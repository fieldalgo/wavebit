"""Host-managed JAX evaluations with automatic geometry re-preparation."""
from threading import RLock

import numpy as np

from .._common.backend import eager_under_torch_compile
from .._common.execution import Execution
from ..errors import IncompatibleProblemError, NumericalError, UnsupportedFeatureError, ValidationError
from . import _adapt, _lifecycle
from .numerics import Numerics


_TOPOLOGY_CODES = frozenset(('prepared.topology', 'prepared.rectilinear_topology',
                             'prepared.raster_partition'))
_TOPOLOGY_CHECKS = frozenset(('shape_topology', 'rectilinear_topology', 'raster_partition'))


@eager_under_torch_compile
class Compiled:
    """A host callable owning one preparation and its JAX-compiled observations.

    Construct with :func:`wavebit.rcwa.compile`. Calls accept a concrete real
    floating-point parameter array (including a scalar). Re-preparation is automatic
    at the same parameters when the recorded geometry stops fitting. Methods return
    JAX arrays; use the derivative methods instead of applying JAX transforms to this
    stateful host controller. ``build`` and ``observe`` must be pure, JAX-traceable
    functions, with fixed model structure and observation shape.
    """

    def __init__(self, build, observe, *, numerics, device='cpu', precision='double'):
        if not callable(build) or not callable(observe):
            raise ValidationError('build and observe must be callable.',
                                  code='compiled.callable', path='build/observe')
        if not isinstance(numerics, Numerics):
            raise ValidationError('numerics must be rcwa.Numerics.', code='numerics.type', path='numerics')
        self._execution = Execution(backend='jax', device=device, precision=precision)
        self._execution.validate()
        # Choosing this controller explicitly chooses host-managed geometry checks.
        self._numerics = numerics.replace(topology_check='host')
        self._build, self._observe = build, observe
        self._prepared = None
        self._functions = {}
        self._preparation_count = 0
        self._lock = RLock()

    @property
    def preparation_count(self):
        """Number of successfully installed preparations, including the first one."""
        return self._preparation_count

    def __call__(self, parameters):
        """Evaluate the observation, rebuilding stale geometry automatically."""
        return self._evaluate(parameters, 'value')

    def value_and_grad(self, parameters):
        """Return a real scalar observation and its reverse-mode gradient."""
        return self._evaluate(parameters, 'gradient')

    def value_and_jacobian(self, parameters):
        """Return the observation array and its forward-mode Jacobian."""
        return self._evaluate(parameters, 'jacobian')

    def jacobian(self, parameters):
        """Return the observation Jacobian, rebuilding stale geometry automatically."""
        return self.value_and_jacobian(parameters)[1]

    def _parameters(self, parameters):
        import jax
        import jax.numpy as jnp

        if _adapt.has_tracer(parameters):
            raise UnsupportedFeatureError(
                'A compiled RCWA controller must be called outside jax.jit, grad and vmap. '
                'It compiles internally; use value_and_grad or value_and_jacobian for derivatives.',
                code='compiled.host_call', path='parameters')
        values = np.asarray(parameters)
        if values.dtype.kind != 'f' or not np.isfinite(values).all():
            raise ValidationError('parameters must be a finite real floating-point array or scalar.',
                                  code='compiled.parameters', path='parameters')
        platform, _, index = self._execution.device.partition(':')
        device = jax.devices(platform)[int(index or 0)]
        dtype = jnp.float64 if self._execution.precision == 'double' else jnp.float32
        return jax.device_put(jnp.asarray(values, dtype=dtype), device)

    def _prepare(self, problem):
        return _lifecycle.prepare(problem, numerics=self._numerics, backend='jax',
                                  device=self._execution.device, precision=self._execution.precision)

    def _function(self, prepared, mode):
        import jax
        import jax.numpy as jnp

        build, observe = self._build, self._observe

        def evaluate(parameters):
            result = prepared.run(build(parameters))
            value = jnp.asarray(observe(result))
            if value.dtype.kind not in 'fc':
                raise ValidationError('observe must return a floating-point or complex array.',
                                      code='compiled.observation', path='observe')
            if mode == 'gradient' and (value.ndim != 0 or value.dtype.kind != 'f'):
                raise ValidationError('value_and_grad requires a real scalar observation.',
                                      code='compiled.scalar_objective', path='observe')
            return value, dict(result.validity.checks)

        if mode == 'value':
            return jax.jit(evaluate)
        if mode == 'gradient':
            return jax.jit(jax.value_and_grad(evaluate, has_aux=True))

        def with_aux(parameters):
            value, checks = evaluate(parameters)
            return value, (value, checks)

        return jax.jit(jax.jacfwd(with_aux, has_aux=True))

    def _evaluate(self, parameters, mode):
        with self._lock, self._execution._scope():
            parameters = self._parameters(parameters)
            problem = self._build(parameters)
            prepared = self._prepared
            fresh = prepared is None
            if prepared is not None:
                report = prepared.validate(problem)
                failures = [issue for issue in report.issues if issue.severity == 'error']
                if failures:
                    if not all(issue.code in _TOPOLOGY_CODES for issue in failures):
                        report.raise_if_invalid()
                    fresh = True
            if fresh:
                prepared = self._prepare(problem)
            functions = {} if fresh else dict(self._functions)

            # Preflight usually detects stale geometry. A device/host arithmetic
            # difference can still fail a runtime guard: retry an old plan once at
            # this exact input, never loop or return the previous point's result.
            for attempt in range(2):
                if mode not in functions:
                    functions[mode] = self._function(prepared, mode)
                answer = functions[mode](parameters)
                if mode == 'value':
                    value, checks = answer
                    derivative = None
                elif mode == 'gradient':
                    (value, checks), derivative = answer
                else:
                    derivative, (value, checks) = answer
                failed = {name for name, passed in checks.items() if not bool(passed)}
                admission = failed - {'numerical_finite'}
                if admission:
                    if admission <= _TOPOLOGY_CHECKS and not fresh and attempt == 0:
                        prepared = self._prepare(problem)
                        functions = {}
                        fresh = True
                        continue
                    name = sorted(admission)[0]
                    if name in _TOPOLOGY_CHECKS:
                        raise IncompatibleProblemError(
                            'Fresh geometry failed its runtime check at the requested parameters.',
                            code=f'prepared.{name}', path='stack')
                    _lifecycle._refuse_guard(problem, name)
                if failed or not np.isfinite(np.asarray(value)).all():
                    raise NumericalError('The requested observation is nonfinite.',
                                         code='rcwa.nonfinite', path='result')
                if derivative is not None and not np.isfinite(np.asarray(derivative)).all():
                    raise NumericalError('The requested observation derivative is nonfinite.',
                                         code='rcwa.nonfinite_derivative', path='result')
                # Failed inputs, observers and numerical solves never replace a
                # working cache. Keep only this plan; no unbounded topology history.
                self._prepared, self._functions = prepared, functions
                self._preparation_count += int(fresh)
                return value if derivative is None else (value, derivative)


@eager_under_torch_compile
def compile(build, observe, *, numerics, device='cpu', precision='double'):
    """Compile an RCWA observation with automatic geometry re-preparation.

    ``build(parameters)`` returns a Problem; ``observe(result)`` returns a scalar
    or array. This JAX controller uses ``topology_check='host'`` and its supported
    scalar-material, single-plane-wave scope. Other numerical choices are retained.
    Call it directly from a host optimizer; it manages preparation, JIT and first
    derivatives internally. Re-preparation does not assert smoothness at topology
    boundaries or convergence of the geometry discretization.
    """
    return Compiled(build, observe, numerics=numerics, device=device, precision=precision)
