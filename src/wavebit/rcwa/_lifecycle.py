"""Validated RCWA preparation and stateless execution over the numerical engine."""
from dataclasses import dataclass, field
from contextlib import contextmanager
from types import MappingProxyType

import numpy as np

from .._common.execution import Execution, CheckReport
from .._common.backend import eager_under_torch_compile
from ..errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError, NumericalError, Issue
from .problem import Problem
from .numerics import Numerics
from . import _adapt


def _issue(error, path='problem'):
    if isinstance(error, ValidationError):
        return error.issue
    return Issue(code='rcwa.unsupported' if isinstance(error, NotImplementedError) else 'rcwa.invalid',
                 path=path, message=str(error))


@contextmanager
def _normalize_errors():
    """Translate legacy admission failures at the public API boundary."""
    try:
        yield
    except ValidationError:
        raise
    except NotImplementedError as error:
        raise UnsupportedFeatureError(str(error), code='rcwa.unsupported', path='problem') from error
    except (TypeError, ValueError) as error:
        raise ValidationError(str(error), code='rcwa.invalid', path='problem') from error


def _engine_scene(solver):
    return getattr(solver, '_scene', solver)


def _resolve(problem, numerics, execution):
    if not isinstance(problem, Problem):
        raise ValidationError('rcwa lifecycle functions require rcwa.Problem.', code='problem.type', path='problem')
    if not isinstance(numerics, Numerics):
        raise ValidationError('numerics must be rcwa.Numerics.', code='numerics.type', path='numerics')
    execution.validate()
    _adapt.validate_values(problem)
    if problem.outputs.scattering and numerics.algorithm != 'smm':
        raise UnsupportedFeatureError('Scattering matrices require algorithm="smm".',
                                      code='rcwa.scattering_algorithm', path='outputs.scattering')
    if not _adapt.single_plane(problem):
        if problem.outputs.diffraction:
            raise UnsupportedFeatureError('Coherent beams and internal sources currently expose boundary power and requested fields. Use Outputs(diffraction=False); no unique single-incident-port R/T is invented.',
                                          code='rcwa.coherent_diffraction', path='outputs.diffraction')
        if problem.outputs.scattering:
            raise UnsupportedFeatureError('A coherent-source result does not retain a unique group-independent scattering matrix. Use rcwa.operators for explicit operators.',
                                          code='rcwa.coherent_scattering', path='outputs.scattering')
        if numerics.detuning is not None:
            raise UnsupportedFeatureError('Detuning is not implemented for coherent-source runs.',
                                          code='rcwa.coherent_detuning', path='numerics.detuning')
        if _adapt.has_tracer(problem):
            raise UnsupportedFeatureError('The coherent-source assembly is an eager capability. Its host grouping does not support external JAX transforms.',
                                          code='rcwa.coherent_transform', path='excitation.sources')
    solvers = []
    with execution.scope():
        for wavelength in _adapt.wavelengths(problem):
            solver = _adapt.build_solver(problem, numerics, execution, wavelength)
            scene = _engine_scene(solver)
            scene._check_before_building()
            solvers.append(solver)
    return solvers


def _report(problem, numerics, execution, solvers):
    scene = _engine_scene(solvers[0])
    window = tuple(tuple(v) for v in scene.order_window)
    modes = (2*window[0][1]+1)*(2*window[1][1]+1)
    cbytes = 16 if execution.precision == 'double' else 8
    single = _adapt.single_plane(problem)
    return CheckReport(
        estimates={'retained_spatial_modes': modes,
                   'one_dense_material_matrix_bytes': modes*modes*cbytes,
                   'one_first_order_matrix_bytes': (4*modes)**2*cbytes,
                   'scope': 'individual dense matrix storage only; excludes temporaries, layer copies, source groups, and backend runtime; not a peak-memory bound'},
        capabilities={'eager': True, 'material_gradients': execution.backend != 'numpy' and single,
                      'jit': execution.backend == 'jax' and single,
                      'vmap': execution.backend == 'jax' and single,
                      'geometry_reuse': 'concrete topology validation, on the primal of an eager jax.grad tracer too; geometry traced under jit or vmap is refused',
                      'diffraction': single, 'coherent_boundary_power': not single,
                      'physical_convergence': 'not established by admission or finite arithmetic'},
        resolved={'order_window': window, 'wavelength_count': len(solvers),
                  'layer_count': len(problem.stack.layers), 'fourier_coefficients': scene.fourier_coefficients.value,
                  'factorization': scene.factorization.value,
                  'algorithm': numerics.algorithm, 'formulation': numerics.formulation,
                  'shape_nvf_grid_size': scene.shape_nvf_grid_size if scene.factorization.value == 'nvf' else None,
                  'backend': execution.backend, 'device': execution.device, 'precision': execution.precision,
                  'retained_fields': problem.outputs.fields, 'retained_scattering': problem.outputs.scattering,
                  'length_unit': problem.units.length})


@eager_under_torch_compile
def check(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Return admission and scoped resource estimates without solving scattering."""
    execution = Execution(backend=backend, device=device, precision=precision)
    try:
        if _adapt.has_tracer(problem):
            raise UnsupportedFeatureError('check requires concrete inputs; call it outside an autodiff transform.',
                                          code='check.concrete', path='problem')
        solvers = _resolve(problem, numerics, execution)
        return _report(problem, numerics, execution, solvers)
    except (ValidationError, TypeError, ValueError, NotImplementedError) as error:
        return CheckReport(issues=(_issue(error),))


def _partition_guards(problem, solvers):
    """Freeze only equality partitions needed by raster host preparation.

    New material values may change, but the recorded phase partition must remain
    identical. This conservative guard prevents a traced replay from silently
    changing compression or NVF interfaces.
    """
    scene = _engine_scene(solvers[0])
    if not any(layer.epsilon is not None for layer in problem.stack.layers):
        return ()
    if scene.factorization.value == 'li' and scene.fourier_coefficients.value == 'sampled':
        return ()
    out = []
    for i, layer in enumerate(problem.stack.layers):
        for name in ('epsilon', 'mu'):
            value = getattr(layer, name)
            if value is None:
                continue
            a = _adapt.host(value, f'stack.layers[{i}].{name}')
            _, first, inverse = np.unique(a.reshape(-1), return_index=True, return_inverse=True)
            out.append((i, name, first.copy(), inverse.copy()))
    return tuple(out)


def _runtime_checks(problem, execution, partitions):
    xp = _adapt._namespace(execution)
    finite = xp.asarray(True)
    for value in _adapt.leaves(problem):
        a = _adapt._array(xp, value, execution, complex_value=True)
        finite = finite & xp.all(xp.isfinite(a))
    positive_wavelength = xp.asarray(True)
    for value in _adapt.wavelengths(problem):
        a = _adapt._array(xp, value, execution, True)
        positive_wavelength = positive_wavelength & xp.all((xp.real(a) > 0) & (xp.imag(a) == 0))
    positive_thickness = xp.asarray(True)
    for layer in problem.stack.layers:
        a = _adapt._array(xp, layer.thickness, execution, True)
        positive_thickness = positive_thickness & xp.all((xp.real(a) >= 0) & (xp.imag(a) == 0))
    polarization = xp.asarray(True)
    for source in problem.excitation.sources:
        if hasattr(source, 'polarization') and not isinstance(source.polarization, str):
            a = _adapt._array(xp, source.polarization, execution, True)
            polarization = polarization & (xp.sum(xp.abs(a)**2) > 0)
    if _adapt.single_plane(problem):
        polarization = polarization & (xp.abs(_adapt._array(xp, problem.excitation.sources[0].amplitude,
                                                           execution, True)) > 0)
    partition = xp.asarray(True)
    for i, name, first, inverse in partitions:
        if _adapt.has_tracer(getattr(problem.stack.layers[i], name)) and len(first) > 256:
            raise UnsupportedFeatureError('The prepared raster has more than 256 distinct phases. Its equality partition cannot be economically guarded under a trace; use sampled Li, or prepare each concrete design outside the transform.',
                                          code='rcwa.raster_partition_transform', path=f'stack.layers[{i}].{name}')
        a = _adapt._array(xp, getattr(problem.stack.layers[i], name), execution, True).reshape(-1)
        # Backend integer constants preserve indexing under JIT and Torch grad.
        from .._common.backend import as_array
        idx = as_array(xp, first, dtype=xp.int64, device=getattr(a, 'device', None))
        owners = as_array(xp, inverse, dtype=xp.int64, device=getattr(a, 'device', None))
        phases = a[idx]
        partition = partition & xp.all(a == phases[owners])
        if len(first) <= 256:
            identity = xp.eye(len(first), dtype=xp.bool, device=getattr(a, 'device', None))
            partition = partition & xp.all((phases[:, None] != phases[None, :]) | identity)
        elif not _adapt.has_tracer(getattr(problem.stack.layers[i], name)):
            current = _adapt.host(getattr(problem.stack.layers[i], name), 'raster').reshape(-1)[first]
            partition = partition & xp.asarray(len(np.unique(current)) == len(first))
    return {'finite_inputs': finite, 'positive_wavelength': positive_wavelength,
            'nonnegative_thickness': positive_thickness, 'nonzero_polarization': polarization,
            'raster_partition': partition}


@eager_under_torch_compile
@dataclass(frozen=True)
class Prepared:
    """A fixed numerical contract and static geometry, with no last-call state."""
    numerics: Numerics
    execution: Execution
    report: CheckReport
    contract: object
    _structure: object = field(repr=False)
    _geometry: tuple = field(repr=False)
    _partitions: tuple = field(repr=False)
    _coherent_dependencies: object = field(repr=False)

    def _solvers(self, problem):
        if not isinstance(problem, Problem):
            raise ValidationError('Prepared.run/validate require rcwa.Problem.',
                                  code='problem.type', path='problem')
        _adapt.check_structure(problem, self._structure)
        if _adapt.coherent_dependencies(problem) != self._coherent_dependencies:
            raise IncompatibleProblemError('Coherent source grouping dependencies changed. Prepare again after changing directions, wavelengths, sampling, cell, or half-space response.',
                                           code='prepared.source_grouping', path='excitation.sources')
        traced_geometry = False
        if _adapt.has_tracer(problem):
            for path, value in _adapt.geometry_values(problem).items():
                if not _adapt.has_tracer(value):
                    continue
                traced_geometry = True
                # An eager jax.grad tracer carries its primal, so the topology is checked
                # on that below, as a PyTorch tensor is. Under jit or vmap there is none.
                if not _adapt.carries_primal(value):
                    raise UnsupportedFeatureError('This geometry decision has no traced topology guard. Prepare each concrete geometry outside the transform; material, wavelength, and layer-thickness derivatives retain their supported graphs.',
                                                  code='prepared.host_topology', path=path)
            if self.numerics.detuning is not None:
                raise UnsupportedFeatureError('Adaptive detuning is eager-only. Use exact-point numerics under a transform, or explicitly change the physical input and prepare that problem.',
                                              code='prepared.detuning_transform', path='numerics.detuning')
        solvers = _resolve(problem, self.numerics, self.execution)
        # Built only to be checked: the run keeps the traced values and their tangents.
        checked = (_resolve(_adapt.primal_geometry(problem), self.numerics, self.execution)
                   if traced_geometry else solvers)
        for solver, geometry in zip(checked, self._geometry):
            try:
                _engine_scene(solver)._refuse_a_stale_geometry_record(geometry)
            except (ValueError, RuntimeError) as error:
                raise IncompatibleProblemError(str(error), code='prepared.topology', path='stack') from error
        return solvers

    def validate(self, problem):
        """Eagerly report whether a concrete problem can reuse this preparation."""
        try:
            if _adapt.has_tracer(problem):
                raise UnsupportedFeatureError('validate requires concrete inputs; call it outside the transform.',
                                              code='validate.concrete', path='problem')
            with self.execution.scope():
                solvers = self._solvers(problem)
                checks = _runtime_checks(problem, self.execution, self._partitions)
                for name, passed in checks.items():
                    if not bool(_adapt.host(passed, name)):
                        raise IncompatibleProblemError('The prepared runtime guard failed.',
                                                       code=f'prepared.{name}', path='problem')
                return _report(problem, self.numerics, self.execution, solvers)
        except (ValidationError, TypeError, ValueError, NotImplementedError, RuntimeError) as error:
            return CheckReport(issues=(_issue(error),))

    def run(self, problem):
        """Execute live values after structural and numerical guard checks."""
        from .results import pack
        with self.execution.scope(), _normalize_errors():
            solvers = self._solvers(problem)
            checks = _runtime_checks(problem, self.execution, self._partitions)
            for name, passed in checks.items():
                if not _adapt.has_tracer(passed) and not bool(_adapt.host(passed, name)):
                    raise IncompatibleProblemError('The prepared runtime guard failed.', code=f'prepared.{name}', path='problem')
            solved = []
            for solver, geometry in zip(solvers, self._geometry):
                try:
                    if _adapt.single_plane(problem):
                        result = solver.solve(geometry=geometry, auto_detune=self.numerics.detuning or False)
                    else:
                        result = solver.solve(geometry=geometry)
                except (np.linalg.LinAlgError, FloatingPointError) as error:
                    raise NumericalError(str(error), code='rcwa.singular', path='problem') from error
                solved.append((solver, result))
            result = pack(problem, self.numerics, self.execution, solved, checks, self.report)
            finite = result.validity.checks['numerical_finite']
            if not _adapt.has_tracer(finite) and not bool(_adapt.host(finite, 'result')):
                raise NumericalError('The requested solve produced nonfinite observables; no implicit detuning was applied.',
                                     code='rcwa.nonfinite', path='result')
            return result


@eager_under_torch_compile
def prepare(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Validate concrete values and freeze the static geometry needed by the engine."""
    execution = Execution(backend=backend, device=device, precision=precision)
    from .._common.pytrees import register_dataclass
    from dataclasses import fields
    from .problem import Outputs
    register_dataclass(Numerics, static_fields=tuple(f.name for f in fields(Numerics)))
    register_dataclass(Outputs, static_fields=tuple(f.name for f in fields(Outputs)))
    register_dataclass(Execution, static_fields=('backend', 'device', 'precision'))
    if _adapt.has_tracer(problem):
        raise UnsupportedFeatureError('prepare requires a concrete problem. Prepare before JAX grad/jit/vmap and pass live Problems to Prepared.run.',
                                      code='prepare.concrete', path='problem')
    with execution.scope(), _normalize_errors():
        solvers = _resolve(problem, numerics, execution)
        geometry = tuple(_engine_scene(solver).freeze_geometry() for solver in solvers)
        report = _report(problem, numerics, execution, solvers)
        partitions = _partition_guards(problem, solvers)
    contract = MappingProxyType({'static': ('numerics', 'execution', 'units', 'layer/source/model counts',
                                          'array/tensor shapes', 'magnetic presence', 'outputs', 'spectral shape'),
                                 'live': ('material coefficients', 'wavelength', 'source amplitude and polarization',
                                          'layer thickness', 'concrete geometry within validated topology for a single plane wave'),
                                 'coherent_grouping': 'cell, wavelength, source directions/sampling and half-space responses are fixed; prepare again to change them',
                                 'runtime_guards': ('finite inputs', 'positive wavelength', 'nonnegative thickness',
                                                    'nonzero polarization', 'raster phase partition'),
                                 'traced_geometry': 'checked on the primal under eager jax.grad; refused under jit or vmap, which carry none',
                                 'cache_policy': 'static geometry only; every run rebuilds material, mode, and propagation operators'})
    return Prepared(numerics=numerics, execution=execution, report=report, contract=contract,
                    _structure=_adapt.structure_key(problem), _geometry=geometry, _partitions=partitions,
                    _coherent_dependencies=_adapt.coherent_dependencies(problem))


@eager_under_torch_compile
def run(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Prepare and run one complete physical problem."""
    return prepare(problem, numerics=numerics, backend=backend, device=device, precision=precision).run(problem)
