"""The JAX backend: the namespace, the device scope, the pytree, and the optimizer mixin.

Was `wavebit/on_jax/`, five files deep. See :mod:`wavebit.backends.numpy_backend` for why the
depth went.

**Importing this module lifts JAX's 32-bit cap**, and that is deliberate rather than a side
effect nobody chose: RCWA needs double precision, and with x64 off JAX silently truncates
float64 and complex128 at array creation. It happens here, on first use of this backend, rather
than on `import wavebit`, because the setting is process-wide and belongs to the caller. The
line sits above every other import in this file for the reason it sat above the solver import
in the package this module replaces: whatever runs first must not build an array under the cap.

`wavebit.enable_x64()` is the public way to lift it earlier, which is what a caller who builds
JAX arrays *before* the first solve needs -- see :class:`~wavebit._core.backend.PrecisionLossWarning`.
"""
import jax
from .._core.backend import eager_under_torch_compile as _eager_under_torch_compile

# Above the rest, and before anything here can build an array. See the module docstring.
jax.config.update('jax_enable_x64', True)

from collections.abc import Mapping                              # noqa: E402
from contextlib import contextmanager                            # noqa: E402
from functools import partial                                    # noqa: E402
from inspect import signature                                    # noqa: E402

import jax.numpy as jnp                                          # noqa: E402
import jax.numpy as _xp                                          # noqa: E402
import numpy as np                                              # noqa: E402
from jax import tree_util                                        # noqa: E402
from tqdm import tqdm                                            # noqa: E402

from .._core.engine import (CoreFactorizationMixin, CoreSolveMixin,  # noqa: E402
                            _concrete_2x2, _copy_condition_containers, _holds_tracer,
                            device_scoped)
from .._core import fields as _fields                            # noqa: E402
from .._core import lattice as _lattice_module                   # noqa: E402
from .._core.enums import Backend                                # noqa: E402
from .._core.enums import DIFFERENTIABLE_BACKENDS as _DIFFERENTIABLE_BACKENDS  # noqa: E402
from .._core.result import Result, ResultSub                     # noqa: E402
from .._core.geometry import StaticGeometry                      # noqa: E402


@_eager_under_torch_compile
class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the parametric builder, which starts from
    #: parametric shapes and so cannot infer a namespace from its input -- used to reach a
    #: `_namespace_for_backend(self.backend)` living in `_core` that imported jax and torch by
    #: name and fell through to numpy on an unrecognised code, with no `else`. The solver
    #: already knows which namespace it is; asking it is cheaper than deciding again, and it
    #: keeps `_core` from naming an optional dependency.
    xp = _xp

    #: Which backend this class is, as the member `rcwa(backend=...)` parses to. A declared
    #: member because the shared constructor cannot spell three literals -- and it has to
    #: stay an accepted *keyword* as well: `SolverJax._tree_flatten` puts `backend` in
    #: aux_data, so it is part of the treedef and of every jit cache key, and
    #: `_tree_unflatten` hands it straight back to `__init__` on every jitted call.
    _BACKEND_ID = Backend.JAX

    #: Derived, not declared: `_core.enums.DIFFERENTIABLE_BACKENDS` is the one table,
    #: so `manifest.py` can publish this without importing this module to ask.
    _DIFFERENTIABLE = _BACKEND_ID in _DIFFERENTIABLE_BACKENDS


    def _device_scope(self):
        """`jax.default_device`, which is how JAX is told where to place what it builds.

        `self.device` is a *list* here -- `jax.devices('cpu')` -- so it is indexed. That is
        the one thing about it that is not obvious, and it is why this is an override rather
        than a value the base could hold.

        The scope places what the core builds from Python scalars inside a call, which has no
        operand to follow. It does not decide where a computation *on the solver's arrays*
        runs -- a committed operand outranks it -- and that is `_array_device`'s job.
        """
        return jax.default_device(self.device[0])

    def _eager(self):
        """`jax.ensure_compile_time_eval`: a device move is carried out in a trace, not staged.

        See :meth:`CoreSolveMixin._eager`. Inside ``jax.jit`` a ``device_put`` of a concrete
        array is otherwise a step of the traced program and hands back a tracer; under this
        context it moves the array now and hands back the array (measured on jax 0.11.0 with two
        host devices). A tracer still stages, which is what a tracer should do.
        """
        return jax.ensure_compile_time_eval()

    @property
    def _array_device(self):
        """The first of `self.device`, so every array a setter builds is *committed* there.

        The scope above is not enough on its own, and that was measured, not argued
        (2026-09-11, jax 0.11.0, a CUDA jaxlib, `device='cpu'`). The constructor and every
        setter run outside any decorated method, so the structure, the thickness and the rest
        were built on JAX's default device -- the GPU -- while `solver.device` read
        `[CpuDevice(id=0)]`. The builders then place their index arrays with
        `device=<the structure's device>`, which committed the convolution matrices to the GPU,
        and a committed operand outranks `jax.default_device`: `solve_conv`'s scope was open and
        the eig ran on the GPU anyway. On a 2D raster T came back 8.77000489428177987e-01,
        against 8.77000489428339414e-01 from a CPU-only jaxlib.

        Committing at the setters makes the device a property of the value, which is what JAX
        consults once a computation has a committed operand -- at the dispatch of a jitted call
        too, where a `device_put` inside the trace moves nothing. It is Torch's arrangement as
        well: there a tensor is placed at creation, and there is nowhere else to say it.
        """
        return self.device[0]

    #: A JAX array -- and a `DynamicJaxprTracer` too, which is what lets a period be
    #: differentiated or closed over inside `jax.jit`.
    _ARRAY_TYPES = (jnp.ndarray,)

    def _resolve_device(self, value):
        """A list of :class:`jax.Device`, which is what `jax.default_device` takes.

        A single already-resolved device is wrapped rather than refused, so
        `solver.device = solver.device` round-trips against the getter. The membership test is
        against the public `jax.Device`: matching the implementation class's repr instead broke
        silently when jaxlib moved it from `xla_extension` to `_jax`.
        """
        if value is None or value in (0, 'cpu'):
            return jax.devices('cpu')
        if value in (1, 'gpu', 'cuda'):
            return jax.devices('gpu')
        if isinstance(value, jax.Device):
            return [value]
        if (isinstance(value, (list, tuple)) and value
                and all(isinstance(d, jax.Device) for d in value)):
            return list(value)
        raise ValueError(f'unsupported jax device: {value!r}')


from .._core.field_result import FieldState
jax.tree_util.register_pytree_node_class(FieldState)

# `Port` registers itself when a port is built, so a matrix built before JAX arrived kept its
# channel as a string leaf until another one was built. Registering on this import -- the first
# use of the backend -- repairs it. See `_core.scattering._register_pytree_if_jax_loaded`.
from .._core import scattering as _scattering
_scattering._register_pytree_if_jax_loaded()


@_eager_under_torch_compile
@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@_eager_under_torch_compile
@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""


@_eager_under_torch_compile
class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    #: The pytree-registered pair, so a jitted `solve_field` can return one. See the
    #: declaration in `CoreFactorizationMixin`.
    _RESULT = ResultJax
    _RESULT_SUB = ResultSubJax

    @device_scoped
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('incident',),
                    geometry=None, **kwargs):
        """Solve and reconstruct the field in one jit-compiled call.

        ``geometry`` is required here, unlike on :meth:`solve`. This method jits over
        ``self``, so the structure arrives as tracers and the discrete decisions inside the
        factorization and the solve have nothing concrete to branch on; a
        :class:`~wavebit._core.geometry.StaticGeometry` from
        :meth:`~wavebit._core.engine.CoreFactorizationMixin.freeze_geometry` supplies them as
        static data. It is a static argument -- reuse one freeze across calls and you compile
        once.

        A stale ``geometry`` is refused here as ``solve`` refuses it: raster normal fields,
        compression records, shape topology and tensor records are checked against the live
        solver before the compiled call, by the builders' own checks without Fourier assembly
        (:meth:`~wavebit._core.engine.CoreFactorizationMixin._refuse_a_stale_geometry_record`).
        They were replayed on trust until 2026-09-25, when keywords that move the cell reached
        this door: a normal field frozen on one oblique cell and replayed on another answered
        ``sum(T)`` wrong by up to 0.86 with nothing said. When the geometry inputs are traced
        under an outer ``jax.jit`` or ``jax.grad``, they cannot be compared, so the record is
        replayed as frozen. Traced conditions unrelated to geometry, or retained outputs of
        an earlier traced solve, do not suppress this check on concrete geometry inputs.

        ``sample_x`` / ``sample_y`` / ``sample_z`` and ``incidences`` are static too: they set the
        output shape and select which incidences to reconstruct. Concrete coordinate lists
        and arrays are normalized to immutable tuples before compilation, so equivalent
        containers share a cache entry. Differentiate x/y sample positions through the eager
        or result-owned reconstruction API. Explicit z heights require concrete layer
        thicknesses and must also be reconstructed outside this compiled method.

        Use :meth:`solve_field_eager` for the eager equivalent that auto-detects.

        ``kwargs`` are conditions, as on :meth:`solve`, and they are assigned *here*, on the
        live solver, before the compiled call rather than inside it. Inside, they were handed to
        the copy of ``self`` the trace rebuilds, and that copy is gone when the call returns.
        Measured 2026-09-24: ``solve_field(thickness=[900.], geometry=g)`` answered for 900 and
        left ``solver.thickness`` at 300, where NumPy and Torch keep 900, so the same sweep
        loop's next ``solve()`` answered different scenes on different backends. And a condition
        that is not an array could not be passed at all: ``tetm='tm'``, ``layer_algorithm``
        and ``device`` were refused by ``jax.jit`` as arguments it cannot trace, and
        ``truncation_order`` arrived traced at a setter that reads integers. Assigned out here,
        a concrete condition stays, as :meth:`solve` keeps one, and reaches the compiled call
        on ``self`` -- as a traced child, or as static aux data that compiles once more for a
        new value, the way a new solver would. A traced one -- this call under an outer
        ``jax.jit`` or ``jax.grad`` -- is put back when the call returns
        (:meth:`~wavebit._core.engine.CoreFactorizationMixin._conditions`). Both refusals come
        first, so neither leaves a keyword assigned: the missing freeze, and a solver that did
        not declare ``answers='field'``.

        **The solve itself stays inside the compiled call.** It runs on the copy of ``self``
        the trace rebuilds, so the modes and amplitudes it leaves behind are that copy's; the
        field this returns is their reconstruction. The live solver's own solve state is the
        last *eager* solve's, and once a kept condition has moved the scene it describes
        another one. Measured 2026-09-24 on a 2D raster: after an eager ``solve()``,
        ``solve_field(top=2.25)`` and then ``reconstruct_field()`` returned a field 2.12 away
        from the moved scene's and 0.83 from the unmoved one's -- neither; after a new
        ``structure`` it returned the old scene's field to 3e-14 while the solver reported the
        new raster; and after ``truncation_order=[2, 2]`` it raised a reshape ``TypeError``.
        When a keyword changes the scene, that state is dropped here and
        :meth:`reconstruct_field` refuses, naming both ways out
        (:meth:`~wavebit._core.engine.CoreFactorizationMixin._forget_the_solve`). A call that
        leaves nothing on the solver -- no keyword, or only traced ones, which go back -- keeps
        it: the scene it describes is still the one the solver reports.

        A structure declared through ``Lattice.drawn`` is checked against the cell here, before
        the compiled call, because inside it the cell is a traced child and cannot be compared;
        the flatten that starts the call records that it was (``_tree_flatten``), and a traced
        cell anywhere else is refused rather than skipped. Parametric shape declarations
        cross the compiled call through the static leaves of ``_ShapeStructure``.

        Whether the call moved the scene is decided by the flattened solver before and after --
        the scene the compiled call itself reads -- not by which objects changed, so a keyword
        equal to what the solver held keeps the solve state. A call that fails after a concrete
        keyword was assigned keeps that keyword, as :meth:`solve` does, and so has moved the
        scene: the state is dropped then too.
        """
        if geometry is None:
            raise ValueError(
                'solve_field is jit-compiled and needs the geometry frozen up front:\n'
                '    geometry = solver.freeze_geometry()\n'
                '    result, field_cell = solver.solve_field(..., geometry=geometry)\n'
                'Reuse that one geometry across calls to compile only once. '
                'For the eager, auto-detecting path use solver.solve_field_eager(...).')
        self._require_field_declaration()
        self._refuse_unknown_conditions(kwargs)
        before = self._scene_as_the_compiled_call_reads_it()
        try:
            with self._conditions(kwargs):
                self._refuse_a_raster_drawn_on_another_cell()
                # The dispatcher validates scalar/tensor shapes as well as raster records.
                # Retained traced outputs must not hide readable geometry inputs.
                self._refuse_a_stale_geometry_record(geometry)
                sample_x = _static_field_samples(sample_x, 'sample_x')
                sample_y = _static_field_samples(sample_y, 'sample_y')
                sample_z = _static_field_samples(sample_z, 'sample_z')
                incidences = tuple(_fields.INCIDENCE_NAMES[i]
                                   for i in _fields.incidence_indices(incidences))
                return self._solve_field_compiled(sample_x, sample_y, sample_z, incidences,
                                                  geometry)
        finally:
            after = self._scene_as_the_compiled_call_reads_it()
            if before is None or after is None or not _same_scene(before, after):
                self._forget_the_solve()

    def _scene_as_the_compiled_call_reads_it(self):
        """This solver flattened -- the scene the compiled call would solve -- or ``None``.

        What decides whether a call moved the scene. It was object identity over
        ``vars(self)``, and every concrete keyword stores a new object, so
        ``solve_field(top=1.0)`` on a solver already at ``top=1.0`` dropped a solve state that
        still described its scene, and ``reconstruct_field`` refused where it had answered
        exactly (measured 2026-09-24). The flattened solver is the scene by construction: it is
        what the compiled call rebuilds and solves. ``None`` when it cannot be flattened or read,
        which the caller treats as moved.
        """
        try:
            leaves, treedef = tree_util.tree_flatten(self)
        except Exception:                                  # noqa: BLE001 - unreadable = moved
            return None
        return treedef, leaves

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    @device_scoped
    def _solve_field_compiled(self, sample_x, sample_y, sample_z, incidences, geometry):
        """:meth:`solve_field`'s compiled body. The conditions are already on ``self``.

        This, not :meth:`solve_field`, is what ``jax.jit`` wraps with ``self`` at argnum 0: every
        call flattens the solver (:meth:`SolverJax._tree_flatten`) and runs on the solver
        :meth:`SolverJax._tree_unflatten` rebuilds inside the trace.
        """
        return self._solve_and_field(sample_x, sample_y, sample_z, incidences, geometry)


# The eager path, which is `CoreFactorizationMixin.solve_field` under its other name -- one
# function, two names, rather than a body written twice. JAX is the only backend that needs
# both, because it is the only one whose `solve_field` is jitted and so demands a frozen
# geometry; the other two have nothing to be eager *relative to*. Bound here rather than inside
# the class because the name it aliases is the one this class overrides.
RCWAJax.solve_field_eager = CoreFactorizationMixin.solve_field


def _static_field_samples(samples, name):
    """Keep concrete coordinate spellings equivalent at the compiled cache boundary."""
    try:
        _fields._check_spec(samples, name)
        if _fields.is_count(samples):
            return int(samples)
        return tuple(np.asarray(samples).tolist())
    except (TypeError, ValueError) as error:
        if _holds_tracer(samples):
            alternative = ('use a count for sample_z, or reconstruct concrete heights '
                           'outside the transform' if name == 'sample_z' else
                           'use solve_field_eager or result.reconstruct_field for traced x/y coordinates')
            raise ValueError(
                f'{name}: compiled solve_field requires concrete static sampling points; '
                f'{alternative}') from error
        raise


def _same_scene(a, b):
    """Two flattened solvers describe the same scene: one tree, and equal leaves.

    A leaf that cannot be compared -- a tracer -- counts as a difference, so an unreadable
    scene is never taken for an unmoved one.
    """
    (tree_a, leaves_a), (tree_b, leaves_b) = a, b
    if tree_a != tree_b or len(leaves_a) != len(leaves_b):
        return False
    try:
        return all(x is y or bool(jnp.array_equal(x, y)) for x, y in zip(leaves_a, leaves_b))
    except Exception:                                      # noqa: BLE001 - a tracer: moved
        return False


try:
    import optax
except (TypeError, ImportError, ModuleNotFoundError) as e:
    import warnings

    warnings.warn('Importing optax failed. You can run RCWA but not optimization on the JAX backend. '
                  'One possible reason is python version: optax support python>=3.9.')

@contextmanager
def _temporary_solver_state(solver, *, keep_keywords=False, parameters=()):
    """Release traced evaluations; optionally retain explicitly kept condition keywords."""
    state = dict(solver.__dict__)
    kept = {'values': [], 'parameters': frozenset(parameters)} if keep_keywords else None
    try:
        if kept is not None:
            solver.__dict__['_kept_condition_updates'] = kept
        cell_checked = solver._cell_checked_when_flattened()
        for name in ('_structure', '_structure_dispersion', '_excitation',
                     '_lattice_basis', '_structure_drawn_on', '_top', '_bot'):
            if name in state:
                solver.__dict__[name] = _copy_condition_containers(state[name], snapshot_numpy=True)
        if cell_checked:
            # Equal-value snapshots preserve the declaration validation at flatten.
            solver.__dict__['_drawn_checked_against'] = solver._cell_objects()
        with solver._solve_state_scope():
            yield
    finally:
        # Direct custom writes are temporary. Concrete keywords passed through a public
        # condition scope keep their existing persistence contract after restoration.
        solver.__dict__.clear()
        solver.__dict__.update(state)
        try:
            if kept is not None:
                with solver._eager():
                    for name, value in kept['values']:
                        with solver._conditions({name: value}):
                            pass
        except (TypeError, ValueError) as error:
            solver.__dict__.clear()
            solver.__dict__.update(state)
            raise ValueError(
                'cannot restore explicitly kept conditions on the original scene; '
                'the scene before this gradient call has been restored') from error
        finally:
            solver._forget_the_solve()


def _same_prepared_geometry(left, right):
    """Exact, eager equality for the fit's last two records, including array-bearing ones.

    StaticGeometry deliberately hashes raster/tensor arrays by identity. Keep that global
    cache contract; locally retain the previous object only after comparing every field,
    array shape, dtype and value. A fresh but identical raster freeze need not compile the
    optimizer again. Unknown or incomparable records conservatively keep their new identity.
    """
    if type(left) is not StaticGeometry or type(right) is not StaticGeometry:
        return False
    if left == right:
        return True
    leaves_a, tree_a = tree_util.tree_flatten(vars(left))
    leaves_b, tree_b = tree_util.tree_flatten(vars(right))
    if tree_a != tree_b:
        return False
    try:
        for a, b in zip(leaves_a, leaves_b):
            if a is b:
                continue
            if isinstance(a, (np.ndarray, jax.Array)) or isinstance(b, (np.ndarray, jax.Array)):
                a, b = np.asarray(a), np.asarray(b)
                if a.dtype != b.dtype or not np.array_equal(a, b):
                    return False
            elif not bool(a == b):
                return False
    except (TypeError, ValueError):
        return False
    return True


@_eager_under_torch_compile
class OptimizerJax:

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def _grad(params, forward, loss_fn):
        def forward_pass(params, forward, loss):
            result = forward(**params)
            loss_value = loss(result)
            return loss_value

        loss_value, grads = jax.value_and_grad(forward_pass)(params, forward, loss_fn)

        # Conjugated HERE, once, because both callers need it and only one of them did it.
        # For a real-valued loss of a complex parameter, `jax.value_and_grad` returns
        # `dL/dx - i dL/dy`, which is the conjugate of the steepest-ascent direction. `fit`
        # knew that and conjugated in its own `step`; `grad` returned the raw value, so the
        # two shipped answers for one question were complex conjugates of each other -- and
        # `grad`'s was the one a caller reads and steps along. Measured against finite
        # differences on a lossy grating (eps = 4 + 0.6j), dL/dx = -0.000489034 and
        # dL/dy = -0.029081472: stepping `z - lr * grad` moved the loss by -8.460e-07 with the
        # conjugated value and +8.455e-07 with the raw one. Downhill and uphill.
        #
        # Real parameters are unaffected -- `conj` of a real array is itself -- which is why a
        # thickness sweep never showed it and why this went out at all.
        return loss_value, tree_util.tree_map(jnp.conj, grads)

    # Scoped, so what `forward` and the optimizer build from scalars -- `optimizer.init`'s
    # zeros, a loss's constants -- lands where the solve runs rather than on JAX's default.
    @device_scoped
    def grad(self, parameters, forward, loss_fn):
        """Return gradients while restoring the complete parameter assignments.

        Both the forward and loss evaluate on temporary solver state, including custom
        assignments to derived conditions and edits to shape recipes or excitation dictionaries. Reassigning resolved
        property values after differentiation loses declarations such as a default
        polarization, a drawn cell, or a tabulated material definition. Restore the original
        scene instead, retaining concrete condition keywords passed by the forward as on
        a direct solve. Named gradient parameters still return to their original declarations.
        Release traced outputs before leaving differentiation, including on failure.
        """
        params = {name: (getattr(self, name)) for name in parameters}
        def evaluate(**values):
            with _temporary_solver_state(self, keep_keywords=True, parameters=params), self._conditions(values):
                return loss_fn(forward(**values))
        _, grads = self._grad(params, evaluate, lambda value: value)
        return grads

    @device_scoped
    def fit(self, parameters, forward, loss_fn, optimizer, iteration=1, *,
            parameterize=None, prepare_geometry=None):
        """Optimize with a geometry prepared from the current design before every JIT step.

        ``parameters`` is either a sequence of solver attribute names (the original API),
        or a mapping of initial numeric design parameters. ``parameterize(**params)`` may
        map those design parameters to solver conditions, for example a width to a shape
        ``structure``. It runs both eagerly to prepare the geometry and under differentiation
        to carry the design derivatives; keep it pure and use traceable array operations.

        For this solver's ``solve``, ``solve_field`` or ``solve_field_eager`` (including
        ``functools.partial``), the loop calls ``freeze_geometry`` on the current conditions
        outside JIT. Fixed keyword conditions in a partial participate too. A ``geometry``
        captured by keyword or position in that partial is replaced, so an initial freeze
        cannot go stale. The record is an explicit static argument of the compiled step.
        Exactly equal records
        reuse the previous object (including raster arrays), changed records recompile.
        Continuous design parameters stay traced.

        An arbitrary ``forward`` cannot reveal which scene it solves. For that case provide
        ``prepare_geometry(**conditions)``; it runs outside JIT at every step and its return
        is passed explicitly as ``forward(geometry=..., **conditions)``. The forward must use
        that record, not a freeze hidden in a closure. Without this preparation contract an
        opaque forward is refused before optimization, rather than replayed unchecked.
        Preparation also receives a partial's fixed keyword defaults (except ``geometry``),
        overridden by the current conditions. For a built-in forward these are restricted
        to solver conditions, excluding field sampling and other call options.

        Returns the final design parameters. The built-in forward also leaves their mapped
        conditions on this solver. For a custom forward, the original attribute-name API
        still writes those attributes back; a mapping of custom design variables is returned
        without creating arbitrary solver attributes. The forward and loss use temporary
        solver state during each compiled step; their assignments and mutable condition edits are
        restored to the eagerly prepared scene, including on failure. Retained solve state
        is discarded before differentiation exits.
        """
        named_attributes = not isinstance(parameters, Mapping)
        params = ({name: getattr(self, name) for name in parameters}
                  if named_attributes else dict(parameters))

        method, fixed, positional = forward, {}, ()
        while isinstance(method, partial):
            fixed = {**method.keywords, **fixed}
            positional = (*method.args, *positional)
            method = method.func
        # Inspect the class methods, not unrelated instance attributes: callers may wrap
        # one entry point for instrumentation while fitting through another. A replaced
        # forward itself remains a custom callable and needs explicit preparation.
        automatic = (getattr(method, '__self__', None) is self
                     and any(getattr(method, '__func__', None) is getattr(type(self), name)
                             for name in ('solve', 'solve_field', 'solve_field_eager')))
        if not automatic and prepare_geometry is None:
            raise TypeError(
                'fit needs this solver\'s solve, solve_field or solve_field_eager '
                '(or functools.partial of one) to prepare geometry automatically. '
                'For a custom forward, provide prepare_geometry(**conditions) and accept '
                'the resulting geometry= in forward. Use parameterize= to map design '
                'variables to solver conditions.')

        if positional:
            # Remove an old record even when it occupied a positional slot. BoundArguments
            # preserves the other positions/options and moves arguments after the gap to
            # keywords, so the fresh geometry below cannot collide with the captured one.
            bound = signature(method).bind_partial(*positional, **fixed)
            bound.arguments.pop('geometry', None)
            forward = partial(method, *bound.args, **bound.kwargs)

        def conditions(values):
            mapped = values if parameterize is None else parameterize(**values)
            if not isinstance(mapped, Mapping):
                raise TypeError('parameterize must return a mapping of conditions')
            if 'geometry' in mapped:
                raise ValueError('fit prepares geometry each step; it is not a design condition')
            return dict(mapped)

        fixed_conditions = {name: value for name, value in fixed.items()
                            if name != 'geometry'
                            and (not automatic or name in self._condition_names())}
        opt_state = optimizer.init(params)

        @partial(jax.jit, static_argnames=('geometry',))
        def step(params, opt_state, geometry):
            def evaluate(**values):
                with _temporary_solver_state(self):
                    return loss_fn(forward(**conditions(values), geometry=geometry))

            loss_value, grads = self._grad(params, evaluate, lambda value: value)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        started = False
        previous_geometry = None
        try:
            for _ in tqdm(range(iteration)):
                started = True
                current = conditions(params)
                prepare = self.freeze_geometry if prepare_geometry is None else prepare_geometry
                geometry = prepare(**(fixed_conditions | current))
                if _same_prepared_geometry(previous_geometry, geometry):
                    geometry = previous_geometry
                previous_geometry = geometry
                params, opt_state, _ = step(params, opt_state, geometry)

            if started:
                if automatic:
                    with self._conditions(fixed_conditions | conditions(params)):
                        pass
                elif named_attributes:
                    for name, value in params.items():
                        setattr(self, name, value)
        finally:
            if started:
                self._forget_the_solve()

        return params


#: The flattened children, in order, by the name each takes on `__init__`. `polarization` and
#: `psi` are the two that cannot ride through the constructor -- see `_tree_unflatten` -- and
#: are popped there; everything else goes in as a keyword, so this order is a *listing* rather
#: than a contract with `__init__`'s signature.
CHILD_NAMES = ('top', 'bot', 'theta', 'phi', 'psi', 'period', 'wavelength', 'structure',
               'thickness', 'chi', 'polarization', 'mu', 'excitation', 'kx', 'ky', 'lattice',
               'structure_drawn_on')

#: Read through the private slot rather than the property: these report a *resolved* value,
#: and a round trip has to carry what was asked for. `lattice` materialises `diag(period)` on
#: a solver given only a period, which is not a basis anybody named; its slot is `None` there.
_SLOTS = {'chi': '_chi', 'polarization': '_polarization', 'mu': '_mu',
          'lattice': '_lattice_basis', 'structure_drawn_on': '_structure_drawn_on'}


@_eager_under_torch_compile
@tree_util.register_pytree_node_class
class _ShapeStructure:
    """Internal pytree view: shape strings are declarations, numbers remain children.

    Preserve JAX's treedef for lists, dictionaries, and Materials, including None
    slots. Only string leaves move to aux data; freezing the entire shape recipe
    would silently remove geometry and material derivatives.
    """
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        leaves, tree = tree_util.tree_flatten(self.value)
        static = tuple((i, leaf) for i, leaf in enumerate(leaves) if isinstance(leaf, str))
        dynamic = tuple(leaf for leaf in leaves if not isinstance(leaf, str))
        return dynamic, (tree, static)

    @classmethod
    def tree_unflatten(cls, aux, children):
        tree, static = aux
        declarations = dict(static)
        values = iter(children)
        leaves = [declarations[i] if i in declarations else next(values)
                  for i in range(tree.num_leaves)]
        return cls(tree_util.tree_unflatten(tree, leaves))


@_eager_under_torch_compile
class SolverJax(RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

    def _declaration_checked_for_the_trace(self):
        """Compare a ``Lattice.drawn`` structure with a concrete cell now; ``True`` if it held.

        ``False`` when there is no declaration, or either side has no value to compare -- the
        declaration is a child too, so a solver flattened inside a trace holds a traced one --
        and a rebuilt copy then refuses a traced side rather than trusting it. A cell that
        disagrees with the declaration is refused here, by name, as ``solve`` refuses it.
        """
        if (self._structure_drawn_on is None or self._read_cell() is None
                or _concrete_2x2(self._structure_drawn_on) is None):
            return False
        self._refuse_a_raster_drawn_on_another_cell()
        return True

    def _tree_flatten(self):
        # `polarization` is a *value* of the incidence, so it is a child rather than aux data.
        # It used to ride in aux, which made it a compile-time constant: differentiating a
        # solver whose incidence was a pair returned exactly `0j` for it -- not an error, a
        # zero, and indistinguishable from the true zeros the same slot produces (a pure-s
        # incidence on a mirror-symmetric structure gives one). It rides last so the leading
        # tuple still lines up with `__init__`'s positional parameters; `None` is an empty
        # subtree, so a solver without a pair flattens exactly as it did before.
        #
        # `top` and `bot` are the two Materials, not the derived indices. They ride as
        # *children* because a Material is a registered pytree whose children are
        # `(epsilon, mu)`, so a swept or differentiated half-space permittivity is a traced
        # leaf here rather than a compile-time constant hashed into the cache key -- the same
        # argument `polarization` makes just below. Handing back the derived `n_top` instead
        # would also be lossy: `sqrt(eps*mu)` cannot be taken apart again, so a magnetic
        # half-space would come back through `_tree_unflatten` as a non-magnetic one of a
        # different permittivity, silently.
        #
        # `excitation` is a child, and JAX's own dict handling is what makes that the right
        # slot: a dict flattens with its **keys in the treedef** and its **values as leaves**,
        # which is exactly the split the property documents -- "they may be traced ... while
        # the keys may not, because which entries of the incident column are non-zero decides
        # array structure rather than a value". Amplitudes are a legitimate design variable,
        # so putting the whole dict in aux would have made a differentiated beam weight
        # return a compile-time constant, the way `polarization` did above.
        #
        # It was in *neither* slot until now, which is worse than the wrong slot: the jitted
        # `solve_field` rebuilds `self` on every call, so a multimode scene came back with
        # `excitation = None` and the solve answered the single-specular-wave problem.
        # Measured on a 1D grating, eager against jitted: the field parted by 1.22 and R read
        # 0.148814934 where eager gave 0.164651118 -- and 0.148814934 is exactly the R of the
        # same scene built with no excitation at all, which is the tell. `order_window`
        # narrowed with it, ((0, 3), (0, 0)) -> ((0, 2), (0, 0)).
        #
        # `mu` is the raster permeability field, a child for the same reason `structure` is:
        # it is material values, and material values are what gradients are for. It rides
        # last, after `polarization`, so the leading tuple still lines up with `__init__`'s
        # positional parameters; `None` is an empty subtree, so a non-magnetic solver
        # flattens exactly as it did before.
        # The names, beside the values, because `_tree_unflatten` rebuilds by keyword. They
        # used to be matched by *position* against `__init__`'s leading parameters, which made
        # the two orders one undeclared contract: adding a positional parameter anywhere but
        # the end would have shifted every child by one and rebuilt a different solver, in
        # silence. Two comments below warned about it; now nothing has to.
        #
        # `lattice` is a child, last, and the `Lattice` in the slot is itself a registered
        # pytree: its two vectors are leaves and `axis_aligned` -- the one fact about a cell
        # that decides a branch -- is its aux data, so it still keys the compile. It rode here
        # as a nested tuple in aux until 2026-09-24, which made every basis a compile-time
        # constant: a concrete cell sweep through the jitted `solve_field` compiled once per
        # value, and a *traced* basis -- `jax.grad` of `solve_field(lattice=L)` -- could not be
        # flattened at all (`.tolist()` on a tracer, `ConcretizationTypeError`). Nothing inside
        # the trace needs the numbers concrete: the frozen geometry carries the discrete
        # decisions, and the one check that compares a basis, `Lattice.drawn`'s, runs here
        # while the cell is still concrete (`drawn_checked` below).
        #
        # `structure_drawn_on` -- the cell a `Lattice.drawn(...)` structure declared -- is a
        # child beside it, last, for the same reason. It rode in aux as a nested tuple, so a
        # declared structure's cell sweep compiled once per cell while the same sweep over a
        # bare array compiled once. Whether a declaration *exists* is still in the treedef --
        # `None` is an empty subtree -- and so is whether it was compared (`drawn_checked`);
        # only the numbers are values. `_tree_unflatten` restores it to the slot, not through
        # the constructor, which has no keyword for it.
        children = tuple(getattr(self, _SLOTS[n], None) if n in _SLOTS else getattr(self, n)
                         for n in CHILD_NAMES)
        children = tuple(self._material_input(name) if name in ('top', 'bot', 'structure')
                         else value for name, value in zip(CHILD_NAMES, children))
        if isinstance(self.structure, list):
            children = tuple(_ShapeStructure(value) if name == 'structure' else value
                             for name, value in zip(CHILD_NAMES, children))
        aux_data = {
            'backend': self.backend,
            'tetm': self.tetm,
            'truncation_order': self.truncation_order,
            'layer_algorithm': self.layer_algorithm,
            'device': self.device,
            'complex_dtype': self.complex_dtype,
            # The *requested* values, not the resolved ones the properties report. Both axes
            # resolve against the modeling path, and a pytree round-trip has to reconstruct the
            # solver it flattened: writing the resolved value here would hand the rebuilt solver
            # an explicit request nobody made, so a structure that had merely accepted its
            # path's only option would come back out asserting it -- and, on a solver that did
            # name an unavailable one, the refusal would fire here, inside JAX's tree
            # machinery, rather than at the solve that is actually being refused.
            'fourier_coefficients': self._fourier_coefficients,
            'factorization': self._factorization,
            'raster_normal_method': self._normal_method,
            # Beside it, and for the same reason: the *raw* slot, so an unset axis comes back
            # unset. `solve_field` flattens `self` on every call, so a resolved value written
            # here would come back as an explicit request and `_reject_unread_projector` would
            # then refuse a raster solver for a value the library itself put there -- which is
            # exactly what `shape_nvf_grid_size` did, one entry down.
            'nvf_projector': self._projector,
            # The raw slot again, for the same reason, one rule over: an unset nesting comes back
            # unset, so a rebuilt NVF solver is not handed a request `_reject_unread_nesting`
            # would refuse. Aux rather than a child because the builds branch on it.
            'li_nesting': self._nesting,
            # Whether the `Lattice.drawn` declaration (a child, above) was compared against the
            # cell here, while both were concrete. Inside the rebuilt copy both are traced
            # children and cannot be compared, so the copy trusts this answer for as long as it
            # holds the objects it was rebuilt with (`_cell_checked_when_flattened`). A moved
            # cell is refused here, by name, before any trace starts.
            'drawn_checked': self._declaration_checked_for_the_trace(),
            # A dispatch key like `pol`: the solves branch on it (which cascade blocks are
            # read, which way the stack is walked), so it must come back static, never traced.
            'incidence_side': self.incidence_side,
            # The last settable condition to reach this dict, 2026-08-30. It is a *declaration*
            # -- `'first_order'` asks for the first-order form where `auto` would have picked the
            # second-order one -- and a declaration that does not survive the round trip is worse
            # than one that is refused: with the key absent, two solvers differing only here
            # produce identical aux_data, so their treedefs coincide and **one jit compilation
            # serves both**.
            # Measured on a 16x16 raster with `answers='field'`: declaring `'first_order'` after
            # `auto` did not recompile (0.05 s against 0.93 s) and returned `auto`'s number to the
            # last digit. The declared cross-check silently became a no-op, and the declared
            # *refusal* -- `'second_order'` over a z row -- silently did not fire. It cannot
            # produce a wrong answer (max |T_second - T_first| = 1.4e-14 on a patterned raster, and
            # each override degrades to the correct path), which is why it outranks nothing; it is
            # here because a declaration must mean the same thing on both sides of a trace.
            # It rides as the *raw slot*: `rcwa()` does not take `formulation` as a keyword,
            # so `_tree_unflatten` restores it directly rather than through the constructor.
            'formulation': self._formulation,
            # Aux data, and it has to be. A plain instance attribute survives __init__ but
            # nothing puts it here, so `_tree_unflatten` never passes it to the constructor
            # and the rebuilt solver takes the default: measured on this pair, a solver that
            # declared 'field' comes back reporting 'amplitude' and retaining 0.000 MB where
            # it should hold 1.253. And with the key absent the two declarations produce
            # *identical* aux_data, so their treedefs coincide and one jit compilation serves
            # both -- the loss cannot even be recovered by recompiling. Static here, so the
            # core's `covers` branches are read at trace time, the only time they can be.
            # See docs/what-a-solve-answers.md section 6.
            'answers': self.answers,
            # Not the pair itself (that is a child now) but the fact that `psi` was given
            # explicitly, which the always-present psi child would otherwise overwrite.
            'psi_given': self._psi_given,
            # `normal_grid` was missing beside the cell basis, and both were silent when they
            # were: the compiled `solve_field` flattens `self` on every call, so the solve ran
            # on whatever the *rebuilt* solver carries, not on what the caller configured.
            # Measured on a 60-degree cell: the basis came back `None`,
            # `lattice_is_axis_aligned` flipped False -> True (taking
            # `_reject_unsupported_lattice`'s guard with it), and the jitted call answered the
            # rectangular-cell problem -- T = 0.6565 against a correct 0.5460, energy conserving
            # throughout. `shape_nvf_grid_size` fell 4096 -> 1024, which moves a shape gradient's
            # sawtooth floor by an order of magnitude. The basis is a child now (see
            # `CHILD_NAMES`); `shape_nvf_grid_size` sizes an array, so it stays here.
            #
            # The **raw** slot, not the property. `shape_nvf_grid_size` reads back resolved --
            # `None` becomes `max(1024, 4*truncation_order+1)` -- and putting that in aux_data
            # makes the round trip rewrite what the caller asked for: an unset grid comes back
            # as an explicit one, and `_reject_unread_normal_grid` then refuses a raster solver
            # for a value the library itself put there. The compiled `solve_field` flattens
            # `self` on every call, so it took out 23 tests at once.
            'shape_nvf_grid_size': self._normal_grid,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # `polarization` cannot ride through __init__: the psi child is always present (it
        # is the traced leaf, non-None even when it was merely derived from `pol`), so the
        # constructor would see both spellings of the incidence and refuse a state that was
        # legal when it was flattened. Rebuild without it, then restore the slot directly.
        aux_data = dict(aux_data)
        psi_given = aux_data.pop('psi_given')
        # `formulation` is settable on the instance and on `solve()`, but not on `rcwa()`,
        # so it cannot go through the constructor either. Same treatment as `polarization`.
        formulation = aux_data.pop('formulation')
        drawn_checked = aux_data.pop('drawn_checked')
        named = dict(zip(CHILD_NAMES, children))
        # Same treatment, same reason: `structure` is assigned inside `__init__` and would
        # overwrite the slot, so the declaration is restored after the rebuild rather than
        # handed in. `__init__` has no keyword for it either -- it is a fact the structure
        # carried, not a parameter anyone passes.
        drawn_on = named.pop('structure_drawn_on')
        if isinstance(named['structure'], _ShapeStructure):
            named['structure'] = named['structure'].value
        polarization = named.pop('polarization')
        # `psi` too, and for the same kind of reason: the slot holds a value whenever `psi`
        # was not declared away, and the constructor reads a `psi` keyword as a *given* one.
        # So a solver whose `tetm` was assigned after it was built -- the slot still holding
        # the default angle, which is what the setter leaves -- came back through
        # `cls(psi=..., tetm='te')` and was refused as naming the incidence twice, inside
        # JAX's tree machinery: `solve_field(tetm='te')` raised there on every 1D solver built
        # without it (measured 2026-09-24). Restored as the slot it was, beside `_psi_given`.
        psi = named.pop('psi')
        # `period` and `lattice` are two spellings of one cell and naming both is refused at
        # the constructor, but this path holds both by construction -- both are children,
        # `period` because it is a design variable on a rectangular cell and the basis because
        # it is one on any cell. Dropping it here is provably a no-op rather than a loss:
        # `__init__` assigns `self.lattice` *after* `self.period`, and that setter overwrites
        # the period slot with `periods_of(A)`, so a period passed beside a basis has never
        # reached anything.
        if named.get('lattice') is not None:
            named.pop('period', None)
        solver = cls(**named, **aux_data)
        solver._formulation = formulation
        # Straight into the slot, bypassing the setter's validation: the pair may now be a
        # tuple of tracers, and the setter's `(0, 0)` check reads concrete values. It already
        # guards that with a try/except, but there is nothing left to validate here anyway --
        # this pair was validated when the solver it came from was built.
        solver._polarization = polarization
        solver._psi = psi
        solver._psi_given = psi_given
        solver._structure_drawn_on = drawn_on
        if drawn_checked:
            solver._drawn_checked_against = solver._cell_objects()

        return solver


# Registered here rather than in a package `__init__`, which is where it lived: the class is
# defined in this module, so the registration cannot be reached without it and cannot run
# before it. `SolverJax` is a pytree because `_solve_field_compiled`, the body of `solve_field`,
# is `jax.jit`-decorated at class level with `self` at argnum 0 -- every call flattens the
# solver and rebuilds it through `__init__`. What that costs when a slot is forgotten is in
# `_tree_flatten`'s comments.
tree_util.register_pytree_node(SolverJax,
                               SolverJax._tree_flatten,
                               SolverJax._tree_unflatten)
# The cell basis is a child of that tree, so its own class has to be a pytree before the first
# flatten. `Lattice` registers itself lazily when one is built with JAX loaded, and a basis
# built *before* this module was imported -- `Lattice.hexagonal(400.)` ahead of the first
# `rcwa(backend='jax')` -- is handed on as it is, so nothing would register it in time.
_lattice_module._register_pytree_if_jax_loaded()
