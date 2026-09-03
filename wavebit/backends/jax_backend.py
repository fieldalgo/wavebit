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

# Above the rest, and before anything here can build an array. See the module docstring.
jax.config.update('jax_enable_x64', True)

from functools import partial                                    # noqa: E402

import jax.numpy as jnp                                          # noqa: E402
import jax.numpy as _xp                                          # noqa: E402
from jax import tree_util                                        # noqa: E402
from tqdm import tqdm                                            # noqa: E402

from .._core.engine import (CoreFactorizationMixin, CoreSolveMixin,  # noqa: E402
                            device_scoped)
from .._core.enums import DIFFERENTIABLE_BACKENDS as _DIFFERENTIABLE_BACKENDS  # noqa: E402
from .._core.result import Result, ResultSub                     # noqa: E402


class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the parametric builder, which starts from
    #: parametric shapes and so cannot infer a namespace from its input -- used to reach a
    #: `_namespace_for_backend(self.backend)` living in `_core` that imported jax and torch by
    #: name and fell through to numpy on an unrecognised code, with no `else`. The solver
    #: already knows which namespace it is; asking it is cheaper than deciding again, and it
    #: keeps `_core` from naming an optional dependency.
    xp = _xp

    #: Which backend this class is, as the integer `rcwa(backend=...)` takes. A declared
    #: member because the shared constructor cannot spell three literals -- and it has to
    #: stay an accepted *keyword* as well: `SolverJax._tree_flatten` puts `backend` in
    #: aux_data, so it is part of the treedef and of every jit cache key, and
    #: `_tree_unflatten` hands it straight back to `__init__` on every jitted call.
    _BACKEND_ID = 1

    #: Derived, not declared: `_core.enums.DIFFERENTIABLE_BACKENDS` is the one table,
    #: so `manifest.py` can publish this without importing this module to ask.
    _DIFFERENTIABLE = _BACKEND_ID in _DIFFERENTIABLE_BACKENDS


    def _device_scope(self):
        """`jax.default_device`, which is how JAX is told where to place what it builds.

        `self.device` is a *list* here -- `jax.devices('cpu')` -- so it is indexed. That is
        the one thing about it that is not obvious, and it is why this is an override rather
        than a value the base could hold.
        """
        return jax.default_device(self.device[0])

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


@jax.tree_util.register_pytree_node_class
class ResultJax(Result):
    """Shared container, registered as a pytree so a jitted method can return one."""


@jax.tree_util.register_pytree_node_class
class ResultSubJax(ResultSub):
    """Shared container; a pytree for the same reason as :class:`ResultJax`."""


class RCWAJax(CoreFactorizationMixin, _BaseRCWA):
    #: The pytree-registered pair, so a jitted `solve_field` can return one. See the
    #: declaration in `CoreFactorizationMixin`.
    _RESULT = ResultJax
    _RESULT_SUB = ResultSubJax

    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5),
             static_argnames=('sample_x', 'sample_y', 'sample_z', 'incidences', 'geometry'))
    @device_scoped
    def solve_field(self, sample_x=20, sample_y=20, sample_z=20, incidences=('given',),
                    geometry=None, **kwargs):
        """Solve and reconstruct the field in one jit-compiled call.

        ``geometry`` is required here, unlike on :meth:`solve`. This method jits over
        ``self``, so the structure arrives as tracers and the discrete decisions inside the
        factorization and the solve have nothing concrete to branch on; a
        :class:`~wavebit._core.geometry.StaticGeometry` from
        :meth:`~wavebit._core.engine.CoreFactorizationMixin.freeze_geometry` supplies them as
        static data. It is a static argument -- reuse one freeze across calls and you compile
        once.

        Because this method always traces, a stale ``geometry`` is replayed on trust here: the
        comparison that would catch it is the thing the freeze exists to avoid. ``solve``
        refuses one eagerly, so check a long fit there, or re-freeze on a schedule.

        ``sample_x`` / ``sample_y`` / ``sample_z`` and ``incidences`` are static too: they set the
        output shape and select which incidences to reconstruct.

        Use :meth:`solve_field_eager` for the eager equivalent that auto-detects.
        """
        if geometry is None:
            raise ValueError(
                'solve_field is jit-compiled and needs the geometry frozen up front:\n'
                '    geometry = solver.freeze_geometry()\n'
                '    result, field_cell = solver.solve_field(..., geometry=geometry)\n'
                'Reuse that one geometry across calls to compile only once. '
                'For the eager, auto-detecting path use solver.solve_field_eager(...).')
        return self._solve_and_field(sample_x, sample_y, sample_z, incidences, geometry,
                                     **kwargs)


# The eager path, which is `CoreFactorizationMixin.solve_field` under its other name -- one
# function, two names, rather than a body written twice. JAX is the only backend that needs
# both, because it is the only one whose `solve_field` is jitted and so demands a frozen
# geometry; the other two have nothing to be eager *relative to*. Bound here rather than inside
# the class because the name it aliases is the one this class overrides.
RCWAJax.solve_field_eager = CoreFactorizationMixin.solve_field


try:
    import optax
except (TypeError, ImportError, ModuleNotFoundError) as e:
    import warnings

    warnings.warn('Importing optax failed. You can run RCWA but not optimization on the JAX backend. '
                  'One possible reason is python version: optax support python>=3.9.')

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
        return loss_value, {k: jnp.conj(v) for k, v in grads.items()}

    def grad(self, parameters, forward, loss_fn):
        params = {name: (getattr(self, name)) for name in parameters}
        _, grads = self._grad(params, forward, loss_fn)
        [setattr(self, name, params[name]) for name in parameters]

        return grads

    def fit(self, parameters, forward, loss_fn, optimizer, iteration=1):
        params = {name: (getattr(self, name)) for name in parameters}
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = self._grad(params, forward, loss_fn)   # already conjugated
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for _ in tqdm(range(iteration)):
            params, opt_state, loss_value = step(params, opt_state)

        [setattr(self, name, params[name]) for name in parameters]

        return params


#: The flattened children, in order, by the name each takes on `__init__`. `polarization` is
#: the one that cannot ride through the constructor -- see `_tree_unflatten` -- and is popped
#: there; everything else goes in as a keyword, so this order is a *listing* rather than a
#: contract with `__init__`'s signature.
CHILD_NAMES = ('top', 'bot', 'theta', 'phi', 'psi', 'period', 'wavelength', 'structure',
               'thickness', 'chi', 'polarization', 'mu', 'excitation')

#: Read through the private slot rather than the property: these three report a *resolved*
#: value, and a round trip has to carry what was asked for.
_PRIVATE = frozenset({'chi', 'polarization', 'mu'})


class SolverJax(RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

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
        children = tuple(getattr(self, '_' + n, None) if n in _PRIVATE else getattr(self, n)
                         for n in CHILD_NAMES)
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
            'normal_method': self._normal_method,
            # Beside it, and for the same reason: the *raw* slot, so an unset axis comes back
            # unset. `solve_field` flattens `self` on every call, so a resolved value written
            # here would come back as an explicit request and `_reject_unread_projector` would
            # then refuse a raster solver for a value the library itself put there -- which is
            # exactly what `normal_grid` did, one entry down.
            'projector': self._projector,
            # A dispatch key like `pol`: the solves branch on it (which cascade blocks are
            # read, which way the stack is walked), so it must come back static, never traced.
            # The cell a `Lattice.drawn(...)` structure declared, if any. A dispatch fact about
            # the *structure*, not a value in it -- and it must survive, or a jitted solve
            # would rebuild without the declaration and stop checking it. Carried as the plain
            # nested list the wrapper stored, so it hashes into the treedef.
            'structure_drawn_on': (None if self._structure_drawn_on is None
                                   else tuple(map(tuple, self._structure_drawn_on.tolist()))),
            'incidence_side': self.incidence_side,
            # The last settable condition to reach this dict, 2026-08-30. It is a *declaration*
            # -- `'4ff'` asks for the four-field form where `auto` would have picked two -- and
            # a declaration that does not survive the round trip is worse than one that is
            # refused: with the key absent, two solvers differing only here produce identical
            # aux_data, so their treedefs coincide and **one jit compilation serves both**.
            # Measured on a 16x16 raster with `answers='field'`: declaring `'4ff'` after `auto`
            # did not recompile (0.05 s against 0.93 s) and returned `auto`'s number to the
            # last digit. The declared cross-check silently became a no-op, and the declared
            # *refusal* -- `'2ff'` over a z row -- silently did not fire. It cannot produce a
            # wrong answer (max |T_2ff - T_4ff| = 1.4e-14 on a patterned raster, and each
            # override degrades to the correct path), which is why it outranks nothing; it is
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
            # Both of these were missing, and both are silent when they are. `solve_field` is
            # `jax.jit`-decorated at class level, so `self` is flattened on every call -- which
            # means the solve ran on whatever the *rebuilt* solver carries, not on what the
            # caller configured. Measured on a 60-degree cell: the basis came back `None`,
            # `lattice_is_axis_aligned` flipped False -> True (taking
            # `_reject_unsupported_lattice`'s guard with it), and the jitted call answered the
            # rectangular-cell problem -- T = 0.6565 against a correct 0.5460, energy conserving
            # throughout. `normal_grid` fell 4096 -> 1024, which moves a shape gradient's
            # sawtooth floor by an order of magnitude.
            #
            # The basis rides as a nested tuple because aux_data has to be hashable and this is
            # a `(2, 2)` ndarray; `lattice.as_lattice` takes it back through `np.asarray`, so
            # the round trip is exact. It belongs in aux rather than in children for the reason
            # its own docstring gives -- a basis decides control flow and reaches SciPy, so it
            # is concrete by construction and can never be a tracer.
            #
            # Ordering note for anyone adding to this dict: `__init__` assigns `lattice` after
            # `period` because the setter derives `period` from the basis. That still holds on
            # the way back in, and it is correct: with a basis in play `period` is a reported
            # quantity rather than an independent one.
            # `.columns`, not `.tolist()` on the basis itself: `_lattice_basis` holds a
            # `Lattice` since the cell became a design variable, and a `Lattice` has no
            # `tolist`. Flattening an oblique solver raised `AttributeError` -- and because
            # `solve_field` is jitted at class level, `self` is flattened on every call, so
            # the *entire jitted field path* was unreachable on a skewed cell. Nothing caught
            # it: jax, an oblique cell and a field reconstruction are three features the suite
            # covered one at a time and never together.
            'lattice': (None if self._lattice_basis is None
                        else tuple(map(tuple, self._lattice_basis.columns.tolist()))),
            # The **raw** slot, not the property. `normal_grid` reads back resolved --
            # `None` becomes `max(1024, 4*truncation_order+1)` -- and putting that in aux_data
            # makes the round trip rewrite what the caller asked for: an unset grid comes back
            # as an explicit one, and `_reject_unread_normal_grid` then refuses a raster solver
            # for a value the library itself put there. `solve_field` is jitted at class level,
            # so this flattens on every call and it took out 23 tests at once.
            'normal_grid': self._normal_grid,
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
        # Same treatment, same reason: `structure` is assigned inside `__init__` and would
        # overwrite the slot, so the declaration is restored after the rebuild rather than
        # handed in. `__init__` has no keyword for it either -- it is a fact the structure
        # carried, not a parameter anyone passes.
        drawn_on = aux_data.pop('structure_drawn_on')
        named = dict(zip(CHILD_NAMES, children))
        polarization = named.pop('polarization')
        # `period` and `lattice` are two spellings of one cell and naming both is refused at
        # the constructor, but this path holds both by construction -- `period` is a child
        # (it has to be, it is a design variable on a rectangular cell) and `lattice` is aux
        # data. Dropping it here is provably a no-op rather than a loss: `__init__` assigns
        # `self.lattice` *after* `self.period`, and that setter overwrites the period slot
        # with `periods_of(A)`, so a period passed beside a basis has never reached anything.
        if named.get('lattice') is not None or aux_data.get('lattice') is not None:
            named.pop('period', None)
        solver = cls(**named, **aux_data)
        solver._formulation = formulation
        # Straight into the slot, bypassing the setter's validation: the pair may now be a
        # tuple of tracers, and the setter's `(0, 0)` check reads concrete values. It already
        # guards that with a try/except, but there is nothing left to validate here anyway --
        # this pair was validated when the solver it came from was built.
        solver._polarization = polarization
        solver._psi_given = psi_given
        # Left as the nested tuple it hashed as; the refusal converts when it compares, so
        # this module needs no numpy of its own.
        solver._structure_drawn_on = drawn_on

        return solver


# Registered here rather than in a package `__init__`, which is where it lived: the class is
# defined in this module, so the registration cannot be reached without it and cannot run
# before it. `SolverJax` is a pytree because `solve_field` is `jax.jit`-decorated at class
# level with `self` at argnum 0 -- every call flattens the solver and rebuilds it through
# `__init__`. What that costs when a slot is forgotten is in `_tree_flatten`'s comments.
tree_util.register_pytree_node(SolverJax,
                               SolverJax._tree_flatten,
                               SolverJax._tree_unflatten)
