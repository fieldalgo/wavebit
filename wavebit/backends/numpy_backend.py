"""The NumPy backend: the namespace it computes in, and the classes `rcwa(backend=0)` builds.

Was `wavebit/on_numpy/`, three files and two `__init__.py` deep -- a layout that was accurate
when `on_numpy` held a whole solver. It holds no solver now; the physics is in `_core` and what
is left here is which namespace to call and which integer this backend is.
"""
import array_api_compat.numpy as _xp

from .._core.engine import _NO_AUTODIFF, CoreFactorizationMixin, CoreSolveMixin
from .._core.enums import DIFFERENTIABLE_BACKENDS as _DIFFERENTIABLE_BACKENDS


class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the analytic builder, which starts from
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
    _BACKEND_ID = 0

    #: Derived, not declared: `_core.enums.DIFFERENTIABLE_BACKENDS` is the one table,
    #: so `manifest.py` can publish this without importing this module to ask.
    _DIFFERENTIABLE = _BACKEND_ID in _DIFFERENTIABLE_BACKENDS



class RCWANumpy(CoreFactorizationMixin, _BaseRCWA):
    """The NumPy solver, which is the two mixins and the declared members on `_BaseRCWA`.

    Empty on purpose. Everything this class used to hold -- a 104-line constructor -- is in
    `CoreFactorizationMixin.__init__`, and everything NumPy does differently is a declared
    member one level down. The class survives because it is the name `rcwa(backend=0)` builds
    and the MRO position that pairs the two halves.
    """


class SolverNumpy(RCWANumpy):
    """The NumPy solver. It solves and it differentiates nothing.

    `grad` and `fit` are *here*, refusing, rather than absent. The MRO is why they cannot live
    in `_core` as a shared refusal that the other two override: `CoreSolveMixin` sits **before**
    `OptimizerJax` and `OptimizerTorch` in the MRO of their solvers, so a `grad` defined there
    would shadow the working ones rather than be shadowed by them.
    """

    def __init__(self, *args, **kwargs):
        RCWANumpy.__init__(self, *args, **kwargs)

    def grad(self, *args, **kwargs):
        raise NotImplementedError(_NO_AUTODIFF.format(name='grad'))

    def fit(self, *args, **kwargs):
        raise NotImplementedError(_NO_AUTODIFF.format(name='fit'))
