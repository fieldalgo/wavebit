"""The PyTorch backend: the namespace, the device handling, and the optimizer mixin.

Was `wavebit/on_torch/`, four files deep. See :mod:`wavebit.backends.numpy_backend` for why the
depth went.
"""
import torch
import array_api_compat.torch as _xp
from tqdm import tqdm

from .._core.engine import CoreFactorizationMixin, CoreSolveMixin
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
    _BACKEND_ID = 2

    #: Derived, not declared: `_core.enums.DIFFERENTIABLE_BACKENDS` is the one table,
    #: so `manifest.py` can publish this without importing this module to ask.
    _DIFFERENTIABLE = _BACKEND_ID in _DIFFERENTIABLE_BACKENDS


    @property
    def _array_device(self):
        """Torch places a tensor at creation, so this is the only place to say where."""
        return self.device

    #: A tensor, so a caller's optimizer leaf reaches the solver as itself.
    _ARRAY_TYPES = (torch.Tensor,)

    def _resolve_device(self, value):
        """A :class:`torch.device`.

        Strings are accepted, which they were not: `rcwa(backend=2, device='cpu')` raised a
        bare `ValueError` while `RCWATorch.__init__` declared `device='cpu'` as its own
        default, so the documented default could not be written out. JAX took the strings all
        along, and `_core.modeler._resolve_device` -- the same mapping, for the modeler --
        has taken them since it was written.
        """
        if value is None or value in (0, 'cpu'):
            return torch.device('cpu')
        if value in (1, 'gpu', 'cuda'):
            return torch.device('cuda')
        if isinstance(value, torch.device):
            return value
        raise ValueError(
            f"device: expected 0 / 'cpu', 1 / 'gpu' / 'cuda', or a torch.device; "
            f'got {value!r}.')


class RCWATorch(CoreFactorizationMixin, _BaseRCWA):
    """The Torch solver. Empty for the same reason as :class:`RCWANumpy`."""


class OptimizerTorch:

    def __init__(self, *args, **kwargs):

        super().__init__()

    def grad(self, parameters, forward, loss_fn):
        leaves = [getattr(self, name) for name in parameters]
        for leaf in leaves:
            leaf.requires_grad = True
            # Torch *accumulates* into `.grad`, and nothing here used to clear it. So the
            # same question asked twice answered twice the truth, three times the third time
            # -- measured exactly 1x, 2x, 3x on a thickness derivative. A gradient that
            # depends on how many times you have asked for one is not a gradient, and the
            # scaling is silent: every step of a hand-written loop moves further than the
            # last, which reads as a learning rate that is merely too high.
            leaf.grad = None

        loss_fn(forward()).backward()
        return {name: getattr(self, name).grad for name in parameters}

    def _build_optimizer(self, _parameters, _opt, *args, **kwargs):
        _parameters_to_fit = [(getattr(self, name)) for name in _parameters]
        res = _opt(_parameters_to_fit, *args, **kwargs)
        return res

    def fit(self, parameters, forward, loss_fn, optimizer, opt_options, iteration=1):
        optimizer = self._build_optimizer(parameters, optimizer, **opt_options)
        [setattr(getattr(self, name), 'requires_grad', True) for name in parameters]

        for _ in tqdm(range(iteration)):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss_value = loss_fn(result)  # Loss

            loss_value.backward()  # Back Prop.
            optimizer.step()

        return [getattr(self, name) for name in parameters]


class SolverTorch(RCWATorch, OptimizerTorch):

    # `(*args, **kwargs)` and nothing else, which is what `SolverNumpy` and `SolverJax` are.
    # This one used to name `device` and `complex_dtype` before them, and naming them here is
    # not a default -- it is a claim on the first two POSITIONS. `rcwa()` forwards `*args`
    # positionally, so `rcwa(0, 1.5, 2.25)` and `rcwa(1, 1.5, 2.25)` build a solver with
    # `top=1.5, bot=2.25` while `rcwa(2, 1.5, 2.25)` read 1.5 as a device and refused. Same
    # call, one backend answering a different question -- and the two names it intercepted are
    # the two that already had defaults one layer down, so it was not adding anything either.
    #
    # The forwarding was equally redundant: `device=0` and `complex_dtype=0` resolve to exactly
    # what `RCWATorch.__init__`'s own `'cpu'` and `torch.complex128` resolve to, through
    # `_resolve_device` and `_resolve_precision`. Removing them leaves construction identical.
    def __init__(self, *args, **kwargs):
        RCWATorch.__init__(self, *args, **kwargs)
        OptimizerTorch.__init__(self, *args, **kwargs)
