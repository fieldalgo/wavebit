import torch
import array_api_compat.torch as _xp

from ..._core.engine import CoreSolveMixin


class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the analytic builder, which starts from
    #: parametric shapes and so cannot infer a namespace from its input -- used to reach a
    #: `_namespace_for_backend(self.backend)` living in `_core` that imported jax and torch by
    #: name and fell through to numpy on an unrecognised code, with no `else`. The solver
    #: already knows which namespace it is; asking it is cheaper than deciding again, and it
    #: keeps `_core` from naming an optional dependency.
    xp = _xp

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

