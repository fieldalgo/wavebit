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

    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 truncation_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM', excitation=None,
                 device='cpu', complex_dtype=torch.complex128, polarization=None):

        self.device = device

        # The width, once. `_resolve_precision` is in `_core`; it is not the setter, because
        # the setter re-casts every stored array and none of them exist yet.
        self._complex_dtype, self._float_dtype, self._int_dtype = (
            self._resolve_precision(complex_dtype))

        # Materials in, an index out: the two setters store the Material and derive the
        # `sqrt(eps*mu)` every solve reads. See `CoreSolveMixin.top`.
        self.top = top
        self.bot = bot

        self.theta = theta
        self.phi = phi
        # Order matters: `tetm` refuses to coexist with a named incidence, so it goes first
        # and the two spellings below raise against it rather than silently winning.
        self.tetm = tetm
        self.psi = psi
        self.chi = chi
        self.polarization = polarization
        if self._tetm is None and self._psi is None and polarization is None:
            self._default_psi()

        self.truncation_order = truncation_order
        self.period = period
        self.wavelength = wavelength
        self.thickness = thickness
        self.layer_algorithm = layer_algorithm
        self.excitation = excitation

        self.layer_info_list = []
        self.T1 = None
        # SMM only, from `answers='amplitude'` up; see `CoreSolveMixin._finish`.
        self._scattering = None

    def _separable_kxy(self, wavelength):
        (cx, hx), (cy, hy) = self.order_window

        max_order_x_range = torch.arange(cx - hx, cx + hx + 1, device=self.device,
                                   dtype=self.float_dtype)
        max_order_y_range = torch.arange(cy - hy, cy + hy + 1, device=self.device,
                                   dtype=self.float_dtype)

        k_inc_x, k_inc_y = self._incident_kxy(wavelength)

        kx = (k_inc_x + max_order_x_range * (
                wavelength / self.period[0])).type(self.complex_dtype)

        ky = (k_inc_y + max_order_y_range * (
                wavelength / self.period[1])).type(self.complex_dtype)

        return kx, ky

