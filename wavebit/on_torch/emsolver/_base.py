import torch
import array_api_compat.torch as _xp

import numpy as np


from ..._core.engine import CoreSolveMixin
from wavebit._core import conventions as _conventions
from wavebit._core.backend import check_input_precision, forward_strided


def _as_tensor(v, device=None, dtype=None):
    """``v`` as a torch tensor, keeping any autograd graph intact.

    A second copy of :func:`wavebit._core.backend.as_array`, kept only until its remaining
    callers -- the ``period`` and ``thickness`` setters and ``_coerce_raster`` -- fold into
    ``_core`` alongside the angles. Verified equivalent to ``as_array`` across eleven
    value/dtype combinations including a grad-carrying tensor, so the two can be swapped one
    caller at a time rather than all at once.
    """
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype)
    return torch.tensor(forward_strided(v), device=device, dtype=dtype)


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

    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 truncation_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM', excitation=None,
                 device='cpu', complex_dtype=torch.complex128, polarization=None):

        # device
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')

        # complex_dtype
        if complex_dtype in (0, torch.complex128, np.complex128):
            self._complex_dtype = torch.complex128
        elif complex_dtype in (1, torch.complex64, np.complex64):
            self._complex_dtype = torch.complex64
        else:
            raise ValueError('Torch complex_dtype')

        self._float_dtype = (torch.float64 if self._complex_dtype is not torch.complex64
                             else torch.float32)
        self._int_dtype = torch.int64 if self._complex_dtype is not torch.complex64 else torch.int32

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

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device == 0:
            self._device = torch.device('cpu')
        elif device == 1:
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError

        if hasattr(self, 'modeler') and self.modeler is not None:
            self.modeler.device = self._device

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, complex_dtype):
        if complex_dtype in (0, torch.complex128, np.complex128):
            self._complex_dtype = torch.complex128
        elif complex_dtype in (1, torch.complex64, np.complex64):
            self._complex_dtype = torch.complex64
        else:
            raise ValueError('complex_dtype')

        self._float_dtype = (torch.float64 if self.complex_dtype is not torch.complex64
                             else torch.float32)
        self._int_dtype = (torch.int64 if self.complex_dtype is not torch.complex64
                           else torch.int32)
        self.theta = self.theta
        self.phi = self.phi
        self._psi = self.psi

        self.truncation_order = self.truncation_order
        self.thickness = self.thickness

        if hasattr(self, 'modeler') and self.modeler is not None:
            self.modeler.complex_dtype = self._complex_dtype

    @property
    def float_dtype(self):
        return self._float_dtype

    @property
    def int_dtype(self):
        return self._int_dtype

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

