import torch

import numpy as np


from ..._core.engine import CoreSolveMixin
from wavebit._core.backend import check_input_precision, forward_strided


def _as_tensor(v, device=None, dtype=None):
    """``v`` as a torch tensor, keeping any autograd graph intact.

    ``torch.tensor(x)`` copies and detaches, so wrapping an incidence angle or a period that
    the caller is optimizing would silently drop it from the graph — the solve still returns a
    tensor, just one with nothing behind it. Tensors are moved with ``.to()``; everything else
    is created as before.

    This is the boundary where a caller's value becomes a torch value, so it is also where a
    reversed NumPy view (``np.rot90``, ``np.flip``, ``a[::-1]``) is normalised — torch rejects
    negative strides outright where the other two backends accept them. See
    :func:`wavebit._core.backend.forward_strided`.
    """
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype)
    return torch.tensor(forward_strided(v), device=device, dtype=dtype)


class _BaseRCWA(CoreSolveMixin):
    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 max_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM',
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

        self.max_order = max_order
        self.period = period
        self.wavelength = wavelength
        self.thickness = thickness
        self.layer_algorithm = layer_algorithm

        self.layer_info_list = []
        self.T1 = None
        # SMM only, from `answers='amplitude'` up; see `CoreSolveMixin._finish`.
        self.scattering = None

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

        self.max_order = self.max_order
        self.thickness = self.thickness

        if hasattr(self, 'modeler') and self.modeler is not None:
            self.modeler.complex_dtype = self._complex_dtype

    @property
    def float_dtype(self):
        return self._float_dtype

    @property
    def int_dtype(self):
        return self._int_dtype

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        if theta is None:
            self._theta = None
        else:
            check_input_precision('theta', theta, self.complex_dtype)
            self._theta = _as_tensor(theta, device=self.device, dtype=self.complex_dtype)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            check_input_precision('phi', phi, self.complex_dtype)
            self._phi = _as_tensor(phi, device=self.device, dtype=self.complex_dtype)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi_set_explicitly()
            self._reject_complex_angle(psi)
            # Real dtype: see the numpy twin, and `CoreSolveMixin._reject_complex_angle`.
            # `torch.real` rather than `np.real` on a tensor: it is a no-op on a real one and
            # keeps the autograd graph, which `np.real` would drop on the way out to numpy.
            psi = torch.real(psi) if torch.is_tensor(psi) else np.real(psi)
            self._psi = _as_tensor(psi, device=self.device, dtype=self.float_dtype)

    def _as_complex(self, value):
        """An incident amplitude in the solve's complex dtype."""
        return _as_tensor(value, device=self.device, dtype=self.complex_dtype)

    def _as_real_angle(self, value):
        """An angle in the real dtype — the storage `chi` shares with `psi`."""
        value = torch.real(value) if torch.is_tensor(value) else np.real(value)
        return _as_tensor(value, device=self.device, dtype=self.float_dtype)

    def _default_psi(self):
        """Pure s, applied when no incidence was named and none was declared away."""
        self._psi = torch.tensor(torch.pi / 2, device=self.device, dtype=self.float_dtype)

    @property
    def max_order(self):
        return self._max_order

    @max_order.setter
    def max_order(self, max_order):

        if isinstance(max_order, (list, tuple)):
            if len(max_order) == 1:
                self._max_order = [int(max_order[0]), 0]
            elif len(max_order) == 2:
                self._max_order = [int(v) for v in max_order]
            else:
                raise ValueError('Torch max_order')
        elif isinstance(max_order, (np.ndarray, torch.Tensor)):
            max_order_list = max_order.tolist()
            if isinstance(max_order_list, list):
                if len(max_order_list) == 1:
                    self._max_order = [int(max_order_list[0]), 0]
                elif len(max_order_list) == 2:
                    self._max_order = [int(v) for v in max_order_list]
                else:
                    raise ValueError('Torch max_order')
            elif isinstance(max_order_list, (int, float)):
                self._max_order = [int(max_order_list), 0]
            else:
                raise ValueError('Torch max_order')
        elif isinstance(max_order, (int, float)):
            self._max_order = [int(max_order), 0]
        else:
            raise ValueError('Torch max_order')

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = _as_tensor([period, period], device=self.device, dtype=self.float_dtype)
        elif type(period) in (list, tuple, np.ndarray) or isinstance(period, torch.Tensor):
            if len(period) == 1:
                period = [period[0], period[0]]
            if isinstance(period, (list, tuple)) and any(isinstance(p, torch.Tensor) for p in period):
                # a sequence holding tensors: stack rather than re-wrap, or the caller's
                # graph is lost. Each element is cast first so nothing is created at float32.
                self._period = torch.stack([_as_tensor(p, device=self.device, dtype=self.float_dtype)
                                            for p in period])
            else:
                self._period = _as_tensor(period, device=self.device, dtype=self.float_dtype)
        else:
            raise ValueError

        if hasattr(self, 'modeler') and self.modeler is not None:
            self.modeler.period = self._period

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if type(thickness) in (int, float):
            self._thickness = _as_tensor([thickness], device=self.device, dtype=self.float_dtype)
        elif type(thickness) in (list, tuple, np.ndarray):
            if isinstance(thickness, (list, tuple)) and any(isinstance(t, torch.Tensor) for t in thickness):
                self._thickness = torch.stack([_as_tensor(t, device=self.device, dtype=self.float_dtype)
                                               for t in thickness])
            else:
                self._thickness = _as_tensor(thickness, device=self.device, dtype=self.float_dtype)
        elif type(thickness) is torch.Tensor:
            self._thickness = thickness.to(device=self.device, dtype=self.float_dtype)
        else:
            raise ValueError

    def _separable_kxy(self, wavelength):

        max_order_x_range = torch.arange(-self.max_order[0], self.max_order[0] + 1, device=self.device,
                                   dtype=self.float_dtype)
        max_order_y_range = torch.arange(-self.max_order[1], self.max_order[1] + 1, device=self.device,
                                   dtype=self.float_dtype)

        theta = (torch.tensor(0, device=self.device, dtype=self.complex_dtype)
                 if self.theta is None else self.theta)

        if theta.real >= torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32):
            # https://github.com/numpy/numpy/issues/27306
            sin_theta = torch.sin(
                torch.nextafter(
                    torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32),
                    torch.tensor(0, device=self.device, dtype=torch.float32),
                ) + theta.imag * 1j
            )
        else:
            sin_theta = torch.sin(theta)

        if self.phi is None:
            phi = torch.tensor(0, device=self.device, dtype=self.complex_dtype)
        else:
            phi = self.phi

        # n_inc, not n_top: theta lives in the incidence medium, which incidence_side picks.
        kx = (self.n_inc * sin_theta * torch.cos(phi) + max_order_x_range * (
                wavelength / self.period[0])).type(self.complex_dtype)

        ky = (self.n_inc * sin_theta * torch.sin(phi) + max_order_y_range * (
                wavelength / self.period[1])).type(self.complex_dtype)

        return kx, ky

