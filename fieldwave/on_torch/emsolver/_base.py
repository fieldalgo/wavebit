import torch

import numpy as np


from ..._core.engine import CoreSolveMixin


class _BaseRCWA(CoreSolveMixin):
    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, pol=0., fto=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-20,
                 device='cpu', type_complex=torch.complex128, use_pinv=False):

        # device
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')

        # type_complex
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('Torch type_complex')

        self._type_float = torch.float64 if self._type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self._type_complex is not torch.complex64 else torch.int32
        self.perturbation = perturbation

        self.n_top = n_top
        self.n_bot = n_bot

        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.psi = psi

        self.fto = fto
        self.period = period
        self.wavelength = wavelength
        self.thickness = thickness
        self.connecting_algo = connecting_algo
        self.use_pinv = use_pinv

        self.layer_info_list = []
        self.T1 = None

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
    def type_complex(self):
        return self._type_complex

    @type_complex.setter
    def type_complex(self, type_complex):
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('type_complex')

        self._type_float = torch.float64 if self.type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self.type_complex is not torch.complex64 else torch.int32
        self.theta = self.theta
        self.phi = self.phi
        self._psi = self.psi

        self.fto = self.fto
        self.thickness = self.thickness

        if hasattr(self, 'modeler') and self.modeler is not None:
            self.modeler.type_complex = self._type_complex

    @property
    def type_float(self):
        return self._type_float

    @property
    def type_int(self):
        return self._type_int

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        if theta is None:
            self._theta = None
        else:
            self._theta = torch.tensor(theta, device=self.device, dtype=self.type_complex)
            self._theta = torch.where(self._theta == 0, self.perturbation, self._theta)  # perturbation

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            self._phi = torch.tensor(phi, device=self.device, dtype=self.type_complex)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = torch.tensor(psi, dtype=self.type_complex)
            pol = -(2 * psi / torch.pi - 1)
            self._pol = pol

    @property
    def pol(self):
        """
        portion of TM. 0: full TE, 1: full TM

        Returns: polarization ratio

        """
        return self._pol

    @pol.setter
    def pol(self, pol):
        if not 0 <= pol <= 1:
            raise ValueError

        self._pol = pol
        psi = torch.tensor(torch.pi / 2 * (1 - self.pol), device=self.device, dtype=self.type_complex)
        self._psi = psi

    @property
    def fto(self):
        return self._fto

    @fto.setter
    def fto(self, fto):

        if isinstance(fto, (list, tuple)):
            if len(fto) == 1:
                self._fto = [int(fto[0]), 0]
            elif len(fto) == 2:
                self._fto = [int(v) for v in fto]
            else:
                raise ValueError('Torch fto')
        elif isinstance(fto, (np.ndarray, torch.Tensor)):
            fto_list = fto.tolist()
            if isinstance(fto_list, list):
                if len(fto_list) == 1:
                    self._fto = [int(fto_list[0]), 0]
                elif len(fto_list) == 2:
                    self._fto = [int(v) for v in fto_list]
                else:
                    raise ValueError('Torch fto')
            elif isinstance(fto_list, (int, float)):
                self._fto = [int(fto_list), 0]
            else:
                raise ValueError('Torch fto')
        elif isinstance(fto, (int, float)):
            self._fto = [int(fto), 0]
        else:
            raise ValueError('Torch fto')

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = torch.tensor([period, period], device=self.device, dtype=self.type_float)
        elif type(period) in (list, tuple, np.ndarray) or isinstance(period, torch.Tensor):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = torch.tensor(period, device=self.device, dtype=self.type_float)
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
            self._thickness = torch.tensor([thickness], device=self.device, dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = torch.tensor(thickness, device=self.device, dtype=self.type_float)
        elif type(thickness) is torch.Tensor:
            self._thickness = thickness.to(device=self.device, dtype=self.type_float)
        else:
            raise ValueError

    def get_kx_ky_vector(self, wavelength):

        fto_x_range = torch.arange(-self.fto[0], self.fto[0] + 1, device=self.device,
                                   dtype=self.type_float)
        fto_y_range = torch.arange(-self.fto[1], self.fto[1] + 1, device=self.device,
                                   dtype=self.type_float)

        theta = torch.tensor(0, device=self.device, dtype=self.type_complex) if self.theta is None else self.theta

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
            phi = torch.tensor(0, device=self.device, dtype=self.type_complex)
        else:
            phi = self.phi

        kx = (self.n_top * sin_theta * torch.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).type(self.type_complex).conj()

        ky = (self.n_top * sin_theta * torch.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).type(self.type_complex).conj()

        return kx, ky

