import numpy as np

from ..._core.engine import CoreSolveMixin


class _BaseRCWA(CoreSolveMixin):
    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, pol=0., fto=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-20,
                 device=0, type_complex=np.complex128):

        self._device = 0

        # type_complex
        self._type_complex: type
        if type_complex in (0, np.complex128):
            self._type_complex = np.complex128
        elif type_complex in (1, np.complex64):
            self._type_complex = np.complex64
        else:
            raise ValueError('Numpy type_complex')

        # currently these two are not used. Only TorchMeent uses.
        self._type_float = np.float64 if self.type_complex is not np.complex64 else np.float32
        self._type_int = np.int64 if self.type_complex is not np.complex64 else np.int32
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

        self.layer_info_list = []
        self.T1 = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        print('NumpyMeent support only CPU.')

    @property
    def type_complex(self):
        return self._type_complex

    @type_complex.setter
    def type_complex(self, type_complex):
        if type_complex == 0:
            self._type_complex = np.complex128
        elif type_complex == 1:
            self._type_complex = np.complex64
        elif type_complex in (np.complex128, np.complex64):
            self._type_complex = type_complex
        else:
            raise ValueError

        self._type_float = np.float64 if self.type_complex is not np.complex64 else np.float32
        self._type_int = np.int64 if self.type_complex is not np.complex64 else np.int32
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
            self._theta = np.array(theta, dtype=self.type_complex)
            self._theta = np.where(self._theta == 0, self.perturbation, self._theta)  # perturbation

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            self._phi = np.array(phi, dtype=self.type_complex)
        # self._phi = np.array(phi, dtype=self.type_complex) if phi is not None else None

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = np.array(psi, dtype=self.type_complex)  # TODO: complex, QA
            pol = -(2 * psi / np.pi - 1)
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
        psi = np.array(np.pi / 2 * (1 - self.pol), dtype=self.type_complex)
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
                raise ValueError
        elif isinstance(fto, np.ndarray):
            fto_list = fto.tolist()
            if isinstance(fto_list, list):
                if len(fto_list) == 1:
                    self._fto = [int(fto_list[0]), 0]
                elif len(fto_list) == 2:
                    self._fto = [int(v) for v in fto_list]
                else:
                    raise ValueError
            elif isinstance(fto_list, (int, float)):
                self._fto = [int(fto_list), 0]
            else:
                raise ValueError
        elif isinstance(fto, (int, float)):
            self._fto = [int(fto), 0]
        else:
            raise ValueError

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = np.array([period, period], dtype=self.type_float)
        elif type(period) in (list, tuple, np.ndarray):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = np.array(period, dtype=self.type_float)
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
            self._thickness = np.array([thickness], dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = np.array(thickness, dtype=self.type_float)
        else:
            raise ValueError

    def get_kx_ky_vector(self, wavelength):

        fto_x_range = np.arange(-self.fto[0], self.fto[0] + 1)
        fto_y_range = np.arange(-self.fto[1], self.fto[1] + 1)

        theta = 0 if self.theta is None else self.theta

        if theta.real >= np.float32(np.pi / 2):
            # https://github.com/numpy/numpy/issues/27306
            sin_theta = np.sin(np.nextafter(np.float32(np.pi / 2), np.float32(0)) + theta.imag * np.complex64(1j))
        else:
            sin_theta = np.sin(theta)

        phi = 0 if self.phi is None else self.phi  # phi is None -> 1D TE TM case

        kx = (self.n_top * sin_theta * np.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).astype(self.type_complex)

        ky = (self.n_top * sin_theta * np.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).astype(self.type_complex)

        return kx, ky

