import numpy as np

from ..._core.engine import CoreSolveMixin
from wavebit._core.backend import check_input_precision


class _BaseRCWA(CoreSolveMixin):
    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 max_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM',
                 device=0, complex_dtype=np.complex128, polarization=None):

        self._device = 0

        # complex_dtype
        self._complex_dtype: type
        if complex_dtype in (0, np.complex128):
            self._complex_dtype = np.complex128
        elif complex_dtype in (1, np.complex64):
            self._complex_dtype = np.complex64
        else:
            raise ValueError('Numpy complex_dtype')

        # currently these two are not used. Only the torch backend uses them.
        self._float_dtype = np.float64 if self.complex_dtype is not np.complex64 else np.float32
        self._int_dtype = np.int64 if self.complex_dtype is not np.complex64 else np.int32

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
        self._scattering = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        print('The NumPy backend runs on CPU only.')

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, complex_dtype):
        if complex_dtype == 0:
            self._complex_dtype = np.complex128
        elif complex_dtype == 1:
            self._complex_dtype = np.complex64
        elif complex_dtype in (np.complex128, np.complex64):
            self._complex_dtype = complex_dtype
        else:
            raise ValueError

        self._float_dtype = np.float64 if self.complex_dtype is not np.complex64 else np.float32
        self._int_dtype = np.int64 if self.complex_dtype is not np.complex64 else np.int32
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
            self._theta = np.array(theta, dtype=self.complex_dtype)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            check_input_precision('phi', phi, self.complex_dtype)
            self._phi = np.array(phi, dtype=self.complex_dtype)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi_set_explicitly()
            self._reject_complex_angle(psi)
            # Real dtype, unlike `theta`: a complex theta means evanescent incidence and is
            # a feature, while a complex psi is only ever the mistake `_reject_complex_angle`
            # describes. Storing it real is what lets the refusal read a dtype, and it keeps
            # the autodiff leaf real — the imaginary direction it used to carry was the
            # unnormalized cosh path, not physics.
            self._psi = np.array(np.real(psi), dtype=self.float_dtype)

    def _as_complex(self, value):
        """An incident amplitude in the solve's complex dtype."""
        return np.array(value, dtype=self.complex_dtype)

    def _as_real_angle(self, value):
        """An angle in the real dtype — the storage `chi` shares with `psi`."""
        return np.array(np.real(value), dtype=self.float_dtype)

    def _default_psi(self):
        """Pure s, applied when no incidence was named and none was declared away.

        Not marked as *given*, so ``polarization`` may still be assigned afterwards — the
        same courtesy the retired ``pol``-derived ``psi`` had.
        """
        self._psi = np.array(np.pi / 2, dtype=self.float_dtype)

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
                raise ValueError
        elif isinstance(max_order, np.ndarray):
            max_order_list = max_order.tolist()
            if isinstance(max_order_list, list):
                if len(max_order_list) == 1:
                    self._max_order = [int(max_order_list[0]), 0]
                elif len(max_order_list) == 2:
                    self._max_order = [int(v) for v in max_order_list]
                else:
                    raise ValueError
            elif isinstance(max_order_list, (int, float)):
                self._max_order = [int(max_order_list), 0]
            else:
                raise ValueError
        elif isinstance(max_order, (int, float)):
            self._max_order = [int(max_order), 0]
        else:
            raise ValueError

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = np.array([period, period], dtype=self.float_dtype)
        elif type(period) in (list, tuple, np.ndarray):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = np.array(period, dtype=self.float_dtype)
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
            self._thickness = np.array([thickness], dtype=self.float_dtype)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = np.array(thickness, dtype=self.float_dtype)
        else:
            raise ValueError

    def _separable_kxy(self, wavelength):

        max_order_x_range = np.arange(-self.max_order[0], self.max_order[0] + 1)
        max_order_y_range = np.arange(-self.max_order[1], self.max_order[1] + 1)

        theta = 0 if self.theta is None else self.theta

        if theta.real >= np.float32(np.pi / 2):
            # https://github.com/numpy/numpy/issues/27306
            sin_theta = np.sin(np.nextafter(np.float32(np.pi / 2), np.float32(0)) + theta.imag * np.complex64(1j))
        else:
            sin_theta = np.sin(theta)

        phi = 0 if self.phi is None else self.phi  # phi is None -> 1D TE TM case

        # n_inc, not n_top: theta lives in the incidence medium, which incidence_side picks.
        kx = (self.n_inc * sin_theta * np.cos(phi) + max_order_x_range * (
                wavelength / self.period[0])).astype(self.complex_dtype)

        ky = (self.n_inc * sin_theta * np.sin(phi) + max_order_y_range * (
                wavelength / self.period[1])).astype(self.complex_dtype)

        return kx, ky

