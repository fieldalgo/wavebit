import functools
import jax

import jax.numpy as jnp
import numpy as np

from ..._core.engine import CoreSolveMixin, _POL_RETIRED
from wavebit._core.backend import check_input_precision


def jax_device_set(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        self, *_ = args
        with jax.default_device(self.device[0]):
            res = func(*args, **kwargs)
            return res

    return wrap


class _BaseRCWA(CoreSolveMixin):

    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None, fto=(0, 0), pol=None,
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='ETM',
                 device=0, type_complex=jnp.complex128, polarization=None):

        self.device = device

        # type_complex
        if type_complex in (0, jnp.complex128, np.complex128):
            self._type_complex = jnp.complex128
        elif type_complex in (1, jnp.complex64, np.complex64):
            self._type_complex = jnp.complex64
        else:
            raise ValueError('JAX type_complex')

        # currently these two are not used. Only the torch backend uses them.
        self._type_float = jnp.float64 if self._type_complex is not jnp.complex64 else jnp.float32
        self._type_int = jnp.int64 if self._type_complex is not jnp.complex64 else jnp.int32

        self.n_top = n_top
        self.n_bot = n_bot

        if pol is not None:
            raise ValueError(_POL_RETIRED)
        # degree to radian due to JAX JIT
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
        if device in (0, 'cpu'):
            self._device = jax.devices('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = jax.devices('gpu')
        # Already-resolved devices, which is what the getter hands back — so assigning
        # `self.device = self.device` has to round-trip. Test against the public
        # ``jax.Device``: matching the repr of the implementation class instead broke
        # silently when jaxlib moved it from ``xla_extension`` to ``_jax``.
        elif isinstance(device, jax.Device):
            self._device = [device]
        elif (isinstance(device, (list, tuple)) and device
                and all(isinstance(d, jax.Device) for d in device)):
            self._device = list(device)
        else:
            raise ValueError(f'unsupported jax device: {device!r}')

    @property
    def type_complex(self):
        return self._type_complex

    @type_complex.setter
    def type_complex(self, type_complex):
        # type_complex
        if type_complex in (0, jnp.complex128, np.complex128):
            self._type_complex = jnp.complex128
        elif type_complex in (1, jnp.complex64, np.complex64):
            self._type_complex = jnp.complex64
        else:
            raise ValueError('JAX type_complex')

        self._type_float = jnp.float64 if self.type_complex is not jnp.complex64 else jnp.float32
        self._type_int = jnp.int64 if self.type_complex is not jnp.complex64 else jnp.int32
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
            check_input_precision('theta', theta, self.type_complex)
            self._theta = jnp.array(theta, dtype=self.type_complex)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            check_input_precision('phi', phi, self.type_complex)
            self._phi = jnp.array(phi, dtype=self.type_complex)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi_set_explicitly()
            self._reject_complex_angle(psi)
            # Real dtype: see the numpy twin, and `CoreSolveMixin._reject_complex_angle`.
            self._psi = jnp.array(jnp.real(psi), dtype=self.type_float)

    def _as_complex(self, value):
        """An incident amplitude in the solve's complex dtype."""
        return jnp.array(value, dtype=self.type_complex)

    def _as_real_angle(self, value):
        """An angle in the real dtype — the storage `chi` shares with `psi`."""
        return jnp.array(jnp.real(value), dtype=self.type_float)

    def _default_psi(self):
        """Pure s, applied when no incidence was named and none was declared away."""
        self._psi = jnp.array(jnp.pi / 2, dtype=self.type_float)

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
        elif isinstance(fto, (np.ndarray, jnp.ndarray)):
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
            self._period = jnp.array([period, period], dtype=self.type_float)
        elif type(period) in (list, tuple, np.ndarray) or isinstance(period, jnp.ndarray):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = jnp.array(period, dtype=self.type_float)
        elif type(period) is jax.interpreters.partial_eval.DynamicJaxprTracer:
            print('init period')
            jax.debug.print('init period')
            self._period = period
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
            self._thickness = jnp.array([thickness], dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = jnp.array(thickness, dtype=self.type_float)
        elif isinstance(thickness, jnp.ndarray):
            self._thickness = jnp.array(thickness, dtype=self.type_float)
        elif type(thickness) is jax.interpreters.partial_eval.DynamicJaxprTracer:
            print('init period')
            jax.debug.print('init period')
            self._thickness = thickness
        else:
            raise ValueError


    @jax_device_set
    def get_kx_ky_vector(self, wavelength):

        fto_x_range = jnp.arange(-self.fto[0], self.fto[0] + 1)
        fto_y_range = jnp.arange(-self.fto[1], self.fto[1] + 1)

        def adjust_theta():
            theta = 0 if self.theta is None else self.theta
            # https://github.com/numpy/numpy/issues/27306
            check = theta.real >= jnp.float32(jnp.pi / 2)
            sin_theta_true_case = jnp.sin(
                jnp.nextafter(jnp.float32(jnp.pi / 2), jnp.float32(0)) + theta.imag * jnp.complex64(1j))
            sin_theta_false_case = jnp.sin(theta)
            return jnp.where(check, sin_theta_true_case, sin_theta_false_case)

        sin_theta = adjust_theta()

        phi = 0 if self.phi is None else self.phi  # phi is None -> 1D TE TM case

        # n_inc, not n_top: theta lives in the incidence medium, which incidence_side picks.
        kx = (self.n_inc * sin_theta * jnp.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).astype(self.type_complex)

        ky = (self.n_inc * sin_theta * jnp.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).astype(self.type_complex)

        return kx, ky

