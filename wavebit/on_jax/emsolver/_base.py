import jax

import jax.numpy as jnp
import jax.numpy as _xp

from ..._core.engine import CoreSolveMixin, device_scoped


class _BaseRCWA(CoreSolveMixin):
    #: The array namespace this solver's arrays live in. A class attribute rather than a
    #: lookup, because the one caller that needs it -- the analytic builder, which starts from
    #: parametric shapes and so cannot infer a namespace from its input -- used to reach a
    #: `_namespace_for_backend(self.backend)` living in `_core` that imported jax and torch by
    #: name and fell through to numpy on an unrecognised code, with no `else`. The solver
    #: already knows which namespace it is; asking it is cheaper than deciding again, and it
    #: keeps `_core` from naming an optional dependency.
    xp = _xp

    def _device_scope(self):
        """`jax.default_device`, which is how JAX is told where to place what it builds.

        `self.device` is a *list* here -- `jax.devices('cpu')` -- so it is indexed. That is
        the one thing about it that is not obvious, and it is why this is an override rather
        than a value the base could hold.
        """
        return jax.default_device(self.device[0])


    #: A JAX array -- and a `DynamicJaxprTracer` too, which is what lets a period be
    #: differentiated or closed over inside `jax.jit`.
    _ARRAY_TYPES = (jnp.ndarray,)

    def __init__(self, top=1., bot=1., theta=0., phi=None, psi=None, chi=None, tetm=None,
                 truncation_order=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), layer_algorithm='ETM', excitation=None,
                 device=0, complex_dtype=jnp.complex128, polarization=None):

        self.device = device

        # The width, once. `_resolve_precision` is in `_core`; it is not the setter, because
        # the setter re-casts every stored array and none of them exist yet.
        self._complex_dtype, self._float_dtype, self._int_dtype = (
            self._resolve_precision(complex_dtype))

        # Materials in, an index out: the two setters store the Material and derive the
        # `sqrt(eps*mu)` every solve reads. See `CoreSolveMixin.top`.
        self.top = top
        self.bot = bot

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

    @device_scoped
    def _separable_kxy(self, wavelength):
        (cx, hx), (cy, hy) = self.order_window

        max_order_x_range = jnp.arange(cx - hx, cx + hx + 1)
        max_order_y_range = jnp.arange(cy - hy, cy + hy + 1)

        k_inc_x, k_inc_y = self._incident_kxy(wavelength)

        kx = (k_inc_x + max_order_x_range * (
                wavelength / self.period[0])).astype(self.complex_dtype)

        ky = (k_inc_y + max_order_y_range * (
                wavelength / self.period[1])).astype(self.complex_dtype)

        return kx, ky

