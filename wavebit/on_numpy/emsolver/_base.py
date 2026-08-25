import numpy as np
import array_api_compat.numpy as _xp

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

    def _separable_kxy(self, wavelength):
        (cx, hx), (cy, hy) = self.order_window

        max_order_x_range = np.arange(cx - hx, cx + hx + 1)
        max_order_y_range = np.arange(cy - hy, cy + hy + 1)

        k_inc_x, k_inc_y = self._incident_kxy(wavelength)

        kx = (k_inc_x + max_order_x_range * (
                wavelength / self.period[0])).astype(self.complex_dtype)

        ky = (k_inc_y + max_order_y_range * (
                wavelength / self.period[1])).astype(self.complex_dtype)

        return kx, ky

