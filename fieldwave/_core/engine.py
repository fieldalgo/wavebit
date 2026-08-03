"""Bridge from the solver classes to the single-source core.

``CoreSolveMixin`` reimplements the three ``_BaseRCWA`` solve methods on top of
:mod:`fieldwave._core.solve_1d` / :mod:`fieldwave._core.solve_2d`. It keeps their exact contract —
same arguments, same returned result mapping, and the same ``layer_info_list`` / ``T1``
side-effects that field reconstruction reads — so mixing it in swaps the *implementation*
without touching the public API.

Because the core is backend-agnostic, one mixin serves NumPy, JAX and PyTorch alike; the
per-backend ``transfer_method`` chains it replaces existed in triplicate.
"""
from .solve_1d import solve_1d as _solve_1d, solve_1d_conical as _solve_1d_conical
from .solve_2d import solve_2d as _solve_2d


class CoreSolveMixin:
    """Provides ``solve_1d`` / ``solve_1d_conical`` / ``solve_2d`` backed by the core."""

    def _check_algo(self):
        if self.connecting_algo != 'TMM':
            raise ValueError(f'connecting_algo {self.connecting_algo!r} is not supported')

    def _finish(self, out):
        """Store the field-reconstruction state and return just the result mapping."""
        self.layer_info_list = out.pop('layer_info_list')
        self.T1 = out.pop('T1')
        return out

    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self._check_algo()
        kx, _ = self.get_kx_ky_vector(wavelength)
        return self._finish(_solve_1d(
            self.pol, wavelength, kx, self.n_top, self.n_bot, self.theta, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            use_pinv=self.use_pinv, full=True))

    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self._check_algo()
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(_solve_1d_conical(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            use_pinv=self.use_pinv, full=True))

    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None):
        self._check_algo()
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(_solve_2d(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
            use_pinv=self.use_pinv, full=True))
