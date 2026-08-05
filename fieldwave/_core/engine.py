"""Bridge from the solver classes to the single-source core.

``CoreSolveMixin`` reimplements the three ``_BaseRCWA`` solve methods on top of
:mod:`fieldwave._core.solve_1d` / :mod:`fieldwave._core.solve_2d`. It keeps their exact contract —
same arguments, same returned result mapping, and the same ``layer_info_list`` / ``T1``
side-effects that field reconstruction reads — so mixing it in swaps the *implementation*
without touching the public API.

Because the core is backend-agnostic, one mixin serves NumPy, JAX and PyTorch alike; the
per-backend ``transfer_method`` chains it replaces existed in triplicate.
"""
from . import analytic as _analytic
from . import factorization as _fac
from .geometry import StaticGeometry, detect_same_material
from .scattering import smm_1d as _smm_1d, smm_1d_conical as _smm_1d_conical, smm_2d as _smm_2d
from .solve_1d import solve_1d as _solve_1d, solve_1d_conical as _solve_1d_conical
from .solve_2d import solve_2d as _solve_2d

# Connecting algorithm -> the core solve for each grating type. TMM propagates an enhanced
# transmittance matrix; SMM cascades per-layer scattering matrices. Same modes, same boundary
# conventions, same answers (~1e-14); SMM costs ~1.1-1.4x and cannot reconstruct fields. The
# two solves in each row take identical arguments, which is what makes the dispatch a lookup.
_SOLVES = {
    'TMM': {'1d': _solve_1d, 'conical': _solve_1d_conical, '2d': _solve_2d},
    'SMM': {'1d': _smm_1d, 'conical': _smm_1d_conical, '2d': _smm_2d},
}


class CoreSolveMixin:
    """Provides ``solve_1d`` / ``solve_1d_conical`` / ``solve_2d`` backed by the core."""

    def _solve_fn(self, grating_type):
        """The core solve for this instance's ``connecting_algo`` and the given grating type."""
        try:
            return _SOLVES[self.connecting_algo][grating_type]
        except KeyError:
            raise ValueError(
                f'connecting_algo {self.connecting_algo!r} is not supported '
                f'(expected one of {sorted(_SOLVES)})') from None

    _frozen_geometry = None

    def _same_material(self):
        g = self._frozen_geometry
        return None if g is None else g.same_material

    def _finish(self, out):
        """Store the field-reconstruction state and return just the result mapping."""
        self.layer_info_list = out.pop('layer_info_list')
        self.T1 = out.pop('T1')
        return out

    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        solve = self._solve_fn('1d')
        kx, _ = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            self.pol, wavelength, kx, self.n_top, self.n_bot, self.theta, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            same_material=self._same_material(), full=True))

    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        solve = self._solve_fn('conical')
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            same_material=self._same_material(), full=True))

    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None):
        solve = self._solve_fn('2d')
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all,
            same_material=self._same_material(), full=True))


class CoreFactorizationMixin:
    """Provides ``conv_solve`` — factorization through the core, then the core solve.

    Dispatches the same way the per-backend ``conv_solve`` implementations did (modeling type,
    then Fourier type) but builds every convolution matrix with
    :mod:`fieldwave._core.factorization`. Two consequences beyond de-duplication:

    * NVF (``fourier_type=2``) now works on every backend — the per-backend builders had it
      only in NumPy, so JAX and PyTorch used to raise here.
    * The eager geometry steps (NVF normal field, CFS block compression) are computed inside
      the builders, which needs concrete values. For JAX-traced use, call the core builders
      directly with a precomputed ``normal_field`` / ``compression``.

    The analytic shape path (``modeling_type == 2``) is built from parametric shapes rather
    than an array, so there is no input array to read the namespace from — it comes from the
    backend code instead.
    """

    def freeze_geometry(self, **kwargs):
        """Take this structure's discrete geometry decisions once, eagerly.

        Runs the factorization on the current (concrete) ``ucell`` to capture the
        same-material flags, plus whatever the active Fourier strategy needs. Hand the result
        back to :meth:`conv_solve` as ``geometry=`` to make the call traceable.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]
        fto_x, fto_y = self.fto[0], self.fto[1]

        normal_field = compression = None
        if self._modeling_type_assigned == 0:
            if self.fourier_type in (2, 'nvf', 'NVF'):
                normal_field = _fac.normal_field_for_ucell(
                    self.ucell, fto_x, fto_y, enhanced_dfs=self.enhanced_dfs)
            elif self.fourier_type == 1:
                compression = [_fac.compression_indices(self.ucell[i])
                               for i in range(self.ucell.shape[0])]

        geometry = StaticGeometry(normal_field=normal_field, compression=compression)
        epx = self._conv_matrices(geometry)[0]
        if self._modeling_type_assigned == 1:
            geometry.vector_grid = self.modeler.grid_plans
            geometry.shape_choices = self.modeler.shape_choices
        geometry.same_material = detect_same_material(epx, self.n_bot)
        return geometry

    def conv_solve(self, geometry=None, **kwargs):
        """Build the convolution matrices, then solve.

        ``geometry`` is an optional :class:`~fieldwave._core.geometry.StaticGeometry` — the
        discrete decisions frozen by :meth:`freeze_geometry`. Supply it to keep the whole call
        traceable under ``jax.jit`` / ``jax.grad``; omit it and each decision is auto-detected
        eagerly, which needs concrete values.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization sweeps
        self._frozen_geometry = geometry
        return self.solve_for_conv(self.wavelength, *self._conv_matrices(geometry))

    def _conv_matrices(self, geometry=None):
        """Convolution matrices for the current structure: ``(epx, epy, epz_i, epxy, epyx)``."""
        fto_x, fto_y = self.fto[0], self.fto[1]
        epxy_conv_all = epyx_conv_all = None

        if self._modeling_type_assigned == 0:                      # raster
            if self.fourier_type == 0:
                epx_conv_all, epy_conv_all, epz_conv_i_all = _fac.to_conv_mat_raster_discrete(
                    self.ucell, fto_x, fto_y, enhanced_dfs=self.enhanced_dfs)
            elif self.fourier_type == 1:
                epx_conv_all, epy_conv_all, epz_conv_i_all = _fac.to_conv_mat_raster_continuous(
                    self.ucell, fto_x, fto_y,
                    compression=None if geometry is None else geometry.compression)
            elif self.fourier_type in (2, 'nvf', 'NVF'):
                (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
                 epz_conv_i_all) = _fac.to_conv_mat_raster_nvf(
                    self.ucell, fto_x, fto_y,
                    enhanced_dfs=self.enhanced_dfs,
                    normal_field=None if geometry is None else geometry.normal_field)
            else:
                raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        elif self._modeling_type_assigned == 1:                    # vector
            ucell_vector = self.modeling_vector_instruction(
                self.ucell, plans=None if geometry is None else geometry.vector_grid,
                shape_choices=None if geometry is None else geometry.shape_choices)
            epx_conv_all, epy_conv_all, epz_conv_i_all = _fac.to_conv_mat_vector(
                ucell_vector, fto_x, fto_y)

        elif self._modeling_type_assigned == 2:                    # analytic shapes
            xp = _namespace_for_backend(self.backend)
            (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
             epz_conv_i_all) = _analytic.to_conv_mat_analytic_layers(
                self.ucell, fto_x, fto_y, self.period,
                normal_grid=self.analytic_normal_grid,
                type_complex=xp.complex128, xp=xp)

        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        return epx_conv_all, epy_conv_all, epz_conv_i_all, epxy_conv_all, epyx_conv_all


def _namespace_for_backend(backend):
    """Array namespace for a ``Backend`` code.

    The analytic builder starts from parametric shapes, not arrays, so unlike every other
    entry point it cannot infer the namespace from its input.
    """
    from .backend import array_namespace
    if backend == 1:
        import jax.numpy as jnp
        return array_namespace(jnp.zeros(1))
    if backend == 2:
        import torch
        return array_namespace(torch.zeros(1))
    import numpy as np
    return array_namespace(np.zeros(1))
