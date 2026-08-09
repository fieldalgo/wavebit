"""Bridge from the solver classes to the single-source core.

``CoreSolveMixin`` reimplements the three ``_BaseRCWA`` solve methods on top of
:mod:`fieldwave._core.solve_1d` / :mod:`fieldwave._core.solve_2d`. It keeps their exact contract —
same arguments, same returned result mapping, and the same ``layer_info_list`` / ``T1``
side-effects that field reconstruction reads — so mixing it in swaps the *implementation*
without touching the public API.

Because the core is backend-agnostic, one mixin serves NumPy, JAX and PyTorch alike; the
per-backend ``transfer_method`` chains it replaces existed in triplicate.
"""
import warnings

from . import analytic as _analytic
from . import boundary as _boundary
from . import factorization as _fac
from .enums import (Factorization, FourierCoefficients, coefficients_of, factorization_of,
                    resolve_fourier_options, to_fourier_type)
from .geometry import StaticGeometry
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
            epx_conv_all, epy_conv_all, epz_conv_i_all, full=True))

    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        solve = self._solve_fn('conical')
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all, full=True))

    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all,
                 epxy_conv_all=None, epyx_conv_all=None):
        solve = self._solve_fn('2d')
        kx, ky = self.get_kx_ky_vector(wavelength)
        return self._finish(solve(
            wavelength, kx, ky, self.n_top, self.n_bot, self.theta, self.psi, self.thickness,
            epx_conv_all, epy_conv_all, epz_conv_i_all,
            epxy_conv_all=epxy_conv_all, epyx_conv_all=epyx_conv_all, full=True))


class CoreFactorizationMixin:
    """Provides ``conv_solve`` — factorization through the core, then the core solve.

    Dispatches the same way the per-backend ``conv_solve`` implementations did (modeling type,
    then the Fourier options) but builds every convolution matrix with
    :mod:`fieldwave._core.factorization`. Two consequences beyond de-duplication:

    * NVF works on every backend — the per-backend builders had it only in NumPy, so JAX and
      PyTorch used to raise here.
    * The eager geometry steps (NVF normal field, CFS block compression) are computed inside
      the builders, which needs concrete values. For JAX-traced use, call the core builders
      directly with a precomputed ``normal_field`` / ``compression``.

    The analytic shape path (``modeling_type == 2``) is built from parametric shapes rather
    than an array, so there is no input array to read the namespace from — it comes from the
    backend code instead.

    Also owns the two Fourier option attributes and the deprecated ``fourier_type`` alias, so
    the three backend constructors resolve them identically.
    """

    _fourier_coefficients = FourierCoefficients.SAMPLED
    _factorization = Factorization.LI

    def _set_fourier_options(self, fourier_coefficients, factorization, fourier_type):
        """Settle ``fourier_coefficients`` / ``factorization`` from a constructor's arguments."""
        self._fourier_coefficients, self._factorization = resolve_fourier_options(
            fourier_coefficients, factorization, fourier_type)

    @property
    def fourier_coefficients(self):
        """How each layer's Fourier coefficients are obtained: ``'sampled'`` | ``'integrated'``."""
        return self._fourier_coefficients

    @fourier_coefficients.setter
    def fourier_coefficients(self, value):
        self._fourier_coefficients = coefficients_of(value)

    @property
    def factorization(self):
        """Which rule closes the discontinuous products: ``'li'`` | ``'nvf'``."""
        return self._factorization

    @factorization.setter
    def factorization(self, value):
        self._factorization = factorization_of(value)

    @property
    def fourier_type(self):
        """Deprecated alias for the ``(fourier_coefficients, factorization)`` pair.

        Reading it raises for ``('integrated', 'nvf')``: that combination is new and has no
        legacy integer, which is the whole reason the keyword was split.
        """
        warnings.warn(
            "'fourier_type' is deprecated and will be removed; read 'fourier_coefficients' "
            "and 'factorization' instead.", DeprecationWarning, stacklevel=2)
        code = to_fourier_type(self._fourier_coefficients, self._factorization)
        if code is None:
            raise ValueError(
                f"fourier_coefficients={self._fourier_coefficients.value!r} with "
                f"factorization={self._factorization.value!r} has no 'fourier_type' code; "
                f"read those two attributes instead.")
        return code

    @fourier_type.setter
    def fourier_type(self, value):
        self._fourier_coefficients, self._factorization = resolve_fourier_options(
            fourier_type=value)


    def freeze_geometry(self, **kwargs):
        """Take this structure's discrete geometry decisions once, eagerly.

        Captures whatever the active Fourier strategy decides from the concrete ``ucell``.
        Hand the result back to :meth:`conv_solve` as ``geometry=`` to make the call
        traceable. The default ('sampled', 'li') build decides nothing, so its frozen geometry
        is empty — passing it is still what keeps ``conv_solve`` off the eager path.

        The record describes *this* structure. Replaying it onto one whose geometry has moved
        is refused rather than answered, eagerly — see :mod:`fieldwave._core.geometry` for what
        each record checks and why a traced replay cannot.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]
        fto_x, fto_y = self.fto[0], self.fto[1]

        # The two axes decide independently what has to be frozen: NVF needs the normal
        # field, the integrated series needs the block compression, and ('integrated', 'nvf')
        # -- the combination the old integer could not spell -- needs both.
        normal_field = normal_field_source = compression = None
        if self._modeling_type_assigned == 0:
            if self._factorization == Factorization.NVF:
                normal_field = _fac.normal_field_for_ucell(
                    self.ucell, fto_x, fto_y)
                normal_field_source = _fac.normal_field_source_for_ucell(
                    self.ucell, fto_x, fto_y)
            if self._fourier_coefficients == FourierCoefficients.INTEGRATED:
                compression = [_fac.compression_indices(self.ucell[i])
                               for i in range(self.ucell.shape[0])]

        geometry = StaticGeometry(normal_field=normal_field, compression=compression,
                                  normal_field_source=normal_field_source)
        if self._modeling_type_assigned == 1:
            # The vector modeler records its plans while it builds, so run the build once to
            # get them -- the *modeler*, not the whole factorization. Freezing used to go
            # through `_conv_matrices` and throw the convolution matrices away, which it paid
            # for only because the same-material detector read them; that detector is gone.
            # Dropping them takes a freeze from 25.6 ms to 10.4 ms, which is what makes
            # re-freezing every step of a fit cheaper than nursing one along.
            self.modeling_vector_instruction(self.ucell)
            geometry.vector_grid = self.modeler.grid_plans
            geometry.shape_choices = self.modeler.shape_choices
        if self._modeling_type_assigned == 2:
            # Only a painted layer that actually overlaps has a plan to freeze. A summed one
            # has no discrete decision in it, and a painted one with nothing overlapping means
            # the same structure either way -- both stay on the closed-form path, so their
            # slots stay None and the replay never looks.
            plans = [_boundary.plan_layer(layer.get('shapes', []))
                     if _analytic.layer_needs_resolving(layer, self.period) else None
                     for layer in self.ucell]
            if any(p is not None for p in plans):
                geometry.boundary_plans = plans
        return geometry

    def conv_solve(self, geometry=None, **kwargs):
        """Build the convolution matrices, then solve.

        ``geometry`` is an optional :class:`~fieldwave._core.geometry.StaticGeometry` — the
        discrete decisions frozen by :meth:`freeze_geometry`. Supply it to keep the whole call
        traceable under ``jax.jit`` / ``jax.grad``; omit it and each decision is auto-detected
        eagerly, which needs concrete values.

        Raises:
            StaleGeometryError: eagerly, when ``geometry`` no longer describes the structure it
                is being replayed onto. A traced call cannot check — the comparison is the
                thing the record exists to avoid — so a long fit should re-freeze on a schedule
                rather than wait to be told.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization sweeps
        self._frozen_geometry = geometry
        return self.solve_for_conv(self.wavelength, *self._conv_matrices(geometry))

    def _conv_matrices(self, geometry=None):
        """Convolution matrices for the current structure: ``(epx, epy, epz_i, epxy, epyx)``."""
        fto_x, fto_y = self.fto[0], self.fto[1]
        epxy_conv_all = epyx_conv_all = None

        if self._modeling_type_assigned == 0:                      # raster
            # The two axes are dispatched independently. Splitting on the factorization
            # first is not arbitrary: Li's rules return three matrices and NVF returns five
            # (it couples x and y, so epxy / epyx stop being zero).
            sampled = self._fourier_coefficients == FourierCoefficients.SAMPLED
            normal_field = None if geometry is None else geometry.normal_field
            normal_field_source = None if geometry is None else geometry.normal_field_source
            compression = None if geometry is None else geometry.compression

            if self._factorization == Factorization.LI:
                if sampled:
                    epx_conv_all, epy_conv_all, epz_conv_i_all = \
                        _fac.to_conv_mat_raster_discrete(
                            self.ucell, fto_x, fto_y)
                else:
                    epx_conv_all, epy_conv_all, epz_conv_i_all = \
                        _fac.to_conv_mat_raster_continuous(
                            self.ucell, fto_x, fto_y, compression=compression)
            else:
                if sampled:
                    (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
                     epz_conv_i_all) = _fac.to_conv_mat_raster_nvf(
                        self.ucell, fto_x, fto_y,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source)
                else:
                    (epx_conv_all, epy_conv_all, epxy_conv_all, epyx_conv_all,
                     epz_conv_i_all) = _fac.to_conv_mat_raster_continuous_nvf(
                        self.ucell, fto_x, fto_y,
                        normal_field=normal_field,
                        normal_field_source=normal_field_source,
                        compression=compression)

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
                type_complex=xp.complex128, xp=xp,
                boundary_plans=None if geometry is None else geometry.boundary_plans)

        else:
            raise ValueError("Check 'modeling_type' in 'conv_solve'.")

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
