"""Primal/dual error control for the closed constrained constitutive cell.

The real H1 electric potential and rotated real H1 flux potential give
independent conforming fields. Their constitutive mismatch bounds the electric
cell error by coercivity. This NumPy diagnostic uses the existing dense finite
families; it does not replace their tensors, differentiate an adaptive choice,
or certify Maxwell/Fourier-order/contact-derivative convergence.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np

from .cartesian_pair_galerkin import CartesianPairGalerkinGapCell, _response, _solve, _relative
from .cartesian_pair_opening import CartesianPairOpeningPlan, real_array
from .periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from .cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from .material_sector import minimum_rotated_storage
from .equilibrated_flux import least_squares_flux


@dataclass(frozen=True)
class ClosedErrorEstimate:
    """Floating evaluation of an exact-arithmetic constitutive error bound.

    ``operator_error_bound`` bounds the spectral norm of the transverse tensor
    error at fixed retained modes. ``relative_error_bound`` uses a lower bound
    on the exact tensor norm and is infinite when that lower bound is zero.
    ``field_error_bound`` uses the rotated-storage energy norm. No bound on an
    optical observable or on the opening derivative follows from this object.
    ``dual_tensor`` is the retained admissible flux; for ``least_squares``
    it comes from the weighted projection, not a separate constitutive solve.
    """

    tensor: np.ndarray
    dual_tensor: np.ndarray
    residual_gram: np.ndarray
    field_error_bound: float
    operator_error_bound: float
    relative_error_bound: float
    exact_tensor_norm_lower_bound: float
    diagnostics: dict


@dataclass(frozen=True)
class ClosedRefinementResult:
    """A bounded refinement run, including explicit nonconvergence at its cap."""

    estimate: ClosedErrorEstimate
    refinement: int
    converged: bool
    reason: str
    history: tuple


def _pair_settings(*, basis, widths, height, epsilon, background=1., offset=(0., 0.)):
    return dict(basis=np.asarray(basis), widths=np.asarray(widths), height=height,
                epsilon=np.asarray(epsilon), background=background, offset=np.asarray(offset))


def _periodic_settings(*, period=(700., 700.), width=350., epsilon=12.25,
                       background=1., offset=(0., 0.)):
    period = real_array(period, (2,), 'period')
    width = real_array(width, (), 'width').item()
    if np.any(period <= 0):
        raise ValueError('period must be positive')
    return _pair_settings(basis=np.diag(period), widths=(width/2, width/2), height=period[1],
                          epsilon=(epsilon, epsilon), background=background, offset=offset)


def _settings(cell, geometry):
    # Reject unknown subclasses too: their geometry and physical spaces may
    # differ from the exact rectangle indicators used in the estimator.
    if type(cell) in (CartesianPairGalerkinGapCell, CartesianPairPolynomialGalerkinGapCell):
        return _pair_settings(**geometry)
    if type(cell) is PeriodicRectangleGalerkinGapCell:
        return _periodic_settings(**geometry)
    raise TypeError('closed error control requires an explicit pair or periodic rectangle Galerkin cell')


def _closed_state(cell, geometry):
    assembled = cell._assemble(**geometry, gap=0., gap_tangent=0.,
                               epsilon_tangent=(0., 0.), background_tangent=0.)
    k, dk, q, dq, r, dr, qt, dqt, source, dsource, e, de, info = assembled
    if not all(np.isfinite(v).all() for v in assembled[:-1]):
        raise ValueError('nonfinite closed error-control assembly')
    direct = np.kron(np.eye(2), e)
    form = cell._elimination(len(k), 2*len(e))
    u, du, x, dx, h, dh, condition = _response(
        k, dk, q, dq, r, dr, qt, dqt, source, dsource, direct, np.zeros_like(direct), form)
    original, _ = cell._original_residual(x, dx, h, dh, u, du, info)
    residuals = dict(
        weak_residual=_relative(k@u+source@x-qt@h, k@u, source@x, qt@h),
        retained_residual=_relative(x+q@u-np.eye(len(h)), np.eye(len(h))),
        flux_residual=_relative(direct@x+r@u-h, direct@x, r@u, h),
        original_residual=original,
    )
    reverse = info['reverse']
    reciprocal = _relative(h-h[reverse][:, reverse].T, h)
    if (not all(np.isfinite(v).all() for v in (u, x, h))
            or max(residuals.values()) > 1e-10 or reciprocal > 1e-9
            or minimum_rotated_storage(h, cell._phase) <= 0
            or np.linalg.eigvalsh((h-h.conj().T)/2j)[0] < -1e-9):
        raise ValueError('closed error-control state failed original/physical/reciprocity guards')
    return dict(potential=u, field=x, tensor=h, info=info, moments=q,
                diagnostics=dict(residuals, reciprocity_defect=reciprocal,
                                 solve_condition=condition, elimination=form))


def _dual_fields(primal, surrogate):
    """Convert passive 1/conj(epsilon) to a normal-continuous flux field.

    Conjugation reverses retained Fourier indices. A real quarter-turn rotates
    a gradient into a divergence-free field; inversion sets its retained
    electric input to identity. Trial potentials remain real and geometric.
    """
    size = primal['info']['size']
    reverse = primal['info']['reverse']
    identity, zero = np.eye(size), np.zeros((size, size))
    rotation = np.block([[zero, -identity], [identity, zero]])
    inverse = rotation@surrogate['tensor'].conj()[reverse][:, reverse]@rotation.T
    h = _solve(inverse, np.eye(2*size))
    x = rotation@surrogate['field'].conj()[reverse][:, reverse]@rotation.T@h
    u = surrogate['potential'].conj()[:, reverse]@rotation.T@h
    residuals = dict(
        dual_inverse_residual=_relative(inverse@h-np.eye(2*size), np.eye(2*size)),
        dual_retained_residual=_relative(x+rotation@primal['moments']@u-h, h),
    )
    if not all(np.isfinite(v).all() for v in (h, x, u)) or max(residuals.values()) > 1e-10:
        raise ValueError('closed dual field failed its inverse/retained guards')
    return h, x, u, residuals


def _indicators(cell, geometry):
    """Exact pure-Fourier products, separate from all triangle moments."""
    nx, ny = cell.order
    modes = np.array([(m, n) for n in range(-ny, ny+1) for m in range(-nx, nx+1)])
    basis, widths, height, offset = (geometry[k] for k in ('basis', 'widths', 'height', 'offset'))
    waves = 2*np.pi*(modes[:, None]-modes)@np.linalg.inv(basis)
    center = basis.sum(axis=1)/2+offset

    def rectangle(center, width, height):
        return (width*height/np.linalg.det(basis)
                *np.sinc(waves[..., 0]*width/(2*np.pi))
                *np.sinc(waves[..., 1]*height/(2*np.pi))*np.exp(-1j*(waves@center)))

    if type(cell) is PeriodicRectangleGalerkinGapCell:
        bodies = [rectangle(center, sum(widths), basis[1, 1]), np.zeros((len(modes), len(modes)), complex)]
    else:
        bodies = [rectangle(center+[shift, 0.], width, height)
                  for shift, width in zip((-widths[1]/2, widths[0]/2), widths)]
    return [np.eye(len(modes))-sum(bodies), *bodies]


def _validate_flux(flux):
    if not isinstance(flux, str) or flux not in ('surrogate', 'least_squares'):
        raise ValueError("flux must be 'surrogate' or 'least_squares'")


def estimate_closed_error(cell, *, flux='surrogate', **geometry):
    """Estimate closed transverse cell error using a conforming flux solve.

    ``cell`` must be an explicitly supported pair, polynomial pair, or
    periodic rectangle Galerkin cell. Geometry/material keywords follow
    its ``closure`` signature, omitting gap and all tangents. The original
    coercivity and physical guards apply at every supported degree.
    ``flux='surrogate'`` preserves the reciprocal-material solve.
    ``flux='least_squares'`` minimizes the weighted physical mismatch over
    the same admissible flux space, leaving the electric tensor unchanged.

    The bound controls the infinite constrained H1 cell problem at the given
    Fourier set. It is evaluated in floating point, with a reported roundoff
    allowance, not certified by interval arithmetic. See the derivation in
    ``docs/nvf-galerkin-error.md``.
    """
    _validate_flux(flux)
    settings = _settings(cell, geometry)
    primal = _closed_state(cell, settings)
    if flux == 'surrogate':
        surrogate = _closed_state(cell, dict(settings, epsilon=1/np.conj(settings['epsilon']),
                                            background=1/np.conj(settings['background'])))
        hd, xd, ud, dual_residuals = _dual_fields(primal, surrogate)
        dual_residuals['surrogate'] = surrogate['diagnostics']
    else:
        hd, xd, ud, dual_residuals = least_squares_flux(cell, settings, primal)
    info = primal['info']
    epsilon = info['material']
    alpha = np.real(cell._phase*epsilon)
    moments = [np.zeros_like(hd) for _ in epsilon]
    gram = np.zeros_like(hd)
    term_scale = 0.
    for tag, local in zip(cell.data['tags'], info['local']):
        _, ci, _, _, _, c, _, _, _, determinant, _, thin, eps, _, average, _ = local
        if thin:
            continue
        electric = c@primal['potential'][ci]/determinant
        flux_gradient = c@ud[ci]/determinant
        displacement = np.stack([-flux_gradient[..., 1, :], flux_gradient[..., 0, :]], axis=-2)
        residual = eps*electric-displacement
        moments[tag] += determinant*cell._moment(residual, average).reshape(hd.shape)
        term = determinant*cell._inner(residual.conj(), residual)/alpha[tag]
        gram += term
        term_scale += np.linalg.norm(term)
    weight = np.zeros_like(hd)
    for eps, storage, indicator, moment in zip(epsilon, alpha, _indicators(cell, settings), moments):
        direct = np.kron(np.eye(2), indicator)
        free = eps*primal['field']-xd
        pure, mixed = free.conj().T@direct@free/storage, free.conj().T@moment/storage
        gram += pure+mixed+mixed.conj().T
        term_scale += np.linalg.norm(pure)+2*np.linalg.norm(mixed)
        weight += abs(eps)**2/storage*direct
    # Keep the raw Gram as evidence. Hermitian parts are used only for the
    # eigenvalue calculation, after measuring their defects. The allowance
    # exposes cancellation; it is not a relaxed physical residual guard.
    hermitian = _relative(gram-gram.conj().T, gram)
    roundoff = 64*np.finfo(float).eps*len(hd)*term_scale
    eigenvalues = np.linalg.eigvalsh((gram+gram.conj().T)/2)
    weight_eigenvalues = np.linalg.eigvalsh((weight+weight.conj().T)/2)
    if (not np.isfinite(gram).all() or not np.isfinite(weight).all()
            or hermitian > 1e-10 or eigenvalues[0] < -roundoff or weight_eigenvalues[0] <= 0):
        raise ValueError('constitutive residual Gram failed Hermitian/positivity guards')
    field_bound = float(np.sqrt(max(0., eigenvalues[-1])+roundoff))
    operator_bound = float(field_bound*np.sqrt(weight_eigenvalues[-1]))
    exact_lower = max(0., float(np.linalg.norm(primal['tensor'], 2))-operator_bound)
    relative_bound = operator_bound/exact_lower if exact_lower else float('inf')
    diagnostics = dict(primal['diagnostics'], **dual_residuals,
                       flux_method=flux, order=cell.order, degree=cell.degree,
                       refinement=cell.plan.refinement, minimum_material_storage=float(min(alpha)),
                       residual_gram_hermitian_defect=hermitian,
                       residual_gram_minimum_eigenvalue=float(eigenvalues[0]),
                       residual_gram_roundoff_allowance=float(roundoff),
                       flux_weight_norm=float(weight_eigenvalues[-1]))
    return ClosedErrorEstimate(primal['tensor'], hd, gram, field_bound, operator_bound,
                               relative_bound, exact_lower, diagnostics)


def refine_closed_error(cell, *, absolute_tolerance=0., relative_tolerance=0.,
                        max_refinement=None, flux='surrogate', **geometry):
    """Refine the closed mesh until an explicit constitutive error budget passes.

    The criterion is B <= atol + rtol * max(0, ||H_h|| - B), so its relative
    term uses a lower bound on the exact tensor norm. At least one tolerance
    must be positive. Exhausting the supported dense refinement range returns
    ``converged=False`` with all measurements; it never accepts a plateau.
    This is global mesh refinement at fixed Fourier order, not an NVF router.
    ``flux`` selects the same estimator strategy at every refinement level.
    """
    _validate_flux(flux)
    _settings(cell, geometry)
    absolute_tolerance = real_array(absolute_tolerance, (), 'absolute_tolerance').item()
    relative_tolerance = real_array(relative_tolerance, (), 'relative_tolerance').item()
    if (min(absolute_tolerance, relative_tolerance) < 0
            or max(absolute_tolerance, relative_tolerance) <= 0):
        raise ValueError('nonnegative tolerances with at least one positive value required')
    limit = (min(3, 4-(cell.degree-1).bit_length())
             if type(cell) is CartesianPairPolynomialGalerkinGapCell else 3)
    if max_refinement is None:
        max_refinement = limit
    if (isinstance(max_refinement, (bool, np.bool_)) or not isinstance(max_refinement, Integral)
            or not cell.plan.refinement <= max_refinement <= limit):
        raise ValueError('maximum refinement must lie inside the current family\'s supported range')
    history = []
    for refinement in range(cell.plan.refinement, max_refinement+1):
        if refinement != cell.plan.refinement:
            if type(cell) is PeriodicRectangleGalerkinGapCell:
                cell = PeriodicRectangleGalerkinGapCell(cell.order, degree=cell.degree,
                    refinement=refinement, coercivity_angle=cell.coercivity_angle)
            else:
                cell = type(cell)(CartesianPairOpeningPlan(cell.plan.shift, refinement),
                    cell.order, degree=cell.degree, coercivity_angle=cell.coercivity_angle)
        result = estimate_closed_error(cell, flux=flux, **geometry)
        budget = absolute_tolerance+relative_tolerance*result.exact_tensor_norm_lower_bound
        converged = result.operator_error_bound <= budget
        history.append(dict(refinement=refinement, operator_error_bound=result.operator_error_bound,
                            relative_error_bound=result.relative_error_bound, budget=budget))
        if converged:
            break
    return ClosedRefinementResult(result, refinement, converged,
                                  'tolerance' if converged else 'refinement_limit', tuple(history))
