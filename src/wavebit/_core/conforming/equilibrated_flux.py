"""Weighted projection onto admissible closed-cell displacement fields.

This explicit NumPy diagnostic uses the existing dense conforming spaces.
It minimizes the error estimator's mismatch without changing the primal
electric field or selecting/truncating any trial functions.
"""
import numpy as np

from .cartesian_pair_galerkin import _relative, _solve
from .periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell


def least_squares_flux(cell, settings, primal):
    """Return retained flux and coefficients of Phi*y + R grad(v).

    The real geometric potential space is the original closed quotient C.
    Retained Fourier coefficients are free. With alpha = Re(phase*epsilon),
    this is the full positive weighted projection of epsilon*E onto that
    space in the norm integral(|epsilon*E-D|**2/alpha).
    """
    epsilon = primal['info']['material']
    alpha = np.real(cell._phase*epsilon)
    # The metric materials are positive real, independently of the physical
    # material sector. Reusing an angle beyond pi/2 would reject this metric.
    if type(cell) is PeriodicRectangleGalerkinGapCell:
        metric = type(cell)(cell.order, degree=cell.degree,
                            refinement=cell.plan.refinement, coercivity_angle=0.)
    else:
        metric = type(cell)(cell.plan, cell.order, degree=cell.degree, coercivity_angle=0.)

    def assemble(obj, materials):
        return obj._assemble(**dict(settings, epsilon=materials[1:], background=materials[0]),
                             gap=0., gap_tangent=0., epsilon_tangent=(0., 0.),
                             background_tangent=0.)

    weighted = assemble(metric, 1/alpha)
    forced = assemble(cell, epsilon/alpha)
    nc, size = primal['info']['nc'], primal['info']['size']
    identity, zero = np.eye(size), np.zeros((size, size))
    rotation = np.block([[zero, -identity], [identity, zero]])
    direct = np.kron(np.eye(2), weighted[10])
    mixed = rotation@weighted[4][:, :nc]
    matrix = np.block([[direct, mixed], [mixed.conj().T, weighted[0][:nc, :nc]]])
    cross = np.zeros((nc, nc), complex)
    for tag, local in zip(cell.data['tags'], primal['info']['local']):
        _, ci, _, _, _, c, _, _, _, determinant, _, thin, *_ = local
        if thin:
            continue
        rotated = np.stack([-c[..., 1, :], c[..., 0, :]], axis=-2)
        cross[np.ix_(ci, ci)] += epsilon[tag]/alpha[tag]*cell._inner(rotated, c)/determinant
    x, u = primal['field'], primal['potential'][:nc]
    rhs = np.vstack([np.kron(np.eye(2), forced[10])@x+forced[4][:, :nc]@u,
                     forced[8][:nc]@rotation.T@x+cross@u])
    hermitian = _relative(matrix-matrix.conj().T, matrix)
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all() or hermitian > 1e-10:
        raise ValueError('least-squares flux metric failed finite/Hermitian guards')
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('least-squares flux metric is not positive definite') from error
    solution = _solve(matrix, rhs)
    residual = _relative(matrix@solution-rhs, matrix@solution, rhs)
    y, v = solution[:2*size], solution[2*size:]
    potential = np.zeros_like(primal['potential'])
    potential[:nc] = v
    retained = y+rotation@primal['moments'][:, :nc]@v
    if (not all(np.isfinite(value).all() for value in (solution, retained))
            or not np.isfinite(residual) or residual > 1e-10):
        raise ValueError('least-squares flux failed its weighted stationarity guard')
    return retained, y, potential, dict(flux_stationarity_residual=residual,
                                       flux_metric_hermitian_defect=hermitian)
