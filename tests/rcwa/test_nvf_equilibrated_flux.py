"""Admissible weighted flux projection against independent physical integrals."""
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming import galerkin_error as ge, equilibrated_flux
from wavebit.rcwa._engine.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from tests.rcwa.test_nvf_galerkin_error import cell, geometry, laminate, PERIODIC
from tests.rcwa.test_nvf_cartesian_pair_opening import _oracle_geometry
from tests.rcwa.test_nvf_periodic_galerkin import oracle_chart
from tests.rcwa.test_nvf_pair_quadratic import duffy, nodal_gradients
from tests.rcwa.test_nvf_polynomial_galerkin import cardinal_gradient, indices, bernstein

pytestmark = pytest.mark.research


def sampled_projection(obj, kind, settings, primal, quadrature):
    """Solve directly from sampled physical columns, with no candidate moments.

    The original electric coefficients are shared; rational triangle geometry,
    cardinal gradients, direct Fourier samples, the weighted QR solve and its
    residual integration are independent of the candidate flux assembly.
    """
    args = ge._settings(obj, settings)
    length = np.sqrt(np.linalg.det(args['basis']))
    if kind == 'pair':
        graph = _oracle_geometry(obj.data, {k: np.asarray(args[k])/length
                                           for k in ('basis', 'widths', 'height', 'offset')})
    else:
        graph, _ = oracle_chart(obj, settings)
    points = np.asarray(graph['vertices0'], float)
    collapse = np.eye(max(graph['pmap'])+1)[graph['pmap']][:, 1:]
    nx, ny = obj.order
    modes = np.array([(m, n) for n in range(-ny, ny+1) for m in range(-nx, nx+1)])
    waves = 2*np.pi*modes@np.linalg.inv(args['basis']/length)
    bary, weights = duffy(quadrature)
    columns, targets, projections = [], [], []
    for tri, thin, tag in zip(graph['triangles'], graph['thin'], graph['tags']):
        if thin:
            continue
        vertex = points[tri[:3]]
        jac = np.column_stack([vertex[1]-vertex[0], vertex[2]-vertex[0]])
        c = collapse[graph['periodic'][tri]]
        if type(obj) is CartesianPairPolynomialGalerkinGapCell:
            grad = cardinal_gradient(bary, obj.degree)@np.linalg.inv(jac)
            c = bernstein(np.asarray(indices(obj.degree))/obj.degree, obj.degree)@c
        elif obj.degree == 2:
            grad = nodal_gradients(bary)@np.linalg.inv(jac)
        else:
            grad = np.broadcast_to(np.array([[-1., -1.], [1., 0.], [0., 1.]])
                                   @np.linalg.inv(jac), (len(bary), 3, 2))
        gradients = np.einsum('qnd,nc->qdc', grad, c)
        phase = np.exp(1j*(bary@vertex)@waves.T)
        phi = np.zeros((len(bary), 2, 2*len(modes)), complex)
        phi[:, 0, :len(modes)] = phase
        phi[:, 1, len(modes):] = phase
        rotated = np.stack([-gradients[:, 1], gradients[:, 0]], axis=1)
        trial = np.concatenate([phi, rotated], axis=2)
        eps = [args['background'], *args['epsilon']][tag]
        electric = phi@primal['field']+gradients@primal['potential'][:c.shape[1]]
        area = weights*np.linalg.det(jac)
        root = np.sqrt(area/np.real(obj._phase*eps))[:, None, None]
        columns.append((root*trial).reshape(-1, trial.shape[-1]))
        targets.append((root*eps*electric).reshape(-1, electric.shape[-1]))
        projections.append(np.einsum('q,qdi,qdj->ij', area, phi.conj(), trial, optimize=True))
    design, target = np.vstack(columns), np.vstack(targets)
    q, r = np.linalg.qr(design, mode='reduced')
    coefficients = np.linalg.solve(r, q.conj().T@target)
    residual = target-design@coefficients
    return residual.conj().T@residual, sum(projections)@coefficients


@pytest.mark.parametrize('kind,degree', (('pair', 1), ('pair', 2), ('pair', 3),
                                       ('pair', 4), ('periodic', 1), ('periodic', 2)))
@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10.+1j)))
def test_weighted_flux_matches_independent_sampled_projection(kind, degree, epsilon):
    obj = (CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0), (1, 0),
            degree=degree, coercivity_angle=1.53) if degree > 2 else cell(kind, degree, order=(1, 0)))
    settings = geometry(kind, epsilon)
    primal = ge._closed_state(obj, ge._settings(obj, settings))
    expected = [sampled_projection(obj, kind, settings, primal, q) for q in (20, 28)]
    actual = ge.estimate_closed_error(obj, flux='least_squares', **settings)
    for index, value in enumerate((actual.residual_gram, actual.dual_tensor)):
        scale = max(1., np.linalg.norm(expected[1][index]))
        np.testing.assert_allclose(expected[0][index]/scale, expected[1][index]/scale, atol=3e-11, rtol=0)
        np.testing.assert_allclose(value/scale, expected[1][index]/scale, atol=3e-11, rtol=0)
    previous = ge.estimate_closed_error(obj, **settings)
    np.testing.assert_array_equal(actual.tensor, previous.tensor)
    difference = previous.residual_gram-actual.residual_gram
    allowance = sum(x.diagnostics['residual_gram_roundoff_allowance'] for x in (previous, actual))
    assert np.linalg.eigvalsh((difference+difference.conj().T)/2)[0] >= -allowance
    assert actual.field_error_bound <= previous.field_error_bound+np.sqrt(allowance)
    assert actual.diagnostics['flux_stationarity_residual'] < 1e-10


@pytest.mark.parametrize('kind', ('pair', 'periodic'))
def test_negative_real_sector_uniform_medium_uses_positive_metric(kind):
    obj = cell(kind, angle=2.8)
    eps = -10.+1j
    result = ge.estimate_closed_error(obj, flux='least_squares',
                                     **dict(geometry(kind, (eps, eps)), background=eps))
    np.testing.assert_allclose(result.tensor, eps*np.eye(len(result.tensor)), atol=3e-10, rtol=0)
    assert result.operator_error_bound < 3e-10
    assert obj.coercivity_angle == 2.8


@pytest.mark.parametrize('epsilon', (4., -10.+1j))
def test_least_squares_bound_contains_exact_laminate_error(epsilon):
    settings = dict(PERIODIC, epsilon=epsilon, background=1.)
    result = ge.estimate_closed_error(cell('periodic', refinement=1, order=(2, 0)),
                                     flux='least_squares', **settings)
    assert np.linalg.norm(result.tensor-laminate(2, settings), 2) <= result.operator_error_bound+3e-10


@pytest.mark.parametrize('kind', ('pair', 'periodic'))
def test_real_energy_identity_is_preserved_by_projection(kind):
    obj = cell(kind, refinement=1, order=(0, 1), angle=0.)
    result = ge.estimate_closed_error(obj, flux='least_squares', **geometry(kind, (4., 9.)))
    np.testing.assert_allclose(result.residual_gram, result.tensor-result.dual_tensor, atol=3e-10, rtol=0)


def test_projection_preserves_units_and_translation():
    obj = cell('pair', order=(0, 1))
    settings = geometry('pair', (12.25, -10.+1j))
    first = ge.estimate_closed_error(obj, flux='least_squares', **settings)
    scaled = dict(settings, **{k: np.asarray(settings[k])*.001 for k in ('basis', 'widths', 'height', 'offset')})
    second = ge.estimate_closed_error(obj, flux='least_squares', **scaled)
    np.testing.assert_allclose(first.residual_gram, second.residual_gram, atol=3e-8, rtol=0)
    moved = ge.estimate_closed_error(obj, flux='least_squares',
                                    **dict(settings, offset=settings['offset']+np.array([13., -17.])))
    np.testing.assert_allclose(first.operator_error_bound, moved.operator_error_bound, atol=3e-9, rtol=0)


def test_refinement_keeps_projection_and_reports_unmet_budget():
    obj = cell('periodic', order=(1, 0), angle=0.)
    settings = dict(PERIODIC, epsilon=4., background=1.)
    first = ge.estimate_closed_error(obj, flux='least_squares', **settings)
    second = ge.estimate_closed_error(cell('periodic', refinement=1, order=(1, 0), angle=0.),
                                     flux='least_squares', **settings)
    tolerance = (first.operator_error_bound+second.operator_error_bound)/2
    result = ge.refine_closed_error(obj, flux='least_squares', absolute_tolerance=tolerance,
                                   max_refinement=1, **settings)
    assert result.converged and result.refinement == 1
    assert result.estimate.diagnostics['flux_method'] == 'least_squares'
    exhausted = ge.refine_closed_error(obj, flux='least_squares', absolute_tolerance=1e-14,
                                      max_refinement=1, **settings)
    assert not exhausted.converged and exhausted.reason == 'refinement_limit'


@pytest.mark.parametrize('flux', ('unknown', None, 1, ['least_squares']))
def test_invalid_strategy_fails_before_solving(monkeypatch, flux):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid strategy reached a solve')
    monkeypatch.setattr(ge, '_closed_state', forbidden)
    with pytest.raises(ValueError, match='flux must be'):
        ge.estimate_closed_error(cell('pair'), flux=flux, **geometry('pair', (4., 9.)))
    with pytest.raises(ValueError, match='flux must be'):
        ge.refine_closed_error(cell('pair'), flux=flux, absolute_tolerance=1., **geometry('pair', (4., 9.)))


def test_corrupted_weighted_solution_cannot_return_bound(monkeypatch):
    solve = equilibrated_flux._solve
    monkeypatch.setattr(equilibrated_flux, '_solve', lambda a, b: solve(a, b)+.01)
    with pytest.raises(ValueError, match='weighted stationarity'):
        ge.estimate_closed_error(cell('pair'), flux='least_squares', **geometry('pair', (4., 9.)))
