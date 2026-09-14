"""Constitutive error control against physical integrals and exact laminates."""
import numpy as np
import pytest

from wavebit._core.conforming.cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.periodic_rectangle_galerkin import PeriodicRectangleGalerkinGapCell
from wavebit._core.conforming.galerkin_error import (
    estimate_closed_error, refine_closed_error, _closed_state, _dual_fields, _settings,
)
from test_nvf_cartesian_pair_opening import _oracle_geometry
from test_nvf_periodic_galerkin import oracle_chart
from test_nvf_pair_quadratic import duffy, nodal_gradients


PAIR = dict(basis=np.array([[700., 35.], [0., 560.]]), widths=(170., 230.),
            height=280., offset=(21., -14.))
PERIODIC = dict(period=(700., 560.), width=310., offset=(21., -14.))


def cell(kind, degree=2, refinement=0, order=(1, 2), angle=1.53):
    if kind == 'pair':
        return CartesianPairGalerkinGapCell(CartesianPairOpeningPlan(0, refinement), order,
                                           degree=degree, coercivity_angle=angle)
    return PeriodicRectangleGalerkinGapCell(order, degree=degree, refinement=refinement,
                                            coercivity_angle=angle)


def geometry(kind, epsilon):
    return dict(PAIR if kind == 'pair' else PERIODIC,
                epsilon=epsilon if kind == 'pair' else epsilon[0], background=1.)


def physical_gram(obj, kind, settings, quadrature):
    """Integrate the actual electric/flux mismatch with independent gradients.

    Only solved coefficients are shared. Rational geometry, Vandermonde nodal
    gradients, Duffy quadrature and direct Fourier fields are independent of
    the estimator's exact polynomial and phase moments.
    """
    args = _settings(obj, settings)
    primal = _closed_state(obj, args)
    dual = _closed_state(obj, dict(args, epsilon=1/np.conj(args['epsilon']),
                                  background=1/np.conj(args['background'])))
    hd, xd, ud, _ = _dual_fields(primal, dual)
    length = np.sqrt(np.linalg.det(args['basis']))
    if kind == 'pair':
        graph = _oracle_geometry(obj.data, {k: np.asarray(args[k])/length
                                           for k in ('basis', 'widths', 'height', 'offset')})
    else:
        graph, _ = oracle_chart(obj, settings)
    points = np.asarray(graph['vertices0'], float)
    nx, ny = obj.order
    modes = np.array([(m, n) for n in range(-ny, ny+1) for m in range(-nx, nx+1)])
    waves = 2*np.pi*modes@np.linalg.inv(args['basis']/length)
    bary, weights = duffy(quadrature)
    result = np.zeros_like(hd)
    # Expand the real collapsed quotient independently from its node labels.
    collapse = np.eye(max(graph['pmap'])+1)[graph['pmap']][:, 1:]
    for triangle, thin, tag in zip(graph['triangles'], graph['thin'], graph['tags']):
        if thin:
            continue
        vertex = points[triangle[:3]]
        jac = np.column_stack([vertex[1]-vertex[0], vertex[2]-vertex[0]])
        if obj.degree == 2:
            grad = nodal_gradients(bary)@np.linalg.inv(jac)
        else:
            grad = np.broadcast_to(np.array([[-1., -1.], [1., 0.], [0., 1.]])
                                   @np.linalg.inv(jac), (len(bary), 3, 2))
        c = collapse[graph['periodic'][triangle]]
        p = np.einsum('qnd,ni->qdi', grad, c@primal['potential'][:c.shape[1]])
        d = np.einsum('qnd,ni->qdi', grad, c@ud[:c.shape[1]])
        phase = np.exp(1j*(bary@vertex)@waves.T)
        phi = np.zeros((len(bary), 2, 2*len(modes)), complex)
        phi[:, 0, :len(modes)] = phase
        phi[:, 1, len(modes):] = phase
        eps = [args['background'], *args['epsilon']][tag]
        residual = eps*(phi@primal['field']+p)-(phi@xd+np.stack([-d[:, 1], d[:, 0]], axis=1))
        result += np.einsum('q,qdi,qdj->ij', weights*np.linalg.det(jac)/np.real(obj._phase*eps),
                            residual.conj(), residual, optimize=True)
    return result


@pytest.mark.parametrize('kind', ('pair', 'periodic'))
@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('epsilon', ((4., 9.), (-10.+1j, 12.25)))
def test_residual_gram_matches_independent_physical_integrals(kind, degree, epsilon):
    obj = cell(kind, degree)
    settings = geometry(kind, epsilon)
    expected = [physical_gram(obj, kind, settings, q) for q in (24, 32)]
    scale = max(1., np.linalg.norm(expected[1]))
    np.testing.assert_allclose(expected[0]/scale, expected[1]/scale, atol=3e-11, rtol=0)
    actual = estimate_closed_error(obj, **settings)
    np.testing.assert_allclose(actual.residual_gram/scale, expected[1]/scale, atol=3e-11, rtol=0)
    assert actual.diagnostics['residual_gram_minimum_eigenvalue'] >= -1e-10*scale
    assert actual.diagnostics['original_residual'] < 1e-10


@pytest.mark.parametrize('kind', ('pair', 'periodic'))
@pytest.mark.parametrize('degree', (1, 2))
@pytest.mark.parametrize('refinement', (0, 1))
def test_real_positive_hypercircle_identity(kind, degree, refinement):
    obj = cell(kind, degree, refinement, angle=0.)
    result = estimate_closed_error(obj, **geometry(kind, (4., 9.)))
    np.testing.assert_allclose(result.residual_gram, result.tensor-result.dual_tensor,
                               atol=3e-10, rtol=0)


def laminate(order, settings):
    modes = np.arange(-order, order+1)
    difference = modes[:, None]-modes
    fraction = settings['width']/settings['period'][0]
    center = .5+settings['offset'][0]/settings['period'][0]
    indicator = fraction*np.sinc(difference*fraction)*np.exp(-2j*np.pi*difference*center)
    identity = np.eye(len(modes))
    eps, bg = settings['epsilon'], settings['background']
    direct = bg*identity+(eps-bg)*indicator
    inverse = identity/bg+(1/eps-1/bg)*indicator
    zero = np.zeros_like(direct)
    return np.block([[np.linalg.solve(inverse, identity), zero], [zero, direct]])


@pytest.mark.parametrize('epsilon', (4., -10.+1j))
@pytest.mark.parametrize('refinement', (0, 1, 2))
def test_operator_bound_contains_exact_lamination_error(epsilon, refinement):
    settings = dict(PERIODIC, epsilon=epsilon, background=1.)
    result = estimate_closed_error(cell('periodic', refinement=refinement, order=(2, 0)), **settings)
    error = np.linalg.norm(result.tensor-laminate(2, settings), 2)
    assert error <= result.operator_error_bound+3e-10
    assert np.isfinite(result.operator_error_bound)


@pytest.mark.parametrize('kind', ('pair', 'periodic'))
@pytest.mark.parametrize('epsilon', (3.+.4j, -10.+1j))
def test_uniform_material_has_zero_constitutive_error(kind, epsilon):
    obj = cell(kind)
    result = estimate_closed_error(obj, **dict(geometry(kind, (epsilon, epsilon)), background=epsilon))
    np.testing.assert_allclose(result.tensor, epsilon*np.eye(len(result.tensor)), atol=3e-10, rtol=0)
    assert result.operator_error_bound < 3e-10


def test_unit_and_translation_invariance():
    obj = cell('pair', order=(1, 1))
    settings = geometry('pair', (12.25, -10.+1j))
    first = estimate_closed_error(obj, **settings)
    scaled = dict(settings, **{k: np.asarray(v)*.001 for k, v in PAIR.items()})
    second = estimate_closed_error(obj, **scaled)
    np.testing.assert_allclose(first.residual_gram, second.residual_gram, atol=3e-8, rtol=0)
    shift = np.array([13., -17.])
    moved = estimate_closed_error(obj, **dict(settings, offset=settings['offset']+shift))
    np.testing.assert_allclose(first.operator_error_bound, moved.operator_error_bound, atol=3e-9, rtol=0)


def test_refinement_runs_until_requested_constitutive_bound_or_reports_exhaustion():
    settings = dict(PERIODIC, epsilon=4., background=1.)
    obj = cell('periodic', order=(1, 0), angle=0.)
    # A tolerance between independently evaluated levels must select level 1.
    first = estimate_closed_error(obj, **settings)
    second = estimate_closed_error(cell('periodic', refinement=1, order=(1, 0), angle=0.), **settings)
    assert second.operator_error_bound < first.operator_error_bound
    tolerance = (first.operator_error_bound+second.operator_error_bound)/2
    result = refine_closed_error(obj, absolute_tolerance=tolerance, max_refinement=2, **settings)
    assert result.converged and result.refinement == 1 and len(result.history) == 2
    assert result.estimate.operator_error_bound <= tolerance
    exhausted = refine_closed_error(obj, absolute_tolerance=1e-14, max_refinement=1, **settings)
    assert not exhausted.converged and exhausted.reason == 'refinement_limit'
    assert len(exhausted.history) == 2


def test_relative_budget_uses_the_exact_norm_lower_bound():
    settings = dict(PERIODIC, epsilon=4., background=1.)
    obj = cell('periodic', order=(1, 0), angle=0.)
    estimate = estimate_closed_error(obj, **settings)
    loose = estimate.operator_error_bound/np.linalg.norm(estimate.tensor, 2)
    strict = estimate.relative_error_bound
    assert strict > loose
    # This budget would pass using the approximate norm. It must fail with
    # the guaranteed lower bound on the exact tensor norm.
    result = refine_closed_error(obj, relative_tolerance=(loose+strict)/2,
                                 max_refinement=0, **settings)
    assert not result.converged
    assert estimate.operator_error_bound > result.history[0]['budget']


@pytest.mark.parametrize('shift', (0, 1, 2, 3))
def test_all_frozen_pair_plans_preserve_the_real_energy_identity(shift):
    obj = CartesianPairGalerkinGapCell(CartesianPairOpeningPlan(shift, 0), (0, 1), degree=2)
    settings = geometry('pair', (4., 9.))
    turn = np.linalg.matrix_power(np.array([[0., -1.], [1., 0.]]), shift)
    settings['basis'] = settings['basis']@turn
    result = estimate_closed_error(obj, **settings)
    np.testing.assert_allclose(result.residual_gram, result.tensor-result.dual_tensor,
                               atol=3e-10, rtol=0)


@pytest.mark.parametrize('flux', ('surrogate', 'least_squares'))
def test_corrupt_original_response_cannot_return_an_error_bound(monkeypatch, flux):
    from wavebit._core.conforming import galerkin_error
    response = galerkin_error._response
    def corrupt(*args):
        output = list(response(*args))
        output[4] = output[4]+np.eye(len(output[4]))
        return output
    monkeypatch.setattr(galerkin_error, '_response', corrupt)
    with pytest.raises(ValueError, match='original/physical/reciprocity'):
        estimate_closed_error(cell('pair'), flux=flux, **geometry('pair', (4., 9.)))


@pytest.mark.parametrize('options', ({'absolute_tolerance': -1}, {'relative_tolerance': np.nan},
                                   {'max_refinement': 4}, {'max_refinement': True}))
def test_invalid_refinement_requests_fail_before_solving(options):
    with pytest.raises(ValueError):
        refine_closed_error(cell('pair'), **options, **geometry('pair', (4., 9.)))


def test_error_estimate_refuses_unsupported_geometry_and_materials():
    with pytest.raises(TypeError):
        estimate_closed_error(object(), **geometry('pair', (4., 9.)))
    with pytest.raises((TypeError, ValueError)):
        estimate_closed_error(cell('pair'), gap=.1, **geometry('pair', (4., 9.)))
    with pytest.raises(ValueError):
        estimate_closed_error(cell('pair'), **geometry('pair', (4., -10.-1j)))
