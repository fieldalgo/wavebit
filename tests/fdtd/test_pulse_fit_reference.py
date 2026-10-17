"""Pure preflight for the prescribed four-coordinate pulse fitting trajectories."""
import numpy as np
import pytest

from tests.fdtd import _pulse_fit_reference as fit


@pytest.mark.parametrize('seed', [0, 1])
def test_fixed_residual_trajectory_is_valid_and_nonvacuous(seed, record_property):
    result = fit.trajectory(seed)
    rows = result['rows']
    assert len(rows) == 12
    for left, right in zip(rows[:-1], rows[1:]):
        np.testing.assert_array_equal(left['after'], right['before'])
    record_property('rates', repr(fit.plan()['rates'].tolist()))
    record_property('trajectory', repr([{k: np.asarray(v).tolist() for k, v in row.items()}
                                         for row in rows]))
    record_property('final_value', result['final_value'])
    assert 0 < result['final_value'] < rows[0]['value']
    if seed == 1:
        assert any(np.any(row['projected']) for row in rows)


def test_original_unconstrained_second_update_exits_fixed_wavelength_box():
    # Preserve the failed unprojected protocol as an explicit regression. Rates,
    # bounds and target are unchanged; the optimizer now projects its proposals.
    point = np.asarray(fit.SEEDS[1])
    rates = fit.plan()['rates']
    point = point-rates*fit.oracle(point)['gradient']
    assert fit.in_bounds(point)
    raw = point-rates*fit.oracle(point)['gradient']
    assert raw[0] < fit.BOUNDS[0, 0]
    assert not fit.in_bounds(raw)


@pytest.mark.parametrize('seed', [0, 1])
def test_residual_hessian_matches_two_gradient_secants(seed, record_property):
    point = np.asarray(fit.SEEDS[seed])
    exact = fit.oracle(point)
    errors = []
    for spacing in fit.pulse.SECANT_STEPS:
        columns = []
        for i in range(4):
            direction = np.eye(4)[i]*spacing
            columns.append((fit.oracle(point+direction)['gradient']
                            - fit.oracle(point-direction)['gradient'])/(2*spacing))
        measured = np.stack(columns, axis=1)
        errors.append(float(np.max(abs((measured-exact['hessian'])/exact['hessian']))))
        np.testing.assert_allclose(measured, exact['hessian'],
                                   rtol=fit.pulse.SECANT_TOL, atol=0.)
    record_property('relative_secant_errors', repr(errors))


@pytest.mark.parametrize('seed', [0, 1])
def test_omitting_residual_curvature_changes_directional_hessian(seed, record_property):
    exact = fit.oracle(fit.SEEDS[seed])
    _, _, mutant = fit.residual_objective(exact['expected'], omit_curvature=True)
    error = float(np.max(abs(mutant@fit.DIRECTION-exact['hvp']))
                  /np.max(abs(exact['hvp'])))
    record_property('omitted_curvature_error', error)
    assert error > 1e-4


def test_residual_jet_matches_literal_polynomial_with_signed_residual():
    # A scalar nonlinear observation with a fixed, different target. Its
    # residual second derivative contributes to H even away from a minimum.
    point = np.array([.7, -.3, .2, .5])
    value = point[0]*point[1]+point[2]**2+point[3]
    first = np.array([point[1], point[0], 2*point[2], 1.])
    second = np.zeros((4, 4))
    second[0, 1] = second[1, 0] = 1.
    second[2, 2] = 2.
    residual = value-.9
    actual = fit.pulse.weighted_jet(np.array([residual]), first[:, None],
                                    second[:, :, None], 1.)
    expected = (residual**2, 2*residual*first,
                2*np.outer(first, first)+2*residual*second)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=fit.RTOL, atol=0.)
