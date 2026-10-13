"""Whole-state contact elimination with unchanged physical acceptance checks."""
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming import cartesian_pair_seam_galerkin as module
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from tests.rcwa.test_nvf_pair_seam_galerkin import GEOMETRY

pytestmark = pytest.mark.research


def cell(degree=3, strategy='auto'):
    return module.CartesianPairSeamGalerkinGapCell(CartesianPairOpeningPlan(0), (1, 1),
        degree=degree, quadrature=24, coercivity_angle=1.53, contact_solver=strategy)


@pytest.mark.parametrize('degree', (3, 4))
@pytest.mark.parametrize('epsilon', ((4., 9.), (12.25, -10+1j)))
@pytest.mark.parametrize('direction', ({}, dict(gap_tangent=.7,
    epsilon_tangent=(.3+.2j, -.4+.1j), background_tangent=.2+.3j)))
def test_retained_whole_state_matches_independent_physical_qr(degree, epsilon, direction):
    settings = dict(GEOMETRY, gap=0., epsilon=epsilon, **direction)
    actual = cell(degree).closure(**settings)
    expected = cell(degree, 'projected').closure(**settings)
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-10, rtol=0)
    # Use the established dimensionless gap tolerance, and the material
    # tolerance for simultaneous finite material directions.
    scale = 1. if direction else np.sqrt(np.linalg.det(GEOMETRY['basis']))
    np.testing.assert_allclose(scale*np.asarray(actual[1]), scale*np.asarray(expected[1]), atol=3e-9, rtol=0)
    assert actual[2]['contact_solver_attempts'] == ('retained',)
    assert expected[2]['contact_solver_attempts'] == ('physical_projected',)
    assert max(v for k, v in actual[2].items() if k.endswith('residual') and v is not None) < 1e-10


def test_accepted_retained_state_does_not_run_physical_qr(monkeypatch):
    def forbidden(*args):
        pytest.fail('accepted retained equations unnecessarily ran physical QR')
    monkeypatch.setattr(module, 'projected_contact_response', forbidden)
    result = cell().closure(gap=0., epsilon=(12.25, -10+1j), **GEOMETRY)
    assert result[2]['closed_elimination'] == 'retained'


@pytest.mark.parametrize('failure', ('guard', 'singular'))
def test_failed_retained_candidate_is_replaced_as_one_complete_state(monkeypatch, failure):
    settings = dict(GEOMETRY, gap=0., epsilon=(12.25, -10+1j))
    expected = cell(strategy='projected').closure(**settings)
    original = module.coupled_contact
    def candidate(*args, **kwargs):
        if kwargs.get('response_solver') is not None:
            return original(*args, **kwargs)
        if failure == 'singular':
            raise np.linalg.LinAlgError('injected singular retained system')
        value, tangent, meta = original(*args, **kwargs)
        # A rejected candidate must contribute no field or derivative block
        # to the independent whole-state QR result.
        meta['energy_derivative_residual'] = 1e-4
        return np.asarray(value)+.3, np.asarray(tangent)+.2, meta
    monkeypatch.setattr(module, 'coupled_contact', candidate)
    actual = cell().closure(**settings)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2]['contact_solver_attempts'] == ('retained', 'physical_projected')
    assert actual[2]['retained_contact_rejection']


def test_failure_of_both_complete_solvers_is_reported(monkeypatch):
    original = module.coupled_contact
    def corrupt(*args, **kwargs):
        value, tangent, metadata = original(*args, **kwargs)
        metadata['energy_derivative_residual'] = 1e-4
        return value, tangent, metadata
    monkeypatch.setattr(module, 'coupled_contact', corrupt)
    with pytest.raises(ValueError, match='original/energy/physical guards'):
        cell().closure(gap=0., epsilon=(12.25, -10+1j), **GEOMETRY)


@pytest.mark.parametrize('key', ('reciprocity', 'derivative_reciprocity',
                               'minimum_rotated_storage', 'minimum_dissipation'))
def test_nonfinite_physical_diagnostic_cannot_pass_a_candidate(key):
    value, tangent, metadata = cell().closure(gap=0., epsilon=(4., 9.), **GEOMETRY)
    metadata[key] = np.nan
    with pytest.raises(ValueError, match=f'{key}=nan'):
        module._check_response(value, tangent, metadata)


def test_guard_failure_reports_each_failed_metric_and_its_bound():
    value, tangent, metadata = cell().closure(gap=0., epsilon=(4., 9.), **GEOMETRY)
    metadata['original_residual'] = 5.19e-9
    metadata['energy_derivative_residual'] = 3e-10
    with pytest.raises(ValueError, match='original/energy/physical guards') as error:
        module._check_response(value, tangent, metadata)
    message = str(error.value)
    assert 'original_residual=5.19e-09 (required <= 1e-10)' in message
    assert 'energy_derivative_residual=3e-10 (required <= 1e-10)' in message
    assert 'minimum_rotated_storage' not in message


def test_guard_diagnostics_preserve_existing_threshold_boundaries():
    # These are acceptance-policy boundaries, independent of any numerical
    # integration or solver representation.
    metadata = dict(original_residual=1e-10, weak_derivative_residual=None,
                    reciprocity=1e-9, derivative_reciprocity=1e-9,
                    minimum_rotated_storage=np.nextafter(0., 1.),
                    minimum_dissipation=-1e-9)
    module._check_response(np.eye(2), np.zeros((2, 2)), metadata)
    metadata['minimum_rotated_storage'] = 0.
    with pytest.raises(ValueError, match='minimum_rotated_storage=0 .*required > 0'):
        module._check_response(np.eye(2), np.zeros((2, 2)), metadata)


@pytest.mark.parametrize('target', ('value', 'tangent'))
def test_guard_diagnostics_identify_nonfinite_response_blocks(target):
    arrays = dict(value=np.eye(2), tangent=np.zeros((2, 2)))
    arrays[target][0, 0] = np.nan
    metadata = dict(energy_residual=0., reciprocity=0., derivative_reciprocity=0.,
                    minimum_rotated_storage=1., minimum_dissipation=0.)
    with pytest.raises(ValueError, match=f'{target} contains nonfinite entries'):
        module._check_response(**arrays, metadata=metadata)


@pytest.mark.parametrize('residuals', ({}, {'weak_derivative_residual': None}))
def test_guard_cannot_accept_a_state_without_computed_residuals(residuals):
    metadata = dict(reciprocity=0., derivative_reciprocity=0.,
                    minimum_rotated_storage=1., minimum_dissipation=0., **residuals)
    with pytest.raises(ValueError, match='no computed residuals'):
        module._check_response(np.eye(2), np.zeros((2, 2)), metadata)


@pytest.mark.parametrize('strategy', ('invalid', None, 1, ['auto']))
def test_invalid_solver_selection_is_refused(strategy):
    with pytest.raises(ValueError, match='contact_solver'):
        cell(strategy=strategy)


def test_quadratic_family_refuses_unsupported_forced_projection():
    with pytest.raises(ValueError, match='requires degree 3 or 4'):
        cell(degree=2, strategy='projected')
