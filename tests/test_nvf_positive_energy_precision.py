"""Independent positive physical integrals reject response/field corruption."""
import numpy as np
import pytest

from wavebit._core.conforming import pair_positive_precision as precision
from wavebit._core.conforming import pair_positive_energy_precision as energy
from wavebit._core.conforming import cartesian_pair_corner_galerkin as corner
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell

pytestmark = pytest.mark.research


@pytest.mark.parametrize('target,metric', (
    ('value', 'energy_residual'),
    ('tangent', 'energy_derivative_residual'),
    ('potential', 'energy_residual'),
    ('retained_field', 'retained_residual'),
))
def test_complete_positive_physical_integrals_reject_corruption(monkeypatch, target, metric):
    response, verify, check = precision.response, energy.verify, corner._check_response
    observed = []

    def corrupt_response(*args):
        state, change, condition = response(*args)
        if target in ('value', 'tangent'):
            index = 4 if target == 'value' else 5
            state = list(state)
            hi, lo = state[index]
            state[index] = hi+.01*np.eye(len(hi)), lo
        return state, change, condition

    def corrupt_field(base, whole, divided, fields, potential, field, change):
        if target == 'potential':
            potential = tuple(v.copy() for v in potential)
            potential[0][0, 0] += .05
        if target == 'retained_field':
            field = tuple(v.copy() for v in field)
            field[0][:] += .05*np.eye(len(field[0]))
        return verify(base, whole, divided, fields, potential, field, change)

    def record(value, tangent, metadata):
        observed.append(metadata)
        return check(value, tangent, metadata)

    monkeypatch.setattr(precision, 'response', corrupt_response)
    monkeypatch.setattr(energy, 'verify', corrupt_field)
    monkeypatch.setattr(corner, '_check_response', record)
    cell = CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0), 0,
        reference=(2., 2., 2.), degree=5, quadrature=48, angular_quadrature=12)
    with pytest.raises(ValueError, match='original/energy/physical guards'):
        cell.closure(gap=.02, basis=np.eye(2), widths=(.3, .3), height=.4,
                     offset=(-.5, -.5), epsilon=(4., 9.), background=1.,
                     gap_tangent=.7)
    assert observed[0][metric] > 1e-10
    if target in ('potential', 'retained_field'):
        assert observed[0]['weak_residual'] < 1e-10
        assert observed[0]['rounded_retained_defect'] < 1e-10
        assert observed[0]['flux_residual'] < 1e-10
