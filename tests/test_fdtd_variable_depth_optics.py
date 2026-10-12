"""Independent moving-exit film optics and total-depth derivative accuracy."""
import numpy as np
import pytest

from studies.fdtd_layer_gradients import airy, BACKGROUND
from studies.fdtd_variable_depth import reference, measure, complex_amplitudes


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('smoothing', [0., .025])
def test_moving_exit_reference_derivative_and_reciprocity(side, smoothing):
    d = .173
    value, derivative = reference(d, smoothing, side=side)
    np.testing.assert_allclose(value[-4:-2]+value[-2:], 1., atol=4e-13)
    for h in (1e-5, 5e-6):
        finite = (reference(d+h, smoothing, side=side)[0]-reference(d-h, smoothing, side=side)[0])/(2*h)
        np.testing.assert_allclose(derivative, finite, atol=3e-8, rtol=0)
    fine = reference(d, smoothing, side=side, slices=1024)
    np.testing.assert_allclose(value, fine[0], atol=2e-7, rtol=0)
    np.testing.assert_allclose(derivative, fine[1], atol=3e-6, rtol=0)
    if not smoothing:
        background = BACKGROUND if side == 'top' else (*BACKGROUND[2:], *BACKGROUND[:2])
        np.testing.assert_allclose(value, airy(d, depth=d, background=background), atol=3e-14)


def test_total_depth_scattering_refines_to_the_independent_graded_film(record_property):
    pytest.importorskip('jax')
    result = measure()
    for side in ('top', 'bottom'):
        rows = [r for r in result['rows'] if r['side'] == side]
        for key in ('amplitude_error', 'derivative_error'):
            assert rows[-1][key] < rows[0][key]
        assert rows[-1]['amplitude_error'] < .002
        assert rows[-1]['derivative_error'] < .02
        assert rows[-1]['energy_error'] < .001
        value, jac = map(np.asarray, (rows[-1]['values'], rows[-1]['derivatives']))
        assert np.isfinite(complex_amplitudes(value)).all() and np.isfinite(jac).all()
    record_property('variable_depth_refinement', result)
