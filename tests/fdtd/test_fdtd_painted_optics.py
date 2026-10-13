"""Preselected Fourier-target and optical-refinement budgets for painted shapes."""
import numpy as np
import pytest

from studies.cross_solver.painted_shapes.fdtd_painted_shapes import density, reference, response
from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import finite_difference


def test_independent_inclusion_exclusion_mean_and_parameter_slopes():
    p = np.array([.37, .055, .22])
    def mean(p):
        return density(p, clip=False).mean()
    overlap = .11-(p[1]-p[2]/2)
    expected = (.8*.3*.18+1.25*p[2]*.22-.8*overlap*.16)/(.36*1.25)
    np.testing.assert_allclose(mean(p), expected, atol=3e-15)
    derivative = np.array([0., .8*.16, 1.25*.22-.8*.16/2])/(.36*1.25)
    for h in (2e-6, 1e-6):
        np.testing.assert_allclose(finite_difference(mean, p, h), derivative, atol=2e-10)
    # Adding both contrasts would leave the overlap counted twice.
    superposed = (.8*.3*.18+1.25*p[2]*.22)/(.36*1.25)
    np.testing.assert_allclose(superposed-mean(p), .8*overlap*.16/(.36*1.25), atol=3e-15)


@pytest.fixture(scope='module')
def target():
    p = np.array([.37, .055, .22])
    return p, reference(p), finite_difference(reference, p, 1e-5)


def test_independent_painted_reference_grid_order_and_step_controls(target):
    p, value, jac = target
    for f in (lambda p: reference(p, grid=512), lambda p: reference(p, order=10)):
        assert np.max(abs(f(p)-value)) < 2e-5
        assert np.max(abs(finite_difference(f, p, 1e-5)-jac)) < .003
    assert np.max(abs(finite_difference(reference, p, 5e-6)-jac)) < 1e-5


@pytest.mark.slow
def test_painted_eight_complex_channels_and_three_shape_derivatives_refine(target, record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p, value, jac = target
    errors = []
    for resolution in (20, 30):
        f = lambda p: response(p, resolution)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        error = np.r_[np.max(abs(a-value)), np.max(abs(j-jac), axis=0)]
        errors.append(error)
        record_property(f'r{resolution}_amplitude_angle_separation_width_errors', error.tolist())
        for col in (0, 4):
            assert abs(sum(abs(a[col:col+2])**2)+1.1*sum(abs(a[col+2:col+4])**2)-1) < .005
    assert np.all(errors[1] < errors[0])
    assert np.all(errors[1] < [.01, .01, .05, .05])
