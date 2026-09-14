"""Independent ellipse Fourier derivatives and graded optical refinement budgets."""
import numpy as np
import pytest

from studies.fdtd_ellipse_gradients import density, reference, response
from studies.fdtd_circle_gradients import density as disk_density
from studies.fdtd_rotation_gradients import finite_difference


def test_ellipse_fourier_mean_and_three_analytic_parameter_derivatives():
    p = np.array([.37, .34, .2])
    value, jac = density(p, clip=False, return_jacobian=True)
    np.testing.assert_allclose(value.mean(), np.pi*p[1]*p[2]/(4*.6**2), atol=2e-15)
    np.testing.assert_allclose(jac.mean(axis=(0, 1)), [0., np.pi*p[2]/(4*.6**2), np.pi*p[1]/(4*.6**2)], atol=2e-14)
    for h in (1e-6, 5e-7):
        np.testing.assert_allclose(jac, finite_difference(lambda p: density(p, clip=False), p, h), atol=3e-7, rtol=3e-6)


def test_fourier_circle_limit_has_zero_rotation_and_the_shared_diameter_derivative():
    a, j = density([.63, .3, .3], clip=False, return_jacobian=True)
    circle, diameter = disk_density([.3], clip=False, return_jacobian=True)
    np.testing.assert_allclose(a, circle, atol=2e-14)
    np.testing.assert_allclose(j[..., 0], 0., atol=2e-14)
    np.testing.assert_allclose(j[..., 1]+j[..., 2], diameter[..., 0], atol=3e-13)


@pytest.fixture(scope='module')
def target():
    p = np.array([.37, .34, .2])
    return p, reference(p), finite_difference(reference, p, 1e-5)


def test_independent_reference_grid_order_and_difference_controls(target):
    p, value, jac = target
    for f in (lambda p: reference(p, grid=512), lambda p: reference(p, order=10)):
        assert np.max(abs(f(p)-value)) < 1e-5
        assert np.max(abs(finite_difference(f, p, 1e-5)-jac)) < .002
    assert np.max(abs(finite_difference(reference, p, 5e-6)-jac)) < 2e-6


@pytest.mark.slow
def test_eight_complex_channels_and_three_shape_derivatives_refine(target, record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p, value, jac = target
    errors = []
    for resolution in (20, 30):
        f = lambda p: response(p, resolution)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        error = np.r_[np.max(abs(a-value)), np.max(abs(j-jac), axis=0)]
        errors.append(error)
        record_property(f'r{resolution}_amplitude_angle_width_height_errors', error.tolist())
        for col in (0, 4):
            assert abs(sum(abs(a[col:col+2])**2)+1.1*sum(abs(a[col+2:col+4])**2)-1) < .005
    assert np.all(errors[1] < errors[0])
    assert np.all(errors[1] < [.01, .006, .05, .05])
