"""Independent Bessel-series controls and FDTD disk refinement budgets."""
import numpy as np
import pytest

from studies.cross_solver.circle_gradients.fdtd_circle_gradients import density, reference, response
from studies.cross_solver.rotation_gradients.fdtd_rotation_gradients import finite_difference


def test_fourier_profile_has_exact_disk_mean_and_diameter_derivative():
    p = np.array([.3])
    value, jac = density(p, clip=False, return_jacobian=True)
    np.testing.assert_allclose(value.mean(), np.pi*.15**2/.6**2, atol=2e-15)
    np.testing.assert_allclose(jac.mean(), np.pi*.15/.6**2, atol=2e-14)
    for step in (1e-6, 5e-7):
        np.testing.assert_allclose(jac, finite_difference(lambda p: density(p, clip=False), p, step),
                                   atol=2e-7, rtol=2e-6)


@pytest.fixture(scope='module')
def target():
    p = np.array([.3])
    return p, reference(p), finite_difference(reference, p, 1e-5)


def test_graded_reference_fourier_grid_order_and_difference_controls(target):
    p, value, jac = target
    for f in (lambda p: reference(p, grid=512), lambda p: reference(p, order=10)):
        assert np.max(abs(f(p)-value)) < 1e-5
        assert np.max(abs(finite_difference(f, p, 1e-5)-jac)) < 1e-3
    assert np.max(abs(finite_difference(reference, p, 5e-6)-jac)) < 1e-6


@pytest.mark.slow
def test_complex_scattering_and_diameter_gradient_refine_toward_fourier_optics(target, record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p, value, jac = target
    errors = []
    for resolution in (20, 30):
        f = lambda p: response(p, resolution)
        a, j = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p))
        error = np.array([np.max(abs(a-value)), np.max(abs(j-jac))])
        errors.append(error)
        record_property(f'r{resolution}_amplitude_diameter_errors', error.tolist())
        for column in (0, 4):
            assert abs(sum(abs(a[column:column+2])**2)+1.1*sum(abs(a[column+2:column+4])**2)-1) < .005
    assert np.all(errors[1] < errors[0])
    assert errors[1][0] < .01 and errors[1][1] < .04
