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


@pytest.fixture(scope='module')
def meep_at_20():
    """Meep's own resolution-20 answer for the same painted scene, from the kept golden."""
    from tools.repository_paths import repository_path
    archive = dict(np.load(repository_path('tests/references/golden_meep_painted_shapes_coarse.npz')))
    assert archive['resolution'] == 20
    np.testing.assert_array_equal(archive['parameters'], [.37, .055, .22])
    return archive['amplitudes'][-1], archive['jacobians'][-1, -1]      # the longest duration, finest step


@pytest.mark.slow
def test_painted_eight_complex_channels_and_three_shape_derivatives_at_resolution_20(
        target, meep_at_20, record_property):
    """The native FDTD at resolution 20 against two independent answers for the same painted
    scene: the Fourier-profile RCWA reference, and Meep at the same resolution.

    The study's refinement (resolutions 20, 30, 40; ``fdtd_painted_shapes.measure``) is recorded
    in docs/validation/fdtd/fdtd-painted-shapes.md from the desktop; the resolution-30 Jacobian
    under jit asks XLA for one 22.9 GB buffer (CI run 38738075697, slow shard 1), so this case
    stays at 20 and checks it two ways. Measured (this Mac, 2026-10-16, equal to the page's
    table): against RCWA 7.95e-3 in amplitude and 1.59e-3, 7.25e-3, 9.65e-3 in the angle, position
    and width derivatives; Meep at 20 is 2.09e-2 and 3.25e-3, 4.12e-2, 9.44e-2 from the same
    reference, and the two FDTDs are 1.50e-2 and 4.8e-3, 3.4e-2, 8.7e-2 apart.

    No outer ``jax.jit``: wrapped around the value and ``jacfwd``, it took the process to
    13.6 GB peak RSS (Linux CPU, jax 0.11.0) against 3.6 GB without it, in the same time and
    within 2.3e-15 of the same numbers. At 13.3 GB on a 16.4 GB CI runner, beside a second
    xdist worker, it shut the runner down (slow shard 3/3 of run 39098232669, both attempts;
    measured by the diagnostics run 39145409231).
    """
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p, value, jac = target
    meep_a, meep_j = meep_at_20
    f = lambda p: response(p, 20)
    a, j = np.asarray(f(p)), np.asarray(jax.jacfwd(f)(p))
    error = np.r_[np.max(abs(a-value)), np.max(abs(j-jac), axis=0)]
    meep_error = np.r_[np.max(abs(meep_a-value)), np.max(abs(meep_j-jac), axis=0)]
    apart = np.r_[np.max(abs(a-meep_a)), np.max(abs(j-meep_j), axis=0)]
    record_property('r20_amplitude_angle_separation_width_errors', error.tolist())
    record_property('r20_amplitude_angle_separation_width_from_meep', apart.tolist())
    for col in (0, 4):
        assert abs(sum(abs(a[col:col+2])**2)+1.1*sum(abs(a[col+2:col+4])**2)-1) < .005
    # Within the study's budget against the RCWA reference, and at least as close to it as Meep
    # is at the same resolution.
    assert np.all(error < [.01, .01, .05, .05]), error
    assert np.all(error <= meep_error), (error, meep_error)
    # And near Meep itself: two FDTDs of one graded profile, within twice the measured gap.
    assert np.all(apart < [.03, .01, .07, .18]), apart


@pytest.mark.slow
def test_painted_jacobian_under_jit_equals_the_eager_one_at_resolution_6():
    """The resolution-20 case above runs eagerly; this keeps the outer ``jax.jit`` around the
    value and ``jacfwd`` covered, on the same painted scene at resolution 6, where it costs
    5.3 GB peak RSS rather than 13.6 (Linux CPU, jax 0.11.0; the jit's cost is mostly
    compilation, 6.2 GB at resolution 8 and 7.2 at 10). Measured: both paths agree to 1.1e-15
    in value and Jacobian, and every Jacobian column reaches 1.2e-2, so the comparison is not
    of zeros."""
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p = np.array([.37, .055, .22])
    f = lambda p: response(p, 6, duration=40)
    eager = np.asarray(f(p)), np.asarray(jax.jacfwd(f)(p))
    traced = tuple(map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p)))
    assert np.min(np.max(abs(eager[1]), axis=0)) > 1e-3
    for actual, expected in zip(traced, eager):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
