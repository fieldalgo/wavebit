"""Constrained joint FDTD design derivatives and independent multilayer optics."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.fdtd._periodic.runtime import time_loop_backend
from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian
from studies.fdtd.multilayer_optimization.fdtd_multilayer_optimization import sharp_stack, decode_numpy, graded_stack


@pytest.mark.parametrize('width', [0., .05, .15])
def test_graded_continuum_reference_conservation_refinement_and_uniform_limit(width):
    from examples.fdtd.fdtd_multilayer_optimization import INITIAL, BAND
    eps, lengths = decode_numpy(INITIAL)
    actual = graded_stack(eps, lengths, BAND, width)
    refined = graded_stack(eps, lengths, BAND, width, slices=256)
    for key in ('r', 't', 'R', 'T'):
        np.testing.assert_allclose(actual[key], refined[key], atol=1.5e-6, rtol=0)
    np.testing.assert_allclose(actual['R']+actual['T'], 1., atol=2e-13, rtol=0)
    uniform = graded_stack(np.full(4, 4.), lengths, BAND, width)
    np.testing.assert_allclose(uniform['r'], -1/3, atol=2e-13, rtol=0)
    np.testing.assert_allclose(uniform['t'], 2/3*np.exp(4j*np.pi*.8/BAND), atol=2e-13, rtol=0)


@pytest.mark.parametrize('wavelength', [.8, 1., 1.2])
def test_independent_multilayer_matrix_matches_fresnel_recursion_and_rcwa(wavelength):
    from examples.fdtd.fdtd_multilayer_optimization import INITIAL, sharp_reflection
    eps, lengths = decode_numpy(INITIAL)
    target = sharp_stack(eps, lengths, [wavelength])
    # The independent matrix traverses top-to-bottom; public layer arrays start below.
    result = _legacy_wavebit.rcwa(structure=eps[::-1, None, None], thickness=lengths[::-1],
                          top=1., bot=4., period=.05, wavelength=wavelength,
                          truncation_order=0, backend='numpy').solve()
    for name in ('r', 't'):
        np.testing.assert_allclose(getattr(result.s_inc, name+'_s').item(), target[name][0], atol=2e-14)
    np.testing.assert_allclose(target['R']+target['T'], 1., atol=2e-14)
    np.testing.assert_allclose(target['R'], sharp_reflection(eps, lengths, np.array([wavelength])), atol=2e-14)
    # Uniform spacer merging leaves the external reference plane fixed.
    split = sharp_stack(np.r_[eps, eps[-1]], np.r_[lengths[:-1], lengths[-1]/2, lengths[-1]/2], [wavelength])
    np.testing.assert_allclose(split['r'], target['r'], atol=2e-14)
    np.testing.assert_allclose(split['t'], target['t'], atol=2e-14)


@pytest.mark.parametrize('wavelength', [.8, 1., 1.2])
def test_bottom_incidence_matches_the_reversed_independent_multilayer_matrix(wavelength):
    from examples.fdtd.fdtd_multilayer_optimization import INITIAL, sharp_reflection
    eps, lengths = decode_numpy(INITIAL)
    # Keep the physical stack and half-spaces fixed; reverse the oracle's traversal.
    target = sharp_stack(eps[::-1], lengths[::-1], [wavelength], top=4., bottom=1.)
    result = _legacy_wavebit.rcwa(structure=eps[::-1, None, None], thickness=lengths[::-1],
                          top=1., bot=4., period=.05, wavelength=wavelength,
                          incidence_side='bottom', truncation_order=0, backend='numpy').solve()
    for name in ('r', 't'):
        np.testing.assert_allclose(getattr(result.s_inc, name+'_s').item(), target[name][0], atol=2e-14)
    np.testing.assert_allclose(target['R']+target['T'], 1., atol=2e-14)
    # Lossless reciprocal stacks have the same power reflection from either side.
    np.testing.assert_allclose(target['R'], sharp_reflection(eps, lengths, np.array([wavelength])), atol=2e-14)


def test_example_places_the_spacer_at_the_bottom_and_matches_its_independent_reference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from examples.fdtd.fdtd_multilayer_optimization import INITIAL, BAND, design
    eps, lengths = decode_numpy(INITIAL)
    stack = design(jax.numpy.asarray(INITIAL))
    np.testing.assert_array_equal(stack.structure, eps[::-1, None, None])
    np.testing.assert_allclose(stack.thickness, lengths[::-1], atol=2e-16, rtol=0)
    assert np.asarray(stack.structure)[0].item() == 4.  # spacer touches the substrate
    target = sharp_stack(eps, lengths, BAND)
    for index, wavelength in enumerate(BAND):
        result = _legacy_wavebit.rcwa(stack=stack, wavelength=wavelength,
                              truncation_order=0, backend='jax').solve()
        for name in ('r', 't'):
            np.testing.assert_allclose(getattr(result.s_inc, name+'_s').item(),
                                       target[name][index], atol=2e-14)


def joint_objective(p, backend, *, diagnostics=False):
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    weights = xp.concat((p[:3], array([0.])))
    lengths = _legacy_wavebit.layer_thicknesses(weights, .8, minimum=.04)
    epsilon = xp.concat((p[3:], array([4.])))
    scene = _legacy_wavebit.Stack.from_structure(epsilon[:, None, None], lengths, top=1., bot=4., period=.05)
    light = _legacy_wavebit.Incidence(wavelength=[.95, 1.05], theta=.2, phi=.3,
                              polarization=(1., .6j), incidence_side='bottom')
    result, out = _legacy_wavebit.fdtd(scene, light, backend=backend, resolution=10, n_pml=6,
                               mesh_thickness=[.2]*4, layer_smoothing=.06, align_layers=True,
                               n_periods=35, pulse_width=1.5, truncation_order=0,
                               diffraction_method='directional', return_diagnostics=True)
    value = sum((r.incident.R+.1*xp.real(r.incident.r_s)-.025*xp.real(r.incident.t_p)).sum() for r in result)
    return (value, out) if diagnostics else value.reshape(1)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_six_joint_design_derivatives_match_two_finite_difference_steps(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    from examples.fdtd.fdtd_multilayer_optimization import INITIAL
    p = INITIAL.copy()
    value, gradient = value_and_jacobian(lambda p: joint_objective(p, backend), p, backend)
    expected, mesh = joint_objective(p, 'numpy', diagnostics=True)
    np.testing.assert_allclose(value, expected, atol=2e-11)
    for step in (2e-5, 1e-5):
        finite = np.stack([(joint_objective(p+v, 'numpy')-joint_objective(p-v, 'numpy'))/(2*step)
                           for v in step*np.eye(len(p))], axis=-1)
        np.testing.assert_allclose(gradient, finite, atol=3e-8, rtol=3e-5)
    assert np.min(abs(gradient)) > 1e-4
    _, other = joint_objective(p+np.array([.2, -.1, .3, -.1, .2, -.2]), 'numpy', diagnostics=True)
    for key in ('z_nodes', 'z_spacing', 'dt', 'steps', 'source_z', 'pml_bounds', 'mesh_thickness'):
        np.testing.assert_array_equal(mesh[key], other[key])
    assert not np.array_equal(mesh['layer_planes'], other['layer_planes'])
