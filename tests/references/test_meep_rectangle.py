"""Independent rectangle scattering, derivative uncertainty and refinement."""
from pathlib import Path

import numpy as np
import pytest

from studies.fdtd_rectangle_gradients import rcwa_reference


def reference(fine=False):
    name = 'golden_meep_rectangle_fine.npz' if fine else 'golden_meep_rectangle.npz'
    with np.load(Path(__file__).with_name(name)) as data:
        return dict(data)


def test_meep_reference_uncertainty_and_independent_fourier_optics(record_property):
    coarse, fine = reference(), reference(True)
    p = coarse['parameters']
    target = rcwa_reference(p)[:2]
    translation = -2j*np.pi*np.array([-1, 0, 1])/1.3*target
    width = (rcwa_reference(p+[1e-5, 0])-rcwa_reference(p-[1e-5, 0]))[:2]/2e-5
    for data, error_bound in ((coarse, .004), (fine, .0006)):
        np.testing.assert_array_equal(data['parameters'], p)
        a, d = data['amplitudes'], data['jacobians']
        np.testing.assert_allclose(a[0], a[1], atol=4e-6, rtol=0)
        np.testing.assert_allclose(d[0], d[1], atol=2e-5, rtol=0)
        np.testing.assert_allclose(d[:, 0], d[:, 1], atol=5e-8, rtol=0)
        np.testing.assert_allclose(a[-1], target, atol=error_bound, rtol=0)
        np.testing.assert_allclose(d[-1, -1, ..., 0], width, atol=.03, rtol=0)
        np.testing.assert_allclose(d[-1, -1, ..., 1], translation, atol=.1, rtol=0)
    # Sharp finite-grid derivatives carry substantially more position error
    # than scattering values. Do not equate duration/step stability with accuracy.
    coarse_error = np.max(abs(coarse['amplitudes'][-1]-target))
    fine_error = np.max(abs(fine['amplitudes'][-1]-target))
    assert fine_error < .2*coarse_error
    for index, expected in ((0, width), (1, translation)):
        assert np.max(abs(fine['jacobians'][-1, -1, ..., index]-expected)) < np.max(abs(coarse['jacobians'][-1, -1, ..., index]-expected))
    record_property('meep_fine_amplitude_error', float(fine_error))


def test_wavebit_shape_derivatives_refine_against_independent_phase_and_rcwa(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd_rectangle_gradients import response
    data = reference()
    p = data['parameters']
    target = rcwa_reference(p)
    translation = -2j*np.pi*np.array([-1, 0, 1])/1.3*target
    def finite(step, order=32):
        return (rcwa_reference(p+[step, 0], order)-rcwa_reference(p-[step, 0], order))/(2*step)
    width = finite(1e-5)
    np.testing.assert_allclose(width, finite(5e-6), atol=2e-8, rtol=0)
    np.testing.assert_allclose(width, finite(1e-5, 48), atol=4e-4, rtol=0)
    def evaluate(resolution, cells):
        f = lambda p: response(p, resolution, cells)
        return tuple(map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p)))
    sharp, sharp_jacobian = evaluate(40, 0.)
    np.testing.assert_allclose(sharp[:2], data['amplitudes'][-1], atol=.003, rtol=0)
    np.testing.assert_allclose(sharp_jacobian[:2], data['jacobians'][-1, -1], atol=.095, rtol=0)
    errors = []
    for resolution in (20, 40):
        a, d = evaluate(resolution, 1.)
        errors.append(np.array([np.max(abs(a-target)), np.max(abs(d[..., 0]-width)),
                                np.max(abs(d[..., 1]-translation))]))
    assert np.all(errors[1] < .5*errors[0])
    assert np.all(errors[1] < [.0025, .007, .015])
    assert errors[1][1] < .2*np.max(abs(sharp_jacobian[..., 0]-width))
    assert errors[1][2] < .2*np.max(abs(sharp_jacobian[..., 1]-translation))
    record_property('sharp_meep_derivative_gap', float(np.max(abs(sharp_jacobian[:2]-data['jacobians'][-1, -1]))))
    record_property('smoothed_width_error', float(errors[1][1]))
    record_property('smoothed_position_error', float(errors[1][2]))


def test_averaged_rectangle_derivatives_against_independent_meep(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd_rectangle_gradients import response
    def load(name):
        with np.load(Path(__file__).with_name(name)) as data:
            return dict(data)
    base = load('golden_meep_rectangle_smoothed.npz')
    refined = load('golden_meep_rectangle_smoothed_refined.npz')
    assert base['shape_smoothing'] == .025 and base['resolution'] == 40
    assert base['subpixel_tol'] == 1e-8 and refined['subpixel_tol'] == 1e-9
    for key in base.keys()-{'subpixel_tol', 'amplitudes', 'jacobians'}:
        np.testing.assert_array_equal(base[key], refined[key])
    np.testing.assert_allclose(base['amplitudes'], refined['amplitudes'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(base['jacobians'], refined['jacobians'], atol=6e-7, rtol=0)
    for data in (base, refined):
        a, d = data['amplitudes'], data['jacobians']
        np.testing.assert_allclose(a[0], a[1], atol=4e-6, rtol=0)
        np.testing.assert_allclose(d[0], d[1], atol=2e-5, rtol=0)
        np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-5, rtol=0)
    f = lambda p: response(p, 40, 1.)[:2]
    a, d = map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(base['parameters']))
    np.testing.assert_allclose(a, base['amplitudes'][-1], atol=.002, rtol=0)
    np.testing.assert_allclose(d, base['jacobians'][-1, -1], atol=.003, rtol=0)
    record_property('averaged_meep_amplitude_gap', float(np.max(abs(a-base['amplitudes'][-1]))))
    record_property('averaged_meep_derivative_gap', float(np.max(abs(d-base['jacobians'][-1, -1]))))
