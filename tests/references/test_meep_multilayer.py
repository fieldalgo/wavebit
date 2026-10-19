"""Six coating variables checked against Meep and independent continuum optics."""

from wavebit import Units
from wavebit.fdtd import periodic as fdtd
from pathlib import Path

import numpy as np
import pytest


def test_constrained_coating_gradients_against_meep_and_continuum(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from examples.fdtd.fdtd_multilayer_optimization import design
    from studies.fdtd.multilayer_optimization.fdtd_multilayer_optimization import decode_numpy, graded_stack
    def load(name):
        with np.load(Path(__file__).with_name(name)) as data:
            return {key: data[key] for key in data.files}
    ref = load('golden_meep_multilayer.npz')
    refined = load('golden_meep_multilayer_refined.npz')
    assert ref['subpixel_tol'] == 1e-8 and refined['subpixel_tol'] == 1e-9
    assert ref['dimensions'] == ref['transverse_cells']+2 == 3
    assert ref['resolution'] == 80 and ref['layer_smoothing'] == 1/80
    for key in ref.keys()-{'subpixel_tol', 'amplitudes', 'jacobians'}:
        np.testing.assert_array_equal(ref[key], refined[key])
    values, derivatives = ref['amplitudes'], ref['jacobians']
    np.testing.assert_allclose(values, refined['amplitudes'], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivatives, refined['jacobians'], atol=5e-7, rtol=0)
    np.testing.assert_allclose(values[0], values[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(derivatives[:, 0], derivatives[:, 1], atol=1.5e-4, rtol=0)
    p, band = ref['parameters'], ref['wavelengths']
    eps, lengths = decode_numpy(p)
    np.testing.assert_allclose(eps, ref['epsilon'], atol=1e-15)
    np.testing.assert_allclose(lengths, ref['thickness'], atol=1e-15)

    def response(p, resolution):
        units = Units(length='um')
        problem = fdtd.Problem(units=units, stack=design(p),
            excitation=fdtd.Excitation(wavelength=band, sources=(fdtd.PlaneWave(polarization=(1., 0.)),)),
            pulse=fdtd.Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*3.*units.time_unit_seconds)))
        result = fdtd.run(problem, backend='jax', numerics=fdtd.Numerics(
            spacing=1/resolution, pml_cells=resolution,
            nominal_layer_thickness=(.2,)*4, layer_smoothing=1/resolution, align_layers=True,
            duration_seconds=65*.5*(float(np.min(band))+float(np.max(band)))*units.time_unit_seconds,
            orders=0, extraction_method='directional'))
        return jnp.stack((result.reflection.amplitude_s.reshape(-1),
                          result.transmission.amplitude_s.reshape(-1)))
    def evaluate(resolution):
        f = lambda p: response(p, resolution)
        return tuple(map(np.asarray, jax.jit(lambda p: (f(p), jax.jacfwd(f)(p)))(p)))
    actual, derivative = evaluate(80)
    np.testing.assert_allclose(actual, values[-1], atol=.0065, rtol=0)
    np.testing.assert_allclose(derivative, derivatives[-1, -1], atol=.0035, rtol=0)
    assert np.min(np.max(abs(derivatives[-1, -1]), axis=(0, 1))) > .04

    def continuum(p, width):
        eps, lengths = decode_numpy(p)
        target = graded_stack(eps, lengths, band, width)
        return np.stack((target['r'], target['t']))
    def finite(step, width):
        return np.stack([(continuum(p+v, width)-continuum(p-v, width))/(2*step)
                         for v in np.eye(6)*step], axis=-1)
    graded, graded_derivative = continuum(p, 1/80), finite(1e-5, 1/80)
    np.testing.assert_allclose(graded_derivative, finite(5e-6, 1/80), atol=2e-8, rtol=0)
    # Both finite-grid methods are checked independently; mutual agreement alone
    # would also accept the same convention or physical-profile mistake twice.
    for a, d in ((actual, derivative), (values[-1], derivatives[-1, -1])):
        np.testing.assert_allclose(a, graded, atol=.012, rtol=0)
        np.testing.assert_allclose(d, graded_derivative, atol=.006, rtol=0)

    sharp, sharp_derivative = continuum(p, 0.), finite(1e-5, 0.)
    target_R = abs(sharp[0])**2
    target_gradient = 2*np.real(sharp[0, :, None].conj()*sharp_derivative[0])
    coarse, coarse_derivative = evaluate(40)
    def errors(a, d):
        gradient = 2*np.real(a[0, :, None].conj()*d[0])
        return np.max(abs(abs(a[0])**2-target_R)), np.max(abs(gradient-target_gradient))
    fine_errors, coarse_errors = np.array(errors(actual, derivative)), np.array(errors(coarse, coarse_derivative))
    assert np.all(fine_errors < .3*coarse_errors)
    assert fine_errors[0] < 5e-6 and fine_errors[1] < 2e-4
    record_property('meep_amplitude_gap', float(np.max(abs(actual-values[-1]))))
    record_property('meep_derivative_gap', float(np.max(abs(derivative-derivatives[-1, -1]))))
    record_property('sharp_reflection_gap', float(fine_errors[0]))
    record_property('sharp_reflection_derivative_gap', float(fine_errors[1]))
