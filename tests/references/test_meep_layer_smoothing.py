"""Graded magnetic-film derivatives against Meep's public material-function API."""
from pathlib import Path

import numpy as np
import pytest


def test_averaged_interface_derivative_against_independent_meep(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd_layer_gradients import scene_vector
    from studies.fdtd_layer_smoothing import graded_film
    def load(name):
        with np.load(Path(__file__).with_name(name)) as data:
            return {key: data[key] for key in data.files}
    ref = load('golden_meep_layer_smoothing.npz')
    refined = load('golden_meep_layer_smoothing_refined.npz')
    assert ref['subpixel_tol'] == 1e-8 and refined['subpixel_tol'] == 1e-9
    assert ref['dimensions'] == refined['dimensions'] == 3
    for key in ('parameters', 'layer_smoothing', 'transverse_cells', 'resolution', 'durations', 'deltas'):
        np.testing.assert_array_equal(ref[key], refined[key])
    for key in ('amplitudes', 'jacobians'):
        np.testing.assert_allclose(ref[key], refined[key], atol=2e-6, rtol=0)
    values, derivatives = ref['amplitudes'], ref['jacobians'][..., 0]
    np.testing.assert_allclose(values[0], values[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(derivatives[:, 0], derivatives[:, 1], atol=2e-5, rtol=0)
    options = dict(band=ref['wavelengths'], depth=float(ref['reference_depth']),
                   background=ref['background'], film=(float(ref['epsilon']), float(ref['mu'])),
                   smoothing=float(ref['layer_smoothing']))
    def objective(d):
        return scene_vector(d, int(ref['resolution']), **options)
    value, gradient = jax.jit(lambda d: (objective(d), jax.jacfwd(objective)(d)))(ref['thickness'])
    def amplitudes(vector):
        r1, r2, t1, t2 = np.asarray(vector[:8]).reshape(4, 2)
        return np.stack((r1+1j*r2, t1+1j*t2))
    actual, derivative = amplitudes(value), amplitudes(gradient)
    np.testing.assert_allclose(actual, values[-1], atol=.0015, rtol=0)
    np.testing.assert_allclose(derivative, derivatives[-1, -1], atol=.002, rtol=0)
    assert np.min(abs(derivatives[-1, -1])) > .5
    continuum = amplitudes(graded_film(float(ref['thickness']), **options)[1])
    # External finite-grid agreement and graded-continuum agreement are distinct.
    np.testing.assert_allclose(derivatives[-1, -1], continuum, atol=.006, rtol=0)
    np.testing.assert_allclose(derivative, continuum, atol=.006, rtol=0)
    record_property('meep_amplitude_gap', float(np.max(abs(actual-values[-1]))))
    record_property('meep_derivative_gap', float(np.max(abs(derivative-derivatives[-1, -1]))))
    record_property('meep_graded_derivative_gap', float(np.max(abs(derivatives[-1, -1]-continuum))))
