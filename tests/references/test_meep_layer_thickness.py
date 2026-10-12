"""Fixed-reference-plane thickness AD versus independently perturbed Meep runs."""
from pathlib import Path

import numpy as np
import pytest


def test_magnetic_film_thickness_derivative_against_meep(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    from studies.fdtd_layer_gradients import airy, scene_vector
    with np.load(Path(__file__).with_name('golden_meep_layer_thickness.npz')) as data:
        ref = {key: data[key] for key in data.files}
    values, derivatives = ref['amplitudes'], ref['jacobians'][..., 0]
    np.testing.assert_allclose(values[0], values[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(derivatives[:, 0], derivatives[:, 1], atol=3e-6, rtol=0)
    options = dict(band=ref['wavelengths'], depth=float(ref['reference_depth']),
                   background=ref['background'], film=(float(ref['epsilon']), float(ref['mu'])))
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
    d, step = float(ref['thickness']), 1e-5
    continuum = amplitudes((airy(d+step, **options)-airy(d-step, **options))/(2*step))
    # Both discretizations share interface-position error: agreement with one
    # another is not continuum derivative accuracy at this resolution.
    assert np.max(abs(derivatives[-1, -1]-continuum)) < .06
    record_property('meep_amplitude_gap', float(np.max(abs(actual-values[-1]))))
    record_property('meep_derivative_gap', float(np.max(abs(derivative-derivatives[-1, -1]))))
    record_property('meep_continuum_derivative_gap', float(np.max(abs(derivatives[-1, -1]-continuum))))
