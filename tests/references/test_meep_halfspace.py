"""Frozen public-API Meep differences versus live half-space autodiff."""
from pathlib import Path

import numpy as np
import pytest

import wavebit


@pytest.mark.slow
def test_halfspace_amplitudes_and_jacobian_against_meep(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    with np.load(Path(__file__).with_name('golden_meep_halfspace.npz')) as data:
        ref = {key: data[key] for key in data.files}
    values, derivatives = ref['amplitudes'], ref['jacobians']
    # Settling and perturbation controls protect the external derivative oracle.
    np.testing.assert_allclose(values[0], values[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(derivatives[0], derivatives[1], atol=2e-7, rtol=0)
    np.testing.assert_allclose(derivatives[:, 0], derivatives[:, 1], atol=2e-6, rtol=0)
    def objective(p):
        results = wavebit.fdtd(structure=np.full((1, 1, 1), float(ref['epsilon'])),
                               mu=np.full((1, 1, 1), float(ref['mu'])), thickness=[float(ref['thickness'])],
                               top=wavebit.Material(p[0], p[1]), bot=wavebit.Material(p[2], p[3]),
                               period=.05, wavelength=ref['wavelengths'], backend='jax',
                               mesh_indices=(1.1, 1.4), resolution=int(ref['resolution']),
                               n_pml=int(ref['resolution']), n_periods=65, pulse_width=3.,
                               align_layers=True, diffraction_method='directional', truncation_order=0)
        amplitudes = jnp.stack([jnp.stack([getattr(r.s_inc, key).reshape(()) for r in results])
                                for key in ('r_s', 't_s')])
        return jnp.stack((amplitudes.real, amplitudes.imag))
    value, jacobian = jax.jit(lambda p: (objective(p), jax.jacrev(objective)(p)))(ref['parameters'])
    actual = np.asarray(value[0]+1j*value[1])
    gradient = np.asarray(jacobian[0]+1j*jacobian[1])
    np.testing.assert_allclose(actual, values[-1], atol=.003, rtol=0)
    np.testing.assert_allclose(gradient, derivatives[-1, -1], atol=.005, rtol=0)
    assert np.min(np.max(abs(derivatives[-1, -1]), axis=(0, 1))) > .1
    record_property('maximum_meep_amplitude_difference', float(np.max(abs(actual-values[-1]))))
    record_property('maximum_meep_derivative_difference', float(np.max(abs(gradient-derivatives[-1, -1]))))
