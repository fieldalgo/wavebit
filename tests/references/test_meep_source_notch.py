"""Matched uniform ADE records and mode suppression against public-API Meep."""
from pathlib import Path

import numpy as np
import pytest

import wavebit
from wavebit._core.scene_fdtd import _notched_pulse


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('name', ['gaussian', 'notched'])
def test_native_source_records_match_meep_at_the_same_current_times(backend, name):
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    elif backend == 2:
        pytest.importorskip('torch')
    with np.load(Path(__file__).with_name('golden_meep_source_notch.npz'), allow_pickle=False) as data:
        dt, frequency, width = (float(data[k]) for k in ('dt', 'frequency', 'width'))
        times, expected = data[name+'_t'], data[name+'_signal']
        steps = int(np.ceil(times[-1]/dt))
        # Meep's electric current is sampled at the half time step. Native
        # pulse[n] drives that same update; native E records are at (n+1)*dt.
        current_time = (np.arange(steps)+.5)*dt
        pulse = np.where(current_time <= 10*width,
                         np.exp(-((current_time-5*width)/width)**2)
                         *np.exp(-2j*np.pi*frequency*current_time), 0j)
        if name == 'notched':
            pulse, _ = _notched_pulse(pulse, dt, np.array([.4, .6]), frequency,
                                      np.finfo(float).eps, frequency)
        source = np.zeros((3, 2, 2, 2)); source[1] = 1
        out = wavebit.fdtd_grid(np.full((2, 2, 2), float(data['epsilon'])), spacing=(.1,)*3,
                                dt=dt, steps=steps, backend=backend, source=source, pulse=pulse,
                                poles=[(float(data['omega0']), 0., float(data['strength']))],
                                record_field=1, record_components=['ey'], record_region=(slice(0, 1),)*3)
        field = out['fields']['ey']['values'][:, 0, 0, 0]
        field = field.detach().numpy() if backend == 2 else np.asarray(field)
        indices = np.rint(times/dt).astype(int)-1
        np.testing.assert_allclose(field[indices], expected, rtol=0, atol=1e-10)


def test_meep_notch_suppresses_the_independent_undamped_material_pole():
    with np.load(Path(__file__).with_name('golden_meep_source_notch.npz'), allow_pickle=False) as data:
        dt = float(data['dt'])
        rate = float(data['omega0'])**2+float(data['strength'])/float(data['epsilon'])
        expected = np.arccos(1-rate*dt*dt/2)/(2*np.pi*dt)
        mode = wavebit.fit_fdtd_modes(data['gaussian_t'], data['gaussian_signal'], modes=1)
        np.testing.assert_allclose(mode['frequency'], [expected], atol=1e-9)
        np.testing.assert_allclose(mode['decay_rate'], 0, atol=1e-9)
        assert np.linalg.norm(data['gaussian_signal']) > .1
        assert np.linalg.norm(data['notched_signal']) < 1e-8*np.linalg.norm(data['gaussian_signal'])
