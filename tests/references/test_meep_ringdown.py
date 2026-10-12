"""Fit frozen public-API Meep records against independently derived time poles."""
from pathlib import Path

import numpy as np
import pytest
import wavebit


@pytest.mark.parametrize('name', ['lorentz', 'cavity'])
def test_meep_ringdown_recovers_discrete_material_and_maxwell_modes(name):
    with np.load(Path(__file__).with_name('golden_meep_ringdown.npz'), allow_pickle=False) as data:
        dt, eps, gamma = (float(data[k]) for k in (name+'_dt', 'epsilon', 'gamma'))
        hold = (1-gamma*dt/2)/(1+gamma*dt/2)
        if name == 'lorentz':
            rate = float(data['omega0'])**2+float(data['strength'])/eps
            gain = (2-rate*dt**2)/(1+gamma*dt/2)
            modes = 3
        else:
            dx, length = float(data['cavity_dx']), float(data['cavity_length'])
            spatial = 2*np.sin(np.pi*dx/length)/dx
            gain = 1+hold-dt**2*spatial**2/(eps*(1+gamma*dt/2))
            modes = 2
        roots = np.linalg.eigvals([[gain, -hold], [1., 0.]])
        if name == 'lorentz':
            roots = np.r_[roots, 1.+0j]
        frequency = -np.angle(roots)/(2*np.pi*dt)
        order = np.argsort(frequency)
        fitted = wavebit.fit_fdtd_modes(data[name+'_t'], data[name+'_signal'], modes=modes)
        np.testing.assert_allclose(fitted['frequency'], frequency[order], atol=1e-9)
        np.testing.assert_allclose(fitted['decay_rate'], -np.log(abs(roots[order]))/dt, atol=1e-9)
        assert fitted['relative_residual'] < 1e-9
