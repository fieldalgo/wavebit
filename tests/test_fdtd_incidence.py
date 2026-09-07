"""Unsupported incidence must be refused before polarization dispatch loses information."""
import numpy as np
import pytest

import wavebit
from wavebit._core import fdtd as core


@pytest.mark.parametrize('crossed', [False, True])
@pytest.mark.parametrize('incidence', [dict(tetm='te'), dict(psi=.4, phi=0.),
                                      dict(psi=.4, phi=.3)])
@pytest.mark.parametrize('override, message', [
    (dict(chi=.2), 'chi'),
    (dict(polarization='rcp'), 'polarization'),
    (dict(theta=.3+.1j), 'theta.*complex'),
    (dict(phi=.3+.1j), 'phi.*complex'),
    (dict(theta=np.nan), 'theta.*finite'),
    (dict(phi=np.inf), 'phi.*finite'),
])
def test_incidence_refused_before_time_loop(monkeypatch, crossed, incidence, override, message):
    def must_not_run(*args, **kwargs):
        pytest.fail('invalid incidence reached the time loop')
    monkeypatch.setattr(core, '_time_loop', must_not_run)
    kw = dict(structure=np.ones((1, 2 if crossed else 1, 2)), thickness=[100.],
              top=1., bot=1., period=[300., 400.] if crossed else [300.],
              wavelength=900., theta=.3, resolution=.015, n_periods=60)
    kw.update(incidence)
    kw.update(override)
    with pytest.raises(ValueError, match=message):
        wavebit.fdtd(**kw)


@pytest.mark.parametrize('psi', [.4+.1j, np.nan, np.inf])
def test_psi_must_be_a_finite_real_angle(psi):
    with pytest.raises(ValueError, match='psi.*(complex|finite)'):
        wavebit.fdtd(structure=np.ones((1, 1, 2)), thickness=[100.], top=1., bot=1.,
                     period=[300.], wavelength=900., theta=.3, phi=.4, psi=psi,
                     resolution=.015, n_periods=60)
