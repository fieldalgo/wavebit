"""The modal gradient guard preserves the public numerical retry contract."""
import warnings
import numpy as np
import pytest
import wavebit

CFG = dict(period=[1000.], wavelength=500., truncation_order=(3, 0),
           top=1.21, bot=1.69, theta=0., phi=0., psi=np.pi/2, thickness=[300., 200.])
PILLAR = np.ones((1, 20)); PILLAR[:, 6:14] = 12.25


def value(backend, parameter, knob, auto_detune=True):
    if backend == 'jax':
        import jax.numpy as xp
    elif backend == 'torch':
        import torch as xp
    else:
        xp = np
    eps = parameter if knob == 'epsilon' else 2.25
    structure = xp.stack([xp.ones((1, 20), dtype=xp.float64)*eps, xp.asarray(PILLAR)])
    cfg = CFG | ({'wavelength': parameter} if knob == 'wavelength' else {})
    s = wavebit.rcwa(backend=backend, **cfg, structure=structure)
    return xp.real(s.solve(auto_detune=auto_detune).R).sum()


@pytest.mark.parametrize('detune', [True, 1e-10, 1e-8, 1e-6])
@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('knob', ['epsilon', 'wavelength'])
def test_rayleigh_parameter_gradient_reaches_the_detuned_retry(backend, knob, detune):
    point = 2.25 if knob == 'epsilon' else 500.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if backend == 'jax':
            jax = pytest.importorskip('jax')
            import jax.numpy as jnp
            wavebit.enable_x64()
            got = float(jax.grad(lambda t: value('jax', t, knob, detune))(jnp.asarray(point)))
        else:
            torch = pytest.importorskip('torch')
            t = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            value('torch', t, knob, detune).backward()
            got = float(t.grad)
        h = 1e-4
        expected = (float(value('numpy', point+h, knob))-float(value('numpy', point-h, knob)))/(2*h)
    print(backend, knob, detune, got, expected, flush=True)
    assert np.isfinite(got)
    assert got == pytest.approx(expected, rel=2e-4, abs=2e-6)
