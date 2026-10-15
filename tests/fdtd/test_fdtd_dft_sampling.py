"""Subsampled DFT quadrature, timestamps and backend sensitivity."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest


def carrier(backend, amplitude=1., stride=4):
    steps, dt = 23, .02
    e = np.zeros((3, 4, 2, 2), complex)
    e[0] = 1
    h = np.zeros_like(e)
    h[1] = .7
    times = np.arange(steps+1)*dt
    # Force a spatially constant exact exp(-i omega t) electric field.
    signal = np.exp(-2j*np.pi*times)
    pulse = (signal[:-1]-signal[1:])/dt
    return _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.),
                            dt=dt, steps=steps, electric=e*amplitude, magnetic=h,
                            source=e*amplitude, pulse=pulse, wavelengths=(1.,), monitor_z=(1,),
                            backend=backend, dft_subsample=stride, convergence_window=6,
                            record_field=5, record_components='ex')


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_exact_carrier_and_staggered_magnetic_quadrature(backend):
    if backend == 'jax':
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    if backend == 'torch':
        pytest.importorskip('torch')
    result = carrier(backend)
    assert result['dft_samples'] == 6
    # The demodulated electric signal is exactly constant; weights sum to 23.
    assert complex(result['electric_spectrum'][0, 0, 0, 0, 0]) == pytest.approx(.46, abs=1e-14)
    c = result['convergence']
    assert complex(c['electric_spectrum_before_window'][0, 0, 0, 0, 0]) == pytest.approx(.34, abs=1e-14)
    n = np.arange(0, 23, 4)
    expected_h = np.sum(np.minimum(4, 23-n)*.02*.7*np.exp(2j*np.pi*(n+.5)*.02))
    assert complex(result['magnetic_spectrum'][0, 0, 1, 0, 0]) == pytest.approx(expected_h, abs=1e-14)
    exact = carrier(backend, stride=1)
    host = (lambda v: v.detach().numpy()) if backend == 'torch' else np.asarray
    np.testing.assert_array_equal(host(result['E']), host(exact['E']))
    np.testing.assert_array_equal(host(result['fields']['ex']['values']), host(exact['fields']['ex']['values']))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_subsampled_dft_gradient(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        value = jax.jit(jax.grad(lambda a: carrier('jax', a)['electric_spectrum'].real.sum()))(1.)
    else:
        torch = pytest.importorskip('torch')
        amplitude = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        # NumPy multiplication would detach Torch scalars: use a tensor pattern.
        e = torch.zeros((3, 4, 2, 2), dtype=torch.complex128)
        e[0] = 1
        pulse = (np.exp(-2j*np.pi*np.arange(23)*.02)-np.exp(-2j*np.pi*np.arange(1, 24)*.02))/.02
        result = _legacy_wavebit_fdtd_grid(torch.ones((4, 2, 2), dtype=torch.float64),
                                  spacing=(1., 1., 1.), dt=.02, steps=23,
                                  electric=e*amplitude, source=e*amplitude, pulse=pulse,
                                  wavelengths=(1.,), monitor_z=(1,), backend='torch',
                                  dft_subsample=4, convergence_window=6)
        value, = torch.autograd.grad(result['electric_spectrum'].real.sum(), amplitude)
    assert float(value) == pytest.approx(4*.46, abs=1e-12)


@pytest.mark.parametrize('stride', [True, 0, -1, 24, 1.5])
def test_invalid_stride(stride):
    with pytest.raises(ValueError, match='dft_subsample'):
        carrier('numpy', stride=stride)


def test_sampling_nyquist_is_rejected():
    with pytest.raises(ValueError, match='Nyquist'):
        _legacy_wavebit_fdtd_grid(np.ones((4, 2, 2)), spacing=(1., 1., 1.), dt=.1, steps=20,
                          wavelengths=(1.,), monitor_z=(1,), dft_subsample=5)


def test_gaussian_scene_band_agrees_with_every_step_and_prefix_run():
    kw = dict(structure=np.array([[[2.25]]]), thickness=[.3], top=1., bot=1.,
              period=.2, wavelength=[.95, 1.05], backend='numpy', resolution=20, n_periods=30,
              truncation_order=0, tetm='te', return_diagnostics=True, grading=True,
              incidence_side='bottom')
    exact, _ = _legacy_wavebit_fdtd(**kw)
    sampled, d = _legacy_wavebit_fdtd(**kw, dft_subsample=3, convergence_window=77)
    for a, b in zip(sampled, exact):
        np.testing.assert_allclose(a.incident.t_s, b.incident.t_s, atol=1e-5, rtol=0)
        np.testing.assert_allclose(a.incident.r_s, b.incident.r_s, atol=1e-5, rtol=0)
    assert d['dft_samples'] == (d['steps']-1)//3+1
    short_kw = dict(kw, n_periods=(d['steps']-77-.25)*d['dt'])
    prefix, _ = _legacy_wavebit_fdtd(**short_kw, dft_subsample=3)
    expected = np.stack([a.incident.t_s-b.incident.t_s for a, b in zip(sampled, prefix)])
    np.testing.assert_allclose(d['convergence']['amplitude_change']['t_s'], expected, atol=2e-14, rtol=0)
