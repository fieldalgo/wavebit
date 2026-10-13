"""Ringdown inversion against exponentials and independent discrete ADE poles."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest



def trace(t, frequency, decay, amplitude):
    return np.exp(t[:, None]*(-np.asarray(decay)-2j*np.pi*np.asarray(frequency))) @ amplitude


@pytest.mark.parametrize('pencil', [None, 3, 50, 237])
def test_complex_modes_preserve_frequency_sign_decay_and_time_origin(pencil):
    t = 12.+np.arange(240)*.1
    frequency, decay = np.array([-.31, .27, .41]), np.array([.04, .03, -.02])
    amplitude = np.array([1+.2j, -.3+.6j, .2j])
    signal = trace(t-t[0], frequency, decay, amplitude)
    out = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=3, pencil=pencil)
    np.testing.assert_allclose(out['frequency'], frequency, atol=2e-11)
    np.testing.assert_allclose(out['decay_rate'], decay, atol=2e-11)
    np.testing.assert_allclose(out['amplitude'], amplitude, atol=2e-10)
    np.testing.assert_allclose(out['quality_factor'][:2], np.pi*abs(frequency[:2])/decay[:2], rtol=2e-9)
    assert np.isnan(out['quality_factor'][2])
    np.testing.assert_allclose(out['complex_frequency'], frequency-1j*decay/(2*np.pi), atol=2e-11)
    assert out['time_origin'] == t[0] and out['relative_residual'] < 1e-10
    assert out['singular_values'][0] == 1
    np.testing.assert_allclose(out['fitted'], signal, atol=3e-10)


def test_real_cosine_keeps_both_conjugate_modes_and_selected_sample_origin():
    t = np.arange(400)*.05
    signal = 2*np.exp(-.06*t)*np.cos(2*np.pi*.7*t+.3)
    saved = signal.copy()
    out = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=2, start=2.02, stop=17.04)
    start = t[t >= 2.02][0]
    amplitudes = np.exp(-.06*start)*np.exp(1j*np.array([1., -1.])*(2*np.pi*.7*start+.3))
    np.testing.assert_allclose(out['frequency'], [-.7, .7], atol=1e-12)
    np.testing.assert_allclose(out['decay_rate'], .06, atol=1e-12)
    np.testing.assert_allclose(out['amplitude'], amplitudes, atol=1e-12)
    assert out['time_origin'] == start and out['t'][-1] <= 17.04
    np.testing.assert_array_equal(signal, saved)


@pytest.mark.parametrize('scale', [1e-240, 1e240])
def test_signal_scaling_does_not_overflow_or_change_poles(scale):
    t = np.arange(300)*.1
    unit = trace(t, [-.2, .3], [.08, -.03], [1+.4j, .2-.3j])
    out = _legacy_wavebit.fit_fdtd_modes(t, scale*unit, modes=2)
    np.testing.assert_allclose(out['frequency'], [-.2, .3], atol=1e-12)
    np.testing.assert_allclose(out['decay_rate'], [.08, -.03], atol=1e-12)
    np.testing.assert_allclose(out['amplitude']/scale, [1+.4j, .2-.3j], atol=2e-12)
    np.testing.assert_allclose(out['fitted']/scale, unit, atol=2e-12)
    assert np.isfinite(out['relative_residual'])


def test_noisy_close_modes_and_underfitting_report_different_residuals():
    t = np.arange(500)*.1
    signal = trace(t, [1., 1.007], [.005, .009], [1., .6j])
    rng = np.random.default_rng(31)
    signal += 1e-6*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
    out = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=2, pencil=200)
    np.testing.assert_allclose(out['frequency'], [1., 1.007], atol=2e-5)
    np.testing.assert_allclose(out['decay_rate'], [.005, .009], atol=2e-5)
    assert out['relative_residual'] < 1e-5
    underfit = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=1, pencil=200)
    assert underfit['relative_residual'] > .01


def test_sampling_alias_is_reported_on_the_signed_nyquist_branch():
    t = np.arange(100)*.5
    out = _legacy_wavebit.fit_fdtd_modes(t, np.exp((-0.03-2j*np.pi*1.2)*t))
    np.testing.assert_allclose(out['frequency'], [-.8], atol=1e-12)
    np.testing.assert_allclose(out['decay_rate'], [.03], atol=1e-12)


def test_an_algebraic_tail_does_not_produce_a_window_independent_lifetime():
    t = np.arange(801)*.1
    signal = (t+10)**-1.5*np.exp(-2j*np.pi*.3*t)
    early = _legacy_wavebit.fit_fdtd_modes(t, signal, stop=40.)
    late = _legacy_wavebit.fit_fdtd_modes(t, signal, start=40.)
    np.testing.assert_allclose(early['frequency'], [.3], atol=1e-12)
    np.testing.assert_allclose(late['frequency'], [.3], atol=1e-12)
    assert early['decay_rate'][0] > 2*late['decay_rate'][0]
    assert early['relative_residual'] > .01


def material_record(dt, backend='numpy'):
    # Constant displacement epsilon*E+P makes the source-free Lorentz medium
    # a scalar oscillator at omega^2=omega0^2+strength/epsilon, plus a DC term.
    e = np.zeros((3, 4, 2, 2)); e[1] = 1
    result = _legacy_wavebit.fdtd_grid(np.full((4, 2, 2), 2.), spacing=(1.,)*3, dt=dt,
                                steps=round(16/dt), electric=e, poles=[(3., .1, 2.)],
                                backend=backend, record_field=1, record_components=['ey'],
                                record_region=(1, 0, 0))
    field = result['fields']['ey']
    host = lambda v: v.detach().cpu().numpy() if backend == 'torch' else np.asarray(v)
    return host(field['t']), host(field['values'])[:, 0, 0, 0]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_actual_yee_ade_records_recover_independent_discrete_poles(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    dt = .05
    t, signal = material_record(dt, backend)
    out = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=3, start=1.)
    hold = (1-.1*dt/2)/(1+.1*dt/2)
    gain = (2-(3.**2+2./2.)*dt**2)/(1+.1*dt/2)
    roots = np.linalg.eigvals([[gain, -hold], [1., 0.]])
    expected = np.r_[1.+0j, roots]
    frequency = -np.angle(expected)/(2*np.pi*dt)
    order = np.argsort(frequency)
    np.testing.assert_allclose(out['frequency'], frequency[order], atol=2e-12)
    np.testing.assert_allclose(out['decay_rate'], -np.log(abs(expected[order]))/dt, atol=2e-12)
    assert out['relative_residual'] < 1e-12


def test_damped_medium_modes_converge_to_the_continuous_oscillator():
    expected = np.linalg.eigvals([[0., 1.], [-10., -.1]])
    frequency = max(abs(expected.imag))/(2*np.pi)
    errors = []
    for dt in (.08, .04, .02):
        t, signal = material_record(dt)
        out = _legacy_wavebit.fit_fdtd_modes(t, signal, modes=3)
        errors.append(abs(out['frequency'][-1]-frequency))
        np.testing.assert_allclose(out['decay_rate'][-1], .05, rtol=1e-5)
    assert errors[-1] < 1e-4
    assert errors[0]/errors[1] > 3.9 and errors[1]/errors[2] > 3.9


@pytest.mark.parametrize('changes', [
    {'modes': True}, {'modes': 0}, {'modes': 1.5}, {'modes': 11},
    {'pencil': True}, {'pencil': 0}, {'pencil': 21}, {'pencil': 2.5},
    {'rcond': True}, {'rcond': 0}, {'rcond': 1}, {'rcond': np.nan}, {'rcond': [1e-8]},
    {'start': -1.}, {'stop': 30.}, {'start': 10., 'stop': 9.}, {'start': 19.5},
])
def test_invalid_fit_options_are_refused(changes):
    t = np.arange(21, dtype=float)
    with pytest.raises(ValueError):
        _legacy_wavebit.fit_fdtd_modes(t, np.exp(-.02*t), **changes)


@pytest.mark.parametrize('kind', ['irregular', 'repeated', 'complex_time', 'short', 'nan',
                                  'wrong_shape', 'zero', 'rank', 'impulse', 'object'])
def test_bad_records_and_unresolved_models_are_refused(kind):
    t = np.arange(21, dtype=float); signal = np.exp(-.03*t); modes = 1
    if kind == 'irregular':
        t[10] += .1
    elif kind == 'repeated':
        t[10] = t[9]
    elif kind == 'complex_time':
        t = t+0j
    elif kind == 'short':
        t, signal = t[:2], signal[:2]
    elif kind == 'nan':
        signal[10] = np.nan
    elif kind == 'wrong_shape':
        signal = signal[:, None]
    elif kind == 'zero':
        signal *= 0
    elif kind == 'rank':
        modes = 2
    elif kind == 'impulse':
        signal[:] = 0; signal[0] = 1
    else:
        signal = np.array(['a']*21)
    with pytest.raises(ValueError):
        _legacy_wavebit.fit_fdtd_modes(t, signal, modes=modes)
