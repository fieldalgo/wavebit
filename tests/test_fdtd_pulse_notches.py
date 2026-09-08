"""Source spectral zeros, calibrated optics and shared-backend sensitivities."""
import numpy as np
import pytest

import wavebit
from wavebit._core.scene_fdtd import _notched_pulse


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=.1, wavelength=[.9, 1., 1.1], backend=0, resolution=10,
                     n_periods=45, n_pml=4, truncation_order=0, tetm='te',
                     pulse_width=2., return_diagnostics=True), **changes)


@pytest.mark.parametrize('notches', [.8, [.8, 1.2], [.8, .8]])
def test_finite_spectrum_has_requested_zeros_and_the_correct_delay_sign(notches):
    dt, count = .02, 701
    t = np.arange(count)*dt
    original = np.exp(-((t-5)/1.7)**2)*np.exp(-2j*np.pi*t)
    frequencies = np.array([.91, 1., 1.11])
    pulse, info = _notched_pulse(original, dt, frequencies, notches, np.finfo(float).eps, 1.)
    zeros = np.atleast_1d(notches)
    all_f = np.r_[frequencies, zeros]
    spectra = np.exp(2j*np.pi*all_f[:, None]*t)@pulse
    scale = np.sum(abs(pulse))
    np.testing.assert_allclose(spectra[len(frequencies):]/scale, 0, atol=2e-12)
    # Finite convolution theorem, including the discarded base endpoint samples.
    base = np.exp(2j*np.pi*frequencies[:, None]*t[:-len(zeros)])@original[:-len(zeros)]
    multiplier = np.prod(1-np.exp(2j*np.pi*(frequencies[:, None]-zeros)*dt), axis=1)
    expected_ratio = base*multiplier/(base[1]*multiplier[1])
    np.testing.assert_allclose(spectra[:3]/spectra[1], expected_ratio, atol=2e-10)
    np.testing.assert_allclose(info['relative_spectrum'], abs(spectra[:3])/scale, atol=2e-15)
    assert len(pulse) == count and np.max(abs(pulse)) == pytest.approx(1.)
    assert info['spectrum_kind'] == 'sampled_l1'
    np.testing.assert_array_equal(original, np.exp(-((t-5)/1.7)**2)*np.exp(-2j*np.pi*t))


@pytest.mark.parametrize('notches', [[], [[.8]], True, 0., -.1, np.nan, np.inf, 1j, '1', [True], 100., [.8]*3])
def test_invalid_notches_fail_before_time_stepping(monkeypatch, notches):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid notch reached a time-domain run')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='pulse_notches'):
        wavebit.fdtd(**options(pulse_notches=notches))


@pytest.mark.parametrize('changes', [
    {'pulse_notches': 1/.9},
    {'wavelength': [.9, 1.1], 'pulse_notches': 1., 'record_field': 10},
    {'pulse_notches': 1.+1e-15},  # Indistinguishable within the 32-epsilon frequency guard.
])
def test_notches_cannot_remove_required_calibration(changes):
    with pytest.raises(ValueError, match='pulse_notches'):
        wavebit.fdtd(**options(**changes))


def test_weak_excitation_guard_tracks_backend_precision():
    t = np.arange(1201)*.02
    original = np.exp(-((t-10)/2)**2)*np.exp(-2j*np.pi*t)
    kw = dict(pulse=original, dt=.02, frequencies=np.array([1.]), notches=1.+1e-7, carrier=1.)
    _notched_pulse(**kw, epsilon=np.finfo(np.float64).eps)
    with pytest.raises(ValueError, match='numerical source tail'):
        _notched_pulse(**kw, epsilon=np.finfo(np.float32).eps)


def test_notch_prevents_excitation_of_an_independent_undamped_material_mode():
    # Uniform D/P dynamics have omega_continuous^2 = omega0^2+strength/eps.
    # The leapfrog oscillator's frequency is known before any field solve.
    dt, count, omega0, strength, eps = .04, 1500, 3., 2., 2.
    frequency = np.arcsin(dt*np.sqrt(omega0**2+strength/eps)/2)/(np.pi*dt)
    t = np.arange(count)*dt
    pulse = np.exp(-((t-25)/5)**2)*np.exp(-2j*np.pi*frequency*t)
    shaped, _ = _notched_pulse(pulse, dt, np.array([.4, .6]), frequency,
                               np.finfo(float).eps, frequency)
    source = np.zeros((3, 2, 2, 2)); source[1] = 1
    options = dict(spacing=(.2,)*3, dt=dt, steps=count, source=source,
                   poles=[(omega0, 0., strength)])
    baseline = wavebit.fdtd_grid(np.full((2, 2, 2), eps), pulse=pulse, **options)
    filtered = wavebit.fdtd_grid(np.full((2, 2, 2), eps), pulse=shaped, **options)
    assert np.linalg.norm(baseline['E']) > .1
    assert np.linalg.norm(filtered['E']) < 1e-8*np.linalg.norm(baseline['E'])
    assert np.linalg.norm(filtered['polarization'][0]) < 1e-8*np.linalg.norm(baseline['polarization'][0])


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_same_filtered_pulse_reaches_both_bases_and_references(monkeypatch, side):
    import wavebit._core.scene_fdtd as scene
    original = scene.fdtd_grid
    pulses = []
    def capture(*args, **kwargs):
        pulses.append(np.array(kwargs['pulse']))
        return original(*args, **kwargs)
    monkeypatch.setattr(scene, 'fdtd_grid', capture)
    _, d = wavebit.fdtd(**options(pulse_notches=.8, incidence_side=side,
                                  record_field=101, record_components=['ey'],
                                  record_bounds={'z': (.05, .15)}, grading=True))
    assert len(pulses) == 4
    for pulse in pulses[1:]:
        np.testing.assert_array_equal(pulse, pulses[0])
    t = np.arange(d['steps'])*d['dt']
    assert abs(np.sum(pulses[0]*np.exp(2j*np.pi*.8*t)))/np.sum(abs(pulses[0])) < 1e-12
    assert np.isfinite(d['fields']['ey']['values']).all()
    assert d['pulse']['carrier_fraction'] > .1
    np.testing.assert_array_equal(d['pulse']['notches'], [.8])


def test_unrecorded_carrier_can_be_notched_between_requested_frequencies():
    results, d = wavebit.fdtd(**options(wavelength=[.9, 1.1], pulse_notches=1.))
    assert d['pulse']['carrier_fraction'] < 1e-12
    assert np.min(d['pulse']['relative_spectrum']) > .1
    for result in results:
        assert np.isfinite(result.given.r_s).all()


def test_notch_frequency_scales_inversely_with_scene_length():
    first, a = wavebit.fdtd(**options(pulse_notches=.8))
    second, b = wavebit.fdtd(**options(pulse_notches=.4, pulse_width=4., thickness=[.4],
                                      period=.2, wavelength=[1.8, 2., 2.2], resolution=5))
    for before, after in zip(first, second):
        for name in ('r_s', 't_s', 'R', 'T'):
            np.testing.assert_allclose(getattr(before.given, name), getattr(after.given, name), atol=1e-11)
    np.testing.assert_allclose(a['pulse']['relative_spectrum'], b['pulse']['relative_spectrum'], atol=1e-14)
    assert b['pulse']['sampled_peak_time'] == 2*a['pulse']['sampled_peak_time']


def test_notched_oblique_slab_refines_to_independent_airy_amplitudes():
    band = np.array([.9, 1., 1.1])
    transverse = band*np.sin(.25)
    q0, q1 = np.sqrt(1-transverse**2), np.sqrt(2.25-transverse**2)
    phase = np.exp(2j*np.pi*q1*.213/band)
    expected = {}
    for component, r in [('s', (q0-q1)/(q0+q1)), ('p', (2.25*q0-q1)/(2.25*q0+q1))]:
        denominator = 1-r*r*phase*phase
        expected[component] = np.array([r*(1-phase*phase)/denominator,
                                         (1-r*r)*phase/denominator])
    errors = []
    for resolution in (40, 80):
        results, d = wavebit.fdtd(**options(resolution=resolution, n_pml=round(.3*resolution),
                                            thickness=[.213], theta=.25, pulse_notches=[.75, 1.25],
                                            align_layers=True, diffraction_method='directional',
                                            diffraction_distance=.2, convergence_window=200))
        error = 0.
        for component, basis in [('s', 's_inc'), ('p', 'p_inc')]:
            actual = np.array([[getattr(getattr(r, basis), key+'_'+component).item()
                                for r in results] for key in ('r', 't')])
            error = max(error, np.max(abs(actual-expected[component])))
        errors.append(error)
        assert d['convergence']['scattering_settled']
    assert errors[1] < .003 and errors[1] < .4*errors[0]


@pytest.mark.parametrize('backend', [1, 2])
def test_notched_source_preserves_material_mu_pole_and_record_gradients(backend):
    def objective(p, which):
        result, d = wavebit.fdtd(**options(
            structure=p[:1].reshape(1, 1, 1), mu=p[1:2].reshape(1, 1, 1), wavelength=1.,
            dispersion=np.ones((1, 1, 1), int), poles={1: [(8., .3, p[2])]}, backend=which,
            pulse_notches=[.8, 1.3], grading=True, incidence_side='bottom',
            tetm=None, theta=.2, phi=.4, polarization=(1., .6j), dft_subsample=2,
            diffraction_method='directional', diffraction_distance=.2, convergence_window=100,
            record_field=101, record_components=['ex'], record_bounds={'z': (.05, .15)}))
        r = result.given
        return ((.2*r.r_s+.3j*r.r_p+.4*r.t_s+.5j*r.t_p).real.sum()
                +.1*(abs(d['fields']['ex']['values'])**2).mean()
                +.3*(d['convergence']['scattering_change_bound']['r_p']**2).sum())
    p = np.array([2.25, 1.05, 1.])
    expected = objective(p, 0)
    delta = 1e-4
    finite = np.array([(objective(p+v, 0)-objective(p-v, 0))/(2*delta) for v in np.eye(3)*delta])
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
        value, gradient = jax.jit(jax.value_and_grad(lambda v: objective(v, 1)))(p)
    else:
        torch = pytest.importorskip('torch')
        parameter = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        value = objective(parameter, 2)
        gradient, = torch.autograd.grad(value, parameter)
        value, gradient = value.detach().numpy(), gradient.detach().numpy()
    np.testing.assert_allclose(value, expected, atol=1e-11)
    np.testing.assert_allclose(gradient, finite, rtol=3e-5, atol=1e-8)
    assert np.min(abs(finite)) > 1e-5
