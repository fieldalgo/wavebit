"""Jones phase, coherent interference and normalization on each FDTD route."""
import numpy as np
import pytest

import wavebit
from wavebit._core import fdtd as core
from wavebit._core.result import Result, ResultSub


AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')


def scene(crossed=False, theta=.3, phi=.4):
    return dict(structure=np.full((1, 2 if crossed else 1, 2), 2.25), thickness=[.1],
                top=1., bot=1.44, period=[.1, .14] if crossed else [.1],
                wavelength=.9, theta=theta, phi=phi, truncation_order=dict(x=0, y=0))


@pytest.mark.parametrize('crossed,theta,phi', [(False, 0., .4), (False, .3, 0.),
                                               (False, .3, .4), (True, .3, 0.),
                                               (True, .3, .4)])
def test_circular_planar_amplitudes_match_rcwa(crossed, theta, phi):
    # Opposite handedness changes a relative phase, not simply the two intensities.
    common = scene(crossed, theta, phi)
    for pair in ((1., 1j), (1., -1j)):
        result = wavebit.fdtd(**common, polarization=pair, resolution=30., n_periods=80)
        reference = wavebit.rcwa(**common, polarization=pair, backend=0).solve()
        for key in AMPLITUDES:
            np.testing.assert_allclose(getattr(result.given, key),
                                       getattr(reference.given, key), atol=.015, rtol=0)
        assert abs(float(result.R.sum() + result.T.sum()) - 1) < .015


@pytest.fixture(scope='module')
def patterned():
    common = scene(True)
    common.update(structure=np.array([[[2.25, 1.], [1., 1.44]]]), period=[.6, .8])
    result = wavebit.fdtd(**common, polarization=(1., 1j), resolution=20., n_periods=80)
    return common, result


def test_crossed_ellipse_retains_complex_interference_and_overall_phase(patterned):
    common, circular = patterned
    psi, chi, phase = .37, -.22, np.exp(.63j)
    # Stated independently from the production resolver.
    pair = np.array([np.sin(psi)*np.cos(chi) + 1j*np.cos(psi)*np.sin(chi),
                     np.cos(psi)*np.cos(chi) - 1j*np.sin(psi)*np.sin(chi)])
    chart = wavebit.fdtd(**common, psi=psi, chi=chi, resolution=20., n_periods=80)
    phased = wavebit.fdtd(**common, polarization=3*phase*pair, resolution=20., n_periods=80)
    for key in AMPLITUDES:
        expected = pair[0]*getattr(circular.s_inc, key) + pair[1]*getattr(circular.p_inc, key)
        np.testing.assert_allclose(getattr(chart.given, key), expected, atol=2e-14)
        np.testing.assert_allclose(getattr(phased.given, key), phase*expected, atol=2e-14)
    np.testing.assert_allclose(phased.R, chart.R, atol=2e-14)
    np.testing.assert_allclose(phased.T, chart.T, atol=2e-14)
    # A separate discretization checks the physical amplitudes, not only the
    # algebraic identity of combining this solver's own basis responses.
    reference_kw = dict(common, truncation_order=dict(x=5, y=5))
    reference = wavebit.rcwa(**reference_kw, psi=psi, chi=chi, backend=0).solve()
    for key in AMPLITUDES:
        np.testing.assert_allclose(getattr(chart.given, key)[0, 0],
                                   getattr(reference.given, key)[5, 5], atol=.025, rtol=0)
    incoherent = abs(pair[0])**2*circular.s_inc.R + abs(pair[1])**2*circular.p_inc.R
    assert np.max(np.abs(chart.R - incoherent)) > 1e-5


def test_typed_broadband_ellipse_uses_same_state_at_each_wavelength():
    stack = wavebit.Stack(layers=[wavebit.Layer(thickness=.1, structure=np.full((1, 2), 2.25))],
                          top=1., bot=1.44, period=(.1,))
    incidence = wavebit.Incidence(wavelength=[.85, .9, .95], theta=0., phi=.4, chi=.2)
    results = wavebit.fdtd(stack=stack, incidence=incidence, resolution=30., n_periods=80)
    for wavelength, result in zip(incidence.wavelength, results):
        reference = wavebit.rcwa(stack=stack, wavelength=wavelength, theta=0., phi=.4,
                                 chi=.2, backend=0).solve()
        for key in AMPLITUDES:
            np.testing.assert_allclose(getattr(result.given, key),
                                       getattr(reference.given, key), atol=.015, rtol=0)
    assert len(results) == 3


@pytest.mark.parametrize('pair', [(0, 0), (np.nan, 1), (1, np.inf), [1], [1, 2, 3],
                                 [[1], [1j]], 'rcp', 1])
def test_invalid_jones_vector_never_reaches_time_loop(monkeypatch, pair):
    monkeypatch.setattr(core, '_time_loop', lambda *a, **k: pytest.fail('time loop reached'))
    with pytest.raises(ValueError, match='polarization'):
        wavebit.fdtd(**scene(), polarization=pair)


@pytest.mark.parametrize('state', [dict(tetm='te', chi=.2), dict(tetm='tm', polarization=(1, 1j)),
                                   dict(psi=.4, polarization=(1, 1j)),
                                   dict(chi=.2, polarization=(1, 1j))])
def test_conflicting_polarization_spellings_are_refused(state):
    with pytest.raises(ValueError):
        wavebit.fdtd(**scene(), **state)


@pytest.mark.parametrize('scale', [1e-320, 1e-300, 1e300])
def test_extreme_jones_scale_preserves_phase(scale):
    pair = core._incident_pair(None, None, None, (scale, 1j*scale))
    np.testing.assert_allclose(pair, np.array([1, 1j])/np.sqrt(2), atol=1e-15)


def test_complex_time_records_are_not_misrepresented_as_a_real_pulse():
    with pytest.raises(ValueError, match='quadrature pulse'):
        wavebit.fdtd(**scene(), polarization=(1, 1j), record_field=10)


def test_flux_weight_falls_back_per_order_when_first_basis_has_a_dark_channel():
    # The two orders have different physical flux weights. Each basis has one dark
    # channel; choosing a single basis for the whole array silently drops an order.
    def leg(amplitude):
        a = np.array([amplitude], dtype=complex)
        power = abs(a)**2*np.array([[.4, .7]])
        sub = ResultSub(a, a, a, a, 2*power, power, power, 2*power, power, power)
        return Result(given=sub)
    s, p = leg([1, 0]), leg([0, 1])
    a = np.array([[1, 1j]])/np.sqrt(2)
    mixed = ResultSub(a, a, a, a, *([None]*6))
    result = core._efficiencies_from(mixed, s, p, 1/np.sqrt(2), 1j/np.sqrt(2))
    np.testing.assert_allclose(result.R, [[.4, .7]])
    np.testing.assert_allclose(result.T, [[.4, .7]])


def test_real_jones_vector_keeps_recording_contract():
    common = scene(False, theta=.3, phi=0.)
    angle = .37
    _, chart = wavebit.fdtd(**common, psi=angle, resolution=20., n_periods=60,
                            record_field=20, return_diagnostics=True)
    _, pair = wavebit.fdtd(**common, polarization=(np.sin(angle), np.cos(angle)),
                           resolution=20., n_periods=60, record_field=20,
                           return_diagnostics=True)
    for key in ('field_e', 'field_ex'):
        np.testing.assert_allclose(pair[key], chart[key], atol=1e-14)
