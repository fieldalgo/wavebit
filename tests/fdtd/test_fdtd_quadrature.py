"""Temporal quadrature must rotate a physical field, not its spatial Bloch phase."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.fdtd._periodic import legacy as core


def test_quadrature_source_has_positive_i_phase_in_the_published_dft_convention():
    t = np.linspace(-12., 12., 24001)
    sine = core.gaussian_pulse(t, 1., 2., 0.)
    quadrature = core.gaussian_pulse(t, 1., 2., 0., quadrature=True)
    assert core.gaussian_pulse(0., 1., 2., 0., quadrature=True) == -1
    for frequency in (.8, 1., 1.2):
        kernel = np.exp(2j*np.pi*frequency*t)
        np.testing.assert_allclose(np.sum(quadrature*kernel)/np.sum(sine*kernel),
                                   1j, atol=2e-13)


@pytest.mark.parametrize('crossed', [False, True])
@pytest.mark.parametrize('hand', [-1, 1])
def test_vacuum_circular_record_rotates_with_the_requested_handedness(crossed, hand):
    result, diag = _legacy_wavebit.fdtd(
        structure=np.ones((1, 2 if crossed else 1, 2)), thickness=[.1], top=1., bot=1.,
        period=[.1, .14] if crossed else [.1], wavelength=.9, theta=0., phi=0.,
        polarization=(1., hand*1j), resolution=20., n_periods=160,
        record_field=3, return_diagnostics=True)
    z = int(np.argmin(abs(diag['field_z'] - .05)))
    ey = diag['field_e'][:, z, 0, 0].real
    ex = diag['field_ex'][:, z, 0, 0].real
    t = diag['field_t']
    assert np.max(abs(ey)) > .05 and np.max(abs(ex)) > .05
    # These vacuum components share z/t; no spatial Bloch or interpolation phase.
    np.testing.assert_allclose(np.sum(ex**2)/np.sum(ey**2), 1., atol=1e-9)
    assert abs(np.dot(ex, ey))/np.sqrt(np.dot(ex, ex)*np.dot(ey, ey)) < 1e-9
    kernel = np.exp(2j*np.pi*t/.9)
    # The Gaussian starts at finite t=dt, four envelope widths before its peak.
    # Its omitted tail is exp(-16), so an exactly analytic +/-i ratio is not the
    # right oracle at 1e-9. Compute the specified finite source independently.
    source_t = (np.arange(diag['n_steps'])+1)*diag['grid'].dt
    width = core.SOURCE_CYCLES*.9
    relative_t = source_t-core.SOURCE_DELAY*width
    envelope = np.exp(-(relative_t/width)**2)
    phase = 2*np.pi*relative_t/.9
    source_kernel = np.exp(2j*np.pi*source_t/.9)
    source_ratio = (np.sum(-np.cos(phase)*envelope*source_kernel)
                    / np.sum(np.sin(phase)*envelope*source_kernel))
    assert abs(source_ratio-1j) < 1e-8
    np.testing.assert_allclose(np.sum(ex*kernel)/np.sum(ey*kernel),
                               -hand*source_ratio, atol=1e-9, rtol=0)
    # Time-domain handedness also checked directly, without the production DFT.
    area = np.sum(ey[:-1]*ex[1:] - ex[:-1]*ey[1:])
    assert hand*area < 0
    assert abs(float(result.R.sum()+result.T.sum())-1) < .01


@pytest.mark.parametrize('crossed,theta,phi', [(False, 0., .4), (False, .3, 0.),
                                               (False, .3, .4), (True, .3, .4)])
def test_elliptical_recording_preserves_diffraction_and_physical_superposition(crossed, theta, phi):
    common = dict(structure=np.full((1, 2 if crossed else 1, 2), 2.25), thickness=[.1],
                  top=1., bot=1.44, period=[.1, .14] if crossed else [.1],
                  wavelength=.9, theta=theta, phi=phi, resolution=20., n_periods=60)
    pair = np.array([.3+.4j, .7-.2j])
    pair /= np.linalg.norm(pair)
    recorded, diag = _legacy_wavebit.fdtd(**common, polarization=pair, record_field=10,
                                  return_diagnostics=True)
    plain = _legacy_wavebit.fdtd(**common, polarization=pair)
    for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        np.testing.assert_array_equal(getattr(recorded.incident, key), getattr(plain.incident, key))
    assert np.isfinite(diag['field_e']).all() and np.isfinite(diag['field_ex']).all()
    # A global sign reverses every instantaneous field, including the quadrature,
    # while leaving power unchanged. It must not become a spatial conjugation.
    negative, flipped = _legacy_wavebit.fdtd(**common, polarization=-pair, record_field=10,
                                     return_diagnostics=True)
    for key in ('field_e', 'field_ex'):
        np.testing.assert_allclose(flipped[key], -diag[key], atol=1e-14)
    np.testing.assert_allclose(negative.R, recorded.R, atol=1e-14)


def test_normal_incidence_real_coefficients_do_not_launch_quadrature_runs(monkeypatch):
    original = core._time_loop
    quadratures = []
    def count(*args, **kwargs):
        quadratures.append(kwargs.get('source_quadrature', False))
        return original(*args, **kwargs)
    monkeypatch.setattr(core, '_time_loop', count)
    common = dict(structure=np.ones((1, 1, 2)), thickness=[.1], top=1., bot=1.,
                  period=[.1], wavelength=.9, resolution=15., n_periods=60, record_field=20)
    _legacy_wavebit.fdtd(**common, psi=.4)
    assert quadratures == [False, False]
    quadratures.clear()
    _legacy_wavebit.fdtd(**common, polarization=(1., 1j))
    assert quadratures == [False, False, True]


def test_broadband_elliptical_record_keeps_every_frequency_result():
    common = dict(structure=np.full((1, 1, 2), 2.25), thickness=[.1], top=1., bot=1.44,
                  period=[.1], wavelength=[.85, .9, .95], theta=0., phi=.4,
                  psi=.3, chi=.2, resolution=20., n_periods=80)
    recorded, diag = _legacy_wavebit.fdtd(**common, record_field=5, return_diagnostics=True)
    plain = _legacy_wavebit.fdtd(**common)
    assert len(recorded) == len(plain) == 3
    for left, right in zip(recorded, plain):
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            np.testing.assert_array_equal(getattr(left.incident, key), getattr(right.incident, key))
    assert np.isfinite(diag['field_e']).all() and np.isfinite(diag['field_ex']).all()
