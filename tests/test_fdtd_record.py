"""The public psi route must return the requested field, not just one basis solve."""
import numpy as np
import pytest

import wavebit


def _record(psi, theta, phi, crossed=False, **extra):
    return wavebit.fdtd(
        structure=np.ones((1, 2 if crossed else 1, 2)), thickness=[100.0],
        top=1.0, bot=1.0, period=[300.0, 400.0] if crossed else [300.0],
        wavelength=900.0, psi=psi, theta=theta, phi=phi, resolution=0.015,
        n_periods=60, record_field=10, return_diagnostics=True, **extra)


@pytest.mark.parametrize('theta,phi,crossed', [(0., .4, False), (.3, 0., False),
                                             (.3, .4, False), (.3, .4, True),
                                             (0., 0., True), (0., .4, True), (.3, 0., True)])
def test_the_record_of_a_mixed_incidence_obeys_field_superposition(theta, phi, crossed):
    psi = .37
    _, s = _record(np.pi / 2, theta, phi, crossed)
    _, p = _record(0., theta, phi, crossed)
    result, mixed = _record(psi, theta, phi, crossed)
    for key in ('field_e', 'field_ex'):
        assert key in mixed, 'psi silently dropped record_field'
        np.testing.assert_allclose(mixed[key], np.sin(psi)*s[key] + np.cos(psi)*p[key],
                                   rtol=1e-12, atol=1e-14)
        assert np.max(np.abs(mixed[key])) > .01
    assert mixed['field_e_component'] == 'ey'
    assert mixed['residual'] == max(s['residual'], p['residual'])
    for key in ('field_t', 'field_z', 'field_x', 'field_y', 'field_ex_x', 'field_ex_y'):
        np.testing.assert_array_equal(mixed[key], s[key])
        np.testing.assert_array_equal(mixed[key], p[key])
    # Recording must not change the frequency-domain result.
    plain = wavebit.fdtd(
        structure=np.ones((1, 2 if crossed else 1, 2)), thickness=[100.], top=1., bot=1.,
        period=[300., 400.] if crossed else [300.], wavelength=900., psi=psi,
        theta=theta, phi=phi, resolution=.015, n_periods=60)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_array_equal(getattr(result.incident, name), getattr(plain.incident, name))


@pytest.mark.parametrize('theta,phi', [(0., .4), (.3, 0.)])
def test_scalar_records_use_the_incident_electric_vector(theta, phi):
    # At ky=0, s=(-sin(phi), cos(phi)) and p=-cos(theta)*(cos(phi), sin(phi)).
    # Public scalar records use the same unit incident s/p electric vectors;
    # at phi=0 their transverse fields are +Ey and -cos(theta)*Ex.
    psi = .37
    _, mixed = _record(psi, theta, phi)
    components = {}
    for tetm in ('te', 'tm'):
        _, components[tetm] = wavebit.fdtd(
            structure=np.ones((1, 1, 2)), thickness=[100.], top=1., bot=1.,
            period=[300.], wavelength=900., tetm=tetm, theta=theta,
            resolution=.015, n_periods=60, record_field=10, return_diagnostics=True)
    ey = np.sin(psi)*np.cos(phi) - np.cos(psi)*np.cos(theta)*np.sin(phi)
    ex = -np.sin(psi)*np.sin(phi) - np.cos(psi)*np.cos(theta)*np.cos(phi)
    np.testing.assert_allclose(mixed['field_e'], ey*components['te']['field_e'], atol=1e-14)
    np.testing.assert_allclose(mixed['field_ex'],
                               ex/(-np.cos(theta))*components['tm']['field_e'], atol=1e-14)
    np.testing.assert_array_equal(mixed['field_ex_x'], components['tm']['field_x'])


@pytest.mark.parametrize('stride', [0, -1, 1.5, True])
def test_psi_does_not_hide_an_invalid_record_stride(stride):
    with pytest.raises(ValueError, match='record_field'):
        wavebit.fdtd(structure=np.ones((1, 1, 2)), thickness=[100.], top=1., bot=1.,
                     period=[300.], wavelength=900., psi=.4, theta=.3, phi=.4,
                     resolution=.015, n_periods=60, record_field=stride)


@pytest.mark.parametrize('theta', [0., .3])
def test_crossed_psi_at_zero_azimuth_retains_both_order_axes(theta):
    common = dict(structure=np.array([[[2.25, 1.], [1., 1.44]]]), thickness=[100.],
                  top=1., bot=1., period=[1000., 1400.], wavelength=900., theta=theta,
                  resolution=.015, n_periods=60)
    s = wavebit.fdtd(tetm='te', **common)
    p, pd = wavebit.fdtd(tetm='tm', return_diagnostics=True, **common)
    psi = .37
    mixed = wavebit.fdtd(psi=psi, phi=0., **common)
    assert len(mixed.orders_v) > 1
    assert np.isfinite(pd['incident_balance'])
    assert pd['residual'] > 0  # a p source has zero Ey, but a nonzero Ex residual
    np.testing.assert_array_equal(mixed.orders_u, s.orders_u)
    np.testing.assert_array_equal(mixed.orders_v, s.orders_v)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        expected = np.sin(psi)*getattr(s.incident, name) + np.cos(psi)*getattr(p.incident, name)
        np.testing.assert_allclose(getattr(mixed.incident, name), expected, atol=1e-13)
