"""Common-point interpolation and physical instantaneous E cross H, not phasor power."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.fdtd._periodic.probes import fdtd_poynting as _legacy_wavebit_fdtd_poynting

from wavebit.fdtd._periodic.probes import sample_fdtd_fields as _legacy_wavebit_sample_fdtd_fields
from types import SimpleNamespace

import numpy as np
import pytest



@pytest.fixture
def affine():
    fields = {}
    for index, name in enumerate(('ex', 'ey', 'ez', 'hx', 'hy', 'hz')):
        shift = .1*index
        axes = dict(t=np.array([0., 1., 2., 4.])+shift,
                    z=np.array([-2., -1., 1., 3.])+shift,
                    y=np.array([-2., 0., 3.])+shift,
                    x=np.array([-2., -.3, 1., 3.])+shift)
        t, z, y, x = np.meshgrid(*(axes[a] for a in ('t', 'z', 'y', 'x')), indexing='ij')
        # Cross terms detect interpolation on the wrong axis; multilinear values
        # are reproduced exactly on nonuniform and differently staggered grids.
        values = (index+1)*(t + 2*z - 3*y + 4*x + .2*t*x + .3*z*y) + 1j*(7-index)
        fields[name] = dict(values=values, **axes)
    return dict(fields=fields)


def test_interpolation_is_exact_for_multilinear_fields_on_staggered_nonuniform_axes(affine):
    times = np.array([.75, 1.3, 2.])
    got = _legacy_wavebit_sample_fdtd_fields(affine, x=.2, y=.4, z=.5, t=times)
    base = times + 1 - 1.2 + .8 + .2*times*.2 + .3*.5*.4
    expected = np.stack([(i+1)*base+1j*(7-i) for i in range(6)], axis=1)
    np.testing.assert_allclose(got['E'], expected[:, :3], atol=1e-13)
    np.testing.assert_allclose(got['H'], expected[:, 3:], atol=1e-13)
    assert got['region'] is None
    assert (got['x'], got['y'], got['z']) == (.2, .4, .5)


def test_default_times_use_only_common_support_and_scalar_time_has_one_row(affine):
    default = _legacy_wavebit_sample_fdtd_fields(affine, z=0.)
    np.testing.assert_array_equal(default['t'], [1., 2., 4.])
    single = _legacy_wavebit_sample_fdtd_fields(affine, z=0., t=1.5)
    assert single['E'].shape == single['H'].shape == (1, 3)


def test_instantaneous_power_uses_real_fields_not_conjugate_phasor_product(affine):
    for i, name in enumerate(('ex', 'ey', 'ez', 'hx', 'hy', 'hz')):
        affine['fields'][name]['values'][:] = [1+5j, 2+6j, 3+7j, 4+8j, 5+9j, 6+10j][i]
    result = _legacy_wavebit_fdtd_poynting(affine, z=0., t=1.)
    np.testing.assert_allclose(result['S'], [[-3., 6., -3.]], atol=1e-13, rtol=0)


@pytest.mark.parametrize('kw', [dict(x=-3.), dict(y=5.), dict(z=9.), dict(t=0.),
                               dict(t=[1., .9]), dict(t=[]), dict(t=np.nan),
                               dict(t=1j), dict(x=1j), dict(z=np.inf)])
def test_outside_invalid_or_unordered_queries_are_refused(affine, kw):
    with pytest.raises(ValueError):
        _legacy_wavebit_sample_fdtd_fields(affine, **{'z': 0., **kw})


def test_missing_components_and_single_snapshot_have_clear_failures(affine):
    with pytest.raises(ValueError, match='all six'):
        _legacy_wavebit_fdtd_poynting({}, z=0.)
    for field in affine['fields'].values():
        field['t'] = field['t'][:1]
        field['values'] = field['values'][:1]
    with pytest.raises(ValueError, match='time vector'):
        _legacy_wavebit_sample_fdtd_fields(affine, z=0.)


def test_source_discontinuity_is_not_interpolated_and_regions_are_reported(affine):
    affine['grid'] = SimpleNamespace(z_node=np.array([-1., 1.]), k_tfsf=1)
    with pytest.raises(ValueError, match='TFSF'):
        _legacy_wavebit_fdtd_poynting(affine, z=.3)
    assert _legacy_wavebit_fdtd_poynting(affine, z=-1.4)['region'] == 'scattered'
    # This point still interpolates above/below the real material grid but lies
    # entirely on the total-field side for all manufactured staggered records.
    assert _legacy_wavebit_fdtd_poynting(affine, z=2.)['region'] == 'total'


def test_undersampled_records_warn_instead_of_claiming_temporal_reconstruction(affine):
    affine['wavelengths'] = np.array([1.])
    with pytest.warns(UserWarning, match='undersamples'):
        _legacy_wavebit_fdtd_poynting(affine, z=0.)


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_vacuum_energy_flows_forward_and_improves_with_record_stride(tetm):
    _, diag = _legacy_wavebit_fdtd(structure=np.ones((1, 1, 2)), thickness=[.1],
                          top=1., bot=1., period=[.2], wavelength=.9, tetm=tetm,
                          resolution=30., n_periods=80, record_field=1,
                          record_components='all', return_diagnostics=True)
    probe = _legacy_wavebit_fdtd_poynting(diag, z=.05)
    assert probe['region'] == 'total'
    assert np.sum(probe['S'][:, 2]) < 0
    np.testing.assert_allclose(probe['S'][:, :2], 0., atol=1e-14)
    # A downward vacuum wave satisfies Sz=-|E|^2; interpolation attenuates the
    # half-step H carrier. More frequent recording must improve that estimate.
    def error(sample):
        electric = np.sum(sample['E'].real**2, axis=1)
        return abs(-np.sum(sample['S'][:, 2])/np.sum(electric)-1)
    coarse = dict(diag, fields={name: dict(field, values=field['values'][::5], t=field['t'][::5])
                               for name, field in diag['fields'].items()})
    assert error(probe) < .02
    assert error(probe) < error(_legacy_wavebit_fdtd_poynting(coarse, z=.05))


@pytest.mark.parametrize('theta,phi,pair', [(.3, .4, (1., 1j)), (.3, 2.7, (1., 0.))])
def test_oblique_pulse_energy_follows_its_wavevector(theta, phi, pair):
    _, diag = _legacy_wavebit_fdtd(structure=np.ones((1, 2, 2)), thickness=[.1],
                          top=1., bot=1., period=[.1, .14], wavelength=.9,
                          theta=theta, phi=phi, polarization=pair,
                          resolution=20., n_periods=80, record_field=2,
                          record_components='all', return_diagnostics=True)
    probe = _legacy_wavebit_fdtd_poynting(diag, z=.05)
    flow = np.sum(probe['S'], axis=0)
    expected = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)])
    np.testing.assert_allclose(flow/np.linalg.norm(flow), expected, atol=.01, rtol=0)


def test_legacy_real_carrier_bloch_records_are_not_reported_as_physical_flux(affine):
    affine['grid'] = SimpleNamespace(kx_inc=.2, ky_inc=0., z_node=np.array([-1., 1.]), k_tfsf=1)
    with pytest.raises(ValueError, match='positive-frequency'):
        _legacy_wavebit_fdtd_poynting(affine, z=2.)


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_pure_oblique_record_uses_positive_carrier_without_changing_diffraction(tetm):
    common = dict(structure=np.ones((1, 1, 2)), thickness=[.1], top=1., bot=1.,
                  period=[.1], wavelength=.9, tetm=tetm, theta=.3,
                  resolution=20., n_periods=80)
    result, diag = _legacy_wavebit_fdtd(**common, record_field=2, record_components='all',
                                return_diagnostics=True)
    plain = _legacy_wavebit_fdtd(**common)
    assert diag['field_temporal_representation'] == 'positive_frequency_carrier'
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_array_equal(getattr(result.incident, key), getattr(plain.incident, key))
    flux = _legacy_wavebit_fdtd_poynting(diag, z=.05)['S'].sum(axis=0)
    np.testing.assert_allclose(flux/np.linalg.norm(flux), [np.sin(.3), 0., -np.cos(.3)],
                               atol=.01, rtol=0)
