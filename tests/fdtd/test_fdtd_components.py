"""Native Yee records must carry spatial staggering, time staggering and field signs."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest



def scene(**extra):
    kw = dict(structure=np.ones((1, 1, 2)), thickness=[.1], top=1., bot=1.,
              period=[.2], wavelength=.9, resolution=20., n_periods=80,
              record_field=3, return_diagnostics=True)
    kw.update(extra)
    return kw


@pytest.mark.parametrize('tetm,nonzero', [('te', ('ey', 'hx')), ('tm', ('ex', 'hy'))])
def test_scalar_components_share_maxwell_sign_and_native_timestamps(tetm, nonzero):
    _, diag = _legacy_wavebit.fdtd(**scene(tetm=tetm, record_components='all'))
    fields, grid = diag['fields'], diag['grid']
    assert set(fields) == {'ex', 'ey', 'ez', 'hx', 'hy', 'hz'}
    for name, field in fields.items():
        assert field['values'].shape == tuple(len(field[a]) for a in ('t', 'z', 'y', 'x'))
        if name.startswith('h'):
            np.testing.assert_allclose(field['t'], diag['field_t']-grid.dt/2, atol=1e-14)
        else:
            np.testing.assert_array_equal(field['t'], diag['field_t'])
        if name not in nonzero:
            np.testing.assert_allclose(field['values'], 0., atol=1e-14)
    # Fourier-transform each at ITS OWN z and t. In vacuum downward propagation
    # has Hx=Ey, Hy=-Ex in public +z-up coordinates. Comparing raw array
    # indices would introduce two phases.
    amplitudes = {}
    for name in nonzero:
        field = fields[name]
        index = int(np.argmin(abs(field['z']-.05)))
        trace = field['values'][:, index, 0, 0]
        kernel = np.exp(2j*np.pi*field['t']/.9)
        amplitudes[name] = np.sum(trace*kernel)*np.exp(2j*np.pi*field['z'][index]/.9)
    e, h = nonzero
    np.testing.assert_allclose(amplitudes[h]/amplitudes[e], 1 if tetm == 'te' else -1,
                               atol=.004, rtol=0)


@pytest.mark.parametrize('crossed,theta,phi,grading', [(False, .3, 0., None),
                                                    (False, .3, .4, 1.04),
                                                    (True, .3, .4, None)])
def test_all_components_preserve_diffraction_axes_and_legacy_records(crossed, theta, phi, grading):
    kw = scene(structure=np.full((1, 2 if crossed else 1, 2), 2.25),
               period=[.2, .3] if crossed else [.2], psi=.4, chi=.2,
               theta=theta, phi=phi, grading=grading, n_periods=80)
    got, diag = _legacy_wavebit.fdtd(**kw, record_components='all')
    plain, legacy = _legacy_wavebit.fdtd(**kw)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_array_equal(getattr(got.incident, name), getattr(plain.incident, name))
    assert 'fields' not in legacy
    f, g = diag['fields'], diag['grid']
    for name, field in f.items():
        assert np.isfinite(field['values']).all()
        assert field['values'].shape == tuple(len(field[a]) for a in ('t', 'z', 'y', 'x'))
        half = name in ('ez', 'hx', 'hy')
        nodes = diag['z_nodes']
        np.testing.assert_allclose(field['z'], (nodes[:-1]+nodes[1:])/2 if half else nodes,
                                   atol=1e-14, rtol=0)
    np.testing.assert_allclose(f['ex']['x']-f['ey']['x'], g.dx/2)
    np.testing.assert_array_equal(f['ez']['x'], f['ey']['x'])
    np.testing.assert_array_equal(f['hz']['x'], f['ex']['x'])
    if crossed:
        np.testing.assert_allclose(f['ey']['y']-f['ex']['y'], g.dy/2)
        np.testing.assert_array_equal(f['hz']['y'], f['ey']['y'])
        np.testing.assert_array_equal(f['ez']['y'], f['ex']['y'])
    for name, key in (('ey', 'field_e'), ('ex', 'field_ex')):
        np.testing.assert_array_equal(f[name]['values'], diag[key])
        np.testing.assert_array_equal(f[name]['values'], legacy[key])
    assert np.max(abs(f['ez']['values'])) > .01
    assert np.max(abs(f['hz']['values'])) > .01


@pytest.mark.parametrize('components', [[], ['hx', 'hx'], ['invalid'], 42])
def test_invalid_component_requests_fail_early(components):
    with pytest.raises(ValueError, match='record_components'):
        _legacy_wavebit.fdtd(**scene(tetm='te', record_components=components))


def test_component_selection_needs_a_record_and_allocates_only_selected_extras():
    with pytest.raises(ValueError, match='record_field'):
        _legacy_wavebit.fdtd(**scene(tetm='te', record_field=None, record_components='hx'))
    _, d = _legacy_wavebit.fdtd(**scene(tetm='te', record_components=['hx', 'hz']))
    assert set(d['fields']) == {'hx', 'hz'}


def test_six_component_vacuum_record_obeys_transversality_and_maxwell_curl():
    theta, phi, wavelength = .3, .4, .9
    _, diag = _legacy_wavebit.fdtd(**scene(theta=theta, phi=phi, polarization=(1., 1j),
                                  resolution=30., n_periods=100, record_components='all'))
    direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)])
    amplitudes = []
    for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        field = diag['fields'][name]
        iz = int(np.argmin(abs(field['z']-.05)))
        position = np.array([field['x'][0], field['y'][0], field['z'][iz]])
        kernel = np.exp(2j*np.pi*field['t']/wavelength)
        amplitudes.append(np.sum(field['values'][:, iz, 0, 0]*kernel)
                          * np.exp(-2j*np.pi*np.dot(direction, position)/wavelength))
    electric, magnetic = np.asarray(amplitudes[:3]), np.asarray(amplitudes[3:])
    scale = np.linalg.norm(electric)
    assert abs(np.dot(direction, electric))/scale < .008
    assert abs(np.dot(direction, magnetic))/scale < .008
    assert np.linalg.norm(magnetic-np.cross(direction, electric))/scale < .008
