"""Diffraction plane placement and continuum/Yee cutoff diagnostics."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest
import wavebit


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.213],
                     top=1., bot=1., period=.1, wavelength=1., tetm='tm', theta=.25,
                     backend='numpy', resolution=20, n_pml=6, n_periods=35,
                     align_layers=True, truncation_order=0, return_diagnostics=True), **changes)


@pytest.mark.parametrize('side,grading', [('top', None), ('bottom', True)])
def test_moving_planes_preserves_mesh_and_dynamics_under_recalibration(side, grading):
    kw = options(incidence_side=side, grading=grading, pulse_width=3., n_periods=60,
                 convergence_window=100, record_field=43, record_components=['ex', 'hy'],
                 record_bounds={'z': (-.05, .25)}, field_monitors={'out': -.25})
    _, a = _legacy_wavebit.fdtd(**kw)
    _, b = _legacy_wavebit.fdtd(**kw, diffraction_distance=(.18, .27))
    assert bool(a['convergence']['scattering_settled'])
    assert bool(b['convergence']['scattering_settled'])
    for key in ('z_nodes', 'z_spacing', 'shape', 'dt', 'steps', 'source_z', 'pml_bounds'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('reflection_z', 'transmission_z'):
        assert a[key] == b[key]
    # Grid/PML reflections change incident calibration with position. For pure
    # TM at one carrier the independently measured frequency-plane ratio must
    # account for the entire time record: moving a detector changes no dynamics.
    ratio = b['monitors']['out']['fields']['ex'][0, 0, 0]/a['monitors']['out']['fields']['ex'][0, 0, 0]
    assert np.isfinite(ratio) and abs(ratio) > 0
    for name in a['fields']:
        np.testing.assert_allclose(ratio*a['fields'][name]['values'], b['fields'][name]['values'], atol=1e-12)
        for axis in ('x', 'y', 'z', 't'):
            np.testing.assert_array_equal(a['fields'][name][axis], b['fields'][name][axis])
    reflection_z, transmission_z = [b['diffraction'][key] for key in ('reflection_monitor_z', 'transmission_monitor_z')]
    target = (.213+.18, -.27) if side == 'top' else (-.18, .213+.27)
    np.testing.assert_allclose([reflection_z, transmission_z], target, atol=.05)
    assert reflection_z != a['diffraction']['reflection_monitor_z']


def test_near_planes_refine_to_fresnel_airy():
    q0 = np.cos(.25)
    q1 = np.sqrt(2.25-np.sin(.25)**2)
    r = (2.25*q0-q1)/(2.25*q0+q1)
    phase = np.exp(2j*np.pi*q1*.213)
    expected = np.array([r*(1-phase**2)/(1-r*r*phase**2), (1-r*r)*phase/(1-r*r*phase**2)])
    errors = []
    for resolution in (40, 80):
        result, _ = _legacy_wavebit.fdtd(**options(resolution=resolution, n_pml=round(.3*resolution),
                                           diffraction_distance=.2))
        errors.append(np.max(abs(np.array([result.incident.r_p.item(), result.incident.t_p.item()])-expected)))
    assert errors[1] < .002
    assert errors[1] < .4*errors[0]


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_near_plane_material_gradient_and_backend_metadata(backend):
    def objective(value, which):
        result, d = _legacy_wavebit.fdtd(**options(structure=value.reshape(1, 1, 1), backend=which,
                                           incidence_side='bottom', grading=True,
                                           diffraction_distance=(.2, .3), convergence_window=50))
        return (result.incident.r_p+.4j*result.incident.t_p).real.sum(), d['diffraction']['reflection']['q']
    expected, expected_q = objective(np.array(2.25), 'numpy')
    step = 1e-4
    finite = (objective(np.array(2.25+step), 'numpy')[0]-objective(np.array(2.25-step), 'numpy')[0])/(2*step)
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        (value, q), gradient = jax.jit(jax.value_and_grad(lambda x: objective(x, 'jax'), has_aux=True))(2.25)
        host = np.asarray
    else:
        torch = pytest.importorskip('torch')
        parameter = torch.tensor(2.25, dtype=torch.float64, requires_grad=True)
        value, q = objective(parameter, 'torch')
        gradient, = torch.autograd.grad(value, parameter)
        host = lambda x: x.detach().numpy()
    np.testing.assert_allclose(host(value), expected, atol=1e-11)
    np.testing.assert_allclose(host(q), expected_q, atol=1e-14)
    np.testing.assert_allclose(host(gradient), finite, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_conical_magnetic_half_space_metadata_is_incidence_relative(side):
    results, d = _legacy_wavebit.fdtd(**options(top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(1.8, 1.2),
                                 period=[.4, .6], wavelength=[.51, .55], theta=.2, phi=.6,
                                 tetm=None, polarization=(1., 1j), incidence_side=side,
                                 truncation_order=[1, 1], diffraction_distance=(.2, .3)))
    info = d['diffraction']
    np.testing.assert_array_equal(info['order_x'], np.tile([-1, 0, 1], (3, 1)))
    np.testing.assert_array_equal(info['order_y'], np.tile([[-1], [0], [1]], (1, 3)))
    other_orders = (info['order_x'] != 0) | (info['order_y'] != 0)
    for result in results:
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            # A uniform planar layer cannot diffract, even in a coupled frame.
            np.testing.assert_allclose(getattr(result.incident, key)[other_orders], 0., atol=1e-8)
    index = np.sqrt([1.2*1.1, 1.8*1.2])
    if side == 'bottom':
        index = index[::-1]
    w = 2*np.pi/np.array([.51, .55])[:, None, None]
    for name, n in zip(('reflection', 'transmission'), index):
        q2 = n*n-(info['kx']/w)**2-(info['ky']/w)**2
        q = np.sqrt(q2.astype(complex))
        np.testing.assert_allclose(info[name]['q'], q, atol=1e-14)
        np.testing.assert_array_equal(info[name]['propagating'], q2 > 0)
        np.testing.assert_allclose(info[name]['p_projection_gain'], n/abs(q), atol=1e-13)
        np.testing.assert_allclose(info[name]['cutoff_wavelength'], 2*np.pi*n/np.hypot(info['kx'], info['ky']))


def test_refinement_resolves_a_false_open_channel_without_changing_power_mask():
    band = .96*np.array([.998, 1.002])
    for resolution, resolved in ((20, False), (40, True)):
        results, d = _legacy_wavebit.fdtd(**options(period=.8, wavelength=band, theta=np.arcsin(.2),
                                            resolution=resolution, truncation_order=1, diffraction_distance=.2))
        info = d['diffraction']
        assert bool(info['cutoff_resolved']) is resolved
        expected = np.array([[[True, True, False]], [[False, True, False]]])
        for name in ('reflection', 'transmission'):
            np.testing.assert_array_equal(info[name]['propagating'], expected)
            assert bool(info[name]['yee_cutoff_parameter'][1, 0, 0] > 0) is not resolved
            assert info[name]['cutoff_wavelength'][0, 0] == pytest.approx(.96)
        # Closed continuum channels carry no outgoing power even if the mesh
        # disagrees. This is a classification test, not a scattering accuracy test.
        assert results[1].R[0, 0] == 0
        assert results[1].T[0, 0] == 0


@pytest.mark.parametrize('wavelength', [.8, np.nextafter(.8, np.inf)])
def test_grazing_is_rejected_before_time_stepping(monkeypatch, wavelength):
    import wavebit.fdtd._periodic.scene as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('grazing validation ran after the expensive solver')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='grazing'):
        _legacy_wavebit.fdtd(**options(period=.8, wavelength=wavelength, theta=0., truncation_order=1))


def test_plane_touching_a_mixed_interface_cell_is_refused():
    with pytest.raises(ValueError, match='diffraction_distance'):
        _legacy_wavebit.fdtd(**options(thickness=[.249], align_layers=False, diffraction_distance=(.2, .004)))


@pytest.mark.parametrize('distance', [0., -1., np.nan, True, [1.], [.1, 0.], [.1, .2, .3], 1e-6, 100.])
def test_invalid_or_unplaceable_distances_are_refused(distance):
    with pytest.raises(ValueError, match='diffraction_distance'):
        _legacy_wavebit.fdtd(**options(diffraction_distance=distance))
