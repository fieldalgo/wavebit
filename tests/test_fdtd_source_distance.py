"""Independent current-sheet position, reference phase and material graphs."""
import numpy as np
import pytest

import wavebit


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.213], top=1., bot=1.,
                     period=.1, wavelength=[.9, 1., 1.1], backend='numpy', resolution=20,
                     n_periods=50, n_pml=6, truncation_order=0, tetm='tm', theta=.25,
                     pulse_width=3., align_layers=True, diffraction_method='directional',
                     diffraction_distance=.2, return_diagnostics=True), **changes)


@pytest.mark.parametrize('value', [0., -.1, np.nan, np.inf, True, 1j, '1', [1.], .05, 100.])
def test_bad_source_position_fails_before_time_stepping(monkeypatch, value):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid source position reached time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='source_distance'):
        wavebit.fdtd(**options(source_distance=value))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_explicit_default_source_is_bit_identical_and_mesh_is_unchanged(side):
    old, a = wavebit.fdtd(**options(incidence_side=side))
    physical = abs(a['source_z']-a['reflection_z'])
    same, b = wavebit.fdtd(**options(incidence_side=side, source_distance=physical))
    moved, c = wavebit.fdtd(**options(incidence_side=side, source_distance=1.))
    for before, after in zip(old, same):
        for basis in ('incident', 's_inc', 'p_inc'):
            for key in ('r_s', 'r_p', 't_s', 't_p'):
                np.testing.assert_array_equal(getattr(getattr(before, basis), key), getattr(getattr(after, basis), key))
    for key in ('z_nodes', 'z_spacing', 'shape', 'pml_bounds'):
        np.testing.assert_array_equal(a[key], c[key])
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        assert a['diffraction'][key] == c['diffraction'][key]
    assert c['source_z'] == pytest.approx(1.213 if side == 'top' else -1.)
    assert b['source_z'] == a['source_z'] and c['source_z'] != a['source_z']
    for before, after in zip(old, moved):
        for basis in ('s_inc', 'p_inc'):
            for key in ('r_s', 'r_p', 't_s', 't_p'):
                np.testing.assert_allclose(getattr(getattr(before, basis), key), getattr(getattr(after, basis), key), atol=1e-7)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_fixed_source_and_monitors_stay_put_when_padding_changes(side):
    _, a = wavebit.fdtd(**options(source_distance=1., incidence_side=side, pad_scale=1.))
    _, b = wavebit.fdtd(**options(source_distance=1., incidence_side=side, pad_scale=2.))
    assert a['source_z'] == pytest.approx(b['source_z'], abs=1e-13)
    assert a['shape'] != b['shape']
    assert not np.allclose(a['pml_bounds'], b['pml_bounds'])
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        assert a['diffraction'][key] == pytest.approx(b['diffraction'][key], abs=1e-13)


def test_graded_source_uses_nearest_node_and_avoids_user_monitor(monkeypatch):
    import wavebit._core.scene_fdtd as scene
    original = scene.fdtd_grid
    calls = []
    def capture(*args, **kwargs):
        index = np.flatnonzero(np.any(abs(kwargs['source']) > 0, axis=(0, 2, 3)))
        calls.append(index)
        return original(*args, **kwargs)
    monkeypatch.setattr(scene, 'fdtd_grid', capture)
    _, d = wavebit.fdtd(**options(source_distance=1.013, grading=True))
    expected = int(np.argmin(abs(np.asarray(d['z_nodes'])-(.213+1.013))))
    assert len(calls) == 4
    # The private time-stepper observes the reflected incident-first grid.
    internal_index = len(d['z_nodes'])-1-expected
    for index in calls:
        np.testing.assert_array_equal(index, [internal_index])
    assert d['source_z'] == d['z_nodes'][expected]
    with pytest.raises(ValueError, match='current sheet'):
        wavebit.fdtd(**options(source_distance=1.013, grading=True, field_monitors={'source': d['source_z']}))


@pytest.mark.slow
def test_moved_source_refines_to_independent_oblique_airy():
    band = np.array([.9, 1., 1.1])
    q0 = np.sqrt(1-(band*np.sin(.25))**2)
    q1 = np.sqrt(2.25-(band*np.sin(.25))**2)
    r = (2.25*q0-q1)/(2.25*q0+q1)
    phase = np.exp(2j*np.pi*q1*.213/band)
    expected = np.array([r*(1-phase**2)/(1-r*r*phase**2), (1-r*r)*phase/(1-r*r*phase**2)])
    errors = []
    for resolution in (40, 80):
        results, _ = wavebit.fdtd(**options(source_distance=.75, resolution=resolution,
                                            n_pml=round(.3*resolution), pulse_notches=.8))
        actual = np.array([[getattr(r.incident, key).item() for r in results] for key in ('r_p', 't_p')])
        errors.append(np.max(abs(actual-expected)))
    assert errors[1] < .003 and errors[1] < .4*errors[0]


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_position_control_preserves_bottom_material_and_record_gradients(backend):
    def objective(p, which):
        result, d = wavebit.fdtd(**options(
            structure=p[:1].reshape(1, 1, 1), mu=p[1:2].reshape(1, 1, 1), wavelength=1.,
            dispersion=np.ones((1, 1, 1), int), poles={1: [(8., .3, p[2])]}, backend=which,
            source_distance=1., pulse_notches=.8, grading=True, incidence_side='bottom',
            resolution=10, n_pml=4, n_periods=40, pulse_width=2.,
            tetm=None, phi=.4, polarization=(1., .6j), convergence_window=100,
            record_field=101, record_components=['ex'], record_bounds={'z': (.05, .15)}))
        r = result.incident
        return ((.2*r.r_s+.3j*r.r_p+.4*r.t_s+.5j*r.t_p).real.sum()
                +.1*(abs(d['fields']['ex']['values'])**2).mean()
                +.3*(d['convergence']['scattering_change_bound']['r_p']**2).sum())
    p = np.array([2.25, 1.05, 1.])
    expected = objective(p, 'numpy')
    step = 1e-4
    finite = np.array([(objective(p+v, 'numpy')-objective(p-v, 'numpy'))/(2*step) for v in np.eye(3)*step])
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
        value, gradient = jax.jit(jax.value_and_grad(lambda v: objective(v, 'jax')))(p)
    else:
        torch = pytest.importorskip('torch')
        parameter = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        value = objective(parameter, 'torch')
        gradient, = torch.autograd.grad(value, parameter)
        value, gradient = value.detach().numpy(), gradient.detach().numpy()
    np.testing.assert_allclose(value, expected, atol=1e-11)
    np.testing.assert_allclose(gradient, finite, rtol=3e-5, atol=1e-8)
    assert np.min(abs(finite)) > 1e-5


@pytest.mark.parametrize('distance', [.8, 1.])
def test_source_must_precede_the_default_reflection_plane(monkeypatch, distance):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid source crossed the default reflection plane')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='source_distance'):
        wavebit.fdtd(**options(source_distance=distance, diffraction_distance=None))


def test_bottom_magnetic_conical_phase_is_independent_of_source_position():
    common = dict(structure=np.full((1, 1, 1), 2.25), mu=np.full((1, 1, 1), 1.15),
                  thickness=[.213], top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(2., 1.3),
                  period=.1, wavelength=1., theta=.28, phi=.41, polarization=(1., .6j),
                  incidence_side='bottom', truncation_order=0)
    reference = wavebit.rcwa(backend='numpy', **common).solve().incident
    results = []
    for distance in (1., 1.3):
        result = wavebit.fdtd(backend='numpy', **common, resolution=40, n_pml=12,
                              pulse_width=3., n_periods=60, align_layers=True,
                              diffraction_method='directional', diffraction_distance=.2,
                              source_distance=distance).incident
        results.append(result)
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            np.testing.assert_allclose(getattr(result, key), getattr(reference, key), atol=.005)
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(getattr(results[0], key), getattr(results[1], key), atol=1e-7)
