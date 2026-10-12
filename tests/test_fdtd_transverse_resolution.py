"""Independent transverse sampling: optics, coordinate symmetry and material graphs."""
import numpy as np
import pytest

import wavebit


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=.2, wavelength=1., backend='numpy', resolution=10, n_pml=4,
                     n_periods=30, truncation_order=0, tetm='tm', theta=.25,
                     align_layers=True, return_diagnostics=True), **changes)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, True, 1j, '20', [20.],
                                   [20., 0.], [20., np.inf], [20., 30., 40.], [[20., 30.]]])
def test_invalid_transverse_resolution_is_refused_before_time_stepping(monkeypatch, value):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid mesh request reached time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='transverse_resolution'):
        wavebit.fdtd(**options(transverse_resolution=value))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_explicit_original_sampling_preserves_default_results_exactly(side):
    old, a = wavebit.fdtd(**options(incidence_side=side))
    same, b = wavebit.fdtd(**options(incidence_side=side, transverse_resolution=[10., 10.]))
    for basis in ('incident', 's_inc', 'p_inc'):
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_array_equal(getattr(getattr(old, basis), key), getattr(getattr(same, basis), key))
    for key in ('shape', 'spacing', 'dt', 'z_nodes', 'z_spacing', 'pml_bounds', 'source_z'):
        np.testing.assert_array_equal(a[key], b[key])


def test_rectangular_sampling_keeps_z_geometry_and_recomputes_cfl():
    kw = options(structure=np.full((1, 2, 3), 2.25), thickness=[.213], period=[.41, .37],
                 tetm=None, phi=.3, polarization=(1., .4j), resolution=20, grading=True,
                 source_distance=1., diffraction_distance=.2)
    _, old = wavebit.fdtd(**kw)
    _, new = wavebit.fdtd(**dict(kw, transverse_resolution=(30., 10.)))
    assert new['shape'][1:] == (4, 13)
    assert new['transverse_resolution'] == (30., 10.)
    dx, dy = .41/13, .37/4
    np.testing.assert_allclose(new['spacing'][1:], [dy, dx], atol=1e-15)
    assert new['dt'] == pytest.approx(.8/np.sqrt(new['z_spacing'].min()**-2+dy**-2+dx**-2))
    assert new['dt'] != old['dt']
    for key in ('z_nodes', 'z_spacing', 'source_z', 'layer_planes', 'pml_bounds'):
        np.testing.assert_array_equal(old[key], new[key])
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        assert old['diffraction'][key] == new['diffraction'][key]


@pytest.mark.parametrize('explicit_y_period', [False, True])
def test_invariant_y_retains_two_cells_with_declared_or_implicit_period(explicit_y_period):
    _, d = wavebit.fdtd(**options(period=[.2, .6] if explicit_y_period else .2,
                                  transverse_resolution=(20., 15.)))
    assert d['shape'][1:] == (2, 4)
    assert d['spacing'][1] == pytest.approx(.3 if explicit_y_period else 1/15)


def test_coarse_transverse_sampling_cannot_hide_a_requested_nyquist_violation(monkeypatch):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('undersampled diffraction reached time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='spatial Nyquist'):
        wavebit.fdtd(**options(truncation_order=1, transverse_resolution=(5., 10.)))


def test_normal_then_tangential_means_match_independent_series_parallel_values(monkeypatch):
    import wavebit._core.scene_fdtd as scene
    captured, calls = {}, 0
    class Captured(Exception):
        pass
    def capture(epsilon, **settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {}
        captured.update(epsilon=np.asarray(epsilon), mu=np.asarray(settings['mu']))
        raise Captured()
    monkeypatch.setattr(scene, 'fdtd_grid', capture)
    pattern = np.array([[[1., 4.], [9., 16.]]])
    with pytest.raises(Captured):
        wavebit.fdtd(**options(structure=pattern, mu=pattern, thickness=[.3],
                                period=[.2, .2], transverse_resolution=15.))
    # Both transverse overlaps are one half. Average each series pair in y
    # first, then the two parallel columns: ((1||9) + (4||16))/2 = 4.1.
    expected = .5*(2/(1/1+1/9)+2/(1/4+1/16))
    assert expected == pytest.approx(4.1)
    # Ey is nodal in x and half-shifted in y; Hy has the opposite staggering.
    np.testing.assert_allclose(captured['epsilon'][1, :, 1, 0].max(), expected, atol=1e-13)
    np.testing.assert_allclose(captured['mu'][1, :, 0, 1].max(), expected, atol=1e-13)


@pytest.mark.parametrize('subpixel', [False, True])
@pytest.mark.parametrize('separate', [False, True])
def test_axis_exchange_rotates_the_same_discrete_maxwell_problem(subpixel, separate):
    structure = np.array([[[1.3, 2.2, 1.6], [2.5, 1.7, 1.1]]])
    mu = np.array([[[1.1, 1., 1.2], [1., 1.3, 1.1]]])
    kw = options(structure=structure, mu=mu, period=[.4, .3], tetm=None,
                 polarization=(1., .7j), phi=.31, transverse_resolution=(12., 13.) if separate else None,
                 diffraction_method='directional', incidence_side='bottom',
                 source_distance=1., diffraction_distance=.3, subpixel=subpixel)
    original, a = wavebit.fdtd(**kw)
    exchanged, b = wavebit.fdtd(**dict(kw, structure=structure.transpose(0, 2, 1),
                                       mu=mu.transpose(0, 2, 1), period=[.3, .4],
                                       transverse_resolution=(13., 12.) if separate else None, phi=np.pi/2-.31))
    assert a['shape'][1:] == b['shape'][:0:-1]
    assert a['dt'] == pytest.approx(b['dt'], abs=1e-15)
    # An x/y exchange reverses handedness. The s basis changes sign, whereas
    # p does not; diagonal Jones entries stay and off-diagonal entries negate.
    for basis, incident in (('s_inc', 's'), ('p_inc', 'p')):
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            sign = 1 if key[-1] == incident else -1
            np.testing.assert_allclose(getattr(getattr(original, basis), key),
                                       sign*getattr(getattr(exchanged, basis), key), atol=1e-11)


def test_unequal_axis_refinement_converges_to_conical_magnetic_airy():
    band, theta, epsilon, mu = np.array([.98, 1.02]), .3, 2.25, 1.1
    q0 = np.sqrt(1-(band*np.sin(theta))**2)
    q1 = np.sqrt(epsilon*mu-(band*np.sin(theta))**2)
    phase = np.exp(2j*np.pi*q1*.2/band)
    expected = {}
    for pol, admittance in (('s', mu), ('p', epsilon)):
        r = (admittance*q0-q1)/(admittance*q0+q1)
        expected[pol] = np.array([r*(1-phase**2)/(1-r*r*phase**2),
                                  (1-r*r)*phase/(1-r*r*phase**2)]).T
    errors = []
    for resolution in (40, 80):
        results, _ = wavebit.fdtd(**options(
            structure=np.full((1, 1, 1), epsilon), mu=np.full((1, 1, 1), mu),
            wavelength=band, period=.05, theta=theta, tetm=None, phi=.4, polarization=(1., 1j),
            resolution=resolution, transverse_resolution=(1.5*resolution, .75*resolution),
            n_pml=resolution, n_periods=60, pulse_width=3., diffraction_method='directional'))
        error = 0.
        for pol, basis in (('s', 's_inc'), ('p', 'p_inc')):
            actual = np.array([[getattr(getattr(result, basis), key+'_'+pol).item()
                                for key in ('r', 't')] for result in results])
            error = max(error, np.max(abs(actual-expected[pol])))
        errors.append(error)
    assert errors[1] < .004 and errors[1] < .4*errors[0]


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_anisotropic_scene_preserves_material_pole_and_record_gradients(backend):
    pattern = np.array([[[1., 1.2], [1.4, 1.1]]])
    def objective(p, which):
        from wavebit.backends.time_domain import time_loop_backend
        array = time_loop_backend(which).array
        result, d = wavebit.fdtd(**options(
            structure=p[0]*array(pattern), mu=p[1]*array(np.ones_like(pattern)), period=[.2, .2],
            dispersion=np.ones_like(pattern, dtype=int), poles={1: [(8., .3, p[2])]},
            backend=which, transverse_resolution=(20., 15.), tetm=None, phi=.4,
            polarization=(1., .6j), pulse_width=1.5, pulse_notches=.7,
            incidence_side='bottom', source_distance=1., diffraction_distance=.3,
            diffraction_method='directional', convergence_window=100,
            record_field=103, record_components=['ex'], record_bounds={'z': (.05, .15)}))
        r = result.incident
        return ((.2*r.r_s+.3j*r.r_p+.4*r.t_s+.5j*r.t_p).real.sum()
                +.1*(abs(d['fields']['ex']['values'])**2).mean()
                +.3*(d['convergence']['scattering_change_bound']['r_p']**2).sum())
    p, step = np.array([1.8, 1.05, .8]), 1e-4
    expected = objective(p, 'numpy')
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
    np.testing.assert_allclose(value, expected, atol=1e-10)
    np.testing.assert_allclose(gradient, finite, rtol=3e-5, atol=1e-8)
    assert np.min(abs(finite)) > 1e-5


def test_crossed_magnetic_pattern_refines_against_independent_rcwa():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    physical = dict(structure=np.array([[[1.3, 2.2, 1.6], [2.5, 1.7, 1.1]]]),
                    mu=np.array([[[1.1, 1., 1.2], [1., 1.3, 1.1]]]),
                    thickness=[.2], top=1., bot=1., period=[.4, .3], wavelength=1.,
                    theta=.25, phi=.31, polarization=(1., .7j), incidence_side='bottom')
    def central(result):
        return np.array([np.asarray(getattr(getattr(result, basis), key)).ravel()[result.incident.R.size//2]
                         for basis in ('s_inc', 'p_inc') for key in ('r_s', 'r_p', 't_s', 't_p')])
    references = [central(wavebit.rcwa(backend='numpy', truncation_order=dict(x=order, y=order), **physical).solve())
                  for order in (3, 5)]
    assert np.max(abs(references[1]-references[0])) < .001
    errors = []
    for resolution in (20, 40):
        result = wavebit.fdtd(**physical, backend='jax', truncation_order=0, resolution=resolution,
                              transverse_resolution=(1.5*resolution, resolution),
                              n_pml=round(.4*resolution), n_periods=60, pulse_width=3.,
                              align_layers=True, diffraction_method='directional',
                              diffraction_distance=.3, source_distance=1.)
        errors.append(np.max(abs(central(result)-references[1])))
    assert errors[1] < .008 and errors[1] < .6*errors[0]
