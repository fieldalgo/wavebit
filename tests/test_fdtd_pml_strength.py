"""Static CPML strength, outgoing-pulse removal and backend sensitivities."""
import numpy as np
import pytest

import wavebit


@pytest.mark.parametrize('value', [0., 1., -.1, np.nan, np.inf, True, [.1], .1+0j, '0.1'])
def test_invalid_strength_is_refused_before_scene_time_stepping(monkeypatch, value):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid strength reached the time loop')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='pml_reflection'):
        wavebit.fdtd(structure=np.ones((1, 1, 1)), thickness=[.2], top=1., bot=1.,
                     period=1., wavelength=1., backend=0, resolution=10, pml_reflection=value)
    with pytest.raises(ValueError, match='pml_reflection'):
        wavebit.fdtd_grid(np.ones((8, 2, 2)), spacing=(1.,)*3, dt=.1, steps=2,
                          pml_reflection=value)


@pytest.mark.parametrize('nonuniform', [False, True])
def test_default_strength_and_explicit_uniform_profiles_preserve_fields(nonuniform):
    source = np.zeros((3, 24, 2, 2)); source[0, 8] = 1
    kw = dict(spacing=(.2,)*3, dt=.05, steps=100, pml_cells=5, source=source,
              pulse=np.exp(-((np.arange(100)*.05-1.)/.3)**2), wavelengths=[1.], monitor_z=[15])
    if nonuniform:
        kw['z_spacing'] = np.full(24, .2)
    omitted = wavebit.fdtd_grid(np.ones((24, 2, 2)), **kw)
    explicit = wavebit.fdtd_grid(np.ones((24, 2, 2)), **kw, pml_reflection=1e-8)
    other_profile = dict(kw)
    if nonuniform:
        other_profile.pop('z_spacing')
    else:
        other_profile['z_spacing'] = np.full(24, .2)
    uniform = wavebit.fdtd_grid(np.ones((24, 2, 2)), **other_profile, pml_reflection=1e-8)
    for key in ('E', 'H', 'electric_spectrum', 'magnetic_spectrum'):
        np.testing.assert_array_equal(omitted[key], explicit[key])
        np.testing.assert_allclose(omitted[key], uniform[key], atol=1e-14)
    assert explicit['pml_reflection'] == 1e-8
    strong = wavebit.fdtd_grid(np.ones((24, 2, 2)), **kw, pml_reflection=1e-30)
    strong_uniform = wavebit.fdtd_grid(np.ones((24, 2, 2)), **other_profile, pml_reflection=1e-30)
    for key in ('E', 'H', 'electric_spectrum', 'magnetic_spectrum'):
        np.testing.assert_allclose(strong[key], strong_uniform[key], atol=1e-14)


@pytest.mark.parametrize('nonuniform', [False, True])
def test_custom_strength_removes_a_pulse_in_a_magnetic_background(nonuniform):
    # A closed periodic box retains the pulse. Resolved homogeneous CPML must
    # remove it; the deliberately weak profile is an independent return control.
    shape = (120, 2, 2)
    source = np.zeros((3,)+shape); source[1, 60] = 1
    dt, steps = .04, 1200
    time = (np.arange(steps)+.5)*dt
    kw = dict(spacing=(.1,)*3, dt=dt, steps=steps, source=source,
              pulse=np.exp(-((time-5)/1.5)**2)*np.exp(-2j*np.pi*time/1.8),
              mu=1.44, pml_background=((2.25, 1.44),)*2, pml_wavelength=1.8)
    if nonuniform:
        widths = np.full(120, .1)
        widths[30:90] += .005*np.sin(np.arange(60)*2*np.pi/60)
        kw['z_spacing'] = widths
    epsilon = np.full(shape, 2.25)
    periodic = wavebit.fdtd_grid(epsilon, **kw)
    weak = wavebit.fdtd_grid(epsilon, **kw, pml_cells=20, pml_reflection=.9)
    strong = wavebit.fdtd_grid(epsilon, **kw, pml_cells=20, pml_reflection=1e-20)
    weights = kw.get('z_spacing', np.full(120, .1))[:, None, None]
    def energy(result):
        return np.sum(weights*(2.25*abs(result['E'])**2+1.44*abs(result['H'])**2))
    assert energy(strong) < 1e-5*energy(periodic)
    assert energy(strong) < .01*energy(weak)


@pytest.mark.parametrize('component', [1, 2])
def test_z_cpml_does_not_absorb_an_exact_parallel_yee_mode(component):
    # A z-independent mode has no z derivative to stretch. Neither strength
    # nor thickness removes this exact mode of the periodic native grid.
    shape, dx, dt, steps, kx = (20, 2, 4), .1, .04, 71, .7
    omega = 2*np.arcsin(dt*np.sin(kx*dx/2)/dx)/dt
    e = np.zeros((3,)+shape, complex); h = np.zeros_like(e)
    e[component] = np.exp(1j*kx*dx*np.arange(4))
    h[3-component] = (3-2*component)*np.exp(1j*(kx*dx*(np.arange(4)+.5)+omega*dt/2))
    for target in (1e-8, 1e-40):
        out = wavebit.fdtd_grid(np.ones(shape), spacing=(dx,)*3, dt=dt, steps=steps,
                                electric=e, magnetic=h, bloch_phase=(0., 0., kx*dx*4),
                                pml_cells=5, pml_reflection=target)
        for key, initial in (('E', e), ('H', h)):
            np.testing.assert_allclose(out[key], initial*np.exp(-1j*omega*steps*dt), atol=2e-14)


@pytest.mark.parametrize('backend', [1, 2])
def test_nonuniform_custom_cpml_preserves_material_and_pole_gradients(backend):
    shape = (20, 2, 3)
    mask = np.zeros(shape); mask[8:12] = 1
    source = np.zeros((3,)+shape); source[1, 6] = 1
    kw = dict(spacing=(.2, .5, .5), z_spacing=np.r_[.3, .28, .26, .24, np.full(12, .2), .24, .26, .28, .3],
              dt=.05, steps=120, source=source, pulse=np.exp(-((np.arange(120)*.05-1.)/.4)**2),
              wavelengths=[2.], monitor_z=[14], pml_cells=4, pml_reflection=1e-30,
              record_field=17, record_components=['ey'], checkpoint_steps=11)
    def objective(p, which):
        m = mask if which == 0 else native_mask
        out = wavebit.fdtd_grid(1+p[0]*m, mu=1+p[1]*m, poles=[(2., .3, p[2]*m)], backend=which, **kw)
        return out['power_z'].sum()+.1*(abs(out['fields']['ey']['values'])**2).sum()
    p = np.array([.3, .1, .5])
    expected = objective(p, 0)
    delta = 1e-5
    finite = np.array([(objective(p+v, 0)-objective(p-v, 0))/(2*delta) for v in np.eye(3)*delta])
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        native_mask = jnp.asarray(mask)
        value, derivative = jax.jit(jax.value_and_grad(lambda v: objective(v, 1)))(p)
    else:
        torch = pytest.importorskip('torch')
        native_mask = torch.as_tensor(mask, dtype=torch.float64)
        parameters = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        value = objective(parameters, 2)
        derivative, = torch.autograd.grad(value, parameters)
        value, derivative = value.detach().numpy(), derivative.detach().numpy()
    np.testing.assert_allclose(value, expected, atol=1e-12)
    np.testing.assert_allclose(derivative, finite, rtol=2e-6, atol=1e-10)
    assert np.min(abs(finite)) > 1e-6


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_scene_forwards_strength_to_both_calibration_bases_and_backgrounds(monkeypatch, side):
    import wavebit._core.scene_fdtd as scene
    run = scene.fdtd_grid
    seen = []
    def capture(*args, **kwargs):
        seen.append((kwargs['pml_reflection'], kwargs['pml_background']))
        return run(*args, **kwargs)
    monkeypatch.setattr(scene, 'fdtd_grid', capture)
    result, diagnostics = wavebit.fdtd(
        structure=np.array([[[2.]]]), thickness=[.2], top=1., bot=1.2, period=1., wavelength=1.,
        backend=0, resolution=10, n_pml=4, n_periods=25, pml_reflection=1e-24,
        incidence_side=side, diffraction_method='directional', truncation_order=0, return_diagnostics=True)
    assert len(seen) == 4
    assert all(target == 1e-24 for target, _ in seen)
    assert seen[:2] == seen[2:]
    reference, actual = np.asarray(seen[0][1]), np.asarray(seen[1][1])
    np.testing.assert_array_equal(reference[0], reference[1])
    np.testing.assert_array_equal(reference[0], actual[0])
    assert not np.array_equal(actual[0], actual[1])
    assert diagnostics['pml_reflection'] == 1e-24
    assert diagnostics['incidence_side'] == side
    assert np.isfinite(result.given.r_s).all() and np.isfinite(result.given.t_p).all()
