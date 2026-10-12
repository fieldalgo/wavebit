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
                     period=1., wavelength=1., backend='numpy', resolution=10, pml_reflection=value)
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


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_nonuniform_custom_cpml_preserves_material_and_pole_gradients(backend):
    shape = (20, 2, 3)
    mask = np.zeros(shape); mask[8:12] = 1
    source = np.zeros((3,)+shape); source[1, 6] = 1
    kw = dict(spacing=(.2, .5, .5), z_spacing=np.r_[.3, .28, .26, .24, np.full(12, .2), .24, .26, .28, .3],
              dt=.05, steps=120, source=source, pulse=np.exp(-((np.arange(120)*.05-1.)/.4)**2),
              wavelengths=[2.], monitor_z=[14], pml_cells=4, pml_reflection=1e-30,
              record_field=17, record_components=['ey'], checkpoint_steps=11)
    def objective(p, which):
        m = mask if which == 'numpy' else native_mask
        out = wavebit.fdtd_grid(1+p[0]*m, mu=1+p[1]*m, poles=[(2., .3, p[2]*m)], backend=which, **kw)
        return out['power_z'].sum()+.1*(abs(out['fields']['ey']['values'])**2).sum()
    p = np.array([.3, .1, .5])
    expected = objective(p, 'numpy')
    delta = 1e-5
    finite = np.array([(objective(p+v, 'numpy')-objective(p-v, 'numpy'))/(2*delta) for v in np.eye(3)*delta])
    if backend == 'jax':
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        native_mask = jnp.asarray(mask)
        value, derivative = jax.jit(jax.value_and_grad(lambda v: objective(v, 'jax')))(p)
    else:
        torch = pytest.importorskip('torch')
        native_mask = torch.as_tensor(mask, dtype=torch.float64)
        parameters = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        value = objective(parameters, 'torch')
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
        backend='numpy', resolution=10, n_pml=4, n_periods=25, pml_reflection=1e-24,
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
    assert np.isfinite(result.incident.r_s).all() and np.isfinite(result.incident.t_p).all()


# The legacy route (``backend`` omitted) builds two CPMLs: its grid's, and the one-dimensional
# grid that carries the incident wave. A target that reaches only the first moves the answers by
# 2.25e-7 on the route-parity study's magnetic twin (measured with the incident grid left at the
# default), so no comparison of results can see it; these tests watch the profiles being built.
LEGACY_SCENE = dict(structure=np.full((1, 1, 1), 2.25), thickness=[.3], top=1., bot=1.44,
                    period=.5, wavelength=.9)


class _Built(Exception):
    """Raised once the incident grid exists: both CPMLs are built, nothing is stepped."""


@pytest.mark.parametrize('incidence', [dict(tetm='te'), dict(psi=.3)], ids=['tetm', 'psi'])
@pytest.mark.parametrize('value', [0., 1., -.1, np.nan, np.inf, True, [.1], .1+0j, '0.1', None])
def test_legacy_invalid_strength_is_refused_before_a_grid_is_built(monkeypatch, value, incidence):
    import wavebit._core.fdtd as legacy
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid strength reached the grid')
    monkeypatch.setattr(legacy, '_Grid', forbidden)
    with pytest.raises(ValueError, match='pml_reflection'):
        wavebit.fdtd(**LEGACY_SCENE, **incidence, pml_reflection=value)


@pytest.mark.parametrize('incidence', [
    dict(tetm='te'), dict(psi=.3), dict(psi=.3, theta=.2, phi=.3)],
    ids=['tetm', 'psi-normal', 'psi-conical'])
def test_legacy_strength_reaches_both_cpmls_on_every_route(monkeypatch, incidence):
    import wavebit._core.fdtd as legacy
    built = legacy._Incident1D.__init__

    def stop(self, *args, **kwargs):
        built(self, *args, **kwargs)
        raise _Built

    def profiles(target):
        profile, calls = legacy.cpml_profiles, []

        def spy(*args, **kwargs):
            out = profile(*args, **kwargs)
            calls.append((kwargs.get('target_reflection', 'missing'), out))
            return out
        with monkeypatch.context() as patch:
            patch.setattr(legacy, 'cpml_profiles', spy)
            patch.setattr(legacy._Incident1D, '__init__', stop)
            with pytest.raises(_Built):
                wavebit.fdtd(**LEGACY_SCENE, **incidence, n_pml=4,
                             **({} if target is None else dict(pml_reflection=target)))
        return calls

    default, strong = profiles(None), profiles(1e-16)
    # The grid's CPML, then the incident grid's -- the order `solve_fdtd` builds them in.
    assert [target for target, _ in default] == [legacy.PML_TARGET_REFLECTION]*2
    assert [target for target, _ in strong] == [1e-16]*2
    for (_, (node_default, _)), (_, (node_strong, _)) in zip(default, strong):
        assert not np.allclose(node_default['b'], node_strong['b']), 'the target changed nothing'


def test_legacy_default_strength_is_the_old_profile_and_is_recorded():
    import inspect

    import wavebit._core.fdtd as legacy
    assert (inspect.signature(legacy.solve_fdtd).parameters['pml_reflection'].default
            == legacy.PML_TARGET_REFLECTION == 1e-8)
    run = dict(LEGACY_SCENE, tetm='te', resolution=20., n_pml=4, return_diagnostics=True)
    omitted, omitted_diagnostics = wavebit.fdtd(**run)
    explicit, explicit_diagnostics = wavebit.fdtd(**run, pml_reflection=1e-8)
    strong, strong_diagnostics = wavebit.fdtd(**run, pml_reflection=1e-16)
    for name in ('R', 'T', 'r_s', 't_s'):
        np.testing.assert_array_equal(getattr(omitted.incident, name), getattr(explicit.incident, name))
    assert not np.array_equal(omitted.incident.r_s, strong.incident.r_s)
    assert omitted_diagnostics['pml_reflection'] == explicit_diagnostics['pml_reflection'] == 1e-8
    assert strong_diagnostics['pml_reflection'] == 1e-16


@pytest.mark.parametrize('incidence', [dict(tetm='te'), dict(psi=.3)], ids=['tetm', 'psi'])
@pytest.mark.parametrize('value', [0, 1, -3, 1.5, np.nan, np.inf, True, '12', None])
def test_legacy_absorber_thinner_than_two_cells_is_refused_before_a_grid_is_built(
        monkeypatch, value, incidence):
    # The explicit-backend path already refuses these; the legacy one took n_pml=0 and returned
    # T = 2e-4 on a slab RCWA puts at 0.885.
    import wavebit._core.fdtd as legacy
    def forbidden(*args, **kwargs):
        raise AssertionError('an invalid absorber depth reached the grid')
    monkeypatch.setattr(legacy, '_Grid', forbidden)
    with pytest.raises(ValueError, match='n_pml'):
        wavebit.fdtd(**LEGACY_SCENE, **incidence, n_pml=value)


@pytest.mark.parametrize('value', [2, 12.0, np.int64(24)])
def test_legacy_absorber_depth_takes_whole_numbers_from_two(monkeypatch, value):
    import wavebit._core.fdtd as legacy
    built, grid, grids = legacy._Incident1D.__init__, legacy._Grid, []

    def stop(self, *args, **kwargs):
        built(self, *args, **kwargs)
        raise _Built

    def capture(*args, **kwargs):
        grids.append(grid(*args, **kwargs))
        return grids[-1]
    monkeypatch.setattr(legacy._Incident1D, '__init__', stop)
    monkeypatch.setattr(legacy, '_Grid', capture)
    with pytest.raises(_Built):
        wavebit.fdtd(**LEGACY_SCENE, tetm='te', n_pml=value)
    assert grids and grids[-1].n_pml == int(value)
