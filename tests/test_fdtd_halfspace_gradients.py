"""Live exterior materials: closed-form CPML, fixed geometry and optical graphs."""
import numpy as np
import pytest

import wavebit
from wavebit.backends.time_domain import time_loop_backend

from test_fdtd_scene_angles import value_and_jacobian


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2],
                     top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(1.6, 1.15),
                     period=.2, wavelength=1., resolution=10, n_pml=6, n_periods=35,
                     backend=0, truncation_order=0, theta=.3, phi=.4,
                     polarization=(1., .6j), align_layers=True, mesh_indices=(1.1, 1.4)), **changes)


@pytest.mark.parametrize('value', [1., [1.], [1., 0.], [1., -1.], [1., np.nan],
                                       [1., np.inf], [1., 1j], [True, False], [[1., 1.]]])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_invalid_mesh_indices_are_refused_before_stepping(monkeypatch, value, side):
    import wavebit._core.scene_fdtd as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='mesh_indices'):
        wavebit.fdtd(**options(mesh_indices=value, incidence_side=side))


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, .1j, [1.], True])
def test_invalid_live_material_is_refused_before_stepping(monkeypatch, value):
    import wavebit._core.scene_fdtd as scene
    monkeypatch.setattr(scene, 'fdtd_grid', lambda *a, **kw: pytest.fail('reached time stepping'))
    with pytest.raises(ValueError, match='Half spaces'):
        wavebit.fdtd(**options(top=wavebit.Material(np.asarray(value), 1.)))


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_array_half_spaces_require_both_static_mesh_and_order_window(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    runtime = time_loop_backend(backend)
    for missing in ('mesh_indices', 'truncation_order'):
        with pytest.raises(ValueError, match='explicit mesh_indices and explicit truncation_order'):
            wavebit.fdtd(**options(top=runtime.array(1.2), backend=backend, **{missing: None}))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_fixed_indices_preserve_geometry_and_default_scalar_results(side):
    kw = options(incidence_side=side, grading=True, return_diagnostics=True)
    runs = [wavebit.fdtd(**dict(kw, top=wavebit.Material(np.array(e), np.array(m))))
            for e, m in ((1.2, 1.1), (2., 1.3))]
    first, second = (d for _, d in runs)
    for key in ('shape', 'spacing', 'dt', 'steps', 'z_nodes', 'z_spacing', 'source_z', 'pml_bounds', 'mesh_indices'):
        np.testing.assert_array_equal(first[key], second[key])
    for key in ('order_x', 'order_y', 'reflection_monitor_z', 'transmission_monitor_z'):
        np.testing.assert_array_equal(first['diffraction'][key], second['diffraction'][key])
    assert first['mesh_indices'] == (1.1, 1.4)
    np.testing.assert_allclose(second['background'], [[2., 1.3], [1.6, 1.15]])
    default, d = wavebit.fdtd(**dict(kw, mesh_indices=None))
    explicit, _ = wavebit.fdtd(**dict(kw, mesh_indices=d['mesh_indices']))
    for basis in ('given', 's_inc', 'p_inc'):
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_array_equal(getattr(getattr(default, basis), key),
                                           getattr(getattr(explicit, basis), key))


@pytest.mark.parametrize('backend', [1, 2])
def test_cpml_profile_jacobian_matches_closed_form_on_both_ends(backend):
    from wavebit._core.fdtd import cpml_profiles
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    nodes = np.array([0., .1, .25, .5, .8, 1.2, 1.5])
    halves = (nodes[1:]+nodes[:-1])/2
    p = np.array([1.2, 1.1, 2., 1.3])
    def objective(v):
        profiles = cpml_profiles(nodes, halves, .25, .3, .03, 1.2, v[0]*v[1], v[2]*v[3],
                                  target_reflection=1e-10, xp=xp, array=array)
        return xp.concat([profile[key] for profile in profiles for key in ('b', 'a', 'kappa')])
    value, derivative = value_and_jacobian(objective, p, backend)
    expected, gradients = [], []
    for positions in (nodes, halves):
        triples, jacobians = [], []
        for z in positions:
            side = 0 if .25-z >= z-1.2 else 2
            thickness, depth = (.25, .25-z) if side == 0 else (.3, z-1.2)
            if depth <= 0:
                triples.append([1., 0., 1.]); jacobians.append(np.zeros((3, 4)))
                continue
            f = min(depth/thickness, 1.)
            sigma = -5*np.log(1e-10)*f**4/(2*np.sqrt(p[side]*p[side+1])*thickness)
            kappa, alpha = 1+f**4, 3*np.pi/1.2*(1-f)
            b = np.exp(-.03*(sigma/kappa+alpha))
            denominator = kappa*(sigma+kappa*alpha)
            a = sigma*(b-1)/denominator
            jac = np.zeros((3, 4))
            for j in (side, side+1):
                ds = -sigma/(2*p[j])
                db = -.03*ds*b/kappa
                jac[0, j] = db
                jac[1, j] = ((ds*(b-1)+sigma*db)*denominator-sigma*(b-1)*kappa*ds)/denominator**2
            triples.append([b, a, kappa]); jacobians.append(jac)
        expected.extend(np.asarray(triples).T.ravel())
        gradients.extend(np.asarray(jacobians).transpose(1, 0, 2).reshape(-1, 4))
    np.testing.assert_allclose(value, expected, atol=2e-15)
    np.testing.assert_allclose(derivative, gradients, atol=2e-15)
    assert np.min(np.max(abs(np.asarray(gradients)), axis=0)) > .01


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('nonuniform', [False, True])
def test_native_exterior_gradients_include_cpml_and_recorded_fields(backend, nonuniform):
    shape = (24, 2, 3)
    low = np.zeros(shape); low[:12] = 1
    source = np.zeros((3,)+shape); source[1, 10] = 1.
    kw = dict(spacing=(.2, .4, .3), dt=.05, steps=150, source=source,
              pulse=np.exp(-((np.arange(150)*.05-1.2)/.5)**2), pml_cells=4,
              pml_reflection=1e-5, wavelengths=[2.], monitor_z=[9, 15],
              record_field=23, record_components=['ey', 'hx'], checkpoint_steps=13)
    if nonuniform:
        kw['z_spacing'] = np.r_[.26, .24, .22, np.full(18, .2), .23, .26, .29]
    def objective(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        mask = array(low)
        out = wavebit.fdtd_grid(p[0]*mask+p[2]*(1-mask), mu=p[1]*mask+p[3]*(1-mask),
                                pml_background=((p[0], p[1]), (p[2], p[3])), backend=which, **kw)
        return xp.stack((out['power_z'].sum(), .1*xp.sum(xp.abs(out['fields']['ey']['values'])**2)
                          +.2*xp.sum(xp.abs(out['H'])**2)))
    p = np.array([1.2, 1.1, 1.6, 1.15])
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, objective(p, 0), atol=1e-12)
    for step in (2e-5, 1e-5):
        finite = np.stack([(objective(p+v, 0)-objective(p-v, 0))/(2*step) for v in np.eye(4)*step], axis=-1)
        np.testing.assert_allclose(derivative, finite, rtol=2e-6, atol=2e-9)
    assert np.min(np.max(abs(finite), axis=0)) > 1e-4


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('method', ['electric', 'directional'])
def test_joint_halfspace_angle_pole_graphs_through_bottom_scene(backend, method):
    pattern = np.array([[[1.2, 1.8], [1.5, 1.1]]])
    def objective(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        stack = wavebit.Stack.from_structure(array(pattern), [.2],
                                             top=wavebit.Material(p[0], p[1]),
                                             bot=wavebit.Material(p[2], p[3]), period=[.3, .4])
        incidence = wavebit.Incidence(wavelength=[.98, 1.02], theta=p[4], phi=.4,
                                     polarization=(1., .6j), incidence_side='bottom')
        results, d = wavebit.fdtd(stack, incidence, backend=which, mesh_indices=(1.1, 1.4),
                                 resolution=10, transverse_resolution=(12., 13.),
                                 n_pml=6, n_periods=35, grading=True, align_layers=True,
                                 pulse_width=1.5, pulse_notches=.7, source_distance=1., diffraction_distance=.3,
                                 dispersion=np.ones_like(pattern, int), poles={1: [(8., .3, p[5])]},
                                 diffraction_method=method, truncation_order=[1, 1],
                                 return_diagnostics=True, convergence_window=100,
                                 field_monitors={'inside': .1}, record_field=113,
                                 record_components=['ex'], record_bounds={'z': (.05, .15)})
        amplitude = sum(xp.real(.2*r.given.r_s+.3j*r.given.r_p+.4*r.given.t_s+.5j*r.given.t_p).sum()
                        +.1*r.given.T.sum() for r in results)
        return xp.stack((amplitude, .1*xp.mean(xp.abs(d['field_e'])**2)
                         +.2*xp.mean(xp.abs(d['monitors']['inside']['fields']['ey'])**2),
                         .3*xp.sum(d['convergence']['scattering_change_bound']['r_p']**2)))
    p = np.array([1.2, 1.1, 1.6, 1.15, .3, .8])
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, objective(p, 0), atol=2e-10)
    for step in (2e-4, 1e-4):
        finite = np.stack([(objective(p+v, 0)-objective(p-v, 0))/(2*step) for v in np.eye(6)*step], axis=-1)
        np.testing.assert_allclose(derivative, finite, rtol=6e-5, atol=4e-8)
    assert np.min(np.max(abs(finite), axis=0)) > 1e-6


def test_jit_keeps_one_mesh_and_masks_invalid_background_and_duration():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traced = []
    def solve(e):
        traced.append(None)
        r, d = wavebit.fdtd(**options(top=e, backend=1, return_diagnostics=True,
                                      field_monitors={'inside': .1}, record_field=151, record_components=['ey']))
        return r.given.T, d['stable'], d['z_nodes'], d['dt'], d['field_e'], d['monitors']['inside']['fields']['ey']
    compiled = jax.jit(solve)
    initial = compiled(np.array(1.2))
    changed = compiled(np.array(2.))
    assert initial[1] and changed[1]
    np.testing.assert_array_equal(initial[2], changed[2])
    np.testing.assert_array_equal(initial[3], changed[3])
    for e in (-1., np.nan, .001, 100.):
        result = compiled(np.array(e))
        assert not result[1]
        for i in (0, 4, 5):
            assert np.isnan(result[i]).all()
    assert len(traced) == 1
    with pytest.raises(ValueError, match='n_periods is too short'):
        wavebit.fdtd(**options(top=np.array(100.)))


@pytest.mark.parametrize('backend', [1, 2])
def test_native_background_validation_remains_active(backend):
    if backend:
        pytest.importorskip('jax' if backend == 1 else 'torch')
    array = time_loop_backend(backend).array
    kw = dict(spacing=(.2,)*3, dt=.05, steps=2, pml_cells=3, backend=backend)
    for background in ([[1., 1.], [-1., 1.]], [[1., 1.], [np.nan, 1.]], [[1., 1.], [2., 1.]]):
        with pytest.raises(ValueError, match='pml_background|PML background'):
            wavebit.fdtd_grid(array(np.ones((16, 2, 2))), pml_background=array(background), **kw)


def test_halfspace_cutoff_can_change_on_one_trace_but_exact_grazing_is_masked():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    critical = 4*np.sin(.5)**2
    def solve(epsilon):
        traces.append(None)
        r, d = wavebit.fdtd(**options(top=4., bot=epsilon, theta=.5, phi=0., courant=.5,
                                      mesh_indices=(2., 1.), backend=1, return_diagnostics=True))
        channel = d['diffraction']['transmission']
        return r.given.T, d['stable'], channel['q'], channel['propagating'], d['z_nodes']
    compiled = jax.jit(solve)
    below, above = compiled(np.array(.8)), compiled(np.array(1.1))
    for epsilon, result in ((.8, below), (1.1, above)):
        assert result[1]
        np.testing.assert_allclose(result[2], np.sqrt(epsilon-critical+0j), atol=1e-14)
        assert result[3].item() == (epsilon > critical)
        assert np.isfinite(result[0]).all()
    np.testing.assert_array_equal(below[4], above[4])
    grazing = compiled(np.array(critical))
    assert not grazing[1]
    assert np.isnan(grazing[0]).all()
    assert len(traces) == 1
    with pytest.raises(ValueError, match='exactly grazing'):
        wavebit.fdtd(**options(top=4., bot=np.array(critical), theta=.5, phi=0., courant=.5,
                               mesh_indices=(2., 1.)))


def test_magnetic_halfspace_sensitivities_refine_to_independent_airy(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    p = np.array([1.2, 1.1, 1.6, 1.15])
    band, depth, epsilon, mu = np.array([.98, 1.02]), .2, 2.25, 1.12
    def airy(v):
        y0, y1, y2 = np.sqrt(v[0]/v[1]), np.sqrt(epsilon/mu), np.sqrt(v[2]/v[3])
        a, b = (y0-y1)/(y0+y1), (y1-y2)/(y1+y2)
        phase = np.exp(2j*np.pi*np.sqrt(epsilon*mu)*depth/band)
        r = (a+b*phase**2)/(1+a*b*phase**2)
        t = (1+a)*(1+b)*phase/(1+a*b*phase**2)
        return np.r_[r.real, r.imag, t.real, t.imag, abs(r)**2, y2/y0*abs(t)**2]
    expected = airy(p)
    step = 1e-5
    target = np.stack([(airy(p+v)-airy(p-v))/(2*step) for v in np.eye(4)*step], axis=-1)
    smaller = np.stack([(airy(p+v/2)-airy(p-v/2))/step for v in np.eye(4)*step], axis=-1)
    np.testing.assert_allclose(target, smaller, atol=1e-9)
    errors = []
    for resolution in (40, 80):
        def objective(v):
            results = wavebit.fdtd(**options(top=wavebit.Material(v[0], v[1]),
                                             bot=wavebit.Material(v[2], v[3]), theta=0., phi=0.,
                                             mu=np.full((1, 1, 1), mu), backend=1, wavelength=band,
                                             period=.05, resolution=resolution, n_pml=resolution,
                                             n_periods=65, pulse_width=3., diffraction_method='directional'))
            import jax.numpy as jnp
            r = jnp.stack([s.s_inc.r_s.reshape(()) for s in results])
            t = jnp.stack([s.s_inc.t_s.reshape(()) for s in results])
            return jnp.concatenate((r.real, r.imag, t.real, t.imag,
                                    jnp.stack([s.s_inc.R.reshape(()) for s in results]),
                                    jnp.stack([s.s_inc.T.reshape(()) for s in results])))
        value, derivative = value_and_jacobian(objective, p, 1)
        errors.append(np.array([np.max(abs(value-expected)), np.max(abs(derivative-target))]))
    assert np.all(errors[1] < .005)
    assert np.all(errors[1] < .45*errors[0])
    for resolution, error in zip((40, 80), errors):
        record_property(f'value_error_{resolution}', float(error[0]))
        record_property(f'derivative_error_{resolution}', float(error[1]))
