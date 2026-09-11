"""Azimuth graphs with fixed y sampling, independent dispersion and rotations."""
import numpy as np
import pytest

import wavebit

from test_fdtd_scene_angles import value_and_jacobian


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=[.2, .5], wavelength=1., resolution=10, n_pml=6, n_periods=30,
                     backend=0, truncation_order=0, theta=.3, phi=.4,
                     polarization=(1., .6j), align_layers=True), **changes)


@pytest.mark.parametrize('phi', [np.nan, np.inf, -np.inf, .2j, True, [.2], [[.2]]])
def test_invalid_azimuth_is_refused_before_time_stepping(monkeypatch, phi):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid azimuth reached time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='phi'):
        wavebit.fdtd(**options(phi=phi))


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_array_azimuth_requires_explicit_window_and_jones_polarization(backend):
    from wavebit.backends.time_domain import time_loop_backend
    if backend == 1:
        pytest.importorskip('jax')
    elif backend == 2:
        pytest.importorskip('torch')
    phi = time_loop_backend(backend).array(0.)
    with pytest.raises(ValueError, match='explicit truncation_order'):
        wavebit.fdtd(**options(phi=phi, backend=backend, truncation_order=None))
    with pytest.raises(ValueError, match='psi or polarization'):
        wavebit.fdtd(**options(phi=phi, backend=backend, tetm='te', polarization=None))


@pytest.mark.parametrize('period', [.2, [.2, .5]])
def test_array_azimuth_keeps_y_cells_and_time_step_while_crossing_zero(period):
    runs = [wavebit.fdtd(**options(phi=np.array(phi), period=period, return_diagnostics=True))
            for phi in (-.2, 0., .2)]
    first = runs[0][1]
    assert first['shape'][1] == (5 if isinstance(period, list) else 2)
    for _, d in runs[1:]:
        for key in ('shape', 'spacing', 'dt', 'steps', 'z_nodes', 'z_spacing', 'source_z', 'pml_bounds'):
            np.testing.assert_array_equal(first[key], d[key])
        for key in ('order_x', 'order_y', 'reflection_monitor_z', 'transmission_monitor_z'):
            np.testing.assert_array_equal(first['diffraction'][key], d['diffraction'][key])
    _, static = wavebit.fdtd(**options(phi=0., period=period, return_diagnostics=True))
    assert static['shape'][1] == 2
    # A nonzero host angle already uses the full y period.
    original, _ = wavebit.fdtd(**options(phi=.2, period=period, return_diagnostics=True))
    for basis in ('given', 's_inc', 'p_inc'):
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_array_equal(getattr(getattr(original, basis), key),
                                           getattr(getattr(runs[-1][0], basis), key))


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_two_angle_jacobian_matches_independent_uniform_yee_phase(backend, side, record_property):
    from wavebit.backends.time_domain import time_loop_backend
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    runtime = time_loop_backend(backend)
    xp = runtime.xp
    band, theta, phi, depth = np.array([.98, 1.02]), .6, .6, .2
    kw = options(structure=np.ones((1, 1, 1)), wavelength=band, period=[.2, .3],
                 n_pml=10, n_periods=80, pulse_width=4., incidence_side=side,
                 transverse_resolution=(20., 5.), diffraction_method='directional')
    def objective(p):
        results, d = wavebit.fdtd(**dict(kw, theta=p[0], phi=p[1], backend=backend, return_diagnostics=True))
        values = xp.stack([getattr(getattr(r, basis), key).reshape(()) for r in results
                           for basis, key in (('s_inc', 't_s'), ('p_inc', 't_p'))])
        return xp.concat((xp.real(values), xp.imag(values), d['angles'], d['azimuth'].reshape(1)))
    value, derivative = value_and_jacobian(objective, np.array([theta, phi]), backend)
    _, d = wavebit.fdtd(**dict(kw, theta=theta, phi=np.array(phi), return_diagnostics=True))
    dz, dy, dx = d['spacing']
    dt, k0 = d['dt'], 2*np.pi/band
    kx, ky = 2*np.pi*np.sin(theta)*np.array([np.cos(phi), np.sin(phi)])
    dkx = np.array([2*np.pi*np.cos(theta)*np.cos(phi), -ky])
    dky = np.array([2*np.pi*np.cos(theta)*np.sin(phi), kx])
    x, y = 2*np.sin(kx*dx/2)/dx, 2*np.sin(ky*dy/2)/dy
    omega = 2*np.sin(k0*dt/2)/dt
    longitudinal = np.sqrt(omega**2-x*x-y*y)
    phase = 2*np.arcsin(dz*longitudinal/2)/dz
    dsquare = -2*x*np.cos(kx*dx/2)*dkx-2*y*np.cos(ky*dy/2)*dky
    dphase = dsquare/(2*longitudinal[:, None]*np.sqrt(1-(dz*longitudinal[:, None]/2)**2))
    expected = np.repeat(np.exp(1j*depth*phase), 2)
    gradient = np.repeat(1j*depth*dphase*np.exp(1j*depth*phase[:, None]), 2, axis=0)
    angles = np.arcsin(band*np.sin(theta))
    dangles = np.stack((band*np.cos(theta)/np.sqrt(1-(band*np.sin(theta))**2), np.zeros(2)), axis=1)
    target = np.r_[gradient.real, gradient.imag, dangles, [[0., 1.]]]
    np.testing.assert_allclose(value, np.r_[expected.real, expected.imag, angles, phi], atol=2e-6)
    np.testing.assert_allclose(derivative, target, atol=2e-5)
    assert np.max(abs(gradient[:, 1])) > 10*2e-5
    record_property('maximum_angle_jacobian_error', float(np.max(abs(derivative-target))))


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_normal_incidence_rotates_calibrated_monitors_and_records(backend, side):
    from wavebit.backends.time_domain import time_loop_backend
    def objective(phi, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        result, d = wavebit.fdtd(**options(phi=phi, theta=0., backend=which, incidence_side=side,
                                          polarization=(1., 0.), diffraction_method='directional',
                                          field_monitors={'inside': .1}, return_diagnostics=True,
                                          record_field=17, record_components=['ex', 'ey'],
                                          record_bounds={'z': (.05, .15)}))
        temporal = xp.exp(2j*np.pi*d['field_t'])
        values = xp.stack([xp.mean(d['monitors']['inside']['fields'][name]) for name in ('ex', 'ey')]
                          +[xp.mean(temporal[:, None, None, None]*d['fields'][name]['values']) for name in ('ex', 'ey')])
        return xp.concat((xp.real(values), xp.imag(values), array([1.])*result.given.T.reshape(1)))
    phi = .37
    base = objective(np.array(0.), 0)
    base_complex = base[:4]+1j*base[4:8]
    scales = base_complex[[1, 3]]
    expected = (scales[:, None]*np.array([-np.sin(phi), np.cos(phi)])).ravel()
    gradient = (scales[:, None]*np.array([-np.cos(phi), -np.sin(phi)])).ravel()
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), phi, backend)
    np.testing.assert_allclose(value, np.r_[expected.real, expected.imag, base[-1]], atol=2e-11)
    np.testing.assert_allclose(derivative, np.r_[gradient.real, gradient.imag, 0.], atol=2e-10)
    assert np.max(abs(gradient)) > .1


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('method', ['electric', 'directional'])
def test_azimuth_and_material_graphs_match_two_finite_difference_steps(backend, method):
    from wavebit.backends.time_domain import time_loop_backend
    pattern = np.array([[[1., 1.2], [1.4, 1.1]]])
    def objective(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        scene = wavebit.Stack.from_structure(p[2]*array(pattern), [.2], top=1., bot=1., period=[.3, .4])
        incidence = wavebit.Incidence(wavelength=[.98, 1.02], theta=p[0], phi=p[1],
                                     polarization=(1., .6j), incidence_side='bottom')
        results, d = wavebit.fdtd(scene, incidence, backend=which, resolution=10, n_pml=6, n_periods=30,
                                 mu=1.1*array(np.ones_like(pattern)), dispersion=np.ones_like(pattern, int),
                                 poles={1: [(8., .3, p[3])]}, truncation_order=[1, 1],
                                 transverse_resolution=(12., 13.), grading=True, align_layers=True,
                                 pulse_width=1.5, pulse_notches=.7, source_distance=1., diffraction_distance=.3,
                                 diffraction_method=method, convergence_window=100, return_diagnostics=True,
                                 field_monitors={'inside': .1}, record_field=103, record_components=['ex'],
                                 record_bounds={'z': (.05, .15)})
        amplitude = sum(xp.sum(xp.real(.2*r.given.r_s+.3j*r.given.r_p+.4*r.given.t_s+.5j*r.given.t_p))
                        +.1*r.given.T.sum() for r in results)
        return xp.stack((amplitude, .1*xp.mean(xp.abs(d['field_e'])**2)
                         +.2*xp.mean(xp.abs(d['monitors']['inside']['fields']['ey'])**2),
                         .3*xp.sum(d['convergence']['scattering_change_bound']['r_p']**2)))
    p = np.array([.3, .4, 1.8, .8])
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, objective(p, 0), atol=1e-10)
    for step in (2e-4, 1e-4):
        finite = np.stack([(objective(p+v, 0)-objective(p-v, 0))/(2*step)
                           for v in np.eye(4)*step], axis=-1)
        np.testing.assert_allclose(derivative, finite, rtol=5e-5, atol=3e-8)
        assert np.max(abs(finite[:, 1])) > 100*3e-8


def test_jitted_nonfinite_azimuth_masks_results_and_normalized_fields():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    def solve(phi):
        result, d = wavebit.fdtd(**options(phi=phi, backend=1, return_diagnostics=True,
                                          field_monitors={'inside': .1}, record_field=103,
                                          record_components=['ex'], record_bounds={'z': (.05, .15)}))
        return result.given.T, d['stable'], d['monitors']['inside']['fields']['ex'], d['field_e']
    compiled = jax.jit(solve)
    good = compiled(0.)
    assert bool(good[1]) and np.isfinite(good[0]).all()
    for phi in (np.nan, np.inf, -np.inf):
        result, stable, monitor, record = compiled(phi)
        assert not bool(stable)
        for value in (result, monitor, record):
            assert np.isnan(value).all()


def test_jitted_azimuth_crosses_zero_without_resizing_or_retracing():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    shapes = []
    kw = options(structure=np.array([[[1.3, 2.2, 1.6]]]), period=[.3, .5],
                 truncation_order=[1, 1], diffraction_method='directional', diffraction_distance=.3)
    def objective(phi, backend):
        result, d = wavebit.fdtd(**dict(kw, phi=phi, backend=backend, return_diagnostics=True))
        if backend == 1:
            shapes.append(d['shape'])
        return (.2*result.s_inc.r_p+.3j*result.p_inc.t_s).real.sum()
    compiled = jax.jit(jax.value_and_grad(lambda p: objective(p, 1)))
    for phi in (-.1, 0., .1):
        value, derivative = compiled(phi)
        step = 1e-4
        expected = objective(np.array(phi), 0)
        finite = (objective(np.array(phi+step), 0)-objective(np.array(phi-step), 0))/(2*step)
        np.testing.assert_allclose(value, expected, atol=1e-11)
        np.testing.assert_allclose(derivative, finite, rtol=3e-5, atol=1e-8)
    assert len(shapes) == 1 and shapes[0][1] == 5


def test_azimuth_dependent_grazing_is_invalid_in_eager_and_jit_calls():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    kw = options(period=[1., .5], truncation_order=1, diffraction_method='directional')
    phi = np.arccos(-np.sin(kw['theta'])/2)
    # For m=+1 at lambda=Lx, |k_parallel/k0|^2 = 1+s^2+2s*cos(phi).
    with pytest.raises(ValueError, match='grazing'):
        wavebit.fdtd(**dict(kw, phi=phi))
    def solve(phi):
        result, d = wavebit.fdtd(**dict(kw, phi=phi, backend=1, return_diagnostics=True))
        return result.given.T, d['stable']
    compiled = jax.jit(solve)
    assert bool(compiled(.4)[1])
    result, valid = compiled(phi)
    assert not bool(valid) and np.isnan(result).all()


@pytest.mark.parametrize('backend', [1, 2])
def test_azimuth_derivative_respects_axis_exchange(backend):
    from wavebit.backends.time_domain import time_loop_backend
    pattern = np.array([[[1.3, 2.2, 1.6], [2.5, 1.7, 1.1]]])
    mu = np.array([[[1.1, 1., 1.2], [1., 1.3, 1.1]]])
    def objective(phi, exchange):
        runtime = time_loop_backend(backend)
        xp = runtime.xp
        result = wavebit.fdtd(**options(
            structure=pattern.transpose(0, 2, 1) if exchange else pattern,
            mu=mu.transpose(0, 2, 1) if exchange else mu, phi=np.pi/2-phi if exchange else phi,
            period=[.3, .4] if exchange else [.4, .3], backend=backend,
            transverse_resolution=(13., 12.) if exchange else (12., 13.),
            incidence_side='bottom', diffraction_method='directional', diffraction_distance=.3))
        values = xp.stack([getattr(getattr(result, basis), key).reshape(())
                           *(1 if not exchange or key[-1] == incident else -1)
                           for basis, incident in (('s_inc', 's'), ('p_inc', 'p'))
                           for key in ('r_s', 'r_p', 't_s', 't_p')])
        return xp.concat((xp.real(values), xp.imag(values)))
    value, gradient = value_and_jacobian(lambda p: objective(p, False), .31, backend)
    exchanged, rotated = value_and_jacobian(lambda p: objective(p, True), .31, backend)
    np.testing.assert_allclose(exchanged, value, atol=1e-11)
    np.testing.assert_allclose(rotated, gradient, atol=1e-10)


def test_magnetic_airy_azimuth_invariance_is_recovered_under_refinement(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    theta, phi, epsilon, mu = .55, .41, 2.25, 1.1
    band = np.array([.98, 1.02])
    q0 = np.sqrt(1-(band*np.sin(theta))**2)
    q1 = np.sqrt(epsilon*mu-(band*np.sin(theta))**2)
    phase = np.exp(2j*np.pi*q1*.2/band)
    expected = 0.
    for coefficient in (mu, epsilon):
        r = (coefficient*q0-q1)/(coefficient*q0+q1)
        reflected = r*(1-phase**2)/(1-r*r*phase**2)
        transmitted = (1-r*r)*phase/(1-r*r*phase**2)
        expected += np.real((.2+.3j)*reflected+(-.4+.1j)*transmitted).sum()
    errors = []
    for resolution in (40, 80):
        def objective(azimuth):
            results = wavebit.fdtd(**options(
                theta=theta, phi=azimuth, structure=np.full((1, 1, 1), epsilon),
                mu=np.full((1, 1, 1), mu), wavelength=band, period=.05,
                backend=1, resolution=resolution, n_pml=resolution, n_periods=60, pulse_width=3.,
                transverse_resolution=(1.5*resolution, .75*resolution), diffraction_method='directional'))
            return sum(((.2+.3j)*getattr(getattr(result, basis), 'r_'+pol)
                         +(-.4+.1j)*getattr(getattr(result, basis), 't_'+pol)).real.sum()
                       for result in results for pol, basis in (('s', 's_inc'), ('p', 'p_inc')))
        value, derivative = jax.jit(jax.value_and_grad(objective))(phi)
        # The isotropic planar continuum has no azimuth dependence. Unequal
        # Yee cell widths break that invariance at finite resolution.
        errors.append(np.array([abs(float(value)-expected), abs(float(derivative))]))
        record_property(f'objective_error_{resolution}', float(errors[-1][0]))
        record_property(f'azimuth_derivative_{resolution}', float(derivative))
    assert np.all(errors[1] < .005)
    assert np.all(errors[1] < .45*errors[0])
