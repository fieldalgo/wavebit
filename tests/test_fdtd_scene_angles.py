"""Fixed-grid incidence sensitivities: independent dispersion and full graph checks."""
import numpy as np
import pytest

import wavebit


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=.2, wavelength=1., resolution=10, n_pml=6, n_periods=30,
                     backend=0, truncation_order=0, theta=.3, tetm='te', align_layers=True), **changes)


def value_and_jacobian(function, value, backend):
    if backend == 1:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        # Return a real vector so reverse mode checks phase as well as power.
        return jax.jit(lambda p: (function(p), jax.jacrev(function)(p)))(value)
    torch = pytest.importorskip('torch')
    parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    output = function(parameter)
    derivative = torch.autograd.functional.jacobian(function, parameter, vectorize=True)
    return output.detach().numpy(), derivative.detach().numpy()


@pytest.mark.parametrize('theta', [-.01, np.pi/2, np.inf, np.nan, .2j, [.2], True])
def test_invalid_angle_is_refused_before_time_stepping(monkeypatch, theta):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid angle reached time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    with pytest.raises(ValueError, match='theta|incidence angles'):
        wavebit.fdtd(**options(theta=theta))


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_array_angle_requires_static_diffraction_window(backend):
    from wavebit.backends.time_domain import time_loop_backend
    if backend == 1:
        pytest.importorskip('jax')
    elif backend == 2:
        pytest.importorskip('torch')
    theta = time_loop_backend(backend).array(.3)
    with pytest.raises(ValueError, match='explicit truncation_order'):
        wavebit.fdtd(**options(theta=theta, backend=backend, truncation_order=None))


def test_host_scalar_auto_window_preserves_material_jit():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    def solve(epsilon):
        return wavebit.fdtd(**options(structure=epsilon.reshape(1, 1, 1), theta=.3,
                                      truncation_order=None, backend=1)).given.T.sum()
    value, derivative = jax.jit(jax.value_and_grad(solve))(np.array(2.25))
    expected = wavebit.fdtd(**options(truncation_order=None)).given.T.sum()
    step = 1e-4
    finite = (wavebit.fdtd(**options(structure=np.full((1, 1, 1), 2.25+step))).given.T.sum()
              - wavebit.fdtd(**options(structure=np.full((1, 1, 1), 2.25-step))).given.T.sum())/(2*step)
    np.testing.assert_allclose(value, expected, atol=1e-11)
    np.testing.assert_allclose(derivative, finite, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_uniform_phase_angle_derivative_matches_independent_yee_dispersion(backend, side, record_property):
    from wavebit.backends.time_domain import time_loop_backend
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    runtime = time_loop_backend(backend)
    xp = runtime.xp
    band, phi, theta, depth = np.array([.98, 1.02]), .4, .3, .2
    kw = options(structure=np.ones((1, 1, 1)), wavelength=band, tetm=None, phi=phi,
                 n_periods=50, pulse_width=2., n_pml=10, backend=backend,
                 transverse_resolution=(15., 12.), incidence_side=side,
                 diffraction_method='directional', return_diagnostics=True)
    def objective(angle):
        results, d = wavebit.fdtd(**dict(kw, theta=angle))
        values = xp.stack([getattr(getattr(r, basis), key).reshape(()) for r in results
                           for basis, key in (('s_inc', 't_s'), ('p_inc', 't_p'))])
        return xp.concat((xp.real(values), xp.imag(values), d['angles']))
    value, derivative = value_and_jacobian(objective, theta, backend)
    _, d = wavebit.fdtd(**dict(kw, theta=theta, backend=0))
    dz, dy, dx = d['spacing']
    dt, k0 = d['dt'], 2*np.pi/band
    kx, ky = 2*np.pi*np.sin(theta)*np.array([np.cos(phi), np.sin(phi)])
    dkx, dky = 2*np.pi*np.cos(theta)*np.array([np.cos(phi), np.sin(phi)])
    x, y = 2*np.sin(kx*dx/2)/dx, 2*np.sin(ky*dy/2)/dy
    omega = 2*np.sin(k0*dt/2)/dt
    square = omega**2-x*x-y*y
    longitudinal = np.sqrt(square)
    phase = 2*np.arcsin(dz*longitudinal/2)/dz
    dsquare = -2*x*np.cos(kx*dx/2)*dkx-2*y*np.cos(ky*dy/2)*dky
    dphase = dsquare/(2*longitudinal*np.sqrt(1-(dz*longitudinal/2)**2))
    expected = np.repeat(np.exp(1j*depth*phase), 2)
    gradient = np.repeat(1j*depth*dphase*np.exp(1j*depth*phase), 2)
    angles = np.arcsin(band*np.sin(theta))
    dangles = band*np.cos(theta)/np.sqrt(1-(band*np.sin(theta))**2)
    np.testing.assert_allclose(value, np.r_[expected.real, expected.imag, angles], atol=2e-6)
    np.testing.assert_allclose(derivative, np.r_[gradient.real, gradient.imag, dangles], atol=2e-5)
    record_property('maximum_phase_derivative_error', float(np.max(abs(derivative-np.r_[gradient.real, gradient.imag, dangles]))))


@pytest.mark.parametrize('backend', [1, 2])
@pytest.mark.parametrize('method', ['electric', 'directional'])
@pytest.mark.slow
def test_angle_material_poles_monitors_and_records_share_one_graph(backend, method):
    from wavebit.backends.time_domain import time_loop_backend
    pattern = np.array([[[1., 1.2], [1.4, 1.1]]])
    def objective(p, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        stack = wavebit.Stack.from_structure(p[1]*array(pattern), [.2], period=[.4, .4], top=1., bot=1.)
        incidence = wavebit.Incidence(wavelength=[.98, 1.02], theta=p[0], phi=.4,
                                     polarization=(1., .6j), incidence_side='bottom')
        results, d = wavebit.fdtd(stack, incidence, backend=which, mu=p[2]*array(np.ones_like(pattern)),
                                 dispersion=np.ones_like(pattern, dtype=int), poles={1: [(8., .3, p[3])]},
                                 resolution=10, transverse_resolution=(12., 13.), n_pml=6,
                                 n_periods=30, truncation_order=[1, 1], grading=True, align_layers=True,
                                 pulse_width=1.5, pulse_notches=.7, source_distance=1., diffraction_distance=.3,
                                 diffraction_method=method, return_diagnostics=True, convergence_window=100,
                                 field_monitors={'inside': .1}, record_field=103, record_components=['ex'],
                                 record_bounds={'z': (.05, .15)})
        amplitude = sum(xp.real(.2*r.given.r_s+.3j*r.given.r_p+.4*r.given.t_s+.5j*r.given.t_p).sum()
                        +.1*r.given.T.sum() for r in results)
        return xp.stack((amplitude,
                         .1*xp.mean(xp.abs(d['fields']['ex']['values'])**2)
                         +.2*xp.mean(xp.abs(d['monitors']['inside']['fields']['ey'])**2),
                         .3*xp.sum(d['convergence']['scattering_change_bound']['r_p']**2)))
    p, step = np.array([.3, 1.8, 1.05, .8]), 1e-4
    expected = objective(p, 0)
    finite = np.stack([(objective(p+v, 0)-objective(p-v, 0))/(2*step) for v in np.eye(4)*step], axis=-1)
    value, derivative = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, expected, atol=1e-10)
    np.testing.assert_allclose(derivative, finite, rtol=4e-5, atol=2e-8)
    assert np.max(abs(finite[:, 0])) > .01


def test_jitted_invalid_angle_masks_results_monitors_and_time_records():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    def solve(theta):
        result, d = wavebit.fdtd(**options(theta=theta, backend=1, return_diagnostics=True,
                                          field_monitors={'inside': .1}, record_field=103,
                                          record_components=['ey'], record_bounds={'z': (.05, .15)}))
        return result.given.T, d['stable'], d['monitors']['inside']['fields']['ey'], d['field_e']
    compiled = jax.jit(solve)
    good = compiled(.3)
    assert bool(good[1]) and np.isfinite(good[0]).all()
    for angle in (-.1, np.pi/2, np.nan):
        result, stable, monitor, record = compiled(angle)
        assert not bool(stable)
        for value in (result, monitor, record):
            assert np.isnan(value).all()


@pytest.mark.parametrize('kind', ['physical', 'discrete', 'band'])
def test_jitted_angle_dependent_singularities_cannot_report_stable(kind):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    kw = options(period=1., truncation_order=1, diffraction_method='directional')
    if kind == 'physical':
        angle = 0.  # The first orders are exactly grazing at lambda = period.
    elif kind == 'discrete':
        spacing = .1
        dt = .8/np.sqrt(3/spacing**2)
        cutoff_k = 2/spacing*np.arcsin(spacing*np.sin(np.pi*dt)/dt)
        angle = np.arcsin((cutoff_k-2*np.pi)/(2*np.pi))
    else:
        kw.update(wavelength=[.8, 1.2], truncation_order=0)
        angle = 1.1  # The incident wave cannot propagate at the longest wavelength.
    def solve(theta):
        result, d = wavebit.fdtd(**dict(kw, theta=theta, backend=1, return_diagnostics=True))
        result = result if isinstance(result, list) else [result]
        return tuple(r.given.T for r in result), d['stable']
    compiled = jax.jit(solve)
    _, good = compiled(.2)
    assert bool(good)
    values, stable = compiled(angle)
    assert not bool(stable)
    assert all(np.isnan(value).all() for value in values)
    with pytest.raises(ValueError, match='grazing|discrete cutoff|propagate'):
        wavebit.fdtd(**dict(kw, theta=angle))


@pytest.mark.parametrize('backend', [1, 2])
def test_normal_incidence_has_finite_zero_angle_derivative(backend):
    def objective(theta):
        r = wavebit.fdtd(**options(theta=theta, backend=backend, tetm=None, phi=.4)).given
        return r.T.reshape(1)
    value, derivative = value_and_jacobian(objective, 0., backend)
    assert np.isfinite(value).all()
    np.testing.assert_allclose(derivative, 0., atol=1e-10)


def test_magnetic_slab_angle_sensitivity_refines_to_differentiated_airy(record_property):
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    band, theta, epsilon, mu, depth = np.array([.98, 1.02]), .3, 2.25, 1.1, .2
    transverse = (band*np.sin(theta))**2
    slope = 2*band**2*np.sin(theta)*np.cos(theta)
    q0, q1 = np.sqrt(1-transverse), np.sqrt(epsilon*mu-transverse)
    dq0, dq1 = -slope/(2*q0), -slope/(2*q1)
    phase = np.exp(2j*np.pi*q1*depth/band)
    dphase = 2j*np.pi*depth/band*dq1*phase
    expected, gradient = 0., 0.
    for admittance in (mu, epsilon):
        a, b, da, db = admittance*q0, q1, admittance*dq0, dq1
        r = (a-b)/(a+b)
        dr = 2*(da*b-a*db)/(a+b)**2
        denominator = 1-r*r*phase**2
        dd = -2*r*dr*phase**2-2*r*r*phase*dphase
        nr, nt = r*(1-phase**2), (1-r*r)*phase
        dnr = dr*(1-phase**2)-2*r*phase*dphase
        dnt = -2*r*dr*phase+(1-r*r)*dphase
        expected += np.real((.2+.3j)*nr/denominator+(-.4+.1j)*nt/denominator).sum()
        gradient += np.real((.2+.3j)*(dnr*denominator-nr*dd)/denominator**2
                            +(-.4+.1j)*(dnt*denominator-nt*dd)/denominator**2).sum()
    errors = []
    for resolution in (40, 80):
        def objective(angle):
            results = wavebit.fdtd(**options(theta=angle, structure=np.full((1, 1, 1), epsilon),
                                             mu=np.full((1, 1, 1), mu), wavelength=band, period=.05,
                                             tetm=None, phi=.4, backend=1, resolution=resolution,
                                             transverse_resolution=(1.5*resolution, .75*resolution),
                                             n_pml=resolution, n_periods=60, pulse_width=3.,
                                             diffraction_method='directional'))
            return sum(((.2+.3j)*getattr(getattr(result, basis), 'r_'+pol)
                         +(-.4+.1j)*getattr(getattr(result, basis), 't_'+pol)).real.sum()
                       for result in results for pol, basis in (('s', 's_inc'), ('p', 'p_inc')))
        value, derivative = jax.jit(jax.value_and_grad(objective))(theta)
        errors.append(np.array([abs(float(value)-expected), abs(float(derivative)-gradient)]))
    assert np.all(errors[1] < .005)
    assert np.all(errors[1] < .45*errors[0])
    record_property('analytic_objective', float(expected))
    record_property('analytic_angle_derivative', float(gradient))
    for resolution, error in zip((40, 80), errors):
        record_property(f'value_error_{resolution}', float(error[0]))
        record_property(f'derivative_error_{resolution}', float(error[1]))


def test_angle_sweeps_leave_mesh_source_planes_and_run_length_fixed():
    diagnostics = []
    for angle in (.1, .4):
        _, d = wavebit.fdtd(**options(theta=np.array(angle), phi=.4, tetm=None, grading=True,
                                      transverse_resolution=(15., 12.), return_diagnostics=True))
        diagnostics.append(d)
    first, second = diagnostics
    for key in ('shape', 'spacing', 'dt', 'steps', 'z_nodes', 'z_spacing', 'source_z', 'pml_bounds'):
        np.testing.assert_array_equal(first[key], second[key])
    for key in ('order_x', 'order_y', 'reflection_monitor_z', 'transmission_monitor_z'):
        np.testing.assert_array_equal(first['diffraction'][key], second['diffraction'][key])
    assert first['diffraction']['kx'].item() != second['diffraction']['kx'].item()


@pytest.mark.parametrize('backend', [1, 2])
def test_real_order_frame_derivatives_are_continuous_on_both_axes(backend):
    from wavebit._core.conventions import real_order_frame
    from wavebit.backends.time_domain import time_loop_backend
    if backend == 1:
        jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    def frame(p):
        cosine, sine = real_order_frame(xp, array([2., -2., 0., 0.])+p[1],
                                        array([0., 0., 2., -2.])+p[0], .4)
        return xp.concat((cosine, sine))
    value, derivative = value_and_jacobian(frame, np.zeros(2), backend)
    np.testing.assert_allclose(value, [0, 0, 1, -1, 1, -1, 0, 0], atol=1e-15)
    expected = np.zeros((8, 2))
    expected[:2, 0] = .5
    expected[6:, 1] = .5
    np.testing.assert_allclose(derivative, expected, atol=1e-15)


@pytest.mark.parametrize('backend', [1, 2])
def test_normal_crossed_diffraction_derivative_matches_one_sided_angle_change(backend):
    from wavebit.backends.time_domain import time_loop_backend
    def objective(theta, which):
        runtime = time_loop_backend(which)
        xp, array = runtime.xp, runtime.array
        r = wavebit.fdtd(**options(structure=np.array([[[1.3, 2.1], [1.8, 1.1]]]), period=[.4, .4],
                                   theta=theta, tetm=None, phi=.4, polarization=(1., .6j),
                                   backend=which, truncation_order=[1, 1], diffraction_method='directional',
                                   diffraction_distance=.3)).given
        weights = array(np.arange(1, 10).reshape(3, 3)/10)
        return xp.stack((xp.sum(weights*xp.real(.2*r.r_s+.3j*r.r_p)),
                         xp.sum(weights*xp.real(.4*r.t_s+.5j*r.t_p))))
    step = 1e-4
    expected = objective(0., 0)
    finite = (-3*expected+4*objective(step, 0)-objective(2*step, 0))/(2*step)
    value, derivative = value_and_jacobian(lambda a: objective(a, backend), 0., backend)
    np.testing.assert_allclose(value, expected, atol=1e-10)
    np.testing.assert_allclose(derivative, finite, rtol=5e-5, atol=2e-7)
    # A missing angle graph must exceed the absolute comparison tolerance.
    assert np.max(abs(finite)) > 100*2e-7
