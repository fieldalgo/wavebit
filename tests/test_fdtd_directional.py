"""Travelling-wave extraction from discrete Maxwell fields, including returns."""
import numpy as np
import pytest

import wavebit
from wavebit._core.scene_fdtd import _directional_projection, _split_directions


def options(**changes):
    return dict(dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
                     period=.1, wavelength=1., theta=.25, tetm='tm', backend='numpy',
                     resolution=20, n_pml=6, n_periods=60, pulse_width=3., align_layers=True,
                     truncation_order=0, return_diagnostics=True, diffraction_method='directional'), **changes)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_mixed_discrete_maxwell_modes_separate_at_wrapped_planes(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    from wavebit.backends.time_domain import time_loop_backend
    runtime = time_loop_backend(backend)
    xp, array = runtime.xp, runtime.array
    shape, spacing, dt, steps = (8, 5, 6), np.array([.2, .3, .25]), .04, 23
    eps, mu = 1.7, 1.2
    k = np.array([.72, -.43, 2*np.pi/(shape[0]*spacing[0])])
    discrete = 2*np.sin(k*spacing[::-1]/2)/spacing[::-1]
    frequency = np.linalg.norm(discrete)/np.sqrt(eps*mu)
    omega = 2*np.arcsin(dt*frequency/2)/dt
    amplitudes = np.array([[.4+.2j, -.7+.1j], [-.3+.4j, .2-.6j]])
    coords = np.meshgrid(*[np.arange(n)*d for n, d in zip(shape, spacing)], indexing='ij')
    e, h = np.zeros((3,)+shape, complex), np.zeros((3,)+shape, complex)
    for sign, tangential in zip((1, -1), amplitudes):
        vector = k*np.array([1, 1, sign])
        q = discrete*np.array([1, 1, sign])
        electric = np.r_[tangential, -np.dot(q[:2], tangential)/q[2]]
        magnetic = np.cross(q, electric)/(mu*frequency)
        for component in range(3):
            for target, offset, amplitude, time in (
                (e, np.eye(3)[component]/2, electric[component], 0.),
                (h, (1-np.eye(3)[component])/2, magnetic[component], dt/2)):
                phase = sum(vector[2-a]*(coords[a]+offset[2-a]*spacing[a]) for a in range(3))
                target[component] += amplitude*np.exp(1j*(phase+omega*time))
    indices = [0, 2, 5]
    out = wavebit.fdtd_grid(np.full(shape, eps), mu=mu, spacing=spacing, dt=dt, steps=steps,
                            electric=e, magnetic=h, bloch_phase=(0., k[1]*shape[1]*spacing[1], k[0]*shape[2]*spacing[2]),
                            wavelengths=[2*np.pi/omega], monitor_z=indices, backend=backend)
    spatial = array(np.exp(-1j*(k[1]*coords[1][0]+k[0]*coords[2][0])))
    divisor = array(np.cos(k[:2]*spacing[:0:-1]/2))
    fe = xp.sum(out['electric_spectrum'][:, :, :2]*spatial, axis=(-2, -1))/(shape[1]*shape[2])/divisor
    fh = xp.sum(out['magnetic_spectrum'][:, :, :2]*spatial, axis=(-2, -1))/(shape[1]*shape[2])/xp.flip(divisor, axis=0)
    matrix = _directional_projection(np.array([2*np.pi/omega]), k[:1].reshape(1, 1), k[1:2].reshape(1, 1),
                                     eps, mu, dt, spacing[2], spacing[1], spacing[0])
    projection = array(np.repeat(matrix[:, None], len(indices), axis=1))
    separated = _split_directions(xp, fe[..., None, None], fh[..., None, None], projection)
    expected = np.stack([amplitudes*np.exp(1j*np.array([1, -1])[:, None]*k[2]*spacing[0]*i)
                         for i in indices])[None, ..., None, None]*steps*dt
    host = separated.detach().numpy() if backend == 'torch' else np.asarray(separated)
    np.testing.assert_allclose(host, expected, atol=3e-13)


def test_evanescent_decaying_and_growing_fields_separate_without_conjugation():
    band, dt, dx, dy, dz, eps, mu = np.array([3.]), .03, .1, .12, .2, 1.7, 1.2
    kx, ky = np.array([[5.]]), np.array([[.7]])
    frequency = float(2*np.sin(np.pi*dt/band[0])/dt)
    x, y = float(2*np.sin(kx.item()*dx/2)/dx), float(2*np.sin(ky.item()*dy/2)/dy)
    z = 1j*np.sqrt(x*x+y*y-eps*mu*frequency**2)
    amplitudes = np.array([[.4+.1j, -.3+.2j], [.2-.5j, .7+.6j]])
    electric, magnetic = np.zeros(2, complex), np.zeros(2, complex)
    for sign, tangential in zip((1, -1), amplitudes):
        q = np.array([x, y, sign*z])
        field = np.r_[tangential, -np.dot(q[:2], tangential)/q[2]]
        electric += field[:2]
        magnetic += np.cross(q, field)[:2]/(mu*frequency)*np.cos(np.arcsin(dz*z/2))
    projection = _directional_projection(band, kx, ky, eps, mu, dt, dx, dy, dz)[:, None]
    result = _split_directions(np, electric[None, None, :, None, None], magnetic[None, None, :, None, None], projection)
    np.testing.assert_allclose(result[0, 0, :, :, 0, 0], amplitudes, atol=1e-13)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_conical_two_dimensional_window_in_uniform_magnetic_medium(backend):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    elif backend == 'torch':
        pytest.importorskip('torch')
    material = wavebit.Material(1.7, 1.2)
    results, d = wavebit.fdtd(**options(structure=np.full((1, 2, 2), 1.7), mu=np.full((1, 2, 2), 1.2),
                                       top=material, bot=material, period=[.4, .6], wavelength=[.51, .55],
                                       theta=.2, tetm=None, phi=.6, polarization=(1., 1j),
                                       truncation_order=[1, 1], backend=backend, pulse_width=1.5, n_periods=70,
                                       diffraction_distance=.2))
    def host(value):
        return value.detach().numpy() if backend == 'torch' else np.asarray(value)
    for result in results:
        np.testing.assert_allclose(host(result.incident.R), 0., atol=1e-15)
        expected = np.zeros((3, 3)); expected[1, 1] = 1.
        np.testing.assert_allclose(host(result.incident.T), expected, atol=1e-8)
    assert host(d['diffraction']['incoming']['p_inc']['t_p']).shape == (2, 3, 3)


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_planes_do_not_mix_returning_waves_into_scattering(side):
    kw = options(top=wavebit.Material(1.2, 1.1), bot=wavebit.Material(1.8, 1.2),
                 tetm=None, phi=.4, polarization=(.4j, .7), incidence_side=side, convergence_window=100,
                 record_field=37, record_components=['ex', 'hy'], record_bounds={'z': (-.05, .25)})
    full, a = wavebit.fdtd(**kw)
    near, b = wavebit.fdtd(**kw, diffraction_distance=(.2, .3))
    for column in ('incident', 's_inc', 'p_inc'):
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            np.testing.assert_allclose(getattr(getattr(full, column), key), getattr(getattr(near, column), key), atol=1e-8)
            np.testing.assert_allclose(a['diffraction']['incoming'][column][key], b['diffraction']['incoming'][column][key], atol=1e-8)
    incoming = b['diffraction']['incoming']
    for name in a['fields']:
        np.testing.assert_allclose(a['fields'][name]['values'], b['fields'][name]['values'], atol=1e-8)
        for axis in ('x', 'y', 'z', 't'):
            np.testing.assert_array_equal(a['fields'][name][axis], b['fields'][name][axis])
    for key in ('r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(incoming['incident'][key], (.4j*incoming['s_inc'][key]+.7*incoming['p_inc'][key])/np.sqrt(.65), atol=1e-13)
    assert bool(a['convergence']['scattering_settled']) and bool(b['convergence']['scattering_settled'])


def test_directional_slab_refines_to_airy():
    q0, q1 = np.cos(.25), np.sqrt(2.25-np.sin(.25)**2)
    r = (2.25*q0-q1)/(2.25*q0+q1)
    phase = np.exp(2j*np.pi*q1*.213)
    expected = np.array([r*(1-phase**2)/(1-r*r*phase**2), (1-r*r)*phase/(1-r*r*phase**2)])
    errors = []
    for resolution in (40, 80):
        result, _ = wavebit.fdtd(**options(thickness=[.213], resolution=resolution,
                                           n_pml=round(.3*resolution), diffraction_distance=.2))
        errors.append(np.max(abs(np.array([result.incident.r_p.item(), result.incident.t_p.item()])-expected)))
    assert errors[1] < .002 and errors[1] < .4*errors[0]


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_directional_material_mu_pole_and_incoming_gradients(backend):
    def objective(parameters, which):
        result, d = wavebit.fdtd(**options(structure=parameters[:1].reshape(1, 1, 1),
                                           mu=parameters[1:2].reshape(1, 1, 1), dispersion=np.ones((1, 1, 1), int),
                                           poles={1: [(8., .3, parameters[2])]}, backend=which,
                                           resolution=10, n_pml=4, n_periods=40, pulse_width=2.,
                                           tetm=None, phi=.6, polarization=(1., .6j),
                                           incidence_side='bottom', diffraction_distance=.2, convergence_window=100,
                                           record_field=101, record_components=['ex'], record_bounds={'z': (.05, .15)}))
        r = result.incident
        return ((.2*r.r_s+.3j*r.r_p+.4*r.t_s+.5j*r.t_p).real.sum()
                +.2*(abs(d['diffraction']['incoming']['p_inc']['t_p'])**2).sum()
                +.1*(abs(d['fields']['ex']['values'])**2).mean()
                +.3*(d['convergence']['scattering_change_bound']['r_p']**2).sum())
    parameters = np.array([2.25, 1.05, 1.])
    expected = objective(parameters, 'numpy')
    step = 1e-4
    finite = np.array([(objective(parameters+v, 'numpy')-objective(parameters-v, 'numpy'))/(2*step)
                       for v in np.eye(3)*step])
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, 'jax')))(parameters)
    else:
        torch = pytest.importorskip('torch')
        p = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        value = objective(p, 'torch')
        gradient, = torch.autograd.grad(value, p)
        value, gradient = value.detach().numpy(), gradient.detach().numpy()
    np.testing.assert_allclose(value, expected, atol=1e-11)
    np.testing.assert_allclose(gradient, finite, rtol=3e-5, atol=1e-8)


def test_record_carrier_can_have_unrequested_grazing_orders():
    _, d = wavebit.fdtd(**options(period=.8, wavelength=[.79, .81], theta=0., truncation_order=1,
                                  diffraction_distance=.2, record_field=50, record_components=['ex', 'hy'],
                                  record_bounds={'z': (0., .2)}))
    assert all(np.isfinite(v['values']).all() for v in d['fields'].values())


@pytest.mark.parametrize('stride', [1, 3])
def test_directional_tail_matches_an_independent_truncated_solve(stride):
    kw = options(resolution=10, n_pml=4, n_periods=40, pulse_width=2.,
                 dispersion=np.ones((1, 1, 1), int), poles={1: [(8., .02, 5.)]},
                 diffraction_distance=.2, dft_subsample=stride)
    full, d = wavebit.fdtd(**kw, convergence_window=77)
    prefix, _ = wavebit.fdtd(**dict(kw, n_periods=(d['steps']-77-.01)*d['dt']))
    for basis in ('incident', 's_inc', 'p_inc'):
        diagnostics = d['convergence'] if basis == 'incident' else d['convergence'][basis]
        for key in ('r_s', 'r_p', 't_s', 't_p'):
            expected = getattr(getattr(full, basis), key)-getattr(getattr(prefix, basis), key)
            np.testing.assert_allclose(diagnostics['amplitude_change'][key][0], expected, atol=1e-10)


@pytest.mark.parametrize('kind', ['method', 'grading'])
def test_bad_method_or_nonuniform_plane_refused_before_time_stepping(monkeypatch, kind):
    import wavebit._core.scene_fdtd as scene
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight validation ran after time stepping')
    monkeypatch.setattr(scene, 'fdtd_grid', forbidden)
    changes = dict(diffraction_method='unknown') if kind == 'method' else dict(grading=True, resolution=80, diffraction_distance=.05)
    with pytest.raises(ValueError, match='diffraction|Directional'):
        wavebit.fdtd(**options(**changes))


@pytest.mark.parametrize('kind', ['cutoff', 'longitudinal', 'temporal'])
def test_discrete_singularities_are_not_divided_through(kind):
    dt, dz, x = .02, .1, 0.
    if kind == 'cutoff':
        frequency = 2*np.sin(np.pi*dt)/dt
        x = 20*np.arcsin(.1*frequency/2)
    elif kind == 'longitudinal':
        dz = 1.
    else:
        dt = .6
    with pytest.raises(ValueError, match='Directional'):
        _directional_projection(np.array([1.]), np.array([[x]]), np.zeros((1, 1)), 1., 1., dt, .1, .1, dz)
