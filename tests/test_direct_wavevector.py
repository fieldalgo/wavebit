"""Direct physical transverse k: boundary physics, fields, AD and public API."""
import json

import numpy as np
import pytest

import wavebit
from test_half_space_power import ROUTES, ROUTE_IDS, _interface, _solver, _slab_heat


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('eps', [2.25, 4. + 3j])
@pytest.mark.parametrize('theta,phi', [(.4, .7), (-.3, 2.4), (0., 0.)])
def test_angles_and_direct_k_agree_in_fields_and_power(backend, side, eps, theta, phi):
    wavebit.enable_x64()
    common = dict(backend=backend, top=eps, bot=eps, incidence_side=side,
                  wavelength=500., period=[600., 700.], truncation_order=dict(x=1, y=1),
                  structure=np.array([[[1., 2.], [3., 4.]]]), thickness=[80.],
                  polarization=(.6, .8j), answers='field')
    angled = wavebit.rcwa(**common, theta=theta, phi=phi)
    x, y = map(_numpy, angled.order_kxy())
    k0 = 2*np.pi/500.
    direct = wavebit.rcwa(**common, kx=x[1].real*k0, ky=y[1].real*k0)
    a, b = angled.solve(), direct.solve()
    for attr in ('R', 'T', 'power_entering', 'absorption', 'r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(_numpy(getattr(a.incident, attr)),
                                   _numpy(getattr(b.incident, attr)), atol=2e-12, rtol=0)
    fa, fb = [s.reconstruct_field(3, 3, 3) for s in (angled, direct)]
    np.testing.assert_allclose(_numpy(fa), _numpy(fb), atol=1e-11, rtol=0)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pol', ['te', 'tm'])
@pytest.mark.parametrize('beta', [.5, 2.7])
def test_direct_k_matches_independent_magnetic_lossy_interface(route, side, pol, beta):
    medium = (4. + 3j, 1.2 + .05j, 2.25, 1.)
    s = _solver(route, medium, beta, pol, side)
    s.theta, s.phi = None, None
    s.kx = beta * (2*np.pi/500.)
    result = s.solve()
    r, t, R, T, _ = _interface(medium, beta)
    j = int(pol == 'tm')
    for value, want in [(result.R.sum(), R[j]), (result.T.sum(), T[j]),
                        (result.power_entering, T[j]), (result.absorption, 0.)]:
        assert float(value) == pytest.approx(want, abs=3e-12)
    np.testing.assert_allclose(getattr(result.incident, 'r_' + 'sp'[j]), r[j], atol=3e-12)
    np.testing.assert_allclose(getattr(result.incident, 't_' + 'sp'[j]), t[j], atol=3e-12)


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_direct_k_stays_fixed_during_wavelength_and_material_sweep(backend):
    s = wavebit.rcwa(backend=backend, kx=.003, ky=-.002, wavelength=500.,
                     top=wavebit.TabulatedMaterial([500., 700.], [2.25, 4.+3j]),
                     lattice=wavebit.Lattice.from_columns([[900., 300.], [0., 800.]]),
                     truncation_order=dict(u=1, v=1))
    A = np.array([[900., 300.], [0., 800.]])
    G = 2*np.pi*np.linalg.inv(A).T
    u, v = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    want = np.array([.003, -.002])[:, None, None] + np.einsum('ij,jmn->imn', G, [u, v])
    for wavelength in (500., 700.):
        s.wavelength = wavelength
        got = np.stack([_numpy(k) for k in s.order_kxy()]) * (2*np.pi/wavelength)
        np.testing.assert_allclose(got, want, atol=4e-18, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(kx=0., theta=0.), dict(ky=0., phi=0.),
                                     dict(kx=.1, theta=.2, phi=.3)])
def test_direction_inputs_are_exclusive(kwargs):
    with pytest.raises(ValueError, match='alternative incident directions'):
        wavebit.rcwa(**kwargs)
    assert any('alternative incident directions' in p for p in wavebit.Incidence(**kwargs).check())


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('value', [.01j, float('nan'), float('inf'), [0.], np.array([0.])])
def test_bad_k_is_not_silently_coerced(backend, value):
    with pytest.raises(ValueError, match='finite real scalar'):
        wavebit.rcwa(backend=backend, kx=value)
    assert wavebit.Incidence(kx=value).check()


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
def test_zero_power_evanescent_illumination_is_refused(route):
    s = _solver(route, (1., 1., 1., 1.), .5, 'te')
    s.theta, s.phi = None, None
    s.kx = 2. * (2*np.pi/500.)
    with pytest.raises(ValueError, match='no positive incident power'):
        s.solve()


def test_ky_cannot_be_dropped_by_pure_1d():
    with pytest.raises(ValueError, match='tetm requires ky=0'):
        wavebit.rcwa(structure=np.ones((1, 1, 4)), period=700., tetm='te', kx=.001, ky=.001).solve()
    s = wavebit.rcwa(structure=np.ones((1, 1, 4)), period=700., tetm='te', kx=.001)
    s.solve()
    assert s.grating_type == 0


def test_incidence_spec_roundtrip_and_sweep():
    spec = wavebit.JobSpec(
        stack=wavebit.Stack(layers=[wavebit.Layer(100., np.full((2, 2), 2.25))],
                            period=[700., 700.]),
        incidence=wavebit.Incidence(wavelength=500., kx=.002, ky=.001),
        sweep=wavebit.Sweep.from_dict({'kx': [.002, .003]}))
    assert spec.check() == ()
    clone = wavebit.JobSpec.from_json(json.dumps(spec.to_dict()))
    assert clone.key() == spec.key()
    points = list(clone.expand())
    assert [p.incidence.kx for p in points] == [.002, .003]
    actual = [float(wavebit.run(p).T.sum()) for p in points]
    assert abs(actual[0]-actual[1]) > 1e-4


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_direct_conditions_recast_and_stale_field(backend):
    s = wavebit.rcwa(backend=backend, kx=.002, ky=.001, wavelength=500.,
                     period=700., structure=np.full((1, 2, 2), 2.25),
                     thickness=[100.], answers='field')
    s.solve()
    s.kx = .003
    with pytest.raises(RuntimeError, match='changed|stale|scene'):
        s.reconstruct_field(2, 2, 2)
    s.complex_dtype = 1
    assert str(s.kx.dtype).endswith('float32')
    assert str(s.ky.dtype).endswith('float32')


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_fixed_k_absorption_gradient_matches_independent_heat(backend):
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        wavebit.enable_x64()
    else:
        import torch as xp
    k0 = 2*np.pi/500.
    def objective(v):
        s = wavebit.rcwa(backend=1 if backend == 'jax' else 2, kx=v[0], ky=v[1],
                         top=2.25+.7j, bot=1., wavelength=500., period=700.,
                         structure=np.full((1, 2, 2), 4.+.8j), thickness=[75.],
                         polarization=(.6, .8j))
        return s.solve().absorption
    def reference(v):
        beta = np.linalg.norm(v)/k0
        _, heat, weights = _slab_heat(2.25+.7j, 4.+.8j, 1., beta, 75.)
        weights *= [.36, .64]
        return np.dot(heat, weights)/weights.sum()
    v = np.array([.004, .002])
    h = 1e-7
    want_grad = [(reference(v+step)-reference(v-step))/(2*h) for step in np.eye(2)*h]
    if backend == 'jax':
        actual, gradient = jax.jit(jax.value_and_grad(objective))(xp.asarray(v))
        got_batch = jax.jit(jax.vmap(objective))(xp.asarray([v, v*1.1]))
        np.testing.assert_allclose(got_batch, [reference(v), reference(v*1.1)], atol=2e-12)
    else:
        arg = xp.tensor(v, dtype=xp.float64, requires_grad=True)
        actual = objective(arg)
        gradient, = xp.autograd.grad(actual, arg)
    assert float(_numpy(actual)) == pytest.approx(reference(v), abs=2e-12)
    np.testing.assert_allclose(_numpy(gradient), want_grad, rtol=0, atol=2e-6)


def test_jax_pytree_and_traced_condition_restore():
    import jax
    import jax.numpy as jnp
    wavebit.enable_x64()
    s = wavebit.rcwa(backend=1, kx=.002, ky=.001, wavelength=500.,period=700.,
                     structure=np.full((1, 2, 2), 2.25), thickness=[100.], answers='field')
    leaves, tree = jax.tree_util.tree_flatten(s)
    clone = jax.tree_util.tree_unflatten(tree, leaves)
    assert clone.theta is None and clone.phi is None
    np.testing.assert_allclose(clone.order_kxy(), s.order_kxy())
    original = s.solve().T
    np.testing.assert_allclose(jax.jit(lambda obj: obj.solve().T)(s), original, atol=1e-14)
    field_result, field = s.solve_field(2, 2, 2, geometry=s.freeze_geometry())
    np.testing.assert_allclose(field_result.T, original, atol=1e-14)
    assert np.isfinite(field).all()
    changed = jax.jit(lambda k: s.solve(kx=k).T)(.004)
    assert abs(float(changed.sum()-original.sum())) > 1e-4
    assert float(s.kx) == .002
    np.testing.assert_allclose(s.solve().T, original, atol=1e-14)
    @jax.jit
    def order(k):
        return wavebit.rcwa(backend=1, kx=k).order_kxy()[0]
    assert np.isnan(order(jnp.asarray(.01j))).all()


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
def test_direct_k_superposition_has_correct_order_powers(route):
    eps, beta, d = 2.25+.7j, .2, 75.
    s = _solver(route, (eps, 1., eps, 1.), beta, 'tm',
                excitation={(0, 0): 1., (1, 0): .5j}, thickness=d)
    s.theta, s.phi = None, None
    s.kx = beta*2*np.pi/500.
    result = s.solve()
    orders = result.orders_u
    q = np.sqrt(eps-(beta+.5*orders)**2)
    weights = (q/eps).real * np.array([1. if m == 0 else .25 if m == 1 else 0. for m in orders])
    expected = weights*np.exp(-4*np.pi/500.*q.imag*d)/weights.sum()
    np.testing.assert_allclose(np.asarray(result.T).ravel(), expected, atol=2e-12)
    assert float(result.absorption) == pytest.approx(1-expected.sum(), abs=2e-12)


@pytest.mark.parametrize('value', [float('nan'), .1j, [0., 1.]])
def test_spec_and_sweep_reject_invalid_components(value):
    from wavebit._core.spec import SPEC_VERSION
    with pytest.raises(wavebit.SpecError, match='finite real scalar'):
        wavebit.JobSpec.from_dict({'version': SPEC_VERSION, 'incidence': {'kx': value}})
    with pytest.raises(wavebit.SpecError, match='finite real scalar'):
        wavebit.Sweep.from_dict({'ky': [value]})


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_ky_only_and_single_precision(backend):
    s = wavebit.rcwa(backend=backend, ky=.002, wavelength=500.,complex_dtype=1)
    x, y = map(_numpy, s.order_kxy())
    np.testing.assert_allclose(x, 0.)
    np.testing.assert_allclose(y, .002*500./(2*np.pi), rtol=2e-7)
