"""Real phase-angle illumination in an absorbing incident half-space.

References use prescribed phase/decay vectors or a numerical inversion of
the complex dispersion relation, never the production angle-conversion formula.
"""

from wavebit.materials.tabulated import TabulatedMaterial as _legacy_wavebit_TabulatedMaterial

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

import wavebit
from tests.rcwa.test_half_space_power import ROUTES, ROUTE_IDS, _interface, _slab_heat, _solver


def _numpy(value):
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value)


def _beta(epsilon, mu, theta):
    """Invert the phase angle by bisection on real transverse k/k0."""
    if theta == 0:
        return 0.
    target = abs(theta)
    em = complex(epsilon * mu)
    def angle(beta):
        return np.arctan2(beta, np.sqrt(em - beta**2).real)
    low, high = 0., 1.
    while angle(high) < target:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if angle(middle) < target:
            low = middle
        else:
            high = middle
    return np.copysign((low + high) / 2, theta)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('theta,phi', [(0., .7), (.3, 0.), (-.6, .4), (1.2, -2.4)])
@pytest.mark.parametrize('mu', [1., 1.2 + .02j])
def test_prescribed_phase_and_decay_vectors(backend, side, theta, phi, mu):
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    phase, decay = 1.7, .12
    em = phase**2 - decay**2 + 2j * phase * np.cos(theta) * decay
    incoming = wavebit.Material(em / mu, mu=mu)
    solver = _legacy_wavebit_rcwa(backend=backend, top=incoming if side == 'top' else 1.,
                          bot=incoming if side == 'bottom' else 1., incidence_side=side,
                          theta=theta, phi=phi, period=[900., 1100.], truncation_order=0)
    kx, ky = map(_numpy, solver.order_kxy())
    want = phase * np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
    np.testing.assert_allclose([kx.item(), ky.item()], want, rtol=0, atol=8e-15)
    assert np.imag(kx).item() == 0 and np.imag(ky).item() == 0
    inward = np.sqrt(em - kx.item()**2 - ky.item()**2)
    assert inward == pytest.approx(phase * np.cos(theta) + 1j * decay, abs=8e-15)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('theta', [0., .4, 1.3])
def test_lossless_limit_is_the_standard_angle_formula(backend, theta):
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    for mu in (1., 1.4):
        solver = _legacy_wavebit_rcwa(backend=backend, top=wavebit.Material(2.7, mu=mu),
                              theta=theta, phi=.6, period=[900., 900.], truncation_order=0)
        actual = np.array([_numpy(k).item() for k in solver.order_kxy()])
        want = np.sqrt(2.7 * mu) * np.sin(theta) * np.array([np.cos(.6), np.sin(.6)])
        np.testing.assert_allclose(actual, want, rtol=0, atol=8e-15)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pol', ['te', 'tm'])
@pytest.mark.parametrize('mu', [1., 1.2 + .08j])
def test_interface_power_at_the_requested_phase_angle(route, side, pol, mu):
    medium = (4. + 3j, mu, 2.25, 1.)
    theta = .4
    beta = _beta(medium[0], mu, theta)
    solver = _solver(route, medium, beta, pol, side)
    solver.theta = theta
    result = solver.solve()
    r, t, R, T, _ = _interface(medium, beta)
    j = int(pol == 'tm')
    assert result.R.sum() == pytest.approx(R[j], abs=4e-12)
    assert result.T.sum() == pytest.approx(T[j], abs=4e-12)
    assert result.power_entering == pytest.approx(T[j], abs=4e-12)
    assert result.absorption == pytest.approx(0., abs=4e-12)
    np.testing.assert_allclose(getattr(result.incident, 'r_' + 'sp'[j]), r[j], atol=4e-12, rtol=0)
    np.testing.assert_allclose(getattr(result.incident, 't_' + 'sp'[j]), t[j], atol=4e-12, rtol=0)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('angles', [(.4 + .2j, .3), (.4, .3 + .1j)])
def test_explicit_complex_angles_keep_the_formal_parameterization(backend, angles):
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    theta, phi = angles
    solver = _legacy_wavebit_rcwa(backend=backend, top=4. + 3j, theta=theta, phi=phi,
                          period=[1000., 1000.], truncation_order=0)
    want = np.sqrt(4. + 3j) * np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
    np.testing.assert_allclose([_numpy(k).item() for k in solver.order_kxy()], want,
                               atol=4e-15, rtol=0)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_weakly_absorbing_negative_epsilon_keeps_its_phase_direction(backend):
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    eps, theta = -4. + 1e-8j, .4
    solver = _legacy_wavebit_rcwa(backend=backend, top=eps, theta=theta,
                          period=[1000., 1000.], truncation_order=0)
    beta = _numpy(solver.order_kxy()[0]).item()
    q = np.sqrt(eps - beta**2)
    assert beta.real > 0
    assert beta.imag == 0
    assert np.arctan2(beta.real, q.real) == pytest.approx(theta, abs=1e-14)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('theta', [np.pi / 2, -np.pi / 2, 1.7])
def test_absorbing_grazing_phase_angle_is_refused(backend, theta):
    if backend == 'jax':
        _legacy_wavebit_enable_x64()
    solver = _legacy_wavebit_rcwa(backend=backend, top=4. + 3j, theta=theta,
                          period=[1000., 1000.], truncation_order=0)
    with pytest.raises(ValueError, match=r'abs\(theta\).*pi/2'):
        solver.order_kxy()


@pytest.mark.parametrize('top', [-4.0, wavebit.Material(-4.0, mu=1.0),
                                 wavebit.Material(-2.0, mu=2.0)], ids=['metal', 'mu-one', 'mng'])
def test_a_lossless_lit_medium_that_carries_no_wave_is_refused_by_name(top):
    """A lossless lit half-space with ``Re(eps*mu) <= 0`` carries no wave, so no real angle
    names a direction in it; the refusal names that condition, not only the absorbing medium's
    ``abs(theta) < pi/2``, which such a medium meets."""
    solver = _legacy_wavebit_rcwa(top=top, theta=0.3, period=[1000., 1000.], truncation_order=0)
    with pytest.raises(ValueError, match=r'lossless lit half-space must carry one'):
        solver.order_kxy()


def test_oblique_lattice_and_dispersive_conditions_use_the_same_phase_vector():
    material = _legacy_wavebit_TabulatedMaterial([500., 700.], [4. + 1j, 6. + 2j])
    lattice = wavebit.Lattice.from_columns([[900., 300.], [0., 800.]])
    solver = _legacy_wavebit_rcwa(top=1., bot=material, incidence_side='bottom', theta=.4, phi=.6,
                          wavelength=500., lattice=lattice, truncation_order=dict(u=1, v=1))
    reciprocal = 2 * np.pi * np.linalg.inv(np.column_stack(([900., 0.], [300., 800.]))).T
    for wavelength, eps in [(500., 4. + 1j), (700., 6. + 2j)]:
        solver.wavelength = wavelength
        x, y = map(_numpy, solver.order_kxy())
        base = _beta(eps, 1., .4) * np.array([np.cos(.6), np.sin(.6)])
        u, v = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        want = base[:, None, None] + wavelength / (2 * np.pi) * np.einsum(
            'ij,jmn->imn', reciprocal, np.stack([u, v]))
        np.testing.assert_allclose(np.stack([x, y]), want, atol=4e-15, rtol=0)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_absorption_gradients_at_fixed_phase_angle(backend, algorithm):
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        _legacy_wavebit_enable_x64()
    else:
        import torch as xp

    def quantities(v):
        loss, theta = v[0], v[1]
        solver = _legacy_wavebit_rcwa(
            backend='jax' if backend == 'jax' else 'torch', top=2.25 + 1j * loss, bot=1.,
            theta=theta, phi=.3, polarization=(.6, .8j), wavelength=500.,
            period=[1000., 1000.], truncation_order=0, thickness=[75.],
            structure=xp.full((1, 2, 2), 4. + .8j, dtype=xp.complex128),
            layer_algorithm=algorithm)
        result = solver.solve()
        return xp.stack([result.power_entering, result.absorption])

    def reference(v):
        ei = 2.25 + 1j * v[0]
        beta = _beta(ei, 1., v[1])
        entering, absorbed, weights = _slab_heat(ei, 4. + .8j, 1., beta, 75.)
        power = weights * [.36, .64]
        return np.array([np.dot(entering, power), np.dot(absorbed, power)]) / power.sum()

    v = np.array([.7, .4])
    h = 1e-5
    steps = np.eye(2) * h
    want_grad = np.column_stack([(reference(v + step) - reference(v - step)) / (2*h)
                                for step in steps])
    if backend == 'jax':
        actual = jax.jit(quantities)(xp.asarray(v))
        gradient = jax.jit(jax.jacrev(quantities))(xp.asarray(v))
    else:
        arg = xp.tensor(v, dtype=xp.float64, requires_grad=True)
        actual = quantities(arg)
        gradient = xp.autograd.functional.jacobian(quantities, arg)
    np.testing.assert_allclose(_numpy(actual), reference(v), rtol=0, atol=5e-12)
    np.testing.assert_allclose(_numpy(gradient), want_grad, rtol=0, atol=3e-7)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_lossless_material_gradient_is_continuous(backend):
    if backend == 'jax':
        import jax
        import jax.numpy as xp
        _legacy_wavebit_enable_x64()
    else:
        import torch as xp

    def component(loss):
        solver = _legacy_wavebit_rcwa(backend='jax' if backend == 'jax' else 'torch', top=2.25 + 1j*loss,
                              theta=.4, period=[1000., 1000.], truncation_order=0)
        return xp.real(solver.order_kxy()[0][0])

    if backend == 'jax':
        derivative = jax.jit(jax.grad(component))(xp.asarray(0.))
        second = jax.jit(jax.grad(jax.grad(component)))(xp.asarray(0.))
    else:
        loss = xp.tensor(0., dtype=xp.float64, requires_grad=True)
        derivative, = xp.autograd.grad(component(loss), loss, create_graph=True)
        second, = xp.autograd.grad(derivative, loss)
    assert _numpy(derivative).item() == pytest.approx(0., abs=1e-14)
    # Curvature at b=0, derived from the dispersion relation at fixed phase angle.
    want_second = np.sin(.4) / (4 * 2.25**1.5 * np.cos(.4)**2)
    assert _numpy(second).item() == pytest.approx(want_second, abs=1e-13)


@pytest.mark.parametrize('route', ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_reconstructed_field_has_uniform_surface_magnitude_and_normal_decay(route, side):
    eps, theta, d = 2.25 + .7j, .4, 100.
    beta = _beta(eps, 1., theta)
    solver = _solver(route, (eps, 1., eps, 1.), beta, 'te', side,
                     thickness=d, answers='field')
    solver.theta = theta
    solver.solve()
    xs, zs = np.array([-100., 0., 100.]), np.array([10., 40., 80.])
    field = np.asarray(solver.reconstruct_field(sample_x=xs, sample_y=1, sample_z=zs))
    if field.ndim == 5:
        field = field[0]  # declared illumination; the other entries are pure s/p solves
    ey = field[..., 0 if route[0] == '1d' else 1][:, 0, :]
    k0 = 2*np.pi/500.
    q = np.sqrt(eps - beta**2)
    direction = -1 if side == 'top' else 1
    expected = np.exp(1j*k0*(beta*(xs[None, :] - xs[0])
                            + direction*q*(zs[:, None] - zs[0])))
    np.testing.assert_allclose(ey / ey[0, 0], expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('eps,theta', [(2.25, .4), (4. + 3j, .5),
                                       (-4. + 1e-4j, .4), (4. + 3j, 1.55)])
def test_single_precision_phase_angle(backend, eps, theta):
    solver = _legacy_wavebit_rcwa(backend=backend, complex_dtype=np.complex64, top=eps,
                          theta=theta, phi=.3, period=[1000., 1000.], truncation_order=0)
    x, y = [_numpy(k).item() for k in solver.order_kxy()]
    beta = _beta(eps, 1., theta)
    np.testing.assert_allclose([x, y], beta*np.array([np.cos(.3), np.sin(.3)]),
                               rtol=2e-6, atol=1e-12)
    assert x.imag == 0 and y.imag == 0


def test_jit_vmap_and_invalid_traced_phase_angle():
    import jax
    import jax.numpy as jnp
    _legacy_wavebit_enable_x64()

    def transverse(loss, theta):
        solver = _legacy_wavebit_rcwa(backend='jax', top=2.25 + 1j*loss, theta=theta,
                              period=[1000., 1000.], truncation_order=0)
        return solver.order_kxy()[0][0]

    losses = np.array([0., .4, .8])
    angles = np.array([.2, .4, .6])
    got = jax.jit(jax.vmap(transverse))(jnp.asarray(losses), jnp.asarray(angles))
    want = [_beta(2.25 + 1j*b, 1., theta) for b, theta in zip(losses, angles)]
    np.testing.assert_allclose(got, want, rtol=0, atol=4e-15)
    invalid = jax.jit(transverse)(jnp.asarray(.5), jnp.asarray(np.pi/2))
    assert np.isnan(np.asarray(invalid))
