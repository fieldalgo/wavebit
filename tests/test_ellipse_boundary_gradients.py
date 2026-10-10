"""An NVF sample on a smooth ellipse boundary must not poison shape gradients."""
import numpy as np
import pytest

import wavebit
from wavebit._core.parametric import _ellipse_distance, plan_gamma
from wavebit._core.lattice import as_lattice


BASIS = np.array([[600., 0.], [416., 672.]])
POINT = np.array([787.5, 100., 0.])


def structure(parameters, magnetic=False):
    material = (wavebit.Material(4., 1.3)
                if magnetic else 4.)
    return [dict(background=1., shapes=[
        dict(type='ellipse', cx=0., cy=0., lx=parameters[0], ly=parameters[1],
             angle=parameters[2], material=material),
        dict(type='circle', cx=0., cy=200., lx=70., material=2.25)])]


def solver(backend, magnetic):
    return wavebit.rcwa(backend=backend, lattice=BASIS, wavelength=900.,
                       thickness=[170.], truncation_order=(1, 1), theta=.17, phi=.3,
                       polarization=(.8, .6j), factorization='nvf',
                       shape_nvf_grid_size=16, answers='field',
                       structure=structure(POINT, magnetic))


def objective(s, xp, parameters, magnetic, geometry=None):
    r = s.solve(structure=structure(parameters, magnetic), geometry=geometry, auto_detune=False)
    field = r.reconstruct_field([17.], [-23.], [70.])
    return xp.real(r.T.sum())+.1*xp.sum(xp.abs(field)**2)


def value_and_gradient(backend, function, value):
    if backend in ('jax', 'jit'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        calculate = jax.value_and_grad(lambda p: function(jnp, p))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        with jax.checking_leaks():
            result, gradient = calculate(jnp.asarray(value))
        return float(result), np.asarray(gradient)
    torch = pytest.importorskip('torch')
    p = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    result = function(torch, p)
    gradient, = torch.autograd.grad(result, p)
    return float(result.detach()), gradient.detach().numpy()


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('sign', [-1., 1.])
@pytest.mark.parametrize('offset', [-.125, 0., .125])
def test_fourth_power_of_boundary_distance_has_its_exact_smooth_derivative(backend, sign, offset):
    # Along the major axis, near the tip, distance**4 is exactly (a - |u|)**4.
    # The composition is smooth even though unsigned distance itself has a cusp.
    u = sign*(393.75+offset)
    def loss(xp, a):
        return _ellipse_distance(xp, xp.asarray(np.array([u])),
                                 xp.asarray(np.array([0.])), a, 50.)[0]**4
    value, gradient = value_and_gradient(backend, loss, 393.75)
    assert value == pytest.approx(offset**4, abs=1e-15)
    assert float(gradient) == pytest.approx(-4*offset**3, abs=1e-12)


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
@pytest.mark.parametrize('magnetic', [False, True], ids=['nonmagnetic', 'magnetic'])
def test_public_shape_gradient_is_finite_and_correct_at_an_ellipse_grid_hit(backend, magnetic):
    pytest.importorskip('torch' if backend == 'torch' else 'jax')
    if backend != 'torch':
        wavebit.enable_x64()
    s = solver('torch' if backend == 'torch' else 'jax', magnetic)
    geometry = s.freeze_geometry() if backend == 'jit' else None
    plan = plan_gamma(structure(POINT, magnetic)[0]['shapes'], as_lattice(lattice=BASIS))
    assert len(plan.pieces[0].imgs) > 1
    # At grid (j=1, i=2), relative coordinates (-206.25,-416) are exactly
    # (393.75,0) relative to the ellipse's periodic image at (-600,-416).
    u, v = (2.5/16-.5), (1.5/16-.5)
    np.testing.assert_array_equal(BASIS@np.array([u+1, v]), [393.75, 0.])
    value, gradient = value_and_gradient(backend,
        lambda xp, p: objective(s, xp, p, magnetic, geometry), POINT)
    expected_value = objective(solver('numpy', magnetic), np, POINT, magnetic)
    assert value == pytest.approx(expected_value, rel=2e-11, abs=2e-11)
    expected = []
    for i, h in enumerate([.01, .002, 1e-5]):
        plus, minus = POINT.copy(), POINT.copy()
        plus[i] += h
        minus[i] -= h
        expected.append((objective(solver('numpy', magnetic), np, plus, magnetic)
                         -objective(solver('numpy', magnetic), np, minus, magnetic))/(2*h))
    np.testing.assert_allclose(gradient, expected, rtol=5e-5, atol=2e-8)
    assert np.linalg.norm(expected) > 1e-5


def test_one_compiled_gradient_handles_repeated_boundary_crossings():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    s = solver('jax', False)
    geometry = s.freeze_geometry()
    calculate = jax.jit(jax.value_and_grad(
        lambda p: objective(s, jnp, p, False, geometry)))
    with jax.checking_leaks():
        for width in (787.49, 787.5, 787.51, 787.5):
            point = POINT.copy()
            point[0] = width
            value, gradient = calculate(jnp.asarray(point))
            expected_value = objective(solver('numpy', False), np, point, False)
            assert float(value) == pytest.approx(expected_value, abs=2e-11)
            plus, minus = point.copy(), point.copy()
            plus[0] += .001
            minus[0] -= .001
            fd = (objective(solver('numpy', False), np, plus, False)
                  -objective(solver('numpy', False), np, minus, False))/.002
            assert np.isfinite(np.asarray(gradient)).all()
            assert float(gradient[0]) == pytest.approx(fd, rel=5e-5, abs=2e-8)
