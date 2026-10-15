"""Lattice container spelling preserves physical solve and field derivatives."""

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

CFG = dict(thickness=(170.,), wavelength=900., theta=.13, phi=.21,
           truncation_order=(1, 1), answers='field')


def structure(family):
    if family == 'raster':
        return np.array([[[1., 4., 1.], [1., 4., 4.], [1., 1., 1.]]])
    return [dict(background=1., shapes=[dict(type='rectangle', cx=20., cy=-30.,
                                            lx=150., ly=190., angle=0., material=4.)])]


def matrix(p, xp, spelling):
    if spelling == 'list':
        return [[p[0], p[1]], [0., 750.]]
    if spelling == 'mixed':
        return [(p[0], p[1]), np.array([0., 750.])]
    first = xp.stack([p[0], p[1]])
    second = xp.asarray([0., 750.], dtype=p.dtype)
    if spelling == 'rows':
        return (first, second)
    return xp.stack([first, second])


def solver(backend, family, p):
    return _legacy_wavebit_rcwa(backend=backend, **CFG, lattice=np.array([[p[0], p[1]], [0., 750.]]),
                       structure=structure(family), factorization='laurent' if family == 'raster' else 'nvf',
                       shape_nvf_grid_size=16 if family == 'shape' else None)


def score(result, field=None):
    if field is None:
        return result.T.sum().real
    return result.T.sum().real+.01*(abs(field)**2).mean().real


@pytest.mark.parametrize('family', ['raster', 'shape'])
@pytest.mark.parametrize('spelling', ['list', 'mixed', 'rows', 'array'])
@pytest.mark.parametrize('door', ['solve', 'solve_field'])
def test_nested_lattice_jit_derivative_matches_numpy(family, spelling, door):
    _legacy_wavebit_enable_x64()
    point = np.array([600., 40.])
    s = solver('jax', family, point)
    original = np.asarray(s.lattice).copy()
    geometry = s.freeze_geometry()
    def loss(p):
        kwargs = dict(lattice=matrix(p, jnp, spelling), geometry=geometry)
        if door == 'solve':
            return score(s.solve(**kwargs))
        return score(*s.solve_field(2, 2, 2, **kwargs))
    got = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(point)))
    def reference(p):
        fresh = solver('numpy', family, p)
        return float(score(fresh.solve()) if door == 'solve' else score(*fresh.solve_field(2, 2, 2)))
    fd = []
    for i in range(2):
        h = 1e-3
        plus, minus = point.copy(), point.copy()
        plus[i] += h
        minus[i] -= h
        fd.append((reference(plus)-reference(minus))/(2*h))
    assert np.linalg.norm(fd) > 1e-7
    np.testing.assert_allclose(got, fd, rtol=2e-5, atol=2e-8)
    np.testing.assert_array_equal(np.asarray(s.lattice), original)
