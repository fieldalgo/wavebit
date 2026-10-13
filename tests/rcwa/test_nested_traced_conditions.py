"""A nested traced condition must restore its declaration before the next solve."""

from wavebit.compat import legacy as _legacy_wavebit
from functools import partial
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

CFG = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
           truncation_order=(0, 0), theta=.13, phi=.21, answers='field',
           factorization='li')


def design(value):
    return [dict(background=1., shapes=[dict(type='rectangle', cx=0., cy=0.,
                                            lx=100., ly=120., material=value)])]


def solver(backend='jax', **kwargs):
    return _legacy_wavebit.rcwa(backend=backend, **CFG, **kwargs)


@pytest.mark.parametrize('door', ['solve', 'solve_field', 'solve_field_eager'])
@pytest.mark.parametrize('failure', [False, True])
def test_nested_material_gradient_restores_modeling_path_before_following_solve(door, failure):
    raster = np.full((1, 2, 2), 4.)
    s = solver(structure=raster)
    original_geometry = s.freeze_geometry()
    shape_geometry = solver(structure=design(4.)).freeze_geometry()
    method = getattr(s, door)
    if door != 'solve':
        method = partial(method, 2, 2, 2)
    def power(result):
        return (result if door == 'solve' else result[0]).T.sum().real
    observed = []
    def forward(structure):
        if failure:
            from wavebit.rcwa._engine.geometry import StaticGeometry, StaleGeometryError
            with pytest.raises(StaleGeometryError, match='factorization'):
                method(structure=design(jnp.mean(structure)),
                       geometry=StaticGeometry(factorization='nvf'))
            first_power = 0.
        else:
            first_power = power(method(structure=design(jnp.mean(structure)), geometry=shape_geometry))
        observed.append(s.modeling_path)
        second = method(geometry=original_geometry)
        return first_power + power(second)
    with jax.checking_leaks():
        got = np.asarray(s.grad(['structure'], forward, lambda value: value)['structure'])
    assert observed == ['raster']
    def reference(values):
        first = solver('numpy', structure=design(float(np.mean(values)))).solve().T.sum().real
        second = solver('numpy', structure=values).solve().T.sum().real
        return float((0. if failure else first) + second)
    expected = np.empty_like(raster)
    for index in np.ndindex(raster.shape):
        a, b = raster.copy(), raster.copy()
        a[index] += 1e-4
        b[index] -= 1e-4
        expected[index] = (reference(a)-reference(b))/(2e-4)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=1e-9)
    assert s.modeling_path == 'raster'
    np.testing.assert_array_equal(s.structure, raster)
