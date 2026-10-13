"""An integer cell parameter must not truncate fractional constants under transforms."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest


jax = pytest.importorskip('jax')
import jax.numpy as jnp


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float64'])
@pytest.mark.parametrize('quantity', ['period', 'lattice'])
@pytest.mark.parametrize('transform', ['jit', 'vmap'])
def test_fractional_cell_literals_survive_integer_traced_inputs(dtype, quantity, transform):
    _legacy_wavebit.enable_x64()
    def inputs(p):
        return ({'period': [p, 750.123456789]} if quantity == 'period' else
                {'lattice': [[p, 180.123456789], [0., 750.123456789]]})
    def grid(p):
        return _legacy_wavebit.meshgrid_xy((3, 5), backend='jax', **inputs(p))
    points = np.array([600, 650])
    if transform == 'jit':
        function = jax.jit(grid)
        actual = [function(jnp.asarray(point, dtype=getattr(jnp, dtype))) for point in points]
    else:
        x, y = jax.vmap(grid)(jnp.asarray(points, dtype=getattr(jnp, dtype)))
        actual = list(zip(x, y))
    for point, got in zip(points, actual):
        expected = _legacy_wavebit.meshgrid_xy((3, 5), backend='numpy', **inputs(float(point)))
        for a, b in zip(got, expected):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
