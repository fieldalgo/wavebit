"""``sample_grid`` places every sample at the same bits on NumPy, PyTorch and JAX.

The shape normal fields fold each sample to its nearest periodic image, and on a hexagonal cell
some samples sit exactly on the edge between two images; which one wins is then decided by the
last bit of the position. JAX computed ``(i + 0.5)/N`` as a multiply by ``1/N`` (XLA's rewrite of
a division by a constant), one ulp off on about a third of the samples, and picked the other
image (tests/api/test_eager_grad_parity.py, hexagonal sample ties).
"""
import numpy as np
import pytest

from wavebit import Lattice
from wavebit._common.backend import array_namespace
from wavebit.lattice import sample_grid

CELLS = {'hexagonal': Lattice.hexagonal(1.), 'oblique': Lattice.oblique(1., 1.1, 1.3),
         'rectangular': np.array([[2., 0.], [0., .7]])}
SHAPES = [(n, n) for n in (13, 14, 17, 21, 65, 1024)] + [(13, 21), (7, 64)]


def positions(x, grid_shape, cell, dtype):
    xp = array_namespace(x)
    return [np.asarray(a.detach() if hasattr(a, 'detach') else a)
            for a in sample_grid(xp, grid_shape, cell, getattr(xp, dtype))]


# PyTorch's float32 `u * a1x + v * a2x` rounds apart from NumPy's, before and after the fractions
# moved to NumPy, so its float32 positions are not compared; the shape solvers run in float64.
@pytest.mark.parametrize(('backend', 'dtype'), [('jax', 'float64'), ('jax', 'float32'), ('torch', 'float64')])
@pytest.mark.parametrize('cell', sorted(CELLS))
def test_sample_positions_are_the_same_bits_on_every_backend(backend, dtype, cell):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        with jax.enable_x64(True):
            seen = {s: positions(jnp.zeros(1), s, CELLS[cell], dtype) for s in SHAPES}
    else:
        torch = pytest.importorskip('torch')
        seen = {s: positions(torch.zeros(1), s, CELLS[cell], dtype) for s in SHAPES}
    for s in SHAPES:
        expected = positions(np.zeros(1), s, CELLS[cell], dtype)
        for got, want in zip(seen[s], expected):
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)
