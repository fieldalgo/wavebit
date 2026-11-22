"""The chain tables formed a batch of rows at a time are the tables formed at once, bit for bit.

``modal_gradient._frechet_chain_tables`` forms its divided-difference tables through
``_dd_lattice_rows`` since its path capacity grew to ``4 c`` (#219): at once, a 66-mode
first-order layer's ``jit(grad(sum vmap))`` held 8.9 GB, past what a CI runner holds beside a
second worker. The batches change where the lattice's intermediates live, not one entry.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

from wavebit.rcwa._engine.modal_gradient import _dd_lattice, _dd_lattice_rows  # noqa: E402


def _points(m, batch=()):
    """Eigenvalues with a near-degenerate cluster, as at an exceptional point, and ``exp``'s
    first divided differences ``F1`` formed as ``_coupled_frechet`` forms them."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=batch + (m,)) + 1j * rng.normal(size=batch + (m,))
    a[..., :4] = 0.3 + 0.2j + 1e-9 * np.arange(4)                       # a cluster of four
    a = jnp.asarray(a)
    ea = jnp.exp(a)
    d = a[..., :, None] - a[..., None, :]
    right = jnp.real(d) >= 0
    x = jnp.expm1(jnp.where(right, -d, d))
    num = jnp.where(right, ea[..., :, None] * (-x), ea[..., None, :] * x)
    nz = d != 0
    F1 = jnp.where(nz, num / jnp.where(nz, d, jnp.ones_like(d)),
                   jnp.broadcast_to(ea[..., :, None], d.shape))
    return a, ea, F1


@pytest.mark.parametrize('batch', [(), (3,)])
@pytest.mark.parametrize('block', [1, 4, 7])
def test_a_table_formed_in_batches_of_rows_is_the_table_formed_at_once(batch, block):
    m, cp, c, cl = 12, 10, 6, 4
    a, ea, F1 = _points(m, batch)
    rng = np.random.default_rng(3)
    i2, k2, s2 = (jnp.asarray(rng.integers(0, m, cp)) for _ in range(3))
    X = jnp.asarray(rng.integers(0, m, (c, cl)))
    K = jnp.asarray(rng.integers(0, m, c))
    idx = jnp.arange(m)
    tables = [
        [i2[:, None], k2[:, None], s2[:, None], idx[None, :]],                         # F20
        [i2[:, None, None], k2[:, None, None], s2[:, None, None],
         X[None, :, :], K[None, :, None]],                                              # F21
        [i2[:, None], k2[:, None], s2[:, None], i2[None, :], k2[None, :], s2[None, :]],  # F22
    ]
    for points in tables:
        whole = np.asarray(jax.jit(lambda a, ea, F1: _dd_lattice(a, ea, F1, points))(a, ea, F1))
        rows = np.asarray(jax.jit(lambda a, ea, F1: _dd_lattice_rows(a, ea, F1, points, block))(
            a, ea, F1))
        assert rows.shape == whole.shape == batch + jnp.broadcast_shapes(
            *(p.shape for p in points))
        np.testing.assert_array_equal(rows, whole)
