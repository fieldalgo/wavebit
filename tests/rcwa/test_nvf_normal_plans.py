"""The rebuild a replay differentiates is the construction it replays.

``normal_plans`` rebuilds a raster normal field from the choices its construction recorded, in a
cell that carries a derivative. The solve-level tests (``test_nvf_record_under_a_cell_derivative``)
judge the derivative that reaches a loss; these judge the field itself, where a drift between
the rebuild and the NumPy construction would show first: the rebuilt value has to be the
construction's to rounding, and its Jacobian in the cell has to be the construction's central
difference -- on both of ``'smooth'``'s branches, on the delegations, and on the cells that
send each construction somewhere else.
"""

import numpy as np
import pytest

from wavebit.rcwa._engine import normal_methods as nm
from wavebit.rcwa._engine import normal_plans
from wavebit.rcwa._engine.factorization import _interface_mask
from wavebit.rcwa._engine.factorization import normal_vector_field

pytestmark = pytest.mark.shard_unit(per='test')

#: A skewed cell and an axis-aligned one with non-square pixels: ``'smooth'`` takes its
#: transform branch on the first and its separable one on the second. Neither has pixel steps
#: in a small rational ratio, which would make two interface pixels exactly as near as each
#: other somewhere -- a kink in the field, where a derivative is one-sided (1000 x 600 on a
#: 16-pixel grid is 62.5 : 37.5 = 5 : 3, and has them).
CELLS = {'skewed': np.array([[1000., 40.], [0., 900.]]),
         'rectangular': np.array([[1000., 0.], [0., 613.]])}


def _disc(n=24):
    eps = np.ones((n, n))
    j, i = np.mgrid[0:n, 0:n]
    eps[(i - 0.4 * n) ** 2 + (j - 0.55 * n) ** 2 < (0.3 * n) ** 2] = 12.25
    return eps


def _wrapped_bar(n=20):
    """A bar across the wrap in v and a step across it in u."""
    eps = np.ones((n, n))
    eps[-3:, :] = 4.0
    eps[:4, :] = 4.0
    eps[:, 5:9] = 4.0
    return eps


def _three_phases(n=16):
    eps = np.ones((n, n))
    eps[2:7, 2:9] = 4.0
    eps[9:14, 6:13] = 9.0
    return eps


SCENES = {'disc': _disc, 'wrapped_bar': _wrapped_bar, 'three_phases': _three_phases}


def _jax():
    pytest.importorskip('jax')
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    import jax
    import jax.numpy as jnp
    return jax, jnp


def _record(eps, cell, method, magnitude=False):
    mask = _interface_mask(eps)
    decisions = {}
    field = normal_vector_field(eps, mask, lattice=cell, method=method,
                                with_magnitude=magnitude, plan=decisions)
    return field, normal_plans.plan_for(decisions, mask, cell, eps.shape, magnitude)


def _construction(eps, cell, method, magnitude=False):
    return np.stack(normal_vector_field(eps, _interface_mask(eps), lattice=cell, method=method,
                                        with_magnitude=magnitude))


def _rebuilt(field, plan, cell, shape, magnitude=False):
    """The rebuild alone (no carry), as a stacked array, in a traced JAX cell -- the only cell
    a replay ever rebuilds in. A NumPy ``cell`` is traced here under ``jax.jit``."""
    jax, jnp = _jax()

    def rebuild(a):
        rebuilt = normal_plans._rebuilt(jnp, plan, a, shape, field, magnitude)
        return jnp.stack([jnp.asarray(f) if r is None else r for f, r in zip(field, rebuilt)])
    if isinstance(cell, np.ndarray):
        return jax.jit(rebuild)(jnp.asarray(cell))
    return rebuild(cell)


@pytest.mark.parametrize('magnitude', [False, True])
@pytest.mark.parametrize('scene', sorted(SCENES))
@pytest.mark.parametrize('cell', sorted(CELLS))
@pytest.mark.parametrize('method', ['distance', 'spectral', 'smooth'])
def test_the_rebuild_is_the_construction_and_differentiates_like_it(method, cell, scene,
                                                                     magnitude):
    """Value to rounding, Jacobian to a central difference of the NumPy construction."""
    jax, jnp = _jax()
    eps, A = SCENES[scene](), CELLS[cell]
    field, plan = _record(eps, A, method, magnitude)
    if scene == 'three_phases':
        assert plan.direction == 'distance'          # the delegation is recorded as what ran
    else:
        assert plan.direction == method
    value = np.asarray(_rebuilt(field, plan, A, eps.shape, magnitude))
    np.testing.assert_allclose(value, np.stack(field), rtol=0, atol=1e-9)

    jac = np.asarray(jax.jacfwd(lambda a: _rebuilt(field, plan, a, eps.shape, magnitude))(
        jnp.asarray(A)))
    # A pixel the construction left without a direction -- exactly (0, 0), on a skeleton --
    # becomes a unit vector under any change of the cell: the field jumps there, by 1/(2h) in a
    # central difference, and has no derivative to compare. The rebuild keeps it at (0, 0).
    flat = (field[0] == 0) & (field[1] == 0)
    assert flat.mean() < 0.15, flat.mean()
    centre = _construction(eps, A, method, magnitude)
    h = 1e-3
    for r, c in ((0, 0), (0, 1), (1, 1)):
        if cell == 'rectangular' and (r, c) == (0, 1):
            continue                                   # a shear would leave the separable branch
        Ap, Am = A.copy(), A.copy()
        Ap[r, c] += h
        Am[r, c] -= h
        ahead, behind = _construction(eps, Ap, method, magnitude), _construction(eps, Am, method,
                                                                               magnitude)
        fd = (ahead - behind) / (2 * h)
        fd[:2, flat] = jac[:2, flat, r, c]
        scale = float(np.max(np.abs(fd)))
        assert scale > 1e-6, (r, c)
        # Each pixel's allowance includes its own difference's error, gauged by how far its two
        # one-sided differences disagree. It matters where ``|g|`` sits a millionth of its peak,
        # just above ``'smooth'``'s snap to zero: normalising it multiplies the construction's
        # own rounding by ~1e6, which a 2e-3 step turns into ~5e-8 of difference -- 1e-5 of the
        # derivative, where a wrong rebuild misses by a sizeable fraction of it.
        spread = np.abs((ahead - centre) - (centre - behind)) / h
        spread[:2, flat] = 0.0
        assert np.all(np.abs(jac[..., r, c] - fd) <= 1e-5 * scale + 2 * spread), (r, c)


def test_the_value_handed_on_is_the_record_bit_for_bit():
    """The rebuilt field contributes its tangent only: what the projector reads is the record."""
    jax, jnp = _jax()
    eps, A = _disc(), CELLS['skewed']
    for method in ('distance', 'spectral', 'smooth'):
        field, plan = _record(eps, A, method, True)

        def handed_on(a):
            return normal_plans.replayed(field, plan, a, eps.shape, True)
        eager = jax.jvp(handed_on, (jnp.asarray(A),), (jnp.ones((2, 2)),))[0]
        for got in (eager, jax.jit(handed_on)(jnp.asarray(A))):
            for g, f in zip(got, field):
                assert np.array_equal(np.asarray(g), f), method
        assert handed_on(A) is field                   # a constant cell is not touched at all


def test_a_one_row_raster_carries_the_reciprocal_direction():
    """A one-cell axis records ``'axis'``: the normal is the cell's reciprocal vector, rebuilt."""
    jax, jnp = _jax()
    eps = np.ones((1, 12))
    eps[0, 3:7] = 4.0
    A = CELLS['skewed']
    field, plan = _record(eps, A, 'contour')
    assert plan.direction == 'axis' and plan.offsets is None
    value = np.asarray(_rebuilt(field, plan, A, eps.shape))
    np.testing.assert_allclose(value, np.stack(field), atol=1e-15)
    jac = np.asarray(jax.jacfwd(lambda a: _rebuilt(field, plan, a, eps.shape))(jnp.asarray(A)))
    h = 1e-3
    Ap, Am = A.copy(), A.copy()
    Ap[0, 1] += h
    Am[0, 1] -= h
    fd = (_construction(eps, Ap, 'contour') - _construction(eps, Am, 'contour')) / (2 * h)
    np.testing.assert_allclose(jac[..., 0, 1], fd, rtol=1e-6, atol=1e-12)


def test_what_does_not_depend_on_the_cell_records_nothing_to_rebuild():
    eps = np.full((12, 12), 2.25)
    for method in ('distance', 'contour', 'spectral', 'smooth'):
        field, plan = _record(eps, CELLS['skewed'], method, True)
        assert plan.direction == 'constant' and plan.offsets is None, method
    field, plan = _record(_disc(), CELLS['skewed'], 'contour')
    assert plan.direction == 'contour'


@pytest.mark.parametrize('cell', sorted(CELLS))
def test_the_recorded_nearest_pixel_measures_the_signed_distance(cell):
    """``_signed_distance_offsets`` is the choice ``_periodic_signed_distance`` made: its length
    in the metric is that distance, on the transform and on the kd-tree alike."""
    eps = _wrapped_bar()
    inside = nm._two_phase_indicator(eps).astype(bool)
    M = nm._metric(CELLS[cell], eps.shape)
    d_v, d_u = nm._signed_distance_offsets(inside, M)
    length = np.hypot(M[0, 0] * d_u + M[0, 1] * d_v, M[1, 0] * d_u + M[1, 1] * d_v)
    np.testing.assert_allclose(np.where(inside, -length, length),
                               nm._periodic_signed_distance(inside, M), rtol=0, atol=1e-12)


@pytest.mark.parametrize('n, sigma', [(16, 1.3), (16, 3.7), (10, 4.9), (9, 7.2)])
def test_the_circulants_are_scipy_s_periodic_gaussian(n, sigma):
    """Order 0 and 1, with a radius up to three times the axis: ``'wrap'`` folds the kernel."""
    from scipy.ndimage import gaussian_filter1d
    f = np.random.default_rng(0).normal(size=(n, 5))
    radius = int(4.0 * sigma + 0.5)
    smooth, slope = normal_plans._circulants(np, sigma, radius, n, f)
    for order, C in ((0, smooth), (1, slope)):
        np.testing.assert_allclose(C @ f, gaussian_filter1d(f, sigma, axis=0, order=order,
                                                            mode='wrap'), rtol=0, atol=1e-15)
