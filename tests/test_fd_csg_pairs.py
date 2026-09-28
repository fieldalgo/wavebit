"""The CSG pair predicate against an independent primal linear program."""
import numpy as np
import pytest
from scipy.optimize import linprog

from wavebit.fd._runtime import runtime
from wavebit.fd._shapes._bodies import _no_common_volume


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch', 'jit'))
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_pair_separation_matches_primal_volume_feasibility(backend, dtype):
    rng = np.random.default_rng(20260928)
    normals = rng.normal(size=(96, 2, 3))
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    distances = rng.uniform(-.75, .75, (96, 2))
    expected = []
    for normal, distance in zip(normals, distances):
        # Minimize the maximum signed plane value over the unit voxel.
        # This primal LP uses box coordinates, not the kernel's dual knots.
        result = linprog([0., 0., 0., 1.],
                         A_ub=np.column_stack([normal, -np.ones(2)]),
                         b_ub=-distance, bounds=[(-.5, .5)]*3+[(None, None)],
                         method='highs')
        assert result.success
        # Exclude solver-boundary ambiguity: the minimum |optimum| measured
        # on 2026-09-28 is .004070378091359254, safely above the LP's 1e-7
        # default feasibility tolerance. Keep a 1e-4 fixture guard below it.
        # This is a fixture check, not a tolerance on the boolean predicate.
        assert abs(result.fun) > 1e-4
        expected.append(result.fun >= 0)
    # Binary-exact boundaries and their two neighbouring representable
    # values: empty contact must stay distinct from positive overlap.
    precision = getattr(np, dtype)
    lower = np.nextafter(precision(.25), precision(0.))
    upper = np.nextafter(precision(.25), precision(1.))
    for axis in range(3):
        n = np.eye(3)[axis]
        for d, apart in ((lower, False), (.25, True), (upper, True)):
            normals = np.concatenate([normals, [[n, -n]]])
            distances = np.concatenate([distances, [[d, -.25]]])
            expected.append(apart)
    # Parallel planes put the answer at an endpoint, with no interior knot.
    for d_a, d_b, apart in ((.75, 0., True), (0., .75, True),
                            (0., 0., False), (-.75, -.75, False)):
        normals = np.concatenate([normals, [[[1., 0., 0.], [1., 0., 0.]]]])
        distances = np.concatenate([distances, [[d_a, d_b]]])
        expected.append(apart)
    rt = runtime('jax' if backend == 'jit' else backend)
    xp = rt.xp
    values = [xp.asarray(v, dtype=getattr(xp, dtype)) for v in
              (distances[:, 0], normals[:, 0], distances[:, 1], normals[:, 1])]
    evaluate = lambda a, b, c, d: _no_common_volume(a, b, c, d, 1., xp)
    if backend == 'jit':
        import jax
        evaluate = jax.jit(evaluate)
    np.testing.assert_array_equal(np.asarray(evaluate(*values)), expected)
    # Swapping the children is the same feasibility problem.
    np.testing.assert_array_equal(np.asarray(evaluate(*values[2:], *values[:2])), expected)
