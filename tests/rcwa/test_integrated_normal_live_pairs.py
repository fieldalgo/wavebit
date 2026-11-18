"""The NumPy painted normal field from live pairs only is the dense evaluation, bit for bit.

``integrated_normal._sparse_fields`` and ``boundary._continued_in_pair_near`` skip work whose
result is an exact zero or an exact false. Neither may move a value by one ulp, so every
comparison here is ``array_equal`` against the dense evaluation they replace, which is still
the path JAX and PyTorch take and the fallback NumPy takes if its summation changes.
"""

import numpy as np
import pytest
import array_api_compat.numpy as xp

from wavebit.rcwa._engine import boundary as b
from wavebit.rcwa._engine import integrated_normal as inormal


def dense_tree(monkeypatch):
    monkeypatch.setattr(inormal, "_tree_matches_numpy", lambda n, dtype: False)


def dense_pairs(monkeypatch):
    monkeypatch.setattr(b, "_continued_in_pair_near", lambda *args: None)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 5, 8, 13, 64, 128, 129, 554, 1000, 8864, 20001])
def test_tree_sum_reproduces_numpy_sum(n, dtype):
    # The live-pair path relies on NumPy reducing a row as 0 + pairwise(row). If a NumPy
    # release changes that, the path falls back to the dense evaluation (still exact) and this
    # test says the speed went with it.
    assert inormal._tree_matches_numpy(n, dtype)
    rng = np.random.default_rng(n)
    for density in (0.002, 0.05, 0.5, 1.0):
        dense = (rng.standard_normal((9, n)) * np.exp(rng.uniform(-8.0, 8.0, (9, n)))).astype(
            dtype
        )
        dense[rng.random((9, n)) >= density] = 0
        dense[1] = 0
        rows, columns = np.nonzero(dense)
        values = np.stack([dense[rows, columns], -dense[rows, columns] ** 2])
        got = inormal._tree_sum(rows, columns, values, 9, inormal._pairwise_plan(n))
        np.testing.assert_array_equal(got[0], np.sum(dense, axis=1))
        np.testing.assert_array_equal(got[1], np.sum(-(dense**2), axis=1))


def circles(cell):
    return [
        dict(type="circle", lx=300.0, cx=-60.0, cy=0.0, material=4.0),
        dict(type="circle", lx=240.0, cx=60.0, cy=0.0, material=9.0),
    ]


SCENES = {
    "overlapping circles": (((600.0, 0.0), (0.0, 600.0)), circles),
    # Axis-aligned edges put samples exactly on interfaces (the pinned rows) and exactly at
    # junctions; the ellipse adds a closed-form row next to them.
    "boxes on the grid": (
        ((640.0, 0.0), (0.0, 640.0)),
        lambda cell: [
            # On the 32 x 32 grid samples sit at 10 + 20 k.
            dict(type="rectangle", lx=340.0, ly=180.0, cx=0.0, cy=0.0, material=4.0),
            dict(type="rectangle", lx=160.0, ly=320.0, cx=90.0, cy=50.0, material=9.0),
            dict(type="ellipse", lx=200.0, ly=90.0, angle=0.4, cx=-150.0, cy=-170.0,
                 material=2.0 + 0.1j),
        ],
    ),
    # A skewed cell folds samples onto a reduced cell whose tiles straddle the fold.
    "skewed cell": (
        ((500.0, 0.0), (230.0, 450.0)),
        lambda cell: [
            dict(type="circle", lx=260.0, cx=100.0, cy=60.0, material=6.25),
            dict(type="rectangle", lx=200.0, ly=120.0, angle=0.3, cx=180.0, cy=170.0,
                 material=2.25),
        ],
    ),
}


def fields(cell, shapes, grid, dtype, chunk=None):
    transform = b.reduction_transform(cell)
    recs, loops, transform, plan = b.plan_layer(shapes, cell)
    verts = [b.shape_vertices(xp, s, cell) for s in shapes]
    vals = b._clipper.eval_table(xp, recs, verts)
    regions = [
        (s["material"], [b._clipper.loop_verts(xp, lp, vals) for lp in ll])
        for s, ll in zip(shapes, loops)
    ]
    segments = b.replay_interfaces(xp, regions, 1.0, cell, transform, plan)
    segments = tuple(np.asarray(x, dtype=dtype) if x.dtype.kind == "f" else x for x in segments)
    contrasts = np.asarray(b.interface_contrasts(xp, regions, 1.0, plan, dtype), dtype=dtype)
    return np.stack(
        b.material_projector_fields(
            xp, grid, cell, dtype, segments, contrasts, plan, transform, chunk=chunk
        )
    )


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("grid", [(32, 32), (30, 41)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_live_pair_fields_are_the_dense_fields(monkeypatch, scene, grid, dtype):
    cell, shapes = SCENES[scene]
    got = fields(cell, shapes(cell), grid, dtype)
    with monkeypatch.context() as patch:
        dense_tree(patch)
        dense_pairs(patch)
        want = fields(cell, shapes(cell), grid, dtype)
        # The dense rows do not depend on the run length either.
        np.testing.assert_array_equal(fields(cell, shapes(cell), grid, dtype, chunk=7), want)
    np.testing.assert_array_equal(got, want)
    # The scene reaches the branches the live path splits apart.
    assert np.isfinite(got).all()


def test_live_pair_layer_matrices_are_the_dense_ones(monkeypatch):
    cell, shapes = SCENES["overlapping circles"]
    args = (np, shapes(cell), 1.0, 2, 2, cell, np.complex128)
    got = b.to_conv_mat_boundary_layer(*args, shape_nvf_grid_size=64)
    dense_tree(monkeypatch)
    dense_pairs(monkeypatch)
    want = b.to_conv_mat_boundary_layer(*args, shape_nvf_grid_size=64)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize("seed", range(4))
def test_live_mass_is_the_dense_interval_mass(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, (60, 2))
    e = rng.uniform(-0.3, 0.3, (60, 2))
    e[::7] = e[1::7][: len(e[::7])] * 1e-7  # nearly vanishing
    a[3::11] = a[2::11][: len(a[3::11])] + e[2::11][: len(a[3::11])]  # shared endpoints
    l2 = np.sum(e * e, 1)
    valid = rng.random(60) < 0.9
    p = np.concatenate(
        [
            rng.uniform(-1.5, 1.5, (200, 2)),
            a[:20] + 0.37 * e[:20],  # on an interval
            a[20:30],  # at an endpoint
            [[40.0, 40.0]],  # beyond every support
        ]
    )
    threshold2 = (64 * np.finfo(float).eps) ** 2
    want = inormal._interval_mass(np, p, a, e, l2, 1.0, threshold2, valid)
    rows, columns, mass = inormal._live_mass(
        p, a, e, l2, 1.0, threshold2, valid, lambda r, live: live < len(a)
    )
    got = np.zeros_like(want)
    got[rows, columns] = mass
    np.testing.assert_array_equal(got, want)


def random_intervals(rng, kind, m):
    if kind == "split polygon":
        t = np.sort(rng.uniform(0, 1, m + 1))
        pts = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], -1) * 300
        pts = np.round(pts) if rng.random() < 0.5 else pts
        a, e = pts[:-1], pts[1:] - pts[:-1]
        back = rng.integers(0, m, m // 4)  # retraced, reversed
        part = rng.integers(0, m, m // 4)  # split pieces of one line
        a = np.concatenate([a, a[back] + e[back], a[part]])
        e = np.concatenate([e, -e[back], e[part] * rng.uniform(0.2, 0.8, (len(part), 1))])
    elif kind == "grid lines":
        a = rng.integers(0, 5, (m, 2)) * 100.0
        length = rng.integers(1, 4, m) * 50.0
        horizontal = rng.random(m) < 0.5
        e = np.where(
            horizontal[:, None],
            np.stack([length, 0 * length], -1),
            np.stack([0 * length, length], -1),
        )
    else:
        a = rng.uniform(-300, 300, (m, 2))
        e = rng.uniform(-50, 50, (m, 2))
        shared = rng.integers(0, m, m // 3)
        source = rng.integers(0, m, len(shared))
        # Some start on another interval's line, along it; some at its start, any direction.
        a[shared] = a[source] + e[source] * rng.integers(0, 3, (len(shared), 1)) / 2
        along = rng.random(len(shared)) < 0.5
        e[shared[along]] = e[source[along]] * rng.uniform(0.3, 2.0, (along.sum(), 1))
    return a, e


@pytest.mark.parametrize("kind", ["split polygon", "grid lines", "scattered"])
def test_continued_endpoints_near_pairs_are_the_dense_ones(kind):
    rng = np.random.default_rng(len(kind))
    tol2 = (b._SAME_POINT * 600.0) ** 2
    hits = 0
    for _ in range(40):
        a, e = random_intervals(rng, kind, int(rng.integers(2, 300)))
        l2 = np.sum(e * e, 1)
        l2 = np.where(l2 > 0, l2, 1.0)
        valid = (np.sum(e * e, 1) > 0) & (rng.random(len(a)) < 0.9)
        got = b._continued_in_pair_near(a, e, l2, valid, tol2)
        assert got is not None
        for rows in (len(a), max(1, len(a) // 7)):
            want = b._continued_in_pair(xp, a, e, l2, valid, tol2, rows=rows)
            for g, w in zip(got, want):
                np.testing.assert_array_equal(g, w)
        hits += sum(int(np.sum(g)) for g in got)
    assert hits > 0


def test_continued_endpoints_near_pairs_defer_a_valid_zero_length_interval():
    a = np.array([[0.0, 0.0], [5.0, 5.0]])
    e = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert b._continued_in_pair_near(a, e, np.ones(2), np.ones(2, dtype=bool), 1e-12) is None
