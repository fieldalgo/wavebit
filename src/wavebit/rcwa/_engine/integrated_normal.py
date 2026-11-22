"""Finite-support boundary integrals for the scalar painted normal projector.

Each physical interval contributes its normal dyad times
    contrast**4 * integral (1 / |p-q|**2 - 1 / radius**2)**2 ds,
inside the radius and zero outside. This positive measure is additive under straight
subdivision and tends to zero with a disappearing interval away from its endpoint.
The kernel and its first derivative vanish at the support boundary. Existing image
bounds retain every segment within half the reduced cell's longest diagonal.

Two intervals bounding a sliver thinner than ``interfaces.facing_gap`` are blended toward
the single interval its closing leaves (:func:`_facing_corrections`): without that, a
contact replaced two weights by one at once and stepped the field everywhere.

An inverse-distance pole recovers the exact normal on an interface. Evaluate all
integrals in units of the support radius and multiply by the common nearest distance
cubed; that common scale cancels when the dyads are normalised. A series avoids
cancellation on extensions of a segment's supporting line.
"""

from wavebit.rcwa._engine import boundary as b
from wavebit._common.geometry.interfaces import facing_gap


_CANDIDATE_CAPACITY = 1024


def _interval_mass(xp, p, a, e, l2, radius, threshold2, valid):
    # Units of the compact-support radius. Integrate (1/r^2 - 1)^2 along each line.
    norm = xp.sqrt(l2)
    t = e / norm[:, None]
    v = (p[:, None, :] - a[None, :, :]) / radius
    s0 = -xp.sum(v * t[None, :, :], axis=-1)
    h2 = (v[:, :, 0] * t[None, :, 1] - v[:, :, 1] * t[None, :, 0]) ** 2
    half = xp.sqrt(xp.maximum(1 - h2, xp.full_like(h2, 1e-30)))
    lo = xp.maximum(s0, -half)
    hi = xp.minimum(s0 + norm[None, :] / radius, half)
    length = xp.maximum(hi - lo, xp.zeros_like(lo))
    live = (length > 0) & (h2 < 1) & valid[None, :]
    # Scale all weights by the closest distance cubed before evaluating the pole.
    d2 = xp.where(lo > 0, h2 + lo * lo, xp.where(hi < 0, h2 + hi * hi, h2))
    minimum = xp.min(xp.where(live, d2, 1.0), axis=1)
    on = minimum <= threshold2 / radius**2
    d = xp.sqrt(xp.where(on, 1.0, minimum))[:, None]
    # A split endpoint owns half the line pole; two halves equal an unsplit line,
    # including a grid sample exactly at a junction of different normals.
    tolerance = xp.sqrt(threshold2) / radius
    pole = xp.where((lo < -tolerance) & (hi > tolerance), 1.0, 0.5)
    lo = xp.where(on[:, None], 0.0, lo)
    hi = xp.where(on[:, None], 0.0, hi)
    length = xp.where(on[:, None], 0.0, length)
    h2 = xp.where(on[:, None], 1.0, h2)
    D = h2 + lo * hi
    small = (D > 0) & (h2 * length**2 < 0.01 * D**2)
    ds = xp.where(small, D, 1.0)
    x2 = h2 * (length / ds) ** 2
    # Stable atan(x)/x and (atan(x)-x/(1+x*x))/(2*x**3).
    # Horner form avoids repeated generic powers on NumPy and bounds the trace.
    f = xp.full_like(x2, 9 / 19)
    g = xp.full_like(x2, 1 / 17)
    for k in range(7, -1, -1):
        f = (k + 1) / (2 * k + 3) - x2 * f
        g = 1 / (2 * k + 1) - x2 * g
    w2small = d**3 * (length / ds) * g
    w4small = d**3 * length / (ds**2 + h2 * length**2) + (d * length / ds) ** 3 * f
    h = xp.sqrt(xp.where(small | on[:, None] | ~live, 1.0, h2))
    angle = xp.atan2(h * length, h * h + lo * hi)
    denom0 = h * h + lo * lo
    denom1 = h * h + hi * hi
    w2large = d**3 * angle / h
    w4large = (d / h) ** 3 * (angle + h * (hi / denom1 - lo / denom0)) / 2
    w2 = xp.where(small, w2small, w2large)
    w4 = xp.where(small, w4small, w4large)
    mass = xp.maximum(w4 - 2 * w2 + d**3 * length, xp.zeros_like(w4))
    # The singular limit only retains the interfaces through the sample.
    mass = xp.where(
        on[:, None], pole * xp.astype(d2 <= threshold2 / radius**2, p.dtype), mass
    )
    return xp.where(live, mass, 0.0)


def _smootherstep(xp, x):
    x = xp.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (6.0 * x - 15.0))


def _facing_corrections(xp, A, E, L2, contrasts, merged, plan, basis, gap):
    """Signed sub-interval weights that carry each facing pair toward its closed state.

    For a facing pair ``(p, q, m, n)`` bounding a sliver of region Y between X and Z, the
    part of ``p`` covered by ``q``'s projection, and the part of ``q`` covered by ``p``'s,
    are weighted ``s * c**4 + (1 - s) * c_XZ**4 / 2`` instead of ``c**4``. ``s`` rises from
    zero at contact to one at the separation ``gap`` with a C2 smootherstep, so a pair at
    or beyond that separation is unchanged. At contact the two halves add to the single
    X|Z interval the plan then holds. Straight subdivision leaves the mass integral
    unchanged, so weighting a sub-interval is the same as splitting the interval there.

    Returns per pair side: parent piece, start and end fractions on it, and the weight to
    add over that sub-interval (the blended weight minus the parent's own).
    """
    parents, lows, highs, weights = [], [], [], []
    for k, (p, q, m, n) in enumerate(plan.facing):
        # The basis entries are plain floats for a concrete cell and arrays for a traced one.
        shift = xp.stack(
            [
                xp.asarray(m * basis[0][0] + n * basis[0][1], dtype=A.dtype),
                xp.asarray(m * basis[1][0] + n * basis[1][1], dtype=A.dtype),
            ]
        )
        ap, ep, aq, eq = A[p], E[p], A[q] + shift, E[q]
        sides = []
        for a0, e0, a1, e1 in ((ap, ep, aq, eq), (aq, eq, ap, ep)):
            # Fractions of the other interval's ends along this one, clipped to it.
            t0 = xp.sum((a1 - a0) * e0) / xp.sum(e0 * e0)
            t1 = xp.sum((a1 + e1 - a0) * e0) / xp.sum(e0 * e0)
            lo = xp.clip(xp.minimum(t0, t1), 0.0, 1.0)
            hi = xp.clip(xp.maximum(t0, t1), 0.0, 1.0)
            sides.append((a0, e0, lo, hi))
        # The ends of each whole interval against the other's line, as the planner measures it
        # (`interfaces._facing_separation`).
        separation = []
        for (a0, e0, _, _), (a1, e1, _, _) in ((sides[0], sides[1]), (sides[1], sides[0])):
            for x in (a0 - a1, a0 + e0 - a1):
                separation.append(xp.abs(x[0] * e1[1] - x[1] * e1[0]) / xp.sqrt(xp.sum(e1 * e1)))
        blend = 1.0 - _smootherstep(xp, xp.max(xp.stack(separation)) / gap)
        closed = merged[k] ** 4 / 2
        for piece, (_, _, lo, hi) in zip((p, q), sides):
            parents.append(piece)
            lows.append(lo)
            highs.append(hi)
            weights.append(blend * (closed - contrasts[piece] ** 4))
    return parents, xp.stack(lows), xp.stack(highs), xp.stack(weights)


def projector_fields(
    xp,
    grid,
    lattice,
    dtype,
    segment_data,
    contrasts,
    plan,
    transform=None,
    chunk=None,
    facing_contrasts=None,
):
    A, E, L2, _active = segment_data
    ids = [i for g in plan.groups for i in g]
    if not ids:
        return (
            xp.ones(grid, dtype=dtype),
            xp.zeros(grid, dtype=dtype),
            xp.zeros(grid, dtype=dtype),
        )
    A, E, L2 = b.periodic_segments(
        xp, A, E, L2, lattice, transform, plan.images, plan.groups
    )
    pairs = [tuple(sorted((plan.pieces[i].left, plan.pieces[i].right))) for i in ids]
    keys = sorted(set(pairs))
    labels = [keys.index(x) for x in pairs]
    scale = b._cell_scale(xp, lattice, dtype)
    repeat = b._continued_endpoints(
        xp, A, E, L2, xp.sum(E * E, axis=1) > 0, labels, scale
    )[2]
    strength = contrasts[xp.asarray(ids)]
    distinct = xp.asarray([pair[0] != pair[1] for pair in pairs])
    valid = (strength > 0) & distinct & (xp.sum(E * E, axis=1) > 0) & ~repeat
    t = E / xp.sqrt(L2)[:, None]
    nx, ny = t[:, 1], -t[:, 0]
    contrast = strength**4
    basis = b.reduced_basis(lattice, transform)
    basis = xp.stack(
        [
            xp.stack([xp.asarray(basis[i][j], dtype=dtype) for j in range(2)])
            for i in range(2)
        ]
    )
    radius = 0.5 * xp.sqrt(
        xp.maximum(
            xp.sum((basis[:, 0] + basis[:, 1]) ** 2),
            xp.sum((basis[:, 0] - basis[:, 1]) ** 2),
        )
    )
    pts = b._folded_sample_points(xp, grid, lattice, dtype, transform)
    threshold2 = b._on_interface_threshold2(xp, lattice, dtype)

    extra = None
    if plan.facing and facing_contrasts is not None:
        base_A, base_E, base_L2, _ = segment_data
        gap = facing_gap(xp, b.reduced_basis(lattice, transform), dtype)
        parents, lows, highs, weights = _facing_corrections(
            xp, base_A, base_E, base_L2, contrasts, facing_contrasts, plan,
            b.reduced_basis(lattice, transform), gap,
        )
        # Every periodic copy of a parent carries its sub-interval, at the same fractions.
        copies, sides = [], []
        for side, parent in enumerate(parents):
            for position, piece in enumerate(ids):
                if piece == parent:
                    copies.append(position)
                    sides.append(side)
        copies = xp.asarray(copies, dtype=xp.int64)
        sides = xp.asarray(sides, dtype=xp.int64)
        lo, hi = lows[sides], highs[sides]
        sub_A = A[copies] + lo[:, None] * E[copies]
        sub_E = (hi - lo)[:, None] * E[copies]
        sub_L2 = xp.sum(sub_E * sub_E, axis=1)
        sub_weight = weights[sides]
        # A pair at or beyond the gap adds nothing and is left out entirely, so the field of
        # a layer whose slivers are all wider is the field without this correction, bit for bit.
        sub_live = (sub_weight != 0) & (sub_L2 > 0) & ~repeat[copies]
        extra = (
            sub_A,
            sub_E,
            xp.where(sub_L2 > 0, sub_L2, xp.ones_like(sub_L2)),
            sub_live,
            sub_weight,
            nx[copies],
            ny[copies],
        )

    def integrate(p, a, e, l2, live, weight, normal_x, normal_y, indices=None):
        count = a.shape[0]
        if extra is not None:
            a = xp.concat([a, extra[0]], axis=0)
            e = xp.concat([e, extra[1]], axis=0)
            l2 = xp.concat([l2, extra[2]], axis=0)
            live = xp.concat([live, extra[3]], axis=0)
        mass = _interval_mass(xp, p, a, e, l2, radius, threshold2, live)
        if extra is not None:
            # Elementwise but for the minimum distance, which a sub-interval of a live
            # interval never lowers: the parents' masses are those computed without it.
            mass, sub_mass = mass[:, :count], mass[:, count:] * extra[4][None, :]
        mass = mass * weight[None, :]
        if indices is not None:
            # Restore the full ordered reduction, including exact zeros, so NumPy
            # chunk size cannot change its pairwise summation tree by one ulp.
            full_mass = xp.zeros((p.shape[0], len(A)), dtype=mass.dtype)
            full_mass[:, indices] = mass
            mass, normal_x, normal_y = full_mass, nx, ny
        total = xp.sum(mass, axis=1)
        xx = xp.sum(mass * normal_x**2, axis=1)
        xy = xp.sum(mass * normal_x * normal_y, axis=1)
        yy = xp.sum(mass * normal_y**2, axis=1)
        if extra is not None:
            total = total + xp.sum(sub_mass, axis=1)
            xx = xx + xp.sum(sub_mass * extra[5] ** 2, axis=1)
            xy = xy + xp.sum(sub_mass * extra[5] * extra[6], axis=1)
            yy = yy + xp.sum(sub_mass * extra[6] ** 2, axis=1)
        nonempty = total > 0
        total = xp.where(nonempty, total, 1.0)
        return xp.stack(
            [
                xp.where(nonempty, xx / total, 1.0),
                xy / total,
                yy / total,
            ],
            axis=-1,
        )

    def full(p):
        return integrate(p, A, E, L2, valid, contrast, nx, ny)

    lower, upper = xp.minimum(A, A + E), xp.maximum(A, A + E)

    def evaluate(p):
        # A box farther than the support radius contributes exactly zero, including
        # its first derivative. This is an exact sparse evaluation, not a length
        # cutoff. The conservative box test can retain extra intervals harmlessly.
        gap = xp.maximum(lower - xp.max(p, axis=0), xp.min(p, axis=0) - upper)
        gap = xp.maximum(gap, xp.zeros_like(gap))
        keep = valid & (xp.sum(gap**2, axis=1) <= radius**2)
        if "jax" in xp.__name__ and len(A) > _CANDIDATE_CAPACITY:
            # Static-size gather for JIT/AD, with a full fallback for dense scenes.
            # No assumption about a maximum physical interface count is made.
            import jax

            count = xp.sum(xp.astype(keep, xp.int64))
            index = xp.nonzero(keep, size=_CANDIDATE_CAPACITY, fill_value=0)[0]
            selected = xp.arange(_CANDIDATE_CAPACITY) < count
            return jax.lax.cond(
                count <= _CANDIDATE_CAPACITY,
                lambda _: integrate(
                    p,
                    A[index],
                    E[index],
                    L2[index],
                    selected,
                    contrast[index],
                    nx[index],
                    ny[index],
                ),
                lambda _: full(p),
                operand=None,
            )
        if "numpy" in xp.__name__ and "jax" not in xp.__name__:
            index = xp.nonzero(keep)[0]
            if not len(index) and extra is None:
                return xp.stack(
                    [
                        xp.ones_like(p[:, 0]),
                        xp.zeros_like(p[:, 0]),
                        xp.zeros_like(p[:, 0]),
                    ],
                    axis=-1,
                )
            return integrate(
                p,
                A[index],
                E[index],
                L2[index],
                valid[index],
                contrast[index],
                nx[index],
                ny[index],
                indices=index,
            )
        return full(p)

    if (
        "numpy" in xp.__name__
        and "jax" not in xp.__name__
        and _tree_matches_numpy(len(A), pts.dtype)
        and (extra is None or _tree_matches_numpy(extra[0].shape[0], pts.dtype))
    ):
        result = _sparse_fields(
            pts, grid, A, E, L2, valid, contrast, nx, ny, radius, threshold2,
            chunk or 256, extra,
        )
    else:
        result = b._chunked(xp, evaluate, pts, chunk or 256, 3, A, E, contrasts)
    return tuple(xp.reshape(result[:, i], grid) for i in range(3))


# ------------------------------------------------------------ NumPy: live pairs only
#
# The dense NumPy evaluation above forms every (sample, interval) pair of a chunk and sums
# each row over every imaged interval, zeros included, so that the pairwise summation tree --
# and therefore the last bit -- does not depend on which intervals a chunk kept. About half
# of a chunk's kept pairs are inside the support, and a kept interval is a few of the
# thousands imaged, so most of that work multiplies and adds zeros. The functions below
# evaluate the same expressions on the live pairs alone and rebuild NumPy's own summation
# tree around them, which leaves every value bit for bit as the dense evaluation produces it.
#
# NumPy reduces a contiguous float row as ``0 + pairwise(row)``: blocks of at most 128 summed
# in eight interleaved accumulators, a sequential tail, halves split at a multiple of eight.
# Adding a zero changes no nonzero partial sum, so a leaf accumulator is the sequential sum of
# its nonzero entries and an empty subtree contributes an exact zero. Only the sign of an
# all-zero partial sum can differ, and the leading ``0 +`` removes that. Whether the installed
# NumPy still reduces this way is checked once per width against ``numpy.sum`` itself
# (:func:`_tree_matches_numpy`); on any mismatch the dense evaluation runs instead.

_TILE = 8


def _pairwise_plan(n):
    """NumPy's pairwise summation tree for an ``n``-wide row.

    For each column: its leaf; its slot, 0-7 one of the leaf's eight accumulators and 8 the
    sequential tail; and its place in that slot's sequence. Also whether each leaf starts
    with accumulators (it is at least eight wide) and the leaves as nested pairs in the
    order their sums are added."""
    import numpy as np
    leaf = np.empty(n, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    place = np.empty(n, dtype=np.int64)
    eight = []

    def split(lo, m):
        if m <= 128:
            index = len(eight)
            eight.append(m >= 8)
            leaf[lo:lo + m] = index
            body = m - m % 8 if m >= 8 else 0
            slot[lo:lo + body] = np.arange(body) % 8
            place[lo:lo + body] = np.arange(body) // 8
            slot[lo + body:lo + m] = 8
            place[lo + body:lo + m] = np.arange(m - body)
            return index
        half = m // 2
        half -= half % 8
        return (split(lo, half), split(lo + half, m - half))

    tree = split(0, n) if n else None
    return leaf, slot, place, np.asarray(eight, dtype=bool), tree


def _in_place_order(start, key, place, values):
    """Add ``values`` into ``start[key]`` one place at a time, in increasing ``place``.

    Entries sharing a key are a sequence; within one place every key is distinct."""
    import numpy as np
    if not len(key):
        return start
    order = np.argsort(place.astype(np.uint8), kind='stable')
    bounds = np.searchsorted(place[order], np.arange(int(place.max()) + 2))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        at = order[lo:hi]
        start[key[at]] = start[key[at]] + values[at]
    return start


def _tree_sum(rows, columns, values, count, plan):
    """``numpy.sum(dense, axis=-1)`` for each row of ``values``, where ``dense[w]`` is the
    ``(count, n)`` array holding ``values[w, i]`` at ``(rows[i], columns[i])`` and zero
    elsewhere: ``(width, count)``. The pairs must be distinct and in row-major order."""
    import numpy as np
    leaf, slot, place, eight, tree = plan
    width = values.shape[0]
    leaves = len(eight)
    if tree is None or not len(rows):
        return np.zeros((width, count), dtype=values.dtype)
    cell = rows * leaves + leaf[columns]
    first = np.empty(len(cell), dtype=bool)
    first[0] = True
    np.not_equal(cell[1:], cell[:-1], out=first[1:])
    used = cell[first]
    which = np.cumsum(first) - 1
    sl, pl = slot[columns], place[columns]
    body = sl < 8
    # Each accumulator is a sequence over places, in column order; a missing place is an
    # exact zero.
    if values.dtype == np.float64:
        # bincount adds its weights into float64 bins in input order, from zero.
        key = which[body] * 8 + sl[body]
        r = np.zeros((width, len(used) * 8), dtype=values.dtype)
        if body.any():
            r = np.stack([np.bincount(key, weights=v, minlength=len(used) * 8)
                          for v in (values[:, body] if not body.all() else values)])
        r = r.reshape(width, len(used), 8)
    else:
        places = int(pl[body].max()) + 1 if body.any() else 1
        grid = np.zeros((width, len(used) * places * 8), dtype=values.dtype)
        grid[:, (which[body] * places + pl[body]) * 8 + sl[body]] = values[:, body]
        grid = grid.reshape(width, len(used), places, 8)
        r = grid[:, :, 0]
        for k in range(1, places):
            r = r + grid[:, :, k]
        del grid
    res = ((r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])) + (
        (r[..., 4] + r[..., 5]) + (r[..., 6] + r[..., 7]))
    # A leaf narrower than eight is a plain sequential sum from zero.
    res = np.where(eight[used % leaves][None, :], res, np.zeros_like(res))
    tail = ~body
    if tail.any():
        res = _in_place_order(res.T.copy(), which[tail], pl[tail], values[:, tail].T).T
    dense = np.zeros((width, leaves * count), dtype=values.dtype)
    dense[:, (used % leaves) * count + used // leaves] = res
    dense = dense.reshape(width, leaves, count)

    def combine(node):
        if isinstance(node, tuple):
            return combine(node[0]) + combine(node[1])
        return dense[:, node]

    return np.zeros((width, count), dtype=values.dtype) + combine(tree)


_TREE_CHECKED = {}


def _tree_matches_numpy(n, dtype):
    """Whether :func:`_tree_sum` reproduces ``numpy.sum`` on this NumPy, for width ``n``."""
    import numpy as np
    key = (n, np.dtype(dtype).str)
    if key not in _TREE_CHECKED:
        rng = np.random.default_rng(n)
        dense = (rng.standard_normal((6, n))*np.exp(rng.uniform(-30., 30., (6, n)))).astype(dtype)
        dense[rng.random((6, n)) < .8] = 0
        dense[0] = 0
        rows, columns = np.nonzero(dense)
        got = _tree_sum(rows, columns, dense[rows, columns][None, :], 6, _pairwise_plan(n))
        _TREE_CHECKED[key] = bool(np.array_equal(got[0], np.sum(dense, axis=1)))
    return _TREE_CHECKED[key]


def _kept(p, lower, upper, valid, radius):
    """The intervals whose boxes lie within the support radius of ``p``'s box, as the dense
    evaluation keeps them."""
    import numpy as np
    gap = np.maximum(lower - np.max(p, axis=0), np.min(p, axis=0) - upper)
    gap = np.maximum(gap, np.zeros_like(gap))
    return np.nonzero(valid & (np.sum(gap**2, axis=1) <= radius**2))[0]


def _live_mass(p, a, e, l2, radius, threshold2, valid, padded):
    """:func:`_interval_mass` on its live pairs only: ``(rows, columns, mass)``.

    Every expression is :func:`_interval_mass`'s, on the same operands, element by element;
    a branch it discards with ``where`` is evaluated only where it is kept, except that the
    series is also evaluated on the few live pairs that take the closed form, and then
    discarded. ``padded(rows, live)`` says whether the dense evaluation's row also held an
    interval that was kept but not live, whose 1 then entered the row's minimum."""
    import numpy as np
    norm = np.sqrt(l2)
    t = e / norm[:, None]
    width = a.shape[0]
    vx = (p[:, None, 0] - a[None, :, 0]) / radius
    vy = (p[:, None, 1] - a[None, :, 1]) / radius
    # -sum(v * t, axis=-1): NumPy adds the two products into its zero identity.
    s0 = vx * t[None, :, 0]
    s0 += vy * t[None, :, 1]
    s0 += 0.0
    np.negative(s0, out=s0)
    h2 = vx * t[None, :, 1]
    h2 -= vy * t[None, :, 0]
    del vx, vy
    np.square(h2, out=h2)
    half = np.maximum(1 - h2, 1e-30)
    np.sqrt(half, out=half)
    hi = s0 + norm / radius
    np.minimum(hi, half, out=hi)
    np.negative(half, out=half)
    lo = np.maximum(s0, half, out=s0)
    del half
    live = hi > lo  # length = max(hi - lo, 0) > 0
    live &= h2 < 1
    live &= valid[None, :]
    flat = np.flatnonzero(live)
    del live
    rows = flat // width
    columns = flat - rows * width
    lo, hi, h2 = lo.ravel().take(flat), hi.ravel().take(flat), h2.ravel().take(flat)
    length = np.maximum(hi - lo, 0)
    d2 = np.where(lo > 0, h2 + lo * lo, np.where(hi < 0, h2 + hi * hi, h2))
    # The row minimum over live pairs; min is exact in any order. A kept pair that is not
    # live contributed 1 to it.
    count = p.shape[0]
    minimum = np.full(count, np.inf, dtype=d2.dtype)
    if len(rows):
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        minimum[rows[starts]] = np.minimum.reduceat(d2, starts)
    number = np.bincount(rows, minlength=count)
    above = np.nonzero(minimum > 1.0)[0]
    if len(above):
        minimum[above[padded(above, number[above])]] = 1.0
    on = minimum <= threshold2 / radius**2
    d = np.sqrt(np.where(on, 1.0, minimum))
    d3 = d**3
    pinned = on[rows]
    if pinned.any():
        mass = np.empty_like(lo)
        tolerance = np.sqrt(threshold2) / radius
        pole = np.where((lo[pinned] < -tolerance) & (hi[pinned] > tolerance), 1.0, 0.5)
        mass[pinned] = pole * (d2[pinned] <= threshold2 / radius**2).astype(p.dtype)
        free = ~pinned
        mass[free] = _free_mass(lo[free], hi[free], length[free], h2[free],
                                d[rows[free]], d3[rows[free]])
    else:
        mass = _free_mass(lo, hi, length, h2, d[rows], d3[rows])
    return rows, columns, mass


def _free_mass(lo, hi, length, h2, d, d3):
    import numpy as np
    D = h2 + lo * hi
    small = (D > 0) & (h2 * length**2 < 0.01 * D**2)
    ds = np.where(small, D, 1.0)
    q = length / ds
    x2 = h2 * q**2
    f = np.full_like(x2, 9 / 19)
    g = np.full_like(x2, 1 / 17)
    scratch = np.empty_like(x2)
    for k in range(7, -1, -1):
        np.multiply(x2, f, out=scratch)
        np.subtract((k + 1) / (2 * k + 3), scratch, out=f)
        np.multiply(x2, g, out=scratch)
        np.subtract(1 / (2 * k + 1), scratch, out=g)
    w2 = d3 * q * g
    w4 = d3 * length / (ds**2 + h2 * length**2) + (d * length / ds) ** 3 * f
    s = np.flatnonzero(~small)
    if len(s):
        lg, ll, hh, ld3, dl = lo[s], length[s], hi[s], d3[s], d[s]
        h = np.sqrt(h2[s])
        angle = np.arctan2(h * ll, h * h + lg * hh)
        denom0 = h * h + lg * lg
        denom1 = h * h + hh * hh
        w2[s] = ld3 * angle / h
        w4[s] = (dl / h) ** 3 * (angle + h * (hh / denom1 - lg / denom0)) / 2
    return np.maximum(w4 - 2 * w2 + d3 * length, 0)


def _sparse_fields(pts, grid, A, E, L2, valid, contrast, nx, ny, radius, threshold2, chunk,
                   extra=None):
    """The NumPy fields of :func:`projector_fields`, bit for bit, from live pairs only.

    Samples are taken in square tiles, whose boxes keep fewer intervals than the dense
    evaluation's runs of ``chunk`` consecutive samples; ``chunk`` is still consulted for the
    one thing a run's kept set decides, the 1 a kept but dead interval puts in a row's
    minimum.

    ``extra`` is :func:`projector_fields`' facing sub-intervals. The dense evaluation appends
    all of them to every run's kept intervals, so they share each row's minimum, and sums
    their signed masses in a second reduction of their own width; so does this."""
    import numpy as np
    plan = _pairwise_plan(len(A))
    lower, upper = np.minimum(A, A + E), np.maximum(A, A + E)
    flat = np.arange(pts.shape[0]).reshape(grid)
    out = np.empty((pts.shape[0], 3), dtype=pts.dtype)
    kept_in_run = {}

    def padded_for(sel):
        def padded(rows, live):
            runs = sel[rows] // chunk
            for run in set(runs.tolist()):
                if run not in kept_in_run:
                    kept_in_run[run] = len(_kept(pts[run * chunk:(run + 1) * chunk],
                                                 lower, upper, valid, radius))
            kept = np.asarray([kept_in_run[r] for r in runs.tolist()], dtype=np.int64)
            return kept + appended > live
        return padded

    appended = 0 if extra is None else extra[0].shape[0]
    if appended:
        sub_A, sub_E, sub_L2, sub_live, sub_weight, sub_nx, sub_ny = extra
        sub_plan = _pairwise_plan(appended)
        sub_nx2, sub_ny2 = sub_nx**2, sub_ny**2

    nx2, ny2 = nx**2, ny**2
    for i0 in range(0, grid[0], _TILE):
        for j0 in range(0, grid[1], _TILE):
            sel = flat[i0:i0 + _TILE, j0:j0 + _TILE].reshape(-1)
            p = pts[sel]
            index = _kept(p, lower, upper, valid, radius)
            a, e, l2, live = A[index], E[index], L2[index], valid[index]
            if appended:
                a = np.concatenate([a, sub_A], axis=0)
                e = np.concatenate([e, sub_E], axis=0)
                l2 = np.concatenate([l2, sub_L2], axis=0)
                live = np.concatenate([live, sub_live], axis=0)
            rows, columns, mass = _live_mass(p, a, e, l2, radius, threshold2, live,
                                             padded_for(sel))
            if appended:
                parent = columns < len(index)
                sub_rows, sub_columns = rows[~parent], columns[~parent] - len(index)
                sub_mass = mass[~parent] * sub_weight[sub_columns]
                rows, columns, mass = rows[parent], columns[parent], mass[parent]
            columns = index[columns]
            mass = mass * contrast[columns]
            terms = np.empty((4, len(mass)), dtype=mass.dtype)
            terms[0] = mass
            np.multiply(mass, nx2[columns], out=terms[1])
            np.multiply(mass, nx[columns], out=terms[2])
            terms[2] *= ny[columns]
            np.multiply(mass, ny2[columns], out=terms[3])
            sums = _tree_sum(rows, columns, terms, len(sel), plan)
            if appended:
                terms = np.empty((4, len(sub_mass)), dtype=sub_mass.dtype)
                terms[0] = sub_mass
                np.multiply(sub_mass, sub_nx2[sub_columns], out=terms[1])
                np.multiply(sub_mass, sub_nx[sub_columns], out=terms[2])
                terms[2] *= sub_ny[sub_columns]
                np.multiply(sub_mass, sub_ny2[sub_columns], out=terms[3])
                sums = sums + _tree_sum(sub_rows, sub_columns, terms, len(sel), sub_plan)
            total = sums[0]
            nonempty = total > 0
            total = np.where(nonempty, total, 1.0)
            out[sel] = np.stack([np.where(nonempty, sums[1] / total, 1.0),
                                 sums[2] / total, sums[3] / total], axis=-1)
    return out
