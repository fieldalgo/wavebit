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
        shift = xp.stack(
            [m * basis[0][0] + n * basis[0][1], m * basis[1][0] + n * basis[1][1]]
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
        separation = []
        for (a0, e0, lo, hi), (a1, e1, _, _) in ((sides[0], sides[1]), (sides[1], sides[0])):
            for t in (lo, hi):
                x = a0 + t * e0 - a1
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
        gap = facing_gap(xp, b.reduced_basis(lattice, transform))
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

    result = b._chunked(xp, evaluate, pts, chunk or 256, 3, A, E, contrasts)
    return tuple(xp.reshape(result[:, i], grid) for i in range(3))
