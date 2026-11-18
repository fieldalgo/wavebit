"""Actual polygon interfaces: concrete interval topology and differentiable replay.

Resolved loops are oriented with material on their left (including clockwise holes).
Split collinear edges at every overlapping endpoint and sum signed coverage per region
to recover the two sides. This cancels zero-area retraces as well as shared contours,
without allowing a retrace to erase a third, genuine exterior edge. Compare the two
material values live during replay.
The plan stores endpoint references and integer images, never coordinates or materials.
"""
from collections import namedtuple
import math
import numpy as np

from wavebit._common.backend import checked_float
from wavebit.materials.response import as_material
from wavebit.lattice import reduced_basis


class Interval(namedtuple('Interval', 'edge lower upper left right centroid_sign')):
    """Atomic oriented segment; len(regions) denotes background on either side."""
    __slots__ = ()


class InterfacePlan(namedtuple('InterfacePlan', 'pieces images groups facing',
                               defaults=((),))):
    """Pure topology: intervals, bounded reduced-basis images, and indices per image.

    An interval's centroid_sign selects one oriented representative of each exterior
    interval of the unwrapped union; that area contour includes periodic abutting
    faces. It is separate from the material-jump mask used by the normal field.

    ``facing`` lists ``(p, q, m, n)``: pieces ``p`` and ``q`` (``q`` translated by the
    reduced-basis image ``(m, n)``) bound one sliver of a single region, on the right side
    of both, and lie within :func:`facing_gap` of each other over their common extent.
    Closing that sliver turns the two intervals into one; see :func:`facing_contrasts`.
    """
    __slots__ = ()


#: The facing-gap scale as a fraction of the longest reduced basis vector: about one
#: sample of the default normal grid (1024 per axis). A sliver this thin or thinner has its
#: two bounding intervals blended toward the single interval its closing leaves.
FACING_GAP_FRACTION = 2.0 ** -10


def facing_gap(xp, basis):
    """The width below which two facing intervals are blended: :data:`FACING_GAP_FRACTION`
    of the longest reduced basis vector. ``basis`` holds the vectors as columns."""
    first = basis[0][0] * basis[0][0] + basis[1][0] * basis[1][0]
    second = basis[0][1] * basis[0][1] + basis[1][1] * basis[1][1]
    return FACING_GAP_FRACTION * xp.sqrt(xp.maximum(xp.asarray(first), xp.asarray(second)))


def facing_contrasts(xp, regions, background, plan, dtype):
    """Contrast between the two outer materials of each facing pair in ``plan.facing``.

    A sliver of region Y between X and Z closes into one X|Z interval. That interval's
    contrast is what the two bounding X|Y and Y|Z intervals tend to as the sliver closes.
    """
    cdtype = xp.complex128 if dtype == xp.float64 else xp.complex64
    values = xp.stack([xp.asarray(as_material(m).epsilon, dtype=cdtype)
                       for m, _ in regions] + [xp.asarray(as_material(background).epsilon,
                                                        dtype=cdtype)])
    left = xp.asarray([plan.pieces[p].left for p, _, _, _ in plan.facing], dtype=xp.int64)
    right = xp.asarray([plan.pieces[q].left for _, q, _, _ in plan.facing], dtype=xp.int64)
    a, b = values[left], values[right]
    scale = xp.abs(a) + xp.abs(b)
    return xp.abs(a-b) / xp.where(scale > 0., scale, xp.ones_like(scale))


def interface_contrasts(xp, regions, background, plan, dtype):
    """Live contrast of each interval, including zero at material equality.

    The interval topology remains independent of the material values. The magnetic
    pass supplies its permeability view through the same scalar slots.
    """
    cdtype = xp.complex128 if dtype == xp.float64 else xp.complex64
    values = xp.stack([xp.asarray(as_material(m).epsilon, dtype=cdtype)
                       for m, _ in regions] + [xp.asarray(as_material(background).epsilon,
                                                        dtype=cdtype)])
    left = xp.asarray([p.left for p in plan.pieces], dtype=xp.int64)
    right = xp.asarray([p.right for p in plan.pieces], dtype=xp.int64)
    a, b = values[left], values[right]
    scale = xp.abs(a) + xp.abs(b)
    return xp.abs(a-b) / xp.where(scale > 0., scale, xp.ones_like(scale))


def plan_interfaces(regions, lattice, transform):
    """Plan on concrete resolved loops; integer images count the reduced basis.

    Contact candidates are bounded by intersecting fractional endpoint bounding boxes:
    a translate outside either coordinate interval cannot intersect the base edge.
    Distance images use a separate bound. On a reduced basis a point's nearest image
    is in the 3x3 about its fractional fold. A segment can reach ceil(|fractional E|)
    further cells, so enlarge that ring by its coordinate extent. Long segments are
    covered without assuming a shape fits a fixed 3x3 ring.
    """
    basis = np.asarray([[checked_float(v) for v in row]
                        for row in reduced_basis(lattice, transform)])
    inv = np.linalg.inv(basis)
    starts, ends, owners = [], [], []
    for owner, (_, loops) in enumerate(regions):
        for loop in loops:
            v = np.asarray([[checked_float(c) for c in p] for p in loop])
            starts.extend(v)
            ends.extend(np.roll(v, -1, axis=0))
            owners.extend([owner] * len(v))
    if not starts:
        return InterfacePlan((), ((0, 0),), ((),))
    a, z = np.asarray(starts), np.asarray(ends)
    e = z - a
    lengths = np.linalg.norm(e, axis=1)
    scale = max(np.linalg.norm(basis, axis=0).max(), lengths.max())
    tol = 128 * np.finfo(float).eps * scale
    fa, fz = a @ inv.T, z @ inv.T
    fmin, fmax = np.minimum(fa, fz), np.maximum(fa, fz)
    ftol = tol * np.linalg.norm(inv, axis=1)
    pieces, fractions, spans = [], [], []
    for i in range(len(a)):
        if lengths[i] <= tol:
            continue
        unit = e[i] / lengths[i]
        # Vectorized direction filter keeps polygonised curves inexpensive.
        cross = e[:, 0] * unit[1] - e[:, 1] * unit[0]
        candidates = np.flatnonzero((lengths > tol) &
                                   (np.abs(cross) <= tol * np.maximum(1., lengths / lengths[i])))
        events = [(0., (i, 0, 0, 0)), (1., (i, 1, 0, 0))]
        runs = []
        for j in candidates:
            lower = np.ceil(fmin[i] - fmax[j] - ftol).astype(int)
            upper = np.floor(fmax[i] - fmin[j] + ftol).astype(int)
            for m in range(int(lower[0]), int(upper[0]) + 1):
                for n in range(int(lower[1]), int(upper[1]) + 1):
                    shift = basis @ np.array([m, n])
                    delta = a[j] + shift - a[i]
                    # Use the longer edge's line. Normalising a tiny clipped remainder
                    # amplifies endpoint rounding, so a coincident full facet can otherwise
                    # be classified as nonparallel or displaced from it. The direction
                    # filter above likewise measures the shorter edge's transverse extent.
                    line = e[j] / lengths[j] if lengths[j] > lengths[i] else unit
                    if abs(delta[0] * line[1] - delta[1] * line[0]) > tol:
                        continue
                    t0 = float(delta @ unit / lengths[i])
                    t1 = float((z[j] + shift - a[i]) @ unit / lengths[i])
                    forward = t1 > t0
                    lo, hi = max(0., min(t0, t1)), min(1., max(t0, t1))
                    if (hi - lo) * lengths[i] <= tol:
                        continue
                    runs.append((lo, hi, owners[j], 1 if forward else -1, int(j), m, n))
                    if tol < lo * lengths[i] < lengths[i] - tol:
                        events.append((lo, (int(j), 0 if forward else 1, m, n)))
                    if tol < hi * lengths[i] < lengths[i] - tol:
                        events.append((hi, (int(j), 1 if forward else 0, m, n)))
        events.sort(key=lambda x: (x[0], x[1]))
        unique = []
        for t, ref in events:
            if not unique or (t - unique[-1][0]) * lengths[i] > tol:
                unique.append((t, ref))
            elif ref < unique[-1][1]:
                unique[-1] = (unique[-1][0], ref)
        for (lo, lref), (hi, href) in zip(unique, unique[1:]):
            mid = .5 * (lo + hi)
            covering = [run for run in runs if run[0] < mid < run[1]]
            winding = {}
            for _, _, owner, sign, _, _, _ in covering:
                winding[owner] = winding.get(owner, 0) + sign
            left = [owner for owner, count in winding.items() if count > 0]
            right = [owner for owner, count in winding.items() if count < 0]
            if len(left) > 1 or len(right) > 1:
                # The clipper reads outlines within about 1e-12 of their size as touching, and
                # an overlap that thin comes back from it uncut, two regions sharing the strip.
                # Here, at the rounding of the coordinates, the strip has two sides. Said as the
                # painting the planner cannot express, like the clipper's own refusals, rather
                # than as a malformed input.
                from wavebit._common.geometry.clipper import DegenerateClipError
                pair = sorted(left if len(left) > 1 else right)
                raise DegenerateClipError(
                    f'regions {pair[0]} and {pair[1]} (counted from 0 in the order declared) '
                    f'overlap in a strip thinner than the boundary planner resolves, about 1e-12 '
                    f'of their size: it reads them as touching there and then finds both on one '
                    f'side of the edge they share. Write them touching, or overlapping by more.')
            # A resolved contour can contain zero-area out-and-back spurs. Sum
            # signed coverage per region before assigning either side: a spur
            # cannot hide the true contour of a third, co-oriented edge.
            same_cell = [run for run in covering if run[5] == run[6] == 0]
            net = sum(run[3] for run in same_cell)
            canonical = min(run[4] for run in same_cell)
            centroid_sign = (1 if net > 0 else -1) if net and i == canonical else 0
            pieces.append(Interval(i, lref, href,
                                   left[0] if left else len(regions),
                                   right[0] if right else len(regions), centroid_sign))
            fractions.append((hi - lo) * (e[i] @ inv.T))
            spans.append((a[i] + lo * e[i], a[i] + hi * e[i]))
    reach = np.max(np.abs(e @ inv.T), axis=0)
    radius = [1 + int(math.ceil(float(v))) for v in reach]
    images = tuple((m, n) for m in range(-radius[0], radius[0] + 1)
                   for n in range(-radius[1], radius[1] + 1))
    # Any live endpoint has an image within rho of every sample: round the two
    # fractional coordinates, leaving a vector in the centered parallelogram.
    # A candidate whose coordinate interval is farther than rho therefore cannot
    # win. Use only segment lengths, not positions, so rigid translation never
    # changes the record. This avoids replicating every tiny curved facet on the
    # larger ring needed by a different, long edge in the same layer.
    rho = .5 * max(np.linalg.norm(basis[:, 0] + basis[:, 1]),
                   np.linalg.norm(basis[:, 0] - basis[:, 1]))
    norms = np.linalg.norm(inv, axis=1)
    df = np.asarray(fractions).reshape((-1, 2))
    # Round extents outward to whole cells. Otherwise harmless changes in a
    # clipped edge length can cross the pruning threshold and invalidate an
    # unchanged painting topology. Only a whole-cell extent event changes it.
    extent_lower = np.floor(np.minimum(0., df) + ftol)
    extent_upper = np.ceil(np.maximum(0., df) - ftol)
    selected, groups = [], []
    for image in images:
        lower = extent_lower + image
        upper = 1. + extent_upper + image
        gap = np.maximum(np.maximum(lower - 1., -upper), 0.) / norms
        ids = tuple(int(i) for i in np.flatnonzero(np.all(gap <= rho + tol, axis=1)))
        if ids:
            selected.append(image)
            groups.append(ids)
    facing = _plan_facing(pieces, spans, basis, inv, tol)
    return InterfacePlan(tuple(pieces), tuple(selected), tuple(groups), facing)


def _plan_facing(pieces, spans, basis, inv, tol):
    """Pairs of intervals that bound a sliver of one region thinner than :func:`facing_gap`.

    Both intervals carry the sliver's region on their right, run in opposite directions and
    overlap when projected on each other. The separation is the largest distance from an
    end of either common extent to the other interval's line, so a pair that is not parallel
    enters only once all of its common extent is within the gap. The plan holds no
    coordinates; replay recomputes the extents and the separation from traced vertices.
    """
    if not pieces:
        return ()
    gap = float(facing_gap(np, basis))
    s = np.asarray([span[0] for span in spans])
    z = np.asarray([span[1] for span in spans])
    d = z - s
    length = np.linalg.norm(d, axis=1)
    unit = d / np.where(length > 0, length, 1.)[:, None]
    left = np.asarray([piece.left for piece in pieces])
    right = np.asarray([piece.right for piece in pieces])
    usable = (length > tol) & (left != right)
    fs, fz = s @ inv.T, z @ inv.T
    fmin, fmax = np.minimum(fs, fz), np.maximum(fs, fz)
    freach = gap * np.linalg.norm(inv, axis=1) + tol * np.linalg.norm(inv, axis=1)
    facing = []
    for p in np.flatnonzero(usable):
        q = np.arange(p + 1, len(pieces))
        q = q[usable[q] & (right[q] == right[p]) & (unit[q] @ unit[p] < 0.)]
        if not len(q):
            continue
        # An integer image brings q within the gap only if these bounds admit one.
        lower = np.ceil(fmin[p] - fmax[q] - freach)
        upper = np.floor(fmax[p] - fmin[q] + freach)
        for j in q[np.all(lower <= upper, axis=1)]:
            lo = np.ceil(fmin[p] - fmax[j] - freach).astype(int)
            hi = np.floor(fmax[p] - fmin[j] + freach).astype(int)
            for m in range(int(lo[0]), int(hi[0]) + 1):
                for n in range(int(lo[1]), int(hi[1]) + 1):
                    shift = basis @ np.array([m, n])
                    if _facing_separation(s[p], z[p], s[j] + shift, z[j] + shift, tol) < gap:
                        facing.append((int(p), int(j), m, n))
    return tuple(facing)


def _facing_separation(s0, z0, s1, z1, tol):
    """Largest separation of two facing intervals over their common extent, or infinity.

    The common extent of each interval is the part covered by the other's projection. The
    separation is the largest distance from an end of either common extent to the other
    interval's line, measured on that line's right. A pair whose projections overlap by no
    more than ``tol``, or that does not face across its right sides, bounds no sliver.
    """
    extents = []
    for start, end, other_start, other_end in ((s0, z0, s1, z1), (s1, z1, s0, z0)):
        length = math.hypot(*(end - start))
        t = (end - start) / length
        u = sorted(((other_start - start) @ t, (other_end - start) @ t))
        lo, hi = max(0., u[0]), min(length, u[1])
        if hi - lo <= tol:
            return math.inf
        extents.append((start + lo * t, start + hi * t, start, np.array([t[1], -t[0]])))
    distances = [float((x - other[2]) @ other[3])
                 for own, other in ((extents[0], extents[1]), (extents[1], extents[0]))
                 for x in own[:2]]
    if min(distances) < -tol:
        return math.inf
    return max(distances)


def replay_interfaces(xp, regions, background, lattice, transform, plan, material_values=None):
    """Return starts, vectors, safe squared lengths, and live material-jump flags.

    Endpoint projections are recomputed from traced vertices and traced lattice vectors.
    Equality is per current scalar property, including its imaginary part: the magnetic
    pass supplies mu_value in epsilon slots, just as in the parametric builder.
    ``material_values`` optionally supplies region values followed by background;
    trailing tensor axes are reduced together, so equal traces do not erase a seam.
    """
    loops = [v for _, polys in regions for v in polys]
    if not loops:
        return None
    a = xp.concat(loops, axis=0)
    e = xp.concat([xp.roll(v, -1, axis=0) - v for v in loops], axis=0)
    basis = reduced_basis(lattice, transform)
    cdtype = xp.complex128 if a.dtype == xp.float64 else xp.complex64
    values = material_values if material_values is not None else xp.stack([xp.asarray(as_material(m).epsilon, dtype=cdtype)
                       for m, _ in regions] + [xp.asarray(as_material(background).epsilon,
                                                        dtype=cdtype)])
    if not plan.pieces:
        return None
    source = xp.asarray([p[0] for p in plan.pieces], dtype=xp.int64)
    left = xp.asarray([p.left for p in plan.pieces], dtype=xp.int64)
    right = xp.asarray([p.right for p in plan.pieces], dtype=xp.int64)
    base, edge = a[source], e[source]
    denom = xp.sum(edge * edge, axis=-1)

    def endpoint(which):
        refs = [p[which] for p in plan.pieces]
        j = xp.asarray([r[0] for r in refs], dtype=xp.int64)
        end = xp.asarray([r[1] for r in refs], dtype=a.dtype)
        m = xp.asarray([r[2] for r in refs], dtype=a.dtype)
        n = xp.asarray([r[3] for r in refs], dtype=a.dtype)
        shift = xp.stack([m * basis[0][0] + n * basis[0][1],
                          m * basis[1][0] + n * basis[1][1]], axis=-1)
        t = xp.sum((a[j] + end[:, None] * e[j] + shift - base) * edge, axis=-1) / denom
        own = (j == source) & (m == 0) & (n == 0)
        t = xp.where(own, end, xp.clip(t, 0., 1.))
        return base + t[:, None] * edge

    starts, ends = endpoint(1), endpoint(2)
    live = values[left] != values[right]
    if live.ndim > 1:
        live = xp.any(live, axis=tuple(range(1, live.ndim)))
    edge = ends - starts
    l2 = xp.sum(edge * edge, axis=-1)
    live = live & (l2 > 0)
    return starts, edge, xp.where(l2 > 0, l2, xp.ones_like(l2)), live
