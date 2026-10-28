# -*- coding: utf-8 -*-
"""Resolve painting order into disjoint polygons -- the planning half of the boundary route.

Everything here is NumPy on concrete values and runs **outside any trace**. What it produces is
a *plan*: which vertices a loop is made of, and which pairs of edges cross. The plan holds
indices, never coordinates -- ``('src', 'B', 7)`` or ``('x', 3, 5, 8, 9)`` -- so replaying it
against traced vertices (:func:`eval_table`) is pure arithmetic and differentiates. That split
is why a discrete geometric decision does not stop a gradient.

The clipping is Greiner-Hormann: G. Greiner and K. Hormann, "Efficient clipping of arbitrary
polygons", *ACM Transactions on Graphics* **17**(2), 71-83 (1998), doi:10.1145/274363.274364.
Two things about it cost a debugging round each and are kept in the code rather than only in a
comment:

* **Degeneracies are the norm, not the exception.** Plain GH fails on 49.9% of grid-snapped
  rectangle pairs -- shared edges and corners-on-edges are what real designs are made of, and
  "is this vertex inside?" has no answer on the boundary. :func:`plan_difference_robust` nudges
  the subtractor by ~1e-9 of its span **for planning only**, then replays the resulting plan
  against the unperturbed vertices. Sweeping the nudge over 400x moves the answer by ~3e-17.
* **The difference walk runs A backward**, so loops come out clockwise -- and the winding is
  *not* consistent between loops, so each loop's orientation is normalised on its own. When
  crossings exist the difference has no holes, which is what justifies forcing counter-clockwise.

Painting several shapes closes on an **oriented loop** representation: a region is a list of
loops, counter-clockwise for material and clockwise for a hole. Both the area and the polygon
transform are signed contour integrals, so a region's value is the plain sum over its loops and
nothing downstream special-cases a hole -- or *can* forget to. Subtracting from a region is
then subtracting from each loop, so every clipper call stays a simple-polygon pair and the loop
count grows roughly linearly instead of the 2^m an inclusion-exclusion formulation would need.

The orientation was carried in a separate ``sign`` field for a while, with every loop stored
counter-clockwise. That is the same information, but it is information *about* the data rather
than *in* it, and it has to be re-applied by hand at every use: `boundary.conv_mats` did not,
and a shape painted entirely inside another one was silently solved as a different structure
while staying Hermitian and conserving energy exactly. The conversion was also self-defeating --
:func:`plan_difference` already emits a hole clockwise, and :func:`subtract_loop` was flipping
it back.

**Every tolerance is relative to the geometry it judges, so the plan does not depend on the
length unit.** ``EPS`` used to be compared with cross products and coordinates as plain numbers --
lengths squared and lengths -- so it meant something different in every unit: on a sub-micron
design spelled in metres, the default unit, every pair of edges came out "parallel", no crossing
was found, and an overlap was painted wrongly -- cut out as a hole, taken as covered, or left
summed -- or the nudge ladder ran out and raised, with ``check`` passing and ``R + T = 1``. Now
two edges are parallel when the sine of the angle between them is below ``EPS``
(:func:`_seg_cross`), and a vertex is on an outline when it lies within ``EPS`` of that outline's
own size from it (:func:`point_in_polygon`); neither is ever finer than the rounding of the
coordinates it reads (:func:`_grain`). All of it is dimensionless, so the plan for ``2**k * x`` is
the plan for ``x`` row for row, and a layout re-spelled in another unit plans the same up to the
rounding of its decimal conversion.

They are also **local**, and that is not incidental. Scaling the whole layer by its largest
coordinate before planning is the obvious alternative and is wrong twice over: the scale then
depends on where the cell sits, so a layout written whole periods away from it -- the same
structure -- had true crossings read as parallel and painted wrongly; and it depends on every
shape, so moving one shape across a power of two changed how two others were judged, which a
frozen plan then reported as a topology change.
"""
import math

import numpy as np

# Dimensionless: a sine, a fraction along a segment, or a distance in units of an outline's size.
EPS = 1e-12

# Entry/exit flag convention for the difference. Set empirically against a raster rather than
# by argument -- the four combinations are easy to reason wrongly about, and only one matches.
FLIP_A = True
FLIP_B = False


class DegenerateClipError(NotImplementedError):
    """A configuration this clipper does not handle (touching, collinear overlap).

    A ``NotImplementedError``, because that is what it is: a request the planner cannot express
    yet, not a defect of the input. ``prepare`` and ``run`` therefore report it as
    ``rcwa.unsupported`` rather than letting a private class escape; :func:`resolve_painting`
    names the two regions. ``Prepared.run`` refuses new values it cannot plan the way it refuses
    any that leave its record, as ``prepared.topology``.
    """


class StalePlanError(RuntimeError):
    """The recorded walk no longer describes this parameter set."""


def _cross(p, q):
    return p[0] * q[1] - p[1] * q[0]


#: Sixteen units in the last place, as a fraction of the largest coordinate.
_ROUNDING = 16 * np.finfo(float).eps


def _grain(*arrays):
    """The rounding of coordinates this large: no tolerance can be finer than this.

    The tolerances here are relative to the outlines they judge, so they shrink with a feature;
    the rounding of the coordinates a feature is written in does not. Written a few thousand
    feature sizes from the origin -- a layout moved by whole periods -- a vertex exactly on an
    edge comes out several 1e-13 off it in um, outside ``EPS`` of a 0.15-wide outline, and the
    walk then ran on decisions that disagreed with each other (``T`` 5e-2 off, measured).
    """
    return _ROUNDING * max(float(np.max(np.abs(V))) for V in arrays)


def _seg_cross(a1, a2, b1, b2, l1, l2, grain):
    """Proper crossing of two open segments -> (t, u), else None.

    ``l1`` and ``l2`` are the segments' lengths and ``grain`` the rounding of their coordinates
    (:func:`_grain`), both taken once per pair of outlines by :func:`_rings`.
    """
    d1, d2 = a2 - a1, b2 - b1
    den = _cross(d1, d2)
    # Parallel when the sine of the angle between them is below EPS, or when `den` is within the
    # rounding of the coordinates it was formed from. `<=`, so that a zero-length edge -- both
    # sides 0 -- is parallel rather than a division by zero.
    if abs(den) <= max(EPS * l1 * l2, grain * (l1 + l2)):
        return None                                   # parallel or collinear
    diff = b1 - a1
    t = _cross(diff, d2) / den
    u = _cross(diff, d1) / den
    on = (EPS < t < 1 - EPS) and (EPS < u < 1 - EPS)
    touching = (-EPS < t < 1 + EPS) and (-EPS < u < 1 + EPS) and not on
    if touching:
        raise DegenerateClipError(
            f'segment endpoint lies on the other segment (t={t:.6g}, u={u:.6g}); '
            'nudge the geometry or extend the clipper')
    return (t, u) if on else None


def point_in_polygon(p, V):
    """Crossing-number test. Boundary cases are rejected as degenerate.

    "On the boundary" means within ``EPS`` times the outline's own size of an edge, and never
    closer than the coordinates' rounding (:func:`_grain`). A test on the cross product alone,
    ``|(b - a) x (p - a)| < EPS``, is a distance below ``EPS / |b - a|``: in length units, and
    longer the shorter the edge -- on a 64-gon's short edges in um it held a vertex that every
    nudge had moved off the edge.
    """
    n = len(V)
    inside = False
    tol = max(EPS * float(np.max(V.max(0) - V.min(0))), _grain(V, p))
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        if abs(_cross(b - a, p - a)) <= tol * math.hypot(b[0] - a[0], b[1] - a[1]) and \
           min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and \
           min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol:
            raise DegenerateClipError('vertex lies exactly on the other polygon boundary')
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            if x > p[0]:
                inside = not inside
    return inside


# --------------------------------------------------------------------- plan

def _rings(A, B):
    """Both vertex rings with crossings spliced in, cross-linked by ``twin``.

    A node is ``dict(kind='src'|'x', ...)``; ``src`` carries the source index, ``x`` carries
    the crossing's ``(i, j)`` edge pair and its concrete point.
    """
    na, nb = len(A), len(B)
    hits_a = [[] for _ in range(na)]
    hits_b = [[] for _ in range(nb)]
    la = [math.hypot(*(A[(i + 1) % na] - A[i])) for i in range(na)]
    lb = [math.hypot(*(B[(j + 1) % nb] - B[j])) for j in range(nb)]
    grain = _grain(A, B)
    for i in range(na):
        for j in range(nb):
            r = _seg_cross(A[i], A[(i + 1) % na], B[j], B[(j + 1) % nb], la[i], lb[j], grain)
            if r is not None:
                t, u = r
                pt = A[i] + t * (A[(i + 1) % na] - A[i])
                hits_a[i].append((t, i, j, pt))
                hits_b[j].append((u, i, j, pt))

    def ring(V, hits, which):
        out = []
        for k in range(len(V)):
            out.append(dict(kind='src', poly=which, idx=k, pt=V[k], twin=None))
            for (a, i, j, pt) in sorted(hits[k], key=lambda h: h[0]):
                out.append(dict(kind='x', i=i, j=j, pt=pt, alpha=a, twin=None,
                                entry=None, done=False))
        return out

    ra, rb = ring(A, hits_a, 'A'), ring(B, hits_b, 'B')
    key_b = {(n['i'], n['j']): m for m, n in enumerate(rb) if n['kind'] == 'x'}
    for m, n in enumerate(ra):
        if n['kind'] == 'x':
            n['twin'] = key_b[(n['i'], n['j'])]
            rb[n['twin']]['twin'] = m
    return ra, rb


def _mark(ring, other_poly, flip):
    """Entry/exit flags: alternate around the ring, seeded by the first vertex's status."""
    status = point_in_polygon(ring[0]['pt'], other_poly)
    if flip:
        status = not status
    for n in ring:
        if n['kind'] == 'x':
            n['entry'] = not status
            status = not status


def _walk(ra, rb):
    """Follow the marked rings and emit closed loops of node references.

    Each crossing is emitted once, at the top of the outer loop; the source vertices between
    two crossings are emitted as they are passed. The loop closes when the walk reaches a node
    it has already emitted.
    """
    loops = []
    rings = {'A': ra, 'B': rb}
    budget = 8 * (len(ra) + len(rb)) + 16
    for start in [m for m, n in enumerate(ra) if n['kind'] == 'x']:
        if ra[start]['done']:
            continue
        loop, cur, m, steps = [], 'A', start, 0
        while True:
            R = rings[cur]
            node = R[m]
            if node['done']:
                break
            node['done'] = True
            rings['B' if cur == 'A' else 'A'][node['twin']]['done'] = True
            loop.append((cur, m))

            fwd = node['entry']
            while True:
                m = (m + 1) % len(R) if fwd else (m - 1) % len(R)
                steps += 1
                if steps > budget:
                    raise DegenerateClipError('walk did not close')
                if R[m]['kind'] == 'x':
                    break
                loop.append((cur, m))
            m, cur = R[m]['twin'], ('B' if cur == 'A' else 'A')
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def plan_difference(A, B):
    """Plan for ``A \\ B``. Returns ``(plan, polygons)`` with concrete vertices.

    ``plan`` is a list of loops; each loop is a list of records:
      ``('src', 'A'|'B', index)``  or  ``('x', i, j)`` for the crossing of A-edge i, B-edge j.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    ra, rb = _rings(A, B)
    n_crossings = sum(1 for n in ra if n['kind'] == 'x')

    if n_crossings == 0:                                       # no crossings: containment or disjoint
        if point_in_polygon(A[0], B):
            return [], []                              # A entirely covered
        if point_in_polygon(B[0], A):                  # B punches a hole in A
            plan = [[('src', 'A', k) for k in range(len(A))],
                    [('src', 'B', k) for k in range(len(B) - 1, -1, -1)]]
            return plan, _eval_plan_np(plan, A, B)
        plan = [[('src', 'A', k) for k in range(len(A))]]
        return plan, _eval_plan_np(plan, A, B)

    if n_crossings % 2:
        raise DegenerateClipError(f'odd number of crossings ({n_crossings}) — tangency?')

    # Flag convention for A\B, pinned against a raster in `_selftest_flags` below:
    # walk A forward where it is OUTSIDE B, and B backward where it is INSIDE A.
    _mark(ra, B, flip=FLIP_A)
    _mark(rb, A, flip=FLIP_B)
    loops = _walk(ra, rb)

    plan = []
    for loop in loops:
        recs, pts = [], []
        for (which, m) in loop:
            n = (ra if which == 'A' else rb)[m]
            recs.append(('src', n['poly'], n['idx']) if n['kind'] == 'src'
                        else ('x', n['i'], n['j']))
            pts.append(n['pt'])
        # collapse consecutive duplicates the walk can emit at switch points
        keep = [k for k in range(len(recs)) if k == 0 or recs[k] != recs[k - 1]]
        ded = [recs[k] for k in keep]
        dpt = np.array([pts[k] for k in keep])
        if len(ded) > 1 and ded[0] == ded[-1]:
            ded, dpt = ded[:-1], dpt[:-1]
        if len(ded) < 3:
            continue
        # With crossings present, A\B is a set of disjoint simple pieces -- no holes, so every
        # loop must be counter-clockwise. The walk does not produce a consistent winding
        # (it runs backward along A and forward along B), so normalise each loop on its own.
        if signed_area(dpt) < 0:
            ded = ded[::-1]
        plan.append(ded)
    return plan, _eval_plan_np(plan, A, B)


# ------------------------------------------------------------------- replay

def replay(xp, plan, A, B, validate=None):
    """Evaluate a plan against (possibly traced) vertex arrays. Returns a list of polygons."""
    out = []
    for loop in plan:
        pts = []
        for rec in loop:
            if rec[0] == 'src':
                V = A if rec[1] == 'A' else B
                pts.append(V[rec[2]])
            else:
                _, i, j = rec
                a1, a2 = A[i], A[(i + 1) % A.shape[0]]
                b1, b2 = B[j], B[(j + 1) % B.shape[0]]
                d1, d2, diff = a2 - a1, b2 - b1, b1 - a1
                den = d1[0] * d2[1] - d1[1] * d2[0]
                t = (diff[0] * d2[1] - diff[1] * d2[0]) / den
                if validate is not None:
                    u = (diff[0] * d1[1] - diff[1] * d1[0]) / den
                    tc, uc = float(t), float(u)
                    if not (validate[0] < tc < 1 - validate[0]) or \
                       not (validate[0] < uc < 1 - validate[0]):
                        raise StalePlanError(
                            f'crossing (A edge {i}, B edge {j}) left its segments: '
                            f't={tc:.4f}, u={uc:.4f}')
                pts.append(a1 + t * (a2 - a1))
        out.append(xp.stack(pts))
    return out


def _eval_plan_np(plan, A, B):
    return replay(np, plan, A, B)


def signed_area(V):
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ------------------------------------------------------- degenerate configs

# A direction that is never axis-aligned and never parallel to a 45-degree edge, so nudging
# along it breaks ties that grid-aligned designs create by construction.
_NUDGE = np.array([np.cos(1.0), np.sin(1.0)])
# Tried when every step along the first fails. A contact along an edge parallel to `_NUDGE` --
# an edge at 1 rad, which `Rectangle(angle=1.0)` has -- slides along itself under every step of
# it, and was refused until this direction moved it off; only a pair touching along edges
# parallel to both at once is left.
_NUDGE_ACROSS = np.array([np.cos(2.3), np.sin(2.3)])

# Relative nudge sizes tried in order. They only ever decide *combinatorics*; see
# `plan_difference_robust` for why none of this reaches the returned coordinates.
_NUDGE_STEPS = (1e-9, -2.7e-9, 7.1e-9, -1.9e-8, 5.3e-8)


def plan_difference_robust(A, B, steps=_NUDGE_STEPS):
    """``A \\ B`` that survives exactly-touching geometry.

    Grid-aligned designs violate general position on purpose: shapes share edges, corners land
    exactly on edges, rectangles sit flush. Plain Greiner-Hormann has no answer there -- "is
    this vertex inside?" has no truth value when it is *on* the boundary.

    The fix is to break the tie with a tiny nudge, but to nudge **only while planning**. The
    plan is pure bookkeeping -- ``('src', 'B', 7)``, ``('x', 3, 5)`` -- indices, never
    coordinates. So the nudged geometry decides *which* vertices and crossings form the loop,
    and then the loop is evaluated against the **unperturbed** input. The nudge size therefore
    does not appear in the result at all; it is symbolic perturbation, not an approximation.

    What a tie-break can still change is the *interpretation* of a degenerate contact, and the
    two readings differ by a zero-area spur -- which contributes nothing to the area or to the
    Fourier transform. Returns ``(plan, polygons, nudge_used)``, the nudge as a fraction of the
    pair's size along its direction (zero when none was needed).
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    try:
        plan, polys = plan_difference(A, B)
        return plan, polys, np.zeros(2)
    except DegenerateClipError:
        pass

    # The pair's own size, so the nudge is the same fraction of it in every unit. Only a pair
    # with no extent at all, which no outline that passed `polygon_defect` is, falls back to 1.
    span = float(max(A.max(0) - A.min(0)).max() if A.size else 0.0)
    span = max(span, float(max(B.max(0) - B.min(0)).max() if B.size else 0.0)) or 1.0

    for direction in (_NUDGE, _NUDGE_ACROSS):
        for s in steps:
            try:
                plan, _ = plan_difference(A, B + s * span * direction)
            except DegenerateClipError:
                continue
            # Replay the nudged plan against the ORIGINAL vertices.
            return plan, replay(np, plan, A, B), s * direction

    raise DegenerateClipError(
        'no nudge along either direction resolved this contact')


# ==================================================================== painting
class Table:
    """Vertex table: rows are ``('src', shape, vertex)`` or ``('x', ia1, ia2, ib1, ib2)``."""

    def __init__(self, shapes):
        self.shapes = [np.asarray(s, float) for s in shapes]
        self.recs = []
        self.vals = []
        self._crossings = {}

    def src(self, si, vi):
        self.recs.append(('src', si, vi))
        self.vals.append(self.shapes[si][vi])
        return len(self.recs) - 1

    def cross(self, ia1, ia2, ib1, ib2):
        key = (ia1, ia2, ib1, ib2)
        if key in self._crossings:
            return self._crossings[key]
        a1, a2 = self.vals[ia1], self.vals[ia2]
        b1, b2 = self.vals[ib1], self.vals[ib2]
        self.recs.append(('x', ia1, ia2, ib1, ib2))
        self.vals.append(_seg_point(np, a1, a2, b1, b2))
        self._crossings[key] = len(self.recs) - 1
        return len(self.recs) - 1

    def poly(self, ids):
        return np.array([self.vals[i] for i in ids])


def _seg_point(xp, a1, a2, b1, b2):
    d1, d2, diff = a2 - a1, b2 - b1, b1 - a1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / den
    return a1 + t * (a2 - a1)


def eval_table(xp, recs, shapes):
    """Evaluate every table row against (possibly traced) shape vertex arrays."""
    vals = []
    for rec in recs:
        if rec[0] == 'src':
            vals.append(shapes[rec[1]][rec[2]])
        else:
            _, ia1, ia2, ib1, ib2 = rec
            vals.append(_seg_point(xp, vals[ia1], vals[ia2], vals[ib1], vals[ib2]))
    return vals


def loop_verts(xp, ids, vals):
    return xp.stack([vals[i] for i in ids])


# ------------------------------------------------------------------ subtract

def _bbox(P):
    return P.min(0), P.max(0)


def _disjoint(P, Q):
    (p0, p1), (q0, q1) = _bbox(P), _bbox(Q)
    return bool((p1 < q0).any() or (q1 < p0).any())


def subtract_loop(table, ids, sub_ids):
    """``loop \\ subtractor`` as a list of loops, appending crossings to the table.

    **The orientation is the answer, not a label attached to it.** A hole comes back wound
    clockwise, which is what makes it subtract itself in every consumer -- the signed shoelace
    and the signed contour integral both read it directly, with no sign to carry alongside and
    so no sign for a consumer to forget. :func:`plan_difference` already emits it that way (the
    containment branch walks ``B`` backwards on purpose); this function used to flip it back to
    counter-clockwise and record a separate ``-1``, which is the conversion that put the
    invariant somewhere a reader had to re-apply it. One of four readers did not, and a shape
    painted entirely inside another was solved as a different structure.

    What genuinely needs counter-clockwise input is :func:`plan_difference`: its entry/exit flag
    convention is oriented ("walk A forward where it is OUTSIDE B, and B backward where it is
    INSIDE A"), so a clockwise ``A`` would invert the walk. That is handled where it belongs --
    turn a clockwise input the right way up, difference it, and turn the results back. The plan
    holds indices into whatever array it was handed, so ``ids`` is reversed *before* the call
    and the reversed list is what the records are read against.

    Turning the results back is exactly the old ``sign * rel`` product, spelled as geometry:
    subtracting from a hole leaves a hole, and the pieces a hole is cut into are holes too.
    """
    A, B = table.poly(ids), table.poly(sub_ids)
    if _disjoint(A, B):
        return [ids]

    # `plan_difference` needs A counter-clockwise; see the docstring.
    flipped = signed_area(A) < 0
    if flipped:
        ids = ids[::-1]
        A = table.poly(ids)

    # The concrete polygons are not needed any more: the winding they would have been consulted
    # for is now carried by `flipped` and the plan's own order.
    plan, _polys, _nudge = plan_difference_robust(A, B)
    if not plan:
        return []                                   # the loop is entirely covered

    out = []
    na, nb = len(ids), len(sub_ids)
    for rec_loop in plan:
        new = []
        for rec in rec_loop:
            if rec[0] == 'src':
                new.append(ids[rec[2]] if rec[1] == 'A' else sub_ids[rec[2]])
            else:
                _, i, j = rec
                new.append(table.cross(ids[i], ids[(i + 1) % na],
                                       sub_ids[j], sub_ids[(j + 1) % nb]))
        if flipped:
            new = new[::-1]
        if len(new) >= 3:
            out.append(new)
    return out


def resolve_painting(shapes):
    """Resolve drawing order into one region per shape.

    ``shapes`` is a list of simple vertex arrays, earliest drawn first; later shapes paint over
    earlier ones. Returns ``(table_records, regions)`` where ``regions[i]`` is a list of
    ``loop_ids`` for shape ``i``'s visible part -- **counter-clockwise for material, clockwise
    for a hole**, which is the whole of the bookkeeping. A hole inside a hole comes back
    counter-clockwise again, i.e. material, by composition rather than by multiplying signs.

    **An outline may arrive wound either way**, and one wound clockwise is walked backwards:
    its table rows are taken in reverse index order, so every loop starts out counter-clockwise
    and the rows still name the vertices as they were declared. The primitives are built
    counter-clockwise and never take that branch; a user's polygon may. Reversing the *rows*
    rather than the array is what lets a plan replay on traced vertices: the winding was read
    here, once, on concrete values, and what survives of the decision is an index order --
    so a replay never has to ask the sign of an area it cannot read. Taken as given instead,
    a clockwise outline would be its own hole: the transform is signed, and S4, which takes
    its polygons as given, solves a clockwise rectangle as a different structure (``sum(T)``
    0.859 against 0.974, measured) without a word.

    A contact the nudge ladder cannot resolve is refused with :class:`DegenerateClipError`
    naming the two regions: ``j``, the one being painted, and the earlier region whose outline
    it meets (:func:`_met_region`), which is not always the region being cut -- an outline
    between them may have left its edge in that region as a hole.
    """
    table = Table(shapes)
    ids = []
    for i, V in enumerate(table.shapes):
        order = range(len(V)) if signed_area(V) >= 0 else range(len(V) - 1, -1, -1)
        ids.append([table.src(i, k) for k in order])

    regions = []
    for i in range(len(shapes)):
        region = [ids[i]]
        for j in range(i + 1, len(shapes)):
            nxt = []
            for lp in region:
                try:
                    nxt.extend(subtract_loop(table, lp, ids[j]))
                except DegenerateClipError as error:
                    met = _met_region(table, lp, ids, j, i)
                    raise DegenerateClipError(
                        f'region {j} cannot be painted over region {met} '
                        f'(regions counted from 0 in the order declared): the boundary planner '
                        f'found no consistent way to cut one outline by the other where they '
                        f'touch exactly: along edges at, or within about 2e-5 rad of, 1 rad and '
                        f'2.3 rad to the x axis at once, the two directions it shifts an outline '
                        f'in to decide a contact, or so far from the cell that the coordinates '
                        f'cannot resolve the shift. Move region {j} by a small fraction of its '
                        f'size.'
                    ) from error
            region = nxt
            if not region:
                break
        regions.append(region)
    return table.recs, regions


def _met_region(table, loop, ids, j, fallback):
    """The earlier region whose outline region ``j`` could not be cut against, for a message.

    The loop that failed is built from the outlines it was cut by as well as its own, so the
    culprit is looked for among those: the first whose outline alone, against ``j``'s, is
    unresolvable too. Only reached on a refusal, so the cost does not matter.
    """
    owners, seen, todo = set(), set(), list(loop)
    while todo:
        row = todo.pop()
        if row in seen:
            continue
        seen.add(row)
        rec = table.recs[row]
        if rec[0] == 'src':
            owners.add(rec[1])
        else:
            todo.extend(rec[1:])
    for k in sorted(owners - {j}):
        try:
            plan_difference_robust(table.poly(ids[k]), table.poly(ids[j]))
        except DegenerateClipError:
            return k
    return fallback


def region_area(xp, region, vals):
    """Area of a region. The shoelace is signed, so a clockwise hole subtracts itself."""
    tot = 0.0
    for lp in region:
        V = loop_verts(xp, lp, vals)
        x, y = V[:, 0], V[:, 1]
        tot = tot + 0.5 * xp.sum(x * xp.roll(y, -1) - xp.roll(x, -1) * y)
    return tot
