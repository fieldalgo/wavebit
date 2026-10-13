"""Do two parametric shapes overlap? — the geometry behind the layer-semantics check.

:mod:`wavebit.rcwa._engine.parametric` sums its shapes and :mod:`wavebit.rcwa._engine.boundary` paints them,
so the two only describe the same structure while no two shapes intersect. That makes "do these
intersect?" the question the modeling layer has to answer before it can let a layer through,
and this module answers it for the parametric primitives: rectangle / square (a rotated box) and
circle / disk / ellipse (a rotated ellipse). Both are convex, which is what makes it cheap.

Everything here works on the *signed gap* between two shapes — positive is the distance
between them, negative is how deep they interpenetrate, and zero is touching. Two convex sets
are disjoint exactly when some direction separates them, so with each shape's support radius
``r_S(d)`` (how far it reaches from its centre along ``d``) the gap is

    gap = max over unit d of  (c_A - c_B) . d - r_A(d) - r_B(d)

which is concave in ``d``, so the maximum is unique in the sense that matters — any direction
with a positive value is a *proof* of disjointness. Three pairings never need the search
because a closed form exists: box/box (the separating-axis theorem, where only the four edge
normals can separate, and the same maximum is the exact penetration depth), circle/circle
(centre distance minus the radii), and circle/box (clamp the centre into the box). Only a
genuine ellipse falls back to :func:`_gap_searched`, which samples the circle and then refines
the best direction; a positive result there is still a proof, so the one thing it can get
wrong is calling a hair's-breadth gap an overlap. That is the harmless direction — the caller
warns instead of silently building a different structure, and says how to proceed anyway.

Shapes are checked against each other's *periodic images* as well, because the layer is a unit
cell: two rods that miss inside the cell can still meet across the boundary, and a rod wider
than the period runs into itself. Touching is not overlapping, so the caller's tolerance is
what keeps abutting rectangles — a checkerboard, a rod exactly filling the period — from
being reported.

**Touching is the second question this module answers**, and it is asked of *faces* rather
than of shapes. Two boxes that abut -- a shape against its own periodic image, or against
another box -- share a line over some run, and along that run the material continues: there is
no interface there, whatever the outlines say. :func:`face_pairs` finds every such run over the
ring of images, and :func:`~wavebit.rcwa._engine.parametric.plan_gamma` turns the sweep into the
combinatorial record the normal field replays. :func:`interfaces_of` is the independent
instrument for both -- the outlines of every image, sampled densely, with each sample kept
only if a step along its outward normal lands in no image at all -- and it shares no arithmetic
with the sweep, which is what makes it an oracle rather than a second copy.
"""
import math
from collections import namedtuple

import numpy as np

from wavebit._common.backend import checked_float
from wavebit.lattice import reduced_basis

#: Penetration below this fraction of the period is touching, not overlapping.
OVERLAP_TOL_REL = 1e-6

_INV_PHI = (5.0 ** 0.5 - 1.0) / 2.0

BOX = 'box'
ELLIPSE = 'ellipse'

_BOX_TYPES = ('rectangle', 'square')
_ELLIPSE_TYPES = ('circle', 'disk', 'ellipse')

# Periodic images of the second shape, in cells of the *reduced* basis (`reduced_basis`, which
# every sweep here applies first): there a shape reaching past one cell on both sides at once
# is already reported against its own image, so one ring is enough. On the basis as given it
# was not -- `a2 = 10 a1 + (50, 700)` has its (50, 700) neighbour at (-10, 1).
_IMAGES = tuple((ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1))

#: The four faces of a box in the order every reader of a face index uses: outward normal
#: along the shape's own ``+u``, ``-u``, ``+v``, ``-v``. A face's *tangent* is the other axis.
FACES = ('+u', '-u', '+v', '-v')

#: **Rounding-scale thresholds, and they step.** Two face lines are "the same line" when their
#: offsets along the shared normal agree to this fraction of the longer basis vector, and two
#: face normals are "opposed" when ``|n_p + n_q|^2`` is under ``ANTIPARALLEL_TOL2``. Both are
#: the size of the arithmetic's own error, not of any geometry: ``cos(pi/2)`` is 6.1e-17 rather
#: than 0, so a box turned a quarter turn misses an *exact* test on a 700 cell by 4.3e-14 and
#: passes this one by ten orders. That is all the tolerances are for, and it is worth saying
#: what they are not: a face 7.1e-10 off its partner's line on a 700 cell is two interfaces, and
#: one 6.9e-10 off is a pair whose members cancel identically -- the field steps between the two
#: by a band of depth ``(4 * delta * D**4) ** (1/5)`` (about two units for ``D = 350``), which is
#: 1.4e-5 in ``sum(T)`` on the square grating at order 5 and 1.7e-4 on the sheared box (N = 256)
#: at the rungs ``tests/rcwa/test_gamma_field.py`` puts either side of it -- a step in the value at
#: a rounding-scale location, and one that does not decay with order on the sheared cell
#: (1.8e-5 / 1.9e-5 / 5.2e-5 at orders 4 / 8 / 12; 1.1e-5 / 4.1e-6 / 1.4e-6 on the square),
#: since the band's depth is independent of the grid and the samples it holds grow with N. The
#: same shape of threshold the deleted lattice-direction test had, one level down: it can sit at
#: rounding scale because everything above it is carried continuously by the blend, so there
#: is no geometry for a wider window to swallow.
COLLINEAR_TOL_REL = 1e-12
ANTIPARALLEL_TOL2 = 1e-24


def images(r):
    """Every lattice offset ``(m, n)`` with ``|m|, |n| <= r``, ``(0, 0)`` included.

    ``r`` is the plan's ring (:func:`~wavebit.rcwa._engine.parametric.plan_gamma`), read here by the
    pair sweep and by the oracle so the two cannot disagree about which images exist. Both
    count it in cells of the reduced basis, as the plan does.
    """
    return tuple((m, n) for m in range(-r, r + 1) for n in range(-r, r + 1))


def box_faces(p):
    """The four faces of box ``p`` in :data:`FACES` order: ``(mx, my, nx, ny, sx, sy, h)`` each.

    Midpoint, outward unit normal, unit tangent and half-length. The tangent of a ``u`` face
    is ``+v`` and of a ``v`` face ``+u`` -- the *same* direction for the two members of an
    opposite pair, which is what lets a run found on one be read off the other by a sign.
    """
    _, cx, cy, a, b, ct, st = p
    ux, uy = ct, st
    vx, vy = -st, ct
    return ((cx + a * ux, cy + a * uy, ux, uy, vx, vy, b),
            (cx - a * ux, cy - a * uy, -ux, -uy, vx, vy, b),
            (cx + b * vx, cy + b * vy, vx, vy, ux, uy, a),
            (cx - b * vx, cy - b * vy, -vx, -vy, ux, uy, a))


def primitive(stype, cx, cy, lx, ly, angle):
    """``(kind, cx, cy, a, b, cos, sin)`` for one resolved shape, or ``None`` if not a primitive.

    ``a`` / ``b`` are semi-axes, so an ``lx``-wide box and an ``lx``-diameter circle both
    become ``a = lx / 2``. Unknown shape types return ``None`` rather than raising: this
    module is a check running ahead of the build, and the build's own error for an unknown
    type is the better one to surface.
    """
    if stype in _BOX_TYPES:
        kind = BOX
    elif stype in _ELLIPSE_TYPES:
        kind = ELLIPSE
    else:
        return None
    # `checked_float`, not `float`: a `jax.grad` tracer refuses `float()` while carrying a
    # perfectly concrete primal, and `parametric._primitives` reads that refusal as "traced --
    # skip the check". See the helper for what that cost.
    return (kind, checked_float(cx), checked_float(cy),
            checked_float(lx) / 2.0, checked_float(ly) / 2.0,
            math.cos(checked_float(angle)), math.sin(checked_float(angle)))


def _is_circle(p):
    return p[0] == ELLIPSE and p[3] == p[4]


def _support_radius(p, dx, dy):
    """How far ``p`` reaches from its centre along unit ``(dx, dy)`` — even in the direction."""
    kind, _, _, a, b, ct, st = p
    u = dx * ct + dy * st                 # direction in the shape's own frame
    v = -dx * st + dy * ct
    if kind == BOX:
        return a * abs(u) + b * abs(v)
    return math.hypot(a * u, b * v)


def _gap_along(p, q, dx, dy):
    """Signed separation of ``p`` from ``q`` along unit ``(dx, dy)``; positive means a gap."""
    return ((p[1] - q[1]) * dx + (p[2] - q[2]) * dy
            - _support_radius(p, dx, dy) - _support_radius(q, dx, dy))


def _gap_on_axis(p, q, dx, dy):
    """The better of ``+d`` and ``-d``.

    Only the centre term changes sign with the direction — the support radii do not — so an
    *axis* separates when the absolute projection clears both reaches. A caller enumerating
    axes rather than directions has to say so, or it reads a box's own edge normal pointing
    away from the other box as a 20-unit interpenetration.
    """
    return (abs((p[1] - q[1]) * dx + (p[2] - q[2]) * dy)
            - _support_radius(p, dx, dy) - _support_radius(q, dx, dy))


def _gap_box_box(p, q):
    """Separating-axis theorem: for two boxes only the four edge normals can separate them."""
    best = -math.inf
    for s in (p, q):
        ct, st = s[5], s[6]
        for dx, dy in ((ct, st), (-st, ct)):
            best = max(best, _gap_on_axis(p, q, dx, dy))
    return best


def _gap_circle_circle(p, q):
    return math.hypot(p[1] - q[1], p[2] - q[2]) - p[3] - q[3]


def _gap_circle_box(c, b):
    """Clamp the circle's centre into the box: the exact distance, negative when inside."""
    dx, dy = c[1] - b[1], c[2] - b[2]
    u = dx * b[5] + dy * b[6]
    v = -dx * b[6] + dy * b[5]
    du, dv = abs(u) - b[3], abs(v) - b[4]
    if du <= 0.0 and dv <= 0.0:                   # centre inside: nearest face decides
        return max(du, dv) - c[3]
    return math.hypot(max(du, 0.0), max(dv, 0.0)) - c[3]


def _gap_searched(p, q, n_coarse=1024, n_refine=64):
    """Maximise the gap over directions: sample the circle, then refine the best bracket.

    The sampled maximum alone is already a disjointness proof whenever it is positive; the
    golden-section pass is what makes a *near*-tangent pair come out positive too, since the
    cone of separating directions narrows to nothing as two shapes approach touching.
    """
    step = 2.0 * math.pi / n_coarse
    best_i, best = 0, -math.inf
    for i in range(n_coarse):
        t = i * step
        v = _gap_along(p, q, math.cos(t), math.sin(t))
        if v > best:
            best_i, best = i, v

    def g(t):
        return _gap_along(p, q, math.cos(t), math.sin(t))

    lo, hi = (best_i - 1) * step, (best_i + 1) * step
    c, d = hi - _INV_PHI * (hi - lo), lo + _INV_PHI * (hi - lo)
    gc, gd = g(c), g(d)
    for _ in range(n_refine):
        if gc < gd:
            lo, c, gc = c, d, gd
            d = lo + _INV_PHI * (hi - lo)
            gd = g(d)
        else:
            hi, d, gd = d, c, gc
            c = hi - _INV_PHI * (hi - lo)
            gc = g(c)
    return max(best, gc, gd)


def signed_gap(p, q):
    """Distance between two primitives, negative when they interpenetrate.

    Exact for every pairing that has a closed form; only a genuine (non-circular) ellipse
    reaches the direction search. See the module docstring for what that costs.
    """
    p_circle, q_circle = _is_circle(p), _is_circle(q)
    if p[0] == BOX and q[0] == BOX:
        return _gap_box_box(p, q)
    if p_circle and q_circle:
        return _gap_circle_circle(p, q)
    if p_circle and q[0] == BOX:
        return _gap_circle_box(p, q)
    if q_circle and p[0] == BOX:
        return _gap_circle_box(q, p)
    return _gap_searched(p, q)


def _bounding_radius(p):
    """Radius of a circle about the centre that contains the shape."""
    return math.hypot(p[3], p[4]) if p[0] == BOX else max(p[3], p[4])


def _shifted(p, dx, dy):
    return (p[0], p[1] + dx, p[2] + dy, p[3], p[4], p[5], p[6])


def overlapping_pairs(prims, tol, lattice, *, self_only=False, image_only=False):
    """Every pair of primitives that interpenetrates by more than ``tol``.

    Yields ``(i, j, image_x, image_y, gap)`` with ``i <= j``; ``image_*`` is which periodic
    copy of ``j`` was hit (``0, 0`` for the one inside the cell), and ``i == j`` means the
    shape runs into its own image, i.e. it is bigger than the period. One entry per pair —
    the first image that overlaps is enough to say the layer is ambiguous.

    ``self_only`` drops the cross terms and reports just the ``i == j`` entries, turning the
    sweep from quadratic into linear: 17 ms to 0.3 ms on a 144-shape layer, paid on every
    rebuild of a fit. It was the whole question the removed block path's painting check asked.

    ``image_only`` drops the in-cell offset for *every* pair, so a hit means the layer needs a
    periodic image and not merely an overlap. That is the question the shape paths ask before
    building: an overlap inside the cell resolves into disjoint polygons, but one that happens
    through an edge does not — :func:`~wavebit.rcwa._engine.boundary.resolve_painting` clips loops in
    unwrapped Cartesian coordinates and enumerates no images, so it answers such a layer with a
    number that does not converge (measured 0.309 / 0.632 / 0.913 in ``sum(T)`` at truncation
    order 5 / 9 / 13 against a wrapped raster's 0.997). It cannot ride on ``self_only``: two
    sub-period shapes can meet only through an edge, and neither runs into its *own* image.

    ``(0, 0)`` is the *fifth* entry of ``_IMAGES``, not the first, which matters here. Only one
    entry per pair is reported, so a pair overlapping both in the cell and through an edge can
    come back either way — asking for the images alone is what makes the answer definite.

    ``tol`` keeps *touching* out of the report: abutting rectangles and a shape exactly
    filling the period both come out at a gap of zero, and they are the same structure
    whether the layer superposes or paints.

    Bounding circles reject a pair before :func:`signed_gap` sees it. That is not just an
    optimisation on paper: most of the work here is the eight periodic images of every pair,
    which are far apart by construction, and the direction search is a thousand evaluations
    of Python arithmetic. The bound is one-sided — the true gap is never smaller than the
    circles' — so rejecting on it can only skip a pair that was never going to be reported.

    The images are the 3x3 of the **reduced** basis (:func:`~wavebit.lattice.reduced_basis`),
    whatever spelling arrives: on ``a2 = 10 a1 + (50, 700)`` the neighbour that matters sits
    at ``(-10, 1)`` of the basis as given, which a 3x3 on that basis never reaches, and a
    100 x 800 box on that cell -- which overlaps its ``(50, 700)`` image over a 50 x 100
    patch -- was accepted. ``image_*`` in the result count cells of the reduced basis too.
    """
    # The basis is read once, not per image: it is loop-invariant, and it is read the way the
    # shapes are, because `lattice` is a design variable too (2026-08-23). Spelled `float` it
    # refused a `jax.grad` tracer, and the caller reads that refusal as "skip the check" -- so
    # differentiating the *cell* walked past a refusal that differentiating the *shape* hits.
    # Measured on one 1.6-wide rod in a period of 1.0: refused eagerly, and `grad` wrt the
    # period returned 0.0265. `reduced_basis` reads it the same way, and raises the same way.
    lattice = reduced_basis(lattice)
    a1x, a2x = checked_float(lattice[0][0]), checked_float(lattice[0][1])
    a1y, a2y = checked_float(lattice[1][0]), checked_float(lattice[1][1])

    found = []
    radii = [None if p is None else _bounding_radius(p) for p in prims]
    for i, p in enumerate(prims):
        if p is None:
            continue
        for j in ((i,) if self_only else range(i, len(prims))):
            q = prims[j]
            if q is None:
                continue
            reach = radii[i] + radii[j]
            for ox, oy in _IMAGES:
                if (ox, oy) == (0, 0) and (i == j or image_only):
                    continue                      # a shape always overlaps itself
                # Images sit on the lattice, not on a rectangle: a skewed cell's neighbour
                # is ox*a1 + oy*a2. Getting this wrong does not raise -- it reports the
                # wrong pairs as overlapping, and the layer takes the wrong build path.
                dx = ox * a1x + oy * a2x
                dy = ox * a1y + oy * a2y
                shifted = _shifted(q, dx, dy)
                if math.hypot(p[1] - shifted[1], p[2] - shifted[2]) - reach >= -tol:
                    continue
                gap = signed_gap(p, shifted)
                if gap < -tol:
                    found.append((i, j, ox, oy, gap))
                    break
    return found


class Contact(namedtuple('Contact', 'i f j g m n lo hi slide h_p h_q')):
    """One record of :func:`face_pairs`: which faces, which image, the run, and the slide."""
    __slots__ = ()


def face_pairs(prims, lattice, r, tol):
    """Every run along which a face of an in-cell box shares its line, opposed, with a face of an image.

    Yields :class:`Contact` records: face ``f`` of primitive ``i`` (in the cell) and face ``g``
    of primitive ``j``'s image ``(m, n)`` are anti-parallel and collinear, and their tangential
    extents overlap over ``[lo, hi]`` -- measured along ``f``'s tangent from ``f``'s midpoint, so
    that ``[-h_p, h_p]`` is the whole of ``f``; ``slide`` is where ``g``'s midpoint sits on that
    axis, and ``h_q`` its half-length. Only runs longer than ``tol`` count: two faces meeting at
    an endpoint share a point, not a run. ``j == i`` with ``(m, n) != (0, 0)`` is the shape
    against its own image; ``j != i`` includes ``(0, 0)``, two boxes abutting inside the cell.
    Every ordered face is swept, so a coincidence is reported from both of its faces -- the
    caller that wants one record per coincidence keeps the canonical orientation, and the one
    that wants each face's *coverage* reads all of them.

    **A face is a segment, not a line, and that is the whole content of this function.** The
    image of a face under a lattice vector ``t`` lands on a parallel line offset perpendicular
    by ``t . n`` and slid *along* the face by ``t . s``. Two opposite faces therefore coincide
    over positive measure -- the material runs on unbroken and there is no interface there at
    all -- when ``|t . n|`` is the separation between them *and* ``|t . s|`` is short enough
    that the segments still overlap. :func:`overlapping_pairs` cannot answer this: the two boxes
    only *touch*, so it reports a gap of zero and passes them, which is the right answer to the
    question it asks (does the layer need a periodic image to be built) and the wrong one to
    this one (is every face of this box an interface).

    ``t . s == 0`` is not special here. It used to be: the previous mechanism folded each face
    distance by the repeat along the shape's own axis, which reaches exactly the zero-slide
    case and nothing else, and the partial run was refused by name because no closed form
    over one box reaches it -- along the run the nearest interface belongs to the neighbouring
    image. The field now evaluates every face over its ring of images, so a run is a run
    whatever its slide, and the sweep is the one that finds them.

    The sweep is over every face of every image of every box, ``16 M^2 (2r+1)^2`` Python
    comparisons: microseconds for a handful of shapes at the ``r = 1..3`` an accepted layer
    reaches (the L box, 689 x 200 on a 700 square, gives 3 -- a 49-image ring per pair).

    Round types never appear. A strictly convex boundary meets its own translate, or a box, at
    isolated points only, which is measure zero -- so ``ellipse`` / ``circle`` / ``disk`` are
    skipped rather than tested.

    ``(m, n)`` count cells of the **reduced** basis (:func:`~wavebit.lattice.reduced_basis`),
    whichever spelling ``lattice`` is, so ``r`` rings of it reach what ``r`` rings of the
    reduced cell reach and a record is the same record on every spelling of one cell. The
    primitives are absolute positions and do not care.
    """
    lattice = reduced_basis(lattice)
    a1x, a2x = checked_float(lattice[0][0]), checked_float(lattice[0][1])
    a1y, a2y = checked_float(lattice[1][0]), checked_float(lattice[1][1])
    faces = [box_faces(p) if p is not None and p[0] == BOX else None for p in prims]
    ring = images(r)

    found = []
    for i, fi in enumerate(faces):
        if fi is None:
            continue
        for f, (mx, my, nx, ny, sx, sy, h) in enumerate(fi):
            c_p = nx * mx + ny * my                    # the line, as n . X = c_p
            for j, fj in enumerate(faces):
                if fj is None:
                    continue
                for m, n in ring:
                    if j == i and m == 0 and n == 0:
                        continue                       # a face against itself is parallel
                    tx = m * a1x + n * a2x
                    ty = m * a1y + n * a2y
                    for g, (qx, qy, qnx, qny, _qsx, _qsy, qh) in enumerate(fj):
                        # Opposed normals, compared squared so the test needs no root.
                        if (nx + qnx) ** 2 + (ny + qny) ** 2 > ANTIPARALLEL_TOL2:
                            continue
                        qmx, qmy = qx + tx, qy + ty
                        if abs(nx * qmx + ny * qmy - c_p) > tol:
                            continue                   # parallel lines, not the same line
                        # q's midpoint along p's tangent; the run is where the extents overlap.
                        slid = sx * (qmx - mx) + sy * (qmy - my)
                        lo, hi = max(-h, slid - qh), min(h, slid + qh)
                        if hi - lo > tol:
                            found.append(Contact(i, f, j, g, m, n, lo, hi, slid, h, qh))
    return found


def covered_length(runs, tol):
    """Length of the union of ``(lo, hi)`` intervals -- what several images cover *between* them.

    **The union, not any single image.** Two images can take half a face each and cover it
    between them, and then there is no interface along it at all -- the material runs on. Li's
    checkerboard cell reaches that with an ordinary rectangle: on ``a1 = (1.25, 1.25)``,
    ``a2 = (1.25, -1.25)`` a 2.5 x 1.25 bar has each horizontal face covered by ``a1`` and
    ``-a2`` taking half each. A per-image reading calls that partial; the face is dead.
    """
    covered, edge = 0.0, None
    for lo, hi in sorted(runs):
        if edge is None or lo > edge + tol:
            covered += hi - lo
            edge = hi
        elif hi > edge:
            covered += hi - edge
            edge = hi
    return covered


def _outline(p, n):
    """``(x, y, nx, ny)`` rows: ``p``'s outline sampled densely, with the outward unit normal.

    A box gives ``n`` cell-centred samples per face -- the corners themselves are never
    sampled, so no sample has to choose between two normals -- and a round shape ``4n`` in
    the parameter of the ellipse, whose normal is its level-set gradient.
    """
    kind, cx, cy, a, b, ct, st = p
    if kind == BOX:
        rows = []
        for mx, my, nx, ny, sx, sy, h in box_faces(p):
            s = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * h
            rows.append(np.column_stack([mx + s * sx, my + s * sy,
                                         np.full(n, nx), np.full(n, ny)]))
        return np.concatenate(rows)
    phi = (np.arange(4 * n) + 0.5) / (4 * n) * 2.0 * np.pi
    u, v = a * np.cos(phi), b * np.sin(phi)
    gu, gv = np.cos(phi) / a, np.sin(phi) / b
    gx, gy = gu * ct - gv * st, gu * st + gv * ct
    g = np.hypot(gx, gy)
    return np.column_stack([cx + u * ct - v * st, cy + u * st + v * ct, gx / g, gy / g])


def strictly_inside(p, x, y):
    """Boolean mask: which of the points ``(x, y)`` lie strictly inside primitive ``p``."""
    kind, cx, cy, a, b, ct, st = p
    dx, dy = np.asarray(x) - cx, np.asarray(y) - cy
    u = dx * ct + dy * st
    v = -dx * st + dy * ct
    if kind == BOX:
        return (np.abs(u) < a) & (np.abs(v) < b)
    return (u / a) ** 2 + (v / b) ** 2 < 1.0


def interfaces_of(prims, lattice, r, n=1024, step=None, groups=None):
    """The exact interface set: ``(x, y, nx, ny)`` rows of Gamma, from the outlines alone.

    Every image ``(m, n)`` in :func:`images` of every primitive has its outline sampled, and a
    sample is kept only if a step of ``step`` along its **outward** normal lands strictly inside
    no image at all (over one ring more, so the outermost outlines are judged against their
    own neighbours). A face a neighbouring image covers loses exactly its covered run; a face
    facing background keeps all of it; a round shape keeps its whole boundary. Nothing here
    knows about faces, runs, folds or blends -- it is point-in-shape tests on dense samples --
    which is what qualifies it to judge :func:`face_pairs` and the field built from it.

    ``groups`` makes it material-aware: one hashable label per primitive, and a step that
    lands inside an image of a primitive with the *same* label is buried while one landing in
    a different label's image is kept -- a seam between two boxes of different material is an
    interface and its normal is the first box's outward one. Left ``None`` every primitive is
    one material and the seam is dropped, which is the reading the geometry-only tests want.
    The background is never a label: a shape whose material equals the background keeps its
    outline, as the field keeps its faces (case E), so what this judges is the field's own
    contract rather than the physics of an inert boundary.

    ``step`` defaults to ``1e-6`` of the longer basis vector. It is the oracle's own
    resolution: two faces closer than it read as one seam, which is also what a solve sees.

    Validated before it was trusted: on a 1-D grating (700 square, ``lx = 350``, ``ly = 700``)
    the field reads 0.000 deg against it over the boundary band, so the oracle carries no
    y-normal a bar with no y-face could be measured against.

    ``r`` rings of the **reduced** basis, as for :func:`face_pairs`: the plan's ring is
    counted there, and an oracle sweeping the basis as given would miss on an unreduced
    spelling the very images the plan is judged on.
    """
    lattice = reduced_basis(lattice)
    a1x, a2x = checked_float(lattice[0][0]), checked_float(lattice[0][1])
    a1y, a2y = checked_float(lattice[1][0]), checked_float(lattice[1][1])
    if step is None:
        step = 1e-6 * max(math.hypot(a1x, a1y), math.hypot(a2x, a2y))
    if groups is None:
        groups = [0] * len(prims)
    live = [(p, g) for p, g in zip(prims, groups) if p is not None]
    rows, labels = [], []
    for p, g in live:
        for m, k in images(r):
            outline = _outline(_shifted(p, m * a1x + k * a2x, m * a1y + k * a2y), n)
            rows.append(outline)
            labels.append(np.full(len(outline), hash(g)))
    if not rows:
        return np.zeros((0, 4))
    pts = np.concatenate(rows)
    label = np.concatenate(labels)
    probe_x = pts[:, 0] + step * pts[:, 2]
    probe_y = pts[:, 1] + step * pts[:, 3]
    buried = np.zeros(len(pts), dtype=bool)
    for p, g in live:
        for m, k in images(r + 1):
            hit = strictly_inside(_shifted(p, m * a1x + k * a2x, m * a1y + k * a2y),
                                  probe_x, probe_y)
            buried |= hit & (label == hash(g))
    return pts[~buried]
