"""Do two analytic shapes overlap? — the geometry behind the layer-semantics check.

:mod:`wavebit._core.analytic` superposes its shapes and the block path paints them, so the
two only describe the same structure while no two shapes intersect. That makes "do these
intersect?" the question the modeling layer has to answer before it can let a layer through,
and this module answers it for the analytic primitives: rectangle / square (a rotated box) and
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
"""
import math

#: What a layer may say about shapes that overlap. Both shape paths read this: the analytic
#: path superposes (``'add'``), the block path paints later shapes over earlier ones
#: (``'paint'``), and neither can build the other's rule. A recipe that means one of them has
#: to be able to say so, or moving it between paths silently changes the structure.
COMBINE_MODES = ('add', 'paint')

#: Penetration below this fraction of the period is touching, not overlapping.
OVERLAP_TOL_REL = 1e-6

_INV_PHI = (5.0 ** 0.5 - 1.0) / 2.0

BOX = 'box'
ELLIPSE = 'ellipse'

_BOX_TYPES = ('rectangle', 'square')
_ELLIPSE_TYPES = ('circle', 'disk', 'ellipse')

# Periodic images of the second shape, in units of the period. A shape reaching past one cell
# on both sides at once is already reported against its own image, so one ring is enough.
_IMAGES = tuple((ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1))


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
    return (kind, float(cx), float(cy), float(lx) / 2.0, float(ly) / 2.0,
            math.cos(float(angle)), math.sin(float(angle)))


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


def overlapping_pairs(prims, tol, lattice, *, self_only=False):
    """Every pair of primitives that interpenetrates by more than ``tol``.

    Yields ``(i, j, image_x, image_y, gap)`` with ``i <= j``; ``image_*`` is which periodic
    copy of ``j`` was hit (``0, 0`` for the one inside the cell), and ``i == j`` means the
    shape runs into its own image, i.e. it is bigger than the period. One entry per pair —
    the first image that overlaps is enough to say the layer is ambiguous.

    ``self_only`` drops the cross terms and reports just the ``i == j`` entries. That is the
    whole question the block path's painting check asks — it paints a genuine overlap and only
    a shape's own image defeats it — and it turns the sweep from quadratic into linear: 17 ms
    to 0.3 ms on a 144-shape layer, paid on every rebuild of a fit.

    ``tol`` keeps *touching* out of the report: abutting rectangles and a shape exactly
    filling the period both come out at a gap of zero, and they are the same structure
    whether the layer superposes or paints.

    Bounding circles reject a pair before :func:`signed_gap` sees it. That is not just an
    optimisation on paper: most of the work here is the eight periodic images of every pair,
    which are far apart by construction, and the direction search is a thousand evaluations
    of Python arithmetic. The bound is one-sided — the true gap is never smaller than the
    circles' — so rejecting on it can only skip a pair that was never going to be reported.
    """
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
                if i == j and (ox, oy) == (0, 0):
                    continue                      # a shape always overlaps itself
                # Images sit on the lattice, not on a rectangle: a skewed cell's neighbour
                # is ox*a1 + oy*a2. Getting this wrong does not raise -- it reports the
                # wrong pairs as overlapping, and the layer takes the wrong build path.
                dx = ox * float(lattice[0][0]) + oy * float(lattice[0][1])
                dy = ox * float(lattice[1][0]) + oy * float(lattice[1][1])
                shifted = _shifted(q, dx, dy)
                if math.hypot(p[1] - shifted[1], p[2] - shifted[2]) - reach >= -tol:
                    continue
                gap = signed_gap(p, shifted)
                if gap < -tol:
                    found.append((i, j, ox, oy, gap))
                    break
    return found
