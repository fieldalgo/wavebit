"""Two smoothed surfaces in one voxel: per-medium fills and the multi-phase rule (stage 4, C10).

Door 5 (``checks.check_smoothed``) refuses two smoothed bodies whose surfaces pass through one
voxel, because :func:`coefficients.mix` takes two phases and one surface and composes wrongly
there. This module answers such a pair where every quantity the answer needs is either exact or
the rule already used for one body, and returns an explanatory string -- which keeps door 5's
refusal -- everywhere else. The acceptance it is held to was registered before it was written:
``docs/validation/fdtd/fd-polytope-clip.md``, "Registered acceptance".

**The media of a voxel.** For the pair ``A`` (listed first) and ``B`` (listed later, so it wins
where the two overlap) the voxel holds

    B           fill  F_B
    A           fill  F_A - S
    the ground  fill  (1 - F_A) - F_B + S      what the voxel held before A was painted

with ``F_A`` and ``F_B`` the bodies' own exact fills and ``S`` the volume of the voxel they share.
``S`` is taken only where it is exact, in this order:

``disjoint``
    The bodies share no volume of the voxel, proved per voxel by the exact test of their
    supporting half spaces (:func:`_bodies._no_common_volume`; a convex body lies inside its own),
    or for the whole pair where one is a sphere whose centre is at least its radius from the
    partner's exact signed distance. ``S = 0``.
``flat pair``
    Both bodies are **flat** in the voxel -- convex, with an exact fill equal to their tangent
    half space's to :data:`_bodies._FLAT` -- so each *is* its half space there and the exact
    two-plane cut (:func:`_cut.two_plane`, validated against a ConvexHull oracle) is ``S``.
``axis-flat member``
    One member is flat with an axis-aligned normal (an unturned ``fd.Box`` face), so the part of
    the voxel on that member's side of its face is an axis-aligned box, and ``S`` is the
    partner's own closed-form volume over that box: a ball-box clip for a sphere, a clipped
    cross-section times a 1-D overlap for an extrusion.

Anything else -- curved bodies that overlap in a voxel, a crease meeting a crease -- has only a
bracket, not a value, and is refused. So is a **flush joint**, two coincident back-to-back flat
faces: which material owns the shared plane is the junction's question (``junction.py``,
``Structure(junction=)``), measured there, and a second rule for it is not offered here.

**The interface.** ``w_c`` is the second moment of every interface in the voxel: each member's own
area and moment (:meth:`Body.parts` -- exact for polygonal bodies, the tangent plane's for curved
ones, exactly as for one body), less the part of ``A``'s surface lying inside ``B``, which separates
``B`` from ``B`` and is no interface. ``B``'s whole surface separates ``B`` from ``A`` or from the
ground and is kept. The buried part is the two-plane cut's face area where both are flat; the
partner's exact cross-section on the face's plane where ``A`` is the axis-flat member; and the
area of ``A``'s tangent plane inside ``B`` where ``B`` is, which is ``A``'s own tangent-plane reading
and is admitted only where ``A``'s interface in the voxel *is* its tangent plane.

**The rule.** With ``<1/eps> = sum_m f_m / eps_m`` and ``<eps> = sum_m f_m eps_m``::

    1/eps_c = w_c <1/eps> + (1 - w_c) / <eps>                       at component c's own point
    K       = 1/<eps> I + (<1/eps> - 1/<eps>) P                     at the common node

``P`` the area-weighted mean of the members' nearest-surface ``n n^T`` (the existing node proxy,
with an unturned Box's cross entries kept zero, as ``constitutive.cross_free`` keeps them). For
parallel planar interfaces it is the **exact laminate**: the field normal to the layers sees the
harmonic mean of every layer, the field along them the arithmetic mean
(Farjadpour et al., Opt. Lett. **31**, 2972 (2006); Kottke, Farjadpour & Johnson, Phys. Rev. E
**77**, 036611 (2008), whose two-phase construction this generalises to the layered medium it is
exact for). For interfaces that are not parallel it is the same projection with ``P`` averaged
over them, as the one-body rule averages one creased surface; how well that does is what the
registered convergence study measures. **With one medium's fill exactly zero both forms are the
two-phase rule bit for bit** -- the order of every sum below is chosen for that, and
``tests/fdtd/test_fd_clip.py`` holds it with ``numpy.array_equal``.

**Static geometry, live materials.** The fills, ``w`` and ``P`` are computed once, on NumPy, while
planning, and stored in the plan under a digest like the junction's masks; lowering reads only the
three material values, on the run's backend, so a material derivative flows through the rule and
no host test reads a traced number. Traced geometry on a member is refused before this module is
reached (``shape_ranges``: a concrete partner may not reach a traced body's halo).
"""
import dataclasses

import numpy as np

#: Index of the six symmetric entries of ``P`` in a record's ``tensor``: diagonal, then
#: ``constitutive``'s off-diagonal order ``(0, 1), (0, 2), (1, 2)``.
PAIRS = ((0, 1), (0, 2), (1, 2))

#: Families of points a record covers: the three electric components, then the common node.
FAMILIES = ('Ex', 'Ey', 'Ez', 'E*')

#: Per-family count columns: written, disjoint, flat pair, axis-flat member.
CASES = ('disjoint', 'flat pair', 'axis-flat member')


@dataclasses.dataclass(frozen=True)
class Clip:
    """One clipped pair: static geometry of every written point, keyed by family.

    ``first`` is painted before ``second``. ``rows`` is the x-first cell box the arrays cover.
    ``write`` ``(4, nz, ny, nx)`` marks the written points of the three electric components and
    the common node; ``fills`` ``(4, 3, nz, ny, nx)`` holds ``(first, second, ground)`` there,
    ``weights`` ``(3, nz, ny, nx)`` each component's ``w_c`` at its own point and ``tensor``
    ``(6, nz, ny, nx)`` the node's ``P``. Points not written hold the ground alone, so the rule
    is finite and differentiable everywhere it is evaluated.
    """
    first: int
    second: int
    rows: tuple
    counts: tuple
    source: str
    media: tuple
    write: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})
    fills: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})
    weights: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})
    tensor: object = dataclasses.field(default=None, compare=False, metadata={'digest': True})


# ---------------------------------------------------------------------------------------------
# The rule. Pure arithmetic on backend values; the order of each sum is part of the contract.
# ---------------------------------------------------------------------------------------------

def means(f1, f2, fg, e1, e2, eg):
    """``(<1/eps>, <eps>)`` over the three media, summed first, ground, second.

    That order is the one :func:`coefficients.mix` writes its two terms in, so with ``f2 = 0``
    the third term adds an exact zero and both means are the two-phase ones to the bit.
    """
    harmonic = (f1 / e1 + fg / eg) + f2 / e2
    arithmetic = (f1 * e1 + fg * eg) + f2 * e2
    return harmonic, arithmetic


def diagonal(f1, f2, fg, e1, e2, eg, w):
    """``eps_c`` at one component's own point: :func:`coefficients.mix` over three media."""
    harmonic, arithmetic = means(f1, f2, fg, e1, e2, eg)
    return 1.0 / (w * harmonic + (1.0 - w) / arithmetic)


def contrast(f1, f2, fg, e1, e2, eg, arithmetic):
    """``<1/eps> - 1/<eps>`` without its cancellation, a sum of non-negative pair terms.

    ``(sum f/e)(sum f e) - (sum f)^2 = sum_{m<n} f_m f_n (e_m - e_n)^2 / (e_m e_n)`` for fills
    summing to one, divided by ``<eps>``. The first term is ``constitutive.lower``'s two-phase
    expression written in its own order, and the other two are exact zeros when ``f2 = 0``.
    """
    return ((f1 * fg * (e1 - eg) ** 2 / (e1 * eg * arithmetic)
             + f1 * f2 * (e1 - e2) ** 2 / (e1 * e2 * arithmetic))
            + f2 * fg * (e2 - eg) ** 2 / (e2 * eg * arithmetic))


def node(f1, f2, fg, e1, e2, eg, p, xp):
    """``(diagonal (3, ...), cross (3, ...), <eps>)`` of the node tensor ``1/<eps> + c P``."""
    _, arithmetic = means(f1, f2, fg, e1, e2, eg)
    c = contrast(f1, f2, fg, e1, e2, eg, arithmetic)
    diag = xp.stack([1.0 / arithmetic + c * p[a] for a in range(3)])
    cross = xp.stack([c * p[3 + k] for k in range(3)])
    return diag, cross, arithmetic


# ---------------------------------------------------------------------------------------------
# Static geometry, on NumPy, while planning.
# ---------------------------------------------------------------------------------------------

def _points(grid, box, c):
    from wavebit.fdtd._prepare.constitutive import common_points
    from wavebit.fdtd._prepare.occupancy import yee_points
    return np.asarray(yee_points(grid, box, 'E', c) if c < 3 else common_points(grid, box, 'E'))


def _axis_of(n):
    """The axis an exactly axis-aligned unit normal lies along, else -1, per point."""
    exact = np.abs(n) == 1.0
    return np.where(exact.sum(axis=-1) == 1, np.argmax(exact, axis=-1), -1)


def _box_volume(body, lo, hi):
    """``body``'s exact volume inside the axis-aligned boxes ``[lo, hi]`` (``(N, 3)`` each)."""
    from wavebit.fdtd._shapes import Sphere
    from wavebit.fdtd._shapes._bodies import AXES
    from wavebit.fdtd._shapes._bodies import _transverse
    from wavebit.fdtd._shapes._exact import clipped_ball_volume
    hi = np.maximum(hi, lo)
    if isinstance(body, Sphere):
        c = [float(v) for v in body.center]
        return clipped_ball_volume(np.full(len(lo), c[0]), np.full(len(lo), c[1]),
                                   np.full(len(lo), c[2]), np.full(len(lo), float(body.radius)),
                                   lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2], np)
    u, v = _transverse(body.axis)
    k = AXES[body.axis]
    centre = float(body.center[k])
    half = float(body.height) / 2
    probe = (lo + hi) / 2
    area = body._area((lo[:, u], hi[:, u], lo[:, v], hi[:, v]), probe)
    along = np.maximum(np.minimum(hi[:, k], centre + half) - np.maximum(lo[:, k], centre - half), 0.)
    return area * along


def _section(body, axis, plane, lo, hi):
    """``body``'s exact cross-section on ``x_axis = plane`` inside the square ``[lo, hi]``.

    ``None`` where no closed form is written here: an extrusion lying across the plane.
    """
    from wavebit.fdtd._shapes import Sphere
    from wavebit.fdtd._shapes._bodies import AXES
    from wavebit.fdtd._shapes._bodies import _transverse
    from wavebit.fdtd._shapes._exact import clipped_disc_area
    u, v = (j for j in range(3) if j != axis)
    if isinstance(body, Sphere):
        c = [float(x) for x in body.center]
        rho2 = float(body.radius) ** 2 - (plane - c[axis]) ** 2
        live = rho2 > 0
        rho = np.sqrt(np.where(live, rho2, 1.0))
        area = clipped_disc_area(np.full(len(lo), c[u]), np.full(len(lo), c[v]), rho,
                                 lo[:, u], hi[:, u], lo[:, v], hi[:, v], np)
        return np.where(live, area, 0.0)
    if AXES[body.axis] != axis:
        return None
    bu, bv = _transverse(body.axis)
    centre = float(body.center[axis])
    half = float(body.height) / 2
    probe = (lo + hi) / 2
    inside = (plane > centre - half) & (plane < centre + half)
    return np.where(inside, body._area((lo[:, bu], hi[:, bu], lo[:, bv], hi[:, bv]), probe), 0.0)


def _caps_on(body, axis, plane, dx, tol):
    """Whether an extrusion along ``axis`` has a cap on ``plane``: a flush joint."""
    from wavebit.fdtd._shapes import Sphere
    from wavebit.fdtd._shapes._bodies import AXES
    if isinstance(body, Sphere) or AXES[body.axis] != axis:
        return np.zeros(np.shape(plane), bool)
    centre = float(body.center[axis])
    half = float(body.height) / 2
    return ((np.abs(plane - (centre - half)) <= tol * dx)
            | (np.abs(plane - (centre + half)) <= tol * dx))


def _figure(body):
    """``('poly', vertices (nv, 2))`` or ``('disc', centre (2,), radius)`` of an extrusion's
    cross-section in its own transverse frame, or ``None`` where no exact test is written."""
    from wavebit.fdtd._shapes import Cylinder
    from wavebit.fdtd._shapes._bodies import _Polygonal
    from wavebit.fdtd._shapes._bodies import _transverse
    probe = np.zeros((1, 3))
    if isinstance(body, Cylinder):
        u, v = _transverse(body.axis)
        return 'disc', np.array([float(body.center[u]), float(body.center[v])]), float(body.radius)
    if isinstance(body, _Polygonal) and body._convex:
        return 'poly', np.asarray(body._vertices(probe), float).reshape(-1, 2)
    return None


def _separated(first, second):
    """Exact separating-axis test of two convex figures; touching counts as apart (no area)."""
    if first[0] == 'disc' and second[0] == 'disc':
        return float(np.hypot(*(first[1] - second[1]))) >= first[2] + second[2]
    if first[0] == 'disc' or second[0] == 'disc':
        disc, poly = (first, second) if first[0] == 'disc' else (second, first)
        ring = poly[1]
        best = np.inf
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            t = np.clip(np.dot(disc[1] - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            best = min(best, float(np.hypot(*(a + t * (b - a) - disc[1]))))
        inside = all((b - a)[0] * (disc[1] - a)[1] - (b - a)[1] * (disc[1] - a)[0] >= 0
                     for a, b in zip(ring, np.roll(ring, -1, axis=0)))
        return (not inside) and best >= disc[2]
    for ring in (first[1], second[1]):
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            normal = np.array([b[1] - a[1], a[0] - b[0]])           # outward for CCW
            if np.max(first[1] @ normal) <= np.min(second[1] @ normal) or \
                    np.max(second[1] @ normal) <= np.min(first[1] @ normal):
                return True
    return False


def _globally_apart(a, b):
    """Whether the two bodies provably share no volume anywhere -- exactly, in closed form.

    * a sphere whose centre is at least its radius from the partner's exact signed distance;
    * two extrusions along one axis whose axial spans, or whose convex cross-sections
      (separating-axis test; a disc by its distance), do not overlap;
    * two unturned Boxes, which are axis-aligned boxes, separated along some axis.

    Anything else is left to the per-voxel test of the supporting half spaces.
    """
    from wavebit.fdtd._shapes import Box
    from wavebit.fdtd._shapes import Sphere
    from wavebit.fdtd._shapes._bodies import AXES
    from wavebit.fdtd._shapes._bodies import _Extrusion
    for one, other in ((a, b), (b, a)):
        if isinstance(one, Sphere):
            centre = np.asarray([[float(v) for v in one.center]])
            d = float(np.asarray(other.surface(centre)[0])[0])
            return d >= float(one.radius)
    if isinstance(a, Box) and isinstance(b, Box) and a.angle == 0 and b.angle == 0:
        lo_a, hi_a = (np.asarray(a.center, float) + s * np.asarray(a.size, float) / 2
                      for s in (-1, 1))
        lo_b, hi_b = (np.asarray(b.center, float) + s * np.asarray(b.size, float) / 2
                      for s in (-1, 1))
        return bool(np.any((hi_a <= lo_b) | (hi_b <= lo_a)))
    if isinstance(a, _Extrusion) and isinstance(b, _Extrusion) and a.axis == b.axis:
        k = AXES[a.axis]
        span = [(float(x.center[k]) - float(x.height) / 2, float(x.center[k]) + float(x.height) / 2)
                for x in (a, b)]
        if span[0][1] <= span[1][0] or span[1][1] <= span[0][0]:
            return True
        fa, fb = _figure(a), _figure(b)
        if fa is not None and fb is not None:
            return _separated(fa, fb)
    return False


def geometry(body_a, body_b, points, dx, tol):
    """Everything the rule needs at ``points`` ``(N, 3)``, or the reason it cannot be had.

    Returns ``dict(fa, fb, fg, w (N, 3), p (N, 6), case (N,), shared, area, moment (N, 3))``
    with ``case`` indexing :data:`CASES` and ``fa`` the first body's **own medium** (its fill
    less the shared volume), or ``(reason, index)`` naming the first point that fails.
    """
    from wavebit.fdtd._shapes._bodies import _FLAT
    from wavebit.fdtd._shapes._bodies import _SAME_FACE
    from wavebit.fdtd._shapes._bodies import _no_common_volume
    from wavebit.fdtd._shapes._cut import _mean_of_the_moment
    from wavebit.fdtd._shapes._cut import _parts
    from wavebit.fdtd._shapes._cut import two_plane
    n = len(points)
    fa, area_a, moment_a = (np.asarray(v) for v in body_a.parts(points, dx))
    fb, area_b, moment_b = (np.asarray(v) for v in body_b.parts(points, dx))
    da, na = (np.asarray(v) for v in body_a.support_surface(points))
    db, nb = (np.asarray(v) for v in body_b.support_surface(points))
    plane_a, plane_b = _parts(da, na, dx), _parts(db, nb, dx)
    flat_a = bool(body_a.convex) & (np.abs(np.asarray(plane_a[0]) - fa) <= _FLAT)
    flat_b = bool(body_b.convex) & (np.abs(np.asarray(plane_b[0]) - fb) <= _FLAT)

    # A flush joint: two coincident back-to-back flat faces. The junction's question.
    flush = (flat_a & flat_b & (1 + np.sum(na * nb, axis=-1) <= _SAME_FACE)
             & (np.abs(da + db) <= _SAME_FACE * dx))
    if flush.any():
        return ('a flush joint (two coincident back-to-back faces); which medium owns the '
                'shared plane is the junction rule\'s question', int(np.argmax(flush)))

    apart = np.zeros(n, bool)
    if _globally_apart(body_a, body_b):
        apart[:] = True
    elif body_a.convex and body_b.convex:
        apart = np.asarray(_no_common_volume(da, na, db, nb, dx, np))
    volume, cut_a, _ = (np.asarray(v) for v in two_plane(da, na, db, nb, dx, plane_a[:2],
                                                         plane_b[:2]))
    pair = ~apart & flat_a & flat_b

    axis_a, axis_b = _axis_of(na), _axis_of(nb)
    lo, hi = points - dx / 2, points + dx / 2
    shared = np.zeros(n)
    buried = np.zeros(n)
    case = np.full(n, -1)
    case[apart] = 0
    shared[pair], buried[pair] = volume[pair], cut_a[pair]
    case[pair] = 1
    for member, partner, flat, axis, d, normal in (
            (0, body_b, flat_a, axis_a, da, na), (1, body_a, flat_b, axis_b, db, nb)):
        take = (case < 0) & flat & (axis >= 0)
        if not take.any():
            continue
        for k in range(3):
            sel = take & (axis == k)
            if not sel.any():
                continue
            sign = normal[sel, k]
            plane = np.where(sign > 0, points[sel, k] - d[sel], points[sel, k] + d[sel])
            if _caps_on(partner, k, plane, dx, tol).any():
                return ('a flush joint (a cap of one body on the other\'s face); which medium '
                        'owns the shared plane is the junction rule\'s question',
                        int(np.flatnonzero(sel)[0]))
            sub_lo, sub_hi = lo[sel].copy(), hi[sel].copy()
            sub_hi[:, k] = np.where(sign > 0, np.minimum(sub_hi[:, k], plane), sub_hi[:, k])
            sub_lo[:, k] = np.where(sign > 0, sub_lo[:, k], np.maximum(sub_lo[:, k], plane))
            shared[sel] = _box_volume(partner, sub_lo, sub_hi) / dx ** 3
            if member == 0:
                section = _section(partner, k, plane, lo[sel], hi[sel])
                if section is None:
                    return ('a body lying across an unturned Box face, whose section on that '
                            'face has no closed form here', int(np.flatnonzero(sel)[0]))
                # A is flat here AND fractional (door 5's mark), so its face lies strictly
                # inside the voxel and is counted whole by A's own moment: no boundary share.
                buried[sel] = section
            else:
                # A's tangent plane inside B, B being its half space here: admitted only where
                # A's interface in the voxel IS its tangent plane.
                consistent = ((np.abs(area_a[sel] - np.asarray(plane_a[1])[sel]) <= _FLAT * dx * dx)
                              & np.all(np.abs(moment_a[sel] - np.asarray(plane_a[2])[sel])
                                       <= _FLAT * dx * dx, axis=-1))
                if not consistent.all():
                    return ('a creased surface buried in an unturned Box face listed after it',
                            int(np.flatnonzero(sel)[~consistent][0]))
                buried[sel] = cut_a[sel]
            case[sel] = 2
    if (case < 0).any():
        return ('a voxel where the two bodies overlap and neither is flat in it, so their '
                'shared volume has only a bracket and not a value', int(np.argmax(case < 0)))

    f1 = fa - shared
    fg = (1.0 - fa) - fb + shared
    for name, value in (('the first body\'s', f1), ('the ground\'s', fg)):
        bad = (value < -tol) | (value > 1 + tol)
        if bad.any():
            return (f'{name} fill leaving [0, 1] ({float(value[bad][0]):.3g})',
                    int(np.argmax(bad)))
    exposed = area_a - buried
    if (exposed < -tol * dx * dx).any():
        return ('a buried area larger than the face', int(np.argmax(exposed < -tol * dx * dx)))
    # A face buried whole reads its exposed part as -3e-12 dx^2 of round-off (a disc section
    # against a square), and a negative area would give P a negative weight and K an
    # eigenvalue below its arithmetic floor. Below -tol it was refused above; above, it is zero.
    whole = exposed < 0
    exposed = np.where(whole, 0.0, exposed)
    moment = np.where(whole[:, None], 0.0, moment_a - buried[:, None] * na * na) + moment_b
    w = np.asarray(_mean_of_the_moment(exposed + area_b, moment, dx, np))
    t = np.where(exposed + area_b > 0, exposed / np.where(exposed + area_b > 0,
                                                          exposed + area_b, 1.0), 0.5)
    _, nna = (np.asarray(v) for v in body_a.surface(points))
    _, nnb = (np.asarray(v) for v in body_b.surface(points))
    from wavebit.fdtd._prepare.constitutive import cross_free
    p = np.zeros((n, 6))
    for a in range(3):
        p[:, a] = t * nna[:, a] ** 2 + (1 - t) * nnb[:, a] ** 2
    for idx, (i, j) in enumerate(PAIRS):
        pa = 0.0 if cross_free(body_a) else nna[:, i] * nna[:, j]
        pb = 0.0 if cross_free(body_b) else nnb[:, i] * nnb[:, j]
        p[:, 3 + idx] = t * pa + (1 - t) * pb
    return dict(fa=f1, fb=fb, fg=fg, w=w, p=p, case=case, shared=shared,
                area=exposed + area_b, moment=moment)


# ---------------------------------------------------------------------------------------------
# Planning: classify a door-5 pair, or say why not.
# ---------------------------------------------------------------------------------------------

def _same_material(a, b):
    """Static: one material object, or two whose ``eps`` are equal concrete numbers."""
    from wavebit.materials._time_domain import concrete
    if a is b:
        return True
    values = [concrete(v) for v in tuple(a.eps) + tuple(b.eps)]
    return None not in values and len(set(values)) == 1


def _scope(material, where):
    """The junction's material door, read as a reason: lossless, non-dispersive, isotropic,
    non-magnetic -- the media every number on this rule's page was measured with."""
    from wavebit.materials._time_domain import Material
    from wavebit.materials._time_domain import concrete
    if not isinstance(material, Material) or material.poles:
        return f'{where} is conductive or dispersive; the clip mixes lossless media only'
    if material.is_magnetic:
        return f'{where} declares mu; only the electric clip has been measured'
    eps = material.eps
    if not (eps[0] is eps[1] is eps[2]):
        values = [concrete(v) for v in eps]
        if None in values or len(set(values)) != 1:
            return f'{where} is anisotropic; the clip mixes isotropic media only'
    return ''


def classify(scene, grid, ia, ib, marks):
    """A :class:`Clip` for the door-5 pair ``(ia, ib)``, or a string saying why not.

    ``marks`` is door 5's own marking (``checks._shared_surface_voxels(..., masks=True)``) over
    the intersection of the two supports: the points where both fills are fractional. Those are
    the points written; every other point composes exactly under the painters already.
    """
    from wavebit.fdtd._prepare.checks import _clip_tolerance
    from wavebit.fdtd._prepare.coefficients import materials_of
    from wavebit.fdtd._prepare.junction import _share
    from wavebit.fdtd._prepare.junction import _signature
    from wavebit.fdtd._prepare.junction import _fractional
    background, structures = materials_of(scene)
    first, second = min(ia, ib), max(ia, ib)
    sa, sb = structures[first], structures[second]
    if 'stencil' in (sa.discretisation, sb.discretisation):
        return "discretisation='stencil' is a single-structure operator"
    for material, where in ((background, 'Scene(background=)'),
                            (sa.material, f'structures[{first}]'),
                            (sb.material, f'structures[{second}]')):
        reason = _scope(material, where)
        if reason:
            return reason
    body_a, body_b = sa.shape.body(), sb.shape.body()
    tol = _clip_tolerance(grid)
    box_a, box_b = sa.shape.support(grid), sb.shape.support(grid)
    rows = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(box_a, box_b))
    shape = tuple(hi - lo for lo, hi in reversed(rows))
    write = np.zeros((4,) + shape, bool)
    # Points not written read as all ground: the rule is evaluated over the whole box and then
    # selected, and a zero <eps> in the branch a `where` discards still carries a NaN into
    # every cotangent (the package's own rule, ``_cut._second_difference``'s docstring).
    fills = np.zeros((4, 3) + shape)
    fills[:, 2] = 1.0
    weights = np.zeros((3,) + shape)
    tensor = np.zeros((6,) + shape)
    counts, media = [], {first, second}
    overlap = False
    for c in range(4):
        mark = np.asarray(marks['E', c])
        if not mark.any():
            counts.append((FAMILIES[c], 0, 0, 0, 0))
            continue
        points = _points(grid, rows, c)[mark]
        out = geometry(body_a, body_b, points, grid.dx, tol)
        if isinstance(out, tuple):
            reason, k = out
            where = tuple(round(float(v), 6) for v in points[k])
            return f'{reason}, at the {FAMILIES[c]} point {where}'
        overlap |= bool(np.any(out['shared'] > tol))
        write[c][mark] = True
        fills[c, 0][mark], fills[c, 1][mark], fills[c, 2][mark] = out['fa'], out['fb'], out['fg']
        if c < 3:
            weights[c][mark] = out['w'][:, c]
        else:
            for k in range(6):
                tensor[k][mark] = out['p'][:, k]
        counts.append((FAMILIES[c], int(mark.sum()),
                       *(int((out['case'] == j).sum()) for j in range(3))))
        for index, other in enumerate(structures):
            if index in (first, second):
                continue
            share = _share(other.shape, points, grid)
            if not np.any(share > tol):
                continue
            if index > first:
                return (f'structures[{index}] is listed after the first pair member and reaches '
                        'a voxel the clip writes')
            if np.any(_fractional(share)):
                return f'structures[{index}] leaves a fractional ground in a voxel the clip writes'
            reason = _scope(other.material, f'structures[{index}]')
            if reason:
                return reason
            media.add(index)
    if overlap and not _same_material(sa.material, sb.material):
        from wavebit.materials._time_domain import concrete
        if any(concrete(v) is None for v in tuple(sa.material.eps) + tuple(sb.material.eps)):
            return ('the two bodies overlap and a material is traced; an overlapping pair needs '
                    'concrete materials, so that the face buried between them is decided')
    if overlap and _same_material(sa.material, sb.material):
        return ('the two bodies overlap and are one material: that is a union, whose face '
                'between them is no interface, and the CSG spelling that says so is not public')
    for array in (write, fills, weights, tensor):
        array.flags.writeable = False
    return Clip(first, second, rows, tuple(counts), _signature(scene, grid),
                tuple(sorted(media)), write, fills, weights, tensor)


def _view(box, origin):
    return tuple(slice(lo - base[0], hi - base[0])
                 for (lo, hi), base in zip(reversed(box), reversed(origin)))


def finish(records, junctions):
    """Refuse two write sets that share a point: a clip's with another clip's or a junction's."""
    from wavebit.fdtd._prepare.constitutive import _intersect
    for i, record in enumerate(records):
        for other in tuple(records[:i]) + tuple(junctions):
            box = _intersect(record.rows, other.rows)
            if box is None:
                continue
            mine = record.write[(slice(None),) + _view(box, record.rows)]
            theirs = other.write[(slice(None),) + _view(box, other.rows)]
            if np.any(mine & theirs):
                raise ValueError(
                    f'ClipOverlapRefused: the clip of structures[{record.first}] and '
                    f'structures[{record.second}] writes a point another two-surface rule writes '
                    'too; a voxel holding three smoothed surfaces is outside every measured rule. '
                    'Keep the third surface a voxel away.')
    return tuple(sorted(records, key=lambda r: (r.first, r.second)))


# ---------------------------------------------------------------------------------------------
# Lowering: live material values over the static geometry.
# ---------------------------------------------------------------------------------------------

def _values(rt, scene, record):
    structures = scene.structures
    return (rt.array(structures[record.first].material.eps[0], 'float64'),
            rt.array(structures[record.second].material.eps[0], 'float64'))


def paint(rt, record, scene, plan, out, ground, component):
    """``out`` with this record's written points of ``component`` replaced by the clip's ``eps``."""
    from wavebit.fdtd._prepare.coefficients import _paste
    xp, box = rt.xp, record.rows
    view = _view(box, ((0, 0),) * 3)
    shape = (plan.nz, plan.ny, plan.nx)
    full = xp.broadcast_to(out, shape)
    eg = xp.broadcast_to(ground, shape)[view]
    e1, e2 = _values(rt, scene, record)
    f = rt.array(np.array(record.fills[component], copy=True), 'float64')
    w = rt.array(np.array(record.weights[component], copy=True), 'float64')
    value = diagonal(f[0], f[1], f[2], e1, e2, eg, w)
    write = rt.array(np.array(record.write[component], copy=True))
    return _paste(xp, full, box, xp.where(write, value, full[view]))


def paint_nodes(rt, record, scene, plan, term, tensor, cross, held, ground):
    """The node tensor, cross terms and ``<eps>`` of this record's written common nodes."""
    from wavebit.fdtd._prepare.coefficients import _paste
    from wavebit.fdtd._prepare.constitutive import _intersect
    box = tuple(reversed(term.box))
    cut = _intersect(box, record.rows)
    if cut is None or term.field != 'E':
        return tensor, cross, held
    mine, view = _view(cut, record.rows), _view(cut, box)
    mask = record.write[(3,) + mine]
    if not mask.any():
        return tensor, cross, held
    xp = rt.xp
    eg = ground[view]
    e1, e2 = _values(rt, scene, record)
    f = rt.array(np.array(record.fills[(3, slice(None)) + mine], copy=True), 'float64')
    p = rt.array(np.array(record.tensor[(slice(None),) + mine], copy=True), 'float64')
    diag, off, arithmetic = node(f[0], f[1], f[2], e1, e2, eg, p, xp)
    local = tuple((lo - origin[0], hi - origin[0]) for (lo, hi), origin in zip(cut, box))
    write = rt.array(np.array(mask, copy=True))
    tensor = _paste(xp, tensor, local, xp.where(write[None], diag, tensor[(slice(None),) + view]))
    cross = _paste(xp, cross, local, xp.where(write[None], off, cross[(slice(None),) + view]))
    held = _paste(xp, held, local, xp.where(write, arithmetic, held[view]))
    return tensor, cross, held


def proxy_mask(plan, index, box):
    """Host-only report mask: a clipped node is not the single-body proxy."""
    from wavebit.fdtd._prepare.constitutive import _intersect
    keep = np.ones(tuple(hi - lo for lo, hi in reversed(box)), dtype=bool)
    for record in plan.clips:
        if index not in (record.first, record.second):
            continue
        cut = _intersect(box, record.rows)
        if cut is not None:
            keep[_view(cut, box)] &= ~record.write[(3,) + _view(cut, record.rows)]
    return keep


def refuse_mismatch(scene, plan):
    """The scene's clipped pairs are the plan's: same geometry signature, checked at lowering."""
    from wavebit.fdtd._prepare.junction import _signature
    from wavebit.fdtd._prepare.shape_ranges import planning_scene
    from wavebit.fdtd._prepare.shape_ranges import traced
    if not plan.clips:
        return
    if traced(scene):
        scene = planning_scene(scene, plan)
    signature = _signature(scene, plan.grid)
    if any(r.source != signature for r in plan.clips):
        raise ValueError('ClipMismatchRefused: the geometry differs from the plan its clipped '
                         'pairs were computed for; rebuild it with fd.plan(scene).')


def report_rows(plan):
    """``(first, second, written, disjoint, flat pair, axis-flat member)`` over the E families."""
    return tuple((r.first, r.second,
                  *(sum(row[k] for row in r.counts if row[0] != 'E*') for k in (1, 2, 3, 4)))
                 for r in plan.clips)


def notes(scene, plan):
    from wavebit.fdtd.diagnostics import SubpixelAccuracyWarning
    if not plan.clips:
        return []
    lines = []
    for first, second, written, apart, pair, axis in report_rows(plan):
        lines.append(
            f'structures[{first}] and structures[{second}] share {written} electric voxels '
            f'answered by the C10 clip ({apart} where the two are proved apart, {pair} where both '
            f'are flat, {axis} beside an unturned Box face): exact per-medium fills and one '
            'projection over both interfaces, exact for parallel planar layers.')
    return [(SubpixelAccuracyWarning, ' '.join(lines) + ' See '
             'docs/validation/fdtd/fd-polytope-clip.md for what was measured; it is not an error '
             'bound for this scene.')]

