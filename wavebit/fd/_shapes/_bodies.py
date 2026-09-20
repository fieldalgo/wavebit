"""The bodies: a signed distance, its gradient, and the closed forms each one happens to own.

A **body** is the smoothed, differentiable thing stage 4 adds, and it is not the same object as
a ``Shape`` in ``wavebit/fd/geometry.py``. A ``Slab`` or a ``Box`` there is cell-snapped and
answers questions about *cell indices* -- which cells it fills, which it may touch, whether it
is uniform across a layer. A body answers questions about a *point*: how far is your surface
from here, which way does it face, and -- where it can do better than a plane -- exactly how
much of this voxel do you fill. The two meet in C2, which is the only place that will know both.

Every body answers three things:

``surface(points) -> (d, n)``
    ``d`` negative inside, and ``n`` the **unit** outward normal, unit *everywhere* including
    at the points where the true gradient does not exist (a sphere's centre, a box's medial
    axis, a polygon's vertex). That is a contract ``_cut.cut_parts`` depends on: it divides by
    ``max_i |n_i| dx``, which is at least ``dx/sqrt(3)`` for a unit vector and zero for a zero
    one. Where the gradient is genuinely undefined the body names the direction it returns; the
    fill there is 0 or 1 whatever the direction is, so the choice cannot be observed.

``exact_fill(points, dx) -> fill | None``
    The exact filled fraction where a closed form is free, ``None`` otherwise -- and ``None``
    is the *common* case, which is why the plane exists.

``exact_moment(points, dx) -> (area, moment) | None``
    The exact interface area in the voxel and ``area * n_c^2`` over it. Separate from
    ``exact_fill`` because the two are not free together: a cylinder's fill is a closed form
    the repository already owns and its curved side's second moment is not.

**Where the exact forms are, and where they are not**

    body                fill                              moment
    Box, Prism          polygon-box clip x 1-D overlap    edge moments x overlap + cap areas
    Cylinder            disc-box clip x 1-D overlap       *tangent plane*
    Sphere              ball-box clip                     *tangent plane*
    Union/Intersection/
    Difference          *tangent plane*                   *tangent plane*

A cylinder's exact moment would need the second moment of the arc of a circle clipped to a
rectangle -- the angular interval set, not the area -- and the repository owns no such form;
the arc-weighted *area* forms it does own (``_weighted_circle_area``) are a bilinear window for
a different geometry. A sphere's would need the second moment of a spherical patch, which by
the divergence theorem needs the *volume's* second moments and the clipped disc's; the patch's
plain area is free that way and the moment is not (see :class:`Sphere`). Measured cost of using
the plane's moment on a cylinder instead of the truth, against 20001-point quadrature over the
true circumference
(``tests/test_fd_shapes.py::test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs``,
cut voxels only, worst ``|w - w_exact|`` then rms):

    R/h          away from the rim              on the rim
      4        1.31e-2   9.63e-3            3.98e-1   2.04e-1
      8        1.92e-2   1.02e-2            5.09e-1   2.35e-1
     16        7.19e-3   3.50e-3            5.18e-1   1.68e-1

Away from the rim the plane's normal *is* the true normal at the nearest point and the error is
the arc's own spread. It does **not** fall measurably over this sweep and the first writing of
this table said it did: the rms falls (9.63e-3, 1.02e-2, 3.50e-3) and the worst does not, and
an independent surface oracle in the C1 review read 7.6e-3 / 1.8e-2 / 6.9e-3 on the same three
resolutions -- non-monotone in both datasets. What the table shows is the worst.
**On the rim it does not converge at all.** A rim is a crease, and the plane sees exactly one
surface where the voxel holds two. The stage-4 design's section 3 says a creased two-media
interface is "handled exactly by the ``exact_fill``/moment forms"; for a cylinder's cap rim that
is not true as this table stands, and it is C2's to decide whether to write the clipped arc's
second moment or to record the price beside the rule.

**The crease is also where ``w`` jumps**, and the number to price is the cylinder's, not the
box's. Sliding a body past one fixed voxel in steps of 0.0025 cell and keeping only the steps
where *both* ends are cut voxels -- so that the allowed ``w -> 0`` step is excluded -- the
largest single-step change in ``w`` is

    sphere 0.0000, axis-aligned box 0.0000, rotated box 0.0106 (a slope, not a jump),
    cylinder cap rim 0.9961

so on a cap rim ``w`` swaps from the side wall's ``(0.5, 0.5, 0)`` to the cap's ``(0, 0, 1)``
inside one 0.0025-cell move. The design's G4 ("largest single-step change <= 1e-5") was measured
on a sphere and is out by five orders of magnitude there. The fill is smooth throughout.
"""
import numpy as _np

from wavebit._core.backend import array_namespace

from ._cut import _parts
from ._exact import (clipped_ball_volume, clipped_disc_area, clipped_polygon_area, overlap,
                     polygon_edge_moments)

#: The axis names a body may be extruded along, and the index each one is on a typed x-first
#: triple. Arrays in this package are ``[component, z, y, x]``; the triples here are x-first.
AXES = {'x': 0, 'y': 1, 'z': 2}


def readable(value):
    """``value`` as a plain NumPy array **for a check**, or ``None`` when there is no value.

    The array twin of ``wavebit._core.backend.checked_float``, and it draws the line in the same
    place and for the same measured reason: a Torch tensor is detached and read, an un-jitted
    ``jax.grad`` tracer carries a concrete primal and is read through ``stop_gradient``, and a
    ``jax.jit`` tracer has no number at all and comes back ``None``. Reading a differentiated
    value to *check* geometry is the case that most needs checking -- the memory this repository
    carries is a rod grown by gradient ascent to three times its period with every refusal
    silently skipped, because the check had been spelled ``float(x)``.

    Every door in this module is therefore skipped, and only skipped, under ``jax.jit``.
    """
    detach = getattr(value, 'detach', None)
    if detach is not None:
        value = detach()
    try:
        return _np.asarray(value, dtype=float)
    except Exception:                       # noqa: BLE001 - a tracer of some kind
        pass
    if type(value).__module__.split('.')[0] == 'jax':
        import jax                          # local: jax is optional
        try:
            return _np.asarray(jax.lax.stop_gradient(value), dtype=float)
        except Exception:                   # noqa: BLE001 - a jit tracer, as it must
            return None
    return None


def check_sizes(where, **named):
    """Refuse a non-positive length by name. ``a length is a size``, and a size is positive.

    The repository's own sentence, and its own scar: a negative ``lx`` on the RCWA path
    *subtracted* the shape instead of refusing. Measured on this kernel before these doors,
    from three constructors whose docstrings declare a length: ``Cylinder(radius=-0.2)`` read
    ``fill = 16.0`` with ``sum_c w_c = 0`` -- both invariants broken at once, because
    ``clipped_disc_area`` with ``r < 0`` returns sixteen times the box it is clipped to while
    the plane reports no interface; ``Box(size=(-0.3, 0.2, 0.2))`` read 0.0000 and
    ``Sphere(radius=-0.2)`` read 0.0000. Three bodies, three different wrong answers, no
    refusal anywhere.
    """
    for what, value in named.items():
        value = readable(value)
        if value is None:
            continue
        if not _np.all(_np.isfinite(value)):
            raise ValueError(f'{where}({what}=...): a finite length; got {value.tolist()}.')
        if _np.any(value <= 0):
            raise ValueError(f'{where}({what}=...): a length is a size and a size is positive; '
                             f'got {value.tolist()}. To place the body elsewhere use center=.')


def _crossing(p, q, r, s):
    """Do the closed segments ``pq`` and ``rs`` meet? Orientation test, collinear cases included."""
    def side(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    def between(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    d1, d2, d3, d4 = side(r, s, p), side(r, s, q), side(p, q, r), side(p, q, s)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and between(r, s, p)) or (d2 == 0 and between(r, s, q))
            or (d3 == 0 and between(p, q, r)) or (d4 == 0 and between(p, q, s)))


def polygon_shape(where, vertices):
    """``(nv, convex)`` after refusing everything ``clipped_polygon_area`` is not written for.

    Three refusals, each measured on this kernel before it had them:

    * **fewer than three vertices** -- ``clipped_polygon_area``'s own docstring says ``nv >= 3``
      and a two-vertex ``Prism`` returned 0.0000 rather than saying so.
    * **clockwise** -- ``clipped_polygon_area`` ends with the source's rounding guard
      ``where(area < 0, 0, area)``, so a reversed vertex list reads area 0 in every cell while
      nothing else in the body is orientation-sensitive: the even-odd inside test and
      ``polygon_edge_moments`` (which returns ``n_c^2``) are both orientation-blind. Measured on
      ``HEX`` against ``HEX[::-1]`` at ``h = 0.05``: volume 0.030951 (true 0.030951) against
      **0.000000**, with an identical signed distance and 240 voxels still reporting
      ``sum_c w_c = 1``. A body that is not there, with a live interface.
    * **self-intersecting** -- the shoelace sum is a *signed* area, so a figure-eight returns
      the difference of its lobes and the even-odd test disagrees with it.

    ``convex`` is returned rather than refused: it is a static promise
    (:func:`_exact.clip_capacity`) that buys the smaller clip buffer, and a simple non-convex
    cross-section is a legal body -- ``fd.Union`` of convex prisms is not the only spelling.
    """
    read = readable(vertices)
    if read is None:
        return None, False                  # a jit tracer: no value; take the safe buffer
    if read.ndim != 2 or read.shape[-1] != 2:
        raise ValueError(f'{where}(vertices=...): a sequence of (u, v) pairs, shape (nv, 2); '
                         f'got shape {read.shape}.')
    nv = read.shape[0]
    if nv < 3:
        raise ValueError(f'{where}(vertices=...): a polygon has at least three vertices; '
                         f'got {nv}.')
    if not _np.all(_np.isfinite(read)):
        raise ValueError(f'{where}(vertices=...): every vertex must be finite.')
    points = [tuple(v) for v in read]
    for i in range(nv):
        for j in range(i + 2, nv):
            if i == 0 and j == nv - 1:
                continue                    # adjacent round the ring
            if _crossing(points[i], points[(i + 1) % nv], points[j], points[(j + 1) % nv]):
                raise ValueError(f'{where}(vertices=...): a simple polygon -- edges {i} and {j} '
                                 'meet. A self-intersecting outline has no filled fraction to '
                                 'report (the shoelace sum returns the difference of its lobes '
                                 'and the even-odd inside test disagrees with it); build it as '
                                 'an fd.Union of simple pieces instead.')
    following = _np.roll(read, -1, axis=0)
    area = .5 * float(_np.sum(read[:, 0] * following[:, 1] - following[:, 0] * read[:, 1]))
    if area <= 0:
        raise ValueError(f'{where}(vertices=...): counter-clockwise, in the right-handed (u, v) '
                         f'pair across the extrusion axis; this list has signed area {area:.6g}. '
                         'Reverse it (vertices[::-1]) -- the clipped area of a clockwise list is '
                         'zero in every cell and the body silently disappears.')
    edge = following - read
    turn = _np.roll(edge, -1, axis=0)
    cross = edge[:, 0] * turn[:, 1] - edge[:, 1] * turn[:, 0]
    return nv, bool(_np.all(cross >= -1e-12 * _np.max(_np.abs(cross), initial=1.0)))


def _as(xp, value, like):
    """``value`` as an array beside ``like`` -- and untouched if it already is one (it may be traced).

    ``device=`` is read off ``like`` with a default because a JAX tracer has a ``dtype`` and no
    ``device``: inside ``jax.jit`` the placement is the trace's, not an attribute to copy.
    """
    if hasattr(value, 'dtype') and hasattr(value, 'device'):
        return value
    return xp.asarray(value, dtype=like.dtype, device=getattr(like, 'device', None))


def _transverse(axis):
    """The two axes across ``axis``, in the cyclic order that keeps ``(u, v, axis)`` right-handed.

    ``z -> (x, y)``, ``x -> (y, z)``, ``y -> (z, x)``. Right-handed is what makes
    "counter-clockwise" mean one thing for a prism whatever axis it is extruded along.
    """
    k = AXES[axis]
    return (k + 1) % 3, (k + 2) % 3


def _stack3(xp, parts, order, like):
    """A typed x-first triple from ``parts`` given for axes ``order``."""
    out = [None, None, None]
    for value, k in zip(parts, order):
        out[k] = value
    return xp.stack([p if p is not None else xp.zeros_like(like) for p in out], axis=-1)


class Body:
    """A shape that answers for a point, not for a cell. See the module docstring."""

    def surface(self, points):
        raise NotImplementedError

    def exact_fill(self, points, dx):
        return None

    def exact_moment(self, points, dx):
        return None

    def plane(self, points, dx):
        """``(fill, area, moment)`` from the tangent plane -- what every body falls back to.

        Goes to ``_cut._parts`` rather than the exported ``cut_parts``: the bodies here
        *guarantee* a unit normal, so the door on the public spelling would be a reduction and a
        host synchronisation per body per image, paid on every voxel of every solve to check a
        precondition this file establishes by construction.
        """
        d, n = self.surface(points)
        return _parts(d, n, dx)


class Sphere(Body):
    """``|x - centre| <= radius``, with an **exact** fill and the plane's moment.

    At the centre the gradient does not exist and the normal returned is ``+x``. Nothing can
    observe it: the fill there is 1 for any direction whenever ``radius >= dx sqrt(3)/2``, and a
    sphere smaller than a voxel is refused for other reasons (C5's "too few cells across").

    **Why the exact fill is written here after all.** Study S3(a) and the first draft of this
    package both decided the tangent plane was enough, on the grounds that its ``+0.250 (h/R)^2``
    over-fill is second order and one-signed, so the convergence study that is the gate fits it
    away. Two measurements taken by the C1 review overturned that. S2's companion claim -- order
    1.83 on a curved interface, the number the whole smoothing rule is sold on -- was itself
    measured with an *exact* fill, so it says nothing about a plane-filled sphere; and the
    review then measured the tangent plane's bias on top of it at 0.99x to 1.19x S2's own fitted
    errors at S2's own four resolutions, same sign. Together they read about twice the error at
    an order that still looks like two. The sphere is stage 4's pass mark (C3, Mie), so the one
    body the grade is taken on may not be the one body carrying a removable bias.

    The tangent plane stays for :class:`Union`, :class:`Intersection`, :class:`Difference` and
    anything else with no closed form; what it costs there is measured in ``_cut.py``.

    **The moment is still the plane's**, and that is not an oversight. By the divergence theorem
    the interface *area* would be free -- ``3V = R S + sum_faces d_f A_f`` with ``A_f`` a clipped
    disc -- but the second moment needs ``int (x_c - c_c)^2 dS`` over the spherical patch, which
    by the same theorem needs the volume's own second moments *and* the clipped disc's, neither
    of which this repository owns. What the plane's moment costs here is
    ``tests/test_fd_shapes.py::test_w_vanishes_only_where_the_fill_is_already_decided``'s
    measured sphere row.
    """

    def __init__(self, radius, center=(0., 0., 0.)):
        check_sizes('Sphere', radius=radius)
        self.radius, self.center = radius, center

    def exact_fill(self, points, dx):
        xp = array_namespace(points)
        centre = _as(xp, self.center, points)
        half = dx / 2
        return clipped_ball_volume(
            centre[..., 0], centre[..., 1], centre[..., 2], _as(xp, self.radius, points),
            points[..., 0] - half, points[..., 0] + half,
            points[..., 1] - half, points[..., 1] + half,
            points[..., 2] - half, points[..., 2] + half, xp) / dx ** 3

    def surface(self, points):
        xp = array_namespace(points)
        v = points - _as(xp, self.center, points)
        r2 = xp.sum(v * v, axis=-1)
        live = r2 > 0
        r = xp.where(live, xp.sqrt(xp.where(live, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        safe = xp.where(live, r, xp.ones_like(r))[..., None]
        fallback = xp.stack([xp.ones_like(r), xp.zeros_like(r), xp.zeros_like(r)], axis=-1)
        return r - _as(xp, self.radius, points), xp.where(live[..., None], v / safe, fallback)


class _Extrusion(Body):
    """A plane figure swept along a grid axis: the half of a body that Box, Prism and Cylinder share.

    The exact signed distance of the intersection of an infinite prism with a slab, in the
    spelling that is exact rather than merely conservative (Quilez's elongation form; the same
    combination the boundary path's rounded joins use)::

        d = min(max(d_r, d_a), 0) + |max((d_r, d_a), 0)|

    -- outside both, the distance is the hypotenuse and the normal mixes the two; inside at
    least one, it is the larger of the two and the normal is that one's. ``angle`` rotates the
    plane figure about the extrusion axis only, so the cross-section does not change along the
    axis. That is exactly the property ``check_absorbing_layers`` reads, and it is why a tilted
    extrusion axis is stage 5 and refused by name.
    """

    def __init__(self, height, axis, center, angle):
        if axis not in AXES:
            raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}.")
        check_sizes(type(self).__name__, height=height)
        self.height, self.axis, self.center, self.angle = height, axis, center, angle

    def _transverse_surface(self, q):
        """``(d, g)`` of the plane figure at ``q``, shape ``(..., 2)``; ``g`` a unit 2-vector.

        ``q`` is in **absolute** transverse coordinates, the same frame the clipped-area forms
        work in, so that one description of the figure serves both and a centre cannot be
        subtracted twice (it was, once, and only a direct test of the distance itself saw it:
        every volume and every AD check passed, because they were consistent with each other).
        """
        raise NotImplementedError

    def _split(self, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        k = AXES[self.axis]
        centre = _as(xp, self.center, points)
        q = xp.stack([points[..., u], points[..., v]], axis=-1)
        s = points[..., k] - centre[..., k]
        return xp, (u, v, k), q, s

    def surface(self, points):
        xp, (u, v, k), q, s = self._split(points)
        dr, g = self._transverse_surface(q)
        half = _as(xp, self.height, points) / 2
        da = xp.abs(s) - half
        one = xp.ones_like(s)
        sign = xp.where(s >= 0, one, -one)
        out = (dr > 0) & (da > 0)
        hyp = xp.sqrt(xp.where(out, dr * dr + da * da, xp.ones_like(dr)))
        d = xp.where(out, hyp, xp.maximum(dr, da))
        radial = dr > da
        safe = xp.where(out, hyp, one)
        nu = xp.where(out, dr * g[..., 0] / safe, xp.where(radial, g[..., 0], xp.zeros_like(dr)))
        nv = xp.where(out, dr * g[..., 1] / safe, xp.where(radial, g[..., 1], xp.zeros_like(dr)))
        na = xp.where(out, da * sign / safe, xp.where(radial, xp.zeros_like(dr), sign))
        return d, _stack3(xp, (nu, nv, na), (u, v, k), s)

    def _boxes(self, points, dx):
        """The voxel's transverse rectangle and axial interval, and the shape's axial span."""
        xp, (u, v, k), _, _ = self._split(points)
        centre = _as(xp, self.center, points)
        half = _as(xp, self.height, points) / 2
        return (xp, (u, v, k),
                (points[..., u] - dx / 2, points[..., u] + dx / 2,
                 points[..., v] - dx / 2, points[..., v] + dx / 2),
                (centre[..., k] - half, centre[..., k] + half, points[..., k]))

    def _area(self, box, points):
        raise NotImplementedError

    def exact_fill(self, points, dx):
        xp, _, box, (lo, hi, s) = self._boxes(points, dx)
        return self._area(box, points) * overlap(lo, hi, s, dx, xp) / dx ** 3


class _Polygonal(_Extrusion):
    """An extrusion whose cross-section is a simple counter-clockwise polygon.

    ``self._convex`` is a **static** promise about that polygon, read once at the constructor
    from concrete vertices and never from a traced value -- it sizes an array, so it could not be
    otherwise. :func:`_exact.clip_capacity` says what it buys and what the safe answer is when
    there is no value to read.
    """

    #: Set by every concrete subclass' constructor; False is always safe (a bigger buffer).
    _convex = False

    def _vertices(self, points):
        """The polygon in the grid's transverse frame, absolute, ``(nv, 2)``."""
        raise NotImplementedError

    def _transverse_surface(self, q):
        xp = array_namespace(q)
        vertices = self._vertices(q)
        nv = vertices.shape[-2]
        zero = xp.zeros_like(q[..., 0])
        best = zero + float('inf')
        near = xp.stack([zero, zero], axis=-1)
        inside = xp.zeros_like(zero, dtype=xp.bool)
        for i in range(nv):
            a, b = vertices[..., i, :], vertices[..., (i + 1) % nv, :]
            e = b - a
            w = q - a
            ee = xp.sum(e * e, axis=-1)
            live = ee > 0
            t = xp.clip(xp.sum(w * e, axis=-1) / xp.where(live, ee, xp.ones_like(ee)),
                        zero, xp.ones_like(zero))
            t = xp.where(live, t, zero)
            off = w - e * t[..., None]
            d2 = xp.sum(off * off, axis=-1)
            take = d2 < best
            best = xp.where(take, d2, best)
            near = xp.where(take[..., None], off, near)
            # Even-odd crossing count, the same test the RCWA path's membership uses: a ray
            # along +u from q crosses this edge iff the edge straddles q's v and the crossing
            # is to the right. A horizontal edge straddles nothing and is skipped by the guard.
            dv = b[..., 1] - a[..., 1]
            straddle = (a[..., 1] > q[..., 1]) != (b[..., 1] > q[..., 1])
            cross = a[..., 0] + (e[..., 0] / xp.where(straddle, dv, xp.ones_like(dv))) * (q[..., 1] - a[..., 1])
            inside = inside ^ (straddle & (q[..., 0] < cross))
        one = xp.ones_like(zero)
        sign = xp.where(inside, -one, one)
        live = best > 0
        norm = xp.where(live, xp.sqrt(xp.where(live, best, one)), one)
        # On the boundary itself the direction is genuinely undefined; +u is returned and the
        # fill is the same for any of them, because the voxel is cut through its own centre.
        fallback = xp.stack([one, zero], axis=-1)
        g = xp.where(live[..., None], sign[..., None] * near / norm[..., None], fallback)
        return sign * xp.sqrt(best), g

    def _area(self, box, points):
        xp = array_namespace(points)
        return clipped_polygon_area(self._vertices(points), *box, xp, convex=self._convex)

    def exact_moment(self, points, dx):
        xp, (u, v, k), box, (lo, hi, s) = self._boxes(points, dx)
        length, muu, mvv = polygon_edge_moments(self._vertices(points), *box, xp)
        along = overlap(lo, hi, s, dx, xp)
        # A cap contributes the whole clipped cross-section, or nothing: the face is a plane at
        # a fixed axial coordinate, so its area inside the voxel jumps as it leaves. That step
        # is the geometry's and not an artefact -- the surface really does leave the voxel --
        # and it is the reason C2 has a choice to make here; see the module docstring.
        half = dx / 2
        caps = xp.astype((lo > s - half) & (lo < s + half), points.dtype) \
            + xp.astype((hi > s - half) & (hi < s + half), points.dtype)
        cap_area = caps * self._area(box, points)
        area = length * along + cap_area
        return area, _stack3(xp, (muu * along, mvv * along, cap_area), (u, v, k), area)


class Prism(_Polygonal):
    """A polygon extruded along ``axis``: ``vertices`` are ``(u, v)`` offsets from ``center``.

    Counter-clockwise, simple, absolute lengths -- not fractions of anything (memory: analytic
    shape coordinates are absolute). ``(u, v)`` is the right-handed pair across ``axis``
    (:func:`_transverse`), so "counter-clockwise" means one thing on all three axes. All three
    of those words are refused by name when they are not true (:func:`polygon_shape`).

    **Non-convex is legal**, and gives the same number on all three backends; it costs the
    larger clip buffer (:func:`_exact.clip_capacity`) and nothing else.
    """

    def __init__(self, vertices, height, axis='z', center=(0., 0., 0.), angle=0.):
        super().__init__(height, axis, center, angle)
        _, self._convex = polygon_shape('Prism', vertices)
        self.vertices = vertices

    def _vertices(self, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        local = _as(xp, self.vertices, points)
        angle = _as(xp, self.angle, points)
        co, si = xp.cos(angle), xp.sin(angle)
        centre = _as(xp, self.center, points)
        return xp.stack([local[..., 0] * co - local[..., 1] * si + centre[..., u],
                         local[..., 0] * si + local[..., 1] * co + centre[..., v]], axis=-1)


class Box(_Polygonal):
    """A rectangular body: ``size`` across all three axes, rotatable about ``axis`` by ``angle``.

    It is the four-vertex :class:`Prism` and shares every line of its machinery; it exists under
    its own name because ``size=(a, b, c)`` is the spelling a user writes for a rectangle and
    a vertex list is not. **Not** ``wavebit.fd.geometry.Box``, which is the cell-snapped region
    a monitor or a ``TFSF`` takes -- the two are different discretisations under one word today,
    and the plan's C4 is where that is resolved (``fd.Box`` becomes this, the snapped one
    becomes ``fd.Region``). Nothing imports both while the word means two things.
    """

    #: A rectangle, built here from ``size``, so convexity is a property of the constructor.
    _convex = True

    def __init__(self, size, center=(0., 0., 0.), angle=0., axis='z'):
        if axis in AXES:
            check_sizes('Box', size=size)
        super().__init__(size[AXES[axis]] if axis in AXES else None, axis, center, angle)
        self.size, self._uv = size, _transverse(axis)

    def _vertices(self, points):
        xp = array_namespace(points)
        u, v = self._uv
        size = _as(xp, self.size, points)
        a, b = size[..., u] / 2, size[..., v] / 2
        zero = xp.zeros_like(a)
        local = xp.stack([xp.stack([-a, -b], axis=-1), xp.stack([a, -b], axis=-1),
                          xp.stack([a, b], axis=-1), xp.stack([-a, b], axis=-1)], axis=-2)
        angle = _as(xp, self.angle, points) + zero
        co, si = xp.cos(angle), xp.sin(angle)
        centre = _as(xp, self.center, points)
        return xp.stack([local[..., 0] * co - local[..., 1] * si + centre[..., u],
                         local[..., 0] * si + local[..., 1] * co + centre[..., v]], axis=-1)


class Cylinder(_Extrusion):
    """A disc extruded along ``axis``.

    ``angle`` is accepted because the stage-4 API names it, and it is a rotation of a circle:
    measured here, the gradient of a loss over 4096 voxels with respect to it is **exactly
    0.0** (``tests/test_fd_shapes.py::test_a_circular_cylinders_angle_carries_no_gradient``).
    That is the same defect the design
    refuses by name on ``Sphere(angle=)`` -- a key whose gradient cannot be anything but zero,
    the ``circle``/``ly`` lesson at ``wavebit/_core/parametric.py:348-355``. C5 should either
    refuse it on key *presence* as it does for a sphere, or the cross-section should become an
    ellipse, which would give the key something to turn.
    """

    def __init__(self, radius, height, axis='z', center=(0., 0., 0.), angle=0.):
        super().__init__(height, axis, center, angle)
        check_sizes('Cylinder', radius=radius)
        self.radius = radius

    def _transverse_surface(self, q):
        xp = array_namespace(q)
        u, v = _transverse(self.axis)
        centre = _as(xp, self.center, q)
        q = q - xp.stack([centre[..., u], centre[..., v]], axis=-1)
        r2 = xp.sum(q * q, axis=-1)
        live = r2 > 0
        r = xp.where(live, xp.sqrt(xp.where(live, r2, xp.ones_like(r2))), xp.zeros_like(r2))
        safe = xp.where(live, r, xp.ones_like(r))[..., None]
        zero = xp.zeros_like(r)
        fallback = xp.stack([xp.ones_like(r), zero], axis=-1)
        return r - _as(xp, self.radius, q), xp.where(live[..., None], q / safe, fallback)

    def _area(self, box, points):
        xp = array_namespace(points)
        u, v = _transverse(self.axis)
        centre = _as(xp, self.center, points)
        return clipped_disc_area(centre[..., u], centre[..., v], _as(xp, self.radius, points),
                                 *box, xp)


class _Combination(Body):
    """A CSG node. The rule is on the signed distance, resolved once, before any fill is taken.

    ``Union = min``, ``Intersection = max``, ``Difference(a, b, ...) = max(d_a, -d_b, ...)``.
    Painting order *is* this: structure ``k``'s effective body is the difference of it and
    everything after it, which on cell-snapped geometry reproduces ``owner_map``'s last-wins
    exactly and on smoothed geometry makes the overlap exact and differentiable, with no
    second algebra competing with the first.

    The normal is the **selected** child's, so it is a unit vector by induction. The selection
    is C0 and not C1 at a tie -- the crease where two surfaces meet -- which is a real crease in
    the geometry and not an artefact of the spelling. No exact fill and no exact moment: an
    exact clip of a CSG node does not exist, which is the reason the tangent plane is the
    kernel's default and not its fallback.

    **This is the one class whose ``w`` rests entirely on this normal**, and the C1 review found
    that nothing held it: a mutant returning ``self.bodies[0]``'s normal from every node passed
    the whole gate while moving ``w`` by 0.88 in 616 voxels. It is now held by
    ``tests/test_fd_shapes.py::test_the_signed_distance_is_the_distance``, which reads each
    node's distance a second way (``min`` / ``max`` / ``max`` of the negation over the children's
    own closed forms) and its normal against a central difference of that distance. The
    *direction* of a normal is read nowhere else in this package -- ``_cut.cut_parts`` consumes
    ``n`` only as ``|n| dx`` and ``n n^T`` -- so that test is also the only thing holding
    :class:`Difference`'s reversal, and without it ``-other_n`` is a line with nothing but a
    docstring behind it.
    """

    def __init__(self, *bodies):
        if len(bodies) < 2:
            raise ValueError('a CSG node combines at least two bodies.')
        self.bodies = bodies

    def _compose(self, d, n, other_d, other_n, xp):
        raise NotImplementedError

    def surface(self, points):
        xp = array_namespace(points)
        d, n = self.bodies[0].surface(points)
        for body in self.bodies[1:]:
            d, n = self._compose(d, n, *body.surface(points), xp)
        return d, n


class Union(_Combination):
    """Everything in any of them."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d < d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)


class Intersection(_Combination):
    """Only what is in all of them."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = other_d > d
        return xp.where(take, other_d, d), xp.where(take[..., None], other_n, n)


class Difference(_Combination):
    """The first, minus everything after it. The removed body's normal comes back reversed."""

    def _compose(self, d, n, other_d, other_n, xp):
        take = -other_d > d
        return xp.where(take, -other_d, d), xp.where(take[..., None], -other_n, n)
