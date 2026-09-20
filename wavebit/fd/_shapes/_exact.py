"""Closed forms the repository already owns: the overlap, the clipped disc, the clipped polygon.

Three of the four bodies in ``_bodies.py`` are an extrusion of a plane figure along a grid
axis, and for those the tangent plane is not the best a voxel can be told. The plane figure's
area inside the voxel's transverse square, times the extrusion's overlap with the voxel's axial
interval, is the **exact** filled volume -- no bias, no ``(h/R)^2``, and still one differentiable
expression. The same walk that clips the polygon gives the exact ``<n n^T>`` over its edges.

**These are ported, not imported.** Nothing under ``wavebit/fd/`` imports the old engines
(``wavebit/fd/_lower/cpml.py`` states the rule and the reason); what is copied here is the
arithmetic and the guards, with the source named at each one:

* :func:`overlap` <- ``wavebit/_core/fdtd.py::_overlap`` (:334), one line, arrays instead of
  a shared edge vector.
* :func:`clipped_disc_area` <- ``wavebit/_core/scene_fdtd.py::_clipped_circle_area`` (:672),
  the sharp branch: four signed-quadrant primitives in disc-centred coordinates, including the
  series that removes the cancellation of the elementary primitive's divergent derivatives near
  ``t = r``. The ``moment_weights`` half of the original (a bilinear window for the RCWA path's
  translation-averaged smoothing) is **not** carried across: stage 4 averages over the
  component's own Yee cell and nothing else (``docs/fdtd-circle-gradients.md`` prices the gap
  between the two geometries at 0.005933 in amplitude).
* :func:`clipped_polygon_area` <- ``wavebit/_core/scene_fdtd.py::_clipped_rectangle_area``
  (:306), Sutherland-Hodgman against the four faces with the stable repack that keeps the
  cyclic order without a host-side geometry plan. **Generalised from four vertices to n**: the
  original's eight-slot buffer is ``2 * 4``; four half-plane clips of an ``n``-gon produce at
  most ``n + 4`` vertices, so the buffer is that. Everything else is unchanged.
* :func:`polygon_edge_moments` <- ``wavebit/_core/scene_fdtd.py::_rectangle_edge_moments``
  (:823), Liang-Barsky parametric clipping per edge (Liang & Barsky, ACM TOG **3**, 1 (1984),
  1-22), with the same guarded normalisation of a zero-length edge. The original **refuses**
  anything but four vertices by name; the generalisation to ``n`` is the whole reason it is
  ported rather than called. Its docstring also carries the fact this stage is built on --
  that the *second* normal moment is the right object and the first is not -- and that
  paragraph is quoted in ``__init__.py`` where the choice is made.

**The coordinates are not the same and that is the other reason to port.** The RCWA path works
in a cell whose origin is its centre and whose arrays are ``(len(y), len(x))`` with a fixed
grid of cell boxes (memory: analytic shape coordinates are absolute lengths, not fractions of
the period; arrays are v-first while typed pairs are x-first). Stage 4 works in absolute scene
coordinates on an arbitrary array of points, one voxel box per point, and its typed triples are
x-first throughout. A call across that boundary would have to transpose and re-origin at every
site; the arithmetic is the same either way, and
``tests/test_fd_shapes.py::test_the_ported_closed_forms_reproduce_the_rcwa_paths_own`` holds
the two against each other over random configurations so the port cannot drift.
"""


def overlap(lo, hi, centre, dx, xp):
    """Length of ``[centre - dx/2, centre + dx/2]`` inside ``[lo, hi]``.

    Ported from ``wavebit/_core/fdtd.py::_overlap`` (:334), which walks a shared edge vector;
    here each site carries its own interval, so it is the same ``max(min - max, 0)``.
    """
    return xp.maximum(xp.minimum(hi, centre + dx / 2) - xp.maximum(lo, centre - dx / 2),
                      xp.zeros_like(centre))


def clipped_disc_area(cx, cy, radius, left, right, bottom, top, xp):
    """Exact area of ``disc(c, r)`` inside the axis-aligned box, differentiably.

    Ported from ``wavebit/_core/scene_fdtd.py::_clipped_circle_area`` (:672) -- the sharp
    branch, without the bilinear ``moment_weights`` window. Its own docstring states the
    design, and all of it is carried: integrate vertical disc chords on each signed quadrant,
    four corner primitives in disc-centred coordinates, no polygon and no quadrature; inactive
    square roots and arcsines guarded **before** evaluation so a derivative is finite; a signed
    branch chosen at zero so a centre derivative survives on the axes.
    """
    r2 = radius * radius

    def primitive(x, y):
        one = xp.ones_like(x)
        sx = xp.where(x >= 0, one, -one)
        sy = xp.where(y >= 0, one, -one)
        a, b = xp.minimum(sx * x, radius), xp.minimum(sy * y, radius)
        inside = b < radius
        split = xp.where(inside, xp.sqrt(xp.where(inside, r2 - b * b, xp.ones_like(b))),
                         xp.zeros_like(b))
        s = xp.where(a < split, a, split)

        def arc(t):
            # The elementary primitive subtracts divergent derivatives near t = r. Integrating
            # sqrt(2z - z^2) as a series at that endpoint removes the cancellation (the omitted
            # term is < 1e-22). Verbatim from the source named above.
            near = t > .99 * radius
            u = xp.where(near, xp.zeros_like(t), t / radius)
            z = xp.where(near, (radius - t) / radius, xp.zeros_like(t))
            cap = 2. ** .5 * z ** 1.5 * (2 / 3 + z * (-1 / 10 + z * (-1 / 112 + z * (-1 / 576
                  + z * (-5 / 11264 + z * (-7 / 53248 + z * (-7 / 163840 - z * 33 / 2228224)))))))
            pi_over_4 = xp.zeros_like(t) + 0.7853981633974483
            return r2 * xp.where(near, pi_over_4 - cap, .5 * (u * xp.sqrt(1 - u * u) + xp.asin(u)))

        return sx * sy * (b * s + arc(a) - arc(s))

    left, right = left - cx, right - cx
    bottom, top = bottom - cy, top - cy
    area = primitive(right, top) - primitive(left, top) - primitive(right, bottom) + primitive(left, bottom)
    return xp.where(area < 0, xp.zeros_like(area), area)


def clipped_polygon_area(vertices, left, right, bottom, top, xp):
    """Exact area of a simple polygon inside the axis-aligned box, differentiably.

    ``vertices`` is ``(..., nv, 2)``, counter-clockwise, broadcasting against the four bounds.

    Ported from ``wavebit/_core/scene_fdtd.py::_clipped_rectangle_area`` (:306) and generalised
    from ``nv = 4`` to any ``nv >= 3``: four half-plane clips of an ``nv``-gon produce at most
    ``nv + 4`` vertices, so the fixed buffer is that rather than the original's ``2 * 4``. The
    stable packing that keeps the survivors' cyclic order without a host-side plan, and the safe
    inactive denominators that keep a derivative finite on an edge parallel to a face, are the
    original's and are unchanged. Combinatorics are piecewise; an edge lying exactly on a face
    is admitted to both neighbours, as there.
    """
    nv = vertices.shape[-2]
    cap = nv + 4
    shape = xp.broadcast_shapes(left.shape, right.shape, bottom.shape, top.shape,
                                vertices.shape[:-2])
    copies = -(-cap // nv)
    polygon = xp.broadcast_to(xp.concat([vertices] * copies, axis=-2)[..., :cap, :],
                              (*shape, cap, 2))
    count = xp.zeros(shape, dtype=xp.int64) + nv
    index = xp.arange(cap)
    for axis, bound, sign in ((0, left, 1), (0, right, -1), (1, bottom, 1), (1, top, -1)):
        bound = xp.broadcast_to(bound, shape)
        previous = xp.where(index == 0, count[..., None] - 1, index - 1)
        previous = xp.maximum(previous, xp.zeros_like(previous))
        a = xp.take_along_axis(polygon, xp.broadcast_to(previous[..., None], polygon.shape), axis=-2)
        b = polygon
        da = sign * (a[..., axis] - bound[..., None])
        db = sign * (b[..., axis] - bound[..., None])
        active = index < count[..., None]
        inside_a, inside_b = da >= 0, db >= 0
        crossing = active & (inside_a != inside_b)
        denominator = xp.where(crossing, da - db, xp.ones_like(da))
        t = xp.where(crossing, da / denominator, xp.zeros_like(da))
        intersection = a + t[..., None] * (b - a)
        candidates = xp.stack((intersection, b), axis=-2).reshape(*shape, 2 * cap, 2)
        valid = xp.stack((crossing, active & inside_b), axis=-1).reshape(*shape, 2 * cap)
        order = xp.argsort(xp.astype(~valid, xp.int32), axis=-1, stable=True)[..., :cap]
        polygon = xp.take_along_axis(candidates, xp.broadcast_to(order[..., None], (*shape, cap, 2)),
                                     axis=-2)
        count = xp.sum(xp.astype(valid, xp.int64), axis=-1)
    following = xp.where(index + 1 < count[..., None], index + 1, xp.zeros_like(index))
    other = xp.take_along_axis(polygon, xp.broadcast_to(following[..., None], polygon.shape), axis=-2)
    # Cell-relative coordinates reduce cancellation for tiny clipped pieces (the source's note).
    origin = xp.stack((xp.broadcast_to(left, shape), xp.broadcast_to(bottom, shape)), axis=-1)[..., None, :]
    a, b = polygon - origin, other - origin
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    cross = xp.where(index < count[..., None], cross, xp.zeros_like(cross))
    area = .5 * xp.sum(cross, axis=-1)
    return xp.where(area < 0, xp.zeros_like(area), area)


def polygon_edge_moments(vertices, left, right, bottom, top, xp):
    """``(L, M_uu, M_vv)``: the polygon's boundary length inside the box, and ``n_c^2`` over it.

    ``M_cc`` integrates ``n_c^2`` along the shape's own boundary inside the box, so ``M/L`` is
    the mean of ``n_c^2`` there and ``M_uu + M_vv == L`` identically -- which is what makes the
    three-component invariant ``sum_c w_c == 1`` exact rather than approximate for an extruded
    body.

    Ported from ``wavebit/_core/scene_fdtd.py::_rectangle_edge_moments`` (:823) and generalised
    from four vertices to ``n`` -- the original refuses anything else by name, which is the
    reason this is a port and not a call. Liang-Barsky parametric clipping (Liang & Barsky, ACM
    TOG **3**, 1 (1984), 1-22): each face is one inequality ``slope * t <= offset`` on the
    edge's own parameter, so the four faces only ever move the entry and exit limits and never
    rebuild a polygon -- and the clipped fraction of the parameter interval times the **full**
    edge's length and the **full** edge's normal means nothing is read off a clipped piece,
    where a near-tangent chord's direction is ill-conditioned.

    The off-diagonal ``M_uv`` is one line from here and is deliberately not returned: stage 4's
    rule is the diagonal of the effective tensor's inverse, nothing reads an off-diagonal, and
    an unread return value is a value nothing can hold to account.
    """
    edge = xp.roll(vertices, -1, axis=-2) - vertices
    edge_u, edge_v = edge[..., 0], edge[..., 1]
    # The source's guarded normalisation: a zero-length edge leaves a zero normal and a zero
    # length, so it contributes nothing and differentiates without sqrt's infinite slope at 0.
    squared = edge_u * edge_u + edge_v * edge_v
    extended = squared > 0
    safe = xp.where(extended, squared, xp.ones_like(squared))
    length = xp.where(extended, xp.sqrt(safe), xp.zeros_like(squared))
    inverse = 1 / xp.sqrt(safe)
    normal_u, normal_v = -edge_v * inverse, edge_u * inverse
    start_u, start_v = vertices[..., 0], vertices[..., 1]
    tol = 32 * xp.finfo(vertices.dtype).eps * xp.max(length)
    enter = xp.zeros_like(start_u + left[..., None])
    leave = xp.ones_like(enter)
    outside = xp.zeros_like(enter, dtype=xp.bool)
    for slope, offset in ((-edge_u, start_u - left[..., None]), (edge_u, right[..., None] - start_u),
                          (-edge_v, start_v - bottom[..., None]), (edge_v, top[..., None] - start_v)):
        slope, offset = slope + xp.zeros_like(enter), offset + xp.zeros_like(enter)
        moving = xp.abs(slope) > tol
        t = offset / xp.where(moving, slope, xp.ones_like(slope))
        enter = xp.where(moving & (slope < 0), xp.maximum(enter, t), enter)
        leave = xp.where(moving & (slope > 0), xp.minimum(leave, t), leave)
        # A face the edge runs parallel to admits the whole segment or none of it.
        outside = outside | (~moving & (offset < 0))
    span = xp.where(outside, xp.zeros_like(enter), xp.maximum(leave - enter, xp.zeros_like(enter)))
    arclength = span * length
    return (xp.sum(arclength, axis=-1),
            xp.sum(arclength * normal_u * normal_u, axis=-1),
            xp.sum(arclength * normal_v * normal_v, axis=-1))
