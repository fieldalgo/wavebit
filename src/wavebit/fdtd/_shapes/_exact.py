"""Closed forms: the overlap, the clipped disc, the clipped polygon -- and the clipped ball.

Three of the four bodies in ``_bodies.py`` are an extrusion of a plane figure along a grid
axis, and for those the tangent plane is not the best a voxel can be told. The plane figure's
area inside the voxel's transverse square, times the extrusion's overlap with the voxel's axial
interval, is the **exact** filled volume -- no bias, no ``(h/R)^2``, and still one differentiable
expression. The same walk that clips the polygon gives the exact ``<n n^T>`` over its edges.
The fourth body, the sphere, is the one this repository owned no form for, and
:func:`clipped_ball_volume` is written here (see its own docstring for the derivation and for
why it is written rather than left to the plane).

**Four of the five are ported, not imported.** Nothing under ``src/wavebit/fd/`` imports the old
engines (``src/wavebit/fdtd/_prepare/cpml.py`` states the rule and the reason); what is copied here is
the arithmetic and the guards, with the source named at each one. :func:`clipped_ball_volume` is
the exception and has no source: it is derived in this file, and it mirrors
:func:`clipped_disc_area`'s structure (an odd-extended corner primitive summed over the box's
corners) one dimension up.

* :func:`overlap` <- ``src/wavebit/fdtd/_periodic/legacy.py::_overlap`` (:334), one line, arrays instead of
  a shared edge vector.
* :func:`clipped_disc_area` <- ``src/wavebit/fdtd/_periodic/scene.py::_clipped_circle_area`` (:672),
  the sharp branch: four signed-quadrant primitives in disc-centred coordinates, including the
  series that removes the cancellation of the elementary primitive's divergent derivatives near
  ``t = r``. The ``moment_weights`` half of the original (a bilinear window for the RCWA path's
  translation-averaged smoothing) is **not** carried across: stage 4 averages over the
  component's own Yee cell and nothing else (``docs/validation/fdtd/fdtd-circle-gradients.md`` prices the gap
  between the two geometries at 0.005933 in amplitude).
* :func:`clipped_polygon_area` <- ``src/wavebit/fdtd/_periodic/scene.py::_clipped_rectangle_area``
  (:306), Sutherland-Hodgman against the four faces with the stable repack that keeps the
  cyclic order without a host-side geometry plan. **Generalised from four vertices to n**, and
  from convex to any simple polygon; :func:`clip_capacity` carries the two buffer bounds and
  the proof of the second one.
* :func:`polygon_edge_moments` <- ``src/wavebit/fdtd/_periodic/scene.py::_rectangle_edge_moments``
  (:823), Liang-Barsky parametric clipping per edge (Liang & Barsky, ACM TOG **3**, 1 (1984),
  1-22), with the same guarded normalisation of a zero-length edge. The original **refuses**
  anything but four vertices by name; the generalisation to ``n`` is the whole reason it is
  ported rather than called. Boundary-aligned edges additionally share half their length
  with each adjacent voxel, as axial caps do through :func:`plane_share`.
  Its docstring also carries the fact this stage is built on --
  that the *second* normal moment is the right object and the first is not -- and that
  paragraph is quoted in ``__init__.py`` where the choice is made.

**The coordinates are not the same and that is the other reason to port.** The RCWA path works
in a cell whose origin is its centre and whose arrays are ``(len(y), len(x))`` with a fixed
grid of cell boxes (memory: analytic shape coordinates are absolute lengths, not fractions of
the period; arrays are v-first while typed pairs are x-first). Stage 4 works in absolute scene
coordinates on an arbitrary array of points, one voxel box per point, and its typed triples are
x-first throughout. A call across that boundary would have to transpose and re-origin at every
site; the arithmetic is the same either way, and
``tests/fdtd/test_fd_shapes.py::test_the_ported_closed_forms_reproduce_the_rcwa_paths_own`` holds
the two against each other over random configurations so the port cannot drift.
"""


def overlap(lo, hi, centre, dx, xp):
    """Length of ``[centre - dx/2, centre + dx/2]`` inside ``[lo, hi]``.

    Ported from ``src/wavebit/fdtd/_periodic/legacy.py::_overlap`` (:334), which walks a shared edge vector;
    here each site carries its own interval, so it is the same ``max(min - max, 0)``.
    """
    return xp.maximum(xp.minimum(hi, centre + dx / 2) - xp.maximum(lo, centre - dx / 2),
                      xp.zeros_like(centre))


def plane_share(position, centre, dx, xp):
    """Share of a plane in a voxel: half on either boundary, one strictly inside.

    Splitting a boundary face between its neighbours preserves total surface area
    and reflection symmetry. Reuse the polygon clip's 32-epsilon arithmetic margin,
    scaled by the coordinates involved in the subtraction. Translating the .4 by .2
    bar by .137 moved a face by one ulp and changed its old moment by 1/3.
    """
    scale = xp.maximum(xp.maximum(xp.abs(position), xp.abs(centre)), dx+xp.zeros_like(centre))
    tol = 32*xp.finfo(centre.dtype).eps*scale
    distance = xp.abs(position-centre)-dx/2
    return xp.where(xp.abs(distance) <= tol, .5,
                    xp.astype(distance < 0, centre.dtype))


def clipped_disc_area(cx, cy, radius, left, right, bottom, top, xp):
    """Exact area of ``disc(c, r)`` inside the axis-aligned box, differentiably.

    Ported from ``src/wavebit/fdtd/_periodic/scene.py::_clipped_circle_area`` (:672) -- the sharp
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


def clip_capacity(nv, convex):
    """How many slots the Sutherland-Hodgman buffer needs for an ``nv``-gon, and why.

    One clip against a half plane emits, per edge of the current polygon, the crossing point if
    the edge crosses and the end point if the end point is inside. So with ``i`` vertices inside
    the half plane, ``o = n - i`` outside and ``x`` crossings::

        n' = i + x

    and ``x`` is the number of sign changes around a **closed** ring, hence even, hence
    ``x = 2 r`` where ``r`` is the number of maximal inside runs. Every inside run holds at least
    one inside vertex and every outside run at least one outside vertex, so ``r <= min(i, o)``
    and ``n' <= i + 2 min(i, o) <= n + n//2``. That bound uses nothing but the sign sequence, so
    it holds for **any** simple polygon, convex or not, and four clips give
    ``n -> n + n//2`` four times over::

        nv          3    4    6    8   12   16
        convex      7    8   10   12   16   20
        simple     13   19   28   38   60   81

    For a **convex** polygon the standard ``nv + 4`` holds instead: the clipped figure is convex,
    its vertices are original vertices or crossings, and each of the four face lines meets a
    convex boundary at most twice while removing at least one original vertex whenever it meets
    it twice. That is the bound the RCWA source
    (``src/wavebit/fdtd/_periodic/scene.py::_clipped_rectangle_area``, whose ``2 * 4`` is this bound for
    a quadrilateral) was written against, and it is **only** a convex bound -- the C1 review
    measured what happens when it is used anyway: a 12-vertex star overflows the buffer, NumPy
    raises ``IndexError``, JAX returns ``nan`` (``where(area < 0, 0, area)`` does not catch a
    NaN) and Torch returns a number wrong by up to 20.8%. One legal input, three answers, which
    is the one thing "one body of code for three backends" may not do.

    The choice is therefore made **statically**, at the body's constructor, from concrete
    vertices (``_bodies._Polygonal``); a polygon whose vertices cannot be read -- a ``jax.jit``
    tracer -- gets the simple bound, which is correct for both. The price of the simple bound is
    the argsort over ``2 cap`` slots per clip, in time *and in memory*; measured for a hexagonal
    prism over 110592 voxels it is the difference recorded in ``tests/fdtd/test_fd_shapes.py``'s
    module docstring, where the memory column is the one that decides how C2 must chunk.

    **The promise that sizes this buffer has to be free of the coordinate scale**, and the first
    writing's was not: the turn test compared a cross product against an absolute floor, so the
    same star in metres was called convex, took ``nv + 4`` and overflowed it -- the three
    answers above, from the same body in a different unit. ``_bodies._TURN_SINE`` is now a sine,
    and ``tests/fdtd/test_fd_shapes.py::test_every_geometric_predicate_is_free_of_the_coordinate_scale``
    sweeps fifteen decades.
    """
    if convex:
        return nv + 4
    for _ in range(4):
        nv += nv // 2
    return nv


_PI = 3.141592653589793


def _root(v, xp):
    """``sqrt(max(v, 0))`` with the branch guarded **before** the root, not after.

    ``sqrt(maximum(v, 0))`` is right in forward mode and NaN in reverse: at ``v <= 0`` the
    ``maximum`` hands ``sqrt`` an exact zero, whose derivative is infinite, and the zero
    cotangent times that infinity is a NaN -- the same ``0 * inf`` the rest of this package
    guards with ``where(live, sqrt(where(live, ...)), 0)``. Measured: the first writing of
    :func:`clipped_ball_volume` spelled it the short way and ``jax.grad`` of a sphere's radius
    over 4096 voxels returned ``nan`` against a finite difference of 1056.3.

    The radicand must also have a stable sign. In particular, callers factor differences of
    squares BEFORE calling this function. At a wedge endpoint ``t = sqrt(k2)``, the old
    ``k2 - t*t`` changed sign when XLA contracted or recomputed the arithmetic in different
    fused kernels. The inner and outer ``where`` then disagreed: the inactive value 1 reached
    the result. This was a finite wrong answer, not NaN propagation or float32 arithmetic.
    At R=0.3, dx=0.05, p=(0.00161135212954675, -0.02089849705058622,
    -0.3005275396021332), a sphere inside a jitted CSG returned -200.9404 instead of 0.4608063.
    Changing ONLY the inactive radicand from 1 to 4 or 0.25 changed that wrong answer to
    -186.0971 or -203.4363. With factored radicands all three give the same physical answer.
    See ``tests/fdtd/test_fd_sphere_clip_jit.py`` and ``docs/validation/fdtd/fd-sphere-clip-jit.md`` for the sweep,
    the causal diagnostic, and the restored-formula mutants. Keep the derivative guard here;
    replacing its inactive value by zero merely hides the forward defect and restores NaN AD.
    """
    live = v > 0
    return xp.where(live, xp.sqrt(xp.where(live, v, xp.ones_like(v))), xp.zeros_like(v))


def _angle(y, x, xp):
    """``atan2(y, x)``, with the origin guarded: ``(0, 0)`` is a zero angle and a zero gradient.

    Every ``atan2`` below is an angle between two lengths that vanish together at the end of a
    regime -- ``a`` and ``sqrt(rho^2 - a^2)`` at ``a = 0, rho = 0``. The value there is
    multiplied by a coefficient that is also zero, so forward mode is right either way; the
    reverse-mode derivative of ``atan2`` at the origin is ``0/0``, and on JAX (not on Torch,
    which returns zero) that NaN survives the multiplication and poisons the whole gradient.
    Same ``0 * inf`` family as :func:`_root`, found the same way.
    """
    live = (y * y + x * x) > 0
    return xp.where(live, xp.atan2(xp.where(live, y, xp.zeros_like(y)),
                                   xp.where(live, x, xp.ones_like(x))), xp.zeros_like(y))


def _arc_integral(z, k, xp):
    """``int_0^z sqrt(k^2 - t^2) dt``, with ``z`` clamped to ``k`` so the branch is never left."""
    t = xp.minimum(z, k)
    # t <= k: this is non-negative and EXACTLY zero at the endpoint, without subtracting
    # independently rounded squares. The same identity is essential in _wedge_integral.
    rest = _root((k - t) * (k + t), xp)
    return .5 * (t * rest + k * k * _angle(t, rest, xp))


def _cubic_integral(z, radius, xp):
    """``int_0^z (R^2 - t^2) dt`` -- the area of the disc chord swept along the axis."""
    return radius * radius * z - z ** 3 / 3


def _wedge_integral(z, a, radius, xp):
    """``int_0^z (R^2 - t^2) asin(a / sqrt(R^2 - t^2)) dt``, for ``0 <= z <= sqrt(R^2 - a^2)``.

    Integration by parts against ``R^2 t - t^3/3``; the remaining integral splits into
    ``int t^2/sqrt(k^2-t^2)``, ``int dt/sqrt(k^2-t^2)`` and
    ``int dt/((R^2-t^2) sqrt(k^2-t^2))`` with ``k^2 = R^2 - a^2``, the last of which is
    ``(1/(aR)) atan(t a /(R sqrt(k^2-t^2)))`` because ``k < R``.

    **Every inverse trigonometric argument is written as an ``atan2`` of two lengths, never as
    an ``asin`` of a ratio.** ``asin(a/rho)`` at the end of the wedge's own range has
    ``a/rho -> 1``, where ``asin`` loses half of its digits: measured, the first writing of this
    module put a jump of 5.5e-08 into the volume at the regime boundary -- 2.4e-06 relative --
    and the same expression as ``atan2(a, sqrt(rho^2 - a^2))`` is exact to 5.6e-16 against
    quadrature. It is the same defect ``clipped_disc_area``'s series near ``t = r`` exists for,
    in the place where a series is not needed because the stable spelling is free.
    """
    k2 = (radius - a) * (radius + a)
    k = _root(k2, xp)
    t = xp.minimum(z, k)
    # Do not restore k2 - t*t: squaring sqrt(k2) loses the exact endpoint, and the two
    # evaluations of _root's predicate can disagree under JIT. See _root's measured case.
    rest = _root((k - t) * (k + t), xp)
    return (_cubic_integral(t, radius, xp) * _angle(a, rest, xp)
            - (a / 6) * (k2 * _angle(t, rest, xp) - t * rest)
            + (2 * a * radius * radius / 3) * _angle(t, rest, xp)
            - (2 * radius ** 3 / 3) * _angle(t * a, radius * rest, xp))


def _ball_corner(a, b, c, radius, xp):
    """``vol(ball(R) ∩ [0,a] x [0,b] x [0,c])`` for ``a, b, c >= 0`` -- the cumulative primitive.

    Integrate the **clipped disc** along the axis: at height ``t`` the cross-section is the
    quarter-plane rectangle ``[0,a] x [0,b]`` against a disc of radius ``rho = sqrt(R^2 - t^2)``,
    whose area is exactly :func:`clipped_disc_area`'s first-quadrant primitive. That area has
    four regimes in ``t``, separated by ``rho^2 = a^2 + b^2``, ``rho = a`` and ``rho = b``::

        rho^2 >= a^2 + b^2           a b                            (the rectangle is inside)
        a, b <= rho < hypot(a, b)    (a t_a + b t_b
                                      + rho^2 (atan2(a,t_a) + atan2(b,t_b) - pi/2)) / 2
        min(a,b) <= rho < max(a,b)   (m t_m + rho^2 atan2(m, t_m)) / 2,  m = min(a, b)
        rho < min(a, b)              pi rho^2 / 4                   (the disc is inside)

    with ``t_a = sqrt(R^2 - a^2 - t^2)``. Each regime's antiderivative in ``t`` is elementary
    (:func:`_arc_integral`, :func:`_cubic_integral`, :func:`_wedge_integral`), and the four
    pieces are summed between limits clamped to ``[0, min(c, R)]``, so an empty regime
    contributes exactly zero rather than being branched away.

    **An empty regime's limit is a constant zero, not ``min(0, c)``.** The value is the same,
    but at ``c = 0`` -- a box face through the centre plane on the integration axis -- the
    clamp ties two zeros, and ``minimum`` hands half the derivative to each side on JAX and
    on Torch. ``F_c(a, b, 0)`` is the cross-section's area at ``t = 0``; with the tie, every
    regime that is empty there contributed half of a derivative it does not have. Measured on
    a ball of radius .22 at ``dx = .1``: per-voxel centre derivatives up to 2.2 against a
    central difference of 0.41, and the energy derivative of a sphere centred on a half-node
    plane read 28.7 (JAX) against 20.5 (``docs/validation/fdtd/fd-subpixel-occupancy.md``, the centre tie).
    A regime that is present has a positive limit, so ``min(limit, 0)`` picks ``c`` without a
    tie and its derivative, one, is the true one.
    """
    a = xp.minimum(a, radius)
    b = xp.minimum(b, radius)
    c = xp.minimum(c, radius)
    ka = _root((radius - a) * (radius + a), xp)
    kb = _root((radius - b) * (radius + b), xp)
    zero = xp.zeros_like(c)
    # sqrt(R^2-a^2-b^2) = sqrt((ka-b)(ka+b)). Carry the same ka into the regime boundary
    # and its antiderivatives, so their shared endpoint is a subtraction of equal lengths.
    corner = _root((ka - b) * (ka + b), xp)
    rect = xp.where(corner > 0, xp.minimum(corner, c), zero)
    shorter = xp.minimum(ka, kb)
    both = xp.where(shorter > 0, xp.minimum(shorter, c), zero)
    lesser, outer = xp.minimum(a, b), xp.maximum(ka, kb)
    one = xp.where(outer > 0, xp.minimum(outer, c), zero)

    def two_chords(z):
        return .5 * (a * _arc_integral(z, ka, xp) + b * _arc_integral(z, kb, xp)
                     + _wedge_integral(z, a, radius, xp) + _wedge_integral(z, b, radius, xp)
                     - (_PI / 2) * _cubic_integral(z, radius, xp))

    def one_chord(z):
        return .5 * (lesser * _arc_integral(z, outer, xp)
                     + _wedge_integral(z, lesser, radius, xp))

    return (a * b * rect
            + (two_chords(both) - two_chords(rect))
            + (one_chord(one) - one_chord(both))
            + (_PI / 4) * (_cubic_integral(c, radius, xp) - _cubic_integral(one, radius, xp)))


def _fold(v, xp):
    """``(s, |v|)`` with ``s = +1`` where ``v >= 0``, spelt ``s v + 0`` rather than ``abs(v)``.

    The same number to the bit: ``s v`` is ``|v|`` exactly, and ``+ 0.`` turns the ``-0.0`` a
    ``v = -0.0`` leaves into ``+0.0``. The derivative differs at ``v = 0``, the only place it
    can: ``abs`` gives 0 on Torch (1 on JAX), ``s v`` gives ``s = +1`` on both. An odd extension
    ``s F(|v|)`` has the even derivative ``F'(|v|)``, which is ``F'(0)`` at zero, not zero.
    """
    one = xp.ones_like(v)
    s = xp.where(v >= 0, one, -one)
    return s, s * v + 0.


def clipped_ball_volume(cx, cy, cz, radius, x0, x1, y0, y1, z0, z1, xp):
    """Exact volume of ``ball(c, R)`` inside the axis-aligned box, differentiably.

    Written here rather than ported: this repository owned no sphere-box clip. The structure is
    the three-dimensional twin of :func:`clipped_disc_area`'s -- a cumulative primitive in
    disc-centred (here ball-centred) coordinates, extended oddly in each coordinate by
    ``sgn(a) sgn(b) sgn(c)``, and summed over the box's eight corners with alternating signs.
    :func:`_ball_corner` is the primitive and carries the derivation.

    **The two clamps are the function, not an approximation of it.** A box whose farthest corner
    is inside the ball is full and a box whose nearest point is outside is empty, and in both the
    true derivative with respect to the radius, the centre and the box is exactly zero -- so the
    ``where`` costs no accuracy and buys the two invariants. Without them the eight-corner sum
    differences numbers of order ``R^3`` to produce one of order ``h^3``, and at ``R/h = 16``
    that reads ``fill = -1.4e-03`` for voxels that are simply outside; with them, measured over
    the same grid, ``fill.min() == 0.0`` and ``fill.max() == 1.0`` exactly.

    Accuracy on genuinely cut voxels, against adaptive quadrature of
    :func:`clipped_disc_area` along the axis (300 random ball/box configurations,
    ``tests/fdtd/test_fd_shapes.py::test_the_clipped_ball_is_what_quadrature_says_it_is``): worst
    9.1e-14 absolute, and the primitive itself 5.6e-16. As a fraction of one **voxel** the
    residual is ``1.5 eps (R/dx)^3`` -- the eight corner primitives are of order ``R^3`` and
    their alternating sum is of order ``dx^3``, so the cancellation is inherent to any
    inclusion-exclusion form and no summation order removes it. Measured 8.7e-15 at
    ``R/dx = 3`` and 8.2e-12 at 32;
    ``tests/fdtd/test_fd_shapes.py::test_a_sphere_fills_a_voxel_exactly_and_this_is_what_that_replaced``
    carries the table.

    **The fold is** :func:`_fold`, not ``xp.abs`` -- :func:`clipped_disc_area`'s "signed branch
    chosen at zero so a centre derivative survives on the axes", which this function first
    missed.
    """
    corners = ((x0 - cx, x1 - cx), (y0 - cy, y1 - cy), (z0 - cz, z1 - cz))
    total = None
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                (sa, a), (sb, b), (sc, c) = (_fold(v, xp) for v in (
                    corners[0][i], corners[1][j], corners[2][k]))
                sign = 1. if (i + j + k) % 2 == 1 else -1.
                term = sign * (sa * sb * sc) * _ball_corner(a, b, c, radius, xp)
                total = term if total is None else total + term
    zero = xp.zeros_like(total)

    def outside(lo, hi):
        return xp.maximum(xp.maximum(lo, -hi), xp.zeros_like(lo))

    far = sum(xp.maximum(lo * lo, hi * hi) for lo, hi in corners)
    near = sum(outside(lo, hi) ** 2 for lo, hi in corners)
    box = (x1 - x0) * (y1 - y0) * (z1 - z0) + zero
    return xp.where(near >= radius * radius, zero, xp.where(far <= radius * radius, box, total))


def clipped_polygon_area(vertices, left, right, bottom, top, xp, convex=False):
    """Exact area of a simple polygon inside the axis-aligned box, differentiably.

    ``vertices`` is ``(..., nv, 2)``, counter-clockwise and simple, broadcasting against the four
    bounds. ``convex`` is a **static** promise about the polygon and buys the smaller buffer;
    it is not read from the data and never branches on a value. See :func:`clip_capacity`.

    Ported from ``src/wavebit/fdtd/_periodic/scene.py::_clipped_rectangle_area`` (:306) and generalised
    from ``nv = 4`` to any ``nv >= 3`` and from convex to simple. The stable packing that keeps
    the survivors' cyclic order without a host-side plan, and the safe inactive denominators that
    keep a derivative finite on an edge parallel to a face, are the original's and are unchanged.
    Combinatorics are piecewise; an edge lying exactly on a face is admitted to both neighbours,
    as there.

    On a non-convex polygon Sutherland-Hodgman leaves **degenerate bridge edges** running along
    the clip box's own faces. They carry no area -- each traverses a face line and comes back --
    so the shoelace sum below is still the exact clipped area, and the final
    ``where(area < 0, 0, area)`` is the source's rounding guard and nothing more. Measured
    against an unbounded-list reference clipper in
    ``tests/fdtd/test_fd_shapes.py::test_a_non_convex_cross_section_is_the_same_number_on_three_backends``.
    """
    nv = vertices.shape[-2]
    cap = clip_capacity(nv, convex)
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

    Ported from ``src/wavebit/fdtd/_periodic/scene.py::_rectangle_edge_moments`` (:823) and generalised
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
    share = xp.ones_like(enter)
    coordinate = xp.maximum(xp.max(xp.abs(vertices), axis=-1), length)
    voxel_scale = xp.maximum(xp.maximum(xp.abs(left), xp.abs(right)),
                            xp.maximum(xp.abs(bottom), xp.abs(top)))
    # Use the existing 32-epsilon clipping margin for coordinate cancellation,
    # separately from the slope test. The .137 translation probe lost a face
    # under an exact offset comparison; no geometric smoothing width is added.
    boundary_tol = 32*xp.finfo(vertices.dtype).eps*xp.maximum(coordinate,
                                                           voxel_scale[..., None])
    for slope, offset in ((-edge_u, start_u - left[..., None]), (edge_u, right[..., None] - start_u),
                          (-edge_v, start_v - bottom[..., None]), (edge_v, top[..., None] - start_v)):
        slope, offset = slope + xp.zeros_like(enter), offset + xp.zeros_like(enter)
        moving = xp.abs(slope) > tol
        t = offset / xp.where(moving, slope, xp.ones_like(slope))
        enter = xp.where(moving & (slope < 0), xp.maximum(enter, t), enter)
        leave = xp.where(moving & (slope > 0), xp.minimum(leave, t), leave)
        # A parallel boundary edge is shared equally by the adjacent voxels.
        # The moving-edge threshold above is unchanged (a 1e-9 slope is resolved).
        outside = outside | (~moving & (offset < -boundary_tol))
        share = xp.where(~moving & (xp.abs(offset) <= boundary_tol), .5*share, share)
    span = xp.where(outside, xp.zeros_like(enter), xp.maximum(leave - enter, xp.zeros_like(enter)))
    arclength = span * length * share
    return (xp.sum(arclength, axis=-1),
            xp.sum(arclength * normal_u * normal_u, axis=-1),
            xp.sum(arclength * normal_v * normal_v, axis=-1))
