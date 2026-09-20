"""The cut: what fraction of a voxel a half space fills, and which way its surface faces.

One routine everything in stage 4 goes through::

    cut(d, n, dx) -> (fill, w)

``d`` is the shape's signed distance at the voxel centre (negative inside), ``n`` the unit
outward normal there, ``dx`` the voxel's side. The shape is replaced by its **tangent plane**
``n . (x - centre) <= -d`` and the answer is the pair the whole stage rests on: the fraction
``fill`` of the cube the half space holds, and ``w_c``, the mean of ``n_c^2`` over the piece of
surface inside the cube, which sums to one over the three components wherever there is any
surface at all.

**Why the tangent plane and not an exact clip.** The plane is one closed form for every shape,
including a CSG node that has no clip at all, and it is C1 in ``(n, d)`` -- so it
differentiates, which an exact clip of a CSG node does not. It is not free: a *convex* body
lies inside its own tangent half space at every surface point, so the plane always holds more
of the voxel than the body does, and summing it over a grid **over**-fills. Measured here
(``tests/test_fd_shapes.py::test_the_tangent_plane_over_fills_a_sphere_at_order_two``), over 24
low-discrepancy sub-cell offsets of the centre and at every one of them:

    R/h        3         4         6         8        12        16        32        64
    dV/V   +2.767e-2 +1.567e-2 +6.946e-3 +3.902e-3 +1.736e-3 +9.767e-4 +2.441e-4 +6.104e-5
    x(h/R)^-2 +0.2491   +0.2507   +0.2501   +0.2497   +0.2500   +0.2500   +0.2499   +0.2500

i.e. **+0.250 (h/R)^2**, fitted order 1.9997, over the true volume at 24 of 24 offsets at every
one of those resolutions. On the diameter -- the axis the "too few cells across" refusal is
drawn on -- that is ``1.000 (h/D)^2``. Against the other three bodies at the *same* cells
across (a rotated box, the same box spelled as a CSG max, a hexagonal prism) the four
constants sit between 0.81 and 1.16, so the body does not decide anything; only the sphere's is
stable enough to quote.

The bias is a bias in the *volume*, so it is a bias in the effective radius: ``dR/R = dV/3V``,
which is ``+5.2e-3`` at 4 cells of radius and ``+8.3e-4`` at 10.

**Which bodies still pay it.** Every body in ``_bodies.py`` with a closed form now overrides the
plane's *fill*: ``Box``, ``Prism``, ``Cylinder`` and -- since the C1 review -- ``Sphere``. The
plane's fill is what ``Union``, ``Intersection`` and ``Difference`` get, and the sphere above is
its calibration rather than a live cost. That the calibration is *still* taken on a sphere is
deliberate: it is the one body whose constant is stable enough to quote, and the closed form is
now there to check it against. The plane's *moment* is what a cylinder, a sphere and every CSG
node get, and its price is in ``_bodies.py``'s table.

**The constant is a value, not a bound, and it cannot be exposed per body.** The C1 review
derived the closed form ``C = D^2 int (k1 + k2) dA / (24 V)`` for the same second-order term and
measured 3.01 for a regular tetrahedron built as a CSG ``max`` -- three times the sphere's --
against 1.008 for a rotated cube and 1.000 for an octahedron. The sphere's ``C = 1`` is the one
value that form gives exactly (``int(k1+k2)dA = 8 pi R`` and ``V = 4 pi R^3/3``), and this file
reproduces it numerically: 0.9989 over six sub-cell offsets at 24 cells across. The tetrahedron
is the review's number and is quoted as such -- a ``Body`` here cannot be given an arbitrary
half space, so the shape is not constructible in this package to check.

What *is* settled here is that the constant cannot be handed to C5 per body. It is a surface
integral of the mean curvature: free for a sphere, a cylinder and a box, and unavailable for the
one class that needs it most -- a CSG node, whose edges and their turning angles do not exist
until the geometry is resolved, which is the whole reason that class has no exact fill either.
A polyhedron's *measured* constant cannot arbitrate it either, because the sub-cell offset
lottery moves it by +/-0.35 (a polyhedron's plane error lives on its edges, a set whose voxel
count the offset moves): the same cube reads 0.85 as a rotated ``Box`` and 1.16 as an
``Intersection`` of three slabs, at the same resolution. So C5's door has to be drawn on a
*measured worst* over the body classes it admits, and it must be told that a CSG intersection
can sit at three times the sphere it was calibrated on.

**The closed form, and why it is not written the way it is usually written.** The volume of a
cube cut by a half space is the inclusion-exclusion sum over the cube's eight corners::

    a_i = |n_i| dx,   c = clip(-d + (sum_i a_i)/2, 0, sum_i a_i),   <t> = max(t, 0)
    V/dx^3 = [ <c>^3 - sum_i <c-a_i>^3 + sum_{i<j} <c-a_i-a_j>^3 - <c-sum a>^3 ] / (6 a_x a_y a_z)

and written like that it is unusable twice over. The ``6 a_x a_y a_z`` denominator **vanishes
for an axis-aligned normal** -- every voxel of a slab, of an axis-aligned box face and of a
sphere's three great circles -- so it returns ``nan`` there and needs 2-D, 1-D and 0-D limit
branches bolted on with a tolerance. And just above that tolerance it is a divided difference
of eight cubes that cancel: the surviving digits go like ``eps * a_max / a_min``.

This module writes the same function as a **third divided difference taken in a stable order**.
With ``a_1 >= a_2 >= a_3`` the sorted ``a_i``, and ``P(t; a, b)`` the second divided difference
of ``<t>^3`` over steps ``a >= b`` (:func:`_second_difference` below, five polynomial pieces,
each with the shared factors cancelled *algebraically* rather than numerically)::

    fill = [ P(c; a_2, a_3) - P(c - a_1; a_2, a_3) ] / (6 a_1)

The only division by a step is by ``a_1``, the *largest*, and ``a_1 >= dx/sqrt(3)`` for any
unit normal, so it can never be small. The 2-D, 1-D and 0-D limits are then not branches bolted
on: ``P`` is exact at ``a_3 = 0`` and at ``a_2 = a_3 = 0``, because those regions of its own
piecewise definition are empty there. Measured against the naive form (both against exact
rational arithmetic, ``tests/test_fd_shapes.py::test_the_stable_form_holds_where_the_naive_one_loses_its_digits``):

    a_min/a_max       1e-1      1e-3      1e-5      1e-7       0
    eight corners  4.2e-15   3.8e-13   3.2e-11   3.5e-09      nan
    this form      2.2e-16   3.3e-16   3.3e-16   2.2e-16   2.2e-16

**The interface area comes out for free**, and it is what lets images and CSG nodes be
combined: ``dV/dc`` is exactly the area of the plane's cross-section inside the cube, so the
same five pieces differentiated once give the weight each contribution carries into ``w``.

**What is a branch and what is not.** Every selection here is :func:`xp.where` over an array
predicate -- never a Python ``if`` on a value -- so a traced radius, centre or angle takes the
same path as a concrete one and ``jax.grad`` and a finite difference compute the same function
(the reason there is no ``smoothing=`` keyword anywhere in stage 4). Denominators that a
selection makes inactive are replaced by one **before** they are used, the idiom
``wavebit/_core/scene_fdtd.py::_clipped_rectangle_area`` states as "safe inactive denominators
prevent NaN derivatives on parallel edges": a dead branch's value is discarded in forward mode
but its *cotangent* is not, so ``0/0`` in a branch nothing selects still poisons a gradient.
"""
from wavebit._core.backend import _concrete, array_namespace


def _ramp(t, xp):
    """``max(t, 0)``, whose square and cube are C1 and C2 at the knot."""
    return xp.where(t > 0, t, xp.zeros_like(t))


def _second_difference(t, a, b, xp):
    """``[<t>^3 - <t-a>^3 - <t-b>^3 + <t-a-b>^3] / (a b)`` without the cancellation, ``a >= b >= 0``.

    The second divided difference of ``<t>^3``. Five pieces, each written with ``a b`` or ``a``
    cancelled algebraically so that no piece ever divides a small numerator by a small step:

        t <= 0        0
        0 < t <= b    t^3 / (a b)
        b < t <= a    (3t^2 - 3tb + b^2) / a
        a < t <= a+b  the same, minus (t-a)^3 / (a b)
        t > a+b       6t - 3a - 3b

    Value and slope agree at all four knots (``b^2/a``, ``0``, ``3(a+b)`` and ``3b/a``, ``6``),
    so this is C1 in ``t`` -- which is the whole reason the fill differentiates. It is also
    exact in both degenerate limits without a tolerance: at ``b = 0`` the second piece's region
    is empty and the fourth's collapses onto the third, leaving the first derivative of the
    *first* divided difference, and at ``a = b = 0`` only the first and last survive, leaving
    ``6 <t>`` -- the second derivative of ``<t>^3`` itself.
    """
    ab, one = a * b, xp.ones_like(a)
    ab_safe = xp.where(ab > 0, ab, one)
    a_safe = xp.where(a > 0, a, one)
    u = _ramp(t, xp)
    middle = (3 * t * t - 3 * t * b + b * b) / a_safe
    return xp.where(
        t <= 0, xp.zeros_like(t),
        xp.where(t <= b, u * u * u / ab_safe,
                 xp.where(t <= a, middle,
                          xp.where(t <= a + b, middle - _ramp(t - a, xp) ** 3 / ab_safe,
                                   6 * t - 3 * a - 3 * b))))


def _second_difference_slope(t, a, b, xp):
    """:func:`_second_difference` differentiated in ``t`` -- the five pieces, each once."""
    ab, one = a * b, xp.ones_like(a)
    ab_safe = xp.where(ab > 0, ab, one)
    a_safe = xp.where(a > 0, a, one)
    u = _ramp(t, xp)
    middle = (6 * t - 3 * b) / a_safe
    return xp.where(
        t <= 0, xp.zeros_like(t),
        xp.where(t <= b, 3 * u * u / ab_safe,
                 xp.where(t <= a, middle,
                          xp.where(t <= a + b, middle - 3 * _ramp(t - a, xp) ** 2 / ab_safe,
                                   6 * xp.ones_like(t)))))


def _unit(n, xp):
    """Refuse a normal that is not a unit vector, where there is a value to read.

    ``cut`` and ``cut_parts`` are exported, so C2 will call them with a normal this package did
    not produce, and every line below assumes ``|n| = 1``: the plane is ``n.(x - centre) <= -d``,
    so a scaled ``n`` silently rescales ``d`` as well. Measured on the exported spelling before
    this door: ``n`` scaled by two reads ``fill = 0.450`` where the truth is 0.400 and
    ``w = [0, 0, 4]``, which does not sum to one; a zero ``n`` divides by zero. Neither is
    caught by anything downstream, because ``w`` is only ever *compared* with one inside this
    package and is a free input outside it.

    The check is a single reduction over the array and one host read, so it is skipped under
    ``jax.jit`` (there is no value) and paid once per call everywhere else. ``_bodies.Body.plane``
    goes to :func:`_parts` instead and never pays it: the bodies construct a unit normal.
    """
    worst = _concrete(xp.max(xp.abs(xp.sum(n * n, axis=-1) - 1)) > 1e-9)
    if worst:
        raise ValueError('cut(n=...): n must be the unit outward normal, one per point; the '
                         'worst |n| here is off one by more than 1e-9. The plane is '
                         'n.(x - centre) <= -d, so scaling n rescales the offset too and the '
                         'fill is silently wrong (measured: 0.450 against a true 0.400 at '
                         '|n| = 2), and w stops summing to one.')


def _parts(d, n, dx):
    """:func:`cut_parts` with the door left off -- the spelling every body in this package uses."""
    xp = array_namespace(d, n)
    a = xp.abs(n) * dx
    ordered = xp.flip(xp.sort(a, axis=-1), axis=-1)
    a1, a2, a3 = ordered[..., 0], ordered[..., 1], ordered[..., 2]
    total = a1 + a2 + a3
    offset = -d + total / 2
    # Outside [0, total] the fill is exactly 0 or 1 and its gradient is exactly zero, so the
    # clip is the function rather than an approximation of it -- and it stops the top piece
    # (6t - 3a - 3b) from differencing two large numbers for a voxel deep inside the shape,
    # where -d can be many cells. Without it a fill reads 1 +/- 2e-13 at d = -10 dx.
    c = xp.clip(offset, xp.zeros_like(total), total)
    fill = (_second_difference(c, a2, a3, xp)
            - _second_difference(c - a1, a2, a3, xp)) / (6 * a1)
    # The area is taken from the UNclipped offset. The clip lands a deep-inside voxel exactly
    # on the knot t = 0, where the density of an axis-aligned normal jumps 0 -> 6, and the
    # difference would read a full face of interface for a voxel the surface is nowhere near.
    # No cancellation to avoid here: the two slopes are equal and opposite, not large and close.
    slope = (_second_difference_slope(offset, a2, a3, xp)
             - _second_difference_slope(offset - a1, a2, a3, xp)) / (6 * a1)
    area = slope * dx ** 3
    return fill, area, area[..., None] * n * n


def cut_parts(d, n, dx):
    """``(fill, area, moment)`` for one tangent plane: the pair of :func:`cut`, unnormalised.

    ``area`` is the area of the plane's cross-section inside the voxel and ``moment`` is
    ``area * n_c^2``. They are separate from ``w`` because a shape is not always one plane: the
    periodic image ring and a CSG node add contributions whose normals differ, and a mean of
    ``n n^T`` over a surface is an **area-weighted** mean. Dividing too early would average the
    averages. :func:`cut` is this with the division done, for the one-plane case.

    ``n`` must be a unit vector, and :func:`_unit` says so where there is a value to read; the
    bodies in ``_bodies.py`` guarantee it by construction, including at the points where the true
    gradient is undefined (a sphere's centre, a box's medial axis), so
    ``max_i |n_i| dx >= dx/sqrt(3)`` and the division inside can never be by a small number.
    """
    _unit(n, array_namespace(d, n))
    return _parts(d, n, dx)


def cut(d, n, dx):
    """``(fill, w)``: the fraction of the voxel the tangent half space fills, and ``<n_c^2>``.

    ``w`` is zero where no surface reaches the voxel -- there is no surface to take a mean
    over -- and sums to one wherever there is. **For one plane** that step is invisible to the
    rule that reads it: the plane's own area is zero exactly where its own fill is 0 or 1, and
    ``1/eps = w (f/e_a + (1-f)/e_b) + (1-w)/(f e_a + (1-f) e_b)`` does not depend on ``w`` at
    either end.

    A **body** is not one plane, and there the pairing is only tight, not exact: a body whose
    fill comes from a closed form and whose moment comes from the plane can have the two
    disagree about whether a voxel is cut at all. How far, measured over a sweep of placements
    rather than fitted to one, is in
    ``tests/test_fd_shapes.py::test_w_vanishes_only_where_the_fill_is_already_decided`` -- that
    test, and not this docstring, is the number C2 has to price.
    """
    xp = array_namespace(d, n)
    fill, area, moment = cut_parts(d, n, dx)
    live = area > 0
    return fill, xp.where(live[..., None], moment / xp.where(live, area, xp.ones_like(area))[..., None],
                          xp.zeros_like(moment))
