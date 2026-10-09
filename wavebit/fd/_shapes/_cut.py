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
from operator import index as _index

from wavebit._core.backend import _concrete, array_namespace


#: A voxel is called **cut** when the interface area inside it clears this fraction of the
#: voxel's own face ``dx^2``. The first writing tested ``area > 0``, an absolute predicate on a
#: number that is a *product of coordinates*, so the same scene written in a different unit
#: landed on opposite sides of it: measured in the C1 review's cross-backend fuzz over 200
#: random bodies, a
#: star prism at coordinate scale 3.29e+05 with ``h = 3.23e+04`` gave ``area = 0`` on NumPy and
#: ``area = 1.80e-09`` on JAX -- a relative 1.7e-18 of the voxel face, 1.04e+09 -- and ``w``
#: swapped from ``(0, 0, 0)`` to a full ``(1, 0, 0)`` between the two backends on one legal
#: input. A ratio to ``dx^2`` says the same thing at every scale (memory: a predicate compared
#: against an absolute floor is a predicate about the coordinate unit, and the same argument
#: rewrote ``_bodies._TURN_SINE`` last round). The margin is far below anything physical: a
#: cross-section of ``1e-12 dx^2`` is a corner tetrahedron of volume about ``1e-18 dx^3``.
#:
#: **That last step is a plane's argument, and the fill is not the plane's**, so it does not
#: bound the fill in a voxel this predicate calls dead. ``area`` is the tangent plane's
#: cross-section at the voxel's *own point*, and a curved body can clip a corner of a voxel its
#: tangent plane at the centre misses entirely: there the exact fill is fractional and ``w`` is
#: exactly ``(0, 0, 0)``. Measured on ``Sphere(radius=0.25, center=(0.013, -0.007, 0.021))``
#: at ``dx = 0.05``, **32** of the six components' voxels are dead with a fill further than
#: 1e-9 from both ends, the worst **2.88e-04** from an end. Over ``dx`` 0.2 down to 0.0125 at
#: four placements each -- centred on a node, a quarter and a half cell off it on every axis,
#: and ``(0.13, -0.37, 0.41)`` cells off -- the worst is **1.35e-03** away from the node, and
#: **1.67e-02** centred on it at ``dx = 0.05``, halving with ``dx`` (8.3e-03 at 0.025, 4.2e-03
#: at 0.0125). That one is a tie and not a corner: a voxel on the ball's axis whose point sits
#: exactly half a cell inside the surface has its tangent plane lying on its own face, so the
#: plane's cross-section is zero while the curved clip leaves the face's corners outside the
#: ball (fill 0.983294). These are geometric fill measurements, not coefficient-error bounds
#: for replacing a nonzero normal moment by ``w = 0``. The sensitivity of ``coefficients.mix``
#: depends on the positive material values and their ratio, so the fill's distance from an
#: endpoint alone does not bound that error. The independent area-mask and fill measurements
#: are pinned by
#: ``tests/test_fd_occupancy.py::test_the_normal_moment_sums_to_one_wherever_the_plane_cuts_the_voxel``.
_LIVE_AREA = 1e-12


def _mean_of_the_moment(area, moment, dx, xp):
    """``w = moment / area`` where a surface reaches the voxel, and exactly zero where none does."""
    live = area > _LIVE_AREA * dx * dx
    safe = xp.where(live, area, xp.ones_like(area))[..., None]
    return xp.where(live[..., None], moment / safe, xp.zeros_like(moment))


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


def _unit(n, xp, where='cut'):
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

    **The test is written as the complement on purpose.** ``worst > 1e-9`` is ``False`` for a
    NaN, so the first writing let a non-finite normal straight through: measured on the exported
    spelling, ``n = (nan, nan, nan)`` with ``d = 0.01`` read ``fill = nan`` with
    ``sum_c w_c = 0`` -- which is exactly what a door exists to stop, and exactly what a broken
    upstream hands one. ``~(worst <= 1e-9)`` refuses it, and the tolerance is dimensionless
    because ``|n|`` is (memory: a predicate compared against an absolute floor is a predicate
    about the coordinate unit).
    """
    worst = _concrete(~(xp.max(xp.abs(xp.sum(n * n, axis=-1) - 1)) <= 1e-9))
    if worst:
        raise ValueError(f'{where}(n=...): n must be the unit outward normal, one per point, and '
                         'finite; the worst |n| here is off one by more than 1e-9 or is not a '
                         'number. The plane is '
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


#: ``1 - (n_a . n_b)^2`` below this and the pair is treated as **parallel**, where the volume and
#: the two face areas have closed forms of their own. The general form below divides by
#: ``sqrt(1 - mu^2)``, so its round-off grows as ``eps / sqrt(1 - mu^2)``; the parallel form's own
#: error at an angle ``theta`` off parallel is the wedge it ignores, about ``theta/2`` of the
#: voxel. The two cross at ``1 - mu^2 ~ 2e-16``, where both are about 7e-9 -- so the switch is set
#: a hair above the last bit of a dot product of unit vectors and nowhere near a real angle.
#: Measured against the ConvexHull oracle over 300 nearly-anti-parallel, nearly-flush joints:
#: worst ``|dV|`` 4.3e-13 and worst ``|dA|/dx^2`` 1.3e-09 against the oracle named below.
_PARALLEL = 1e-14

#: How close two tangent faces must be to count as lying **on each other**: ``d_a + d_b`` below
#: this many voxels, for a pair whose normals are exactly opposite. It decides whether an
#: anti-parallel pair's two faces are buried in each other's material (a device's flush joint) or
#: separated by a gap that is real interface. The twin of ``_bodies._SAME_FACE`` and the same
#: argument: ``(d_a + d_b)/dx`` is a length over a length, and a joint placed flush hits zero to
#: the last bit.
_ON_EACH_OTHER = 1e-12

#: The other two axes of a box face whose normal is along axis ``k``, in increasing order.
_OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _half_line(lo, hi, dead, alpha, beta, xp):
    """``[lo, hi]`` intersected with ``{t : alpha t <= beta}``, branch-free.

    ``alpha = 0`` is the case a ``/`` cannot answer: the constraint is then either vacuous or
    impossible, and which one is a property of ``beta`` alone. The division is done with a safe
    denominator first (the idiom this package states in :func:`_second_difference`), so the dead
    branch's *cotangent* is zero rather than ``nan``.
    """
    safe = xp.where(alpha != 0, alpha, xp.ones_like(alpha))
    reach = beta / safe
    return (xp.where(alpha < 0, xp.maximum(lo, reach), lo),
            xp.where(alpha > 0, xp.minimum(hi, reach), hi),
            dead | ((alpha == 0) & (beta < 0)))


def _span(lo, hi, dead, xp, zero):
    return xp.where(dead, zero, xp.maximum(hi - lo, zero))


def _edge_key(m, k, s_k, r, s_r):
    return (m, s_k, s_r) if k < r else (m, s_r, s_k)


def two_plane(d_a, n_a, d_b, n_b, dx, plane_a, plane_b):
    """The voxel cut by **two** half spaces: the shared fill, and each face's area inside the other.

    ``(shared, area_a, area_b)`` for ``B = [-dx/2, dx/2]^3`` centred on the point and
    ``H_i = {n_i . x + d_i <= 0}``:

        shared  = vol(B n H_a n H_b) / dx^3
        area_a  = area(dH_a n B n H_b)      the part of plane a lying inside H_b
        area_b  = area(dH_b n B n H_a)

    Those three are everything a CSG node needs from a pair of surfaces that share a voxel.
    ``B n (H_a u H_b)`` is ``f_a + f_b - shared``; the union's interface is
    ``(a_a - area_a) + (a_b - area_b)``; the intersection's is ``area_a + area_b``; and a
    difference takes one of each. :func:`_parts` supplies ``f_i`` and ``a_i``.

    **Why this and not the general recursion.** The previous round priced Lasserre's recursion
    for ``k`` half spaces (J. B. Lasserre, *An analytical expression and an algorithm for the
    volume of a convex polyhedron in R^n*, J. Optim. Theory Appl. **39**, 363 (1983)) at 33 s to
    440 s per component per lowering and up to 22 GiB, and rejected it. What is written here is
    the **k = 2** case of the same divergence-theorem identity, unrolled, with the recursion
    replaced by arithmetic:

        3 vol(P) = sum_f s_f A_f          over the at most 8 faces of P, s_f the signed
                                          support of face f from the voxel centre
        2 A_f    = sum_e b_e L_e          the same identity one dimension down, inside face f

    and the innermost quantity ``L_e`` -- the length of an edge of ``P`` -- is a **line clipped by
    a handful of half spaces**, which is one pass of :func:`_half_line` and no geometry at all.
    There are twelve box edges (each clipped by the two planes), twelve traces of a cut plane on
    a box face (each clipped by the face's four sides and the other half space), and one line
    where the two cut planes meet (clipped by the six box sides): 25 segments, about 1100 scalar
    operations per voxel, no branch on a value and nothing allocated per voxel. The cost is in
    :class:`_bodies._Combination`'s table.

    **Validated against an oracle that shares no line with it** -- brute-force vertex enumeration
    over all 56 triples of the 8 planes, feasibility, and ``scipy.spatial.ConvexHull``
    -- it is written out in
    ``tests/test_fd_shapes.py::test_the_exact_two_plane_cut_is_what_an_independent_oracle_says``,
    where these numbers are re-run (2026-09-21):

        family                                        worst |dV|   worst |dA|/dx^2
        600 random pairs, both cutting the box         4.5e-13        7.8e-13
        300 axis-aligned pairs (faces, slabs, corners) 9.8e-13        5.0e-13
        300 nearly anti-parallel, nearly flush joints  4.3e-13        1.3e-09
        300 with one plane grazing a corner            5.3e-17        1.0e-14

    and against 4e5-point Monte Carlo on 60 random configurations, 1.5e-03 at a 1.6e-03 standard
    error. The identity's own consistency checks hold to round-off over a 4000-configuration
    sweep: ``V <= min(f_a, f_b)`` to 5.6e-16, ``V >= max(0, f_a + f_b - 1)`` to 1.2e-15,
    ``area_i <= a_i`` to 1.2e-15. With one plane pushed clear of the voxel it reproduces
    :func:`_parts` to 5.6e-16 in the fill and 1.1e-16 in the area -- the check that says the two
    forms are the same function. The answer is bit-identical across five decades of coordinate
    scale (1e-6 to 1e+9).

    **The parallel branch is a different formula, not a tolerance on this one.** Two parallel
    planes have no intersection line and ``sqrt(1 - mu^2)`` is the denominator that says so;
    below :data:`_PARALLEL` the pair is a slab (``max(0, f_a + f_b - 1)``, exact, and the case
    every device joint is) or a nested pair of half spaces (``min(f_a, f_b)``, exact). Where two
    *coincident* faces meet, which of them is "inside" the other is a measure-zero question with
    two defensible answers; it is broken one way (``<=`` for a, ``<`` for b) so that exactly one
    of them counts, which is what makes a union of two coincident faces report one interface and
    an intersection of them report one too, rather than two or none.
    """
    xp = array_namespace(d_a, n_a)
    h = dx / 2
    c = (-d_a, -d_b)
    n = (n_a, n_b)
    zero, one = xp.zeros_like(d_a), xp.ones_like(d_a)
    off = zero > one
    far = 4 * dx * one                  # any segment of P is inside [-dx sqrt(3)/2, +dx sqrt(3)/2]

    edge = {}                           # the twelve box edges, clipped by both half spaces
    for m in range(3):
        p, q = _OTHER[m]
        for s_p in (1.0, -1.0):
            for s_q in (1.0, -1.0):
                lo, hi, dead = -h * one, h * one, off
                for i in (0, 1):
                    foot = s_p * h * n[i][..., p] + s_q * h * n[i][..., q]
                    lo, hi, dead = _half_line(lo, hi, dead, n[i][..., m], c[i] - foot, xp)
                edge[m, s_p, s_q] = _span(lo, hi, dead, xp, zero)

    trace, offs, norm = {}, {}, {}      # each cut plane's trace on each box face
    for k in range(3):
        u, v = _OTHER[k]
        for s in (1.0, -1.0):
            for i in (0, 1):
                j = 1 - i
                nu, nv = n[i][..., u], n[i][..., v]
                g = c[i] - s * h * n[i][..., k]
                q2 = nu * nu + nv * nv
                live = q2 > 0           # false only where plane i is parallel to this face
                q2s = xp.where(live, q2, one)
                root = xp.sqrt(q2s)
                du, dv = -nv / root, nu / root
                pu, pv = g * nu / q2s, g * nv / q2s
                lo, hi, dead = -far, far, ~live
                lo, hi, dead = _half_line(lo, hi, dead, du, h - pu, xp)
                lo, hi, dead = _half_line(lo, hi, dead, -du, h + pu, xp)
                lo, hi, dead = _half_line(lo, hi, dead, dv, h - pv, xp)
                lo, hi, dead = _half_line(lo, hi, dead, -dv, h + pv, xp)
                ou, ov = n[j][..., u], n[j][..., v]
                lo, hi, dead = _half_line(lo, hi, dead, ou * du + ov * dv,
                                          c[j] - s * h * n[j][..., k] - (ou * pu + ov * pv), xp)
                trace[k, s, i] = _span(lo, hi, dead, xp, zero)
                offs[k, s, i] = g
                norm[k, s, i] = xp.where(live, root, one)

    mu = xp.sum(n_a * n_b, axis=-1)     # the line where the two cut planes meet
    apart = 1 - mu * mu
    crossed = apart > _PARALLEL
    safe = xp.where(crossed, apart, one)
    base = (((c[0] - mu * c[1]) / safe)[..., None] * n_a
            + ((c[1] - mu * c[0]) / safe)[..., None] * n_b)
    along = xp.stack([n_a[..., 1] * n_b[..., 2] - n_a[..., 2] * n_b[..., 1],
                      n_a[..., 2] * n_b[..., 0] - n_a[..., 0] * n_b[..., 2],
                      n_a[..., 0] * n_b[..., 1] - n_a[..., 1] * n_b[..., 0]], axis=-1)
    root = xp.sqrt(safe)
    along = along / root[..., None]
    lo, hi, dead = -far, far, ~crossed
    for k in range(3):
        lo, hi, dead = _half_line(lo, hi, dead, along[..., k], h - base[..., k], xp)
        lo, hi, dead = _half_line(lo, hi, dead, -along[..., k], h + base[..., k], xp)
    ridge = _span(lo, hi, dead, xp, zero)

    total = zero                        # 3 vol(P), face by face
    for k in range(3):
        u, v = _OTHER[k]
        for s in (1.0, -1.0):
            face = zero
            for m in (u, v):
                r = v if m == u else u
                for s_r in (1.0, -1.0):
                    face = face + h * edge[_edge_key(m, k, s, r, s_r)]
            for i in (0, 1):
                face = face + (offs[k, s, i] / norm[k, s, i]) * trace[k, s, i]
            total = total + h * 0.5 * face

    cut = []
    for i in (0, 1):
        j = 1 - i
        face = zero
        for k in range(3):
            nik = n[i][..., k]
            rise = xp.sqrt(xp.maximum(1 - nik * nik, zero))
            live = rise > 0
            rises = xp.where(live, rise, one)
            for s in (1.0, -1.0):
                face = face + xp.where(live, (h - s * c[i] * nik) / rises, zero) * trace[k, s, i]
        face = face + xp.where(crossed, (c[j] - mu * c[i]) / root, zero) * ridge
        cut.append(0.5 * face)
    total = (total + c[0] * cut[0] + c[1] * cut[1]) / 3

    same = mu > 0                       # the parallel branch: a slab, or one inside the other
    meet = (c[0] + c[1]) >= -_ON_EACH_OTHER * dx
    flat_vol = xp.where(same, xp.minimum(plane_a[0], plane_b[0]),
                        xp.maximum(plane_a[0] + plane_b[0] - one, zero)) * dx ** 3
    flat_a = xp.where(same, xp.where(c[0] <= c[1], plane_a[1], zero),
                      xp.where(meet, plane_a[1], zero))
    flat_b = xp.where(same, xp.where(c[1] < c[0], plane_b[1], zero),
                      xp.where(meet, plane_b[1], zero))
    return (xp.where(crossed, total, flat_vol) / dx ** 3,
            xp.where(crossed, cut[0], flat_a),
            xp.where(crossed, cut[1], flat_b))


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

    ``dx`` is doored here and not in :func:`_parts`, for the same reason ``n`` is: the bodies
    reach ``_parts`` through ``occupancy``, which doors ``dx`` once per call, and a door inside
    ``_parts`` would be paid once per body per image instead.
    """
    from ._bodies import check_sizes       # local: _bodies imports _parts from this module
    dx, = check_sizes('cut', dx=dx)
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
    return fill, _mean_of_the_moment(area, moment, dx, xp)


# ----------------------------------------------------------------------------------------
# the same cut one and two dimensions down: a component's edge, and its dual face
# ----------------------------------------------------------------------------------------
#
# The material operator C4 is built towards does not ask a body only for the fraction of a
# *voxel* it fills. Werner, Bauer and Cary, *A more accurate, stable, FDTD algorithm for
# electromagnetics in anisotropic dielectrics*, J. Comput. Phys. **255**, 436 (2013), sections 4
# and 7, constrain the fields on the two objects the Yee grid actually integrates over: the
# **edge** a component's line integral of E runs along, and the **dual face** its flux of D
# crosses. So the pair each needs is the same question one and two dimensions down --
#
#     edge_cut(d, n, dx, axis)   the fraction of that segment the tangent half space holds
#     face_cut(d, n, dx, axis)   the fraction of that square it holds
#
# -- and the reason to write them here, beside :func:`cut`, rather than as a closed form per
# body is the reason :func:`cut` exists: one form for every shape, **including a CSG node**,
# which has no exact clip of its own and whose composed distance and normal are all it can offer.
#
# ``d`` and ``n`` are the shape's signed distance and unit outward normal **at the centre of
# that edge or that face** -- not at the voxel centre. A component's edge and its dual face
# share that centre (the component's own Yee point), which is why one ``(d, n)`` serves both.
#
# The forms are the same inclusion-exclusion sum :func:`_parts` writes, with the corner count
# dropping from eight to four and to two. Over the ``m`` live axes of the cell, with
# ``a_i = |n_i| dx``, ``A = sum a_i`` and ``c = clip(-d + A/2, 0, A)``::
#
#     m = 3   [ <c>^3 - sum <c-a_i>^3 + sum <c-a_i-a_j>^3 - <c-A>^3 ] / (6 a_1 a_2 a_3)
#     m = 2   [ <c>^2 - <c-a_1>^2 - <c-a_2>^2 + <c-A>^2 ]            / (2 a_1 a_2)
#     m = 1   [ <c>   - <c-a_1> ]                                    / a_1
#
# (the clip on ``c`` is what ``m = 3`` and ``m = 2`` need for a cell deep inside the body and
# ``m = 1`` provably does not; :func:`_edge_parts` carries the measurement)
#
# and each is written the way :func:`_parts` writes the first: as a divided difference taken in
# a **stable order**, so that the only division by a step is by the largest one, and so that the
# lower-dimensional limits are regions of the pieces' own definitions rather than branches
# bolted on with a tolerance. :func:`_first_difference_square` is to ``m = 2`` what
# :func:`_second_difference` is to ``m = 3``.
#
# **What does need a branch, and why it is a limit and not a tolerance.** In three dimensions
# the largest step can never be small: ``|n| = 1`` forces ``a_1 >= dx/sqrt(3)``. One and two
# dimensions down that is gone -- the component's edge may be exactly *parallel* to the plane
# (``n_axis = 0``: every voxel of a slab, of an axis-aligned box face, and of two of a sphere's
# three great circles), and the dual face may be parallel to it too (``n`` along ``axis``).
# There the cell is wholly inside or wholly outside and no divided difference can say which;
# :func:`_decided` reads it off the sign of ``d`` instead. The predicate is an exact ``a > 0``
# and not a tolerance, because the branch it selects **is** the limit of the one it leaves: as
# ``a -> 0`` with ``d`` fixed and non-zero, ``c`` saturates at 0 or ``A`` and the general form
# already returns the same 0 or 1. Only the single point ``a = 0, d = 0`` -- a plane containing
# the cell -- has no limit to follow (along ``a -> 0`` it is 1/2, along ``d -> 0`` it is 0 or 1),
# and it is answered by the measure of the closed half space, which is 1; :func:`_decided` says
# why.
#
# **The regularity is not the same one dimension down, and that is geometry and not a defect.**
# The ``m``-dimensional cut of a box is C^(m-1) in ``(d, n)``: the cube's is C1, the dual face's
# is C1, and **the edge's is only C0** -- piecewise linear in ``d``, with a corner where an end
# of the segment crosses the plane. No exact form can do better, the sphere's exact override in
# ``_bodies.py`` has the same two corners, and
# ``tests/test_fd_shapes.py::test_the_face_cut_is_c1_and_the_edge_cut_is_only_c0`` measures
# both. What that costs a design loop is a derivative that exists almost everywhere and jumps
# on a measure-zero set, with a slope ``1/a_1`` that grows without bound as the plane turns
# parallel to the edge.


def _first_difference(t, a, xp):
    """``[<t> - <t-a>]/a``, ``a >= 0``: the first divided difference of ``<t>``.

    Three pieces, the middle one written so that the division is of a number in ``[0, a]`` by
    ``a`` rather than of a difference of two numbers near ``t``:

        t <= 0        0
        0 < t <= a    t / a
        t > a         1

    Exact in its degenerate limit without a tolerance: at ``a = 0`` the middle piece's region is
    empty and what is left is ``<t>``'s own derivative, the step. The value agrees at both knots
    (0 and 1); the slope does **not** (0, then ``1/a``, then 0), and that is the function -- the
    length of a segment inside a half space has a corner where an end of it crosses the plane.
    """
    one = xp.ones_like(a)
    a_safe = xp.where(a > 0, a, one)
    return xp.where(t <= 0, xp.zeros_like(t),
                    xp.where(t <= a, _ramp(t, xp) / a_safe, one))


def _first_difference_square(t, a, xp):
    """``[<t>^2 - <t-a>^2]/a``, ``a >= 0``: the first divided difference of ``<t>^2``.

    Three pieces, each with the step cancelled algebraically:

        t <= 0        0
        0 < t <= a    t^2 / a
        t > a         2t - a

    Value and slope agree at both knots (``a``, and ``2a/a`` against ``2``), so this is C1 in
    ``t`` -- which is what makes :func:`face_cut` C1. It is exact at ``a = 0``, where the middle
    region is empty and ``2<t>`` is left: the derivative of ``<t>^2`` itself, which is the
    one-dimensional cut the face degenerates to when the normal lies along one of its own axes.
    """
    one = xp.ones_like(a)
    a_safe = xp.where(a > 0, a, one)
    u = _ramp(t, xp)
    return xp.where(t <= 0, xp.zeros_like(t),
                    xp.where(t <= a, u * u / a_safe, 2 * t - a))


def _decided(d, xp):
    """1 where the centre is inside: the cut of a cell the plane is parallel to and misses.

    The ``a_1 = 0`` branch of both forms below, and it is the **measure** of the closed half
    space rather than the limit of the general form. ``d <= 0`` and not ``d < 0``: a cell lying
    in the boundary plane of ``n.(x - centre) <= -d`` is contained in it, so its fraction is one.
    The other defensible spelling is the pointwise limit along ``a -> 0+``, which is
    ``1/2`` at ``d = 0`` because that is what the general form returns for every ``a > 0``
    there; it is not taken, because the oracle these two forms are held against
    (``tests/test_fd_shapes.py::test_the_edge_and_face_cuts_are_what_an_independent_clip_says``)
    computes a measure, and a measure has no say in which limit a degenerate slice ought to
    follow. Either way the disagreement is one point of a two-parameter family.
    """
    return xp.where(d <= 0, xp.ones_like(d), xp.zeros_like(d))


def _edge_parts(d, n, dx, axis, xp):
    """:func:`edge_cut` with the doors left off -- the spelling this package uses internally.

    **No clip on ``c`` here, and that is measured rather than assumed.** ``_parts`` and
    :func:`_face_parts` clip because their top piece differences two large numbers for a cell
    deep inside the body; :func:`_first_difference` has no such piece -- it *returns* the
    constant 1 -- so there is nothing to cancel. Over 2000 normals at depths of 1, 10, 1e3, 1e6
    and 1e12 voxels, clipped and unclipped agree to 0.0 exactly, while the face at the same
    depths read 1 +/- 2.2e-16 with its ``c`` clipped and 1 +/- 1.9e-3 without it. A clip on
    ``c`` written here anyway would be arithmetic no input can reach, so it is not written.

    **A decided cell needs no select here either, and that is the same sentence one more
    time.** The face selects its two decided cases (:func:`_face_parts`) because its divided
    difference lands an ulp either side of one; this one's middle piece divides ``t <= a`` by
    ``a`` and its top piece is the literal constant, so 0 and 1 come out of the form itself.
    The same sweep measures its worst at exactly 1.0 and its least at exactly 0.0.
    """
    a = xp.abs(n[..., axis]) * dx
    return xp.where(a > 0, _first_difference(-d + a / 2, a, xp), _decided(d, xp))


def _face_parts(d, n, dx, axis, xp):
    """:func:`face_cut` with the doors left off -- the spelling this package uses internally.

    **A decided square is answered exactly, and that is not the clip on ``c``.** The clip pins
    ``c`` to ``[0, total]``, but the divided difference then taken at ``c = total`` is still
    three roundings of ``(2 a_1 + 2 a_2 - a_2 - a_2) / (2 a_1)`` -- one in exact arithmetic and
    one **plus or minus an ulp** in floating point. Measured over 2000 unit normals at depths of
    1, 10, 1e3, 1e6 and 1e12 voxels, at every one of those depths: 151 of them read
    ``1 + 2.220446e-16`` and 151 read ``1 - 2.220446e-16``, and the worst over a slide of the
    plane across the cell was ``1.0000000000000002``. Neither end of that is a rounding detail
    to the rule that reads it -- the mixing rule divides by ``f`` and by ``1 - f``, so an ulp
    above one is a sign change in a denominator -- and this package already refused the same
    thing once, in ``_bodies.Sphere.exact_edge_fill``, whose world-coordinate spelling was
    rejected for returning ``1 + 8.88178e-16``.

    So the two decided cases are selected rather than computed: ``total/2`` is the square's own
    support in the direction ``n``, so it lies wholly inside the half space exactly when
    ``-d >= total/2`` and wholly outside exactly when ``-d <= -total/2``. That is an exact
    predicate on the same numbers the difference is taken of, not a tolerance, and it is the
    same argument :func:`_decided` makes one dimension further down. Strictly between them
    nothing changes, and the gradient is untouched: where the select bites, the exact
    function's own derivative is zero.

    The edge needs neither, and that is its form and not luck: :func:`_first_difference`'s top
    piece *is* the literal constant and its middle piece divides ``t <= a`` by ``a``, so the
    same sweep measures its worst at exactly 1.0 and its least at exactly 0.0.

    **The three-dimensional :func:`_parts` is deliberately left alone, and it is worse.**
    Measured on the same 2000 normals at the same five depths, a voxel the interface does not
    reach reads between ``0.99999999999999933`` and ``1.0000000000000007`` -- 519 of 2000 above
    one and 1050 of 2000 not exactly 0 or 1 -- three ulps where the face had one. It is not
    fixed here because it is the *routed* form: every ``occupancy`` of every scene goes through
    it, and the same select there moves coefficients the frozen archives pin, which is a
    measurement and a regeneration rather than a minor. Recorded with its number in
    ``docs/fd-subpixel-occupancy.md``.
    """
    u, v = _OTHER[axis]
    a_u, a_v = xp.abs(n[..., u]) * dx, xp.abs(n[..., v]) * dx
    a1, a2 = xp.maximum(a_u, a_v), xp.minimum(a_u, a_v)
    total = a1 + a2
    zero, one = xp.zeros_like(total), xp.ones_like(total)
    offset = -d + total / 2
    # The clip is the function outside [0, total], exactly as in `_parts`, and it is what stops
    # the top piece (2t - a) from differencing two large numbers for a face deep inside a body.
    c = xp.clip(offset, zero, total)
    a1_safe = xp.where(a1 > 0, a1, one)
    fill = (_first_difference_square(c, a2, xp)
            - _first_difference_square(c - a1, a2, xp)) / (2 * a1_safe)
    fill = xp.where(offset >= total, one, xp.where(offset <= zero, zero, fill))
    return xp.where(a1 > 0, fill, _decided(d, xp))


def _axis_index(where, axis):
    """``axis`` as 0, 1 or 2, refusing anything else by name -- including ``True`` and ``1.0``.

    A component index, not a letter: these two are the low-level spelling :func:`cut` is, and
    ``_OTHER`` and the ``(..., 3)`` typed triple are indexed by the same integer.
    ``operator.index`` is what admits a NumPy integer and refuses a float, and ``True`` is
    refused on top of it because ``bool`` **is** an ``int`` and ``True == 1`` is how a bool
    reaches an integer door (memory: a bool is not a length, and this repository has paid for
    that one already).
    """
    try:
        index = -1 if isinstance(axis, bool) else _index(axis)
    except TypeError:
        index = -1
    if index not in (0, 1, 2):
        raise ValueError(f'{where}(axis=...): axis is a component index -- 0 for x, 1 for y, '
                         f'2 for z; got {axis!r}.')
    return index


def edge_cut(d, n, dx, axis):
    """The fraction of a component's own edge the tangent half space holds.

    The segment of length ``dx`` along ``axis``, centred on the point ``(d, n)`` was taken at,
    intersected with ``n . (x - centre) <= -d``, as a fraction of ``dx``. See the section note
    above for the form, for what its one branch is, and for the regularity: this is C0 and not
    C1, because the exact function is.
    """
    from ._bodies import check_sizes       # local: _bodies imports _parts from this module
    xp = array_namespace(d, n)
    dx, = check_sizes('edge_cut', dx=dx)
    _unit(n, xp, 'edge_cut')
    return _edge_parts(d, n, dx, _axis_index('edge_cut', axis), xp)


def face_cut(d, n, dx, axis):
    """The fraction of a component's own dual face the tangent half space holds.

    The ``dx`` by ``dx`` square normal to ``axis``, centred on the point ``(d, n)`` was taken
    at, intersected with ``n . (x - centre) <= -d``, as a fraction of ``dx^2``. See the section
    note above.
    """
    from ._bodies import check_sizes       # local: _bodies imports _parts from this module
    xp = array_namespace(d, n)
    dx, = check_sizes('face_cut', dx=dx)
    _unit(n, xp, 'face_cut')
    return _face_parts(d, n, dx, _axis_index('face_cut', axis), xp)
