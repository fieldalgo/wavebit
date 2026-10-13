"""Gate 1 -- the geometry kernel, before any solver touches it.

``src/wavebit/fd/_shapes/`` answers one question, per body and per voxel: what fraction of you is
in it, and what is the mean of ``n_c^2`` over your surface inside it? Every number stage 4 will
claim stands on that pair, and a fault in it would first show as a scattering cross-section
that converges at the wrong order -- a 400-second run to see, and a week to attribute. So the
kernel is gated on its own, against oracles that are not it.

**The oracles, and why each is independent of the thing it judges**

* *Exact rational arithmetic.* The cube-cut fill is a piecewise cubic in rational data, so
  ``fractions.Fraction`` computes it with no rounding at all, by the inclusion-exclusion sum
  over ``k`` live axes. That is what says the kernel's stable rewriting of the same function is
  the same function, and what puts a number on the digits the usual writing loses.
* *Monte Carlo with a stated interval.* 4e6 uniform points in the cube per plane, so the
  standard error of the estimated fill is ``0.5/sqrt(4e6) = 2.5e-4``; the test asserts 4 sigma.
* *Analytic volumes.* ``4 pi R^3/3``, a prism's shoelace area times its height, a box's product
  of sides. Nothing in the kernel is consulted to produce them.
* *The distance, written a second way.* A sphere's, a rotated box's and a capped cylinder's
  signed distance have two-line closed forms; the kernel reaches them through a polygon walk
  and a disc, and this file writes them straight. A hexagonal prism has no such form, so its
  oracle is the minimum distance to 720k points laid analytically on its own surface (mean
  spacing 9.5e-4, worst disagreement measured 5.1e-4). This is the oracle that caught the one
  real bug in this commit -- the extrusion's transverse centre was subtracted twice, which
  every volume test and every AD-against-finite-difference test passed, because they were
  consistent with each other and with a body that was simply somewhere else.
* *Dense surface sampling.* For ``<n n^T>``: 1200x1200 points per face of a rotated box, and
  20001 points around a cylinder's circumference, binned into voxels.
* *Adaptive quadrature of a form this file does not use.* The ball-box clip is held against
  ``scipy.integrate.quad`` of :func:`clipped_disc_area` along the axis -- the same volume read
  as a stack of exactly-clipped discs rather than as eight corner primitives.
* *An unbounded-list clipper.* The polygon clip is held against a plain Python
  Sutherland-Hodgman with no fixed buffer at all, which is the only way to see a buffer bound.
* *The RCWA path's own routines.* ``src/wavebit/fdtd/_shapes/_exact.py`` is a port, so it is held against
  its source -- measured bit-identical for the disc and the polygon area, 5.6e-17 for the edge
  moments.

**The mutants**, each edited into ``src/wavebit/fd/_shapes/``, run against this file, watched to
fail, and reverted:

    mutant                                                            red
    the eight-corner form without its limit branches (`_cut.py`)       6
    the guarded denominators of `_second_difference` removed           2
    the *first* normal moment in place of the second (`_exact.py`)    12
    the periodic image ring dropped (`_occupancy.py`)                  3
    `fill` clamped to 1 after the image sum                            1
    `exact_fill` ignored, the plane used for every body                6
    the transverse centre subtracted twice (the bug that was here)     4
    `clip_capacity` returning the convex bound for every polygon       2
    every CSG node reporting its first child's normal                  2
    `Difference` composing `other_n` instead of `-other_n`             1
    `asin(a/rho)` in place of `atan2(a, sqrt(rho^2-a^2))`              9
    `sqrt(maximum(v, 0))` in place of the guarded root                 5
    the ball clip's full/empty `where` guards dropped                  1
    the non-positive-length refusal removed                            1
    the counter-clockwise refusal removed                              1
    `cut`'s unit-normal refusal removed                                1
    the convexity tolerance written as an ABSOLUTE floor               1
    `clipped_polygon_area`'s `convex=` default flipped to True         1
    the cap interval closed at both ends instead of half open          1
    Liang-Barsky's parallel tolerance loosened to `1e-6 L`             1
    `isfinite` weakened to `~isnan` (an infinite length passes)        1
    `_unit`'s test left as `worst > 1e-9` (a NaN normal passes)        1
    `readable` returning None for a torch.func value                   1
    a traced parameter on a CSG node accepted                          1
    the back-to-back margin widened to a tenth                         1
    the outline test absolute again (`best > 0`)                       1
    the nearest EDGE normal back to a fixed `+u`                       2
    the slab term dropped (the apart branch alone)                     1
    `occupancy`'s `dx` door removed                                    1
    a complex value cast instead of refused                            1
    `_as` back to testing `dtype` and `device`                         1
    the liveness test absolute again (`area > 0`)                      1
    a ZERO length accepted under jit                                   1
    the orientation factor dropped from `exact_moment`                 1
    a union of flush children composed by the plane                    2
    a union subtracting the MOST its pair can share, not the least     1
    the shared volume not clamped into the two fills' own interval     1
    every child called flat / no child ever called flat                2
    "wholly inside" decided on the half space, not the circumradius    1
    the two-pairs refusal dropped                                      1
    a `Difference` fusing a flush pair across its own cut              1
    the back-to-back margin widened to 0.4 of a voxel                  1
    a shared set of measure zero counted as shared (`> 0` for `>= 0`)  1
    an `Intersection` taking its children's sum, not the cut           1
    `two_plane`'s parallel branch never taken                          1
    two faces lying on each other not buried                           1
    `_ON_EACH_OTHER` widened to 0.4 of a voxel                         1
    `two_plane`'s cut faces returned as the whole plane                1
    the line where the two planes meet dropped from the face areas     1
    the burial taken whether or not the pair was proved flat           1
    a bool a length again / `period` undoored / `points` undoored      3

The last nineteen are the round that rebuilt the CSG rule, and four of them are the four of the
round-5 review's seven that survived the 51-test gate this file then had: the blocker's own fix
(a union subtracting the most rather than the least its pair can share), the back-to-back margin
widened to a tenth, `Intersection`'s rule switched off, and `_no_common_volume`'s LP boundary
flipped from `>= 0` to `> 0`. Each needed a gate written for it and not a claim softened: a voxel
proved interior to one child with a second child overlapping it; two opposed faces a FRACTION of
a voxel apart, which must keep their interface; the Intersection rows of the first-contact test;
and a T-junction of three boxes whose union is one box, where calling a shared plane a shared
volume makes three pairs share one voxel and the node refuse a case it can answer exactly.

The eleven before them are the round that added the rule. Three of them did not go red on the first sweep and the gate
was widened rather than the claim: a margin on the separation test that nothing could make fail
was **deleted** (the case it was written for is answered by the back-to-back branch instead); the
orientation guard's assertion was taken at a point INSIDE a hexagon, where the interface area is
zero whatever the guard does, and now runs at a point on its edge; and the slab term produces a
fill that runs continuously past 1 rather than stepping, so the first-contact test checks the
range and not only the step. The first of those decided the round: the convexity tolerance as shipped *was* absolute
below unit scale, so the mutant spelling it that way was behaviourally identical to the code and
passed all 45 tests -- the gate was holding the classifier at one coordinate scale only.

The second one is worth reading twice: removing the guards leaves ``0/0`` in a branch that
``xp.where`` never selects, so **every forward value is still right** and only two tests go
red -- the gradient of a cylinder's height on JAX and on Torch, where an axis-aligned normal
makes the dead branch's cotangent a NaN. A gate of forward values alone would have passed it.
Three rows are the same family, found the same way, in the ball clip this commit adds: an
``asin`` of a ratio is right to eight digits and wrong in the ninth exactly where a regime ends;
an unguarded ``sqrt`` and an unguarded ``atan2`` are right forwards and NaN backwards, the
``sqrt`` on both autodiff backends and the ``atan2`` on JAX only, because Torch's ``atan2``
returns zero at the origin where JAX returns NaN. And the last three rows say what "a door
refuses rather than returning a silently wrong number" is worth: each is one line, and removing
it takes exactly one test with it, which is the point.

**The cost, measured on this machine** (2026-09-21, CPU, 6 cores, ``OMP_NUM_THREADS=2``), for
one component of a 48^3 support box at ``dx = 0.0125`` -- 110592 voxels (2.53 MiB of points),
which is a sphere of 20 cells of radius with room around it. The memory column is the NumPy
peak from ``tracemalloc``, and it belongs beside the time column because it is the one that
decides whether C2 may call ``occupancy`` on a whole component at once:

    body                          numpy     jax (jit)     torch     numpy peak
    Sphere (exact ball)          785.8 ms    40.2 ms    266.2 ms      33.0 MiB
    Sphere (plane, for scale)     53.9 ms       --          --        23.3 MiB
    Cylinder (exact disc)         83.5 ms     5.8 ms     44.3 ms      24.2 MiB
    Box                         1260.9 ms   131.0 ms    978.4 ms     191.6 MiB
    Difference (CSG, plane)       61.4 ms     5.6 ms     25.6 ms      23.3 MiB
    Prism, hexagon (convex cap) 1564.6 ms   175.5 ms   1158.9 ms     236.7 MiB
    Prism, 6-point star         8875.2 ms  1706.6 ms   9131.4 ms    1365.3 MiB

Two prices are visible and both were paid deliberately. The exact ball costs **15x** the
tangent plane, because the eight corner primitives hold 48 wedge integrals and about 144
``atan2`` calls between them, against the disc clip's four; it removes a ``+0.250 (h/R)^2``
volume bias from the one body stage 4 is graded on, and it is a setup cost paid once per
component and not per step. The star prism costs **5.7x** the hexagon in time and **5.8x** in
memory at the same vertex-count class, because a simple polygon needs a clip buffer of 60 slots
where a convex one needs 16 (``_exact.clip_capacity``); the alternative measured on the same
input is NumPy raising, JAX returning NaN and Torch returning a number wrong by 20.8%.

**The memory is the number C2 has to plan against, not the time.** The clip's working set is
``(points, 2 cap, 2)`` float64 plus an ``int64`` argsort over the same, i.e. about
``80 cap`` bytes per voxel per clip step, which at ``cap = 60`` is 4.8 kB of live temporaries
for every voxel. 48^3 is 1/72 of a 200^3 grid, so a star prism called on a whole component at
once would need of order **96 GiB** and even a hexagon of order 17 GiB. C2 must chunk
``points`` -- this repository already carries the memory that a laptop *panics* rather than
slows when it runs out. The bound ``n + n//2`` per clip is loose (the realised counts are 20 of
60 and 28 of 81), and a tighter proof buys this back in the same proportion.

The periodic ring multiplies all of it by the image count, as it must: on NumPy the sphere
reads 33.9 ms with no ring, 289.2 ms over two periodic axes (9 images) and 819.5 ms over three
(27), on the plane path those numbers were measured on. That is the price of the static
``rings=`` prune C2 will make from a declared ``range=``.

**What a CSG node costs now that it answers from its children**, with the shipped node's own
column present -- the round that added the rule priced it with a two-child measurement and a
column belonging to the path it replaced. A substrate plus ``k - 1`` flush pillars, 66^3 = 287496
voxels at ``dx`` = 0.0152, float64, ``OMP_NUM_THREADS=2``, best of three:

    k   pairs  backend    NODE    children alone  composed plane  pair test  two_plane
    2     1    numpy     4.484 s     2.998 s         0.229 s       0.065 s    0.627 s
    2     1    torch     3.052 s     2.568 s         0.187 s       0.015 s    0.339 s
    2     1    jax jit   0.516 s
    4     6    numpy     5.381 s     3.580 s         0.295 s       0.394 s    0.629 s
    4     6    torch     3.807 s     2.913 s         0.215 s       0.082 s    0.340 s
   10    45    numpy    12.324 s     5.599 s         0.484 s       3.330 s    0.683 s
   10    45    torch     6.049 s     3.366 s         0.357 s       0.652 s    0.341 s
   10    45    jax jit   2.302 s

On the 3x3 pillar array -- ten children, forty-five pairs, an ordinary metasurface cell -- the
node is **12.3 s** on NumPy: 5.6 s of it its children's own exact fills, 3.3 s the O(k^2) pair
test, 0.68 s the one exact two-plane cut, which is **flat in k** because it runs once per voxel
and not once per pair. The "composed plane" column reproduces the previous round's 0.483 s for
the path this rule replaced, so the node is **25x** the figure that round offered as its price.
Over six sub-cell grid origins the node reads between 11.1 s and 12.3 s, so it is a property of
the scene and not of an alignment. The alternative -- the exact multi-plane cut over all ``k``
planes, priced in ``_bodies._Combination`` -- was 440.4 s and 22.2 GiB on this same scene. The
pair test is where the next factor is: 27% of the node at ``k`` = 10, and growing as ``k^2``.
"""
import numpy as np
import pytest

from wavebit.fdtd._shapes import Approximations, Box, Cylinder, Difference, Intersection, Prism, Sphere, Union, clip_capacity, clipped_ball_volume, clipped_disc_area, clipped_polygon_area, cut, cut_parts, edge_cut, face_cut, image_shifts, occupancy, polygon_edge_moments
from wavebit.fdtd._shapes._bodies import _loud_if_not_a_size, _loud_if_not_counter_clockwise, polygon_shape
from wavebit.fdtd._shapes._cut import _first_difference, _first_difference_square, _parts, _second_difference, two_plane

from tests.fixtures.fdtd.fd_helpers import backends

#: The stage-2/3 autodiff budget, unchanged: ``|g_ad - g_fd| <= RTOL |g_fd| + FTOL |f|``.
RTOL, FTOL = 1e-7, 1e-10

#: The finite-difference step, set by the sweep in
#: :func:`test_a_finite_difference_says_what_the_backend_says`'s docstring. It is **not** the
#: stage-3 ``1e-5``: the fill is C1 in every leaf and only piecewise C2, so a central
#: difference carries an ``h^2 f'''/6`` truncation with a third derivative that jumps at every
#: knot the stencil straddles, and 1e-5 leaves 3.1e-7 relative on a vertex -- three times the
#: budget, all of it the difference's.
H_STEP = 1e-6

#: A hexagon, in the transverse plane, counter-clockwise, turned off every symmetry.
HEX = tuple((0.19 * np.cos(2 * np.pi * k / 6 + 0.3), 0.19 * np.sin(2 * np.pi * k / 6 + 0.3))
            for k in range(6))
HEX_AREA = 0.5 * abs(sum(HEX[k][0] * HEX[(k + 1) % 6][1] - HEX[(k + 1) % 6][0] * HEX[k][1]
                         for k in range(6)))


def star(points, outer, inner):
    """A counter-clockwise star: simple, non-convex, ``2 * points`` vertices."""
    return tuple(((outer if k % 2 == 0 else inner) * np.cos(np.pi * k / points),
                  (outer if k % 2 == 0 else inner) * np.sin(np.pi * k / points))
                 for k in range(2 * points))


#: Simple, counter-clockwise, and every one of them non-convex. An L, a comb and a cross do not
#: overflow the convex buffer; the stars do, which is why both kinds are here.
L_HEX = ((-0.30, -0.20), (0.30, -0.20), (0.30, 0.00), (0.00, 0.00), (0.00, 0.22), (-0.30, 0.22))
COMB = ((-0.33, -0.18), (0.33, -0.18), (0.33, 0.20), (0.21, 0.20), (0.21, -0.02), (0.11, -0.02),
        (0.11, 0.20), (-0.11, 0.20), (-0.11, -0.02), (-0.21, -0.02), (-0.21, 0.20), (-0.33, 0.20))
CROSS = ((-0.09, -0.30), (0.09, -0.30), (0.09, -0.09), (0.30, -0.09), (0.30, 0.09), (0.09, 0.09),
         (0.09, 0.30), (-0.09, 0.30), (-0.09, 0.09), (-0.30, 0.09), (-0.30, -0.09), (-0.09, -0.09))
NON_CONVEX = {'L': L_HEX, 'star5': star(5, 0.42, 0.10), 'star6': star(6, 0.42, 0.10),
              'star8': star(8, 0.42, 0.10), 'comb': COMB, 'cross': CROSS}


def reference_clip(vertices, left, right, bottom, top):
    """Sutherland-Hodgman in a plain Python list with **no buffer at all**.

    The kernel's clip is the same recurrence in a fixed-width array, so this is the only oracle
    that can say whether the fixed width is wide enough: it returns the clipped area and the
    largest slot count the kernel would have had to hold.
    """
    poly = [tuple(v) for v in vertices]
    worst = len(poly)
    for axis, bound, keep_low in ((0, left, True), (0, right, False),
                                  (1, bottom, True), (1, top, False)):
        out = []
        for i, b in enumerate(poly):
            a = poly[i - 1]
            ina = (a[axis] >= bound) if keep_low else (a[axis] <= bound)
            inb = (b[axis] >= bound) if keep_low else (b[axis] <= bound)
            if ina != inb:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if inb:
                out.append(b)
        poly = out
        worst = max(worst, len(poly))
        if not poly:
            return 0.0, worst
    area = .5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                    - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    return abs(area), worst


def grid(n, lo, h):
    """``(n, n, n, 3)`` voxel centres, x-first triples."""
    c = lo + (np.arange(n) + 0.5) * h
    return np.stack(np.meshgrid(c, c, c, indexing='ij'), axis=-1)


def halton(n, base):
    out = []
    for i in range(1, n + 1):
        f, r, k = 1.0, 0.0, i
        while k:
            f /= base
            r += f * (k % base)
            k //= base
        out.append(r)
    return np.array(out)


#: 24 low-discrepancy sub-cell offsets: the placement lottery, sampled rather than chosen.
OFFSETS = np.stack([halton(24, 2), halton(24, 3), halton(24, 5)], axis=-1) - 0.5


def plane_volume(body, h, half):
    """The summed TANGENT-PLANE fill over a cube of side ``2*half``, in z slabs."""
    n = int(np.ceil(2 * half / h))
    c = (np.arange(n) + 0.5) * h - n * h / 2
    total = 0.0
    x, y = np.meshgrid(c, c, indexing='ij')
    for k0 in range(0, n, 24):
        z = c[k0:k0 + 24]
        pts = np.stack(np.broadcast_arrays(x[..., None], y[..., None], z[None, None, :]), axis=-1)
        total += body.plane(pts, h)[0].sum()
    return total * h ** 3


def order(ns, errs):
    return -np.polyfit(np.log(np.asarray(ns, float)), np.log(np.abs(errs)), 1)[0]


# ----------------------------------------------------------------------------------------
# the cut itself
# ----------------------------------------------------------------------------------------

def exact_rational_fill(a, c):
    """The cube-cut fill in exact arithmetic, over however many steps are nonzero."""
    import itertools
    import math
    from fractions import Fraction as F
    live = [v for v in a if v != 0]
    k = len(live)
    total = sum(live, F(0))
    if c <= 0:
        return F(0)
    if c >= total or k == 0:
        return F(1)
    num = F(0)
    for m in range(k + 1):
        for combo in itertools.combinations(live, m):
            num += (-1) ** m * (lambda t: t ** k if t > 0 else F(0))(c - sum(combo, F(0)))
    den = math.factorial(k)
    for v in live:
        den *= v
    return num / den


def eight_corner_fill(a, c):
    """The formula as it is usually written: eight corners over ``6 a_x a_y a_z``.

    This is the **mutant** of ``_cut.cut_parts``, kept here rather than in the package: it is
    the same function wherever it is defined at all, and it is not defined for an axis-aligned
    normal. Where it is defined it is the cross-check, and where it is not it is the reason the
    kernel is written the way it is.
    """
    a = np.asarray(a, dtype=float)
    ramp = lambda t: max(t, 0.) ** 3
    num = ramp(c)
    for i in range(3):
        num -= ramp(c - a[i])
    for i in range(3):
        for j in range(i + 1, 3):
            num += ramp(c - a[i] - a[j])
    num -= ramp(c - a.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        return num / (6 * a[0] * a[1] * a[2])


def stable_fill(a, c):
    a1, a2, a3 = [np.asarray([float(v)]) for v in a]
    t = np.asarray([float(c)])
    return float(((_second_difference(t, a2, a3, np)
                   - _second_difference(t - a1, a2, a3, np)) / (6 * a1))[0])


def test_the_stable_form_holds_where_the_naive_one_loses_its_digits():
    """Both against exact rational arithmetic, as the smallest step shrinks to zero.

    Measured (worst over 200 random thresholds per row, 2026-09-20):

        a_min/a_max      1e-1     1e-3     1e-5     1e-7        0
        eight corners  4.2e-15  3.8e-13  3.2e-11  3.5e-09      nan
        this kernel    2.2e-16  3.3e-16  3.3e-16  2.2e-16  2.2e-16

    The naive form's error goes like ``eps * a_max/a_min`` because it is a divided difference of
    eight cubes that cancel; the kernel's does not, because the two smallest steps are
    eliminated algebraically and only the largest is ever divided by.
    """
    from fractions import Fraction as F
    rng = np.random.default_rng(7)
    worst = {}
    for ratio in (1e-1, 1e-3, 1e-5, 1e-7, 0.0):
        naive, stable = [], 0.0
        for _ in range(60):
            a = (F(1), F(int(rng.integers(3, 10)), 10), F(ratio).limit_denominator(10 ** 12))
            c = sum(a, F(0)) * F(int(rng.integers(1, 999)), 1000)
            reference = float(exact_rational_fill(a, c))
            naive.append(abs(eight_corner_fill([float(v) for v in a], float(c)) - reference))
            stable = max(stable, abs(stable_fill(a, c) - reference))
        worst[ratio] = (max(naive), stable)
    assert worst[1e-1][1] < 1e-15, worst
    assert worst[0.0][1] < 1e-15, worst
    # the mutant is not merely worse at an axis-aligned normal: it has no value there at all
    assert not np.isfinite(worst[0.0][0]), worst
    assert worst[1e-7][0] > 100 * worst[1e-7][1], worst
    assert worst[1e-1][0] < worst[1e-3][0] < worst[1e-5][0] < worst[1e-7][0], worst


def test_the_closed_form_answers_an_axis_aligned_normal():
    """The 2-D, 1-D and 0-D limits, which are every voxel of a slab and of a box face.

    **(M)** Replacing ``cut_parts`` with :func:`eight_corner_fill` -- "drop the limit branches"
    -- makes every assertion here read ``nan``.
    """
    dx = 1.0
    for n in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)):
        d = np.linspace(-0.9, 0.9, 37)
        axis = np.broadcast_to(np.asarray(n), (37, 3))
        fill, area, _ = cut_parts(d, axis, dx)
        w = cut(d, axis, dx)[1]
        assert np.all(np.isfinite(fill)) and np.all(np.isfinite(area))
        assert np.abs(fill - np.clip(0.5 - d, 0, 1)).max() < 1e-15, n
        # The cross-section of a plane normal to an axis is the whole face, and nothing
        # outside: the 1-D limit, where the eight-corner form divides by zero twice over. The
        # interval is half open because <t> = max(t, 0) closes on the left -- a plane exactly
        # on a voxel face counts as interface for the face it arrives at and not for the one
        # it leaves. A measure-zero configuration, and the fill is 1 on that side either way.
        assert np.abs(area - 1.0 * ((d >= -0.5) & (d < 0.5))).max() < 1e-15, n
        live = area > 0
        assert live.sum() == 20 and np.abs(w[live] - np.asarray(n)).max() < 1e-15
        assert np.abs(w[~live]).max() == 0.0
    # the 2-D limit: one component zero, and one a millionth of the others
    for third in (0.0, 1e-6, 1e-12):
        n = np.asarray([[1.0, 1.0, third]])
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        fill, w = cut(np.asarray([0.0]), n, dx)
        assert abs(fill[0] - 0.5) < 1e-15, third
        assert abs(w.sum() - 1.0) < 1e-14, third


def test_a_cut_is_what_a_monte_carlo_says_it_is():
    """The fill and the interface area against 4e6 uniform samples per plane, at 4 sigma."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(16, 3))
    n = v / np.linalg.norm(v, axis=-1, keepdims=True)
    d = rng.uniform(-0.45, 0.45, size=16)
    fill, area, _ = cut_parts(d, n, 1.0)
    u = rng.uniform(-0.5, 0.5, size=(4_000_000, 3))
    sigma = 0.5 / np.sqrt(4_000_000)
    for k in range(16):
        assert abs((u @ n[k] <= -d[k]).mean() - fill[k]) < 4 * sigma, k
    # the area is dV/dc, so a central difference of the fill in the offset must reproduce it
    # (worst measured 1.4e-9 at this step -- the difference's own truncation, not the area's)
    step = 1e-5
    lo = cut_parts(d + step, n, 1.0)[0]
    hi = cut_parts(d - step, n, 1.0)[0]
    assert np.abs((hi - lo) / (2 * step) - area).max() < 1e-8


def test_the_free_invariants_hold_on_every_body():
    """``0 <= fill <= 1``, ``|sum_c w_c - 1| < 1e-14`` where there is a surface, ``w = 0`` where not.

    **The upper bound is 1 + 3.4e-13 and not 1 + 1e-14**, and the number is the exact fills'
    own round-off rather than a slack. A body centred on the grid's own symmetry is the worst
    placement, because every one of the eight corner primitives then cancels against its mirror.
    Measured, the worst ``max(fill) - 1`` over twenty radii from 0.10 to 0.35, per grid:

        grid          sphere     cylinder      box
        16^3         3.02e-14     6.22e-15   1.78e-15
        24^3         2.22e-16     1.33e-14   1.78e-15
        32^3         3.34e-13     4.60e-14   4.66e-15

    It grows with the grid, for the reason
    :func:`test_w_vanishes_only_where_the_fill_is_already_decided` records for the same clip. The
    shipped bound was 1e-14 and was met only because the two bodies listed here missed the worst
    placement: ``Sphere(0.3)`` on this grid reads 1.0000000000000144, and it now has a row.
    """
    h = 0.05
    pts = grid(16, -0.4, h)
    joint = float(-0.4 + 8.5 * h)
    bodies = {
        'sphere': Sphere(0.26, (0.013, -0.021, 0.007)),
        'sphere on the lattice': Sphere(0.3),
        'cylinder on the lattice': Cylinder(0.25, 0.4, 'z'),
        'union, flush': Union(Box((0.5, 0.5, 0.2), (0., 0., joint - 0.1)),
                              Cylinder(0.12, 0.2, 'z', (0., 0., joint + 0.1))),
        'cylinder': Cylinder(0.21, 0.33, 'y', (0.01, -0.02, 0.0), 0.4),
        'prism': Prism(HEX, 0.33, 'x', (0.01, -0.02, 0.0), 0.9),
        'box': Box((0.34, 0.22, 0.29), (0.013, -0.021, 0.007), 0.7),
        'union': Union(Sphere(0.2, (0.1, 0., 0.)), Box((0.2, 0.2, 0.2), (-0.1, 0., 0.), 0.3)),
        'difference': Difference(Sphere(0.3, (0., 0., 0.)), Cylinder(0.12, 1.0, 'z', (0., 0., 0.))),
        'intersection': Intersection(Sphere(0.3, (0.05, 0., 0.)), Sphere(0.3, (-0.05, 0., 0.))),
    }
    for name, body in bodies.items():
        fill, w = occupancy(body, pts, h)
        assert fill.min() >= 0.0, name
        assert fill.max() <= 1.0 + 3.4e-13, name
        total = w.sum(-1)
        live = total > 0
        assert np.abs(total[live] - 1).max() < 1e-14, name
        assert np.abs(w[~live]).max() == 0.0, name
        assert live.sum() > 50, name


#: The placements this file measures a *worst* over: the first twelve of the Halton cover, plus
#: the four that read the worst gap in the 24-placement sweep below, named so that the gate runs
#: the case the number is quoted from. A bound asserted over placements that do not include the
#: worst one is a bound fitted to a sample.
W_GAP_PLACEMENTS = np.concatenate([OFFSETS[:12], OFFSETS[[16, 22, 15, 1]]], axis=0)


def test_w_vanishes_only_where_the_fill_is_already_decided():
    """The step in ``w`` at a voxel the mixing rule does not read ``w`` in -- and how far it is not.

    ``w`` steps from ``n_c^2`` to zero when the last of the surface leaves the voxel. That is
    allowed only because the fill there is 0 or 1, where
    ``1/eps = w(f/e_a + (1-f)/e_b) + (1-w)/(f e_a + (1-f) e_b)`` is ``1/e_b`` or ``1/e_a``
    whatever ``w`` is. If this pairing ever breaks, the step becomes a step in a coefficient.

    It is **exact** where the fill and the moment come from the same route, and only *tight*
    where they do not: a body whose fill is a closed form and whose moment is the plane's can
    have the two disagree about whether a voxel is cut, because the plane misses a voxel once
    ``|d| > (sum_i |n_i| dx)/2`` while the true curved surface may still poke into a corner of
    it.

    **This bound has now been fitted twice and re-derived twice, and this is the sweep it comes
    from.** The first writing asserted 1e-6 from one cylinder; the C1 review moved that
    cylinder and read 1.09e-2. The second writing asserted 5e-3 for the sphere from six draws at
    two resolutions, and the C1 review read 8.41e-3 from four hundred. Measured here over the
    24-placement Halton cover at five resolutions, worst gap from 0 or 1 over the ``w = 0``
    voxels (2026-09-21):

        body         R/h = 4        6         8        12        16
        box          1.44e-15  2.67e-15  2.44e-15  6.22e-15  5.66e-15
        prism        1.11e-15  2.00e-15  1.44e-15  4.66e-15  3.33e-15
        CSG          6.66e-16  8.88e-16  8.88e-16  8.88e-16  8.88e-16
        sphere       1.16e-02  1.94e-03  2.54e-03  6.44e-04  5.55e-04
        cylinder     3.26e-02  2.79e-02  2.93e-02  3.08e-02  2.53e-02

    **The CSG row is no longer the control it was.** A CSG node used to answer entirely from
    the tangent plane, one route, and read a flat 8.9e-16; it now answers with its *children's
    own* routes wherever they hold no volume of the voxel in common (``_bodies._Combination``),
    so a node built from spheres inherits the sphere's gap exactly -- measured 1.16e-02 at
    ``R/h = 4``, the sphere's own number. The control is kept by a second CSG row built from
    **boxes**, whose fill and moment are both exact: it reads 1.1e-15 and 2.4e-15 at ``R/h`` = 4
    and 8, i.e. the clip's own accumulated round-off and nothing else, and its bound scales with
    ``R/h`` exactly as the single box's does. What the old flat CSG row said -- that the growth
    in the box and prism rows is the clip's and not the pairing's -- this row still says.

    Three things in that table are not what the previous writing said. The sphere's worst at
    four cells of radius is **1.16e-02** and not 5e-03, so it is not "a coefficient in the third
    digit" -- it is the second, and the exact interface area that would remove it is worth more
    to C2 than the last writing priced it at. The sphere does fall but **not monotonically**
    (1.94e-03 at 6 against 2.54e-03 at 8), so "does shrink" is a trend and not a property. And
    the one-route rows are **not** a resolution-free 1e-14: they grow with the grid, because
    they are accumulated round-off in the clip and the overlap, measured 1.44e-15 / 2.11e-15 /
    5.66e-15 / 1.30e-14 / 2.04e-14 for the box at R/h = 4, 8, 16, 24, 32. The CSG row, which is
    one route in the *plane* and takes no clip, stays flat at 8.9e-16 across the same sweep --
    which is what says the growth is the clip's and not the pairing's. So the bound asserted
    here for those rows scales with ``R/h`` and the assertion at 32 cells would fail a constant.

    The remedy for the sphere and the cylinder is the exact interface **area**, which the
    divergence theorem makes free for both (``3V = R S + sum_f d_f A_f`` for a ball,
    ``2A = R L + sum_e d_e l_e`` for a disc), and which would make the area vanish exactly where
    the surface is absent without touching the direction the plane supplies. It is still not
    taken here -- it is six more clipped-disc evaluations per voxel on a body whose exact fill
    already costs 15x the plane -- but at 1.16e-2 it is C2's decision and not an afterthought.
    """
    bounds = {'box': lambda npr: 1e-15 * npr, 'prism': lambda npr: 1e-15 * npr,
              'csg box': lambda npr: 2e-15 * npr, 'csg': lambda npr: 2e-2,
              'sphere': lambda npr: 2e-2,
              'cylinder': lambda npr: 5e-2}
    bodies = {
        'box': lambda R, c: Box((1.7 * R, 1.3 * R, 1.9 * R), c, 0.37),
        'prism': lambda R, c: Prism(HEX, 1.6 * R, 'z', c, 0.4),
        'csg box': lambda R, c: Difference(Box((1.7 * R, 1.3 * R, 1.9 * R), c, 0.37),
                                           Box((R, R, 4 * R), tuple(np.add(c, 0.6 * R)), 0.2)),
        'csg': lambda R, c: Difference(Sphere(R, c), Sphere(0.5 * R, tuple(np.add(c, 0.6 * R)))),
        'sphere': lambda R, c: Sphere(R, c),
        'cylinder': lambda R, c: Cylinder(R, 1.6 * R, 'z', c),
    }
    R = 0.21
    worst_of = {}
    for name, make in bodies.items():
        for npr in ((4, 8, 16) if name in ('sphere', 'cylinder') else (4, 8)):
            h = R / npr
            n = int(np.ceil(2.6 * R / h))
            pts = grid(n, -n * h / 2, h)
            worst, dead_total, live_total = 0.0, 0, 0
            for offset in W_GAP_PLACEMENTS:
                fill, w = occupancy(make(R, tuple(offset * 1.2 * h)), pts, h)
                dead = w.sum(-1) == 0
                worst = max(worst, float(np.abs(np.minimum(fill[dead], 1 - fill[dead])).max()))
                dead_total += int(dead.sum())
                live_total += int((~dead).sum())
            worst_of[name, npr] = worst
            assert dead_total > 100 and live_total > 100, name
            assert worst < bounds[name](npr), (name, npr, worst)
            if name in ('sphere', 'cylinder'):
                assert worst > 1e-12, (name, npr, worst)   # the row is real, not a tolerance
    # the two numbers the last writing got wrong, pinned so they cannot be understated again
    assert worst_of['sphere', 4] > 5e-3, worst_of          # not "the third digit"
    assert worst_of['cylinder', 16] > 1e-2, worst_of       # and the cylinder does not fall


# ----------------------------------------------------------------------------------------
# what the plane costs, and where it is not used
# ----------------------------------------------------------------------------------------

def test_the_tangent_plane_over_fills_a_sphere_at_order_two():
    """+0.250 (h/R)^2, one-signed, fitted 2.00 -- the constant the docstring carries.

    Measured here over 24 sub-cell offsets (2026-09-20):

        R/h        3         4         6         8        16        32        64
        dV/V  +2.767e-2 +1.567e-2 +6.946e-3 +3.902e-3 +9.767e-4 +2.441e-4 +6.104e-5
        x(h/R)^2   0.2491    0.2507    0.2501    0.2497    0.2500    0.2499    0.2500

    over the true volume at 24 of 24 offsets at every one of them, fitted order 1.9997. A
    convex body lies inside its own tangent half space, so the sign cannot be otherwise; the
    stage-4 design's word "under-fills" is wrong and this is the correction. The test runs the
    first four rows; 12, 16, 32 and 64 are in this docstring and not in the suite because they
    cost 35 s for a number the first four already fix to three digits.
    """
    R, want = 0.5, 0.250
    errors, ns = [], (3, 4, 6, 8)
    for npr in ns:
        h = R / npr
        vals = np.array([plane_volume(Sphere(R, tuple(o * h)), h, R + 3 * h) for o in OFFSETS])
        vals = vals / (4 / 3 * np.pi * R ** 3) - 1
        assert (vals > 0).all(), (npr, vals)             # one-signed, at 24 of 24
        errors.append(vals.mean())
        assert abs(vals.mean() * npr ** 2 - want) < 0.005, (npr, vals.mean() * npr ** 2)
    assert order(ns, errors) > 1.9, errors
    assert np.all(np.diff(errors) < 0), errors           # monotone


def test_the_plane_serves_all_four_bodies_about_equally():
    """Neither the study's reading nor its review's inversion of it survives one length axis.

    Study S3(a) concluded "the shapes that have exact forms available are the ones the plane
    serves worst"; its review inverted that to "at equal resolution the sphere is the worst of
    the four". Re-measured here with both sweeps indexed by cells across the **smallest full
    extent**, over 6 sub-cell offsets (2026-09-20):

        body        n=12              n=24              n=48
        sphere    +6.943e-3  0.9997  +1.738e-3  1.0010  +4.340e-4  0.9998
        box rot   +5.955e-3  0.8574  +1.630e-3  0.9387  +3.821e-4  0.8805
        box CSG   +5.630e-3  0.8107  +1.408e-3  0.8107  +5.028e-4  1.1585
        hex prism +6.345e-3  0.9137  +1.689e-3  0.9726  +4.005e-4  0.9227

    All four sit between 0.81 and 1.16 ``(h/D)^2``, so the body does not decide anything -- and
    only the sphere's constant is stable enough to quote (three readings within 0.0013; the
    other three bounce by up to 0.35 with the offset lottery, because a polyhedron's plane error
    lives on its edges, a set whose voxel count and alignment the offset moves; at the four
    offsets this test can afford they read 0.70 to 1.35). What decides against writing
    ``Sphere.exact_fill`` in stage 4 is therefore neither ranking: it is that the bias is second
    order and one-signed, so the convergence study that is the gate fits it away, and the
    alternative is a quadrature per voxel where the others are a closed form.
    """
    D, n = 0.5, 24
    h = D / n
    bodies = {
        'sphere': (lambda c: Sphere(D / 2, c), 4 / 3 * np.pi * (D / 2) ** 3),
        'box rot': (lambda c: Box((D, D, D), c, 0.4), D ** 3),
        'box CSG': (lambda c: Intersection(Box((D, 10., 10.), c, 0., 'z'),
                                           Box((10., D, 10.), c, 0., 'z'),
                                           Box((10., 10., D), c, 0., 'x')), D ** 3),
        'hex prism': (lambda c: Prism(HEX, D, 'z', c, 0.), HEX_AREA * D),
    }
    constants = {}
    for name, (make, exact) in bodies.items():
        vals = [plane_volume(make(tuple(o * h)), h, D * 1.15) for o in OFFSETS[:4]]
        constants[name] = (np.mean(vals) / exact - 1) * n ** 2
    assert all(0.6 < c < 1.45 for c in constants.values()), constants
    assert max(constants.values()) / min(constants.values()) < 2.1, constants
    assert abs(constants['sphere'] - 1.0) < 0.01, constants


def test_an_axis_aligned_box_a_cylinder_and_a_prism_are_exact():
    """The closed forms, to 1e-14 -- per voxel against the analytic overlap, and summed.

    **(M)** Making ``exact_fill`` return ``None`` for every body (so the plane is used
    everywhere) takes the summed errors to +2.6e-3, +3.1e-3 and +2.9e-3 and this fails three
    ways.
    """
    h = 0.1
    pts = grid(10, -0.5, h)
    centre = (0.013, -0.021, 0.007)
    cases = {
        'box': (Box((0.34, 0.22, 0.47), centre, 0.0), 0.34 * 0.22 * 0.47),
        'box rotated': (Box((0.34, 0.22, 0.47), centre, 0.63), 0.34 * 0.22 * 0.47),
        'box on x': (Box((0.34, 0.22, 0.47), centre, 0.63, 'x'), 0.34 * 0.22 * 0.47),
        'cylinder': (Cylinder(0.21, 0.33, 'z', centre), np.pi * 0.21 ** 2 * 0.33),
        'cylinder on y': (Cylinder(0.21, 0.33, 'y', centre), np.pi * 0.21 ** 2 * 0.33),
        'prism': (Prism(HEX, 0.33, 'z', centre, 0.41), HEX_AREA * 0.33),
        'prism on x': (Prism(HEX, 0.33, 'x', centre, 0.41), HEX_AREA * 0.33),
    }
    for name, (body, volume) in cases.items():
        fill, _ = occupancy(body, pts, h)
        assert abs(fill.sum() * h ** 3 - volume) < 1e-14, name
    # per voxel, against the product of three analytic overlaps
    box = Box((0.34, 0.22, 0.47), centre, 0.0)
    want = np.ones(pts.shape[:-1])
    for k, size in enumerate((0.34, 0.22, 0.47)):
        lo, hi = centre[k] - size / 2, centre[k] + size / 2
        want = want * np.maximum(np.minimum(hi, pts[..., k] + h / 2)
                                 - np.maximum(lo, pts[..., k] - h / 2), 0.) / h
    assert np.abs(occupancy(box, pts, h)[0] - want).max() < 1e-15


def test_a_non_convex_cross_section_is_the_same_number_on_three_backends():
    """The C1 blocker: one legal input, three answers, from a buffer sized for convex only.

    ``_Polygonal`` and ``Prism`` both declare a "simple counter-clockwise polygon" and
    ``clipped_polygon_area`` says "any ``nv >= 3``"; the buffer was ``nv + 4``, which is the
    bound for a **convex** ``nv``-gon and not for a simple one. Measured on a 12-vertex star
    before the fix: NumPy raised ``IndexError: index 16 is out of bounds for axis 2 with size
    16``, JAX returned ``fill = nan`` (``where(area < 0, 0, area)`` does not catch a NaN) and
    Torch returned ``0.38101489`` where the truth is ``0.34210489`` -- 11% wrong, silently, and
    up to 20.8% over a wider sweep. All three now read ``0.34210489``.

    Held against :func:`reference_clip`, a plain Python Sutherland-Hodgman with no buffer:
    worst 5.6e-17 over 300 random boxes for each of six non-convex outlines, and the three
    backends agree with each other to 2.8e-17.
    """
    import jax
    import jax.numpy as jnp
    import torch
    from array_api_compat import torch as xtorch
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(5)
    for name, outline in dict(NON_CONVEX, hexagon=HEX).items():
        verts = np.asarray(outline)
        side = rng.uniform(0.02, 0.5, 300)
        lo = rng.uniform(-0.6, 0.55, (300, 2))
        box = (lo[:, 0], lo[:, 0] + side, lo[:, 1], lo[:, 1] + side)
        convex = name == 'hexagon'
        got = clipped_polygon_area(verts[None], *box, np, convex=convex)
        want = np.array([reference_clip(verts, *(b[i] for b in box))[0] for i in range(300)])
        assert np.abs(got - want).max() < 1e-15, name
        on_jax = np.asarray(clipped_polygon_area(jnp.asarray(verts)[None],
                                                 *(jnp.asarray(b) for b in box), jnp,
                                                 convex=convex))
        on_torch = clipped_polygon_area(torch.tensor(verts)[None],
                                        *(torch.tensor(b) for b in box), xtorch,
                                        convex=convex).numpy()
        assert np.abs(on_jax - got).max() < 1e-15, name
        assert np.abs(on_torch - got).max() < 1e-15, name
        assert got.max() > 1e-3, name
    # and the whole body, through occupancy, on the input that used to raise
    star6 = Prism(star(6, 0.42, 0.10), 0.4, 'z', (0., 0., 0.), 0.)
    assert abs(float(occupancy(star6, np.zeros((1, 1, 3)), 0.5)[0][0, 0]) - 0.34210489) < 1e-7
    # the exported clip's DEFAULT is the safe one, and a caller who does not think about
    # `convex=` gets the correct answer. Nothing held that before: every call site in this file
    # passed `convex=` explicitly, so flipping the default to True passed the whole gate while
    # a bare call raised `IndexError: index 16 is out of bounds`.
    verts = np.asarray(star(6, 0.42, 0.10))
    one_box = (np.asarray([-0.2]), np.asarray([0.3]), np.asarray([-0.25]), np.asarray([0.35]))
    bare = float(np.ravel(clipped_polygon_area(verts[None], *one_box, np))[0])
    assert abs(bare - reference_clip(verts, -0.2, 0.3, -0.25, 0.35)[0]) < 1e-15, bare
    assert bare > 1e-3


def test_the_clip_buffer_holds_what_the_bound_says():
    """``clip_capacity`` is an upper bound on the slot count, and the convex one is not.

    The bound is combinatorial (``n' <= n + n//2`` per clip, four clips), so the only way to
    falsify it is to run the same recurrence with no buffer and watch the counts. 4000 random
    boxes per outline:

        outline    nv   convex cap   simple cap   worst realised
        hexagon     6       10           28            10
        L            6       10           28             6
        star5      10       14           49            17
        star6      12       16           60            20
        star8      16       20           81            28
        comb       12       16           60            12
        cross      12       16           60            12

    The three stars are the proof that the convex bound is not a bound: 17 > 14, 20 > 16,
    28 > 20. An L, a comb and a cross are non-convex and do **not** overflow it, which is why
    a gate built only from those would have passed the defect too.

    **(M)** ``clip_capacity`` returning ``nv + 4`` for every polygon fails this and the test
    above on all three backends.
    """
    rng = np.random.default_rng(12)
    overflows = 0
    for name, outline in dict(NON_CONVEX, hexagon=HEX).items():
        nv = len(outline)
        worst = 0
        for _ in range(1500):
            side = rng.uniform(0.01, 0.6)
            lo = rng.uniform(-0.6, 0.6, 2)
            worst = max(worst, reference_clip(outline, lo[0], lo[0] + side,
                                              lo[1], lo[1] + side)[1])
        assert worst <= clip_capacity(nv, False), (name, worst)
        overflows += worst > clip_capacity(nv, True)
        if name == 'hexagon':
            assert worst <= clip_capacity(nv, True), worst
    assert overflows >= 3, 'the stars must actually exceed the convex bound'
    assert clip_capacity(4, True) == 8 and clip_capacity(12, False) == 60


def test_the_clipped_ball_is_what_quadrature_says_it_is():
    """The one closed form here that is not a port, against an integral of a form that is.

    ``clipped_ball_volume`` sums eight corner primitives; the oracle reads the same volume as a
    stack of exactly-clipped **discs**, ``int clipped_disc_area(sqrt(R^2 - z^2)) dz``, by
    adaptive quadrature broken at the ball's poles. The two share no line. Measured over 300
    random ball/box configurations: worst **9.1e-14** absolute, and on the corner primitive
    alone 5.6e-16.

    **(M)** Writing ``asin(a/rho)`` for ``atan2(a, sqrt(rho^2 - a^2))`` -- the same angle,
    spelled the way it is usually written -- puts a 5.5e-08 jump at a regime boundary and takes
    the worst here to 1.3e-07, because ``asin`` near 1 loses half its digits. This test and the
    exactness test below are the two that see it.
    """
    from scipy.integrate import quad
    rng = np.random.default_rng(1)

    def by_discs(centre, radius, lo, hi):
        def area(z):
            rho2 = radius ** 2 - (z - centre[2]) ** 2
            return 0.0 if rho2 <= 0 else float(clipped_disc_area(
                np.float64(centre[0]), np.float64(centre[1]), np.sqrt(rho2), np.float64(lo[0]),
                np.float64(hi[0]), np.float64(lo[1]), np.float64(hi[1]), np))
        a, b = max(lo[2], centre[2] - radius), min(hi[2], centre[2] + radius)
        if b <= a:
            return 0.0
        knots = sorted({a, b} | {v for v in (centre[2] - radius, centre[2] + radius, centre[2])
                                 if a < v < b})
        return sum(quad(area, knots[i], knots[i + 1], limit=400, epsabs=1e-14, epsrel=1e-13)[0]
                   for i in range(len(knots) - 1))

    worst, nonzero = 0.0, 0
    for _ in range(120):
        radius = rng.uniform(0.05, 1.2)
        centre = rng.uniform(-1, 1, 3)
        lo = rng.uniform(-1.2, 0.9, 3)
        hi = lo + rng.uniform(0.01, 0.8)
        got = float(clipped_ball_volume(*(np.float64(v) for v in centre), radius,
                                        *(np.float64(v) for pair in zip(lo, hi) for v in pair),
                                        np))
        want = by_discs(centre, radius, lo, hi)
        nonzero += want > 1e-6
        worst = max(worst, abs(got - want))
    assert nonzero > 20, 'the configurations must actually intersect'
    assert worst < 1e-12, worst


def test_a_sphere_fills_a_voxel_exactly_and_this_is_what_that_replaced():
    """``Sphere.exact_fill`` against the analytic volume, and the plane it replaced beside it.

    Measured over 8 sub-cell offsets, the summed relative volume error of a sphere of radius
    0.5 on a grid of ``h = R/n``:

        R/h            3           4           6           8          12          16
        exact     -1.7e-16    +2.2e-16    +2.8e-16    +3.1e-16    +4.9e-16    +1.4e-16
        plane     +2.77e-02   +1.57e-02   +6.95e-03   +3.90e-03   +1.74e-03   +9.78e-04
        plane x (h/R)^-2  0.2492  0.2512  0.2501  0.2497  0.2502  0.2503

    That is the whole case for writing it: the bias the plane leaves is one-signed, so it does
    not cancel over a grid, and at G6's resolutions it is the same size as the error S2 fitted
    an order 1.83 through -- on a study that had measured that order with an exact fill in the
    first place.

    A voxel wholly inside the ball reads **exactly** 1.0 and one wholly outside **exactly** 0.0,
    which is the two ``where`` guards doing their job: without them the eight-corner sum
    differences numbers of order ``R^3`` into one of order ``h^3`` and reads ``-1.4e-03`` for
    voxels that are simply outside the ball at R/h = 16.

    What the guards cannot remove is the same cancellation on a genuinely *cut* voxel, where the
    answer really is of order ``h^3``. It is ``eps (R/h)^3`` of one voxel and it is measured,
    not bounded away:

        R/h              3        4        8       16       32
        fill.min    -8.7e-15    0.0      0.0      0.0      0.0
        fill.max-1      0.0      0.0      0.0   +4.6e-13  +8.2e-12
        eps (R/h)^3  5.9e-15  1.4e-14  1.1e-13  9.0e-13  7.2e-12

    -- worst/(R/h)^3 reads 3.2e-16, 2.5e-16 and 2.5e-16 on the three rows that have a number at
    all, i.e. between 1.1 and 1.5 ulp, which is what one cancellation costs. That is
    the same class as the disc clip this repository already ships (the C1 review measured
    ``Cylinder.exact_fill`` at 9.8e-11 against an independent quadrature), and it is inherent to
    any form built by inclusion-exclusion over corners: the rounding lives in the primitive, so
    no summation order removes it.

    **(M)** Dropping either guard fails this and the free-invariants test.
    """
    radius = 0.5
    rows = {}
    for npr in (3, 4, 8):
        h = radius / npr
        n = int(np.ceil((2 * radius + 6 * h) / h))
        exact, planed = [], []
        for offset in OFFSETS[:4]:
            body = Sphere(radius, tuple(offset * h))
            centres = (np.arange(n) + 0.5) * h - n * h / 2
            x, y = np.meshgrid(centres, centres, indexing='ij')
            total_e = total_p = 0.0
            for k0 in range(0, n, 16):
                z = centres[k0:k0 + 16]
                pts = np.stack(np.broadcast_arrays(x[..., None], y[..., None], z[None, None, :]),
                               axis=-1)
                fill = body.exact_fill(pts, h)
                slack = 4e-16 * npr ** 3          # 1.5 ulp of a voxel per (R/h)^3, measured
                assert fill.min() >= -slack, (npr, fill.min())
                assert fill.max() <= 1.0 + slack, (npr, fill.max())
                total_e += fill.sum()
                total_p += body.plane(pts, h)[0].sum()
            true = 4 / 3 * np.pi * radius ** 3
            exact.append(total_e * h ** 3 / true - 1)
            planed.append(total_p * h ** 3 / true - 1)
        rows[npr] = (np.mean(exact), np.mean(planed))
        assert abs(rows[npr][0]) < 1e-14, (npr, rows[npr])
        assert rows[npr][1] > 0, (npr, rows[npr])
        assert abs(rows[npr][1] * npr ** 2 - 0.250) < 0.005, (npr, rows[npr][1] * npr ** 2)
    assert rows[3][1] > rows[4][1] > rows[8][1], rows


def test_the_ported_closed_forms_reproduce_the_rcwa_paths_own():
    """The port is held against its source, so it cannot drift where nothing would notice.

    Measured: the disc-box and polygon-box areas are **bit-identical** over 500 random
    configurations; the edge moments agree to 5.6e-17 (the RCWA routine sums a 3x3 image ring,
    so it is compared at a period 40 times the shape, where the images contribute nothing).
    """
    from wavebit.fdtd._periodic.scene import _clipped_circle_area, _clipped_rectangle_area, _rectangle_edge_moments
    rng = np.random.default_rng(3)
    n = 500
    cx, cy = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    radius = rng.uniform(0.05, 1.0, n)
    left, bottom = rng.uniform(-1.2, 0.8, n), rng.uniform(-1.2, 0.8, n)
    side = rng.uniform(0.02, 0.6, n)
    right, top = left + side, bottom + side
    assert np.abs(clipped_disc_area(cx, cy, radius, left, right, bottom, top, np)
                  - _clipped_circle_area(np.stack([cx, cy, radius], axis=-1),
                                         left, right, bottom, top, xp=np)).max() == 0.0
    angle = rng.uniform(0, np.pi, n)
    hx, hy = rng.uniform(0.05, 0.7, n), rng.uniform(0.05, 0.7, n)
    co, si = np.cos(angle)[:, None], np.sin(angle)[:, None]
    loc = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    ux, uy = loc[:, 0][None, :] * hx[:, None], loc[:, 1][None, :] * hy[:, None]
    verts = np.stack([ux * co - uy * si + cx[:, None], ux * si + uy * co + cy[:, None]], axis=-1)
    assert np.abs(clipped_polygon_area(verts, left, right, bottom, top, np)
                  - _clipped_rectangle_area(verts, left, right, bottom, top, xp=np)).max() == 0.0
    # An edge 1e-9 rad from a voxel face, which the port's `32 eps max(L)` tolerance has to
    # treat as MOVING and not as parallel. Nothing placed one before: a mutant loosening that
    # tolerance by ten orders of magnitude -- every edge within 1e-6 of parallel then admitted
    # or dropped whole -- passed the entire gate. Here the edge enters the unit voxel at
    # t = 1/6 (it crosses u = 0) and leaves at t = 1/3 (it crosses v = 1), and |e| = 3, so the
    # arclength inside is exactly 0.5; treating the top face as parallel admits the whole
    # u-span and reads 1.0.
    near = np.asarray([(-0.5, 1.0 - 1e-9), (2.5, 1.0 + 2e-9), (2.5, 50.0)])
    unit = (np.asarray([0.0]), np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0]))
    grazed = polygon_edge_moments(near[None], *unit, np)
    assert abs(float(grazed[0][0]) - 3 * (1 / 3 - 1 / 6)) < 1e-9, float(grazed[0][0])
    assert abs(float(grazed[1][0] + grazed[2][0]) - float(grazed[0][0])) < 1e-15
    step, nx, ny = 0.37, 7, 5
    x, y = (np.arange(nx) + 0.5) * step, (np.arange(ny) + 0.5) * step
    rect = np.array([[0.4, 0.3], [1.9, 0.5], [1.7, 1.6], [0.2, 1.4]])
    length, mxx, _, myy = _rectangle_edge_moments(rect, x, y, step, step, (40., 40.), xp=np,
                                                  array=np.asarray, check=lambda c, m: None)
    left = (x - step / 2)[None, :] + np.zeros((ny, 1))
    bottom = (y - step / 2)[:, None] + np.zeros((1, nx))
    mine = polygon_edge_moments(rect[None, None, ...], left, left + step, bottom, bottom + step, np)
    assert length.sum() > 1.0
    for got, want in zip(mine, (length, mxx, myy)):
        assert np.abs(got - want).max() < 1e-16


def test_the_exact_face_moments_are_the_surfaces_own():
    """``<n n^T>`` for a rotated box, against 1200x1200 sampled points per face.

    Measured: worst 2.2e-3, rms 7.2e-4 -- which is the sampler's own 1/1200 discretisation. On
    the same voxels the tangent plane reads worst 5.5e-1 and rms 2.9e-1, because the nearest
    surface point of a corner voxel is on one face and the voxel holds two or three.

    **(M)** Using the *first* normal moment there -- ``N N^T/|N|^2`` over the same edges --
    is caught by the last two blocks, and it is worth saying exactly where, because the design's
    own wording points at a case that does **not** catch it. At a *symmetric* 90-degree corner
    the first moment's diagonal is ``(1/2, 1/2)``, which is what ``n n^T`` reads too: the two
    rules differ there only off the diagonal, and stage 4 reads the diagonal. What separates
    them on the diagonal is asymmetry -- faces of unequal area, where the second moment reads
    ``(L_1, L_2)/(L_1+L_2)`` and the first reads ``(L_1^2, L_2^2)/(L_1^2+L_2^2)``, so 2/3 and
    1/3 become 0.8 and 0.2 -- and cancellation, where a feature thinner than a cell puts two
    antiparallel faces in one voxel, ``N`` sums to exactly zero and the first rule has no answer
    at all. That is the laminate for which ``eps = <eps> I - Delta n n^T`` is exact, and it is
    the defect on record at ``src/wavebit/fdtd/_periodic/scene.py``:844-846.
    """
    h = 0.1
    size, centre, angle = (0.34, 0.26, 0.29), (0.013, -0.021, 0.007), 0.63
    pts = grid(8, -0.4, h).reshape(-1, 3)
    body = Box(size, centre, angle)
    fill, w = occupancy(body, pts, h)
    cutv = (fill > 1e-9) & (fill < 1 - 1e-9)
    area, want = _sampled_box_moment(size, centre, angle, pts[cutv], h)
    assert cutv.sum() > 40
    assert np.abs(w[cutv] - want).max() < 5e-3
    plane_area, plane_moment = body.plane(pts[cutv], h)[1:]
    plane_w = plane_moment / np.where(plane_area > 0, plane_area, 1.0)[:, None]
    assert np.abs(plane_w - want).max() > 0.3     # what the plane would have cost here
    origin = np.zeros((1, 3))
    # a symmetric corner: two faces of equal area, w = (1/2, 1/2, 0) -- and the first moment
    # reads the same diagonal here, which is why this case alone would not be a gate
    area, moment = Box((0.4, 0.4, 0.4), (0.2, 0.2, 0.0), 0.0).exact_moment(origin, h)
    assert np.abs(moment[0] / area[0] - np.array([0.5, 0.5, 0.0])).max() < 1e-14
    # an ASYMMETRIC corner: 0.06 of x face against 0.03 of y face, so (2/3, 1/3, 0) -- where
    # the first moment would read (L^2)/(sum L^2) = (0.8, 0.2, 0)
    area, moment = Box((0.4, 0.4, 0.4), (0.22, 0.19, 0.0), 0.0).exact_moment(origin, h)
    assert np.abs(moment[0] / area[0] - np.array([2 / 3, 1 / 3, 0.0])).max() < 1e-14
    assert abs((moment[0] / area[0]).sum() - 1.0) < 1e-14
    # a feature thinner than a cell: two antiparallel faces in one voxel, where a first moment
    # cancels to exactly zero and has no direction left to normalise
    area, moment = Box((0.03, 0.4, 0.4), (0., 0., 0.), 0.0).exact_moment(origin, h)
    assert abs(area[0] - 0.02) < 1e-15
    assert np.abs(moment[0] / area[0] - np.array([1.0, 0.0, 0.0])).max() < 1e-14
    # A cap landing EXACTLY on a voxel face is split equally between its two neighbours.
    # Its total area still counts once: measured with two open ends (the first writing) the total
    # cap area of a hexagonal prism whose caps sit on a face read 0.000000 against a truth of
    # 0.020000 -- the interface left the grid altogether -- and with two closed ends it is
    # 0.040000, counted by both neighbours. Nothing in the gate held either way.
    zs = (np.arange(-8, 9) + 0.5) * h
    column = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=-1)
    for centre, where in ((0.0, 'caps exactly on a voxel face'), (0.03, 'caps inside a voxel')):
        cap = Prism(HEX, 0.4, 'z', (0., 0., centre), 0.).exact_moment(column, h)[1][:, 2]
        assert abs(float(cap.sum()) - 0.02) < 1e-15, (where, cap.sum())
        assert int((cap > 0).sum()) == (4 if centre == 0 else 2), (where, cap)


def _sampled_box_moment(size, centre, angle, pts, h, n=1200):
    co, si = np.cos(angle), np.sin(angle)
    g = (np.arange(n) + 0.5) / n - 0.5
    a, b = np.meshgrid(g, g, indexing='ij')
    area = np.zeros(len(pts))
    moment = np.zeros((len(pts), 3))
    lo, hi = pts - h / 2, pts + h / 2
    for k, (wa, wb, wc) in enumerate(((size[1], size[2], size[0]), (size[2], size[0], size[1]),
                                      (size[0], size[1], size[2]))):
        for sgn in (-1., 1.):
            loc = np.zeros((n * n, 3))
            loc[:, (k + 1) % 3] = (a * wa).ravel()
            loc[:, (k + 2) % 3] = (b * wb).ravel()
            loc[:, k] = sgn * wc / 2
            nrm = np.zeros(3)
            nrm[k] = sgn
            world = np.stack([loc[:, 0] * co - loc[:, 1] * si + centre[0],
                              loc[:, 0] * si + loc[:, 1] * co + centre[1],
                              loc[:, 2] + centre[2]], axis=-1)
            turned = np.array([nrm[0] * co - nrm[1] * si, nrm[0] * si + nrm[1] * co, nrm[2]])
            hits = np.all((world[None] > lo[:, None]) & (world[None] < hi[:, None]), axis=-1)
            count = hits.sum(-1) * wa * wb / (n * n)
            area += count
            moment += count[:, None] * (turned * turned)[None, :]
    return area, moment / np.where(area > 0, area, 1.0)[:, None]


def test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs():
    """A cylinder has an exact fill and no exact moment. Measured, this is the price.

    Against 20001-point quadrature around the true circumference, cut voxels only
    (2026-09-20, radius 0.5, height 0.6):

        R/h      away from the rim            on the rim (|z - cap| < h)
                 voxels  worst    rms         voxels  worst    rms
         4          64  1.31e-2  9.63e-3        192  3.98e-1  2.04e-1
         8         512  1.92e-2  1.02e-2        456  5.09e-1  2.35e-1
        16        2304  7.19e-3  3.50e-3       1720  5.18e-1  1.68e-1

    Away from the rim the plane's normal is the true normal at the nearest point and the error
    is the arc's own spread, O(h/R), falling. **On the rim it does not converge at all**: the
    plane sees one surface where the voxel holds two, and a rim is a crease, not a smooth
    patch. The design's section 3 says a creased two-media interface is "handled exactly by the
    exact_fill/moment forms" -- for a cylinder's rim that is not true as the table stands, and
    C2 either writes the clipped arc's second moment or records this price where the rule is.
    """
    radius, height, h = 0.5, 0.6, 0.5 / 8
    n = int(np.ceil(1.5 / h))
    c = (np.arange(n) + 0.5) * h - n * h / 2
    pts = np.stack(np.meshgrid(c, c, c, indexing='ij'), axis=-1).reshape(-1, 3)
    cz = 0.007 * h
    body = Cylinder(radius, height, 'z', (0.013 * h, -0.021 * h, cz))
    fill, w = occupancy(body, pts, h)
    cutv = (fill > 1e-9) & (fill < 1 - 1e-9)
    pts, w = pts[cutv], w[cutv]
    area, want = _sampled_cylinder_moment(body, pts, h)
    rim = (np.abs(pts[:, 2] - (cz - height / 2)) < h) | (np.abs(pts[:, 2] - (cz + height / 2)) < h)
    side = (area > 0) & ~rim
    assert np.abs(w - want)[side].max() < 3e-2, 'the smooth side is what the plane is good for'
    assert np.abs(w - want)[(area > 0) & rim].max() > 0.3, 'and the rim is what it is not'


def test_a_crease_is_where_w_jumps_and_this_is_how_far():
    """What G4's "largest single-step change <= 1e-5" is worth on a body that is not smooth.

    Slide a body past one fixed voxel along a generic direction in steps of 0.0025 cell, and
    keep only the steps where **both** ends are cut voxels with a live ``w`` -- so the allowed
    ``w -> 0`` step is excluded and what is left is the crease alone. Measured (2026-09-21,
    ``h = 0.1``, 2400 steps per body):

        body                    steps   max |dw|   max |dfill|
        sphere                    668   5.0e-16      3.3e-03
        box, axis aligned         825    0.4455      3.3e-03
        box, rotated             1019    0.3170      2.7e-03
        hexagonal prism edge     1332    0.4577      2.1e-03
        cylinder cap rim         1485    1.0000      2.0e-03
        sphere minus a rod       2292   2.08e-02     3.0e-03
        sphere u box              842   1.69e-03     3.1e-03
        two spheres, n           1501   6.85e-03     2.7e-03
        box minus box            1484    0.4909      2.9e-03
        prism u cylinder         1674    0.1405      2.4e-03
        three-way (two cuts)     2217    0.5309      3.1e-03

    The cylinder's rim is the worst *leaf* case: ``w`` swaps from the side wall's
    ``(0.5, 0.5, 0)`` to the cap's ``(0, 0, 1)`` inside one 0.0025-cell move, a jump of the full
    range. G4 was measured on a sphere, where this file reads 5e-16, and is out by five orders of
    magnitude on a rim.

    **The last six rows are the news, and they are the rebuilt CSG rule's.** Every one of the
    ``max |dfill|`` entries is now 3.1e-03 or less -- the same size as a *leaf*'s, which is the
    slide step times the fill's own slope. The three previous writings of this paragraph said a
    CSG node's fill was discontinuous and quoted 3.15e-02, then 4.29e-02, then 4.48e-02 with a
    full 1.000 in ``w``, and built the constructor's refusal of a traced parameter on it. That
    step came from the composed plane: its fill is a function of the **selected** child's normal,
    and at a crease the selection swaps. A node now answers from its children's own fills and one
    exact two-plane cut, neither of which reads a selected normal, so the step is gone. Bisected
    at the worst point of the ``sphere minus a rod`` slide, over a geometry change of 5.6e-17:

        fill   0.9135725164 -> 0.9135725164      |dfill| 9.1e-15   (it was 3.15e-02)
        a central difference of fill^2 reads 7.62, 7.62, 7.62, 7.62, 7.62 at steps 1e-3..1e-7

    -- convergent, where the same probe diverged as ``1/step`` before.

    **``w`` still jumps, and that is now the whole of the reason a CSG node's geometry may not be
    traced.** At the worst live-to-live step of the ``box minus box`` slide, bisected to 2.8e-17,
    the fill is identical to ten digits and

        w  (0.0200, 0.4877, 0.4923)  ->  (0.0395, 0.9605, 0.0000)      |dw| 4.91e-01

    with a central difference of ``w_y`` reading 233, 2361, 23639, 236424, 2364267 at the same
    five steps: ``1/step``, the signature of a jump in the objective. ``occupancy`` returns the
    pair, and C2's mixing rule reads both halves of it, so a loss through a node is still
    discontinuous in a geometry parameter even though its fill is not. See
    :func:`test_a_traced_parameter_on_a_csg_node_is_refused_by_name`.
    """
    h = 0.1
    point = np.zeros((1, 3))
    step = 0.0025 * h
    direction = np.array([0.37, 0.61, 0.70])
    direction = direction / np.linalg.norm(direction)
    cases = {
        'sphere': (lambda c: Sphere(0.31, c), tuple(0.31 * direction), None),
        'box axis': (lambda c: Box((0.34, 0.26, 0.29), c, 0.0), (0.17, 0.13, 0.145), 0.3),
        'cyl rim': (lambda c: Cylinder(0.31, 0.44, 'z', c), (0.31, 0.0, 0.22), 0.9),
        'csg': (lambda c: Difference(Sphere(0.31, c), Cylinder(0.12, 1.0, 'z', c)),
                (0.12, 0.0, 0.0), None),
        'csg box minus box': (lambda c: Difference(Box((0.34, 0.26, 0.29), c, 0.2),
                                                   Box((0.2, 0.2, 0.5), tuple(np.add(c, 0.1)),
                                                       0.4)),
                              (0.15, 0.05, 0.05), 0.4),
        'csg prism u cylinder': (lambda c: Union(Prism(HEX, 0.4, 'z', c),
                                                 Cylinder(0.14, 0.5, 'z', c)),
                                 (0.17, 0.05, 0.1), 0.1),
    }
    for name, (make, start, want_w) in cases.items():
        ws, fills, live = [], [], []
        for i in range(2400):
            centre = np.asarray(start) + direction * (i - 1200) * step
            fill, w = occupancy(make(tuple(centre)), point, h)
            ws.append(w[0])
            fills.append(float(fill[0]))
            live.append(float(w[0].sum()) > 0)
        ws, fills, live = np.asarray(ws), np.asarray(fills), np.asarray(live)
        cut_here = (fills > 1e-9) & (fills < 1 - 1e-9) & live
        both = cut_here[:-1] & cut_here[1:]
        assert both.sum() > 100, name
        step_w = np.abs(np.diff(ws, axis=0)).max(-1)[both].max()
        step_f = np.abs(np.diff(fills))[both].max()
        if name == 'sphere':
            assert step_w < 1e-12, (name, step_w)    # smooth surface, smooth moment
        elif want_w is not None:
            assert step_w > want_w, (name, step_w)   # a crease, and this is how big
        # EVERY row, leaf and node alike: the fill is continuous at the slide's resolution
        assert step_f < 4e-3, (name, step_f)
    # the node fill's old jump, bisected rather than sampled: it is not there any more
    make, start = cases['csg'][0], cases['csg'][1]

    def fill_at(s):
        centre = np.asarray(start) + direction * s
        return float(occupancy(make(tuple(centre)), point, h)[0][0])

    lo, hi = -0.2785, -0.27825
    low, high = fill_at(lo), fill_at(hi)
    for _ in range(45):
        mid = (lo + hi) / 2
        if abs(fill_at(mid) - low) < abs(fill_at(mid) - high):
            lo = mid
        else:
            hi = mid
    assert hi - lo < 1e-15
    assert abs(fill_at(hi) - fill_at(lo)) < 1e-12, (fill_at(lo), fill_at(hi))
    slope = [(fill_at(-0.2785 + s) ** 2 - fill_at(-0.2785 - s) ** 2) / (2 * s)
             for s in (1e-3, 1e-5, 1e-7)]
    assert max(slope) - min(slope) < 1e-2, slope     # convergent, not 1/step

    # and what IS still discontinuous: w, at a crease, with both ends cut voxels
    make, start = cases['csg box minus box'][0], cases['csg box minus box'][1]

    def pair_at(s):
        centre = np.asarray(start) + direction * s
        fill, w = occupancy(make(tuple(centre)), point, h)
        return float(fill[0]), np.asarray(w)[0]

    lo, hi = -0.20725, -0.207
    f_lo, w_lo = pair_at(lo)
    f_hi, w_hi = pair_at(hi)
    for _ in range(55):
        mid = (lo + hi) / 2
        f_m, w_m = pair_at(mid)
        if np.abs(w_m - w_lo).max() < np.abs(w_m - w_hi).max():
            lo, f_lo, w_lo = mid, f_m, w_m
        else:
            hi, f_hi, w_hi = mid, f_m, w_m
    assert hi - lo < 1e-15
    assert abs(f_hi - f_lo) < 1e-12, (f_hi, f_lo)            # the fill does not step
    # Half-area boundary sharing splits the old jump into two. The final
    # floating-point bracket now meets the half-area state (gap .326510).
    # Keep the >.4 whole-crease gate on BOTH existing one-sided ladder rungs:
    # measured .491610 and .492277, with unchanged slopes 2361 and 236424.
    rung = []
    for s in (1e-4, 1e-6):
        left = pair_at((lo + hi) / 2 - s)[1]
        right = pair_at((lo + hi) / 2 + s)[1]
        assert np.abs(right - left).max() > 0.4, np.abs(right - left).max()
        rung.append((right[1] - left[1]) / (2 * s))
    assert abs(rung[1]) > 50 * abs(rung[0]), rung           # 1/step: the loss itself jumps


def test_a_union_of_flush_bodies_reads_solid_material_on_the_seam():
    """A rod standing on a substrate of the same material: the joint is solid, not half solid.

    ``min`` of two signed distances is the union's distance *outside* it and not inside, so where
    two children's surfaces are flush both ``d`` are zero in the middle of solid material and the
    composed tangent plane cuts the voxel in half. Measured on the kernel before the rule in
    ``_bodies._Combination`` (``h = 0.02``, 40^3, the joint on a voxel CENTRE):

        60 voxels straddling the joint, well inside the rod's footprint
          fill    = 0.5        <- the truth is 1.0
          sum_c w = 1.0        <- the free invariant is satisfied BY the artefact
          w       = (0, 0, 1)  <- a full z-facing interface through solid material

    It was first order in ``h``, identical on NumPy, JAX and Torch, and invisible to both free
    invariants -- and ``fd.Union`` is the design's own instruction to a user whose bodies share a
    material. It now reads ``fill = 1`` to 1.2e-15 and ``w = 0`` **exactly**, on all three.

    The second half is what says the rule is not a special case of one axis: two boxes meeting on
    a plane, whose union is exactly ``Box((0.8, 0.4, 0.4))``, with the seam placed at 0, 1/4 and
    **1/2** of a cell -- **and the grid origin swept with it**, which is what the round-5 review
    found missing. The previous rule read "worst |dfill| 0.0000" at the one alignment this test
    used and 0.0625 / 0.1875 / 0.0625 a quarter of a cell away, 0.2500 at half a cell: the
    offending voxels were the ones whose centre lies on the union's OUTER face as well as on the
    seam. Over the product of 3 seam positions and **7 grid origins** (2026-09-21):

        grid origin   0.000  0.125  0.250  0.375  0.500  0.625  0.750
        worst |dfill| 2.7e-15 at every one of the 21 cells, union volume 0.12800000 at every one

    so the number is a property of the rule and not of an alignment.
    """
    h = 0.02
    n = 40
    c = -n * h / 2 + (np.arange(n) + 0.5) * h
    pts = np.stack(np.meshgrid(c, c, c, indexing='ij'), axis=-1)
    joint = float(c[n // 2])                      # a voxel CENTRE: the placement that was worst
    body = Union(Box((0.7, 0.7, 0.3), center=(0., 0., joint - 0.15)),
                 Cylinder(0.12, 0.3, 'z', center=(0., 0., joint + 0.15)))
    radius = np.hypot(pts[..., 0], pts[..., 1])
    band = (np.abs(pts[..., 2] - joint) < h / 4) & (radius < 0.12 - 1.5 * h)
    assert band.sum() > 50
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    for lib, name in ((np, 'numpy'), (jnp, 'jax'), (pytest.importorskip('torch'), 'torch')):
        fill, w = occupancy(body, lib.asarray(pts, dtype=lib.float64), h)
        fill, w = np.asarray(fill), np.asarray(w)
        assert np.abs(fill[band] - 1.0).max() < 1e-14, (name, fill[band].min())
        assert np.abs(w[band]).max() == 0.0, (name, np.abs(w[band]).max())
    # the volume, against the analytic truth, on the same scene
    truth = 0.7 * 0.7 * 0.3 + np.pi * 0.12 ** 2 * 0.3
    got = float(occupancy(body, pts, h)[0].sum()) * h ** 3
    assert abs(got - truth) / truth < 3e-4, (got, truth)
    # two boxes meeting on a plane: the union of them IS a box, at three sub-cell placements
    hh = 0.05
    m = 24
    for origin in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75):
        cc = -m * hh / 2 + (np.arange(m) + 0.5) * hh + origin * hh
        grid3 = np.stack(np.meshgrid(cc, cc, cc, indexing='ij'), axis=-1)
        exact = occupancy(Box((0.8, 0.4, 0.4)), grid3, hh)[0]
        for frac in (0.0, 0.25, 0.5):
            seam = frac * hh
            left = Box((0.4 + seam, 0.4, 0.4), center=(-0.4 + (0.4 + seam) / 2, 0., 0.))
            right = Box((0.4 - seam, 0.4, 0.4), center=(0.4 - (0.4 - seam) / 2, 0., 0.))
            got = occupancy(Union(left, right), grid3, hh)[0]
            assert np.abs(got - exact).max() < 1e-13, (origin, frac, np.abs(got - exact).max())
            assert abs(float(got.sum()) * hh ** 3 - 0.128) < 1e-12, (origin, frac)
    # A T-junction: three boxes whose union is one box, with TWO coincident faces meeting in the
    # same voxel. It is the case that says `_bodies._no_common_volume`'s boundary must be `>= 0`
    # and not `> 0` -- a pair sharing a plane shares no VOLUME -- and it is the case the round-6
    # review found the per-voxel refusal firing on: the three bodies are pairwise DISJOINT and
    # the node answers them exactly, but it refused at 16 of 16 random sub-cell grid origins and
    # answered at the 1 this test used. **The origin is swept now**, both the aligned lattice
    # (a voxel centre on both seams) and the half-offset one, at 18 placements.
    tee = Union(Box((1.0, 1.0, 0.4), center=(0., 0., -0.2)),
                Box((0.5, 1.0, 0.4), center=(-0.25, 0., 0.2)),
                Box((0.5, 1.0, 0.4), center=(0.25, 0., 0.2)))
    rng = np.random.default_rng(4)
    worst_tee = 0.0
    for origin in [0.0, 0.5] + list(rng.random(7)):
        for centred in (0.0, 0.5):
            cc = (np.arange(25) - 12 + centred + origin) * hh
            grid3 = np.stack(np.meshgrid(cc, cc, cc, indexing='ij'), axis=-1)
            exact = np.asarray(occupancy(Box((1.0, 1.0, 0.8)), grid3, hh)[0])
            got = np.asarray(occupancy(tee, grid3, hh)[0])
            worst_tee = max(worst_tee, float(np.abs(got - exact).max()))
    assert worst_tee < 1e-13, worst_tee
    # and the same union read under jax.jit, which is where the refusal could not fire and the
    # node fell back to the composed plane: 0.7500 against a true 1.0, with w = (1, 0, 0) in
    # solid material, in 19 voxels. The two paths are one function now.
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    for origin in (0.0, 0.25):
        cc = (np.arange(25) - 12 + origin) * hh
        grid3 = np.stack(np.meshgrid(cc, cc, cc, indexing='ij'), axis=-1)
        exact = np.asarray(occupancy(Box((1.0, 1.0, 0.8)), grid3, hh)[0])
        jitted = np.asarray(jax.jit(lambda p: occupancy(tee, p, hh)[0])(jnp.asarray(grid3)))
        assert np.abs(jitted - exact).max() < 1e-13, (origin, np.abs(jitted - exact).max())
    cc = (np.arange(25) - 12) * hh              # back to a voxel CENTRE on the joint
    grid3 = np.stack(np.meshgrid(cc, cc, cc, indexing='ij'), axis=-1)
    # And the remedy the constructor's warning names: children OVERLAPPING by a fraction of a
    # cell, so that both faces are in one voxel and the material is solid between them. That is
    # the pair `_cut.two_plane` answers on its anti-parallel branch, and the branch that decides
    # whether the two faces are buried in each other's material is `_cut._ON_EACH_OTHER`.
    for fraction in (0.2, 0.4, 0.8):
        sunk = fraction * hh
        under = Box((0.5, 0.5, 0.4), center=(0., 0., -0.2 + sunk / 2))
        over = Box((0.5, 0.5, 0.4), center=(0., 0., 0.2 - sunk / 2))
        fill, w = occupancy(Union(under, over), grid3, hh)
        band = (np.abs(grid3[..., 0]) < 0.15) & (np.abs(grid3[..., 1]) < 0.15) \
            & (np.abs(grid3[..., 2]) < hh / 4)
        assert band.sum() > 20
        assert np.abs(np.asarray(fill)[band] - 1.0).max() < 1e-13, fraction
        assert np.abs(np.asarray(w)[band]).max() == 0.0, fraction
    # and the case the rule CANNOT do says so, by name, at construction and not per voxel: a
    # node whose own answer is one end of an interval cannot be a child, because which end is
    # safe depends on the operation above it.
    for spelling in (lambda: Union(Union(Sphere(0.2), Box((0.2, 0.2, 0.2))),
                                   Cylinder(0.1, 0.4, 'z')),
                     lambda: Intersection(Intersection(Sphere(0.3), Box((0.4, 0.4, 0.4))),
                                          Cylinder(0.1, 0.4, 'z')),
                     lambda: Difference(Sphere(0.3), Union(Box((0.2, 0.2, 0.2)),
                                                           Cylinder(0.1, 0.4, 'z'))),
                     lambda: Union(Intersection(Sphere(0.3), Box((0.4, 0.4, 0.4))),
                                   Cylinder(0.1, 0.4, 'z'))):
        with pytest.raises(ValueError, match='must answer its own filled fraction'):
            spelling()
    # a flat node of leaves is legal, and a NON-CONVEX Prism is a legal child: it used to put the
    # whole node on the composed plane with a warning, i.e. back to the half-a-voxel seam.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        Union(Sphere(0.2), Box((0.2, 0.2, 0.2)), Cylinder(0.1, 0.4, 'z'))
        loose = Union(Prism(star(6, 0.42, 0.10), 0.3, 'z'), Sphere(0.2))
    cc = (np.arange(20) - 10 + 0.63) * hh
    grid3 = np.stack(np.meshgrid(cc, cc, cc, indexing='ij'), axis=-1).reshape(-1, 3)
    loose_fill = np.asarray(occupancy(loose, grid3, hh)[0])
    kids = np.stack([np.asarray(b.parts(grid3, hh)[0]) for b in loose.bodies], axis=0)
    assert float((kids.max(axis=0) - loose_fill).max()) < 1e-12      # no hole
    assert float((loose_fill - np.minimum(1.0, kids.sum(axis=0))).max()) < 1e-12


def test_no_node_kind_steps_at_first_contact():
    """Drive two children head on through each other: nothing jumps, for any node kind.

    The round-5 review found that :class:`Intersection`'s first contact stepped by **0.5002** --
    the largest step a fill can take -- in a branch the gate's step test never touched, and that
    :class:`Union`'s angled case stepped by 3.384e-01, which the previous round measured, priced
    and called "the honest cost". Both came from the same thing: the rule applied to some pairs
    and not others, and switching between the rule and the composed plane is a jump.

    There is nothing to switch between now. Every voxel is answered by the same three cases --
    a boolean, one child's own cut, or one exact two-plane cut -- and each agrees with the next
    at the boundary between them, because a child that is about to stop crossing the voxel
    contributes a fill of 0 or 1 and an area of 0 either way, and a pair that is about to start
    sharing volume shares none of it at the moment of contact. Measured, one voxel, 4000 slide
    steps of 3e-05 cell (2026-09-21):

        node                                   max |dfill|     before
        Union(box, box) head on                  3.000e-04     4.998e-01 without the slab term
        Union(sphere, sphere)                    3.000e-04
        Union(box, cylinder) standing            3.000e-04
        Union(sphere, box tilted 0.4)            3.000e-04     3.384e-01
        Union(prism, cylinder)                   3.000e-04
        Intersection(box, box) head on           3.000e-04     5.002e-01
        Intersection(sphere, sphere)             3.000e-04     5.001e-01
        Intersection(sphere, box tilted)         3.000e-04
        Difference(box, box)                     3.000e-04
        Difference(sphere, cylinder)             3.000e-04
        Difference(box, sphere tilted in)        3.000e-04
        Difference(box, box) tilted in           3.000e-04

    3.0e-04 **is** the slide step: the fill moves by one slide step per slide step, which is no
    jump at all. (The table's numbers are 4000 steps from a gap of 0.06; the test runs 1000 from
    0.015, which straddles contact at the same resolution and reads the same 3.000e-04.) The ``before`` column is the round-5 review's own instrument on the code it
    reviewed. The fill is also asserted to stay in ``[0, 1]`` through the whole slide, because
    the failure mode the slab term was originally added for was a fill running continuously past
    1, which a step test alone cannot see.
    """
    h = 0.1
    point = np.zeros((1, 3))

    def worst_step(make, start=0.015, step=3e-5, n=1000):
        previous, worst = None, 0.0
        for i in range(n):
            gap = start - i * step
            fill = float(occupancy(make(gap), point, h)[0][0])
            assert -1e-12 <= fill <= 1 + 1e-12, (gap, fill)
            if previous is not None:
                worst = max(worst, abs(fill - previous))
            previous = fill
        return worst

    cases = {
        'union boxes': lambda g: Union(Box((0.3, 0.4, 0.4), center=(-0.15 - g / 2, 0., 0.)),
                                       Box((0.3, 0.4, 0.4), center=(0.15 + g / 2, 0., 0.))),
        'union spheres': lambda g: Union(Sphere(0.25, (-0.25 - g / 2, 0., 0.)),
                                         Sphere(0.25, (0.25 + g / 2, 0., 0.))),
        'union cylinder on a box': lambda g: Union(
            Box((0.5, 0.5, 0.3), center=(0., 0., -0.15 - g / 2)),
            Cylinder(0.12, 0.3, 'z', center=(0., 0., 0.15 + g / 2))),
        'union tilted 0.4': lambda g: Union(
            Sphere(0.25, (-0.25 - g / 2, 0., 0.)),
            Box((0.3, 0.4, 0.4), center=(0.15 + g / 2, 0.02, 0.01), angle=0.4)),
        'intersection boxes': lambda g: Intersection(
            Box((0.3, 0.4, 0.4), center=(-0.15 + g / 2, 0., 0.)),
            Box((0.3, 0.4, 0.4), center=(0.15 - g / 2, 0., 0.))),
        'intersection spheres': lambda g: Intersection(Sphere(0.25, (-0.25 + g / 2, 0., 0.)),
                                                       Sphere(0.25, (0.25 - g / 2, 0., 0.))),
        'intersection tilted': lambda g: Intersection(
            Sphere(0.3, (-0.28 + g / 2, 0., 0.)),
            Box((0.34, 0.4, 0.4), center=(0.17 - g / 2, 0.02, 0.01), angle=0.4)),
        'difference boxes': lambda g: Difference(
            Box((0.4, 0.4, 0.4)), Box((0.3, 0.4, 0.4), center=(g + 0.15, 0., 0.))),
        'difference sphere minus rod': lambda g: Difference(
            Sphere(0.3), Cylinder(0.12, 0.6, 'x', center=(0., g + 0.12, 0.02))),
        'difference tilted in': lambda g: Difference(
            Box((0.4, 0.4, 0.4)), Box((0.3, 0.4, 0.4), center=(g + 0.16, 0.02, 0.01), angle=0.4)),
    }
    for name, make in cases.items():
        got = worst_step(make)
        assert got < 1e-3, (name, got)            # 3.0e-04 measured: the slide step, no jump


def _device_scenes(h, gap):
    """A substrate and one pillar, four pillar shapes, at a given joint offset in cells.

    ``gap = 0`` is flush -- the pillar's foot plane and the substrate's top plane coincide, which
    is what a device is -- and a negative ``gap`` sinks the pillar into the substrate by that
    many cells, which is what :class:`Union`'s own warning tells a user to do.
    """
    z0 = gap * h
    sub = Box((3.0, 3.0, 0.6), center=(0.0, 0.0, -0.3))
    tall = 0.5 - z0
    high = (0.5 + z0) / 2
    return {
        'round pillar': Union(sub, Cylinder(0.24, tall, 'z', center=(0.0, 0.0, high))),
        'angled pillar': Union(sub, Box((0.34, 0.34, tall), center=(0.0, 0.0, high), angle=0.4)),
        'hex pillar': Union(sub, Prism(HEX_BIG, tall, 'z', center=(0.0, 0.0, high))),
        'ball on a substrate': Union(sub, Sphere(0.3, (0.0, 0.0, 0.3 + z0))),
        'a three-way stack': Union(Box((1.0, 1.0, 0.3), center=(0.0, 0.0, -0.15 + z0)),
                                   Box((0.4, 0.4, 0.3), center=(0.0, 0.0, 0.15)),
                                   Box((0.2, 0.2, 0.3), center=(0.0, 0.0, 0.45 - z0))),
    }


#: A hexagon big enough to be a pillar rather than a feature, counter-clockwise.
HEX_BIG = tuple((0.22 * np.cos(2 * np.pi * k / 6), 0.22 * np.sin(2 * np.pi * k / 6))
                for k in range(6))

#: Six sub-cell origins for the grid itself. A worst quoted at one alignment is a sample, and
#: this file has quoted four of those in four rounds; the product with the scene is the claim.
GRID_ORIGINS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.5), (0.23, 0.11, 0.5),
                (0.37, 0.37, 0.37), (0.5, 0.5, 0.5))


def _voxels(n, h, origin):
    c = -n * h / 2 + (np.arange(n) + 0.5) * h
    return np.stack(np.meshgrid(c, c, c, indexing='ij'), axis=-1) + np.asarray(origin) * h


def _sampled_fill(body, pts, h, m, seed):
    """The Monte-Carlo fill of every voxel, from the body's own MEMBERSHIP and nothing else.

    A node's composed distance has the right sign everywhere -- ``min`` / ``max`` / ``max`` of
    the negation of exact signed distances -- whatever its magnitude does, so ``d <= 0`` is an
    exact point-in-body test and stratified sampling of it is an unbiased oracle that shares no
    line of code with the fill it judges. ``m^3`` samples per voxel, jittered inside their own
    cells, so the standard error is at most ``0.5 / m^1.5``.
    """
    rng = np.random.default_rng(seed)
    lattice = (np.arange(m) + 0.5) / m - 0.5
    offsets = np.stack(np.meshgrid(lattice, lattice, lattice, indexing='ij'),
                       axis=-1).reshape(-1, 3)
    jitter = (rng.random(offsets.shape) - 0.5) / m
    total = np.zeros(pts.shape[:-1])
    for one, extra in zip(offsets, jitter):
        total = total + (np.asarray(body.surface(pts + (one + extra) * h)[0]) <= 0)
    return total / len(offsets)


def test_a_union_is_never_read_below_the_child_that_fills_the_voxel():
    """The blocker of the round-5 review, and the two theorems that make it unwritable.

    The rule this replaces used ``max(0, f_i + f_j - 1)`` -- the volume two tangent HALF SPACES
    share -- as the volume two BODIES share, for every pair, including pairs whose second child's
    surface was nowhere near the voxel. On the docstring's own recommended remedy (overlap the
    children by a cell) that subtraction over-fired: a voxel lying **entirely inside** a
    substrate, with a pillar's curved wall passing through it, read ``fill = 0.77`` where the
    truth is 1.0, and up to 0.31 over five pillar shapes.

    Two statements make that arithmetic impossible now, and both are theorems about any two sets
    in one voxel rather than margins::

        max(0, F_a + F_b - 1)  <=  shared  <=  min(F_a, F_b)

    The upper one gives ``fill = sum F_i - shared >= max_i F_i``; the lower one gives
    ``fill <= min(1, sum F_i)``. Measured over 5 scenes x 3 joint offsets x 6 **grid origins**,
    24^3 voxels at ``h`` = 0.05 (2026-09-21):

        worst  max_i F_i - fill           4.9e-14
        worst  fill - min(1, sum F_i)     2.2e-14

    -- round-off, on both sides, and the same numbers on NumPy, JAX and Torch.

    **And where the children are disjoint the node is exactly their sum.** A pillar standing
    flush on a substrate touches it on a plane and shares no volume with it, so the union's fill
    *is* ``F_substrate + F_pillar`` -- an identity with no Monte Carlo in it and nothing to
    tune. Over the same product of shapes and grid origins the worst departure is **1.1e-14**.
    That is the seam measurement the review asked to be re-quoted over origins: the previous
    rule's "worst |dfill| 0.0000" held at the one alignment its test used and read 0.2500 half a
    cell away.
    """
    h = 0.05
    n = 24
    worst_floor = worst_roof = worst_sum = 0.0
    for gap in (0.0, -1.0, -2.0):
        for name, body in _device_scenes(h, gap).items():
            for origin in GRID_ORIGINS:
                pts = _voxels(n, h, origin)
                fill = np.asarray(occupancy(body, pts, h)[0])
                own = [np.asarray(child.parts(pts, h)[0]) for child in body.bodies]
                worst_floor = max(worst_floor, float((np.maximum.reduce(own) - fill).max()))
                worst_roof = max(worst_roof, float((fill - np.minimum(1.0, sum(own))).max()))
                if gap == 0.0:                   # flush: the children are disjoint
                    worst_sum = max(worst_sum, float(np.abs(fill - sum(own)).max()))
    assert worst_floor < 1e-12, worst_floor      # 4.9e-14: no voxel reads below its fullest child
    assert worst_roof < 1e-12, worst_roof        # 2.2e-14
    assert worst_sum < 1e-12, worst_sum          # 1.1e-14: disjoint children add, exactly


def test_a_union_of_flush_bodies_is_the_same_number_on_three_backends():
    """One body of code, three backends, on the scene the rule exists for."""
    h = 0.05
    pts = _voxels(16, h, (0.23, 0.11, 0.5))
    body = _device_scenes(h, -1.0)['round pillar']
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    base = None
    for lib, name in ((np, 'numpy'), (jnp, 'jax'), (pytest.importorskip('torch'), 'torch')):
        fill, w = occupancy(body, lib.asarray(pts, dtype=lib.float64), h)
        got = (np.asarray(fill), np.asarray(w))
        if base is None:
            base = got
        assert np.abs(got[0] - base[0]).max() < 1e-13, name
        assert np.abs(got[1] - base[1]).max() < 1e-13, name


def test_a_node_never_deletes_an_interface_its_children_report():
    """The second half of the round-5 review: ``w = 0`` where there is a full interface.

    The rule this replaces deleted **both** children's whole face wherever a pair was back to
    back and flush. At the foot perimeter of a pillar that is wrong twice over: the substrate's
    top is still exposed outside the pillar's footprint, and the pillar's side wall is in the
    same voxel. 108 of 5360 truly cut voxels of a 3x3 pillar array reported ``sum_c w_c = 0`` --
    no interface at all -- where the answer before that rule was a full z-facing one, and the
    liveness test could not flag it because it reads the area the rule had just zeroed.

    A subtraction now happens only where both children are proved **flat** in the voxel: their
    own exact fill equals their own tangent half space's, which for a convex body says the two
    sets are the same set there. A box with an **edge** in the voxel is not flat -- at the
    review's own voxel its exact fill is 0.21 against its half space's 0.30 -- so nothing is
    subtracted and both faces stand.

    What is asserted here is the invariant, not a count: in every voxel the Monte Carlo says is
    cut **and** in which some crossing child reports a live interface of its own, the node
    reports one too. Measured over 5 scenes x 3 joint offsets x 6 grid origins: 0 violations of
    98326 such voxels, against 108 of 5360 on one scene before.
    """
    h = 0.06
    n = 14
    bad = live_cut = 0
    for gap in (0.0, -1.0):
        for name, body in _device_scenes(h, gap).items():
            for origin in GRID_ORIGINS[:3]:
                pts = _voxels(n, h, origin)
                w = np.asarray(occupancy(body, pts, h)[1])
                child_area = np.zeros(w.shape[:-1])
                for child in body.bodies:
                    child_area = np.maximum(child_area, np.asarray(child.parts(pts, h)[1]))
                exact = _sampled_fill(body, pts, h, m=5, seed=4)
                cut = (exact > 1e-9) & (exact < 1 - 1e-9) & (child_area > 1e-12 * h * h)
                live_cut += int(cut.sum())
                bad += int((cut & (np.abs(w).sum(-1) < 0.5)).sum())
    assert live_cut > 2000, live_cut
    assert bad == 0, (bad, live_cut)
    # A flush pair is fused in a UNION and must not be in a DIFFERENCE: two children of a union
    # that meet flush have material on both sides and no interface between them, but a removed
    # body resting flush on the face of the body it is taken from leaves vacuum there, and the
    # face is still the difference's own boundary. Same geometry, opposite answers.
    hh = 0.05
    pts = _voxels(12, hh, (0.0, 0.0, 0.5))
    kept = Box((0.5, 0.5, 0.4), center=(0.0, 0.0, 0.0))
    resting = Box((0.5, 0.5, 0.4), center=(0.0, 0.0, 0.4))          # its foot IS kept's top face
    deep = (np.abs(pts[..., 0]) < 0.15) & (np.abs(pts[..., 1]) < 0.15) \
        & (np.abs(pts[..., 2] - 0.2) < hh / 4)
    assert deep.sum() > 20
    union_w = np.asarray(occupancy(Union(kept, resting), pts, hh)[1])
    cut_fill, cut_w = occupancy(Difference(kept, resting), pts, hh)
    assert np.abs(union_w[deep]).max() == 0.0                        # buried: solid on both sides
    assert np.abs(np.asarray(cut_fill)[deep] - 0.5).max() < 1e-12    # the face is kept's own top
    assert np.abs(np.asarray(cut_w)[deep] - np.array([0., 0., 1.])).max() < 1e-12
    # Two opposed faces a FRACTION of a voxel apart are two interfaces and a gap, not a joint.
    # This is what holds `_bodies._SAME_FACE` at 1e-12: widen it to 0.4 of a voxel and the pair
    # below is called "on each other", both faces come out of the sum, and a 0.2-cell air gap
    # reads as solid material with no interface in it.
    for fraction in (0.2, 0.4, 0.8):
        space = fraction * hh
        lower = Box((0.5, 0.5, 0.4), center=(0.0, 0.0, -0.2 - space / 2))
        upper = Box((0.5, 0.5, 0.4), center=(0.0, 0.0, 0.2 + space / 2))
        fill, w = occupancy(Union(lower, upper), pts, hh)
        middle = (np.abs(pts[..., 0]) < 0.15) & (np.abs(pts[..., 1]) < 0.15) \
            & (np.abs(pts[..., 2]) < hh / 4)
        assert middle.sum() > 20
        assert np.abs(np.asarray(fill)[middle] - (1 - fraction)).max() < 1e-12, fraction
        assert np.abs(np.asarray(w)[middle] - np.array([0., 0., 1.])).max() < 1e-12, fraction


def test_the_union_of_two_bodies_is_at_or_above_the_truth_and_this_is_how_far_above():
    """"An approximation that can remove material that is there is not an option", measured.

    :func:`_cut.two_plane` answers exactly for the two tangent **half spaces**. A convex body is
    inside its own, so that volume is an upper bound on what the two bodies share, and
    ``V - gap_a - gap_b`` is a lower one, with ``gap_i = f_i - F_i`` the child's own
    tangent-plane over-fill of this voxel (``H_a n H_b`` minus ``A n B`` is inside
    ``(H_a \\ A) u (H_b \\ B)``). A union subtracts the **lower** bound, so its fill is provably
    at or above the truth; an intersection **is** the shared volume, so it takes the lower bound
    too, and reads at or below the truth, which is the side on which it cannot invent material.
    Where both children are flat the two bounds are the same number and the answer is exact.

    Measured against 14^3 = 2744 stratified membership samples per voxel (standard error 9.5e-3),
    16^3 voxels at ``h`` = 0.06, over 6 grid origins (2026-09-21):

        scene                          max(truth - fill)   max(fill - truth)
        round pillar, flush                  0.0091             0.0048
        angled pillar, flush                 0.0086             0.0045
        hex pillar, flush                    0.0078             0.0058
        ball on a substrate, flush           0.0070             0.0069
        round pillar, sunk one cell          0.0091             0.0111
        angled pillar, sunk one cell         0.0086             0.0927
        hex pillar, sunk one cell            0.0078             0.0553
        ball on a substrate, sunk one cell   0.0070             0.0200
        a three-way stack, flush             0.0047             0.0064

    The left column never clears the sampling error: nothing removes material. The right column
    is what never removing it costs -- 0.093 of a voxel where a *box* pillar overlaps a
    substrate, because a box with an edge in the voxel is not its own tangent half space and the
    bound is then a bound and not the answer. Before this round the same left column read 0.2500
    (and 0.77 against a true 1.0 in the review's own probe).
    """
    h = 0.06
    n = 12
    error = 0.5 / 10 ** 1.5
    worst_removed = 0.0
    for gap in (0.0, -1.0):
        for name, body in _device_scenes(h, gap).items():
            for origin in GRID_ORIGINS[:3]:
                pts = _voxels(n, h, origin)
                fill = np.asarray(occupancy(body, pts, h)[0])
                exact = _sampled_fill(body, pts, h, m=10, seed=5)
                worst_removed = max(worst_removed, float((exact - fill).max()))
    assert worst_removed < 4 * error, (worst_removed, error)


def test_three_surfaces_in_one_voxel_are_answered_on_the_safe_side_and_counted():
    """Two pairs of children sharing one voxel is the case this kernel cannot write down exactly.

    The shared volume of two half spaces in a box is a closed form; of three it is Lasserre's
    recursion, which an earlier round measured at 440 s and 22 GiB for one component of one
    metasurface cell. The round-5 writing **refused** such a voxel, and the round-6 review
    measured what that cost: the refusal fired on a T-junction of three *pairwise disjoint*
    boxes at 16 of 16 random sub-cell grid origins and not at the 1 the gate used, and under
    ``jax.jit`` -- where there is no value to read, so it could not fire at all -- the same
    voxels silently read 0.7500 against a true 1.0. A door that the grid origin decides, and
    that becomes a wrong answer under a transform, is not a door.

    What ships instead: the node brackets the fill and returns the end that is safe for its own
    operation, and **counts** the voxels it had to bracket. Two pillars placed exactly tangent
    to each other on a substrate is the case; it is answered, it never removes material that is
    there, and the count comes back through :class:`Approximations`. Moving one of them by a
    fifth of a cell makes the pairs disjoint again and the answer exact, with a count of zero.
    """
    h = 0.05
    pts = _voxels(16, h, (0.0, 0.0, 0.5))
    sub = Box((3.0, 3.0, 0.4), center=(0.0, 0.0, -0.2))
    touching = Union(sub, Cylinder(0.2, 0.5, 'z', center=(-0.2, 0.0, 0.25)),
                     Cylinder(0.2, 0.5, 'z', center=(0.2, 0.0, 0.25)))
    report = Approximations()
    fill = np.asarray(occupancy(touching, pts, h, report=report)[0])
    assert -1e-12 <= fill.min() and fill.max() <= 1 + 1e-12
    kids = np.stack([np.asarray(b.parts(pts, h)[0]) for b in touching.bodies], axis=0)
    assert float((kids.max(axis=0) - fill).max()) < 1e-12            # no hole in solid material
    assert report.voxels > 0 and 0.0 < report.width <= 1.0
    assert 'BOUND' in report.message() and 'Union(Box, Cylinder, Cylinder)' in report.message()
    # and a scene the rule answers exactly -- two flat boxes meeting flush, whose union IS one
    # box -- reports a count of exactly zero, so the count is not decoration
    seam = Union(Box((0.4, 0.4, 0.4), center=(-0.2, 0.0, 0.0)),
                 Box((0.4, 0.4, 0.4), center=(0.2, 0.0, 0.0)))
    quiet = Approximations()
    fill = np.asarray(occupancy(seam, _voxels(20, h, (0.0, 0.0, 0.0)), h, report=quiet)[0])
    assert -1e-12 <= fill.min() and fill.max() <= 1 + 1e-12
    assert quiet.voxels == 0 and quiet.width < 1e-9 and quiet.message() is None
    # and the report is the node's own, not the scene's: it names the body
    assert quiet.entries and quiet.entries[0].node == 'Union(Box, Box)'


def _spiky_prism(h):
    """A prism whose tangent half space at the NEAREST face does not contain it.

    The voxel is centred on the origin. The body's nearest boundary point is on a face at
    ``x = 0.51 h`` whose outward normal is exactly ``-x``, so the tangent test
    ``d >= (h/2)|n|_1`` reads ``0.51 h >= 0.5 h`` and calls the voxel EMPTY -- while a thin
    spike of the same polygon reaches ``(0.40 h, 0.40 h)``, which is **inside** the voxel. Every
    point of the spike is further from the centre than the face is (0.57 h against 0.51 h), so
    the signed distance is the face's and so is its normal. A convex body cannot be written like
    this, which is the whole of why the tangent test is sound for one and not for the other.
    """
    face = 0.51 * h
    outline = [(face, -0.30), (0.30, -0.30), (0.30, 0.30), (face, 0.30),
               (face, 0.36 * h), (0.40 * h, 0.40 * h), (face, 0.24 * h)]
    return Prism(np.asarray(outline), 0.60, 'z')


def test_a_non_convex_child_is_classified_on_the_side_that_cannot_lie():
    """Three promises a CSG node may make about a **convex** child and not about any other,
    and the bracket on what one pair shares -- each asserted where it is decided.

    A convex body lies inside its own tangent half space. Everything the rule reads off that
    half space is therefore a statement about the body: that the voxel misses it (``outside``),
    that the two half spaces share no volume (:func:`_bodies._no_common_volume`), and that equal
    volumes mean equal sets (``flat``). A non-convex :class:`Prism` -- the one non-convex body
    this package admits -- has none of that, so each of the three is either re-spelled on
    something that does hold (the circumradius, which needs only ``|d|`` at most the true
    distance) or switched off. Each is a gate mutant, and the first one is measurable:
    ``_spiky_prism``'s tangent test calls a voxel empty that the body fills, and dropping the
    child there leaves a **cut voxel with no interface at all** -- ``sum_c w_c = 0``, this
    package's own worst failure mode.

    The fourth assertion is the pair bracket itself: ``_cut.two_plane``'s volume, clamped into
    ``[max(0, F_a + F_b - 1), min(F_a, F_b)]``, which any two sets in one voxel obey.
    """
    h = 0.05
    spiky = _spiky_prism(h)
    assert not spiky.convex and spiky.has_exact_fill
    point = np.zeros((1, 3))
    d, n = (np.asarray(v) for v in spiky.surface(point))
    assert float(d[0]) > (h / 2) * float(np.abs(n[0]).sum())    # the tangent test says EMPTY
    assert float(d[0]) < np.sqrt(3) / 2 * h                     # the circumradius says CROSSING
    assert float(np.asarray(spiky.parts(point, h)[0])[0]) > 0   # and the body is really in there
    # so a node that believed the tangent test would report a cut voxel with no interface
    far = Box((0.2, 0.2, 0.6), center=(-0.6, 0.0, 0.0))
    fill, w = occupancy(Union(spiky, far), point, h)
    fill, w = float(np.asarray(fill)[0]), np.asarray(w)[0]
    assert 0 < fill < 1 and abs(w.sum() - 1.0) < 1e-12, (fill, w)

    # the other three are read where they are decided, over a family of voxels
    node = Union(spiky, Box((0.25, 0.25, 0.6), center=(0.18, 0.0, 0.0)),
                 Cylinder(0.12, 0.6, 'z', center=(-0.10, 0.20, 0.0)))
    pts = _voxels(14, h, (0.13, 0.37, 0.5))
    faces = [body.surface(pts) for body in node.bodies]
    child = [body.parts(pts, h, surface=face) for body, face in zip(node.bodies, faces)]
    state = node._state(faces, child, h, np)
    split, _, first = node._pair_state(state, h, np)
    crease = node._crease(state, first, h, np)
    corner = np.sqrt(3) / 2 * h
    # (1) the non-convex child's `outside` is the circumradius and not the tangent support
    assert np.array_equal(np.asarray(state[0].outside), np.asarray(state[0].d) >= corner)
    assert not np.array_equal(np.asarray(state[1].outside), np.asarray(state[1].d) >= corner)
    # (2) it is never called flat, however close its two fills come
    assert not bool(np.asarray(state[0].flat).any())
    assert bool(np.asarray(state[1].flat).any())            # a convex child is, and often
    # (3) no pair containing it ever claims to share no volume
    assert not bool(np.asarray(split[0, 1]).any()) and not bool(np.asarray(split[0, 2]).any())
    assert bool(np.asarray(split[1, 2]).any())
    assert crease.least is not None
    # (4) the pair bracket lies inside what the two children's own fills force, on a pair that
    # really does share the voxel -- a sunk pillar on a substrate, where the two-plane cut runs
    seen = 0
    for gap in (0.0, -1.0):
        for _, body in _device_scenes(h, gap).items():
            pts = _voxels(12, h, (0.23, 0.11, 0.5))
            faces = [one.surface(pts) for one in body.bodies]
            child = [one.parts(pts, h, surface=face) for one, face in zip(body.bodies, faces)]
            st = body._state(faces, child, h, np)
            _, _, first = body._pair_state(st, h, np)
            cr = body._crease(st, first, h, np)
            f_a = np.zeros(pts.shape[:-1])
            f_b = np.zeros(pts.shape[:-1])
            taken = np.zeros(pts.shape[:-1], dtype=bool)
            for (i, j), take in first.items():
                take = np.asarray(take)
                f_a = np.where(take, np.asarray(st[i].part[0]), f_a)
                f_b = np.where(take, np.asarray(st[j].part[0]), f_b)
                taken |= take
            seen += int(taken.sum())
            least, most = np.asarray(cr.least), np.asarray(cr.most)
            floor, roof = np.maximum(f_a + f_b - 1.0, 0.0), np.minimum(f_a, f_b)
            assert float((floor[taken] - least[taken]).max(initial=0.0)) <= 1e-15
            assert float((least[taken] - most[taken]).max(initial=0.0)) <= 1e-15
            assert float((most[taken] - roof[taken]).max(initial=0.0)) <= 1e-15
    assert seen > 200, seen


def test_wholly_inside_is_the_voxels_circumradius_and_not_its_tangent_planes_support():
    """``d <= -dx sqrt(3)/2`` proves a child fills the voxel; ``d <= -(dx/2)|n|_1`` does not.

    The second is the condition for the voxel to sit inside the child's tangent HALF SPACE, and
    a convex body is inside that half space, not the other way round -- so it proves nothing
    about the body. The difference is a real surface: a sphere of radius ``h`` whose centre is
    ``0.4 h`` below the voxel centre reads ``d = -0.6 h`` with ``n = +z``, so the half-space test
    (threshold ``-0.5 h``) calls the voxel FULL while the sphere's own surface passes through it
    at ``(0.5 h, 0, 0.466 h)`` -- on the voxel's own face. A node that believed it would report
    solid material with **no interface at all**, which is this package's worst failure mode.

    The circumradius needs only ``|d|`` to be at most the true distance to the surface, which is
    what every body here gives and what a nested node would have given too.
    """
    h = 0.05
    ball = Sphere(h, center=(0.0, 0.0, -0.4 * h))
    point = np.zeros((1, 3))
    d, n = (np.asarray(v) for v in ball.surface(point))
    assert float(d[0]) <= -(h / 2) * float(np.abs(n[0]).sum())     # the half space says FULL
    assert float(d[0]) > -np.sqrt(3) / 2 * h                       # the circumradius says CROSSING
    own_fill = float(np.asarray(ball.parts(point, h)[0])[0])
    assert 0 < own_fill < 1, own_fill              # the sphere does NOT fill the voxel
    # A child called wholly inside puts the node in its `solid` branch, which zeroes the
    # interface of EVERY child -- so the face that disappears is the one next to it. Here a box
    # whose own face runs through the voxel centre reports a full h^2 of it.
    wall = Box((0.4, 2.0, 2.0), center=(-0.2, 0.0, 0.0))
    assert float(np.asarray(wall.parts(point, h)[1])[0]) > 0.9 * h * h
    node = Union(ball, wall)
    area = float(np.asarray(node.parts(point, h)[1])[0])
    assert area > 0.9 * h * h, area                  # the wall's face survives, whole
    w = np.asarray(occupancy(node, point, h)[1])[0]
    assert abs(w.sum() - 1.0) < 1e-12 and abs(w[0] - 1.0) < 1e-12, w


def test_a_third_crossing_child_keeps_its_own_face():
    """A pair's buried area replaces the face of the two children in that pair and of no other.

    :meth:`_Combination._inside_face` **replaces** a child's whole area with what the two-plane
    cut proved lies inside its partner, so a crossing child that is in no such pair would be
    left with the zero that belongs to a child that has one -- no interface where there is one,
    which is this package's own worst failure mode and the one an earlier writing of this class
    shipped (108 of 5360 truly cut voxels of a pillar array). The gate is an
    :class:`Intersection` of three bodies whose surfaces all cross the same voxels.
    """
    h = 0.05
    nodes = [Intersection(Sphere(0.30), Box((0.34, 2.0, 2.0)), Box((2.0, 0.36, 2.0))),
             # and a UNION of three, which is where a child wrongly called WHOLLY INSIDE the
             # voxel shows: the node reports solid material and no interface where a surface is
             Union(Sphere(0.26, (-0.12, 0.0, 0.0)), Cylinder(0.14, 0.5, 'z', (0.14, 0.0, 0.0)),
                   Box((0.5, 0.18, 0.18), center=(0.0, 0.26, 0.0)))]
    live = 0
    for node, origin in [(n, o) for n in nodes for o in GRID_ORIGINS[:4]]:
        pts = _voxels(16, h, origin)
        fill, w = occupancy(node, pts, h)
        fill, w = np.asarray(fill), np.asarray(w)
        own = np.zeros(fill.shape)
        for child in node.bodies:
            own = np.maximum(own, np.asarray(child.parts(pts, h)[1]))
        # a voxel is TRULY cut when some child reports a live interface of its own, which is the
        # same criterion `test_a_node_never_deletes_an_interface_its_children_report` uses: a
        # voxel a surface merely passes near is not one the node owes an interface for
        cut = (fill > 1e-6) & (fill < 1 - 1e-6) & (own > 1e-12 * h * h)
        assert cut.sum() > 50, (node, origin)
        dead = cut & (w.sum(axis=-1) < 0.5)
        assert not dead.any(), (node, origin, int(dead.sum()))   # every cut voxel keeps a face
        live += int(cut.sum())
    assert live > 900, live
    # and, precisely: where the gate fires on ONE pair and a third child crosses the same voxel
    # in neither half of it, the node's own area still contains that child's whole face. It is
    # the same statement the loop above makes, read where the substitution happens, because a
    # node whose other two children happen to bury nothing would hide it behind a live w.
    # the two flat boxes first, so that THEY are the pair the cut is spent on and the sphere --
    # which is never flat, and so could never be half of a gated pair -- is the third child
    node = Intersection(Box((0.34, 2.0, 2.0)), Box((2.0, 0.36, 2.0)), Sphere(0.30))
    seen = 0
    for origin in GRID_ORIGINS[:4]:
        pts = _voxels(16, h, origin)
        faces = [one.surface(pts) for one in node.bodies]
        child = [one.parts(pts, h, surface=face) for one, face in zip(node.bodies, faces)]
        st = node._state(faces, child, h, np)
        _, _, first = node._pair_state(st, h, np)
        cr = node._crease(st, first, h, np)
        fill, area = (np.asarray(v) for v in node.parts(pts, h)[:2])
        for k in range(len(node.bodies)):
            own = np.asarray(child[k][1])
            pick = (np.asarray(cr.gate) & ~np.asarray(cr.chosen[k]) & np.asarray(st[k].cross)
                    & (fill > 1e-9) & (own > 1e-12 * h * h))
            seen += int(pick.sum())
            assert float((own[pick] - area[pick]).max(initial=0.0)) <= 1e-15, (origin, k)
    assert seen > 20, seen


def test_the_node_reports_what_it_approximated_and_says_nothing_when_it_did_not():
    """The interface C2 and C5 warn from: a count, a width, and the body's own name.

    It is the half of "no per-voxel refusal" that keeps the package honest -- a node that answers
    everything with a bound and says nothing about it is worse than one that refuses. Four things
    are held here: a count of **exactly zero** on a scene the rule answers exactly (so the count
    is not decoration); a positive count and width on one it does not; one entry per node **per
    periodic image**, because the ring calls ``parts`` once per image; and the behaviour under
    ``jax.jit``, where there is no number to read -- ``voxels`` and ``width`` come back ``None``
    and the two reductions are still on the entry as the backend's own arrays, which is what a
    jitted caller returns from its own function.
    """
    h = 0.05
    pts = _voxels(14, h, (0.31, 0.11, 0.5))
    sub = Box((3.0, 3.0, 0.4), center=(0.0, 0.0, -0.2))
    # lattice-aligned, so every cut voxel holds ONE flat face and the interval is closed
    exact = Union(Box((0.4, 0.4, 0.4), center=(-0.2, 0.0, 0.0)),
                  Box((0.4, 0.4, 0.4), center=(0.2, 0.0, 0.0)))
    bounded = Union(sub, Cylinder(0.16, 0.5, 'z', center=(-0.11, 0.0, 0.22)),
                    Cylinder(0.16, 0.5, 'z', center=(0.11, 0.0, 0.22)))
    quiet = Approximations()
    occupancy(exact, _voxels(20, h, (0.0, 0.0, 0.0)), h, report=quiet)
    assert quiet.voxels == 0 and quiet.width < 1e-9 and quiet.message() is None
    loud = Approximations()
    occupancy(bounded, pts, h, report=loud)
    assert loud.voxels > 0 and loud.width > 0
    assert str(loud.voxels) in loud.message()
    # one entry per image: a 3x3 ring of a periodic axis calls parts nine times
    ring = Approximations()
    occupancy(bounded, pts, h, period=(1.0, 1.0, None), report=ring)
    assert len(ring.entries) == 9 and len(loud.entries) == 1
    assert ring.voxels >= loud.voxels
    # a value that is not a collector is refused by name, like every other argument
    with pytest.raises(ValueError, match='occupancy\\(report=...\\)'):
        occupancy(bounded, pts, h, report=[])
    # under jax.jit there is no number; the arrays are still there
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    traced = Approximations()

    def run(p):
        occupancy(bounded, p, h, report=traced)
        entry = traced.entries[-1]
        return entry.counted, entry.widest

    counted, widest = jax.jit(run)(jnp.asarray(pts))
    assert traced.entries[-1].voxels is None and traced.entries[-1].width is None
    assert int(counted) == loud.voxels and abs(float(widest) - loud.width) < 1e-12


def test_each_node_kind_takes_the_side_that_cannot_lie_about_material():
    """The sign argument, per operation, against a membership oracle that shares no line with it.

    ``Union`` and ``Difference`` must never **remove** material that is there; ``Intersection``
    must never **add** material that is not. Truth is ``body.surface(points)[0] <= 0``, stratified
    over the voxel -- the composed signed distance's SIGN, which is exact for every node built
    from convex leaves, and which touches none of ``parts``, ``_cut`` or ``_resolved``.

    Measured over 12^3 = 1728 stratified samples per voxel (standard error 1.2e-2), 14^3 voxels
    at ``h`` = 0.05, three grid origins, 20 scenes (2026-09-21). Worst violation of the safe side
    over every one of them: **2.3e-03**, i.e. inside the sampling error and consistent with
    exactly zero. The other tail -- what being on that side costs -- runs to 0.19 of a voxel on
    an ``Intersection`` of three slabs, which is the corner where three faces meet in one voxel
    and is the widest case in the package. The one this replaces removed **0.0206** of a voxel
    over 72 voxels of a slab with two overlapping vias, one-signed: the shared volume of the two
    removed children entered ``1 - F_1 - F_2 + shared`` with the lower bound where the sign calls
    for the upper one.
    """
    h = 0.05
    m = 8
    error = 0.5 / m ** 1.5
    slab = Box((1.6, 1.6, 0.6))
    sub = Box((3.0, 3.0, 0.4), center=(0.0, 0.0, -0.2))
    cases = [
        ('hi', Union(sub, Cylinder(0.16, 0.6, 'z', center=(-0.11, 0.0, 0.22)),
                     Cylinder(0.16, 0.6, 'z', center=(0.11, 0.0, 0.22)))),
        ('hi', Union(sub, Box((0.18, 0.18, 0.5), center=(-0.17, 0.0, 0.22)),
                     Box((0.18, 0.18, 0.5), center=(0.0, 0.0, 0.22)),
                     Box((0.18, 0.18, 0.5), center=(0.17, 0.0, 0.22)))),
        ('lo', Intersection(Sphere(0.4), Box((0.5, 2.0, 2.0)), Box((2.0, 0.55, 2.0)))),
        ('lo', Intersection(Box((0.6, 2.0, 2.0)), Box((2.0, 0.6, 2.0)), Box((2.0, 2.0, 0.6)))),
        ('lo', Intersection(Cylinder(0.3, 0.6, 'z'), Sphere(0.35, (0.1, 0.0, 0.05)))),
        # the C2 review's own repro: two overlapping bodies REMOVED from a slab
        ('hi', Difference(slab, Cylinder(0.18, 1.0, 'z', center=(-0.10, 0.0, 0.0)),
                          Cylinder(0.18, 1.0, 'z', center=(0.10, 0.0, 0.0)))),
        ('hi', Difference(slab, Sphere(0.20, (-0.12, 0.0, 0.0)), Sphere(0.20, (0.12, 0.0, 0.0)))),
        ('hi', Difference(slab, Box((0.6, 0.15, 0.3), center=(0.1, 0.0, 0.1)),
                          Box((0.15, 0.6, 0.3), center=(-0.1, 0.1, 0.1)))),
    ]
    worst = 0.0
    for side, node in cases:
        for origin in GRID_ORIGINS[:3]:
            pts = _voxels(12, h, origin)
            fill = np.asarray(occupancy(node, pts, h)[0])
            truth = _sampled_fill(node, pts, h, m=m, seed=11)
            unsafe = (truth - fill) if side == 'hi' else (fill - truth)
            worst = max(worst, float(unsafe.max()))
    assert worst < 3 * error, (worst, error)


def test_the_exact_two_plane_cut_is_what_an_independent_oracle_says():
    """``_cut.two_plane`` against brute-force vertex enumeration and a convex hull.

    The oracle shares no line with it: it forms all 56 triples of the eight planes (six box
    faces and the two cuts), solves each 3x3 system, keeps the feasible vertices and takes
    ``scipy.spatial.ConvexHull`` of them, then of the subset on each cut plane for that face's
    area. Four families, and the degenerate ones are the point -- a device joint is exactly
    anti-parallel and exactly flush, which is where the general form's ``sqrt(1 - mu^2)``
    denominator goes to zero and a separate closed form takes over.

        family                                        worst |dV|   worst |dA|/dx^2
        600 random pairs, both cutting the box         4.5e-13        7.8e-13
        300 axis-aligned pairs (faces, slabs, corners) 9.8e-13        5.0e-13
        300 nearly anti-parallel, nearly flush joints  4.3e-13        1.3e-09
        300 with one plane grazing a corner            5.3e-17        1.0e-14

    and, with one plane pushed clear of the voxel, it reproduces :func:`_cut._parts` to 5.6e-16
    in the fill and 1.1e-16 in the area -- which is what says the two forms are one function.
    """
    hull = pytest.importorskip('scipy.spatial').ConvexHull
    import itertools
    rng = np.random.default_rng(20260921)
    dx = 0.37
    half = dx / 2

    def oracle(d_a, n_a, d_b, n_b):
        rows = [(np.eye(3)[k] * s, half) for k in range(3) for s in (1.0, -1.0)]
        rows += [(np.asarray(n_a), -d_a), (np.asarray(n_b), -d_b)]
        a = np.stack([r[0] for r in rows])
        c = np.asarray([r[1] for r in rows])
        pts = []
        for tri in itertools.combinations(range(8), 3):
            m = a[list(tri)]
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            x = np.linalg.solve(m, c[list(tri)])
            if np.all(a @ x <= c + 1e-11 * dx):
                pts.append(x)
        pts = np.unique(np.round(np.asarray(pts) / dx, 12), axis=0) * dx if pts else np.zeros((0,))
        if len(pts) < 4:
            return 0.0, 0.0, 0.0
        out = [hull(pts).volume / dx ** 3]
        for idx in (6, 7):
            on = pts[np.abs(pts @ a[idx] - c[idx]) < 1e-8 * dx]
            if len(on) < 3:
                out.append(0.0)
                continue
            e1 = np.cross(a[idx], [1.0, 0.0, 0.0])
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(a[idx], [0.0, 1.0, 0.0])
            e1 = e1 / np.linalg.norm(e1)
            flat = np.stack([on @ e1, on @ np.cross(a[idx], e1)], axis=-1)
            out.append(hull(flat).volume)
        return tuple(out)

    def unit():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    families = {}
    families['random'] = [(rng.uniform(-1.7 * half, 1.7 * half), unit(),
                           rng.uniform(-1.7 * half, 1.7 * half), unit()) for _ in range(60)]
    axis = np.eye(3)
    families['axis aligned'] = [
        (rng.uniform(-half, half), rng.choice([-1.0, 1.0]) * axis[rng.integers(0, 3)],
         rng.uniform(-half, half), rng.choice([-1.0, 1.0]) * axis[rng.integers(0, 3)])
        for _ in range(60)]
    joints = []
    for _ in range(60):
        n_a = unit()
        perp = np.cross(n_a, unit())
        perp = perp / np.linalg.norm(perp)
        theta = rng.uniform(0, 1e-3)
        d_a = rng.uniform(-half, half)
        joints.append((d_a, n_a, -d_a + rng.normal(scale=1e-3 * dx),
                       np.cos(theta) * (-n_a) + np.sin(theta) * perp))
    families['nearly flush joints'] = joints
    # EXACTLY anti-parallel and EXACTLY parallel: the pairs the general form cannot answer,
    # because its 1/sqrt(1 - mu^2) is 1/0 there and a separate closed form takes over. A device
    # joint is exactly this, and so is every layer stack, so the branch is the common case and
    # not the corner -- including the part of it that decides whether the two faces are buried
    # in each other's material (a slab) or are two interfaces with a gap between them.
    exact_pairs = []
    for _ in range(60):
        n_a = unit()
        d_a = rng.uniform(-half, half)
        exact_pairs.append((d_a, n_a, -d_a + rng.uniform(-half, half), -n_a))
    families['exactly anti-parallel'] = exact_pairs
    families['exactly parallel'] = [(rng.uniform(-half, half), n, rng.uniform(-half, half), n)
                                    for n in (unit() for _ in range(60))]
    for name, cases in families.items():
        worst_v = worst_a = 0.0
        for d_a, n_a, d_b, n_b in cases:
            da, db = np.asarray([d_a]), np.asarray([d_b])
            na, nb = np.asarray([n_a]), np.asarray([n_b])
            got = two_plane(da, na, db, nb, dx, _parts(da, na, dx), _parts(db, nb, dx))
            want = oracle(d_a, n_a, d_b, n_b)
            worst_v = max(worst_v, abs(float(got[0][0]) - want[0]))
            worst_a = max(worst_a, abs(float(got[1][0]) - want[1]) / dx ** 2,
                          abs(float(got[2][0]) - want[2]) / dx ** 2)
        assert worst_v < 1e-11, (name, worst_v)
        assert worst_a < 1e-8, (name, worst_a)
    # one plane clear of the voxel: the two-plane form IS the one-plane form
    worst = 0.0
    for _ in range(200):
        n_a, n_b = unit(), unit()
        d_a = rng.uniform(-1.8 * half, 1.8 * half)
        da, na, nb = np.asarray([d_a]), np.asarray([n_a]), np.asarray([n_b])
        one = _parts(da, na, dx)
        got = two_plane(da, na, np.asarray([-10 * dx]), nb, dx, one, _parts(np.asarray([-10 * dx]),
                                                                           nb, dx))
        worst = max(worst, abs(float(got[0][0]) - float(one[0][0])),
                    abs(float(got[1][0]) - float(one[1][0])) / dx ** 2, abs(float(got[2][0])))
    assert worst < 1e-14, worst


def test_the_two_plane_cut_obeys_its_own_bounds_at_every_coordinate_scale():
    """``0 <= V <= min(f_a, f_b)``, ``V >= max(0, f_a + f_b - 1)``, ``A_i <= a_i`` -- and scale.

    The three inequalities are what the CSG rule's clamp rests on, so they are asserted on the
    kernel itself rather than inferred from a node. The scale row is this file's standing
    question of every predicate: a geometric test compared against an absolute floor is a test
    about the coordinate unit, and this kernel is bit-identical from a micrometre scene to a
    metre one.
    """
    rng = np.random.default_rng(11)
    n_a = rng.normal(size=(2000, 3))
    n_a = n_a / np.linalg.norm(n_a, axis=-1, keepdims=True)
    n_b = rng.normal(size=(2000, 3))
    n_b = n_b / np.linalg.norm(n_b, axis=-1, keepdims=True)
    unit_a, unit_b = rng.uniform(-0.37, 0.37, 2000), rng.uniform(-0.37, 0.37, 2000)
    reference = None
    for scale in (1e-6, 1e-3, 1.0, 3.29e5, 1e9):
        dx = 0.37 * scale
        d_a, d_b = unit_a * scale, unit_b * scale
        one_a, one_b = _parts(d_a, n_a, dx), _parts(d_b, n_b, dx)
        shared, area_a, area_b = two_plane(d_a, n_a, d_b, n_b, dx, one_a, one_b)
        assert shared.min() > -1e-13, (scale, shared.min())
        assert float(np.maximum(shared - np.minimum(one_a[0], one_b[0]), 0).max()) < 1e-13, scale
        assert float(np.maximum(np.maximum(one_a[0] + one_b[0] - 1, 0) - shared, 0).max()) < 1e-13
        assert float(np.maximum(area_a - one_a[1], 0).max()) / dx ** 2 < 1e-13, scale
        assert float(np.maximum(area_b - one_b[1], 0).max()) / dx ** 2 < 1e-13, scale
        if reference is None:
            reference = np.asarray(shared)
        else:
            assert np.abs(np.asarray(shared) - reference).max() < 1e-15, scale


def test_the_doors_this_round_adds_refuse_a_typo_by_name():
    """Four more doors, each closing a case that came back as an incidental exception.

    The repository's rule is that a typo is refused on every door and by name, and the round-5
    review found the round's own doors had holes in exactly the places the rule names. Measured
    on the kernel before each of these (2026-09-21):

        spelling                                    was                       is now
        occupancy(period=(0, None, None))           fill 3.0000, silently     refused
        occupancy(period=(-1., None, None))         accepted and ignored      refused
        occupancy(rings=(-1, 0, 0))                 bare TypeError from range refused
        occupancy(rings=(1.5, 0, 0))                bare TypeError            refused
        occupancy(..., dx=True)                     solved at dx = 1.0        refused
        Sphere(radius=True)                         a unit sphere             refused
        Box(size=(True, True, True))                a unit cube               refused
        occupancy(points=np.zeros((1, 2)))          IndexError from inside    refused
        occupancy(points=np.zeros((1, 4)))          broadcast ValueError      refused
        occupancy(points=complex)                   TypeError from arctan2    refused
        occupancy(body='x')                         AttributeError .leaves    refused
        Union(Sphere(0.2), 'x')                     AttributeError .leaves    refused
        occupancy(..., dx=np.array([h, h]))         broadcast ValueError      refused

    Each refusal names the call the reader made: ``occupancy(period[x]=...)`` and not
    ``image_shifts(period[x]=...)``, which is what the pattern below holds, because
    ``image_shifts`` doors the same value one level down and a door that answers in the callee's
    name leaves the caller's own door untested.

    ``period = 0`` is the one worth reading twice: it stacks all three images of the ring on the
    same point, so the fill comes back **exactly tripled** with ``sum_c w_c = 1`` in every voxel.
    Both free invariants are satisfied by the artefact, which is the class of failure every door
    in this file exists for.
    """
    point = np.zeros((1, 3))
    cases = {
        'period zero': (lambda: occupancy(Sphere(0.2), point, 0.1, period=(0, None, None)),
                        r'occupancy\(period\[x\]'),
        'period negative': (lambda: occupancy(Sphere(0.2), point, 0.1, period=(-1., None, None)),
                            r'occupancy\(period\[x\]'),
        'rings negative': (lambda: occupancy(Sphere(0.2), point, 0.1, rings=(-1, 0, 0)), 'rings'),
        'rings fractional': (lambda: occupancy(Sphere(0.2), point, 0.1, rings=(1.5, 0, 0)),
                             'rings'),
        'dx a bool': (lambda: occupancy(Sphere(0.2), point, True), 'this is a bool'),
        'radius a bool': (lambda: Sphere(radius=True), 'this is a bool'),
        'size a bool': (lambda: Box(size=(True, True, True)), 'this is a bool'),
        'height a bool': (lambda: Cylinder(0.2, True), 'this is a bool'),
        'points too narrow': (lambda: occupancy(Sphere(0.2), np.zeros((1, 2)), 0.1), 'points'),
        'points too wide': (lambda: occupancy(Sphere(0.2), np.zeros((1, 4)), 0.1), 'points'),
        'points complex': (lambda: occupancy(Sphere(0.2), np.zeros((1, 3), dtype=complex), 0.1),
                           'complex'),
        'body not a body': (lambda: occupancy('x', point, 0.1), 'a Body'),
        'child not a body': (lambda: Union(Sphere(0.2), 'x'), 'every child is a Body'),
        'child a number': (lambda: Intersection(Sphere(0.2), 3.0), 'every child is a Body'),
        'dx an array': (lambda: occupancy(Sphere(0.2), point, np.array([0.05, 0.05])),
                        'one number'),
    }
    for name, (call, pattern) in cases.items():
        with pytest.raises(ValueError, match=pattern):
            call()
    # and the artefact the period door stops, so the number in the docstring is this test's
    assert float(occupancy(Sphere(0.2), point, 0.1, period=(1., 1., 1.))[0][0]) == 1.0


def test_a_geometry_value_that_is_not_a_real_number_is_refused_by_name():
    """A complex parameter was checked on its real part and then used whole.

    ``readable`` asked NumPy for ``dtype=float``, which silently discards a complex value's
    imaginary part, and ``_as`` then handed the **original** to the kernel -- so the door and the
    computation saw two different bodies. Measured before this (``h = 0.05``, 16^3):

        Prism(vertices = HEX + 0.03j)      ACCEPTED, volume 0.03417706 against 0.03095088 (+10.4%)
        Box(size = [0.34+0.2j, .22, .29])  ACCEPTED, fill dtype complex128
        Cylinder(radius = 0.21+0.3j)       ACCEPTED, fill dtype complex128
        Sphere(center = [0.01+0.5j, 0, 0]) ACCEPTED, fill dtype complex128
        Sphere(radius = -0.26+0.9j)        refused -- but only because its REAL part is negative

    The package already refuses a complex eps by name; a length, a coordinate and an angle are
    real for the same reason. The second half of this test is the same door's other branch: a
    value that is not a number at all is refused by name rather than deferred to an obscure
    failure inside ``array_namespace`` (memory: typos are refused on every door).
    """
    match = 'a real number, and this is complex'
    with pytest.raises(ValueError, match=match):
        Sphere(0.26 + 0.1j)
    with pytest.raises(ValueError, match=match):
        Sphere(-0.26 + 0.9j)                  # refused for being complex, not for its real part
    with pytest.raises(ValueError, match=match):
        Sphere(0.26, (0.01 + 0.5j, 0., 0.))
    with pytest.raises(ValueError, match=match):
        Cylinder(0.21 + 0.3j, 0.3, 'z')
    with pytest.raises(ValueError, match=match):
        Cylinder(0.21, 0.3, 'z', (0., 0., 0.), 0.4 + 0.1j)
    with pytest.raises(ValueError, match=match):
        Box((0.34 + 0.2j, 0.22, 0.29))
    with pytest.raises(ValueError, match=match):
        Prism(tuple((u + 0.03j, v) for u, v in HEX), 0.33, 'z')
    with pytest.raises(ValueError, match=match):
        Prism(HEX, 0.33 + 0.01j, 'z')
    # a complex value with a zero imaginary part is still complex: the dtype is what reaches the
    # kernel, and `Sphere(0.26+0j)` used to die inside as "ufunc 'arctan2' not supported"
    with pytest.raises(ValueError, match=match):
        Sphere(0.26 + 0j)
    # not a number at all
    not_a_number = 'a real number, and this is not a number at all'
    for rubbish in ('abc', {'r': 0.2}, object()):
        with pytest.raises(ValueError, match=not_a_number):
            Sphere(rubbish)
    with pytest.raises(ValueError, match=not_a_number):
        Cylinder(0.2, 0.3, 'z', ('a', 'b', 'c'))
    # and every legal spelling still passes, on every backend's own array type
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    for radius in (0.2, np.float64(0.2), np.asarray(0.2), jnp.asarray(0.2), torch.tensor(0.2)):
        assert Sphere(radius) is not None


def test_a_numpy_parameter_is_the_same_number_on_every_backend():
    """``Prism(np.asarray(HEX), ...)`` is what a user writes, and Torch used to raise on it.

    ``_as`` returned a value untouched when it had both ``dtype`` and ``device``, which under
    numpy 2.5.1 an ``ndarray`` has -- so a NumPy-array parameter was never converted and then met
    a Torch tensor. Measured before the fix:

        parameter written as     numpy          jax            torch
        Sphere(radius) python    0.07362218     0.07362218     0.07362218
        Sphere(radius) numpy     0.07362218     0.07362218     TypeError
        Box(size) numpy          0.02169200     0.02169200     TypeError
        Prism(vertices) numpy    0.03095088     0.03095088     TypeError
        Sphere(center) numpy     0.07362218     0.07362218     0.07362218   <- and this worked

    ``TypeError: unsupported operand type(s) for *: 'numpy.ndarray' and 'Tensor'``, deep inside
    ``_vertices``, naming no door -- and inconsistent inside one body. Comparing namespaces
    instead of attributes converts a foreign concrete value and leaves a traced one alone.
    """
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    h = 0.08
    pts = grid(8, -0.32, h)
    cases = {
        'Sphere(radius)': lambda box: Sphere(box(0.22), (0.01, 0., 0.)),
        'Sphere(center)': lambda box: Sphere(0.22, box((0.01, 0., 0.))),
        'Box(size)': lambda box: Box(box((0.34, 0.22, 0.29)), (0.01, 0., 0.), 0.3),
        'Cylinder(radius)': lambda box: Cylinder(box(0.19), 0.3, 'z', (0.01, 0., 0.)),
        'Prism(vertices)': lambda box: Prism(box(HEX), 0.33, 'z', (0.01, 0., 0.)),
        'Prism(angle)': lambda box: Prism(HEX, 0.33, 'z', (0.01, 0., 0.), box(0.4)),
    }
    for name, make in cases.items():
        answers = []
        for plain in (lambda v: v, np.asarray):
            for lib in (np, jnp, torch):
                got = occupancy(make(plain), lib.asarray(pts, dtype=lib.float64), h)[0]
                answers.append(float(np.asarray(got).sum()))
        assert max(answers) - min(answers) < 1e-12, (name, answers)
        assert answers[0] > 0.0, name


def test_the_liveness_of_w_is_free_of_the_coordinate_scale():
    """``area > 0`` is a predicate about the coordinate unit; ``area > tiny dx^2`` is not.

    A randomised cross-backend fuzz in the C1 review found one row that split: a star prism at
    coordinate scale 3.29e+05 with ``h = 3.23e+04`` gave ``area = 0`` on NumPy and
    ``area = 1.7978064238510632e-09`` on JAX -- a relative 1.7e-18 of the voxel face 1.04e+09 --
    so ``live = area > 0`` fell on opposite sides and ``w`` swapped from ``(0, 0, 0)`` to a full
    ``(1, 0, 0)`` between two backends on one legal input. The package's headline rule is that an
    input that reaches the kernel must never give three different answers, and it carries no
    carve-out for ``w``.

    The fix is the one ``_TURN_SINE`` had last round: compare against the voxel's own face rather
    than against zero (``_cut._LIVE_AREA``). Measured here over the same body at four coordinate
    scales spanning fifteen decades, worst ``|dw|`` between NumPy and each of JAX and Torch:
    5.6e-16, 2.2e-16, 7.8e-16, 1.1e-16 -- no swap anywhere.
    """
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    outline = star(6, 0.42, 0.10)
    for scale in (1e-6, 1.0, 3.29e5, 1e9):
        body = Prism(tuple((u * scale, v * scale) for u, v in outline), 0.5 * scale, 'z',
                     (0.01 * scale, -0.02 * scale, 0.0))
        h = 0.11 * scale
        n = 9
        c = -n * h / 2 + (np.arange(n) + 0.5) * h
        pts = np.stack(np.meshgrid(c, c, c, indexing='ij'), axis=-1)
        answers = [tuple(np.asarray(v) for v in occupancy(body, lib.asarray(pts, dtype=lib.float64),
                                                          h))
                   for lib in (np, jnp, torch)]
        for other in answers[1:]:
            assert np.abs(answers[0][0] - other[0]).max() < 1e-13, scale
            assert np.abs(answers[0][1] - other[1]).max() < 1e-13, scale
        # and the body is really there: a dead w and a live one both occur at every scale
        total = answers[0][1].sum(-1)
        assert (total == 0).sum() > 10 and (total > 0).sum() > 10, scale


def _sampled_cylinder_moment(cyl, pts, h, nth=20001):
    radius, height = cyl.radius, cyl.height
    cx, cy, cz = cyl.center
    th = (np.arange(nth) + 0.5) * 2 * np.pi / nth
    x, y = cx + radius * np.cos(th), cy + radius * np.sin(th)
    ds = radius * 2 * np.pi / nth
    lo, hi = pts - h / 2, pts + h / 2
    live = ((x[None, :] > lo[:, 0, None]) & (x[None, :] < hi[:, 0, None])
            & (y[None, :] > lo[:, 1, None]) & (y[None, :] < hi[:, 1, None]))
    along = np.maximum(np.minimum(hi[:, 2], cz + height / 2)
                       - np.maximum(lo[:, 2], cz - height / 2), 0.)
    arc = live.sum(-1) * ds * along
    mx = (live * np.cos(th) ** 2).sum(-1) * ds * along
    my = (live * np.sin(th) ** 2).sum(-1) * ds * along
    disc = clipped_disc_area(cx, cy, radius, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], np)
    caps = ((cz - height / 2 > lo[:, 2]) & (cz - height / 2 < hi[:, 2])).astype(float) \
        + ((cz + height / 2 > lo[:, 2]) & (cz + height / 2 < hi[:, 2])).astype(float)
    cap = caps * disc
    area = arc + cap
    safe = np.where(area > 0, area, 1.0)
    return area, np.stack([mx / safe, my / safe, cap / safe], axis=-1)


# ----------------------------------------------------------------------------------------
# the distance itself, and the normal
# ----------------------------------------------------------------------------------------

CENTRE_D = (0.05, -0.02, 0.03)


def _sphere_distance(pts):
    return np.linalg.norm(pts - np.asarray(CENTRE_D), axis=-1) - 0.31


def _box_distance(pts, size=(0.4, 0.28, 0.33), angle=0.6):
    """The textbook rotated-box distance, written straight -- not through the polygon walk."""
    co, si = np.cos(angle), np.sin(angle)
    v = pts - np.asarray(CENTRE_D)
    local = np.stack([v[..., 0] * co + v[..., 1] * si, -v[..., 0] * si + v[..., 1] * co,
                      v[..., 2]], axis=-1)
    q = np.abs(local) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(q, 0.), axis=-1) + np.minimum(q.max(-1), 0.)


def _cylinder_distance(pts, radius=0.24, height=0.4):
    """The same, for a disc extruded along x."""
    v = pts - np.asarray(CENTRE_D)
    dr = np.hypot(v[..., 1], v[..., 2]) - radius
    da = np.abs(v[..., 0]) - height / 2
    return np.minimum(np.maximum(dr, da), 0.) + np.hypot(np.maximum(dr, 0.), np.maximum(da, 0.))


def _prism_surface(angle=0.2, height=0.4, per_face=90000):
    """Points on a hexagonal prism's surface: six side rectangles and two fan-cut caps."""
    rng = np.random.default_rng(5)
    co, si = np.cos(angle), np.sin(angle)
    verts = np.asarray(HEX)
    out = []
    for i in range(6):
        a, b = verts[i], verts[(i + 1) % 6]
        t = rng.uniform(0, 1, per_face)[:, None]
        z = rng.uniform(-height / 2, height / 2, per_face)
        edge = a + t * (b - a)
        out.append(np.stack([edge[:, 0], edge[:, 1], z], axis=-1))
    for sign in (-1., 1.):
        for i in range(1, 5):
            a, b, c = verts[0], verts[i], verts[i + 1]
            u, v = rng.uniform(0, 1, (2, per_face // 4))
            flip = u + v > 1
            u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
            face = a + u[:, None] * (b - a) + v[:, None] * (c - a)
            out.append(np.stack([face[:, 0], face[:, 1],
                                 np.full(len(face), sign * height / 2)], axis=-1))
    local = np.concatenate(out)
    return np.stack([local[:, 0] * co - local[:, 1] * si + CENTRE_D[0],
                     local[:, 0] * si + local[:, 1] * co + CENTRE_D[1],
                     local[:, 2] + CENTRE_D[2]], axis=-1)


def test_the_signed_distance_is_the_distance():
    """Against closed forms written independently of the walk that computes them.

    This is the oracle that caught the one real bug in this commit: a transverse centre
    subtracted twice put every extruded body somewhere else, and the volumes, the invariants
    and ``jax.grad`` against a finite difference all still agreed, because they agreed with
    each other and with a body that was simply elsewhere. A distance is the one thing that
    cannot be self-consistent and wrong.

    The box and the cylinder go through the polygon/disc machinery in the kernel and through
    the textbook two-line formula here. The prism has no such formula, so its oracle is the
    minimum distance to 720k points laid analytically on its own surface: mean spacing 9.5e-4,
    worst disagreement measured 5.1e-4 and median 2.9e-5, held to 2e-3.

    **The CSG rows are the C1 review's finding**, and they are the ones with teeth. Before them
    nothing in this file read a ``Union``, ``Intersection`` or ``Difference``'s normal at all:
    the CSG bodies appeared only in invariant tests (``0 <= fill <= 1``, ``sum_c w_c == 1``,
    ``w = 0`` where the fill is decided), every one of which a wrong-but-unit normal satisfies.
    A mutant returning ``self.bodies[0].surface(points)[1]`` from every node passed the whole
    gate while moving ``w`` by 0.8783 in 616 of 13824 voxels -- for the one body class that has
    **no** exact form and therefore rests entirely on that normal. Their distance is written
    here a second way, as ``min`` / ``max`` / ``max`` of the negation over closed forms this
    file already owns, and their normal is held against the central difference of it.

    That is also the only thing holding ``Difference``'s ``-other_n``: ``_cut.cut_parts`` reads
    ``n`` as ``|n| dx`` and ``n n^T``, so a reversed normal is invisible everywhere else. A
    mutant returning ``other_n`` there passed the gate before this row and fails it now.

    **(M)** Restoring that double subtraction (``q = points - centre`` in ``_Extrusion._split``
    while ``_vertices`` stays absolute) fails this and four other tests.
    """
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.6, 0.6, size=(600, 3))
    prism = Prism(HEX, 0.4, 'z', CENTRE_D, 0.2)
    samples = _prism_surface()
    # Ten points at a time. In one broadcast the (600, 720k, 3) difference and its square are
    # 10.4 GB each and both alive at once: 21.1 GB peak, measured, which ran the 32 GB desktop
    # out of memory under ``-n 3``. Each row is the same arithmetic either way (bit-identical).
    brute = np.concatenate([np.sqrt(((p[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(-1)
                            for p in np.split(pts, 60)])
    ball = _sphere_distance(pts)
    brick = _box_distance(pts)
    rod = _cylinder_distance(pts)
    cases = ((Sphere(0.31, CENTRE_D), ball, 1e-15),
             (Box((0.4, 0.28, 0.33), CENTRE_D, 0.6), brick, 1e-14),
             (Cylinder(0.24, 0.4, 'x', CENTRE_D), rod, 1e-14),
             (prism, np.sign(prism.surface(pts)[0]) * brute, 2e-3),
             (Union(Sphere(0.31, CENTRE_D), Box((0.4, 0.28, 0.33), CENTRE_D, 0.6)),
              np.minimum(ball, brick), 1e-14),
             (Intersection(Sphere(0.31, CENTRE_D), Cylinder(0.24, 0.4, 'x', CENTRE_D)),
              np.maximum(ball, rod), 1e-14),
             (Difference(Box((0.4, 0.28, 0.33), CENTRE_D, 0.6), Sphere(0.31, CENTRE_D)),
              np.maximum(brick, -ball), 1e-14),
             (Difference(Sphere(0.31, CENTRE_D), Cylinder(0.24, 0.4, 'x', CENTRE_D),
                         Box((0.4, 0.28, 0.33), CENTRE_D, 0.6)),
              np.maximum(np.maximum(ball, -rod), -brick), 1e-14))
    for body, want, tol in cases:
        d, n = body.surface(pts)
        assert np.abs(d - want).max() < tol, (body, np.abs(d - want).max())
        assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-14, body
        # the normal is the gradient: a central difference of d must reproduce it
        fd = np.stack([(body.surface(pts + e * 1e-6)[0] - body.surface(pts - e * 1e-6)[0]) / 2e-6
                       for e in np.eye(3)], axis=-1)
        keep = np.abs(np.linalg.norm(fd, axis=-1) - 1) < 1e-4       # off the medial axis
        assert keep.sum() > 250, body
        assert np.abs(fd - n)[keep].max() < 1e-5, body


def test_a_normal_is_a_unit_vector_even_where_the_gradient_is_not():
    """At a sphere's centre, on a box's medial axis and on a cylinder's own axis."""
    for body, pts in ((Sphere(0.3, (0., 0., 0.)), np.zeros((1, 3))),
                      (Box((0.4, 0.4, 0.4), (0., 0., 0.), 0.3), np.zeros((1, 3))),
                      (Cylinder(0.2, 0.4, 'z', (0., 0., 0.)),
                       np.stack([np.zeros(5), np.zeros(5), np.linspace(-0.1, 0.1, 5)], axis=-1)),
                      (Prism(HEX, 0.4, 'z', (0., 0., 0.), 0.), np.zeros((1, 3)))):
        d, n = body.surface(pts)
        assert np.all(np.isfinite(d))
        assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-14, body


# ----------------------------------------------------------------------------------------
# translation, and the periodic image ring
# ----------------------------------------------------------------------------------------

def test_a_one_cell_translation_of_the_whole_scene_rolls_the_fill_bitwise():
    """Not "to 1e-14": the same bits. Powers of two make every coordinate exact.

    A rule that is right to a tolerance under translation is a rule with a preferred origin,
    and on a periodic grid that shows up as a fill that depends on where the user put the cell
    boundaries (memory: the NVF raster read the torus flat).
    """
    h = 2.0 ** -5
    pts = grid(20, -0.3125, h)
    for body, shifted in ((Sphere(0.25, (0.0625, -0.03125, 0.09375)),
                           Sphere(0.25, (0.0625 + h, -0.03125, 0.09375))),
                          (Box((0.375, 0.25, 0.3125), (0.0625, -0.03125, 0.09375), 0.0),
                           Box((0.375, 0.25, 0.3125), (0.0625 + h, -0.03125, 0.09375), 0.0)),
                          (Cylinder(0.1875, 0.3125, 'z', (0.0625, -0.03125, 0.09375)),
                           Cylinder(0.1875, 0.3125, 'z', (0.0625 + h, -0.03125, 0.09375)))):
        a, wa = occupancy(body, pts, h)
        b, wb = occupancy(shifted, pts, h)
        assert (b[1:] == a[:-1]).all(), body
        assert (wb[1:] == wa[:-1]).all(), body
        assert 0.0 < a.mean() < 1.0


def test_a_corner_centred_sphere_comes_back_whole_through_the_ring():
    """``4 pi R^3/3`` through the 3x3x3 ring -- now to machine precision, not to a bias.

    Before ``Sphere.exact_fill`` this test could only assert the plane's own ``+0.250 (h/R)^2``
    over-fill and that it was positive. With the exact ball clip the ring has nothing left to
    hide behind: eight wrapped lobes reassemble into the analytic volume to 4.4e-16 relative.

    **(M)** Dropping the ring (``image_shifts`` returning ``[(0,0,0)]``) loses the seven
    wrapped lobes: the same scene reads -87.45% of the volume.
    """
    radius = 0.25
    h = radius / 8
    n = int(round(1.0 / h))
    pts = grid(n, 0.0, h)
    body = Sphere(radius, (0., 0., 0.))
    exact = 4 / 3 * np.pi * radius ** 3
    with_ring = occupancy(body, pts, h, period=(1.0, 1.0, 1.0))[0].sum() * h ** 3
    without = occupancy(body, pts, h)[0].sum() * h ** 3
    assert abs(with_ring / exact - 1) < 1e-14
    assert without / exact - 1 < -0.8
    # the plane, on the same scene, is the bias this replaced -- still positive, still (h/R)^2
    planed = sum(body.plane(pts[i:i + 8] - np.asarray(s), h)[0].sum()
                 for s in image_shifts((1.0, 1.0, 1.0)) for i in range(0, n, 8)) * h ** 3
    assert 0 < planed / exact - 1 < 4.2e-3
    # the ring is also what makes the answer independent of where the seam is put
    moved = Sphere(radius, (0.5, 0.5, 0.5))
    centred = occupancy(moved, grid(n, 0.0, h), h, period=(1.0, 1.0, 1.0))[0].sum() * h ** 3
    assert abs(centred - with_ring) < 1e-14


def test_the_image_ring_agrees_with_minimising_the_distance_over_it():
    """Two spellings of one rule, and the assumption under which they are one.

    The design says the distance is minimised over the ring; this kernel sums each image's
    own contribution, which is what lets an exact fill and an area-weighted moment be composed.
    They are the same answer exactly while the images are disjoint -- which is the same
    condition C5's "wider than its period" refusal draws -- and this holds them together so
    that neither spelling can drift from the other.
    """
    h = 0.04
    pts = grid(25, 0.0, h)
    # a CSG node, because only a body on the plane path has a "minimise the distance" spelling
    # at all: an exact clip belongs to one image and cannot be read off a minimised distance.
    body = Difference(Sphere(0.3, (0.05, 0.05, 0.5)), Sphere(0.12, (0.05, 0.05, 0.62)))
    summed = occupancy(body, pts, h, period=(1.0, 1.0, None))[0]
    shifts = image_shifts((1.0, 1.0, None))
    assert len(shifts) == 9
    best = None
    for shift in shifts:
        d = body.surface(pts - np.asarray(shift))[0]
        best = d if best is None else np.minimum(best, d)
    nearest = np.zeros_like(best)
    for shift in shifts:
        d, _ = body.surface(pts - np.asarray(shift))
        # the node's OWN answer for that image, not a bare plane: what this test holds is the
        # ring's two spellings, and re-deriving the node's composition here would only restate it
        nearest = np.where(d == best, body.parts(pts - np.asarray(shift), h)[0], nearest)
    assert np.abs(summed - nearest).max() < 1e-14
    assert 0.0 < summed.mean() < 1.0


def test_nothing_is_clamped_when_a_body_is_wider_than_its_period():
    """The kernel hands C5's refusal the evidence rather than a plausible number.

    A cylinder of diameter 1.4 periods reads ``max fill = 2.0000``. An ordinary substrate is
    worse and is the case C5 will actually meet: ``Box((3.0, 3.0, 0.3))`` on
    ``period=(1, 1, None)`` reads **9.0000**, with ``sum_c w_c = 1.0000`` in those voxels, so
    the free invariant does not flag it either. **9, not 2, is the number the refusal has to be
    written against.**

    **(M)** ``fill = where(fill > 1, 1, fill)`` after the image sum -- the RCWA path's own line
    (``src/wavebit/fdtd/_periodic/scene.py``:802) -- makes both read 1.0000, and the only door that
    could have refused this geometry has nothing left to read.
    """
    h = 0.05
    pts = grid(20, 0.0, h)
    fill, _ = occupancy(Cylinder(0.7, 0.4, 'z', (0.5, 0.5, 0.5)), pts, h, period=(1.0, 1.0, None))
    assert fill.max() > 1.99
    fill, w = occupancy(Box((3.0, 3.0, 0.3), (0.5, 0.5, 0.5)), pts, h, period=(1.0, 1.0, None))
    assert abs(fill.max() - 9.0) < 1e-12
    assert abs(w.sum(-1).max() - 1.0) < 1e-14


def test_the_ring_is_periodic_in_the_body_for_one_period_only():
    """The precondition ``occupancy`` cannot enforce, pinned so it cannot be forgotten silently.

    The ``+/-1`` ring is exact while the body's extent is smaller than its period **and the body
    lies in the cell**. Move it two periods and it vanishes, with no warning:

        centre + 0 periods in x            V = 0.06544985
        centre + 1 period                  V = 0.06544985   worst |dfill| 7.8e-14
        centre + 2 periods                 V = 0.01252029   worst |dfill| 1.00
        centre + 3 periods                 V = 0.00000000   worst |dfill| 1.00
        centre + 2 periods, rings=(2,1,1)  V = 0.06544985

    ``occupancy`` is given a period and an array of points and is never told where the cell's
    origin is, so it cannot fold a body whose position it also cannot ask for -- a CSG node has
    none. C2 owns the cell and owns the fold; this test owns the statement. It is the same shape
    of defect as the repository's own memory "NVF not periodic under translation -- the raster
    read the torus flat", and the reason it needs a test rather than a docstring is that the
    wrong answer is a plausible one.
    """
    h = 0.05
    pts = grid(20, 0.0, h)
    reference = occupancy(Sphere(0.25, (0.11, -0.07, 0.03)), pts, h, period=(1., 1., 1.))[0]
    assert abs(reference.sum() * h ** 3 - 0.06544985) < 1e-7
    one = occupancy(Sphere(0.25, (1.11, -0.07, 0.03)), pts, h, period=(1., 1., 1.))[0]
    assert np.abs(one - reference).max() < 1e-13
    for periods in (2, 3):
        gone = occupancy(Sphere(0.25, (0.11 + periods, -0.07, 0.03)), pts, h,
                         period=(1., 1., 1.))[0]
        assert np.abs(gone - reference).max() > 0.9, periods
        wider = occupancy(Sphere(0.25, (0.11 + periods, -0.07, 0.03)), pts, h,
                          period=(1., 1., 1.), rings=(periods + 1, 1, 1))[0]
        assert np.abs(wider - reference).max() < 1e-13, periods


def test_the_image_ring_is_enumerated_in_one_fixed_order():
    """x outer, z inner, and a non-periodic axis contributes one term, not three."""
    assert image_shifts((None, None, None)) == [(0., 0., 0.)]
    assert len(image_shifts((1.0, 2.0, None))) == 9
    assert len(image_shifts((1.0, 2.0, 3.0))) == 27
    assert image_shifts((1.0, 2.0, None))[:3] == [(-1., -2., 0.), (-1., 0., 0.), (-1., 2., 0.)]
    assert len(image_shifts((1.0, 1.0, 1.0), rings=(1, 0, 0))) == 3


# ----------------------------------------------------------------------------------------
# the derivatives
# ----------------------------------------------------------------------------------------

def _loss(body_of, value, pts, xp):
    fill, w = occupancy(body_of(value), pts, 0.08)
    return xp.sum(fill * fill) + xp.sum(w * w)


CENTRE = (0.013, -0.021, 0.007)
LEAVES = {
    'radius': (lambda v: Sphere(v, CENTRE), 0.22),
    'centre': (lambda v: Sphere(0.2, v), np.asarray(CENTRE)),
    'height': (lambda v: Cylinder(0.19, v, 'z', CENTRE), 0.27),
    'angle': (lambda v: Box((0.30, 0.21, 0.26), CENTRE, v), 0.37),
    'size': (lambda v: Box(v, CENTRE, 0.4), np.asarray([0.30, 0.21, 0.26])),
    'vertices': (lambda v: Prism(v, 0.33, 'z', CENTRE, 0.0), np.asarray(HEX)),
}


def _central_difference(make, value, pts):
    value = np.asarray(value, dtype=float)
    flat = value.reshape(-1).copy()
    out = np.zeros_like(flat)
    step = H_STEP * max(1.0, float(np.abs(value).max()))
    for i in range(flat.size):
        total = 0.0
        for sign in (+1., -1.):
            probe = flat.copy()
            probe[i] += sign * step
            total += sign * _loss(make, probe.reshape(value.shape), pts, np)
        out[i] = total / (2 * step)
    return out.reshape(value.shape)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('leaf', list(LEAVES))
def test_a_finite_difference_says_what_the_backend_says(backend, leaf):
    """Radius, centre, height, angle, size and vertices, on all three backends.

    NumPy has no autodiff, so its row is the reference every other row is compared against;
    JAX and Torch are each held to the stage-2/3 budget ``1e-7 |g_fd| + 1e-10 |f|``.

    **The step.** Measured 2026-09-20, ``jax.grad`` against the central difference, worst
    relative error over each leaf's components:

        leaf           |g|      1e-3     1e-4     1e-5     1e-6     1e-7
        radius      1.08e+03  5.1e-01  8.0e-01  6.6e-09  7.0e-11  7.0e-11
        centre      1.38e+01  9.7e-01  6.4e-06  6.4e-08  6.2e-10  8.0e-09
        height      2.63e+02  1.3e-14  4.0e-14  2.1e-12  5.8e-13  1.4e-10
        angle       3.90e+01  3.5e-06  3.5e-08  3.0e-10  8.4e-11  2.7e-10
        size        1.60e+02  5.6e-04  5.6e-06  5.6e-08  5.8e-10  3.1e-10
        vertices    1.66e+02  3.2e-03  3.1e-05  3.1e-07  3.1e-09  6.1e-10

    Truncation falling as ``h^2`` to a floor near ``1e-6``, so :data:`H_STEP` is ``1e-6`` and
    the worst reading there is 3.1e-9 -- thirty times inside the budget. One row does not
    follow that shape and it says something: ``height`` is 1e-14 at every step because the
    axial overlap is *piecewise linear* in the height, so a central difference of it has no
    truncation at all; and ``radius`` reads O(1) at 1e-3 and 1e-4 because a step that large
    moves whole voxels across a knot, which is the same effect the design's G8 note is about
    seen from the other side.

    **There is no CSG row here any more, and its absence is the finding.** The previous writing
    carried ``csg radius`` and it passed, because that particular placement put no voxel on the
    node's crease. Where one does, the loss is *discontinuous* in the leaf and no step converges
    -- see :func:`test_a_traced_parameter_on_a_csg_node_is_refused_by_name`, which is now what
    holds that case.

    Every gradient here is order 1e1 to 1e3, so none of them is the trap an AD-against-FD gate
    falls into on a staircase, where both sides are exactly zero and the comparison cannot
    fail.
    """
    make, value = LEAVES[leaf]
    pts = grid(8, -0.32, 0.08)
    reference = _central_difference(make, value, pts)
    assert np.abs(reference).max() > 1.0, 'a gradient of zero would make this test vacuous'
    if backend == 'numpy':
        pytest.skip('numpy is the reference the other two are held to')
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        got = np.asarray(jax.grad(lambda v: _loss(make, v, jnp.asarray(pts), jnp))(
            jnp.asarray(value, dtype=jnp.float64)))
        value_of = float(_loss(make, jnp.asarray(value, dtype=jnp.float64), jnp.asarray(pts), jnp))
    else:
        import torch
        leaf_value = torch.tensor(np.asarray(value, dtype=float), requires_grad=True)
        loss = _loss(make, leaf_value, torch.tensor(pts), torch)
        loss.backward()
        got, value_of = leaf_value.grad.numpy(), float(loss.detach())
    budget = RTOL * np.abs(reference) + FTOL * abs(value_of)
    assert np.all(np.abs(got - reference) <= budget), (leaf, got, reference)


def test_a_circular_cylinders_angle_carries_no_gradient():
    """Measured 0.0 exactly, over 4096 voxels -- the key has nothing to turn.

    A circle is its own rotation, so ``angle=`` on a :class:`Cylinder` is the defect the design
    refuses by name on ``Sphere(angle=)``: a keyword whose derivative cannot be anything but
    zero, so a design loop that reaches for it silently optimises nothing (the ``circle``/``ly``
    lesson, ``src/wavebit/rcwa/_engine/parametric.py``:348-355). The kernel accepts it because its
    extrusions share one constructor; the public ``fd.Cylinder`` does not take it (C4). The
    rotation itself is not dead code: the same line
    turns a :class:`Prism` and a :class:`Box`, and this test's sibling above measures those
    gradients at order 1e1.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    pts = jnp.asarray(grid(16, -0.8, 0.1))

    def loss(angle):
        fill, w = occupancy(Cylinder(0.31, 0.44, 'z', CENTRE, angle), pts, 0.1)
        return jnp.sum(fill * fill) + jnp.sum(w * w)

    assert float(jax.grad(loss)(jnp.asarray(0.37))) == 0.0
    assert float(loss(jnp.asarray(0.37))) > 1.0


def test_the_kernel_survives_a_jit_boundary():
    """The solver crosses ``jax.jit``; so must this (memory: the test must make it cross).

    A Python branch on a traced value would raise here rather than quietly take the concrete
    path, and a value that is right eagerly and wrong under trace would show as a mismatch.

    **The doors are part of what crosses.** ``jax.jit`` is the one transform with no value to
    read (``_bodies.readable``'s table), so the size and orientation refusals become array
    expressions there rather than raises. That the factor is exactly ``1.0`` for a legal body is
    held bitwise in
    :func:`test_a_door_holds_under_every_transform_that_carries_a_value` and cannot be held
    here: XLA reassociates, so a jitted result differs from an eager one in the last bits
    whatever the guard does.

    The CSG row traces the **points** and not the radius, because a traced parameter on a CSG
    node is refused (:func:`test_a_traced_parameter_on_a_csg_node_is_refused_by_name`). That is
    what a forward solve does with any body, and it still makes the whole kernel cross the
    boundary.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    h = 0.08
    pts = grid(10, -0.4, h)
    for body_of, value in ((lambda r: Sphere(r, CENTRE), 0.22),
                           (lambda r: Cylinder(r, 0.3, 'z', CENTRE, 0.3), 0.19),
                           (lambda v: Prism(v, 0.33, 'z', CENTRE, 0.0), np.asarray(HEX))):
        def run(r, p):
            return occupancy(body_of(r), p, h)
        eager = run(jnp.asarray(value), jnp.asarray(pts))
        traced = jax.jit(run)(jnp.asarray(value), jnp.asarray(pts))
        fill_atol = 1e-14
        body = body_of(value)
        if isinstance(body, Sphere):
            # Eight O(R^3) corner primitives cancel to O(h^3): the fill's roundoff
            # scale is eps * (R/h)^3. This is a budget, not a libm error bound.
            # Original d1119445 and the integration both differ by 1.2823e-14 on
            # JAX 0.11 CPU; keep the moment and other bodies' original budget.
            fill_atol = max(fill_atol, 8 * np.finfo(np.float64).eps * (body.radius / h) ** 3)
        for (a, b), atol in zip(zip(eager, traced), (fill_atol, 1e-14)):
            assert np.abs(np.asarray(a) - np.asarray(b)).max() < atol
    # a CSG node: concrete geometry, traced points -- what a forward solve does
    node = Difference(Sphere(0.3), Box((0.2, 0.2, 0.9), (0., 0., 0.), 0.5))
    eager = occupancy(node, jnp.asarray(pts), 0.08)
    traced = jax.jit(lambda p: occupancy(node, p, 0.08))(jnp.asarray(pts))
    for a, b in zip(eager, traced):
        # 1e-13 and not the leaves' 1e-14: a node now sums its children's own routes where they
        # hold no volume of a voxel in common, which is more arithmetic for XLA to reassociate.
        # Measured worst 2.03e-14; this bounds a REASSOCIATION, not a rule (see the docstring).
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-13


# ----------------------------------------------------------------------------------------
# the doors this kernel does have
# ----------------------------------------------------------------------------------------

def test_an_axis_that_is_not_an_axis_is_refused_by_name():
    for make in (lambda a: Cylinder(0.2, 0.3, a), lambda a: Prism(HEX, 0.3, a),
                 lambda a: Box((0.2, 0.2, 0.2), (0., 0., 0.), 0., a)):
        with pytest.raises(ValueError, match="axis must be one of 'x', 'y', 'z'"):
            make('w')


def test_a_csg_node_of_one_body_is_refused():
    """``Union(a)`` is ``a`` spelled twice, and a second spelling is what this repository deletes."""
    for node in (Union, Intersection, Difference):
        with pytest.raises(ValueError, match='at least two bodies'):
            node(Sphere(0.2, (0., 0., 0.)))


def test_a_length_is_a_size_and_a_non_positive_one_is_refused_by_name():
    """Three bodies gave three different wrong answers from a length that is not a size.

    Measured before these doors, and each of the three is a different failure:

        Cylinder(radius=-0.2)      fill = 16.0000  with sum_c w_c = 0
        Box(size=(-0.3, 0.2, 0.2)) fill =  0.0000
        Sphere(radius=-0.2)        fill =  0.0000

    The cylinder is the instructive one: ``clipped_disc_area`` with a negative radius returns
    sixteen times the box it is clipped to, while the plane reports no interface at all -- both
    of the kernel's free invariants broken at once, by a constructor that had no door. The
    repository already carries the memory ("a length is a size -- a negative ``lx`` subtracted
    the shape"); this is the same defect in the same week.
    """
    match = 'a length is a size'
    with pytest.raises(ValueError, match=match):
        Sphere(-0.2)
    with pytest.raises(ValueError, match=match):
        Sphere(0.0)
    with pytest.raises(ValueError, match=match):
        Cylinder(-0.2, 0.3, 'z')
    with pytest.raises(ValueError, match=match):
        Cylinder(0.2, -0.3, 'z')
    with pytest.raises(ValueError, match=match):
        Box((-0.3, 0.2, 0.2))
    with pytest.raises(ValueError, match=match):
        Prism(HEX, -0.3, 'z')
    with pytest.raises(ValueError, match='a finite length'):
        Sphere(float('nan'))
    with pytest.raises(ValueError, match='a finite length'):
        Sphere(float('inf'))          # (M) `isfinite` weakened to `~isnan` fails here alone
    with pytest.raises(ValueError, match='a finite length'):
        Cylinder(0.2, float('-inf'), 'z')
    with pytest.raises(ValueError, match='there is none'):
        Sphere(None)                  # a missing argument, named as one and not as a NaN
    # `dx` is a length too, and it was the one length in the package with no door. Measured
    # before it, on a 10^3 grid at |dx| = 0.05:
    #     body        dx = +0.05                   dx = -0.05
    #     Sphere      fill sum 268.0826, live 296  fill sum 136.0000, live 0   <- half a body,
    #     Box         fill sum 120.0000, live 192  fill sum   0.0000, live 0      no interface
    #     Cylinder    fill sum 301.5929, live 364  fill sum   0.0000, live 0   <- not there
    # -- and `occupancy`'s advertised contract (fill in [0, 1], sum_c w_c == 1) is SATISFIED by
    # both, so nothing downstream could flag them. On JAX there was not even a RuntimeWarning.
    small = grid(4, -0.16, 0.08)
    for bad in (-0.05, 0.0, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='length'):
            occupancy(Sphere(0.2), small, bad)
        with pytest.raises(ValueError, match='length'):
            cut(np.zeros(1), np.asarray([[0., 0., 1.]]), bad)
        with pytest.raises(ValueError, match='length'):
            cut_parts(np.zeros(1), np.asarray([[0., 0., 1.]]), bad)
    # and the same doors let a traced leaf through, which is the only thing they must not stop
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    assert float(jax.grad(lambda r: occupancy(Sphere(r, CENTRE), jnp.asarray(grid(6, -0.24, 0.08)),
                                              0.08)[0].sum())(jnp.asarray(0.2))) > 0.0
    with pytest.raises(ValueError, match=match):          # a grad tracer still carries a value
        jax.grad(lambda r: occupancy(Sphere(-r, CENTRE), jnp.asarray(grid(4, -0.16, 0.08)),
                                     0.08)[0].sum())(jnp.asarray(0.2))


def test_an_outline_that_is_not_a_simple_counter_clockwise_polygon_is_refused_by_name():
    """Clockwise, self-intersecting, and fewer than three vertices.

    The clockwise case is the C1 review's, and it is the quiet one. ``clipped_polygon_area``
    ends with the RCWA source's rounding guard ``where(area < 0, 0, area)``, so a reversed list
    reads area 0 in every cell, while the signed distance, the even-odd inside test and
    ``polygon_edge_moments`` are all orientation-blind. Measured on ``HEX`` against
    ``HEX[::-1]`` at ``h = 0.05``: volume **0.030951** against **0.000000**, an identical
    distance field, and 240 voxels still reporting ``sum_c w_c = 1`` -- a body that is not
    there, with a live interface, which is exactly the state
    :func:`test_w_vanishes_only_where_the_fill_is_already_decided` forbids in the other
    direction.
    """
    with pytest.raises(ValueError, match='counter-clockwise'):
        Prism(HEX[::-1], 0.33, 'z')
    with pytest.raises(ValueError, match='counter-clockwise'):
        Prism(star(6, 0.42, 0.10)[::-1], 0.33, 'z')
    with pytest.raises(ValueError, match='at least three vertices'):
        Prism(((0., 0.), (0.2, 0.1)), 0.33, 'z')
    with pytest.raises(ValueError, match='a simple polygon'):        # a bow tie
        Prism(((-0.2, -0.2), (0.2, 0.2), (0.2, -0.2), (-0.2, 0.2)), 0.33, 'z')
    with pytest.raises(ValueError, match='a simple polygon'):        # a vertex on an edge
        Prism(((-0.2, -0.2), (0.2, -0.2), (0.0, -0.2), (0.0, 0.2)), 0.33, 'z')
    with pytest.raises(ValueError, match=r'shape \(nv, 2\)'):
        Prism(((0., 0., 0.), (0.2, 0.1, 0.0), (0.1, 0.3, 0.0)), 0.33, 'z')
    # a vertex list NumPy cannot read at all -- ragged, or not numbers -- used to skip all four
    # refusals silently, because "there is no value here" and "this is not a vertex array" were
    # the same answer. They are told apart by whether the thing has a shape.
    for rubbish in ([[0., 0.], [1., 0.], [1., 1., 5.]], [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError, match=r'shape \(nv, 2\)'):
            Prism(rubbish, 0.33, 'z')
    # a string and a None are refused one door earlier now, by the branch that refuses a complex
    # vertex list: they are not numbers at all, and the message names the type rather than the
    # shape it could not read (:func:`test_a_geometry_value_that_is_not_a_real_number_...`)
    for rubbish in ('abc', None, {'u': 1}):
        with pytest.raises(ValueError, match='not a number at all'):
            Prism(rubbish, 0.33, 'z')
    # every legal outline in this file passes, including the six non-convex ones
    for outline in dict(NON_CONVEX, hexagon=HEX).values():
        Prism(outline, 0.33, 'z')


def test_every_geometric_predicate_is_free_of_the_coordinate_scale():
    """The same body in metres and in micrometres is the same body, or nothing here is safe.

    The C1 blocker came back once already because the convexity test compared a cross product
    against ``1e-12 * max(|cross|, initial=1.0)``: an **absolute** floor, so every polygon whose
    coordinates are below about 3e-6 was classified convex, took the ``nv + 4`` clip buffer and
    overflowed it. ``src/wavebit/fdtd/_scene.py`` fixes units as "lengths in the user's unit", so a
    visible-wavelength structure written in SI metres is exactly that input. Measured on the
    shipped code before this test, the gate's own 6-point
    star at 1e-6:

        numpy  IndexError: index 16 is out of bounds for axis 2 with size 16
        jax    fill = nan
        torch  fill = 0.38101489        against a truth of 0.34210489

    -- the same three answers the previous round existed to abolish, from the same input in a
    different unit. A mutant that spells the tolerance absolutely passed all 45 tests, because
    every outline in this file is of order 0.4.

    So this test sweeps the scale rather than the shape: six outlines over fifteen decades. The
    classification must be identical at every scale, the buffer must hold, and the fill must be
    the same number on all three backends -- which is scale-*invariant*, because a fill is a
    ratio of volumes.
    """
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(5)
    scales = (1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6)
    for name, outline in dict(NON_CONVEX, hexagon=HEX).items():
        verts = np.asarray(outline)
        at_one = polygon_shape('P', verts)
        for scale in scales:
            assert polygon_shape('P', verts * scale) == at_one, (name, scale)
        # the buffer the classification buys really does hold, at the two ends of the sweep
        for scale in (1e-9, 1e6):
            worst = 0
            for _ in range(200):
                side = rng.uniform(0.02, 0.6) * scale
                lo = rng.uniform(-0.6, 0.6, 2) * scale
                worst = max(worst, reference_clip(verts * scale, lo[0], lo[0] + side,
                                                  lo[1], lo[1] + side)[1])
            assert worst <= clip_capacity(len(outline), at_one[1]), (name, scale, worst)
    # and the whole body, through the public entry point, on the input that used to break
    star6 = star(6, 0.42, 0.10)
    truth = float(occupancy(Prism(star6, 0.4, 'z'), np.zeros((1, 1, 3)), 0.5)[0][0, 0])
    assert abs(truth - 0.34210489) < 1e-7
    for scale in (1e-9, 1e-6, 1e3, 1e6):
        body = Prism(tuple((u * scale, v * scale) for u, v in star6), 0.4 * scale, 'z')
        pts, dx = np.zeros((1, 1, 3)), 0.5 * scale
        on_numpy = float(occupancy(body, pts, dx)[0][0, 0])
        on_jax = float(occupancy(body, jnp.asarray(pts), dx)[0][0, 0])
        on_torch = float(occupancy(body, torch.tensor(pts), dx)[0][0, 0])
        assert abs(on_numpy - truth) < 1e-12, (scale, on_numpy)
        assert abs(on_jax - truth) < 1e-12, (scale, on_jax)
        assert abs(on_torch - truth) < 1e-12, (scale, on_torch)


def test_a_door_holds_under_every_transform_that_carries_a_value():
    """``Sphere(radius=-0.3)`` must be refused wherever there is a number to read.

    ``_bodies.readable``'s docstring carries this as a table and this is the table. The previous
    writing claimed the doors were "skipped, and only skipped, under ``jax.jit``" and was wrong
    on three rows: ``torch.func.grad``, ``torch.func.vmap`` and ``jax.vmap`` all came back with
    no value, and ``Sphere(-0.3)`` was **accepted** under ``torch.func.grad`` with a finite
    gradient of 135.00000000000014 -- a negative radius differentiated, which is this
    repository's own gradient-ascent scar in the modern Torch functional API.

    ``jax.jit`` is the one transform that cannot be doored: a ``DynamicJaxprTracer`` carries a
    ``Var`` and not a value, and a *jitted gradient* is one too, so there is nothing to read and
    nothing to branch on. What holds there instead is a NaN, which is the other half of this
    test: a body that is not a size must not come back as a plausible number.
    """
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    pts = np.zeros((2, 3))
    pts[1] = (0.05, 0.0, 0.0)
    match = 'a length is a size'

    def fill_of(radius, p):
        return occupancy(Sphere(radius, (0., 0., 0.)), p, 0.1)[0]

    refusals = {
        'eager': lambda: fill_of(-0.3, pts),
        'jax concrete': lambda: fill_of(jnp.asarray(-0.3), jnp.asarray(pts)),
        'jax.grad': lambda: jax.grad(lambda r: fill_of(-r, jnp.asarray(pts)).sum())(
            jnp.asarray(0.3)),
        'jax.vmap': lambda: jax.vmap(lambda r: fill_of(-r, jnp.asarray(pts)))(
            jnp.asarray([0.3, 0.4])),
        'jax.vmap(grad)': lambda: jax.vmap(jax.grad(
            lambda r: fill_of(-r, jnp.asarray(pts)).sum()))(jnp.asarray([0.3, 0.4])),
        'torch plain': lambda: fill_of(torch.tensor(-0.3), torch.tensor(pts)),
        'torch autograd': lambda: fill_of(-torch.tensor(0.3, requires_grad=True),
                                          torch.tensor(pts)),
        'torch.func.grad': lambda: torch.func.grad(
            lambda r: fill_of(-r, torch.tensor(pts)).sum())(torch.tensor(0.3)),
        'torch.func.jacrev': lambda: torch.func.jacrev(
            lambda r: fill_of(-r, torch.tensor(pts)).sum())(torch.tensor(0.3)),
        'torch.func.jacfwd': lambda: torch.func.jacfwd(
            lambda r: fill_of(-r, torch.tensor(pts)).sum())(torch.tensor(0.3)),
        'torch.func.vmap': lambda: torch.func.vmap(
            lambda r: fill_of(-r, torch.tensor(pts)))(torch.tensor([0.3, 0.4])),
        'torch.func.vmap(vmap)': lambda: torch.func.vmap(torch.func.vmap(
            lambda r: fill_of(-r, torch.tensor(pts))))(torch.tensor([[0.3, 0.4]])),
    }
    for where, call in refusals.items():
        with pytest.raises(ValueError, match=match):
            call()
    # the ones that cannot be a refusal, and what stands there instead. The shipped table claimed
    # `torch.func.vmap(grad)` was READ and it is not: instrumented, `readable` walks three
    # torch.Tensor layers and the third has no storage, so item(), detach().item(), float() and
    # unwrap_if_dead().item() all raise `vmap: It looks like you're ... calling .item()` and
    # detach().cpu().numpy() raises `Cannot access data pointer of Tensor that doesn't have
    # storage`. `vmap(vmap)` above is the control: the same three layers, and the third READS.
    dead_ends = {
        'jax.jit': lambda: jax.jit(lambda r: fill_of(-r, jnp.asarray(pts)))(jnp.asarray(0.3)),
        'jax.jit(jax.grad)': lambda: jax.jit(jax.grad(
            lambda r: fill_of(-r, jnp.asarray(pts)).sum()))(jnp.asarray(0.3)),
        'torch.func.vmap(grad)': lambda: torch.func.vmap(torch.func.grad(
            lambda r: fill_of(-r, torch.tensor(pts)).sum()))(torch.tensor([0.3, 0.4])),
        'torch.func.jacrev(vmap)': lambda: torch.func.jacrev(torch.func.vmap(
            lambda r: fill_of(-r, torch.tensor(pts)).sum()))(torch.tensor([0.3, 0.4])),
    }
    for where, call in dead_ends.items():
        got = np.asarray(call().detach() if hasattr(call(), 'detach') else call())
        assert np.isnan(got).all(), (where, got)
    # a length of exactly ZERO under jit: the eager door refuses it by name and the array guard
    # has to as well, which `value >= 0` would not -- and the -0.3 row above cannot see that.
    for zero in (jnp.asarray(0.0), jnp.asarray([0.3, 0.0, 0.2])):
        assert np.isnan(np.asarray(_loud_if_not_a_size(zero))).all(), zero
    under_jit = np.asarray(jax.jit(lambda r: fill_of(r, jnp.asarray(pts)))(jnp.asarray(0.0)))
    assert np.isnan(under_jit).all(), under_jit
    # and a legal body is untouched by the guard, BIT FOR BIT -- which is the property that lets
    # it be applied unconditionally. Held on the guard itself, because comparing a jitted result
    # with an eager one cannot say this: XLA reassociates, so those two differ in the last bits
    # whether the guard is exactly one or not.
    for legal in (jnp.asarray(0.3), jnp.asarray([0.30, 0.21, 0.26]), torch.tensor(0.3)):
        guarded = np.asarray(_loud_if_not_a_size(legal))
        assert np.array_equal(guarded, np.asarray(legal)), guarded
    assert np.isnan(np.asarray(_loud_if_not_a_size(jnp.asarray(-0.3)))).all()
    assert np.isnan(np.asarray(_loud_if_not_a_size(jnp.asarray([0.3, -0.1, 0.2])))).all()
    assert float(_loud_if_not_counter_clockwise(jnp.asarray(HEX))) == 1.0
    assert np.isnan(float(_loud_if_not_counter_clockwise(jnp.asarray(HEX[::-1]))))
    # the same two halves for the quietest polygon refusal: a clockwise outline reads area 0 in
    # every cell, so under jit it is a NaN and not a body that silently is not there
    with pytest.raises(ValueError, match='counter-clockwise'):
        Prism(HEX[::-1], 0.33, 'z')
    # a point ON the outline's edge, so that the voxel carries a live interface -- inside the
    # hexagon `area` is zero whatever the guard does, and an assertion taken there says nothing
    edge_pts = np.concatenate([pts, np.asarray([[0.17, 0.0, 0.0], [0.0, 0.17, 0.0]])], axis=0)
    assert (occupancy(Prism(HEX, 0.33, 'z'), edge_pts, 0.1)[1][-2:].sum(-1) > 0).all()
    turned_fill, turned_w = jax.jit(
        lambda v: occupancy(Prism(v, 0.33, 'z'), jnp.asarray(edge_pts), 0.1))(
            jnp.asarray(HEX[::-1]))
    assert np.isnan(np.asarray(turned_fill)).all(), turned_fill
    # `w` as well as the fill, and DEAD rather than NaN: a NaN area fails the liveness test, so
    # what a poisoned moment produces is `w = 0`. Asserting on the fill alone left the moment's
    # own factor free -- with `_turn` dropped from `exact_moment` the area came back finite, `w`
    # came back live, and the whole gate passed while reporting a live interface on a body that
    # is not there, which is the exact state this refusal exists to stop.
    assert (np.asarray(turned_w) == 0.0).all(), turned_w
    straight = np.asarray(jax.jit(lambda v: occupancy(Prism(v, 0.33, 'z'), jnp.asarray(pts),
                                                      0.1)[0])(jnp.asarray(HEX)))
    assert np.abs(straight - occupancy(Prism(HEX, 0.33, 'z'), pts, 0.1)[0]).max() < 1e-15
    # the vertex count and the array's shape are STATIC, so they are refused even under jit
    with pytest.raises(ValueError, match='at least three vertices'):
        jax.jit(lambda v: occupancy(Prism(v, 0.33, 'z'), jnp.asarray(pts), 0.1)[0])(
            jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match=r'shape \(nv, 2\)'):
        jax.jit(lambda v: occupancy(Prism(v, 0.33, 'z'), jnp.asarray(pts), 0.1)[0])(
            jnp.zeros((4, 3)))


def test_a_traced_parameter_on_a_csg_node_is_refused_by_name():
    """A CSG node's fill is discontinuous, so its geometry may not be a design variable.

    The C1 review found, and this file's crease test records, that a CSG node's **fill** steps
    by 3.15e-02 across a slide of 5.5e-17 -- the tangent plane's fill is a function of the
    *selected* child's normal, and at a crease the selection swaps. Measured here
    off-tree with a prototype of the fix, a central difference of a loss in one centre coordinate,
    at the slide value the jump sits at:

        step      1e-3      1e-4      1e-5      1e-6      1e-7
        plane    268.81   2685.85  26855.47 268551.57  2685512.50

    It **diverges as 1/step**: the loss itself is discontinuous in the leaf, so refining the
    step never converges, and ``jax.grad`` reads -5.98 where the one-sided slope is +24.81 --
    wrong in sign, not merely in the last digits. An optimiser's line search sees a genuine 3e-2
    cliff in ``eps`` from a geometry change of order 1e-17.

    So the parameter is refused. A forward-only CSG node is honest -- nothing about a *value* is
    wrong, and this refusal is on the leaf and not on tracing, so a forward solve that traces
    the **points** is untouched (:func:`test_the_kernel_survives_a_jit_boundary`). Every body
    with an exact fill is immune and remains differentiable, which is where a design variable
    belongs. Why the alternative was rejected, with its numbers, is in ``_bodies._Combination``.
    """
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update('jax_enable_x64', True)
    match = "geometry must be concrete"
    pts = np.zeros((2, 3))

    def node(radius):
        return Difference(Sphere(radius, (0., 0., 0.)), Cylinder(0.12, 1.0, 'z'))

    with pytest.raises(ValueError, match=match):
        jax.grad(lambda r: occupancy(node(r), jnp.asarray(pts), 0.1)[0].sum())(jnp.asarray(0.3))
    with pytest.raises(ValueError, match=match):
        jax.jit(lambda r: occupancy(node(r), jnp.asarray(pts), 0.1)[0])(jnp.asarray(0.3))
    with pytest.raises(ValueError, match=match):
        jax.vmap(lambda r: occupancy(node(r), jnp.asarray(pts), 0.1)[0])(jnp.asarray([0.3, 0.4]))
    with pytest.raises(ValueError, match=match):
        node(torch.tensor(0.3, requires_grad=True))
    with pytest.raises(ValueError, match=match):
        torch.func.grad(lambda r: occupancy(node(r), torch.tensor(pts), 0.1)[0].sum())(
            torch.tensor(0.3))
    # a centre, a size, an angle and a vertex list are leaves too, at any depth of nesting
    with pytest.raises(ValueError, match=match):
        jax.grad(lambda c: occupancy(Union(Sphere(0.2, c), Box((0.2, 0.2, 0.2))),
                                     jnp.asarray(pts), 0.1)[0].sum())(jnp.zeros(3))
    with pytest.raises(ValueError, match=match):
        jax.grad(lambda a: occupancy(Intersection(Sphere(0.3), Union(Box((0.2, 0.2, 0.2),
                                                                        (0., 0., 0.), a),
                                                                     Sphere(0.25))),
                                     jnp.asarray(pts), 0.1)[0].sum())(jnp.asarray(0.4))
    # concrete geometry is accepted on every backend, including a concrete backend array
    for radius in (0.3, np.float64(0.3), jnp.asarray(0.3), torch.tensor(0.3)):
        assert node(radius) is not None
    # and the body with an exact fill, which is where a design variable belongs, still works
    got = float(jax.grad(lambda r: occupancy(Sphere(r, (0.01, 0., 0.)),
                                             jnp.asarray(grid(6, -0.24, 0.08)), 0.08)[0].sum())(
        jnp.asarray(0.2)))
    assert abs(got) > 1.0


def test_cut_refuses_a_normal_that_is_not_a_unit_vector():
    """``cut`` and ``cut_parts`` are exported, so their one precondition is a door.

    Every line of the cut assumes ``|n| = 1``: the plane is ``n.(x - centre) <= -d``, so a
    scaled ``n`` rescales the offset too. Measured on the exported spelling before this door,
    with ``n`` scaled by two: ``fill = 0.450`` where the truth is 0.400, and
    ``w = [0, 0, 4]``, which does not sum to one. A zero ``n`` divides by zero.

    ``_bodies.Body.plane`` goes to the undoored ``_cut._parts`` instead, because the bodies
    construct a unit normal and the check is a reduction and a host read per call.
    """
    n = np.asarray([[0.0, 0.0, 1.0]])
    d = np.asarray([0.1])
    assert abs(cut(d, n, 1.0)[0][0] - 0.4) < 1e-15
    nan_n = np.full((1, 3), np.nan)
    for bad in (2 * n, 0.5 * n, np.zeros((1, 3)), nan_n, np.full((1, 3), np.inf)):
        with pytest.raises(ValueError, match='unit outward normal'):
            cut(d, bad, 1.0)
        with pytest.raises(ValueError, match='unit outward normal'):
            cut_parts(d, bad, 1.0)
    # the NaN row is the one the first writing let through: `worst > 1e-9` is False for a NaN,
    # so the door opened and the answer came back `fill = nan` with `sum_c w_c = 0`. Testing
    # the complement -- `~(worst <= 1e-9)` -- is what makes a non-finite normal a refusal.
    # one bad row among many is enough
    many = np.broadcast_to(n, (64, 3)).copy()
    many[17] *= 1.5
    with pytest.raises(ValueError, match='unit outward normal'):
        cut(np.zeros(64), many, 1.0)
    many = np.broadcast_to(n, (64, 3)).copy()
    many[41] = np.nan
    with pytest.raises(ValueError, match='unit outward normal'):
        cut(np.zeros(64), many, 1.0)


# ----------------------------------------------------------------------------------------
# the same cut one and two dimensions down: a component's edge, and its dual face
# ----------------------------------------------------------------------------------------
#
# `_cut.edge_cut` and `_cut.face_cut` answer, for the tangent half space, what fraction of a
# component's own edge and of its own dual face it holds -- the pair the Werner/Bauer/Cary
# material operator constrains, and the pair the C3 research operator has been taking from
# closed forms inside `studies/`. They ship here as a **preparation**: nothing in the package
# dispatches to them, `Sphere` carries exact forms of the same two quantities itself, and the
# section below measures how far apart those two answers are, which is the whole reason there
# is no fallback wiring them together.
#
# The oracles, and why each is independent of the thing it judges:
#
# * *Exact clipping, written with Python `if`s.* A segment clipped by one inequality, and a
#   square clipped by a half plane through Sutherland-Hodgman plus a shoelace area. Neither
#   shares a line with the divided-difference form under test, and both are branch-on-a-value
#   code, which is exactly what the kernel may not be.
# * *Quadrature.* The same square area read as an integral of its chord length along one axis,
#   split at the kinks, `scipy.integrate.quad`.
# * *The `Sphere` overrides' own two oracles*: the exact chord of a ball for the edge, and the
#   study gate's adaptive integration of clipped disc chords for the face
#   (`tests/fdtd/test_fd_triplet_sphere.py::independent_face`).
#
# The mutants, each edited into `src/wavebit/fd/_shapes/` or into the two study files, run, watched
# to fail, and reverted (2026-09-24):
#
#     mutant                                                            red
#     both `Sphere` interface overrides deleted (back to `None`)        30
#     `_first_difference_square`'s own `a = 0` piece dropped             6
#     the degenerate branches dropped (`where(a > 0, ...)` removed)      2
#     the face's two steps not sorted (`a1, a2 = a_u, a_v`)              2
#     `_first_difference`'s safe denominator removed                     2
#     the face's clip on `c` dropped                                     2
#     `_decided` reading `d < 0` for `d <= 0`                            1
#     the `axis` door removed                                            1
#     `circle_fraction`'s `xp` argument dropped                          1
#     `circle_fraction`'s lift built with `np.stack` again               1
#
# Two of those did not go red on the first sweep, and both of them changed something.
# `_first_difference`'s safe denominator passed all twelve tests, because the caller pins its
# `c` to the dead end of a `clip` whose transpose is a `select` and therefore *discards* the
# NaN cotangent rather than multiplying by it; the guard is right and stays, and what was added
# is a check of the invariant where it lives
# (`_each_divided_difference_is_guarded_at_a_zero_step`). Dropping the **edge's** clip passed
# all twelve too, and that mutant was correct: measured over 2000 normals at depths of 1 to 1e12
# voxels, clipped and unclipped agree to 0.0 exactly, because `_first_difference` returns the
# constant 1 rather than differencing two large numbers. The clip was deleted from
# `_edge_parts`, and the mutant that remains is the **face's** clip, which at those depths is
# worth 1.9e-3.


def _oracle_edge(d, n, dx, k):
    """The segment clipped by the closed half space, in plain Python with branches."""
    h = dx / 2
    a = float(n[k])
    lo, hi = -h, h
    if a > 0:
        hi = min(hi, -d / a)
    elif a < 0:
        lo = max(lo, -d / a)
    else:                                   # the plane is parallel to the edge: all or nothing
        return 1.0 if d <= 0 else 0.0
    return max(0.0, hi - lo) / dx


def _oracle_face_clip(d, n, dx, k):
    """The dual square clipped by the half plane -- Sutherland-Hodgman, then a shoelace area."""
    u, v = (0, 1, 2)[:k] + (0, 1, 2)[k + 1:]
    nu, nv = float(n[u]), float(n[v])
    h = dx / 2
    poly, out = [(-h, -h), (h, -h), (h, h), (-h, h)], []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % 4]
        sp, sq = nu * p[0] + nv * p[1] + d, nu * q[0] + nv * q[1] + d
        if sp <= 0:
            out.append(p)
        if (sp < 0 < sq) or (sq < 0 < sp):
            t = sp / (sp - sq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    if len(out) < 3:
        return 0.0
    area = sum(out[i][0] * out[(i + 1) % len(out)][1] - out[(i + 1) % len(out)][0] * out[i][1]
               for i in range(len(out)))
    return abs(area) / 2 / dx ** 2


def _oracle_face_quad(d, n, dx, k):
    """The same area as a quadrature of the chord length, split at its kinks."""
    from scipy.integrate import quad
    u, v = (0, 1, 2)[:k] + (0, 1, 2)[k + 1:]
    nu, nv = float(n[u]), float(n[v])
    h = dx / 2

    def chord(x):
        b = -d - nu * x
        if nv > 0:
            return max(0.0, min(h, b / nv) + h)
        if nv < 0:
            return max(0.0, h - max(-h, b / nv))
        return 2 * h if b >= 0 else 0.0

    cuts = [-h, h]
    for s in (1, -1):
        if nu != 0:
            x = (-d - s * nv * h) / nu
            if -h < x < h:
                cuts.append(x)
    cuts.sort()
    return sum(quad(chord, a, b, limit=200)[0] for a, b in zip(cuts[:-1], cuts[1:])) / dx ** 2


def _interface_families():
    """``{name: (normals, offsets)}`` -- random, exactly axis-aligned, exactly diagonal, and
    the two decided ones. The exact families are the point: an axis-aligned normal is what
    kills the eight-corner form's lower-dimensional twins, and a diagonal one is where two
    steps are equal and the sort has nothing to order."""
    rng = np.random.default_rng(7717)
    dx = 0.0125
    out = {}
    n = rng.normal(size=(400, 3))
    out['400 random normals'] = (n / np.linalg.norm(n, axis=-1)[:, None],
                                 rng.uniform(-0.9, 0.9, 400) * dx)
    ax = np.repeat(np.concatenate([np.eye(3), -np.eye(3)]), 40, axis=0)
    out['240 exactly axis-aligned normals'] = (ax, rng.uniform(-0.9, 0.9, len(ax)) * dx)
    body = np.array([[sx, sy, sz] for sx in (1., -1.) for sy in (1., -1.)
                     for sz in (1., -1.)]) / np.sqrt(3)
    face = []
    for i in range(3):
        j, k = (0, 1, 2)[:i] + (0, 1, 2)[i + 1:]
        for sj in (1., -1.):
            for sk in (1., -1.):
                e = np.zeros(3)
                e[j], e[k] = sj, sk
                face.append(e / np.sqrt(2))
    dg = np.repeat(np.concatenate([body, np.array(face)]), 20, axis=0)
    out['400 exactly diagonal normals'] = (dg, rng.uniform(-0.9, 0.9, len(dg)) * dx)
    far = rng.normal(size=(100, 3))
    far = far / np.linalg.norm(far, axis=-1)[:, None]
    out['100 cells wholly inside (d = -5 dx)'] = (far, np.full(100, -5 * dx))
    out['100 cells wholly outside (d = +5 dx)'] = (far, np.full(100, +5 * dx))
    return dx, out


def test_the_edge_and_face_cuts_are_what_an_independent_clip_says():
    """The two new forms against an exact clip and a quadrature, over five families and a node.

    Measured 2026-09-24 at ``dx = 0.0125``, worst ``|difference|`` over the family and all
    three axes:

        family                                        edge     face (clip)  face (quad)
        400 random normals                           2.2e-16     3.3e-16      3.3e-16
        240 exactly axis-aligned normals             0           2.2e-16      3.3e-16
        400 exactly diagonal normals                 1.1e-16     2.2e-16      2.2e-16
        100 cells wholly inside  (d = -5 dx)         0           2.2e-16      2.2e-16
        100 cells wholly outside (d = +5 dx)         0           0            0
        500 points on Union(Sphere, Box)             2.2e-16     2.2e-16
        500 points on Difference(Sphere, Box)        2.2e-16     2.2e-16

    **The two exact families are what the degenerate branches are for.** An exactly
    axis-aligned normal is the case with no volume at all in the lower-dimensional cell -- the
    edge parallel to the plane, the dual face parallel to it -- and it is not rare: it is every
    voxel of a slab, of an axis-aligned box face, and of two of a sphere's three great circles.
    An exactly diagonal normal is where the two steps of the face are equal and the sort has
    nothing to order.

    **The CSG rows are the reason these live in ``_cut.py`` and not in a body.** A node has no
    exact clip of its own -- that is what ``_Combination``'s whole docstring is about -- and
    these two forms take its composed ``(d, n)`` exactly as they take a leaf's.
    """
    dx, families = _interface_families()
    for name, (n, d) in families.items():
        for axis in range(3):
            got_e = np.asarray(edge_cut(d, n, dx, axis))
            got_f = np.asarray(face_cut(d, n, dx, axis))
            for i in range(len(d)):
                assert abs(got_e[i] - _oracle_edge(d[i], n[i], dx, axis)) < 1e-14, (name, i, axis)
                assert abs(got_f[i] - _oracle_face_clip(d[i], n[i], dx, axis)) < 1e-14, (name, i)
                assert abs(got_f[i] - _oracle_face_quad(d[i], n[i], dx, axis)) < 1e-13, (name, i)
    rng = np.random.default_rng(31337)
    pts = rng.uniform(-0.4, 0.4, size=(500, 3))
    for node in (Union(Sphere(0.3), Box((0.4, 0.3, 0.5), (0.1, 0., 0.))),
                 Difference(Sphere(0.3), Box((0.2, 0.2, 0.9), (0., 0., 0.), 0.5))):
        d, n = node.surface(pts)
        d, n = np.asarray(d), np.asarray(n)
        for axis in range(3):
            got_e = np.asarray(edge_cut(d, n, dx, axis))
            got_f = np.asarray(face_cut(d, n, dx, axis))
            for i in range(len(d)):
                assert abs(got_e[i] - _oracle_edge(d[i], n[i], dx, axis)) < 1e-14, (node, i)
                assert abs(got_f[i] - _oracle_face_clip(d[i], n[i], dx, axis)) < 1e-14, (node, i)


def test_the_face_cut_is_c1_and_the_edge_cut_is_only_c0():
    """Slide the plane through the cell and read the slope's jumps. Both numbers are the point.

    Measured 2026-09-24, ``dx = 0.1``, ``n = (0.3, 0.48, 0.8246)`` normalised, ``d`` swept
    across the cell in steps of ``h dx``; the slope is a forward difference and the column is
    the largest jump between two consecutive slopes:

        h          edge_cut max|df'|   face_cut max|df'|
        1e-3           32.34              2.527e-2
        5e-4           31.34              1.264e-2
        2.5e-4         29.34              6.319e-3

    ``face_cut``'s halves when ``h`` halves -- a Lipschitz derivative, so C1 -- while
    ``edge_cut``'s stays at the size of the slope itself (``max |f'| = 33.34 = 1/a_1``, with
    ``a_1 = |n_x| dx = 0.03``). **That is the exact function and not the writing of it.** The
    length of a segment inside a half space is piecewise linear in the offset, with a corner
    where an end of the segment crosses the plane; the ``m``-dimensional cut of a box is
    C^(m-1), so one dimension down from :func:`cut` there is no derivative to be continuous.

    What it costs C4 is written here rather than discovered there: a design loop that
    differentiates through an edge fraction gets a gradient that exists almost everywhere and
    jumps on a measure-zero set, and a slope ``1/a_1`` that grows without bound as the plane
    turns parallel to the edge.
    """
    dx = 0.1
    n = np.asarray([0.3, 0.48, 0.8246])
    n = n / np.linalg.norm(n)
    face_jumps = []
    for h in (1e-3, 5e-4, 2.5e-4):
        d = np.arange(-0.6, 0.6, h) * dx
        nn = np.broadcast_to(n, (len(d), 3))
        edge = np.asarray(edge_cut(d, nn, dx, 0))
        face = np.asarray(face_cut(d, nn, dx, 0))
        edge_slope, face_slope = np.diff(edge) / (h * dx), np.diff(face) / (h * dx)
        assert abs(np.abs(edge_slope).max() - 1 / (n[0] * dx)) < 1e-9, 'the slope is 1/a_1'
        # C0: the jump stays the size of the slope however fine the sampling gets
        assert np.abs(np.diff(edge_slope)).max() > 0.5 * np.abs(edge_slope).max()
        face_jumps.append(float(np.abs(np.diff(face_slope)).max()))
    # C1: halving the step halves the largest jump, to better than a percent at both halvings
    for a, b in zip(face_jumps[:-1], face_jumps[1:]):
        assert abs(a / b - 2.0) < 0.01, face_jumps
    assert face_jumps[-1] < 1e-2, face_jumps


def test_the_exact_sphere_fills_are_what_quadrature_says_and_they_are_c0_too():
    """``Sphere``'s two overrides against their own oracles -- and where each one has a corner.

    Measured 2026-09-24 over 200 placements about a radius-0.25 sphere at ``dx = 0.025``, all
    three axes: worst ``|difference|`` **1.1e-16** for the edge against the exact chord, and
    **3.7e-14** for the face against the adaptive integration of clipped disc chords that
    ``tests/fdtd/test_fd_triplet_sphere.py`` holds the study's form to.

    **Both are C0, and the face's corner is not where the edge's is.** On the same slide as
    :func:`test_the_face_cut_is_c1_and_the_edge_cut_is_only_c0`, at ``dx = 0.1`` with a
    radius-0.25 sphere: the edge's largest slope jump is 9.72 against a largest slope of 10.0
    (a corner where an end of the segment leaves the ball), and the face's is 157.05, which is
    exactly its own largest slope and exactly ``2 pi R / dx^2`` -- the corner where the
    cross-section disc *appears*, its area ``pi (R^2 - v^2)/dx^2`` starting with a finite slope
    against the zero it leaves. So the exact form is **less** regular than the tangent-plane
    ``face_cut`` it would replace, which is C1. That is a real trade for C4 to price, not a
    defect of either: one is the body and the other is its tangent plane.
    """
    quad = pytest.importorskip('scipy.integrate').quad
    dx, radius = 0.025, 0.25
    rng = np.random.default_rng(23923)
    pts = rng.normal(size=(200, 3))
    pts = pts / np.linalg.norm(pts, axis=-1)[:, None] * radius + rng.uniform(-.5, .5, (200, 3)) * dx
    body = Sphere(radius)
    for axis in range(3):
        got_e = np.asarray(body.exact_edge_fill(pts, dx, axis))
        got_f = np.asarray(body.exact_face_fill(pts, dx, axis))
        for i, p in enumerate(pts):
            t2 = sum(p[j] ** 2 for j in range(3) if j != axis)
            if radius ** 2 > t2:
                half = np.sqrt(radius ** 2 - t2)
                lo, hi = max(-dx / 2, -half - p[axis]), min(dx / 2, half - p[axis])
                truth = max(0.0, hi - lo) / dx
            else:
                truth = 0.0
            assert abs(got_e[i] - truth) < 1e-14, (axis, i)
            assert abs(got_f[i] - _disc_chord_quadrature(quad, p, dx, axis, radius)) < 1e-12
    # the two corners, on the slide the tangent plane's regularity was measured on
    dx, radius = 0.1, 0.25
    h = 2.5e-4
    shift = np.arange(-0.6, 0.6, h) * dx
    pts = np.stack([shift + 0.231, np.full_like(shift, 0.041),
                    np.full_like(shift, -0.033)], axis=-1)
    body = Sphere(radius)
    for name, f in (('edge', np.asarray(body.exact_edge_fill(pts, dx, 0))),
                    ('face', np.asarray(body.exact_face_fill(pts, dx, 0)))):
        slope = np.diff(f) / (h * dx)
        assert np.abs(np.diff(slope)).max() > 0.5 * np.abs(slope).max(), name
    face = np.asarray(body.exact_face_fill(pts, dx, 0))
    # the face's corner is the disc appearing: 2 pi R / dx^2, to a sixth of a percent at this h
    assert abs(np.abs(np.diff(face)).max() / (h * dx) / (2 * np.pi * radius / dx ** 2) - 1) < 2e-3


def _disc_chord_quadrature(quad, point, dx, axis, radius):
    """The study gate's own face oracle: adaptive integration of clipped chords, split at the
    four places the disc meets the square's sides. No corner primitive is used."""
    a, b = (0, 1, 2)[:axis] + (0, 1, 2)[axis + 1:]
    r2 = radius ** 2 - point[axis] ** 2
    if r2 <= 0:
        return 0.
    r = np.sqrt(r2)
    lo, hi = max(point[a] - dx / 2, -r), min(point[a] + dx / 2, r)
    if hi <= lo:
        return 0.
    low, high = point[b] - dx / 2, point[b] + dx / 2
    cuts = [lo, hi]
    for y in (low, high):
        if abs(y) < r:
            x = np.sqrt(r2 - y * y)
            cuts.extend(t for t in (-x, x) if lo < t < hi)
    cuts.sort()

    def chord(x):
        hh = np.sqrt(max(0., r2 - x * x))
        return max(0., min(high, hh) - max(low, -hh))

    return sum(quad(chord, u, v, epsabs=1e-15, epsrel=1e-12)[0]
               for u, v in zip(cuts[:-1], cuts[1:])) / dx ** 2


def test_the_plane_is_not_a_drop_in_for_the_exact_edge_and_face_fills():
    """The scope rule, measured: swapping the generic forms in would change every number.

    C3 reaches only the exact ``Sphere`` overrides. The generic tangent-plane forms ship beside
    them for the bodies that have no closed form -- a CSG node above all -- and this is the
    test that says the two are **not** interchangeable, which is why no ``parts``-style
    dispatcher pairs them. 4000 points about a radius-0.25 sphere per row, one sub-cell jitter
    per point, all three axes, measured 2026-09-24:

        R/h                 3        4        6        8       12       16       32
        worst |edge gap|  5.8e-1   6.7e-1   4.5e-1   7.5e-1   3.3e-1   1.6e-1   1.7e-1
        worst |face gap|  4.6e-1   1.6e-1   4.5e-1   2.7e-1   9.8e-2   8.9e-2   4.9e-2
        mean  |edge gap|  1.0e-2   7.7e-3   5.2e-3   3.9e-3   2.7e-3   1.8e-3   9.2e-4
        mean  |face gap|  2.0e-2   1.5e-2   9.7e-3   7.4e-3   4.9e-3   3.7e-3   1.8e-3

    The **mean** falls like ``h`` -- first order, not second -- and the **worst** barely falls
    at all: a cell where the tangent plane misses a corner of the body is wrong by a finite
    fraction of the cell however fine the grid is. Against that, the research operator's
    measured convergence orders (2.137819 on the cylinder, 2.063617 and 2.057240 on the glass
    and weak spheres, 2.468123 percent on the small high-index sphere) were all taken with the
    exact forms, so a silent substitution would invalidate every one of them while still
    returning a plausible number -- the failure mode this repository names on every door.

    The other half of the rule is that no body **but** ``Sphere`` answers these at all: a CSG
    node, a box, a prism and a cylinder all return ``None``, and a caller that wants an answer
    for them has to reach for the plane in the open.
    """
    rng = np.random.default_rng(4242)
    radius = 0.25
    worst_seen = 0.0
    for cells, mean_floor in ((8, 3.0e-3), (16, 1.5e-3), (32, 7.0e-4)):
        dx = radius / cells
        pts = rng.normal(size=(4000, 3))
        pts = pts / np.linalg.norm(pts, axis=-1)[:, None] * radius
        pts = pts + rng.uniform(-.5, .5, pts.shape) * dx
        body = Sphere(radius)
        d, n = body.surface(pts)
        for axis in range(3):
            for plane, exact in ((edge_cut(d, n, dx, axis), body.exact_edge_fill(pts, dx, axis)),
                                 (face_cut(d, n, dx, axis), body.exact_face_fill(pts, dx, axis))):
                gap = np.abs(np.asarray(plane) - np.asarray(exact))
                assert gap.mean() > mean_floor, (cells, axis, gap.mean())
                worst_seen = max(worst_seen, gap.max())
    assert worst_seen > 0.1, worst_seen
    pts = grid(4, -0.2, 0.1)
    for body in (Box((0.2, 0.2, 0.2)), Cylinder(0.2, 0.3), Prism(HEX, 0.3),
                 Union(Sphere(0.2), Box((0.2, 0.2, 0.2))),
                 Difference(Sphere(0.2), Box((0.1, 0.1, 0.5)))):
        assert body.exact_edge_fill(pts, 0.1, 0) is None, body
        assert body.exact_face_fill(pts, 0.1, 0) is None, body
    assert Sphere(0.2).exact_edge_fill(pts, 0.1, 0) is not None
    assert Sphere(0.2).exact_face_fill(pts, 0.1, 0) is not None


@pytest.mark.parametrize('backend', backends())
def test_the_interface_cuts_are_one_number_on_every_backend_and_every_scale(backend):
    """NumPy, JAX eager, ``jax.jit`` and Torch, at five coordinate scales six decades apart.

    200 placements per scale, all three axes, both generic forms and both ``Sphere``
    overrides; the column is the worst ``|difference|`` from the NumPy answer, measured
    2026-09-24:

        form                       jax eager   jax.jit     torch
        edge_cut                    2.2e-16    2.2e-16    2.2e-16
        face_cut                    0          0          0
        Sphere.exact_edge_fill      1.7e-14    1.7e-14    1.7e-14
        Sphere.exact_face_fill      2.0e-13    2.0e-13    2.0e-13

    The two generic forms are **exact** across the backends -- they are additions, comparisons
    and one division -- and the two exact ones carry the disc clip's own cross-backend spread,
    which is this package's known price for an ``asin`` and a guarded root (``_exact.py``).
    The scales are the other half: every one of these is a function of ``d/dx`` and of ``n``
    alone, so the same numbers come back at ``1e-6`` and at ``1e+6``, and on NumPy the worst
    difference between the unit scale and either end is 4.4e-16.
    """
    xp, cast, jit = _backend_pieces(backend)
    rng = np.random.default_rng(555)
    n = rng.normal(size=(200, 3))
    n = n / np.linalg.norm(n, axis=-1)[:, None]
    d0 = rng.uniform(-0.9, 0.9, 200)
    p0 = rng.normal(size=(200, 3))
    p0 = p0 / np.linalg.norm(p0, axis=-1)[:, None] * 0.25 + rng.uniform(-.6, .6, (200, 3)) * 0.0125
    unit_scale = None
    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        dx, d, p = 0.0125 * scale, d0 * 0.0125 * scale, p0 * scale
        body = Sphere(0.25 * scale)
        calls = []
        for axis in range(3):
            calls.append((1e-15, lambda D, N, P, a=axis: edge_cut(D, N, dx, a)))
            calls.append((1e-15, lambda D, N, P, a=axis: face_cut(D, N, dx, a)))
            calls.append((1e-13, lambda D, N, P, a=axis: body.exact_edge_fill(P, dx, a)))
            calls.append((1e-12, lambda D, N, P, a=axis: body.exact_face_fill(P, dx, a)))
        reference = [np.asarray(f(d, n, p)) for _, f in calls]
        if scale == 1.0:
            unit_scale = reference
        for wrap in ((lambda f: f),) if jit is None else ((lambda f: f), jit):
            for (tol, f), want in zip(calls, reference):
                got = np.asarray(wrap(f)(cast(d), cast(n), cast(p)))
                assert np.abs(got - want).max() < tol, (backend, scale, tol)
    # the same function of d/dx and n at every scale, on this backend's own answers
    dx = 0.0125
    body = Sphere(0.25)
    for axis in range(3):
        for got, want in ((edge_cut(cast(d0 * dx), cast(n), dx, axis), unit_scale[4 * axis]),
                          (face_cut(cast(d0 * dx), cast(n), dx, axis), unit_scale[4 * axis + 1]),
                          (body.exact_edge_fill(cast(p0), dx, axis), unit_scale[4 * axis + 2]),
                          (body.exact_face_fill(cast(p0), dx, axis), unit_scale[4 * axis + 3])):
            assert np.abs(np.asarray(got) - want).max() < 1e-12, axis


def _backend_pieces(backend):
    """``(xp, cast, jit)`` for a backend name -- ``jit`` is ``None`` where there is no trace."""
    if backend == 'numpy':
        return np, np.asarray, None
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        return jnp, jnp.asarray, jax.jit
    torch = pytest.importorskip('torch')
    return torch, (lambda v: torch.tensor(np.asarray(v, dtype=float))), None


@pytest.mark.parametrize('backend', backends())
def test_a_finite_difference_says_what_the_backend_says_on_the_interface_fills(backend):
    """``Sphere``'s two overrides differentiate, against NumPy's central difference.

    The loss is ``sum e^2 + 1.7 sum f^2`` over 300 points and all three axes, so both overrides
    and every one of their guards are in it. Leaves: the radius and the three components of the
    centre. Measured 2026-09-24 at ``H_STEP``, gradients of order 1e2 to 5e3 -- so this is not
    the vacuous comparison of two zeros -- and the worst absolute gap is **1.31e-6** on JAX and
    **1.31e-6** on Torch, against a budget ``RTOL |g_fd| + FTOL |f|`` of 5.5e-4.

    The guards are what is being tested. The edge's root is taken of a value a ``where`` has
    already replaced where the line misses the ball, and the face's disc clip is fed a unit
    disc and a centred square wherever it is inactive; without either, the forward values are
    unchanged and the backward pass is NaN -- the defect family this file already carries three
    rows of.
    """
    dx = 0.05
    rng = np.random.default_rng(1234)
    pts = rng.uniform(-0.35, 0.35, size=(300, 3))
    radius, centre = 0.25, np.asarray([0.011, -0.007, 0.019])

    def loss(points, r, c, xp):
        body = Sphere(r, c)
        out = None
        for axis in range(3):
            e, f = body.exact_edge_fill(points, dx, axis), body.exact_face_fill(points, dx, axis)
            piece = xp.sum(e * e) + 1.7 * xp.sum(f * f)
            out = piece if out is None else out + piece
        return out

    reference = [(loss(pts, radius + H_STEP, centre, np)
                  - loss(pts, radius - H_STEP, centre, np)) / (2 * H_STEP)]
    for i in range(3):
        up, down = centre.copy(), centre.copy()
        up[i] += H_STEP
        down[i] -= H_STEP
        reference.append((loss(pts, radius, up, np) - loss(pts, radius, down, np)) / (2 * H_STEP))
    reference = np.asarray(reference)
    assert np.abs(reference).min() > 1.0, 'a gradient of zero would make this test vacuous'
    if backend == 'numpy':
        pytest.skip('numpy is the reference the other two are held to')
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        points = jnp.asarray(pts)
        grad = jax.grad(lambda r, c: loss(points, r, c, jnp), argnums=(0, 1))(
            jnp.asarray(radius), jnp.asarray(centre))
        got = np.concatenate([[float(grad[0])], np.asarray(grad[1])])
        value = float(loss(points, jnp.asarray(radius), jnp.asarray(centre), jnp))
    else:
        torch = pytest.importorskip('torch')
        r = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
        c = torch.tensor(centre, dtype=torch.float64, requires_grad=True)
        out = loss(torch.tensor(pts), r, c, torch)
        out.backward()
        got = np.concatenate([[float(r.grad)], c.grad.numpy()])
        value = float(out.detach())
    assert np.all(np.isfinite(got)), got
    assert np.all(np.abs(got - reference) <= RTOL * np.abs(reference) + FTOL * abs(value)), (
        backend, got, reference)


def test_the_interface_cuts_refuse_a_typo_by_name():
    """``axis``, ``dx`` and ``n`` -- the preconditions, each a door with its own message.

    ``True`` is in the axis row on purpose: ``True == 1`` is how a bool reaches an integer door,
    and this repository has already paid for a bool arriving where a length was expected.

    The two ``Sphere`` overrides get the ``axis`` and ``dx`` rows and not the ``n`` one, which
    is the difference between them rather than an omission: they take a *point* and build the
    normal themselves. ``dx`` is theirs to door because they are public methods reached
    directly -- not through ``occupancy``, which doors it once for :meth:`Sphere.exact_fill` --
    and everything inside them is written in units of it. Measured with the doors removed,
    three points about a radius-0.25 ball at ``|dx| = 0.05``: a negative ``dx`` divides the
    edge's two clips by it and swaps them, so two segments that read ``1.0`` at ``+0.05`` read
    ``0.0``; the same input reads exactly the positive answer on the face, whose every quantity
    is squared; and ``dx = 0`` reads ``nan`` on the face and ``1.0`` on the edge through a clip
    of an infinity. One invalid input, three behaviours, and only one of them looks wrong.

    Mutant: remove either ``check_sizes`` call. Measured 2026-09-24, over the whole file with
    three workers: this case is the only one that goes red, for each of the two.
    """
    n = np.asarray([[0.0, 0.0, 1.0]])
    d = np.asarray([0.1])
    for call in (edge_cut, face_cut):
        assert np.isfinite(np.asarray(call(d, n, 1.0, 2))).all()
        for bad in ('z', 3, -1, True, False, 1.0, None):
            with pytest.raises(ValueError, match='axis is a component index'):
                call(d, n, 1.0, bad)
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match='a length is a size'):
                call(d, n, bad, 0)
        for bad in (2 * n, np.zeros((1, 3)), np.full((1, 3), np.nan)):
            with pytest.raises(ValueError, match='unit outward normal'):
                call(d, bad, 1.0, 0)
    points = grid(2, -0.1, 0.1)
    for name in ('exact_edge_fill', 'exact_face_fill'):
        call = getattr(Sphere(0.2), name)
        assert np.isfinite(np.asarray(call(points, 0.1, 2))).all()
        for bad in ('z', 3, -1, True, False, 1.0, None):
            with pytest.raises(ValueError, match='axis is a component index'):
                call(points, 0.1, bad)
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match='a length is a size') as caught:
                call(points, bad, 0)
            assert str(caught.value).startswith(f'Sphere.{name}(dx=...)')
        with pytest.raises(ValueError, match='a length, and there is none'):
            call(points, None, 0)


#: An exactly axis-aligned normal, an exactly in-face diagonal one, and six general ones --
#: the batch the guarded denominators are tested on. The first six make a step exactly zero.
_MIXED_NORMALS = np.concatenate([
    np.eye(3), -np.eye(3),
    np.asarray([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]]) / np.sqrt(2),
    np.asarray([[0.3, 0.48, 0.8246], [-0.7, 0.2, 0.55], [0.1, -0.9, 0.35],
                [0.62, -0.31, -0.7], [-0.2, -0.4, 0.88], [0.5, 0.5, 0.5]])
    / np.linalg.norm([[0.3, 0.48, 0.8246], [-0.7, 0.2, 0.55], [0.1, -0.9, 0.35],
                      [0.62, -0.31, -0.7], [-0.2, -0.4, 0.88], [0.5, 0.5, 0.5]],
                     axis=-1)[:, None]])


@pytest.mark.parametrize('backend', backends())
def test_the_interface_cuts_differentiate_where_a_step_is_exactly_zero(backend):
    """The guarded denominators, tested the only way they can be: backwards.

    An exactly axis-aligned normal makes ``a_1`` exactly zero for the edge and ``a_2`` exactly
    zero for the face, and every division by a step in these two forms is written with the
    denominator replaced by one **before** it is used. Take the guards out and nothing changes
    forwards -- the ``where`` that made the branch dead still discards its value -- while the
    dead branch's *cotangent* is ``nan``, which poisons the whole sum. This file already
    carries three rows of that family from the ball clip; these are the fourth and fifth.

    The batch mixes six exactly axis-aligned normals and three exactly in-face diagonal ones
    with six general ones, so the loss has a real gradient (measured -822.196 against a value
    of 43.808) rather than the zero a batch of degenerate rows alone would give -- a gradient
    of zero cannot fail an AD-against-finite-difference comparison. Measured 2026-09-24: the
    gap to the central difference is 6.49e-8 on JAX and 6.49e-8 on Torch, against a budget of
    8.2e-5.
    """
    dx = 0.1
    n = _MIXED_NORMALS
    d = np.linspace(-0.37, 0.41, len(n)) * dx

    def loss(shift, xp, cast):
        out = None
        offsets = cast(d) + shift
        normals = cast(n)
        for axis in range(3):
            e = edge_cut(offsets, normals, dx, axis)
            f = face_cut(offsets, normals, dx, axis)
            piece = xp.sum(e * e) + 1.7 * xp.sum(f * f)
            out = piece if out is None else out + piece
        return out

    reference = float((loss(H_STEP, np, np.asarray) - loss(-H_STEP, np, np.asarray))
                      / (2 * H_STEP))
    assert abs(reference) > 1.0, 'a gradient of zero would make this test vacuous'
    if backend == 'numpy':
        pytest.skip('numpy is the reference the other two are held to')
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        got = float(jax.grad(lambda s: loss(s, jnp, jnp.asarray))(jnp.asarray(0.0)))
        value = float(loss(jnp.asarray(0.0), jnp, jnp.asarray))
    else:
        torch = pytest.importorskip('torch')
        shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        out = loss(shift, torch, lambda v: torch.tensor(np.asarray(v, dtype=float)))
        out.backward()
        got, value = float(shift.grad), float(out.detach())
    assert np.isfinite(got), (backend, got)
    assert abs(got - reference) <= RTOL * abs(reference) + FTOL * abs(value), (backend, got,
                                                                              reference)
    _each_divided_difference_is_guarded_at_a_zero_step(backend)


def _each_divided_difference_is_guarded_at_a_zero_step(backend):
    """The two helpers' own denominators, called directly at ``a = 0``.

    **This paragraph is a correction, and the correction is the measurement.** It used to say
    that the loss above cannot reach these denominators, because ``_edge_parts`` pins their
    argument to the dead end of a ``clip`` whose transpose is a ``select`` -- a select
    *discards* a NaN cotangent instead of multiplying it by zero -- so the caller masked its
    own guard. That was true when it was written and is not true as landed: the same commit
    deleted ``_edge_parts``' clip, having measured it a no-op at depths from 1 to 1e12 voxels,
    and deleting it took the mask with it.

    Re-measured 2026-09-24, the mutant that removes ``_first_difference``'s safe denominator:
    the loss above now returns a **NaN gradient** on both JAX and Torch, and
    ``assert np.isfinite(got)`` fires before this helper is ever called -- two red cases,
    ``[jax]`` and ``[torch]``, and nothing else in the file. The remaining ``where(a > 0, ...)``
    at the end of ``_edge_parts`` does not mask it: a ``where``'s unselected branch is handed a
    zero cotangent, and zero times the infinity that ``u**3 / 0`` differentiates to is NaN,
    which is the whole reason the safe denominator is written before the division rather than
    after it.

    This check is kept anyway, and not as a second argument about the caller: it states the
    invariant where it lives -- **every division by a step in these two forms is finite in
    reverse mode at a zero step**, whatever the caller happens to pass -- and it costs
    milliseconds. It is also what would still fail if some later caller re-introduced a mask.
    """
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        for f in (_first_difference, _first_difference_square):
            for t in (-0.3, 0.0, 0.7):
                for a in (0.0, 0.5):
                    slope = float(jax.grad(lambda x, f=f, a=a: f(x, jnp.asarray(a), jnp))(
                        jnp.asarray(t)))
                    assert np.isfinite(slope), (f.__name__, t, a, slope)
    else:
        import torch
        for f in (_first_difference, _first_difference_square):
            for t in (-0.3, 0.0, 0.7):
                for a in (0.0, 0.5):
                    x = torch.tensor(t, dtype=torch.float64, requires_grad=True)
                    f(x, torch.tensor(a, dtype=torch.float64), torch).backward()
                    assert np.isfinite(float(x.grad)), (f.__name__, t, a, float(x.grad))


def test_an_axis_aligned_normal_is_a_slab_on_the_edge_and_on_its_dual_face():
    """The degenerate limits, against numbers written out by hand rather than by an oracle.

    A normal along ``+z`` is a slab, and every fraction it produces can be read off without any
    of this file's machinery. With ``dx = 0.1`` and the plane at ``d``:

        cell                       a_1          fraction
        x edge,  n = +z            0            1 if d <= 0 else 0
        z edge,  n = +z            dx           clip(1/2 - d/dx, 0, 1)
        face _|_ z, n = +z         0            1 if d <= 0 else 0
        face _|_ x, n = +z         dx (a_2 = 0) clip(1/2 - d/dx, 0, 1)

    The first and third rows are the whole reason ``_decided`` exists: the cell lies *in* a
    plane parallel to the interface, no divided difference can say which side of it the cell is
    on, and the measure of the closed half space -- **1 at ``d = 0``** -- answers it. The second
    and fourth are the other limit: the face's two steps are ``dx`` and exactly zero, and
    ``_first_difference_square`` has to collapse onto the one-dimensional answer without a
    tolerance deciding that it should.

    The last row of each block is the deep cell: at ``d = -1e6 dx`` the fraction is **exactly**
    1.0 and at ``+1e6 dx`` exactly 0.0, which is what the clip on ``c`` buys the face (measured
    unclipped: 1 +/- 2.3e-9 at that depth, and 1 +/- 1.9e-3 at 1e12 voxels).
    """
    dx = 0.1
    for k in range(3):
        n = np.zeros((1, 3))
        for sign in (1.0, -1.0):
            n[0, k] = sign
            for d, along in ((-0.3 * dx, 0.8), (0.3 * dx, 0.2), (0.0, 0.5),
                             (0.5 * dx, 0.0), (-0.5 * dx, 1.0),
                             (-1e6 * dx, 1.0), (1e6 * dx, 0.0)):
                offset = np.asarray([d])
                across = 1.0 if d <= 0 else 0.0
                for axis in range(3):
                    # the edge along the normal's own axis sees the slab; the other two are
                    # parallel to it and are wholly in or wholly out
                    want = along if axis == k else across
                    got = float(edge_cut(offset, n, dx, axis)[0])
                    assert abs(got - want) <= _slab_budget(want), (k, sign, d, axis, got)
                    # the dual face normal to the normal's axis is the parallel one here
                    want = across if axis == k else along
                    got = float(face_cut(offset, n, dx, axis)[0])
                    assert abs(got - want) <= _slab_budget(want), (k, sign, d, axis, got)


def _slab_budget(want):
    """Exactly 0 and exactly 1 are exact; a fraction of a voxel carries one ulp of ``d/dx``.

    The two ends are the claim -- a cell the interface does not reach is not ``1 - 2e-16`` of a
    cell, because the rule that reads this divides by ``1 - f``. In between,
    ``(-d + a/2)/a`` is three roundings of numbers that are not binary fractions, and 0.8 comes
    back as 0.7999999999999999.
    """
    return 0.0 if want in (0.0, 1.0) else 1e-15


def test_neither_interface_cut_ever_leaves_the_unit_interval():
    """A fraction outside ``[0, 1]`` is not a rounding detail, and one of them used to return one.

    The rule these two feed divides by ``f`` and by ``1 - f``, so a fraction an ulp above one
    is a sign change in a denominator, not a small error. ``_bodies.Sphere``'s own docstrings
    say it in as many words -- the world-coordinate spelling of its exact chord was rejected
    for returning ``1 + 8.88178e-16`` -- and :func:`face_cut` was landed returning
    ``1 +/- 2.220446e-16`` itself: its divided difference at ``c = total`` evaluates
    ``(2 a_1 + 2 a_2 - a_2 - a_2) / (2 a_1)``, one in exact arithmetic and one plus or minus an
    ulp in three roundings. Measured before the fix, over the same 2000 unit normals used
    below and at each of five depths: **151 of them** read ``1 + 2.220446e-16`` and 151 read
    ``1 - 2.220446e-16``, and the worst over a slide of the plane across the cell was
    ``1.0000000000000002``. The edge never did, and that is its form and not luck:
    :func:`_first_difference`'s top piece is the literal constant and its middle piece divides
    ``t <= a`` by ``a``.

    The claim is therefore stronger than an interval: a cell the plane does not reach is
    **exactly** 0 or 1, which is what ``_slab_budget`` below already demands of the
    axis-aligned family and what this asserts for every normal. Five depths inside and five
    outside (1 to 1e12 voxels), plus the plane slid across the cell at 500 offsets for each of
    2000 normals and all three axes -- 3e6 fractions per form -- for the interval itself.

    Mutant: drop the ``offset >= total`` / ``offset <= 0`` select from ``_face_parts``, leaving
    the clip on ``c`` that was there before. This goes red on the first depth, naming the
    rows; the oracle comparisons beside it do not, because they are all at 1e-14.
    """
    dx = 0.1
    rng = np.random.default_rng(7)
    n = rng.normal(size=(2000, 3))
    n = n / np.linalg.norm(n, axis=-1)[:, None]
    for depth in (1., 10., 1e3, 1e6, 1e12):
        for sign in (-1., 1.):
            d = np.full(len(n), sign * depth * dx)
            for axis in range(3):
                for got in (np.asarray(edge_cut(d, n, dx, axis)),
                            np.asarray(face_cut(d, n, dx, axis))):
                    assert got.min() >= 0.0 and got.max() <= 1.0, (depth, sign, axis,
                                                                   got.min(), got.max())
                    # a cell that deep is decided: nothing in between is allowed either
                    assert np.isin(got, (0.0, 1.0)).all(), (depth, sign, axis)
    offsets = np.linspace(-3, 3, 500) * dx
    worst, least = 0.0, 1.0
    for row in n:
        broadcast = np.broadcast_to(row, (len(offsets), 3))
        for axis in range(3):
            for got in (np.asarray(edge_cut(offsets, broadcast, dx, axis)),
                        np.asarray(face_cut(offsets, broadcast, dx, axis))):
                worst, least = max(worst, got.max()), min(least, got.min())
    assert worst == 1.0 and least == 0.0, (worst, least)
