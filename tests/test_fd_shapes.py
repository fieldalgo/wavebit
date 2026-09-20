"""Gate 1 -- the geometry kernel, before any solver touches it.

``wavebit/fd/_shapes/`` answers one question, per body and per voxel: what fraction of you is
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
* *The RCWA path's own routines.* ``wavebit/fd/_shapes/_exact.py`` is a port, so it is held against
  its source -- measured bit-identical for the disc and the polygon area, 5.6e-17 for the edge
  moments.

**The mutants**, each edited into ``wavebit/fd/_shapes/``, run against this file, watched to
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

The last eight are this round's, and six of them are mutants that **survived** the previous
gate. The first of those decided the round: the convexity tolerance as shipped *was* absolute
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
"""
import numpy as np
import pytest

from wavebit.fd._shapes import (Box, Cylinder, Difference, Intersection, Prism, Sphere, Union,
                                clip_capacity, clipped_ball_volume, clipped_disc_area,
                                clipped_polygon_area, cut, cut_parts, image_shifts, occupancy,
                                polygon_edge_moments)
from wavebit.fd._shapes._bodies import (_loud_if_not_a_size, _loud_if_not_counter_clockwise,
                                        polygon_shape)
from wavebit.fd._shapes._cut import _second_difference

from fd_helpers import backends

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
    """``0 <= fill <= 1``, ``|sum_c w_c - 1| < 1e-14`` where there is a surface, ``w = 0`` where not."""
    h = 0.05
    pts = grid(16, -0.4, h)
    bodies = {
        'sphere': Sphere(0.26, (0.013, -0.021, 0.007)),
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
        assert fill.max() <= 1.0 + 1e-14, name
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
              'csg': lambda npr: 2e-15, 'sphere': lambda npr: 2e-2,
              'cylinder': lambda npr: 5e-2}
    bodies = {
        'box': lambda R, c: Box((1.7 * R, 1.3 * R, 1.9 * R), c, 0.37),
        'prism': lambda R, c: Prism(HEX, 1.6 * R, 'z', c, 0.4),
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
    from wavebit._core.scene_fdtd import (_clipped_circle_area, _clipped_rectangle_area,
                                          _rectangle_edge_moments)
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
    the defect on record at ``wavebit/_core/scene_fdtd.py``:844-846.
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
    # A cap landing EXACTLY on a voxel face belongs to the voxel it enters, once. The interval
    # is half open for that reason: measured with two open ends (the first writing) the total
    # cap area of a hexagonal prism whose caps sit on a face read 0.000000 against a truth of
    # 0.020000 -- the interface left the grid altogether -- and with two closed ends it is
    # 0.040000, counted by both neighbours. Nothing in the gate held either way.
    zs = (np.arange(-8, 9) + 0.5) * h
    column = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=-1)
    for centre, where in ((0.0, 'caps exactly on a voxel face'), (0.03, 'caps inside a voxel')):
        cap = Prism(HEX, 0.4, 'z', (0., 0., centre), 0.).exact_moment(column, h)[1][:, 2]
        assert abs(float(cap.sum()) - 0.02) < 1e-15, (where, cap.sum())
        assert int((cap > 0).sum()) == 2, (where, cap)


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
        sphere                    668    0.0000      3.3e-03
        box, axis aligned         825    0.4455      3.3e-03
        box, rotated             1102    0.3028      3.0e-03
        hexagonal prism edge     1485    0.3222      1.6e-03
        cylinder cap rim         1485    1.0000      2.0e-03
        CSG (sphere minus rod)   2301    0.5660      3.1e-02

    The cylinder's rim is the worst case and the one C2 has to price: ``w`` swaps from the side
    wall's ``(0.5, 0.5, 0)`` to the cap's ``(0, 0, 1)`` inside one 0.0025-cell move, a jump of
    the full range. G4 was measured on a sphere, where this file reads 0.0000, and is out by
    five orders of magnitude on a rim.

    **The last row is new and it is not a jump in ``w``.** A CSG node's *fill* is discontinuous
    too, which no earlier measurement had looked for: the tangent plane's fill depends on the
    **selected** normal, and at a crease the selection swaps between children, so ``fill`` jumps
    although ``d`` is continuous. Bisected to 5.5e-17 in the slide parameter, the fill steps
    0.96063617 -> 0.99217239, a jump of **3.15e-02**, while ``d`` holds to twelve digits. Every
    body with an exact fill is immune (their fill never reads a normal); every CSG node has it.

    The consequence is worse than a disagreement between a gradient and a difference: at that
    slide value a central difference of a loss reads 268.81, 2685.85, 26855.47, 268551.57 and
    2685512.50 at steps of 1e-3 down to 1e-7 -- it **diverges as 1/step**, because the loss
    itself steps. That is what makes a CSG node's geometry a refused design variable
    (:func:`test_a_traced_parameter_on_a_csg_node_is_refused_by_name`), and why
    ``_bodies._Combination`` prices the fix rather than merely naming it.
    """
    h = 0.1
    point = np.zeros((1, 3))
    step = 0.0025 * h
    direction = np.array([0.37, 0.61, 0.70])
    direction = direction / np.linalg.norm(direction)
    cases = {
        'sphere': (lambda c: Sphere(0.31, c), tuple(0.31 * direction), 1e-12, 1e-2),
        'box axis': (lambda c: Box((0.34, 0.26, 0.29), c, 0.0), (0.17, 0.13, 0.145), 0.3, 1e-2),
        'cyl rim': (lambda c: Cylinder(0.31, 0.44, 'z', c), (0.31, 0.0, 0.22), 0.9, 1e-2),
        'csg': (lambda c: Difference(Sphere(0.31, c), Cylinder(0.12, 1.0, 'z', c)),
                (0.12, 0.0, 0.0), 0.4, 5e-2),
    }
    for name, (make, start, want_w, want_f) in cases.items():
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
            assert step_w < want_w, (name, step_w)   # smooth surface, smooth moment
        else:
            assert step_w > want_w, (name, step_w)   # a crease, and this is how big
        assert step_f < want_f, (name, step_f)
        if name == 'csg':
            # the fill steps too, which is the new finding and not a matter of resolution
            assert step_f > 3e-2, step_f
        else:
            assert step_f < 4e-3, (name, step_f)     # every exact fill is continuous
    # the CSG fill's own jump, bisected rather than sampled
    make = cases['csg'][0]

    def fill_at(s):
        centre = np.asarray(cases['csg'][1]) + direction * s
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
    assert abs(fill_at(hi) - fill_at(lo)) > 3e-2, (fill_at(lo), fill_at(hi))


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
    brute = np.sqrt(((pts[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(-1)
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
        d, n = body.surface(pts - np.asarray(shift))
        nearest = np.where(d == best, cut_parts(d, n, h)[0], nearest)
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
    (``wavebit/_core/scene_fdtd.py``:802) -- makes both read 1.0000, and the only door that
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
    lesson, ``wavebit/_core/parametric.py``:348-355). It is accepted here because the stage-4
    API names it; C5 should refuse it on key *presence* as it does for a sphere, or the
    cross-section should become an ellipse. The rotation itself is not dead code: the same line
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
    pts = grid(10, -0.4, 0.08)
    for body_of, value in ((lambda r: Sphere(r, CENTRE), 0.22),
                           (lambda r: Cylinder(r, 0.3, 'z', CENTRE, 0.3), 0.19),
                           (lambda v: Prism(v, 0.33, 'z', CENTRE, 0.0), np.asarray(HEX))):
        def run(r, p):
            return occupancy(body_of(r), p, 0.08)
        eager = run(jnp.asarray(value), jnp.asarray(pts))
        traced = jax.jit(run)(jnp.asarray(value), jnp.asarray(pts))
        for a, b in zip(eager, traced):
            assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-14
    # a CSG node: concrete geometry, traced points -- what a forward solve does
    node = Difference(Sphere(0.3), Box((0.2, 0.2, 0.9), (0., 0., 0.), 0.5))
    eager = occupancy(node, jnp.asarray(pts), 0.08)
    traced = jax.jit(lambda p: occupancy(node, p, 0.08))(jnp.asarray(pts))
    for a, b in zip(eager, traced):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-14


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
    for rubbish in ([[0., 0.], [1., 0.], [1., 1., 5.]], 'abc', [1.0, 2.0, 3.0], None):
        with pytest.raises(ValueError, match=r'shape \(nv, 2\)'):
            Prism(rubbish, 0.33, 'z')
    # every legal outline in this file passes, including the six non-convex ones
    for outline in dict(NON_CONVEX, hexagon=HEX).values():
        Prism(outline, 0.33, 'z')


def test_every_geometric_predicate_is_free_of_the_coordinate_scale():
    """The same body in metres and in micrometres is the same body, or nothing here is safe.

    The C1 blocker came back once already because the convexity test compared a cross product
    against ``1e-12 * max(|cross|, initial=1.0)``: an **absolute** floor, so every polygon whose
    coordinates are below about 3e-6 was classified convex, took the ``nv + 4`` clip buffer and
    overflowed it. ``wavebit/fd/scene.py`` fixes units as "lengths in the user's unit", so a
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
        'torch.func.vmap': lambda: torch.func.vmap(
            lambda r: fill_of(-r, torch.tensor(pts)))(torch.tensor([0.3, 0.4])),
    }
    for where, call in refusals.items():
        with pytest.raises(ValueError, match=match):
            call()
    # the one that cannot be a refusal, and what stands there instead
    under_jit = np.asarray(jax.jit(lambda r: fill_of(-r, jnp.asarray(pts)))(jnp.asarray(0.3)))
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
    turned = np.asarray(jax.jit(lambda v: occupancy(Prism(v, 0.33, 'z'), jnp.asarray(pts),
                                                    0.1)[0])(jnp.asarray(HEX[::-1])))
    assert np.isnan(turned).all(), turned
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
