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
    `exact_fill` ignored, the plane used for every body                2
    the transverse centre subtracted twice (the bug that was here)     4

The second one is worth reading twice: removing the guards leaves ``0/0`` in a branch that
``xp.where`` never selects, so **every forward value is still right** and only two tests go
red -- the gradient of a cylinder's height on JAX and on Torch, where an axis-aligned normal
makes the dead branch's cotangent a NaN. A gate of forward values alone would have passed it.

**The cost, measured on this machine** (2026-09-20, CPU, 6 cores, ``OMP_NUM_THREADS=2``), for
one component of a 48^3 support box at ``dx = 0.0125`` -- 110592 voxels, which is a sphere of
20 cells of radius with room around it:

    body                      numpy     jax (jit)    torch
    Sphere (plane)            49.1 ms     4.8 ms    17.6 ms
    Cylinder (exact fill)    112.1 ms     5.9 ms    40.4 ms
    Difference (CSG, plane)   51.8 ms     5.0 ms    19.3 ms

and the periodic ring multiplies it by the image count, as it must: on NumPy the sphere reads
33.9 ms with no ring, 289.2 ms over two periodic axes (9 images) and 819.5 ms over three (27).
That is the price of the static ``rings=`` prune C2 will make from a declared ``range=``.
Memory, NumPy, same box: 23.3 MB peak against 2.53 MB of points and 3.38 MB returned, i.e.
about nine times the points -- roughly 21 live float64 temporaries per voxel, which is what an
unfused NumPy expression of this length costs and what ``jax.jit`` fuses away.
"""
import numpy as np
import pytest

from wavebit.fd._shapes import (Box, Cylinder, Difference, Intersection, Prism, Sphere, Union,
                                clipped_disc_area, clipped_polygon_area, cut, cut_parts,
                                image_shifts, occupancy, polygon_edge_moments)
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


def test_w_vanishes_only_where_the_fill_is_already_decided():
    """The one discontinuity in ``w`` is at a voxel the mixing rule does not read ``w`` in.

    ``w`` steps from ``n_c^2`` to zero when the last of the surface leaves the voxel. That is
    allowed only because the fill there is 0 or 1, where
    ``1/eps = w(f/e_a + (1-f)/e_b) + (1-w)/(f e_a + (1-f) e_b)`` is ``1/e_b`` or ``1/e_a``
    whatever ``w`` is. If this pairing ever breaks, the step becomes a step in a coefficient.

    It is **exact** where the fill and the moment come from the same route, and merely tight
    where they do not. A :class:`Cylinder` takes its fill from a closed form and its moment
    from the plane, and the two can disagree about whether a voxel is cut: the plane misses a
    voxel once ``|d| > (sum_i |n_i| dx)/2``, while a curved surface's own corners may still
    poke into it. Measured worst gap from 0 or 1 where ``w = 0``, over a body of 4 / 8 / 16
    cells of radius:

        sphere, prism, box, CSG difference   6.7e-16   2.6e-15   5.6e-15
        cylinder (exact fill, plane moment)  8.1e-08   1.3e-14   1.1e-13

    so the coefficient a stage-4 cell would get is wrong there in the eighth digit at four
    cells of radius, and the effect shrinks. It is recorded rather than removed because the
    remedy is the cylinder's exact moment, which is C2's decision, not a tolerance.
    """
    h = 0.05
    pts = grid(16, -0.4, h)
    exact_route = (Sphere(0.26, (0.013, -0.021, 0.007)),
                   Box((0.34, 0.22, 0.29), (0.013, -0.021, 0.007), 0.7),
                   Prism(HEX, 0.33, 'z', (0.01, -0.02, 0.0), 0.4),
                   Difference(Sphere(0.3, (0., 0., 0.)), Sphere(0.14, (0.05, 0., 0.))))
    for body in exact_route:
        fill, w = occupancy(body, pts, h)
        dead = w.sum(-1) == 0
        assert np.abs(np.minimum(fill[dead], 1 - fill[dead])).max() < 1e-14, body
        assert dead.sum() > 100 and (~dead).sum() > 100
    fill, w = occupancy(Cylinder(0.21, 0.33, 'z', (0.01, -0.02, 0.0)), pts, h)
    dead = w.sum(-1) == 0
    assert np.abs(np.minimum(fill[dead], 1 - fill[dead])).max() < 1e-6


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

    **(M)** Restoring that double subtraction (``q = points - centre`` in ``_Extrusion._split``
    while ``_vertices`` stays absolute) fails this and four other tests.
    """
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.6, 0.6, size=(600, 3))
    prism = Prism(HEX, 0.4, 'z', CENTRE_D, 0.2)
    samples = _prism_surface()
    brute = np.sqrt(((pts[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(-1)
    cases = ((Sphere(0.31, CENTRE_D), _sphere_distance(pts), 1e-15),
             (Box((0.4, 0.28, 0.33), CENTRE_D, 0.6), _box_distance(pts), 1e-14),
             (Cylinder(0.24, 0.4, 'x', CENTRE_D), _cylinder_distance(pts), 1e-14),
             (prism, np.sign(prism.surface(pts)[0]) * brute, 2e-3))
    for body, want, tol in cases:
        d, n = body.surface(pts)
        assert np.abs(d - want).max() < tol, (body, np.abs(d - want).max())
        assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 1e-14, body
        # the normal is the gradient: a central difference of d must reproduce it
        fd = np.stack([(body.surface(pts + e * 1e-6)[0] - body.surface(pts - e * 1e-6)[0]) / 2e-6
                       for e in np.eye(3)], axis=-1)
        keep = np.abs(np.linalg.norm(fd, axis=-1) - 1) < 1e-4       # off the medial axis
        assert keep.sum() > 400, body
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
    """``4 pi R^3/3`` through the 3x3x3 ring, to the plane's own bias and no more.

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
    assert abs(with_ring / exact - 1) < 4.2e-3            # the plane's +0.250 (h/R)^2 at R/h = 8
    assert with_ring / exact - 1 > 0                      # and it is the over-fill, not a loss
    assert without / exact - 1 < -0.8
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
    body = Sphere(0.3, (0.05, 0.05, 0.5))       # the plane path, so both spellings apply
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

    **(M)** ``fill = where(fill > 1, 1, fill)`` after the image sum -- the RCWA path's own line
    (``wavebit/_core/scene_fdtd.py``:802) -- makes the measured 2.0000 read 1.0000, and the
    only door that could have refused this geometry has nothing left to read.
    """
    h = 0.05
    pts = grid(20, 0.0, h)
    fill, _ = occupancy(Cylinder(0.7, 0.4, 'z', (0.5, 0.5, 0.5)), pts, h, period=(1.0, 1.0, None))
    assert fill.max() > 1.99


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
    'csg radius': (lambda v: Difference(Sphere(v, (0., 0., 0.)), Sphere(0.12, (0.06, 0., 0.))), 0.25),
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
        csg radius  1.42e+03  9.2e-01  9.9e-01  1.0e-08  1.1e-10  9.6e-11

    Truncation falling as ``h^2`` to a floor near ``1e-6``, so :data:`H_STEP` is ``1e-6`` and
    the worst reading there is 3.1e-9 -- thirty times inside the budget. Two rows do not
    follow that shape and both say something: ``height`` is 1e-14 at every step because the
    axial overlap is *piecewise linear* in the height, so a central difference of it has no
    truncation at all; and ``radius`` and ``csg radius`` read O(1) at 1e-3 and 1e-4 because a
    step that large moves whole voxels across a knot, which is the same effect the design's G8
    note is about seen from the other side.

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
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    pts = grid(10, -0.4, 0.08)
    for body_of, value in ((lambda r: Sphere(r, CENTRE), 0.22),
                           (lambda r: Cylinder(r, 0.3, 'z', CENTRE, 0.3), 0.19),
                           (lambda r: Difference(Sphere(r, (0., 0., 0.)), Box((0.2, 0.2, 0.9),
                                                                              (0., 0., 0.), 0.5)),
                            0.3)):
        def run(r, p):
            return occupancy(body_of(r), p, 0.08)
        eager = run(jnp.asarray(value), jnp.asarray(pts))
        traced = jax.jit(run)(jnp.asarray(value), jnp.asarray(pts))
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
