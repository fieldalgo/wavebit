"""Gate 2, 3 and 4 -- the lowering stage that turns a body into coefficients.

``src/wavebit/fd/_shapes`` was gated on its own (``tests/fdtd/test_fd_shapes.py``): given a body and a
point, it answers with a fill fraction and a normal moment. This file gates everything between
that answer and ``be``/``bh``: **which** points each component is asked at, **what** the two
media are mixed by, and what the two together do to a number a user reads.

**The oracles, and why each is independent of the thing it judges**

* *A closed form at a known sub-cell position.* A flat z interface put at 0.3 of a cell: the
  fill of every one of the six components' voxels is a fraction this file writes down from the
  Yee offsets, and the mixed permittivity is arithmetic or harmonic by hand. Nothing in
  ``_lower/`` is consulted to produce either. That is the co-location gate: ``Ez`` sits half a
  cell from ``Ex``, so a rule that sampled all three at the cell centre gives ``f = 0.8`` where
  the truth is 0.3, and the closed form says so.
* *The continuum transfer matrix.* For the mixing rule itself, with no FDTD anywhere: a
  lamellar stack of period ``p``, the exact band at Bloch phase zero from
  ``tr(M_b M_a)/2 = 1`` bisected to 1e-13, against the same band read off a 1-D Yee
  eigenproblem whose two permittivities come from :func:`wavebit.fdtd._prepare.coefficients.mix`.
  The wave runs **along** the layers in p polarisation, so ``E_x`` is tangential (the
  arithmetic half of the rule) and ``E_z`` is normal (the harmonic half) and both halves are
  under test at once. 24 translations of the stack through one cell at each of four
  resolutions, so the claim is a fitted order and not a lucky alignment.
* *Itself, slid.* For smoothness in position there is no external oracle and none is wanted:
  the question is whether a scalar a user reads moves continuously as the geometry does, and
  the measurement is the largest single step over a sweep.
* *The same geometry lowered as one shape.* For composition, the oracle is the scene itself
  said differently: two half spaces of the same material *are* the deeper one alone, and one
  painter answers that exactly, so the two-painter lowering is compared with a one-painter
  lowering of the same geometry and no tolerance enters the comparison at all.
* *The archives.* The four frozen stage-0..3 goldens, byte compared, plus a call counter that
  says the mixing rule was never reached at all by a scene of ``Slab``s and ``Region``s.
* *Finite differences, on JAX and on Torch.* For the gradient, central differences of the same
  function the AD is taken of -- the only oracle that is not the AD.

**The mutants**, each edited into ``src/wavebit/fd/``, run against this file, watched to fail, and
reverted. The count is the number of tests in this file that went red:

    mutant                                                                  red
    all three components sampled at the cell centre (`occupancy.py`)         2
    `mu` read at the E points instead of the H points (`coefficients.py`)    1
    arithmetic and harmonic swapped in `mix`                                 6
    `w` forced to 1/3 -- an isotropic scalar volume average                  6
    the fill rounded to 0/1 in `component_occupancy`                         6
    `check_smoothed`'s conductive/dispersive door removed                    1
    `check_smoothed`'s periodic-face door removed                            1
    `Sphere.support`'s one-cell margin dropped                               2
    `Sphere.bounded_axes` copying `Region`'s "axes it does not span" rule       4
    `Box.body()` returning a `_shapes.Box` instead of None                   2

and the doors the composition round added, each mutated in both directions -- removed, and
widened until it refuses a scene that is legal:

    door 5 (two smoothed surfaces in one voxel) removed                      1
    door 5 drawn on the bounding boxes instead of the voxels                 3
    door 5 keeping the distance mark and dropping the fill test              1
    the `Axis1D` arm of door 2 removed                                       1
    door 3 (a shape the grid cannot carry whole at a face) removed           1
    door 3's uniform-across-the-face exemption dropped                       5
    door 4 (anisotropy) removed                                              1
    `_isotropic` reading a triple of distinct numbers as one                 1

and the round that made "both directions" true of the two doors it was not true of, gave every
door a tolerance tied to the clip's own round-off, and wired the kernel's bound report. Those
run against this file **and** `test_fd_shape_protocol.py` (baseline 63 passed), because
`Sphere.support` is a protocol answer and both files read it:

    door 5's fill tolerance dropped (comparing against 0 and 1 themselves)   1
    door 5's fill tolerance widened to 1e-3                                  1
    `_VOXEL_REACH` cut from half a voxel DIAGONAL to half an EDGE            3
    `_VOXEL_REACH` widened to two cells                                      2
    door 2's `Bloch` arm removed (`(Periodic, Bloch)` -> `(Periodic,)`)      1
    door 2's wrapping arm widened to every face                             14
    door 3's two arms given each other's message                             2
    door 3's `+` arm skipping the components whose point is on the wall      2
    door 3's threshold slackened to 1.0 / tightened to 0.0                   2 / 7
    door 1's self clause removed / used for every medium                     1 / 1
    `Sphere.support`'s margin dropped / dropped on the high side only        9 / 9
    `Approximations.record` doing nothing                                    1
    an empty `Approximations` warning anyway                                 1
    the bound width summed instead of maximised                              1
    the report not threaded to the body at all                               1

None of those eighteen survived; the two of them that had no scene to die on before -- the
half-voxel reach and the ``Bloch`` arm -- are what
`test_the_shared_surface_mark_reaches_a_voxel_diagonal_and_not_an_edge` and the parametrized
periodic-face gate exist for.

And the close-out round, which split door 3 by CAUSE rather than by side, gave the clip
tolerance the grid it is read on, and read the `w` mask from the predicate's own input. It was
written against C2 and landed after earlier bounded C8 work. The following table records
that historical fe92caa0 round-7 run against this file, `test_fd_shape_protocol.py` and
`test_fd_symmetry_door.py` (93 cases); it is not the current integration's rerun count:

    every probe counted as wholly outside (no straddling arm at all)          2
    split by SIDE (the `+` face's wholly-outside probes read as straddling)   2
    `_TRIMMED` returned in preference to `_LEAVES`                            2
    the two arms' messages swapped                                            2
    `_clip_tolerance` = `_CLIP_ROUNDOFF` alone (the constant as it stood)     1
    `_clip_tolerance` = the relative arm alone, with no floor                 2
    `_LIVE_AREA` raised to 1e-3 / dropped to 0                                1 / 0

The second and third are the ones the round exists for: split by side,
`fd.Sphere(radius=0.2, center=(0, 0.59, 0))` -- three probe voxels wholly outside the domain,
at fill 1.0, with 3.8 cells of ball past the wall -- was told that "nothing outside the box is
being counted here" and that "there is nothing sticking out to truncate", and the suite as it
stood asserted that message on that very scene. **`_LIVE_AREA = 0` survives the recorded gates**: over the moment gate's sweep (``dx``
0.2 to 0.0125, four placements) no sampled voxel has an area in ``(0, 1e-12 dx^2]``. A
separately constructed tangent plane cutting a corner off by ``1e-8 dx`` has area
``6.9e-16 dx^2``; its nonzero ``w = moment / area`` stays finite and sums to 1 to the last
digit. That quotient check does not make the threshold change equivalent: the production
threshold zeros this constructed moment, and the recorded gates do not exercise that case.

The same re-run of the earlier rounds' door-3 rows, whose code the close-out rewrote: door 3
removed 2, its uniform-across exemption dropped 6, its `+` arm skipping the components on the
wall 2, its threshold slackened to 1.0 2 -- and **tightened to 0.0, 0**: the end door 3 reads
is the one a body misses, and that end is exactly 0.0 on this kernel in every scene here
(``test_the_clip_tolerance_follows_the_grid_and_not_a_number`` measures it), so no tolerance
at that end refuses or admits anything the bare comparison would not. Door 3 answering at a
``Symmetry`` face fails ``test_fd_symmetry_door.py`` once.

Two mutants of the earlier rounds **survived a first draft**, and both notes are left where
they were fixed.
``test_a_ball_varies_along_every_axis``: with only a small ball in the scene `Region`'s
`bounded_axes` rule gives the same answer, so the gate had to grow a ball wide enough for its
support to clip to the whole grid (one cell short of inscribed, so that door 3 does not answer
first). ``test_two_smoothed_bodies_a_voxel_apart_compose_in_either_order``: the dimer on the
45 degree diagonal is 1.9 cells apart, so the door's cheap first stage never marks a voxel at
all and its second stage is never reached -- a pair one whole cell apart, marked in 36 voxels
and sharing none of them, is what makes the fill test load-bearing. That is the
surviving-mutant rule earning its keep (memory ``mutate-the-code-to-test-the-tests``).

The cost of the stage is measured rather than asserted, and lives in this module's own
``COST`` table below -- an assertion on wall clock is a flake on a shared machine, and the
number a reader wants is what a sphere costs, not that it is under a threshold.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._kernel.grid import OFFSETS
from wavebit.fdtd._prepare import coefficients as coef
from wavebit.fdtd._prepare.occupancy import component_occupancy, occupancy, yee_points
from wavebit.fdtd._runtime import runtime
from wavebit.fdtd._shapes import Box as BoxBody
from wavebit.fdtd._shapes import Sphere as BallBody
from wavebit.fdtd._shapes import occupancy as body_occupancy
from wavebit.fdtd._legacy_geometry import AXES, Shape

from tests.fixtures.fdtd.fd_helpers import backends

#: What this stage costs, measured on the machine this commit was written on, for one
#: ``fd.Sphere(radius=0.3)`` at **12 cells per radius** in a 64^3 box with a 12-cell PML on
#: every face -- ``fd.lower`` only, best of three, against the same scene with the ball
#: replaced by a cell-snapped ``fd.Region`` of the same bounding size::
#:
#:     backend    vacuum    snapped Region    one Sphere    x Region    RSS delta
#:     numpy     0.0005 s          0.0066 s      0.4211 s         64x       6.2 MB
#:     jax       0.0040 s          0.0074 s      0.6892 s         93x       6.3 MB
#:     torch     0.0009 s          0.0050 s      0.3112 s         62x       6.3 MB
#:
#: Three things a reader should take from it.
#:
#: **The peak memory is the coefficient arrays and nothing else.** 6.3 MB is exactly one
#: ``(3, 64, 64, 64)`` float64 array, and the snapped arm pays the same -- so the fills add
#: nothing measurable, because they live over the support box (26^3 voxels, 0.14 MB per
#: component) and not over the grid. That is what ``_Smoothed`` touching only its box buys.
#:
#: **The time is the exact clipped-ball fill, not the rule and not the moment.** Per component
#: array of 17 576 voxels on NumPy: the points 0.0003 s, ``exact_fill`` **0.1181 s**, the
#: tangent plane that carries the moment 0.0032 s. The fill is 97% of it, at 6.7 us a voxel.
#:
#: **A magnetic material doubles it**, because ``mu`` is read at the H points: E only against
#: E and H, 2.04x / 1.97x / 2.13x on numpy / jax / torch. That is why ``coefficients.build``
#: asks for the H pass only when something in the scene is magnetic.
COST_IS_MEASURED = True

DX = 0.05
N = 24


# ---------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------

def _grid_scene(structures, n=N, dx=DX, steps=4):
    """A closed PEC box: no absorbing layer, no source, so nothing but geometry is under test."""
    return fd.Scene(
        domain=fd.Domain(size=(n * dx,) * 3),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps), structures=structures)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def _lowered(scene, rt='numpy'):
    p = _quiet(fd.plan, scene)
    return p, _quiet(fd.lower, scene, p, rt)


def _eps_of(plan, params):
    """``eps`` per component, recovered from ``be`` exactly as the kernel would read it."""
    return (plan.dt / plan.dx) / np.asarray(params['be'])


def _mu_of(plan, params):
    return (plan.dt / plan.dx) / np.asarray(params['bh'])


# ---------------------------------------------------------------------------------------------
# A test-local shape: a half space with a flat z face, spelled through the protocol
# ---------------------------------------------------------------------------------------------

class _HalfSpace(Shape):
    """Everything below ``top``, smoothed -- the one geometry with a closed-form fill.

    A ``Sphere`` cannot put a flat interface at a named sub-cell position, and that is what the
    co-location gate needs: a fill this file computes by hand from the Yee offsets. So it is
    spelled as a ``_shapes.Box`` far wider than the domain on x and y, reaching below it on z,
    which is the body C4 made public as ``fd.Box``, doing what it ships for. The protocol is the public door; this
    class answers it and nothing here reaches inside the package.
    """

    def __init__(self, top, wide=40.0):
        self.top, self.wide = float(top), float(wide)

    def body(self):
        return BoxBody(size=(self.wide, self.wide, self.wide),
                       center=(0.0, 0.0, self.top - self.wide / 2))

    def support(self, grid):
        k1 = min(max(int(math.ceil((self.top - grid.low[2]) / grid.dx)) + 1, 0), grid.n[2])
        return ((0, grid.n[0]), (0, grid.n[1]), (0, k1))

    def uniform_along(self, axis, cells, grid):
        """No interface anywhere but the one cell layer its face lands in.

        The obvious "does my support reach these layers" answer is **wrong for this shape**
        and the wrongness is load-bearing from C2 on: a half space is uniform along x and y
        everywhere, and uniform along z everywhere except at its own face, so it may leave the
        box on five of its six faces without losing anything to the clip. That is exactly the
        exemption ``checks._reaches_outside`` reads, and a conservative ``False`` here would
        have this shape refused for running out of the domain -- which is what it is for.
        """
        if axis != 2:
            return True
        edge = int(math.floor((self.top - grid.low[2]) / grid.dx))
        return cells[1] <= edge - 1 or cells[0] >= edge + 2

    def bounded_axes(self, grid):
        return (0, 1, 2)

    def describe(self):
        return f'HalfSpace(top={self.top})'

    def span_rows(self, grid):
        return (('half.z', -self.wide, self.top, -self.wide, self.top),)

    def min_extent(self, grid):
        return self.wide


class _Ball(Shape):
    """A ``Sphere`` whose radius may be **traced**, with a static declared support box.

    This is the shape C6 will ship as ``fd.Sphere(radius=..., range=...)``: the radius is a
    leaf, the support box is declared and static, and everything between them is one body of
    code. C2's public ``fd.Sphere`` keeps a static radius because nothing yet spells the
    declaration; the chain from the geometry to ``be`` is already differentiable, and this
    class is how that is gated without inventing the public keyword a commit later owns.
    """

    def __init__(self, radius, center=(0.0, 0.0, 0.0), bound=None):
        self.radius, self.center = radius, tuple(float(c) for c in center)
        self.bound = float(bound if bound is not None else radius)

    def body(self):
        return BallBody(radius=self.radius, center=self.center)

    def support(self, grid):
        out = []
        for a in range(3):
            k0 = math.floor((self.center[a] - self.bound - grid.low[a]) / grid.dx) - 1
            k1 = math.ceil((self.center[a] + self.bound - grid.low[a]) / grid.dx) + 1
            out.append((min(max(int(k0), 0), grid.n[a]), min(max(int(k1), 0), grid.n[a])))
        return tuple(out)

    def uniform_along(self, axis, cells, grid):
        k0, k1 = self.support(grid)[axis]
        return k1 <= cells[0] or k0 >= cells[1]

    def bounded_axes(self, grid):
        return (0, 1, 2)

    def describe(self):
        return f'Ball(bound={self.bound}, center={self.center})'

    def span_rows(self, grid):
        return tuple((f'ball.{axis}', self.center[a] - self.bound, self.center[a] + self.bound,
                      self.center[a] - self.bound, self.center[a] + self.bound)
                     for a, axis in enumerate(AXES))

    def min_extent(self, grid):
        return 2 * self.bound


# ---------------------------------------------------------------------------------------------
# G2 -- co-location
# ---------------------------------------------------------------------------------------------

def _overlap_below(top, centre, dx):
    """Fraction of ``[centre - dx/2, centre + dx/2]`` that lies below ``top``, exactly."""
    return min(max((top - (centre - dx / 2)) / dx, 0.0), 1.0)


def _closed_form(top, grid, field, c, k, eps_a, eps_b):
    """The mixed value of component ``c`` of ``field`` in cell layer ``k``, by hand.

    The interface normal is ``z``, so ``w = (0, 0, 1)``: the two in-plane components are the
    arithmetic mean and the z one is the harmonic mean, which is the axis-aligned rule the
    legacy engine writes at ``src/wavebit/fdtd/_periodic/legacy.py``:375 and which the Farjadpour/Kottke form
    reduces to exactly.
    """
    centre = grid.low[2] + (k + 0.5 * OFFSETS[field][c][2]) * grid.dx
    f = _overlap_below(top, centre, grid.dx)
    if c == 2:
        return 1.0 / (f / eps_a + (1.0 - f) / eps_b)
    return f * eps_a + (1.0 - f) * eps_b


def test_each_component_reads_the_interface_at_its_own_yee_point():
    """A z interface at 0.3 of a cell, and every component's fill read back from ``be``.

    ``Ex`` and ``Ey`` sit at the node in z and their voxel is 80% below the interface; ``Ez``
    sits half a cell above and its voxel is 30% below. A rule that sampled the cell centre for
    all three would give 0.8 three times and a plausible permittivity, which is the defect this
    repository already paid for once with ``Ey``'s pole state (memory
    ``legacy-ey-pole-fill-half-cell``, fix ``a6cb803``: a published crossed-metal absorption
    moved 0.400 -> 0.433 against an RCWA 0.431).
    """
    top = -N * DX / 2 + 12.3 * DX
    scene = _grid_scene((fd.Structure(_HalfSpace(top), fd.Material(eps=4.0)),))
    plan, params = _lowered(scene)
    eps = _eps_of(plan, params)
    for k in range(9, 15):
        for c in range(3):
            want = _closed_form(top, plan.grid, 'E', c, k, 4.0, 1.0)
            got = eps[c, k, N // 2, N // 2]
            assert got == pytest.approx(want, rel=1e-12), (c, k, got, want)
    # The half cell is visible in the answer, not only in the construction.
    assert eps[0, 12, N // 2, N // 2] != eps[2, 12, N // 2, N // 2]
    assert eps[0, 12, N // 2, N // 2] == pytest.approx(0.8 * 4.0 + 0.2 * 1.0, rel=1e-12)
    assert eps[2, 12, N // 2, N // 2] == pytest.approx(1.0 / (0.3 / 4.0 + 0.7), rel=1e-12)


def test_mu_is_read_at_the_h_points_and_not_at_the_e_points():
    """``Hx`` is half a cell along z from ``Ex``, so a magnetic body's ``mu`` must differ there.

    Same interface, same cell. ``Ex`` reads 0.8 of a cell of material and ``Hx`` reads 0.3, and
    the arithmetic/harmonic split swaps too: for a z interface ``Hz`` is the normal component
    and ``Hx``/``Hy`` are tangential.
    """
    top = -N * DX / 2 + 12.3 * DX
    scene = _grid_scene((fd.Structure(_HalfSpace(top), fd.Material(eps=4.0, mu=2.0)),))
    plan, params = _lowered(scene)
    mu = _mu_of(plan, params)
    for k in range(9, 15):
        for c in range(3):
            want = _closed_form(top, plan.grid, 'H', c, k, 2.0, 1.0)
            assert mu[c, k, N // 2, N // 2] == pytest.approx(want, rel=1e-12), (c, k)
    assert mu[0, 12, N // 2, N // 2] == pytest.approx(0.3 * 2.0 + 0.7 * 1.0, rel=1e-12)
    assert mu[2, 12, N // 2, N // 2] == pytest.approx(1.0 / (0.8 / 2.0 + 0.2), rel=1e-12)


def test_the_rule_is_the_axis_aligned_rule_at_the_two_ends():
    """``w = 1`` is harmonic and ``w = 0`` is arithmetic, to the bit, for every fill.

    That is the whole claim that stage 4's rule *contains* the legacy engine's rather than
    competing with it (``src/wavebit/fdtd/_periodic/legacy.py``:375 ``component_property``). Both ends are
    closed forms here, so the comparison is exact rather than approximate.
    """
    for f in (0.0, 0.125, 0.3, 0.5, 0.75, 1.0):
        for a, b in ((4.0, 1.0), (1.0, 12.0), (2.25, 2.25)):
            assert coef.mix(np, b, a, f, 0.0) == 1.0 / (1.0 / (f * a + (1 - f) * b))
            assert coef.mix(np, b, a, f, 1.0) == 1.0 / (f / a + (1 - f) / b)
    # and it is monotone between them: the harmonic mean never exceeds the arithmetic one
    for f in (0.1, 0.4, 0.9):
        lo = coef.mix(np, 1.0, 12.0, f, 1.0)
        hi = coef.mix(np, 1.0, 12.0, f, 0.0)
        mid = coef.mix(np, 1.0, 12.0, f, 0.5)
        assert lo < mid < hi


# ---------------------------------------------------------------------------------------------
# G3 -- the rule itself, against the continuum transfer matrix, with no FDTD
# ---------------------------------------------------------------------------------------------

PERIOD = 1.0
DUTY = 0.5
KT = 2.0


def _layer_transfer(omega, kt, eps, thickness):
    """The 2x2 transfer matrix of one layer for p polarisation (``E_t``, ``H_z``).

    From Maxwell with ``exp(-i omega t)``, ``c = mu = 1``, a field ``~ exp(i k_t t)``:
    ``d/dn (u, v) = i [[0, a], [b, 0]] (u, v)`` with ``a = omega - k_t^2/(omega eps)`` and
    ``b = omega eps``. p is the polarisation with a field component **across** the layers,
    which is the branch the subpixel rule is about.
    """
    a, b = omega - kt * kt / (omega * eps), omega * eps
    kn = np.sqrt(complex(a * b))
    if abs(kn) < 1e-13:
        return np.array([[1.0, 1j * a * thickness], [1j * b * thickness, 1.0]], complex)
    c, s = np.cos(kn * thickness), np.sin(kn * thickness) / kn
    return np.array([[c, 1j * a * s], [1j * b * s, c]], complex)


def _exact_omega(kt, eps_hi, eps_lo, steps=4000):
    """The lowest ``omega`` with ``tr(M)/2 = 1``: Bloch phase zero across the period."""
    def f(w):
        m = (_layer_transfer(w, kt, eps_lo, PERIOD * (1 - DUTY))
             @ _layer_transfer(w, kt, eps_hi, PERIOD * DUTY))
        return float(np.real(np.trace(m)) / 2.0 - 1.0)

    lo, hi = kt / (math.sqrt(eps_hi) * 1.001), kt * 1.001
    prev, w0 = f(lo), lo
    for i in range(1, steps + 1):
        w = lo + (hi - lo) * i / steps
        cur = f(w)
        if prev * cur <= 0.0:
            a, b = w0, w
            for _ in range(200):
                mid = 0.5 * (a + b)
                if f(a) * f(mid) <= 0.0:
                    b = mid
                else:
                    a = mid
            return 0.5 * (a + b)
        prev, w0 = cur, w
    raise RuntimeError('no band in the physical window')


def _stack_fill(centres, h, shift):
    """The fraction of each voxel of side ``h`` holding the high-index layer, exactly.

    The stack is ``{ (z - shift) mod p < duty p }``. Summed over enough images that a voxel
    wider than a layer is still right, and this is a closed form, not the package's.
    """
    out = np.zeros_like(centres)
    for k in range(-2, 3):
        lo = k * PERIOD + shift
        hi = lo + DUTY * PERIOD
        a = np.minimum(np.maximum(centres + h / 2, lo), hi)
        b = np.minimum(np.maximum(centres - h / 2, lo), hi)
        out = out + (a - b) / h
    return out


def _grid_omega(n, shift, eps_hi, eps_lo, rule='kottke'):
    """The band at Bloch phase zero of a 1-D Yee p-polarised operator, ``n`` cells per period.

    ``E_x`` at the nodes, ``E_z`` and ``H_y`` at the half nodes. Eliminating ``E`` gives a
    symmetric tridiagonal generalised eigenproblem in ``H_y``::

        omega^2 H[m] = k_t^2/eps_z[m] H[m]
                       + (1/h^2) [ (H[m] - H[m+1])/eps_x[m+1] + (H[m] - H[m-1])/eps_x[m] ]

    ``eps_x`` is the tangential component (``w = 0``) and ``eps_z`` the normal one (``w = 1``),
    both produced by :func:`wavebit.fdtd._prepare.coefficients.mix`, which is the thing under test.
    """
    h = PERIOD / n
    nodes = np.arange(n) * h
    halves = nodes + h / 2
    if rule == 'snapped':
        fx = np.round(_stack_fill(nodes, h, shift))
        fz = np.round(_stack_fill(halves, h, shift))
        wx = wz = 0.0
    else:
        fx, fz = _stack_fill(nodes, h, shift), _stack_fill(halves, h, shift)
        wx, wz = (1 / 3.0, 1 / 3.0) if rule == 'isotropic' else (0.0, 1.0)
    eps_x = coef.mix(np, eps_lo, eps_hi, fx, wx)
    eps_z = coef.mix(np, eps_lo, eps_hi, fz, wz)
    a = np.zeros((n, n))
    for m in range(n):
        a[m, m] = KT * KT / eps_z[m] + (1 / h ** 2) * (1 / eps_x[(m + 1) % n] + 1 / eps_x[m])
        a[m, (m + 1) % n] += -(1 / h ** 2) / eps_x[(m + 1) % n]
        a[m, (m - 1) % n] += -(1 / h ** 2) / eps_x[m]
    return math.sqrt(float(np.min(np.linalg.eigvalsh(0.5 * (a + a.T)))))


def _order(ns, errs):
    return float(np.polyfit(np.log(np.asarray(ns, float)), np.log(np.asarray(errs)), 1)[0])


TRANSLATIONS = 24
RESOLUTIONS = (8, 16, 32, 64)


def _rms_over_translations(n, eps_hi, eps_lo, rule):
    exact = _exact_omega(KT, eps_hi, eps_lo)
    errs = [abs(_grid_omega(n, k * PERIOD / TRANSLATIONS, eps_hi, eps_lo, rule) - exact) / exact
            for k in range(TRANSLATIONS)]
    return float(np.sqrt(np.mean(np.square(errs))))


def test_the_rule_matches_the_continuum_band_at_second_order():
    """RMS over 24 translations against the exact band, fitted order >= 1.8.

    This is the rule on its own: no FDTD, no geometry kernel, no scene. Both halves are under
    test at once -- ``E_x`` runs along the layers and must be arithmetic, ``E_z`` runs across
    them and must be harmonic -- which is why the polarisation is p and the wave runs along the
    stack. Measured on this machine at eps 12:1, N = 8/16/32/64:
    6.81e-03 / 1.70e-03 / 4.25e-04 / 1.06e-04, fitted order **2.00**.
    """
    errs = [_rms_over_translations(n, 12.0, 1.0, 'kottke') for n in RESOLUTIONS]
    assert _order(RESOLUTIONS, errs) <= -1.8, errs
    assert errs[-1] < 5e-4, errs
    assert all(b < a for a, b in zip(errs, errs[1:])), errs


def test_the_snapped_rule_is_first_order_and_jumps():
    """The control: round the fill to 0/1 and the same harness reads order ~1.

    Without this arm "order 2" is a number with nothing to be better than. Measured:
    3.46e-02 / 1.86e-02 / 9.61e-03 / 4.88e-03, fitted order **0.94**, and the peak-to-peak over
    the 24 translations at N = 32 is 1.68e-02 against the Kottke rule's 8.46e-04 -- twenty
    times, and all of the snapped arm's arriving in one step as the interface crosses a cell
    boundary, which is what a gradient with respect to a position cannot survive.
    """
    errs = [_rms_over_translations(n, 12.0, 1.0, 'snapped') for n in RESOLUTIONS]
    assert -1.3 < _order(RESOLUTIONS, errs) < -0.7, errs
    assert errs[-1] > 10 * _rms_over_translations(RESOLUTIONS[-1], 12.0, 1.0, 'kottke')


def test_an_isotropic_volume_average_loses_the_order():
    """Farjadpour's own control: ``w = 1/3`` everywhere, and the order falls off.

    A scalar volume average is what "subpixel smoothing" means to a reader who has not read
    the paper, and it is exactly the thing the normal's second moment replaces. It is written
    here as an arm rather than as a mutant because the comparison is the argument. Measured:
    2.71e-02 / 1.28e-02 / 6.30e-03 / 3.13e-03, fitted order **1.04**: thirty times the Kottke
    rule's error at the finest resolution, and converging an order slower.
    """
    errs = [_rms_over_translations(n, 12.0, 1.0, 'isotropic') for n in RESOLUTIONS]
    kottke = [_rms_over_translations(n, 12.0, 1.0, 'kottke') for n in RESOLUTIONS]
    assert _order(RESOLUTIONS, errs) > -1.5, errs
    assert errs[-1] > 20 * kottke[-1], (errs, kottke)


# ---------------------------------------------------------------------------------------------
# G4 -- smoothness in position
# ---------------------------------------------------------------------------------------------

def _polarizability(scene):
    """``sum (1 - 1/eps)`` over the whole array: a linear functional of what the kernel uses.

    ``be`` is proportional to ``1/eps``, so this is a sum of the coefficients themselves rather
    than of anything derived, and its continuum limit is proportional to the ball's volume.
    """
    plan, params = _lowered(scene)
    return float(np.sum(1.0 - np.asarray(params['be']) / (plan.dt / plan.dx)))


STEPS = 41


def test_the_coefficient_moves_continuously_as_a_ball_crosses_a_cell():
    """Slide a ball through one cell; the largest single step in a scalar output is tiny.

    The measurement, at R = 0.2 and dx = 0.05 (four cells of radius), over 41 positions
    spanning one cell along x: the smoothed rule's largest single step is **2.46e-4** of the
    mean, while the cell-snapped ``fd.Region`` of the same volume jumps **1.56e-1** -- **634x**,
    and the snapped jump arrives in a single step as a face crosses a cell boundary, which is
    what makes a gradient with respect to a position meaningless under it. The mutant that
    rounds the fill to 0/1 inside ``component_occupancy`` returns the smoothed arm to the
    snapped arm's jump.
    """
    smooth, snapped = [], []
    for k in range(STEPS):
        t = k * DX / (STEPS - 1)
        smooth.append(_polarizability(_grid_scene(
            (fd.Structure(fd.Sphere(radius=0.2, center=(t, 0.0, 0.0)), fd.Material(eps=4.0)),))))
        side = (4 / 3 * math.pi * 0.2 ** 3) ** (1 / 3)
        snapped.append(_polarizability(_grid_scene(
            (fd.Structure(fd.Region(size=(side,) * 3, center=(t, 0.0, 0.0)),
                          fd.Material(eps=4.0)),))))
    step = float(np.max(np.abs(np.diff(smooth)))) / float(np.mean(smooth))
    jump = float(np.max(np.abs(np.diff(snapped)))) / float(np.mean(snapped))
    assert step < 1e-3, step
    assert jump > 100 * step, (step, jump)


@pytest.mark.parametrize('centre, n, worst_without', [
    ((0.013, -0.007, 0.021), N, 3.79e-3),
    ((0.0, 0.0, 0.0), 16, 1.45e-2),
    ((DX / 2, DX / 2, DX / 2), 16, 0.0),
], ids=['off-node', 'node-centred', 'cell-centred'])
def test_the_support_box_holds_the_whole_ball(centre, n, worst_without):
    """The lowered fill sums to ``4 pi R^3/3``, so the support box loses none of the surface.

    ``Sphere.support`` carries one cell of margin because a cell's six component voxels reach
    a whole ``dx`` past its node. The fill is the exact clipped-ball volume, not an
    approximation, so with the margin every placement here reads the truth to ~1e-15.

    **What the margin is worth is placement dependent, and the three rows are the three
    placements that matter.** Without it the three components whose Yee offset along an axis
    is zero cover only ``[k0 - 1/2, k1 - 1/2]`` cells of a box running ``k0 .. k1``, so the
    ball's top surface falls out of their reach as soon as its centre sits more than half a
    cell above a node:

        placement (R = 0.25, dx = 0.05, five cells per radius)   margin dropped
        centred ON A NODE            (n even, so 0 is a node)        1.45e-2
        this file's scene            (worst axis 0.86 of a cell)     3.79e-3
        centred at a CELL CENTRE                                     2.1e-16

    -- so the node-centred ball is the **worst** placement there is and the cell-centred one
    loses nothing at all. The previous writing of this gate and of
    ``Sphere.support`` said the opposite ("a ball centred on a node loses nothing at any
    resolution"), which is what measuring at an odd cell count does: ``Grid.from_domain`` puts
    the low corner at ``centre - n dx / 2``, so the box centre is a node plane only when ``n``
    is **even**, and at an odd ``n`` a ball at the origin is a cell-centred ball. That is why
    the two even-``n`` rows carry their own ``n``. The table over resolution
    (1.45e-2 / 5.74e-3 / 3.69e-3 / 1.45e-3 / 9.30e-4 at 5 / 8 / 10 / 16 / 20 cells per radius,
    node-centred) is in ``src/wavebit/fdtd/_legacy_geometry.py::Sphere.support``.
    """
    radius = 0.25
    ball = fd.Sphere(radius=radius, center=centre)
    scene = _grid_scene((fd.Structure(ball, fd.Material(eps=2.25)),), n=n)
    plan = _quiet(fd.plan, scene)
    occ = occupancy(scene, plan, runtime('numpy'), fields=('E', 'H'))
    truth = 4 / 3 * math.pi * radius ** 3
    for field in ('E', 'H'):
        for c in range(3):
            fill = np.asarray(occ[0][field][c][0])
            volume = float(np.sum(fill)) * plan.dx ** 3
            assert volume == pytest.approx(truth, rel=1e-12), (field, c, volume, truth)

    # the mutant, run here rather than by hand: the support with no margin at all
    box = tuple((min(max(int(math.floor((ball.low[a] - plan.grid.low[a]) / plan.dx)), 0),
                     plan.grid.n[a]),
                 min(max(int(math.ceil((ball.high[a] - plan.grid.low[a]) / plan.dx)), 0),
                     plan.grid.n[a])) for a in range(3))
    lost = max(abs(float(np.sum(np.asarray(
        component_occupancy(ball.body(), plan.grid, box, 'E', c, runtime('numpy'))[0])))
        * plan.dx ** 3 / truth - 1.0) for c in range(3))
    if worst_without == 0.0:
        assert lost < 1e-14, lost
    else:
        assert lost == pytest.approx(worst_without, rel=3e-3), lost


def test_the_normal_moment_sums_to_one_wherever_the_plane_cuts_the_voxel():
    """``sum_c w_c == 1`` wherever the **tangent plane** cuts the voxel -- and what that is not.

    It has to be read at **one** point across the three components; summing ``w_x`` at the
    ``Ex`` points, ``w_y`` at the ``Ey`` points and ``w_z`` at the ``Ez`` points adds three
    different voxels and is not this invariant at all. Measured worst 4.4e-16 over all six
    components' point sets, on 2856 voxels of the ball below.

    **The mask is the kernel's own predicate and it is read from the predicate's own input.**
    ``_cut._mean_of_the_moment`` returns ``w = 0`` exactly where ``area <= _LIVE_AREA dx^2``,
    so this gate takes ``area`` from ``Body.parts`` and asserts that the set it marks is
    *exactly* the set where ``sum_c w_c != 0``. An earlier writing masked on ``sum_c w_c != 0``
    alone, which is the failure condition itself: a mutant that zeroed ``w`` on the voxels it
    found hardest would have shrunk the mask rather than failed the assert.

    **And the invariant is not "wherever the body has any surface in the voxel".** ``fill`` is
    the exact *curved* clip and ``area`` is the *tangent plane's* cross-section at the voxel's
    own point, so where a ball clips a corner whose plane at the centre misses the voxel
    altogether, the fill is fractional and ``w`` is exactly ``(0, 0, 0)``. Measured on this
    ball: **32** of the six components' voxels have ``w = 0`` with a fill further than 1e-9
    from both ends, the worst **2.88e-04** from an end; over ``dx`` 0.2 to 0.0125 at four
    placements each, worst **1.35e-03** off a node and **1.67e-02** for a ball centred on one
    at ``dx = 0.05`` (a voxel whose tangent plane lies on its own face; it halves with ``dx``).
    That is the number ``_LIVE_AREA``'s own docstring used to put at "0 or 1 to eighteen
    digits" -- a plane's argument applied to a curved clip -- and it is corrected there. So the
    **mask is right and the invariant's old wording was wrong**. This gate measures the
    geometric fill distance, not the coefficient error from dropping a normal moment; that
    error also depends on the positive material values and their ratio.
    """
    from wavebit.fdtd._shapes._cut import _LIVE_AREA
    ball = fd.Sphere(radius=0.25, center=(0.013, -0.007, 0.021))
    scene = _grid_scene((fd.Structure(ball, fd.Material(eps=2.25)),))
    plan = _quiet(fd.plan, scene)
    dx, box, body = plan.grid.dx, ball.support(plan.grid), ball.body()
    live, worst_live, dropped, worst_dropped = 0, 0.0, 0, 0.0
    for field in ('E', 'H'):
        for c in range(3):
            points = np.asarray(yee_points(plan.grid, box, field, c))
            fill, area, _ = body.parts(points, dx)
            fill, area = np.asarray(fill), np.asarray(area)
            total = np.sum(np.asarray(body_occupancy(body, points, dx)[1]), axis=-1)
            cut = area > _LIVE_AREA * dx * dx
            assert np.array_equal(cut, total != 0), (field, c)
            live += int(cut.sum())
            worst_live = max(worst_live, float(np.max(np.abs(total[cut] - 1.0))))
            edge = np.minimum(fill[~cut], 1.0 - fill[~cut])
            dropped += int(np.sum(edge > 1e-9))
            worst_dropped = max(worst_dropped, float(np.max(edge)))
    assert live == 2856, live
    assert worst_live < 1e-14, worst_live
    assert dropped == 32, dropped
    assert worst_dropped == pytest.approx(2.8819e-04, rel=1e-3), worst_dropped


def test_a_ball_varies_along_every_axis():
    """``bounded_axes`` is all three, and the coefficient arrays are full on all three.

    A ``Region`` may leave out an axis it spans, because a box clipped to the domain really is
    uniform along it. A ball is not, and an extent-1 axis would broadcast one plane of fills
    over the grid. ``painters`` refuses a smoothed shape whose array is not full, so the mutant
    that copies ``Region``'s rule fails loudly here rather than returning that plane.

    **The second ball is the one that kills the mutant**, and the first on its own does not:
    a small ball's support does not span any axis, so ``Region``'s rule would give it (0, 1, 2)
    too and the mutant would live. A ball wide enough that its support clips to the whole grid
    on every axis is the case where the two rules differ -- and there the answer that matters
    is still all three, because the ball is not uniform along an axis merely because its box
    is. This is the surviving mutant the first draft of this test had.

    It is ``R = 0.55`` and not the box's own half width, because a ball inscribed in the box
    reaches the last half cell of it -- which three of the six components have no point in --
    and ``check_smoothed``'s door 3 refuses that. One cell in from the wall the *geometry*
    clears every face while the one-cell margin on ``Sphere.support`` still clips, so the case
    the two rules differ on is reached by a ball the scene may legally hold.
    """
    ball = fd.Sphere(radius=0.25, center=(0.0, 0.0, 0.0))
    scene = _grid_scene((fd.Structure(ball, fd.Material(eps=2.25)),))
    plan, params = _lowered(scene)
    assert ball.bounded_axes(plan.grid) == (0, 1, 2)
    assert np.asarray(params['be']).shape == (3, N, N, N)
    eps = _eps_of(plan, params)
    for axis in (1, 2, 3):
        assert float(np.std(eps, axis=axis).max()) > 0.1

    wide = fd.Sphere(radius=N * DX / 2 - DX, center=(0.0, 0.0, 0.0))
    scene = _grid_scene((fd.Structure(wide, fd.Material(eps=2.25)),))
    plan, params = _lowered(scene)
    assert wide.support(plan.grid) == ((0, N),) * 3, 'the two rules must differ here'
    assert wide.bounded_axes(plan.grid) == (0, 1, 2)
    assert np.asarray(params['be']).shape == (3, N, N, N)
    eps = _eps_of(plan, params)
    for axis in (1, 2, 3):
        assert float(np.std(eps, axis=axis).max()) > 0.1


# ---------------------------------------------------------------------------------------------
# The archives: a snapped scene never reaches the rule at all
# ---------------------------------------------------------------------------------------------

def test_a_snapped_scene_never_reaches_the_mixing_rule(monkeypatch):
    """``Slab`` and ``Region`` take the stage-1 ``where`` and nothing else, and the skeleton holds.

    The four frozen archives are byte compared by
    ``tests/references/interface/freeze_fd_stage{0,1,2,3}.py`` -- all 1141 arrays (143 + 288 +
    140 + 570) reproduce bit for bit with this commit in place. What this adds is *why*: the
    mixing rule is never called at all, so there is no arithmetic for a rounding to differ in.

    The mutant that gives ``fd.Region`` a body was run against the archives too, and what it did
    there is worth recording. The stage-1 archive, which has no ``Region`` structure, is
    unmoved -- 288 of 288 still bit for bit. The stage-3 archive does not move either: it
    **refuses to freeze**, because its lamellar grating is a ``Region`` 10 units wide on a
    one-cell periodic y axis, and a smoothed shape reaching a periodic face is turned away by
    ``check_smoothed``. That is the door doing its job, and it is also the note C4 needs: a
    smoothed rectangular body cannot spell "runs the whole period" by being clipped the way
    ``fd.Region`` does, so C4 has to land the periodic fold with it.
    """
    calls = []
    real = coef.mix
    monkeypatch.setattr(coef, 'mix', lambda *a, **k: calls.append(1) or real(*a, **k))
    scene = _grid_scene((fd.Structure(fd.Slab(z=(-0.2, 0.0)), fd.Material(eps=2.25)),
                         fd.Structure(fd.Region(size=(0.2, 0.2, 0.2), center=(0.1, 0.0, 0.3)),
                                      fd.Material(eps=4.0, mu=1.5))))
    plan, params = _lowered(scene)
    assert calls == []
    got = sorted(np.unique(np.asarray(params['be'])))
    want = sorted((plan.dt / plan.dx) / e for e in (1.0, 2.25, 4.0))
    assert len(got) == 3 and all(a == b for a, b in zip(got, want)), (got, want)
    assert fd.Slab(z=(-0.2, 0.0)).body() is None
    assert fd.Region(size=(0.2,) * 3).body() is None


def test_a_smoothed_scene_composes_in_painting_order():
    """A later structure mixes into what the earlier ones left, so order still means order.

    The rule's second medium is *what the cell already holds*. Two balls, one inside the other:
    painting them (outer, inner) leaves the inner material in the middle, and painting them
    (inner, outer) leaves the outer -- the same sentence ``Structure`` has carried since stage
    1, now on a fractional edge.

    **How far the sentence carries, and the three tests that draw the line.** A core inside a
    shell is the case where it carries all the way: each ball's surface sits in voxels the
    other fills whole (or not at all), so each mix has its two phases. The same holds for a
    smoothed shape over a cell-snapped one, measured at 2.0e-16 by
    ``test_a_smoothed_shape_over_a_snapped_one_composes_exactly``, and for two smoothed bodies
    a voxel apart, by
    ``test_two_smoothed_bodies_a_voxel_apart_compose_in_either_order``. It does **not** carry
    where two smoothed surfaces are in one voxel, and that is refused rather than approximated
    by ``test_two_smoothed_surfaces_in_one_voxel_are_refused``; what it would have given
    instead is measured in
    ``test_the_size_of_the_composition_error_the_door_refuses``.
    """
    outer = fd.Structure(fd.Sphere(radius=0.25), fd.Material(eps=4.0))
    inner = fd.Structure(fd.Sphere(radius=0.12), fd.Material(eps=9.0))
    _, first = _lowered(_grid_scene((outer, inner)))
    _, second = _lowered(_grid_scene((inner, outer)))
    a, b = np.asarray(first['be']), np.asarray(second['be'])
    mid = (slice(None), N // 2, N // 2, N // 2)
    assert a[mid][0] == pytest.approx(b[mid][0] * 4.0 / 9.0, rel=1e-12)


# ---------------------------------------------------------------------------------------------
# Three backends, and the gradient
# ---------------------------------------------------------------------------------------------

#: The measured cross-backend spread of a smoothed coefficient, worst relative difference
#: from NumPy over ``be``, ``bh`` and ``ae``, against cells per radius::
#:
#:     grid           R/h        jax        torch
#:     24^3, R=0.23    4.6   8.14e-14     4.54e-14
#:     40^3, R=0.3    12.0   1.51e-12     1.27e-12
#:     24^3, R=0.55   11.0   1.29e-12     7.74e-13
#:     48^3, R=0.4    20.0   9.95e-12     8.83e-12
#:
#: It grows with resolution, roughly as ``(R/h)^2``, and the cause is not the mixing rule: the
#: exact clipped-ball volume goes through ``acos``/``atan2``, whose last bits are each
#: library's own, and its round-off accumulates with the grid -- the same growth
#: ``tests/fdtd/test_fd_shapes.py`` already records for the clip's own one-route rows. So a
#: smoothed scene has **no bit-identical cross-backend claim**, and the suite's usual 1e-12
#: (``tests/fdtd/test_fd_stage1_backends.py``) is already reached at 12 cells per radius. The gate
#: below asserts the measured bound at each resolution rather than one number for all of them,
#: because one number would either be slack at 4.6 or a failure at 20.
#: The third row is also the ball whose *support* covers the whole array, so it is the case
#: where ``_paste`` concatenates two empty blocks on every axis. It is ``R = 0.55`` and not
#: the 0.6 this table first carried, because 0.6 puts the ball's surface 0.011 **past** the
#: box on three faces, which door 3 of ``check_smoothed`` now refuses: at 0.55 the geometry
#: clears every face by one cell and the one-cell margin on ``Sphere.support`` still clips, so
#: the empty-block path is reached by a scene that is also legal.
BACKEND_SPREAD = ((0.23, 2e-13), (0.55, 5e-12))


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('radius, bound', BACKEND_SPREAD)
def test_the_three_backends_agree_on_a_smoothed_scene(backend, radius, bound):
    """One body of code: the same ball lowers to the same coefficients on all three.

    They are **not** bit-identical and were not expected to be; see :data:`BACKEND_SPREAD`
    above for the measured table, what causes it and why it grows with resolution. The second
    radius is also the one whose support box covers the whole array, so it is the case where
    ``_paste`` concatenates two empty blocks on every axis -- which is a different code path on
    each backend and reads the same number on all three.
    """
    scene = _grid_scene((fd.Structure(fd.Sphere(radius=radius, center=(0.011, -0.006, 0.017)),
                                      fd.Material(eps=4.0, mu=1.5)),))
    _, base = _lowered(scene, 'numpy')
    _, got = _lowered(scene, backend)
    for key in ('be', 'bh', 'ae'):
        want = np.asarray(base[key])
        have = np.asarray(runtime(backend).numpy(got[key]))
        assert have.shape == want.shape
        assert float(np.max(np.abs(have - want) / np.abs(want))) < bound, key


def _be_sum(radius, rt, plan, bound=0.26):
    """``sum(be[0])`` under a concrete static plan, differentiating numerical lowering only."""
    scene = _grid_scene((fd.Structure(_Ball(radius, center=(0.011, -0.006, 0.017), bound=bound),
                                      fd.Material(eps=4.0)),))
    params = fd.lower(scene, plan, rt)
    return params['be'][0].sum()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_a_traced_radius_reaches_the_coefficient(backend):
    """AD through geometry, occupancy and the mixing rule, against central differences.

    The whole chain: ``radius`` -> ``_shapes.Sphere`` -> the per-component fill and moment at
    each Yee point -> :func:`~wavebit.fdtd._prepare.coefficients.mix` -> ``be``. The declared
    support box is static and the radius is a leaf, which is exactly the shape C6 ships as
    ``fd.Sphere(radius=..., range=...)``; C2's public ``fd.Sphere`` keeps a static radius
    because nothing yet spells the declaration, and this gate is why that is a door and not a
    missing derivative.

    Measured, ``d sum(be[0]) / d radius`` at R = 0.23 and dx = 0.05. ``jax.grad`` reads
    **-1216.6452974131578** and Torch autograd **-1216.6452974131562**: they agree to
    **1.3e-15 relative, which is 7 ulp**, and not to the bit. That distinction is kept
    deliberately -- the same commit writes "the three backends are NOT bit-identical on a
    smoothed scene" for the forward values, and the reason is the same one, the clip's
    ``acos``/``atan2``. The central difference walks towards both as the step falls::

        h = 1e-3    -1216.6247246332      relative to the AD   1.7e-05
        h = 1e-4    -1216.6461109791                           6.7e-07
        h = 1e-5    -1216.6452928113                           3.8e-09
        h = 1e-6    -1216.6452975180                           8.6e-11

    The disagreement falling like ``h^2`` is what says the AD is the right number and the
    difference is carrying its own truncation, which a single step could not say. The gate
    runs at ``h = 1e-6``.

    **The seed has to be float64, and a loose tolerance hides that it is not.** Measured in a
    fresh process with JAX's 32-bit cap unlifted, a bare ``jax.grad(f)(0.23)`` seeds a weakly
    typed **float32** tracer; the whole chain then runs at single precision behind a correctly
    asked-for float64 and the gradient comes back **-1216.6452636719**, agreeing to 2.8e-8 --
    which passes anything written loosely for a finite difference. That is memory
    ``jax-x64-import-order`` happening at the seed rather than at the import, so the cap is
    lifted here, the seed is made explicitly, its dtype is asserted, and the tolerance is three
    orders tighter than a float32 seed can reach.
    """
    pytest.importorskip(backend)
    r0, h = 0.23, 1e-6
    scene = _grid_scene((fd.Structure(_Ball(r0, center=(0.011, -0.006, 0.017), bound=0.26),
                                     fd.Material(eps=4.0)),))
    plan = _quiet(fd.plan, scene)
    fd_slope = float((_be_sum(r0 + h, 'numpy', plan)
                      - _be_sum(r0 - h, 'numpy', plan)) / (2 * h))
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        from wavebit._common.backend import lift_jax_x64
        assert lift_jax_x64(required=True)
        seeded = jax.grad(lambda r: _be_sum(r, 'jax', plan))(jnp.asarray(r0, jnp.float64))
        assert str(seeded.dtype) == 'float64'
        grad = float(seeded)
    else:
        import torch
        r = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
        _be_sum(r, 'torch', plan).backward()
        grad = float(r.grad)
    assert grad == pytest.approx(fd_slope, rel=1e-9), (grad, fd_slope)
    assert abs(grad) > 1.0


# ---------------------------------------------------------------------------------------------
# The doors C2 opens
# ---------------------------------------------------------------------------------------------

def test_a_conductive_or_dispersive_material_beside_a_smoothed_shape_is_refused():
    """``sigma`` and the poles are not smoothed in C2, so nothing may hide behind a fill.

    Four cases, because the wrong number has four routes in: the ball's own conductivity, the
    ball's own poles, the background's (it fills every cell a ball is in), and another
    structure sharing its cells. Without the door the ball's partly filled voxels carry the
    **full** conductivity or polarisation of whatever they overlap, which is a plausible
    number -- ``_Smoothed`` leaves ``sigma`` alone by design, and the door is what makes "left
    alone" the same as "zero".

    A *dispersive* structure overlapping another was refused in stage 1 (``check_overlaps``,
    for a different reason -- the pole term is not routed through the permittivity map), so the
    fourth case here is spelled with a conductor. Since C5 ``check_overlaps`` leaves any pair
    with a smoothed shape to this door, so a dispersive neighbour reaches it too
    (``tests/fdtd/test_fd_c5_texts.py`` pins that mixed pair); the conductor case stays as written.

    **A structure does not share cells with itself**, and the first of the four cases is the
    smoothed shape's *own* material. The message is asserted for that, because it read
    "structures[0] (Sphere(...)) is an fd.Conductor and shares cells with structures[0]
    (Sphere(...)), which is smoothed" -- the first door a user meets, disagreeing with door 4
    one loop below, which had already been given the self case its own clause.
    """
    lossy = fd.Conductor(eps=2.25, sigma=0.2)
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=9.0, gamma=0.1),))
    ball = fd.Sphere(radius=0.2)
    for structures, background, own in (
            ((fd.Structure(ball, lossy),), None, True),
            ((fd.Structure(ball, metal),), None, True),
            ((fd.Structure(ball, fd.Material(eps=2.25)),), lossy, False),
            ((fd.Structure(ball, fd.Material(eps=2.25)),
              fd.Structure(fd.Region(size=(0.5,) * 3), lossy)), None, False)):
        scene = fd.Scene(
            domain=fd.Domain(size=(N * DX,) * 3),
            boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
            resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
            background=background, structures=structures)
        with pytest.raises(ValueError, match='smoothed') as e:
            _quiet(fd.plan, scene)
        message = str(e.value)
        if own:
            assert 'is smoothed and its own material is an fd.' in message, message
            assert 'shares cells with' not in message, message
        else:
            assert 'shares cells with structures[0]' in message, message
    # the same scene with the smoothed shape replaced by a snapped one is untouched
    ok = _grid_scene((fd.Structure(fd.Region(size=(0.4,) * 3), lossy),))
    _quiet(fd.plan, ok)


@pytest.mark.parametrize('face', [fd.Periodic(), fd.Bloch(k=0.7)],
                         ids=['Periodic', 'Bloch'])
def test_a_smoothed_shape_at_a_periodic_face_is_refused(face):
    """C2 does not sum the image ring, so a ball may not reach a Periodic or Bloch face.

    Held open, a ``Sphere(radius=0.2)`` centred on the ``x-`` face of a 1.2-wide periodic cell
    lowers **59.3%** of its volume, against 1.0000 for the same ball at the centre of the same
    cell, and every invariant the lowering checks is still satisfied -- ``sum_c w_c`` reads 1
    to 4.4e-16 on every voxel that has any surface in it. The refusal is drawn on
    the support box, which carries a cell of margin, so it also turns away a ball that merely
    comes within a cell of the face; the message says how far to move it.

    **Both faces are run, because the door names both and only one was ever asserted.** The
    door reads ``isinstance(face, (Periodic, Bloch))``; with ``fd.Bloch`` never put in front
    of a smoothed shape anywhere in the suite, cutting that tuple down to ``(Periodic,)``
    left every test green while a ball on a Bloch face lowered with 40.7% of its volume
    missing. A Bloch face is the *same* geometry question -- the image ring is what makes the
    body whole and the phase is only what each image is multiplied by -- so there is nothing
    to measure separately here, only a scene to assert.
    """
    edge = fd.Scene(
        domain=fd.Domain(size=(N * DX,) * 3),
        boundaries=fd.Boundaries(x=face, y=fd.Periodic(), z=fd.PEC()),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
        structures=(fd.Structure(fd.Sphere(radius=0.2, center=(-0.6, 0.0, 0.0)),
                                 fd.Material(eps=2.25)),))
    with pytest.raises(ValueError, match='periodic face') as e:
        _quiet(fd.plan, edge)
    assert f'reaches the {type(face).__name__} x face' in str(e.value)
    # well inside the same cell it is fine
    inside = fd.Scene(
        domain=fd.Domain(size=(N * DX,) * 3),
        boundaries=fd.Boundaries(x=face, y=fd.Periodic(), z=fd.PEC()),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
        structures=(fd.Structure(fd.Sphere(radius=0.2), fd.Material(eps=2.25)),))
    _quiet(fd.plan, inside)


def test_a_ball_outside_the_box_is_refused_rather_than_trimmed():
    """A smoothed shape is not clipped the way a cell-snapped ``Region`` is, and says so."""
    scene = _grid_scene((fd.Structure(fd.Sphere(radius=0.1, center=(4.0, 0.0, 0.0)),
                                      fd.Material(eps=2.25)),))
    with pytest.raises(ValueError, match='lies outside the box'):
        _quiet(fd.plan, scene)


def test_a_ball_that_ends_inside_an_absorbing_layer_is_refused():
    """``uniform_along`` is ``False`` wherever a ball reaches, which is what the layer asks.

    A bounding-box test would wave through a ball whose *box* spans the layer, and that case is
    measured at 1.9e-1 reflection against 6.0e-4 -- 317x (``tests/fdtd/test_fd_absorber.py``).

    The ball sits **tangent** to the z- wall rather than across it, so that this door and not
    ``check_smoothed``'s door 3 (a shape the grid cannot carry whole at a face) is the one under
    test: the two are different refusals and a scene that trips both says nothing about either.
    """
    scene = fd.Scene(
        domain=fd.Domain(size=(N * DX,) * 3),
        boundaries=fd.Boundaries.open(fd.PML(cells=4, wavelength=1.0, allow_thin=True)),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
        structures=(fd.Structure(fd.Sphere(radius=0.2, center=(0.0, 0.0, -0.4)),
                                 fd.Material(eps=2.25)),))
    with pytest.raises(ValueError, match='ends inside'):
        _quiet(fd.plan, scene)


def test_a_smoothed_shape_refuses_the_question_a_snapped_one_answers():
    """``filled_cells`` and ``body`` are the same question asked of the two discretisations."""
    grid = fd.plan(_grid_scene(())).grid
    with pytest.raises(NotImplementedError, match='fractionally'):
        fd.Sphere(radius=0.2).filled_cells(grid)
    assert fd.Sphere(radius=0.2).body() is not None
    assert fd.Slab(z=(-0.1, 0.1)).filled_cells(grid) == fd.Slab(z=(-0.1, 0.1)).support(grid)


def test_a_sphere_refuses_a_traced_or_impossible_radius():
    """The C2 door: a radius is a static, positive length, and ``range=`` is named where it is
    not.

    A length is a size here (memory ``a-length-is-a-size``): a negative radius would subtract
    the shape rather than draw it.
    """
    with pytest.raises(ValueError, match='positive length'):
        fd.Sphere(radius=-0.1)
    with pytest.raises(ValueError, match='positive length'):
        fd.Sphere(radius=0.0)
    with pytest.raises(ValueError, match='triple'):
        fd.Sphere(radius=0.1, center=(0.0, 0.0))
    torch = pytest.importorskip('torch')
    r = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='static'):
        fd.Sphere(radius=r)


# ---------------------------------------------------------------------------------------------
# What two shapes may be to each other: the composition doors
# ---------------------------------------------------------------------------------------------

def _hand_mix(held, value, fill, w):
    """The rule, written here rather than imported: the oracle may not be the thing judged."""
    harmonic = fill / value + (1.0 - fill) / held
    arithmetic = fill * value + (1.0 - fill) * held
    return 1.0 / (w * harmonic + (1.0 - w) / arithmetic)


def test_a_smoothed_shape_clear_of_a_snapped_face_composes_exactly():
    """A ball over a cell-snapped substrate, a voxel from its face: two phases, and exact.

    Where no voxel the ball's surface cuts is also cut by the slab's face, every such voxel
    holds exactly two media -- the ball's and whatever the slab's painter left there, which is
    all slab or all not -- and the mixing rule has what it needs. The oracle is written out
    here from the snapped mask and the kernel's ``(fill, w)`` pair -- ``coefficients`` is not
    consulted -- and the lowered ``eps`` matches it to round-off on all three components over
    the whole array, for a ball wholly inside the slab and for one wholly above it.

    In the other order the snapped shape is painted last and **replaces** the cells it fills:
    every voxel of the slab reads exactly 4.0 (bit for bit, because ``_Snapped`` is a
    ``where``) and nothing above it moves by a bit from the ball-alone scene. Both halves of "a
    later one overwrites an earlier one" hold across the seam between the two discretisations.

    Until C4 this test put the ball's centre ON the slab's face and called the result exact,
    against an oracle that was the snapped mask and not the geometry: the three components
    whose Yee point lies on the face have a voxel half inside the slab that the mask calls
    outside. That scene is refused now, and
    :func:`test_a_smoothed_surface_in_a_voxel_a_snapped_face_cuts_is_refused` measures why.

    The first half reads the mix, ``coefficients.build``'s ``be``: the star lift acts on it
    afterwards (``stars.lift``), a different rule, tested in ``tests/fdtd/test_fd_star_lift.py`` (the
    first, eight-block form of the lift moved this ball's ``be`` by up to 1.6%). The second half
    reads ``fd.lower``, lift and all.
    """
    lo = -N * DX / 2
    top = lo + 12 * DX              # exactly on a cell boundary: the snap moves nothing
    slab = fd.Slab(z=(lo, top))
    for centre_z in (top - 0.2 - 2 * DX, top + 0.2 + 2 * DX):
        ball = fd.Sphere(radius=0.2, center=(0.0, 0.0, centre_z))
        scene = _grid_scene((fd.Structure(slab, 4.0), fd.Structure(ball, 12.0)))
        plan = _quiet(fd.plan, scene)
        eps = _eps_of(plan, coef.build(scene, plan, runtime('numpy')))
        grid, box = plan.grid, ball.support(plan.grid)
        (i0, i1), (j0, j1), (k0, k1) = box
        truth = np.empty_like(eps)
        for c in range(3):
            held = np.ones((plan.nz, plan.ny, plan.nx))
            held[:12] = 4.0
            fill, w = body_occupancy(ball.body(), yee_points(grid, box, 'E', c), grid.dx)
            out = held.copy()
            out[k0:k1, j0:j1, i0:i1] = _hand_mix(held[k0:k1, j0:j1, i0:i1], 12.0, fill,
                                                 w[..., c])
            truth[c] = out
        worst = float(np.max(np.abs(eps - truth) / np.abs(truth)))
        assert worst < 1e-15, (centre_z, worst)

    ball = fd.Sphere(radius=0.2, center=(0.0, 0.0, top + 0.2 + 2 * DX))
    plan, _ = _lowered(_grid_scene((fd.Structure(slab, 4.0), fd.Structure(ball, 12.0))))
    _, reversed_ = _lowered(_grid_scene((fd.Structure(ball, 12.0), fd.Structure(slab, 4.0))))
    back = _eps_of(plan, reversed_)
    assert np.all(back[:, :12] == 4.0), 'the snapped painter replaces, it does not mix'
    _, alone = _lowered(_grid_scene((fd.Structure(ball, 12.0),)))
    assert np.array_equal(back[:, 12:], _eps_of(plan, alone)[:, 12:])


def test_a_smoothed_surface_in_a_voxel_a_snapped_face_cuts_is_refused(monkeypatch):
    """Door 6: a body's surface and a snapped shape's face in one voxel is three media.

    A snapped shape owns whole cells, so the three components whose Yee point sits on one of
    its faces have a voxel half on each side, painted all one side or the other; a smoothed
    body cutting that voxel mixes into what the painter left, and one of the voxel's two
    other media is lost. Held open, the door's own measurement: a pillar ``fd.Box`` of
    ``eps`` 12 flush on an ``fd.Slab`` substrate of ``eps`` 2.25 reads tangential ``eps``
    **6.5** at the joint node where the voxel holds 7.125 -- written out here from the
    geometry, half substrate and half pillar -- and nothing else in the lowering notices.

    Refused: the ball centred on the substrate's face (the scene the previous writing of the
    composition test called exact), the pillar flush, 0.3 of a cell above, and half a cell
    sunk into the substrate (its rim, where its sides meet the face, still holds three media).
    Not refused: the pillar half a cell above the face, whose surface cuts no voxel the face
    cuts -- the voxel between them is the slab's own documented half-cell rule, which the
    body does not touch -- the pillar a voxel clear, and a ball inside a full-width slab near
    the domain's ``x-`` wall, which is the slab reaching the wall and no face at all.

    Mutants, each alone, measured 2026-09-26 over this file and the four beside it that plan
    smoothed scenes (141 fast cases): the door never refusing, the domain's walls read as
    faces, and every straddling voxel near the surface refused whether the body cuts it or
    not -- each fails this case alone.
    """
    from wavebit.fdtd._prepare import checks
    lo = -N * DX / 2
    substrate = fd.Structure(fd.Slab(z=(lo, 0.0)), 2.25)

    def pillar(base):
        return fd.Structure(fd.Box(size=(0.3, 0.3, 0.3), center=(0.0, 0.0, base + 0.15)), 12.0)
    ball = fd.Structure(fd.Sphere(radius=0.2, center=(0.0, 0.0, 0.0)), 12.0)
    for refused in (ball, pillar(0.0), pillar(0.3 * DX), pillar(-0.5 * DX)):
        with pytest.raises(ValueError, match='cell-snapped structures\\[0\\]') as e:
            _quiet(fd.plan, _grid_scene((substrate, refused)))
        assert 'three media' in str(e.value) and '6.5 where the voxel holds 7.125' in str(e.value)
    for accepted in (pillar(0.5 * DX), pillar(1.2 * DX)):
        _quiet(fd.plan, _grid_scene((substrate, accepted)))
    near_wall = fd.Structure(fd.Sphere(radius=0.2, center=(lo + 0.2 + 0.4 * DX, 0.0, 0.0)), 12.0)
    _quiet(fd.plan, _grid_scene((fd.Structure(fd.Slab(z=(lo, 0.4)), 2.25), near_wall)))

    monkeypatch.setattr(checks, '_snapped_joint', lambda *a, **k: None)
    plan, params = _lowered(_grid_scene((substrate, pillar(0.0))))
    eps = _eps_of(plan, params)
    joint = N // 2                                  # the node at z = 0, x = y = 0
    assert eps[0, joint, joint, joint] == pytest.approx(6.5, rel=1e-12)
    assert 0.5 * 2.25 + 0.5 * 12.0 == 7.125         # the voxel, half substrate, half pillar
    assert eps[0, joint - 1, joint, joint] == pytest.approx(2.25, rel=1e-12)
    assert eps[0, joint + 1, joint, joint] == pytest.approx(12.0, rel=1e-12)


def test_two_smoothed_surfaces_in_one_voxel_are_refused():
    """Painting order does not compose there, and the refusal names what does.

    The mixing rule takes **two** phases and **one** surface. A second smoothed painter reads
    the first mixture as a medium and puts a second interface across it, so the background
    survives with weight ``(1-f1)(1-f2)`` in a voxel that holds none of it. The size of that
    is measured in ``test_the_size_of_the_composition_error_the_door_refuses`` below, with
    this door held open; here the door is shut and what is under test is that it fires on
    both spellings of the case and names the spellings that work.
    """
    ball = fd.Sphere(radius=0.2, center=(0.013, -0.007, 0.021))
    glass = fd.Material(eps=12.0)
    twice = _grid_scene((fd.Structure(ball, glass), fd.Structure(ball, glass)))
    with pytest.raises(ValueError, match='surfaces pass through') as e:
        _quiet(fd.plan, twice)
    message = str(e.value)
    assert 'fd.Union' in message and 'fd.Intersection' in message and 'fd.Difference' in message
    assert 'not on bounding boxes' in message, 'a dimer with a gap must not read this as a ban'

    lo = -N * DX / 2
    pair = _grid_scene((fd.Structure(_HalfSpace(lo + 12.5 * DX), 12.0),
                        fd.Structure(_HalfSpace(lo + 12.5000001 * DX), 12.0)))
    with pytest.raises(ValueError, match='surfaces pass through'):
        _quiet(fd.plan, pair)

    # two balls that touch, and two that overlap: both are still refused. C10's clip does not
    # answer two curved surfaces in one voxel -- its registered dimer gate failed
    # (docs/validation/fdtd/fd-polytope-clip.md) -- and says so.
    for gap in (0.0, -0.1):
        touching = _grid_scene((
            fd.Structure(fd.Sphere(radius=0.15, center=(-0.15 - gap / 2, 0.0, 0.0)), 4.0),
            fd.Structure(fd.Sphere(radius=0.15, center=(0.15 + gap / 2, 0.0, 0.0)), 4.0)))
        with pytest.raises(ValueError, match='surfaces pass through.*two curved surfaces'):
            _quiet(fd.plan, touching)


def test_the_size_of_the_composition_error_the_door_refuses(monkeypatch):
    """Hold door 5 open and read what the two half spaces give: 73% against a closed form.

    Two half spaces of the same material with faces at 12.5 and 12.5000001 cells **are** the
    one half space at 12.5000001 cells, which the single-painter lowering answers exactly (it
    is gated against a hand closed form by the co-location tests above). So the one-shape
    scene is the oracle and no tolerance is involved in the comparison -- the two numbers are
    two lowerings of the same geometry, one of which composes two painters::

        tops (cells)          got        truth      relative
        12.5 / 12.5000001     3.20000    1.84615    0.733
        12.45 / 12.55         3.22364    2.01681    0.598
        12.2 / 12.8           4.34783    3.75000    0.159
        12.1 / 13.9          12.00000   12.00000    5.2e-13

    The last row is the control: a whole cell apart, no voxel holds both surfaces and the two
    lowerings agree to round-off, so what the first three rows measure is "two smoothed
    surfaces in one voxel" and not "two structures".

    Through the public door the same ``fd.Sphere`` declared **twice** instead of once moves
    ``eps`` by up to 1.22 relative (8.144451 against 3.666166), on 648 voxels by more than 10%
    and 796 by more than 1%, with nothing in the scene saying so.
    """
    from wavebit.fdtd._prepare import checks
    monkeypatch.setattr(checks, '_shared_surface_voxels', lambda *a, **k: (0, None))
    # This measures the historical DIAGONAL composition defect with its door
    # held open. The coupled operator now has an independent positivity door;
    # leave the constitutive terms out of this deliberately invalid comparator.
    from wavebit.fdtd._prepare import constitutive
    monkeypatch.setattr(constitutive, 'terms', lambda *a: ())
    lo = -N * DX / 2
    rows = []
    for t1, t2 in ((12.5, 12.5000001), (12.45, 12.55), (12.2, 12.8), (12.1, 13.9)):
        two = _grid_scene((fd.Structure(_HalfSpace(lo + t1 * DX), 12.0),
                           fd.Structure(_HalfSpace(lo + t2 * DX), 12.0)))
        one = _grid_scene((fd.Structure(_HalfSpace(lo + max(t1, t2) * DX), 12.0),))
        got, truth = _eps_of(*_lowered(two)), _eps_of(*_lowered(one))
        rows.append(float(np.max(np.abs(got - truth) / np.abs(truth))))
    assert rows[0] == pytest.approx(0.733, abs=5e-3), rows
    assert rows[1] == pytest.approx(0.598, abs=5e-3), rows
    assert rows[2] == pytest.approx(0.159, abs=5e-3), rows
    assert rows[3] < 1e-12, rows

    ball = fd.Sphere(radius=0.2, center=(0.013, -0.007, 0.021))
    glass = fd.Material(eps=12.0)
    twice = _eps_of(*_lowered(_grid_scene((fd.Structure(ball, glass),) * 2)))
    once = _eps_of(*_lowered(_grid_scene((fd.Structure(ball, glass),))))
    moved = np.abs(twice - once) / np.abs(once)
    assert float(moved.max()) == pytest.approx(1.22, abs=0.01), float(moved.max())
    assert int((moved > 0.10).sum()) == 648, int((moved > 0.10).sum())
    assert int((moved > 0.01).sum()) == 796, int((moved > 0.01).sum())


def test_two_smoothed_bodies_a_voxel_apart_compose_in_either_order():
    """A dimer with a gap is accepted, and the door is drawn on voxels and not on boxes.

    The two balls sit on a 45 degree diagonal, so their axis-aligned bounding boxes **overlap**
    while their surfaces are 1.9 cells apart. The design names this as the failure of a
    bounding-box test by name: two equal spheres on that diagonal have overlapping boxes out
    to a centre distance of ``2 sqrt(2) R``, so a box test would falsely refuse a gold dimer
    at any gap below 0.414 diameters -- the canonical plasmonic scene.

    That they are accepted is half the claim; the other half is that they really do compose,
    and the evidence is that **painting order does not matter**: with no voxel shared, the two
    orders agree to 1e-14 (they are not bit-identical, because a voxel only one ball touches
    still goes through the other's ``f = 0`` double reciprocal). On the refused scene the same
    comparison is meaningless, which is the point of refusing it.

    **The second pair is the one that kills the two-stage mutant.** The door marks a voxel
    with the two signed distances first and only pays for the exact fill where both are
    marked, and the first pair never reaches the second stage at all: at 1.9 cells apart *no*
    voxel is even marked. Two balls a **whole cell** apart along x are marked in 36 voxels and
    share none of them, so the fill test is the only thing standing between that scene and a
    refusal -- and the mutant that keeps the distance mark and drops the fill test survived
    this gate's first draft with the diagonal pair alone (memory
    ``mutate-the-code-to-test-the-tests``).
    """
    a = fd.Structure(fd.Sphere(radius=0.2, center=(-0.175, -0.175, 0.0)), fd.Material(eps=4.0))
    b = fd.Structure(fd.Sphere(radius=0.2, center=(0.175, 0.175, 0.0)), fd.Material(eps=9.0))
    grid = _quiet(fd.plan, _grid_scene(())).grid
    boxa, boxb = a.shape.support(grid), b.shape.support(grid)
    assert all(u[0] < v[1] and v[0] < u[1] for u, v in zip(boxa, boxb)), 'the boxes must overlap'
    gap = math.dist(a.shape.center, b.shape.center) - 2 * 0.2
    assert 1.5 * DX < gap < 2.5 * DX, gap

    forward = _eps_of(*_lowered(_grid_scene((a, b))))
    backward = _eps_of(*_lowered(_grid_scene((b, a))))
    worst = float(np.max(np.abs(forward - backward) / np.abs(backward)))
    assert worst < 1e-14, worst
    assert float(forward.max()) > 8.9, 'both balls are really in the scene'

    close = (fd.Structure(fd.Sphere(radius=0.2, center=(-0.225, 0.0, 0.0)), fd.Material(eps=4.0)),
             fd.Structure(fd.Sphere(radius=0.2, center=(0.225, 0.0, 0.0)), fd.Material(eps=9.0)))
    assert math.isclose(0.45 - 0.4, DX, rel_tol=1e-9), 'exactly one cell of gap'
    first = _eps_of(*_lowered(_grid_scene(close)))
    second = _eps_of(*_lowered(_grid_scene(close[::-1])))
    assert float(np.max(np.abs(first - second) / np.abs(second))) < 1e-14


def test_a_voxel_one_body_fills_whole_is_not_a_shared_surface():
    """Door 5 reads its fills against the clip's own round-off, and the ends are never exact.

    Two ``_HalfSpace``es of the **same** material, faces at 13.0 and 12.5 cells above the low
    wall of this file's 24^3 box. Nothing about that is illegal: 13.0 lands exactly on a cell
    boundary, so in every voxel one of the two bodies is whole or absent and the composition is
    the two-painter case that composes exactly. It **is** the deeper half space alone, which is
    this file's own composition oracle, and the two painting orders agree to 3.8e-14 while the
    two-painter lowering agrees with the one-painter lowering of the same geometry to 9.3e-14.

    Written ``0 < f < 1``, on the argument that "the two ends carry meaning", the door refused
    it on **740** voxels -- because the exact clip returns neither end exactly. At ``dx = 0.05``
    a voxel the half space fills whole reads 0.99999999999991 (8.8e-14 short), so ``f < 1`` is
    true for it and every one of the 740 was a voxel one body fills whole or misses; not one of
    them had ``min(f, 1-f) > 1e-9`` for both bodies. The refusal even printed "they fill 1.0000
    and 0.5000" as it turned the scene away, because its "worst" pick is an ``argmax`` over the
    same quantity.

    The tolerance is :func:`~wavebit.fdtd._prepare.checks._clip_tolerance`, what door 3 already
    reads at the other end of the same clip, and it is measured where it is defined
    (worst 4.0e-13 at the whole end, 1.2e-13 at the missed end, over both bodies C2 can reach
    at five resolutions and four placements each). **The mutant is the code as it stood**:
    comparing against 0 and 1 themselves, which refuses this scene.

    It does not slacken the door. The scenes door 5 is right about keep every genuinely cut
    voxel -- 1728 of the gate's own 12.5/12.5000001 scene and 1815 of the sphere-twice scene --
    and lose only the round-off ones (736 and 68), which were inflating the counts the refusals
    printed.
    """
    lo = -N * DX / 2
    legal = (fd.Structure(_HalfSpace(lo + 13.0 * DX), 12.0),
             fd.Structure(_HalfSpace(lo + 12.5 * DX), 12.0))
    try:
        forward = _eps_of(*_lowered(_grid_scene(legal)))
        backward = _eps_of(*_lowered(_grid_scene(legal[::-1])))
        one = _eps_of(*_lowered(_grid_scene((legal[0],))))
    except ValueError as error:
        raise AssertionError('known-valid whole-voxel composition was refused') from error
    assert float(np.max(np.abs(forward - backward) / np.abs(backward))) < 1e-12
    assert float(np.max(np.abs(forward - one) / np.abs(one))) < 1e-12

    # the mutant: no tolerance at either end
    from wavebit.fdtd._prepare import checks
    grid = _quiet(fd.plan, _grid_scene(())).grid
    bodies = [s.shape.body() for s in legal]
    boxes = [s.shape.support(grid) for s in legal]
    shared = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(*boxes))
    assert checks._shared_surface_voxels(bodies[0], bodies[1], grid, shared)[0] == 0
    saved = checks._CLIP_ROUNDOFF, checks._CLIP_ULPS
    checks._CLIP_ROUNDOFF, checks._CLIP_ULPS = 0.0, 0.0
    try:
        count, worst = checks._shared_surface_voxels(bodies[0], bodies[1], grid, shared)
    finally:
        checks._CLIP_ROUNDOFF, checks._CLIP_ULPS = saved
    assert count == 740, count
    assert worst[0] < 1e-12, worst          # every one of them is round-off and nothing else

    # and the door still refuses what it is right about, on the voxels that are really cut
    ball = fd.Sphere(radius=0.2, center=(0.013, -0.007, 0.021))
    twice = _grid_scene((fd.Structure(ball, 12.0), fd.Structure(ball, 12.0)))
    # C11 adds the E common nodes and H common centres: measured 2422
    # shared voxels, versus the historical 1815 on the six Yee components.
    with pytest.raises(ValueError, match='surfaces pass through the same voxel: 2422'):
        _quiet(fd.plan, twice)


def test_the_clip_tolerance_follows_the_grid_and_not_a_number():
    """A fill's round-off is ``(|x|/dx) eps``, so the tolerance that reads it cannot be a number.

    ``_CLIP_ROUNDOFF`` was measured at the grids this file runs -- ``|x|/dx`` about 24 -- and
    its own first sentence says the error grows with that ratio, which is a statement the
    constant itself could not keep. It does grow, cleanly and over eleven decades. This gate
    re-measures the law by moving the **whole scene**, domain and ball together, away from the
    coordinate origin and reading the voxels the ball fills *whole*::

        |x|/dx     |f - 1| where the ball fills whole    multiple of (|x|/dx) eps
        1.2e+01                6.7e-16                            0.25
        2.0e+03                6.8e-13                            1.53
        2.0e+05                4.4e-11                            0.98
        2.0e+06                7.0e-10                            1.57
        2.0e+07                2.8e-09                            0.63
        2.0e+08                4.5e-08                            1.01
        2.0e+09                7.2e-07                            1.61

    The law is a staircase: the error steps with the binade of ``|x|``, so within a step the
    multiple swings -- 0.62 to 2.31 over forty ratios from 4.0e5 to 8.0e6, all six
    components -- and the factor of 2 asserted below holds at these seven ratios, not between
    them. The absolute 1e-9 is first crossed at ``|x|/dx`` of about **5.2e6**, where ``|x|``
    passes ``2^18`` and the error steps from 7.0e-10 to 2.8e-9; from there a door written
    against it alone reads a voxel the body fills whole as *fractional* -- which is
    precisely the 740-voxel false refusal of
    ``test_a_voxel_one_body_fills_whole_is_not_a_shared_surface``, brought back by the
    coordinates rather than by the comparison.

    **The end the body MISSES does not drift at all**: exactly 0.0 at every one of those
    ratios. That is the whole reason door 3, which reads only that end, was never in danger
    and door 5, which reads both, was. The gate asserts both halves, so a mutant that makes
    the tolerance relative at the wrong end is as dead as one that leaves it absolute.

    ``_CLIP_ULPS = 0`` is the mutant -- the constant as it stood -- and it fails the one
    assertion that matters: the tolerance must stay above the measured round-off at every
    grid, because falling below it is a refusal of a legal scene and not a lost digit.
    """
    from wavebit.fdtd._prepare import checks
    ulp = float(np.finfo(np.float64).eps)
    rows = []
    for centre in (0.0, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8):
        scene = fd.Scene(domain=fd.Domain(size=(N * DX,) * 3, center=(centre,) * 3),
                         boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                         resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4))
        grid = _quiet(fd.plan, scene).grid
        ball = fd.Sphere(radius=0.2, center=(centre,) * 3)
        body = ball.body()
        points = np.asarray(yee_points(grid, ball.support(grid), 'E', 0))
        fill = np.asarray(body_occupancy(body, points, grid.dx)[0])
        d = np.asarray(body.surface(points)[0])
        whole, missed = d <= -math.sqrt(3) / 2 * grid.dx, d >= math.sqrt(3) / 2 * grid.dx
        assert whole.sum() > 100 and missed.sum() > 100
        at_one = float(np.max(np.abs(fill[whole] - 1.0)))
        at_zero = float(np.max(np.abs(fill[missed])))
        ratio = max(max(abs(grid.low[a]), abs(grid.low[a] + N * DX)) for a in range(3)) / DX
        rows.append((ratio, at_one, at_zero, checks._clip_tolerance(grid)))

    for ratio, at_one, at_zero, tol in rows:
        assert at_zero == 0.0, (ratio, at_zero)         # the missed end never drifts
        assert at_one <= 2.0 * ratio * ulp, (ratio, at_one)   # ... and the whole end obeys the law
        assert tol > at_one, (ratio, tol, at_one)       # what the doors read is above both
        assert tol < 1e-2, (ratio, tol)                 # and far below a fraction of a voxel
    # the law is a law and not a plateau: three orders of |x|/dx buy three orders of error
    assert rows[-1][1] / rows[2][1] == pytest.approx(rows[-1][0] / rows[2][0], rel=0.7)
    # the floor rules at the grids this suite runs, and the relative arm takes over beyond
    assert rows[0][3] == checks._CLIP_ROUNDOFF and rows[1][3] == checks._CLIP_ROUNDOFF
    assert rows[-1][3] == pytest.approx(checks._CLIP_ULPS * rows[-1][0] * ulp, rel=1e-9)
    # the mutant: the constant alone, which falls under the measured round-off from 5.2e6 on
    saved = checks._CLIP_ULPS
    checks._CLIP_ULPS = 0.0
    try:
        under = [(r[0], r[1]) for r in rows if checks._CLIP_ROUNDOFF <= r[1]]
    finally:
        checks._CLIP_ULPS = saved
    assert [round(r[0], -6) for r in under] == [2e7, 2e8, 2e9], under


@pytest.mark.parametrize('case', ('below', 'at', 'above', 'nonfinite', 'outside', 'shared'))
def test_the_clip_tolerance_refuses_when_endpoint_neighborhoods_overlap(case, monkeypatch):
    """A range failure must not turn either geometric refusal into an empty predicate.

    The floor cases isolate the exact endpoint-neighbourhood boundary without relying on a
    rounded coordinate product. The two live doors then use real sphere fills and a binary
    grid translated by 2**38: nodes are still separated by 512 coordinate ulps, but the
    unbounded tolerance exceeds one and would hide both an outside fill and shared surfaces.
    """
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare import checks

    grid = Grid(n=(24, 24, 24), dx=1.0 / 32, low=(0.0, 0.0, 0.0))
    if case in ('below', 'at', 'above'):
        floor = {'below': 0.25, 'at': 0.5, 'above': 0.75}[case]
        monkeypatch.setattr(checks, '_CLIP_ROUNDOFF', floor)
        if case == 'below':
            tol = checks._clip_tolerance(grid)
            assert tol == floor and tol < 0.5 < 1.0 - tol
        else:
            assert not (floor < 0.5 < 1.0 - floor)
            with pytest.raises(ValueError, match='ClipPrecisionRefused'):
                checks._clip_tolerance(grid)
        return
    if case == 'nonfinite':
        grid = Grid(n=(1, 1, 1), dx=1e-308, low=(1e308, 0.0, 0.0))
        with pytest.raises(ValueError, match='ClipPrecisionRefused'):
            checks._clip_tolerance(grid)
        return

    offset = 2.0 ** 38
    moved = Grid(n=grid.n, dx=grid.dx, low=(offset, 0.0, 0.0))
    assert grid.dx / math.ulp(offset) == 512
    cx = 0.75 if case == 'outside' else 0.375
    shape = fd.Sphere(radius=0.125, center=(cx, 0.375, 0.375))
    shifted = fd.Sphere(radius=0.125, center=(offset + cx, 0.375, 0.375))
    if case == 'outside':
        control = checks._reaches_outside(shape, shape.body(), grid, shape.support(grid))
        assert control[:3] == (checks._LEAVES, 0, '+') and control[3] > 0
        with pytest.raises(ValueError, match='ClipPrecisionRefused'):
            checks._reaches_outside(shifted, shifted.body(), moved, shifted.support(moved))
    else:
        control = checks._shared_surface_voxels(
            shape.body(), shape.body(), grid, shape.support(grid), include_common=True)
        assert control[0] > 0 and control[1][0] > 0
        with pytest.raises(ValueError, match='ClipPrecisionRefused'):
            checks._shared_surface_voxels(
                shifted.body(), shifted.body(), moved, shifted.support(moved),
                include_common=True)


def test_the_shared_surface_mark_reaches_a_voxel_diagonal_and_not_an_edge():
    """``_VOXEL_REACH`` is half a voxel **diagonal**, and a dimer at 0.95 cells proves it.

    Door 5's cheap first stage marks a voxel when a body's signed distance is within
    ``_VOXEL_REACH * dx`` of that voxel's own point, and the whole soundness argument of that
    stage is one sentence: a surface *inside* a voxel is within half the voxel's **diagonal**
    of the voxel's centre, so no voxel whose fill will turn out to be fractional can be missed.
    Half an **edge** is not that bound -- it misses a surface that cuts a corner -- and the
    suite had no scene that could tell the two apart, so ``sqrt(3)/2 -> 1/2`` was a mutant that
    lived.

    Two ``fd.Sphere(radius=0.2)`` whose surfaces are **0.95 cells** apart are the scene that
    tells them apart: they share **8** voxels under the diagonal and **0** under half an edge,
    so the mutant accepts a scene with eight genuinely doubly-cut voxels in it. The two reaches
    disagree everywhere the door works, not only here -- the same ball twice 1815 against 1219,
    two touching balls 48 against 32, two overlapping balls 220 against 96, and the dimer 16
    against 4 at a gap of 0.9 cells -- but this is the placement where the disagreement crosses
    zero and turns a refusal into an acceptance.

    The other direction is already gated: at a gap of one whole cell the shipped reach counts
    none and the scene is accepted
    (``test_two_smoothed_bodies_a_voxel_apart_compose_in_either_order``).
    """
    from wavebit.fdtd._prepare import checks
    assert checks._VOXEL_REACH == pytest.approx(math.sqrt(3.0) / 2.0)
    half = (0.4 + 0.95 * DX) / 2
    pair = (fd.Structure(fd.Sphere(radius=0.2, center=(-half, 0.0, 0.0)), 4.0),
            fd.Structure(fd.Sphere(radius=0.2, center=(half, 0.0, 0.0)), 9.0))
    grid = _quiet(fd.plan, _grid_scene(())).grid
    boxes = [s.shape.support(grid) for s in pair]
    shared = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(*boxes))
    bodies = [s.shape.body() for s in pair]

    def count(reach):
        saved = checks._VOXEL_REACH
        checks._VOXEL_REACH = reach
        try:
            return checks._shared_surface_voxels(bodies[0], bodies[1], grid, shared)[0]
        finally:
            checks._VOXEL_REACH = saved

    assert count(math.sqrt(3.0) / 2.0) == 8
    assert count(0.5) == 0
    # The common E node at the midpoint adds one more shared voxel in C11.
    with pytest.raises(ValueError, match='surfaces pass through the same voxel: 9'):
        _quiet(fd.plan, _grid_scene(pair))


def test_a_smoothed_shape_on_a_one_cell_invariant_axis_is_refused():
    """``Axis1D`` is a wrapping face too, and its refusal is not the image-ring one.

    ``boundaries._WRAPPING`` is ``(Periodic, Bloch, Axis1D)`` and the door used to name only
    the first two, so the identical scene was refused on a one-cell ``Periodic`` y axis and
    accepted on a one-cell ``Axis1D`` one. Held open, it lowers to ``be`` of shape
    ``(3, 24, 1, 24)`` holding **18.4%** of the ball's volume -- the slab cross-section, which
    is neither the ball asked for nor a cylinder -- and no warning fires.

    Its sentence is its own, because the reason is not the missing images: an ``Axis1D`` axis
    stands for one the field is *invariant* along, so no fold over any number of images would
    make a body that curves right on it. That is why it names ``fd.Region`` and C4's
    ``fd.Cylinder``/``fd.Prism`` rather than "keep it one cell clear of the face".
    """
    def scene(y, structure):
        return fd.Scene(
            domain=fd.Domain(size=(1.2, DX, 1.2)),
            boundaries=fd.Boundaries(x=fd.Periodic(), y=y, z=fd.PEC()),
            resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4), structures=(structure,))

    ball = fd.Structure(fd.Sphere(radius=0.2), 4.0)
    with pytest.raises(ValueError, match='fdtd.InvariantAxis face') as e:
        _quiet(fd.plan, scene(fd.Axis1D(k=0.3, k_max=1.0), ball))
    assert 'invariant' in str(e.value) and 'fd.Cylinder' in str(e.value)
    with pytest.raises(ValueError, match='periodic face'):
        _quiet(fd.plan, scene(fd.Periodic(), ball))
    # a cell-snapped shape on the same axis is what that axis is for, and is untouched
    _quiet(fd.plan, scene(fd.Axis1D(k=0.3, k_max=1.0),
                          fd.Structure(fd.Region(size=(0.4, 10.0, 0.4)), 4.0)))


def test_a_smoothed_shape_that_leaves_the_domain_is_refused():
    """Door 3's first arm: material in a voxel that lies **wholly outside** the box.

    Which three of the six components read that material, and whether they gain it or lose it
    along with the rest, flips between the two ends of an axis -- so a body that leaves the box
    is none of the shapes a reader would name, and the numbers are in
    ``test_the_volume_a_clipped_ball_would_have_lowered`` and
    ``test_the_two_ends_of_an_axis_disagree_in_opposite_directions``. The arm fires at a ``-``
    face and at a ``+`` face alike, because **the cause and not the side is what it is drawn
    on**: the second scene here sticks 3.8 cells past a ``+`` wall.

    **The exemption is why the test is on the body and not on the support box.** A shape that
    is uniform *across* the face loses nothing to the clip, answers ``uniform_along`` ``True``
    there and is let through -- which is what ``_HalfSpace``, the co-location oracle of this
    whole file, is: it runs out of the box on five faces. So this gate asserts both
    directions, and the mutant that drops the exemption takes the co-location gates with it.

    **The last case is a ball that swallows the whole domain**, and it is here because it is
    the only one that reaches the outside layer with *no* surface anywhere near it: every
    voxel checked is deep inside the body. It is what makes the door's cheap first stage
    ``d <= reach`` and not ``|d| <= reach`` -- the absolute value marks a voxel the surface
    passes through and steps straight over one the body fills whole, and the mutant that
    writes it that way is invisible to every other scene in this file.

    That ball would in fact have lowered *correctly*, to a uniform medium, and it is refused
    all the same. Door 3 is drawn on the one statement that can be made exactly -- the body's
    material may not reach past a face -- and that statement is true here. The refusal names
    the spelling: a medium that fills the whole box is ``Scene(background=...)``.
    """
    for centre, face in (((-0.6, 0.0, 0.0), 'x-'), ((0.0, 0.59, 0.0), 'y+')):
        with pytest.raises(ValueError, match='leaves the domain') as e:
            _quiet(fd.plan, _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=centre),
                                                      2.25),)))
        message = str(e.value)
        assert f'leaves the domain through the {face} face' in message, message
        assert 'wholly outside the domain' in message, message
        assert 'uniform ACROSS the face' in message, message
        # ... and never the other arm's words, on either side
        assert 'is inside the domain' not in message, message
        assert 'nothing sticking out to truncate' not in message, message
    # tangent to the wall is inside the box, and a half space across it is exempt
    _quiet(fd.plan, _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=(-0.4, 0.0, 0.0)),
                                              2.25),)))
    _quiet(fd.plan, _grid_scene((fd.Structure(_HalfSpace(-0.1), 2.25),)))
    # a ball that contains the whole box: the only scene whose outside voxels are all interior
    with pytest.raises(ValueError, match='leaves the domain') as e:
        _quiet(fd.plan, _grid_scene((fd.Structure(fd.Sphere(radius=10.0), 2.25),)))
    assert 'fills 1.0000 of a voxel' in str(e.value)


def test_door_three_splits_by_cause_and_not_by_side():
    """The two arms answer two questions, and a ``+`` face can be either of them.

    ``_reaches_outside`` probes the cell layer one cell beyond each face the support reaches.
    At a ``-`` face all six components' probe voxels lie wholly outside the domain. At a ``+``
    face they do **not**: the three whose Yee offset along the axis is half a cell sit at
    ``wall + dx/2``, so their voxel is ``[wall, wall + dx]`` and is wholly outside, while the
    other three sit exactly on the wall and theirs is ``[wall - dx/2, wall + dx/2]``, half
    inside. So the side does not say what went wrong and the probe voxel does, and this gate
    reads the probe itself rather than trusting the message.

    Measured here, both ways round at the ``x+`` / ``y+`` wall of this file's 1.2 box:

        scene                                     wholly-outside probes   straddling probe
        Sphere(0.2, (0, 0.59, 0)) -- 3.8 cells out   1.000000 (37, 30,       1.000000
                                                      30 voxels at fill 1)
        Sphere(0.2, (0.376, 0, 0)) -- inside         0.000000                0.002509

    **The defect this replaces** split on ``side`` alone, so the first row got the second's
    message: a body with three probe voxels wholly outside the domain and full of material was
    told that "nothing outside the box is being counted here" and that "there is nothing
    sticking out to truncate". Both halves of the table are asserted, so neither the mutant
    that folds the straddling probe into the outside one (the old code) nor the one that folds
    the outside probes into the straddling one survives.
    """
    from wavebit.fdtd._prepare import checks
    grid = _quiet(fd.plan, _grid_scene(())).grid
    for shape, axis, away, straddling in (
            (fd.Sphere(radius=0.2, center=(0.0, 0.59, 0.0)), 1, 1.0, 1.0),
            (fd.Sphere(radius=0.2, center=(0.376, 0.0, 0.0)), 0, 0.0, 0.002509)):
        body, box = shape.body(), shape.support(grid)
        beyond = [(box[b][0] - 1, box[b][1] + 1) for b in range(3)]
        beyond[axis] = (grid.n[axis], grid.n[axis] + 1)
        wall = grid.low[axis] + grid.n[axis] * grid.dx
        got = {True: 0.0, False: 0.0}
        for field in ('E', 'H'):
            for c in range(3):
                points = np.asarray(yee_points(grid, tuple(beyond), field, c))
                fill = np.asarray(body_occupancy(body, points, grid.dx)[0])
                outside = bool(OFFSETS[field][c][axis])
                # the classification is a fact about the point, so check it and do not assume
                point = float(points.reshape(-1, 3)[0][axis])
                assert (point - 0.5 * grid.dx >= wall - 1e-12) is outside, (field, c, point)
                got[outside] = max(got[outside], float(np.max(fill)))
        assert got[True] == pytest.approx(away, abs=1e-6), (shape, got)
        assert got[False] == pytest.approx(straddling, abs=1e-6), (shape, got)
        cause = checks._reaches_outside(shape, body, grid, box)[0]
        assert cause is (checks._LEAVES if away else checks._TRIMMED), (shape, cause)

    # x+ trims first in axis order, but the later y+ face actually leaves the domain.
    shape = fd.Sphere(radius=0.6, center=(0.0, 0.1, 0.0))
    got = checks._reaches_outside(shape, shape.body(), grid, shape.support(grid))
    assert got[:3] == (checks._LEAVES, 1, '+'), got


def test_the_trimmed_refusal_names_the_material_it_loses():
    """Door 3's second arm: the body is **inside** the box, and the message says so.

    At a ``+`` face the three components whose Yee offset along the axis is zero have their
    last point on the last node, so their last voxel stops at ``high - dx/2`` and material
    between there and the wall is **lost** -- with nothing outside the box at all. So
    ``fd.Sphere(radius=0.2, center=(0.376, 0, 0))``, whose surface is **0.48 cells clear** of
    the ``x+`` wall and which never leaves the box, is refused -- and its mirror image at the
    ``x-`` wall, and a ball tangent to the ``x-`` wall from inside, are accepted. That
    asymmetry is the grid's and is kept; what this gate holds is that the message names it,
    quotes its shortfall as the **relative** number it is, and offers only the remedies that
    work for a body already inside the domain.

    What it costs is in ``test_the_plus_face_loses_the_last_half_cell``.
    """
    inside = fd.Sphere(radius=0.2, center=(0.376, 0.0, 0.0))
    assert inside.high[0] < N * DX / 2, 'the ball must be entirely inside the box'
    with pytest.raises(ValueError, match='is inside the domain') as e:
        _quiet(fd.plan, _grid_scene((fd.Structure(inside, 2.25),)))
    message = str(e.value)
    assert 'LOST' in message, message
    assert 'leaves the domain through' not in message, message   # the other arm's opening
    assert 'wholly outside' not in message, message
    assert 'Nothing is outside the box' in message, message
    assert 'may be entirely inside and this still happens' in message, message
    assert 'truncate' in message and 'already inside' in message, message
    assert 'half a cell (0.025) clear of the x+ wall' in message, message
    # the shortfall is quoted as a fraction of the volume, and said to be one
    assert '1.872e-05 OF that volume (6.27e-07 of volume)' in message, message
    assert '1.114e-02 of it (3.73e-04) at tangency' in message, message
    # the mirror image, and a ball tangent to the x- wall from inside, are both accepted
    for cx in (-0.376, -0.3999):
        _quiet(fd.plan, _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=(cx, 0.0, 0.0)),
                                                  2.25),)))


def test_the_two_ends_of_an_axis_disagree_in_opposite_directions(monkeypatch):
    """The numbers the ``leaves`` arm quotes: one body, one wall, six answers.

    ``Sphere(radius=0.2, center=(0, 0.59, 0))`` sticks **3.8 cells** past the ``y+`` wall of
    this file's 1.2 box, so 53.75% of ``4 pi R^3/3`` is inside it. Held open::

        component set             y+ centre (0, 0.59, 0)      y- mirror (0, -0.59, 0)
        Ey, Hx, Hz  (offset 1/2)      0.01801075                  0.01801075
        Ex, Ez, Hy  (offset 0)        0.01487374  (-17.4%)        0.02110849  (+17.2%)

    The first row is the same at both ends and is exactly the part inside the wall -- those
    three components' voxels stop on the wall at both ends. The second is short at ``y+``,
    because their last voxel stops half a cell inside, and over at ``y-``, because their first
    voxel starts half a cell outside. **Which three components are wrong, and in which
    direction, is a property of the end of the axis**, which is the whole reason the refusal
    cannot be written as "three of them count material beyond the wall".
    """
    from wavebit.fdtd._prepare import checks
    monkeypatch.setattr(checks, '_reaches_outside', lambda *a, **k: None)
    truth = 4 / 3 * math.pi * 0.2 ** 3
    got = {}
    for cy in (0.59, -0.59):
        scene = _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=(0.0, cy, 0.0)), 2.25),))
        plan = _quiet(fd.plan, scene)
        occ = occupancy(scene, plan, runtime('numpy'), fields=('E', 'H'))
        got[cy] = {f'{f}{"xyz"[c]}': float(np.sum(occ[0][f][c][0])) * plan.dx ** 3
                   for f in ('E', 'H') for c in range(3)}
    cap = math.pi * 0.19 ** 2 * (3 * 0.2 - 0.19) / 3          # the part outside the y+ wall
    for cy in (0.59, -0.59):
        for name in ('Ey', 'Hx', 'Hz'):
            assert got[cy][name] == pytest.approx(truth - cap, rel=1e-6), (cy, got[cy])
            assert got[cy][name] == pytest.approx(0.01801075, abs=5e-8), (cy, got[cy])
    for name in ('Ex', 'Ez', 'Hy'):
        assert got[0.59][name] == pytest.approx(0.01487374, abs=5e-8), got[0.59]
        assert got[-0.59][name] == pytest.approx(0.02110849, abs=5e-8), got[-0.59]
    assert got[0.59]['Ex'] / got[0.59]['Ey'] - 1 == pytest.approx(-0.174, abs=1e-3)
    assert got[-0.59]['Ex'] / got[-0.59]['Ey'] - 1 == pytest.approx(+0.172, abs=1e-3)


def test_the_plus_face_loses_the_last_half_cell(monkeypatch):
    """Hold door 3 open at a ``+`` face and read what the three node-centred components lose.

    The same invariant as the ``-`` arm's measurement, from the other end: each component's
    fill should sum to ``4 pi R^3/3``. At a ``+`` face ``Ex`` (whose points sit half a cell
    inside the wall) is exact and ``Ey``/``Ez`` (whose points sit on the nodes) fall **short**,
    which is the opposite sign to the ``-`` face, where they came out 18.7% **over**.

        surface clear of the x+ wall     Ex          Ey = Ez
        0.48 cells                    +2.2e-16      -1.872e-05
        0.002 cells (tangent)         +2.2e-16      -1.114e-02

    and the mirror of each at the ``x-`` wall loses nothing at all (worst 1.3e-15). Those two
    numbers are what the ``+`` arm's message quotes.
    """
    from wavebit.fdtd._prepare import checks
    monkeypatch.setattr(checks, '_reaches_outside', lambda *a, **k: None)
    truth = 4 / 3 * math.pi * 0.2 ** 3
    got = {}
    for cx in (0.376, -0.376, 0.3999, -0.3999):
        scene = _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=(cx, 0.0, 0.0)), 2.25),))
        plan = _quiet(fd.plan, scene)
        occ = occupancy(scene, plan, runtime('numpy'), fields=('E',))
        got[cx] = [float(np.sum(occ[0]['E'][c][0])) * plan.dx ** 3 / truth - 1.0
                   for c in range(3)]
    assert abs(got[0.376][0]) < 1e-15, got[0.376]
    assert got[0.376][1] == pytest.approx(-1.872e-05, rel=2e-3), got[0.376]
    assert got[0.376][1] == pytest.approx(got[0.376][2], rel=1e-9), got[0.376]
    assert got[0.3999][1] == pytest.approx(-1.114e-02, rel=2e-3), got[0.3999]
    for cx in (-0.376, -0.3999):
        assert max(abs(v) for v in got[cx]) < 1e-14, got[cx]


def test_the_volume_a_clipped_ball_would_have_lowered(monkeypatch):
    """Hold door 3 open and read the six components' volumes: they do not agree.

    The invariant every other smoothed gate rests on -- each component's fill sums to the
    body's volume -- is what fails here, and by how much is the number the refusal quotes.
    ``Sphere(radius=0.2)`` centred exactly on the ``x-`` PEC face of a 1.2 box at
    ``dx = 0.05``: ``Ex``, whose points sit half a cell inside the wall, sums to
    **0.01675516**, exactly half the ball (``2 pi R^3 / 3``); ``Ey`` and ``Ez``, whose points
    sit **on** the wall, sum to **0.01988039** -- **18.7%** more, the half cell beyond the
    wall counted in.
    """
    from wavebit.fdtd._prepare import checks
    monkeypatch.setattr(checks, '_reaches_outside', lambda *a, **k: None)
    scene = _grid_scene((fd.Structure(fd.Sphere(radius=0.2, center=(-N * DX / 2, 0.0, 0.0)),
                                      2.25),))
    plan = _quiet(fd.plan, scene)
    occ = occupancy(scene, plan, runtime('numpy'), fields=('E',))
    volume = [float(np.sum(occ[0]['E'][c][0])) * plan.dx ** 3 for c in range(3)]
    half = 2 / 3 * math.pi * 0.2 ** 3
    assert volume[0] == pytest.approx(half, rel=1e-12), volume
    assert volume[1] == pytest.approx(0.01988039, abs=5e-8), volume
    assert volume[1] == pytest.approx(volume[2], rel=1e-12), volume
    assert volume[1] / volume[0] == pytest.approx(1.187, abs=1e-3), volume


def test_an_anisotropic_material_beside_a_smoothed_shape_is_refused():
    """The mixing rule is derived for two **isotropic** media, and says so at the door.

    Blending each component separately with that component's own ``eps`` is not the
    tau-transform the Kottke construction calls for when a phase is anisotropic, and on a ball
    the surface normal is almost never along a principal axis -- so the case it is wrong for is
    the generic one, not a corner. Held open, a ``Sphere`` with ``eps=(2.0, 5.0, 9.0)`` lowers
    with per-component maxima ``[2.0, 5.0, 9.0]`` and no sign of the missing derivation.

    The door covers the same three media as the conductive one -- the shape's own material,
    the background and any structure sharing its cells -- because the rule's *second* phase is
    whatever the cell already holds, so an anisotropic background is the same defect from the
    other side. ``mu`` is read as well as ``eps``, because the same rule runs at the H points.
    """
    aniso = fd.Material(eps=(2.0, 5.0, 9.0))
    ball = fd.Sphere(radius=0.2)
    cases = (((fd.Structure(ball, aniso),), None),
             ((fd.Structure(ball, 2.25),), aniso),
             ((fd.Structure(ball, 2.25), fd.Structure(fd.Region(size=(0.5,) * 3), aniso)), None),
             ((fd.Structure(ball, fd.Material(eps=2.25, mu=(1.0, 1.0, 2.0))),), None))
    for structures, background in cases:
        scene = fd.Scene(
            domain=fd.Domain(size=(N * DX,) * 3),
            boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
            resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
            background=background, structures=structures)
        with pytest.raises(ValueError, match='anisotropic') as e:
            _quiet(fd.plan, scene)
        assert 'tau-transform' in str(e.value)
    # the same anisotropy on cell-snapped geometry is untouched: nothing is mixed there
    _quiet(fd.plan, _grid_scene((fd.Structure(fd.Region(size=(0.4,) * 3), aniso),)))
    # and a scalar eps spelled as a triple of equal numbers is isotropic, not anisotropic
    _quiet(fd.plan, _grid_scene((fd.Structure(ball, fd.Material(eps=(2.25, 2.25, 2.25))),)))


# ---------------------------------------------------------------------------------------------
# What the kernel could not answer exactly: the bound report, from the body to the user
# ---------------------------------------------------------------------------------------------

class _BoundedBall(BallBody):
    """A ball that reports bounds -- the producer side of the report, written here.

    ``fd.Sphere`` itself uses an exact clipped-ball volume. This test double isolates report
    aggregation from the actual CSG geometry tested in ``test_fd_occupancy_integration.py``.
    It answers exactly like its parent and then reports a known count and width through
    the producer interface: ``parts(..., report=)`` and
    ``report.record(node, count, width)``.
    """

    #: The count and the width one call records, chosen to be recognisable in the message.
    RECORDS = (7, 0.0625)

    def parts(self, points, dx, surface=None, report=None):
        out = super().parts(points, dx, surface=surface, report=report)
        if report is not None:
            report.record(type(self).__name__, *self.RECORDS)
        return out


class _ReportingBall(_Ball):
    """A ``_Ball`` whose body is the reporting one."""

    def body(self):
        return _BoundedBall(radius=self.radius, center=self.center)

    def describe(self):
        return f'ReportingBall(bound={self.bound}, center={self.center})'


def test_a_body_that_took_bounds_is_reported_once_for_the_scene():
    """Counts add, widths do not; plan warns once and reports also work with given params.

    The synthetic body isolates scene aggregation. Real CSG and compiled lowering are
    checked independently in test_fd_occupancy_integration.py.
    """
    from wavebit.fdtd._prepare.report import report as describe
    from wavebit.fdtd._shapes import Approximations
    from wavebit.fdtd._prepare.occupancy import bound_note

    quiet = _grid_scene((fd.Structure(fd.Sphere(radius=0.2), 2.25),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fd.plan(quiet)
    assert not [w for w in caught if issubclass(w.category, fd.BoundedVoxelWarning)]

    for count, shapes in (
        # Three E Yee locations and the common E node: four calls of seven.
        (28, (_ReportingBall(0.2),)),
        (56, (_ReportingBall(0.2, center=(-0.25, 0., 0.)),
              _ReportingBall(0.2, center=(0.25, 0., 0.)))),
    ):
        scene = _grid_scene(tuple(fd.Structure(s, 2.25) for s in shapes))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plan = fd.plan(scene)
        mine = [w for w in caught if issubclass(w.category, fd.BoundedVoxelWarning)]
        assert len(mine) == 1
        message = str(mine[0].message)
        assert f'{count} voxel(s)' in message
        assert '0.0625 of a voxel' in message
        assert 'Intersection has not' in message
        with warnings.catch_warnings(record=True) as lowered:
            warnings.simplefilter('always')
            params = fd.lower(scene, plan, 'numpy')
        assert not [w for w in lowered if issubclass(w.category, fd.BoundedVoxelWarning)]
        for given in (None, params):
            printed = describe(scene, plan, given)
            assert printed.warnings.count(message) == 1
            assert message in str(printed)

    report = Approximations()
    assert report.voxels == 0 and bound_note(report) is None
    report.record('empty', 0, 0.0)
    assert bound_note(report) is None
    occupancy(scene, plan, runtime('numpy'), fields=('E',), report=report)
    assert (report.voxels, report.width) == (42, 0.0625)
