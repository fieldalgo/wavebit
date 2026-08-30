"""Sliding the raster by whole pixels must not move the answer.

A unit cell is periodic in both directions -- topologically a torus, not a rectangle -- so
rolling its raster by an integer number of pixels is an exact lattice translation of one
structure, not a different structure. Every efficiency must come back bit-for-bit. Li's rules
do (1e-15 throughout); the NVF path did not, by ~1e-2, at every ``truncation_order``.

**Why it does not converge away, which is what makes it a defect rather than a residual.** The
lift to the ``4*truncation_order+1`` floor is a block repeat -- the same staircase spelled with more pixels
(``docs/nvf-upsampling.md``) -- so raising ``truncation_order`` does not refine the normal field. It only
lets the convolution matrices read more orders of a field that is wrong by the same amount, and
the measured error wanders between 3e-3 and 1.7e-2 without a trend.

**Three ingredients had to be fixed, and no partial fix helps.** The ``'distance'`` construction
read the cell as a flat array in three places: ``_interface_mask`` compared only inside the
array, ``distance_transform_edt`` measured to the in-cell interface rather than the periodically
nearest one, and ``np.gradient`` is one-sided on the first and last row and column. Fixing any
one or two of them still leaves 7e-2 to 1e0 on some case here; only all three give exact zero.
That is why this file parametrizes over three *placements* of the same shape -- a partial fix
passes some of them.

**It is not a feature-touches-the-edge bug**, which is the natural guess and the wrong one.
``test_a_shape_that_never_approaches_the_wall_is_covariant_too`` is the case that rules it out:
the distance map is nonzero right up to the array border whatever the shape does, so the
one-sided gradient corrupts wall pixels regardless.

``'smooth'`` was already exactly covariant before any of this -- it already tiled 3x3 and
smoothed with ``mode='wrap'`` -- and is kept here as the control that says periodicity is the
right answer rather than a tolerance being tuned. ``'spectral'`` leaked only through the shared
mask. ``'contour'`` had a second defect of its own, in the aperture its level set is extracted
through, and needed both a full-cell pad and the clipped-fragment handling; the shapes that
separate those two halves are parametrized at the end of this file.

The sibling defect in the *analytic* (non-raster) path is older and pinned elsewhere:
``test_shape_position_gradients.test_a_multishape_layer_is_periodic_across_the_cell_edge`` --
which pinned the defect under the opposite name until ``c128fd8`` fixed it.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import normal_vector_field

SOLVE_KW = dict(backend=0, top=1.0, bot=1.0, thickness=[120.0], wavelength=250.0,
                theta=np.deg2rad(25.0), phi=np.pi / 2, psi=0.7, period=[500.0, 800.0])

#: Sign is free -- the factorization consumes ``n n^T`` only -- so covariance is compared on the
#: three independent products rather than on the components.
def _frame_defect(a, b):
    (ax, ay), (bx, by) = a, b
    return float(np.maximum.reduce([np.abs(ax * ax - bx * bx),
                                    np.abs(ay * ay - by * by),
                                    np.abs(ax * ay - bx * by)]).max())


def _field_defect(eps, shift, axis, method, lattice=None):
    """Roll the raster, take the field, roll it back, compare against the unrolled field."""
    ref = normal_vector_field(eps, method=method, lattice=lattice)
    got = normal_vector_field(np.roll(eps, shift, axis=axis), method=method, lattice=lattice)
    return _frame_defect(ref, tuple(np.roll(c, -shift, axis=axis) for c in got))


def _block(ny=16, nx=16, rows=(5, 10), cols=(4, 9), value=3.48):
    eps = np.ones((ny, nx))
    eps[rows[0]:rows[1], cols[0]:cols[1]] = value
    return eps


#: One shape, three placements after the roll. A partial fix passes some and fails others, so
#: all three are needed to hold the whole construction periodic.
PLACEMENTS = [
    pytest.param(4, 1, id='stays-interior'),
    pytest.param(7, 1, id='lands-flush-against-the-wall'),
    pytest.param(9, 1, id='crosses-the-wrap'),
    pytest.param(5, 0, id='crosses-the-wrap-in-y'),
]

COVARIANT_METHODS = ['distance', 'contour', 'spectral', 'smooth']


@pytest.mark.parametrize('method', COVARIANT_METHODS)
@pytest.mark.parametrize('shift,axis', PLACEMENTS)
def test_the_normal_field_is_covariant_under_a_lattice_translation(method, shift, axis):
    """The field itself, before any solver enters -- the tightest place to state it."""
    defect = _field_defect(_block(), shift, axis, method)
    assert defect < 1e-12, f"{method}: n n^T moved by {defect:.2e} under a roll of {shift}"


@pytest.mark.parametrize('method', COVARIANT_METHODS)
def test_a_shape_that_never_approaches_the_wall_is_covariant_too(method):
    """The case that rules out 'it only affects features touching the cell edge'.

    Columns 4..8 roll to 8..12 -- three clear pixels from either wall, before and after. The
    pre-fix field still moved by a full 90 degrees at 74 of 256 pixels, because the distance
    map it differentiates does not stop where the shape does: it grows all the way to the array
    border, where ``np.gradient`` switches to a one-sided stencil and the untiled transform
    measures to the wrong copy.
    """
    defect = _field_defect(_block(), 4, 1, method)
    assert defect < 1e-12, f'{method}: interior-only roll moved n n^T by {defect:.2e}'


#: Shapes whose level set exercises the fragment handling: a wrap-crossing rod, two disjoint
#: rods (several loops at once), a ring (nested loops), and a stripe (percolating — no closed
#: 0.5 crossing exists at all, so the open curves are all there is to keep).
def _ring(n=20, outer=8.0, inner=4.0):
    eps = np.ones((n, n))
    gy, gx = np.ogrid[:n, :n]
    r = np.hypot(gx - (n / 2 - 0.5), gy - (n / 2 - 0.5))
    eps[(r <= outer) & (r >= inner)] = 3.48
    return eps


def _two_rods():
    eps = np.ones((16, 16))
    eps[2:6, 2:6] = 3.48
    eps[9:14, 8:13] = 3.48
    return eps


def _stripe():
    eps = np.ones((16, 16))
    eps[:, 4:9] = 3.48
    return eps


@pytest.mark.parametrize('name,eps,shift,axis', [
    ('wrap-crossing rod', _block(), 9, 1),
    ('wrap-crossing rod in y', _block(), 5, 0),
    ('two disjoint loops', _two_rods(), 11, 1),
    ('nested loops', _ring(), 13, 1),
    ('percolating, no closed loop', _stripe(), 7, 1),
])
def test_contour_survives_the_shapes_that_stress_its_fragment_handling(name, eps, shift, axis):
    """``'contour'`` was the last construction to become covariant, and it needed both halves.

    Its window used to be ``4*sigma+3`` px wide, so the level set was extracted through an
    aperture narrower than a period and the roll slid the shape past a fixed cut. Widening it
    to a full cell fixes the shapes that stay clear of the wrap and *not* the ones that cross
    it — measured 1.7e-3 to 1.5e-14 on an interior roll, and 2.1e-3 to 9.8e-4 on a wrapping
    one, sometimes worse. The rest is the fragments: a clipped copy carries endpoints that are
    not curve ends and is resampled by a different rule than a whole loop, so it contributes
    sample positions no whole loop has.

    These five shapes are the ones that separate a half-fix from a whole one. A single centred
    rod does not: it passes on the pad change alone.
    """
    defect = _field_defect(eps, shift, axis, 'contour')
    assert defect < 1e-12, f'{name}: n n^T moved by {defect:.2e}'


#: A skewed cell, where the roll is still an exact lattice translation -- of a lattice whose
#: two directions are not perpendicular. The metric enters every construction here, so this is
#: where a metric applied to the *field* but not to the periodic tiling it is built on would
#: show: the roll would move the answer even though the structure did not move.
HEX = np.column_stack(([500.0, 0.0], [250.0, 250.0 * np.sqrt(3)]))


@pytest.mark.parametrize('method', COVARIANT_METHODS)
@pytest.mark.parametrize('shift,axis', PLACEMENTS)
def test_the_field_is_covariant_under_a_translation_of_a_skewed_lattice(method, shift, axis):
    """The same property on an oblique cell, where all four constructions now run.

    Periodicity and the metric are independent corrections and both are per-construction, so
    holding one on a square cell says nothing about holding it on a skewed one: 'contour'
    folds its level-set loops back by whole periods, 'spectral' copies bulk normals across a
    3x3 tiling, and 'smooth' measures its distance on one -- each of those is a place where a
    tiling that was built in index space and a field that is measured in Cartesian space can
    part company without either looking wrong on its own.
    """
    defect = _field_defect(_block(), shift, axis, method, lattice=HEX)
    assert defect < 1e-12, f'{method}: n n^T moved by {defect:.2e} under a roll of {shift}'


@pytest.mark.parametrize('shift,axis', [(7, 2), (5, 1)])
def test_the_solve_is_covariant_on_a_multi_phase_raster(shift, axis):
    """End to end, on the raster that first showed this -- four phases, both roll axes.

    Four phases matter to the *reading* of this test rather than to the physics: a raster with
    more than two distinct values delegates to ``'distance'`` whatever ``normal_method`` says,
    so all four methods return bit-identical numbers here. That delegation is what hid the
    defect's location originally -- ``'contour'`` and ``'spectral'`` looked equally broken while
    neither was being exercised.
    """
    structure = np.ones((1, 16, 16))
    structure[0, 2:7, 3:12] = 3.48
    structure[0, 6:13, 9:13] = 2.10
    structure[0, 11:14, 2:6] = 1.60

    def efficiencies(uc):
        solver = wavebit.rcwa(structure=uc, truncation_order=[5, 3], factorization='nvf', **SOLVE_KW)
        return np.asarray(solver.solve().given.R, dtype=complex)

    moved = np.abs(efficiencies(structure) - efficiencies(np.roll(structure, shift, axis=axis))).max()
    assert moved < 1e-12, f'R moved by {moved:.2e} under an exact lattice translation'


def test_the_defect_did_not_converge_with_max_order_which_is_why_it_was_a_defect():
    """Covariance holds at every truncation, not merely at a large one.

    Stated as a *sweep* deliberately. The pre-fix error did not decay with ``truncation_order`` -- it sat
    between 3e-3 and 1.7e-2 without a trend -- so a single-``truncation_order`` guard could be passed by an
    implementation that merely made the error small somewhere. Raising ``truncation_order`` changes the lift
    factor, and the lift is a block repeat, so a genuinely non-periodic field stays non-periodic
    however fine the grid is.
    """
    structure = np.ones((1, 16, 16))
    structure[0, 2:7, 3:12] = 3.48
    structure[0, 6:13, 9:13] = 2.10

    rolled = np.roll(structure, 7, axis=2)
    for truncation_order in ([3, 2], [5, 3], [8, 5]):
        def efficiencies(uc):
            solver = wavebit.rcwa(structure=uc, truncation_order=truncation_order, factorization='nvf', **SOLVE_KW)
            return np.asarray(solver.solve().given.R, dtype=complex)

        moved = np.abs(efficiencies(structure) - efficiencies(rolled)).max()
        assert moved < 1e-12, f'truncation_order={truncation_order}: R moved by {moved:.2e}'


def _disc(X, Y, r=30.0):
    return np.where(((X - 32) ** 2 + (Y - 32) ** 2) <= r ** 2, 12.25, 1.0)


def _annulus(X, Y, r_out=30.0, r_in=14.0):
    cell = _disc(X, Y, r_out)
    cell[((X - 32) ** 2 + (Y - 32) ** 2) <= r_in ** 2] = 1.0
    return cell


def test_contour_is_periodic_under_a_roll_that_lands_on_the_corner():
    """Rolling a raster is the same structure written from a different corner.

    Every answer must be identical. Three of the four constructions always were, to 4e-14.
    `normal_method='contour'` -- the one `docs/nvf-normal-methods.md` and `README.md` recommend
    -- was not, until 2026-08-29: `sum(T)` moved by 3.9e-4 on a square cell and 9.7e-4 on a 2:1
    cell for the disc below, and did not converge with truncation order.

    **The cause was `_two_phase_indicator`, and it was not in the contour block at all.** That
    function keyed the background on the *corner pixel* -- `eps != eps.flat[0]` -- which is
    invariant under relabeling the two phases, the property its docstring defends, and is not
    invariant under a translation. A roll can put the corner *inside* the feature, and then the
    indicator inverted wholesale. It partitioned these ten rolls exactly:

    ======================================  =============  ================
    roll                                    corner inside  ``sum(T)``
    ======================================  =============  ================
    (0,0) (7,11) (1,63) (2,2) (5,5) (17,3)  no             0.328909134249
    (23,5) (31,17) (40,40) (13,29)          yes            0.329294462911
    ======================================  =============  ================

    Two values, nothing between, each class tight to 5e-13 -- which is the tell, because a
    sample-phase offset gives a spread and not two classes. Orienting the indicator by area
    instead collapses all ten to 4.157e-13.

    **The small disc is here because it refutes how this was first described.** The standing
    account was that `'contour'` is unsafe "where a feature is large compared to the cell". It
    is not a size problem: a disc of radius 12 in the same 64 cell breaks on exactly one roll
    of the ten -- (40,40), which moves its centre to within 12 pixels of the corner -- at
    1.5e-4. Any feature does, given the placement, and a shape written at the origin is an
    ordinary thing to write.

    **Three refuted explanations**, kept because each is what the next person would try.

    * *the tile ring.* `_contour_polylines` folds each loop on its area-mean centre, so a
      loop's bounding box can overshoot the cell by a full period (0.81 to 1.00 cells on these
      scenes) while `contour_normal_field` lays down exactly one ring. Deriving the ring from
      that overshoot moves the field by **5e-14**.
    * *the resample phase.* `_resample` lays its samples from `pts[0]`, wherever contourpy
      entered the loop, which a roll moves. This was the shipped explanation until 2026-08-28,
      and it scoped the fix as a nearest-segment rewrite deleting `_resample`. It is the
      amplifier, not the cause: an inverted indicator is the *same curve*, 236 points either
      way, reached in a different traversal. Canonicalizing the traversal directly was tried
      and is worse -- shoelace direction alone leaves 4.5e-4 and still two classes, and adding
      a lexicographic entry vertex gives 1.5e-3 across ten distinct values, because the entry
      is itself unstable.
    * *the fold's `np.round` half-way tie.* Real -- `np.round(0.5)` is 0, so the centred disc
      keeps two loop representatives where every offset roll keeps one -- and harmless: those
      two agree to 4.7e-13, because the 3x3 tiling regenerates the copies anyway.

    What the fix picks between are two readings that converge, so it is a choice and not a
    correction: the same disc at 64 / 128 / 256 samples a side has its branches 3.85e-4,
    2.51e-5 and 6.73e-7 apart. At exactly half fill the area rule ties and the corner rule
    stands, which a 50/50 stripe shows is harmless (1.144e-14 across these rolls, before and
    after alike). Making the tie flip instead -- `>=` for `>` -- is a mutant this test does
    not kill, and deliberately so: at half fill both readings give the same curve and the
    same traversal, so nothing downstream can see which was taken. The two that matter do
    die -- restoring the corner rule, and marking the larger phase instead of the smaller.
    """
    import wavebit

    n = 64
    ax = np.arange(n)
    X, Y = np.meshgrid(ax, ax)
    cfg = dict(backend=0, thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               period=[1.0, 1.0], normal_method='contour')

    def spread(cell, rolls):
        got = [float(np.sum(np.asarray(wavebit.rcwa(
            structure=np.roll(np.roll(cell, r[0], axis=0), r[1], axis=1)[None],
            **cfg).solve().T))) for r in rolls]
        return max(got) - min(got), got

    # Six rolls whose corner falls outside and four whose corner falls inside -- the partition
    # the defect produced. It is not enough to roll *somewhere*: eight of these ten agree with
    # each other even with the defect in place.
    rolls = ((0, 0), (7, 11), (23, 5), (31, 17), (1, 63), (2, 2), (40, 40), (5, 5),
             (13, 29), (17, 3))
    inside = {(23, 5), (31, 17), (40, 40), (13, 29)}

    from wavebit._core.normal_methods import _two_phase_indicator
    big = np.where(((X - 32) ** 2 + (Y - 32) ** 2) <= 30 ** 2, 12.25, 1.0)

    # Which phase gets marked is a convention, and it is the *smaller* one -- asserted because
    # marking the larger is equally periodic and so passes everything below, while moving every
    # `'contour'` number on a raster whose feature covers more than half the cell.
    assert _two_phase_indicator(big).mean() == pytest.approx(1.0 - 0.6887, abs=1e-3)

    # The corner rule really did invert on exactly those four, and the area rule does not.
    for r in rolls:
        rolled = np.roll(np.roll(big, r[0], axis=0), r[1], axis=1)
        assert bool(rolled.flat[0] > 1.5) == (r in inside), r
        moved = np.abs(_two_phase_indicator(rolled)
                       - np.roll(np.roll(_two_phase_indicator(big), r[0], axis=0),
                                 r[1], axis=1)).max()
        assert moved < 1e-12, f'roll {r}: indicator moved by {moved}'

    moved, got = spread(big, rolls)
    assert moved < 1e-9, f'disc r=30: sum(T) moved by {moved:.2e} across rolls -- {got}'

    # A feature far too small for the "large compared to the cell" story. Only (40,40) reaches
    # the corner, so a test rolling anywhere else would pass with the defect in place.
    small = np.where(((X - 32) ** 2 + (Y - 32) ** 2) <= 12 ** 2, 12.25, 1.0)
    moved, got = spread(small, ((0, 0), (40, 40)))
    assert moved < 1e-9, f'disc r=12: sum(T) moved by {moved:.2e} across rolls -- {got}'

    # Exactly half fill: the area rule ties, the corner rule stands, and it does not matter.
    stripe = np.where(X < 32, 12.25, 1.0)
    moved, got = spread(stripe, rolls)
    assert moved < 1e-9, f'50/50 stripe: sum(T) moved by {moved:.2e} across rolls -- {got}'


@pytest.mark.xfail(strict=True, reason='the medial axis carries no defined normal, and which '
                                       'segment wins that tie moves with the roll')
@pytest.mark.parametrize('name,build,lattice', [
    ('annulus, square cell', _annulus, None),
    ('disc, hex cell', _disc, np.column_stack(([1.0, 0.0], [0.5, np.sqrt(3) / 2]))),
])
def test_contour_is_periodic_where_the_medial_axis_is_reachable(name, build, lattice):
    """What the 2026-08-29 indicator fix does **not** reach, pinned so it is not claimed away.

    Marking the smaller phase makes `'contour'` exactly periodic for a *single* loop on an
    orthogonal cell -- 4.2e-13 over the ten rolls below, from 3.9e-4. Two scenes stay broken:

    =========================  ==========  ==========
    scene                      after       before
    =========================  ==========  ==========
    disc, square cell          4.2e-13     3.9e-4
    disc, [1,2] / [2,1] cells  1.2e-13     --
    **annulus, square cell**   **1.2e-5**  7.5e-5
    **disc, hex cell**         **2.3e-4**  3.0e-4
    annulus, hex cell          2.2e-3      2.2e-3
    =========================  ==========  ==========

    `'spectral'`, `'distance'` and `'smooth'` sit at 1e-13 on every one of them, which is what
    makes this `'contour'`'s and not the raster's.

    **It is not another labelling bug, and no tie-break rule fixes it.** Comparing the fields
    pixel by pixel rather than through `sum(T)` -- which is the only way to see this -- the
    annulus disagrees on **one pixel**, at the exact centre of its hole; the hex disc on 1 to
    18 pixels at radius 35 to 42, where the curve is at 30, i.e. *outside* the shape, in the
    deep gaps between its periodic images. Both sets are the **medial axis**: the nearest
    boundary is equidistant in every direction, the normal is genuinely undefined, and which
    segment wins the tie is an accident of where the samples fell. A roll changes the accident.

    A nearest-segment query -- the rewrite this file's docstrings proposed for two weeks --
    hits the same wall from the other side: a prototype differs from the shipped construction
    on exactly 8 pixels, all within 4.2 px of a disc's centre, and comes out *less* periodic
    (2.8e-3, ten distinct values).

    This is the raster twin of the exact-tangency singularity in the analytic path
    (`analytic._blend`): a normal field asked for a direction where geometry defines none.

    The scene tests above do not catch it, which is worth stating -- their nested ring and
    their hex cell both pass. What these two add is a feature large enough that its medial axis
    lands on a *sampled* pixel.
    """
    n = 64
    ax = np.arange(n)
    X, Y = np.meshgrid(ax, ax)
    cell = build(X, Y)
    cfg = dict(backend=0, thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               normal_method='contour')
    cfg['lattice' if lattice is not None else 'period'] = (lattice if lattice is not None
                                                           else [1.0, 1.0])
    rolls = ((0, 0), (7, 11), (23, 5), (31, 17), (1, 63), (2, 2), (40, 40), (5, 5),
             (13, 29), (17, 3))
    got = [float(np.sum(np.asarray(wavebit.rcwa(
        structure=np.roll(np.roll(cell, r[0], axis=0), r[1], axis=1)[None],
        **cfg).solve().T))) for r in rolls]
    assert max(got) - min(got) < 1e-9, f'{name}: sum(T) moved by {max(got) - min(got):.2e}'
