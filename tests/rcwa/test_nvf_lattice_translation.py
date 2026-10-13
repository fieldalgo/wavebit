"""Sliding the raster by whole pixels must not move the answer.

A unit cell is periodic in both directions -- topologically a torus, not a rectangle -- so
rolling its raster by an integer number of pixels is an exact lattice translation of one
structure, not a different structure. Every efficiency must come back bit-for-bit. Li's rules
do (1e-15 throughout); the NVF path did not, by ~1e-2, at every ``truncation_order``.

**Why it does not converge away, which is what makes it a defect rather than a residual.** The
lift to the ``4*truncation_order+1`` floor is a block repeat -- the same staircase spelled with more pixels
(``docs/validation/rcwa/nvf-upsampling.md``) -- so raising ``truncation_order`` does not refine the normal field. It only
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

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest

from wavebit.rcwa._engine.factorization import normal_vector_field

SOLVE_KW = dict(backend='numpy', top=1.0, bot=1.0, thickness=[120.0], wavelength=250.0,
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
    more than two distinct values delegates to ``'distance'`` whatever ``raster_normal_method`` says,
    so all four methods return bit-identical numbers here. That delegation is what hid the
    defect's location originally -- ``'contour'`` and ``'spectral'`` looked equally broken while
    neither was being exercised.
    """
    structure = np.ones((1, 16, 16))
    structure[0, 2:7, 3:12] = 3.48
    structure[0, 6:13, 9:13] = 2.10
    structure[0, 11:14, 2:6] = 1.60

    def efficiencies(uc):
        solver = _legacy_wavebit.rcwa(structure=uc, truncation_order=[5, 3], factorization='nvf', **SOLVE_KW)
        return np.asarray(solver.solve().incident.R, dtype=complex)

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
            solver = _legacy_wavebit.rcwa(structure=uc, truncation_order=truncation_order, factorization='nvf', **SOLVE_KW)
            return np.asarray(solver.solve().incident.R, dtype=complex)

        moved = np.abs(efficiencies(structure) - efficiencies(rolled)).max()
        assert moved < 1e-12, f'truncation_order={truncation_order}: R moved by {moved:.2e}'


def _index_disc(U, V, r=30.0):
    """Round **in index space**, which off a square cell is not round in the cell.

    ``U`` and ``V`` are raster indices, not Cartesian coordinates: the raster is read at
    ``(i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2``, so this mask is the image of a disc under the
    basis. Measured aspect ratio of what it actually builds: 1.00 at 90 degrees, **1.73 at 60,
    3.73 at 30, 5.67 at 20**. Naming the arguments ``X``/``Y`` hid that, and a caveat in
    ``docs/theory/rcwa/oblique-lattice.md`` called the 60-degree case "a disc of radius 30".

    That is fine for what this file measures -- **periodicity is a property of the raster and
    of the code that reads it, not of the shape being round** -- and an ellipse is a legitimate
    structure. It is not fine as a description. Where a genuine Cartesian circle on a skewed
    cell is wanted, ``tests/rcwa/test_nvf_normal_methods.py::skewed_disc`` builds one through
    ``wavebit.meshgrid_xy`` with a minimum-image membership test.
    """
    return np.where(((U - 32) ** 2 + (V - 32) ** 2) <= r ** 2, 12.25, 1.0)


def _index_annulus(U, V, r_out=30.0, r_in=14.0):
    """:func:`_index_disc` with its middle removed -- an elliptical annulus off a square cell."""
    cell = _index_disc(U, V, r_out)
    cell[((U - 32) ** 2 + (V - 32) ** 2) <= r_in ** 2] = 1.0
    return cell


def _two_index_discs(U, V, r=11.0, sep=12.0):
    """Two discs with a saddle between them -- the worst medial axis found so far."""
    return np.where(((U - (32 - sep)) ** 2 + (V - 32) ** 2 <= r ** 2)
                    | ((U - (32 + sep)) ** 2 + (V - 32) ** 2 <= r ** 2), 12.25, 1.0)


def test_contour_is_periodic_under_a_roll_that_lands_on_the_corner():
    """Rolling a raster is the same structure written from a different corner.

    Every answer must be identical. Three of the four constructions always were, to 4e-14.
    `raster_normal_method='contour'` -- the one `docs/theory/rcwa/nvf-normal-methods.md` and `README.md` recommend
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

    All three are written in the tense of the code they were measured against. Two of them no
    longer describe it: the 3x3 window and its dedup were replaced by a seam-duplicated single
    period on 2026-09-01, and `_resample` has no phase left to be the amplifier -- it
    subdivides each segment. Neither retraction touches the finding, which is about the
    indicator and is asserted below.

    What the fix picks between are two readings that converge, so it is a choice and not a
    correction: the same disc at 64 / 128 / 256 samples a side has its branches 3.85e-4,
    2.51e-5 and 6.73e-7 apart. At exactly half fill the area rule still ties;
    the contour consumer now canonicalizes traversal so complementing the
    indicator samples the same curve. The asymmetric half-fill test below
    exercises that fix; a stripe alone cannot detect the old defect.
    """

    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    cfg = dict(backend='numpy', thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               period=[1.0, 1.0], raster_normal_method='contour')

    def spread(cell, rolls):
        got = [float(np.sum(np.asarray(_legacy_wavebit.rcwa(
            structure=np.roll(np.roll(cell, r[0], axis=0), r[1], axis=1)[None],
            **cfg).solve().T))) for r in rolls]
        return max(got) - min(got), got

    # Six rolls whose corner falls outside and four whose corner falls inside -- the partition
    # the defect produced. It is not enough to roll *somewhere*: eight of these ten agree with
    # each other even with the defect in place.
    rolls = ((0, 0), (7, 11), (23, 5), (31, 17), (1, 63), (2, 2), (40, 40), (5, 5),
             (13, 29), (17, 3))
    inside = {(23, 5), (31, 17), (40, 40), (13, 29)}

    from wavebit.rcwa._engine.normal_methods import _two_phase_indicator
    big = np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 30 ** 2, 12.25, 1.0)

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
    small = np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 12 ** 2, 12.25, 1.0)
    moved, got = spread(small, ((0, 0), (40, 40)))
    assert moved < 1e-9, f'disc r=12: sum(T) moved by {moved:.2e} across rolls -- {got}'

    # Half-fill stripe control; the asymmetric blob below is the discriminating case.
    stripe = np.where(U < 32, 12.25, 1.0)
    moved, got = spread(stripe, rolls)
    assert moved < 1e-9, f'50/50 stripe: sum(T) moved by {moved:.2e} across rolls -- {got}'


@pytest.mark.parametrize('name,build,rolls', [
    ('annulus, square cell', _index_annulus,
     ((0, 0), (7, 11), (23, 5), (31, 17), (1, 63), (2, 2), (40, 40), (5, 5), (13, 29), (17, 3))),
    ('two discs, square cell', _two_index_discs,
     ((0, 0), (48, 32), (42, 32), (38, 32), (2, 32))),
])
def test_contour_is_periodic_where_the_medial_axis_is_reachable(name, build, rolls):
    """The two scenes that survived the 2026-08-29 indicator fix, and what settled them.

    **The oblique half of this was a second, unrelated defect and was fixed first** (the tile
    ring; see `test_the_tile_ring_follows_the_cell`). What is left, and what these two scenes
    reach, is the medial axis on an orthogonal cell.

    **How much there was, swept rather than sampled.** With the translation identity there is
    no reason to guess: 6 scenes x 4 constructions x 2 lattices x 1023 rolls, ~49k
    measurements in 395 s. `'distance'`, `'spectral'` and `'smooth'` were clean on every one
    of the 48 combinations, and so was `'contour'` on every oblique one. Exactly two scenes
    survived, both on a square cell:

    ========================  ==============  ==========
    scene                     broken          worst
    ========================  ==============  ==========
    annulus                   621 of 1023     4.06e-05
    **two discs (saddle)**    26 of 1023      **1.20e-03**
    ========================  ==============  ==========

    (Re-taken on 2026-09-01 after `_translation_defect` stopped dividing each slot by its own
    norm; the figures this file carried before that -- 435 of 1023 at 9.87e-05, and 1.70e-02
    -- were larger for that reason and not because the code was different.)

    A disc, a small disc, a stripe and a cross were all clean, on both lattices. The saddle
    between two discs is the worse of the two by two orders of magnitude and was not in this
    file until 2026-08-31 -- the annulus had been standing in for a class whose worst member
    it is not.

    **The saddle's trigger is sharper than "a roll".** All 26 of its broken rolls had the same
    ``roll_u = 32`` -- exactly half the period, along the axis joining the two discs, which is
    the shift that puts the saddle point on the *cell wrap*. Rolling anywhere else was clean,
    which is why the eight rolls first written here all passed and the case looked fixed. The
    rolls below are chosen from the failing set for that reason, and they are kept now that it
    passes for exactly the same reason: a pin that never reached the defect proves nothing.
    Measured on precisely those rolls, before and after: the annulus `sum(T)` 1.19e-05 ->
    4.67e-14, the saddle 3.16e-04 -> 1.61e-12, and the translation identity 9.87e-05 ->
    9.7e-16 and 1.20e-03 -> 1.1e-15.

    **It was never a labelling bug and no tie-break *rule* fixes it.** Comparing the fields
    pixel by pixel rather than through `sum(T)` -- which is the only way to see this -- the
    annulus disagreed on **one pixel**, at the exact centre of its hole; a disc in a hex cell
    on 1 to 18 pixels at radius 35 to 42, where the curve is at 30, i.e. *outside* the shape,
    in the deep gaps between its periodic images. Both sets are the **medial axis**: the
    nearest boundary is equidistant in every direction, the normal is genuinely undefined, and
    which point wins the tie is an accident of where the samples fell. A roll changes the
    accident.

    A nearest-*segment* query -- the rewrite this file's docstrings proposed for two weeks --
    hits the same wall from the other side, and is worth recording as a dead end: a prototype
    differed from the shipped construction on exactly 8 pixels, all within 4.2 px of a disc's
    centre, and came out *less* periodic (2.8e-3, ten distinct values). Exactness in the query
    was never the missing thing.

    **What fixed it, 2026-09-01, and precisely what it does not do.** The tie is not resolved;
    it is *settled the same way every time*, by answering the nearest-sample query in a metric
    sheared by `_CONTOUR_TIE_SHEAR` -- a fixed, translation-invariant, infinitesimal
    perturbation, the standard device for a geometric predicate whose exact ties have to be
    single valued (Edelsbrunner & Muecke 1990). The direction it then reports on the medial
    axis is still arbitrary, and still undefined by the geometry. What is no longer arbitrary
    is *which* arbitrary answer, which is the whole of what covariance asks for. Swept the
    same way as above, 1023 rolls: 0 of 1023 on both, and on the hex cell with them.

    Two things worth keeping about how this arrived. It was **not sought**: the percolation
    fix replaced arc-length resampling, whose phase had been breaking these ties by accident
    all along, and the accident had to be replaced by something deliberate before the
    percolating scenes would hold. And the exactness is the whole difficulty -- a level set
    drawn on a raster inherits the raster's mirrors, so the two competing points are equal to
    the last bit rather than merely close, which is why 1e-15 of round-off decides and why a
    perturbation is what separates them. Turning the shear off puts these two scenes back at
    9.9e-05 on 48 of 63 rolls and 2.9e-04 on 8, and breaks five more with them.

    **The accuracy half is still open.** Both S4 and fmmax answer the null by *not choosing a
    direction*: they open the linear state into an ellipse whose ellipticity follows the field
    magnitude, `eta = (pi/8)(1 + cos(pi|v|))`, so `|v| -> 0` gives an exactly circular state
    whose projector is `Pxx = Pyy = 1/2`, `Pxy = -i/2`, `Pyx = +i/2` -- independent of the
    arbitrary direction, and a *limit* rather than a branch, with no threshold anywhere
    (`fmm_PolBasisJones.cpp:65`, `fmmax/vector.py:285`; Liu & Fan, CPC 183, 2233 (2012)).
    Prototyped here with `|v| = |grad d|`, the distance field's gradient magnitude -- 1 almost
    everywhere, dropping only on the ridge that *is* the medial axis, and independent of which
    construction produced the field. It took the annulus to 4.56e-15 and the saddle down by
    about one order and no further -- the saddle's two nearest boundary points sit on
    *different* discs, so the tie there is a discrete jump rather than a direction that fades
    out. (Read that as a ratio: both sides of it were measured before `_translation_defect`
    stopped dividing each slot by its own norm, so the values it was quoted with, 1.70e-2 and
    1.58e-3, are larger than the same prototype would report today.) It is not landed because it is a trade and not a free win:
    against an exact circle at 512x512 the error goes 3.18e-5 to 3.95e-5. And it needs the
    field API to carry a magnitude (the constructions return unit vectors), complex projector
    coefficients, and distinct `eps_xy` / `eps_yx` -- today one object fills both slots.

    This is the raster twin of the exact-tangency singularity in the parametric path
    (`analytic._blend`): a normal field asked for a direction where geometry defines none.
    The shear makes that answer stable. It does not make it right, and nothing here does.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    cell = build(U, V)
    cfg = dict(backend='numpy', thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               raster_normal_method='contour')
    cfg['period'] = [1.0, 1.0]
    got = [float(np.sum(np.asarray(_legacy_wavebit.rcwa(
        structure=np.roll(np.roll(cell, r[0], axis=0), r[1], axis=1)[None],
        **cfg).solve().T))) for r in rolls]
    assert max(got) - min(got) < 1e-9, f'{name}: sum(T) moved by {max(got) - min(got):.2e}'


def _phase_diagonal(to, shift_v, shift_u, n_v, n_u):
    """``D = diag(exp(-i G.d))`` for an integer pixel shift, in ``conv_matrix_2d``'s order."""
    mv = np.arange(-to, to + 1)
    mu = np.arange(-to, to + 1)
    V, U = np.meshgrid(mv, mu, indexing='ij')
    return np.exp(-2j * np.pi * (V.ravel() * shift_v / n_v + U.ravel() * shift_u / n_u))


def _translation_defect(cell, shift, method, lattice=None, to=3):
    """``max |E(rolled) - D E D^-1| / max|E|`` over the three slots that carry the field.

    **The instrument, and it is worth more than the test it serves.** Translating a periodic
    structure by ``d`` multiplies every Fourier coefficient by ``exp(-i G.d)``, so every
    convolution matrix transforms as a *similarity*, ``E -> D E D^-1``. A covariant builder
    satisfies that exactly -- and the identity holds at **any** truncation, converged or not,
    with **no solve**, no S-matrix and no flux convention in the way. 36 measurements take
    under a second where the equivalent `sum(T)` sweep took twenty minutes.

    ``epz_i`` is the control: it is ``inv(<eps>)`` and reads no normal field at all, so it
    pins the convention itself at 5e-15 whatever the field does.

    It is also what made the defect's *size* honest. Ten hand-picked rolls said the oblique
    residual was 2.2e-3; sweeping all 4095 said 2.7e-2 on 39 of 63, and that the annulus on a
    square cell breaks on 35% of rolls rather than the one in ten a sample had found.

    **One trap, and it is in the normalisation.** Every slot is divided by the *largest* of
    the three, not by its own norm. Dividing a slot by itself reports round-off over round-off
    on a slot that is identically zero -- `epxy` on a stripe is `1.2e-19` against `epx`'s
    `7.8`, and the per-slot spelling calls a perfectly covariant construction **2.09** there
    while `sum(T)` over the same rolls moves 1.2e-14. That cost an hour on 2026-09-01, twice.
    The cost of the shared denominator is that a defect living only in a slot orders below the
    others is under-reported, which is the right trade: such a slot cannot move an answer.
    """
    kw = dict(backend='numpy', thickness=[0.3], truncation_order=(to, to), top=1.0, bot=1.0,
              wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2,
              factorization='nvf', raster_normal_method=method)
    kw['lattice' if lattice is not None else 'period'] = (lattice if lattice is not None
                                                          else [1.0, 1.0])
    n_v, n_u = cell.shape
    rolled = np.roll(np.roll(cell, shift[0], axis=0), shift[1], axis=1)
    A = _legacy_wavebit.rcwa(structure=cell[None], **kw).eps_conv_matrices()
    B = _legacy_wavebit.rcwa(structure=rolled[None], **kw).eps_conv_matrices()
    d = _phase_diagonal(to, shift[0], shift[1], n_v, n_u)
    slots = [(np.asarray(A[k])[0], np.asarray(B[k])[0]) for k in (0, 1, 3)]
    scale = max(max(float(np.abs(a).max()) for a, _ in slots), 1e-30)
    worst = 0.0
    for a, b in slots:                      # epx, epy, epxy -- the slots the field reaches
        pred = (d[:, None] * a) * np.conj(d)[None, :]
        worst = max(worst, float(np.abs(b - pred).max()) / scale)
    return worst


@pytest.mark.parametrize('gamma', [90, 75, 60, 45, 30, 20])
def test_the_tile_ring_follows_the_cell(gamma):
    """One ring of periodic images is enough on a square cell and not on a skewed one.

    `contour_normal_field` answers each pixel with the nearest point of the level set *over
    the periodic images*, so it lays the curve down on a ring of tiles and queries that. The
    ring was fixed at 3x3 until 2026-08-30.

    Measured with that in place, disc of radius 30 in a 64 cell at 60 degrees: 2.68e-2, on 39
    of 63 rolls, where `'distance'`, `'spectral'` and `'smooth'` all held at 1e-15. Deriving
    the ring from the lattice takes every angle here to the floor. That much is solid and is
    what this test pins.

    **The mechanism, and the trap in measuring it.** One ring genuinely fails to reach the
    nearest point of the level set on a skewed cell -- but *not at every placement*, which is
    what makes it hard to see. On the unrolled disc the one-ring nearest distance equals the
    true minimum at every angle (excess at most 2.2e-14 on 0 of 4096 pixels), so a single
    measurement there says the ring is sufficient and it is wrong. Rolling the same raster:

    ==========  ==========  ================  =================
    cell        roll (0,0)  roll (2,2)        roll (1,63)
    ==========  ==========  ================  =================
    90 degrees  0 px        0 px              0 px
    60 degrees  0 px        14 px, 2.60 far   18 px, 2.99 far
    20 degrees  0 px        96 px, 11.9 far   65 px, 3.87 far
    ==========  ==========  ================  =================

    Those pixel counts are exactly the pixels whose field changes between ring 1 and ring 2
    (14 / 1 / 18 at 60 degrees), which closes the chain: skewed cell -> one ring misses at
    *some* placements -> those pixels take a farther point -> the field depends on placement.
    `test_one_ring_misses_the_nearest_point_only_at_some_placements` pins the middle step.

    The ring it picks is 2 on a square cell -- one more than the code used to take. That is
    not a no-op: an orthogonal cell moves by 4.3e-14, which is the kd-tree choosing between
    samples at equal distance. The cost is 3 ms -> 6 ms on the field build at N=64.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    disc = np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 30 ** 2, 12.25, 1.0)
    t = np.deg2rad(gamma)
    lattice = None if gamma == 90 else np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)]))
    for shift in ((1, 1), (2, 3), (3, 0), (0, 2), (5, 7)):
        got = _translation_defect(disc, shift, 'contour', lattice)
        assert got < 1e-12, f'gamma={gamma}, roll {shift}: E moved by {got:.2e}'


def test_the_translation_identity_holds_for_every_construction():
    """The instrument itself, on all four fields and on the permittivity-only control.

    A defect this pins that `sum(T)` cannot: it names the *slot*. `epz_i` is `inv(<eps>)` and
    reads no field, so it holds at 5e-15 no matter what any construction does -- which is how
    a field defect is told apart from a convention error in the identity itself.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    disc = np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 30 ** 2, 12.25, 1.0)
    hexl = np.column_stack(([1.0, 0.0], [0.5, np.sqrt(3) / 2]))
    for lattice in (None, hexl):
        for method in ('distance', 'contour', 'spectral', 'smooth'):
            got = _translation_defect(disc, (13, 29), method, lattice)
            assert got < 1e-12, f'{method}: E moved by {got:.2e}'


def test_the_ring_count_is_read_from_the_cell_not_assumed():
    """`_tile_rings`' two terms, pinned separately, because each hides the other's absence.

    The covering term alone is right on a square cell and the stray term alone is right at 45
    degrees, so a test that only checks the *outcome* at a few angles passes with either one
    deleted. These are the values the formula owes:

    ==========  ==========  ============  =======  =====
    gamma       covering/   shortest      stray    rings
                shortest    is
    ==========  ==========  ============  =======  =====
    90          0.707       |a1| = 1      1        2
    60          0.866       |a1-a2| = 1   1        2
    45          1.207       |a1-a2|=0.77  1        3
    20          2.836       |a1-a2|=0.35  1        4
    135         1.208       |a1+a2|=0.77  1        3
    160         2.838       |a1+a2|=0.35  1        4
    ==========  ==========  ============  =======  =====

    The `shortest` column is why the minimum runs over `a1 +- a2` and not just the two basis
    vectors: away from 90 degrees the *diagonal* is the shortest lattice vector, and using
    `|a1|` instead gives 2 where the geometry needs 4 at 20 degrees -- a mutant the angle
    sweep above does not kill, because two rings happen to be enough for the one disc it draws.

    The **obtuse** rows are here for the other half of the same point: `covering` is half the
    *longer* diagonal, and which diagonal that is swaps at 90 degrees. Taking `|a1+a2|` alone
    is right for every acute cell and gives 1 where 2 and 3 are needed at 135 and 160.
    """
    from wavebit.rcwa._engine.normal_methods import _tile_rings

    n = 64
    # samples exactly inside the cell: the stray term is then 0 and the covering term is alone
    for gamma, want in ((90, 1), (60, 1), (45, 2), (20, 3), (135, 2), (160, 3)):
        t = np.deg2rad(gamma)
        M = np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)])) / n
        pts = (M @ np.array([[1.0, 1.0], [n - 1.0, n - 1.0]]).T).T
        assert _tile_rings(M, pts, n, n) == want, f'gamma={gamma}'

    # and the stray term: push one sample a full cell out, expect exactly one more ring
    t = np.deg2rad(90.0)
    M = np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)])) / n
    inside = (M @ np.array([[1.0, 1.0], [n - 1.0, n - 1.0]]).T).T
    outside = np.vstack([inside, (M @ np.array([[1.4 * n], [0.5 * n]])).T])
    assert _tile_rings(M, inside, n, n) == 1
    assert _tile_rings(M, outside, n, n) == 2


@pytest.mark.parametrize('gamma', [90, 60, 45, 20])
def test_one_ring_reaches_the_nearest_point_once_the_fold_is_right(gamma):
    """The tile ring is not a coverage question any more, and the history is why this exists.

    `contour_normal_field` answers each pixel with the nearest point of the level set over the
    periodic images, and lays the curve on a ring of tiles to do it. That ring was a hard-coded
    3x3, and on a skewed cell `'contour'` came back non-periodic under a rigid translation of
    the raster -- 2.68e-2 in `sum(T)` at 60 degrees, on 39 of 63 rolls, where the other three
    constructions held at 1e-15.

    The first fix widened the ring, deriving it from the covering radius over the shortest
    lattice vector: 2 rings at 90 degrees, 3 at 45, 4 at 20, 13 (729 tiles) at 5. That worked
    and it treated the symptom. The cause was one line in `_contour_polylines`, which folded
    each loop onto the **origin** rather than onto the cell -- `round(c / n) * n` lands a centre
    in `[-n/2, n/2)`, so the representative straddled the cell's corner and could sit a full
    period outside the cell it describes (index centre (-30, -30) for a 64 cell, every sample
    at a negative index). Tiles are laid around the *representative*, so the cell sat at the
    edge of the tiled region instead of its middle.

    With the fold corrected, **one ring misses nothing**: 0 of 4096 pixels at 90 / 60 / 45 / 20
    degrees, on a disc and on a percolating stripe, at five placements each. That is what this
    asserts, and it is the statement the covering-radius rule was standing in for.

    `_tile_rings` was then kept for a second reason that has since gone away: with a single
    ring the percolating stripe was still non-periodic at 45 and 20 degrees (61 and 62 of 63
    rolls) -- *not* because a nearest point was missed, which this test rules out, but because
    open curves got no fold and no dedup, so the padding window cut them into a
    placement-dependent set of fragments, and the wider ring masked it. That is fixed at the
    extraction now (`test_a_percolating_level_set_is_periodic`), and forcing `_tile_rings` to
    1 leaves all fourteen scenes of the translation sweep clean. It stays for the
    covering-radius guarantee alone, which is about a cell far flatter than any measured here.
    """
    from scipy.spatial import cKDTree

    from wavebit.rcwa._engine import normal_methods as nm

    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    scenes = {
        'disc (closed loop)': np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 30 ** 2, 12.25, 1.0),
        'stripe (open curve)': np.where(np.abs(U - 32) < 22, 12.25, 1.0),
    }
    t = np.deg2rad(float(gamma))
    lattice = None if gamma == 90 else np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)]))
    M = nm._metric(lattice, (n, n))
    sigma = nm._CONTOUR_SIGMA_COEF * float(np.sqrt(np.sqrt(n * n))) * nm._pixel_size(M)
    Ii, Jj = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    P = np.column_stack([Ii.ravel(), Jj.ravel()]) @ M.T

    for name, base in scenes.items():
        for shift in ((0, 0), (2, 2), (17, 3), (1, 63), (7, 11)):
            cell = np.roll(np.roll(base, shift[0], axis=0), shift[1], axis=1)
            smoothed = nm._isotropic_blur(nm._two_phase_indicator(cell), sigma, M)
            pts = nm._assemble(nm._contour_polylines(smoothed),
                               nm._CONTOUR_RESAMPLE_H)[0] @ M.T

            def nearest(k):
                tiles = [pts + M @ np.array([dx * n, dy * n], dtype=float)
                         for dy in range(-k, k + 1) for dx in range(-k, k + 1)]
                return cKDTree(np.vstack(tiles)).query(P, workers=-1)[0]

            missed = int((nearest(1) - nearest(5) > 1e-9).sum())
            assert missed == 0, (
                f'gamma={gamma}, {name}, roll {shift}: one ring missed the nearest point on '
                f'{missed} of {P.shape[0]} pixels')


def test_half_fill_contour_translation_is_invariant():
    """An asymmetric half-filled blob must not choose a field from its corner.

    No translation of this blob exchanges it with its complement. Before
    contour traversal was canonicalized, rolls that changed the corner phase
    changed transmission by about 2e-4. Keep the original 1e-9 solve-level gate.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    blob = (V >= 8) & (V < 56) & (U >= 8) & (U < 50)
    blob = blob | ((V >= 8) & (V < 40) & (U == 50))
    assert blob.sum() * 2 == blob.size, 'the scene must sit exactly on the tie'
    assert not (blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any()), (
        'the blob must not touch a cell edge, or open curves confound this')

    cell = np.where(blob, 12.25, 1.0)
    cfg = dict(backend='numpy', thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               raster_normal_method='contour', period=[1.0, 1.0])
    got = [float(np.sum(np.asarray(_legacy_wavebit.rcwa(
        structure=np.roll(np.roll(cell, a, axis=0), b, axis=1)[None], **cfg).solve().T)))
        for a in range(0, n, 16) for b in range(0, n, 16)]
    assert max(got) - min(got) < 1e-9, f'sum(T) moved by {max(got) - min(got):.2e}'


def _percolating_stripe(U, V, half=22):
    """A stripe spanning the cell: its level set has no closed 0.5 crossing at all."""
    return np.where(np.abs(U - 32) < half, 12.25, 1.0)


def _post_and_arm(U, V):
    """A column percolating in y with an arm on it -- the level set is open and complicated."""
    return np.where(((U < 40) & (V < 26)) | ((U < 18) & (V >= 26)), 12.25, 1.0)


@pytest.mark.parametrize('name,build,gamma', [
    ('stripe, 45 degrees', _percolating_stripe, 45),
    ('stripe, 20 degrees', _percolating_stripe, 20),
    ('post and arm, square cell', _post_and_arm, 90),
])
def test_a_percolating_level_set_is_periodic(name, build, gamma):
    """A structure that crosses the cell is closed *on the torus*, and the code now sees that.

    `_contour_polylines` used to pad the raster 3x3 with `mode='wrap'` and hand contourpy a
    **finite rectangle**. A level set that runs off the cell and comes back is a closed curve
    on the torus, but in that rectangle it is cut at the border and came back as open curves
    whose endpoints were cut marks, not curve ends. Closed loops were given a fold onto the
    cell and a dedup, so exactly one whole representative survived whatever the placement;
    open curves got neither, and where the cut fell moved with the structure, so the fragment
    count did too: 5 / 6 / 9 / 12 across rolls of the post-and-arm scene.

    ====================  ==========  =============
    scene                 worst       rolls broken
    ====================  ==========  =============
    simple stripe, 90 deg 1.3e-15     0 of 63
    simple stripe, 45 deg 5.3e-05     62 of 63
    simple stripe, 20 deg 1.4e-05     56 of 63
    post and arm, 90 deg  2.8e-03     56 of 63
    post and arm, 45 deg  1.1e-03     63 of 63
    ====================  ==========  =============

    All four sit at 2e-15 now, which is where the other three constructions are.

    The 90-degree stripe is the control and it is the reason this went unseen: a stripe
    fragments the same way at every placement, so the commonest percolating scene is clean.
    Add an arm, or skew the cell, and it does not.

    **Fixed 2026-09-01, and the fix is the window rather than the fragments.** contourpy has
    no periodic domain, so the answer is to hand it a window whose cut has an exact partner:
    one period with the seam duplicated. Every torus cell of the raster then appears exactly
    once, the level set leaving through column `n_u` re-enters at column `0` at a coordinate
    interpolated from the same two samples, and `_stitch_across_the_seam` walks the cycle back
    together. What comes out is closed -- with winding `(0, 0)` if the component merely
    straddles the seam, non-zero if it percolates -- so there is no fragment left to be
    placement-dependent, and no fold or dedup to give it, because there is only ever one copy.

    The matching is by position on the torus and not by "the opposite edge", which is what
    makes it work on a skewed cell: the raster's periodicity is an index shift of a whole cell
    whatever the lattice does with those indices, so the metric never enters the stitch. An
    earlier probe that matched *opposite edges* paired 2 of 4 ends on a square-cell stripe and
    0 of 8 on a 20-degree cell, and it is the window rather than the rule that was wrong.

    Two further changes were needed and neither is about percolation, which is worth stating
    because either one alone leaves this failing:

    * a reassembled curve has no intrinsic starting vertex, so arc-length resampling from one
      end is no longer covariant. `_resample` subdivides each *segment* instead, which has no
      phase. Putting the arc-length spelling back breaks 8 of the 14 scenes swept -- 7.0e-03
      on a circle overlapping its images, 1.2e-03 on a disc merely straddling the seam.
    * a sample set with no phase in it inherits the raster's symmetries, and the medial-axis
      ties then flip on the last bit. `_CONTOUR_TIE_SHEAR` settles them the same way at every
      placement. Without it, this scene moves 1.2e-03 on 56 of 63 rolls.

    Both are pinned by their own scenes -- see
    `test_contour_is_periodic_where_the_medial_axis_is_reachable`.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    cell = build(U, V)
    t = np.deg2rad(float(gamma))
    lattice = None if gamma == 90 else np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)]))
    worst = max(_translation_defect(cell, (a, b), 'contour', lattice)
                for a in range(8) for b in range(8) if (a, b) != (0, 0))
    assert worst < 1e-12, f'{name}: E moved by {worst:.2e} under a translation'


@pytest.mark.parametrize('radius,percolates', [(100, False), (140, True)])
def test_a_round_feature_overlapping_its_own_images_is_periodic(radius, percolates):
    """The scene that makes percolation ordinary rather than exotic, and it is a real circle.

    At 20 degrees with `|a1| = |a2| = 640` the lattice rows sit 219 apart, so a circle past
    radius **109** overlaps its own periodic images and its level set percolates. Both radii
    here are drawn the same way and only one of them crosses that line, which is what makes
    this a pair rather than a scene:

    ======  ==================  ==========  =============================
    R       closed / open       `contour`   `distance` / `spectral` / `smooth`
    ======  ==================  ==========  =============================
    100     9 / 0               2.1e-15     ~2e-15
    **140** **0 / 12**          **8.8e-03** ~4e-15
    ======  ==================  ==========  =============================

    The closed/open counts are what the 3x3 window returned; R=140 percolates, so nothing came
    back closed and the open fragments moved with the placement. Both radii sit at 2e-15 now.

    That was the state until 2026-09-01. It matters more than a percentage suggests, because
    the *parametric* path refuses a shape this size outright ("larger than the period and runs
    into its own image"), so the raster is the only route to it -- and `'contour'` is the
    construction the oblique documentation tells you to reach for.

    **The circle is built through `wavebit.meshgrid_xy`, and that is not a detail.** A circle written
    from raster indices -- `(U - 32)**2 + (V - 32)**2 <= r**2` -- is the image of a disc under
    the basis, which at 20 degrees is a **5.67:1 ellipse**; `_index_disc` in this file is
    named for that reason. An ellipse that thin percolates at a quite different radius, so
    drawing it that way would have made both rows of the table say the same thing.
    """
    n = 64
    t = np.deg2rad(20.0)
    lattice = np.column_stack(([640.0, 0.0], [640.0 * np.cos(t), 640.0 * np.sin(t)]))
    X, Y = _legacy_wavebit.meshgrid_xy((n, n), lattice=lattice)
    cell = np.where(X ** 2 + Y ** 2 <= float(radius) ** 2, 12.25, 1.0)

    from wavebit.rcwa._engine.normal_methods import _contour_polylines, _isotropic_blur, _two_phase_indicator, _metric, _pixel_size, _CONTOUR_SIGMA_COEF
    M = _metric(lattice, (n, n))
    sigma = _CONTOUR_SIGMA_COEF * float(np.sqrt(np.sqrt(n * n))) * _pixel_size(M)
    curves = _contour_polylines(_isotropic_blur(_two_phase_indicator(cell), sigma, M))
    winding = sorted(w for _, w in curves)
    assert (winding != [(0, 0)]) == percolates, (
        f'R={radius} should {"" if percolates else "not "}percolate; windings {winding}')

    worst = max(_translation_defect(cell, (a, b), 'contour', lattice)
                for a in range(8) for b in range(8) if (a, b) != (0, 0))
    assert worst < 1e-12, f'R={radius}: E moved by {worst:.2e} under a translation'


def test_a_percolating_level_set_comes_back_closed_on_the_torus():
    """The stitch itself, under the solve -- the statement the periodicity pins rest on.

    Three things, and each fails differently. A percolating stripe is **two** curves, each
    winding once along the axis it spans and neither of them open. A disc drawn across the
    seam is cut into fragments by the window and has to come back as **one** loop of winding
    `(0, 0)` -- the case that says the stitch reassembles rather than merely tolerating a cut.
    And a layer holding both keeps both, which is what the shipped code got wrong by dropping
    every open curve whenever a closed loop survived.

    `winding is None` is the degenerate fallback for a fragment whose partner is missing or
    ambiguous. Nothing here should reach it, and asserting that is the point: a scene that
    silently fell back would pass a periodicity test and lose the reassembly.
    """
    from wavebit.rcwa._engine.normal_methods import _contour_polylines, _isotropic_blur, _two_phase_indicator, _metric, _pixel_size, _CONTOUR_SIGMA_COEF

    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)

    def windings(cell, gamma=90):
        t = np.deg2rad(float(gamma))
        lat = None if gamma == 90 else np.column_stack(([1.0, 0.0], [np.cos(t), np.sin(t)]))
        M = _metric(lat, (n, n))
        sigma = _CONTOUR_SIGMA_COEF * float(np.sqrt(np.sqrt(n * n))) * _pixel_size(M)
        out = _contour_polylines(_isotropic_blur(_two_phase_indicator(cell), sigma, M))
        for pts, w in out:
            assert w is not None, 'a fragment fell back to an open curve'
            assert pts[-1] == pytest.approx(pts[0] + np.array([w[0] * n, w[1] * n]), abs=1e-9)
        return sorted(w for _, w in out)

    assert windings(_percolating_stripe(U, V)) == [(0, -1), (0, 1)]
    assert windings(_percolating_stripe(U, V), gamma=20) == [(0, -1), (0, 1)]
    # the same stripe read the other way round: it percolates along u, not v
    assert windings(np.where(np.abs(V - 32) < 22, 12.25, 1.0)) == [(-1, 0), (1, 0)]

    # a disc sitting on the seam -- cut by the window, and one loop again once stitched
    seam_disc = np.roll(np.where(((U - 32) ** 2 + (V - 32) ** 2) <= 12 ** 2, 12.25, 1.0),
                        (32, 32), axis=(0, 1))
    assert windings(seam_disc) == [(0, 0)]

    # and both at once, which is the layer that used to lose its stripe entirely
    assert windings(np.where((np.abs(V - 12) < 6)
                             | ((U - 32) ** 2 + (V - 42) ** 2 <= 12 ** 2), 12.25, 1.0)) == [
        (-1, 0), (0, 0), (1, 0)]


def test_a_closed_loop_does_not_delete_the_percolating_interface_beside_it():
    """The one defect found here that returned a **wrong** answer rather than an unstable one.

    `_contour_polylines` used to end with `if out: return out` -- if any closed loop survived
    the fold, every open curve was thrown away. A layer holding both is ordinary: a percolating
    stripe and a disc, which is one line of a recipe.

    Measured on exactly that: contourpy found 9 closed curves and 6 open ones in the padded
    array, and the code kept 1 -- the disc's representative. The stripe's boundary was **not
    in the normal field at all**. Every pixel then answered "where is the nearest interface?"
    with the disc, including the pixels sitting on the stripe's own edge.

    The stripe here is horizontal, so its normal must be `(0, +-1)` everywhere along it:

    =====  ==================  ==================
    x      `'contour'` (was)   `'spectral'`
    =====  ==================  ==================
    8      (-0.660, +0.751)    (0.000, +1.000)
    20     (-0.388, +0.922)    (0.000, +1.000)
    50     (+0.554, +0.832)    (0.000, +1.000)
    =====  ==================  ==================

    Up to **45 degrees** wrong, silently, with energy conserved. It was not a translation
    defect -- the field was wrong at every placement, consistently -- which is why the
    periodicity pins in this file could not see it and why it needs its own.

    The discard is gone with the thing that made it look necessary: once the level set is
    extracted through a seam-duplicated single period and stitched
    (`test_a_percolating_level_set_is_periodic`), there are no fragments to prefer a whole
    loop over. Every component comes back once, closed on the torus, and both are kept.

    The vertical stripe used elsewhere here hides this by coincidence: both interfaces are
    then vertical, so the disc's normal happens to point the way the stripe's would.
    """
    n = 64
    ax = np.arange(n)
    U, V = np.meshgrid(ax, ax)
    cell = np.where((np.abs(V - 12) < 6) | ((U - 32) ** 2 + (V - 42) ** 2 <= 12 ** 2),
                    12.25, 1.0)

    from wavebit.rcwa._engine.factorization import _interface_mask
    from wavebit.rcwa._engine.normal_methods import contour_normal_field

    nx, ny = contour_normal_field(cell, _interface_mask(cell))
    along = np.abs(ny[6, 4:60])                       # the stripe's upper edge, away from the disc
    worst = float(np.degrees(np.arccos(np.clip(along.min(), 0.0, 1.0))))
    assert worst < 1.0, (
        f'on a horizontal interface the normal must be (0, +-1); it is off by {worst:.1f} '
        f'degrees, because the stripe was dropped and the nearest kept curve is the disc')
