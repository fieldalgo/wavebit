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


@pytest.mark.xfail(strict=True, reason='the two-phase indicator keys the background on the '
                                        'corner pixel, which a roll can move inside the shape')
def test_contour_is_not_periodic_on_a_loop_that_fills_the_cell():
    """A known defect, pinned as one. `xfail`, not deleted, so it fails when it is fixed.

    The pin is a `strict` *marker*, and that is load-bearing. It was an imperative
    `pytest.xfail(...)` call at the top of the body until 2026-08-28, which raises immediately:
    the assertion below never ran, so the test reported XFAIL whether or not the defect was
    still there and could not have told anyone it was fixed -- which is exactly what the line
    above claims it does. Two one-line tests with identical `assert True` bodies show it:
    the imperative call gives XFAIL, `@pytest.mark.xfail(strict=True)` gives
    `FAILED [XPASS(strict)]`.

    Rolling a raster is the same structure written from a different corner, so every answer
    must be identical. Three of the four constructions are, to 4e-14. `normal_method='contour'`
    -- the one `docs/nvf-normal-methods.md` and `README.md` recommend -- moves `sum(T)` by
    3.9e-4 on a square cell and 9.7e-4 on a 2:1 cell, for a disc of radius 30 in a 64-pixel
    cell. It does not converge with truncation order.

    **The cause is `_two_phase_indicator`, and it is not in the contour block at all.** That
    function keys the background on the *corner pixel* -- `eps != eps.flat[0]` -- which is
    invariant under relabeling the two phases, the property its docstring defends, and is not
    invariant under a translation. This disc has radius 30 in a 64 cell, so a roll can put the
    corner *inside* it, and then the indicator inverts wholesale.

    That is measurable directly, and it partitions the rolls exactly:

    ====================================  =============  ================
    roll                                  corner inside  ``sum(T)``
    ====================================  =============  ================
    (0,0) (7,11) (1,63) (2,2) (5,5) (17,3)  no           0.328909134249
    (23,5) (31,17) (40,40) (13,29)          yes          0.329294462911
    ====================================  =============  ================

    Two values, nothing between, each class internally tight to 5e-13. A sample-phase offset
    would give a spread, not two classes. Replacing the corner rule with the majority phase --
    a probe, not a proposed fix, since it ties at exactly half fill -- collapses all ten rolls
    to one value, spread 4.157e-13, the floor the other three constructions sit at. So the
    indicator is the whole of it on this scene.

    **Two earlier hypotheses, both recorded as refused.** The tile ring: `_contour_polylines`
    folds each loop on its area-mean centre, so a loop's bounding box can overshoot the cell by
    a full period (0.81 to 1.00 cells on the three scenes here) while `contour_normal_field`
    lays down exactly one ring -- deriving the ring from that overshoot moves the field by
    **5e-14**. And the resample phase: `_resample` lays its samples from `pts[0]`, wherever
    contourpy entered the loop, which a roll moves. That was the standing explanation until
    2026-08-28 and is why the fix was scoped as a nearest-segment rewrite deleting `_resample`.
    It is the amplifier, not the cause: an inverted indicator is the *same curve* -- 236 points
    either way -- reached in a different traversal, and `_resample`'s phase is what turns that
    into a different field. `'distance'`, `'spectral'` and `'smooth'` read the same indicator
    and do not notice, because an inversion flips their field globally and NVF consumes only
    `n n^T`.

    Two further probes, both refused, because each is the obvious thing to try next:

    * forcing every closed loop counter-clockwise by its shoelace sign does **not** fix it
      (4.543e-4, still exactly two classes). Reversing the point order moves the entry point
      too, so it canonicalizes the wrong thing.
    * the fold's `np.round` half-way tie is real and is **not** what this is.
      `_contour_polylines` folds on `np.round(centre / n)`, and `np.round(0.5)` is 0, so the
      centred disc keeps *two* representatives at (-32,+32) and (-32,-32) where every offset
      roll keeps one. Harmless: roll (0,0) with two and roll (7,11) with one agree to 4.7e-13,
      because the 3x3 tiling regenerates the copies either way.

    Until then `'spectral'` is the honest recommendation wherever a feature is large compared
    to the cell, and on a skewed cell.
    """
    import wavebit

    n = 64
    ax = np.arange(n)
    X, Y = np.meshgrid(ax, ax)
    disc = np.where(((X - 32) ** 2 + (Y - 32) ** 2) <= 30 ** 2, 12.25, 1.0)
    cfg = dict(backend=0, thickness=[0.3], truncation_order=(6, 6), top=1.0, bot=1.0,
               wavelength=0.94, psi=np.pi / 2, theta=0.1, phi=0.2, factorization='nvf',
               period=[1.0, 1.0], normal_method='contour')

    def total(roll):
        cell = np.roll(np.roll(disc, roll[0], axis=0), roll[1], axis=1)
        return float(np.sum(np.asarray(wavebit.rcwa(structure=cell[None], **cfg).solve().T)))

    got = [total(r) for r in ((0, 0), (7, 11), (23, 5))]
    assert max(got) - min(got) < 1e-9, got
