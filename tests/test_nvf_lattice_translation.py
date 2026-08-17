"""Sliding the raster by whole pixels must not move the answer.

A unit cell is periodic in both directions -- topologically a torus, not a rectangle -- so
rolling its raster by an integer number of pixels is an exact lattice translation of one
structure, not a different structure. Every efficiency must come back bit-for-bit. Li's rules
do (1e-15 throughout); the NVF path did not, by ~1e-2, at every ``fto``.

**Why it does not converge away, which is what makes it a defect rather than a residual.** The
lift to the ``4*fto+1`` floor is a block repeat -- the same staircase spelled with more pixels
(``docs/nvf-upsampling.md``) -- so raising ``fto`` does not refine the normal field. It only
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
mask. ``'contour'`` is still broken, by a different mechanism, and is pinned as a strict xfail
below rather than quietly omitted.

The sibling defect in the *analytic* (non-raster) path is older and pinned elsewhere:
``test_shape_position_gradients.test_a_multishape_layer_is_not_periodic_across_the_cell_edge``.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import normal_vector_field

SOLVE_KW = dict(backend=0, n_top=1.0, n_bot=1.0, thickness=[120.0], wavelength=250.0,
                theta=np.deg2rad(25.0), phi=np.pi / 2, psi=0.7, period=[500.0, 800.0])

#: Sign is free -- the factorization consumes ``n n^T`` only -- so covariance is compared on the
#: three independent products rather than on the components.
def _frame_defect(a, b):
    (ax, ay), (bx, by) = a, b
    return float(np.maximum.reduce([np.abs(ax * ax - bx * bx),
                                    np.abs(ay * ay - by * by),
                                    np.abs(ax * ay - bx * by)]).max())


def _field_defect(eps, shift, axis, method):
    """Roll the raster, take the field, roll it back, compare against the unrolled field."""
    ref = normal_vector_field(eps, method=method)
    got = normal_vector_field(np.roll(eps, shift, axis=axis), method=method)
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

#: ``'contour'`` is excluded and pinned separately below.
COVARIANT_METHODS = ['distance', 'spectral', 'smooth']


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


@pytest.mark.xfail(strict=True, reason=(
    "'contour' has a second, unrelated periodicity defect: _contour_polylines pads by "
    "4*sigma+3 px, which on a 16 px cell is far narrower than a full period, so the level set "
    "is extracted from a window that cuts the periodic copies at a FIXED place while the roll "
    "moves the shape past it. The set of cut fragments therefore changes with the roll (171 / "
    "187 / 224 resampled points at rolls 0 / 4 / 9), and a fragment's arbitrary endpoint can "
    "win a nearest-point query. Widening the pad to a full cell makes it exact off-wrap and "
    "leaves a p99 ~4 deg resampling tie-break on the medial ridge -- the same class already "
    "documented for mirrors. Not fixed here: it is a separate defect in a separate "
    "construction, and it overlaps work in flight on claude/nvf-grayscale-contour."))
def test_contour_is_not_yet_covariant():
    assert _field_defect(_block(), 4, 1, 'contour') < 1e-12


@pytest.mark.parametrize('shift,axis', [(7, 2), (5, 1)])
def test_the_solve_is_covariant_on_a_multi_phase_raster(shift, axis):
    """End to end, on the raster that first showed this -- four phases, both roll axes.

    Four phases matter to the *reading* of this test rather than to the physics: a raster with
    more than two distinct values delegates to ``'distance'`` whatever ``normal_method`` says,
    so all four methods return bit-identical numbers here. That delegation is what hid the
    defect's location originally -- ``'contour'`` and ``'spectral'`` looked equally broken while
    neither was being exercised.
    """
    ucell = np.ones((1, 16, 16))
    ucell[0, 2:7, 3:12] = 3.48
    ucell[0, 6:13, 9:13] = 2.10
    ucell[0, 11:14, 2:6] = 1.60

    def efficiencies(uc):
        solver = wavebit.rcwa(ucell=uc, fto=[5, 3], factorization='nvf', **SOLVE_KW)
        return np.asarray(solver.conv_solve().res.R, dtype=complex)

    moved = np.abs(efficiencies(ucell) - efficiencies(np.roll(ucell, shift, axis=axis))).max()
    assert moved < 1e-12, f'R moved by {moved:.2e} under an exact lattice translation'


def test_the_defect_did_not_converge_with_fto_which_is_why_it_was_a_defect():
    """Covariance holds at every truncation, not merely at a large one.

    Stated as a *sweep* deliberately. The pre-fix error did not decay with ``fto`` -- it sat
    between 3e-3 and 1.7e-2 without a trend -- so a single-``fto`` guard could be passed by an
    implementation that merely made the error small somewhere. Raising ``fto`` changes the lift
    factor, and the lift is a block repeat, so a genuinely non-periodic field stays non-periodic
    however fine the grid is.
    """
    ucell = np.ones((1, 16, 16))
    ucell[0, 2:7, 3:12] = 3.48
    ucell[0, 6:13, 9:13] = 2.10

    rolled = np.roll(ucell, 7, axis=2)
    for fto in ([3, 2], [5, 3], [8, 5]):
        def efficiencies(uc):
            solver = wavebit.rcwa(ucell=uc, fto=fto, factorization='nvf', **SOLVE_KW)
            return np.asarray(solver.conv_solve().res.R, dtype=complex)

        moved = np.abs(efficiencies(ucell) - efficiencies(rolled)).max()
        assert moved < 1e-12, f'fto={fto}: R moved by {moved:.2e}'
