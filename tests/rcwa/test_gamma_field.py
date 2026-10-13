"""The piece field through a solve: the digit it keeps, the threshold it has, and what it moved.

``test_nvf_self_coincident_faces.py`` judges the field against the exact interface oracle;
``test_gamma_plan.py`` judges the plan. This file is the solve-level half: the one number the
new field must reproduce to the last digit (the 1-D grating, where the previous fold was exact
too), the ladder across the only threshold the mechanism has (two lines being one line, at
rounding scale), the invariances that need no reference value, the golden that moved and
what it moved against, and the mutation harness that says which instrument kills which mutant.

**On rotations.** Every rotation here transports the incidence frame with the structure:
``phi -> phi + alpha`` with ``psi`` unchanged is an exact invariance of the solve (measured
1.6e-12 at 17 degrees and 4e-13 at 90 on the previous mechanism, at ``theta = 0.2``), a
structure-only rotation is a different problem (1.8e-3), and at normal incidence the same
transport can be spelled ``psi -> psi - alpha`` instead. A rotation test that forgets the frame
is blind at normal incidence and wrong at oblique.
"""

from wavebit.compat import legacy as _legacy_wavebit

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import importlib
import pathlib
import sys

import numpy as np
import pytest

from wavebit.materials import response as D
from wavebit._common.geometry import overlap as O
from wavebit.rcwa._engine import parametric as P
from wavebit.lattice import as_lattice, reduced_basis

sys.path.insert(0, str(repository_path('tests')))
from tests.fixtures.rcwa.gamma_instruments import band_error, gamma_points, mutated  # noqa: E402
from tests.rcwa.test_gamma_layers import _stacked_off_centre_cross_pair_is_exact  # noqa: E402

PERIOD = 700.0
EPS_HI = 12.25
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])
TILT = float(np.arctan2(1.0, 2.0))
TILT_LY = PERIOD * np.sin(TILT)
L_ANGLE = float(np.arctan2(117.9, 690.0))

#: The operating point of ``test_nvf_self_coincident_faces.py``: eps 12.25 on 1, substrate 2.25.
SOLVE_KW = dict(backend='numpy', wavelength=900.0, top=1.0, bot=2.25,
                theta=0.2, phi=0.0, psi=0.0, thickness=[300.0])


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=EPS_HI), **geom)


def _total(lattice, shape, order=6, shape_nvf_grid_size=512, **incidence):
    P._PROJECTOR_CACHE.clear()
    kw = dict(SOLVE_KW, **incidence)
    res = _legacy_wavebit.rcwa(lattice=lattice, structure=[dict(background=1.0, shapes=[shape])],
                       truncation_order=dict(u=order, v=order), shape_nvf_grid_size=shape_nvf_grid_size,
                       **kw).solve()
    return float(np.sum(np.asarray(res.T)))


# ---------------------------------------------------------------------- the digit it keeps

def test_the_grating_keeps_the_previous_mechanisms_digits():
    """``ly = P``: the fold of 2026-09-03 and the piece field are both exact, so same answer.

    0.9763112427108831 is the number the fold gave at ``truncation_order (5, 5)`` on the scene
    ``test_every_backend_spans_the_same_way`` pins to 1e-9. Pinned here to 1e-13 instead,
    because for this scene the two mechanisms build the *same* field -- ``(+-1, 0)`` at every
    sample -- and a difference in the last digits would mean the field is not that.
    """
    P._PROJECTOR_CACHE.clear()
    s = _legacy_wavebit.rcwa(backend='numpy', thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                     truncation_order=dict(x=5, y=5), top=1, bot=1, wavelength=550.0, psi=0.0)
    s.structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                     lx=350.0, ly=PERIOD, angle=0.0,
                                                     material=4.0)])]
    assert abs(float(np.sum(s.solve().incident.T)) - 0.9763112427108831) < 1e-13


# ------------------------------------------------------------------- the threshold it has

#: The owner's rungs, descending, with the two that straddle the one threshold the mechanism
#: has: 1e-12 of the cell is 7e-10 here, so 7.1e-10 is two lines and 6.9e-10 is one.
RUNGS = (1e-1, 1e-2, 7.1e-4, 7.01e-4, 6.99e-4, 6.9e-4, 1e-4, 1e-6, 1e-9, 7.1e-10, 6.9e-10, 0.0)


def _coefficient_distance(lattice, shape, other):
    A = as_lattice(lattice=lattice)
    got = P.normal_projector_coeffs(np, [shape], (256, 256), A, 5, 5, np.complex128)
    want = P.normal_projector_coeffs(np, [other], (256, 256), A, 5, 5, np.complex128)
    return max(float(np.abs(g - w).max()) for g, w in zip(got, want))


def test_the_ladder_through_the_pair_threshold_is_monotone():
    """``ly = P - gap``: no step in the value anywhere, the threshold included, on both cells.

    The value only. Its derivative diverges as the gap closes -- the cusp the next test pins.

    Above the threshold the two faces are two lines ``gap`` apart with no pair record, and
    the blend cancels them everywhere except where the *other* pair's own weight difference is
    smaller than the residual ``4 gap / y`` -- a band of depth about ``(gap D^5 / 2h)^(1/5)``
    along the seam (1.7 units at ``gap = 1e-9`` for this box) that shrinks with the gap; at
    and below it they are one line and cancel identically. The step at the threshold is that
    band's last two units: 1.4e-5 in ``sum(T)`` on the square grating at order 5, 2e-4 on the
    sheared box, whose answer is more sensitive to the projector. Both are rounding-scale
    events (1e-12 of the cell) and both are what a threshold costs; the ladder is what says
    nothing else steps.

    Two cells, two readings. On the square grating ``sum(T)`` itself falls monotonically down
    the owner's twelve rungs. On the sheared cell -- the scene the previous mechanism refused
    below 7e-4 -- the coarse rungs are a real change of geometry whose physics is not
    monotone (2.1e-3 at 0.1, 2.9e-3 at 0.01), so ``sum(T)`` is required monotone only from
    the refusal's rung down, and the projector coefficients' distance from the spanning field
    is required monotone throughout, since that is the quantity the mechanism controls. The
    threshold there is ``1e-12 * |a2| = 7.28e-10``, not 7e-10, and the straddling rungs scale
    with it. The plan is asserted on both sides too, so the threshold sits where its
    docstring says.

    The floor under the last rung is the geometry's, not the arithmetic's: below the threshold
    the pair cancels identically over its run, and beyond the run the two faces are their own
    segments a gap apart -- 7e-10 of a 700 cell, which moves the projector coefficients by
    1.1e-12 at N = 256 (a self pair substitutes per record inside its run window since the
    double-covered face was fixed; the unconditional substitution before it read the partner
    off the primary's line beyond the run too, and sat under 1e-12 for that reason alone).
    """
    at_span = _total(SQUARE, _box(lx=350.0, ly=PERIOD), order=5, shape_nvf_grid_size=256)
    A = as_lattice(lattice=SQUARE)
    previous = None
    for gap in RUNGS[:-1]:
        shape = _box(lx=350.0, ly=PERIOD - gap)
        assert bool(P.plan_gamma([shape], A).pairs) is (gap <= 7.0e-10), gap
        value = abs(_total(SQUARE, shape, order=5, shape_nvf_grid_size=256) - at_span)
        if previous is not None:
            assert value <= previous, f'square: not approaching the span at gap {gap}'
        previous = value
    assert previous < 1e-10

    A = as_lattice(lattice=SHEAR)
    tol = 1e-12 * float(np.hypot(*SHEAR[:, 1]))
    rungs = (1e-1, 1e-2, 7.1e-4, 7.01e-4, 6.99e-4, 6.9e-4, 1e-4, 1e-6, 1e-9,
             tol * (1.0 + 1.0 / 70.0), tol * (1.0 - 1.0 / 70.0))
    spanning = _box(lx=500.0, ly=PERIOD)
    at_span = _total(SHEAR, spanning, order=5, shape_nvf_grid_size=256)
    previous_t = previous_c = None
    for gap in rungs:
        shape = _box(lx=500.0, ly=PERIOD - gap)
        assert bool(P.plan_gamma([shape], A).pairs) is (gap <= tol), gap
        distance = _coefficient_distance(SHEAR, shape, spanning)
        if previous_c is not None:
            assert distance <= previous_c, f'shear: the field is not approaching at gap {gap}'
        previous_c = distance
        value = abs(_total(SHEAR, shape, order=5, shape_nvf_grid_size=256) - at_span)
        if gap <= 7.1e-4 and previous_t is not None:
            assert value <= previous_t, f'shear: not approaching the span at gap {gap}'
        previous_t = value
    assert previous_t < 1e-10 and previous_c < 5e-12


def test_closing_gap_has_large_slopes_on_the_documented_grid():
    """Preserve the observed defect ladder without claiming an asymptotic theorem.

    At order (5, 5), grid 256, central secants with h=gap/10 grow from
    0.010405 to 0.73117 to 50.640 at gaps 1e-2, 1e-4, 1e-6. The value
    approaches contact while the sampled slope is large. These finite-grid
    observations do not establish an unbounded physical derivative, a unique
    power law, or that a tuned near-pair ramp is the only possible remedy.
    Grid refinement changes the small-gap ladder; see nvf-contact-followup.md.
    The assertions below still detect the previously recorded solver behavior.
    """
    at_span = _total(SQUARE, _box(lx=350.0, ly=PERIOD), order=5, shape_nvf_grid_size=256)
    slopes = []
    for gap in (1e-2, 1e-4, 1e-6):
        h = gap / 10.0
        value = _total(SQUARE, _box(lx=350.0, ly=PERIOD - gap), order=5, shape_nvf_grid_size=256)
        assert value < at_span                                   # approaching from below
        assert abs(value - at_span) / gap ** 0.2 == pytest.approx(2.0e-3, rel=0.35)
        slopes.append((_total(SQUARE, _box(lx=350.0, ly=PERIOD - gap + h), order=5, shape_nvf_grid_size=256)
                       - _total(SQUARE, _box(lx=350.0, ly=PERIOD - gap - h), order=5,
                                shape_nvf_grid_size=256)) / (2.0 * h))
    assert slopes[0] == pytest.approx(1.0405e-2, rel=1e-2)
    assert slopes[1] / slopes[0] > 30.0 and slopes[2] / slopes[1] > 30.0, slopes
    assert slopes[2] > 40.0


# --------------------------------------------------------------- invariances, frame carried

@pytest.mark.parametrize('name, lattice, shape', [
    ('B shear', SHEAR, _box(lx=500.0, ly=PERIOD)),
    ('B tilted', SQUARE, _box(lx=640.0, ly=TILT_LY, angle=TILT)),
    ('L box', SQUARE, _box(lx=689.0, ly=200.0, angle=L_ANGLE)),
    ('600x650 at 10 deg', SQUARE, _box(lx=600.0, ly=650.0, angle=np.deg2rad(10.0))),
])
@pytest.mark.parametrize('alpha', [np.deg2rad(17.0), np.pi / 2])
def test_a_rigid_rotation_with_the_incidence_frame_is_an_invariance(name, lattice, shape, alpha):
    """Lattice, shape and azimuth turned together: the same answer to 1e-12.

    Nothing in the plan or the replay names an axis -- pieces live in the shape's frame,
    images are lattice vectors, the tie reads the shape's own axes -- so a rotated scene
    plans identically and replays to the same coefficients up to the rotation of the retained
    order set, which the transported ``phi`` undoes. The control asserts that leaving ``phi``
    behind is *not* an invariance, so the test can fail.
    """
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    turned = dict(shape, angle=shape['angle'] + alpha)
    base = _total(lattice, shape, order=5, shape_nvf_grid_size=256, phi=0.3)
    same = _total(rot @ lattice, turned, order=5, shape_nvf_grid_size=256, phi=0.3 + alpha)
    assert abs(same - base) < 5e-12, (name, same - base)
    left_behind = _total(rot @ lattice, turned, order=5, shape_nvf_grid_size=256, phi=0.3)
    assert abs(left_behind - base) > 1e-5, (name, left_behind - base)


#: A face covered from BOTH ends by two different images, its middle left live: on
#: ``a1 = (700, 0)``, ``a2 = (400, 700)`` a 600 x 700 box has its +v face under the (0, 1) image
#: over [100, 300] and under the (-1, 1) image over [-300, 0], and [0, 100] is an interface. The
#: hex cell is the ordinary way to reach it: a grating written as a box on ``a2 = (350, 606.2)``
#: is covered by (0, 1) over [50, 300] and by (-1, 1) over [-300, -50].
SHEAR2 = np.array([[PERIOD, 400.0], [0.0, PERIOD]])
HEX = np.array([[PERIOD, PERIOD / 2.0], [0.0, PERIOD * np.sqrt(3.0) / 2.0]])
DOUBLE_COVERED = {
    'two images from both ends': (SHEAR2, _box(lx=600.0, ly=PERIOD)),
    'hex-cell grating as a box': (HEX, _box(lx=600.0, ly=PERIOD * np.sqrt(3.0) / 2.0)),
}


def _both_dead_runs_read_the_live_axis(lattice, shape, turn):
    """On each covered run of the +v face, on the line and up to 0.1 off it: no v component."""
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    A = as_lattice(lattice=rot @ lattice)
    turned = dict(shape, angle=turn)
    plan = P.plan_gamma([turned], A)
    assert len(plan.pairs) == 2 and not any(pr.flush for pr in plan.pairs)
    prim = P._primitives([turned], A)[0]
    runs = [(c.lo, c.hi) for c in O.face_pairs([prim], A, plan.ring, O.COLLINEAR_TOL_REL * PERIOD)
            if c.f == 2 and c.j == 0]
    assert len(runs) == 2
    centre = 0.5 * (np.asarray(A[:, 0]) + np.asarray(A[:, 1]))
    b = turned['ly'] / 2.0
    for lo, hi in runs:
        u = np.linspace(lo + 1e-3, hi - 1e-3, 29)
        for off in (0.0, 1e-4, 1e-2, 1e-1):
            v = np.full_like(u, b + off)
            X = centre[0] + u * np.cos(turn) - v * np.sin(turn)
            Y = centre[1] + u * np.sin(turn) + v * np.cos(turn)
            n_x, n_y = P._piece_field(np, X, Y, [turned], plan, None, A, np.float64)
            assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0
            along_v = np.abs(-n_x * np.sin(turn) + n_y * np.cos(turn))
            assert along_v.max() < 1e-15, ((lo, hi), off, along_v.max())


@pytest.mark.parametrize('name', list(DOUBLE_COVERED))
@pytest.mark.parametrize('turn', [0.3, 1.1])
def test_a_face_covered_from_both_ends_cancels_on_both_runs(name, turn):
    """Two self records on one face: each dead run reads the primary at ITS coincident instance.

    The replay substituted the partner's normal coordinate per ``(face, instance)`` in a loop
    over the records, so the second record overwrote the first for every instance and the
    first run's partner read the primary at the wrong one. The two primaries lie on one line
    mathematically, but their ``at()`` values differ by the lattice vector's own-axis
    component -- 2.8e-14 at 0.3 rad, -5.7e-14 at 1.1, and exactly 0.0 at 0 and at 17 deg,
    which is why the suite's rotations could not see it -- so on the first run the two
    members' distances differed by 5e-14, ``on_gamma`` handed the whole weight to the one
    that read 0.0, and a covered face carried a full normal (``|n . v| = 1.0`` on 11 of 29
    samples on the line at 0.3 rad, 29 of 29 at 1.1, and out to 0.1 off it). Through a solve
    that was a rotation invariance of 1.1e-9 at N = 256 where the un-turned scene reads
    5e-14. Substituted per record inside that record's run window, both runs read the live
    axis exactly at every turn, and the invariance is back to 1e-11 with the frame carried.
    """
    lattice, shape = DOUBLE_COVERED[name]
    _both_dead_runs_read_the_live_axis(lattice, shape, turn)
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    base = _total(lattice, shape, order=5, shape_nvf_grid_size=256, phi=0.3)
    same = _total(rot @ lattice, dict(shape, angle=turn), order=5, shape_nvf_grid_size=256,
                  phi=0.3 + turn)
    assert abs(same - base) < 1e-11, (name, turn, same - base)


def test_x_orders_cannot_change_an_x_invariant_answer():
    """The transpose of ``test_y_orders_cannot_change_a_y_invariant_answer``, ``psi`` swapped.

    A stripe of ``lx = Px`` varies in y alone, so retaining x orders can only add zeros:
    ``(0, n)`` and ``(n, n)`` must agree, at the polarization that puts the field across the
    grooves of *this* orientation. The y-invariant twin lives in the spanning-shape file; a
    field that reads the u pair right and the v pair wrong passes that one and fails this.
    """
    def total(tox, toy):
        P._PROJECTOR_CACHE.clear()
        s = _legacy_wavebit.rcwa(backend='numpy', thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                         truncation_order=dict(x=tox, y=toy), top=1, bot=1, wavelength=550.0,
                         psi=np.pi / 2)
        s.structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                         lx=PERIOD, ly=350.0, angle=0.0,
                                                         material=4.0)])]
        return float(np.sum(s.solve().incident.T))

    for n in (3, 5, 7):
        assert abs(total(0, n) - total(n, n)) < 1e-12


# ------------------------------------------------- the threshold a blended ellipse crosses

#: ``test_nvf_self_coincident_faces.py``'s ellipse: 720 x 200 at 10 deg leaves its
#: Wigner-Seitz cell on the 700 square between ``lx = 709.90`` and ``709.95``.
L_ELLIPSE = dict(type='ellipse', material=EPS_HI, cx=0.0, cy=0.0, ly=200.0,
                 angle=np.deg2rad(10.0))
DISC = dict(type='circle', cx=0.0, cy=300.0, lx=160.0, material=4.0)


def test_a_blended_ellipse_steps_where_its_support_leaves_its_cell():
    """The known limit (case L) beside another shape: ``sum(T)`` steps at the threshold.

    Inside its cell the ellipse's single-image piece reads its nearest *centre*, with the
    algebraic distance ``|f| / |grad f|`` (``parametric._round_field``); past the edge its
    ring reads the nearest boundary with the exact distance. Beside the disc the distance is
    the piece's weight in the blend, so the two sides disagree off the curve and, at order 6
    and ``N = 256``, the step of 0.05 across the threshold reads +3.879e-3 between steps of
    -2.888e-5 and -2.191e-5 (alone the same ellipse steps +4.759e-3). Reading the single
    image's nearest boundary exactly, over the 3x3 around its fold, removes this step and was
    not taken: it biases the gradient over translation and rotation windows of a blended
    ellipse (``docs/validation/rcwa/normal-field-known-limits.md``). A change that removes the step must
    update that record.
    """
    A = as_lattice(lattice=SQUARE)
    assert [len(P.plan_gamma([dict(L_ELLIPSE, lx=lx), DISC], A).pieces[0].imgs)
            for lx in (709.90, 709.95)] == [1, 11]
    totals = [_layer_total([dict(L_ELLIPSE, lx=lx), DISC], order=6, shape_nvf_grid_size=256)
              for lx in (709.85, 709.90, 709.95, 710.00)]
    before, across, after = np.diff(totals)
    assert max(abs(before), abs(after)) < 1e-4 and across > 1e-3, (before, across, after)


# ------------------------------------------------------------------ the golden that moved

def test_the_350_box_golden_and_what_it_moved_against():
    """A lone 350 x 350 box on a 700 square, order 6: 0.8918880 -> 0.8922841 (+4.0e-4).

    Every lone box golden moved by this class of amount when the corner-radial extension gave
    way to fixed face normals, and the justification is the one C3 states, not a judgment of
    either value: a face piece's direction is its outward normal everywhere, so outside a
    corner the two adjacent faces are equidistant and the blend reads their diagonal where
    the fold read the radial direction from the corner. That is what buys the exact grating
    (the foot rule reinforces a covered face and its partner at their shared endpoints,
    ``max|n_y|`` 0.76 where it must be 0), a zero step at every merge, and the sheared and
    tilted boxes -- and what a lone box that touches nothing pays for it, measured at equal
    grid against the exact interface oracle over a 1.5-pixel band:

        N = 256   fold 0.358 deg mean, 0.39% past 45 deg    pieces 0.473 deg, 0.00%
        N = 512   fold 0.179 deg mean, 0.20% past 45 deg    pieces 0.237 deg, 0.00%

    The mean is higher on the pieces at both grids (the diagonal at every corner) and only
    the fraction past 45 deg favours them; the 0.237-against-0.358 that stood here before
    compared the pieces at N = 512 with the fold at N = 256. Off the interface a normal field
    is a numerical device, so neither value is "the answer"; what can be measured is where
    each sits against a converged reference of the same geometry that shares no code with
    either -- a raster of the box with its edges on pixel boundaries, factorized by NVF with
    the ``'contour'`` field, and by Li's rule, which is exact for an axis-aligned box:

        order 6   fold 0.891888043   pieces 0.892284142   raster N=700 nvf   0.891754321
                                                          raster N=1400 nvf  0.891758246
                                                          raster N=1400 li   0.891257562
        order 9   fold 0.895686917   pieces 0.895966448   raster N=700 nvf   0.895609343
                                                          raster N=1400 nvf  0.895618806
                                                          raster N=1400 li   0.892943538
        order 12  fold 0.893094875   pieces 0.892740124   raster N=700 nvf   0.892945272
                                                          raster N=1400 nvf  0.892973722
                                                          raster N=1400 li   0.892629330

    From the N = 1400 ``'contour'`` reference the pieces sit 5.3e-4 / 3.5e-4 / 2.3e-4 at
    orders 6 / 9 / 12 and the fold 1.3e-4 / 6.8e-5 / 1.2e-4: the instrument sits nearer the
    fold at every order, as it does on the rod and the two-shape goldens. Its ``'contour'``
    construction is distance-based and extends off the interface the way the corner-radial
    rule did, so that is a resemblance of extensions; it is stated here as what the
    measurement shows, and the price is accepted for what fixed normals buy.
    """
    got = _total(SQUARE, _box(lx=350.0, ly=350.0), order=6, shape_nvf_grid_size=512)
    assert got == pytest.approx(0.892284142298975, abs=1e-9)
    assert abs(got - 0.8918880433497915) < 5e-4          # the class of movement, as booked


# ------------------------------------------------------------- the blind plan's own bound

def test_the_blind_plan_replays_within_rounding_of_the_planned_field():
    """What ``jax.jit`` without a frozen record gets: every piece, no pairs, and ulps.

    On the grating the two y faces cancel to rounding instead of exactly -- the same values,
    minus the exact zero -- so the field is within 1e-12 of the planned one everywhere. What
    it does not get is the tie: on the checkerboard the planned field reads ``v`` on the
    surviving pair's midline and the blind one reads whatever the rounding left, which is why
    a scene that leans on the tie should freeze first.
    """
    A = as_lattice(lattice=SQUARE)
    shape = _box(lx=350.0, ly=PERIOD)
    planned = P.normal_vector_field_analytical(np, (128, 128), [shape], np.float64, A)
    blind = P.normal_vector_field_analytical(np, (128, 128), [shape], np.float64, A,
                                             plan=P._blind_plan([shape], A))
    assert np.abs(np.abs(blind[0]) - np.abs(planned[0])).max() < 1e-12
    assert np.abs(blind[1]).max() < 1e-12
    assert np.abs(planned[1]).max() == 0.0


# ------------------------------------------------------------------------- the mutants

def _own_axis_residual_at_quarter_turn():
    A = as_lattice(lattice=SQUARE)
    shape = _box(lx=350.0, ly=PERIOD, angle=np.pi / 2)
    n_x, n_y = P.normal_vector_field_analytical(np, (64, 64), [shape], np.float64, A)
    assert np.abs(-n_x * np.sin(np.pi / 2) + n_y * np.cos(np.pi / 2)).max() < 1e-15


def _checkerboard_tie():
    A = as_lattice(lattice=CHECKER)
    n_x, n_y = P.normal_vector_field_analytical(np, (256, 256), [_box(lx=2.5, ly=0.8)],
                                                np.float64, A)
    assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0 and np.abs(n_x).max() == 0.0


def _grating_exact_off_centre():
    # The partner's own arithmetic, `(e - 700) + 350` against `e - 350`, rounds differently
    # for most `e`; a centre on an awkward fraction keeps the samples off the few that agree.
    A = as_lattice(lattice=SQUARE)
    shape = _box(lx=350.0, ly=PERIOD, cy=123.456789)
    n_x, n_y = P.normal_vector_field_analytical(np, (128, 128), [shape], np.float64, A)
    assert np.abs(n_y).max() == 0.0


def _strip_tie_reads_the_turned_axis():
    turn = 0.3
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    A = as_lattice(lattice=rot @ SHEAR)
    shape = _box(lx=500.0, ly=PERIOD, angle=turn)
    plan = P.plan_gamma([shape], A)
    v = np.linspace(-300.0, 300.0, 61)
    centre = 0.5 * (np.asarray(A[:, 0]) + np.asarray(A[:, 1]))
    X, Y = centre[0] - v * np.sin(turn), centre[1] + v * np.cos(turn)
    n_x, n_y = P._piece_field(np, X, Y, [shape], plan, None, A, np.float64)
    assert np.abs(np.abs(n_x * np.cos(turn) + n_y * np.sin(turn)) - 1.0).max() < 1e-15


def _band_gate(lattice, shape):
    n_x, n_y = P.normal_vector_field_analytical(np, (512, 512), [shape], np.float64,
                                                as_lattice(lattice=lattice))
    mean, over, _ = band_error(n_x, n_y, (512, 512), lattice, gamma_points([shape], lattice))
    assert mean < 0.3 and over < 0.002, (mean, over)


def _tiling_bar_is_empty():
    assert P.plan_gamma([_box(lx=2.5, ly=1.25)], as_lattice(lattice=CHECKER)).empty


L_ELLIPSE = dict(type='ellipse', cx=0.0, cy=0.0, lx=720.0, ly=200.0, angle=np.deg2rad(10.0),
                 material=EPS_HI)


def _l_ellipse_band_gate():
    A = as_lattice(lattice=SQUARE)
    n_x, n_y = P.normal_vector_field_analytical(np, (512, 512), [L_ELLIPSE], np.float64, A)
    mean, over, _ = band_error(n_x, n_y, (512, 512), SQUARE, gamma_points([L_ELLIPSE], SQUARE))
    assert mean < 0.3 and over == 0.0, (mean, over)


def _l_ellipse_centre_reads_its_own_level_set():
    # Along the major axis inside the ellipse the nearest boundary is its own, and the piece
    # must read its own level set there; a distance unbounded at the centre hands it to the
    # neighbour instead.
    A = as_lattice(lattice=SQUARE)
    plan = P.plan_gamma([L_ELLIPSE], A)
    centre = 0.5 * (np.asarray(A[:, 0]) + np.asarray(A[:, 1]))
    u = np.linspace(-216.0, 216.0, 37)
    turn = L_ELLIPSE['angle']
    X = centre[0] + u * np.cos(turn) - 3.0 * np.sin(turn)
    Y = centre[1] + u * np.sin(turn) + 3.0 * np.cos(turn)
    n_x, n_y = P._piece_field(np, X, Y, [L_ELLIPSE], plan, None, A, np.float64)
    _, gx, gy = P._round_field(np, X, Y, L_ELLIPSE, A, reduced_basis(A), np.float64)
    assert np.abs(n_x * gx + n_y * gy).min() > 1.0 - 1e-12


def _l_ellipse_beside_a_disc_reads_its_own_boundary():
    # The ring read inside the blend: with a second shape in the layer the ellipse is one piece
    # of several, and the lone-piece shortcut the two instruments above go through is not taken.
    A = as_lattice(lattice=SQUARE)
    shapes = [L_ELLIPSE, dict(type='circle', cx=0.0, cy=300.0, lx=160.0, material=4.0)]
    plan = P.plan_gamma(shapes, A)
    gamma = gamma_points([L_ELLIPSE], SQUARE)[::5]
    n_x, n_y = P._piece_field(np, gamma[:, 0], gamma[:, 1], shapes, plan, None, A, np.float64)
    assert np.abs(n_x * gamma[:, 2] + n_y * gamma[:, 3]).min() > 1.0 - 1e-9


def _the_350_golden():
    assert _total(SQUARE, _box(lx=350.0, ly=350.0)) == pytest.approx(0.892284142298975, abs=1e-9)


def _layer_total(shapes, order=4, shape_nvf_grid_size=128):
    """``sum(T)`` of a multi-shape layer on the square cell, at this file's operating point."""
    P._PROJECTOR_CACHE.clear()
    res = _legacy_wavebit.rcwa(lattice=SQUARE, structure=[dict(background=1.0, shapes=shapes)],
                       truncation_order=dict(u=order, v=order), shape_nvf_grid_size=shape_nvf_grid_size,
                       **SOLVE_KW).solve()
    return float(np.sum(np.asarray(res.T)))


def _k_short(left=4.0, right=9.0):
    """K: two 300 x 300 boxes abutting inside the cell, materials ``left | right``."""
    return [_box(cx=-150.0, lx=300.0, ly=300.0, material=left),
            _box(cx=150.0, lx=300.0, ly=300.0, material=right)]


#: The seam contrast's three factors (``_piece_field``, block 3b), one instrument each. Every
#: number here was measured on 2026-09-11 at order 4, against the mutant named beside it.

def _the_seam_contrast_vanishes_with_the_contrast():
    # The equality limit: T(9 + 1e-7 | 9) - T(9 | 9) reads 2.4e-9 at N = 128, Li's rule
    # 2.3e-9. The contrast enters to the blend's own power, which is what makes the strip the
    # seam claims linear in the contrast; to the first power it is a quarter-power wide and
    # the step reads 2.3e-5 -- the width that sank the 2026-09-09 candidate.
    step = _layer_total(_k_short(9.0 + 1e-7, 9.0)) - _layer_total(_k_short(9.0, 9.0))
    assert abs(step) < 1e-8


def _the_seam_is_oriented_by_the_materials():
    # [A, B] against [B, A] at N = 504, where the first-shape anchor's translation is a whole
    # number of pixels (108) and the two listings sample one field: 6e-15. An orientation read
    # off the listing -- the co-oriented vector along the earlier shape's normal, which is
    # what the record's sign did -- puts them 7.7e-4 apart, at N = 504 and at 512 alike.
    assert abs(_layer_total(_k_short(), shape_nvf_grid_size=504)
               - _layer_total(list(reversed(_k_short())), shape_nvf_grid_size=504)) < 1e-12


def _the_seam_contrast_reads_opposed_pairs_only():
    # Directly above the K seam's end (x = the seam line, y in (150, 300)) the four near faces
    # tie at weight one: the two top faces read (0, 2), the one opposed pair of different owners
    # leaks 2c along x with c = (5/13)**4, and the far faces and images make it
    # c (1 - w_u) / (1 - w_v), 0.0240 at most on this column. Without the max(0, -n_a.n_b)
    # gate the two perpendicular pairs add their (1, +-1) as well and double it, to 0.051.
    A = as_lattice(lattice=SQUARE)
    y = np.linspace(151.0, 299.0, 75)
    n_x, _ = P._piece_field(np, np.full_like(y, PERIOD / 2.0), PERIOD / 2.0 + y, _k_short(),
                            P.plan_gamma(_k_short(), A), None, A, np.float64)
    assert np.abs(n_x).max() < 1.5 * (5.0 / 13.0) ** P._BLEND_EXPONENT


def _an_off_centre_windowed_cross_pair_cancels_bitwise_over_its_run():
    # The windowed twin of `_stacked_off_centre_cross_pair_is_exact`: a 500-tall box listed
    # first, a 300-tall one abutting it 100 higher, both at an awkward offset, so the run is
    # [-50, 250] along the primary's face and the two faces' own roundings of the normal
    # coordinate differ (8.1e-28 against 3.2e-27 in n2). Over the run the partner reads the
    # primary's n2, the two weights are one number, the dead seam cancels exactly and the tie
    # reads the v axis: |n_x| 1.6e-15 at most on the line. A window built from the partner's
    # half-length is [-50, 150] here -- too narrow, since the primary is the taller -- and
    # the 50 unsubstituted samples over (150, 250) read their own rounding, weights 1 and
    # 1/16, and the seam normal: |n_x| = 1.0. On the exact-centre columns every face reads an
    # exact zero and no window can be told from another, which is why the A|B|C gates passed
    # this mutant once the window stopped gating a sign.
    off = 0.123456
    tall = [_box(cx=-150.0 + off, lx=300.0, ly=500.0, material=4.0),
            _box(cx=150.0 + off, lx=300.0, ly=300.0, cy=100.0, material=4.0)]
    A = as_lattice(lattice=SQUARE)
    plan = P.plan_gamma(tall, A)
    assert {(pr.m, pr.n, pr.flush) for pr in plan.pairs} == {(0, 0, False)}
    y = np.linspace(-49.0, 249.0, 150)
    n_x, n_y = P._piece_field(np, np.full_like(y, PERIOD / 2.0 + off), PERIOD / 2.0 + y, tall,
                              plan, None, A, np.float64)
    assert np.abs(n_x).max() < 1e-12 and np.abs(np.abs(n_y) - 1.0).max() < 1e-12


def _the_bar_along_two_a1_plus_a2_has_its_flush_pair():
    # A bar as long as `2 a1 + a2` along it threads between the 3x3 images and meets the (2, 1)
    # one end to end -- the one accepted shape whose contact sits past the 3x3 on a reduced
    # basis (the unreduced spelling that used to stand here plans as its reduction now).
    A = as_lattice(lattice=SQUARE)
    bar = _box(lx=PERIOD * 5.0 ** 0.5, ly=50.0, angle=float(np.arctan2(1.0, 2.0)))
    plan = P.plan_gamma([bar], A)
    assert {(pr.m, pr.n, pr.flush) for pr in plan.pairs} == {(2, 1, True)}


def _unreduced_circle_reads_its_own_level_set():
    # `a2 = 3 a1 + (50, 700)`: the first spelling whose Wigner-Seitz cell reaches past the
    # fold's 3x3 on the basis as given (8.89 deg mean on Gamma there); on the reduced one 2e-7.
    shape = dict(type='circle', cx=0.0, cy=0.0, lx=693.0, material=EPS_HI)
    A = as_lattice(lattice=np.array([[PERIOD, 2150.0], [0.0, PERIOD]]))
    plan = P.plan_gamma([shape], A)
    centre = 0.5 * (np.asarray(A[:, 0]) + np.asarray(A[:, 1]))
    t = (np.arange(4000) + 0.5) / 4000 * 2.0 * np.pi
    X, Y = centre[0] + 346.5 * np.cos(t), centre[1] + 346.5 * np.sin(t)
    n_x, n_y = P._piece_field(np, X, Y, [shape], plan, None, A, np.float64)
    cos = np.clip(np.abs(n_x * np.cos(t) + n_y * np.sin(t)), 0.0, 1.0)
    assert np.degrees(np.arccos(cos)).mean() < 1e-5


def _unreduced_box_plans_as_its_reduction():
    A = as_lattice(lattice=np.array([[PERIOD, 7050.0], [0.0, PERIOD]]))
    plan = P.plan_gamma([_box(lx=500.0, ly=PERIOD)], A)
    assert plan.ring == 3 and len(plan.pieces[0].imgs) == 21
    assert {(pr.m, pr.n) for pr in plan.pairs} == {(0, 1)}


#: What a fresh interpreter runs for the one mutant that cannot be reloaded in place
#: (``materials/response.py`` -- see :func:`gamma_instruments.run_isolated`): a box naming ``mu = 1.0``
#: beside one naming none, in the permeability view ``_mu_structure`` would build. Their
#: slots are both 1.0, so the seam cancels and the field equals the spelled-out one; read
#: absence as anything else and the seam co-orients instead.
MU_SPELLINGS = """
import numpy as np
from wavebit import Material
from wavebit.rcwa._engine import parametric as P
from wavebit.materials.response import as_material
from wavebit.lattice import as_lattice
A = as_lattice(period=[700.0, 700.0])
left = dict(type='rectangle', cx=-150.0, cy=0.0, lx=300.0, ly=300.0, angle=0.0)
right = dict(left, cx=150.0)
def mu_view(shapes):
    return [dict(s, material=as_material(s['material']).mu_value) for s in shapes]
absent = mu_view([dict(left, material=Material(4.0, mu=1.0)), dict(right, material=Material(9.0))])
named = mu_view([dict(left, material=Material(4.0, mu=1.0)), dict(right, material=Material(9.0, mu=1.0))])
got = P.normal_vector_field_analytical(np, (64, 64), absent, np.float64, A)
want = P.normal_vector_field_analytical(np, (64, 64), named, np.float64, A)
assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
# The field cannot arbitrate this alone any more: the seam contrast is continuous in the
# materials, so a permeability of 1 + 1e-9 beside 1.0 reads a contrast of (5e-10) ** 4 and
# moves the field by nothing a float can hold. The slot the field reads is the instrument.
slots = P._material_slots(np, absent, np.float64)
assert complex(slots[1]) == 1.0 and complex(slots[0]) == 1.0, [complex(v) for v in slots]
assert as_material(Material(9.0)).mu_value == 1.0
"""


def _mu_absence_is_one_in_a_fresh_interpreter():
    from tests.fixtures.rcwa.gamma_instruments import run_isolated
    passed, stderr = run_isolated(MU_SPELLINGS)
    assert passed, stderr


#: The four grouping lines a face's direction goes through, and the foot rule in their place:
#: beyond a face's tangential extent the direction becomes the vector from the nearest
#: endpoint to the sample (at the instance on the box's own fold), the corner-radial rule the
#: fold's box branch used. Inside the extent it is the fixed normal, as before.
GROUPED_SUM = """        wu = w[index[(i, '+u')]] - w[index[(i, '-u')]]
        wv = w[index[(i, '+v')]] - w[index[(i, '-v')]]
        sx = sx + (wu * frame.cos - wv * frame.sin)
        sy = sy + (wu * frame.sin + wv * frame.cos)"""
FOOT_SUM = """        du, dv = frame.at(frame.km, frame.kn)
        for face in _overlap.FACES:
            if face[1] == 'u':
                nu = du - frame.a if face == '+u' else du + frame.a
                exc = dv - xp.clip(dv, -frame.b, frame.b)
                fx, fy = nu * frame.cos - exc * frame.sin, nu * frame.sin + exc * frame.cos
            else:
                nu = dv - frame.b if face == '+v' else dv + frame.b
                exc = du - xp.clip(du, -frame.a, frame.a)
                fx, fy = exc * frame.cos - nu * frame.sin, exc * frame.sin + nu * frame.cos
            nrm = _hypot_safe(xp, fx, fy)
            sx = sx + w[index[(i, face)]] * fx / nrm
            sy = sy + w[index[(i, face)]] * fy / nrm"""

SOURCE = pathlib.Path(P.__file__)
OVERLAP_SOURCE = pathlib.Path(O.__file__)
DOMAIN_SOURCE = pathlib.Path(D.__file__)

#: ``(what, module to mutate and reload -- None runs the instrument in a fresh interpreter --
#: old, new, the instrument that must fail under it)``.
MUTANTS = [
    ('a flush pair reads its own tangential excess', P,
     'e2, n2 = pe2, pn2                   # flush: the same segment, bitwise',
     'n2 = pn2                            # flush: the same segment, bitwise',
     _own_axis_residual_at_quarter_turn),
    ('no self pair substitution at all', P,
     'if q_face != face or offset not in own:',
     'if True:',
     _grating_exact_off_centre),
    ('the self substitution keyed by instance alone, the last record winning', P,
     'n2 = xp.where((psigma >= lo) & (psigma <= hi), pn2, n2)',
     'n2 = pn2',
     lambda: _both_dead_runs_read_the_live_axis(*DOUBLE_COVERED['two images from both ends'],
                                               0.3)),
    ('the tie substitution removed', P,
     'sx = xp.where(tie, axis_x, sx)\n    sy = xp.where(tie, axis_y, sy)',
     'pass',
     _checkerboard_tie),
    ('the tie always answers (1, 0)', P,
     'axis_x = xp.where(pick_u, frame.cos * one, xp.where(pick_v, -frame.sin * one, axis_x))',
     'axis_x = one',
     _strip_tie_reads_the_turned_axis),
    ('one face loses its image list', P,
     'for m, n in piece.imgs:',
     "for m, n in (piece.imgs if face != '+v' else ((0, 0),)):",
     lambda: _band_gate(SQUARE, _box(lx=689.0, ly=200.0, angle=L_ANGLE))),
    ('the v faces lose their images', P,
     'for m, n in piece.imgs:',
     "for m, n in (piece.imgs if face[1] == 'u' else ((0, 0),)):",
     lambda: _band_gate(SQUARE, _box(lx=689.0, ly=200.0, angle=L_ANGLE))),
    ('the u faces lose their images', P,
     'for m, n in piece.imgs:',
     "for m, n in (piece.imgs if face[1] == 'v' else ((0, 0),)):",
     lambda: _band_gate(SQUARE, _box(lx=200.0, ly=689.0, angle=L_ANGLE + np.pi / 2))),
    ('coverage per image instead of the union', P,
     'dead = _overlap.covered_length(runs, tol) >= 2.0 * halves[f] - tol',
     'dead = any(hi - lo >= 2.0 * halves[f] - tol for lo, hi in runs)',
     _tiling_bar_is_empty),
    ('blend exponent 4 -> 2', P,
     '_BLEND_EXPONENT = 4',
     '_BLEND_EXPONENT = 2',
     _the_350_golden),
    ("the winner's weight is d_min^4 (the synthesis's formula)", P,
     'w.append(xp.where(on_gamma, xp.where(on_face, one, zero), ratio ** _BLEND_EXPONENT))',
     'w.append((d_min / xp.where(dk == d_min, one, dk)) ** _BLEND_EXPONENT)',
     _the_350_golden),
    # The seam contrast, 2026-09-11: a mutant per factor of block 3b. These replaced the three
    # mutants of the record's material sign ('forced opposed', 'keyed on the q piece', 'keyed
    # on the p piece'), whose line is gone.
    ('the contrast enters to the first power', P,
     'contrast = xp.astype(contrast, dtype) ** _BLEND_EXPONENT',
     'contrast = xp.astype(contrast, dtype) ** 1',
     _the_seam_contrast_vanishes_with_the_contrast),
    ('the seam oriented by the listing instead of the materials', P,
     'corr = contrast * mean * opposed * phase',
     'corr = contrast * mean * opposed',
     _the_seam_is_oriented_by_the_materials),
    ('the opposed gate removed: every cross pair of pieces adds its difference', P,
     'opposed = xp.where(dot < 0.0, -dot, zero)',
     'opposed = one',
     _the_seam_contrast_reads_opposed_pairs_only),
    ('fixed normals replaced by the foot direction', P,
     GROUPED_SUM, FOOT_SUM,
     _grating_exact_off_centre),
    ('the pair sweep reads a fixed 3x3 instead of the ring', O,
     '    ring = images(r)\n',
     '    ring = images(1)\n',
     _the_bar_along_two_a1_plus_a2_has_its_flush_pair),
    # The reduced basis, 2026-09-06: the fold and the offsets count cells of it. The replay
    # dropping the plan's transform folds on the basis as given, which the circle on the
    # `3 a1` spelling reads at 8.89 deg; the planner dropping it plans that spelling on ring 14.
    ('the replay folds on the basis as given', P,
     '    reduced = reduced_basis(lattice, plan.reduction)\n',
     '    reduced = lattice\n',
     _unreduced_circle_reads_its_own_level_set),
    ('the planner reduces nothing', P,
     '        transform = reduction_transform(lattice)\n        reduced = reduced_basis(lattice, transform)\n',
     '        transform = ((1, 0), (0, 1))\n        reduced = reduced_basis(lattice, transform)\n',
     _unreduced_box_plans_as_its_reduction),
    # C4, the per-run window. Of the four mutants that passed the 26 tests of this file's
    # neighbours (every cross pair the suite had was flush with equal extents and no face
    # carried two runs), the half-length window below is what is left; the substitution
    # mutant after it was never one of the four. 'The sign window always open'
    # (`hits[r] = one`) is dropped, not moved: since 2026-09-11 the sum reads no sign from a
    # record, and the transfer that `hits` gated took `moved` from both members of a record,
    # which are opposed on one line -- neutral in the vector sum, so no instrument at the value
    # could see it (the K contact step, the equality step and the listing gap all moved by
    # less than 1e-14 under it; measured), and on 2026-09-12 the transfer itself was removed. The run window still decides the substitution, and its
    # mutant moved from the A|B|C band gate (3.71 deg when the window gated a sign; 0.3371 now,
    # the unmutated field's own figure, a survivor) to the off-centre windowed pair, where a
    # narrowed window leaves rounding uncancelled on the line.
    ("the run window built from the partner's half-length", P,
     'tx, ty, h_p = -pframe.sin, pframe.cos, pframe.b        # the u faces run along v',
     'tx, ty, h_p = -pframe.sin, pframe.cos, qframe.b        # the u faces run along v',
     _an_off_centre_windowed_cross_pair_cancels_bitwise_over_its_run),
    ('the cross-pair substitution removed, flush and windowed', P,
     'if pr.flush:\n                        e2, n2 = pe2, pn2\n                    else:\n'
     '                        n2 = xp.where(on_segment, pn2, n2)',
     'if pr.flush:\n                        pass\n                    else:\n'
     '                        pass',
     _stacked_off_centre_cross_pair_is_exact),
    ('the arc ring dropped: every round shape at its own instance', P,
     '    if not leaves:\n        return ((0, 0),)',
     '    if True:\n        return ((0, 0),)',
     _l_ellipse_band_gate),
    ('the ring picks its instance by the algebraic distance', P,
     '        dist = _ellipse_distance(xp, xr, yr, a, b)',
     '        dist = xp.abs((xr / a) ** 2 + (yr / b) ** 2 - 1.0) / _hypot_safe(xp, gxr, gyr)',
     _l_ellipse_centre_reads_its_own_level_set),
    ('the blend reads a leaving arc at its own instance only', P,
     '        if len(pc.imgs) == 1:\n            d[k], gx, gy = _round_field(',
     '        if True:\n            d[k], gx, gy = _round_field(',
     _l_ellipse_beside_a_disc_reads_its_own_boundary),
    ('mu = None read as unequal to 1.0', None,
     'return 1.0 if self.mu is None else self.mu',
     'return 1.0 + 1e-9 if self.mu is None else self.mu',
     _mu_absence_is_one_in_a_fresh_interpreter),
]


@pytest.mark.parametrize('what, module, old, new, instrument', MUTANTS, ids=[m[0] for m in MUTANTS])
def test_the_mutant_is_killed(what, module, old, new, instrument):
    """Each mutant names the instrument that kills it, and the harness says if it never ran.

    A ``DID-NOT-APPLY`` (the text is gone) or a ``NO-OP`` (the replacement changes nothing)
    is a failure of this test, not a survivor; a survivor is the instrument passing under the
    mutation, which is the one outcome the harness cannot tell from a passing test on its
    own -- so it is asserted. A mutant of ``overlap`` is reloaded in place, since
    ``parametric`` reads that module through its name; one of ``materials.response`` cannot be, and runs
    its instrument in a fresh interpreter instead.
    """
    path = {P: SOURCE, O: OVERLAP_SOURCE, None: DOMAIN_SOURCE}[module]
    instrument()  # An import/setup failure must not count as a killed mutant.
    try:
        with mutated(path, old, new):
            if module is not None:
                importlib.reload(module)
            try:
                instrument()
            except AssertionError:
                killed = True
            else:
                killed = False
    finally:
        if module is not None:
            importlib.reload(module)
    assert killed, f'SURVIVED: {what}'
