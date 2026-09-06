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
import importlib
import pathlib
import sys

import numpy as np
import pytest

import wavebit
from wavebit._core import domain as D
from wavebit._core import overlap as O
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gamma_instruments import band_error, gamma_points, mutated  # noqa: E402
from test_gamma_layers import (_abc_band_gate, _abc_column_exact,  # noqa: E402
                               _k_column_above_the_seam_end)

PERIOD = 700.0
EPS_HI = 12.25
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])
TILT = float(np.arctan2(1.0, 2.0))
TILT_LY = PERIOD * np.sin(TILT)
L_ANGLE = float(np.arctan2(117.9, 690.0))

#: The operating point of ``test_nvf_self_coincident_faces.py``: eps 12.25 on 1, substrate 2.25.
SOLVE_KW = dict(backend=0, wavelength=900.0, top=1.0, bot=2.25,
                theta=0.2, phi=0.0, psi=0.0, thickness=[300.0])


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=EPS_HI), **geom)


def _total(lattice, shape, order=6, normal_grid=512, **incidence):
    P._PROJECTOR_CACHE.clear()
    kw = dict(SOLVE_KW, **incidence)
    res = wavebit.rcwa(lattice=lattice, structure=[dict(background=1.0, shapes=[shape])],
                       truncation_order=dict(u=order, v=order), normal_grid=normal_grid,
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
    s = wavebit.rcwa(backend=0, thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                     truncation_order=dict(x=5, y=5), top=1, bot=1, wavelength=550.0, psi=0.0)
    s.structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                     lx=350.0, ly=PERIOD, angle=0.0,
                                                     material=4.0)])]
    assert abs(float(np.sum(s.solve().given.T)) - 0.9763112427108831) < 1e-13


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
    """``ly = P - gap``: no cliff anywhere, the threshold included, on both cells.

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
    at_span = _total(SQUARE, _box(lx=350.0, ly=PERIOD), order=5, normal_grid=256)
    A = as_lattice(lattice=SQUARE)
    previous = None
    for gap in RUNGS[:-1]:
        shape = _box(lx=350.0, ly=PERIOD - gap)
        assert bool(P.plan_gamma([shape], A).pairs) is (gap <= 7.0e-10), gap
        value = abs(_total(SQUARE, shape, order=5, normal_grid=256) - at_span)
        if previous is not None:
            assert value <= previous, f'square: not approaching the span at gap {gap}'
        previous = value
    assert previous < 1e-10

    A = as_lattice(lattice=SHEAR)
    tol = 1e-12 * float(np.hypot(*SHEAR[:, 1]))
    rungs = (1e-1, 1e-2, 7.1e-4, 7.01e-4, 6.99e-4, 6.9e-4, 1e-4, 1e-6, 1e-9,
             tol * (1.0 + 1.0 / 70.0), tol * (1.0 - 1.0 / 70.0))
    spanning = _box(lx=500.0, ly=PERIOD)
    at_span = _total(SHEAR, spanning, order=5, normal_grid=256)
    previous_t = previous_c = None
    for gap in rungs:
        shape = _box(lx=500.0, ly=PERIOD - gap)
        assert bool(P.plan_gamma([shape], A).pairs) is (gap <= tol), gap
        distance = _coefficient_distance(SHEAR, shape, spanning)
        if previous_c is not None:
            assert distance <= previous_c, f'shear: the field is not approaching at gap {gap}'
        previous_c = distance
        value = abs(_total(SHEAR, shape, order=5, normal_grid=256) - at_span)
        if gap <= 7.1e-4 and previous_t is not None:
            assert value <= previous_t, f'shear: not approaching the span at gap {gap}'
        previous_t = value
    assert previous_t < 1e-10 and previous_c < 5e-12


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
    base = _total(lattice, shape, order=5, normal_grid=256, phi=0.3)
    same = _total(rot @ lattice, turned, order=5, normal_grid=256, phi=0.3 + alpha)
    assert abs(same - base) < 5e-12, (name, same - base)
    left_behind = _total(rot @ lattice, turned, order=5, normal_grid=256, phi=0.3)
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
    base = _total(lattice, shape, order=5, normal_grid=256, phi=0.3)
    same = _total(rot @ lattice, dict(shape, angle=turn), order=5, normal_grid=256,
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
        s = wavebit.rcwa(backend=0, thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                         truncation_order=dict(x=tox, y=toy), top=1, bot=1, wavelength=550.0,
                         psi=np.pi / 2)
        s.structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                         lx=PERIOD, ly=350.0, angle=0.0,
                                                         material=4.0)])]
        return float(np.sum(s.solve().given.T))

    for n in (3, 5, 7):
        assert abs(total(0, n) - total(n, n)) < 1e-12


# ------------------------------------------------------------------ the golden that moved

def test_the_350_box_golden_and_what_it_moved_against():
    """A lone 350 x 350 box on a 700 square, order 6: 0.8918880 -> 0.8922841 (+4.0e-4).

    Every lone box golden moved by this class of amount when the corner-radial extension gave
    way to fixed face normals. Off the interface a normal field is a numerical device, so
    neither value is "the answer"; what can be measured is where each sits against a
    converged reference of the same geometry that shares no code with either -- a raster of
    the box with its edges on pixel boundaries, factorized by NVF with the ``'contour'`` field:

        order 6   parametric, fold (HEAD)  0.891888043
                  parametric, pieces       0.892284142
                  raster N=700  nvf        0.891754321
                  raster N=1400 nvf        0.891758246
                  raster N=1400 li         0.891257562
        order 9   parametric, pieces       0.895966448   raster N=1400 nvf  0.895618806
        order 12  parametric, pieces       0.892740124   raster N=1400 nvf  0.892973722

    The pieces sit 5.3e-4 above the reference at order 6, 3.5e-4 above at 9 and 2.3e-4 below
    at 12 -- decaying -- where the fold sat 1.3e-4 / 6.8e-5 / 1.2e-4 from it. The fold tracked
    the reference more closely because the ``'contour'`` construction is itself distance-based
    and extends off the interface the way the corner-radial rule did; that is a resemblance of
    extensions, not a verdict. The field itself is judged in
    ``test_nvf_self_coincident_faces.py``'s oracle band: 0.237 deg mean against 0.358 for the
    fold, with 0.00% against 0.39% past 45 deg -- the corner quadrants read the diagonal of
    their two faces now rather than the radial direction from the corner, which is what the
    change is, and what C3 chose over the foot rule that keeps HEAD's corners: that rule
    breaks the 1-D grating (``max|n_y|`` 0.76) and steps 4e-4 at every merge.
    """
    got = _total(SQUARE, _box(lx=350.0, ly=350.0), order=6, normal_grid=512)
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
    _, gx, gy = P._round_field(np, X, Y, L_ELLIPSE, A, np.float64)
    assert np.abs(n_x * gx + n_y * gy).min() > 1.0 - 1e-12


def _the_350_golden():
    assert _total(SQUARE, _box(lx=350.0, ly=350.0)) == pytest.approx(0.892284142298975, abs=1e-9)


def _rotated_seam_reads_its_own_axis():
    # K turned by 0.3 rad with its cell: the seam normal is (cos, sin), which the tie's (1, 0)
    # fallback -- where a sign forced opposed sends every sample -- is 17 deg away from.
    turn = 0.3
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    bars = [dict(_box(lx=350.0, ly=PERIOD, angle=turn, material=4.0),
                 cx=-175.0 * np.cos(turn), cy=-175.0 * np.sin(turn)),
            dict(_box(lx=350.0, ly=PERIOD, angle=turn, material=9.0),
                 cx=175.0 * np.cos(turn), cy=175.0 * np.sin(turn))]
    n_x, n_y = P.normal_vector_field_analytical(np, (128, 128), bars, np.float64,
                                                as_lattice(lattice=rot @ SQUARE))
    assert np.abs(np.abs(n_x * np.cos(turn) + n_y * np.sin(turn)) - 1.0).max() < 1e-15


def _unreduced_basis_has_its_pair():
    # The sheared cell written with an unreduced basis: the abutting image sits at (-2, 1).
    A = as_lattice(lattice=np.array([[PERIOD, 1450.0], [0.0, PERIOD]]))
    plan = P.plan_gamma([_box(lx=500.0, ly=PERIOD)], A)
    assert {(pr.m, pr.n) for pr in plan.pairs} == {(-2, 1)}


#: What a fresh interpreter runs for the one mutant that cannot be reloaded in place
#: (``domain.py`` -- see :func:`gamma_instruments.run_isolated`): a box naming ``mu = 1.0``
#: beside one naming none, in the permeability view ``_mu_structure`` would build. Their
#: slots are both 1.0, so the seam cancels and the field equals the spelled-out one; read
#: absence as anything else and the seam co-orients instead.
MU_SPELLINGS = """
import numpy as np
from wavebit import Material
from wavebit._core import parametric as P
from wavebit._core.domain import as_material
from wavebit._core.lattice import as_lattice
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
"""


def _mu_absence_is_one_in_a_fresh_interpreter():
    from gamma_instruments import run_isolated
    passed, stderr = run_isolated(MU_SPELLINGS)
    assert passed, stderr


#: The four grouping lines a face's direction goes through, and the foot rule in their place:
#: beyond a face's tangential extent the direction becomes the vector from the nearest
#: endpoint to the sample (at the instance on the box's own fold), the corner-radial rule the
#: fold's box branch used. Inside the extent it is the fixed normal, as before.
GROUPED_SUM = """        wu = own_w[index[(i, '+u')]] - own_w[index[(i, '-u')]]
        wv = own_w[index[(i, '+v')]] - own_w[index[(i, '-v')]]
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
            sx = sx + own_w[index[(i, face)]] * fx / nrm
            sy = sy + own_w[index[(i, face)]] * fy / nrm"""

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
    ('the seam sign forced opposed', P,
     'signs[r] = xp.where(equal, as_array(xp, -1.0, dtype=dtype), as_array(xp, 1.0, dtype=dtype))',
     'signs[r] = as_array(xp, -1.0, dtype=dtype)',
     _rotated_seam_reads_its_own_axis),
    ('fixed normals replaced by the foot direction', P,
     GROUPED_SUM, FOOT_SUM,
     _grating_exact_off_centre),
    ('the pair sweep reads a fixed 3x3 instead of the ring', O,
     '    ring = images(r)\n',
     '    ring = images(1)\n',
     _unreduced_basis_has_its_pair),
    # C4, the per-run window and the per-pair sign. Four mutants that passed the 26 tests of
    # this file's neighbours: every cross pair the suite had was flush with equal extents and
    # no face carried two runs, so the window was the whole face everywhere they looked.
    ('the sign window always open', P,
     'hits[r] = (sigma >= lo) & (sigma <= hi)',
     'hits[r] = (sigma == sigma)',
     _k_column_above_the_seam_end),
    ('the sign keyed on the q piece', P,
     'signs[r] = xp.where(equal, as_array(xp, -1.0, dtype=dtype), as_array(xp, 1.0, dtype=dtype))',
     'signs[r] = next((signs[k] for k, pk in enumerate(plan.pairs[:r]) if pk.q == pr.q and k in '
     'signs), xp.where(equal, as_array(xp, -1.0, dtype=dtype), as_array(xp, 1.0, dtype=dtype)))',
     _abc_column_exact),
    ('the sign keyed on the p piece', P,
     'signs[r] = xp.where(equal, as_array(xp, -1.0, dtype=dtype), as_array(xp, 1.0, dtype=dtype))',
     'signs[r] = next((signs[k] for k, pk in enumerate(plan.pairs[:r]) if pk.p == pr.p and k in '
     'signs), xp.where(equal, as_array(xp, -1.0, dtype=dtype), as_array(xp, 1.0, dtype=dtype)))',
     _abc_band_gate),
    ("the run window built from the partner's half-length", P,
     'tx, ty, h_p = -pframe.sin, pframe.cos, pframe.b        # the u faces run along v',
     'tx, ty, h_p = -pframe.sin, pframe.cos, qframe.b        # the u faces run along v',
     _abc_band_gate),
    ('the arc ring dropped: every round shape at its own instance', P,
     '    if not leaves:\n        return ((0, 0),)',
     '    if True:\n        return ((0, 0),)',
     _l_ellipse_band_gate),
    ('the ring picks its instance by the algebraic distance', P,
     '        dist = _ellipse_distance(xp, xr, yr, a, b)',
     '        dist = xp.abs((xr / a) ** 2 + (yr / b) ** 2 - 1.0) / _hypot_safe(xp, gxr, gyr)',
     _l_ellipse_centre_reads_its_own_level_set),
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
    ``parametric`` reads that module through its name; one of ``domain`` cannot be, and runs
    its instrument in a fresh interpreter instead.
    """
    path = {P: SOURCE, O: OVERLAP_SOURCE, None: DOMAIN_SOURCE}[module]
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
