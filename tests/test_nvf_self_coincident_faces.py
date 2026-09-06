"""A face a shape's own periodic image covers is not an interface, and carries no normal.

The parametric normal field is built from **pieces**: a box is its four faces, each a segment
evaluated at every periodic instance that can be its nearest, and the plan
(:class:`~wavebit._core.parametric.GammaPlan`) records which face lies on which image's face,
opposed, over what run. Along such a run the material continues into the image and there is no
interface -- the two pieces' distances are bitwise one number there and their opposite normals
cancel to an exact zero.

**Two cases, and since 2026-09-06 they are one case.** The image can cover a face end to end,
or it can be slid *along* the face and cover part of it. A face is a segment, so which one
happens is decided by the lattice vector's component along the face:

* **Covered end to end** (``t . s == 0``) -- a rectangle spanning the cell, i.e. a 1-D grating
  written on a 2-D cell. Measured against the exact nearest-interface oracle over the boundary
  band, 0.000 deg -- and the answer is HEAD's to the last digit, because the previous fold was
  exact here too.
* **Covered in part** (``t . s != 0``) -- along the dead run the nearest interface belongs to
  the *neighbouring image*, which a field built from one box alone never looked at, so this
  was refused by name. It is built now, from the image's own face: 0.16 deg against the oracle
  on the sheared cell and 0.18 on the tilted box, where the one-box field read 90 deg on the
  run (1.4 to 17 deg averaged over the band, and 3e-2 in ``sum(T)`` that did not decay).

Three tests here exist because a mutation pass reached their branches and nothing else did:
``test_a_stripe_along_x_is_the_mirrored_case`` (the ``u`` pair, which every grating-shaped test
leaves alone), ``test_a_box_that_tiles_the_plane_keeps_a_unit_field`` (the empty plan, whose
absence gives a zero vector rather than an error), and
``test_the_onset_thresholds_themselves_are_built`` (a run of zero length is a point, not a run).

``test_the_covering_image_can_be_a_diagonal_of_the_ring`` is here for the same reason and came
from a second session working the same defect: on every cell used above, the covering
translation *is* a basis vector, so a search over ``a1`` and ``a2`` alone passes the whole file
and still leaves 46.9% of Li's checkerboard cell wrong. It also found the tie that
``test_the_surviving_pair_does_not_cancel_to_a_zero_vector`` now pins: two opposed survivors at
equal distance sum to the zero vector, until the surviving axis is substituted there.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit
from wavebit._core import overlap as O
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice
from wavebit._core.overlap import primitive

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gamma_instruments import band_error, gamma_points  # noqa: E402

PERIOD = 700.0
EPS_HI = 12.25
GRID = (512, 512)

SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
#: ``a1 = (700, 0)``, ``a2 = (200, 700)``. ``a2 . v = 700`` is the face separation of a box
#: spanning the cell in y, and ``a2 . u = 200`` slides the image along the face -- so the two
#: y-faces coincide over ``lx - 200`` and only for ``lx > 200``.
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])

#: A box turned by ``atan(1/2)`` whose ``ly`` is exactly ``700 sin(theta)``: then ``t = (700, 0)``
#: has ``t . v = -2b`` exactly and ``t . u = 700 cos(theta)``, so the same partial coincidence
#: happens on a *square* cell. ``ly`` must be spelled to full precision -- a rounded 313.05
#: misses the equality and nothing fires.
TILT = float(np.arctan2(1.0, 2.0))
TILT_LY = PERIOD * np.sin(TILT)
TILT_ONSET = PERIOD * np.cos(TILT)              # 626.099...

#: The L box: 689 x 200 turned by ``atan2(117.9, 690)``. Its short faces sit one unit off the
#: image's, its long faces slide 117.9 past each other, so it touches nothing and still leaves
#: its Wigner-Seitz cell -- the nearest interface at 0.5% of its boundary band belonged to a
#: neighbouring image, which the one-box field could not see (90 deg there).
L_ANGLE = float(np.arctan2(117.9, 690.0))

#: Li's checkerboard primitive cell. Neither basis vector runs along an unrotated shape's axes,
#: but their **sum** ``a1 + a2 = (2.5, 0)`` does, and their difference ``(0, 2.5)`` does for the
#: other pair -- so the covering image here is reached by a diagonal of the ring, not by a basis
#: vector. A search over ``a1`` and ``a2`` alone misses it and leaves the same defect standing on
#: a cell one shape over.
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])

SOLVE_KW = dict(backend=0, wavelength=900.0, top=1.0, bot=2.25,
                theta=0.2, phi=0.0, psi=0.0, thickness=[300.0])


def _field(lattice, **geom):
    shape = dict(type='rectangle', material=EPS_HI, **geom)
    return P.normal_vector_field_analytical(np, GRID, [shape], np.float64,
                                            as_lattice(lattice=lattice))


def _field_at(lattice, n, **geom):
    shape = dict(type='rectangle', material=EPS_HI, **geom)
    return P.normal_vector_field_analytical(np, (n, n), [shape], np.float64,
                                            as_lattice(lattice=lattice))


def _solve(lattice, truncation_order=8, **geom):
    P._PROJECTOR_CACHE.clear()
    layer = [dict(background=1.0, shapes=[dict(type='rectangle', material=EPS_HI, **geom)])]
    return wavebit.rcwa(lattice=lattice, structure=layer,
                        truncation_order=dict(u=truncation_order, v=truncation_order),
                        normal_grid=512, **SOLVE_KW).solve()


# ------------------------------------------------------- covered end to end: built, and exact

def test_a_rectangle_spanning_the_cell_has_no_normal_across_the_seam():
    """``ly = Py`` is one unbroken ridge in y, so the field is ``(+-1, 0)`` everywhere.

    This is the simplest scene the parametric path has -- a 1-D grating on a 2-D cell -- and it
    read ``|n_y| = 1`` over the two corner triangles where a y-face is nearest, 11.5% of the
    cell, until the covered pair was dropped. The check is on the exact zero rather than a
    tolerance: the two y-faces are a *flush* pair in the plan, their distances are one number
    at every sample, and there is no y component left to be small.
    """
    n_x, n_y = _field(SQUARE, cx=0.0, cy=0.0, lx=350.0, ly=PERIOD, angle=0.0)
    assert np.abs(n_y).max() == 0.0
    assert np.abs(np.abs(n_x) - 1.0).max() == 0.0


def test_a_stripe_along_x_is_the_mirrored_case():
    """``lx = Px`` kills the *other* pair. Nothing else in this file reaches that branch.

    Every case built around a grating spans the cell in y, so a fix written for the ``v`` pair
    alone passes all of them. This is the transpose of the test above and fails against it.
    """
    n_x, n_y = _field(SQUARE, cx=0.0, cy=0.0, lx=PERIOD, ly=200.0, angle=0.0)
    assert np.abs(n_x).max() == 0.0
    assert np.abs(np.abs(n_y) - 1.0).max() == 0.0


@pytest.mark.parametrize('geom, dead_axis, live_axis', [
    (dict(lx=2.5, ly=0.8, angle=0.0), 0, 1),      # spans a1 + a2 = (2.5, 0)
    (dict(lx=0.8, ly=2.5, angle=0.0), 1, 0),      # spans a1 - a2 = (0, 2.5)
])
def test_the_covering_image_can_be_a_diagonal_of_the_ring(geom, dead_axis, live_axis):
    """The repeat direction need not be a basis vector, and on a checkerboard cell it is not.

    Every other scene in this file sits on a cell where ``a1`` or ``a2`` is itself the covering
    translation, so a search over the two basis vectors passes all of them -- and then reports
    ``max|n_x| = 1.000`` over 46.9% of Li's checkerboard cell, on a structure that repeats end
    to end along x and has no vertical face at all. The sweep here is the whole ring of images
    the plan states, which is also the ring the oracle samples, so the two cannot disagree
    about which images exist.
    """
    field = _field(CHECKER, cx=0.0, cy=0.0, **geom)
    assert np.abs(field[dead_axis]).max() == 0.0
    assert np.abs(np.abs(field[live_axis]) - 1.0).max() == 0.0


@pytest.mark.parametrize('lx', [1.0, 2.0])
def test_a_checkerboard_shape_short_of_the_repeat_keeps_its_faces(lx):
    """The control for the test above: below the repeat every face is still an interface."""
    n_x, n_y = _field(CHECKER, cx=0.0, cy=0.0, lx=lx, ly=0.8, angle=0.0)
    assert np.abs(n_x).max() > 0.9 and np.abs(n_y).max() > 0.9


def test_a_box_that_tiles_the_plane_keeps_a_unit_field():
    """Both pairs covered leaves the blend nothing to sum -- and a zero vector is not a normal.

    Such a layer is uniform, so no projector acts on an interface and the value is free; what
    is *not* free is that it be finite and of unit length, since ``n n^T`` goes into the
    factorization either way. An empty plan answers the constant ``(1, 0)``; without that arm
    four cancelling faces leave ``(0, 0)``, which no assertion elsewhere in the suite notices.
    """
    n_x, n_y = _field(SQUARE, cx=0.0, cy=0.0, lx=PERIOD, ly=PERIOD, angle=0.0)
    assert np.isfinite(n_x).all() and np.isfinite(n_y).all()
    assert np.abs(np.hypot(n_x, n_y) - 1.0).max() < 1e-12
    assert P.plan_gamma([dict(type='rectangle', cx=0.0, cy=0.0, lx=PERIOD, ly=PERIOD,
                              material=EPS_HI)], as_lattice(lattice=SQUARE)).empty


def test_the_grating_solves_as_the_same_grating_under_lis_rule():
    """Two spellings of one structure, one answer.

    A rectangle with ``ly = Py`` *is* a lamellar grating, and the exact normal field of one is
    the constant ``(1, 0)`` -- under which the NVF projector reduces to Li's 1-D rule outright.
    So the parametric path and a raster of the same grating factorized by Li must agree to
    machine precision, and this is the check that says the dead pair left the *right* field
    behind rather than merely a smaller ``n_y``. Before the drop the two differed by 5e-4 to
    9e-3, which reads as an ordinary truncation residual and is not one.

    ``lx = 350`` on a 4096-sample raster lands on a pixel boundary exactly, so the two
    structures really are the same one; a width that did not would compare a staircase.
    """
    x = (np.arange(4096) + 0.5) / 4096 * PERIOD
    row = np.where(np.abs(x - PERIOD / 2.0) <= 175.0, EPS_HI, 1.0)
    raster = np.tile(row, (8, 1))[None, :, :]

    for order in (6, 10):
        shaped = _solve(SQUARE, truncation_order=order, cx=0.0, cy=0.0, lx=350.0, ly=PERIOD)
        li = wavebit.rcwa(period=[PERIOD, PERIOD], structure=raster, factorization='li',
                          truncation_order=dict(x=order, y=order), **SOLVE_KW).solve()
        assert abs(float(shaped.T.sum()) - float(li.T.sum())) < 1e-11


# --------------------------------------------------- covered in part: built, from the image's face

@pytest.mark.parametrize('name, lattice, geom, run', [
    ('shear, lx just past the onset', SHEAR, dict(lx=300.0, ly=PERIOD, angle=0.0), 100.0),
    ('shear, half the cell', SHEAR, dict(lx=500.0, ly=PERIOD, angle=0.0), 300.0),
    ('tilted on a square cell', SQUARE, dict(lx=640.0, ly=TILT_LY, angle=TILT), 13.9),
    ('tilted, a longer run', SQUARE, dict(lx=690.0, ly=TILT_LY, angle=TILT), 63.9),
])
def test_a_partially_covered_face_is_built_from_the_neighbours_face(name, lattice, geom, run):
    """The run is recorded by the plan, the layer builds, and the field is the oracle's.

    Until 2026-09-06 this raised ``runs into its own periodic image``: no field over one box
    reaches the dead run, because the nearest interface there is the neighbouring image's face.
    The piece field carries every face over its images, so that face is simply one of the
    candidates, and the run itself reads no normal at all. Judged over a band of 1.5 pixels
    around the exact interface set: under 0.3 deg mean and under 0.2% of samples more than
    45 deg out, where the one-box field read 90 deg along the whole run.
    """
    A = as_lattice(lattice=lattice)
    prim = primitive('rectangle', 0.0, 0.0, geom['lx'], geom['ly'], geom['angle'])
    plan = P.plan_gamma([dict(type='rectangle', material=EPS_HI, cx=0.0, cy=0.0, **geom)], A)
    contacts = O.face_pairs([prim], A, plan.ring, O.COLLINEAR_TOL_REL * PERIOD)
    assert contacts and abs(contacts[0].hi - contacts[0].lo - run) < 0.05
    assert plan.pairs and not any(pr.flush for pr in plan.pairs)

    res = _solve(lattice, cx=0.0, cy=0.0, **geom)
    assert 0.0 < float(res.T.sum()) < 1.0

    shape = dict(type='rectangle', material=EPS_HI, cx=0.0, cy=0.0, **geom)
    n_x, n_y = _field(lattice, cx=0.0, cy=0.0, **geom)
    mean, over, count = band_error(n_x, n_y, GRID, lattice, gamma_points([shape], lattice))
    assert count > 1000
    assert mean < 0.3, (name, mean)
    assert over < 0.002, (name, over)


def test_the_l_box_reads_its_neighbours_face_without_touching_it():
    """A box that leaves its Wigner-Seitz cell takes its normal from the nearest instance.

    No pair here: the L box touches nothing. Yet 0.5% of its boundary band is nearer a
    neighbouring image's long face than any face of the in-cell box, and the one-box field
    read 90 deg there (0.65 deg averaged over the band, 0.64% of it past 45 deg). Every face
    is evaluated over its ring of images now: 0.19 deg, 0.05%.
    """
    geom = dict(cx=0.0, cy=0.0, lx=689.0, ly=200.0, angle=L_ANGLE)
    shape = dict(type='rectangle', material=EPS_HI, **geom)
    plan = P.plan_gamma([shape], as_lattice(lattice=SQUARE))
    assert not plan.pairs and plan.ring == 3
    n_x, n_y = _field(SQUARE, **geom)
    mean, over, count = band_error(n_x, n_y, GRID, SQUARE, gamma_points([shape], SQUARE))
    assert count > 1000 and mean < 0.3 and over < 0.002, (mean, over)


@pytest.mark.parametrize('name, lattice, geom', [
    # `t . s == 0`: covered end to end, which is the case with an exact answer.
    ('grating spanning the cell', SQUARE, dict(lx=350.0, ly=PERIOD, angle=0.0)),
    ('stripe spanning the cell', SQUARE, dict(lx=PERIOD, ly=200.0, angle=0.0)),
    # Exactly at the onset the run has zero length, so the faces meet along a line of measure
    # zero and there is nothing to record. An over-eager sweep records a pair here and fails
    # nowhere else.
    ('shear, exactly at the onset', SHEAR, dict(lx=200.0, ly=PERIOD, angle=0.0)),
    ('tilted, exactly at the onset', SQUARE, dict(lx=TILT_ONSET, ly=TILT_LY, angle=TILT)),
    ('shear, short of the onset', SHEAR, dict(lx=150.0, ly=PERIOD, angle=0.0)),
    # A strictly convex boundary meets its translate at isolated points only, so a round type
    # cannot self-coincide however large it is -- and must not be tested as though it could.
    ('a disc filling the cell', SQUARE, dict(lx=PERIOD, ly=PERIOD, angle=0.0)),
])
def test_the_onset_thresholds_themselves_are_built(name, lattice, geom):
    if name.startswith('a disc'):
        layer = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0,
                                                   lx=geom['lx'], material=EPS_HI)])]
        res = wavebit.rcwa(lattice=lattice, structure=layer, normal_grid=512,
                           truncation_order=dict(u=6, v=6), **SOLVE_KW).solve()
    else:
        res = _solve(lattice, truncation_order=6, cx=0.0, cy=0.0, **geom)
        plan = P.plan_gamma([dict(type='rectangle', material=EPS_HI, cx=0.0, cy=0.0, **geom)],
                            as_lattice(lattice=lattice))
        assert bool(plan.pairs) is name.endswith('spanning the cell'), name
    assert 0.0 <= float(res.T.sum()) <= 1.0


# ------------------------------------------------------------------------ inert where it should be

@pytest.mark.parametrize('geom', [
    dict(lx=350.0, ly=350.0, angle=0.0),
    dict(lx=200.0, ly=150.0, angle=0.0),
    dict(lx=300.0, ly=200.0, angle=np.deg2rad(25.0)),
    dict(lx=600.0, ly=650.0, angle=np.deg2rad(10.0)),
])
def test_a_shape_that_meets_no_image_keeps_both_face_pairs(geom):
    """A shape meeting no image of its own keeps every face, and the merge is behavioural.

    Asserted on the field rather than on a predicate, so it survives a change of mechanism --
    which is exactly what happened to the first version of this test, written against a helper
    that no longer exists, and again to the second.
    """
    n_x, n_y = _field(SQUARE, cx=0.0, cy=0.0, **geom)
    assert np.abs(n_x).max() > 0.9 and np.abs(n_y).max() > 0.9
    assert np.abs(np.hypot(n_x, n_y) - 1.0).max() < 1e-12


def test_the_drop_survives_jit_and_grad():
    """No Python branch on a shape parameter, so a traced solve takes the same road.

    The plan is taken eagerly from the frozen record's pass and the replay is array ops on
    the traced values -- no ``float()`` of a tracer anywhere on the way, and a jitted solve of
    a grating is the most ordinary thing an optimizer does.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    wavebit.enable_x64()

    def total(lx):
        layer = [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0,
                                                   lx=lx, ly=PERIOD, material=EPS_HI)])]
        solver = wavebit.rcwa(backend=1, period=[PERIOD, PERIOD], structure=layer,
                              truncation_order=dict(x=4, y=4), normal_grid=128,
                              **{k: v for k, v in SOLVE_KW.items() if k != 'backend'})
        return jnp.sum(solver.solve(geometry=solver.freeze_geometry()).T)

    value = float(jax.jit(total)(jnp.asarray(350.0)))
    assert 0.0 < value < 1.0
    g = float(jax.grad(total)(jnp.asarray(350.0)))
    assert np.isfinite(g)


# --------------------------------------------------------- the tie the surviving pair cannot break

def test_the_surviving_pair_does_not_cancel_to_a_zero_vector():
    """Two exactly opposed survivors cancel where equidistant, and the answer is their axis.

    A dead pair cancels everywhere, and on the surviving pair's own midline the two survivors
    cancel too: the sum is ``(0, 0)``, which is not a normal. The substitution puts the axis of
    the pair that did *not* cancel there -- read off the plan's pair records by array equality
    of the members' distances -- which is what the geometry says: sign is free because the
    factorization reads ``n n^T``.

    The cell matters more than the shape. A centred shape on an axis-aligned cell never reaches
    the tie at all, because ``(i + 0.5)/N`` cannot land on ``N/2``; the checkerboard's sample
    diagonal lands on it once per row, so the zero set was one line however fine the grid --
    0.78% / 0.39% / 0.20% at N = 128 / 256 / 512, not shrinking. That is why the assertion
    sweeps N: a single resolution cannot tell a line from a rounding artefact.

    Those points read ``(+-1, 0)`` *before* the pair was dropped -- along the one axis with no
    interface on it -- so the sequence here is wrong, then null, then right.
    """
    for n in (128, 256, 512):
        n_x, n_y = _field_at(CHECKER, n, cx=0.0, cy=0.0, lx=2.5, ly=0.8, angle=0.0)
        mag = np.hypot(n_x, n_y)
        assert int((mag == 0.0).sum()) == 0
        assert np.abs(mag - 1.0).max() < 1e-15
        assert np.abs(n_x).max() == 0.0          # the dead axis stays dead

    # the mirrored pair, and a cell that never samples the tie -- both must be untouched
    n_x, n_y = _field_at(CHECKER, 256, cx=0.0, cy=0.0, lx=0.8, ly=2.5, angle=0.0)
    assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0 and np.abs(n_y).max() == 0.0
    for n in (128, 512):
        n_x, n_y = _field_at(SQUARE, n, cx=0.0, cy=0.0, lx=350.0, ly=PERIOD, angle=0.0)
        assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0


def test_a_run_strip_tie_reads_the_surviving_axis():
    """On the sheared cell the two v runs overlap over a strip, and the tie there is ``u``.

    Both v faces are dead over ``x in [-50, 50]`` of the box's own frame (each run covers it),
    so on the box's midline ``x = 0`` inside that strip every weight difference is exactly
    zero. The u faces did not cancel -- they are not a pair -- so the substitution reads
    *their* axis, whichever way the box is turned. The whole scene is rotated with its cell so
    that the answer is ``(cos, sin)`` of the turn and not the ``(1, 0)`` a lazy fallback
    would give; the sample points are placed on the midline by construction, since a grid
    lands on it only by luck of rounding.
    """
    turn = 0.3
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    lattice = as_lattice(lattice=rot @ SHEAR)
    shape = dict(type='rectangle', material=EPS_HI, cx=0.0, cy=0.0, lx=500.0, ly=PERIOD,
                 angle=turn)
    plan = P.plan_gamma([shape], lattice)
    assert plan.pairs and not any(pr.flush for pr in plan.pairs)

    # points on the shape's own midline (u = 0), spanning the strip in v, in the cell frame
    v = np.linspace(-300.0, 300.0, 61)
    centre = 0.5 * (np.asarray(lattice[:, 0]) + np.asarray(lattice[:, 1]))
    X = centre[0] + v * (-np.sin(turn))
    Y = centre[1] + v * np.cos(turn)
    n_x, n_y = P._piece_field(np, X, Y, [shape], plan, None, lattice, np.float64)
    assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0
    along_u = np.abs(n_x * np.cos(turn) + n_y * np.sin(turn))
    assert np.abs(along_u - 1.0).max() < 1e-15

    # the control: off the strip the v faces are live and the midline reads +-u for a
    # different reason (they are farther than the u faces), so the tie needs the strip
    off = np.linspace(-345.0, 345.0, 7)
    X2 = centre[0] + 200.0 * np.cos(turn) + off * (-np.sin(turn))
    Y2 = centre[1] + 200.0 * np.sin(turn) + off * np.cos(turn)
    n_x2, n_y2 = P._piece_field(np, X2, Y2, [shape], plan, None, lattice, np.float64)
    assert np.abs(n_x2 * np.cos(turn) + n_y2 * np.sin(turn)).min() < 0.9
