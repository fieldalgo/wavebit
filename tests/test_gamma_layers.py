"""Several shapes in one layer: one blend over every piece, and a seam decided by the materials.

The lone-shape half of the piece field is judged in ``test_nvf_self_coincident_faces.py`` and
``test_gamma_field.py``. This file is the multi-shape half (2026-09-06, the second commit of the
mechanism): two boxes that abut share a face line over a run, the plan records the pair, and
the pair cancels there whatever the materials; a seam between *different* materials is put
back co-oriented in proportion to their contrast (the seam contrast, 2026-09-11 -- it replaced
an equality bit that stepped at contact, at material equality and with the listing order), so
the seam's normal survives. There is no per-shape stage: a two-stage blend switches extension
at the moment of contact (measured as a 1.1e-2 step in ``sum(T)`` at order 6), and a single sum
over every piece has nothing to switch.

The cases carry the design's letters. **D**: two touching same-material bars are the one bar
they make -- the invariants a 1-D structure must satisfy hold to float noise, and the two
spellings of a 2-D structure differ by a residual that decays with order. **K**: an ``eps 4 |
eps 9`` seam is an interface, exact where the structure is 1-D and 0.23 deg against a
material-aware oracle where it is not (it was a null band), and it is one formula at any gap:
opening a contact by 1e-9, sweeping a material through equality and swapping the listing
order each move ``sum(T)`` by rounding, where the equality bit stepped 2.7e-3, 5e-3 and
1.5e-3. **E**: a shape whose material equals the background keeps its faces, a material sweep
never moves a pair-free field, and a touching pair's field is a continuous function of the
pair's contrast -- not of an equality bit. And the seam is decided per quantity: the
permeability pass reads the permeability slots.

Every rotation transports the incidence frame with the structure (``psi`` stated, ``phi``
turned), for the reason ``test_gamma_field.py`` gives.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit
from wavebit import Material
from wavebit._core import overlap as O
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice, sample_grid

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gamma_instruments import band_error  # noqa: E402

PERIOD = 700.0
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
GRID = (512, 512)

#: The operating point of ``test_nvf_self_coincident_faces.py``.
SOLVE_KW = dict(backend=0, wavelength=900.0, top=1.0, bot=2.25,
                theta=0.2, phi=0.0, psi=0.0, thickness=[300.0])


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=4.0), **geom)


#: D. Two spellings of the 350 x 700 grating and two of the 700 x 350 stripe.
ONE_TALL = [_box(lx=350.0, ly=PERIOD)]
TWO_STACKED = [_box(lx=350.0, ly=350.0, cy=-175.0), _box(lx=350.0, ly=350.0, cy=175.0)]
ONE_WIDE = [_box(lx=PERIOD, ly=350.0)]
TWO_SIDE = [_box(lx=350.0, ly=350.0, cx=-175.0), _box(lx=350.0, ly=350.0, cx=175.0)]
#: D in two dimensions: a 350 x 350 box and its two halves.
ONE_SQUARE = [_box(lx=350.0, ly=350.0)]
TWO_HALVES = [_box(lx=175.0, ly=350.0, cx=-87.5), _box(lx=175.0, ly=350.0, cx=87.5)]
#: K. Abutting bars of different material, spanning the cell (a 1-D structure with no
#: background left) and not (a seam of finite length inside the cell).
K_SPANNING = [_box(cx=-175.0, lx=350.0, ly=PERIOD, material=4.0),
              _box(cx=175.0, lx=350.0, ly=PERIOD, material=9.0)]
K_SHORT = [_box(cx=-150.0, lx=300.0, ly=300.0, material=4.0),
           _box(cx=150.0, lx=300.0, ly=300.0, material=9.0)]
D_SHORT = [dict(s, material=4.0) for s in K_SHORT]
ONE_SHORT = [_box(lx=600.0, ly=300.0)]
#: Three materials in a row, two seams of different contrast: ``4 | 9 | 12``.
THREE_TOUCHING = [_box(cx=-200.0, lx=200.0, ly=300.0, material=4.0),
                  _box(cx=0.0, lx=200.0, ly=300.0, material=9.0),
                  _box(cx=200.0, lx=200.0, ly=300.0, material=12.0)]


def _field(shapes, lattice=SQUARE, grid=GRID):
    return P.normal_vector_field_analytical(np, grid, shapes, np.float64,
                                            as_lattice(lattice=lattice))


def _gamma(shapes, lattice=SQUARE, groups=None):
    A = as_lattice(lattice=lattice)
    return O.interfaces_of(P._primitives(shapes, A), A, P.plan_gamma(shapes, A).ring,
                           groups=groups)


def _grating_total(shapes, tox, toy, psi):
    """The 550 nm, eps 4 on 1 scene of ``test_y_orders_cannot_change_a_y_invariant_answer``."""
    P._PROJECTOR_CACHE.clear()
    s = wavebit.rcwa(backend=0, thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                     truncation_order=dict(x=tox, y=toy), top=1, bot=1, wavelength=550.0,
                     psi=psi, normal_grid=256)
    s.structure = [dict(background=1.0, shapes=shapes)]
    return float(np.sum(s.solve().given.T))


def _total(shapes, order=6, lattice=SQUARE, normal_grid=512, **incidence):
    P._PROJECTOR_CACHE.clear()
    res = wavebit.rcwa(lattice=lattice, structure=[dict(background=1.0, shapes=shapes)],
                       truncation_order=dict(u=order, v=order), normal_grid=normal_grid,
                       **dict(SOLVE_KW, **incidence)).solve()
    return float(np.sum(np.asarray(res.T)))


# ------------------------------------------------------------------------------------- D

def test_the_plan_pairs_the_two_bars_across_their_seam_and_across_the_cell_edge():
    """Stacked: ``+v`` of the lower against ``-v`` of the upper at ``(0, 0)``, flush, and the
    lower's ``-v`` against the upper's ``+v`` one cell down. No self pair: each bar is 350 of a
    700 repeat. Recorded once each, from the lower shape."""
    plan = P.plan_gamma(TWO_STACKED, as_lattice(lattice=SQUARE))
    pairs = {(plan.pieces[pr.p].shape, plan.pieces[pr.p].face, plan.pieces[pr.q].shape,
              plan.pieces[pr.q].face, pr.m, pr.n, pr.flush) for pr in plan.pairs}
    assert pairs == {(0, '+v', 1, '-v', 0, 0, True), (0, '-v', 1, '+v', 0, -1, True)}
    assert not any(pc.dead for pc in plan.pieces) and not plan.empty


def test_two_touching_same_material_bars_solve_as_the_one_bar_they_are():
    """The invariants of a 1-D structure, on its two-bar spelling, to float noise.

    Stacked in y the pair is a 350 x 700 grating: orders along y can only carry zero, so
    ``(n, 0)`` and ``(n, n)`` agree at ``psi = 0``; side by side in x it is the stripe, and
    ``(0, n)`` and ``(n, n)`` agree at ``psi = pi / 2``. Both hold because every face that
    survives the two cross pairs points along one axis and the pairs cancel *exactly* -- the
    partner's distance is the primary's own expression at the coincident absolute image, so the
    two weights are one number and the box's own axis sum is an exact zero. The two-bar answer
    also equals the one-bar answer, where the only difference is the permittivity transform
    written as two sincs instead of one.

    Measured: 4e-15 to 2.4e-14 on every row at orders 3, 5, 7, against a one-bar floor of
    4e-15 -- the same float floor with more terms in it. The bound is where a lost pair would
    land (the field then carries a y-normal on 11% of the cell, 5e-4 to 9e-3 here).
    """
    for n in (3, 5, 7):
        stacked = _grating_total(TWO_STACKED, n, n, psi=0.0)
        assert abs(_grating_total(TWO_STACKED, n, 0, psi=0.0) - stacked) < 5e-14, n
        assert abs(_grating_total(ONE_TALL, n, n, psi=0.0) - stacked) < 5e-14, n
        side = _grating_total(TWO_SIDE, n, n, psi=np.pi / 2)
        assert abs(_grating_total(TWO_SIDE, 0, n, psi=np.pi / 2) - side) < 5e-14, n
        assert abs(_grating_total(ONE_WIDE, n, n, psi=np.pi / 2) - side) < 5e-14, n


def test_the_two_spellings_of_a_two_dimensional_structure_converge_together():
    """One 350 x 350 box against its two halves: a residual, and it decays with order.

    On the seam the two halves cancel exactly and on every face both spellings read the same
    normal (the band error against the oracle is identical, see the control below), so the
    difference is in the *extension* off the interface: the ratio weights are length-blind,
    and a cut face weighs twice near the cut. That is an admissible continuation replaced by
    another, which is the class of difference that has to vanish as the order rises and has
    no business plateauing. Measured, ``sum(T)`` one box minus two halves:

        order   4         6         8         10        12
                -1.09e-3  -5.11e-4  +4.73e-5  +1.51e-4  -8.05e-5

    Not monotone past order 8 -- the residual changes sign -- but an order of magnitude down
    from where it starts, which is what "decays" can mean under this laptop's order cap.
    """
    gap = {n: abs(_grating_total(ONE_SQUARE, n, n, 0.0) - _grating_total(TWO_HALVES, n, n, 0.0))
           for n in (4, 6, 8, 10, 12)}
    assert gap[4] > 5e-4                                   # there is a residual to decay
    assert gap[6] < gap[4]
    assert max(gap[8], gap[10], gap[12]) < 0.2 * gap[4]


def test_the_same_material_control_reads_the_one_box_field_on_gamma():
    """Two 300 x 300 boxes of one material abutting inside the cell, against the 600 x 300 box.

    On the interface set both spellings are the same field: the oracle drops the seam (a step
    across it lands in the other box), and the band error of the two-box field against it is
    the one-box field's to seven digits. The control that this instrument can fail: the same
    two boxes with the seam *kept* in the oracle read 90 deg along it.
    """
    gamma = _gamma(D_SHORT)
    two = band_error(*_field(D_SHORT), GRID, SQUARE, gamma)
    one = band_error(*_field(ONE_SHORT), GRID, SQUARE, gamma)
    assert two[0] < 0.3 and two[1] < 0.002, two
    assert abs(two[0] - one[0]) < 1e-5 and two[1] == one[1], (two, one)
    with_seam = band_error(*_field(D_SHORT), GRID, SQUARE, _gamma(D_SHORT, groups=(0, 1)))
    assert with_seam[0] > 5.0, with_seam


# ------------------------------------------------------------------------------------- K

def test_a_seam_between_different_materials_is_an_interface():
    """``eps 4 | eps 9`` bars spanning the cell: the seam's normal survives, and the answer is Li's.

    Every face here is a pair member -- the two seams are cross pairs, the top and bottom faces
    self pairs -- and with the materials different the seam contrast puts the two seams back
    co-oriented after their pairs cancel. The structure is a 1-D grating with no background
    left, so its exact field is the constant ``(+-1, 0)``, under which the NVF projector *is*
    Li's 1-D rule: the parametric solve and a raster of the same profile factorized by
    ``'li'`` must agree to machine precision. They do (1.4e-14 and 1.0e-13 at orders 6 and
    10). The one-box fold read this seam as a null band.
    """
    n_x, n_y = _field(K_SPANNING)
    assert np.abs(n_y).max() == 0.0 and np.abs(np.abs(n_x) - 1.0).max() == 0.0
    assert int((np.hypot(n_x, n_y) == 0.0).sum()) == 0

    x = (np.arange(4096) + 0.5) / 4096 * PERIOD
    raster = np.tile(np.where(x < 350.0, 4.0, 9.0), (8, 1))[None, :, :]
    for order in (6, 10):
        shaped = _total(K_SPANNING, order=order)
        li = wavebit.rcwa(period=[PERIOD, PERIOD], structure=raster, factorization='li',
                          truncation_order=dict(x=order, y=order), **SOLVE_KW).solve()
        assert abs(shaped - float(li.T.sum())) < 1e-11, order


def test_the_seam_normal_turns_with_the_structure():
    """The rotated K scene reads its own seam axis everywhere -- not ``(1, 0)``.

    The unrotated scene cannot tell a co-oriented seam from the tie's fallback, because its
    seam normal *is* ``(1, 0)``; turning the cell and both bars by 0.3 rad separates the two.
    A seam contrast forced to zero leaves every pair cancelled, sends every sample to the tie,
    and the tie, with both of each box's axes cancelled, answers ``(1, 0)``: 17 deg off.
    """
    turn = 0.3
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    turned = [dict(s, angle=turn, cx=s['cx'] * np.cos(turn), cy=s['cx'] * np.sin(turn))
              for s in K_SPANNING]
    n_x, n_y = _field(turned, lattice=rot @ SQUARE, grid=(256, 256))
    assert np.abs(np.abs(n_x * np.cos(turn) + n_y * np.sin(turn)) - 1.0).max() < 1e-15


def test_a_seam_of_finite_length_is_an_interface_over_its_run_and_not_beyond():
    """Two 300 x 300 boxes of different material abutting inside the cell.

    Judged against the oracle made material-aware (``groups``): a step across the seam lands
    in a box of the *other* label, so the seam's samples are kept with the left box's outward
    normal, while its two ends are corners like any other. Over the run the seam is exact --
    both weights are one there and nothing else survives. Beyond the seam's end no interface
    continues, and the two faces no longer cancel exactly there: the seam contrast is
    windowless, so it leaks past the run at strength ``c = (5/13) ** 4`` (the 0.024 that
    ``_k_column_above_the_seam_end`` pins), which the blind oracle (seam dropped) still reads
    worse, having no seam to agree with -- 0.238 deg against the aware 0.230 at N = 512. Under
    the equality bit the two read 0.351 and 0.22: the band against the seam-blind oracle, which
    judges the whole field, improved by more than the leak cost the aware one, and the 0.05
    margin the old gate carried went with the gap it measured.
    """
    n_x, n_y = _field(K_SHORT)
    aware = band_error(n_x, n_y, GRID, SQUARE, _gamma(K_SHORT, groups=(4.0, 9.0)))
    blind = band_error(n_x, n_y, GRID, SQUARE, _gamma(K_SHORT))
    assert aware[2] > 4000 and aware[0] < 0.3 and aware[1] < 0.002, aware
    assert blind[0] > aware[0], (blind, aware)


#: C4. One face on two runs of different equality: B's ``-u`` face (``x = 0``, ``y`` in
#: [-300, 300]) meets same-material A over [10, 300] (dead: the material runs on), background
#: over [-10, 10], and different-material C over [-300, -10] (a seam). Three orderings, because
#: a pair's ``p`` / ``q`` roles and the order of the records follow the shape order, and each of
#: the four mutants of this window in ``test_gamma_field.py`` survives two of the three.
A_BOX = _box(cx=-150.0, cy=160.0, lx=300.0, ly=300.0, material=4.0)
B_BOX = _box(cx=150.0, cy=0.0, lx=300.0, ly=600.0, material=4.0)
C_BOX = _box(cx=-150.0, cy=-160.0, lx=300.0, ly=300.0, material=9.0)
ABC = {'A B C': [A_BOX, B_BOX, C_BOX], 'B A C': [B_BOX, A_BOX, C_BOX],
       'A C B': [A_BOX, C_BOX, B_BOX]}


def _abc_band_gate():
    """Every ordering under 0.35 deg / 0.3% against the material-aware oracle (0.337 / 0.21%).

    The gate is this scene's, not a constant of the mechanism: a contact scene's band carries
    the fixed-normal wedge beyond each run's end, which reads 0.23 deg on ``K_SHORT``, 0.34
    here and 0.32 on two same-material boxes 300 and 400 tall -- identical whether the taller
    one is written upright or turned a quarter turn, so it is the wedge and not a defect. What
    justifies 0.35 is the separation: the sign keyed on a piece reads 5.87 deg / 6.80% and the
    window built from the partner's half-length 3.71 / 4.14%. The seam contrast (2026-09-11)
    moved this scene from 0.265 / 0.11% to 0.337 / 0.21%, identical to 1e-9 across the three
    orderings; the margin to the gate is now 0.013 deg.
    """
    for name, shapes in ABC.items():
        mean, over, count = band_error(*_field(shapes), GRID, SQUARE,
                                       _gamma(shapes, groups=(4.0, 4.0, 9.0) if name != 'A C B'
                                              else (4.0, 9.0, 4.0)))
        assert count > 4000 and mean < 0.35 and over < 0.003, (name, mean, over)


def _abc_column_exact():
    """On B's ``-u`` face line, exactly: ``n_x == 0`` over the dead run, ``|n_x| == 1`` on the seam and in the gap."""
    A = as_lattice(lattice=SQUARE)
    y = np.concatenate([np.linspace(11.0, 299.0, 41), np.linspace(-299.0, -11.0, 41),
                        np.linspace(-9.0, 9.0, 7)])
    for name, shapes in ABC.items():
        n_x, _ = P._piece_field(np, np.full_like(y, PERIOD / 2.0), PERIOD / 2.0 + y, shapes,
                                P.plan_gamma(shapes, A), None, A, np.float64)
        assert np.abs(n_x[:41]).max() == 0.0, name                     # dead: the v axis
        assert np.abs(np.abs(n_x[41:]) - 1.0).max() == 0.0, name       # the seam, and B alone


def _seam_column(shapes, y):
    """The field on ``K_SHORT``'s seam line ``x = 0`` at the centred ``y``, off any grid."""
    A = as_lattice(lattice=SQUARE)
    return P._piece_field(np, np.full_like(y, PERIOD / 2.0), PERIOD / 2.0 + y, shapes,
                          P.plan_gamma(shapes, A), None, A, np.float64)


def _k_column_above_the_seam_end():
    """Directly above the end of the ``K_SHORT`` seam (``x`` = the seam line, ``y`` in (150, 300)).

    The two seam faces are beyond their run there: the record's transfer leaves them cancelled,
    the two top faces carry the field and it is ``(0, +-1)`` -- ``n_x == 0.0`` exactly under
    the equality bit, whose window was the run. The seam contrast has no window: it is one
    formula at any gap and past any run, so the cancelled pair comes back here too, at
    strength ``c = (5/13) ** 4 = 0.022`` times the seam faces' shared weight. Measured over the
    column: ``|n_x|`` 0.0219 to 0.0240, ``|n_y|`` at least 0.9997. The leak is the price of a
    windowless term: a contrast windowed by the *record* would read the record, which exists
    only at contact, and put back the contact step the term exists to remove (a window cut
    from the transfer's off-run tail is the candidate the docs leave unmeasured), and the
    whole-field band against the seam-blind oracle improved with the leak in it (0.351 to
    0.238 deg at N = 512). The old mutant, a sign window always open, read 0.75 here; the gate
    is a fifteenth of that and twice the leak.
    """
    n_x, n_y = _seam_column(K_SHORT, np.linspace(151.0, 299.0, 75))
    assert np.abs(n_x).max() < 0.05 and np.abs(n_y).min() > 0.998, (np.abs(n_x).max(),
                                                                     np.abs(n_y).min())


@pytest.mark.parametrize('name', list(ABC))
def test_one_face_on_two_runs_of_different_equality(name):
    """The sign is a window per record in the traced run bounds, never one scalar per piece.

    B's ``-u`` face is a pair member twice: with A (equal materials, the run [10, 300] cancels
    and the tie reads B's own v axis there) and with C (different, the run [-300, -10] keeps
    the seam's normal), with 20 units of plain interface between them. Both records are on the
    plan whatever the shape order; the field reads 0.337 deg / 0.21% against the
    material-aware oracle in every ordering (0.265 / 0.11% under the equality bit; the seam
    contrast leaks past the C run's ends), and on the face line itself the exact pattern
    ``0 | 1 | 1`` in ``|n_x|`` over dead run, gap and seam. Neither held on any shipped scene
    before: every cross pair in the suite was flush with equal extents and no face carried two
    runs, so a window that was always open, a sign keyed on either piece, and a window built
    from the partner's half-length all passed 26 tests (``test_gamma_field.py`` names them).
    """
    shapes = ABC[name]
    plan = P.plan_gamma(shapes, as_lattice(lattice=SQUARE))
    b = shapes.index(B_BOX)
    runs = {(plan.pieces[pr.p].shape, plan.pieces[pr.p].face, plan.pieces[pr.q].shape,
             plan.pieces[pr.q].face, pr.m, pr.n) for pr in plan.pairs}
    others = {shapes.index(A_BOX): '+u', shapes.index(C_BOX): '+u'}
    for i, face in others.items():
        assert (min(i, b), face if i < b else '-u', max(i, b), '-u' if i < b else face,
                0, 0) in runs, name
    assert not any(pr.flush for pr in plan.pairs), name
    _abc_band_gate()
    _abc_column_exact()


def test_the_sign_window_ends_where_the_run_ends():
    """Beyond the seam's end the pair stays cancelled but for the contrast's leak: the column
    above ``K_SHORT`` reads the top faces' ``(0, +-1)`` to within ``|n_x| < 0.05``."""
    _k_column_above_the_seam_end()


def test_the_run_end_is_continuous_at_every_grid():
    """The old run window made a 5.23e-6 step at N=128, independent of step size.

    Moving the endpoint now changes the field continuously off its line. Check
    that reducing the displacement 100x reduces the transmission difference,
    rather than merely repinning a smaller jump at one chosen displacement.
    Same-material transfer remains an algebraic no-op.
    """
    A = as_lattice(lattice=SQUARE)
    def layer(ly, right=9.0):
        return [_box(cx=-150.0, lx=300.0, ly=300.0, material=4.0),
                _box(cx=150.0, lx=300.0, ly=ly, material=right)]
    for grid in (128, 256, 512):
        _, Y = sample_grid(np, (grid, grid), A, np.float64)
        rows = np.unique(np.round(Y[:, 0] - PERIOD / 2.0, 9))
        ly_star = 2.0 * rows[rows < 150.0].max()
        def difference(h, right=9.0):
            return abs(_total(layer(ly_star+h,right), order=4, normal_grid=grid)
                       - _total(layer(ly_star-h,right), order=4, normal_grid=grid))
        large, small = difference(1e-4), difference(1e-6)
        assert small < .02*large
        assert small < 2e-9
        assert difference(1e-6, 4.) < 1e-9


def test_an_off_centre_flush_cross_pair_cancels_bitwise():
    """The stacked bars shifted to an awkward centre, sampled just off the dead seam's line.

    The partner reads the primary's expression at the coincident absolute image, so the two
    seam faces are one number wherever both are the nearest instance of their piece -- not
    only over the run, since the pair is flush -- and their opposed normals cancel to an
    exact zero. Beyond the bars' ends, at ``y = seam + 0.123456`` for ``x`` in (180, 340),
    the u faces decide the direction and ``n_y`` must be ``0.0`` exactly; with the
    substitution removed the two read their own rounding and ``n_y`` is 2.2e-16 -- which the
    5e-14 invariants of D never see, and only a rotation gate at 1e-11 caught by 3.4e-11.
    ``cy = 12.345678`` keeps the seam off every row the grid could land on.
    """
    _stacked_off_centre_cross_pair_is_exact()


def _stacked_off_centre_cross_pair_is_exact():
    A = as_lattice(lattice=SQUARE)
    off = 12.345678
    bars = [_box(lx=350.0, ly=350.0, cy=-175.0 + off), _box(lx=350.0, ly=350.0, cy=175.0 + off)]
    plan = P.plan_gamma(bars, A)
    assert {(pr.m, pr.n, pr.flush) for pr in plan.pairs} == {(0, 0, True), (0, -1, True)}
    x = np.linspace(180.0, 340.0, 81)
    n_x, n_y = P._piece_field(np, PERIOD / 2.0 + x, np.full_like(x, PERIOD / 2.0 + off + 0.123456),
                              bars, plan, None, A, np.float64)
    assert np.abs(n_y).max() == 0.0 and np.abs(np.abs(n_x) - 1.0).max() == 0.0


def test_touching_different_materials_is_built_and_summed_as_before():
    """The control whose kind must not change: touching is not overlapping, on any material.

    ``K_SHORT`` passes the layer refusals, is not routed to the polygon route (nothing
    overlaps), and builds through the closed form -- the same answer in kind as before the
    seam had a normal, and the seam's presence is what moved the value.
    """
    A = as_lattice(lattice=SQUARE)
    P.check_layer_shapes(dict(background=1.0, shapes=K_SHORT), A)
    assert not P.layer_needs_resolving(dict(background=1.0, shapes=K_SHORT), A)
    assert 0.0 < _total(K_SHORT) < 1.0


def _opened(shapes, i, by):
    """``shapes`` with box ``i`` slid along ``x`` by ``by``: its seam opened to ``|by|``."""
    return [dict(s, cx=s['cx'] + by) if j == i else s for j, s in enumerate(shapes)]


def test_a_different_material_contact_has_no_step():
    """Opening a seam by 1e-9 moves ``sum(T)`` by rounding, on one seam and on two.

    The seam contrast reads no record: two opposed faces a positive gap apart cancel by
    arithmetic and the contrast puts the pair's cancelled part back in proportion to the
    materials, the same on both sides of contact. The equality bit read the record only, so
    the strip between the boxes read the far faces' normal until the gap closed and the answer
    stepped there: -2.7e-3 at order 4 on ``K_SHORT`` (-8.7e-4 to -3.4e-4 at orders 6 to 12,
    against 1.4e-9 for the explicit Li rule) and -5.8e-4 / -2.7e-2 on the two seams of ``4 | 9
    | 12``. Measured now at order 4, N = 128: 2.0e-12 on ``K_SHORT``, -1.7e-12 and 8.3e-12 on
    the three-box seams (1e-12 to 2.2e-12 on ``K_SHORT`` at N = 512 across orders 4 to 12).
    The 1e-9 bound is three decades above rounding and five below the smaller old step; a box
    slid by 1e-9 also moves the structure, and that is inside it.
    """
    base = _total(K_SHORT, order=4, normal_grid=128)
    assert abs(_total(_opened(K_SHORT, 1, 1e-9), order=4, normal_grid=128) - base) < 1e-9
    base = _total(THREE_TOUCHING, order=4, normal_grid=128)
    for i, by in ((0, -1e-9), (2, 1e-9)):
        step = _total(_opened(THREE_TOUCHING, i, by), order=4, normal_grid=128) - base
        assert abs(step) < 1e-9, (i, step)


def test_the_material_equality_limit_is_continuous():
    """``T(9 + delta | 9) - T(9 | 9)`` is Li's ramp, linear in ``delta`` down to 1e-7.

    Measured at order 4, N = 128: 2.36e-9, 2.36e-7 and 2.36e-5 for ``delta`` of 1e-7, 1e-5 and
    1e-3 -- the explicit Li rule on the same profile gives 2.27e-9 at 1e-7, and the ratio of
    the first two is 100 to 4e-6. The equality bit switched the seam's whole normal on the
    moment the two slots differed, whatever ``delta``, and stepped -5.0e-3 at order 4 (-4.8e-4
    to -4.0e-5 at 6 to 12; 9.2e-2 on ``12.25 | 12.25 + 1e-3``); the contrast switches on as
    ``delta ** 4``, so the strip the seam claims shrinks linearly with the material difference
    and the total follows the material. A residual step ``s`` would read as a ratio
    ``(s + 100 a) / (s + a)``, not 100: the second gate is the one a small step cannot pass.
    """
    def total(delta):
        return _total([dict(K_SHORT[0], material=9.0 + delta), dict(K_SHORT[1], material=9.0)],
                      order=4, normal_grid=128)

    base = total(0.0)
    step = {delta: total(delta) - base for delta in (1e-7, 1e-5)}
    assert abs(step[1e-7]) < 1e-8, step
    assert abs(step[1e-5] / step[1e-7] - 100.0) < 1.0, step


def test_the_listing_order_of_a_touching_pair_moves_nothing():
    """``K_SHORT`` and ``K_SHORT[::-1]`` are one structure and solve as one: 6.4e-15 apart.

    The co-oriented vector on a seam is the materials' -- pointed from the lower-``|eps|``
    shape toward the higher -- where the equality bit laid it along the *primary's* normal,
    the earlier-listed shape's, so the same structure listed both ways solved 1.5e-3 apart at
    every grid. The projector anchors a multi-shape layer on its first shape, so the two
    listings sample the structure 300 units apart: N = 504 makes that 216 whole pixels and
    the two totals differ by 6.4e-15 at order 4; at N = 128 (54.9 pixels) they differ by
    1.7e-4, which is the anchor's sampling offset and not the listing's.
    """
    ab = _total(K_SHORT, order=4, normal_grid=504)
    ba = _total(K_SHORT[::-1], order=4, normal_grid=504)
    assert abs(ab - ba) < 1e-12, ab - ba


# ------------------------------------------------------------------------------------- E

def test_a_pair_free_field_reads_no_material_and_a_touching_pair_reads_its_contrast():
    """A pair-free field reads no material; a touching pair's field follows its contrast.

    Dropping a shape's faces when its material equals the background is a cliff in a material
    design variable and puts a material fact into a frozen record (C7). So the field of a
    pair-free layer is bit-identical across any material sweep, including through the
    background's value -- a layer swept as a whole, both shapes one material, is one field.
    A layer with a cross pair reads the pair's contrast ``|eps_a - eps_b| / (|eps_a| +
    |eps_b|)`` and nothing else of the materials: equal materials add an exact zero, so ``4 |
    4``, ``1 | 1`` and ``9 | 9`` are one field; ``4 | 9`` and ``4 | 1`` are two (the equality
    bit made them one); and the field of ``9 | 9 + delta`` tends to the field of ``9 | 9`` as
    ``delta`` does to zero. Measured at N = 128, the largest move of a component: 0.0 at
    ``delta`` 1e-9 and 1e-6 (the contrast's fourth power is below an ulp of every sum), 5.9e-10
    at 1e-3, 5.8e-2 at 1e-1 -- so the gate at 1e-3 has a thousandfold margin and the control at
    1e-1 shows there is a seam to grow. The two listings of one pair are one field, since the
    seam's vector is oriented by the materials and not by the earlier-listed shape: 1.3e-15
    apart at N = 252, the summation order's rounding. (``_field`` samples the shapes where
    they are, so any grid would do here; 252 keeps the listings' 300-unit anchor offset on 108
    whole pixels, the condition the solved total needs in
    ``test_the_listing_order_of_a_touching_pair_moves_nothing``.)
    """
    lone = [_box(lx=300.0, ly=200.0, angle=0.3)]
    apart = [_box(lx=200.0, ly=150.0, cx=-200.0), _box(lx=150.0, ly=200.0, cx=180.0, cy=100.0)]
    for shapes in (lone, apart, K_SHORT):
        fields = [_field([dict(s, material=m) for s in shapes], grid=(128, 128))
                  for m in (4.0, 1.0, 12.25)]
        for got in fields[1:]:
            assert np.array_equal(got[0], fields[0][0]) and np.array_equal(got[1], fields[0][1])

    def seam(left, right, grid=128):
        return _field([dict(K_SHORT[0], material=left), dict(K_SHORT[1], material=right)],
                      grid=(grid, grid))

    def apart_by(one, other):
        return max(np.abs(one[0] - other[0]).max(), np.abs(one[1] - other[1]).max())

    assert apart_by(seam(4.0, 9.0), seam(4.0, 1.0)) > 1.0
    merged = seam(9.0, 9.0)
    assert apart_by(seam(9.0, 9.0 + 1e-9), merged) < 1e-9
    assert apart_by(seam(9.0, 9.0 + 1e-3), merged) < 1e-6
    assert 1e-2 < apart_by(seam(9.0, 9.0 + 1e-1), merged) < 1e-1
    swapped = _field(K_SHORT[::-1], grid=(252, 252))
    assert apart_by(seam(4.0, 9.0, grid=252), swapped) < 1e-12


def test_the_seam_is_decided_per_quantity():
    """Same permittivity, different permeability: no seam in the eps field, a seam in the mu one.

    The permeability pass hands the builder a view whose material slots hold ``mu_value`` --
    ``1.0`` for a material that named none -- so the same reader decides each quantity's
    interface set from its own slots, as the raster path's twin records do. On the seam line
    inside the run the eps view (``4 | 4``) reads ``n_x == 0`` exactly -- the pair cancels and
    the tie answers the boxes' v axis -- and the mu view (``mu 2 | 3``) reads ``|n_x| == 1``
    exactly: both weights are one there and nothing else survives, so any contrast above zero
    gives the seam's direction, ``(1/5) ** 4`` here. Off the line the strength is the
    contrast's, which is why the mu field is no longer array-equal to the ``eps 4 | 9`` field
    (``(5/13) ** 4``) as it was under the equality bit. A box that names ``mu = 1.0`` beside
    one that names none is one permeability, and the mu field says so.
    """
    left, right = K_SHORT
    eps_alike = [dict(left, material=Material(4.0, mu=2.0)),
                 dict(right, material=Material(4.0, mu=3.0))]

    def view(shapes, quantity):
        key = {'permittivity': 'epsilon', 'permeability': 'mu_value'}[quantity]
        return [dict(s, material=getattr(s['material'], key)) for s in shapes]

    eps_field = _field(view(eps_alike, 'permittivity'), grid=(128, 128))
    mu_field = _field(view(eps_alike, 'permeability'), grid=(128, 128))
    assert np.array_equal(eps_field[0], _field(D_SHORT, grid=(128, 128))[0])
    assert not np.array_equal(eps_field[0], mu_field[0])
    y = np.linspace(-149.0, 149.0, 75)
    e_x, e_y = _seam_column(view(eps_alike, 'permittivity'), y)
    m_x, m_y = _seam_column(view(eps_alike, 'permeability'), y)
    assert np.abs(e_x).max() == 0.0 and np.abs(np.abs(e_y) - 1.0).max() == 0.0
    assert np.abs(np.abs(m_x) - 1.0).max() == 0.0 and np.abs(m_y).max() == 0.0

    named = [dict(left, material=Material(4.0, mu=1.0)), dict(right, material=Material(9.0, mu=1.0))]
    absent = [dict(left, material=Material(4.0, mu=1.0)), dict(right, material=Material(9.0))]
    for got, want in zip(_field(view(absent, 'permeability'), grid=(128, 128)),
                         _field(view(named, 'permeability'), grid=(128, 128))):
        assert np.array_equal(got, want)

    # and through the solver, where `_mu_structure` builds that view
    solver = wavebit.rcwa(lattice=SQUARE, structure=[dict(background=1.0, shapes=absent)],
                          truncation_order=dict(u=3, v=3), **SOLVE_KW)
    assert [s['material'] for s in solver._mu_structure()[0]['shapes']] == [1.0, 1.0]
    assert 0.0 < float(np.sum(np.asarray(solver.solve().T))) < 1.0


@pytest.mark.parametrize('alpha', [np.deg2rad(17.0), np.pi / 2])
def test_a_rigid_rotation_of_a_two_shape_layer_with_its_frame_is_an_invariance(alpha):
    """K and D turned with the lattice and ``phi``: the same answer to 1e-11; ``phi`` left
    behind is a different problem. Nothing in a cross pair names an axis either: the run window
    is measured along the primary's own tangent."""
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    for shapes in (K_SHORT, D_SHORT, TWO_STACKED):
        turned = [dict(s, angle=alpha,
                       cx=s['cx'] * np.cos(alpha) - s['cy'] * np.sin(alpha),
                       cy=s['cx'] * np.sin(alpha) + s['cy'] * np.cos(alpha)) for s in shapes]
        base = _total(shapes, order=5, normal_grid=256, phi=0.3)
        same = _total(turned, order=5, lattice=rot @ SQUARE, normal_grid=256, phi=0.3 + alpha)
        assert abs(same - base) < 1e-11, (alpha, same - base)
        left_behind = _total(turned, order=5, lattice=rot @ SQUARE, normal_grid=256, phi=0.3)
        assert abs(left_behind - base) > 1e-6, (alpha, left_behind - base)
