"""Several shapes in one layer: one blend over every piece, and a seam decided by the materials.

The lone-shape half of the piece field is judged in ``test_nvf_self_coincident_faces.py`` and
``test_gamma_field.py``. This file is the multi-shape half (2026-09-06, the second commit of the
mechanism): two boxes that abut share a face line over a run, the plan records the pair, and the
two material slots decide the sign -- equal materials cancel there, different ones co-orient so
the seam's normal survives. There is no per-shape stage: a two-stage blend switches extension
at the moment of contact (measured as a 1.1e-2 step in ``sum(T)`` at order 6), and a single sum
over every piece has nothing to switch.

The cases carry the design's letters. **D**: two touching same-material bars are the one bar
they make -- the invariants a 1-D structure must satisfy hold to float noise, and the two
spellings of a 2-D structure differ by a residual that decays with order. **K**: an ``eps 4 |
eps 9`` seam is an interface, exact where the structure is 1-D and 0.22 deg against a
material-aware oracle where it is not (it was a null band). **E**: a shape whose material
equals the background keeps its faces, so the field is a function of the *equality bits* alone
and a material sweep never moves a pair-free field. And the seam is decided per quantity: the
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
from wavebit._core.lattice import as_lattice

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
    self pairs -- and with the materials different the seams co-orient instead of cancelling.
    The structure is a 1-D grating with no background left, so its exact field is the constant
    ``(+-1, 0)``, under which the NVF projector *is* Li's 1-D rule: the parametric solve and a
    raster of the same profile factorized by ``'li'`` must agree to machine precision. They do
    (1.4e-14 and 1.0e-13 at orders 6 and 10). The one-box fold read this seam as a null band.
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
    A sign forced opposed cancels every pair, sends every sample to the tie, and the tie, with
    both of each box's axes cancelled, answers ``(1, 0)``: 17 deg off.
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
    normal, while its two ends are corners like any other. The sign window is the run: beyond
    the seam's end the two faces cancel, which is right -- no interface continues there -- and
    is what the blind oracle (seam dropped) reads worse, since it has no seam to agree with.
    """
    n_x, n_y = _field(K_SHORT)
    aware = band_error(n_x, n_y, GRID, SQUARE, _gamma(K_SHORT, groups=(4.0, 9.0)))
    blind = band_error(n_x, n_y, GRID, SQUARE, _gamma(K_SHORT))
    assert aware[2] > 4000 and aware[0] < 0.3 and aware[1] < 0.002, aware
    assert blind[0] > aware[0] + 0.05, (blind, aware)


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


# ------------------------------------------------------------------------------------- E

def test_a_field_reads_no_material_beyond_the_equality_of_touching_pairs():
    """A background-material shape keeps its faces; only a cross pair's equality bit is read.

    Dropping a shape's faces when its material equals the background is a cliff in a material
    design variable and puts a material fact into a frozen record (C7). So the field of a
    pair-free layer is bit-identical across any material sweep, including through the
    background's value, and a layer with a cross pair depends on nothing but whether the two
    members agree: ``4 | 9`` and ``4 | 1`` are one field, ``4 | 4`` and ``1 | 1`` another.
    """
    lone = [_box(lx=300.0, ly=200.0, angle=0.3)]
    apart = [_box(lx=200.0, ly=150.0, cx=-200.0), _box(lx=150.0, ly=200.0, cx=180.0, cy=100.0)]
    for shapes in (lone, apart):
        fields = [_field([dict(s, material=m) for s in shapes], grid=(128, 128))
                  for m in (4.0, 1.0, 12.25)]
        for got in fields[1:]:
            assert np.array_equal(got[0], fields[0][0]) and np.array_equal(got[1], fields[0][1])

    def seam(left, right):
        return _field([dict(K_SHORT[0], material=left), dict(K_SHORT[1], material=right)],
                      grid=(128, 128))

    differ = [seam(4.0, 9.0), seam(4.0, 1.0), seam(1.0, 4.0)]
    same = [seam(4.0, 4.0), seam(1.0, 1.0), seam(9.0, 9.0)]
    for group in (differ, same):
        for got in group[1:]:
            assert np.array_equal(got[0], group[0][0]) and np.array_equal(got[1], group[0][1])
    assert not np.array_equal(differ[0][0], same[0][0])


def test_the_seam_is_decided_per_quantity():
    """Same permittivity, different permeability: no seam in the eps field, a seam in the mu one.

    The permeability pass hands the builder a view whose material slots hold ``mu_value`` --
    ``1.0`` for a material that named none -- so the same reader decides each quantity's
    interface set from its own slots, as the raster path's twin records do. A box that names
    ``mu = 1.0`` beside one that names none is one permeability, and the mu field says so.
    """
    A = as_lattice(lattice=SQUARE)
    left, right = K_SHORT
    eps_alike = [dict(left, material=Material(4.0, mu=2.0)),
                 dict(right, material=Material(4.0, mu=3.0))]

    def view(shapes, quantity):
        key = {'permittivity': 'epsilon', 'permeability': 'mu_value'}[quantity]
        return [dict(s, material=getattr(s['material'], key)) for s in shapes]

    eps_field = _field(view(eps_alike, 'permittivity'), grid=(128, 128))
    mu_field = _field(view(eps_alike, 'permeability'), grid=(128, 128))
    assert np.array_equal(eps_field[0], _field(D_SHORT, grid=(128, 128))[0])
    assert np.array_equal(mu_field[0], _field(K_SHORT, grid=(128, 128))[0])
    assert not np.array_equal(eps_field[0], mu_field[0])

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
