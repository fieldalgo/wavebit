"""The plan of a layer's interface set, judged against the dense-outline oracle.

:func:`~wavebit._core.parametric.plan_gamma` sweeps every face of every image of every box for
the runs along which two opposed faces share a line (:func:`~wavebit._core.overlap.face_pairs`),
and records the combinatorics -- which pieces, which pairs, what is dead, whether anything is
left -- with no coordinate and no material in it. The oracle here
(:func:`~wavebit._core.overlap.interfaces_of`, through ``gamma_instruments``) knows none of that:
it samples outlines and asks, point by point, whether a step outward lands inside some image.
Where the two agree, the plan's runs are the geometry's runs.

The scenes are the ones the mechanism was designed on, named by their letters where the design
gave them one: **A** the 1-D grating, its transpose the stripe along x, **C** the bar that tiles
Li's checkerboard cell, **B** the sheared cell and the tilted box on a square cell (both refused
until 2026-09-06), the **L** box that leaves its Wigner-Seitz cell without touching anything,
two boxes that touch nothing, and an ellipse -- which can never be a pair member.
"""
import pathlib
import sys

import numpy as np
import pytest

from wavebit._core import overlap as O
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice, reduced_basis

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gamma_instruments import band_error, gamma_points  # noqa: E402

PERIOD = 700.0
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
#: ``a1 = (700, 0)``, ``a2 = (200, 700)``: the image across ``a2`` sits 700 up and 200 over, so a
#: box of ``ly = 700`` meets it face to face over ``lx - 200`` of its width.
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])
#: Li's checkerboard primitive cell, where ``a1 + a2 = (2.5, 0)`` and ``a1 - a2 = (0, 2.5)``.
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])
#: The sheared cell ``a2 = (50, 700)`` written with the unreduced basis ``a2 = 2 a1 + (50, 700)``:
#: the same structure, and since 2026-09-06 the same plan -- every image is counted on the
#: reduced basis, so the abutting image is ``(0, 1)`` and the ring 3, with the transform to that
#: spelling recorded. On the basis as given the image sat at ``(-2, 1)``, outside any fixed 3x3,
#: which is what a reader of the ring that fell back to one would miss.
UNREDUCED = np.array([[PERIOD, 1450.0], [0.0, PERIOD]])
TILT = float(np.arctan2(1.0, 2.0))
TILT_LY = PERIOD * np.sin(TILT)
L_ANGLE = float(np.arctan2(117.9, 690.0))


def _box(**geom):
    return dict(dict(type='rectangle', cx=0.0, cy=0.0, angle=0.0, material=12.25), **geom)


SCENES = {
    'A grating': (SQUARE, _box(lx=350.0, ly=PERIOD)),
    'stripe along x': (SQUARE, _box(lx=PERIOD, ly=200.0)),
    'C tiling bar': (CHECKER, _box(lx=2.5, ly=1.25)),
    'checkerboard span': (CHECKER, _box(lx=2.5, ly=0.8)),
    'B shear': (SHEAR, _box(lx=500.0, ly=PERIOD)),
    'B tilted': (SQUARE, _box(lx=640.0, ly=TILT_LY, angle=TILT)),
    'L box': (SQUARE, _box(lx=689.0, ly=200.0, angle=L_ANGLE)),
    '350 box': (SQUARE, _box(lx=350.0, ly=350.0)),
    '600x650 at 10 deg': (SQUARE, _box(lx=600.0, ly=650.0, angle=np.deg2rad(10.0))),
    'ellipse': (SQUARE, dict(type='ellipse', cx=0.0, cy=0.0, lx=400.0, ly=200.0, angle=0.3,
                             material=12.25)),
    'unreduced basis': (UNREDUCED, _box(lx=500.0, ly=PERIOD)),
}

#: What the geometry says about each scene: pairs as ``(p face, q face, m, n, flush)``, which
#: faces are dead, whether the plan is empty, and the ring the size rule gives.
EXPECTED = {
    'A grating': dict(pairs={('+v', '-v', 0, 1, True)}, dead={'+v', '-v'}, empty=False, ring=3),
    'stripe along x': dict(pairs={('+u', '-u', 1, 0, True)}, dead={'+u', '-u'}, empty=False,
                           ring=3),
    'C tiling bar': dict(pairs={('+u', '-u', 1, 1, True), ('+v', '-v', 0, -1, False),
                                ('+v', '-v', 1, 0, False)},
                         dead={'+u', '-u', '+v', '-v'}, empty=True, ring=3),
    'checkerboard span': dict(pairs={('+u', '-u', 1, 1, True)}, dead={'+u', '-u'},
                              empty=False, ring=3),
    'B shear': dict(pairs={('+v', '-v', 0, 1, False)}, dead=set(), empty=False, ring=3),
    'B tilted': dict(pairs={('+v', '-v', -1, 0, False)}, dead=set(), empty=False, ring=3),
    'L box': dict(pairs=set(), dead=set(), empty=False, ring=3),
    '350 box': dict(pairs=set(), dead=set(), empty=False, ring=2),
    '600x650 at 10 deg': dict(pairs=set(), dead=set(), empty=False, ring=3),
    'ellipse': dict(pairs=set(), dead=set(), empty=False, ring=2),
    'unreduced basis': dict(pairs={('+v', '-v', 0, 1, False)}, dead=set(), empty=False, ring=3),
}


def _plan(name):
    lattice, shape = SCENES[name]
    return P.plan_gamma([shape], as_lattice(lattice=lattice))


# ------------------------------------------------------------------ the oracle, validated first

def test_the_oracle_reads_the_grating_exactly():
    """A bar spanning the cell has no y-face, and the oracle must not carry one either.

    Validated on the one scene whose field is provable: ``ly = P`` on a square cell gives
    ``(+-1, 0)`` everywhere, so any y-normal in the band would be the oracle's, not the
    field's. It reads 0.000 deg exactly, and keeps no sample at all on the seam.
    """
    lattice, shape = SCENES['A grating']
    gamma = gamma_points([shape], lattice)
    assert np.abs(gamma[:, 3]).max() == 0.0                    # no y-normal survives
    assert np.abs(np.abs(gamma[:, 2]) - 1.0).max() == 0.0
    n_x, n_y = P.normal_vector_field_analytical(np, (256, 256), [shape], np.float64,
                                                as_lattice(lattice=lattice))
    mean, over, count = band_error(n_x, n_y, (256, 256), lattice, gamma)
    assert count > 0 and mean == 0.0 and over == 0.0


def test_the_oracle_keeps_the_neighbours_face_where_it_bounds_background():
    """On the sheared cell the image's bottom face is an interface where it overhangs the box.

    The in-cell box's top face is covered over ``x in [-50, 250]``; the image's bottom face runs
    on to ``x = 450`` and bounds background there, so the oracle keeps *that* part with the
    image's normal ``-v``. A field built from the in-cell box alone never looks there, which
    is the whole reason the piece field carries every face over its images.
    """
    lattice, shape = SCENES['B shear']
    gamma = gamma_points([shape], lattice)
    mid = 0.5 * (lattice[:, 0] + lattice[:, 1])              # the shape sits at the centre
    x, y = gamma[:, 0] - mid[0], gamma[:, 1] - mid[1]
    on_seam = np.abs(y - 350.0) < 1e-9
    assert on_seam.any()
    assert np.all((x[on_seam] < -50.0 + 1e-6) | (x[on_seam] > 250.0 - 1e-6))
    assert (x[on_seam] > 250.0).any() and (x[on_seam] < -50.0).any()


# --------------------------------------------------------------------- the plan vs the oracle

@pytest.mark.parametrize('name', [n for n in SCENES if SCENES[n][1]['type'] == 'rectangle'])
def test_the_runs_are_the_oracles_runs(name):
    """Along every face, a point is covered by a self run iff a step outward lands in an image.

    The sweep works on lines, offsets and extents; the oracle on point-in-box tests of dense
    samples. They agree everywhere except within one sample of a run endpoint, which is the
    sampling's resolution and not a disagreement.
    """
    lattice, shape = SCENES[name]
    A = as_lattice(lattice=lattice)
    prims = P._primitives([shape], A)
    plan = P.plan_gamma([shape], A)
    scale = max(np.hypot(*lattice[:, 0]), np.hypot(*lattice[:, 1]))
    tol = O.COLLINEAR_TOL_REL * scale
    records = O.face_pairs(prims, A, plan.ring, tol)

    R = np.asarray(reduced_basis(A))          # the sweep counts images on the reduced basis
    a1, a2 = R[:, 0], R[:, 1]
    step = 1e-6 * scale
    n = 4000
    for f, (mx, my, nx, ny, sx, sy, h) in enumerate(O.box_faces(prims[0])):
        sigma = ((np.arange(n) + 0.5) / n * 2.0 - 1.0) * h
        px, py = mx + sigma * sx + step * nx, my + sigma * sy + step * ny
        buried = np.zeros(n, dtype=bool)
        for m, k in O.images(plan.ring + 1):
            buried |= O.strictly_inside(O._shifted(prims[0], m * a1[0] + k * a2[0],
                                                   m * a1[1] + k * a2[1]), px, py)
        runs = [(c.lo, c.hi) for c in records if c.f == f and c.j == 0]
        covered = np.zeros(n, dtype=bool)
        near_end = np.zeros(n, dtype=bool)
        for lo, hi in runs:
            covered |= (sigma >= lo) & (sigma <= hi)
            near_end |= (np.abs(sigma - lo) <= 2.0 * h / n) | (np.abs(sigma - hi) <= 2.0 * h / n)
        assert np.array_equal(covered[~near_end], buried[~near_end]), (name, O.FACES[f])
        # and the plan's `dead` is what the oracle's *whole face* says
        assert plan.pieces[f].dead == bool(buried.all()), (name, O.FACES[f])


@pytest.mark.parametrize('name', list(SCENES))
def test_the_plan_is_what_the_geometry_says(name):
    lattice, shape = SCENES[name]
    plan = _plan(name)
    want = EXPECTED[name]
    pairs = {(plan.pieces[pr.p].face, plan.pieces[pr.q].face, pr.m, pr.n, pr.flush)
             for pr in plan.pairs}
    assert pairs == want['pairs'], name
    assert {pc.face for pc in plan.pieces if pc.dead} == want['dead'], name
    assert plan.empty is want['empty'], name
    assert plan.ring == want['ring'], name
    if shape['type'] == 'ellipse':
        assert [pc.face for pc in plan.pieces] == ['arc']
    else:
        assert [pc.face for pc in plan.pieces] == list(O.FACES)


def test_coverage_is_the_union_of_the_runs_not_any_one_of_them():
    """On the checkerboard the bar's v faces are covered by two images taking half each.

    A per-image reading calls each face partial; the union says dead, and dead on all four is
    ``empty`` -- a uniform layer, whatever the field. The single covering vector is
    ``a1 + a2 = (2.5, 0)`` for the u faces, a diagonal of the ring and not a basis vector.
    """
    plan = _plan('C tiling bar')
    lattice, shape = SCENES['C tiling bar']
    A = as_lattice(lattice=lattice)
    records = O.face_pairs(P._primitives([shape], A), A, plan.ring, O.COLLINEAR_TOL_REL * 2.5)
    for f in (2, 3):                                             # the v faces, half-length 1.25
        runs = sorted((c.lo, c.hi) for c in records if c.f == f and c.j == 0)
        assert len(runs) == 2
        assert max(hi - lo for lo, hi in runs) == pytest.approx(1.25, abs=1e-12)
        assert O.covered_length(runs, 1e-12) == pytest.approx(2.5, abs=1e-12)
    assert plan.empty


def test_the_union_counts_a_nested_run_once_and_closes_a_gap_only_below_the_tolerance():
    """:func:`~wavebit._core.overlap.covered_length` on the run layouts no scene above produces.

    Every scene in this file hands it abutting or disjoint runs. A run *nested* in another
    takes the one branch that adds nothing, and summing it instead would over-count the
    coverage and call a face dead that has an interface left on it. The tolerance is the
    plan's rounding-scale one: a gap narrower than it is the same line and is covered, a gap
    wider is an interface and is not.
    """
    tol = 1e-9
    assert O.covered_length([(0.0, 2.0), (0.5, 1.5)], tol) == 2.0            # nested, once
    assert O.covered_length([(0.5, 1.5), (0.0, 2.0)], tol) == 2.0            # order-free
    assert O.covered_length([(0.0, 1.5), (1.0, 2.0)], tol) == 2.0            # overlapping
    assert O.covered_length([(0.0, 1.0), (0.2, 0.4), (1.5, 2.0)], tol) == 1.5  # a gap stays
    assert O.covered_length([(0.0, 1.0), (1.0 + tol / 2, 2.0)], tol) == 2.0  # one line
    assert O.covered_length([(0.0, 1.0), (1.0 + 4 * tol, 2.0)], tol) == pytest.approx(
        2.0 - 4 * tol, abs=1e-15)
    assert O.covered_length([], tol) == 0.0


def test_the_ring_is_the_size_rule_and_every_contact_lies_inside_it():
    """``r = 1 + ceil(2R / h_min)`` -- the L box needs 3 on an ordinary square cell.

    Cost-only in one direction (a wider ring finds nothing new) and load-bearing in the other,
    so every recorded contact must sit inside it and the rule must be the stated one. ``h_min``
    is the shortest lattice height *of the reduced basis*, which is the basis the ring counts
    cells of: on the unreduced spelling the rule read off the basis as given gives 4, and the
    plan is the reduced cell's 3.
    """
    for name, (lattice, shape) in SCENES.items():
        A = as_lattice(lattice=lattice)
        prim = P._primitives([shape], A)[0]
        R = np.asarray(reduced_basis(A))
        a1, a2 = R[:, 0], R[:, 1]
        h_min = abs(np.linalg.det(R)) / max(np.hypot(*a1), np.hypot(*a2))
        want = 1 + int(np.ceil(2.0 * O._bounding_radius(prim) / h_min))
        plan = P.plan_gamma([shape], A)
        assert plan.ring == want, name
        assert all(max(abs(pr.m), abs(pr.n)) <= plan.ring for pr in plan.pairs), name
    assert _plan('L box').ring == 3 and _plan('350 box').ring == 2


def test_a_pair_is_recorded_once_from_its_plus_face():
    """The sweep sees every coincidence twice; the plan keeps the ``+`` face's reading."""
    for name in ('A grating', 'stripe along x', 'B shear', 'B tilted', 'checkerboard span'):
        plan = _plan(name)
        for pr in plan.pairs:
            assert plan.pieces[pr.p].face[0] == '+' and plan.pieces[pr.q].face[0] == '-'
            assert plan.pieces[pr.p].face[1] == plan.pieces[pr.q].face[1]
        mirrored = {(pr.q, pr.p, -pr.m, -pr.n) for pr in plan.pairs}
        assert not mirrored & {(pr.p, pr.q, pr.m, pr.n) for pr in plan.pairs}


def test_the_plan_holds_no_coordinate_and_no_material():
    """Translate the shape, name it a period over, change its material: the plan is the same.

    That is the contract ``geometry.py`` states for a frozen record -- a sweep over the values
    never invalidates it -- and it holds because nothing here is a float: the record hashes.
    """
    for name in ('A grating', 'B shear', 'B tilted', '350 box', 'L box', 'ellipse'):
        lattice, shape = SCENES[name]
        A = as_lattice(lattice=lattice)
        base = P.plan_gamma([shape], A)
        hash(base)
        for other in (dict(shape, cx=123.0, cy=-40.0), dict(shape, cx=PERIOD),
                      dict(shape, material=2.0), dict(shape, cy=-0.75 * PERIOD)):
            assert P.plan_gamma([other], A) == base, (name, other)
        leaves = [v for pc in base.pieces for v in (pc.shape, pc.face, pc.dead)]
        leaves += [v for pc in base.pieces for mn in pc.imgs for v in mn]
        leaves += [v for pr in base.pairs for v in pr]
        assert all(isinstance(v, (int, bool, str)) for v in leaves)


def test_image_lists_are_bounded_by_size_alone():
    """Every offset within ``diam + 2R`` of the fold is listed, ``<=`` at the boundary.

    Checked against a brute-force enumeration over a wide box of offsets, so the search box
    the helper derives from the lattice heights cannot silently clip the set.
    """
    for name in ('350 box', 'L box', 'B shear', 'checkerboard span', 'A grating'):
        lattice, shape = SCENES[name]
        A = as_lattice(lattice=lattice)
        prim = P._primitives([shape], A)[0]
        a1, a2 = lattice[:, 0], lattice[:, 1]
        reach = max(np.hypot(*(a1 + a2)), np.hypot(*(a1 - a2))) + 2.0 * O._bounding_radius(prim)
        brute = {(m, n) for m in range(-12, 13) for n in range(-12, 13)
                 if np.hypot(*(m * a1 + n * a2)) <= reach * (1.0 + 1e-12)}
        plan = P.plan_gamma([shape], A)
        assert set(plan.pieces[0].imgs) == brute, name
        assert (0, 0) in brute
        assert all(pc.imgs == plan.pieces[0].imgs for pc in plan.pieces)


@pytest.mark.parametrize('name, lattice, shape, leaves', [
    ('690 x 200 at 10 deg, 9.8 inside', SQUARE,
     dict(type='ellipse', lx=690.0, ly=200.0, angle=np.deg2rad(10.0)), False),
    ('710 x 200 at 10 deg, 0.1 outside', SQUARE,
     dict(type='ellipse', lx=710.0, ly=200.0, angle=np.deg2rad(10.0)), True),
    ('720 x 200 at 10 deg', SQUARE,
     dict(type='ellipse', lx=720.0, ly=200.0, angle=np.deg2rad(10.0)), True),
    ('a disc filling the cell: on the edge, inside', SQUARE,
     dict(type='circle', lx=PERIOD), False),
    ('700 x 200 at 0: touching the edge, inside', SQUARE,
     dict(type='ellipse', lx=PERIOD, ly=200.0, angle=0.0), False),
    ('sheared cell, 740 x 200 tilted 10 deg off a2, 0.8 outside', SHEAR,
     dict(type='ellipse', lx=740.0, ly=200.0,
          angle=float(np.arctan2(700.0, 200.0)) + np.deg2rad(10.0)), True),
    ('sheared cell, 720 x 200 tilted 10 deg off a2, 9 inside', SHEAR,
     dict(type='ellipse', lx=720.0, ly=200.0,
          angle=float(np.arctan2(700.0, 200.0)) + np.deg2rad(10.0)), False),
])
def test_a_round_shape_takes_the_ring_iff_its_support_leaves_its_wigner_seitz_cell(
        name, lattice, shape, leaves):
    """``h(t) > |t| / 2`` for some lattice vector ``t``: then, and only then, an instance list.

    Inside the cell the nearest centre's level set is the only one whose interface enters the
    cell, and the arc keeps the single instance it always had (off the curve a neighbour's tip
    can be nearer, and the arc reads its own level set there by design); outside it a
    neighbour's boundary enters the cell and
    the piece is carried over the ring with the exact distance. The ring is bounded by
    ``|A s| <= diam + min(a, b) + h(s)`` and is checked against a brute-force enumeration of
    that bound, so the search box cannot silently clip it. A disc filling its cell and an
    ellipse touching its image at a point sit on the threshold and stay inside, by the
    rounding-scale tolerance.
    """
    A = as_lattice(lattice=lattice)
    shape = dict(shape, cx=0.0, cy=0.0, material=12.25)
    plan = P.plan_gamma([shape], A)
    imgs = plan.pieces[0].imgs
    assert (len(imgs) > 1) is leaves, (name, imgs)
    if not leaves:
        assert imgs == ((0, 0),)
        return
    prim = P._primitives([shape], A)[0]
    a1, a2 = lattice[:, 0], lattice[:, 1]
    diam = max(np.hypot(*(a1 + a2)), np.hypot(*(a1 - a2)))
    a, b = prim[3], prim[4]
    brute = {(0, 0)}
    for m in range(-8, 9):
        for n in range(-8, 9):
            t = m * a1 + n * a2
            length = np.hypot(*t)
            if (m, n) != (0, 0) and length <= (diam + min(a, b) + O._support_radius(
                    prim, t[0] / length, t[1] / length)) * (1.0 + 1e-12):
                brute.add((m, n))
    assert set(imgs) == brute, name
    assert set(imgs) <= set(P._ring_offsets(*(float(v) for v in (a1[0], a1[1], a2[0], a2[1])),
                                            diam + a + b))


def test_the_exact_ellipse_distance_is_exact():
    """:func:`_ellipse_distance` against a brute-force search, and its derivative against differences.

    Sixty points per ellipse on three ellipses (a thin one, a swapped-axes one, a near-circle):
    random points inside and out, the major axis inside and outside the evolute, a point
    1e-7 off it, the centre, and boundary points -- the brute force samples the boundary at
    20000 angles and refines the best bracket by golden section. The gradient with respect to
    the semi-axes is the implicit one, because the distance is stationary in the nearest
    point's angle; it is compared with central differences at ``h = 1e-4``, since the value
    is exact only to 5e-13 and a smaller step reads that as derivative (4e-7 at ``h = 1e-6``,
    a relative 3e-4 on the 1.5e-3 that ``d/db`` is near the major-axis tip).
    """
    rng = np.random.default_rng(3)

    def brute(u, v, a, b, n=20000):
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        bx, by = a * np.cos(t), b * np.sin(t)
        out = np.empty(len(u))
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        for i in range(len(u)):
            k = int(np.argmin((u[i] - bx) ** 2 + (v[i] - by) ** 2))
            lo, hi = t[k] - 2.0 * np.pi / n, t[k] + 2.0 * np.pi / n
            g = lambda s: (u[i] - a * np.cos(s)) ** 2 + (v[i] - b * np.sin(s)) ** 2
            c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
            for _ in range(100):
                if g(c) < g(d):
                    hi, d = d, c
                    c = hi - phi * (hi - lo)
                else:
                    lo, c = c, d
                    d = lo + phi * (hi - lo)
            out[i] = np.sqrt(g(0.5 * (lo + hi)))
        return out

    for a, b in ((360.0, 100.0), (100.0, 360.0), (200.0, 199.0)):
        ev = a * (a * a - b * b) / (a * a) if a > b else 0.0
        theta = rng.uniform(0.0, 2.0 * np.pi, 12)
        u = np.concatenate([rng.uniform(-900.0, 900.0, 24), np.linspace(-a * 0.999, a * 0.999, 8),
                            [0.0, ev, ev * 0.999, ev * 1.001, 111.612], a * np.cos(theta),
                            rng.uniform(-a, a, 11)])
        v = np.concatenate([rng.uniform(-900.0, 900.0, 24), np.zeros(8),
                            [0.0, 0.0, 0.0, 0.0, 1e-7], b * np.sin(theta), np.full(11, 1e-7)])
        d = P._ellipse_distance(np, u, v, a, b)
        assert np.abs(d - brute(u, v, a, b)).max() < 1e-11, (a, b)
        assert np.abs(d[-23:-11]).max() < 1e-12                         # the boundary points
        assert d[32] == pytest.approx(min(a, b), abs=1e-12)              # the centre
    # a circle is the radial distance, not a special case
    u, v = rng.uniform(-500.0, 500.0, 50), rng.uniform(-500.0, 500.0, 50)
    assert np.abs(P._ellipse_distance(np, u, v, 120.0, 120.0)
                  - np.abs(np.hypot(u, v) - 120.0)).max() < 1e-12

    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)
    for uu, vv in ((300.0, 40.0), (100.0, 5.0), (359.0, 2.0), (111.612, 1e-7), (-50.0, -90.0)):
        f = lambda a, b, uu=uu, vv=vv: P._ellipse_distance(
            jnp, jnp.asarray([uu]), jnp.asarray([vv]), a, b)[0]
        h = 1e-4
        for arg, fd in ((0, (float(f(360.0 + h, 100.0)) - float(f(360.0 - h, 100.0))) / (2 * h)),
                        (1, (float(f(360.0, 100.0 + h)) - float(f(360.0, 100.0 - h))) / (2 * h))):
            ad = float(jax.grad(f, arg)(360.0, 100.0))
            assert ad == pytest.approx(fd, rel=1e-6, abs=1e-8), (uu, vv, arg)


def test_a_round_shape_is_never_a_pair_member():
    """A strictly convex boundary meets its translate at points only -- even when it touches."""
    A = as_lattice(lattice=SQUARE)
    for shape in (SCENES['ellipse'][1],
                  dict(type='circle', cx=0.0, cy=0.0, lx=PERIOD, material=4.0),
                  dict(type='ellipse', cx=0.0, cy=0.0, lx=PERIOD, ly=200.0, material=4.0)):
        plan = P.plan_gamma([shape], A)
        assert [pc.face for pc in plan.pieces] == ['arc'] and not plan.pairs
        assert not plan.empty and plan.pieces[0].imgs == ((0, 0),)


def test_a_traced_layer_gets_no_plan_and_the_blind_one_has_every_piece():
    """``jax.jit`` has no primal to read: ``None``, then a blind plan -- pieces, ring, no pairs.

    Every piece the *types* name: four faces per box and one arc per circle. An ellipse is
    the one shape whose instance list is a question of its size (:func:`_arc_offsets`), and
    the blind plan cannot read a size, so it refuses one by name rather than build the fold's
    single instance -- which is what it did until 2026-09-06, 8.2e-3 in ``sum(T)`` on the
    720 x 200 ellipse (``test_gamma_frozen.py``).
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)
    A = as_lattice(period=[PERIOD, PERIOD])
    seen = []

    def probe(lx):
        shape = dict(type='rectangle', cx=0.0, cy=0.0, lx=lx, ly=PERIOD, angle=0.0,
                     material=4.0)
        seen.append(P.plan_gamma([shape], A))
        return lx

    jax.jit(probe)(jnp.asarray(350.0))
    assert seen == [None]
    # under grad the primal is there, and the plan is the eager one
    jax.grad(probe)(jnp.asarray(350.0))
    assert seen[1] == P.plan_gamma([SCENES['A grating'][1]], A)

    circle = dict(type='circle', cx=0.0, cy=0.0, lx=400.0, material=12.25)
    blind = P._blind_plan([SCENES['A grating'][1], circle], A)
    assert [pc.face for pc in blind.pieces] == list(O.FACES) + ['arc']
    assert blind.pairs == () and not blind.empty
    assert set(P.plan_gamma([SCENES['A grating'][1]], A).pieces[0].imgs) <= set(blind.pieces[0].imgs)
    assert blind.pieces[-1].imgs == ((0, 0),)           # a circle never leaves its cell
    # An ellipse can leave, and whether it does is its size: not readable here, so refused.
    with pytest.raises(ValueError, match='freeze_geometry'):
        P._blind_plan([SCENES['A grating'][1], SCENES['ellipse'][1]], A)


def test_the_blind_ring_is_read_from_the_basis_and_a_traced_basis_is_refused():
    """The blind plan's ring comes from the cell, and a cell it cannot read is not guessed.

    ``a2 = 10 a1 + (50, 700)`` is the cell of ``a2 = (50, 700)`` written on an unreduced
    basis, accepted as it is. On the basis as given its planned list ran to ring 14 -- 479
    offsets, 460 of them outside a 7x7 -- with the contact at ``(m, n) = (-10, 1)``, and the
    blind plan used to hand a *traced* basis that 7x7 with no argument: the field built on it
    disagreed with the planned one on two thirds of the samples (max 90 deg; 6e-3 in
    ``sum(T)`` through a jitted solve, ``test_gamma_frozen.py``). Since 2026-09-06 every
    image is counted on the reduced basis, so this cell plans exactly as ``a2 = (50, 700)``
    does -- ring 3, 21 offsets, the contact at ``(0, 1)`` -- with the transform recorded, and
    the readable blind ring is the 25 offsets of the reduced cell rather than 1559 (the
    ``2 * diam`` superset of the planned list, for any shape whose bounding radius is at most
    half the reduced cell's diameter). A traced basis still raises the freeze message: it
    cannot be reduced either.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)
    lattice = np.array([[PERIOD, 7050.0], [0.0, PERIOD]])
    A = as_lattice(lattice=lattice)
    shape = _box(lx=500.0, ly=PERIOD)
    plan = P.plan_gamma([shape], A)
    assert plan.ring == 3 and plan.reduction == ((1, -10), (0, 1))
    assert {(pr.m, pr.n) for pr in plan.pairs} == {(0, 1)}
    planned = set(plan.pieces[0].imgs)
    assert len(planned) == 21 and planned <= set(O.images(3))
    spelled = P.plan_gamma([shape], as_lattice(lattice=np.array([[PERIOD, 50.0], [0.0, PERIOD]])))
    assert spelled.reduction == ((1, 0), (0, 1))
    assert plan._replace(reduction=spelled.reduction) == spelled
    blind = P._blind_plan([shape], A)
    assert planned <= set(blind.pieces[0].imgs) and len(blind.pieces[0].imgs) == 25
    assert blind.reduction == plan.reduction

    seen = []

    def probe(L):
        try:
            P._blind_plan([shape], L)
        except ValueError as e:
            seen.append(str(e))
        return jnp.sum(L)

    jax.jit(probe)(jnp.asarray(lattice))
    assert len(seen) == 1 and 'freeze_geometry' in seen[0]


# ----------------------------------------------------------------- the reduced basis

def _own_boundary(shape, A, n=4000):
    """Points on a round shape's boundary with the level set's normal -- no shared arithmetic."""
    _, cx, cy, lx, ly, ang = P.shape_geometry(shape, A)
    a, b = lx / 2.0, ly / 2.0
    ct, st = np.cos(ang), np.sin(ang)
    t = (np.arange(n) + 0.5) / n * 2.0 * np.pi
    u, v = a * np.cos(t), b * np.sin(t)
    gu, gv = np.cos(t) / a, np.sin(t) / b
    gx, gy = gu * ct - gv * st, gu * st + gv * ct
    g = np.hypot(gx, gy)
    return float(cx) + u * ct - v * st, float(cy) + u * st + v * ct, gx / g, gy / g


def _angle_deg(fx, fy, ox, oy):
    cos = np.clip(np.abs(fx * ox + fy * oy) / np.maximum(np.hypot(fx, fy), 1e-300), 0.0, 1.0)
    return np.degrees(np.arccos(cos))


CIRCLE_693 = dict(type='circle', cx=0.0, cy=0.0, lx=693.0, material=12.25)
L_ELLIPSE = dict(type='ellipse', cx=0.0, cy=0.0, lx=720.0, ly=200.0, angle=np.deg2rad(10.0),
                 material=12.25)


def _spelling(k):
    """The cell ``a2 = (50, 700)`` written as ``a2 = k a1 + (50, 700)``; reduced only at ``k = 0``."""
    return np.array([[PERIOD, 50.0 + k * PERIOD], [0.0, PERIOD]])


@pytest.mark.parametrize('name, shape', [('693 circle', CIRCLE_693), ('720 x 200 ellipse', L_ELLIPSE)])
def test_a_round_shape_on_an_unreduced_basis_reads_its_own_level_set(name, shape):
    """``a2 = k a1 + (50, 700)``, ``k = 0..10``: on its own boundary the field is its own normal.

    The fold rounds fractional coordinates and sweeps one cell a side, which finds the nearest
    centre only on a reduced basis (``parametric._fold_to_nearest_image`` has the argument).
    ``as_lattice`` accepts these spellings as they are, and until 2026-09-06 a lone 693 circle
    read its own level set 8.89 deg off in the mean on its boundary at ``k = 3`` (69.9 deg
    max, 900 of 4000 points past 1 deg) and 43.8 / 90.0 at ``k = 10``, on every backend, in
    silence -- identical digits on the fold tree (8bfc6aa), so pre-existing rather than a
    regression. Every image is counted on the reduced basis now, and the ladder reads the
    ``k = 0`` value, 2.4e-7 deg, at every rung; the ellipse keeps its eleven instances on
    every spelling. The oracle here is the shape's own boundary and level set, built from the
    shape dict alone.
    """
    for k in (0, 1, 2, 3, 4, 10):
        A = as_lattice(lattice=_spelling(k))
        P.check_layer_shapes(dict(background=1.0, shapes=[shape]), A)
        plan = P.plan_gamma([shape], A)
        assert len(plan.pieces[0].imgs) == (1 if shape['type'] == 'circle' else 11), (name, k)
        assert plan.reduction == ((1, -k), (0, 1)), (name, k)
        x, y, ox, oy = _own_boundary(shape, A)
        n_x, n_y = P._piece_field(np, x, y, [shape], plan, None, A, np.float64)
        angle = _angle_deg(n_x, n_y, ox, oy)
        assert angle.mean() < 1e-5 and angle.max() < 1e-4, (name, k, angle.mean(), angle.max())


@pytest.mark.parametrize('name, shape', [('350 box', _box(lx=350.0, ly=350.0)),
                                         ('500 x 700 box', _box(lx=500.0, ly=PERIOD)),
                                         ('693 circle', CIRCLE_693),
                                         ('720 x 200 ellipse', L_ELLIPSE)])
def test_every_spelling_of_a_cell_builds_the_same_field(name, shape):
    """The field at the same physical points, ``k = 10`` against ``k = 0``: one field.

    The plan is the same plan up to the recorded transform, and the replay reduces with it, so
    the two spellings run the same arithmetic on the same reduced vectors; what differs is
    the origin each centre is measured from (the centre of the cell as given, five ``a1``
    apart here -- a lattice vector, so the same periodic structure) and the rounding of the
    larger coordinates: the two fields agree to 1.2e-13 in ``|n0 x n10|`` and in their
    components, which is the ulp of a coordinate near 3875. The instrument is the cross
    product, not an angle -- ``arccos`` of a dot product one ulp below 1 reads 1.2e-6 deg, a
    floor of the function rather than of the field. Sampled on a grid around the centre that
    avoids the centre itself: a lone box's centre sample is a tie decided by ulps
    (``_piece_field``, "The tie"), not a difference between spellings.

    The solve is deliberately not compared. ``truncation_order`` cuts a rectangle of
    ``(m, n)`` on each spelling's *own* reciprocal basis, and those are different sets of
    retained orders -- ``|b1|`` is ten times longer on the unreduced spelling -- so the two
    solves differ by truncation, not by geometry, at any order this suite can afford. The
    field is the quantity the reduction touches, and it is compared here point for point.
    """
    grid = np.linspace(-345.0, 345.0, 47) + 1.234567       # off every symmetry line and the centre
    px, py = np.meshgrid(grid, grid)
    fields = {}
    for k in (0, 10):
        A = as_lattice(lattice=_spelling(k))
        plan = P.plan_gamma([shape], A)
        centre = 0.5 * (np.asarray(A[:, 0]) + np.asarray(A[:, 1]))
        fields[k] = P._piece_field(np, centre[0] + px, centre[1] + py, [shape], plan, None, A,
                                   np.float64)
    (n0x, n0y), (n1x, n1y) = fields[0], fields[10]
    cross = np.abs(n0x * n1y - n0y * n1x)                   # sin of the angle, sign-free
    apart = np.minimum(np.hypot(n0x - n1x, n0y - n1y), np.hypot(n0x + n1x, n0y + n1y))
    assert cross.max() < 1e-11 and apart.max() < 1e-11, (name, cross.max(), apart.max())


def test_a_thin_bar_threading_between_its_images_meets_the_far_one():
    """A bar along ``2 a1 + a2`` on the square, as long as that vector: a flush pair at ``(2, 1)``.

    The overlap check accepts it -- it threads between every image of the 3x3 without touching
    one -- and its ends meet its ``(2, 1)`` image face to face, outside the 3x3 on a reduced
    basis. The ring the size rule gives (4) reaches it, both ``u`` faces are dead, and it is the
    scene that separates "the pair sweep reads the ring" from "the pair sweep reads a 3x3"
    now that no accepted spelling of a cell puts a nearer contact past the 3x3.
    """
    A = as_lattice(lattice=SQUARE)
    bar = _box(lx=PERIOD * 5.0 ** 0.5, ly=50.0, angle=float(np.arctan2(1.0, 2.0)))
    P.check_layer_shapes(dict(background=1.0, shapes=[bar]), A)
    plan = P.plan_gamma([bar], A)
    assert plan.ring == 4
    assert {(pr.m, pr.n, pr.flush) for pr in plan.pairs} == {(2, 1, True)}
    assert {pc.face for pc in plan.pieces if pc.dead} == {'+u', '-u'}


def test_an_empty_plan_is_a_uniform_layer():
    """The tiling bar's coefficients are an empty layer's, and its solve is a slab's.

    ``every conv matrix is eps * I whatever the normal field says`` -- checked here on the
    field's own output: the projector coefficients of the bar equal those of a layer with no
    shape at all, because both are the constant ``(1, 0)``.
    """
    import wavebit
    lattice, shape = SCENES['C tiling bar']
    A = as_lattice(lattice=lattice)
    bar = P.normal_projector_coeffs(np, [shape], (64, 64), A, 4, 4, np.complex128)
    none = P.normal_projector_coeffs(np, [], (64, 64), A, 4, 4, np.complex128)
    for got, want in zip(bar, none):
        assert np.array_equal(got, want)

    kw = dict(backend=0, lattice=lattice, thickness=[0.3], wavelength=0.94,
              truncation_order=dict(u=4, v=4), top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3)
    bar_t = float(np.sum(wavebit.rcwa(structure=[dict(background=1.0, shapes=[shape])],
                                      **kw).solve().T))
    slab_t = float(np.sum(wavebit.rcwa(structure=[dict(background=12.25, shapes=[])],
                                       **kw).solve().T))
    assert bar_t == pytest.approx(slab_t, abs=1e-12)
