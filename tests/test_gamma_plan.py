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
from wavebit._core.lattice import as_lattice

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gamma_instruments import band_error, gamma_points  # noqa: E402

PERIOD = 700.0
SQUARE = np.array([[PERIOD, 0.0], [0.0, PERIOD]])
#: ``a1 = (700, 0)``, ``a2 = (200, 700)``: the image across ``a2`` sits 700 up and 200 over, so a
#: box of ``ly = 700`` meets it face to face over ``lx - 200`` of its width.
SHEAR = np.array([[PERIOD, 200.0], [0.0, PERIOD]])
#: Li's checkerboard primitive cell, where ``a1 + a2 = (2.5, 0)`` and ``a1 - a2 = (0, 2.5)``.
CHECKER = np.array([[1.25, 1.25], [1.25, -1.25]])
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

    a1, a2 = lattice[:, 0], lattice[:, 1]
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


def test_the_ring_is_the_size_rule_and_every_contact_lies_inside_it():
    """``r = 1 + ceil(2R / h_min)`` -- the L box needs 3 on an ordinary square cell.

    Cost-only in one direction (a wider ring finds nothing new) and load-bearing in the other,
    so every recorded contact must sit inside it and the rule must be the stated one.
    """
    for name, (lattice, shape) in SCENES.items():
        A = as_lattice(lattice=lattice)
        prim = P._primitives([shape], A)[0]
        a1, a2 = lattice[:, 0], lattice[:, 1]
        h_min = abs(np.linalg.det(lattice)) / max(np.hypot(*a1), np.hypot(*a2))
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
    """``jax.jit`` has no primal to read: ``None``, then a blind plan -- pieces, ring, no pairs."""
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

    blind = P._blind_plan([SCENES['A grating'][1], SCENES['ellipse'][1]], A)
    assert [pc.face for pc in blind.pieces] == list(O.FACES) + ['arc']
    assert blind.pairs == () and not blind.empty
    assert set(P.plan_gamma([SCENES['A grating'][1]], A).pieces[0].imgs) <= set(blind.pieces[0].imgs)


def test_an_empty_plan_is_a_uniform_layer():
    """The tiling bar's solve is a slab's.

    ``every conv matrix is eps * I whatever the normal field says`` -- so the plan may call
    the layer empty, and the solve cannot tell the bar from the slab.
    """
    import wavebit
    lattice, shape = SCENES['C tiling bar']
    assert P.plan_gamma([shape], as_lattice(lattice=lattice)).empty

    kw = dict(backend=0, lattice=lattice, thickness=[0.3], wavelength=0.94,
              truncation_order=dict(u=4, v=4), top=1.0, bot=1.0, theta=0.1, phi=0.2, psi=0.3)
    bar_t = float(np.sum(wavebit.rcwa(structure=[dict(background=1.0, shapes=[shape])],
                                      **kw).solve().T))
    slab_t = float(np.sum(wavebit.rcwa(structure=[dict(background=12.25, shapes=[])],
                                       **kw).solve().T))
    assert bar_t == pytest.approx(slab_t, abs=1e-12)
