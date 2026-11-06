"""Stage 4, C10 -- two smoothed surfaces in one voxel, against oracles that are not the clip.

The acceptance this file holds was registered before the clip was written
(``docs/validation/fdtd/fd-polytope-clip.md``, "Registered acceptance", gates 1 and 4); every
tolerance below is the registered one and is quoted beside its assertion.

**The oracles, and why each is independent of what it judges**

* *Convex polytopes.* A voxel and each polyhedral body written as half spaces from the body's
  **public** description (centre, size, angle in degrees, vertices) -- not from the kernel's
  vertex code -- intersected by ``scipy.spatial.HalfspaceIntersection`` and measured by
  ``ConvexHull``: the volume, and each facet's area and normal. Nothing in ``_cut``, ``_exact``
  or ``_bodies`` is consulted.
* *Adaptive quadrature.* A sphere's share of a voxel, alone or below a substrate's face, as
  ``scipy.integrate.quad`` along z of the exactly clipped disc section, broken at the poles and
  the face -- the pattern ``tests/fdtd/test_fd_shapes.py`` holds the ball-box clip to, and
  sharing no line with ``clipped_ball_volume``, which the clip uses.
* *The laminate.* Parallel layers in a voxel have a closed form, harmonic across and arithmetic
  along, from the layer thicknesses alone.

**Mutations** -- each made in ``src/wavebit/fdtd/_prepare/clip.py``, run against this file,
watched to fail, and restored exactly; the failing tests are recorded in the validation page.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_root  # noqa: F401
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare import clip, coefficients, constitutive
from wavebit.fdtd._prepare.checks import _clip_tolerance
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import to_numpy

DX = 0.05


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def scene_of(structures, n=24, dx=DX, periodic=True, steps=4, background=1.0, sources=(),
             center=(0., 0., 0.)):
    faces = (dict(x=fd.Periodic(), y=fd.Periodic(), z=fd.PEC()) if periodic
             else dict(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()))
    return fd.Scene(domain=fd.Domain(size=(n * dx,) * 3, center=center),
                    boundaries=fd.Boundaries(**faces), resolution=fd.Resolution(dx=dx),
                    run=fd.Run(steps=steps), background=background, structures=structures,
                    sources=sources)


def substrate(top=0.0, eps=2.25):
    return fd.Structure(fd.Box(size=(4., 4., 1.), center=(0., 0., top - .5)), eps)


# ---------------------------------------------------------------------------------------------
# The scenes each gate reads. Every admitted case of the clip appears in at least one.
# ---------------------------------------------------------------------------------------------

def hexagon_free_triangle():
    return ((-.16, -.11), (.17, -.07), (-.03, .17))


POLYHEDRAL = {
    # two boxes turned alike, faces 0.02 apart along their own u axis: proved disjoint
    'turned boxes apart': (fd.Structure(fd.Box(size=(.3, .2, .3), center=(-.1611, .013, .007),
                                               angle=20), 6.),
                           fd.Structure(fd.Box(size=(.25, .3, .3),
                                               center=(0.111663061269, 0.126111946351, .007),
                                               angle=20), 2.25)),
    # a box pillar sunk into a substrate: flat pairs below the walls, the Box face at the rim
    'box sunk': (substrate(), fd.Structure(fd.Box(size=(.3, .25, .4), center=(.013, -.007, .18)),
                                           12.)),
    # a turned box sunk: its walls are flat and tilted, so the pair is cut by two planes
    'turned box sunk': (substrate(), fd.Structure(fd.Box(size=(.3, .25, .4),
                                                         center=(.013, -.007, .171), angle=27),
                                                  12.)),
    # a convex prism sunk
    'prism sunk': (substrate(), fd.Structure(fd.Prism(vertices=hexagon_free_triangle(),
                                                      height=.4, center=(.011, .006, .183),
                                                      angle=11), 12.)),
}

S7_DIMER = (fd.Structure(fd.Sphere(radius=.25, center=(-.2625, 0., .021)), 12.),
            fd.Structure(fd.Sphere(radius=.25, center=(.2625, 0., .021)), 2.25))

SPHERICAL = {
    'sphere beside a box': (fd.Structure(fd.Box(size=(.3, .3, .3), center=(-.2017, .013, .007)),
                                         4.),
                            fd.Structure(fd.Sphere(radius=.2, center=(.1604, -.011, .019)), 12.)),
    'sphere sunk': (substrate(), fd.Structure(fd.Sphere(radius=.2, center=(.013, -.007, .12)), 12.)),
    'sphere touching': (substrate(), fd.Structure(fd.Sphere(radius=.2, center=(.013, -.007, .2)),
                                                  12.)),
}


def planned(structures, **kwargs):
    scene = scene_of(structures, n=kwargs.pop('n', 32), **kwargs)
    plan = quiet(fd.plan, scene)
    assert len(plan.clips) == 1 and not plan.junctions, (plan.clips, plan.junctions)
    return scene, plan, plan.clips[0]


def written(plan, record, c):
    points = clip._points(plan.grid, record.rows, c)
    return points[record.write[c]], record.fills[c][:, record.write[c]].T


# ---------------------------------------------------------------------------------------------
# The polytope oracle
# ---------------------------------------------------------------------------------------------

def _halfspaces_of(structure):
    """``[(n, offset), ...]`` with ``n . x <= offset``, from the public shape alone."""
    shape = structure.shape
    centre = np.asarray(shape.center, float)
    if isinstance(shape, fd.Box):
        hx, hy = shape.size[0] / 2, shape.size[1] / 2
        outline = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
        height = shape.size[2]
    else:
        outline, height = shape.vertices, shape.height
    turn = math.radians(getattr(shape, 'angle', 0.0) or 0.0)
    co, si = math.cos(turn), math.sin(turn)
    ring = [np.array([u * co - v * si + centre[0], u * si + v * co + centre[1]]) for u, v in outline]
    out = []
    for a, b in zip(ring, ring[1:] + ring[:1]):
        edge = b - a
        normal = np.array([edge[1], -edge[0], 0.]) / np.hypot(*edge)   # outward for CCW
        out.append((normal, float(normal[:2] @ a)))
    out.append((np.array([0., 0., 1.]), centre[2] + height / 2))
    out.append((np.array([0., 0., -1.]), -(centre[2] - height / 2)))
    return out


def _voxel(point, dx):
    out = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.
        out.append((e, point[k] + dx / 2))
        out.append((-e, -(point[k] - dx / 2)))
    return out


def _polytope(planes):
    """``(volume, [(normal, area, offset), ...])`` of ``{n . x <= o}``; zero if it is empty."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from scipy.spatial import HalfspaceIntersection
    a = np.array([n for n, _ in planes])
    b = np.array([o for _, o in planes])
    norm = np.linalg.norm(a, axis=1)
    res = linprog(np.r_[np.zeros(3), -1.], A_ub=np.c_[a, norm], b_ub=b,
                  bounds=[(None, None)] * 3 + [(0, None)], method='highs')
    if res.status != 0 or res.x[3] < 1e-11:
        return 0.0, []
    inside = res.x[:3]
    hs = HalfspaceIntersection(np.c_[a, -b], inside)
    hull = ConvexHull(hs.intersections)
    facets = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        p = hull.points[simplex]
        area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 2
        facets.append((equation[:3], area, -equation[3]))
    return hull.volume, facets


def _on(facets, planes, tol=1e-9):
    """Area and per-component ``area n_c^2`` of the facets lying on any of ``planes``."""
    area, moment = 0.0, np.zeros(3)
    for normal, a, offset in facets:
        for n, o in planes:
            if np.allclose(normal, n, atol=tol) and abs(offset - o) <= tol:
                area += a
                moment += a * normal ** 2
                break
    return area, moment


def oracle(sa, sb, point, dx):
    """Per-medium fills and the exposed interface (area, moment) of one voxel."""
    va, vb = _halfspaces_of(sa), _halfspaces_of(sb)
    cube = _voxel(point, dx)
    vol_a, fa_facets = _polytope(cube + va)
    vol_b, fb_facets = _polytope(cube + vb)
    vol_ab, fab_facets = _polytope(cube + va + vb)
    v = dx ** 3
    fills = ((vol_a - vol_ab) / v, vol_b / v, 1.0 - (vol_a + vol_b - vol_ab) / v)
    a_area, a_moment = _on(fa_facets, va)
    buried_area, buried_moment = _on(fab_facets, va)
    b_area, b_moment = _on(fb_facets, vb)
    return fills, (a_area - buried_area + b_area), (a_moment - buried_moment + b_moment)


@pytest.mark.parametrize('name', list(POLYHEDRAL))
def test_polyhedral_fills_and_interfaces_are_the_polytopes(name):
    """G1a and G1c: every medium fill to 2e-12 of a voxel, the exposed interface to 2e-9 dx^2.

    Registered: 2e-12 rounds up the two-plane cut's 9.8e-13 against this kind of oracle; 2e-9
    rounds up its 1.3e-9 on nearly anti-parallel faces.
    """
    sa, sb = POLYHEDRAL[name]
    scene, plan, record = planned((sa, sb))
    grid = plan.grid
    seen = set()
    for c in range(4):
        points, fills = written(plan, record, c)
        if not len(points):
            continue
        geo = clip.geometry(sa.shape.body(), sb.shape.body(), points, grid.dx,
                            _clip_tolerance(grid))
        seen.update(np.unique(geo['case']).tolist())
        for k, p in enumerate(points):
            want, area, moment = oracle(sa, sb, p, grid.dx)
            assert np.max(np.abs(np.array(want) - fills[k][[0, 1, 2]])) <= 2e-12, (c, p, want,
                                                                                     fills[k])
            assert abs(geo['area'][k] - area) <= 2e-9 * grid.dx ** 2, (c, p, geo['area'][k], area)
            assert np.max(np.abs(geo['moment'][k] - moment)) <= 2e-9 * grid.dx ** 2, (c, p)
    assert seen, 'the scene must write something'
    return seen


def test_every_polyhedral_case_is_exercised():
    seen = set()
    for name in POLYHEDRAL:
        seen |= test_polyhedral_fills_and_interfaces_are_the_polytopes(name)
    assert seen == {0, 1, 2}, seen


# ---------------------------------------------------------------------------------------------
# The quadrature oracle
# ---------------------------------------------------------------------------------------------

def _disc_column(sphere, lo, hi, z0, z1):
    """Volume of the sphere inside the box ``lo..hi`` restricted to ``z0 <= z <= z1``."""
    from scipy.integrate import quad
    from wavebit.fdtd._shapes._exact import clipped_disc_area
    c, r = np.asarray(sphere.center, float), float(sphere.radius)
    a, b = max(lo[2], z0, c[2] - r), min(hi[2], z1, c[2] + r)
    if b <= a:
        return 0.0

    def area(z):
        rho2 = r * r - (z - c[2]) ** 2
        return 0.0 if rho2 <= 0 else float(clipped_disc_area(
            np.float64(c[0]), np.float64(c[1]), np.sqrt(rho2), np.float64(lo[0]),
            np.float64(hi[0]), np.float64(lo[1]), np.float64(hi[1]), np))
    knots = sorted({a, b} | {v for v in (c[2], c[2] - r, c[2] + r) if a < v < b})
    return sum(quad(area, knots[i], knots[i + 1], limit=400, epsabs=1e-16, epsrel=1e-13)[0]
               for i in range(len(knots) - 1))


@pytest.mark.parametrize('name', list(SPHERICAL))
def test_fills_with_a_sphere_are_what_quadrature_says(name):
    """G1b: every medium fill within 1e-10 of a voxel of adaptive quadrature."""
    sa, sb = SPHERICAL[name]
    scene, plan, record = planned((sa, sb))
    dx = plan.dx
    total = 0
    for c in range(4):
        points, fills = written(plan, record, c)
        for p, got in zip(points, fills):
            lo, hi = p - dx / 2, p + dx / 2
            if sa.shape.center[0] < -.1:            # the box beside the sphere: apart
                b = sa.shape
                box = np.prod([max(0.0, min(hi[k], b.center[k] + b.size[k] / 2)
                                   - max(lo[k], b.center[k] - b.size[k] / 2)) for k in range(3)])
                fa = box / dx ** 3
                fb = _disc_column(sb.shape, lo, hi, -np.inf, np.inf) / dx ** 3
                want = (fa, fb, 1 - fa - fb)
            else:
                top = sa.shape.center[2] + sa.shape.size[2] / 2
                below = max(0.0, min(hi[2], top) - lo[2]) * dx * dx / dx ** 3
                fb = _disc_column(sb.shape, lo, hi, -np.inf, np.inf) / dx ** 3
                shared = _disc_column(sb.shape, lo, hi, -np.inf, top) / dx ** 3
                want = (below - shared, fb, 1 - below - fb + shared)
            assert np.max(np.abs(np.array(want) - got)) <= 1e-10, (name, c, p, want, got)
            total += 1
    assert total > 0


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------

def _stack(gap=.003, center=.0137):
    return (fd.Structure(fd.Box(size=(4., 4., .3), center=(0., 0., -.15)), 6.),
            fd.Structure(fd.Box(size=(4., 4., .2), center=(0., 0., gap + .1)), 2.25))


def test_parallel_layers_lower_to_the_laminate():
    """G1d(i): three parallel layers in one voxel, each component to the closed form, 1e-12."""
    scene = scene_of(_stack(), n=24, center=(0., 0., .0137))
    plan = quiet(fd.plan, scene)
    assert plan.clips, 'the gap must share voxels'
    eps = to_numpy(coefficients.build(scene, plan, runtime('numpy'))['be'])
    eps = (plan.dt / plan.dx) / np.broadcast_to(eps, plan.shape)
    layers = ((-.30, 0., 6.), (0., .003, 1.), (.003, .203, 2.25))
    checked = {0: 0, 2: 0}
    record = plan.clips[0]
    for c in (0, 2):
        pts = clip._points(plan.grid, record.rows, c)
        mask = record.write[c]
        z0, y0, x0 = (lo for lo, _ in reversed(record.rows))
        for idx in zip(*np.nonzero(mask)):
            z = pts[idx][2]
            lo, hi = z - plan.dx / 2, z + plan.dx / 2
            f = {}
            for a, b, e in layers:
                f[e] = f.get(e, 0.) + max(0., min(hi, b) - max(lo, a)) / plan.dx
            f[1.] = f.get(1., 0.) + 1 - sum(f.values())
            want = (sum(v * e for e, v in f.items()) if c == 0
                    else 1 / sum(v / e for e, v in f.items()))
            got = eps[(c, idx[0] + z0, idx[1] + y0, idx[2] + x0)]
            assert got == pytest.approx(want, rel=1e-12), (c, z, got, want)
            checked[c] += 1
    assert checked[0] and checked[2], checked


def test_one_absent_medium_is_the_two_phase_rule_to_the_bit():
    """G1d(ii): with ``f2 = 0`` the clip is ``coefficients.mix`` and ``constitutive.lower``'s
    single-body node, bit for bit."""
    rng = np.random.default_rng(11)
    n = 4001
    f1 = rng.uniform(0, 1, n)
    f2 = np.zeros(n)
    fg = 1.0 - f1
    e1, e2, eg = rng.uniform(1, 16, n), rng.uniform(1, 16, n), rng.uniform(1, 16, n)
    w = rng.uniform(0, 1, n)
    assert np.array_equal(clip.diagonal(f1, f2, fg, e1, e2, eg, w),
                          coefficients.mix(np, eg, e1, f1, w))
    normal = rng.normal(size=(n, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    p = np.stack([normal[:, 0] ** 2, normal[:, 1] ** 2, normal[:, 2] ** 2,
                  normal[:, 0] * normal[:, 1], normal[:, 0] * normal[:, 2],
                  normal[:, 1] * normal[:, 2]])
    diag, cross, arithmetic = clip.node(f1, f2, fg, e1, e2, eg, p, np)
    # constitutive.lower's single-body expressions, written as it writes them:
    fill, value, old = f1, e1, eg
    want_arithmetic = fill * value + (1 - fill) * old
    contrast = fill * (1 - fill) * (value - old) ** 2 / (value * old * want_arithmetic)
    weights = np.stack([normal[..., i] * normal[..., j] for i, j in ((0, 1), (0, 2), (1, 2))])
    assert np.array_equal(arithmetic, want_arithmetic)
    assert np.array_equal(cross, contrast[None] * weights)
    assert np.array_equal(diag, np.stack([1.0 / want_arithmetic + contrast * normal[..., a] ** 2
                                          for a in range(3)]))


def test_the_node_tensor_is_never_below_its_arithmetic_floor():
    """The star lift needs ``K >= 1/<eps>``; the clip's ``P`` is a mean of projectors."""
    for name, pair in {**POLYHEDRAL, **SPHERICAL}.items():
        _, plan, record = planned(pair)
        mask = record.write[3]
        if not mask.any():
            continue
        p = record.tensor[:, mask]
        full = np.zeros((mask.sum(), 3, 3))
        for a in range(3):
            full[:, a, a] = p[a]
        for k, (i, j) in enumerate(clip.PAIRS):
            full[:, i, j] = full[:, j, i] = p[3 + k]
        lowest = np.linalg.eigvalsh(full).min()
        assert lowest >= -1e-15 and np.allclose(np.trace(full, axis1=1, axis2=2), 1), name


# ---------------------------------------------------------------------------------------------
# What stays refused, by name (G1e)
# ---------------------------------------------------------------------------------------------

REFUSED = {
    'flush prism': ((substrate(), fd.Structure(fd.Prism(vertices=hexagon_free_triangle(), height=.4,
                                                        center=(0., 0., .2)), 12.)),
                    'flush joint'),
    'one sphere twice': ((fd.Structure(fd.Sphere(radius=.2, center=(.013, -.007, .021)), 12.),) * 2,
                         'two curved surfaces'),
    'union of one material': ((substrate(eps=4.), fd.Structure(fd.Box(size=(.3, .25, .4),
                                                                      center=(.013, -.007, .18)), 4.)),
                              'one material'),
    'cylinder lying across a face': ((substrate(), fd.Structure(fd.Cylinder(
        radius=.12, height=.5, axis='x', center=(0., .011, .1)), 12.)), 'lying across'),
    'two curved surfaces (the S7 dimer)': (S7_DIMER, 'two curved surfaces'),
    'magnetic member': ((substrate(), fd.Structure(fd.Sphere(radius=.2, center=(.013, -.007, .12)),
                                                   fd.Material(eps=12., mu=1.5))), 'declares mu'),
    'a third body in a written voxel': (
        (substrate(), fd.Structure(fd.Sphere(radius=.2, center=(.013, -.007, .12)), 12.),
         fd.Structure(fd.Region(size=(.1, .1, .1), center=(.2, 0., .0)), 3.)),
        'listed after the first pair member'),
}


@pytest.mark.parametrize('name', list(REFUSED))
def test_what_is_not_shown_right_stays_refused(name):
    structures, phrase = REFUSED[name]
    with pytest.raises(ValueError) as err:
        quiet(fd.plan, scene_of(structures, n=32))
    message = str(err.value)
    assert phrase in message, message[-600:]
    assert 'surfaces pass through' in message or 'cell-snapped' in message


def test_three_surfaces_in_one_voxel_stay_refused():
    """Two clips that would write one point: a voxel holding three smoothed surfaces."""
    # three unturned boxes whose faces meet in one corner voxel, a hair apart (no flush faces):
    # every pair is flat there, so each pair alone is a clip, and the corner holds all three
    trio = (fd.Structure(fd.Box(size=(.3, .6, .3), center=(-.1427, .013, .011)), 12.),
            fd.Structure(fd.Box(size=(.3, .3, .3), center=(.1583, -.1361, .011)), 2.25),
            fd.Structure(fd.Box(size=(.3, .3, .3), center=(.1583, .1649, .011)), 4.))
    with pytest.raises(ValueError, match='structures\[2\] is listed after the first pair member'):
        quiet(fd.plan, scene_of(trio, n=32, periodic=False))


def test_traced_geometry_on_a_member_is_refused_by_name():
    import jax
    pair = SPHERICAL['sphere sunk']
    scene = scene_of(pair, n=32)
    plan = quiet(fd.plan, scene)

    def lowered(radius):
        moving = fd.Structure(fd.Sphere(radius=radius, center=(.013, -.007, .12),
                                        range=((-.23, .26), (-.23, .22), (-.1, .34))), 12.)
        return fd.lower(scene.replace(structures=(pair[0], moving)), plan, 'jax')['be'].sum()
    with pytest.raises(ValueError, match='halo|range|traced'):
        jax.grad(lowered)(jax.numpy.asarray(.25))


def test_an_overlapping_pair_with_a_traced_material_is_refused():
    """Which face is buried between two overlapping bodies depends on whether they are one
    material; a traced value cannot say, so a plan is refused rather than guessed."""
    import jax
    from wavebit.fdtd._prepare.clip import classify
    from wavebit.fdtd._prepare.checks import _shared_surface_voxels
    sub, pillar = POLYHEDRAL['box sunk']
    grid = quiet(fd.plan, scene_of((sub, pillar), n=32)).grid
    boxes = [s.shape.support(grid) for s in (sub, pillar)]
    shared = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(*boxes))
    _, _, marks = _shared_surface_voxels(sub.shape.body(), pillar.shape.body(), grid, shared,
                                         include_common=True, masks=True)
    answers = []

    def probe(eps):
        traced = pillar.replace(material=fd.Material(eps=eps))
        answers.append(classify(scene_of((sub, traced), n=32), grid, 0, 1, marks))
        return eps
    jax.jit(probe)(12.)
    assert isinstance(answers[0], str) and 'traced' in answers[0], answers[0]
    concrete = classify(scene_of((sub, pillar), n=32), grid, 0, 1, marks)
    assert isinstance(concrete, clip.Clip), concrete


# ---------------------------------------------------------------------------------------------
# Plans, reports and the executor
# ---------------------------------------------------------------------------------------------

def test_the_plan_records_the_clip_and_execution_drops_its_arrays():
    from wavebit.fdtd._execution import _program
    scene, plan, record = planned(SPHERICAL['sphere sunk'])
    stripped = _program(plan)
    assert stripped.clips[0].write is None and stripped.clips[0].fills is None
    assert stripped.skeleton == plan.skeleton
    report = quiet(fd.check, scene)
    assert 'clip with structures[0]' in str(report)


def test_a_moved_body_is_refused_against_the_plan():
    scene, plan, record = planned(SPHERICAL['sphere sunk'])
    sub, ball = scene.structures
    moved = scene.replace(structures=(sub, ball.replace(shape=fd.Sphere(radius=.2,
                                                                         center=(.014, -.007, .12)))))
    with pytest.raises(ValueError, match='ClipMismatchRefused'):
        fd.lower(moved, plan)


# ---------------------------------------------------------------------------------------------
# Gate 4: backends and derivatives
# ---------------------------------------------------------------------------------------------

def _dipole():
    return (fd.Dipole('d', center=(.031, .017, .45), component='Ex',
                      band=fd.Band(wavelength=4.0, df=.05)),)


GATE4 = {'beside': SPHERICAL['sphere beside a box'], 'sunk': SPHERICAL['sphere sunk'],
         'box sunk': POLYHEDRAL['box sunk']}


@pytest.mark.parametrize('backend', ('jax', 'jax-jit', 'torch'))
@pytest.mark.parametrize('name', list(GATE4))
def test_lowered_coefficients_agree_across_backends(backend, name):
    """Registered: maximum relative difference 1e-12 (``be``, cross terms, lifted ``be``)."""
    scene = scene_of(GATE4[name], n=32)
    plan = quiet(fd.plan, scene)
    expected = fd.lower(scene, plan, 'numpy')
    if backend == 'jax-jit':
        import jax
        actual = jax.jit(lambda: fd.lower(scene, plan, 'jax'))()
    else:
        actual = fd.lower(scene, plan, backend)
    for key in ('be',):
        got, want = to_numpy(actual[key]), np.asarray(expected[key])
        assert np.max(np.abs(got - want) / np.abs(want)) <= 1e-12
    for term, want in expected.get('constitutive', {}).items():
        got = to_numpy(actual['constitutive'][term])
        scale = max(np.max(np.abs(want)), 1e-300)
        assert np.max(np.abs(got - np.asarray(want))) <= 1e-12 * scale


def _evolve(scene, plan, backend, eps=None, medium=None):
    """The final state of ``plan`` (40 steps, the dipole on) with one medium's ``eps`` live."""
    from wavebit.fdtd._execution import execute
    from wavebit.fdtd._runtime import runtime as rt_of
    rt = rt_of(backend)
    if eps is not None:
        structures = list(scene.structures)
        if medium == 2:
            scene = scene.replace(background=fd.Material(eps=eps))
        else:
            structures[medium] = structures[medium].replace(material=fd.Material(eps=eps))
            scene = scene.replace(structures=tuple(structures))
    return execute(plan, fd.lower(scene, plan, rt), rt)


def test_fields_after_forty_steps_agree_across_backends():
    """Registered: at most 1e-11 relative, dipole-driven clip scene."""
    scene = scene_of(GATE4['sunk'], n=32, steps=40, sources=_dipole())
    plan = quiet(fd.plan, scene)
    want = _evolve(scene, plan, 'numpy')
    for backend in ('jax', 'torch'):
        got = _evolve(scene, plan, backend)
        for field in ('E', 'H'):
            w = np.asarray(want[field])
            g = to_numpy(got[field])
            assert np.max(np.abs(g - w)) <= 1e-11 * np.max(np.abs(w)), (backend, field)


@pytest.mark.parametrize('medium', (0, 1, 2))
def test_material_derivatives_through_the_clip(medium):
    """Registered: AD within 1e-6 of a central difference at relative step 1e-4, on JAX and on
    Torch, and JAX within 1e-10 of Torch. ``medium`` 0 and 1 are the pair, 2 the ground."""
    import jax
    scene = scene_of(GATE4['sunk'], n=32, steps=40, sources=_dipole())
    plan = quiet(fd.plan, scene)
    value = (2.25, 12., 1.)[medium]

    def loss(eps, name):
        state = _evolve(scene, plan, name, eps=eps, medium=medium)
        return (state['E'] ** 2 + .3 * state['H'] ** 2).sum() / plan.cells

    h = 1e-4 * value
    finite = (float(loss(value + h, 'numpy')) - float(loss(value - h, 'numpy'))) / (2 * h)
    by_jax = float(jax.jit(jax.grad(lambda e: loss(e, 'jax')))(runtime('jax').array(value,
                                                                                   'float64')))
    parameter = runtime('torch').array(value, 'float64').requires_grad_()
    loss(parameter, 'torch').backward()
    by_torch = float(parameter.grad)
    for got in (by_jax, by_torch):
        assert abs(got - finite) <= 1e-6 * abs(got), (got, finite)
    assert abs(by_jax - by_torch) <= 1e-10 * abs(by_jax), (by_jax, by_torch)
    assert by_jax != 0.0


def test_a_pair_called_apart_shares_no_volume_and_one_that_overlaps_is_not():
    """``clip._globally_apart`` against the polytope oracle (convex polygons and boxes) and
    against dense sampling of the disc (cylinders), on configurations a margin away from contact.

    A wrong 'apart' would set the shared volume to zero where it is not, so this is the proof's
    own gate. Two hundred random pairs of each kind; both answers must occur.
    """
    from wavebit.fdtd._prepare.clip import _globally_apart
    rng = np.random.default_rng(5)
    verdicts = {True: 0, False: 0}
    for _ in range(200):
        boxes = [fd.Structure(fd.Box(size=tuple(rng.uniform(.1, .4, 3)),
                                     center=tuple(rng.uniform(-.2, .2, 3)),
                                     angle=float(rng.choice([0., rng.uniform(-60, 60)]))), 2.)
                 for _ in range(2)]
        volume, _ = _polytope(_halfspaces_of(boxes[0]) + _halfspaces_of(boxes[1]))
        apart = _globally_apart(*(b.shape.body() for b in boxes))
        if 0 < volume < 1e-7:
            continue                                   # too near contact to call either way
        assert apart == (volume == 0.0), (apart, volume, boxes)
        verdicts[apart] += 1
    for _ in range(200):
        cyl = fd.Cylinder(radius=float(rng.uniform(.05, .2)), height=.3,
                          center=tuple(rng.uniform(-.2, .2, 3)))
        box = fd.Box(size=tuple(rng.uniform(.1, .4, 3)), center=tuple(rng.uniform(-.2, .2, 3)),
                     angle=float(rng.uniform(-60, 60)))
        planes = _halfspaces_of(fd.Structure(box, 2.))
        theta = np.linspace(0, 2 * np.pi, 721)[:-1]
        rad = np.sqrt(np.linspace(0, 1, 60))[:, None]
        disc = np.stack([cyl.center[0] + cyl.radius * rad * np.cos(theta),
                         cyl.center[1] + cyl.radius * rad * np.sin(theta)], -1).reshape(-1, 2)
        lo, hi = cyl.center[2] - .15, cyl.center[2] + .15
        zs = np.linspace(lo, hi, 7)
        inside = np.zeros(len(disc), bool)
        for z in zs:
            pts = np.c_[disc, np.full(len(disc), z)]
            margin = np.min([o - pts @ n for n, o in planes], axis=0)
            inside |= margin > 2e-3
        near = np.zeros(len(disc), bool)
        for z in zs:
            pts = np.c_[disc, np.full(len(disc), z)]
            margin = np.min([o - pts @ n for n, o in planes], axis=0)
            near |= np.abs(margin) <= 2e-3
        apart = _globally_apart(cyl.shape.body() if hasattr(cyl, 'shape') else cyl.body(),
                                box.body())
        if inside.any():
            assert not apart, (cyl, box)
            verdicts[False] += 1
        elif not near.any():
            assert apart, (cyl, box)
            verdicts[True] += 1
    assert verdicts[True] > 20 and verdicts[False] > 20, verdicts


def test_the_node_tensor_is_the_three_medium_projection():
    """At every written common node: ``K = 1/<eps> I + (<1/eps> - 1/<eps>) P`` from the
    record's fills and ``P``, written here with the plain means (not the clip's
    cancellation-free form), against what ``constitutive.lower`` hands the star lift."""
    scene, plan, record = planned(SPHERICAL['sphere sunk'])
    nodes = {}
    constitutive.lower(scene, plan, runtime('numpy'), nodes=nodes)
    (term,) = plan.constitutive
    tensor, inverse = (np.asarray(v) for v in nodes[term.name])
    box = tuple(reversed(term.box))
    mask = record.write[3]
    assert mask.any()
    e = (2.25, 12., 1.)
    checked = 0
    for idx in zip(*np.nonzero(mask)):
        f = record.fills[(3, slice(None)) + idx]
        p = record.tensor[(slice(None),) + idx]
        a = sum(fi / ei for fi, ei in zip(f, e))
        b = 1 / sum(fi * ei for fi, ei in zip(f, e))
        at = tuple(i + r[0] - t[0] for i, r, t in zip(idx, reversed(record.rows), reversed(box)))
        for k in range(3):
            assert tensor[(k,) + at] == pytest.approx(b + (a - b) * p[k], rel=1e-12)
        assert inverse[at] == pytest.approx(b, rel=1e-12)
        checked += 1
    assert checked == mask.sum()


def test_points_the_clip_does_not_write_keep_the_painters_values(monkeypatch):
    """Every coefficient outside the write set is what the painters alone produce, bit for bit."""
    scene, plan, record = planned(SPHERICAL['sphere sunk'])
    rt = runtime('numpy')
    with_clip = np.asarray(coefficients.build(scene, plan, rt)['be'])
    monkeypatch.setattr(clip, 'paint', lambda rt, record, scene, plan, out, ground, c: out)
    without = np.asarray(coefficients.build(scene, plan, rt)['be'])
    written = np.zeros(with_clip.shape, bool)
    view = tuple(slice(lo, hi) for lo, hi in reversed(record.rows))
    for c in range(3):
        written[(c,) + view] = record.write[c]
    assert np.array_equal(with_clip[~written], without[~written])
    assert not np.array_equal(with_clip[written], without[written])


def test_the_public_lifecycle_reaches_the_clip():
    """``wavebit.fdtd``: a sphere sunk into a substrate is checked, prepared and run, where
    before C10 it was refused at door 5."""
    from wavebit import Material, Units, fdtd as public
    from wavebit.fdtd._lower_public import lower_problem
    sub = public.Structure('substrate', shape=public.Box(size=(2., 2., 1.), center=(0., 0., -.5)),
                           material=Material(epsilon=2.25))
    ball = public.Structure('ball', shape=public.Sphere(radius=.2, center=(.013, -.007, .12)),
                            material=Material(epsilon=12.))
    problem = public.Problem(units=Units(length='um'), scene=public.Scene(
        domain=public.Domain(size=(1.6, 1.6, 1.6)), structures=(sub, ball),
        boundaries=public.Boundaries(x=public.Periodic(), y=public.Periodic(), z=public.PEC())))
    numerics = public.Numerics(grid=public.UniformGrid(spacing=.05),
                               duration=public.Duration(steps=3))
    report = public.check(problem, numerics=numerics)
    assert report.ok, report.issues
    scene, _ = lower_problem(problem, numerics)
    assert len(quiet(fd.plan, scene).clips) == 1
    result = public.prepare(problem, numerics=numerics).run(problem)
    assert result is not None
