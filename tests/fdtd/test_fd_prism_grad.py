"""Public prism derivatives, fixed vertex count, and runtime polygon validity."""
import dataclasses
import itertools
import math

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_fd_sphere_range import quiet


OUTLINE = np.array(((-.25, -.21), (.18, -.24), (.28, -.03),
                    (.22, .23), (-.04, .27), (-.24, .14)))
CONCAVE = OUTLINE.copy()
CONCAVE[3] = (.02, .04)
RANGE = ((-.36, .36),)*3
POINT = np.concatenate((OUTLINE.ravel(), (.47, -.031, .017, -.013)))
CASES = (('x', 23., False), ('y', 23., False), ('z', 23., False), ('z', -23., True))
case = pytest.mark.parametrize('axis,angle,concave', CASES, ids=('x', 'y', 'z', 'concave'))


def point(concave=False):
    value = POINT.copy()
    if concave:
        value[:12] = CONCAVE.ravel()
    return value


def scene(v=POINT, axis='z', angle=23., bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Prism(vertices=v[:12].reshape(6, 2), height=v[12], center=v[13:],
                     axis=axis, angle=angle, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def objective(v, axis, angle, backend, plan):
    result = quiet(fd.run, scene(v, axis, angle), backend, plan=plan)
    return (result.final.E[:, 5:11, 5:11, 5:11]**2).sum()


def corners(vertices, height, center, axis, angle):
    k = 'xyz'.index(axis)
    u, v = ((1, 2), (2, 0), (0, 1))[k]
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    out = []
    for (x, y), sign in itertools.product(vertices, (-1, 1)):
        p = list(center)
        p[u], p[v] = p[u]+x*co-y*si, p[v]+x*si+y*co
        p[k] += sign*height/2
        out.append(p)
    return tuple((min(p[a] for p in out), max(p[a] for p in out)) for a in range(3))


@pytest.mark.parametrize('axis', tuple('xyz'))
@pytest.mark.parametrize('angle', (0., 23., -23., 123., 203.))
def test_range_contains_every_rotated_vertex_and_both_caps(axis, angle):
    bounds = corners(OUTLINE, .47, (0.,)*3, axis, angle)
    shape = fd.Prism(vertices=OUTLINE, height=.47, axis=axis, angle=angle,
                     range=[list(p) for p in bounds])
    assert shape.range == bounds
    assert all(isinstance(v, float) for p in shape.vertices+shape.range for v in p)
    for a in range(3):
        for side in (0, 1):
            changed = [list(p) for p in bounds]
            changed[a][side] = np.nextafter(changed[a][side], 0.)
            with pytest.raises(ValueError, match='whole prism'):
                shape.replace(range=changed)
    for bad in ((), ((-.4, .4),)*2, ((-.4,),)*3, ((0., 0.),)*3,
                ((.4, -.4),)*3, ((-math.inf, .4),)*3, ((-.4, math.nan),)*3):
        with pytest.raises(ValueError, match='Prism.*range'):
            shape.replace(range=bad)
    for bad in (0., -.1, math.nan, math.inf, True, 1j, None):
        with pytest.raises(ValueError, match='Prism.*height'):
            shape.replace(height=bad)
    import torch
    with pytest.raises(ValueError, match='range.*static'):
        shape.replace(range=((torch.tensor(-.4, requires_grad=True), .4),)*3)
    with pytest.raises(ValueError, match='angle.*static'):
        shape.replace(angle=torch.tensor(angle, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='axis'):
        shape.replace(axis=torch.tensor(2.))
    with pytest.raises(ValueError, match='stencil.*Prism.*range'):
        fd.Structure(shape, fd.Material(eps=2.25), discretisation='stencil')


def test_unranged_vertices_stay_static_in_every_transform_and_bad_outlines_refuse():
    import torch
    v = torch.tensor(OUTLINE, dtype=torch.float64, requires_grad=True)
    fn = lambda vertices: fd.Prism(vertices=vertices, height=.47).volume()
    with pytest.raises(ValueError, match='vertices.*static'):
        fn(v)
    with pytest.raises(ValueError, match='vertices.*static'):
        torch.func.jvp(fn, (v.detach(),), (torch.ones_like(v),))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
        with pytest.raises(ValueError, match='vertices.*static'):
            fn(dual)
    assert fn(v.detach()) == fn(OUTLINE)
    triangle = ((0, 0), (1, 0), (0, 1))
    for bad in (np.array(triangle, dtype=bool), np.array(triangle, dtype=np.complex64),
                torch.tensor(triangle, dtype=torch.bool),
                torch.tensor(triangle, dtype=torch.complex64)):
        with pytest.raises(ValueError, match='Prism.*vertices.*real'):
            fd.Prism(vertices=bad, height=.47)
    shape = scene().structures[0].shape
    for bad in (OUTLINE[:2], OUTLINE[::-1], OUTLINE[[0, 2, 1, 3, 4, 5]],
                ((0., 0.),)*3, ((0.,),)*3, 'outline'):
        with pytest.raises(ValueError, match='Prism.*vertices'):
            shape.replace(vertices=bad)
    for bad in (math.nan, math.inf, 1j, True, None):
        vertices = [list(p) for p in OUTLINE]
        vertices[2][1] = bad
        with pytest.raises(ValueError, match='Prism.*vertices'):
            shape.replace(vertices=vertices)


@case
def test_reserved_support_recovers_volume_at_opposite_corners(axis, angle, concave):
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare.occupancy import component_occupancy
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    vertices = CONCAVE if concave else OUTLINE
    v = np.asarray(vertices)
    # Shoelace sum independent of the public volume() and clipping implementations.
    want = .47*sum(v[i, 0]*v[(i+1) % len(v), 1]-v[(i+1) % len(v), 0]*v[i, 1]
                    for i in range(len(v)))/2
    bounds = corners(vertices, .47, (0.,)*3, axis, angle)
    for side in (-1, 1):
        center = tuple(.4-hi if side > 0 else -.4-lo for lo, hi in bounds)
        shape = fd.Prism(vertices=vertices, height=.47, center=center, axis=axis,
                         angle=angle, range=((-.4, .4),)*3)
        assert shape.support(grid) == ((3, 14),)*3
        for field in ('E', 'H'):
            for c in range(3):
                fill, _ = component_occupancy(shape.body(), grid, shape.support(grid),
                                              field, c, runtime('numpy'))
                # Probe maximum 1.12e-15; retain the existing exact-volume budget.
                assert float(fill.sum())*grid.dx**3 == pytest.approx(want, rel=1e-12)


@case
@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_convex_and_concave_sweeps_share_the_plan_capacity_and_fields(axis, angle, concave, backend):
    rt = runtime(backend)
    p = quiet(fd.plan, scene(point(concave), axis, angle))
    compiled = None
    for other in (concave, not concave):
        at = point(other)
        at[12:] = (.49, .023, -.019, .031)
        current = scene(at, axis, angle)
        shape = current.structures[0].shape
        assert shape.body()._convex is False
        fresh = quiet(fd.plan, current)
        assert fresh == p and fresh.skeleton == p.skeleton
        result = quiet(fd.run, current, backend, plan=p)
        direct = quiet(fd.run, scene(at, axis, angle, None), 'numpy')
        assert result.plan is p and result.report.bodies[0].asked == shape.volume()
        assert all(row[0].startswith('prism.range.') for row in result.report.spans)
        for field in ('E', 'H'):
            want = getattr(direct.final, field)
            # Probe maximum 3.66e-16 of the peak field; retain the existing budget.
            np.testing.assert_allclose(rt.numpy(getattr(result.final, field)), want, rtol=0,
                                       atol=128*np.finfo(float).eps*np.max(abs(want)))
        if backend == 'jax':
            if compiled is None:
                compiled = rt._compiled[p]
            assert rt._compiled[p] is compiled and compiled._cache_size() == 1


@case
@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_all_vertex_height_and_center_derivatives_match_numpy(axis, angle, concave, backend):
    runtime(backend)
    at = point(concave)
    p = quiet(fd.plan, scene(at, axis, angle))
    fn = lambda v: objective(v, axis, angle, backend, p)
    # Four probe steps measured a worst relative error of 8.06e-7 at 3e-6;
    # retain the existing 2e-6 budget. A fixed central observation avoids subtracting
    # two source-dominated ~1070 full-field norms for tiny vertex sensitivities.
    h = 3e-6
    expected = np.array([(objective(at+d, axis, angle, 'numpy', p)
                          - objective(at-d, axis, angle, 'numpy', p))/(2*h)
                         for d in np.eye(16)*h])
    value = objective(at, axis, angle, 'numpy', p)

    def partial(coordinate):
        current = scene(at, axis, angle)
        structure = current.structures[0]
        vertices = [list(row) for row in structure.shape.vertices]
        vertices[2][1] = coordinate  # one leaf among otherwise plain numeric coordinates
        current = current.replace(structures=(structure.replace(
            shape=structure.shape.replace(vertices=vertices)),))
        result = quiet(fd.run, current, backend, plan=p)
        return (result.final.E[:, 5:11, 5:11, 5:11]**2).sum()

    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        got, derivative = quiet(jax.value_and_grad(fn), jnp.asarray(at))
        derivative = np.asarray(derivative)
    else:
        import torch
        v = torch.tensor(at, dtype=torch.float64, requires_grad=True)
        got = fn(v)
        derivative, = torch.autograd.grad(got, v)
        got, derivative = float(got.detach()), derivative.detach().numpy()
    # Check the reverse result before trying another transform: a detached
    # vertex mutant must fail for its zero sensitivities, not a later jvp error.
    assert float(got) == pytest.approx(float(value), rel=128*np.finfo(float).eps)
    assert np.all(expected != 0)
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=0.)
    if backend == 'jax':
        one = quiet(jax.grad(partial), jnp.asarray(at[5]))
    else:
        _, tangent = quiet(torch.func.jvp, fn, (v.detach(),), (torch.ones_like(v),))
        assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
            _, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
            assert tangent is not None
            assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
        coordinate = torch.tensor(at[5], dtype=torch.float64, requires_grad=True)
        one, = torch.autograd.grad(partial(coordinate), coordinate)
        one = one.detach()
    assert float(one) == pytest.approx(float(expected[5]), rel=2e-6, abs=0.)


def test_exact_range_orientation_and_vertex_count_are_part_of_the_plan():
    original = scene()
    p = quiet(fd.plan, original)
    assert p.shape_ranges == ((0, 'prism', RANGE, 'z', 23., 6),)
    shape = original.structures[0].shape
    midpoint = tuple((a+b)/2 for a, b in zip(shape.vertices[0], shape.vertices[1]))
    seven = (shape.vertices[0], midpoint, *shape.vertices[1:])
    for changed in (shape.replace(axis='x'), shape.replace(angle=23.00001),
                    shape.replace(vertices=seven), shape.replace(range=((-.361, .361),)*3),
                    shape.replace(range=None)):
        current = original.replace(structures=(original.structures[0].replace(shape=changed),))
        assert quiet(fd.plan, current).skeleton != p.skeleton
        with pytest.raises(ValueError, match='declarations differ'):
            fd.lower(current, p)
        with pytest.raises(ValueError, match='different scene'):
            quiet(fd.run, current, plan=p)


def test_jitted_run_guards_vertices_height_center_and_rotated_containment():
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene())
    def run(v):
        result = quiet(fd.run, scene(v), 'jax', plan=p)
        return (result.final.E**2).sum(), result.diagnostics['stable']
    compiled = jax.jit(run)
    for at in (POINT, point(True)):
        good, stable = compiled(jnp.asarray(at))
        assert bool(stable) and np.isfinite(float(good))
    invalid = []
    for a in range(16):
        values = (.8, math.nan, math.inf) if a < 12 else (
            (.1, 1., math.nan, math.inf) if a == 12 else (.2, math.nan, math.inf))
        for value in values:
            at = POINT.copy()
            at[a] = value
            invalid.append(at)
    # The unturned outline fits y=0.36; its rotated upper vertex does not.
    at = POINT.copy()
    at[14] = .08
    invalid.append(at)
    for vertices in (OUTLINE[::-1], OUTLINE[[0, 2, 1, 3, 4, 5]], OUTLINE*.5,
                     np.zeros_like(OUTLINE)):
        at = POINT.copy()
        at[:12] = vertices.ravel()
        invalid.append(at)
    for at in invalid:
        value, stable = compiled(jnp.asarray(at))
        assert not bool(stable), at
        assert np.isnan(float(value)), at
    assert compiled._cache_size() == 1


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_runtime_outline_door_matches_topology_and_hull_width(backend):
    from types import SimpleNamespace
    from wavebit.fdtd._legacy_geometry import _hull_width
    from wavebit.fdtd._prepare.shape_ranges import _prism_outline_valid
    rt = runtime(backend)
    def valid(vertices, minimum):
        shape = SimpleNamespace(vertices=tuple(tuple(row[i] for i in range(2)) for row in vertices))
        return _prism_outline_valid(shape, rt, minimum)
    if backend == 'jax':
        import jax
        valid = jax.jit(valid)
    for outline in (OUTLINE, CONCAVE, OUTLINE[:3]):
        for scale in (1e-6, 1., 1e6):
            vertices = outline*scale
            width = _hull_width(tuple(map(tuple, vertices)))
            array = rt.array(vertices, 'float64')
            assert bool(valid(array, width*.99))
            assert not bool(valid(array, width*1.01))
    crossing = OUTLINE[[0, 2, 1, 3, 4, 5]]
    # Positive shoelace area alone does not prove a simple outline.
    after = np.roll(crossing, -1, axis=0)
    assert np.sum(crossing[:, 0]*after[:, 1]-after[:, 0]*crossing[:, 1]) > 0
    # Decimal midpoint construction left a 6.07e-18 signed cross product;
    # both the concrete and runtime doors correctly accepted that tiny gap.
    # These dyadic coordinates put vertex 3 exactly on edge 0.
    touching = ((-.25, -.25), (.25, -.25), (.25, .25),
                (0., -.25), (0., .25), (-.25, .25))
    for bad in (crossing, touching, OUTLINE[::-1].copy(), np.zeros_like(OUTLINE)):
        assert not bool(valid(rt.array(bad, 'float64'), 0.))


def test_traced_prisms_need_a_concrete_isolated_seed_and_full_range_clearance():
    import torch
    from wavebit.fdtd._prepare.shape_ranges import planning_scene
    v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
    current = scene(v)
    p = quiet(fd.plan, scene())
    plain = scene().structures[0]
    for changed in (plain.shape.replace(vertices=v[:12].reshape(6, 2)),
                    plain.shape.replace(height=v[12]), plain.shape.replace(center=v[13:])):
        with pytest.raises(ValueError, match='concrete Prism'):
            quiet(fd.plan, scene().replace(structures=(plain.replace(shape=changed),)))
    with pytest.raises(ValueError, match='requires plan='):
        quiet(fd.run, current, 'torch')
    with pytest.raises(ValueError, match='requires plan='):
        fd.lower(current, dataclasses.replace(p), 'torch')
    with pytest.raises(ValueError, match='different scene'):
        fd.lower(current.replace(run=fd.Run(steps=25)), p, 'torch')
    extra = fd.Structure(fd.Region(size=(.1,)*3, center=(.6,)*3), fd.Material(eps=1.))
    with pytest.raises(ValueError, match='reaches the interpolation halo'):
        planning_scene(current.replace(structures=current.structures+(extra,)), p)
    wide = ((-.61, .61),)*3
    wp = quiet(fd.plan, scene(bounds=wide))
    with pytest.raises(ValueError, match='interpolation halo'):
        planning_scene(scene(v, bounds=wide), wp)
    expected = ((0, tuple(map(tuple, OUTLINE)), POINT[12], tuple(POINT[13:])),)
    assert p.with_static(schedule=8)._shape_seeds == expected
    seed = planning_scene(current, p).structures[0].shape
    assert seed.vertices == expected[0][1] and seed.height == POINT[12]
    assert seed.center == tuple(POINT[13:])
    assert all(isinstance(x, float) for row in seed.vertices for x in row)
    for at in (POINT*np.array([.5]*12+[1]*4), POINT*np.array([1]*12+[.5, 1, 1, 1])):
        with pytest.raises(ValueError, match='cells across'):
            quiet(fd.run, scene(at), plan=p)


def test_traced_reports_and_monitors_use_the_current_prism_near_pml():
    import torch
    runtime('jax')
    import jax
    import jax.numpy as jnp
    monitors = (fd.Record('r', at=(.2, .03, -.02)),
                fd.Field('f', box=fd.Region(size=(.2,)*3, center=(.2, 0., 0.)),
                         components=('Ey',), wavelengths=(3.,)))
    def current(v):
        return scene(v).replace(
            domain=fd.Domain(size=(2.4,)*3), monitors=monitors,
            boundaries=fd.Boundaries.open(fd.PML(cells=2, wavelength=3., allow_thin=True)))
    p = quiet(fd.plan, current(POINT))
    uniform = jax.jit(lambda v: current(v).structures[0].shape.uniform_along(
        2, (0, p.grid.n[2]), p.grid))
    assert not bool(uniform(jnp.asarray(POINT)))
    at = point(True)
    at[13:] = (.023, -.019, .031)
    v = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    shape = current(v).structures[0].shape
    assert not shape.uniform_along(2, (0, p.grid.n[2]), p.grid)
    assert shape.uniform_along(2, (0, 1), p.grid)
    result = quiet(fd.run, current(v), 'torch', plan=p)
    direct = quiet(fd.run, current(at), 'numpy', plan=p)
    assert result.report.bodies == () and result.report.lifts == ()
    assert any('Prism geometry is traced' in m and 'unavailable' in m
               for m in result.report.warnings)
    assert any('inner face of the absorber' in m for m in result.report.warnings)
    assert direct.report.bodies[0].asked == current(at).structures[0].shape.volume()
    for a, b in ((result['r'].values, direct['r'].values),
                 (result['f'].fields['Ey'], direct['f'].fields['Ey'])):
        np.testing.assert_allclose(a.detach().numpy(), b, rtol=0,
                                   atol=128*np.finfo(float).eps*np.max(abs(b)))
