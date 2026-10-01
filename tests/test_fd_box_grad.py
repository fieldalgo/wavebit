"""Public box derivatives with static orientation and a whole-body range certificate.

Measurements and real source-mutation witnesses live in studies/results/fdtd_box_grad_*.
"""
import dataclasses
import itertools
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._runtime import runtime
from test_fd_sphere_range import quiet


RANGE = ((-.36, .36),)*3
POINT = np.array((.43, .46, .49, -.031, .017, -.013))
SECOND = np.array((.46, .43, .51, .023, -.019, .031))
CASES = (('z', 0.), ('x', 23.), ('y', 23.), ('z', 23.))
case = pytest.mark.parametrize('axis,angle', CASES, ids=('unturned', 'x', 'y', 'z'))
RANGE_CASES = CASES + tuple((a, t) for a in 'xyz' for t in (-23., 123., 203.))
RANGE_IDS = ('unturned', 'x', 'y', 'z') + tuple(
    f'{a}-{q}' for a in 'xyz' for q in ('negative', 'second', 'third'))


def scene(v=POINT, axis='z', angle=23., bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Box(size=v[:3], center=v[3:], axis=axis, angle=angle, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def objective(v, axis, angle, backend, plan):
    result = quiet(fd.run, scene(v, axis, angle), backend, plan=plan)
    return (result.final.E**2).sum()


def corner_bounds(size, axis, angle):
    """Independent eight-corner construction, without the absolute-cosine shortcut."""
    k = 'xyz'.index(axis)
    u, v = ((1, 2), (2, 0), (0, 1))[k]
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    points = []
    for signs in itertools.product((-1, 1), repeat=3):
        p = [s*d/2 for s, d in zip(signs, size)]
        p[u], p[v] = p[u]*co-p[v]*si, p[u]*si+p[v]*co
        points.append(p)
    return tuple((min(p[a] for p in points), max(p[a] for p in points)) for a in range(3))


@pytest.mark.parametrize('axis,angle', RANGE_CASES, ids=RANGE_IDS)
def test_range_contains_all_rotated_corners_and_is_static(axis, angle):
    bounds = corner_bounds(POINT[:3], axis, angle)
    box = fd.Box(size=POINT[:3], axis=axis, angle=angle, range=[list(p) for p in bounds])
    assert box.range == bounds
    assert all(isinstance(v, float) for pair in box.range for v in pair)
    # Exact contact is legal; one representable step inward on any face is not.
    for a in range(3):
        for side in (0, 1):
            changed = [list(p) for p in bounds]
            changed[a][side] = np.nextafter(changed[a][side], 0.)
            with pytest.raises(ValueError, match='whole box'):
                box.replace(range=changed)
    for bad in ((), ((-.4, .4),)*2, ((-.4,),)*3, ((0., 0.),)*3,
                ((.4, -.4),)*3, ((-math.inf, .4),)*3, ((-.4, math.nan),)*3):
        with pytest.raises(ValueError, match='Box.*range'):
            box.replace(range=bad)
    for a in range(3):
        for bad in (0., -.1, math.nan, math.inf, True, 1j, None):
            size = list(POINT[:3])
            size[a] = bad
            with pytest.raises(ValueError, match='Box.*size'):
                box.replace(size=size)
    import torch
    with pytest.raises(ValueError, match='range.*static'):
        box.replace(range=((torch.tensor(-.4, requires_grad=True), .4),)*3)
    with pytest.raises(ValueError, match='angle.*static'):
        box.replace(angle=torch.tensor(angle, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='axis'):
        box.replace(axis=torch.tensor(2.))
    with pytest.raises(ValueError, match='stencil.*Box.*range'):
        fd.Structure(box, fd.Material(eps=2.25), discretisation='stencil')


@case
def test_reserved_support_keeps_exact_volume_at_opposite_corners(axis, angle):
    from wavebit.fd._kernel.grid import Grid
    from wavebit.fd._lower.occupancy import component_occupancy
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    for side in (-1, 1):
        half = [hi for lo, hi in corner_bounds(POINT[:3], axis, angle)]
        center = tuple(side*(.4-h) for h in half)
        box = fd.Box(size=POINT[:3], center=center, axis=axis, angle=angle,
                     range=((-.4, .4),)*3)
        assert box.support(grid) == ((3, 14),)*3
        for field in ('E', 'H'):
            for c in range(3):
                fill, _ = component_occupancy(box.body(), grid, box.support(grid), field, c,
                                              runtime('numpy'))
                # Probe maximum 1.12e-15; retain the existing exact-volume budget.
                assert float(fill.sum())*grid.dx**3 == pytest.approx(math.prod(POINT[:3]),
                                                                   rel=1e-12)


@case
@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_sweeps_reuse_the_operator_and_match_unranged_fields(axis, angle, backend):
    rt = runtime(backend)
    p = quiet(fd.plan, scene(axis=axis, angle=angle))
    assert bool(p.constitutive) == bool(angle)
    compiled = None
    for point in (POINT, SECOND):
        current = scene(point, axis, angle)
        fresh = quiet(fd.plan, current)
        assert fresh == p and fresh.skeleton == p.skeleton
        result = quiet(fd.run, current, backend, plan=p)
        direct = quiet(fd.run, scene(point, axis, angle, None), 'numpy')
        assert result.plan is p
        assert result.report.bodies[0].asked == math.prod(point[:3])
        assert all(row[0].startswith('cuboid.range.') for row in result.report.spans)
        for field in ('E', 'H'):
            want = getattr(direct.final, field)
            # The retained probe measures <= 4.90e-16 of the largest field.
            np.testing.assert_allclose(rt.numpy(getattr(result.final, field)), want, rtol=0,
                                       atol=128*np.finfo(float).eps*np.max(abs(want)))
        if backend == 'jax':
            if compiled is None:
                compiled = rt._compiled[p]
            assert rt._compiled[p] is compiled and compiled._cache_size() == 1


@case
@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_all_six_size_and_center_derivatives_match_numpy(axis, angle, backend):
    runtime(backend)
    p = quiet(fd.plan, scene(axis=axis, angle=angle))
    fn = lambda v: objective(v, axis, angle, backend, p)
    # Steps 1e-4..1e-6 were measured first. At 3e-6 the worst relative error is
    # 4.44e-8; the 1e-4 z-rotation step crosses a nearest-face transition.
    # Keep the sphere/cylinder 2e-6 budget and zero absolute allowance.
    h = 3e-6
    expected = np.array([(objective(POINT+d, axis, angle, 'numpy', p)
                          - objective(POINT-d, axis, angle, 'numpy', p))/(2*h)
                         for d in np.eye(6)*h])
    value = objective(POINT, axis, angle, 'numpy', p)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        got, derivative = quiet(jax.value_and_grad(fn), jnp.asarray(POINT))
        derivative = np.asarray(derivative)
    else:
        import torch
        v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
        got = fn(v)
        derivative, = torch.autograd.grad(got, v)
        got, derivative = float(got.detach()), derivative.detach().numpy()
        _, tangent = quiet(torch.func.jvp, fn, (v.detach(),), (torch.ones_like(v),))
        assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
            _, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
            assert tangent is not None
            assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
    assert float(got) == pytest.approx(float(value), rel=128*np.finfo(float).eps)
    assert np.all(abs(expected) > .5)  # The measured smallest sensitivity is 0.570436.
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=0.)


@pytest.mark.parametrize('angle', (0., 23.))
def test_jitted_run_guards_every_size_center_and_rotated_extent(angle):
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene(angle=angle))
    def run(v):
        result = quiet(fd.run, scene(v, angle=angle), 'jax', plan=p)
        return (result.final.E**2).sum(), result.diagnostics['stable']
    compiled = jax.jit(run)
    good, stable = compiled(jnp.asarray(POINT))
    assert bool(stable) and np.isfinite(float(good))
    invalid = []
    for a in range(6):
        for value in ((.1, 1., math.nan, math.inf) if a < 3 else (.2, math.nan, math.inf)):
            v = POINT.copy()
            v[a] = value
            invalid.append(v)
    if angle:
        # The unrotated half-width fits here; the rotated x corner exceeds x=0.36.
        v = POINT.copy()
        v[3] = .1
        invalid.append(v)
    for v in invalid:
        value, stable = compiled(jnp.asarray(v))
        assert not bool(stable), v
        assert np.isnan(float(value)), v
    assert compiled._cache_size() == 1


def test_exact_range_axis_and_angle_are_part_of_the_plan():
    original = scene()
    p = quiet(fd.plan, original)
    assert p.shape_ranges == ((0, 'box', RANGE, 'z', 23.),)
    for current in (scene(axis='x'), scene(angle=23.00001), scene(angle=0.),
                    scene(bounds=((-.361, .361),)*3), scene(bounds=None)):
        shape = current.structures[0].shape
        if shape.range is not None:
            assert shape.support(p.grid) == original.structures[0].shape.support(p.grid)
        assert quiet(fd.plan, current).skeleton != p.skeleton
        with pytest.raises(ValueError, match='declarations differ'):
            fd.lower(current, p)
        with pytest.raises(ValueError, match='different scene'):
            quiet(fd.run, current, plan=p)


def test_traced_boxes_need_a_concrete_isolated_seed_and_full_range_clearance():
    import torch
    from wavebit.fd._lower.shape_ranges import planning_scene
    v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
    current = scene(v)
    p = quiet(fd.plan, scene())
    for size, center in ((v[:3], POINT[3:]), (POINT[:3], v[3:])):
        base = scene().structures[0]
        one = scene().replace(structures=(base.replace(
            shape=base.shape.replace(size=size, center=center)),))
        with pytest.raises(ValueError, match='concrete Box'):
            quiet(fd.plan, one)
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
    assert p.with_static(schedule=8)._shape_seeds == ((0, tuple(POINT[:3]), tuple(POINT[3:])),)
    seed = planning_scene(current, p).structures[0].shape
    assert seed.size == tuple(POINT[:3]) and seed.center == tuple(POINT[3:])
    assert all(isinstance(s, float) for s in (*seed.size, *seed.center))
    for a in range(3):
        thin = POINT.copy()
        thin[a] = .1
        with pytest.raises(ValueError, match='cells across'):
            quiet(fd.run, scene(thin), plan=p)


def test_unranged_box_sizes_stay_static_in_every_torch_transform():
    import torch
    v = torch.tensor(POINT[:3], dtype=torch.float64, requires_grad=True)
    fn = lambda size: fd.Box(size=size).volume()
    with pytest.raises(ValueError, match='size.*static'):
        fn(v)
    with pytest.raises(ValueError, match='size.*static'):
        torch.func.jvp(fn, (v.detach(),), (torch.ones_like(v),))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
        with pytest.raises(ValueError, match='size.*static'):
            fn(dual)
    assert fn(v.detach()) == math.prod(POINT[:3])


def test_traced_reports_and_monitors_use_the_current_box_near_pml():
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
    v = torch.tensor(SECOND, dtype=torch.float64, requires_grad=True)
    shape = current(v).structures[0].shape
    assert not shape.uniform_along(2, (0, p.grid.n[2]), p.grid)
    assert shape.uniform_along(2, (0, 1), p.grid)
    result = quiet(fd.run, current(v), 'torch', plan=p)
    direct = quiet(fd.run, current(SECOND), 'numpy', plan=p)
    assert result.report.bodies == () and result.report.lifts == ()
    assert any('Box geometry is traced' in m and 'unavailable' in m
               for m in result.report.warnings)
    assert any('inner face of the absorber' in m for m in result.report.warnings)
    assert direct.report.bodies[0].asked == math.prod(SECOND[:3])
    for a, b in ((result['r'].values, direct['r'].values),
                 (result['f'].fields['Ey'], direct['f'].fields['Ey'])):
        np.testing.assert_allclose(a.detach().numpy(), b, rtol=0,
                                   atol=128*np.finfo(float).eps*np.max(abs(b)))
