"""Fixed cylinder allocations, public derivatives and the full-range certificate.

Measurements: studies/fdtd/cylinder_grad/records/fdtd_cylinder_grad_probe.json and
fdtd_cylinder_grad_volumes.json; the source-mutation witnesses are archived outside the
repository.
"""
import dataclasses
import math

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_fd_sphere_range import quiet


RANGE = ((-.36, .36),)*3
POINT = np.array((.22, .47, -.073, .017, -.009))
SECOND = np.array((.26, .51, .071, -.037, .023))


def scene(v=POINT, axis='z', bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Cylinder(radius=v[0], height=v[1], center=v[2:], axis=axis, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def objective(v, axis, backend, plan):
    result = quiet(fd.run, scene(v, axis), backend, plan=plan)
    return (result.final.E**2).sum()


@pytest.mark.parametrize('axis', 'xyz')
def test_range_contains_the_whole_cylinder_and_stays_static(axis):
    half = [.25]*3
    half['xyz'.index(axis)] = .3
    bounds = [[-h, h] for h in half]
    rod = fd.Cylinder(radius=.25, height=.6, axis=axis, range=bounds)
    assert rod.range == tuple(tuple(p) for p in bounds)
    assert all(isinstance(v, float) for p in rod.range for v in p)
    for a in range(3):
        for side in (0, 1):
            changed = [p.copy() for p in bounds]
            changed[a][side] = np.nextafter(changed[a][side], 0.)
            with pytest.raises(ValueError, match='whole cylinder'):
                rod.replace(range=changed)
    for bad in ((), ((-.3, .3),)*2, ((-.3,),)*3, ((0., 0.),)*3,
                ((.3, -.3),)*3, ((-math.inf, .3),)*3, ((-.3, math.nan),)*3):
        with pytest.raises(ValueError, match='Cylinder.*range'):
            rod.replace(range=bad)
    for field in ('radius', 'height'):
        for value in (-.1, 0., math.inf, math.nan, True, 1j, None, [1.]):
            with pytest.raises(ValueError, match='Cylinder'):
                rod.replace(**{field: value})
    import torch
    with pytest.raises(ValueError, match='range.*static'):
        rod.replace(range=((torch.tensor(-.4, requires_grad=True), .4),)*3)
    with pytest.raises(ValueError, match='stencil.*Cylinder.*range'):
        fd.Structure(rod, fd.Material(eps=2.25), discretisation='stencil')


@pytest.mark.parametrize('axis', 'xyz')
def test_reserved_support_preserves_volume_at_opposite_range_ends(axis):
    from wavebit.fdtd._kernel.grid import Grid
    from wavebit.fdtd._prepare.occupancy import component_occupancy
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    for side in (-1, 1):
        rod = fd.Cylinder(radius=.25, height=.5, axis=axis, center=(side*.15,)*3,
                          range=((-.4, .4),)*3)
        # Same floor/ceil and existing one-cell margin as the sphere reservation.
        assert rod.support(grid) == ((3, 14),)*3
        for field in ('E', 'H'):
            for c in range(3):
                fill, _ = component_occupancy(rod.body(), grid, rod.support(grid), field, c,
                                              runtime('numpy'))
                # The retained volume probe measured <= 1.12e-15 relative error;
                # keep the existing exact-occupancy budget of 1e-12.
                assert float(fill.sum())*grid.dx**3 == pytest.approx(rod.volume(), rel=1e-12)


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_concrete_sweeps_reuse_the_plan_and_match_unranged_fields(axis, backend):
    rt = runtime(backend)
    p = quiet(fd.plan, scene(axis=axis))
    compiled = None
    for point in (POINT, SECOND):
        current = scene(point, axis)
        fresh = quiet(fd.plan, current)
        assert fresh == p and fresh.skeleton == p.skeleton
        result = quiet(fd.run, current, backend, plan=p)
        direct = quiet(fd.run, scene(point, axis, None), 'numpy')
        assert result.plan is p
        assert result.report.bodies[0].asked == math.pi*point[0]**2*point[1]
        assert all(row[0].startswith('cylinder.range.') for row in result.report.spans)
        for field in ('E', 'H'):
            want = getattr(direct.final, field)
            # Probe maximum: 3.25e-16 of the largest field; reuse the sphere budget.
            np.testing.assert_allclose(rt.numpy(getattr(result.final, field)), want, rtol=0,
                                       atol=128*np.finfo(float).eps*np.max(abs(want)))
        if backend == 'jax':
            if compiled is None:
                compiled = rt._compiled[p]
            assert rt._compiled[p] is compiled and compiled._cache_size() == 1


@pytest.mark.parametrize('axis', 'xyz')
@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_all_five_public_geometry_derivatives_match_numpy(axis, backend):
    runtime(backend)
    p = quiet(fd.plan, scene(axis=axis))
    fn = lambda v: objective(v, axis, backend, p)
    # The retained 1e-4..1e-6 step sweep measures <= 2.44e-7 relative error at 3e-6.
    # Keep the sphere derivative budget, with no absolute allowance.
    h = 3e-6
    expected = np.array([(objective(POINT+d, axis, 'numpy', p)
                          - objective(POINT-d, axis, 'numpy', p))/(2*h)
                         for d in np.eye(5)*h])
    value = objective(POINT, axis, 'numpy', p)
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
    assert np.all(abs(expected) > .3)  # Probe's smallest sensitivity is 0.386698.
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=0.)


@pytest.mark.parametrize('axis', 'xyz')
def test_jitted_run_marks_invalid_radius_height_and_centers_unstable(axis):
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene(axis=axis))
    def run(v):
        result = quiet(fd.run, scene(v, axis), 'jax', plan=p)
        return (result.final.E**2).sum(), result.diagnostics['stable']
    compiled = jax.jit(run)
    good, stable = compiled(jnp.asarray(POINT))
    assert bool(stable) and np.isfinite(float(good))
    invalid = []
    for a, values in ((0, (.1, .4, math.nan, math.inf)),
                      (1, (.1, .8, math.nan, math.inf))):
        for value in values:
            v = POINT.copy()
            v[a] = value
            invalid.append(v)
    for a in range(3):
        for coordinate in (.2, math.nan, math.inf):
            v = POINT.copy()
            v[a+2] = coordinate
            invalid.append(v)
    # Only the cap exceeds its range here; using radius on the extrusion axis misses it.
    cap = POINT.copy()
    cap[2+'xyz'.index(axis)] = .14
    invalid.append(cap)
    for v in invalid:
        value, stable = compiled(jnp.asarray(v))
        assert not bool(stable), v
        assert np.isnan(float(value)), v
    assert compiled._cache_size() == 1


def test_range_and_axis_changes_require_a_new_plan():
    original = scene()
    p = quiet(fd.plan, original)
    assert p.shape_ranges == ((0, 'cylinder', RANGE, 'z'),)
    for current in (scene(axis='x'), scene(bounds=((-.361, .361),)*3), scene(bounds=None)):
        shape = current.structures[0].shape
        if shape.range is not None:
            assert shape.support(p.grid) == original.structures[0].shape.support(p.grid)
        assert quiet(fd.plan, current).skeleton != p.skeleton
        with pytest.raises(ValueError, match='declarations differ'):
            fd.lower(current, p)
        with pytest.raises(ValueError, match='different scene'):
            quiet(fd.run, current, plan=p)


def test_traced_cylinder_requires_a_certified_scene_and_concrete_seed():
    import torch
    from wavebit.fdtd._prepare.shape_ranges import planning_scene
    v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
    current = scene(v)
    p = quiet(fd.plan, scene())
    base = scene().structures[0]
    only_height = scene().replace(structures=(base.replace(shape=base.shape.replace(
        height=torch.tensor(.47, dtype=torch.float64, requires_grad=True))),))
    with pytest.raises(ValueError, match='concrete Cylinder'):
        quiet(fd.plan, only_height)
    assert planning_scene(only_height, p).structures[0].shape.height == .47
    with pytest.raises(ValueError, match='concrete Cylinder'):
        quiet(fd.plan, current)
    with pytest.raises(ValueError, match='requires plan='):
        quiet(fd.run, current, 'torch')
    with pytest.raises(ValueError, match='requires plan='):
        fd.lower(current, dataclasses.replace(p), 'torch')
    with pytest.raises(ValueError, match='different scene'):
        fd.lower(current.replace(run=fd.Run(steps=25)), p, 'torch')
    s = current.structures[0]
    extra = fd.Structure(fd.Region(size=(.1,)*3, center=(.6,)*3), fd.Material(eps=1.))
    with pytest.raises(ValueError, match='reaches the interpolation halo'):
        planning_scene(current.replace(structures=(s, extra)), p)
    wide = ((-.61, .61),)*3
    wp = quiet(fd.plan, scene(bounds=wide))
    with pytest.raises(ValueError, match='interpolation halo'):
        planning_scene(scene(v, bounds=wide), wp)
    assert p.with_static(schedule=8)._shape_seeds == ((0, .22, tuple(POINT[2:]), .47),)
    seed = planning_scene(current, p).structures[0].shape
    assert seed.radius == .22 and seed.height == .47 and seed.center == tuple(POINT[2:])
    thin = POINT.copy()
    thin[1] = .1
    with pytest.raises(ValueError, match='cells across'):
        quiet(fd.run, scene(thin), plan=p)


@pytest.mark.parametrize('kind', ('Cylinder', 'Box', 'Prism'))
def test_unranged_vector_centers_cannot_silently_detach(kind):
    import torch
    options = dict(Cylinder=dict(radius=.25, height=.5), Box=dict(size=(.5,)*3),
                   Prism=dict(vertices=((-1., -1.), (1., -1.), (0., 1.)), height=.5))
    def construct(center):
        return getattr(fd, kind)(center=center, **options[kind])
    v = torch.tensor((.01, .02, .03), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match='center.*static'):
        construct(v)
    with pytest.raises(ValueError, match='center.*static'):
        torch.func.jvp(lambda x: construct(x).center[0], (v.detach(),), (torch.ones_like(v),))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
        with pytest.raises(ValueError, match='center.*static'):
            construct(dual)
    assert construct(v.detach()).center == tuple(v.detach().numpy())


def test_traced_cylinder_reports_and_monitors_use_current_geometry_near_pml():
    import torch
    monitors = (fd.Record('r', at=(.2, .03, -.02)),
                fd.Field('f', box=fd.Region(size=(.2,)*3, center=(.2, 0., 0.)),
                         components=('Ey',), wavelengths=(3.,)))
    def current(v):
        return scene(v, 'x').replace(
            domain=fd.Domain(size=(2.4,)*3), monitors=monitors,
            boundaries=fd.Boundaries.open(fd.PML(cells=2, wavelength=3., allow_thin=True)))
    p = quiet(fd.plan, current(POINT))
    v = torch.tensor(SECOND, dtype=torch.float64, requires_grad=True)
    rod = current(v).structures[0].shape
    assert not rod.uniform_along(0, (0, p.grid.n[0]), p.grid)
    assert rod.uniform_along(0, (0, 1), p.grid)
    result = quiet(fd.run, current(v), 'torch', plan=p)
    direct = quiet(fd.run, current(SECOND), 'numpy', plan=p)
    assert result.report.bodies == () and result.report.lifts == ()
    assert any('Cylinder geometry is traced' in m and 'unavailable' in m
               for m in result.report.warnings)
    assert any('inner face of the absorber' in m for m in result.report.warnings)
    assert direct.report.bodies[0].asked == math.pi*SECOND[0]**2*SECOND[1]
    for a, b in ((result['r'].values, direct['r'].values),
                 (result['f'].fields['Ey'], direct['f'].fields['Ey'])):
        np.testing.assert_allclose(a.detach().numpy(), b, rtol=0,
                                   atol=128*np.finfo(float).eps*np.max(abs(b)))
