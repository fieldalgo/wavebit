"""Fixed support for concrete sphere sweeps, before the traced-geometry door of C6.

The measured numerical errors are in studies/fdtd/sphere_range/records/fdtd_sphere_range_probe.json;
the source-mutation witnesses are archived outside the repository.
"""
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare.occupancy import component_occupancy
from wavebit.fdtd._runtime import runtime

from tests.fixtures.fdtd.fd_helpers import backends


RANGE = ((-.36, .36),)*3
POINTS = ((.22, (-.073, .017, -.009)), (.27, (.071, -.037, .023)))


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def scene(radius=.22, center=POINTS[0][1], bounds=RANGE):
    return fd.Scene(
        domain=fd.Domain(size=(1.6,)*3), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=24), structures=(fd.Structure(
            fd.Sphere(radius=radius, center=center, range=bounds),
            fd.Material(eps=2.25, mu=1.2)),), sources=(fd.Dipole(
                'drive', center=(-.55, .03, -.02), component='Ey',
                band=fd.Band(wavelength=3., df=.03), delay=-18.),))


def test_range_is_finite_static_and_contains_the_whole_sphere():
    ball = fd.Sphere(radius=.25, range=[[-.25, .25]]*3)
    assert ball.range == ((-.25, .25),)*3
    assert all(isinstance(v, float) for pair in ball.range for v in pair)
    invalid = [(), ((-.3, .3),)*2, ((-.3,),)*3,
               ((0., 0.),)*3, ((.3, -.3),)*3,
               ((-math.inf, .3),)*3, ((-.3, math.inf),)*3,
               ((math.nan, .3),)*3, ((-.3, math.nan),)*3]
    for bounds in invalid:
        with pytest.raises(ValueError, match='Sphere.*range'):
            fd.Sphere(radius=.25, range=bounds)
    # One representable step outside on each face is already outside: no guessed tolerance.
    for axis in range(3):
        for side in (0, 1):
            bounds = [[-.25, .25] for _ in range(3)]
            bounds[axis][side] = np.nextafter(bounds[axis][side], 0.)
            with pytest.raises(ValueError, match='whole sphere'):
                fd.Sphere(radius=.25, range=bounds)


@pytest.mark.parametrize('axis', range(3))
def test_reserved_support_holds_volume_at_both_range_ends(axis):
    from wavebit.fdtd._kernel.grid import Grid
    grid = Grid.from_domain((1.6,)*3, (0.,)*3, .1)
    bounds = ((-.4, .4),)*3
    expected = ((3, 14),)*3  # floor/ceil of [4,12], plus the existing one-cell margin
    # The high endpoint is 12.000000000000002 in binary; ceil reserves cell 13 too.
    supports = []
    for side in (-1, 1):
        center = [0., 0., 0.]
        center[axis] = side*.15
        ball = fd.Sphere(radius=.25, center=center, range=bounds)
        support = ball.support(grid)
        assert support == expected
        supports.append(support)
        for field in ('E', 'H'):
            for c in range(3):
                fill, _ = component_occupancy(ball.body(), grid, support, field, c,
                                              runtime('numpy'))
                # Same 1e-12 relative budget as test_fd_occupancy's exact-volume gate.
                assert float(fill.sum())*grid.dx**3 == pytest.approx(ball.volume(), rel=1e-12)
    assert supports[0] == supports[1]


@pytest.mark.parametrize('backend', backends())
def test_sweep_reuses_plan_and_matches_unranged_fields_and_volume(backend):
    rt = runtime(backend)
    plan = quiet(fd.plan, scene())
    compiled = None
    for radius, center in POINTS:
        current = scene(radius, center)
        fresh = quiet(fd.plan, current)
        assert fresh == plan and fresh.skeleton == plan.skeleton
        result = quiet(fd.run, current, backend, plan=plan)
        direct = quiet(fd.run, scene(radius, center, None), 'numpy')
        assert result.plan is plan
        assert result.report.bodies[0].asked == 4*math.pi*radius**3/3
        assert all(row[0].startswith('sphere.range.') for row in result.report.spans)
        for field in ('E', 'H'):
            got = rt.numpy(getattr(result.final, field))
            want = getattr(direct.final, field)
            # 128 double-precision epsilons of the largest field. The retained probe
            # measures both allocation and backend changes before choosing this budget.
            budget = 128*np.finfo(float).eps*np.max(abs(want))
            assert np.max(abs(want)) > 0
            np.testing.assert_allclose(got, want, rtol=0, atol=budget)
        if backend == 'jax':
            if compiled is None:
                compiled = rt._compiled[plan]
            assert rt._compiled[plan] is compiled
            assert compiled._cache_size() == 1


def test_lower_refuses_a_different_range_even_when_the_cells_match():
    original = scene()
    plan = quiet(fd.plan, original)
    assert plan.shape_ranges == ((0, 'sphere', RANGE),)
    changed = scene(bounds=((-.361, .361),)*3)
    assert changed.structures[0].shape.support(plan.grid) == original.structures[0].shape.support(plan.grid)
    assert quiet(fd.plan, changed).skeleton != plan.skeleton
    plain = scene(bounds=None)
    moved = original.replace(structures=(
        fd.Structure(fd.Region(size=(.1,)*3, center=(.6, .6, .6)), fd.Material(eps=1.)),
        *original.structures))
    for current in (changed, plain, moved):
        with pytest.raises(ValueError, match='Sphere.*range.*declarations differ'):
            fd.lower(current, plan)
    with pytest.raises(ValueError, match='Sphere.*range.*declarations differ'):
        fd.lower(original, quiet(fd.plan, plain))
    with pytest.raises(ValueError, match='different scene'):
        quiet(fd.run, changed, plan=plan)


def test_reused_plan_still_checks_the_actual_sphere():
    plan = quiet(fd.plan, scene())
    # A generous allocation cannot turn a sub-resolution sphere into a resolved one.
    with pytest.raises(ValueError, match='cells across'):
        quiet(fd.run, scene(radius=.1), plan=plan)
    original = scene()
    lossy = original.replace(structures=(original.structures[0].replace(
        material=fd.Conductor(eps=2.25, sigma=.1)),))
    with pytest.raises(ValueError, match='conductivity|Conductor'):
        quiet(fd.run, lossy, plan=plan)


def test_range_refuses_a_prepared_stencil():
    with pytest.raises(ValueError, match='stencil.*Sphere.*range'):
        fd.Structure(fd.Sphere(radius=.25, range=RANGE), fd.Material(eps=2.25),
                     discretisation='stencil')


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_range_does_not_silently_detach_a_shape_derivative(backend):
    runtime(backend)
    def identity(r):
        ball = fd.Sphere(radius=r, center=(r*.1, 0., 0.), range=RANGE)
        return ball.radius + ball.center[0]
    if backend == 'jax':
        import jax
        assert float(jax.grad(identity)(.25)) == pytest.approx(1.1)
        with pytest.raises(ValueError, match='Sphere.*radius.*static'):
            jax.grad(lambda r: fd.Sphere(radius=r).radius)(.25)
    else:
        import torch
        r = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
        assert float(torch.autograd.grad(identity(r), r)[0]) == pytest.approx(1.1)
        _, tangent = torch.func.jvp(identity, (r.detach(),), (torch.ones_like(r),))
        assert float(tangent) == pytest.approx(1.1)
        with pytest.raises(ValueError, match='Sphere.*radius.*static'):
            fd.Sphere(radius=r)
