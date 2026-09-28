"""Public geometry derivatives and the entire-range safety contract.

Real source mutations and numerical measurements live in studies/results/fdtd_sphere_grad_*.
"""
import dataclasses

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._runtime import runtime
from test_fd_sphere_range import scene, quiet, POINTS


POINT = np.array((POINTS[0][0], *POINTS[0][1]))


def objective(v, backend, plan):
    result = quiet(fd.run, scene(v[0], tuple(v[1:])), backend, plan=plan)
    return (result.final.E**2).sum()


def differences(function, point=POINT, h=3e-6):
    return np.array([(function(point+d)-function(point-d))/(2*h)
                     for d in np.eye(4)*h])


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_public_radius_and_all_center_derivatives_match_numpy(backend):
    runtime(backend)
    p = quiet(fd.plan, scene())
    expected = differences(lambda v: objective(v, 'numpy', p))
    value = objective(POINT, 'numpy', p)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        got, derivative = quiet(jax.value_and_grad(
            lambda v: objective(v, backend, p)), jnp.asarray(POINT))
        derivative = np.asarray(derivative)
    else:
        import torch
        v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
        got = objective(v, backend, p)
        derivative, = torch.autograd.grad(got, v)
        derivative = derivative.detach().numpy()
        got = float(got.detach())
        fn = lambda x: objective(x, backend, p)
        _, tangent = quiet(torch.func.jvp, fn, (v.detach(),), (torch.ones_like(v),))
        assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
            _, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
            assert tangent is not None
            assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
    assert float(got) == pytest.approx(float(value), rel=128*np.finfo(float).eps)
    # The retained step sweep measures 1e-4 through 1e-6; at 3e-6 the
    # worst relative discrepancy is 1.084e-8. A 2e-6 budget leaves room for cancellation.
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=0.)
    assert np.all(abs(expected) > 1.)


def test_jitted_public_run_marks_invalid_geometry_unstable():
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene())
    def run(v):
        result = quiet(fd.run, scene(v[0], tuple(v[1:])), 'jax', plan=p)
        return (result.final.E**2).sum(), result.diagnostics['stable']
    compiled = jax.jit(run)
    good, stable = compiled(jnp.asarray(POINT))
    assert bool(stable) and np.isfinite(float(good))
    invalid = [POINT.copy() for _ in range(7)]
    for v, radius in zip(invalid[:4], (.1, .4, float('nan'), float('inf'))):
        v[0] = radius
    for a, v in enumerate(invalid[4:]):
        v[a+1] = .2
    for v in invalid:
        value, stable = compiled(jnp.asarray(v))
        assert not bool(stable), v
        assert np.isnan(float(value)), v
    assert compiled._cache_size() == 1


def test_traced_preflight_requires_a_certified_scene_and_plan():
    import torch
    from wavebit.fd._lower.shape_ranges import planning_scene
    r = torch.tensor(.22, dtype=torch.float64, requires_grad=True)
    current = scene(radius=r)
    p = quiet(fd.plan, scene())
    with pytest.raises(ValueError, match='concrete Sphere'):
        quiet(fd.plan, current)
    with pytest.raises(ValueError, match='requires plan='):
        quiet(fd.run, current, 'torch')
    with pytest.raises(ValueError, match='requires plan='):
        fd.lower(current, dataclasses.replace(p), 'torch')
    with pytest.raises(ValueError, match='different scene'):
        fd.lower(current.replace(run=fd.Run(steps=25)), p, 'torch')
    with pytest.raises(ValueError, match='declarations differ'):
        fd.lower(scene(radius=r, bounds=((-.361, .361),)*3), p, 'torch')
    s = current.structures[0]
    invalid = [current.replace(structures=(s, fd.Structure(
                   fd.Region(size=(.1,)*3, center=(.6, .6, .6)), fd.Material(eps=2.)))),
               current.replace(initial=fd.InitialField(E=lambda x,y,z: (x*0,y*0,z*0))),
               current.replace(monitors=(fd.Flux('f', plane=fd.Plane(z=.4), wavelengths=(3.,)),))]
    for bad in invalid:
        with pytest.raises(ValueError, match='one isolated ranged sphere'):
            planning_scene(bad, p)
    for material in (fd.Conductor(eps=2.25, sigma=.1), fd.Material(eps=(2., 2.25, 2.)),
                     fd.Material(eps=r+2)):
        with pytest.raises(ValueError, match='isotropic lossless|static'):
            planning_scene(current.replace(structures=(s.replace(material=material),)), p)
    wide = ((-.61, .61),)*3
    wp = quiet(fd.plan, scene(bounds=wide))
    with pytest.raises(ValueError, match='interpolation halo'):
        planning_scene(scene(radius=r, bounds=wide), wp)
    # Scheduling preserves only the concrete numeric seed, not a captured traced scene.
    scheduled = p.with_static(schedule=8)
    assert scheduled._shape_seeds == p._shape_seeds == ((0, .22, POINTS[0][1]),)
    assert planning_scene(current, scheduled).structures[0].shape.radius == .22


def test_traced_monitor_results_and_reports_use_the_actual_geometry():
    import torch
    monitors = (fd.Record('r', at=(.2, .03, -.02)),
                fd.Field('f', box=fd.Region(size=(.2,)*3, center=(.2, 0., 0.)),
                         components=('Ey',), wavelengths=(3.,)))
    base = scene().replace(monitors=monitors)
    p = quiet(fd.plan, base)
    def run(v, backend):
        return quiet(fd.run, scene(v[0], tuple(v[1:])).replace(monitors=monitors),
                     backend, plan=p)
    # A different geometry from the planning seed must reach the fields and both monitors.
    point = np.array((POINTS[1][0], *POINTS[1][1]))
    v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    result = run(v, 'torch')
    direct = run(point, 'numpy')
    assert result.report.bodies == () and result.report.lifts == ()
    assert any('geometry is traced' in message and 'unavailable' in message
               for message in result.report.warnings)
    assert direct.report.bodies[0].asked == pytest.approx(4*np.pi*point[0]**3/3)
    for a, b in ((result['r'].values, direct['r'].values),
                 (result['f'].fields['Ey'], direct['f'].fields['Ey'])):
        np.testing.assert_allclose(a.detach().numpy(), b, rtol=0,
                                   atol=128*np.finfo(float).eps*np.max(abs(b)))
    def reading(res):
        return (abs(res['f'].fields['Ey'])**2).sum() + (res['r'].values**2).sum()
    derivative, = torch.autograd.grad(reading(result), v)
    expected = differences(lambda x: float(reading(run(x, 'numpy'))), point)
    np.testing.assert_allclose(derivative.detach().numpy(), expected, rtol=2e-6, atol=0.)
