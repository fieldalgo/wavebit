"""Simultaneous public geometry derivatives with disjoint reserved interpolation halos."""
import dataclasses
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower.shape_ranges import leaves, planning_scene
from wavebit.fd._runtime import runtime
from studies.fdtd_multi_shape_grad_probe import POINT, scene, objective
from test_fd_sphere_range import quiet


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_all_four_public_body_gradients_match_numpy(backend):
    p = quiet(fd.plan, scene())
    fn = lambda v: quiet(objective, v, backend, p)
    # Four measured steps: 1e-6 gave worst relative error 1.414e-9. Larger
    # steps cross a cylinder wall/cap switch 2.304e-6 away at one Hy point.
    # Keep the existing 2e-6 budget; smaller-step confirmations are retained.
    h = 1e-6
    expected = np.array([(quiet(objective, POINT+d, 'numpy', p)
                          - quiet(objective, POINT-d, 'numpy', p))/(2*h)
                         for d in np.eye(4)*h])
    value = quiet(objective, POINT, 'numpy', p)
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        got, derivative = quiet(jax.value_and_grad(fn), jnp.asarray(POINT))
        got, derivative = float(got), np.asarray(derivative)
    else:
        import torch
        v = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
        got = fn(v)
        assert got.requires_grad, 'The multi-body objective lost its geometry derivative graph.'
        derivative, = torch.autograd.grad(got, v)
        got, derivative = float(got.detach()), derivative.detach().numpy()
    assert got == pytest.approx(float(value), rel=128*np.finfo(float).eps)
    assert np.all(expected != 0)
    np.testing.assert_allclose(derivative, expected, rtol=2e-6, atol=0.)
    if backend == 'torch':
        _, tangent = quiet(torch.func.jvp, fn, (v.detach(),), (torch.ones_like(v),))
        assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(v.detach(), torch.ones_like(v))
            _, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
            assert tangent is not None
            assert float(tangent) == pytest.approx(float(expected.sum()), rel=2e-6)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_sweeps_keep_all_supports_and_match_unranged_and_reversed_fields(backend):
    p = quiet(fd.plan, scene())
    rt = runtime(backend)
    compiled = None
    for offset in (0., 1.):
        at = POINT+offset*np.array((.004, -.003, .013, -.009))
        current = scene(at)
        fresh = quiet(fd.plan, current)
        assert fresh == p and fresh.skeleton == p.skeleton
        result = quiet(fd.run, current, backend, plan=p)
        assert result.plan is p and len(result.report.bodies) == 4
        assert [row.asked for row in result.report.bodies] == [s.shape.volume() for s in current.structures]
        assert all('.range.' in row[0] for row in result.report.spans)
        plain = current.replace(structures=tuple(s.replace(shape=s.shape.replace(range=None))
                                                  for s in current.structures))
        for comparison in (plain, current.replace(structures=current.structures[::-1])):
            direct = quiet(fd.run, comparison)
            for field in ('E', 'H'):
                want = getattr(direct.final, field)
                # Probe: NumPy order and range changes are exact; backend maximum
                # 6.13e-16 of peak. Keep the existing 128-double-epsilon budget.
                np.testing.assert_allclose(rt.numpy(getattr(result.final, field)), want, rtol=0,
                                           atol=128*np.finfo(float).eps*np.max(abs(want)))
        if backend == 'jax':
            if compiled is None:
                compiled = rt._compiled[p]
            assert rt._compiled[p] is compiled and compiled._cache_size() == 1


def test_separated_coefficients_equal_the_sum_of_independent_body_changes():
    current = scene(POINT+np.array((.004, -.003, .013, -.009)))
    def lower(s):
        return fd.lower(s, quiet(fd.plan, s))
    combined = lower(current)
    base = lower(current.replace(structures=()))
    singles = [lower(current.replace(structures=(s,))) for s in current.structures]
    for key in ('be', 'bh'):
        want = base[key]+sum(single[key]-base[key] for single in singles)
        # The retained four-body probe measured exact equality, including star lifts.
        np.testing.assert_allclose(combined[key], want, rtol=128*np.finfo(float).eps, atol=0.)


@pytest.mark.parametrize('index', range(4))
def test_each_traced_seed_is_restored_at_its_index_and_concrete_partners_stay_current(index):
    import torch
    p = quiet(fd.plan, scene())
    at = POINT+np.array((.004, -.003, .013, -.009))
    values = list(at)
    values[index] = torch.tensor(at[index], dtype=torch.float64, requires_grad=True)
    current = scene(values)
    scheduled = p.with_static(schedule=8)
    concrete = planning_scene(current, scheduled)
    expected = at.copy()
    expected[index] = POINT[index]
    assert concrete == scene(expected)
    assert len(concrete.structures) == 4
    assert all(not isinstance(v, torch.Tensor) for s in concrete.structures for v in leaves(s.shape))
    assert scheduled._shape_seeds == p._shape_seeds
    assert [row[0] for row in p._shape_seeds] == list(range(4))
    with pytest.raises(ValueError, match='requires plan='):
        planning_scene(current, dataclasses.replace(p))
    with pytest.raises(ValueError, match='concrete'):
        quiet(fd.plan, current)
    with pytest.raises(ValueError, match='different scene'):
        fd.lower(current.replace(run=fd.Run(steps=33)), p, 'torch')


def halo_scene(axis, overlap=False):
    # Dyadic coordinates isolate a single-cell halo overlap without float ties.
    centers = [[0.]*3 for _ in range(3)]
    centers[0][axis], centers[2][axis] = -.625, .625
    centers[1][(axis+1) % 3] = 1.25  # a remote body between the pair in scene order
    shapes = []
    for i, center in enumerate(centers):
        bounds = [[c-.3125, c+.3125] for c in center]
        if overlap and i == 2:
            bounds[axis][0] = .125
        shapes.append(fd.Sphere(radius=.26, center=center, range=bounds))
    return fd.Scene(domain=fd.Domain(size=(4.,)*3), resolution=fd.Resolution(dx=.125),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=1), structures=tuple(fd.Structure(s, fd.Material(eps=2.))
                                                        for s in shapes))


@pytest.mark.parametrize('axis', range(3))
def test_every_reserved_pair_checks_the_interpolation_halo_in_all_axes(axis):
    import torch
    for overlap in (False, True):
        original = halo_scene(axis, overlap)
        p = quiet(fd.plan, original)
        supports = [s.shape.support(p.grid) for s in original.structures]
        # The actual spheres and their base supports remain disjoint in both cases.
        assert supports[0][axis][1] <= supports[2][axis][0]
        assert supports[0][axis][1]+1 == supports[2][axis][0]-1+(1 if overlap else 0)
        moving = original.structures[2].replace(shape=original.structures[2].shape.replace(
            radius=torch.tensor(.261, dtype=torch.float64, requires_grad=True)))
        current = original.replace(structures=(*original.structures[:2], moving))
        if overlap:
            with pytest.raises(ValueError, match=r'structures\[0\].*structures\[2\].*halos overlap'):
                planning_scene(current, p)
            with pytest.raises(ValueError, match='halos overlap'):
                fd.lower(current, p, 'torch')
            with pytest.raises(ValueError, match='halos overlap'):
                quiet(fd.run, current, 'torch', plan=p)
        else:
            assert planning_scene(current, p) == original


@pytest.mark.parametrize('index', range(4))
def test_every_partner_material_is_isotropic_and_lossless(index):
    """A traced scalar eps is admitted with the geometry (tests/test_fd_traced_material.py);
    a traced eps given per axis is not one number and is refused like an anisotropic one."""
    import torch
    values = list(POINT)
    values[0] = torch.tensor(POINT[0], dtype=torch.float64, requires_grad=True)
    current = scene(values)
    p = quiet(fd.plan, scene())
    for material in (fd.Conductor(eps=2., sigma=.1), fd.Material(eps=(2., 2.1, 2.)),
                     fd.Material(eps=(values[0]+2., values[0]+2.1, values[0]+2.))):
        structures = list(current.structures)
        structures[index] = structures[index].replace(material=material)
        with pytest.raises(ValueError, match='isotropic lossless|static'):
            planning_scene(current.replace(structures=structures), p)
    with pytest.raises(ValueError, match='isotropic lossless'):
        planning_scene(current.replace(background=fd.Material(eps=(1., 1.1, 1.))), p)


def test_a_concrete_partner_is_rechecked_instead_of_using_its_old_seed():
    import torch
    values = list(POINT)
    values[0] = torch.tensor(POINT[0], dtype=torch.float64, requires_grad=True)
    values[3] = .1  # the partner is concrete and its new height is under-resolved
    p = quiet(fd.plan, scene())
    with pytest.raises(ValueError, match='cells across'):
        fd.lower(scene(values), p, 'torch')
    with pytest.raises(ValueError, match='cells across'):
        quiet(fd.run, scene(values), 'torch', plan=p)


def test_jitted_lowering_combines_every_runtime_geometry_guard():
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene())
    def lower(v):
        params = quiet(fd.lower, scene(v), p, 'jax')
        return params['stable'], jnp.sum(params['be'])
    compiled = jax.jit(lower)
    stable, value = compiled(jnp.asarray(POINT))
    assert bool(stable) and np.isfinite(float(value))
    for index in range(4):
        for bad in (.1, 1., math.nan, math.inf):
            at = POINT.copy()
            at[index] = bad
            stable, value = compiled(jnp.asarray(at))
            # lower() returns the numeric 1/NaN guard; run() turns it into a bool.
            assert np.isnan(float(stable)), (index, bad)
            assert np.isnan(float(value)), (index, bad)
    assert compiled._cache_size() == 1


def test_all_range_records_and_static_scene_inputs_protect_the_plan():
    p = quiet(fd.plan, scene())
    assert [(row[0], row[1]) for row in p.shape_ranges] == list(enumerate(
        ('sphere', 'cylinder', 'box', 'prism')))
    for index in range(4):
        current = scene()
        structures = list(current.structures)
        shape = structures[index].shape
        bounds = (shape.range[0], (-.361, .361), shape.range[2])
        structures[index] = structures[index].replace(shape=shape.replace(range=bounds))
        with pytest.raises(ValueError, match='declarations differ'):
            fd.lower(current.replace(structures=structures), p)
    with pytest.raises(ValueError, match='declarations differ'):
        fd.lower(scene().replace(structures=scene().structures[::-1]), p)


def test_unturned_partner_halos_clear_domain_and_absorbers_too():
    import torch
    for upper, reason in ((.5, None), (.5625, 'absorbing layer'), (.8125, 'inside the domain')):
        left = fd.Sphere(radius=.26, center=(-.625, 0., 0.),
                         range=((-.9375, -.3125), (-.3125, .3125), (-.3125, .3125)))
        right = fd.Box(size=(.53,)*3, center=(.625, 0., 0.),
                       range=((.3125, .9375), (-.3125, upper), (-.3125, .3125)))
        boundaries = (fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()) if upper == .8125
                      else fd.Boundaries.open(fd.PML(cells=2, wavelength=3., allow_thin=True)))
        original = fd.Scene(domain=fd.Domain(size=(3., 2., 2.)), resolution=fd.Resolution(dx=.125),
            boundaries=boundaries,
            run=fd.Run(steps=1), structures=(fd.Structure(left, fd.Material(eps=2.)),
                                           fd.Structure(right, fd.Material(eps=2.))))
        p = quiet(fd.plan, original)
        current = original.replace(structures=(original.structures[0].replace(shape=left.replace(
            radius=torch.tensor(.261, dtype=torch.float64, requires_grad=True))), original.structures[1]))
        if reason:
            with pytest.raises(ValueError, match=reason):
                planning_scene(current, p)
        else:
            assert planning_scene(current, p) == original


def test_traced_reports_name_all_body_kinds_and_monitors_use_current_geometry():
    import torch
    monitors = tuple(fd.Record('r'+str(i), at=(s.shape.center[0]+.1, .03, -.02))
                     for i, s in enumerate(scene().structures))
    p = quiet(fd.plan, scene().replace(monitors=monitors))
    at = POINT+np.array((.004, -.003, .013, -.009))
    v = torch.tensor(at, dtype=torch.float64, requires_grad=True)
    result = quiet(fd.run, scene(v).replace(monitors=monitors), 'torch', plan=p)
    direct = quiet(fd.run, scene(at).replace(monitors=monitors), plan=p)
    assert result.report.bodies == () and result.report.lifts == ()
    assert any('Sphere/Cylinder/Box/Prism geometry is traced' in message and 'unavailable' in message
               for message in result.report.warnings)
    assert len(direct.report.bodies) == 4
    for i in range(4):
        want = direct['r'+str(i)].values
        np.testing.assert_allclose(result['r'+str(i)].values.detach().numpy(), want, rtol=0,
                                   atol=128*np.finfo(float).eps*np.max(abs(want)))
