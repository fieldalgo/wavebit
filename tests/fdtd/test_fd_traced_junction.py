"""A traced pillar standing flush on a smoothed substrate: the range-aware 'split' junction.

The scenes are ``studies/fdtd/traced_junction/fdtd_traced_junction_probe.py``'s metasurface cell
(results: ``studies/fdtd/traced_junction/records/fdtd_traced_junction_settled.json``; the probe's
other outputs are archived outside the repository). The write set is recomputed from the live
geometry (``src/wavebit/fdtd/_prepare/junction.py::live_write``), the range certificate is
``shape_ranges._certify_junction`` and the base guard is ``shape_ranges.guard``.
"""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare import junction
from wavebit.fdtd._prepare.shape_ranges import planning_scene
from wavebit.fdtd._runtime import runtime
from studies.fdtd.traced_junction.fdtd_traced_junction_probe import BOX_POINT, PLANE, POINT, objectives, scene
from tests.fdtd.test_fd_sphere_range import quiet

#: The face a quarter cell above a node plane, on a node plane, on a half-node plane. The two
#: ties are where the base-side extension writes points (``Junction.counts``).
PLANES = (PLANE, -.25, -.225)
POINTS = {'cylinder': ((.12, .3, .011), (.1, .25, -.013), (.135, .37, .003), (.127, .21, 0.)),
          'box': ((.23, .21, .3), (.2, .22, .25), (.25, .205, .37), (.214, .233, .21))}


@pytest.mark.parametrize('plane', PLANES)
@pytest.mark.parametrize('kind', ('cylinder', 'box'))
def test_the_live_write_set_is_the_plans_own_at_its_seed(kind, plane):
    rt = runtime('numpy')
    current = scene(kind=kind, plane=plane)
    p = quiet(fd.plan, current)
    (record,) = p.junctions
    for c in range(4):
        points = junction._points(p.grid, record.rows, 'E', c)
        live = np.asarray(junction.live_write(rt, record, current, p.grid, points))
        assert np.array_equal(live, record.write[c]), c
    assert sum(row[1] for row in record.counts) > 0
    if plane != PLANE:
        assert sum(row[3] for row in record.counts) > 0       # base-side points written


def _lowered(scene_, plan, backend):
    try:
        return quiet(fd.lower, scene_, plan, backend)
    except Exception as error:
        raise AssertionError(f'lowering a certified traced scene failed: {error!r}') from error


@pytest.mark.parametrize('plane', PLANES)
@pytest.mark.parametrize('kind', ('cylinder', 'box'))
def test_a_traced_pillar_lowers_and_runs_as_the_concrete_scene_of_its_geometry(kind, plane):
    """Four geometries across the reservation, from the seed's plan: the traced Torch lowering
    and a 40-step run equal the concrete NumPy scene planned and classified at that geometry,
    whose junction writes 26 to 146 electric points. Measured: coefficients within 4.5e-15 of
    their peak, fields within 5.4e-16."""
    import torch
    p = quiet(fd.plan, scene(kind=kind, plane=plane, steps=40))
    rtt = runtime('torch')
    written = set()
    for v in POINTS[kind]:
        traced = scene([torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in v],
                       kind=kind, plane=plane, steps=40)
        concrete = scene(v, kind=kind, plane=plane, steps=40)
        q = quiet(fd.plan, concrete)
        written.add(sum(int(r.write.sum()) for r in q.junctions))
        got, want = _lowered(traced, p, 'torch'), quiet(fd.lower, concrete, q)
        assert got['be'].requires_grad
        pairs = [(got['be'], want['be'])] + [(got['constitutive'][k], want['constitutive'][k])
                                              for k in want.get('constitutive', {})]
        for a, b in pairs:
            np.testing.assert_allclose(rtt.numpy(a.detach()), b, rtol=0,
                                       atol=128 * np.finfo(float).eps * np.max(abs(b)))
        ran = quiet(fd.run, traced, 'torch', plan=p)
        direct = quiet(fd.run, concrete, plan=q)
        for f in ('E', 'H'):
            b = getattr(direct.final, f)
            np.testing.assert_allclose(rtt.numpy(getattr(ran.final, f).detach()), b, rtol=0,
                                       atol=128 * np.finfo(float).eps * np.max(abs(b)))
    assert len(written) > 1                     # the write set really moved with the rim


#: (kind, plane, height override, backend). Measured largest relative Jacobian error of NumPy
#: central differences (each geometry planned on its own) against reverse mode, at steps
#: 1e-4, 1e-5, 3e-6, 1e-6: cylinder 0.27, 1.3e-6, 9.2e-8, 6.4e-8 (a top-rim wall/cap switch
#: of the cylinder's own rule lies 6.4e-5 away in radius); box 3.9e-7, 3.9e-9, 1.2e-9, 1.0e-9;
#: cylinder on a node plane at height .31 2.3e-4, 2.3e-6, 2.2e-7, 3.4e-7. Step 3e-6, the
#: existing 2e-6 budget. JAX and Torch agree to 1.6e-15.
GRADIENT_CASES = (('cylinder', PLANE, None, 'jax'), ('cylinder', PLANE, None, 'torch'),
                  ('box', PLANE, None, 'torch'), ('cylinder', -.25, .31, 'jax'))


@pytest.mark.parametrize('kind,plane,height,backend', GRADIENT_CASES)
def test_a_pillar_on_a_substrate_differentiates_its_orders(kind, plane, height, backend):
    """Transmitted and reflected zero orders of the metasurface cell under a normal s
    PlaneWave, in the pillar's radius (or x and y sizes), height and centre x (or height),
    against the concrete solver as a user runs it: each perturbed geometry planned and
    classified on its own."""
    point = np.array(POINT if kind == 'cylinder' else BOX_POINT)
    if height is not None:
        point[1] = height
    p = quiet(fd.plan, scene(point, kind=kind, plane=plane))
    h = 3e-6

    def values(v):
        return np.array([float(x) for x in quiet(objectives, v, 'numpy', None, kind=kind,
                                                 plane=plane)])
    expected = np.array([(values(point + d) - values(point - d)) / (2 * h)
                         for d in np.eye(3) * h]).T
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp

        def fn(v):
            return jnp.stack(quiet(objectives, v, 'jax', p, kind=kind, plane=plane))
        got = np.asarray(quiet(jax.jacrev(fn), jnp.asarray(point)))
    else:
        import torch
        v = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        out = quiet(objectives, v, 'torch', p, kind=kind, plane=plane)
        assert all(o.requires_grad for o in out), 'the junction lost its geometry graph'
        got = np.array([torch.autograd.grad(o, v, retain_graph=True)[0].numpy() for o in out])
    assert np.all(expected != 0)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.)


def _variant(substrate=None, rule='split', extra=(), kind='cylinder'):
    """The probe's cell without sources or monitors, for planning-only refusals."""
    base = scene(kind=kind).replace(sources=(), monitors=(), run=fd.Run(steps=1))
    structures = list(base.structures)
    if substrate is not None:
        structures[0] = structures[0].replace(shape=substrate)
    structures[1] = structures[1].replace(junction=rule)
    return base.replace(structures=tuple(structures) + tuple(extra))


def _traced(concrete):
    import torch
    s = concrete.structures[1]
    radius = torch.tensor(s.shape.radius, dtype=torch.float64, requires_grad=True)
    return concrete.replace(structures=(concrete.structures[0], s.replace(
        shape=s.shape.replace(radius=radius))) + concrete.structures[2:])


def _admitted(traced, p):
    try:
        return planning_scene(traced, p)
    except ValueError as error:
        raise AssertionError(f'refused a certified traced junction: {error}') from error


def test_the_junction_certificate_covers_the_whole_range():
    """The face must cover the reserved halo and rows with every voxel's half cell, and its far
    face lie more than a cell beyond them; the rule must be 'split'; nothing else may reach
    the traced halo. Each case plans concretely at the seed, so ``classify`` admits it there:
    the refusal is about the rest of the range. The halo and rows span z cells 13..17 (lo 13),
    so a far face at cell 12.5 is refused and one at 11.9 admitted."""
    wide = _variant()
    _admitted(_traced(wide), quiet(fd.plan, wide))
    narrow = _variant(fd.Box(size=(.4, .4, .3), center=(0., 0., PLANE - .15)))
    with pytest.raises(ValueError, match=r"reach past that face's edge"):
        planning_scene(_traced(narrow), quiet(fd.plan, narrow))
    for cells, refused in ((12.5, True), (11.9, False)):
        thickness = (PLANE + 1. - cells * .05)
        thin = _variant(fd.Box(size=(10., 10., thickness),
                               center=(0., 0., PLANE - thickness / 2)))
        q = quiet(fd.plan, thin)
        if refused:
            with pytest.raises(ValueError, match='Thicken the substrate'):
                planning_scene(_traced(thin), q)
        else:
            _admitted(_traced(thin), q)
    stair = _variant(rule='staircase')
    with pytest.raises(ValueError, match=r"takes 'split' only"):
        planning_scene(_traced(stair), quiet(fd.plan, stair))
    corner = fd.Structure(fd.Region(size=(.05, .05, .1), center=(-.225, -.225, PLANE + .2)),
                          fd.Material(eps=1.5))
    crowded = _variant(extra=(corner,))
    with pytest.raises(ValueError, match=r'structures\[2\].*reaches the interpolation halo'):
        planning_scene(_traced(crowded), quiet(fd.plan, crowded))


def pedestal(size=.4):
    """A ranged pedestal Box (the face) carrying a concrete Cylinder standing on its top."""
    face = fd.Box(size=(size, size, .3), center=(0., 0., -.15),
                  range=((-.31, .31), (-.31, .31), (-.31, .01)))
    pillar = fd.Cylinder(radius=.12, height=.3, center=(0., 0., .15))
    return fd.Scene(domain=fd.Domain(size=(1.6,) * 3), resolution=fd.Resolution(dx=.05),
                    boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
                    run=fd.Run(steps=1),
                    structures=(fd.Structure(face, fd.Material(eps=2.25)),
                                fd.Structure(pillar, fd.Material(eps=2.25))))


def test_a_traced_face_is_refused():
    import torch
    p = quiet(fd.plan, pedestal())
    assert len(p.junctions) == 1
    traced = pedestal(torch.tensor(.4, dtype=torch.float64, requires_grad=True))
    with pytest.raises(ValueError, match='differentiated through its standing body only'):
        planning_scene(traced, p)


def test_the_base_must_stay_on_the_face():
    """The flush guard: a centre moved off ``plane + height/2`` by 1e-6 marks the run unstable
    (NaN coefficients); by 1e-12, inside the flush test's own tolerance, it does not."""
    import torch
    p = quiet(fd.plan, scene(steps=4))
    for shift, stable in ((0., True), (1e-12, True), (1e-6, False)):
        v = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in POINT]
        current = scene(v, steps=4)
        s = current.structures[1]
        moved = s.shape.replace(center=(*s.shape.center[:2], s.shape.center[2] + shift))
        params = quiet(fd.lower, current.replace(structures=(current.structures[0],
                                                             s.replace(shape=moved))), p, 'torch')
        assert bool(np.isfinite(float(params['stable']))) is stable, shift


def test_jitted_lowering_follows_the_rim_without_recompiling():
    """``jax.jit(fd.lower)`` at two geometries: one compile, and each answer is the concrete
    scene's at that geometry."""
    runtime('jax')
    import jax
    import jax.numpy as jnp
    p = quiet(fd.plan, scene(steps=4))

    def lower(v):
        params = quiet(fd.lower, scene(v, steps=4), p, 'jax')
        return params['stable'], params['be']
    compiled = jax.jit(lower)
    for v in POINTS['cylinder'][:2]:
        try:
            stable, be = compiled(jnp.asarray(v))
        except Exception as error:
            raise AssertionError(f'jitted lowering failed: {error!r}') from error
        concrete = scene(v, steps=4)
        want = quiet(fd.lower, concrete, quiet(fd.plan, concrete))['be']
        assert float(stable) == 1.
        np.testing.assert_allclose(np.asarray(be), want, rtol=0,
                                   atol=128 * np.finfo(float).eps * np.max(abs(want)))
    assert compiled._cache_size() == 1
