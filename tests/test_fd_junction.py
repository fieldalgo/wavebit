"""Flush interfaces: independent laminate, scope, static plans, and live derivatives."""
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._lower import coefficients, constitutive, junction
from wavebit.fd._runtime import runtime
from fd_helpers import to_numpy


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def scene_of(kind='cylinder', zeta=.25, rule='split', reverse=False, axis=2, side=1,
             eps=(2.25, 4., 1.), gap=0., steps=2):
    dx = .05
    plane = zeta*dx
    center = [.017, -.013, .007]
    center[axis] = plane + side*(.2 + gap*dx)
    size = [.28, .32, .36]
    size[axis] = .4
    pillar = (fd.Cylinder(radius=.15, height=.4, axis='xyz'[axis], center=tuple(center))
              if kind == 'cylinder' else fd.Box(size=tuple(size), center=tuple(center)))
    substrate_size = [2., 2., 2.]
    substrate_size[axis] = 1.2 + side*plane
    substrate_center = [0., 0., 0.]
    substrate_center[axis] = (plane-side*1.2)/2
    pair = (fd.Structure(fd.Box(size=tuple(substrate_size), center=tuple(substrate_center)), eps[0]),
            fd.Structure(pillar, eps[1], junction=rule))
    faces = dict(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic())
    faces['xyz'[axis]] = fd.PEC()
    size = [.8, .8, .8]
    size[axis] = 1.2
    return fd.Scene(domain=fd.Domain(size=tuple(size)), boundaries=fd.Boundaries(**faces),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                    background=eps[2], structures=pair[::-1] if reverse else pair)


def arrays(scene):
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    nodes = {}
    cross = constitutive.lower(scene, plan, rt, nodes=nodes)
    return plan, coefficients.build(scene, plan, rt), cross, nodes


@pytest.mark.parametrize('value', ('splt', 'blend', 'meep', None, 1, 'Split'))
def test_rule_names_are_explicit(value):
    with pytest.raises(ValueError, match='split|staircase'):
        fd.Structure(fd.Cylinder(radius=.15, height=.4), 4., junction=value)


@pytest.mark.parametrize('shape', (fd.Slab(z=(-.2, 0)), fd.Region(size=(.2,)*3),
                                   fd.Sphere(radius=.15)))
def test_unread_rules_on_ineligible_shapes_are_refused(shape):
    with pytest.raises(ValueError, match='no junction rule reads'):
        fd.Structure(shape, 4., junction='staircase')


def test_unread_rule_on_a_lifted_pillar_and_on_the_substrate_is_refused():
    with pytest.raises(ValueError, match='stands on no smoothed face'):
        quiet(fd.plan, scene_of(gap=2., rule='staircase'))
    scene = scene_of()
    pair = scene.structures
    with pytest.raises(ValueError, match=r'set it on structures\[1\]'):
        quiet(fd.plan, scene.replace(structures=(pair[0].replace(junction='staircase'), pair[1])))


def test_refuse_retains_the_shared_surface_refusal():
    with pytest.raises(ValueError, match="same voxel:.*asked for junction='refuse'"):
        quiet(fd.plan, scene_of(rule='refuse'))


def oracle(scene, record, points, node=False):
    """Independent transverse geometry and a 2x2 matrix inverse, not the closed form."""
    from wavebit.fd._shapes._exact import clipped_disc_area, clipped_polygon_area, polygon_edge_moments
    from wavebit.fd._shapes._cut import _parts, _LIVE_AREA
    shape = scene.structures[record.standing].shape
    axis, (u, v) = record.axis, junction.TRANSVERSE[record.axis]
    dx = scene.resolution.dx
    alpha = np.clip(.5-record.side*(points[..., axis]-record.plane)/dx, 0, 1)
    box = (points[..., u]-dx/2, points[..., u]+dx/2,
           points[..., v]-dx/2, points[..., v]+dx/2)
    es, ep, eg = (scene.structures[record.face].material.eps[0],
                  scene.structures[record.standing].material.eps[0], scene.background.eps[0])
    if isinstance(shape, fd.Cylinder):
        fu = clipped_disc_area(shape.center[u], shape.center[v], shape.radius, *box, np)/dx**2
        offsets = points[..., [u, v]] - np.asarray(shape.center)[[u, v]]
        distance = np.linalg.norm(offsets, axis=-1)
        normal = offsets/np.where(distance > 0, distance, 1)[..., None]
        normal = np.where((distance > 0)[..., None], normal, np.array([1., 0.]))
        plane = np.concatenate((normal, np.zeros(normal.shape[:-1]+(1,))), axis=-1)
        _, area, _ = _parts(distance-shape.radius, plane, dx)
        live = ((fu > 1e-9) & (fu < 1-1e-9)) if node else area > _LIVE_AREA*dx**2
        weight = np.einsum('...i,...j->...ij', normal, normal)*live[..., None, None]
    else:
        x, y = shape.center[u], shape.center[v]
        a, b = shape.size[u]/2, shape.size[v]/2
        vertices = np.array([[x-a,y-b], [x+a,y-b], [x+a,y+b], [x-a,y+b]])
        fu = clipped_polygon_area(vertices, *box, np, convex=True)/dx**2
        length, muu, mvv = polygon_edge_moments(vertices, *box, np)
        live = length*dx > _LIVE_AREA*dx**2
        safe = np.where(live, length, 1)
        weight = np.zeros(fu.shape+(2, 2))
        weight[..., 0, 0] = np.where(live, muu/safe, 0)
        weight[..., 1, 1] = np.where(live, mvv/safe, 0)
    up = fu*ep + (1-fu)*eg
    eta = fu/ep + (1-fu)/eg
    tensor_up = (np.eye(2)-weight)/up[..., None, None] + eta[..., None, None]*weight
    tangential = np.linalg.inv((1-alpha)[..., None, None]*np.linalg.inv(tensor_up)
                               + (alpha*es)[..., None, None]*np.eye(2))
    diagonal = np.empty((3,)+fu.shape)
    diagonal[u], diagonal[v] = tangential[..., 0, 0], tangential[..., 1, 1]
    diagonal[axis] = (1-alpha)/up + alpha/es
    cross = np.zeros_like(diagonal)
    cross[junction.PAIRS.index(tuple(sorted((u,v))))] = tangential[..., 0, 1]
    return diagonal, cross, 1/((1-alpha)*up+alpha*es)


@pytest.mark.parametrize('kind', ('cylinder', 'box'))
@pytest.mark.parametrize('zeta', (0., .25, .5, .75))
@pytest.mark.parametrize('axis,side', ((2,1), (2,-1), (0,1), (1,1)))
def test_split_matches_independent_laminate_and_preserves_unwritten_points(kind, zeta, axis, side):
    scene = scene_of(kind, zeta, axis=axis, side=side)
    plan, params, cross, nodes = arrays(scene)
    assert len(plan.junctions) == 1
    record = plan.junctions[0]
    baseline = plan.with_static(junctions=())
    old_params = coefficients.build(scene, baseline, runtime('numpy'))
    old_nodes = {}
    old_cross = constitutive.lower(scene, baseline, runtime('numpy'), nodes=old_nodes)
    full = np.zeros((3, plan.nz, plan.ny, plan.nx), bool)
    full[(slice(None),)+junction._view(record.rows)] = record.write[:3]
    for c in range(3):
        points = junction._points(plan.grid, record.rows, 'E', c)
        expected, _, _ = oracle(scene, record, points)
        got = params['be'][c][junction._view(record.rows)]/(plan.dt/plan.dx)
        np.testing.assert_allclose(got[record.write[c]], expected[c][record.write[c]],
                                   rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(params['be'][~full], old_params['be'][~full])
    for term in plan.constitutive:
        points = constitutive.common_points(plan.grid, tuple(reversed(term.box)), 'E')
        expected, off, inverse = oracle(scene, record, points, node=True)
        mask = np.zeros(term.region.shape, bool)
        cut = constitutive._intersect(tuple(reversed(term.box)), record.rows)
        mask[junction._view(cut, tuple(reversed(term.box)))] = record.write[
            (3,)+junction._view(cut, record.rows)]
        tensor, inv = nodes[term.name]
        for got, want, old in ((tensor, expected, old_nodes[term.name][0]),
                                (cross[term.name], off, old_cross[term.name])):
            np.testing.assert_allclose(got[:,mask], want[:,mask], rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(got[:,~mask], old[:,~mask])
        np.testing.assert_allclose(inv[mask], inverse[mask], rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(inv[~mask], old_nodes[term.name][1][~mask])


@pytest.mark.parametrize('rule', ('split', 'staircase'))
def test_pair_listing_order_does_not_change_the_interface(rule):
    first = arrays(scene_of(rule=rule))
    second = arrays(scene_of(rule=rule, reverse=True))
    record = first[0].junctions[0]
    mask = np.zeros(first[1]['be'].shape, bool)
    mask[(slice(None),)+junction._view(record.rows)] = record.write[:3]
    np.testing.assert_array_equal(first[1]['be'][mask], second[1]['be'][mask])
    np.testing.assert_allclose(first[1]['be'], second[1]['be'], rtol=1e-14, atol=0)
    for name in first[2]:
        np.testing.assert_allclose(first[2][name], second[2][name], rtol=0, atol=1e-14)
        for a, b in zip(first[3][name], second[3][name]):
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


def test_staircase_uses_the_medium_at_each_point():
    scene = scene_of(rule='staircase')
    plan, params, cross, nodes = arrays(scene)
    record = plan.junctions[0]
    for c in range(3):
        points = junction._points(plan.grid, record.rows, 'E', c)
        own = np.where(points[...,2] < record.plane-1e-9*plan.dx, 2.25,
                       np.where(scene.structures[1].shape.body().surface(points)[0] <= 0, 4., 1.))
        got = params['be'][c][junction._view(record.rows)]/(plan.dt/plan.dx)
        mask = record.inside[c]
        np.testing.assert_allclose(got[mask], 1/own[mask], rtol=0, atol=1e-15)


@pytest.mark.parametrize('change', ('rule', 'pillar', 'third', 'missing'))
def test_plan_mismatch_is_refused(change):
    scene = scene_of()
    plan = quiet(fd.plan, scene)
    if change == 'rule':
        altered = scene_of(rule='staircase')
    elif change == 'pillar':
        altered = scene_of(gap=2)
    elif change == 'third':
        extra = fd.Structure(fd.Region(size=(.1,)*3), 3.)
        altered = scene.replace(structures=scene.structures+(extra,))
    else:
        plan = quiet(fd.plan, scene.replace(structures=()))
        altered = scene
    with pytest.raises(ValueError, match='JunctionMismatchRefused'):
        fd.lower(altered, plan)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
def test_material_backends_agree(backend):
    scene = scene_of()
    plan = quiet(fd.plan, scene)
    expected = fd.lower(scene, plan, 'numpy')
    actual = fd.lower(scene, plan, backend)
    np.testing.assert_allclose(to_numpy(actual['be']), expected['be'], rtol=1e-10, atol=1e-13)
    for name, value in actual['constitutive'].items():
        np.testing.assert_allclose(to_numpy(value), expected['constitutive'][name],
                                   rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize('backend', ('jax', 'torch'))
@pytest.mark.parametrize('medium', (0, 1, 2))
def test_material_gradients_follow_the_same_split(backend, medium):
    rt = runtime(backend)
    plan = quiet(fd.plan, scene_of())
    def loss(eps, backend=backend):
        values = [2.25, 4., 1.]
        values[medium] = eps
        return fd.lower(scene_of(eps=values), plan, backend)['be'].sum()
    value = [2.25, 4., 1.][medium]
    h = 1e-6*value
    finite = (float(loss(value+h, 'numpy'))-float(loss(value-h, 'numpy')))/(2*h)
    if backend == 'jax':
        import jax
        measured = jax.jit(jax.grad(loss))(rt.array(value, 'float64'))
    else:
        parameter = rt.array(value, 'float64').requires_grad_()
        loss(parameter).backward()
        measured = parameter.grad
    np.testing.assert_allclose(to_numpy(measured), finite, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('gap', (-.3, .3))
def test_a_nonflush_pair_remains_refused(gap):
    with pytest.raises(ValueError, match='does not stand flush'):
        quiet(fd.plan, scene_of(gap=gap))


def test_magnetic_and_late_overlays_remain_refused():
    scene = scene_of()
    pair = scene.structures
    magnetic = pair[1].replace(material=fd.Material(eps=4., mu=1.1))
    with pytest.raises(ValueError, match='magnetic medium'):
        quiet(fd.plan, scene.replace(structures=(pair[0], magnetic)))
    overlay = fd.Structure(fd.Region(size=(.3, .3, .3)), 3.)
    with pytest.raises(ValueError, match='is listed after'):
        quiet(fd.plan, scene.replace(structures=pair+(overlay,)))


def test_skeleton_records_the_rule_and_execution_drops_masks():
    from wavebit.fd._run import _program
    from pathlib import Path
    import subprocess
    import sys
    split = quiet(fd.plan, scene_of())
    staircase = quiet(fd.plan, scene_of(rule='staircase'))
    assert split.skeleton != staircase.skeleton
    program = _program(split)
    assert program == split and hash(program) == hash(split)
    assert all(r.write is r.inside is r.owner is None for r in program.junctions)
    assert split.junctions[0].write is not None
    command = ("import sys; sys.path.insert(0, 'tests'); "
               "from test_fd_junction import fd, quiet, scene_of; "
               "print(quiet(fd.plan, scene_of()).skeleton)")
    for _ in range(2):
        result = subprocess.run([sys.executable, '-c', command],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == split.skeleton


def test_box_node_limit_preserves_the_existing_nearest_wall_tensor():
    from wavebit.fd._shapes import occupancy
    dx = .05
    box = fd.Box(size=(.28, .32, .4), center=(.017, -.013, .2)).body()
    sub = fd.Box(size=(2., 2., 1.2), center=(0., 0., -.6)).body()
    points = np.array([[x,y,.05] for x in (-.1,.15) for y in (-.15,.15)])
    fill = occupancy(box, points, dx)[0]
    normal = box.surface(points)[1]
    mean = 4*fill+1-fill
    inverse_mean = fill/4+1-fill
    expected = np.stack([(1-normal[...,a]**2)/mean + normal[...,a]**2*inverse_mean
                          for a in range(3)])
    got, cross, floor = junction.values(sub, box, 2, points, dx, 2.25, 4., 1., np, node=True)
    # The exact-edge-moment node mutant misses four corners by .1005220005782;
    # the correct limit differs by at most 1.67e-16 (2026-09-28 review probe).
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)
    assert np.count_nonzero(cross) == 0
    np.testing.assert_allclose(floor, 1/mean, rtol=0, atol=1e-14)


def test_report_counts_the_written_electric_families():
    scene = scene_of()
    plan = quiet(fd.plan, scene)
    report = quiet(fd.check, scene)
    record = plan.junctions[0]
    count = int(record.inside.sum())
    assert report.junctions[0][6] == count
    assert f'split, {count} voxels' in str(report)
    assert 'junction on structures[0]' in str(report)
    assert any('unresolved order' in str(w) for w in report.warnings)


def test_no_junction_never_classifies_or_changes_the_skeleton(monkeypatch):
    from wavebit.fd._lower.plan import skeleton_of
    import dataclasses
    scene = scene_of(gap=2)
    def forbidden(*args, **kwargs):
        raise AssertionError('classification on a scene without shared surface voxels')
    monkeypatch.setattr(junction, 'classify', forbidden)
    plan, params, cross, nodes = arrays(scene)
    assert plan.junctions == ()
    original_static = {f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)
                       if f.name not in ('skeleton', 'junctions')}
    assert plan.skeleton == skeleton_of(**original_static)
    assert not quiet(fd.check, scene).junctions


@pytest.mark.parametrize('case', ('sphere', 'turned', 'thin', 'edge', 'snapped', 'stencil'))
def test_unvalidated_geometries_remain_refused(case):
    scene = scene_of()
    sub, pillar = scene.structures
    if case == 'sphere':
        pillar = fd.Structure(fd.Sphere(radius=.15, center=(0,0,.15+.0125)), 4.)
        phrase = 'sphere touches a face'
    elif case == 'turned':
        pillar = fd.Structure(fd.Box(size=(.25,.25,.4), center=(0,0,.2125), angle=30), 4.)
        phrase = 'turned fd.Box'
    elif case == 'thin':
        sub = fd.Structure(fd.Box(size=(2,2,.035), center=(0,0,.0125-.0175)), 2.25)
        phrase = 'ends within a voxel'
    elif case == 'edge':
        sub = fd.Structure(fd.Box(size=(.3,.3,1.2125), center=(0,0,-.59375)), 2.25)
        phrase = 'rim meets the edge'
    elif case == 'snapped':
        sub = fd.Structure(fd.Slab(z=(-1.2, 0)), 2.25)
        phrase = 'cell-snapped'
    else:
        pillar = pillar.replace(discretisation='stencil')
        phrase = 'Stencil.*Refused'
    with pytest.raises(ValueError, match=phrase):
        quiet(fd.plan, scene.replace(structures=(sub,pillar)))


def test_high_contrast_junction_exercises_the_lift_and_stays_bounded():
    from test_fd_constitutive import seeded_state
    from wavebit.fd._kernel.step import step
    scene = scene_of(eps=(2.1, 100., 1.), steps=300)
    plan = quiet(fd.plan, scene)
    rt = runtime('numpy')
    _, lifted = constitutive.lifted_on_host(scene, plan)
    assert sum(value.moved for value in lifted.values()) > 0
    nodes = {}
    cross = constitutive.lower(scene, plan, rt, nodes=nodes)
    for term in plan.constitutive:
        diagonal, inverse = nodes[term.name]
        tensor = np.zeros(inverse.shape+(3,3))
        for a in range(3):
            tensor[...,a,a] = diagonal[a]
        for k,(a,b) in enumerate(junction.PAIRS):
            tensor[...,a,b] = tensor[...,b,a] = cross[term.name][k]
        record = plan.junctions[0]
        box = tuple(reversed(term.box))
        cut = constitutive._intersect(box, record.rows)
        mask = np.zeros(inverse.shape, bool)
        mask[junction._view(cut, box)] = record.write[(3,)+junction._view(cut, record.rows)]
        # Check the new junction floor on its write set. A first probe over the
        # entire term also tested unchanged single-body nodes: eleven had a
        # relative eigensolver deficit up to 1.5233e-13; the junction's was
        # 2.22e-16. The 300-step check below still exercises the whole operator.
        assert np.all(np.linalg.eigvalsh(tensor)[...,0][mask] >= inverse[mask]*(1-1e-13))
    params = fd.lower(scene, plan)
    state = seeded_state(plan, params, rt)
    norm = lambda state: max(np.linalg.norm(state['E']), np.linalg.norm(state['H']))
    initial = norm(state)
    for n in range(300):
        state = step(plan, params, state, n, rt)
    # Measured 2026-09-28: 101/294 stars moved, largest fraction .7045792972;
    # seeded (224) norm after 300 steps is .9740758014 of its initial value.
    # Reuse the pre-existing sphere stability gate's factor 10, not a fitted gate.
    assert np.isfinite(norm(state)) and norm(state)/initial < 10


@pytest.mark.parametrize('backend', ('jax', 'torch'))
def test_internal_radius_derivative_keeps_geometry_on_backend_arrays(backend):
    from wavebit.fd._shapes import Box, Cylinder
    rt = runtime(backend)
    face = Box(size=(2.,2.,1.2), center=(0.,0.,-.6))
    def value(radius, runtime_=rt):
        body = Cylinder(radius=radius, height=.4, center=(0.,0.,.2))
        p = runtime_.array([[.145,.055,.0125]], 'float64')
        diagonal, cross, inverse = junction.values(face, body, 2, p, .05,
                                                   2.25, 4., 1., runtime_.xp)
        return diagonal.sum()+2*cross.sum()+inverse.sum()
    h = .15e-6
    finite = (value(.15+h, runtime('numpy'))-value(.15-h, runtime('numpy')))/(2*h)
    if backend == 'jax':
        import jax
        got = jax.jit(jax.grad(value))(rt.array(.15, 'float64'))
    else:
        radius = rt.array(.15, 'float64').requires_grad_()
        value(radius).backward()
        got = radius.grad
    assert abs(float(finite)) > 1
    np.testing.assert_allclose(to_numpy(got), finite, rtol=1e-7, atol=1e-8)

@pytest.mark.parametrize('backend', ('jax', 'torch'))
@pytest.mark.parametrize('medium', (0, 1, 2))
def test_material_derivatives_survive_forty_coupled_steps(backend, medium):
    from test_fd_constitutive import seeded_state
    from wavebit.fd._kernel.step import step
    scene = scene_of(steps=40)
    plan = quiet(fd.plan, scene)
    def evolve(eps, name):
        rt = runtime(name)
        values = [2.25, 4., 1.]
        values[medium] = eps
        params = fd.lower(scene_of(eps=values, steps=40), plan, rt)
        state = seeded_state(plan, params, rt)
        return rt.scan(lambda s, n: step(plan, params, s, n, rt), state, 40)
    def loss(eps, name=backend):
        state = evolve(eps, name)
        return (state['E']**2 + .3*state['H']**2).sum()/plan.cells
    value = [2.25, 4., 1.][medium]
    expected = evolve(value, 'numpy')
    got = evolve(value, backend)
    for field in ('E', 'H'):
        np.testing.assert_allclose(to_numpy(got[field]), expected[field], rtol=1e-10, atol=1e-10)
    h = value*1e-6  # Registered material finite-difference step, not fitted to this run.
    finite = (loss(value+h, 'numpy')-loss(value-h, 'numpy'))/(2*h)
    rt = runtime(backend)
    if backend == 'jax':
        import jax
        actual = jax.jit(jax.grad(loss))(rt.array(value, 'float64'))
    else:
        parameter = rt.array(value, 'float64').requires_grad_()
        loss(parameter).backward()
        actual = parameter.grad
    np.testing.assert_allclose(to_numpy(actual), finite, rtol=1e-7, atol=1e-8)
