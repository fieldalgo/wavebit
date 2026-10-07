"""Independent commuting-Absorber contracts; unexecuted draft.

The numerical budgets remain unaccepted until the fixed algebra/field calibration
is disclosed. The dense oracle below never calls production cross_action.
"""
import dataclasses
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.constitutive import ConstitutiveTerm
from wavebit.fd._kernel.constitutive_absorber import (
    AbsorberConstitutiveEnergy, AbsorberConstitutiveTerm,
)
from wavebit.fd._kernel.masks import Region
from wavebit.fd._kernel.observe import InteriorEnergy
from wavebit.fd._kernel.poles import PoleTerm
from wavebit.fd._kernel.step import initial_state, step
from wavebit.fd._lower.constitutive_absorber import (
    _scope, certify_links, physical_diagonal, lifted_delta, refuse_mismatch, validate_links, validate_runtime,
)
from wavebit.fd._runtime import runtime

BACKENDS = ('numpy', 'jax', 'torch')
ACCEPTED_BUDGETS = {'algebra': 1.1368683772161603e-13, 'step': 7.275957614183426e-12, 'physical_alignment': 2.842170943040401e-14}  # only a disclosed calibration acceptance may replace this


def near(actual, expected, key='algebra'):
    assert ACCEPTED_BUDGETS is not None, 'disclose calibration and accept budgets first'
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    assert float(np.max(np.abs(actual-expected), initial=0.0))/scale <= ACCEPTED_BUDGETS[key]


def quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call(*args, **kwargs)


def dense(diagonal, cross, field='E'):
    shape = cross.shape
    out = np.diag(np.broadcast_to(diagonal, shape).ravel())
    sign = -1 if field == 'E' else 1
    for row, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        for node in np.ndindex(shape[1:]):
            for di in (0, sign):
                for dj in (0, sign):
                    a, b = list(node), list(node)
                    a[2-i] += di
                    b[2-j] += dj
                    if any(v < 0 or v >= n for v, n in zip(a, shape[1:])):
                        continue
                    if any(v < 0 or v >= n for v, n in zip(b, shape[1:])):
                        continue
                    u = np.ravel_multi_index((i, *a), shape)
                    v = np.ravel_multi_index((j, *b), shape)
                    out[u, v] += cross[(row, *node)]/4
                    out[v, u] += cross[(row, *node)]/4
    return out


def local_case(axis, complex_):
    rng = np.random.default_rng(629)
    shape = (3, 2, 2, 2)
    pair = tuple(i for i in range(3) if i != axis)
    row = ((0, 1), (0, 2), (1, 2)).index(pair)
    cross = np.zeros(shape)
    cross[row] = rng.uniform(-.08, .08, shape[1:])
    diagonal = rng.uniform(.7, 1.2, shape)
    idx = np.indices(shape[1:])[2-axis]
    rate = np.broadcast_to(.2 + idx, shape).copy()
    def vector():
        x = rng.normal(size=shape)
        return x + 1j*rng.normal(size=shape) if complex_ else x
    return shape, cross, diagonal, rate, vector(), vector(), rng


def private_term(shape, dt):
    return AbsorberConstitutiveTerm('constitutive[E0]', 'E',
        Region(z=(0, shape[1]), y=(0, shape[2]), x=(0, shape[3])), dt)


def params_for(rt, term, cross, diagonal, rate, dt, h_rate=None):
    half = dt*rate/2
    drive = 1/(1+half)
    keep = (1-half)/(1+half)
    if h_rate is None:
        h_rate = rate
    hh = dt*h_rate/2
    return {
        'constitutive': {term.name: rt.array(cross)},
        'constitutive_absorber': {term.name: {'drive': rt.array(drive),
            'keep': rt.array(keep), 'rate': rt.array(rate)}},
        'constitutive_absorber_drive_e': rt.array(drive),
        'constitutive_absorber_rate_h': rt.array(h_rate),
        'ae': rt.array(keep), 'be': rt.array(dt*drive*diagonal),
        'ah': rt.array((1-hh)/(1+hh)), 'bh': rt.array(dt/(1+hh)),
        'minus_bh': rt.array(-dt/(1+hh)),
    }


def extrusion(axis=2, family='cylinder', touching=True):
    length = [1.0]*3
    length[axis] = 1.6
    height = 2.4 if touching else .5
    if family == 'cylinder':
        shape = fd.Cylinder(radius=.22, height=height, axis='xyz'[axis])
    else:
        size = [.44, .48, .44]
        size[axis] = height
        shape = fd.Box(size=tuple(size), angle=25, axis='xyz'[axis])
    faces = {a: fd.PEC() for a in 'xyz'}
    faces['xyz'[axis]] = fd.Absorber(cells=2)
    return fd.Scene(domain=fd.Domain(size=tuple(length)), resolution=fd.Resolution(dx=.1),
        boundaries=fd.Boundaries(**faces), run=fd.Run(steps=8),
        structures=(fd.Structure(shape, fd.Material(eps=2.25)),))


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('complex_', (False, True))
def test_damped_flux_and_field_match_independent_dense_operator(backend, axis, complex_):
    rt = runtime(backend, device='cpu')
    shape, cross, diagonal, rate, D, q, _ = local_case(axis, complex_)
    dt = .125
    term = private_term(shape, dt)
    params = params_for(rt, term, cross, diagonal, rate, dt)
    K = dense(diagonal, cross)
    E = (K @ D.ravel()).reshape(shape)
    A, B = (1-dt*rate/2)/(1+dt*rate/2), 1/(1+dt*rate/2)
    expected_D = A*D + dt*B*q
    mem, delta = term.prepare(rt.array(q), rt.array(D), params, rt)
    actual_E = rt.update(rt.array(E.copy()), rt.array(q), params['ae'], params['be']) + delta
    near(rt.numpy(mem), expected_D)
    near(rt.numpy(actual_E), (K @ expected_D.ravel()).reshape(shape))
    assert certify_links(cross, rate, 'E') is True


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('complex_', (False, True))
@pytest.mark.parametrize('damped', (False, True))
@pytest.mark.parametrize('work', (False, True))
def test_staggered_storage_obeys_independent_dissipation_identity(backend, complex_, damped, work):
    rt = runtime(backend, device='cpu')
    shape, cross, diagonal, rate, D, H, rng = local_case(2, complex_)
    if not damped:
        rate = np.zeros_like(rate)
    dt = .025
    term = private_term(shape, dt)
    rh = rng.uniform(.1, 2., shape) if damped else np.zeros(shape)
    params = params_for(rt, term, cross, diagonal, rate, dt, rh)
    K = dense(diagonal, cross)
    Me = np.linalg.inv(K)
    # Independent finite curl/adjoint pair: a tiny exact algebraic instrument.
    C = rng.normal(size=(D.size, D.size)) / 8
    E = K @ D.ravel()
    hp = rt.update(rt.array(H.copy()), rt.array((C @ E).reshape(shape)),
                   params['ah'], params['minus_bh'])
    current = rng.normal(size=shape) if work else np.zeros(shape)
    q = (C.T @ np.asarray(rt.numpy(hp)).ravel()).reshape(shape) + current
    mem, delta = term.prepare(rt.array(q), rt.array(D), params, rt)
    ep = rt.update(rt.array(E.reshape(shape).copy()), rt.array(q), params['ae'], params['be']) + delta
    Ep, Hp = np.asarray(rt.numpy(ep)).ravel(), np.asarray(rt.numpy(hp)).ravel()
    observer = AbsorberConstitutiveEnergy((0,0,0), shape[1:], 1, dt, dt,
                                          constitutive=(term,))
    before = observer.electric(rt.array(E.reshape(shape)), {term.name: rt.array(D)}, params, rt)
    before += observer.magnetic(rt.array(H), hp, None, params, rt)
    def storage(e, h):
        return np.real(np.vdot(e, Me @ e) + np.vdot(h, h) - dt*np.vdot(h, C @ e))
    w0 = storage(E, H.ravel())
    w1 = storage(Ep, Hp)
    near(rt.numpy(before), w0)
    hnext = params['ah']*hp + params['minus_bh']*rt.array((C @ Ep).reshape(shape))
    after = observer.electric(ep,{term.name:mem},params,rt)
    after += observer.magnetic(hp,hnext,None,params,rt)
    near(rt.numpy(after),w1)
    near(rt.numpy(mem),(Me @ Ep).reshape(shape))
    eb, hb = (Ep+E)/2, (Hp+H.ravel())/2
    loss = 2*dt*np.real(np.vdot(eb, Me @ (rate.ravel()*eb)) + np.vdot(hb, rh.ravel()*hb))
    source_work = 2*dt*np.real(np.vdot(eb,current.ravel()))
    near(w1-w0+loss-source_work, 0.)
    assert w0 > 0 and w1 > 0 and loss >= 0


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('complex_', (False, True))
@pytest.mark.parametrize('large_rate', (False, True))
def test_observer_uses_whole_absorber_and_both_magnetic_clocks(backend, complex_, large_rate):
    rt = runtime(backend, device='cpu')
    rng = np.random.default_rng(841)
    shape = (3, 3, 3, 3)
    old = rng.normal(size=shape)
    cE = rng.normal(size=shape)
    if complex_:
        old = old + 1j*rng.normal(size=shape)
        cE = cE + 1j*rng.normal(size=shape)
    dt = .125
    rate = np.full(shape, 64. if large_rate else .75)
    half = dt*rate/2
    params = {'ah': rt.array((1-half)/(1+half)), 'minus_bh': rt.array(-dt/(1+half)),
              'constitutive_absorber_rate_h': rt.array(rate)}
    new = (1-half)/(1+half)*old-dt/(1+half)*cE
    obs = AbsorberConstitutiveEnergy((0,0,0), shape[1:], 1, dt, dt)
    expected = np.real(np.vdot(old, old)-dt*np.vdot(old, cE))
    pre = obs.magnetic(rt.array(old), None, rt.array(cE), params, rt)
    post = obs.magnetic(rt.array(old), rt.array(new), None, params, rt)
    near(rt.numpy(pre), expected)
    near(rt.numpy(post), expected)
    if large_rate:
        assert np.any((1-half)/(1+half) < 0)


@pytest.mark.parametrize('field', ('E', 'H'))
@pytest.mark.parametrize('pair', ((0,1), (0,2), (1,2)))
@pytest.mark.parametrize('link', ('same', 'i-neighbour', 'j-neighbour', 'both', 'ghost'))
def test_every_live_interpolation_link_is_certified(field, pair, link):
    shape = (3, 3, 3, 3)
    cross = np.zeros(shape)
    rates = np.ones(shape)
    node = [1,1,1]
    i, j = pair
    shift = -1 if field == 'E' else 1
    row = ((0,1), (0,2), (1,2)).index(pair)
    if link == 'ghost':
        node[2-i] = 0 if field == 'E' else 2
        node[2-j] = 0 if field == 'E' else 2
        cross[(row, *node)] = .25
        assert certify_links(cross, rates, field) is True
        return
    cross[(row, *node)] = .25
    target = node.copy()
    component = i if link != 'j-neighbour' else j
    if link in ('i-neighbour', 'both'):
        target[2-i] += shift
    elif link == 'j-neighbour':
        target[2-j] += shift
    rates[(component, *target)] = 2.
    if link == 'both':
        target_j = node.copy()
        target_j[2-j] += shift
        rates[(j, *target_j)] = 3.
    K = dense(np.ones(shape), cross, field)
    commutator = K*rates.ravel()[None,:]-rates.ravel()[:,None]*K
    assert np.max(np.abs(commutator)) > 0
    with pytest.raises(ValueError, match='ConstitutiveAbsorberCommutation'):
        certify_links(cross, rates, field)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('axis', (0,1,2))
def test_lift_and_cfl_read_physical_not_damped_diagonal(backend, axis):
    rt = runtime(backend, device='cpu')
    scene = extrusion(axis)
    plan = quiet(fd.plan, scene)
    params = quiet(fd.lower, scene, plan, rt)
    plain_faces = {a: fd.PEC() for a in 'xyz'}
    plain = scene.replace(boundaries=fd.Boundaries(**plain_faces))
    p0 = quiet(fd.plan, plain)
    q0 = quiet(fd.lower, plain, p0, rt)
    assert plan.shape == p0.shape
    assert all(type(t) is AbsorberConstitutiveTerm for t in plan.constitutive)
    for term, oldterm in zip(plan.constitutive, p0.constitutive):
        new_diag = rt.numpy(physical_diagonal(term, params, rt))
        old_diag = rt.numpy(physical_diagonal(oldterm, q0, rt))
        near(new_diag, old_diag)
        drive = np.asarray(rt.numpy(params['constitutive_absorber'][term.name]['drive']))
        assert np.min(drive) < 1
        increment = np.full((3,)+term.region.shape,.125)
        near(rt.numpy(lifted_delta(term,params,rt.array(increment))),
             term.ratio*drive*increment)
    assert type(plan.energy) is AbsorberConstitutiveEnergy


@pytest.mark.parametrize('change', ('missing', 'ordinary', 'signature', 'shape', 'material',
                                    'boundary', 'magnetic-source', 'initial', 'decay', 'foreign',
                                    'added-source', 'missing-source', 'foreign-memory'))
def test_prepared_plan_cannot_bypass_new_scope(change):
    scene = extrusion()
    plan = quiet(fd.plan, scene)
    term = plan.constitutive[0]
    candidate, held = scene, plan
    if change == 'added-source':
        candidate = scene.replace(sources=(fd.Dipole('d',center=(0,0,0),component='Ey',
            band=fd.Band(wavelength=2.,df=.03)),))
    elif change == 'missing-source':
        driven = scene.replace(sources=(fd.Dipole('d',center=(0,0,0),component='Ey',
            band=fd.Band(wavelength=2.,df=.03)),))
        held = quiet(fd.plan, driven)
    elif change == 'foreign-memory':
        foreign = PoleTerm('foreign-pole',term.region,(),1)
        held = plan.with_static(terms=tuple(plan.terms)+(foreign,))
    elif change == 'foreign':
        foreign = ConstitutiveTerm('foreign',term.field,term.region,term.ratio,term.masks)
        held = plan.with_static(terms=tuple(plan.terms)+(foreign,))
    elif change in ('missing', 'ordinary', 'signature'):
        terms = tuple(t for t in plan.terms if t is not term)
        if change == 'ordinary':
            terms += (ConstitutiveTerm(term.name, term.field, term.region, term.ratio, term.masks),)
        elif change == 'signature':
            terms += (dataclasses.replace(term, signature=('foreign',)),)
        held = plan.with_static(terms=terms)
    elif change == 'shape':
        body = scene.structures[0].shape.replace(radius=.16)
        candidate = scene.replace(structures=(scene.structures[0].replace(shape=body),))
    elif change == 'material':
        candidate = scene.replace(structures=(scene.structures[0].replace(material=fd.Material(eps=3.)),))
    elif change == 'boundary':
        candidate = scene.replace(boundaries=fd.Boundaries(x=fd.PEC(),y=fd.PEC(),z=fd.Absorber(cells=3)))
    elif change == 'magnetic-source':
        candidate = scene.replace(sources=(fd.Dipole('m',center=(0,0,0),component='Hx',band=fd.Band(wavelength=2.,df=.03),kind='magnetic'),))
    elif change == 'initial':
        candidate = scene.replace(initial=fd.InitialField(E=(None,lambda x,y,z: x*0+1,None)))
    else:
        candidate = scene.replace(run=fd.Run(steps=8,decay=fd.Decay(threshold=1e-5,window=1.)))
    with pytest.raises(ValueError, match='ConstitutiveAbsorber'):
        refuse_mismatch(candidate, held)
    with pytest.raises(ValueError, match='ConstitutiveAbsorber|Decay'):
        quiet(fd.lower, candidate, held, 'numpy')


@pytest.mark.parametrize('kind', ('magnetic-material', 'conductor', 'dispersion', 'anisotropic',
                                  'pml', 'two-axes', 'initial', 'decay', 'magnetic-source'))
def test_scope_refuses_unproved_material_source_and_boundary_combinations(kind):
    scene = extrusion()
    plan = quiet(fd.plan, scene)
    material = {'magnetic-material': fd.Material(eps=2.,mu=1.1),
                'conductor': fd.Conductor(eps=2.,sigma=.1),
                'dispersion': fd.Dispersive(eps_inf=2.,poles=(fd.Debye(delta_eps=1.,tau=1.),)),
                'anisotropic': fd.Material(eps=(2.,3.,2.))}.get(kind)
    if material is not None:
        scene = scene.replace(structures=(scene.structures[0].replace(material=material),))
    elif kind == 'pml':
        scene = scene.replace(boundaries=fd.Boundaries(x=fd.PML(cells=2,allow_thin=True),y=fd.PEC(),z=fd.Absorber(cells=2)))
    elif kind == 'two-axes':
        scene = scene.replace(boundaries=fd.Boundaries(x=fd.Absorber(cells=2),y=fd.PEC(),z=fd.Absorber(cells=2)))
    elif kind == 'initial':
        scene = scene.replace(initial=fd.InitialField(E=(None,lambda x,y,z: x*0+1,None)))
    elif kind == 'decay':
        scene = scene.replace(run=fd.Run(steps=8,decay=fd.Decay(threshold=1e-5,window=1.)))
    else:
        scene = scene.replace(sources=(fd.Dipole('m',center=(0,0,0),component='Hx',band=fd.Band(wavelength=2.,df=.03),kind='magnetic'),))
    with pytest.raises(ValueError, match='ConstitutiveAbsorberScope'):
        _scope(scene, plan)


@pytest.mark.parametrize('case', ('torch-grad', 'torch-dual', 'jax-grad', 'torch-gpu', 'jax-gpu', 'nonfinite'))
def test_concrete_cpu_and_finite_undamped_coefficients_are_required(case):
    scene = extrusion()
    plan = quiet(fd.plan, scene)
    if case in ('torch-gpu', 'jax-gpu'):
        name = case.split('-')[0]
        device = SimpleNamespace(type='cuda', platform='gpu')
        with pytest.raises(ValueError, match='ConstitutiveAbsorberRuntime'):
            validate_runtime(SimpleNamespace(name=name, device=device))
    elif case == 'nonfinite':
        rt = runtime('numpy')
        params = quiet(fd.lower, scene, plan, rt)
        params['be'] = np.full(plan.shape, np.inf)
        with pytest.raises(ValueError, match='ConstitutiveAbsorberRange'):
            validate_links(plan, params, rt)
    else:
        def lower(eps):
            candidate = scene.replace(structures=(scene.structures[0].replace(material=fd.Material(eps=eps)),))
            return fd.lower(candidate, plan, 'jax' if case == 'jax-grad' else 'torch')['be']
        if case == 'jax-grad':
            import jax
            with pytest.raises(ValueError, match='ConstitutiveAbsorberConcrete'):
                jax.grad(lambda v: lower(v).sum())(2.25)
        else:
            import torch
            if case == 'torch-grad':
                with pytest.raises(ValueError, match='ConstitutiveAbsorberConcrete'):
                    lower(torch.tensor(2.25,dtype=torch.float64,requires_grad=True))
            else:
                with torch.autograd.forward_ad.dual_level():
                    dual = torch.autograd.forward_ad.make_dual(torch.tensor(2.25,dtype=torch.float64),
                                                               torch.tensor(1.,dtype=torch.float64))
                    with pytest.raises(ValueError, match='ConstitutiveAbsorberConcrete'):
                        lower(dual)


@pytest.mark.parametrize('backend', BACKENDS)
def test_nonoverlapping_scene_keeps_ordinary_term_and_memory_accounting(backend):
    scene = extrusion(touching=False)
    plan = quiet(fd.plan, scene)
    rt = runtime(backend, device='cpu')
    params = quiet(fd.lower, scene, plan, rt)
    assert plan.constitutive and all(type(t) is ConstitutiveTerm for t in plan.constitutive)
    assert type(plan.energy) is InteriorEnergy
    assert not any(k.startswith('constitutive_absorber') for k in params)
    report = quiet(fd.check, scene)
    expected = (2*np.prod(plan.shape)+sum(3*np.prod(t.region.shape) for t in plan.constitutive))*8
    assert report.bytes_per_state == expected


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('wall', ('PEC', 'PMC'))
@pytest.mark.parametrize('axis', (0,1,2))
def test_projected_yee_step_and_sampled_energy_match_independent_flux_oracle(backend, wall, axis):
    scene = extrusion(axis)
    faces = {a: getattr(fd, wall)() for a in 'xyz'}
    faces['xyz'[axis]] = fd.Absorber(cells=2)
    scene = scene.replace(boundaries=fd.Boundaries(**faces))
    rt = runtime(backend, device='cpu')
    plan = quiet(fd.plan, scene)
    params = quiet(fd.lower, scene, plan, rt)
    lossless_faces = {a: getattr(fd, wall)() for a in 'xyz'}
    lossless_faces['xyz'[axis]] = fd.PEC()
    lossless_scene = scene.replace(boundaries=fd.Boundaries(**lossless_faces))
    p0 = quiet(fd.plan, lossless_scene)
    q0 = quiet(fd.lower, lossless_scene, p0, 'numpy')
    diagonal = np.broadcast_to(q0['be']/(p0.dt/p0.dx), plan.shape)
    blocks = []
    for term in plan.constitutive:
        cross = np.asarray(rt.numpy(params['constitutive'][term.name]))
        localshape = cross.shape
        dest, src, weights = [], [], []
        for row, (i,j) in enumerate(((0,1),(0,2),(1,2))):
            for node in np.ndindex(localshape[1:]):
                for di in (0,-1):
                    for dj in (0,-1):
                        a,b = list(node),list(node)
                        a[2-i] += di
                        b[2-j] += dj
                        if any(v<0 or v>=n for v,n in zip(a,localshape[1:])):
                            continue
                        if any(v<0 or v>=n for v,n in zip(b,localshape[1:])):
                            continue
                        u = np.ravel_multi_index((i,*a),localshape)
                        v = np.ravel_multi_index((j,*b),localshape)
                        dest.extend((u,v)); src.extend((v,u))
                        weights.extend((cross[(row,*node)]/4,)*2)
        blocks.append((term,np.asarray(dest),np.asarray(src),np.asarray(weights)))
    def project(value, field):
        out = value.copy()
        for mask in plan.masks:
            if mask.field == field:
                out[mask.slab.slices] = 0
        return out
    def action(flux):
        out = diagonal*flux
        for term,dest,src,weights in blocks:
            addition = np.zeros(3*np.prod(term.region.shape))
            np.add.at(addition,dest,weights*flux[term.view].ravel()[src])
            out[term.view] += addition.reshape((3,)+term.region.shape)
        return project(out, 'E')
    def curl_reference(value, field):
        out = np.zeros_like(value)
        direction = 1 if field == 'E' else -1
        # All six faces here use zero exterior values; PEC/PMC differ by projections.
        for component in range(3):
            for source_component, derivative, sign in (
                ((component+2)%3,(component+1)%3,1),
                ((component+1)%3,(component+2)%3,-1)):
                arr = value[source_component]
                index = np.arange(arr.shape[2-derivative])+direction
                shifted = np.take(arr, np.clip(index,0,arr.shape[2-derivative]-1),axis=2-derivative)
                end = [slice(None)]*3
                end[2-derivative] = -1 if direction == 1 else 0
                shifted[tuple(end)] = 0
                out[component] += sign*(shifted-arr if direction == 1 else arr-shifted)/plan.dx
        return out
    rng = np.random.default_rng(182)
    D = project(rng.normal(size=plan.shape), 'E')
    H = project(rng.normal(size=plan.shape), 'H')
    E = action(D)
    state = initial_state(plan, params, rt)
    state['E'], state['H'] = rt.array(E.copy()), rt.array(H.copy())
    for term in plan.constitutive:
        state['mem'][term.name] = rt.array(D[term.view].copy())
    A = np.broadcast_to(rt.numpy(params['ae']),plan.shape)
    B = np.broadcast_to(rt.numpy(params['constitutive_absorber_drive_e']),plan.shape)
    Rh = np.broadcast_to(rt.numpy(params['constitutive_absorber_rate_h']),plan.shape)
    Re = 2*(1/B-1)/plan.dt
    Ah = (1-plan.dt*Rh/2)/(1+plan.dt*Rh/2)
    Bh = 1/(1+plan.dt*Rh/2)
    rows = {}
    for n in range(plan.steps):
        CE = curl_reference(E,'E')
        W = np.real(np.vdot(E,D)+np.vdot(H,H)-plan.dt*np.vdot(H,CE))
        rows[n] = W
        hp = project(Ah*H-plan.dt*Bh*CE,'H')
        dp = project(A*D+plan.dt*B*curl_reference(hp,'H'),'E')
        ep = action(dp)
        Wnew = np.real(np.vdot(ep,dp)+np.vdot(hp,hp)
                       -plan.dt*np.vdot(hp,curl_reference(ep,'E')))
        eb, db, hb = (ep+E)/2, (dp+D)/2, (hp+H)/2
        loss = 2*plan.dt*np.real(np.vdot(db,Re*eb)+np.vdot(hb,Rh*hb))
        near(Wnew-W+loss,0.,'step')
        state = step(plan,params,state,n,rt)
        near(rt.numpy(state['E']),ep,'step')
        near(rt.numpy(state['H']),hp,'step')
        for term in plan.constitutive:
            near(rt.numpy(state['mem'][term.name]),dp[term.view],'step')
        E,H,D = ep,hp,dp
    expected_rows = np.asarray([rows[n] for n in plan.energy.sampled(plan.steps)])
    near(rt.numpy(state['energy']),expected_rows,'step')
    assert np.all(expected_rows > 0)


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
@pytest.mark.parametrize('family', ('cylinder', 'rotated-box'))
def test_equal_material_new_route_matches_legacy_fields(backend, family):
    # These six configurations are disclosed before accepting the allowance.
    # The shaped scene retains the new zero-cross term; the other omits it.
    from studies.fdtd_constitutive_absorber_probe import measure_zero_pair
    assert ACCEPTED_BUDGETS is not None, 'Disclose calibration before accepting budgets.'
    row = measure_zero_pair(family, backend=backend, n=20, placement=.3,
                            polarization='u', time=16.)
    assert row['checks'] and all(row['checks'].values())
    assert row['energy_equality_claim'] is False
    assert row['max_roundoff_scaled'] <= ACCEPTED_BUDGETS['physical_alignment']
