"""Exact dual-symbol, clock, closed-Yee and optical hybrid film gates.

No calibration ceiling is accepted before six disclosed family/backend rows.
Reference equations use declared material values and independent dense algebra.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._kernel.observe import InteriorEnergy
from wavebit.fdtd._kernel.step import initial_state, step
from wavebit.fdtd._prepare import checks
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import backends, to_numpy

TERM = 'hybrid_film[0]'
CALIBRATED_CEILINGS = {'mixed-metal': (.005, .010), 'mixed-dielectric': (.005, .010)}
ONE = (('debye', .15, 0., 3.), ('lorentz', 8., 1., 10.))
METAL = (('debye', .2, 0., .25), ('drude', 0., 1., 81.), ('lorentz', 8., 2., 10.))
MULTI = (('debye', .08, 0., 2.), ('lorentz', 5., .8, 10.), ('debye', .25, 0., 1.),
         ('lorentz', 8., 1.5, 20.), ('lorentz', 11., 2., 15.))
MANY = tuple(('debye', .07+.03*j, 0., 1.+j) for j in range(4))+tuple(
    ('lorentz', 3.+j, .5+.2*j, 3.+j) for j in range(8))
MEMORY_KEYS = ('P', 'previous', 'delta', 'aux')


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def material(eps=4., pole_specs=MULTI):
    return fd.Dispersive(eps_inf=eps, poles=tuple(
        fd.Debye(tau=w0, delta_eps=strength) if kind == 'debye'
        else fd.Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude'
        else fd.Lorentz(omega_0=w0, gamma=gamma, strength=strength)
        for kind, w0, gamma, strength in pole_specs))


def film_scene(axis=2, eps=4., pole_specs=MULTI, host=1., shift=.03, dx=.1, courant=.2):
    lengths = [.4]*3
    lengths[axis] = 2.
    size = [length+4*dx for length in lengths]
    size[axis] = .54
    center = [0.]*3
    center[axis] = shift
    return fd.Scene(domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=8, courant=courant),
        background=fd.Material(eps=host), structures=(fd.Structure(
            fd.Box(size=tuple(size), center=tuple(center)), material(eps, pole_specs)),))


def fills(scene, plan, axis):
    shape = scene.structures[0].shape
    low, high = (float(shape.center[axis])+sign*float(shape.size[axis])/2 for sign in (-1, 1))
    result = []
    for component in range(3):
        points = np.asarray(plan.grid.yee_points('E', component)[axis])
        left, right = points-plan.dx/2, points+plan.dx/2
        fraction = np.clip((np.minimum(right, high)-np.maximum(left, low))/plan.dx, 0., 1.)
        fraction = np.where((left >= low) & (right <= high), 1., fraction)
        fraction = np.where((right <= low) | (left >= high), 0., fraction)
        dimensions = [1, 1, 1]
        dimensions[2-axis] = len(fraction)
        result.append(np.broadcast_to(fraction.reshape(dimensions), plan.shape[1:]))
    return np.asarray(result)


def geometry_weights(scene, plan, axis, eps=4., host=1.):
    f = fills(scene, plan, axis)
    A = f*host+(1-f)*eps
    eta, alpha, kappa = f*eps+(1-f)*host, np.zeros(plan.shape), f.copy()
    eta[axis] = eps*host/A[axis]
    alpha[axis] = (1-f[axis])/A[axis]
    kappa[axis] = f[axis]*host**2/A[axis]**2
    rho = np.divide(alpha, kappa, out=np.zeros_like(alpha), where=kappa > 0)
    return f, eta, alpha, kappa, rho


def independent_shift(specs, kappa, rho, dt):
    bare = [kappa*strength/(1+2*w0/dt) if kind == 'debye' and strength else np.zeros_like(kappa)
            for kind, w0, _, strength in specs]
    denominator = 1+rho*sum(bare)
    return [value/denominator for value in bare]


def expand(value, plan, term):
    out = np.zeros(plan.shape, dtype=np.asarray(value).dtype)
    out[(slice(None),)+term.region.slices] = value
    return out


def actual_map(plan, params, term, axis):
    """All 4N actual memory coordinates, without a production inverse or formula.

    Spatial coefficients vary along one axis; retain every component/normal-node
    and one equivalent transverse node, so the batch solve is small.
    """
    rt = runtime('numpy')
    count, shape = term.n_poles, (3,)+term.region.shape
    cut = [slice(None), 0, 0, 0]
    cut[3-axis] = slice(None)
    cut = tuple(cut)
    zero = lambda: np.zeros(shape, complex)
    empty = {key: tuple(zero() for _ in range(count)) for key in MEMORY_KEYS}
    sample = lambda mem: np.stack([to_numpy(mem[key][j])[cut] for key in MEMORY_KEYS for j in range(count)])
    columns = []
    for key in MEMORY_KEYS:
        for slot in range(count):
            mem = {name: tuple(np.array(v, copy=True) for v in values) for name, values in empty.items()}
            mem[key] = tuple(np.ones(shape, complex) if j == slot else zero() for j in range(count))
            columns.append(sample(term.advance(mem, np.zeros(plan.shape, complex), None, 0, params, rt)))
    A = np.stack(columns, axis=1)
    B = sample(term.advance(empty, np.ones(plan.shape, complex), None, 0, params, rt))
    return A, B, cut


def steady_response(plan, params, term, A, B, cut, angle):
    z = complex(-1., 0.) if angle == math.pi else np.exp(-1j*angle)
    size = A.shape[0]
    identity = np.eye(size).reshape((size, size)+(1,)*(A.ndim-2))
    matrix = np.moveaxis(z*identity-A, (0, 1), (-2, -1))
    # advance receives E at the NEW observer clock: X=(zI-A)^-1 zB.
    X = np.moveaxis(np.linalg.solve(matrix, np.moveaxis(z*B, 0, -1)[..., None])[..., 0], -1, 0)
    ae, be = (np.broadcast_to(to_numpy(params[key]), plan.shape) for key in ('ae', 'be'))
    update_eps = plan.dt/plan.dx*(1+ae)/(2*be)
    base = update_eps[(slice(None),)+term.region.slices][cut]
    physical = base+np.sum(X[term.n_poles:2*term.n_poles], axis=0)
    return physical, X, z


def direct_response(f, axis, eps, specs, host, angle, dt):
    own = complex(eps)
    for kind, w0, gamma, strength in specs:
        if not strength:
            continue
        if kind == 'debye':
            own += 0. if angle == math.pi else strength/(1-2j*w0*math.tan(angle/2)/dt)
        else:
            q = (0. if kind == 'drude' else w0*w0)-4*math.sin(angle/2)**2/dt**2
            q -= 0j if angle == math.pi else 1j*gamma*math.sin(angle)/dt
            own += strength/q
    expected = f*own+(1-f)*host
    expected[axis] = 1/(f[axis]/own+(1-f[axis])/host)
    return expected


def dense_predictor(mem, electric, specs, eta, kappa, rho, shift, dt):
    """Solve the original coupled equations, with dense L_D at each sample node."""
    N = len(specs)
    shape = electric.shape
    out = {key: np.zeros((N,)+shape, complex) for key in MEMORY_KEYS}
    old = {key: np.asarray(mem[key]) for key in MEMORY_KEYS}
    for point in np.ndindex(shape):
        kap, r = kappa[point], rho[point]
        ds = [j for j,s in enumerate(specs) if s[0] == 'debye' and s[3] > 0 and kap > 0]
        os = [j for j,s in enumerate(specs) if s[0] != 'debye' and s[3] > 0 and kap > 0]
        field = electric[point]
        q = np.array([old['P'][(j,)+point] for j in os])
        D = np.array([old['aux'][(j,)+point] for j in ds])
        c = np.array([shift[j][point] for j in ds])
        R = np.array([specs[j][1]/(kap*specs[j][3]) for j in ds])
        L = np.diag([1/(kap*specs[j][3]) for j in ds])+r*np.ones((len(ds),len(ds)))
        L += np.diag(2*R/dt)
        m = np.linalg.solve(L, 2*R*D/dt-r*sum(q)+field) if ds else np.empty(0,complex)
        Dnext = 2*m-D
        qnext, vbar = [], []
        for j,value in zip(os,q):
            kind,w0,gamma,S = specs[j]
            d = 0. if kind == 'drude' else w0*w0
            previous = old['previous'][(j,)+point]
            nxt = ((2-d*dt**2)*value-(1-gamma*dt/2)*previous
                   +dt**2*kap*S*(field-r*(sum(q)+sum(m))))/(1+gamma*dt/2)
            qnext.append(nxt);vbar.append((nxt-previous)/(2*dt))
        ynext = np.linalg.solve(L, 2*R*Dnext/dt-r*sum(qnext)) if ds else np.empty(0,complex)
        ynow = m-c*field
        for j,y,new_y,aux in zip(ds,ynow,ynext,Dnext):
            for key,value in (('P',new_y),('previous',y),('delta',new_y-y),('aux',aux)):
                out[key][(j,)+point]=value
        for j,q0,q1,v in zip(os,q,qnext,vbar):
            for key,value in (('P',q1),('previous',q0),('delta',q1-q0),('aux',v)):
                out[key][(j,)+point]=value
    return out


def storage_density(mem, field, specs, eta, kappa, rho, shift, dt):
    """The new correction to an eta+c electric observer, at previous-output clock."""
    P = [np.where((kappa > 0) & (s[3] > 0), mem['previous'][j]+(shift[j]*field if s[0]=='debye' else 0),0.)
         for j,s in enumerate(specs)]
    total = sum(P)
    density = rho*abs(total)**2-sum(shift)*abs(field)**2
    rd, ro, physical, velocities = [],[],[],[]
    for j,(kind,w0,gamma,strength) in enumerate(specs):
        active=(kappa > 0) & (strength > 0)
        inverse=np.divide(1.,kappa*strength,out=np.zeros_like(kappa),where=active)
        if kind=='debye':
            density += inverse*abs(P[j])**2
            rd.append(w0*inverse);ro.append(np.zeros_like(kappa));velocities.append(np.zeros_like(field))
        else:
            d=0. if kind=='drude' else w0*w0
            v=np.where(active,mem['aux'][j],0.)
            force=np.where(active,d*inverse*P[j]+rho*total-field,0.)
            density += d*inverse*abs(P[j])**2+(1+(dt*gamma/2)**2)*inverse*abs(v)**2
            density -= dt**2/4*kappa*strength*abs(force)**2
            rd.append(np.zeros_like(kappa));ro.append(gamma*inverse);velocities.append(v)
        physical.append(P[j])
    return density,physical,rd,ro,velocities


def periodic_curl_e(field):
    parts=[]
    for component in range(3):
        a,b=(component+1)%3,(component+2)%3
        parts.append(np.roll(field[b],-1,axis=2-a)-field[b]
                     -np.roll(field[a],-1,axis=2-b)+field[a])
    return np.asarray(parts)


@pytest.mark.parametrize('axis', (0, 1, 2), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('one-each', 'metal', 'multi', 'many', 'equal', 'weak', 'tiny-fill'))
def test_hybrid_actual_recurrence_preserves_both_symbols_and_nyquist(axis, case):
    specs = {'one-each': ONE, 'metal': METAL, 'multi': MULTI, 'many': MANY,
        'equal': (ONE[0], ONE[0], ONE[1], ONE[1]),
        'weak': (('debye', .15, 0., 1e-24), ONE[1]), 'tiny-fill': ONE}[case]
    scene = film_scene(axis=axis, pole_specs=specs, shift=.02+1e-12 if case == 'tiny-fill' else .03)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    term = next(t for t in plan.terms if t.name == TERM)
    assert term.n_poles == len(specs) and term.kinds == tuple(s[0] for s in specs)
    f, eta, _, kappa, rho = geometry_weights(scene, plan, axis)
    shift = independent_shift(specs, kappa, rho, plan.dt)
    coefficients = params['poles'][TERM]
    for j, spec in enumerate(specs):
        got = expand(np.broadcast_to(coefficients['debye_c'][j], (3,)+term.region.shape), plan, term)
        np.testing.assert_allclose(got, shift[j], rtol=64*np.finfo(float).eps, atol=0.)
        if spec[0] == 'debye' and spec[3] > 0:
            assert np.all(got[kappa > 0] > 0)
    A, B, cut = actual_map(plan, params, term, axis)
    for angle in (2*math.pi*plan.dt, .3, 1.2, math.pi-.1, math.pi):
        actual, X, z = steady_response(plan, params, term, A, B, cut, angle)
        expected = direct_response(f, axis, 4., specs, 1., angle, plan.dt)
        expected = expected[(slice(None),)+term.region.slices][cut]
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        if angle == math.pi:
            # An actual alternating-field update, not merely substitution in an oracle.
            shape = (3,)+term.region.shape
            def full(value):
                dimensions = [3, 1, 1, 1]
                dimensions[3-axis] = value.shape[1]
                return np.broadcast_to(value.reshape(dimensions), shape).copy()
            mem = {key: tuple(full(X[k*term.n_poles+j]) for j in range(term.n_poles))
                   for k, key in enumerate(MEMORY_KEYS)}
            result = term.advance(mem, -np.ones(plan.shape, complex), None, 0, params, runtime('numpy'))
            for key in MEMORY_KEYS:
                for actual_value, value in zip(result[key], mem[key]):
                    np.testing.assert_allclose(actual_value, -value, rtol=3e-12, atol=3e-12)
            for j, spec in enumerate(specs):
                if spec[0] == 'debye':
                    cj = np.broadcast_to(coefficients['debye_c'][j], shape)[cut]
                    np.testing.assert_allclose(X[term.n_poles+j]+cj, 0., atol=3e-12)
            assert z == -1
    # Dense coupled equations use consistent histories from an actual first update.
    rt = runtime('numpy')
    first = term.advance(term.init_mem(plan, rt), np.full(plan.shape, .7), None, 0, params, rt)
    field = np.linspace(-.5, .8, math.prod(plan.shape)).reshape(plan.shape)
    result = term.advance(first, field, None, 1, params, rt)
    loc = (slice(None),)+term.region.slices
    expected = dense_predictor(first, field[loc], specs, eta[loc], kappa[loc], rho[loc],
                               [value[loc] for value in shift], plan.dt)
    for key in MEMORY_KEYS:
        np.testing.assert_allclose(np.asarray(result[key]), expected[key], rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('axis', (0, 1, 2), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('zero-debye', 'zero-oscillator', 'all-zero', 'reversed', 'equal-debye', 'equal-oscillator'))
def test_hybrid_inactive_family_endpoints_order_and_memory_are_exact(axis, case):
    specs = MULTI
    if case.startswith('zero-') or case == 'all-zero':
        specs = tuple((kind,w0,gamma,0. if case == 'all-zero' or (kind == 'debye') == (case == 'zero-debye') else strength)
                      for kind,w0,gamma,strength in specs)
    elif case == 'equal-debye':
        specs = (ONE[0], ONE[0], ONE[1])
    elif case == 'equal-oscillator':
        specs = (ONE[0], ONE[1], ONE[1])
    scene = film_scene(axis=axis, pole_specs=specs)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    term = next(t for t in plan.terms if t.name == TERM)
    mem = term.init_mem(plan, runtime('numpy'))
    arrays = [array for key in MEMORY_KEYS for array in mem[key]]
    assert set(mem) == set(MEMORY_KEYS) and len(arrays) == 4*len(specs)
    assert all(array.shape == (3,)+term.region.shape and np.count_nonzero(array) == 0 for array in arrays)
    assert not any(np.shares_memory(left, right) for i,left in enumerate(arrays) for right in arrays[i+1:])
    assert term.memory_arrays_per_pole == 4
    f = fills(scene, plan, axis)
    A,B,cut = actual_map(plan, params, term, axis)
    for angle in (.2, 1.2, math.pi):
        got, _, _ = steady_response(plan, params, term, A, B, cut, angle)
        expected = direct_response(f, axis, 4., specs, 1., angle, plan.dt)
        np.testing.assert_allclose(got, expected[(slice(None),)+term.region.slices][cut], rtol=3e-12, atol=3e-12)
    coefficients = params['poles'][TERM]
    for j,spec in enumerate(specs):
        if not spec[3]:
            for key in ('drive', 'coupling', 'debye_c', 'debye_drive', 'debye_feedback',
                        'energy_inverse_delta', 'energy_inverse_strength', 'energy_active'):
                assert np.count_nonzero(coefficients[key][j]) == 0
    if case == 'all-zero':
        other = scene.replace(structures=(fd.Structure(scene.structures[0].shape, fd.Material(eps=4.)),))
        other_plan = quiet(fd.plan, other)
        other_params = fd.lower(other, other_plan, 'numpy')
        for key in ('ae','be','ah','bh'):
            np.testing.assert_allclose(np.broadcast_to(params[key], plan.shape),
                                       np.broadcast_to(other_params[key], plan.shape), rtol=2e-14, atol=2e-14)
    if case == 'reversed':
        other = film_scene(axis=axis, pole_specs=specs[::-1])
        other_plan = quiet(fd.plan, other)
        other_params = fd.lower(other, other_plan, 'numpy')
        other_term = next(t for t in other_plan.terms if t.name == TERM)
        a,b,c = actual_map(other_plan, other_params, other_term, axis)
        got, _, _ = steady_response(plan, params, term, A,B,cut,.4)
        reverse, _, _ = steady_response(other_plan,other_params,other_term,a,b,c,.4)
        np.testing.assert_allclose(got, reverse, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('mixed', 'multi', 'zero-debye', 'zero-oscillator', 'lossless-oscillator', 'debye-with-lossless-oscillator', 'hold-zero', 'negative-hold'))
def test_hybrid_observer_clock_and_closed_yee_dissipation(backend, axis, case):
    specs = ONE if case == 'mixed' else MULTI
    if case in ('zero-debye', 'zero-oscillator', 'lossless-oscillator', 'debye-with-lossless-oscillator'):
        specs = tuple((kind,w0,0. if case in ('lossless-oscillator','debye-with-lossless-oscillator') else gamma,
                       0. if (kind == 'debye' and case in ('zero-debye','lossless-oscillator')) or (kind != 'debye' and case == 'zero-oscillator') else strength)
                      for kind,w0,gamma,strength in specs)
    scene = film_scene(axis=axis, pole_specs=specs)
    plan = quiet(fd.plan, scene)
    oscillator = next(j for j,spec in enumerate(specs) if spec[0] != 'debye')
    if case in ('hold-zero', 'negative-hold'):
        gamma = (2. if case == 'hold-zero' else 3.)/plan.dt
        specs = tuple(spec[:2]+(gamma,spec[3]) if j == oscillator else spec for j,spec in enumerate(specs))
        scene = film_scene(axis=axis, pole_specs=specs)
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(t for t in plan.terms if t.name == TERM)
    co = params['poles'][TERM]
    if case == 'hold-zero':
        assert np.max(abs(to_numpy(co['hold'][oscillator]))) < 3e-16
    if case == 'negative-hold':
        assert np.min(to_numpy(co['hold'][oscillator])) < 0
    _,eta,_,kappa,rho = geometry_weights(scene,plan,axis)
    shift = independent_shift(specs,kappa,rho,plan.dt)
    rng = np.random.default_rng(814)
    shape = (3,)+term.region.shape
    local = {key: tuple(rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in specs) for key in MEMORY_KEYS}
    mem = {key: tuple(rt.array(v,'complex128') for v in values) for key,values in local.items()}
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    whole = tuple((part.start,part.stop) for part in term.region.slices)
    expanded = {key: [expand(value,plan,term) for value in values] for key,values in local.items()}
    density = storage_density(expanded,field,specs,eta,kappa,rho,shift,plan.dt)[0]
    cut = tuple((lo+1 if hi-lo > 2 else lo,hi) for lo,hi in whole)
    for cells,comps in ((whole,slice(None)),(cut,slice(axis,axis+1))):
        offsets = tuple(lo-part.start for (lo,_),part in zip(cells,term.region.slices))
        got = InteriorEnergy._pole(mem,rt.array(field,'complex128'),co,len(specs),cells,offsets,plan.dt,rt,comps=comps)
        expected = np.sum(density[(comps,)+tuple(slice(lo,hi) for lo,hi in cells)])
        assert float(to_numpy(got)) == pytest.approx(expected,rel=5e-13,abs=5e-13)
    # Private consistent initialization is a kernel proof fixture; public InitialField stays refused.
    state = initial_state(plan,params,rt)
    state['E'] = rt.array(rng.normal(size=plan.shape),plan.dtype)
    state['H'] = rt.array(rng.normal(size=plan.shape),plan.dtype)
    state['mem'][TERM] = term.advance(term.init_mem(plan,rt),state['E'],state['H'],0,params,rt)
    update_eps = eta+sum(shift)
    def storage(current):
        E,H = np.array(to_numpy(current['E']),copy=True),np.array(to_numpy(current['H']),copy=True)
        values = {key: [expand(np.array(to_numpy(v),copy=True),plan,term) for v in current['mem'][TERM][key]] for key in MEMORY_KEYS}
        detail = storage_density(values,E,specs,eta,kappa,rho,shift,plan.dt)
        Hplus = H-plan.dt/plan.dx*periodic_curl_e(E)
        field_value = np.sum(update_eps*E**2+H*Hplus)
        observed = float(to_numpy(InteriorEnergy._pole(current['mem'][TERM],current['E'],co,len(specs),whole,(0,0,0),plan.dt,rt)))
        reference = np.sum(detail[0])
        assert observed == pytest.approx(reference,rel=5e-13,abs=5e-13)
        if case in ('zero-debye','lossless-oscillator'):
            # The old valid oscillator storage differs under damping, even though its transfer agrees.
            old_storage = 0.
            total_new,total_old = np.zeros(plan.shape),np.zeros(plan.shape)
            difference = 0.
            for j,(kind,w0,gamma,strength) in enumerate(specs):
                if kind == 'debye' or not strength:
                    continue
                active = kappa > 0
                inv = np.divide(1.,kappa*strength,out=np.zeros_like(kappa),where=active)
                p,q = (np.where(active,values[key][j],0.) for key in ('P','previous'))
                d = 0. if kind == 'drude' else w0*w0
                old_storage += np.sum(inv*((p-q)**2/plan.dt**2+d*p*q)-E*(p-q))
                total_new += p;total_old += q
                difference += plan.dt*np.sum(detail[3][j]*abs(detail[4][j])**2)
            old_storage += np.sum(rho*total_new*total_old)
            assert observed-old_storage == pytest.approx(difference,rel=2e-11,abs=2e-11)
        return field_value+observed,detail
    previous,details = storage(state)
    for n in range(4):
        state = step(plan,params,state,n,rt)
        current,new_details = storage(state)
        debye_loss,oscillator_loss = 0.,0.
        for j,spec in enumerate(specs):
            if spec[0] == 'debye':
                debye_loss += 2*plan.dt*np.sum(details[2][j]*abs((new_details[1][j]-details[1][j])/plan.dt)**2)
            else:
                oscillator_loss += plan.dt*np.sum(details[3][j]*(abs(new_details[4][j])**2+abs(details[4][j])**2))
        loss = debye_loss+oscillator_loss
        budget = 4096*np.finfo(float).eps*max(1.,abs(previous),abs(current),loss)
        assert loss >= 0 and abs(current-previous+loss) <= budget
        if case == 'debye-with-lossless-oscillator':
            assert debye_loss > 0 and oscillator_loss == 0
        if case == 'lossless-oscillator':
            assert debye_loss == oscillator_loss == 0
        previous,details = current,new_details


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('family', ('mixed-metal', 'mixed-dielectric'))
def test_oblique_hybrid_matches_complex_airy_on_each_backend(backend,family):
    from studies.fdtd.hybrid_film.fdtd_hybrid_film_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review six disclosed hybrid calibration rows before accepting ceilings.'
    row = measure(family=family,n=40,pol='p',periods=120,backend=backend)
    r_limit,t_limit = CALIBRATED_CEILINGS[family]
    assert max(row['error_r']) < r_limit and max(row['error_t']) < t_limit
    assert row['stable'] is True and max(row['decay'].values()) < 1e-5
    assert row['energy_shape'] == [row['energy_rows']] and row['energy_finite'] is True and row['energy_max'] > 0


def refused_scene(case):
    scene = film_scene()
    body, own = scene.structures[0].shape, scene.structures[0].material
    if case in ('finite-box', 'sphere'):
        body = fd.Box(size=(.2, .2, .54)) if case == 'finite-box' else fd.Sphere(radius=.15)
        scene = scene.replace(domain=fd.Domain(size=(1.2, 1.2, 2.)))
    elif case == 'turned':
        body = body.replace(angle=15.)
        scene = scene.replace(domain=fd.Domain(size=(2., 2., 2.)))
    elif case == 'ranged':
        body = body.replace(range=((-1., 1.), (-1., 1.), (-.8, .8)))
    elif case == 'anisotropic':
        own = own.replace(eps_inf=(4., 5., 4.))
    elif case == 'magnetic':
        own = own.replace(mu=2.)
    elif case == 'background-loss':
        return scene.replace(background=fd.Conductor(eps=1., sigma=.2))
    elif case == 'background-magnetic':
        return scene.replace(background=fd.Material(eps=1., mu=2.))
    elif case == 'partner':
        return scene.replace(structures=scene.structures+(
            fd.Structure(fd.Slab(z=(.5, .7)), fd.Material(eps=2.)),))
    elif case == 'initial':
        return scene.replace(initial=fd.InitialField(E=(0., 0., 1.)))
    elif case == 'stencil':
        return scene.replace(structures=(fd.Structure(body, own, discretisation='stencil'),))
    elif case == 'sub-dx':
        body = body.replace(size=(.8, .8, .05))
    elif case == 'transverse-halo':
        body = body.replace(size=(.5, .8, .54))
    elif case == 'transverse-absorber':
        body = body.replace(size=(3.2, .8, .54))
        scene = scene.replace(domain=fd.Domain(size=(2.8, .4, 2.)), boundaries=fd.Boundaries(
            x=(fd.PML(cells=12, wavelength=1.), fd.PEC()), y=fd.Periodic(), z=fd.Periodic()))
    elif case == 'pml-normal':
        body = body.replace(center=(0., 0., -.6))
        scene = scene.replace(boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
            z=(fd.PML(cells=12, wavelength=1.), fd.PEC())))
    else:
        raise AssertionError(case)
    return scene.replace(structures=(fd.Structure(body, own),))


@pytest.mark.parametrize('case', ('finite-box', 'sphere', 'turned', 'ranged', 'anisotropic',
    'magnetic', 'background-loss', 'background-magnetic', 'partner',
    'initial', 'stencil', 'pml-normal', 'sub-dx', 'transverse-halo', 'transverse-absorber',
    'unrepresentable-strength'))
def test_unsupported_hybrid_films_are_refused_by_name(case):
    if case == 'unrepresentable-strength':
        scene = film_scene(pole_specs=(('debye', .15, 0., math.ulp(0.)), ONE[1]))
        body = scene.structures[0].shape
        candidates = [scene]
        for pole in (fd.Drude(omega_p=1e-200, gamma=1.),
                     fd.Lorentz(omega_0=1e-200, gamma=1., strength=1.)):
            candidates.append(scene.replace(structures=(fd.Structure(body,
                fd.Dispersive(eps_inf=4., poles=(fd.Debye(tau=.15, delta_eps=3.), pole))),)))
        for delta,tau in ((1.,1e16),(1e-42,1e-20)):
            candidates.append(film_scene(pole_specs=(('debye',tau,0.,delta),('lorentz',1.,.1,1.))))
        for candidate in candidates:
            with pytest.raises(ValueError, match='(?i)hybrid-film.*float64'):
                plan = quiet(fd.plan,candidate)
                fd.lower(candidate,plan,'numpy')
        # These analytical candidates preserve a representable gap from each endpoint.
        for delta,tau in ((1.,1e12),(1e-32,1e-15)):
            candidate = film_scene(pole_specs=(('debye',tau,0.,delta),('lorentz',1.,.1,1.)))
            plan = quiet(fd.plan,candidate)
            params = fd.lower(candidate,plan,'numpy')
            co = params['poles'][TERM]
            gain,active = np.broadcast_arrays(co['debye_gain'][0],co['energy_debye_active'][0])
            assert np.all(abs(gain[active > 0]) < 1.)
        return
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('dispers', 'film', 'smooth', 'lossless', 'absorbing'))
    if case == 'stencil':
        assert 'smoothed' in message and 'fd.dispersive' in message


@pytest.mark.parametrize('change', ('strength0', 'strength1', 'tau0', 'gamma1', 'omega1',
    'pole-count', 'reversed-poles', 'kind', 'eps', 'center', 'background', 'missing-memory',
    'extra-memory', 'ordinary-memory', 'nonfilm-plan', 'one-pole-plan', 'debye-plan', 'conductive-plan'))
def test_prepared_hybrid_plan_refuses_changed_identity_or_memory(change):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    body, own = scene.structures[0].shape, scene.structures[0].material
    other = scene
    if change in ('missing-memory', 'extra-memory', 'ordinary-memory'):
        term = next(t for t in plan.terms if t.name == TERM)
        if change == 'ordinary-memory':
            from wavebit.fdtd._kernel.poles import PoleTerm
            terms = plan.terms+(PoleTerm(name='poles[0]', region=term.region, slabs=term.slabs, n_poles=3),)
        else:
            terms = (tuple(t for t in plan.terms if t is not term) if change == 'missing-memory'
                     else plan.terms+(dataclasses.replace(term, name='hybrid_film[1]'),))
        plan = dataclasses.replace(plan, terms=terms)
    elif change in ('nonfilm-plan', 'one-pole-plan', 'debye-plan', 'conductive-plan'):
        previous = fd.Material(eps=4.)
        if change == 'one-pole-plan':
            previous = material(pole_specs=(MULTI[1],))
        elif change == 'debye-plan':
            previous = fd.Dispersive(eps_inf=4., poles=(fd.Debye(tau=.15, delta_eps=6.),))
        elif change == 'conductive-plan':
            previous = fd.Conductor(eps=4., sigma=8.)
        plan = quiet(fd.plan, scene.replace(structures=(fd.Structure(body, previous),)))
    elif change == 'background':
        other = scene.replace(background=fd.Material(eps=2.))
    elif change == 'center':
        other = scene.replace(structures=(fd.Structure(body.replace(center=(0., 0., .01)), own),))
    else:
        specs = [list(spec) for spec in MULTI]
        if change.startswith(('strength', 'tau', 'gamma', 'omega')):
            field = 3 if change.startswith('strength') else 2 if change.startswith('gamma') else 1
            specs[int(change[-1])][field] *= 1.2
        elif change == 'pole-count':
            specs.append(['lorentz', 12., 3., 5.])
        elif change == 'reversed-poles':
            specs = specs[::-1]
        own = material(5. if change == 'eps' else 4., specs)
        if change == 'kind':
            own = fd.Dispersive(eps_inf=4., poles=(fd.Debye(tau=.15, delta_eps=6.),))
        other = scene.replace(structures=(fd.Structure(body, own),))
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('strength0', 'strength1', 'tau0', 'gamma1', 'omega1'))
def test_traced_hybrid_parameters_cannot_bypass_the_concrete_door(mode, parameter):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    index = int(parameter[-1])
    field = 3 if parameter.startswith('strength') else 2 if parameter.startswith('gamma') else 1
    initial = MULTI[index][field]

    def lower(value, backend):
        specs = [list(spec) for spec in MULTI]
        specs[index][field] = value
        other = scene.replace(structures=(fd.Structure(scene.structures[0].shape, material(pole_specs=specs)),))
        return fd.lower(other, plan, backend)['be']

    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|hybrid'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|hybrid'):
                    lower(dual, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|hybrid'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')



@pytest.mark.parametrize('count', (1, 2, 4))
def test_hybrid_cfl_uses_physical_eta_and_both_conservative_rates(count):
    specs = tuple(('debye',.12+.02*j,0.,8.+j) for j in range(count))+tuple(
        ('lorentz',4.+j,.5+j,40.+j) for j in range(count))
    scene = film_scene(eps=1.,host=4.,pole_specs=specs,courant=.01)
    plan = quiet(fd.plan,scene)
    _,eta,alpha,kappa,_ = geometry_weights(scene,plan,2,eps=1.,host=4.)
    m1 = sum(s[3]/s[1] for s in specs if s[0] == 'debye')
    m2 = sum(s[3]/s[1]**2 for s in specs if s[0] == 'debye')
    relaxation = max(m2,float(np.max(kappa/eta*(m2+alpha*m1*m1))))
    oscillator = max(s[1]**2 for s in specs if s[0] != 'debye')+sum(s[3] for s in specs if s[0] != 'debye')
    vacuum = plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+(relaxation+oscillator)/4)
    got,reason = checks.cfl_bound(scene,plan)
    assert reason is None and got == pytest.approx(expected,rel=2e-13)
    # The mixed cells increase this retained Debye policy beyond its bulk budget.
    assert relaxation > m2
    unsafe = scene.replace(run=fd.Run(steps=2,courant=(expected/vacuum+1)/2))
    with pytest.raises(ValueError,match='(?i)stability|bound|courant'):
        quiet(fd.plan,unsafe)
    zero = film_scene(eps=1.,pole_specs=(('debye',1e-250,0.,0.),('lorentz',1e200,1e200,0.)))
    zero_plan = quiet(fd.plan,zero)
    zero_params = fd.lower(zero,zero_plan,'numpy')
    bound,reason = checks.cfl_bound(zero,zero_plan)
    assert reason is None and bound == pytest.approx(vacuum,rel=2e-15)
    assert np.count_nonzero(zero_params['poles'][TERM]['energy_hybrid_shift']) == 0
    with pytest.raises(ValueError,match='(?i)stability|bound|courant'):
        checks.check_stability(zero,dataclasses.replace(zero_plan,dt=2*bound))


@pytest.mark.parametrize('case', ('band-debye', 'band-interior-peak', 'active-undamped', 'monitor-phase',
    'monitor-skin', 'resolution-phase', 'pml-phase', 'zero-strength-resonance', 'resolved-metal',
    'traced-band-forward', 'traced-band-jax'))
def test_hybrid_spectral_guard_covers_both_band_families_and_each_sample(case):
    from studies.fdtd.hybrid_film.fdtd_hybrid_film_probe import build, band_certificate, epsilon
    from wavebit.fdtd._prepare.hybrid_resolution import validate
    scene,_ = build()
    if case.startswith('traced-band'):
        grid = quiet(fd.plan,scene).grid
        def check(df):
            incoming = scene.sources[0].replace(band=fd.Band(wavelength=1.,df=df,cutoff=1e-6))
            return validate(scene.replace(sources=(incoming,)),grid)
        if case == 'traced-band-jax':
            jax = pytest.importorskip('jax')
            with pytest.raises(ValueError,match='(?i)concrete|traced|band'):
                jax.jit(check)(.1)
        else:
            torch = pytest.importorskip('torch')
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(.1,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
                with pytest.raises(ValueError,match='(?i)concrete|traced|band'):
                    check(dual)
        return
    if case in ('band-interior-peak','band-debye'):
        specs = ((('debye',.2,0.,.001),('lorentz',6.,.1,10.)) if case == 'band-interior-peak'
                 else (('debye',.15,0.,6.),('lorentz',8.,1.,.001)))
        scene,_ = build(n=20,pole_specs=specs,courant=.01)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape,fd.Material(eps=1.)),))
        grid = quiet(fd.plan,safe).grid
        weakened = band_certificate('mixed-metal',pole_specs=specs,
            endpoints_only=case == 'band-interior-peak',omit_debye=case == 'band-debye')
        assert grid.dx*weakened < 2*math.pi/8
        assert grid.dx*band_certificate('mixed-metal',pole_specs=specs) > 2*math.pi/8
        wavelengths = np.array((.9,1.,1.1))
        wavevector = 2*math.pi/wavelengths*np.sqrt(epsilon(wavelengths,pole_specs=specs))
        assert np.all(2*math.pi/(wavevector.real*grid.dx) > 8)
        assert np.all(1/(wavevector.imag*grid.dx) > 1)
        # A later diagnostic may independently refuse; isolate this certificate first.
        with pytest.raises(ValueError,match='(?i)source .* band.*bulk spectral mesh bound'):
            validate(scene,grid)
    elif case == 'active-undamped':
        scene,_ = build(pole_specs=(('debye',.2,0.,1.),('lorentz',6.,0.,10.)))
    elif case == 'monitor-phase':
        scene = scene.replace(monitors=tuple(m.replace(wavelengths=(.1,)) for m in scene.monitors))
    elif case == 'monitor-skin':
        specs = (('debye',.2,0.,0.),('drude',0.,.1,80.**2),('lorentz',8.,1.,1.))
        scene,_ = build(pole_specs=specs,courant=.01)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape,fd.Material(eps=1.)),))
        grid = quiet(fd.plan,safe).grid
        k = 2*math.pi*np.sqrt(epsilon((1.,),pole_specs=specs)[0])
        assert 2*math.pi/(k.real*grid.dx) > 8 and 1/(k.imag*grid.dx) < 1
        with pytest.raises(ValueError,match='(?i)skin depth'):
            validate(scene.replace(sources=()),grid)
    elif case == 'resolution-phase':
        scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=4,wavelength=.1))
    elif case == 'pml-phase':
        scene = scene.replace(boundaries=scene.boundaries.replace(z=fd.PML(cells=24,wavelength=.1,reflection=1e-12)))
    elif case == 'zero-strength-resonance':
        scene,_ = build(pole_specs=(('debye',1e-250,0.,0.),('lorentz',2*math.pi,0.,0.)))
    if case in ('zero-strength-resonance','resolved-metal'):
        assert quiet(fd.plan,scene).dt > 0
        if case == 'zero-strength-resonance':
            from wavebit.fdtd._assemble import _round_trip
            assert math.isfinite(_round_trip(scene,scene.sources[0],None))
    else:
        with pytest.raises(ValueError,match='(?i)wavelength|skin|band|spectral|cells|resonance'):
            quiet(fd.plan,scene)
