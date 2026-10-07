"""Separated conductive-film stack gates, drafted before any new response.

The local oracle is the declared complex medium on explicit Yee intervals.
No calibration ceiling is accepted before the six disclosed backend rows.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._check import report
from wavebit.fd._kernel.observe import InteriorEnergy
from wavebit.fd._kernel.poles import PoleTerm
from wavebit.fd._kernel.step import initial_state, step
from wavebit.fd._lower import checks
from wavebit.fd._runtime import runtime
from fd_helpers import backends, to_numpy

PREFIX = 'conductive_stack['
MEMORY_KEYS = ('P', 'previous', 'delta')
# Post-disclosure coarse-grid regression limits; see the calibration acceptance record.
CALIBRATED_CEILINGS = (.007, .014)


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def stack_scene(axis=2, materials=((4., 8.), (2.25, 4.)), host=1., dx=.1,
                courant=.2, centers=None, thicknesses=None, complex_fields=False):
    count = len(materials)
    lengths = [.4, .4, .4]
    lengths[axis] = max(4., count*1.2)
    centers = tuple((j-(count-1)/2)*.85+.03 for j in range(count)) if centers is None else centers
    thicknesses = tuple(.34+.04*(j % 3) for j in range(count)) if thicknesses is None else thicknesses
    structures = []
    for (eps, sigma), center, thickness in zip(materials, centers, thicknesses):
        size, position = [length+4*dx for length in lengths], [0., 0., 0.]
        size[axis], position[axis] = thickness, center
        structures.append(fd.Structure(fd.Box(size=tuple(size), center=tuple(position)),
                                       fd.Conductor(eps=eps, sigma=sigma)))
    faces = [fd.Periodic(), fd.Periodic(), fd.Periodic()]
    if complex_fields:
        faces[(axis+1) % 3] = fd.Bloch(phase=.31)
    return fd.Scene(domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(**dict(zip('xyz', faces))),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=5, courant=courant),
        background=fd.Material(eps=host), structures=tuple(structures))


def member_fills(scene, plan, axis, member):
    """E_c is half a cell along c, with zero offset along the other two axes."""
    shape = scene.structures[member].shape
    low = float(shape.center[axis])-float(shape.size[axis])/2
    high = float(shape.center[axis])+float(shape.size[axis])/2
    values = []
    for component in range(3):
        x = plan.grid.low[axis]+(np.arange(plan.grid.n[axis])+.5*(component == axis))*plan.dx
        fraction = np.maximum(0., np.minimum(high, x+plan.dx/2)-np.maximum(low, x-plan.dx/2))/plan.dx
        fraction = np.where((x-plan.dx/2 >= low) & (x+plan.dx/2 <= high), 1., fraction)
        fraction = np.where((x+plan.dx/2 <= low) | (x-plan.dx/2 >= high), 0., fraction)
        extent = [1, 1, 1]
        extent[2-axis] = len(fraction)
        values.append(np.broadcast_to(fraction.reshape(extent), plan.shape[1:]))
    return np.asarray(values)


def stack_terms(plan):
    return tuple(t for t in plan.terms if t.name.startswith(PREFIX))


def expand(value, plan, term):
    out = np.zeros(plan.shape, dtype=np.asarray(value).dtype)
    out[(slice(None),)+term.region.slices] = value
    return out


def independent_weights(scene, plan, axis):
    """Real high-frequency medium and each interface's physical relaxation."""
    host = float(scene.background.eps[0])
    eta, sigma = np.full(plan.shape, host), np.zeros(plan.shape)
    deltas, taus, shifts = [], [], []
    for member, structure in enumerate(scene.structures):
        a, cond = float(structure.material.eps[0]), float(structure.material.sigma[0])
        f = member_fills(scene, plan, axis, member)
        own = f*a+(1-f)*host
        own[axis] = 1/(f[axis]/a+(1-f[axis])/host)
        occupied = f > 0
        eta[occupied] = own[occupied]
        local_sigma = f*cond
        local_sigma[axis] = np.where(f[axis] == 1., cond, 0.)
        sigma[occupied] = local_sigma[occupied]
        active = (f > 0) & (f < 1) & (cond > 0)
        active[np.arange(3) != axis] = False
        safe = np.where(active, f, .5)
        A = safe*host+(1-safe)*a
        delta = np.where(active, safe*host**2/((1-safe)*A), 0.)
        tau = np.where(active, A/((1-safe)*(cond if cond else 1.)), 1.)
        deltas.append(delta)
        taus.append(tau)
        shifts.append(delta*plan.dt/(2*tau+plan.dt))
    return eta, sigma, tuple(deltas), tuple(taus), tuple(shifts)


def recurrence_map(plan, params, term, axis):
    """Read the real production advance on a basis, not its coefficient formulas."""
    rt, shape = runtime('numpy'), (3,)+term.region.shape
    cut = [slice(None), 0, 0, 0]
    cut[3-axis] = slice(None)
    cut = tuple(cut)
    empty = {key: (np.zeros(shape, complex),) for key in MEMORY_KEYS}
    sample = lambda mem: np.stack([to_numpy(mem[key][0])[cut] for key in MEMORY_KEYS])
    columns = []
    for key in MEMORY_KEYS:
        mem = {name: (np.ones(shape, complex) if name == key else np.zeros(shape, complex),)
               for name in MEMORY_KEYS}
        columns.append(sample(term.advance(mem, np.zeros(plan.shape, complex), None, 0, params, rt)))
    A = np.stack(columns, axis=1)
    B = sample(term.advance(empty, np.ones(plan.shape, complex), None, 0, params, rt))
    return A, B


def actual_response(plan, params, axis, angle):
    ae, be = (np.broadcast_to(to_numpy(params[key]), plan.shape) for key in ('ae', 'be'))
    omega = 2*math.tan(angle/2)/plan.dt
    response = np.asarray(plan.dt/plan.dx*(1+ae)/(2*be)+1j*(1-ae)/(plan.dx*be*omega), complex)
    z = np.exp(-1j*angle)
    total_current, expected_current = np.zeros(plan.shape, complex), np.zeros(plan.shape, complex)
    for term in stack_terms(plan):
        A, B = recurrence_map(plan, params, term, axis)
        identity = np.eye(3).reshape((3, 3)+(1,)*(A.ndim-2))
        matrix = np.moveaxis(z*identity-A, (0, 1), (-2, -1))
        X = np.moveaxis(np.linalg.solve(matrix, np.moveaxis(z*B, 0, -1)[..., None])[..., 0], -1, 0)
        compact = [3, 1, 1, 1]
        compact[3-axis] = X.shape[-1]
        local = {key: (np.broadcast_to(X[j].reshape(compact), (3,)+term.region.shape).copy(),)
                 for j, key in enumerate(MEMORY_KEYS)}
        view = (slice(None),)+term.region.slices
        response[view] += local['previous'][0]
        expected_current[view] = -(plan.dx/plan.dt)*(z-1)*local['previous'][0]
        total_current = term.before_E(total_current, None, local, 0, params, runtime('numpy'))
        np.testing.assert_allclose(X[0], z*X[1], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(total_current, expected_current, rtol=3e-13, atol=3e-13)
    return response, omega


@pytest.mark.parametrize('axis', (0, 1, 2), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('pair', 'triple', 'four', 'eight', 'equal-eps', 'one-inactive', 'all-inactive', 'thin', 'tiny-fill'))
def test_every_stack_member_matches_the_independent_discrete_series_parallel_law(axis, case):
    count = {'triple': 3, 'four': 4, 'eight': 8}.get(case, 2)
    materials = tuple((2.+.5*j, 3.+2*j) for j in range(count))
    host = 1.44 if case == 'triple' else 1.
    if case == 'equal-eps':
        materials = ((1., 8.), (1., 3.))
    elif case == 'one-inactive':
        materials = ((4., 0.), (2.25, 4.))
    elif case == 'all-inactive':
        materials = ((4., 0.), (2.25, 0.))
    kwargs = {'thicknesses': (.1, .1)} if case == 'thin' else {}
    if case == 'tiny-fill':
        kwargs = {'dx': .125, 'centers': (-.625, .125+2.**-42), 'thicknesses': (.5, .5)}
    scene = stack_scene(axis, materials, host=host, **kwargs)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    terms = stack_terms(plan)
    assert [t.name for t in terms] == [f'{PREFIX}{j}]' for j in range(count)]
    assert all(t.n_poles == 1 for t in terms)
    for angle in (.071, .39, 1.2):
        got, omega = actual_response(plan, params, axis, angle)
        expected = np.full(plan.shape, complex(host))
        for member, (a, sigma) in enumerate(materials):
            f = member_fills(scene, plan, axis, member)
            own = a+1j*sigma/omega
            local = f*own+(1-f)*host
            local[axis] = 1/(f[axis]/own+(1-f[axis])/host)
            expected[f > 0] = local[f > 0]
        np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)
        assert np.all(np.isfinite(got))
    if case == 'tiny-fill':
        term = terms[1]
        fraction = member_fills(scene, plan, axis, 1)[(slice(None),)+term.region.slices]
        tiny = (fraction > 0) & (fraction < 1e-9)
        tiny[np.arange(3) != axis] = False
        drive = np.broadcast_to(to_numpy(params['poles'][term.name]['drive'][0]), fraction.shape)
        assert np.any(tiny) and np.all(drive[tiny] > 0), 'positive fills cannot be snapped away'

@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('case', ('pair', 'triple', 'all-inactive', 'complex', 'reversed'))
def test_stack_regions_preserve_singletons_order_and_actual_state_allocation(backend, case, capsys):
    materials = ((4., 8.), (2.25, 4.), (3., 0.)) if case == 'triple' else ((4., 8.), (2.25, 4.))
    if case == 'all-inactive':
        materials = tuple((eps, 0.) for eps, _ in materials)
    scene = stack_scene(materials=materials, complex_fields=case == 'complex')
    if case == 'reversed':
        scene = scene.replace(structures=scene.structures[::-1])
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    terms = stack_terms(plan)
    assert len(terms) == len(scene.structures)
    assert tuple(t.member_index for t in terms) == tuple(range(len(terms)))
    assert len({t.name for t in terms}) == len(terms)
    owned = np.zeros(plan.shape, bool)
    for member, term in enumerate(terms):
        only = scene.replace(structures=(scene.structures[member],))
        one_plan = quiet(fd.plan, only)
        single = fd.lower(only, one_plan, rt)
        assert any(t.name == 'conductive[0]' for t in one_plan.terms)
        assert not stack_terms(one_plan), 'the existing singleton route remains selected'
        view = (slice(None),)+term.region.slices
        assert not np.any(owned[view])
        owned[view] = True
        for key in ('ae', 'be', 'ah', 'bh'):
            np.testing.assert_array_equal(np.broadcast_to(to_numpy(params[key]), plan.shape)[view],
                                          np.broadcast_to(to_numpy(single[key]), plan.shape)[view])
        for key, values in single['poles']['conductive[0]'].items():
            actual = params['poles'][term.name][key]
            for got, want in zip(actual, values) if isinstance(values, tuple) else ((actual, values),):
                np.testing.assert_array_equal(to_numpy(got), to_numpy(want))
    host_scene = scene.replace(structures=())
    host_plan = quiet(fd.plan, host_scene)
    host_params = fd.lower(host_scene, host_plan, rt)
    for key in ('ae', 'be'):
        np.testing.assert_array_equal(np.broadcast_to(to_numpy(params[key]), plan.shape)[~owned],
                                      np.broadcast_to(to_numpy(host_params[key]), plan.shape)[~owned])
    reversed_scene = scene.replace(structures=scene.structures[::-1])
    reversed_plan = quiet(fd.plan, reversed_scene)
    reversed_params = fd.lower(reversed_scene, reversed_plan, rt)
    assert terms[0].signature != stack_terms(reversed_plan)[0].signature
    for key in ('ae', 'be', 'ah', 'bh'):
        np.testing.assert_array_equal(to_numpy(params[key]), to_numpy(reversed_params[key]))
    state = initial_state(plan, params, rt)
    assert set(state['mem']) == {t.name for t in terms}
    def payload(tree):
        if isinstance(tree, dict):
            return sum(payload(v) for v in tree.values())
        if isinstance(tree, (tuple, list)):
            return sum(payload(v) for v in tree)
        return 0 if tree is None else to_numpy(tree).nbytes
    wanted = payload({key: state[key] for key in ('E', 'H', 'psi', 'mem')})
    assert report(scene, plan, params).bytes_per_state == wanted
    assert quiet(fd.check, scene).bytes_per_state == wanted
    assert 'CPML and material memory' in capsys.readouterr().out


def periodic_curl_e(field):
    values = []
    for c in range(3):
        a, b = (c+1) % 3, (c+2) % 3
        values.append(np.roll(field[b], -1, axis=2-a)-field[b]
                      -np.roll(field[a], -1, axis=2-b)+field[a])
    return np.asarray(values)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('pair', 'triple', 'all-inactive'))
def test_stack_observer_sums_local_storage_and_closes_the_yee_loss_budget(backend, axis, case):
    materials = ((4., 8.), (2.25, 4.), (3., 0.)) if case == 'triple' else ((4., 8.), (2.25, 4.))
    if case == 'all-inactive':
        materials = tuple((eps, 0.) for eps, _ in materials)
    scene = stack_scene(axis, materials)
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    terms = stack_terms(plan)
    eta, sigma, deltas, taus, shifts = independent_weights(scene, plan, axis)
    rng = np.random.default_rng(901)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    memory = {}
    density = (eta+sum(shifts))*abs(field)**2
    for term, delta, c in zip(terms, deltas, shifts):
        shape = (3,)+term.region.shape
        old = rng.normal(size=shape)+1j*rng.normal(size=shape)
        memory[term.name] = {key: (rt.array(old*factor, 'complex128'),)
                            for key, factor in zip(MEMORY_KEYS, (1.7, 1., .7))}
        inverse = np.divide(1., delta, out=np.zeros_like(delta), where=delta > 0)
        correction = inverse*abs(expand(old, plan, term)+c*field)**2-c*abs(field)**2
        density += correction
        cells = tuple((s.start, s.stop) for s in term.region.slices)
        cut = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in cells)
        for box, comps in ((cells, slice(None)), (cut, slice(axis, axis+1))):
            offsets = tuple(lo-s.start for (lo, _), s in zip(box, term.region.slices))
            got = InteriorEnergy._pole(memory[term.name], rt.array(field, 'complex128'),
                params['poles'][term.name], 1, box, offsets, plan.dt, rt, comps=comps)
            wanted = np.sum(correction[(comps,)+tuple(slice(lo, hi) for lo, hi in box)])
            assert float(to_numpy(got)) == pytest.approx(wanted, rel=5e-13, abs=5e-13)
    observed = plan.energy.electric(rt.array(field, 'complex128'), memory, params, rt)
    assert float(to_numpy(observed)) == pytest.approx(float(np.sum(density)), rel=5e-13, abs=5e-13)
    # This private consistent initialization is a kernel proof fixture, not public InitialField support.
    state = initial_state(plan, params, rt)
    state['E'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    state['H'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    for term in terms:
        state['mem'][term.name] = term.advance(term.init_mem(plan, rt), state['E'], state['H'], 0, params, rt)
    def storage(current):
        E = np.array(to_numpy(current['E']), copy=True)
        H = np.array(to_numpy(current['H']), copy=True)
        physical = []
        value = np.sum(eta*E**2+H*(H-plan.dt/plan.dx*periodic_curl_e(E)))
        electric = np.sum(eta*E**2)
        for term, delta, c in zip(terms, deltas, shifts):
            old = expand(np.array(to_numpy(current['mem'][term.name]['previous'][0]), copy=True), plan, term)
            polarization = old+c*E
            physical.append(polarization)
            inverse = np.divide(1., delta, out=np.zeros_like(delta), where=delta > 0)
            material_energy = np.sum(inverse*polarization**2)
            value += material_energy
            electric += material_energy
        observed = float(to_numpy(plan.energy.electric(current['E'], current['mem'], params, rt)))
        assert observed == pytest.approx(electric, rel=5e-13, abs=5e-13)
        return value, E, physical
    previous, old_e, old_p = storage(state)
    for n in range(4):
        state = step(plan, params, state, n, rt)
        current, new_e, new_p = storage(state)
        loss = 2*plan.dt*np.sum(sigma*((new_e+old_e)/2)**2)
        for old, new, delta, tau in zip(old_p, new_p, deltas, taus):
            resistance = np.divide(tau, delta, out=np.zeros_like(delta), where=delta > 0)
            loss += 2/plan.dt*np.sum(resistance*(new-old)**2)
        budget = 4096*np.finfo(float).eps*max(1., abs(previous), abs(current), loss)
        assert loss >= 0 and abs(current-previous+loss) <= budget
        assert loss == 0 if case == 'all-inactive' else loss > 0
        previous, old_e, old_p = current, new_e, new_p


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('family', ('lossy-pair', 'mixed-triple'))
def test_oblique_conductive_stack_matches_independent_multilayer_on_each_backend(backend, family):
    from studies.fdtd_conductive_stack_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review six disclosed calibration rows before accepting ceilings.'
    row = measure(n=40, pol='p', family=family, periods=120, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS
    assert max(row['error_r']) < r_limit and max(row['error_t']) < t_limit
    assert row['stable'] is True and max(row['decay'].values()) < 1e-5
    assert row['energy_shape'] == [row['energy_rows']]
    assert row['energy_finite'] is True and row['energy_max'] > 0


def replace_member(scene, index=1, shape=None, material=None, **kwargs):
    structures = list(scene.structures)
    previous = structures[index]
    structures[index] = fd.Structure(previous.shape if shape is None else shape,
        previous.material if material is None else material, **kwargs)
    return scene.replace(structures=tuple(structures))


REFUSALS = ('physical-overlap', 'padded-overlap', 'padded-overlap-inactive', 'off-grid-two-dx',
    'finite-box', 'sphere', 'turned', 'ranged', 'anisotropic-eps', 'anisotropic-sigma', 'magnetic',
    'lossy-host', 'conductive-zero-host', 'magnetic-host', 'dispersive-member', 'plain-material-member',
    'slab-member', 'nonparallel', 'initial', 'stencil', 'sub-dx', 'transverse-halo',
    'normal-absorber', 'transverse-absorber', 'coefficient-range', 'exact-support-touch')


@pytest.mark.parametrize('case', REFUSALS)
def test_stack_admission_requires_every_member_and_disjoint_padded_support(case):
    scene = stack_scene()
    body = scene.structures[1].shape
    if case in ('physical-overlap', 'padded-overlap', 'padded-overlap-inactive'):
        gap = -.05 if case == 'physical-overlap' else .1
        left = scene.structures[0].shape
        position = list(body.center)
        position[2] = left.center[2]+left.size[2]/2+gap+body.size[2]/2
        scene = replace_member(scene, shape=body.replace(center=tuple(position)))
        if case.endswith('inactive'):
            scene = scene.replace(structures=tuple(fd.Structure(s.shape, fd.Conductor(eps=s.material.eps, sigma=0.))
                                                   for s in scene.structures))
    elif case in ('off-grid-two-dx', 'exact-support-touch'):
        shift = .03125 if case == 'off-grid-two-dx' else 0.
        scene = stack_scene(dx=.125, centers=(-.5+shift, .25+shift), thicknesses=(.5, .5))
        if case == 'exact-support-touch':
            plan = quiet(fd.plan, scene)
            terms = sorted(stack_terms(plan), key=lambda term: term.region.z[0])
            assert len(terms) == 2 and terms[0].region.z[1] == terms[1].region.z[0]
            return
    elif case in ('finite-box', 'sphere'):
        scene = replace_member(scene, shape=fd.Box(size=(.2, .2, .38), center=body.center)
                               if case == 'finite-box' else fd.Sphere(radius=.15, center=body.center))
    elif case == 'turned':
        scene = replace_member(scene, shape=body.replace(angle=15.))
    elif case == 'ranged':
        scene = replace_member(scene, shape=body.replace(range=((-1., 1.), (-1., 1.), (-1., 1.))))
    elif case in ('anisotropic-eps', 'anisotropic-sigma', 'magnetic'):
        own = fd.Conductor(eps=(2.25, 3., 2.25) if case == 'anisotropic-eps' else 2.25,
                           sigma=(4., 5., 4.) if case == 'anisotropic-sigma' else 4.,
                           mu=2. if case == 'magnetic' else 1.)
        scene = replace_member(scene, material=own)
    elif case in ('lossy-host', 'conductive-zero-host', 'magnetic-host'):
        host = fd.Material(eps=1., mu=2.) if case == 'magnetic-host' else fd.Conductor(
            eps=1., sigma=0. if case == 'conductive-zero-host' else .2)
        scene = scene.replace(background=host)
    elif case in ('dispersive-member', 'plain-material-member'):
        own = fd.Material(eps=2.25) if case == 'plain-material-member' else fd.Dispersive(
            eps_inf=2.25, poles=(fd.Debye(tau=.1, delta_eps=2.),))
        scene = replace_member(scene, material=own)
    elif case == 'slab-member':
        scene = replace_member(scene, shape=fd.Slab(z=(.265, .645)))
    elif case == 'nonparallel':
        scene = scene.replace(domain=fd.Domain(size=(1.2, 1.2, 4.)),
            structures=tuple(fd.Structure(s.shape.replace(size=(1.6, 1.6, s.shape.size[2])), s.material)
                             for s in scene.structures))
        scene = replace_member(scene, shape=fd.Box(size=(.34, 1.6, 4.4), center=(.03, 0., 0.)))
    elif case == 'initial':
        scene = scene.replace(initial=fd.InitialField(E=(0., 0., 1.)))
    elif case == 'stencil':
        scene = replace_member(scene, discretisation='stencil')
    elif case in ('sub-dx', 'transverse-halo'):
        scene = replace_member(scene, shape=body.replace(size=(.8, .8, .05) if case == 'sub-dx' else (.5, .8, .38)))
    elif case == 'normal-absorber':
        scene = replace_member(scene, shape=body.replace(center=(0., 0., 1.5)))
        scene = scene.replace(boundaries=scene.boundaries.replace(z=(fd.PEC(), fd.PML(cells=12, wavelength=1.))))
    elif case == 'transverse-absorber':
        scene = scene.replace(domain=fd.Domain(size=(2.8, .4, 4.)),
            structures=tuple(fd.Structure(s.shape.replace(size=(3.2, .8, s.shape.size[2])), s.material)
                             for s in scene.structures),
            boundaries=scene.boundaries.replace(x=(fd.PML(cells=12, wavelength=1.), fd.PEC())))
    elif case == 'coefficient-range':
        scene = replace_member(scene, material=fd.Conductor(eps=2.25, sigma=1e300))
    else:
        raise AssertionError(case)
    if case in ('physical-overlap', 'padded-overlap', 'padded-overlap-inactive', 'off-grid-two-dx'):
        from wavebit.fd._lower.conductive_stack import validate
        safe_grid = quiet(fd.plan, stack_scene(dx=scene.resolution.dx)).grid
        # Generic smoothed-neighbor refusals must not mask a missing stack support guard.
        with pytest.raises(ValueError, match='(?i)support .*overlap'):
            validate(scene, safe_grid)
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, scene)
    message = str(caught.value).lower()
    assert any(word in message for word in ('conduct', 'stack', 'film', 'smooth', 'absorbing', 'support', 'float64'))

@pytest.mark.parametrize('change', ('first-sigma', 'second-sigma', 'second-eps', 'center', 'thickness',
    'background', 'order', 'count', 'missing-second', 'duplicate-first', 'wrong-name', 'wrong-index',
    'wrong-region', 'ordinary-memory', 'singleton-plan', 'nonfilm-plan', 'unsupported-finite'))
def test_prepared_stack_plan_binds_the_entire_ordered_scene_and_memory(change):
    scene = stack_scene()
    plan = quiet(fd.plan, scene)
    terms = stack_terms(plan)
    other = scene
    if change == 'first-sigma':
        other = replace_member(scene, 0, material=fd.Conductor(eps=4., sigma=9.))
    elif change in ('second-sigma', 'second-eps'):
        other = replace_member(scene, material=fd.Conductor(eps=3. if change == 'second-eps' else 2.25,
                                                          sigma=5. if change == 'second-sigma' else 4.))
    elif change in ('center', 'thickness'):
        body = scene.structures[1].shape
        other = replace_member(scene, shape=body.replace(center=(0., 0., .465)) if change == 'center'
                               else body.replace(size=(.8, .8, .39)))
    elif change == 'background':
        other = scene.replace(background=fd.Material(eps=1.5))
    elif change == 'order':
        other = scene.replace(structures=scene.structures[::-1])
    elif change == 'count':
        other = scene.replace(structures=scene.structures[:1])
    elif change in ('singleton-plan', 'nonfilm-plan'):
        previous = scene.replace(structures=scene.structures[:1]) if change == 'singleton-plan' else scene.replace(structures=())
        plan = quiet(fd.plan, previous)
    elif change == 'unsupported-finite':
        other = replace_member(scene, shape=fd.Box(size=(.2, .2, .38), center=(0., 0., .455)))
    else:
        revised = list(plan.terms)
        index = revised.index(terms[1])
        if change == 'missing-second':
            revised.pop(index)
        elif change == 'duplicate-first':
            revised[index] = terms[0]
        elif change == 'wrong-name':
            revised[index] = dataclasses.replace(terms[1], name='conductive_stack[8]')
        elif change == 'wrong-index':
            revised[index] = dataclasses.replace(terms[1], member_index=0)
        elif change == 'wrong-region':
            revised[index] = dataclasses.replace(terms[1], region=terms[0].region)
        elif change == 'ordinary-memory':
            revised.append(PoleTerm(name='poles[0]', region=terms[0].region, slabs=terms[0].slabs, n_poles=1))
        else:
            raise AssertionError(change)
        plan = dataclasses.replace(plan, terms=tuple(revised))
    if change == 'ordinary-memory':
        from wavebit.fd._lower.conductive_stack import refuse_mismatch
        # Ordinary ADE lowering has another refusal; isolate the stack identity contract.
        with pytest.raises(ValueError, match='(?i)plan|signature|concrete|stack'):
            refuse_mismatch(other, plan)
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete|stack'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('second-sigma', 'second-eps', 'center', 'thickness', 'host'))
def test_traced_later_members_and_host_cannot_bypass_the_stack_door(mode, parameter):
    scene = stack_scene()
    plan = quiet(fd.plan, scene)
    initial = {'second-sigma': 4., 'second-eps': 2.25, 'center': .455, 'thickness': .38, 'host': 1.}[parameter]
    def lower(value, backend):
        body = scene.structures[1].shape
        if parameter == 'host':
            other = scene.replace(background=fd.Material(eps=value))
        elif parameter in ('center', 'thickness'):
            shape = body.replace(center=(0., 0., value)) if parameter == 'center' else body.replace(size=(.8, .8, value))
            other = replace_member(scene, shape=shape)
        else:
            own = fd.Conductor(eps=value if parameter == 'second-eps' else 2.25,
                               sigma=value if parameter == 'second-sigma' else 4.)
            other = replace_member(scene, material=own)
        return fd.lower(other, plan, backend)['be']
    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|conduct|stack|geometry'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                value = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                           torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|conduct|stack|geometry'):
                    lower(value, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|conduct|stack|geometry'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')


@pytest.mark.parametrize('case', ('later-fast', 'earlier-fast', 'four-members', 'all-inactive', 'subunit-eps'))
def test_stack_cfl_uses_the_maximum_local_member_rate_and_physical_curl_floor(case):
    materials = ((4., 8.), (2.25, 150.))
    if case == 'earlier-fast':
        materials = materials[::-1]
    elif case == 'four-members':
        materials = ((4., 8.), (2.25, 20.), (3., 150.), (2., 70.))
    elif case == 'all-inactive':
        materials = ((4., 0.), (2.25, 0.))
    elif case == 'subunit-eps':
        materials = ((4., 8.), (.64, 40.))
    scene = stack_scene(materials=materials, courant=.001)
    plan = quiet(fd.plan, scene)
    eta, _, deltas, taus, _ = independent_weights(scene, plan, 2)
    maxima = tuple(float(np.max(delta/tau**2/eta)) for delta, tau in zip(deltas, taus))
    maximum = max(maxima)
    minimum_eps = min(1., *(eps for eps, _ in materials))
    vacuum = math.sqrt(minimum_eps)*plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+maximum/4)
    actual, reason = checks.cfl_bound(scene, plan)
    assert reason is None and actual == pytest.approx(expected, rel=3e-13)
    if maximum:
        assert expected < .995*vacuum
        summed = 1/math.sqrt(1/vacuum**2+sum(maxima)/4)
        assert not math.isclose(expected, summed, rel_tol=1e-5), 'max and sum are distinguishable'
        unsafe = scene.replace(run=fd.Run(steps=5, courant=(expected/vacuum+1)/2*math.sqrt(minimum_eps)))
        with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
            quiet(fd.plan, unsafe)
    else:
        assert actual == pytest.approx(vacuum, rel=3e-13)


@pytest.mark.parametrize('case', ('band-second', 'monitor-second', 'resolution-second', 'pml-second',
    'resolved', 'traced-band-forward', 'traced-band-jax'))
def test_stack_spectral_guards_check_every_member_and_the_full_source_band(case):
    from wavebit.fd._lower.conductive_stack import validate
    scene = stack_scene(materials=((1., 1.), (1., 1.)), dx=.05 if case == 'band-second' else .025,
                        courant=.01)
    source = fd.Dipole('kick', center=(0., 0., -1.4), component='Ex',
                       band=fd.Band(wavelength=1., df=.1, cutoff=1e-6))
    scene = scene.replace(sources=(source,), monitors=(fd.Field('field', plane=fd.Plane(z=1.4),
        components=('Ex',), wavelengths=(.9, 1., 1.1)),))
    grid = quiet(fd.plan, scene).grid
    if case.startswith('traced-band'):
        def check(df):
            incoming = source.replace(band=fd.Band(wavelength=1., df=df, cutoff=1e-6))
            return validate(scene.replace(sources=(incoming,)), grid)
        if case == 'traced-band-jax':
            jax = pytest.importorskip('jax')
            with pytest.raises(ValueError, match='(?i)concrete|traced|band'):
                jax.jit(check)(.1)
        else:
            torch = pytest.importorskip('torch')
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(.1, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)concrete|traced|band'):
                    check(dual)
        return
    if case == 'band-second':
        scene = replace_member(scene, material=fd.Conductor(eps=1., sigma=30.))
        high = 2*math.pi*(1+math.sqrt(2*math.log(1e6))*.1)
        limit = 2*math.pi/(8*grid.dx)
        assert abs(high*np.sqrt(1+1j/high)) < limit
        assert abs(high*np.sqrt(1+30j/high)) > limit
        for omega in (high, *(2*math.pi/w for w in (.9, 1., 1.1))):
            k = omega*np.sqrt(1+30j/omega)
            assert 2*math.pi/(grid.dx*k.real) >= 8 and 1/(grid.dx*k.imag) >= 1
        with pytest.raises(ValueError, match='(?i)source .*band.*spectral|source .*band.*mesh'):
            validate(scene, grid)
    elif case in ('monitor-second', 'resolution-second', 'pml-second'):
        scene = replace_member(scene, material=fd.Conductor(eps=9., sigma=4.))
        if case == 'monitor-second':
            scene = scene.replace(monitors=(scene.monitors[0].replace(wavelengths=(.35,)),))
        elif case == 'resolution-second':
            scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=14, wavelength=.35))
        else:
            scene = scene.replace(boundaries=scene.boundaries.replace(z=fd.PML(cells=12, wavelength=.35)))
        assert .35/grid.dx > 8, 'the host and first member do not cause this point refusal'
        with pytest.raises(ValueError, match='(?i)wavelength|skin|spectral|cells'):
            validate(scene, grid)
    else:
        assert validate(scene, grid) is True
        assert quiet(fd.plan, scene).dt > 0
        return
    with pytest.raises(ValueError, match='(?i)wavelength|skin|spectral|cells|band'):
        quiet(fd.plan, scene)
