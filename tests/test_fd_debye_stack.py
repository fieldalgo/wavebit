"""Separated Debye-stack gates with independent response and storage references.

Optical ceilings follow the separately preserved six-row calibration disclosure.
Singleton tests are imported only for independent interval/Decimal instruments.
"""
import dataclasses
import math

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._check import report
from wavebit.fd._kernel.constitutive import ConstitutiveTerm
from wavebit.fd._kernel.observe import InteriorEnergy
from wavebit.fd._kernel.poles import PoleTerm
from wavebit.fd._kernel.step import initial_state, step
from wavebit.fd._lower import checks
from wavebit.fd._runtime import runtime
from fd_helpers import backends, to_numpy
from test_fd_conductive_stack import quiet, member_fills, expand, periodic_curl_e
from test_fd_debye_sum_film import rational_modes

PREFIX = 'debye_stack['
MEMORY_KEYS = ('P', 'previous', 'delta')
# Inherited independent discrete-response oracle budget: test_fd_debye_sum_film.py,
# test_debye_sum_discrete_response_is_the_declared_sum_mixture (3e-12).
RESPONSE_TOL = 3e-12
# Inherited storage/closed-Yee budgets: test_fd_conductive_stack.py,
# test_stack_observer_sums_local_storage_and_closes_the_yee_loss_budget.
STORAGE_TOL = 5e-13
STEP_ULPS = 4096
# Inherited summed-moment budget: test_fd_debye_sum_film.py,
# test_debye_sum_cfl_uses_the_local_sum_and_harmonic_high_frequency_moment.
CFL_TOL = 2e-13
CALIBRATED_CEILINGS = (.005, .01)


def material(eps=4., deltas=(6.,), taus=(.15,)):
    assert len(deltas) == len(taus) > 0
    return fd.Dispersive(eps_inf=eps, poles=tuple(
        fd.Debye(delta_eps=d, tau=t) for d, t in zip(deltas, taus)))


def specifications(case):
    first = (4., (6.,), (.15,))
    second = (2.25, (2., 4.), (.05, .2))
    third = (2., (1., 1., 2., 2.), (.04, .08, .16, .32))
    if case == 'triple':
        return first, second, third
    if case == 'four':
        return first, second, (3., (1., 2., 3.), (.05, .12, .3)), third
    if case == 'eight-poles':
        second = (3., (.75,)*8, tuple(.025*1.5**j for j in range(8)))
    elif case == 'equal-times':
        second = (2.25, (1., 1., 2., 2.), (.15,)*4)
    elif case == 'single-active':
        second = (2.25, (0., 0., 6., 0.), (1e-250, 1e250, .15, 1e200))
    elif case == 'adjacent-times':
        second = (2.25, (1., 1., 2., 2.), (.125, math.nextafter(.125, 1.), .25, .5))
    elif case == 'tiny-residue':
        second = (2.25, (1., 1e-24, 2., 2.), (.04, .08, .16, .32))
    elif case == 'all-inactive':
        first = (4., (0.,), (1e-250,))
        second = (2.25, (0.,)*4, (1e-250, 1e250, 1e-200, 1e200))
    return first, second


def stack_scene(axis=2, specs=None, host=1., dx=.1, courant=.2, centers=None,
                thicknesses=None, complex_fields=False):
    specs = specifications('pair') if specs is None else specs
    lengths = [.4]*3
    lengths[axis] = max(4., len(specs)*1.2)
    centers = tuple((j-(len(specs)-1)/2)*.85+.03 for j in range(len(specs))) if centers is None else centers
    thicknesses = tuple(.34+.04*(j % 3) for j in range(len(specs))) if thicknesses is None else thicknesses
    structures = []
    for spec, center, thickness in zip(specs, centers, thicknesses):
        size, position = [length+4*dx for length in lengths], [0., 0., 0.]
        size[axis], position[axis] = thickness, center
        structures.append(fd.Structure(fd.Box(size=tuple(size), center=tuple(position)), material(*spec)))
    faces = [fd.Periodic(), fd.Periodic(), fd.Periodic()]
    if complex_fields:
        faces[(axis+1) % 3] = fd.Bloch(phase=.31)
    return fd.Scene(domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(**dict(zip('xyz', faces))), resolution=fd.Resolution(dx=dx),
        run=fd.Run(steps=5, courant=courant), background=fd.Material(eps=host), structures=tuple(structures))


def stack_terms(plan):
    return tuple(t for t in plan.terms if t.name.startswith(PREFIX))


def replace_member(scene, index=1, shape=None, own=None, **kwargs):
    structures = list(scene.structures)
    old = structures[index]
    structures[index] = fd.Structure(old.shape if shape is None else shape,
                                    old.material if own is None else own, **kwargs)
    return scene.replace(structures=tuple(structures))


def declared(own):
    return float(own.eps_inf[0]), tuple(float(p.delta_eps) for p in own.poles), tuple(float(p.tau) for p in own.poles)


def rational_eps(own, omega):
    a, deltas, taus = declared(own)
    return a+sum(delta/(1-1j*omega*tau) for delta, tau in zip(deltas, taus) if delta)


def actual_response(plan, params, axis, angle):
    """Production advance on a basis, with an independent z-domain solve."""
    rt, z = runtime('numpy'), np.exp(-1j*angle)
    response = np.array(plan.dt/plan.dx/np.broadcast_to(to_numpy(params['be']), plan.shape), complex)
    np.testing.assert_array_equal(to_numpy(params['ae']), 1.)
    current, expected_current = np.zeros(plan.shape, complex), np.zeros(plan.shape, complex)
    for term in stack_terms(plan):
        count, shape = term.n_poles, (3,)+term.region.shape
        cut = [slice(None), 0, 0, 0]
        cut[3-axis] = slice(None)
        cut = tuple(cut)
        def memory(basis=None):
            return {key: tuple(np.full(shape, float(basis == k*count+j), complex)
                for j in range(count)) for k, key in enumerate(MEMORY_KEYS)}
        def sample(mem):
            return np.stack([to_numpy(mem[key][j])[cut] for key in MEMORY_KEYS for j in range(count)])
        zero = np.zeros(plan.shape, complex)
        A = np.stack([sample(term.advance(memory(j), zero, None, 0, params, rt))
                      for j in range(3*count)], axis=1)
        B = sample(term.advance(memory(), np.ones(plan.shape, complex), None, 0, params, rt))
        identity = np.eye(3*count).reshape((3*count, 3*count)+(1,)*(A.ndim-2))
        matrix = np.moveaxis(z*identity-A, (0, 1), (-2, -1))
        X = np.moveaxis(np.linalg.solve(matrix, np.moveaxis(z*B, 0, -1)[..., None])[..., 0], -1, 0)
        compact = [3, 1, 1, 1]
        compact[3-axis] = X.shape[-1]
        local = {key: tuple(np.broadcast_to(X[k*count+j].reshape(compact), shape).copy()
                 for j in range(count)) for k, key in enumerate(MEMORY_KEYS)}
        view = (slice(None),)+term.region.slices
        response[view] += sum(local['previous'])
        expected_current[view] += -(plan.dx/plan.dt)*(z-1)*sum(local['previous'])
        current = term.before_E(current, None, local, 0, params, rt)
        np.testing.assert_allclose(X[:count], z*X[count:2*count], rtol=RESPONSE_TOL, atol=RESPONSE_TOL)
    np.testing.assert_allclose(current, expected_current, rtol=RESPONSE_TOL, atol=RESPONSE_TOL)
    return response, 2*math.tan(angle/2)/plan.dt


@pytest.mark.parametrize('axis', (0, 1, 2), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('pair', 'triple', 'four', 'eight-poles', 'equal-times',
    'single-active', 'adjacent-times', 'tiny-residue', 'all-inactive', 'tiny-fill'))
def test_each_member_advance_matches_the_declared_series_parallel_response(axis, case):
    kwargs = {'dx': .125, 'centers': (-.625, .125+2.**-42), 'thicknesses': (.5, .5)} if case == 'tiny-fill' else {}
    scene = stack_scene(axis, specifications(case), **kwargs)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    terms = stack_terms(plan)
    assert [t.name for t in terms] == [f'{PREFIX}{j}]' for j in range(len(scene.structures))]
    for member, term in enumerate(terms):
        own = scene.structures[member].material
        assert term.n_poles == len(own.poles)
        coefficients = params['poles'][term.name]
        assert all(len(coefficients[k]) == term.n_poles for k in
            ('step_gain', 'hold', 'drive', 'energy_inverse_delta', 'energy_instantaneous'))
        local = member_fills(scene, plan, axis, member)[(slice(None),)+term.region.slices]
        drives = sum(np.broadcast_to(to_numpy(d), local.shape) for d in coefficients['drive'])
        np.testing.assert_array_equal(drives > 0, (local > 0) & any(p.delta_eps > 0 for p in own.poles))
        if case == 'tiny-fill' and member == 1:
            tiny = (local > 0) & (local < 1e-9)
            assert np.any(tiny) and np.all(drives[tiny] > 0)
    host = float(scene.background.eps[0])
    for angle in (.071, .39, 1.2):
        got, omega = actual_response(plan, params, axis, angle)
        expected = np.full(plan.shape, complex(host))
        for member, structure in enumerate(scene.structures):
            f, own = member_fills(scene, plan, axis, member), rational_eps(structure.material, omega)
            local = f*own+(1-f)*host
            local[axis] = 1/(f[axis]/own+(1-f[axis])/host)
            expected[f > 0] = local[f > 0]
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, expected, rtol=RESPONSE_TOL, atol=RESPONSE_TOL)

@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('case', ('pair', 'triple', 'equal-times', 'single-active', 'all-inactive', 'complex'))
def test_composition_preserves_singletons_declared_slots_and_actual_allocation(backend, case, capsys):
    scene = stack_scene(specs=specifications(case), complex_fields=case == 'complex')
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params, terms = fd.lower(scene, plan, rt), stack_terms(plan)
    assert len(terms) == len(scene.structures)
    assert tuple(t.member_index for t in terms) == tuple(range(len(terms)))
    owned = np.zeros(plan.shape, bool)
    for member, term in enumerate(terms):
        only = scene.replace(structures=(scene.structures[member],))
        one_plan = quiet(fd.plan, only)
        single = fd.lower(only, one_plan, rt)
        count = len(only.structures[0].material.poles)
        key = {1: 'debye_film[0]', 2: 'two_debye_film[0]'}.get(count, 'debye_sum_film[0]')
        assert not stack_terms(one_plan)
        assert [t.name for t in one_plan.terms if isinstance(t, PoleTerm)] == [key]
        view = (slice(None),)+term.region.slices
        assert not np.any(owned[view])
        owned[view] = True
        for coefficient in ('ae', 'be', 'ah', 'bh'):
            np.testing.assert_array_equal(np.broadcast_to(to_numpy(params[coefficient]), plan.shape)[view],
                np.broadcast_to(to_numpy(single[coefficient]), plan.shape)[view])
        for coefficient, values in single['poles'][key].items():
            actual = params['poles'][term.name][coefficient]
            for got, want in zip(actual, values) if isinstance(values, tuple) else ((actual, values),):
                np.testing.assert_array_equal(to_numpy(got), to_numpy(want))
        active_count = len({float(p.tau) for p in only.structures[0].material.poles if p.delta_eps})
        for coefficient in ('drive', 'energy_inverse_delta', 'energy_instantaneous'):
            assert all(np.count_nonzero(to_numpy(v)) == 0 for v in params['poles'][term.name][coefficient][active_count:])
    empty = scene.replace(structures=())
    empty_plan = quiet(fd.plan, empty)
    empty_params = fd.lower(empty, empty_plan, rt)
    for coefficient in ('ae', 'be'):
        np.testing.assert_array_equal(np.broadcast_to(to_numpy(params[coefficient]), plan.shape)[~owned],
            np.broadcast_to(to_numpy(empty_params[coefficient]), plan.shape)[~owned])
    reverse = scene.replace(structures=scene.structures[::-1])
    reverse_plan = quiet(fd.plan, reverse)
    reverse_params = fd.lower(reverse, reverse_plan, rt)
    assert terms[0].signature != stack_terms(reverse_plan)[0].signature
    for coefficient in ('ae', 'be', 'ah', 'bh'):
        np.testing.assert_array_equal(to_numpy(params[coefficient]), to_numpy(reverse_params[coefficient]))
    state = initial_state(plan, params, rt)
    assert set(state['mem']) == {t.name for t in terms}
    for term in terms:
        assert set(state['mem'][term.name]) == set(MEMORY_KEYS)
        assert all(len(state['mem'][term.name][k]) == term.n_poles for k in MEMORY_KEYS)
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


def independent_weights(scene, plan, axis):
    """Decimal polynomial residues of each declared rational mixture, no reducer calls."""
    host = float(scene.background.eps[0])
    eta = np.full(plan.shape, host)
    members = []
    for index, structure in enumerate(scene.structures):
        a, ds, ts = declared(structure.material)
        fill = member_fills(scene, plan, axis, index)
        own = fill*a+(1-fill)*host
        own[axis] = 1/(fill[axis]/a+(1-fill[axis])/host)
        eta[fill > 0] = own[fill > 0]
        deltas = np.zeros((len(ds),)+plan.shape)
        taus = np.ones_like(deltas)
        for component in range(3):
            for fraction in np.unique(fill[component]):
                selected = fill[component] == fraction
                for j, (delta, tau) in enumerate(rational_modes(fraction, component == axis, a, host, ds, ts)):
                    deltas[j, component][selected] = delta
                    taus[j, component][selected] = tau
        members.append((deltas, taus, deltas*plan.dt/(2*taus+plan.dt)))
    return eta, members


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('pair', 'triple', 'all-inactive'))
def test_all_members_storage_and_closed_yee_dissipation_use_independent_residues(backend, axis, case):
    scene = stack_scene(axis, specifications(case))
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params, terms = fd.lower(scene, plan, rt), stack_terms(plan)
    eta, members = independent_weights(scene, plan, axis)
    rng = np.random.default_rng(901)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    density = eta*abs(field)**2
    memory = {}
    for term, (deltas, _, shifts) in zip(terms, members):
        old = tuple(rng.normal(size=(3,)+term.region.shape)+1j*rng.normal(size=(3,)+term.region.shape)
                    for _ in range(term.n_poles))
        memory[term.name] = {key: tuple(rt.array(v*factor, 'complex128') for v in old)
                            for key, factor in zip(MEMORY_KEYS, (1.7, 1., .7))}
        correction = np.zeros(plan.shape)
        for value, delta, c in zip(old, deltas, shifts):
            inverse = np.divide(1., delta, out=np.zeros_like(delta), where=delta > 0)
            physical = inverse*abs(expand(value, plan, term)+c*field)**2
            density += physical
            correction += physical-c*abs(field)**2
        cells = tuple((s.start, s.stop) for s in term.region.slices)
        crop = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in cells)
        for box, comps in ((cells, slice(None)), (crop, slice(axis, axis+1))):
            offsets = tuple(lo-s.start for (lo, _), s in zip(box, term.region.slices))
            got = InteriorEnergy._pole(memory[term.name], rt.array(field, 'complex128'),
                params['poles'][term.name], term.n_poles, box, offsets, plan.dt, rt, comps=comps)
            want = np.sum(correction[(comps,)+tuple(slice(lo, hi) for lo, hi in box)])
            assert float(to_numpy(got)) == pytest.approx(want, rel=STORAGE_TOL, abs=STORAGE_TOL)
    observed = plan.energy.electric(rt.array(field, 'complex128'), memory, params, rt)
    assert float(to_numpy(observed)) == pytest.approx(float(np.sum(density)), rel=STORAGE_TOL, abs=STORAGE_TOL)
    # Consistent private kernel initialization; public InitialField remains refused.
    state = initial_state(plan, params, rt)
    state['E'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    state['H'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    for term in terms:
        state['mem'][term.name] = term.advance(term.init_mem(plan, rt), state['E'], state['H'], 0, params, rt)
    def storage(current):
        E, H = (np.array(to_numpy(current[k]), copy=True) for k in ('E', 'H'))
        value = np.sum(eta*E**2+H*(H-plan.dt/plan.dx*periodic_curl_e(E)))
        electric, physical = np.sum(eta*E**2), []
        for term, (deltas, _, shifts) in zip(terms, members):
            member = []
            for j, (delta, c) in enumerate(zip(deltas, shifts)):
                old = expand(np.array(to_numpy(current['mem'][term.name]['previous'][j]), copy=True), plan, term)
                polarization = old+c*E
                member.append(polarization)
                inverse = np.divide(1., delta, out=np.zeros_like(delta), where=delta > 0)
                own_energy = np.sum(inverse*polarization**2)
                value += own_energy
                electric += own_energy
            physical.append(member)
        got = float(to_numpy(plan.energy.electric(current['E'], current['mem'], params, rt)))
        assert got == pytest.approx(electric, rel=STORAGE_TOL, abs=STORAGE_TOL)
        return value, physical
    previous, old_p = storage(state)
    for n in range(4):
        state = step(plan, params, state, n, rt)
        current, new_p = storage(state)
        loss = 0.
        for olds, news, (deltas, taus, _) in zip(old_p, new_p, members):
            for old, new, delta, tau in zip(olds, news, deltas, taus):
                resistance = np.divide(tau, delta, out=np.zeros_like(delta), where=delta > 0)
                loss += 2/plan.dt*np.sum(resistance*(new-old)**2)
        budget = STEP_ULPS*np.finfo(float).eps*max(1., abs(previous), abs(current), loss)
        assert loss >= 0 and abs(current-previous+loss) <= budget
        assert loss == 0 if case == 'all-inactive' else loss > 0
        previous, old_p = current, new_p


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('family', ('debye-pair', 'debye-triple'))
def test_oblique_debye_stack_matches_declared_multilayer_on_each_backend(backend, family):
    from studies.fdtd_debye_stack_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review six disclosed rows before accepting optical ceilings.'
    row = measure(n=40, pol='p', family=family, periods=120, fraction=.3, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS
    assert max(row['error_r']) < r_limit and max(row['error_t']) < t_limit
    assert row['stable'] is True and row['energy_finite'] is True and row['energy_max'] > 0
    assert row['energy_shape'] == [row['energy_rows']]

REFUSALS = ('physical-overlap', 'padded-overlap', 'all-zero-overlap', 'off-grid-two-dx',
    'exact-support-touch', 'finite-box', 'turned', 'ranged', 'anisotropic', 'magnetic',
    'lossy-host', 'conductive-zero-host', 'oscillator-member', 'conductor-member', 'plain-member',
    'slab-member', 'nonparallel', 'initial', 'decay', 'stencil', 'sub-dx', 'transverse-halo',
    'normal-absorber', 'transverse-absorber', 'range-later', 'certificate-later')


@pytest.mark.parametrize('case', REFUSALS)
def test_every_member_must_pass_scope_and_exact_padded_support_separation(case, monkeypatch):
    from wavebit.fd._lower.debye_stack import validate
    scene = stack_scene()
    body = scene.structures[1].shape
    safe_grid = quiet(fd.plan, scene).grid
    if case in ('physical-overlap', 'padded-overlap', 'all-zero-overlap'):
        gap = -.05 if case == 'physical-overlap' else .1
        left = scene.structures[0].shape
        center = left.center[2]+left.size[2]/2+gap+body.size[2]/2
        scene = replace_member(scene, shape=body.replace(center=(0., 0., center)))
        if case == 'all-zero-overlap':
            scene = scene.replace(structures=tuple(fd.Structure(s.shape, material(
                float(s.material.eps_inf[0]), (0.,)*len(s.material.poles), (1e-250,)*len(s.material.poles)))
                for s in scene.structures))
    elif case in ('off-grid-two-dx', 'exact-support-touch'):
        shift = .03125 if case == 'off-grid-two-dx' else 0.
        scene = stack_scene(dx=.125, centers=(-.5+shift, .25+shift), thicknesses=(.5, .5))
        safe_grid = quiet(fd.plan, stack_scene(dx=.125)).grid
        if case == 'exact-support-touch':
            plan = quiet(fd.plan, scene)
            terms = sorted(stack_terms(plan), key=lambda t: t.region.z[0])
            assert len(terms) == 2 and terms[0].region.z[1] == terms[1].region.z[0]
            return
    elif case == 'finite-box':
        scene = replace_member(scene, shape=fd.Box(size=(.2, .2, .38), center=body.center))
    elif case == 'turned':
        scene = replace_member(scene, shape=body.replace(angle=15.))
    elif case == 'ranged':
        scene = replace_member(scene, shape=body.replace(range=((-1., 1.),)*3))
    elif case in ('anisotropic', 'magnetic'):
        scene = replace_member(scene, own=fd.Dispersive(
            eps_inf=(2.25, 3., 2.25) if case == 'anisotropic' else 2.25,
            mu=2. if case == 'magnetic' else 1., poles=(fd.Debye(tau=.15, delta_eps=6.),)))
    elif case in ('lossy-host', 'conductive-zero-host'):
        scene = scene.replace(background=fd.Conductor(eps=1., sigma=.2 if case == 'lossy-host' else 0.))
    elif case == 'oscillator-member':
        scene = replace_member(scene, own=fd.Dispersive(eps_inf=2.25, poles=(fd.Drude(omega_p=3., gamma=1.),)))
    elif case == 'conductor-member':
        scene = replace_member(scene, own=fd.Conductor(eps=2.25, sigma=1.))
    elif case == 'plain-member':
        scene = replace_member(scene, own=fd.Material(eps=2.25))
    elif case == 'slab-member':
        scene = replace_member(scene, shape=fd.Slab(z=(.265, .645)))
    elif case == 'nonparallel':
        scene = scene.replace(domain=fd.Domain(size=(1.2, 1.2, 4.)), structures=tuple(
            fd.Structure(s.shape.replace(size=(1.6, 1.6, s.shape.size[2])), s.material) for s in scene.structures))
        scene = replace_member(scene, shape=fd.Box(size=(.34, 1.6, 4.4), center=(.03, 0., 0.)))
    elif case == 'initial':
        scene = scene.replace(initial=fd.InitialField(E=(0., 0., 1.)))
    elif case == 'decay':
        scene = scene.replace(run=fd.Run(steps=5, courant=.2, decay=fd.Decay(threshold=.1, window=.1)))
        with pytest.raises(ValueError, match='DebyeStackDecayRefused'):
            validate(scene, safe_grid)
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
    elif case == 'range-later':
        scene = replace_member(scene, own=material(4., (math.ulp(0.),)*3, (.2,)*3))
    elif case == 'certificate-later':
        from wavebit.fd._lower import debye_sum_film
        scene = replace_member(scene, own=material(2., (1., 1., 2., 2.), (.04, .08, .16, .32)))
        debye_sum_film._normal_modes.cache_clear()
        monkeypatch.setattr(debye_sum_film, '_MAX_BISECTIONS', 0)
        try:
            with pytest.raises(ValueError, match='(?i)certif|bisection'):
                plan = quiet(fd.plan, scene)
                fd.lower(scene, plan, 'numpy')
        finally:
            debye_sum_film._normal_modes.cache_clear()
        return
    else:
        raise AssertionError(case)
    if case in ('physical-overlap', 'padded-overlap', 'all-zero-overlap', 'off-grid-two-dx'):
        # Isolate the stack door; generic neighbor checks must not mask its removal.
        with pytest.raises(ValueError, match='DebyeStackOverlap'):
            validate(scene, safe_grid)
    with pytest.raises(ValueError, match='(?i)debye|dispers|film|smooth|absorbing|support|float64|decay'):
        plan = quiet(fd.plan, scene)
        fd.lower(scene, plan, 'numpy')


CHANGES = ('later-delta', 'later-tau', 'later-count', 'pole-order', 'eps', 'center', 'host', 'order',
    'missing-later', 'duplicate-first', 'wrong-name', 'wrong-index', 'ordinary-memory', 'foreign-flux',
    'singleton-plan', 'nonfilm-plan', 'added-source', 'removed-source', 'source-component',
    'boundary', 'wrong-region', 'wrong-count')


@pytest.mark.parametrize('change', CHANGES)
def test_prepared_stack_binds_all_members_memories_and_source_allocation(change):
    from wavebit.fd._lower.debye_stack import refuse_mismatch
    scene = stack_scene()
    source = fd.Dipole('kick', center=(0., 0., -1.4), component='Ex', kind='electric',
                       band=fd.Band(wavelength=4., df=.025, cutoff=1e-6))
    if change in ('removed-source', 'source-component'):
        scene = scene.replace(sources=(source,))
    plan = quiet(fd.plan, scene)
    terms, other = stack_terms(plan), scene
    if change in ('later-delta', 'later-tau', 'later-count', 'pole-order', 'eps'):
        own = scene.structures[1].material
        poles = list(own.poles)
        if change in ('later-delta', 'later-tau'):
            poles[-1] = fd.Debye(delta_eps=5. if change == 'later-delta' else 4.,
                                 tau=.25 if change == 'later-tau' else .2)
        elif change == 'later-count':
            poles.append(fd.Debye(delta_eps=0., tau=.3))
        elif change == 'pole-order':
            poles.reverse()
        other = replace_member(scene, own=fd.Dispersive(eps_inf=3. if change == 'eps' else 2.25, poles=tuple(poles)))
    elif change == 'center':
        other = replace_member(scene, shape=scene.structures[1].shape.replace(center=(0., 0., .465)))
    elif change == 'boundary':
        other = scene.replace(boundaries=scene.boundaries.replace(z=fd.PEC()))
    elif change == 'host':
        other = scene.replace(background=fd.Material(eps=1.5))
    elif change == 'order':
        other = scene.replace(structures=scene.structures[::-1])
    elif change in ('singleton-plan', 'nonfilm-plan'):
        plan = quiet(fd.plan, scene.replace(structures=scene.structures[:1] if change == 'singleton-plan' else ()))
    elif change == 'added-source':
        other = scene.replace(sources=(source,))
    elif change == 'removed-source':
        other = scene.replace(sources=())
    elif change == 'source-component':
        other = scene.replace(sources=(source.replace(component='Ey'),))
    else:
        revised = list(plan.terms)
        index = revised.index(terms[1])
        if change == 'missing-later':
            revised.pop(index)
        elif change == 'duplicate-first':
            revised[index] = terms[0]
        elif change == 'wrong-name':
            revised[index] = dataclasses.replace(terms[1], name='debye_stack[8]')
        elif change == 'wrong-index':
            revised[index] = dataclasses.replace(terms[1], member_index=0)
        elif change == 'wrong-region':
            revised[index] = dataclasses.replace(terms[1], region=terms[0].region)
        elif change == 'wrong-count':
            revised[index] = dataclasses.replace(terms[1], n_poles=terms[1].n_poles+1)
        elif change == 'ordinary-memory':
            revised.append(PoleTerm(name='poles[0]', region=terms[0].region, slabs=terms[0].slabs, n_poles=1))
        elif change == 'foreign-flux':
            revised.append(ConstitutiveTerm(name='constitutive[E,9]', field='E', region=terms[0].region, ratio=plan.dt/plan.dx))
        else:
            raise AssertionError(change)
        plan = dataclasses.replace(plan, terms=tuple(revised))
    with pytest.raises(ValueError, match='DebyeStackPlan|DebyeStackScope'):
        refuse_mismatch(other, plan)
    with pytest.raises(ValueError, match='DebyeStackPlan|DebyeStackScope'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('last-delta', 'last-tau', 'eps', 'center', 'host'))
def test_traced_later_poles_geometry_and_host_cannot_detach_into_stack_admission(mode, parameter):
    scene = stack_scene(specs=specifications('triple'))
    plan = quiet(fd.plan, scene)
    initial = {'last-delta': 2., 'last-tau': .32, 'eps': 2., 'center': .88, 'host': 1.}[parameter]
    def lower(value, backend):
        old = scene.structures[-1]
        if parameter == 'host':
            other = scene.replace(background=fd.Material(eps=value))
        elif parameter == 'center':
            other = replace_member(scene, 2, shape=old.shape.replace(center=(0., 0., value)))
        else:
            ds, ts = [1., 1., 2., 2.], [.04, .08, .16, .32]
            if parameter == 'last-delta':
                ds[-1] = value
            elif parameter == 'last-tau':
                ts[-1] = value
            other = replace_member(scene, 2, own=material(value if parameter == 'eps' else 2., tuple(ds), tuple(ts)))
        return fd.lower(other, plan, backend)['be']
    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|debye|stack|geometry'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                value = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                           torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|debye|stack|geometry'):
                    lower(value, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|debye|stack|geometry'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')

@pytest.mark.parametrize('case', ('later-fast', 'earlier-fast', 'four-members', 'all-inactive', 'subunit-eps'))
def test_cfl_uses_summed_local_moments_then_the_maximum_over_separated_members(case):
    specs = ((4., (1.,), (.2,)), (1., (3., 3., 3.), (.01,)*3))
    if case == 'earlier-fast':
        specs = specs[::-1]
    elif case == 'four-members':
        specs = (*specs, (3., (2., 1.), (.015, .05)), (2., (1.,)*4, (.02, .03, .04, .05)))
    elif case == 'all-inactive':
        specs = specifications(case)
    elif case == 'subunit-eps':
        specs = (specs[0], (.64, specs[1][1], specs[1][2]))
    scene = stack_scene(specs=specs, courant=.001)
    plan = quiet(fd.plan, scene)
    b = float(scene.background.eps[0])
    maxima = []
    for member, (a, deltas, taus) in enumerate(specs):
        f = member_fills(scene, plan, 2, member)
        # Independent high-frequency moment; no normal-mode reducer is consulted.
        M1 = sum(d/t for d, t in zip(deltas, taus) if d)
        M2 = sum(d/t**2 for d, t in zip(deltas, taus) if d)
        A = f*b+(1-f)*a
        alpha, kappa, eta = (1-f)/A, f*b*b/A**2, a*b/A
        rates = f*M2/(f*a+(1-f)*b)
        rates[2] = (kappa*(M2+alpha*M1**2)/eta)[2]
        maxima.append(max(M2/a, float(np.max(rates))))
    maximum = max(maxima)
    vacuum = math.sqrt(min(b, *(s[0] for s in specs)))*plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+maximum/4)
    actual, reason = checks.cfl_bound(scene, plan)
    assert reason is None and actual == pytest.approx(expected, rel=CFL_TOL)
    if maximum:
        assert expected < .995*vacuum
        summed = 1/math.sqrt(1/vacuum**2+sum(maxima)/4)
        assert not math.isclose(expected, summed, rel_tol=CFL_TOL)
    else:
        assert actual == pytest.approx(vacuum, rel=CFL_TOL)
    # Also exercises the diagnostic's inactive extreme-tau path, not only its bound.
    unsafe = dataclasses.replace(plan, dt=2*expected)
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        checks.check_stability(scene, unsafe)


@pytest.mark.parametrize('case', ('band-later', 'monitor-later', 'resolution-later', 'pml-later',
    'resolved', 'traced-band-forward', 'traced-band-jax'))
def test_spectral_certificate_checks_every_declared_pole_and_later_member(case):
    from wavebit.fd._lower.debye_stack import validate
    specs = ((1., (0.,), (.15,)), (1., (0.,)*3, (.15,)*3))
    scene = stack_scene(specs=specs, dx=.05 if case == 'band-later' else .025, courant=.01)
    source = fd.Dipole('kick', center=(0., 0., -1.4), component='Ex', kind='electric',
                       band=fd.Band(wavelength=1., df=.1, cutoff=1e-6))
    scene = scene.replace(sources=(source,), monitors=(fd.Field('field', plane=fd.Plane(z=1.4),
        components=('Ex',), wavelengths=(.9, 1., 1.1)),))
    original_plan = quiet(fd.plan, scene)
    grid = original_plan.grid
    if case.startswith('traced-band'):
        def check(df, backend):
            incoming = source.replace(band=fd.Band(wavelength=1., df=df, cutoff=1e-6))
            other = scene.replace(sources=(incoming,))
            assert validate(other, grid) is False, 'tracing is outside the structural stack descriptor'
            return fd.lower(other, original_plan, backend)['be']
        if case == 'traced-band-jax':
            jax = pytest.importorskip('jax')
            with pytest.raises(ValueError, match='DebyeStackPlan|DebyeStackScope'):
                jax.jit(lambda df: check(df, 'jax'))(.1)
        else:
            torch = pytest.importorskip('torch')
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(.1, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='DebyeStackPlan|DebyeStackScope'):
                    check(dual, 'torch')
        return
    if case == 'band-later':
        scene = replace_member(scene, own=material(1., (2., 2., 2.), (.15,)*3))
        high = 2*math.pi*(1+math.sqrt(2*math.log(1e6))*.1)
        limit = 2*math.pi/(8*grid.dx)
        assert high*abs(1+2/(1-1j*high*.15))**.5 < limit
        assert high*abs(1+6/(1-1j*high*.15))**.5 > limit
        # Every explicitly declared point passes; a point door cannot mask this witness.
        for wavelength in (.9, 1., 1.1):
            omega = 2*math.pi/wavelength
            k = omega*np.sqrt(rational_eps(scene.structures[1].material, omega))
            assert 2*math.pi/(grid.dx*k.real) >= 8 and 1/(grid.dx*k.imag) >= 1
        with pytest.raises(ValueError, match='DebyeStackSpectral'):
            validate(scene, grid)
    elif case in ('monitor-later', 'resolution-later', 'pml-later'):
        scene = replace_member(scene, own=material(9., (1., 1.), (.15, .3)))
        if case == 'monitor-later':
            scene = scene.replace(monitors=(scene.monitors[0].replace(wavelengths=(.35,)),))
        elif case == 'resolution-later':
            scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=14, wavelength=.35))
        else:
            scene = scene.replace(boundaries=scene.boundaries.replace(z=fd.PML(cells=12, wavelength=.35)))
        assert .35/grid.dx > 8
        with pytest.raises(ValueError, match='DebyeStackSpectral'):
            validate(scene, grid)
    else:
        assert validate(scene, grid) is True
        assert quiet(fd.plan, scene).dt > 0
        return
    with pytest.raises(ValueError, match='(?i)wavelength|skin|spectral|cells|band'):
        quiet(fd.plan, scene)
