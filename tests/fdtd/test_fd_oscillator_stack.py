"""Separated single-oscillator films: independent response, storage and mesh gates.

The response oracle uses centered first/second derivative symbols, not a
production reducer. Optical ceilings require the six-row calibration disclosure.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import dataclasses
import math

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare.report import report
from wavebit.fdtd._kernel.constitutive import ConstitutiveTerm
from wavebit.fdtd._kernel.observe import InteriorEnergy
from wavebit.fdtd._kernel.poles import PoleTerm
from wavebit.fdtd._kernel.step import initial_state, step
from wavebit.fdtd._prepare import checks
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import backends, to_numpy
from tests.fdtd.test_fd_conductive_stack import quiet, member_fills, expand, periodic_curl_e
from tests.fdtd.test_fd_single_pole_film import material

PREFIX = 'oscillator_stack['
MEMORY_KEYS = ('P', 'previous', 'delta')
# Existing independent single-pole response and stack storage budgets.
RESPONSE_TOL = 3e-12
STORAGE_TOL = 5e-13
STEP_ULPS = 4096
CFL_TOL = 2e-13
CALIBRATED_CEILINGS = (0.005, 0.01)


def specifications(case):
    first = ('drude', 4., 1., 20., 5.)
    second = ('lorentz', 2.25, .8, 30., 8.)
    if case == 'drude-pair':
        second = ('drude', 2.25, .8, 30., 8.)
    elif case == 'lorentz-pair':
        first = ('lorentz', 4., 1., 20., 5.)
    elif case == 'triple':
        return first, second, ('lorentz', 3., 0., 0., 2*math.pi)
    elif case == 'four':
        return first, second, ('lorentz', 3., .4, 10., 3.), ('drude', 2., 2., 50., 1.)
    elif case == 'lossless':
        first, second = ('drude', 4., 0., 20., 5.), ('lorentz', 2.25, 0., 30., 8.)
    elif case == 'all-inactive':
        first, second = ('lorentz', 4., 0., 0., 2*math.pi), ('lorentz', 2.25, 1., 0., 8.)
    elif case == 'tiny-strength':
        second = ('lorentz', 2.25, .8, 1e-24, 8.)
    return first, second


def stack_scene(axis=2, specs=None, host=1., dx=.1, courant=.2, centers=None,
                thicknesses=None, complex_fields=False):
    specs = specifications('mixed') if specs is None else specs
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
    return response


def declared(own):
    pole = own.poles[0]
    omega0 = 0. if type(pole) is fd.Drude else float(pole.omega_0)
    strength = float(pole.omega_p)**2 if type(pole) is fd.Drude else float(pole.strength)
    return float(own.eps_inf[0]), omega0**2, float(pole.gamma), strength


def discrete_eps(own, angle, dt):
    a, w0, gamma, strength = declared(own)
    if strength == 0:
        return complex(a)
    denominator = w0-4*math.sin(angle/2)**2/dt**2-1j*gamma*math.sin(angle)/dt
    return a+strength/denominator


@pytest.mark.parametrize('axis', (0, 1, 2), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('mixed', 'drude-pair', 'lorentz-pair', 'triple',
    'four', 'lossless', 'all-inactive', 'tiny-strength', 'tiny-fill'))
def test_each_member_advance_matches_the_declared_series_parallel_response(axis, case):
    kwargs = {'dx': .125, 'centers': (-.625, .125+2.**-42), 'thicknesses': (.5, .5)} if case == 'tiny-fill' else {}
    scene = stack_scene(axis, specifications(case), **kwargs)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    terms = stack_terms(plan)
    assert [t.name for t in terms] == [f'{PREFIX}{j}]' for j in range(len(scene.structures))]
    for member, term in enumerate(terms):
        assert term.n_poles == 1
        coefficients = params['poles'][term.name]
        assert all(len(coefficients[k]) == 1 for k in ('step_gain', 'hold', 'drive',
            'energy_inverse_strength', 'energy_resonance_over_strength', 'energy_active'))
        local = member_fills(scene, plan, axis, member)[(slice(None),)+term.region.slices]
        drive = np.broadcast_to(to_numpy(coefficients['drive'][0]), local.shape)
        np.testing.assert_array_equal(drive > 0, (local > 0) & (declared(scene.structures[member].material)[3] > 0))
        if case == 'tiny-fill' and member == 1:
            tiny = (local > 0) & (local < 1e-9)
            assert np.any(tiny) and np.all(drive[tiny] > 0)
    host = float(scene.background.eps[0])
    for angle in (.071, .39, 1.2):
        got = actual_response(plan, params, axis, angle)
        expected = np.full(plan.shape, complex(host))
        for member, structure in enumerate(scene.structures):
            f = member_fills(scene, plan, axis, member)
            own = discrete_eps(structure.material, angle, plan.dt)
            local = f*own+(1-f)*host
            local[axis] = 1/(f[axis]/own+(1-f[axis])/host)
            expected[f > 0] = local[f > 0]
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, expected, rtol=RESPONSE_TOL, atol=RESPONSE_TOL)

@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('case', ('mixed', 'triple', 'lossless', 'all-inactive', 'complex'))
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
        key = 'dispersive_film[0]'
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
        if declared(only.structures[0].material)[3] == 0:
            for coefficient in ('drive', 'energy_inverse_strength', 'energy_resonance_over_strength', 'energy_active'):
                assert np.count_nonzero(to_numpy(params['poles'][term.name][coefficient][0])) == 0
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
    """Arithmetic/harmonic high-frequency response and exact oscillator partial fraction."""
    host = float(scene.background.eps[0])
    eta = np.full(plan.shape, host)
    members = []
    for index, structure in enumerate(scene.structures):
        a, w0, gamma, strength = declared(structure.material)
        f = member_fills(scene, plan, axis, index)
        A = f*host+(1-f)*a
        own = f*a+(1-f)*host
        own[axis] = a*host/A[axis]
        eta[f > 0] = own[f > 0]
        S, resonance = f*strength, np.full(plan.shape, w0)
        S[axis] = (f*host*host*strength/A**2)[axis]
        resonance[axis] += ((1-f)*strength/A)[axis]
        inverse = np.divide(1., S, out=np.zeros_like(S), where=S > 0)
        members.append((S, resonance, gamma, inverse))
    return eta, members


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('mixed', 'triple', 'all-inactive', 'lossless', 'hold-zero'))
def test_all_members_storage_and_closed_yee_dissipation_use_independent_oscillators(backend, axis, case):
    scene = stack_scene(axis, specifications(case))
    plan = quiet(fd.plan, scene)
    if case == 'hold-zero':
        own = scene.structures[1].material
        scene = replace_member(scene, own=own.replace(poles=(own.poles[0].replace(gamma=2/plan.dt),)))
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params, terms = fd.lower(scene, plan, rt), stack_terms(plan)
    eta, members = independent_weights(scene, plan, axis)
    if case == 'hold-zero':
        assert np.max(abs(to_numpy(params['poles'][terms[1].name]['hold'][0]))) < 3e-16
    rng = np.random.default_rng(901)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    density = eta*abs(field)**2
    memory = {}
    for term, (strength, resonance, _, inverse) in zip(terms, members):
        old = rng.normal(size=(3,)+term.region.shape)+1j*rng.normal(size=(3,)+term.region.shape)
        new = 1.7*old+.3j
        memory[term.name] = {key: (rt.array(v, 'complex128'),)
                            for key, v in zip(MEMORY_KEYS, (new, old, new-old))}
        old_full, new_full = expand(old, plan, term), expand(new, plan, term)
        delta = new_full-old_full
        correction = np.where(strength > 0, inverse*(abs(delta)**2/plan.dt**2
            +resonance*np.real(new_full*np.conj(old_full)))-np.real(field*np.conj(delta)), 0.)
        density += correction
        cells = tuple((s.start, s.stop) for s in term.region.slices)
        crop = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in cells)
        for box, comps in ((cells, slice(None)), (crop, slice(axis, axis+1))):
            offsets = tuple(lo-s.start for (lo, _), s in zip(box, term.region.slices))
            got = InteriorEnergy._pole(memory[term.name], rt.array(field, 'complex128'),
                params['poles'][term.name], 1, box, offsets, plan.dt, rt, comps=comps)
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
        electric = np.sum(eta*E**2)
        physical = []
        for term, (strength, resonance, _, inverse) in zip(terms, members):
            old = expand(np.array(to_numpy(current['mem'][term.name]['previous'][0]), copy=True), plan, term)
            new = expand(np.array(to_numpy(current['mem'][term.name]['P'][0]), copy=True), plan, term)
            electric += np.sum(np.where(strength > 0, inverse*((new-old)**2/plan.dt**2
                +resonance*new*old)-E*(new-old), 0.))
            physical.append((new, old))
        got = float(to_numpy(plan.energy.electric(current['E'], current['mem'], params, rt)))
        assert got == pytest.approx(electric, rel=STORAGE_TOL, abs=STORAGE_TOL)
        return electric+np.sum(H*(H-plan.dt/plan.dx*periodic_curl_e(E))), physical
    previous, old_p = storage(state)
    for n in range(4):
        state = step(plan, params, state, n, rt)
        current, new_p = storage(state)
        loss = sum(gamma/(2*plan.dt)*np.sum(inverse*(new[0]-old[1])**2)
                   for old, new, (_, _, gamma, inverse) in zip(old_p, new_p, members))
        budget = STEP_ULPS*np.finfo(float).eps*max(1., abs(previous), abs(current), loss)
        assert loss >= 0 and abs(current-previous+loss) <= budget
        assert loss == 0 if case in ('all-inactive', 'lossless') else loss > 0
        previous, old_p = current, new_p


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('family', ('mixed-pair', 'lorentz-triple'))
def test_oblique_oscillator_stack_matches_declared_multilayer_on_each_backend(backend, family, monkeypatch):
    monkeypatch.syspath_prepend(str(repository_path('studies')))
    from studies.fdtd.oscillator_stack.fdtd_oscillator_stack_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review six disclosed rows before accepting optical ceilings.'
    row = measure(n=40, pol='p', family=family, periods=120, fraction=.3, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS
    assert max(row['error_r']) < r_limit and max(row['error_t']) < t_limit
    assert row['stable'] is True and row['energy_finite'] is True and row['energy_max'] > 0
    assert max(row['decay'].values()) < 1e-5
    assert row['energy_shape'] == [row['energy_rows']]

REFUSALS = ('physical-overlap', 'padded-overlap', 'all-zero-overlap', 'off-grid-two-dx',
    'exact-support-touch', 'finite-box', 'turned', 'ranged', 'anisotropic', 'magnetic',
    'lossy-host', 'conductive-zero-host', 'debye-member', 'conductor-member', 'plain-member',
    'slab-member', 'nonparallel', 'initial', 'decay', 'stencil', 'sub-dx', 'transverse-halo',
    'normal-absorber', 'transverse-absorber', 'range-later', 'multiple-poles', 'inactive-extreme-frequency')


@pytest.mark.parametrize('case', REFUSALS)
def test_every_member_must_pass_scope_and_exact_padded_support_separation(case, monkeypatch):
    from wavebit.fdtd._prepare.oscillator_stack import validate
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
                'lorentz', float(s.material.eps_inf[0]), 0., 0., 8.)) for s in scene.structures))
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
            mu=2. if case == 'magnetic' else 1., poles=(fd.Lorentz(omega_0=8., gamma=.8, strength=30.),)))
    elif case in ('lossy-host', 'conductive-zero-host'):
        scene = scene.replace(background=fd.Conductor(eps=1., sigma=.2 if case == 'lossy-host' else 0.))
    elif case == 'debye-member':
        scene = replace_member(scene, own=fd.Dispersive(eps_inf=2.25, poles=(fd.Debye(tau=.15, delta_eps=6.),)))
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
        with pytest.raises(ValueError, match='OscillatorStackDecayRefused'):
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
        scene = replace_member(scene, own=material('lorentz', 4., 1., math.ulp(0.), 8.))
    elif case == 'multiple-poles':
        own = scene.structures[1].material
        scene = replace_member(scene, own=own.replace(poles=own.poles+(fd.Lorentz(omega_0=3., gamma=1., strength=0.),)))
    elif case == 'inactive-extreme-frequency':
        scene = replace_member(scene, own=material('lorentz', 2.25, 0., 0., 1e200))
    else:
        raise AssertionError(case)
    if case in ('physical-overlap', 'padded-overlap', 'all-zero-overlap', 'off-grid-two-dx'):
        # Isolate the stack door; generic neighbor checks must not mask its removal.
        with pytest.raises(ValueError, match='OscillatorStackOverlap'):
            validate(scene, safe_grid)
    with pytest.raises(ValueError, match='(?i)oscillator|dispers|film|smooth|absorbing|support|float64|decay'):
        plan = quiet(fd.plan, scene)
        fd.lower(scene, plan, 'numpy')


CHANGES = ('later-strength', 'later-gamma', 'later-frequency', 'later-kind', 'later-count', 'eps', 'center', 'host', 'order',
    'missing-later', 'duplicate-first', 'wrong-name', 'wrong-index', 'ordinary-memory', 'foreign-flux',
    'singleton-plan', 'nonfilm-plan', 'added-source', 'removed-source', 'source-component',
    'boundary', 'wrong-region', 'wrong-count')


@pytest.mark.parametrize('change', CHANGES)
def test_prepared_stack_binds_all_members_memories_and_source_allocation(change):
    from wavebit.fdtd._prepare.oscillator_stack import refuse_mismatch
    scene = stack_scene()
    source = fd.Dipole('kick', center=(0., 0., -1.4), component='Ex', kind='electric',
                       band=fd.Band(wavelength=4., df=.025, cutoff=1e-6))
    if change in ('removed-source', 'source-component'):
        scene = scene.replace(sources=(source,))
    plan = quiet(fd.plan, scene)
    terms, other = stack_terms(plan), scene
    if change in ('later-strength', 'later-gamma', 'later-frequency', 'later-kind', 'later-count', 'eps'):
        own = material('drude' if change == 'later-kind' else 'lorentz',
            3. if change == 'eps' else 2.25, 1.3 if change == 'later-gamma' else .8,
            31. if change == 'later-strength' else 30., 9. if change == 'later-frequency' else 8.)
        if change == 'later-count':
            own = own.replace(poles=own.poles+(fd.Lorentz(omega_0=3., gamma=1., strength=0.),))
        other = replace_member(scene, own=own)
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
            revised[index] = dataclasses.replace(terms[1], name='oscillator_stack[8]')
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
    with pytest.raises(ValueError, match='OscillatorStackPlan|OscillatorStackScope'):
        refuse_mismatch(other, plan)
    with pytest.raises(ValueError, match='OscillatorStackPlan|OscillatorStackScope'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('strength', 'gamma', 'frequency', 'eps', 'center', 'host'))
def test_traced_later_poles_geometry_and_host_cannot_detach_into_stack_admission(mode, parameter):
    scene = stack_scene()
    plan = quiet(fd.plan, scene)
    initial = {'strength': 30., 'gamma': .8, 'frequency': 8., 'eps': 2.25, 'center': .455, 'host': 1.}[parameter]
    def lower(value, backend):
        old = scene.structures[1]
        if parameter == 'host':
            other = scene.replace(background=fd.Material(eps=value))
        elif parameter == 'center':
            other = replace_member(scene, shape=old.shape.replace(center=(0., 0., value)))
        else:
            other = replace_member(scene, own=material('lorentz', value if parameter == 'eps' else 2.25,
                value if parameter == 'gamma' else .8, value if parameter == 'strength' else 30.,
                value if parameter == 'frequency' else 8.))
        return fd.lower(other, plan, backend)['be']
    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|oscillator|stack|geometry'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                value = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                           torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|oscillator|stack|geometry'):
                    lower(value, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|oscillator|stack|geometry'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')


@pytest.mark.parametrize('case', ('later-fast', 'earlier-fast', 'four-members', 'all-inactive', 'subunit-eps'))
def test_cfl_uses_each_complete_oscillator_rate_then_the_spatial_maximum(case):
    specs = (('drude', 4., 1., 20., 5.), ('lorentz', 1., 1., 4000., 40.))
    if case == 'earlier-fast':
        specs = specs[::-1]
    elif case == 'four-members':
        specs = (*specs, ('lorentz', 3., .1, 100., 15.), ('drude', 2., 3., 200., 1.))
    elif case == 'all-inactive':
        specs = specifications(case)
    elif case == 'subunit-eps':
        specs = (specs[0], ('lorentz', .64, 1., 4000., 40.))
    scene = stack_scene(specs=specs, courant=.001)
    plan = quiet(fd.plan, scene)
    maxima = []
    for member, structure in enumerate(scene.structures):
        a, w0, _, strength = declared(structure.material)
        f = member_fills(scene, plan, 2, member)
        b = float(scene.background.eps[0])
        A = f*b+(1-f)*a
        normal_rate = w0+(1-f)*strength/A+(f*b*b*strength/A**2)/(a*b/A)
        np.testing.assert_allclose(normal_rate, w0+strength/a, rtol=CFL_TOL, atol=CFL_TOL)
        maxima.append(w0+strength/a)
    maximum = max(maxima)
    vacuum = math.sqrt(min(float(scene.background.eps[0]), *(s[1] for s in specs)))*plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+maximum/4)
    actual, reason = checks.cfl_bound(scene, plan)
    assert reason is None and actual == pytest.approx(expected, rel=CFL_TOL)
    summed = 1/math.sqrt(1/vacuum**2+sum(maxima)/4)
    assert not math.isclose(expected, summed, rel_tol=CFL_TOL)
    # The singleton conservative nominal omega0 term is retained even when inactive.
    unsafe = dataclasses.replace(plan, dt=2*expected)
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        checks.check_stability(scene, unsafe)


@pytest.mark.parametrize('case', ('interior-band-later', 'zero-damping-later', 'monitor-phase-later',
    'monitor-skin-later', 'resolution-later', 'pml-later', 'zero-strength-resonance',
    'resolved-metal', 'traced-band-forward', 'traced-band-jax'))
def test_spectral_certificate_includes_later_member_interior_resonances_and_samples(case):
    from wavebit.fdtd._prepare.oscillator_stack import validate
    specs = (('lorentz', 1., 0., 0., 8.), ('lorentz', 1., 0., 0., 8.))
    scene = stack_scene(specs=specs, dx=.05, courant=.001)
    source = fd.Dipole('kick', center=(0., 0., -1.4), component='Ex', kind='electric',
                       band=fd.Band(wavelength=1., df=.1, cutoff=1e-6))
    scene = scene.replace(sources=(source,))
    original_plan = quiet(fd.plan, scene)
    grid = original_plan.grid
    if case.startswith('traced-band'):
        def check(df, backend):
            incoming = source.replace(band=fd.Band(wavelength=1., df=df, cutoff=1e-6))
            other = scene.replace(sources=(incoming,))
            assert validate(other, grid) is False, 'tracing is outside structural stack admission'
            return fd.lower(other, original_plan, backend)['be']
        if case == 'traced-band-jax':
            jax = pytest.importorskip('jax')
            with pytest.raises(ValueError, match='OscillatorStackPlan|OscillatorStackScope'):
                jax.jit(lambda df: check(df, 'jax'))(.1)
        else:
            torch = pytest.importorskip('torch')
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(.1, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='OscillatorStackPlan|OscillatorStackScope'):
                    check(dual, 'torch')
        return
    if case in ('interior-band-later', 'zero-damping-later'):
        scene = replace_member(scene, own=material('lorentz', 1., 0. if case == 'zero-damping-later' else .001, 40., 8.))
        if case == 'interior-band-later':
            low, high = [2*math.pi*(1+sign*math.sqrt(2*math.log(1e6))*.1) for sign in (-1, 1)]
            # An endpoint-only conservative single-oscillator bound would admit this mesh.
            endpoint = max(w*w/abs(64-w*w-1j*.001*w) for w in (low, high))
            assert grid.dx*math.sqrt(high*high+40*endpoint) < 2*math.pi/8
        with pytest.raises(ValueError, match='(?i)band|resonan|spectral'):
            validate(scene, grid)
    elif case in ('monitor-phase-later', 'monitor-skin-later', 'resolution-later', 'pml-later'):
        wavelength = 2*math.pi/(8.-.001 if case != 'monitor-skin-later' else 8.+.001)
        scene = replace_member(scene, own=material('lorentz', 1., .001, 400., 8.))
        scene = scene.replace(sources=())
        if case.startswith('monitor-'):
            scene = scene.replace(monitors=(fd.Field('field', plane=fd.Plane(z=1.4), components=('Ex',), wavelengths=(wavelength,)),))
        elif case == 'resolution-later':
            scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=14, wavelength=wavelength))
        else:
            scene = scene.replace(boundaries=scene.boundaries.replace(z=fd.PML(cells=12, wavelength=wavelength)))
        assert wavelength/grid.dx > 8
        with pytest.raises(ValueError, match='(?i)wavelength|skin|spectral|cells'):
            validate(scene, grid)
    elif case == 'resolved-metal':
        scene = replace_member(scene, own=material('drude', 1., 1., 81., 1.))
        scene = scene.replace(resolution=fd.Resolution(dx=.025))
        assert quiet(fd.plan, scene).dt > 0
        return
    else:
        scene = scene.replace(monitors=(fd.Field('field', plane=fd.Plane(z=1.4),
            components=('Ex',), wavelengths=(2*math.pi/8,)),))
        assert validate(scene, grid) is True
        assert quiet(fd.plan, scene).dt > 0
        return
    with pytest.raises(ValueError, match='(?i)wavelength|skin|spectral|cells|band|resonan'):
        quiet(fd.plan, scene)
