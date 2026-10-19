"""Touching passive films against independent series/parallel constitutive laws."""
import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from tests.fdtd.test_fd_conductive_stack import (stack_scene, quiet, member_fills,
    actual_response, stack_terms)


MATERIALS = (((4., 8.), (2.25, 4.)), ((4., 0.), (2.25, 4.)),
             ((4., 8.), (2.25, 0.)), ((4., 0.), (2.25, 0.)),
             ((4., 8.), (4., 8.)), ((4., 8.), (2., 4.)))


def contact_scene(axis=2, materials=MATERIALS[0], fraction=.375):
    dx = .125
    seam = fraction*dx
    return stack_scene(axis=axis, materials=materials, dx=dx,
                       centers=(seam-.25, seam+.25), thicknesses=(.5, .5))


def oracle(scene, plan, axis, omega):
    f = member_fills(scene, plan, axis, 0)
    g = member_fills(scene, plan, axis, 1)
    host = 1-f-g
    a, b = [complex(float(s.material.eps[0]), float(s.material.sigma[0])/omega)
            for s in scene.structures]
    expected = f*a+g*b+host
    expected[axis] = 1/(f[axis]/a+g[axis]/b+host[axis])
    return expected


@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('fraction', (0., .125, .375, .5, .875))
@pytest.mark.parametrize('materials', MATERIALS)
def test_contact_current_matches_independent_harmonic_complex_medium(axis, fraction, materials):
    scene = contact_scene(axis, materials, fraction)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    assert len(stack_terms(plan)) == 1
    for angle in (.071, .39, 1.2):
        got, omega = actual_response(plan, params, axis, angle)
        np.testing.assert_allclose(got, oracle(scene, plan, axis, omega), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('mutation', ('dc', 'drive'))
def test_contact_oracle_detects_wrong_current_coefficients(mutation):
    scene = contact_scene()
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    term = stack_terms(plan)[0]
    if mutation == 'drive':
        params['poles'][term.name]['drive'] = tuple(v*.9 for v in params['poles'][term.name]['drive'])
    else:
        # Remove the trapezoidal conductive attenuation while preserving epsilon.
        ae, be = params['ae'], params['be']
        params['be'] = be*2/(1+ae)
        params['ae'] = np.ones_like(ae)
    got, omega = actual_response(plan, params, 2, .39)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, oracle(scene, plan, 2, omega), rtol=3e-12, atol=3e-12)


def physical_weights(scene, plan, axis):
    """Extract storage from series admittance, independently of split coefficients."""
    from fractions import Fraction as F
    fractions = [member_fills(scene, plan, axis, i) for i in range(2)]
    fractions.append(1-fractions[0]-fractions[1])
    media = [(float(s.material.eps[0]), float(s.material.sigma[0])) for s in scene.structures]
    media.append((1., 0.))
    eta, dc, delta, tau = [np.zeros(plan.shape) for _ in range(4)]
    tau.fill(1.)
    for index in np.ndindex(plan.shape):
        occupied = [(F(float(f[index])), F(a), F(u)) for f, (a, u) in zip(fractions, media) if f[index] > 0]
        if index[0] != axis:
            eta[index] = float(sum(f*a for f, a, u in occupied))
            dc[index] = float(sum(f*u for f, a, u in occupied))
        else:
            eta[index] = float(1/sum(f/a for f, a, u in occupied))
            if all(u for f, a, u in occupied):
                dc[index] = float(1/sum(f/u for f, a, u in occupied))
            if len(occupied) == 2:
                (f, a, u), (g, b, v) = occupied
                B, A = f*v+g*u, f*b+g*a
                if B:
                    # Polynomial long division of (a*s+u)(b*s+v)/(s*(A*s+B)).
                    strength = (a*v+b*u)/B-a*b/A-A*u*v/B**2
                    assert strength >= 0
                    delta[index], tau[index] = float(strength), float(A/B)
    return eta, dc, delta, tau


@pytest.mark.parametrize('backend', ('numpy', 'jax', 'torch'))
@pytest.mark.parametrize('materials', MATERIALS)
def test_contact_positive_storage_and_discrete_dissipation(backend, materials):
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._kernel.step import initial_state, step
    from tests.fixtures.fdtd.fd_helpers import to_numpy
    from tests.fdtd.test_fd_conductive_stack import expand, periodic_curl_e
    scene = contact_scene(materials=materials)
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term, = stack_terms(plan)
    eta, sigma, delta, tau = physical_weights(scene, plan, 2)
    shift = delta*plan.dt/(2*tau+plan.dt)
    inverse = np.divide(1., delta, out=np.zeros_like(delta), where=delta > 0)
    resistance = inverse*tau
    rng = np.random.default_rng(1701)
    state = initial_state(plan, params, rt)
    state['E'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    state['H'] = rt.array(rng.normal(size=plan.shape), plan.dtype)
    state['mem'][term.name] = term.advance(term.init_mem(plan, rt), state['E'], state['H'], 0, params, rt)
    def storage(state):
        E, H = [np.array(to_numpy(state[k]), copy=True) for k in ('E', 'H')]
        P = expand(np.array(to_numpy(state['mem'][term.name]['previous'][0]), copy=True), plan, term)+shift*E
        electric = np.sum(eta*E**2+inverse*P**2)
        observed = float(to_numpy(plan.energy.electric(state['E'], state['mem'], params, rt)))
        assert observed == pytest.approx(electric, rel=5e-13, abs=5e-13)
        energy = electric+np.sum(H*(H-plan.dt/plan.dx*periodic_curl_e(E)))
        assert energy >= 0
        return energy, E, P
    before, old_e, old_p = storage(state)
    for n in range(50):
        state = step(plan, params, state, n, rt)
        after, E, P = storage(state)
        loss = 2*plan.dt*np.sum(sigma*((E+old_e)/2)**2)+2/plan.dt*np.sum(resistance*(P-old_p)**2)
        assert loss >= 0
        assert abs(after-before+loss) <= 4096*np.finfo(float).eps*max(1., abs(before), abs(after), loss)
        before, old_e, old_p = after, E, P


@pytest.mark.parametrize('change', ('missing-memory', 'changed-sigma', 'reordered', 'gap', 'overlap', 'three-films', 'junction'))
def test_contact_scope_and_prepared_identity_remain_guarded(change):
    import dataclasses
    scene = contact_scene()
    plan = quiet(fd.plan, scene)
    if change == 'missing-memory':
        plan = dataclasses.replace(plan, terms=tuple(t for t in plan.terms if not t.name.startswith('conductive_stack[')))
    else:
        structures = list(scene.structures)
        second = structures[1]
        if change == 'changed-sigma':
            structures[1] = dataclasses.replace(second, material=fd.Conductor(eps=2.25, sigma=5.))
        elif change == 'reordered':
            structures.reverse()
        elif change in ('gap', 'overlap'):
            center = list(second.shape.center)
            center[2] += (1 if change == 'gap' else -1)*2**-30
            structures[1] = dataclasses.replace(second, shape=second.shape.replace(center=tuple(center)))
        elif change == 'three-films':
            structures.append(dataclasses.replace(second, shape=second.shape.replace(center=(0., 0., .796875))))
        else:
            structures[1] = dataclasses.replace(second, junction='staircase')
        scene = scene.replace(structures=tuple(structures))
    with pytest.raises(ValueError):
        fd.lower(scene, plan, 'numpy')
    if change in ('gap', 'overlap', 'three-films', 'junction'):
        with pytest.raises(ValueError):
            quiet(fd.plan, scene)


def test_identical_contact_reduces_to_single_film_and_order_is_irrelevant():
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._kernel.step import initial_state, step
    from tests.fixtures.fdtd.fd_helpers import to_numpy
    scene = contact_scene(materials=((4., 8.), (4., 8.)))
    one = fd.Structure(fd.Box(size=(.9, .9, 1.), center=(0., 0., .375*.125)), fd.Conductor(eps=4., sigma=8.))
    scenes = (scene, scene.replace(structures=scene.structures[::-1]), scene.replace(structures=(one,)))
    results = []
    for item in scenes:
        plan = quiet(fd.plan, item)
        params = fd.lower(item, plan, 'numpy')
        rt = runtime('numpy')
        state = initial_state(plan, params, rt)
        rng = np.random.default_rng(32)
        state['E'] = rng.normal(size=plan.shape)
        state['H'] = rng.normal(size=plan.shape)
        for n in range(10):
            state = step(plan, params, state, n, rt)
        results.append((to_numpy(state['E']), to_numpy(state['H'])))
    for got in results[1:]:
        np.testing.assert_allclose(got, results[0], rtol=3e-12, atol=3e-12)


def test_contact_reference_limits_and_reciprocity():
    from tests.fixtures.fdtd.conductive_contact import FILMS, WAVELENGTHS, reference
    from studies.fdtd.layer_alignment.fdtd_layer_alignment import multilayer
    import math
    omega = 2*np.pi/np.asarray(WAVELENGTHS)
    kz = np.sqrt(omega**2-(2*np.pi*np.sin(math.radians(35.)))**2)
    for pol in ('s', 'p'):
        r, t = reference(pol, ((1., 0., .125), (1., 0., .125)))
        np.testing.assert_allclose(r, 0., atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(t, np.exp(1j*kz*.25), atol=2e-13, rtol=2e-13)
        r, t = reference(pol, ((4., 0., .125), (2.25, 0., .125)))
        np.testing.assert_allclose(abs(r)**2+abs(t)**2, 1., atol=2e-13, rtol=2e-13)
        r, t = reference(pol)
        np.testing.assert_allclose(reference(pol, FILMS[::-1])[1], t, atol=2e-13, rtol=2e-13)
        assert np.all(1-abs(r)**2-abs(t)**2 > 0)
        equal = reference(pol, ((4., 8., .125), (4., 8., .125)))
        single = multilayer(WAVELENGTHS, [4.+8j/omega], [.25], theta=math.radians(35.))['tm' if pol == 'p' else 'te']
        np.testing.assert_allclose(equal, single, atol=2e-13, rtol=2e-13)


def test_public_contact_admission_and_union_memory_accounting():
    from wavebit import fdtd as public, Units
    from wavebit.materials import ConductiveMaterial
    from wavebit.fdtd._lower_public import lower_problem
    from wavebit.fdtd._runtime import runtime
    from wavebit.fdtd._kernel.step import initial_state
    from wavebit.fdtd._prepare.report import report
    problem = public.Problem(units=Units(length='um'), scene=public.Scene(
        domain=public.Domain(size=(.5, .5, 4.)),
        boundaries=public.Boundaries(x=public.Periodic(), y=public.Periodic(), z=public.Periodic()),
        structures=(public.Structure('a', shape=public.Box(size=(1., 1., .5), center=(0., 0., -.25)),
                                     material=ConductiveMaterial(epsilon=4., conductivity=2e4)),
                    public.Structure('b', shape=public.Box(size=(1., 1., .5), center=(0., 0., .25)),
                                     material=ConductiveMaterial(epsilon=2.25, conductivity=1e4)))))
    numerics = public.Numerics(grid=public.UniformGrid(spacing=.125), duration=public.Duration(steps=3))
    assert public.check(problem, numerics=numerics).ok
    scene, _ = lower_problem(problem, numerics)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    state = initial_state(plan, params, runtime('numpy'))
    term, = stack_terms(plan)
    assert set(state['mem']) == {term.name}
    held = sum(v.nbytes for values in state['mem'][term.name].values() for v in values)
    assert report(scene, plan, params).bytes_per_state == state['E'].nbytes+state['H'].nbytes+held
