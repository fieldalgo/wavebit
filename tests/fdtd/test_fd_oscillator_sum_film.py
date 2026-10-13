"""DRAFT: finite unequal-damping oscillator-sum film gates.

Independent centered-difference, storage/dissipation and continuum optical
oracles; no optical ceiling is accepted before disclosed calibration.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._kernel.observe import InteriorEnergy
from wavebit.fdtd._prepare import checks
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import backends, to_numpy

TERM = 'oscillator_sum_film[0]'
CANDIDATE_CEILINGS = (.005, .010)
CALIBRATED_CEILINGS = {'mixed-metal': (.005, .010), 'multi-lorentz': (.005, .010)}
# Specifications are (kind, angular resonance, damping, strength).
MIXED = (('drude', 0., 1., 81.), ('lorentz', 8., 2., 10.))
MULTI = (('lorentz', 5., .8, 10.), ('lorentz', 8., 1.5, 20.), ('lorentz', 11., 2., 15.))
EIGHT = tuple(('lorentz', 3.+j, .5+.2*j, 3.+j) for j in range(8))


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def material(eps=4., pole_specs=MULTI):
    return fd.Dispersive(eps_inf=eps, poles=tuple(
        fd.Drude(omega_p=math.sqrt(strength), gamma=gamma) if kind == 'drude'
        else fd.Lorentz(omega_0=omega0, gamma=gamma, strength=strength)
        for kind, omega0, gamma, strength in pole_specs))


def film_scene(axis=2, eps=4., pole_specs=MULTI, host=1.,
               shift=.03, dx=.1, courant=.3):
    lengths = [.4]*3
    lengths[axis] = 2.
    size = [length+4*dx for length in lengths]
    size[axis] = .54
    center = [0.]*3
    center[axis] = shift
    return fd.Scene(domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=2, courant=courant),
        background=fd.Material(eps=host), structures=(fd.Structure(
            fd.Box(size=tuple(size), center=tuple(center)), material(eps, pole_specs)),))


def fills(scene, plan, axis):
    """Direct interval intersections at public Yee coordinates, no lowering helper."""
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


def response(plan, params, omega):
    """Identify the actual production advance map with basis states, then solve it.

    No production coefficient formula or rank-one inverse is reused here. A
    dropped coupling in advance is visible even if lowered coefficients are right.
    """
    ae, be = (np.broadcast_to(to_numpy(params[key]), plan.shape) for key in ('ae', 'be'))
    value = np.array(plan.dt/plan.dx*(1+ae)/(2*be), dtype=complex)
    terms = [term for term in plan.terms if hasattr(term, 'n_poles')]
    assert len(terms) == 1 and terms[0].name == TERM
    term, rt = terms[0], runtime('numpy')
    count, shape = term.n_poles, (3,)+term.region.shape
    zeros = tuple(np.zeros(shape, complex) for _ in range(count))
    field = np.zeros(plan.shape, complex)
    empty = {'P': zeros, 'previous': zeros, 'delta': zeros}
    A, B = [], []
    for slot in range(count):
        basis = tuple(np.ones(shape, complex) if j == slot else np.zeros(shape, complex) for j in range(count))
        A.append(np.asarray(term.advance({**empty, 'P': basis}, field, None, 0, params, rt)['P']))
        B.append(np.asarray(term.advance({**empty, 'previous': basis}, field, None, 0, params, rt)['P']))
    A, B = np.stack(A, axis=1), np.stack(B, axis=1)
    drive = np.asarray(term.advance(empty, np.ones(plan.shape, complex), None, 0, params, rt)['P'])
    z = np.exp(-1j*omega*plan.dt)
    identity = np.eye(count).reshape((count, count)+(1,)*len(shape))
    matrix = np.moveaxis(z*identity-A-B/z, (0, 1), (-2, -1))
    rhs = np.moveaxis(drive, 0, -1)[..., None]
    modes = np.linalg.solve(matrix, rhs)[..., 0]
    value[(slice(None),)+term.region.slices] += np.sum(modes, axis=-1)
    return value


def direct_response(f, axis, eps, pole_specs, host, omega, dt):
    own = complex(eps)
    for kind, w0, gamma, strength in pole_specs:
        if strength:
            q = (0. if kind == 'drude' else w0*w0)-4*math.sin(omega*dt/2)**2/dt**2
            q -= 1j*gamma*math.sin(omega*dt)/dt
            own += strength/q
    expected = f*own+(1-f)*host
    expected[axis] = 1/(f[axis]/own+(1-f[axis])/host)
    return expected


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('mixed', 'multi', 'n8', 'host', 'equal-eps', 'tiny-fill', 'zero', 'equal-gamma'))
def test_actual_oscillator_sum_recurrence_is_the_declared_discrete_mixture(axis, case):
    specs, eps, host, shift = MULTI, 4., 1., .03
    if case == 'mixed':
        specs = MIXED
    elif case == 'n8':
        specs = EIGHT
    elif case == 'host':
        eps, host = 2.25, 1.44
    elif case == 'equal-eps':
        eps = host = 1.
    elif case == 'tiny-fill':
        shift = -.02+1e-12
    elif case == 'zero':
        specs = tuple((kind, w0, gamma, 0.) for kind, w0, gamma, _ in MULTI)
    elif case == 'equal-gamma':
        specs = tuple((kind, w0, .7, strength) for kind, w0, _, strength in MULTI)
    scene = film_scene(axis=axis, eps=eps, pole_specs=specs, host=host, shift=shift)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    term = next(t for t in plan.terms if t.name == TERM)
    assert term.n_poles == len(specs)
    f = fills(scene, plan, axis)
    local = f[(slice(None),)+term.region.slices]
    for spec, drive in zip(specs, params['poles'][TERM]['drive']):
        np.testing.assert_array_equal(np.broadcast_to(to_numpy(drive), local.shape) > 0,
                                      (local > 0) & (spec[3] > 0))
    if case == 'tiny-fill':
        assert np.any((local > 0) & (local < 1e-9))
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        np.testing.assert_allclose(response(plan, params, omega),
            direct_response(f, axis, eps, specs, host, omega, plan.dt), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('one-active', 'equal-poles', 'reversed', 'all-zero'))
def test_oscillator_sum_endpoints_zero_strength_and_pole_order(axis, case):
    specs = MULTI
    if case == 'one-active':
        specs = (('lorentz', 2*math.pi, 0., 0.), MULTI[1], ('lorentz', 12., 0., 0.))
    elif case == 'equal-poles':
        specs = tuple(('lorentz', 8., 1.5, strength) for strength in (5., 7., 8.))
    elif case == 'all-zero':
        specs = tuple(('lorentz', 2*math.pi, 0., 0.) for _ in range(3))
    scene = film_scene(axis=axis, pole_specs=specs)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    f = fills(scene, plan, axis)
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        np.testing.assert_allclose(response(plan, params, omega),
            direct_response(f, axis, 4., specs, 1., omega, plan.dt), rtol=3e-12, atol=3e-12)
    coefficients = params['poles'][TERM]
    for j, spec in enumerate(specs):
        if spec[3] == 0:
            for key in ('drive', 'coupling', 'energy_inverse_strength', 'energy_resonance_over_strength', 'energy_active'):
                assert np.count_nonzero(to_numpy(coefficients[key][j])) == 0
    if case == 'all-zero':
        assert np.count_nonzero(to_numpy(coefficients['energy_rank_one'])) == 0
        reference = scene.replace(structures=(fd.Structure(scene.structures[0].shape, fd.Material(eps=4.)),))
        refplan = quiet(fd.plan, reference)
        refparams = fd.lower(reference, refplan, 'numpy')
        for key in ('ae', 'be', 'ah', 'bh'):
            np.testing.assert_allclose(np.broadcast_to(params[key], plan.shape),
                                       np.broadcast_to(refparams[key], plan.shape), rtol=2e-14, atol=2e-14)
        from studies.fdtd.oscillator_sum_film.fdtd_oscillator_sum_film_probe import build
        from wavebit.fdtd._assemble import _round_trip
        diagnostic, _ = build(eps=4., pole_specs=specs, theta=0.)
        assert _round_trip(diagnostic, diagnostic.sources[0], None) == pytest.approx(4*diagnostic.domain.size[2], rel=2e-15)
    elif case == 'reversed':
        reverse = scene.replace(structures=(fd.Structure(scene.structures[0].shape, material(pole_specs=specs[::-1])),))
        reverse_plan = quiet(fd.plan, reverse)
        reverse_params = fd.lower(reverse, reverse_plan, 'numpy')
        for key in ('step_gain', 'hold', 'drive', 'coupling', 'energy_inverse_strength', 'energy_resonance_over_strength', 'energy_active'):
            for actual, expected in zip(coefficients[key], reverse_params['poles'][TERM][key][::-1]):
                np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(response(plan, params, 2*math.pi),
                                   response(reverse_plan, reverse_params, 2*math.pi), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('mixed', 'multi', 'n8', 'lossless', 'hold-zero', 'negative-hold', 'zero'))
def test_oscillator_sum_storage_and_actual_recurrence_dissipation(backend, axis, case):
    specs = MIXED if case == 'mixed' else EIGHT if case == 'n8' else MULTI
    if case == 'lossless':
        specs = tuple((kind, w0, 0., strength) for kind, w0, _, strength in specs)
    elif case == 'zero':
        specs = (MULTI[0], ('lorentz', 8., 0., 0.), MULTI[2])
    scene = film_scene(axis=axis, pole_specs=specs)
    plan = quiet(fd.plan, scene)
    if case in ('hold-zero', 'negative-hold'):
        gamma = (2. if case == 'hold-zero' else 3.)/plan.dt
        specs = (specs[0][:2]+(gamma, specs[0][3]),)+specs[1:]
        scene = film_scene(axis=axis, pole_specs=specs)
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(t for t in plan.terms if t.name == TERM)
    coefficients = params['poles'][TERM]
    if case == 'hold-zero':
        assert np.max(abs(to_numpy(coefficients['hold'][0]))) < 3e-16
    if case == 'negative-hold':
        assert np.min(to_numpy(coefficients['hold'][0])) < 0
    rng = np.random.default_rng(814)
    shape = (3,)+term.region.shape
    old = tuple(rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in specs)
    new = tuple(rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in specs)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    mem = {'P': tuple(rt.array(v, 'complex128') for v in new),
           'previous': tuple(rt.array(v, 'complex128') for v in old),
           'delta': tuple(rt.array(v-u, 'complex128') for u, v in zip(old, new))}
    f = fills(scene, plan, axis)
    A = f+(1-f)*4.
    kappa, alpha = f.copy(), np.zeros(plan.shape)
    kappa[axis] = f[axis]/A[axis]**2
    alpha[axis] = (1-f[axis])/A[axis]
    density, total_new, total_old = np.zeros(plan.shape), np.zeros(plan.shape, complex), np.zeros(plan.shape, complex)
    inverses = []
    for j, (kind, w0, _, strength) in enumerate(specs):
        active = (kappa > 0) & (strength > 0)
        effective = kappa*strength
        inverse = np.where(active, 1/np.where(active, effective, 1.), 0.)
        inverses.append(inverse)
        p, q = np.zeros(plan.shape, complex), np.zeros(plan.shape, complex)
        p[(slice(None),)+term.region.slices] = new[j]
        q[(slice(None),)+term.region.slices] = old[j]
        step = p-q
        d = 0. if kind == 'drude' else w0*w0
        density += (abs(step/plan.dt)**2+d*np.real(p*np.conj(q)))*inverse-np.where(active, np.real(field*np.conj(step)), 0.)
        total_new += np.where(active, p, 0.)
        total_old += np.where(active, q, 0.)
    cross = np.where(kappa > 0, alpha/np.where(kappa > 0, kappa, 1.), 0.)
    density += cross*np.real(total_new*np.conj(total_old))
    whole = tuple((part.start, part.stop) for part in term.region.slices)
    cut = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in whole)
    for cells, comps in ((whole, slice(None)), (cut, slice(axis, axis+1))):
        offsets = tuple(lo-part.start for (lo, _), part in zip(cells, term.region.slices))
        got = InteriorEnergy._pole(mem, rt.array(field, 'complex128'), coefficients,
                                   term.n_poles, cells, offsets, plan.dt, rt, comps=comps)
        expected = np.sum(density[(comps,)+tuple(slice(lo, hi) for lo, hi in cells)])
        assert float(to_numpy(got)) == pytest.approx(expected, rel=4e-13, abs=4e-13)
    # Independent zero-drive conservation/dissipation identity, using actual
    # advance and the observer on consecutive stored states, not just coefficients.
    zero = rt.array(np.zeros(plan.shape, complex), 'complex128')
    after = term.advance(mem, zero, None, 0, params, rt)
    v0 = float(to_numpy(InteriorEnergy._pole(mem, zero, coefficients, term.n_poles,
                      whole, (0, 0, 0), plan.dt, rt)))
    v1 = float(to_numpy(InteriorEnergy._pole(after, zero, coefficients, term.n_poles,
                      whole, (0, 0, 0), plan.dt, rt)))
    dissipated = 0.
    for j, (_, _, gamma, _) in enumerate(specs):
        velocity_sum = (to_numpy(after['P'][j])-old[j])/plan.dt
        local_inverse = inverses[j][(slice(None),)+term.region.slices]
        dissipated += plan.dt*.5*gamma*np.sum(abs(velocity_sum)**2*local_inverse)
    assert dissipated >= 0
    budget = 512*np.finfo(float).eps*max(1., abs(v0), abs(v1), dissipated)
    assert abs(v1-v0+dissipated) <= budget


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('family', ('mixed-metal', 'multi-lorentz'))
def test_oblique_oscillator_sum_matches_complex_airy_on_each_backend(backend, family):
    from studies.fdtd.oscillator_sum_film.fdtd_oscillator_sum_film_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review six disclosed calibration rows before accepting ceilings.'
    row = measure(family=family, n=40, pol='p', periods=120, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS[family]
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
    # Mixed pole families are admitted only in a lossless host; this combination stays unsupported.
    elif case == 'appended-debye-lossy-host':
        scene = scene.replace(background=fd.Conductor(eps=1., sigma=.2))
        own = own.replace(poles=own.poles+(fd.Debye(tau=.2, delta_eps=1.),))
    elif case == 'mixed-poles-lossy-host':
        scene = scene.replace(background=fd.Conductor(eps=1., sigma=.2))
        own = own.replace(poles=(own.poles[0], fd.Debye(tau=.2, delta_eps=1.)))
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
    'magnetic', 'appended-debye-lossy-host', 'mixed-poles-lossy-host', 'background-loss', 'background-magnetic', 'partner',
    'initial', 'stencil', 'pml-normal', 'sub-dx', 'transverse-halo', 'transverse-absorber',
    'unrepresentable-strength'))
def test_unsupported_oscillator_sum_films_are_refused_by_name(case):
    if case == 'unrepresentable-strength':
        specs = tuple(('lorentz', 8., 1., math.ulp(0.0)) for _ in range(2))
        scene = film_scene(pole_specs=specs)
        body = scene.structures[0].shape
        ordinary = fd.Lorentz(omega_0=8., gamma=1., strength=1.)
        scenes = [scene]
        # Construct actual angular frequencies directly: a strength-only test
        # helper would underflow before reaching the production descriptor.
        for tiny in (fd.Drude(omega_p=1e-200, gamma=1.),
                     fd.Lorentz(omega_0=1e-200, gamma=1., strength=1.)):
            scenes.append(scene.replace(structures=(fd.Structure(body,
                fd.Dispersive(eps_inf=4., poles=(tiny, ordinary))),)))
        for candidate in scenes:
            with pytest.raises(ValueError, match='(?i)oscillator-sum-film.*float64'):
                plan = quiet(fd.plan, candidate)
                fd.lower(candidate, plan, 'numpy')
        return
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('dispers', 'film', 'smooth', 'lossless', 'absorbing'))
    if case == 'stencil':
        assert 'smoothed' in message and 'fd.dispersive' in message


@pytest.mark.parametrize('change', ('strength0', 'strength2', 'gamma0', 'gamma2', 'omega2',
    'pole-count', 'reversed-poles', 'kind', 'eps', 'center', 'background', 'missing-memory',
    'extra-memory', 'ordinary-memory', 'nonfilm-plan', 'one-pole-plan', 'debye-plan', 'conductive-plan'))
def test_prepared_oscillator_sum_plan_refuses_changed_identity_or_memory(change):
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
                     else plan.terms+(dataclasses.replace(term, name='oscillator_sum_film[1]'),))
        plan = dataclasses.replace(plan, terms=terms)
    elif change in ('nonfilm-plan', 'one-pole-plan', 'debye-plan', 'conductive-plan'):
        previous = fd.Material(eps=4.)
        if change == 'one-pole-plan':
            previous = material(pole_specs=(MULTI[0],))
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
        if change.startswith(('strength', 'gamma', 'omega')):
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
@pytest.mark.parametrize('parameter', ('strength0', 'strength2', 'gamma0', 'gamma2', 'omega2'))
def test_traced_oscillator_sum_parameters_cannot_bypass_the_concrete_door(mode, parameter):
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
        with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|oscillator'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|oscillator'):
                    lower(dual, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|oscillator'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')


@pytest.mark.parametrize('count', (2, 3, 8))
def test_oscillator_sum_cfl_uses_every_active_strength_and_largest_resonance(count):
    specs = tuple(('lorentz', 4.+j, .5+j, 40.+j) for j in range(count))
    scene = film_scene(eps=1., pole_specs=specs, courant=.01)
    plan = quiet(fd.plan, scene)
    rate = max(spec[1]**2 for spec in specs)+sum(spec[3] for spec in specs)
    vacuum = plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+rate/4)
    got, reason = checks.cfl_bound(scene, plan)
    assert reason is None and got == pytest.approx(expected, rel=2e-13)
    unsafe = scene.replace(run=fd.Run(steps=2, courant=(expected/vacuum+1)/2))
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        quiet(fd.plan, unsafe)
    inactive = film_scene(eps=1., pole_specs=specs+(('lorentz', 1e200, 1e200, 0.),), courant=.01)
    inactive_plan = quiet(fd.plan, inactive)
    assert checks.cfl_bound(inactive, inactive_plan)[0] == pytest.approx(expected, rel=2e-13)
    zero = film_scene(eps=1., pole_specs=(('lorentz', 1e200, 1e200, 0.),)*2)
    zero_plan = quiet(fd.plan, zero)
    bound, reason = checks.cfl_bound(zero, zero_plan)
    assert reason is None and bound == pytest.approx(vacuum, rel=2e-15)
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        checks.check_stability(zero, dataclasses.replace(zero_plan, dt=2*bound))


@pytest.mark.parametrize('case', ('band-interior-peak', 'active-undamped', 'monitor-phase',
    'monitor-skin', 'resolution-phase', 'pml-phase', 'zero-strength-resonance', 'resolved-metal',
    'traced-band-forward', 'traced-band-jax'))
def test_oscillator_sum_spectral_guard_covers_each_band_peak_and_sample(case):
    from studies.fdtd.oscillator_sum_film.fdtd_oscillator_sum_film_probe import build, band_certificate, epsilon
    from wavebit.fdtd._prepare.oscillator_sum_resolution import validate
    scene, _ = build()
    if case.startswith('traced-band'):
        grid = quiet(fd.plan, scene).grid

        def check(df):
            incoming = scene.sources[0].replace(band=fd.Band(wavelength=1., df=df, cutoff=1e-6))
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
    if case == 'band-interior-peak':
        specs = (('drude', 0., 1., 1.), ('lorentz', 6., .1, 10.))
        scene, _ = build(n=20, pole_specs=specs)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape, fd.Material(eps=1.)),))
        grid = quiet(fd.plan, safe).grid
        assert grid.dx*band_certificate('mixed-metal', pole_specs=specs, endpoints_only=True) < 2*math.pi/8
        assert grid.dx*band_certificate('mixed-metal', pole_specs=specs) > 2*math.pi/8
        with pytest.raises(ValueError, match='(?i)source .* band.*bulk spectral mesh bound'):
            validate(scene, grid)
    elif case == 'active-undamped':
        scene, _ = build(pole_specs=(('lorentz', 6., 0., 10.), ('lorentz', 10., 1., 5.)))
    elif case == 'monitor-phase':
        scene = scene.replace(monitors=tuple(m.replace(wavelengths=(.1,)) for m in scene.monitors))
    elif case == 'monitor-skin':
        scene, _ = build(pole_specs=(('drude', 0., .1, 80.**2), ('lorentz', 8., 1., 1.)), courant=.01)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape, fd.Material(eps=1.)),))
        grid = quiet(fd.plan, safe).grid
        k = 2*math.pi*np.sqrt(epsilon((1.,), pole_specs=(('drude', 0., .1, 80.**2),
                                                        ('lorentz', 8., 1., 1.)))[0])
        assert 2*math.pi/(k.real*grid.dx) > 8 and 1/(k.imag*grid.dx) < 1
        with pytest.raises(ValueError, match='(?i)skin depth'):
            validate(scene.replace(sources=()), grid)
    elif case == 'resolution-phase':
        scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=4, wavelength=.1))
    elif case == 'pml-phase':
        scene = scene.replace(boundaries=scene.boundaries.replace(z=fd.PML(cells=24, wavelength=.1, reflection=1e-12)))
    elif case == 'zero-strength-resonance':
        scene, _ = build(pole_specs=(('lorentz', 2*math.pi, 0., 0.), ('lorentz', 8., 0., 0.)))
    if case in ('zero-strength-resonance', 'resolved-metal'):
        assert quiet(fd.plan, scene).dt > 0
    else:
        with pytest.raises(ValueError, match='(?i)wavelength|skin|band|spectral|cells|resonance'):
            quiet(fd.plan, scene)
