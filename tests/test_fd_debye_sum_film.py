"""Finite Debye-sum film gates with independent rational, storage and optical oracles.

The continuum/discrete references sum the declared poles before applying the
interface law. Optical ceilings retain the inherited values after disclosed calibration review.
"""
import dataclasses
from decimal import Decimal, localcontext
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.observe import InteriorEnergy
from wavebit.fd._lower import checks
from wavebit.fd._runtime import runtime
from fd_helpers import backends, to_numpy


CANDIDATE_CEILINGS = (.005, .010)
# Disclosed attempt1: n40/p/T120/PML24/fraction.3 on NumPy/JAX/Torch.
# Maximum |r-error|=.003993837708; |t-error|=.009375278239; decay<1.018e-8.
# Backend scaled differences <=9.993e-16; all three runs have finite energy.
# Raw results and acceptance are retained under studies/results/fdtd_debye_sum_film_calibration*.
# Inherited ceilings are unchanged; r/t margins are .001006162292/.000624721761.
# This is a fixed-configuration regression ceiling, not a convergence guarantee.
CALIBRATED_CEILINGS = (.005, .010)
TERM = 'debye_sum_film[0]'


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def material(eps=4., deltas=(1., 1., 2., 2.), taus=(.04, .08, .16, .32)):
    return fd.Dispersive(eps_inf=eps, poles=tuple(
        fd.Debye(tau=tau, delta_eps=delta) for delta, tau in zip(deltas, taus)))


def film_scene(axis=2, eps=4., deltas=(1., 1., 2., 2.), taus=(.04, .08, .16, .32), host=1.,
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
            fd.Box(size=tuple(size), center=tuple(center)), material(eps, deltas, taus)),))


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
    ae, be = (np.broadcast_to(to_numpy(params[key]), plan.shape) for key in ('ae', 'be'))
    value = np.array(plan.dt/plan.dx*(1+ae)/(2*be), dtype=complex)
    np.testing.assert_array_equal(ae, np.ones(plan.shape))
    z = np.exp(-1j*omega*plan.dt)
    terms = [term for term in plan.terms if hasattr(term, 'n_poles')]
    assert len(terms) == 1 and terms[0].name == TERM and terms[0].n_poles >= 3
    term = terms[0]
    coefficients = params['poles'][term.name]
    for key in ('step_gain', 'hold', 'drive', 'energy_inverse_delta', 'energy_instantaneous'):
        assert len(coefficients[key]) == term.n_poles
    for pole in range(term.n_poles):
        gain, hold, drive = (to_numpy(coefficients[key][pole])
                             for key in ('step_gain', 'hold', 'drive'))
        np.testing.assert_array_equal(hold, 0.)
        value[(slice(None),)+term.region.slices] += drive/(z-gain+hold/z)
    return value


def direct_response(f, axis, eps, deltas, taus, host, omega, dt):
    s = -2j*math.tan(omega*dt/2)/dt
    own = complex(eps)
    for delta, tau in zip(deltas, taus):
        if delta:
            own += delta/(1+tau*s)
    expected = f*own+(1-f)*host
    expected[axis] = 1/(f[axis]/own+(1-f[axis])/host)
    return expected


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('eps,deltas,taus,host,shift', [
    (4., (1., 2., 3.), (.04, .12, .32), 1., .03),
    (4., (1., 1., 2., 2.), (.04, .08, .16, .32), 1., .03),
    (4., (.75,)*8, tuple(.025*1.5**j for j in range(8)), 1., .03),
    (1., (1., 2., 3.), (.05, .1, .2), 1., -.011),
    (2.25, (1.5, .7, .9), (.4, .07, .15), 1.44, .017),
    (4., (0.,)*4, (.04, .08, .16, .32), 1., .025),
    (4., (1., 1., 2., 2.), (.04, .08, .16, .32), 1., -.02+1e-12),
    (1., (1e-42, 2e-42, 3e-42), (1e-20, 2e-20, 3e-20), 1., .03),
    (4., (1., 2., 3.), (1e18, 2e18, 3e18), 1., .03),
], ids=('n3', 'n4', 'n8', 'equal-eps', 'host', 'zero', 'tiny-fill', 'fast', 'slow'))
def test_debye_sum_discrete_response_is_the_declared_sum_mixture(axis, eps, deltas, taus, host, shift):
    scene = film_scene(axis, eps, deltas, taus, host, shift)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    f = fills(scene, plan, axis)
    term = next(t for t in plan.terms if t.name == TERM)
    local = f[(slice(None),)+term.region.slices]
    coefficients = params['poles'][TERM]
    drives = [np.broadcast_to(to_numpy(value), local.shape) for value in coefficients['drive']]
    np.testing.assert_array_equal(sum(drives) > 0, (local > 0) & (sum(deltas) > 0))
    if max(taus) < 1e-18:
        assert any(np.any((to_numpy(gain) == -1.) & (to_numpy(drive) > 0))
                   for gain, drive in zip(coefficients['step_gain'], coefficients['drive']))
    if abs(shift+.02) < 1e-10:
        assert np.any((local > 0) & (local < 1e-9)), 'retain every positive fraction'
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        expected = direct_response(f, axis, eps, deltas, taus, host, omega, plan.dt)
        got = response(plan, params, omega)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(got[f == 0], host, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('case', ('equal-times', 'single-active', 'partial-equal',
    'all-zero', 'reversed', 'adjacent-times', 'tiny-residue'))
def test_debye_sum_exact_collapses_and_declared_order_preserve_response(axis, case):
    deltas, taus = (1., 1., 2., 2.), (.04, .08, .16, .32)
    if case == 'equal-times':
        taus = (.15,)*4
    elif case == 'single-active':
        deltas, taus = (0., 0., 6., 0.), (1e-250, 1e250, .15, 1e200)
    elif case == 'partial-equal':
        taus = (.1, .2, .1, .2)
    elif case == 'all-zero':
        deltas, taus = (0.,)*4, (1e-250, 1e250, 1e-200, 1e200)
    elif case == 'adjacent-times':
        taus = (.125, np.nextafter(.125, 1.), .25, .5)
    elif case == 'tiny-residue':
        deltas = (1., 1e-24, 2., 2.)
    scene = film_scene(axis=axis, deltas=deltas, taus=taus)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    coefficients = params['poles'][TERM]
    term = next(t for t in plan.terms if t.name == TERM)
    assert term.n_poles == len(deltas)
    f = fills(scene, plan, axis)
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        np.testing.assert_allclose(response(plan, params, omega),
            direct_response(f, axis, 4., deltas, taus, 1., omega, plan.dt), rtol=3e-12, atol=3e-12)
    merged = {}
    for delta, tau in zip(deltas, taus):
        if delta:
            merged[tau] = merged.get(tau, 0.)+delta
    if case in ('equal-times', 'single-active', 'partial-equal', 'all-zero'):
        for key in ('drive', 'energy_inverse_delta', 'energy_instantaneous'):
            assert all(np.count_nonzero(to_numpy(v)) == 0 for v in coefficients[key][len(merged):])
        own = (fd.Material(eps=4.) if not merged else fd.Dispersive(eps_inf=4., poles=tuple(
            fd.Debye(delta_eps=delta, tau=tau) for tau, delta in sorted(merged.items()))))
        reduced = scene.replace(structures=(fd.Structure(scene.structures[0].shape, own),))
        refplan = quiet(fd.plan, reduced)
        reference = fd.lower(reduced, refplan, 'numpy')
        for key in ('ae', 'be', 'ah', 'bh'):
            np.testing.assert_allclose(np.broadcast_to(params[key], plan.shape),
                np.broadcast_to(reference[key], plan.shape), rtol=2e-14, atol=2e-14)
        assert checks.cfl_bound(scene, plan)[0] == pytest.approx(checks.cfl_bound(reduced, refplan)[0], rel=2e-14)
        if not merged:
            from studies.fdtd_debye_sum_film_probe import build
            from wavebit.fd._assemble import _round_trip
            diagnostic, _ = build(eps=4., deltas=deltas, taus=taus, theta=0.)
            assert _round_trip(diagnostic, diagnostic.sources[0], None) == pytest.approx(
                4*diagnostic.domain.size[2], rel=2e-15)
    elif case == 'reversed':
        reverse = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
            material(deltas=deltas[::-1], taus=taus[::-1])),))
        reverse_plan = quiet(fd.plan, reverse)
        reverse_params = fd.lower(reverse, reverse_plan, 'numpy')
        for key in ('ae', 'be'):
            np.testing.assert_allclose(params[key], reverse_params[key], rtol=2e-14, atol=2e-14)
        for key in ('step_gain', 'drive', 'energy_inverse_delta', 'energy_instantaneous'):
            for actual, expected in zip(coefficients[key], reverse_params['poles'][TERM][key]):
                np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    else:
        local = f[(slice(None),)+term.region.slices]
        for drive, inverse in zip(coefficients['drive'], coefficients['energy_inverse_delta']):
            np.testing.assert_array_equal(np.broadcast_to(to_numpy(drive), local.shape) > 0, local > 0)
            np.testing.assert_array_equal(np.broadcast_to(to_numpy(inverse), local.shape) > 0, local > 0)


def rational_modes(fraction, normal, eps, host, deltas, taus):
    """Independent 100-digit roots/residues of numerator/denominator polynomials.

    This evaluates the direct mixture polynomials rather than the production
    secular interval certificate. Fixtures here have well separated active times.
    """
    with localcontext() as context:
        context.prec = 100
        D = Decimal.from_float
        f, a, b = (D(float(v)) for v in (fraction, eps, host))
        if f == 0:
            return []
        active = {}
        for delta, tau in zip(deltas, taus):
            if delta:
                key = D(float(tau))
                active[key] = active.get(key, Decimal(0))+D(float(delta))
        declared = sorted((tau, delta) for tau, delta in active.items())
        if not normal or f == 1:
            return [(float(f*delta), float(tau)) for tau, delta in declared]
        if not declared:
            return []

        def product(times):
            coefficients = [Decimal(1)]
            for tau in times:
                newer = [Decimal(0)]*(len(coefficients)+1)
                for j, value in enumerate(coefficients):
                    newer[j] += value
                    newer[j+1] += tau*value
                coefficients = newer
            return coefficients

        def evaluate(coefficients, point):
            value = Decimal(0)
            for coefficient in coefficients[::-1]:
                value = value*point+coefficient
            return value

        times = [tau for tau, _ in declared]
        den = product(times)
        numerator = [a*value for value in den]
        for j, (_, delta) in enumerate(declared):
            for k, value in enumerate(product(times[:j]+times[j+1:])):
                numerator[k] += delta*value
        denominator = [f*b*x+(1-f)*y for x, y in zip(den, numerator)]
        derivative = [Decimal(j)*v for j, v in enumerate(denominator)][1:]
        poles = sorted(-1/tau for tau in times)
        A = f*b+(1-f)*a
        bound = max(1/tau for tau in times)+(1-f)/A*sum(delta/tau for tau, delta in declared)
        boundaries = [-2*bound, *poles]
        modes = []
        for left, right in zip(boundaries, boundaries[1:]):
            sign = evaluate(denominator, left)
            assert sign*evaluate(denominator, right) < 0
            for _ in range(280):
                middle = (left+right)/2
                value = evaluate(denominator, middle)
                if value == 0:
                    left = right = middle
                    break
                if value*sign > 0:
                    left = middle
                else:
                    right = middle
            root = (left+right)/2
            tau = -1/root
            delta = tau*b*evaluate(numerator, root)/evaluate(derivative, root)
            assert delta > 0 and tau > 0
            modes.append((float(delta), float(tau)))
        return sorted(modes, key=lambda pair: pair[1])


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('n3', 'n4', 'n8', 'equal-times', 'gain-zero', 'negative-gain', 'all-zero'))
def test_debye_sum_energy_sums_all_local_memories_before_region_reduction(backend, axis, case):
    deltas, taus, courant = (1., 1., 2., 2.), (.04, .08, .16, .32), .3
    if case == 'n3':
        deltas, taus = (1., 2., 3.), (.04, .12, .32)
    elif case == 'n8':
        deltas, taus = (.75,)*8, tuple(.025*1.5**j for j in range(8))
    if case == 'equal-times':
        taus = (.15,)*4
    elif case == 'all-zero':
        deltas = (0.,)*4
    elif case == 'negative-gain':
        deltas, taus, courant = (1e-5, 2e-5, 3e-5, 4e-5), (1e-5, 2e-5, 3e-5, 4e-5), .01
    elif case == 'gain-zero':
        deltas = (.01, .02, .03, .04)
    scene = film_scene(axis=axis, deltas=deltas, taus=taus, courant=courant)
    plan = quiet(fd.plan, scene)
    if case == 'gain-zero':
        taus = (plan.dt/2, .08, .16, .32)
        scene = film_scene(axis=axis, deltas=deltas, taus=taus, courant=courant)
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(t for t in plan.terms if t.name == TERM)
    coefficients = params['poles'][TERM]
    if case == 'gain-zero':
        assert any(np.any((to_numpy(gain) == 0.) & (to_numpy(drive) > 0))
                   for gain, drive in zip(coefficients['step_gain'], coefficients['drive']))
    if case == 'negative-gain':
        assert all(np.min(to_numpy(gain)) < 0 for gain in coefficients['step_gain'])
    rng = np.random.default_rng(279)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    old = tuple(rng.normal(size=(3,)+term.region.shape)+1j*rng.normal(size=(3,)+term.region.shape)
                for _ in range(term.n_poles))
    mem = {name: tuple(rt.array(value*scale, 'complex128') for value in old)
           for name, scale in (('previous', 1.), ('P', 1.7), ('delta', .7))}
    f = fills(scene, plan, axis)
    strengths, relaxations = np.zeros((term.n_poles,)+plan.shape), np.ones((term.n_poles,)+plan.shape)
    for component in range(3):
        for fraction in np.unique(f[component]):
            modes = rational_modes(fraction, component == axis, 4., 1., deltas, taus)
            mask = f[component] == fraction
            for pole, (delta, tau) in enumerate(modes):
                strengths[pole, component][mask] = delta
                relaxations[pole, component][mask] = tau
    density = np.zeros(plan.shape)
    for pole in range(term.n_poles):
        strength, tau = strengths[pole], relaxations[pole]
        shift = strength*plan.dt/(2*tau+plan.dt)
        inverse = np.where(strength > 0, 1/np.where(strength > 0, strength, 1.), 0.)
        full_old = np.zeros(plan.shape, complex)
        full_old[(slice(None),)+term.region.slices] = old[pole]
        density += abs(full_old+shift*field)**2*inverse-shift*abs(field)**2
    whole = tuple((part.start, part.stop) for part in term.region.slices)
    cut = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in whole)
    for cells, comps in ((whole, slice(None)), (cut, slice(axis, axis+1))):
        offsets = tuple(lo-part.start for (lo, _), part in zip(cells, term.region.slices))
        got = InteriorEnergy._pole(mem, rt.array(field, 'complex128'), coefficients,
                                   term.n_poles, cells, offsets, plan.dt, rt, comps=comps)
        expected = np.sum(density[(comps,)+tuple(slice(lo, hi) for lo, hi in cells)])
        assert np.asarray(to_numpy(got)).shape == ()
        assert float(to_numpy(got)) == pytest.approx(expected, rel=4e-13, abs=4e-13)


@pytest.mark.parametrize('backend', backends())
def test_oblique_debye_sum_film_matches_complex_airy_on_each_backend(backend):
    from studies.fdtd_debye_sum_film_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review the disclosed calibration before freezing ceilings.'
    row = measure(n=40, pol='p', periods=120, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS
    assert max(row['error_r']) < r_limit and max(row['error_t']) < t_limit
    assert row['stable'] is True and max(row['decay'].values()) < 1e-5
    assert row['energy_shape'] == [row['energy_rows']]
    assert row['energy_finite'] is True and row['energy_max'] > 0


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
    # Finite passive Debye sums are admitted; mixed pole families remain refused.
    elif case == 'appended-lorentz':
        own = own.replace(poles=own.poles+(fd.Lorentz(omega_0=5., gamma=1., strength=2.),))
    elif case == 'mixed-poles':
        own = own.replace(poles=(own.poles[0], fd.Lorentz(omega_0=5., gamma=1., strength=2.)))
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
    'magnetic', 'appended-lorentz', 'mixed-poles', 'background-loss', 'background-magnetic', 'partner',
    'initial', 'stencil', 'pml-normal', 'sub-dx', 'transverse-halo', 'transverse-absorber',
    'unrepresentable-strength', 'certificate-cap'))
def test_unsupported_debye_sum_films_are_refused_by_name(case, monkeypatch):
    if case in ('unrepresentable-strength', 'certificate-cap'):
        scene = film_scene(deltas=(math.ulp(0.0),)*3, taus=(.2,)*3)
        from wavebit.fd._lower import debye_sum_film
        if case == 'certificate-cap':
            # Clear successes explicitly so test order cannot bypass the
            # deliberately exhausted numerical admission safeguard.
            debye_sum_film._normal_modes.cache_clear()
            scene = film_scene(eps=4.375)
            monkeypatch.setattr(debye_sum_film, '_MAX_BISECTIONS', 0)
        try:
            with pytest.raises(ValueError, match='(?i)debye-sum-film.*float64'):
                plan = quiet(fd.plan, scene)
                fd.lower(scene, plan, 'numpy')
        finally:
            if case == 'certificate-cap':
                debye_sum_film._normal_modes.cache_clear()
        return
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('dispers', 'film', 'smooth', 'lossless', 'absorbing'))
    if case == 'stencil':
        assert 'smoothed' in message and 'fd.dispersive' in message


@pytest.mark.parametrize('change', ('delta0', 'delta2', 'delta3', 'tau0', 'tau2', 'tau3', 'pole-count', 'reversed-poles', 'kind',
    'eps', 'center', 'background', 'missing-memory', 'extra-memory', 'ordinary-memory',
    'nonfilm-plan', 'unsupported-nonfilm-plan', 'one-debye-plan', 'two-debye-plan', 'oscillator-plan', 'conductor-plan'))
def test_prepared_debye_sum_plan_refuses_changed_identity_or_memory(change):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    body, own = scene.structures[0].shape, scene.structures[0].material
    other = scene
    if change in ('missing-memory', 'extra-memory', 'ordinary-memory'):
        term = next(t for t in plan.terms if t.name == TERM)
        if change == 'ordinary-memory':
            from wavebit.fd._kernel.poles import PoleTerm
            terms = plan.terms+(PoleTerm(name='poles[0]', region=term.region,
                                         slabs=term.slabs, n_poles=2),)
        else:
            terms = (tuple(t for t in plan.terms if t is not term) if change == 'missing-memory'
                     else plan.terms+(dataclasses.replace(term, name='debye_sum_film[1]'),))
        plan = dataclasses.replace(plan, terms=terms)
    elif change in ('nonfilm-plan', 'unsupported-nonfilm-plan', 'one-debye-plan', 'two-debye-plan', 'oscillator-plan', 'conductor-plan'):
        previous = fd.Material(eps=4.)
        if change == 'one-debye-plan':
            previous = fd.Dispersive(eps_inf=4., poles=(fd.Debye(tau=.05, delta_eps=2.),))
        elif change == 'two-debye-plan':
            previous = material(deltas=(2., 4.), taus=(.05, .2))
        elif change == 'oscillator-plan':
            previous = fd.Dispersive(eps_inf=4., poles=(fd.Lorentz(omega_0=5., gamma=1., strength=20.),))
        elif change == 'conductor-plan':
            previous = fd.Conductor(eps=4., sigma=8.)
        plan = quiet(fd.plan, scene.replace(structures=(fd.Structure(body, previous),)))
        if change == 'unsupported-nonfilm-plan':
            body = fd.Box(size=(.2, .2, .54))
        other = scene.replace(structures=(fd.Structure(body, own),))
    elif change == 'background':
        other = scene.replace(background=fd.Material(eps=2.))
    elif change == 'center':
        other = scene.replace(structures=(fd.Structure(body.replace(center=(0., 0., .01)), own),))
    else:
        deltas, taus = [1., 1., 2., 2.], [.04, .08, .16, .32]
        if change.startswith('delta'):
            deltas[int(change[-1])] += 1.
        elif change.startswith('tau'):
            taus[int(change[-1])] *= 1.2
        elif change == 'pole-count':
            deltas, taus = deltas+[1.], taus+[.64]
        elif change == 'reversed-poles':
            deltas, taus = deltas[::-1], taus[::-1]
        own = material(5. if change == 'eps' else 4., deltas, taus)
        if change == 'kind':
            own = fd.Dispersive(eps_inf=4., poles=(fd.Lorentz(omega_0=5., gamma=1., strength=20.),))
        other = scene.replace(structures=(fd.Structure(body, own),))
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('delta0', 'delta2', 'delta3', 'tau0', 'tau2', 'tau3'))
def test_traced_any_debye_pole_cannot_bypass_the_concrete_door(mode, parameter):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    index = int(parameter[-1])
    initial = (1., 1., 2., 2.)[index] if parameter.startswith('delta') else (.04, .08, .16, .32)[index]

    def lower(value, backend):
        deltas, taus = [1., 1., 2., 2.], [.04, .08, .16, .32]
        (deltas if parameter.startswith('delta') else taus)[index] = value
        own = material(deltas=deltas, taus=taus)
        other = scene.replace(structures=(fd.Structure(scene.structures[0].shape, own),))
        return fd.lower(other, plan, backend)['be']

    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
            jax.jit(lambda value: lower(value, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(torch.tensor(initial, dtype=torch.float64),
                                                         torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
                    lower(dual, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')


@pytest.mark.parametrize('count', (3, 4, 8))
def test_debye_sum_cfl_uses_the_local_sum_and_harmonic_high_frequency_moment(count):
    deltas, taus = (9./count,)*count, tuple(.01*(1+j/count) for j in range(count))
    scene = film_scene(eps=1., deltas=deltas, taus=taus, courant=.01)
    plan = quiet(fd.plan, scene)
    f = fills(scene, plan, 2)
    # -[1/s**2] in the direct response is sum(delta_eff/tau_eff**2).
    B = sum(delta/tau for delta, tau in zip(deltas, taus))
    C = sum(delta/tau**2 for delta, tau in zip(deltas, taus))
    rate = f*C
    rate[2] = f[2]*(C+(1-f[2])*B**2)
    maximum = max(C, float(np.max(rate)))
    assert maximum > 1.5*C, 'bulk or single-pole maxima do not bound these mixed cells'
    vacuum = plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+maximum/4)
    got, reason = checks.cfl_bound(scene, plan)
    assert reason is None and got == pytest.approx(expected, rel=2e-13)
    unsafe = scene.replace(run=fd.Run(steps=2, courant=(expected/vacuum+1)/2))
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        quiet(fd.plan, unsafe)
    zero = film_scene(eps=1., deltas=(0.,)*4, taus=(1e-250, 1e250, 1e-200, 1e200))
    zero_plan = quiet(fd.plan, zero)
    bound, reason = checks.cfl_bound(zero, zero_plan)
    assert reason is None
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        checks.check_stability(zero, dataclasses.replace(zero_plan, dt=2*bound))


@pytest.mark.parametrize('case', ('band-high', 'monitor-phase', 'resolution-phase', 'pml-phase',
    'zero-strength-extreme-times', 'resolved-loss', 'traced-band-forward', 'traced-band-jax'))
def test_debye_sum_spatial_guard_covers_the_whole_sum_band_and_declared_points(case):
    from studies.fdtd_debye_sum_film_probe import build
    from wavebit.fd._lower.debye_sum_resolution import validate
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
    if case == 'band-high':
        scene, _ = build(n=20, eps=1., deltas=(2., 2., 2.), taus=(.15,)*3, courant=.01)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
            material(eps=1., deltas=(0.,)*3, taus=(.15,)*3)),))
        grid = quiet(fd.plan, safe).grid
        # A later diagnostic point refusal must not mask omission of a band pole.
        high = 2*math.pi*(1+math.sqrt(2*math.log(1e6))*.1)
        single = 1+2/(1-1j*high*.15)
        total = 1+6/(1-1j*high*.15)
        assert grid.dx*high*abs(single)**.5 < 2*math.pi/8
        assert grid.dx*high*abs(total)**.5 > 2*math.pi/8
        with pytest.raises(ValueError, match='(?i)source .* band.*bulk spectral mesh bound'):
            validate(scene, grid)
    elif case == 'monitor-phase':
        scene = scene.replace(monitors=tuple(m.replace(wavelengths=(.2,)) for m in scene.monitors))
    elif case == 'resolution-phase':
        scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=8, wavelength=.2))
    elif case == 'pml-phase':
        scene = scene.replace(boundaries=scene.boundaries.replace(
            z=fd.PML(cells=24, wavelength=.2, reflection=1e-12)))
    elif case == 'zero-strength-extreme-times':
        scene, _ = build(deltas=(0.,)*4, taus=(1e-250, 1e250, 1e-200, 1e200))
    if case in ('zero-strength-extreme-times', 'resolved-loss'):
        assert quiet(fd.plan, scene).dt > 0
    else:
        with pytest.raises(ValueError, match='(?i)wavelength|skin|band|spectral|cells'):
            quiet(fd.plan, scene)
