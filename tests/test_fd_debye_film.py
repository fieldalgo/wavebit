"""Bounded Debye film gates against independent mixture, energy and optical laws.

Discrete constitutive references use the trapezoidal frequency, not the production
mixture helper. Optical ceilings retain the inherited values after the disclosed calibration.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.observe import InteriorEnergy
from wavebit.fd._lower import checks
from wavebit.fd._runtime import runtime
from fd_helpers import backends, to_numpy


# Disclosed calibration attempt1, n40/p120/PML24/fraction.3, all three backends:
# max absolute r/t errors .003564006/.008260100; largest decay ratio 8.6143e-9.
# Raw hashes/results: studies/results/fdtd_debye_film_calibration_attempt1.json.
# Retain the inherited candidate ceilings unchanged; this is a single-grid
# regression, and the separately registered study decides convergence.
CANDIDATE_CEILINGS = (.005, .010)
CALIBRATED_CEILINGS = (.005, .010)


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def material(eps=4., delta=3., tau=.15):
    return fd.Dispersive(eps_inf=eps, poles=(fd.Debye(tau=tau, delta_eps=delta),))


def film_scene(axis=2, eps=4., delta=3., tau=.15, host=1., shift=.03, dx=.1, courant=.3):
    lengths = [.4]*3
    lengths[axis] = 2.
    size = [length+4*dx for length in lengths]
    size[axis] = .54
    center = [0.]*3
    center[axis] = shift
    return fd.Scene(domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=2, courant=courant),
        background=fd.Material(eps=host),
        structures=(fd.Structure(fd.Box(size=tuple(size), center=tuple(center)),
                                 material(eps, delta, tau)),))


def fills(scene, plan, axis):
    """Direct interval geometry at each component's public Yee coordinates."""
    shape = scene.structures[0].shape
    low, high = (float(shape.center[axis]) + sign*float(shape.size[axis])/2
                 for sign in (-1, 1))
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
    assert len(terms) == 1 and terms[0].name == 'debye_film[0]'
    term = terms[0]
    assert term.n_poles == 1
    coefficients = params['poles'][term.name]
    gain, hold, drive = (to_numpy(coefficients[key][0])
                         for key in ('step_gain', 'hold', 'drive'))
    value[(slice(None),) + term.region.slices] += drive/(z-gain+hold/z)
    return value


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('eps,delta,tau,host,shift', [
    (4., 3., .15, 1., .03), (1., 9., .15, 1., -.011),
    (2.25, 1.5, .4, 1.44, .017), (4., 0., .15, 1., .025),
    (4., 3., .15, 1., -.02+1e-12), (1., 1e-42, 1e-20, 1., .03),
    (4., 3., 1e18, 1., .03),
])
def test_lowered_debye_response_is_the_series_parallel_law(axis, eps, delta, tau, host, shift):
    scene = film_scene(axis, eps, delta, tau, host, shift)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    f = fills(scene, plan, axis)
    assert np.any((f[axis] > 1e-8) & (f[axis] < 1-1e-8))
    term = next(t for t in plan.terms if t.name == 'debye_film[0]')
    local_fill = f[(slice(None),)+term.region.slices]
    coefficients = params['poles'][term.name]
    drive = np.broadcast_to(to_numpy(coefficients['drive'][0]), local_fill.shape)
    np.testing.assert_array_equal(drive > 0, (local_fill > 0) & (delta > 0))
    np.testing.assert_array_equal(to_numpy(coefficients['hold'][0]), 0.)
    if tau < 1e-18:
        gain = np.broadcast_to(to_numpy(coefficients['step_gain'][0]), local_fill.shape)
        assert np.any((gain == -1.) & (drive > 0)), 'do not reconstruct drive from 1+gain'
    if abs(shift+.02) < 1e-10:
        assert np.any((local_fill > 0) & (local_fill < 1e-9)), 'retain every positive fraction'
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        omega_hat = 2*math.tan(omega*plan.dt/2)/plan.dt
        own = eps+delta/(1-1j*omega_hat*tau)
        want = f*own+(1-f)*host
        want[axis] = 1/(f[axis]/own+(1-f[axis])/host)
        got = response(plan, params, omega)
        np.testing.assert_allclose(got, want, rtol=3e-12, atol=3e-12)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got[f == 0], host, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(got[f == 1], own, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('tau', (1e-200, 1e200), ids=('tiny-tau', 'huge-tau'))
def test_zero_debye_strength_is_lossless_even_at_extreme_relaxation_times(axis, tau):
    scene = film_scene(axis=axis, delta=0., tau=tau)
    reference = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                       fd.Material(eps=4.)),))
    plan, refplan = quiet(fd.plan, scene), quiet(fd.plan, reference)
    actual, expected = fd.lower(scene, plan, 'numpy'), fd.lower(reference, refplan, 'numpy')
    for name in ('ae', 'be', 'ah', 'bh'):
        np.testing.assert_allclose(np.broadcast_to(actual[name], plan.shape),
                                   np.broadcast_to(expected[name], plan.shape),
                                   rtol=2e-14, atol=2e-14)
    assert checks.cfl_bound(scene, plan) == checks.cfl_bound(reference, refplan)
    coefficients = actual['poles']['debye_film[0]']
    for key in ('drive', 'energy_inverse_delta', 'energy_instantaneous'):
        assert np.count_nonzero(to_numpy(coefficients[key][0])) == 0
    if axis == 2:
        from studies.fdtd_debye_film_probe import build
        from wavebit.fd._assemble import _round_trip
        diagnostic, _ = build(eps=4., delta=0., tau=tau, theta=0.)
        duration = _round_trip(diagnostic, diagnostic.sources[0], None)
        assert duration == pytest.approx(4*diagnostic.domain.size[2], rel=2e-15)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('ordinary', 'gain-zero', 'negative-gain', 'zero-strength'))
def test_debye_energy_uses_local_split_weights_before_reducing(backend, axis, case):
    delta, tau, courant = 3., .15, .3
    if case == 'zero-strength':
        delta = 0.
    elif case == 'negative-gain':
        delta, tau, courant = 1e-5, 1e-5, .01
    elif case == 'gain-zero':
        delta = .01
    scene = film_scene(axis=axis, delta=delta, tau=tau, courant=courant)
    plan = quiet(fd.plan, scene)
    if case == 'gain-zero':
        tau = plan.dt/2
        scene = film_scene(axis=axis, delta=delta, tau=tau, courant=courant)
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(t for t in plan.terms if t.name == 'debye_film[0]')
    coefficients = params['poles'][term.name]
    gain = to_numpy(coefficients['step_gain'][0])
    if case == 'gain-zero':
        assert np.any((gain == 0.) & (to_numpy(coefficients['drive'][0]) > 0))
    if case == 'negative-gain':
        assert np.min(gain) < 0.
    rng = np.random.default_rng(148)
    field = rng.normal(size=plan.shape)+1j*rng.normal(size=plan.shape)
    old = rng.normal(size=(3,)+term.region.shape)+1j*rng.normal(size=(3,)+term.region.shape)
    mem = {name: (rt.array(old*scale, 'complex128'),)
           for name, scale in (('previous', 1.), ('P', 1.7), ('delta', .7))}
    f = fills(scene, plan, axis)
    A = f+(1-f)*4.
    B = A+(1-f)*delta
    strength = f*delta
    relaxation = np.full(plan.shape, tau)
    strength[axis] = f[axis]*delta/(A[axis]*B[axis])
    relaxation[axis] = A[axis]*tau/B[axis]
    active = strength > 0
    shift = strength*plan.dt/(2*relaxation+plan.dt)
    inverse = np.where(active, 1/np.where(active, strength, 1.), 0.)
    full_old = np.zeros(plan.shape, complex)
    full_old[(slice(None),)+term.region.slices] = old
    density = abs(full_old+shift*field)**2*inverse-shift*abs(field)**2
    whole = tuple((s.start, s.stop) for s in term.region.slices)
    cut = tuple((lo+1 if hi-lo > 2 else lo, hi) for lo, hi in whole)
    for cells, comps in ((whole, slice(None)), (cut, slice(axis, axis+1))):
        offsets = tuple(lo-s.start for (lo, _), s in zip(cells, term.region.slices))
        got = InteriorEnergy._pole(mem, rt.array(field, 'complex128'), coefficients,
                                   term.n_poles, cells, offsets, plan.dt, rt, comps=comps)
        expected = np.sum(density[(comps,)+tuple(slice(lo, hi) for lo, hi in cells)])
        assert np.asarray(to_numpy(got)).shape == ()
        assert float(to_numpy(got)) == pytest.approx(expected, rel=4e-13, abs=4e-13)


@pytest.mark.parametrize('backend', backends())
def test_oblique_debye_film_matches_complex_airy_on_each_backend(backend):
    from studies.fdtd_debye_film_probe import measure
    assert CALIBRATED_CEILINGS is not None, 'Review the disclosed calibration before freezing ceilings.'
    row = measure(n=40, pol='p', periods=120, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS
    assert max(row['error_r']) < r_limit
    assert max(row['error_t']) < t_limit
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
    elif case == 'three-mixed-poles':
        own = own.replace(poles=own.poles + (fd.Debye(tau=.2),
                                             fd.Lorentz(omega_0=5., gamma=1., strength=2.)))
    elif case == 'mixed-poles':
        own = own.replace(poles=own.poles+(fd.Lorentz(omega_0=5., gamma=1., strength=2.),))
    elif case == 'background-loss':
        return scene.replace(background=fd.Conductor(eps=1., sigma=.2))
    elif case == 'background-magnetic':
        return scene.replace(background=fd.Material(eps=1., mu=2.))
    elif case == 'partner':
        return scene.replace(structures=scene.structures + (
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
    'magnetic', 'three-mixed-poles', 'mixed-poles', 'background-loss', 'background-magnetic', 'partner',
    'initial', 'stencil', 'pml-normal', 'sub-dx', 'transverse-halo', 'transverse-absorber'))
def test_unsupported_debye_films_are_refused_by_name(case):
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('dispers', 'film', 'smooth', 'lossless', 'absorbing'))
    if case == 'stencil':
        assert 'smoothed' in message and 'fd.dispersive' in message


@pytest.mark.parametrize('change', ('delta', 'tau', 'kind', 'eps', 'center', 'background',
    'missing-memory', 'extra-memory', 'ordinary-memory', 'nonfilm-plan',
    'unsupported-nonfilm-plan', 'oscillator-plan', 'conductor-plan'))
def test_prepared_debye_film_plan_refuses_changed_input_or_memory(change):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    body, own = scene.structures[0].shape, scene.structures[0].material
    if change in ('missing-memory', 'extra-memory', 'ordinary-memory'):
        other = scene
        term = next(t for t in plan.terms if t.name == 'debye_film[0]')
        if change == 'ordinary-memory':
            from wavebit.fd._kernel.poles import PoleTerm
            extra = PoleTerm(name='poles[0]', region=term.region, slabs=term.slabs, n_poles=1)
            terms = plan.terms+(extra,)
        else:
            terms = (tuple(t for t in plan.terms if t is not term) if change == 'missing-memory'
                     else plan.terms+(dataclasses.replace(term, name='debye_film[1]'),))
        plan = dataclasses.replace(plan, terms=terms)
    elif change in ('nonfilm-plan', 'unsupported-nonfilm-plan', 'oscillator-plan', 'conductor-plan'):
        previous = fd.Material(eps=4.)
        if change == 'oscillator-plan':
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
        own = material(eps=5. if change == 'eps' else 4., delta=4. if change == 'delta' else 3.,
                       tau=.2 if change == 'tau' else .15)
        if change == 'kind':
            own = fd.Dispersive(eps_inf=4., poles=(fd.Lorentz(omega_0=5., gamma=1., strength=20.),))
        other = scene.replace(structures=(fd.Structure(body, own),))
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
@pytest.mark.parametrize('parameter', ('delta', 'tau'))
def test_traced_debye_parameters_cannot_bypass_the_concrete_door(mode, parameter):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    initial = 3. if parameter == 'delta' else .15

    def lower(value, backend):
        own = material(delta=value if parameter == 'delta' else 3.,
                       tau=value if parameter == 'tau' else .15)
        other = scene.replace(structures=(fd.Structure(scene.structures[0].shape, own),))
        return fd.lower(other, plan, backend)['be']

    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
            jax.jit(lambda v: lower(v, 'jax'))(initial)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                value = torch.autograd.forward_ad.make_dual(
                    torch.tensor(initial, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
                    lower(value, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive|debye'):
                lower(torch.tensor(initial, dtype=torch.float64, requires_grad=True), 'torch')


def test_debye_film_cfl_policy_covers_the_actual_transformed_nodes():
    scene = film_scene(eps=1., delta=9., tau=.01, courant=.01)
    plan = quiet(fd.plan, scene)
    f = fills(scene, plan, 2)
    A = f+(1-f)
    B = A+(1-f)*9.
    rate = f*9./.01**2
    rate[2] = f[2]*9.*B[2]/(A[2]**2*.01**2)
    bulk = 9./.01**2
    maximum = max(bulk, float(np.max(rate)))
    assert maximum > 2*bulk, 'the declared bulk policy alone is insufficient'
    vacuum = plan.dx/math.sqrt(3.)
    expected = 1/math.sqrt(1/vacuum**2+maximum/4)
    got, reason = checks.cfl_bound(scene, plan)
    assert reason is None and got == pytest.approx(expected, rel=2e-13)
    unsafe = scene.replace(run=fd.Run(steps=2, courant=(expected/vacuum+1)/2))
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        quiet(fd.plan, unsafe)
    # Refusal diagnostics must also avoid inactive Debye.rate() tau**2 underflow.
    zero = film_scene(eps=1., delta=0., tau=1e-250)
    zero_plan = quiet(fd.plan, zero)
    bound, reason = checks.cfl_bound(zero, zero_plan)
    assert reason is None
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        checks.check_stability(zero, dataclasses.replace(zero_plan, dt=2*bound))


@pytest.mark.parametrize('case', ('band-high', 'monitor-phase', 'resolution-phase', 'pml-phase',
    'zero-strength-tiny-tau', 'resolved-loss', 'traced-band-forward', 'traced-band-jax'))
def test_debye_spatial_guard_covers_source_band_and_requested_points(case):
    from studies.fdtd_debye_film_probe import build
    scene, _ = build()
    if case.startswith('traced-band'):
        from wavebit.fd._lower.debye_resolution import validate
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
                dual = torch.autograd.forward_ad.make_dual(
                    torch.tensor(.1, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)concrete|traced|band'):
                    check(dual)
        return
    if case == 'band-high':
        from wavebit.fd._lower.debye_resolution import validate
        scene, _ = build(n=20, eps=1., delta=6., tau=.15, courant=.01)
        safe = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                      material(eps=1., delta=0., tau=.15)),))
        grid = quiet(fd.plan, safe).grid
        # Isolate the source-band certificate before the planner's later notes
        # independently check its shortest wavelength. A generic planning refusal
        # alone lets that secondary phase check hide an omitted band contribution.
        with pytest.raises(ValueError, match='(?i)source .* band.*bulk spectral mesh bound'):
            validate(scene, grid)
    elif case == 'monitor-phase':
        scene = scene.replace(monitors=tuple(m.replace(wavelengths=(.2,)) for m in scene.monitors))
    elif case == 'resolution-phase':
        scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=8, wavelength=.2))
    elif case == 'pml-phase':
        scene = scene.replace(boundaries=scene.boundaries.replace(
            z=fd.PML(cells=24, wavelength=.2, reflection=1e-12)))
    elif case == 'zero-strength-tiny-tau':
        scene, _ = build(delta=0., tau=1e-200)
    if case in ('zero-strength-tiny-tau', 'resolved-loss'):
        assert quiet(fd.plan, scene).dt > 0
    else:
        with pytest.raises(ValueError, match='(?i)wavelength|skin|band|spectral|cells'):
            quiet(fd.plan, scene)
