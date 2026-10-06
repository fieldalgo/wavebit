"""Bounded one-pole film gates with independent constitutive and optical oracles.

The recurrence is read from actual lowered parameters; its reference uses the two
centered-difference symbols, not production pole/coefficient helpers. End-to-end
regression ceilings are set only after the two disclosed optical calibrations.
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


# Disclosed calibration attempt1, n40/p60/PML24/fraction.3: max r/t errors
# Drude .002943/.001146, Lorentz .002826/.005958; both settled below 8.6e-8.
# Retain the prior conductive-film absolute regression ceilings. The separate
# registered study decides convergence; these ceilings certify only this grid.
CALIBRATED_CEILINGS = {'drude': (.005, .010), 'lorentz': (.005, .010)}


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def material(family='lorentz', eps=4., gamma=1., strength=20., omega0=5.):
    pole = (fd.Drude(omega_p=math.sqrt(strength), gamma=gamma) if family == 'drude'
            else fd.Lorentz(omega_0=omega0, gamma=gamma, strength=strength))
    return fd.Dispersive(eps_inf=eps, poles=(pole,))


def film_scene(axis=2, family='lorentz', eps=4., gamma=1., strength=20., omega0=5.,
               host=1., shift=.03, dx=.1, courant=.3):
    lengths = [.4] * 3
    lengths[axis] = 2.
    size = [length + 4*dx for length in lengths]
    size[axis] = .54
    center = [0.] * 3
    center[axis] = shift
    return fd.Scene(
        domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=2, courant=courant),
        background=fd.Material(eps=host),
        structures=(fd.Structure(fd.Box(size=tuple(size), center=tuple(center)),
                                 material(family, eps, gamma, strength, omega0)),))


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
    assert len(terms) == 1 and terms[0].name == 'dispersive_film[0]'
    term = terms[0]
    assert term.n_poles == 1
    coefficients = params['poles'][term.name]
    gain, hold, drive = (to_numpy(coefficients[key][0])
                         for key in ('step_gain', 'hold', 'drive'))
    value[(slice(None),) + term.region.slices] += drive/(z-gain+hold/z)
    return value


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('family,eps,gamma,strength,host,shift', [
    ('drude', 4., 1., 20., 1., .03), ('lorentz', 4., 1., 20., 1., .03),
    ('lorentz', 1., 0., 20., 1., -.011), ('drude', 2.25, 150., 20., 1.44, .017),
    ('lorentz', 4., 1., 0., 1., .025), ('lorentz', 4., 1., 20., 1., -.02+1e-12),
])
def test_lowered_single_pole_response_is_the_series_parallel_law(
        axis, family, eps, gamma, strength, host, shift):
    scene = film_scene(axis, family, eps, gamma, strength, host=host, shift=shift)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    f = fills(scene, plan, axis)
    assert np.any((f[axis] > 1e-8) & (f[axis] < 1-1e-8))
    term = next(t for t in plan.terms if t.name == 'dispersive_film[0]')
    local_fill = f[(slice(None),)+term.region.slices]
    drives = np.broadcast_to(to_numpy(params['poles'][term.name]['drive'][0]), local_fill.shape)
    np.testing.assert_array_equal(drives > 0, (local_fill > 0) & (strength > 0))
    if abs(shift+.02) < 1e-10:
        assert np.any((local_fill > 0) & (local_fill < 1e-9)), 'retain every positive fraction'
    for wavelength in (.7, 1., 1.4):
        omega = 2*math.pi/wavelength
        q = (0. if family == 'drude' else 25.) - 4*math.sin(omega*plan.dt/2)**2/plan.dt**2
        q -= 1j*gamma*math.sin(omega*plan.dt)/plan.dt
        own = eps + strength/q
        want = f*own+(1-f)*host
        want[axis] = 1/(f[axis]/own+(1-f[axis])/host)
        got = response(plan, params, omega)
        np.testing.assert_allclose(got, want, rtol=3e-12, atol=3e-12)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got[f == 0], host, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(got[f == 1], own, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
def test_zero_strength_at_nominal_resonance_is_the_lossless_box(axis):
    scene = film_scene(axis=axis, strength=0., gamma=0., omega0=2*math.pi, dx=.025)
    scene = scene.replace(resolution=fd.Resolution(cells_per_wavelength=40, wavelength=1.))
    reference = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                       fd.Material(eps=4.)),))
    plan, refplan = quiet(fd.plan, scene), quiet(fd.plan, reference)
    actual, expected = fd.lower(scene, plan, 'numpy'), fd.lower(reference, refplan, 'numpy')
    for name in ('ae', 'be', 'ah', 'bh'):
        np.testing.assert_allclose(np.broadcast_to(actual[name], plan.shape),
                                   np.broadcast_to(expected[name], plan.shape),
                                   rtol=2e-14, atol=2e-14)
    coefficients = actual['poles']['dispersive_film[0]']
    for key in ('drive', 'energy_inverse_strength', 'energy_resonance_over_strength', 'energy_active'):
        assert np.count_nonzero(to_numpy(coefficients[key][0])) == 0
    if axis == 2:
        from studies.fdtd_single_pole_film_probe import build
        from wavebit.fd._assemble import _round_trip
        diagnostic_scene, _ = build(family='lorentz', eps=4., strength=0., gamma=0.,
                                    omega0=2*math.pi, theta=0.)
        # This is the production diagnostic called after a PlaneWave run. Its
        # fallback length needs no source-column state to expose a 0/0 pole.
        duration = _round_trip(diagnostic_scene, diagnostic_scene.sources[0], None)
        assert duration == pytest.approx(4*diagnostic_scene.domain.size[2], rel=2e-15)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
@pytest.mark.parametrize('case', ('drude', 'lorentz', 'hold-zero', 'zero-strength'))
def test_single_pole_energy_weights_active_cells_before_reducing(backend, axis, case):
    strength = 0. if case == 'zero-strength' else 20.
    family = 'drude' if case == 'drude' else 'lorentz'
    scene = film_scene(axis=axis, family=family, strength=strength)
    plan = quiet(fd.plan, scene)
    if case == 'hold-zero':
        scene = film_scene(axis=axis, gamma=2/plan.dt)
        plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(t for t in plan.terms if t.name == 'dispersive_film[0]')
    coefficients = params['poles'][term.name]
    if case == 'hold-zero':
        assert np.max(abs(to_numpy(coefficients['hold'][0]))) < 3e-16
    rng = np.random.default_rng(147)
    field = rng.normal(size=plan.shape) + 1j*rng.normal(size=plan.shape)
    old = rng.normal(size=(3,) + term.region.shape) + 1j*rng.normal(size=(3,) + term.region.shape)
    new = 1.7*old + .3j
    mem = {'previous': (rt.array(old, 'complex128'),), 'P': (rt.array(new, 'complex128'),),
           'delta': (rt.array(new-old, 'complex128'),)}
    f = fills(scene, plan, axis)
    A = f + (1-f)*4.
    s = f*strength
    w0 = 0. if family == 'drude' else 25.
    omega_squared = np.full(plan.shape, w0)
    s[axis] = f[axis]*strength/A[axis]**2
    omega_squared[axis] += (1-f[axis])*strength/A[axis]
    active = s > 0
    safe = np.where(active, s, 1.)
    old_full, new_full = np.zeros(plan.shape, complex), np.zeros(plan.shape, complex)
    old_full[(slice(None),)+term.region.slices] = old
    new_full[(slice(None),)+term.region.slices] = new
    delta = new_full-old_full
    density = np.where(active, (abs(delta)**2/plan.dt**2
                               + omega_squared*np.real(new_full*np.conj(old_full)))/safe
                       - np.real(field*np.conj(delta)), 0.)
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
@pytest.mark.parametrize('family', ('drude', 'lorentz'))
def test_oblique_single_pole_film_matches_complex_airy_on_each_backend(backend, family):
    from studies.fdtd_single_pole_film_probe import measure
    assert family in CALIBRATED_CEILINGS, 'Review the disclosed calibration before setting ceilings.'
    row = measure(family=family, n=40, pol='p', periods=60, backend=backend)
    r_limit, t_limit = CALIBRATED_CEILINGS[family]
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
    elif case == 'two-poles':
        own = own.replace(poles=own.poles + (fd.Drude(omega_p=2., gamma=1.),))
    # Finite passive Debye sums are admitted; mixed pole families remain refused.
    elif case == 'three-mixed-poles':
        own = own.replace(poles=(fd.Debye(tau=.1), fd.Debye(tau=.2),
                                 fd.Lorentz(omega_0=5., gamma=1., strength=2.)))
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
    'magnetic', 'two-poles', 'three-mixed-poles', 'background-loss', 'background-magnetic', 'partner',
    'initial', 'stencil', 'pml-normal', 'sub-dx', 'transverse-halo', 'transverse-absorber'))
def test_unsupported_single_pole_films_are_refused_by_name(case):
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('dispers', 'film', 'smooth', 'lossless', 'absorbing'))
    if case == 'stencil':
        assert 'smoothed' in message and 'fd.dispersive' in message


@pytest.mark.parametrize('change', ('strength', 'gamma', 'frequency', 'kind', 'eps', 'center',
    'background', 'missing-memory', 'extra-memory', 'ordinary-memory', 'nonfilm-plan',
    'unsupported-nonfilm-plan'))
def test_prepared_single_pole_film_plan_refuses_changed_input_or_memory(change):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    body, own = scene.structures[0].shape, scene.structures[0].material
    if change in ('missing-memory', 'extra-memory', 'ordinary-memory'):
        other = scene
        term = next(t for t in plan.terms if t.name == 'dispersive_film[0]')
        if change == 'ordinary-memory':
            from wavebit.fd._kernel.poles import PoleTerm
            extra = PoleTerm(name='poles[0]', region=term.region, slabs=term.slabs, n_poles=1)
            terms = plan.terms+(extra,)
        else:
            terms = (tuple(t for t in plan.terms if t is not term) if change == 'missing-memory'
                     else plan.terms + (dataclasses.replace(term, name='dispersive_film[1]'),))
        plan = dataclasses.replace(plan, terms=terms)
    elif change in ('nonfilm-plan', 'unsupported-nonfilm-plan'):
        plan = quiet(fd.plan, scene.replace(structures=(fd.Structure(body, fd.Material(eps=4.)),)))
        if change == 'unsupported-nonfilm-plan':
            body = fd.Box(size=(.2, .2, .54))
        other = scene.replace(structures=(fd.Structure(body, own),))
    elif change == 'background':
        other = scene.replace(background=fd.Material(eps=2.))
    elif change == 'center':
        other = scene.replace(structures=(fd.Structure(body.replace(center=(0., 0., .01)), own),))
    else:
        own = material('drude' if change == 'kind' else 'lorentz',
                       eps=5. if change == 'eps' else 4., gamma=2. if change == 'gamma' else 1.,
                       strength=21. if change == 'strength' else 20.,
                       omega0=6. if change == 'frequency' else 5.)
        other = scene.replace(structures=(fd.Structure(body, own),))
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
def test_traced_single_pole_strength_cannot_bypass_the_concrete_door(mode):
    scene = film_scene()
    plan = quiet(fd.plan, scene)

    def lower(strength, backend):
        own = fd.Dispersive(eps_inf=4., poles=(fd.Lorentz(omega_0=5., gamma=1., strength=strength),))
        other = scene.replace(structures=(fd.Structure(scene.structures[0].shape, own),))
        return fd.lower(other, plan, backend)['be']

    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive'):
            jax.jit(lambda s: lower(s, 'jax'))(20.)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                value = torch.autograd.forward_ad.make_dual(
                    torch.tensor(20., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive'):
                    lower(value, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|dispersive'):
                lower(torch.tensor(20., dtype=torch.float64, requires_grad=True), 'torch')


@pytest.mark.parametrize('family', ('drude', 'lorentz'))
def test_single_pole_film_uses_the_proved_material_stability_bound(family):
    scene = film_scene(family=family, eps=1., strength=4000., omega0=40., courant=.01)
    plan = quiet(fd.plan, scene)
    vacuum = plan.dx/math.sqrt(3.)
    rate = 4000.+(0. if family == 'drude' else 1600.)
    expected = 1/math.sqrt(1/vacuum**2+rate/4)
    got, reason = checks.cfl_bound(scene, plan)
    assert reason is None and got == pytest.approx(expected, rel=2e-13)
    assert expected < .8*vacuum
    unsafe = scene.replace(run=fd.Run(steps=2, courant=(expected/vacuum+1)/2))
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        quiet(fd.plan, unsafe)


@pytest.mark.parametrize('case', ('interior-band', 'zero-damping-resonance', 'monitor-phase',
    'monitor-skin', 'zero-strength-resonance', 'resolved-metal', 'traced-band-forward',
    'traced-band-jax', 'extreme-critical'))
def test_single_pole_spatial_guard_covers_the_band_and_requested_wavelengths(case):
    """The tested geometry/source are valid before the material-specific mesh refusal."""
    scene = film_scene(eps=1., omega0=8., gamma=.001, strength=40., dx=.05, courant=.001)
    band = fd.Band(wavelength=1., df=.1, cutoff=1e-6)
    source = fd.Dipole('kick', center=(0., 0., .7), component='Ex', band=band)
    scene = scene.replace(sources=(source,))
    if case == 'extreme-critical':
        from wavebit.fd._lower.single_pole_film import descriptor
        from wavebit.fd._lower.single_pole_resolution import _band_bound
        base = film_scene()
        plan = quiet(fd.plan, base)
        film = dataclasses.replace(descriptor(base, plan.grid), eps=1.,
                                   omega0_squared=1e160, gamma=1e78, strength=1e160)
        x = 1/(1-.01**2/2)
        ratio = x/math.sqrt((1-x)**2+.01**2*x)
        expected = math.sqrt(1.1**2+ratio)
        assert _band_bound(film, .9e80, 1.1e80)/1e80 == pytest.approx(expected, rel=2e-13)
        return
    if case.startswith('traced-band'):
        from wavebit.fd._lower.single_pole_resolution import validate
        base = film_scene(family='drude', eps=1., strength=81., gamma=1., dx=.025)
        grid = quiet(fd.plan, base).grid

        def check(df):
            incoming = source.replace(band=fd.Band(wavelength=1., df=df, cutoff=1e-6))
            return validate(base.replace(sources=(incoming,)), grid)

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
    if case in ('monitor-phase', 'monitor-skin'):
        # The asked frequency is outside the narrow source band; it still needs a mesh.
        wavelength = 2*math.pi/(8.-.001 if case == 'monitor-phase' else 8.+.001)
        # A real public Diffraction scene avoids source/periodic-size fixture errors.
        from studies.fdtd_single_pole_film_probe import build
        scene, _ = build(family='lorentz', n=40, pol='p', gamma=.001)
        own = material('lorentz', eps=1., gamma=.001, strength=400., omega0=8.)
        scene = scene.replace(structures=(fd.Structure(scene.structures[0].shape, own),),
            sources=(scene.sources[0].replace(band=fd.Band(wavelength=1., df=.001, cutoff=1e-6)),),
            monitors=tuple(m.replace(wavelengths=(wavelength,)) for m in scene.monitors))
    elif case == 'zero-damping-resonance':
        scene = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                       material(gamma=0., eps=1., omega0=8., strength=40.)),))
    elif case == 'zero-strength-resonance':
        scene = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                       material(gamma=0., eps=1., omega0=8., strength=0.)),))
    elif case == 'resolved-metal':
        scene = scene.replace(resolution=fd.Resolution(dx=.025), structures=(fd.Structure(
            scene.structures[0].shape, material('drude', eps=1., strength=81., gamma=1.)),))
    if case in ('zero-strength-resonance', 'resolved-metal'):
        assert quiet(fd.plan, scene).dt > 0
    else:
        with pytest.raises(ValueError, match='(?i)wavelength|skin|band|resonan|spectral|cells'):
            quiet(fd.plan, scene)
