"""The bounded first C8 door: an axis-aligned conductive film in a dielectric.

The material oracle is the complex series/parallel law evaluated at the trapezoidal
frequency, not the production helper that realizes it. An oblique p-polarized film also
measures the actual update; coefficient agreement alone cannot certify stepping.
The oblique propagation/convergence experiment lives in
``studies/fdtd_conductive_film_probe.py`` and is a separate physical gate.
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


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        return function(*args, **kwargs)


def film_scene(axis=2, eps=4.0, sigma=8.0, host=1.0, shift=.03, dx=.1,
               courant=.3, steps=2, thin=False):
    """An isolated full-width film; both faces cut their own Yee dual cells."""
    lengths = [dx if thin else .4] * 3
    lengths[axis] = 2.0
    size = [length + 4 * dx for length in lengths]
    size[axis] = .54
    center = [0.0] * 3
    center[axis] = shift
    return fd.Scene(
        domain=fd.Domain(size=tuple(lengths)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(), z=fd.Periodic()),
        resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps, courant=courant),
        background=fd.Material(eps=host),
        structures=(fd.Structure(fd.Box(size=tuple(size), center=tuple(center)),
                                 fd.Conductor(eps=eps, sigma=sigma)),))


def fills(scene, plan, axis):
    """Interval overlaps at public grid points, independent of production occupancy."""
    body = scene.structures[0].shape
    low = float(body.center[axis]) - float(body.size[axis]) / 2
    high = float(body.center[axis]) + float(body.size[axis]) / 2
    out = []
    for component in range(3):
        points = np.asarray(plan.grid.yee_points('E', component)[axis])
        f = np.maximum(0.0, np.minimum(points + plan.dx / 2, high)
                       - np.maximum(points - plan.dx / 2, low)) / plan.dx
        shape = [1, 1, 1]
        shape[2 - axis] = len(f)
        out.append(np.broadcast_to(np.clip(f, 0.0, 1.0).reshape(shape), plan.shape[1:]))
    return np.asarray(out)


def recovered_response(plan, params, omega):
    """The response encoded by the update plus memory, for E[n] = exp(-i omega n dt).

    The update's permittivity already includes the instantaneous Debye contribution.
    Conductivity is trapezoidal; the auxiliary polarization is drive/(z - gain).
    No film-specific production formula is called here.
    """
    ae = np.broadcast_to(to_numpy(params['ae']), plan.shape)
    be = np.broadcast_to(to_numpy(params['be']), plan.shape)
    eps_update = plan.dt / plan.dx * (1 + ae) / (2 * be)
    sigma = (1 - ae) / (plan.dx * be)
    omega_hat = 2 * math.tan(omega * plan.dt / 2) / plan.dt
    value = np.asarray(eps_update + 1j * sigma / omega_hat, dtype=complex).copy()
    z = np.exp(-1j * omega * plan.dt)
    for term in plan.terms:
        if not hasattr(term, 'n_poles'):
            continue
        p = params['poles'][term.name]
        region = (slice(None),) + term.region.slices
        for pole in range(term.n_poles):
            gain, hold, drive = (to_numpy(p[key][pole])
                                 for key in ('step_gain', 'hold', 'drive'))
            value[region] += drive / (z - gain + hold / z)
    return value, omega_hat


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
@pytest.mark.parametrize('eps,sigma,host,shift', [
    (4.0, 8.0, 1.0, .03), (1.0, 8.0, 1.0, -.011),
    (2.25, 3.0, 1.44, .017), (4.0, 0.0, 1.0, .025),
])
def test_lowered_response_is_the_complex_series_parallel_law(axis, eps, sigma, host, shift):
    scene = film_scene(axis, eps, sigma, host, shift)
    plan = quiet(fd.plan, scene)
    params = fd.lower(scene, plan, 'numpy')
    f = fills(scene, plan, axis)
    assert np.any((f[axis] > 1e-8) & (f[axis] < 1 - 1e-8)), 'normal memory is exercised'
    for wavelength in (.7, 1.0, 1.4):
        got, omega_hat = recovered_response(plan, params, 2 * math.pi / wavelength)
        own = eps + 1j * sigma / omega_hat
        want = f * own + (1 - f) * host
        want[axis] = 1 / (f[axis] / own + (1 - f[axis]) / host)
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
        assert np.all(np.isfinite(got))
        # Endpoints must remain whole media, without a singular pole at f = 1.
        np.testing.assert_allclose(got[f < 1e-12], host, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(got[f > 1 - 1e-12], own, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('axis', range(3), ids=('x', 'y', 'z'))
def test_zero_conductivity_is_the_existing_lossless_box(axis):
    scene = film_scene(axis=axis, sigma=0.0)
    material = fd.Material(eps=4.0)
    lossless = scene.replace(structures=(fd.Structure(scene.structures[0].shape, material),))
    plan = quiet(fd.plan, scene)
    reference_plan = quiet(fd.plan, lossless)
    actual = fd.lower(scene, plan, 'numpy')
    reference = fd.lower(lossless, reference_plan, 'numpy')
    for name in ('ae', 'be', 'ah', 'bh'):
        np.testing.assert_allclose(np.broadcast_to(actual[name], plan.shape),
                                   np.broadcast_to(reference[name], plan.shape),
                                   atol=2e-14, rtol=2e-14)
    for coefficients in actual['poles'].values():
        for drive in coefficients['drive']:
            assert np.count_nonzero(drive) == 0


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('axis', (0, 2), ids=('x', 'z'))
def test_interface_energy_uses_local_weights_before_reducing(backend, axis):
    """Synthetic memory probes the exact energy identity and nontrivial observer slices.

    Random memory on inactive cells must contribute nothing. A component-only slice is
    the same operation the symmetry-plane correction uses, with the correct region offset.
    """
    scene = film_scene(axis=axis)
    plan = quiet(fd.plan, scene)
    rt = runtime(backend)
    params = fd.lower(scene, plan, rt)
    term = next(term for term in plan.terms if term.name == 'conductive[0]')
    coefficients = params['poles'][term.name]
    rng = np.random.default_rng(146)
    field = rng.normal(size=plan.shape) + 1j * rng.normal(size=plan.shape)
    old = rng.normal(size=(3,) + term.region.shape) + 1j * rng.normal(size=(3,) + term.region.shape)
    mem = {name: (rt.array(old * scale, 'complex128'),)
           for name, scale in (('previous', 1.0), ('P', 1.7), ('delta', .7))}
    fraction = fills(scene, plan, axis)
    active = (fraction > 1e-12) & (fraction < 1 - 1e-12)
    active[np.arange(3) != axis] = False
    f = np.where(active, fraction, .5)
    eps, sigma, host = 4.0, 8.0, 1.0
    q = f * host + (1 - f) * eps
    delta = f * host ** 2 / ((1 - f) * q)
    tau = q / ((1 - f) * sigma)
    c = np.where(active, delta * plan.dt / (2 * tau + plan.dt), 0.0)
    inverse = np.where(active, 1 / delta, 0.0)
    full_old = np.zeros(plan.shape, dtype=complex)
    full_old[(slice(None),) + term.region.slices] = old
    density = inverse * np.abs(full_old + c * field) ** 2 - c * np.abs(field) ** 2
    whole = tuple((s.start, s.stop) for s in term.region.slices)
    cut = list(whole)
    for a, (lo, hi) in enumerate(cut):
        if hi - lo > 2:
            cut[a] = (lo + 1, hi)
    for cells, comps in ((whole, slice(None)), (tuple(cut), slice(axis, axis + 1))):
        offsets = tuple(lo - s.start for (lo, _), s in zip(cells, term.region.slices))
        got = InteriorEnergy._pole(mem, rt.array(field, 'complex128'), coefficients,
                                   term.n_poles, cells, offsets, plan.dt, rt, comps=comps)
        wanted = float(np.sum(density[(comps,) + tuple(slice(lo, hi) for lo, hi in cells)]))
        assert np.asarray(to_numpy(got)).shape == ()
        assert float(to_numpy(got)) == pytest.approx(wanted, rel=3e-13, abs=3e-13)


@pytest.mark.parametrize('backend', backends())
def test_the_oblique_tm_film_matches_complex_airy_on_each_backend(backend):
    """Public source and monitor, with both normal and tangential electric response.

    Calibration at n40, theta35, 60 periods measured max complex errors .004528 in r
    and .009189 in t, with decay below 1.5e-8. The fixed regression ceilings below
    allow about ten percent margin. They assert this finite-grid response; the
    separately registered study decides convergence per wavelength and coefficient.
    """
    from studies.fdtd_conductive_film_probe import measure
    row = measure(n=40, pol='p', periods=60, backend=backend)
    assert max(row['error_r']) < .005
    assert max(row['error_t']) < .010
    assert row['stable'] is True and max(row['decay'].values()) < 1e-5
    assert row['energy_shape'] == [row['energy_rows']]
    assert row['energy_finite'] is True and row['energy_max'] > 0


def refused_scene(case):
    scene = film_scene()
    body, material = scene.structures[0].shape, scene.structures[0].material
    if case == 'finite-box':
        body = fd.Box(size=(.2, .2, .54))
        scene = scene.replace(domain=fd.Domain(size=(1.2, 1.2, 2.0)))
    elif case == 'sphere':
        body = fd.Sphere(radius=.15)
        scene = scene.replace(domain=fd.Domain(size=(1.2, 1.2, 2.0)))
    elif case == 'turned':
        body = body.replace(angle=15.0)
        scene = scene.replace(domain=fd.Domain(size=(2.0, 2.0, 2.0)))
    elif case == 'ranged':
        body = body.replace(range=((-1., 1.), (-1., 1.), (-.8, .8)))
    elif case == 'anisotropic':
        material = fd.Conductor(eps=(4., 5., 4.), sigma=8.0)
    elif case == 'anisotropic-sigma':
        material = fd.Conductor(eps=4., sigma=(8., 9., 8.))
    elif case == 'magnetic':
        material = fd.Conductor(eps=4.0, sigma=8.0, mu=2.0)
    elif case == 'pole':
        material = fd.Dispersive(eps_inf=4.0, poles=(fd.Debye(tau=.1),))
    elif case == 'background-loss':
        return scene.replace(background=fd.Conductor(eps=1.0, sigma=.2))
    elif case == 'background-magnetic':
        return scene.replace(background=fd.Material(eps=1.0, mu=2.0))
    elif case == 'partner':
        extra = fd.Structure(fd.Slab(z=(.5, .7)), fd.Material(eps=2.0))
        return scene.replace(structures=scene.structures + (extra,))
    elif case == 'initial':
        return scene.replace(initial=fd.InitialField(E=(0., 0., 1.)))
    elif case == 'sub-dx':
        body = body.replace(size=(.8, .8, .05))
    elif case == 'transverse-halo':
        body = body.replace(size=(.5, .8, .54))
    elif case == 'transverse-absorber':
        body = body.replace(size=(3.2, .8, .54))
        scene = scene.replace(domain=fd.Domain(size=(2.8, .4, 2.0)),
                              boundaries=fd.Boundaries(
                                  x=(fd.PML(cells=12, wavelength=1.), fd.PEC()),
                                  y=fd.Periodic(), z=fd.Periodic()))
    elif case == 'pml-normal':
        body = body.replace(center=(0., 0., -.6))
        scene = scene.replace(boundaries=fd.Boundaries(
            x=fd.Periodic(), y=fd.Periodic(),
            z=(fd.PML(cells=12, wavelength=1.0), fd.PEC())))
    else:
        raise AssertionError(case)
    return scene.replace(structures=(fd.Structure(body, material),))


@pytest.mark.parametrize('case', ('finite-box', 'sphere', 'turned', 'ranged',
                                  'anisotropic', 'anisotropic-sigma', 'magnetic', 'pole',
                                  'background-loss', 'background-magnetic', 'partner',
                                  'initial', 'pml-normal', 'sub-dx', 'transverse-halo',
                                  'transverse-absorber'))
def test_unsupported_smoothed_lossy_scenes_are_refused_by_name(case):
    with pytest.raises(ValueError) as caught:
        quiet(fd.plan, refused_scene(case))
    message = str(caught.value).lower()
    assert any(word in message for word in ('conduct', 'film', 'smooth', 'c8', 'absorbing'))


@pytest.mark.parametrize('change', ('sigma', 'eps', 'center', 'background', 'missing-memory',
                                    'extra-memory', 'nonfilm-plan', 'unsupported-nonfilm-plan'))
def test_prepared_film_plan_refuses_changed_material_or_geometry(change):
    scene = film_scene()
    plan = quiet(fd.plan, scene)
    body = scene.structures[0].shape
    if change in ('missing-memory', 'extra-memory'):
        other = scene
        term = next(term for term in plan.terms if term.name == 'conductive[0]')
        if change == 'missing-memory':
            plan = dataclasses.replace(plan, terms=tuple(t for t in plan.terms if t is not term))
        else:
            plan = dataclasses.replace(plan, terms=plan.terms + (dataclasses.replace(
                term, name='conductive[1]'),))
    elif change in ('nonfilm-plan', 'unsupported-nonfilm-plan'):
        nonfilm = scene.replace(structures=(fd.Structure(body, fd.Material(eps=4.)),))
        if change == 'unsupported-nonfilm-plan':
            body = fd.Box(size=(.2, .2, .54))
        plan = quiet(fd.plan, nonfilm)
        other = scene.replace(structures=(fd.Structure(body, scene.structures[0].material),))
    elif change == 'background':
        other = scene.replace(background=fd.Material(eps=2.0))
    elif change == 'center':
        other = scene.replace(structures=(fd.Structure(body.replace(center=(0., 0., .01)),
                                                      scene.structures[0].material),))
    else:
        material = fd.Conductor(eps=5.0 if change == 'eps' else 4.0,
                                sigma=9.0 if change == 'sigma' else 8.0)
        other = scene.replace(structures=(fd.Structure(body, material),))
    with pytest.raises(ValueError, match='(?i)plan|signature|concrete'):
        fd.lower(other, plan, 'numpy')


@pytest.mark.parametrize('mode', ('torch-reverse', 'torch-forward', 'jax-jit'))
def test_traced_conductivity_cannot_bypass_the_prepared_film_door(mode):
    scene = film_scene()
    plan = quiet(fd.plan, scene)

    def lower(sigma, backend):
        material = fd.Conductor(eps=4., sigma=sigma)
        traced = scene.replace(structures=(fd.Structure(scene.structures[0].shape, material),))
        return fd.lower(traced, plan, backend)['be']

    if mode == 'jax-jit':
        jax = pytest.importorskip('jax')
        with pytest.raises(ValueError, match='(?i)traced|concrete|conductive'):
            jax.jit(lambda sigma: lower(sigma, 'jax'))(8.)
    else:
        torch = pytest.importorskip('torch')
        if mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(
                    torch.tensor(8., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
                with pytest.raises(ValueError, match='(?i)traced|concrete|conductive'):
                    lower(dual, 'torch')
        else:
            with pytest.raises(ValueError, match='(?i)traced|concrete|conductive'):
                lower(torch.tensor(8., dtype=torch.float64, requires_grad=True), 'torch')


def test_derived_normal_relaxation_tightens_the_planning_bound():
    """Compare the public planning refusal with the independent derived pole's rate."""
    scene = film_scene(sigma=150.0, courant=.01)
    plan = quiet(fd.plan, scene)
    fraction = fills(scene, plan, 2)[2]
    f = fraction[(fraction > 1e-8) & (fraction < 1 - 1e-8)]
    q = f + (1 - f) * 4.0
    eps_inf = 4.0 / q
    tau = q / ((1 - f) * 150.0)
    delta = f / ((1 - f) * q)
    rate = float(np.max(delta / tau ** 2 / eps_inf))
    vacuum = plan.dx / math.sqrt(3.0)
    expected = 1 / math.sqrt(1 / vacuum ** 2 + rate / 4)
    got, reason = checks.cfl_bound(scene, plan)
    assert reason is None
    assert got == pytest.approx(expected, rel=2e-13)
    assert expected < .99 * vacuum
    unsafe = scene.replace(run=fd.Run(steps=2, courant=(expected / vacuum + 1) / 2))
    with pytest.raises(ValueError, match='(?i)stability|bound|courant'):
        quiet(fd.plan, unsafe)


def test_conductivity_is_counted_in_the_source_band_resolution_floor():
    """Large sigma shortens the wavelength even with positive real epsilon.

    The tiny courant keeps the independent derived-memory stability bound clear;
    this refusal must be the spatial material-resolution guard, not the CFL guard.
    """
    scene = film_scene(dx=.025, sigma=4000., courant=1e-5)
    source = fd.Dipole('kick', center=(0., 0., .6), component='Ex',
                       band=fd.Band(wavelength=1., df=.1))
    scene = scene.replace(sources=(source,))
    with pytest.raises(ValueError, match='cells per wavelength|skin depth'):
        quiet(fd.plan, scene)
    resolved = scene.replace(structures=(fd.Structure(scene.structures[0].shape,
                                                     fd.Conductor(eps=4., sigma=8.)),))
    assert quiet(fd.plan, resolved).dt > 0
