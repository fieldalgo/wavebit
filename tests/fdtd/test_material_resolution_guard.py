"""Live material resolution certificates use the eager optical readings."""
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from wavebit.fdtd import _legacy as fd
from wavebit.fdtd._prepare.checks import _index_at
from wavebit.fdtd._prepare.shape_ranges import resolution_guard, _resolution_index
from wavebit.fdtd._runtime import runtime


def scene(material, wavelength=1.):
    return fd.Scene(domain=fd.Domain(size=(1.,)*3),
        resolution=fd.Resolution(cells_per_wavelength=20., wavelength=wavelength) if wavelength is not None else fd.Resolution(dx=.05),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        run=fd.Run(steps=2), structures=(fd.Structure(
            fd.Region(size=(.4,)*3), material),))


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# Each pair straddles the actual eager threshold at wavelength 1, dx .05.
CASES = [
    ('plain', 6.24, True), ('plain', 6.25, True), ('plain', 6.26, False),
    ('conductor', 6.24, True), ('conductor', 6.26, False),
    ('debye', 8.90, True), ('debye', 8.93, False),
    ('drude', 20.8, True), ('drude', 21., False),
    ('drude_inf', 20., True), ('drude_inf', 28., False),
    ('inactive_lorentz', 6.24, True), ('inactive_lorentz', 6.26, False),
    ('active_lorentz', 600., True), ('active_lorentz', 700., False),
]


def material(kind, x):
    if kind == 'plain':
        return fd.Material(eps=x)
    if kind == 'conductor':
        # Outside conductive-film scope eager deliberately reads eps, not sigma.
        return fd.Conductor(eps=x, sigma=100.)
    if kind == 'debye':
        return fd.Dispersive(eps_inf=1., poles=(fd.Debye(delta_eps=x, tau=1/(2*math.pi)),))
    if kind == 'active_lorentz':
        return fd.Dispersive(eps_inf=1., poles=(fd.Lorentz(
            omega_0=4*math.pi, gamma=.5, strength=x),))
    if kind == 'inactive_lorentz':
        return fd.Dispersive(eps_inf=x, poles=(fd.Lorentz(
            omega_0=2*math.pi, gamma=0., strength=0.),))
    return fd.Dispersive(eps_inf=9. if kind == 'drude_inf' else 1.,
                         poles=(fd.Drude(omega_p=x, gamma=0.),))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind,x,accepted', CASES)
def test_threshold_matches_eager_and_poisoned_coefficients(backend, kind, x, accepted):
    rt = runtime(backend)
    seed = quiet(fd.plan, scene(material(kind, CASES[[c[0] for c in CASES].index(kind)][1])))
    concrete = scene(material(kind, x))
    if accepted:
        quiet(fd.plan, concrete)
    else:
        with pytest.raises(ValueError, match='cells per wavelength|skin depth'):
            quiet(fd.plan, concrete)

    def objective(params):
        return params['be'].sum() * (1 + sum(d for pole in params['poles'].values()
                                              for d in pole['drive']))

    def fn(v):
        return objective(quiet(fd.lower, scene(material(kind, v)), seed, backend))
    want = objective(quiet(fd.lower, concrete, seed)) if accepted else None
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        compiled = jax.jit(fn)
        value = float(compiled(jnp.asarray(x)))
        grad = float(jax.grad(fn)(jnp.asarray(x)))
    else:
        import torch
        v = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = fn(v)
        value = float(out.detach())
        grad = float(torch.autograd.grad(out, v)[0])
    if accepted:
        assert value == pytest.approx(want, rel=1e-12)
        h = 1e-5
        def host(v):
            s = scene(material(kind, v))
            return objective(quiet(fd.lower, s, seed))
        expected = (host(x+h)-host(x-h))/(2*h)
        assert expected != 0
        assert grad == pytest.approx(expected, rel=2e-6)
    else:
        assert np.isnan(value)
        assert np.isnan(grad)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_index_selects_same_complex_component_and_conductivity_scope(backend):
    rt = runtime(backend)
    mat = fd.Conductor(eps=(2., 3., 4.), mu=(2., 1., 1.), sigma=(1., 80., 2.))
    for conductive in (False, True):
        n, skin = _index_at(mat, 1., conductive=conductive)
        real, imag = _resolution_index(mat, 1., rt, conductive=conductive)
        assert float(real) == pytest.approx(n, rel=1e-14)
        if skin is not None:
            assert 1/(2*math.pi*float(imag)) == pytest.approx(skin, rel=1e-14)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_no_source_pml_fallback_and_previous_guard(backend):
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=(('z-', 1.),))
    def fn(x):
        return resolution_guard(scene(fd.Material(eps=x), wavelength=None), plan, rt,
                                rt.array(2., 'float64'))
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        fn = jax.jit(fn)
        values = [float(fn(jnp.asarray(x))) for x in (6.24, 6.26)]
    else:
        import torch
        values = [float(fn(torch.tensor(x, requires_grad=True))) for x in (6.24, 6.26)]
    assert values[0] == 2.
    assert np.isnan(values[1])


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('reader', ['band', 'diffraction'])
def test_dispersive_rule_wavelengths_use_live_optical_index(backend, reader):
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=())
    def make(x):
        s = scene(material('debye', x), wavelength=None)
        if reader == 'band':
            return s.replace(sources=(fd.Dipole('d', center=(0.,)*3, component='Ey',
                band=fd.Band(wavelength=1., df=.01)),))
        return s.replace(monitors=(fd.Diffraction('T', z=0., side='transmit',
                                                 wavelengths=(1., .8), orders=0),))
    from wavebit.fdtd._prepare.sources import shortest_wavelength
    waves = (shortest_wavelength(make(1.)),) if reader == 'band' else (1., .8)
    def fn(x):
        return resolution_guard(make(x), plan, rt, None)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        compiled = jax.jit(fn)
        evaluate = lambda x: float(compiled(jnp.asarray(x)))
    else:
        import torch
        evaluate = lambda x: float(fn(torch.tensor(x, dtype=torch.float64, requires_grad=True)))
    for x in (1., 15.):
        accepted = all(w/(_index_at(material('debye', x), w)[0]*plan.dx) >= 8 for w in waves)
        assert np.isfinite(evaluate(x)) == accepted


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_source_free_homogeneous_vacuum_keeps_eager_report_exemption(backend):
    rt = runtime(backend)
    def make(x):
        return scene(fd.Material(eps=1.)).replace(
            background=fd.Material(eps=x), structures=(),
            resolution=fd.Resolution(cells_per_wavelength=4., wavelength=1.))
    plan = quiet(fd.plan, make(1.))
    with pytest.raises(ValueError, match='cells per wavelength'):
        quiet(fd.plan, make(1.01))
    def fn(x):
        return resolution_guard(make(x), plan, rt, None)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        compiled = jax.jit(fn)
        values = [float(compiled(jnp.asarray(x))) for x in (1., 1.01)]
    else:
        import torch
        values = [float(fn(torch.tensor(x, requires_grad=True))) for x in (1., 1.01)]
    assert values[0] == 1.
    assert np.isnan(values[1])


def backend_call(backend, fn):
    """A live double scalar, compiled on JAX and carried by reverse mode on Torch."""
    runtime(backend)
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        compiled = jax.jit(fn)
        return lambda x: np.asarray(compiled(jnp.asarray(x, dtype=jnp.float64)))
    import torch
    return lambda x: fn(torch.tensor(x, dtype=torch.float64, requires_grad=True)).detach().numpy()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('kind,boundary', [('plain', 6.25),
    ('drude', math.sqrt(400 + 4*math.pi**2))])
def test_inclusive_doors_and_nearest_representable_neighbours(backend, kind, boundary):
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=())
    evaluate = backend_call(backend, lambda x: resolution_guard(
        scene(material(kind, x)), plan, rt, None))
    # Adjacent eps values can have the same rounded sqrt; compare the eager
    # decision itself, not a presumed exact-real boundary or a tolerance band.
    points = [boundary]
    for direction in (-np.inf, np.inf):
        x = boundary
        for _ in range(4):
            x = float(np.nextafter(x, direction))
            points.append(x)
    admitted = []
    for x in points:
        n, skin = _index_at(material(kind, x), 1.)
        expected = skin/plan.dx >= 1. if skin is not None else 1./(n*plan.dx) >= 8.
        admitted.append(expected)
        assert bool(np.isfinite(evaluate(x))) == expected, (backend, kind, x)
    assert admitted[0] and any(admitted) and not all(admitted)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('monitor_wavelength', [None, 1., .5, 2.])
def test_vacuum_report_exemption_does_not_exempt_diffraction(backend, monitor_wavelength):
    from wavebit.fdtd._prepare.checks import check_wavelength_resolution, _resolution_notes
    rt = runtime(backend)
    concrete = scene(fd.Material(eps=1.)).replace(
        structures=(), resolution=fd.Resolution(cells_per_wavelength=4., wavelength=1.))
    seed = quiet(fd.plan, concrete)
    if monitor_wavelength is not None:
        concrete = concrete.replace(monitors=(fd.Diffraction('T', z=0., side='transmit',
            wavelengths=(monitor_wavelength,), orders=0),))
    assert _resolution_notes(concrete, seed) == []
    expected = monitor_wavelength is None or monitor_wavelength >= 2.
    if expected:
        check_wavelength_resolution(concrete, seed.grid)
    else:
        with pytest.raises(ValueError, match='cells per wavelength'):
            check_wavelength_resolution(concrete, seed.grid)
    evaluate = backend_call(backend, lambda x: resolution_guard(
        concrete.replace(background=fd.Material(eps=x)), seed, rt, None))
    assert bool(np.isfinite(evaluate(1.))) == expected


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('eps', [(.5, 1.5, 1.), (1.5, .5, 1.), (.5, 1.5, 4.)])
def test_anisotropic_absolute_component_selection_and_first_tie(backend, eps):
    rt = runtime(backend)
    def make(x):
        return fd.Dispersive(eps_inf=eps, mu=(1., 1., 2.),
                             poles=(fd.Drude(omega_p=x, gamma=0.),))
    concrete = make(2*math.pi)
    n, skin = _index_at(concrete, 1.)
    expected_k = 0. if skin is None else 1/(2*math.pi*skin)
    def index(x):
        real, imag = _resolution_index(make(x), 1., rt)
        return rt.xp.stack((real, imag))
    got = backend_call(backend, index)(2*math.pi)
    np.testing.assert_allclose(got, (n, expected_k), rtol=1e-14, atol=1e-15)
    plan = SimpleNamespace(dx=.2, grid=None, pml_wavelengths=())
    expected = skin/.2 >= 1. if skin is not None else 1/(n*.2) >= 8.
    evaluate = backend_call(backend, lambda x: resolution_guard(scene(make(x)), plan, rt, None))
    assert bool(np.isfinite(evaluate(2*math.pi))) == expected


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_inactive_and_live_lorentz_strength_are_not_interchanged(backend):
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=())
    def make(strength):
        return fd.Dispersive(eps_inf=1., poles=(fd.Lorentz(
            omega_0=2*math.pi, gamma=0., strength=strength),))
    # An inactive concrete pole at resonance is removed even with a live eps.
    inactive = backend_call(backend, lambda x: resolution_guard(scene(fd.Dispersive(
        eps_inf=x, poles=make(0.).poles)), plan, rt, None))
    assert float(inactive(1.)) == 1.
    # A carried zero strength is not inactive: 0/0 at resonance is singular.
    live = backend_call(backend, lambda x: resolution_guard(scene(make(x)), plan, rt, None))
    assert np.isnan(live(0.)) and np.isnan(live(1.))
    # Away from resonance a nonzero pole contributes its optical susceptibility.
    def off_resonance(x):
        return fd.Dispersive(eps_inf=1., poles=(fd.Lorentz(
            omega_0=4*math.pi, gamma=.5, strength=x),))
    evaluate = backend_call(backend, lambda x: resolution_guard(
        scene(off_resonance(x)), plan, rt, None))
    for x in (1., 1000.):
        n, skin = _index_at(off_resonance(x), 1.)
        assert skin is None
        assert bool(np.isfinite(evaluate(x))) == (1/(n*.05) >= 8.)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_report_fallback_priority_and_previous_guard_absorption(backend):
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None,
                           pml_wavelengths=(('z-', 1.), ('z+', .1)))
    from wavebit.fdtd._prepare.checks import _wavelength
    # Eager reads the first PML wavelength, not the shortest; Resolution wins.
    for explicit, expected in ((None, True), (.5, False)):
        s = scene(fd.Material(eps=1.), wavelength=explicit)
        assert _wavelength(s, plan) == (1. if explicit is None else explicit)
        for previous in (None, 0., 2., float('nan')):
            evaluate = backend_call(backend, lambda x: resolution_guard(
                s.replace(structures=(s.structures[0].replace(material=fd.Material(eps=x)),)),
                plan, rt, None if previous is None else rt.array(previous, 'float64')))
            value = float(evaluate(2.))
            if expected and (previous is None or np.isfinite(previous)):
                assert value == (1. if previous is None else previous)
            else:
                assert np.isnan(value)
    # With no wavelength, there is no new certificate to multiply in.
    blank = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=())
    previous = rt.array(2., 'float64')
    evaluate = backend_call(backend, lambda x: resolution_guard(
        scene(fd.Material(eps=x), wavelength=None), blank, rt, previous))
    assert float(evaluate(1000.)) == 2.


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_source_band_overrides_resolution_and_pml_report_wavelength(backend):
    from wavebit.fdtd._prepare.checks import _wavelength
    from wavebit.fdtd._prepare.sources import shortest_wavelength
    rt = runtime(backend)
    plan = SimpleNamespace(dx=.05, grid=None, pml_wavelengths=(('z-', .1),))
    s = scene(fd.Material(eps=1.), wavelength=.5).replace(sources=(fd.Dipole(
        'd', center=(0.,)*3, component='Ey', band=fd.Band(wavelength=1., df=.01)),))
    assert _wavelength(s, plan) == shortest_wavelength(s)
    assert _wavelength(s, plan) > .5
    evaluate = backend_call(backend, lambda x: resolution_guard(s.replace(
        structures=(s.structures[0].replace(material=fd.Material(eps=x)),)), plan, rt, None))
    assert float(evaluate(2.)) == 1.
