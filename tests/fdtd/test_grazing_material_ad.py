"""Grazing admission follows live physical media, independently of smoothed coupling."""
import warnings

import numpy as np
import pytest

from wavebit import Material, fdtd
from wavebit.fdtd import _legacy as old
from wavebit.fdtd._prepare.shape_ranges import _coupled
from wavebit.fdtd._runtime import runtime
from tests.fdtd.test_substrate_continuation import pillar, numerics
from studies.fdtd.traced_material.fdtd_traced_material_probe import substrate_scene, SUBSTRATE


def slab(eps=2.1, *, stabilize=0., wavelength=.95):
    s = substrate_scene(tuple(SUBSTRATE[:4]) + (eps,), steps=20)
    s = s.replace(structures=(s.structures[0].replace(shape=old.Slab(z=(-2., -.2375))),),
        monitors=(old.Diffraction('T', z=-.55, side='transmit', wavelengths=(wavelength,), orders=0),))
    return s.replace(boundaries=s.boundaries.replace(
        z=tuple(f.replace(stabilize=stabilize) for f in s.boundaries.z)))


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('wavelength,points', [(.95, (2.5069, 2.507, 2.55)), (.62, (1.05, 1.2, 1.5))])
def test_unsmoothed_live_medium_matches_eager_grazing_admission(backend, wavelength, points):
    runtime(backend)
    p = quiet(old.plan, slab(wavelength=wavelength))
    assert not _coupled(p)
    assert all(s.shape.body() is None for s in slab().structures)

    def coefficients(eps):
        return quiet(old.lower, slab(eps, wavelength=wavelength), p, backend)['be']

    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        evaluate = jax.jit(jax.value_and_grad(lambda e: jnp.sum(coefficients(e)**2)))
    else:
        import torch
        def evaluate(e):
            e = torch.tensor(e, dtype=torch.float64, requires_grad=True)
            value = (coefficients(e)**2).sum()
            derivative, = torch.autograd.grad(value, e)
            return value.detach().numpy(), derivative.detach().numpy()

    for eps in points:
        try:
            direct = quiet(old.lower, slab(eps, wavelength=wavelength),
                           quiet(old.plan, slab(eps, wavelength=wavelength)))['be']
        except ValueError as error:
            assert 'Grazing order refused' in str(error)
            value, derivative = evaluate(eps)
            assert np.isnan(value) and np.isnan(derivative)
        else:
            value, derivative = evaluate(eps)
            np.testing.assert_allclose(value, np.sum(direct**2), rtol=2e-14)
            h = 1e-6
            def objective(e):
                s = slab(e, wavelength=wavelength)
                b = quiet(old.lower, s, quiet(old.plan, s))['be']
                return np.sum(b**2)
            finite = (objective(eps+h)-objective(eps-h))/(2*h)
            assert abs(finite) > 0
            np.testing.assert_allclose(derivative, finite, rtol=2e-6)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_stabilized_unsmoothed_medium_is_refused_before_lowering(backend):
    runtime(backend)
    s = slab(stabilize=.0001)
    p = quiet(old.plan, s)
    assert not _coupled(p)
    def lower(e):
        return quiet(old.lower, slab(e, stabilize=.0001), p, backend)['be']
    with pytest.raises(ValueError, match='traced material value and fills a stabilized PML layer'):
        if backend == 'jax':
            import jax
            jax.jit(lower)(2.1)
        else:
            import torch
            lower(torch.tensor(2.1, dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_public_continuation_uses_physical_medium_with_fixed_reference(backend):
    runtime(backend)
    n = numerics(steps=20)
    n = n.replace(absorber=n.absorber.replace(reference_media={'z-': Material(2.1)}))
    def problem(eps):
        p = pillar(substrate=eps)
        return p.replace(outputs=p.outputs.replace(monitors=(p.outputs.monitors[0].replace(wavelengths=(.95,)),)))
    prepared = quiet(fdtd.prepare, problem(2.1), numerics=n, backend=backend)
    def objective(e):
        r = quiet(prepared.run, problem(e))
        return (sum((v**2).sum() for v in r.final_fields.values['E']),
                r.validity.checks['runtime_stability'])
    # Eager refusal at the same physical material, despite the fixed reference.
    with pytest.raises(ValueError, match='Grazing order refused'):
        quiet(fdtd.run, problem(2.55), numerics=n)
    if backend == 'jax':
        import jax
        (value, stable), derivative = jax.jit(jax.value_and_grad(objective, has_aux=True))(2.55)
    else:
        import torch
        e = torch.tensor(2.55, dtype=torch.float64, requires_grad=True)
        value, stable = objective(e)
        derivative, = torch.autograd.grad(value, e)
        value, derivative = value.detach().numpy(), derivative.detach().numpy()
    assert np.isnan(value) and np.isnan(derivative)
    assert not bool(stable)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_public_stabilization_refuses_traced_continued_medium(backend):
    runtime(backend)
    n = numerics(steps=2)
    n = n.replace(absorber=n.absorber.replace(stabilize=.0001,
                    reference_media={'z-': Material(2.1)}))
    p = quiet(fdtd.prepare, pillar(), numerics=n, backend=backend)
    def run(e):
        return quiet(p.run, pillar(substrate=e)).final_fields.values['E'][1]
    with pytest.raises(ValueError, match='traced material value and fills a stabilized PML layer'):
        if backend == 'jax':
            import jax
            jax.jit(run)(2.1)
        else:
            import torch
            run(torch.tensor(2.1, dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_unsmoothed_run_fails_validity_at_a_grazing_order(backend):
    runtime(backend)
    p = quiet(old.plan, slab())
    def run(e):
        r = quiet(old.run, slab(e), backend, plan=p)
        return (r.final.E**2).sum(), r.diagnostics['stable']
    if backend == 'jax':
        import jax
        evaluate = jax.jit(jax.value_and_grad(run, has_aux=True))
        (value, stable), derivative = evaluate(2.55)
    else:
        import torch
        e = torch.tensor(2.55, dtype=torch.float64, requires_grad=True)
        value, stable = run(e)
        derivative, = torch.autograd.grad(value, e)
        value, derivative = value.detach().numpy(), derivative.detach().numpy()
    assert not bool(stable)
    assert np.isnan(value) and np.isnan(derivative)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_grazing_cutoffs_match_eager_at_adjacent_float_values(backend):
    """The analytic n-squared interval must retain the eager sqrt/k0 rounding."""
    from wavebit.fdtd._prepare.grazing import refused_bands
    runtime(backend)
    s = slab(wavelength=.62)
    p = quiet(old.plan, s)
    intervals = refused_bands(s, p, 1.55)
    # Keep the analytic boundary as an independent reference, not the endpoints
    # returned by the implementation under test.
    from wavebit.fdtd._prepare.grazing import accepted_cos, exponent
    k0 = 2*np.pi/.62
    low = (2*np.pi/.6)**2/k0**2
    c = accepted_cos(1e-8, 8, exponent(.62), p.steps*p.dt)
    high = low/(1-c*c)
    assert intervals
    def lower(e):
        return quiet(old.lower, slab(e, wavelength=.62), p, backend)['be']
    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        finite = jax.jit(lambda e: jnp.isfinite(lower(e)).all())
    else:
        import torch
        def finite(e):
            return torch.isfinite(lower(torch.tensor(e, dtype=torch.float64, requires_grad=True))).all()
    for cut in (low, high):
        # Several neighbouring floats cover sqrt's repeated values as well as
        # the algebraically equivalent k0 computations' different rounding.
        points = [cut]
        for direction in (-np.inf, np.inf):
            x = cut
            for _ in range(4):
                x = np.nextafter(x, direction)
                points.append(x)
        for eps in points:
            try:
                quiet(old.plan, slab(eps, wavelength=.62))
                accepted = True
            except ValueError as exc:
                assert 'Grazing order refused' in str(exc)
                accepted = False
            assert bool(finite(eps)) == accepted, (backend, eps, accepted)
