"""Representable Drude loss tangents must survive large finite damping."""
from decimal import Decimal, localcontext
import math
import sys

import numpy as np
import pytest

from wavebit.materials import DispersiveMaterial, Drude
from wavebit.materials._lower import lower_material
from wavebit.materials._time_domain import Drude as NormalizedDrude
from wavebit.units import C0, Units


def reference(omega, plasma, gamma):
    # Independent real-valued formula, without a complex quotient or its AD rule.
    with localcontext() as context:
        context.prec = 100
        w, p, g = map(Decimal.from_float, (omega, plasma, gamma))
        d = w*w + g*g
        return float(p*p*g/(w*d)), float(-p*p*g*(3*w*w+g*g)/(w*w*d*d))


def response(route, plasma, gamma):
    if route == 'normalized':
        return lambda w: NormalizedDrude(omega_p=plasma, gamma=gamma).eps_term(w).imag
    pole = Drude(omega_p=plasma, gamma=gamma)
    if route == 'physical':
        return lambda w: pole.epsilon_term(w).imag
    material = DispersiveMaterial(1., (pole,))
    units = Units(length='m')
    if route == 'lowered':
        normalized = lower_material(material, units=units)
        return lambda w: normalized.poles[0].eps_term(w*units.time_unit_seconds).imag
    assert route == 'material-at'
    return lambda w: material.at(2*math.pi*C0/w, units=units).epsilon.imag


def differentiated(fn, mode):
    if mode.startswith('jax'):
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        if mode == 'jax-reverse':
            evaluate = jax.jit(jax.value_and_grad(fn))
        else:
            evaluate = jax.jit(lambda x: jax.jvp(fn, (x,), (jnp.ones_like(x),)))

        def run(value):
            with jax.enable_x64():
                return tuple(map(float, evaluate(value)))
        return run
    torch = pytest.importorskip('torch')

    def run(value):
        x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        if mode == 'torch-reverse':
            y = fn(x)
            return float(y.detach()), float(torch.autograd.grad(y, x)[0])
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
            y, tangent = torch.autograd.forward_ad.unpack_dual(fn(dual))
            return float(y.detach()), float(tangent.detach())
    return run


MODES = ('torch-reverse', 'torch-forward', 'jax-reverse', 'jax-forward')


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('route', ('normalized', 'physical', 'lowered', 'material-at'))
def test_large_damping_keeps_unscaled_frequency_tangent(route, mode):
    fn = response(route, 1e50, 1e308)
    evaluate = differentiated(fn, mode)
    old_boundary = sys.float_info.max/1e308
    # Include the original gamma*omega overflow switch and its float64 neighbours.
    for w in (.75, 1., 1.25, float(np.nextafter(old_boundary, 0.)), old_boundary,
              float(np.nextafter(old_boundary, math.inf)), 2.):
        expected = reference(w, 1e50, 1e308)
        assert expected[1] != 0.
        np.testing.assert_allclose(float(fn(np.float64(w))), expected[0], rtol=1e-12, atol=0.)
        # No objective rescaling or absolute tolerance may hide a lost 1e-208 tangent.
        np.testing.assert_allclose(evaluate(w), expected, rtol=1e-12, atol=0.)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('route', ('normalized', 'physical'))
def test_drude_frequency_tangent_at_scaling_boundary_and_ordinary_values(route, mode):
    for w in (.75, 1.25):
        boundary = .5*math.sqrt(1./sys.float_info.min)/max(1., w)
        for g in (0., .5, 2., float(np.nextafter(boundary, 0.)), boundary,
                  float(np.nextafter(boundary, math.inf))):
            plasma = 2. if g < 3. else 1e50
            fn = response(route, plasma, g)
            expected = reference(w, plasma, g)
            np.testing.assert_allclose(float(fn(np.float64(w))), expected[0], rtol=1e-12, atol=0.)
            np.testing.assert_allclose(differentiated(fn, mode)(w), expected, rtol=1e-12, atol=0.)


@pytest.mark.parametrize('backend', ('torch', 'jax'))
@pytest.mark.parametrize('vector', (False, True))
@pytest.mark.parametrize('carrier', ('omega', 'gamma'))
def test_public_single_precision_response_retains_dtype_and_tangent(backend, vector, carrier):
    def fn(x):
        w, g = (x, .5) if carrier == 'omega' else (x*0+1., x)
        # Pole rates are scalar declarations; only wavelength/frequency is batched.
        if carrier == 'gamma' and vector:
            w = w.repeat(2)
        return Drude(omega_p=2., gamma=g).epsilon_term(w)

    size = 2 if vector else 1
    initial = 1. if carrier == 'omega' else .5
    shape = (2,) if vector and carrier == 'omega' else ()
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        x = torch.full(shape, initial, dtype=torch.float32, requires_grad=True)
        value = fn(x)
        assert value.dtype == torch.complex64
        tangent = torch.autograd.grad(value.imag.sum(), x)[0].detach().numpy()
        value = value.detach().numpy()
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        with jax.enable_x64():
            x = jnp.full(shape, initial, dtype=jnp.float32)
            value = jax.jit(fn)(x)
            assert value.dtype == jnp.complex64
            tangent = np.asarray(jax.jit(jax.grad(lambda t: fn(t).imag.sum()))(x))
            value = np.asarray(value)
    expected_tangent = -4.16 if carrier == 'omega' else 1.92*size
    np.testing.assert_allclose(value, -3.2+1.6j, rtol=1e-6, atol=0.)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-6, atol=0.)
