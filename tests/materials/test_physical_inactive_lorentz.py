"""Physical Lorentz response: inactive identities and live zero-strength tangents."""
import numpy as np
import pytest

from wavebit import DispersiveMaterial, Lorentz, Units
from wavebit.units import C0


@pytest.mark.parametrize('omega0,gamma', [(2., 0.), (1e200, 0.), (2., 1e308)])
@pytest.mark.parametrize('carrier', ['python', 'numpy', 'jax', 'torch'])
def test_concrete_inactive_term_is_zero_without_singular_or_overflow_arithmetic(omega0, gamma, carrier):
    strength = 0.
    if carrier == 'numpy':
        strength = np.asarray(-0.)
    elif carrier == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        strength = pytest.importorskip('jax.numpy').asarray(0.)
    elif carrier == 'torch':
        torch = pytest.importorskip('torch')
        strength = torch.tensor(0., dtype=torch.float64)
    pole = Lorentz(omega_0=omega0, gamma=gamma, strength=strength)
    with np.errstate(all='raise'):
        value = pole.epsilon_term(2.)
    assert complex(value) == 0j
    if carrier == 'jax':
        assert complex(jax.jit(pole.epsilon_term)(2.)) == 0j


@pytest.mark.parametrize('length,scale', [('m', 1.), ('um', 1e6), ('nm', 1e9)])
def test_inactive_physical_material_preserves_tensor_baseline_and_active_pole(length, scale):
    omega = 2e15
    material = DispersiveMaterial(epsilon_inf=(2., 3., 5.), poles=(
        Lorentz(omega_0=omega, gamma=0., strength=0.),
        Lorentz(omega_0=3e15, gamma=4e13, strength=5e29)))
    response = material.at(2*np.pi*C0/omega*scale, units=Units(length=length))
    expected = np.asarray((2., 3., 5.)) + 5e29/complex(9e30-4e30, -8e28)
    np.testing.assert_allclose(response.epsilon, expected, rtol=3e-15)


@pytest.mark.parametrize('parameter', ['strength', 'omega0', 'gamma', 'frequency'])
@pytest.mark.parametrize('mode', ['jax-grad', 'jax-jvp', 'jax-jit-jvp',
                                  'torch-grad', 'torch-forward', 'torch-func-jvp'])
def test_zero_primal_keeps_the_independent_rational_derivative(parameter, mode):
    initial = {'strength': 0., 'omega0': 2., 'gamma': 0., 'frequency': 2.}[parameter]
    expected = 1/complex(9.-4., -.7*2.) if parameter == 'strength' else 0j
    def response(v):
        return Lorentz(
            omega_0=v if parameter == 'omega0' else 3. if parameter == 'strength' else 2.,
            gamma=v if parameter == 'gamma' else .7 if parameter == 'strength' else 0.,
            strength=v if parameter == 'strength' else 0.
        ).epsilon_term(v if parameter == 'frequency' else 2.)
    if mode.startswith('jax'):
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        jnp = pytest.importorskip('jax.numpy')
        value = jnp.asarray(initial)
        if mode == 'jax-grad':
            derivative = (jax.grad(lambda x: response(x).real)(value)
                          + 1j*jax.grad(lambda x: response(x).imag)(value))
        else:
            call = lambda x: jax.jvp(response, (x,), (jnp.ones_like(x),))[1]
            derivative = (jax.jit(call) if mode == 'jax-jit-jvp' else call)(value)
    else:
        torch = pytest.importorskip('torch')
        value = torch.tensor(initial, dtype=torch.float64, requires_grad=mode == 'torch-grad')
        if mode == 'torch-grad':
            y = response(value)
            derivative = (torch.autograd.grad(y.real, value, retain_graph=True)[0]
                          + 1j*torch.autograd.grad(y.imag, value)[0])
        elif mode == 'torch-forward':
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
                _, derivative = torch.autograd.forward_ad.unpack_dual(response(dual))
                assert derivative is not None
        else:
            _, derivative = torch.func.jvp(response, (value,), (torch.ones_like(value),))
    assert complex(derivative) == pytest.approx(expected, rel=3e-13, abs=3e-15)


def test_positive_extended_precision_strength_is_not_inactive():
    strength = (np.ldexp(np.longdouble(1), -1075) if np.finfo(np.longdouble).minexp < -1022
                else np.nextafter(np.float64(0), np.float64(1)))
    value = Lorentz(omega_0=.75, gamma=0., strength=strength).epsilon_term(.25)
    assert value == 2*strength and value != 0


def test_inactive_strength_does_not_hide_invalid_traced_rates():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    value = jax.jit(lambda gamma: Lorentz(omega_0=2., gamma=gamma, strength=0.).epsilon_term(2.))(-1.)
    assert np.isnan(value)
