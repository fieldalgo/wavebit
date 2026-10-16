"""Carried frequency must retain the selected susceptibility's derivative."""
import numpy as np
import pytest
from wavebit.materials.time_domain import Drude, Lorentz
from wavebit.fdtd._runtime import runtime


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('family', ['drude', 'lorentz'])
def test_carried_frequency_matches_concrete_response(backend, family):
    runtime(backend)
    pole = (Drude(omega_p=2., gamma=.5) if family == 'drude' else
            Lorentz(omega_0=10., gamma=.5, strength=4.))
    f = lambda w: pole.eps_term(w).imag
    h = 1e-5
    expected = (f(10.+h)-f(10.-h))/(2*h)
    if backend == 'torch':
        import torch
        w = torch.tensor(10., dtype=torch.float64, requires_grad=True)
        value = f(w)
        grad = float(torch.autograd.grad(value, w)[0])
        value = float(value.detach())
    else:
        import jax
        value, grad = map(float, jax.jit(jax.value_and_grad(f))(10.))
    assert value == pytest.approx(f(10.), rel=1e-12)
    assert grad == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_large_damping_resonance_keeps_frequency_tangent(backend):
    runtime(backend)
    pole = Lorentz(omega_0=10., gamma=1e308, strength=1e100)
    # chi.imag * 1e209 = 10/w here, to float64 precision. Its derivative is -0.1.
    f = lambda w: pole.eps_term(w).imag * 1e209
    if backend == 'torch':
        import torch
        w = torch.tensor(10., dtype=torch.float64, requires_grad=True)
        value = f(w)
        grad = float(torch.autograd.grad(value, w)[0])
        value = float(value.detach())
    else:
        import jax
        value, grad = map(float, jax.jit(jax.value_and_grad(f))(10.))
    assert value == pytest.approx(1., rel=1e-12)
    assert np.isfinite(grad)
    assert grad == pytest.approx(-.1, rel=1e-12)
