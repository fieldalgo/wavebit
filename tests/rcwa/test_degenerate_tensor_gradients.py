"""Tensor coupling sensitivities at and approaching an isotropic material."""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

CFG = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
           truncation_order=(0, 0), theta=0., phi=0., answers='field',
           factorization='li', fourier_coefficients='integrated')
MIX = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])


def value(backend, coupling, split, formulation):
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        xp = torch
    else:
        xp = jnp if backend == 'jax' else np
    eps = xp.asarray(np.diag([4.+split, 4., 4.])) + coupling*xp.asarray(MIX)
    s = _legacy_wavebit_rcwa(backend=backend, **CFG, structure=eps[None, None, None, ...])
    s.formulation = formulation
    result = s.solve()
    field = result.reconstruct_field([0.], [0.], [85.])
    ex = field[0, 0, 0, 0]
    return xp.real(ex) + .37*xp.imag(ex)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('split', [0., 1e-3, .1])
@pytest.mark.parametrize('formulation', ['auto', 'first_order'])
def test_tensor_coupling_gradient(backend, split, formulation):
    if backend == 'jax':
        got = float(jax.grad(lambda coupling: value('jax', coupling, split, formulation))(jnp.asarray(0.)))
    else:
        torch = pytest.importorskip('torch')
        coupling = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        value('torch', coupling, split, formulation).backward()
        got = float(coupling.grad)
    h = 1e-4
    expected = (float(value('numpy', h, split, formulation))-float(value('numpy', -h, split, formulation)))/(2*h)
    print(backend, split, formulation, got, expected, flush=True)
    assert abs(expected) > 1e-4
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-8)


def airy_field(epsilon):
    n = np.sqrt(epsilon+0j)
    k0 = 2*np.pi/900.
    r = (n-1)/(n+1)
    phase = np.exp(2j*k0*n*170.)
    a = (2/(1+n))/(1-r*r*phase)
    b = r*a*phase
    return a*np.exp(1j*k0*n*85.) + b*np.exp(-1j*k0*n*85.)


@pytest.mark.parametrize('coupling', [-.2, -.01, .01, .2])
def test_numpy_tensor_cross_field_matches_rotated_independent_airy(coupling):
    ex = (airy_field(4.+coupling)-airy_field(4.-coupling))/2
    expected = float(ex.real+.37*ex.imag)
    assert value('numpy', coupling, 0., 'auto') == pytest.approx(expected, abs=1e-12)
