"""Independent variations of both constitutive fields through each 1D formulation."""
import numpy as np
import pytest
import wavebit
from wavebit._core.backend import array_namespace

DIR = np.array([[[1., -.3, .2]]])
CFG = dict(period=(1400., 1500.), wavelength=900., thickness=(170.,),
           truncation_order=(1, 0), theta=0., phi=0., answers='field',
           factorization='li', fourier_coefficients='integrated', excitation={-1: 1., 1: .7j})


def value(backend, t, path, quantity, algorithm):
    xp = array_namespace(t)
    eps = xp.asarray(np.full((1, 1, 3), 4.+.08j))
    mu = xp.asarray(np.full((1, 1, 3), 1.3+.02j))
    if quantity == 'structure':
        eps = eps+t*xp.asarray(DIR)
    else:
        mu = mu+t*xp.asarray(DIR)
    pol = dict(psi=.37) if path == 'conical' else dict(tetm=path)
    s = wavebit.rcwa(backend=backend, **CFG, **pol, structure=eps, mu=mu,
                     layer_algorithm=algorithm)
    result = s.solve()
    f = result.reconstruct_field([57.], [0.], [83.])[0, 0, 0]
    return result.T.sum().real + .7*xp.real(f[1]) + .13*xp.imag(f[3]) + .1*xp.sum(xp.abs(f)**2)


@pytest.mark.parametrize('mode', ['grad', 'torch'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical'])
@pytest.mark.parametrize('quantity', ['structure', 'mu'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_one_dimensional_material_and_field_derivatives(mode, path, quantity, algorithm):
    if mode == 'torch':
        torch = pytest.importorskip('torch')
        t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        value(2, t, path, quantity, algorithm).backward()
        got = float(t.grad)
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        got = float(jax.grad(lambda t: value(1, t, path, quantity, algorithm))(jnp.asarray(0.)))
    h = 1e-4
    expected = (value(0, np.asarray(h), path, quantity, algorithm)
                - value(0, np.asarray(-h), path, quantity, algorithm))/(2*h)
    assert abs(expected) > 1e-6
    assert got == pytest.approx(float(expected), abs=2e-7, rel=2e-5)
