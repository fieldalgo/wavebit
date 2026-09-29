"""Independent field sensitivities at a symmetric uniform patterned-layer limit."""
import numpy as np
import pytest
import wavebit

jax = pytest.importorskip('jax')
import jax.numpy as jnp

CFG = dict(period=(1400., 1500.), wavelength=900., thickness=(170.,),
           truncation_order=(1, 1), theta=0., phi=0., answers='field',
           factorization='li', fourier_coefficients='integrated')


def value(backend, raster, observe, coherent=False):
    s = wavebit.rcwa(backend=backend, **CFG, structure=raster,
                     excitation={(-1, 0): 1., (1, 0): .7j} if coherent else None)
    result = s.solve()
    if observe == 'power':
        return result.T.sum().real
    field = result.reconstruct_field([57.], [-89.], [83.])
    xp = jnp if backend == 1 else np
    return xp.real(field[0, 0, 0, 1]) if observe == 'amplitude' else xp.sum(xp.abs(field[0, 0, 0, :])**2)


@pytest.mark.parametrize('observe', ['power', 'amplitude', 'intensity'])
@pytest.mark.parametrize('coherent', [False, True])
def test_uniform_field_pixel_derivative(observe, coherent):
    raster = np.full((1, 3, 3), 4.)
    got = np.asarray(jax.grad(lambda eps: value(1, eps, observe, coherent))(jnp.asarray(raster)))
    expected = np.empty_like(raster)
    for index in np.ndindex(raster.shape):
        a, b = raster.copy(), raster.copy()
        a[index] += 1e-4
        b[index] -= 1e-4
        expected[index] = (float(value(0, a, observe, coherent))-float(value(0, b, observe, coherent)))/(2e-4)
    print(observe, coherent, got.tolist(), expected.tolist(), flush=True)
    np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-7)
