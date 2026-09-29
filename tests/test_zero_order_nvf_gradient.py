"""The DC normal-field coefficient is its mean, including under compiled gradients."""
import numpy as np
import pytest

import wavebit
from wavebit._core.parametric import _sampled_coeffs


@pytest.mark.parametrize('backend', [0, 1, 2])
@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('complex_field', [False, True])
def test_dc_values_and_precision_agree_with_the_full_transform(backend, bits, complex_field):
    if backend == 0:
        import array_api_compat.numpy as xp
    elif backend == 1:
        pytest.importorskip('jax')
        import jax.numpy as xp
        wavebit.enable_x64()
    else:
        pytest.importorskip('torch')
        import array_api_compat.torch as xp
    values = np.sin(np.arange(35).reshape(5, 7)/7.)
    if complex_field:
        values = values+1j*np.cos(np.arange(35).reshape(5, 7)/11.)
    dtype = getattr(xp, 'complex'+str(2*bits) if complex_field else 'float'+str(bits))
    field = xp.asarray(values, dtype=dtype)
    got = _sampled_coeffs(xp, field, 0, 0)
    expected = xp.fft.fft2(field)[0, 0]/values.size
    assert got.shape == (1, 1)
    assert got.dtype == getattr(xp, 'complex'+str(2*bits))
    assert complex(got[0, 0]) == pytest.approx(complex(expected), abs=2e-7 if bits == 32 else 1e-14)


@pytest.mark.parametrize('complex_field', [False, True])
def test_compiled_dc_derivative_matches_the_analytic_mean(complex_field):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    a = np.arange(35, dtype=float).reshape(5, 7)/35+.1
    b = a[::-1, ::-1]+.2
    def loss(w):
        field = jnp.sin(w*jnp.asarray(a))
        if complex_field:
            field = field+1j*jnp.cos(w*jnp.asarray(b))
        value = _sampled_coeffs(jnp, field, 0, 0)[0, 0]
        return jnp.abs(value)**2
    got = float(jax.jit(jax.grad(loss))(.7))
    value = np.mean(np.sin(.7*a))
    derivative = np.mean(a*np.cos(.7*a))
    if complex_field:
        value += 1j*np.mean(np.cos(.7*b))
        derivative += -1j*np.mean(b*np.sin(.7*b))
    assert got == pytest.approx(2*np.real(np.conj(value)*derivative), rel=1e-11, abs=1e-12)


@pytest.mark.parametrize('kind', ['rectangle', 'ellipse'])
@pytest.mark.parametrize('transform', ['grad', 'jit_grad', 'grad_jit'])
def test_zero_order_shape_gradient_matches_fresh_numpy(kind, transform):
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    def structure(width):
        return [dict(background=1., shapes=[dict(type=kind, cx=0., cy=0., lx=width,
                                               ly=100., angle=0., material=4.)])]
    cfg = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
               truncation_order=(0, 0), theta=.13, phi=.21,
               shape_nvf_grid_size=32, factorization='nvf')
    s = wavebit.rcwa(backend=1, **cfg, structure=structure(100.))
    geometry = s.freeze_geometry()
    def loss(width):
        return s.solve(structure=structure(width), geometry=geometry).T.sum().real
    derivative = jax.grad(jax.jit(loss) if transform == 'grad_jit' else loss)
    if transform == 'jit_grad':
        derivative = jax.jit(derivative)
    got = float(derivative(100.))
    def reference(width):
        return wavebit.rcwa(backend=0, **cfg, structure=structure(width)).solve().T.sum().real
    h = 1e-3
    expected = (reference(100.+h)-reference(100.-h))/(2*h)
    assert abs(expected) > 1e-8
    assert got == pytest.approx(expected, abs=1e-7, rel=2e-5)
