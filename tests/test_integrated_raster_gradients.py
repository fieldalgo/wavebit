"""Equal raster values are independent design variables, including under a frozen plan."""
from functools import lru_cache

import numpy as np
import pytest

import wavebit


CFG = dict(period=(700., 650.), thickness=(120.,), wavelength=900.,
           truncation_order=(1, 1), theta=.19, phi=.3, psi=.7, top=1., bot=2.25,
           fourier_coefficients='integrated')
PIXELS = np.array([[[4., 4., 2., 2.], [4., 4., 2., 2.]]])


def _solver(backend, quantity, rule, pixels):
    conditions = {quantity: pixels}
    if quantity == 'mu':
        conditions['structure'] = np.full_like(PIXELS, 2.25)
    return wavebit.rcwa(backend=backend, **CFG, factorization=rule, **conditions)


@lru_cache(maxsize=None)
def _reference(quantity, rule):
    def fresh(pixels):
        return float(_solver('numpy', quantity, rule, pixels).solve().T.sum().real)
    gradient = np.empty_like(PIXELS)
    for index in np.ndindex(PIXELS.shape):
        plus, minus = PIXELS.copy(), PIXELS.copy()
        plus[index] += 1e-4
        minus[index] -= 1e-4
        gradient[index] = (fresh(plus)-fresh(minus))/2e-4
    return fresh(PIXELS), gradient


@pytest.mark.parametrize('mode', ['grad', 'jit_grad', 'grad_jit', 'torch'])
@pytest.mark.parametrize('quantity', ['structure', 'mu'])
@pytest.mark.parametrize('rule', ['li', 'laurent'])
def test_every_pixel_gradient_matches_fresh_numpy_under_compression(mode, quantity, rule):
    expected, gradient = _reference(quantity, rule)
    if mode == 'torch':
        torch = pytest.importorskip('torch')
        pixels = torch.tensor(PIXELS, dtype=torch.float64, requires_grad=True)
        solver = _solver('torch', quantity, rule, pixels)
        geometry = solver.freeze_geometry()
        value = solver.solve(**{quantity: pixels}, geometry=geometry).T.sum().real
        value.backward()
        got, actual = float(value.detach()), pixels.grad.numpy()
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        wavebit.enable_x64()
        solver = _solver('jax', quantity, rule, PIXELS)
        geometry = solver.freeze_geometry()
        def objective(pixels):
            return solver.solve(**{quantity: pixels}, geometry=geometry).T.sum().real
        pixels = jnp.asarray(PIXELS)
        if mode == 'grad_jit':
            objective = jax.jit(objective)
            # Populate the forward cache before differentiation. Its trace must already
            # preserve every pixel, even though no JVP/linearization tracer arrived then.
            assert float(objective(pixels)) == pytest.approx(expected, abs=2e-10)
        calculate = jax.value_and_grad(objective)
        if mode == 'jit_grad':
            calculate = jax.jit(calculate)
        value, result = calculate(pixels)
        got, actual = float(value), np.asarray(result)
    assert got == pytest.approx(expected, abs=2e-10)
    # The first pixel is omitted by the compressed gather and has a nonzero derivative.
    assert abs(gradient[0, 0, 0]) > 1e-5
    np.testing.assert_allclose(actual, gradient, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('quantity', ['structure', 'mu'])
def test_fit_updates_every_uniform_pixel_using_independent_derivatives(quantity):
    jax = pytest.importorskip('jax')
    optax = pytest.importorskip('optax')
    wavebit.enable_x64()
    cfg = dict(period=(700., 700.), thickness=(120.,), wavelength=900.,
               truncation_order=(0, 0), theta=.13, phi=.21, psi=.37,
               fourier_coefficients='integrated')
    pixels = np.full((1, 3, 3), 4.)
    if quantity == 'mu':
        cfg['structure'] = np.full_like(pixels, 2.25)
    solver = wavebit.rcwa(backend='jax', **cfg, **{quantity: pixels})
    observed = []
    optimizer = optax.sgd(.1)

    def update(grads, state, params=None):
        def remember(p, g):
            observed.append((np.asarray(p).copy(), np.asarray(g).copy()))
        jax.debug.callback(remember, params[quantity], grads[quantity], ordered=True)
        return optimizer.update(grads, state, params)

    final = solver.fit([quantity], solver.solve, lambda r: r.T.sum().real,
                       optax.GradientTransformation(optimizer.init, update), iteration=2)
    jax.effects_barrier()
    assert len(observed) == 2
    expected = pixels.copy()
    def fresh(p):
        return float(wavebit.rcwa(backend='numpy', **cfg, **{quantity: p}).solve().T.sum().real)
    for point, actual in observed:
        np.testing.assert_allclose(point, expected, atol=1e-9, rtol=0)
        want = np.empty_like(point)
        for index in np.ndindex(point.shape):
            plus, minus = point.copy(), point.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            want[index] = (fresh(plus)-fresh(minus))/2e-4
        np.testing.assert_allclose(actual, want, atol=1e-8, rtol=2e-5)
        expected -= .1 * want
    np.testing.assert_allclose(final[quantity], expected, atol=2e-9, rtol=0)
    assert np.all(np.abs(np.asarray(final[quantity])-pixels) > 1e-6)
