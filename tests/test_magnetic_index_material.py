"""Index and permeability determine both phase and interface impedance."""
import json

import numpy as np
import pytest

import wavebit
from wavebit import Incidence, JobSpec, Layer, Stack, TabulatedMaterial, eps_from_index


def scalar(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value).item()


def test_scalar_conversion_includes_complex_mu_and_preserves_default():
    n, mu = 2.+.2j, 1.3+.1j
    assert eps_from_index(n, mu=mu) * mu == pytest.approx(n*n)
    assert eps_from_index(2) == 4 and isinstance(eps_from_index(2), int)
    assert eps_from_index(n) == n*n
    assert eps_from_index(n, mu=None) == n*n


@pytest.mark.parametrize('mu', [0., 0j, np.nan, np.inf, np.array([1., 0.])])
def test_invalid_permeability_is_not_silently_divided(mu):
    with pytest.raises(ValueError, match='finite, nonzero'):
        eps_from_index(2., mu=mu)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_conversion_keeps_broadcasting_precision_and_both_gradients(backend, precision):
    n_values, mu_values = [[2.], [3.]], [1., 2.]
    if backend == 'numpy':
        n, mu = np.array(n_values, dtype=precision), np.array(mu_values, dtype=precision)
        result = eps_from_index(n, mu=mu)
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        wavebit.enable_x64()
        n, mu = jax.numpy.array(n_values, dtype=precision), jax.numpy.array(mu_values, dtype=precision)
        fn = jax.jit(lambda a, b: eps_from_index(a, mu=b).sum())
        dn, dm = jax.grad(fn, argnums=(0, 1))(n, mu)
        np.testing.assert_allclose(dn, [[6.], [9.]])
        np.testing.assert_allclose(dm, [-13., -3.25])
        result = jax.jit(lambda a, b: eps_from_index(a, mu=b))(n, mu)
    else:
        torch = pytest.importorskip('torch')
        dtype = getattr(torch, precision)
        n = torch.tensor(n_values, dtype=dtype, requires_grad=True)
        mu = torch.tensor(mu_values, dtype=dtype, requires_grad=True)
        result = eps_from_index(n, mu=mu)
        result.sum().backward()
        np.testing.assert_allclose(n.grad, [[6.], [9.]])
        np.testing.assert_allclose(mu.grad, [-13., -3.25])
        result = result.detach().numpy()
    assert str(result.dtype) == precision
    np.testing.assert_allclose(result, [[4., 2.], [9., 4.5]])


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_index_and_mu_are_interpolated_before_conversion(backend):
    model = TabulatedMaterial.from_index([800., 1000.], [2.+.1j, 3.+.3j], mu=[1.+.1j, 2.+.2j])
    wavelengths = [[800., 850.], [900., 1000.]]
    if backend == 'numpy':
        w = np.array(wavelengths)
        actual = model.at(w)
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        wavebit.enable_x64()
        w = jax.numpy.array(wavelengths)
        actual = jax.jit(model.at)(w)
        derivative = jax.jit(jax.grad(lambda x: model.at(x).epsilon.real.sum()))(w)
    else:
        torch = pytest.importorskip('torch')
        w = torch.tensor(wavelengths, dtype=torch.float64, requires_grad=True)
        actual = model.at(w)
        actual.epsilon.real.sum().backward()
        derivative = w.grad.numpy()
        actual = wavebit.Material(actual.epsilon.detach().numpy(), actual.mu.detach().numpy())
    f = (np.array(wavelengths)-800.)/200.
    n, mu = 2.+.1j + f*(1.+.2j), 1.+.1j + f*(1.+.1j)
    np.testing.assert_allclose(actual.epsilon, n*n/mu, rtol=1e-13)
    np.testing.assert_allclose(actual.mu, mu)
    if backend != 'numpy':
        expected = (2*n*(1.+.2j)/200./mu - n*n*(1.+.1j)/200./mu**2).real
        np.testing.assert_allclose(derivative, expected, rtol=1e-12)
    # Converting sample points first and interpolating epsilon would be wrong.
    endpoint_average = ((2.+.1j)**2/(1.+.1j) + (3.+.3j)**2/(2.+.2j))/2
    assert abs(scalar(model.at(900.).epsilon) - endpoint_average) > 1e-3


def test_constant_mu_copy_and_nonmagnetic_default():
    mu = np.array([1.2, 1.4])
    model = TabulatedMaterial.from_index([800., 1000.], [2., 3.], mu=mu)
    mu[:] = 9.
    assert model.at(900.).mu == pytest.approx(1.3)
    constant = TabulatedMaterial.from_index([800., 1000.], [2., 3.], mu=1.25)
    assert constant.mu == (1.25, 1.25)
    assert constant.at(900.).epsilon == pytest.approx(5.)
    assert constant.rotated(.2) is constant
    assert not TabulatedMaterial.from_index([800., 1000.], [2., 3.]).is_magnetic
    assert TabulatedMaterial.from_index([800., 1000.], [2., 3.], mu=1.).is_magnetic


@pytest.mark.parametrize('mu', [0., [1., 0.], [1., np.nan], [1., np.inf],
                               [1.], np.ones((2, 3)), np.ones((2, 3, 3))])
def test_index_import_refuses_invalid_or_tensor_mu(mu):
    with pytest.raises(ValueError, match='mu'):
        TabulatedMaterial.from_index([800., 1000.], [2., 3.], mu=mu)


def test_zero_interpolated_mu_has_eager_and_jit_failure_semantics():
    model = TabulatedMaterial.from_index([800., 1000.], [2., 2.], mu=[-1., 1.])
    with pytest.raises(ValueError, match='nonzero'):
        model.at(900.)
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    w = jax.numpy.array([850., 900., 950.])
    actual = jax.jit(model.at)(w)
    assert np.isnan(actual.epsilon[1])
    np.testing.assert_allclose(np.asarray(actual.epsilon)[[0, 2]], [-8., 8.])
    converted = jax.jit(lambda mu: eps_from_index(2., mu=mu))(jax.numpy.array([1., 0., np.inf]))
    assert converted[0] == 4. and np.all(np.isnan(converted[1:]))


@pytest.mark.parametrize('mu', [1.25, [1.1+.1j, 1.3+.2j]])
def test_magnetic_index_json_preserves_source_tables_and_interpolation(mu):
    model = TabulatedMaterial.from_index([800., 1000.], [2.+.1j, 3.+.2j], mu=mu)
    spec = JobSpec(stack=Stack([Layer(50., dict(background=model, shapes=[]))],
                              period=(600., 600.), top=model), incidence=Incidence(wavelength=900.))
    serialized = spec.to_json()
    table = json.loads(serialized)['stack']['top']['dispersion']
    assert 'index' in table and 'mu' in table and 'epsilon' not in table
    restored = JobSpec.from_json(serialized)
    assert restored.key() == spec.key()
    assert restored.stack.top == model
    assert restored.stack.layers[0].structure['background'] == model
    assert restored.stack.check() == ()
    assert restored.stack.top.at(875.).epsilon == model.at(875.).epsilon


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('slot', ['top', 'bot'])
def test_constant_index_changing_mu_matches_fresnel_impedance(backend, slot):
    pytest.importorskip(backend)
    model = TabulatedMaterial.from_index([800., 1000.], [2., 2.], mu=[1., 3.])
    solver = wavebit.rcwa(backend=backend, wavelength=800., period=600.,
                          structure=np.ones((1, 1, 1)), thickness=[0.],
                          truncation_order=0, tetm='te', **{slot: model})
    for w, mu in [(800., 1.), (900., 2.), (1000., 3.)]:
        result = solver.solve(wavelength=w, auto_detune=False)
        z = mu / 2.  # relative wave impedance mu/n; n stays exactly two
        reflection = ((z-1)/(z+1))**2
        assert scalar(result.R.sum()) == pytest.approx(reflection, abs=1e-12)
        assert scalar(result.T.sum()) == pytest.approx(1-reflection, abs=1e-12)
        assert scalar(getattr(solver, slot).epsilon) == pytest.approx(4./mu)
        assert scalar(getattr(solver, slot).mu) == pytest.approx(mu)
