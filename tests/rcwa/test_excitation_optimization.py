"""Excitation weights remain differentiable values and custom edits stay temporary."""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

from wavebit.rcwa._engine.engine import _holds_tracer

CFG = dict(period=(1800., 900.), wavelength=900.,
           structure=np.array([[[1., 4., 4., 1.]]]), thickness=(170.,),
           truncation_order=(1, 0), theta=.13, tetm='tm', answers='field')


def solver(backend='jax', **kwargs):
    cfg = dict(CFG, excitation={0: 1., 1: .4j}) | kwargs
    return _legacy_wavebit_rcwa(backend=backend, **cfg)


def value(excitation, thickness=170.):
    return float(solver('numpy', excitation=excitation, thickness=(thickness,)).solve().T.sum().real)


def differences(excitation):
    result = {}
    h = 1e-5
    for key in excitation:
        parts = []
        for direction in (1., 1j):
            plus, minus = excitation.copy(), excitation.copy()
            plus[key] += h*direction
            minus[key] -= h*direction
            parts.append((value(plus)-value(minus))/(2*h))
        result[key] = parts[0]+1j*parts[1]
    return result


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('precision', [0, 1])
@pytest.mark.parametrize('amplitude', [1, 1., .4j])
def test_weights_have_the_solvers_complex_precision(backend, precision, amplitude):
    source = {0: 1, 1: amplitude}
    s = solver(backend, excitation=source, complex_dtype=precision)
    assert set(s.excitation) == set(source)
    for order, stored in s.excitation.items():
        assert stored.dtype == s.complex_dtype
        assert complex(stored) == pytest.approx(source[order], rel=1e-7)
    assert source == {0: 1, 1: amplitude}


def test_jax_integer_weights_have_the_correct_complex_gradient():
    pytest.importorskip('jax')
    s = solver(excitation={0: 1, 1: 1})
    got = s.grad(['excitation'], s.solve, lambda result: result.T.sum().real)['excitation']
    expected = differences({0: 1, 1: 1})
    assert np.linalg.norm(list(expected.values())) > 1e-6
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=2e-6, atol=2e-9)
    assert not any(_holds_tracer(v) for v in vars(s).values())


def test_fit_of_integer_weights_matches_independent_complex_descent():
    pytest.importorskip('jax')
    optax = pytest.importorskip('optax')
    s = solver(excitation={0: 1, 1: 1})
    final = s.fit(['excitation'], s.solve, lambda result: result.T.sum().real,
                  optax.sgd(.02), iteration=3)['excitation']
    expected = {0: 1.+0j, 1: 1.+0j}
    for _ in range(3):
        gradient = differences(expected)
        expected = {key: expected[key]-.02*gradient[key] for key in expected}
    for key in expected:
        np.testing.assert_allclose(final[key], expected[key], atol=2e-9, rtol=2e-9)
        np.testing.assert_array_equal(s.excitation[key], final[key])
    assert value(expected) < value({0: 1., 1: 1.})


@pytest.mark.parametrize('door', ['grad', 'fit'])
@pytest.mark.parametrize('failure', [None, 'forward', 'loss'])
def test_custom_in_place_excitation_edits_restore_the_original_dictionary(door, failure):
    jax = pytest.importorskip('jax')
    optax = pytest.importorskip('optax')
    s = solver()
    original = s.excitation
    original_value = complex(original[1])
    def forward(thickness, geometry=None):
        s.excitation[1] = .4j*thickness[0]/170.
        if failure == 'forward':
            raise RuntimeError('intentional custom excitation failure')
        return s.solve(geometry=geometry)
    def loss(result):
        if failure == 'loss':
            raise RuntimeError('intentional custom excitation failure')
        return result.T.sum().real
    def call():
        if door == 'grad':
            return s.grad(['thickness'], forward, loss)['thickness']
        return s.fit(['thickness'], forward, loss, optax.sgd(.1), iteration=1,
                     prepare_geometry=s.freeze_geometry)['thickness']
    with jax.checking_leaks():
        if failure:
            with pytest.raises(RuntimeError, match='intentional custom excitation failure'):
                call()
        else:
            got = float(call()[0])
    if failure is None:
        h = 1e-3
        def reference(t):
            return value({0: 1., 1: .4j*t/170.}, t)
        fd = (reference(170.+h)-reference(170.-h))/(2*h)
        assert abs(fd) > 1e-7
        expected = fd if door == 'grad' else 170.-.1*fd
        assert got == pytest.approx(expected, abs=2e-8, rel=2e-6)
    assert not any(_holds_tracer(v) for v in vars(s).values())
    assert s.excitation is original
    assert complex(original[1]) == original_value
    expected = value({0: 1., 1: .4j}, float(s.thickness[0]))
    assert float(s.solve().T.sum().real) == pytest.approx(expected, abs=1e-12)


def test_torch_excitation_conversion_preserves_the_callers_complex_leaf():
    torch = pytest.importorskip('torch')
    amplitude = torch.tensor(.4+.3j, dtype=torch.complex128, requires_grad=True)
    s = solver('torch', excitation={0: 1, 1: amplitude})
    s.solve().T.sum().real.backward()
    expected = differences({0: 1., 1: .4+.3j})[1]
    assert abs(expected) > 1e-6
    assert complex(amplitude.grad) == pytest.approx(expected, abs=2e-9, rel=2e-6)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('initial', [0, 1])
def test_weight_precision_follows_a_later_precision_assignment(backend, initial):
    s = solver(backend, complex_dtype=initial, excitation={0: 1, 1: .5j})
    s.complex_dtype = 1-initial
    for amplitude in s.excitation.values():
        assert amplitude.dtype == s.complex_dtype
    expected = solver(backend, complex_dtype=1-initial, excitation={0: 1, 1: .5j})
    np.testing.assert_allclose(s.solve().T, expected.solve().T, rtol=2e-5, atol=2e-6)
