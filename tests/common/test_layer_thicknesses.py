"""Independent simplex derivatives and shared physical Stack constraints."""

from wavebit._common.periodic import Incidence as _legacy_wavebit_Incidence

from wavebit._common.periodic import Stack as _legacy_wavebit_Stack

from wavebit._common.periodic import layer_thicknesses as _legacy_wavebit_layer_thicknesses

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

from wavebit.fdtd._periodic.runtime import time_loop_backend
from tests.fdtd.test_fdtd_scene_angles import value_and_jacobian


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_lengths_and_derivatives_match_analytic_constrained_partition(backend):
    if backend != 'numpy':
        pytest.importorskip('jax' if backend == 'jax' else 'torch')
    p = np.array([.2, -.4, .6])
    minimum = np.array([.02, .04, .03])
    ratio = np.exp(p)/np.exp(p).sum()
    expected = minimum+(.5-minimum.sum())*ratio
    jacobian = (.5-minimum.sum())*(np.diag(ratio)-np.outer(ratio, ratio))
    def objective(weights):
        return _legacy_wavebit_layer_thicknesses(weights, .5, minimum=minimum)
    if backend != 'numpy':
        value, derivative = value_and_jacobian(objective, p, backend)
        np.testing.assert_allclose(derivative, jacobian, atol=2e-15)
        np.testing.assert_allclose(derivative.sum(axis=0), 0., atol=2e-15)
    else:
        value = objective(p)
    np.testing.assert_allclose(value, expected, atol=2e-15)
    np.testing.assert_allclose(value.sum(), .5, atol=2e-16)
    assert np.all(value >= minimum)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_additive_gauge_permutation_saturation_and_precision(backend):
    if backend != 'numpy':
        pytest.importorskip('jax' if backend == 'jax' else 'torch')
    runtime = time_loop_backend(backend)
    array, xp = runtime.array, runtime.xp
    p = array([.2, -.4, .6])
    minimum = np.array([.02, .04, .03])
    base = _legacy_wavebit_layer_thicknesses(p, .5, minimum=minimum)
    for shift in (-1000., 1000.):
        np.testing.assert_allclose(_legacy_wavebit_layer_thicknesses(p+shift, .5, minimum=minimum), base, atol=1e-14)
    permuted = _legacy_wavebit_layer_thicknesses(xp.flip(p, axis=0), .5, minimum=minimum[::-1])
    np.testing.assert_allclose(permuted, xp.flip(base, axis=0), atol=2e-15)
    saturated = _legacy_wavebit_layer_thicknesses(array([1000., -1000.]), .5, minimum=.04)
    np.testing.assert_allclose(saturated, [.46, .04], atol=2e-15)
    narrow = _legacy_wavebit_layer_thicknesses(array([1., 2.], dtype=xp.float32), .5, minimum=.04)
    assert narrow.dtype == xp.float32
    assert getattr(narrow, 'device', None) == getattr(p, 'device', None)


def test_plain_lists_integer_weights_and_single_layer():
    np.testing.assert_allclose(_legacy_wavebit_layer_thicknesses([0, 0, 0], .6, minimum=.1), [.2]*3, atol=1e-15)
    assert _legacy_wavebit_layer_thicknesses(np.array([1, 2, 3]), .6).dtype == np.float64
    np.testing.assert_array_equal(_legacy_wavebit_layer_thicknesses([-.7], .6, minimum=.2), [.6])


@pytest.mark.parametrize('weights', [[], 1., [[0., 1.]], [True, False], [0., 1j], ['x', 'y'], [0., np.nan], [np.inf, 0.], [1000., -1000.]])
def test_invalid_weights_raise(weights):
    with pytest.raises(ValueError, match='logits|Thicknesses'):
        _legacy_wavebit_layer_thicknesses(weights, .5)


@pytest.mark.parametrize('changes', [
    dict(total=0.), dict(total=-1.), dict(total=np.inf), dict(total=True),
    dict(total=[.5]), dict(total=.5j), dict(minimum=-.1), dict(minimum=[.02]),
    dict(minimum=[.1, .2, .3]), dict(minimum=.3), dict(minimum=.25),
    dict(minimum=np.nan), dict(minimum=True), dict(minimum=[.1j, .1]),
])
def test_invalid_or_infeasible_static_constraints_raise(changes):
    with pytest.raises(ValueError, match='total|minimum'):
        _legacy_wavebit_layer_thicknesses([0., 0.], **dict(dict(total=.5), **changes))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_total_and_minima_are_static_host_inputs(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    array = time_loop_backend(backend).array
    for changes in (dict(total=array(.5)), dict(minimum=array(.04)), dict(minimum=[array(.04), array(.04)])):
        with pytest.raises(ValueError, match='static real host'):
            _legacy_wavebit_layer_thicknesses(array([0., 0.]), **dict(dict(total=.5), **changes))


def test_jit_masks_nonfinite_weights_and_underflow_on_one_trace():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    traces = []
    def decode(p):
        traces.append(None)
        return _legacy_wavebit_layer_thicknesses(p, .5)
    evaluate = jax.jit(decode)
    np.testing.assert_allclose(evaluate(np.array([0., 0.])), [.25, .25], atol=1e-15)
    assert np.isfinite(evaluate(np.array([1., -1.]))).all()
    for p in ([np.nan, 0.], [np.inf, 0.], [1000., -1000.]):
        assert np.isnan(evaluate(np.array(p))).all()
    assert len(traces) == 1


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_typed_rcwa_preserves_constrained_multilayer_gradient(backend):
    pytest.importorskip('jax' if backend == 'jax' else 'torch')
    def objective(p, which):
        runtime = time_loop_backend(which)
        epsilon = runtime.array([1.4, 2.2, 3.1])[:, None, None]
        lengths = _legacy_wavebit_layer_thicknesses(p, .6, minimum=.04)
        stack = _legacy_wavebit_Stack.from_structure(epsilon, lengths, top=1., bot=4., period=.05)
        light = _legacy_wavebit_Incidence(wavelength=1., theta=.2, polarization=(1., .4j))
        return _legacy_wavebit_rcwa(stack=stack, incidence=light, backend=which, truncation_order=0).solve().incident.R.reshape(1)
    p = np.array([-.2, .3, -.1])
    value, gradient = value_and_jacobian(lambda p: objective(p, backend), p, backend)
    np.testing.assert_allclose(value, objective(p, 'numpy'), atol=1e-12)
    for step in (2e-5, 1e-5):
        finite = np.stack([(objective(p+v, 'numpy')-objective(p-v, 'numpy'))/(2*step) for v in step*np.eye(3)], axis=-1)
        np.testing.assert_allclose(gradient, finite, atol=2e-8, rtol=2e-5)
    assert np.max(abs(gradient)) > .01
