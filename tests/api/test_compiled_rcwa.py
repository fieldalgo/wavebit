"""Optimizers cross geometry-plan boundaries without handling preparations."""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import IncompatibleProblemError, NumericalError, UnsupportedFeatureError, ValidationError


NUMERICS = rcwa.Numerics(orders=(1, 0), factorization='laurent')


@pytest.fixture
def jax():
    module = pytest.importorskip('jax')
    with module.enable_x64(True):
        yield module


def painting(parameters):
    """Overlap at x=160, disjoint at x=200; fixed material and model structure."""
    offset = parameters[0]
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(
                rcwa.Region(geometry=rcwa.Rectangle(size=(250., 200.), center=(-50., 0.)),
                            material=Material(4.)),
                rcwa.Region(geometry=rcwa.Rectangle(size=(200., 240.), center=(offset, 30.)),
                            material=Material(5.)))),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(
            rcwa.PlaneWave(theta=.2, phi=.13, polarization='p'),)))


def reflectance(result):
    return result.R.sum()


def test_automatic_repreparation_matches_fresh_values_and_derivatives(jax):
    import jax.numpy as jnp

    observe = lambda r: jnp.stack((r.R.sum(), r.T.sum()))
    evaluate = rcwa.compile(painting, observe, numerics=NUMERICS)
    initial = None
    for offset, count in ((160., 1), (161., 1), (200., 2), (201., 2), (160., 3)):
        parameters = np.asarray([offset])
        value, jacobian = evaluate.value_and_jacobian(parameters)
        fresh = rcwa.run(painting(parameters), numerics=NUMERICS)
        np.testing.assert_allclose(value, [fresh.R.sum(), fresh.T.sum()], rtol=2e-11, atol=1e-13)
        h = .001
        plus = rcwa.run(painting(parameters+h), numerics=NUMERICS)
        minus = rcwa.run(painting(parameters-h), numerics=NUMERICS)
        finite_difference = np.asarray([plus.R.sum()-minus.R.sum(), plus.T.sum()-minus.T.sum()])/(2*h)
        assert abs(finite_difference[0]) > 1e-6
        np.testing.assert_allclose(jacobian[:, 0], finite_difference, rtol=2e-5, atol=2e-10)
        assert evaluate.preparation_count == count
        if initial is None:
            initial = (value, jacobian)
    np.testing.assert_array_equal(value, initial[0])
    np.testing.assert_array_equal(jacobian, initial[1])
    np.testing.assert_array_equal(evaluate.jacobian(parameters), jacobian)


def test_reverse_gradient_follows_the_new_preparation(jax):
    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS)
    for offset in (160., 200.):
        value, gradient = evaluate.value_and_grad(np.asarray([offset]))
        expected = jax.grad(lambda x: rcwa.run(painting(x), numerics=NUMERICS,
                                              backend='jax').R.sum())(np.asarray([offset]))
        np.testing.assert_allclose(gradient, expected, rtol=2e-10, atol=1e-12)
        np.testing.assert_array_equal(evaluate([offset]), value)
    assert evaluate.preparation_count == 2


def test_nvf_painting_recovers_with_the_same_numerical_settings(jax):
    numerics = NUMERICS.replace(factorization='nvf', nvf_projector='circular', shape_nvf_grid_size=32)
    evaluate = rcwa.compile(painting, reflectance, numerics=numerics)
    for offset in (80., 200.):
        value, gradient = evaluate.value_and_grad([offset])
        expected = rcwa.run(painting([offset]), numerics=numerics).R.sum()
        h = .001
        finite_difference = (rcwa.run(painting([offset+h]), numerics=numerics).R.sum()
                             - rcwa.run(painting([offset-h]), numerics=numerics).R.sum())/(2*h)
        np.testing.assert_allclose(value, expected, rtol=2e-11, atol=1e-13)
        np.testing.assert_allclose(gradient[0], finite_difference, rtol=2e-5, atol=2e-10)
    assert evaluate.preparation_count == 2


def test_complex_observations_have_a_forward_jacobian(jax):
    observe = lambda r: r.transmission.amplitude_ratio_p.ravel()
    evaluate = rcwa.compile(painting, observe, numerics=NUMERICS)
    value, jacobian = evaluate.value_and_jacobian([160.])
    expected = observe(rcwa.run(painting([160.]), numerics=NUMERICS))
    h = .001
    finite_difference = (observe(rcwa.run(painting([160.+h]), numerics=NUMERICS))
                         - observe(rcwa.run(painting([160.-h]), numerics=NUMERICS)))/(2*h)
    np.testing.assert_allclose(value, expected, rtol=2e-11, atol=1e-13)
    np.testing.assert_allclose(jacobian[:, 0], finite_difference, rtol=2e-5, atol=2e-10)


def test_least_squares_crosses_from_overlap_to_disjoint_without_a_retry_handler(jax):
    from scipy.optimize import least_squares

    truth = 200.
    target = float(rcwa.run(painting([truth]), numerics=NUMERICS).R.sum())
    residual = rcwa.compile(painting, lambda r: r.R.sum()/target-1., numerics=NUMERICS)
    fit = least_squares(lambda x: np.atleast_1d(residual(x)), np.asarray([160.]),
                        jac=lambda x: np.asarray(residual.jacobian(x)).reshape(1, 1),
                        bounds=([140.], [240.]), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=30)
    assert fit.success, fit.message
    np.testing.assert_allclose(fit.x, [truth], rtol=0, atol=1e-5)
    assert np.linalg.norm(fit.fun) < 1e-9
    assert residual.preparation_count >= 2


def test_runtime_guard_can_recover_when_preflight_misses_a_plan_change(jax, monkeypatch):
    from wavebit import CheckReport
    from wavebit.rcwa._lifecycle import Prepared

    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS)
    evaluate([160.])
    # Still run the real compiled topology guard on the real stale painting.
    monkeypatch.setattr(Prepared, 'validate', lambda self, problem: CheckReport())
    actual = evaluate([200.])
    expected = rcwa.run(painting([200.]), numerics=NUMERICS).R.sum()
    np.testing.assert_allclose(actual, expected, rtol=2e-11)
    assert evaluate.preparation_count == 2


def test_a_fresh_plan_failure_is_bounded_and_does_not_replace_a_working_plan(jax, monkeypatch):
    import jax.numpy as jnp
    from wavebit.rcwa import _host_topology

    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS)
    good = evaluate([160.])
    with monkeypatch.context() as patch:
        patch.setattr(_host_topology, 'traced_check', lambda *args: jnp.asarray(False))
        with pytest.raises(IncompatibleProblemError, match='Fresh geometry'):
            evaluate([200.])
    assert evaluate.preparation_count == 1
    np.testing.assert_array_equal(evaluate([160.]), good)
    assert np.isfinite(evaluate([200.]))
    assert evaluate.preparation_count == 2


def test_invalid_physical_inputs_are_not_retried_or_hidden(jax):
    def build(parameters):
        p = painting(parameters)
        return p.replace(stack=p.stack.replace(layers=(
            p.stack.layers[0].replace(thickness=parameters[1]),)))

    evaluate = rcwa.compile(build, reflectance, numerics=NUMERICS)
    good = evaluate([160., 180.])
    with pytest.raises(ValidationError) as error:
        evaluate([160., -1.])
    assert error.value.code == 'layer.thickness'
    assert evaluate.preparation_count == 1
    np.testing.assert_array_equal(evaluate([160., 180.]), good)


def test_a_different_model_structure_is_not_a_geometry_retry(jax):
    def build(parameters):
        p = painting(parameters)
        if len(parameters) > 1:
            p = p.replace(stack=p.stack.replace(layers=p.stack.layers+(
                rcwa.Layer(thickness=30., background=Material(2.)),)))
        return p

    evaluate = rcwa.compile(build, reflectance, numerics=NUMERICS)
    good = evaluate([160.])
    with pytest.raises(IncompatibleProblemError) as error:
        evaluate([200., 1.])
    assert error.value.code == 'prepared.structure'
    assert evaluate.preparation_count == 1
    np.testing.assert_array_equal(evaluate([160.]), good)


@pytest.mark.parametrize('mode', ['jit', 'grad', 'vmap'])
def test_outer_transforms_are_refused_before_controller_state_changes(jax, mode):
    import jax.numpy as jnp

    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS)
    parameters = jnp.asarray([[160.], [200.]]) if mode == 'vmap' else jnp.asarray([160.])
    with pytest.raises(UnsupportedFeatureError) as error:
        getattr(jax, mode)(evaluate)(parameters)
    assert error.value.code == 'compiled.host_call'
    assert evaluate.preparation_count == 0


@pytest.mark.parametrize('bad', [[np.nan], [np.inf], [160], [160.+1j]])
def test_invalid_parameter_arrays_do_not_prepare(jax, bad):
    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS)
    with pytest.raises(ValidationError) as error:
        evaluate(bad)
    assert error.value.code == 'compiled.parameters'
    assert evaluate.preparation_count == 0


def test_a_nonfinite_selected_observation_is_a_numerical_error(jax):
    evaluate = rcwa.compile(painting, lambda r: r.R.sum()*np.nan, numerics=NUMERICS)
    with pytest.raises(NumericalError) as error:
        evaluate([160.])
    assert error.value.code == 'rcwa.nonfinite'
    assert evaluate.preparation_count == 0


def test_a_nonfinite_derivative_is_not_returned_as_a_success(jax):
    import jax.numpy as jnp

    evaluate = rcwa.compile(painting, lambda r: jnp.sqrt(0*r.R.sum()), numerics=NUMERICS)
    assert evaluate([160.]) == 0
    with pytest.raises(NumericalError) as error:
        evaluate.value_and_grad([160.])
    assert error.value.code == 'rcwa.nonfinite_derivative'
    assert evaluate.preparation_count == 1


def test_reverse_mode_requires_a_real_scalar_observation(jax):
    evaluate = rcwa.compile(painting, lambda r: r.R, numerics=NUMERICS)
    with pytest.raises(ValidationError) as error:
        evaluate.value_and_grad([160.])
    assert error.value.code == 'compiled.scalar_objective'
    assert evaluate.preparation_count == 0


def test_single_precision_controller_preserves_output_and_derivative_dtypes(jax):
    evaluate = rcwa.compile(painting, reflectance, numerics=NUMERICS, precision='single')
    value, jacobian = evaluate.value_and_jacobian([160.])
    assert value.dtype == jacobian.dtype == np.dtype('float32')


def test_scalar_parameters_reuse_a_plan_for_live_thickness(jax):
    base = painting([160.])

    def build(thickness):
        return base.replace(stack=base.stack.replace(layers=(
            base.stack.layers[0].replace(thickness=thickness),)))

    evaluate = rcwa.compile(build, reflectance, numerics=NUMERICS)
    for thickness in (180., 200.):
        value, gradient = evaluate.value_and_grad(thickness)
        fresh = lambda t: rcwa.run(build(t), numerics=NUMERICS).R.sum()
        h = .001
        finite_difference = (fresh(thickness+h)-fresh(thickness-h))/(2*h)
        np.testing.assert_allclose(value, fresh(thickness), rtol=2e-11, atol=1e-13)
        np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-10)
        assert value.shape == gradient.shape == ()
    assert evaluate.preparation_count == 1
