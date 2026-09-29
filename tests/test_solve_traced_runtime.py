"""High-level traced solves release live state after complete field operations."""
import numpy as np
import pytest
import wavebit
from wavebit._core.engine import _holds_tracer

CFG = dict(period=(600., 750.), wavelength=900., thickness=(170.,),
           truncation_order=(1, 1), theta=.13, phi=.2, answers='field',
           structure=np.array([[[4., 1.], [1., 1.]]]),
           factorization='li', fourier_coefficients='integrated')


@pytest.mark.parametrize('door', ['solve', 'solve_field_eager', 'solve_field'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_captured_solver_field_gradient_releases_runtime_and_keeps_saved_fields(door, algorithm):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    s = wavebit.rcwa(backend=1, **CFG, layer_algorithm=algorithm)
    saved = s.solve()
    saved_field = np.asarray(saved.reconstruct_field(2, 2, 2))
    geometry = s.freeze_geometry()
    def loss(t):
        if door == 'solve':
            r = s.solve(thickness=[t], geometry=geometry)
            field = r.reconstruct_field(2, 2, 2)
        else:
            r, field = getattr(s, door)(2, 2, 2, thickness=[t], geometry=geometry)
        return jnp.sum(jnp.abs(field)**2)
    with jax.checking_leaks():
        evaluate = jax.jit(jax.value_and_grad(loss))
        got_value, got_gradient = evaluate(jnp.asarray(170.))
        again = evaluate(jnp.asarray(185.))
    assert np.isfinite(np.asarray(again)).all()
    assert not any(_holds_tracer(v) for v in s.__dict__.values())
    if door == 'solve_field':
        # Its traced copy owns the new runtime; the unchanged live scene keeps
        # its previous concrete solve, which is still a valid reconstruction.
        np.testing.assert_allclose(s.reconstruct_field(2, 2, 2)[0], saved_field, atol=1e-12)
    else:
        assert s.T1 is None and s.layer_info_list == [] and s._scattering is None
    def reference(t):
        r = wavebit.rcwa(backend=0, **(CFG | {'thickness': [t]}), layer_algorithm=algorithm).solve()
        return np.sum(np.abs(r.reconstruct_field(2, 2, 2))**2)
    h = 1e-3
    expected_gradient = (reference(170.+h)-reference(170.-h))/(2*h)
    assert float(got_value) == pytest.approx(reference(170.), abs=2e-11)
    assert float(got_gradient) == pytest.approx(expected_gradient, abs=2e-8, rel=2e-5)
    np.testing.assert_array_equal(saved.reconstruct_field(2, 2, 2), saved_field)
    np.testing.assert_allclose(s.solve().reconstruct_field(2, 2, 2), saved_field, atol=1e-12)


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_high_level_solve_releases_partial_modes_when_finishing_raises(algorithm):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    wavebit.enable_x64()
    s = wavebit.rcwa(backend=1, **CFG, layer_algorithm=algorithm)
    geometry = s.freeze_geometry()
    finish = s._finish
    def fail(out):
        finish(out)
        raise RuntimeError('intentional post-solve failure')
    s._finish = fail
    with jax.checking_leaks(), pytest.raises(RuntimeError, match='post-solve failure'):
        jax.jit(lambda t: s.solve(thickness=[t], geometry=geometry).T.sum())(jnp.asarray(170.))
    assert not any(_holds_tracer(v) for v in s.__dict__.values())
    assert s.T1 is None and s.layer_info_list == [] and s._scattering is None
    s._finish = finish
    assert np.isfinite(s.solve().T).all()
