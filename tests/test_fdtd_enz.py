"""Lossy ENZ amplitudes, normal electric fields and material sensitivities."""
import numpy as np
import pytest

from studies.fdtd_enz import BAND, reference, run, solve


def test_lossy_enz_resolves_both_basis_tails_and_refines_to_continuum():
    epsilon, _, _ = reference()
    assert abs(epsilon[1].real) < 1e-14
    assert epsilon[0].real > 0 > epsilon[-1].real
    assert np.all(epsilon.imag > 0)
    _, early = run(40, 'numpy', periods=60)
    assert early['amplitudes_settled']['s_inc']
    assert not early['amplitudes_settled']['p_inc']
    assert not early['scattering_settled']
    assert not early['state_decayed']
    errors = []
    for resolution in (40, 80):
        # ENZ is sensitive to ADE time dispersion: this accuracy target uses
        # half the default Courant fraction, independently of pulse duration.
        _, row = run(resolution, 'numpy', courant=.4)
        assert row['state_decayed']
        assert row['scattering_settled']
        assert all(row['amplitudes_settled'].values())
        errors.append(np.array([*row['error'].values(), row['ez_error']]))
    assert np.all(errors[1] < .01)
    assert np.all(errors[1] < .4*errors[0])


@pytest.fixture(scope='module')
def material_reference():
    value = diagnostic_objective(np.ones(3), 0)
    estimates = []
    for delta in (1e-4, 5e-5):
        estimates.append(np.array([(diagnostic_objective(np.ones(3)+step, 0)
                                   - diagnostic_objective(np.ones(3)-step, 0))/(2*delta)
                                  for step in np.eye(3)*delta]))
    np.testing.assert_allclose(*estimates, rtol=3e-5, atol=1e-8)
    assert np.all(abs(estimates[-1]) > 1e-4)
    return value, estimates[-1]


def diagnostic_objective(factors, backend):
    # This is the derivative of a finite-duration calculation, not a claim of
    # steady-state accuracy at this coarse resolution and short duration.
    results, d = solve(factors, backend, 10, periods=60, side='bottom', pml_width=.4)
    c = d['convergence']
    phase = sum((v.p_inc.r_p+.4j*v.p_inc.t_p).real.sum() for v in results)/len(BAND)
    return (phase+.01*(abs(d['monitors']['middle']['fields']['ez'])**2).sum()
            +c['p_inc']['amplitude_change']['r_p'].real.sum()
            +c['scattering_change_bound']['r_p'].sum())


@pytest.mark.parametrize('backend', [1, 2])
def test_enz_material_and_new_diagnostic_gradients_match_finite_difference(material_reference, backend):
    expected, finite = material_reference
    if backend == 1:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        value, gradient = jax.jit(jax.value_and_grad(lambda x: diagnostic_objective(x, 1)))(jnp.ones(3))
        convert = np.asarray
    else:
        torch = pytest.importorskip('torch')
        factors = torch.ones(3, dtype=torch.float64, requires_grad=True)
        value = diagnostic_objective(factors, 2)
        gradient, = torch.autograd.grad(value, factors)
        convert = lambda x: x.detach().numpy()
    np.testing.assert_allclose(convert(value), expected, atol=1e-10)
    np.testing.assert_allclose(convert(gradient), finite, rtol=3e-5, atol=1e-8)
