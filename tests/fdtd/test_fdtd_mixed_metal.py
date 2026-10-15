"""Mixed Drude/Lorentz scene amplitudes and all material-parameter graphs."""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd
import numpy as np
import pytest



def solve(factors, backend):
    # Relative factors for epsilon_inf, Drude gamma/strength, and Lorentz
    # omega0/gamma/strength. Drude's zero resonance remains exactly zero.
    epsilon = 1.5*factors[0]
    poles = [(0., .08*factors[1], 160*factors[2]),
             (10*factors[3], .12*factors[4], 40*factors[5])]
    result = _legacy_wavebit_fdtd(
        structure=epsilon.reshape(1, 1, 1), thickness=[.2], top=1., bot=1.,
        period=.05, wavelength=[.9, 1., 1.1], backend=backend, resolution=10,
        n_periods=30, n_pml=4, truncation_order=0, tetm='te',
        dispersion=np.ones((1, 1, 1), int), poles={1: poles}, checkpoint_steps=32)
    # Include phase, not just power: conjugation/sign errors must be visible.
    objective = sum((v.incident.r_s+.4j*v.incident.t_s).real.sum() for v in result)
    return objective, result


@pytest.fixture(scope='module')
def numpy_reference():
    factors = np.ones(6)
    objective, result = solve(factors, 'numpy')
    finite = []
    for delta in (1e-4, 5e-5):
        derivatives = []
        for index in range(6):
            offset = np.eye(6)[index]*delta
            derivatives.append((solve(factors+offset, 'numpy')[0]-solve(factors-offset, 'numpy')[0])/(2*delta))
        finite.append(np.array(derivatives))
    np.testing.assert_allclose(*finite, rtol=3e-5, atol=2e-8)
    return objective, result, finite[-1]


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_all_mixed_scene_material_gradients_match_numpy(numpy_reference, backend):
    expected, reference, finite = numpy_reference
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        (value, actual), derivative = jax.jit(jax.value_and_grad(
            lambda factors: solve(factors, 'jax'), has_aux=True))(jnp.ones(6))
        convert = np.asarray
    else:
        torch = pytest.importorskip('torch')
        factors = torch.ones(6, dtype=torch.float64, requires_grad=True)
        value, actual = solve(factors, 'torch')
        derivative, = torch.autograd.grad(value, factors)
        convert = lambda v: v.detach().numpy()
    np.testing.assert_allclose(convert(value), expected, atol=1e-10, rtol=0)
    for a, b in zip(actual, reference):
        for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            np.testing.assert_allclose(convert(getattr(a.incident, key)), getattr(b.incident, key),
                                       atol=1e-10, rtol=0)
    assert np.all(abs(finite) > 1e-5), 'Every parameter must affect this objective.'
    np.testing.assert_allclose(convert(derivative), finite, rtol=3e-5, atol=2e-8)


@pytest.mark.slow
def test_mixed_metal_matches_independent_discrete_ade_boundary_problem():
    # Harmonic substitution into centered P''/P' gives the discrete material
    # susceptibility below. Solve the resulting Yee Helmholtz matrix with exact
    # outgoing vacuum leads: no time loop, CPML, material sampler or calibration.
    wavelengths = np.array([.9, 1., 1.1])
    omega = 2*np.pi/wavelengths
    epsilon = 1.5+160/(-omega**2-1j*.08*omega)+40/(100-omega**2-1j*.12*omega)
    index = np.sqrt(epsilon)
    interface = (1-index)/(1+index)
    propagation = np.exp(1j*omega*index*.2)
    denominator = 1-interface**2*propagation**2
    continuum = np.array([interface*(1-propagation**2)/denominator,
                          (1-interface**2)*propagation/denominator])
    errors = []
    for resolution in (40, 80):
        h, dt = 1/resolution, .8/(resolution*np.sqrt(3))
        z = np.arange(-1, round(.2/h)+2)*h
        fraction = np.maximum(0, np.minimum(z+h/2, .2)-np.maximum(z-h/2, 0))/h
        expected = []
        for w in omega:
            frequency2 = (2*np.sin(w*dt/2)/dt)**2
            damping_frequency = np.sin(w*dt)/dt
            material = 1.5+160/(-frequency2-1j*.08*damping_frequency)
            material += 40/(100-frequency2-1j*.12*damping_frequency)
            sampled = 1+fraction*(material-1)
            phase = 2*np.arcsin(h*np.sqrt(frequency2)/2)
            operator = np.diag(h*h*frequency2*sampled-2)
            operator += np.diag(np.ones(len(z)-1), 1)+np.diag(np.ones(len(z)-1), -1)
            operator[0, 0] += np.exp(1j*phase)
            operator[-1, -1] += np.exp(1j*phase)
            drive = np.zeros(len(z), complex)
            drive[0] = 2j*np.sin(phase)*np.exp(1j*phase*z[0]/h)
            field = np.linalg.solve(operator, drive)
            expected.append([(field[0]-np.exp(1j*phase*z[0]/h))*np.exp(1j*phase*z[0]/h),
                             field[-1]*np.exp(-1j*phase*(z[-1]-.2)/h)])
        results, diagnostic = _legacy_wavebit_fdtd(
            structure=np.full((1, 1, 1), 1.5), thickness=[.2], top=1., bot=1.,
            period=.05, wavelength=wavelengths, backend='numpy', resolution=resolution,
            n_periods=160, n_pml=round(.3*resolution), truncation_order=0, tetm='te',
            dispersion=np.ones((1, 1, 1), int), poles={1: [(0., .08, 160.), (10., .12, 40.)]},
            return_diagnostics=True, convergence_window=round(2/dt))
        actual = np.array([[v.incident.r_s.item(), v.incident.t_s.item()] for v in results]).T
        np.testing.assert_allclose(diagnostic['dt'], dt, atol=1e-16)
        np.testing.assert_allclose(actual, np.array(expected).T, atol=3e-4, rtol=0)
        assert diagnostic['convergence']['state_decayed']
        assert diagnostic['convergence']['amplitudes_settled']
        errors.append(np.max(abs(actual-continuum)))
    assert errors[1] < .003
    assert errors[1] < errors[0]*.4
