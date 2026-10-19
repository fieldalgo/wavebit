"""JAX GPU evidence for the public native grid and the periodic slab.

The scenes and tolerances are those of test_native_cuda.py and test_periodic_cuda.py, run
with ``backend='jax', device='gpu'``. Skips without a CUDA-enabled jaxlib are not execution
evidence.
"""
import numpy as np
import pytest

from tests.fdtd.test_native_cuda import pulse_case
from wavebit import Material, Units
from wavebit.fdtd import native as n
from wavebit.fdtd import periodic as p
from wavebit.units import C0, EPSILON0


@pytest.fixture
def gpu():
    jax = pytest.importorskip('jax'); jax.config.update('jax_enable_x64', True)
    try:
        devices = jax.devices('gpu')
    except RuntimeError:
        pytest.skip('requires a CUDA-enabled jaxlib and an NVIDIA GPU')
    return devices[0]


def placed(value, device):
    return value.devices() == {device}


def assert_field_close(actual, expected):
    # The same peak-normalized infinity error as the Torch CUDA tests.
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual-expected)) <= 5e-12*np.max(np.abs(expected))


def test_jax_gpu_conductor_matches_independent_si_recurrence(gpu):
    dt, sigma, eps, current = 1e-17, 2300., 2.4, 1e7
    problem = n.Problem(units=Units(length='nm'),
        medium=n.Medium(epsilon=np.full((3, 2, 2), eps), conductivity=sigma),
        initial=n.InitialField(E=np.full((3, 3, 2, 2), 2.), H=np.full((3, 3, 2, 2), .03)),
        current=n.Current(density=np.full((3, 3, 2, 2), current), samples=np.ones(11)))
    nums = n.Numerics(grid=n.Grid(spacing=(10.,)*3), time_step_seconds=dt, steps=11)
    result = n.run(problem, numerics=nums, backend='jax', device='gpu')
    assert placed(result.E, gpu) and placed(result.H, gpu)
    assert result.E.dtype == np.complex128
    ratio = sigma*dt/(2*EPSILON0*eps)
    expected = 2.
    for _ in range(11):
        expected = ((1-ratio)*expected-dt*current/(EPSILON0*eps))/(1+ratio)
    np.testing.assert_allclose(np.asarray(result.E), expected, rtol=2e-14)
    np.testing.assert_allclose(np.asarray(result.H), .03, rtol=2e-14)


def test_jax_gpu_cpml_dispersion_bloch_records_and_prepared_reuse(gpu):
    problem, nums = pulse_case()
    prepared = n.prepare(problem, numerics=nums, backend='jax', device='gpu')
    assert prepared.report.ok
    for amplitude in (1., 1.3):
        live = problem.replace(current=problem.current.replace(density=problem.current.density*amplitude))
        actual, expected = prepared.run(live), n.run(live, numerics=nums)
        pairs = [(getattr(actual, name), getattr(expected, name)) for name in ('E', 'H', 'spectral_power_z')]
        pairs.append((actual.spectrum.values, expected.spectrum.values))
        pairs += [(actual.snapshots[name].values, expected.snapshots[name].values) for name in actual.snapshots]
        pairs += list(zip(actual.polarization_density, expected.polarization_density))
        for value, reference in pairs:
            assert placed(value, gpu)
            assert_field_close(value, reference)


def test_jax_gpu_source_gradient_matches_cpu_and_secant(gpu):
    import jax
    import jax.numpy as jnp
    problem, nums = pulse_case()
    def objective(amplitude, device):
        live = problem.replace(current=problem.current.replace(
            density=jnp.asarray(problem.current.density)*amplitude))
        return jnp.sum(n.run(live, numerics=nums, backend='jax', device=device).spectral_power_z)
    on_gpu = jax.grad(lambda a: objective(a, 'gpu'))(1.2)
    on_cpu = jax.grad(lambda a: objective(a, 'cpu'))(1.2)
    h = 1e-4
    secant = (float(objective(1.2+h, 'cpu'))-float(objective(1.2-h, 'cpu')))/(2*h)
    assert abs(float(on_gpu)) > 0
    np.testing.assert_allclose(float(on_gpu), float(on_cpu), rtol=1e-10)
    np.testing.assert_allclose(float(on_gpu), secant, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_jax_gpu_slab_refines_towards_independent_airy_and_matches_cpu(gpu, side):
    wavelengths = np.array([.85, .9, .95])
    problem = p.Problem(units=Units(length='um'),
        stack=p.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(p.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=p.Excitation(wavelength=wavelengths, sources=(p.PlaneWave(side=side),)))
    nums = p.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=0)
    prepared = p.prepare(problem, numerics=nums, backend='jax', device='gpu')
    assert prepared.report.ok
    assert prepared.validate(problem).ok
    result = prepared.run(problem)
    cpu = p.run(problem, numerics=nums)
    for name in ('R', 'T'):
        actual = getattr(result, name)
        assert placed(actual, gpu)
        np.testing.assert_allclose(np.asarray(actual), getattr(cpu, name), rtol=2e-12, atol=1e-14)
    actual = result.reflection.amplitude_s
    assert placed(actual, gpu)
    np.testing.assert_allclose(np.asarray(actual), cpu.reflection.amplitude_s, rtol=2e-12, atol=1e-14)
    phase = np.exp(4j*np.pi*1.5*.1/wavelengths)
    exact = (-.2+.2*phase)/(1-.04*phase)
    fine = p.run(problem, numerics=nums.replace(spacing=nums.spacing/2), backend='jax', device='gpu')
    fine_r = np.asarray(fine.reflection.amplitude_s)[:, 0, 0]
    np.testing.assert_allclose(fine_r, exact, atol=.005)
    assert np.linalg.norm(fine_r-exact) < np.linalg.norm(np.asarray(actual)[:, 0, 0]-exact)
