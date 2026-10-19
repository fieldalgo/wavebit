"""CUDA evidence for the public native grid, including an independent SI oracle."""
import numpy as np
import pytest

from wavebit import Units
from wavebit.fdtd import native as n
from wavebit.units import EPSILON0


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and an NVIDIA GPU')
    return torch


def host(value):
    return value.detach().cpu().numpy()


def assert_field_close(actual, expected):
    # SI E, H, polarization and Fourier power have different units and scales.
    # A peak-normalized infinity error also covers components at cancellation zeros.
    actual, expected = host(actual), np.asarray(expected)
    scale = np.max(np.abs(expected))
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual-expected)) <= 5e-12*scale


def test_cuda_conductor_matches_independent_si_recurrence(torch_cuda):
    dt, sigma, eps, current = 1e-17, 2300., 2.4, 1e7
    p = n.Problem(units=Units(length='nm'),
        medium=n.Medium(epsilon=np.full((3, 2, 2), eps), conductivity=sigma),
        initial=n.InitialField(E=np.full((3, 3, 2, 2), 2.), H=np.full((3, 3, 2, 2), .03)),
        current=n.Current(density=np.full((3, 3, 2, 2), current), samples=np.ones(11)))
    nums = n.Numerics(grid=n.Grid(spacing=(10.,)*3), time_step_seconds=dt, steps=11)
    result = n.run(p, numerics=nums, backend='torch', device='cuda:0')
    assert result.E.device.type == result.H.device.type == 'cuda'
    assert result.E.dtype == torch_cuda.complex128
    ratio = sigma*dt/(2*EPSILON0*eps)
    expected = 2.
    for _ in range(11):
        expected = ((1-ratio)*expected-dt*current/(EPSILON0*eps))/(1+ratio)
    np.testing.assert_allclose(host(result.E), expected, rtol=2e-14)
    np.testing.assert_allclose(host(result.H), .03, rtol=2e-14)


def pulse_case():
    shape = (16, 2, 2)
    units = Units(length='um')
    ts = units.time_unit_seconds
    mask = np.zeros(shape); mask[7:9] = 1.
    current = np.zeros((3,)+shape); current[0, 5] = 1e7
    p = n.Problem(units=units,
        medium=n.Medium(epsilon=1.+mask,
            poles=(n.Oscillator(omega_0=2/ts, gamma=.1/ts, strength=.4*mask/ts**2),)),
        current=n.Current(density=current, samples=np.sin(np.arange(30)*.1)),
        bloch_phase=(0., .13, -.08),
        outputs=n.Outputs(spectrum=n.Spectrum(wavelengths=(.8, 1.), z_indices=(6, 10)),
            snapshots=n.Snapshots(every=3, components=('Ex', 'Hy'))))
    nums = n.Numerics(grid=n.Grid(spacing=(.1,)*3, z_spacing=np.linspace(.09, .11, 16)),
        time_step_seconds=.02*ts, steps=30, absorber=n.PML(cells=3, wavelength=1.))
    return p, nums


def test_cuda_cpml_dispersion_bloch_records_and_prepared_reuse(torch_cuda):
    p, nums = pulse_case()
    prepared = n.prepare(p, numerics=nums, backend='torch', device='cuda:0')
    assert prepared.report.ok
    for amplitude in (1., 1.3):
        live = p.replace(current=p.current.replace(density=p.current.density*amplitude))
        actual = prepared.run(live)
        expected = n.run(live, numerics=nums)
        for name in ('E', 'H', 'spectral_power_z'):
            value = getattr(actual, name)
            assert value.device.type == 'cuda'
            assert_field_close(value, getattr(expected, name))
        assert_field_close(actual.spectrum.values, expected.spectrum.values)
        for name in actual.snapshots:
            assert actual.snapshots[name].values.device.type == 'cuda'
            assert_field_close(actual.snapshots[name].values, expected.snapshots[name].values)
        for value, reference in zip(actual.polarization_density, expected.polarization_density):
            assert value.device.type == 'cuda'
            assert_field_close(value, reference)


@pytest.mark.parametrize('checkpoint', [32, 7])
@pytest.mark.parametrize('leaf_device', ['cpu', 'cuda:0'])
def test_cuda_source_gradient_matches_independent_scaling_and_cpu_secant(torch_cuda, checkpoint, leaf_device):
    torch = torch_cuda
    p, nums = pulse_case()
    # Snapshots every three steps split the loop into three-step intervals,
    # hiding the distinct 32/7 checkpoint schedules. Exercise the full 30-step
    # loop here; the recorded-observable derivative is checked separately below.
    p = p.replace(outputs=p.outputs.replace(snapshots=None))
    nums = nums.replace(checkpoint_steps=checkpoint)
    amplitude = torch.tensor(1.2, dtype=torch.float64, device=leaf_device, requires_grad=True)
    density = torch.tensor(p.current.density, dtype=torch.float64, device=leaf_device)
    live = p.replace(current=p.current.replace(density=density*amplitude))
    result = n.run(live, numerics=nums, backend='torch', device='cuda:0')
    loss = result.E.abs().square().sum()
    gradient = torch.autograd.grad(loss, amplitude)[0]
    assert gradient.device == amplitude.device
    assert result.E.device.type == 'cuda'
    # Linear Maxwell evolution with zero initial fields makes this loss quadratic.
    np.testing.assert_allclose(host(gradient), 2*host(loss)/1.2, rtol=2e-12)
    def objective(value):
        q = p.replace(current=p.current.replace(density=p.current.density*value))
        return np.sum(np.abs(n.run(q, numerics=nums).E)**2)
    h = 1e-5
    finite = (objective(1.2+h)-objective(1.2-h))/(2*h)
    np.testing.assert_allclose(host(gradient), finite, rtol=2e-7)


def test_cuda_recorded_observation_keeps_source_derivative(torch_cuda):
    torch = torch_cuda
    p, nums = pulse_case()
    amplitude = torch.tensor(1.2, dtype=torch.float64, device='cuda', requires_grad=True)
    density = torch.tensor(p.current.density, dtype=torch.float64, device='cuda')
    q = p.replace(current=p.current.replace(density=density*amplitude))
    result = n.run(q, numerics=nums, backend='torch', device='cuda:0')
    value = result.snapshots['Ex'].values.abs().square().sum()
    gradient = torch.autograd.grad(value, amplitude)[0]
    assert value.item() > 0
    np.testing.assert_allclose(host(gradient), 2*host(value)/1.2, rtol=2e-12)


def test_cuda_material_and_bloch_gradients_match_cpu_secants(torch_cuda):
    torch = torch_cuda
    p, nums = pulse_case()
    mask = np.zeros((16, 2, 2)); mask[7:9] = 1.
    ts = p.units.time_unit_seconds
    def objective(v, backend):
        weights = torch.tensor(mask, device='cuda', dtype=torch.float64) if backend == 'torch' else mask
        q = p.replace(medium=p.medium.replace(epsilon=1.+v[0]*weights,
            poles=(n.Oscillator(omega_0=2/ts, gamma=.1/ts, strength=v[1]*weights/ts**2),)),
            bloch_phase=(0., v[2], -.08))
        result = n.run(q, numerics=nums, backend=backend,
                       device='cuda:0' if backend == 'torch' else 'cpu')
        return (result.E.real**2+result.E.imag**2).sum()
    x = np.array([1., .4, .13])
    live = torch.tensor(x, device='cuda', dtype=torch.float64, requires_grad=True)
    loss = objective(live, 'torch')
    gradient = host(torch.autograd.grad(loss, live)[0])
    np.testing.assert_allclose(host(loss), objective(x, 'numpy'), rtol=2e-12)
    for h in (1e-4, 5e-5):
        finite = np.array([(objective(x+row*h, 'numpy')-objective(x-row*h, 'numpy'))/(2*h)
                           for row in np.eye(3)])
        assert np.all(np.abs(finite) > 1e-4)
        np.testing.assert_allclose(gradient, finite, rtol=2e-6)
