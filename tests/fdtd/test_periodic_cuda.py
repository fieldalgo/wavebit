"""CUDA periodic scattering against CPU results and an independent Airy slab."""
import numpy as np
import pytest

from wavebit import Material, Units
from wavebit.fdtd import periodic as p
from wavebit.units import C0


@pytest.fixture
def cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA-enabled Torch and an NVIDIA GPU')
    return torch


def host(value):
    return value.detach().cpu().numpy()


@pytest.mark.slow
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_cuda_slab_refines_towards_independent_airy_and_matches_cpu(cuda, side):
    wavelengths = np.array([.85, .9, .95])
    problem = p.Problem(units=Units(length='um'),
        stack=p.Stack(period=(.2,), top=Material(1.), bottom=Material(1.),
            layers=(p.Layer(thickness=.1, epsilon=np.full((1, 2), 2.25)),)),
        excitation=p.Excitation(wavelength=wavelengths, sources=(p.PlaneWave(side=side),)))
    nums = p.Numerics(spacing=1/30, duration_seconds=55*.9e-6/C0, orders=(0, 0))
    prepared = p.prepare(problem, numerics=nums, backend='torch', device='cuda:0')
    assert prepared.report.ok
    assert prepared.validate(problem).ok
    result = prepared.run(problem)
    cpu = p.run(problem, numerics=nums)
    for name in ('R', 'T'):
        actual = getattr(result, name)
        assert actual.device.type == 'cuda'
        np.testing.assert_allclose(host(actual), getattr(cpu, name), rtol=2e-12, atol=1e-14)
    actual = result.reflection.amplitude_s
    assert actual.device.type == 'cuda'
    np.testing.assert_allclose(host(actual), cpu.reflection.amplitude_s, rtol=2e-12, atol=1e-14)
    phase = np.exp(4j*np.pi*1.5*.1/wavelengths)
    exact = (-.2+.2*phase)/(1-.04*phase)
    fine = p.run(problem, numerics=nums.replace(spacing=nums.spacing/2),
                 backend='torch', device='cuda:0')
    fine_r = host(fine.reflection.amplitude_s)[:, 0, 0]
    np.testing.assert_allclose(fine_r, exact, atol=.005)
    assert np.linalg.norm(fine_r-exact) < np.linalg.norm(host(actual)[:, 0, 0]-exact)
