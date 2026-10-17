"""Native device admission and device-local observation contracts."""
import numpy as np
import pytest

from wavebit import Units
from wavebit.errors import UnsupportedFeatureError
from wavebit.fdtd import native as n
from wavebit.units import Z0


def small_problem():
    units = Units(length='um')
    shape = (8, 3, 2)
    density = np.zeros((3,) + shape)
    density[0, 3, 1, 0] = 1e7
    problem = n.Problem(
        units=units, medium=n.Medium(epsilon=np.ones(shape)),
        current=n.Current(density=density, samples=np.ones(5)),
        outputs=n.Outputs(
            spectrum=n.Spectrum(wavelengths=(.8,), planes=(4,)),
            record=n.Record(every=2, components=('ex', 'hy'))))
    numerics = n.Numerics(
        grid=n.Grid(spacing=(.1,) * 3),
        time_step_seconds=.02 * units.time_seconds, steps=5,
        tail=n.Tail(window_seconds=.04 * units.time_seconds))
    return problem, numerics


@pytest.fixture(params=['cpu', 'cuda:0'])
def torch_device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and an NVIDIA GPU')
    return torch.device(request.param)


def test_unavailable_cuda_is_reported_without_execution(monkeypatch):
    torch = pytest.importorskip('torch')
    problem, numerics = small_problem()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = n.check(problem, numerics=numerics, backend='torch', device='cuda:0')
    assert not report.ok
    assert report.issues[0].code == 'device_unavailable'
    assert report.issues[0].path == 'execution.device'
    for action in (n.prepare, n.run):
        with pytest.raises(UnsupportedFeatureError, match='CUDA device is unavailable'):
            action(problem, numerics=numerics, backend='torch', device='cuda:0')


@pytest.mark.parametrize('invalid', ['epsilon', 'cfl', 'pml_medium', 'pml_phase'])
def test_native_rejects_invalid_inputs_before_advancing(torch_device, monkeypatch, invalid):
    from wavebit.fdtd._periodic.runtime import TorchTimeLoop

    problem, numerics = small_problem()
    if invalid == 'epsilon':
        problem = problem.replace(medium=n.Medium(epsilon=-problem.medium.epsilon))
    elif invalid == 'cfl':
        numerics = numerics.replace(time_step_seconds=problem.units.time_seconds)
    elif invalid == 'pml_medium':
        numerics = numerics.replace(absorber=n.PML(cells=2))
        problem = problem.replace(medium=n.Medium(epsilon=2 * problem.medium.epsilon))
    else:
        numerics = numerics.replace(absorber=n.PML(cells=2))
        problem = problem.replace(bloch_phase=(.1, 0., 0.))

    def refuse(*args, **kwargs):
        raise AssertionError('invalid input advanced the time loop')

    monkeypatch.setattr(TorchTimeLoop, 'advance', refuse)
    monkeypatch.setattr(TorchTimeLoop, 'advance_recorded', refuse)
    options = dict(numerics=numerics, backend='torch', device=str(torch_device))
    assert not n.check(problem, **options).ok
    for action in (n.prepare, n.run):
        with pytest.raises(ValueError, match='Invalid material/poles/PML background or dt'):
            action(problem, **options)


def test_native_observations_share_the_execution_device(torch_device):
    torch = pytest.importorskip('torch')
    problem, numerics = small_problem()
    result = n.run(problem, numerics=numerics, backend='torch', device=str(torch_device))
    reference = n.run(problem, numerics=numerics)
    assert result.E.device == result.H.device == torch_device
    assert result.spectrum.values.device == result.spectral_power_z.device == torch_device
    assert result.E.dtype == result.H.dtype == result.spectrum.values.dtype == torch.complex128
    assert result.spectral_power_z.dtype == torch.float64
    for name, coordinate in result.spectrum.coordinates.items():
        assert coordinate.device == torch_device, name
        assert coordinate.dtype == torch.float64, name
        np.testing.assert_allclose(coordinate.detach().cpu(), reference.spectrum.coordinates[name],
                                   rtol=2e-15, atol=0.)
    for name in ('wavelengths', 'frequencies'):
        value = getattr(result.spectrum, name)
        assert value.device == torch_device and value.dtype == torch.float64
        np.testing.assert_allclose(value.detach().cpu(), getattr(reference.spectrum, name),
                                   rtol=2e-15, atol=0.)
    for field in result.fields.values():
        assert field.values.device == field.time.device == torch_device
        assert field.values.dtype == torch.complex128 and field.time.dtype == torch.float64
        assert all(coordinate.device == torch_device for coordinate in field.coordinates.values())
        assert all(coordinate.dtype == torch.float64 for coordinate in field.coordinates.values())
    for value in result.validity.checks.values():
        assert value.device == torch_device
    for value in result.diagnostics['tail'].values():
        if isinstance(value, torch.Tensor):
            assert value.device == torch_device


def test_native_check_and_prepare_do_not_advance_cuda(torch_device, monkeypatch):
    from wavebit.fdtd._periodic.runtime import TorchTimeLoop

    problem, numerics = small_problem()

    def refuse(*args, **kwargs):
        raise AssertionError('admission advanced the time loop')

    monkeypatch.setattr(TorchTimeLoop, 'advance', refuse)
    monkeypatch.setattr(TorchTimeLoop, 'advance_recorded', refuse)
    options = dict(numerics=numerics, backend='torch', device=str(torch_device))
    assert n.check(problem, **options).ok
    prepared = n.prepare(problem, **options)
    assert prepared.report.ok
    assert prepared.validate(problem).ok
    assert not prepared.validate(problem.replace(outputs=n.Outputs())).ok


@pytest.mark.parametrize('phase', [0., .37])
@pytest.mark.parametrize('steps', [37, 513])
def test_native_plane_wave_matches_independent_yee_dispersion(torch_device, phase, steps):
    # In a uniform medium, centered Yee differences admit a discrete plane wave:
    # sin(omega*dt/2) = dt*sin(k*dz/2)/(dz*sqrt(epsilon*mu)), with c=1.
    # The twist exp(i*phase) closes the grid; the chosen mode makes two turns
    # plus that twist. H is offset by dz/2 in space and -dt/2 in time.
    units = Units(length='um')
    shape = (16, 2, 2)
    epsilon, mu, dz, dt = 2.4, 1.7, .05, .02
    k = (4 * np.pi + phase) / (shape[0] * dz)
    omega = 2 / dt * np.arcsin(dt / (dz * np.sqrt(epsilon * mu)) * np.sin(k * dz / 2))
    z = np.arange(shape[0])[:, None, None] * dz
    electric = np.zeros((3,) + shape, dtype=complex)
    magnetic = np.zeros_like(electric)
    electric[0] = np.exp(1j * k * z)
    magnetic[1] = np.sqrt(epsilon / mu) / Z0 * np.exp(1j * k * (z + dz / 2) + 1j * omega * dt / 2)
    problem = n.Problem(
        units=units, medium=n.Medium(epsilon=np.full(shape, epsilon), mu=mu),
        initial=n.Initial(E=electric, H=magnetic), bloch_phase=(phase, 0., 0.))
    numerics = n.Numerics(
        grid=n.Grid(spacing=(dz,) * 3), time_step_seconds=dt * units.time_seconds,
        steps=steps, checkpoint_steps=7)
    result = n.run(problem, numerics=numerics, backend='torch', device=str(torch_device))
    rotation = np.exp(-1j * omega * dt * steps)
    np.testing.assert_allclose(result.E.detach().cpu(), electric * rotation, rtol=3e-13, atol=0.)
    np.testing.assert_allclose(result.H.detach().cpu(), magnetic * rotation, rtol=3e-13, atol=0.)
