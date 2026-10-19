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
            spectrum=n.Spectrum(wavelengths=(.8,), z_indices=(4,)),
            snapshots=n.Snapshots(every=2, components=('Ex', 'Hy'))))
    numerics = n.Numerics(
        grid=n.Grid(spacing=(.1,) * 3),
        time_step_seconds=.02 * units.time_unit_seconds, steps=5,
        decay_check=n.DecayCheck(window_seconds=.04 * units.time_unit_seconds))
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
        numerics = numerics.replace(time_step_seconds=problem.units.time_unit_seconds)
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
    for field in result.snapshots.values():
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


def _yee_wave_case(phase, steps):
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
        initial=n.InitialField(E=electric, H=magnetic), bloch_phase=(phase, 0., 0.))
    numerics = n.Numerics(
        grid=n.Grid(spacing=(dz,) * 3), time_step_seconds=dt * units.time_unit_seconds,
        steps=steps, checkpoint_steps=7)
    rotation = np.exp(-1j * omega * dt * steps)
    return problem, numerics, electric, magnetic, rotation


@pytest.mark.parametrize('phase', [0., .37])
@pytest.mark.parametrize('steps', [37, 513])
def test_native_plane_wave_matches_independent_yee_dispersion(torch_device, phase, steps):
    problem, numerics, electric, magnetic, rotation = _yee_wave_case(phase, steps)
    result = n.run(problem, numerics=numerics, backend='torch', device=str(torch_device))
    np.testing.assert_allclose(result.E.detach().cpu(), electric * rotation, rtol=3e-13, atol=0.)
    np.testing.assert_allclose(result.H.detach().cpu(), magnetic * rotation, rtol=3e-13, atol=0.)


def _long_wave_budget(steps):
    # A predeclared numerical acceptance choice: preserve the existing 513-step
    # allowance per step for coherently accumulated roundoff. This is not a
    # rigorous floating-point bound, and does not loosen either existing gate.
    return 3e-13 * max(1., steps / 513)


def _long_wave_metrics(electric, magnetic, initial_e, initial_h, rotation):
    metrics = {}
    for name, actual, initial in (('E', electric, initial_e), ('H', magnetic, initial_h)):
        assert np.isfinite(actual).all()
        norm_squared = np.vdot(initial, initial).real
        coefficient = np.vdot(initial, actual) / norm_squared
        metrics[name + '_field_error'] = float(
            np.max(np.abs(actual - initial * rotation)) / np.max(np.abs(initial)))
        metrics[name + '_phase_error'] = float(abs(np.angle(coefficient * rotation.conjugate())))
        metrics[name + '_amplitude_error'] = float(abs(abs(coefficient) - 1))
        metrics[name + '_nonmodal_residual'] = float(
            np.linalg.norm(actual - coefficient * initial) / np.sqrt(norm_squared))
    # This single complex traveling eigenmode has constant E/H magnitudes.
    # This norm is not claimed to be the conserved energy of an arbitrary
    # superposition of staggered Yee states; epsilon_0/2 cancels in the ratio.
    def mode_energy(e, h):
        return 2.4 * np.vdot(e, e).real + 1.7 * Z0 ** 2 * np.vdot(h, h).real
    metrics['energy_drift'] = float(abs(
        mode_energy(electric, magnetic) / mode_energy(initial_e, initial_h) - 1))
    return metrics


def _assert_long_wave_metrics(metrics, budget):
    for name, value in metrics.items():
        allowance = 2 * budget + budget ** 2 if name == 'energy_drift' else budget
        assert np.isfinite(value) and value <= allowance, (name, value, allowance, metrics)


@pytest.mark.slow
@pytest.mark.parametrize('steps', [4097, 16385])
def test_cuda_long_native_wave_preserves_phase_amplitude_and_mode(steps, record_property):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and an NVIDIA GPU')
    problem, numerics, initial_e, initial_h, rotation = _yee_wave_case(.37, steps)
    assert problem.outputs.snapshots is None and problem.outputs.spectrum is None
    assert numerics.decay_check is None
    # No field history, diagnostic history or reverse-mode graph is requested.
    # The current engine still allocates a zero complex128 drive[steps] without
    # a source: memory is O(grid + steps), not strictly independent of duration
    # (64/256 KiB of zero drive at these two horizons).
    with torch.no_grad():
        result = n.run(problem, numerics=numerics, backend='torch', device='cuda:0')
    assert result.E.device.type == result.H.device.type == 'cuda'
    assert result.E.dtype == result.H.dtype == torch.complex128
    assert not result.E.requires_grad and not result.H.requires_grad
    assert bool(result.validity.checks['stable']) and bool(result.validity.checks['finite_fields'])
    assert not result.snapshots and result.spectrum is None and result.diagnostics['tail'] is None
    electric, magnetic = result.E.detach().cpu().numpy(), result.H.detach().cpu().numpy()
    budget = _long_wave_budget(steps)
    metrics = _long_wave_metrics(electric, magnetic, initial_e, initial_h, rotation)
    record_property('steps', steps)
    record_property('relative_budget', budget)
    for key, value in metrics.items():
        record_property(key, value)
    print(f'Native long wave: steps={steps}, phase=.37, budget={budget!r}, metrics={metrics!r}',
          flush=True)
    _assert_long_wave_metrics(metrics, budget)
    # Retain elementwise phase/amplitude checks, including exact zeros of the
    # four unexcited components, in addition to the modal diagnostics above.
    np.testing.assert_allclose(electric, initial_e * rotation, rtol=budget, atol=0.)
    np.testing.assert_allclose(magnetic, initial_h * rotation, rtol=budget, atol=0.)


@pytest.mark.parametrize('steps', [4097, 16385])
@pytest.mark.parametrize('mutation', ['phase', 'amplitude', 'orthogonal-component'])
def test_long_wave_acceptance_rejects_nonphysical_mutations(steps, mutation):
    """CPU-only controls: energy alone cannot expose a wrong modal phase."""
    _, _, initial_e, initial_h, rotation = _yee_wave_case(.37, steps)
    e, h = initial_e * rotation, initial_h * rotation
    budget = _long_wave_budget(steps)
    _assert_long_wave_metrics(_long_wave_metrics(e, h, initial_e, initial_h, rotation), budget)
    if mutation == 'phase':
        e, h = e * np.exp(1e-6j), h * np.exp(1e-6j)
        metric = 'E_phase_error'
    elif mutation == 'amplitude':
        e, h = e * (1 + 1e-6), h * (1 + 1e-6)
        metric = 'E_amplitude_error'
    else:
        e[1] = 1e-6
        metric = 'E_nonmodal_residual'
    metrics = _long_wave_metrics(e, h, initial_e, initial_h, rotation)
    assert metrics[metric] > 100 * budget
    if mutation == 'phase':
        assert metrics['energy_drift'] < budget
        assert metrics['E_amplitude_error'] < budget
    with pytest.raises(AssertionError):
        _assert_long_wave_metrics(metrics, budget)
