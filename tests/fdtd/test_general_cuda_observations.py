"""General-scene TimeSeriesMonitor/FieldMonitor/FluxMonitor derivatives on CPU and CUDA."""
import numpy as np
import pytest

from wavebit import fdtd
from wavebit.units import Z0

from tests.fdtd.test_general_cuda import fixed_numerics, slab_problem


@pytest.fixture(params=['cpu', 'cuda:0'])
def torch_device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


def observable_losses(result, units):
    """Undo only fixed SI scale factors, retaining each observable's live graph."""
    record = (result['e'].values ** 2).mean()
    fields = result['field'].values
    # E and Z0 H share field units. Divide the Fourier integrals by the fixed
    # geometry time unit, not a live field norm, to keep the derivative unchanged.
    field = sum(((fields[c].real ** 2 + fields[c].imag ** 2) * scale ** 2).sum()
                for c, scale in (('Ey', 1.), ('Hx', Z0))) / units.time_unit_seconds ** 2
    # The general solver returns a two-sided pulse spectral energy. This fixed
    # factor restores its internal signed plane flux without normalizing it by
    # a source amplitude or detaching the field-dependent result.
    flux = result['flux'].spectral_energy.sum() / (
        2 * units.time_unit_seconds ** 2 * units.length_unit_meters ** 2 / Z0)
    return record, field, flux


def test_record_field_flux_material_and_source_derivatives(torch_device):
    torch, device = torch_device
    # The public general-scene API currently uses a plain scan (schedule=None).
    # Keep a non-round duration so a later explicit scheduling regression can
    # also exercise a tail without changing this physical observable case.
    numerics = fixed_numerics(161)
    problem = slab_problem()
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch', device=device)
    epsilon = torch.tensor(2.25, dtype=torch.float64, device=device, requires_grad=True)
    amplitude = torch.tensor(1.2, dtype=torch.float64, device=device, requires_grad=True)
    result = prepared.run(slab_problem(epsilon=epsilon, amplitude=amplitude))
    assert bool(result.validity.checks['runtime_stability'])
    losses = observable_losses(result, problem.units)
    values, derivatives = [], []
    for index, loss in enumerate(losses):
        assert loss.device == epsilon.device
        pair = torch.autograd.grad(loss, (epsilon, amplitude), retain_graph=index < len(losses) - 1)
        assert all(value.device == epsilon.device for value in pair)
        values.append(float(loss.detach().cpu()))
        derivatives.append([float(value.detach().cpu()) for value in pair])
    values, derivatives = np.asarray(values), np.asarray(derivatives)
    assert np.isfinite(values).all() and np.isfinite(derivatives).all()
    assert np.all(np.abs(values) > 1e-12)
    # Every output used here is quadratic in source amplitude in linear Maxwell
    # evolution with zero initial fields, including the signed spectral flux.
    np.testing.assert_allclose(derivatives[:, 1], 2 * values / 1.2, rtol=2e-11, atol=0.)

    def cpu_values(material_epsilon):
        live = slab_problem(epsilon=material_epsilon, amplitude=1.2)
        answer = fdtd.run(live, numerics=numerics)
        return np.asarray(observable_losses(answer, problem.units))

    np.testing.assert_allclose(values, cpu_values(2.25), rtol=5e-13, atol=0.)
    for step in (2.25e-4, 1.125e-4):
        secant = (cpu_values(2.25 + step) - cpu_values(2.25 - step)) / (2 * step)
        assert np.isfinite(secant).all() and np.all(np.abs(secant) > 1e-12)
        np.testing.assert_allclose(derivatives[:, 0], secant, rtol=2e-6, atol=0.)
