"""Complex Bloch/CPML observations: finite-trajectory reverse-mode AD gates.

NumPy phase-zero and phase-pi/2 runs provide a real Gram matrix for each
complex linear observation. This independently differentiates superposition,
not Maxwell discretisation. Short DFT windows and four-cell PML padding do not
establish optical accuracy or a reflection bound.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.units import Z0


UNITS = Units(length='um')
STEPS = 83
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
CASES = [('cpu', None), ('cuda:0', None), ('cuda:0', 7)]
IDS = ['cpu-plain', 'cuda-plain', 'cuda-k7']


def problem(amplitude=1., phase=0., epsilon=2.25):
    pulse = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15 / UNITS.time_seconds)
    peak = np.sqrt(2 * np.log(1 / pulse.cutoff)) / (2 * np.pi * pulse.frequency_sigma_hz)
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.3), theta=np.pi / 6,
        phi=np.deg2rad(35.), polarization='p', amplitude=amplitude, phase=phase,
        pulse=pulse, delay_seconds=-peak + .12 * UNITS.time_seconds)
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.25, .2, 1.2)), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries(**fdtd.Bloch.from_source(source, units=UNITS), z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.1, .1)),
                                   material=Material(epsilon=epsilon)),)),
        sources=(source,), outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.TimeSeriesMonitor('e', at=(.025, 0., -.25), component='Ey'),
            fdtd.TimeSeriesMonitor('h', at=(.025, 0., -.25), component='Hx'),
            fdtd.FieldMonitor('field', plane=fdtd.Plane(z=-.25), wavelengths=(1.,),
                       components=COMPONENTS),
            fdtd.FluxMonitor('flux', plane=fdtd.Plane(z=-.25), wavelengths=(1.,)))))


def numerics(block=None):
    # 83 = 11*7+6: a nonempty tail with enough transit time to reach the
    # downstream monitor and the open padding, without a long spectral run.
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), courant=.5,
        duration=fdtd.Duration(steps=STEPS), checkpoint_steps=block,
        absorber=fdtd.PML(cells=4, wavelength=1., allow_thin=True))


def groups(result):
    """Fixed SI rescalings only; each tuple contains complex linear outputs."""
    source = result.sources['pw']
    return (
        (result.final_fields.values['E'],),
        (Z0 * result.final_fields.values['H'],),
        (result['e'].values, Z0 * result['h'].values),
        tuple(result['field'].values[c] * (Z0 if c[0] == 'H' else 1.) / UNITS.time_seconds
              for c in COMPONENTS),
        (source['incident'], Z0 * source['incident_H']),
    )


def norm2(values):
    return sum((v.real ** 2 + v.imag ** 2).sum() for v in values)


def losses(result):
    record = norm2(groups(result)[2])
    field = norm2(groups(result)[3])
    flux = result['flux'].energy_density.sum() / (
        2 * UNITS.time_seconds ** 2 * UNITS.length_metres ** 2 / Z0)
    return record, field, flux


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def gram_matrices():
    basis = [groups(fdtd.run(problem(phase=p), numerics=numerics())) for p in (0., np.pi / 2)]
    return [np.array([[sum(np.vdot(x, y).real for x, y in zip(a[g], b[g]))
                       for b in basis] for a in basis]) for g in range(5)]


def analytic(amplitude, phase, gram):
    u = np.array([np.cos(phase), np.sin(phase)])
    du = np.array([-np.sin(phase), np.cos(phase)])
    q = u @ gram @ u
    dq = 2 * du @ gram @ u
    ddq = 2 * (du @ gram @ du - q)
    value = amplitude ** 2 * q
    gradient = np.array([2 * amplitude * q, amplitude ** 2 * dq])
    hessian = np.array([[2 * q, 2 * amplitude * dq],
                        [2 * amplitude * dq, amplitude ** 2 * ddq]])
    assert value > 0 and np.all(np.abs(gradient) > 1e-4 * value)
    assert np.all(np.abs(hessian) > 1e-4 * value)
    return value, gradient, hessian


def runtime(device):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    return torch


def ownership(torch, leaves, originals, derivatives):
    for leaf, original, derivative in zip(leaves, originals, derivatives):
        assert torch.equal(leaf, original) and leaf.grad is None
        assert derivative.device == leaf.device and derivative.dtype == leaf.dtype


def snapshot(result):
    # Snapshot the public buffers themselves, not rescaled temporary tensors.
    source = result.sources['pw']
    values = (*result.final_fields.values.values(), result['e'].values, result['h'].values,
              *result['field'].values.values(), source['incident'], source['incident_H'],
              result['flux'].energy_density)
    return [(v, v.detach().clone()) for v in values]


@pytest.mark.parametrize('device,block', CASES, ids=IDS)
def test_complex_bloch_observation_amplitude_phase_hessians(device, block, record_property):
    torch = runtime(device)
    grams = gram_matrices()
    prepared = fdtd.prepare(problem(), numerics=numerics(block), backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == STEPS
    assert prepared.contract['resolved_checkpoint_steps'] == block
    old = None
    for point, (amplitude, phase) in enumerate(((1.2, .37), (.9, .61))):
        leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
                  torch.tensor(phase, dtype=torch.float64, device=device, requires_grad=True))
        originals = tuple(v.detach().clone() for v in leaves)
        live = problem(*leaves)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        assert bool(result.validity.checks['runtime_stability'])
        actual_groups = groups(result)
        for index, (values, gram) in enumerate(zip(actual_groups, grams)):
            assert all(v.device == torch.device(device) and v.dtype == torch.complex128 for v in values)
            assert all(bool(torch.isfinite(v).all()) for v in values)
            # A polarization may make an individual component vanish (e.g. Hz).
            # Each observation group must nevertheless contain a complex signal.
            assert max(float(v.imag.abs().max().detach().cpu()) for v in values) > 1e-8
            loss = norm2(values)
            gradients = torch.autograd.grad(loss, leaves, create_graph=True)
            rows = [torch.autograd.grad(g, leaves, retain_graph=index < 4 or row == 0)
                    for row, g in enumerate(gradients)]
            actual = (float(host(loss)), np.array([float(host(g)) for g in gradients]),
                      np.array([[float(host(g)) for g in row] for row in rows]))
            for got, expected in zip(actual, analytic(amplitude, phase, gram)):
                np.testing.assert_allclose(got, expected, rtol=2e-12, atol=0.)
            for derivatives in (gradients, *rows):
                ownership(torch, leaves, originals, derivatives)
            record_property(f'point_{point}_group_{index}_value_gradient_hessian',
                            repr(tuple(np.asarray(v).tolist() for v in actual)))
        if old is not None:
            for value, saved in old:
                assert torch.equal(value, saved)
        old = snapshot(result)


@pytest.mark.parametrize('device,block', CASES, ids=IDS)
def test_complex_bloch_record_field_flux_material_phase_gradients(device, block, record_property):
    torch = runtime(device)
    amplitude, phase, epsilon = 1.2, .37, 2.25
    leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
              torch.tensor(phase, dtype=torch.float64, device=device, requires_grad=True),
              torch.tensor(epsilon, dtype=torch.float64, device=device, requires_grad=True))
    originals = tuple(v.detach().clone() for v in leaves)
    prepared = fdtd.prepare(problem(), numerics=numerics(block), backend='torch', device=device)
    result = prepared.run(problem(*leaves))
    assert bool(result.validity.checks['runtime_stability'])
    values, derivatives = [], []
    for index, loss in enumerate(losses(result)):
        gradients = torch.autograd.grad(loss, leaves, retain_graph=index < 2)
        ownership(torch, leaves, originals, gradients)
        values.append(float(host(loss)))
        derivatives.append([float(host(g)) for g in gradients])
    values, derivatives = np.asarray(values), np.asarray(derivatives)
    assert np.isfinite(values).all() and np.isfinite(derivatives).all()
    assert np.all(np.abs(values) > 1e-8) and np.all(np.abs(derivatives) > 1e-8)
    np.testing.assert_allclose(derivatives[:, 0], 2 * values / amplitude, rtol=2e-12, atol=0.)

    def numpy_values(p=phase, e=epsilon):
        reference = fdtd.run(problem(amplitude, p, e), numerics=numerics())
        assert bool(reference.validity.checks['runtime_stability'])
        return np.asarray(losses(reference))

    np.testing.assert_allclose(values, numpy_values(), rtol=5e-13, atol=0.)
    for column, center in ((1, phase), (2, epsilon)):
        for step in (center * 1e-4, center * 5e-5):
            plus = numpy_values(p=phase + step) if column == 1 else numpy_values(e=epsilon + step)
            minus = numpy_values(p=phase - step) if column == 1 else numpy_values(e=epsilon - step)
            secant = (plus - minus) / (2 * step)
            assert np.isfinite(secant).all() and np.all(np.abs(secant) > 1e-8)
            np.testing.assert_allclose(derivatives[:, column], secant, rtol=2e-6, atol=0.)
    record_property('record_field_flux_values', repr(values.tolist()))
    record_property('amplitude_phase_epsilon_gradients', repr(derivatives.tolist()))
