"""Amplitude/phase Hessian from linear superposition, independently of autodiff.

The NumPy basis uses the same spatial discretisation, so this is a derivative
oracle, not an independent Maxwell/PML accuracy reference. Thin padding and a
short trajectory deliberately limit cost. This qualifies eager reverse-mode
derivatives in the stated finite trajectories.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.units import Z0


UNITS = Units(length='um')
STEPS = 41
DX = .05
MOMENT = 1e-16
RTOL = 2e-12


def problem(amplitude=1., phase=0., kind='electric'):
    pulse = fdtd.GaussianPulse(wavelength=2., frequency_sigma_hz=.08 / UNITS.time_unit_seconds)
    peak = np.sqrt(2 * np.log(1 / pulse.cutoff)) / (2 * np.pi * pulse.frequency_sigma_hz)
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.6,) * 3), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries(x=fdtd.Open(), y=fdtd.Open(), z=fdtd.Open())),
        sources=(fdtd.CurrentDipole('d', center=(-.031, .017, -.10),
            component='Ey' if kind == 'electric' else 'Hy', kind=kind,
            pulse=pulse, current_moment=MOMENT * amplitude,
            phase=phase, delay_seconds=-peak + .12 * UNITS.time_unit_seconds),),
        outputs=fdtd.Outputs(final_fields=True))


def numerics(block=None):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=STEPS), checkpoint_steps=block,
        absorber=fdtd.PML(cells=4, wavelength=2., allow_thin=True))


def vector(result, kind):
    fields = result.final_fields.values
    return (fields['E'], Z0 * fields['H'],
            (Z0 if kind == 'magnetic' else 1.) * result.sources['d']['field'])


def objective(result, kind):
    return sum((v * v).sum() for v in vector(result, kind))


def gram_matrix(kind):
    basis = [vector(fdtd.run(problem(phase=phase, kind=kind), numerics=numerics()), kind)
             for phase in (0., np.pi / 2)]
    return np.array([[sum(np.sum(x * y) for x, y in zip(a, b)) for b in basis] for a in basis])


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


@pytest.mark.parametrize('kind', ['electric', 'magnetic'])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_public_cpml_amplitude_phase_hessian(device, block, kind):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    gram = gram_matrix(kind)
    prepared = fdtd.prepare(problem(kind=kind), numerics=numerics(block),
                            backend='torch', device=device)
    old = None
    for amplitude, phase in ((1.2, .37), (.9, .61)):
        leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
                  torch.tensor(phase, dtype=torch.float64, device=device, requires_grad=True))
        result = prepared.run(problem(*leaves, kind=kind))
        assert bool(result.validity.checks['runtime_stability'])
        loss = objective(result, kind)
        gradients = torch.autograd.grad(loss, leaves, create_graph=True)
        rows = [torch.autograd.grad(g, leaves, retain_graph=i == 0)
                for i, g in enumerate(gradients)]
        actual = (float(loss.detach().cpu()),
                  np.array([float(g.detach().cpu()) for g in gradients]),
                  np.array([[float(g.detach().cpu()) for g in row] for row in rows]))
        for got, want in zip(actual, analytic(amplitude, phase, gram)):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.)
        for derivs in (gradients, *rows):
            for leaf, derivative in zip(leaves, derivs):
                assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
                assert leaf.grad is None
        if old is not None:
            assert torch.equal(old[0].final_fields.values['E'], old[1])
            assert torch.equal(old[0].sources['d']['field'], old[2])
        old = (result, result.final_fields.values['E'].detach().clone(),
               result.sources['d']['field'].detach().clone())
