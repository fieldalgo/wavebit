"""Static Bloch material/amplitude derivatives from an exact discrete mode.

The initial H half-step is fixed independently of epsilon. This makes the
17-step electric field a cosine for every live epsilon, without differentiating
an eigenmode initialization. This gate does not establish phase derivatives,
PML accuracy, or Bloch/CurrentDipole support.
"""
import math

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.units import Z0


UNITS = Units(length='um')
CELLS = 16
DX = .05
DT = .5 * DX / math.sqrt(3.)  # Geometry-time units; c=1 in the discrete equations.
STEPS = 17
MU = 1.3
PHASE = np.array([.7, -1.1, .4])
K = (2 * np.pi + PHASE) / (CELLS * DX)
DISCRETE_K = 2 * np.sin(K * DX / 2) / DX
S = float(np.dot(DISCRETE_K, DISCRETE_K) / 4)
POINTS = ((.8, 1.7), (1.3, 2.4))
RTOL = 2e-12


def initial_patterns():
    # Explicit Yee coordinates: E_c is half-shifted along c, H_c on the two
    # other axes. Arrays are component,z,y,x; no plan/kernel helper supplies them.
    polarization = np.array([1., .5,
        -(DISCRETE_K[0] + .5 * DISCRETE_K[1]) / DISCRETE_K[2]])
    polarization /= np.linalg.norm(polarization)
    assert abs(np.dot(DISCRETE_K, polarization)) < 1e-14
    curl_amplitude = 1j * np.cross(DISCRETE_K, polarization)
    arrays = {}
    for field in ('E', 'H'):
        components = []
        for component in range(3):
            axes = [(-CELLS / 2 + np.arange(CELLS) + .5 * (
                axis == component if field == 'E' else axis != component)) * DX
                for axis in range(3)]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            pattern = np.exp(1j * (K[0] * x + K[1] * y + K[2] * z))
            factor = (polarization[component] if field == 'E'
                      else DT * curl_amplitude[component] / (2 * MU * Z0))
            components.append(factor * pattern)
        arrays[field] = np.stack(components)
    return arrays['E'], arrays['H']


def problem(electric, magnetic, amplitude=1., epsilon=2.25):
    return fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(CELLS * DX,) * 3),
        background=Material(epsilon=epsilon, mu=MU),
        boundaries=fdtd.Boundaries(**{
            axis: fdtd.Bloch(phase=float(phase)) for axis, phase in zip('xyz', PHASE)})),
        initial=fdtd.InitialField(E=electric, H=magnetic, amplitude=amplitude),
        outputs=fdtd.Outputs(final_fields=True))


def numerics(block):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=STEPS), courant=.5, checkpoint_steps=block)


def analytic(amplitude, epsilon, electric):
    """L, dL/dA and dL/depsilon with fixed initial arrays and fixed timestep.

    Kd.p=0 gives E_1=(1-2 dt^2 S/(epsilon mu))*E_0. Hence
    E_n=A E_0 cos(n omega dt), omega=2/dt asin(dt sqrt(S/(epsilon mu))).
    The objective is the component-and-cell mean of the complex squared modulus.
    """
    ratio = DT * math.sqrt(S / (epsilon * MU))
    assert 0 < ratio < 1
    omega = 2 / DT * math.asin(ratio)
    phase = STEPS * omega * DT
    assert abs(math.cos(phase)) > .1 and abs(math.sin(2 * phase)) > .1
    coefficient = float(np.mean(np.abs(electric) ** 2))
    domega = -math.sqrt(S) / (epsilon**1.5 * math.sqrt(MU)
                                  * math.sqrt(1 - ratio**2))
    value = amplitude**2 * coefficient * math.cos(phase)**2
    gradient = np.array([2 * value / amplitude,
        -amplitude**2 * coefficient * STEPS * DT * math.sin(2 * phase) * domega])
    assert value > 0 and np.isfinite(gradient).all()
    assert np.all(np.abs(gradient) > 1e-3 * value)
    return value, gradient, omega


@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_static_bloch_material_and_amplitude_derivatives(device, block, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires an actual CUDA device')
    electric, magnetic = initial_patterns()
    saved_inputs = (electric.copy(), magnetic.copy())
    prepared = fdtd.prepare(problem(electric, magnetic), numerics=numerics(block),
                            backend='torch', device=device)
    record_property('static_bloch_phases', PHASE.tolist())
    record_property('steps', STEPS)
    record_property('checkpoint_steps', block)
    old = None
    for index, (amplitude_value, epsilon_value) in enumerate(POINTS):
        # The same static initial arrays are reused at both permittivities;
        # only these two leaves change. Amplitude intentionally starts on CPU.
        amplitude = torch.tensor(amplitude_value, dtype=torch.float64, device='cpu',
                                 requires_grad=True)
        epsilon = torch.tensor(epsilon_value, dtype=torch.float64, device=device,
                               requires_grad=True)
        leaves = (amplitude, epsilon)
        snapshots = tuple(value.detach().clone() for value in leaves)
        live = problem(electric, magnetic, amplitude, epsilon)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        assert bool(result.validity.checks['runtime_stability'])
        np.testing.assert_allclose(result.provenance['time_step_seconds'],
                                   DT * UNITS.time_unit_seconds, rtol=2e-15, atol=0.)
        value = result.final_fields.values['E'].abs().square().mean()
        gradients = torch.autograd.grad(value, leaves)
        expected, exact_gradients, omega = analytic(amplitude_value, epsilon_value, electric)
        np.testing.assert_allclose(float(value.detach().cpu()), expected, rtol=RTOL, atol=0.)
        actual_gradients = np.array([float(g.detach().cpu()) for g in gradients])
        np.testing.assert_allclose(actual_gradients, exact_gradients, rtol=RTOL, atol=0.)
        record_property(f'point_{index}_value', expected)
        record_property(f'point_{index}_gradients', actual_gradients.tolist())
        for leaf, gradient, snapshot in zip(leaves, gradients, snapshots):
            assert leaf.is_leaf and leaf.requires_grad and leaf.grad is None
            assert gradient.device == leaf.device and gradient.dtype == leaf.dtype
            assert bool(torch.isfinite(gradient)) and float(gradient.detach().cpu()) != 0.
            assert torch.equal(leaf, snapshot)
        # Full complex fields catch phase, Yee location and half-step errors
        # which a spatial squared-modulus objective could otherwise hide.
        expected_fields = {
            'E': amplitude_value * electric * math.cos(STEPS * omega * DT),
            'H': -amplitude_value * magnetic * math.sin((STEPS - .5) * omega * DT)
                 / math.sin(omega * DT / 2),
        }
        for field in ('E', 'H'):
            actual = result.final_fields.values[field]
            assert actual.device == torch.device(device) and actual.dtype == torch.complex128
            assert bool(torch.isfinite(actual).all()) and float(actual.imag.abs().max().detach().cpu()) > 1e-4
            np.testing.assert_allclose(actual.detach().cpu().numpy(), expected_fields[field],
                                       rtol=RTOL, atol=0.)
            if old is not None:
                assert torch.equal(old[0].final_fields.values[field], old[1][field])
        for original, snapshot in zip((electric, magnetic), saved_inputs):
            np.testing.assert_array_equal(original, snapshot)
        old = (result, {field: result.final_fields.values[field].detach().clone()
                        for field in ('E', 'H')})
