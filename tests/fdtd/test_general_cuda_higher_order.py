"""Bounded CUDA double backward against the exact SI conductor recurrence.

This covers an eager Hessian of a uniform, periodic conductor with final-field
observations. It does not establish arbitrary geometry Hessians, torch.func
transforms, CPML higher derivatives, or other material/source families.
"""
import numpy as np
import pytest

from wavebit import fdtd
from wavebit.units import EPSILON0

from tests.fdtd.test_general_cuda import closed_problem, fixed_numerics


STEPS = 17
EPSILON = 2.25
RTOL = 2e-12  # Existing exact conductor first-derivative gate; zero absolute allowance.


def conductor_quadratic_derivatives(amplitude, conductivity, dt_seconds):
    """J, gradient and Hessian in (amplitude, SI conductivity) order.

    Spatially uniform periodic fields have zero curl. Trapezoidal conductivity
    therefore gives Ey_N=A*q**N, q=(1-sigma*f)/(1+sigma*f),
    f=dt/(2*epsilon_0*epsilon_r). Differentiate that scalar rational recurrence,
    independently of Wavebit's lowering, stepping and autodiff implementation.
    """
    f = dt_seconds / (2 * EPSILON0 * EPSILON)
    denominator = 1 + conductivity * f
    q = (1 - conductivity * f) / denominator
    dq = -2 * f / denominator ** 2
    ddq = 4 * f ** 2 / denominator ** 3
    m = 2 * STEPS
    value = amplitude ** 2 * q ** m
    gradient = np.asarray([2 * amplitude * q ** m,
                           amplitude ** 2 * m * q ** (m - 1) * dq])
    mixed = 2 * amplitude * m * q ** (m - 1) * dq
    hessian = np.asarray([
        [2 * q ** m, mixed],
        [mixed, amplitude ** 2 * m * (
            (m - 1) * q ** (m - 2) * dq ** 2 + q ** (m - 1) * ddq)],
    ])
    assert 0 < q < 1
    assert np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(hessian).all()
    assert value > 0 and np.all(gradient != 0) and np.all(hessian != 0)
    return value, gradient, hessian


@pytest.fixture
def cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch


def solve_hessian(torch, prepared, amplitude_value, conductivity_value):
    # Exercise a real host/device graph boundary: returned derivatives belong
    # to the original leaves, including both orders of the mixed derivative.
    amplitude = torch.tensor(amplitude_value, dtype=torch.float64, device='cpu',
                             requires_grad=True)
    conductivity = torch.tensor(conductivity_value, dtype=torch.float64, device='cuda:0',
                                requires_grad=True)
    leaves = (amplitude, conductivity)
    live = closed_problem(epsilon=EPSILON, conductivity=conductivity, amplitude=amplitude)
    assert prepared.validate(live).ok
    result = prepared.run(live)
    assert bool(result.validity.checks['runtime_stability'])
    electric = result.final_fields.values['E']
    assert electric.device == conductivity.device and electric.dtype == torch.float64
    loss = (electric[1] ** 2).mean()
    gradient = torch.autograd.grad(loss, leaves, create_graph=True)
    rows = tuple(torch.autograd.grad(g, leaves, retain_graph=index == 0)
                 for index, g in enumerate(gradient))
    for derivatives in (gradient,) + rows:
        for derivative, leaf in zip(derivatives, leaves):
            assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
            assert leaf.is_leaf and leaf.requires_grad and leaf.grad is None
            assert bool(torch.isfinite(derivative))
    host = lambda value: float(value.detach().cpu())
    actual = (host(loss), np.asarray([host(g) for g in gradient]),
              np.asarray([[host(g) for g in row] for row in rows]))
    expected = conductor_quadratic_derivatives(
        amplitude_value, conductivity_value, result.provenance['time_step_seconds'])
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.)
        assert np.all(np.asarray(got) != 0)
    np.testing.assert_allclose(actual[2][0, 1], actual[2][1, 0], rtol=RTOL, atol=0.)
    assert float(amplitude.detach()) == amplitude_value
    assert host(conductivity) == conductivity_value
    return result, actual


@pytest.mark.parametrize('block', [None, 3], ids=['plain', 'checkpoint-3'])
def test_cuda_conductor_exact_hessian_across_device_leaves_and_prepared_reuse(cuda, block):
    numerics = fixed_numerics(STEPS).replace(checkpoint_steps=block)
    base = closed_problem(epsilon=EPSILON)
    prepared = fdtd.prepare(base, numerics=numerics, backend='torch', device='cuda:0')
    assert prepared.contract['resolved_checkpoint_steps'] == block
    assert prepared.contract['resolved_steps'] == STEPS
    if block is not None:
        assert divmod(STEPS, block) == (5, 2)

    first, _ = solve_hessian(cuda, prepared, 1.3, 600.)
    saved_electric = first.final_fields.values['E'].detach().clone()
    reused, reused_derivatives = solve_hessian(cuda, prepared, .9, 850.)
    fresh = fdtd.prepare(base, numerics=numerics, backend='torch', device='cuda:0')
    independent, fresh_derivatives = solve_hessian(cuda, fresh, .9, 850.)
    for got, want in zip(reused_derivatives, fresh_derivatives):
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.)
    assert cuda.equal(reused.final_fields.values['E'], independent.final_fields.values['E'])
    assert cuda.equal(first.final_fields.values['E'], saved_electric)
    assert not cuda.equal(reused.final_fields.values['E'], saved_electric)
