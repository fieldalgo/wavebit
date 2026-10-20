"""ADE TimeSeriesMonitor-objective first and second derivatives.

Each of the six sampled components uses its own physical Yee position and
time level. A fixed weighted square plus a fixed signed linear functional
tests the recorded history, including signs, with final-field output disabled.
The exact oracle is a matrix-power Frechet derivative, not a finite difference.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
from wavebit import fdtd
from wavebit.units import Z0

from tests.fdtd._homogeneous_ade_reference import UNITS, fourier_mode, host, numerics, probe_at
from tests.fdtd.test_general_ade_ad import CELLS, POINTS, STEPS, live_problem, oracle_states


LABELS = tuple(field+'xyz'[c] for field in ('E', 'H') for c in range(3))
QUADRATIC = np.array([.3, .7, 1.1, .5, .9, 1.3])[:, None] * (
    1+np.arange(1, STEPS+1)[None, :]/STEPS)
LINEAR = .17*np.array([.2, -.3, .5, -.7, .11, .13])[:, None] * (
    1+.2*np.sin(np.arange(1, STEPS+1)[None, :]))


def record_problem(kind, amplitude, parameter, electric, magnetic):
    problem = live_problem(kind, amplitude, parameter, electric, magnetic)
    return replace(problem, outputs=replace(problem.outputs, final_fields=False))


def expected(kind, amplitude, parameter, dt, discrete_k, initial, spatial):
    # State after step n contains E(n dt) and H((n-.5) dt); these are exactly
    # the two time levels stored in that step's public TimeSeriesMonitor rows.
    histories = [oracle_states(kind, parameter, dt, discrete_k, initial, steps=n)
                 for n in range(1, STEPS+1)]
    vectors = []
    for derivative in range(3):
        vectors.append(np.stack([
            np.array([(state[derivative][offset+c] *
                       spatial(field, c, probe_at(CELLS, field, c))).real
                      for state in histories])
            for field, offset in (('E', 0), ('H', 3)) for c in range(3)
        ]))
    values, first, second = vectors
    assert np.all(np.max(np.abs(values), axis=1) > .01)
    q = np.sum(QUADRATIC*values**2)
    qp = 2*np.sum(QUADRATIC*values*first)
    qpp = 2*np.sum(QUADRATIC*(first**2+values*second))
    ell, ellp, ellpp = (np.sum(LINEAR*v) for v in vectors)
    loss = amplitude**2*q+amplitude*ell
    gradient = np.array([2*amplitude*q+ell, amplitude**2*qp+amplitude*ellp])
    hessian = np.array([[2*q, 2*amplitude*qp+ellp],
                        [2*amplitude*qp+ellp, amplitude**2*qpp+amplitude*ellpp]])
    assert abs(ell) > 1e-4*q  # The signed contribution must not be a vacuous zero.
    assert loss > .01 and np.all(np.abs(gradient) > 1e-5*loss)
    assert np.all(np.abs(hessian) > 1e-5*loss)
    return loss, gradient, hessian


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_homogeneous_record_amplitude_material_full_hessian(device, block, kind, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    _, discrete_k, initial, electric, magnetic, spatial = fourier_mode(CELLS, 0)
    patterns = electric.copy(), magnetic.copy()
    dt = .5/CELLS/math.sqrt(3)*UNITS.time_unit_seconds
    policy = numerics(CELLS, STEPS).replace(checkpoint_steps=block)
    square_weights = torch.as_tensor(QUADRATIC, dtype=torch.float64, device=device)
    linear_weights = torch.as_tensor(LINEAR, dtype=torch.float64, device=device)
    old = None
    prepared = fdtd.prepare(record_problem(kind, 1., 1., electric, magnetic),
                            numerics=policy, backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == STEPS
    assert prepared.contract['resolved_checkpoint_steps'] == block
    np.testing.assert_allclose(prepared.contract['resolved_time_step_seconds'], dt,
                               rtol=2e-15, atol=0.)
    for point, (amplitude, parameter) in enumerate(POINTS):
        leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
                  torch.tensor(parameter, dtype=torch.float64, device=device, requires_grad=True))
        originals = tuple(value.detach().clone() for value in leaves)
        live = record_problem(kind, *leaves, electric, magnetic)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        assert result.final_fields is None
        assert bool(result.validity.checks['runtime_stability'])
        for label in LABELS:
            values = result[label].values
            assert values.dtype == torch.float64 and values.device == torch.device(device)
            assert bool(torch.isfinite(values).all())
            shift = 1. if label[0] == 'E' else .5
            np.testing.assert_allclose(host(result[label].times_seconds),
                (np.arange(STEPS)+shift)*dt, rtol=2e-14, atol=0.)
        values = torch.stack([result[label].values*(Z0 if label[0] == 'H' else 1.)
                              for label in LABELS])
        loss = (square_weights*values**2+linear_weights*values).sum()
        gradients = torch.autograd.grad(loss, leaves, create_graph=True)
        rows = [torch.autograd.grad(g, leaves, retain_graph=row == 0)
                for row, g in enumerate(gradients)]
        actual = (float(host(loss)), np.array([float(host(g)) for g in gradients]),
                  np.array([[float(host(g)) for g in row] for row in rows]))
        reference = expected(kind, amplitude, parameter, dt, discrete_k, initial, spatial)
        for value, wanted in zip(actual, reference):
            np.testing.assert_allclose(value, wanted, rtol=2e-12, atol=0.)
        for derivatives in (gradients, *rows):
            for leaf, original, derivative in zip(leaves, originals, derivatives):
                assert leaf.is_leaf and leaf.grad is None and torch.equal(leaf, original)
                assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
        if old is not None:
            for value, saved in old:
                assert torch.equal(value, saved)
        old = tuple((result[label].values, result[label].values.detach().clone())
                    for label in LABELS)
        np.testing.assert_array_equal(electric, patterns[0])
        np.testing.assert_array_equal(magnetic, patterns[1])
        record_property(f'point_{point}_amplitude_parameter', repr((amplitude, parameter)))
        record_property(f'point_{point}_value_gradient_hessian',
                        repr(tuple(np.asarray(v).tolist() for v in actual)))
        record_property(f'point_{point}_oracle_value_gradient_hessian',
                        repr(tuple(np.asarray(v).tolist() for v in reference)))
    record_property('steps_checkpoint_tail', repr((STEPS, block, STEPS % block if block else None)))
