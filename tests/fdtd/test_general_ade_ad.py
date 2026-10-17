"""Homogeneous ADE amplitude/material Hessian qualification.

The oracle differentiates a dense SI state matrix by a matrix-power Frechet
construction. It never differentiates Wavebit's coefficients or its time loop.
Only the existing rest-polarization startup is qualified; this is not a
continuum-accuracy, interface, source, PML, or general dispersive-media accuracy claim.
"""
import math

import numpy as np
import pytest
from wavebit import fdtd
from wavebit.materials import DispersiveMaterial, Drude, Lorentz
from wavebit.units import Z0

from tests.fdtd._homogeneous_ade_reference import (
    EPSILON,
    MU,
    UNITS,
    fourier_mode,
    host,
    make_problem,
    mesh,
    numerics,
    pole_parameters,
    transition,
)

CELLS, STEPS = 12, 17
POINTS = ((1.2, .8), (.9, 1.3))


def live_problem(kind, amplitude, parameter, electric, magnetic):
    """p multiplies SI Lorentz strength or SI Drude plasma frequency."""
    omega0, gamma, strength = pole_parameters(kind, True)
    pole = (Drude(omega_p=math.sqrt(strength)*parameter, gamma=gamma)
            if kind == 'drude' else
            Lorentz(omega_0=omega0, gamma=gamma, strength=strength*parameter))
    material = DispersiveMaterial(epsilon_inf=EPSILON, mu=MU, poles=(pole,))
    base = make_problem(kind, True, electric, magnetic, CELLS)
    structure = base.scene.structures[0].replace(material=material)
    return base.replace(scene=base.scene.replace(structures=(structure,)),
                        initial=base.initial.replace(amplitude=amplitude))


def oracle_states(kind, parameter, dt, discrete_k, initial, steps=STEPS):
    """Return x, dx/dp, d2x/dp2 by the exact derivative of M**steps."""
    omega0, gamma, strength0 = pole_parameters(kind, True)
    strength = strength0*(parameter**2 if kind == 'drude' else parameter)
    matrix = transition(dt, discrete_k, omega0, gamma, strength)
    drive0 = strength0*dt**2/(1+gamma*dt/2)
    dp = 2*parameter*drive0 if kind == 'drude' else drive0
    dpp = 2*drive0 if kind == 'drude' else 0.
    first, second = np.zeros_like(matrix), np.zeros_like(matrix)
    # Only Q_new and delta_new depend on the material drive. Their drive field
    # is E_new, whose entire matrix row is independent of these pole leaves.
    for rows in (slice(6, 9), slice(12, 15)):
        first[rows] = dp*matrix[:3]
        second[rows] = dpp*matrix[:3]
    block = np.zeros((45, 45), dtype=complex)
    for start in (0, 15, 30):
        block[start:start+15, start:start+15] = matrix
    block[:15, 15:30] = 2*first
    block[:15, 30:] = second
    block[15:30, 30:] = first
    vector = np.zeros(45, dtype=complex)
    vector[30:] = initial
    second_state, first_state, state = np.split(np.linalg.matrix_power(block, steps)@vector, 3)
    return state, first_state, second_state


def reconstruct(state, spatial):
    """Real E and Z0*H, evaluated at explicit physical Yee sample locations."""
    return np.concatenate([
        np.stack([(state[offset+c]*spatial(field, c, mesh(CELLS, field, c))).real
                  for c in range(3)]).ravel()
        for field, offset in (('E', 0), ('H', 3))
    ])


def expected(kind, amplitude, parameter, dt, discrete_k, initial, spatial):
    field, first, second = [reconstruct(state, spatial) for state in
                           oracle_states(kind, parameter, dt, discrete_k, initial)]
    q = np.dot(field, field)
    qp = 2*np.dot(field, first)
    qpp = 2*(np.dot(first, first)+np.dot(field, second))
    loss = amplitude**2*q
    gradient = np.array([2*amplitude*q, amplitude**2*qp])
    hessian = np.array([[2*q, 2*amplitude*qp],
                        [2*amplitude*qp, amplitude**2*qpp]])
    assert loss > 1. and np.all(np.abs(gradient) > 1e-5*loss)
    assert np.all(np.abs(hessian) > 1e-5*loss)
    return loss, gradient, hessian


def raw_outputs(result):
    return tuple(result.final_fields.values.values()) + tuple(
        result[field+'xyz'[c]].values for field in ('E', 'H') for c in range(3))


@pytest.mark.parametrize('kind', ['lorentz', 'drude'])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_homogeneous_material_amplitude_full_hessian(device, block, kind, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    _, discrete_k, initial, electric, magnetic, spatial = fourier_mode(CELLS, 0)
    patterns = electric.copy(), magnetic.copy()
    policy = numerics(CELLS, STEPS).replace(checkpoint_steps=block)
    dt = .5/CELLS/math.sqrt(3)*UNITS.time_seconds
    prepared = fdtd.prepare(live_problem(kind, 1., 1., electric, magnetic),
                            numerics=policy, backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == STEPS
    assert prepared.contract['resolved_checkpoint_steps'] == block
    np.testing.assert_allclose(prepared.contract['resolved_dt_seconds'], dt,
                               rtol=2e-15, atol=0.)
    old = None
    for point, (amplitude, parameter) in enumerate(POINTS):
        leaves = (torch.tensor(amplitude, dtype=torch.float64, requires_grad=True),
                  torch.tensor(parameter, dtype=torch.float64, device=device, requires_grad=True))
        originals = tuple(leaf.detach().clone() for leaf in leaves)
        live = live_problem(kind, *leaves, electric, magnetic)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        assert bool(result.validity.checks['runtime_stability'])
        for values in raw_outputs(result):
            assert values.dtype == torch.float64 and values.device == torch.device(device)
            assert bool(torch.isfinite(values).all())
        fields = result.final_fields.values
        loss = (fields['E']**2).sum()+((Z0*fields['H'])**2).sum()
        gradients = torch.autograd.grad(loss, leaves, create_graph=True)
        rows = [torch.autograd.grad(derivative, leaves, retain_graph=index == 0)
                for index, derivative in enumerate(gradients)]
        actual = (float(host(loss)), np.array([float(host(g)) for g in gradients]),
                  np.array([[float(host(g)) for g in row] for row in rows]))
        reference = expected(kind, amplitude, parameter, dt, discrete_k, initial, spatial)
        for value, wanted in zip(actual, reference):
            np.testing.assert_allclose(value, wanted, rtol=2e-12, atol=0.)
        for derivatives in (gradients, *rows):
            for leaf, saved, derivative in zip(leaves, originals, derivatives):
                assert leaf.is_leaf and leaf.grad is None and torch.equal(leaf, saved)
                assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
        if old is not None:
            for value, saved in old:
                assert torch.equal(value, saved)
        old = tuple((value, value.detach().clone()) for value in raw_outputs(result))
        np.testing.assert_array_equal(electric, patterns[0])
        np.testing.assert_array_equal(magnetic, patterns[1])
        record_property(f'point_{point}_amplitude_parameter', repr((amplitude, parameter)))
        record_property(f'point_{point}_value_gradient_hessian',
                        repr(tuple(np.asarray(v).tolist() for v in actual)))
        record_property(f'point_{point}_oracle_value_gradient_hessian',
                        repr(tuple(np.asarray(v).tolist() for v in reference)))
    record_property('steps_checkpoint_tail', repr((STEPS, block, STEPS % block if block else None)))
