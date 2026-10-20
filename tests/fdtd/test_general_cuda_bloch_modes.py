"""Public static Bloch faces against independent discrete Yee eigenmodes.

The 16^3, 300-step budgets are those of test_fd_bloch: 1e-11 for the
relative temporal recurrence and 1e-10 for phase/amplitude. No CPU solve is
used as a CUDA oracle. Three cyclic orientations exercise signed Bloch pairs
and the remaining Periodic axis; a fourth case carries Bloch on all axes.
Initial patterns and all boundary declarations are static host values.
"""
import math

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.units import Z0
from tests.fixtures.fdtd.fd_helpers import (
    discrete_omega, plane_wave_polarization, recurrence_residual,
)


pytestmark = pytest.mark.slow
UNITS = Units(length='um')
N = 16
DX = 1 / N
DT = .5 * DX / math.sqrt(3)
STEPS = 300
RECURRENCE_TOL = 1e-11
FIELD_TOL = 1e-10
MUTANT_FLOOR = 1e-4
CASES = [pytest.param(rotation, False, id='mixed-' + axis)
         for rotation, axis in enumerate('xyz')]
CASES.append(pytest.param(0, True, id='all-bloch'))


@pytest.fixture(params=['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def yee_axes(field, component):
    # Arrays are component,z,y,x, but physical axes below are x,y,z. These
    # offsets are the Yee definition, independent of the prepared plan.
    return tuple(-.5 + (np.arange(N) + .5 * (
        axis == component if field == 'E' else axis != component)) * DX
        for axis in range(3))


def mesh(field, component):
    x, y, z = yee_axes(field, component)
    z, y, x = np.meshgrid(z, y, x, indexing='ij')
    return x, y, z


def probe_at(field, component):
    return tuple(values[index] for values, index in
                 zip(yee_axes(field, component), (N // 2 + 1, N // 2, N // 2 - 1)))


def mode(rotation, all_bloch):
    phases = np.roll(np.array([.7, -.4 if all_bloch else 0., -1.1]), rotation)
    k = 2 * np.pi + phases  # L=1, integer winding one on every axis.
    discrete_k = 2 * np.sin(k * DX / 2) / DX
    # Rotate the same transverse eigenpolarization along with the axes.
    polarization = np.roll(plane_wave_polarization(np.roll(k, -rotation), DX), rotation)
    assert abs(discrete_k @ polarization) < 1e-14
    curl_amplitude = 1j * np.cross(discrete_k, polarization)

    def value(field, component, at):
        amplitude = polarization[component] if field == 'E' else curl_amplitude[component]
        return amplitude * np.exp(1j * sum(ki * ai for ki, ai in zip(k, at)))

    electric = np.stack([value('E', c, mesh('E', c)) for c in range(3)])
    curl = np.stack([value('H', c, mesh('H', c)) for c in range(3)])
    faces = {axis: fdtd.Periodic() if phase == 0. else fdtd.Bloch(phase=float(phase))
             for axis, phase in zip('xyz', phases)}
    problem = fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,) * 3), boundaries=fdtd.Boundaries(**faces)),
        initial=fdtd.InitialField(E=electric, H=DT / 2 * curl / Z0),
        outputs=fdtd.Outputs(final_fields=True, monitors=tuple(
            fdtd.TimeSeriesMonitor(field + 'xyz'[c], component=field + 'xyz'[c], at=probe_at(field, c))
            for field in ('E', 'H') for c in range(3))))
    return problem, phases, k, electric, curl, value


@pytest.mark.parametrize('rotation,all_bloch', CASES)
def test_public_static_bloch_has_exact_discrete_fields_and_records(device, rotation, all_bloch):
    torch, name = device
    problem, phases, k, electric, curl, value = mode(rotation, all_bloch)
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=STEPS), courant=.5)
    print(dict(device=name, steps=STEPS, shape=(N,) * 3, phases=phases.tolist(),
               spacing=DX, courant=.5), flush=True)
    snapshots = tuple(np.array(v, copy=True) for v in (problem.initial.E, problem.initial.H))
    with torch.no_grad():
        result = fdtd.run(problem, numerics=numerics, backend='torch', device=name)
    assert bool(result.validity.checks['runtime_stability'])
    assert result.provenance['dt_seconds'] == pytest.approx(
        DT * UNITS.time_unit_seconds, rel=2e-15, abs=0.)
    for original, snapshot in zip((problem.initial.E, problem.initial.H), snapshots):
        np.testing.assert_array_equal(original, snapshot)

    omega = discrete_omega(k, DX, DT)
    factor = DT / (2 * math.sin(omega * DT / 2))
    wrong_frequencies = (
        discrete_omega(2 * np.pi - phases, DX, DT),
        discrete_omega(np.full(3, 2 * np.pi), DX, DT),
    )
    expected_fields = dict(
        E=electric * math.cos(omega * STEPS * DT),
        H=-factor * curl * math.sin(omega * (STEPS - .5) * DT))
    metrics = dict(device=name, phases=phases.tolist(), field_error={}, recurrence={},
                   record_error={}, wrong_phase_residual={})
    active_records = 0
    for field in ('E', 'H'):
        samples = result.final_fields.values[field]
        assert samples.device == torch.device(name) and samples.dtype == torch.complex128
        actual = host(samples) * (Z0 if field == 'H' else 1.)
        expected = expected_fields[field]
        peak = float(np.max(np.abs(expected)))
        assert peak > 1e-3 and np.isfinite(actual).all()
        error = float(np.max(np.abs(actual - expected)) / peak)
        metrics['field_error'][field] = error
        assert error < FIELD_TOL
        # A real-valued implementation cannot pass through an amplitude-only check.
        assert np.max(np.abs(actual.imag)) > .1 * peak
        for c in range(3):
            key = field + 'xyz'[c]
            series = result[key]
            assert series.values.device == torch.device(name)
            assert series.values.dtype == torch.complex128
            values = host(series.values) * (Z0 if field == 'H' else 1.)
            times = (np.arange(STEPS) + (1. if field == 'E' else .5)) * DT
            np.testing.assert_allclose(host(series.times_seconds), times * UNITS.time_unit_seconds,
                                       rtol=2e-14, atol=0.)
            spatial = value(field, c, probe_at(field, c))
            expected = spatial * (np.cos(omega * times) if field == 'E'
                                   else -factor * np.sin(omega * times))
            scale = float(np.max(np.abs(expected)))
            if scale == 0.:
                assert np.max(np.abs(values)) < FIELD_TOL
                continue
            active_records += 1
            assert scale > 1e-3 and np.max(np.abs(values.imag)) > 1e-3
            error = float(np.max(np.abs(values - expected)) / scale)
            recurrence = recurrence_residual(values, omega, DT)
            mutants = [recurrence_residual(values, wrong, DT) for wrong in wrong_frequencies]
            metrics['record_error'][key] = error
            metrics['recurrence'][key] = recurrence
            metrics['wrong_phase_residual'][key] = mutants
            assert error < FIELD_TOL
            assert recurrence < RECURRENCE_TOL
            assert min(mutants) > MUTANT_FLOOR
    assert active_records == 5  # Two transverse E components and all three H components.
    print(metrics, flush=True)
