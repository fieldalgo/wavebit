"""Public oblique Bloch PlaneWave against the independent discrete wave oracle.

The waveform, fixed-k Yee dispersion and staggered polarisation are the existing
stage-2 closed forms, not a second solver run. Public angles are radians and
times are seconds; the oracle explicitly converts back to length/c units.
The bounded conical case uses a damped transform to suppress finite-window
ring-down. This does not qualify arbitrary angles, media or AD.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.units import Z0
from tests.fdtd.test_fd_planewave import (
    A, PHASE, DELAY, CLOSED_TOL, DAMPED_TOL, GAMMA, ROUND_OFF,
    _dft, _waveform, _discrete_kz, _grid_polarisation,
)


pytestmark = pytest.mark.slow
UNITS = Units(length='um')
DX = .05
SIZE = (.25, .2, 3.)
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
OFFSETS = dict(Ex=(.5, 0., 0.), Ey=(0., .5, 0.), Ez=(0., 0., .5),
               Hx=(0., .5, .5), Hy=(.5, 0., .5), Hz=(.5, .5, 0.))
CASES = [(20., 's', 0., '-z', 1100, 0.), (20., 'p', 0., '-z', 1100, 0.),
         (20., 's', 0., '+z', 1100, 0.), (20., 'p', 0., '+z', 1100, 0.),
         (30., 'p', 35., '-z', 1600, GAMMA)]
CASE_IDS = ['theta20-s-minus', 'theta20-p-minus', 'theta20-s-plus',
            'theta20-p-plus', 'theta30-phi35-p-damped']


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def vectors(theta, phi, sign, polarization):
    """Continuum direction defines fixed k; Yee polarisation is derived separately."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    k = np.array([st * cp, st * sp, sign * ct])
    e = np.array([-sp, cp, 0.]) if polarization == 's' else np.array([sign * ct * cp, sign * ct * sp, -st])
    return k, e, np.cross(k, e)


def point(component, node_z, ix=0, iy=0):
    """Explicit Yee point of a physical transverse cell; no planner masks/indices."""
    ox, oy, oz = OFFSETS[component]
    return (-SIZE[0] / 2 + (ix + ox) * DX,
            -SIZE[1] / 2 + (iy + oy) * DX, node_z + oz * DX)


def fixture(case):
    theta_deg, polarization, phi_deg, direction, steps, gamma = case
    theta, phi = np.deg2rad([theta_deg, phi_deg])
    sign = 1 if direction == '+z' else -1
    pulse = fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15 / UNITS.time_unit_seconds)
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=0.), direction=direction,
        theta=theta, phi=phi, polarization=polarization, pulse=pulse,
        amplitude=A, phase=PHASE, delay_seconds=DELAY * UNITS.time_unit_seconds)
    faces = fdtd.Bloch.from_source(source, units=UNITS)
    expected_k = 2 * np.pi * vectors(theta, phi, sign, polarization)[0]
    for axis, index in (('x', 0), ('y', 1)):
        if abs(expected_k[index]) < 1e-12:
            assert isinstance(faces[axis], fdtd.Periodic)
        else:
            assert isinstance(faces[axis], fdtd.Bloch)
            np.testing.assert_allclose(faces[axis].k, expected_k[index], rtol=1e-14, atol=0.)
    monitors = []
    for component in COMPONENTS:
        # Incident normal E and every H use the index of the interior H half plane.
        entry_node = 0. if component in ('Ex', 'Ey') or sign > 0 else -DX
        monitors.append(fdtd.TimeSeriesMonitor('entry_' + component, at=point(component, entry_node), component=component))
        for name, z, ix, iy in (('forward', sign * .8, 0, 0),
                                ('xnext', sign * .8, 1, 0),
                                ('ynext', sign * .8, 0, 1),
                                ('backward', -sign * .6, 0, 0)):
            monitors.append(fdtd.TimeSeriesMonitor(name + '_' + component,
                at=point(component, z, ix, iy), component=component))
    problem = fdtd.Problem(units=UNITS, scene=fdtd.Scene(
        domain=fdtd.Domain(size=SIZE), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries(**faces, z=fdtd.Open())), sources=(source,),
        outputs=fdtd.Outputs(monitors=tuple(monitors)))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=DX),
        duration=fdtd.Duration(steps=steps), courant=.5,
        absorber=fdtd.PML(cells=24, wavelength=1.))
    # Adapter for the established oracle's degree/length-time convention. Its
    # vectors are written above, not taken from lowering or the source column.
    oracle = SimpleNamespace(band=SimpleNamespace(wavelength=1., df=.15, cutoff=pulse.cutoff),
        amplitude=A, phase=PHASE, delay=DELAY, theta=theta_deg,
        polarization=polarization, sign=sign,
        vectors=lambda: vectors(theta, phi, sign, polarization))
    return problem, numerics, oracle, expected_k, gamma


@pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'cuda'])
def test_public_oblique_bloch_wave_matches_discrete_closed_form(case, device, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires CUDA-enabled Torch and an NVIDIA GPU')
    problem, numerics, oracle, k, gamma = fixture(case)
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch', device=device)
    assert prepared.report.ok
    with torch.no_grad():
        result = prepared.run(problem)
    assert bool(result.validity.checks['runtime_stability'])
    dt = prepared.contract['resolved_time_step_seconds'] / UNITS.time_unit_seconds
    plan = SimpleNamespace(dx=DX, dt=dt)
    source = result.source_references['pw']
    assert source['plane'] == pytest.approx(0., abs=1e-14)
    assert source['plane_H'] == pytest.approx(oracle.sign * DX / 2, abs=1e-14)
    incident = (host(source['incident']), Z0 * host(source['incident_H']))
    assert all(value.shape == (case[4], 3) for value in incident)
    forward = max(np.max(np.abs(host(result['forward_' + c].values))) * (Z0 if c[0] == 'H' else 1.)
                  for c in COMPONENTS)
    backward = max(np.max(np.abs(host(result['backward_' + c].values))) * (Z0 if c[0] == 'H' else 1.)
                   for c in COMPONENTS)
    assert forward > .5  # Prevent a missing/weak wave from satisfying relative gates.
    reference_peak = max(np.max(np.abs(values)) for values in incident)
    assert reference_peak > .5
    entry_error, phase_error = 0., 0.
    for index, component in enumerate(COMPONENTS):
        scale = Z0 if component[0] == 'H' else 1.
        entry = result['entry_' + component]
        values = host(entry.values) * scale
        expected = incident[index // 3][:, index % 3]
        entry_error = max(entry_error, np.max(np.abs(values - expected)) / reference_peak)
        expected_times = host(source['magnetic_time' if component[0] == 'H' else 'time'])
        np.testing.assert_array_equal(host(entry.times_seconds), expected_times)
        for name in ('entry', 'forward', 'backward', 'xnext', 'ynext'):
            actual = result[name + '_' + component].values
            assert actual.device == torch.device(device) and actual.dtype == torch.complex128
            assert np.isfinite(host(actual)).all()
        base = host(result['forward_' + component].values) * scale
        for name, axis in (('xnext', 0), ('ynext', 1)):
            neighbour = host(result[name + '_' + component].values) * scale
            phase_error = max(phase_error, np.max(np.abs(neighbour - base * np.exp(1j * k[axis] * DX))) / forward)
    assert backward / forward <= ROUND_OFF
    assert entry_error <= ROUND_OFF
    assert phase_error <= ROUND_OFF

    edge = np.sqrt(2 * np.log(1 / oracle.band.cutoff))
    frequencies = np.linspace(1 - edge * .15, 1 + edge * .15, 41)
    omega = 2 * np.pi * frequencies + 1j * gamma
    times = (np.arange(case[4]) + 1.) * dt
    waveform = _dft(times, _waveform(oracle, times), omega)
    polarisation = _grid_polarisation(oracle, plan)
    normalization = max(np.max(np.abs(value * waveform)) for value in polarisation)
    # The hard-source plane is one cell upstream of the requested sheet. The
    # probe is .8 downstream: derive .85 independently, never read column.ks.
    distance = .8 + DX
    error = 0.
    for c, component in enumerate(('Ex', 'Ey')):
        sample = result['forward_' + component]
        np.testing.assert_allclose(sample.coordinates['sample'][2], oracle.sign * .8, atol=1e-14, rtol=0.)
        expected = polarisation[c] * waveform * np.exp(1j * _discrete_kz(oracle, plan, omega) * distance)
        actual = _dft(host(sample.times_seconds) / UNITS.time_unit_seconds, host(sample.values), omega)
        error = max(error, float(np.max(np.abs(actual - expected)) / normalization))
    record_property('closed_form_peak_relative_error', error)
    record_property('backward_over_forward', float(backward / forward))
    record_property('entry_reference_relative_error', float(entry_error))
    record_property('cell_phase_relative_error', float(phase_error))
    assert error <= (DAMPED_TOL if gamma else CLOSED_TOL)
