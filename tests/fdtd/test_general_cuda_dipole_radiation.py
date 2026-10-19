"""Public CurrentDipole SI radiation, duality and geometry-unit gates on Torch.

The radiation budgets are the existing test_fd_dipole.py gates: vacuum
14/20/28 cells per wavelength, eight periods, a .6-wavelength physical box
and twelve PML cells. No far-field monitor or automatic stopping is used.
CPU cases qualify the public fixture independently of CUDA admission.
"""
import math

import numpy as np
import pytest

from wavebit import Units, fdtd
from wavebit.units import C0, EPSILON0, Z0


pytestmark = pytest.mark.slow
CENTER = (.013, -.021, .007)
RESOLUTIONS = (14, 20, 28)
CURRENT = 1e-12  # Electric current moment in A*m; the magnetic dual is Z0*CURRENT V*m.
TIME_UNIT = 1e-6 / C0
COARSE_ERROR = (-.06, -.03)


def host(value):
    return value.detach().cpu().numpy()


def larmor_energy(current=CURRENT):
    """SI vacuum energy of the Gaussian current element, independent of the solver.

    U = mu0/(6*pi*c) integral (dI/dt)^2 dt. For Gaussian width tau and
    angular frequency omega, the integral is I0^2*sqrt(pi)*tau/2 times
    (omega^2+1/(2*tau^2)), up to the same exponentially small carrier/envelope
    truncation terms covered by the established radiation gate.
    """
    mu0 = Z0 / C0
    tau = TIME_UNIT / (2 * math.pi * .15)
    omega = 2 * math.pi / TIME_UNIT
    return mu0 * current**2 * math.sqrt(math.pi) * tau * (
        omega**2 + .5/tau**2) / (12 * math.pi * C0)


@pytest.fixture(scope='module', params=['cpu', 'cuda:0'])
def device(request):
    torch = pytest.importorskip('torch')
    if request.param.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch, request.param


def problem(*, kind='electric', axis='z', center=CENTER, size=.6,
            periodic=False, delay=0., final_fields=False):
    field = 'E' if kind == 'electric' else 'H'
    boundaries = (fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())
                  if periodic else fdtd.Boundaries.open())
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(size,)*3), boundaries=boundaries),
        sources=(fdtd.CurrentDipole('d', center=center, component=field+axis, kind=kind,
            current_moment=CURRENT if kind == 'electric' else Z0*CURRENT,
            pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=.15/TIME_UNIT),
            delay_seconds=delay),), outputs=fdtd.Outputs(final_fields=final_fields))


def solve(device, physical, numerics):
    torch, name = device
    # Captured pytest logs preserve the precise attempted case even on failure.
    source = physical.sources[0]
    print(dict(device=name, kind=source.kind, component=source.component,
        center=source.center, delay_seconds=source.delay_seconds, spacing=numerics.grid.spacing,
        duration=repr(numerics.duration), units=physical.units.length), flush=True)
    with torch.no_grad():
        result = fdtd.run(physical, numerics=numerics, backend='torch', device=name)
    assert bool(result.validity.checks['runtime_stability'])
    reference = result.sources['d']
    assert reference['drive_units'] == ('A*m' if source.kind == 'electric' else 'V*m')
    assert reference['field_units'] == ('V/m' if source.kind == 'electric' else 'A/m')
    assert reference['power_units'] == 'W' and reference['time_unit'] == 's'
    for key in ('drive', 'field', 'delivered_power'):
        assert reference[key].device == torch.device(name)
        assert reference[key].dtype == torch.float64
        assert np.isfinite(host(reference[key])).all()
    dt = result.provenance['dt_seconds']
    sample = dict(dt=dt, energy=dt*float(host(reference['delivered_power']).sum()),
        drive=host(reference['drive']).copy(), field=host(reference['field']).copy(),
        power=host(reference['delivered_power']).copy(),
        time=np.asarray(reference['time']).copy(), field_time=np.asarray(reference['field_time']).copy())
    if physical.outputs.final_fields:
        sample.update(E=host(result.final_fields.values['E']).copy(),
                      H=host(result.final_fields.values['H']).copy())
    if 'probe' in result.outputs:
        sample['probe'] = host(result['probe'].values).copy()
    return sample


@pytest.fixture(scope='module')
def radiation(device):
    cache = {}

    def measure(cpw, magnetic=False):
        key = (cpw, magnetic)
        if key in cache:
            return cache[key]
        dx = 1/cpw
        dt = .5*dx/math.sqrt(3)*TIME_UNIT
        physical = problem(kind='magnetic' if magnetic else 'electric',
            center=tuple(c+dx/2 for c in CENTER) if magnetic else CENTER,
            delay=-dt/2 if magnetic else 0.)
        numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
            duration=fdtd.Duration(periods=8, source='d'), courant=.5,
            absorber=fdtd.PML(cells=12, wavelength=1.))
        sample = solve(device, physical, numerics)
        assert sample['dt'] == pytest.approx(dt, rel=2e-15, abs=0.)
        assert np.isfinite(sample['energy']) and sample['energy'] > 0
        cache[key] = sample
        return sample

    return measure


def test_public_dipole_larmor_energy_converges_at_second_order(radiation, record_property):
    exact = larmor_energy()
    record_property('larmor_reference_joules', exact)
    assert np.isfinite(exact) and exact > 0
    energies = np.asarray([radiation(cpw)['energy'] for cpw in RESOLUTIONS])
    errors = energies/exact-1
    for cpw, energy, error in zip(RESOLUTIONS, energies, errors):
        record_property(f'energy_joules_cpw_{cpw}', float(energy))
        record_property(f'relative_error_cpw_{cpw}', float(error))
    assert np.all(errors < 0), errors
    order = np.polyfit(np.log(1/np.asarray(RESOLUTIONS)), np.log(np.abs(errors)), 1)[0]
    record_property('fitted_order', float(order))
    assert 1.9 < order < 2.1, (order, errors)
    ratio = (RESOLUTIONS[2]/RESOLUTIONS[1])**2
    limit = (ratio*errors[2]-errors[1])/(ratio-1)
    record_property('richardson_relative_error', float(limit))
    assert abs(limit) < 1e-3, (limit, errors)
    assert COARSE_ERROR[0] < errors[0] < COARSE_ERROR[1], errors


def test_public_electric_magnetic_open_box_radiation_duality(radiation, record_property):
    electric = radiation(RESOLUTIONS[0])['energy']
    magnetic = radiation(RESOLUTIONS[0], magnetic=True)['energy']
    record_property('electric_energy_joules', electric)
    record_property('magnetic_energy_joules', magnetic)
    record_property('open_duality_relative_error', magnetic/electric-1)
    # M=Z0*I gives equal SI radiated energy: eps0*M^2 = mu0*I^2.
    assert EPSILON0*(Z0*CURRENT)**2 == pytest.approx((Z0/C0)*CURRENT**2, rel=2e-15, abs=0.)
    assert math.isclose(electric, magnetic, rel_tol=1e-4), (electric, magnetic)
    error = magnetic/larmor_energy()-1
    assert COARSE_ERROR[0] < error < COARSE_ERROR[1], error


@pytest.mark.parametrize('axis', 'xyz')
def test_public_electric_magnetic_periodic_lattice_duality(device, axis):
    dx = 1/14
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=dx),
        duration=fdtd.Duration(steps=150), courant=.5)
    electric = solve(device, problem(axis=axis, size=16*dx, periodic=True,
                                    final_fields=True), numerics)
    magnetic = solve(device, problem(kind='magnetic', axis=axis, size=16*dx,
        periodic=True, final_fields=True, center=tuple(c+dx/2 for c in CENTER),
        delay=-electric['dt']/2), numerics)
    peak_drive = np.max(np.abs(electric['drive']))
    peak_field = np.max(np.abs(electric['field']))
    assert peak_drive > 0 and peak_field > 0 and electric['energy'] > 0
    assert np.max(np.abs(magnetic['drive']/Z0-electric['drive'])) < 1e-13*peak_drive
    assert np.max(np.abs(magnetic['field']*Z0-electric['field'])) < 1e-13*peak_field
    assert math.isclose(magnetic['energy'], electric['energy'], rel_tol=1e-13)
    peak = np.max(np.abs(electric['E']))
    assert peak > 0
    for component in range(3):
        shifted = np.roll(electric['E'][component], 1, axis=2-component)
        assert np.max(np.abs(magnetic['H'][component]*Z0-shifted)) < 1e-13*peak


@pytest.mark.parametrize('kind', ['electric', 'magnetic'])
def test_public_dipole_si_source_is_invariant_under_geometry_units(device, kind):
    component = 'Ey' if kind == 'electric' else 'Hy'

    def measure(scale, length):
        physical = fdtd.Problem(units=Units(length=length), scene=fdtd.Scene(
            domain=fdtd.Domain(size=(1.6*scale,)*3),
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())),
            sources=(fdtd.CurrentDipole('d', center=(0., 0., 0.), component=component, kind=kind,
                current_moment=CURRENT if kind == 'electric' else Z0*CURRENT,
                pulse=fdtd.GaussianPulse(wavelength=3*scale, frequency_sigma_hz=.03/TIME_UNIT),
                delay_seconds=-18*TIME_UNIT),),
            outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('probe', at=(.3*scale, 0., 0.),
                                                       component=component),)))
        numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1*scale),
            duration=fdtd.Duration(steps=30))
        return solve(device, physical, numerics)

    micrometres, nanometres = measure(1., 'um'), measure(1000., 'nm')
    assert np.max(np.abs(micrometres['probe'])) > 1e-8
    np.testing.assert_allclose(micrometres['probe'], nanometres['probe'], rtol=1e-10, atol=1e-15)
    moment = CURRENT if kind == 'electric' else Z0*CURRENT
    np.testing.assert_allclose(micrometres['drive'], nanometres['drive'], rtol=1e-12, atol=moment*1e-14)
    assert np.max(np.abs(micrometres['drive'])) <= moment
    for key in ('field', 'power'):
        peak = np.max(np.abs(micrometres[key]))
        assert peak > 0
        assert np.max(np.abs(micrometres[key]-nanometres[key])) < 1e-10*peak
    for key in ('time', 'field_time'):
        np.testing.assert_allclose(micrometres[key], nanometres[key], rtol=2e-15, atol=0.)
