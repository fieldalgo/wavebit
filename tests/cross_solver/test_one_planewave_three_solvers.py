"""One PlaneWave, three solvers, one meaning for every diffraction name (fieldalgo/wavebit#54).

A uniform glass film between vacuum half-spaces in a crossed 0.2 um cell, lit from above at
0.2 rad, p polarized, with the drive ``amplitude=2j`` at one wavelength (0.9 um), is solved by
``rcwa.run``, ``fdtd.periodic.run`` and ``fdtd.run``. Each name must mean one physical quantity
in all three:

- ``amplitude_ratio_*`` is the outgoing field over the incident field. It does not see the drive,
  so it agrees across the solvers to their discretization error.
- ``field_*`` is that ratio times the CW drive, in V/m: rcwa and periodic agree, and fdtd.run,
  whose pulse has no single field, has none.
- ``incident_power`` and ``power`` are in W and scale with |2j|**2: rcwa and periodic agree.

Before #54 the periodic ratios were multiplied by the drive, which is a factor 2j here and off
by more than every tolerance below. The periodic solver's own drive test
(tests/fdtd/test_periodic_drive.py) now also catches a ratio that carries the drive. This
comparison pins something no single solver's test can: that the three solvers mean the same
thing by each name, so it also catches a wrong factor that does not follow the drive, in any
of them.
One wavelength, because rcwa fixes the angle while the FDTD solvers fix k_par across a band.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd, rcwa
from wavebit.fdtd import periodic
from wavebit.units import C0, Z0

WAVELENGTH, THETA, DRIVE, PERIOD, THICKNESS = .9, .2, 2j, .2, .1
GLASS, AIR = Material(epsilon=2.25), Material(epsilon=1.)


def spectral():
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), layers=(rcwa.Layer(thickness=THICKNESS, background=GLASS),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH, sources=(
            rcwa.PlaneWave(amplitude=DRIVE, theta=THETA, polarization='p'),)))
    return rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))


def periodic_run():
    problem = periodic.Problem(units=Units(length='um'),
        stack=periodic.Stack(period=(PERIOD, PERIOD), top=AIR, bottom=AIR,
            layers=(periodic.Layer(thickness=THICKNESS, epsilon=np.full((2, 2), 2.25)),)),
        excitation=periodic.Excitation(wavelength=WAVELENGTH, sources=(
            periodic.PlaneWave(amplitude=DRIVE, theta=THETA, polarization='p'),)))
    numerics = periodic.Numerics(spacing=1/40, duration_seconds=55*WAVELENGTH*1e-6/C0, orders=(0, 0))
    return periodic.run(problem, numerics=numerics)


def temporal_run():
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.6), polarization='p', theta=THETA,
                            amplitude=DRIVE,
                            pulse=fdtd.GaussianPulse(wavelength=WAVELENGTH, frequency_sigma_hz=3e13))
    units = Units(length='um')
    faces = fdtd.Bloch.from_source(source, units=units)
    monitors = tuple(fdtd.DiffractionMonitor(name, plane=fdtd.Plane(z=z), wavelengths=(WAVELENGTH,),
        source='pw', wave=wave, orders=(0, 0), reference_plane=reference, incident_plane=THICKNESS)
        for name, z, wave, reference in (('r', .4, 'reflected', THICKNESS), ('t', -.3, 'transmitted', 0.)))
    problem = fdtd.Problem(units=units,
        scene=fdtd.Scene(domain=fdtd.Domain(size=(PERIOD, PERIOD, 1.6)), background=AIR,
            boundaries=fdtd.Boundaries(**faces, z=fdtd.Open()),
            structures=(fdtd.Structure('film', shape=fdtd.Box(size=(2*PERIOD, 2*PERIOD, THICKNESS),
                                                                center=(0., 0., THICKNESS/2)),
                                       material=GLASS),)),
        sources=(source,), outputs=fdtd.Outputs(monitors=monitors))
    return fdtd.run(problem, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
                                                    duration=fdtd.Duration(seconds=100e-15)))


@pytest.fixture(scope='module')
def three():
    return spectral(), periodic_run(), temporal_run()


def test_amplitude_ratios_agree_and_ignore_the_drive(three):
    exact, stepped, pulsed = three
    for a, b, c in ((exact.reflection, stepped.reflection, pulsed['r']),
                    (exact.transmission, stepped.transmission, pulsed['t'])):
        for orders in (a, b, c):
            assert orders.reference['amplitude_ratio_units'] == '1'
        assert abs(a.amplitude_ratio_p[0, 0, 0]) > .05       # a ratio the comparison can see
        np.testing.assert_allclose(b.amplitude_ratio_p, a.amplitude_ratio_p, atol=.02)
        np.testing.assert_allclose(c.amplitude_ratio_p, a.amplitude_ratio_p, atol=.02)
        np.testing.assert_allclose(b.efficiency, a.efficiency, atol=.01)
        np.testing.assert_allclose(c.efficiency, a.efficiency, atol=.01)


def test_fields_carry_the_cw_drive_and_a_pulse_has_none(three):
    exact, stepped, pulsed = three
    for a, b, c in ((exact.reflection, stepped.reflection, pulsed['r']),
                    (exact.transmission, stepped.transmission, pulsed['t'])):
        for orders in (a, b):
            assert orders.reference['incident_field'] == DRIVE
            assert orders.reference['incident_field_units'] == 'V/m'
            np.testing.assert_allclose(orders.field_p, DRIVE*orders.amplitude_ratio_p, rtol=1e-15)
        np.testing.assert_allclose(b.field_p, a.field_p, atol=.02*abs(DRIVE))
        assert c.field_s is None and c.field_p is None and c.field_xyz is None
        assert c.reference['incident_spectrum_units'] == 'V*s/m'
        assert 'incident_field' not in c.reference


def test_powers_are_watts_through_the_same_cell(three):
    exact, stepped, _ = three
    for a, b in ((exact.reflection, stepped.reflection), (exact.transmission, stepped.transmission)):
        assert a.reference['power_units'] == b.reference['power_units'] == 'W'
        assert a.reference['incident_power_units'] == b.reference['incident_power_units'] == 'W'
        # |2j|**2/(2 Z0) cos(0.2) through a 0.2 um square cell: both closed forms.
        expected = abs(DRIVE)**2/(2*Z0)*np.cos(THETA)*(PERIOD*1e-6)**2
        np.testing.assert_allclose(a.reference['incident_power'], expected, rtol=1e-12)
        np.testing.assert_allclose(b.reference['incident_power'], expected, rtol=1e-12)
        np.testing.assert_allclose(b.power, a.power, atol=.01*expected)
