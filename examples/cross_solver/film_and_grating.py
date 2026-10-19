"""Compare the current RCWA and FDTD APIs on one film and one shifted grating.

Both use shared materials and um, a cell-centred lateral origin, +z upward, and a
film occupying z=0..0.2. Amplitudes refer to the film's top/bottom faces, not the
FDTD monitor planes. The odd orders of the shifted grating retain complex phase.
This small spatial/time ladder is a case-specific check, not a universal accuracy
certificate. NumPy only. Use --polarization p for the other linear polarization.
"""
import argparse
import numpy as np

from wavebit import Material, Units, fdtd, rcwa

UNITS = Units(length='um')
PERIOD, THICKNESS, WAVELENGTH = (1.2, .1), .2, .8
AIR, GLASS = Material(epsilon=1.), Material(epsilon=2.25)
ORDERS = (-1, 0, 1)


def problems(profile, polarization):
    # Define the dimensions once; each solver receives its own physical problem.
    width, offset = .6, .15
    if profile == 'grating':
        layer = rcwa.Layer(thickness=THICKNESS, background=AIR, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(width, PERIOD[1]), center=(offset, 0.)), material=GLASS),))
        body = fdtd.Box(size=(width, 3*PERIOD[1], THICKNESS), center=(offset, 0., THICKNESS/2))
    else:
        layer = rcwa.Layer(thickness=THICKNESS, background=GLASS)
        # Cover both periodic faces; the same finite z interval defines the film.
        body = fdtd.Box(size=(PERIOD[0]+.2, 3*PERIOD[1], THICKNESS), center=(0., 0., THICKNESS/2))
    spectral = rcwa.Problem(units=UNITS, stack=rcwa.Stack(period=PERIOD, layers=(layer,)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
            sources=(rcwa.PlaneWave(polarization=polarization),)))
    temporal = fdtd.Problem(units=UNITS,
        scene=fdtd.Scene(domain=fdtd.Domain(size=(*PERIOD, 1.6)), background=AIR,
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
            structures=(fdtd.Structure('film', shape=body, material=GLASS),)),
        sources=(fdtd.PlaneWave('illumination', plane=fdtd.Plane(z=.6), polarization=polarization,
            pulse=fdtd.GaussianPulse(wavelength=WAVELENGTH, frequency_sigma_hz=3e13)),),
        outputs=fdtd.Outputs(monitors=tuple(fdtd.DiffractionMonitor(name, plane=fdtd.Plane(z=z),
            wavelengths=(WAVELENGTH,), source='illumination', wave=wave, orders=(1, 0),
            reference_plane=reference, incident_plane=THICKNESS)
            for name, z, wave, reference in (
                ('r', .4, 'reflected', THICKNESS), ('t', -.3, 'transmitted', 0.)))))
    return spectral, temporal


def arrays(ports):
    # Match order labels explicitly; RCWA also retains higher, evanescent orders.
    amplitude = np.array([[[np.asarray(port.order(m).amplitude_s)[0],
                           np.asarray(port.order(m).amplitude_p)[0]] for m in ORDERS] for port in ports])
    efficiency = np.array([[np.asarray(port.order(m).efficiency)[0] for m in ORDERS] for port in ports])
    return amplitude, efficiency


def spectral_run(problem, order):
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(order, 0)))
    ports = result.reflection, result.transmission
    return ports, arrays(ports)


def compare(profile, polarization):
    spectral, temporal = problems(profile, polarization)
    _, low = spectral_run(spectral, 7)
    reference, expected = spectral_run(spectral, 11)
    assert np.max(np.abs(expected[0]-low[0])) < 5e-4
    assert np.max(np.abs(expected[1]-low[1])) < 1e-4
    if profile == 'grating':
        assert np.max(np.abs(expected[0][:, [0, 2]])) > .1
    print(f'\n{profile}, {polarization}; reference faces z=0.2/0 um, lateral origin=(0,0)')
    print('spacing (nm)   duration (fs)   max complex-amplitude gap   max efficiency gap')
    observed = []
    for spacing, duration in ((.05, 70e-15), (.025, 70e-15), (.025, 100e-15), (.025, 140e-15)):
        result = fdtd.run(temporal, numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=spacing),
            duration=fdtd.Duration(seconds=duration)))
        ports = result['r'], result['t']
        for port, other, z in zip(ports, reference, (THICKNESS, 0.)):
            np.testing.assert_array_equal(port.wavelengths, other.wavelengths)
            assert port.reference['plane'] == other.reference['z'][0] == z
            assert port.reference['incident_plane'] == THICKNESS
            for key in ('lateral_origin', 'outward_normal'):
                assert port.reference[key] == other.reference[key]
            for key in ('epsilon', 'mu', 'cell_area_m2'):
                np.testing.assert_allclose(port.reference[key], other.reference[key])
            assert np.all(port.metadata['source_usable'])
        current = arrays(ports)
        observed.append(current)
        gaps = [np.max(np.abs(a-b)) for a, b in zip(current, expected)]
        print(f'{spacing*1000:12.1f}   {duration*1e15:13.1f}   {gaps[0]:25.6g}   {gaps[1]:18.6g}')
    for index, tolerance in ((0, .012), (1, .005)):
        coarse = np.max(np.abs(observed[0][index]-expected[index]))
        fine = np.max(np.abs(observed[1][index]-expected[index]))
        assert fine < .35*coarse
        assert np.max(np.abs(observed[-1][index]-expected[index])) < tolerance
    early = np.max(np.abs(observed[2][0]-observed[1][0]))
    late = np.max(np.abs(observed[3][0]-observed[2][0]))
    assert late < max(.8*early, 1e-6) and late < 5e-4
    np.testing.assert_allclose(observed[-1][1].sum(), 1., atol=.002, rtol=0)
    print(f'Last time-refinement change: {late:.3g}; summed R+T: {observed[-1][1].sum():.8f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--polarization', choices=('s', 'p'), default='s')
    args = parser.parse_args()
    for profile in ('film', 'grating'):
        compare(profile, args.polarization)


if __name__ == '__main__':
    main()
