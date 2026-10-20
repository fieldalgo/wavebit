"""Current physical APIs agree on port powers and phase under measured refinement.

The asymmetric lateral placement makes odd diffraction orders sensitive to the
cell-centre origin. No compatibility factories, private kernels, or frozen
numerical answers are used. The continuum RCWA solution is refined independently;
FDTD grid and integration duration are varied separately at fixed physical planes.
"""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd, rcwa


def problems(profile, polarization):
    units = Units(length='um')
    air, glass = Material(epsilon=1.), Material(epsilon=2.25)
    if profile == 'grating':
        layer = rcwa.Layer(thickness=.2, background=air, regions=(rcwa.Region(
            geometry=rcwa.Rectangle(size=(.6, .1), center=(.15, 0.)), material=glass),))
        body = fdtd.Box(size=(.6, .3, .2), center=(.15, 0., .1))
    else:
        layer = rcwa.Layer(thickness=.2, background=glass)
        # Extend invariant axes beyond the periodic faces. Box retains fractional
        # Yee-cell filling of the physical z interfaces under grid refinement.
        body = fdtd.Box(size=(1.4, .3, .2), center=(0., 0., .1))
    spectral = rcwa.Problem(units=units,
        stack=rcwa.Stack(period=(1.2, .1), layers=(layer,)),
        excitation=rcwa.Excitation(wavelength=.8,
            sources=(rcwa.PlaneWave(polarization=polarization),)))
    monitors = tuple(fdtd.DiffractionMonitor(name, plane=fdtd.Plane(z=z), wavelengths=(.8,),
        source='pw', wave=wave, orders=(1, 0), reference_plane=reference, incident_plane=.2)
        for name, z, wave, reference in (('r', .4, 'reflected', .2), ('t', -.3, 'transmitted', 0.)))
    temporal = fdtd.Problem(units=units,
        scene=fdtd.Scene(domain=fdtd.Domain(size=(1.2, .1, 1.6)), background=air,
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
            structures=(fdtd.Structure('body', shape=body, material=glass),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.6), polarization=polarization,
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=3e13)),),
        outputs=fdtd.Outputs(monitors=monitors))
    return spectral, temporal


def observations(outputs, indices=None):
    if indices is None:
        indices = [0, 1, 2]
    amplitudes = np.stack([np.stack((o.amplitude_s[..., indices], o.amplitude_p[..., indices]))
                           for o in outputs])
    powers = np.stack([o.efficiency[..., indices] for o in outputs])
    return amplitudes, powers


def spectral_run(problem, orders):
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(orders, 0)))
    indices = [tuple(result.reflection.orders_u).index(m) for m in (-1, 0, 1)]
    return result, observations((result.reflection, result.transmission), indices)


@pytest.mark.parametrize('profile', ['film', 'grating'])
@pytest.mark.parametrize('polarization', ['s', 'p'])
def test_public_order_amplitudes_and_powers_converge_at_shared_reference_planes(profile, polarization):
    spectral, temporal = problems(profile, polarization)
    _, previous = spectral_run(spectral, 7)
    reference, expected = spectral_run(spectral, 11)
    assert np.max(np.abs(expected[0] - previous[0])) < 5e-4
    assert np.max(np.abs(expected[1] - previous[1])) < 1e-4

    if profile == 'grating':
        # Translation of a shape has a known complex Fourier phase. These odd
        # orders cannot pass through a corner/centre phase mismatch unnoticed.
        layer = spectral.stack.layers[0]
        region = layer.regions[0]
        centred = spectral.replace(stack=spectral.stack.replace(layers=(layer.replace(
            regions=(region.replace(geometry=region.geometry.replace(center=(0., 0.))),)),)))
        _, at_origin = spectral_run(centred, 11)
        phase = np.exp(-2j*np.pi*np.array([-1, 0, 1])*.15/1.2)
        np.testing.assert_allclose(expected[0], at_origin[0]*phase, atol=2e-12, rtol=0)
        assert np.max(np.abs(expected[0][..., [0, 2]])) > .1

    measured = []
    for spacing, seconds in ((.05, 70e-15), (.025, 70e-15), (.025, 100e-15), (.025, 140e-15)):
        result = fdtd.run(temporal, numerics=fdtd.Numerics(
            grid=fdtd.UniformGrid(spacing=spacing), duration=fdtd.Duration(seconds=seconds)))
        ports = (result['r'], result['t'])
        for port, other, reference_z in zip(ports, (reference.reflection, reference.transmission), (.2, 0.)):
            assert tuple(port.orders_u) == (-1, 0, 1)
            assert port.amplitude_s.shape == (1, 1, 3)
            assert port.reference['plane'] == other.reference['plane'][0] == reference_z
            assert port.reference['wave'] == other.reference['wave']
            assert (port.reference['port'], other.reference['port']) in (('z+', 'top'), ('z-', 'bottom'))
            assert port.reference['incident_plane'] == .2
            assert port.reference['lateral_origin'] == other.reference['lateral_origin'] == (0., 0.)
            assert port.reference['outward_normal'] == other.reference['outward_normal']
            assert port.reference['epsilon'] == other.reference['epsilon'][0] == 1.
            assert port.reference['mu'] == other.reference['mu'][0] == 1.
            assert port.reference['cell_measure'] == pytest.approx(other.reference['cell_measure'][0])
            assert port.reference['cell_measure_units'] == other.reference['cell_measure_units'] == 'm^2'
            assert bool(port.metadata['source_usable'][0])
        measured.append(observations(ports))

    # Independently refined space and time. Bounds are on complex coefficients,
    # not only squared magnitudes; their phases are part of the public contract.
    for index, ceiling in ((0, .012), (1, .005)):
        coarse = np.max(np.abs(measured[0][index] - expected[index]))
        fine = np.max(np.abs(measured[1][index] - expected[index]))
        assert fine < .35*coarse
        assert np.max(np.abs(measured[-1][index] - expected[index])) < ceiling
    early_tail = np.max(np.abs(measured[2][0] - measured[1][0]))
    late_tail = np.max(np.abs(measured[3][0] - measured[2][0]))
    assert late_tail < max(.8*early_tail, 1e-6)
    assert late_tail < 5e-4
    assert np.sum(measured[-1][1]) == pytest.approx(1., abs=.002)
