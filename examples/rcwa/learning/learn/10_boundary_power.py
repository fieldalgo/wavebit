"""Compare isolated-wave efficiency with the net power at a lossy interface."""

import json

import numpy as np
from wavebit import Material, Units, rcwa


def boundary_case(epsilon):
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(1200., 1200.), top=Material(epsilon), bottom=Material(2.25),
            layers=(rcwa.Layer(thickness=100., background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=1000., sources=(
            rcwa.PlaneWave(theta=np.deg2rad(30.), polarization='s'),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))
    kx, ky = result.transmission.k_parallel[0, 0, 0] / (2*np.pi/1000.)
    beta_squared = (kx.item() ** 2 + ky.item() ** 2).real
    q_in = np.sqrt(complex(epsilon - beta_squared))
    q_out = np.sqrt(complex(2.25 - beta_squared))
    r = (q_in - q_out) / (q_in + q_out)
    t = 1 + r

    # Independent s-wave boundary calculation; the common 1/Z0 cancels.
    incident = 0.5 * q_in.real
    reflected = 0.5 * abs(r) ** 2 * q_in.real
    transmitted = 0.5 * abs(t) ** 2 * q_out.real
    entering = 0.5 * np.real((1 + r) * np.conj(q_in * (1 - r)))
    expected = [reflected / incident, transmitted / incident, entering / incident]
    np.testing.assert_allclose(
        [result.R.sum(), result.T.sum(), result.outputs['power_entering'][0]], expected,
        rtol=2e-12, atol=2e-12,
    )

    # Time-average the real fields, including the two mixed products.
    time_phase = 2 * np.pi * np.arange(4096) / 4096
    phasor = np.exp(-1j * time_phase)
    ei, er = phasor.real, (r * phasor).real
    hi, hr = (q_in * phasor).real, (-q_in * r * phasor).real
    time_average = np.mean((ei + er) * (hi + hr)) / incident
    cross_term = np.mean(ei * hr + er * hi) / incident
    np.testing.assert_allclose(time_average, result.outputs['power_entering'][0], atol=2e-12)
    np.testing.assert_allclose(
        cross_term, 2 * q_in.imag * r.imag / q_in.real, atol=2e-12,
    )
    np.testing.assert_allclose(result.absorption[0], 0., atol=2e-12)
    return {
        'R': float(result.R.sum()), 'T': float(result.T.sum()),
        'power_entering': float(result.outputs['power_entering'][0]),
        'absorption': float(result.absorption[0]),
        'interference': float(cross_term),
        'one_minus_R_minus_T': float(1 - result.R.sum() - result.T.sum()),
        'r_electric_tangential': [float(r.real), float(r.imag)],
        'admittance_times_Z0': [float(q_in.real), float(q_in.imag)],
    }


def absorbing_finite_layer():
    epsilon = 2.16 + 0.9j
    wavelength, thickness = 500., 75.
    problem = rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(1000., 1000.), top=Material(epsilon), bottom=Material(epsilon),
            layers=(rcwa.Layer(thickness=thickness, background=Material(epsilon)),)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(polarization='s'),)))
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(0, 0)))
    expected_T = np.exp(-4 * np.pi * np.sqrt(epsilon).imag * thickness / wavelength)
    np.testing.assert_allclose(result.R.sum(), 0., atol=2e-12)
    np.testing.assert_allclose(result.outputs['power_entering'][0], 1., atol=2e-12)
    np.testing.assert_allclose(result.T.sum(), expected_T, atol=2e-12)
    np.testing.assert_allclose(result.absorption[0], 1 - expected_T, atol=2e-12)
    return {
        'R': float(result.R.sum()), 'T': float(result.T.sum()),
        'power_entering': float(result.outputs['power_entering'][0]),
        'absorption': float(result.absorption[0]), 'exponential_T': float(expected_T),
    }


if __name__ == '__main__':
    print(json.dumps({
        'lossless_incidence': boundary_case(4.),
        'absorbing_incidence': boundary_case(4. + 3j),
        'absorbing_finite_layer': absorbing_finite_layer(),
    }, indent=2))
