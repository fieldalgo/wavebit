"""A Lorentz film: physical time records and raw Fourier E/H fields.

Lengths use um, time uses seconds, omega_0/gamma use rad/s, and Lorentz strength
uses s^-2. Fourier fields are integral F(t) exp(+i 2 pi f t) dt: V*s/m or A*s/m,
not monochromatic field amplitudes. Run with --plot /path/pulse.png for a figure
(requires wavebit[plot]); the default run needs only NumPy.
"""
import argparse
from pathlib import Path
import numpy as np

from wavebit import DispersiveMaterial, Lorentz, Units, fdtd

C0 = 299792458.
UNITS = Units(length='um')
WAVELENGTHS = (.72, .8, .9)


def make_problem():
    omega0 = 2 * np.pi * C0 / (.5e-6)
    film = DispersiveMaterial(epsilon_inf=1.5,
        poles=(Lorentz(omega_0=omega0, gamma=.08*omega0, strength=.8*omega0**2),))
    problem = fdtd.Problem(units=UNITS,
        scene=fdtd.Scene(domain=fdtd.Domain(size=(.1, .1, 1.6)),
            boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
            structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(0., .2)), material=film),)),
        sources=(fdtd.PlaneWave('pulse', plane=fdtd.Plane(z=.6),
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=3e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.TimeSeriesMonitor('time_Ey', at=(0., 0., -.3), component='Ey'),
            fdtd.FieldMonitor('fourier_fields', plane=fdtd.Plane(z=-.3),
                wavelengths=WAVELENGTHS, components=('Ey', 'Hx')),
            fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=.4), wavelengths=WAVELENGTHS,
                source='pulse', wave='reflected', orders=(0, 0), reference_plane=.2, incident_plane=.2),
            fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-.3), wavelengths=WAVELENGTHS,
                source='pulse', wave='transmitted', orders=(0, 0), reference_plane=0., incident_plane=.2),
        )))
    return problem, film


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', type=Path, help='optional figure output path')
    args = parser.parse_args()
    problem, film = make_problem()
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
        duration=fdtd.Duration(seconds=100e-15))
    prepared = fdtd.prepare(problem, numerics=numerics)
    prepared.report.raise_if_invalid()
    result = prepared.run(problem)
    fields = result['fourier_fields']
    time = result['time_Ey']
    assert bool(result.validity.checks['runtime_stability'])
    assert time.units == {'Ey': 'V/m', 'coordinates': 'um'}
    assert fields.units == {'Ey': 'V*s/m', 'Hx': 'A*s/m', 'coordinates': 'um'}
    assert all(np.all(np.isfinite(v)) and np.max(np.abs(v)) > 0 for v in fields.values.values())

    # The shared physical model is evaluated at the same vacuum wavelengths.
    epsilon = np.asarray(film.at(np.asarray(WAVELENGTHS), units=UNITS).epsilon)
    assert np.all(epsilon.imag > 0)  # Passive under exp(-i omega t).
    reflection = np.asarray(result['r'].efficiency)[:, 0, 0]
    transmission = np.asarray(result['t'].efficiency)[:, 0, 0]
    absorption = 1 - reflection - transmission
    assert np.all(result['r'].metadata['source_usable']) and np.all(result['t'].metadata['source_usable'])
    assert np.all(absorption > 0) and np.all(absorption < 1)

    # Reuse the fixed preparation with complete live inputs: Maxwell linearity
    # scales both physical time records and raw electric/magnetic transforms.
    half = problem.replace(sources=(problem.sources[0].replace(amplitude=.5),))
    scaled = prepared.run(half)
    np.testing.assert_allclose(scaled['time_Ey'].values, .5*time.values, rtol=2e-12, atol=1e-15)
    for component in fields.components:
        np.testing.assert_allclose(scaled['fourier_fields'].values[component],
            .5*fields.values[component], rtol=2e-12, atol=1e-28)
    print('wavelength (um)   epsilon                      R          T          A')
    for wavelength, eps, r, t, a in zip(WAVELENGTHS, epsilon, reflection, transmission, absorption):
        print(f'{wavelength:15.2f}   {eps.real:8.5f}+{eps.imag:.5f}j   {r:.6f}   {t:.6f}   {a:.6f}')
    print(f'{len(time.times_seconds)} samples; time in s, Ey in V/m; Fourier field units: {dict(fields.units)}')
    print('Half-amplitude reuse agrees for the time record and both Fourier components.')
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.asarray(time.times_seconds)*1e15, np.asarray(time.values))
        ax.set(xlabel='Time (fs)', ylabel='Transmitted Ey (V/m)', title='Pulse through a Lorentz film')
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(args.plot)


if __name__ == '__main__':
    main()
