"""Physical tensor materials, rotated axes and an explicit wavelength spectrum."""
import numpy as np
from wavebit import Material, TabulatedMaterial, Units, rcwa


def film(material, wavelength=600., polarization='s'):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(600., 600.), layers=(rcwa.Layer(thickness=220., background=material),)),
        excitation=rcwa.Excitation(wavelength=wavelength,
            sources=(rcwa.PlaneWave(theta=0., phi=0., polarization=polarization),)))


def main():
    numerics = rcwa.Numerics()
    crystal = Material(epsilon=(2.25, 4., 2.25))
    answers = {pol: rcwa.run(film(crystal, polarization=pol), numerics=numerics) for pol in ('s','p')}
    for result in answers.values():
        np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=1e-12)
    print(f'Uniform film: R_s={float(answers["s"].R.sum()):.9f}, R_p={float(answers["p"].R.sum()):.9f}')
    rotated = crystal.rotated(np.deg2rad(30.), axis='z')
    result = rcwa.run(film(rotated), numerics=numerics)
    # Vacuum ports and normal incidence make squared amplitude a power fraction.
    cross_power = np.abs(result.transmission.amplitude_p)**2
    assert cross_power.sum() > 0
    np.testing.assert_allclose(result.R.sum()+result.T.sum(), 1., atol=1e-12)
    print(f'Rotated film: cross-polarized T_p={float(cross_power.sum()):.9f}')
    dispersive = TabulatedMaterial(wavelengths=(500.,700.), wavelength_unit='nm',
        epsilon=((2.,3.,2.5),(3.,4.,2.9)))
    spectrum = rcwa.run(film(dispersive, wavelength=(550.,600.,650.)), numerics=numerics)
    for i,wavelength in enumerate(spectrum.wavelengths):
        np.testing.assert_allclose(spectrum.R[i].sum()+spectrum.T[i].sum(), 1., atol=1e-12)
        print(f'{wavelength:.0f} nm: R_s={float(spectrum.R[i].sum()):.9f}')


if __name__ == '__main__':
    main()
