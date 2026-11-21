"""The Psi/Delta spectrum of an SiO2 film on silicon, as an ellipsometer reports it.

One ``rcwa.run_ellipsometry`` call solves the film s- and p-polarized on one preparation and
returns the zeroth-order Jones matrix, ``psi``/``delta`` and the Mueller matrix. ``delta`` is in
the instrument convention (``tan(psi) exp(i delta) = conj(r_p / r_s)``, ``[0, 2 pi)``), so the
numbers compare with a measurement directly; the native ``exp(-i omega t)`` ratio is ``rho``.
The film is checked against an independent Airy formula written with the same tabulated
refractive indices, and the Mueller matrix against the standard isotropic form.
"""
import numpy as np

from wavebit import Material, Units, materials, rcwa
from wavebit.ellipsometry import depolarization_index

ANGLE = np.radians(70.)
WAVELENGTHS = np.arange(300., 1001., 50.)          # nm
THICKNESS = 100.                                   # nm of SiO2


def fresnel(n_in, n_out, cos_in, cos_out, polarization):
    """Nebraska-convention amplitude coefficients: ``r_p = -r_s`` at normal incidence."""
    if polarization == 's':
        return (n_in * cos_in - n_out * cos_out) / (n_in * cos_in + n_out * cos_out)
    return (n_out * cos_in - n_in * cos_out) / (n_out * cos_in + n_in * cos_out)


def airy_rho(wavelength, n_film, n_substrate, thickness, theta):
    """``r_p / r_s`` of one film from the Airy sum, ``exp(-i omega t)``: the round trip is ``exp(+2i beta)``."""
    sin_t = np.sin(theta)
    cos_0 = np.cos(theta)
    cos_1 = np.sqrt(1 - (sin_t / n_film) ** 2 + 0j)
    cos_2 = np.sqrt(1 - (sin_t / n_substrate) ** 2 + 0j)
    cos_2 = np.where((n_substrate * cos_2).imag < 0, -cos_2, cos_2)
    phase = np.exp(2j * (2 * np.pi / wavelength) * n_film * thickness * cos_1)
    r = {}
    for polarization in ('s', 'p'):
        r01 = fresnel(1., n_film, cos_0, cos_1, polarization)
        r12 = fresnel(n_film, n_substrate, cos_1, cos_2, polarization)
        r[polarization] = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    return r['p'] / r['s']


def main():
    silicon, silica = materials.load('Si'), materials.load('SiO2')
    problem = rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=(300., 300.), top=Material(epsilon=1.), bottom=silicon,
                         layers=(rcwa.Layer(thickness=THICKNESS, background=silica),)),
        excitation=rcwa.Excitation(wavelength=WAVELENGTHS,
                                   sources=(rcwa.PlaneWave(theta=ANGLE, phi=0.),)))
    result = rcwa.run_ellipsometry(problem, numerics=rcwa.Numerics(orders=(0, 0)))
    angles = result.reflection.ellipsometry()          # zeroth order
    mueller = result.reflection.mueller()

    # Independent check: the same tabulated indices through the Airy formula.
    n_si = np.sqrt(np.asarray(silicon.at(WAVELENGTHS).epsilon))
    n_ox = np.sqrt(np.asarray(silica.at(WAVELENGTHS).epsilon))
    rho = airy_rho(WAVELENGTHS, n_ox, n_si, THICKNESS, ANGLE)
    np.testing.assert_allclose(angles.rho, rho, rtol=0, atol=1e-12)
    np.testing.assert_allclose(angles.psi, np.arctan(np.abs(rho)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * angles.delta), np.conj(rho) / np.abs(rho), rtol=0, atol=1e-12)
    # The standard isotropic Mueller form, with this delta's sign.
    n, c, s = np.cos(2 * angles.psi), np.sin(2 * angles.psi) * np.cos(angles.delta), np.sin(2 * angles.psi) * np.sin(angles.delta)
    np.testing.assert_allclose(mueller[:, 2, 3], s, atol=1e-13)
    np.testing.assert_allclose(mueller[:, 0, 1], -n, atol=1e-13)
    np.testing.assert_allclose(mueller[:, 2, 2], c, atol=1e-13)
    np.testing.assert_allclose(depolarization_index(mueller), 1., atol=1e-12)

    print(f'{THICKNESS:.0f} nm SiO2 on Si at {np.degrees(ANGLE):.0f} deg (instrument convention)')
    print('  nm     Psi/deg  Delta/deg')
    for wavelength, psi, delta in zip(WAVELENGTHS, np.degrees(angles.psi), np.degrees(angles.delta)):
        print(f'{wavelength:5.0f}  {psi:8.3f}  {delta:9.3f}')
    return angles


if __name__ == '__main__':
    main()
