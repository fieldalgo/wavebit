"""An oxide film on silicon with surface roughness and a mixed interface, as ellipsometry models it.

The stack, bottom to top: silicon, a 2 nm interface layer where oxide mixes into silicon
linearly (``rcwa.Graded.interface``, sliced by Wavebit), 100 nm of oxide, and 3 nm of roughness
(``wavebit.materials.Bruggeman``: oxide and 50% void). The script prints the s and p
reflectance with and without the two thin layers, and checks the full stack against an
independent characteristic-matrix calculation (Born and Wolf, sec. 1.6) of the same layers,
with each effective permittivity found by ``numpy.roots``.
"""
import math

import numpy as np

from wavebit import Material, Units, rcwa
from wavebit.materials import Bruggeman

silicon, oxide, air = Material(epsilon=15.2 + 0.16j), Material(epsilon=2.1025), Material(epsilon=1.0)
wavelengths = np.linspace(0.4, 0.8, 9)
theta = math.radians(70.)
INTERFACE, FILM, ROUGH = 0.002, 0.100, 0.003


def problem(layers, polarization):
    return rcwa.Problem(units=Units(length='um'),
                        stack=rcwa.Stack(period=(1., 1.), bottom=silicon, layers=tuple(layers)),
                        excitation=rcwa.Excitation(wavelength=wavelengths,
                                                   sources=(rcwa.PlaneWave(theta=theta, polarization=polarization),)))


interface = rcwa.ProfiledLayer(thickness=INTERFACE, background=rcwa.Graded.interface(below=silicon, above=oxide))
film = rcwa.Layer(thickness=FILM, background=oxide)
rough = rcwa.Layer(thickness=ROUGH, background=Bruggeman(host=oxide, inclusion=air, fraction=0.5))
numerics = rcwa.Numerics(slicing=rcwa.Slicing(count=16))

reflectance = {}
for name, layers in (('film only', [film]), ('with interface and roughness', [interface, film, rough])):
    for polarization in ('s', 'p'):
        result = rcwa.run(problem(layers, polarization), numerics=numerics)
        reflectance[name, polarization] = np.asarray(result.R).reshape(len(wavelengths))
    print(f'{name:30s} Rs {reflectance[name, "s"].round(4)}')
    print(f'{"":30s} Rp {reflectance[name, "p"].round(4)}')


def bruggeman(host, inclusion, fraction):
    b = (2 - 3*fraction)*host + (3*fraction - 1)*inclusion
    return max(np.roots([2, -b, -host*inclusion]), key=lambda r: (round(r.imag, 12), r.real))


def characteristic_matrix(epsilons, thicknesses, wavelength, polarization):
    """Reflectance of uniform films on silicon under air (Born and Wolf, sec. 1.6)."""
    k0, kx2 = 2*math.pi/wavelength, math.sin(theta)**2
    def admittance(eps):
        q = np.sqrt(complex(eps - kx2))
        q = q if q.imag >= 0 else -q
        return (q, q) if polarization == 's' else (q, eps/q)
    m = np.eye(2, dtype=complex)
    for eps, d in zip(epsilons[::-1], thicknesses[::-1]):
        q, y = admittance(eps)
        delta = k0*q*d
        m = m @ np.array([[np.cos(delta), -1j*np.sin(delta)/y], [-1j*y*np.sin(delta), np.cos(delta)]])
    _, y0 = admittance(1.0)
    _, ys = admittance(15.2 + 0.16j)
    b, c = m @ np.array([1., ys])
    return abs((y0*b - c)/(y0*b + c))**2


# The same staircase the solver cuts: 16 interface slices at their midpoint fractions.
fractions = [(k + 0.5)/16 for k in range(16)]
epsilons = [bruggeman(15.2 + 0.16j, 2.1025, u) for u in fractions] + [2.1025, bruggeman(2.1025, 1.0, 0.5)]
thicknesses = [INTERFACE/16]*16 + [FILM, ROUGH]
for polarization in ('s', 'p'):
    expected = [characteristic_matrix(epsilons, thicknesses, w, polarization) for w in wavelengths]
    assert np.max(np.abs(reflectance['with interface and roughness', polarization] - expected)) < 1e-10
print('checked against the characteristic-matrix calculation of the same layers')
