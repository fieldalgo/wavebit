"""Reflection spectrum of a tapered silicon line grating, swept over its sidewall angle.

An OCD-style line: silicon on silicon, period 120 nm, height 100 nm, mid-height CD 45 nm,
with 5 nm of top rounding. ``rcwa.ProfiledLayer`` slices the profile itself; this script
reads how many slices it used and the estimated error of the z staircase, and checks

* the power balance R + T + A = 1 of every spectrum,
* a vertical wall (sidewall angle pi/2, no rounding) against the same line written as one
  Rectangle layer, which it must equal to the bit,
* the staircase error estimate against the 2e-3 this sweep asks for.

Pass ``--plot`` to draw the spectra (needs matplotlib).
"""
import argparse
import math

import numpy as np

from wavebit import Units, rcwa
from wavebit.materials import catalog

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--plot', action='store_true', help='draw R(wavelength) for each sidewall angle')
args = parser.parse_args()

silicon = catalog.load('Si')
wavelengths = np.linspace(300., 800., 26)
period, height, cd = 120., 100., 45.


def problem(swa, top_rounding=5.):
    line = rcwa.Trapezoid(cd=cd, cd_at='middle', swa=swa, top_rounding=top_rounding)
    layer = rcwa.ProfiledLayer(thickness=height, regions=(rcwa.Region(geometry=line, material=silicon),))
    return rcwa.Problem(units=Units(length='nm'),
                        stack=rcwa.Stack(period=(period, period), bottom=silicon, layers=(layer,)),
                        excitation=rcwa.Excitation(wavelength=wavelengths,
                                                   sources=(rcwa.PlaneWave(theta=math.radians(65.), polarization='p'),)))


# A line's slices are full-height rectangles, on which Li's factorization is exact and fast.
numerics = rcwa.Numerics(orders=(8, 0), factorization='li', slicing=rcwa.Slicing(count=16))

spectra = {}
for degrees in (82., 86., 90.):
    result = rcwa.run(problem(math.radians(degrees)), numerics=numerics)
    report = result.diagnostics['z_slicing'][0]
    reflectance = np.asarray(result.R).sum(axis=(1, 2))
    balance = reflectance + np.asarray(result.T).sum(axis=(1, 2)) + np.asarray(result.outputs['absorption'])
    print(f'SWA {degrees:4.1f} deg: {report["count"]} slices, staircase error ~ {report["staircase_error"]:.1e}, '
          f'R from {reflectance.min():.3f} to {reflectance.max():.3f}')
    assert np.max(np.abs(balance - 1.)) < 1e-9
    assert report['staircase_error'] < 2e-3
    spectra[degrees] = reflectance

# A vertical wall without rounding is the Rectangle layer, to the bit.
vertical = rcwa.run(problem(math.pi/2, top_rounding=0.), numerics=numerics)
rectangle = rcwa.Layer(thickness=height, regions=(
    rcwa.Region(geometry=rcwa.Rectangle(size=(cd, period)), material=silicon),))
by_hand = rcwa.run(rcwa.Problem(units=Units(length='nm'),
                                stack=rcwa.Stack(period=(period, period), bottom=silicon, layers=(rectangle,)),
                                excitation=problem(math.pi/2).excitation), numerics=numerics)
assert np.array_equal(vertical.R, by_hand.R) and vertical.diagnostics['z_slicing'][0]['merged']
print('vertical wall: identical to the Rectangle layer')

if args.plot:
    import matplotlib.pyplot as plt
    for degrees, reflectance in spectra.items():
        plt.plot(wavelengths, reflectance, label=f'SWA {degrees:g} deg')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('R, p polarization, 65 deg incidence')
    plt.legend()
    plt.show()
