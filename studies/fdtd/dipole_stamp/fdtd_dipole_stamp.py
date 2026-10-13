"""Which stamp a ``wavebit.fdtd._legacy.Dipole`` spreads its current with: tent or cubic B-spline.

Both stamps have half-width 2 cells, touch 3-4 points per axis, sum to exactly 1 and
reproduce linear functions, so the moment and the centroid are exact for both. What differs
is how much of the answer depends on where ``center`` sits inside its cell -- a stamp with
more high-wavenumber content couples more to the grid's anisotropic short waves and radiates
a position-dependent power. The probe: an electric ``Ez`` dipole at the centre of an open
vacuum box (3 wavelengths, a 12-cell PML all round), 16 cells per wavelength at the band
centre (``Band(wavelength=1, df=0.15)``), moved along the cell diagonal by 0, 1/4 and 1/2
cell. The quantity is the radiation resistance at three frequencies of the band,

    R(f) = -Re(conj(D) F) / |D|^2 * cos(w dt / 2)

with ``D`` the DFT of the drive series and ``F`` that of the stamp-weighted field, each over
its own sample times (``res.sources[name]``); the ``cos`` is the one the work series' mean of
two E levels puts in. ``spread`` is ``max R / min R - 1`` over the three positions; the
continuum value ``R = k^2 / (6 pi)`` (the time-average ``P = k^2 |I l|^2 / (12 pi)`` over
``|I l|^2 / 2``) is printed beside it.

Measured 2026-09-19, CPU, NumPy (``studies/fdtd/dipole_stamp/records/fdtd_dipole_stamp.log``)::

    R at f/f0 = 0.8 / 1.0 / 1.2      continuum 1.3404 / 2.0944 / 3.0159
    tent     offset 0.00 / 0.25 / 0.50:  1.2813 / 1.9545 / 2.7291 ... 1.2621 / 1.9092 / 2.6379
    tent     spread: 1.5e-2 / 2.4e-2 / 3.5e-2
    bspline  offset 0.00 / 0.25 / 0.50:  1.3064 / 2.0210 / 2.8648 ... 1.3060 / 2.0208 / 2.8642
    bspline  spread: 3.2e-4 / 1.2e-4 / 2.0e-4

**The B-spline wins by 100x on position dependence** (1.2e-4 against 2.4e-2 at the band
centre) and is closer to the continuum as well (-3.5% against -8.8% at f0, the rest being
the grid's dispersion at 16 cells per wavelength). The tent's kink at every node is a
``1/k^2`` spectral tail; the B-spline's is ``1/k^4``. ``fd.Dipole`` takes it by default.

Run from the repository root: ``python studies/fdtd/dipole_stamp/fdtd_dipole_stamp.py``. NumPy, a few seconds
per run. Results in the module docstring of ``src/wavebit/fdtd/_kernel/sources.py``.
"""
import math
import sys
import warnings

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import wavebit.fdtd._legacy as fd                    # noqa: E402

CPW = 16
WAVELENGTH = 1.0
OFFSETS = (0.0, 0.25, 0.5)
FREQS = (0.8, 1.0, 1.2)                     # in units of f0


def resistance(stamp, offset):
    dx = WAVELENGTH / CPW
    band = fd.Band(wavelength=WAVELENGTH, df=0.15)
    c = offset * dx
    scene = fd.Scene(domain=fd.Domain(size=(3.0, 3.0, 3.0)),
                     boundaries=fd.Boundaries.open(fd.PML(cells=12)),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(periods=14),
                     sources=(fd.Dipole('d', center=(c, c, c), component='Ez', band=band,
                                        stamp=stamp),))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fd.run(scene)
    s = res.sources['d']
    dt = res.plan.dt
    out = []
    for f in FREQS:
        w = 2 * math.pi * f / WAVELENGTH
        D = np.sum(np.asarray(s.drive) * np.exp(1j * w * s.t))
        F = np.sum(np.asarray(s.field) * np.exp(1j * w * s.t_field))
        out.append(-np.real(np.conj(D) * F) / abs(D) ** 2 * math.cos(w * dt / 2))
    return out


def main():
    print(f'{CPW} cells per wavelength; R at f/f0 = {FREQS}; continuum k^2/(6 pi) = '
          + ', '.join(f'{(2 * math.pi * f) ** 2 / (6 * math.pi):.4f}' for f in FREQS))
    for stamp in ('tent', 'bspline'):
        rows = [resistance(stamp, o) for o in OFFSETS]
        for o, r in zip(OFFSETS, rows):
            print(f'{stamp:8s} offset {o:4.2f}: ' + ', '.join(f'{v:.5f}' for v in r))
        spread = [max(col) / min(col) - 1 for col in zip(*rows)]
        print(f'{stamp:8s} spread over positions: ' + ', '.join(f'{v:.2e}' for v in spread))


if __name__ == '__main__':
    main()
