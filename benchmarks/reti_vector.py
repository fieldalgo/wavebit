"""Reticolo cross-check of the *vector* (shape-based) path.

The other scripts here compare rasters. This one compares the block lists the vector modeler
builds, which is what nothing else validates: the suite's own references for that path are
either the legacy solver's own output or geometry measured against the shape's equation.
Reticolo is the only oracle in the project that independently validates diffraction orders.

Two comparisons, because they answer different questions:

* ``run_1d`` — a binary grating whose edges land exactly on the texture boundaries, so neither
  side approximates anything. It exercises ``vector_per_layer_numeric``: the edge list, the
  grid-line sharing, the orders.
* ``run_2d`` — a rotated rectangle and a rotated ellipse, with Reticolo handed the *same*
  blocks the modeler produced. Reticolo takes rectangles natively, so the geometry is identical
  on both sides and what is left under test is the factorization and the solve. Handing it a
  pixel raster instead would pit two different approximations of the ellipse against each
  other, and at a raster Octave will chew through in reasonable time — 32x32 is a 9.4-unit
  pixel against a 30-unit semi-axis — the staircase swamps what is being measured.

Needs Octave and the **V9_pristine** copy of Reticolo; the modified ``V9`` beside it fails
inside ``retb`` under Octave 9.4.

Reading the 2D output: 46 of its 48 rows land at 1e-13. Two do not — the ellipse at 1.9 rad
under normal incidence, at 3e-8, which is 6e-7 of the efficiencies involved. It is isolated in
``n_split`` rather than systematic (7e-15 / 3e-14 / **3e-8** / 9e-14 / 5e-14 at 6 / 10 / 14 /
20 / 30) and the grid it comes from is not the finest in that sweep, so it reads as one
structure's conditioning rather than a property of the decomposition.

It is Reticolo's. Sliding the shape along x cannot change an efficiency, so how far a code
moves under that translation is its own error scale on the structure — and at ``n_split`` 14
Reticolo moves 6.0e-8 while FieldWave moves 1.6e-14, the disagreement being the same size as
Reticolo's own wobble. At 10 and 20 both sit at 1e-14. FieldWave also conserves energy to
3e-15 throughout. Read the two rows as the oracle's floor on that texture, not as a finding
about the modeler.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

import fieldwave
from fieldwave._core.modeler import Modeling

RETICOLO_PATH = '<a path outside this repository>'
PERIOD = 300.0
N_BASE, N_ROD = 1.0, 3.5
THICKNESS = [205.0]
WAVELENGTH = 900.0


def _reticolo():
    from oct2py import octave
    octave.addpath(octave.genpath(RETICOLO_PATH))
    from interface.Reticolo import Reticolo
    return Reticolo()


def _scalar(v):
    return float(np.real(np.asarray(v).reshape(-1)[0]))


def run_1d(fto=10):
    """Binary grating, exact on both sides. Returns the largest |FieldWave - Reticolo|."""
    reti = _reticolo()
    segments = np.array([[[N_BASE, N_ROD, N_ROD, N_BASE]]], dtype=complex)   # edges at 75 / 225
    worst = 0.0

    print(f'{"pol":>4} {"theta":>6} {"":>10} {"sum de_ri":>14} {"sum de_ti":>14}')
    for pol, name in ((0, 'TE'), (1, 'TM')):
        for theta in (0.0, 0.2):
            r, t, _, _ = reti.run_res2(grating_type=0, period=[PERIOD], fto=fto,
                                       ucell=segments, thickness=THICKNESS, theta=theta,
                                       phi=0.0, pol=pol, wavelength=WAVELENGTH,
                                       n_top=1.0, n_bot=1.0)
            want = (float(np.sum(np.asarray(r))), float(np.sum(np.asarray(t))))

            ucell = [[N_BASE, [['rectangle', PERIOD / 2, PERIOD / 2, 150.0, PERIOD,
                                N_ROD, 0, 0, 0]]]]
            res = fieldwave.rcwa(backend=0, period=[PERIOD, PERIOD], fto=[fto, 0], ucell=ucell,
                                 thickness=THICKNESS, theta=theta, phi=0.0, pol=pol,
                                 wavelength=WAVELENGTH, n_top=1.0, n_bot=1.0).conv_solve()
            got = (float(np.sum(np.asarray(res.de_ri))), float(np.sum(np.asarray(res.de_ti))))

            print(f'{name:>4} {theta:>6} {"reticolo":>10} {want[0]:>14.10f} {want[1]:>14.10f}')
            print(f'{"":>4} {"":>6} {"fieldwave":>10} {got[0]:>14.10f} {got[1]:>14.10f}')
            worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
    return worst


def run_2d(fto=5):
    """Rotated shapes, Reticolo given the modeler's own blocks. Returns the largest |diff|.

    Period 900 against a 500 wavelength, so nine to eleven orders propagate and the comparison
    is per order rather than on a total that cannot move far. Reticolo returns its efficiencies
    in a different layout, so the two are matched as sorted spectra — which still pins every
    order's value, just not its label.
    """
    reti = _reticolo()
    period, wavelength, thickness = 900.0, 500.0, [300.0]
    cases = [('rectangle', (450.0, 450.0, 400.0, 260.0, N_ROD), 6),
             ('ellipse', (450.0, 450.0, 460.0, 300.0, N_ROD), 14)]
    worst = 0.0

    print(f'{"shape":>10} {"angle":>6} {"pol":>4} {"theta":>6} {"phi":>5} {"orders":>7} '
          f'{"max |diff|":>11}')
    for shape, args, n_split in cases:
        for angle in (0.0, 0.37, -0.9, 1.9):
            raw = getattr(Modeling(period=[period, period]), shape)(*args, angle,
                                                                   n_split, n_split)
            blocks = [(_scalar(b[0][0]), _scalar(b[0][1]), _scalar(b[1][0]), _scalar(b[1][1]))
                      for b in raw]
            rects = [[N_BASE, 0, 0, 0, 0, 0]]
            for y_lo, x_lo, y_hi, x_hi in blocks:
                rects.append([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2,
                              x_hi - x_lo, y_hi - y_lo, N_ROD, 1])
            textures = [1.0, np.array(rects, dtype=complex), 1.0]
            profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])

            for pol, name in ((0, 'TE'), (1, 'TM')):
                for theta, phi in ((0.0, 0.0), (0.21, 0.0), (0.21, 0.7)):
                    r, t, _, _ = reti._run(pol, theta, phi, [period, period], 1.0,
                                           [fto, fto], textures, profile, wavelength,
                                           grating_type=2)
                    ucell = [[N_BASE, [[shape, *args, angle, n_split, n_split]]]]
                    res = fieldwave.rcwa(backend=0, period=[period, period], fto=[fto, fto],
                                         ucell=ucell, thickness=thickness, theta=theta,
                                         phi=phi, pol=pol, wavelength=wavelength,
                                         n_top=1.0, n_bot=1.0).conv_solve()

                    diff, orders = 0.0, 0
                    for want, got in ((r.efficiency, res.de_ri), (t.efficiency, res.de_ti)):
                        want = np.sort(np.asarray(want, dtype=float).ravel())[::-1]
                        got = np.sort(np.asarray(got, dtype=float).ravel())[::-1]
                        orders = min((np.abs(want) > 1e-13).sum(), (np.abs(got) > 1e-13).sum())
                        diff = max(diff, float(np.max(np.abs(want[:orders] - got[:orders]))),
                                   abs(want.sum() - got.sum()))
                    worst = max(worst, diff)
                    print(f'{shape:>10} {angle:>6.2f} {name:>4} {theta:>6.2f} {phi:>5.2f} '
                          f'{orders:>7} {diff:>11.2e}', flush=True)
    return worst


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    print('== 1D, exact geometry on both sides ==')
    worst_1d = run_1d()
    print(f'worst |FieldWave - Reticolo|: {worst_1d:.3e}\n')
    print('== 2D, Reticolo given the modeler\'s blocks ==')
    worst_2d = run_2d()
    print(f'worst |FieldWave - Reticolo|: {worst_2d:.3e}')
