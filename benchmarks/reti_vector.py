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


def run_2d(fto=6):
    """Rotated shapes, Reticolo given the modeler's own blocks. Returns the largest |diff|."""
    reti = _reticolo()
    cases = [('rectangle', (150.0, 150.0, 80.0, 60.0, N_ROD, 0.4), 4),
             ('ellipse', (150.0, 150.0, 90.0, 60.0, N_ROD, 0.4), 12)]
    worst = 0.0

    print(f'{"shape":>10} {"pol":>4} {"blocks":>7} {"":>10} {"sum de_ri":>14} {"sum de_ti":>14}')
    for shape, args, n_split in cases:
        raw = getattr(Modeling(period=[PERIOD, PERIOD]), shape)(*args, n_split, n_split)
        blocks = [(_scalar(b[0][0]), _scalar(b[0][1]), _scalar(b[1][0]), _scalar(b[1][1]))
                  for b in raw]

        rects = [[N_BASE, 0, 0, 0, 0, 0]]
        for y_lo, x_lo, y_hi, x_hi in blocks:
            rects.append([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2,
                          x_hi - x_lo, y_hi - y_lo, N_ROD, 1])
        textures = [1.0, np.array(rects, dtype=complex), 1.0]
        profile = np.array([[0, *THICKNESS, 0], range(1, len(THICKNESS) + 3)])

        for pol, name in ((0, 'TE'), (1, 'TM')):
            r, t, _, _ = reti._run(pol, 0.0, 0.0, [PERIOD, PERIOD], 1.0, [fto, fto],
                                   textures, profile, WAVELENGTH, grating_type=2)
            want = (float(np.sum(np.asarray(r.efficiency))),
                    float(np.sum(np.asarray(t.efficiency))))

            ucell = [[N_BASE, [[shape, *args, n_split, n_split]]]]
            res = fieldwave.rcwa(backend=0, period=[PERIOD, PERIOD], fto=[fto, fto],
                                 ucell=ucell, thickness=THICKNESS, theta=0.0, phi=0.0,
                                 pol=pol, wavelength=WAVELENGTH, n_top=1.0,
                                 n_bot=1.0).conv_solve()
            got = (float(np.sum(np.asarray(res.de_ri))), float(np.sum(np.asarray(res.de_ti))))

            print(f'{shape:>10} {name:>4} {len(blocks):>7} {"reticolo":>10} '
                  f'{want[0]:>14.10f} {want[1]:>14.10f}')
            print(f'{"":>10} {"":>4} {"":>7} {"fieldwave":>10} {got[0]:>14.10f} {got[1]:>14.10f}')
            worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
    return worst


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    print('== 1D, exact geometry on both sides ==')
    worst_1d = run_1d()
    print(f'worst |FieldWave - Reticolo|: {worst_1d:.3e}\n')
    print('== 2D, Reticolo given the modeler\'s blocks ==')
    worst_2d = run_2d()
    print(f'worst |FieldWave - Reticolo|: {worst_2d:.3e}')
