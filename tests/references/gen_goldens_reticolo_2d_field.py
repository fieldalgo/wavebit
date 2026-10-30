"""Freeze RETICOLO's in-layer field of a **crossed grating** into ``golden_reticolo_2d_field.npz``.

Run once, on a machine with Octave and a RETICOLO V10 tree; ``test_reticolo_2d_field_golden.py``
then runs anywhere.

    RETICOLO_PATH=... python tests/references/gen_goldens_reticolo_2d_field.py
    RETICOLO_PATH=... python tests/references/gen_goldens_reticolo_2d_field.py off-grid

Rerun it only to add cases -- never to make a failing test pass. ``off-grid`` writes
``golden_reticolo_2d_field_off_grid.npz`` alone: the same stack with walls on no pixel line, for
a shape structure, under ``champ = 1`` and ``champ = 0``.

**What it stores.** ``res3``'s raw cell for every case of ``_reticolo_2d_field_cases.py``, under
**both** of RETICOLO's in-plane reconstructions: ``champ = 1`` (``E`` normal to a wall rebuilt
from the continuous ``D`` series, line by line, then divided by ``eps`` -- P. Lalanne and M. P.
Jurek, J. Mod. Opt. 45, 1357 (1998)) and ``champ = 0`` (the plain Fourier sum). They pin this
project's ``discontinuous='continuity'`` and its default ``'fourier'`` respectively. ``champ``
changes nothing but the field: the solve and every other component are the same.

**How it runs RETICOLO.** Through ``octave`` as a subprocess on ``interface/reticolo_res3_2d.m``,
with the case written to a ``.mat`` and the cell read back from one -- a 2D texture is a cell
array of cell arrays, which the ``.m`` builds in RETICOLO's own language from plain arrays.

**Citation.** RETICOLO's terms require Hugonin & Lalanne (arXiv:2101.00901), and Lalanne & Jurek
(above) whenever near-field distributions are shown -- which a field golden is. See
``README.md``.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE.parents[1]))

OUT = HERE / 'golden_reticolo_2d_field.npz'
OUT_OFF_GRID = HERE / 'golden_reticolo_2d_field_off_grid.npz'


def _run(reticolo, angle, champ, layers=None, rows=None):
    import scipy.io as sio
    from tests.references._reticolo_2d_field_cases import (
        EINC, LAYERS, MAX_ORDER, NPTS, N_BOT, N_TOP, PERIOD, WL, blocks, samples)
    layers = LAYERS if layers is None else layers
    rows = blocks if rows is None else rows
    theta, phi = angle
    x, y = samples()
    # Octave's `sparse` refuses integer arrays, so every count crosses as a double.
    case = dict(wavelength=WL, period=np.array(PERIOD), nn=np.array([MAX_ORDER, MAX_ORDER], float),
                k_parallel=N_TOP * np.sin(np.deg2rad(theta)), delta0=phi, champ=float(champ),
                n_top=N_TOP, n_bot=N_BOT, n_layers=float(len(layers)),
                thickness=np.array([layer['thickness'] for layer in layers]),
                x=x, y=y, npts=float(NPTS), n_einc=float(len(EINC)))
    for k, layer in enumerate(layers, 1):
        case[f'bg_{k}'] = layer['background']
        case[f'blocks_{k}'] = rows(layer)
    for k, pol in enumerate(EINC, 1):
        case[f'einc_{k}'] = np.array(EINC[pol], dtype=complex)
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = os.path.join(tmp, 'in.mat'), os.path.join(tmp, 'out.mat')
        sio.savemat(infile, case)
        script = (f"addpath(genpath('{reticolo}')); addpath('{HERE / 'interface'}'); "
                  f"reticolo_res3_2d('{infile}', '{outfile}');")
        subprocess.run(['octave', '--no-gui', '--quiet', '--eval', script], check=True)
        got = sio.loadmat(outfile)
    return {pol: got[f'e_{k}'] for k, pol in enumerate(EINC, 1)}, got['z'].ravel(), got['o']


def main():
    reticolo = os.environ.get('RETICOLO_PATH')
    if not reticolo:
        raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                         'golden; the committed .npz is what the tests read.')
    reticolo = os.path.expanduser(reticolo)
    from tests.references._reticolo_2d_field_cases import (
        ANGLES, OFF_GRID_ANGLE, OFF_GRID_LAYERS, key, off_grid_blocks, samples)

    if sys.argv[1:] == ['off-grid']:
        out = {}
        for champ in (1, 0):
            cells, z, o = _run(reticolo, OFF_GRID_ANGLE, champ, OFF_GRID_LAYERS, off_grid_blocks)
            for pol, e in cells.items():
                out[f'champ{champ}__{key(OFF_GRID_ANGLE, pol)}'] = e
        out['z'], out['o'] = z, o
        out['x'], out['y'] = samples()
        np.savez_compressed(str(OUT_OFF_GRID), **out)
        print(f'wrote {len(out)} arrays to {OUT_OFF_GRID.name}; cell {e.shape}')
        return

    out = {}
    for angle in ANGLES:
        for champ in (1, 0):
            cells, z, o = _run(reticolo, angle, champ)
            for pol, e in cells.items():
                out[f'champ{champ}__{key(angle, pol)}'] = e      # raw (z, x, y, 6)
        out['z'], out['o'] = z, o
    out['x'], out['y'] = samples()
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}; cell {e.shape}')


if __name__ == '__main__':
    main()
