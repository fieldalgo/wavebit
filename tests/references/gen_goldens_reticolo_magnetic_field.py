"""Freeze RETICOLO's in-cell field with a **patterned permeability** into ``golden_reticolo_magnetic_field.npz``.

Run once, on a machine with Octave and a RETICOLO V10 tree;
``test_reticolo_magnetic_field_golden.py`` then runs anywhere.

    RETICOLO_PATH=... python tests/references/gen_goldens_reticolo_magnetic_field.py

Rerun it only to add cases -- never to make a failing test pass.

**What it stores.** ``res3``'s raw cell on each path of ``_reticolo_magnetic_field_cases.py``, under
``champ = 1`` and ``champ = 0``. ``champ = 1`` is P. Lalanne and M. P. Jurek, J. Mod. Opt. 45, 1357
(1998) on both fields: RETICOLO's ``retcouche`` builds ``fhx`` and ``fhy`` from ``mu`` as it
builds ``fex`` and ``fey`` from ``eps``, and ``retchamp`` divides the sum by ``mu`` at the sample.

**How it runs RETICOLO.** Through ``octave`` as a subprocess on
``interface/reticolo_res3_magnetic.m``, as ``gen_goldens_reticolo_2d_field.py`` does.

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

OUT = HERE / 'golden_reticolo_magnetic_field.npz'


def _run(reticolo, case):
    import scipy.io as sio
    from tests.references._reticolo_magnetic_field_cases import NPTS, change_index
    case = dict(case, npts=float(NPTS))
    rows = change_index()
    case['n_ci'] = float(len(rows))
    for k, row in enumerate(rows, 1):
        case[f'ci_{k}'] = row
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = os.path.join(tmp, 'in.mat'), os.path.join(tmp, 'out.mat')
        sio.savemat(infile, case)
        script = (f"addpath(genpath('{reticolo}')); addpath('{HERE / 'interface'}'); "
                  f"reticolo_res3_magnetic('{infile}', '{outfile}');")
        subprocess.run(['octave', '--no-gui', '--quiet', '--eval', script], check=True,
                       stdin=subprocess.DEVNULL)
        return sio.loadmat(outfile)


def _two_d(layers, rows, champ):
    from tests.references._reticolo_magnetic_field_cases import (
        ANGLE_2D, EINC, N_BOT, N_TOP, ORDER_2D, PERIOD_2D, WL, samples_2d)
    x, y = samples_2d()
    case = dict(path='2d', wavelength=WL, period=np.array(PERIOD_2D), nn=np.array([ORDER_2D, ORDER_2D], float),
                k_parallel=N_TOP * np.sin(np.deg2rad(ANGLE_2D[0])), delta0=ANGLE_2D[1], champ=float(champ),
                n_top=N_TOP, n_bot=N_BOT, n_layers=float(len(layers)),
                thickness=np.array([layer['thickness'] for layer in layers]),
                x=x, y=y, n_einc=float(len(EINC)))
    for k, layer in enumerate(layers, 1):
        case[f'bg_{k}'] = layer['background']
        case[f'blocks_{k}'] = rows(layer)
    for k, pol in enumerate(EINC, 1):
        case[f'einc_{k}'] = np.array(EINC[pol], dtype=complex)
    got = _run(os.environ['RETICOLO_PATH'], case)
    return {pol: got[f'e_{k}'] for k, pol in enumerate(EINC, 1)}, got['z'].ravel(), got['o']


def _one_d(path, champ):
    from tests.references._reticolo_magnetic_field_cases import (
        ANGLE_1D, ANGLE_CONICAL, EINC, N_BOT, N_TOP, ORDER_1D, PERIOD_1D, STACK_1D, WL, samples_1d)
    conical = path == 'conical'
    theta = ANGLE_CONICAL[0] if conical else ANGLE_1D
    case = dict(path=path, wavelength=WL, period=PERIOD_1D, nn=float(ORDER_1D),
                k_parallel=N_TOP * np.sin(np.deg2rad(theta)), delta0=ANGLE_CONICAL[1], champ=float(champ),
                n_top=N_TOP, n_bot=N_BOT, n_layers=float(len(STACK_1D)),
                thickness=np.array([layer['thickness'] for layer in STACK_1D]),
                x=samples_1d(), y=0.0, n_einc=float(len(EINC)))
    if conical:
        case['nn'] = np.array([float(ORDER_1D), 0.0])
    for k, layer in enumerate(STACK_1D, 1):
        x0, x1, n = layer['bar']
        case[f'edges_{k}'] = np.array([x0, x1, PERIOD_1D])
        case[f'n_{k}'] = np.array([layer['background'], n, layer['background']])
    for k, pol in enumerate(EINC, 1):
        case[f'einc_{k}'] = np.array(EINC[pol], dtype=complex)
    got = _run(os.environ['RETICOLO_PATH'], case)
    if conical:
        return {pol: got[f'e_{k}'] for k, pol in enumerate(EINC, 1)}, got['z'].ravel(), got['o']
    return {'s' if path == 'te' else 'p': got['e_1']}, got['z'].ravel(), got['o']


def main():
    if not os.environ.get('RETICOLO_PATH'):
        raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                         'golden; the committed .npz is what the tests read.')
    os.environ['RETICOLO_PATH'] = os.path.expanduser(os.environ['RETICOLO_PATH'])
    from tests.references._reticolo_magnetic_field_cases import (
        GRID_2D, STACK_2D, grid_blocks, key, samples_1d, samples_2d, shape_blocks)
    out = {}
    for champ in (1, 0):
        for path, cells_z_o in (('shapes', _two_d(STACK_2D, shape_blocks, champ)),
                                ('raster', _two_d(GRID_2D, grid_blocks, champ)),
                                ('te', _one_d('te', champ)), ('tm', _one_d('tm', champ)),
                                ('conical', _one_d('conical', champ))):
            cells, z, o = cells_z_o
            for pol, e in cells.items():
                out[key(path, pol, champ)] = e
            out[f'z_{path}'], out[f'o_{path}'] = z, o
    out['x_2d'], out['y_2d'] = samples_2d()
    out['x_1d'] = samples_1d()
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}')


if __name__ == '__main__':
    main()
