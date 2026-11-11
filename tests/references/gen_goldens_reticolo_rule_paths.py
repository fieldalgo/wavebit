"""Freeze RETICOLO's in-cell field off the scalar rule's first paths into ``golden_reticolo_rule_paths.npz``.

Run once, on a machine with Octave and a RETICOLO V10 tree;
``test_reticolo_rule_paths_golden.py`` then runs anywhere.

    RETICOLO_PATH=... python tests/references/gen_goldens_reticolo_rule_paths.py

Rerun it only to add cases -- never to make a failing test pass.

**What it stores.** ``res3``'s raw cell on each family of ``_reticolo_rule_paths_cases.py`` under
``champ = 1`` and ``champ = 0``, and for a dipole the same two from ``retchamp``. ``champ = 1`` is
P. Lalanne and M. P. Jurek, J. Mod. Opt. 45, 1357 (1998); RETICOLO's ``retcouche`` builds ``fex``,
``fey``, ``fhx`` and ``fhy`` from ``eps_xx``, ``eps_yy``, ``mu_xx`` and ``mu_yy``, and ``retchamp`` divides
the sums by the same entries at the sample.

**One case avoids a branch.** A lossless case must conserve energy in RETICOLO before it can speak
for anything: ``res1`` on a y-invariant texture with ``nn = [N, 0]`` and ``change_index`` takes a branch that does not
(R + T of 0.95 and 1.12 on a diagonal tensor at 30 degrees), so the tensor conical case asks for
``nn = [N, 1]``, where it does.

**How it runs RETICOLO.** Through ``octave`` as a subprocess on ``interface/reticolo_res3_magnetic.m``
and ``interface/reticolo_source.m``.

**Citation.** RETICOLO's terms require Hugonin & Lalanne (arXiv:2101.00901), and Lalanne & Jurek
(above) whenever near-field distributions are shown -- which a field golden is. See ``README.md``.
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

OUT = HERE / 'golden_reticolo_rule_paths.npz'


def _octave(function, case):
    import scipy.io as sio
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = os.path.join(tmp, 'in.mat'), os.path.join(tmp, 'out.mat')
        sio.savemat(infile, case)
        script = (f"addpath(genpath('{os.environ['RETICOLO_PATH']}')); addpath('{HERE / 'interface'}'); "
                  f"{function}('{infile}', '{outfile}');")
        subprocess.run(['octave', '--no-gui', '--quiet', '--eval', script], check=True, stdin=subprocess.DEVNULL)
        return sio.loadmat(outfile)


def _res3(case):
    from tests.references._reticolo_rule_paths_cases import NPTS, change_index
    case = dict(case, npts=float(NPTS))
    rows = change_index()
    case['n_ci'] = float(len(rows))
    for k, row in enumerate(rows, 1):
        case[f'ci_{k}'] = row
    return _octave('reticolo_res3_magnetic', case)


def _two_d(stack, champ, nn=None, angle=None, backgrounds=None, sens=None):
    from tests.references._reticolo_rule_paths_cases import (
        ANGLE_2D, EINC, N_BOT, N_TOP, ORDER_2D, PERIOD_2D, WL, blocks_of, samples_2d)
    theta, phi = ANGLE_2D if angle is None else angle
    x, y = samples_2d()
    case = dict(path='2d', wavelength=WL, period=np.array(PERIOD_2D),
                nn=np.array([ORDER_2D, ORDER_2D] if nn is None else nn, float),
                k_parallel=N_TOP * np.sin(np.deg2rad(theta)), delta0=phi, champ=float(champ),
                n_top=N_TOP, n_bot=N_BOT, n_layers=float(len(stack)),
                thickness=np.array([layer['thickness'] for layer in stack]), x=x, y=y, n_einc=float(len(EINC)))
    for k, layer in enumerate(stack, 1):
        case[f'bg_{k}'] = layer['background'] if backgrounds is None or k - 1 not in backgrounds else backgrounds[k - 1]
        case[f'blocks_{k}'] = blocks_of(layer)
    for k, pol in enumerate(EINC, 1):
        case[f'einc_{k}'] = np.array(EINC[pol], dtype=complex)
    if sens is not None:
        case['sens'] = float(sens)
    got = _res3(case)
    return {pol: got[f'e_{k}'] for k, pol in enumerate(EINC, 1)}, got['z'].ravel()


def _one_d(stack, path, champ):
    from tests.references._reticolo_rule_paths_cases import (
        ANGLE_1D, ANGLE_CONICAL, EINC, N_BOT, N_TOP, ORDER_1D, PERIOD_1D, WL, samples_1d)
    conical = path == 'conical'
    theta = ANGLE_CONICAL[0] if conical else ANGLE_1D
    case = dict(path=path, wavelength=WL, period=PERIOD_1D, nn=float(ORDER_1D),
                k_parallel=N_TOP * np.sin(np.deg2rad(theta)), delta0=ANGLE_CONICAL[1], champ=float(champ),
                n_top=N_TOP, n_bot=N_BOT, n_layers=float(len(stack)),
                thickness=np.array([layer['thickness'] for layer in stack]), x=samples_1d(), y=0.0,
                n_einc=float(len(EINC)))
    if conical:
        case['nn'] = np.array([float(ORDER_1D), 1.0])
    for k, layer in enumerate(stack, 1):
        x0, x1, n = (100.0, 200.0, layer['background']) if layer['bar'] is None else layer['bar']
        case[f'edges_{k}'] = np.array([x0, x1, PERIOD_1D])
        case[f'n_{k}'] = np.array([layer['background'], n, layer['background']])
    for k, pol in enumerate(EINC, 1):
        case[f'einc_{k}'] = np.array(EINC[pol], dtype=complex)
    got = _res3(case)
    if conical:
        return {pol: got[f'e_{k}'] for k, pol in enumerate(EINC, 1)}, got['z'].ravel()
    return {'s' if path == 'te' else 'p': got['e_1']}, got['z'].ravel()


def _source(component, c):
    from tests.references._reticolo_rule_paths_cases import (
        N_BOT, N_TOP, NPTS, PERIOD_2D, SOURCE_BLOCK, SOURCE_HOST, SOURCE_HOST_INDEX, SOURCE_ORDER, SOURCE_PATTERNED,
        SOURCE_POSITION, SOURCE_ROW, WL, samples_2d)
    x, y = samples_2d()
    got = _octave('reticolo_source', dict(
        wavelength=WL, period=np.array(PERIOD_2D), order=float(SOURCE_ORDER), x=x, y=y, n_top=N_TOP, n_bot=N_BOT,
        n_host=SOURCE_HOST_INDEX, block=np.array([*SOURCE_BLOCK, 1.0]), position=np.array(SOURCE_POSITION),
        component=float(component), patterned=SOURCE_PATTERNED, host_above=SOURCE_HOST[0], row=SOURCE_ROW,
        host_below=SOURCE_HOST[1], c=float(c), npts=float(NPTS)))
    return got['e'], got['z'].ravel()


def main():
    if not os.environ.get('RETICOLO_PATH'):
        raise SystemExit('set RETICOLO_PATH to a reticolo_allege_v10 directory to regenerate this '
                         'golden; the committed .npz is what the tests read.')
    os.environ['RETICOLO_PATH'] = os.path.expanduser(os.environ['RETICOLO_PATH'])
    from tests.references._reticolo_rule_paths_cases import (
        MERGED_1D, MERGED_2D, ONE_ORDER_2D, ORDER_2D, SOURCE_MOMENTS, TENSOR_1D, TENSOR_2D, UNIFORM_ANGLE,
        UNIFORM_BLOCK, UNIFORM_TENSOR, UNIFORM_THICKNESS, key, samples_1d, samples_2d)
    out = {}

    def store(path, champ, cells_z):
        cells, z = cells_z
        for pol, e in cells.items():
            out[key(path, pol, champ)] = e
        out[f'z_{path}'] = z

    for champ in (1, 0):
        store('tensor_2d', champ, _two_d(TENSOR_2D, champ))
        store('bottom_2d', champ, _two_d(TENSOR_2D, champ, sens=-1))
        for path in ('te', 'tm', 'conical'):
            store(f'tensor_{path}', champ, _one_d(TENSOR_1D, path, champ))
            store(f'merged_{path}', champ, _one_d(MERGED_1D, path, champ))
        store('merged_2d', champ, _two_d(MERGED_2D, champ))
        store('one_x', champ, _two_d(ONE_ORDER_2D, champ, nn=[ORDER_2D, 0]))
        store('one_y', champ, _two_d(ONE_ORDER_2D, champ, nn=[0, ORDER_2D]))
        for pair, tensor in UNIFORM_TENSOR.items():
            stack = [dict(thickness=UNIFORM_THICKNESS[0], background=1.0, blocks=[UNIFORM_BLOCK]),
                     dict(thickness=UNIFORM_THICKNESS[1], background=1.0, blocks=[])]
            store(f'uniform_{pair}', champ, _two_d(stack, champ, angle=UNIFORM_ANGLE, backgrounds={1: tensor}))
    for component in SOURCE_MOMENTS:
        for c in (1, 0):
            e, z = _source(component, c)
            out[f'source_{component}_{c}'] = e
            out['z_source'] = z
    out['x_2d'], out['y_2d'] = samples_2d()
    out['x_1d'] = samples_1d()
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}')


if __name__ == '__main__':
    main()
