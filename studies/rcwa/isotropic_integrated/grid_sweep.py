"""Grid dimension: dependence of R, T on ``shape_nvf_grid_size`` at fixed orders, both arms.

Run from the repository root with ``PYTHONPATH=src`` (single-threaded BLAS)::

    python studies/rcwa/isotropic_integrated/grid_sweep.py      # fills grid_raw.json (resumable)
    python studies/rcwa/isotropic_integrated/grid_sweep.py --dense CASE...  # extra non-power-of-2 grids
    python studies/rcwa/isotropic_integrated/grid_sweep.py --overlap-ext overlap
    python studies/rcwa/isotropic_integrated/grid_analyze.py    # writes grid.json

Every solve is stored as ``R`` and ``T`` per diffraction order (real parts; all are real),
keyed ``arm|case|M|grid``; a solve the engine refuses is stored with its error.
"""
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
from studies.rcwa.isotropic_integrated import cases as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'grid_raw.json')

ORDERS = (3, 6)
BELOW_FLOOR = (1, 2, 3, 8, 12)           # integrated arm only (the scalar arm refuses < 4M+1)
LADDER = (13, 16, 25, 32, 64, 128, 256, 512, 1024)
EXTRA = {'circle': (2048,), 'rect-tilted': (2048,), 'ellipse': (2048,)}
# --dense: non-power-of-two and odd grids (parity / alignment check) and a 2048 reference.
DENSE = (20, 31, 33, 40, 48, 63, 65, 80, 96, 127, 129, 160, 192, 255, 257, 320, 384, 640, 768, 2048)
# --overlap-ext: a finer reference for the painted case (its scalar arm is not converged at 256).
OVERLAP_EXT = {'scalar': (192, 384, 512), 'integrated': (192, 384, 512, 1024)}
OVERLAP = (13, 16, 25, 32, 48, 64, 96, 128, 256)   # painted: scalar arm costs ~grid**2


def grids(case, arm, M, dense=False, ext=False):
    if ext:
        return list(OVERLAP_EXT[arm]) if case == 'overlap' else []
    if dense:
        return [g for g in DENSE if g >= 4 * M + 1 or arm == 'integrated']
    base = OVERLAP if case == 'overlap' else LADDER + EXTRA.get(case, ())
    g = list(base)
    if arm == 'integrated':
        g = list(BELOW_FLOOR) + g
    if M == 6 and arm == 'integrated':
        g += [24]                         # one below the scalar floor 4M+1 = 25
    if arm == 'scalar':
        g += [4 * M]                      # recorded as the refusal
    return sorted(set(g))


def main():
    import wavebit
    assert wavebit.__file__.startswith(os.path.join(ROOT, 'src')), wavebit.__file__
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    args = sys.argv[1:]
    dense, ext = '--dense' in args, '--overlap-ext' in args
    cases = [a for a in args if not a.startswith('--')] or list(C.CASES)
    for case in cases:
        for arm in C.ARMS:
            with C.arm(arm):
                for M in ORDERS:
                    for g in grids(case, arm, M, dense, ext):
                        key = f'{arm}|{case}|{M}|{g}'
                        if key in store:
                            continue
                        try:
                            R, T = C.solve(case, M, grid=g)
                            rec = dict(R=np.real(R).ravel().tolist(), T=np.real(T).ravel().tolist(),
                                       imag=float(max(np.abs(np.imag(R)).max(), np.abs(np.imag(T)).max())),
                                       shape=list(np.shape(T)))
                        except Exception as e:  # noqa: BLE001
                            rec = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
                            if not str(e):
                                rec['tb'] = traceback.format_exc()[-800:]
                        store[key] = rec
                        tmp = OUT + '.tmp'
                        with open(tmp, 'w') as f:
                            json.dump(store, f)
                        os.replace(tmp, OUT)
                        print(key, 'ERR ' + rec['error'][:80] if 'error' in rec else
                              'sumT=%.12f' % sum(rec['T']), flush=True)


if __name__ == '__main__':
    main()
