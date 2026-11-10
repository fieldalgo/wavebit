"""Fourier-order ladder for both arms of the isotropic-integrated study (raw solves).

Run from the repository root, single-threaded::

    PYTHONPATH=src OMP_NUM_THREADS=1 python studies/rcwa/isotropic_integrated/orders_run.py \
        CASE[:GRID] M1,M2,... [CASE[:GRID] M...]

Each solve (case, arm, M, grid) is appended to ``orders_raw.json`` beside this file as the full
``R`` and ``T`` order arrays; a key already present is skipped, so the run resumes. ``grid`` is
``shape_nvf_grid_size``; omitted means the engine default (``None`` -> max(1024, 4M+1)).
NumPy backend only. No timings are recorded. Environment ``ORDERS_ARMS=scalar`` (or
``integrated``) restricts the run to one arm (used for the metallic M=18/20 scalar-only check).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

import wavebit                                                              # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

RAW = os.path.join(HERE, 'orders_raw.json')


def key(case, arm, m, grid):
    return f'{case}|{arm}|{m}|{grid}'


def load():
    if os.path.exists(RAW):
        with open(RAW) as f:
            return json.load(f)
    return {}


def save(store):
    tmp = RAW + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(store, f)
    os.replace(tmp, RAW)


def pack(a):
    a = np.asarray(a)
    return {'re': a.real.tolist(), 'im': a.imag.tolist()} if np.iscomplexobj(a) else a.tolist()


def main(argv):
    assert wavebit.__file__.startswith(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'src')))
    store = load()
    jobs = []
    for spec, ms in zip(argv[0::2], argv[1::2]):
        case, _, g = spec.partition(':')
        grid = int(g) if g else None
        for m in (int(v) for v in ms.split(',')):
            for arm in os.environ.get('ORDERS_ARMS', ','.join(C.ARMS)).split(','):
                jobs.append((case, arm, m, grid))
    for case, arm, m, grid in jobs:
        k = key(case, arm, m, grid)
        if k in store:
            continue
        try:
            with C.arm(arm):
                r, t = C.solve(case, m, grid=grid)
            store[k] = {'R': pack(r), 'T': pack(t)}
            print(f'{k}: sumR={r.sum():.12f} sumT={t.sum():.12f}', flush=True)
        except Exception as exc:                                            # noqa: BLE001
            store[k] = {'error': f'{type(exc).__name__}: {exc}'[:500]}
            print(f'{k}: ERROR {store[k]["error"]}', flush=True)
        save(store)


if __name__ == '__main__':
    main(sys.argv[1:])
