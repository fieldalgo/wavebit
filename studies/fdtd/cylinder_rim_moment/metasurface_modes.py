"""(b)/(c) C7's cylinder metasurfaces M1 and M2 under each study moment, against stored RCWA.

Reuses ``studies/cross_solver/metasurface/fdtd_metasurface_rcwa.py`` unchanged: its ``build``
(the public scene), ``fd_row`` (one solve, every order) and ``judge`` (worst amplitude and
efficiency error per rung, the fitted slope). The RCWA reference is the stored ladder in that
study's ``records/fdtd_metasurface_rcwa_reference.jsonl``, read under the RCWA provenance its
report names (M1 813427469a4340bf, M2 f14ff6b46e70ac44); nothing is re-solved on the RCWA side.

Usage: python metasurface_modes.py MODE SCENE[,SCENE] [RUNGS]   (writes records/metasurface_MODE.jsonl)
       python metasurface_modes.py report
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import studies.cross_solver.metasurface.fdtd_metasurface_rcwa as c7  # noqa: E402
import wavebit.fdtd._shapes._bodies as bodies  # noqa: E402

RECORDS = HERE / 'records'
STORED = ROOT / 'studies/cross_solver/metasurface/records'
RCWA_PROV = {'M1': '813427469a4340bf', 'M2': 'f14ff6b46e70ac44'}


def setup():
    c7.PROV = c7.provenance()
    c7.PROV['rcwa'] = dict(c7.PROV['rcwa'], **RCWA_PROV)
    c7.RCWA_LOG = STORED / 'fdtd_metasurface_rcwa_reference.jsonl'


def rows(path):
    out = {}
    if path.exists():
        for line in open(path, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                out[r['key']] = r
    return out


def run(mode, names, rungs=None):
    setup()
    bodies._CYLINDER_MOMENT = mode
    path = RECORDS / f'metasurface_{mode}.jsonl'
    done = rows(path)
    for name in names:
        for n in rungs or c7.SCENES[name]['rungs']:
            for pol in c7.POLS:
                s = c7.spec(name, n, pol)
                if s['backend'] == 'jax':
                    s['backend'] = 'jax'
                if c7.key_of(s) in done and done[c7.key_of(s)].get('status') == 'ok':
                    continue
                row = c7.fd_row(s)
                row['moment'] = mode
                c7.emit(path, row)


def stored_plane():
    out = {}
    for line in open(STORED / 'fdtd_metasurface_rcwa.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r.get('kind') == 'fd' and r.get('status') == 'ok' and r['spec']['periods'] == 120:
            out[r['key']] = r
    return out


def report():
    setup()
    sets = {'plane (stored C7 rows)': stored_plane()}
    for mode in ('plane', 'rim', 'split', 'exact'):
        p = RECORDS / f'metasurface_{mode}.jsonl'
        if p.exists():
            sets[f'{mode} (this study)'] = rows(p)
    for name in ('M1', 'M2'):
        ref, ladder = c7.reference(name)
        for pol in c7.POLS:
            print(f'== {name} {pol}')
            for label, got in sets.items():
                runs = [got.get(c7.key_of(c7.spec(name, n, pol))) for n in c7.SCENES[name]['rungs']]
                if any(r is None for r in runs):
                    have = [r['spec']['n'] for r in runs if r is not None]
                    if have:
                        part = [r for r in runs if r is not None]
                        A = [c7.worst(abs(c[2] - c[3]) for c in c7.cells_of(name, pol, r, ref[pol], [l[pol] for l in ladder]) if c[1] == 'amp') for r in part]
                        F = [c7.worst(abs(c[2] - c[3]) for c in c7.cells_of(name, pol, r, ref[pol], [l[pol] for l in ladder]) if c[1] == 'eff') for r in part]
                        print(f'  {label:24s} rungs {have}: A {" ".join(f"{a:.4e}" for a in A)}  F {" ".join(f"{f:.4e}" for f in F)}')
                    continue
                j = c7.judge(name, pol, runs, ref[pol], [l[pol] for l in ladder])
                print(f'  {label:24s} A {" ".join(f"{a:.4e}" for a in j["A"])} (slope {j["slope_A"]:+.3f})'
                      f'  F {" ".join(f"{f:.4e}" for f in j["F"])} (slope {j["slope_F"]:+.3f})'
                      f'  consistent={j["consistent"]} second={j["second"]}')
    # how far each mode moves every cell from the plane rows of this study (same code, same env)
    base = sets.get('plane (this study)') or sets['plane (stored C7 rows)']
    for label, got in sets.items():
        if label.startswith('plane'):
            continue
        for key, r in sorted(got.items()):
            b = base.get(key)
            if b is None or r.get('status') != 'ok':
                continue
            worst_abs, worst_rel = 0., 0.
            for k, vals in r['values'].items():
                for v, w in zip(vals, b['values'][k]):
                    d = abs(complex(*v) - complex(*w))
                    worst_abs = max(worst_abs, d)
                    if abs(complex(*w)) > 1e-3:
                        worst_rel = max(worst_rel, d / abs(complex(*w)))
            print(f'  move {label:20s} {key}: max |d amp/eff| {worst_abs:.3e}, max rel (|ref|>1e-3) {worst_rel:.3e}')


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report()
    else:
        run(sys.argv[1], sys.argv[2].split(','),
            [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else None)
