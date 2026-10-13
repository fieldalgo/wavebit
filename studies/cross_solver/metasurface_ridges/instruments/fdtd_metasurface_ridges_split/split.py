"""The ridges study's rows split by wavelength and by cell -- an analysis made after the
campaign, not part of the registered rule.

Run from the repository root: ``python studies/results/fdtd_metasurface_ridges_split/split.py``.
It reads ``studies/results/fdtd_metasurface_ridges.jsonl`` (``Y12`` against the RCWA ladder's
top, N = 800) and the corner study's ``R12`` rows (zeroth order, against its top, N = 600), and
prints for each wavelength the worst amplitude and efficiency error and its local orders, the
local orders of every cell whose error at rung 320 is at least a fifth of the worst of its
kind (amplitudes and efficiencies apart), and the
reference-free self-convergence order per wavelength.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNGS = (40, 80, 160, 320)
WL = (0.95, 1.0, 1.05)
KEYS = [f'{pol}|{side}|{q}' for pol in ('s', 'p') for side, power in (('r', 'R'), ('t', 'T'))
        for q in (f'{side}_s', f'{side}_p', power)]


def rows_of(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.replace('\0', '').strip()
        if line:
            r = json.loads(line)
            out[r['key']] = r
    return out


ridges = rows_of(ROOT / 'studies/results/fdtd_metasurface_ridges.jsonl')
corners = rows_of(ROOT / 'studies/results/fdtd_metasurface_corners.jsonl')


def y12(n):
    return {k: [complex(*v) for v in ridges[f'fd|Y12|{n}|{k.split("|")[0]}|120']['values'][
        '|'.join(k.split('|')[1:])]] for k in KEYS}


def y_ref():
    return {k: [complex(*ridges[f'rcwa|Y|{i}|800']['values'][k]) for i in range(3)] for k in KEYS}


def r12(n):
    out = {}
    for k in KEYS:
        pol, side, q = k.split('|')
        out[k] = [complex(*v) for v in corners[f'fd|R12|{n}|{pol}']['values'][f'{side}|0|{q}']]
    return out


def r_ref():
    return {k: [complex(*corners[f'rcwa|R12|{i}|600']['values'][
        f'{k.split("|")[0]}|{k.split("|")[1]}|0|{k.split("|")[2]}']) for i in range(3)]
        for k in KEYS}


def orders(X):
    return [math.log(X[j] / X[j + 1]) / math.log(2) for j in range(len(X) - 1)]


def table(label, fd, ref):
    print(label)
    v = {n: fd(n) for n in RUNGS}
    for i, wl in enumerate(WL):
        for eff in (False, True):
            X = [max(abs(v[n][k][i] - ref[k][i]) for k in KEYS
                     if (k.split('|')[-1] in ('R', 'T')) == eff) for n in RUNGS]
            print(f'  {wl} {"F" if eff else "A"}: worst errors '
                  + ', '.join(f'{x:.2e}' for x in X) + '; local orders '
                  + ', '.join(f'{o:.2f}' for o in orders(X)))
    for i, wl in enumerate(WL):
        for eff in (False, True):
            kind = [k for k in KEYS if (k.split('|')[-1] in ('R', 'T')) == eff]
            worst = max(abs(v[320][k][i] - ref[k][i]) for k in kind)
            for k in kind:
                X = [abs(v[n][k][i] - ref[k][i]) for n in RUNGS]
                if X[-1] < 0.2 * worst:
                    continue
                print(f'    {wl} {k}: ' + ', '.join(f'{x:.2e}' for x in X) + '; local orders '
                      + ', '.join(f'{o:.2f}' for o in orders(X)))
    return v


v = table('Y12, the yz ridge, against RCWA N 800', y12, y_ref())
table("R12, the corner study's xz ridge, zeroth order, against RCWA N 600", r12, r_ref())
print('Y12 reference-free self-convergence per wavelength (changes between rungs)')
for i, wl in enumerate(WL):
    for eff in (False, True):
        d = [max(abs(v[RUNGS[j]][k][i] - v[RUNGS[j + 1]][k][i]) for k in KEYS
                 if (k.split('|')[-1] in ('R', 'T')) == eff) for j in range(3)]
        print(f'  {wl} {"F" if eff else "A"}: changes ' + ', '.join(f'{x:.2e}' for x in d)
              + '; orders ' + ', '.join(f'{o:.2f}' for o in orders(d)))
