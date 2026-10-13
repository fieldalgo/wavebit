"""RCWA zeroth-order band scan and control sizing for the ridges study (no FDTD), made before
its registration.

Run from the repository root: ``python studies/results/fdtd_metasurface_ridges_scan/scan.py``.

* The band scan: the yz ridge (epsilon 12 and 6) and the corner study's xz ridge (epsilon 12) at
  the fixed-k angle of each wavelength, truncation 40 along the periodic axis.
* The control sizing: how far the epsilon-12 yz ridge's zeroth-order cells move when the ridge is
  narrower or wider by ``CONTROL_SHIFT`` of a cell at each rung, worst over the cells and the
  three wavelengths as the study takes it, at truncations 40 and 60.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fdtd_metasurface_corners as cs                                            # noqa: E402
import fdtd_metasurface_ridges as L                                              # noqa: E402

GRID = [0.95 + 0.025 * k for k in range(8)]


def show(label, point):
    print(f'{label}: wavelength, R_s, T_s, R_p, T_p (zeroth order)')
    for wl in GRID:
        v = point(wl)
        row = [v[k].real for k in ('s|r|R', 's|t|T', 'p|r|R', 'p|t|T')]
        print(f'  {wl:.3f}  ' + '  '.join(f'{x:.4f}' for x in row), flush=True)


def yz(eps):
    def point(wl):
        L.EPS_P = eps
        return L.rcwa_point('Y', wl, 40)
    return point


def xz(wl):
    saved = cs.WAVELENGTHS
    cs.WAVELENGTHS = (wl,)
    try:
        v = cs.rcwa_point('R12', wl, 40)
    finally:
        cs.WAVELENGTHS = saved
    return {f'{pol}|{side}|{q}': v[f'{pol}|{side}|0|{q}'] for pol in ('s', 'p')
            for side, q in (('r', 'R'), ('t', 'T'))}


def sizing(N):
    L.EPS_P = 12.0
    w0 = L.GEOM['Y']['width']
    base = {wl: L.rcwa_point('Y', wl, N) for wl in L.WAVELENGTHS}
    print(f'control sizing, epsilon 12, truncation {N}: rung, sign, worst A, worst F, '
          f'worst A per wavelength')
    for n in L.RUNGS:
        for sign in L.CONTROL_SIGNS:
            per = []
            A = F = 0.0
            for wl in L.WAVELENGTHS:
                v = L.rcwa_point('Y', wl, N, w0 + sign * L.CONTROL_SHIFT / n)
                a = max(abs(v[k] - base[wl][k]) for k in v if k.split('|')[-1] not in ('R', 'T'))
                f = max(abs(v[k] - base[wl][k]) for k in v if k.split('|')[-1] in ('R', 'T'))
                per.append(a)
                A, F = max(A, a), max(F, f)
            print(f'  {n:4d}  {sign:+d}  {A:.3e}  {F:.3e}  '
                  + '  '.join(f'{wl}: {a:.2e}' for wl, a in zip(L.WAVELENGTHS, per)), flush=True)


show('yz ridge (Y), epsilon 12', yz(12.0))
show('yz ridge (Y), epsilon 6', yz(6.0))
show("the corner study's xz ridge (R12), epsilon 12", xz)
for N in (40, 60):
    sizing(N)
