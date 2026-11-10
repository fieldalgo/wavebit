"""Analyse grid_raw.json (see grid_sweep.py) into grid.json and printed tables.

Error at grid ``g`` = max over diffraction orders of ``|R_g - R_ref|`` and ``|T_g - T_ref|``.
Primary reference = the same arm and orders at grid 1024 (overlap: 256); where a 2048 run exists,
errors against 2048 are reported as well (``err2048``), including the 1024 run's own.
``within_*`` = smallest ladder grid from which every larger grid (below the reference) stays
within the tolerance. ``slope`` = least-squares slope of log10(err) on log10(grid) over
32 <= grid <= ref/2 (all grids run, dense ones included). ``slope_even_32_to_ref_over_4`` fits even grids only (odd grids put a sample on the shape centre
and are systematically worse; ``slope_odd`` fits odd grids >= 31). ``*_even`` thresholds use
even grids only, ``*_all`` every grid run. ``scatter_even_256_768`` = max/min error over even grids
256..768 (vs 2048 where run): ~2-9 for a smooth algebraic decay, much larger for a plateau.
Overlap: reference 512, finer 1024 (integrated arm only) when run.
``energy_defect`` = R + T - 1 per grid (0 for lossless cases on an exact builder).
``odd_even`` = geometric-mean
error of odd grids over that of the neighbouring even grids (63/65 vs 64, ...), >1: odd worse.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ('scalar', 'integrated')
ORDERS = (3, 6)
ORDER = ['rectangle', 'rect-tilted', 'circle', 'ellipse', 'polygon', 'overlap', 'hexagonal',
         'oblique', 'lossy', 'metallic', 'high-contrast']
POW2 = (1, 2, 3, 8, 12, 13, 16, 24, 25, 32, 48, 64, 96, 128, 256, 512)
PARITY = ((31, 33, 32), (63, 65, 64), (127, 129, 128), (255, 257, 256))


def err(a, b):
    return float(max(np.abs(np.subtract(a['R'], b['R'])).max(),
                     np.abs(np.subtract(a['T'], b['T'])).max()))


def first_within(errs, tol):
    ok = None
    for g in sorted(errs, reverse=True):
        if errs[g] > tol:
            break
        ok = g
    return ok


def analyse(raw):
    E = {}
    for k, v in raw.items():
        arm, case, M, g = k.split('|')
        E.setdefault((case, int(M), arm), {})[int(g)] = v
    out = {}
    for case in [c for c in ORDER if any(k[0] == c for k in E)]:
        for M in ORDERS:
            for arm in ARMS:
                d = E.get((case, M, arm), {})
                ok = {g: v for g, v in d.items() if 'error' not in v}
                ref_g = (512 if 512 in ok else 256) if case == 'overlap' else 1024
                if ref_g not in ok:
                    continue
                errs = {g: err(v, ok[ref_g]) for g, v in sorted(ok.items()) if g < ref_g}
                fine = 2048 if 2048 in ok else (1024 if case == 'overlap' and 1024 in ok else None)
                errs2048 = ({g: err(v, ok[fine]) for g, v in sorted(ok.items()) if g < fine}
                            if fine else None)
                even = {g: e for g, e in errs.items() if g % 2 == 0}
                fit_e = [(g, e) for g, e in even.items() if 32 <= g <= ref_g // 4 and e > 0]
                slope_even = (float(np.polyfit(np.log10([g for g, _ in fit_e]),
                                               np.log10([e for _, e in fit_e]), 1)[0]) if len(fit_e) >= 3 else None)
                odd = {g: e for g, e in errs.items() if g % 2 and g >= 31 and e > 0}
                slope_odd = (float(np.polyfit(np.log10(list(odd)), np.log10(list(odd.values())), 1)[0])
                             if len(odd) >= 3 else None)
                # Scatter: max/min error among even grids 256..768 (a smooth decay gives ~3-9).
                band = [e for g, e in (errs2048 or errs).items() if g % 2 == 0 and 256 <= g <= 768]
                scatter = float(max(band) / min(band)) if len(band) >= 2 and min(band) > 0 else None
                ladder = {g: e for g, e in errs.items() if g in POW2 or case == 'overlap'}
                fit = [(g, e) for g, e in errs.items() if 32 <= g <= ref_g // 2 and e > 0]
                slope = (float(np.polyfit(np.log10([g for g, _ in fit]),
                                          np.log10([e for _, e in fit]), 1)[0]) if len(fit) >= 3 else None)
                par = [(errs[o1] * errs[o2]) ** .5 / errs[e] for o1, o2, e in PARITY
                       if all(x in errs and errs[x] > 0 for x in (o1, o2, e))]
                # Non-monotone steps on the power-of-two ladder from 16 up.
                gs = [g for g in sorted(ladder) if g >= 16]
                up = [g2 for g1, g2 in zip(gs, gs[1:]) if ladder[g2] > 1.5 * ladder[g1]]
                out[f'{case}|{M}|{arm}'] = dict(
                    ref_grid=ref_g, fine_grid=fine, err=errs, err2048=errs2048,
                    slope_even_32_to_ref_over_4=slope_even, slope_odd=slope_odd,
                    within_1e4_even=first_within(even, 1e-4), within_1e6_even=first_within(even, 1e-6),
                    scatter_even_256_768=scatter,
                    refused=sorted(g for g, v in d.items() if 'error' in v),
                    refusal=next((v['error'][:160] for v in d.values() if 'error' in v), None),
                    slope=slope, within_1e4=first_within(ladder, 1e-4), within_1e6=first_within(ladder, 1e-6),
                    within_1e4_all=first_within(errs, 1e-4), within_1e6_all=first_within(errs, 1e-6),
                    odd_even=float(np.exp(np.mean(np.log(par)))) if par else None, ladder_increases=up,
                    sumRT_ref=float(sum(ok[ref_g]['R']) + sum(ok[ref_g]['T'])),
                    energy_defect={g: float(sum(v['R']) + sum(v['T']) - 1.) for g, v in sorted(ok.items())},
                    max_imag=max(v.get('imag', 0.) for v in ok.values()))
    return out


def fmt(e):
    return f'{e:7.1e}' if e is not None else '      -'


def main():
    raw = json.load(open(os.path.join(HERE, 'grid_raw.json')))
    res = analyse(raw)
    json.dump(res, open(os.path.join(HERE, 'grid.json'), 'w'), indent=1)
    cols = [1, 2, 3, 8, 12, 13, 16, 25, 32, 64, 128, 256, 512]
    print('ladder error vs own grid-1024 answer (overlap: vs 512)')
    print('case|M|arm'.ljust(26) + ''.join(f'{g:>8d}' for g in cols)
          + '   slope 1e-4 1e-6 | 1e-4* 1e-6*  odd/even  1024vs2048  up@')
    for k, r in res.items():
        e = r['err']
        print(k.ljust(26) + ''.join(' ' + fmt(e.get(g)) for g in cols)
              + f"  {'-' if r['slope'] is None else round(r['slope'], 2)!s:>6} {r['within_1e4']!s:>4} {r['within_1e6']!s:>4}"
              + f" | {r['within_1e4_all']!s:>5} {r['within_1e6_all']!s:>5}"
              + f"  {'-' if r['odd_even'] is None else round(r['odd_even'], 1)!s:>8}"
              + f"  {fmt(r['err2048'].get(1024)) if r['err2048'] else '      -':>10}  {r['ladder_increases']}")
    print('\nsummary (errors vs own 1024 [overlap: 512]; even = even grids only incl. dense)')
    print('case|M|arm'.ljust(26) + '   e@16   e@32   e@64  e@128  e@256  e@512 | slope_e slope_o odd/even'
          ' | 1e-4 1e-6 (pow2) | 1e-4 1e-6 (even) | 1e-4 1e-6 (all) | scatter')
    for k, r in res.items():
        e = r['err']
        f = lambda v: '-' if v is None else f'{v:.2f}' if isinstance(v, float) else str(v)
        print(k.ljust(26) + ''.join(' ' + (f'{e[g]:.0e}' if g in e else '    -').rjust(6) for g in (16, 32, 64, 128, 256, 512))
              + f" | {f(r['slope_even_32_to_ref_over_4']):>7} {f(r['slope_odd']):>7} {f(r['odd_even']):>8}"
              + f" | {f(r['within_1e4']):>4} {f(r['within_1e6']):>4}       "
              + f" | {f(r['within_1e4_even']):>4} {f(r['within_1e6_even']):>4}       "
              + f" | {f(r['within_1e4_all']):>4} {f(r['within_1e6_all']):>4}      | {f(r['scatter_even_256_768'])}")
    print('\nR+T-1 vs grid (lossless cases)')
    ecols = [3, 8, 13, 16, 25, 32, 64, 128, 256, 512, 1024, 2048]
    print('case|M|arm'.ljust(26) + ''.join(f'{g:>10d}' for g in ecols))
    for k, r in res.items():
        if k.split('|')[0] not in ('lossy', 'metallic'):
            print(k.ljust(26) + ''.join(f"{r['energy_defect'][g]:10.1e}" if g in r['energy_defect'] else '         -'
                                        for g in ecols))
    dense = [g for g in (20, 31, 32, 33, 40, 48, 63, 64, 65, 80, 96, 127, 128, 129, 160, 192, 255, 256,
                         257, 320, 384, 512, 640, 768, 1024)]
    print('\nall grids, error vs own finer answer: 2048 (overlap integrated: 1024)')
    print('case|M|arm'.ljust(26) + ''.join(f'{g:>8d}' for g in dense))
    for k, r in res.items():
        if r['err2048']:
            print(k.ljust(26) + ''.join(' ' + fmt(r['err2048'].get(g)) for g in dense))


if __name__ == '__main__':
    main()
