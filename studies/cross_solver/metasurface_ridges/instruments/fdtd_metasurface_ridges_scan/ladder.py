"""The yz ridge's RCWA ladder spread at tops 600 and 800 (no FDTD), before registration.

Run from the repository root: ``python studies/results/fdtd_metasurface_ridges_scan/ladder.py``.
``u`` is the study's spread: the largest distance of the three truncations below the top from
the top, over the zeroth-order cells and the three wavelengths, amplitudes and efficiencies apart.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fdtd_metasurface_ridges as L                                              # noqa: E402

L.EPS_P = 12.0
NS = (100, 200, 300, 400, 600, 800)
vals = {}
for N in NS:
    t0 = time.perf_counter()
    vals[N] = [L.rcwa_point('Y', wl, N) for wl in L.WAVELENGTHS]
    print(f'N {N}: {time.perf_counter() - t0:.0f} s', flush=True)


def spread(top, lower):
    out = []
    for eff in (False, True):
        out.append(max(abs(vals[lo][i][k] - vals[top][i][k]) for lo in lower
                       for i in range(len(L.WAVELENGTHS)) for k in vals[top][i]
                       if (k.split('|')[-1] in ('R', 'T')) == eff))
    return out


for top in (600, 800):
    lower = [N for N in NS if N < top][-3:]
    uA, uF = spread(top, lower)
    print(f'top {top} (lower {lower}): u_A {uA:.3e} u_F {uF:.3e}; gate floor (u / '
          f'{L.REFERENCE_SHARE}) A {uA / L.REFERENCE_SHARE:.3e} F {uF / L.REFERENCE_SHARE:.3e}')
for N in NS[:-1]:
    dA, dF = spread(800, [N])
    print(f'N {N} against 800: A {dA:.3e} F {dF:.3e}')
