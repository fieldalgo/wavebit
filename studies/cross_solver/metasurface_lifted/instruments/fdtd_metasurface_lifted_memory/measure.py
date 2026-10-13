"""Peak allocation of one RCWA solve of M3 (truncation 12): C7's one layer and the lifted two.

Run from the repository root: ``python studies/results/fdtd_metasurface_lifted_memory/measure.py``.
NumPy reports its buffers to ``tracemalloc``; the peak is printed in units of one complex
2N x 2N matrix (N orders), the size the solve's dense blocks have.
"""
import math
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fdtd_metasurface_rcwa as c7                                               # noqa: E402
import numpy as np                                                                # noqa: E402
import wavebit                                                                    # noqa: E402

g = c7.SCENES['M3']
box = c7.box_raster('M3', g['size'])
trunc = (12, 12)
n = (2 * trunc[0] + 1) * (2 * trunc[1] + 1)
unit = (2 * n) ** 2 * 16
for layers in (1, 2):
    for algo in ('ETM', 'SMM'):
        structure = box if layers == 1 else np.concatenate([box, np.ones_like(box)])
        thickness = [g['H']] if layers == 1 else [g['H'], 0.10]
        tracemalloc.start()
        t0 = time.perf_counter()
        wavebit.rcwa(backend=0, top=1.0, bot=g['eps_s'],
                     theta=math.radians(c7.theta_at('M3', 1.0)), phi=math.radians(g['phi']),
                     psi=math.pi / 2, truncation_order=list(trunc), period=list(g['P']),
                     wavelength=1.0, structure=structure, thickness=thickness,
                     fourier_coefficients='integrated', factorization='li',
                     layer_algorithm=algo).solve()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(f'{layers} layer(s) {algo} truncation {trunc}: peak {peak / 1e9:.3f} GB = '
              f'{peak / unit:.1f} matrices of 2N x 2N; wall {time.perf_counter() - t0:.1f} s')
for t in (20, 26, 28, 30, 32):
    m = (2 * t + 1) ** 2
    print(f'truncation ({t}, {t}): N {m}, one 2N x 2N matrix {(2 * m) ** 2 * 16 / 1e9:.3f} GB, '
          f'34.5 of them {34.5 * (2 * m) ** 2 * 16 / 1e9:.1f} GB')
