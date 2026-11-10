"""Fine scan of sum(T) for 'metallic' (eps -20+1.5j circle in air) at orders (4,4), default grid.

The 30-point sweep (dx = 0.0069) of gradients.json is far coarser than the structure of
T(radius) for this case on both arms; this resolves 0.29..0.31 at 41 points (dx = 5e-4) with a
local central difference (h = 1e-5) to tell resonant-but-smooth from jagged.
Results -> gradients_metalscan.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

from wavebit import rcwa                                         # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C         # noqa: E402

CASE, ORDERS, H = 'metallic', 4, 1e-5
xs = np.linspace(.29, .31, 41)
out = {}
for arm in C.ARMS:
    with C.arm(arm):
        build = C.CASES[CASE][0]
        prepared = rcwa.prepare(build(.3), numerics=C.numerics(ORDERS))
        f = lambda v: float(np.sum(np.asarray(prepared.run(build(v)).T)))
        T, G = [], []
        for x in xs:
            T.append(f(x))
            G.append((f(x + H) - f(x - H)) / (2 * H))
        out[arm] = dict(x=xs.tolist(), T=T, fd=G)
    print(arm, flush=True)
    json.dump(out, open(os.path.join(HERE, 'gradients_metalscan.json'), 'w'), indent=1)
