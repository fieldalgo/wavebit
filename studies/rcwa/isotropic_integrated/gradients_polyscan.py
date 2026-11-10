"""Fine scan of sum(T) for 'polygon' near x0 = 0.32 at orders (3,3), default grid, both arms.

The integrated arm's h=1e-5 central difference disagreed with jax.grad (and with h=1e-6) at x0;
this resolves T(x) on x0 + [-3e-5, 3e-5] to see whether it is a step or curvature.
Results -> gradients_polyscan.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

from wavebit import rcwa                                         # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C         # noqa: E402

CASE, ORDERS = 'polygon', 3
x0 = C.CASES[CASE][1]
xs = x0 + np.linspace(-3e-5, 3e-5, 13)
out = {}
for arm in C.ARMS:
    with C.arm(arm):
        build = C.CASES[CASE][0]
        prepared = rcwa.prepare(build(x0), numerics=C.numerics(ORDERS))
        out[arm] = dict(x=xs.tolist(), T=[float(np.sum(np.asarray(prepared.run(build(x)).T))) for x in xs])
    print(arm, flush=True)
    json.dump(out, open(os.path.join(HERE, 'gradients_polyscan.json'), 'w'), indent=1)
for arm, r in out.items():
    T = np.asarray(r['T'])
    print(arm, 'dT/dx per 5e-6 step:', np.round(np.diff(T) / 5e-6, 5).tolist())
