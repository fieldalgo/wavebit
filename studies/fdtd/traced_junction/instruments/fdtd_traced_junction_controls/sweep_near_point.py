"""T0 and R0 of the concrete junction scene on a fine sweep of radius and of height through the
probe's point, each geometry planned and classified on its own: where is the feature that a
1e-4 central difference crosses? Run with the studies directory as the first argument."""
import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
import fdtd_traced_junction_probe as P  # noqa: E402
import wavebit.fd as fd  # noqa: E402

warnings.filterwarnings('ignore')
for k, name in ((0, 'radius'), (1, 'height')):
    for t in np.linspace(-1.2e-4, 1.2e-4, 25):
        v = P.POINT.copy()
        v[k] += t
        s = P.scene(v)
        p = fd.plan(s)
        T, R = (float(x) for x in P.objectives(v, 'numpy', None))
        written = sum(int(r.write.sum()) for r in p.junctions)
        print(f'{name} {v[k]:.8f} T {T!r} R {R!r} written {written}', flush=True)
