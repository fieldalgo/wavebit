"""Does the integrated-arm polygon jaggedness depend on shape_nvf_grid_size?

7-point scan of sum(T) on x0 + [-3e-5, 3e-5] (step 1e-5) for 'polygon', orders (3,3), integrated
arm, at grids 128, 256, 512 (default 1024 is in gradients_polyscan.json), plus jax.grad at x0.
Results -> gradients_polygrid.json.
"""
import json
import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from wavebit import rcwa                                         # noqa: E402
from studies.rcwa.isotropic_integrated import cases as C         # noqa: E402

CASE, ORDERS = 'polygon', 3
build, x0, _ = C.CASES[CASE]
xs = x0 + np.linspace(-3e-5, 3e-5, 7)
out = {}
with C.arm('integrated'):
    for grid in (128, 256, 512):
        num = C.numerics(ORDERS, grid)
        prepared = rcwa.prepare(build(x0), numerics=num)
        T = [float(np.sum(np.asarray(prepared.run(build(x)).T))) for x in xs]
        with jax.enable_x64(True):
            pj = rcwa.prepare(build(x0), numerics=num, backend='jax')
            g = float(jax.grad(lambda v: jnp.sum(pj.run(build(v)).T))(jnp.asarray(x0)))
        out[grid] = dict(x=xs.tolist(), T=T, ad_jax=g)
        print(grid, 'AD', g, 'slopes', np.round(np.diff(T) / 1e-5, 5).tolist(),
              'maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
        json.dump(out, open(os.path.join(HERE, 'gradients_polygrid.json'), 'w'), indent=1)
