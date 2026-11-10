"""Eager jax.grad against PyTorch backward and a NumPy central difference, on today's builder.

    python lattice_gradients.py      # writes results/lattice_gradients.json

The short-grid study (docs/theory/rcwa/scalar-nvf-short-grid-study.md, section (c)) found a
JAX-versus-finite-difference gap on the hexagonal cell at grid 13, where the prototype equals
today's code. This records that the gap is there with the prototype switch off, and on which
cases and grids.
"""
import json
import pathlib

import numpy as np

from cases import CASES, arm
from wavebit import rcwa

OUT = pathlib.Path(__file__).with_name('results')
orders = (3, 3)


def grads(case, grid):
    import jax
    import jax.numpy as jnp
    import torch
    build, x0, _ = CASES[case]
    numerics = rcwa.Numerics(orders=orders, shape_nvf_grid_size=grid)
    with jax.enable_x64(True):
        prepared = rcwa.prepare(build(x0), numerics=numerics, backend='jax')
        g_jax = float(jax.grad(lambda v: jnp.sum(prepared.run(build(v)).T))(jnp.asarray(x0)))
    prepared = rcwa.prepare(build(torch.tensor(x0, dtype=torch.float64)), numerics=numerics, backend='torch')
    live = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    torch.sum(prepared.run(build(live)).T).backward()
    g_torch = float(live.grad)
    concrete = rcwa.prepare(build(x0), numerics=numerics)
    f = lambda v: float(np.sum(concrete.run(build(v)).T))
    fd = {h: (f(x0 + h) - f(x0 - h))/(2*h) for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)}
    return dict(case=case, grid=grid, jax=g_jax, torch=g_torch, fd=fd,
                jax_vs_torch=abs(g_jax - g_torch)/abs(g_torch), torch_vs_fd5=abs(g_torch - fd[1e-5])/abs(fd[1e-5]))


rows = []
plan = [('hexagonal', g) for g in (13, 14, 15, 16, 17, 21, 32, 64, 65, 128, None)]
plan += [(c, g) for c in ('circle', 'polygon', 'oblique') for g in (13, 32, 64)]
with arm(False):
    for case, grid in plan:
        rows.append(grads(case, grid))
        print(rows[-1], flush=True)
(OUT / 'lattice_gradients.json').write_text(json.dumps(rows, indent=1, default=str))
