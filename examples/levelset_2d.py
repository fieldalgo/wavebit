"""A level-set layer, and a short topology optimisation run on it.

The point of the path is that the geometry is a *field*: the boundary is wherever ``phi``
crosses zero, so merging and splitting cost nothing, and the gradient that comes back is
``dJ/d phi`` — a shape derivative, not a per-pixel density gradient. This script shows both
halves: one forward solve, then a handful of ascent steps that move the boundary.

Two things it is *not*. It is not a level-set scheme: a real one reinitialises ``phi`` to a
signed distance function and advects with ``|grad phi|``, and this just adds the raw gradient.
And it is not tuned — the merit is one diffraction order, picked because it moves visibly in a
few steps.

**PyTorch rather than JAX, on purpose.** A level-set step usually moves the sign pattern, which
is a new plan, which under ``jax.jit`` is a new compilation — so at this truncation the eager
backend wins outright (the same crossover ``docs/backend-choice.md`` measures: PyTorch below
``fto`` about 5, JAX above). The path itself runs on either.

See ``docs/levelset-modeling.md`` for what it was measured against.
"""
import numpy as np
import torch

import fieldwave

PERIOD = 1.0
N = 48                                                    # design grid; >= 4*fto+1
FTO = 3
ORDER = (FTO + 1, FTO)                                    # the (0, +1) transmitted order


def initial_phi():
    """Two discs, apart. A signed distance function, which is what a level set wants."""
    j = (np.arange(N) + 0.5) / N * PERIOD
    X, Y = np.meshgrid(j, j, indexing='xy')
    phi = np.maximum(0.14 - np.hypot(X - 0.34, Y - 0.5),
                     0.14 - np.hypot(X - 0.66, Y - 0.5))
    return torch.tensor(phi, dtype=torch.float64)


def solve(phi):
    solver = fieldwave.rcwa(backend=2, period=[PERIOD, PERIOD], thickness=[0.35],
                            wavelength=0.85, theta=0.0, phi=0.0, fto=[FTO, FTO],
                            n_top=1.0, n_bot=1.45,
                            ucell=[dict(n_base=1.0, n_index=3.5, phi=phi)])
    # Re-freeze every step. The plan records one sign pattern and a level-set step is allowed
    # to move it, so it is re-taken rather than nursed along; `freeze_geometry` decides
    # discrete things and therefore reads concrete values, which is why it is called here and
    # not inside anything traced.
    return solver.conv_solve(geometry=solver.freeze_geometry()).res


def run():
    phi = initial_phi()
    res = solve(phi)
    print(f'forward: sum(de_ri) + sum(de_ti) = '
          f'{float(torch.sum(res.de_ri) + torch.sum(res.de_ti)):.12f}')

    print('\nstep   merit      |dJ/dphi|   material fraction')
    for step in range(8):
        phi = phi.detach().requires_grad_(True)
        J = torch.abs(solve(phi).de_ti[ORDER])
        J.backward()
        g = phi.grad
        print(f'{step:4d}   {float(J.detach()):.6f}   {float(torch.linalg.norm(g)):9.2f}   '
              f'{float(torch.mean((phi > 0).double())):.4f}')
        # A fixed move of the *boundary*, not of phi's values: normalising makes the step size
        # mean "this far, in units of the grid" whatever the merit's own scale happens to be.
        phi = phi.detach() + (0.5 * PERIOD / N) * g * (N / torch.linalg.norm(g))

    print(f'\nfinal merit = {float(torch.abs(solve(phi).de_ti[ORDER])):.6f}')


if __name__ == '__main__':
    run()
