"""Cases shared by ``gen_goldens_fmmax_patterned_mu.py`` and ``test_fmmax_patterned_mu_golden.py``.

Same split as the other fmmax pairs — the generator runs in ``fwref``, the test in the project
env, so the shared module imports numpy and nothing else.

**Patterned permeability**: ``mu`` varies across the cell, which needs a factorization on the
``mu`` side exactly as a patterned tensor ``eps`` did. Frozen before that factorization was
wired to ``mu``.

Three things these cases are built to expose, none of which a single patterned-mu shape would:

* **the pattern in ``mu`` alone.** ``uniform_eps_patterned_mu`` leaves ``eps`` constant, so
  every bit of structure in the answer comes from the permeability. A path that factorized
  ``eps`` properly and averaged ``mu`` would still get an interesting-looking answer on any
  case where both are patterned.
* **two different shapes in one layer.** ``different_shapes`` patterns ``eps`` as one rectangle
  and ``mu`` as another. The Popov-Neviere rearrangement is written around *one* interface
  normal, so a layer whose two tensors have different boundaries needs two normal fields — and
  a builder that reuses one for both would pass every same-shape case.
* **a patterned off-diagonal**, which is the only part of ``mu`` that cannot be reached by any
  amount of care with the diagonal.

The raster convention is this project's ``[y, x]``; the generator transposes for fmmax. As with
the patterned ``eps`` golden the pillars are deliberately **not square**, because a square one is
invariant under the axis swap and would hide a mix-up.
"""
import numpy as np

WL = 700.0
PERIOD = 400.0
THICK = 210.0
GRID = 96
THETA = 8.0

# High, because the oracle arbitrates by convergence: fmmax's own two formulations are still
# some way apart at this order and no claim can be sharper than that gap.
MAX_ORDER = 12


def rect(value, background, y0, y1, x0, x1, grid=GRID):
    a = np.full((grid, grid), background, dtype=complex)
    a[int(y0 * grid):int(y1 * grid), int(x0 * grid):int(x1 * grid)] = value
    return a


def pillar(value, background, grid=GRID):
    """The shared shape: wider in x than in y, so the axis convention is visible."""
    return rect(value, background, 1 / 3, 2 / 3, 1 / 4, 2 / 3, grid)


def offset_pillar(value, background, grid=GRID):
    """A *different* rectangle — used where eps and mu must not share a boundary."""
    return rect(value, background, 1 / 5, 3 / 5, 2 / 5, 5 / 6, grid)


def uniform(value, grid=GRID):
    return np.full((grid, grid), value, dtype=complex)


# name -> dict of rasters. `eps` / `mu` are the scalar diagonals (xx = yy = zz) and
# `eps_xy` / `mu_xy` the gyrotropic off-diagonals (the tensor is +g / -g, hence Hermitian and
# lossless). Any key may be absent.
def CASES():
    return {
        'uniform_eps_patterned_mu': dict(
            eps=uniform(2.56), mu=pillar(1.9, 1.0)),
        'patterned_eps_patterned_mu': dict(
            eps=pillar(2.56, 1.0), mu=pillar(1.9, 1.0)),
        'patterned_mu_only_weak': dict(
            eps=uniform(2.2), mu=pillar(1.35, 1.0)),
        'different_shapes': dict(
            eps=pillar(2.56, 1.0), mu=offset_pillar(1.9, 1.0)),
        'patterned_gyromagnetic': dict(
            eps=pillar(2.56, 1.0), mu=pillar(1.9, 1.0), mu_xy=pillar(0.35j, 0.0)),
        'patterned_gyroelectric_and_gyromagnetic': dict(
            eps=pillar(2.56, 1.0), mu=pillar(1.9, 1.0),
            eps_xy=pillar(0.30j, 0.0), mu_xy=pillar(0.35j, 0.0)),
    }


def tensors(case):
    """``(eps, eps_xy, mu, mu_xy)`` rasters, absent keys filled in."""
    z = np.zeros((GRID, GRID), dtype=complex)
    return (case['eps'], case.get('eps_xy', z), case['mu'], case.get('mu_xy', z))
