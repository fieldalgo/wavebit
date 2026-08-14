"""Cases for the fmmax **field** golden, shared by the generator and the test.

Imports numpy and nothing else, because the two sides live in different environments: the
generator runs in ``fwref`` (fmmax, no wavebit) and the test in the project env (wavebit, no
fmmax). See ``README.md``.

**The two settings that make this comparison mean anything**, both found by measurement rather
than assumed — with either one wrong the grating cases sit at 1e-1 and look like a solver
disagreement:

* **The order sets must match.** fmmax's ``generate_expansion`` on a square lattice with
  ``approximate_num_terms=25`` and parallelogramic truncation returns a 5x5 set. Comparing that
  against this project's ``fto=(8, 0)`` — 17 orders along x, one along y — left a 1.2e-1
  residual that is entirely truncation. ``fto=(2, 2)`` is the same 5x5 set and drops it to
  1.9e-5.
* **The factorization rules must match.** fmmax's ``Formulation.FFT`` is the plain Laurent
  rule, which is the one Li's work exists because of: it converges badly in TM. It left the
  p-polarised cases at 2.7e-1 while TE was already at 1.9e-5. A vector formulation
  (``JONES_DIRECT``; ``NORMAL`` gives the same numbers here) brings p to 7.2e-5.

The geometry is 1D — a binary grating varying along x — but is given to both solvers as a **2D
cell tiled along y**. That is deliberate: it is what lets a 1D structure carry fmmax's square
5x5 order set, so the two truncations can be made identical at all.
"""
import numpy as np

WL = 620.0
PERIOD = 400.0
THICK = 260.0
NX = 400                  # raster columns
NY = 8                    # tiled rows; the structure is y-invariant, so this is exact
FTO = 2                   # (2, 2) -> 5x5 = 25 orders, matching fmmax's parallelogramic set
NUM_TERMS = (2 * FTO + 1) ** 2

RES_X = 9                 # fmmax grid_shape[0]; its x samples are CELL-CENTRED
RES_Y = 5                 # fmmax refuses a grid smaller than the expansion on either axis
RES_Z = 7                 # inclusive over [0, THICK] on both sides

N_HI = 2.0
FILL = 0.4

# (name, theta_deg, polarisation)
CASES = [
    ('uniform', 0.0, 's'), ('uniform', 0.0, 'p'),
    ('uniform', 24.0, 's'), ('uniform', 24.0, 'p'),
    ('grating', 0.0, 's'), ('grating', 0.0, 'p'),
    ('grating', 24.0, 's'), ('grating', 24.0, 'p'),
    ('grating', 41.0, 's'), ('grating', 41.0, 'p'),
]


def raster(name):
    """The layer's refractive index as ``[y, x]``, which is this project's raster layout.

    fmmax wants ``[x, y]``; its generator transposes. That transpose is not cosmetic — see
    ``docs`` on the fmmax raster convention, where getting it wrong reverses the s/p split and
    reads like a sign error.
    """
    if name == 'uniform':
        return np.full((NY, NX), 1.5)
    row = np.ones(NX)
    lo = int((0.5 - FILL / 2) * NX)
    row[lo:lo + int(FILL * NX)] = N_HI
    return np.tile(row[None, :], (NY, 1))


def x_samples():
    """fmmax's cell-centred x grid, ``(j + 0.5) * PERIOD / RES_X``.

    This project samples ``linspace(0, period, res_x)`` inclusive, so the grids do not line up.
    Asking it for ``2*RES_X + 1`` points puts index ``2j+1`` exactly on fmmax's ``j``-th — no
    interpolation anywhere in the comparison.
    """
    return (np.arange(RES_X) + 0.5) * PERIOD / RES_X


def fine_res_x():
    return 2 * RES_X + 1
