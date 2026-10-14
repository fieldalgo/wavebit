"""Cases for the fmmax **field** golden, shared by the generator and the test.

Imports numpy and nothing else, because the two sides live in different environments: the
generator runs in ``fwref`` (fmmax, no wavebit) and the test in the project env (wavebit, no
fmmax). See ``README.md``.

**The two settings that make this comparison mean anything**, both found by measurement rather
than assumed — with either one wrong the grating cases sit at 1e-1 and look like a solver
disagreement:

* **The order sets must match.** fmmax's ``generate_expansion`` on a square lattice with
  ``approximate_num_terms=25`` and parallelogramic truncation returns a 5x5 set. Comparing that
  against this project's ``truncation_order=(8, 0)`` — 17 orders along x, one along y — left a 1.2e-1
  residual that is entirely truncation. ``truncation_order=(2, 2)`` is the same 5x5 set and drops it to
  1.9e-5.
* **The factorization rules must match.** fmmax's ``Formulation.FFT`` is the plain Laurent
  rule, which is the one Li's work exists because of: it converges badly in TM. It left the
  p-polarised cases at 2.7e-1 while TE was already at 1.9e-5. A vector formulation
  (``JONES_DIRECT``; ``NORMAL`` gives the same numbers here) brings p to 7.2e-5.

Two geometries live here, and they reach different things.

``CASES`` — a binary grating varying along **x only**, given to both solvers as a 2D cell tiled
along y. The tiling is deliberate: it is what lets a 1D structure carry fmmax's square 5x5 order
set, so the two truncations can be made identical at all. But the structure is y-invariant, so
the reconstructed field is y-constant to machine precision and no ``ky != 0`` order carries
amplitude.

``CASES_2D`` — an x/y-asymmetric crossed patch, which does vary in y. Its settings are
documented at its own block below.
"""
import numpy as np

WL = 620.0
PERIOD = 400.0
THICK = 260.0
NX = 400                  # raster columns
NY = 8                    # tiled rows; the structure is y-invariant, so this is exact
MAX_ORDER = 2                   # (2, 2) -> 5x5 = 25 orders, matching fmmax's parallelogramic set
NUM_TERMS = (2 * MAX_ORDER + 1) ** 2

RES_X = 9                 # fmmax grid_shape[0]; its x samples are CELL-CENTRED
RES_Y = 5                 # fmmax refuses a grid smaller than the expansion on either axis
RES_Z = 7                 # inclusive over [0, THICK] on both sides

EPS_HI = 4.0              # the index-2 bar this case list has always used, as a permittivity
EPS_BG = 2.25             # and the index-1.5 uniform slab
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
    """The layer's **permittivity** as ``[y, x]``, which is this project's raster layout.

    fmmax wants ``[x, y]``; its generator transposes. That transpose is not cosmetic — see
    ``docs`` on the fmmax raster convention, where getting it wrong reverses the s/p split and
    reads like a sign error.

    Permittivity is now what *both* consumers want, which is why the squaring that used to sit
    in the generator is gone rather than moved: fmmax's ``eigensolve_isotropic_media`` always
    took a permittivity, and since 2026-08-20 so does ``wavebit.rcwa``. One quantity, stated
    once — and the numbers are the squares of the indices frozen into the golden, so the scene
    behind it has not moved.
    """
    if name == 'uniform':
        return np.full((NY, NX), EPS_BG)
    row = np.ones(NX)
    lo = int((0.5 - FILL / 2) * NX)
    row[lo:lo + int(FILL * NX)] = EPS_HI
    return np.tile(row[None, :], (NY, 1))


def x_samples():
    """fmmax's cell-centred x grid, ``(j + 0.5) * PERIOD / RES_X``.

    This project samples ``linspace(0, period, sample_x)`` inclusive, so the grids do not line up.
    Asking it for ``2*RES_X + 1`` points puts index ``2j+1`` exactly on fmmax's ``j``-th — no
    interpolation anywhere in the comparison.
    """
    return (np.arange(RES_X) + 0.5) * PERIOD / RES_X


def fine_res_x():
    return 2 * RES_X + 1


# ---------------------------------------------------------------------------------------
# The 2D case: an x/y-asymmetric crossed patch (added 2026-08-15, rewritten 2026-10-14).
#
# Everything above is y-invariant: the reconstructed field's y-variation is 2.5e-15 of
# ``max|F|``, so every ``ky != 0`` order carries zero amplitude and the ``Ky`` terms of the ``Ez``
# / ``Hz`` assignment multiply nothing there. This patch puts amplitude on those orders: its
# golden field varies in y by 3.6e-1.
#
# ``300 x 120`` nm, centred at ``(-70, +60)`` in a 500 nm square cell (centre-origin
# coordinates; ``(180, 310)`` from the corner, where fmmax puts its origin). Unequal in x and y
# and off-centre in both, so no mirror or x<->y symmetry survives for a swap to hide behind, and
# reversing the raster's rows moves the patch.
#
# **Three choices that are not free, each settled by measurement on 2026-08-15:**
#
# * **Dielectric, not metal.** The same patch in gold (``eps = -5.6528 + 1.1456j``) is unusable as
#   an oracle: the two codes disagree by 0.35 to 1.5 even at 49 orders, because metal at that
#   contrast is converged in neither. ``eps = 4.0`` lands at 3e-2 instead.
# * **phi stays 0.** fmmax's mode index ``0`` / ``n`` is s / p only when the incident ``ky`` is
#   zero; at ``phi = 33 deg`` the *uniform* control, which has nothing to truncate, came back at
#   3.8e-1. The ``ky`` amplitude here comes from the **structure**, not from the incidence.
# * **Two truncations, 49 and 361 orders.** The residual does not shrink between them, which is
#   the evidence that the floor is the two vector factorizations disagreeing near the
#   rectangle's corners rather than truncation.

WL_2D = 630.0
PERIOD_2D = 500.0
THICK_2D = 50.0
THETA_2D = 41.0
NRAS_2D = 100             # raster, square so the ``.T`` convention stays testable
EPS_PATCH_2D = 4.0        # the patch, on a vacuum background
EPS_UNIFORM_2D = 2.25     # the control layer
PATCH_CX, PATCH_HX = -70.0, 150.0     # centre and half-width along x, from the cell centre
PATCH_CY, PATCH_HY = 60.0, 60.0       # ... and along y; both differ from x on purpose

RES_X_2D, RES_Y_2D, RES_Z_2D = 9, 7, 5

# (name, truncation order on both axes, polarisation) -- all at THETA_2D, phi = 0
CASES_2D = [('uniform2d', 3, 's'), ('uniform2d', 3, 'p')]
CASES_2D += [('patch', order, pol) for order in (3, 9) for pol in ('s', 'p')]


def raster_2d(name):
    """The layer's **permittivity** as ``[y, x]``, this project's layout; fmmax wants ``[x, y]``.

    Row 0 is the ``-y`` end of the cell. Pixel centres sit at ``(i + 0.5) * P/N - P/2``.
    """
    if name == 'uniform2d':
        return np.full((NRAS_2D, NRAS_2D), EPS_UNIFORM_2D, dtype=complex)
    ax = (np.arange(NRAS_2D) + 0.5) / NRAS_2D * PERIOD_2D - PERIOD_2D / 2
    X, Y = np.meshgrid(ax, ax, indexing='xy')          # X varies along axis 1 == x
    cell = np.ones((NRAS_2D, NRAS_2D), dtype=complex)
    cell[(np.abs(X - PATCH_CX) <= PATCH_HX) & (np.abs(Y - PATCH_CY) <= PATCH_HY)] = EPS_PATCH_2D
    return cell


def x_samples_2d():
    """fmmax's cell-centred x grid, measured from the cell **corner** (fmmax's origin).

    This library measures explicit ``sample_x`` from the cell centre: subtract ``PERIOD_2D/2``.
    """
    return (np.arange(RES_X_2D) + 0.5) * PERIOD_2D / RES_X_2D


def y_samples_2d():
    """fmmax's cell-centred y grid, from the corner -- the axis the 1D cases never sample."""
    return (np.arange(RES_Y_2D) + 0.5) * PERIOD_2D / RES_Y_2D


def z_samples_2d():
    """fmmax's heights inside the layer, from the face the light enters (the top), downward."""
    return np.linspace(0.0, THICK_2D, RES_Z_2D)


def grid_2d():
    """What to ask fmmax for, and where the stored points sit inside it.

    fmmax refuses a grid smaller than the expansion on either axis, so 361 orders cannot be
    sampled on the 9x7 grid that 49 orders use. Asking for **3x** the reference grid clears that
    (27x21 covers up to truncation order 10), and a cell-centred grid of ``3k`` points contains
    the ``k``-point one exactly at index ``3i+1``, so every case is stored on the *same* points
    and no interpolation enters the comparison.
    """
    return ((3 * RES_X_2D, 3 * RES_Y_2D), [3 * i + 1 for i in range(RES_X_2D)],
            [3 * i + 1 for i in range(RES_Y_2D)])
