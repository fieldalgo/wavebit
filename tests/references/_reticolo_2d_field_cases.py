"""Cases for the RETICOLO **2D field** golden, shared by the generator and the test.

The 1D and conical field goldens beside this one cover gratings. This one is a crossed grating:
two patterned layers on a rectangular cell, each a few axis-aligned blocks whose edges sit on
the raster's pixel lines, so the raster this project solves *is* the structure RETICOLO is given
as rectangles and Li's rules build the same matrices on both sides. It is what pins the
line-wise continuity reconstruction of the in-plane ``E`` (``discontinuous='continuity'``)
against RETICOLO's ``champ = 1``.

Two choices are there to make a wrong answer visible rather than plausible:

* the cell is **not square** (400 x 360 nm, 8 x 9 pixels), and no block is symmetric under
  ``x <-> y``, so a swapped axis or a transposed raster disagrees at O(1);
* the two layers have their walls in different places, so a rule reading the other layer's
  permittivity puts a wall where there is none.

The samples avoid every pixel line by at least 8 nm: on a wall the two codes may take the two
sides of the jump.
"""
import numpy as np

WL = 620.0
PERIOD = (400.0, 360.0)              # (x, y)
NX, NY = 8, 9                        # raster columns (x) and rows (y)
MAX_ORDER = 5                        # orders -5..5 on each axis
N_TOP, N_BOT = 1.0, 1.5
NPTS = 5                             # res3 points per profile entry, both faces included

NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

# (row0, row1, col0, col1, refractive index): rows are y pixels, columns x pixels, ends
# exclusive, from the cell's negative corner. Blocks within a layer never overlap.
_LOSSY = 1.6 + 0.05j
LAYERS = [                           # top to bottom, RETICOLO's profile order
    dict(thickness=180.0, background=1.0,
         blocks=[(1, 5, 2, 6, 2.0), (5, 7, 1, 3, 2.5)]),
    dict(thickness=120.0, background=1.5,
         blocks=[(0, 3, 4, 7, _LOSSY), (6, 9, 0, 2, 1.0)]),
]

# (theta, phi) in degrees, and the incidences: RETICOLO's `einc` is `[a_p, a_s]`.
ANGLES = [(0.0, 0.0), (25.0, 30.0)]
EINC = {'p': (1.0, 0.0), 's': (0.0, 1.0)}


def key(angle, pol):
    theta, phi = angle
    return f'{theta:g}_{phi:g}_{pol}'


def raster():
    """The structure as this project takes it: permittivity, ``(layer, NY, NX)``, bottom
    layer first (the public order); rows are y."""
    out = []
    for layer in LAYERS[::-1]:
        grid = np.full((NY, NX), layer['background'] ** 2, dtype=complex)
        for r0, r1, c0, c1, n in layer['blocks']:
            grid[r0:r1, c0:c1] = n ** 2
        out.append(grid)
    return np.stack(out)


def thickness():
    """Layer thicknesses, bottom layer first."""
    return [layer['thickness'] for layer in LAYERS[::-1]]


def blocks(layer):
    """RETICOLO texture rows ``[cx, cy, dx, dy, n, 1]`` of one layer, in centre-origin lengths."""
    px, py = PERIOD[0] / NX, PERIOD[1] / NY
    return np.array([[((c0 + c1) / 2) * px - PERIOD[0] / 2, ((r0 + r1) / 2) * py - PERIOD[1] / 2,
                      (c1 - c0) * px, (r1 - r0) * py, n, 1.0]
                     for r0, r1, c0, c1, n in layer['blocks']], dtype=complex)


def samples():
    """``(x, y)`` sample lines, centre-origin lengths, each at least 8 nm off every pixel line."""
    x = (np.arange(12) + 0.75) * PERIOD[0] / 12 - PERIOD[0] / 2
    y = (np.arange(9) + 0.71) * PERIOD[1] / 9 - PERIOD[1] / 2
    for line, period, n in ((x, PERIOD[0], NX), (y, PERIOD[1], NY)):
        pixel = period / n
        off = np.abs((line + period / 2) / pixel - np.round((line + period / 2) / pixel)) * pixel
        assert off.min() > 8.0, off.min()
    return x, y
