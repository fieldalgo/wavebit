"""Cases for the RETICOLO golden of the Lalanne rule off its first paths, shared by generator and test.

Each family reaches a path the scalar goldens do not, and RETICOLO answers each through ``res1``
and ``res3`` or, for a source, through the ``ret*`` functions under them:

* ``TENSOR_*``: a diagonal tensor in a patterned layer, through ``parm.res1.change_index``. Each
  layer of the 2D stack fails one way of reading the wrong entry: all six entries differ, ``eps_xx``
  alone agrees across the wall, ``eps_yy`` alone agrees, only the z entries differ, only ``mu`` is a
  tensor, and a uniform tensor background.
* ``UNIFORM_*``: a uniform layer with an off-diagonal pair beside a patterned one, on either
  formulation. One pair at ``delta0 = 0`` is the region where RETICOLO's anisotropic solve conserves
  energy (``gen_goldens_reticolo_anisotropic.py``).
* ``MERGED_*``: layers of a half-space's own medium, some of which this project merges into it.
* ``ONE_ORDER``: one retained order along an axis of a cell patterned along both.
* ``SOURCE``: an electric dipole in a uniform layer under a patterned one.
* ``bottom``: the tensor stack lit from below, through ``parm.res3.sens``.

A block's index is its name for RETICOLO, and :data:`TABLE` says what it is on both sides. The names
are unusual on purpose: ``change_index`` replaces any index within rounding of a name, half-spaces
included.
"""
import numpy as np

WL = 620.0
N_TOP, N_BOT = 1.0, 1.5
NPTS = 3
NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
EINC = {'p': (1.0, 0.0), 's': (0.0, 1.0)}   # RETICOLO's `einc` is `[a_p, a_s]`

_ONE = (1.0, 1.0, 1.0)
#: A name in a texture -> ``((eps_xx, eps_yy, eps_zz), (mu_xx, mu_yy, mu_zz))``.
TABLE = {
    0.1101: ((4.0, 6.25, 2.25), (1.21, 1.44, 1.69)),
    0.1102: (((1.6 + 0.05j) ** 2, 3.24, 2.56), (1.1, 1.3, 1.2)),
    0.1103: ((2.25, 4.0, 3.0), _ONE),
    0.1104: ((2.25, 9.0, 3.0), _ONE),
    0.1105: ((4.0, 2.25, 3.0), _ONE),
    0.1106: ((9.0, 2.25, 3.0), _ONE),
    0.1107: ((3.0, 3.0, 4.0), (1.0, 1.0, 1.2)),
    0.1108: ((3.0, 3.0, 7.0), (1.0, 1.0, 2.0)),
    0.1109: ((2.0, 2.0, 2.0), (1.44, 2.25, 1.21)),
    0.1110: ((2.0, 2.0, 2.0), (1.21, 1.0, 1.44)),
    0.1111: ((2.56, 3.61, 4.41), _ONE),
    0.1112: ((4.0, 4.0, 4.0), (2.25, 2.25, 2.25)),
}

# ---------------------------------------------------------------------------------- tensors, 2D

PERIOD_2D = (400.0, 360.0)
ORDER_2D = 3
ANGLE_2D = (25.0, 30.0)
NX, NY = 8, 9
#: ``(row0, row1, col0, col1, name)``, top layer first, walls on the raster's pixel lines.
TENSOR_2D = [
    dict(thickness=120.0, background=1.0, blocks=[(1, 5, 2, 6, 0.1101), (6, 8, 1, 3, 0.1102)]),
    dict(thickness=80.0, background=0.1103, blocks=[(2, 6, 1, 5, 0.1104)]),
    dict(thickness=70.0, background=0.1105, blocks=[(3, 7, 3, 7, 0.1106)]),
    dict(thickness=60.0, background=0.1107, blocks=[(1, 4, 4, 7, 0.1108)]),
    dict(thickness=50.0, background=0.1109, blocks=[(4, 8, 2, 5, 0.1110)]),
    dict(thickness=40.0, background=0.1111, blocks=[]),
]

# ---------------------------------------------------------------------------------- tensors, 1D

PERIOD_1D = 400.0
ORDER_1D = 6
NX_1D = 40
ANGLE_1D = 20.0
ANGLE_CONICAL = (20.0, 30.0)
#: ``(x0, x1, name)`` of one bar, top layer first, walls on the raster's pixel lines.
TENSOR_1D = [
    dict(thickness=100.0, background=1.0, bar=(130.0, 270.0, 0.1101)),
    dict(thickness=70.0, background=0.1103, bar=(40.0, 100.0, 0.1104)),
    dict(thickness=60.0, background=0.1109, bar=(220.0, 330.0, 0.1110)),
]

# ------------------------------------------------------------------------------------ one pair

#: A uniform layer's 3x3 ``eps``, with the patterned layer :data:`UNIFORM_BLOCK` over it.
UNIFORM_TENSOR = {'xy': np.array([[3.0, 0.4, 0.0], [0.4, 3.5, 0.0], [0.0, 0.0, 2.5]], dtype=complex),
                  'xz': np.array([[3.0, 0.0, 0.4], [0.0, 3.5, 0.0], [0.4, 0.0, 2.5]], dtype=complex)}
UNIFORM_BLOCK = (1, 5, 2, 6, 0.1101)
UNIFORM_THICKNESS = (180.0, 90.0)     # the patterned layer, then the uniform one
UNIFORM_ANGLE = (25.0, 0.0)

# ------------------------------------------------------------------------------------- merging

#: Top layer first. ``merges`` marks the outer layers of a half-space's own medium this project
#: merges when they are within the tolerance, and the two inner ones can never merge.
MERGED_2D = [
    dict(thickness=130.0, background=1.0, blocks=[], merges=True),
    dict(thickness=70.0, background=1.0, blocks=[], merges=True),
    dict(thickness=110.0, background=1.0, blocks=[(1, 5, 2, 6, 0.1112)], merges=False),
    dict(thickness=60.0, background=1.0, blocks=[], merges=False),
    dict(thickness=50.0, background=1.5, blocks=[], merges=False),
    dict(thickness=90.0, background=1.3, blocks=[(3, 7, 3, 7, 2.2)], merges=False),
    dict(thickness=120.0, background=1.5, blocks=[], merges=True),
]
MERGED_1D = [
    dict(thickness=130.0, background=1.0, bar=None, merges=True),
    dict(thickness=70.0, background=1.0, bar=None, merges=True),
    dict(thickness=110.0, background=1.0, bar=(130.0, 270.0, 0.1112), merges=False),
    dict(thickness=60.0, background=1.0, bar=None, merges=False),
    dict(thickness=90.0, background=1.3, bar=(40.0, 100.0, 2.2), merges=False),
    dict(thickness=120.0, background=1.5, bar=None, merges=True),
]
#: How far a merging layer sits from its half-space's ``eps``: within ``MERGE_TOLERANCE``, unequal.
MERGE_NUDGE, MERGE_TOLERANCE = 3e-10, 1e-9

# ------------------------------------------------------------------------------------ one order

ONE_ORDER_2D = [
    dict(thickness=120.0, background=1.0, blocks=[(1, 5, 2, 6, 0.1112)]),
    dict(thickness=80.0, background=1.3, blocks=[(3, 7, 3, 7, 2.2)]),
]

# --------------------------------------------------------------------------------------- source

#: From the top: the patterned layer, then the uniform host with a thin row around the dipole.
SOURCE_BLOCK = (-37.3, 23.1, 123.7, 91.9, 2.0)
SOURCE_PATTERNED = 150.0
SOURCE_HOST = (100.0, 97.0)           # above and below the dipole's thin row
SOURCE_ROW = 0.002
SOURCE_HOST_INDEX = 1.3
SOURCE_POSITION = (31.1, -22.7)
#: RETICOLO's source component (1, 2, 3 for ``J_x``, ``J_y``, ``J_z``) and this project's moment.
SOURCE_MOMENTS = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}
SOURCE_ORDER = 5


def key(path, pol, champ):
    return f'champ{champ}__{path}_{pol}'


def material_of(name):
    """``(eps, mu)`` entries of a name in a texture; any other index is scalar with no ``mu``."""
    if name in TABLE:
        return TABLE[name]
    return (name ** 2,) * 3, _ONE


def samples_2d():
    """``(x, y)`` sample lines, centre-origin lengths, clear of every pixel line."""
    x = (np.arange(6) + 0.73) * PERIOD_2D[0] / 6 - PERIOD_2D[0] / 2
    y = (np.arange(5) + 0.71) * PERIOD_2D[1] / 5 - PERIOD_2D[1] / 2
    return x, y


def samples_1d():
    """x samples from the cell's corner, RETICOLO's origin on a 1D texture, clear of every wall."""
    return np.linspace(0.0, PERIOD_1D, 7) + 7.3


def blocks_of(layer):
    """RETICOLO texture rows ``[cx, cy, dx, dy, n, 1]`` of a raster-drawn layer."""
    px, py = PERIOD_2D[0] / NX, PERIOD_2D[1] / NY
    rows = [[((c0 + c1) / 2) * px - PERIOD_2D[0] / 2, ((r0 + r1) / 2) * py - PERIOD_2D[1] / 2,
             (c1 - c0) * px, (r1 - r0) * py, n, 1.0] for r0, r1, c0, c1, n in layer['blocks']]
    return np.array(rows, dtype=complex).reshape(-1, 6)


def rectangle_of(block):
    """A raster-drawn block as ``(cx, cy, lx, ly)``, centre-origin lengths."""
    r0, r1, c0, c1, _ = block
    px, py = PERIOD_2D[0] / NX, PERIOD_2D[1] / NY
    return (((c0 + c1) / 2) * px - PERIOD_2D[0] / 2, ((r0 + r1) / 2) * py - PERIOD_2D[1] / 2,
            (c1 - c0) * px, (r1 - r0) * py)


def change_index():
    """``parm.res1.change_index``'s rows, ``[n, n_x, n_y, n_z, m_x, m_y, m_z]``."""
    return [np.array([n, *np.sqrt(np.array(e, dtype=complex)), *np.sqrt(np.array(m, dtype=complex))], dtype=complex)
            for n, (e, m) in TABLE.items()]


def raster_2d(stack):
    """A stack as tensor rasters ``(eps, mu)``, ``(layer, NY, NX, 3, 3)``, bottom layer first."""
    eps, mu = [], []
    for layer in stack[::-1]:
        e_bg, m_bg = material_of(layer['background'])
        e, m = (np.full((NY, NX, 3), v, dtype=complex) for v in (e_bg, m_bg))
        for r0, r1, c0, c1, n in layer['blocks']:
            e[r0:r1, c0:c1], m[r0:r1, c0:c1] = material_of(n)
        eps.append(e[..., None] * np.eye(3))
        mu.append(m[..., None] * np.eye(3))
    return np.stack(eps), np.stack(mu)


def raster_1d(stack):
    """A 1D stack as one-row tensor rasters, ``(layer, 1, NX_1D, 3, 3)``, bottom layer first."""
    centres = (np.arange(NX_1D) + 0.5) * PERIOD_1D / NX_1D
    eps, mu = [], []
    for layer in stack[::-1]:
        e_bg, m_bg = material_of(layer['background'])
        e, m = np.full((NX_1D, 3), e_bg, dtype=complex), np.full((NX_1D, 3), m_bg, dtype=complex)
        if layer['bar'] is not None:
            x0, x1, n = layer['bar']
            inside = (centres > x0) & (centres < x1)
            e[inside], m[inside] = material_of(n)
        eps.append(e[None, ..., None] * np.eye(3))
        mu.append(m[None, ..., None] * np.eye(3))
    return np.stack(eps), np.stack(mu)
