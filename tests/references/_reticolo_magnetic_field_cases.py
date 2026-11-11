"""Cases for the RETICOLO **magnetic field** golden, shared by the generator and the test.

``parm.res1.change_index`` is how RETICOLO's ``res1`` puts a permeability in a texture: a row
``[n, n_x, n_y, n_z, m_x, m_y, m_z]`` swaps every region drawn with the index ``n`` for
``eps = n_i**2``, ``mu = m_i**2``. So a block's index below is its name for RETICOLO and
:data:`MATERIALS` says what it is on both sides.

Three grating paths, because the rule reaches ``H`` by different algebra on each: ``field_2d``
sums ``H_x`` and ``H_y`` line by line, ``field_1d`` takes TE's ``B_x`` off ``W q``, and
``field_1d_conical`` takes it off Faraday's x row. Each stack has layers with ``mu`` varying (on
2D a lossy one), a layer whose *background* is the magnetic material, a layer where ``mu`` alone
varies, its ``eps`` equal to the last bit so that a raster has no ``eps`` line there, and a last
layer where only ``eps`` varies and nothing names a ``mu``. So a rule reading the wrong field, the
wrong layer, the shapes without the background or the real part alone disagrees at O(1).
"""
import numpy as np

WL = 620.0
N_TOP, N_BOT = 1.0, 1.5
NPTS = 3                             # res3 points per profile entry, both faces included

NAMES = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
EINC = {'p': (1.0, 0.0), 's': (0.0, 1.0)}   # RETICOLO's `einc` is `[a_p, a_s]`

_LOSSY = 1.6 + 0.05j
#: An index in a texture -> ``(eps, mu)``; any other is ``(n**2, 1)``. Index 1.7 next to 1.4 draws
#: ``mu`` alone, whichever of the two is the background, and 1.8 is a magnetic background under
#: blocks of another ``eps``.
MATERIALS = {2.0: (4.0, 2.25),
             2.5: (6.25, 1.0),
             _LOSSY: (_LOSSY ** 2, (1.2 + 0.03j) ** 2),
             1.7: (1.4 ** 2, 1.44),
             1.8: (3.24, 1.21)}

# ------------------------------------------------------------------------------------- 2D

PERIOD_2D = (400.0, 360.0)
ORDER_2D = 5
ANGLE_2D = (25.0, 30.0)
#: ``(cx, cy, dx, dy, index)``, centre-origin lengths, top layer first, on no pixel line.
STACK_2D = [
    dict(thickness=180.0, background=1.0,
         blocks=[(-37.3, 23.1, 123.7, 91.9, 2.0), (101.3, -111.7, 77.7, 61.3, 2.5)]),
    dict(thickness=120.0, background=1.8,
         blocks=[(61.1, -81.3, 97.3, 131.9, _LOSSY), (-133.3, 99.1, 71.1, 83.3, 1.0)]),
    dict(thickness=90.0, background=1.7,
         blocks=[(-61.7, 17.3, 143.3, 91.1, 1.4), (123.1, -93.7, 61.9, 73.3, 1.4)]),
    dict(thickness=60.0, background=1.3, blocks=[(37.9, -41.3, 111.1, 87.7, 2.2)]),
]

# A raster of the same kind, whose walls are on pixel lines so it is what RETICOLO is given.
NX, NY = 8, 9
#: ``(row0, row1, col0, col1, index)``, top layer first.
GRID_2D = [
    dict(thickness=180.0, background=1.0, blocks=[(1, 5, 2, 6, 2.0), (5, 7, 1, 3, 2.5)]),
    dict(thickness=120.0, background=1.8, blocks=[(0, 3, 4, 7, _LOSSY), (6, 9, 0, 2, 1.0)]),
    dict(thickness=90.0, background=1.7, blocks=[(2, 6, 1, 4, 1.4), (6, 8, 5, 7, 1.4)]),
    dict(thickness=60.0, background=1.3, blocks=[(3, 7, 3, 6, 2.2)]),
]

# ------------------------------------------------------------------------------- 1D, conical

PERIOD_1D = 400.0
ORDER_1D = 8
NX_1D = 40
ANGLE_1D = 20.0
ANGLE_CONICAL = (20.0, 30.0)
#: ``(x0, x1, index)`` of one bar, top layer first. The second layer's bar changes ``mu`` alone,
#: to the last bit of ``eps``; the third's background is magnetic; the last has no ``mu``.
STACK_1D = [
    dict(thickness=100.0, background=1.0, bar=(130.0, 270.0, 2.0)),
    dict(thickness=70.0, background=1.4, bar=(40.0, 100.0, 1.7)),
    dict(thickness=60.0, background=1.7, bar=(220.0, 330.0, 2.5)),
    dict(thickness=50.0, background=1.3, bar=(90.0, 250.0, 2.2)),
]


def material_of(index):
    """``(eps, mu)`` of an index in a texture."""
    return MATERIALS.get(index, (index ** 2, 1.0))


def key(path, pol, champ):
    return f'champ{champ}__{path}_{pol}'


def samples_2d():
    """``(x, y)`` sample lines, centre-origin lengths, clear of every wall in both stacks."""
    x = (np.arange(12) + 0.75) * PERIOD_2D[0] / 12 - PERIOD_2D[0] / 2
    y = (np.arange(9) + 0.71) * PERIOD_2D[1] / 9 - PERIOD_2D[1] / 2
    return x, y


def samples_1d():
    """x samples from the cell's corner, RETICOLO's origin on a 1D texture, clear of every wall."""
    return np.linspace(0.0, PERIOD_1D, 11) + 7.3


def grid_blocks(layer):
    """RETICOLO texture rows ``[cx, cy, dx, dy, n, 1]`` of a :data:`GRID_2D` layer."""
    px, py = PERIOD_2D[0] / NX, PERIOD_2D[1] / NY
    return np.array([[((c0 + c1) / 2) * px - PERIOD_2D[0] / 2, ((r0 + r1) / 2) * py - PERIOD_2D[1] / 2,
                      (c1 - c0) * px, (r1 - r0) * py, n, 1.0]
                     for r0, r1, c0, c1, n in layer['blocks']], dtype=complex)


def shape_blocks(layer):
    """RETICOLO texture rows of a :data:`STACK_2D` layer."""
    return np.array([[cx, cy, dx, dy, n, 1.0] for cx, cy, dx, dy, n in layer['blocks']], dtype=complex)


def change_index():
    """``parm.res1.change_index``'s rows."""
    rows = []
    for n, (eps, mu) in MATERIALS.items():
        e, m = np.sqrt(complex(eps)), np.sqrt(complex(mu))
        rows.append(np.array([n, e, e, e, m, m, m], dtype=complex))
    return rows


def grid_fields():
    """:data:`GRID_2D` as this project takes it: ``(eps, mu)``, ``(layer, NY, NX)``, bottom first."""
    eps, mu = [], []
    for layer in GRID_2D[::-1]:
        background_eps, background_mu = material_of(layer['background'])
        e = np.full((NY, NX), background_eps, dtype=complex)
        m = np.full((NY, NX), background_mu, dtype=complex)
        for r0, r1, c0, c1, n in layer['blocks']:
            e[r0:r1, c0:c1], m[r0:r1, c0:c1] = material_of(n)
        eps.append(e)
        mu.append(m)
    return np.stack(eps), np.stack(mu)


def fields_1d():
    """:data:`STACK_1D` as ``(eps, mu)``, ``(layer, 1, NX_1D)``, bottom layer first."""
    centres = (np.arange(NX_1D) + 0.5) * PERIOD_1D / NX_1D
    eps, mu = [], []
    for layer in STACK_1D[::-1]:
        x0, x1, n = layer['bar']
        inside = (centres > x0) & (centres < x1)
        e, m = material_of(n)
        background_eps, background_mu = material_of(layer['background'])
        eps.append(np.where(inside, e, background_eps)[None].astype(complex))
        mu.append(np.where(inside, m, background_mu)[None].astype(complex))
    return np.stack(eps), np.stack(mu)
