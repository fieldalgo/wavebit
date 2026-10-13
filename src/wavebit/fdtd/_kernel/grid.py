"""The uniform Yee grid: where each of the six components sits, and how a point finds one.

Cell ``i`` along x covers ``[x_i, x_i + dx)`` with ``x_i = x_lo + i*dx``, and the same on y
and z with the one ``dx``. Inside the cell the six components sit at the Yee offsets, given
in half cells along ``(x, y, z)``:

    Ex (1, 0, 0)   Ey (0, 1, 0)   Ez (0, 0, 1)
    Hx (0, 1, 1)   Hy (1, 0, 1)   Hz (1, 1, 0)

that is, ``E_c`` is half a cell along its own axis and ``H_c`` half a cell along the other
two. All six components have the same array shape ``(nz, ny, nx)``: the walls are realised by
neighbour rules and masks (``faces.py``), never by an ``N + 1`` array, which is what lets one
``[component, z, y, x]`` block hold a field and one index address the same cell in every
component.

The box is snapped **up** to whole cells about its centre: ``n = ceil(size / dx)`` (with a
relative tolerance of 1e-9 so that ``1.0 / (1/16)`` is 16 cells and not 17), and the low
corner is ``center - n*dx/2``. The report states the snapped size beside the requested one.
"""
import dataclasses
import math

import numpy as np

__all__ = ['Grid', 'OFFSETS', 'cell_count']

#: Yee offsets in half cells, ``OFFSETS[field][component] = (ox, oy, oz)``.
OFFSETS = {
    'E': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    'H': ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
}


def cell_count(size, dx):
    """Whole cells covering ``size``: rounded up, unless the shortfall is round-off."""
    ratio = size / dx
    n = math.floor(ratio + 1e-9)
    if n * dx < size * (1 - 1e-9):
        n += 1
    return max(int(n), 1)


@dataclasses.dataclass(frozen=True)
class Grid:
    """``n = (nx, ny, nz)`` cells of size ``dx`` with low corner ``low = (x_lo, y_lo, z_lo)``."""

    n: tuple
    dx: float
    low: tuple

    @classmethod
    def from_domain(cls, size, center, dx):
        n = tuple(cell_count(s, dx) for s in size)
        low = tuple(c - k * dx / 2 for c, k in zip(center, n))
        return cls(n=n, dx=float(dx), low=low)

    @property
    def size(self):
        """The snapped box size ``(sx, sy, sz)``."""
        return tuple(k * self.dx for k in self.n)

    @property
    def high(self):
        return tuple(lo + s for lo, s in zip(self.low, self.size))

    @property
    def shape(self):
        """One field's array shape ``(3, nz, ny, nx)``."""
        return (3, self.n[2], self.n[1], self.n[0])

    def axis(self, a, half):
        """Coordinates along axis ``a`` (0, 1, 2) at the nodes (``half=0``) or half cells."""
        return self.low[a] + (np.arange(self.n[a]) + 0.5 * half) * self.dx

    def yee_points(self, field, c):
        """``(x, y, z)`` 1-D coordinate arrays of component ``c`` of ``field``."""
        return tuple(self.axis(a, OFFSETS[field][c][a]) for a in range(3))

    def meshgrid(self, field, c):
        """The three coordinate grids of component ``c`` of ``field``, each ``(nz, ny, nx)``."""
        x, y, z = self.yee_points(field, c)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        return X, Y, Z

    def nearest(self, field, c, at):
        """``(k, j, i)`` -- the z, y, x indices of the Yee point of ``field[c]`` nearest ``at``."""
        idx = []
        for a in range(3):
            o = 0.5 * OFFSETS[field][c][a]
            i = int(round((at[a] - self.low[a]) / self.dx - o))
            idx.append(min(max(i, 0), self.n[a] - 1))
        return (idx[2], idx[1], idx[0])
