"""Slabs -- the sub-boxes everything additive or multiplicative touches -- and the masks.

A ``Slab`` names one component's sub-box ``[z0, z1) x [y0, y1) x [x0, x1)`` of a field
array. It is the unit of every write the kernel makes outside the curl and the update: a
term adds into a slab (``rt.slab_add``), a mask multiplies one (``rt.slab_mul``). The slab is
static -- integers only, part of the plan and of its skeleton hash -- so the runtime can
precompute whatever its backend needs for it once: NumPy takes a view, Torch a view, and
JAX the four index arrays of a scatter (see ``wavebit/fd/_runtime.py`` for why a scatter and
not a slice).

One component per slab, on purpose. ``a[[1, 2], ..., 0]`` is a copy in NumPy and in Torch,
so a two-component slab could not be written in place; two one-component slabs can.

A ``Mask`` is a slab of ``E`` or ``H`` multiplied by zero after that field's update. It is
how a wall is realised on the face whose neighbour rule is not itself the wall: PEC
tangential E on the low-face node layer, PMC tangential H on the high-face layer, and the
odd components of a mirror on its plane (``faces.py`` says which).
"""
import dataclasses

__all__ = ['Slab', 'Mask']


@dataclasses.dataclass(frozen=True)
class Slab:
    """Component ``c`` of a ``(3, nz, ny, nx)`` array over ``z``, ``y``, ``x`` half-open ranges."""

    c: int
    z: tuple
    y: tuple
    x: tuple

    @property
    def slices(self):
        """The basic-indexing tuple, a view on NumPy and Torch."""
        return (self.c, slice(*self.z), slice(*self.y), slice(*self.x))

    @property
    def shape(self):
        return (self.z[1] - self.z[0], self.y[1] - self.y[0], self.x[1] - self.x[0])

    @classmethod
    def layer(cls, c, axis, index, n):
        """The whole layer ``index`` of component ``c`` across axis ``axis`` (0 = x, 1 = y, 2 = z)."""
        nx, ny, nz = n
        z, y, x = (0, nz), (0, ny), (0, nx)
        one = (index, index + 1)
        if axis == 0:
            x = one
        elif axis == 1:
            y = one
        else:
            z = one
        return cls(c=c, z=z, y=y, x=x)


@dataclasses.dataclass(frozen=True)
class Mask:
    """Zero ``slab`` of ``field`` (``'E'`` or ``'H'``) after that field's update."""

    field: str
    slab: Slab

    def after_E(self, E, rt):
        return rt.slab_mul(E, self.slab, 0.0) if self.field == 'E' else E

    def after_H(self, H, rt):
        return rt.slab_mul(H, self.slab, 0.0) if self.field == 'H' else H
