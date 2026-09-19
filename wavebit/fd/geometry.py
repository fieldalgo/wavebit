"""Where a material sits: a cell-snapped slab, and the structure that fills it.

Stage 1 has **one** shape, and it is the one a layer stack is made of::

    fd.Slab(z=(-0.2, 0.0))                 # one axis named; x and y run the whole box
    fd.Structure(fd.Slab(z=(-0.2, 0.0)), glass)

A ``Slab`` names exactly one axis. Two axes would be a box, an ``x``/``y`` extent a
rectangle, and every other shape -- a sphere, a cylinder, a raster, a polygon -- is stage 4.
Each is refused by name rather than by the generic keyword message, so the reader learns the
plan and not only the spelling.

**Slabs are cell-snapped, not subpixel.** The interface lands on the nearest *cell boundary*
of the named axis, and everything in the cells between the two snapped boundaries is the
structure's material -- every component of them, including the ones whose Yee point sits
exactly on a boundary. That is one rule for all six components and it is exactly what makes
it cheap: the material is a range of cell indices, so the coefficient arrays are built by
assignment and nothing is averaged.

It is also, deliberately, the *inaccurate* choice, and the report says so: an interface that
does not fall on a cell boundary is moved by up to half a cell, which is a first-order error
in ``dx`` where the rest of the scheme is second order. Until subpixel averaging arrives in
stage 4, **a convergence study in ``dx`` is the only route to a right answer** for a
structure whose thickness is not a whole number of cells -- and ``check`` prints the snapped
span beside the one that was asked for so that the size of the move is never a surprise.

The snapping rule, written once: with the grid's low corner ``lo`` and cell size ``dx``,

    k0 = round((z_lo - lo) / dx),   k1 = round((z_hi - lo) / dx)

clipped to ``[0, n]``, and the structure occupies cells ``k0 .. k1 - 1``. A slab whose two
boundaries snap to the same cell boundary is refused: it asked for a layer and would have
got nothing, which is the one snapping outcome a reader cannot see in a result.
"""
from ._spec import spec, static_float

__all__ = ['Slab', 'Structure', 'Box', 'AXES']

#: The three axes in the order the public API spells a triple.
AXES = ('x', 'y', 'z')

#: Shape keywords stage 4 will take. Refused here by name, with the stage.
_LATER_SHAPES = {
    'radius': 'a sphere or a cylinder', 'center': 'a box or a sphere',
    'size': 'a box', 'vertices': 'a polygon', 'raster': 'a raster',
    'angle': 'a rotated shape', 'height': 'a cylinder',
}


@spec
class Slab:
    """A layer: one axis named as ``(lo, hi)``, the other two running the whole box.

    ``fd.Slab(z=(-0.2, 0.0))`` is a layer 0.2 thick whose faces are at ``z = -0.2`` and
    ``z = 0``, in the scene's length unit and in the scene's coordinates (the box's centre
    is wherever ``Domain(center=...)`` put it, not the origin by default). The bounds are
    static -- they fix which cells carry which coefficients, and so the plan and its
    skeleton -- so a traced bound is refused by name.
    """

    x: tuple = None
    y: tuple = None
    z: tuple = None

    def __post_init__(self):
        named = [a for a in AXES if getattr(self, a) is not None]
        if len(named) != 1:
            if len(named) == 0:
                raise ValueError(
                    'Slab(): name exactly one axis, as Slab(z=(lo, hi)). A slab with no axis '
                    'is the whole box -- that is Scene(background=...).')
            raise NotImplementedError(
                f'Slab({", ".join(f"{a}=..." for a in named)}): a slab names exactly ONE axis '
                f'and runs the whole box across the other two. Two axes is a rectangular rod '
                f'and three is a box; both, with spheres, cylinders, polygons and rasters, '
                f'are stage 4 (shapes).')
        axis = named[0]
        span = getattr(self, axis)
        if not (isinstance(span, (tuple, list)) and len(span) == 2):
            raise ValueError(f'Slab({axis}={span!r}): a span is the pair (lo, hi).')
        lo = static_float('Slab', axis, span[0])
        hi = static_float('Slab', axis, span[1])
        if not lo < hi:
            raise ValueError(f'Slab({axis}=({lo}, {hi})): lo < hi. A length is a size, so a '
                             f'reversed or empty span is refused rather than read as the '
                             f'complement.')
        object.__setattr__(self, axis, (lo, hi))

    @property
    def axis(self):
        """The name of the one axis this slab bounds."""
        return next(a for a in AXES if getattr(self, a) is not None)

    @property
    def span(self):
        """``(lo, hi)`` along :attr:`axis`, as asked for (not snapped)."""
        return getattr(self, self.axis)

    def cells(self, grid):
        """``(k0, k1)``: the half-open range of cell indices along :attr:`axis`, snapped."""
        a = AXES.index(self.axis)
        lo, hi = self.span
        k0 = int(round((lo - grid.low[a]) / grid.dx))
        k1 = int(round((hi - grid.low[a]) / grid.dx))
        k0 = min(max(k0, 0), grid.n[a])
        k1 = min(max(k1, 0), grid.n[a])
        if k1 <= k0:
            snapped = (grid.low[a] + k0 * grid.dx, grid.low[a] + k1 * grid.dx)
            raise ValueError(
                f'Slab({self.axis}={self.span}) snaps to the empty span {snapped} on a grid of '
                f'dx = {grid.dx:g}: it is thinner than a cell, or it lies outside the box. A '
                f'layer that snapped to nothing would run as vacuum and look like a physics '
                f'result, so it is refused. Give at least one cell of thickness (dx <= '
                f'{abs(self.span[1] - self.span[0]):g}) or move it inside the box.')
        return k0, k1

    def snapped(self, grid):
        """``(lo, hi)`` of the cell boundaries the span snapped to."""
        a = AXES.index(self.axis)
        k0, k1 = self.cells(grid)
        return (grid.low[a] + k0 * grid.dx, grid.low[a] + k1 * grid.dx)


_slab_init = Slab.__init__


def _slab_init_with_stages(self, *args, **kwargs):
    later = [k for k in kwargs if k in _LATER_SHAPES]
    if later:
        k = later[0]
        raise NotImplementedError(
            f'Slab({k}=...) is {_LATER_SHAPES[k]}, which belongs to stage 4 (shapes). Stage 1 '
            f'has one shape, the layer: Slab(x=(lo, hi)), Slab(y=...) or Slab(z=...).')
    _slab_init(self, *args, **kwargs)


Slab.__init__ = _slab_init_with_stages


@spec(positional=('shape', 'material'))
class Structure:
    """A shape filled with a material: ``fd.Structure(fd.Slab(z=(-0.2, 0)), glass)``.

    Both may be given positionally, in that order, because there is no other order a reader
    would expect. ``material`` takes the scalar shorthand: ``fd.Structure(slab, 2.25)`` is
    ``fd.Material(eps=2.25)``.

    Structures are applied **in the order they are given**, so a later one overwrites an
    earlier one where they overlap -- the layer-stack rule, and the only overlap rule stage 1
    needs. ``check`` prints the snapped span of each.
    """

    shape: object
    material: object

    def __post_init__(self):
        from .materials import as_material
        if isinstance(self.shape, Box):
            raise NotImplementedError(
                'Structure(fd.Box(...), ...): in stage 2 a Box is only the region of an '
                'fd.TFSF source (and later of monitors); a box filled with a material is a '
                'shape, and shapes other than fd.Slab are stage 4.')
        if not isinstance(self.shape, Slab):
            raise TypeError(
                f'Structure(shape=...): stage 1 has one shape, fd.Slab(z=(lo, hi)); got '
                f'{type(self.shape).__name__}. Spheres, cylinders, boxes, polygons and '
                f'rasters are stage 4 (shapes).')
        object.__setattr__(self, 'material', as_material(self.material, 'Structure(material=)'))


@spec
class Box:
    """An axis-aligned box, ``fd.Box(size=(sx, sy, sz), center=(cx, cy, cz))``.

    Stage 2 introduces it as the region of an :class:`~wavebit.fd.TFSF` source, and nothing
    else: as a *structure* it is a stage-4 shape and ``Structure(fd.Box(...), ...)`` is
    refused naming that stage. Both triples are static lengths in the scene's coordinates
    (the box's faces pick node planes, so they fix the plan).
    """

    size: tuple
    center: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        for name in ('size', 'center'):
            value = getattr(self, name)
            if not (isinstance(value, (tuple, list)) and len(value) == 3):
                raise ValueError(f'Box({name}={value!r}): a triple (x, y, z).')
            object.__setattr__(self, name, tuple(static_float('Box', name, v) for v in value))
        if any(not s > 0 for s in self.size):
            raise ValueError(f'Box(size={self.size}): every size is a positive length.')

    @property
    def low(self):
        return tuple(c - s / 2 for c, s in zip(self.center, self.size))

    @property
    def high(self):
        return tuple(c + s / 2 for c, s in zip(self.center, self.size))
