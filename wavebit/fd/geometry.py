"""Where a material sits: a cell-snapped slab or box, and the structure that fills it.

Stage 1 has **one** shape, and it is the one a layer stack is made of (stage 3 adds the
cell-snapped :class:`Box`, the same rules on all three axes, and the monitor :class:`Plane`)::

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

__all__ = ['Slab', 'Structure', 'Box', 'Plane', 'AXES', 'cell_box', 'slab_axis', 'bounded_axes',
           'describe', 'span_rows']

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
                f'and runs the whole box across the other two. A rectangular rod or block is '
                f'fd.Structure(fd.Box(size=..., center=...), material) (stage 3, cell-snapped); '
                f'spheres, cylinders, polygons and rasters are stage 4 (shapes).')
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
            f'Slab({k}=...) is {_LATER_SHAPES[k]}, which belongs to stage 4 (shapes) -- except a '
            f'cell-snapped block, which is fd.Box(size=..., center=...) as of stage 3. A slab '
            f'is the layer: Slab(x=(lo, hi)), Slab(y=...) or Slab(z=...).')
    _slab_init(self, *args, **kwargs)


Slab.__init__ = _slab_init_with_stages


@spec(positional=('shape', 'material'))
class Structure:
    """A shape filled with a material: ``fd.Structure(fd.Slab(z=(-0.2, 0)), glass)``.

    Both may be given positionally, in that order, because there is no other order a reader
    would expect. ``material`` takes the scalar shorthand: ``fd.Structure(slab, 2.25)`` is
    ``fd.Material(eps=2.25)``.

    ``shape`` is an :class:`Slab` (stage 1) or, from stage 3, a :class:`Box`: a cell-snapped
    rectangular block, the minimum lateral patterning a diffraction-order check needs (a
    lamellar grating is a Box spanning one transverse axis, a crossed pillar grating a Box
    spanning neither). Every other shape is stage 4.

    Structures are applied **in the order they are given**, so a later one overwrites an
    earlier one where they overlap -- the layer-stack rule, and the only overlap rule stage 1
    needs. ``check`` prints the snapped span of each.
    """

    shape: object
    material: object

    def __post_init__(self):
        from .materials import as_material
        if not isinstance(self.shape, (Slab, Box)):
            raise TypeError(
                f'Structure(shape=...): the shapes are fd.Slab(z=(lo, hi)) and, from stage 3, '
                f'the cell-snapped fd.Box(size=..., center=...); got '
                f'{type(self.shape).__name__}. Spheres, cylinders, polygons and rasters, '
                f'subpixel averaging and differentiable edges are stage 4 (shapes).')
        object.__setattr__(self, 'material', as_material(self.material, 'Structure(material=)'))


@spec
class Box:
    """An axis-aligned box, ``fd.Box(size=(sx, sy, sz), center=(cx, cy, cz))``.

    Three uses, one object: the region of an :class:`~wavebit.fd.TFSF` source (stage 2), the
    closed surface of an ``fd.Flux`` or the volume of an ``fd.Field`` monitor (stage 3), and --
    from stage 3 -- a **structure**, ``fd.Structure(fd.Box(...), material)``. Both triples are
    static lengths in the scene's coordinates (the box's faces pick node planes or cells, so
    they fix the plan and its skeleton).

    **As a structure it is cell-snapped exactly like** :class:`Slab`, on every axis at once:
    each face lands on the nearest cell boundary,

        i0 = round((x_lo - x_low) / dx),   i1 = round((x_hi - x_low) / dx)

    clipped to ``[0, n]``, and the cells ``i0 .. i1 - 1`` (times the same on y and z) carry the
    material -- every component of them, with no subpixel averaging -- and **the edges are
    not differentiable**: a size or a centre is static, and a gradient flows through the
    material's values only. ``check`` prints the snapped span beside the one asked for on each
    axis. A face that does not fall on a cell boundary moves by up to half a cell, which is a
    first-order error in ``dx``; stage 4 brings exact fractional filling and differentiable
    edges, and until then a convergence study in ``dx`` is the only route to a right answer. A
    box that extends past the domain on an axis is clipped there, which is how a lamellar
    grating spells "runs the whole period" along its uniform axis.
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

    def cells(self, grid):
        """``((i0, i1), (j0, j1), (k0, k1))``: the snapped half-open cell range per axis."""
        out = []
        for a, axis in enumerate(AXES):
            lo, hi = self.low[a], self.high[a]
            k0 = min(max(int(round((lo - grid.low[a]) / grid.dx)), 0), grid.n[a])
            k1 = min(max(int(round((hi - grid.low[a]) / grid.dx)), 0), grid.n[a])
            if k1 <= k0:
                snapped = (grid.low[a] + k0 * grid.dx, grid.low[a] + k1 * grid.dx)
                raise ValueError(
                    f'Box(size={self.size}, center={self.center}) snaps to the empty span '
                    f'{snapped} along {axis} on a grid of dx = {grid.dx:g}: it is thinner than '
                    f'a cell there, or it lies outside the box. A structure that snapped to '
                    f'nothing would run as the background and look like a physics result, so '
                    f'it is refused. Give it at least one cell along {axis} (dx <= '
                    f'{hi - lo:g}) or move it inside the box.')
            out.append((k0, k1))
        return tuple(out)

    def snapped(self, grid):
        """``(low triple, high triple)`` of the cell boundaries the box snapped to."""
        cells = self.cells(grid)
        return (tuple(grid.low[a] + cells[a][0] * grid.dx for a in range(3)),
                tuple(grid.low[a] + cells[a][1] * grid.dx for a in range(3)))


@spec
class Plane:
    """A plane across one axis, for a monitor: ``fd.Plane(z=0.2)``.

    The axis named with a scalar is the plane's normal; it snaps to the nearest **node plane**
    (where the tangential E lives). The other two axes run the whole box unless given as a
    span, ``fd.Plane(z=0.2, x=(-0.3, 0.3), y=(-0.3, 0.3))``, whose ends snap to node planes too.
    Every number is static: it picks indices, so it fixes the plan.
    """

    x: object = None
    y: object = None
    z: object = None

    def __post_init__(self):
        scalars = [a for a in AXES if getattr(self, a) is not None
                   and not isinstance(getattr(self, a), (tuple, list))]
        if len(scalars) != 1:
            raise ValueError(
                f'Plane({", ".join(f"{a}={getattr(self, a)!r}" for a in AXES if getattr(self, a) is not None)}): '
                f'name the normal axis with ONE scalar position, as Plane(z=0.2); the other '
                f'axes may be given as (lo, hi) spans and otherwise run the whole box.')
        normal = scalars[0]
        object.__setattr__(self, normal, static_float('Plane', normal, getattr(self, normal)))
        for a in AXES:
            v = getattr(self, a)
            if a == normal or v is None:
                continue
            if len(v) != 2:
                raise ValueError(f'Plane({a}={v!r}): a transverse extent is the pair (lo, hi).')
            lo, hi = static_float('Plane', a, v[0]), static_float('Plane', a, v[1])
            if not lo < hi:
                raise ValueError(f'Plane({a}=({lo}, {hi})): lo < hi.')
            object.__setattr__(self, a, (lo, hi))

    @property
    def axis(self):
        """The index (0 = x, 1 = y, 2 = z) of the plane's normal."""
        return next(a for a, name in enumerate(AXES)
                    if getattr(self, name) is not None
                    and not isinstance(getattr(self, name), tuple))

    @property
    def position(self):
        return getattr(self, AXES[self.axis])

    def extent(self, a):
        """``(lo, hi)`` along transverse axis ``a``, or ``None`` for the whole box."""
        return getattr(self, AXES[a])


def slab_axis(shape):
    """The axis index (0 = x, 1 = y, 2 = z) a :class:`Slab` bounds, or ``None`` for a Box."""
    return AXES.index(shape.axis) if isinstance(shape, Slab) else None


def cell_box(shape, grid):
    """``((i0, i1), (j0, j1), (k0, k1))``: the cells a structure's shape occupies, per axis.

    A :class:`Slab` runs the whole box on its two other axes; a :class:`Box` is snapped on
    all three. One rule, so every check that asks "which cells" asks it here.
    """
    if isinstance(shape, Box):
        return shape.cells(grid)
    out = [(0, grid.n[0]), (0, grid.n[1]), (0, grid.n[2])]
    out[AXES.index(shape.axis)] = shape.cells(grid)
    return tuple(out)


def bounded_axes(shape, grid):
    """The axes along which ``shape`` varies the medium: a Slab's own; a Box's cut ones."""
    if isinstance(shape, Slab):
        return (AXES.index(shape.axis),)
    return tuple(a for a, (k0, k1) in enumerate(shape.cells(grid)) if (k0, k1) != (0, grid.n[a]))


def describe(shape):
    """A short spelling of ``shape`` for a message: ``z=(lo, hi)`` or ``Box(size, center)``."""
    if isinstance(shape, Slab):
        return f'{shape.axis}={shape.span}'
    return f'Box(size={shape.size}, center={shape.center})'


def span_rows(shape, grid):
    """The report's ``(label, lo, hi, snapped lo, snapped hi)`` rows for one structure.

    A Slab gives one row labelled by its axis, exactly as stage 1 printed it (the rows are in
    the plan and its skeleton, so a Slab scene's skeleton does not move); a Box gives one row
    per axis, labelled ``box.x``, ``box.y``, ``box.z``.
    """
    if isinstance(shape, Slab):
        return ((shape.axis,) + tuple(shape.span) + tuple(shape.snapped(grid)),)
    lo, hi = shape.snapped(grid)
    return tuple((f'box.{axis}', shape.low[a], shape.high[a], lo[a], hi[a])
                 for a, axis in enumerate(AXES))
