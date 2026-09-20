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

**What a shape is asked, and by whom.** Until this commit there was one function,
``cell_box(shape, grid)``, and nine call sites read it. They were not asking one question.
Sorted by what they do with the answer, they were asking three:

* *which cells does this material fill?* -- the lowering itself
  (``coefficients.owner_map``), which paints an owner index over them;
* *which cells may this shape touch?* -- every containment, clearance and overlap test
  (``check_overlaps``, ``_proximity_notes``, ``_check_scatterer``, ``FarField``'s
  containment, the pole term's region, ``grazing._layer_media``, ``medium_at``);
* *is this shape uniform along an axis over these cell layers?* -- i.e. does it put an
  interface in there (``checks.check_absorbing_layers``, and the "fills the whole cell
  layer" halves of ``medium_at`` and ``_layer_media``).

On a cell-snapped :class:`Slab` or :class:`Box` all three have the same answer, which is why
one function served them; on the first shape whose edge is not a cell boundary they differ,
and a caller reading the wrong one is silent. A sphere whose *bounding box* spans an
absorbing layer would be waved through the refusal in ``check_absorbing_layers``, and that
case is measured at 1.9e-1 reflection against 6.0e-4 -- 317x (``tests/test_fd_absorber.py``).
So :class:`Shape` names the three questions separately, plus the four dispatches that used to
be written ``isinstance(shape, Slab) ... else assume it is a Box``: :meth:`Shape.layer_axis`,
:meth:`Shape.bounded_axes`, :meth:`Shape.describe`, :meth:`Shape.span_rows`. Every one of
them is answered by the shape and none of them has an "else" that guesses.

``cell_box`` is deleted rather than kept beside the protocol: nothing in ``wavebit.fd`` is
released, so a retired name goes outright (see ``docs/`` and the ``fourier_type``/``combine=``
precedents) -- a second spelling of "which cells" is exactly the ambiguity this replaces.
"""
from ._spec import spec, static_float

__all__ = ['Shape', 'Slab', 'Structure', 'Box', 'Plane', 'AXES']

#: The three axes in the order the public API spells a triple.
AXES = ('x', 'y', 'z')

#: Shape keywords stage 4 will take. Refused here by name, with the stage.
_LATER_SHAPES = {
    'radius': 'a sphere or a cylinder', 'center': 'a box or a sphere',
    'size': 'a box', 'vertices': 'a polygon', 'raster': 'a raster',
    'angle': 'a rotated shape', 'height': 'a cylinder',
}


class Shape:
    """What a structure's shape is asked, and the only place those questions are named.

    Eight methods, and the module docstring says which caller asks which. Seven of them have
    no default: a shape that does not answer one is refused **by name**, at the call site, and
    never silently treated as something else. That is the whole point of the class. The one
    default is :meth:`layer_axis`, whose honest answer for anything that is not a layer is
    ``None``, and whose ``None`` makes every reader take the conservative branch.

    ``support``, ``filled_cells``, ``uniform_along``, ``bounded_axes``, ``min_extent`` and
    ``span_rows`` are **static**: they decide cell ranges, array shapes and report rows, so
    they are part of the plan and its skeleton and cannot depend on a traced value.
    """

    def _unanswered(self, method, question):
        return NotImplementedError(
            f'{type(self).__name__}.{method}() is not implemented: {question} '
            f'fd.Shape names one question per method so that no caller has to guess from '
            f'the type, and a shape that does not answer one is refused here rather than '
            f'read as a Slab or a Box.')

    # -- the three questions ``cell_box`` used to answer at once ------------------------------

    def support(self, grid):
        """``((i0, i1), (j0, j1), (k0, k1))``: every cell this shape **may** touch.

        The static integer bounding box, half-open per axis, clipped to the grid. It is an
        over-estimate for a shape with a fractional edge and exact for a cell-snapped one.
        Read by every containment, clearance and overlap question -- the ones whose right
        answer is "no" only when the shape cannot reach at all.
        """
        raise self._unanswered('support', 'which cells it may touch is unknown.')

    def filled_cells(self, grid):
        """``((i0, i1), ...)``: the cells this shape's material fills **wholly**.

        Only a cell-snapped shape has such a box, and it is the one ``coefficients.owner_map``
        paints. A shape with a fractional edge has no answer in this form -- its answer is a
        fill fraction per component per cell -- and refuses, which is what makes the caller
        that must change when such a shape arrives a single, named one.
        """
        raise self._unanswered(
            'filled_cells', 'the cells it fills outright are unknown -- a shape whose edge is '
            'not a cell boundary fills cells fractionally and has no such box.')

    def uniform_along(self, axis, cells, grid):
        """Whether this shape puts **no interface** inside cell layers ``range(*cells)``.

        ``axis`` is 0, 1 or 2 and ``cells`` is the half-open ``(lo, hi)`` layer range along it.
        True means the medium this shape makes does not change anywhere across those layers:
        it either runs straight through them or is uniform along that axis there.

        This, and not a bounding box, is what ``checks.check_absorbing_layers`` asks -- a
        structure that *ends* inside an absorbing layer is a mirror the layer cannot undo, and
        a bounding box that spans the layer says nothing about whether the shape does.
        """
        raise self._unanswered(
            'uniform_along', 'whether it puts an interface inside a range of cell layers is '
            'unknown.')

    # -- the dispatches that used to be ``isinstance(shape, Slab) ... else assume a Box`` -----

    def layer_axis(self):
        """The axis index (0, 1, 2) this shape is a **layer** across, or ``None``.

        A layer runs the whole box on its other two axes, so it is part of the one-dimensional
        background a ``TFSF`` column is built from and it is what ``sources`` selects on.
        ``None`` -- the default, and the answer for everything that is not a :class:`Slab` --
        means "not a layer", and every reader's ``None`` branch is the conservative one: the
        shape is a scatterer, kept out of the column and never waved past a face.

        **A non-``None`` answer needs nothing else.** Its cell span along that axis is
        ``support(grid)[layer_axis()]``, and whether it fills one cell layer ``j`` outright is
        ``uniform_along(layer_axis(), (j, j + 1), grid)`` -- both already on this protocol, and
        both the question the caller means rather than a :class:`Slab` attribute. Until this
        was written the three readers in ``_lower/sources.py`` reached for ``shape.axis`` and
        ``shape.cells(grid)`` after branching on this method, so a shape that answered the
        whole protocol and declared itself a layer died there with a bare ``AttributeError``
        -- the guess-from-the-type failure this class exists to replace, one method later.
        ``tests/test_fd_shape_protocol.py`` walks a protocol-only layer through all three.
        """
        return None

    def bounded_axes(self, grid):
        """The axes along which this shape varies the medium, as a tuple of indices.

        ``coefficients.varying_axes`` unions these to decide which axes a coefficient array
        is broadcast along, so this fixes array shapes and therefore the skeleton. It is not
        derived from :meth:`uniform_along`: a :class:`Slab` names its own axis even where its
        span happens to cover the whole box, and that difference is a frozen array shape.
        """
        raise self._unanswered('bounded_axes', 'which axes it varies the medium along is '
                                               'unknown.')

    def describe(self):
        """A short spelling of this shape for a refusal or a warning, e.g. ``z=(-0.2, 0.0)``.

        Every refusal that names a structure prints this. It has no default because a default
        would print one shape's spelling for another's -- the exact defect this protocol
        replaces, where a sphere would have been described as ``Box(size=..., center=...)``.
        """
        raise self._unanswered('describe', 'it has no spelling for a message.')

    def span_rows(self, grid):
        """The report's ``(label, lo, hi, snapped lo, snapped hi)`` rows for this shape.

        They go into ``Plan.spans`` and so into the skeleton and every frozen archive, which
        is why the labels are part of the shape's contract and not the reporter's guess.
        """
        raise self._unanswered('span_rows', 'it has no report rows.')

    # -- what the grid has to resolve ---------------------------------------------------------

    def min_extent(self, grid):
        """The shortest side this shape **bounds** in ``grid``, or ``None`` if it bounds none.

        The smallest feature the grid has to resolve. Only the axes :meth:`bounded_axes` names
        are measured, and each is measured as it reaches into the box: a shape that runs out
        through a face has no feature size along that axis, and a refusal quoting one would be
        about a side that is not in the grid. ``fd.Box(size=(0.3, 0.3, 1.0))`` in a box 0.2
        wide across x and y bounds z alone, and its answer is 1.0, not 0.3.

        That is why it takes ``grid``: "bounded" is a question about the grid, so a signature
        without one cannot answer the sentence above -- it can only return the shortest side
        as spelled, clipped sides and all.

        Stage 4's "too few cells across" refusal (C5) is the reader; it is on the protocol
        from the start so that every shape answers it in the same units as its own lengths,
        rather than the refusal reconstructing it per type.
        """
        raise self._unanswered('min_extent', 'its smallest bounded side is unknown.')


@spec
class Slab(Shape):
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

    # -- the protocol (see :class:`Shape`) ----------------------------------------------------

    def support(self, grid):
        """Its snapped cells on :attr:`axis`, the whole box on the other two."""
        out = [(0, grid.n[0]), (0, grid.n[1]), (0, grid.n[2])]
        out[AXES.index(self.axis)] = self.cells(grid)
        return tuple(out)

    def filled_cells(self, grid):
        """The same box as :meth:`support`: a slab is cell-snapped, so it fills it outright."""
        return self.support(grid)

    def uniform_along(self, axis, cells, grid):
        """True across its own axis only where its snapped cells cover ``cells`` entirely.

        On the other two axes it is uniform everywhere -- it runs the whole box there, so no
        range of layers along them holds a face of it.
        """
        if axis != AXES.index(self.axis):
            return True
        k0, k1 = self.cells(grid)
        return k0 <= cells[0] and k1 >= cells[1]

    def layer_axis(self):
        return AXES.index(self.axis)

    def bounded_axes(self, grid):
        """Its own axis, always -- see :meth:`Shape.bounded_axes` for why not the derived set."""
        return (AXES.index(self.axis),)

    def describe(self):
        return f'{self.axis}={self.span}'

    def span_rows(self, grid):
        """One row, labelled by its axis, exactly as stage 1 printed it."""
        return ((self.axis,) + tuple(self.span) + tuple(self.snapped(grid)),)

    def min_extent(self, grid):
        """Its thickness inside the box: the one side it bounds, so the only one to resolve."""
        a = AXES.index(self.axis)
        lo, hi = self.span
        return min(hi, grid.high[a]) - max(lo, grid.low[a])


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
    spanning neither). The shapes the package will *ship* next -- spheres, cylinders, prisms,
    rotated boxes and their CSG, each subpixel-smoothed and differentiable -- are stage 4.

    **The door is ``isinstance(shape, fd.Shape)``, and that is deliberately a seam.**
    :class:`Slab` and :class:`Box` are the only two :class:`Shape` in the package, but a shape
    written outside it that answers the protocol plans, lowers and runs with no other type
    check anywhere downstream -- that is what makes stage 4's geometry kernel a new file
    rather than an edit to nine call sites, and it is why :class:`Shape` is exported as
    ``fd.Shape`` instead of kept private. What such a shape gets **today** is the stage-3
    lowering: ``coefficients.owner_map`` paints the cells :meth:`Shape.filled_cells` returns,
    with no fill fraction and no smoothing, so a curved one is a staircase at whatever box it
    calls filled. A fractional edge is meant to refuse ``filled_cells`` by name until C2
    replaces that caller. ``fd.Shape()`` itself is refused here: it is the protocol, not a
    shape. A :class:`Plane` is a monitor's region and not a shape, so this same door refuses
    it too.

    Structures are applied **in the order they are given**, so a later one overwrites an
    earlier one where they overlap -- the layer-stack rule, and the only overlap rule stage 1
    needs. ``check`` prints the snapped span of each.
    """

    shape: object
    material: object

    def __post_init__(self):
        from .materials import as_material
        if type(self.shape) is Shape:
            raise TypeError(
                'Structure(shape=fd.Shape()): fd.Shape is the protocol a shape answers, not a '
                'shape -- it has no extent, and every question below it would refuse. Use '
                'fd.Slab(z=(lo, hi)) or fd.Box(size=..., center=...), or subclass fd.Shape '
                'and answer its methods.')
        if not isinstance(self.shape, Shape):
            raise TypeError(
                f'Structure(shape=...): the shapes are fd.Slab(z=(lo, hi)) and, from stage 3, '
                f'the cell-snapped fd.Box(size=..., center=...); got '
                f'{type(self.shape).__name__}. Anything else gets in by subclassing fd.Shape '
                f'and answering its protocol -- this is the only type check, and nothing '
                f'downstream adds another. The shapes the package itself will ship (spheres, '
                f'cylinders, polygons and rasters), their subpixel averaging and their '
                f'differentiable edges are stage 4 (shapes).')
        object.__setattr__(self, 'material', as_material(self.material, 'Structure(material=)'))


@spec
class Box(Shape):
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

    # -- the protocol (see :class:`Shape`) ----------------------------------------------------

    def support(self, grid):
        """Its snapped cells on all three axes."""
        return self.cells(grid)

    def filled_cells(self, grid):
        """The same box as :meth:`support`: a box is cell-snapped, so it fills it outright."""
        return self.cells(grid)

    def uniform_along(self, axis, cells, grid):
        """True where its snapped cells on ``axis`` cover ``cells`` entirely.

        A box clipped to the domain on an axis covers every range along it, which is how a
        lamellar grating's uniform axis reaches through an absorbing face.
        """
        k0, k1 = self.cells(grid)[axis]
        return k0 <= cells[0] and k1 >= cells[1]

    def bounded_axes(self, grid):
        """The axes it does not span the whole box along -- the ones it actually cuts."""
        return tuple(a for a, (k0, k1) in enumerate(self.cells(grid))
                     if (k0, k1) != (0, grid.n[a]))

    def describe(self):
        return f'Box(size={self.size}, center={self.center})'

    def span_rows(self, grid):
        """One row per axis, labelled ``box.x``, ``box.y``, ``box.z``."""
        lo, hi = self.snapped(grid)
        return tuple((f'box.{axis}', self.low[a], self.high[a], lo[a], hi[a])
                     for a, axis in enumerate(AXES))

    def min_extent(self, grid):
        """Its shortest side over the axes it bounds, as each reaches into the box.

        A clipped axis is not one of them -- a lamellar tooth's "runs the whole period" side
        is the box's width, not a feature -- so a box clipped on every axis answers ``None``.
        """
        sides = [min(self.high[a], grid.high[a]) - max(self.low[a], grid.low[a])
                 for a in self.bounded_axes(grid)]
        return min(sides) if sides else None


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


