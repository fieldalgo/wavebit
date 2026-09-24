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
  (``coefficients.painters``), which paints the material over them;
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

**Two discretisations, and the shape is the one that says which.** From stage 4's C2 a shape
either *fills cells whole* (:class:`Slab`, :class:`Box`) or *fills them fractionally*
(:class:`Sphere`, and the bodies C4 adds). :meth:`Shape.body` is that answer -- the
``wavebit/fd/_shapes`` body this shape is smoothed from, or ``None``. It is a method and not a
``smoothing=`` keyword on :class:`Structure` on purpose: a rule that could be switched, or
that keyed itself on whether a radius happened to be traced, would make ``jax.grad`` and a
finite difference compute two different functions, and it is the route-naming knob
``combine=`` and ``fourier_type=`` were deleted from this repository for. A cell-snapped shape
answers ``None`` and every number it lowers to is unchanged to the bit.

C3's ``Structure(discretisation=)`` is not that keyword and does not undo this paragraph: it
chooses between two discretisations that converge at different orders rather than between two
spellings of one rule, and both of the objections above are answered in its own place -- it is
static, it is in the skeleton, and a traced leaf on a structure that asked for the stencil is
**refused** rather than quietly given the other rule. See :class:`Structure`.
"""
import math

from ._spec import spec, static_float

__all__ = ['Shape', 'Slab', 'Sphere', 'Structure', 'Box', 'Plane', 'AXES']

#: The three axes in the order the public API spells a triple.
AXES = ('x', 'y', 'z')

#: Shape keywords stage 4 will take. Refused here by name, with the stage, except where the
#: spelling now exists -- ``radius`` names :class:`Sphere` (C2) rather than a stage.
_LATER_SHAPES = {
    'radius': 'a sphere or a cylinder', 'center': 'a box or a sphere',
    'size': 'a box', 'vertices': 'a polygon', 'raster': 'a raster',
    'angle': 'a rotated shape', 'height': 'a cylinder',
}


class Shape:
    """What a structure's shape is asked, and the only place those questions are named.

    Nine methods, and the module docstring says which caller asks which. Eight of them have
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

        Only a cell-snapped shape has such a box, and it is the one
        ``coefficients.painters`` paints with a ``where``. A shape with a fractional edge has
        no answer in this form -- its answer is a fill fraction per component per cell -- and
        refuses, which is what makes the caller that must change when such a shape arrives a
        single, named one.
        """
        raise self._unanswered(
            'filled_cells', 'the cells it fills outright are unknown -- a shape whose edge is '
            'not a cell boundary fills cells fractionally and has no such box.')

    def body(self):
        """The ``wavebit/fd/_shapes`` body this shape is **smoothed** from, or ``None``.

        ``None`` means cell-snapped: the shape fills the cells :meth:`filled_cells` names
        whole, and :func:`wavebit.fd._lower.coefficients.build` paints its material over them
        with a ``where``, which is what every stage-0..3 golden holds to the bit.

        A body means fractional: the lowering asks it, at **each component's own Yee point**,
        what fraction of that voxel the shape fills and which way the surface faces there
        (``wavebit/fd/_lower/occupancy.py``), and mixes the two media by the
        Farjadpour/Kottke rule rather than choosing one of them.

        There is no default and no third answer. A shape that returns a body must refuse
        :meth:`filled_cells` -- the two are the same question asked of the two
        discretisations, and a shape that answered both would have said its edge is a cell
        boundary and also that it is not.
        """
        raise self._unanswered(
            'body', 'whether it fills cells whole or fractionally is unknown -- a shape is '
            'cell-snapped (body() is None) or smoothed (body() is a wavebit.fd._shapes body).')

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

    def body(self):
        """``None``: a layer is cell-snapped, and stage 4 deliberately leaves it that way.

        Every stage-1..3 layered oracle and every frozen golden rests on it, and a layer's
        interface is the one case where the user can put the boundary on a cell edge and mean
        it. Whether the layer becomes a smoothed body too is a stage-5 question, and it costs
        a deliberate re-freeze of all four archives. The smoothed rectangular body is C4's.
        """
        return None

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
            f'Slab({k}=...) is {_LATER_SHAPES[k]}. A slab is the layer: Slab(x=(lo, hi)), '
            f'Slab(y=...) or Slab(z=...). A ball is fd.Sphere(radius=..., center=...), '
            f'smoothed at each component\'s own Yee point; a cell-snapped block is '
            f'fd.Box(size=..., center=...); the cylinder, the prism and the smoothed, '
            f'rotatable rectangular body are stage 4 C4.')
    _slab_init(self, *args, **kwargs)


Slab.__init__ = _slab_init_with_stages


@spec(positional=('shape', 'material'))
class Structure:
    """A shape filled with a material: ``fd.Structure(fd.Slab(z=(-0.2, 0)), glass)``.

    Both may be given positionally, in that order, because there is no other order a reader
    would expect. ``material`` takes the scalar shorthand: ``fd.Structure(slab, 2.25)`` is
    ``fd.Material(eps=2.25)``.

    ``shape`` is an :class:`Slab` (stage 1), a :class:`Box` (stage 3): a cell-snapped
    rectangular block, the minimum lateral patterning a diffraction-order check needs (a
    lamellar grating is a Box spanning one transverse axis, a crossed pillar grating a Box
    spanning neither), or -- from stage 4's C2 -- a :class:`Sphere`, the first shape whose
    edge is **not** a cell boundary. The cylinder, the prism, the rotated body and CSG of them
    are C4.

    The door is ``isinstance(shape, fd.Shape)``: external subclasses may answer the
    protocol too. ``body()`` chooses the fractional or cell-snapped lowering. A bare
    ``fd.Shape()`` is the protocol rather than a shape and is refused here. A :class:`Plane` is a monitor's region and not a shape, so this
    same door refuses it.

    Structures are applied **in the order they are given**, so a later one overwrites an
    earlier one where they overlap -- the layer-stack rule. ``check`` prints the span of each.

    **How far that carries once a shape has a fractional edge.** The mixing rule's second
    medium is *whatever the cell already holds*, background or an earlier structure, which is
    two phases and one interface. So painting order still means exactly what it says wherever
    the arriving shape fills the voxel **whole** -- a smoothed ball over a cell-snapped
    :class:`Slab` substrate is right per component to 2.0e-16 against the hand-written
    two-phase mix, in either order, and so is a core inside a shell -- and it does **not**
    compose where two *smoothed* surfaces pass through one voxel: the second mix would treat
    the first mixture as a medium with a second interface across it, measured 73% high on two
    half spaces of the same material and up to 122% on one ``fd.Sphere`` declared twice. That
    case is refused by name (``wavebit/fd/_lower/checks.py::check_smoothed``, door 5), which
    is what keeps the sentence above true for every scene this module accepts.

    **``discretisation=``: which material discretisation this structure gets.** ``'voxel'``,
    the default and every scene's behaviour until stage 4's C3, is *not* the name of a rule --
    it is "whatever this shape's :meth:`Shape.body` already says", a whole-cell replace for a
    cell-snapped shape and the diagonal Farjadpour/Kottke mix plus the constitutive cross terms
    for a smoothed one. What the two have in common, and what the word names, is that every
    coefficient is a function of one component's **own voxel**. ``'stencil'`` asks instead for
    the symmetric extended-stencil operator ``docs/fd-symmetric-stencil.md`` derives: the exact
    edge and dual-face interface constraints, solved once on the host, whose rows couple each
    edge to every edge within two cells. It is opt-in, it is prepared at ``fd.plan`` time and
    never inside a trace, and ``fd.check`` prints it.

    **``memory_bound=``** (bytes, only beside ``discretisation='stencil'``) raises the host
    memory a preparation may be predicted to take before it is refused. The default admits the
    research sphere at its three placements at 40 and 56 cells per wavelength with a stated
    margin (``docs/fd-symmetric-stencil.md``); a larger value is an explicit decision to trust
    the prediction past the sizes it was measured at.

    **Why here, and not the ``smoothing=`` the module docstring refuses.** That paragraph turns
    down a keyword that would *switch a rule* for the same shape, or key itself on whether a
    radius happened to be traced, because either makes ``jax.grad`` and a finite difference
    compute two different functions. Both objections are kept rather than dodged. This value is
    static, a typo is refused outright, it is in the plan and in the skeleton, and it is never
    read from whether anything is a tracer. A structure that asks for the stencil takes plain
    numbers only -- Python ``int``/``float`` and NumPy scalars -- and any array or tensor, traced
    or not, is **refused by name** (``StencilNumberRefused``) rather than quietly given the
    other rule, which is exactly the failure ``combine=`` and ``fourier_type=`` were deleted
    for. The door is a type and not a question about tracing because a Torch forward-mode dual
    reads as a number and passed a tracing test.

    It is on the structure and not on the :class:`Shape` or on :class:`~wavebit.fd.Run`
    because that is where its inputs are. The operator is assembled from the body's geometry
    **and** from the material contrast -- ``eps`` enters the fit's right-hand side -- so a
    shape cannot answer it: one ``fd.Sphere`` object may be shared by two structures of
    different materials and would then have to name two operators. And ``Run`` cannot, because
    a scene may hold one structure the operator can be prepared for and one it cannot; a
    run-wide switch would have to exempt the second silently, which is the fallback this
    commit refuses to write.
    """

    shape: object
    material: object
    discretisation: str = 'voxel'
    memory_bound: object = None

    def __post_init__(self):
        import math
        from numbers import Real

        from .materials import as_material
        from ._lower.stencil import DEFAULT, STENCIL
        if self.discretisation not in (DEFAULT, STENCIL):
            raise ValueError(
                f'Structure(discretisation={self.discretisation!r}): the two spellings are '
                f'{DEFAULT!r} -- the rule this structure\'s shape already names, which is what '
                f'every scene has had -- and {STENCIL!r}, the symmetric extended-stencil '
                f'operator of docs/fd-symmetric-stencil.md. A value nobody reads is silently '
                f'ignored, so a third one is refused here rather than falling back to the '
                f'default: the two converge at different orders and a scene that got the one '
                f'it did not ask for would read as a physics result.')
        if self.memory_bound is not None:
            if self.discretisation != STENCIL:
                raise ValueError(
                    f'Structure(memory_bound={self.memory_bound!r}) bounds the host memory of '
                    f'the extended stencil\'s preparation, and this structure asks for '
                    f'discretisation={self.discretisation!r}, which prepares nothing. A value '
                    f'nobody reads is refused rather than ignored.')
            bound = self.memory_bound
            if isinstance(bound, bool) or not isinstance(bound, Real) \
                    or not math.isfinite(bound) or bound <= 0:
                raise ValueError(
                    f'Structure(memory_bound={bound!r}): a positive, finite number of bytes, '
                    f'e.g. memory_bound=2 * 1024 ** 3 for 2 GiB.')
        if type(self.shape) is Shape:
            raise TypeError(
                'Structure(shape=fd.Shape()): fd.Shape is the protocol a shape answers, not a '
                'shape -- it has no extent, and every question below it would refuse. Use '
                'fd.Slab(z=(lo, hi)) or fd.Box(size=..., center=...), or subclass fd.Shape '
                'and answer its methods.')
        if not isinstance(self.shape, Shape):
            raise TypeError(
                f'Structure(shape=...): the shapes are fd.Slab(z=(lo, hi)), the cell-snapped '
                f'fd.Box(size=..., center=...) and the smoothed fd.Sphere(radius=..., '
                f'center=...); got {type(self.shape).__name__}. External shapes get in by '
                f'subclassing fd.Shape and answering its protocol. Cylinders, prisms, rotated '
                f'bodies, polygons and CSG of them are stage 4 C4; a raster shape is refused '
                f'outright (its interface normal cannot come from the shape).')
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

    def body(self):
        """``None``: this box is the cell-snapped one, and it is also a monitor's region.

        ``fd.Box`` is three things at once -- a ``TFSF`` region, a monitor's surface or
        volume, and a structure -- and a region has to snap to node planes to pick indices.
        One name cannot carry two discretisations, so the smoothed, rotatable, differentiable
        rectangular **body** is a separate class in C4 and this one does not move. Every
        stage-3 golden holds to the bit because of that.
        """
        return None

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
class Sphere(Shape):
    """A ball, ``fd.Sphere(radius=0.2, center=(0, 0, 0))`` -- the first **smoothed** shape.

    Unlike :class:`Slab` and :class:`Box` it does not snap to anything. The lowering asks it,
    at each of the six components' **own** Yee points, what fraction ``f`` of that component's
    voxel the ball fills and how the surface is oriented there (``w_c``, the mean of ``n_c^2``
    over the surface inside the voxel), and mixes the ball's material with whatever the cell
    already held by the Farjadpour/Kottke rule -- see
    ``wavebit/fd/_lower/coefficients.py::mix`` for the formula and its citations, and
    ``wavebit/fd/_lower/occupancy.py`` for the stage that produces the pair. Its exact clipped
    volume changes continuously with the requested position. Electromagnetic convergence
    is a separate measurement, made by the Mie gate.

    That accuracy gate is incomplete. Planning emits ``fd.SubpixelAccuracyWarning`` with
    the controlled glass-sphere benchmark (order 2.023). The kernel includes adjoint
    cross coupling between field directions and stores D/B in compact regions; validation
    across contrasts remains separate. The benchmark is not this sphere's error bound;
    refine and validate the observables required by the application.

    The coupled memory adds preparation-time doors: its interpolation support must be
    lossless and clear of absorbing layers; cross terms at a Symmetry plane need a
    mirror-weighted operator not yet provided. Initial E (and initial H when magnetic)
    needs an inverse constitutive solve not yet provided; use a source instead. The
    positive-energy and time-step certificates are checked before a run starts.

    ``radius`` and ``center`` are **static** in C2. They fix :meth:`support`, which fixes the
    coefficient array shapes and so the plan's skeleton, and the ``range=`` that declares a
    static support box for a traced value -- the ``fd.Dipole(range=)`` spelling -- is C6's. The
    lowering chain from the ball's geometry to ``be`` is already differentiable and is gated
    that way (``tests/test_fd_occupancy.py::test_a_traced_radius_reaches_the_coefficient``);
    what C6 adds is the public door, not the derivative.

    **What a sphere may not meet, in C2: five doors.** All five are
    ``wavebit/fd/_lower/checks.py::check_smoothed``, in the order ``wavebit/fd/__init__.py``
    lists them, and each names the commit that owns the case it turns away.

    1. **No conductivity and no poles where it reaches.** Its own material must not be an
       ``fd.Conductor`` or an ``fd.Dispersive``, and neither may the background or any
       structure whose cells it shares: ``sigma`` and the pole coefficients are *not* smoothed
       here and keep the cell-snapped rule they have today, which a fractional edge has no
       answer for. (C8 measures the rule before writing it.)
    2. **No wrapping face.** A sphere that reaches a ``Periodic`` or ``Bloch`` face is refused
       because resolving its wrapped part means summing the geometry over the image ring, and
       the cell's origin -- which the ring needs and a body does not carry -- is C4's to
       thread. An ``fd.Axis1D`` axis is refused too, with its own sentence and for a different
       reason: that axis stands for one the field is *invariant* along, and a ball is not.
    3. **No reaching a face of the box.** A cell-snapped ``fd.Box`` is clipped exactly; a ball
       is not, and the six components disagree about where the box ends -- three of them count
       material half a cell outside a ``-`` face, and the same three lose the last half cell at
       a ``+`` face. The two arms carry two different messages.
    4. **No anisotropic eps or mu.** The effective tensor is the Farjadpour/Kottke rule for two
       *isotropic* phases; blending each component with its own ``eps`` is not the tau-transform
       the anisotropic construction calls for, and on a ball the normal is almost never along a
       principal axis. The door covers the same three media as door 1.
    5. **No second smoothed surface in one voxel.** A painter mixes into what the cell already
       holds, which is two phases and *one* surface; two smoothed surfaces in one voxel do not
       compose (73% high on two half spaces, 122% on one ``fd.Sphere`` declared twice). A ball
       over a cell-snapped shape, or inside another ball it shares no surface voxel with, is
       fine and is not refused.
    """

    radius: float
    center: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        radius = static_float('Sphere', 'radius', self.radius)
        if not math.isfinite(radius) or not radius > 0:
            raise ValueError(
                f'Sphere(radius={radius}): a radius is a finite positive length. A length is a size '
                f'here, so a negative one is refused rather than read as subtracting the '
                f'shape.')
        object.__setattr__(self, 'radius', radius)
        center = self.center
        if not (isinstance(center, (tuple, list)) and len(center) == 3):
            raise ValueError(f'Sphere(center={center!r}): a triple (x, y, z), absolute, in the '
                             f'scene\'s coordinates -- not the cell-centre origin the rcwa '
                             f'path spells as cx=/cy=.')
        center = tuple(static_float('Sphere', 'center', v) for v in center)
        if not all(math.isfinite(v) for v in center):
            raise ValueError(f'Sphere(center={center!r}): every coordinate must be finite.')
        object.__setattr__(self, 'center', center)

    @property
    def low(self):
        return tuple(c - self.radius for c in self.center)

    @property
    def high(self):
        return tuple(c + self.radius for c in self.center)

    # -- the protocol (see :class:`Shape`) ----------------------------------------------------

    def body(self):
        """The ``_shapes`` ball: an exact clipped-ball fill, and the tangent plane's moment."""
        from ._shapes import Sphere as _Ball
        return _Ball(radius=self.radius, center=self.center)

    def support(self, grid):
        """Every cell whose voxels the ball can reach, **with one cell of margin** per side.

        A cell's six component voxels are not the cell: component ``c`` of ``E`` sits half a
        cell along axis ``c``, so the union of a cell's voxels spans
        ``[low + (i - 1/2) dx, low + (i + 1) dx]`` along each axis and reaches a whole ``dx``
        past the node. The margin is one cell rather than the half this needs, because the
        cost is a shell of voxels whose fill is exactly zero and the alternative is an
        off-by-one that removes material from a real surface. It is gated on the volume:
        ``tests/test_fd_occupancy.py::test_the_support_box_holds_the_whole_ball`` sums the
        lowered fill and compares it with ``4 pi R^3 / 3``, and on that gate's own scene
        (``R = 0.25`` at ``dx = 0.05``, five cells per radius, centre
        ``(0.013, -0.007, 0.021)``) the mutant that drops the margin loses **3.79e-3** of the
        volume against 1.9e-15 with it.

        **The loss is placement dependent, and a ball centred on a node plane is the WORST
        placement there is.** What the margin protects is the three components whose Yee offset
        along an axis is zero: their voxels are centred on the *nodes*, so without the margin
        they cover only ``[k0 - 1/2, k1 - 1/2]`` cells of a box that runs ``k0 .. k1``, and the
        ball's top surface falls out of that as soon as the centre sits more than half a cell
        above a node. Measured, ``R = 0.25`` in a PEC box at ``dx = R/k``, the centre on all
        three axes at once and the cell count **even** so that the box centre really is a node
        plane -- swept through one whole cell in 41 steps at ``k`` = 5, 8, 10 and 20, and at the
        node, the cell centre and one generic offset at ``k`` = 16:

            k (cells per radius)          5        8       10       16       20
            worst, margin dropped   1.45e-2  5.74e-3  3.69e-3  1.45e-3  9.30e-4
            worst, margin kept      2.1e-15  1.7e-15  3.2e-15  1.5e-15  6.6e-15

        and at every resolution the worst placement is the node-centred one. A ball centred at
        a **cell centre** loses nothing at all -- 2.1e-16 at five cells per radius and the same
        at every other -- and between the two the loss is flat at round-off while the centre's
        fractional cell position is at or below one half and rises monotonically above it:
        1.50e-4 at 0.55 of a cell, 5.96e-4 at 0.60, 3.69e-3 at 0.75, 1.18e-2 at 0.95 and
        1.45e-2 at the node, at five cells per radius. The gate's own scene is an off-node
        centre whose worst axis sits at 0.86 of a cell, and it loses **3.79e-3**.

        The grid condition is one clause, and it is what made an earlier measurement of this
        irreproducible: ``Grid.from_domain`` puts the low corner at ``centre - n dx / 2``, so
        the box centre is a node plane only when the cell count ``n`` is **even**. Measured at
        an odd ``n``, a ball at the origin is a cell-centred ball and loses nothing, which is
        the opposite scene.

        The loss falls with resolution and never reaches zero, which is why the margin is not
        traded for the half cell the arithmetic above actually needs.
        """
        import math
        out = []
        for a in range(3):
            k0 = math.floor((self.low[a] - grid.low[a]) / grid.dx) - 1
            k1 = math.ceil((self.high[a] - grid.low[a]) / grid.dx) + 1
            k0 = min(max(int(k0), 0), grid.n[a])
            k1 = min(max(int(k1), 0), grid.n[a])
            out.append((k0, k1))
        return tuple(out)

    def filled_cells(self, grid):
        """Refused: a ball's edge is not a cell boundary, so no cell range is "the ball"."""
        return super().filled_cells(grid)

    def uniform_along(self, axis, cells, grid):
        """``True`` only where the ball cannot reach the layers at all.

        A sphere's surface curves, so every range of cell layers its support touches holds a
        piece of that surface. This is the answer ``checks.check_absorbing_layers`` needs and
        a bounding-box test does not give: a ball whose *box* spans an absorbing layer would
        be waved through it, and that case is measured at 1.9e-1 reflection against 6.0e-4 --
        317x (``tests/test_fd_absorber.py``).
        """
        k0, k1 = self.support(grid)[axis]
        return k1 <= cells[0] or k0 >= cells[1]

    def bounded_axes(self, grid):
        """All three, always -- and not the axes its *box* happens to cut.

        :class:`Box` may leave an axis out because a box clipped to the domain really is
        uniform along it, so a coefficient array of extent 1 there is exact. A ball is never
        uniform along an axis it reaches, whatever its support clipped to, so an extent-1 axis
        would broadcast one plane of fills over the whole grid. The mutant that copies Box's
        rule here is ``tests/test_fd_occupancy.py::test_a_ball_varies_along_every_axis``.
        """
        return (0, 1, 2)

    def describe(self):
        return f'Sphere(radius={self.radius}, center={self.center})'

    def span_rows(self, grid):
        """One row per axis, labelled ``sphere.x/y/z``, asked and got the **same**.

        That is the report telling the truth rather than a missing feature: a smoothed shape
        does not snap, so ``check`` prints no ``<- snapped`` note for one, and the difference
        from the ``Slab``/``Box`` rows above it is the whole point of the shape.
        """
        return tuple((f'sphere.{axis}', self.low[a], self.high[a], self.low[a], self.high[a])
                     for a, axis in enumerate(AXES))

    def min_extent(self, grid):
        """Its diameter."""
        return 2 * self.radius


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


