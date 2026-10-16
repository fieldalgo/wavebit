"""Where a material sits: a cell-snapped slab or box, and the structure that fills it.

Stage 1 has **one** shape, and it is the one a layer stack is made of (stage 3 adds the
cell-snapped :class:`Region`, the same rules on all three axes, and the monitor :class:`Plane`)::

    fd.Slab(z=(-0.2, 0.0))                 # one axis named; x and y run the whole box
    fd.Structure(fd.Slab(z=(-0.2, 0.0)), glass)

A ``Slab`` names exactly one axis. Two axes would be a box, an ``x``/``y`` extent a
rectangle, and every other shape is one of stage 4's smoothed bodies -- :class:`Sphere`,
:class:`Box`, :class:`Cylinder`, :class:`Prism` -- or refused (a raster). Each keyword is
refused by name rather than by the generic keyword message, so the reader learns the spelling
that exists.

**Slabs are cell-snapped, not subpixel.** The interface lands on the nearest *cell boundary*
of the named axis, and everything in the cells between the two snapped boundaries is the
structure's material -- every component of them, including the ones whose Yee point sits
exactly on a boundary. That is one rule for all six components and it is exactly what makes
it cheap: the material is a range of cell indices, so the coefficient arrays are built by
assignment and nothing is averaged.

It is also, deliberately, the *inaccurate* choice, and the report says so: an interface that
does not fall on a cell boundary is moved by up to half a cell, which is a first-order error
in ``dx`` where the rest of the scheme is second order. For a cell-snapped structure whose
thickness is not a whole number of cells, **a convergence study in ``dx`` is the route to a
right answer** -- or, for a lossless non-dispersive structure, the smoothed spelling of the
same geometry (stage 4: the unturned ``fd.Box`` puts a layer's faces where they were asked, gate
G5; conductive and dispersive bodies outside C8's bounded flat-film scopes remain
refused smoothed) -- and ``check`` prints the
snapped span beside the one that was asked for so that the size of the move is never a
surprise.

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

On a cell-snapped :class:`Slab` or :class:`Region` all three have the same answer, which is why
one function served them; on the first shape whose edge is not a cell boundary they differ,
and a caller reading the wrong one is silent. A sphere whose *bounding box* spans an
absorbing layer would be waved through the refusal in ``check_absorbing_layers``, and that
case is measured at 1.9e-1 reflection against 6.0e-4 -- 317x (``tests/fdtd/test_fd_absorber.py``).
So :class:`Shape` names the three questions separately, plus the four dispatches that used to
be written ``isinstance(shape, Slab) ... else assume it is a Box``: :meth:`Shape.layer_axis`,
:meth:`Shape.bounded_axes`, :meth:`Shape.describe`, :meth:`Shape.span_rows`. Every one of
them is answered by the shape and none of them has an "else" that guesses.

``cell_box`` is deleted rather than kept beside the protocol: nothing in ``wavebit.fdtd._legacy`` is
released, so a retired name goes outright (see ``docs/`` and the ``fourier_type``/``combine=``
precedents) -- a second spelling of "which cells" is exactly the ambiguity this replaces.

**Two discretisations, and the shape is the one that says which.** From stage 4's C2 a shape
either *fills cells whole* (:class:`Slab`, :class:`Region`) or *fills them fractionally*
(:class:`Sphere`, and from C4 the extruded bodies :class:`Box`, :class:`Cylinder` and
:class:`Prism`). :meth:`Shape.body` is that answer -- the
``src/wavebit/fdtd/_shapes`` body this shape is smoothed from, or ``None``. It is a method and not a
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
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
import math

from wavebit._common.declarations import spec
from wavebit._common.declarations import static_float
from wavebit._common.declarations import torch_carried
from wavebit._common.declarations import is_jax_typed

__all__ = ['Shape', 'Slab', 'Sphere', 'Box', 'Cylinder', 'Prism', 'Structure', 'Region',
           'Plane', 'AXES']

#: The three axes in the order the public API spells a triple.
AXES = ('x', 'y', 'z')

#: Shape keywords stage 4 will take. Refused here by name, with the stage, except where the
#: spelling now exists -- ``radius`` names :class:`Sphere` (C2) rather than a stage.
_LATER_SHAPES = {
    'radius': 'a sphere or a cylinder', 'center': 'a positioned body',
    'size': 'a box', 'vertices': 'a prism', 'raster': 'a raster',
    'angle': 'a turned body', 'height': 'a cylinder or a prism',
}


from wavebit._common.backend import eager_under_torch_compile


@eager_under_torch_compile
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
            f'read as a Slab or a Region.')

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
        """The ``src/wavebit/fdtd/_shapes`` body this shape is **smoothed** from, or ``None``.

        ``None`` means cell-snapped: the shape fills the cells :meth:`filled_cells` names
        whole, and :func:`wavebit.fdtd._prepare.coefficients.build` paints its material over them
        with a ``where``, which is what every stage-0..3 golden holds to the bit.

        A body means fractional: the lowering asks it, at **each component's own Yee point**,
        what fraction of that voxel the shape fills and which way the surface faces there
        (``src/wavebit/fdtd/_prepare/occupancy.py``), and mixes the two media by the
        Farjadpour/Kottke rule rather than choosing one of them.

        There is no default and no third answer. A shape that returns a body must refuse
        :meth:`filled_cells` -- the two are the same question asked of the two
        discretisations, and a shape that answered both would have said its edge is a cell
        boundary and also that it is not.
        """
        raise self._unanswered(
            'body', 'whether it fills cells whole or fractionally is unknown -- a shape is '
            'cell-snapped (body() is None) or smoothed (body() is a wavebit.fdtd._shapes body).')

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
        was written the three readers in ``src/wavebit/fdtd/_prepare/sources.py`` reached for ``shape.axis`` and
        ``shape.cells(grid)`` after branching on this method, so a shape that answered the
        whole protocol and declared itself a layer died there with a bare ``AttributeError``
        -- the guess-from-the-type failure this class exists to replace, one method later.
        ``tests/fdtd/test_fd_shape_protocol.py`` walks a protocol-only layer through all three.
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
        about a side that is not in the grid. ``Region(size=(0.3, 0.3, 1.0))`` in a box 0.2
        wide across x and y bounds z alone, and its answer is 1.0, not 0.3.

        That is why it takes ``grid``: "bounded" is a question about the grid, so a signature
        without one cannot answer the sentence above -- it can only return the shortest side
        as spelled, clipped sides and all.

        Stage 4's "too few cells across" refusal (C5) is the reader; it is on the protocol
        from the start so that every shape answers it in the same units as its own lengths,
        rather than the refusal reconstructing it per type.
        """
        raise self._unanswered('min_extent', 'its smallest bounded side is unknown.')


@eager_under_torch_compile
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
                f'fd.Structure(fd.Region(size=..., center=...), material) (stage 3, cell-snapped) '
                f'or the smoothed fd.Box(size=..., center=...); spheres, cylinders and polygons '
                f'are stage 4\'s other smoothed bodies, fd.Sphere, fd.Cylinder and fd.Prism.')
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
        a deliberate re-freeze of all four archives. The smoothed rectangular body is
        :class:`Box`.
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
            f'Slab(y=...) or Slab(z=...). The smoothed bodies of stage 4, filled at each '
            f'component\'s own Yee point, are fd.Sphere(radius=..., center=...), '
            f'fd.Cylinder(radius=..., height=..., axis=...), fd.Prism(vertices=..., height=..., '
            f'axis=...) and the rectangular fd.Box(size=..., center=..., angle=...); a '
            f'cell-snapped block is fd.Region(size=..., center=...). A raster shape is refused '
            f'outright: its interface normal cannot come from the shape.')
    _slab_init(self, *args, **kwargs)


Slab.__init__ = eager_under_torch_compile(_slab_init_with_stages)


@spec(positional=('shape', 'material'))
class Structure:
    """A shape filled with a material: ``fd.Structure(fd.Slab(z=(-0.2, 0)), glass)``.

    Both may be given positionally, in that order, because there is no other order a reader
    would expect. ``material`` takes the scalar shorthand: ``fd.Structure(slab, 2.25)`` is
    ``fd.Material(eps=2.25)``.

    ``shape`` is an :class:`Slab` (stage 1), a :class:`Region` (stage 3): a cell-snapped
    rectangular block, the minimum lateral patterning a diffraction-order check needs (a
    lamellar grating is a Region spanning one transverse axis, a crossed pillar grating a Region
    spanning neither), or -- from stage 4's C2 -- a :class:`Sphere`, the first shape whose
    edge is **not** a cell boundary, and from C4 the extruded bodies :class:`Box`,
    :class:`Cylinder` and :class:`Prism`. CSG of them is later in stage 4.

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
    case is refused by name (``src/wavebit/fdtd/_prepare/checks.py::check_smoothed``, door 5), except
    for the separately lowered flush junction below.

    **``junction=``: a Cylinder or unturned Box standing flush on a smoothed Box face.**
    The standing body's ``'split'`` default layers the substrate below the pillar/background
    mixture. ``'staircase'`` assigns each junction point its owning medium; ``'refuse'``
    retains the shared-surface refusal. The choice is static in the plan. The substrate
    must be an unturned :class:`Box`, with its rim clear of the junction; a cell-snapped
    :class:`Slab` or :class:`Region` remains refused. Junction media must be isotropic,
    lossless, non-dispersive and non-magnetic. Split measured second order on the flat-film
    anchor, but the rim order on one pillar scene remains unresolved; the report states
    this limitation. See ``docs/validation/fdtd/fd-subpixel-occupancy.md``, The junction.

    **``discretisation=``: which material discretisation this structure gets.** ``'voxel'``,
    the default and every scene's behaviour until stage 4's C3, is *not* the name of a rule --
    it is "whatever this shape's :meth:`Shape.body` already says", a whole-cell replace for a
    cell-snapped shape and the diagonal Farjadpour/Kottke mix plus the constitutive cross terms
    for a smoothed one. What the two have in common, and what the word names, is that every
    coefficient is a function of one component's **own voxel**. ``'stencil'`` asks instead for
    the symmetric extended-stencil operator ``docs/design/fd-symmetric-stencil.md`` derives: the exact
    edge and dual-face interface constraints, solved once on the host, whose rows couple each
    edge to every edge within two cells. It is opt-in, it is prepared at ``fd.plan`` time and
    never inside a trace, and ``fd.check`` prints it.

    **``memory_bound=``** (bytes, only beside ``discretisation='stencil'``) raises the host
    memory a preparation may be predicted to take before it is refused. The default admits the
    research sphere at its three placements at 40 and 56 cells per wavelength with a stated
    margin (``docs/design/fd-symmetric-stencil.md``); a larger value is an explicit decision to trust
    the prediction past the sizes it was measured at.

    **Why here, and not the ``smoothing=`` the module docstring refuses.** That paragraph turns
    down a keyword that would *switch a rule* for the same shape, or key itself on whether a
    radius happened to be traced, because either makes ``jax.grad`` and a finite difference
    compute two different functions. Both objections are kept rather than dodged. This value is
    static, a typo is refused outright, it is in the plan and in the skeleton, and it is never
    read from whether anything is a tracer. A structure that asks for the stencil takes plain
    numbers only in its material values -- Python ``int``/``float`` and NumPy scalars -- and
    any array or tensor, traced or not, is **refused by name** (``StencilNumberRefused``) rather
    than quietly given the other rule, which is exactly the failure ``combine=`` and
    ``fourier_type=`` were deleted for. That door is a type and not a question about tracing
    because a Torch forward-mode dual reads as a number and passed a tracing test. The ball's
    radius and centre need no second door: they are :class:`Sphere`'s static fields, and
    ``static_float`` refuses every traced value there -- forward mode included -- on either
    discretisation.

    It is on the structure and not on the :class:`Shape` or on :class:`~wavebit.fdtd._legacy.Run`
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
    junction: str = 'split'

    def __post_init__(self):
        import math
        from numbers import Real

        from wavebit.materials.time_domain import as_material
        from wavebit.fdtd._prepare.stencil import DEFAULT
        from wavebit.fdtd._prepare.stencil import STENCIL
        if self.discretisation not in (DEFAULT, STENCIL):
            raise ValueError(
                f'Structure(discretisation={self.discretisation!r}): the two spellings are '
                f'{DEFAULT!r} -- the rule this structure\'s shape already names, which is what '
                f'every scene has had -- and {STENCIL!r}, the symmetric extended-stencil '
                f'operator of docs/design/fd-symmetric-stencil.md. A value nobody reads is silently '
                f'ignored, so a third one is refused here rather than falling back to the '
                f'default: the two converge at different orders and a scene that got the one '
                f'it did not ask for would read as a physics result.')
        if (self.discretisation == STENCIL and isinstance(self.shape, (Sphere, Cylinder, Box, Prism))
                and self.shape.range is not None):
            raise ValueError(f"Structure(discretisation='stencil') does not support "
                             f"{type(self.shape).__name__}(range=): "
                             'the prepared operator depends on the actual geometry. Use the '
                             'voxel discretisation for a sweep with a fixed support range.')
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
                'fd.Slab(z=(lo, hi)) or fd.Region(size=..., center=...), or subclass fd.Shape '
                'and answer its methods.')
        if not isinstance(self.shape, Shape):
            raise TypeError(
                f'Structure(shape=...): the shapes are fd.Slab(z=(lo, hi)), the cell-snapped '
                f'fd.Region(size=..., center=...), and stage 4\'s smoothed bodies '
                f'fd.Sphere(radius=..., center=...), fd.Box(size=..., center=..., angle=...), '
                f'fd.Cylinder(radius=..., height=..., axis=...) and fd.Prism(vertices=..., '
                f'height=..., axis=...); got {type(self.shape).__name__}. External shapes get in '
                f'by subclassing fd.Shape and answering its protocol. CSG of the bodies is later '
                f'in stage 4; a raster shape is refused outright (its interface normal cannot '
                f'come from the shape).')
        from wavebit.fdtd._prepare.junction import DEFAULT as JUNCTION_DEFAULT
        from wavebit.fdtd._prepare.junction import RULES
        if not isinstance(self.junction, str) or self.junction not in RULES:
            import difflib
            hint = difflib.get_close_matches(str(self.junction), RULES, n=1)
            message = (f'Structure(junction={self.junction!r}): choose split (the layered '
                       'interface), staircase (the medium at each point), or refuse.')
            if str(self.junction).lower() in ('blend', 'mix', 'average', 'fdtdx'):
                message += (' The FDTDX-style blend is not offered: its junction term '
                            'is first order and depends on structure listing order.')
            elif str(self.junction) == 'meep':
                message += " Did you mean 'staircase'?"
            elif hint:
                message += f' Did you mean {hint[0]!r}?'
            raise ValueError(message)
        if self.junction != JUNCTION_DEFAULT and (self.shape.body() is None
                                                   or isinstance(self.shape, Sphere)):
            raise ValueError(f'Structure(junction={self.junction!r}): this shape cannot '
                             'stand on a smoothed face; no junction rule reads this value.')
        object.__setattr__(self, 'material', as_material(self.material, 'Structure(material=)'))


@spec
class Region(Shape):
    """An axis-aligned box, ``fd.Region(size=(sx, sy, sz), center=(cx, cy, cz))``.

    Three uses, one object: the region of an :class:`~wavebit.fdtd._legacy.TFSF` source (stage 2), the
    closed surface of an ``fd.Flux`` or the volume of an ``fd.Field`` monitor (stage 3), and --
    from stage 3 -- a **structure**, ``fd.Structure(fd.Region(...), material)``. Both triples are
    static lengths in the scene's coordinates (the box's faces pick node planes or cells, so
    they fix the plan and its skeleton).

    It was ``fd.Box`` until stage 4's C4, which gave that name to the smoothed, rotatable
    rectangular **body** -- one name per concept, and the snapped box is the one the monitors
    and ``TFSF`` take. Its report rows keep the labels ``box.x``/``box.y``/``box.z`` the
    stage-3 archive froze, so the rename moved no skeleton.

    **As a structure it is cell-snapped exactly like** :class:`Slab`, on every axis at once:
    each face lands on the nearest cell boundary,

        i0 = round((x_lo - x_low) / dx),   i1 = round((x_hi - x_low) / dx)

    clipped to ``[0, n]``, and the cells ``i0 .. i1 - 1`` (times the same on y and z) carry the
    material -- every component of them, with no subpixel averaging -- and **the edges are
    not differentiable**: a size or a centre is static, and a gradient flows through the
    material's values only. ``check`` prints the snapped span beside the one asked for on each
    axis. A face that does not fall on a cell boundary moves by up to half a cell, which is a
    first-order error in ``dx``; the smoothed :class:`Box` fills fractionally and exactly
    (stage 4's C4), and its edge is to become differentiable with ``range=`` (C6); for this one a
    convergence study in ``dx`` is the route to a right answer. A
    box that extends past the domain on an axis is clipped there, which is how a lamellar
    grating spells "runs the whole period" along its uniform axis.
    """

    size: tuple
    center: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        for name in ('size', 'center'):
            value = getattr(self, name)
            if not (isinstance(value, (tuple, list)) and len(value) == 3):
                raise ValueError(f'Region({name}={value!r}): a triple (x, y, z).')
            object.__setattr__(self, name, tuple(static_float('Region', name, v) for v in value))
        if any(not s > 0 for s in self.size):
            raise ValueError(f'Region(size={self.size}): every size is a positive length.')

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
                    f'Region(size={self.size}, center={self.center}) snaps to the empty span '
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

        ``fd.Region`` is three things at once -- a ``TFSF`` region, a monitor's surface or
        volume, and a structure -- and a region has to snap to node planes to pick indices.
        One name cannot carry two discretisations, so the smoothed, rotatable, differentiable
        rectangular **body** is a separate class, ``fd.Box`` since C4, and this one does not
        move. Every stage-3 golden holds to the bit because of that.
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
        return f'Region(size={self.size}, center={self.center})'

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

    Unlike :class:`Slab` and :class:`Region` it does not snap to anything. The lowering asks it,
    at each of the six components' **own** Yee points, what fraction ``f`` of that component's
    voxel the ball fills and how the surface is oriented there (``w_c``, the mean of ``n_c^2``
    over the surface inside the voxel), and mixes the ball's material with whatever the cell
    already held by the Farjadpour/Kottke rule -- see
    ``src/wavebit/fdtd/_prepare/coefficients.py::mix`` for the formula and its citations, and
    ``src/wavebit/fdtd/_prepare/occupancy.py`` for the stage that produces the pair. Its exact clipped
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
    operator is made positive definite by the star lift (``src/wavebit/fdtd/_prepare/stars.py``), which is
    checked, with the time-step bound, before a run starts.

    ``range=((x0, x1), (y0, y1), (z0, z1))`` reserves a fixed support box for a sweep.
    It contains the **whole sphere**, including its radius, rather than just its centre
    (unlike ``Dipole.range``). Every sphere in the sweep must fit inside it. With the default
    voxel discretisation, changes of radius and centre inside the same range can reuse a
    plan and its compiled stepping program, even across cell boundaries. The range fixes
    allocation, not material: :meth:`body`, the volume report and every planning check still
    use the actual sphere. An oversized range costs memory and may reach a boundary or an
    absorbing layer the coupled operator cannot touch. The prepared stencil is refused.

    With a range, radius and centre can be differentiated through ``fd.run`` or ``fd.lower``
    using a plan built from concrete geometry first. One or more ranged Sphere, Cylinder,
    Box and Prism bodies may be differentiated together when all reserved supports,
    including their interpolation halos, are pairwise disjoint. Every ranged body needs
    isotropic lossless material, and no initial field is allowed. Concrete partners
    without a range (a substrate Slab, a Region, an unranged body, a lossy or dispersive
    layer) may join them when each, widened by one cell, stays off every traced halo; a
    flush junction among them lowers as usual. A body's or partner's ``eps`` and ``mu`` may be
    traced with the geometry, each as one scalar of a plain Material, a substrate's in a PML
    included; a medium filling a stabilized PML layer, the background of a scene with a
    plane-wave source or a PML matched to it, conductivities and pole parameters stay
    concrete. A traced
    Cylinder or unturned Box may stand flush on an unranged smoothed Box face under
    ``junction='split'``, with its base held on the face. Every
    source and monitor family is admitted, provided the cell layers where a PlaneWave or TFSF
    normalisation or a Diffraction plane reads its plain medium stay outside every traced
    body's reserved support. Every full interpolation halo must stay inside the domain and
    clear of absorbing layers, including for an unturned Box. Runtime guards mark
    out-of-range or under-resolved geometry unstable and return NaN fields. Traced reports
    leave geometry measurements unavailable. Without a range, radius and centre remain
    static. The derivative is that of the discrete solver; it is piecewise smooth, not a
    guarantee of continuum sensitivity or convergence.

    **What a sphere may not meet: six doors of C2 and C4.** All six are
    ``src/wavebit/fdtd/_prepare/checks.py::check_smoothed``, in the order ``src/wavebit/fdtd/_legacy.py``
    lists them, and each names the commit that owns the case it turns away.

    1. **No conductivity and no poles where it reaches.** Its own material must not be an
       ``fd.Conductor`` or an ``fd.Dispersive``, and neither may the background or any
       structure whose cells it shares. C8's film exceptions require a single concrete,
       unturned and unranged :class:`Box` spanning exactly two full grid axes in a
       plain isotropic, non-magnetic dielectric host; it is not an exception for a
       sphere. That film's separated faces and support stay clear of absorbing
       layers and the normal domain walls, and it admits no partner, initial field
       or geometry/material differentiation. A conductive film's normal mixed cells carry an induced
       Debye memory; a film with one passive Drude or Lorentz pole carries a spatial
       oscillator for the exact harmonic normal response. A single passive Debye pole
       has a spatial relaxation time and strength, with an instantaneous update shift.
       A finite sum of passive Debye poles uses coupled normal relaxation modes,
       with one reserved memory slot per declared pole and exact zero/equal-time
       reductions. The existing one- and two-pole routes retain their calculations.
       A finite Drude/Lorentz sum may have unequal damping; its physical
       polarizations share a normal restoring force and the corresponding energy
       cross term. A declared mixture of Debye and Drude/Lorentz poles uses
       a hybrid recurrence with four arrays per pole and its own passive
       integer-time storage. A separate stack scope admits two or more of the
       concrete Conductor films, including sigma=0, with a common normal and
       pairwise disjoint padded grid supports in the same plain host. It keeps
       one interface memory per member and checks all members. It does not
       admit touching films or stacks of different declared pole families.
       General conductive bodies, lossy junctions,
       lossy hosts and curved dispersive interfaces remain refused. See ``docs/validation/fdtd/fd-conductive-film.md``,
       ``docs/validation/fdtd/fd-single-pole-film.md``, ``docs/validation/fdtd/fd-debye-film.md``,
       ``docs/validation/fdtd/fd-two-debye-film.md``, ``docs/validation/fdtd/fd-debye-sum-film.md`` and
       ``docs/validation/fdtd/fd-oscillator-sum-film.md`` and ``docs/validation/fdtd/fd-hybrid-film.md`` for the exact
       contracts and validation status.
    2. **No wrapping face.** A sphere that reaches a ``Periodic`` or ``Bloch`` face is refused
       because resolving its wrapped part means summing the geometry over the image ring, and
       that fold -- which needs the cell's origin, which a body does not carry -- is not
       written. What is carried without it is a body **uniform along** the periodic axis over
       the whole domain (an extruded body along its own axis, or an unturned :class:`Box`
       spanning it), and a ball never is. An ``fd.Axis1D`` axis is refused too, with its own
       sentence and for a different reason: that axis stands for one the field is *invariant*
       along, and a ball is not.
    3. **No reaching a face of the box in a way the grid cannot carry whole.** A cell-snapped
       ``fd.Region`` is clipped exactly; a ball is not, and the six components disagree about
       where the box ends. The two arms are split by **cause** and not by side: one for a ball
       that *leaves the domain*, at either end of an axis, and one for a ball that is inside
       and whose last half cell before a ``+`` face the plan *trims*. Each carries its own
       message and its own remedy, because the remedies are not interchangeable.
    4. **No anisotropic eps or mu.** The effective tensor is the Farjadpour/Kottke rule for two
       *isotropic* phases; blending each component with its own ``eps`` is not the tau-transform
       the anisotropic construction calls for, and on a ball the normal is almost never along a
       principal axis. The door covers the same three media as door 1.
    5. **No second smoothed surface in one voxel.** A painter mixes into what the cell already
       holds, which is two phases and *one* surface; two smoothed surfaces in one voxel do not
       compose (73% high on two half spaces, 122% on one ``fd.Sphere`` declared twice). A ball
       inside another ball it shares no surface voxel with is fine and is not refused.
    6. **No surface in a voxel a cell-snapped face cuts** (C4). The snapped cells are all or
       nothing to the painter where the geometry splits them, so such a voxel holds three
       media for a rule of two; a ball resting on a ``Slab`` substrate is the case, and a ball
       over or inside a snapped shape a voxel from its faces is fine.
    """

    radius: float
    center: tuple = (0.0, 0.0, 0.0)
    range: object = None

    def __post_init__(self):
        radius = (_shape_leaf('Sphere', 'radius', self.radius) if self.range is not None
                  else static_float('Sphere', 'radius', self.radius))
        if not _shape_traced(radius) and (not math.isfinite(radius) or not radius > 0):
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
        center = tuple(_shape_leaf('Sphere', 'center', v) if self.range is not None
                       else static_float('Sphere', 'center', v) for v in center)
        if not all(_shape_traced(v) or math.isfinite(v) for v in center):
            raise ValueError(f'Sphere(center={center!r}): every coordinate must be finite.')
        object.__setattr__(self, 'center', center)
        if self.range is not None:
            rng = self.range
            if not (isinstance(rng, (tuple, list)) and len(rng) == 3
                    and all(isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in rng)):
                raise ValueError('Sphere(range=...): three finite (lo, hi) pairs, '
                                 '((x0, x1), (y0, y1), (z0, z1)), containing the whole sphere.')
            pairs = tuple(tuple(_finite('Sphere', 'range', v) for v in pair) for pair in rng)
            for axis, (lo, hi), c in zip(AXES, pairs, center):
                if not lo < hi:
                    raise ValueError(f'Sphere(range=...): the {axis} pair must have lo < hi.')
                if (not _shape_traced(c) and not _shape_traced(radius)
                        and (c - radius < lo or c + radius > hi)):
                    raise ValueError(f'Sphere(range=...): the whole sphere must lie inside '
                                     f'the declared {axis} range ({lo}, {hi}), including its '
                                     f'radius; containing only the centre is not enough.')
            object.__setattr__(self, 'range', pairs)

    @property
    def low(self):
        return tuple(c - self.radius for c in self.center)

    @property
    def high(self):
        return tuple(c + self.radius for c in self.center)

    # -- the protocol (see :class:`Shape`) ----------------------------------------------------

    def body(self):
        """The ``_shapes`` ball: an exact clipped-ball fill, and the tangent plane's moment."""
        from wavebit.fdtd._shapes import Sphere as _Ball
        return _Ball(radius=self.radius, center=self.center)

    def support(self, grid):
        """Every cell whose voxels the ball can reach, **with one cell of margin** per side.

        A cell's six component voxels are not the cell: component ``c`` of ``E`` sits half a
        cell along axis ``c``, so the union of a cell's voxels spans
        ``[low + (i - 1/2) dx, low + (i + 1) dx]`` along each axis and reaches a whole ``dx``
        past the node. The margin is one cell rather than the half this needs, because the
        cost is a shell of voxels whose fill is exactly zero and the alternative is an
        off-by-one that removes material from a real surface. It is gated on the volume:
        ``tests/fdtd/test_fd_occupancy.py::test_the_support_box_holds_the_whole_ball`` sums the
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

        With ``range=`` the declared box supplies these bounds, retaining the same margin.
        It must contain the whole ball, so the fixed allocation contains every sweep point.
        """
        import math
        bounds = self.range if self.range is not None else tuple(zip(self.low, self.high))
        out = []
        for a in range(3):
            k0 = math.floor((bounds[a][0] - grid.low[a]) / grid.dx) - 1
            k1 = math.ceil((bounds[a][1] - grid.low[a]) / grid.dx) + 1
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
        317x (``tests/fdtd/test_fd_absorber.py``).
        """
        k0, k1 = self.support(grid)[axis]
        return k1 <= cells[0] or k0 >= cells[1]

    def bounded_axes(self, grid):
        """All three, always -- and not the axes its *box* happens to cut.

        :class:`Region` may leave an axis out because a box clipped to the domain really is
        uniform along it, so a coefficient array of extent 1 there is exact. A ball is never
        uniform along an axis it reaches, whatever its support clipped to, so an extent-1 axis
        would broadcast one plane of fills over the whole grid. The mutant that copies Region's
        rule here is ``tests/fdtd/test_fd_occupancy.py::test_a_ball_varies_along_every_axis``.
        """
        return (0, 1, 2)

    def describe(self):
        return f'Sphere(radius={self.radius}, center={self.center})'

    def span_rows(self, grid):
        """One row per axis, labelled ``sphere.x/y/z``: asked, and got as the box clips it.

        A smoothed shape does not snap, so ``check`` never prints ``<- snapped`` for one --
        the difference from the ``Slab``/``Region`` rows is the whole point of the shape -- and
        where got differs from asked it is the box's face, printed ``<- clipped`` (C5; before,
        the row claimed asked and got were the same even with a quarter of the body outside).
        With ``range=`` the explicitly labelled range rows describe the fixed allocation;
        the report's body block still measures the actual sphere.
        """
        if self.range is not None:
            return tuple((f'sphere.range.{axis}', lo, hi, max(lo, grid.low[a]),
                          min(hi, grid.high[a]))
                         for a, (axis, (lo, hi)) in enumerate(zip(AXES, self.range)))
        return tuple((f'sphere.{axis}', self.low[a], self.high[a], max(self.low[a], grid.low[a]),
                      min(self.high[a], grid.high[a])) for a, axis in enumerate(AXES))

    def min_extent(self, grid):
        """Its diameter."""
        return 2 * self.radius

    def volume(self):
        """``4 pi r^3 / 3``: what ``fd.check`` compares the lowered fill with (C5)."""
        return 4.0 * math.pi * self.radius ** 3 / 3.0


def _extrusion_axis(where, axis):
    """``'x'``, ``'y'`` or ``'z'``; anything else refused by name, a tilted axis as stage 5."""
    if isinstance(axis, str) and axis in AXES:
        return axis
    raise ValueError(
        f'{where}(axis={axis!r}): the extrusion axis is one of \'x\', \'y\', \'z\'. A body '
        f'turns only about its own axis (angle=), which keeps it uniform along that axis -- '
        f'what the absorbing-layer and periodic-face rules read. A tilted axis is stage 5, and '
        f'is refused here rather than approximated.')


def _shape_traced(value):
    """A carried scalar, including Torch forward-mode tangents invisible to float()."""
    if torch_carried(value):
        return True
    if is_jax_typed(value):
        try:
            float(value)
        except (TypeError, ValueError):
            return True
    return False


def _shape_leaf(where, name, value):
    """Keep ranged shape leaves intact; normalise ordinary numbers as before."""
    if (isinstance(value, (bool, str, complex)) or value is None
            or 'complex' in str(getattr(value, 'dtype', ''))
            or tuple(getattr(value, 'shape', ())) != ()):
        raise ValueError(f'{where}({name}=...): a real scalar is required.')
    return value if _shape_traced(value) else static_float(where, name, value)


def _ranged_coordinate(where, name, value):
    value = _shape_leaf(where, name, value)
    if not _shape_traced(value) and not math.isfinite(value):
        raise ValueError(f'{where}({name}=...): a finite number is required.')
    return value


def _ranged_length(where, name, value):
    value = _ranged_coordinate(where, name, value)
    if not _shape_traced(value) and not value > 0:
        raise ValueError(f'{where}({name}=...): a finite positive length is required.')
    return value


def _contained_range(where, bounds, center, halves):
    if not (isinstance(bounds, (tuple, list)) and len(bounds) == 3
            and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in bounds)):
        raise ValueError(f'{where}(range=...): three finite (lo, hi) pairs are required.')
    pairs = tuple(tuple(_finite(where, 'range', v) for v in p) for p in bounds)
    for axis, (lo, hi), c, half in zip(AXES, pairs, center, halves):
        if not lo < hi:
            raise ValueError(f'{where}(range=...): the {axis} pair must have lo < hi.')
        if (not _shape_traced(c) and not _shape_traced(half)
                and (c - half < lo or c + half > hi)):
            raise ValueError(f'{where}(range=...): the whole {where.lower()} must lie inside '
                             f'the declared {axis} range, including its full extent.')
    return pairs


def _finite(where, name, value):
    """A static, finite real: a traced one is refused by ``static_float``, a NaN or inf here.

    ``name`` may carry an index, ``size[1]``, so the refusal names the element and not the
    whole keyword.
    """
    value = static_float(where, name, value)
    if not math.isfinite(value):
        raise ValueError(f'{where}({name}={value!r}): a finite number.')
    return value


def _length(where, name, value):
    """A static, finite, positive length. A length is a size: a negative one is refused."""
    value = _finite(where, name, value)
    if not value > 0:
        raise ValueError(f'{where}({name}={value!r}): a finite positive length. A length is a '
                         f'size here, so a negative one is refused rather than read as '
                         f'subtracting the shape.')
    return value


def _triple(where, name, value, check):
    """Check three values element by element, preserving backend leaves for the check."""
    if hasattr(value, 'tolist') and getattr(value, 'shape', None) == (3,):
        # Indexing preserves backend leaves; tolist() detaches a Torch vector.
        value = tuple(value[i] for i in range(3))
    if not (isinstance(value, (tuple, list)) and len(value) == 3):
        what = ('a triple (x, y, z), absolute, in the scene\'s coordinates -- not the '
                'cell-centre origin the rcwa path spells as cx=/cy=' if name == 'center'
                else 'three lengths (x, y, z)')
        raise ValueError(f'{where}({name}={value!r}): {what}.')
    return tuple(check(where, f'{name}[{i}]', v) for i, v in enumerate(value))


def _turned(vertices, degrees):
    """``vertices`` turned counter-clockwise by ``degrees`` -- the kernel bodies' own formula."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return tuple((u * c - v * s, u * s + v * c) for u, v in vertices)


def _hull_width(points):
    """The width of the convex hull of ``points``: its narrowest caliper distance."""
    pts = sorted(set(points))

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    best = math.inf
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        side = math.hypot(b[0] - a[0], b[1] - a[1])
        best = min(best, max(abs(turn(a, b, p)) for p in hull) / side)
    return best


#: The right-handed pair across each axis -- ``src/wavebit/fdtd/_shapes/_bodies.py::_transverse``'s.
_TRANSVERSE = ((1, 2), (2, 0), (0, 1))


@_eager_under_torch_compile
class _Extruded(Shape):
    """What :class:`Box`, :class:`Cylinder` and :class:`Prism` share: a section along an axis.

    Private; the three classes below are the API. Each is a ``src/wavebit/fdtd/_shapes`` body with a
    cross-section, an extrusion ``axis`` and a ``center``, smoothed at every component's
    own Yee point exactly as :class:`Sphere` is -- the same ``coefficients.mix``, the same
    doors. What an extrusion adds is an axis the body is **uniform along**, and that is the one
    question its protocol answers differently from a ball's: the absorbing-layer rule, the
    ``Axis1D`` and periodic-face doors and ``_reaches_outside`` all read
    :meth:`uniform_along`, so a pillar may run the whole height of a periodic cell and a
    substrate spelled as a body may run out through a face.
    """

    #: The ``span_rows`` label. ``box`` is :class:`Region`'s, frozen by the stage-3 archive.
    _label = ''

    def _section(self):
        """``((u_lo, u_hi), (v_lo, v_hi))``: the cross-section's extent about the centre."""
        raise NotImplementedError

    def _feature(self, grid):
        """The narrowest width of the cross-section, for :meth:`min_extent`."""
        raise NotImplementedError

    def _bounds(self):
        a = AXES.index(self.axis)
        low, high = [0.0] * 3, [0.0] * 3
        low[a] = self.center[a] - self.height / 2
        high[a] = self.center[a] + self.height / 2
        for b, (lo, hi) in zip(_TRANSVERSE[a], self._section()):
            low[b], high[b] = self.center[b] + lo, self.center[b] + hi
        return tuple(low), tuple(high)

    @property
    def low(self):
        return self._bounds()[0]

    @property
    def high(self):
        return self._bounds()[1]

    def support(self, grid):
        """Every cell its voxels can reach, with one cell of margin: :meth:`Sphere.support`'s
        rule and its measured reason, on the body's axis-aligned bounding box."""
        bounds = getattr(self, 'range', None)
        bounds = bounds if bounds is not None else tuple(zip(self.low, self.high))
        out = []
        for a in range(3):
            k0 = math.floor((bounds[a][0] - grid.low[a]) / grid.dx) - 1
            k1 = math.ceil((bounds[a][1] - grid.low[a]) / grid.dx) + 1
            out.append((min(max(int(k0), 0), grid.n[a]), min(max(int(k1), 0), grid.n[a])))
        return tuple(out)

    def filled_cells(self, grid):
        """Refused: a body's edge is not a cell boundary, so no cell range is "the body"."""
        return super().filled_cells(grid)

    def _spans(self, a, cells, grid):
        """Whether the body's extent along ``a`` covers every voxel of those cell layers.

        The voxels of layers ``c0 .. c1 - 1`` run from ``c0 - 1/2`` to ``c1`` cells -- the three
        components whose Yee offset along ``a`` is zero are centred on the nodes -- and the test
        asks a whole cell either side, the margin :meth:`support` keeps too.
        """
        lo = grid.low[a] + (cells[0] - 1) * grid.dx
        hi = grid.low[a] + (cells[1] + 1) * grid.dx
        return self.low[a] <= lo and self.high[a] >= hi

    def _across(self, a):
        """Whether the section is made of faces normal to the grid's own axes (Box, unturned)."""
        return False

    def uniform_along(self, axis, cells, grid):
        """True where it cannot reach those layers, or where it spans them along a straight axis.

        Along its own axis the section is the same everywhere between the caps, so a body that
        spans the layers is uniform there. Across it, a bounded section changes wherever it is
        -- except an unturned :class:`Box`, whose faces are planes of the grid's own axes and
        which is uniform along any axis it spans. A turned one is not: ``angle=90`` is not
        exactly a quarter turn in floating point, and a box meant to be axis-aligned is written
        with its sizes swapped.
        """
        k0, k1 = self.support(grid)[axis]
        if k1 <= cells[0] or k0 >= cells[1]:
            return True
        if isinstance(self, (Box, Cylinder, Prism)) and self.range is not None:
            if isinstance(self, Prism):
                dimensions = (self.height, *(v for p in self.vertices for v in p))
            elif isinstance(self, Box):
                dimensions = self.size
            else:
                dimensions = (self.radius, self.height)
            if any(_shape_traced(v) for v in (*dimensions, *self.center)):
                # No host truth test on moving faces. False is conservative; public
                # traced bodies are certified wholly inside the domain before lowering.
                return False
        straight = axis == AXES.index(self.axis) or self._across(axis)
        return straight and self._spans(axis, cells, grid)

    def bounded_axes(self, grid):
        """All three, always, as :class:`Sphere`: every coefficient array is sampled in full.

        Along an axis it spans an extent-1 array would be exact; the saving is not taken, so
        that one painter rule serves every body.
        """
        return (0, 1, 2)

    def span_rows(self, grid):
        """One row per axis, labelled by the body: asked, and got as the box clips it.

        It does not snap; a difference is the box's face (:meth:`Sphere.span_rows`).
        """
        bounds = getattr(self, 'range', None)
        if bounds is not None:
            return tuple((f'{self._label}.range.{axis}', lo, hi, max(lo, grid.low[a]),
                          min(hi, grid.high[a]))
                         for a, (axis, (lo, hi)) in enumerate(zip(AXES, bounds)))
        return tuple((f'{self._label}.{axis}', self.low[a], self.high[a],
                      max(self.low[a], grid.low[a]), min(self.high[a], grid.high[a]))
                     for a, axis in enumerate(AXES))

    def _inside(self, a, grid):
        """Its extent along ``a`` as it reaches into the box (:meth:`Shape.min_extent`)."""
        return min(self.high[a], grid.high[a]) - max(self.low[a], grid.low[a])

    def min_extent(self, grid):
        """The narrowest section width, and the height as it reaches into the box.

        An axis the body runs out of the domain through on both sides has no feature size along
        it (:meth:`Shape.min_extent`), and one it leaves through one face is measured inside;
        so is the across-axis size of an unturned :class:`Box`.
        """
        a = AXES.index(self.axis)
        sides = [self._feature(grid)]
        if not self._spans(a, (0, grid.n[a]), grid):
            sides.append(self._inside(a, grid))
        sides = [s for s in sides if s is not None]
        return min(sides) if sides else None


@spec
class Box(_Extruded):
    """A smoothed rectangular body: ``fd.Box(size=(sx, sy, sz), center=(cx, cy, cz))``.

    The body a structure fills fractionally, like :class:`Sphere`, turned by ``angle``
    (degrees, counter-clockwise about ``+axis``) about its own ``axis`` through its centre.
    Not :class:`Region`, the cell-snapped box a monitor or a ``TFSF`` takes and a structure
    may still fill whole cells with; that one was ``fd.Box`` until stage 4's C4. One name per
    concept, and both docstrings point at the other.

    At ``angle=0`` its faces are planes of the grid's own axes, and three things follow that do
    not for any other body: it is uniform along every axis it spans (so a substrate spelled as
    a box may run out through the domain and through a periodic cell), its fill in a voxel is
    the product of three 1-D overlaps, and its constitutive cross terms are exactly zero, so the
    plan gives it none. With ``range=((x0, x1), (y0, y1), (z0, z1))``, size and
    centre may be differentiated under :class:`Sphere`'s isolated-body restrictions,
    using a concrete prebuilt plan. The range contains the entire rotated box.
    The angle and axis remain static and are recorded in the plan, so crossing
    between the zero-angle and rotated operators requires a new plan.
    Without a range all geometry stays static. It is ``src/wavebit/fdtd/_shapes``'
    :class:`~wavebit.fdtd._shapes.Box`, the four-vertex prism, whose fill and diagonal
    moment are exact at its faces, edges and corners. Rotated constitutive cross
    terms still use the nearest-surface proxy; derivatives do not certify convergence.
    """

    size: tuple
    center: tuple = (0.0, 0.0, 0.0)
    angle: float = 0.0
    axis: str = 'z'
    range: object = None

    _label = 'cuboid'

    def __post_init__(self):
        object.__setattr__(self, 'axis', _extrusion_axis('Box', self.axis))
        length = _length if self.range is None else _ranged_length
        coordinate = _finite if self.range is None else _ranged_coordinate
        object.__setattr__(self, 'size', _triple('Box', 'size', self.size, length))
        object.__setattr__(self, 'center', _triple('Box', 'center', self.center, coordinate))
        object.__setattr__(self, 'angle', _finite('Box', 'angle', self.angle))
        if self.range is not None:
            object.__setattr__(self, 'range', _contained_range(
                'Box', self.range, self.center, self._range_halves()))

    def _range_halves(self):
        """Axis-aligned half extents without host comparisons on traced side lengths."""
        u, v = _TRANSVERSE[AXES.index(self.axis)]
        half = [s / 2 for s in self.size]
        co, si = abs(math.cos(math.radians(self.angle))), abs(math.sin(math.radians(self.angle)))
        a, b = half[u], half[v]
        half[u], half[v] = co*a + si*b, si*a + co*b
        return tuple(half)

    @property
    def height(self):
        return self.size[AXES.index(self.axis)]

    def _half(self):
        u, v = _TRANSVERSE[AXES.index(self.axis)]
        return self.size[u] / 2, self.size[v] / 2

    def _section(self):
        a, b = self._half()
        corners = _turned(((-a, -b), (a, -b), (a, b), (-a, b)), self.angle)
        return ((min(p[0] for p in corners), max(p[0] for p in corners)),
                (min(p[1] for p in corners), max(p[1] for p in corners)))

    def _across(self, a):
        return self.angle == 0

    def _feature(self, grid):
        u, v = _TRANSVERSE[AXES.index(self.axis)]
        if self.angle:
            return min(self.size[u], self.size[v])
        sides = [self._inside(b, grid) for b in (u, v) if not self._spans(b, (0, grid.n[b]), grid)]
        return min(sides) if sides else None

    def volume(self):
        """``sx sy sz``, turned or not (:meth:`Sphere.volume`)."""
        return math.prod(self.size)

    def body(self):
        """The ``_shapes`` box: the polygon clip times the axial overlap, exact moments."""
        from wavebit.fdtd._shapes import Box as _Cuboid
        return _Cuboid(size=self.size, center=self.center, angle=math.radians(self.angle),
                       axis=self.axis)

    def describe(self):
        turn = f', angle={self.angle}, axis={self.axis!r}' if self.angle else ''
        return f'Box(size={self.size}, center={self.center}{turn})'


@spec
class Cylinder(_Extruded):
    """A smoothed circular cylinder: ``fd.Cylinder(radius=0.2, height=0.5, axis='z')``.

    A disc of ``radius`` extruded ``height`` along ``axis``, centred on ``center``, filled
    fractionally at every component's own Yee point as :class:`Sphere` is. There is no
    ``angle``: a circle turned about its own axis is the same circle, and a key whose gradient
    can only be zero is refused rather than accepted
    (``src/wavebit/fdtd/_shapes/_bodies.py::Cylinder`` measured it). Its fill is the exact
    disc-rectangle clip times the axial overlap; its surface moment is the tangent plane's,
    which on the rim -- where the side meets a cap -- is off by up to half (0.40-0.52 in ``w``
    at 4 to 16 cells per radius; ``docs/validation/fdtd/fd-subpixel-occupancy.md``, S7, counts those voxels).
    With ``range=((x0, x1), (y0, y1), (z0, z1))`` the whole cylinder must stay
    in a static box. Radius, height and centre may then be differentiated through
    ``fd.run`` or ``fd.lower`` using a concrete prebuilt plan, under the isolated-body
    restrictions of :class:`Sphere`. The axis remains static and is recorded in the plan.
    Without a range all geometry remains static. The derivative is of this discrete
    solver; the rim approximation and the need for physical convergence checks remain.
    """

    radius: float
    height: float
    axis: str = 'z'
    center: tuple = (0.0, 0.0, 0.0)
    range: object = None

    _label = 'cylinder'

    def __post_init__(self):
        object.__setattr__(self, 'axis', _extrusion_axis('Cylinder', self.axis))
        length = _length if self.range is None else _ranged_length
        coordinate = _finite if self.range is None else _ranged_coordinate
        object.__setattr__(self, 'radius', length('Cylinder', 'radius', self.radius))
        object.__setattr__(self, 'height', length('Cylinder', 'height', self.height))
        object.__setattr__(self, 'center', _triple('Cylinder', 'center', self.center, coordinate))
        if self.range is not None:
            axis = AXES.index(self.axis)
            halves = tuple(self.height / 2 if a == axis else self.radius for a in range(3))
            object.__setattr__(self, 'range', _contained_range(
                'Cylinder', self.range, self.center, halves))

    def _section(self):
        return (-self.radius, self.radius), (-self.radius, self.radius)

    def _feature(self, grid):
        return 2 * self.radius

    def volume(self):
        """``pi r^2 h`` (:meth:`Sphere.volume`)."""
        return math.pi * self.radius ** 2 * self.height

    def body(self):
        """The ``_shapes`` cylinder: the exact disc clip times the axial overlap."""
        from wavebit.fdtd._shapes import Cylinder as _Rod
        return _Rod(radius=self.radius, height=self.height, axis=self.axis, center=self.center)

    def describe(self):
        return (f'Cylinder(radius={self.radius}, height={self.height}, axis={self.axis!r}, '
                f'center={self.center})')


@spec
class Prism(_Extruded):
    """A smoothed polygon extruded along an axis: ``fd.Prism(vertices=..., height=0.5)``.

    ``vertices`` are ``(u, v)`` offsets from ``center`` in the right-handed pair across
    ``axis`` -- ``(x, y)`` for ``'z'``, ``(y, z)`` for ``'x'``, ``(z, x)`` for ``'y'`` --
    counter-clockwise, simple, absolute lengths; a clockwise, self-crossing or two-vertex
    outline is refused here, at construction, by the kernel's own check. A non-convex outline
    is legal. ``angle`` (degrees) turns the outline counter-clockwise about ``+axis`` through
    ``center``. Fill and diagonal moment are exact at every edge and corner
    (``src/wavebit/fdtd/_shapes``' :class:`~wavebit.fdtd._shapes.Prism`). With ``range=`` and a
    concrete prebuilt plan, vertices, height and centre may be differentiated under
    :class:`Sphere`'s isolated-body restrictions. The vertex count, axis and angle stay
    static. The full rotated outline and both caps must stay inside the range.
    Runtime checks preserve a finite, simple, counter-clockwise outline and the existing
    convex-hull-width floor; that floor does not resolve a concave outline's thin arms.
    Both convex and concave ranged outlines use the general clipping capacity so that
    changing convexity cannot change the compiled allocation. Without a range all
    geometry stays static. Derivatives belong to the existing discrete operator,
    including its nearest-surface cross-term approximation and its nondifferentiable ties.
    """

    vertices: tuple
    height: float
    axis: str = 'z'
    center: tuple = (0.0, 0.0, 0.0)
    angle: float = 0.0
    range: object = None

    _label = 'prism'

    def __post_init__(self):
        from wavebit.fdtd._shapes._bodies import polygon_shape
        object.__setattr__(self, 'axis', _extrusion_axis('Prism', self.axis))
        vertices = self.vertices
        if hasattr(vertices, 'tolist') and len(getattr(vertices, 'shape', ())) == 2:
            vertices = tuple(tuple(vertices[i, j] for j in range(vertices.shape[1]))
                             for i in range(vertices.shape[0]))
        if not (isinstance(vertices, (tuple, list))
                and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in vertices)):
            raise ValueError(f'Prism(vertices={vertices!r}): a tuple or list of (u, v) pairs, '
                             f'or an (nv, 2) array.')
        # Indexing keeps derivative leaves, unlike tolist(), but also keeps scalar
        # dtypes: float(np.bool_) or float(a real-valued complex tensor) would pass.
        if any(kind in str(getattr(v, 'dtype', '')) for p in vertices for v in p
               for kind in ('bool', 'complex')):
            raise ValueError('Prism(vertices=...): real coordinates, not boolean or complex values.')
        coordinate = _finite if self.range is None else _ranged_coordinate
        vertices = tuple((coordinate('Prism', f'vertices[{i},0]', u),
                          coordinate('Prism', f'vertices[{i},1]', v))
                         for i, (u, v) in enumerate(vertices))
        if len(vertices) < 3:
            raise ValueError('Prism(vertices=...): a polygon has at least three vertices.')
        if not any(_shape_traced(v) for p in vertices for v in p):
            polygon_shape('Prism', vertices)
        object.__setattr__(self, 'vertices', vertices)
        length = _length if self.range is None else _ranged_length
        object.__setattr__(self, 'height', length('Prism', 'height', self.height))
        object.__setattr__(self, 'center', _triple('Prism', 'center', self.center, coordinate))
        object.__setattr__(self, 'angle', _finite('Prism', 'angle', self.angle))
        if self.range is not None:
            a = AXES.index(self.axis)
            halves = tuple(self.height / 2 if k == a else 0. for k in range(3))
            for point in self._range_points():
                object.__setattr__(self, 'range', _contained_range(
                    'Prism', self.range, point, halves))

    def _range_points(self):
        """Rotated section vertices at the axial centre, without host extrema."""
        u, v = _TRANSVERSE[AXES.index(self.axis)]
        for x, y in _turned(self.vertices, self.angle):
            point = list(self.center)
            point[u], point[v] = point[u] + x, point[v] + y
            yield tuple(point)

    def _section(self):
        turned = _turned(self.vertices, self.angle)
        return ((min(p[0] for p in turned), max(p[0] for p in turned)),
                (min(p[1] for p in turned), max(p[1] for p in turned)))

    def _feature(self, grid):
        """The width of the outline's convex hull, which C5's cells-across door reads.

        A non-convex outline's thinnest arm can be narrower -- 2.83x for study S2b's L-shaped
        outline -- and whether its error follows the arm or the hull was not measured, so the
        door can pass a prism with an arm under four cells across (an open item of S2b).
        """
        return _hull_width(self.vertices)

    def volume(self):
        """The shoelace area of the outline times the height (:meth:`Sphere.volume`)."""
        v = self.vertices
        area = 0.5 * sum(u0 * v1 - u1 * v0 for (u0, v0), (u1, v1) in zip(v, v[1:] + v[:1]))
        return abs(area) * self.height

    def body(self):
        """The ``_shapes`` prism: the exact polygon clip times the axial overlap."""
        from wavebit.fdtd._shapes import Prism as _Extruded_polygon
        vertices = self.vertices
        typed = tuple(v for p in vertices for v in p if _shape_traced(v))
        if typed:
            from wavebit._common.backend import array_namespace
            from wavebit.fdtd._shapes._bodies import _as
            xp = array_namespace(*typed)
            vertices = xp.stack([_as(xp, p, typed[0]) for p in vertices])
        body = _Extruded_polygon(vertices=vertices, height=self.height, axis=self.axis,
                                 center=self.center, angle=math.radians(self.angle))
        if self.range is not None:
            # The general capacity is safe for every simple outline with this vertex
            # count, including a sweep that changes between convex and concave.
            body._convex = body.convex = False
        return body

    def describe(self):
        turn = f', angle={self.angle}' if self.angle else ''
        return (f'Prism(vertices={self.vertices}, height={self.height}, axis={self.axis!r}, '
                f'center={self.center}{turn})')


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
