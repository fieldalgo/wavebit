"""Gate G0: what a shape is asked, and by whom -- one test per caller, one mutant each.

``geometry.cell_box(shape, grid)`` was one function answering three questions at once (the
module docstring of ``wavebit/fd/geometry.py`` lists the nine call sites and sorts them). On a
cell-snapped ``Slab`` or ``Box`` the three answers coincide, so nothing here could have gone
wrong; on the first shape whose edge is not a cell boundary they part company, and a caller
reading the wrong one is **silent**. This file makes them part company now, before any such
shape exists in the package, by handing the callers shapes that answer the protocol and
nothing else.

They are :class:`_Pebble`, a round thing whose *support* (the cells it may touch) is strictly
larger than its *filled cells* (the ones it owns outright) and which is uniform along no axis
at all; and :class:`_Wafer`, a cell-snapped **layer** with none of :class:`Slab`'s own
attributes, for the three readers that branch on ``layer_axis()``. Every test below is a
caller asking one of the protocol's questions, with the answer that caller must get.

The mutants, each written first, edited into the package, run and watched to fail (the
surviving-mutant rule). Counts are out of the 30 tests in this file, measured 2026-09-20,
each mutant applied to the package on its own and reverted from a pristine copy taken off
disk before the next.

    mutant, edited into the package                  red here (30 tests, exit 1 each)
    -------------------------------------------------------------------------------------
    THE THREE QUESTIONS, ASKED OF THE WRONG CALLER
    ``check_absorbing_layers`` reads the support     2: ..._an_absorbing_layer_asks_whether
      (``k0 <= layer[0] and k1 >= layer[1]``)          _the_shape_is_uniform_not_its_box,
      instead of ``uniform_along``                     ..._a_refusal_names_the_shape... (its
                                                       refusal is this one). A pebble ending
                                                       inside a PML is waved through, and
                                                       that case is measured at 1.9e-1
                                                       against 6.0e-4 -- 317x
                                                       (``tests/test_fd_absorber.py``)
    ``painters`` reads ``support`` instead of       1: ..._the_lowering_paints_the_cells_a
      ``filled_cells``                                 _shape_fills_not_the_ones_it_touches
                                                       -- the pebble's whole bounding box is
                                                       painted with its material, corners
                                                       and all
    ``farfield_checks`` reads ``filled_cells``       1: ..._a_clearance_question_asks_the
      instead of ``support``                           _support_not_the_filled_cells -- a
                                                       pebble clipping the box's surface is
                                                       declared strictly inside it
    ``plan._pole_terms`` reads ``filled_cells``      1: ..._a_pole_terms_region_is_the
      instead of ``support``                           _support... -- the region drops from
                                                       (4, 12)^3 to (6, 10)^3, so the pole
                                                       stops driving 448 of the 512 cells
                                                       its own shape reaches
    ``grazing._layer_media`` reads                   1: ..._the_media_in_an_absorbing_layer
      ``filled_cells`` instead of ``support``          _are_every_shape_that_may_reach_it --
                                                       ``structures[0]`` drops out of the
                                                       layer's media and the grazing rule
                                                       judges the face against n = 1 where a
                                                       ball of n = 1.5 clips it
    ``sources._structure_cells`` reads               1: ..._a_tfsf_scatterer_is_judged_by
      ``filled_cells`` instead of ``support``          _its_support_not_by_the_cube_inside
                                                       _it -- a ball on the TFSF surface is
                                                       declared strictly inside, with a cell
                                                       to spare
    ``_post.medium_at`` reads ``filled_cells``       1: ..._the_medium_at_a_plane_sees_every
      instead of ``support``                           _shape_that_may_reach_the_layer -- the
                                                       ball is invisible in the layer it
                                                       clips and the plane reads vacuum
    ``medium_at``'s ``full`` test restored to the    1: ..._a_plane_wave_layer_is_one_medium
      bounding-box comparison                          _only_if_a_shape_is_uniform_across_it

    THE DISPATCHES A SHAPE NOW ANSWERS ITSELF
    the deleted ``describe(shape)`` restored with    3: ..._a_refusal_names_the_shape_it_is
      its ``isinstance(shape, Slab) ... else``         _about_in_the_shapes_own_words (the
      Box branch, and ``checks.py`` routed             refusal calls a pebble ``Box(size=
      through it                                       (0.36, 0.36, 0.36), center=(0.0, 0.0,
                                                       -0.22))``), ..._an_overlap_question...,
                                                       ..._the_cell_box_family_is_gone...
    ``Slab.bounded_axes`` derived from               1: ..._a_slab_names_its_own_axis_even
      ``uniform_along`` rather than answered by        _where_its_span_fills_the_box -- a
      the shape                                        full-box slab's coefficient arrays
                                                       lose their long axis, which would
                                                       move every frozen skeleton
    ``medium_at``'s refusal back to the hardcoded    2: ..._a_plane_wave_layer_is_one_medium
      "a Box structure cuts it"                        ..., ..._the_medium_at_a_plane_sees...
                                                       -- a sentence about a box, in a
                                                       message about a ball

    LAYER_AXIS'S TWO FORMER PREREQUISITES, RESTORED
    the two refusal messages read                    2: ..._a_layer_shape_needs_no_slab
      ``structure.shape.axis`` again                   _attribute_to_reach_a_plane_waves
                                                       _refusal, ..._to_reach_the_tfsf
                                                       _crossing_refusal -- both with
                                                       ``AttributeError: '_Wafer' object has
                                                       no attribute 'axis'`` out of
                                                       ``wavebit/fd/_lower/sources.py``
    ``_tfsf_driver``'s owner scan reads              1: ..._the_tfsf_drivers_owner_scan_asks
      ``s.shape.cells(grid)`` again                    _uniform_along_not_a_slabs_span --
                                                       ``AttributeError: '_Wafer' object has
                                                       no attribute 'cells'``

    THE TWO ANSWERS NOTHING IN THE PACKAGE READS YET
    ``_proximity_notes`` drops the                   1: ..._a_slab_filling_the_box_on_the
      ``layer_axis() != fa`` half of its skip          _absorbing_axis_still_gets_the
                                                       _proximity_note -- both z faces' notes
                                                       vanish, and nothing else in the fast
                                                       tier notices
    ``min_extent`` ignores the grid (``min(size)``   1: ..._min_extent_is_the_shortest_side
      and ``hi - lo``, as it did before the            _the_shape_bounds_in_this_grid -- a box
      signature took one)                              clipped to 0.2 on x and y answers 0.3,
                                                       a side that is not in the grid, and
                                                       C5's refusal would quote it
    the ``Structure`` door takes a bare              1: ..._the_structure_door_is_fd_Shape
      ``fd.Shape()`` again, and ``Shape`` goes         _and_the_bare_protocol_is_not_a_shape
      back to being unexported

The bit-identity half of G0 is not here: it is ``git archive`` before and after plus a
byte-compare of all 23 golden scenes' lowered arrays and skeletons, recorded in the commit
message (memory ``a-pure-move-is-provable``). What is here is the half a frozen archive
cannot see -- that each caller now asks the question it means.
"""
import math
import warnings

import pytest

import wavebit.fd as fd
from wavebit.fd._kernel.grid import Grid
from wavebit.fd._lower.checks import check_overlaps
from wavebit.fd._lower.coefficients import painters, varying_axes
from wavebit.fd._lower.monitors import farfield_checks
from wavebit.fd._post import medium_at
from wavebit.fd.geometry import AXES, Box, Shape, Slab

DX = 0.05


# ---------------------------------------------------------------------------------------------
# A shape that answers the protocol and nothing else
# ---------------------------------------------------------------------------------------------

class _Pebble(Shape):
    """A round shape spelled with the protocol alone: stage 4's first one, borrowed early.

    It is a ball of radius ``radius`` at ``center``, and it is here only so that the three
    questions have three different answers:

    * :meth:`support` is the outward-rounded bounding box -- every cell the ball may touch;
    * :meth:`filled_cells` is the inward-rounded box of the **inscribed cube** (half-side
      ``radius / sqrt(3)``) -- cells the ball certainly owns outright. Strictly smaller.
    * :meth:`uniform_along` is ``False`` everywhere: a sphere's surface curves, so any range
      of cell layers its support reaches holds a piece of that surface.

    It is deliberately *not* what stage 4 will ship (there is no fill fraction and no normal
    here, which is the whole of C1 and C2). It is the smallest object that makes a caller's
    choice of question observable, and it carries ``size`` and ``center`` attributes so that
    the deleted ``else: assume it is a Box`` branch would read it **silently** rather than
    raising ``AttributeError`` -- a mutant that crashes proves nothing about a silent defect.
    """

    def __init__(self, radius, center=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        self.center = tuple(float(c) for c in center)

    @property
    def size(self):
        """The bounding cube's sides -- the attribute the old Box branch would have read."""
        return (2 * self.radius,) * 3

    def _cells(self, grid, half, outward):
        lo_round, hi_round = (math.floor, math.ceil) if outward else (math.ceil, math.floor)
        out = []
        for a in range(3):
            k0 = lo_round((self.center[a] - half - grid.low[a]) / grid.dx)
            k1 = hi_round((self.center[a] + half - grid.low[a]) / grid.dx)
            k0 = min(max(int(k0), 0), grid.n[a])
            k1 = min(max(int(k1), 0), grid.n[a])
            out.append((k0, max(k1, k0)))
        return tuple(out)

    def support(self, grid):
        return self._cells(grid, self.radius, outward=True)

    def filled_cells(self, grid):
        return self._cells(grid, self.radius / math.sqrt(3.0), outward=False)

    def uniform_along(self, axis, cells, grid):
        return False

    def body(self):
        """``None``: this pebble is cell-snapped, so the lowering paints it with a mask.

        Stage 4's C2 added this question to the protocol, and the pebble's answer keeps it on
        exactly the path it was written to exercise -- the three questions it separates are
        the cell-snapped ones, and a fill fraction would replace two of them rather than
        sharpen them.
        """
        return None

    def bounded_axes(self, grid):
        return (0, 1, 2)

    def describe(self):
        return f'Pebble(radius={self.radius}, center={self.center})'

    def span_rows(self, grid):
        box = self.support(grid)
        return tuple((f'pebble.{axis}',
                      self.center[a] - self.radius, self.center[a] + self.radius,
                      grid.low[a] + box[a][0] * grid.dx, grid.low[a] + box[a][1] * grid.dx)
                     for a, axis in enumerate(AXES))

    def min_extent(self, grid):
        return 2 * self.radius


class _Wafer(Shape):
    """A **layer** spelled with the protocol alone: no ``axis``, no ``cells(grid)``.

    :class:`Slab` is the package's only layer, and until this commit three readers in
    ``wavebit/fd/_lower/sources.py`` reached past :meth:`Shape.layer_axis` for Slab-only
    attributes -- ``shape.axis`` in two refusal messages and ``shape.cells(grid)`` in the
    TFSF driver's ownership scan. A shape that answered the whole protocol and declared itself a layer
    therefore died there with a bare ``AttributeError``: the guess-from-the-type failure the
    class was written to replace, one method later.

    So this wafer has neither attribute, on purpose. It is a cell-snapped layer across
    ``across``, which is what :class:`Slab` is, spelled in the eight methods and nothing else.
    Unlike :class:`_Pebble` it does **not** carry the attributes a mutant would read: the
    point here is the crash, and a mutant that reads a wrong value silently is a different
    defect with its own pebble.
    """

    def __init__(self, span, across):
        self.span = tuple(float(v) for v in span)
        self.across = int(across)

    def _cells(self, grid):
        a = self.across
        k0 = int(round((self.span[0] - grid.low[a]) / grid.dx))
        k1 = int(round((self.span[1] - grid.low[a]) / grid.dx))
        return (min(max(k0, 0), grid.n[a]), min(max(k1, 0), grid.n[a]))

    def support(self, grid):
        out = [(0, grid.n[a]) for a in range(3)]
        out[self.across] = self._cells(grid)
        return tuple(out)

    def filled_cells(self, grid):
        return self.support(grid)                       # cell-snapped: it fills it outright

    def body(self):
        return None

    def uniform_along(self, axis, cells, grid):
        if axis != self.across:
            return True
        k0, k1 = self._cells(grid)
        return k0 <= cells[0] and k1 >= cells[1]

    def layer_axis(self):
        return self.across

    def bounded_axes(self, grid):
        return (self.across,)

    def describe(self):
        return f'Wafer({AXES[self.across]}={self.span})'

    def span_rows(self, grid):
        a, (k0, k1) = self.across, self._cells(grid)
        return ((f'wafer.{AXES[a]}',) + self.span
                + (grid.low[a] + k0 * grid.dx, grid.low[a] + k1 * grid.dx),)

    def min_extent(self, grid):
        a = self.across
        return min(self.span[1], grid.high[a]) - max(self.span[0], grid.low[a])


class _Mute(Shape):
    """A shape that answers nothing: every protocol method must refuse it by name."""


def _grid(n=(16, 16, 16), dx=DX):
    return Grid(n=n, dx=dx, low=tuple(-k * dx / 2 for k in n))


def _band(df=0.1):
    """A band whose shortest wavelength still resolves eps = 2.25 at ``DX`` (9.7 cells)."""
    return fd.Band(wavelength=1.0, df=df)


def _scene(structures=(), boundaries=None, size=16 * DX, steps=6, sources=()):
    return fd.Scene(
        domain=fd.Domain(size=(size,) * 3),
        boundaries=boundaries or fd.Boundaries.open(
            fd.PML(cells=4, wavelength=1.0, allow_thin=True)),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=steps),
        structures=structures, sources=sources)


def _quiet(fn, *args, **kwargs):
    """Run ``fn``, ignoring the diagnostics warnings -- this file judges refusals, not notes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('method, args', [
    ('support', ('grid',)), ('filled_cells', ('grid',)), ('body', ()),
    ('uniform_along', (0, (0, 4), 'grid')), ('bounded_axes', ('grid',)),
    ('describe', ()), ('span_rows', ('grid',)), ('min_extent', ('grid',)),
])
def test_a_shape_that_does_not_answer_a_question_is_refused_by_name(method, args):
    """No default, and no "else assume it is a Box": the refusal names the method and the type.

    This is the door the old dispatch did not have. ``describe(sphere)`` fell through to
    ``f'Box(size={shape.size}, center={shape.center})'``, ``bounded_axes(sphere, grid)`` to
    ``shape.cells(grid)``, ``cell_box(sphere, grid)`` to ``shape.axis``: three different
    guesses about what a shape it had never seen must be.
    """
    grid = _grid()
    args = tuple(grid if a == 'grid' else a for a in args)
    with pytest.raises(NotImplementedError) as e:
        getattr(_Mute(), method)(*args)
    assert method in str(e.value) and '_Mute' in str(e.value)


def test_layer_axis_is_the_one_question_with_a_default_and_its_default_is_not_a_layer():
    """``None`` is the truthful answer for anything that is not a :class:`Slab`, and it is safe.

    Every reader of ``layer_axis`` takes its conservative branch on ``None``: the shape is a
    scatterer, so it is kept out of a TFSF column's one-dimensional background and it is not
    waved past an absorbing face. A shape that forgets to answer therefore errs toward the
    refusal, which is why this one method may have a default at all.
    """
    assert _Mute().layer_axis() is None
    assert _Pebble(0.1).layer_axis() is None
    assert Box(size=(0.2, 0.2, 0.2)).layer_axis() is None
    assert Slab(z=(-0.1, 0.1)).layer_axis() == 2
    assert Slab(x=(-0.1, 0.1)).layer_axis() == 0


def test_the_cell_box_family_is_gone_from_geometry():
    """The retired names are deleted, not aliased: nothing in ``wavebit.fd`` is released."""
    import wavebit.fd.geometry as geometry
    for name in ('cell_box', 'slab_axis', 'bounded_axes', 'describe', 'span_rows'):
        assert not hasattr(geometry, name), f'geometry.{name} is still there'
    assert 'Shape' in geometry.__all__


# ---------------------------------------------------------------------------------------------
# Slab and Box: today's answers, pinned
# ---------------------------------------------------------------------------------------------

def test_a_slab_answers_the_protocol_with_the_layer_rules_it_always_had():
    grid = _grid()
    slab = Slab(z=(-0.1, 0.1))                       # cells 6..10 of 16, dx = 0.05
    assert slab.support(grid) == ((0, 16), (0, 16), (6, 10))
    assert slab.filled_cells(grid) == slab.support(grid)
    assert slab.bounded_axes(grid) == (2,)
    assert slab.describe() == 'z=(-0.1, 0.1)'
    row, = slab.span_rows(grid)
    assert row[0] == 'z' and row[1:] == pytest.approx((-0.1, 0.1, -0.1, 0.1))
    assert slab.min_extent(grid) == pytest.approx(0.2)
    # Uniform across the two axes it runs the whole box along, and along its own axis only
    # where its cells cover the range asked about.
    assert slab.uniform_along(0, (0, 16), grid) and slab.uniform_along(1, (3, 5), grid)
    assert slab.uniform_along(2, (7, 9), grid)
    assert not slab.uniform_along(2, (0, 4), grid)
    assert not slab.uniform_along(2, (4, 8), grid)


def test_a_box_answers_the_protocol_with_the_snapped_rules_it_always_had():
    grid = _grid()
    box = Box(size=(0.2, 0.2, 40.0), center=(0.0, 0.0, 0.0))   # clipped through z
    assert box.support(grid) == ((6, 10), (6, 10), (0, 16))
    assert box.filled_cells(grid) == box.support(grid)
    assert box.bounded_axes(grid) == (0, 1)          # z is the whole box, so it cuts nothing
    assert box.describe() == 'Box(size=(0.2, 0.2, 40.0), center=(0.0, 0.0, 0.0))'
    assert [row[0] for row in box.span_rows(grid)] == ['box.x', 'box.y', 'box.z']
    assert box.min_extent(grid) == pytest.approx(0.2)
    assert box.uniform_along(2, (0, 4), grid)        # reaches through the z faces
    assert not box.uniform_along(0, (0, 4), grid)


def test_a_slab_names_its_own_axis_even_where_its_span_fills_the_box():
    """``bounded_axes`` is the shape's own answer and not derived from ``uniform_along``.

    A slab spanning the whole box is uniform along its own axis by the ``uniform_along``
    rule -- there is no face of it inside the grid -- so the derived set would be empty and
    the coefficient arrays would collapse to one cell along z. They do not: stage 1 gave the
    slab its axis unconditionally, those array shapes are in every frozen skeleton, and a
    scene that meant a uniform box meant ``Scene(background=)``. The mutant is the derivation.
    """
    grid = _grid()
    whole = Slab(z=(grid.low[2], grid.low[2] + grid.n[2] * grid.dx))
    assert whole.support(grid)[2] == (0, 16)
    assert whole.uniform_along(2, (0, 16), grid)     # no face of it anywhere in the grid
    assert whole.bounded_axes(grid) == (2,)          # and yet: its own axis, always
    scene = _scene((fd.Structure(whole, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()))
    assert 2 in varying_axes(scene, _quiet(fd.plan, scene))


# ---------------------------------------------------------------------------------------------
# One caller, one question
# ---------------------------------------------------------------------------------------------

def test_an_absorbing_layer_asks_whether_the_shape_is_uniform_not_its_box():
    """``check_absorbing_layers``: a bounding box that spans the layer proves nothing.

    A structure that *ends* inside an absorbing layer is a mirror the layer cannot undo --
    1.9e-1 reflection against 6.0e-4 running it through, 317x, measured in
    ``tests/test_fd_absorber.py``. The old test was ``k0 <= layer[0] and k1 >= layer[1]`` on
    the cell box, which a pebble's bounding box satisfies while the pebble's own surface
    curves through the middle of the layer. Restore that test and this passes silently.
    """
    pebble = _Pebble(0.18, center=(0.0, 0.0, -0.22))     # its box reaches through the z- PML
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),))
    grid = _grid()
    assert pebble.support(grid)[2][0] == 0               # the bounding box does span the layer
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert 'ends inside the PML' in str(e.value)


def test_a_refusal_names_the_shape_it_is_about_in_the_shapes_own_words():
    """``describe`` is the shape's, so a refusal cannot print one shape's spelling for another.

    The mutant is the deleted module function restored: ``isinstance(shape, Slab)`` and
    otherwise ``f'Box(size={shape.size}, center={shape.center})'``. ``_Pebble`` carries both
    attributes, so under that mutant the refusal below reads ``Box(size=(0.36, 0.36, 0.36),
    center=(0.0, 0.0, -0.22))`` -- a sentence about a box, in a message about a ball.
    """
    pebble = _Pebble(0.18, center=(0.0, 0.0, -0.22))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),))
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert 'Pebble(radius=0.18' in str(e.value)
    assert 'Box(size=' not in str(e.value)


def test_the_lowering_paints_the_cells_a_shape_fills_not_the_ones_it_touches():
    """``painters`` is the one reader of ``filled_cells``, and stage 4's C2 kept it that way.

    Painting a cell-snapped mask over a cell the shape only clips would give that whole cell
    the shape's material -- the staircase, but silently and at the shape's *outer* radius. The
    mutant is ``support`` here; it shows up as the pebble's corner cells painted. C2 replaced
    the owner index with one painter per structure (a mask for a cell-snapped shape, the
    per-component fill for a smoothed one) and this question did not move: a shape that
    answers ``body()`` with ``None`` still gets exactly this box and exactly this ``where``.
    """
    pebble = _Pebble(0.18, center=(0.0, 0.0, 0.0))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()))
    p = _quiet(fd.plan, scene)
    grid = p.grid
    painted = painters(scene, p, {0, 1, 2}, {})
    assert len(painted) == 1
    mask = painted[0].mask
    filled, support = pebble.filled_cells(grid), pebble.support(grid)
    assert filled != support, 'the two questions must differ or this test proves nothing'
    (i0, i1), (j0, j1), (k0, k1) = filled
    assert mask[k0:k1, j0:j1, i0:i1].all()
    # Nothing outside the filled box is painted: in particular not the corner of the support,
    # which is where the ball is not.
    assert int(mask.sum()) == (i1 - i0) * (j1 - j0) * (k1 - k0)
    assert not mask[support[2][0], support[1][0], support[0][0]]


def test_a_clearance_question_asks_the_support_not_the_filled_cells():
    """``farfield_checks`` and ``_check_scatterer``: a cell the shape merely clips is enough.

    The transform assumes one uniform medium outside its box, and the TFSF corrections read
    an incident field computed in a 1-D column that knows only the background. A cell the
    shape touches at all breaks both, so the question is the support. The mutant is
    ``filled_cells``, under which the pebble below is declared strictly inside.
    """
    pebble = _Pebble(0.18, center=(0.0, 0.0, 0.0))
    grid = _grid(n=(24, 24, 24))
    monitor = fd.FarField('ff', box=fd.Box(size=(0.4, 0.4, 0.4)), wavelengths=(1.0,))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),), size=24 * DX)
    # The far-field box's nodes leave the inscribed cube clear but not the support.
    assert pebble.filled_cells(grid) != pebble.support(grid)
    with pytest.raises(ValueError) as e:
        farfield_checks(monitor, scene, grid)
    assert 'not\nstrictly inside the box'.replace('\n', ' ') in ' '.join(str(e.value).split())


def test_a_plane_wave_layer_is_one_medium_only_if_a_shape_is_uniform_across_it():
    """``medium_at``: "fills the layer" is uniformity across both transverse axes.

    An order, and a source normalisation, are plane waves in a uniform medium. The old test
    compared the structure's cell box with ``(0, n)`` on the two transverse axes; a pebble
    whose support happens to reach both ends of a small box would pass it while leaving the
    layer two media. The question is ``uniform_along``, and a pebble is uniform along nothing.
    """
    grid = _grid(n=(8, 8, 16), dx=DX)
    pebble = _Pebble(0.25, center=(0.0, 0.0, 0.0))       # its box spans x and y entirely
    assert pebble.support(grid)[0] == (0, 8) and pebble.support(grid)[1] == (0, 8)
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),))
    with pytest.raises(ValueError) as e:
        medium_at(scene, grid, 2, range(8, 9), 'Diffraction(test)')
    assert 'is not one medium' in str(e.value)
    # And it names the structure that cut the layer, in that shape's own words: the message
    # used to read "a Box structure cuts it" whatever the shape was.
    assert 'structures[0]' in str(e.value) and 'Pebble(radius=0.25' in str(e.value)
    assert 'a Box structure' not in str(e.value)


def test_an_overlap_question_asks_the_support_because_a_missed_overlap_is_the_silent_one():
    """``check_overlaps``: the dispersive-overlap refusal reads every cell either may touch.

    An overlap the door misses is answered wrongly and quietly (the pole term keeps driving
    cells its structure no longer owns -- 72% of the peak, ``check_overlaps``' own docstring),
    so the conservative question is the right one here even though it can refuse a pair whose
    materials never actually meet.
    """
    grid = _grid()
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=4.0, gamma=1.0),))
    pebble = _Pebble(0.1, center=(0.0, 0.0, 0.14))
    slab = fd.Slab(z=(-0.1, 0.1))
    assert pebble.filled_cells(grid)[2][0] >= slab.support(grid)[2][1]   # filled: clear of it
    assert pebble.support(grid)[2][0] < slab.support(grid)[2][1]         # support: they meet
    scene = _scene((fd.Structure(slab, metal), fd.Structure(pebble, fd.Material(eps=2.25))))
    with pytest.raises(ValueError) as e:
        check_overlaps(scene, grid)
    assert 'share cells' in str(e.value) and 'Pebble(radius=0.1' in str(e.value)


def test_the_report_rows_are_the_shapes_own_and_reach_the_plan():
    """``span_rows`` is on the shape because its labels are in ``Plan.spans`` and the skeleton."""
    pebble = _Pebble(0.1, center=(0.0, 0.0, 0.0))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()))
    p = _quiet(fd.plan, scene)
    assert [row[0] for row in p.spans] == ['pebble.x', 'pebble.y', 'pebble.z']


def test_a_shape_that_answers_the_protocol_needs_nothing_else_to_reach_the_kernel():
    """The protocol is the whole interface: a scene of pebbles plans, lowers and steps.

    Stage 4 puts its geometry kernel and its occupancy stage behind these same methods, so
    this is the pin that says the seam is the protocol and not a type check somewhere
    downstream.
    """
    pebble = _Pebble(0.1, center=(0.0, 0.0, 0.0))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()), steps=4)
    res = _quiet(fd.run, scene)
    assert res is not None
    assert _quiet(fd.plan, scene).skeleton == _quiet(fd.plan, scene).skeleton


# ---------------------------------------------------------------------------------------------
# The four callers the first round left unpinned
# ---------------------------------------------------------------------------------------------

def test_a_pole_terms_region_is_the_support_because_the_poles_state_lives_there():
    """``plan._pole_terms``: the region is where the pole's state array and ``drive`` are.

    So it has to hold every cell the shape reaches, not only the ones it fills outright: a
    pole that stopped driving the cells its shape merely clips is a metal with a hollow shell,
    and nothing says so. On the pebble below the two questions are 8^3 cells and 4^3, so the
    mutant (``filled_cells`` here) leaves 448 of the 512 cells the shape reaches undriven and
    returns a number shaped like an answer. What *fraction* of each of those cells the shape
    fills is the coefficients' business (C2), not the region's.
    """
    grid = _grid()
    pebble = _Pebble(0.18, center=(0.0, 0.0, 0.0))
    metal = fd.Dispersive(eps_inf=1.0, poles=(fd.Drude(omega_p=4.0, gamma=1.0),))
    scene = _scene((fd.Structure(pebble, metal),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()))
    support, filled = pebble.support(grid), pebble.filled_cells(grid)
    assert support != filled, 'the two questions must differ or this test proves nothing'
    term, = _quiet(fd.plan, scene).terms
    assert (term.region.x, term.region.y, term.region.z) == support
    assert term.region.shape == (8, 8, 8)               # 512 cells, not the inscribed 64


def test_the_media_in_an_absorbing_layer_are_every_shape_that_may_reach_it():
    """``grazing._layer_media``: the grazing rule reads the worst index *in* the layer.

    A cell of the layer the shape merely clips already carries that material's index, and the
    layer's return is judged against the largest index anywhere in it, so the question is the
    support. The second half of the same line is ``uniform_along``: the background drops out of
    the list only where some structure fills the layer's whole cross-section, which a curved
    shape never does. Asserting the exact list pins both -- ``filled_cells`` drops
    ``structures[0]`` and a ``uniform_along -> True`` mutant drops ``the background``.

    The scene is one ``check_absorbing_layers`` refuses (a pebble whose surface ends inside a
    PML is the 317x mirror), which is why this calls ``_layer_media`` directly: with today's
    shapes the two answers part company only there. The legal case arrives with stage 4's fill
    fractions -- a taper that runs *through* a layer while filling no whole cell in it -- and
    this is the pin that says which question it will be asked then.
    """
    from wavebit.fd._lower.grazing import _layer_media
    grid = _grid()
    pebble = _Pebble(0.18, center=(0.0, 0.0, -0.05))
    assert pebble.support(grid)[2] == (3, 11)           # reaches cell 3, the layer's innermost
    assert pebble.filled_cells(grid)[2] == (5, 9)       # ... and fills none of the layer
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),))
    media = _layer_media(scene, grid, 2, '-', 4)
    assert [label for label, _ in media] == ['the background', 'structures[0]']
    assert [index for _, index in media] == [pytest.approx(1.0), pytest.approx(1.5)]


def test_a_tfsf_scatterer_is_judged_by_its_support_not_by_the_cube_inside_it():
    """``sources._structure_cells``, read by ``_check_scatterer``: one clipped cell is enough.

    The corrections read the incident field on the box's node planes and write the cells one
    outside them, and that field came from a 1-D column that knows only the background. A
    scatterer that touches those cells at all makes it wrong, so the question is the support.
    Under ``filled_cells`` the pebble below is declared strictly inside with a cell to spare
    -- its inscribed cube is (10, 14) against the box's 9..15 -- while the ball itself reaches
    cells 8..15 and sits on the surface.
    """
    grid = _grid(n=(24, 24, 24))
    pebble = _Pebble(0.18, center=(0.0, 0.0, 0.0))
    assert pebble.support(grid)[2] == (8, 16) and pebble.filled_cells(grid)[2] == (10, 14)
    tfsf = fd.TFSF('t', box=fd.Box(size=(0.3, 0.3, 0.3)), direction='-z', polarization='x',
                   band=_band())
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),), size=24 * DX,
                   sources=(tfsf,))
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert 'touches the surface of' in str(e.value)
    assert 'Pebble(radius=0.18' in str(e.value)


def test_the_medium_at_a_plane_sees_every_shape_that_may_reach_the_layer():
    """``_post.medium_at``'s other half: the cell layer's owners come from the support.

    The uniform half is the test above it; this is the one that decides whether the shape is
    *looked at* in layer ``k`` at all. Under ``filled_cells`` the pebble below is invisible at
    cell layer 3 -- it fills 5..9 -- and ``medium_at`` hands back the background as if the
    layer were empty, so an order or a normalisation is computed in vacuum where a ball is.
    """
    grid = _grid()
    pebble = _Pebble(0.18, center=(0.0, 0.0, -0.05))
    assert pebble.support(grid)[2] == (3, 11) and pebble.filled_cells(grid)[2] == (5, 9)
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),))
    with pytest.raises(ValueError) as e:
        medium_at(scene, grid, 2, range(3, 4), 'Diffraction(test)')
    assert 'the cell layer 3 across z is not one medium' in str(e.value)
    assert 'structures[0]' in str(e.value) and 'Pebble(radius=0.18' in str(e.value)


# ---------------------------------------------------------------------------------------------
# A layer needs nothing but the protocol
# ---------------------------------------------------------------------------------------------

def test_a_layer_shape_needs_no_slab_attribute_to_reach_a_plane_waves_refusal():
    """``sources._plane_wave_term``: the axis in the message is ``AXES[layer_axis()]``.

    It used to be ``structure.shape.axis``, a :class:`Slab` property that is not on the
    protocol, so this refusal -- about a layer that is not across the propagation axis -- was
    an ``AttributeError`` for any other layer shape. The wafer has no ``axis``, so restoring
    that read turns this test red with a crash inside ``_lower``.
    """
    wafer = _Wafer((0.0, 0.4), across=0)                # across x, cells 8..16 of 16
    assert wafer.layer_axis() == 0 and not hasattr(wafer, 'axis')
    pw = fd.PlaneWave('pw', z=0.1, direction='-z', polarization='s', band=_band())
    scene = _scene((fd.Structure(wafer, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(
                       x=fd.Periodic(), y=fd.Periodic(),
                       z=fd.PML(cells=4, wavelength=1.0, allow_thin=True)),
                   sources=(pw,))
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert 'a slab across x' in str(e.value) and 'not uniform across the' in str(e.value)


def test_a_layer_shape_needs_no_slab_attribute_to_reach_the_tfsf_crossing_refusal():
    """``sources._tfsf_term``: the same read, in the message about a layer crossing the box."""
    wafer = _Wafer((0.0, 0.6), across=0)                # across x, cells 12..24 of 24
    tfsf = fd.TFSF('t', box=fd.Box(size=(0.3, 0.3, 0.3)), direction='-z', polarization='x',
                   band=_band())
    scene = _scene((fd.Structure(wafer, fd.Material(eps=2.25)),), size=24 * DX,
                   sources=(tfsf,))
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert "crosses the faces of TFSF('t')" in str(e.value) and 'slab across x' in str(e.value)


def test_the_tfsf_drivers_owner_scan_asks_uniform_along_not_a_slabs_span():
    """``sources._tfsf_driver``: "this layer owns cell j" is ``uniform_along(P, (j, j+1))``.

    It used to be ``shape.cells(grid)``, a :class:`Slab` method that is not on the protocol,
    so a wafer filling the incident side crashed here instead of being the driver's medium.
    The two are the same inequality for a cell-snapped layer; they part company on a layer
    whose face falls *inside* cell ``j``, where ``uniform_along`` is False, the cell keeps
    whatever was under it and ``owners`` holds two -- the refusal, which is the right answer
    for a half-filled cell handed to a plane wave.
    """
    glass = fd.Material(eps=2.25)
    tfsf = fd.TFSF('t', box=fd.Box(size=(0.3, 0.3, 0.3)), direction='-z', polarization='x',
                   band=_band())
    # A wafer over the whole incident side (cells 15..24, the box's entry plane out to the z+
    # face): one medium, so the driver column is glass and the plan is built.
    whole = _Wafer((0.15, 0.6), across=2)
    assert whole.layer_axis() == 2 and not hasattr(whole, 'cells')
    scene = _scene((fd.Structure(whole, glass),), size=24 * DX, sources=(tfsf,))
    assert _quiet(fd.plan, scene) is not None
    # The same wafer stopping short of the entry plane leaves the background in there too.
    part = _Wafer((0.3, 0.6), across=2)
    scene = _scene((fd.Structure(part, glass),), size=24 * DX, sources=(tfsf,))
    with pytest.raises(ValueError) as e:
        _quiet(fd.plan, scene)
    assert 'is not one medium' in str(e.value) and 'the background' in str(e.value)


# ---------------------------------------------------------------------------------------------
# The three loose ends of the first round
# ---------------------------------------------------------------------------------------------

def test_a_slab_filling_the_box_on_the_absorbing_axis_still_gets_the_proximity_note():
    """``checks._proximity_notes``' ``layer_axis() != fa`` half, which is otherwise unreached.

    The skip is for a structure that runs *through* a transverse face with no face of its own
    in the way -- there is no clearance to measure it against. A layer **across this very
    axis** whose span happens to fill the box is uniform along it by the same rule and would
    be skipped too, but it is a different scene: it meant ``Scene(background=)``, its material
    is what the layer would have to be matched to rather than the background, and stage 1
    printed the note for it. Drop the second half of the guard and the two warnings below
    vanish with every other test still green -- which is what made this an unreached branch.
    """
    grid = Grid(n=(16, 16, 60), dx=DX, low=(-0.4, -0.4, -1.5))
    slab = Slab(z=(-1.5, 1.5))                          # the whole box, on the absorbing axis
    assert slab.uniform_along(2, (0, 60), grid) and slab.layer_axis() == 2
    scene = fd.Scene(
        domain=fd.Domain(size=(0.8, 0.8, 3.0)),
        boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                 z=fd.PML(cells=12, wavelength=1.0)),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=4),
        structures=(fd.Structure(slab, fd.Material(eps=2.25)),))
    with pytest.warns(fd.AbsorberProximityWarning) as record:
        fd.plan(scene)
    faces = sorted({m for w in record for m in ("'z-'", "'z+'") if m in str(w.message)})
    assert faces == ["'z+'", "'z-'"], [str(w.message) for w in record]


def test_min_extent_is_the_shortest_side_the_shape_bounds_in_this_grid():
    """The smallest feature C5's resolution refusal has to judge -- and only a real one.

    A side clipped by the box is not a feature: the shape runs out through that face, there is
    nothing of that size in the grid, and a refusal quoting it would be about a side that is
    not there. So the measure is over :meth:`Shape.bounded_axes`, which needs the grid, which
    is why the signature takes one -- ``min_extent()`` could only ever have returned the
    shortest side as spelled, clipped sides and all, contradicting its own docstring.
    """
    grid = Grid(n=(4, 4, 40), dx=DX, low=(-0.1, -0.1, -1.0))
    box = Box(size=(0.3, 0.3, 1.0), center=(0.0, 0.0, 0.0))
    assert box.support(grid) == ((0, 4), (0, 4), (10, 30))
    assert box.bounded_axes(grid) == (2,)               # x and y run out through the faces
    assert box.min_extent(grid) == pytest.approx(1.0)   # the bounded side, not the 0.3 stub
    slab = Slab(z=(-2.0, 2.0))                          # asked for 4.0 in a box 2.0 long
    assert slab.min_extent(grid) == pytest.approx(2.0)
    # A box clipped on every axis bounds nothing: it is the background, and there is no
    # feature size to refuse.
    assert Box(size=(9.0, 9.0, 9.0)).min_extent(grid) is None


def test_the_structure_door_is_fd_Shape_and_the_bare_protocol_is_not_a_shape():
    """The seam is exported, and named, rather than left to drift.

    ``isinstance(shape, Shape)`` is the only type check between a shape and the kernel, so a
    shape written outside this package runs -- that is what makes stage 4's geometry kernel a
    new file instead of an edit to nine call sites. It is therefore public as ``fd.Shape``,
    and the door says what it takes. ``fd.Shape()`` itself is refused: it is the protocol, and
    every question below it would raise at some later call site instead of here.
    """
    assert fd.Shape is Shape and 'Shape' in fd.__all__
    fd.Structure(_Pebble(0.1), fd.Material(eps=2.0))            # from outside the package: in
    with pytest.raises(TypeError) as e:
        fd.Structure(fd.Shape(), fd.Material(eps=2.0))
    assert 'the protocol' in str(e.value) and 'not a' in str(e.value)
    with pytest.raises(TypeError) as e:
        fd.Structure(fd.Plane(z=0.0), fd.Material(eps=2.0))     # a monitor region, not a shape
    assert 'subclassing fd.Shape' in str(e.value) and 'Plane' in str(e.value)
