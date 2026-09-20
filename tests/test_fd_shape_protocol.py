"""Gate G0: what a shape is asked, and by whom -- one test per question, one mutant each.

``geometry.cell_box(shape, grid)`` was one function answering three questions at once (the
module docstring of ``wavebit/fd/geometry.py`` lists the nine call sites and sorts them). On a
cell-snapped ``Slab`` or ``Box`` the three answers coincide, so nothing here could have gone
wrong; on the first shape whose edge is not a cell boundary they part company, and a caller
reading the wrong one is **silent**. This file makes them part company now, before any such
shape exists in the package, by handing the callers a shape that answers the protocol and
nothing else.

That shape is :class:`_Pebble`: a round thing whose *support* (the cells it may touch) is
strictly larger than its *filled cells* (the ones it owns outright), and which is uniform
along no axis at all. Every test below is a caller asking one of its questions, with the
answer that caller must get. The mutants, each written first, edited into the package, run and
watched to fail (the surviving-mutant rule):

    mutant, edited into the package                 red here (2026-09-20, 20 tests)
    ------------------------------------------------------------------------------------
    ``check_absorbing_layers`` reads the support    2 failed: ..._an_absorbing_layer_asks
      (``k0 <= layer[0] and k1 >= layer[1]``)         _whether_the_shape_is_uniform_not_its
      instead of ``uniform_along``                    _box, and ..._a_refusal_names_the
                                                      _shape... (its refusal is this one).
                                                      A pebble ending inside a PML is waved
                                                      through, and that case is measured at
                                                      1.9e-1 against 6.0e-4 -- 317x
                                                      (``tests/test_fd_absorber.py``)
    ``owner_map`` reads ``support`` instead of      1 failed: ..._the_lowering_paints_the
      ``filled_cells``                                _cells_a_shape_fills_not_the_ones_it
                                                      _touches -- the pebble's whole
                                                      bounding box is painted with its
                                                      material, corners and all
    ``farfield_checks`` reads ``filled_cells``      1 failed: ..._a_clearance_question_asks
      instead of ``support``                          _the_support_not_the_filled_cells --
                                                      a pebble clipping the box's surface
                                                      is declared strictly inside it
    the deleted ``describe(shape)`` restored with   3 failed: ..._a_refusal_names_the_shape
      its ``isinstance(shape, Slab) ... else``        _it_is_about_in_the_shapes_own_words
      Box branch, and ``checks.py`` routed            (the refusal calls a pebble
      through it                                      ``Box(size=(0.36, 0.36, 0.36),
                                                      center=(0.0, 0.0, -0.22))``),
                                                      ..._an_overlap_question..., and
                                                      ..._the_cell_box_family_is_gone...
    ``medium_at``'s ``full`` test restored to the   1 failed: ..._a_plane_wave_layer_is_one
      bounding-box comparison                         _medium_only_if_a_shape_is_uniform
                                                      _across_it
    ``Slab.bounded_axes`` derived from              1 failed: ..._a_slab_names_its_own_axis
      ``uniform_along`` rather than answered by       _even_where_its_span_fills_the_box --
      the shape                                       a full-box slab's coefficient arrays
                                                      lose their long axis, which would
                                                      move every frozen skeleton

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
from wavebit.fd._lower.coefficients import owner_map, varying_axes
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

    def min_extent(self):
        return 2 * self.radius


class _Mute(Shape):
    """A shape that answers nothing: every protocol method must refuse it by name."""


def _grid(n=(16, 16, 16), dx=DX):
    return Grid(n=n, dx=dx, low=tuple(-k * dx / 2 for k in n))


def _scene(structures=(), boundaries=None, size=16 * DX, steps=6):
    return fd.Scene(
        domain=fd.Domain(size=(size,) * 3),
        boundaries=boundaries or fd.Boundaries.open(
            fd.PML(cells=4, wavelength=1.0, allow_thin=True)),
        resolution=fd.Resolution(dx=DX), run=fd.Run(steps=steps),
        structures=structures)


def _quiet(fn, *args, **kwargs):
    """Run ``fn``, ignoring the diagnostics warnings -- this file judges refusals, not notes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('method, args', [
    ('support', ('grid',)), ('filled_cells', ('grid',)),
    ('uniform_along', (0, (0, 4), 'grid')), ('bounded_axes', ('grid',)),
    ('describe', ()), ('span_rows', ('grid',)), ('min_extent', ()),
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
    assert slab.min_extent() == pytest.approx(0.2)
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
    assert box.min_extent() == pytest.approx(0.2)
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
    """``owner_map`` is the one reader of ``filled_cells``, and the one C2 replaces.

    Painting an owner index over a cell the shape only clips would give that whole cell the
    shape's material -- the staircase, but silently and at the shape's *outer* radius. The
    mutant is ``support`` here; it shows up as the pebble's corner cells owned.
    """
    pebble = _Pebble(0.18, center=(0.0, 0.0, 0.0))
    scene = _scene((fd.Structure(pebble, fd.Material(eps=2.25)),),
                   boundaries=fd.Boundaries(x=fd.Periodic(), y=fd.Periodic(),
                                            z=fd.Periodic()))
    p = _quiet(fd.plan, scene)
    grid = p.grid
    owner = owner_map(scene, p, {0, 1, 2})
    filled, support = pebble.filled_cells(grid), pebble.support(grid)
    assert filled != support, 'the two questions must differ or this test proves nothing'
    (i0, i1), (j0, j1), (k0, k1) = filled
    assert (owner[k0:k1, j0:j1, i0:i1] == 0).all()
    # Nothing outside the filled box is owned: in particular not the corner of the support,
    # which is where the ball is not.
    assert (owner == 0).sum() == (i1 - i0) * (j1 - j0) * (k1 - k0)
    assert owner[support[2][0], support[1][0], support[0][0]] == -1


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
