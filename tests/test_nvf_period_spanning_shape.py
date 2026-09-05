"""A shape as long as the period has no face there — its images abut and the material runs on.

``ly = period`` is the **only** way to write a 1-D grating on the parametric path: ``ly`` above
the period is refused by name (``shape 0 (rectangle) is larger than the period and runs into
its own image``), and anything below it is a 2-D structure. The bundled
``examples/*.py`` writes its grating exactly that way. So the boundary value was the one value
in the family that was accepted -- and then read as two material interfaces, giving ``|n_y| = 1``
with |n_y| above 0.01 over 32.9% of the cell (above 0.99 over 8.5%, at a 512 grid), where the
raster builder for the same grating gives ``0.000e+00`` everywhere.

**Nothing raised and nothing looked wrong.** The permittivity coefficients are exact either way
(replace the field with a constant and ``epz_i`` is bit-identical), so ``R + T`` conserved to 1e-14
and the only symptom was slow convergence -- which reads as a property of the path rather than a
defect in it. It reached a published convergence study before it was caught.

**What the fix is.** :func:`~wavebit._core.parametric._shape_surface_normal` measures each face
distance to the nearest *image* of that face, folding by the repeat along the shape's own axis
(:func:`~wavebit._core.parametric._axis_lattice_periods`). At ``ly = Pv`` the ``+y`` and ``-y``
families become the same set of lines, so the two distances are equal everywhere and the two
opposite normals cancel identically. **The cancellation is an algebraic identity**, not a special
case, and a shape one ulp short of the period is carried continuously by the same fold.

The path does carry an exact equality on ``ly`` (``merged_v``, and ``merged_u`` for ``lx``), and
it is worth being precise about what that test is for: the cancellation alone leaves the
*surviving* pair's weights holding a factor that vanishes on the line where the merged family
sits, so the equality drops the merged pair outright rather than letting it cancel. It decides
nothing about the field elsewhere -- a shape one ulp short never reaches it. It is the move ``5e0561b`` already made for the
displacement -- closing the level set on the torus -- applied to the faces.

The tests are ordered by what they can rule out:

* against the **raster builder**, which is independent of this one and reads the geometry as an
  array (``test_a_period_spanning_bar_matches_the_raster_field``);
* against the **structure's own symmetry** -- a y-invariant layer cannot care how many y orders
  it is given, and that holds without any second builder to trust
  (``test_y_orders_cannot_change_a_y_invariant_answer``);
* against the **other modeling path**, order for order on a geometry both spell exactly
  (``test_the_two_paths_agree_order_for_order``).

All three separate cleanly. The field comparison is out by the full 1.0 before the fix
(``max|n_y|`` of 1 where the raster gives 0) and exact after; the two solve-level ones are
1.1e-5 and 1e-5 before, and machine precision after.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core import factorization as F
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice, sample_grid

PERIOD = 700.0
GRID = 64


def _analytic(shapes, lattice, n=GRID):
    return P.normal_vector_field_analytical(np, (n, n), shapes, np.float64, lattice)


def _bar(**kw):
    return dict({'type': 'rectangle', 'cx': 0.0, 'cy': 0.0, 'material': 4.0}, **kw)


def _solver(tox, toy, backend=0):
    return wavebit.rcwa(backend=backend, thickness=[300.0],
                        period=dict(x=PERIOD, y=PERIOD),
                        truncation_order=dict(x=tox, y=toy),
                        top=1, bot=1, wavelength=550.0, psi=0.0)


def _grating_layer(lx=350.0, ly=PERIOD):
    return [dict(background=1.0, shapes=[_bar(lx=lx, ly=ly, angle=0.0)])]


# --------------------------------------------------------------------------------------------
# the field itself
# --------------------------------------------------------------------------------------------

def test_a_period_spanning_bar_matches_the_raster_field():
    """The same grating, two builders that share no code: the fields must be the same field.

    The raster builder reads the geometry off an array and knows nothing about shapes, so it is
    the one oracle here that cannot inherit this builder's mistakes. It answers ``n_y = 0``
    exactly, because a bar that runs the height of the cell has no horizontal edge to find.

    Measured before the fix: ``max|n_y| = 1.00``, with |n_y| above 0.01 over 32.9% of the cell.
    """
    shapes = [_bar(cx=175.0, lx=PERIOD / 2, ly=PERIOD)]
    lattice = as_lattice(period=[PERIOD, PERIOD])
    normal_x, normal_y = _analytic(shapes, lattice)

    raster = np.ones((GRID, GRID))
    raster[:, :GRID // 2] = 4.0
    raster_x, raster_y = F.normal_vector_field(raster, lattice=lattice)

    # The oracle has to be answering the question before its answer means anything.
    # `normal_vector_field` short-circuits to a constant field when a raster axis is one deep,
    # so a 1 x N raster of this grating returns (1, 0) everywhere *by construction* and would
    # agree with anything. Verified: at 1x64 the field is constant, at 16x64 and 64x64 it is
    # computed. This asserts we are on the computed side.
    thin_x, thin_y = F.normal_vector_field(raster[:1], lattice=lattice)
    assert np.all(thin_x == thin_x.flat[0]) and np.all(thin_y == thin_y.flat[0]), (
        'the one-deep short-circuit is gone -- this guard no longer distinguishes anything')
    assert not np.all(raster_x == raster_x.flat[0]), (
        'the oracle returned a constant field, so it is not measuring this geometry')

    assert np.abs(normal_y).max() == 0.0
    # Sign is free -- a normal field is defined up to it, and the factorization reads only
    # quadratic combinations -- so the two are compared as directions.
    np.testing.assert_array_equal(np.abs(normal_x), np.abs(raster_x))
    np.testing.assert_array_equal(np.abs(normal_y), np.abs(raster_y))


def test_the_field_stays_a_unit_field_on_a_spanning_bar():
    """No sample may come back as ``(0, 0)``.

    Dropping the merged face pair rather than leaving it to cancel is what this pins. The
    cancellation alone is exact, but it leaves the surviving pair's weights carrying a factor
    that vanishes on the one line where the merged family sits -- half a repeat from the shape's
    centre. A ``cy`` that puts that line on a pixel row then produced 32 zero-normal samples of
    4096, and a zero normal is a null projector at that pixel, not a harmless one.

    ``cy`` here is chosen to do exactly that: half a pixel off centre.
    """
    half_pixel = 0.5 * PERIOD / GRID
    for cy in (0.0, half_pixel, -half_pixel, 123.0):
        shapes = [_bar(cy=cy, lx=PERIOD / 2, ly=PERIOD)]
        normal_x, normal_y = _analytic(shapes, as_lattice(period=[PERIOD, PERIOD]))
        norm = np.hypot(normal_x, normal_y)
        assert norm.min() == pytest.approx(1.0, abs=1e-12), f'zero normal at cy={cy}'


@pytest.mark.parametrize('angle, spans', [(0.0, True), (np.pi / 2, True), (np.pi, True),
                                          (np.pi / 4, False)])
def test_the_span_is_read_on_the_shapes_own_axis_not_on_x_and_y(angle, spans):
    """``ly`` spans when a *lattice vector* runs along the shape's y axis with that length.

    A quarter turn puts the shape's y axis along the cell's x, so ``ly = period`` still spans --
    on the other axis. Forty-five degrees does not span at all: the images step diagonally and
    every face is a real interface. It is accepted by :func:`check_layer_shapes` (the separating
    axis test clears it), so a rule written on ``ly == period`` alone would have silently erased
    four faces that are there.
    """
    shapes = [_bar(lx=PERIOD / 2, ly=PERIOD, angle=angle)]
    normal_x, normal_y = _analytic(shapes, as_lattice(period=[PERIOD, PERIOD]))
    # the component along the shape's own y axis -- what a spanning shape cannot have
    on_own_y = np.abs(-normal_x * np.sin(angle) + normal_y * np.cos(angle)).max()
    if spans:
        assert on_own_y < 1e-15
    else:
        assert on_own_y == pytest.approx(1.0, abs=1e-9)


def test_a_sheared_cell_keeps_the_face_it_still_has():
    """On ``a1 = (700, 0), a2 = (200, 700)`` a bar of ``ly = 700`` does **not** span.

    Its images are displaced 200 in x as well, so they step past each other rather than abut and
    the horizontal edge is a real material interface. The repeat is asked of the *lattice*, not
    of ``period`` -- ``periods_of`` reports ``(700, 700)`` for this cell and would have said span.

    The x direction of the same cell is the control: ``a1`` really does run along it, so a bar
    of ``lx = 700`` spans there and its vertical faces go.
    """
    sheared = as_lattice(lattice=np.array([[PERIOD, 200.0], [0.0, PERIOD]]))

    _, spans_nothing_y = _analytic([_bar(lx=PERIOD / 2, ly=PERIOD)], sheared)
    assert np.abs(spans_nothing_y).max() == pytest.approx(1.0, abs=1e-9)

    spans_x, _ = _analytic([_bar(lx=PERIOD, ly=PERIOD / 2)], sheared)
    assert np.abs(spans_x).max() == 0.0


def test_a_shape_filling_the_cell_gets_the_no_interface_constant():
    """Both pairs merged is a uniform slab written as a shape -- there is no interface anywhere.

    It answers with the same constant ``(1, 0)``
    :func:`~wavebit._core.parametric.normal_vector_field_analytical` uses when no shape has a
    field at all, rather than the ``(0, 0)`` four dropped faces would otherwise leave.
    """
    normal_x, normal_y = _analytic([_bar(lx=PERIOD, ly=PERIOD)],
                                   as_lattice(period=[PERIOD, PERIOD]))
    np.testing.assert_array_equal(normal_x, np.ones((GRID, GRID)))
    np.testing.assert_array_equal(normal_y, np.zeros((GRID, GRID)))


def test_an_ellipse_touching_the_edge_still_has_its_boundary():
    """A tangent point is not an abutting face, so the round branch must be untouched.

    An ellipse of ``ly = period`` meets its image at a single point and is a curved interface
    everywhere else. This is here because "the shape is as long as the period" is the tempting
    way to state the rule, and it is the wrong one: what matters is that the images abut *face
    to face*, which only a flat side can do.
    """
    shapes = [dict(type='ellipse', cx=0.0, cy=0.0, lx=PERIOD / 2, ly=PERIOD, material=4.0)]
    _, normal_y = _analytic(shapes, as_lattice(period=[PERIOD, PERIOD]))
    assert np.abs(normal_y).max() > 0.9


def test_the_projector_is_the_exact_one_dimensional_factorization():
    """``<nx^2> = 1``, ``<nx ny> = 0``, ``<ny^2> = 0``, and no off-DC term at all.

    This is Li's rule with x as the normal everywhere, which is *the* correct factorization of a
    grating that only varies in x -- so on this structure the NVF path has an exact answer to
    reproduce rather than a tolerance to sit inside. It read 0.860 / 0.140 before the fix: 14% of
    the projector pointing across the grooves.
    """
    shapes = [_bar(lx=PERIOD / 2, ly=PERIOD)]
    c_xx, c_xy, c_yy = P.normal_projector_coeffs(
        np, shapes, (256, 256), as_lattice(period=[PERIOD, PERIOD]), 6, 6, np.float64)
    dc = (c_xx.shape[0] // 2, c_xx.shape[1] // 2)

    assert c_xx[dc].real == pytest.approx(1.0, abs=1e-12)
    assert np.abs(c_yy).max() == 0.0
    assert np.abs(c_xy).max() == 0.0
    # every coefficient other than DC is a variation this structure does not have
    off_dc = np.abs(c_xx).sum() - abs(c_xx[dc])
    assert off_dc == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------------
# what it does to a solve
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('tox', [3, 5, 7])
def test_y_orders_cannot_change_a_y_invariant_answer(tox):
    """A layer uniform in y must give the same answer with 0 y orders as with ``tox`` of them.

    The strongest of these, because it needs no second builder and no reference value: adding
    orders along a direction the structure does not vary in can only add zeros, and the
    permittivity coefficients supply them on both settings. Any gap is the normal field alone.

    Measured 1.1e-5 at ``tox = 3`` before the fix and 6.3e-7 at ``tox = 13`` -- a ratio of
    0.056 against 0.053 for ``1/tox**2`` and 0.231 for ``1/tox``, so it decays like a second-order
    truncation error, which is exactly what made it look like ordinary truncation error in the
    convergence study it corrupted. ("Before the fix" here means both halves removed: with only
    the fold deleted the equality still fires at ``ly == period`` and the invariant holds.)
    """
    flat = _solver(tox, 0)
    flat.structure = _grating_layer()
    square = _solver(tox, tox)
    square.structure = _grating_layer()

    assert float(np.sum(flat.solve().given.T)) == pytest.approx(
        float(np.sum(square.solve().given.T)), abs=1e-12)


@pytest.mark.parametrize('tox', [3, 7, 13])
def test_the_two_paths_agree_order_for_order(tox):
    """The same grating, spelled as a shape and as a raster, must solve to the same number.

    ``lx = 350`` on a 700 period puts both edges exactly on the pixel grid of an ``N = 1400``
    raster, so the two paths describe the *same* structure to the last bit and any gap is
    factorization rather than geometry. It was 1.3e-4 at ``truncation_order`` 3 and still 5.4e-6
    at 13 -- converging, slowly, from a normal field that was wrong by a constant.
    """
    n = 1400
    shape = _solver(tox, 0)
    shape.structure = _grating_layer()

    raster = _solver(tox, 0)
    x = (np.arange(n) + 0.5) / n * PERIOD - PERIOD / 2
    raster.structure = (np.where(np.abs(x)[None, :] <= 175.0, 4.0, 1.0)
                        * np.ones((n, 1)))[None]

    assert float(np.sum(shape.solve().given.T)) == pytest.approx(
        float(np.sum(raster.solve().given.T)), abs=1e-12)


def test_growing_a_rod_into_a_slab_has_no_cliff_at_the_period():
    """``sum(T)`` must run smoothly into its spanning value, not jump onto it.

    An optimiser reaches ``ly = period`` by the most ordinary move it has, and the refusal above
    it means this is the edge of the feasible set -- exactly where a discontinuity would be read
    as a gradient. Refusing ``ly == period`` outright, the other defensible fix, puts a step of
    4.3e-5 here; erasing the faces on an equality test alone puts the same step here. Folding the
    face distance leaves no step, because the fold is what a shape one ulp short already gets.
    """
    span = _solver(5, 5)
    span.structure = _grating_layer()
    at_period = float(np.sum(span.solve().given.T))

    previous = None
    for gap in (1e-1, 1e-2, 1e-4, 1e-6):
        near = _solver(5, 5)
        near.structure = _grating_layer(ly=PERIOD - gap)
        value = abs(float(np.sum(near.solve().given.T)) - at_period)
        if previous is not None:
            assert value < previous, f'not approaching the spanning value at gap {gap}'
        previous = value
    assert previous < 1e-5


@pytest.mark.parametrize('backend', [0, 1, 2])
def test_every_backend_spans_the_same_way(backend):
    """numpy, jax and torch build this field through three different namespaces.

    The fold is written in ``xp`` calls that each backend spells its own way -- ``round``,
    ``where``, and a reciprocal guarded so that "no repeat along this axis" is a zero multiplier
    rather than an infinity. An ``inf`` there reads better and does not survive autograd:
    ``P * round(s/inf)`` is ``inf * 0``, a NaN in the branch a ``where`` discards and
    reverse-mode still reads.
    """
    pytest.importorskip({0: 'numpy', 1: 'jax', 2: 'torch'}[backend])
    solver = _solver(5, 5, backend=backend)
    solver.structure = _grating_layer()
    assert float(np.sum(np.asarray(solver.solve().given.T))) == pytest.approx(
        0.976311242711, abs=1e-9)


def test_a_spanning_shape_still_traces_and_differentiates():
    """``jit`` and ``grad`` must cross the fold, which is why it carries no Python branch.

    Every question the fold asks -- is this lattice vector along the shape's axis, does the
    shape's own length land on the repeat -- is answered with ``where`` on the value rather than
    with ``if``, so a traced ``lx`` / ``ly`` / ``angle`` reaches the same code the eager call
    does. Branching on any of them would raise ``TracerBoolConversionError`` on every jitted
    parametric solve, not just on a spanning one.
    """
    jax = pytest.importorskip('jax')
    jnp = pytest.importorskip('jax.numpy')
    jax.config.update('jax_enable_x64', True)

    def sum_t(ly):
        solver = _solver(4, 4, backend=1)
        solver.structure = [dict(background=1.0, shapes=[dict(
            type='rectangle', cx=0.0, cy=0.0, lx=jnp.asarray(350.0), ly=ly,
            angle=jnp.asarray(0.0), material=4.0)])]
        return jnp.sum(solver.solve().given.T)

    eager = float(sum_t(jnp.asarray(PERIOD)))
    assert float(jax.jit(sum_t)(jnp.asarray(PERIOD))) == pytest.approx(eager, abs=1e-12)

    # and the gradient is a real one just below the span, where the fold is doing its work
    grad = float(jax.grad(sum_t)(jnp.asarray(690.0)))
    step = 0.5
    central = (float(sum_t(jnp.asarray(690.0 + step)))
               - float(sum_t(jnp.asarray(690.0 - step)))) / (2 * step)
    assert grad == pytest.approx(central, rel=2e-2)


def test_the_lattice_direction_test_is_a_threshold():
    """The one tolerance in the fold is recorded here rather than left to be discovered.

    "Does a lattice vector run along the shape's axis" cannot be asked exactly: ``pi/2`` is not
    a representable angle, so a shape turned a quarter turn misses an exact test by 4.3e-14 on
    a 700 cell. :data:`~wavebit._core.parametric._LATTICE_DIR_TOL2` is where the line is drawn,
    and a shape more than half a repeat long steps across it -- it folds on one side and not on
    the other.

    Two things make that acceptable, and both are pinned here rather than asserted in prose. The
    step is at 1e-11 rad, which is 5.7e-10 degrees. And a shape at or below half a repeat does not
    step at all, because the fold is arithmetically inert there -- which is most shapes.
    """
    lattice = as_lattice(period=[PERIOD, PERIOD])

    def field(angle, ly):
        return np.stack(_analytic([_bar(lx=250.0, ly=ly, angle=angle)], lattice, n=128))

    # above half a repeat: inert below the threshold, a step above it
    tall = field(0.0, 0.9 * PERIOD)
    assert np.abs(field(1e-12, 0.9 * PERIOD) - tall).max() < 1e-9
    assert np.abs(field(1e-11, 0.9 * PERIOD) - tall).max() > 1e-2

    # at or below half a repeat there is no threshold to cross -- the fold never applies
    squat = field(0.0, 0.5 * PERIOD)
    for angle in (1e-12, 1e-11, 1e-9, 1e-6):
        assert np.abs(field(angle, 0.5 * PERIOD) - squat).max() < 1e-3


def test_a_rhombic_cell_spans_along_the_sum_of_its_basis_vectors():
    """``a1 + a2`` is a lattice direction too, and on some cells it is the only one that matters.

    This is the case that killed the first version of
    :func:`~wavebit._core.parametric._axis_lattice_periods`, which offered only ``+-a1`` and
    ``+-a2`` as candidates on the reasoning that a shape reaching across the sum would run into
    a nearer image and be refused upstream. It would not: on the Li checkerboard basis the two
    basis vectors sit at +-45 degrees, their sum is axis-aligned, and ``check_layer_shapes``
    accepts ``lx`` right up to the full ``|a1 + a2|``.

    The structure at that length repeats end to end along x and has no vertical face at all.
    The two-vector version answered ``max|n_x| = 1.000``, above 0.01 over 46.9% of the cell.

    The checkerboard *itself* is the control, and it is why the candidate list is checked with a
    strict inequality: ``lx = ly = 1.25`` is exactly half of ``|a1 + a2| = 2.5``, the fold is
    inert at exactly half, and the field is bit-identical to what it was before any of this.
    That matters because that scene is a published-number comparison.
    """
    checker = as_lattice(lattice=np.array([[1.25, 1.25], [1.25, -1.25]]))
    span = np.hypot(1.25 + 1.25, 1.25 - 1.25)          # |a1 + a2| = 2.5, along x
    assert span == pytest.approx(2.5, abs=1e-12)

    # accepted by the layer check, so the field builder is what has to be right about it
    P.check_layer_shapes({'background': 1.0, 'shapes': [
        dict(type='rectangle', cx=0.0, cy=0.0, lx=span, ly=1.25, angle=0.0, material=4.0)]},
        checker)

    normal_x, _ = _analytic([dict(type='rectangle', cx=0.0, cy=0.0, lx=span, ly=1.25,
                                  angle=0.0, material=4.0)], checker)
    assert np.abs(normal_x).max() == 0.0

    # and the candidate set matches `overlap._IMAGES` -- the same 3x3 the refusal tests against
    from wavebit._core import overlap as _overlap
    assert set(_overlap._IMAGES) == {(m, n) for m in (-1, 0, 1) for n in (-1, 0, 1)}


# --------------------------------------------------------------------------------------------
# the three branches a mutation pass found nothing reaching
# --------------------------------------------------------------------------------------------

def test_the_x_fold_carries_its_own_weight():
    """Deleting the fold on the **x** face pair alone must break something. It used to not.

    Every other test here spans ``ly``, so a mutant that folds only the y faces passed 217 tests
    across five files while leaving the x half unfolded. That is the exact half of this change
    the original report asked about by name -- "check ``lx == period`` symmetrically; this was
    only measured on ``ly``" -- and it was implemented and then pinned by nothing.

    The invariant is a mirror, which is what makes it need no reference value: a structure and
    its 90-degree rotation are the same structure with s and p exchanged, so ``sum(T)`` at
    ``psi = 0`` on one must equal ``sum(T)`` at ``psi = pi/2`` on the other, to the last digit.
    That holds only if both axes fold. With the x fold deleted the two sides part by 2.7e-4,
    while every ``ly`` number in the file stays byte-identical.

    ``lx = 630`` is above half of 700 and below it, so the fold is live and the equality is not
    -- this reaches the fold and not ``merged_u``, which
    ``test_a_shape_filling_the_cell_gets_the_no_interface_constant`` already covers.
    """
    def total(lx, ly, psi):
        s = wavebit.rcwa(backend=0, thickness=[300.0], period=dict(x=PERIOD, y=PERIOD),
                         truncation_order=dict(x=5, y=5), top=1, bot=1,
                         wavelength=550.0, psi=psi)
        s.structure = [dict(background=1.0, shapes=[_bar(lx=lx, ly=ly, angle=0.0)])]
        return float(np.sum(s.solve().given.T))

    # 1.3e-14 measured, against 2.7e-4 with the x fold deleted -- six orders either side of
    # this line. Not exact equality: the s/p exchange reaches the same physics through a
    # different rotation, so the last bits differ for reasons that have nothing to do with folds.
    wide, tall = total(630.0, 250.0, 0.0), total(250.0, 630.0, np.pi / 2)
    assert abs(wide - tall) < 1e-10, (wide, tall)
    assert abs(total(630.0, 250.0, np.pi / 2) - total(250.0, 630.0, 0.0)) < 1e-10

    # and the x field really is flat where the shape spans x, which is the direct statement
    normal_x, _ = _analytic([_bar(lx=PERIOD, ly=250.0, angle=0.0)],
                            as_lattice(period=[PERIOD, PERIOD]))
    assert np.abs(normal_x).max() == 0.0


def test_a_merged_pair_never_becomes_the_nearest_surface():
    """A dropped face pair must leave the *distance* too, not only the blend. Nothing reached it.

    ``_shape_surface_normal`` returns a distance alongside the normal, and that distance decides
    which shape owns a point when a layer holds several. A merged pair is not a surface, so it
    must not be allowed to be the nearest one -- and collapsing that reduction to a plain
    ``xp.minimum(near_u, near_v)`` passed every other test in this file, because they all build
    a layer holding one shape, where the reduction has nothing to arbitrate.

    This checks the distance directly instead, against a closed form, which is what makes it
    need neither a golden nor a second builder: for a bar spanning ``ly`` the only interfaces are
    its two vertical faces, so inside it the nearest surface is exactly ``a - |x|``. Clean code
    reproduces that to 0.0; reading the phantom horizontal face puts it 64.5 out.

    The non-spanning control is the same shape with ``ly`` short of the period. There the
    horizontal faces are real, the same closed form is *wrong*, and the assertion below would
    fail -- which is how this test shows it can fail at all.
    """
    lattice = as_lattice(period=[PERIOD, PERIOD])
    n = 128
    X, Y = sample_grid(np, (n, n), lattice, np.float64)
    # Two frames, and they are half a cell apart. `_shape_surface_normal` wants the *absolute*
    # sample point, so `X`/`Y` go into it untouched; the closed form below is a statement about
    # where the point sits relative to the shape, whose `cx = cy = 0` names the cell centre. So
    # the assertions read `Xc`/`Yc`. `sample_grid` is corner-origin and `meshgrid_xy` is the
    # public door that re-bases -- mixing them silently misplaces everything by `0.5*(a1 + a2)`.
    mid_x = 0.5 * (lattice[0][0] + lattice[0][1])
    mid_y = 0.5 * (lattice[1][0] + lattice[1][1])
    Xc, Yc = X - mid_x, Y - mid_y
    a = 70.0

    spanning = _bar(lx=2 * a, ly=PERIOD, angle=0.0)
    dist, _, _ = P._shape_surface_normal(np, X, Y, spanning, np.float64, lattice)
    inside = np.abs(Xc) < a
    assert np.abs(np.asarray(dist)[inside] - (a - np.abs(Xc))[inside]).max() == 0.0

    # the control: with real horizontal faces that closed form is not the answer
    short = _bar(lx=2 * a, ly=300.0, angle=0.0)
    d2, _, _ = P._shape_surface_normal(np, X, Y, short, np.float64, lattice)
    box = inside & (np.abs(Yc) < 150.0)
    assert np.abs(np.asarray(d2)[box] - (a - np.abs(Xc))[box]).max() > 1.0


def test_the_merged_test_is_an_equality_and_not_a_window():
    """``fold(2a, Pu) == 0`` is exact on purpose. A tolerance window passes every other test.

    The code and its comment call the exactness a decided choice; replacing it with a 7e-4
    absolute window on a 700 period passed all 23 tests. It has to be exact because the fold
    already carries every length *below* the span continuously -- a window would take shapes the
    fold is handling correctly and delete their faces instead, and the wider the window the more
    of them.

    A bar one part in 1e5 short of the period still has two real (if very close) interfaces, and
    its answer has to differ from the spanning one by about the amount its geometry differs --
    not by nothing, which is what a window that swallowed it would give.
    """
    lattice = as_lattice(period=[PERIOD, PERIOD])
    spanning = _analytic([_bar(lx=350.0, ly=PERIOD, angle=0.0)], lattice)
    just_short = _analytic([_bar(lx=350.0, ly=PERIOD * (1 - 1e-5), angle=0.0)], lattice)

    # the merged case is flat in y; the one just short of it is not, and not by a little
    assert np.abs(spanning[1]).max() == 0.0
    assert np.abs(just_short[1]).max() > 0.5, np.abs(just_short[1]).max()

    # ... while staying continuous: closer to the span means closer to flat
    closer = _analytic([_bar(lx=350.0, ly=PERIOD * (1 - 1e-9), angle=0.0)], lattice)
    assert np.abs(closer[1]).sum() < np.abs(just_short[1]).sum()


def test_a_merged_pair_leaves_no_null_normal_behind():
    """Once one face pair is gone, the other cancels on its centre line. Open, and not the drop.

    A rectangle spanning ``|a1 + a2|`` on the Li checkerboard has its u pair merged, so ``n_x`` is
    zero identically -- correct, that direction has no interface. What is left is the v pair, and
    a four-face blend gives the two members equal weight on their own midline, where their
    opposite normals cancel to ``(0, 0)``. A null normal is not a harmless projector; it is a null
    projection.

    **It is the fold, not the equality test.** Forcing ``merged_u`` / ``merged_v`` to False leaves
    the count unchanged at 256 with ``max|n_x|`` still 0.000000: at ``lx = |a1 + a2|`` the fold
    alone already makes the two u distances equal everywhere, so the pair cancels whether or not
    anything drops it. Any mechanism that removes a pair inherits this, which is why the sibling
    session's independent construction -- an explicit face-pair drop rather than a fold -- lands on
    exactly the same 128 / 256 / 512.

    A square cell hides it. The surviving pair's midline is ``x = cx`` there, and a centred
    ``(i + 0.5)/N`` grid with even ``N`` never samples it: the same shape on a 700 square cell
    gives zero null normals at every ``N``. The checkerboard's skewed grid crosses the line once
    per row, so the count is ``N`` however fine the grid -- one line, not a shrinking set.

    The fix is to break the tie rather than to remove the pair differently: where the blend lands
    on zero and exactly one pair is gone, substitute the surviving axis. It needs no sign, since
    the factorization reads ``n n^T``. Bit-identical on every scene with no pair dropped, and on
    the 1-D grating and the cell-filling slab, which have no such point to substitute at.
    """
    checker = as_lattice(lattice=np.array([[1.25, 1.25], [1.25, -1.25]]))
    span = _bar(lx=2.5, ly=0.8, angle=0.0)
    for n in (128, 256):
        normal_x, normal_y = _analytic([span], checker, n=n)
        assert np.abs(normal_x).max() == 0.0          # the merged direction, correct
        assert int((np.hypot(normal_x, normal_y) == 0.0).sum()) == 0
