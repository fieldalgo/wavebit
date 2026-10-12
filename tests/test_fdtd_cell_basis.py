"""The FDTD scene geometry reads the cell basis: general where the geometry is, refused where the Yee box is.

``_core/scene_fdtd.py`` took only ``periods`` -- the edge lengths ``(|a1|, |a2|)``, with the cell
angle already gone -- and ``tests/test_cell_shape_gate.py`` flagged eleven lines of period-pair
arithmetic there and a dropped lattice (2026-09-11). A shape recipe means the same thing on any
cell, so ``_shape_partition``'s area route now reads a skewed cell for what it is: it folds the
centre into the cell on the reduced basis and checks sizes against the cell's width across each
pair of its edges. The functions that stand on the Cartesian Yee grid's rectangular periodic box
refuse a skewed cell beside their own arithmetic (``_yee_box``), not only at the scene engine's
door. On a rectangle every result is the one it was, which the byte comparison run with the
change measured; the tests here pin the parts that comparison cannot see.
"""
import numpy as np
import pytest

from wavebit._core.lattice import as_lattice, is_axis_aligned, periods_of, reduced_basis
from wavebit._core.parametric import shape_geometry
from wavebit._core.scene_fdtd import (_image_shifts, _painted_cell_materials, _painted_rectangle_edges,
                                     _rectangle_edge_moments, _require_scene, _shape_cell_fractions,
                                     _shape_partition, _yee_box)
from wavebit.backends.time_domain import time_loop_backend

PERIODS = [.6, .4]
HEX = as_lattice(lattice=np.column_stack(([.6, 0.], [.3, .6*np.sqrt(3)/2])))
OBLIQUE = as_lattice(lattice=np.column_stack(([.7, .1], [.25, .55])))


def area_partition(shapes, lattice):
    runtime = time_loop_backend('numpy')
    return _shape_partition([dict(background=1., shapes=shapes)], [float(v) for v in periods_of(lattice)],
                            xp=runtime.xp, array=runtime.array, check=_require_scene,
                            integration='area', lattice=lattice)


@pytest.mark.parametrize('periods', [[.6, .4], np.array([.6, .4]), [1., .8]])
@pytest.mark.parametrize('rings', [(1, 1), (2, 2), (2, 1)])
def test_the_image_ring_is_the_literal_it_replaced_to_the_bit(periods, rings):
    """Six copies of this literal became one helper; the sums over it must not regroup."""
    runtime = time_loop_backend('numpy')
    new = np.asarray(_image_shifts(periods, as_lattice(period=periods), rings, array=runtime.array))
    old = np.asarray(runtime.array([(i*periods[0], j*periods[1]) for i in range(-rings[0], rings[0]+1)
                                    for j in range(-rings[1], rings[1]+1)]))
    assert (new.dtype, new.shape) == (old.dtype, old.shape) and new.tobytes() == old.tobytes()


def test_the_yee_box_is_the_callers_periods_and_nothing_else():
    """The box hands back the caller's own objects, and refuses a cell it was not built on."""
    box = _yee_box(PERIODS, as_lattice(period=PERIODS))
    assert box[0] is PERIODS[0] and box[1] is PERIODS[1]
    with pytest.raises(ValueError, match='Cartesian periodic box'):
        _yee_box(PERIODS, HEX)
    with pytest.raises(ValueError, match='periods and lattice disagree'):
        _yee_box(PERIODS, as_lattice(period=[.6, .5]))
    # Periods equal to a skewed cell's own diagonal: only the skew refusal stands between this
    # call and a rectangle's answer, so it has to be the one that fires.
    skewed = as_lattice(lattice=np.column_stack(([.6, 0.], [.2, .5])))
    with pytest.raises(ValueError, match='Cartesian periodic box'):
        _yee_box([.6, .5], skewed)


def test_every_function_that_stands_on_the_yee_box_refuses_a_skewed_cell():
    """The guard sits beside the arithmetic now, not only a thousand lines away at the door."""
    runtime = time_loop_backend('numpy')
    array = runtime.array
    square = array(np.array([[.1, .1], [.2, .1], [.2, .2], [.1, .2]]))
    material = (array(2.), array(1.))
    background = (array(1.), array(1.))
    x, y = np.arange(6)*.1, np.arange(4)*.1
    calls = {
        '_painted_rectangle_edges': lambda: _painted_rectangle_edges(
            [(square, material), (square, material)], background, PERIODS, xp=runtime.xp, array=array,
            lattice=HEX),
        '_painted_cell_materials': lambda: _painted_cell_materials(
            [], background, x, y, .1, .1, PERIODS, xp=runtime.xp, array=array, lattice=HEX),
        '_shape_cell_fractions': lambda: _shape_cell_fractions(
            square, x, y, .1, .1, PERIODS, xp=runtime.xp, array=array, lattice=HEX),
        '_rectangle_edge_moments': lambda: _rectangle_edge_moments(
            square, x, y, .1, .1, PERIODS, xp=runtime.xp, array=array, lattice=HEX),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match='Cartesian periodic box'):
            call()


def test_several_painted_rectangles_on_a_skewed_cell_are_refused_by_the_box_they_fold_into():
    with pytest.raises(ValueError, match='Cartesian periodic box'):
        area_partition([dict(lx=.1, ly=.1, cx=.05, material=2.), dict(lx=.1, ly=.1, cx=-.1, material=3.)], HEX)


def test_the_separable_partition_refuses_a_skewed_cell():
    runtime = time_loop_backend('numpy')
    with pytest.raises(ValueError, match='face intervals'):
        _shape_partition([dict(background=1., shapes=[dict(lx=.2, ly=.2, material=2.)])],
                         [float(v) for v in periods_of(HEX)], xp=runtime.xp, array=runtime.array,
                         check=_require_scene, lattice=HEX)


def fractional_span(kind, lx, ly, angle, lattice):
    """The largest span of a shape in either fractional coordinate of the reduced basis.

    Independent of the implementation's normals and widths: the reciprocal rows ``b_k`` of the
    inverted basis, and the support function ``h`` of the shape along each, give a span of
    ``2*h(b_k)`` cells in coordinate ``k``.
    """
    reciprocal = np.linalg.inv(np.asarray(reduced_basis(lattice)))
    u, v = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
    spans = []
    for w in reciprocal:
        if kind == 'circle':
            h = lx/2*np.linalg.norm(w)
        elif kind == 'ellipse':
            h = np.hypot(lx/2*(w @ u), ly/2*(w @ v))
        else:
            h = lx/2*abs(w @ u)+ly/2*abs(w @ v)
        spans.append(2*h)
    return max(spans)


def accepted(kind, lx, ly, angle, lattice):
    shape = dict(type=kind, cx=.05, cy=-.02, lx=lx, angle=angle, material=2.)
    if kind != 'circle':
        shape['ly'] = ly
    try:
        area_partition([shape], lattice)
    except ValueError as exc:
        if 'must not exceed' in str(exc) or 'no larger than' in str(exc):
            return False
        raise
    return True


@pytest.mark.parametrize('lattice', [HEX, OBLIQUE], ids=['hex', 'oblique'])
@pytest.mark.parametrize('kind', ['rectangle', 'ellipse', 'circle'])
def test_a_shape_fits_a_skewed_cell_exactly_when_it_spans_at_most_one_cell(lattice, kind):
    """Sampled across the whole range, not only at the ends: both outcomes must occur."""
    rng = np.random.default_rng(7)
    outcomes = set()
    for _ in range(120):
        lx, ly = rng.uniform(.05, 1.), rng.uniform(.05, 1.)
        angle = rng.uniform(0., np.pi)
        span = fractional_span(kind, lx, ly, angle, lattice)
        if abs(span-1) < 1e-9:
            continue
        got = accepted(kind, lx, ly, angle, lattice)
        assert got == (span <= 1), (kind, lx, ly, angle, span)
        outcomes.add(got)
    assert outcomes == {True, False}


@pytest.mark.parametrize('kind', ['rectangle', 'ellipse', 'circle'])
def test_a_sheared_spelling_of_a_rectangle_accepts_what_the_rectangle_accepts(kind):
    """``a2 = 2*a1 + (0, .4)`` is the .6 x .4 rectangle spelled skewed; it takes the general branch."""
    sheared = as_lattice(lattice=np.column_stack(([.6, 0.], [1.2, .4])))
    rectangle = as_lattice(period=PERIODS)
    assert not is_axis_aligned(sheared) and is_axis_aligned(rectangle)
    rng = np.random.default_rng(11)
    for _ in range(60):
        lx, ly, angle = rng.uniform(.05, .8), rng.uniform(.05, .8), rng.uniform(0., np.pi)
        if abs(fractional_span(kind, lx, ly, angle, rectangle)-1) < 1e-9:
            continue
        assert accepted(kind, lx, ly, angle, sheared) == accepted(kind, lx, ly, angle, rectangle)


SHEAR = as_lattice(lattice=np.column_stack(([.4, 0.], [1.25, .5])))     # reduces to a1, a2 - 3*a1


@pytest.mark.parametrize('lattice, given', [(HEX, (1.37/.6, -2.11/.52)), (OBLIQUE, (.8, .6)), (SHEAR, (.3, .4))],
                         ids=['hex', 'oblique', 'shear'])
def test_a_centre_on_a_skewed_cell_folds_into_the_reduced_cell_by_a_lattice_vector(lattice, given):
    """Folded on the *reduced* basis -- the cell the fit is measured on -- not on the basis as spelled.

    OBLIQUE and SHEAR are not reduced, and the centre is placed inside the cell the basis as given
    spells but outside the reduced one, so a fold on the given basis would leave it where it is
    and fail here. HEX is its own reduction; it keeps the far-away centre.
    """
    reduced = np.asarray(reduced_basis(lattice))
    target = np.asarray(lattice) @ np.asarray(given)
    probe = dict(type='ellipse', cx=0., cy=0., lx=.1, ly=.05, angle=.3, material=2.)
    _, ox, oy, *_ = shape_geometry(probe, lattice)             # where the recipe's (0, 0) lands
    shape = dict(probe, cx=float(target[0]-float(ox)), cy=float(target[1]-float(oy)))
    _, _, layers = area_partition([shape], lattice)
    folded = np.asarray(layers[0][1][0][:2])     # (base, (vector, material)); the vector leads with the centre
    fraction = np.linalg.solve(reduced, folded)
    assert np.all((fraction >= 0) & (fraction < 1)), fraction
    steps = np.linalg.solve(reduced, target-folded)
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-12)
    start = np.linalg.solve(reduced, target)
    assert np.any((start < 0) | (start >= 1)), 'the centre already sat in the reduced cell: nothing tested'
