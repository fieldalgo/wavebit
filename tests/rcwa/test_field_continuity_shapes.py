"""``discontinuous='continuity'`` on a shape structure: the rule along lines through the shapes.

The rule itself is pinned elsewhere, on a raster, against RETICOLO
(``tests/references/test_reticolo_2d_field_golden.py``). What is new here is where it reads a line
from, so the first anchor is a raster that draws the same walls: both then have the same operators
and the same eps, and any gap is this route. Walls off every pixel line have no such raster. There
RETICOLO pins Li's rules, a fine raster holds the operators, and ``eps E_x`` having no orders past
the truncation holds ``eps`` at the samples on every factorization, with no reference or label.
The derivative is against finite differences, and what it does not cover is stated by a test of
its own.
"""
import copy
import pickle
from collections import deque, namedtuple
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

import wavebit
from wavebit._common.backend import _enable_x64
from wavebit.lattice import Lattice
from wavebit.rcwa._engine.continuity import solved_structure
from wavebit.rcwa._engine.factorization import line_inverse_rules
from wavebit.rcwa._engine.line_crossings import (
    edge_crossings, eps_at, line_operators, read_lines, segments)
from wavebit.rcwa._factory import rcwa

PX, PY, NX, NY = 400.0, 360.0, 8, 9
DX, DY = PX / NX, PY / NY
SAMPLES = {
    'count': (7, 5),
    'walls': (np.array([-150.0, -100.0, -7.3, 0.0, 50.0, 100.0, 175.0]),
              np.array([-160.0, -80.0, -13.1, 0.0, 40.0, 120.0])),
    'away': (np.array([-950.0, 1234.5, 2000.0]), np.array([-1000.0, 77.7, 1440.0])),
}


def _rect(i0, i1, j0, j1, material, **over):
    """A rectangle over raster columns ``i0:i1`` and rows ``j0:j1``."""
    return dict(type='rectangle', cx=(i0 + i1) / 2 * DX - PX / 2, cy=(j0 + j1) / 2 * DY - PY / 2,
                lx=(i1 - i0) * DX, ly=(j1 - j0) * DY, material=material, **over)


def _paint(background, *rects):
    """The raster of a layer of pixel-aligned rectangles, each over the ones before it."""
    grid = np.full((NY, NX), background, dtype=complex)
    for i0, i1, j0, j1, material in rects:
        grid[j0:j1, i0:i1] = material
    return grid


LOWER = (2.25, (4, 7, 0, 3, (1.6 + 0.05j) ** 2), (0, 2, 6, 9, 1.0))
UPPER = (1.0, (2, 6, 1, 5, 4.0), (1, 3, 5, 7, 6.25))
PAINTED = (1.5, (1, 6, 2, 7, 4.0), (4, 7, 0, 4, 9.0))        # the second over the first
NESTED = (1.0, (1, 7, 1, 8, 4.0), (3, 5, 3, 6, 1.0))         # a hole
SCENES = {'one': (UPPER,), 'two': (LOWER, UPPER), 'painted': (PAINTED,), 'nested': (NESTED,)}


def _both(scene, factorization='li', **over):
    """The same walls as shapes and as a raster, each solved."""
    kw = dict(backend='numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[3, 3],
              wavelength=620.0, thickness=[150.0, 90.0][:len(scene)], theta=0.3, phi=0.4,
              answers='field', factorization=factorization)
    kw.update(over)
    shapes = rcwa(structure=[dict(background=bg, shapes=[_rect(*r[:5]) for r in rects])
                             for bg, *rects in scene], **kw)
    raster = rcwa(structure=np.stack([_paint(*layer) for layer in scene]),
                  fourier_coefficients='integrated', **kw)
    shapes.solve()
    raster.solve()
    return shapes, raster


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('factorization', ['li', 'laurent'])
@pytest.mark.parametrize('name', list(SCENES))
def test_the_same_walls_read_the_same_as_a_raster(name, factorization, side):
    """The two routes agree on every grid, both sides of every wall and outside the cell.

    The sides decide the layer order the rule reads in, the second layer the order itself, the
    painted scene which shape is on top, and ``'walls'`` the positive side of a wall.
    """
    shapes, raster = _both(SCENES[name], factorization, incidence_side=side)
    for x, y in SAMPLES.values():
        rule = shapes.reconstruct_field(x, y, 3, discontinuous='continuity')
        assert _rel(rule, raster.reconstruct_field(x, y, 3, discontinuous='continuity')) < 5e-12
        assert _rel(rule, shapes.reconstruct_field(x, y, 3)) > 0.01     # the rule did something


def test_walls_that_round_below_a_sample_are_lifted_back_onto_it():
    """Period 400 over 6 pixels puts the wall at 66.66666666666669 and the sample a rounding below.

    On 400 over 8 every wall and sample are exact, and the anchor above cannot tell.
    """
    period = (400.0, 350.0)
    grid = np.full((7, 6), 1.0, dtype=complex)
    grid[2:5, 1:4] = 4.0
    wall_x = (np.arange(7) / 6 - 0.5) * period[0]
    wall_y = (np.arange(8) / 7 - 0.5) * period[1]
    cx, lx = (1 + 4) / 2 / 6 * period[0] - period[0] / 2, 3 / 6 * period[0]
    cy, ly = (2 + 5) / 2 / 7 * period[1] - period[1] / 2, 3 / 7 * period[1]
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=list(period), truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='li')
    shapes = rcwa(structure=[dict(background=1.0, shapes=[dict(type='rectangle', cx=cx, cy=cy, lx=lx,
                                                               ly=ly, material=4.0)])], **kw)
    raster = rcwa(structure=grid[None], fourier_coefficients='integrated', **kw)
    shapes.solve()
    raster.solve()
    assert _rel(shapes.reconstruct_field(wall_x, wall_y, 3, discontinuous='continuity'),
                raster.reconstruct_field(wall_x, wall_y, 3, discontinuous='continuity')) < 5e-12


L_OUTLINE = [(-150.0, -140.0), (100.0, -140.0), (100.0, -20.0), (0.0, -20.0), (0.0, 100.0),
             (-150.0, 100.0)]


@pytest.mark.parametrize('winding', [1, -1])
def test_an_outline_in_either_winding_reads_as_its_raster(winding):
    grid = np.full((NY, NX), 1.0, dtype=complex)
    grid[1:4, 1:6] = 4.0
    grid[4:7, 1:4] = 4.0
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent')
    outline = dict(type='polygon', vertices=L_OUTLINE[::winding], material=4.0)
    shapes = rcwa(structure=[dict(background=1.0, shapes=[outline])], **kw)
    raster = rcwa(structure=grid[None], fourier_coefficients='integrated', **kw)
    shapes.solve()
    raster.solve()
    x, y = SAMPLES['walls']
    assert _rel(shapes.reconstruct_field(x, y, 3, discontinuous='continuity'),
                raster.reconstruct_field(x, y, 3, discontinuous='continuity')) < 5e-12


def _solved_layer(**rect):
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent',
              theta=0.2, phi=0.3)
    s = rcwa(structure=[dict(background=1.0, shapes=[dict(type='rectangle', material=4.0, **rect)])],
             **kw)
    s.solve()
    return s


def test_spellings_of_one_rectangle_read_the_same():
    """A quarter turn is its lengths swapped, and whole periods over is the same structure."""
    x, y = SAMPLES['walls']
    plain = _solved_layer(cx=30.0, cy=-20.0, lx=150.0, ly=90.0)
    field = plain.reconstruct_field(x, y, 3, discontinuous='continuity')
    turned = _solved_layer(cx=30.0, cy=-20.0, lx=90.0, ly=150.0, angle=np.pi / 2)
    assert _rel(turned.reconstruct_field(x, y, 3, discontinuous='continuity'), field) < 1e-12
    moved = _solved_layer(cx=30.0 + 2 * PX, cy=-20.0 - 3 * PY, lx=150.0, ly=90.0)
    assert _rel(moved.reconstruct_field(x, y, 3, discontinuous='continuity'), field) < 1e-12


def test_a_shape_through_both_cell_edges_reads_as_the_raster_that_wraps():
    x, y = SAMPLES['walls']
    grid = np.full((NY, NX), 1.0, dtype=complex)
    grid[np.ix_([8, 0, 1], [7, 0, 1])] = 4.0
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent')
    raster = rcwa(structure=grid[None], fourier_coefficients='integrated', **kw)
    raster.solve()
    corner = dict(type='rectangle', cx=-PX / 2 + DX / 2, cy=-PY / 2 + DY / 2, lx=3 * DX, ly=3 * DY,
                  material=4.0)
    shapes = rcwa(structure=[dict(background=1.0, shapes=[corner])], **kw)
    shapes.solve()
    assert _rel(shapes.reconstruct_field(x, y, 3, discontinuous='continuity'),
                raster.reconstruct_field(x, y, 3, discontinuous='continuity')) < 5e-12


def test_a_shape_filling_the_cell_changes_nothing_in_the_sum():
    """Every line is one eps from end to end: ``eps * I / eps``, up to the rule's own arithmetic."""
    s = _solved_layer(cx=0.0, cy=0.0, lx=PX, ly=PY)
    x, y = SAMPLES['walls']
    assert _rel(s.reconstruct_field(x, y, 3, discontinuous='continuity'),
                s.reconstruct_field(x, y, 3)) < 1e-13


def test_single_precision_reads_as_its_raster():
    """On Li, where the two solves agree to 3e-7 in the sum: Laurent's differ by 6e-6 there."""
    shapes, raster = _both((UPPER,), 'li', complex_dtype=1)
    for x, y in SAMPLES.values():
        assert _rel(shapes.reconstruct_field(x, y, 3, discontinuous='continuity'),
                    raster.reconstruct_field(x, y, 3, discontinuous='continuity')) < 2e-6


# ---------------------------------------------------------------- the kernel, on its own


def _lattice():
    return wavebit.Lattice.rectangular(PX, PY)


def _lines(values):
    return np.asarray(values, dtype=float)


def test_the_operators_and_eps_are_the_rasters_along_a_pixel_line():
    bg, *rects = PAINTED
    layer = dict(background=bg, shapes=[_rect(*r[:5]) for r in rects])
    grid = _paint(bg, *rects)
    rows = (np.arange(NY) + 0.5) * DY
    columns = (np.arange(NX) + 0.5) * DX
    paint = read_lines(np, layer, _lattice(), 0, rows, PX, PY, np.complex128)
    np.testing.assert_allclose(line_operators(np, paint, 3), line_inverse_rules(np, grid, 3),
                               atol=1e-13)
    np.testing.assert_array_equal(eps_at(np, paint, columns / PX), grid)
    down = read_lines(np, layer, _lattice(), 1, columns, PY, PX, np.complex128)
    np.testing.assert_allclose(line_operators(np, down, 3), line_inverse_rules(np, grid.T, 3),
                               atol=1e-13)
    np.testing.assert_array_equal(eps_at(np, down, rows / PY), grid.T)


def test_an_outline_crosses_a_line_an_even_number_of_times_through_its_vertices():
    """A line through a vertex takes the edge above it, a line along an edge takes neither end."""
    p = np.array([100.0, 300.0, 300.0, 100.0])
    q = np.array([50.0, 50.0, 200.0, 200.0])
    at, sign = edge_crossings(np, p, q, _lines([50.0, 200.0, 125.0, 50.0 + PY, 125.0 - PY]), PY)
    np.testing.assert_array_equal(np.sum(sign != 0, axis=1), [2, 0, 2, 2, 2])
    np.testing.assert_array_equal(np.sum(sign, axis=1), [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(at[0][sign[0] != 0], [300.0, 100.0])


def test_a_sample_just_before_a_wall_stays_before_it():
    """The lift puts rounding on the positive side, and a real gap of 1e-9 nm stays a gap."""
    s = _rule_on([dict(type='rectangle', cx=0.0, cy=0.0, lx=200.0, ly=180.0, material=4.0)])
    rows = s._continuity_rows(np.array([-100.0 - 1e-9, -100.0, 100.0 - 1e-9, 100.0]),
                                    np.array([0.0]), True)
    np.testing.assert_array_equal(rows[0][2], [[1.0, 4.0, 4.0, 1.0]])


def test_a_sample_exactly_on_a_crossing_is_past_it():
    """Unlifted, so that this is the kernel's own rule and not the lift the engine puts over it."""
    layer = dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=200.0, ly=90.0,
                                              material=4.0)])
    paint = read_lines(np, layer, _lattice(), 0, _lines([180.0]), PX, PY, np.complex128)
    np.testing.assert_array_equal(eps_at(np, paint, np.array([0.25, 0.75])), [[4.0, 1.0]])


def test_the_kernel_paints_a_turned_outline_as_a_brute_painter_does():
    """Slanted edges stay refused at the door, but a rectangle's derivative in ``angle`` is here.

    So the slope term has to be right, and :func:`test_the_angle_derivative_is_the_slope_term`
    relies on this.
    """
    half_x, half_y, angle = 90.0, 40.0, 0.3
    centre = np.array([230.0, 170.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    box = np.array([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]])
    vertices = [tuple(c) for c in box @ rotation.T + centre - [PX / 2, PY / 2]]
    layer = dict(background=1.0, shapes=[dict(type='polygon', material=4.0, vertices=vertices)])
    line = _lines([170.0, 145.0, 215.0, 40.0])
    left, _, after = segments(np, read_lines(np, layer, _lattice(), 0, line, PX, PY, np.complex128))
    n = 2 ** 15
    u = (np.arange(n) + 0.5) / n
    for j, y in enumerate(line):
        along = np.stack([u * PX - centre[0], np.full(n, y) - centre[1]], axis=1) @ rotation
        brute = np.where((np.abs(along[:, 0]) < half_x) & (np.abs(along[:, 1]) < half_y), 0.25, 1.0)
        painted = after[j][np.searchsorted(left[j], u, side='right') - 1]
        assert np.count_nonzero(painted != brute) <= 2      # one sample of error at either wall


def test_the_angle_derivative_is_the_slope_term():
    """``d/d angle`` at a right angle moves a wall by ``(y - cy) d angle``, not rigidly."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _enable_x64()

    def operators(angle):
        layer = dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=150.0,
                                                  ly=90.0, angle=angle, material=4.0)])
        line = jnp.asarray([170.0, 200.0])
        paint = read_lines(jnp, layer, _lattice(), 0, line, PX, PY, jnp.complex128)
        return line_operators(jnp, paint, 2)

    ad = jax.jacfwd(operators)(jnp.asarray(0.0))
    h = 1e-6
    fd = (operators(jnp.asarray(h)) - operators(jnp.asarray(-h))) / (2 * h)
    np.testing.assert_allclose(ad, fd, atol=1e-6)
    assert float(jnp.abs(fd[0] - fd[1]).max()) > 1e-3        # two lines, two different motions


# ---------------------------------------------------------------- walls off every pixel line


def _inside(x, y, shape):
    dx = (x - shape['cx'] + PX / 2) % PX - PX / 2
    dy = (y - shape['cy'] + PY / 2) % PY - PY / 2
    return (np.abs(dx) < shape['lx'] / 2) & (np.abs(dy) < shape['ly'] / 2)


OFF_GRID = (dict(type='rectangle', cx=-37.3, cy=23.1, lx=123.7, ly=91.9, material=4.0),
            dict(type='rectangle', cx=-17.9, cy=-7.7, lx=77.7, ly=61.3, material=9.0 + 0.2j),
            dict(type='rectangle', cx=-PX / 2 + 11.3, cy=-PY / 2 + 7.9, lx=71.1, ly=83.3, material=2.25))


def _eps_of(x, y, shapes):
    eps = np.full(np.broadcast(x, y).shape, 1.5, dtype=complex)
    for shape in shapes:
        eps = np.where(_inside(x, y, shape), shape['material'], eps)
    return eps


@pytest.mark.parametrize('factorization', ['li', 'laurent', 'nvf'])
def test_d_has_no_orders_past_the_truncation(factorization):
    """``D_x`` along a row, and ``D_y`` along a column, are finite series: nothing else is.

    So ``eps E`` on a uniform grid transforms to exactly the kept orders, with the rule dividing
    by its ``eps`` and this multiplying by the test's own painter's. This reads the rule's ``eps``
    at the samples, on any factorization and with nothing to compare against: a wall read in the
    wrong place with a sample inside the error, a wrong side or the wrong shape on top each spill
    past the kept orders. It cannot see the operators, since ``D`` is a finite series whichever
    built it: the raster anchors and RETICOLO hold those. Li's builder takes no shape through a
    cell edge, so it does without the third shape.
    """
    half = 3
    shapes = OFF_GRID[:2] if factorization == 'li' else OFF_GRID
    s = rcwa(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[half, half],
             wavelength=620.0, thickness=[100.0], theta=0.2, phi=0.3, answers='field',
             factorization=factorization, structure=[dict(background=1.5, shapes=list(shapes))],
             **(dict(shape_nvf_grid_size=64) if factorization == 'nvf' else {}))
    s.solve()
    n = 64
    x = np.arange(n) / n * PX - PX / 2 + 0.37
    y = np.arange(n) / n * PY - PY / 2 + 0.53
    cell = np.asarray(s.reconstruct_field(x, y, np.array([50.0]), discontinuous='continuity'))[0, 0]
    kx, ky = s.order_kxy()
    k0 = 2 * np.pi / 620.0
    eps = _eps_of(*np.meshgrid(x, y), shapes)
    # Take the Bloch phase off, so that the series left is periodic on the grid.
    dx = eps * cell[..., 0] * np.exp(-1j * k0 * float(np.real(kx[half])) * x)[None, :]
    dy = eps * cell[..., 1] * np.exp(-1j * k0 * float(np.real(ky[half])) * y)[:, None]
    for d, axis in ((dx, 1), (dy, 0)):
        spectrum = np.abs(np.fft.fft(d, axis=axis)) / n
        order = np.abs(np.fft.fftfreq(n, 1.0 / n))
        kept = order <= half
        far = np.take(spectrum, np.nonzero(~kept)[0], axis=axis).max()
        assert far < 1e-10 * np.take(spectrum, np.nonzero(kept)[0], axis=axis).max()


def test_a_fine_raster_comes_to_the_same_walls():
    """Off the grid, a raster's walls round to pixels, and its gap goes with the pixel size.

    On ``fourier_coefficients='integrated'`` and Laurent, so the modes are of the same rule, and at
    one truncation, so all that changes is where the raster put the walls.
    """
    shape = dict(type='rectangle', cx=-37.3, cy=23.1, lx=123.7, ly=91.9, material=4.0)
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent')
    shapes = rcwa(structure=[dict(background=1.0, shapes=[shape])], **kw)
    shapes.solve()
    x = np.array([-150.0, -20.0, 111.1])
    y = np.array([-130.0, 44.4, 150.0])
    rule = np.asarray(shapes.reconstruct_field(x, y, 3, discontinuous='continuity'))
    gaps = []
    for n in (64, 512):
        cx, cy = (np.arange(n) + 0.5) / n * PX - PX / 2, (np.arange(n) + 0.5) / n * PY - PY / 2
        grid = np.where(_inside(*np.meshgrid(cx, cy), shape), 4.0 + 0j, 1.0 + 0j)
        raster = rcwa(structure=grid[None], fourier_coefficients='integrated', **kw)
        raster.solve()
        gaps.append(_rel(rule, raster.reconstruct_field(x, y, 3, discontinuous='continuity')))
    assert gaps[1] < gaps[0] / 3
    assert gaps[1] < 0.01


def test_a_nudged_eps_moves_the_field_by_a_nudge():
    """A rule that ties on equal values jumps here (fieldalgo/wavebit#129); this one must not."""
    def field(eps):
        shapes = [dict(type='rectangle', cx=-50.0, cy=0.0, lx=100.0, ly=200.0, material=4.0),
                  dict(type='rectangle', cx=50.0, cy=0.0, lx=100.0, ly=200.0, material=eps)]
        s = rcwa(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3],
                 wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent',
                 structure=[dict(background=1.0, shapes=shapes)])
        s.solve()
        return np.asarray(s.reconstruct_field(9, 9, 3, discontinuous='continuity'))

    base = field(4.0)
    assert _rel(field(4.0 + 1e-6), base) < 1e-5
    assert _rel(field(4.0 + 1e-3), base) < 1e-2


@pytest.mark.parametrize('orders', [[2, 4], [1, 4], [5, 2]])
def test_unequal_orders_read_as_a_raster(orders):
    """Each axis's lines take that axis's orders, which equal orders cannot show."""
    shapes, raster = _both((UPPER,), 'li', truncation_order=orders)
    x, y = SAMPLES['walls']
    assert _rel(shapes.reconstruct_field(x, y, 3, discontinuous='continuity'),
                raster.reconstruct_field(x, y, 3, discontinuous='continuity')) < 5e-12


def _ridge(height, *bump):
    """A ridge across the cell along y, with a step on one side, so its long sides differ."""
    return [(-60.0, -height), (60.0, -height), (60.0, -40.0), (90.0, -40.0), (90.0, 40.0),
            (60.0, 40.0), (60.0, height), (-60.0, height), *bump]


@pytest.mark.parametrize('excess', [0.0, 2e-14, 2e-8, 9e-7])
def test_a_ridge_a_period_tall_reads_the_same_when_its_height_rounds_up(excess):
    """``check_layer_shapes`` admits an excess under its tolerance, so the edge is over a period.

    The long edge meets the lines in that band twice, and the two shorter edges facing it once
    each. Counting the long edge once left the winding on past the far wall: 0.27 in ``E_x`` and
    0.48 in ``E_y`` on the count grid's first and last rows, from ``166.66667`` for ``Py / 2``.
    """
    period_y = 1000.0 / 3.0
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, period_y], truncation_order=[3, 3],
              wavelength=620.0, thickness=[100.0], theta=0.2, phi=0.3, answers='field',
              factorization='laurent')

    def field(height):
        s = rcwa(structure=[dict(background=1.0, shapes=[dict(type='polygon', material=4.0,
                                                              vertices=_ridge(height))])], **kw)
        s.solve()
        return s.reconstruct_field(9, 5, 3, discontinuous='continuity')

    assert _rel(field(period_y / 2 * (1 + excess)), field(period_y / 2)) < 1e-5


def test_a_sliver_with_arms_over_a_period_paints_only_the_sliver():
    """Thinner than the overlap tolerance, so it is admitted with an edge 1.5 periods long."""
    w, y0 = 1e-4, -50.0
    vertices = [(0.0, y0), (w, y0), (w, y0 + 0.8 * PY), (2 * w, y0 + 0.8 * PY), (2 * w, y0 + 1.5 * PY),
                (0.0, y0 + 1.5 * PY)]
    layer = dict(background=1.0, shapes=[dict(type='polygon', material=4.0, vertices=vertices)])
    lines = _lines([135.0, 230.0, 330.0, 430.0])
    paint = read_lines(np, layer, _lattice(), 0, lines, PX, PY, np.complex128)
    np.testing.assert_array_equal(np.sum(paint.sign, axis=1), 0.0)
    np.testing.assert_array_equal(eps_at(np, paint, np.array([0.125, 0.375, 0.625, 0.875])),
                                  np.ones((4, 4)))


@pytest.mark.parametrize('period_y', [360.0, 333.3, 1000.0 / 3.0])
def test_an_edge_exactly_a_period_long_meets_every_line(period_y):
    """Compared as positions, a line a rounding under its lower end folded up onto its upper end.

    Then neither wall met it and the whole shape was gone from that row, by a factor of 4 in
    ``E_y``. The lines are the ones around every image of that end.
    """
    low = 80.0
    q = np.array([low, low, low + period_y, low + period_y])
    p = np.array([120.0, 280.0, 280.0, 120.0])
    near = np.concatenate([low + k * period_y + np.array([-1, 1]) * d
                           for k in range(-4, 5) for d in 10.0 ** np.arange(-15.0, -9.0)])
    at, sign = edge_crossings(np, p, q, near, period_y)
    np.testing.assert_array_equal(np.sum(sign != 0, axis=1), 2)
    np.testing.assert_array_equal(np.sum(sign, axis=1), 0)


_RIDGE = [(-60.0, -200.0), (60.0, -200.0), (60.0, -40.0), (90.0, -40.0), (90.0, 40.0), (60.0, 40.0),
          (60.0, 200.0), (-60.0, 200.0)]
_ELL = [(-160.0, -160.0), (120.0, -160.0), (120.0, -40.0), (0.0, -40.0), (0.0, 120.0), (-160.0, 120.0)]


@pytest.mark.parametrize('period,outline', [(400.0, _RIDGE), (400.0, _ELL), (1000 / 3, _ELL),
                                            (333.3, _ELL), (632.893004116449, _RIDGE)])
@pytest.mark.parametrize('axis', [0, 1])
def test_the_winding_steps_of_an_outline_sum_to_zero_on_every_line(period, outline, axis):
    """Each edge folded on its own left steps of 1 here, on 2% of the lines near a vertex image.

    Such a line read the shape on from its one wall through the cell edge, on neither side of any
    wall, and a sum over the crossings cannot see it unless every line is counted.
    """
    scale = period / 400.0
    vertices = np.array(outline) * scale
    p, q = vertices[:, axis], vertices[:, 1 - axis]
    units = np.array([0, 1, 2, 3, 5, 8, 64, 65, 128]) * np.finfo(float).eps * period
    lines = np.concatenate([height + k * period + sign * units for height in np.unique(q)
                            for k in (-1000, -100, -5, -1, 0, 1, 5, 100, 1000) for sign in (-1, 1)])
    _, sign = edge_crossings(np, p, q, lines, period)
    np.testing.assert_array_equal(np.sum(sign, axis=1), 0)
    np.testing.assert_array_equal(np.sum(sign != 0, axis=1) % 2, 0)


def _lines_through(outline, period, y, xs):
    layer = dict(background=1.0, shapes=[dict(type='polygon', material=4.0, vertices=outline)])
    lattice = wavebit.Lattice.rectangular(period, period)
    paint = read_lines(np, layer, lattice, 0, np.atleast_1d(y) + period / 2, period, period, np.complex128)
    return eps_at(np, paint, xs / period + 0.5).real[0]


@pytest.mark.parametrize('turns', [0, 1, 50, 99, 100, 101, 500, 1000])
def test_a_line_on_a_wall_far_from_the_cell_reads_one_side_of_it(turns):
    """At 100 periods out this was neither side, 0.32 and 0.75 from the raster with no error."""
    period = 1000 / 3
    scale = period / 400.0
    outline = [(a * scale, b * scale) for a, b in _ELL]
    xs = np.array([-170.0, -75.0, -30.0, 10.0, 75.0, 150.0]) * scale
    wall = -40.0 * scale + turns * period
    above = np.where((xs > -160 * scale) & (xs < 0.0), 4.0, 1.0)
    below = np.where((xs > -160 * scale) & (xs < 120 * scale), 4.0, 1.0)
    read = _lines_through(outline, period, wall, xs)
    assert np.array_equal(read, above) or np.array_equal(read, below)


def test_a_line_just_under_a_ridge_a_period_tall_keeps_the_ridge():
    """The line misses both walls of a rectangle whose height rounded a unit short of the period."""
    period = 333.3
    cx, cy, lx, ly = 0.05 * period, period / 7, 0.3 * period, period
    outline = [(cx - lx / 2, cy - ly / 2), (cx + lx / 2, cy - ly / 2),
               (cx + lx / 2, cy + ly / 2), (cx - lx / 2, cy + ly / 2)]
    us = np.array([0.1, 0.5, 0.6, 0.7, 0.9]) * period - period / 2
    read = _lines_through(outline, period, 47.61428571428566 - period / 2, us)
    np.testing.assert_array_equal(read, [1.0, 4.0, 4.0, 1.0, 1.0])


@pytest.mark.parametrize('units', [-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
def test_a_ridge_a_period_tall_has_no_gap_of_a_rounding(units):
    """Its height comes out a unit short of this period, and a line in that unit met neither wall.

    Measured without taking the height as whole periods: on 5% of such ridges some line lost
    the shape.
    """
    period, cy = 111.1111111111111, 25.49961788711093
    low, high = cy - period / 2, cy + period / 2
    p = np.array([0.0, 30.0, 30.0, 0.0])
    q = np.array([low, low, high, high])
    lines = np.array([edge + units * np.spacing(period) for edge in (low, high, low - period, high + period)])
    _, sign = edge_crossings(np, p, q, lines, period)
    np.testing.assert_array_equal(np.sum(sign != 0, axis=1), 2)
    np.testing.assert_array_equal(np.sum(sign, axis=1), 0)


def test_a_line_a_rounding_under_a_vertex_image_reads_the_ridge_as_it_is():
    """Measured before: [1, 1, 1, 4, 4, 4, 4, 4], the ridge read on through the cell edge."""
    xs = (np.arange(8) + 0.5) / 8 * 400.0 - 200.0
    read = _lines_through(_RIDGE, 400.0, -800.0000000000001 - 200.0, xs)
    np.testing.assert_array_equal(read, [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------- what must not move


def test_only_the_transverse_e_of_a_patterned_layer_changes():
    """Bit for bit: ``E_z`` and all of ``H``, and every component of a layer with no shapes."""
    s = rcwa(backend='numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[3, 3],
             wavelength=620.0, thickness=[100.0, 60.0], theta=0.3, phi=0.4, answers='field',
             factorization='laurent',
             structure=[dict(background=2.25, shapes=[]),
                        dict(background=1.0, shapes=[_rect(2, 6, 1, 5, 4.0)])])
    s.solve()
    x, y = SAMPLES['walls']
    z = np.array([10.0, 30.0, 110.0, 150.0])
    sum_ = np.asarray(s.reconstruct_field(x, y, z))
    rule = np.asarray(s.reconstruct_field(x, y, z, discontinuous='continuity'))
    np.testing.assert_array_equal(rule[..., :2, :, :, :], sum_[..., :2, :, :, :])
    np.testing.assert_array_equal(rule[..., 2:], sum_[..., 2:])
    assert _rel(rule[..., 2:, :, :, :2], sum_[..., 2:, :, :, :2]) > 0.01


# ---------------------------------------------------------------- backends and derivatives


SCENE_XY = (np.array([-120.0, -40.0, 33.0, 170.0]), np.array([-130.0, -20.0, 70.0]))
KW = dict(top=1.0, bot=1.0, period=[PX, PY], truncation_order=[3, 3], wavelength=620.0,
          thickness=[100.0], theta=0.2, phi=0.3, answers='field')


def _painted(cx=30.0):
    return [dict(background=1.0,
                 shapes=[dict(type='rectangle', cx=cx, cy=-20.0, lx=150.0, ly=90.0, material=4.0),
                         dict(type='rectangle', cx=cx + 40.0, cy=-10.0, lx=60.0, ly=50.0,
                              material=9.0)])]


@pytest.mark.parametrize('factorization', ['li', 'laurent'])
def test_every_backend_and_a_compiled_call_agree(factorization):
    jax = pytest.importorskip('jax')
    pytest.importorskip('torch')
    _enable_x64()
    x, y = SCENE_XY
    reference = None
    for backend in ('numpy', 'torch', 'jax'):
        s = rcwa(backend=backend, structure=_painted(), factorization=factorization, **KW)
        s.solve()
        field = s.reconstruct_field(x, y, 3, discontinuous='continuity')
        if backend == 'numpy':
            reference = field
        assert _rel(np.asarray(field), reference) < 1e-12
    frozen = rcwa(backend='numpy', structure=_painted(), factorization=factorization,
                  **KW).freeze_geometry()

    def compiled(cx):
        s = rcwa(backend='jax', structure=_painted(cx), factorization=factorization, **KW)
        s.solve(geometry=frozen)
        return s.reconstruct_field(x, y, 3, discontinuous='continuity')

    assert _rel(jax.jit(compiled)(30.0), reference) < 1e-12


def _fixed_points_power(field):
    return (field[..., :2].real ** 2 + field[..., :2].imag ** 2).sum()


@pytest.mark.parametrize('parameter', ['cx', 'cy', 'lx', 'ly', 'eps'])
def test_the_derivative_at_fixed_points_is_the_finite_difference(parameter):
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    _enable_x64()
    x, y = SCENE_XY
    start = dict(cx=30.0, cy=-20.0, lx=150.0, ly=90.0, eps=4.0)

    def scene(value):
        chosen = dict(start, **{parameter: value})
        material = chosen.pop('eps')
        return [dict(background=1.0, shapes=[dict(type='rectangle', material=material, **chosen)])]

    frozen = rcwa(backend='numpy', structure=scene(start[parameter]), factorization='laurent',
                  **KW).freeze_geometry()

    def objective(value):
        s = rcwa(backend='jax', structure=scene(value), factorization='laurent', **KW)
        s.solve(geometry=frozen)
        return _fixed_points_power(s.reconstruct_field(x, y, 3, discontinuous='continuity'))

    def plain(value):
        s = rcwa(backend='numpy', structure=scene(value), factorization='laurent', **KW)
        s.solve()
        return _fixed_points_power(np.asarray(s.reconstruct_field(x, y, 3,
                                                                  discontinuous='continuity')))

    ad = float(jax.grad(objective)(start[parameter]))
    h = 1e-4
    assert ad == pytest.approx((plain(start[parameter] + h) - plain(start[parameter] - h)) / (2 * h),
                               rel=1e-5)

    leaf = torch.tensor(start[parameter], dtype=torch.float64, requires_grad=True)
    s = rcwa(backend='torch', structure=scene(leaf), factorization='laurent', **KW)
    s.solve()
    _fixed_points_power(s.reconstruct_field(x, y, 3, discontinuous='continuity')).backward()
    assert float(leaf.grad) == pytest.approx(ad, rel=1e-9)


def test_an_outline_written_in_grad_tracers_is_read_by_value():
    """A list of vertices is how an outline is written, and ``jax.grad`` must reach through it."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    x, y = SCENE_XY

    def scene(wall):
        return [dict(background=1.0, shapes=[dict(
            type='polygon', material=4.0,
            vertices=[(-100.0, -60.0), (wall, -60.0), (wall, 60.0), (-100.0, 60.0)])])]

    frozen = rcwa(backend='numpy', structure=scene(70.0), factorization='laurent',
                  **KW).freeze_geometry()

    def objective(wall):
        s = rcwa(backend='jax', structure=scene(wall), factorization='laurent', **KW)
        s.solve(geometry=frozen)
        return _fixed_points_power(s.reconstruct_field(x, y, 3, discontinuous='continuity'))

    def plain(wall):
        s = rcwa(backend='numpy', structure=scene(wall), factorization='laurent', **KW)
        s.solve()
        return _fixed_points_power(np.asarray(s.reconstruct_field(x, y, 3,
                                                                  discontinuous='continuity')))

    h = 1e-4
    assert float(jax.grad(objective)(70.0)) == pytest.approx(
        (plain(70.0 + h) - plain(70.0 - h)) / (2 * h), rel=1e-5)


def test_the_derivative_of_an_area_objective_misses_the_wall_crossing_a_sample():
    """Stated limit, not a target: ``eps`` at a sample is locally constant in the shape.

    Moving a whole shape cannot change ``|E_x|^2`` averaged over the cell, so the true derivative
    is 0. The Fourier sum's is, to rounding. The rule's derivative holds each point fixed, so it
    leaves out what a wall sweeping across a sample does, and that term is large: ``E_n`` jumps by
    the contrast there. On a rectangle growing in ``lx`` the sign itself comes out wrong
    (+4.2e-4 against -2.7e-4 for a fine finite difference of the mean, at this order).
    ``docs/guide/rcwa/fields.md`` tells an area objective to take ``'fourier'``.
    """
    jax = pytest.importorskip('jax')
    _enable_x64()
    n = 32
    grid = (np.arange(n) + 0.5) / n * 400.0 - 200.0
    kw = dict(top=1.0, bot=1.0, period=[400.0, 400.0], wavelength=900.0, thickness=[150.0],
              answers='field', truncation_order=[3, 3], factorization='li')

    def scene(cx):
        return [dict(background=1.0, shapes=[dict(type='rectangle', cx=cx, cy=0.0, lx=200.0,
                                                  ly=200.0, material=12.25)])]

    frozen = rcwa(backend='numpy', structure=scene(13.0), **kw).freeze_geometry()

    def mean(cx, rule):
        s = rcwa(backend='jax', structure=scene(cx), **kw)
        s.solve(geometry=frozen)
        field = s.reconstruct_field(grid, grid, np.array([75.0]), discontinuous=rule)
        return (field[..., 0].real ** 2 + field[..., 0].imag ** 2).mean()

    assert abs(float(jax.grad(mean)(13.0, 'fourier'))) < 1e-14
    assert abs(float(jax.grad(mean)(13.0, 'continuity'))) > 1e-6


# ---------------------------------------------------------------- refusals


def _rule_on(shapes, **over):
    kw = dict(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[2, 2],
              wavelength=620.0, thickness=[100.0], answers='field', factorization='laurent')
    kw.update(over)
    s = rcwa(structure=[dict(background=1.0, shapes=shapes)], **kw)
    s.solve()
    return s


SLANTED = {
    'circle': dict(type='circle', cx=0.0, cy=0.0, lx=100.0, material=4.0),
    'ellipse': dict(type='ellipse', cx=0.0, cy=0.0, lx=100.0, ly=60.0, material=4.0),
    'rectangle': dict(type='rectangle', cx=0.0, cy=0.0, lx=100.0, ly=60.0, angle=0.3, material=4.0),
    'diamond': dict(type='polygon', vertices=[(0.0, -50.0), (50.0, 0.0), (0.0, 50.0), (-50.0, 0.0)],
                    material=4.0),
}


@pytest.mark.parametrize('name', list(SLANTED))
def test_a_slanted_or_curved_wall_is_refused_and_the_sum_is_not(name):
    s = _rule_on([SLANTED[name]])
    s.reconstruct_field(3, 3, 3)
    with pytest.raises(NotImplementedError, match='slanted or curved wall'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


@pytest.mark.parametrize('shape', [
    dict(type='rectangle', cx=0.0, cy=0.0, lx=100.0, ly=60.0, angle=1e-6, material=4.0),
    dict(type='polygon', material=4.0,
         vertices=[(0.0, 0.0), (100.0, 1e-6), (100.0, 100.0), (0.0, 100.0)])])
def test_the_smallest_slant_past_rounding_is_refused(shape):
    s = _rule_on([shape])
    with pytest.raises(NotImplementedError, match='slanted or curved wall'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


def test_a_traced_angle_cannot_show_its_walls_and_is_refused():
    jax = pytest.importorskip('jax')
    _enable_x64()
    kw = dict(top=1.0, bot=1.0, period=[PX, PY], truncation_order=[2, 2], wavelength=620.0,
              thickness=[100.0], answers='field', factorization='laurent')

    def scene(angle):
        return [dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=100.0,
                                                  ly=60.0, angle=angle, material=4.0)])]

    frozen = rcwa(backend='numpy', structure=scene(0.0), **kw).freeze_geometry()

    def field(angle):
        s = rcwa(backend='jax', structure=scene(angle), **kw)
        s.solve(geometry=frozen)
        return s.reconstruct_field(3, 3, 3, discontinuous='continuity')

    with pytest.raises(NotImplementedError, match='slanted or curved wall'):
        jax.jit(field)(0.0)


def test_traced_vertices_cannot_show_their_walls_and_are_refused():
    jax = pytest.importorskip('jax')
    _enable_x64()
    kw = dict(top=1.0, bot=1.0, period=[PX, PY], truncation_order=[2, 2], wavelength=620.0,
              thickness=[100.0], answers='field', factorization='laurent')

    def scene(wall):
        return [dict(background=1.0, shapes=[dict(
            type='polygon', material=4.0,
            vertices=[(-100.0, -60.0), (wall, -60.0), (wall, 60.0), (-100.0, 60.0)])])]

    frozen = rcwa(backend='numpy', structure=scene(70.0), **kw).freeze_geometry()

    def field(wall):
        s = rcwa(backend='jax', structure=scene(wall), **kw)
        s.solve(geometry=frozen)
        return s.reconstruct_field(3, 3, 3, discontinuous='continuity')

    with pytest.raises(NotImplementedError, match='slanted or curved wall'):
        jax.jit(field)(70.0)


@pytest.mark.parametrize('truncation', [[2, 0], [0, 2]])
def test_one_order_along_an_axis_is_refused(truncation):
    """Under Laurent ``period=[400]`` with ``ly=100`` and eps 4 is solved as ``ly=400``, eps 1.75."""
    s = _rule_on([_rect(2, 6, 1, 5, 4.0)], truncation_order=truncation)
    with pytest.raises(NotImplementedError, match='one order along an axis'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


@pytest.mark.parametrize('columns', [[[-PX, 0.0], [0.0, PY]], [[PX, 0.0], [0.0, -PY]]])
def test_a_negative_side_of_the_lattice_is_refused(columns):
    kw = dict(backend='numpy', top=1.0, bot=1.0, truncation_order=[2, 2], wavelength=620.0,
              thickness=[100.0], answers='field', factorization='laurent',
              lattice=np.asarray(columns))
    s = rcwa(structure=[dict(background=1.0, shapes=[_rect(2, 6, 1, 5, 4.0)])], **kw)
    s.solve()
    with pytest.raises(NotImplementedError, match='positive sides'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


def test_a_tensor_material_in_a_shape_is_refused():
    tensor = wavebit.Material(np.diag([4.0, 4.0, 9.0]).astype(complex))
    s = _rule_on([_rect(2, 6, 1, 5, tensor)], factorization='nvf', shape_nvf_grid_size=32)
    with pytest.raises(NotImplementedError, match='tensor material'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


def test_a_solve_that_merged_a_layer_into_a_half_space_is_refused():
    s = rcwa(backend='numpy', top=1.0, bot=2.25, period=[PX, PY], truncation_order=[2, 2],
             wavelength=620.0, thickness=[100.0, 80.0], answers='field', factorization='laurent',
             half_space_merge_tolerance=1e-8,
             structure=[dict(background=2.25 * (1 + 3e-9), shapes=[]),
                        dict(background=1.0, shapes=[_rect(2, 6, 1, 5, 4.0)])])
    s.solve(auto_detune=False)
    assert s._merged_field_state is not None
    with pytest.raises(NotImplementedError, match='merged layers'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


def test_the_rule_reads_the_shapes_the_solve_was_posed_with():
    s = _rule_on([_rect(2, 6, 1, 5, 4.0)])
    before = np.asarray(s.reconstruct_field(3, 3, 3, discontinuous='continuity'))

    s.structure[0]['shapes'][0]['lx'] = 300.0
    np.testing.assert_array_equal(
        np.asarray(s.reconstruct_field(3, 3, 3, discontinuous='continuity')), before)
    s.solve()
    moved = np.asarray(s.reconstruct_field(3, 3, 3, discontinuous='continuity'))
    assert np.max(np.abs(moved - before)) > 1e-2

    assigned = _rule_on([_rect(2, 6, 1, 5, wavebit.Material(4.0))])
    assigned.structure = [dict(background=1.0, shapes=[_rect(2, 6, 1, 5, wavebit.Material(9.0))])]
    np.testing.assert_array_equal(
        np.asarray(assigned.reconstruct_field(3, 3, 3, discontinuous='continuity')), before)


def test_an_array_inside_a_material_is_copied_with_the_structure():
    held = np.array(4.0)
    s = _rule_on([_rect(2, 6, 1, 5, wavebit.Material(held))])
    before = np.asarray(s.reconstruct_field(3, 3, 3, discontinuous='continuity'))
    held += 5.0
    np.testing.assert_array_equal(
        np.asarray(s.reconstruct_field(3, 3, 3, discontinuous='continuity')), before)


def test_the_copy_counts_as_solve_state():
    """The scope that forgets a traced solve, and auto_detune's derivative check, read conditions.

    A leaf in the copy of an earlier solve is not a condition of this one.
    """
    from wavebit.rcwa._engine.engine import _SOLVE_STATE
    assert '_solved_structure' in _SOLVE_STATE


def test_the_gate_asks_about_the_shapes_the_solve_was_posed_with():
    circle = [dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=120.0,
                                                material=4.0)])]
    posed_on_a_rectangle = _rule_on([_rect(2, 6, 1, 5, 4.0)])
    before = np.asarray(posed_on_a_rectangle.reconstruct_field(3, 3, 3, discontinuous='continuity'))
    posed_on_a_rectangle.structure = circle
    np.testing.assert_array_equal(
        np.asarray(posed_on_a_rectangle.reconstruct_field(3, 3, 3, discontinuous='continuity')), before)

    posed_on_a_circle = rcwa(backend='numpy', top=1.0, bot=1.0, period=[PX, PY], truncation_order=[2, 2],
                             wavelength=620.0, thickness=[100.0], answers='field',
                             factorization='laurent', structure=circle)
    posed_on_a_circle.solve()
    posed_on_a_circle.structure = [dict(background=1.0, shapes=[_rect(2, 6, 1, 5, 4.0)])]
    with pytest.raises(NotImplementedError, match='slanted or curved'):
        posed_on_a_circle.reconstruct_field(3, 3, 3, discontinuous='continuity')


def _rectangle_scene(lx=200.0, cx=0.0):
    return [dict(background=1.0, shapes=[dict(type='rectangle', cx=cx, cy=0.0, lx=lx, ly=90.0,
                                              material=4.0)])]


_LASER = dict(top=1.0, bot=1.0, period=[PX, PY], truncation_order=[2, 2], wavelength=620.0,
              thickness=[100.0], answers='field', factorization='laurent')


def test_a_read_has_the_derivative_its_solve_had():
    """Modes without one under operators with one read 5.8 times too large, with no error."""
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(200.0, dtype=torch.float64, requires_grad=True)
    s = rcwa(backend='torch', structure=_rectangle_scene(leaf), **_LASER)
    with torch.no_grad():
        s.solve()
    assert not s.reconstruct_field(3, 3, 3, discontinuous='continuity').requires_grad

    s.solve()
    assert s.reconstruct_field(3, 3, 3, discontinuous='continuity').requires_grad
    with torch.no_grad():
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')

    plain = rcwa(backend='torch', structure=_rectangle_scene(200.0), **_LASER)
    plain.solve()
    plain.structure = _rectangle_scene(leaf)
    assert not plain.reconstruct_field(3, 3, 3, discontinuous='continuity').requires_grad


def test_a_leaf_stepped_in_place_after_the_solve_is_not_read():
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(200.0, dtype=torch.float64, requires_grad=True)
    s = rcwa(backend='torch', structure=_rectangle_scene(leaf), **_LASER)
    s.solve()
    before = s.reconstruct_field(3, 3, 3, discontinuous='continuity').detach().numpy()
    with torch.no_grad():
        leaf += 40.0
    after = s.reconstruct_field(3, 3, 3, discontinuous='continuity').detach().numpy()
    np.testing.assert_array_equal(after, before)


def test_a_grad_over_an_eager_solve_has_nothing_to_differentiate():
    """As the Fourier sum: the read answers the solved scene, whatever is assigned over it."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    s = rcwa(backend='jax', structure=_rectangle_scene(200.0), **_LASER)
    s.solve()

    def objective(lx):
        s.structure = _rectangle_scene(lx)
        return jax.numpy.sum(jax.numpy.abs(s.reconstruct_field(3, 3, 3, discontinuous='continuity')))

    assert float(jax.grad(objective)(200.0)) == 0.0


def _power(field):
    import jax.numpy as jnp
    return jnp.sum(jnp.abs(field[..., :2]) ** 2)


_OFF_WALLS = (np.array([-120.0, -40.0, 33.0, 170.0]), np.array([-130.0, -20.0, 70.0]), 3)


def _rule_gradient_against_a_finite_difference():
    """``dJ/dcx`` of a numpy solve at 30 by central difference, for the traced cases below."""
    def value(cx):
        s = rcwa(backend='numpy', structure=_rectangle_scene(150.0, cx), **_LASER)
        s.solve()
        return float(np.sum(np.abs(np.asarray(s.reconstruct_field(
            *_OFF_WALLS, discontinuous='continuity'))[..., :2]) ** 2))
    return (value(30.0 + 1e-4) - value(30.0 - 1e-4)) / 2e-4


@pytest.mark.parametrize('how', ['put back by the solve', 'assigned equal', 'assigned back'])
def test_a_solve_with_a_derivative_reads_with_it_whatever_is_assigned_after(how):
    """An equal concrete structure over traced modes lost the operators' term, 1.6 to 9.3% off."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    frozen = rcwa(backend='numpy', structure=_rectangle_scene(150.0, 30.0), **_LASER).freeze_geometry()

    def objective(cx):
        if how == 'put back by the solve':
            s = rcwa(backend='jax', structure=_rectangle_scene(150.0, 30.0),
                     **dict(_LASER, wavelength=620.0 + 0.0 * cx))
            s.solve(structure=_rectangle_scene(150.0, cx), geometry=frozen)
        else:
            s = rcwa(backend='jax', structure=_rectangle_scene(150.0, cx), **_LASER)
            s.solve(geometry=frozen)
            s.structure = (_rectangle_scene(150.0, 30.0) if how == 'assigned equal'
                           else _rectangle_scene(150.0, cx))
        return _power(s.reconstruct_field(*_OFF_WALLS, discontinuous='continuity'))

    assert float(jax.grad(objective)(30.0)) == pytest.approx(
        _rule_gradient_against_a_finite_difference(), rel=1e-6)


def test_a_structure_edited_after_a_solve_inside_a_trace_is_not_read():
    jax = pytest.importorskip('jax')
    _enable_x64()
    frozen = rcwa(backend='numpy', structure=_rectangle_scene(), **_LASER).freeze_geometry()

    def edited(lx):
        s = rcwa(backend='jax', structure=_rectangle_scene(lx), **_LASER)
        s.solve(geometry=frozen)
        s.structure = _rectangle_scene(lx + 20.0)
        return s.reconstruct_field(3, 3, 3, discontinuous='continuity')

    eager = rcwa(backend='numpy', structure=_rectangle_scene(), **_LASER)
    eager.solve()
    np.testing.assert_allclose(np.asarray(jax.jit(edited)(200.0)),
                               np.asarray(eager.reconstruct_field(3, 3, 3, discontinuous='continuity')),
                               rtol=0, atol=1e-11)


def test_a_fit_whose_loss_takes_the_rule_steps_along_its_gradient():
    """``fit`` solves with the structure traced, so the loss reads the traced copy.

    Before the copy the first step's ``dJ/dlx`` read -0.0249 where a finite difference reads +0.0052.
    """
    pytest.importorskip('jax')
    optax = pytest.importorskip('optax')
    import jax.numpy as jnp
    _enable_x64()
    kw = dict(top=1.0, bot=2.25, period=[PX, PY], truncation_order=[2, 2], wavelength=620.0,
              thickness=[150.0], answers='field', factorization='laurent')

    def scene(lx):
        return [dict(background=1.0, shapes=[dict(type='rectangle', cx=10.0, cy=-5.0, lx=lx, ly=160.0,
                                                  material=4.0)])]

    def value(lx):
        s = rcwa(backend='numpy', structure=scene(lx), **kw)
        s.solve()
        return float(np.sum(np.abs(np.asarray(s.reconstruct_field(
            3, 3, 3, discontinuous='continuity'))) ** 2))

    s = rcwa(backend='jax', structure=scene(200.0), **kw)

    def loss(result):
        return jnp.sum(jnp.abs(s.reconstruct_field(3, 3, 3, discontinuous='continuity')) ** 2)

    learned = s.fit({'lx': jnp.asarray(200.0)}, s.solve, loss, optax.sgd(1.0), iteration=1,
                    parameterize=lambda lx: {'structure': scene(lx)})
    step = 200.0 - float(np.asarray(learned[0]['lx'] if isinstance(learned, tuple) else learned['lx']))
    assert step == pytest.approx((value(200.0 + 1e-4) - value(200.0 - 1e-4)) / 2e-4, rel=1e-5)


def test_a_negative_side_under_a_trace_answers_nan_where_the_rule_reads():
    """A trace has no sign to refuse, and the rule's lines are wrong there: E_x off by 0.92."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    kw = dict(_LASER, factorization='laurent')
    del kw['period']
    frozen = rcwa(backend='numpy', structure=_rectangle_scene(150.0, 30.0), period=[PX, PY],
                  **kw).freeze_geometry()

    def field(px):
        s = rcwa(backend='jax', structure=_rectangle_scene(150.0, 30.0), period=[px, PY], **kw)
        s.solve(geometry=frozen)
        return s.reconstruct_field(*_OFF_WALLS, discontinuous='continuity')

    negative = np.asarray(jax.jit(field)(-PX))
    assert np.all(np.isnan(negative[..., :2]))
    assert np.all(np.isfinite(negative[..., 2:]))
    positive = np.asarray(jax.jit(field)(PX))
    assert np.all(np.isfinite(positive))


def test_matrices_from_solve_conv_are_refused_on_shapes():
    s = _rule_on([_rect(2, 6, 1, 5, 4.0)])
    s.solve_conv(*s.eps_conv_matrices())
    with pytest.raises(NotImplementedError, match='solve_conv'):
        s.reconstruct_field(3, 3, 3, discontinuous='continuity')


def test_matrices_after_a_solve_are_refused_on_what_that_solve_returns():
    """The solver still holds the solve before them while they run, and the result they return must not."""
    s = _rule_on([_rect(2, 6, 1, 5, 4.0)])
    matrices = s.eps_conv_matrices()
    s.solve()
    state = s.solve_conv(*matrices)._field_state
    with pytest.raises(NotImplementedError, match='solve_conv'):
        state.reconstruct(3, 3, 3, 'incident', 'continuity')


def _rectangle_with(spelling):
    return [dict(background=1.0, shapes=spelling), dict(background=2.25, shapes=[])]


_A_RECTANGLE = dict(type='rectangle', cx=30.0, cy=-20.0, lx=150.0, ly=90.0, material=4.0)
_AT = dict(sample_x=[-120.0, -45.0, 33.0, 105.0, 170.0], sample_y=[-130.0, -65.0, 25.0, 70.0],
           sample_z=[10.0, 50.0, 120.0], discontinuous='continuity')


def _copy_solver(structure):
    return rcwa(backend='numpy', period=[PX, PY], wavelength=620.0, truncation_order=[2, 2],
                thickness=[100.0, 60.0], theta=0.2, phi=0.3, polarization=(0.0, 1.0), answers='field',
                factorization='laurent', structure=structure)


def test_a_namedtuple_in_a_structure_still_solves_and_reads():
    """The copy handed a generator to its constructor, on every field solve, rule or no rule."""
    point = namedtuple('point', 'x y')
    outline = namedtuple('outline', 'a b c d')
    spellings = (
        [point(-100.0, -80.0), point(100.0, -80.0), point(100.0, 80.0), point(-100.0, 80.0)],
        outline((-100.0, -80.0), (100.0, -80.0), (100.0, 80.0), (-100.0, 80.0)))
    plain = _copy_solver(_rectangle_with([dict(type='polygon', material=4.0,
                                               vertices=[(-100.0, -80.0), (100.0, -80.0), (100.0, 80.0), (-100.0, 80.0)])]))
    plain.solve()
    expected = plain.reconstruct_field(**_AT)
    for vertices in spellings:
        s = _copy_solver(_rectangle_with([dict(type='polygon', material=4.0, vertices=vertices)]))
        s.solve()
        np.testing.assert_array_equal(s.reconstruct_field(**_AT), expected)


class _Live:
    """An array-like a shape may be written with, whose value changes after the solve."""

    def __init__(self, value):
        self.value = value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def __float__(self):
        return float(self.value)


@pytest.mark.parametrize('spell, edit', [
    (lambda shape: [MappingProxyType(shape)], lambda shape, s: shape.update(cx=99.0)),
    (lambda shape: deque([shape]), lambda shape, s: shape.update(cx=99.0)),
    (lambda shape: deque([shape]), lambda shape, s: s.append(dict(shape, cx=-99.0))),
], ids=['read-only mapping', 'deque edited', 'deque appended'])
def test_no_spelling_of_a_container_reaches_the_copy(spell, edit):
    """Measured before: these moved the rule by 0.68 with the sum at 0, on the door that reads the copy."""
    shape = dict(_A_RECTANGLE)
    shapes = spell(shape)
    s = _copy_solver(_rectangle_with(shapes))
    s.solve()
    before = s.reconstruct_field(**_AT)
    edit(shape, shapes)
    np.testing.assert_array_equal(s.reconstruct_field(**_AT), before)


def test_an_array_like_is_copied_with_the_structure():
    live = _Live(30.0)
    copy = solved_structure([dict(shapes=[dict(_A_RECTANGLE, cx=live)])])
    live.value = 99.0
    assert float(copy[0]['shapes'][0]['cx']) == 30.0


def test_a_lattice_is_copied_with_its_arrays():
    u, v = np.array([PX, 0.0]), np.array([0.0, PY])
    copy = solved_structure(Lattice(u, v, True))
    u[0] = 999.0
    assert isinstance(copy, Lattice) and copy.axis_aligned
    assert copy.columns[0, 0] == PX


def test_asked_before_any_solve_the_rule_says_to_solve():
    s = _copy_solver(_rectangle_with([dict(_A_RECTANGLE)]))
    with pytest.raises(RuntimeError, match='run solve'):
        s.reconstruct_field(**_AT)


def test_a_solve_that_kept_no_field_state_is_not_called_matrices():
    s = _copy_solver(_rectangle_with([dict(_A_RECTANGLE)]))
    s.answers = 'amplitude'
    s.solve()
    s.answers = 'field'
    with pytest.raises(RuntimeError, match='per-layer amplitudes'):
        s.reconstruct_field(**_AT)


@pytest.mark.parametrize('through', [copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))],
                         ids=['deepcopy', 'pickle'])
def test_a_copied_solver_still_refuses_matrices_by_name(through):
    s = _copy_solver(_rectangle_with([dict(_A_RECTANGLE)]))
    s.solve_conv(*s.eps_conv_matrices())
    with pytest.raises(NotImplementedError, match='solve_conv'):
        through(s).reconstruct_field(**_AT)


class _Square(Lattice):
    def __init__(self, side):
        super().__init__(np.array([side, 0.0]), np.array([0.0, side]), True)


class _Glass(wavebit.Material):
    def __init__(self, index=1.5):
        super().__init__(index ** 2)


def test_a_subclass_with_its_own_constructor_still_solves():
    """The copy went through ``dataclasses.replace``, on every field solve, rule or no rule."""
    kw = dict(backend='numpy', wavelength=620.0, truncation_order=[2, 2], thickness=[100.0, 60.0],
              theta=0.2, phi=0.3, polarization=(0.0, 1.0), answers='field', factorization='laurent')
    glass = [dict(background=1.0, shapes=[dict(_A_RECTANGLE, material=_Glass(2.0))]), dict(background=2.25, shapes=[])]
    plain = rcwa(lattice=np.diag([400.0, 400.0]), structure=_rectangle_with([dict(_A_RECTANGLE)]), **kw)
    plain.solve()
    subclassed = rcwa(lattice=_Square(400.0), structure=glass, **kw)
    subclassed.solve()
    np.testing.assert_array_equal(subclassed.reconstruct_field(**_AT), plain.reconstruct_field(**_AT))


class _ArrayLike:
    """What a pandas or xarray scalar shows: a shape, a dtype and ``__array__``, and no array namespace."""
    shape, dtype = (), np.dtype(float)

    def __init__(self, value):
        self.value = value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def __float__(self):
        return float(self.value)

    def clone(self):
        raise RuntimeError('this clone is not a tensor copy')


def test_an_array_of_another_library_is_read_and_copied_through_its_array():
    s = _copy_solver(_rectangle_with([dict(_A_RECTANGLE, material=_ArrayLike(4.0))]))
    s.solve()
    plain = _copy_solver(_rectangle_with([dict(_A_RECTANGLE)]))
    plain.solve()
    np.testing.assert_array_equal(s.reconstruct_field(**_AT), plain.reconstruct_field(**_AT))


def test_an_array_with_nothing_but_its_array_is_copied_through_it():
    class Only:
        def __init__(self, value):
            self.value = value

        def __array__(self, dtype=None, copy=None):
            return np.array(self.value, dtype=dtype, copy=copy)

    values = np.array([1.0, 2.0])
    copied = solved_structure(Only(values))
    values[0] = 9.0
    np.testing.assert_array_equal(copied, [1.0, 2.0])


class _FloatOnly:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


def test_what_reads_as_a_number_is_copied_as_that_number():
    live = _FloatOnly(4.0)
    copied = solved_structure([dict(material=live, exact=Fraction(1, 2), name=np.str_('rectangle'), count=3)])
    live.value = 9.0
    assert copied[0]['material'] == 4.0 and copied[0]['exact'] == 0.5 and copied[0]['count'] == 3
    assert isinstance(copied[0]['name'], str)


def _jax_result(shape):
    return rcwa(backend='jax', structure=_rectangle_with([shape]), period=[PX, PY], wavelength=620.0,
                truncation_order=[2, 2], thickness=[100.0, 60.0], theta=0.2, phi=0.3, answers='field',
                factorization='laurent').solve()


def test_a_result_that_went_through_a_pytree_can_still_be_pickled():
    """The pytree's read-only mappings could not, and a Result from a jit is one of those."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    result = jax.tree_util.tree_map(lambda leaf: leaf, _jax_result(dict(_A_RECTANGLE)))
    expected = np.asarray(result.reconstruct_field(**_AT))
    for through in (copy.deepcopy, lambda value: pickle.loads(pickle.dumps(jax.device_get(value)))):
        np.testing.assert_allclose(np.asarray(through(result).reconstruct_field(**_AT)), expected, rtol=0, atol=1e-13)


def test_a_result_holds_only_leaves_jax_can_carry():
    """An ``np.str_`` was a string array, and a ``Fraction`` a number no transform can take."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    odd = rcwa(backend='jax', structure=[dict(background=Fraction(1), shapes=[dict(_A_RECTANGLE, type=np.str_('rectangle'))]),
                                         dict(background=2.25, shapes=[])],
               period=[PX, PY], wavelength=620.0, truncation_order=[2, 2], thickness=[100.0, 60.0], theta=0.2, phi=0.3,
               answers='field', factorization='laurent').solve()
    expected = np.asarray(odd.reconstruct_field(**_AT))
    jax.tree_util.tree_map(lambda leaf: leaf, odd)
    jax.jit(lambda result: result.R)(odd)
    plain = _jax_result(dict(_A_RECTANGLE))
    np.testing.assert_allclose(expected, np.asarray(plain.reconstruct_field(**_AT)), rtol=0, atol=1e-14)


def test_the_order_of_a_dicts_keys_is_not_part_of_a_results_pytree():
    jax = pytest.importorskip('jax')
    _enable_x64()
    one = _jax_result(dict(_A_RECTANGLE))
    other = _jax_result(dict(reversed(list(_A_RECTANGLE.items()))))
    assert jax.tree_util.tree_structure(one) == jax.tree_util.tree_structure(other)


def test_a_negative_side_under_a_trace_answers_nan_on_a_raster_too():
    """Counts run down such a cell and the pixels are read up it; a trace cannot tell them apart."""
    jax = pytest.importorskip('jax')
    _enable_x64()
    grid = np.ones((2, 9, 8))
    grid[0, 2:7, 3:6] = 4.0
    grid[1] = 2.25
    kw = dict(backend='jax', structure=grid, wavelength=620.0, truncation_order=[2, 2], thickness=[100.0, 60.0],
              theta=0.2, phi=0.3, polarization=(0.0, 1.0), answers='field')

    def field(side):
        out = rcwa(lattice=Lattice(jax.numpy.stack([side, 0.0]), jax.numpy.array([0.0, PY]), True), **kw).solve()
        return out.reconstruct_field(5, 4, 3, discontinuous='continuity')

    negative = np.asarray(jax.jit(field)(-PX))
    assert np.all(np.isnan(negative[..., :2][:3])) and np.all(np.isfinite(negative[..., 2:]))
    assert np.all(np.isfinite(np.asarray(jax.jit(field)(PX))))


def test_a_negative_side_under_a_trace_answers_nan_on_a_one_row_raster_too():
    jax = pytest.importorskip('jax')
    _enable_x64()
    grid = np.ones((2, 1, 8))
    grid[0, 0, 3:6] = 4.0
    grid[1] = 2.25
    kw = dict(backend='jax', structure=grid, wavelength=620.0, truncation_order=[3, 0], thickness=[100.0, 60.0],
              theta=0.2, phi=0.0, polarization=(0.0, 1.0), answers='field')

    def field(side):
        out = rcwa(lattice=Lattice(jax.numpy.stack([side, 0.0]), jax.numpy.array([0.0, PY]), True), **kw).solve()
        return out.reconstruct_field(5, 1, 3, discontinuous='continuity')

    negative = np.asarray(jax.jit(field)(-PX))
    assert np.all(np.isnan(negative[..., 0][:3])) and np.all(np.isfinite(negative[..., 2]))
    assert np.all(np.isfinite(np.asarray(jax.jit(field)(PX))))
