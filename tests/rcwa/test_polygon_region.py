"""``rcwa.Polygon``: a region whose outline is a list of vertices, solved on the painting route.

It was declared and refused until the region could be built (develop 0cd24a45). What these
tests hold it to, beside the S4 identity in ``tests/references/test_s4_polygon_golden.py``:

* the outline is checked where its values are -- at the declaration when they are concrete,
  and again at the layer funnel, which a traced declaration reaches with values later;
* a rectangle written as its four corners *is* the rectangle: the same transform to rounding,
  the same painting-route matrices, and under NVF the same limit;
* either winding is the same region, bit for bit;
* the image test is exact on outlines, so a non-convex outline wider than the period that
  threads between its own images is built, and one that reaches into a copy of itself, or into
  another region's copy, is refused;
* a frozen plan replays on traced vertices and goes stale when the painting changes;
* the vertex derivative is the solve's derivative, window-integrated, under Laurent's rule and
  under NVF -- under NVF since the painting route's normal field averages its medial-axis
  switches over the sample spacing (it missed 11% of this window before).
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError, ValidationError

#: Non-convex: the reflex corner at (-0.1, -0.1).
L_SHAPE = ((-.35, -.35), (.3, -.35), (.3, -.1), (-.1, -.1), (-.1, .3), (-.35, .3))
TRIANGLE = ((-.3, -.25), (.32, -.2), (.05, .33))
_C, _S = np.cos(.2), np.sin(.2)
#: The L turned by 0.2 rad, so no edge is axis-aligned. An axis-aligned edge has a fractional
#: extent of exactly zero, which the interface planner rounds to a whole cell, so the first
#: move off the axis re-plans its image bounds; a tilted outline can be moved and replayed.
L_TILTED = tuple(tuple(map(float, p)) for p in np.asarray(L_SHAPE) @ np.array([[_C, _S], [-_S, _C]]))
LAURENT = rcwa.Numerics(orders=(3, 3), factorization='laurent')
NVF = rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=128)


def region(geometry, epsilon=6.25):
    if not isinstance(geometry, (rcwa.Rectangle, rcwa.Circle, rcwa.Ellipse, rcwa.Polygon)):
        geometry = rcwa.Polygon(vertices=geometry)
    return rcwa.Region(geometry=geometry, material=Material(epsilon))


def problem(*regions, lattice=None, polarization='s', theta=.35, phi=.6):
    layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=regions)
    cell = dict(period=(1., 1.)) if lattice is None else dict(lattice=lattice)
    return rcwa.Problem(stack=rcwa.Stack(layers=(layer,), bottom=Material(2.25), **cell),
                        excitation=rcwa.Excitation(wavelength=1.3, sources=(
                            rcwa.PlaneWave(theta=theta, phi=phi, polarization=polarization),)))


def efficiencies(result):
    return np.asarray(result.R)[0], np.asarray(result.T)[0]


def gap(a, b):
    return max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, b))


# ------------------------------------------------------------------------------ declaration

def test_vertices_are_owned_and_kept_in_the_order_given():
    vertices = [list(p) for p in L_SHAPE]
    polygon = rcwa.Polygon(vertices=vertices)
    vertices[0][0] = 9.
    assert polygon.vertices == L_SHAPE
    # Clockwise is a region too, and it is not rewritten: the planner reads the winding.
    assert rcwa.Polygon(vertices=L_SHAPE[::-1]).vertices == L_SHAPE[::-1]


BOW_TIE = ((-.3, -.3), (.3, .3), (.3, -.3), (-.3, .3))


@pytest.mark.parametrize('vertices,code,says', [
    (((0., 0.), (.2, 0.)), 'shape.vertices', 'at least three'),
    (((0., 0., 0.), (.2, 0., 0.), (0., .2, 0.)), 'shape.vertices', '(x, y) pairs'),
    (np.zeros((4, 3)), 'shape.vertices', 'got shape (4, 3)'),
    (.3, 'shape.vertices', 'got float'),
    (((0., 0.), (.2 + 1j, 0.), (0., .2)), 'shape.vertices', 'must be real'),
    (((0., 0.), (np.nan, 0.), (0., .2)), 'shape.outline', 'not a finite point'),
    (((0., 0.), (.2, 0.), (.2, 0.), (0., .2)), 'shape.outline', 'zero-length edge'),
    (BOW_TIE, 'shape.outline', 'cross or touch'),
    # pinched: vertex 3 lands on edge 0, so the outline touches itself at a point
    (((-.3, -.3), (.3, -.3), (.3, .3), (0., -.3), (-.3, .3)), 'shape.outline', 'cross or touch'),
    (((0., 0.), (.3, 0.), (.1, 0.), (0., .2)), 'shape.outline', 'doubles back'),
    (((0., 0.), (.1, .1), (.2, .2)), 'shape.outline', 'doubles back'),
])
def test_an_outline_that_is_not_a_simple_polygon_is_refused_where_it_is_declared(vertices, code,
                                                                                   says):
    with pytest.raises(ValidationError) as error:
        rcwa.Polygon(vertices=vertices)
    assert (error.value.issue.code, error.value.issue.path) == (code, 'geometry.vertices')
    assert says in str(error.value)


def test_a_traced_outline_is_checked_once_it_has_a_value():
    """Under ``jit`` there is no value and only the layout is read; ``grad`` has its primal."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    bow_tie = jnp.asarray(BOW_TIE)
    # no number under jit: the declaration takes it, and the layout is still checked
    assert float(jax.jit(lambda v: jnp.sum(rcwa.Polygon(vertices=v).vertices))(bow_tie)) == 0.
    with pytest.raises(ValidationError, match='got shape'):
        jax.jit(lambda v: jnp.sum(rcwa.Polygon(vertices=v[:, :1]).vertices))(bow_tie)
    # a grad tracer carries its primal, and the bow tie is refused on it
    with pytest.raises(ValidationError, match='cross or touch'):
        jax.grad(lambda v: jnp.sum(rcwa.Polygon(vertices=v).vertices))(bow_tie)


def test_the_shape_dict_takes_vertices_and_nothing_that_would_place_it():
    from wavebit._common.geometry.recipes import check_shape_spec
    check_shape_spec(dict(type='polygon', vertices=L_SHAPE, material=4.))
    for extra, says in ((dict(cx=.1), 'placed, sized and turned by its vertices'),
                        (dict(angle=.3), 'placed, sized and turned by its vertices'),
                        (dict(vertex=L_SHAPE), "did you mean 'vertices'")):
        with pytest.raises(ValueError, match=says):
            check_shape_spec(dict(type='polygon', vertices=L_SHAPE, material=4., **extra))
    with pytest.raises(ValueError, match='needs vertices='):
        check_shape_spec(dict(type='polygon', material=4.))
    with pytest.raises(ValueError, match='at least three'):
        check_shape_spec(dict(type='polygon', vertices=L_SHAPE[:2], material=4.))


def test_the_primitive_reader_refuses_a_polygon_rather_than_reading_it_as_a_point():
    """``shape_geometry``'s defaults would make a polygon a zero-size box at the cell centre."""
    from wavebit._common.geometry.recipes import shape_geometry
    with pytest.raises(NotImplementedError, match='placed by its'):
        shape_geometry(dict(type='polygon', vertices=L_SHAPE), ((1., 0.), (0., 1.)))


# ------------------------------------------------------------------------------- solving

@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_polygon_solves_on_every_backend(backend):
    if backend != 'numpy':
        pytest.importorskip(backend)
    reference = efficiencies(rcwa.run(problem(region(L_SHAPE), region(TRIANGLE, 3.)),
                                      numerics=NVF))
    result = efficiencies(rcwa.run(problem(region(L_SHAPE), region(TRIANGLE, 3.)),
                                   numerics=NVF, backend=backend))
    assert abs(sum(np.sum(x) for x in result) - 1.) < 1e-12        # lossless
    assert gap(result, reference) < 1e-11, backend


def test_a_rectangle_written_as_its_corners_is_the_rectangle():
    """The polygon transform of four corners is the closed-form box, to rounding."""
    from wavebit._common.geometry.polygon import rect_verts
    rectangle = rcwa.Rectangle(size=(.46, .31), center=(.07, -.04), angle=.37)
    corners = tuple(map(tuple, np.asarray(rect_verts(np, .07, -.04, .46, .31, .37))))
    numerics = rcwa.Numerics(orders=(4, 4), factorization='laurent')
    for polarization in ('s', 'p'):
        a = efficiencies(rcwa.run(problem(region(rectangle), polarization=polarization),
                                  numerics=numerics))
        b = efficiencies(rcwa.run(problem(region(corners), polarization=polarization),
                                  numerics=numerics))
        assert gap(a, b) < 1e-12, polarization          # measured 5.5e-14 (orders (8, 8): 1.5e-13)


def test_on_the_painting_route_the_corners_build_the_rectangles_matrices():
    """Same loops, same interface plan, same normal field: NVF included, the matrices agree."""
    import array_api_compat.numpy as xnp
    from wavebit.lattice import as_lattice
    from wavebit.rcwa._engine.boundary import to_conv_mat_boundary_layer
    from wavebit._common.geometry.polygon import rect_verts
    cell = as_lattice(period=(1., 1.))
    rectangle = dict(type='rectangle', cx=.07, cy=-.04, lx=.46, ly=.31, angle=.37, material=6.25)
    corners = dict(type='polygon', vertices=np.asarray(rect_verts(np, .07, -.04, .46, .31, .37)),
                   material=6.25)
    a = to_conv_mat_boundary_layer(xnp, [rectangle], 1., 3, 3, cell, xnp.complex128,
                                   shape_nvf_grid_size=64)
    b = to_conv_mat_boundary_layer(xnp, [corners], 1., 3, 3, cell, xnp.complex128,
                                   shape_nvf_grid_size=64)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=0, atol=1e-14)


def test_under_nvf_the_corners_converge_on_the_closed_form_rectangle():
    """Two normal fields, one limit.

    A lone Rectangle is built in closed form, with its per-face blended field; its corners as a
    Polygon take the painting route's nearest-interface field. The geometry is identical (above),
    so what separates them is the field construction, and it closes with truncation. Measured,
    per-order efficiency: 7.1e-3 at orders (4, 4) and 4.8e-4 at (8, 8), at
    ``shape_nvf_grid_size=256`` and at the default 1024 alike -- the numbers the Polygon
    docstring quotes.
    """
    from wavebit._common.geometry.polygon import rect_verts
    rectangle = rcwa.Rectangle(size=(.46, .31), center=(.07, -.04), angle=.37)
    corners = tuple(map(tuple, np.asarray(rect_verts(np, .07, -.04, .46, .31, .37))))
    gaps = []
    for order in (4, 8):
        numerics = rcwa.Numerics(orders=(order, order), shape_nvf_grid_size=256)
        a = efficiencies(rcwa.run(problem(region(rectangle), theta=.2, phi=.3), numerics=numerics))
        b = efficiencies(rcwa.run(problem(region(corners), theta=.2, phi=.3), numerics=numerics))
        gaps.append(gap(a, b))
    assert gaps[0] < 1e-2 and gaps[1] < 1e-3 and gaps[1] < gaps[0] / 5, gaps


@pytest.mark.slow
def test_the_default_solve_lands_on_an_exact_pixel_li_reference():
    """The NVF polygon solve, checked by a construction that shares none of its field.

    An axis-aligned L at normal incidence, near a resonance, so the factorization matters. A
    20 x 20 raster draws it exactly -- every edge on a pixel boundary -- and Li's rule, the
    right one for axis-aligned edges, converges on it: 0.450860 at orders (12, 12), 0.451066 at
    (16, 16). Laurent, the operator S4 arbitrates in the golden, is still 0.054 above it at
    (12, 12). The polygon's NVF solve sits on the reference at orders 4 and 8 (3.2e-5, 5.6e-5)
    with an excursion at 6 (1.5e-2), and is nearer than Laurent at every order, measured.
    """
    inside = lambda X, Y: (((X > -.35) & (X < .3) & (Y > -.35) & (Y < -.1))
                           | ((X > -.35) & (X < -.1) & (Y > -.35) & (Y < .3)))
    c = (np.arange(20) + .5) / 20 - .5
    raster = np.where(inside(*np.meshgrid(c, c)), 6.25, 1.).astype(complex)

    def normal(layer):
        return rcwa.Problem(stack=rcwa.Stack(period=(1., 1.), layers=(layer,)),
                            excitation=rcwa.Excitation(wavelength=1.3, sources=(rcwa.PlaneWave(),)))

    reference = float(np.sum(rcwa.run(normal(rcwa.Layer(thickness=.4, epsilon=raster)),
                                      numerics=rcwa.Numerics(orders=(12, 12), factorization='li',
                                                             fourier_coefficients='integrated')).T))
    polygon = rcwa.Layer(thickness=.4, background=Material(1.), regions=(region(L_SHAPE),))
    for order in (4, 6, 8):
        nvf = float(np.sum(rcwa.run(normal(polygon), numerics=rcwa.Numerics(
            orders=(order, order), shape_nvf_grid_size=256)).T))
        laurent = float(np.sum(rcwa.run(normal(polygon), numerics=rcwa.Numerics(
            orders=(order, order), factorization='laurent')).T))
        assert abs(nvf - reference) < 2e-2, (order, nvf, reference)
        assert abs(nvf - reference) < abs(laurent - reference) / 3, (order, nvf, laurent)


def test_a_clockwise_outline_solves_as_the_same_region_bit_for_bit():
    forward = efficiencies(rcwa.run(problem(region(L_SHAPE), region(TRIANGLE, 3.)), numerics=NVF))
    reverse = efficiencies(rcwa.run(problem(region(L_SHAPE[::-1]), region(TRIANGLE[::-1], 3.)),
                                    numerics=NVF))
    for a, b in zip(forward, reverse):
        np.testing.assert_array_equal(a, b)


def test_moving_the_outline_by_a_lattice_vector_moves_nothing():
    """A cell is a torus: the outline one period over is the same structure, normal field and all."""
    moved = tuple((x + 1., y - 1.) for x, y in L_TILTED)
    a = efficiencies(rcwa.run(problem(region(L_TILTED)), numerics=NVF))
    b = efficiencies(rcwa.run(problem(region(moved)), numerics=NVF))
    assert gap(a, b) < 1e-12


def _winding(vertices, x, y):
    """Signed winding number of a closed outline about every point -- written here, for the raster."""
    v = np.asarray(vertices, float)
    a, b = v, np.roll(v, -1, axis=0)
    count = np.zeros(np.broadcast(x, y).shape, dtype=int)
    for (ax, ay), (bx, by) in zip(a, b):
        left = (bx - ax) * (y - ay) - (x - ax) * (by - ay)
        count += ((ay <= y) & (by > y) & (left > 0)).astype(int)
        count -= ((by <= y) & (ay > y) & (left < 0)).astype(int)
    return count


def _mean_epsilon(shapes):
    """``<eps>``: the DC entry of Laurent's ``[[eps]]`` for one painted layer."""
    solver = _legacy_wavebit_rcwa(structure=[dict(background=1., shapes=shapes)], period=(1., 1.),
                                  wavelength=1.3, thickness=(.4,), truncation_order=(2, 2),
                                  factorization='laurent')
    epx = np.asarray(solver.eps_conv_matrices()[0])[0]
    return epx[epx.shape[0] // 2, epx.shape[0] // 2].real


def _area(vertices):
    v = np.asarray(vertices, float)
    return abs(0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))


def test_a_polygon_paints_over_a_region_and_under_one():
    """The mean permittivity is the painted areas', checked two independent ways.

    Painting is the clipper's (Greiner-Hormann); the exact areas here come from a Green's-theorem
    intersection that shares nothing with it, and a point-sampled raster shares nothing with
    either. A box crosses the L's arm on top of it; the L covers part of a triangle under it.
    """
    from wavebit._common.geometry.overlap import polygon_intersection
    from wavebit._common.geometry.polygon import rect_verts
    box = np.asarray(rect_verts(np, .05, -.2, .5, .12, .3))
    ell, tri = np.asarray(L_SHAPE), np.asarray(TRIANGLE)
    n = 2000
    s = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(s, s)
    in_box, in_ell, in_tri = (_winding(v, X, Y) != 0 for v in (box, ell, tri))
    cases = [
        ([dict(type='polygon', vertices=L_SHAPE, material=6.25),
          dict(type='rectangle', cx=.05, cy=-.2, lx=.5, ly=.12, angle=.3, material=9.)],
         1. + 5.25 * (_area(ell) - polygon_intersection(ell, box, 1e-12)[0]) + 8. * _area(box),
         np.where(in_box, 9., np.where(in_ell, 6.25, 1.)).mean()),
        ([dict(type='polygon', vertices=TRIANGLE, material=2.),
          dict(type='polygon', vertices=L_SHAPE, material=6.25)],
         1. + 1. * (_area(tri) - polygon_intersection(tri, ell, 1e-12)[0]) + 5.25 * _area(ell),
         np.where(in_ell, 6.25, np.where(in_tri, 2., 1.)).mean()),
    ]
    for shapes, exact, raster in cases:
        built = _mean_epsilon(shapes)
        assert abs(built - exact) < 1e-13, (built, exact)
        assert abs(built - raster) < 2e-3, (built, raster)     # the raster's own O(1/n)
        assert abs(built - _mean_epsilon(shapes[::-1])) > 1e-2   # the order is the painting


def _painted(scale, length, case):
    """One painted layer from ``PAINTINGS``, every length multiplied by ``scale``."""
    def geometry(spec):
        if spec[0] == 'box':
            _, cx, cy, lx, ly, angle = spec
            return rcwa.Rectangle(size=(lx * scale, ly * scale), center=(cx * scale, cy * scale),
                                  angle=angle)
        return rcwa.Polygon(vertices=tuple((x * scale, y * scale) for x, y in spec))
    layer = rcwa.Layer(thickness=.4 * scale, background=Material(1.),
                       regions=tuple(region(geometry(g), e) for g, e in PAINTINGS[case]))
    return rcwa.Problem(units=Units(length=length),
                        stack=rcwa.Stack(layers=(layer,), bottom=Material(2.25),
                                         period=(scale, scale)),
                        excitation=rcwa.Excitation(wavelength=1.3 * scale, sources=(
                            rcwa.PlaneWave(theta=.35, phi=.6, polarization='s'),)))


def _painted_mean(case):
    """``<eps>`` of a ``PAINTINGS`` layer from Green's-theorem areas, which share nothing with
    the clipper."""
    from wavebit._common.geometry.overlap import polygon_intersection
    from wavebit._common.geometry.polygon import rect_verts
    (under, e0), (over, e1) = PAINTINGS[case]
    under, over = (np.asarray(rect_verts(np, *g[1:])) if g[0] == 'box' else np.asarray(g)
                   for g in (under, over))
    return (1. + (e0 - 1.) * (_area(under) - polygon_intersection(under, over, 1e-12)[0])
            + (e1 - 1.) * _area(over))


def _gon_over_a_box_corner():
    """A 60-degree box with one corner exactly on a short edge of a 64-gon painted over it."""
    from wavebit._common.geometry.polygon import ellipse_verts
    gon = np.asarray(ellipse_verts(np, .0524139475264455, .0843644987235117, .19130764018455063,
                                   .16384685936997645, .5235987755982988, 64))
    box = ('box', .057851650540178, .1421205263084714, .09238232698180014, .21906341970310506,
           1.0471975511965976)
    return ((box, 4.), (tuple(map(tuple, gon)), 9.))


#: Two regions each, the later painting over the earlier: a box across the L's arm, the L over
#: part of a triangle, the box the bug report found refused in metres instead of summed, and a
#: box corner on an edge 0.009 long, which an on-edge test in length units held on the edge
#: through every nudge in um and m.
PAINTINGS = {
    'box over L': ((L_SHAPE, 6.25), (('box', .05, -.2, .5, .12, .3), 9.)),
    'L over triangle': ((TRIANGLE, 2.), (L_SHAPE, 6.25)),
    'box over L near the edge': (
        # the outline `L_NEAR_THE_EDGE`, which is defined below the tests that use this table
        (((.1, -.35), (.45, -.35), (.45, -.1), (.3, -.1), (.3, .3), (.1, .3)), 4.),
        (('box', .4, .1, .2, .2, 0.), 9.)),
    'polygonised ellipse over a box corner on its edge': _gon_over_a_box_corner(),
}


@pytest.mark.parametrize('length,scale', [('m', 1e-9), ('m', 1e-6), ('m', 1e-3), ('um', 1.),
                                          ('nm', 1e3), ('m', 1e3)])
@pytest.mark.parametrize('case', list(PAINTINGS))
def test_a_painting_is_the_same_in_any_length_unit(case, length, scale):
    """The same layout spelled in any unit paints the same layer and solves the same.

    The planner compared its tolerances with lengths and areas as plain numbers, so in the
    default unit, metres, a sub-micron overlap was painted wrongly -- the box cut out of the L as
    a hole, the triangle taken as covered: ``<eps>`` 9.5e-3 and 9.3e-2 off -- or refused with a
    private exception (the box near the edge, and the box corner on the 64-gon's edge), while
    ``check`` passed and ``R + T`` stayed 1.
    Planning now reads them relative to the outlines it judges.
    """
    from wavebit.rcwa import operators
    epx = np.asarray(operators.assemble(_painted(scale, length, case),
                                        numerics=LAURENT).epsilon[0].epx)
    epx = epx[0] if epx.ndim == 3 else epx
    mean = epx[epx.shape[0] // 2, epx.shape[1] // 2].real
    assert abs(mean - _painted_mean(case)) < 1e-13, mean
    reference = efficiencies(rcwa.run(_painted(1., 'm', case), numerics=LAURENT))
    assert gap(efficiencies(rcwa.run(_painted(scale, length, case), numerics=LAURENT)),
               reference) < 1e-12


# ------------------------------------------------------------------------------- refusals

#: A U whose right arm reaches into the next copy's left arm.
U_INTO_ITS_IMAGE = ((-.45, -.4), (.6, -.4), (.6, .4), (.4, .4), (.4, -.2), (-.25, -.2),
                    (-.25, .4), (-.45, .4))
#: 1.2 wide on a period of 1, clear of its images: each arm runs into the notch the
#: neighbouring copy leaves. Its bounding box and its convex hull both overlap their images.
Z_THREADED = ((-.5, -.4), (.2, -.4), (.2, .2), (.7, .2), (.7, .4), (0., .4), (0., -.2), (-.5, -.2))
L_NEAR_THE_EDGE = ((.1, -.35), (.45, -.35), (.45, -.1), (.3, -.1), (.3, .3), (.1, .3))


def _l_spanning(overhang):
    return ((-.5, -.4), (.5 + overhang, -.4), (.5 + overhang, -.2), (-.2, -.2), (-.2, .3),
            (-.5, .3))


def _box(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)


_ELL = np.array([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)], float)


@pytest.mark.parametrize('P,Q,area,length', [
    (_box(0, 0, 1, 1), _box(.5, 0, 1.5, 1), .5, 3.),
    (_box(-1, -.1, 1, .1), _box(-.1, -1, .1, 1), .04, .8),        # a cross: no vertex inside
    (_box(0, 0, 1, 1), _box(.2, .2, .4, .5), .06, 1.),             # contained
    (_box(0, 0, 1, 1), _box(0, 0, 1, 1), 1., 4.),                  # identical
    (_box(0, 0, 1, 1), _box(0, 0, .5, .5), .25, 2.),               # inside, sharing two edges
    (_box(0, 0, 1, 1), _box(1, 0, 2, 1), 0., 0.),                  # touching along an edge
    (_box(0, 0, 1, 1), _box(1, 1, 2, 2), 0., 0.),                  # touching at a corner
    (_box(0, 0, 1, 1)[::-1], _box(.5, 0, 1.5, 1), .5, 3.),         # clockwise
    (_ELL, _box(1.5, 1.5, 2.5, 2.5), 0., 0.),                      # in the L's notch
    (_ELL, _box(.5, .5, 2, 2), 1.25, 6.),                          # over the reflex corner
], ids=['half', 'cross', 'contained', 'identical', 'corner-inside', 'edge', 'corner',
        'clockwise', 'notch', 'reflex'])
def test_the_intersection_of_two_outlines_is_exact(P, Q, area, length):
    """Area and boundary length of ``P & Q``, hand-computed, either way round."""
    from wavebit._common.geometry.overlap import polygon_intersection
    for a, b in ((P, Q), (Q, P)):
        got_area, got_length = polygon_intersection(a, b, 1e-12)
        assert abs(got_area - area) < 1e-12 and abs(got_length - length) < 1e-12


@pytest.mark.parametrize('regions,refused', [
    ((region(U_INTO_ITS_IMAGE),), 'runs into its own periodic image across the x period'),
    ((region(Z_THREADED),), None),
    ((region(_l_spanning(0.)),), None),              # touching its image is not overlapping
    ((region(_l_spanning(1e-9)),), None),            # rounding is not overlapping either
    ((region(_l_spanning(1e-4)),), 'runs into its own periodic image'),
    ((region(L_NEAR_THE_EDGE), region(rcwa.Circle(radius=.12, center=(-.45, -.22)), 9.)),
     'overlap only across the x period boundary'),
    # the same circle moved to touch the L's image: tangent, and the 256-gon standing in for it
    # pokes 5e-5 of a radius past the curve, which the sweep allows for
    ((region(L_NEAR_THE_EDGE), region(rcwa.Circle(radius=.12, center=(-.43, -.22)), 9.)), None),
    ((region(L_NEAR_THE_EDGE), region(rcwa.Rectangle(size=(.2, .2), center=(.4, .1)), 9.)), None),
], ids=['into-its-image', 'threaded-z', 'touching-image', 'rounding', 'past-its-image',
        'across-the-edge', 'tangent-across-the-edge', 'overlap-in-the-cell'])
def test_the_image_test_reads_the_outline_not_its_hull(regions, refused):
    numerics = rcwa.Numerics(orders=(1, 1), factorization='laurent')
    if refused is None:
        result = rcwa.run(problem(*regions), numerics=numerics)
        assert abs(sum(np.sum(x) for x in efficiencies(result)) - 1.) < 1e-12
        return
    with pytest.raises(ValidationError, match=refused):
        rcwa.run(problem(*regions), numerics=numerics)


@pytest.mark.parametrize('numerics,says', [
    (rcwa.Numerics(orders=(2, 2), factorization='li'), 'supports only axis-aligned rectangles'),
    (rcwa.Numerics(orders=(2, 2), nvf_projector='circular'),
     "nvf_projector='circular' requires rectangles, circles or ellipses"),
])
def test_a_build_that_cannot_take_a_polygon_says_so_at_check(numerics, says):
    report = rcwa.check(problem(region(L_SHAPE)), numerics=numerics)
    assert [i.code for i in report.issues] == ['rcwa.unsupported']
    assert says in report.issues[0].message


def test_tensor_materials_refuse_a_polygon_by_name():
    tensor = Material(np.diag([4., 5., 6.]).astype(complex))
    tensor_region = rcwa.Region(geometry=rcwa.Polygon(vertices=L_SHAPE), material=tensor)
    with pytest.raises(UnsupportedFeatureError, match='holds a polygon, and tensor materials'):
        rcwa.run(problem(tensor_region), numerics=rcwa.Numerics(orders=(1, 1)))


def test_the_periodic_fdtd_scene_refuses_a_polygon_by_name():
    from wavebit.fdtd._periodic.scene import _shape_partition
    structure = [dict(background=1., shapes=[dict(type='polygon', vertices=L_SHAPE,
                                                  material=4.)])]
    with pytest.raises(ValueError, match='does not build a polygon'):
        _shape_partition(structure, (1., 1.), xp=np, array=np.asarray,
                         check=lambda valid, message: None, integration='area')


#: A parallelogram with edges at 1 rad and at 2.3 rad to the x axis and a smaller one fitted into
#: its corner, so that a nudge along either angle slides the smaller one along a shared edge.
_D1, _D2 = np.array([np.cos(1.), np.sin(1.)]), np.array([np.cos(2.3), np.sin(2.3)])
IN_THE_CORNER = tuple(np.array([0. * _D1, a * _D1, a * _D1 + b * _D2, b * _D2]) - .15 * (_D1 + _D2)
                      for a, b in ((.3, .3), (.12, .1)))
@pytest.mark.parametrize('length,scale', [('um', 1.), ('nm', 1e3), ('m', 1e-6)])
@pytest.mark.parametrize('pad,named', [(False, 'Layer 0: region 1 cannot be painted over region 0'),
                                       (True, 'Layer 0: region 2 cannot be painted over region 1')])
def test_a_painting_the_planner_cannot_resolve_is_refused_with_a_code(length, scale, pad, named):
    """A contact the planner cannot resolve reached ``prepare`` and ``run`` as the clipper's
    private exception, uncoded. It is the ``rcwa.unsupported`` engine refusal now, the same in
    every unit, naming the layer and the two regions that meet -- under a pad too, where the
    region being cut when it failed is the pad, and the contact is between the two above it.
    """
    from wavebit._common.geometry.clipper import DegenerateClipError
    regions = tuple(region(tuple(map(tuple, v * scale)), e)
                    for v, e in zip(IN_THE_CORNER, (4., 9.)))
    if pad:
        regions = (region(rcwa.Rectangle(size=(.95 * scale, .95 * scale)), 2.),) + regions
    p = rcwa.Problem(units=Units(length=length),
                     stack=rcwa.Stack(period=(scale, scale),
                                      layers=(rcwa.Layer(thickness=.3 * scale, regions=regions),)),
                     excitation=rcwa.Excitation(wavelength=1.3 * scale,
                                                sources=(rcwa.PlaneWave(polarization='s'),)))
    for door in (rcwa.prepare, rcwa.run):
        with pytest.raises(UnsupportedFeatureError) as caught:
            door(p, numerics=rcwa.Numerics(orders=(1, 1), factorization='laurent'))
        assert (caught.value.code, caught.value.path) == ('rcwa.unsupported', 'problem')
        assert named in caught.value.message
        assert isinstance(caught.value.__cause__, DegenerateClipError)


# ------------------------------------------------------------- frozen geometry and staleness

def _engine_structure(vertices, cx=.3):
    return [dict(background=1., shapes=[dict(type='polygon', vertices=vertices, material=6.25),
                                        dict(type='circle', cx=cx, cy=.3, lx=.16, material=3.)])]


ENGINE = dict(period=(1., 1.), wavelength=1.3, thickness=(.4,), truncation_order=(2, 2),
              top=1., bot=2.25, theta=.35, phi=.6, psi=np.pi / 2, shape_nvf_grid_size=64)


def test_a_frozen_polygon_plan_goes_stale_when_the_painting_changes():
    from wavebit.rcwa._engine.geometry import StaleGeometryError
    frozen = _legacy_wavebit_rcwa(structure=_engine_structure(L_TILTED), **ENGINE).freeze_geometry()
    assert frozen.boundary_plans[0] is not None and frozen.gamma_plans[0] is None
    nudged = tuple((x + 1e-3, y) for x, y in L_TILTED)
    _legacy_wavebit_rcwa(structure=_engine_structure(nudged), **ENGINE).solve(geometry=frozen)
    # the circle slides onto the L's arm: the painting the record resolves is gone
    onto = _legacy_wavebit_rcwa(structure=_engine_structure(L_TILTED, cx=-.2), **ENGINE)
    with pytest.raises(StaleGeometryError, match='no longer fits'):
        onto.solve(geometry=frozen)


def test_jax_replays_a_frozen_polygon_under_jit_and_needs_the_freeze():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def total(v, geometry=None):
        solver = _legacy_wavebit_rcwa(backend='jax', structure=_engine_structure(v), **ENGINE)
        return jnp.sum((solver.solve() if geometry is None else solver.solve(geometry=geometry)).T).real

    with pytest.raises(ValueError, match='freeze_geometry'):
        jax.jit(total)(jnp.asarray(L_TILTED))
    frozen = _legacy_wavebit_rcwa(structure=_engine_structure(L_TILTED), **ENGINE).freeze_geometry()
    value, grad = jax.jit(jax.value_and_grad(lambda v: total(v, frozen)))(jnp.asarray(L_TILTED))
    eager = float(np.sum(np.asarray(
        _legacy_wavebit_rcwa(structure=_engine_structure(L_TILTED), **ENGINE).solve().T)).real)
    assert abs(float(value) - eager) < 1e-13
    # and the unjitted gradient plans from its primal: the same plan, the same derivative
    np.testing.assert_allclose(np.asarray(jax.grad(total)(jnp.asarray(L_TILTED))), np.asarray(grad),
                               rtol=0, atol=1e-12)


def test_the_lifecycle_replays_moved_vertices_and_refuses_a_changed_painting():
    torch = pytest.importorskip('torch')
    circle = region(rcwa.Circle(radius=.08, center=(.3, .3)), 3.)

    def live(vertices):
        return problem(region(vertices), circle)

    # Prepared with the spelling it runs with: an (N, 2) array and N pairs are two layouts.
    prepared = rcwa.prepare(live(torch.tensor(L_TILTED, dtype=torch.float64)), numerics=NVF,
                            backend='torch')
    moved = torch.tensor(L_TILTED, dtype=torch.float64) + torch.tensor([1e-3, -2e-3],
                                                                       dtype=torch.float64)
    moved.requires_grad_(True)
    total = prepared.run(live(moved)).T.sum()
    total.backward()
    assert torch.isfinite(moved.grad).all() and float(moved.grad.abs().max()) > 0
    onto = torch.tensor(L_TILTED, dtype=torch.float64) + torch.tensor([.45, .45], dtype=torch.float64)
    with pytest.raises(IncompatibleProblemError) as error:
        prepared.run(live(onto))
    assert error.value.issue.code == 'prepared.topology'


def test_moving_one_region_leaves_how_two_others_are_painted():
    """Two rotated boxes touching corner to corner, and a small square elsewhere in the cell.

    Each contact is judged on its own pair. With the layer scaled by its largest coordinate for
    planning, moving the square's edge across x = 1 in the cell's frame changed how the two
    boxes' contact was read, and the preparation refused the move as a topology change.
    """
    cell = 1.2

    def box(cx, cy, lx, ly, angle):
        return rcwa.Rectangle(size=(lx, ly), center=(cx - cell / 2, cy - cell / 2), angle=angle)

    def layout(edge):
        regions = (region(box(.46046370366608047, .4575047770503867, .24458258820914708,
                              .25062090039513996, 1.0471975511965976), 4.),
                   region(box(.5597759597229324, .4263373448297414, .22735072559901187,
                              .18603952836162518, .5235987755982988), 9.),
                   region(box(edge - .01, .05, .02, .02, 0.), 2.))
        layer = rcwa.Layer(thickness=.4, background=Material(1.), regions=regions)
        return rcwa.Problem(units=Units(length='um'),
                            stack=rcwa.Stack(layers=(layer,), bottom=Material(2.25),
                                             period=(cell, cell)),
                            excitation=rcwa.Excitation(wavelength=1.3, sources=(
                                rcwa.PlaneWave(theta=.35, phi=.6, polarization='s'),)))

    prepared = rcwa.prepare(layout(.99), numerics=LAURENT)
    assert gap(efficiencies(prepared.run(layout(1.01))),
               efficiencies(rcwa.run(layout(1.01), numerics=LAURENT))) < 1e-13


def test_the_lifecycle_differentiates_jax_vertices_under_grad_and_refuses_them_under_jit():
    """As all region geometry: eager ``jax.grad`` is checked on its primal, ``jit`` has none."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import jax.numpy as jnp
    with jax.enable_x64(True):
        prepared = rcwa.prepare(problem(region(jnp.asarray(L_TILTED))), numerics=LAURENT,
                                backend='jax')
        loss = lambda v: jnp.sum(prepared.run(problem(region(v))).T)
        derivative = np.asarray(jax.grad(loss)(jnp.asarray(L_TILTED)))
        with pytest.raises(UnsupportedFeatureError) as error:
            jax.jit(loss)(jnp.asarray(L_TILTED))
    assert error.value.issue.code == 'prepared.host_topology'
    assert error.value.issue.path.endswith('geometry.vertices')
    prepared = rcwa.prepare(problem(region(torch.tensor(L_TILTED, dtype=torch.float64))),
                            numerics=LAURENT, backend='torch')
    v = torch.tensor(L_TILTED, dtype=torch.float64, requires_grad=True)
    prepared.run(problem(region(v))).T.sum().backward()
    # measured 1.4e-13 against derivatives up to 8.6e-2
    np.testing.assert_allclose(derivative, v.grad.numpy(), rtol=0, atol=1e-11)


# ----------------------------------------------------------------------------- derivatives

def _torch_value_and_grad(vertices, numerics, vertex=3, axis=0):
    """``sum(T)`` and its derivative in one coordinate, through the public lifecycle."""
    import torch
    v = torch.tensor(vertices, dtype=torch.float64)
    q = v[vertex, axis].clone().requires_grad_(True)
    rows = [v[i] if i != vertex else torch.stack([q, v[i, 1]] if axis == 0 else [v[i, 0], q])
            for i in range(len(vertices))]
    total = rcwa.run(problem(region(torch.stack(rows))), numerics=numerics,
                     backend='torch').T.sum()
    total.backward()
    return float(total.detach()), float(q.grad)


def test_under_laurent_every_vertex_derivative_is_the_central_difference():
    """No normal field, so nothing is sampled and a pointwise difference is a fair judge."""
    torch = pytest.importorskip('torch')
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    v = torch.tensor(L_SHAPE, dtype=torch.float64, requires_grad=True)
    rcwa.run(problem(region(v)), numerics=numerics, backend='torch').T.sum().backward()
    base, h = np.asarray(L_SHAPE), 1e-6
    fd = np.zeros_like(base)
    for i in range(base.shape[0]):
        for k in range(2):
            step = np.zeros_like(base)
            step[i, k] = h
            plus, minus = (float(np.sum(rcwa.run(problem(region(base + s * step)),
                                                 numerics=numerics).T)) for s in (1, -1))
            fd[i, k] = (plus - minus) / (2 * h)
    # measured 6e-9 at worst against derivatives up to 7e-2: the difference's own error
    np.testing.assert_allclose(v.grad.numpy(), fd, rtol=0, atol=5e-8)


def _window(numerics, grid_pixels, per_pixel):
    """Integrated AD across a window, against the change of the loss across it.

    The abscissae of the gradient are the loss's own (``position-gradient`` rule): the
    derivative of a sampled normal field fluctuates below a pixel, and a coarser gradient grid
    would alias into whatever its spacing selected.
    """
    pixel = 1. / 128
    p0 = L_TILTED[3][0]
    ps = p0 + np.linspace(-grid_pixels / 2, grid_pixels / 2, int(grid_pixels * per_pixel) + 1) * pixel
    vals, grads = zip(*(_torch_value_and_grad(tuple(
        (p, y) if i == 3 else (x, y) for i, (x, y) in enumerate(L_TILTED)), numerics)
        for p in ps))
    change = vals[-1] - vals[0]
    return abs(np.trapezoid(grads, ps) - change) / abs(change)


def test_under_laurent_the_vertex_derivative_integrates_to_the_change():
    pytest.importorskip('torch')
    # quadrature error only: 1.5e-4 at 8 samples a pixel on this window (grid-free rule)
    assert _window(LAURENT, 4, 8) < 1e-3


def test_under_nvf_the_vertex_derivative_integrates_to_the_change():
    """A strict xfail until the painting route's normal field averaged its switches.

    The nearest-interface direction turns across the medial axis, and the turn moves with the
    vertex: every sample it crossed stepped, and autodiff of the sampled field saw none of the
    steps -- 9.4-11% of this window's change missed at ``shape_nvf_grid_size`` 64 to 512 alike,
    a missing term rather than a grid error. Averaged over half a sample step
    (``boundary.material_projector_fields``) every sample moves smoothly: measured 0.15%.
    """
    pytest.importorskip('torch')
    assert _window(NVF, 4, 8) < 1e-2


@pytest.mark.slow
def test_under_nvf_the_vertex_derivative_integrates_to_the_change_on_a_finer_normal_grid():
    """The same window, sixteen pixels of a 512 grid: a switch is half a step wide at any grid,
    so the derivative holds as the grid refines (measured 0.11%; 0.24% at 256)."""
    pytest.importorskip('torch')
    assert _window(rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=512), 4, 8) < 1e-2


# ------------------------------------------------------------------------------- pictures

def test_the_structure_picture_is_the_painted_outline():
    from wavebit.rcwa._plotting import slice_structure
    layers = [dict(background=1., shapes=[dict(type='polygon', vertices=L_SHAPE, material=6.25),
                                          dict(type='polygon', vertices=TRIANGLE[::-1],
                                               material=2.)])]
    solver = _legacy_wavebit_rcwa(structure=layers, period=(1., 1.), wavelength=1.3,
                                  thickness=(.4,), truncation_order=(2, 2))
    n = 256
    picture = slice_structure(solver, 0, n=n)
    s = (np.arange(n) + .5) / n - .5                 # the picture samples the cell from its corner
    X, Y = np.meshgrid(s, s)
    expected = np.where(_winding(TRIANGLE, X, Y) != 0, 2.,
                        np.where(_winding(L_SHAPE, X, Y) != 0, 6.25, 1.))
    np.testing.assert_array_equal(picture.real, expected)
    skewed = _legacy_wavebit_rcwa(structure=layers, lattice=[[1., .3], [0., 1.]], wavelength=1.3,
                                  thickness=(.4,), truncation_order=(2, 2))
    with pytest.raises(NotImplementedError, match='or a polygon'):
        slice_structure(skewed, 0, n=32)


def test_a_polygon_job_round_trips_in_either_spelling():
    from wavebit import Execution, Job
    for vertices in (L_SHAPE, np.asarray(L_SHAPE)):
        job = Job(solver='rcwa', problem=problem(region(vertices)), numerics=LAURENT,
                  execution=Execution(backend='numpy'))
        back = Job.from_json(job.to_json())
        assert back.key == job.key
        np.testing.assert_array_equal(
            np.asarray(back.problem.stack.layers[0].regions[0].geometry.vertices), np.asarray(L_SHAPE))
