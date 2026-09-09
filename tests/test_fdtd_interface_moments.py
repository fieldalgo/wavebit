"""Per-cell interface arclength and normal second moment: oracle, roll, closed form, gradient."""
import numpy as np
import pytest

from wavebit._core.scene_fdtd import _rectangle_edge_moments, _shape_cell_fractions
from wavebit.backends.time_domain import time_loop_backend


def rectangle_vertices(cx, cy, lx, ly, angle):
    """Counter-clockwise physical corners, the recipe every straight-sided path here uses."""
    c, s = np.cos(angle), np.sin(angle)
    local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*[lx/2, ly/2]
    return local @ np.array([[c, s], [-s, c]])+[cx, cy]


def moments(vertices, mesh, backend=0):
    runtime = time_loop_backend(backend)
    return [np.asarray(v) for v in _rectangle_edge_moments(
        runtime.array(np.asarray(vertices, float)), mesh['x'], mesh['y'], mesh['dx'], mesh['dy'],
        mesh['periods'], xp=runtime.xp, array=runtime.array)]


def fractions(vertices, mesh):
    runtime = time_loop_backend(0)
    return np.asarray(_shape_cell_fractions(
        runtime.array(np.asarray(vertices, float)), mesh['x'], mesh['y'], mesh['dx'], mesh['dy'],
        mesh['periods'], xp=runtime.xp, array=runtime.array))


def edge_moment_oracle(vertices, mesh):
    """Exact arclength bookkeeping along each edge's own parameter; no clipping code shared.

    Cut every oriented edge of every periodic image at each parameter where it meets a cell
    line, then give each resulting piece to the cells whose closed box contains its midpoint.
    Membership is piecewise constant along the edge, so subdividing at the discontinuities
    makes the 1D integral exact and the comparison lands at roundoff rather than at a
    quadrature tolerance. A piece whose midpoint sits exactly on a cell line -- an edge lying
    along the mesh -- is handed to both neighbours, which is what a closed box means.

    This restates the 3x3 ring rather than importing it, so it cannot arbitrate the ring's
    sufficiency; that is what the refusal below, and the stated totals, are for.
    """
    x, y, dx, dy = mesh['x'], mesh['y'], mesh['dx'], mesh['dy']
    left_x, bottom_y = np.asarray(x)-dx/2, np.asarray(y)-dy/2
    interface = np.zeros((len(y), len(x)))
    moment = np.zeros((3, len(y), len(x)))
    for shift_x in (-mesh['periods'][0], 0., mesh['periods'][0]):
        for shift_y in (-mesh['periods'][1], 0., mesh['periods'][1]):
            image = np.asarray(vertices, float)+[shift_x, shift_y]
            for corner in range(4):
                a, b = image[corner], image[(corner+1) % 4]
                direction = b-a
                length = np.hypot(direction[0], direction[1])
                if length == 0:
                    continue
                normal_x, normal_y = -direction[1]/length, direction[0]/length
                cuts = [0., 1.]
                for axis, faces in ((0, np.r_[left_x, left_x+dx]), (1, np.r_[bottom_y, bottom_y+dy])):
                    if direction[axis] != 0:
                        t = (faces-a[axis])/direction[axis]
                        cuts += list(t[(t > 0) & (t < 1)])
                cuts = np.unique(cuts)
                for low, high in zip(cuts[:-1], cuts[1:]):
                    point = a+(low+high)/2*direction
                    columns = np.nonzero((left_x <= point[0]) & (point[0] <= left_x+dx))[0]
                    rows = np.nonzero((bottom_y <= point[1]) & (point[1] <= bottom_y+dy))[0]
                    piece = (high-low)*length
                    for row in rows:
                        for column in columns:
                            interface[row, column] += piece
                            moment[:, row, column] += piece*np.array(
                                [normal_x*normal_x, normal_x*normal_y, normal_y*normal_y])
    return [interface, *moment]


def mesh_of(nx, ny, step, periods):
    return dict(x=np.arange(nx)*step[0], y=np.arange(ny)*step[1],
                dx=step[0], dy=step[1], periods=list(periods))


#: The coincidences the clipping has to survive, plus one plainly asymmetric shape.
#: ``seam`` crosses a periodic seam on a mesh with dx != dy, which is the only thing that can
#: separate the two cell-box metrics: every check that runs implementation and oracle on a square
#: cell passes with dx and dy interchanged. ``near_contact`` leaves a 0.02 gap between periodic
#: copies, so two vertical faces sit in the same cells without coinciding -- an exact contact
#: (lx == period) is refused rather than answered, see the refusal test below. ``coincident`` is
#: axis aligned with both vertical faces on cell lines and both horizontal faces off them, which
#: is the mixed case an ordinary unrotated rectangle falls into by floating-point luck.
#: ``tangency`` sits on a dyadic mesh so all four faces land *exactly* on cell lines, which no
#: non-dyadic mesh actually achieves (0.15 against a grid line stored as 0.15000000000000002 is
#: a near miss, not a coincidence).
#:
#: The third entry is the total arclength the whole mesh must report, worked out on paper: every
#: mesh here tiles exactly one period, so it is the shape's perimeter plus the length of every
#: face lying on a cell line, which the closed box hands to both neighbours. It is a statement
#: per geometry and not a property of the function -- the factor is 1 here, 2 for ``tangency``
#: and 21/13 for ``coincident`` -- and it is blind to *where* the arclength sits: two faces at
#: the same place total the same as two faces at different places. It establishes only that the
#: image ring dropped and duplicated nothing in aggregate; the per-cell question is the oracle's.
GEOMETRIES = {
    'seam': (rectangle_vertices(.03, .05, .38, .19, .37), mesh_of(9, 5, (1/9., .16), (1., .8)), 1.14),
    'near_contact': (rectangle_vertices(0., .07, .98, .3, 0.), mesh_of(10, 8, (.1, .1), (1., .8)), 2.56),
    'corner': (rectangle_vertices(.5, .4, .24, .16, .4), mesh_of(10, 8, (.1, .1), (1., .8)), .8),
    'coincident': (rectangle_vertices(.4375, .5, .25, .4, 0.), mesh_of(8, 8, (.125, .125), (1., 1.)), 2.1),
    'tangency': (rectangle_vertices(.1875, .1875, .5, .25, 0.), mesh_of(8, 8, (.125, .125), (1., 1.)), 3.),
    'asymmetric': (rectangle_vertices(.21, -.13, .44, .11, 1.1), mesh_of(10, 8, (.1, .1), (1., .8)), 1.1),
}

#: Geometries with no face on a cell line, where the cells carrying an interface and the cells
#: with a strictly partial fill are the same set. ``tangency`` and ``coincident`` are excluded
#: because a coincident face gives both neighbours arclength while both are uniformly filled.
GENERIC = ['asymmetric', 'corner', 'near_contact', 'seam']


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
@pytest.mark.parametrize('backend', [0, 1, 2])
def test_moments_match_an_independent_arclength_integration(name, backend):
    if backend == 1:
        pytest.importorskip('jax').config.update('jax_enable_x64', True)
    elif backend == 2:
        pytest.importorskip('torch')
    vertices, mesh, total = GEOMETRIES[name]
    result = moments(vertices, mesh, backend)
    expected = edge_moment_oracle(vertices, mesh)
    for value, reference in zip(result, expected):
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, reference, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result[0].sum(), total, rtol=1e-14)


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_trace_identity_holds_everywhere(name):
    """M_xx + M_yy == L: n is a unit vector, so the trace of the moment is the arclength."""
    vertices, mesh, _ = GEOMETRIES[name]
    interface, m_xx, _, m_yy = moments(vertices, mesh)
    np.testing.assert_allclose(m_xx+m_yy, interface, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('name', GENERIC)
def test_the_moment_and_the_fill_fraction_stand_on_the_same_geometry(name):
    """A cell carries interface exactly when its fill is strictly partial.

    C2 consumes the pair, so the premise it stands on is that the two arrays describe one
    structure -- the same image ring and the same cell boxes. Nothing else in this file compares
    the moment against _shape_cell_fractions, so a future change to either routine's ring or box
    would break the pairing in silence. The fraction is an area sum and saturates to 1 only up to
    roundoff (1 - 2.2e-16 on this mesh), so it is the saturation that carries the slack, not the
    arclength: L is exactly zero in a cell no edge reaches.
    """
    vertices, mesh, _ = GEOMETRIES[name]
    fraction = fractions(vertices, mesh)
    assert np.array_equal(moments(vertices, mesh)[0] > 0, (fraction > 1e-12) & (fraction < 1-1e-12))


def test_touching_or_oversized_periodic_images_are_refused():
    """The 3x3 ring sums each image's boundary, so it must not be handed touching images.

    At lx == period one copy's right face and the next copy's left face are the same segment, the
    material is continuous across it and there is no interface there at all -- yet a sign-blind
    second moment adds the two antiparallel copies instead of cancelling them. Measured before the
    refusal went in, on a mesh of 0.1 cells: a strictly partial cell (fill 0.7) read
    M/L = diag(0.583, 0.417) where the answer is diag(0, 1), and the sweep lx = 0.93, 0.97, 0.999
    ran to 0.82, 0.67, 0.59 and then did not collapse at lx = 1.0 where the gap width is zero.
    Beyond a period the ring is too narrow as well and arclength goes on the floor silently
    (sum 5.31 against a perimeter of 5.98 at a y extent of 1.19). One statement excludes both.

    The oracle cannot arbitrate this: it restates the same ring. So it is pinned as a refusal.
    """
    mesh = mesh_of(10, 8, (.1, .1), (1., .8))
    for lx, ly in ((1., .3), (1.2, .3), (.3, .8), (.3, 1.1)):
        with pytest.raises(ValueError, match='strictly smaller than their periods'):
            moments(rectangle_vertices(0., .07, lx, ly, 0.), mesh)
    # A real gap of any width, however thin, is ordinary geometry and stays admissible.
    moments(rectangle_vertices(0., .07, 1.-1e-9, .3, 0.), mesh)


def test_an_edge_on_a_cell_line_belongs_to_both_neighbours():
    """The closed cell box, stated as a number rather than left to the oracle to agree with."""
    vertices, mesh, _ = GEOMETRIES['tangency']
    interface, m_xx, m_xy, m_yy = moments(vertices, mesh)
    # Faces at x = -0.0625 and 0.4375, y = 0.0625 and 0.3125 are cell lines of this mesh.
    columns = np.searchsorted(mesh['x']-mesh['dx']/2, [-.0625, .4375])
    rows = np.searchsorted(mesh['y']-mesh['dy']/2, [.0625, .3125])
    for column in columns:
        for neighbour in (column-1, column):
            # A vertical face carries a normal along x: it lands wholly in M_xx.
            assert m_xx[rows[0], neighbour % len(mesh['x'])] > 0
    for row in rows:
        for neighbour in (row-1, row):
            assert m_yy[neighbour % len(mesh['y']), columns[0]] > 0
    # Axis-aligned faces only: every normal is a coordinate axis, so the cross term is zero.
    np.testing.assert_array_equal(m_xy, np.zeros_like(m_xy))
    # Perimeter 2*(0.5+0.25) = 1.5, every face on a cell line, so the mesh sees 3.0 -- the
    # convention stated as a number, not inferred from an oracle that shares it.
    np.testing.assert_allclose(interface.sum(), 2*2*(.5+.25), atol=1e-15)


def test_a_coincident_face_steps_the_arclength_while_the_fill_is_smooth():
    """The parallel branch's discontinuity, asserted as numbers rather than crossed by a step.

    An axis-parallel face landing on a mesh line is ordinary traffic -- it happens whenever the
    mesh was sized from the shape -- and the closed box then hands that face to both neighbours.
    Where the face is a cell's only interface both neighbours are uniformly filled and the step is
    harmless, but that is a precondition and not a general argument: here a second, transverse
    face cuts the same cell, the fill stays at 0.1 through the coincidence, and L still steps.
    The 16-offset Jacobian sweep below cannot see any of this, because a rotated edge meets a cell
    line at a point and never along a segment, so this is the parallel branch's only exactness
    check besides the bitwise roll survey.

    Cell (2, 3) of a 0.125 mesh: box x in [0.3125, 0.4375], y in [0.1875, 0.3125]. The rectangle's
    left face sits on x = 0.3125 and its bottom face at y = 0.30, strictly inside the row.
    """
    mesh = GEOMETRIES['coincident'][1]
    def state(offset):
        vertices = rectangle_vertices(.4375+offset, .5, .25, .4, 0.)
        interface, m_xx, _, m_yy = moments(vertices, mesh)
        return fractions(vertices, mesh)[2, 3], interface[2, 3], m_xx[2, 3], m_yy[2, 3]
    before, at = state(-1e-9), state(0.)
    # The fill is continuous and strictly partial on both sides: 0.125 wide by 0.0125 tall.
    np.testing.assert_allclose([before[0], at[0]], [.1, .1], atol=1e-9)
    # L steps by the coincident face's clipped length, 0.3125 - 0.30, and all of it into M_xx.
    np.testing.assert_allclose([before[1], before[2], before[3]], [.125, 0., .125], atol=1e-15)
    np.testing.assert_allclose([at[1], at[2], at[3]], [.1375, .0125, .125], atol=1e-15)
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    traced = time_loop_backend(1)
    columns = len(mesh['x'])
    gradient = jax.jit(jax.grad(lambda p: parametric_moments(p, traced, mesh)[2*columns+3]))
    # And the derivative is blind to the step: exactly zero in every parameter on the near side.
    np.testing.assert_array_equal(
        np.asarray(gradient(traced.array(np.array([.4375-1e-9, .5, .25, .4, 0.])))), np.zeros(5))


@pytest.mark.parametrize('base', [np.array([[.25, .375], [.5, .5], [.4375, .625], [.1875, .5]]),
                                  rectangle_vertices(.4375, .5, .25, .375, 0.)])
def test_one_cell_translation_rolls_the_moment_bitwise(base):
    """Exact roll covariance over every single-cell translation, in the NVF periodicity format.

    Dyadic vertices on a dyadic mesh, so a one-cell translation changes no rounding anywhere
    and the requirement can be bitwise rather than a tolerance -- a tolerance here would pass
    on a path that quietly re-rounds the cell boxes against the shape.

    Two bases, because the first has no edge with a zero component and so never reaches the
    parallel branch of the clipping: its sides are (0.25, 0.125) and (-0.0625, 0.125), exactly
    perpendicular, exactly representable and asymmetric in x and y. The second is axis aligned
    with all four faces on cell lines, which is that branch and nothing else.
    """
    step = .125
    mesh = mesh_of(8, 8, (step, step), (1., 1.))
    reference = moments(base, mesh)
    failures, total = [], 0
    for row in range(8):
        for column in range(8):
            total += 1
            moved = moments(base+[column*step, row*step], mesh)
            if any(not np.array_equal(v, np.roll(r, (row, column), axis=(0, 1)))
                   for v, r in zip(moved, reference)):
                failures.append((row, column))
    assert not failures, f'{len(failures)}/{total} rolls'
    assert f'{len(failures)}/{total} rolls' == '0/64 rolls'


def test_single_cell_closed_form_pins_the_moment_definition():
    """One straight interface across one cell, against arithmetic done on paper.

    tan(theta) = 1/2 for the edge, so sin^2 = 1/5 and cos^2 = 4/5 -- deliberately unequal,
    because M_xx and M_yy swapped is invisible at 0, 45 and 90 degrees. The chord of a line
    through the centre of a unit cell at that angle is 1/cos(theta) = sqrt(5)/2, and the
    normal of an edge at angle theta is (sin, -cos), so M/L is [[1/5, -2/5], [-2/5, 4/5]].
    """
    angle = np.arctan(.5)
    c, s = np.cos(angle), np.sin(angle)
    # A 6 by 6 square rotated by theta, slid so its top edge's midpoint sits on the cell
    # centre. The other three edges and every periodic image stay clear of the one cell.
    vertices = rectangle_vertices(5+3*s, 5-3*c, 6., 6., angle)
    mesh = dict(x=np.array([5.]), y=np.array([5.]), dx=1., dy=1., periods=[10., 10.])
    interface, m_xx, m_xy, m_yy = moments(vertices, mesh)
    np.testing.assert_allclose(interface, [[np.sqrt(5)/2]], rtol=2e-15)
    np.testing.assert_allclose(m_xx/interface, [[.2]], rtol=2e-15)
    np.testing.assert_allclose(m_xy/interface, [[-.4]], rtol=2e-15)
    np.testing.assert_allclose(m_yy/interface, [[.8]], rtol=2e-15)


def parametric_moments(p, runtime, mesh):
    """(cx, cy, lx, ly, angle) to a flat vector of all four fields, in the caller's backend."""
    xp, array = runtime.xp, runtime.array
    c, s = xp.cos(p[4]), xp.sin(p[4])
    local = array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    rotation = xp.stack((xp.stack((c, s)), xp.stack((-s, c))))
    vertices = (local*(xp.stack((p[2], p[3]))/2)) @ rotation+xp.stack((p[0], p[1]))
    fields = _rectangle_edge_moments(vertices, mesh['x'], mesh['y'], mesh['dx'], mesh['dy'],
                                     mesh['periods'], xp=xp, array=array)
    return xp.concat([v.reshape(-1) for v in fields])


@pytest.mark.parametrize('angle', [.41, 0., np.pi/2])
def test_shape_parameter_jacobian_is_finite_and_matches_finite_differences(angle):
    """Property (e): AD against two finite-difference steps at 16 sub-cell offsets.

    The offsets sweep cx across one whole cell in sixteenths, which walks a corner through a cell
    interior and a face through a mesh face in turn. This is the property that decides whether the
    moment is usable in an optimisation at all, and it is settled here on pure geometry rather
    than after a Maxwell run.

    The angle is swept because a rotated shape has no axis-parallel edge at all: at 0.41 the
    guarded Liang-Barsky denominator is never exercised, and replacing it with a bare division
    passes every other check in this file. At 0 and pi/2 that mutation returns 1600 NaNs where the
    real Jacobian is finite and matches central differences to 5e-11.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    mesh = mesh_of(6, 5, (1/6, .16), (1., .8))
    host = time_loop_backend(0)
    traced = time_loop_backend(1)
    jacobian = jax.jit(jax.jacrev(lambda p: parametric_moments(p, traced, mesh)))
    for k in range(16):
        p = np.array([.031+k/16*mesh['dx'], -.017, .37, .23, angle])
        derivative = np.asarray(jacobian(traced.array(p)))
        value = np.asarray(parametric_moments(traced.array(p), traced, mesh))
        assert np.isfinite(derivative).all() and np.isfinite(value).all()
        # Every one of the five parameters must actually move something, or "matches finite
        # differences" is a statement about two columns of zeros.
        assert np.min(np.max(abs(derivative), axis=0)) > 1e-3
        np.testing.assert_allclose(value, np.asarray(parametric_moments(p, host, mesh)), atol=1e-15)
        for step in (2e-6, 1e-6):
            finite = np.stack([(np.asarray(parametric_moments(p+d, host, mesh))
                                -np.asarray(parametric_moments(p-d, host, mesh)))/(2*step)
                               for d in np.eye(5)*step], axis=-1)
            np.testing.assert_allclose(derivative, finite, rtol=1e-8, atol=1e-8)


def test_a_degenerate_edge_contributes_nothing_and_still_differentiates():
    """The zero-length guard, which the docstring promises and nothing else here reaches.

    Dropping the sqrt guard passes every other test in this file, because none of them hands the
    function an edge of zero length. A flattened rectangle is not a contrived input: it is a state
    an optimiser walks into on its own, and it has a closed-form answer -- the two surviving faces
    are the same segment traversed both ways, so the total is 2*lx and every normal is along y.
    """
    mesh = mesh_of(8, 8, (.125, .125), (1., 1.))
    flat = rectangle_vertices(.4375, .5, .4, 0., 0.)
    interface, m_xx, m_xy, m_yy = moments(flat, mesh)
    assert all(np.isfinite(v).all() for v in (interface, m_xx, m_xy, m_yy))
    np.testing.assert_allclose([interface.sum(), m_xx.sum(), m_xy.sum(), m_yy.sum()],
                               [.8, 0., 0., .8], atol=1e-15)
    # A repeated vertex is the same degeneracy arriving from a vertex list rather than a length.
    triangle = np.array([[.3, .3], [.3, .3], [.6, .55], [.3, .55]])
    result = moments(triangle, mesh)
    assert all(np.isfinite(v).all() for v in result)
    np.testing.assert_allclose(result[0].sum(), .3+.25+np.hypot(.3, .25), rtol=1e-14)
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    traced = time_loop_backend(1)
    jacobian = jax.jit(jax.jacrev(lambda p: parametric_moments(p, traced, mesh)))
    for ly in (.05, 1e-12, 0.):
        assert np.isfinite(np.asarray(jacobian(traced.array(np.array([.4375, .5, .4, ly, 0.]))))).all()


def test_a_round_recipe_is_refused_rather_than_chorded():
    runtime = time_loop_backend(0)
    mesh = mesh_of(4, 4, (.25, .25), (1., 1.))
    for bad in (runtime.array([.5, .5, .2]), runtime.array(np.zeros((3, 2)))):
        with pytest.raises(ValueError, match='four oriented edges'):
            _rectangle_edge_moments(bad, mesh['x'], mesh['y'], mesh['dx'], mesh['dy'],
                                    mesh['periods'], xp=runtime.xp, array=runtime.array)
