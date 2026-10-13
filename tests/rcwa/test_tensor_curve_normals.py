"""Smooth tensor primitives must not acquire corners from an interface mesh."""

from wavebit.compat import legacy as _legacy_wavebit
from functools import partial

import numpy as np
import pytest

import wavebit
from wavebit.rcwa._engine import boundary, parametric, tensor_shapes
from wavebit.lattice import as_lattice, sample_grid


CELL = np.array([[600., 0.], [416., 672.]])
POINT = np.array([787.5, 100., 0.])
HEX = np.array([[600., 300.], [0., 300.*np.sqrt(3)]])
SQUARE = np.diag([600., 600.])
TENSOR = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y')


def layer(p, quantity='epsilon', circle='circle'):
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y').epsilon
    material = (wavebit.Material(tensor) if quantity == 'epsilon'
                else wavebit.Material(1., tensor))
    other = 2.25 if quantity == 'epsilon' else wavebit.Material(1., 2.25)
    return dict(background=1., shapes=[
        dict(type='ellipse', lx=p[0], ly=p[1], angle=p[2], material=material),
        dict(type=circle, cy=200., lx=70., material=other)])


def solver(backend='numpy', quantity='epsilon', p=POINT, n=16, **options):
    kw = dict(lattice=CELL, wavelength=900., thickness=[170.],
              truncation_order=(1, 1), theta=.17, phi=.3, polarization=(.8, .6j),
              factorization='nvf', shape_nvf_grid_size=n, answers='field',
              structure=[layer(p, quantity)])
    kw.update(options)
    return _legacy_wavebit.rcwa(backend=backend, **kw)


def objective(s, xp, p, quantity='epsilon', geometry=None):
    r = s.solve(structure=[layer(p, quantity)], geometry=geometry, auto_detune=False)
    f = r.reconstruct_field([17.], [-23.], [70.])
    return xp.real(r.T.sum()) + .1*xp.sum(xp.abs(f)**2)


def reference(p, quantity='epsilon', n=16):
    return float(objective(solver(quantity=quantity, p=p, n=n), np, p, quantity))


def finite_difference(p, quantity='epsilon', n=16):
    gradient = []
    for i, h in enumerate((.001, .001, 1e-6)):
        plus, minus = p.copy(), p.copy()
        plus[i] += h
        minus[i] -= h
        gradient.append((reference(plus, quantity, n)-reference(minus, quantity, n))/(2*h))
    return np.array(gradient)


@pytest.mark.parametrize('quantity', ['epsilon', 'mu'])
@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
def test_shape_derivatives_at_the_old_polygon_vertex(backend, quantity):
    # The curve's planning polygon has a vertex here.
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        p = torch.tensor(POINT, dtype=torch.float64, requires_grad=True)
        value = objective(solver('torch', quantity), torch, p, quantity)
        gradient, = torch.autograd.grad(value, p)
        value, gradient = float(value.detach()), gradient.detach().numpy()
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _legacy_wavebit.enable_x64()
        s = solver('jax', quantity)
        geometry = s.freeze_geometry() if backend == 'jit' else None
        calculate = jax.value_and_grad(lambda p: objective(s, jnp, p, quantity, geometry))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        with jax.checking_leaks():
            value, gradient = calculate(jnp.asarray(POINT))
        value, gradient = float(value), np.asarray(gradient)
    assert value == pytest.approx(reference(POINT, quantity), abs=2e-11)
    np.testing.assert_allclose(gradient, finite_difference(POINT, quantity),
                               rtol=4e-5, atol=2e-8)


@pytest.mark.parametrize('n', [16, 32])
@pytest.mark.parametrize('quantity', ['epsilon', 'mu'])
def test_both_sides_converge_to_the_same_value_and_derivative(quantity, n):
    # The polygon route jumped by 4.56e-5 here whatever the step at n = 16: a sample crossing
    # a vertex of the planning polygon. At n = 32 two samples also lie on the ellipse's medial
    # segment, where the polygon route and the nearest-point vector jump by 5.9e-5
    # (test_the_nearest_point_vector_would_switch_on_the_medial_segment); the level-set
    # direction is continuous there.
    value = reference(POINT, quantity, n)
    deviations, slopes = [], []
    for h in (1e-5, 1e-6, 1e-7):
        plus, minus = POINT.copy(), POINT.copy()
        plus[2], minus[2] = h, -h
        upper, lower = reference(plus, quantity, n), reference(minus, quantity, n)
        deviations.append(max(abs(upper-value), abs(lower-value)))
        slopes.append([(upper-value)/h, (value-lower)/h])
    assert deviations[1] < .11*deviations[0]
    assert deviations[2] < .11*deviations[1]
    np.testing.assert_allclose(slopes[-1], np.mean(slopes[-2]), rtol=1e-4, atol=3e-6)


def _on_the_medial_segment(n):
    """How many samples of the ``n x n`` grid lie on the test ellipse's medial segment at
    angle 0 -- the stretch of an instance's major axis between the evolute's cusps, where the
    nearest boundary point is two-valued. Counted on the unfolded grid, independently of the
    candidate."""
    lattice = as_lattice(lattice=CELL)
    shapes = tensor_shapes._shifted(layer(POINT))[0]['shapes']
    _, cx, cy, lx, ly, _ = parametric.shape_geometry(shapes[0], lattice)
    a, b = lx/2, ly/2
    X, Y = sample_grid(np, (n, n), lattice, np.float64)
    count = 0
    for m in range(-2, 3):
        for k in range(-2, 3):
            ox, oy = CELL @ np.array([m, k], dtype=float)
            count += np.count_nonzero((Y - cy - oy == 0.) & (np.abs(X - cx - ox) < (a*a - b*b)/a))
    return count


def _nearest_point_direction(xp, xr, yr, a, b, c, s, on, cos_a, sin_a, circle=False):
    """``tensor_shapes._curve_direction`` with the nearest-point vector ``(p - foot) / |p - foot|``
    in place of the level-set direction off the curve: the limit of the polygon route, and
    two-valued on an ellipse's medial segment. Patched in, it is the control that shows the
    checks below can see the switch."""
    one = xp.ones_like(xr)
    nu = xp.where(xr < 0., -one, one)*(b*c)
    nv = xp.where(yr < 0., -one, one)*(a*s)
    side = xp.where(((xr/a)**2 + (yr/b)**2 < 1.) & ~on, -one, one)
    scale = side/xp.sqrt(nu*nu + nv*nv)
    nu, nv = nu*scale, nv*scale
    return nu*cos_a - nv*sin_a, nu*sin_a + nv*cos_a


def test_the_nearest_point_vector_would_switch_on_the_medial_segment(monkeypatch):
    # The control for test_both_sides_converge_to_the_same_value_and_derivative at n = 32:
    # the same grid and steps show a switch where there is one. The nearest-point vector is
    # two-valued on an ellipse's medial segment; at angle 0 the n = 32 grid puts two samples
    # on it (n = 16 none), and turning the ellipse through 0 carries them across. Patched in,
    # it drops the objective by 5.89e-5 whatever the step, and so does the polygon route the
    # curve candidates replaced (5.86e-5 at 1024 vertices), which converges to that vector.
    # The level-set direction moves it by 1.5e-10 at the smaller step, a slope's worth.
    assert (_on_the_medial_segment(16), _on_the_medial_segment(32)) == (0, 2)

    def jump(h):
        plus, minus = POINT.copy(), POINT.copy()
        plus[2], minus[2] = h, -h
        return reference(plus, n=32) - reference(minus, n=32)

    level_set = jump(1e-8)
    with monkeypatch.context() as patch:
        patch.setattr(tensor_shapes, '_curve_direction', _nearest_point_direction)
        jumps = [jump(h) for h in (1e-6, 1e-8)]
    np.testing.assert_allclose(jumps, -5.890e-5, rtol=2e-3)
    assert abs(level_set) < 1e-8
    plan_layer = boundary.plan_layer
    with monkeypatch.context() as patch:
        # The planner's vertex count and the replay's must change together.
        patch.setattr(boundary, 'POLYGON_VERTICES', 1024)
        patch.setattr(boundary, 'plan_layer', partial(plan_layer, n_poly=1024))
        patch.setattr(tensor_shapes, '_curve_fields', lambda *args: {})
        polygon = jump(1e-8)
    assert polygon == pytest.approx(jumps[-1], rel=1e-2)


def _hex_scene(angle=.35, cx=0., cy=5.):
    """A hexagonal cell holding a box (shape 0), a tensor ellipse and a circle."""
    return [dict(lx=60., ly=60., cy=290., material=2.25),
            dict(type='ellipse', lx=420., ly=130., angle=angle, cx=cx, cy=cy, material=TENSOR),
            dict(type='circle', lx=100., cx=-250., cy=60., material=3.)]


def _upright_hex_scene(angle=np.pi/2, cx=0., cy=290.-150.*np.sqrt(3)):
    """The same cell with an upright ellipse whose centre is half the cell's height below
    shape 0's: on every even grid midway between two rows of samples, its medial segment
    across them."""
    return [dict(lx=60., ly=60., cy=290., material=2.25),
            dict(type='ellipse', lx=360., ly=120., angle=angle, cx=cx, cy=cy, material=TENSOR),
            dict(type='circle', lx=100., cx=-250., cy=60., material=3.)]


def _square_scene(angle=0., cx=-150., cy=-100.):
    """A square cell holding a box (shape 0), a level tensor ellipse half a period from it in
    both axes -- on every even grid midway between rows and between columns of samples, its
    medial segment along a row -- and a circle."""
    return [dict(lx=80., ly=80., cx=150., cy=200., material=2.25),
            dict(type='ellipse', lx=280., ly=110., angle=angle, cx=cx, cy=cy, material=TENSOR),
            dict(type='circle', lx=100., cx=150., cy=-150., material=3.)]


#: cell, scene, the ellipse parameter varied, the window's centre and half-width, abscissae
WINDOWS = {
    'hex angle': (HEX, _hex_scene, 'angle', .35, .05, 21),
    'upright hex angle': (HEX, _upright_hex_scene, 'angle', np.pi/2, .05, 21),
    'upright hex cx': (HEX, _upright_hex_scene, 'cx', 0., 20., 101),
    'square cy': (SQUARE, _square_scene, 'cy', -100., 20., 81),
}


def _window(name, n=32):
    """How much sum T changes across a window of one ellipse parameter, and the trapezoid
    integral of its torch gradient on the same abscissae; order 1."""
    torch = pytest.importorskip('torch')
    cell, scene, parameter, centre, half, points = WINDOWS[name]
    xs = centre + np.linspace(-half, half, points)
    values, gradients = [], []
    for x in xs:
        p = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        s = _legacy_wavebit.rcwa(backend='torch', lattice=cell, wavelength=900., thickness=[170.],
                         truncation_order=(1, 1), theta=.17, phi=.3, polarization=(.8, .6j),
                         factorization='nvf', shape_nvf_grid_size=n,
                         structure=[dict(background=1., shapes=scene(**{parameter: p}))])
        value = torch.real(s.solve(auto_detune=False).T.sum())
        gradient, = torch.autograd.grad(value, p)
        values.append(float(value.detach()))
        gradients.append(float(gradient))
    g = np.array(gradients)
    return values[-1] - values[0], float(np.sum(g[1:] + g[:-1])/2*(xs[1] - xs[0]))


def _axis_ratio(name):
    """The window's ellipse's minor over major semi-axis."""
    _, scene, *_ = WINDOWS[name]
    ellipse = scene()[1]
    return min(ellipse['lx'], ellipse['ly'])/max(ellipse['lx'], ellipse['ly'])


def _centre_clearance(name, n=32):
    """The nearest a window carries its ellipse's centre to a sample of the ``n x n`` grid, over
    every periodic image. Counted on the unfolded grid, independently of the candidate."""
    cell, scene, parameter, centre, half, _ = WINDOWS[name]
    lattice = as_lattice(lattice=cell)
    X, Y = sample_grid(np, (n, n), lattice, np.float64)
    clearance = np.inf
    for x in centre + np.linspace(-half, half, 401):
        shapes = tensor_shapes._shifted(dict(shapes=scene(**{parameter: x})))[0]['shapes']
        _, cx, cy, *_ = parametric.shape_geometry(shapes[1], lattice)
        offset = np.linalg.solve(cell, np.stack([X.ravel() - cx, Y.ravel() - cy]))
        offset -= np.round(offset)
        for image in np.array([(m, k) for m in (-1, 0, 1) for k in (-1, 0, 1)]):
            clearance = min(clearance, float(np.min(np.hypot(*(cell @ (offset + image[:, None]))))))
    return clearance


@pytest.mark.parametrize('name', list(WINDOWS))
def test_the_gradient_integrates_to_the_change_in_the_loss_across_a_window(monkeypatch, name):
    # A gradient judged against the loss's own change across a window several samples wide:
    # a central difference with a sub-pixel step stays on one smooth piece and agrees with
    # autodiff by construction. Turning or moving an ellipse under the nearest-point vector
    # carries normal-grid samples across its medial segment, where the vector changes sides and
    # the loss steps; autodiff sees none of the steps, and they do not average away on finer
    # grids. The level-set direction has no medial switch. Measured at n = 32, its integral
    # was within 0.5%, 1.4%, 0.5% and 0.7% of the change on these four windows; with the
    # nearest-point vector patched in, the integral was -0.84, -0.16, 1.30 and -0.17 times
    # the change. At n = 64 the level-set integrals close to within 0.2% on all four, where
    # the nearest-point vector reads -0.77, -0.21, 1.29 and -0.37
    # (docs/guide/rcwa/tensor-shape-integration.md).
    if WINDOWS[name][2] != 'angle':
        # Around its centre the level-set direction turns through a half-turn, continuously:
        # a sample the motion carries past the centre at a distance d turns over a stretch of
        # about d (b/a)**2 of the motion along the minor axis, and d (a/b)**2 along the major,
        # where the gradient peaks roughly as 1 / d. Abscissae coarser than that stretch skip
        # the peak; this guard takes the shorter stretch, whichever way the window moves.
        _, _, _, _, half, points = WINDOWS[name]
        spacing = 2*half/(points - 1)
        assert _centre_clearance(name)*_axis_ratio(name)**2 > 2*spacing
    change, integral = _window(name)
    assert abs(integral/change - 1.) < .04, (change, integral)
    monkeypatch.setattr(tensor_shapes, '_curve_direction', _nearest_point_direction)
    change, integral = _window(name)
    assert abs(integral/change - 1.) > .2, (change, integral)


def test_regression_sum_T_at_order_1_on_a_64_grid(monkeypatch):
    # Regression: sum T on this file's scene at order 1, n = 64. The parent of the curve
    # candidates, the 256-vertex polygon route, read 0.9773203484 here (0.9773154819 at 1024
    # vertices). The nearest-point vector, which that route converges to, reads 0.9773155720:
    # the direction rule, not a resolution, moves this order by 6.6e-4.
    r = solver(n=64).solve(auto_detune=False)
    assert float(np.real(np.sum(r.T))) == pytest.approx(0.9766510524697125, abs=1e-9)
    monkeypatch.setattr(tensor_shapes, '_curve_direction', _nearest_point_direction)
    r = solver(n=64).solve(auto_detune=False)
    assert float(np.real(np.sum(r.T))) == pytest.approx(0.9773155720012778, abs=1e-9)


def test_regression_sum_T_at_order_7_on_a_256_grid():
    # Regression: sum T on this file's scene at order 7, n = 256: 0.9741492, 3.2e-4 above its
    # limit, 0.9738276 +- 5e-5 from orders up to 18 under both rules (n = 512 moves order 12 by
    # < 2e-6). The nearest-point vector read 0.9738408 and the 256-vertex polygon route of the
    # parent 0.9738413. From order 7 on, the nearest-point values all lie within a band 1e-4
    # wide; the level-set values only from order 16, whose solve took 54 s against 1.0 s at
    # order 7. Within +-1e-4 of the limit the nearest-point values settle from order 7 and
    # the level-set values from order 9 (3.0 s) -- the price of a gradient that integrates to
    # the loss (test_the_gradient_integrates_to_the_change_in_the_loss_across_a_window).
    r = solver(n=256, truncation_order=(7, 7)).solve(auto_detune=False)
    assert float(np.real(np.sum(r.T))) == pytest.approx(0.9741492466241387, abs=1e-9)


def test_one_jit_gradient_replays_across_the_old_corner():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    s = solver('jax')
    geometry = s.freeze_geometry()
    calculate = jax.jit(jax.value_and_grad(lambda p: objective(s, jnp, p, geometry=geometry)))
    with jax.checking_leaks():
        for angle in (-1e-5, 0., 1e-5, 0.):
            p = POINT.copy()
            p[2] = angle
            value, gradient = calculate(jnp.asarray(p))
            assert float(value) == pytest.approx(reference(p), abs=2e-11)
            np.testing.assert_allclose(np.asarray(gradient), finite_difference(p),
                                       rtol=4e-5, atol=2e-8)


@pytest.mark.parametrize('kind', ['circle', 'disk', 'ellipse'])
def test_on_curve_samples_read_zero_distance_and_the_outward_normal_on_every_instance(kind):
    lattice = as_lattice(lattice=CELL)
    shapes = [dict(type=kind, lx=190., **({'ly': 130.} if kind == 'ellipse' else {}),
                   angle=.27, cx=15., cy=-23., material=4.),
              dict(type='circle', lx=70., cy=200., material=2.25)]
    curves = tensor_shapes._curve_plan(parametric.plan_gamma(shapes, lattice))
    fields = tensor_shapes._curve_fields(np, shapes, lattice, np.float64, curves)
    # On the curve the candidate carries the outward normal, as a segment's n_edge does (its
    # material on the left), whichever side rounding puts a sample: its distance is within
    # the blend's on-interface threshold, where the level-set sign is noise.
    angle = np.array([0., .13, .47, 1.3, 2.9, np.pi, 4.4, 5.7])
    a, b = 95., 65. if kind == 'ellipse' else 95.
    rotation = np.array([[np.cos(.27), -np.sin(.27)], [np.sin(.27), np.cos(.27)]])
    points = np.stack([a*np.cos(angle), b*np.sin(angle)], axis=-1)@rotation.T
    points += CELL@np.array([.5, .5])+[15., -23.]
    expected = np.stack([np.cos(angle)/a, np.sin(angle)/b], axis=-1)@rotation.T
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    for image in ([0, 0], [1, -2], [-3, 1]):
        distance2, normals = fields[(0, 2)](points+CELL@np.array(image))
        np.testing.assert_allclose(distance2, 0., atol=1e-23)
        np.testing.assert_allclose(normals, expected, atol=2e-14)


def _sample_on_an_ellipse(n=16, b=60., angle=.3):
    """The first sample of the ``n x n`` grid at which an ellipse centred on the cell, of minor
    semi-axis ``b`` and turned by ``angle``, can be sized to pass away from its axes, and the
    major axis that puts it there."""
    lattice = as_lattice(lattice=CELL)
    X, Y = sample_grid(np, (n, n), lattice, np.float64)
    dx, dy = X.ravel() - CELL[0].sum()/2, Y.ravel() - CELL[1].sum()/2
    u, v = dx*np.cos(angle) + dy*np.sin(angle), -dx*np.sin(angle) + dy*np.cos(angle)
    for u_, v_ in zip(u, v):
        if 15. < abs(v_) < .8*b:
            a = abs(u_)/np.sqrt(1. - (v_/b)**2)
            if 150. < a < 250.:
                return 2*a


def _on_curve_transmission(backend, xp, p):
    shapes = [dict(type='ellipse', lx=p[0], ly=120., angle=p[1], material=TENSOR),
              dict(type='rectangle', lx=60., ly=60., cx=-200., cy=250., material=2.25)]
    s = solver(backend, truncation_order=(2, 2), n=16, structure=[dict(background=1., shapes=shapes)])
    return xp.real(s.solve(auto_detune=False).T.sum())


def test_a_sample_on_an_ellipse_differentiates_as_the_field_around_it():
    # An ellipse sized to put a sample of the grid -- and its mirror image through the centre
    # -- on its curve, away from its axes. There the candidate carries the outward normal, read
    # from the level-set gradient at the sample as off the curve: the nearest point's normal
    # has the same value and not the same derivative, and moved the angle slope 8.5% (the
    # size slope 0.7%) off a central difference whose step carries the sample off the curve.
    point = np.array([_sample_on_an_ellipse(), .3])
    lattice = as_lattice(lattice=CELL)
    shapes = tensor_shapes._shifted(dict(shapes=[
        dict(type='ellipse', lx=point[0], ly=120., angle=.3, material=TENSOR),
        dict(type='rectangle', lx=60., ly=60., cx=-200., cy=250., material=2.25)]))[0]['shapes']
    fields = tensor_shapes._curve_fields(np, shapes, lattice, np.float64,
                                         tensor_shapes._curve_plan(parametric.plan_gamma(shapes, lattice)))
    points = boundary._folded_sample_points(np, (16, 16), lattice, np.float64,
                                            boundary.plan_layer(shapes, lattice)[2])
    distance2, _ = fields[(0, 2)](points)
    assert np.count_nonzero(distance2 <= boundary._on_interface_threshold2(np, lattice, np.float64)) == 2
    torch = pytest.importorskip('torch')
    p = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(_on_curve_transmission('torch', torch, p), p)
    expected = []
    for i, h in ((0, 1e-4), (1, 1e-7)):
        plus, minus = point.copy(), point.copy()
        plus[i] += h
        minus[i] -= h
        expected.append((_on_curve_transmission('numpy', np, plus) - _on_curve_transmission('numpy', np, minus))/(2*h))
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=1e-4)


@pytest.mark.parametrize('quantity', ['epsilon', 'mu'])
@pytest.mark.parametrize('circle', ['circle', 'disk'])
def test_disjoint_primitive_coefficients_do_not_depend_on_polygon_resolution(monkeypatch, quantity, circle):
    coefficients = []
    plan_layer = boundary.plan_layer
    for facets in (32, 256):
        monkeypatch.setattr(boundary, 'POLYGON_VERTICES', facets)
        monkeypatch.setattr(boundary, 'plan_layer', partial(plan_layer, n_poly=facets))
        s = solver(quantity=quantity, structure=[layer(POINT, quantity, circle)])
        coefficients.append(s.eps_conv_matrices() if quantity == 'epsilon' else s.mu_conv_matrices())
    for actual, expected in zip(*coefficients):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0.)


@pytest.mark.parametrize('quantity', ['epsilon', 'mu'])
@pytest.mark.parametrize('ellipse', ['in-cell', 'ring'])
def test_an_ellipse_as_the_only_curve_takes_the_analytic_candidate(monkeypatch, quantity, ellipse):
    # Every other scene here holds a circle or disk; an ellipse beside a rectangle alone must
    # also be planned onto the analytic candidate rather than back onto its polygon.
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y').epsilon
    material = wavebit.Material(tensor) if quantity == 'epsilon' else wavebit.Material(1., tensor)
    other = 2.25 if quantity == 'epsilon' else wavebit.Material(1., 2.25)
    size = dict(lx=300., ly=120., angle=1.1) if ellipse == 'in-cell' else dict(lx=787.5, ly=100.)
    shapes = [dict(type='ellipse', material=material, **size),
              dict(type='rectangle', lx=80., ly=50., cx=250., cy=260., material=other)]
    plan = tensor_shapes._plan(tensor_shapes._shifted(dict(shapes=shapes))[0],
                               as_lattice(lattice=CELL), 0)
    assert plan[0] == 'disjoint' and [i for i, _ in plan[2][1]] == [0]
    coefficients = []
    plan_layer = boundary.plan_layer
    for facets in (32, 256):
        # The planner's vertex count and the replay's must change together.
        monkeypatch.setattr(boundary, 'POLYGON_VERTICES', facets)
        monkeypatch.setattr(boundary, 'plan_layer', partial(plan_layer, n_poly=facets))
        s = solver(quantity=quantity, structure=[dict(background=1., shapes=shapes)])
        coefficients.append(s.eps_conv_matrices() if quantity == 'epsilon' else s.mu_conv_matrices())
    for actual, expected in zip(*coefficients):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0.)


def test_implicit_rectangle_type_is_supported_with_a_curve():
    shapes = [dict(lx=180., ly=150., cx=-100., material=wavebit.Material([2., 3., 4.])),
              dict(type='disk', lx=80., cx=170., material=2.25)]
    implicit = solver(structure=[dict(background=1., shapes=shapes)]).eps_conv_matrices()
    explicit = solver(structure=[dict(background=1., shapes=[dict(shapes[0], type='rectangle'),
                                                            shapes[1]])]).eps_conv_matrices()
    for a, b in zip(implicit, explicit):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('n', [16, 32])
def test_electric_magnetic_duality(n):
    results = []
    for quantity in ('epsilon', 'mu'):
        scene = layer(POINT, quantity)
        material = scene['shapes'][0]['material']
        tensor = np.real(material.epsilon if quantity == 'epsilon' else material.mu_value)
        scene['shapes'][0]['material'] = (wavebit.Material(tensor) if quantity == 'epsilon'
                                          else wavebit.Material(1., tensor))
        r = solver(n=n, structure=[scene]).solve(auto_detune=False)
        results.append(r)
    for actual, dual in ((results[0].s_inc, results[1].p_inc),
                        (results[0].p_inc, results[1].s_inc)):
        np.testing.assert_allclose(actual.R, dual.R, atol=3e-11, rtol=0.)
        np.testing.assert_allclose(actual.T, dual.T, atol=3e-11, rtol=0.)


def scaled_transmission(backend, xp, p, scale, precision=1, geometry=None):
    scene = layer(p)
    for shape in scene['shapes']:
        for key in ('lx', 'ly', 'cx', 'cy'):
            if key in shape:
                shape[key] = shape[key]*scale
    dtype = xp.float32 if precision else xp.float64
    s = solver(backend, structure=[scene], complex_dtype=precision,
               lattice=xp.asarray(CELL, dtype=dtype)*scale,
               wavelength=900.*scale, thickness=[170.*scale])
    return xp.real(s.solve(geometry=geometry, auto_detune=False).T.sum())


@pytest.mark.parametrize('backend', ['jax', 'jit', 'torch'])
def test_float32_boundary_gradients_are_finite_in_different_length_units(backend):
    # A tiny nonzero distance residual may overflow a discarded inverse-distance
    # weight. Gradients use unscaled design variables so unit changes must cancel.
    expected = []
    for i, h in enumerate((.001, .001, 1e-6)):
        plus, minus = POINT.copy(), POINT.copy()
        plus[i] += h
        minus[i] -= h
        expected.append((scaled_transmission('numpy', np, plus, 1., 0)
                         -scaled_transmission('numpy', np, minus, 1., 0))/(2*h))
    expected_value = scaled_transmission('numpy', np, POINT, 1., 0)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        for scale in (1., 1e-3, 1e-6):
            p = torch.tensor(POINT, dtype=torch.float32, requires_grad=True)
            value = scaled_transmission('torch', torch, p, scale)
            gradient, = torch.autograd.grad(value, p)
            assert float(value.detach()) == pytest.approx(expected_value, abs=3e-6)
            np.testing.assert_allclose(gradient.detach().numpy(), expected, rtol=3e-3, atol=2e-6)
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        geometry = solver().freeze_geometry()
        calculate = jax.value_and_grad(
            lambda p, scale: scaled_transmission('jax', jnp, p, scale, geometry=geometry))
        if backend == 'jit':
            calculate = jax.jit(calculate)
        with jax.checking_leaks():
            for scale in (1., 1e-3, 1e-6):
                value, gradient = calculate(jnp.asarray(POINT, dtype=jnp.float32),
                                             jnp.asarray(scale, dtype=jnp.float32))
                assert float(value) == pytest.approx(expected_value, abs=3e-6)
                np.testing.assert_allclose(np.asarray(gradient), expected, rtol=3e-3, atol=2e-6)


def zero_strength_transmission(backend, xp, p, scale, precision=1, geometry=None):
    """Each pass holds a curve with zero contrast: the magnetic-only circle in the
    permittivity pass and the dielectric-only one in the permeability pass. The centred
    circle's radius is two samples of the odd grid, so samples lie on it."""
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y').epsilon
    dtype = xp.float32 if precision else xp.float64
    s = _legacy_wavebit.rcwa(backend=backend, lattice=xp.asarray(np.diag([600., 600.]), dtype=dtype)*scale,
                     wavelength=900.*scale, thickness=[170.*scale], truncation_order=(1, 1),
                     theta=.17, phi=.3, polarization=(.8, .6j), factorization='nvf',
                     shape_nvf_grid_size=15, complex_dtype=precision,
                     structure=[dict(background=1., shapes=[
                         dict(type='circle', lx=p[0]*scale, material=wavebit.Material(1., 2.25)),
                         dict(type='circle', cx=220.*scale, lx=p[1]*scale,
                              material=wavebit.Material(tensor))])])
    return xp.real(s.solve(geometry=geometry, auto_detune=False).T.sum())


@pytest.mark.parametrize('backend', ['jit', 'torch'])
def test_float32_gradients_are_finite_with_a_zero_strength_curve(backend):
    # Each pass holds a curve candidate of zero strength, whose inverse distance the blend
    # discards, and samples on the centred circle leave its distance a tiny residual: the
    # float32 guard must form neither, or the discarded cotangent is 0 * inf.
    point = np.array([160., 100.])
    expected = []
    for i in range(2):
        plus, minus = point.copy(), point.copy()
        plus[i] += 1e-3
        minus[i] -= 1e-3
        expected.append((zero_strength_transmission('numpy', np, plus, 1., 0)
                         -zero_strength_transmission('numpy', np, minus, 1., 0))/2e-3)
    expected_value = zero_strength_transmission('numpy', np, point, 1., 0)
    scales = (1., 1e-3, 1e-6, 1e-9)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        for scale in scales:
            p = torch.tensor(point, dtype=torch.float32, requires_grad=True)
            value = zero_strength_transmission('torch', torch, p, scale)
            gradient, = torch.autograd.grad(value, p)
            assert float(value.detach()) == pytest.approx(expected_value, abs=5e-6)
            np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-3, atol=1e-9)
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _legacy_wavebit.enable_x64()
        geometry = _legacy_wavebit.rcwa(
            backend='numpy', lattice=np.diag([600., 600.]), wavelength=900., thickness=[170.],
            truncation_order=(1, 1), factorization='nvf', shape_nvf_grid_size=15,
            structure=[dict(background=1., shapes=[
                dict(type='circle', lx=160., material=wavebit.Material(1., 2.25)),
                dict(type='circle', cx=220., lx=100., material=wavebit.Material(
                    wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y').epsilon))])],
        ).freeze_geometry()
        calculate = jax.jit(jax.value_and_grad(
            lambda p, scale: zero_strength_transmission('jax', jnp, p, scale, geometry=geometry)))
        with jax.checking_leaks():
            for scale in scales:
                value, gradient = calculate(jnp.asarray(point, dtype=jnp.float32),
                                             jnp.asarray(scale, dtype=jnp.float32))
                assert float(value) == pytest.approx(expected_value, abs=5e-6)
                np.testing.assert_allclose(np.asarray(gradient), expected, rtol=2e-3, atol=1e-9)


def off_centre_transmission(backend, xp, p, scale, precision=1, geometry=None):
    """A centred circle beside an off-centre tensor ellipse; ``p`` is the ellipse's lx, ly
    and cx."""
    dtype = xp.float32 if precision else xp.float64
    s = _legacy_wavebit.rcwa(backend=backend, lattice=xp.asarray(CELL, dtype=dtype)*scale,
                     wavelength=900.*scale, thickness=[170.*scale], truncation_order=(1, 1),
                     theta=.17, phi=.3, polarization=(.8, .6j), factorization='nvf',
                     shape_nvf_grid_size=16, complex_dtype=precision,
                     structure=[dict(background=1., shapes=[
                         dict(type='circle', lx=150.*scale, material=2.25),
                         dict(type='ellipse', cx=p[2]*scale, cy=250.*scale, lx=p[0]*scale,
                              ly=p[1]*scale, angle=.3, material=TENSOR)])])
    return xp.real(s.solve(geometry=geometry, auto_detune=False).T.sum())


@pytest.mark.parametrize('backend', ['jit', 'torch'])
def test_float32_gradients_of_an_off_centre_ellipse_are_finite_in_any_length_unit(backend):
    # The kept inverse distances of the blend: samples near the ellipse leave squared
    # distances that, in metres, square below float32's smallest normal number, which JAX
    # flushes to zero. Formed in units of the cell they do not; without that the jitted
    # gradient is NaN at 1e-9.
    point = np.array([300., 110., 120.])
    expected = []
    for i in range(3):
        plus, minus = point.copy(), point.copy()
        plus[i] += 1e-3
        minus[i] -= 1e-3
        expected.append((off_centre_transmission('numpy', np, plus, 1., 0)
                         - off_centre_transmission('numpy', np, minus, 1., 0))/2e-3)
    expected_value = off_centre_transmission('numpy', np, point, 1., 0)
    scales = (1., 1e-3, 1e-6, 1e-9)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        for scale in scales:
            p = torch.tensor(point, dtype=torch.float32, requires_grad=True)
            value = off_centre_transmission('torch', torch, p, scale)
            gradient, = torch.autograd.grad(value, p)
            assert float(value.detach()) == pytest.approx(expected_value, abs=5e-6)
            np.testing.assert_allclose(gradient.numpy(), expected, rtol=2e-3, atol=2e-9)
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _legacy_wavebit.enable_x64()
        geometry = _legacy_wavebit.rcwa(
            backend='numpy', lattice=CELL, wavelength=900., thickness=[170.], truncation_order=(1, 1),
            factorization='nvf', shape_nvf_grid_size=16, structure=[dict(background=1., shapes=[
                dict(type='circle', lx=150., material=2.25),
                dict(type='ellipse', cx=120., cy=250., lx=300., ly=110., angle=.3,
                     material=TENSOR)])]).freeze_geometry()
        calculate = jax.jit(jax.value_and_grad(
            lambda p, scale: off_centre_transmission('jax', jnp, p, scale, geometry=geometry)))
        with jax.checking_leaks():
            for scale in scales:
                value, gradient = calculate(jnp.asarray(point, dtype=jnp.float32),
                                             jnp.asarray(scale, dtype=jnp.float32))
                assert float(value) == pytest.approx(expected_value, abs=5e-6)
                np.testing.assert_allclose(np.asarray(gradient), expected, rtol=2e-3, atol=2e-9)


CENTRED_CURVES = {
    'ring ellipse': dict(type='ellipse', lx=787.5, ly=100.),
    'circle': dict(type='circle', lx=150.),
    'in-cell ellipse': dict(type='ellipse', lx=300., ly=120., angle=1.1),
}


def centred_curve(kind, p):
    """Shape 0 of kind ``kind`` and diameter (or major axis) ``p[0]`` beside a circle of
    diameter ``p[1]`` centred at ``x = p[2]``, 40 placing it above shape 0's centre."""
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y').epsilon
    return [dict(CENTRED_CURVES[kind], lx=p[0], cx=40., cy=-30.,
                 material=wavebit.Material(tensor, tensor)),
            dict(type='circle', cx=p[2], cy=170., lx=p[1], material=2.25)]


@pytest.mark.parametrize('n', [15, 17])
@pytest.mark.parametrize('kind', list(CENTRED_CURVES))
def test_an_odd_grid_samples_a_curve_centre(kind, n):
    # The integrated build moves shape 0 to the cell centre, and an odd grid samples that
    # centre exactly, where a curve candidate may have no natural direction.
    point = np.array([CENTRED_CURVES[kind]['lx'], 70., 40.])
    shapes = centred_curve(kind, point)
    lattice = as_lattice(lattice=CELL)
    centred = tensor_shapes._shifted(dict(shapes=shapes))[0]['shapes']
    fields = tensor_shapes._curve_fields(
        np, centred, lattice, np.float64,
        tensor_shapes._curve_plan(parametric.plan_gamma(centred, lattice)))
    points = boundary._folded_sample_points(np, (n, n), lattice, np.float64,
                                            boundary.plan_layer(centred, lattice)[2])
    offset = np.linalg.solve(CELL, (points - CELL@np.array([.5, .5])).T)
    at_centre = np.all(np.abs(offset - np.round(offset)) < 1e-12, axis=0)
    assert np.count_nonzero(at_centre) == 1, 'the grid must sample an image of the centre'
    _, direction = fields[(0, 2)](points[at_centre])
    assert np.all(np.isfinite(direction)) and np.isclose(np.linalg.norm(direction), 1.)
    s = solver(n=n, structure=[dict(background=1., shapes=shapes)])
    for matrices in (s.eps_conv_matrices(), s.mu_conv_matrices()):
        assert all(np.all(np.isfinite(np.asarray(m))) for m in matrices if m is not None)
    r = s.solve(auto_detune=False)
    assert np.all(np.isfinite(r.R)) and np.all(np.isfinite(r.T))
    assert .5 < float(np.sum(r.R) + np.sum(r.T)) < 1.
    if n != 15:
        return
    # And the derivative there. The signed-fold step of ``_ellipse_nearest_point`` divides by
    # G' only where G' < 0, and a circle's G' vanishes at its centre: without that guard every
    # value above stays finite and the gradient is NaN. The second circle sits on a column of
    # samples, on its vertical axis, and one sample lies on its diagonal: moving it sideways
    # turns their normals, which autodiff reads through that step's signed fold -- on the
    # diagonal, where the bisection meets the root exactly, only if the step may leave the
    # bracket by rounding (beside the circle, the slope read -1.2468e-7 against -1.1947e-7 when
    # it could not, and -1.113e-6 with no step at all).
    torch = pytest.importorskip('torch')

    def transmission(backend, xp, p):
        s = solver(backend, n=n, structure=[dict(background=1., shapes=centred_curve(kind, p))])
        return xp.real(s.solve(auto_detune=False).T.sum())
    expected = []
    for i in range(3):
        plus, minus = point.copy(), point.copy()
        plus[i] += 1e-3
        minus[i] -= 1e-3
        expected.append((transmission('numpy', np, plus) - transmission('numpy', np, minus))/2e-3)
    p = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(transmission('torch', torch, p), p)
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=1e-5, atol=1e-10)


def _turned_circle_transmission(backend, xp, angle, n):
    """sum T at order 1 with shape 0, a tensor circle, turned by ``angle``."""
    shapes = centred_curve('circle', np.array([150., 70., 40.]))
    shapes[0]['angle'] = angle
    s = solver(backend, n=n, structure=[dict(background=1., shapes=shapes)])
    return xp.real(s.solve(auto_detune=False).T.sum())


@pytest.mark.parametrize('n', [15, 16])
def test_a_circle_does_not_turn_with_its_angle(n):
    # A circle's angle describes nothing, and its candidate is read in an unrotated frame. At
    # its centre, which an odd grid samples when the circle is shape 0, there is no radial
    # direction and the candidate carries a fixed vector; read in the circle's own frame,
    # that vector turned with the angle. In the circle's own frame sum T moved by 3.1e-4
    # across these angles at n = 15, with a slope of 3.3e-4 that a central difference
    # confirmed, and by 4e-15 of rounding at n = 16; unrotated, nothing reads the angle and
    # every value is the same to the bit.
    values = [_turned_circle_transmission('numpy', np, angle, n) for angle in (0., .4, 1.1, 2.3)]
    assert len(set(values)) == 1, values
    torch = pytest.importorskip('torch')
    angle = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    slope, = torch.autograd.grad(_turned_circle_transmission('torch', torch, angle, n), angle)
    assert float(slope) == 0., float(slope)


def _round_beside_a_box(kind, material):
    """The eps convolution matrices and sum T of a single-image round shape beside a box."""
    s = solver(lattice=SQUARE, n=32, truncation_order=(2, 2), structure=[dict(
        background=1., shapes=[dict(type=kind, lx=300., material=material),
                               dict(lx=100., ly=100., cx=230., cy=230., material=4.)])])
    matrices = [np.asarray(m) for m in s.eps_conv_matrices() if m is not None]
    return matrices, float(np.real(np.sum(s.solve(auto_detune=False).T)))


@pytest.mark.parametrize('material', ['scalar', 'tensor'])
def test_a_disk_beside_another_shape_is_a_circle_to_the_bit(material):
    # 'disk' spells 'circle'. The curve candidate dispatches on the type it is handed, so a
    # dispatch that missed the alias would send a disk through the ellipse branch: the 3x3
    # and the level-set direction, the same radial vector to rounding but not to the bit. The
    # scalar layer is the same check on the blend's round piece.
    material = TENSOR if material == 'tensor' else 12.25
    circle, disk = _round_beside_a_box('circle', material), _round_beside_a_box('disk', material)
    assert disk[1] == circle[1]
    assert len(disk[0]) == len(circle[0])
    assert all(np.array_equal(a, b) for a, b in zip(disk[0], circle[0]))


LONE_ELLIPSES = {
    # Leaving its Wigner-Seitz cell: both paths take the instance at the least exact distance.
    'ring': (CELL, dict(type='ellipse', lx=787.5, ly=100., angle=0.)),
    # Inside its cell: the one-primitive path reads the nearest centre's instance and the
    # disjoint path the nearest boundary's, which are one at every sample here (asserted).
    'in-cell': (SQUARE, dict(type='ellipse', lx=240., ly=200., angle=.3, cx=-60., cy=20.)),
}


def _lone_and_beside(cell, ellipse, n):
    """The eps convolution matrices of a lone tensor ellipse, and of the same ellipse beside a
    small box of the background's own material at the far corner of its cell."""
    corner = cell @ np.array([.5, .5])
    box = dict(lx=20., ly=20., cx=ellipse.get('cx', 0.)+corner[0],
               cy=ellipse.get('cy', 0.)+corner[1], material=1.)
    out = []
    for shapes in ([ellipse], [ellipse, box]):
        s = solver(lattice=cell, truncation_order=(3, 3), n=n,
                   structure=[dict(background=1., shapes=shapes)])
        out.append([np.asarray(m) for m in s.eps_conv_matrices() if m is not None])
    return out


@pytest.mark.parametrize('scene', list(LONE_ELLIPSES))
def test_a_lone_ellipse_and_the_same_ellipse_beside_a_shape_of_no_contrast_agree(
        monkeypatch, scene):
    # A lone ellipse takes the one-primitive path: its own level set, through
    # parametric.normal_vector_field_analytical. Beside a shape of the background's own
    # material it takes the disjoint path, whose blend gives that shape no weight, so the
    # ellipse's curve candidate alone sets the normal. Two builds that share the level-set
    # rule and nothing past the shape geometry: measured 4e-16 and 7e-16 apart in the
    # convolution matrices, where the nearest-point vector, patched in, is 4.0e-2 and 1.6e-2
    # away. Where an in-cell ellipse's nearest centre is not its nearest boundary -- 9-25% of
    # a 32-point grid's samples on three such scenes -- the two read different instances and
    # differed by 1.8e-5 to 1.1e-4 (docs/guide/rcwa/tensor-shape-integration.md).
    cell, ellipse = LONE_ELLIPSES[scene]
    ellipse = dict(ellipse, material=TENSOR)
    if scene == 'in-cell':
        lattice = as_lattice(lattice=cell)
        shapes = tensor_shapes._shifted(dict(shapes=[ellipse]))[0]['shapes']
        points = boundary._folded_sample_points(np, (32, 32), lattice, np.float64,
                                                boundary.plan_layer(shapes, lattice)[2])
        _, centre_x, centre_y, *_ = parametric.shape_geometry(shapes[0], lattice)
        _, nearest, _ = _brute_force_boundary_distance(points, shapes[0], lattice)
        images = np.array([(m, k) for m in range(-2, 3) for k in range(-2, 3)])
        centres = np.array([centre_x, centre_y]) + images @ np.asarray(lattice.columns, dtype=float).T
        nearest_centre = images[np.argmin(np.hypot(*(points[:, None, :] - centres).T).T, axis=1)]
        assert np.array_equal(nearest, nearest_centre), 'the two instances must be one'
    lone, beside = _lone_and_beside(cell, ellipse, 32)
    assert _gap(lone, beside) < 1e-12
    monkeypatch.setattr(tensor_shapes, '_curve_direction', _nearest_point_direction)
    lone, beside = _lone_and_beside(cell, ellipse, 32)
    assert _gap(lone, beside) > 1e-3


OFF_CURVE_SCENES = {
    # Every single-image piece below has samples that a farther periodic instance would
    # misread: a circle folded to its parallelogram image rather than its nearest centre,
    # and an ellipse read at its nearest *centre*, whose boundary is not the nearest one.
    'hex disc': (HEX, [dict(type='circle', lx=520.),
                       dict(type='rectangle', lx=40., ly=40., cx=300., cy=100.*np.sqrt(3))]),
    'hex in-cell ellipse': (HEX, [dict(type='ellipse', lx=560., ly=90., angle=.5),
                                  dict(type='rectangle', lx=40., ly=40., cx=300.,
                                       cy=100.*np.sqrt(3)+60.)]),
    'oblique in-cell ellipse and disk': (CELL, [
        dict(type='ellipse', lx=300., ly=120., angle=1.1, cx=20., cy=-15.),
        dict(type='disk', lx=90., cx=250., cy=260.)]),
    'square tilted ellipse': (np.diag([600., 600.]), [
        dict(type='ellipse', lx=560., ly=40., angle=.5),
        dict(type='circle', lx=60., cx=-250., cy=250.)]),
    # A ring piece (the ellipse leaves its Wigner-Seitz cell) beside a single-image circle.
    'oblique ring ellipse': (CELL, [dict(type='ellipse', lx=787.5, ly=100.),
                                    dict(type='circle', cy=200., lx=70.)]),
}


def _brute_force_boundary_distance(points, shape, lattice, samples=2000, ring=3):
    """Distance to a fine inscribed polygon over a ring of images, the nearest image, and
    the distance to the second-nearest image.

    Independent of the analytic distance and of every fold. Sampled uniformly in the
    parameter, the chords' sagitta is below 5e-4 on every curve used here (4.9e-4 at the
    oblique ring ellipse's thin tips), which bounds the difference from the exact distance.
    """
    _, cx, cy, lx, ly, angle = parametric.shape_geometry(shape, lattice)
    t = np.linspace(0., 2*np.pi, samples, endpoint=False)
    c, s = np.cos(angle), np.sin(angle)
    ax = lx/2*np.cos(t)*c - ly/2*np.sin(t)*s + cx
    ay = lx/2*np.cos(t)*s + ly/2*np.sin(t)*c + cy
    ex, ey = np.roll(ax, -1) - ax, np.roll(ay, -1) - ay
    basis = np.asarray(lattice.columns, dtype=float)
    best = second = np.full(len(points), np.inf)
    nearest = np.zeros((len(points), 2), dtype=int)
    for m in range(-ring, ring+1):
        for n in range(-ring, ring+1):
            ox, oy = basis @ np.array([m, n], dtype=float)
            dx, dy = points[:, :1]-ax-ox, points[:, 1:]-ay-oy
            u = np.clip((dx*ex + dy*ey)/(ex*ex + ey*ey), 0., 1.)
            d2 = np.min((dx - u*ex)**2 + (dy - u*ey)**2, axis=1)
            take = d2 < best
            second = np.where(take, best, np.minimum(second, d2))
            best = np.where(take, d2, best)
            nearest[take] = (m, n)
    return np.sqrt(best), nearest, np.sqrt(second)


def _level_set_direction(points, shape, lattice, nearest):
    """``sign(f) grad f / |grad f|`` of the instance ``nearest`` of a round shape, for
    ``f = (u/a)^2 + (v/b)^2 - 1`` in its own frame, and ``|grad f|``.

    In closed form from the shape's geometry, at the instance the brute-force ring found
    nearest: independent of the candidate's fold, instance sweep and spelling.
    """
    _, cx, cy, lx, ly, angle = parametric.shape_geometry(shape, lattice)
    c, s = np.cos(angle), np.sin(angle)
    a, b = lx/2, ly/2
    d = points - np.array([cx, cy]) - nearest @ np.asarray(lattice.columns, dtype=float).T
    u, v = d[:, 0]*c + d[:, 1]*s, -d[:, 0]*s + d[:, 1]*c
    gu, gv = 2*u/a**2, 2*v/b**2
    gradient = np.hypot(gu, gv)
    side = np.sign((u/a)**2 + (v/b)**2 - 1.)
    gu, gv = side*gu/gradient, side*gv/gradient
    return np.stack([gu*c - gv*s, gu*s + gv*c], axis=-1), gradient


@pytest.mark.parametrize('scene', list(OFF_CURVE_SCENES))
def test_curve_candidates_read_the_nearest_periodic_instance_off_the_curve(scene):
    cell, shapes = OFF_CURVE_SCENES[scene]
    shapes = [dict(s, material=2.25) for s in shapes]
    lattice = as_lattice(lattice=cell)
    curves = tensor_shapes._curve_plan(parametric.plan_gamma(shapes, lattice))
    transform = boundary.plan_layer(shapes, lattice)[2]
    points = boundary._folded_sample_points(np, (32, 32), lattice, np.float64, transform)
    fields = tensor_shapes._curve_fields(np, shapes, lattice, np.float64, curves)
    assert fields
    for (index, _), field in fields.items():
        distance2, direction = field(points)
        distance = np.sqrt(distance2)
        brute, nearest, runner_up = _brute_force_boundary_distance(points, shapes[index], lattice)
        # A farther instance misreads by tens to hundreds of nm on these scenes.
        np.testing.assert_allclose(distance, brute, rtol=0., atol=1e-3, err_msg=scene)
        # The nearest instance's level-set direction, pointing from the curve toward the
        # sample: inward inside the shape, as a segment candidate's vector does. Radial for a
        # circle; for an ellipse not the nearest-point vector, which is up to 62-78 degrees off
        # here. Samples equidistant from two instances may take either; on the curve the
        # candidate carries the outward normal (the test above), and at a centre, where the
        # gradient vanishes, the minor axis. Elsewhere the agreement measured 4.4e-16.
        expected, gradient = _level_set_direction(points, shapes[index], lattice, nearest)
        decided = (runner_up - brute > 1e-2) & (brute > 1e-3) & (gradient > 1e-9)
        assert np.count_nonzero(decided) > .9*len(points)
        np.testing.assert_allclose(direction[decided], expected[decided], rtol=0., atol=1e-12,
                                   err_msg=scene)


def in_cell_solver(backend, p):
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y')
    shapes = [dict(type='ellipse', lx=p[0], ly=p[1], angle=p[2], cx=p[3], cy=p[4], material=tensor),
              dict(type='disk', lx=90., cx=250., cy=260., material=2.25)]
    return solver(backend, truncation_order=(2, 2), n=32, structure=[dict(background=1., shapes=shapes)])


def in_cell_transmission(backend, xp, p, geometry=None):
    return xp.real(in_cell_solver(backend, p).solve(geometry=geometry, auto_detune=False).T.sum())


@pytest.mark.parametrize('backend', ['jit', 'torch'])
def test_gradients_through_the_neighbour_sweep_of_an_in_cell_ellipse(backend):
    # A single-image ellipse takes the hard minimum over its 3x3 neighbours, a disk its
    # nearest centre: both are static in the plan and differentiable through the minimum.
    point = np.array([300., 120., 1.1, 20., -15.])
    expected = []
    for i, h in enumerate((1e-3, 1e-3, 1e-6, 1e-3, 1e-3)):
        plus, minus = point.copy(), point.copy()
        plus[i] += h
        minus[i] -= h
        expected.append((in_cell_transmission('numpy', np, plus)-in_cell_transmission('numpy', np, minus))/(2*h))
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        p = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(in_cell_transmission('torch', torch, p), p)
        gradient = gradient.numpy()
    else:
        jax = pytest.importorskip('jax')
        import jax.numpy as jnp
        _legacy_wavebit.enable_x64()
        geometry = in_cell_solver('numpy', point).freeze_geometry()
        with jax.checking_leaks():
            gradient = np.asarray(jax.jit(jax.grad(
                lambda p: in_cell_transmission('jax', jnp, p, geometry)))(jnp.asarray(point)))
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-10)


POLYGON_ORACLE_SCENES = {
    # The disc is not shape 0 and the rectangle is nudged off the hexagon's symmetry. With
    # the disc at the grid's centre, samples on the hexagonal diagonal are exactly equidistant
    # from two disc instances, where either is a valid nearest-interface pick: a tie, whose
    # O(1) dyad difference at a few samples would hide the convergence this test measures.
    'hex disc': (HEX, [dict(type='rectangle', lx=40., ly=40., cx=307.3, cy=100.*np.sqrt(3)-4.1,
                            material=2.25),
                       dict(type='circle', lx=520., material=TENSOR)]),
    'oblique disk': (CELL, [dict(type='rectangle', lx=60., ly=40., cx=261.9, cy=253.7,
                                 material=2.25),
                            dict(type='disk', lx=380., material=TENSOR)]),
}


def _oracle_coefficients(monkeypatch, cell, shapes, facets=None):
    """The eps convolution matrices of a disjoint layer at order 2 on a 32-point grid, through
    the analytic curve candidates or, given ``facets``, the polygon route they replaced."""
    plan_layer = boundary.plan_layer
    with monkeypatch.context() as patch:
        if facets is not None:
            # The planner's vertex count and the replay's must change together.
            patch.setattr(boundary, 'POLYGON_VERTICES', facets)
            patch.setattr(boundary, 'plan_layer', partial(plan_layer, n_poly=facets))
            patch.setattr(tensor_shapes, '_curve_fields', lambda *args: {})
        s = solver(lattice=cell, truncation_order=(2, 2), n=32,
                   structure=[dict(background=1., shapes=shapes)])
        return [np.asarray(m) for m in s.eps_conv_matrices() if m is not None]


def _gap(left, right):
    return max(np.max(np.abs(a - b)) for a, b in zip(left, right))


@pytest.mark.parametrize('scene', list(POLYGON_ORACLE_SCENES))
def test_analytic_curve_coefficients_are_the_limit_of_the_polygon_route(monkeypatch, scene):
    # An independent route to the same operator: the polygon candidate the analytic one
    # replaced, whose facets converge to the curve and whose vectors are the nearest-point
    # ones -- for a circle the level-set direction itself. The disc dominates the blend inside
    # itself, so the limit does not depend on which way a circle's candidate points there. A
    # farther periodic instance leaves a floor that more facets do not remove (1.1e-3 on the
    # hex scene).
    cell, shapes = POLYGON_ORACLE_SCENES[scene]
    analytic = _oracle_coefficients(monkeypatch, cell, shapes)
    gaps = [_gap(analytic, _oracle_coefficients(monkeypatch, cell, shapes, facets))
            for facets in (512, 8192)]
    # Facet normals converge as 1 / facets: 16 times closer, measured 15-16.
    assert gaps[1] < 3e-5 and gaps[0] > 8*gaps[1], gaps


WEAK_ELLIPSE = [dict(type='ellipse', lx=320., ly=110., angle=.7, cx=-40., cy=-25.,
                     material=wavebit.Material([1.3, 1.2, 1.25])),
                dict(type='rectangle', lx=90., ly=200., angle=.7, cx=150., cy=150., material=TENSOR)]


def test_the_polygon_route_converges_to_the_nearest_point_vector_and_not_to_the_level_set(
        monkeypatch):
    # A weak ellipse beside a strong box: inside the ellipse the box carries mass as well, so
    # the coefficients depend on the direction the ellipse's candidate carries and on which
    # way it points. Patched in, the nearest-point vector is the polygon route's limit, as the
    # disc scenes above are -- the oracle can tell the rules apart. The level-set direction
    # stays 3.9e-4 from it at both facet counts (the nearest-point vector: 2.6e-6 and 1.7e-7):
    # the rule, not a resolution.
    polygon = [_oracle_coefficients(monkeypatch, CELL, WEAK_ELLIPSE, facets)
               for facets in (512, 8192)]
    level_set = [_gap(_oracle_coefficients(monkeypatch, CELL, WEAK_ELLIPSE), p) for p in polygon]
    with monkeypatch.context() as patch:
        patch.setattr(tensor_shapes, '_curve_direction', _nearest_point_direction)
        nearest_point = [_gap(_oracle_coefficients(monkeypatch, CELL, WEAK_ELLIPSE), p)
                         for p in polygon]
    assert nearest_point[1] < 3e-5 and nearest_point[0] > 8*nearest_point[1], nearest_point
    np.testing.assert_allclose(level_set, [3.886e-4, 3.869e-4], rtol=1e-2)


def stale_solver(backend, shapes, cell):
    if backend == 'jax':
        _legacy_wavebit.enable_x64()
    return _legacy_wavebit.rcwa(backend=backend, lattice=cell, wavelength=900., thickness=[170.],
                        truncation_order=(1, 1), theta=.17, phi=.3, polarization=(.8, .6j),
                        factorization='nvf', shape_nvf_grid_size=16,
                        structure=[dict(background=1., shapes=shapes)])


def growing_box(lx):
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y')
    return [dict(type='circle', lx=200., cx=160., material=2.25),
            dict(type='rectangle', lx=lx, ly=280., cx=-150., material=tensor)]


def growing_ellipse(lx, ly=200.):
    tensor = wavebit.Material([3.8+.04j, 4.2+.06j, 4.5+.08j]).rotated(.2, 'y')
    return [dict(type='ellipse', lx=lx, ly=ly, angle=float(np.radians(10.)), material=tensor),
            dict(type='rectangle', lx=60., ly=60., cy=250., material=2.25)]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_box_crossing_its_own_ring_threshold_replays_as_a_fresh_solve(backend):
    # The rectangle's image list grows (9 -> 13): the full interface-set plan changes, the
    # curve candidates read nothing of it, and the frozen record must still replay.
    if backend == 'torch':
        pytest.importorskip('torch')
    cell = np.diag([600., 600.])
    lattice = as_lattice(lattice=cell)
    before, after = (parametric.plan_gamma(growing_box(w), lattice) for w in (200., 220.))
    assert before != after
    assert tensor_shapes._curve_plan(before) == tensor_shapes._curve_plan(after)
    geometry = stale_solver('numpy', growing_box(200.), cell).freeze_geometry()
    replay = stale_solver(backend, growing_box(220.), cell).solve(geometry=geometry,
                                                                  auto_detune=False)
    fresh = stale_solver(backend, growing_box(220.), cell).solve(auto_detune=False)
    for a, b in ((replay.R, fresh.R), (replay.T, fresh.T)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


ELLIPSE_CELL = np.diag([700., 700.])


def _ellipse_images(lx, ly=200.):
    """The image list of the growing ellipse's arc."""
    lattice = as_lattice(lattice=ELLIPSE_CELL)
    return tensor_shapes._curve_plan(parametric.plan_gamma(growing_ellipse(lx, ly), lattice))[1][0][1]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_an_ellipse_growing_out_of_its_wigner_seitz_cell_is_refused_by_name(backend):
    # 709 x 200 at 10 deg stays inside the Wigner-Seitz cell of a 700 square, 710 leaves it:
    # its arc grows from one instance, searched over the 3x3, to a ring of 11 that the 3x3
    # does not hold, which a replay must not search short. The polygon interface plan is the
    # same at both sizes, so only the curve check can refuse.
    if backend == 'torch':
        pytest.importorskip('torch')
    lattice = as_lattice(lattice=ELLIPSE_CELL)
    assert (boundary._plan_key(boundary.plan_layer(growing_ellipse(709.), lattice))
            == boundary._plan_key(boundary.plan_layer(growing_ellipse(710.), lattice)))
    assert (len(_ellipse_images(709.)), len(_ellipse_images(710.))) == (1, 11)
    geometry = stale_solver('numpy', growing_ellipse(709.), ELLIPSE_CELL).freeze_geometry()
    with pytest.raises(_legacy_wavebit.StaleGeometryError) as refusal:
        stale_solver(backend, growing_ellipse(710.), ELLIPSE_CELL).solve(geometry=geometry,
                                                                         auto_detune=False)
    message = str(refusal.value)
    assert ("shape 0 (ellipse)'s image list changed (1 -> 11 instances: its support crossed "
            "the edge of its Wigner-Seitz cell)") in message
    assert 'layer 0' in message and 'cancel an interface' not in message
    stale_solver(backend, growing_ellipse(710.), ELLIPSE_CELL).solve(auto_detune=False)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_an_ellipse_shrinking_into_its_wigner_seitz_cell_replays_as_a_fresh_solve(backend):
    # The other way: the recorded ring of 11 holds the 3x3 a fresh plan searches, so the
    # replay finds the same nearest boundary by the same arithmetic per instance.
    if backend == 'torch':
        pytest.importorskip('torch')
    geometry = stale_solver('numpy', growing_ellipse(710.), ELLIPSE_CELL).freeze_geometry()
    replay = stale_solver(backend, growing_ellipse(709.), ELLIPSE_CELL).solve(
        geometry=geometry, auto_detune=False)
    fresh = stale_solver(backend, growing_ellipse(709.), ELLIPSE_CELL).solve(auto_detune=False)
    for a, b in ((replay.R, fresh.R), (replay.T, fresh.T)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_an_ellipse_leaving_its_cell_into_a_ring_the_3x3_holds_replays_as_a_fresh_solve(backend):
    # 60 wide instead of 200, the ellipse leaves its Wigner-Seitz cell at 711 rather than 710,
    # and its ring is the 3x3 itself, which the single-image record already searches: the
    # refusal reads the instances searched, not the length of the image list.
    if backend == 'torch':
        pytest.importorskip('torch')
    lattice = as_lattice(lattice=ELLIPSE_CELL)
    assert (boundary._plan_key(boundary.plan_layer(growing_ellipse(709., 60.), lattice))
            == boundary._plan_key(boundary.plan_layer(growing_ellipse(711., 60.), lattice)))
    assert _ellipse_images(709., 60.) == ((0, 0),)
    assert _ellipse_images(711., 60.) == tensor_shapes._NEIGHBOURS
    geometry = stale_solver('numpy', growing_ellipse(709., 60.), ELLIPSE_CELL).freeze_geometry()
    replay = stale_solver(backend, growing_ellipse(711., 60.), ELLIPSE_CELL).solve(
        geometry=geometry, auto_detune=False)
    fresh = stale_solver(backend, growing_ellipse(711., 60.), ELLIPSE_CELL).solve(auto_detune=False)
    for a, b in ((replay.R, fresh.R), (replay.T, fresh.T)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _value_and_gradient(differentiate, shapes_of, lx, cell, geometry=None):
    """``sum T`` and its derivative in ``lx``, taken by ``jax.grad`` or torch autograd."""
    if differentiate == 'torch':
        torch = pytest.importorskip('torch')
        width = torch.tensor(lx, dtype=torch.float64, requires_grad=True)
        value = torch.real(stale_solver('torch', shapes_of(width), cell).solve(
            geometry=geometry, auto_detune=False).T.sum())
        gradient, = torch.autograd.grad(value, width)
        return float(value.detach()), float(gradient)
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    _legacy_wavebit.enable_x64()
    value, gradient = jax.value_and_grad(lambda width: jnp.real(stale_solver(
        'jax', shapes_of(width), cell).solve(geometry=geometry, auto_detune=False).T.sum()))(
        jnp.asarray(lx))
    return float(value), float(gradient)


@pytest.mark.parametrize('differentiate', ['jax.grad', 'torch'])
def test_the_curve_check_reads_shape_values_under_differentiation(differentiate):
    # jax.grad and torch autograd carry concrete values, so the check runs there as eagerly
    # as on floats: the growing ellipse is refused, and the box crossing its ring and the
    # shrinking ellipse replay with a fresh plan's value and derivative. Bit for bit, but
    # for one: JAX compiles the blend's evaluation (a scan, checkpointed under jax.grad), the
    # recorded ring of 11 compiles to a different program from a fresh 3x3, and under
    # jax.grad the two round the blend differently -- 5.6e-16 in the value, with identical
    # inputs to the blend and candidates that agree bitwise on their own.
    square = np.diag([600., 600.])
    geometry = stale_solver('numpy', growing_box(200.), square).freeze_geometry()
    assert (_value_and_gradient(differentiate, growing_box, 220., square, geometry)
            == _value_and_gradient(differentiate, growing_box, 220., square))
    geometry = stale_solver('numpy', growing_ellipse(710.), ELLIPSE_CELL).freeze_geometry()
    replay = _value_and_gradient(differentiate, growing_ellipse, 709., ELLIPSE_CELL, geometry)
    fresh = _value_and_gradient(differentiate, growing_ellipse, 709., ELLIPSE_CELL)
    if differentiate == 'torch':
        assert replay == fresh
    else:
        np.testing.assert_allclose(replay, fresh, rtol=1e-13, atol=0.)
    geometry = stale_solver('numpy', growing_ellipse(709.), ELLIPSE_CELL).freeze_geometry()
    with pytest.raises(_legacy_wavebit.StaleGeometryError, match=r"\(1 -> 11 instances"):
        _value_and_gradient(differentiate, growing_ellipse, 710., ELLIPSE_CELL, geometry)
