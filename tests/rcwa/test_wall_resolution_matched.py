"""``Numerics(wall_resolution=G)`` on one circle or axis-aligned ellipse: the matched coordinates.

The second stage of the stretched solve. The coordinates crowd at the lines of a mesh that carries a
rectangle onto the ellipse (Weiss et al. 2009), so the structure is a full in-plane tensor in the
computational cell. What is pinned here holds whatever the accuracy: the map and its inverse, the
plane waves written in its basis, a uniform layer that the map must not change (the closest thing
to the rectangles' slope-one limit, which a map that bends lines has not), moving the shape,
energy, the backends, and every refusal by its code. The accuracy against finite-element fields is
in ``tests/references/test_gyptis_matched_field_golden.py``.
"""
import math

import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError
from wavebit.rcwa._engine.stretch import Matched

P, H = 400.0, 150.0
METAL = -20 + 1.5j
pytestmark = pytest.mark.filterwarnings('ignore::wavebit.rcwa.diagnostics.ExperimentalWarning')


def _layer(geometry, material, thickness=H, background=None):
    return rcwa.Layer(thickness=thickness, regions=[rcwa.Region(geometry=geometry, material=Material(epsilon=material))],
                      **({} if background is None else {'background': Material(epsilon=background)}))


def _problem(material=12.25, geometry=None, wavelength=900.0, theta=0.0, phi=0.0, polarization='p', side='top',
             layers=None, bottom=None, top=None, period=(P, P)):
    if layers is None:
        layers = [_layer(rcwa.Circle(radius=100.0) if geometry is None else geometry, material)]
    media = {name: Material(epsilon=value) for name, value in (('bottom', bottom), ('top', top)) if value is not None}
    return rcwa.Problem(
        units=Units(length='nm'), stack=rcwa.Stack(period=period, layers=layers, **media),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=[rcwa.PlaneWave(
            side=side, polarization=polarization, theta=theta, phi=phi)]),
        outputs=rcwa.Outputs(fields=True))


def _numerics(orders=(4, 4), slope=0.02, **more):
    return rcwa.Numerics(orders=orders, factorization='li', **({} if slope is None else {'wall_resolution': slope}),
                         **more)


def _run(problem, **numerics):
    return rcwa.run(problem, numerics=_numerics(**numerics))


def _totals(result):
    return float(np.sum(np.asarray(result.R))), float(np.sum(np.asarray(result.T)))


# --------------------------------------------------------------------------- the map


@pytest.mark.parametrize('centre, radii, periods', [
    ((200.0, 200.0), (100.0, 100.0), (400.0, 400.0)),
    ((150.0, 260.0), (130.0, 80.0), (400.0, 400.0)),
    ((380.0, 20.0), (150.0, 50.0), (400.0, 400.0)),           # across the cell's corner
    ((300.0, 200.0), (200.0, 120.0), (500.0, 300.0)),          # a rectangular cell
])
def test_the_map_carries_the_rectangle_onto_the_ellipse_and_is_inverted(centre, radii, periods):
    m = Matched(centre, radii, periods, 0.02)
    (cx, cy), (rx, ry) = centre, radii
    # The four lines sit half a period apart in the computational cell, at the slope.
    assert np.allclose(np.diff(m.x.wall_u), periods[0] / 2) and np.allclose(np.diff(m.y.wall_u), periods[1] / 2)
    # The ellipse's own pieces are centred on the inverse of its centre, a quarter period either side.
    mid_u, mid_v = float(m.x.inverse(cx)), float(m.y.inverse(cy))
    assert np.allclose(np.sort((np.array([mid_u - periods[0] / 4, mid_u + periods[0] / 4])) % periods[0]),
                       m.x.wall_u, atol=1e-8)
    for u in (mid_u - periods[0] / 4, mid_u + periods[0] / 4):   # a line u = const inside is an arc
        v = np.linspace(mid_v - periods[1] / 4, mid_v + periods[1] / 4, 41)
        x, y, *_ = m.at(np.full_like(v, u), v)
        wrap = lambda t, p: t - p * np.floor(t / p + 0.5)
        assert np.abs(np.hypot(wrap(x - cx, periods[0]) / rx, wrap(y - cy, periods[1]) / ry) - 1).max() < 1e-12
    u = np.linspace(-periods[0], 2 * periods[0], 301)
    v = np.linspace(-periods[1], 2 * periods[1], 307)
    U, V = np.meshgrid(u, v)
    x, y, xu, xv, yu, yv, inside = m.at(U, V)
    x2, y2, *_ = m.at(U + periods[0], V - periods[1])
    assert np.abs((x2 - x) - periods[0]).max() < 1e-9 and np.abs((y2 - y) + periods[1]).max() < 1e-9   # periodic
    assert (xu * yv - xv * yu).min() > 0                         # it does not fold
    back_u, back_v = m.inverse(x, y)
    again_x, again_y, *_ = m.at(back_u, back_v)
    assert np.abs(again_x - x).max() < 1e-8 and np.abs(again_y - y).max() < 1e-8
    # The analytic Jacobian, where the map is smooth (away from the four lines).
    step = 1e-5
    smooth = np.ones_like(U, bool)
    for axis, W, lines, period in ((m.x, U, m.x.wall_u, periods[0]), (m.y, V, m.y.wall_u, periods[1])):
        for line in lines:
            smooth &= np.abs((W - line + period / 2) % period - period / 2) > 1e-3 * period
    xa, ya, *_ = m.at(U + step, V)
    xb, yb, *_ = m.at(U - step, V)
    assert np.abs((xa - xb) / (2 * step) - xu)[smooth].max() < 1e-5
    assert np.abs((ya - yb) / (2 * step) - yu)[smooth].max() < 1e-5
    # The ellipse's pixels: the rectangle, half a period by half a period of the computational cell.
    assert inside.mean() == pytest.approx(0.25, abs=0.01)


def test_a_circle_is_the_ellipse_of_equal_radii_and_a_corner_patch_is_the_stretch_alone():
    circle = _run(_problem(METAL, rcwa.Circle(radius=90.0)), orders=(3, 3))
    ellipse = _run(_problem(METAL, rcwa.Ellipse(radii=(90.0, 90.0))), orders=(3, 3))
    assert np.array_equal(np.asarray(circle.R), np.asarray(ellipse.R))
    m = Matched((200.0, 200.0), (90.0, 90.0), (P, P), 0.02)
    u = np.linspace(0.0, m.x.wall_u[0], 7)
    x, y, *_ = m.at(u, u)
    assert np.abs(x - m.x.F(u)).max() < 1e-11 and np.abs(y - m.y.F(u)).max() < 1e-11


@pytest.mark.parametrize('radii, periods', [((200.0, 100.0), (P, P)), ((100.0, 210.0), (P, P)), ((0.0, 50.0), (P, P))])
def test_an_ellipse_that_leaves_no_room_to_the_cell_edge_is_refused(radii, periods):
    with pytest.raises(ValueError, match='does not fit inside a cell'):
        Matched((200.0, 200.0), radii, periods, 0.02)


def test_the_raster_puts_the_matched_lines_on_pixel_edges():
    m = Matched((170.0, 230.0), (120.0, 70.0), (P, P), 0.02)
    edges_u, edges_v, unit, inside = m.raster(4, 4)
    for edges, axis in ((edges_u, m.x), (edges_v, m.y)):
        assert edges[-1] == P and np.all(np.diff(edges) > 0)
        for line in axis.wall_u:
            assert np.abs(edges - line).min() < 1e-9 * P
    det = unit[..., 2, 2]
    assert det.min() > 0
    # det(J) (J^T J)^-1 has determinant one in plane.
    product = unit[..., 0, 0] * unit[..., 1, 1]
    plane = product - unit[..., 0, 1] * unit[..., 1, 0]
    assert np.all(np.abs(plane - 1) < 1e-14 * product + 1e-14)       # rounding of two large products


def test_a_plane_wave_is_written_in_the_matched_basis():
    """``wave_columns`` against a brute midpoint sum of ``x_u exp(i k.r)`` and its three siblings."""
    m = Matched((180.0, 215.0), (120.0, 70.0), (P, P), 0.02)
    half, k_u, k_v = 3, 0.004, -0.002
    orders = [(0, 0), (1, -1), (-2, 0)]
    columns = m.wave_columns(np, k_u, k_v, half, half, orders, np.complex128)
    count = 1600
    u = (np.arange(count) + 0.5) / count * P
    U, V = np.meshgrid(u, u)
    x, y, xu, xv, yu, yv, _ = m.at(U, V)
    harmonics = np.arange(-half, half + 1)
    rows = np.exp(-2j * np.pi / P * np.outer(harmonics, u))
    for j, (mm, nn) in enumerate(orders):
        wave = np.exp(1j * ((k_u + 2 * np.pi * mm / P) * x + (k_v + 2 * np.pi * nn / P) * y - k_u * U - k_v * V))
        for column, slope in zip(columns, (xu, yu, xv, yv)):
            brute = (rows @ (slope * wave) @ rows.T / count ** 2).reshape(-1)
            assert np.abs(brute - column[:, j]).max() < 2e-4          # the midpoint sum's own error


# --------------------------------------------------------------------------- limits and invariants


@pytest.mark.parametrize('shape', [rcwa.Circle(radius=100.0), rcwa.Ellipse(radii=(150.0, 60.0), center=(20.0, -30.0))])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_uniform_stack_in_matched_coordinates_is_the_unstretched_solve(shape, side):
    """The shape is of the background's own material: the physics is a stack of uniform layers.

    The map bends every line of the cell, and the half-spaces and uniform layers go through an
    eigenproblem of full tensors, yet at order 6 R, T and the specular amplitudes are the plane-wave
    answer to 6e-9 (circle) and 6e-7 (the 150 x 60 ellipse), lit obliquely from either side, on glass.
    The electric field is the plane-wave field to 0.15% and 0.41% RMS over three layers, 0.29% and
    1.3% at the worst sample. Across a matched line the component of ``E~`` normal to it jumps even in
    a uniform medium; the reading takes it from ``D~`` there, and what is left is that window's.
    """
    layers = [rcwa.Layer(thickness=60.0, background=Material(epsilon=1.7)),
              _layer(shape, 4.0, background=4.0),
              rcwa.Layer(thickness=40.0, background=Material(epsilon=2.25))]
    problem = _problem(layers=layers, bottom=2.25, theta=0.3, phi=0.4, polarization='s', side=side)
    x, y, z = np.linspace(-190, 190, 9), np.linspace(-150, 150, 5), [30.0, 130.0, 240.0]
    matched = _run(problem, orders=(6, 6))
    plain = rcwa.run(problem, numerics=rcwa.Numerics(orders=(6, 6)))
    assert np.abs(np.asarray(matched.R) - np.asarray(plain.R)).max() < 2e-6
    assert np.abs(np.asarray(matched.T) - np.asarray(plain.T)).max() < 2e-6
    centre = (slice(None), 6, 6)
    for port in ('reflection', 'transmission'):
        a, b = (np.asarray(getattr(r, port).amplitude_ratio_s)[centre] for r in (matched, plain))
        assert np.abs(a - b).max() < 2e-6
    fields = [np.asarray(r.field(x=x, y=y, z=z).values) for r in (matched, plain)]
    for part in (slice(0, 3), slice(3, 6)):                     # E, then H, each on its own scale
        got, want = fields[0][..., part], fields[1][..., part]
        difference = np.abs(got - want)
        assert np.sqrt(np.mean(difference ** 2) / np.mean(np.abs(want) ** 2)) < 1e-2
        assert difference.max() < 3e-2 * np.abs(want).max()


@pytest.mark.parametrize('material', [12.25, METAL])
@pytest.mark.parametrize('shape, shift', [
    (rcwa.Circle(radius=100.0), (37.5, -25.0)),
    (rcwa.Circle(radius=100.0), (150.0, 0.0)),                   # across the cell's edge
    (rcwa.Ellipse(radii=(130.0, 80.0)), (-60.0, 110.0)),
])
def test_moving_the_shape_in_the_cell_moves_nothing_else(material, shape, shift):
    x, y = np.linspace(-160, 160, 7), np.linspace(-170, 170, 5)
    kind = type(shape)
    moved_shape = (rcwa.Circle(radius=shape.radius, center=shift) if kind is rcwa.Circle
                   else rcwa.Ellipse(radii=shape.radii, center=shift))
    at_centre = _run(_problem(material, shape), orders=(4, 4))
    moved = _run(_problem(material, moved_shape), orders=(4, 4))
    assert _totals(moved) == pytest.approx(_totals(at_centre), abs=1e-9)
    a = np.asarray(at_centre.field(x=x, y=y, z=[75.0]).values)
    b = np.asarray(moved.field(x=x + shift[0], y=y + shift[1], z=[75.0]).values)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()


@pytest.mark.parametrize('wavelength, theta, phi, polarization, orders', [
    (900.0, 0.0, 0.0, 'p', 1), (900.0, np.deg2rad(20), np.deg2rad(30), 's', 1),
    (350.0, 0.0, 0.0, 'p', 5), (350.0, np.deg2rad(20), 0.0, 'p', 4)])
def test_a_lossless_layer_conserves_energy_with_any_number_of_propagating_orders(
        wavelength, theta, phi, polarization, orders):
    result = _run(_problem(12.25, rcwa.Ellipse(radii=(130.0, 80.0)), wavelength=wavelength, theta=theta, phi=phi,
                           polarization=polarization), orders=(6, 6))
    R, T = _totals(result)
    assert abs(R + T - 1) < 1e-4
    assert int(np.sum(np.asarray(result.R) > 1e-12)) == orders


def test_a_rectangular_cell_and_two_layers_of_the_same_circle():
    """One map serves the stack: the same circle in two layers, a spacer between, a 500 x 300 cell."""
    circle = rcwa.Circle(radius=110.0, center=(30.0, 0.0))
    layers = [_layer(circle, METAL, thickness=60.0), rcwa.Layer(thickness=30.0),
              _layer(circle, 12.25, thickness=90.0, background=2.25)]
    result = _run(_problem(layers=layers, period=(500.0, 300.0), theta=0.2), orders=(5, 4))
    R, T = _totals(result)
    assert 0 < R < 1 and 0 < T < 1 and R + T < 1
    field = np.asarray(result.field(x=[0.0, 100.0], y=[0.0], z=[30.0, 75.0, 130.0]).values)
    assert np.all(np.isfinite(field))


# --------------------------------------------------------------------------- backends


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('material, wavelength', [(12.25, 900.0), (METAL, 350.0)])
def test_the_backends_agree(backend, material, wavelength):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    problem = _problem(material, rcwa.Ellipse(radii=(130.0, 80.0), center=(20.0, -10.0)), wavelength=wavelength,
                       theta=np.deg2rad(20), phi=np.deg2rad(30))
    x, y = np.linspace(-190, 190, 7), np.linspace(-170, 170, 5)
    results = [rcwa.run(problem, numerics=_numerics(orders=(4, 4)), backend=name) for name in ('numpy', backend)]
    R, T, field = ([np.asarray(getter(r)) for r in results] for getter in (
        lambda r: r.R, lambda r: r.T, lambda r: r.field(x=x, y=y, z=[75.0]).values))
    assert np.abs(R[0] - R[1]).max() < 1e-9 and np.abs(T[0] - T[1]).max() < 1e-9
    assert np.abs(field[0] - field[1]).max() < 1e-8 * np.abs(field[0]).max()


# --------------------------------------------------------------------------- the refusals


def _refused(problem, numerics):
    report = rcwa.check(problem, numerics=numerics)
    assert len(report.issues) == 1
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(problem, numerics=numerics)
    assert caught.value.code == report.issues[0].code
    return report.issues[0].code, str(caught.value)


def _two(first, second, same_layer=False):
    if same_layer:
        return [rcwa.Layer(thickness=H, regions=[rcwa.Region(geometry=g, material=Material(epsilon=4.0))
                                                 for g in (first, second)])]
    return [_layer(first, 4.0), _layer(second, 4.0)]


@pytest.mark.parametrize('layers, numerics, said', [
    (lambda: _two(rcwa.Circle(radius=60.0, center=(-100.0, 0.0)), rcwa.Circle(radius=60.0, center=(100.0, 0.0)),
                  same_layer=True), lambda: _numerics(), 'layer 0 holds circle, circle'),
    (lambda: _two(rcwa.Circle(radius=100.0), rcwa.Rectangle(size=(100.0, 100.0))), lambda: _numerics(),
     'layer 1 holds rectangle'),
    (lambda: _two(rcwa.Circle(radius=100.0), rcwa.Circle(radius=90.0)), lambda: _numerics(),
     'layer 1 holds another circle'),
    (lambda: [_layer(rcwa.Ellipse(radii=(130.0, 80.0), angle=0.3), 4.0)], lambda: _numerics(),
     'is turned'),
    (lambda: [_layer(rcwa.Ellipse(radii=(199.9, 80.0)), 4.0)], lambda: _numerics(orders=(1, 1)),
     None),
    (lambda: [_layer(rcwa.Circle(radius=100.0), 4.0)], lambda: _numerics(orders=(4, 0)), 'single harmonic'),
    (lambda: [_layer(rcwa.Circle(radius=100.0), 4.0)], lambda: rcwa.Numerics(orders=(2, 2), wall_resolution=0.02),
     "factorization is 'nvf'"),
    (lambda: [_layer(rcwa.Circle(radius=100.0), 4.0)], lambda: _numerics(algorithm='etm'),
     "layer algorithm is 'ETM'"),
])
def test_what_is_not_implemented_is_refused_by_name(layers, numerics, said):
    problem = _problem(layers=layers())
    if said is None:                              # it fits, with room: solved, not refused
        _run(problem, orders=(1, 1))
        return
    code, message = _refused(problem, numerics())
    assert code == 'rcwa.unsupported' and said in message


def test_an_ellipse_reaching_the_cell_edge_is_refused():
    problem = _problem(layers=[_layer(rcwa.Ellipse(radii=(200.0, 80.0)), 4.0)])
    code, message = _refused(problem, _numerics())
    assert code == 'rcwa.unsupported' and 'does not fit inside a cell' in message


def test_a_matched_solve_warns_that_it_is_experimental():
    from wavebit.rcwa.diagnostics import ExperimentalWarning
    with pytest.warns(ExperimentalWarning, match='wall_resolution is experimental') as caught:
        _run(_problem(), orders=(2, 2))
    assert len(caught) == 1 and caught[0].filename == __file__


def test_the_result_refuses_what_a_matched_solve_does_not_hold():
    result = _run(_problem(METAL), orders=(3, 3))
    with pytest.raises(UnsupportedFeatureError) as caught:
        result.boundary_fields(side='top', x=[0.0], y=[0.0])
    assert caught.value.code == 'rcwa.unsupported'
    for rule in ('lalanne', 'split', 'splitw'):
        with pytest.raises(UnsupportedFeatureError) as caught:
            result.field(x=[0.0], y=[0.0], z=[75.0], reconstruction=rule)
        assert caught.value.code == 'field.reconstruction'


def test_the_check_reports_the_option_and_no_issue():
    report = rcwa.check(_problem(METAL, rcwa.Ellipse(radii=(130.0, 80.0))), numerics=_numerics())
    assert not report.issues and report.resolved['wall_resolution'] == 0.02
    assert math.isclose(report.resolved['wall_resolution'], 0.02)


@pytest.mark.slow
def test_the_published_gold_cylinder():
    """Kuechenmeister, Zebrowski and Busch, Opt. Express 20, 17319 (2012), section 9.2: the matched mesh's R
    is flat at about 0.823 from roughly 500 plane waves on (read off their figure; no table is printed).
    Gold of their Drude model at 829 nm, radius 150 nm, height 50 nm, period 700 nm, in air, normal incidence."""
    omega = 2 * np.pi * 299792458.0 / 829e-9
    gold = 9.0685 - 1.3544e16 ** 2 / (omega * (omega + 1j * 1.1536e14))
    problem = _problem(gold, rcwa.Circle(radius=150.0), wavelength=829.0, period=(700.0, 700.0),
                       layers=[_layer(rcwa.Circle(radius=150.0), gold, thickness=50.0)])
    R = _totals(_run(problem, orders=(12, 12)))[0]
    assert R == pytest.approx(0.82306, abs=2e-5)                  # measured; 0.82344 at order 14
    assert R == pytest.approx(0.823, abs=1e-3)
