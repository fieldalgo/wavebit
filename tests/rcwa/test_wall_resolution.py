"""``Numerics(wall_resolution=G)``: the solve in coordinates that crowd their lines at rectangle walls.

What is pinned here is what the construction has to satisfy whatever the accuracy turns out to be:
the coordinate function and its tables, the limit in which the coordinates are the cell's own, the
freedom to move the structure in the cell, energy conservation, the three backends, the entries an
evanescent order answers with, and every refusal by its code. The accuracy against finite-element
fields is in ``tests/references/test_gyptis_rectangles_field_golden.py``.
"""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.rcwa._engine import rectilinear
from wavebit.rcwa._engine.stretch import Axis, Stretch, plane_wave_matrix

P, H = 400.0, 150.0
METAL = -20 + 1.5j


def _problem(material=12.25, size=(200.0, 200.0), center=None, wavelength=900.0, theta=0.0, phi=0.0,
             polarization='p', side='top', layers=None, bottom=None, top=None, fields=True, **outputs):
    if layers is None:
        geometry = rcwa.Rectangle(size=size, **({} if center is None else {'center': center}))
        layers = [rcwa.Layer(thickness=H, regions=[rcwa.Region(geometry=geometry,
                                                               material=Material(epsilon=material))])]
    media = {name: Material(epsilon=value) for name, value in (('bottom', bottom), ('top', top))
             if value is not None}
    return rcwa.Problem(
        units=Units(length='nm'), stack=rcwa.Stack(period=(P, P), layers=layers, **media),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=[rcwa.PlaneWave(
            side=side, polarization=polarization, theta=theta, phi=phi)]),
        outputs=rcwa.Outputs(fields=fields, **outputs))


def _numerics(orders=(4, 4), slope=0.02, **more):
    return rcwa.Numerics(orders=orders, factorization='li',
                         **({} if slope is None else {'wall_resolution': slope}), **more)


def _run(problem, **numerics):
    return rcwa.run(problem, numerics=_numerics(**numerics))


def _totals(result):
    return float(np.sum(np.asarray(result.R))), float(np.sum(np.asarray(result.T)))


# --------------------------------------------------------------------------- the function


@pytest.mark.parametrize('walls', [[100.0, 300.0], [75.0, 325.0], [50.0, 170.0, 290.0, 320.0],
                                   [0.0, 250.0], [137.5, 387.5]])
def test_the_coordinate_function_is_what_its_docstring_says(walls):
    axis = Axis(walls, P, 0.02)
    u = np.linspace(-P, 2 * P, 6001)
    F, f = axis.F(u), axis.dF(u)
    assert axis.F(0.0) == 0.0 and axis.F(P) == pytest.approx(P, abs=1e-12)
    assert np.allclose(axis.F(u + P), F + P, atol=1e-10)                    # periodic
    assert f.min() >= 0.02 - 1e-12                                          # slowest at the walls
    assert np.allclose(axis.dF(axis.wall_u), 0.02, atol=1e-12)
    assert np.allclose(axis.F(axis.wall_u), np.sort(walls), atol=1e-10)     # a wall is at its place
    assert np.allclose(np.diff(axis.wall_u), P / len(walls))                # equidistant in u
    assert np.allclose(axis.inverse(F), u, atol=1e-8)
    assert np.allclose(np.gradient(F, u)[3:-3], f[3:-3], atol=2e-4)


def test_an_axis_without_walls_is_the_identity_and_two_walls_too_close_are_refused():
    axis = Axis([], P, 0.02)
    u = np.linspace(-10.0, 500.0, 7)
    assert axis.identity and np.array_equal(axis.F(u), u) and np.array_equal(axis.inverse(u), u)
    assert np.array_equal(axis.dF(u), np.ones_like(u))
    # Pieces are P / K wide whatever their physical width; one of 1 nm at slope 0.02 cannot rise.
    with pytest.raises(ValueError, match='closer than the stretch can hold'):
        Axis([100.0, 101.0], P, 0.02)
    for slope in (0.0, -0.1, 1.5, float('nan')):
        with pytest.raises(ValueError):
            Axis([100.0, 300.0], P, slope)


def test_the_weighted_tables_are_the_integrals_they_stand_for():
    axis = Axis([50.0, 170.0, 290.0, 320.0], P, 0.02)
    half = 5
    edges = np.array([50.0, 170.0, 200.0, 290.0, 320.0, 400.0]) / P
    table = axis.segment_table(edges, half)
    count = 400_000
    u = (np.arange(count) + 0.5) / count * P
    x, f = axis.F(u), axis.dF(u)
    orders = np.arange(-2 * half, 2 * half + 1)
    left = np.concatenate([[0.0], edges[:-1]]) * P
    for j in range(len(edges)):
        inside = (x >= left[j]) & (x < edges[j] * P)
        brute = (f[inside] * np.exp(-2j * np.pi * orders[:, None] * u[inside] / P)).sum(axis=1) / count
        assert np.abs(brute - table[:, j]).max() < 5e-6                     # the midpoint sum's own error
    assert np.abs(table.sum(axis=1) - axis.period_table(half, 'slope')).max() < 1e-13
    for name, weight in (('slope', f), ('reciprocal', 1.0 / f), ('one', np.ones(count))):
        brute = (weight * np.exp(-2j * np.pi * orders[:, None] * u / P)).sum(axis=1) / count
        assert np.abs(brute - axis.period_table(half, name)).max() < 1e-9
    assert axis.period_table(half, 'slope')[2 * half] == pytest.approx(1.0, abs=1e-13)   # F rises by P


def test_the_identity_axis_gives_the_closed_form_segment_coefficients():
    table = Axis([], P, 0.02).segment_table(np.array([0.25, 0.75, 1.0]), 4)
    k = np.arange(-8, 9)
    closed = np.where(k == 0, 0.5, (np.exp(-2j * np.pi * k * 0.25) - np.exp(-2j * np.pi * k * 0.75))
                      / (2j * np.pi * np.where(k == 0, 1, k)))
    assert np.abs(table[:, 1] - closed).max() < 1e-14


def test_a_plane_wave_is_written_in_the_computational_basis():
    axis = Axis([50.0, 170.0, 290.0, 320.0], P, 0.02)
    half, k_incident = 4, 0.003
    count = 400_000
    u = (np.arange(count) + 0.5) / count * P
    x, f = axis.F(u), axis.dF(u)
    orders = np.arange(-half, half + 1)
    nodes = axis.wave_nodes(half)
    for weighted in (False, True):
        matrix = plane_wave_matrix(np, nodes, P, k_incident, half, weighted, np.complex128)
        weight = f if weighted else 1.0
        for q in (-2, 0, 3):
            wave = weight * np.exp(1j * (k_incident + 2 * np.pi * q / P) * x)
            brute = (wave * np.exp(-1j * (k_incident + 2 * np.pi * orders[:, None] / P) * u)).sum(axis=1) / count
            assert np.abs(brute - matrix[:, q + half]).max() < 1e-9
    identity = plane_wave_matrix(np, Axis([], P, 0.02).wave_nodes(half), P, k_incident, half, True, np.complex128)
    assert np.abs(identity - np.eye(2 * half + 1)).max() < 1e-13


def test_an_edge_that_is_no_wall_is_not_stretched():
    """A rectangle as long as the period has the same material either side of its two edges."""
    lattice = [[P, 0.0], [0.0, P]]
    layers = [{'shapes': [{'type': 'rectangle', 'lx': 200.0, 'ly': P, 'cx': 0.0, 'cy': 0.0, 'material': 4.0}],
               'background': 1.0}]
    walls = rectilinear.wall_positions(layers, lattice)
    assert walls == ([100.0, 300.0], [])
    stretch = Stretch(*walls, P, P, 0.02)
    assert not stretch.x.identity and stretch.y.identity


def _rectangle(x0, x1, y0, y1, material=METAL):
    return {'type': 'rectangle', 'lx': x1 - x0, 'ly': y1 - y0, 'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
            'material': material}


@pytest.mark.parametrize('shapes, walls', [
    # a block, and the same block written as two rectangles that touch at x = 0
    ([_rectangle(-100, 100, -100, 100)], ([100.0, 300.0], [100.0, 300.0])),
    ([_rectangle(-100, 0, -100, 100), _rectangle(0, 100, -100, 100)], ([100.0, 300.0], [100.0, 300.0])),
    # the block cut by the cell's corner into four, the cut edges on the cell's edges
    ([_rectangle(46, 200, 30, 200), _rectangle(-200, -154, 30, 200), _rectangle(46, 200, -200, -170),
      _rectangle(-200, -154, -200, -170)], ([246.0, 46.0], [230.0, 30.0])),
    # touching rectangles of two materials keep the wall where they touch
    ([_rectangle(-100, 0, -100, 100), _rectangle(0, 100, -100, 100, 12.25)],
     ([100.0, 200.0, 300.0], [100.0, 300.0])),
    # a rectangle inside another: of the same material it is no wall, of another it is
    ([_rectangle(-100, 100, -100, 100), _rectangle(-50, 50, -50, 50)], ([100.0, 300.0], [100.0, 300.0])),
    ([_rectangle(-100, 100, -100, 100), _rectangle(-50, 50, -50, 50, 12.25)],
     ([100.0, 150.0, 250.0, 300.0], [100.0, 150.0, 250.0, 300.0])),
    # a T: where its two rectangles touch, metal meets air above and below the narrow one
    ([_rectangle(-120, 0, -100, 100), _rectangle(0, 80, -50, 50)],
     ([80.0, 200.0, 280.0], [100.0, 150.0, 250.0, 300.0])),
])
def test_a_wall_is_an_edge_with_two_materials_beside_it(shapes, walls):
    """Two rectangles of one material that touch make no wall where they touch.

    Before 2026-11-28 every rectangle edge was a wall, so a block written as two touching
    rectangles was solved on other coordinates than the block (``R`` -9.9e-4, the field 4.9% at
    order 12), and a block across the cell's edge, which has to be written as the rectangles the
    edges cut it into, got walls on the cell's edges.
    """
    found = rectilinear.wall_positions([{'shapes': shapes, 'background': 1.0}], [[P, 0.0], [0.0, P]])
    assert tuple(sorted(w) for w in found) == tuple(sorted(w) for w in walls)


# --------------------------------------------------------------------------- limits and invariants


@pytest.mark.parametrize('material', [12.25, METAL])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_at_slope_one_a_half_period_block_is_the_unstretched_solve(material, side):
    """Walls already at their equidistant places and slope 1: the coordinates are the cell's own.

    Four layers on glass, two of them uniform, lit obliquely from either side: R, T, the amplitudes
    and all six field components in every layer agree with the solve that never stretched, though the
    half-spaces and the uniform layers went through an eigenproblem instead of plane waves.
    """
    block = rcwa.Rectangle(size=(200.0, 200.0))
    layers = [rcwa.Layer(thickness=60.0, background=Material(epsilon=1.7)),
              rcwa.Layer(thickness=H, regions=[rcwa.Region(geometry=block, material=Material(epsilon=material))]),
              rcwa.Layer(thickness=80.0, regions=[rcwa.Region(geometry=block, material=Material(epsilon=4.0))],
                         background=Material(epsilon=2.0)),
              rcwa.Layer(thickness=40.0, background=Material(epsilon=2.25))]
    problem = _problem(layers=layers, bottom=2.25, theta=0.3, phi=0.4, polarization='s', side=side)
    x, y, z = np.linspace(-190, 190, 9), np.linspace(-150, 150, 5), [30.0, 130.0, 250.0, 310.0]
    stretched, plain = _run(problem, slope=1.0), _run(problem, slope=None)
    assert np.abs(np.asarray(stretched.R) - np.asarray(plain.R)).max() < 1e-11
    assert np.abs(np.asarray(stretched.T) - np.asarray(plain.T)).max() < 1e-11
    fields = [np.asarray(r.field(x=x, y=y, z=z).values) for r in (stretched, plain)]
    assert np.abs(fields[0] - fields[1]).max() < 1e-10 * np.abs(fields[1]).max()
    # The specular amplitudes are those of the same plane waves.
    centre = (slice(None), 4, 4)
    for port in ('reflection', 'transmission'):
        a, b = (np.asarray(getattr(r, port).amplitude_ratio_s)[centre] for r in (stretched, plain))
        assert np.abs(a - b).max() < 1e-11


def test_a_stack_without_walls_is_the_unstretched_solve():
    layers = [rcwa.Layer(thickness=80.0, background=Material(epsilon=2.25)),
              rcwa.Layer(thickness=40.0, background=Material(epsilon=4.0))]
    problem = _problem(layers=layers, theta=0.2, phi=0.1)
    assert _totals(_run(problem)) == pytest.approx(_totals(_run(problem, slope=None)), abs=1e-12)


@pytest.mark.parametrize('material', [12.25, METAL])
def test_moving_the_rectangle_in_the_cell_moves_nothing_else(material):
    """The walls' places in the computational cell depend on their order alone."""
    x, y = np.linspace(-160, 160, 7), np.linspace(-170, 170, 5)
    shift = (37.5, -25.0)
    at_centre = _run(_problem(material, size=(250.0, 150.0)), orders=(5, 5))
    moved = _run(_problem(material, size=(250.0, 150.0), center=shift), orders=(5, 5))
    assert _totals(moved) == pytest.approx(_totals(at_centre), abs=1e-9)
    a = np.asarray(at_centre.field(x=x, y=y, z=[75.0]).values)
    b = np.asarray(moved.field(x=x + shift[0], y=y + shift[1], z=[75.0]).values)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()


def _blocks(rectangles):
    return [rcwa.Layer(thickness=H, regions=[rcwa.Region(
        geometry=rcwa.Rectangle(size=(x1 - x0, y1 - y0), center=((x0 + x1) / 2, (y0 + y1) / 2)),
        material=Material(epsilon=METAL)) for x0, x1, y0, y1 in rectangles])]


def test_a_block_written_as_touching_pieces_is_the_block():
    """The 200 nm metal block at theta 20 deg, as one rectangle, as two touching at x = 0, and
    moved by (146, 130) nm across the cell's corner as the four rectangles the edges cut it into
    (a rectangle may not cross the cell's edge): the same walls, so the same solve -- ``R`` and
    ``T`` to 1e-9 and the field, moved back with the Bloch phase, to 1e-8 of its largest value.
    At order 12 the three agree to 4e-11 and 5e-10. Before 2026-11-28 the joins were walls and the
    moved block's field was 6.7% from the block's at order 12."""
    theta, shift = np.radians(20.0), (146.0, 130.0)
    kx = 2 * np.pi / 900.0 * np.sin(theta)
    x, y = np.linspace(-190, 190, 9), np.linspace(-185, 185, 7)
    solves = {
        'one': (_blocks([(-100, 100, -100, 100)]), (0.0, 0.0)),
        'two': (_blocks([(-100, 0, -100, 100), (0, 100, -100, 100)]), (0.0, 0.0)),
        'four': (_blocks([(46, 200, 30, 200), (-200, -154, 30, 200), (46, 200, -200, -170),
                          (-200, -154, -200, -170)]), shift),
    }
    totals, fields = {}, {}
    for name, (layers, (dx, dy)) in solves.items():
        result = _run(_problem(METAL, layers=layers, theta=theta), orders=(6, 6))
        totals[name] = _totals(result)
        xs, ys = x + dx, y + dy
        folded_x, folded_y = (xs + P / 2) % P - P / 2, (ys + P / 2) % P - P / 2
        field = np.asarray(result.field(x=folded_x, y=folded_y, z=[75.0]).values)
        # E at the unfolded point is the folded one times the Bloch phase; the moved structure's
        # field at x + dx is the block's at x times exp(i kx dx)
        fields[name] = field * np.exp(1j * kx * (xs - folded_x - dx))[None, None, None, :, None]
    for name in ('two', 'four'):
        assert totals[name] == pytest.approx(totals['one'], abs=1e-9)
        assert np.abs(fields[name] - fields['one']).max() < 1e-8 * np.abs(fields['one']).max()


def test_a_rectangle_across_the_cell_edge_is_refused_for_what_it_is():
    """The stretched solve says what the unstretched one says, and how to write the rectangle
    instead (until 2026-11-28 it blamed a traced size)."""
    problem = _problem(METAL, center=(150.0, 0.0))
    for numerics in (_numerics(slope=None), _numerics()):
        with pytest.raises(ValidationError, match='crosses the cell edge.*cut it into'):
            rcwa.run(problem, numerics=numerics)


@pytest.mark.parametrize('wavelength, theta, phi, polarization, orders', [
    (900.0, 0.0, 0.0, 'p', 1), (900.0, np.deg2rad(20), np.deg2rad(30), 's', 1),
    (350.0, 0.0, 0.0, 'p', 5), (350.0, np.deg2rad(20), 0.0, 'p', 4)])
def test_a_lossless_layer_conserves_energy_with_any_number_of_propagating_orders(
        wavelength, theta, phi, polarization, orders):
    result = _run(_problem(12.25, wavelength=wavelength, theta=theta, phi=phi, polarization=polarization),
                  orders=(8, 8))
    R, T = _totals(result)
    assert abs(R + T - 1) < 1e-4
    assert int(np.sum(np.asarray(result.R) > 1e-12)) == orders


def test_an_evanescent_order_answers_zero_and_a_propagating_one_its_plane_wave():
    """In these coordinates only a propagating order of a half-space is a plane wave with an amplitude."""
    result = _run(_problem(12.25, wavelength=350.0), orders=(4, 4))
    propagating = np.zeros((9, 9), bool)
    for m, n in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        propagating[4 + n, 4 + m] = True
    for port in (result.reflection, result.transmission):
        for name in ('amplitude_ratio_s', 'amplitude_ratio_p'):
            amplitude = np.asarray(getattr(port, name))[0]
            assert np.all(amplitude[~propagating] == 0)
        both = sum(np.abs(np.asarray(getattr(port, name))[0]) ** 2
                   for name in ('amplitude_ratio_s', 'amplitude_ratio_p'))
        assert np.all(both[propagating] > 1e-8)
    assert np.all(np.asarray(result.R)[0][~propagating] == 0)


def test_a_grating_uniform_along_y_converges_to_what_the_unstretched_solve_approaches():
    """A lamellar grating in p polarization: the stretched solve is settled where the plain one still moves."""
    problem = _problem(12.25, size=(200.0, P))
    settled = [_totals(_run(problem, orders=(n, 0)))[0] for n in (8, 16)]
    plain = [_totals(_run(problem, orders=(n, 0), slope=None))[0] for n in (8, 16, 40)]
    assert settled[0] == pytest.approx(settled[1], abs=2e-5)
    gaps = [abs(value - settled[1]) for value in plain]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-4


# --------------------------------------------------------------------------- backends


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('material, wavelength', [(12.25, 900.0), (METAL, 350.0)])
def test_the_backends_agree(backend, material, wavelength):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    else:
        pytest.importorskip('torch')
    problem = _problem(material, size=(250.0, 150.0), center=(20.0, -10.0), wavelength=wavelength,
                       theta=np.deg2rad(20), phi=np.deg2rad(30))
    x, y = np.linspace(-190, 190, 7), np.linspace(-170, 170, 5)
    results = [rcwa.run(problem, numerics=_numerics(orders=(5, 5)), backend=name) for name in ('numpy', backend)]
    R, T, field = ([np.asarray(getter(r)) for r in results] for getter in (
        lambda r: r.R, lambda r: r.T, lambda r: r.field(x=x, y=y, z=[75.0]).values))
    assert np.abs(R[0] - R[1]).max() < 1e-9 and np.abs(T[0] - T[1]).max() < 1e-9
    # Measured 4e-10 at worst (the metal at 350 nm, order 6); the eigenvectors differ between LAPACKs.
    assert np.abs(field[0] - field[1]).max() < 1e-8 * np.abs(field[0]).max()


# --------------------------------------------------------------------------- the option and its refusals


def test_the_option_is_off_by_default_and_reported_when_set():
    assert rcwa.Numerics(orders=(1, 1)).wall_resolution is None
    assert 'wall_resolution' not in rcwa.Numerics(orders=(1, 1))._engine_kwargs()
    assert rcwa.Numerics(orders=(1, 1), wall_resolution=None) == rcwa.Numerics(orders=(1, 1))
    report = rcwa.check(_problem(), numerics=_numerics())
    assert not report.issues and report.resolved['wall_resolution'] == 0.02
    assert rcwa.check(_problem(), numerics=_numerics(slope=None)).resolved['wall_resolution'] is None
    # Eager only for now: which orders propagate, and a field's sample points, are read as numbers.
    assert report.capabilities['eager'] and not report.capabilities['jit'] and not report.capabilities['vmap']
    assert not report.capabilities['material_gradients']


@pytest.mark.parametrize('value', [0.0, -0.02, 1.5, float('nan'), '0.02', True])
def test_a_value_outside_the_interval_is_refused(value):
    with pytest.raises(ValidationError) as caught:
        rcwa.Numerics(orders=(1, 1), wall_resolution=value)
    assert caught.value.code == 'numerics.wall_resolution'


def _refused(problem, numerics):
    """The code and message of the refusal, which the check gives as the run does."""
    report = rcwa.check(problem, numerics=numerics)
    assert len(report.issues) == 1
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(problem, numerics=numerics)
    assert caught.value.code == report.issues[0].code
    return report.issues[0].code, str(caught.value)


@pytest.mark.parametrize('problem, numerics, said', [
    (lambda: _problem(), lambda: rcwa.Numerics(orders=(2, 2), wall_resolution=0.02), "factorization is 'nvf'"),
    (lambda: _problem(layers=[rcwa.Layer(thickness=H, regions=[rcwa.Region(
        geometry=rcwa.Circle(radius=100.0), material=Material(epsilon=4.0))])]),
     lambda: rcwa.Numerics(orders=(2, 2), wall_resolution=0.02), "factorization is 'nvf'"),
    (lambda: _problem(), lambda: _numerics(algorithm='etm'), "layer algorithm is 'ETM'"),
    (lambda: _problem(), lambda: _numerics(formulation='first_order'), "formulation='first_order'"),
    (lambda: _problem(bottom=12 + 0.5j), lambda: _numerics(), 'bottom half-space is not a lossless dielectric'),
    (lambda: _problem(top=-4.0), lambda: _numerics(), 'top half-space is not a lossless dielectric'),
    (lambda: _problem(fields=False, scattering=True), lambda: _numerics(), 'scattering matrix'),
    (lambda: _problem(), lambda: _numerics(orders=(4, 0)), 'single harmonic'),
    (lambda: _problem(size=(1.0, 200.0)), lambda: _numerics(), 'closer than the stretch can hold'),
])
def test_what_is_not_implemented_is_refused_by_name(problem, numerics, said):
    code, message = _refused(problem(), numerics())
    assert code == 'rcwa.unsupported' and said in message


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('layers', [None, [rcwa.Layer(thickness=H, regions=[rcwa.Region(
    geometry=rcwa.Circle(radius=100.0), material=Material(epsilon=METAL))])]], ids=['rectangle', 'circle'])
def test_single_precision_is_refused_by_name(backend, layers):
    """Measured 2026-11-28 against the wide finite-element table (the validation page): at single
    precision the stretched solve missed its criteria by up to a few percent, and at order 12 on the
    circle refused a half-space. The rounding the stretch amplifies is in the solve, not its tables.
    """
    if backend != 'numpy':
        pytest.importorskip(backend)
    problem, numerics = _problem(METAL, layers=layers), _numerics()
    report = rcwa.check(problem, numerics=numerics, backend=backend, precision='single')
    assert [issue.code for issue in report.issues] == ['rcwa.unsupported']
    with pytest.raises(UnsupportedFeatureError) as caught:
        rcwa.run(problem, numerics=numerics, backend=backend, precision='single')
    assert caught.value.code == 'rcwa.unsupported' and "precision='single'" in str(caught.value)
    # The default solve at single precision is not affected, nor the stretch at double.
    assert not rcwa.check(problem, numerics=rcwa.Numerics(orders=(4, 4)), backend=backend,
                          precision='single').issues
    if backend == 'numpy':
        assert not rcwa.check(problem, numerics=numerics).issues


def test_the_result_refuses_what_a_stretched_solve_does_not_hold():
    result = _run(_problem(METAL))
    with pytest.raises(UnsupportedFeatureError) as caught:
        result.boundary_fields(side='top', x=[0.0], y=[0.0])
    assert caught.value.code == 'rcwa.unsupported' and 'spacer layer' in str(caught.value)
    for rule in ('lalanne', 'split', 'splitw'):
        with pytest.raises(UnsupportedFeatureError) as caught:
            result.field(x=[0.0], y=[0.0], z=[75.0], reconstruction=rule)
        assert caught.value.code == 'field.reconstruction'
    with pytest.raises(UnsupportedFeatureError):
        result.scattering


def test_a_spacer_layer_gives_the_field_above_the_structure():
    """What ``boundary_fields`` cannot: a layer of the half-space's medium is sampled like any other.

    At slope 1 on the half-period block the field in the spacer is the unstretched solve's, whose
    spacer is merged into the half-space and answered analytically.
    """
    block = rcwa.Layer(thickness=H, regions=[rcwa.Region(geometry=rcwa.Rectangle(size=(200.0, 200.0)),
                                                         material=Material(epsilon=METAL))])
    problem = _problem(layers=[block, rcwa.Layer(thickness=50.0)])
    x = y = np.linspace(-180, 180, 5)
    fields = [np.asarray(_run(problem, slope=slope).field(x=x, y=y, z=[H + 10.0, H + 40.0]).values)
              for slope in (1.0, None)]
    assert np.abs(fields[0] - fields[1]).max() < 1e-9 * np.abs(fields[1]).max()


def test_a_stretched_solve_does_not_warn():
    """The option is stable (2026-11-22): neither a solve nor a refusal warns."""
    import warnings
    from wavebit.rcwa.diagnostics import ExperimentalWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error', ExperimentalWarning)
        _run(_problem(), orders=(2, 2))
        with pytest.raises(UnsupportedFeatureError):
            _run(_problem(), orders=(2, 2), algorithm='etm')
