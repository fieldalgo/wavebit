"""The windowed split rule, ``Result.field(reconstruction='splitw')``, on a layer that holds a circle.

What the rule is worth is pinned against a finite-element field in
``tests/references/test_gyptis_cylinder_field_golden.py``. These pin what it is: which
components it changes, what it makes continuous across the curved wall, that the two doors, the
three backends and the two formulations agree on it, that its derivative is a finite
difference's, and that everything it was not measured on is refused by name.
"""
import numpy as np
import pytest

import wavebit
from wavebit import Material, Units, rcwa
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.rcwa._engine import splitw
from wavebit.rcwa._engine.factorization import INVERSE_RULE_BOUND
from wavebit.rcwa._factory import rcwa as engine
from wavebit.units import Z0

PERIOD = (400.0, 360.0)
#: Not round numbers: a length held in single precision anywhere on the way shows.
RADIUS, CENTRE, EPSILON = 91.3, (10.7, -5.2), 6.0
X = np.array([-150.0, -80.0, -45.0, 33.0, 60.0, 170.0])
Y = np.array([-130.0, -65.0, 25.0, 70.0])
Z = np.array([30.0, 75.0])
NUMERICS = rcwa.Numerics(orders=(3, 3), factorization='nvf')


def _circle(radius=RADIUS, centre=CENTRE, epsilon=EPSILON):
    return (rcwa.Region(geometry=rcwa.Circle(radius=radius, center=centre), material=Material(epsilon=epsilon)),)


def _problem(regions=None, *, side='top', theta=0.2, phi=0.3, period=PERIOD, thickness=150.0,
             sources=None, layers=None, outputs=None, **stack):
    layers = layers or (rcwa.Layer(thickness=thickness, background=Material(1.0),
                                   regions=_circle() if regions is None else regions),)
    stack = dict(dict(period=period), **stack)
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(top=Material(1.0), bottom=Material(2.25), layers=layers, **stack),
        excitation=rcwa.Excitation(wavelength=900.0, sources=sources or (rcwa.PlaneWave(
            side=side, theta=theta, phi=phi, polarization='p'),)),
        outputs=outputs or rcwa.Outputs(fields=True))


def _read(result, rule='splitw', x=X, y=Y, z=Z):
    values = result.field(x=x, y=y, z=z, reconstruction=rule).values
    return np.asarray(values.detach() if hasattr(values, 'detach') else values)


@pytest.fixture(scope='module')
def result():
    return rcwa.run(_problem(), numerics=NUMERICS)


@pytest.fixture(scope='module')
def finer():
    return rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(6, 6), factorization='nvf'))


def test_only_the_in_plane_e_of_the_patterned_layer_changes():
    layers = (rcwa.Layer(thickness=150.0, background=Material(1.0), regions=_circle()),
              rcwa.Layer(thickness=60.0, background=Material(2.25)))
    result = rcwa.run(_problem(layers=layers), numerics=NUMERICS)
    z = np.array([30.0, 75.0, 180.0])
    plain, rule = _read(result, 'plain', z=z), _read(result, z=z)
    np.testing.assert_array_equal(rule[..., 2:], plain[..., 2:])
    np.testing.assert_array_equal(rule[:, 2], plain[:, 2])            # the uniform layer keeps the sum
    assert np.max(np.abs(rule[:, :2, ..., :2] - plain[:, :2, ..., :2])) > 0.05
    assert result.field(x=X, y=Y, z=z, reconstruction='splitw').metadata['reconstruction'] == 'splitw'


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers(side):
    public = _read(rcwa.run(_problem(side=side), numerics=NUMERICS))[0]
    solver = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[3, 3],
                    wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                    incidence_side=side, answers='field', factorization='nvf',
                    structure=[dict(background=1.0, shapes=[dict(
                        type='circle', cx=CENTRE[0], cy=CENTRE[1], lx=2 * RADIUS, material=EPSILON)])])
    solver.solve()
    cell = np.asarray(solver.reconstruct_field(X, Y, Z, reconstruction='splitw'))[0]
    np.testing.assert_allclose(public[..., :3], cell[..., :3], rtol=0, atol=1e-13)
    np.testing.assert_allclose(public[..., 3:] * Z0, cell[..., 3:], rtol=0, atol=1e-13)


def _across_the_wall(result, rule, angle, offset=1e-5):
    """The normal and tangential ``E`` just outside and just inside the wall, along one ray."""
    n = np.array([np.cos(angle), np.sin(angle)])
    t = np.array([-n[1], n[0]])
    out = []
    for radius in (RADIUS + offset, RADIUS - offset):
        point = np.array(CENTRE) + radius * n
        e = np.asarray(result.field(x=[point[0]], y=[point[1]], z=[75.0], reconstruction=rule).values)[0, 0, 0, 0, :2]
        out.append((e @ n, e @ t))
    return out


@pytest.mark.parametrize('angle', [0.0, 0.6, 2.3, 4.0])
def test_the_rule_carries_the_walls_jump_and_the_sum_does_not(finer, angle):
    """``eps E_n`` and ``E_t`` are continuous across the wall, as far as the truncated projector is one.

    Measured at order 6 on these four rays: ``eps E_n`` mismatches by 0.4 to 1.4% of the outer
    value and ``E_t`` by 0.2 to 2.3%. The sum is continuous in both, so its ``eps E_n`` mismatches
    by the contrast, 5. A wrong frame or sign in the rule's operators fails this; an accurate
    field needs more (``tests/references/test_gyptis_cylinder_field_golden.py``).
    """
    (n_out, t_out), (n_in, t_in) = _across_the_wall(finer, 'splitw', angle)
    assert abs(n_out - EPSILON * n_in) < 0.04 * abs(n_out)
    assert abs(t_out - t_in) < 0.04 * max(abs(t_out), abs(n_out))
    (n_out, _), (n_in, _) = _across_the_wall(finer, 'plain', angle)
    assert abs(n_out - EPSILON * n_in) > 4.9 * abs(n_out)


def test_a_circle_moved_across_the_cell_gives_the_same_field_moved(result):
    shift = np.array([37.0, -21.0])
    moved = rcwa.run(_problem(_circle(centre=tuple(np.array(CENTRE) + shift))), numerics=NUMERICS)
    # An oblique wave carries its phase to the new centre.
    plain = _read(moved, 'plain', x=X + shift[0], y=Y + shift[1]) / _read(result, 'plain')
    np.testing.assert_allclose(_read(moved, x=X + shift[0], y=Y + shift[1]),
                               _read(result) * plain[0, 0, 0, 0, 2] / abs(plain[0, 0, 0, 0, 2]), rtol=0, atol=1e-11)


def test_a_sample_reads_the_nearest_image_of_the_circle(result):
    np.testing.assert_allclose(_read(result, x=X + PERIOD[0], y=Y - PERIOD[1])[..., :2],
                               _read(result)[..., :2] * (_read(result, 'plain', x=X + PERIOD[0], y=Y - PERIOD[1])
                                                         / _read(result, 'plain'))[..., 2:3], rtol=0, atol=1e-11)


def test_counts_sample_the_points_they_name(result):
    points = result.field(x=5, y=4, z=3).coordinates
    counted = np.asarray(result.field(x=5, y=4, z=3, reconstruction='splitw').values)
    explicit = _read(result, x=np.asarray(points['x']), y=np.asarray(points['y']), z=[75.0])
    np.testing.assert_allclose(counted[:, 1], explicit[:, 0], rtol=0, atol=1e-12)


def test_every_wavelength_reads_its_own_scene(result):
    problem = _problem()
    both = rcwa.run(problem.replace(excitation=rcwa.Excitation(
        wavelength=np.array([900.0, 950.0]), sources=problem.excitation.sources)), numerics=NUMERICS)
    values = _read(both)
    np.testing.assert_array_equal(values[0], _read(result)[0])
    assert np.max(np.abs(values[1] - values[0])) > 0.01


def test_a_circle_of_the_backgrounds_permittivity_keeps_the_sum_bit_for_bit():
    result = rcwa.run(_problem(_circle(epsilon=1.0)), numerics=NUMERICS)
    np.testing.assert_array_equal(_read(result), _read(result, 'plain'))


def test_the_default_is_untouched(result):
    assert 'reconstruction' not in result.field(x=X, y=Y, z=Z).metadata
    np.testing.assert_array_equal(_read(result, 'plain'), np.asarray(result.field(x=X, y=Y, z=Z).values))


@pytest.mark.parametrize('formulation', ['first_order'])
def test_the_first_order_formulation_reads_the_same_rule(result, formulation):
    other = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(3, 3), factorization='nvf', formulation=formulation))
    np.testing.assert_allclose(_read(other), _read(result), rtol=0, atol=1e-10)


def test_a_wave_as_a_source_answers_what_the_ordinary_solve_answers(result):
    lit = rcwa.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p')
    dark = rcwa.PlaneWave(side='bottom', theta=0.1, phi=0.3, polarization='p', amplitude=0.0)
    source = rcwa.run(_problem(sources=(lit, dark), outputs=rcwa.Outputs(
        fields=True, diffraction=False, power_region='cell')), numerics=NUMERICS)
    assert source._field_records[0].kind == 'source'
    np.testing.assert_allclose(_read(source), _read(result), rtol=0, atol=1e-10 * np.abs(_read(result)).max())
    assert np.abs(_read(source) - _read(source, 'plain')).max() > 1e-2


def test_a_layer_merged_into_a_half_space_keeps_the_sum_and_the_circle_takes_the_rule():
    layers = (rcwa.Layer(thickness=40.0, background=Material(2.25)),
              rcwa.Layer(thickness=150.0, background=Material(1.0), regions=_circle()),
              rcwa.Layer(thickness=30.0, background=Material(1.0)))
    result = rcwa.run(_problem(layers=layers), numerics=NUMERICS)
    lone = rcwa.run(_problem(), numerics=NUMERICS)
    z = np.array([10.0, 70.0, 115.0, 200.0])
    plain, rule = _read(result, 'plain', z=z), _read(result, z=z)
    np.testing.assert_array_equal(rule[:, [0, 3]], plain[:, [0, 3]])
    # The same scene with the uniform layers left to the half-spaces, up to the phase of 40 nm.
    np.testing.assert_allclose(np.abs(rule[:, 1:3, ..., :2]), np.abs(_read(lone)[..., :2]), rtol=0, atol=1e-9)


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_every_backend_reads_the_rule_alike(result, backend):
    pytest.importorskip(backend)
    if backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    other = rcwa.run(_problem(), numerics=NUMERICS, backend=backend)
    np.testing.assert_allclose(_read(other), _read(result), rtol=0, atol=1e-11)


OFF = dict(x=X, y=Y, z=Z)       # off the wall of the circle, where a derivative at a fixed point exists


def _objective(values):
    return np.sum(np.abs(np.asarray(values)[..., :2]) ** 2)


def _finite_difference(name, at, h):
    def value(v):
        regions = _circle(**{name: v}) if name != 'cx' else _circle(centre=(v, CENTRE[1]))
        return _objective(_read(rcwa.run(_problem(regions), numerics=NUMERICS)))
    return (value(at + h) - value(at - h)) / (2 * h)


def test_jax_grad_matches_a_finite_difference():
    """Through the window and the wall (the radius), the phase (the centre) and the inverse rule (``eps``)."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    prepared = rcwa.prepare(_problem(), numerics=NUMERICS, backend='jax')
    for name, at, h in (('radius', RADIUS, 1e-4), ('cx', CENTRE[0], 1e-4), ('epsilon', EPSILON, 1e-5)):
        def objective(v):
            regions = _circle(**{name: v}) if name != 'cx' else _circle(centre=(v, CENTRE[1]))
            values = prepared.run(_problem(regions)).field(reconstruction='splitw', **OFF).values
            return jnp.sum(jnp.abs(values[..., :2]) ** 2)

        assert float(jax.grad(objective)(at)) == pytest.approx(_finite_difference(name, at, h), rel=1e-5), name


def test_a_jit_over_a_material_reads_the_rule():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    prepared = rcwa.prepare(_problem(), numerics=NUMERICS, backend='jax')
    eager = _read(rcwa.run(_problem(), numerics=NUMERICS))
    traced = jax.jit(lambda eps: prepared.run(_problem(_circle(epsilon=eps))).field(
        reconstruction='splitw', **OFF).values)(EPSILON)
    np.testing.assert_allclose(np.asarray(traced), eager, rtol=0, atol=1e-11)
    crossed = jax.jit(lambda eps: prepared.run(_problem(_circle(epsilon=eps))))(EPSILON)
    np.testing.assert_allclose(_read(crossed), eager, rtol=0, atol=1e-11)


def test_torch_gradient_matches_a_finite_difference():
    torch = pytest.importorskip('torch')
    leaf = torch.tensor(RADIUS, dtype=torch.float64, requires_grad=True)
    prepared = rcwa.prepare(_problem(), numerics=NUMERICS, backend='torch')
    values = prepared.run(_problem(_circle(radius=leaf))).field(reconstruction='splitw', **OFF).values
    torch.sum(torch.abs(values[..., :2]) ** 2).backward()
    assert float(leaf.grad) == pytest.approx(_finite_difference('radius', RADIUS, 1e-4), rel=1e-5)


def test_the_window_is_one_at_the_wall_and_zero_past_its_reach():
    distance = np.array([0.0, 2.0, 2.5, 5.0, 9.99, 10.0, 40.0])
    window = splitw._window(np, distance, np.float64(10.0))
    np.testing.assert_array_equal(window[[0, 1, 2]], 1.0)
    np.testing.assert_allclose(window[[5, 6]], 0.0, atol=1e-30)
    assert np.all(np.diff(window) <= 0) and 0 < window[4] < window[3] < 1


@pytest.mark.parametrize('radius, period, reach', [(100.0, (400.0, 400.0), 100.0), (150.0, (400.0, 400.0), 50.0),
                                                   (90.0, (400.0, 240.0), 30.0)])
def test_the_window_reaches_the_axis_or_half_the_gap_to_the_next_wall(radius, period, reach):
    layer = dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=2 * radius, material=4.0)])
    lattice = wavebit.Lattice.rectangular(*period)
    circle = splitw._circle(np, layer, lattice, np.float64, None)
    assert float(circle.reach) == pytest.approx(reach)


def test_a_metal_takes_the_floor_the_nvf_solve_takes():
    """Without it ``[[1/eps]]^-1`` has the smeared interface's pole, far above any ``eps`` of the layer."""
    metal = -20.0 + 1.5j
    layer = dict(background=1.0, shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=200.0, material=metal)])
    lattice = wavebit.Lattice.rectangular(400.0, 400.0)
    inverse = splitw._inverse_rule(np, layer, lattice, 4, 4, np.complex128)
    assert np.linalg.norm(inverse, 2) <= INVERSE_RULE_BOUND * abs(metal) * (1 + 1e-12)
    result = rcwa.run(_problem(_circle(epsilon=metal)), numerics=NUMERICS)
    assert np.all(np.isfinite(_read(result)))


def _ellipse():
    return (rcwa.Region(geometry=rcwa.Ellipse(radii=(90.0, 60.0)), material=Material(EPSILON)),)


def _rectangle():
    return (rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0)), material=Material(EPSILON)),)


def _two_circles():
    return _circle(radius=50.0, centre=(-90.0, 0.0)) + _circle(radius=50.0, centre=(90.0, 0.0))


def _raster():
    grid = np.ones((36, 40))
    xx, yy = np.meshgrid(np.arange(40) - 19.5, np.arange(36) - 17.5)
    grid[xx ** 2 + yy ** 2 < 81.0] = EPSILON
    return grid


REFUSED = [
    (lambda: _problem(_rectangle()), NUMERICS, 'holds one circle'),
    (lambda: _problem(_ellipse()), NUMERICS, 'holds one circle'),
    (lambda: _problem(_two_circles()), NUMERICS, 'layer 0 holds 2 circles'),
    (lambda: _problem((rcwa.Region(geometry=rcwa.Polygon(vertices=((-90.0, -60.0), (80.0, -60.0), (0.0, 90.0))),
                                   material=Material(EPSILON)),)), NUMERICS, 'layer 0 holds 1 polygon'),
    (lambda: _problem(_circle(epsilon=(4.0, 5.0, 6.0))), NUMERICS, 'holds a tensor'),
    (lambda: _problem((rcwa.Region(geometry=rcwa.Circle(radius=RADIUS), material=Material(EPSILON, mu=2.0)),)),
     NUMERICS, 'rebuilds E alone'),
    (lambda: _problem(layers=(rcwa.Layer(thickness=150.0, epsilon=_raster()),)),
     rcwa.Numerics(orders=(3, 3), factorization='nvf'), 'a raster names none'),
    (lambda: _problem(layers=(rcwa.Layer(thickness=150.0, epsilon=np.where(np.arange(16) < 6, EPSILON, 1.0)[None, :]),),
                      phi=0.0), rcwa.Numerics(orders=(4, 0)), "1D grating's walls"),
    (lambda: _problem(period=None, lattice=wavebit.Lattice.hexagonal(400.0)), NUMERICS, 'rectangular cell'),
]


@pytest.mark.parametrize('problem, numerics, says', REFUSED)
def test_what_the_rule_was_not_measured_on_is_refused_by_name(problem, numerics, says):
    result = rcwa.run(problem(), numerics=numerics)
    with pytest.raises(UnsupportedFeatureError, match=says) as refused:
        result.field(x=X, y=Y, z=Z, reconstruction='splitw')
    assert refused.value.code == 'field.reconstruction'
    result.field(x=X, y=Y, z=Z)                      # the sum still answers


def test_the_solvers_door_refuses_as_the_result_does():
    solver = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[2, 2],
                    wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                    answers='field', factorization='nvf',
                    structure=[dict(background=1.0, shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=180.0,
                                                                 ly=120.0, material=EPSILON)])])
    solver.solve()
    with pytest.raises(NotImplementedError, match='holds one circle'):
        solver.reconstruct_field(X, Y, Z, reconstruction='splitw')


def test_an_unknown_rule_names_the_three(result):
    with pytest.raises(ValidationError, match="'plain', 'lalanne' or 'splitw'") as refused:
        result.field(x=X, y=Y, z=Z, reconstruction='split')
    assert refused.value.code == 'field.reconstruction'


def test_lalanne_still_refuses_the_circle_and_points_at_nothing_new(result):
    with pytest.raises(UnsupportedFeatureError, match='slanted or curved'):
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')
