"""The split rules, ``Result.field(reconstruction='split' | 'splitw')``, on a layer of round shapes.

What the rules are worth is pinned against a finite-element field in
``tests/references/test_gyptis_cylinder_field_golden.py``. These pin what they are: which
components they change, what they make continuous across the curved wall, that the two doors,
the three backends and the two formulations agree on them, that their derivative is a finite
difference's, and that everything they were not measured on is refused by name. ``'splitw'`` is
the windowed rule and the one most tests read; ``'split'`` runs the same operators with the
window at 1, and the tests that could tell the two apart take both. One circle is the layer most
tests read. An ellipse and several circles are built another way (the solve's analytic normal
field, a window on the distance to the nearest wall) and have their own tests at the end.
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

# One CI shard unit per test function (tests.yml, --shard): the file is about 15 worker-minutes,
# and no test reads what another left behind (the module fixtures are solves each shard repeats).
pytestmark = pytest.mark.shard_unit(per='test')


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


@pytest.mark.parametrize('rule', ['split', 'splitw'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers(side, rule):
    public = _read(rcwa.run(_problem(side=side), numerics=NUMERICS), rule)[0]
    solver = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[3, 3],
                    wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                    incidence_side=side, answers='field', factorization='nvf',
                    structure=[dict(background=1.0, shapes=[dict(
                        type='circle', cx=CENTRE[0], cy=CENTRE[1], lx=2 * RADIUS, material=EPSILON)])])
    solver.solve()
    cell = np.asarray(solver.reconstruct_field(X, Y, Z, reconstruction=rule))[0]
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


@pytest.mark.parametrize('rule', ['split', 'splitw'])
@pytest.mark.parametrize('angle', [0.0, 0.6, 2.3, 4.0])
def test_the_rule_carries_the_walls_jump_and_the_sum_does_not(finer, angle, rule):
    """``eps E_n`` and ``E_t`` are continuous across the wall, as far as the truncated projector is one.

    Measured at order 6 on these four rays: ``eps E_n`` mismatches by 0.4 to 1.4% of the outer
    value and ``E_t`` by 0.2 to 2.3%. The sum is continuous in both, so its ``eps E_n`` mismatches
    by the contrast, 5. A wrong frame or sign in the rule's operators fails this; an accurate
    field needs more (``tests/references/test_gyptis_cylinder_field_golden.py``).
    """
    (n_out, t_out), (n_in, t_in) = _across_the_wall(finer, rule, angle)
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
    """The uniform layers differ from their half-spaces by 1e-9, inside a tolerance that merges them."""
    layers = (rcwa.Layer(thickness=40.0, background=Material(2.25 * (1 + 1e-9))),
              rcwa.Layer(thickness=150.0, background=Material(1.0), regions=_circle()),
              rcwa.Layer(thickness=30.0, background=Material(1.0 + 1e-9)))
    numerics = rcwa.Numerics(orders=(3, 3), factorization='nvf', half_space_merge_tolerance=1e-6)
    result = rcwa.run(_problem(layers=layers), numerics=numerics)
    assert result._field_records[0].merged[:2] == (1, 1)
    lone = rcwa.run(_problem(), numerics=NUMERICS)
    z = np.array([10.0, 70.0, 115.0, 200.0])
    plain, rule = _read(result, 'plain', z=z), _read(result, z=z)
    np.testing.assert_array_equal(rule[:, [0, 3]], plain[:, [0, 3]])
    # The same scene with the uniform layers left to the half-spaces, up to the phase of 40 nm.
    np.testing.assert_allclose(np.abs(rule[:, 1:3, ..., :2]), np.abs(_read(lone)[..., :2]), rtol=0, atol=1e-7)
    assert np.abs(rule[:, 1:3, ..., :2] - plain[:, 1:3, ..., :2]).max() > 1e-2


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_every_backend_reads_the_rules_alike(result, backend):
    pytest.importorskip(backend)
    if backend == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
    other = rcwa.run(_problem(), numerics=NUMERICS, backend=backend)
    for rule in ('split', 'splitw'):
        np.testing.assert_allclose(_read(other, rule), _read(result, rule), rtol=0, atol=1e-11, err_msg=rule)


def test_the_two_rules_differ_only_by_the_window(result):
    """``'split'`` is ``'splitw'`` with the window at 1: the same door, other operators, another field."""
    split, windowed = _read(result, 'split'), _read(result)
    np.testing.assert_array_equal(split[..., 2:], windowed[..., 2:])
    assert np.max(np.abs(split[..., :2] - windowed[..., :2])) > 1e-3
    assert result.field(x=X, y=Y, z=Z, reconstruction='split').metadata['reconstruction'] == 'split'


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


@pytest.mark.parametrize('rule', ['split', 'splitw'])
@pytest.mark.parametrize('disk, background', [(-20.0 + 1.5j, 1.0), (2.25, -20.0 + 1.5j)])
def test_where_eps_changes_sign_the_rule_answers_on_the_negative_side_only(rule, disk, background):
    """A metal disk in air takes the rule inside the disk; a hole in a metal takes it outside the hole."""
    layers = (rcwa.Layer(thickness=150.0, background=Material(background), regions=_circle(epsilon=disk)),)
    result = rcwa.run(_problem(layers=layers), numerics=NUMERICS)
    x = np.linspace(-190.0, 190.0, 39)
    y = np.linspace(-170.0, 170.0, 35)
    plain, ruled = _read(result, 'plain', x=x, y=y), _read(result, rule, x=x, y=y)
    inside = np.hypot(*np.meshgrid(x - CENTRE[0], y - CENTRE[1])) < RADIUS
    metal = inside if np.real(disk) < 0 else ~inside
    np.testing.assert_allclose(ruled[:, :, ~metal], plain[:, :, ~metal], rtol=0, atol=1e-12)
    assert np.min(np.max(np.abs(ruled[:, :, metal, :2] - plain[:, :, metal, :2]), axis=-1)) > 1e-6
    np.testing.assert_array_equal(ruled[..., 2:], plain[..., 2:])


def test_two_metals_take_the_rule_on_both_sides():
    layers = (rcwa.Layer(thickness=150.0, background=Material(-8.0 + 1.0j), regions=_circle(epsilon=-20.0 + 1.5j)),)
    result = rcwa.run(_problem(layers=layers), numerics=NUMERICS)
    assert np.min(np.max(np.abs(_read(result) - _read(result, 'plain'))[..., :2], axis=-1)) > 1e-9


#: One compile each: the windowed rule on the dielectric, the unwindowed one on the metal, whose
#: samples outside it are the ones a refusal could lose to the sum.
_TRACED = [('splitw', EPSILON), ('split', -20.0 + 1.5j)]


def _solver(backend, radius=RADIUS, period=PERIOD, material=EPSILON):
    return engine(backend=backend, top=1.0, bot=2.25, period=list(period), truncation_order=[2, 2],
                  wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                  answers='field', factorization='nvf',
                  structure=[dict(background=1.0, shapes=[dict(type='circle', cx=CENTRE[0], cy=CENTRE[1],
                                                               lx=2 * radius, material=material)])])


@pytest.mark.parametrize('rule, material', _TRACED, ids=['dielectric', 'metal'])
def test_a_negative_side_under_a_trace_answers_nan_where_the_rule_reads(rule, material):
    """A trace has no sign to refuse, and the operators are built for positive sides."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    frozen = _solver('numpy', material=material).freeze_geometry()

    def field(px):
        s = _solver('jax', period=(px, PERIOD[1]), material=material)
        s.solve(geometry=frozen)
        return s.reconstruct_field(X, Y, 3, reconstruction=rule)

    traced = jax.jit(field)
    negative = np.asarray(traced(-PERIOD[0]))
    assert np.all(np.isnan(negative[..., :2])) and np.all(np.isfinite(negative[..., 2:]))
    eager = _solver('numpy', material=material)
    eager.solve()
    np.testing.assert_allclose(np.asarray(traced(PERIOD[0])),
                               np.asarray(eager.reconstruct_field(X, Y, 3, reconstruction=rule)), rtol=0, atol=1e-10)


@pytest.mark.parametrize('rule, material', _TRACED, ids=['dielectric', 'metal'])
def test_a_circle_a_trace_hides_the_fit_of_answers_nan_on_both_sides_of_its_wall(rule, material):
    """Eagerly the solve refuses the circle. On a metal the samples outside it must not fall back to the sum."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    with pytest.raises(ValueError, match='larger than the period'):
        _solver('numpy', radius=185.0, material=material).solve()
    frozen = _solver('numpy', material=material).freeze_geometry()

    def field(radius):
        s = _solver('jax', radius=radius, material=material)
        s.solve(geometry=frozen)
        return s.reconstruct_field(X, Y, 3, reconstruction=rule)

    traced = jax.jit(field)
    wide = np.asarray(traced(185.0))
    assert np.all(np.isnan(wide[..., :2])) and np.all(np.isfinite(wide[..., 2:]))
    assert np.all(np.isfinite(np.asarray(traced(RADIUS))))


def _ellipse(radii=(90.0, 60.0), centre=(0.0, 0.0), epsilon=EPSILON, angle=0.0):
    return (rcwa.Region(geometry=rcwa.Ellipse(radii=radii, center=centre, angle=angle), material=Material(epsilon)),)


def _rectangle():
    return (rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0)), material=Material(EPSILON)),)


def _two_circles(radius=50.0, centres=((-90.0, 0.0), (90.0, 0.0)), epsilon=EPSILON, other=None):
    return (_circle(radius=radius, centre=centres[0], epsilon=epsilon)
            + _circle(radius=radius, centre=centres[1], epsilon=epsilon if other is None else other))


def _raster():
    grid = np.ones((36, 40))
    xx, yy = np.meshgrid(np.arange(40) - 19.5, np.arange(36) - 17.5)
    grid[xx ** 2 + yy ** 2 < 81.0] = EPSILON
    return grid


REFUSED = [
    (lambda: _problem(_rectangle()), NUMERICS, 'layer 0 holds 1 rectangle'),
    (lambda: _problem(_ellipse(angle=0.4)), NUMERICS, 'is turned'),
    (lambda: _problem(_ellipse(radii=(70.0, 50.0), centre=(-100.0, 0.0)) + _circle(radius=40.0, centre=(110.0, 0.0))),
     NUMERICS, 'layer 0 holds 1 ellipse, 1 circle'),
    (lambda: _problem(_two_circles(other=4.0)), NUMERICS, 'are of several'),
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
def test_what_the_rules_were_not_measured_on_is_refused_by_name(problem, numerics, says):
    result = rcwa.run(problem(), numerics=numerics)
    for rule in ('split', 'splitw'):
        with pytest.raises(UnsupportedFeatureError, match=says) as refused:
            result.field(x=X, y=Y, z=Z, reconstruction=rule)
        assert refused.value.code == 'field.reconstruction'
        assert f'reconstruction={rule!r}' in str(refused.value)
    result.field(x=X, y=Y, z=Z)                      # the sum still answers


def test_circles_that_touch_are_refused_images_counted():
    """Asked of the refusal itself: solving overlapping circles to ask it would cost minutes."""
    lattice = wavebit.Lattice.rectangular(*PERIOD)

    def refused(*circles):
        layer = dict(background=1.0, shapes=[dict(type='circle', cx=cx, cy=cy, lx=2 * r, material=EPSILON)
                                             for cx, cy, r in circles])
        return splitw._layer_refusal(layer, lattice, 0, "reconstruction='splitw'")

    assert refused((-90.0, 0.0, 50.0), (90.0, 0.0, 50.0)) is None
    assert 'touch or overlap' in refused((-90.0, 0.0, 90.0), (90.0, 0.0, 90.0))        # tangent
    assert 'touch or overlap' in refused((-90.0, 0.0, 95.0), (90.0, 0.0, 95.0))
    assert 'touch or overlap' in refused((-150.0, 0.0, 55.0), (150.0, 0.0, 55.0))      # across the cell's edge: 100 apart
    assert 'leaves no room' in refused((-90.0, 0.0, 180.0), (90.0, 0.0, 20.0))


def test_the_solvers_door_refuses_as_the_result_does():
    solver = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[2, 2],
                    wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                    answers='field', factorization='nvf',
                    structure=[dict(background=1.0, shapes=[dict(type='rectangle', cx=0.0, cy=0.0, lx=180.0,
                                                                 ly=120.0, material=EPSILON)])])
    solver.solve()
    with pytest.raises(NotImplementedError, match='holds 1 rectangle'):
        solver.reconstruct_field(X, Y, Z, reconstruction='splitw')


def test_an_unknown_rule_names_the_four(result):
    with pytest.raises(ValidationError, match="'plain', 'lalanne', 'split' or 'splitw'") as refused:
        result.field(x=X, y=Y, z=Z, reconstruction='splitv')
    assert refused.value.code == 'field.reconstruction'


def test_lalanne_still_refuses_the_circle_and_points_at_nothing_new(result):
    with pytest.raises(UnsupportedFeatureError, match='slanted or curved'):
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')


# ------------------------------------------------------------ an ellipse, several circles

SHAPES = {'ellipse': lambda **kw: _ellipse(radii=(110.0, 70.0), centre=(8.0, -6.0), **kw),
          'circles': lambda **kw: _two_circles(radius=52.0, centres=((-88.0, 12.0), (95.0, -20.0)), **kw)}


@pytest.fixture(scope='module', params=sorted(SHAPES))
def shaped(request):
    return request.param, rcwa.run(_problem(SHAPES[request.param]()), numerics=NUMERICS)


def test_other_round_shapes_change_the_in_plane_e_alone_and_each_rule_its_own_way(shaped):
    _, result = shaped
    plain, split, windowed = (_read(result, rule) for rule in ('plain', 'split', 'splitw'))
    for ruled in (split, windowed):
        np.testing.assert_array_equal(ruled[..., 2:], plain[..., 2:])
        assert np.max(np.abs(ruled[..., :2] - plain[..., :2])) > 1e-2
    assert np.max(np.abs(split[..., :2] - windowed[..., :2])) > 1e-3


def test_other_round_shapes_answer_alike_at_both_doors_and_on_every_backend(shaped):
    name, result = shaped
    structure = [dict(background=1.0, shapes=[
        dict(type='ellipse', cx=8.0, cy=-6.0, lx=220.0, ly=140.0, material=EPSILON)] if name == 'ellipse' else [
        dict(type='circle', cx=cx, cy=cy, lx=104.0, material=EPSILON) for cx, cy in ((-88.0, 12.0), (95.0, -20.0))])]
    solver = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[3, 3],
                    wavelength=900.0, thickness=[150.0], theta=0.2, phi=0.3, polarization=(0, 1),
                    answers='field', factorization='nvf', structure=structure)
    solver.solve()
    for rule in ('split', 'splitw'):
        public, cell = _read(result, rule)[0], np.asarray(solver.reconstruct_field(X, Y, Z, reconstruction=rule))[0]
        np.testing.assert_allclose(public[..., :3], cell[..., :3], rtol=0, atol=1e-12, err_msg=rule)
        np.testing.assert_allclose(public[..., 3:] * Z0, cell[..., 3:], rtol=0, atol=1e-12, err_msg=rule)
    for backend in ('jax', 'torch'):
        if backend == 'jax':
            jax = pytest.importorskip('jax')
            jax.config.update('jax_enable_x64', True)
        else:
            pytest.importorskip('torch')
        other = rcwa.run(_problem(SHAPES[name]()), numerics=NUMERICS, backend=backend)
        np.testing.assert_allclose(_read(other), _read(result), rtol=0, atol=1e-10, err_msg=backend)


def test_two_circles_read_the_same_rule_at_mirrored_samples():
    """Normal incidence with E along x, two equal circles placed symmetrically: a half turn about the
    cell's centre maps the scene to itself and E_x, E_y to themselves, and the rule must keep that."""
    result = rcwa.run(_problem(_two_circles(radius=30.0, centres=((-300.0, 0.0), (300.0, 0.0))), period=(1200.0, 360.0),
                               theta=0.0, phi=0.0), numerics=rcwa.Numerics(orders=(6, 3), factorization='nvf'))
    x, y = np.array([-345.0, -322.0, -300.0, -275.0, -262.0]), np.array([-28.0, 0.0, 17.0])
    here, there = _read(result, x=x, y=y), _read(result, x=-x[::-1], y=-y[::-1])
    np.testing.assert_allclose(there[:, :, ::-1, ::-1, :2], here[..., :2], rtol=0, atol=1e-9)
    assert np.max(np.abs(here[..., :2] - _read(result, 'plain', x=x, y=y)[..., :2])) > 1e-2


def test_shapes_moved_together_give_the_same_field_moved_to_what_the_grid_resolves(shaped):
    """One circle is carried by a phase and moves exactly. These are sampled where they are, and the
    grid meets the seams of their normal field elsewhere after a move: measured 2.9e-4 of a field of
    order 1 for ``'split'`` on the ellipse and 1.4e-4 on the circles, 1.1e-6 and 2.7e-5 for ``'splitw'``,
    whose window is zero at most seams. The sum itself moves to 2e-14."""
    name, result = shaped
    shift = (37.0, -22.0)
    regions = (_ellipse(radii=(110.0, 70.0), centre=(8.0 + shift[0], -6.0 + shift[1])) if name == 'ellipse' else
               _two_circles(radius=52.0, centres=((-88.0 + shift[0], 12.0 + shift[1]), (95.0 + shift[0], -20.0 + shift[1]))))
    moved = rcwa.run(_problem(regions, theta=0.0, phi=0.0), numerics=NUMERICS)
    still = rcwa.run(_problem(SHAPES[name](), theta=0.0, phi=0.0), numerics=NUMERICS)
    for rule, atol in (('split', 1e-3), ('splitw', 1e-4)):
        np.testing.assert_allclose(_read(moved, rule, x=X + shift[0], y=Y + shift[1])[..., :2],
                                   _read(still, rule)[..., :2], rtol=0, atol=atol, err_msg=rule)
    np.testing.assert_allclose(_read(moved, 'plain', x=X + shift[0], y=Y + shift[1]), _read(still, 'plain'),
                               rtol=0, atol=1e-12)


def test_shapes_of_the_backgrounds_permittivity_keep_the_sum_bit_for_bit():
    for name in SHAPES:
        result = rcwa.run(_problem(SHAPES[name](epsilon=1.0)), numerics=NUMERICS)
        np.testing.assert_array_equal(_read(result), _read(result, 'plain'))


@pytest.mark.parametrize('name', sorted(SHAPES))
def test_on_other_metal_shapes_the_rule_answers_inside_them_only(name):
    result = rcwa.run(_problem(SHAPES[name](epsilon=-20.0 + 1.5j)), numerics=NUMERICS)
    x, y = np.linspace(-190.0, 190.0, 39), np.linspace(-170.0, 170.0, 35)
    plain, ruled = _read(result, 'plain', x=x, y=y), _read(result, x=x, y=y)
    gx, gy = np.meshgrid(x, y)
    if name == 'ellipse':
        inside = ((gx - 8.0) / 110.0) ** 2 + ((gy + 6.0) / 70.0) ** 2 < 1
    else:
        inside = (np.hypot(gx + 88.0, gy - 12.0) < 52.0) | (np.hypot(gx - 95.0, gy + 20.0) < 52.0)
    np.testing.assert_allclose(ruled[:, :, ~inside], plain[:, :, ~inside], rtol=0, atol=1e-12)
    assert np.min(np.max(np.abs(ruled[:, :, inside, :2] - plain[:, :, inside, :2]), axis=-1)) > 1e-6


def test_the_reach_of_other_shapes_is_their_smallest_inradius_and_their_room_is_the_narrowest_gap():
    lattice = wavebit.Lattice.rectangular(400.0, 360.0)
    ellipse = dict(shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=220.0, ly=140.0, material=EPSILON)])
    rows, sides = splitw._round_shapes(np, ellipse, lattice, np.float64, None)
    assert float(splitw._reach(np, rows)) == 70.0
    assert float(splitw._clearance(np, rows, sides)) == 180.0            # 400 - 220 against 360 - 140
    circles = dict(shapes=[dict(type='circle', cx=cx, cy=0.0, lx=2 * r, material=EPSILON)
                           for cx, r in ((-90.0, 60.0), (90.0, 40.0))])
    rows, sides = splitw._round_shapes(np, circles, lattice, np.float64, None)
    assert float(splitw._reach(np, rows)) == 40.0
    assert float(splitw._clearance(np, rows, sides)) == pytest.approx(80.0)    # 180 - 100, the pair in the cell
    x, y = np.array([[-90.0, -25.0, 0.0, 135.0]]), np.zeros((1, 4))
    distance, inside = splitw._wall_distance(np, rows, sides, x + 200.0, y + 180.0)
    np.testing.assert_allclose(distance[0], [60.0, 5.0, 30.0, 5.0], atol=1e-9)
    assert inside[0].tolist() == [True, False, False, False]


def test_jax_grad_on_other_shapes_matches_a_finite_difference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    cases = (('circles', lambda v: _two_circles(radius=v, centres=((-88.0, 12.0), (95.0, -20.0))), 52.0),
             ('ellipse', lambda v: _ellipse(radii=(110.0, v), centre=(8.0, -6.0)), 70.0))
    for name, regions, at in cases:
        def numeric(v):
            return _objective(_read(rcwa.run(_problem(regions(v)), numerics=NUMERICS)))

        prepared = rcwa.prepare(_problem(regions(at)), numerics=NUMERICS, backend='jax')

        def objective(v):
            values = prepared.run(_problem(regions(v))).field(reconstruction='splitw', **OFF).values
            return jnp.sum(jnp.abs(values[..., :2]) ** 2)

        difference = (numeric(at + 1e-4) - numeric(at - 1e-4)) / 2e-4
        assert float(jax.grad(objective)(at)) == pytest.approx(difference, rel=1e-5), name

