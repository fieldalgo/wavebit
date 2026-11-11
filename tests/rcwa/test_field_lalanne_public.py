"""The Lalanne rule through the current result API, ``Result.field(reconstruction=...)``.

A result answers from its own snapshot, so these pin two things: that it answers what the solver's
own door answers, to rounding, on every path the rule takes, and that the snapshot does not reach
past the solve it belongs to. What the rule itself is worth is pinned against RETICOLO and a raster
in ``test_field_lalanne.py``, ``test_field_lalanne_shapes.py`` and
``tests/references/test_reticolo_2d_field_golden.py``.
"""
import numpy as np
import pytest

from wavebit import Material, Units, rcwa
from wavebit.errors import NumericalError, UnsupportedFeatureError, ValidationError
from wavebit.rcwa._factory import rcwa as engine
from wavebit.units import Z0

PERIOD = (400.0, 360.0)
X = np.array([-120.0, -45.0, 33.0, 105.0, 170.0])
Y = np.array([-130.0, -65.0, 25.0, 70.0])
Z = np.array([10.0, 50.0, 99.0, 120.0])
RECTANGLE = dict(cx=30.0, cy=-20.0, lx=150.0, ly=90.0)
OUTLINE = ((-150.0, -140.0), (100.0, -140.0), (100.0, -20.0), (0.0, -20.0), (0.0, 100.0), (-150.0, 100.0))


def _rectangles(cx=30.0):
    return (rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0), center=(cx, -20.0)),
                        material=Material(4.0)),)


def _polygon():
    return (rcwa.Region(geometry=rcwa.Polygon(vertices=OUTLINE), material=Material(4.0)),)


def _problem(regions=None, *, side='top', theta=0.2, phi=0.3, wavelength=620.0, epsilon=None,
             period=PERIOD, source=None, thickness=100.0, diffraction=True):
    first = (rcwa.Layer(thickness=thickness, epsilon=epsilon) if epsilon is not None else
             rcwa.Layer(thickness=thickness, background=Material(1.0),
                        regions=_rectangles() if regions is None else regions))
    second = (rcwa.Layer(thickness=60.0, epsilon=np.full(np.shape(epsilon), 2.25)) if epsilon is not None
              else rcwa.Layer(thickness=60.0, background=Material(2.25)))
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=period, top=Material(1.0), bottom=Material(2.25), layers=(first, second)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(source or rcwa.PlaneWave(
            side=side, theta=theta, phi=phi, polarization='p'),)),
        outputs=rcwa.Outputs(fields=True, diffraction=diffraction))


def _solver(structure, *, side='top', theta=0.2, phi=0.3, wavelength=620.0, orders=(2, 2), **kw):
    s = engine(backend='numpy', top=1.0, bot=2.25, period=list(PERIOD), truncation_order=list(orders),
               wavelength=wavelength, thickness=[100.0, 60.0], theta=theta, phi=phi, polarization=(0, 1),
               incidence_side=side, answers='field', structure=structure, **kw)
    s.solve()
    return s


def _shape_structure(shape):
    return [dict(background=1.0, shapes=[dict(shape, material=4.0)]), dict(background=2.25, shapes=[])]


def _raster():
    grid = np.ones((9, 8), dtype=float)
    grid[2:7, 3:6] = 4.0
    return grid


def _assert_doors_agree(public, solver, x=X, y=Y, z=Z):
    cell = np.asarray(solver.reconstruct_field(x, y, z, reconstruction='lalanne'))[0]
    np.testing.assert_allclose(public[..., :3], cell[..., :3], rtol=0, atol=1e-13)
    np.testing.assert_allclose(public[..., 3:] * Z0, cell[..., 3:], rtol=0, atol=1e-13)


@pytest.mark.parametrize('factorization', ['li', 'laurent', 'nvf'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers_on_shapes(factorization, side):
    result = rcwa.run(_problem(side=side), numerics=rcwa.Numerics(orders=(2, 2), factorization=factorization))
    public = result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values[0]
    _assert_doors_agree(public, _solver(_shape_structure(dict(type='rectangle', **RECTANGLE)), side=side,
                                        factorization=factorization))
    assert np.max(np.abs(public - result.field(x=X, y=Y, z=Z).values[0])) > 0.1


def test_a_result_answers_what_the_solver_answers_on_an_outline():
    result = rcwa.run(_problem(_polygon()), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    _assert_doors_agree(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values[0],
                        _solver(_shape_structure(dict(type='polygon', vertices=OUTLINE)), factorization='laurent'))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers_on_a_raster(side):
    result = rcwa.run(_problem(epsilon=_raster(), side=side), numerics=rcwa.Numerics(orders=(2, 2)))
    structure = np.stack([_raster(), np.full((9, 8), 2.25)])
    _assert_doors_agree(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values[0],
                        _solver(structure, side=side))


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('phi', [0.0, 0.3])
def test_a_result_answers_what_the_solver_answers_on_a_one_row_raster(phi, side):
    grid = np.ones((1, 16), dtype=float)
    grid[0, 4:10] = 4.0
    result = rcwa.run(_problem(epsilon=grid, phi=phi, side=side), numerics=rcwa.Numerics(orders=(4, 0)))
    assert result._field_records[0].kind in ('tm', 'conical', 'second_order')
    structure = np.stack([grid, np.full((1, 16), 2.25)])
    _assert_doors_agree(result.field(x=X, y=Y[:1], z=Z, reconstruction='lalanne').values[0],
                        _solver(structure, phi=phi, orders=(4, 0), side=side), y=Y[:1])


def test_counts_name_the_same_samples_as_the_solver():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    s = _solver(_shape_structure(dict(type='rectangle', **RECTANGLE)), factorization='laurent')
    _assert_doors_agree(result.field(x=5, y=4, z=3, reconstruction='lalanne').values[0], s, 5, 4, 3)
    _assert_doors_agree(result.field(x=5, y=Y, z=Z, reconstruction='lalanne').values[0], s, 5, Y, Z)


def test_only_the_transverse_e_of_a_patterned_layer_changes():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    plain = result.field(x=X, y=Y, z=Z).values
    rule = result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values
    np.testing.assert_array_equal(rule[..., 2:], plain[..., 2:])
    np.testing.assert_array_equal(rule[:, Z > 100.0], plain[:, Z > 100.0])
    assert np.max(np.abs(rule[:, Z < 100.0, :, :, :2] - plain[:, Z < 100.0, :, :, :2])) > 0.1


def test_the_default_is_the_sum_and_says_nothing():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    default = result.field(x=X, y=Y, z=Z)
    named = result.field(x=X, y=Y, z=Z, reconstruction='plain')
    np.testing.assert_array_equal(default.values, named.values)
    assert 'reconstruction' not in default.metadata
    assert result.field(x=X, y=Y, z=Z, reconstruction='lalanne').metadata['reconstruction'] == 'lalanne'


def test_every_wavelength_reads_its_own_scene():
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    both = rcwa.run(_problem(wavelength=np.array([620.0, 700.0])), numerics=numerics)
    rule = both.field(x=X, y=Y, z=Z, reconstruction='lalanne').values
    for index, wavelength in enumerate((620.0, 700.0)):
        alone = rcwa.run(_problem(wavelength=wavelength), numerics=numerics)
        np.testing.assert_allclose(rule[index],
                                   alone.field(x=X, y=Y, z=Z, reconstruction='lalanne').values[0],
                                   rtol=0, atol=1e-13)


def test_a_result_keeps_its_own_structure():
    """A later run, and an edit to the array it was given, reach neither the modes nor the rule."""
    grid = _raster()
    prepared = rcwa.prepare(_problem(epsilon=grid), numerics=rcwa.Numerics(orders=(2, 2)))
    first = prepared.run(_problem(epsilon=grid))
    before = first.field(x=X, y=Y, z=Z, reconstruction='lalanne').values.copy()
    grid[2:7, 3:6] = 9.0
    prepared.run(_problem(epsilon=grid))
    np.testing.assert_array_equal(first.field(x=X, y=Y, z=Z, reconstruction='lalanne').values, before)


@pytest.mark.parametrize('problem,numerics,says', [
    (lambda: _problem((rcwa.Region(geometry=rcwa.Circle(radius=80.0), material=Material(4.0)),)),
     rcwa.Numerics(orders=(2, 2), factorization='laurent'), 'slanted or curved'),
    (lambda: _problem((rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0), center=(30.0, -20.0)),
                                   material=Material(np.array([[4.0, 0.3, 0.0], [0.3, 4.0, 0.0], [0.0, 0.0, 9.0]],
                                                              dtype=complex))),)),
     rcwa.Numerics(orders=(2, 2), factorization='nvf'), 'off-diagonal'),
])
def test_what_the_rule_does_not_take_is_refused_by_name(problem, numerics, says):
    result = rcwa.run(problem(), numerics=numerics)
    result.field(x=X, y=Y, z=Z)
    with pytest.raises(UnsupportedFeatureError, match=says) as caught:
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')
    assert caught.value.code == 'field.reconstruction' and caught.value.path == 'reconstruction'


def test_an_unknown_rule_is_refused_before_anything_is_read():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    with pytest.raises(ValidationError, match='plain') as caught:
        result.field(x=X, y=Y, z=Z, reconstruction='nearest')
    assert caught.value.code == 'field.reconstruction' and caught.value.path == 'reconstruction'


def test_a_result_without_fields_says_so_first():
    problem = _problem().replace(outputs=rcwa.Outputs())
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    with pytest.raises(UnsupportedFeatureError) as caught:
        result.field(x=X, y=Y, z=Z, reconstruction='lalanne')
    assert caught.value.code == 'outputs.unavailable'


#: Off every wall of the rectangle at ``cx = 30``, where a derivative at a fixed point exists.
OFF_X, OFF_Y, OFF_Z = np.array([-120.0, -40.0, 33.0, 110.0, 170.0]), np.array([-130.0, -60.0, 20.0, 70.0]), Z[:2]


def _read(result):
    return result.field(x=OFF_X, y=OFF_Y, z=OFF_Z, reconstruction='lalanne').values


def _finite_difference(numerics, h=1e-4):
    def value(cx):
        return np.sum(np.abs(_read(rcwa.run(_problem(_rectangles(cx)), numerics=numerics))[..., :2]) ** 2)
    return (value(30.0 + h) - value(30.0 - h)) / (2 * h)


def test_jit_and_grad_through_a_prepared_problem_match_a_finite_difference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    numerics = rcwa.Numerics(orders=(2, 2), factorization='li')
    prepared = rcwa.prepare(_problem(), numerics=numerics, backend='jax')

    def values(cx):
        return _read(prepared.run(_problem(_rectangles(cx))))

    eager = _read(rcwa.run(_problem(_rectangles(30.0)), numerics=numerics))
    np.testing.assert_allclose(np.asarray(jax.jit(values)(30.0)), eager, rtol=0, atol=1e-11)
    gradient = jax.jit(jax.grad(lambda cx: jnp.sum(jnp.abs(values(cx)[..., :2]) ** 2)))(30.0)
    assert float(gradient) == pytest.approx(_finite_difference(numerics), rel=1e-5)


def test_grad_on_laurent_matches_a_finite_difference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    prepared = rcwa.prepare(_problem(), numerics=numerics, backend='jax')
    gradient = jax.grad(lambda cx: jnp.sum(jnp.abs(_read(prepared.run(_problem(_rectangles(cx))))[..., :2]) ** 2))(30.0)
    assert float(gradient) == pytest.approx(_finite_difference(numerics), rel=1e-5)


def test_grad_of_a_raster_matches_a_finite_difference():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    jnp = jax.numpy
    numerics = rcwa.Numerics(orders=(2, 2))
    grid = _raster()

    def problem(contrast):
        return _problem(epsilon=1.0 + (grid - 1.0) * contrast)

    prepared = rcwa.prepare(problem(1.0), numerics=numerics, backend='jax')
    gradient = jax.grad(lambda v: jnp.sum(jnp.abs(_read(prepared.run(problem(v)))[..., :2]) ** 2))(1.0)

    def value(contrast):
        return np.sum(np.abs(_read(rcwa.run(problem(contrast), numerics=numerics))[..., :2]) ** 2)

    assert float(gradient) == pytest.approx((value(1.0 + 1e-5) - value(1.0 - 1e-5)) / 2e-5, rel=1e-5)
    with pytest.raises(UnsupportedFeatureError, match='concrete period') as refused:
        jax.jit(lambda v: _read(prepared.run(problem(v))))(1.0)
    assert refused.value.code == 'field.reconstruction'


def test_a_result_returned_from_a_jit_still_takes_the_rule():
    """The copy of the structure crosses the trace with the modes, strings and all."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    numerics = rcwa.Numerics(orders=(2, 2), factorization='li')
    prepared = rcwa.prepare(_problem(), numerics=numerics, backend='jax')
    result = jax.jit(lambda cx: prepared.run(_problem(_rectangles(cx))))(30.0)
    eager = rcwa.run(_problem(), numerics=numerics)
    np.testing.assert_allclose(np.asarray(_read(result)), _read(eager), rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values),
                               eager.field(x=X, y=Y, z=Z, reconstruction='lalanne').values,
                               rtol=0, atol=1e-11)


def test_torch_gradient_matches_a_finite_difference():
    torch = pytest.importorskip('torch')
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    leaf = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    prepared = rcwa.prepare(_problem(), numerics=numerics, backend='torch')
    torch.sum(torch.abs(_read(prepared.run(_problem(_rectangles(leaf))))[..., :2]) ** 2).backward()
    assert float(leaf.grad) == pytest.approx(_finite_difference(numerics), rel=1e-5)


def _off_centre_raster():
    """Not symmetric about the cell's centre, so a phase or a mirror cannot hide."""
    grid = np.ones((36, 32))
    grid[8:28, 12:24] = 4.0
    return grid


def _raster_on(lattice):
    return rcwa.Problem(
        stack=rcwa.Stack(lattice=lattice, top=Material(1.0), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=100.0, epsilon=_off_centre_raster()),
                                 rcwa.Layer(thickness=60.0, epsilon=np.full((36, 32), 2.25)))),
        excitation=rcwa.Excitation(wavelength=620.0,
                                   sources=(rcwa.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p'),)),
        units=Units(length='nm'),
        outputs=rcwa.Outputs(fields=True, power_region='cell'))


@pytest.mark.parametrize('sign_x, sign_y', [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_a_raster_on_a_negative_side_is_refused_by_name(sign_x, sign_y):
    """A diagonal lattice with a negative side is read as spelled, on the general path, which has
    no x and y lines for the rule to follow: refused by name, while the default sum answers there."""
    numerics = rcwa.Numerics(orders=(4, 3), factorization='laurent')
    result = rcwa.run(_raster_on(((sign_x * 400.0, 0.0), (0.0, sign_y * 360.0))), numerics=numerics)
    z = [50.0, 130.0]
    assert np.all(np.isfinite(result.field(x=9, y=5, z=z).values))
    with pytest.raises(UnsupportedFeatureError, match='positive sides') as refused:
        result.field(x=9, y=5, z=z, reconstruction='lalanne')
    assert refused.value.code == 'field.reconstruction'


def test_a_result_keeps_its_own_lattice():
    """Measured before the copy: an edit moved E_x and E_y by 0.74 and the default by 0."""
    cell = np.array([[400.0, 0.0], [0.0, 360.0]])
    problem = _problem(_rectangles())
    problem = rcwa.Problem(
        stack=rcwa.Stack(lattice=cell, top=problem.stack.top, bottom=problem.stack.bottom,
                         layers=problem.stack.layers),
        excitation=problem.excitation, units=problem.units, outputs=problem.outputs)
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    before = np.array(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values)
    cell[0, 0] = 700.0
    np.testing.assert_array_equal(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values, before)
    cell[0, 0] = -400.0
    np.testing.assert_array_equal(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values, before)


def test_a_stack_with_nothing_patterned_answers_at_the_default_orders():
    """It was refused for not resolving a layer along an axis, and no layer has a wall."""
    layers = (rcwa.Layer(thickness=60.0, background=Material(2.0)),
              rcwa.Layer(thickness=60.0, background=Material(3.0)))
    problem = rcwa.Problem(
        stack=rcwa.Stack(period=(400.0, 360.0), top=Material(1.0), bottom=Material(2.25), layers=layers),
        excitation=rcwa.Excitation(wavelength=620.0,
                                   sources=(rcwa.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p'),)),
        units=Units(length='nm'),
        outputs=rcwa.Outputs(fields=True, power_region='cell'))
    for orders in ((0, 0), (2, 0), (2, 2)):
        result = rcwa.run(problem, numerics=rcwa.Numerics(orders=orders))
        np.testing.assert_array_equal(result.field(x=X, y=Y, z=[10.0, 90.0], reconstruction='lalanne').values,
                                      result.field(x=X, y=Y, z=[10.0, 90.0]).values)


def _metal_row_raster():
    grid = np.full((9, 8), 2.0)
    grid[4, :4] = -2.0
    return grid


def _metal_problem():
    return rcwa.Problem(
        stack=rcwa.Stack(period=(400.0, 360.0), top=Material(1.0), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=100.0, epsilon=_metal_row_raster()),
                                 rcwa.Layer(thickness=60.0, epsilon=np.full((9, 8), 2.25)))),
        excitation=rcwa.Excitation(wavelength=620.0,
                                   sources=(rcwa.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p'),)),
        units=Units(length='nm'),
        outputs=rcwa.Outputs(fields=True, power_region='cell'))


def test_a_singular_line_no_sample_is_on_does_not_fail_the_read():
    """Row 4 is singular: half its pixels are -2, half 2. These samples are on rows 1, 2, 5 and 6."""
    result = rcwa.run(_metal_problem(), numerics=rcwa.Numerics(orders=(1, 1), factorization='laurent'))
    values = result.field(x=[-120.0, -45.0, 33.0, 105.0], y=[-130.0, -65.0, 25.0, 70.0], z=[10.0, 50.0],
                          reconstruction='lalanne').values
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_a_singular_line_a_sample_is_on_fails_by_name(backend):
    pytest.importorskip(backend)
    result = rcwa.run(_metal_problem(), numerics=rcwa.Numerics(orders=(1, 1), factorization='laurent'),
                      backend=backend)
    with pytest.raises(NumericalError, match='singular') as failed:
        result.field(x=[-120.0], y=[0.0], z=[50.0], reconstruction='lalanne')
    assert failed.value.code == 'rcwa.singular'
    assert failed.value.path == 'reconstruction'


def test_the_samples_may_share_a_line_and_read_it_once():
    """A line read under many samples is the same as under each alone."""
    result = rcwa.run(_raster_on(((400.0, 0.0), (0.0, 360.0))), numerics=rcwa.Numerics(orders=(3, 3)))
    x, y = [-120.0, -118.0, 33.0, 34.0], [-130.0, -129.0, 25.0, 26.0]
    together = result.field(x=x, y=y, z=[50.0], reconstruction='lalanne').values
    for i, one_x in enumerate(x):
        for j, one_y in enumerate(y):
            alone = result.field(x=[one_x], y=[one_y], z=[50.0], reconstruction='lalanne').values
            np.testing.assert_allclose(alone[0, 0, 0, 0], together[0, 0, j, i], rtol=0, atol=1e-13)


def _uniform_raster_problem(contrast, direction, xp=np):
    grid = 2.0 * xp.ones((9, 8)) + contrast * direction
    return rcwa.Problem(
        stack=rcwa.Stack(period=(400.0, 360.0), top=Material(1.0), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=100.0, epsilon=grid),
                                 rcwa.Layer(thickness=60.0, epsilon=xp.full((9, 8), 2.25)))),
        excitation=rcwa.Excitation(wavelength=620.0,
                                   sources=(rcwa.PlaneWave(side='top', theta=0.2, phi=0.3, polarization='p'),)),
        units=Units(length='nm'),
        outputs=rcwa.Outputs(fields=True, power_region='cell'))


#: Inside pixels of the 9 x 8 raster, where a derivative at a fixed point exists.
_UNIFORM_AT = dict(x=[-120.0, -40.0, 33.0, 110.0, 170.0], y=[-130.0, -60.0, 20.0, 70.0], z=[10.0, 50.0, 120.0])


def _uniform_objective(values, xp):
    return xp.sum(xp.abs(values[..., :2]) ** 2)


def _uniform_derivative_by_difference(direction, numerics, step=1e-5):
    def value(contrast):
        result = rcwa.run(_uniform_raster_problem(contrast, direction), numerics=numerics)
        return _uniform_objective(result.field(reconstruction='lalanne', **_UNIFORM_AT).values, np)
    return (value(step) - value(-step)) / (2 * step)


def test_a_uniform_raster_takes_the_derivative_of_the_rule_not_of_the_sum():
    """Measured before the fix: the sum's derivative, 124% off and of the wrong sign, with no error."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    direction = np.random.default_rng(3).normal(size=(9, 8))
    numerics = rcwa.Numerics(orders=(2, 2))
    prepared = rcwa.prepare(_uniform_raster_problem(0.0, direction), numerics=numerics, backend='jax')

    def objective(contrast):
        result = prepared.run(_uniform_raster_problem(contrast, jnp.asarray(direction), jnp))
        return _uniform_objective(result.field(reconstruction='lalanne', **_UNIFORM_AT).values, jnp)

    assert float(jax.grad(objective)(0.0)) == pytest.approx(
        _uniform_derivative_by_difference(direction, numerics), rel=1e-5)


def test_a_uniform_raster_takes_the_rules_derivative_on_torch_too():
    torch = pytest.importorskip('torch')
    direction = np.random.default_rng(3).normal(size=(9, 8))
    numerics = rcwa.Numerics(orders=(2, 2))
    leaf = torch.zeros((), dtype=torch.float64, requires_grad=True)
    prepared = rcwa.prepare(_uniform_raster_problem(0.0, direction), numerics=numerics, backend='torch')
    result = prepared.run(_uniform_raster_problem(leaf, torch.as_tensor(direction), torch))
    _uniform_objective(result.field(reconstruction='lalanne', **_UNIFORM_AT).values, torch).backward()
    assert float(leaf.grad) == pytest.approx(_uniform_derivative_by_difference(direction, numerics), rel=1e-5)


def test_without_a_derivative_a_uniform_raster_stays_the_default_bit_for_bit():
    result = rcwa.run(_uniform_raster_problem(0.0, np.zeros((9, 8))), numerics=rcwa.Numerics(orders=(2, 2)))
    np.testing.assert_array_equal(result.field(reconstruction='lalanne', **_UNIFORM_AT).values,
                                  result.field(**_UNIFORM_AT).values)


def test_a_differentiated_angle_is_refused_as_a_traced_one_is():
    """Every neighbour of angle 0 is refused, so the number had nothing to be checked against."""
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')

    def problem(angle):
        return _problem((rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0), center=(30.0, -20.0), angle=angle),
                                     material=Material(4.0)),))

    prepared = rcwa.prepare(problem(0.0), numerics=numerics, backend='jax')
    with pytest.raises(UnsupportedFeatureError, match='slanted') as refused:
        jax.grad(lambda t: jnp.sum(jnp.abs(prepared.run(problem(t)).field(
            x=X, y=Y, z=[50.0], reconstruction='lalanne').values) ** 2))(0.0)
    assert refused.value.code == 'field.reconstruction'
    jax.grad(lambda t: jnp.sum(jnp.abs(prepared.run(problem(t)).field(x=X, y=Y, z=[50.0]).values) ** 2))(0.0)


# ---------------------------------------------------------------- H, where a layer names a mu
# What the rule is worth on H is pinned against RETICOLO in
# ``tests/references/test_reticolo_magnetic_field_golden.py``.

MAGNETIC = Material(4.0, mu=2.25)


def _magnetic_problem(side='top', xp=np, mu=2.25, cx=30.0):
    region = rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0), center=(cx, -20.0)), material=Material(4.0, mu=mu))
    return _problem((region,), side=side)


def _across_the_wall(x_inside=104.0, x_outside=106.0):
    """Samples either side of the rectangle's wall normal to x, inside the first layer."""
    return dict(x=np.array([x_inside, x_outside]), y=np.array([-13.0]), z=np.array([50.0]))


NUMERICS = rcwa.Numerics(orders=(8, 8), factorization='laurent')


def test_h_normal_to_a_magnetic_wall_jumps_by_mu_and_the_sum_does_not():
    """``mu H_x`` is what stays continuous across the wall, and that has no reference to lean on."""
    result = rcwa.run(_magnetic_problem(), numerics=NUMERICS)
    rule = result.field(**_across_the_wall(), reconstruction='lalanne').values[0, 0, 0]
    plain = result.field(**_across_the_wall()).values[0, 0, 0]
    assert abs(abs(rule[1, 3]) / abs(rule[0, 3]) - 2.25) < 0.2
    assert abs(abs(plain[1, 3]) / abs(plain[0, 3]) - 1.0) < 0.1


def test_h_along_a_magnetic_wall_stays_continuous():
    result = rcwa.run(_magnetic_problem(), numerics=NUMERICS)
    rule = result.field(**_across_the_wall(), reconstruction='lalanne').values[0, 0, 0]
    assert abs(abs(rule[1, 4]) / abs(rule[0, 4]) - 1.0) < 0.1


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers_on_h(side):
    result = rcwa.run(_magnetic_problem(side), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    solver = _solver(_shape_structure_with(MAGNETIC), side=side, factorization='laurent')
    door = np.asarray(solver.reconstruct_field(X, Y, Z, reconstruction='lalanne'))[0]
    ours = np.array(result.field(x=X, y=Y, z=Z, reconstruction='lalanne').values[0])
    ours[..., 3:] *= Z0
    assert np.abs(ours - door).max() < 1e-11 * np.abs(door).max()


def _shape_structure_with(material):
    return [dict(background=1.0, shapes=[dict(type='rectangle', **RECTANGLE, material=material)]),
            dict(background=2.25, shapes=[])]


def test_a_permeability_of_one_gives_the_sum_on_h_to_rounding():
    plain = rcwa.run(_magnetic_problem(mu=1.0), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    rule = plain.field(x=X, y=Y, z=Z, reconstruction='lalanne').values
    sum_ = plain.field(x=X, y=Y, z=Z).values
    assert np.abs(rule[..., 3:5] - sum_[..., 3:5]).max() < 1e-12 * np.abs(sum_[..., 3:5]).max()


def _raster_stack():
    return np.stack([_raster(), np.full_like(_raster(), 2.25)])


def test_a_uniform_permeability_field_stays_the_sum_on_h_bit_for_bit():
    grid = _raster_stack()
    s = _solver(grid, mu=np.full(grid.shape, 1.44, dtype=complex))
    rule = np.asarray(s.reconstruct_field(X, Y, Z, reconstruction='lalanne'))
    plain = np.asarray(s.reconstruct_field(X, Y, Z))
    np.testing.assert_array_equal(rule[..., 3:], plain[..., 3:])


def test_a_mu_assigned_after_the_solve_does_not_reach_the_rule():
    grid = _raster_stack()
    mu = np.ones(grid.shape, dtype=complex)
    mu[0, 2:7, 3:6] = 2.25
    s = _solver(grid, mu=mu)
    before = np.asarray(s.reconstruct_field(X, Y, Z, reconstruction='lalanne'))
    mu[0, 2:7, 3:6] = 9.0
    s.mu[0, 2:7, 3:6] = 9.0
    np.testing.assert_array_equal(s.reconstruct_field(X, Y, Z, reconstruction='lalanne'), before)
    s.mu = np.full(grid.shape, 3.0, dtype=complex)
    np.testing.assert_array_equal(s.reconstruct_field(X, Y, Z, reconstruction='lalanne'), before)
    state = s.solve()._field_state
    held = np.array(state.mu)
    s.mu = np.full(grid.shape, 3.0, dtype=complex)
    np.testing.assert_array_equal(state.mu, held)


def test_the_gate_asks_about_the_mu_the_solve_was_posed_with():
    grid = _raster_stack()
    mu = np.ones(grid.shape, dtype=complex)
    mu[0, 2:7, 3:6] = 2.25
    tensor = (mu[..., None, None] * np.eye(3)).copy()
    tensor[..., 0, 1] = tensor[..., 1, 0] = 0.1

    scalar = _solver(grid, mu=mu)
    before = np.asarray(scalar.reconstruct_field(X, Y, Z, reconstruction='lalanne'))
    scalar.mu = tensor
    np.testing.assert_array_equal(scalar.reconstruct_field(X, Y, Z, reconstruction='lalanne'), before)

    posed_on_a_tensor = _solver(grid, mu=tensor)
    posed_on_a_tensor.mu = mu
    with pytest.raises(NotImplementedError, match='tensor'):
        posed_on_a_tensor.reconstruct_field(X, Y, Z, reconstruction='lalanne')


def test_a_te_solve_without_a_mu_has_nothing_for_the_rule_to_change():
    grid = np.ones((1, 1, 8), dtype=complex)
    grid[0, 0, 3:6] = 4.0
    s = engine(backend='numpy', top=1.0, bot=2.25, period=[400.0], truncation_order=[3], wavelength=620.0,
               thickness=[100.0], theta=0.2, phi=None, tetm='te', answers='field', structure=grid)
    state = s.solve()._field_state
    np.testing.assert_array_equal(state.reconstruct(5, 1, 3, 'incident', 'lalanne'),
                                  state.reconstruct(5, 1, 3, 'incident'))


@pytest.mark.parametrize('tetm, slot', [('te', 0), ('tm', 3)])
def test_a_path_without_the_component_divides_nothing_for_it(tetm, slot):
    """TE has no E_x and TM no H_x, so a zero at the sample is not the 0/0 of a rule that read one."""
    grid = np.full((1, 1, 8), 4.0, dtype=complex)
    mu = np.ones((1, 1, 8), dtype=complex)
    grid[0, 0, 3:6], mu[0, 0, 3:6] = 2.25, 1.44
    (grid if tetm == 'te' else mu)[0, 0, 1] = 0.0
    s = engine(backend='numpy', top=1.0, bot=2.25, period=[400.0], truncation_order=[3], wavelength=620.0,
               thickness=[100.0], theta=0.2, phi=None, tetm=tetm, answers='field', structure=grid, mu=mu,
               factorization='laurent')
    read = np.asarray(s.solve()._field_state.reconstruct(np.array([-125.0]), np.array([0.0]), np.array([50.0]),
                                                         'incident', 'lalanne'))
    assert read[..., slot].item() == 0


def test_h_on_a_te_grating_changes_only_where_mu_does():
    grid = np.ones((1, 1, 8), dtype=complex)
    grid[0, 0, 3:6] = 4.0
    mu = np.ones_like(grid)
    mu[0, 0, 3:6] = 2.25
    s = engine(backend='numpy', top=1.0, bot=2.25, period=[400.0], truncation_order=[3], wavelength=620.0,
               thickness=[100.0], theta=0.2, phi=None, tetm='te', answers='field', structure=grid, mu=mu)
    state = s.solve()._field_state
    x = np.array([-120.0, 33.0, 170.0])
    rule = np.asarray(state.reconstruct(x, np.array([0.0]), np.array([50.0]), 'incident', 'lalanne'))
    plain = np.asarray(state.reconstruct(x, np.array([0.0]), np.array([50.0]), 'incident'))
    assert np.abs(rule[..., 3] - plain[..., 3]).max() > 1e-3
    np.testing.assert_array_equal(np.delete(rule, 3, axis=-1), np.delete(plain, 3, axis=-1))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_every_backend_reads_h_alike(backend):
    pytest.importorskip(backend)
    if backend == 'jax':
        from wavebit._common.backend import _enable_x64
        _enable_x64()
    grid = _raster_stack()
    mu = np.ones(grid.shape, dtype=complex)
    mu[0, 2:7, 3:6] = 2.25
    door = np.asarray(_solver(grid, mu=mu).reconstruct_field(X, Y, Z, reconstruction='lalanne'))
    s = engine(backend=backend, top=1.0, bot=2.25, period=list(PERIOD), truncation_order=[2, 2], wavelength=620.0,
               thickness=[100.0, 60.0], theta=0.2, phi=0.3, polarization=(0, 1), answers='field', structure=grid, mu=mu)
    s.solve()
    ours = np.asarray(s.reconstruct_field(X, Y, Z, reconstruction='lalanne'))
    assert np.abs(ours - door).max() < 1e-11 * np.abs(door).max()


def test_a_differentiated_mu_matches_a_finite_difference():
    jax = pytest.importorskip('jax')
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    at = _across_the_wall()

    def read(mu):
        result = rcwa.run(_magnetic_problem(mu=mu), numerics=numerics, backend='jax')
        return jax.numpy.real(result.field(**at, reconstruction='lalanne').values[0, 0, 0, 1, 3] * Z0)

    grad = float(jax.grad(read)(jax.numpy.asarray(2.25)))
    step = 1e-5
    difference = (float(read(2.25 + step)) - float(read(2.25 - step))) / (2 * step)
    assert grad == pytest.approx(difference, rel=1e-5)


def _raster_mu_problem(epsilon, mu):
    return rcwa.Problem(
        units=Units(length='nm'),
        stack=rcwa.Stack(period=PERIOD, top=Material(1.0), bottom=Material(2.25),
                         layers=(rcwa.Layer(thickness=100.0, epsilon=epsilon, mu=mu),)),
        excitation=rcwa.Excitation(wavelength=620.0, sources=(rcwa.PlaneWave(theta=0.3, phi=0.4, polarization='p'),)),
        outputs=rcwa.Outputs(fields=True))


def test_a_result_that_crossed_a_pytree_keeps_the_mu_the_rule_reads():
    """Without ``mu`` in the pytree the rule's ``H_x`` is the sum's, silently, after any crossing."""
    jax = pytest.importorskip('jax')
    from wavebit._common.backend import _enable_x64
    _enable_x64()
    grid = np.ones((9, 8), dtype=complex)
    grid[2:7, 3:6] = 4.0
    mu = np.ones((9, 8), dtype=complex)
    mu[2:7, 3:6] = 2.25
    numerics = rcwa.Numerics(orders=(3, 3), factorization='li')
    at = dict(x=np.array([-120.0, 33.0]), y=np.array([-13.0]), z=np.array([50.0]))
    eager = rcwa.run(_raster_mu_problem(grid, mu), numerics=numerics, backend='jax')
    rule = np.asarray(eager.field(**at, reconstruction='lalanne').values)
    assert np.abs(rule[..., 3] - np.asarray(eager.field(**at).values)[..., 3]).max() > 1e-5
    prepared = rcwa.prepare(_raster_mu_problem(grid, mu), numerics=numerics, backend='jax')
    crossed = jax.jit(lambda epsilon: prepared.run(_raster_mu_problem(epsilon, mu)))(jax.numpy.asarray(grid))
    mapped = jax.tree_util.tree_map(lambda leaf: leaf, eager)
    for result in (crossed, mapped):
        got = np.asarray(result.field(**at, reconstruction='lalanne').values)
        np.testing.assert_allclose(got, rule, rtol=0, atol=1e-11 * np.abs(rule).max())


def test_a_singular_mu_line_fails_by_name_and_names_mu():
    """``eps`` is positive and uniform, so a message blaming a negative ``eps`` alone is false."""
    mu = np.ones((9, 8), dtype=complex)
    mu[4, :4] = -1.0
    result = rcwa.run(_raster_mu_problem(np.full((9, 8), 4.0, dtype=complex), mu),
                      numerics=rcwa.Numerics(orders=(3, 3), factorization='laurent'))
    with pytest.raises(NumericalError, match='negative eps or mu') as failed:
        result.field(x=np.array([-120.0]), y=np.array([0.0]), z=np.array([50.0]), reconstruction='lalanne')
    assert failed.value.code == 'rcwa.singular'


def test_the_general_conical_layout_reads_h_as_the_family_one():
    """An x-cutoff layer arrives in the general layout, and these are the same modes written in it."""
    grid = np.ones((2, 1, 16), dtype=complex)
    grid[1, 0, 5:11] = 4.0
    grid[0] = 2.25
    mu = np.ones_like(grid)
    mu[1, 0, 5:11] = 2.25
    s = engine(backend='numpy', top=1.0, bot=2.25, period=[400.0], truncation_order=[3], wavelength=620.0,
               thickness=[100.0, 60.0], theta=0.2, phi=0.3, psi=0.4, answers='field', structure=grid, mu=mu)
    s.solve()
    x, y, z = np.array([-120.0, 33.0, 170.0]), np.array([0.0]), np.array([10.0, 120.0])
    family = np.asarray(s.reconstruct_field(x, y, z, reconstruction='lalanne'))
    general = []
    for epz, W, V, q, d, A_i, B, muz, amplitudes in s.layer_info_list:
        n = W.shape[0]
        zero = np.zeros((n, n), dtype=W.dtype)
        general.append((epz, np.block([[zero, W[:, n:]], [V[:n, :n], V[:n, n:]]]),
                        np.block([[W[:, :n], zero], [V[n:, :n], V[n:, n:]]]), np.diag(q), d, A_i, B, muz, amplitudes))
    s.layer_info_list = general
    assert s.layer_info_list[0][1].shape[0] == 2 * n
    np.testing.assert_allclose(s.reconstruct_field(x, y, z, reconstruction='lalanne'), family, rtol=0,
                               atol=1e-11 * np.abs(family).max())


@pytest.mark.parametrize('word, pair', [('p', (0, 1)), ('s', (1, 0))])
@pytest.mark.parametrize('theta', [0.0, 0.2])
def test_the_pure_1d_path_gives_the_conical_paths_rule(word, pair, theta):
    """A polarization word solves a one-row raster on the pure TE or TM path, a Jones pair on the
    conical one, which the RETICOLO goldens pin. Both rules agree across the two paths (measured
    1.6e-14 of the peak for lalanne on TM)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
    from test_unified_rcwa_pure_1d import ORDERS, grating
    pure = rcwa.run(grating(word, theta=theta), numerics=ORDERS)
    conical = rcwa.run(grating(pair, theta=theta), numerics=ORDERS)
    assert pure.diagnostics['solve_path'] == (f'1d_{"tm" if word == "p" else "te"}',)
    assert conical.diagnostics['solve_path'] == ('1d_conical',)
    x, y, z = np.linspace(-240.0, 240.0, 97), [0.0], [30.0, 70.0]
    for rule in ('plain', 'lalanne'):
        a = np.asarray(pure.field(x=x, y=y, z=z, reconstruction=rule).values)
        b = np.asarray(conical.field(x=x, y=y, z=z, reconstruction=rule).values)
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b)), rule
