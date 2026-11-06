"""The continuity rule through the current result API, ``Result.field(discontinuous=...)``.

A result answers from its own snapshot, so these pin two things: that it answers what the solver's
own door answers, to rounding, on every path the rule takes, and that the snapshot does not reach
past the solve it belongs to. What the rule itself is worth is pinned against RETICOLO and a raster
in ``test_field_continuity.py``, ``test_field_continuity_shapes.py`` and
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
    cell = np.asarray(solver.reconstruct_field(x, y, z, discontinuous='continuity'))[0]
    np.testing.assert_allclose(public[..., :3], cell[..., :3], rtol=0, atol=1e-13)
    np.testing.assert_allclose(public[..., 3:] * Z0, cell[..., 3:], rtol=0, atol=1e-13)


@pytest.mark.parametrize('factorization', ['li', 'laurent', 'nvf'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers_on_shapes(factorization, side):
    result = rcwa.run(_problem(side=side), numerics=rcwa.Numerics(orders=(2, 2), factorization=factorization))
    public = result.field(x=X, y=Y, z=Z, discontinuous='continuity').values[0]
    _assert_doors_agree(public, _solver(_shape_structure(dict(type='rectangle', **RECTANGLE)), side=side,
                                        factorization=factorization))
    assert np.max(np.abs(public - result.field(x=X, y=Y, z=Z).values[0])) > 0.1


def test_a_result_answers_what_the_solver_answers_on_an_outline():
    result = rcwa.run(_problem(_polygon()), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    _assert_doors_agree(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values[0],
                        _solver(_shape_structure(dict(type='polygon', vertices=OUTLINE)), factorization='laurent'))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_result_answers_what_the_solver_answers_on_a_raster(side):
    result = rcwa.run(_problem(epsilon=_raster(), side=side), numerics=rcwa.Numerics(orders=(2, 2)))
    structure = np.stack([_raster(), np.full((9, 8), 2.25)])
    _assert_doors_agree(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values[0],
                        _solver(structure, side=side))


@pytest.mark.parametrize('phi', [0.0, 0.3])
def test_a_result_answers_what_the_solver_answers_on_a_one_row_raster(phi):
    grid = np.ones((1, 16), dtype=float)
    grid[0, 4:10] = 4.0
    result = rcwa.run(_problem(epsilon=grid, phi=phi), numerics=rcwa.Numerics(orders=(4, 0)))
    assert result._field_records[0].kind in ('tm', 'conical', 'second_order')
    structure = np.stack([grid, np.full((1, 16), 2.25)])
    _assert_doors_agree(result.field(x=X, y=Y[:1], z=Z, discontinuous='continuity').values[0],
                        _solver(structure, phi=phi, orders=(4, 0)), y=Y[:1])


def test_counts_name_the_same_samples_as_the_solver():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    s = _solver(_shape_structure(dict(type='rectangle', **RECTANGLE)), factorization='laurent')
    _assert_doors_agree(result.field(x=5, y=4, z=3, discontinuous='continuity').values[0], s, 5, 4, 3)
    _assert_doors_agree(result.field(x=5, y=Y, z=Z, discontinuous='continuity').values[0], s, 5, Y, Z)


def test_only_the_transverse_e_of_a_patterned_layer_changes():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    plain = result.field(x=X, y=Y, z=Z).values
    rule = result.field(x=X, y=Y, z=Z, discontinuous='continuity').values
    np.testing.assert_array_equal(rule[..., 2:], plain[..., 2:])
    np.testing.assert_array_equal(rule[:, Z > 100.0], plain[:, Z > 100.0])
    assert np.max(np.abs(rule[:, Z < 100.0, :, :, :2] - plain[:, Z < 100.0, :, :, :2])) > 0.1


def test_the_default_is_the_sum_and_says_nothing():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    default = result.field(x=X, y=Y, z=Z)
    named = result.field(x=X, y=Y, z=Z, discontinuous='fourier')
    np.testing.assert_array_equal(default.values, named.values)
    assert 'discontinuous' not in default.metadata
    assert result.field(x=X, y=Y, z=Z, discontinuous='continuity').metadata['discontinuous'] == 'continuity'


def test_every_wavelength_reads_its_own_scene():
    numerics = rcwa.Numerics(orders=(2, 2), factorization='laurent')
    both = rcwa.run(_problem(wavelength=np.array([620.0, 700.0])), numerics=numerics)
    rule = both.field(x=X, y=Y, z=Z, discontinuous='continuity').values
    for index, wavelength in enumerate((620.0, 700.0)):
        alone = rcwa.run(_problem(wavelength=wavelength), numerics=numerics)
        np.testing.assert_allclose(rule[index],
                                   alone.field(x=X, y=Y, z=Z, discontinuous='continuity').values[0],
                                   rtol=0, atol=1e-13)


def test_a_result_keeps_its_own_structure():
    """A later run, and an edit to the array it was given, reach neither the modes nor the rule."""
    grid = _raster()
    prepared = rcwa.prepare(_problem(epsilon=grid), numerics=rcwa.Numerics(orders=(2, 2)))
    first = prepared.run(_problem(epsilon=grid))
    before = first.field(x=X, y=Y, z=Z, discontinuous='continuity').values.copy()
    grid[2:7, 3:6] = 9.0
    prepared.run(_problem(epsilon=grid))
    np.testing.assert_array_equal(first.field(x=X, y=Y, z=Z, discontinuous='continuity').values, before)


@pytest.mark.parametrize('problem,numerics,says', [
    (lambda: _problem((rcwa.Region(geometry=rcwa.Circle(radius=80.0), material=Material(4.0)),)),
     rcwa.Numerics(orders=(2, 2), factorization='laurent'), 'slanted or curved'),
    (lambda: _problem(), rcwa.Numerics(orders=(2, 2), formulation='first_order', factorization='laurent'),
     'first-order'),
    (lambda: _problem(), rcwa.Numerics(orders=(2, 0), factorization='laurent'), 'one order along an axis'),
    (lambda: _problem((rcwa.Region(geometry=rcwa.Rectangle(size=(150.0, 90.0), center=(30.0, -20.0)),
                                   material=Material(np.diag([4.0, 4.0, 9.0]).astype(complex))),)),
     rcwa.Numerics(orders=(2, 2), factorization='nvf'), 'tensor material'),
])
def test_what_the_rule_does_not_take_is_refused_by_name(problem, numerics, says):
    result = rcwa.run(problem(), numerics=numerics)
    result.field(x=X, y=Y, z=Z)
    with pytest.raises(UnsupportedFeatureError, match=says) as caught:
        result.field(x=X, y=Y, z=Z, discontinuous='continuity')
    assert caught.value.code == 'field.discontinuous' and caught.value.path == 'discontinuous'


def test_a_solve_of_sources_is_refused_by_name():
    source = rcwa.ElectricDipole(position=(0.0, 0.0, 130.0), moment=(1.0, 0.0, 0.0))
    result = rcwa.run(_problem(source=source, diffraction=False),
                      numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    result.field(x=X, y=Y, z=Z)
    with pytest.raises(UnsupportedFeatureError, match='plane-wave') as caught:
        result.field(x=X, y=Y, z=Z, discontinuous='continuity')
    assert caught.value.code == 'field.discontinuous'


def test_an_unknown_rule_is_refused_before_anything_is_read():
    result = rcwa.run(_problem(), numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    with pytest.raises(ValidationError, match='fourier') as caught:
        result.field(x=X, y=Y, z=Z, discontinuous='nearest')
    assert caught.value.code == 'field.discontinuous' and caught.value.path == 'discontinuous'


def test_a_result_without_fields_says_so_first():
    problem = _problem().replace(outputs=rcwa.Outputs())
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    with pytest.raises(UnsupportedFeatureError) as caught:
        result.field(x=X, y=Y, z=Z, discontinuous='continuity')
    assert caught.value.code == 'outputs.unavailable'


#: Off every wall of the rectangle at ``cx = 30``, where a derivative at a fixed point exists.
OFF_X, OFF_Y, OFF_Z = np.array([-120.0, -40.0, 33.0, 110.0, 170.0]), np.array([-130.0, -60.0, 20.0, 70.0]), Z[:2]


def _read(result):
    return result.field(x=OFF_X, y=OFF_Y, z=OFF_Z, discontinuous='continuity').values


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
    assert refused.value.code == 'field.discontinuous'


def test_a_result_returned_from_a_jit_still_takes_the_rule():
    """The copy of the structure crosses the trace with the modes, strings and all."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    numerics = rcwa.Numerics(orders=(2, 2), factorization='li')
    prepared = rcwa.prepare(_problem(), numerics=numerics, backend='jax')
    result = jax.jit(lambda cx: prepared.run(_problem(_rectangles(cx))))(30.0)
    eager = rcwa.run(_problem(), numerics=numerics)
    np.testing.assert_allclose(np.asarray(_read(result)), _read(eager), rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values),
                               eager.field(x=X, y=Y, z=Z, discontinuous='continuity').values,
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
def test_a_raster_on_a_negative_side_is_the_positive_basis_in_every_component(sign_x, sign_y):
    """Measured before the fix: E_x and E_y one phase away, k_x P_x, from E_z, H and the default."""
    numerics = rcwa.Numerics(orders=(4, 4))
    x, y, z = [-175.0, -75.0, 25.0, 125.0], [-160.0, -20.0, 80.0], [50.0, 130.0]
    positive = rcwa.run(_raster_on(((400.0, 0.0), (0.0, 360.0))), numerics=numerics)
    negative = rcwa.run(_raster_on(((sign_x * 400.0, 0.0), (0.0, sign_y * 360.0))), numerics=numerics)
    np.testing.assert_allclose(negative.field(x=x, y=y, z=z, discontinuous='continuity').values,
                               positive.field(x=x, y=y, z=z, discontinuous='continuity').values,
                               rtol=0, atol=1e-11)


def test_a_result_keeps_its_own_lattice():
    """Measured before the copy: an edit moved E_x and E_y by 0.74 and the default by 0."""
    cell = np.array([[400.0, 0.0], [0.0, 360.0]])
    problem = _problem(_rectangles())
    problem = rcwa.Problem(
        stack=rcwa.Stack(lattice=cell, top=problem.stack.top, bottom=problem.stack.bottom,
                         layers=problem.stack.layers),
        excitation=problem.excitation, units=problem.units, outputs=problem.outputs)
    result = rcwa.run(problem, numerics=rcwa.Numerics(orders=(2, 2), factorization='laurent'))
    before = np.array(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values)
    cell[0, 0] = 700.0
    np.testing.assert_array_equal(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values, before)
    cell[0, 0] = -400.0
    np.testing.assert_array_equal(result.field(x=X, y=Y, z=Z, discontinuous='continuity').values, before)


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
        np.testing.assert_array_equal(result.field(x=X, y=Y, z=[10.0, 90.0], discontinuous='continuity').values,
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
                          discontinuous='continuity').values
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_a_singular_line_a_sample_is_on_fails_by_name(backend):
    pytest.importorskip(backend)
    result = rcwa.run(_metal_problem(), numerics=rcwa.Numerics(orders=(1, 1), factorization='laurent'),
                      backend=backend)
    with pytest.raises(NumericalError, match='singular') as failed:
        result.field(x=[-120.0], y=[0.0], z=[50.0], discontinuous='continuity')
    assert failed.value.code == 'rcwa.singular'
    assert failed.value.path == 'discontinuous'


def test_the_samples_may_share_a_line_and_read_it_once():
    """A line read under many samples is the same as under each alone."""
    result = rcwa.run(_raster_on(((400.0, 0.0), (0.0, 360.0))), numerics=rcwa.Numerics(orders=(3, 3)))
    x, y = [-120.0, -118.0, 33.0, 34.0], [-130.0, -129.0, 25.0, 26.0]
    together = result.field(x=x, y=y, z=[50.0], discontinuous='continuity').values
    for i, one_x in enumerate(x):
        for j, one_y in enumerate(y):
            alone = result.field(x=[one_x], y=[one_y], z=[50.0], discontinuous='continuity').values
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
        return _uniform_objective(result.field(discontinuous='continuity', **_UNIFORM_AT).values, np)
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
        return _uniform_objective(result.field(discontinuous='continuity', **_UNIFORM_AT).values, jnp)

    assert float(jax.grad(objective)(0.0)) == pytest.approx(
        _uniform_derivative_by_difference(direction, numerics), rel=1e-5)


def test_a_uniform_raster_takes_the_rules_derivative_on_torch_too():
    torch = pytest.importorskip('torch')
    direction = np.random.default_rng(3).normal(size=(9, 8))
    numerics = rcwa.Numerics(orders=(2, 2))
    leaf = torch.zeros((), dtype=torch.float64, requires_grad=True)
    prepared = rcwa.prepare(_uniform_raster_problem(0.0, direction), numerics=numerics, backend='torch')
    result = prepared.run(_uniform_raster_problem(leaf, torch.as_tensor(direction), torch))
    _uniform_objective(result.field(discontinuous='continuity', **_UNIFORM_AT).values, torch).backward()
    assert float(leaf.grad) == pytest.approx(_uniform_derivative_by_difference(direction, numerics), rel=1e-5)


def test_without_a_derivative_a_uniform_raster_stays_the_default_bit_for_bit():
    result = rcwa.run(_uniform_raster_problem(0.0, np.zeros((9, 8))), numerics=rcwa.Numerics(orders=(2, 2)))
    np.testing.assert_array_equal(result.field(discontinuous='continuity', **_UNIFORM_AT).values,
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
            x=X, y=Y, z=[50.0], discontinuous='continuity').values) ** 2))(0.0)
    assert refused.value.code == 'field.discontinuous'
    jax.grad(lambda t: jnp.sum(jnp.abs(prepared.run(problem(t)).field(x=X, y=Y, z=[50.0]).values) ** 2))(0.0)


@pytest.mark.parametrize('sign_x, sign_y', [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)])
def test_counts_on_a_negative_side_put_every_component_on_one_grid(sign_x, sign_y):
    """A count runs down the cell on a negative side. Measured before: E_x and E_y on the other grid, 0.6 away."""
    numerics = rcwa.Numerics(orders=(4, 3))
    z = [50.0, 130.0]
    negative = rcwa.run(_raster_on(((sign_x * 400.0, 0.0), (0.0, sign_y * 360.0))), numerics=numerics)
    positive = rcwa.run(_raster_on(((400.0, 0.0), (0.0, 360.0))), numerics=numerics)
    x, y = np.linspace(-0.5, 0.5, 9) * sign_x * 400.0, np.linspace(-0.5, 0.5, 5) * sign_y * 360.0
    expected = positive.field(x=x, y=y, z=z, discontinuous='continuity').values
    np.testing.assert_allclose(negative.field(x=9, y=5, z=z, discontinuous='continuity').values, expected,
                               rtol=0, atol=1e-11)
