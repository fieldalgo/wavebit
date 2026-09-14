"""Real corner fields, complete pair constraints and physical variations."""
from functools import lru_cache
import numpy as np
import pytest
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_seam_galerkin import CartesianPairSeamGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit._core.conforming.pair_corner_fields import PairCornerFields


PLAN = CartesianPairOpeningPlan(0, 0)
GEOMETRY = dict(basis=np.eye(2), widths=(.3, .4), height=.45, offset=(.01, -.03))


@lru_cache(None)
def closure(gap=.003, epsilon=(4., 9.), background=1., gap_tangent=1.,
            epsilon_tangent=(0., 0.), background_tangent=0., exponents=(.6, .8),
            quadrature=32, degree=3, length=1.):
    cell = CartesianPairCornerGalerkinGapCell(
        PLAN, 1, degree=degree, quadrature=quadrature,
        exponents=exponents, coercivity_angle=1.53)
    geometry = {key: np.asarray(value)*length for key, value in GEOMETRY.items()}
    result = cell.closure(gap=gap*length, epsilon=epsilon, background=background,
                         gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                         background_tangent=background_tangent, **geometry)
    assert max(value for key, value in result[2].items()
               if key.endswith('residual') and value is not None) < 1e-10
    return np.asarray(result[0]), np.asarray(result[1]), result[2]


@pytest.mark.parametrize('degree', (3, 4))
def test_corner_field_gradients_and_moving_gradients_match_finite_variations(degree):
    obj = CartesianPairPolynomialGalerkinGapCell(PLAN, 1, degree=degree)
    fields = PairCornerFields(obj)
    jacobian = np.array([[.7, .1], [.05, .6]])
    direction = np.array([[.13, -.04], [.01, -.03]])
    inverse = np.linalg.inv(jacobian)
    dinverse = -inverse@direction@inverse
    bary = np.array([[.2, .3, .5], [.65, .25, .1]])
    dbary = np.array([[.04, -.02, -.02], [-.06, .03, .03]])
    triangle = next(tri for tri, thin in zip(obj.data['triangles'], obj.data['thin'])
                    if not thin and np.any(fields.hats[tri[:3]][:, [4, 6, 7, 9]]))
    value, gradient, tangent = fields.sample(triangle, bary, inverse, dinverse, dbary)
    # Independently evaluate the defining powers minus their short Taylor
    # polynomials away from the singular corner, without fifth-root factors.
    hats = fields.hats[triangle[:3]]
    hv = bary@hats
    gh = (np.array([[-1., -1.], [1., 0.], [0., 1.]])@inverse).T@hats
    definitions = [(c, (n,), a-1, degree-2) for c, n, a in fields.powers]
    definitions += [(c, (n, m), a-2, degree-3) for c, n, m, a in fields.bubbles]
    for column, (corner, neighbors, beta, cutoff) in enumerate(definitions):
        if not np.any(hats[:, corner]):
            np.testing.assert_array_equal(value[:, column], 0.)
            continue
        h = hv[:, corner]
        coefficients = [1.]
        for j in range(1, cutoff+1):
            coefficients.append(coefficients[-1]*(j-1-beta)/j)
        remainder = (1-h)**beta-sum(c*h**j for j, c in enumerate(coefficients))
        derivative = -beta*(1-h)**(beta-1)-sum(
            j*c*h**(j-1) for j, c in enumerate(coefficients) if j)
        product = np.prod(hv[:, neighbors], axis=1)
        gp = sum(np.prod(hv[:, tuple(n for n in neighbors if n != other)], axis=1)[:, None]
                 *gh[:, other] for other in neighbors)
        expected_value = h*product*remainder
        expected_gradient = product[:, None]*(remainder+h*derivative)[:, None]*gh[:, corner]
        expected_gradient += (h*remainder)[:, None]*gp
        np.testing.assert_allclose(value[:, column], expected_value, atol=3e-13, rtol=0)
        np.testing.assert_allclose(gradient[:, :, column], expected_gradient, atol=3e-13, rtol=0)
    h = 1e-4
    for axis in range(2):
        dref = inverse[:, axis]
        move = np.r_[-dref.sum(), dref]
        values = [fields.sample(triangle, bary+j*h*move, inverse, np.zeros((2, 2)))[0]
                  for j in (-2, -1, 1, 2)]
        expected = (values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
        np.testing.assert_allclose(gradient[:, axis], expected, atol=3e-9, rtol=0)
    values = [fields.sample(triangle, bary+j*h*dbary,
                            np.linalg.inv(jacobian+j*h*direction), np.zeros((2, 2)))[1]
              for j in (-2, -1, 1, 2)]
    expected = (values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
    np.testing.assert_allclose(tangent, expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(degree=2), dict(degree=True), dict(degree=3.),
                                   dict(quadrature=23), dict(quadrature=True),
                                   dict(exponents=(.5,)), dict(exponents=(.6, .6)),
                                   dict(exponents=(True,)), dict(exponents=(complex(.6),)),
                                   dict(exponents=(np.nan,))])
def test_unsupported_corner_spaces_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CartesianPairCornerGalerkinGapCell(PLAN, **kwargs)


@pytest.mark.parametrize('gap', (0., .003))
def test_empty_enrichment_preserves_original_whole_seam_value_and_jvp(gap):
    old = CartesianPairSeamGalerkinGapCell(
        PLAN, 1, degree=3, quadrature=48 if gap == 0 else 32,
        coercivity_angle=1.53, contact_solver='projected').closure(
            gap=gap, epsilon=(4., 9.), **GEOMETRY)
    value, tangent, _ = closure(gap=gap, exponents=())
    np.testing.assert_allclose(value, old[0], atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, old[1], atol=3e-9, rtol=0)


def test_uniform_material_has_no_corner_or_gap_correction():
    value, tangent, _ = closure(gap=1e-7, epsilon=(1., 1.))
    expected = np.zeros_like(value)
    expected[0, 0] = expected[1, 0] = expected[2, 0] = np.eye(9)
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, 0., atol=3e-9, rtol=0)


def test_material_directions_are_complex_linear_and_add_to_gap_direction():
    _, gap, _ = closure()
    directions = dict(epsilon_tangent=(.3+.2j, -.4+.1j), background_tangent=.2+.3j)
    _, material, _ = closure(gap_tangent=0., **directions)
    imaginary = dict(epsilon_tangent=tuple(1j*v for v in directions['epsilon_tangent']),
                     background_tangent=1j*directions['background_tangent'])
    _, imaterial, _ = closure(gap_tangent=0., **imaginary)
    _, together, _ = closure(**directions)
    np.testing.assert_allclose(imaterial, 1j*material, atol=3e-9, rtol=0)
    np.testing.assert_allclose(together, gap+material, atol=3e-9, rtol=0)


def test_length_units_preserve_value_and_inverse_length_gap_derivative():
    value, tangent, _ = closure()
    scaled, dscaled, _ = closure(length=700.)
    np.testing.assert_allclose(scaled, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(700*dscaled, tangent, atol=3e-9, rtol=0)


@pytest.mark.parametrize('degree', (3, 4))
def test_quadrature_and_independent_finite_gap_variations(degree):
    value, tangent, _ = closure(degree=degree)
    refined, drefined, _ = closure(quadrature=40, degree=degree)
    np.testing.assert_allclose(value, refined, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, drefined, atol=3e-9, rtol=0)
    for h in (3e-5, 1.5e-5):
        values = [closure(gap=.003+j*h, degree=degree)[0] for j in (-2, -1, 1, 2)]
        difference = (values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
        np.testing.assert_allclose(tangent, difference, atol=3e-9, rtol=0)


def test_metal_positive_values_and_jvps_approach_same_enriched_contact():
    cell = CartesianPairCornerGalerkinGapCell(PLAN, 1, quadrature=32, coercivity_angle=1.53)
    settings = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.,
                    epsilon=(12.25, -10+1j))
    value0, tangent0, _ = cell.closure(gap=0., **settings)
    errors = []
    for gap in (.001, 1e-7, 1e-9):
        value, tangent, meta = cell.closure(gap=gap, **settings)
        assert max(v for k, v in meta.items() if k.endswith('residual') and v is not None) < 1e-10
        assert np.linalg.norm(np.asarray(value)-value0) < 2*gap*np.linalg.norm(tangent0)
        errors.append(np.linalg.norm(np.asarray(tangent)-tangent0))
    assert errors[1] < .02*errors[0]
    assert errors[2] < .2*errors[1]


def test_independent_stationary_energy_rejects_corrupted_positive_jvp(monkeypatch):
    from wavebit._core.conforming import cartesian_pair_corner_galerkin as module
    original = module.solve_positive
    observed = []
    check = module._check_response
    def corrupt(*args, **kwargs):
        values = list(original(*args, **kwargs))
        values[5] = values[5]+.01*np.eye(len(values[5]))
        return tuple(values)
    def record(value, tangent, meta):
        observed.append(meta)
        return check(value, tangent, meta)
    monkeypatch.setattr(module, 'solve_positive', corrupt)
    monkeypatch.setattr(module, '_check_response', record)
    cell = CartesianPairCornerGalerkinGapCell(PLAN, 1, quadrature=32, coercivity_angle=1.53)
    with pytest.raises(ValueError, match='original/energy/physical guards'):
        cell.closure(gap=.003, epsilon=(4., 9.), **GEOMETRY)
    assert observed[0]['energy_derivative_residual'] > 1e-10


@pytest.mark.parametrize('gap', (.001, 1e-7, 1e-9))
def test_exact_clipped_partitions_preserve_area_and_zero_partition_variation(gap):
    from fractions import Fraction
    from wavebit._core.conforming.cartesian_pair_seam_galerkin import _geometry
    from wavebit._core.conforming.pair_corner_partition import partitions, det
    obj = CartesianPairPolynomialGalerkinGapCell(PLAN, 1, degree=3)
    settings = dict(GEOMETRY, gap=gap, epsilon=(4., 9.), background=1., gap_tangent=1.,
                    epsilon_tangent=(0., 0.), background_tangent=0.)
    base = obj._assemble(**settings)
    areas = {}
    changes = {}
    for tri, local, p, dp, polygon in partitions(obj, base, settings, _geometry(obj, settings), True):
        key = tuple(tri)
        for j in range(1, len(polygon)-1):
            vertices = [polygon[0], polygon[j], polygon[j+1]]
            b = [v[0] for v in vertices]
            db = [v[1] for v in vertices]
            area = det(b)
            assert area >= 0
            change = sum(det([db[k] if k == i else b[k] for k in range(3)]) for i in range(3))
            areas[key] = areas.get(key, Fraction(0))+area
            changes[key] = changes.get(key, Fraction(0))+change
    assert len(areas) == np.count_nonzero(~obj.data['thin'])
    assert all(area == 1 for area in areas.values())
    assert all(change == 0 for change in changes.values())


def test_nested_real_corner_space_cannot_increase_minimum_energy():
    def tensor(value):
        xx, yy, zz, xy, yx = value[:, 0]
        return np.block([[xx, -xy], [-yx, yy]])
    original = tensor(closure(exponents=())[0])
    enriched = tensor(closure()[0])
    np.testing.assert_allclose(enriched, enriched.conj().T, atol=3e-10, rtol=0)
    difference = original-enriched
    assert np.linalg.eigvalsh((difference+difference.conj().T)/2)[0] > -3e-10
