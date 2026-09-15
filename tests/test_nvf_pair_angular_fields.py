"""Nested angular powers in the complete corner and junction field family."""
from functools import lru_cache
import numpy as np
import pytest
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit._core.conforming.pair_corner_fields import PairCornerFields


@lru_cache(None)
def closure(gap=.003, quadrature=32, angular_order=2, epsilon=(4., 9.),
            epsilon_tangent=(0., 0.), background_tangent=0., gap_tangent=1., length=1.):
    cell = CartesianPairCornerGalerkinGapCell(
        CartesianPairOpeningPlan(0, 0), 1, degree=4, quadrature=quadrature,
        angular_order=angular_order, junctions=True, coercivity_angle=1.53)
    result = cell.closure(
        basis=np.eye(2)*length, widths=np.array([.3, .4])*length,
        height=.45*length, offset=np.array([.01, -.03])*length,
        gap=gap*length, epsilon=epsilon, epsilon_tangent=epsilon_tangent,
        background_tangent=background_tangent, gap_tangent=gap_tangent)
    assert max(v for k, v in result[2].items()
               if k.endswith('residual') and v is not None) < 1e-10
    assert result[2]['angular_order'] == angular_order
    return np.asarray(result[0]), np.asarray(result[1])


@pytest.mark.parametrize('angular_order', (True, 1., -1, 5, None))
def test_unsupported_angular_orders_are_rejected(angular_order):
    with pytest.raises(ValueError, match='angular order'):
        CartesianPairCornerGalerkinGapCell(
            CartesianPairOpeningPlan(0, 0), angular_order=angular_order)


@pytest.mark.parametrize('degree', (3, 4))
@pytest.mark.parametrize('angular_order', (1, 2, 3, 4))
@pytest.mark.parametrize('junctions', (False, True))
def test_angular_powers_and_gradients_on_every_triangle(degree, angular_order, junctions):
    obj = CartesianPairPolynomialGalerkinGapCell(
        CartesianPairOpeningPlan(0, 1), 1, degree=degree)
    fields = PairCornerFields(obj, junctions=junctions, angular_order=angular_order)
    jac = np.array([[.7, .1], [.05, .6]])
    direction = np.array([[.13, -.04], [.01, -.03]])
    inverse = np.linalg.inv(jac)
    dinverse = -inverse@direction@inverse
    bary = np.array([[.2, .3, .5], [.61, .29, .1]])
    dbary = np.array([[.04, -.02, -.02], [-.06, .03, .03]])
    for tri in obj.data['triangles']:
        value, gradient, tangent = fields.sample(tri, bary, inverse, dinverse, dbary)
        hv = bary@fields.hats[tri[:3]]
        first = len(fields.powers)+len(fields.bubbles)
        for index, (c, n, m, alpha, j) in enumerate(fields.angular, first):
            h = hv[:, c]
            cutoff, beta = degree-3-j, alpha-2-j
            coefficients = [] if cutoff < 0 else [1.]
            for k in range(1, cutoff+1):
                coefficients.append(coefficients[-1]*(k-1-beta)/k)
            remainder = (1-h)**beta-sum(x*h**k for k, x in enumerate(coefficients))
            expected = h*hv[:, n]*hv[:, m]*(hv[:, n]-hv[:, m])**j*remainder
            np.testing.assert_allclose(value[:, index], expected, atol=3e-13, rtol=0)
        step = 1e-4
        for axis in range(2):
            dref = inverse[:, axis]
            move = np.r_[-dref.sum(), dref]
            values = [fields.sample(tri, bary+k*step*move, inverse, np.zeros((2, 2)))[0]
                      for k in (-2, -1, 1, 2)]
            difference = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
            np.testing.assert_allclose(gradient[:, axis], difference, atol=3e-9, rtol=0)
        values = [fields.sample(tri, bary+k*step*dbary,
                                np.linalg.inv(jac+k*step*direction), np.zeros((2, 2)))[1]
                  for k in (-2, -1, 1, 2)]
        difference = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(tangent, difference, atol=3e-9, rtol=0)


@pytest.mark.parametrize('angular_order', (2,4))
def test_complete_angular_quadrature_and_two_finite_gap_variations(angular_order):
    value, tangent = closure(angular_order=angular_order)
    refined, drefined = closure(quadrature=40,angular_order=angular_order)
    np.testing.assert_allclose(value, refined, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, drefined, atol=3e-9, rtol=0)
    for step in (3e-5, 1.5e-5):
        values = [closure(gap=.003+k*step,angular_order=angular_order)[0]
                  for k in (-2, -1, 1, 2)]
        difference = (values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(tangent, difference, atol=3e-9, rtol=0)


def test_nested_angular_contact_spaces_reduce_dielectric_energy():
    previous = None
    for angular_order in (0, 1, 2, 3, 4):
        xx, yy, zz, xy, yx = closure(gap=0., angular_order=angular_order)[0][:, 0]
        tensor = np.block([[xx, -xy], [-yx, yy]])
        np.testing.assert_allclose(tensor, tensor.conj().T, atol=3e-10, rtol=0)
        if previous is not None:
            difference = previous-tensor
            assert np.linalg.eigvalsh((difference+difference.conj().T)/2)[0] > -3e-10
        previous = tensor


@pytest.mark.parametrize('gap', (0., 1e-7))
def test_angular_uniform_medium_is_exact(gap):
    value, tangent = closure(gap=gap, epsilon=(1., 1.))
    expected = np.zeros_like(value)
    expected[0, 0] = expected[1, 0] = expected[2, 0] = np.eye(9)
    np.testing.assert_allclose(value, expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(tangent, 0., atol=3e-9, rtol=0)


def test_angular_material_linearity_and_length_units():
    value, gap = closure()
    scaled, dscaled = closure(length=700.)
    np.testing.assert_allclose(scaled, value, atol=3e-10, rtol=0)
    np.testing.assert_allclose(700*dscaled, gap, atol=3e-9, rtol=0)
    direction = dict(epsilon_tangent=(.3+.2j, -.4+.1j), background_tangent=.2+.3j)
    _, material = closure(gap_tangent=0., **direction)
    imaginary = dict(epsilon_tangent=tuple(1j*v for v in direction['epsilon_tangent']),
                     background_tangent=1j*direction['background_tangent'])
    _, imaterial = closure(gap_tangent=0., **imaginary)
    _, together = closure(**direction)
    np.testing.assert_allclose(imaterial, 1j*material, atol=3e-9, rtol=0)
    np.testing.assert_allclose(together, gap+material, atol=3e-9, rtol=0)


def test_complete_angular_metal_family_approaches_its_contact_value_and_jvp():
    cell = CartesianPairCornerGalerkinGapCell(
        CartesianPairOpeningPlan(0, 0), 1, degree=3, quadrature=32,
        angular_order=2, junctions=True, coercivity_angle=1.53)
    settings = dict(basis=np.diag([700., 700.]), widths=(300., 300.),
                    height=300., epsilon=(12.25, -10+1j))
    value0, tangent0, _ = cell.closure(gap=0., **settings)
    errors = []
    for gap in (1e-3, 1e-7, 1e-9):
        value, tangent, meta = cell.closure(gap=gap, **settings)
        assert max(v for k, v in meta.items() if k.endswith('residual') and v is not None) < 1e-10
        assert np.linalg.norm(np.asarray(value)-value0) < 2*gap*np.linalg.norm(tangent0)
        errors.append(np.linalg.norm(np.asarray(tangent)-tangent0))
    assert errors[2] < errors[1] < errors[0]
