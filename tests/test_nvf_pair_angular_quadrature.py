"""Independent triangle moments and complete responses with split Gauss orders."""
from math import factorial
import numpy as np
import pytest
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit._core.conforming.pair_corner_fields import PairCornerFields


@pytest.mark.parametrize('option',(True,False,0,11,12.5,'24'))
def test_angular_quadrature_requires_an_explicit_valid_integer(option):
    with pytest.raises(ValueError,match='angular_quadrature'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
                                          angular_quadrature=option)


@pytest.mark.parametrize('exponents',((.6,.8),(.55,.6,.8,.95)))
@pytest.mark.parametrize('angular',(12,24))
def test_separate_orders_integrate_exact_triangle_moments(exponents,angular):
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),0,degree=3)
    fields=PairCornerFields(obj,exponents,junctions=True,angular_order=4,
                            angular_quadrature=angular)
    for tri in obj.data['triangles']:
        moments=np.zeros((7,7))
        for bary,weight in fields.quadrature(tri,80):
            assert len(weight)==80*angular
            for i in range(7):
                for j in range(7-i):moments[i,j]+=weight@(bary[:,1]**i*bary[:,2]**j)
        for i in range(7):
            for j in range(7-i):
                expected=factorial(i)*factorial(j)/factorial(i+j+2)
                assert abs(moments[i,j]-expected)<2e-13


def test_explicit_equal_orders_preserve_default_field_quadrature():
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),0,degree=3)
    default=PairCornerFields(obj,junctions=True)
    explicit=PairCornerFields(obj,junctions=True,angular_quadrature=48)
    for tri in obj.data['triangles']:
        for left,right in zip(default.quadrature(tri,48),explicit.quadrature(tri,48),strict=True):
            for a,b in zip(left,right):assert a.tobytes()==b.tobytes()


@pytest.mark.parametrize('gap',(0.,.02))
def test_independent_angular_orders_preserve_uniform_material_and_direction(gap):
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=3,quadrature=48,junctions=True,mesh='symmetric',angular_quadrature=12)
    value,tangent,metadata=cell.closure(gap=gap,basis=np.eye(2),widths=(3/7,3/7),
        height=3/7,epsilon=(3.,3.),background=3.,epsilon_tangent=(5.,5.),background_tangent=5.)
    expected=np.array([3.,3.,1/3,0.,0.])[:,None,None,None]
    derivative=np.array([5.,5.,-5/9,0.,0.])[:,None,None,None]
    np.testing.assert_allclose(value,expected,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,derivative,atol=3e-9,rtol=0)
    assert metadata['angular_integration_quadrature']==12
    assert metadata['integration_quadrature']==48
