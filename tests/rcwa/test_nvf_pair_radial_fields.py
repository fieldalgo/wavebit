"""Independent exact-power controls for expanded radial fields."""
from decimal import Decimal, localcontext
from functools import lru_cache
from math import factorial

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit.rcwa._engine.conforming.pair_corner_fields import PairCornerFields
from wavebit.rcwa._engine.conforming.pair_radial_powers import normalized_power_remainder

pytestmark = pytest.mark.research


def test_exact_normalized_radial_remainders():
 maximum=0.;count=0
 for prefix in (1,2,3,4):
     nodes=(.55,.6,.8,.95)[:prefix]
     for k in (1,2,3,4):
         r=max(0,4-k)
         for s in (1e-90,1e-20,.01,.5,.9,1-1e-12):
             h=1-s;actual=normalized_power_remainder(np.array([s]),np.array([h]),k,r,nodes)
             # Exact divided differences of the exponential at the combined
             # integer interpolation nodes and real exponent nodes.
             with localcontext() as context:
                 context.prec=180;sd=Decimal.from_float(s)
                 nd=[Decimal(i) for i in range(k,k+r)]+[Decimal(str(a)) for a in nodes]
                 from math import factorial
                 for derivative in range(3):
                     expected=Decimal(0)
                     for i,a in enumerate(nd):
                         weight=Decimal((-1)**r*factorial(r))
                         for j,b in enumerate(nd):
                             if i!=j:weight/=a-b
                         for j in range(derivative):weight*=a-j
                         expected+=weight*(sd.ln()*(a-derivative)).exp()
                     expected=float(expected)
                     error=abs(actual[derivative][0]-expected)/max(abs(expected),1e-280)
                     maximum=max(maximum,error);assert error<3e-12,(k,r,nodes,s,derivative,error)
                     count+=1


@pytest.mark.parametrize('exponents', ((.5,), (.55,.55), (True,), ('0.55',)))
def test_unsupported_radial_exponents_are_rejected(exponents):
    with pytest.raises(ValueError,match='corner exponents'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0,0),exponents=exponents)


def test_expanded_radial_quadrature_has_a_concrete_lower_bound():
    with pytest.raises(ValueError,match='quadrature at least 48'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0,0),quadrature=32,exponents=(.55,))


def direct_values(fields,tri,bary,degree):
    hv=bary@fields.hats[tri[:3]]
    rows=[(c,(n,),0,a) for c,n,a in fields.powers]
    rows += [(c,(n,m),0,a) for c,n,m,a in fields.bubbles]
    rows += [(c,(n,m),j,a) for c,n,m,a,j in fields.angular]
    groups={};result=np.zeros((len(bary),len(rows)))
    for index,(c,ns,angular,alpha) in enumerate(rows):
        nodes=groups.setdefault((c,ns,angular),[]);nodes.append(alpha)
        h=hv[:,c];s=1-h;k=len(ns)+angular;r=max(0,degree-k)
        product=np.prod(hv[:,ns],axis=1)
        if angular:product*=(hv[:,ns[0]]-hv[:,ns[1]])**angular
        remainder=np.zeros_like(h)
        for a in nodes:
            weight=1.
            for b in nodes:
                if a!=b:weight/=a-b
            coefficient=1.;polynomial=np.zeros_like(h)
            for j in range(r):
                polynomial+=coefficient*h**j
                coefficient*=(k+j-a)/(j+1)
            remainder+=weight*(s**(a-k)-polynomial)/coefficient
        scale=(-1.)**(len(nodes)-1)*factorial(r+len(nodes)-1)/factorial(r)
        result[:,index]=scale*h*product*remainder
    return result


@pytest.mark.parametrize('degree',(3,4))
@pytest.mark.parametrize('junctions',(False,True))
@pytest.mark.parametrize('exponents',((.55,.6,.8,.95),(.95,.8,.6,.55)))
def test_radial_span_and_both_field_variations(degree,junctions,exponents):
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0,0),1,degree=degree)
    fields=PairCornerFields(obj,exponents,junctions=junctions,angular_order=2)
    jac=np.array([[.7,.1],[.05,.6]]);direction=np.array([[.13,-.04],[.01,-.03]])
    inverse=np.linalg.inv(jac);dinverse=-inverse@direction@inverse
    bary=np.array([[.2,.3,.5],[.61,.29,.1]])
    dbary=np.array([[.04,-.02,-.02],[-.06,.03,.03]])
    step=1e-4
    for tri in obj.data['triangles']:
        value,gradient,tangent=fields.sample(tri,bary,inverse,dinverse,dbary)
        np.testing.assert_allclose(value,direct_values(fields,tri,bary,degree),atol=3e-10,rtol=0)
        for axis in range(2):
            dref=inverse[:,axis];move=np.r_[-dref.sum(),dref]
            values=[fields.sample(tri,bary+i*step*move,inverse,np.zeros((2,2)))[0] for i in (-2,-1,1,2)]
            difference=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
            np.testing.assert_allclose(gradient[:,axis],difference,atol=3e-9,rtol=0)
        values=[fields.sample(tri,bary+i*step*dbary,np.linalg.inv(jac+i*step*direction),np.zeros((2,2)))[1] for i in (-2,-1,1,2)]
        difference=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(tangent,difference,atol=3e-9,rtol=0)


@lru_cache(None)
def radial_closure(gap=.003,epsilon=(4.,9.),length=1.,junctions=True,
                   exponents=(.55,.6,.8,.95),gap_tangent=1.,
                   epsilon_tangent=(0.,0.),background_tangent=0.):
    cell=CartesianPairCornerGalerkinGapCell(
        CartesianPairOpeningPlan(0,0),0,degree=3,quadrature=48,
        exponents=exponents,junctions=junctions,coercivity_angle=1.53)
    value,tangent,meta=cell.closure(
        gap=gap*length,basis=np.eye(2)*length,widths=np.array([.3,.4])*length,
        height=.45*length,offset=np.array([.01,-.03])*length,epsilon=epsilon,
        gap_tangent=gap_tangent,epsilon_tangent=epsilon_tangent,
        background_tangent=background_tangent)
    assert meta['radial_root_order']==20
    assert meta['integration_quadrature']==80
    assert max(value for key,value in meta.items()
               if key.endswith('residual') and value is not None)<1e-10
    return np.asarray(value),np.asarray(tangent)


def test_radial_whole_material_directions_and_length_units():
    value,gap=radial_closure()
    scaled,ds=radial_closure(length=700.)
    np.testing.assert_allclose(scaled,value,atol=3e-10,rtol=0)
    np.testing.assert_allclose(700*ds,gap,atol=3e-9,rtol=0)
    direction=dict(epsilon_tangent=(.3+.2j,-.4+.1j),background_tangent=.2+.3j)
    _,material=radial_closure(gap_tangent=0.,**direction)
    imaginary=dict(epsilon_tangent=tuple(1j*a for a in direction['epsilon_tangent']),
                   background_tangent=1j*direction['background_tangent'])
    _,imaterial=radial_closure(gap_tangent=0.,**imaginary)
    _,together=radial_closure(**direction)
    np.testing.assert_allclose(imaterial,1j*material,atol=3e-9,rtol=0)
    np.testing.assert_allclose(together,gap+material,atol=3e-9,rtol=0)


@pytest.mark.parametrize('gap',(0.,1e-7))
def test_radial_uniform_medium_at_contact_and_positive_gap(gap):
    value,tangent=radial_closure(gap=gap,epsilon=(1.,1.))
    expected=np.zeros_like(value)
    expected[0,0]=expected[1,0]=expected[2,0]=np.eye(1)
    np.testing.assert_allclose(value,expected,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,0.,atol=3e-9,rtol=0)


def test_radial_junction_space_cannot_increase_positive_dielectric_energy():
    def tensor(value):
        xx,yy,zz,xy,yx=value[:,0]
        return np.block([[xx,-xy],[-yx,yy]])
    difference=tensor(radial_closure(junctions=False)[0])-tensor(radial_closure()[0])
    np.testing.assert_allclose(difference,difference.conj().T,atol=3e-10,rtol=0)
    assert np.linalg.eigvalsh((difference+difference.conj().T)/2)[0]>-3e-10


def test_radial_metal_family_reaches_the_same_contact_value_and_derivative():
    settings=dict(epsilon=(12.25,-10+1j),length=700.)
    value0,tangent0=radial_closure(gap=0.,**settings)
    errors=[]
    for gap in (1e-3,1e-7,1e-9):
        value,tangent=radial_closure(gap=gap/700,**settings)
        assert np.linalg.norm(value-value0)<2*gap*np.linalg.norm(tangent0)
        errors.append(np.linalg.norm(tangent-tangent0))
    assert errors[2]<errors[1]<errors[0]


def test_radial_coordinate_order_preserves_the_complete_metal_response():
    settings=dict(gap=0.,epsilon=(12.25,-10+1j),length=700.)
    reference=radial_closure(**settings)
    reversed_order=radial_closure(exponents=(.95,.8,.6,.55),**settings)
    for value,expected,tolerance in zip(reversed_order,reference,(3e-10,3e-9)):
        np.testing.assert_allclose(value,expected,atol=tolerance,rtol=0)


def test_every_declared_radial_subset_has_exact_signed_root_factors():
    from fractions import Fraction as F
    from itertools import combinations
    from wavebit.rcwa._engine.conforming.pair_radial_powers import factored_coefficients
    allowed=(.55,.6,.8,.95)
    for count in range(1,5):
        for subset in combinations(allowed,count):
            for degree in (3,4):
                for k in (1,2,3,4):
                    r=max(0,degree-k)
                    coefficients=factored_coefficients(k,r,subset)
                    np.testing.assert_array_equal(coefficients,
                        factored_coefficients(k,r,subset[::-1]))
                    assert np.all((-1)**(count-1)*coefficients>=0)
                    assert np.any(coefficients!=0)
                    nodes=tuple(F(i) for i in range(k,k+r))+tuple(F(str(a)) for a in subset)
                    for root in (F(1,2),F(3,4)):
                        s=root**20
                        actual=normalized_power_remainder(
                            np.array([float(s)]),np.array([float(1-s)]),k,r,subset)
                        for derivative in range(3):
                            expected=F(0)
                            for i,a in enumerate(nodes):
                                weight=F((-1)**r*factorial(r))
                                for j,b in enumerate(nodes):
                                    if i!=j:weight/=a-b
                                for j in range(derivative):weight*=a-j
                                expected+=weight*root**int(20*(a-derivative))
                            np.testing.assert_allclose(actual[derivative],float(expected),
                                                       atol=0,rtol=3e-12)
