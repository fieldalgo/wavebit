"""Independent Decimal fields for the unchanged contact normal potential bank."""
from decimal import Decimal as Z,localcontext
from math import factorial,prod
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_normal_precision import trace_values,polynomial_gradients
from wavebit._core.conforming.polynomial_pair_trace import PolynomialPairTrace
from wavebit._core.conforming.pair_corner_precision import D
from wavebit._core.conforming.pair_contact_quadrature import Exact
from wavebit._core.conforming.triangle_polynomials import multi_indices
from wavebit._core.conforming.pair_contact_normal_precision import normal_map
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.pair_symmetric_mesh import SymmetricPairPolynomialCell
from wavebit._core.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
from test_nvf_contact_fourier_precision import exponential

pytestmark = pytest.mark.research

def dec(x):return Z.from_float(float(x))


def test_complete_contact_normals_retain_transverse_trace_span():
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(0,1),
        reference=(12.25,-10+1j,1.),degree=5,quadrature=100,
        angular_quadrature=48,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)
    assert metadata['transmission_directions']==20
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual'):
        assert metadata[key]<1e-10
    reflection=np.array([2,1,0])
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(response,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*slot[:,reflection][:,:,reflection],atol=tolerance,rtol=0)

def chebyshev_monomials(n):
    result=[[1]]
    if n:result.append([0,1])
    for k in range(2,n+1):
        coefficients=[0]*(k+1)
        for power,value in enumerate(result[-1]):coefficients[power+1]+=2*value
        for power,value in enumerate(result[-2]):coefficients[power]-=value
        result.append(coefficients)
    return result


@pytest.mark.parametrize('refinement',(0,1))
def test_normal_maps_match_independent_positive_gap_limits(refinement):
    obj=SymmetricPairPolynomialCell(CartesianPairOpeningPlan(0,refinement),(1,1),degree=5)
    geometry=dict(basis=np.diag([1.2,.9]),widths=np.array([.3,.4]),
                  height=np.array(.45),offset=np.array([-.6,-.45]))
    _,_,points,velocity=obj._chart_geometry(0.,geometry)
    with localcontext() as context:
        context.prec=100
        step=Z('1e-35')
        for tri,thin in zip(obj.data['triangles'],obj.data['thin']):
            p,v=points[tri[:3]],velocity[tri[:3]]
            observed=normal_map(p,v,thin)
            pp=[[dec(x) for x in row] for row in p]
            vv=[[dec(x) for x in row] for row in v]
            def inverse(gap):
                q=[[pp[i][a]+gap*vv[i][a] for a in range(2)] for i in range(3)]
                j=[[q[k+1][a]-q[0][a] for k in range(2)] for a in range(2)]
                determinant=j[0][0]*j[1][1]-j[0][1]*j[1][0]
                return [[j[1][1]/determinant,-j[0][1]/determinant],
                        [-j[1][0]/determinant,j[0][0]/determinant]]
            if thin:
                coarse,fine=inverse(step),inverse(step/2)
                # Extrapolate s*inv(J(s)); no common-gradient annihilation
                # is imposed on a disappearing normal potential.
                expected=[[step*(fine[i][j]-coarse[i][j]) for j in range(2)] for i in range(2)]
            else:
                expected=inverse(Z(0))
            for i in range(2):
                for j in range(2):
                    actual=dec(observed.hi[i,j])+dec(observed.lo[i,j])
                    assert abs(actual-expected[i][j])/max(Z(1),abs(expected[i][j]))<Z('3e-28')

@pytest.mark.parametrize('degree',(3,4,5,6,7,8))
@pytest.mark.parametrize('order',(0,1,4))
def test_complete_stored_trace_matches_decimal_monomials(degree,order):
    with localcontext() as context:
        context.prec=100
        worst=0.
        trace=PolynomialPairTrace(order,degree);geometry=trace._geometry(0.,(700.,700.),300.,0.)
        y=D(np.array([-.21428571428571427,-.143,-.013,0.,.059,.192,.21428571428571427]),np.array([2.**-60,-2.**-60,2.**-65,0.,-2.**-60,2.**-60,-2.**-60]))
        result=trace_values(trace,geometry,y);data=geometry.get('polynomial_frame')
        for row in range(len(y)):
         actual_y=dec(y.hi[row])+dec(y.lo[row])
         if data is None:expected=[(1/dec(geometry['transform'][0,0]),Z(0))]
         else:
          frequency=dec(data['frequency']);cosine,sine=exponential((Z(0),frequency*actual_y));u=sine/dec(data['amplitude']);du=frequency*cosine/dec(data['amplitude']);raw=[]
          for index in range(2*order+1):
           if index==0:coefficients=data['residual']
           elif index==1 and data['odd'] is not None:coefficients=data['odd']
           else:coefficients=np.eye(1,index+1,index).ravel()
           monomials=[Z(0)]*len(coefficients)
           for coefficient,terms in zip(coefficients,chebyshev_monomials(len(coefficients)-1)):
            for power,term in enumerate(terms):monomials[power]+=dec(coefficient)*term
           value=sum(c*(u**k if k else Z(1)) for k,c in enumerate(monomials))
           derivative=sum(k*c*(u**(k-1) if k>1 else Z(1)) for k,c in enumerate(monomials) if k)*du
           if index%2:value,derivative=cosine*value,-frequency*sine*value+cosine*derivative
           raw.append((value,derivative))
          expected=[tuple(sum(raw[j][component]*dec(data['frame'][j,k]) for j in range(len(raw))) for component in range(2)) for k in range(2*order+1)]
         for column,pair in enumerate(expected):
          for component,target in enumerate(pair):
           value=result[component];observed=dec(value.hi[row,column])+dec(value.lo[row,column]);error=abs(observed-target)/max(Z(1),abs(target));worst=max(worst,float(error));assert error<Z('3e-28'),(degree,order,row,column,component,error)


@pytest.mark.parametrize('degree',(3,4,5,6,7,8))
def test_complete_nodal_gradient_matches_decimal_potential(degree):
    rng=np.random.default_rng(742)
    with localcontext() as context:
        context.prec=100
        worst=0.
        bary=np.array([[.21,.32,.47],[2.**-60,.3,.7],[.5,.5-2.**-40,2.**-40]])
        # Exact simplex values: supply the third word as the complete complement.
        points=D(bary);points[:,2]=1-points[:,0]-points[:,1]
        rule=Exact(points);coeff=rng.normal(size=(len(multi_indices(degree)),4));inverse=D(np.array([[1.3,.1],[-.2,.9]]),np.array([[2.**-60,0.],[-2.**-61,2.**-59]]))
        result=polynomial_gradients(degree,rule,coeff,inverse)
        for row in range(len(bary)):
         values=[dec(points.hi[row,k])+dec(points.lo[row,k]) for k in range(3)]
         for column in range(4):
          ref=[Z(0),Z(0)]
          for alpha,c in zip(multi_indices(degree),coeff[:,column]):
           scale=dec(c)*factorial(degree)/prod(factorial(i) for i in alpha)
           for coordinate,power in enumerate(alpha):
            if not power:continue
            monomial=scale*power
            for k,n in enumerate(alpha):monomial*=values[k]**(n-(k==coordinate))
            for axis,sign in enumerate(((-1,-1),(1,0),(0,1))[coordinate]):ref[axis]+=sign*monomial
          for component in range(2):
           target=sum(ref[k]*(dec(inverse.hi[k,component])+dec(inverse.lo[k,component])) for k in range(2));observed=dec(result.hi[row,component,column])+dec(result.lo[row,component,column]);error=abs(observed-target)/max(Z(1),abs(target));worst=max(worst,float(error));assert error<Z('3e-28'),(degree,row,column,component,error)
