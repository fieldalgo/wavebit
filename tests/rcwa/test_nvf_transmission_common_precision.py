"""Independent high-degree transmission fields and complete physical controls."""
from decimal import Decimal as Z, localcontext
import numpy as np
from wavebit.rcwa._engine.conforming.pair_transmission_modes import TransmissionMode
from wavebit.rcwa._engine.conforming.pair_transmission_common_precision import D, local_mode
import pytest

def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[0],-a[1]
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def div(a,b):
 d=b[0]*b[0]+b[1]*b[1];return (a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d
def scalar(a):
 if hasattr(a,'hi') and hasattr(a,'lo'):return add(scalar(a.hi),scalar(a.lo))
 return Z.from_float(float(np.real(a))),Z.from_float(float(np.imag(a)))
def exponential(z):
 n=0
 while abs(z[0])+abs(z[1])>Z('.125'):z=(z[0]/2,z[1]/2);n+=1
 term=result=(Z(1),Z(0))
 for k in range(1,101):
  a,b=mul(term,z);term=(a/k,b/k);result=add(result,term)
 for _ in range(n):result=mul(result,result)
 return result
def atan(z):
 sign=1 if z>=0 else -1;z=abs(z)
 if z>1:return sign*(PI/2-atan(1/z))
 for _ in range(3):z=z/(1+(1+z*z).sqrt())
 term=total=z
 for k in range(1,81):term*=-z*z;total+=term/(2*k+1)
 return sign*8*total
def phase(x,y):
 if x==0:return PI/2 if y>0 else -PI/2
 a=atan(y/x)
 return a if x>0 else a+(PI if y>=0 else -PI)

@pytest.mark.research
@pytest.mark.parametrize('coefficients', ((12.25,1,1),(-10+1j,1,1),(12.25,-10+1j,1)))
@pytest.mark.parametrize('branch', range(4))
@pytest.mark.parametrize('quarter_turn', (0,1,2,-1))
def test_compensated_mode_matches_independent_cartesian_power(coefficients,branch,quarter_turn):
    global PI
    with localcontext() as context:

        context.prec=100
        PI=16*atan(Z(1)/5)-4*atan(Z(1)/239)
        errors=[]
        mode=TransmissionMode(coefficients,branch=branch,start=quarter_turn*np.pi/2)
        for sector in range(3):
            center=(sum(mode.angles[:sector])+sum(mode.angles[:sector+1]))/2
            for radius in (.2,1e-70):
                absolute=center+mode.start+.137
                x0,y0=radius*np.cos(absolute),radius*np.sin(absolute)
                x,y=Z.from_float(float(x0)),Z.from_float(float(y0));r=(x*x+y*y).sqrt();theta=phase(x,y)
                shift=round((center+mode.start-np.arctan2(y0,x0))/(2*np.pi))
                local=theta+(2*shift-Z(quarter_turn+sum((1,1,2)[:sector]))/2)*PI
                lam=scalar(mode.complete_exponent);u,q=map(scalar,mode.complete_states[sector]);qa=div(q,scalar(mode.complete_coefficients[sector]))
                one=(Z(1),Z(0));ii=(Z(0),Z(1));two=(Z(2),Z(0))
                A=div(add(u,neg(mul(ii,qa))),two);B=div(add(u,mul(ii,qa)),two)
                positive=mul(A,exponential(mul(lam,(r.ln(),local))));negative=mul(B,exponential(mul(lam,(r.ln(),-local))))
                expected=[add(positive,negative)]
                left,right=div(positive,(x,y)),div(negative,(x,-y))
                expected.extend([mul(lam,add(left,right)),mul(mul(ii,lam),add(left,neg(right)))])
                actual=local_mode(mode,D([x0]),D([y0]),sector,center)
                for a,b in zip(actual,expected):
                    real=Z.from_float(float(a.hi[0].real))+Z.from_float(float(a.lo[0].real))
                    imag=Z.from_float(float(a.hi[0].imag))+Z.from_float(float(a.lo[0].imag))
                    error=max(abs(real-b[0]),abs(imag-b[1]))/max(abs(b[0]),abs(b[1]),Z('1e-300'))
                    errors.append(float(error));assert error<Z('3e-28'),(coefficients,branch,sector,radius,error)


@pytest.mark.research
@pytest.mark.parametrize('degree', (5,6,7,8))
def test_high_degree_flat_reference_retains_uniform_tensor_and_joint_variation(degree):
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=(2.,2.,2.),degree=degree,angular_quadrature=12)
    epsilon=3.+.25j
    direction=.125+.0625j
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2),widths=(.25,.125),
        height=.5,offset=(-.5,-.5),epsilon=(epsilon,epsilon),background=epsilon,
        gap_tangent=.5,epsilon_tangent=(direction,direction),background_tangent=direction)
    assert metadata['transmission_directions']==0
    assert metadata['degree']==degree
    assert metadata['common_coordinate_precision']=='compensated'
    assert metadata['retained_verification']=='physical_field_integral'
    assert metadata['retained_residual']<1e-10
    assert np.isfinite(metadata['rounded_retained_defect'])
    expected=np.array([epsilon,epsilon,1/epsilon,0.,0.])[:,None,None,None]
    delta=np.array([direction,direction,-direction/epsilon**2,0.,0.])[:,None,None,None]
    np.testing.assert_allclose(value,expected,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,delta,atol=3e-9,rtol=0)


def test_high_degree_scope_retains_original_family_and_resource_limits():
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
    from wavebit.rcwa._engine.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
    plan=CartesianPairOpeningPlan(0)
    for family in (CartesianPairCornerGalerkinGapCell,CartesianPairPolynomialGalerkinGapCell):
        with pytest.raises(ValueError,match='degree'):
            family(plan,0,degree=5)
    for degree in (True,5.,2,9):
        with pytest.raises(ValueError,match='degree'):
            CartesianPairTransmissionGalerkinGapCell(plan,0,reference=(4.,9.,1.),degree=degree)
    with pytest.raises(ValueError,match='resolution'):
        CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0,2),0,
            reference=(4.,9.,1.),degree=6)
    for degree in (7,8):
        with pytest.raises(ValueError,match='resolution'):
            CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0,2),0,
                reference=(4.,9.,1.),degree=degree)


# Five complete independently verified states take about 30 minutes on the
# documented single-BLAS-thread reference run. Keep this one bounded slow
# integration control without reducing its field bank or quadrature.
@pytest.mark.research
@pytest.mark.timeout(3600)
def test_quintic_positive_gap_joint_jvp_matches_independent_values():
    from tests.rcwa.test_nvf_pair_transmission import GEOMETRY, REFERENCE
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        reference=REFERENCE,branches=2,degree=5,quadrature=100,angular_quadrature=32,
        coercivity_angle=1.53)
    epsilon=np.array([4.+.1j,9.+.2j]);direction=np.array([.1+.02j,.2+.03j])
    background,dbg=1.+.02j,.05+.01j
    def solve(step):
        return cell.closure(gap=.02+.7*step,**GEOMETRY,epsilon=epsilon+step*direction,
            background=background+step*dbg,gap_tangent=.7,
            epsilon_tangent=direction,background_tangent=dbg)
    value,tangent,metadata=solve(0.)
    assert metadata['degree']==5
    assert metadata['transmission_reference']==REFERENCE
    h=3e-5
    samples=[np.asarray(solve(k*h)[0]) for k in (-2,-1,1,2)]
    finite=(samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
    np.testing.assert_allclose(tangent,finite,atol=3e-9,rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('degree,refinement', ((5,0),(6,0),(5,1)))
def test_complete_common_gradients_match_independent_potential_differences(degree,refinement):
    from math import factorial, prod
    from tests.rcwa.test_nvf_pair_transmission import GEOMETRY, REFERENCE
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    from wavebit.rcwa._engine.conforming.pair_transmission_fields import PairTransmissionFields
    from wavebit.rcwa._engine.conforming.pair_transmission_common_precision import CommonCoordinates
    from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0,refinement),0,
        reference=REFERENCE,degree=degree,branches=2,angular_quadrature=32)
    fields=PairTransmissionFields(cell._base,GEOMETRY,REFERENCE,2,32)
    data=cell._base.data;nold=data['Cfull'].shape[1]-1
    mapping=np.random.default_rng(71).normal(size=(nold+fields.count,3))*[1.,32.,1/32]
    common=CommonCoordinates(fields,mapping)
    bary=np.array([[.23,.31,.46],[.52,.17,.31]])
    inverse=np.array([[1.25,.25],[.125,.875]])
    dinverse=np.array([[.03125,.0625],[-.015625,.03125]])
    for tri in data['triangles'][[0,7,16,-1]]:
        coefficients=data['Cfull'][data['periodic'][tri],1:]@mapping[:nold]
        def potential(b):
            bernstein=np.array([factorial(degree)/prod(factorial(int(a)) for a in alpha)
                *np.prod(b**np.asarray(alpha),axis=1) for alpha in multi_indices(degree)]).T
            value=fields.sample(tri,b,np.eye(2),np.zeros((2,2)))[0]
            return bernstein@coefficients+value@mapping[nold:]
        actual=common.sample(tri,bary,inverse,dinverse)
        h=1e-3
        for slot,matrix in enumerate((inverse,dinverse)):
            for axis in range(2):
                direction=np.r_[-sum(matrix[:,axis]),matrix[:,axis]]
                samples=[potential(bary+k*h*direction) for k in (-3,-2,-1,1,2,3)]
                finite=(-samples[0]+9*samples[1]-45*samples[2]+45*samples[3]-9*samples[4]+samples[5])/(60*h)
                value=actual[2*slot+axis]
                np.testing.assert_allclose(value.hi+value.lo,finite,atol=3e-9,rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('degree,background', ((5,1.),(6,1.),(6,1.+.0625j)),
                         ids=('quintic','sextic','sextic_complex_background'))
def test_high_degree_metal_fourier_projection_retains_original_energy_identity(degree,background):
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    # The flat reference isolates the complete polynomial/Fourier projection.
    # Rounding its free material block before projection fails the independent
    # pointwise energy identity, even with all common fields compensated.
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(4,0),
        reference=(2.,2.,2.),degree=degree,quadrature=20*degree,angular_quadrature=24,
        coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=background,gap_tangent=1.)
    assert metadata['transmission_directions']==0
    assert metadata['common_coordinate_precision']=='compensated'
    assert metadata['retained_verification']=='physical_field_integral'
    assert metadata['retained_residual']<1e-10
    assert np.isfinite(metadata['rounded_retained_defect'])
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    # With no y Fourier variation, reflection makes both off-diagonal slots zero.
    np.testing.assert_allclose(value[3:],0.,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent[3:],0.,atol=3e-9,rtol=0)


@pytest.mark.research
def test_sextic_positive_gap_retains_independent_energy_and_fourier_constraints():
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(4,0),
        reference=(2.,2.,2.),degree=6,quadrature=120,angular_quadrature=24,
        coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=.01,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=1.)
    assert metadata['transmission_directions']==0
    assert metadata['retained_verification']=='physical_field_integral'
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    assert metadata['retained_residual']<1e-10
    np.testing.assert_allclose(value[3:],0.,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent[3:],0.,atol=3e-9,rtol=0)


@pytest.mark.research
def test_complete_batched_transmission_contact_retains_metal_field_energy():
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(1,0),
        reference=(12.25,-10+1j,1.),degree=5,quadrature=100,angular_quadrature=24,
        coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=0.,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=1.)
    assert metadata['transmission_directions']==20
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    np.testing.assert_allclose(value[3:],0.,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent[3:],0.,atol=3e-9,rtol=0)


@pytest.mark.research
def test_complete_positive_gap_retains_transverse_fourier_trace_fields():
    from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
    # Nonzero y order exercises every transverse seam-trace column in the
    # complete positive system, beyond the scalar and x-only controls.
    # Angular order 32 underresolves this metal case: the original nodal
    # check detects the same defect as a finer independent integral.
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),(0,1),
        reference=(12.25,-10+1j,1.),degree=5,quadrature=100,
        angular_quadrature=48,coercivity_angle=1.53)
    value,tangent,metadata=cell.closure(gap=.01,basis=np.eye(2)*700,
        widths=(300.,300.),height=300.,offset=(-350.,-350.),
        epsilon=(12.25,-10+1j),background=1.,gap_tangent=1.)
    assert metadata['transmission_directions']==20
    assert metadata['retained_verification']=='physical_field_integral'
    for key in ('energy_residual','energy_derivative_residual','retained_residual'):
        assert metadata[key]<1e-10
    reflection=np.array([2,1,0])
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(response,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*slot[:,reflection][:,:,reflection],
                                       atol=tolerance,rtol=0)
