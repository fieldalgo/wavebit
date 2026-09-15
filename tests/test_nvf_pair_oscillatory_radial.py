"""Independent powers, physical variations and whole oscillatory pair controls."""
from decimal import Decimal as D, localcontext
from functools import lru_cache
from math import factorial
import numpy as np
import pytest
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit._core.conforming.pair_corner_fields import PairCornerFields
from wavebit._core.conforming.pair_oscillatory_radial import normalized_oscillatory_remainder


def _mul(a,b):
    return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]


def _divide(a,b):
    denominator=b[0]*b[0]+b[1]*b[1]
    return ((a[0]*b[0]+a[1]*b[1])/denominator,
            (a[1]*b[0]-a[0]*b[1])/denominator)


def _trigonometric(x):
    # Independent Decimal power series; no matrix exponential or production
    # coefficient builder is used by this reference.
    sine=st=x;cosine=ct=D(1);n=1
    while True:
        st*=-x*x/((2*n)*(2*n+1));ct*=-x*x/((2*n-1)*(2*n));n+=1
        sn,cn=sine+st,cosine+ct
        if sn==sine and cn==cosine:return cn,sn
        sine,cosine=sn,cn


def _reference(s,k,r,count,derivative,*,as_decimal=False):
    nodes=[(D(i),D(0)) for i in range(k,k+r)]
    nodes += [(D('.6'),D(0)),(D('.8'),D(0)),
              (D('.95'),D('.5')),(D('.95'),D('-.5'))][:count]
    total=D(0);log=s.ln()
    for i,a in enumerate(nodes):
        weight=(D((-1)**r*factorial(r)),D(0))
        for j,b in enumerate(nodes):
            if i!=j:weight=_divide(weight,(a[0]-b[0],a[1]-b[1]))
        for j in range(derivative):weight=_mul(weight,(a[0]-j,a[1]))
        cosine,sine=_trigonometric(a[1]*log)
        magnitude=((a[0]-derivative)*log).exp()
        total+=_mul(weight,(magnitude*cosine,magnitude*sine))[0]
    return total if as_decimal else float(total)


def test_oscillatory_radial_values_and_two_derivatives_against_decimal_powers():
    with localcontext() as context:
        context.prec=180
        for degree in (3,4):
            for k in range(1,7):
                r=max(0,degree-k)
                for count in range(1,5):
                    for text in ('1e-78','1e-12','.01','.2','.5','.7','.99','.99999999'):
                        s=D(text);h=1-s
                        actual=normalized_oscillatory_remainder([float(s)],[float(h)],k,r,count)
                        for derivative,value in enumerate(actual):
                            expected=_reference(s,k,r,count,derivative)
                            assert abs(value[0]-expected)<3e-10*max(abs(expected),1e-280)


@pytest.mark.parametrize('option',(1,0,'yes',None))
def test_oscillatory_option_requires_boolean(option):
    with pytest.raises(ValueError,match='oscillatory_radial must be a boolean'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,oscillatory_radial=option)


@pytest.mark.parametrize('exponents',((),(.6,),(.8,.6),(.55,.6,.8,.95)))
def test_oscillatory_option_requires_its_complete_fixed_real_prefix(exponents):
    with pytest.raises(ValueError,match='oscillatory radial fields require'):
        CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
                                         exponents=exponents,oscillatory_radial=True)


def _direct_values(fields,tri,bary,degree):
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
        remainder=np.zeros_like(h,dtype=complex)
        for a in nodes:
            weight=1.
            for b in nodes:
                if a!=b:weight/=a-b
            coefficient=1.;polynomial=np.zeros_like(h,dtype=complex)
            for j in range(r):
                polynomial+=coefficient*h**j
                coefficient*=(k+j-a)/(j+1)
            remainder+=weight*(s**(a-k)-polynomial)/coefficient
        scale=(-1.)**(len(nodes)-1)*factorial(r+len(nodes)-1)/factorial(r)
        result[:,index]=(scale*h*product*remainder).real
    return result


@pytest.mark.parametrize('degree,angular',((3,2),(4,4)))
def test_complete_real_oscillatory_fields_and_moving_gradients(degree,angular):
    obj=CartesianPairPolynomialGalerkinGapCell(CartesianPairOpeningPlan(0),0,degree=degree)
    fields=PairCornerFields(obj,junctions=True,angular_order=angular,oscillatory_radial=True)
    jac=np.array([[.7,.1],[.05,.6]]);direction=np.array([[.13,-.04],[.01,-.03]])
    inverse=np.linalg.inv(jac);dinverse=-inverse@direction@inverse
    bary=np.array([[.2,.3,.5],[.61,.29,.1]])
    dbary=np.array([[.04,-.02,-.02],[-.06,.03,.03]]);step=1e-4
    for tri in obj.data['triangles']:
        value,gradient,tangent=fields.sample(tri,bary,inverse,dinverse,dbary)
        np.testing.assert_allclose(value,_direct_values(fields,tri,bary,degree),atol=3e-10,rtol=0)
        assert np.isrealobj(value) and np.isrealobj(gradient) and np.isrealobj(tangent)
        for axis in range(2):
            dref=inverse[:,axis];move=np.r_[-dref.sum(),dref]
            values=[fields.sample(tri,bary+i*step*move,inverse,np.zeros((2,2)))[0] for i in (-2,-1,1,2)]
            difference=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
            np.testing.assert_allclose(gradient[:,axis],difference,atol=3e-9,rtol=0)
        values=[fields.sample(tri,bary+i*step*dbary,np.linalg.inv(jac+i*step*direction),np.zeros((2,2)))[1] for i in (-2,-1,1,2)]
        difference=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(tangent,difference,atol=3e-9,rtol=0)


def _cell(order=0,angular=0,*,mesh='symmetric',junctions=True,angular_quadrature=16):
    return CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),order,
        degree=3,quadrature=80,junctions=junctions,angular_order=angular,
        coercivity_angle=1.53,mesh=mesh,oscillatory_radial=True,angular_quadrature=angular_quadrature)


def _settings(gap,scale=700.):
    return dict(gap=gap*scale,basis=np.eye(2)*scale,widths=np.array([3/7,3/7])*scale,
                height=3/7*scale,epsilon=(12.25,-10+1j))


def test_oscillatory_contact_preserves_full_fourier_reflection():
    cell=_cell(order=1,angular=2)
    value,tangent,metadata=cell.closure(**_settings(0.))
    assert metadata['corner_exponents']==(.6,.8)
    assert metadata['oscillatory_radial'] is True
    assert metadata['oscillatory_radial_power']==.95
    assert metadata['oscillatory_radial_frequency']==.5
    assert metadata['radial_root_order']==20
    modes=[(i,j) for j in range(-1,2) for i in range(-1,2)]
    reflection=np.array([modes.index((i,-j)) for i,j in modes])
    for array,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(array,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*slot[:,reflection][:,:,reflection],atol=tolerance,rtol=0)


@pytest.mark.parametrize('gap',(0.,.003))
def test_oscillatory_family_preserves_live_units_and_joint_material_gap_directions(gap):
    # Bound multi-state positive-gap controls by using the original mesh.
    # Contact and the independent full-coordinate oracle cover the symmetric space.
    cell=_cell(mesh='original' if gap else 'symmetric')
    reference=cell.closure(**_settings(gap,1.))
    settings=_settings(gap)
    factor=1+.02j;direction=.3+.2j
    settings['epsilon']=factor*np.asarray(settings['epsilon'])
    settings.update(background=factor,gap_tangent=700.,
                    epsilon_tangent=direction*settings['epsilon'],
                    background_tangent=direction*factor)
    value,tangent,_=cell.closure(**settings)
    slot_scale=np.array([factor,factor,1/factor,factor,factor])[:,None,None,None]
    sign=np.array([1.,1.,-1.,1.,1.])[:,None,None,None]
    np.testing.assert_allclose(value,slot_scale*reference[0],atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,slot_scale*(reference[1]+direction*sign*reference[0]),atol=3e-9,rtol=0)


@lru_cache(maxsize=1)
def _physical_difference_cell():
    return _cell(mesh='original',junctions=False,angular_quadrature=12)


@lru_cache(maxsize=8)
def _physical_difference_state(gap):
    settings=_settings(gap,1.);settings['epsilon']=(4.,9.)
    return tuple(np.asarray(a) for a in _physical_difference_cell().closure(**settings)[:2])


@pytest.mark.parametrize('step',(3e-5,1.5e-5))
def test_oscillatory_positive_gap_derivative_matches_physical_differences(step):
    # Keep each complete physical stencil within the CI time limit using
    # original-mesh outer-corner fields. Independent raw/complete-coordinate
    # variations and whole joint-direction controls include junction fields.
    # The two stencils share only exactly identical complete physical states.
    gap=.02
    _,tangent=_physical_difference_state(gap)
    values=[_physical_difference_state(gap+i*step)[0] for i in (-2,-1,1,2)]
    difference=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
    np.testing.assert_allclose(tangent,difference,atol=3e-9,rtol=0)


@pytest.mark.parametrize('gap',(0.,.02))
def test_oscillatory_quadrature_resolves_the_same_complete_response(gap):
    settings=_settings(gap,1.);settings['epsilon']=(4.,9.)
    mesh='original' if gap else 'symmetric'
    low=_cell(mesh=mesh).closure(**settings)
    cell=_cell(mesh=mesh);cell.quadrature=96
    high=cell.closure(**settings)
    for value,reference,tolerance in zip(high[:2],low[:2],(3e-10,3e-9)):
        np.testing.assert_allclose(value,reference,atol=tolerance,rtol=0)


def test_radial_cache_eviction_and_oversized_profiles_preserve_exact_powers(monkeypatch):
    from collections import OrderedDict
    import wavebit._core.conforming.pair_oscillatory_radial as radial
    monkeypatch.setattr(radial,'_cache',OrderedDict())
    monkeypatch.setattr(radial,'_cache_bytes',0)
    monkeypatch.setattr(radial,'_CACHE_BYTES',3000)
    profiles=[np.array([.2,.3,.4,.5])+shift for shift in (0.,.01,.02,.03,0.)]
    profiles.append(np.linspace(.2,.5,25))
    with localcontext() as context:
        context.prec=180
        for s in profiles:
            result=radial.normalized_oscillatory_remainder(s,1-s,1,2,4,3)
            for derivative,actual in enumerate(result):
                expected=np.array([_reference(D.from_float(x),1,2,4,derivative) for x in s])
                np.testing.assert_allclose(actual,expected,atol=0,rtol=3e-10)
            payload=sum(value.nbytes+len(key[0])+len(key[1])
                        for key,(value,_) in radial._cache.items())
            assert payload<=3000


@pytest.mark.parametrize('degree',(3,4))
def test_scalar_columns_preserve_the_matrix_evaluator_rounding(degree):
    import wavebit._core.conforming.pair_oscillatory_radial as radial
    h=np.r_[np.linspace(0.,.9999,4097),np.ones(24)]
    s=np.r_[1-h[:4097],np.geomspace(1e-12,1e-78,24)]
    np.testing.assert_array_equal(radial._values(h.tobytes(),s.tobytes(),degree),
                                  radial._matrix_values(h.tobytes(),s.tobytes(),degree))
