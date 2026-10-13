"""Independent complete real potentials and original contact-response guards."""
from decimal import Decimal as Decimal, localcontext
from functools import lru_cache
from math import factorial, prod
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from wavebit.rcwa._engine.conforming.pair_corner_common_precision import CommonCoordinates
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices

pytestmark = pytest.mark.research


def dec(value):
    return Decimal.from_float(float(value))


def monomials(bary,degree):
    return [Decimal(factorial(degree)//prod(factorial(int(v)) for v in alpha))
            *prod(bary[i]**int(v) for i,v in enumerate(alpha) if v)
            for alpha in multi_indices(degree)]


@pytest.fixture(scope='module',params=(3,4),ids=('cubic','quartic'))
def common(request):
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=request.param,quadrature=80,exponents=(.55,.6,.8,.95),
        junctions=True,angular_order=0,angular_quadrature=12,mesh='symmetric')
    fields=cell._fields
    count=fields.obj.data['Cfull'].shape[1]-1+fields.count
    mapping=np.random.default_rng(31).normal(size=(count,3))*[1.,256.,1/256]
    return CommonCoordinates(fields,mapping)


def check(common,tri,bary,*,corner=False):
    fields=common.fields;data=fields.obj.data
    inverse=np.array([[1.25,.25],[.125,.875]])
    dinverse=np.array([[.03125,.0625],[-.015625,.03125]])
    actual=common.sample(tri,bary,inverse,dinverse)
    with localcontext() as context:
        context.prec=420 if corner else 100
        nold=data['Cfull'].shape[1]-1;columns=common.mapping.hi.shape[1]
        matrix=common.mapping.hi
        local=fields.interpolant[data['pmap'][data['periodic'][tri]]]
        raw=np.full((fields.count,columns),Decimal(0),object)
        for ids,transform in zip(fields.groups,fields.transforms):
            for row,index in enumerate(ids):
                for col in range(columns):
                    raw[index,col]=dec(fields.column_scale[index])*sum(
                        dec(transform[row,j])*dec(matrix[nold+i,col])
                        for j,i in enumerate(ids))
        polynomial=np.empty((len(tri),columns),object)
        for row,node in enumerate(tri):
            for col in range(columns):
                polynomial[row,col]=sum(
                    dec(v)*dec(matrix[i,col])
                    for i,v in enumerate(data['Cfull'][data['periodic'][node],1:]) if v
                )-sum(dec(local[row,i])*raw[i,col] for i in range(fields.count))
        hats=[[Decimal(int(n))/int(data['weight_denominator']) for n in row]
              for row in data['contact_numerators'][data['mapping'][tri[:3]]]]
        active=np.any(fields.hats[tri[:3]]!=0,axis=0)

        @lru_cache(None)
        def potential(b):
            hv=[sum(b[v]*hats[v][i] for v in range(3)) for i in range(10)]
            result=[sum(value*polynomial[row,col]
                        for row,value in enumerate(monomials(b,fields.obj.degree)))
                    for col in range(columns)]
            radial={}
            for index,c,neighbors,cutoff,angular,nodes in fields.newton:
                if not active[c] or any(not active[n] for n in neighbors):
                    continue
                h=hv[c];s=sum(hv[i] for i in range(10) if i!=c)
                k=len(neighbors)+angular;r=max(0,cutoff+1)
                key=c,k,r,nodes
                if key not in radial:
                    values=tuple(Decimal(str(a)) for a in nodes)
                    scale=Decimal((-1)**(len(nodes)-1)*factorial(r+len(nodes)-1))/factorial(r)
                    total=Decimal(0)
                    for alpha in values:
                        lead=Decimal(1);overlap=Decimal(0)
                        for j in range(r):
                            overlap+=lead*h**j
                            lead*=(Decimal(k+j)-alpha)/Decimal(j+1)
                        weight=scale/lead
                        for other in values:
                            if other!=alpha:
                                weight/=alpha-other
                        total+=weight*((s.ln()*(alpha-k)).exp()-overlap)
                    radial[key]=total
                factor=h*prod(hv[n] for n in neighbors)*radial[key]
                if angular:
                    factor*=(hv[neighbors[0]]-hv[neighbors[1]])**angular
                for col in range(columns):
                    result[col]+=factor*raw[index,col]
            return result

        for q,point in enumerate(bary):
            b=list(map(dec,point));anchor=int(np.argmax(abs(point)))
            b[anchor]=Decimal(1)-sum(b[i] for i in range(3) if i!=anchor)
            step=(min(v for v in b if v>0)*Decimal('1e-20')
                  if corner else Decimal('1e-20'))
            for slot,transform in enumerate((inverse,dinverse)):
                for axis in range(2):
                    direction=[-sum(dec(transform[i,axis]) for i in range(2)),
                               dec(transform[0,axis]),dec(transform[1,axis])]
                    left=potential(tuple(v-step*dv for v,dv in zip(b,direction)))
                    right=potential(tuple(v+step*dv for v,dv in zip(b,direction)))
                    expected=np.array([float((r-l)/(2*step)) for l,r in zip(left,right)])
                    component=actual[2*slot+axis]
                    observed=(component.hi+component.lo)[q]
                    error=np.max(abs(observed-expected)/np.maximum(1.,abs(expected)))
                    assert error<3e-14,(q,slot,axis,error,observed,expected)


def test_complete_common_fields_match_independent_nondyadic_potentials(common):
    bary=np.array([[.23,.31,.46],[.58,.13,.29],[.31,.46,.23]])
    for tri in common.fields.obj.data['triangles'][::23]:
        check(common,tri,bary)


def test_complete_common_fields_preserve_extreme_corner_distances(common):
    fields=common.fields
    triangles=[tri for tri in fields.obj.data['triangles']
               if np.any(fields.hats[tri[:3],4]==1)][:2]
    for tri in triangles:
        anchor=int(np.flatnonzero(fields.hats[tri[:3],4]==1)[0])
        others=[i for i in range(3) if i!=anchor]
        bary=np.zeros((3,3))
        for row,power in enumerate((20,80,200)):
            radius=2.**(-power)
            bary[row,anchor]=1-radius
            bary[row,others]=radius*np.array([.25,.75])
        check(common,tri,bary,corner=True)


@lru_cache(maxsize=2)
def metal_response(angular):
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),(1,1),
        degree=3,quadrature=80,exponents=(.55,.6,.8,.95),junctions=True,
        angular_order=0,angular_quadrature=angular,mesh='symmetric',coercivity_angle=1.53)
    return cell.closure(gap=0.,basis=np.diag([1.2,.9]),widths=(.3,.4),height=.45,
        offset=(.17,-.08),epsilon=(12.25,-10+1j),background=1.,gap_tangent=.7,
        epsilon_tangent=(.2+.1j,-.4+.2j),background_tangent=.1-.03j)


def test_non_square_metal_contact_preserves_entire_joint_reflection():
    value,tangent,metadata=metal_response(12)
    assert metadata['common_coordinate_precision']=='compensated'
    assert metadata['energy_residual']<1e-10
    assert metadata['energy_derivative_residual']<1e-10
    modes=np.array([(i,j) for j in range(-1,2) for i in range(-1,2)])
    lookup={tuple(mode):i for i,mode in enumerate(modes)}
    reflected=np.array([lookup[i,-j] for i,j in modes])
    phase=np.exp(4j*np.pi*(modes[:,1][None,:]-modes[:,1][:,None])*(-.08/.9))
    for response,tolerance in ((value,3e-10),(tangent,3e-9)):
        for slot,sign in zip(response,(1,1,1,-1,-1)):
            np.testing.assert_allclose(slot,sign*phase*slot[:,reflected][:,:,reflected],
                                       atol=tolerance,rtol=0)


def test_non_square_metal_contact_retains_complete_quadrature_accuracy():
    reference=metal_response(12)
    refined=metal_response(24)
    for actual,expected,tolerance in zip(refined[:2],reference[:2],(3e-10,3e-9)):
        np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=0)


def test_quartic_single_radial_contact_has_exact_uniform_material_variation():
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=4,quadrature=80,exponents=(.55,),junctions=True,
        angular_order=0,angular_quadrature=12,mesh='symmetric')
    epsilon=5+.1j;direction=.7-.2j
    value,tangent,metadata=cell.closure(gap=0.,basis=np.diag([1.2,.9]),
        widths=(.3,.4),height=.45,offset=(.17,-.08),epsilon=(epsilon,epsilon),
        background=epsilon,gap_tangent=.9,epsilon_tangent=(direction,direction),
        background_tangent=direction)
    assert metadata['common_coordinate_precision']=='compensated'
    expected=np.array([epsilon,epsilon,1/epsilon,0.,0.])[:,None,None,None]
    change=np.array([direction,direction,-direction/epsilon**2,0.,0.])[:,None,None,None]
    np.testing.assert_allclose(value,expected,atol=3e-10,rtol=0)
    np.testing.assert_allclose(tangent,change,atol=3e-9,rtol=0)
