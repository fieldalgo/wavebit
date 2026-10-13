"""Independent high-precision potentials and physical variations of full coordinates."""
from collections import OrderedDict
from decimal import Decimal as D, localcontext
from functools import lru_cache
from math import factorial, prod
import numpy as np
import pytest
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell
from tests.rcwa.test_nvf_pair_oscillatory_radial import _reference

pytestmark = pytest.mark.research


@pytest.fixture(scope='module',params=(3,4),ids=('cubic','quartic'))
def fields(request):
    return CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,degree=request.param,
        quadrature=80,junctions=True,angular_order=4,mesh='symmetric',
        coercivity_angle=1.53,oscillatory_radial=True)._fields


def dec(a):return D.from_float(float(a))

def monomials(bary,degree):
    return [D(factorial(degree)//prod(factorial(n) for n in alpha))*prod(bary[i]**n for i,n in enumerate(alpha) if n) for alpha in multi_indices(degree)]


def check(fields,tri,bary,dbary,inverse,dinverse,*,corner=False):
    actual=fields.sample(tri,bary,inverse,dinverse,dbary)
    for gradient,reference in zip(fields.gradients(tri,bary,inverse,dinverse,dbary),actual[1:]):
        assert gradient.tobytes()==reference.tobytes()
    with localcontext() as context:
        context.prec=420 if corner else 100
        step=D('1e-18')
        data=fields.obj.data;local=fields.interpolant[data['pmap'][data['periodic'][tri]]]
        numerators=data['contact_numerators'][data['mapping'][tri[:3]]]
        hats=[[D(int(n))/int(data['weight_denominator']) for n in row] for row in numerators]
        active=np.any(fields.hats[tri[:3]]!=0,axis=0)
        chosen=[]
        for control,ids,transform in zip(fields.controls,fields.groups,fields.transforms):
            if active[control] and len(ids):
                chosen.extend((ids,j,transform[:,j]) for j in (0,len(ids)//2,len(ids)-1))
        @lru_cache(None)
        def evaluate(b):
            hv=[sum(b[v]*hats[v][i] for v in range(3)) for i in range(10)]
            radial={};raw=[D(0)]*fields.count
            for index,c,ns,cutoff,j,nodes in fields.newton:
                if not active[c] or any(not active[n] for n in ns):continue
                h=hv[c];s=sum(hv[n] for n in range(10) if n!=c)
                k=len(ns)+j;r=max(0,cutoff+1);key=c,k,r,len(nodes)
                if key not in radial:
                    scale=D((-1)**(len(nodes)-1)*factorial(r+len(nodes)-1))/factorial(r)
                    radial[key]=scale*_reference(s,k,r,len(nodes),0,as_decimal=True)/s**k
                prefactor=h*prod(hv[n] for n in ns)
                if j:prefactor*=(hv[ns[0]]-hv[ns[1]])**j
                raw[index]=prefactor*radial[key]
            p=monomials(b,fields.obj.degree);result=[]
            for ids,j,transform in chosen:
                result.append(sum((raw[i]-sum(p[n]*dec(local[n,i]) for n in range(len(p))))*dec(fields.column_scale[i])*dec(a) for i,a in zip(ids,transform)))
            return result
        def gradient(b,inv,axis):
            direction=[-sum(inv[a][axis] for a in range(2)),inv[0][axis],inv[1][axis]]
            left=evaluate(tuple(v-step*dv for v,dv in zip(b,direction)))
            right=evaluate(tuple(v+step*dv for v,dv in zip(b,direction)))
            return [(r-l)/(2*step) for l,r in zip(left,right)]
        inv=[[dec(a) for a in row] for row in inverse];dinv=[[dec(a) for a in row] for row in dinverse]
        for q in range(len(bary)):
            # A triangle has two independent coordinates. Keep the smaller
            # inputs, including sub-ulp corner distances, and define the
            # remaining coordinate on the exact Decimal unit simplex.
            b=list(map(dec,bary[q]));db=list(map(dec,dbary[q]))
            anchor=int(np.argmax(abs(bary[q])))
            b[anchor]=D(1)-sum(b[i] for i in range(3) if i!=anchor)
            db[anchor]=-sum(db[i] for i in range(3) if i!=anchor)
            b,db=tuple(b),tuple(db)
            if corner:step=min(v for v in b if v>0)*D('1e-18')
            value=evaluate(b)
            g=[gradient(b,inv,a) for a in range(2)]
            if corner:
                dg=[[sum(dec(dinverse[a,axis])*g[a][n] for a in range(2))
                     for n in range(len(chosen))] for axis in range(2)]
            else:
                left=[gradient(tuple(v-step*dv for v,dv in zip(b,db)),[[v-step*dv for v,dv in zip(row,drow)] for row,drow in zip(inv,dinv)],a) for a in range(2)]
                right=[gradient(tuple(v+step*dv for v,dv in zip(b,db)),[[v+step*dv for v,dv in zip(row,drow)] for row,drow in zip(inv,dinv)],a) for a in range(2)]
                dg=[[(r-l)/(2*step) for l,r in zip(ls,rs)] for ls,rs in zip(left,right)]
            for n,(ids,j,_) in enumerate(chosen):
                for slot,observed,expected in ((0,actual[0][q,ids[j]],value[n]),(1,actual[1][q,0,ids[j]],g[0][n]),(1,actual[1][q,1,ids[j]],g[1][n]),(2,actual[2][q,0,ids[j]],dg[0][n]),(2,actual[2][q,1,ids[j]],dg[1][n])):
                    error=abs(observed-float(expected))/max(1.,abs(float(expected)))
                    assert error<3e-14,(tuple(tri),q,n,slot,observed,float(expected),error)
        evaluate.cache_clear()


def test_complete_coordinates_and_live_shape_variations_against_decimal(fields):
    bary=np.array([[.25,.25,.5],[.625,.25,.125]])
    dbary=np.array([[.0625,-.03125,-.03125],[-.0625,.03125,.03125]])
    inverse=np.array([[1.25,.25],[.125,.875]])
    dinverse=np.array([[.03125,.0625],[-.015625,.03125]])
    for tri in fields.obj.data['triangles'][::23]:
        check(fields,tri,bary,dbary,inverse,dinverse)


def test_nondyadic_simplex_coordinates_and_variations_against_decimal(fields):
    bary=np.array([[.23,.31,.46],[.58,.13,.29],[.31,.46,.23]])
    dbary=np.array([[.04,-.01,-.03],[-.03,.02,.01],[.01,-.03,.02]])
    inverse=np.array([[1.25,.25],[.125,.875]])
    dinverse=np.array([[.03125,.0625],[-.015625,.03125]])
    for tri in fields.obj.data['triangles'][::23]:
        check(fields,tri,bary,dbary,inverse,dinverse)


def test_complete_singular_corner_coordinates_against_decimal(fields):
    triangles=[tri for tri in fields.obj.data['triangles'] if np.any(fields.hats[tri[:3],4]==1)][:2]
    for tri in triangles:
        anchor=int(np.flatnonzero(fields.hats[tri[:3],4]==1)[0])
        others=[i for i in range(3) if i!=anchor]
        bary=np.zeros((4,3))
        for row,exponent in enumerate((20,40,80,200)):
            radius=2.**(-exponent);bary[row,anchor]=1-radius
            bary[row,others]=radius*np.array([.25,.75])
        check(fields,tri,bary,np.zeros_like(bary),np.eye(2),
              np.array([[.03125,.0625],[-.015625,.03125]]),corner=True)


def test_oversized_coordinate_tables_keep_every_exact_field(fields,monkeypatch):
    import wavebit.rcwa._engine.conforming.pair_oscillatory_fields as evaluator
    monkeypatch.setattr(evaluator,'_COEFFICIENT_BYTES',1024)
    monkeypatch.setattr(fields,'_oscillatory_coordinate_cache',OrderedDict())
    monkeypatch.setattr(fields,'_oscillatory_coordinate_bytes',0)
    bary=np.array([[.25,.25,.5]])
    check(fields,fields.obj.data['triangles'][0],bary,np.zeros_like(bary),np.eye(2),np.zeros((2,2)))
    assert fields._oscillatory_coordinate_bytes<=1024
    assert not fields._oscillatory_coordinate_cache


def test_singular_vertex_gradient_requires_a_positive_complement(fields):
    tri=next(tri for tri in fields.obj.data['triangles'] if np.any(fields.hats[tri[:3],4]==1))
    anchor=int(np.flatnonzero(fields.hats[tri[:3],4]==1)[0])
    bary=np.zeros((1,3));bary[0,anchor]=1.
    with pytest.raises(ValueError,match='positive complement'):
        fields.gradients(tri,bary,np.eye(2),np.zeros((2,2)))
