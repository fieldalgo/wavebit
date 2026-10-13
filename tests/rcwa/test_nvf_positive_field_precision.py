"""Independent Decimal potential differences for complete positive polynomial fields."""
import numpy as np
import pytest
from decimal import Decimal as DD, localcontext
from math import factorial, prod
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin import CartesianPairTransmissionGalerkinGapCell
from wavebit.rcwa._engine.conforming.pair_transmission_fields import PairTransmissionFields
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import Exact
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_positive_fields import builder
from wavebit.rcwa._engine.conforming.pair_corner_assembly import augment

pytestmark = pytest.mark.research

@pytest.mark.parametrize('gap', (.01,1e-9,1e-13))
@pytest.mark.parametrize('degree',(6,7,8))
@pytest.mark.parametrize('exact_points',(False,True))
def test_complete_physical_fields_match_decimal_potentials(gap,degree,exact_points):
    settings=dict(gap=gap,basis=np.eye(2)*700,widths=(300.,300.),height=300.,offset=(-350.,-350.),epsilon=(12.25,-10+1j),background=1.,gap_tangent=1.,epsilon_tangent=(0.,0.),background_tangent=0.)
    cell=CartesianPairTransmissionGalerkinGapCell(CartesianPairOpeningPlan(0),0,reference=(2.,2.,2.),degree=degree,quadrature=20*degree,angular_quadrature=24,coercivity_angle=1.53)
    obj=cell._base
    fields=PairTransmissionFields(obj,settings,(2.,2.,2.),2,24)
    base=augment(obj,settings,exponents=(),quadrature=12,junctions=True,prepared_fields=fields,angular_quadrature=24)
    info=base[-1];make=builder(obj,settings,base)
    _,_,zero,velocity=obj._chart_geometry(settings['gap'],{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    indices=[0,3,7,-1,next(i for i,v in enumerate(info['local']) if v[11])];rng=np.random.default_rng(79)
    def dec(v):return DD.from_float(float(v))
    with localcontext() as context:
     context.prec=160;hs=dec(info['s'])*DD('1e-25');ht=dec(info['s'])*DD('1e-20')
     for index in indices:
      tri=obj.data['triangles'][index];local=info['local'][index];ci,zi=local[1:3]
      bary=np.array([[.2,.3,.5],[.4,.4,.2]]);db=np.array([[.11,-.07,-.04],[-.04,.1,-.06]])
      if exact_points:
       bary=Exact(D(bary,np.array([[1.,-2.,1.],[-2.,1.,1.]])*2.**-59))
      row=[None]*23;row[1:3]=ci,zi;row[3:7]=np.ones(2),np.zeros(2),local[12],local[13]
      row[13]=row[14]=np.zeros((2,2,1));row.append(dict(triangle=tri,barycentric=bary,tangent=db,weight=Exact(D(np.ones(2)))))
      actual=make(tuple(row),slice(None));g,dg=actual[7:9]
      weights=rng.normal(size=len(ci)+len(zi));g=D(*tuple(v.reshape(2,2,-1) for v in g));dg=D(*tuple(v.reshape(2,2,-1) for v in dg))
      c=obj.data['Cfull'][obj.data['periodic'][tri],1:][:,ci];z=obj.data['Zfull'][obj.data['periodic'][tri]][:,zi]
      common=[sum(dec(v)*dec(w) for v,w in zip(row,weights[:len(ci)])) for row in c]
      normal=[sum(dec(v)*dec(w) for v,w in zip(row,weights[len(ci):])) for row in z]
      alphas=multi_indices(degree)
      def potential(b,s):
       return sum(DD(factorial(degree)//prod(factorial(int(a)) for a in alpha))*prod(b[i]**int(alpha[i]) for i in range(3))*(u+s*v) for alpha,u,v in zip(alphas,common,normal))
      def gradient(b,s,axis):
       p=[[dec(zero[v,a])+s*dec(velocity[v,a]) for a in range(2)] for v in tri[:3]]
       jac=[[p[j+1][i]-p[0][i] for j in range(2)] for i in range(2)];det=jac[0][0]*jac[1][1]-jac[0][1]*jac[1][0]
       inv=[[jac[1][1]/det,-jac[0][1]/det],[-jac[1][0]/det,jac[0][0]/det]]
       direction=[-inv[0][axis]-inv[1][axis],inv[0][axis],inv[1][axis]]
       return (potential([v+hs*d for v,d in zip(b,direction)],s)-potential([v-hs*d for v,d in zip(b,direction)],s))/(2*hs)
      worst=DD(0)
      for q in range(2):
       b=list(map(dec,bary[q]));dbb=list(map(dec,db[q]));anchor=int(np.argmax(abs(bary[q])))
       if exact_points:b=[dec(bary.exact.hi[q,i])+dec(bary.exact.lo[q,i]) for i in range(3)]
       b[anchor]=1-sum(v for i,v in enumerate(b) if i!=anchor);dbb[anchor]=-sum(v for i,v in enumerate(dbb) if i!=anchor)
       for axis in range(2):
        expected=gradient(b,dec(info['s']),axis)
        changed=(gradient([v+ht*d for v,d in zip(b,dbb)],dec(info['s'])+ht*dec(info['ds']),axis)-gradient([v-ht*d for v,d in zip(b,dbb)],dec(info['s'])-ht*dec(info['ds']),axis))/(2*ht)
        for value,reference in ((g,expected),(dg,changed)):
         observed=sum((dec(value.hi[q,axis,j])+dec(value.lo[q,axis,j]))*dec(weights[j]) for j in range(len(weights)))
         error=abs(observed-reference)/max(DD(1),abs(reference));worst=max(worst,error)
         assert error<DD('3e-28'),(index,q,axis,error)
