"""Complete moving physical fields for higher-degree positive transmission gaps.

Every common polynomial, fixed transmission mode, disappearing potential and
continuous trace survives. Collapsing gradients cancel their gap analytically
before compensated evaluation. Clipped trace rows retain the original
partition and transport; no sampled field is fitted or removed.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_corner_precision import simplex_coordinates
from wavebit.rcwa._engine.conforming.pair_complex_precision import C
from wavebit.rcwa._engine.conforming.pair_complex_exponential import exponential
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import quadrature
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import coordinates
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import PI
import wavebit.rcwa._engine.conforming.pair_oscillatory_fields as jet
from wavebit.rcwa._engine.conforming.pair_contact_precision import pair
from wavebit.rcwa._engine.conforming.pair_transmission_common_precision import CommonCoordinates


def unsplit(obj,settings,base,order):
    """Original exact triangular rule with no unnecessary rounded field work."""
    info=base[-1];fields=info['corner_fields']
    extra=np.arange(info['original_nc'],info['nc'])
    for tri,local in zip(obj.data['triangles'],info['local']):
        for bary,weight in quadrature(fields,tri,order):
            row=[None]*23
            row[1]=np.r_[local[1],extra];row[2]=local[2]
            row[3]=np.asarray(weight);row[5:7]=local[12:14]
            row[13]=row[14]=np.zeros((len(bary),2,0))
            yield (*row,dict(triangle=tri,barycentric=bary,
                            tangent=np.zeros_like(bary),weight=weight))


def builder(obj,settings,base):
 """Evaluate complete physical fields before rounding any common coefficient."""
 info=base[-1];nc,nz,N,s,ds=(info[k] for k in ('nc','nz','size','s','ds'));fields=info['corner_fields']
 length,_,zero,velocity=obj._chart_geometry(settings['gap'],{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
 locals_by_triangle={tuple(t):v for t,v in zip(obj.data['triangles'],info['local'])};providers={}
 period=D(np.diag(settings['basis'])/length);nx,ny=obj.order;modes=D(np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)]).T)
 def stack(values,axis=0):return D(np.stack([v.hi for v in values],axis),np.stack([v.lo for v in values],axis))
 def adj(j):return stack([stack([j[1,1],-j[0,1]]),stack([-j[1,0],j[0,0]])])
 def lift(v):
  out=[np.zeros((len(v.real),2,2*N),complex) for _ in range(2)]
  for target,part in zip(out,v.pair):target[:,0,:N]=part;target[:,1,N:]=part
  return tuple(v.reshape(-1,2*N) for v in out)
 def make(row,sl):
  meta=row[-1];tri=meta['triangle'];bary=meta['barycentric'][sl];db0=np.asarray(meta['tangent'][sl]);b=coordinates(bary);db=simplex_coordinates(np.asarray(bary),db0)[1]
  ci,zi,ep,dep=(row[k] for k in (1,2,5,6));indices=np.r_[ci,nc+zi]
  points=D(zero[tri[:3]])+D(s)*D(velocity[tri[:3]]);speeds=D(ds)*D(velocity[tri[:3]])
  jac=stack([points[1]-points[0],points[2]-points[0]],1);djac=stack([speeds[1]-speeds[0],speeds[2]-speeds[0]],1)
  thin=locals_by_triangle[tuple(tri)][11]
  if thin:
   zz=D(zero[tri[:3]]);vv=D(velocity[tri[:3]])
   j0=stack([zz[1]-zz[0],zz[2]-zz[0]],1);j1=stack([vv[1]-vv[0],vv[2]-vv[0]],1)
   d1=j0[0,0]*j1[1,1]+j1[0,0]*j0[1,1]-j0[0,1]*j1[1,0]-j1[0,1]*j0[1,0]
   invc=adj(j1)/d1;dic=D.zeros((2,2));invz=(adj(j0)+D(s)*adj(j1))/d1;diz=D(ds)*invc
   det=D(s)*d1;ddet=D(ds)*d1
  else:
   det=jac[0,0]*jac[1,1]-jac[0,1]*jac[1,0]
   ddet=djac[0,0]*jac[1,1]+jac[0,0]*djac[1,1]-djac[0,1]*jac[1,0]-jac[0,1]*djac[1,0]
   invc=adj(jac)/det;dic=-(invc@djac@invc);invz=D(s)*invc;diz=D(ds)*invc+D(s)*dic
  key=tuple(tri)
  if key not in providers:providers[key]=getattr(fields,'common_coordinate_type',CommonCoordinates)(fields,np.eye(nc)[:,ci])
  common_db=meta.get('common_tangent',meta['tangent'])[sl]
  common=providers[key].sample(tri,bary,invc,dic,common_db)
  static=not any(np.any(v.hi) or np.any(v.lo) for v in db)
  mapping=D(obj.data['Zfull'][obj.data['periodic'][tri]][:,zi])
  if obj.degree>=7:
   from wavebit.rcwa._engine.conforming.pair_extended_precision import Triple as T
   from wavebit.rcwa._engine.conforming.pair_transmission_extended_precision import _coordinates
   from wavebit.rcwa._engine.conforming.pair_transmission_extended_precision import _polynomial
   # Differentiate every Bernstein coefficient before the physical map.
   # Higher degrees exceed the older small-product kernel's proven size.
   grad,delta=_polynomial(obj.degree,_coordinates(bary),list(map(T,db)),T(mapping),not static)
   gz=[D(*sum(grad[a]*T(invz[a,k]) for a in range(2)).pair) for k in range(2)]
   dgz=[D(*sum(delta[a]*T(invz[a,k])+grad[a]*T(diz[a,k]) for a in range(2)).pair) for k in range(2)]
  else:
   ref=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
   jets=[jet.Jet((b[i],db[i],D(ref[i,0]),D(ref[i,1]),D(0.),D(0.)),static) for i in range(3)]
   polynomial=jet.bernstein(jets,obj.degree)
   normal=jet.mapped(polynomial,mapping,static,True);grad=normal.parts[2:4];delta=normal.parts[4:6]
   gz=[sum(grad[a]*invz[a,k] for a in range(2)) for k in range(2)]
   dgz=[sum(delta[a]*invz[a,k]+grad[a]*diz[a,k] for a in range(2)) for k in range(2)]
  def combine(regular,normal):
   a=stack(regular,1);z=stack(normal,1)
   return D(np.concatenate([a.hi,z.hi],2),np.concatenate([a.lo,z.lo],2))
  g=combine(common[:2],gz);dg=combine(common[2:],dgz)
  bb=stack(b,1);dbb=stack(db,1);xy=bb@points;vxy=dbb@points+bb@speeds
  theta=2*PI*(xy/period@modes);dtheta=2*PI*(vxy/period@modes)
  phase=exponential(1j*C(theta));dphase=phase*1j*C(dtheta)
  if 'weight' in meta:
   weight=meta['weight'].exact[sl];w=weight*det;dw=weight*ddet
  else:w=D(row[3][sl]);dw=D(row[4][sl])
  ww=tuple(np.repeat(v,2)[:,None] for v in w.pair);dww=tuple(np.repeat(v,2)[:,None] for v in dw.pair)
  count=len(bary)
  def flatten(value):return tuple(v.reshape(2*count,v.shape[-1]) for v in value.pair)
  le=pair(row[13][sl].reshape(2*count,row[13].shape[-1]));dle=pair(row[14][sl].reshape(2*count,row[14].shape[-1]))
  return indices,ww,dww,ep,dep,lift(phase),lift(dphase),flatten(g),flatten(dg),le,dle
 return make
