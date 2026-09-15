"""Independent physical contact energy from complete high/low field coordinates.

The verifier reconstructs all retained, common and disappearing fields before
forming pointwise energy and its stationary physical variation. Its inputs
contain no candidate tensor, tensor derivative or solved common variation.
The same reconstructed field supplies its directly integrated Fourier mean.
"""
import numpy as np
from .pair_contact_precision import pair,mat,add,cp,retained_field,add_at
from .pair_corner_precision import D
from .pair_contact_quadrature import quadrature,phase_values,triangle_measures
from .pair_corner_common_precision import CommonCoordinates

def verifier(obj,settings,base,transform,chart,order,records,edge_rows):
 info=base[-1];nc,nz,N,ds=(info[k] for k in ('nc','nz','size','ds'));fields=info['corner_fields'];size=2*N;rev=info['reverse']
 nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
 _,_,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
 provider=getattr(fields,'common_coordinate_type',CommonCoordinates)
 def energy(field,field_change,potential):
  coefficients=potential;field_pair=field;change_pair=field_change
  common=provider(fields,D(*(np.hstack([v[:nc].real,v[:nc].imag]) for v in coefficients)))
  value=pair(np.zeros((size,size),complex));delta=pair(np.zeros((size,size),complex));extra=tuple(v[nc+nz:] for v in coefficients);rows=iter(records())
  retained=pair(np.zeros((size,size),complex))
  def combined(sample,offset):
   return tuple(np.stack([getattr(v,part)[:,:size]+1j*getattr(v,part)[:,size:] for v in sample[offset:offset+2]],axis=1).reshape(-1,size) for part in ('hi','lo'))
  def linear(field,coefficient):return mat(pair(field.reshape(field.shape[0]*field.shape[1],field.shape[-1])),coefficient)
  def integral(left,right,measure,*,conjugate=False,components=2):
   ll=tuple(v.conj().T if conjugate else v[:,rev].T for v in left)
   weight=tuple(np.repeat(v,components)[:,None] for v in D(measure).pair)
   return mat(ll,cp(weight,right))
  for tri,local in zip(obj.data['triangles'],info['local']):
   p=zero[tri[:3]];dp=ds*velocity[tri[:3]]
   area,slope=triangle_measures(p,velocity[tri[:3]])
   if local[11]:
    j1=np.column_stack([velocity[tri[1]]-velocity[tri[0]],velocity[tri[2]]-velocity[tri[0]]]);inv=np.array([[j1[1,1],-j1[0,1]],[-j1[1,0],j1[0,0]]])/local[10];dinv=np.zeros_like(inv)
   else:
    jac=np.column_stack([p[1]-p[0],p[2]-p[0]]);dj=np.column_stack([dp[1]-dp[0],dp[2]-dp[0]]);inv=np.linalg.inv(jac);dinv=-inv@dj@inv
   for bary,weight in quadrature(fields,tri,order):
    row,_,_,_,_,added=next(rows);_,ci,zi,w,dw,ep,dep,_,_,gc,dgc,gz,*_=row
    for first in range(0,len(bary),512):
     sl=slice(first,first+512);sample=common.sample(tri,bary[sl],inv,dinv);ph,dph=phase_values(bary[sl],p,dp,modes,chart[4])
     e=add(retained_field(ph,field_pair),combined(sample,0));trace=add(linear(gz[sl],tuple(v[nc+zi] for v in coefficients)),linear(added[sl],extra))
     if local[11]:
      e=add(e,trace)
      delta=add(delta,cp(pair(ds),cp(pair(ep),integral(e,e,weight.exact[sl]*slope))))
     else:
      de=add(retained_field(dph,field_pair),retained_field(ph,change_pair),combined(sample,2),cp(pair(ds),trace))
      ww=weight.exact[sl]*area;dww=weight.exact[sl]*D(ds)*slope
      for component in range(2):
       physical=tuple(v.reshape(-1,2,size)[:,component] for v in e)
       mean=mat(tuple(v.conj().T for v in ph),cp(tuple(v[:,None] for v in ww.pair),physical))
       retained=add_at(retained,(slice(component*N,(component+1)*N),slice(None)),mean)
      value=add(value,cp(pair(ep),integral(e,e,ww,conjugate=True)))
      delta=add(delta,cp(pair(dep),integral(e,e,ww)),cp(pair(ep),integral(e,e,dww)),cp(pair(ep),integral(de,e,ww)),cp(pair(ep),integral(e,de,ww)))
  for ci,ep,p,gc,weight,edge,tri,bary,inverse in edge_rows(obj,settings,base,chart,order,coordinates=True):
   sample=common.sample(tri,bary,inverse,np.zeros_like(inverse));ph,_=phase_values(bary,zero[tri[:3]],np.zeros((3,2)),modes,chart[4]);e=add(retained_field(ph,field_pair),combined(sample,0));ey=tuple(v.reshape(-1,2,size)[:,1] for v in e)
   j=mat(mat(pair(chart[8]._basis(np.array([edge]),chart[9])[0]),pair(transform[2])),extra);jump=tuple(np.broadcast_to(v,(len(weight),size)) for v in j)
   delta=add(delta,cp(pair(ds),cp(pair(ep),integral(ey,jump,D(weight),components=1))),cp(pair(ds),cp(pair(ep),integral(jump,ey,D(weight),components=1))))
  return value[0]+value[1],delta[0]+delta[1],retained[0]+retained[1]
 return energy
