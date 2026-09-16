"""Independent physical contact energy from complete high/low field coordinates.

The verifier reconstructs all retained, common and disappearing fields before
forming pointwise energy and its stationary physical variation. Its inputs
contain no candidate tensor, tensor derivative or solved common variation.
The same reconstructed field supplies its directly integrated Fourier mean
and the physical variation of that mean, including collapsed cells and edges.
"""
import numpy as np
from .pair_contact_precision import pair,mat,add,cp,retained_field,add_at
from .pair_corner_precision import D
from .pair_contact_quadrature import quadrature,phase_values,triangle_measures,contact_maps
from .pair_corner_common_precision import CommonCoordinates
from .pair_contact_normal_precision import contact_normals,trace_values

def verifier(obj,settings,base,transform,chart,order,records,edge_rows):
 info=base[-1];nc,nz,N,ds=(info[k] for k in ('nc','nz','size','ds'));fields=info['corner_fields'];size=2*N;rev=info['reverse']
 nx,ny=obj.order;modes=np.array([(i,j) for j in range(-ny,ny+1) for i in range(-nx,nx+1)])
 _,_,zero,velocity=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
 provider=getattr(fields,'common_coordinate_type',CommonCoordinates)
 def energy(field,field_change,potential):
  coefficients=potential;field_pair=field;change_pair=field_change
  common=provider(fields,D(*(np.hstack([v[:nc].real,v[:nc].imag]) for v in coefficients)))
  value=pair(np.zeros((size,size),complex));delta=pair(np.zeros((size,size),complex));extra=tuple(v[nc+nz:] for v in coefficients);rows=iter(records())
  retained=pair(np.zeros((size,size),complex));dretained=pair(np.zeros((size,size),complex))
  def combined(sample,offset):
   return tuple(np.stack([getattr(v,part)[:,:size]+1j*getattr(v,part)[:,size:] for v in sample[offset:offset+2]],axis=1).reshape(-1,size) for part in ('hi','lo'))
  def linear(field,coefficient):return mat(field.reshape(field.hi.shape[0]*field.hi.shape[1],field.hi.shape[-1]).pair,coefficient)
  def integral(left,right,measure,*,conjugate=False,components=2):
   ll=tuple(v.conj().T if conjugate else v[:,rev].T for v in left)
   weight=tuple(np.repeat(v,components)[:,None] for v in D(measure).pair)
   return mat(ll,cp(weight,right))
  def mean(phase,physical,measure):
   parts=[]
   for component in range(2):
    block=tuple(v.reshape(-1,2,size)[:,component] for v in physical)
    parts.append(mat(tuple(v.conj().T for v in phase),cp(tuple(v[:,None] for v in measure.pair),block)))
   return tuple(np.vstack([block[word] for block in parts]) for word in range(2))
  for tri,tag,local in zip(obj.data['triangles'],obj.data['tags'],info['local']):
   p=zero[tri[:3]];dp=D(ds)*D(velocity[tri[:3]])
   area,slope=triangle_measures(p,velocity[tri[:3]])
   inv,dinv=contact_maps(p,velocity[tri[:3]],ds,local[11])
   for bary,weight in quadrature(fields,tri,order):
    row,_,_,_,_,added=next(rows);_,ci,zi,w,dw,ep,dep,_,_,gc,dgc,gz,*_=row
    gz,added=contact_normals(obj,transform,chart,tri,tag,local,bary,zero,velocity)
    for first in range(0,len(bary),512):
     sl=slice(first,first+512);sample=common.sample(tri,bary[sl],inv,dinv);ph,dph=phase_values(bary[sl],p,dp,modes,chart[4])
     e=add(retained_field(ph,field_pair),combined(sample,0));trace=add(linear(gz[sl],tuple(v[nc+zi] for v in coefficients)),linear(added[sl],extra))
     if local[11]:
      e=add(e,trace)
      delta=add(delta,cp(pair(ds),cp(pair(ep),integral(e,e,weight.exact[sl]*slope))))
      dretained=add(dretained,mean(ph,e,weight.exact[sl]*D(ds)*slope))
     else:
      de=add(retained_field(dph,field_pair),retained_field(ph,change_pair),combined(sample,2),cp(pair(ds),trace))
      ww=weight.exact[sl]*area;dww=weight.exact[sl]*D(ds)*slope
      retained=add(retained,mean(ph,e,ww))
      dretained=add(dretained,mean(dph,e,ww),mean(ph,de,ww),mean(ph,e,dww))
      value=add(value,cp(pair(ep),integral(e,e,ww,conjugate=True)))
      delta=add(delta,cp(pair(dep),integral(e,e,ww)),cp(pair(ep),integral(e,e,dww)),cp(pair(ep),integral(de,e,ww)),cp(pair(ep),integral(e,de,ww)))
  for ci,ep,p,gc,weight,edge,tri,bary,inverse in edge_rows(obj,settings,base,chart,order,coordinates=True):
   inverse,dinverse=contact_maps(zero[tri[:3]],np.zeros((3,2)),0.)
   sample=common.sample(tri,bary,inverse,dinverse);ph,_=phase_values(bary,zero[tri[:3]],np.zeros((3,2)),modes,chart[4]);e=add(retained_field(ph,field_pair),combined(sample,0));ey=tuple(v.reshape(-1,2,size)[:,1] for v in e)
   j=mat(mat(trace_values(chart[8],chart[9],D(np.array([edge])))[0].pair,pair(transform[2])),extra);jump=tuple(np.broadcast_to(v,(len(weight),size)) for v in j)
   delta=add(delta,cp(pair(ds),cp(pair(ep),integral(ey,jump,D(weight),components=1))),cp(pair(ds),cp(pair(ep),integral(jump,ey,D(weight),components=1))))
   moment=mat(tuple(v.conj().T for v in ph),cp((D(ds)*D(weight)[:,None]).pair,jump))
   dretained=add_at(dretained,(slice(N,2*N),slice(None)),moment)
  return value[0]+value[1],delta[0]+delta[1],retained[0]+retained[1],dretained[0]+dretained[1]
 return energy
