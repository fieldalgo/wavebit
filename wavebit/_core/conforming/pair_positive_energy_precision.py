"""Independent physical energy and retained Fourier integrals at positive gaps.

Reconstruct the actual E with every complete coefficient. Its stationary
variation holds auxiliary potentials fixed and receives no candidate H, DH,
solved coefficient derivative or assembled weak matrix. The retained mean is
also sampled directly; no rounded Fourier constraint matrix is reused.
"""
import numpy as np
from .pair_contact_precision import pair,mat,add,cp,transpose

def verify(base,unsplit_rows,split_rows,raw_fields,coefficients,field,field_change):
 info=base[-1];nc,nz,N=(info[k] for k in ('nc','nz','size'));size=2*N;rev=info['reverse']
 energy=pair(np.zeros((size,size),complex));derivative=pair(np.zeros((size,size),complex));retained=pair(np.zeros((size,size),complex))
 extra=tuple(v[nc+nz:] for v in coefficients)
 def integral(left,right,w,conjugate=False):
  return mat(transpose(left,True) if conjugate else tuple(v[:,rev].T for v in left),cp(w,right))
 def integrate(w,dw,ep,dep,a,da,b,db):
  value=cp(pair(ep),integral(a,b,w,True))
  delta=add(cp(pair(ep),integral(a,b,dw)),cp(pair(ep),integral(da,b,w)),cp(pair(ep),integral(a,db,w)))
  if dep!=0:delta=add(delta,cp(pair(dep),integral(a,b,w)))
  return value,delta
 for divided,rows in ((False,unsplit_rows()),(True,split_rows())):
  for row in rows:
   for first in range(0,len(row[3]),512):
    indices,w,dw,ep,dep,phi,dphi,g,dg,le,dle=raw_fields(row,slice(first,first+512))
    local=tuple(v[indices] for v in coefficients)
    E=add(mat(phi,field),mat(g,local));DE=add(mat(dphi,field),mat(phi,field_change),mat(dg,local))
    if divided:
     a=mat(le,extra);da=mat(dle,extra);terms=((E,DE,a,da),(a,da,E,DE),(a,da,a,da));mean=a
    else:terms=((E,DE,E,DE),);mean=E
    retained=add(retained,mat(transpose(phi,True),cp(w,mean)))
    for a,da,b,db in terms:
     v,dv=integrate(w,dw,ep,dep,a,da,b,db);energy=add(energy,v);derivative=add(derivative,dv)
 return energy[0]+energy[1],derivative[0]+derivative[1],retained[0]+retained[1]
