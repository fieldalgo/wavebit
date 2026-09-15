"""Complete positive-gap projection with a material-independent geometric frame.

All physical K, R and Fourier moments are integrated before subtraction.
The complete projected geometric Gram supplies an invertible coordinate
change; strict positive Cholesky pivots reject failure without shifting the
metric or removing a field. Values and fixed-coordinate physical variations
remain compensated through the same weak solve.
"""
import numpy as np
from .pair_contact_precision import pair,mat,cp,add,neg,transpose,solve,add_at
from .pair_corner_precision import D
from .projected_pair_contact import _real_geometry



def cholesky(matrix):
 """Factor every coordinate with positive pivots and complete residuals."""
 matrix=D(matrix)
 if matrix.hi.ndim!=2 or matrix.hi.shape[0]!=matrix.hi.shape[1]:raise ValueError('square metric required')
 n=matrix.hi.shape[0]
 out=D.zeros((n,n))
 for column in range(n):
  value=matrix[column:,column]
  if column:value=value-out[column:,:column]@out[column,:column]
  pivot=value[0]
  if not np.isfinite(pivot.hi+ pivot.lo) or pivot.hi+pivot.lo<=0:raise ArithmeticError('complete metric is not positive definite')
  root=D(np.sqrt(pivot.hi));root=root+(pivot-root*root)/(2*root)
  out[column,column]=root
  if column+1<n:out[column+1:,column]=value[1:]/root
 return out


def assemble(base,unsplit_rows,split_rows,fields):
 info=base[-1];nc,nz,N=(info[k] for k in ('nc','nz','size'));size=2*N
 ne=2*info['corner_fields'].obj.order[1]+1;total=nc+nz+ne
 K=pair(np.zeros((total,total),complex));DK=pair(np.zeros((total,total),complex))
 R=pair(np.zeros((size,total),complex));DR=pair(np.zeros((size,total),complex))
 A=pair(np.zeros((size,size),complex));DA=pair(np.zeros((size,size),complex))
 Q=pair(np.zeros((size,total),complex));DQ=pair(np.zeros((size,total),complex))
 G=pair(np.zeros((total,total),complex));F=pair(np.zeros((size,size),complex))
 def products(w,dw,ep,dep,left,dleft,right,dright,conjugate=False):
  ll=transpose(left,conjugate);dl=transpose(dleft,conjugate);wr=cp(w,right)
  raw=mat(ll,wr);value=cp(pair(ep),raw)
  tangent=add(cp(pair(ep),mat(ll,cp(dw,right))),cp(pair(ep),mat(dl,wr)),cp(pair(ep),mat(ll,cp(w,dright))))
  if dep!=0:tangent=add(tangent,cp(pair(dep),raw))
  return value,tangent
 for divided,rows in ((False,unsplit_rows()),(True,split_rows())):
  for row in rows:
   for first in range(0,len(row[3]),512):
    indices,w,dw,ep,dep,ph,dph,g,dg,le,dle=fields(row,slice(first,first+512))
    if not divided:
     v,dv=products(w,dw,1.,0.,g,dg,g,dg);G=add_at(G,np.ix_(indices,indices),v);K=add_at(K,np.ix_(indices,indices),cp(pair(ep),v));DK=add_at(DK,np.ix_(indices,indices),cp(pair(ep),dv),cp(pair(dep),v))
     v,dv=products(w,dw,1.,0.,ph,dph,g,dg,True);Q=add_at(Q,(slice(None),indices),v);DQ=add_at(DQ,(slice(None),indices),dv);R=add_at(R,(slice(None),indices),cp(pair(ep),v));DR=add_at(DR,(slice(None),indices),cp(pair(ep),dv),cp(pair(dep),v))
     v,dv=products(w,dw,1.,0.,ph,dph,ph,dph,True);F=add(F,v);A=add(A,cp(pair(ep),v));DA=add(DA,cp(pair(ep),dv),cp(pair(dep),v))
    else:
     extra=np.arange(nc+nz,total)
     v,dv=products(w,dw,1.,0.,le,dle,g,dg)
     G=add_at(G,np.ix_(extra,indices),v);G=add_at(G,np.ix_(indices,extra),transpose(v))
     kv=cp(pair(ep),v);kd=add(cp(pair(ep),dv),cp(pair(dep),v))
     K=add_at(K,np.ix_(extra,indices),kv);K=add_at(K,np.ix_(indices,extra),transpose(kv))
     DK=add_at(DK,np.ix_(extra,indices),kd);DK=add_at(DK,np.ix_(indices,extra),transpose(kd))
     v,dv=products(w,dw,1.,0.,le,dle,le,dle);G=add_at(G,np.ix_(extra,extra),v);K=add_at(K,np.ix_(extra,extra),cp(pair(ep),v));DK=add_at(DK,np.ix_(extra,extra),cp(pair(ep),dv),cp(pair(dep),v))
     v,dv=products(w,dw,1.,0.,ph,dph,le,dle,True);Q=add_at(Q,(slice(None),extra),v);DQ=add_at(DQ,(slice(None),extra),dv);R=add_at(R,(slice(None),extra),cp(pair(ep),v));DR=add_at(DR,(slice(None),extra),cp(pair(ep),dv),cp(pair(dep),v))
 return K,DK,R,DR,A,DA,Q,DQ,G,F


def response(base,blocks):
 info=base[-1];nc,s,rev=(info[k] for k in ('nc','s','reverse'));K,DK,R,DR,A,DA=blocks[:6]
 Q,DQ=blocks[6:8];QT=transpose(Q,True);DQT=transpose(DQ,True)
 S=tuple(v[rev].T for v in R);DS=tuple(v[rev].T for v in DR)
 M=add(K,neg(mat(S,Q)),neg(mat(QT,add(R,neg(mat(A,Q))))))
 DM=add(DK,neg(mat(DS,Q)),neg(mat(S,DQ)),neg(mat(DQT,R)),neg(mat(QT,DR)),mat(mat(DQT,A),Q),mat(mat(QT,DA),Q),mat(mat(QT,A),DQ))
 SC=add(S,neg(mat(QT,A)));DSC=add(DS,neg(mat(DQT,A)),neg(mat(QT,DA)))
 G,F=blocks[8:10]
 geometry=add(G,neg(mat(tuple(v[rev].T for v in Q),Q)),neg(mat(QT,Q)),mat(mat(QT,F),Q))
 _real_geometry(geometry[0]+geometry[1])
 lower=cholesky(D(*(v.real for v in geometry)))
 T=pair(np.linalg.solve(lower.hi,np.eye(len(lower.hi))).T);TT=transpose(T)
 mm=mat(TT,mat(M,T));ss=mat(TT,SC);u=solve(mm,neg(ss));U=mat(T,u)
 DU=mat(T,solve(mm,neg(mat(TT,add(DSC,mat(DM,U))))))
 x=add(pair(np.eye(len(A[0]))),neg(mat(Q,U)));dxgeo=neg(mat(DQ,U));dx=add(dxgeo,neg(mat(Q,DU)))
 H=add(A,mat(tuple(v[:,rev].T for v in SC),U))
 DH=add(DA,mat(tuple(v[:,rev].T for v in DSC),U),mat(tuple(v[:,rev].T for v in U),DSC),mat(mat(tuple(v[:,rev].T for v in U),DM),U))
 condition=float(np.linalg.cond(mm[0]+mm[1]))
 return (U,DU,x,dx,H,DH),dxgeo,condition


def coupled(obj,settings,base,order,transform,ordinary,normalized,free):
 """Solve and independently verify every complete positive-gap coordinate."""
 from .pair_positive_fields import builder,unsplit
 from .pair_positive_energy_precision import verify
 from .pair_junction_quadrature import quadrature
 from .real_trace_coupling import coupled_positive
 fields=builder(obj,settings,base)
 whole=lambda:unsplit(obj,settings,base,order)
 divided=lambda:quadrature(obj,base,settings,order,transform,True,True,coordinate_records=True)
 blocks=assemble(base,whole,divided,fields)
 state,dxgeo,condition=response(base,blocks)
 U,DU,x,dx,H,DH=state
 def solved(*unused):return (*(hi+lo for hi,lo in state),condition)
 def energy(field,change,potential):
  return verify(base,whole,divided,fields,potential,field,change)
 return coupled_positive(obj,base,ordinary,free,obj.order[1],
                         response_solver=solved,normalized_quadrature=normalized,
                         energy_verifier=energy,energy_coordinates=(x,dxgeo,U))
