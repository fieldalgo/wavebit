"""Compensated Gaussian points, weights and phases for real contact equations.

Legendre roots and weights use their defining polynomial recurrence in the
standard-library decimal arithmetic. High/low radial maps preserve the same
integration rule through singular corner substitutions and affine partitions.
No physical fields or tensor entries are removed or projected.
"""
from decimal import Decimal,localcontext
from functools import lru_cache
import numpy as np
from .pair_corner_precision import D,simplex_coordinates
from . import pair_oscillatory_scalar as scalar

class Exact(np.ndarray):
    def __new__(cls,value):
        obj=(value.hi+value.lo).view(cls);obj.exact=value;return obj
    def __array_finalize__(self,obj):self.exact=getattr(obj,'exact',None)
    def __array_ufunc__(self,ufunc,method,*args,**kwargs):
        args=tuple(np.asarray(a) if isinstance(a,Exact) else a for a in args)
        if kwargs.get("out") is not None:kwargs["out"]=tuple(np.asarray(a) if isinstance(a,Exact) else a for a in kwargs["out"])
        return getattr(ufunc,method)(*args,**kwargs)
    def __getitem__(self,key):
        out=super().__getitem__(key)
        if isinstance(out,Exact):out.exact=self.exact[key]
        return out

def stack(values,axis=0):return D(np.stack([v.hi for v in values],axis),np.stack([v.lo for v in values],axis))
def convert(values):
    hi=np.array([float(v) for v in values]);lo=np.array([float(v-Decimal.from_float(float(v))) for v in values]);return D(hi,lo)

def opening_measure(ds,local):
    """Share the normalized disappearing-domain measure without early rounding.

    Collapsed energies carry ds outside their normalized d1 integral. Their
    free and common moments must retain the same product. Bulk records keep
    their existing stored measure variation.
    """
    return D(ds)*D(local[10]) if local[11] else D(ds*local[10])

@lru_cache(maxsize=16)
def gauss(n):
    with localcontext() as ctx:
        ctx.prec=90;initial=np.polynomial.legendre.leggauss(n)[0];nodes=[];weights=[]
        for guess in initial:
            x=Decimal.from_float(float(guess))
            for _ in range(5):
                a=Decimal(1);b=x
                for k in range(2,n+1):a,b=b,((2*k-1)*x*b-(k-1)*a)/k
                derivative=n*(a-x*b)/(1-x*x);x-=b/derivative
            a=Decimal(1);b=x
            for k in range(2,n+1):a,b=b,((2*k-1)*x*b-(k-1)*a)/k
            derivative=n*(a-x*b)/(1-x*x)
            nodes.append((1+x)/2);weights.append(1/((1-x*x)*derivative*derivative))
        result=convert(nodes),convert(weights)
        for value in result:
            value.hi.flags.writeable=False;value.lo.flags.writeable=False
        return result

def quadrature(self,tri,order):
    t,w=gauss(order);count=self.angular_quadrature or order;angle,aw=gauss(count)
    refs=D(np.eye(3));seam_hats=self.hats[tri[:3]][:,[5,8]]
    singular_edges=np.flatnonzero(np.sum(seam_hats==1,axis=0)==2)
    if self.junctions and len(singular_edges):
        edge=np.flatnonzero(seam_hats[:,singular_edges[0]]==1);other=next(i for i in range(3) if i not in edge)
        small=.5*t**self.root_order;measure=(self.root_order/2)*t**(self.root_order-1)
        for radial,complement in ((small,1-small),(1-small,small)):
            bary=D.zeros((order,count,3));bary[:,:,other]=D.broadcast(radial[:,None],(order,count));bary[:,:,edge[0]]=complement[:,None]*(1-angle);bary[:,:,edge[1]]=complement[:,None]*angle
            yield Exact(bary.reshape(-1,3)),Exact((complement[:,None]*measure[:,None]*w[:,None]*aw).reshape(-1))
        return
    singular=np.flatnonzero(np.any(self.hats[tri[:3]][:,self.controls]==1,axis=1))
    if len(singular)==1:
        sub=D(np.roll(refs.hi,-int(singular[0]),axis=0));subs=[(sub,D(1.))];power=self.root_order
    elif not len(singular):subs=[(refs,D(1.))];power=1
    else:
        center=D(np.ones(3))/3;subs=[];power=self.root_order
        for i in range(3):
            j,k=(i+1)%3,(i+2)%3
            subs.extend([(stack([refs[i],(refs[i]+refs[j])/2,center]),D(1.)/6),(stack([refs[i],center,(refs[i]+refs[k])/2]),D(1.)/6)])
    radial=t**power;dr=power*t**(power-1)
    uv=stack([D.broadcast(1-radial[:,None],(order,count)),radial[:,None]*(1-angle),radial[:,None]*angle],axis=-1).reshape(-1,3)
    weight=(radial[:,None]*dr[:,None]*w[:,None]*aw).reshape(-1)
    for sub,det in subs:yield Exact(uv@sub),Exact(weight*det)

def coordinates(bary):
    exact=getattr(bary,'exact',None)
    if exact is None:return simplex_coordinates(np.asarray(bary),np.zeros_like(bary))[0]
    # Keep the exact affine constraint after finite high/low polynomial products.
    anchor=np.argmax(abs(exact.hi),axis=1);out=[exact[:,i].copy() for i in range(3)]
    for i in range(3):
        rows=np.flatnonzero(anchor==i)
        if len(rows):out[i][rows]=D(1.)-sum(out[j][rows] for j in range(3) if j!=i)
    return out

with localcontext() as ctx:
    ctx.prec=90
    def atan(x):
        square=x*x;term=x;value=x
        for n in range(1,160):term=-term*square;value+=term/(2*n+1)
        return value
    pi=16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239)
    PI=convert([pi])[0]

def phases(bary,p,dp,modes,period):
    b=stack(coordinates(bary),1);argument=2*PI*((b@D(p))/D(period)@D(modes.T));delta=2*PI*((b@D(dp))/D(period)@D(modes.T))
    phase=scalar.exp_pair((1j*argument.hi,1j*argument.lo));dphase=scalar.cp((1j*delta.hi,1j*delta.lo),phase)
    n=len(modes);rows=len(bary)
    def lift(pair):
        out=[np.zeros((rows,2,2*n),complex) for _ in range(2)]
        for target,part in zip(out,pair):target[:,0,:n]=part;target[:,1,n:]=part
        return tuple(v.reshape(-1,2*n) for v in out)
    return lift(phase),lift(dphase)
