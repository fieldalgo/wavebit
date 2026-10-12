"""Compensated real geometry arithmetic for complete corner coordinates.

This NumPy-only evaluator keeps high/low components through coordinate
composition. Original physical residuals continue to use their own complete
three-chunk products. Small Bernstein products use exact leading integer
partial sums and every cross product with the complete floating remainder.
"""
from decimal import Decimal,localcontext
import numpy as np
from . import pair_oscillatory_radial as arithmetic
from .pair_residual_arithmetic import _real_product,_sum_pairs,_buffered_sum

class D:
    __array_priority__=10000
    def __init__(self,hi,lo=None):
        if isinstance(hi,D):self.hi,self.lo=hi.hi,hi.lo
        else:
            self.hi=np.asarray(hi,float)
            self.lo=np.zeros_like(self.hi) if lo is None else np.asarray(lo,float)
    @property
    def pair(self):return self.hi,self.lo
    @property
    def T(self):return D(self.hi.T,self.lo.T)
    def __len__(self):return len(self.hi)
    def __getitem__(self,key):return D(self.hi[key],self.lo[key])
    def __setitem__(self,key,value):
        value=D(value);self.hi[key]=value.hi;self.lo[key]=value.lo
    def copy(self):return D(self.hi.copy(),self.lo.copy())
    def reshape(self,*shape):return D(self.hi.reshape(*shape),self.lo.reshape(*shape))
    def __add__(self,other):return D(*arithmetic.add(self.pair,D(other).pair))
    __radd__=__add__
    def __neg__(self):return D(-self.hi,-self.lo)
    def __sub__(self,other):return self+-D(other)
    def __rsub__(self,other):return D(other)+-self
    def __mul__(self,other):
        other=D(other);hi,lo=arithmetic.two_product(self.hi,other.hi)
        return D(*arithmetic.add((hi,lo),(self.hi*other.lo+self.lo*other.hi,self.lo*other.lo)))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=D(other);q=self.hi/other.hi
        remainder=self-D(q)*other
        return D(q)+D((remainder.hi+remainder.lo)/other.hi)
    def __rtruediv__(self,other):return D(other)/self
    def __pow__(self,power):
        assert isinstance(power,(int,np.integer)) and power>=0
        value=D(np.ones_like(self.hi))
        for _ in range(power):value=value*self
        return value
    def __matmul__(self,other):
        other=D(other);a,b=self.hi,other.hi
        left=a.ndim==1;right=b.ndim==1
        aa=self.reshape(1,-1) if left else self
        bb=other.reshape(-1,1) if right else other
        out=D(*_sum_pairs([_real_product(aa.hi,bb.hi),
                          (aa.hi@bb.lo+aa.lo@bb.hi,aa.lo@bb.lo)]))
        if left and right:return out[0,0]
        if left:return out[0]
        if right:return out[:,0]
        return out
    def __rmatmul__(self,other):return D(other)@self
    def __ne__(self,other):return self.hi+self.lo!=D(other).hi+D(other).lo
    def __le__(self,other):return self.hi+self.lo<=D(other).hi+D(other).lo
    def sum(self,axis):
        hi=np.moveaxis(self.hi,axis,0);lo=np.moveaxis(self.lo,axis,0)
        value=D(np.zeros_like(hi[0]))
        for a,b in zip(hi,lo):value=value+D(a,b)
        return value
    @staticmethod
    def zeros(shape):return D(np.zeros(shape))
    @staticmethod
    def zeros_like(a):return D(np.zeros_like(D(a).hi))
    @staticmethod
    def broadcast(a,shape):return D(np.broadcast_to(D(a).hi,shape),np.broadcast_to(D(a).lo,shape))


def simplex_coordinates(barycentric,tangent):
    """Preserve small coordinates and the exact affine partition of unity.

    Rounded quadrature triples need not sum to one in compensated arithmetic.
    Homogenized polynomials and their radial expressions agree on the unit
    simplex, so retain the two smaller coordinates and reconstruct the largest
    as their compensated complement. Its variation is the negative sum of the
    other two variations. This changes no potential or physical derivative.
    """
    anchor=np.argmax(abs(barycentric),axis=1)
    values=[D(barycentric[:,i].copy()) for i in range(3)]
    changes=[D(tangent[:,i].copy()) for i in range(3)]
    for i in range(3):
        rows=np.flatnonzero(anchor==i)
        if len(rows):
            values[i][rows]=D(1.)-sum(values[j][rows] for j in range(3) if j!=i)
            changes[i][rows]=-sum(changes[j][rows] for j in range(3) if j!=i)
    return values,changes


with localcontext() as context:
    context.prec=90
    ln2=Decimal(2).ln();high=float(ln2)
    LN2=D(high,float(ln2-Decimal.from_float(high)))

def logarithm(value):
    value=D(value);_,exponent=np.frexp(value.hi)
    mantissa=D(np.ldexp(value.hi,-exponent),np.ldexp(value.lo,-exponent))
    lower=mantissa.hi<np.sqrt(.5)
    factor=np.where(lower,2.,1.)
    mantissa=mantissa*factor;exponent=exponent-lower
    z=(mantissa-1)/(mantissa+1);square=z*z;term=z;total=z
    for n in range(1,25):
        term=term*square;total=total+term/(2*n+1)
    return 2*total+exponent*LN2


def real_product(a,b):
    if not 0<a.shape[1]<=28:raise ValueError('corner geometry products require 1 through 28 Bernstein columns')
    ea=np.frexp(np.max(abs(a),axis=1))[1];eb=np.frexp(np.max(abs(b),axis=0))[1]
    x=np.ldexp(a,-ea[:,None]);y=np.ldexp(b,-eb[None,:])
    bits=(50-(a.shape[1]-1).bit_length())//2;p=float(2**bits)
    def split(z):
        first=np.rint(z*p);rest=z*p-first
        second=np.rint(rest*p);third=rest*p-second
        return first,second,third
    xx,yy=split(x),split(y);exponents=(bits,2*bits,2*bits)
    def terms():
        for i,j in sorted(((i,j) for i in range(3) for j in range(3)),key=lambda ij:sum(ij)):
            term=np.ldexp(xx[i]@yy[j],-exponents[i]-exponents[j])
            yield term,np.zeros_like(term)
    hi,lo=_buffered_sum(terms());exponent=ea[:,None]+eb[None,:]
    return np.ldexp(hi,exponent),np.ldexp(lo,exponent)

def product(a,b):
    a,b=D(a),D(b)
    return D(*_sum_pairs([real_product(a.hi,b.hi),
                         (a.hi@b.lo+a.lo@b.hi,a.lo@b.lo)]))
