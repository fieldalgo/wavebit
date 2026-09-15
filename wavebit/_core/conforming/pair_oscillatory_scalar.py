"""Compensated scalar evaluation of the same real exponent matrix columns.

A convergent exact-coefficient series handles small corner hats. Elsewhere,
compensated scalar powers retain complete polynomial remainders and both
logarithmic derivatives before division by the physical complement. The same
final column normalization as the matrix evaluator is applied after rounding.
Only material-independent scalar coefficients are cached in this module.
"""
from functools import lru_cache
from fractions import Fraction as F
from math import factorial
import numpy as np
from .pair_oscillatory_radial import two_sum,two_product,add
from .pair_oscillatory_coefficients import coefficients,NODES,div,mul

def cp(a,b):
    rr=two_product(a[0].real,b[0].real);ii=two_product(a[0].imag,b[0].imag)
    ri=two_product(a[0].real,b[0].imag);ir=two_product(a[0].imag,b[0].real)
    re=add(rr,tuple(-v for v in ii));im=add(ri,ir)
    return add((re[0]+1j*im[0],re[1]+1j*im[1]),
               (a[0]*b[1]+a[1]*b[0],a[1]*b[1]))

def exact(pair):
    hi=complex(*(float(v) for v in pair))
    lo=complex(float(pair[0]-F.from_float(hi.real)),float(pair[1]-F.from_float(hi.imag)))
    return np.asarray(hi),np.asarray(lo)

def real(value):return np.asarray(value,dtype=complex),np.zeros_like(value,dtype=complex)
def minus(a):return tuple(-v for v in a)

def divide(a,b):
    q=a[0]/b;r=add(a,minus(cp(real(q),real(b))))
    return two_sum(q,(r[0]+r[1])/b)

def exp_pair(argument):
    bound=abs(argument[0]);powers=np.zeros(bound.shape,int);large=bound>.25
    powers[large]=np.ceil(np.log2(bound[large]/.25)).astype(int)
    b=tuple(np.ldexp(v.real,-powers)+1j*np.ldexp(v.imag,-powers) for v in argument)
    value=real(np.ones_like(bound));term=value
    for n in range(1,27):
        term=divide(cp(term,b),float(n));value=add(value,term)
    for n in range(int(powers.max(initial=0))):
        active=powers>n;squared=cp(tuple(v[active] for v in value),tuple(v[active] for v in value))
        for array,changed in zip(value,squared):array[active]=changed
    return value

def powers(h,s):
    log=np.empty_like(s);low=h<=.5
    log[low]=np.log1p(-h[low]);log[~low]=np.log(s[~low])
    values=[exp_pair(cp(exact(node),real(log))) for node in NODES[:3]]
    values.append(tuple(v.conj() for v in values[-1]))
    return values,exp_pair(real(log))

def polynomial(h,coeff,derivative=0):
    out=real(np.zeros_like(h[0]))
    for j in range(len(coeff)-1,derivative-1,-1):
        c=coeff[j]
        for d in range(derivative):c=mul(c,(F(j-d),F(0)))
        out=add(cp(out,h),exact(c))
    return out

class Double:
    __array_priority__=10000
    def __init__(self,hi,lo=None):
        if isinstance(hi,Double):self.hi,self.lo=hi.hi,hi.lo
        else:
            self.hi=np.asarray(hi,float)
            self.lo=np.zeros_like(self.hi) if lo is None else np.asarray(lo,float)
    @property
    def pair(self):return self.hi,self.lo
    def __getitem__(self,key):return Double(self.hi[key],self.lo[key])
    def __add__(self,other):return Double(*add(self.pair,Double(other).pair))
    __radd__=__add__
    def __neg__(self):return Double(-self.hi,-self.lo)
    def __sub__(self,other):return self+-Double(other)
    def __rsub__(self,other):return Double(other)+-self
    def __mul__(self,other):
        other=Double(other);hi,lo=two_product(self.hi,other.hi)
        return Double(*add((hi,lo),(self.hi*other.lo+self.lo*other.hi,self.lo*other.lo)))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=Double(other);q=self.hi/other.hi
        remainder=self-Double(q)*other
        return Double(q)+Double((remainder.hi+remainder.lo)/other.hi)
    def __rtruediv__(self,other):return Double(other)/self
    def __pow__(self,power):
        assert isinstance(power,(int,np.integer)) and power>=0
        value=Double(np.ones_like(self.hi))
        for _ in range(power):value=value*self
        return value

@lru_cache(None)
def series_coefficients(k,r):
    columns=[[F(0)]*r+list(coefficients(k,r,count,96)[-1]) for count in range(1,5)]
    hi=np.array([[float(column[n]) for column in columns] for n in range(r+96)])
    lo=np.array([[float(column[n]-F.from_float(float(column[n]))) for column in columns] for n in range(r+96)])
    return hi,lo

def family(h,s,k,r,spectral):
    small=h<=.25;out=[np.zeros((4,len(h))) for _ in range(3)]
    if np.any(small):
        x,y=h[small],s[small]
        _,complement=spectral[0]
        comp=Double(complement[0].real,complement[1].real)
        hat=Double(np.ones_like(x))-comp
        hi,lo=series_coefficients(k,r)
        a=Double(np.zeros((len(x),4)));b=a;c=a
        for ph,pl in zip(hi[::-1],lo[::-1]):
            c=c*hat[:,None]+2*b
            b=b*hat[:,None]+a
            a=a*hat[:,None]+Double(ph,pl)
        sp=comp**k
        value=sp[:,None]*a
        first=sp[:,None]*(comp[:,None]*b-k*a)/Double(y)[:,None]
        second=sp[:,None]*(comp[:,None]**2*c-2*k*comp[:,None]*b+k*(k-1)*a)/Double(y)[:,None]/Double(y)[:,None]
        canonical=np.array([(-1.)**(r+count-1)*factorial(r+count-1) for count in range(1,5)])
        for target,v in zip(out,(value,first,second)):
            v=v/Double(canonical)
            target[:,small]=((v.hi+v.lo)*canonical).T
    if np.any(~small):
        x,y=h[~small],s[~small]
        es,complement=spectral[1]
        hat=add(real(np.ones_like(x)),minus(complement));fields=[]
        taylor=coefficients(k,r,4,96)[2]
        for node,t,e in zip(NODES,taylor,es):
            alpha=exact(node);v=e;d=minus(cp(alpha,e))
            dd=cp(cp(alpha,add(alpha,real(-1.))),e)
            if r:
                sp=real(np.ones_like(y))
                for _ in range(k):sp=cp(sp,complement)
                p,dp,ddp=(polynomial(hat,t,j) for j in range(3))
                v=add(v,minus(cp(sp,p)))
                d=add(add(d,cp(real(float(k)),cp(sp,p))),minus(cp(cp(sp,dp),complement)))
                dd=add(add(dd,cp(real(float(2*k)),cp(cp(sp,dp),complement))),
                       minus(cp(cp(cp(sp,ddp),complement),complement)))
                if k>1:dd=add(dd,minus(cp(real(float(k*(k-1))),cp(sp,p))))
            fields.append((v,d,dd))
        for count in range(1,5):
            weights,lead,_,scale,_=coefficients(k,r,count,96)
            rows=[real(np.zeros_like(x)) for _ in range(3)]
            for w,l,values in zip(weights,lead,fields):
                weight=exact(div(mul((scale,F(0)),w),l))
                for i,value in enumerate(values):rows[i]=add(rows[i],cp(weight,value))
            rows[1]=divide(rows[1],y);rows[2]=divide(divide(rows[2],y),y)
            canonical=(-1.)**(r+count-1)*factorial(r+count-1)
            for target,row in zip(out,rows):
                row=divide(row,canonical)
                target[count-1,~small]=(row[0]+row[1]).real*canonical
    return tuple(out)

def values(h,s,degree):
    small=h<=.25
    # The small-hat series needs only the shared complement. Its real
    # coefficients already contain every radial node, so evaluating those
    # separate spectral powers would produce unused arrays.
    spectral=((None,exp_pair(real(np.log1p(-h[small])))) if np.any(small) else None,
              powers(h[~small],s[~small]) if np.any(~small) else None)
    return np.asarray([np.asarray(family(h,s,k,degree-k,spectral)).transpose(0,2,1)
                       for k in range(1,degree+1)])
