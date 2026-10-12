"""Complete triangle exponential moments through a scaled triangular matrix."""
import numpy as np
from .pair_complex_precision import C

class J:
    def __init__(self,value,tangent=0):
        if isinstance(value,J):self.v,self.d=value.v,value.d
        else:self.v,self.d=C(value),C(tangent)
    def __add__(self,other):
        other=J(other);return J(self.v+other.v,self.d+other.d)
    __radd__=__add__
    def __neg__(self):return J(-self.v,-self.d)
    def __sub__(self,other):return self+-J(other)
    def __mul__(self,other):
        other=J(other);return J(self.v*other.v,self.d*other.v+self.v*other.d)
    __rmul__=__mul__
    def __truediv__(self,other):return J(self.v/other,self.d/other)
    def __getitem__(self,key):return J(self.v[key],self.d[key])

def product(a,b):
    return [a[0]*b[0],a[0]*b[1]+a[1]*b[3],a[0]*b[2]+a[1]*b[4]+a[2]*b[5],a[3]*b[3],a[3]*b[4]+a[4]*b[5],a[5]*b[5]]

def moment(phase,velocity):
    """Return integral_ref exp(i phase dot bary), and its exact physical JVP.

    exp([[z0,1,0],[0,z1,1],[0,0,z2]])[0,2] is the complete second
    divided difference, including coincident phases. Scaling and squaring
    evaluates it without subtracting nearly equal exponentials or frequencies.
    """
    z=[J(1j*C(phase[i]),1j*C(velocity[i])) for i in range(3)]
    a,b=z[1]-z[0],z[2]-z[0]
    bound=np.maximum(np.maximum(abs(a.v.hi),abs(b.v.hi)),1.)
    powers=np.maximum(0,np.ceil(np.log2(bound/.125)).astype(int));scale=np.ldexp(np.ones_like(bound),-powers)
    zero=J(np.zeros_like(bound));one=J(np.ones_like(bound))
    B=[zero,J(scale),zero,a*J(scale),J(scale),b*J(scale)]
    total=[one,zero,zero,one,zero,one];term=total
    for n in range(1,28):
        term=[value/n for value in product(term,B)]
        total=[v+t for v,t in zip(total,term)]
    for index in range(int(powers.max(initial=0))):
        active=powers>index;squared=product([v[active] for v in total],[v[active] for v in total])
        for target,value in zip(total,squared):
            for name in ('v','d'):
                for part in ('hi','lo'):getattr(getattr(target,name),part)[active]=getattr(getattr(value,name),part)
    factor=z[0].v.exp();factor=J(factor,factor*z[0].d)
    result=factor*total[2]
    return result.v,result.d
