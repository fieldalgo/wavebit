"""Three-word geometry arithmetic for strongly cancelling trial potentials.

This internal NumPy evaluator extends the existing error-free sum/product
kernels. Every cross term survives before normalization. It is used only to
construct complete real fields; the physical equations keep their independent
high/low arithmetic and their original residual criteria.
"""
import numpy as np
from math import factorial
from decimal import Decimal, localcontext
from .pair_oscillatory_radial import two_sum,two_product
from .pair_corner_precision import D as Double
from .pair_complex_precision import C as ComplexDouble

def _renormalize(terms):
    shape=np.broadcast_shapes(*(np.shape(x) for x in terms));h=np.zeros(shape);m=h.copy();l=h.copy()
    for x in terms:
        h,e=two_sum(h,x);m,e=two_sum(m,e);l=l+e
    for _ in range(3):m,l=two_sum(m,l);h,m=two_sum(h,m)
    return h,m,l

class Triple:
    __array_priority__=10000
    def __init__(self,h,m=None,l=None):
        if isinstance(h,Triple):self.words=h.words
        elif isinstance(h,Double):self.words=(h.hi,h.lo,np.zeros_like(h.hi))
        else:
            h=np.asarray(h,float);self.words=(h,np.zeros_like(h) if m is None else np.asarray(m,float),np.zeros_like(h) if l is None else np.asarray(l,float))
    @property
    def T(self):return Triple(*(x.T for x in self.words))
    @property
    def hi(self):return self.words[0]
    @property
    def lo(self):return self.words[1]+self.words[2]
    @property
    def pair(self):return self.hi,self.lo
    def __getitem__(self,k):return Triple(*(x[k] for x in self.words))
    def __setitem__(self,k,v):
        for a,b in zip(self.words,Triple(v).words):a[k]=b
    def __len__(self):return len(self.hi)
    def reshape(self,*shape):return Triple(*(v.reshape(*shape) for v in self.words))
    def __add__(self,v):
        a,b=self.words,Triple(v).words
        h,e=two_sum(a[0],b[0]);m,f=two_sum(a[1],b[1]);m,g=two_sum(e,m)
        l=((a[2]+b[2])+f)+g
        for _ in range(3):m,l=two_sum(m,l);h,m=two_sum(h,m)
        return Triple(h,m,l)
    __radd__=__add__
    def __neg__(self):return Triple(*(-x for x in self.words))
    def __sub__(self,v):return self+-Triple(v)
    def __rsub__(self,v):return Triple(v)+-self
    def __mul__(self,v):
        a,b=self.words,Triple(v).words
        h,e=two_product(a[0],b[0]);p,dp=two_product(a[0],b[1]);q,dq=two_product(a[1],b[0])
        m,f=two_sum(e,p);m,g=two_sum(m,q)
        l=f+g+dp+dq+a[0]*b[2]+a[1]*b[1]+a[2]*b[0]+a[1]*b[2]+a[2]*b[1]+a[2]*b[2]
        for _ in range(3):m,l=two_sum(m,l);h,m=two_sum(h,m)
        return Triple(h,m,l)
    __rmul__=__mul__
    def __truediv__(self,v):
        v=Triple(v);v=Triple(*_renormalize(v.words));q=Triple(self.hi/v.hi)
        for _ in range(2):q=q+Triple((self-q*v).hi/v.hi)
        return q
    def __rtruediv__(self,v):return Triple(v)/self
    def __pow__(self,n):
        if not isinstance(n,(int,np.integer)) or n<0:raise ValueError('triple powers require nonnegative integers')
        out=Triple(np.ones_like(self.hi))
        for _ in range(n):out=out*self
        return out
    def __matmul__(self,v):
        v=Triple(v);left=self.hi.ndim==1;right=v.hi.ndim==1
        if self.hi.ndim not in (1,2) or v.hi.ndim not in (1,2):
            raise ValueError('triple products require vectors or matrices')
        if self.hi.shape[-1]!=v.hi.shape[0]:
            raise ValueError('triple product inner dimensions do not match')
        if not all(np.all(np.isfinite(word)) for word in (*self.words,*v.words)):
            raise ArithmeticError('triple products require finite coefficients')
        a=self.reshape(1,-1) if left else self;b=v.reshape(-1,1) if right else v
        active=[k for k in range(a.hi.shape[1])
                if any(np.any(x[:,k]) for x in a.words) and any(np.any(x[k]) for x in b.words)]
        def accumulate(rows):
            out=Triple.zeros((rows.hi.shape[0],b.hi.shape[1]))
            for k in active:out=out+rows[:,k,None]*b[k,None,:]
            return out
        if a.hi.shape[0]>128 and a.hi.shape[0]*b.hi.shape[1]>=65536:
            # Bound the temporary arrays without changing any entry's
            # multiplication/addition order or its three complete words.
            # Keep the same global exact-zero decisions in every row tile.
            out=Triple.zeros((a.hi.shape[0],b.hi.shape[1]))
            for first in range(0,len(a),128):out[first:first+128]=accumulate(a[first:first+128])
        else:out=accumulate(a)
        if left and right:return out[0,0]
        if left:return out[0]
        if right:return out[:,0]
        return out
    def __rmatmul__(self,v):return Triple(v)@self
    @staticmethod
    def zeros(shape):return Triple(np.zeros(shape))
    @staticmethod
    def zeros_like(x):return Triple(np.zeros_like(Triple(x).hi))
    @staticmethod
    def broadcast(x,shape):return Triple(*(np.broadcast_to(v,shape) for v in Triple(x).words))

class ComplexTriple:
    def __init__(self,x,imag=None):
        if isinstance(x,ComplexTriple):self.real,self.imag=x.real,x.imag
        elif isinstance(x,Triple):self.real,self.imag=x,Triple.zeros_like(x) if imag is None else Triple(imag)
        elif isinstance(x,ComplexDouble):self.real,self.imag=Triple(x.hi.real,x.lo.real),Triple(x.hi.imag,x.lo.imag)
        else:
            a=np.asarray(x,complex);self.real,self.imag=Triple(a.real),Triple(a.imag)
    def __getitem__(self,k):return ComplexTriple(self.real[k],self.imag[k])
    def __add__(self,b):b=ComplexTriple(b);return ComplexTriple(self.real+b.real,self.imag+b.imag)
    __radd__=__add__
    def __neg__(self):return ComplexTriple(-self.real,-self.imag)
    def __sub__(self,b):return self+-ComplexTriple(b)
    def __rsub__(self,b):return ComplexTriple(b)+-self
    def __mul__(self,b):b=ComplexTriple(b);return ComplexTriple(self.real*b.real-self.imag*b.imag,self.real*b.imag+self.imag*b.real)
    __rmul__=__mul__
    def __truediv__(self,b):
        b=ComplexTriple(b);den=b.real*b.real+b.imag*b.imag
        return ComplexTriple((self.real*b.real+self.imag*b.imag)/den,(self.imag*b.real-self.real*b.imag)/den)
    def __rtruediv__(self,b):return ComplexTriple(b)/self

def _constant(v):
    h=float(v);v-=Decimal.from_float(h);m=float(v);return Triple(h,m,float(v-Decimal.from_float(m)))
with localcontext() as ctx:
    ctx.prec=90
    def atan(x):
        term=total=x
        for k in range(1,200):term*=-x*x;total+=term/(2*k+1)
        return total
    PI=_constant(16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239));LN2=_constant(Decimal(2).ln())
    EXPC=[_constant(Decimal(1)/factorial(n)) for n in range(36)]
    COSC=[_constant(Decimal((-1)**n)/factorial(2*n)) for n in range(22)]
    SINC=[_constant(Decimal((-1)**n)/factorial(2*n+1)) for n in range(22)]
    LOGC=[_constant(Decimal(1)/(2*n+1)) for n in range(40)]

def logarithm(v):
    v=Triple(v)
    if not all(np.all(np.isfinite(x)) for x in v.words) or np.any(v.hi<=0):
        raise ArithmeticError('triple logarithms require finite positive coordinates')
    _,e=np.frexp(v.hi);m=Triple(*(np.ldexp(x,-e) for x in v.words));lower=m.hi<np.sqrt(.5);m=m*np.where(lower,2.,1.);e=e-lower
    z=(m-1)/(m+1);square=z*z;term=total=z
    for n in range(1,40):term=term*square;total=total+term*LOGC[n]
    return 2*total+e*LN2

def polynomial(x,coefficients):
    value=Triple.broadcast(coefficients[-1],x.hi.shape)
    for c in coefficients[-2::-1]:value=value*x+c
    return value

def exponential(v):
    v=ComplexTriple(v);real,angle=v.real,v.imag
    if not all(np.all(np.isfinite(x)) for x in (*real.words,*angle.words)):
        raise ArithmeticError('triple exponentials require finite coordinates')
    bound=abs(real.hi);powers=np.zeros(bound.shape,int);large=bound>.25;powers[large]=np.ceil(np.log2(bound[large]/.25)).astype(int)
    if not any(np.any(w) for w in real.words):radial=Triple(np.ones_like(bound))
    else:
        z=Triple(*(np.ldexp(x,-powers) for x in real.words));radial=polynomial(z,EXPC)
        for n in range(int(powers.max(initial=0))):
            active=powers>n;radial[active]=radial[active]*radial[active]
    if not any(np.any(w) for w in angle.words):cosine,sine=Triple(np.ones_like(bound)),Triple(np.zeros_like(bound))
    else:
        quarter=np.rint(angle.hi/(np.pi/2));residual=angle-Triple(quarter)*PI/2;square=residual*residual
        cosine=polynomial(square,COSC);sine=residual*polynomial(square,SINC);phase=quarter.astype(np.int64)%4
        def orient(c,s):return Triple(*(np.select([phase==0,phase==1,phase==2],[a,-b,-a],default=b) for a,b in zip(c.words,s.words)))
        cosine,sine=orient(cosine,sine),orient(sine,-cosine)
    return ComplexTriple(radial*cosine,radial*sine)
