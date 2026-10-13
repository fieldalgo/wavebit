"""Fixed real oscillatory radial coordinates with compensated evaluation.

The real block with diagonal 19/20 and off-diagonals -1/4 and 1 has
conjugate eigenvalues 19/20 +/- i/2. Combined with 3/5, 4/5 and the integer
polynomial nodes, its exponential column defines both oscillatory directions
without subtracting nearly equal sampled powers. All arithmetic is NumPy,
like the explicit pair family; no external matrix-exponential code is used.
The scalar evaluator preserves the same rounded columns. The full small
matrix evaluator remains a reference for that arithmetic compatibility.
"""
from collections import OrderedDict
from functools import lru_cache
from threading import RLock
from fractions import Fraction
from math import factorial
import numpy as np

def two_sum(a,b):
    value=a+b;part=value-a
    return value,(a-(value-part))+(b-part)

def two_product(a,b):
    scale=float(2**27+1)
    c=scale*a;ah=c-(c-a);al=a-ah
    c=scale*b;bh=c-(c-b);bl=b-bh
    value=a*b
    return value,((ah*bh-value)+ah*bl+al*bh)+al*bl

def add(a,b):
    value,error=two_sum(a[0],b[0])
    return two_sum(value,error+a[1]+b[1])

def product(a,b):
    hi,lo=two_product(a[..., :, :,None],b[...,None,:,:])
    total=(hi[...,0,:],lo[...,0,:])
    for i in range(1,a.shape[-1]):total=add(total,(hi[...,i,:],lo[...,i,:]))
    return total

def multiply(a,b):
    hi,lo=product(a[0],b[0])
    return add((hi,lo),(a[0]@b[1]+a[1]@b[0],a[1]@b[1]))

def scale(a,b):
    hi,lo=two_product(a[0],b)
    return add((hi,lo),(a[1]*b,np.zeros_like(hi)))

def divide(a,b):
    q=a[0]/b;hi,lo=two_product(q,b)
    r=add(a,(-hi,-lo))
    return two_sum(q,(r[0]+r[1])/b)

@lru_cache(None)
def matrix(k,r):
    exact=[Fraction(v) for v in range(k,k+r)]+[Fraction(3,5),Fraction(4,5),Fraction(19,20),Fraction(19,20)]
    values=np.array([float(v) for v in exact])
    m=np.diag(values)+np.diag(np.ones(len(values)-1),-1);m[-2,-1]=-.25
    low=np.diag([float(v-Fraction.from_float(float(x))) for v,x in zip(exact,values)])
    return m,low

def _column_block(h,s,k,r):
    m=matrix(k,r);size=len(m[0])
    log=np.empty_like(s);low=h<=.5
    log[low]=np.log1p(-h[low]);log[~low]=np.log(s[~low])
    bound=abs(log)*np.linalg.norm(m[0],ord=np.inf)
    powers=np.zeros(len(s),int);large=bound>.25
    powers[large]=np.ceil(np.log2(bound[large]/.25)).astype(int)
    b=scale(m,log[:,None,None])
    b=tuple(np.ldexp(v,-powers[:,None,None]) for v in b)
    value=(np.broadcast_to(np.eye(size),(len(s),size,size)).copy(),np.zeros((len(s),size,size)))
    term=value
    for n in range(1,27):
        term=divide(multiply(term,b),float(n));value=add(value,term)
    for n in range(int(powers.max(initial=0))):
        active=powers>n
        squared=multiply(tuple(v[active] for v in value),tuple(v[active] for v in value))
        for array,changed in zip(value,squared):array[active]=changed
    return value

def _matrix_values(hbytes,sbytes,degree):
    h=np.frombuffer(hbytes,float);s=np.frombuffer(sbytes,float);blocks=[]
    if not len(h):return np.empty((degree,3,0,4))
    for begin in range(0,len(h),256):
        hs,ss=h[begin:begin+256],s[begin:begin+256]
        exponential=_column_block(hs,ss,1,degree-1);families=[]
        for k in range(1,degree+1):
            r=degree-k;offset=k-1
            e=tuple(v[:,offset:,offset] for v in exponential)
            m=matrix(k if r else 0,r)
            first=multiply(e,tuple(v.T for v in m))
            second=multiply(e,tuple(v.T for v in multiply(m,add(m,(-np.eye(len(m[0])),np.zeros_like(m[0]))))))
            dh=divide(tuple(-v for v in first),ss[:,None])
            ddh=divide(divide(second,ss[:,None]),ss[:,None])
            scales=np.array([(-1.)**(r+c-1)*factorial(r+c-1) for c in range(1,5)])
            families.append(np.array([(v[0]+v[1])[:,r:r+4]*scales for v in (e,dh,ddh)]))
        blocks.append(np.array(families))
    return np.concatenate(blocks,axis=2)

def _values(hbytes,sbytes,degree):
    from wavebit.rcwa._engine.conforming.pair_oscillatory_scalar import values as scalar_values
    h=np.frombuffer(hbytes,float);s=np.frombuffer(sbytes,float)
    if not len(h):return np.empty((degree,3,0,4))
    blocks=[scalar_values(h[i:i+4096],s[i:i+4096],degree) for i in range(0,len(h),4096)]
    return np.concatenate(blocks,axis=2)


# Keep sampled array/key payload bounded even when quadrature grows. Small
# exact profiles may share the budget; full sampled matrices are never kept.
_CACHE_BYTES=64*1024**2
_cache=OrderedDict()
_cache_bytes=0
_cache_lock=RLock()


def values(hbytes,sbytes,degree):
    global _cache_bytes
    key=hbytes,sbytes,degree
    with _cache_lock:
        if key in _cache:
            value,cost=_cache.pop(key);_cache[key]=value,cost
            return value
    value=_values(hbytes,sbytes,degree)
    cost=value.nbytes+len(hbytes)+len(sbytes)
    if cost>_CACHE_BYTES:return value
    value.flags.writeable=False
    with _cache_lock:
        # Another caller may have prepared the identical profile while NumPy
        # released the GIL. Reuse it without accounting for the payload twice.
        if key in _cache:
            previous,stored=_cache.pop(key);_cache[key]=previous,stored
            return previous
        while _cache and _cache_bytes+cost>_CACHE_BYTES:
            _,(_,removed)=_cache.popitem(last=False);_cache_bytes-=removed
        _cache[key]=value,cost;_cache_bytes+=cost
    return value


def evaluate(h,s,k,r,count,degree):
    h,s=np.broadcast_arrays(np.asarray(h,float),np.asarray(s,float));shape=h.shape
    pairs=np.column_stack([h.ravel(),s.ravel()])
    unique,reverse=np.unique(pairs.view(np.dtype((np.void,16))).ravel(),return_inverse=True)
    points=unique.view(float).reshape(-1,2)
    result=values(points[:,0].tobytes(),points[:,1].tobytes(),degree)[k-1 if r else degree-1]
    return tuple(v[:,count-1][reverse].reshape(shape) for v in result)

def normalized_oscillatory_remainder(s,h,k,r,count,degree=None):
    """Return the complete normalized radial value and two s derivatives.

    Count selects a coordinate of the full real pair. A polynomial suffix
    of the exponent matrix is the matrix for the corresponding lower-degree
    overlap, so one exponential supplies every homogeneous field family.
    Only final radial outputs are retained, never the sampled full matrices.
    Cached radial array/key payload is bounded by 64 MiB across all profiles.
    """
    if degree is None:degree=k+r if r else 3
    value,dh,ddh=evaluate(h,s,k,r,count,degree)
    normalization=(-1.)**(count-1)*factorial(r+count-1)/factorial(r)
    return value/normalization,-dh/normalization,ddh/normalization
