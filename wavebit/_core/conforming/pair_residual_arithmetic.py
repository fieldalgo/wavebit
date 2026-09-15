"""Accurate original pair constraints formed from large cancelling matrices.

NumPy-only double-precision residual arithmetic. Normalize rows and columns
by powers of two and split each input into three bounded integer chunks and
its complete floating remainder. Leading dot products then have exact integer
partial sums below 2**50. Keep every cross product and accumulate high/low
parts before rounding the small residual. No potential or matrix entry is
selected or removed by magnitude, rank or material.
"""
import numpy as np


def _two_sum(a,b):
    value=a+b;part=value-a
    return value,(a-(value-part))+(b-part)


def _sum_pairs(pairs):
    pairs=iter(pairs)
    value,error=next(pairs)
    for hi,lo in pairs:
        value,new=_two_sum(value,hi)
        value,error=_two_sum(value,error+new+lo)
    return value,error


def _real_product(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError('physical residual terms must be compatible matrices')
    if not a.shape[1]:
        zero=np.zeros((a.shape[0],b.shape[1]));return zero,zero
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ArithmeticError('physical residual requires finite matrix entries')
    # An identically zero operand has an exactly zero product. Test only
    # after the finite-input guard; no nonzero entry or field is discarded.
    if not np.any(a) or not np.any(b):
        zero=np.zeros((a.shape[0],b.shape[1]));return zero,zero
    ea=np.frexp(np.max(abs(a),axis=1))[1]
    eb=np.frexp(np.max(abs(b),axis=0))[1]
    x=np.ldexp(a,-ea[:,None]);y=np.ldexp(b,-eb[None,:])
    # The dimension chooses integer units, never a retained-field subset.
    bits=min(20,(50-(a.shape[1]-1).bit_length())//2)
    if bits<=0:raise ArithmeticError('residual dot dimension exceeds exact integer accumulation')
    p=float(2**bits)
    def split(z):
        first=np.rint(z*p);rest=z*p-first
        second=np.rint(rest*p);rest=rest*p-second
        third=np.rint(rest*p);fourth=rest*p-third
        return first,second,third,fourth
    xx,yy=split(x),split(y)
    exponents=(bits,2*bits,3*bits,3*bits)
    def terms():
        for i,j in sorted(((i,j) for i in range(4) for j in range(4)),key=lambda ij:sum(ij)):
            term=np.ldexp(xx[i]@yy[j],-exponents[i]-exponents[j])
            yield term,np.zeros_like(term)
    value,error=_sum_pairs(terms())
    exponent=ea[:,None]+eb[None,:]
    return np.ldexp(value,exponent),np.ldexp(error,exponent)


def _product(a,b):
    a,b=np.asarray(a),np.asarray(b)
    rr=_real_product(a.real,b.real)
    if not np.iscomplexobj(a) and not np.iscomplexobj(b):return rr
    # A real coordinate map has no imaginary matrix to multiply. Preserve
    # both components of the complex operand without forming zero products.
    if not np.iscomplexobj(a):
        ri=_real_product(a,b.imag)
        return rr[0]+1j*ri[0],rr[1]+1j*ri[1]
    if not np.iscomplexobj(b):
        ir=_real_product(a.imag,b)
        return rr[0]+1j*ir[0],rr[1]+1j*ir[1]
    ii=_real_product(a.imag,b.imag)
    ri=_real_product(a.real,b.imag);ir=_real_product(a.imag,b.real)
    re=_sum_pairs([rr,(-ii[0],-ii[1])]);im=_sum_pairs([ri,ir])
    return re[0]+1j*im[0],re[1]+1j*im[1]


def residual(constant,*terms):
    """Round constant + sum(A @ B) after combining all product expansions."""
    constant=np.asarray(constant)
    if not np.all(np.isfinite(constant)):
        raise ArithmeticError('physical residual requires a finite constant')
    def values():
        yield constant,np.zeros_like(constant)
        for a,b in terms:yield _product(a,b)
    hi,lo=_sum_pairs(values())
    return hi+lo
