"""Accurate original pair constraints formed from large cancelling matrices.

NumPy-only double-precision residual arithmetic. Normalize rows and columns
by powers of two and split each input into three bounded integer chunks and
its complete floating remainder. Leading dot products then have exact integer
partial sums below 2**50. Keep every cross product and accumulate high/low
parts before rounding the small residual. Binary inner scaling balances
anticorrelated sample magnitudes. A rounding bound identifies floating
remainder products that require complete scalar-product refinement; this
changes arithmetic only. No potential or matrix entry is
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


def _distill(values):
    """Return a tree sum and every exact addition remainder."""
    if len(values)==1:return values[0],np.zeros_like(values)
    errors=[]
    while len(values)>1:
        count=len(values)//2
        high,low=_two_sum(values[:2*count:2],values[1:2*count:2])
        errors.append(low)
        values=np.concatenate([high,values[-1:]]) if len(values)%2 else high
    return values[0],np.concatenate(errors)


def _exact_dot(left,right):
    """Accumulate every finite binary product exactly with integer shifts.

    Float denominators are powers of two. Align them in one integer sum
    instead of reducing a rational after each multiply and add. Conversion
    to the final two words uses the same exact rational rounding as before.
    """
    from fractions import Fraction
    total=0;power=0
    for u,v in zip(left,right):
        un,ud=float(u).as_integer_ratio();vn,vd=float(v).as_integer_ratio()
        numerator=un*vn
        exponent=ud.bit_length()+vd.bit_length()-2
        if exponent>power:
            total<<=exponent-power;power=exponent
        total+=numerator<<(power-exponent)
    exact=Fraction(total,1<<power)
    high=float(exact)
    return high,float(exact-Fraction.from_float(high))


def _refine_dots(a,b,bad,value,error):
    """Refine complete products, in bounded batches, without field selection.

    Three sum-distillation passes preserve addition errors beyond the final
    high/low precision. Dots whose largest scalar product lies outside the safe exponent range use
    exact standard-library fractions before rounding their final two words.
    """
    from .pair_oscillatory_radial import two_product
    rows,columns=np.nonzero(bad)
    for begin in range(0,len(rows),64):
        ri=rows[begin:begin+64];ci=columns[begin:begin+64]
        am,ax=np.frexp(a[ri].T);bm,bx=np.frexp(b[:,ci]);exponent=ax+bx
        active=(am!=0)&(bm!=0)
        # Even the maximum admitted inner dimension cannot accumulate an
        # underflow error near two-word precision when one term has exponent
        # >= -900. Smaller terms are still multiplied; no field is omitted.
        maximum=np.max(np.where(active,exponent,-4096),axis=0)
        safe=(maximum>=-900)&(maximum<=960)
        empty=~np.any(active,axis=0)
        value[ri[empty],ci[empty]]=0.;error[ri[empty],ci[empty]]=0.
        safe|=empty
        for row,column in zip(ri[~safe],ci[~safe]):
            value[row,column],error[row,column]=_exact_dot(a[row],b[:,column])
        if not np.any(safe):continue
        high,low=two_product(am[:,safe],bm[:,safe])
        high=np.ldexp(high,exponent[:,safe]);low=np.ldexp(low,exponent[:,safe])
        first,rest=_distill(np.concatenate([high,low]))
        second,rest=_distill(rest)
        third,_=_distill(rest)
        high,low=_sum_pairs([(first,second),(third,np.zeros_like(third))])
        value[ri[safe],ci[safe]],error[ri[safe],ci[safe]]=high,low


def _real_product(a,b,*,_chunks=3):
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
    ae=np.frexp(np.max(abs(a),axis=0))[1];be=np.frexp(np.max(abs(b),axis=1))[1]
    shift=(ae-be)//2
    # If scaling would erase or overflow a finite operand, keep that original
    # inner coordinate. Exact products below still recover a difficult dot.
    with np.errstate(under='ignore',over='ignore'):
        aa=np.ldexp(a,-shift[None,:]);bb=np.ldexp(b,shift[:,None])
        valid=np.all(np.isfinite(aa)&(np.ldexp(aa,shift[None,:])==a),axis=0)&np.all(np.isfinite(bb)&(np.ldexp(bb,-shift[:,None])==b),axis=1)
    aa[:,~valid]=a[:,~valid];bb[~valid]=b[~valid]
    a,b=aa,bb
    ea=np.frexp(np.max(abs(a),axis=1))[1];eb=np.frexp(np.max(abs(b),axis=0))[1]
    x=np.ldexp(a,-ea[:,None]);y=np.ldexp(b,-eb[None,:])
    bits=min(20,(50-(a.shape[1]-1).bit_length())//2)
    if bits<=0:raise ArithmeticError('residual dot dimension exceeds exact integer accumulation')
    unit=float(2**bits)
    def split(z):
        chunks=[];remainder=z
        for _ in range(_chunks):
            scaled=remainder*unit;chunk=np.rint(scaled)
            remainder=scaled-chunk;chunks.append(chunk)
        return (*chunks,remainder)
    xx,yy=split(x),split(y)
    exponents=tuple(bits*(i+1) for i in range(_chunks))+(_chunks*bits,)
    present_x=tuple(np.any(v) for v in xx);present_y=tuple(np.any(v) for v in yy)
    def terms():
        for i,j in sorted(((i,j) for i in range(_chunks+1) for j in range(_chunks+1)),key=lambda ij:sum(ij)):
            if not present_x[i] or not present_y[j]:continue
            scale=-exponents[i]-exponents[j]
            value=np.ldexp(xx[i]@yy[j],scale)
            yield value,np.zeros_like(value)
    value,error=_sum_pairs(terms());exponent=ea[:,None]+eb[None,:]
    bad=np.zeros_like(value,dtype=bool)
    if present_x[_chunks] or present_y[_chunks]:
        # Sum the integer-chunk magnitudes before their absolute
        # products. This bounds the same complete remainder products with at
        # most three matrix multiplications, retaining every cross term.
        tail=np.zeros_like(value)
        if present_y[_chunks]:
            leading=sum(np.ldexp(abs(xx[i]),-exponents[i]) for i in range(_chunks))
            tail+=np.ldexp(leading@abs(yy[_chunks]),-_chunks*bits)
        if present_x[_chunks]:
            leading=sum(np.ldexp(abs(yy[j]),-exponents[j]) for j in range(_chunks))
            tail+=np.ldexp(abs(xx[_chunks])@leading,-_chunks*bits)
        if present_x[_chunks] and present_y[_chunks]:
            tail+=np.ldexp(abs(xx[_chunks])@abs(yy[_chunks]),-2*_chunks*bits)
        eps=np.finfo(float).eps
        absolute=abs(x)@abs(y);gamma=2*a.shape[1]*eps/(1-2*a.shape[1]*eps)
        bad=gamma*tail+32*a.shape[1]*np.nextafter(0.,1.)>4*eps**2*absolute
    # Rescaling a cancellation across the normal exponent range can erase
    # its surviving term even when the large absolute-dot bound is benign.
    bad|=(exponent>960)|(exponent<-900)
    bad&=np.any(a!=0,axis=1)[:,None]&np.any(b!=0,axis=0)[None,:]
    # Also repair any underflow introduced by the row/column normalization.
    bad|=(np.any(np.ldexp(x,ea[:,None])!=a,axis=1)[:,None]|np.any(np.ldexp(y,eb[None,:])!=b,axis=0)[None,:])
    value,error=np.ldexp(value,exponent),np.ldexp(error,exponent)
    # Dense difficult products are cheaper as one wider integer expansion
    # than as many separate scalar-product batches. The remaining floating
    # tail is still retained and checked by the identical error bound.
    if _chunks==3 and np.count_nonzero(bad)*a.shape[1]>8192:
        return _real_product(a,b,_chunks=6)
    _refine_dots(a,b,bad,value,error)
    return value,error


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
