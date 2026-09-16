"""Independent exact-input oracles for small residuals of large matrix terms."""
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_residual_arithmetic import residual, _product, _real_product


@pytest.mark.parametrize('size,span', [(31,40), (100,160), (631,400), (8192,450)])
@pytest.mark.parametrize('rows', [1,3])
@pytest.mark.parametrize('complex_', [False,True])
def test_anticorrelated_scales_preserve_the_complete_dot(size,span,rows,complex_):
    """Large columns at different samples must not hide a small physical dot."""
    rng=np.random.default_rng(712)
    exponents=np.linspace(-span,span,size).astype(int)
    def draw(shape):
        value=rng.normal(size=shape)
        return value+1j*rng.normal(size=shape) if complex_ else value
    a=draw((rows,size))*np.exp2(exponents)[None,:]
    b=draw((size,4))*np.exp2(-exponents)[:,None]
    if rows>1:
        # Unrelated large entries prevent one global coordinate scaling
        # from balancing every output; these dots need tail refinement.
        a[1,0]=np.exp2(float(span))
        b[-1,2]=np.exp2(float(span))
    high,low=_product(a,b)
    with localcontext() as context:
        context.prec=300
        for i in range(rows):
            for j in range(4):
                real=imag=scale=D(0)
                for u,v in zip(a[i],b[:,j]):
                    ur,ui,vr,vi=(D.from_float(float(x)) for x in (u.real,u.imag,v.real,v.imag))
                    real+=ur*vr-ui*vi
                    imag+=ur*vi+ui*vr
                    scale+=(abs(ur)+abs(ui))*(abs(vr)+abs(vi))
                actual_real=sum(D.from_float(float(x.real)) for x in (high[i,j],low[i,j]))
                actual_imag=sum(D.from_float(float(x.imag)) for x in (high[i,j],low[i,j]))
                assert abs(actual_real-real)+abs(actual_imag-imag)<=D.from_float(16*np.finfo(float).eps**2)*scale


@pytest.mark.parametrize('left,right', [
    ([1.,2.**-80],[1.,1.]),
    ([2.**1023,2.**1023,1.],[1.,-1.,2.**-1020]),
    ([2.**-1022,2.**-1022],[.5,.25]),
    ([2.**1000,2.**-1000],[2.**-1000,2.**1000]),
])
def test_product_refinement_preserves_extreme_finite_dyadic_inputs(left,right):
    from fractions import Fraction
    expected=sum((Fraction.from_float(a)*Fraction.from_float(b)
                  for a,b in zip(left,right)),Fraction(0))
    high,low=_real_product(np.array([left]),np.array(right)[:,None])
    assert Fraction.from_float(float(high[0,0]))+Fraction.from_float(float(low[0,0]))==expected


def test_structural_zero_padding_preserves_the_same_complete_dot():
    rng=np.random.default_rng(91)
    powers=np.linspace(-160,160,100).astype(int)
    a=rng.normal(size=(1,100))*np.exp2(powers)
    b=rng.normal(size=(100,7))*np.exp2(-powers)[:,None]
    short=_real_product(a,b)
    padded=_real_product(np.vstack([np.zeros((150,100)),a]),b)
    difference=(short[0]-padded[0][-1:])+(short[1]-padded[1][-1:])
    assert np.max(abs(difference)/(abs(a)@abs(b)))<16*np.finfo(float).eps**2

@pytest.mark.parametrize('size',[0,3,631,8192])
@pytest.mark.parametrize('complex_left,complex_right',[(False,False),(True,False),(False,True),(True,True)])
def test_cancelling_residual_matches_one_hundred_digit_arithmetic(size,complex_left,complex_right):
    rng=np.random.default_rng(271)
    def draw(shape,complex_):
        value=rng.normal(size=shape)
        if complex_:value=value+1j*rng.normal(size=shape)
        return value*np.exp2(rng.integers(-12,13,size=shape))
    a,b=draw((2,size),complex_left),draw((size,3),complex_right)
    c,d=draw((2,5),complex_left),draw((5,3),complex_right)
    constant=-(a@b+c@d)
    result=residual(constant,(a,b),(c,d))
    with localcontext() as context:
        context.prec=100
        for i in range(2):
            for j in range(3):
                re,im=D.from_float(float(constant[i,j].real)),D.from_float(float(constant[i,j].imag))
                for left,right in ((a,b),(c,d)):
                    for v,w in zip(left[i],right[:,j]):
                        vr,vi,wr,wi=(D.from_float(float(z)) for z in (v.real,v.imag,w.real,w.imag))
                        re+=vr*wr-vi*wi;im+=vr*wi+vi*wr
                expected=complex(float(re),float(im))
                scale=abs(constant[i,j])+sum(np.sum(abs(left[i])*abs(right[:,j])) for left,right in ((a,b),(c,d)))
                assert abs(result[i,j]-expected)<=16*np.finfo(float).eps**2*scale

def test_an_empty_product_preserves_the_constant_exactly():
    constant=np.array([[1+.3j,4-2j],[-3+2j,0.]])
    np.testing.assert_array_equal(residual(constant,(np.empty((2,0)),np.empty((0,2)))),constant)


def test_empty_inner_dimension_does_not_accept_incompatible_matrices():
    with pytest.raises(ValueError,match='compatible matrices'):
        residual(np.zeros((2,2)),(np.empty((2,0)),np.ones((1,2))))


@pytest.mark.parametrize('constant,term',[(np.array([[np.nan]]),np.ones((1,1))),
                                       (np.ones((1,1)),np.array([[np.inf]]))])
def test_nonfinite_constraints_are_rejected(constant,term):
    with pytest.raises(ArithmeticError,match='finite'):
        residual(constant,(term,np.ones((1,1))))
