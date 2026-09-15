"""Independent exact-input oracles for small residuals of large matrix terms."""
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_residual_arithmetic import residual

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
