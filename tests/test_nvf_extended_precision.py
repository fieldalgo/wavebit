"""Independent Decimal checks for three-word geometry before field cancellation."""
from decimal import Decimal as Z,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_extended_precision import Triple as T,ComplexTriple as C,logarithm,exponential
from test_nvf_transmission_common_precision import exponential as decimal_exponential

pytestmark = pytest.mark.research


def dec(value):return Z.from_float(float(value))
def scalar(value,index):return sum(dec(word[index]) for word in value.words)


@pytest.mark.parametrize('operation',('add','multiply','divide','cancel'))
def test_three_word_arithmetic_matches_decimal(operation):
    rng=np.random.default_rng(945)
    a=T(rng.normal(size=31),rng.normal(size=31)*2**-54,rng.normal(size=31)*2**-108)
    b=T(rng.normal(size=31),rng.normal(size=31)*2**-54,rng.normal(size=31)*2**-108)
    operations={'add':lambda a,b:a+b,'multiply':lambda a,b:a*b,
                'divide':lambda a,b:a/b,'cancel':lambda a,b:(a+b)-b}
    function=operations[operation];actual=function(a,b)
    with localcontext() as context:
        context.prec=100
        for index in range(31):
            expected=function(scalar(a,index),scalar(b,index))
            assert abs(scalar(actual,index)-expected)/max(Z(1),abs(expected))<Z('3e-44')


def test_three_word_cancellation_normalizes_the_leading_remainder():
    # A division correction reads the leading remainder. A cancelled high
    # word must not conceal a surviving middle or low word.
    low=2.**-110
    a=T([1.],[2.**-55],[low]);b=T([1.],[2.**-55])
    result=a-b
    np.testing.assert_array_equal(result.hi,[low])
    np.testing.assert_array_equal(result.words[1],[0.])
    np.testing.assert_array_equal(result.words[2],[0.])


@pytest.mark.parametrize('empty',(False,True))
def test_three_word_matrix_product_preserves_every_small_word(empty):
    rng=np.random.default_rng(832);n=0 if empty else 23
    a=T(rng.normal(size=(3,n)),rng.normal(size=(3,n))*2**-55,rng.normal(size=(3,n))*2**-110)
    b=T(rng.normal(size=(n,4)),rng.normal(size=(n,4))*2**-55,rng.normal(size=(n,4))*2**-110)
    actual=a@b
    with localcontext() as context:
        context.prec=100
        for i in range(3):
            for j in range(4):
                expected=sum(scalar(a,(i,k))*scalar(b,(k,j)) for k in range(n))
                assert abs(scalar(actual,(i,j))-expected)/max(Z(1),abs(expected))<Z('3e-43')
    if not empty:
        for observed,expected in ((a[0]@b,(a@b)[0]),(a@b[:,0],(a@b)[:,0]),(a[0]@b[:,0],(a@b)[0,0])):
            for x,y in zip(observed.words,expected.words):np.testing.assert_array_equal(x,y)


def test_three_word_logarithm_matches_decimal():
    rng=np.random.default_rng(619)
    value=T(np.exp(rng.normal(size=31)*2),rng.normal(size=31)*2**-54,rng.normal(size=31)*2**-108)
    result=logarithm(value)
    with localcontext() as context:
        context.prec=100
        for index in range(31):
            expected=scalar(value,index).ln()
            assert abs(scalar(result,index)-expected)/max(Z(1),abs(expected))<Z('3e-43')


@pytest.mark.parametrize('rows,columns,inner',((257,257,0),(257,257,23),(512,324,36)))
def test_large_three_word_products_preserve_independent_rows_and_decimal_dots(rows,columns,inner):
    rng=np.random.default_rng(508)
    a=T(*(rng.normal(size=(rows,inner))*2.**(-55*k) for k in range(3)))
    b=T(*(rng.normal(size=(inner,columns))*2.**(-55*k) for k in range(3)))
    if inner:
        # Whole zero rows and directions that survive only in a low word
        # must behave identically inside and outside a large field bank.
        a.hi[:129,:3]=0.;a.words[1][:129,:2]=0.
        for word in a.words:word[130:135]=0.;word[:128,4]=0.
        for word in b.words:word[5]=0.
    actual=a@b
    pieces=[a[first:first+31]@b for first in range(0,rows,31)]
    for word,observed in enumerate(actual.words):
        expected=np.concatenate([piece.words[word] for piece in pieces])
        np.testing.assert_array_equal(observed.view(np.uint64),expected.view(np.uint64))
    with localcontext() as context:
        context.prec=100
        for row in (0,127,128,130,255,rows-1):
            for column in (0,columns//2,columns-1):
                exact=sum(scalar(a,(row,k))*scalar(b,(k,column)) for k in range(inner))
                error=abs(scalar(actual,(row,column))-exact)/max(Z(1),abs(exact))
                assert error<Z('3e-43')


def test_three_word_products_keep_low_only_operands_and_finite_guards():
    left=T(np.zeros((2,3)),np.full((2,3),2.**-100),np.full((2,3),2.**-200))
    right=T(np.ones((3,2)))
    result=left@right
    np.testing.assert_array_equal(result.hi,np.full((2,2),3*2.**-100))
    np.testing.assert_array_equal(result.lo,np.full((2,2),3*2.**-200))
    with pytest.raises(ArithmeticError,match='finite'):
        T.zeros((2,3))@T(np.full((3,2),np.nan))
    with pytest.raises(ArithmeticError,match='finite'):
        T(np.ones((2,3)),np.full((2,3),np.inf))@T.zeros((3,2))


@pytest.mark.parametrize('left,right',(((2,2),(3,2)),((2,3,4),(4,2)),((),(2,2))))
def test_three_word_products_reject_incompatible_shapes(left,right):
    with pytest.raises(ValueError,match='triple product'):
        T.zeros(left)@T.zeros(right)


def test_three_word_complex_exponential_matches_decimal_series():
    real=np.array([-500.,-100.,-4.,-.125,0.,.125,4.,100.,500.])[:,None]
    angle=np.array([-700.,-32*np.pi,-np.pi/2,-.25,0.,np.pi/2,np.nextafter(np.pi/2,np.inf),12*np.pi,700.])[None,:]
    value=real+1j*angle;rng=np.random.default_rng(11)
    low=rng.normal(size=value.shape+(2,))*2**-60
    third=rng.normal(size=value.shape+(2,))*2**-115
    argument=C(T(value.real,low[...,0],third[...,0]),T(value.imag,low[...,1],third[...,1]))
    actual=exponential(argument)
    with localcontext() as context:
        context.prec=100
        for index in np.ndindex(value.shape):
            expected=decimal_exponential((scalar(argument.real,index),scalar(argument.imag,index)))
            observed=(scalar(actual.real,index),scalar(actual.imag,index))
            error=max(abs(a-b) for a,b in zip(observed,expected))
            assert error/max(*map(abs,expected),Z('1e-300'))<Z('3e-43')


@pytest.mark.parametrize('argument',(np.nan,np.inf,complex(0.,np.nan)))
def test_three_word_exponential_rejects_nonfinite_coordinates(argument):
    with pytest.raises(ArithmeticError,match='finite coordinates'):exponential(C(argument))


@pytest.mark.parametrize('argument',(0.,-1.,np.nan,np.inf))
def test_three_word_logarithm_rejects_invalid_coordinates(argument):
    with pytest.raises(ArithmeticError,match='positive coordinates'):logarithm(T(argument))
