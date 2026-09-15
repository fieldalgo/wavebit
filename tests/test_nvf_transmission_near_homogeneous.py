"""Independent reference roots near equal transmission coefficients."""
import cmath
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_transmission_modes import TransmissionMode


def atan(value):
    term=total=value
    for n in range(1,150):
        term*=-value*value
        total+=term/(2*n+1)
    return total


def asin(value):
    term=total=value
    for n in range(1,50):
        term*=value*value*D((2*n-1)**2)/D((2*n)*(2*n+1))
        total+=term
    return total


@pytest.mark.parametrize('delta', (1e-4,1e-6,1e-8,1e-10,1e-12))
@pytest.mark.parametrize('sign', (-1,1))
@pytest.mark.parametrize('branch', range(4))
def test_near_equal_reference_root_matches_independent_decimal(delta,sign,branch):
    coefficient=1.+sign*delta
    mode=TransmissionMode((coefficient,1.,1.),branch=branch)
    with localcontext() as context:
        context.prec=100
        a=D.from_float(coefficient)
        pi=16*atan(D(1)/5)-4*atan(D(1)/239)
        principal=1-2*asin(abs(a-1)/(2*(a+1)))/pi
        expected=2*((branch+1)//2)+(-principal if branch%2 else principal)
        assert abs(D.from_float(mode.exponent.real)-expected)<D('2e-15')
        assert mode.exponent.imag==0
        assert mode.exponent.real!=round(mode.exponent.real)
    # The complete turn also satisfies the unchanged transmission condition.
    np.testing.assert_allclose(mode.states[-1],mode.states[0],atol=2e-12,rtol=0)


@pytest.mark.parametrize('delta', (1e-4,1e-8,1e-12))
@pytest.mark.parametrize('direction', (1+.5j,-1+.5j,1-.5j,-1-.5j))
def test_near_equal_complex_reference_preserves_closed_growing_mode(delta,direction):
    mode=TransmissionMode((1+delta*direction,1.,1.))
    assert 0<mode.exponent.real<1
    assert mode.exponent.imag!=0
    np.testing.assert_allclose(mode.states[-1],mode.states[0],atol=2e-12,rtol=0)


@pytest.mark.parametrize('coefficients', ((12.25,1.,1.),(-10+1j,1.,1.),(12.25,-10+1j,1.)))
@pytest.mark.parametrize('branch', range(4))
def test_standard_reference_roots_keep_their_existing_values(coefficients,branch):
    a,b,c=map(complex,coefficients)
    first=a/b+b/a;second=(a+b)/c+c*(1/a+1/b)
    root=cmath.acos(-(4+second)/(2+first+second))/np.pi
    expected=2*((branch+1)//2)+(-root if branch%2 else root)
    assert TransmissionMode(coefficients,branch=branch).exponent==expected
