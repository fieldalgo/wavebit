"""Independent decimal controls for complete high/low contact matrix products."""
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_precision import mat,retained_field
@pytest.mark.parametrize("n", (1,21,64,161,512))
@pytest.mark.parametrize("complex_values", (False,True,'left','right'))
def test_complete_cancelling_products_match_independent_decimal(n,complex_values):
 rng=np.random.default_rng(67)
 with localcontext() as context:
  context.prec=100
  def scalar(x):return D.from_float(float(x.real)),D.from_float(float(x.imag))
  def add(a,b):return a[0]+b[0],a[1]+b[1]
  def multiply(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
  a=rng.normal(size=(3,n));b=rng.normal(size=(n,4))
  if complex_values is True or complex_values=='left':a=a+1j*rng.normal(size=a.shape)
  if complex_values is True or complex_values=='right':b=b+1j*rng.normal(size=b.shape)
  # Opposing adjacent products leave a small complete dot product.
  for k in range(1,n,2):a[:,k]=a[:,k-1];b[k]=-b[k-1]
  al=np.spacing(abs(a))*rng.uniform(-.5,.5,a.shape);bl=np.spacing(abs(b))*rng.uniform(-.5,.5,b.shape)
  if np.iscomplexobj(a):al=al+1j*np.spacing(abs(a))*rng.uniform(-.5,.5,a.shape)
  if np.iscomplexobj(b):bl=bl+1j*np.spacing(abs(b))*rng.uniform(-.5,.5,b.shape)
  ah,blarge=a.copy(),b.copy();a=(ah,al);b=(blarge,bl)
  high,low=mat(a,b)
  for i in range(3):
   for j in range(4):
    expected=(D(0),D(0));scale=D(0)
    for k in range(n):
     aa=add(scalar(a[0][i,k]),scalar(a[1][i,k]));bb=add(scalar(b[0][k,j]),scalar(b[1][k,j]));v=multiply(aa,bb)
     expected=add(expected,v);scale+=abs(v[0])+abs(v[1])
    actual=add(scalar(high[i,j]),scalar(low[i,j]));error=max(abs(actual[k]-expected[k]) for k in range(2))/max(D(1),scale)
    assert error<D('3e-28'),(n,complex_values,i,j,error)


@pytest.mark.parametrize('n',(1,9,81))
def test_complete_cartesian_fourier_field_matches_independent_decimal(n):
 rng=np.random.default_rng(72)
 phase=rng.normal(size=(3,n))+1j*rng.normal(size=(3,n))
 coefficients=1e6*(rng.normal(size=(2*n,4))+1j*rng.normal(size=(2*n,4)))
 # Each component has cancelling neighboring modes and independent low parts.
 for k in range(1,n,2):
  phase[:,k]=phase[:,k-1]
  for component in range(2):coefficients[component*n+k]=-coefficients[component*n+k-1]
 phase=(phase,1e-17*(rng.normal(size=phase.shape)+1j*rng.normal(size=phase.shape)))
 coefficients=(coefficients,1e-11*(rng.normal(size=coefficients.shape)+1j*rng.normal(size=coefficients.shape)))
 high,low=retained_field(phase,coefficients)
 with localcontext() as context:
  context.prec=100
  def scalar(x):return D.from_float(float(x.real)),D.from_float(float(x.imag))
  def add(a,b):return a[0]+b[0],a[1]+b[1]
  def multiply(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
  for q in range(3):
   for component in range(2):
    for column in range(4):
     expected=(D(0),D(0));scale=D(0)
     for k in range(n):
      a=add(*(scalar(v[q,k]) for v in phase));b=add(*(scalar(v[component*n+k,column]) for v in coefficients));product=multiply(a,b)
      expected=add(expected,product);scale+=abs(product[0])+abs(product[1])
     actual=add(scalar(high[2*q+component,column]),scalar(low[2*q+component,column]))
     error=max(abs(actual[i]-expected[i]) for i in range(2))/max(D(1),scale)
     assert error<D('3e-28'),(n,q,component,column,error)


def test_zero_geometric_fourier_phase_preserves_finite_input_guard():
 phase=(np.zeros((3,9),complex),np.zeros((3,9),complex))
 coefficients=(np.ones((18,4),complex),np.zeros((18,4),complex))
 assert all(not np.any(v) for v in retained_field(phase,coefficients))
 coefficients[1][0,0]=np.nan
 with pytest.raises(ArithmeticError,match='finite'):
  retained_field(phase,coefficients)


@pytest.mark.parametrize('side', ('left', 'right'))
def test_zero_operand_products_keep_low_parts_and_finite_guards(side):
 from wavebit._core.conforming.pair_residual_arithmetic import _product
 left=np.zeros((3,4));right=np.ones((4,2))
 if side=='right':left,right=right.T,left.T
 assert all(not np.any(v) for v in _product(left,right))
 zero=left if side=='left' else right
 zero.flat[0]=2.**-500
 high,low=_product(left,right)
 np.testing.assert_array_equal(high+low,left@right)
 zero.fill(0.)
 other=right if side=='left' else left
 other.flat[0]=np.nan
 with pytest.raises(ArithmeticError,match='finite'):_product(left,right)
 # A zero high part must not remove the nonzero low coefficient.
 left=(np.zeros((2,3)),np.full((2,3),2.**-100))
 right=(np.ones((3,2)),np.full((3,2),2.**-55))
 high,low=mat(left,right)
 np.testing.assert_array_equal(high,np.full((2,2),3*2.**-100))
 np.testing.assert_array_equal(low,np.full((2,2),3*2.**-155))
