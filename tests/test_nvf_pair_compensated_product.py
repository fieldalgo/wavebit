"""Independent decimal controls for complete high/low contact matrix products."""
from decimal import Decimal as D,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_precision import mat
@pytest.mark.parametrize("n", (1,21,64,161,512))
@pytest.mark.parametrize("complex_values", (False,True))
def test_complete_cancelling_products_match_independent_decimal(n,complex_values):
 rng=np.random.default_rng(67)
 with localcontext() as context:
  context.prec=100
  def scalar(x):return D.from_float(float(x.real)),D.from_float(float(x.imag))
  def add(a,b):return a[0]+b[0],a[1]+b[1]
  def multiply(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
  a=rng.normal(size=(3,n));b=rng.normal(size=(n,4))
  if complex_values:a=a+1j*rng.normal(size=a.shape);b=b+1j*rng.normal(size=b.shape)
  # Opposing adjacent products leave a small complete dot product.
  for k in range(1,n,2):a[:,k]=a[:,k-1];b[k]=-b[k-1]
  al=np.spacing(abs(a))*rng.uniform(-.5,.5,a.shape);bl=np.spacing(abs(b))*rng.uniform(-.5,.5,b.shape)
  if complex_values:al=al+1j*np.spacing(abs(a))*rng.uniform(-.5,.5,a.shape);bl=bl+1j*np.spacing(abs(b))*rng.uniform(-.5,.5,b.shape)
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
