"""Independent Decimal solves for complete retained contact elimination."""
from decimal import Decimal as Z, localcontext
import numpy as np
import pytest
from wavebit.rcwa._engine.conforming.pair_complete_solve import factor, apply, solve
from wavebit.rcwa._engine.conforming.pair_complex_precision import C
from tests.rcwa.test_nvf_contact_fourier_precision import add, sub, mul, div, scalar

pytestmark = pytest.mark.research

def dc(value):return add(scalar(value[0]),scalar(value[1]))
def decimal_solve(a,b):
 n=len(a);a=[list(row)+list(right) for row,right in zip(a,b)];m=len(b[0])
 for k in range(n):
  pivot=max(range(k,n),key=lambda row:abs(a[row][k][0])+abs(a[row][k][1]));a[k],a[pivot]=a[pivot],a[k]
  for row in range(k+1,n):
   factor=div(a[row][k],a[k][k])
   for col in range(k+1,n+m):a[row][col]=sub(a[row][col],mul(factor,a[k][col]))
 x=[[(Z(0),Z(0)) for _ in range(m)] for _ in range(n)]
 for k in range(n-1,-1,-1):
  for j in range(m):
   rhs=a[k][n+j]
   for i in range(k+1,n):rhs=sub(rhs,mul(a[k][i],x[i][j]))
   x[k][j]=div(rhs,a[k][k])
 return x

@pytest.mark.parametrize('n,bound', ((3,'3e-28'),(6,'3e-24'),(10,'3e-17'),(14,'3e-11')))
def test_complete_pivoted_solve_matches_independent_decimal(n,bound):
 with localcontext() as context:
  context.prec=100;bound=Z(bound)
  raw=[[div((Z(1),Z(0)),(Z(i+j+1),Z(0))) for j in range(n)] for i in range(n)]
  phase=[(Z(1),Z('.2')*int(i%3-1)) for i in range(n)]
  raw=[[mul(v,phase[i]) for v in row] for i,row in enumerate(raw)]
  hi=np.array([[complex(float(v[0]),float(v[1])) for v in row] for row in raw]);lo=np.array([[complex(float(sub(v,scalar(hi[i,j]))[0]),float(sub(v,scalar(hi[i,j]))[1])) for j,v in enumerate(row)] for i,row in enumerate(raw)])
  b=(np.stack((np.arange(1,n+1),np.ones(n)),axis=1).astype(complex),np.zeros((n,2),complex))
  target=decimal_solve([[dc((hi[i,j],lo[i,j])) for j in range(n)] for i in range(n)],[[dc((b[0][i,j],b[1][i,j])) for j in range(2)] for i in range(n)])
  actual=solve((hi,lo),b)
  for i in range(n):
   for j in range(2):
    diff=sub(dc((actual[0][i,j],actual[1][i,j])),target[i][j]);error=max(map(abs,diff))/max(*map(abs,target[i][j]),Z(1));assert error<bound,(n,error)


def test_complete_low_word_resolves_a_singular_leading_matrix_and_reuses_factor():
    matrix=C(np.ones((2,2)), np.array([[0.,0.],[0.,2.**-70]]))
    expected=np.array([[2.+.5j,-1.],[3.-.25j,4.+1j]])
    rhs=matrix[:,0,None]*C(expected[0][None,:])+matrix[:,1,None]*C(expected[1][None,:])
    saved=tuple(value.copy() for value in matrix.pair)
    decomposition=factor(matrix.pair)
    for sign in (1.,-1.):
        result=apply(decomposition,tuple(sign*value for value in rhs.pair))
        difference=C(*result)-C(sign*expected)
        assert np.max(abs(difference.hi+difference.lo))<3e-28
    for actual,original in zip(matrix.pair,saved):
        np.testing.assert_array_equal(actual,original)


def test_complete_factor_rejects_exact_singularity_without_a_pivot_shift():
    with pytest.raises(np.linalg.LinAlgError,match='singular complete matrix'):
        factor((np.array([[1.,2.],[2.,4.]],complex),np.zeros((2,2),complex)))


@pytest.mark.parametrize('matrix', (
    (np.zeros((2,3)),np.zeros((2,3))),
    (np.eye(2),np.zeros((2,1))),
    (np.array([[np.nan]]),np.zeros((1,1))),
))
def test_complete_factor_rejects_invalid_input(matrix):
    with pytest.raises(ValueError,match='finite square'):
        factor(matrix)
