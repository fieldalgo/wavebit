"""Independent physical contact maps and their opening limits."""
from decimal import Decimal as Z,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_quadrature import contact_maps as maps
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.pair_symmetric_mesh import SymmetricPairPolynomialCell

pytestmark = pytest.mark.research

def dec(v):return Z.from_float(float(v))
@pytest.mark.parametrize('refinement',(0,1))
def test_complete_contact_maps_match_decimal_inverses_and_gap_differences(refinement):
 obj=SymmetricPairPolynomialCell(CartesianPairOpeningPlan(0,refinement),(1,1),degree=5)
 geometry=dict(basis=np.diag([1.2,.9]),widths=np.array([.3,.4]),height=np.array(.45),offset=np.array([-.6,-.45]))
 length,_,points,velocity=obj._chart_geometry(0.,geometry);ds=.7/length
 with localcontext() as context:
  context.prec=100;step=Z('1e-35')
  for tri,thin in zip(obj.data['triangles'],obj.data['thin']):
   p,v=points[tri[:3]],velocity[tri[:3]];actual=maps(p,v,ds,thin)
   pp=[[dec(x) for x in row] for row in p];vv=[[dec(x) for x in row] for row in v]
   j=[[pp[k+1][a]-pp[0][a] for k in range(2)] for a in range(2)];dj=[[vv[k+1][a]-vv[0][a] for k in range(2)] for a in range(2)]
   def inverse(t):
    a=[[j[i][k]+t*dec(ds)*dj[i][k] for k in range(2)] for i in range(2)];det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return [[a[1][1]/det,-a[0][1]/det],[-a[1][0]/det,a[0][0]/det]]
   if thin:
    # A reference gradient from a linear physical potential annihilates
    # the collapsed adjugate. Its complete opening limit is finite.
    reference=[j[0][k]*Z('.137')+j[1][k]*Z('-.23') for k in range(2)]
    x=inverse(step)
    for axis in range(2):
     expected=sum(reference[k]*x[k][axis] for k in range(2))
     observed=sum(reference[k]*(dec(actual[0].hi[k,axis])+dec(actual[0].lo[k,axis])) for k in range(2))
     error=abs(observed-expected)/max(abs(expected),Z(1));assert error<Z('3e-28'),(refinement,thin,error)
   else:
    center=inverse(Z(0));left,right=inverse(-step),inverse(step)
    for part in range(2):
     for i in range(2):
      for k in range(2):
       expected=center[i][k] if part==0 else (right[i][k]-left[i][k])/(2*step)
       observed=dec(actual[part].hi[i,k])+dec(actual[part].lo[i,k]);error=abs(observed-expected)/max(abs(expected),Z(1));assert error<Z('3e-28'),(refinement,thin,error)
