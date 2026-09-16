"""Independent bulk and distributional-jump means of a piecewise linear lift."""
from decimal import Decimal as Z,localcontext
from types import SimpleNamespace
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_normal_mean import moments
from test_nvf_contact_fourier_precision import decimal_pi,add,mul,times,exponential


def dec(value):return Z.from_float(float(value))


def test_large_resolved_mean_cannot_mask_an_unresolved_small_coefficient(monkeypatch):
    from wavebit._core.conforming import pair_contact_normal_mean as module
    def unresolved(obj,settings,base,transform,chart,order):
        high=np.array([[1e6,order*1e-25]],complex)
        return high,np.zeros_like(high)
    monkeypatch.setattr(module,'_integrate',unresolved)
    obj=SimpleNamespace(degree=5,order=(1,1))
    with pytest.raises(ArithmeticError,match='normal Fourier means did not resolve'):
        module.moments(obj,None,None,None,None,40)


def polynomial_integral(left,right,slope,constant,wave):
    """Integrate the independent monomial exponential series in Decimal."""
    total=(Z(0),Z(0));coefficient=(Z(1),Z(0))
    for n in range(180):
        integral=(slope*(right**(n+2)-left**(n+2))/(n+2)
                  +constant*(right**(n+1)-left**(n+1))/(n+1))
        total=add(total,times(coefficient,integral))
        coefficient=times(mul(coefficient,(Z(0),-wave)),Z(1)/(n+1))
    return total


@pytest.mark.parametrize('order',((0,0),(2,1)))
@pytest.mark.parametrize('center',((0.,0.),(.0625,-.03125)))
def test_periodic_potential_mean_matches_bulk_and_jump_integrals(order,center):
    a,b,height=.25,.125,.5;period=(1.2,.9);points=[]
    for left,right in ((-a,0.),(0.,b)):
        lo,hi=-height/2,height/2
        points.extend([(left,lo),(right,lo),(right,hi),
                       (left,lo),(right,hi),(left,hi)])
    points=np.array(points)+center;triangles=np.arange(12).reshape(4,3)
    empty=np.array([],int);local=(None,empty,empty,*([None]*8),False)
    modes=[(i,j) for j in range(-order[1],order[1]+1)
           for i in range(-order[0],order[0]+1)]
    base=({'nz':0,'size':len(modes),'local':[local]*4},)
    obj=SimpleNamespace(order=order,degree=1,data=dict(triangles=triangles,
        tags=[1,1,2,2],periodic=np.arange(12),Cfull=np.zeros((12,1)),
        Zfull=np.zeros((12,0))))
    obj._chart_geometry=lambda *args:(None,None,points,np.zeros_like(points))
    settings=dict(basis=np.diag(period),widths=(a,b),height=height,offset=center)
    transform=(np.zeros((0,1)),np.zeros((0,1)),np.array([[.7]]))
    chart=(None,None,None,None,period,(a,b),height,np.array(center),
           SimpleNamespace(order=0),{'transform':np.array([[1.3]])})
    observed=moments(obj,settings,base,transform,chart,48)
    with localcontext() as context:
        context.prec=100;pi=decimal_pi();a,b,height=map(dec,(a,b,height))
        scale=dec(.7)/dec(1.3)
        for index,(i,j) in enumerate(modes):
            kx,ky=2*pi*i/dec(period[0]),2*pi*j/dec(period[1])
            yy=polynomial_integral(-height/2,height/2,Z(0),Z(1),ky)
            left=polynomial_integral(-a,Z(0),Z(0),Z(1),kx)
            right=polynomial_integral(Z(0),b,Z(0),Z(1),kx)
            # Two constant bulk x gradients and the unit seam jump.
            xx=add(add(times(left,-1/(2*a)),times(right,-1/(2*b))),(Z(1),Z(0)))
            dx=mul(xx,yy)
            # The y derivative consists only of the bottom and top jumps.
            profile=add(polynomial_integral(-a,Z(0),-1/(2*a),-Z('.5'),kx),
                        polynomial_integral(Z(0),b,-1/(2*b),Z('.5'),kx))
            ends=add(exponential((Z(0),ky*height/2)),
                     times(exponential((Z(0),-ky*height/2)),-1))
            dy=mul(profile,ends)
            shift=exponential((Z(0),-kx*dec(center[0])-ky*dec(center[1])))
            for component,value in enumerate((dx,dy)):
                expected=times(mul(value,shift),scale);row=component*len(modes)+index
                actual=tuple(sum(dec(getattr(word[row,0],part)) for word in observed)
                             for part in ('real','imag'))
                assert max(abs(x-y) for x,y in zip(actual,expected))<Z('3e-28')
    reverse=np.r_[np.arange(len(modes))[::-1],len(modes)+np.arange(len(modes))[::-1]]
    for word in observed:np.testing.assert_array_equal(word,word[reverse].conj())
