"""Independent exponential divided differences and physical free moments."""
from decimal import Decimal as Z,localcontext
import numpy as np
from wavebit._core.conforming.triangle_exponential_precision import moment
from wavebit._core.conforming.pair_corner_precision import D
import pytest
def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[0],-a[1]
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def div(a,b):
 d=b[0]*b[0]+b[1]*b[1];return (a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d
def scalar(a):return Z.from_float(float(np.real(a))),Z.from_float(float(np.imag(a)))
def exponential(z):
 n=0
 while abs(z[0])+abs(z[1])>Z('.125'):z=(z[0]/2,z[1]/2);n+=1
 term=result=(Z(1),Z(0))
 for k in range(1,181):
  a,b=mul(term,z);term=(a/k,b/k);result=add(result,term)
 for _ in range(n):result=mul(result,result)
 return result
def sub(a,b):return add(a,neg(b))
def times(a,x):return a[0]*x,a[1]*x

def reference(theta,velocity):
 z=[scalar(1j*v) for v in theta];dz=[scalar(1j*v) for v in velocity]
 if theta[0]==theta[1]==theta[2]:
  e=exponential(z[0]);return times(e,Z(1)/2),times(mul(e,add(add(dz[0],dz[1]),dz[2])),Z(1)/6)
 pairs=[(i,j) for i,j in ((0,1),(0,2),(1,2)) if theta[i]==theta[j]]
 if pairs:
  i,j=pairs[0];k=next(k for k in range(3) if k not in (i,j));a=sub(z[k],z[i]);e=exponential(a);one=(Z(1),Z(0));a2=mul(a,a)
  numerator=sub(sub(e,one),a);f=div(numerator,a2)
  df=div(sub(mul(sub(e,one),a),times(numerator,Z(2))),mul(a2,a))
  delta=add(times(mul(add(dz[i],dz[j]),sub(f,df)),Z(1)/2),mul(dz[k],df));factor=exponential(z[i])
  return mul(factor,f),mul(factor,delta)
 value=delta=(Z(0),Z(0))
 for i in range(3):
  den=(Z(1),Z(0));logarithmic=dz[i]
  for j in range(3):
   if i==j:continue
   difference=sub(z[i],z[j]);den=mul(den,difference);logarithmic=sub(logarithmic,div(sub(dz[i],dz[j]),difference))
  term=div(exponential(z[i]),den);value=add(value,term);delta=add(delta,mul(term,logarithmic))
 return value,delta

cases=[(0.,0.,0.),(2.2,2.2,2.2),(0.,0.,3.7),(0.,3.7,0.),(3.7,0.,0.),(0.,1e-80,-1e-80),(0.,1e-12,2e-12),(1.,1.+1e-12,1.-1e-12),(0.,23.,-41.),(100.,-70.,13.),(-100.,100.,.3)]
@pytest.mark.parametrize('case', cases)
def test_complete_exponential_moment_and_jvp_match_independent_decimal(case):
    velocity=np.array([.137,-.41,.23])
    actual=moment([D([v]) for v in case],[D([v]) for v in velocity])
    with localcontext() as context:
        context.prec=400
        expected=reference(case,velocity)
        for a,b in zip(actual,expected):
            real=Z.from_float(float(a.hi[0].real))+Z.from_float(float(a.lo[0].real))
            imag=Z.from_float(float(a.hi[0].imag))+Z.from_float(float(a.lo[0].imag))
            assert max(abs(real-b[0]),abs(imag-b[1]))<Z('3e-28')


def decimal_pi():
    def arctangent_inverse(n):
        x=Z(1)/n
        term=value=x
        for k in range(1,180):
            term=-term*x*x
            value+=term/(2*k+1)
        return value
    return 16*arctangent_inverse(5)-4*arctangent_inverse(239)


@pytest.mark.parametrize('order,refinement', [((0,0),0),((1,2),0),((1,1),1)])
def test_entire_free_matrix_and_joint_jvp_match_decimal_rectangles(order,refinement):
    from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit._core.conforming.pair_symmetric_mesh import SymmetricPairPolynomialCell
    from wavebit._core.conforming.pair_contact_fourier import moments
    obj=SymmetricPairPolynomialCell(CartesianPairOpeningPlan(0,refinement=refinement),
                                    order,degree=3)
    settings=dict(gap=0.,basis=np.eye(2),widths=(.25,.125),height=.5,
        offset=(-.5,-.5),epsilon=(4+.125j,9+.25j),background=1+.0625j,
        gap_tangent=.5,epsilon_tangent=(.25+.125j,-.5+.25j),
        background_tangent=.125-.0625j)
    base=obj._assemble(**settings)
    # Supply exact dyadic area records for this independent rectangle oracle.
    # The separate constant-mode check below covers the original stored areas.
    _,_,points,_=obj._chart_geometry(0.,{k:np.asarray(settings[k]) for k in
                                       ('basis','widths','height','offset')})
    local=[]
    for tri,row in zip(obj.data['triangles'],base[-1]['local']):
        p0,p1,p2=points[tri[:3]]
        a,b=p1-p0,p2-p0
        row=list(row)
        row[9]=a[0]*b[1]-a[1]*b[0]
        local.append(tuple(row))
    base=(*base[:-1],dict(base[-1],local=local))
    free,derivative,gram=moments(obj,settings,base)
    modes=[(i,j) for j in range(-order[1],order[1]+1)
                 for i in range(-order[0],order[0]+1)]
    with localcontext() as context:
        context.prec=100
        pi=decimal_pi()
        bg,db=scalar(settings['background']),scalar(settings['background_tangent'])
        for row,(mx,my) in enumerate(modes):
            for col,(nx,ny) in enumerate(modes):
                x,y=mx-nx,my-ny
                expected=bg if row==col else (Z(0),Z(0))
                delta=db if row==col else (Z(0),Z(0))
                for width,ep,dep,shift,dshift in zip(settings['widths'],settings['epsilon'],
                        settings['epsilon_tangent'],(-.0625,.125),(-.25,.25)):
                    w,h=Z.from_float(width),Z('.5')
                    def sinc(z):
                        return Z(1) if z==0 else exponential((Z(0),z))[1]/z
                    indicator=times(exponential((Z(0),-2*pi*x*Z.from_float(shift))),
                                    w*h*sinc(pi*x*w)*sinc(pi*y*h))
                    change=mul((Z(0),-2*pi*x*Z.from_float(dshift)),indicator)
                    expected=add(expected,mul(sub(scalar(ep),bg),indicator))
                    delta=add(delta,add(mul(sub(scalar(dep),db),indicator),
                                       mul(sub(scalar(ep),bg),change)))
                for actual,target in ((free,expected),(derivative,delta),
                                      (gram,(Z(int(row==col)),Z(0)))):
                    high,low=(value[row,col] for value in actual)
                    re=Z.from_float(float(high.real))+Z.from_float(float(low.real))
                    im=Z.from_float(float(high.imag))+Z.from_float(float(low.imag))
                    assert max(abs(re-target[0]),abs(im-target[1]))<Z('3e-28')


@pytest.mark.parametrize('value', (0.,2.5+.137j,-.8+.23j,100j))
def test_scalar_complex_exponential_retains_its_low_part(value):
    from wavebit._core.conforming.pair_complex_precision import C
    actual=C(value).exp()
    with localcontext() as context:
        context.prec=100
        re,im=exponential(scalar(value))
        high,low=actual.pair
        assert abs(Z.from_float(float(high.real))+Z.from_float(float(low.real))-re)<Z('3e-28')
        assert abs(Z.from_float(float(high.imag))+Z.from_float(float(low.imag))-im)<Z('3e-28')


def test_constant_free_moments_preserve_original_stored_measures():
    from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
    from wavebit._core.conforming.pair_symmetric_mesh import SymmetricPairPolynomialCell
    from wavebit._core.conforming.pair_contact_fourier import moments
    obj=SymmetricPairPolynomialCell(CartesianPairOpeningPlan(0),0,degree=3)
    settings=dict(gap=0.,basis=np.eye(2),widths=(.25,.125),height=.5,
        offset=(-.5,-.5),epsilon=(4+.125j,9+.25j),background=1+.0625j,
        gap_tangent=.5,epsilon_tangent=(.25+.125j,-.5+.25j),
        background_tangent=.125-.0625j)
    base=obj._assemble(**settings)
    actual=moments(obj,settings,base)
    with localcontext() as context:
        context.prec=100
        expected=[(Z(0),Z(0)) for _ in range(3)]
        for row in base[-1]['local']:
            area=Z.from_float(row[9])/2
            change=Z.from_float(base[-1]['ds']*row[10])/2
            expected[0]=add(expected[0],times(scalar(row[12]),area))
            expected[1]=add(expected[1],add(times(scalar(row[13]),area),
                                          times(scalar(row[12]),change)))
            expected[2]=add(expected[2],(area,Z(0)))
        for result,target in zip(actual,expected):
            high,low=(v[0,0] for v in result)
            re=Z.from_float(float(high.real))+Z.from_float(float(low.real))
            im=Z.from_float(float(high.imag))+Z.from_float(float(low.imag))
            assert max(abs(re-target[0]),abs(im-target[1]))<Z('3e-28')
