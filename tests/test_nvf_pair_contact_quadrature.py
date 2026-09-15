"""Independent exact moments and Decimal Fourier phases at contact."""
from decimal import Decimal,localcontext
import numpy as np
import pytest
from wavebit._core.conforming.pair_contact_quadrature import gauss,quadrature,phases
from wavebit._core.conforming.pair_corner_precision import D
from wavebit._core.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit._core.conforming.cartesian_pair_corner_galerkin import CartesianPairCornerGalerkinGapCell


@pytest.mark.parametrize('order',(12,24,32,80,96))
def test_gaussian_rule_retains_every_exact_polynomial_moment(order):
    nodes,weights=gauss(order);point=2*nodes-1;power=D(np.ones(order))
    for degree in range(2*order):
        expected=D(0.) if degree%2 else D(1.)/(degree+1)
        error=(weights*power).sum(0)-expected
        assert abs(error.hi+error.lo)<3e-30,(order,degree,error.pair)
        power=power*point


def test_all_contact_partitions_preserve_exact_triangle_moments():
    cell=CartesianPairCornerGalerkinGapCell(CartesianPairOpeningPlan(0),0,
        degree=3,exponents=(.55,.6,.8,.95),junctions=True,quadrature=80,
        angular_quadrature=12,mesh='symmetric')
    fields=cell._fields
    for tri in fields.obj.data['triangles']:
        area=D(0.);first=D(np.zeros(3));second=D(np.zeros(3))
        for bary,weights in quadrature(fields,tri,80):
            area+=weights.exact.sum(0)
            first+=weights.exact@bary.exact
            second+=weights.exact@(bary.exact*bary.exact)
            partition=bary.exact.sum(1)-1
            assert np.max(abs(partition.hi+partition.lo))<3e-30
        for value,expected in ((area,D(.5)),(first,D(1.)/6),(second,D(1.)/12)):
            error=value-expected
            assert np.max(abs(error.hi+error.lo))<3e-30


@pytest.mark.parametrize("order", (1, 4))
def test_complete_fourier_phase_and_variation_match_decimal_reference(order):
    bary=np.array([[.23,.31,.46],[.58,.13,.29],[1.,2.**-81,2.**-80]])
    points=np.array([[-.35,-.225],[.35,-.225],[.05,.225]])
    velocity=np.array([[-.125,0.],[.375,0.],[.25,0.]])
    period=np.array([1.2,.9])
    modes=np.array([(i,j) for j in range(-order,order+1) for i in range(-order,order+1)])
    phase,tangent=phases(bary,points,velocity,modes,period)
    with localcontext() as context:
        context.prec=100
        pi=Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
        dec=lambda value:Decimal.from_float(float(value))
        def trig(argument):
            sine=term=argument;cosine=cterm=Decimal(1)
            for k in range(1,100):
                term*=-argument*argument/((2*k)*(2*k+1));sine+=term
                cterm*=-argument*argument/((2*k-1)*(2*k));cosine+=cterm
            return cosine,sine
        for q,row in enumerate(bary):
            values=list(map(dec,row));anchor=int(np.argmax(abs(row)))
            values[anchor]=Decimal(1)-sum(values[i] for i in range(3) if i!=anchor)
            for m,mode in enumerate(modes):
                argument=2*pi*sum(sum(values[i]*dec(points[i,a]) for i in range(3))*int(mode[a])/dec(period[a]) for a in range(2))
                delta=2*pi*sum(sum(values[i]*dec(velocity[i,a]) for i in range(3))*int(mode[a])/dec(period[a]) for a in range(2))
                cosine,sine=trig(argument)
                for component in range(2):
                    index=(2*q+component,component*len(modes)+m)
                    for observed,expected in ((phase,(cosine,sine)),(tangent,(-delta*sine,delta*cosine))):
                        for part,value in zip(('real','imag'),expected):
                            actual=dec(getattr(observed[0][index],part))+dec(getattr(observed[1][index],part))
                            assert abs(actual-value)<3*Decimal('1e-29'),(q,m,component,part,actual-value)
