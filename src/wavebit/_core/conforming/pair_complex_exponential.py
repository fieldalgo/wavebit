"""Complete complex exponentials with real compensated factors.

Real exponential range reduction and quadrant-reduced real sine/cosine
polynomials evaluate every component. Coefficients retain the exact factorial
remainders. The reduced exponential keeps degree 26; sine and cosine keep
all terms through degrees 31 and 30 respectively. No mode is selected by size.
"""
import numpy as np
from .pair_corner_precision import D
from .pair_complex_precision import C
from .pair_contact_quadrature import PI
from decimal import Decimal,localcontext
from math import factorial
with localcontext() as context:
    context.prec=90
    def coefficient(value):
        hi=float(value);return D(hi,float(value-Decimal.from_float(hi)))
    _exp_coefficients=[coefficient(Decimal(1)/factorial(n)) for n in range(27)]
    _cos_coefficients=[coefficient(Decimal((-1)**n)/factorial(2*n)) for n in range(16)]
    _sin_coefficients=[coefficient(Decimal((-1)**n)/factorial(2*n+1)) for n in range(16)]

def exponential(argument):
    argument=C(argument)
    if not all(np.all(np.isfinite(v)) for v in argument.pair):
        raise ArithmeticError('complex exponential requires finite coordinates')
    real=argument.real;angle=argument.imag
    bound=abs(real.hi);powers=np.zeros(bound.shape,int);large=bound>.25
    powers[large]=np.ceil(np.log2(bound[large]/.25)).astype(int)
    reduced=D(np.ldexp(real.hi,-powers),np.ldexp(real.lo,-powers))
    def polynomial(x,coefficients):
        value=D(np.broadcast_to(coefficients[-1].hi,x.hi.shape).copy(),np.broadcast_to(coefficients[-1].lo,x.hi.shape).copy())
        for coefficient in coefficients[-2::-1]:value=value*x+coefficient
        return value
    if not np.any(real.hi) and not np.any(real.lo):radial=D(np.ones_like(bound))
    else:
        radial=polynomial(reduced,_exp_coefficients)
        for n in range(int(powers.max(initial=0))):
            active=powers>n;radial[active]=radial[active]*radial[active]
    if not np.any(angle.hi) and not np.any(angle.lo):
        cosine=D(np.ones_like(bound));sine=D(np.zeros_like(bound))
    else:
        quarter=np.rint(angle.hi/(np.pi/2));residual=angle-D(quarter)*PI/2;square=residual*residual
        cosine=polynomial(square,_cos_coefficients);sine=residual*polynomial(square,_sin_coefficients)
        phase=quarter.astype(np.int64)%4
        def orient(c,s):
            return tuple(np.select([phase==0,phase==1,phase==2],[a,-b,-a],default=b) for a,b in zip(c.pair,s.pair))
        cosine,sine=D(*orient(cosine,sine)),D(*orient(sine,-cosine))
    re,im=radial*cosine,radial*sine
    return C(re.hi+1j*im.hi,re.lo+1j*im.lo)
