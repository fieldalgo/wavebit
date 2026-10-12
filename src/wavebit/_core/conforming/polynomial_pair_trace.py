"""Analytic polynomial remainders for the complete real pair trace space.

Subtract representable cubic/quartic nodal bubbles before evaluating nearly
coincident restricted trigonometric fields. This is an invertible geometry-only
coordinate change, with no lost trace column and no material-dependent frame.
The NumPy polynomial kernel uses only positive, bounded power-series sums.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss
from .periodic_seam_trace import PeriodicSeamTrace


def _even_coefficients(count, frequency, amplitude, half_width):
    """Chebyshev coefficients of asin(a*u)**2/k**2 - half_width**2.

    For j>=1, c[2*j] = sum_n a**(2*n)/k**2/n**2 times
    binom(2*n,n-j)/binom(2*n,n). All summands are positive. Keeping the
    entire degree-2*count polynomial leaves only the original Taylor tail.
    """
    coefficients = np.zeros(2*count+1)
    power = amplitude**2
    for n in range(1, count+1):
        term = power/(frequency*n)**2
        coefficients[0] += term/2
        ratio = 1.
        for j in range(1, n+1):
            ratio *= (n-j+1)/(n+j)
            coefficients[2*j] += term*ratio
        power *= amplitude**2
    coefficients[0] -= half_width**2
    return coefficients


def _tail_bounds(count, frequency, half_width, q):
    """Bound both bubbles' values and two y derivatives beyond this Taylor degree.

    All asin(z)**2 Taylor coefficients are at most one. Its first three
    z derivatives are bounded by 8*sum(n**3*q**n)/q**1.5. Leibniz' rule
    then bounds the differentiated square used by the odd bubble.
    """
    n, r = count+1, 1-q
    sum_cubes = q**n*(n**3/r+3*n*n*q/r**2+3*n*q*(1+q)/r**3
                      +q*(1+4*q+q*q)/r**4)
    tail = 8*sum_cubes/(q**1.5*frequency**2)
    z, root = np.sqrt(q), np.sqrt(r)
    angle = np.arcsin(z)
    bounds = (half_width**2+(angle/frequency)**2,
              2*angle/(frequency**2*root),
              2/frequency**2*(1/r+angle*z/root**3),
              2/frequency**2*(3*z/r**2+angle/root**3+3*angle*z*z/root**5))
    even = max(1., frequency, 2*frequency**2)*tail
    odd = 20*max(frequency, frequency**2, frequency**3)*(max(bounds)*tail+tail**2/2)
    return even, odd


@lru_cache(maxsize=16)
def _frame(order, degree, period, width, quadrature):
    frequency = np.pi/period
    amplitude = np.sin(frequency*width/2)
    # The taper is always strictly below this cap. The bound therefore covers
    # every positive gap using one fixed, gap-independent coordinate chart.
    outer = width/2+min(width, period-width)/4
    q = np.sin(frequency*outer)**2
    if not 0 < q < 1:
        raise ValueError('polynomial trace analytic interval is unresolved')
    count = max(16, 2*order+4)
    while True:
        coefficients = _even_coefficients(count, frequency, amplitude, width/2)
        scale = abs(coefficients[2*order+2])
        odd, odd_scale = None, None
        if degree == 4:
            # y*(y*y-h*h/4) = k*cos(k*y)/(4*a) * d(B2**2)/du.
            odd = frequency/(4*amplitude)*cheb.chebder(cheb.chebmul(coefficients, coefficients))
            odd_scale = abs(odd[2*order+1])
        even_tail, odd_tail = _tail_bounds(count, frequency, width/2, q)
        tolerance = np.finfo(float).eps/16
        if (scale > 0 and even_tail < tolerance*scale
                and (degree != 4 or odd_scale > 0 and odd_tail < tolerance*odd_scale)):
            break
        count *= 2
        if count > 4096:
            raise ValueError('polynomial trace series tail is unresolved')
    residual = coefficients.copy()
    residual[:2*order+1] = 0.
    residual /= -scale
    if degree == 4:
        odd[:2*order] = 0.
        odd /= -odd_scale
    data = dict(frequency=frequency, amplitude=amplitude, residual=residual,
                odd=odd, scale=scale, odd_scale=odd_scale, width=width)
    nodes, weights = leggauss(quadrature)
    raw = _raw_basis(width*nodes/2, order, data)[0]
    upper = np.linalg.qr(np.sqrt(width*weights/2)[:,None]*raw, mode='r')
    data['frame'] = np.linalg.solve(upper, np.eye(2*order+1))
    gram = (raw@data['frame']).T@((width*weights/2)[:,None]*(raw@data['frame']))
    if np.linalg.norm(gram-np.eye(2*order+1)) > 1e-10:
        raise ValueError('polynomial residual trace frame is unresolved')
    return data


def _raw_basis(y, order, data):
    frequency, amplitude = data['frequency'], data['amplitude']
    angle = frequency*y
    cosine, sine = np.cos(angle), np.sin(angle)
    u = sine/amplitude
    du, ddu = frequency*cosine/amplitude, -frequency**2*u
    values = []
    first = []
    second = []
    for index in range(2*order+1):
        if index == 0:
            coefficients = data['residual']
        elif index == 1 and data['odd'] is not None:
            coefficients = data['odd']
        else:
            coefficients = np.zeros(index+1)
            coefficients[index] = 1.
        p = cheb.chebval(u, coefficients)
        dp = cheb.chebval(u, cheb.chebder(coefficients))
        ddp = cheb.chebval(u, cheb.chebder(coefficients, 2))
        v, d, dd = p, dp*du, ddp*du**2+dp*ddu
        if index % 2:
            v, d, dd = (cosine*v, -frequency*sine*v+cosine*d,
                        -frequency**2*cosine*v-2*frequency*sine*d+cosine*dd)
        values.append(v); first.append(d); second.append(dd)
    return tuple(np.column_stack(value) for value in (values, first, second))


class PolynomialPairTrace(PeriodicSeamTrace):
    """Same trace span modulo exact nodal bubbles, evaluated as remainders."""

    def __init__(self, order, degree, *, quadrature=64):
        super().__init__(order, quadrature=quadrature)
        self.degree = degree

    def _geometry(self, gap, period, width, offset):
        geometry = super()._geometry(gap, period, width, offset)
        if self.order:
            geometry['polynomial_frame'] = _frame(self.order, self.degree, geometry['px'],
                                                 geometry['width'], self.quadrature)
        return geometry

    def _basis(self, y, geometry):
        data = geometry.get('polynomial_frame')
        if data is None:
            return super()._basis(y, geometry)
        return tuple(value@data['frame'] for value in _raw_basis(y, self.order, data))

    def tapered_basis(self, y, dy, geometry, hat, dhat, hy, dhy, side, fraction, delta):
        """Evaluate hat*P - indicator*B with a stable small polynomial cap.

        The subtracted potential f(x)*B(y)*indicator is an exact member of
        the original nodal space. B vanishes at both horizontal endpoints.
        Both material sides remain in the moving quadrature partition.
        """
        if not side and not np.any(hat):
            zero = np.zeros((len(y), 2*self.order+1))
            return tuple(zero.copy() for _ in range(4))
        b, by, byy = self._basis(y, geometry)
        value = hat[:,None]*b
        change = dhat[:,None]*b+hat[:,None]*by*dy[:,None]
        first = hy[:,None]*b+hat[:,None]*by
        dfirst = (dhy[:,None]*b+hy[:,None]*by*dy[:,None]
                  +dhat[:,None]*by+hat[:,None]*byy*dy[:,None])
        data = geometry.get('polynomial_frame')
        if data is None or not side:
            return value, change, first, dfirst
        half = geometry['width']/2
        # Recover the small edge distance from the known taper fraction,
        # instead of subtracting two cell-sized rounded coordinates.
        distance = 2*delta*(fraction-.5)/side
        bubble = distance*(distance-side*2*half)
        bubble_y, bubble_yy = 2*y, np.full_like(y, 2.)
        inside = fraction >= .5
        beta = np.where(inside, -(1-fraction)**2*(1+2*fraction), hat)
        B = np.zeros_like(b); By = np.zeros_like(b); Byy = np.zeros_like(b)
        B[:,0], By[:,0], Byy[:,0] = (v/data['scale'] for v in (bubble, bubble_y, bubble_yy))
        if self.degree == 4:
            B[:,1] = y*bubble/data['odd_scale']
            By[:,1] = (bubble+y*bubble_y)/data['odd_scale']
            Byy[:,1] = 6*y/data['odd_scale']
        B, By, Byy = (v@data['frame'] for v in (B, By, Byy))
        value += beta[:,None]*B
        change += dhat[:,None]*B+beta[:,None]*By*dy[:,None]
        first += hy[:,None]*B+beta[:,None]*By
        dfirst += (dhy[:,None]*B+hy[:,None]*By*dy[:,None]
                   +dhat[:,None]*By+beta[:,None]*Byy*dy[:,None])
        return value, change, first, dfirst
