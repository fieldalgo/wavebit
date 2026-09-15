"""Fixed real trial directions from local scalar transmission equations.

These are auxiliary-potential exponents, not radiation square-root branches.
Reference coefficients define a fixed space; changing physical materials does
not differentiate or refit this space. NumPy is used only by the explicit
NumPy pair-cell family. The separated solution and transfer matrices are
derived here without an external eigensolver or copied implementation.
"""
import cmath
from numbers import Integral
import numpy as np


def transfer(exponent, coefficient, angle):
    """Propagate (potential, coefficient*angular_derivative/exponent)."""
    cosine, sine = cmath.cos(exponent*angle), cmath.sin(exponent*angle)
    return np.array([[cosine, sine/coefficient],
                     [-coefficient*sine, cosine]], complex)


class TransmissionMode:
    """A growing r**lambda mode in two quarter sectors and one half sector.

    The two noninteger root families are lambda+2k and 2-lambda+2k. Integer
    piecewise-polynomial solutions already belong to the fitted Pp space.
    The positive-real-part condition is the exact finite-energy requirement,
    not a tolerance for discarding weak modes.
    """

    angles = (np.pi/2, np.pi/2, np.pi)

    def __init__(self, coefficients, *, start=0., branch=0):
        coefficients = np.asarray(coefficients, complex)
        if (coefficients.shape != (3,) or not np.all(np.isfinite(coefficients))
                or np.any(coefficients == 0)):
            raise ValueError('transmission reference requires three finite nonzero coefficients')
        if (isinstance(branch, (bool, np.bool_)) or not isinstance(branch, Integral)
                or not 0 <= branch < 4):
            raise ValueError('transmission branch must be an integer from 0 through 3')
        if not np.isfinite(start):
            raise ValueError('transmission sector origin must be finite')
        # A common nonzero coefficient factor does not change the local
        # equation or its real trial span. Remove it before choosing real
        # directions, including purely imaginary multiples of real media.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            normalized = coefficients/coefficients[-1]
        if not np.all(np.isfinite(normalized)) or np.any(normalized == 0):
            raise ValueError('the relative reference coefficients must be representable')
        self.coefficients = tuple(normalized)
        self.start = float(start)
        a, b, c = self.coefficients
        first = a/b+b/a
        second = (a+b)/c+c*(1/a+1/b)
        denominator = 2+first+second
        if denominator == 0 or not np.isfinite(denominator):
            raise ValueError('the reference transmission characteristic is degenerate')
        characteristic = -(4+second)/denominator
        root = cmath.acos(characteristic)/np.pi
        if abs(characteristic+1) < np.sqrt(np.finfo(float).eps):
            # Near a homogeneous reference, acos loses the small separation
            # from the integer root. Factor (1+characteristic)/2 directly;
            # dimensionless quotients avoid squaring cell-sized coefficients.
            half = ((a-b)/(a+b))*((a-b)/(a+c))*(c/(b+c))/2
            root = 1-2*cmath.asin(cmath.sqrt(half))/np.pi
        self.exponent = 2*((branch+1)//2)+(-root if branch % 2 else root)
        if not np.isfinite(self.exponent) or self.exponent.real <= 0:
            raise ValueError('the reference transmission mode must have finite energy')
        matrix = np.eye(2, dtype=complex)
        for coefficient, angle in zip(self.coefficients, self.angles):
            matrix = transfer(self.exponent, coefficient, angle)@matrix
        _, _, right = np.linalg.svd(matrix-np.eye(2))
        state = right[-1].conj()
        state /= state[np.argmax(abs(state))]
        states = [state]
        for coefficient, angle in zip(self.coefficients, self.angles):
            states.append(transfer(self.exponent, coefficient, angle)@states[-1])
        if not np.allclose(states[-1], states[0], atol=2e-12, rtol=0):
            raise ArithmeticError('the complete reference mode does not close across all sectors')
        self.states = tuple(tuple(row) for row in states)

    def sample(self, points, *, sector, center_angle, hessian=False):
        """Return potential and Cartesian derivatives from one material side.

        A triangle supplies its material sector even on an interface: normal
        derivatives have distinct one-sided limits. Values alone cannot choose
        this side. The corner has no finite pointwise gradient and is excluded
        from Gaussian samples; its potential is zero for nodal interpolation.
        """
        points = np.asarray(points, float)
        radius = np.hypot(points[:, 0], points[:, 1])
        if np.any(radius == 0):
            raise ValueError('a transmission corner gradient is not a point value')
        theta = np.arctan2(points[:, 1], points[:, 0])
        angle = (theta-self.start) % (2*np.pi)
        angle += 2*np.pi*np.round((center_angle-angle)/(2*np.pi))
        local = angle-sum(self.angles[:sector])
        cosine, sine = np.cos(self.exponent*local), np.sin(self.exponent*local)
        u, flux = self.states[sector]
        coefficient = self.coefficients[sector]
        f = u*cosine+flux/coefficient*sine
        df = self.exponent*(-u*sine+flux/coefficient*cosine)
        power = np.exp(self.exponent*np.log(radius))
        value = power*f
        radial = self.exponent*power/radius*f
        angular = power/radius*df
        cosine, sine = np.cos(theta), np.sin(theta)
        gradient = np.stack([radial*cosine-angular*sine,
                             radial*sine+angular*cosine], axis=1)
        if not hessian:
            return value, gradient
        scale = (self.exponent-1)/radius
        xx = scale*(gradient[:, 0]*cosine-gradient[:, 1]*sine)
        xy = scale*(gradient[:, 0]*sine+gradient[:, 1]*cosine)
        second = np.empty((len(points), 2, 2), complex)
        second[:, 0, 0] = xx
        second[:, 0, 1] = second[:, 1, 0] = xy
        second[:, 1, 1] = -xx
        return value, gradient, second
