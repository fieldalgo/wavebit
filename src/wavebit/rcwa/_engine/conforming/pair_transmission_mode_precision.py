"""Complete fixed roots and continuous sector states for transmission fields.

Divide the monodromy defect by the quarter-sector contrast algebraically.
This keeps the full noninteger mode when nearly equal coefficients make a
numerically formed transfer matrix indistinguishable from the identity.
The ordinary root selects the existing branch; complete arithmetic refines
that root and propagates every sector state without fitting sampled fields.
"""
import cmath
import numpy as np
from wavebit.rcwa._engine.conforming.pair_complex_precision import C
from wavebit.rcwa._engine.conforming.pair_contact_quadrature import PI


def _value(x):
    return np.asarray(x.hi+x.lo).item()


def _sqrt(x):
    root = C(cmath.sqrt(_value(x)))
    if _value(root) == 0:
        return root
    for _ in range(3):
        root = root+(x-root*root)/(2*root)
    return root


def _sincos(x):
    """Resolve a small root separation without subtracting exponentials."""
    count = max(0, int(np.ceil(np.log2(max(abs(_value(x)), .25)/.25))))
    x = x/(2**count)
    square = x*x
    sine = sine_term = x
    cosine = cosine_term = C(1.)
    for k in range(1, 25):
        sine_term = -sine_term*square/((2*k)*(2*k+1))
        cosine_term = -cosine_term*square/((2*k-1)*(2*k))
        sine = sine+sine_term
        cosine = cosine+cosine_term
    for _ in range(count):
        sine, cosine = 2*sine*cosine, cosine*cosine-sine*sine
    return sine, cosine


def parameters(coefficients, branch, ordinary_root):
    """Return the selected complete exponent, sector states and coefficients."""
    a, b, c = (C(v)/C(coefficients[-1]) for v in coefficients)
    delta = a-b
    factor = c/(2*(a+b)*(a+c)*(b+c))
    h = _sqrt(factor)
    t = delta*h
    target = cmath.cos(ordinary_root*np.pi/2)
    if abs(_value(t)-target) > abs(-_value(t)-target):
        h, t = -h, -t
    u = t*t
    s = _sqrt(1-u)
    root = C(ordinary_root)
    for _ in range(4):
        sine, cosine = _sincos((1-root)*C(PI)/2)
        root = root+(sine-t)/(C(PI)/2*cosine)
    exponent = C(2*((branch+1)//2))+(-root if branch % 2 else root)

    # M-I is proportional to delta. Form its quotient directly, using
    # t=cos(pi*root/2)=delta*h, s=sin(pi*root/2), and u=t*t.
    # No nearly equal identity matrices or weak field cutoff is involved.
    hs = h*s*(-1 if branch % 2 else 1)
    m00 = 1/b+delta*factor*(2*u*(a+b)*(b+c)/(b*c)-(1+3*a/b+2*(a+b)/c))
    m11 = -1/a+delta*factor*(2*u*(a+b)*(b+c)/(a*b)-(1+3*b/a+2*c*(a+b)/(a*b)))
    m01 = hs*((2*u-1)*(a+b)/(a*b)+2/c*(u-b/a*(1-u)))
    m10 = -hs*(2*c*(u-a/b*(1-u))+(2*u-1)*(a+b))
    if abs(_value(m00))+abs(_value(m01)) >= abs(_value(m10))+abs(_value(m11)):
        state = (-m01, m00)
    else:
        state = (-m11, m10)
    pivot = max(state, key=lambda value: abs(_value(value)))
    state = tuple(value/pivot for value in state)

    # All four branches share the same squared quarter-angle cosine.
    # Keep their exact signs when propagating through the full turn.
    quarter = (t*(1, -1, -1, 1)[branch], s*(1, 1, -1, -1)[branch])
    half = (2*u-1, 2*t*s*(-1 if branch % 2 else 1))
    states = [state]
    for coefficient, (cosine, sine) in zip((a, b, c), (quarter, quarter, half)):
        potential, flux = states[-1]
        states.append((cosine*potential+sine/coefficient*flux,
                       -coefficient*sine*potential+cosine*flux))
    return exponent, tuple(states), (a, b, c)
