"""Complete complex high/low arithmetic for contact geometry expressions.

The real and imaginary parts retain their product and sum remainders. This
module uses the existing NumPy scalar kernels and introduces no dependency.
"""
import numpy as np
from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_oscillatory_scalar import cp
from wavebit.rcwa._engine.conforming.pair_oscillatory_scalar import exp_pair
from wavebit.rcwa._engine.conforming.pair_residual_arithmetic import _sum_pairs


class C:
    def __init__(self, hi, lo=None):
        if isinstance(hi, C):
            self.hi, self.lo = hi.hi, hi.lo
        elif isinstance(hi, D):
            self.hi, self.lo = hi.hi.astype(complex), hi.lo.astype(complex)
        else:
            self.hi = np.asarray(hi, complex)
            self.lo = np.zeros_like(self.hi) if lo is None else np.asarray(lo, complex)

    @property
    def pair(self):
        return self.hi, self.lo

    @property
    def real(self):
        return D(self.hi.real, self.lo.real)

    @property
    def imag(self):
        return D(self.hi.imag, self.lo.imag)

    def __getitem__(self, key):
        return C(self.hi[key], self.lo[key])

    def __add__(self, other):
        return C(*_sum_pairs([self.pair, C(other).pair]))

    __radd__ = __add__

    def __neg__(self):
        return C(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -C(other)

    def __rsub__(self, other):
        return C(other) + -self

    def __mul__(self, other):
        return C(*cp(self.pair, C(other).pair))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = C(other)
        quotient = C(self.hi / other.hi)
        remainder = self - other * quotient
        return quotient + C((remainder.hi + remainder.lo) / other.hi)

    def __rtruediv__(self, other):
        return C(other) / self

    def exp(self):
        value = C(*exp_pair((np.atleast_1d(self.hi), np.atleast_1d(self.lo))))
        return value[0] if self.hi.ndim == 0 else value
