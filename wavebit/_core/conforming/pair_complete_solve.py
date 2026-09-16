"""Complex high/low partial-pivot elimination retaining every coordinate.

The finite retained block can be ill-conditioned even when the physical
field is well-conditioned. Factor both words, rather than using a singular
or inaccurate leading-word factor as a residual-correction preconditioner.
A zero pivot raises; no numerical rank, pivot shift or field cutoff is used.
"""
import numpy as np
from .pair_complex_precision import C


def factor(matrix):
    """Factor the complete square matrix, recording every row permutation."""
    a = C(*(value.copy() for value in matrix))
    if (a.hi.ndim != 2 or a.hi.shape[0] != a.hi.shape[1]
            or a.lo.shape != a.hi.shape
            or not all(np.all(np.isfinite(value)) for value in a.pair)):
        raise ValueError('finite square complete matrix required')
    a = a+0
    n, swaps = len(a.hi), []
    for k in range(n):
        row = k+int(np.argmax(abs(a.hi[k:, k]+a.lo[k:, k])))
        if not (a.hi[row, k] != 0 or a.lo[row, k] != 0):
            raise np.linalg.LinAlgError('singular complete matrix')
        swaps.append(row)
        if row != k:
            for part in a.pair:
                part[[k, row]] = part[[row, k]]
        column = a[k+1:, k]/a[k, k]
        update = a[k+1:, k+1:]-column[:, None]*a[k, k+1:][None, :]
        for part, multiplier, tail in zip(a.pair, column.pair, update.pair):
            part[k+1:, k] = multiplier
            part[k+1:, k+1:] = tail
    if not all(np.all(np.isfinite(value)) for value in a.pair):
        raise np.linalg.LinAlgError('nonfinite complete factor')
    return a, swaps


def apply(factorization, rhs):
    """Apply the same full factorization to every right-hand-side column."""
    a, swaps = factorization
    b = C(*(value.copy() for value in rhs))
    n = len(a.hi)
    if (b.hi.ndim != 2 or len(b.hi) != n or b.lo.shape != b.hi.shape
            or not all(np.all(np.isfinite(value)) for value in b.pair)):
        raise ValueError('finite compatible complete right-hand side required')
    b = b+0
    for k, row in enumerate(swaps):
        if row != k:
            for part in b.pair:
                part[[k, row]] = part[[row, k]]
    for k in range(n):
        update = b[k+1:]-a[k+1:, k, None]*b[k][None, :]
        for part, value in zip(b.pair, update.pair):
            part[k+1:] = value
    for k in range(n-1, -1, -1):
        row = b[k]/a[k, k]
        update = b[:k]-a[:k, k, None]*row[None, :]
        for part, value, top in zip(b.pair, row.pair, update.pair):
            part[k] = value
            part[:k] = top
    return b.pair


def solve(matrix, rhs):
    return apply(factor(matrix), rhs)
