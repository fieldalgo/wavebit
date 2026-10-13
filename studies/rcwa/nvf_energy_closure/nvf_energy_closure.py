"""Memory-conscious exact energy congruence for NumPy NVF research solves."""
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_solve, norm
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs


def congruence_solve(change, gram, *, overwrite=False):
    """Compute Q^(-H) G Q^(-1) without forming Q's inverse.

    One LU factorization supplies the adjoint and right-side solves. No
    Hermitian projection or positive-definiteness assumption is introduced.
    With overwrite=True, callers relinquish their two input arrays; supplying
    Fortran-contiguous arrays lets LAPACK/BLAS reuse them. The returned array
    reuses the Gram matrix storage when possible.
    """
    change, gram = np.asarray(change), np.asarray(gram)
    if (change.ndim != 2 or change.shape[0] != change.shape[1] or not len(change)
            or gram.shape != change.shape or not np.isfinite(change).all() or not np.isfinite(gram).all()):
        raise ValueError('energy congruence requires finite, equally sized nonempty square matrices')
    dtype = np.result_type(change.dtype, gram.dtype, np.float64)
    q = np.asfortranarray(change, dtype=dtype) if overwrite else np.array(change, dtype=dtype, order='F', copy=True)
    g = np.asfortranarray(gram, dtype=dtype) if overwrite else np.array(gram, dtype=dtype, order='F', copy=True)
    if np.shares_memory(q, g):
        g = g.copy(order='F')
    # Reusing LU must not discard the singularity/conditioning diagnostics
    # supplied by scipy.linalg.solve in the explicit-inverse implementation.
    qnorm = norm(q, 1, check_finite=False)
    getrf, gecon = get_lapack_funcs(('getrf', 'gecon'), (q,))
    lu, pivots, info = getrf(q, overwrite_a=True)
    if info > 0:
        raise np.linalg.LinAlgError('singular NVF change-of-field matrix')
    if info < 0:
        raise ValueError(f'LU factorization reported invalid argument {-info}')
    rcond, info = gecon(lu, qnorm, norm='1')
    if info:
        raise ValueError(f'LU condition estimate failed with code {info}')
    if not np.isfinite(rcond) or rcond < np.finfo(q.real.dtype).eps:
        warnings.warn(f'Ill-conditioned NVF change-of-field matrix: rcond={rcond}', LinAlgWarning, stacklevel=2)
    factors = lu, pivots
    left = lu_solve(factors, g, trans=2, overwrite_b=True, check_finite=False)
    # Right-side triangular solves avoid a dense transpose copy for GETRS.
    # P Q = L U, so D = left U^-1 L^-1 P. These are ordinary, not
    # conjugate, triangular solves. Right multiplication applies the pivot
    # swaps in reverse order, because P = P_last ... P_first.
    trsm = get_blas_funcs('trsm', (lu, left))
    right = trsm(1., lu, left, side=1, lower=0, overwrite_b=True)
    right = trsm(1., lu, right, side=1, lower=1, diag=1, overwrite_b=True)
    for i in range(len(pivots)-1, -1, -1):
        if pivots[i] != i:
            right[:, [i, pivots[i]]] = right[:, [pivots[i], i]]
    return right


def hermitian_defect(matrix):
    """Maximum absolute Hermitian defect with bounded temporary storage."""
    matrix = np.asarray(matrix)
    error = 0.
    for start in range(0, len(matrix), 128):
        block = matrix[start:start+128]-matrix[:, start:start+128].conj().T
        error = np.maximum(error, np.max(abs(block)))
    return float(error)
