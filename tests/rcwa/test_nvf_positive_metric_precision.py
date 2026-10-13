"""Independent Decimal complete metric factors and strict pivot rejection."""
from decimal import Decimal as Z, localcontext

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.pair_corner_precision import D
from wavebit.rcwa._engine.conforming.pair_positive_precision import cholesky

pytestmark = pytest.mark.research


@pytest.mark.parametrize('size,bound', ((3, '3e-28'), (6, '3e-28'), (10, '3e-25')))
def test_complete_metric_factor_matches_independent_decimal(size, bound):
    with localcontext() as context:
        context.prec = 100
        exact = [[Z(1)/(i+j+1) for j in range(size)] for i in range(size)]
        hi = np.array([[float(v) for v in row] for row in exact])
        lo = np.array([[float(v-Z.from_float(float(v))) for v in row] for row in exact])
        expected = [[Z(0)]*size for _ in range(size)]
        for j in range(size):
            expected[j][j] = (exact[j][j]-sum(expected[j][k]**2 for k in range(j))).sqrt()
            for i in range(j+1, size):
                expected[i][j] = (exact[i][j]-sum(expected[i][k]*expected[j][k]
                                                 for k in range(j)))/expected[j][j]
        actual = cholesky(D(hi, lo))
        error = max(abs(Z.from_float(actual.hi[i,j])+Z.from_float(actual.lo[i,j])
                        -expected[i][j]) for i in range(size) for j in range(i+1))
        assert error < Z(bound)
        residual = D(hi, lo)-actual@actual.T
        assert np.max(abs(residual.hi+residual.lo)) < 3e-28


@pytest.mark.parametrize('metric', (np.diag([1., 0.]), np.diag([1., -1e-30]),
                                   np.array([[1., 1.], [1., 1.]]),
                                   np.diag([1., np.nan])))
def test_complete_metric_rejects_bad_pivots_without_shifts_or_column_removal(metric):
    with pytest.raises(ArithmeticError, match='positive definite'):
        cholesky(metric)


@pytest.mark.parametrize('metric', (1., np.ones(3), np.ones((2, 3))))
def test_complete_metric_requires_a_square_matrix(metric):
    with pytest.raises(ValueError, match='square metric'):
        cholesky(metric)
