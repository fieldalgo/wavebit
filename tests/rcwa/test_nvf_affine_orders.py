"""Exact retained Fourier sets for affine-cell matrix export."""
import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.affine_cell import AffineCellGapModel
from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import projected_matrices
from tests.rcwa.test_nvf_affine_cell import BASIS, SETTINGS
from tests.rcwa.test_conforming_nvf import solve

pytestmark = pytest.mark.research


@pytest.mark.parametrize('order', [(2, 1), (1, 2), (1, 0), (0, 1), (0, 0)])
@pytest.mark.parametrize('gap', [0., 17.])
def test_directional_export_closes_the_exact_fourier_set_before_inversion(order, gap):
    model = AffineCellGapModel(order, **SETTINGS)
    assert model.truncation_order == order
    actual = model.conv_matrices(gap, 4., basis=700*BASIS)
    c, _ = AffineCellGapModel(max(order), **SETTINGS).coefficients(gap, 4., basis=700*BASIS)
    expected = projected_matrices(c[:8], order)
    count = (2*order[0]+1)*(2*order[1]+1)
    for got, want in zip(actual, expected):
        assert got.shape == (1, count, count)
        np.testing.assert_allclose(got[0], want, atol=2e-11, rtol=0)
    powers = solve(actual, 700*BASIS, order, kernel_frame=True)
    np.testing.assert_allclose(powers.sum(axis=(1, 2, 3)), 1., atol=2e-10, rtol=0)


def test_directional_export_is_not_a_submatrix_of_an_already_closed_square():
    square = AffineCellGapModel(2, **SETTINGS).conv_matrices(17., 4., basis=700*BASIS)
    directional = AffineCellGapModel((2, 1), **SETTINGS).conv_matrices(17., 4., basis=700*BASIS)
    n, m = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    keep = np.flatnonzero(abs(n.ravel()) <= 1)
    cropped = [slot[0][np.ix_(keep, keep)] for slot in square]
    assert any(not np.allclose(got[0], wrong, atol=1e-9, rtol=1e-9) for got, wrong in zip(directional, cropped))


@pytest.mark.parametrize('order', [True, -1, 1.5, (1,), (1, 2, 3), (1, -1), (True, 1), (1., 2), None])
def test_invalid_directional_orders_raise(order):
    with pytest.raises(ValueError, match='order'):
        AffineCellGapModel(order, **SETTINGS)
