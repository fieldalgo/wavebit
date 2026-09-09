"""Static-cell diagnostics must not be mistaken for optical error bounds."""
import numpy as np
import pytest

from studies.nvf_cell_bounds import energy_bounds, fourier_bounds, projected_matrices


def test_uniform_cell_bounds_collapse_to_exact_material():
    result = fourier_bounds(3, [dict(lx=700., ly=700., epsilon=4.)])
    for key in ('upper', 'lower'):
        np.testing.assert_allclose(result[key], 4*np.eye(2), atol=2e-13, rtol=0)


def test_bar_bounds_enclose_exact_tensor_and_shrink_under_refinement():
    exact = np.diag([1/(.5+.5/12.25), .5+.5*12.25])
    previous = None
    for inner in (7, 15, 31):
        result = fourier_bounds(inner, [dict(lx=350., ly=700., epsilon=12.25)])
        assert np.linalg.eigvalsh(result['upper']-exact).min() > -1e-11
        assert np.linalg.eigvalsh(exact-result['lower']).min() > -1e-11
        if previous is not None:
            assert np.linalg.eigvalsh(previous['upper']-result['upper']).min() > -1e-11
            assert np.linalg.eigvalsh(result['lower']-previous['lower']).min() > -1e-11
            assert result['relative_gap'] < previous['relative_gap']
        previous = result


def test_material_partition_does_not_change_cell_bounds():
    merged = [dict(lx=600., ly=300., cx=21., epsilon=4.)]
    split = [dict(lx=300., ly=300., cx=cx, epsilon=4.) for cx in (-129., 171.)]
    left, right = fourier_bounds(7, merged), fourier_bounds(7, split)
    for key in ('upper', 'lower'):
        np.testing.assert_allclose(left[key], right[key], atol=2e-12, rtol=0)


@pytest.mark.parametrize('upper,dual', [
    (np.eye(2), .5*np.eye(2)),
    (np.array([[1., 1.], [0., 1.]]), np.eye(2)),
    (-np.eye(2), np.eye(2)),
    (np.full((2, 2), np.nan), np.eye(2)),
])
def test_invalid_energy_enclosures_are_rejected(upper, dual):
    with pytest.raises(ValueError):
        energy_bounds(upper, dual)


def test_projected_uniform_matrices_and_invalid_exports():
    c = np.zeros((8, 9, 9), complex)
    c[[0, 3], 4, 4] = 1
    c[[4, 6, 7], 4, 4] = 4
    identity = np.eye(25)
    for actual, expected in zip(projected_matrices(c, 2),
                                (4*identity, 4*identity, identity/4, 0*identity, 0*identity)):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match='cap'):
        projected_matrices(c, 3)
    with pytest.raises(ValueError, match='eight'):
        projected_matrices(c[:7], 2)
    c[0, 0, 0] = 1j
    with pytest.raises(ValueError, match='real-field'):
        projected_matrices(c, 2)
