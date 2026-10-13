"""Static-cell diagnostics must not be mistaken for optical error bounds."""
import numpy as np
import pytest

from studies.rcwa.nvf_cell_bounds.nvf_cell_bounds import energy_bounds, fourier_bounds, projected_matrices

pytestmark = pytest.mark.research


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


@pytest.mark.parametrize('order', [(0, 0), (3, 1), (1, 3), (7, 4), (4, 7)])
def test_projected_fields_obey_energy_identity_across_assembly_tiles(order):
    # A nonsymmetric real change of field with no reflection reduction. Build
    # its Gram field pointwise, independently of the closure's linear solves.
    cap = 2*max(order)+6
    size = 2*cap+1
    y, x = np.meshgrid(2*np.pi*np.arange(size)/size, 2*np.pi*np.arange(size)/size, indexing='ij')
    bxx, bxy = 1+.15*np.cos(x), .1*np.sin(x+y)
    byx, byy = .12*np.cos(2*x-y), 1+.2*np.sin(y)
    epsilon = 3+.4*np.cos(x-2*y)
    fields = np.array([bxx, bxy, byx, byy, epsilon*(bxx*bxx+byx*byx),
                       epsilon*(bxx*bxy+byx*byy), epsilon*(bxy*bxy+byy*byy), epsilon])
    c = np.fft.fftshift(np.fft.fft2(fields), axes=(-2, -1))/size**2
    original = c.copy()
    # Explicit mode enumeration also checks the rectangular ordering and the
    # final short tile (135 modes for (7,4) and (4,7)).
    modes = np.array([(m, n) for n in range(-order[1], order[1]+1)
                      for m in range(-order[0], order[0]+1)])
    conv = [np.array([[row[cap+ni-nj, cap+mi-mj] for mj, nj in modes] for mi, ni in modes]) for row in c]
    change = np.block([[conv[0], conv[1]], [conv[2], conv[3]]])
    gram = np.block([[conv[4], conv[5]], [conv[5], conv[6]]])
    xx, yy, zi, neg_xy, neg_yx = projected_matrices(c, order)
    effective = np.block([[xx, -neg_xy], [-neg_yx, yy]])
    np.testing.assert_allclose(change.conj().T@effective@change, gram, atol=2e-13, rtol=0)
    np.testing.assert_allclose(effective, effective.conj().T, atol=2e-13, rtol=0)
    np.testing.assert_allclose(conv[7]@zi, np.eye(len(modes)), atol=2e-13, rtol=0)
    np.testing.assert_array_equal(c, original)


@pytest.mark.parametrize('singular_row', [0, 3, 7])
def test_singular_projected_fields_still_raise(singular_row):
    c = np.zeros((8, 1, 1), complex)
    c[[0, 3], 0, 0] = 1
    c[[4, 6, 7], 0, 0] = 4
    c[singular_row] = 0
    with pytest.raises(np.linalg.LinAlgError):
        projected_matrices(c, 0)
