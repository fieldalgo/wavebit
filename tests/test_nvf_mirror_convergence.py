"""Independent slab and unreduced-Maxwell controls for mirror acceleration."""
import numpy as np
import pytest
from scipy.linalg import block_diag

from studies.nvf_cell_bounds import projected_matrices
from studies.nvf_fourier_orders import order_grid, orders_xy
from studies.nvf_gap_convergence import power_record
from studies.nvf_harmonic_energy import PERIOD, THETA, WAVELENGTH
from studies.nvf_mirror_convergence import mirror_powers, mirror_sector, parity_basis, parity_convolution, sector_geometry, validate_mirror
from studies.nvf_polynomial_body import PolynomialBodyGapModel


def dense_parity(order, parity):
    n, m = order_grid(order)
    x, y = orders_xy(order)
    columns = []
    for j in range(0 if parity == 1 else 1, y+1):
        for i in range(-x, x+1):
            positive = ((n == j) & (m == i)).ravel().astype(float)
            negative = ((n == -j) & (m == i)).ravel().astype(float)
            columns.append(positive if j == 0 else (positive+parity*negative)/np.sqrt(2))
    return np.column_stack(columns) if columns else np.zeros((n.size, 0))


@pytest.mark.parametrize('order', [(0, 3), (3, 0), (2, 4), (7, 9)])
def test_direct_convolution_matches_explicit_orthonormal_fourier_basis(order):
    n, m = order_grid(order)
    cap = 2*max(order)
    c = np.random.default_rng(732).normal(size=(2*cap+1, 2*cap+1)).astype(complex)
    c += 1j*np.random.default_rng(83).normal(size=c.shape)
    full = c[n.ravel()[:, None]-n.ravel()+cap, m.ravel()[:, None]-m.ravel()+cap]
    u = {parity: dense_parity(order, parity) for parity in (1, -1)}
    all_basis = np.column_stack([u[1], u[-1]])
    np.testing.assert_allclose(all_basis.T@all_basis, np.eye(n.size), atol=3e-16, rtol=0)
    for row in (1, -1):
        for column in (1, -1):
            got = parity_convolution(c, parity_basis(order, row), parity_basis(order, column))
            np.testing.assert_allclose(got, u[row].T@full@u[column], atol=2e-15, rtol=0)


@pytest.mark.parametrize('order', [(0, 3), (3, 0), (2, 4)])
def test_mirror_closure_is_the_restriction_of_the_same_full_finite_closure(order):
    c, _ = PolynomialBodyGapModel(max(order), rows=4).coefficients(17., dict(seams=[31.], values=[4., 9.]))
    xx, yy, zi, xy, yx = projected_matrices(c, order)
    full = np.block([[xx, -xy], [-yx, yy]])
    for pol, parity in (('p', 1), ('s', -1)):
        sector = mirror_sector(c, order, pol)
        a, b = dense_parity(order, parity), dense_parity(order, -parity)
        u = block_diag(a, b)
        np.testing.assert_allclose(sector['effective'], u.T@full@u, atol=3e-11, rtol=0)
        np.testing.assert_allclose(sector['z_inverse'], a.T@zi@a, atol=3e-13, rtol=0)


@pytest.mark.parametrize('order', [0, (0, 3), (3, 0), (2, 4)])
def test_sparse_ky_map_matches_full_fourier_multiplication(order):
    n, _ = order_grid(order)
    for parity in (1, -1):
        a, b = parity_basis(order, parity), parity_basis(order, -parity)
        ua, ub = dense_parity(order, parity), dense_parity(order, -parity)
        expected = ua.T@((WAVELENGTH/PERIOD*n.ravel())[:, None]*ub)
        *_, dense = sector_geometry(dict(a=a, b=b))
        *_, sparse = sector_geometry(dict(a=a, b=b), sparse=True)
        np.testing.assert_array_equal(sparse.toarray(), dense)
        np.testing.assert_allclose(sparse.toarray(), expected, atol=2e-15, rtol=0)
        assert sparse.nnz <= min(len(a['n']), len(b['n']))


@pytest.mark.parametrize('order', [0, (0, 3), (3, 0), (2, 4)])
@pytest.mark.parametrize('thickness', [0., 300., 2400.])
def test_uniform_mirror_slab_matches_independent_airy_formula(order, thickness, monkeypatch):
    monkeypatch.setattr('studies.nvf_mirror_convergence.THICKNESS', thickness)
    cap = 2*max(orders_xy(order))
    c = np.zeros((8, 2*cap+1, 2*cap+1), complex)
    c[[0, 3], cap, cap] = 1
    c[[4, 6, 7], cap, cap] = 4
    result = mirror_powers(c, order)
    epsilon = np.array([1., 4., 2.25])
    kz = np.sqrt(epsilon-np.sin(THETA)**2)
    for pol in ('p', 's'):
        top, body, bottom = epsilon/kz if pol == 'p' else kz
        phase = np.exp(2j*np.pi*thickness/WAVELENGTH*kz[1])
        amplitude = 4*top*body*phase/((top+body)*(body+bottom)+(top-body)*(body-bottom)*phase**2)
        assert result[pol+'_T'] == pytest.approx(bottom/top*abs(amplitude)**2, abs=4e-13, rel=0)
    assert result['energy_error'] < 3e-13


@pytest.mark.parametrize('order,gap', [(2, 0.), (3, 17.), ((2, 4), 17.), ((4, 2), 400.)])
def test_both_mirror_polarizations_match_unreduced_scattering(order, gap):
    c, _ = PolynomialBodyGapModel(max(orders_xy(order)), rows=4).coefficients(gap, dict(seams=[31.], values=[4., 9.]))
    actual = mirror_powers(c, order)
    expected = power_record(order, projected_matrices(c, order))
    for key in ('p_T', 's_T', 'p_R', 's_R'):
        assert actual[key] == pytest.approx(expected[key], abs=2e-10, rel=0)


def test_asymmetric_or_invalid_coefficients_are_refused_without_symmetrizing():
    c, _ = PolynomialBodyGapModel(3, rows=4).coefficients(17., dict(seams=[31.], values=[4., 9.]))
    n = np.arange(-6, 7)
    shifted = c*np.exp(-2j*np.pi*n*.07)[None, :, None]
    with pytest.raises(ValueError, match='y-reflection'):
        mirror_powers(shifted, 3)
    with pytest.raises(ValueError, match='cap'):
        mirror_powers(c, 4)
    with pytest.raises(ValueError, match='eight square'):
        validate_mirror(c[:, :-1], 2)
    invalid = c.copy()
    invalid[0, 6, 6] += .01j
    with pytest.raises(ValueError, match='real-field'):
        validate_mirror(invalid, 2)
    with pytest.raises(ValueError, match='polarization'):
        mirror_sector(c, 2, 'invalid')


def test_near_grazing_mode_is_refused_instead_of_changing_the_root():
    c = np.zeros((8, 5, 5), complex)
    c[[0, 3], 2, 2] = 1
    c[[4, 6, 7], 2, 2] = np.sin(THETA)**2
    with pytest.raises(FloatingPointError, match='near-grazing'):
        mirror_powers(c, 1)


@pytest.mark.filterwarnings('error::scipy.linalg.LinAlgWarning')
def test_mirror_retains_both_independent_contact_derivatives_at_the_final_resolution():
    from studies.nvf_gap_convergence import profile_li
    model = PolynomialBodyGapModel(7, rows=8, body_degree=4, gap_degree=6, x_subdivisions=3, edge_fraction=1e-5)
    candidate, reference = [], []
    for gap in (0., 1e-4, 5e-5):
        c, _ = model.coefficients(gap, 12.25)
        fast = mirror_powers(c, 7)
        control = power_record(7, profile_li(7, gap, 350., [], [12.25]))
        candidate.append([fast['p_T'], fast['s_T']])
        reference.append([control['p_T'], control['s_T']])
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    np.testing.assert_allclose(candidate[0], reference[0], atol=2e-10, rtol=0)
    def derivative(values):
        return 2*(values[2]-values[0])/5e-5-(values[1]-values[0])/1e-4
    np.testing.assert_allclose(derivative(candidate), derivative(reference), atol=1e-7, rtol=0)
