"""Directional Fourier truncation retains the physical problem and conventions."""
import numpy as np
import pytest

from studies.nvf_cell_bounds import projected_matrices
from studies.nvf_continuous_flux import ContinuousFluxGapModel
from studies.nvf_fourier_orders import order_grid, orders_xy, parse_order
from studies.nvf_gap_convergence import profile_li
from studies.nvf_harmonic_energy import PERIOD, THETA, THICKNESS, WAVELENGTH, polarized_powers

pytestmark = pytest.mark.research


def test_directional_order_domain_and_scalar_backwards_compatibility():
    assert parse_order('3') == 3
    assert parse_order('0:7') == (0, 7)
    for invalid in (True, -1, 1.5, (2,), (1, 2, 3), (1, -1), (1, 2.5)):
        with pytest.raises(ValueError, match='order'):
            orders_xy(invalid)
    for invalid in ('1:2:3', '-1', '0:-2', '2.5'):
        with pytest.raises(ValueError):
            parse_order(invalid)
    for got, want in zip(order_grid(2), order_grid((2, 2))):
        np.testing.assert_array_equal(got, want)
    slots = profile_li(2, 17., 350., [31.], [4., 9.])
    for got, want in zip(profile_li((2, 2), 17., 350., [31.], [4., 9.]), slots):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(polarized_powers((2, 2), slots), polarized_powers(2, slots)):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('order', [(0, 3), (3, 0), (2, 4)])
def test_uniform_directional_projection_matches_independent_slab_formula(order):
    cap = 2*max(order)
    c = np.zeros((8, 2*cap+1, 2*cap+1), complex)
    c[[0, 3], cap, cap] = 1
    c[[4, 6, 7], cap, cap] = 4
    slots = projected_matrices(c, order)
    count = (2*order[0]+1)*(2*order[1]+1)
    identity = np.eye(count)
    for got, want in zip(slots, (4*identity, 4*identity, identity/4, 0*identity, 0*identity)):
        np.testing.assert_allclose(got, want, atol=2e-13, rtol=0)
    t, r = polarized_powers(order, slots)
    for i, polarization in enumerate(('p', 's')):
        eps = np.array([1., 4., 2.25])
        kz = np.sqrt(eps-np.sin(THETA)**2)
        a, b, c = eps/kz if polarization == 'p' else kz
        phase = np.exp(2j*np.pi*THICKNESS/WAVELENGTH*kz[1])
        amplitude = 4*a*b*phase/((a+b)*(b+c)+(a-b)*(b-c)*phase**2)
        assert t[i] == pytest.approx(c/a*abs(amplitude)**2, abs=3e-13)
    np.testing.assert_allclose(t+r, 1., atol=3e-13, rtol=0)


def test_swapping_physical_axes_swaps_directional_closure_tensor():
    model = ContinuousFluxGapModel(3, rows=4)
    coefficients, _ = model.coefficients(17., dict(seams=[31.], values=[4., 9.]))
    swapped = coefficients[[3, 2, 1, 0, 6, 5, 4, 7]].transpose(0, 2, 1)
    original = projected_matrices(coefficients, (2, 3))
    actual = projected_matrices(swapped, (3, 2))
    permutation = np.arange(35).reshape(7, 5).T.ravel()
    for got, want in zip(actual, (original[1], original[0], original[2], original[4], original[3])):
        np.testing.assert_allclose(got, want[np.ix_(permutation, permutation)], atol=5e-12, rtol=0)


@pytest.mark.parametrize('order', [(2, 4), (4, 2)])
def test_directional_mixed_cell_powers_match_public_scattering_path(order):
    import wavebit
    model = ContinuousFluxGapModel(max(order), rows=4)
    coefficients, _ = model.coefficients(17., dict(seams=[31.], values=[4., 9.]))
    slots = projected_matrices(coefficients, order)
    t, r = polarized_powers(order, slots)
    for i, psi in enumerate((0., np.pi/2)):
        solver = wavebit.rcwa(backend='numpy', period=(PERIOD, PERIOD), wavelength=WAVELENGTH,
                              top=1., bot=2.25, theta=THETA, phi=0., psi=psi,
                              thickness=(THICKNESS,), truncation_order=order)
        result = solver._solve_conv(*(s[None] for s in slots))
        assert float(result.T.sum()) == pytest.approx(t[i], abs=2e-12, rel=0)
        assert float(result.R.sum()) == pytest.approx(r[i], abs=2e-12, rel=0)


def test_closed_laminate_is_independent_of_extra_transverse_orders():
    values = [4., 9.]
    profile = dict(seams=[31.], values=values)
    model = ContinuousFluxGapModel(4, rows=4)
    coefficients, _ = model.coefficients(0., profile)
    reference = polarized_powers((4, 0), profile_li((4, 0), 0., 350., [31.], values))
    for order in ((4, 0), (4, 2), (4, 4)):
        actual = polarized_powers(order, projected_matrices(coefficients, order))
        for got, want in zip(actual, reference):
            np.testing.assert_allclose(got, want, atol=2e-12, rtol=0)


@pytest.mark.filterwarnings('error::scipy.linalg.LinAlgWarning')
@pytest.mark.parametrize('body_degree,y_subdivisions,x_subdivisions,flux_degree',
                         [(1, 4, 2, 1), (4, 1, 2, 1), (3, 1, 3, 1), (4, 1, 3, 1),
                          (4, 1, 1, 2), (3, 1, 1, 3)])
def test_refined_cell_setting_retains_both_independent_contact_derivatives(
        body_degree, y_subdivisions, x_subdivisions, flux_degree):
    from studies.nvf_interface_resolution import MaterialGradedGapModel
    from studies.nvf_polynomial_body import PolynomialBodyGapModel
    from studies.nvf_polynomial_flux import PolynomialFluxGapModel
    # The coarser edge_fraction=1e-4 used in positive-gap sweeps does not meet
    # the s contact gate. Resolving that edge, at the same DOF count, is required.
    model_class = MaterialGradedGapModel if body_degree == 1 else PolynomialBodyGapModel
    extra = {} if body_degree == 1 else dict(body_degree=body_degree)
    if flux_degree > 1:
        model_class = PolynomialFluxGapModel
        extra['flux_degree'] = flux_degree
    model = model_class(7, rows=8, x_subdivisions=x_subdivisions, y_subdivisions=y_subdivisions,
                         gap_degree=6, edge_fraction=1e-5, **extra)
    candidate, reference = [], []
    for gap in (0., 1e-4, 5e-5):
        c, _ = model.coefficients(gap, 12.25)
        candidate.append(np.asarray(polarized_powers(7, projected_matrices(c, 7))[0]))
        reference.append(np.asarray(polarized_powers(7, profile_li(7, gap, 350., [], [12.25]))[0]))
    np.testing.assert_allclose(candidate[0], reference[0], atol=2e-10, rtol=0)
    def derivative(values):
        return 2*(values[2]-values[0])/5e-5-(values[1]-values[0])/1e-4
    np.testing.assert_allclose(derivative(candidate), derivative(reference), atol=1e-7, rtol=0)
