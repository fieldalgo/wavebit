"""Independent complex-transfer and physical controls for parity doubling."""
import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from studies.rcwa.nvf_fourier_orders.nvf_fourier_orders import orders_xy
from studies.rcwa.nvf_gap_convergence.nvf_gap_convergence import profile_li, projected_matrices
from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import THETA, WAVELENGTH, polarized_powers
from studies.rcwa.nvf_mirror_cartesian.nvf_mirror_cartesian import cartesian_powers
from studies.rcwa.nvf_mirror_convergence.nvf_mirror_convergence import mirror_powers, mirror_sector
from studies.rcwa.nvf_mirror_doubling.nvf_mirror_doubling import mirror_doubling_powers, sector_factors
from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
from studies.rcwa.nvf_scattering_doubling.nvf_scattering_doubling import canonical_factors, doubling_powers, maxwell_factors, symmetric_scattering
from tests.rcwa.test_nvf_mirror_convergence import dense_parity
from tests.rcwa.test_nvf_scattering_doubling import uniform_slots

pytestmark = pytest.mark.research


def test_z_symmetric_doubling_matches_all_complex_blocks_of_independent_matrix_exponential():
    random = np.random.default_rng(42)
    matrix = random.normal(size=(18, 18))+1j*random.normal(size=(18, 18))
    epsilon = matrix.conj().T@matrix+np.eye(18)
    z = np.eye(9)/4
    slots = epsilon[:9, :9], epsilon[9:, 9:], z, -epsilon[:9, 9:], -epsilon[9:, :9]
    factors = canonical_factors(*maxwell_factors(1, slots), 9, 9)
    left, right = factors[:2]
    identity, zero = np.eye(18), np.zeros((18, 18))
    thickness = 27.
    transfer = expm(-2*np.pi*thickness/WAVELENGTH*np.block([[zero, left], [right, zero]]))
    waves = np.block([[identity, identity], [identity, -identity]])/np.sqrt(2)
    matrix = waves.T@transfer@waves
    a, b, c, d = matrix[:18, :18], matrix[:18, 18:], matrix[18:, :18], matrix[18:, 18:]
    reflection = -np.linalg.solve(d, c)
    transmission = np.linalg.solve(d, identity)
    expected = reflection, transmission, a+b@reflection, b@transmission
    for doublings in (2, 5, 8):
        actual, _ = symmetric_scattering(*factors, thickness=thickness, doublings=doublings)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=7e-13, rtol=0)


@pytest.mark.parametrize('order', [0, (0, 3), (3, 0), (2, 3)])
def test_parity_factors_and_both_products_match_full_orthonormal_projection(order):
    c, _ = PolynomialBodyGapModel(max(orders_xy(order)), rows=4).coefficients(17., dict(seams=[31.], values=[4., 9.]))
    full = maxwell_factors(order, projected_matrices(c, order))
    for pol, parity in (('p', 1), ('s', -1)):
        a, b = dense_parity(order, parity), dense_parity(order, -parity)
        electric, magnetic = block_diag(a, b), block_diag(b, a)
        bases = (electric, magnetic), (magnetic, electric), (electric, electric), (magnetic, magnetic)
        actual = sector_factors(mirror_sector(c, order, pol))
        for got, matrix, (row, column) in zip(actual, full, bases):
            np.testing.assert_allclose(got, row.T@matrix@column, atol=3e-11, rtol=2e-13)
        np.testing.assert_allclose(actual[2], actual[0]@actual[1], atol=3e-11, rtol=2e-13)
        np.testing.assert_allclose(actual[3], actual[1]@actual[0], atol=3e-11, rtol=2e-13)


@pytest.mark.parametrize('order,gap', [((0, 3), 17.), ((3, 0), 400.), ((2, 3), 0.), ((3, 2), 17.)])
def test_mirror_doubling_matches_unreduced_doubling_and_modal_powers(order, gap):
    c, _ = PolynomialBodyGapModel(max(orders_xy(order)), rows=4).coefficients(gap, dict(seams=[31.], values=[4., 9.]))
    actual = mirror_powers(c, order, propagation='doubling')
    modal = mirror_powers(c, order)
    full = doubling_powers(order, projected_matrices(c, order))
    for i, pol in enumerate(('p', 's')):
        for j, quantity in enumerate(('T', 'R')):
            assert actual[pol+'_'+quantity] == pytest.approx(modal[pol+'_'+quantity], abs=2e-11, rel=0)
            assert actual[pol+'_'+quantity] == pytest.approx(full[j][i], abs=2e-11, rel=0)
    control = cartesian_powers(order, gap=gap, propagation='doubling')
    expected = polarized_powers(order, profile_li(order, gap, 350., [31.], [4., 9.]))
    for i, pol in enumerate(('p', 's')):
        for j, quantity in enumerate(('T', 'R')):
            assert control[pol+'_'+quantity] == pytest.approx(expected[j][i], abs=2e-11, rel=0)


def test_layer_graze_is_admitted_by_explicit_doubling_without_changing_the_modal_default():
    c = np.zeros((8, 5, 5), complex)
    epsilon = np.sin(THETA)**2
    c[[0, 3], 2, 2], c[[4, 6, 7], 2, 2] = 1., epsilon
    expected = doubling_powers(1, uniform_slots(1, epsilon))
    with pytest.raises(FloatingPointError, match='near-grazing'):
        mirror_powers(c, 1)
    actual = mirror_powers(c, 1, propagation='doubling')
    for i, pol in enumerate(('p', 's')):
        assert actual[pol+'_T'] == pytest.approx(expected[0][i], abs=3e-12, rel=0)
    with pytest.raises(ValueError, match='propagation'):
        mirror_powers(c, 1, propagation='unknown')
    c[[4, 6, 7], 2, 2] = 12.25
    with pytest.raises(FloatingPointError, match='Taylor tail'):
        mirror_powers(c, 1, propagation='doubling', doublings=0, terms=3)
    c[0, 0, 2] += .01
    c[0, 4, 2] += .01
    c[0, 0, 3] += .01
    c[0, 4, 1] += .01
    with pytest.raises(ValueError, match='y-reflection'):
        mirror_powers(c, 1, propagation='doubling')


@pytest.mark.filterwarnings('error::scipy.linalg.LinAlgWarning')
def test_mirror_doubling_retains_both_contact_derivatives_at_the_final_cell_resolution():
    model = PolynomialBodyGapModel(7, rows=8, body_degree=4, gap_degree=6, x_subdivisions=3, edge_fraction=1e-5)
    candidate, control = [], []
    for gap in (0., 1e-4, 5e-5):
        c, _ = model.coefficients(gap, 12.25)
        actual = mirror_powers(c, 7, propagation='doubling')
        candidate.append([actual['p_T'], actual['s_T']])
        control.append(polarized_powers(7, profile_li(7, gap, 350., [], [12.25]))[0])
    candidate, control = np.array(candidate), np.array(control)
    np.testing.assert_allclose(candidate[0], control[0], atol=2e-10, rtol=0)
    def derivative(values):
        return 2*(values[2]-values[0])/5e-5-(values[1]-values[0])/1e-4
    np.testing.assert_allclose(derivative(candidate), derivative(control), atol=1e-7, rtol=0)


def test_live_mirror_closure_and_propagation_jit_derivatives_match_independent_modal_secants():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PolynomialBodyGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    def profile(p):
        return dict(seams=p[1:2], values=p[2:])
    def objective(p):
        c, _ = model.coefficients(p[0], profile(p), xp=jnp)
        return mirror_doubling_powers(c, 2, jnp)[0]
    def control(p):
        return polarized_powers(2, model.matrices(p[0], profile(p))[0])[0]
    value, gradient = jax.jit(objective), jax.jit(jax.jacrev(objective))
    for p in (np.array([17., 0., 4., 9.]), np.array([17., 0., 6.25, 6.25]), np.array([17., 0., 1., 1.])):
        derivative = gradient(jnp.asarray(p))
        np.testing.assert_allclose(value(jnp.asarray(p)), control(p), atol=2e-10, rtol=0)
        for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4)):
            for scale in (1., .5):
                delta = np.eye(4)[i]*step*scale
                secant = (control(p+delta)-control(p-delta))/(2*step*scale)
                np.testing.assert_allclose(derivative[:, i], secant, atol=2e-9, rtol=3e-6)
    p = jnp.array([0., 0., 12.25, 12.25])
    reference = np.array([polarized_powers(2, profile_li(2, gap, 350., [], [12.25]))[0]
                          for gap in (0., 1e-4, 5e-5)])
    expected = 2*(reference[2]-reference[0])/5e-5-(reference[1]-reference[0])/1e-4
    np.testing.assert_allclose(value(p), reference[0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(gradient(p)[:, 0], expected, atol=1e-7, rtol=0)
    jax.clear_caches()


def test_jit_retains_exact_layer_graze_and_refuses_asymmetric_or_nonreal_coefficients():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    def uniform(epsilon):
        c = jnp.zeros((8, 5, 5), dtype=jnp.complex128)
        c = c.at[jnp.array([0, 3]), 2, 2].set(1.)
        return c.at[jnp.array([4, 6, 7]), 2, 2].set(epsilon)
    objective = lambda epsilon: mirror_doubling_powers(uniform(epsilon), 1, jnp)[0]
    expected = lambda epsilon: doubling_powers(1, uniform_slots(1, epsilon, jnp), jnp)[0]
    epsilon = np.sin(THETA)**2
    np.testing.assert_allclose(jax.jit(objective)(epsilon), expected(epsilon), atol=3e-12, rtol=0)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(objective))(epsilon), jax.jacfwd(expected)(epsilon), atol=3e-10, rtol=0)
    c, _ = PolynomialBodyGapModel(1, rows=4).coefficients(17., dict(seams=[31.], values=[4., 9.]))
    function = jax.jit(lambda c: mirror_doubling_powers(c, 1, jnp)[0])
    assert np.isfinite(function(jnp.asarray(c))).all()
    shifted = c*np.exp(-2j*np.pi*np.arange(-2, 3)*.03)[None, :, None]
    assert np.isnan(function(jnp.asarray(shifted))).all()
    c[0, 2, 2] += .01j
    assert np.isnan(function(jnp.asarray(c))).all()
    jax.clear_caches()
