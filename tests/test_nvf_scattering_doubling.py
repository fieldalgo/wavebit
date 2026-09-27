"""Physical and derivative controls for eigenvector-free slab propagation."""
import numpy as np
import pytest

from studies.nvf_fourier_orders import orders_xy
from studies.nvf_gap_convergence import profile_li
from studies.nvf_harmonic_energy import THETA, THICKNESS, WAVELENGTH, polarized_powers
from studies.nvf_scattering_doubling import ScatteringDoublingGapModel, doubling_powers, maxwell_factors

pytestmark = pytest.mark.research


def uniform_slots(order, epsilon, xp=np):
    x, y = orders_xy(order)
    identity = xp.eye((2*x+1)*(2*y+1), dtype=xp.complex128)
    return epsilon*identity, epsilon*identity, identity/epsilon, 0*identity, 0*identity


def slab_power(epsilon, thickness, polarization):
    eps = np.array([1., epsilon, 2.25])
    kz = np.sqrt((eps-np.sin(THETA)**2).astype(complex))
    top, body, bottom = eps/kz if polarization == 'p' else kz
    phase = np.exp(2j*np.pi*thickness/WAVELENGTH*kz[1])
    amplitude = 4*top*body*phase/((top+body)*(body+bottom)+(top-body)*(body-bottom)*phase**2)
    return float((bottom/top).real*abs(amplitude)**2)


def test_expanded_products_match_both_first_order_factors_with_complex_cross_blocks():
    random = np.random.default_rng(87)
    matrix = random.normal(size=(18, 18))+1j*random.normal(size=(18, 18))
    epsilon = matrix.conj().T@matrix+np.eye(18)
    other = random.normal(size=(9, 9))+1j*random.normal(size=(9, 9))
    z = np.linalg.inv(other.conj().T@other+np.eye(9))
    slots = epsilon[:9, :9], epsilon[9:, 9:], z, -epsilon[:9, 9:], -epsilon[9:, :9]
    left, right, lr, rl = maxwell_factors(1, slots)
    np.testing.assert_allclose(lr, left@right, atol=6e-14, rtol=2e-15)
    np.testing.assert_allclose(rl, right@left, atol=6e-14, rtol=2e-15)


@pytest.mark.parametrize('order,thickness,doublings', [(0, 0., 6), (2, 300., 6),
                                                     ((2, 3), 300., 6), (2, 30000., 12)])
def test_uniform_layer_matches_independent_slab_formula_without_eigenvectors(order, thickness, doublings):
    t, r, _ = doubling_powers(order, uniform_slots(order, 4.), thickness=thickness, doublings=doublings)
    for i, pol in enumerate(('p', 's')):
        assert t[i] == pytest.approx(slab_power(4., thickness, pol), abs=2e-10, rel=0)
    np.testing.assert_allclose(t+r, 1., atol=2e-10, rtol=0)


@pytest.mark.parametrize('gap', [0., 17., 400.])
def test_mixed_layer_matches_modal_scattering_and_is_stable_under_slice_refinement(gap):
    slots = profile_li(3, gap, 350., [-31., 73.], [4., 9., 6.25])
    expected = polarized_powers(3, slots)
    for doublings in (3, 6, 9):
        actual = doubling_powers(3, slots, doublings=doublings)
        for got, want in zip(actual[:2], expected):
            np.testing.assert_allclose(got, want, atol=3e-11, rtol=0)


def test_exact_layer_graze_matches_independent_jordan_transfer_limit():
    epsilon = np.sin(THETA)**2
    t, _, _ = doubling_powers(1, uniform_slots(1, epsilon))
    kz = np.sqrt(np.array([1., 2.25])-np.sin(THETA)**2)
    for i, pol in enumerate(('p', 's')):
        top, bottom = np.array([1., 2.25])/kz if pol == 'p' else kz
        transfer = (np.array([[1., 0.], [2j*np.pi*THICKNESS/WAVELENGTH*epsilon, 1.]]) if pol == 'p'
                    else np.array([[1., 2j*np.pi*THICKNESS/WAVELENGTH], [0., 1.]]))
        matrix = np.column_stack([transfer@np.array([1., -top]), -np.array([1., bottom])])
        _, amplitude = np.linalg.solve(matrix, -transfer@np.array([1., top]))
        assert t[i] == pytest.approx(bottom/top*abs(amplitude)**2, abs=3e-12, rel=0)


def test_unresolved_taylor_slice_is_refused_and_invalid_settings_are_rejected():
    slots = uniform_slots(3, 12.25)
    with pytest.raises(FloatingPointError, match='Taylor tail'):
        doubling_powers(3, slots, doublings=0, terms=3)
    for setting in (dict(doublings=-1), dict(terms=2), dict(thickness=-1.)):
        with pytest.raises(ValueError):
            doubling_powers(3, slots, **setting)


def test_jit_removes_the_recorded_material_gradient_bias_without_native_eigen_derivatives():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from studies.nvf_harmonic_energy_jax import HarmonicEnergyModel
    from studies.nvf_harmonic_energy import matrices
    model = HarmonicEnergyModel(2, 7)
    def shapes(p):
        return [dict(lx=p[0], ly=p[1], angle=p[2], epsilon=p[3], cx=17., cy=-21.)]
    def objective(p):
        return doubling_powers(2, model.matrices(shapes(p))[0], jnp)[0]
    p = np.array([440., 270., .31, 6.25])
    actual = jax.jit(objective)(jnp.asarray(p))
    gradient = jax.jit(jax.jacrev(objective))(jnp.asarray(p))
    np.testing.assert_allclose(actual, polarized_powers(2, matrices(2, 7, shapes(p))[0])[0], atol=2e-11, rtol=0)
    # Keep the original physical tolerance and both secant step sizes.
    for i, step in enumerate((1e-3, 1e-3, 1e-6, 1e-5)):
        for scale in (1., .5):
            delta = np.eye(4)[i]*step*scale
            plus = polarized_powers(2, matrices(2, 7, shapes(p+delta))[0])[0]
            minus = polarized_powers(2, matrices(2, 7, shapes(p-delta))[0])[0]
            np.testing.assert_allclose(gradient[:, i], (plus-minus)/(2*step*scale), atol=2e-9, rtol=3e-6)
    jax.clear_caches()


@pytest.mark.parametrize('epsilon', [1., 4., np.sin(THETA)**2])
def test_uniform_and_exact_graze_jit_material_derivatives_remain_finite(epsilon):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    objective = lambda e: doubling_powers(1, uniform_slots(1, e, jnp), jnp)[0]
    derivative = np.asarray(jax.jit(jax.jacfwd(objective))(epsilon))
    assert np.isfinite(derivative).all()
    # At exact graze the 1e-5 Airy secant has a visible O(h^2) error.
    # Extrapolate two independently evaluated step pairs; do not relax the gate.
    def secant(step):
        return np.array([(slab_power(epsilon+step, THICKNESS, pol)
                          -slab_power(epsilon-step, THICKNESS, pol))/(2*step) for pol in ('p', 's')])
    expected = (4*secant(5e-6)-secant(1e-5))/3
    refined = (4*secant(2.5e-6)-secant(5e-6))/3
    np.testing.assert_allclose(expected, refined, atol=3e-8, rtol=2e-6)
    np.testing.assert_allclose(derivative, expected, atol=3e-8, rtol=2e-6)
    jax.clear_caches()


def test_current_nvf_jit_gap_seam_and_material_derivatives_including_zero_contrast():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from studies.nvf_polynomial_body import PolynomialBodyGapModel
    settings = dict(rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    model, modal = ScatteringDoublingGapModel(2, **settings), PolynomialBodyGapModel(2, **settings)
    def objective(p):
        return model.transmission(p[0], dict(seams=p[1:2], values=p[2:]), xp=jnp, polarization='both')['T']
    value, gradient = jax.jit(objective), jax.jit(jax.jacrev(objective))
    def control(p):
        return polarized_powers(2, modal.matrices(p[0], dict(seams=p[1:2], values=p[2:]))[0])[0]
    for p in (np.array([17., 0., 4., 9.]), np.array([17., 0., 6.25, 6.25]), np.array([17., 0., 1., 1.])):
        derivative = np.asarray(gradient(jnp.asarray(p)))
        np.testing.assert_allclose(value(jnp.asarray(p)), control(p), atol=2e-10, rtol=0)
        for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4)):
            for scale in (1., .5):
                delta = np.eye(4)[i]*step*scale
                secant = (control(p+delta)-control(p-delta))/(2*step*scale)
                np.testing.assert_allclose(derivative[:, i], secant, atol=2e-9, rtol=3e-6)
        if p[2] == p[3]:
            np.testing.assert_allclose(derivative[:, 1], 0., atol=2e-12, rtol=0)
        if p[2] == 1.:
            np.testing.assert_allclose(derivative[:, :2], 0., atol=2e-12, rtol=0)
    jax.clear_caches()


def test_current_nvf_contact_jit_derivative_matches_independent_factorization():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ScatteringDoublingGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    objective = lambda gap: model.transmission(gap, 12.25, xp=jnp, polarization='both')['T']
    actual = jax.jit(objective)(0.)
    derivative = jax.jit(jax.jacfwd(objective))(0.)
    control = np.array([polarized_powers(2, profile_li(2, gap, 350., [], [12.25]))[0]
                        for gap in (0., 1e-4, 5e-5)])
    expected = 2*(control[2]-control[0])/5e-5-(control[1]-control[0])/1e-4
    np.testing.assert_allclose(actual, control[0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(derivative, expected, atol=1e-7, rtol=0)
    jax.clear_caches()


def test_translated_shape_adapter_preserves_partition_equality_and_translation_derivatives():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = ScatteringDoublingGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    def shapes(translation, split):
        base = dict(ly=683., epsilon=6.25, cy=translation[1])
        if split:
            return [dict(base, lx=175., cx=translation[0]+cx) for cx in (-87.5, 87.5)]
        return [dict(base, lx=350., cx=translation[0])]
    translation = np.array([17., -21.])
    expected = model.transmission(17., 6.25, polarization='both')['T']
    for split in (False, True):
        actual = model.shape_transmission(shapes(translation, split), polarization='both')['T']
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    objective = lambda p: model.shape_transmission(shapes(p, True), xp=jnp, polarization='both')['T']
    np.testing.assert_allclose(jax.jit(jax.jacrev(objective))(jnp.asarray(translation)), 0., atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match='polarization'):
        model.transmission(17., polarization='bad')
    jax.clear_caches()


def test_jit_marks_unresolved_slices_invalid_instead_of_returning_plausible_powers():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    def objective(thickness):
        return doubling_powers(1, uniform_slots(1, 12.25, jnp), jnp, doublings=0, terms=3,
                               thickness=thickness)[0]
    function = jax.jit(objective)
    assert np.isnan(function(300.)).all()
    assert np.isnan(function(-1.)).all()
    assert np.isfinite(function(1.)).all()
    jax.clear_caches()
