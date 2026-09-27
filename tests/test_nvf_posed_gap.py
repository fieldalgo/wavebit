"""Pose, angular derivatives and complex phases for the same conforming NVF."""
import numpy as np
import pytest

from studies.nvf_fourier_orders import order_grid, orders_xy
from studies.nvf_gap_convergence import profile_li, projected_matrices
from studies.nvf_harmonic_energy import PERIOD, WAVELENGTH, THICKNESS, polarized_powers
from studies.nvf_posed_gap import PosedGapModel, translated_coefficients
from studies.nvf_scattering_doubling import doubling_powers, doubling_response, incident_orders
from test_nvf_scattering_doubling import uniform_slots

pytestmark = pytest.mark.research


def slab_power(epsilon, theta, polarization):
    eps = np.array([1., epsilon, 2.25])
    kz = np.sqrt((eps-np.sin(theta)**2).astype(complex))
    top, body, bottom = eps/kz if polarization == 'p' else kz
    phase = np.exp(2j*np.pi*THICKNESS/WAVELENGTH*kz[1])
    amplitude = 4*top*body*phase/((top+body)*(body+bottom)+(top-body)*(body-bottom)*phase**2)
    return float((bottom/top).real*abs(amplitude)**2)


@pytest.mark.parametrize('theta,azimuth', [(0., .71), (.27, .43), (-.27, .43), (.4, -.6)])
def test_conical_uniform_slab_matches_independent_airy_formula(theta, azimuth):
    t, r, _ = doubling_powers((2, 3), uniform_slots((2, 3), 4.), theta=theta, azimuth=azimuth)
    np.testing.assert_allclose(t, [slab_power(4., theta, p) for p in ('p', 's')], atol=3e-12, rtol=0)
    np.testing.assert_allclose(t+r, 1., atol=3e-12, rtol=0)


@pytest.mark.parametrize('order,theta,azimuth', [(0, 0., .71), ((2, 3), -.2, .53),
                                              ((3, 2), .31, -.8), (3, .43, 1.27)])
def test_nonzero_azimuth_nvf_matches_independent_modal_propagation(order, theta, azimuth):
    model = PosedGapModel(max(orders_xy(order)), rows=4)
    c, _ = model.cell.coefficients(17., dict(seams=[31.], values=[4., 9.]))
    slots = projected_matrices(c, order)
    actual = doubling_powers(order, slots, theta=theta, azimuth=azimuth)
    expected = polarized_powers(order, slots, theta=theta, azimuth=azimuth)
    for got, want in zip(actual[:2], expected):
        np.testing.assert_allclose(got, want, atol=2e-10, rtol=0)


def test_quarter_turn_rotates_tensor_components_directional_orders_and_all_complex_amplitudes():
    order, rotated_order = (2, 3), (3, 2)
    model = PosedGapModel(3, rows=4)
    c, _ = model.cell.coefficients(17., dict(seams=[31.], values=[4., 9.]))
    turned = np.rot90(c, k=-1, axes=(-2, -1))[[3, 2, 1, 0, 6, 5, 4, 7]]
    turned *= np.array([1, -1, -1, 1, 1, -1, 1, 1])[:, None, None]
    original = doubling_response(order, projected_matrices(c, order), theta=.23, azimuth=.34)
    rotated = doubling_response(rotated_order, projected_matrices(turned, rotated_order), theta=.23, azimuth=.34+np.pi/2)
    for name in ('T', 'R'):
        np.testing.assert_allclose(rotated[name], original[name], atol=3e-11, rtol=0)
    indices = np.arange(35).reshape(7, 5)
    permutation = np.rot90(indices, k=-1).ravel()
    permutation = np.r_[permutation, permutation+35]
    for name in ('reflected', 'transmitted'):
        np.testing.assert_allclose(rotated[name], original[name][permutation], atol=5e-11, rtol=0)


def test_laboratory_translation_has_exact_fourier_phase_and_jit_amplitude_derivatives():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PosedGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    profile, rotation, center = dict(seams=[31.], values=[4., 9.]), .31, np.array([17., -21.])
    original = model.response(17., profile, rotation=rotation, azimuth=.47)
    shifted = model.response(17., profile, rotation=rotation, azimuth=.47, center=center)
    n, m = order_grid(2)
    gx = 2*np.pi/PERIOD*(np.cos(rotation)*m.ravel()-np.sin(rotation)*n.ravel())
    gy = 2*np.pi/PERIOD*(np.sin(rotation)*m.ravel()+np.cos(rotation)*n.ravel())
    wave = np.tile(np.column_stack([gx, gy]), (2, 1))
    phase = np.exp(-1j*wave@center)
    for name in ('T', 'R'):
        np.testing.assert_allclose(shifted[name], original[name], atol=2e-11, rtol=0)
    for name in ('reflected', 'transmitted'):
        np.testing.assert_allclose(shifted[name], phase[:, None]*original[name], atol=2e-11, rtol=0)
    # The cell is independent of translation, so its exact coefficient field
    # may be reused while differentiating this physical phase control.
    c = jnp.asarray(model.cell.coefficients(17., profile)[0])
    def amplitudes(center):
        field = translated_coefficients(c, center, rotation, jnp)
        slots, _ = model.cell._matrices_from_coefficients(field, {}, jnp)
        result = doubling_response(2, slots, jnp, azimuth=.47-rotation)
        return jnp.stack([result['reflected'], result['transmitted']])
    derivative = jax.jit(jax.jacfwd(amplitudes))(jnp.asarray(center))
    expected = -1j*np.asarray(amplitudes(jnp.asarray(center)))[..., None]*wave[None, :, None, :]
    np.testing.assert_allclose(derivative, expected, atol=3e-11, rtol=0)
    jax.clear_caches()


def test_normal_incidence_and_layer_graze_have_finite_correct_angular_derivatives():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    objective = lambda angles, epsilon: doubling_powers(1, uniform_slots(1, epsilon, jnp), jnp,
                                                        theta=angles[0], azimuth=angles[1])[0]
    gradient = jax.jit(jax.jacfwd(objective))
    for theta, epsilon in ((0., 4.), (.27, 4.), (.2, np.sin(.2)**2)):
        actual = np.asarray(gradient(jnp.array([theta, .63]), epsilon))
        def secant(step):
            return np.array([(slab_power(epsilon, theta+step, pol)-slab_power(epsilon, theta-step, pol))/(2*step)
                              for pol in ('p', 's')])
        expected = (4*secant(5e-6)-secant(1e-5))/3
        np.testing.assert_allclose(actual[:, 0], expected, atol=3e-8, rtol=2e-6)
        np.testing.assert_allclose(actual[:, 1], 0., atol=3e-11, rtol=0)
    jax.clear_caches()


def test_full_live_pose_geometry_and_material_jit_derivatives_match_two_modal_secants():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PosedGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    def profile(p):
        return dict(seams=p[7:8], values=p[8:])
    def objective(p):
        return model.transmission(p[0], profile(p), p[1], jnp, center=p[2:4], rotation=p[4], theta=p[5],
                                   azimuth=p[6], polarization='both')['T']
    def control(p):
        slots, _ = model.cell.matrices(p[0], profile(p), p[1])
        return polarized_powers(2, slots, theta=p[5], azimuth=p[6]-p[4])[0]
    p = np.array([17., 350., 17., -21., .31, .2, .47, 31., 4., 9.])
    value = jax.jit(objective)(jnp.asarray(p))
    derivative = np.asarray(jax.jit(jax.jacrev(objective))(jnp.asarray(p)))
    np.testing.assert_allclose(value, control(p), atol=2e-10, rtol=0)
    for i, step in enumerate((1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)):
        for scale in (1., .5):
            delta = np.eye(len(p))[i]*step*scale
            expected = (control(p+delta)-control(p-delta))/(2*step*scale)
            np.testing.assert_allclose(derivative[:, i], expected, atol=2e-9, rtol=3e-6)
    np.testing.assert_allclose(derivative[:, 2:4], 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(derivative[:, 4]+derivative[:, 6], 0., atol=2e-12, rtol=0)
    jax.clear_caches()


def test_oblique_contact_jit_derivative_retains_independent_cartesian_gate():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = PosedGapModel(2, rows=4, body_degree=4, gap_degree=6, edge_fraction=1e-5)
    objective = lambda gap: model.transmission(gap, 12.25, xp=jnp, rotation=.31, azimuth=.68,
                                               center=(17., -21.), polarization='both')['T']
    value = jax.jit(objective)(0.)
    derivative = jax.jit(jax.jacfwd(objective))(0.)
    control = np.array([polarized_powers(2, profile_li(2, gap, 350., [], [12.25]), azimuth=.37)[0]
                        for gap in (0., 1e-4, 5e-5)])
    expected = 2*(control[2]-control[0])/5e-5-(control[1]-control[0])/1e-4
    np.testing.assert_allclose(value, control[0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(derivative, expected, atol=1e-7, rtol=0)
    jax.clear_caches()


def test_invalid_pose_and_exact_exterior_graze_are_refused_in_numpy_and_jit():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    for theta, azimuth in ((np.pi/2, 0.), (np.inf, 0.), (.2, np.nan), (1j, 0.), ([.2], 0.)):
        with pytest.raises(ValueError, match='incidence angles'):
            incident_orders(1, theta, azimuth)
    for center, rotation in (((1.,), 0.), ((np.nan, 0.), 0.), ((0., 0.), np.inf), ((1j, 0.), 0.)):
        with pytest.raises(ValueError, match='center'):
            translated_coefficients(np.ones((8, 5, 5)), center, rotation)
    cutoff = np.arcsin(1.5-WAVELENGTH/PERIOD)
    with pytest.raises(FloatingPointError, match='nongrazing exterior'):
        doubling_powers(1, uniform_slots(1, 4.), theta=cutoff)
    function = jax.jit(lambda theta: doubling_powers(1, uniform_slots(1, 4., jnp), jnp, theta=theta)[0])
    assert np.isnan(function(cutoff)).all()
    assert np.isnan(function(np.pi/2)).all()
    assert np.isfinite(function(0.)).all()
    jax.clear_caches()
