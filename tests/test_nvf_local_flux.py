"""Physical and live-derivative controls for the localized, identical cell space."""
import warnings

import numpy as np
import pytest
from scipy.linalg import LinAlgWarning

from studies.nvf_derivative_reference import central_reference
from studies.nvf_gap_convergence import profile_li
from studies.nvf_harmonic_energy import polarized_powers
from studies.nvf_polynomial_body import PolynomialBodyGapModel
from studies.nvf_polynomial_flux import PolynomialFluxGapModel
from studies.nvf_posed_gap import PosedGapModel

pytestmark = pytest.mark.research


@pytest.mark.parametrize('gap', [0., 17., 400.])
def test_local_basis_preserves_the_global_space_and_redundant_material_partitions(gap):
    settings = dict(rows=4, body_degree=2, edge_fraction=1e-4)
    profile = dict(seams=[-73., 31.], values=[4., 9., 6.25])
    split = dict(seams=[-100., -73., 0., 31., 100.], values=[4., 4., 9., 9., 6.25, 6.25])
    global_model = PolynomialBodyGapModel(2, **settings)
    local_model = PolynomialBodyGapModel(2, flux_basis='local', **settings)
    original, _ = global_model.coefficients(gap, profile)
    localized, _ = local_model.coefficients(gap, profile)
    np.testing.assert_allclose(localized, original, atol=3e-11, rtol=0)
    np.testing.assert_array_equal(localized, local_model.coefficients(gap, split)[0])


def sharp_model(subdivisions=3):
    return PosedGapModel(2, rows=4, body_degree=1, x_subdivisions=subdivisions, gap_degree=6,
                          edge_fraction=1e-5, grading_radius=.0001, flux_basis='local')


@pytest.mark.parametrize('profile,subdivisions,angles', [
    (dict(seams=[31.], values=[4., 9.]), 3, (0., .37)),
    (dict(seams=[-73., 31.], values=[4., 9., 6.25]), 6, (0., 1.1)),
])
def test_sharp_grading_jit_mixed_material_contact_passes_cartesian_gate_at_two_azimuths(profile, subdivisions, angles):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = sharp_model(subdivisions)
    def objective(gap, azimuth):
        return model.transmission(gap, profile, xp=jnp, rotation=.31, azimuth=azimuth+.31,
                                   center=(17., -21.), polarization='both')['T']
    evaluate = jax.jit(lambda gap, angle: jax.jvp(lambda g: objective(g, angle), (gap,), (jnp.ones_like(gap),)))
    for azimuth in angles:
        value, derivative = evaluate(0., azimuth)
        control = np.array([polarized_powers(2, profile_li(2, g, 350., **profile), azimuth=azimuth)[0]
                            for g in (0., 1e-4, 5e-5)])
        expected = 2*(control[2]-control[0])/5e-5-(control[1]-control[0])/1e-4
        np.testing.assert_allclose(value, control[0], atol=2e-10, rtol=0)
        np.testing.assert_allclose(derivative, expected, atol=1e-7, rtol=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            finite = [model.transmission(g, profile, azimuth=azimuth, polarization='both')['T']
                      for g in (0., 1e-4, 5e-5)]
        own = 2*(finite[2]-finite[0])/5e-5-(finite[1]-finite[0])/1e-4
        np.testing.assert_allclose(derivative, own, atol=1e-7, rtol=0)
    jax.clear_caches()


def test_sharp_mesh_positive_gap_jit_seam_and_material_derivatives_match_modal_secants():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = sharp_model()
    def objective(p):
        return model.transmission(p[0], dict(seams=p[1:2], values=p[2:]), xp=jnp,
                                   azimuth=.37, polarization='both')['T']
    def control(p):
        slots, _ = model.cell.matrices(p[0], dict(seams=p[1:2], values=p[2:]))
        return polarized_powers(2, slots, azimuth=.37)[0]
    evaluate, differentiate = jax.jit(objective), jax.jit(jax.jacfwd(objective))
    for p in map(np.array, ([17., 31., 4., 9.], [17., 0., 4., 9.], [17., 31., 6.25, 6.25])):
        value, derivative = evaluate(jnp.asarray(p)), differentiate(jnp.asarray(p))
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            center = control(p)
            np.testing.assert_allclose(value, center, atol=2e-10, rtol=0)
            for i, step in enumerate((1e-3, 1e-4, 1e-4, 1e-4)):
                # The sharp mesh's modal values have ~1e-13 roundoff. Raw
                # 5e-5 seam secants amplify it past the derivative criterion.
                # Select a bounded, resolved reference window independently
                # of AD; failure to resolve that window is still an error.
                resolved = central_reference(control, p, np.eye(4)[i], 10*step)
                for reference in resolved['references']:
                    np.testing.assert_allclose(derivative[:, i], reference, atol=2e-9, rtol=3e-6)
                if i == 1 and p[1] == 0.:
                    for scale in (1., .5):
                        delta = np.eye(4)[i]*step*scale
                        plus, minus = control(p+delta), control(p-delta)
                        np.testing.assert_allclose(derivative[:, i], (plus-center)/(step*scale), atol=1e-7, rtol=0)
                        np.testing.assert_allclose(derivative[:, i], (center-minus)/(step*scale), atol=1e-7, rtol=0)
    jax.clear_caches()


def test_unsupported_basis_and_sharper_global_mesh_are_refused():
    for settings in (dict(flux_basis='unknown'), dict(grading_radius=.001),
                     dict(flux_basis='local', grading_radius=.00001)):
        with pytest.raises(ValueError, match='flux|grading'):
            PolynomialBodyGapModel(1, **settings)
    with pytest.raises(ValueError, match='flux_degree'):
        PolynomialFluxGapModel(1, flux_basis='local', flux_degree=2)
    original = PolynomialBodyGapModel(1, flux_basis='local')
    control = PolynomialFluxGapModel(1, flux_basis='local', flux_degree=1)
    profile = dict(seams=[31.], values=[4., 9.])
    np.testing.assert_array_equal(original.coefficients(17., profile)[0], control.coefficients(17., profile)[0])
