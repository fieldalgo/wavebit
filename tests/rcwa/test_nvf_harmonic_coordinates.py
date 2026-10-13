"""Independent controls for the live harmonic coordinate and its affine lift."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from studies.rcwa.nvf_harmonic_coordinates.nvf_harmonic_coordinates import HarmonicCoordinateGapModel, coordinate_integrals
from studies.rcwa.nvf_harmonic_energy.nvf_harmonic_energy import powers
from studies.rcwa.nvf_continuous_flux.nvf_continuous_flux import ContinuousFluxGapModel
from studies.rcwa.nvf_interface_resolution.nvf_interface_resolution import MaterialGradedGapModel
from studies.rcwa.nvf_polynomial_body.nvf_polynomial_body import PolynomialBodyGapModel
from studies.rcwa.nvf_polynomial_flux.nvf_polynomial_flux import PolynomialFluxGapModel

pytestmark = pytest.mark.research

MODEL_CLASSES = [HarmonicCoordinateGapModel, ContinuousFluxGapModel, MaterialGradedGapModel,
                 PolynomialBodyGapModel, PolynomialFluxGapModel]


def _closed_li(order, seams, values, width=350.):
    from wavebit.rcwa._engine.factorization import fourier2d_integrated

    edges = np.r_[-width/2, seams, width/2]/700+.5
    x = np.r_[edges, 1.][:, None]
    cell = np.r_[1., values, 1.][None].astype(complex)
    y = np.ones((1, 1))
    ex = fourier2d_integrated(np, cell, x, y, order, order, inverse_rule='u')
    ey = fourier2d_integrated(np, cell, x, y, order, order, inverse_rule='v')
    ez = np.linalg.inv(fourier2d_integrated(np, cell, x, y, order, order, inverse_rule=None))
    return ex, ey, ez, np.zeros_like(ex), np.zeros_like(ex)


def test_harmonic_basis_and_lift_integrals_match_independent_piece_quadrature():
    nodes = np.array([-.5, -.25, 0., .25])
    lengths = np.full(4, .25)
    seams, values = np.array([-.11, .13]), np.array([4., 9., 6.25])
    inside, modes = np.array([False, True, True, False]), np.arange(-7, 8)
    body, gap, _, _, _ = coordinate_integrals(nodes, lengths, seams, values, inside, modes)
    fields = ('mass', 'derivative', 'stiffness', 'lift', 'lift_squared', 'lift_derivative')
    expected = [{key: np.zeros_like(pack[key]) for key in fields} for pack in (body, gap)]
    q, weights = leggauss(48)
    for i, (node, length) in enumerate(zip(nodes, lengths)):
        cuts = np.r_[node, seams[(seams > node) & (seams < node+length)], node+length]
        permittivities = np.array([values[np.searchsorted(seams, (a+b)/2)] if inside[i] else 1.
                                   for a, b in zip(cuts[:-1], cuts[1:])])
        total = np.sum(np.diff(cuts)/permittivities)
        before = 0.
        for a, b, epsilon in zip(cuts[:-1], cuts[1:], permittivities):
            x = (a+b)/2+(b-a)/2*q
            coordinate = (before+(x-a)/epsilon)/total
            derivative = np.full_like(x, 1/(epsilon*total))
            lift = length*derivative
            exp = np.exp(-2j*np.pi*modes[:, None]*x[None])*(b-a)/2*weights
            for pack, material in zip(expected, (epsilon, 1.)):
                pack['mass'][:, i] += np.stack([exp@(material*coordinate**k) for k in range(3)], axis=-1)
                for key, value in zip(fields[1:], (derivative, derivative**2, lift, lift**2, lift*derivative)):
                    pack[key][:, i] += exp@(material*value)
            before += (b-a)/epsilon
    for actual, want in zip((body, gap), expected):
        for key in fields:
            np.testing.assert_allclose(actual[key], want[key], atol=2e-12, rtol=1e-13)


@pytest.mark.parametrize('seam', [0., 31., -31.])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_static_value_and_jit_shape_material_gradient_match_exact_laminate(seam, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(0, rows=4, edge_fraction=1e-4)

    def candidate(p):
        s, e0, e1, width = p
        return model.coefficients(0., dict(seams=jnp.array([s]), values=jnp.array([e0, e1])),
                                   width, jnp)[0][4, 0, 0].real

    def analytic(p):
        s, e0, e1, width = p
        return 1/(1-width/700+(width/2+s)/700/e0+(width/2-s)/700/e1)

    parameters = jnp.array([seam, 4., 9., 350.])
    got = jax.jit(jax.value_and_grad(candidate))(parameters)
    want = jax.value_and_grad(analytic)(parameters)
    np.testing.assert_allclose(got[0], want[0], atol=2e-12, rtol=0)
    np.testing.assert_allclose(got[1], want[1], atol=2e-11, rtol=0)
    jax.clear_caches()


@pytest.mark.parametrize('seams,values', [([0.], [4., 9.]), ([31.], [4., 9.]), ([-73., 19.], [4., 9., 6.25])])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_closed_constitutive_matrices_are_exact_even_for_unfitted_material_interfaces(seams, values, model_class):
    order = 3
    model = model_class(order, rows=4, edge_fraction=1e-4)
    actual, _ = model.matrices(0., dict(seams=seams, values=values))
    n, m = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1), indexing='ij')
    phase = (-1.)**(n.ravel()+m.ravel())
    for got, want in zip(actual, _closed_li(order, seams, values)):
        np.testing.assert_allclose(got, phase[:, None]*want*phase[None], atol=2e-10, rtol=0)


@pytest.mark.parametrize('polarization', ['p', 's'])
@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_closed_optical_material_and_seam_derivatives_match_independent_li(polarization, model_class):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    model = model_class(2, rows=4, edge_fraction=1e-4)
    objective = jax.jit(jax.value_and_grad(lambda p: model.transmission(
        0., dict(seams=p[:1], values=p[1:]), xp=jnp, polarization=polarization)['T']))

    def independent(p):
        return float(powers(2, _closed_li(2, p[:1], p[1:]), polarization=polarization)[0])

    for seam in (0., 31.):
        p = np.array([seam, 4., 9.])
        value, derivative = objective(p)
        assert float(value) == pytest.approx(independent(p), abs=2e-10)
        for i, step in enumerate((1e-3, 1e-4, 1e-4)):
            delta = np.eye(3)[i]*step
            secant = (independent(p+delta)-independent(p-delta))/(2*step)
            assert float(derivative[i]) == pytest.approx(secant, rel=3e-6, abs=2e-9)
    jax.clear_caches()


@pytest.mark.parametrize('model_class', MODEL_CLASSES)
def test_positive_gap_fields_have_unit_mean_and_conserve_lossless_power(model_class):
    model = model_class(3, rows=4, edge_fraction=1e-4)
    profile = dict(seams=[31.], values=[4., 9.])
    coefficients, metadata = model.coefficients(17., profile)
    np.testing.assert_allclose(coefficients[:4, 6, 6].reshape(2, 2), np.eye(2), atol=2e-12, rtol=0)
    assert metadata['cell_residual'] < 1e-9
    for pol in ('p', 's'):
        result = model.transmission(17., profile, polarization=pol)
        assert result['energy_error'] < 1e-11
        assert result['hermiticity'] < 1e-11


def test_positive_gap_node_sensitivity_discrepancy_decreases_with_mesh_refinement():
    jumps = []
    for subdivisions in (1, 2, 4):
        model = HarmonicCoordinateGapModel(2, rows=4, edge_fraction=1e-4, x_subdivisions=subdivisions)
        power = lambda s: model.transmission(17., dict(seams=[s], values=[4., 9.]), polarization='s')['T']
        h, center = 1e-4, power(0.)
        left, right = (center-power(-h))/h, (power(h)-center)/h
        jumps.append(abs(right-left))
    assert jumps[2] < jumps[1] < jumps[0]
    assert jumps[2] < .6*jumps[0]


@pytest.mark.parametrize('model_class', [
    pytest.param(HarmonicCoordinateGapModel, marks=pytest.mark.xfail(
        strict=True, reason='local harmonic basis retains a finite-cell seam cusp at a positive gap')),
    ContinuousFluxGapModel,
    MaterialGradedGapModel,
    PolynomialBodyGapModel,
    PolynomialFluxGapModel,
])
def test_positive_gap_real_seam_derivative_is_continuous_at_mesh_node(model_class):
    # Closed-laminate exactness does not establish general 2D interface
    # sensitivity. Preserve this unresolved gate in addition to the refinement
    # diagnostic so AD/FD consistency cannot be mistaken for adoption.
    model = model_class(2, rows=4, edge_fraction=1e-4)
    power = lambda s: model.transmission(17., dict(seams=[s], values=[4., 9.]), polarization='s')['T']
    center = power(0.)
    for h in (1e-4, 5e-5):
        left, right = (center-power(-h))/h, (power(h)-center)/h
        assert abs(left-right) < 1e-7
