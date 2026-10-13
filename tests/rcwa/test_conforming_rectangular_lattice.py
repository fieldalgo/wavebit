"""Physical cell metric, unequal periods, and differentiable rectangular lattices."""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
from tests.fixtures.common._conv_frame import about_centre
import pytest

from wavebit.rcwa._engine.conforming.polynomial_body import PolynomialBodyGapModel
from wavebit.rcwa._engine.parametric import assemble_eps_coeffs, conv_matrix_2d
from tests.rcwa.test_conforming_nvf import SETTINGS, _objective, solve


PARAMETERS = np.array([350., 29., 31., 4., 9., 1.7, 700., 430., .31, 17., -21.])


def scene(p=PARAMETERS, xp=np):
    width, gap, seam, e1, e2, background, px, py, angle, cx, cy = p
    rotation = xp.stack([xp.stack([xp.cos(angle), -xp.sin(angle)]),
                         xp.stack([xp.sin(angle), xp.cos(angle)])])
    shapes = []
    for left, right, epsilon in ((-width/2, seam, e1), (seam, width/2, e2)):
        center = rotation@xp.stack([cx+(left+right)/2, cy])
        shapes.append(dict(lx=right-left, ly=py-gap, cx=center[0], cy=center[1],
                           angle=angle, material=epsilon))
    return [dict(background=background, shapes=shapes[::-1])], rotation*xp.stack([px, py])[None]


@pytest.mark.research
@pytest.mark.parametrize('aspect', [.25, .8, 1.7, 4.])
@pytest.mark.parametrize('gap', [0., 29.])
def test_unequal_period_material_integrals_and_exact_contact(aspect, gap):
    p = PARAMETERS.copy()
    p[1], p[7] = gap, p[6]/aspect
    structure, lattice = scene(p)
    order = (2, 1)
    actual = _legacy_wavebit.ConformingNVF(order, **SETTINGS).conv_matrices(structure, lattice)
    layer = structure[0]
    eps, inverse = assemble_eps_coeffs(np, layer['shapes'], layer['background'],
                                       *order, lattice, np.complex128)
    tangent = about_centre(conv_matrix_2d(np, eps, *order), order)
    np.testing.assert_allclose(actual.epz_i[0], np.linalg.inv(tangent), atol=2e-12, rtol=0)
    if gap == 0:
        normal = np.linalg.inv(about_centre(conv_matrix_2d(np, inverse, *order), order))
        c, s = np.cos(p[8]), np.sin(p[8])
        expected = (c*c*normal+s*s*tangent, s*s*normal+c*c*tangent,
                    np.linalg.inv(tangent), c*s*(tangent-normal), c*s*(tangent-normal))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got[0], want, atol=4e-12, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('order', [(2, 1), (1, 0), (0, 1)])
def test_rectangular_lattice_rigid_motion_and_uniform_scaling_preserve_engine_powers(order):
    builder = _legacy_wavebit.ConformingNVF(order, **SETTINGS)
    structure, lattice = scene()
    baseline = solve(builder.conv_matrices(structure, lattice), lattice, order)
    p = PARAMETERS.copy()
    p[8] += .73
    p[9:] += [761., -582.]
    p[[0, 1, 2, 6, 7, 9, 10]] *= .004
    structure, lattice = scene(p)
    actual = solve(builder.conv_matrices(structure, lattice), lattice, order, scale=.004, angle=p[8])
    np.testing.assert_allclose(actual, baseline, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual.sum(axis=(-1, -2)).sum(axis=1), 1., atol=2e-10, rtol=0)


@pytest.mark.research
@pytest.mark.parametrize('body_degree,gap_degree', [(1, 1), (1, 4), (3, 1), (3, 4)])
@pytest.mark.parametrize('aspect', [.6, 1.8])
def test_cell_solution_is_stationary_for_the_physical_energy_and_has_curl_free_fields(
        monkeypatch, body_degree, gap_degree, aspect):
    model = PolynomialBodyGapModel(1, rows=2, body_degree=body_degree, gap_degree=gap_degree,
                                   grade_y=False, flux_basis='local', edge_fraction=1e-4)
    profile = dict(seams=[31.], values=[4., 9.])
    recorded = []
    original = model._solve_flux_cell
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded.append(result[:3])
        return result
    monkeypatch.setattr(model, '_solve_flux_cell', capture)
    coefficients, _ = model.coefficients(210., profile, aspect=aspect)
    np.testing.assert_allclose(coefficients[:4, 2, 2].reshape(2, 2), np.eye(2), atol=3e-12, rtol=0)
    n, m = np.meshgrid(model.modes, model.modes, indexing='ij')
    for bx, by in ((coefficients[0], coefficients[2]), (coefficients[1], coefficients[3])):
        np.testing.assert_allclose(m*by-aspect*n*bx, 0., atol=2e-11, rtol=0)

    g, edges, values, nodes, lengths, _ = model._profile_geometry(210., profile, 350., np)
    body, background = model._x_integrals(nodes, lengths, edges[1:-1], values, np)
    def physical_energy(v, a, b):
        # Independent variation of the projected physical energy. Projection
        # has separate tensor-Gauss controls, and does not read the cell system.
        _, gb, _ = model._project_body(v, a, g, body, background, np, metric_y=aspect**2)
        _, gg, _ = model._project_gap(v, a, b, g, background, np, metric_y=aspect**2)
        gram = (gb+gg)[2, 2]
        return gram[0, 0]+gram[1, 1]/aspect**2
    rng = np.random.default_rng(817)
    directions = [tuple(rng.normal(size=a.shape) for a in recorded[0]) for _ in range(3)]
    def variations(state):
        # Complex-step differentiates this real quadratic polynomial without
        # subtracting nearly equal energies; it is not a complex material solve.
        return np.array([physical_energy(*(a+1e-20j*d for a, d in zip(state, direction))).imag/1e-20
                         for direction in directions])
    np.testing.assert_allclose(variations(recorded[0]), 0., atol=3e-8, rtol=0)

    # Coordinate stretching alone is curl-free but does not minimize physical
    # energy. This control makes sure the stationarity test detects that error.
    model.coefficients(210., profile, aspect=1.)
    assert np.linalg.norm(variations(recorded[1])) > 1e-3


@pytest.mark.research
def test_independent_periods_remain_live_in_jit_and_in_a_reused_plan():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene())
    def objective(p, xp):
        return _objective(builder.conv_matrices(*scene(p, xp), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda p: objective(p, jnp)))(jnp.asarray(PARAMETERS))
    assert value == pytest.approx(objective(PARAMETERS, np), abs=3e-10)
    for i, step in enumerate([1e-3]*3+[1e-4]*3+[1e-3]*2+[1e-5]+[1e-3]*2):
        for scale in (1., .5):
            delta = np.eye(11)[i]*step*scale
            secant = (objective(PARAMETERS+delta, np)-objective(PARAMETERS-delta, np))/(2*step*scale)
            assert gradient[i] == pytest.approx(secant, abs=3e-7, rel=5e-6)
    p = PARAMETERS.copy()
    p[7] = 1100.
    np.testing.assert_allclose(objective(p, np), _objective(builder.conv_matrices(*scene(p)), np),
                               atol=2e-12, rtol=0)
    jax.clear_caches()


@pytest.mark.research
@pytest.mark.parametrize('angle', [0., .83])
@pytest.mark.parametrize('roundoff', [0., -np.spacing(430.)])
def test_rectangular_contact_retains_the_full_one_sided_gap_derivative(angle, roundoff):
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    p = PARAMETERS.copy()
    p[1], p[8] = roundoff, angle
    builder = _legacy_wavebit.ConformingNVF(1, **SETTINGS)
    plan = builder.plan(*scene(p))
    def objective(gap, xp):
        live = xp.asarray(p)+xp.asarray(np.eye(11)[1])*gap
        return _objective(builder.conv_matrices(*scene(live, xp), plan=plan, xp=xp), xp)
    value, gradient = jax.jit(jax.value_and_grad(lambda gap: objective(gap, jnp)))(0.)
    baseline = objective(0., np)
    assert value == pytest.approx(baseline, abs=3e-10)
    for h in (5e-7, 1e-7):
        secant = 2*(objective(h/2, np)-baseline)/(h/2)-(objective(h, np)-baseline)/h
        assert gradient == pytest.approx(secant, abs=3e-6)
    assert abs(gradient) > 1e-3
    with pytest.raises(_legacy_wavebit.StaleGeometryError):
        objective(-1e-8, np)
    jax.clear_caches()


@pytest.mark.parametrize('aspect', [0., -1., np.nan, np.inf, 1j, [1.]])
def test_invalid_cell_aspect_is_refused(aspect):
    with pytest.raises(ValueError, match='aspect'):
        PolynomialBodyGapModel(0, **SETTINGS).coefficients(17., aspect=aspect)


def test_zero_or_sheared_traced_second_axis_poison_every_layer():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    builder = _legacy_wavebit.ConformingNVF(0, **SETTINGS)
    p = PARAMETERS.copy()
    p[8] = 0.
    structure, lattice = scene(p)
    structure.append(dict(background=2.))
    plan = builder.plan(structure, lattice)
    build = jax.jit(lambda axis: builder.conv_matrices(
        structure, jnp.stack([jnp.asarray(lattice[:, 0]), axis], axis=1), plan=plan, xp=jnp))
    for axis in ([0., 0.], [1., 430.], [0., -430.], [0., np.inf]):
        assert all(np.isnan(np.asarray(slot)).all() for slot in build(jnp.asarray(axis)))
    jax.clear_caches()
