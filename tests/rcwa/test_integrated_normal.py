"""Independent quadrature and representation invariants of the painted line measure."""

import numpy as np
import pytest
from scipy.integrate import quad
from wavebit.rcwa._engine.integrated_normal import _interval_mass
from wavebit.rcwa._engine import boundary as b


@pytest.mark.parametrize(
    "point",
    [
        (0.15, 0.23),
        (0.7, 0.7),
        (0.1, -0.00000001),
        (0.1, 0.00000001),
        (0.9, 0.4),
        (0.9, 0.4000000001),
        (0.9, 0.3999999999),
    ],
)
def test_interval_mass_matches_independent_adaptive_quadrature(point):
    a = np.array([[0.1, 0.2], [0.3, 0.5], [-0.3, -0.2]])
    e = np.array([[0.4, 0.1], [0.2, -0.5], [0.5, 0.6]])
    p = np.array([point])
    l2 = (e * e).sum(1)
    got = _interval_mass(np, p, a, e, l2, 1.0, 1e-26, np.ones(3, dtype=bool))[0]
    expected = []
    for start, edge in zip(a, e):

        def integrand(t):
            r2 = np.sum((start + t * edge - p[0]) ** 2)
            return max(1 / r2 - 1, 0) ** 2 * np.linalg.norm(edge)

        expected.append(quad(integrand, 0, 1, epsabs=1e-9, epsrel=1e-11, limit=200)[0])
    expected = np.array(expected)
    np.testing.assert_allclose(
        got / got.sum(), expected / expected.sum(), rtol=2e-10, atol=2e-12
    )


def projector(vertices, grid=(32, 32)):
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    regs = [(4.0, [vertices])]
    plan = b.plan_interfaces(regs, cell, transform)
    segments = b.replay_interfaces(np, regs, 1.0, cell, transform, plan)
    contrasts = b.interface_contrasts(np, regs, 1.0, plan, np.float64)
    return np.array(
        b.material_projector_fields(
            np, grid, cell, np.float64, segments, contrasts, plan, transform
        )
    )


def test_subdivision_and_periodic_translation_preserve_the_projector():
    whole = np.array([[0.1, 0.2], [0.7, 0.2], [0.7, 0.6], [0.1, 0.6]])
    split = np.insert(whole, 1, [0.3, 0.2], axis=0)
    want = projector(whole)
    for vertices in (split, whole + [1.0, 0.0], split + [-2.0, 1.0]):
        np.testing.assert_allclose(projector(vertices), want, atol=2e-12, rtol=0)


def test_a_vanishing_interval_has_vanishing_mass():
    point = np.array([[0.3, 0.2]])
    a = np.array([[0.0, 0.0], [0.5, 0.5]])
    ratios = []
    for length in (1e-3, 1e-5, 1e-7):
        e = np.array([[length, 0.0], [0.2, 0.0]])
        mass = _interval_mass(
            np, point, a, e, (e * e).sum(1), 1.0, 1e-26, np.ones(2, dtype=bool)
        )[0]
        ratios.append(mass[0] / mass[1])
    np.testing.assert_allclose(np.array(ratios[1:]) / ratios[:-1], 0.01, rtol=0.02)


@pytest.mark.parametrize(
    "parameter,centre", [("angle", 0.30354672169878616), ("centre", 91.87187983049787)]
)
def test_disappearing_painted_facet_has_no_transmission_step(parameter, centre):
    from wavebit import Material, Units, rcwa

    def value(x):
        rectangle = rcwa.Rectangle(
            size=(420.0, 210.0),
            center=(-60.0, 0.0),
            angle=x if parameter == "angle" else 0.3,
        )
        ellipse = rcwa.Ellipse(
            radii=(160.0, 100.0),
            center=(x if parameter == "centre" else 90.0, 40.0),
            angle=-0.4,
        )
        layer = rcwa.Layer(
            thickness=300.0,
            background=Material(1.0),
            regions=(
                rcwa.Region(geometry=rectangle, material=Material(12.25)),
                rcwa.Region(geometry=ellipse, material=Material(4.0)),
            ),
        )
        problem = rcwa.Problem(
            units=Units(length="nm"),
            stack=rcwa.Stack(
                period=(700.0, 700.0), bottom=Material(2.25), layers=(layer,)
            ),
            excitation=rcwa.Excitation(
                wavelength=900.0,
                sources=(rcwa.PlaneWave(theta=0.2, phi=0.3, polarization="p"),),
            ),
        )
        return float(
            rcwa.run(
                problem, numerics=rcwa.Numerics(orders=(4, 4), shape_nvf_grid_size=64)
            ).T.sum()
        )

    differences = [abs(value(centre + h) - value(centre - h)) for h in (1e-8, 1e-10)]
    assert differences[0] < 1e-9
    assert differences[1] < 0.05 * differences[0] + 2e-13


def test_splitting_at_a_junction_preserves_the_singular_limit():
    point = np.zeros((1, 2))

    def weights(a, e):
        a, e = np.array(a), np.array(e)
        return _interval_mass(
            np, point, a, e, (e * e).sum(1), 1.0, 1e-26, np.ones(len(a), dtype=bool)
        )[0]

    whole = weights([[-0.3, 0.0], [0.0, -0.4]], [[0.6, 0.0], [0.0, 0.8]])
    split = weights(
        [[-0.3, 0.0], [0.0, 0.0], [0.0, -0.4]], [[0.3, 0.0], [0.3, 0.0], [0.0, 0.8]]
    )
    assert split[:2].sum() / split.sum() == pytest.approx(
        whole[0] / whole.sum(), abs=1e-14
    )


@pytest.mark.parametrize("capacity", [32, 1024])
def test_jax_sparse_and_dense_fallback_match_values_and_shape_derivatives(
    monkeypatch, capacity
):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from wavebit.rcwa._engine import integrated_normal

    jax.config.update("jax_enable_x64", True)
    monkeypatch.setattr(integrated_normal, "_CANDIDATE_CAPACITY", capacity)
    angles = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    vertices = np.stack(
        [0.2 * np.cos(angles) + 0.5, 0.14 * np.sin(angles) + 0.5], axis=-1
    )
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    plan = b.plan_interfaces([(4.0, [vertices])], cell, transform)
    weights = np.arange(63).reshape(7, 9) / 63

    def field(xp, x, chunk):
        v = xp.asarray(vertices) * xp.stack([x, xp.asarray(1.0)])
        regions = [(4.0, [v])]
        segments = b.replay_interfaces(xp, regions, 1.0, cell, transform, plan)
        contrasts = b.interface_contrasts(xp, regions, 1.0, plan, xp.float64)
        return b.material_projector_fields(
            xp,
            (7, 9),
            cell,
            xp.float64,
            segments,
            contrasts,
            plan,
            transform,
            chunk=chunk,
        )[0]

    want = field(np, np.asarray(1.0), 8)

    def objective(x):
        return jnp.sum(field(jnp, x, 16) * jnp.asarray(weights))

    got, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    assert float(got) == pytest.approx(float((want * weights).sum()), abs=2e-12)
    h = 1e-5
    secant = (
        (field(np, np.asarray(1.0 + h), 8) - field(np, np.asarray(1.0 - h), 8))
        * weights
    ).sum() / (2 * h)
    assert float(derivative) == pytest.approx(float(secant), rel=2e-5, abs=2e-7)


def test_compact_support_has_no_value_or_slope_step():
    a = np.array([[-0.2, 0.0], [-0.2, 1.3]])
    e = np.array([[0.4, 0.0], [0.4, 0.0]])

    def fraction(y):
        mass = _interval_mass(
            np,
            np.array([[0.0, y]]),
            a,
            e,
            (e * e).sum(1),
            1.0,
            1e-26,
            np.ones(2, dtype=bool),
        )[0]
        return mass[0] / mass.sum()

    assert fraction(1.001) == 0.0
    assert fraction(1.0) == 0.0
    coarse = fraction(1.0 - 1e-3)
    fine = fraction(1.0 - 1e-4)
    assert 0.0 < fine < 0.01 * coarse


def test_single_precision_slots_keep_high_precision_polygon_area():
    # Mean permittivity is background + area * contrast, independently of NVF.
    # Downcasting these input vertices before intersection replay erases their area.
    mats = b.to_conv_mat_boundary_layer(
        np,
        [dict(type="rectangle", lx=1e-5, ly=1e-5, material=1e10 + 1)],
        1.0,
        0,
        0,
        ((1.0, 0.0), (0.0, 1.0)),
        np.complex64,
        laurent=True,
    )
    assert mats[0][0, 0] == pytest.approx(2.0, abs=1e-6)
