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


def test_spectral_candidates_keep_conditional_execution_and_zero_contrast_mask():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    angles = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    ellipse = np.stack(
        [0.12 * np.cos(angles) + 0.65, 0.09 * np.sin(angles) + 0.65], axis=-1
    )
    rectangle = np.array([[0., 0.], [.2, 0.], [.2, .2], [0., .2]])
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    plan = b.plan_interfaces([(4., [rectangle]), (9., [ellipse])], cell, transform)

    def field(xp, epsilon):
        regions = [(epsilon, [xp.asarray(rectangle)]), (9., [xp.asarray(ellipse)])]
        segments = b.replay_interfaces(xp, regions, 1., cell, transform, plan)
        contrasts = b.interface_contrasts(xp, regions, 1., plan, xp.float64)
        return xp.stack(b.material_projector_fields(
            xp, (5, 5), cell, xp.float64, segments, contrasts, plan, transform,
        ))

    epsilons = np.array([1., 4., 9.])
    batched = jax.vmap(lambda epsilon: field(jnp, epsilon))
    # The rectangle disappears optically in lane zero, including at grid points
    # on its boundary. Retaining it as a candidate must not set the pole scale.
    want = np.stack([field(np, epsilon) for epsilon in epsilons])
    np.testing.assert_allclose(jax.jit(batched)(jnp.asarray(epsilons)), want,
                               rtol=0, atol=2e-12)

    def has_conditional(value):
        if hasattr(value, "eqns"):
            return any(e.primitive.name == "cond" or has_conditional(e.params)
                       for e in value.eqns)
        if hasattr(value, "jaxpr"):
            return has_conditional(value.jaxpr)
        if isinstance(value, dict):
            return any(has_conditional(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(has_conditional(v) for v in value)
        return False

    # A batched predicate becomes select, which executes both branches. Material
    # variation alone must leave this geometry-dependent conditional intact.
    assert has_conditional(jax.make_jaxpr(batched)(jnp.asarray(epsilons)))


def test_batched_dense_integral_bounds_memory_and_preserves_derivatives(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from wavebit.rcwa._engine import integrated_normal

    jax.config.update("jax_enable_x64", True)
    # Exercise overflow cheaply, with independently varying geometry/materials
    # and a zero-contrast lane that must retain the empty-field convention.
    monkeypatch.setattr(integrated_normal, "_CANDIDATE_CAPACITY", 32)
    angles = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    vertices = np.stack(
        [0.2 * np.cos(angles) + 0.5, 0.14 * np.sin(angles) + 0.5], axis=-1
    )
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    plan = b.plan_interfaces([(4.0, [vertices])], cell, transform)
    weights = np.arange(63).reshape(7, 9) / 63
    lanes = np.linspace(-3.0, 1.0, 32)

    def objective(xp, q, lane, chunk):
        v = xp.asarray(vertices) * q + xp.stack([lane * 0.001, xp.asarray(0.0)])
        regions = [(4.0 + lane, [v])]
        segments = b.replay_interfaces(xp, regions, 1.0, cell, transform, plan)
        contrasts = b.interface_contrasts(xp, regions, 1.0, plan, xp.float64)
        field = b.material_projector_fields(
            xp, (7, 9), cell, xp.float64, segments, contrasts, plan, transform,
            chunk=chunk,
        )[0]
        return xp.sum(field * xp.asarray(weights))

    def values(q, chunk):
        return jax.vmap(lambda lane: objective(jnp, q, lane, chunk))(jnp.asarray(lanes))

    q = jnp.ones(2)
    pointwise = jax.jit(jax.jacfwd(lambda q: values(q, 1))).lower(q).compile()
    batched = jax.jit(jax.jacfwd(lambda q: values(q, 63))).lower(q).compile()
    small = pointwise.memory_analysis().temp_size_in_bytes
    large = batched.memory_analysis().temp_size_in_bytes
    # A whole grid of points must not multiply the dense working set. Compare
    # the same compiler/input against its one-point control, not a platform RSS.
    assert large <= 2 * small, (small, large)
    got = np.asarray(batched(q))
    np.testing.assert_allclose(got, pointwise(q), rtol=2e-12, atol=2e-12)
    want = np.array([objective(np, np.ones(2), lane, 8) for lane in lanes])
    np.testing.assert_allclose(jax.jit(lambda q: values(q, 63))(q), want, rtol=0, atol=2e-12)
    h = 1e-5
    secant = np.stack([
        [(objective(np, np.ones(2) + h * axis, lane, 8)
          - objective(np, np.ones(2) - h * axis, lane, 8)) / (2 * h)
         for lane in lanes]
        for axis in np.eye(2)
    ], axis=1)
    np.testing.assert_allclose(got, secant, rtol=2e-5, atol=2e-7)
    reverse = jax.jit(jax.jacrev(lambda q: values(q, 63)))(q)
    np.testing.assert_allclose(reverse, secant, rtol=2e-5, atol=2e-7)


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


# -------------------------------------------------- a sliver that closes (2026-11-18)
#
# Two intervals bounding a sliver of one region (a gap between two shapes, a shape nearly
# meeting its own periodic image, a slot in one shape) become one interval when the sliver
# closes. Weighted separately, contrast**4 each, they stepped the field everywhere at contact:
# 2.6e-4 in total transmission for two of the boxes below at orders (5, 5) and the default grid. They are now
# blended toward the closed interval below `interfaces.facing_gap`.

PERIOD = 600.0


def _two_boxes(gap, angle=0.0, offset=0.0, height=0.3):
    from wavebit import rcwa

    c, s = np.cos(angle), np.sin(angle)
    wa, wb, hb = 0.3 * PERIOD, (0.3 if offset == 0.0 else 0.2) * PERIOD, height * PERIOD

    def box(cx, cy, w, h):
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        x0, y0 = c * cx - s * cy, s * cx + c * cy
        return tuple((x0 + c * x - s * y, y0 + s * x + c * y) for x, y in corners)

    return (
        rcwa.Region(
            geometry=rcwa.Polygon(vertices=box(-(wa + gap) / 2, 0.0, wa, wa)),
            material=_material(4.0),
        ),
        rcwa.Region(
            geometry=rcwa.Polygon(vertices=box((wb + gap) / 2, offset, wb, hb)),
            material=_material(9.0),
        ),
    )


def _material(value):
    from wavebit import Material

    return Material(value)


def _own_image(gap):
    from wavebit import rcwa

    w = PERIOD - gap
    box = rcwa.Polygon(vertices=((-w / 2, -90.0), (w / 2, -90.0), (w / 2, 90.0), (-w / 2, 90.0)))
    return (
        rcwa.Region(geometry=box, material=_material(4.0)),
        rcwa.Region(geometry=rcwa.Circle(radius=60.0, center=(0.0, 200.0)),
                    material=_material(9.0)),
    )


def _transmission(regions, grid=64):
    from wavebit import Units, rcwa

    layer = rcwa.Layer(thickness=0.4 * PERIOD, background=_material(1.0), regions=regions)
    problem = rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(period=(PERIOD, PERIOD), bottom=_material(2.25), layers=(layer,)),
        excitation=rcwa.Excitation(
            wavelength=840.0,
            sources=(rcwa.PlaneWave(theta=0.25, phi=0.4, polarization="p"),),
        ),
    )
    numerics = rcwa.Numerics(orders=(3, 3), shape_nvf_grid_size=grid)
    return float(rcwa.run(problem, numerics=numerics).T.sum())


@pytest.mark.parametrize(
    "scene",
    [
        lambda g: _two_boxes(g),
        lambda g: _two_boxes(g, angle=0.3),
        lambda g: _two_boxes(g, offset=0.1 * PERIOD, height=0.45),
        _own_image,
    ],
    ids=["boxes", "turned-boxes", "partial-overlap", "own-periodic-image"],
)
def test_closing_a_sliver_has_no_transmission_step(scene):
    # Before 2026-11-18 the first difference was 4.3e-4, 5.7e-4, 6.0e-4 and 1.6e-4 here.
    closed = _transmission(scene(0.0))
    differences = [abs(_transmission(scene(h)) - closed) for h in (1e-8, 1e-10)]
    assert differences[0] < 1e-9
    assert differences[1] < 0.05 * differences[0] + 2e-13


def test_closing_a_slot_in_one_shape_meets_the_filled_shape():
    # The slot's two sides face across background inside one region: closed, they leave no
    # interface at all, so their blended weight falls to zero. Before: 2.0e-4.
    from wavebit import rcwa

    a, t, g = 120.0, 60.0, 1e-8
    slot = ((-a, -a), (a, -a), (a, a), (g / 2, a), (g / 2, -a + t), (-g / 2, -a + t),
            (-g / 2, a), (-a, a))
    filled = ((-a, -a), (a, -a), (a, a), (-a, a))
    values = [
        _transmission((rcwa.Region(geometry=rcwa.Polygon(vertices=v), material=_material(4.0)),))
        for v in (slot, filled)
    ]
    assert abs(values[0] - values[1]) < 1e-9


def _box_pair(gap):
    w = 0.3
    left = np.array([[-w, -w / 2], [0.0, -w / 2], [0.0, w / 2], [-w, w / 2]]) - [gap / 2, 0.0]
    right = np.array([[0.0, -w / 2], [w, -w / 2], [w, w / 2], [0.0, w / 2]]) + [gap / 2, 0.0]
    return [(4.0, [left + 0.5]), (9.0, [right + 0.5])]


def _pair_field(xp, regions, plan, cell, transform, grid=(24, 24), facing=True):
    segments = b.replay_interfaces(xp, regions, 1.0, cell, transform, plan)
    contrasts = b.interface_contrasts(xp, regions, 1.0, plan, xp.float64)
    merged = b.facing_contrasts(xp, regions, 1.0, plan, xp.float64) if facing else None
    return b.material_projector_fields(
        xp, grid, cell, xp.float64, segments, contrasts, plan, transform,
        facing_contrasts=merged,
    )


def test_a_facing_pair_is_planned_only_below_the_gap_and_changes_nothing_at_it():
    from wavebit._common.geometry.interfaces import FACING_GAP_FRACTION

    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    ell = FACING_GAP_FRACTION
    for gap, planned in ((0.5 * ell, 1), (0.999 * ell, 1), (1.001 * ell, 0)):
        regions = _box_pair(gap)
        plan = b.plan_interfaces(regions, cell, transform)
        assert len(plan.facing) == planned
    # A pair whose separation has reached the gap carries no correction, bit for bit.
    regions = _box_pair(ell)
    plan = b.plan_interfaces(regions, cell, transform)
    plan = plan._replace(facing=b.plan_interfaces(_box_pair(0.5 * ell), cell, transform).facing)
    with_pair = np.array(_pair_field(np, regions, plan, cell, transform))
    without = np.array(_pair_field(np, regions, plan, cell, transform, facing=False))
    assert np.array_equal(with_pair, without)


def test_the_blend_reaches_the_contact_field():
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    fields = []
    for gap in (1e-9, 0.0):
        regions = _box_pair(gap)
        plan = b.plan_interfaces(regions, cell, transform)
        fields.append(np.array(_pair_field(np, regions, plan, cell, transform)))
    assert np.max(np.abs(fields[0] - fields[1])) < 1e-7
    # Without the blend the two weights stay separate: the step this replaced.
    regions = _box_pair(1e-9)
    plan = b.plan_interfaces(regions, cell, transform)
    unblended = np.array(_pair_field(np, regions, plan, cell, transform, facing=False))
    assert np.max(np.abs(unblended - fields[1])) > 0.1


def test_jax_derivative_through_the_blend_matches_differences():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from wavebit._common.geometry.interfaces import FACING_GAP_FRACTION

    jax.config.update("jax_enable_x64", True)
    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    gap = 0.4 * FACING_GAP_FRACTION
    plan = b.plan_interfaces(_box_pair(gap), cell, transform)
    assert plan.facing
    weights = np.arange(24 * 24).reshape(24, 24) / 576.0

    def objective(xp, g):
        regions = [(m, [xp.asarray(v) for v in loops]) for m, loops in _box_pair(0.0)]
        shift = xp.stack([g / 2, xp.asarray(0.0)])
        regions = [(regions[0][0], [regions[0][1][0] - shift]),
                   (regions[1][0], [regions[1][1][0] + shift])]
        return xp.sum(_pair_field(xp, regions, plan, cell, transform)[0] * xp.asarray(weights))

    derivative = jax.grad(lambda g: objective(jnp, g))(jnp.asarray(gap))
    h = 1e-3 * gap
    secant = (objective(np, np.asarray(gap + h)) - objective(np, np.asarray(gap - h))) / (2 * h)
    assert float(derivative) == pytest.approx(float(secant), rel=1e-5)
    assert abs(float(secant)) > 1.0


def test_torch_derivative_through_the_blend_matches_differences():
    torch = pytest.importorskip("torch")
    from array_api_compat import array_namespace
    from wavebit._common.geometry.interfaces import FACING_GAP_FRACTION

    cell = ((1.0, 0.0), (0.0, 1.0))
    transform = b.reduction_transform(cell)
    gap = 0.4 * FACING_GAP_FRACTION
    plan = b.plan_interfaces(_box_pair(gap), cell, transform)
    assert plan.facing
    weights = np.arange(24 * 24).reshape(24, 24) / 576.0

    def objective(xp, g):
        loops = [(m, [xp.asarray(v, dtype=xp.float64) for v in vs]) for m, vs in _box_pair(0.0)]
        shift = xp.stack([g / 2, xp.asarray(0.0, dtype=xp.float64)])
        regions = [(loops[0][0], [loops[0][1][0] - shift]), (loops[1][0], [loops[1][1][0] + shift])]
        field = _pair_field(xp, regions, plan, cell, transform)[0]
        return xp.sum(field * xp.asarray(weights, dtype=xp.float64))

    g = torch.tensor(gap, dtype=torch.float64, requires_grad=True)
    xp = array_namespace(g)
    objective(xp, g).backward()
    h = 1e-3 * gap
    secant = (objective(np, np.asarray(gap + h)) - objective(np, np.asarray(gap - h))) / (2 * h)
    assert float(g.grad) == pytest.approx(float(secant), rel=1e-5)
