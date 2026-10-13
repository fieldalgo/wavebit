"""Gap-only automatic differentiation of the exact finite pair family.

Independent references retain original rational nodal/Laurent equations and
moving-simplex Duffy moments. No private arrays or scratch imports are used.
Only gap derivatives are qualified; live geometry/material replay does not
qualify their arbitrary tangents. Optional JAX is imported inside fixtures.
"""

from functools import lru_cache

import numpy as np
import pytest

from wavebit.rcwa._engine.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit.rcwa._engine.conforming.cartesian_pair_gap_autodiff import CartesianPairGapEvaluator
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from tests.rcwa.test_nvf_cartesian_pair_gap import CASES, CRITERIA, _oracle_differentiated_native_solve_reference, _reference, analytic_material, settings_for

pytestmark = pytest.mark.research

DETAILS = ("M", "f", "solution", "weights", "fields", "original_flux", "original_load")
GAPS = (0.0, 1 / 64, 1 / 16)
UNIT_CASES = ("main_j0_r0", "main_j0_r1", "different_r0", "different_r1")
VALUE_GUARD, DERIVATIVE_GUARD, RESIDUAL_GUARD = 3e-10, 3e-9, 1e-10


def _declarations():
    rows = []
    for kind in ("primary", "homogeneous"):
        rows += [
            dict(id=f"{kind}_{name}_{i}", case=name, s=s, kind=kind)
            for name in CASES
            for i, s in enumerate(GAPS)
        ]
    rows += [
        dict(id=f"units_{name}_{i}", case=name, s=s, kind="units", alpha=0.01)
        for name in UNIT_CASES
        for i, s in enumerate(GAPS)
    ]
    rows += [
        dict(
            id=f"orders_{x}{y}_{i}", case="main_j0_r0", s=s, kind="orders", order=(x, y)
        )
        for x, y in ((0, 0), (1, 0), (0, 1))
        for i, s in enumerate(GAPS[:2])
    ]
    rows += [
        dict(id=f"lossless_main_j0_r0_{i}", case="main_j0_r0", s=s, kind="lossless")
        for i, s in enumerate(GAPS[:2])
    ]
    return {row["id"]: row for row in rows}


DECLARATIONS = _declarations()


def _maximum(value):
    value = np.asarray(value)
    assert np.isfinite(value).all()
    return float(np.max(np.abs(value)))


def _finite(*values):
    assert all(np.isfinite(np.asarray(value)).all() for value in values)


def _scene(identifier):
    row = DECLARATIONS[identifier]
    case = CASES[row["case"]]
    settings = {
        name: np.asarray(value) for name, value in settings_for(CRITERIA, case).items()
    }
    if row["kind"] == "units":
        for key in ("basis", "widths", "height", "offset"):
            settings[key] = settings[key] * row["alpha"]
    if row["kind"] == "homogeneous":
        settings["epsilon"] = np.repeat(settings["background"], 2)
    if row["kind"] == "lossless":
        settings["epsilon"], settings["background"] = (
            settings["epsilon"].real,
            settings["background"].real,
        )
    plan = CartesianPairOpeningPlan.from_geometry(
        **{
            key: value
            for key, value in settings.items()
            if key not in ("epsilon", "background")
        },
        shift=case["shift"],
        refinement=case["refinement"],
    )
    cell = CartesianPairGapCell(plan, row.get("order", case["order"]))
    length = float(np.sqrt(np.linalg.det(settings["basis"])))
    return settings, cell, length, length * row["s"]


@lru_cache(None)
def _independent(identifier):
    settings, cell, length, gap = _scene(identifier)
    q16, dq16, r16 = _reference(cell, settings, gap, 16)
    q24, dq24, r24 = _reference(cell, settings, gap, 24)
    slots, dslots = _oracle_differentiated_native_solve_reference(q24, dq24, cell.order)
    nv, nd, nmeta, _ = cell.coefficients(gap=gap, **settings)
    ns, nds, _, _ = cell.closure(gap=gap, **settings)
    material, dmaterial = analytic_material(settings, cell.order, gap)
    _finite(
        q16,
        dq16,
        q24,
        dq24,
        slots,
        dslots,
        nv,
        nd,
        *ns,
        *nds,
        r16["residuals"],
        r24["residuals"],
    )
    assert _maximum(q16 - q24) <= VALUE_GUARD
    assert length * _maximum(dq16 - dq24) <= VALUE_GUARD
    assert max(*r16["residuals"], *r24["residuals"]) <= RESIDUAL_GUARD
    return dict(
        settings=settings,
        cell=cell,
        length=length,
        gap=gap,
        value=q24,
        derivative=dq24,
        slots=slots,
        dslots=dslots,
        reference=r24,
        native_value=nv,
        native_derivative=nd,
        native_slots=np.stack(ns),
        native_dslots=np.stack(nds),
        native_meta=nmeta,
        material=material,
        dmaterial=dmaterial,
    )


def _closure(value, derivative, slots, dslots, order, length):
    ox, oy = order
    labels = np.array([(m, n) for n in range(-oy, oy + 1) for m in range(-ox, ox + 1)])
    differences = labels[:, None] - labels[None]
    c = value[:, differences[..., 1] + 2 * oy, differences[..., 0] + 2 * ox]
    dc = (
        length
        * derivative[:, differences[..., 1] + 2 * oy, differences[..., 0] + 2 * ox]
    )
    q, dq = [np.block([[v[0], v[1]], [v[2], v[3]]]) for v in (c, dc)]
    g, dg = [np.block([[v[4], v[5]], [v[8], v[6]]]) for v in (c, dc)]
    h, dh = [
        np.block([[v[0, 0], -v[3, 0]], [-v[4, 0], v[1, 0]]])
        for v in (slots, length * dslots)
    ]
    z, dz = slots[2, 0], length * dslots[2, 0]
    residuals = [
        np.linalg.norm(q.conj().T @ h @ q - g) / max(np.linalg.norm(g), 1.0),
        np.linalg.norm(
            dq.conj().T @ h @ q + q.conj().T @ dh @ q + q.conj().T @ h @ dq - dg
        )
        / max(np.linalg.norm(dg), 1.0),
        np.linalg.norm(c[7] @ z - np.eye(len(z))),
        np.linalg.norm(dc[7] @ z + c[7] @ dz)
        / max(np.linalg.norm(dc[7]) * np.linalg.norm(z), 1.0),
    ]
    _finite(residuals)
    assert max(residuals) <= RESIDUAL_GUARD
    assert np.linalg.eigvalsh((h + h.conj().T) / 2).min() > 0
    assert np.linalg.eigvalsh((h - h.conj().T) / (2j)).min() > -1e-9
    return h, dh


@pytest.mark.parametrize("identifier", DECLARATIONS)
def test_numpy_values_against_original_equations(identifier):
    r = _independent(identifier)
    evaluator = CartesianPairGapEvaluator(r["cell"].plan, r["cell"].order)
    value, slots, metadata, detail = evaluator.evaluate(gap=r["gap"], **r["settings"])
    _finite(value, *slots, *detail.values())
    assert bool(metadata["valid"])
    assert _maximum(value - r["value"]) <= VALUE_GUARD
    assert _maximum(np.stack(slots) - r["slots"]) <= VALUE_GUARD
    assert _maximum(value - r["native_value"]) <= VALUE_GUARD
    assert _maximum(np.stack(slots) - r["native_slots"]) <= VALUE_GUARD
    assert metadata["linear_residual"] <= RESIDUAL_GUARD
    assert metadata["original_flux_residual"] <= RESIDUAL_GUARD


@pytest.fixture(scope="module")
def jax_runtime():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    @lru_cache(None)
    def program(shift, refinement, order):
        cell = CartesianPairGapEvaluator(
            CartesianPairOpeningPlan(shift, refinement), order
        )

        def outputs(gap, basis, widths, height, offset, epsilon, background):
            value, slots, _, detail = cell.evaluate(
                gap=gap,
                basis=basis,
                widths=widths,
                height=height,
                offset=offset,
                epsilon=epsilon,
                background=background,
                xp=jnp,
            )
            return value, jnp.stack(slots), *(detail[name] for name in DETAILS)

        def tangent(gap, dgap, *args):
            return jax.jvp(lambda g: outputs(g, *args), (gap,), (dgap,))

        return cell, outputs, tangent, jax.jit(tangent)

    @lru_cache(None)
    def observe(identifier):
        r = _independent(identifier)
        cell = r["cell"]
        program_value = program(
            cell.plan.shift, cell.plan.refinement, tuple(cell.order)
        )
        args = tuple(
            jnp.asarray(
                r["settings"][key],
                dtype=jnp.complex128
                if key in ("epsilon", "background")
                else jnp.float64,
            )
            for key in ("basis", "widths", "height", "offset", "epsilon", "background")
        )
        v, d = program_value[3](jnp.asarray(r["gap"]), jnp.asarray(1.0), *args)
        return (
            tuple(np.asarray(x) for x in v),
            tuple(np.asarray(x) for x in d),
            args,
            program_value,
        )

    yield jax, jnp, program, observe
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("identifier", DECLARATIONS)
def test_actual_gap_jvp_against_original_directional_solution(identifier, jax_runtime):
    _, jnp, _, observe = jax_runtime
    r = _independent(identifier)
    length = r["length"]
    v, d, args, program = observe(identifier)
    _finite(*v, *d)
    value, slots = v[:2]
    dv, ds = d[:2]
    assert _maximum(value - r["value"]) <= VALUE_GUARD
    assert _maximum(slots - r["slots"]) <= VALUE_GUARD
    assert length * _maximum(dv - r["derivative"]) <= DERIVATIVE_GUARD
    assert length * _maximum(ds - r["dslots"]) <= DERIVATIVE_GUARD
    assert length * _maximum(dv - r["native_derivative"]) <= DERIVATIVE_GUARD
    assert length * _maximum(ds - r["native_dslots"]) <= DERIVATIVE_GUARD
    assert _maximum(value[7] - r["material"]) <= VALUE_GUARD
    assert length * _maximum(dv[7] - r["dmaterial"]) <= DERIVATIVE_GUARD
    matrix, f, x = v[2:5]
    dm, df, dx = (length * z for z in d[2:5])
    residuals = [
        np.linalg.norm(matrix @ x - f) / max(np.linalg.norm(f), 1.0),
        np.linalg.norm(matrix @ dx + dm @ x - df) / max(np.linalg.norm(df), 1.0),
        np.linalg.norm(v[7]) / max(np.linalg.norm(v[8]), 1.0),
        np.linalg.norm(length * d[7]) / max(np.linalg.norm(length * d[8]), 1.0),
    ]
    _finite(residuals)
    assert max(residuals) <= RESIDUAL_GUARD
    h, dh = _closure(value, dv, slots, ds, r["cell"].order, length)
    ox, oy = r["cell"].order
    delta = np.zeros(value.shape[1:])
    delta[2 * oy, 2 * ox] = 1
    assert _maximum(v[5].sum(axis=0) - delta) <= VALUE_GUARD
    assert _maximum((length * d[5]).sum(axis=0)) <= DERIVATIVE_GUARD
    assert _maximum(value[:4, 2 * oy, 2 * ox] - [1, 0, 0, 1]) <= VALUE_GUARD
    assert length * _maximum(dv[:4, 2 * oy, 2 * ox]) <= DERIVATIVE_GUARD
    assert (
        _maximum(v[5].reshape(len(v[5]), -1) - r["reference"]["weights"]) <= VALUE_GUARD
    )
    assert (
        _maximum(
            (length * d[5]).reshape(len(v[5]), -1)
            - length * r["reference"]["dweights_per_unit"]
        )
        <= DERIVATIVE_GUARD
    )
    assert _maximum(v[6] - r["reference"]["fields"]) <= VALUE_GUARD
    mask = ~r["cell"].data["thin"] if r["gap"] == 0 else np.ones(len(v[6]), bool)
    assert (
        _maximum(length * d[6][mask] - length * r["reference"]["regular_dfields"][mask])
        <= DERIVATIVE_GUARD
    )
    row = DECLARATIONS[identifier]
    if row["kind"] == "primary":
        ev, ed = program[2](jnp.asarray(r["gap"]), jnp.asarray(1.0), *args)
        _finite(*ev, *ed)
        assert max(_maximum(np.asarray(ev[i]) - v[i]) for i in (0, 1)) <= VALUE_GUARD
        assert (
            length * max(_maximum(np.asarray(ed[i]) - d[i]) for i in (0, 1))
            <= DERIVATIVE_GUARD
        )
    if row["kind"] == "homogeneous":
        bg = r["settings"]["background"]
        exact = np.zeros_like(value)
        exact[:, 2 * oy, 2 * ox] = [1, 0, 0, 1, bg, 0, bg, bg, 0]
        assert _maximum(value - exact) <= VALUE_GUARD
        assert length * max(_maximum(dv), _maximum(ds)) <= DERIVATIVE_GUARD
    if row["kind"] == "lossless":
        assert _maximum(h - h.conj().T) <= VALUE_GUARD
        assert _maximum((h - h.conj().T) / (2j)) <= VALUE_GUARD
        assert _maximum(dh - dh.conj().T) <= DERIVATIVE_GUARD
        assert _maximum((dh - dh.conj().T) / (2j)) <= DERIVATIVE_GUARD


@pytest.mark.parametrize(
    "refinement,power,index",
    [(r, j, i) for r in (0, 1) for j in (1, 2, 3) for i in range(3)],
)
def test_signed_basis_maps_both_selected_closure_axes(
    refinement, power, index, jax_runtime
):
    observe = jax_runtime[3]
    key = f"primary_main_j0_r{refinement}_{index}"
    target = f"primary_main_j{power}_r{refinement}_{index}"
    bv, bd, *_ = observe(key)
    tv, td, *_ = observe(target)
    ox, oy = _independent(target)["cell"].order
    p = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), power)
    n, m = np.meshgrid(
        np.arange(-2 * oy, 2 * oy + 1), np.arange(-2 * ox, 2 * ox + 1), indexing="ij"
    )
    old = np.stack([m, n], axis=-1) @ p.T
    lookup = {
        (m, n): i
        for i, (m, n) in enumerate((m, n) for n in range(-2, 3) for m in range(-1, 2))
    }
    indices = np.array(
        [
            lookup[tuple(np.array([m, n]) @ p.T)]
            for n in range(-oy, oy + 1)
            for m in range(-ox, ox + 1)
        ]
    )
    length = _independent(key)["length"]
    for a, b, guard in (
        (tv, bv, VALUE_GUARD),
        (
            tuple(length * x for x in td),
            tuple(length * x for x in bd),
            DERIVATIVE_GUARD,
        ),
    ):
        assert _maximum(a[0] - b[0][:, old[..., 1] + 4, old[..., 0] + 2]) <= guard
        assert _maximum(a[1] - b[1][:, :, indices[:, None], indices]) <= guard


@pytest.mark.parametrize("name,index", [(n, i) for n in UNIT_CASES for i in range(3)])
def test_common_units_preserve_values_and_inverse_length_derivative(
    name, index, jax_runtime
):
    observe = jax_runtime[3]
    key = f"primary_{name}_{index}"
    bv, bd, *_ = observe(key)
    uv, ud, *_ = observe(f"units_{name}_{index}")
    length = _independent(key)["length"]
    for o in (0, 1):
        assert _maximum(uv[o] - bv[o]) <= VALUE_GUARD
        assert length * _maximum(0.01 * ud[o] - bd[o]) <= DERIVATIVE_GUARD


@pytest.mark.parametrize(
    "name,tangent", [(n, t) for n in UNIT_CASES for t in (0.0, -1.0)]
)
def test_endpoint_tangent_linearity_is_formal_for_negative_tangent(
    name, tangent, jax_runtime
):
    _, jnp, _, observe = jax_runtime
    key = f"primary_{name}_0"
    bv, bd, args, program = observe(key)
    r = _independent(key)
    value, derivative = program[3](jnp.asarray(0.0), jnp.asarray(tangent), *args)
    _finite(*value, *derivative)
    for o in (0, 1):
        assert _maximum(np.asarray(value[o]) - bv[o]) <= VALUE_GUARD
        assert (
            r["length"] * _maximum(np.asarray(derivative[o]) - tangent * bd[o])
            <= DERIVATIVE_GUARD
        )


@pytest.mark.parametrize("index,output", [(i, o) for i in (0, 1) for o in (0, 1)])
def test_scalar_reverse_gap_uses_independent_conjugate_seed(index, output, jax_runtime):
    jax, jnp, _, observe = jax_runtime
    key = f"primary_main_j0_r1_{index}"
    v, d, args, program = observe(key)
    r = _independent(key)
    axis = np.arange(v[output].size)
    seed = ((1 + axis % 7) / 7 + 1j * (axis % 5) / 5).reshape(v[output].shape)
    reference = r["derivative" if output == 0 else "dslots"]
    expected = np.real(np.vdot(seed, reference))
    actual = np.asarray(
        jax.jit(
            jax.grad(
                lambda gap: jnp.real(
                    jnp.vdot(jnp.asarray(seed), program[1](gap, *args)[output])
                )
            )
        )(jnp.asarray(r["gap"]))
    )
    _finite(actual, expected)
    assert r["length"] * abs(actual - expected) <= DERIVATIVE_GUARD
    assert (
        r["length"] * abs(actual - np.real(np.vdot(seed, d[output])))
        <= DERIVATIVE_GUARD
    )


def test_one_compiled_chart_accepts_two_live_physical_scenes(jax_runtime):
    observe = jax_runtime[3]
    for i in range(3):
        first = observe(f"primary_main_j1_r1_{i}")
        second = observe(f"primary_different_r1_{i}")
        assert first[3][3] is second[3][3]
    # This control also runs independently of the parametrized test order.
    for case in CASES.values():
        if case["shift"] is not None:
            jax_runtime[2](case["shift"], case["refinement"], tuple(case["order"]))
    for order in ((0, 0), (1, 0), (0, 1)):
        jax_runtime[2](0, 0, order)
    assert jax_runtime[2].cache_info().currsize == 11


INVALIDS = (
    "negative_gap",
    "nan_gap",
    "zero_basis",
    "overflow_basis",
    "zero_width",
    "zero_height",
    "escaped_height",
    "nan_offset",
    "gain",
    "zero_background",
)


def _invalid(name):
    settings, cell, _, _ = _scene("primary_main_j0_r0_0")
    settings = dict(settings, gap=0.0)
    replacements = {
        "negative_gap": ("gap", -1.0),
        "nan_gap": ("gap", np.nan),
        "zero_basis": ("basis", np.zeros((2, 2))),
        "overflow_basis": ("basis", np.eye(2) * 1e200),
        "zero_width": ("widths", np.array([0.0, 193.0])),
        "zero_height": ("height", 0.0),
        "escaped_height": ("height", 1e6),
        "nan_offset": ("offset", np.array([np.nan, 0.0])),
        "gain": ("epsilon", np.array([4 - 0.1j, 9.0])),
        "zero_background": ("background", 0.0),
    }
    key, value = replacements[name]
    settings[key] = value
    return settings, cell


@pytest.mark.parametrize("name", INVALIDS)
def test_invalid_numpy_raises(name):
    settings, cell = _invalid(name)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        CartesianPairGapEvaluator(cell.plan, cell.order).evaluate(**settings)


@pytest.mark.parametrize("name", INVALIDS)
def test_invalid_jit_primal_is_nan_and_eager_refuses(name, jax_runtime):
    _, jnp, program, _ = jax_runtime
    settings, cell = _invalid(name)
    evaluator, _, _, compiled = program(
        cell.plan.shift, cell.plan.refinement, tuple(cell.order)
    )
    args = tuple(
        jnp.asarray(
            settings[key],
            dtype=jnp.complex128 if key in ("epsilon", "background") else jnp.float64,
        )
        for key in ("basis", "widths", "height", "offset", "epsilon", "background")
    )
    value, _ = compiled(jnp.asarray(settings["gap"]), jnp.asarray(1.0), *args)
    assert np.isnan(np.asarray(value[0])).all() and np.isnan(np.asarray(value[1])).all()
    # Derivatives outside the admitted primal domain are deliberately undefined.
    with pytest.raises(ValueError):
        evaluator.evaluate(**settings, xp=jnp)


@pytest.mark.parametrize(
    "kind",
    (
        "complex_gap",
        "boolean_gap",
        "unsupported_namespace",
        "plan_assignment",
        "order_assignment",
        "mapping_write",
    ),
)
def test_static_admission_and_frozen_choices(kind):
    settings, cell, _, gap = _scene("primary_main_j0_r0_0")
    evaluator = CartesianPairGapEvaluator(cell.plan, cell.order)
    actions = {
        "complex_gap": lambda: evaluator.evaluate(gap=0j, **settings),
        "boolean_gap": lambda: evaluator.evaluate(gap=True, **settings),
        "unsupported_namespace": lambda: evaluator.evaluate(
            gap=gap, **settings, xp=object()
        ),
        "plan_assignment": lambda: setattr(evaluator, "plan", None),
        "order_assignment": lambda: setattr(evaluator, "order", None),
        "mapping_write": lambda: evaluator.data["mapping"].setflags(write=True),
    }
    with pytest.raises((ValueError, TypeError, AttributeError)):
        actions[kind]()


def test_planning_refuses_traced_geometry(jax_runtime):
    jax, jnp, _, _ = jax_runtime
    settings, _, _, _ = _scene("primary_main_j0_r0_0")
    planner = jax.jit(
        lambda width: CartesianPairOpeningPlan.from_geometry(
            basis=settings["basis"],
            widths=width,
            height=settings["height"],
            offset=settings["offset"],
        )
    )
    with pytest.raises(jax.errors.TracerArrayConversionError):
        planner(jnp.asarray(settings["widths"]))


def test_x64_disabled_refuses(jax_runtime):
    jax, jnp, _, _ = jax_runtime
    settings, cell, _, gap = _scene("primary_main_j0_r0_0")
    evaluator = CartesianPairGapEvaluator(cell.plan, cell.order)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError):
            evaluator.evaluate(gap=gap, **settings, xp=jnp)
    finally:
        jax.config.update("jax_enable_x64", True)
