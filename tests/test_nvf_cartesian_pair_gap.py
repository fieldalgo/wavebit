"""Exact finite-gap verification using original nodal and moving-simplex equations.

The independently qualified endpoint helpers share immutable topology only.
Positive-gap references use rational physical Vandermonde matrices and Duffy
quadrature, not the candidate rescaled blocks. No private checkpoints, AD or
optical solve is required. The fixed guards apply per g/sqrt(det(basis)).
"""

from fractions import Fraction as Q
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from wavebit._core.conforming.cartesian_pair import CartesianPairCell, CartesianPairPlan
from wavebit._core.conforming.cartesian_pair_gap import CartesianPairGapCell
from wavebit._core.conforming.cartesian_pair_opening import (
    CartesianPairOpeningCell,
    CartesianPairOpeningPlan,
)
from test_nvf_cartesian_pair_opening import (
    CASES,
    CRITERIA,
    POWERS,
    _oracle_differentiated_native_solve_reference,
    _oracle_geometry,
    _oracle_inverse,
    _oracle_reference,
    maximum,
    settings_for,
)

GAPS = (0.0, 1 / 64, 1 / 16)


def _reference(cell, settings, gap, quadrature):
    if gap == 0:
        value, derivative, metadata = _oracle_reference(cell, settings, quadrature)
        return value, derivative, metadata
    if gap < 0:
        raise ValueError("reference gap must be positive or exact contact")
    g = _oracle_geometry(cell.data, settings)
    vertices = g["vertices0"] + Q(float(gap)) * g["vertices1"]
    velocity = g["vertices1"]
    size = len(g["pmap"])
    tagged_k, tagged_dk = (np.full((3, size, size), Q(0), object) for _ in range(2))
    tagged_f, tagged_df = (np.full((3, size, 2), Q(0), object) for _ in range(2))
    local = []
    for triangle, tag in zip(g["triangles"], g["tags"]):
        p, dp = vertices[triangle], velocity[triangle]
        v = np.column_stack([np.ones(3, dtype=int), p])
        dv = np.column_stack([np.zeros(3, dtype=int), dp])
        vi = _oracle_inverse(v)
        dvi = -vi @ dv @ vi
        grad, dgrad = vi[1:].T, dvi[1:].T
        area = (
            sum(
                p[i, 0] * p[(i + 1) % 3, 1] - p[(i + 1) % 3, 0] * p[i, 1]
                for i in range(3)
            )
            / 2
        )
        darea = (
            sum(
                dp[i, 0] * p[(i + 1) % 3, 1]
                + p[i, 0] * dp[(i + 1) % 3, 1]
                - dp[(i + 1) % 3, 0] * p[i, 1]
                - p[(i + 1) % 3, 0] * dp[i, 1]
                for i in range(3)
            )
            / 2
        )
        if area <= 0:
            raise ValueError("positive-gap nodal oracle needs positive areas")
        k = area * (grad @ grad.T)
        dk = darea * (grad @ grad.T) + area * (dgrad @ grad.T + grad @ dgrad.T)
        f, df = -area * grad, -darea * grad - area * dgrad
        ids = g["periodic"][triangle]
        for i in range(3):
            tagged_f[tag, ids[i]] += f[i]
            tagged_df[tag, ids[i]] += df[i]
            for j in range(3):
                tagged_k[tag, ids[i], ids[j]] += k[i, j]
                tagged_dk[tag, ids[i], ids[j]] += dk[i, j]
        local.append(
            (
                float(area),
                float(darea),
                np.asarray(grad, float),
                np.asarray(dgrad, float),
            )
        )
    materials = np.r_[settings.get("background", 1.0), settings["epsilon"]].astype(
        complex
    )
    k, dk, f, df = (
        np.tensordot(materials, np.asarray(v, float), axes=(0, 0))
        for v in (tagged_k, tagged_dk, tagged_f, tagged_df)
    )
    w = np.vstack([np.zeros((1, 2)), np.linalg.solve(k[1:, 1:], f[1:])])
    dw = np.vstack([np.zeros((1, 2)), np.linalg.solve(k[1:, 1:], (df - dk @ w)[1:])])
    residuals = [
        np.linalg.norm(k @ w - f) / max(np.linalg.norm(f), 1.0),
        np.linalg.norm(k @ dw + dk @ w - df) / max(np.linalg.norm(df), 1.0),
    ]
    q, weights = leggauss(quadrature)
    q, weights = (q + 1) / 2, weights / 2
    r, s = np.meshgrid(q, q, indexing="ij")
    barycentric = np.stack([1 - r, r * (1 - s), r * s], axis=-1).reshape(-1, 3)
    weights = (weights[:, None] * weights[None, :] * r).ravel()
    basis = np.asarray(settings["basis"], float)
    cell_area = np.linalg.det(basis)
    ox, oy = cell.order
    n, m = np.meshgrid(
        np.arange(-2 * oy, 2 * oy + 1), np.arange(-2 * ox, 2 * ox + 1), indexing="ij"
    )
    wave = 2 * np.pi * np.column_stack([m.ravel(), n.ravel()]) @ np.linalg.inv(basis)
    value, derivative = (
        np.zeros((9, len(wave)), complex),
        np.zeros((9, len(wave)), complex),
    )
    fields, changes, transforms, dtransforms = [], [], [], []
    for triangle, tag, (area, darea, grad, dgrad) in zip(
        g["triangles"], g["tags"], local
    ):
        ids = g["periodic"][triangle]
        b = np.eye(2) + grad.T @ w[ids]
        db = dgrad.T @ w[ids] + grad.T @ dw[ids]
        e = materials[tag]
        gram = e * b.conj().T @ b
        dgram = e * (db.conj().T @ b + b.conj().T @ db)
        components = np.array(
            [*b.ravel(), gram[0, 0], gram[0, 1], gram[1, 1], e, gram[1, 0]]
        )
        dcomponents = np.array(
            [*db.ravel(), dgram[0, 0], dgram[0, 1], dgram[1, 1], 0j, dgram[1, 0]]
        )
        points = barycentric @ np.asarray(vertices[triangle], float)
        speed = barycentric @ np.asarray(velocity[triangle], float)
        phase = np.exp(-1j * points @ wave.T)
        moment = weights @ phase
        dmoment = weights @ (-1j * (speed @ wave.T) * phase)
        ft = 2 * area / cell_area * moment
        dft = 2 / cell_area * (darea * moment + area * dmoment)
        value += components[:, None] * ft
        derivative += dcomponents[:, None] * ft + components[:, None] * dft
        fields.append(b)
        changes.append(db)
        transforms.append(ft)
        dtransforms.append(dft)
    return (
        value.reshape(9, 4 * oy + 1, 4 * ox + 1),
        derivative.reshape(9, 4 * oy + 1, 4 * ox + 1),
        dict(
            residuals=residuals,
            original_condition=np.linalg.cond(k[1:, 1:]),
            fields=np.array(fields),
            regular_dfields=np.array(changes),
            weights=np.array(transforms),
            dweights_per_unit=np.array(dtransforms),
            original_K=k,
            original_dK=dk,
            original_F=f,
            original_dF=df,
        ),
    )


def analytic_material(settings, order, gap):
    n, m = np.meshgrid(
        np.arange(-2 * order[1], 2 * order[1] + 1),
        np.arange(-2 * order[0], 2 * order[0] + 1),
        indexing="ij",
    )
    wave = 2 * np.pi * np.stack([m, n], axis=-1) @ np.linalg.inv(settings["basis"])
    area = np.linalg.det(settings["basis"])
    center = np.sum(settings["basis"], axis=1) / 2 + settings["offset"]
    w1, w2 = settings["widths"]
    height = settings["height"]
    transforms = []
    for width, dx in ((w1, -(w2 + gap) / 2), (w2, (w1 + gap) / 2)):
        point = center + np.array([dx, 0.0])
        transforms.append(
            width
            * height
            / area
            * np.exp(-1j * (wave @ point))
            * np.sinc(wave[..., 0] * width / (2 * np.pi))
            * np.sinc(wave[..., 1] * height / (2 * np.pi))
        )
    left, right = transforms
    bg = settings["background"]
    e1, e2 = settings["epsilon"]
    return bg * ((m == 0) & (n == 0)) + (e1 - bg) * left + (
        e2 - bg
    ) * right, 0.5j * wave[..., 0] * ((e1 - bg) * left - (e2 - bg) * right)


def native(settings, cell, gap):
    return CartesianPairCell(
        CartesianPairPlan(gap == 0.0, cell.plan.shift, cell.plan.refinement), cell.order
    )


def _qualification(case, s):
    criteria = CRITERIA
    settings = settings_for(criteria, case)
    plan = CartesianPairOpeningPlan.from_geometry(
        **{k: v for k, v in settings.items() if k not in ("epsilon", "background")},
        refinement=case["refinement"],
        shift=case["shift"],
    )
    cell = CartesianPairGapCell(plan, case["order"])
    length = np.sqrt(np.linalg.det(settings["basis"]))
    gap = length * s
    value, derivative, meta, detail = cell.coefficients(gap=gap, **settings)
    q16, dq16, r16 = _reference(cell, settings, gap, 16)
    q24, dq24, r24 = _reference(cell, settings, gap, 24)
    slots, dslots, cmeta, _ = cell.closure(gap=gap, **settings)
    slots, dslots = np.stack(slots), np.stack(dslots)
    rs, rds = _oracle_differentiated_native_solve_reference(q24, dq24, cell.order)
    nv, nm = native(settings, cell, gap).coefficients(gap=gap, **settings)
    ns, nsm = native(settings, cell, gap).conv_matrices(gap=gap, **settings)
    ns = np.stack(ns)
    analytical, danalytical = analytic_material(settings, cell.order, gap)
    ox, oy = cell.order
    delta = np.zeros(value.shape[1:])
    delta[2 * oy, 2 * ox] = 1.0
    mask = ~cell.data["thin"] if s == 0 else np.ones(len(cell.data["thin"]), bool)
    errors = dict(
        q_value=maximum(q16 - q24),
        q_derivative=length * maximum(dq16 - dq24),
        value=maximum(value - q24),
        derivative=length * maximum(derivative - dq24),
        native_value=maximum(value - nv),
        native_closure_value=maximum(slots - ns),
        closure_value=maximum(slots - rs),
        closure_derivative=length * maximum(dslots - rds),
        material_value=maximum(value[7] - analytical),
        material_derivative=length * maximum(derivative[7] - danalytical),
        field_value=maximum(detail["fields"] - r24["fields"]),
        field_derivative=maximum(
            detail["field_derivative_ds"][mask] - length * r24["regular_dfields"][mask]
        ),
        weight_value=maximum(detail["weights"] - r24["weights"]),
        weight_derivative=maximum(
            detail["weight_derivative_ds"] - length * r24["dweights_per_unit"]
        ),
        partition_value=maximum(detail["weights"].sum(axis=0) - delta.ravel()),
        partition_derivative=maximum(detail["weight_derivative_ds"].sum(axis=0)),
        mean_B_value=maximum(value[:4, 2 * oy, 2 * ox] - [1, 0, 0, 1]),
        mean_B_derivative=length * maximum(derivative[:4, 2 * oy, 2 * ox]),
    )
    if s == 0:
        endpoint = CartesianPairOpeningCell(plan, cell.order)
        ev, ed, _, _ = endpoint.coefficients(**settings)
        es, eds, _, _ = endpoint.closure(**settings)
        errors.update(
            endpoint_value=maximum(value - ev),
            endpoint_derivative=length * maximum(derivative - ed),
            endpoint_closure_value=maximum(slots - np.stack(es)),
            endpoint_closure_derivative=length * maximum(dslots - np.stack(eds)),
        )
    homogeneous = dict(settings, epsilon=np.array([settings["background"]] * 2))
    hv, hd, _, _ = cell.coefficients(gap=gap, **homogeneous)
    expected = np.zeros_like(hv)
    bg = settings["background"]
    expected[:, 2 * oy, 2 * ox] = [1, 0, 0, 1, bg, 0, bg, bg, 0]
    errors.update(
        homogeneous_value=maximum(hv - expected),
        homogeneous_derivative=length * maximum(hd),
    )
    residuals = [
        meta[k]
        for k in (
            "linear_residual",
            "linear_derivative_residual",
            "original_flux_residual",
            "original_flux_derivative_residual",
        )
    ]
    residuals += (
        list(r16["residuals"])
        + list(r24["residuals"])
        + [nm["cell_residual"], nsm["cell_residual"]]
    )
    checks = dict(
        expected_chart=plan.shift
        == (case["shift"] if case["shift"] is not None else 1),
        value=max(v for k, v in errors.items() if "derivative" not in k) <= 3e-10,
        derivative=max(v for k, v in errors.items() if "derivative" in k) <= 3e-9,
        quadrature=max(errors["q_value"], errors["q_derivative"]) <= 3e-10,
        weak_residual=max(residuals) <= 1e-10,
        inverse_residual=max(cmeta["inverse_residuals"]) <= 1e-10,
        counts=(
            meta["open_free_dofs"],
            meta["contact_free_dofs"],
            meta["difference_dofs"],
        )
        == ((8, 6, 2) if plan.refinement == 0 else (35, 27, 8)),
        endpoint_triangular=bool(
            s != 0
            or np.all(
                detail["M"][: meta["contact_free_dofs"], meta["contact_free_dofs"] :]
                == 0
            )
        ),
    )
    assert all(checks.values()), dict(errors=errors, checks=checks)
    assert meta["derivative_units"] == "per caller length unit"
    assert cmeta["minimum_storage"] > 0
    assert cmeta["minimum_dissipation"] >= -1e-10
    return cell, settings, value, derivative, slots, dslots


@lru_cache(None)
def _qualified(name, s):
    return _qualification(CASES[name], s)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("s", GAPS)
def test_original_nodal_equations_and_moving_integrals(name, s):
    _qualified(name, s)


@pytest.mark.parametrize("gap", GAPS)
@pytest.mark.parametrize("refinement", (0, 1))
@pytest.mark.parametrize("power", (1, 2, 3))
def test_signed_basis_label_covariance(refinement, power, gap):
    base = _qualified(f"main_j0_r{refinement}", gap)
    answer = _qualified(f"main_j{power}_r{refinement}", gap)
    _, settings, value, derivative, slots, dslots = base
    cell, _, v, dv, s, ds = answer
    length = np.sqrt(np.linalg.det(settings["basis"]))
    ox, oy = cell.order
    n, m = np.meshgrid(
        np.arange(-2 * oy, 2 * oy + 1), np.arange(-2 * ox, 2 * ox + 1), indexing="ij"
    )
    old = np.stack([m, n], axis=-1) @ POWERS[power].T
    assert maximum(v - value[:, old[..., 1] + 4, old[..., 0] + 2]) <= 3e-10
    assert (
        length * maximum(dv - derivative[:, old[..., 1] + 4, old[..., 0] + 2]) <= 3e-9
    )
    modes = [(m, n) for n in range(-oy, oy + 1) for m in range(-ox, ox + 1)]
    lookup = {
        (m, n): i
        for i, (m, n) in enumerate((m, n) for n in range(-2, 3) for m in range(-1, 2))
    }
    indices = np.array(
        [lookup[tuple(np.asarray(label) @ POWERS[power].T)] for label in modes]
    )
    assert maximum(s - slots[:, :, indices[:, None], indices]) <= 3e-10
    assert length * maximum(ds - dslots[:, :, indices[:, None], indices]) <= 3e-9


@pytest.mark.parametrize("name", CRITERIA["unit_scaling"]["cases"])
@pytest.mark.parametrize("s", GAPS)
def test_common_units_include_physical_gap(name, s):
    cell, settings, value, derivative, slots, dslots = _qualified(name, s)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    alpha = 0.01
    changed = dict(
        settings,
        **{
            key: np.asarray(settings[key]) * alpha
            for key in ("basis", "widths", "height", "offset")
        },
    )
    cv, cd, _, _ = cell.coefficients(gap=alpha * length * s, **changed)
    cs, cds, _, _ = cell.closure(gap=alpha * length * s, **changed)
    assert maximum(cv - value) <= 3e-10
    assert length * maximum(alpha * cd - derivative) <= 3e-9
    assert maximum(np.stack(cs) - slots) <= 3e-10
    assert length * maximum(alpha * np.stack(cds) - dslots) <= 3e-9


@pytest.mark.parametrize("order", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("s", (0.0, 1 / 64))
def test_zero_direction_labels_before_closure(order, s):
    previous, settings, *_ = _qualified("main_j0_r0", 0.0)
    cell = CartesianPairGapCell(previous.plan, order)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    gap = length * s
    cv, cd, _, _ = cell.coefficients(gap=gap, **settings)
    q16, dq16, _ = _reference(cell, settings, gap, 16)
    q24, dq24, _ = _reference(cell, settings, gap, 24)
    cs, cds, _, _ = cell.closure(gap=gap, **settings)
    rs, rds = _oracle_differentiated_native_solve_reference(q24, dq24, order)
    assert maximum(q16 - q24) <= 3e-10
    assert length * maximum(dq16 - dq24) <= 3e-10
    assert maximum(cv - q24) <= 3e-10
    assert length * maximum(cd - dq24) <= 3e-9
    assert maximum(np.stack(cs) - rs) <= 3e-10
    assert length * maximum(np.stack(cds) - rds) <= 3e-9
    assert cv.shape == (9, 4 * order[1] + 1, 4 * order[0] + 1)
    size = (2 * order[0] + 1) * (2 * order[1] + 1)
    assert all(slot.shape == (1, size, size) for slot in cs)


@pytest.mark.parametrize("name", CRITERIA["unit_scaling"]["cases"])
def test_predetermined_taylor_remainder_verifies_analytic_derivative(name):
    cell, settings, value, derivative, slots, dslots = _qualified(name, 0.0)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    remainders, slot_remainders = [], []
    for s in (0.002, 0.001, 0.0005, 0.00025):
        cv, _, _, _ = cell.coefficients(gap=length * s, **settings)
        cs, _, _, _ = cell.closure(gap=length * s, **settings)
        remainders.append(
            np.max(np.abs(cv - value - length * s * derivative), axis=(1, 2))
        )
        slot_remainders.append(
            np.max(
                np.abs(np.stack(cs) - slots - length * s * dslots), axis=(1, 2, 3)
            )
        )
    remainders, slot_remainders = np.array(remainders), np.array(slot_remainders)
    assert np.all(remainders[1:] <= 0.35 * remainders[:-1] + 3e-10)
    assert np.all(slot_remainders[1:] <= 0.35 * slot_remainders[:-1] + 3e-10)


@pytest.mark.parametrize("index", range(9))
def test_invalid_gap_and_immutable_replay(index):
    cell, settings, *_ = _qualified("main_j0_r0", 0.0)
    actions = (
        lambda: cell.coefficients(gap=-1.0, **settings),
        lambda: cell.coefficients(gap=0j, **settings),
        lambda: cell.coefficients(gap=True, **settings),
        lambda: cell.coefficients(gap=np.nan, **settings),
        lambda: cell.coefficients(gap=1e6, **settings),
        lambda: setattr(cell, "plan", None),
        lambda: setattr(cell, "order", None),
        lambda: cell.data["mapping"].setflags(write=True),
        lambda: cell.closure(gap=0.0, coefficients=None, **settings),
    )
    with pytest.raises((ValueError, TypeError, AttributeError)):
        actions[index]()
