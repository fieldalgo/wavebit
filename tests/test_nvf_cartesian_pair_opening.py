"""Finite right-opening verification from original nodal Laurent equations.

The reference shares immutable discrete topology only. It independently forms
physical vertices, exact rational Vandermonde operators, moving-simplex Duffy
moments and implicit closure derivatives. Positive-gap samples verify the
analytic derivative; they do not define it. No JAX or optical solve is used.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from wavebit._core.conforming.cartesian_pair import CartesianPairCell, CartesianPairPlan
from wavebit._core.conforming.cartesian_pair_opening import (
    CartesianPairOpeningCell,
    CartesianPairOpeningPlan,
)

CRITERIA = {
    "main": {
        "basis": [[700, 315], [0, 490]],
        "widths": [140, 193],
        "height": 210,
        "offset": [49, -21],
        "epsilon": [[4, 0.8], [9, 0.3]],
        "background": [1, 0.1],
    },
    "different": {
        "basis": [[640, -160], [70, 560]],
        "widths": [160, 110],
        "height": 190,
        "offset": [31, -19],
        "epsilon": [[2.4, 0.17], [5.8, 0.46]],
        "background": [1.2, 0.08],
    },
    "cases": [
        {
            "name": "main_j0_r0",
            "fixture": "main",
            "basis_power": 0,
            "shift": 0,
            "refinement": 0,
            "order": [1, 2],
        },
        {
            "name": "main_j0_r1",
            "fixture": "main",
            "basis_power": 0,
            "shift": 0,
            "refinement": 1,
            "order": [1, 2],
        },
        {
            "name": "main_j1_r0",
            "fixture": "main",
            "basis_power": 1,
            "shift": 1,
            "refinement": 0,
            "order": [2, 1],
        },
        {
            "name": "main_j1_r1",
            "fixture": "main",
            "basis_power": 1,
            "shift": 1,
            "refinement": 1,
            "order": [2, 1],
        },
        {
            "name": "main_j2_r0",
            "fixture": "main",
            "basis_power": 2,
            "shift": 2,
            "refinement": 0,
            "order": [1, 2],
        },
        {
            "name": "main_j2_r1",
            "fixture": "main",
            "basis_power": 2,
            "shift": 2,
            "refinement": 1,
            "order": [1, 2],
        },
        {
            "name": "main_j3_r0",
            "fixture": "main",
            "basis_power": 3,
            "shift": 3,
            "refinement": 0,
            "order": [2, 1],
        },
        {
            "name": "main_j3_r1",
            "fixture": "main",
            "basis_power": 3,
            "shift": 3,
            "refinement": 1,
            "order": [2, 1],
        },
        {
            "name": "different_r0",
            "fixture": "different",
            "basis_power": 0,
            "shift": None,
            "refinement": 0,
            "order": [2, 1],
        },
        {
            "name": "different_r1",
            "fixture": "different",
            "basis_power": 0,
            "shift": None,
            "refinement": 1,
            "order": [2, 1],
        },
    ],
    "unit_scaling": {
        "alpha": 0.01,
        "cases": ["main_j0_r0", "main_j0_r1", "different_r0", "different_r1"],
        "expected": "A,widths,height,offset scale byalpha; c0/slots "
        "unchanged, dc/dg and dslots/dg scale by1/alpha",
    },
    "extra_retained_sets": {
        "case": "main_j0_r0",
        "orders": [[0, 0], [1, 0], [0, 1]],
        "reference": "independent q24 and guarded native exact-contact closure",
    },
    "taylor_verification_only": {
        "s": [0.002, 0.001, 0.0005, 0.00025],
        "physical_gap": "g=L*s",
        "per_component_and_slot_guard": "E_next<=.35*E_previous+3e-10; "
        "analytic "
        "derivative "
        "constructed "
        "first",
    },
}
CASES = {case["name"]: case for case in CRITERIA["cases"]}
POWERS = [
    np.array(p, int)
    for p in (
        [[1, 0], [0, 1]],
        [[0, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
    )
]


def _oracle_geometry(data, settings):
    def rational(value):
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(
                "_oracle_reference _oracle_geometry must be finite and real"
            )
        return np.array([Q(v) for v in array.ravel().tolist()], object).reshape(
            array.shape
        )

    a, offset = (
        rational(settings["basis"]),
        rational(settings.get("offset", (0.0, 0.0))),
    )
    w1, w2 = rational(settings["widths"])
    height = rational(settings["height"]).item()
    center = a.sum(axis=1) / 2 + offset
    x = [-(w1 + w2) / 2, (w1 - w2) / 2, (w1 + w2) / 2]
    inner = (
        np.array(
            [[v, -height / 2] for v in x] + [[v, height / 2] for v in x[::-1]], object
        )
        + center
    )
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], object)
    controls = np.vstack([unit @ a.T + offset, inner])
    dcontrols = np.full((12, 2), Q(0), object)
    dcontrols[4:, 0] = [
        Q(-1, 2),
        Q(-1, 2),
        Q(1, 2),
        Q(1, 2),
        Q(1, 2),
        Q(1, 2),
        Q(-1, 2),
        Q(-1, 2),
    ]
    den = data["weight_denominator"]

    def weights(values):
        return np.array([Q(int(v), den) for v in values.ravel()], object).reshape(
            values.shape
        )

    contact = weights(data["contact_numerators"]) @ controls
    result = dict(data)
    result.update(
        vertices0=contact[data["mapping"]],
        vertices1=weights(data["open_numerators"]) @ dcontrols,
        contact_vertices=contact,
    )
    return result


def _oracle_polynomial_vandermonde(points, velocity):
    v = np.full((3, 3, 2), Q(0), object)
    v[:, 0, 0] = Q(1)
    v[:, 1:, 0], v[:, 1:, 1] = points, velocity
    adj = np.full((3, 3, 3), Q(0), object)
    for row in range(3):
        for column in range(3):
            minor = v[[i for i in range(3) if i != row]][
                :, [i for i in range(3) if i != column]
            ]
            adj[column, row] = (-1) ** (row + column) * (
                np.convolve(minor[0, 0], minor[1, 1])
                - np.convolve(minor[0, 1], minor[1, 0])
            )
    determinant = sum(
        (np.convolve(v[0, j], adj[j, 0]) for j in range(3)), np.full(4, Q(0), object)
    )
    assert np.all(determinant[2:] == 0)
    numerator = adj[1:, :].transpose(2, 1, 0)
    assert np.all(numerator[2] == 0)
    return determinant[:2], numerator[:2]


def _oracle_original_limit(g, materials):
    count = len(g["pmap"])
    kt = np.full((3, 3, count, count), Q(0), object)
    ft = np.full((3, 2, count, 2), Q(0), object)
    gradients = []
    for triangle, tag in zip(g["triangles"], g["tags"]):
        (d0, d1), (p0, p1) = _oracle_polynomial_vandermonde(
            g["vertices0"][triangle], g["vertices1"][triangle]
        )
        assert d0 > 0 or (d0 == 0 and d1 > 0)
        if d0:
            grad0 = p0 / d0
            grad1 = p1 / d0 - p0 * d1 / d0**2
            kminus = np.full((3, 3), Q(0), object)
            k0 = d0 / 2 * (grad0 @ grad0.T)
            k1 = d1 / 2 * (grad0 @ grad0.T) + d0 / 2 * (
                grad1 @ grad0.T + grad0 @ grad1.T
            )
            f0, f1 = -d0 / 2 * grad0, -d1 / 2 * grad0 - d0 / 2 * grad1
        else:
            grad0, grad1 = p0 / d1, p1 / d1  # Coefficients of g^-1 and g^0.
            kminus = d1 / 2 * (grad0 @ grad0.T)
            k0 = d1 / 2 * (grad0 @ grad1.T + grad1 @ grad0.T)
            k1 = d1 / 2 * (grad1 @ grad1.T)
            f0, f1 = -d1 / 2 * grad0, -d1 / 2 * grad1
        ids = g["periodic"][triangle]
        for i in range(3):
            ft[tag, 0, ids[i]] += f0[i]
            ft[tag, 1, ids[i]] += f1[i]
            for j in range(3):
                kt[tag, :, ids[i], ids[j]] += np.array(
                    [kminus[i, j], k0[i, j], k1[i, j]]
                )
        gradients.append(
            (float(d0), float(d1), np.asarray(grad0, float), np.asarray(grad1, float))
        )
    k, f = (
        np.tensordot(materials, np.asarray(v, float), axes=(0, 0)) for v in (kt, ft)
    )
    c, z = np.asarray(g["C"], float), np.asarray(g["Z"], float)
    km, k0, k1 = k[:, 1:, 1:]
    f0, f1 = f[:, 1:]
    contact = c.T @ k0 @ c
    differences = z.T @ km @ z
    u = np.linalg.solve(contact, c.T @ f0)
    v = np.linalg.solve(differences, z.T @ (f0 - k0 @ c @ u))
    du = np.linalg.solve(contact, c.T @ (f1 - k1 @ c @ u - k0 @ z @ v))
    w0 = np.vstack([np.zeros((1, 2)), c @ u])
    w1 = np.vstack([np.zeros((1, 2)), c @ du + z @ v])
    remainder = k[0] @ w1 + k[1] @ w0 - f[0]
    projected_derivative = g["Cfull"].T @ (k[1] @ w1 + k[2] @ w0 - f[1])
    residuals = [
        np.linalg.norm(remainder) / max(np.linalg.norm(f[0]), 1.0),
        np.linalg.norm(projected_derivative)
        / max(np.linalg.norm(g["Cfull"].T @ f[1]), 1.0),
    ]
    return w0, w1, gradients, kt, ft, np.asarray(residuals)


def _oracle_reference(cell, settings, quadrature):
    g = _oracle_geometry(cell.data, settings)
    materials = np.r_[settings.get("background", 1.0), settings["epsilon"]].astype(
        complex
    )
    w0, w1, gradients, kt, ft, residuals = _oracle_original_limit(g, materials)
    q, w = leggauss(quadrature)
    q, w = (q + 1) / 2, w / 2
    r, s = np.meshgrid(q, q, indexing="ij")
    weights = (w[:, None] * w[None, :] * r).ravel()
    barycentric = np.stack([1 - r, r * (1 - s), r * s], axis=-1).reshape(-1, 3)
    basis = np.asarray(settings["basis"], float)
    cell_area = np.linalg.det(basis)
    ox, oy = cell.order
    n, m = np.meshgrid(
        np.arange(-2 * oy, 2 * oy + 1), np.arange(-2 * ox, 2 * ox + 1), indexing="ij"
    )
    wave = 2 * np.pi * np.column_stack([m.ravel(), n.ravel()]) @ np.linalg.inv(basis)
    coeff = np.zeros((9, len(wave)), complex)
    derivative = np.zeros_like(coeff)
    all_weights, all_dweights, fields, dfields = [], [], [], []
    for triangle, tag, (d0, d1, grad0, grad1) in zip(
        g["triangles"], g["tags"], gradients
    ):
        epsilon = materials[tag]
        ids = g["periodic"][triangle]
        if d0:
            field = np.eye(2) + grad0.T @ w0[ids]
            dfield = grad1.T @ w0[ids] + grad0.T @ w1[ids]
        else:
            field = np.eye(2) + grad1.T @ w0[ids] + grad0.T @ w1[ids]
            dfield = np.zeros((2, 2), complex)  # Only W1 times field contributes here.
        gram = epsilon * field.conj().T @ field
        dgram = epsilon * (dfield.conj().T @ field + field.conj().T @ dfield)
        values = np.array(
            [*field.ravel(), gram[0, 0], gram[0, 1], gram[1, 1], epsilon, gram[1, 0]]
        )
        changes = np.array(
            [*dfield.ravel(), dgram[0, 0], dgram[0, 1], dgram[1, 1], 0j, dgram[1, 0]]
        )
        points = barycentric @ np.asarray(g["vertices0"][triangle], float)
        velocity = barycentric @ np.asarray(g["vertices1"][triangle], float)
        phase = np.exp(-1j * points @ wave.T)
        moment = weights @ phase
        moment_motion = weights @ (-1j * (velocity @ wave.T) * phase)
        transform = d0 / cell_area * moment
        change = (d1 * moment + d0 * moment_motion) / cell_area
        coeff += values[:, None] * transform
        derivative += changes[:, None] * transform + values[:, None] * change
        all_weights.append(transform)
        all_dweights.append(change)
        fields.append(field)
        dfields.append(dfield)
    return (
        coeff.reshape(9, 4 * oy + 1, 4 * ox + 1),
        derivative.reshape(9, 4 * oy + 1, 4 * ox + 1),
        dict(
            weights=np.asarray(all_weights),
            dweights_per_unit=np.asarray(all_dweights),
            fields=np.asarray(fields),
            regular_dfields=np.asarray(dfields),
            exact_tagged_K=kt,
            exact_tagged_F=ft,
            residuals=residuals,
        ),
    )


def _oracle_differentiated_native_solve_reference(coefficients, derivative, order):
    """Independent integer-label gather and separate implicit linear equations."""
    modes = [
        (m, n)
        for n in range(-order[1], order[1] + 1)
        for m in range(-order[0], order[0] + 1)
    ]
    lookup = {
        (m, n): (
            coefficients[:, n + 2 * order[1], m + 2 * order[0]],
            derivative[:, n + 2 * order[1], m + 2 * order[0]],
        )
        for n in range(-2 * order[1], 2 * order[1] + 1)
        for m in range(-2 * order[0], 2 * order[0] + 1)
    }
    c = np.empty((9, len(modes), len(modes)), complex)
    dc = np.empty_like(c)
    for i, (mi, ni) in enumerate(modes):
        for j, (mj, nj) in enumerate(modes):
            c[:, i, j], dc[:, i, j] = lookup[(mi - mj, ni - nj)]
    size = len(modes)
    q, dq, g, dg = (np.empty((2 * size, 2 * size), complex) for _ in range(4))
    for row, column, b, energy in (
        (0, 0, 0, 4),
        (0, 1, 1, 5),
        (1, 0, 2, 8),
        (1, 1, 3, 6),
    ):
        selection = (
            slice(row * size, (row + 1) * size),
            slice(column * size, (column + 1) * size),
        )
        q[selection], dq[selection] = c[b], dc[b]
        g[selection], dg[selection] = c[energy], dc[energy]
    left = np.linalg.solve(q.conj().T, g)
    h = np.linalg.solve(q.T, left.T).T
    dleft = np.linalg.solve(q.conj().T, dg - dq.conj().T @ left)
    dh = np.linalg.solve(q.T, (dleft - h @ dq).T).T
    z = np.linalg.solve(c[7], np.eye(size))
    dz = np.linalg.solve(c[7], -dc[7] @ z)
    result, dresult = (
        np.empty((5, 1, size, size), complex),
        np.empty((5, 1, size, size), complex),
    )
    for dest, hmat, zmat in ((result, h, z), (dresult, dh, dz)):
        dest[:, 0] = [
            hmat[:size, :size],
            hmat[size:, size:],
            zmat,
            -hmat[:size, size:],
            -hmat[size:, :size],
        ]
    return result, dresult


def _oracle_inverse(matrix):
    n = len(matrix)
    a = np.concatenate(
        [np.asarray(matrix, object), np.eye(n, dtype=int).astype(object)], axis=1
    )
    for column in range(n):
        pivot = next(i for i in range(column, n) if a[i, column] != 0)
        a[[column, pivot]] = a[[pivot, column]]
        scale = a[column, column]
        a[column] = np.array([Q(v) / scale for v in a[column]], object)
        for i in range(n):
            if i != column:
                a[i] -= a[i, column] * a[column]
    assert np.array_equal(np.asarray(matrix, object) @ a[:, n:], np.eye(n))
    return a[:, n:]


def _oracle_assemble(vertices, triangles, tags, periodic):
    """Per-material K/F on all periodic DOFs, including the fixed gauge row."""
    size = int(max(periodic)) + 1
    stiffness = np.full((3, size, size), Q(0), object)
    load = np.full((3, size, 2), Q(0), object)
    for triangle, tag in zip(triangles, tags):
        points = vertices[triangle]
        twice = sum(
            points[i, 0] * points[(i + 1) % 3, 1]
            - points[(i + 1) % 3, 0] * points[i, 1]
            for i in range(3)
        )
        if twice <= 0:
            raise ValueError(
                "independent nodal oracle requires strictly positive triangles"
            )
        coefficients = _oracle_inverse(np.column_stack([np.ones(3, dtype=int), points]))
        gradient = coefficients[1:, :].T
        k = (twice / 2) * (gradient @ gradient.T)
        f = -(twice / 2) * gradient
        ids = periodic[triangle]
        for i in range(3):
            load[tag, ids[i]] += f[i]
            for j in range(3):
                stiffness[tag, ids[i], ids[j]] += k[i, j]
    return stiffness, load


def maximum(value):
    return float(np.max(np.abs(value)))


def normalized_difference(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0))


def settings_for(criteria, case):
    fixture = criteria[case["fixture"]]
    return dict(
        basis=np.asarray(fixture["basis"], float) @ POWERS[case["basis_power"]],
        widths=np.asarray(fixture["widths"], float),
        height=float(fixture["height"]),
        offset=np.asarray(fixture["offset"], float),
        epsilon=np.array([complex(*v) for v in fixture["epsilon"]]),
        background=complex(*fixture["background"]),
    )


def analytic_material(settings, order):
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
    for width, dx in ((w1, -w2 / 2), (w2, w1 / 2)):
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
    return (
        bg * ((m == 0) & (n == 0)) + (e1 - bg) * left + (e2 - bg) * right,
        0.5j * wave[..., 0] * ((e1 - bg) * left - (e2 - bg) * right),
    )


def selected_native(settings, plan, order, gap=0.0):
    native = CartesianPairCell(
        CartesianPairPlan(gap == 0.0, plan.shift, plan.refinement), order
    )
    slots, metadata = native.conv_matrices(gap=gap, **settings)
    return np.stack(slots), metadata


def _qualification(criteria, case):
    settings = settings_for(criteria, case)
    geometry = {k: v for k, v in settings.items() if k not in ("epsilon", "background")}
    plan = CartesianPairOpeningPlan.from_geometry(
        **geometry, refinement=case["refinement"], shift=case["shift"]
    )
    cell = CartesianPairOpeningCell(plan, case["order"])
    L = float(np.sqrt(np.linalg.det(settings["basis"])))
    value, derivative, meta, detail = cell.coefficients(**settings)
    q16, dq16, r16 = _oracle_reference(cell, settings, 16)
    q24, dq24, r24 = _oracle_reference(cell, settings, 24)
    g = _oracle_geometry(cell.data, settings)
    materials = np.r_[settings["background"], settings["epsilon"]]
    rk, rf = (
        np.tensordot(materials, np.asarray(v, float), axes=(0, 0))
        for v in (r24["exact_tagged_K"], r24["exact_tagged_F"])
    )
    ck, cf = _oracle_assemble(
        g["contact_vertices"],
        g["contact_triangles"],
        g["contact_tags"],
        g["contact_periodic"],
    )
    ck, cf = (
        np.tensordot(materials, np.asarray(v, float), axes=(0, 0)) for v in (ck, cf)
    )
    C = cell.data["Cfull"]
    native = CartesianPairCell(
        CartesianPairPlan(True, plan.shift, plan.refinement), cell.order
    )
    contact, contact_meta = native.coefficients(gap=0.0, **settings)
    slots, dslots, closure_meta, _ = cell.closure(**settings)
    slots, dslots = np.stack(slots), np.stack(dslots)
    reference, dreference = _oracle_differentiated_native_solve_reference(
        q24, dq24, cell.order
    )
    native_slots, nmeta = selected_native(settings, plan, cell.order)
    analytical, danalytical = analytic_material(settings, cell.order)
    ox, oy = cell.order
    thin = cell.data["thin"]
    delta = np.zeros(value.shape[1:])
    delta[2 * oy, 2 * ox] = 1.0
    errors = dict(
        q_value=maximum(q16 - q24),
        q_derivative=L * maximum(dq16 - dq24),
        value=maximum(value - q24),
        derivative=L * maximum(derivative - dq24),
        contact_value=maximum(value - contact),
        material_value=maximum(value[7] - analytical),
        material_derivative=L * maximum(derivative[7] - danalytical),
        field_value=maximum(detail["fields"] - r24["fields"]),
        regular_field_derivative=L
        * maximum(
            detail["regular_field_derivative"][~thin] - r24["regular_dfields"][~thin]
        ),
        phi2_derivative=L * maximum(detail["phi2_weights"] - detail["dweights"][thin]),
        phi2_reference_derivative=L
        * maximum(detail["phi2_weights"] - r24["dweights_per_unit"][thin]),
        partition_value=maximum(detail["weights"].sum(axis=0) - delta.ravel()),
        partition_derivative=L * maximum(detail["dweights"].sum(axis=0)),
        meanB_value=maximum(value[:4, 2 * oy, 2 * ox] - [1, 0, 0, 1]),
        meanB_derivative=L * maximum(derivative[:4, 2 * oy, 2 * ox]),
        closure_value=maximum(slots - reference),
        closure_derivative=L * maximum(dslots - dreference),
        native_closure_value=maximum(slots - native_slots),
    )
    operator_errors = {
        f"Laurent_K_{power}": normalized_difference(detail["K"][i], rk[i])
        for i, power in enumerate((-1, 0, 1))
    }
    operator_errors.update(
        {f"Laurent_F_{i}": normalized_difference(detail["F"][i], rf[i]) for i in (0, 1)}
    )
    operator_errors.update(
        contact_K=normalized_difference(C.T @ detail["K"][1] @ C, ck),
        contact_F=normalized_difference(C.T @ detail["F"][0], cf),
    )
    homogeneous = dict(settings, epsilon=np.array([settings["background"]] * 2))
    hv, hd, hm, _ = cell.coefficients(**homogeneous)
    expected = np.zeros_like(hv)
    bg = settings["background"]
    expected[:, 2 * oy, 2 * ox] = [1, 0, 0, 1, bg, 0, bg, bg, 0]
    errors.update(
        homogeneous_value=maximum(hv - expected), homogeneous_derivative=L * maximum(hd)
    )
    remainder, slot_remainder, positive, native_weak = (
        [],
        [],
        [],
        [float(contact_meta["cell_residual"])],
    )
    for s in criteria["taylor_verification_only"]["s"]:
        pc, pm = CartesianPairCell(
            CartesianPairPlan(False, plan.shift, plan.refinement), cell.order
        ).coefficients(gap=L * s, **settings)
        ps, psm = selected_native(settings, plan, cell.order, gap=L * s)
        remainder.append(np.max(np.abs(pc - value - L * s * derivative), axis=(1, 2)))
        slot_remainder.append(
            np.max(np.abs(ps - slots - L * s * dslots), axis=(1, 2, 3))
        )
        positive.append(pc)
        native_weak.extend([float(pm["cell_residual"]), float(psm["cell_residual"])])
    remainder, slot_remainder = np.array(remainder), np.array(slot_remainder)
    refinement = plan.refinement
    checks = dict(
        q_guard=max(errors["q_value"], errors["q_derivative"]) <= 3e-10,
        value_guards=max(v for k, v in errors.items() if "derivative" not in k)
        <= 3e-10,
        derivative_guards=max(v for k, v in errors.items() if "derivative" in k)
        <= 3e-9,
        operator_guards=max(operator_errors.values()) <= 3e-12,
        original_residuals=max(
            *meta["original_residuals"], *r24["residuals"], *native_weak
        )
        <= 1e-10,
        inverse_residuals=max(closure_meta["inverse_residuals"]) <= 1e-10,
        expected_counts=(
            meta["open_free_dofs"],
            meta["contact_free_dofs"],
            meta["difference_dofs"],
            meta["collapsed_triangles"],
        )
        == ((8, 6, 2, 4) if refinement == 0 else (35, 27, 8, 16)),
        coefficient_Taylor=bool(np.all(remainder[1:] <= 0.35 * remainder[:-1] + 3e-10)),
        closure_Taylor=bool(
            np.all(slot_remainder[1:] <= 0.35 * slot_remainder[:-1] + 3e-10)
        ),
    )
    metadata = dict(
        case=case,
        actual_shift=plan.shift,
        comparison_length=L,
        errors=errors,
        operator_errors=operator_errors,
        checks=checks,
        cell=meta,
        closure=closure_meta,
        native_weak_residuals=native_weak,
        coefficient_remainders=remainder,
        slot_remainders=slot_remainder,
    )
    return metadata, (cell, settings, value, derivative, slots, dslots)


@lru_cache(None)
def _qualified(name):
    return _qualification(CRITERIA, CASES[name])


@pytest.mark.research
@pytest.mark.parametrize("name", CASES)
def test_independent_finite_opening(name):
    metadata, _ = _qualified(name)
    assert all(metadata["checks"].values()), metadata
    if name.startswith("different"):
        assert metadata["actual_shift"] == 1


@pytest.mark.research
@pytest.mark.parametrize("refinement", (0, 1))
@pytest.mark.parametrize("power", (1, 2, 3))
def test_signed_basis_label_covariance(refinement, power):
    _, base = _qualified(f"main_j0_r{refinement}")
    _, answer = _qualified(f"main_j{power}_r{refinement}")
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


@pytest.mark.research
@pytest.mark.parametrize("name", CRITERIA["unit_scaling"]["cases"])
def test_derivative_per_caller_unit(name):
    _, (cell, settings, value, derivative, slots, dslots) = _qualified(name)
    alpha = CRITERIA["unit_scaling"]["alpha"]
    changed = dict(
        settings,
        **{
            k: np.asarray(settings[k]) * alpha
            for k in ("basis", "widths", "height", "offset")
        },
    )
    cv, cd, _, _ = cell.coefficients(**changed)
    cs, cds, _, _ = cell.closure(**changed)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    assert maximum(cv - value) <= 3e-10
    assert length * maximum(alpha * cd - derivative) <= 3e-9
    assert maximum(np.stack(cs) - slots) <= 3e-10
    assert length * maximum(alpha * np.stack(cds) - dslots) <= 3e-9


@pytest.mark.research
@pytest.mark.parametrize("order", [(0, 0), (1, 0), (0, 1)])
def test_zero_direction_orders_before_closure(order):
    _, (previous, settings, *_) = _qualified("main_j0_r0")
    cell = CartesianPairOpeningCell(previous.plan, order)
    cv, cd, _, _ = cell.coefficients(**settings)
    rv, rd, _ = _oracle_reference(cell, settings, 24)
    cs, cds, _, _ = cell.closure(**settings)
    rs, rds = _oracle_differentiated_native_solve_reference(rv, rd, order)
    native, _ = selected_native(settings, cell.plan, order)
    length = np.sqrt(np.linalg.det(settings["basis"]))
    assert maximum(cv - rv) <= 3e-10
    assert length * maximum(cd - rd) <= 3e-9
    assert maximum(np.stack(cs) - rs) <= 3e-10
    assert length * maximum(np.stack(cds) - rds) <= 3e-9
    assert maximum(np.stack(cs) - native) <= 3e-10


# The exported plan's own contract needs no qualified cell, so it is the part of
# this file that stays in the default set; the cell's cases below are research.
@pytest.mark.parametrize("index", range(3))
def test_plan_invalid_and_immutable(index):
    plan = CartesianPairOpeningPlan(0)
    actions = (
        lambda: setattr(plan, "shift", 2),
        lambda: CartesianPairOpeningPlan(True, 0),
        lambda: CartesianPairOpeningPlan(0, 4),
    )
    with pytest.raises((ValueError, FrozenInstanceError)):
        actions[index]()
    assert (plan.shift, plan.refinement) == (0, 0)


@pytest.mark.research
@pytest.mark.parametrize("index", range(11))
def test_invalid_and_immutable(index):
    _, (cell, settings, *_) = _qualified("main_j0_r0")
    actions = (
        lambda: cell.data.__setitem__("extras", ()),
        lambda: cell.data["mapping"].__setitem__(0, 3),
        lambda: cell.data["mapping"].setflags(write=True),
        lambda: cell.closure(coefficients=(None,) * 4, **settings),
        lambda: CartesianPairOpeningCell(cell.plan, True),
        lambda: cell.coefficients(
            **dict(settings, basis=np.asarray(settings["basis"], complex))
        ),
        lambda: cell.coefficients(**dict(settings, widths=[0, 110])),
        lambda: cell.coefficients(**dict(settings, height=190 + 0j)),
        lambda: cell.coefficients(**dict(settings, basis=np.zeros((2, 2)))),
        lambda: cell.coefficients(**dict(settings, epsilon=[-1, 3])),
        lambda: cell.coefficients(**dict(settings, background=1 - 0.1j)),
    )
    with pytest.raises((ValueError, TypeError, AttributeError, FrozenInstanceError)):
        actions[index]()
    assert isinstance(cell.data["groups"], tuple)
    assert all(isinstance(group, tuple) for group in cell.data["groups"])
