"""Reusable NumPy finite right-opening response on explicit refined pair charts.

Lengths use the caller's common physical unit. Returned derivatives are per
unit gap, with fixed basis, widths, shared height, offset and material values.
Only the immutable topology comes from the qualified pair cell. The analytic
right-opening response is NumPy-only; it does not change ordinary fixed-contact
replay or JAX tangents. See docs/nvf-cartesian-pair-opening.md for the finite
weak limit and independent qualification.
"""

from dataclasses import dataclass
from numbers import Integral
from collections import Counter
from types import MappingProxyType
import numpy as np
from .cartesian_pair import CartesianPairPlan
from ..factorization import ConvMatrices
from ..polygon import polygon_ft
from .material_sector import (
    passive_pair_materials, phase_from_angle, minimum_rotated_storage,
)


COMPONENTS = ("Bxx", "Bxy", "Byx", "Byy", "Gxx", "Gxy", "Gyy", "epsilon", "Gyx")


def real_array(value, shape, name):
    result = np.asarray(value)
    if (
        result.shape != shape
        or result.dtype.kind not in "iuf"
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite real numeric data with shape {shape}")
    return result.astype(np.float64)


def materials_array(epsilon, background, phase=1.0):
    return passive_pair_materials(epsilon, background, phase)[0]


def validate_geometry(basis, widths, height, offset):
    a = real_array(basis, (2, 2), "basis")
    w = real_array(widths, (2,), "widths")
    h = real_array(height, (), "height")
    o = real_array(offset, (2,), "offset")
    if (
        np.any(w <= 0)
        or h <= 0
        or not np.isfinite(np.linalg.det(a))
        or np.linalg.det(a) <= 0
    ):
        raise ValueError(
            "positive widths, height and finite positive determinant are required"
        )
    return dict(basis=a, widths=w, height=h, offset=o, gap=0.0)


@dataclass(frozen=True)
class CartesianPairOpeningPlan:
    """Contact/open topologies share a concrete cyclic index and refinement."""

    shift: int
    refinement: int = 0

    def __post_init__(self):
        CartesianPairPlan(True, self.shift, self.refinement)

    @classmethod
    def from_geometry(
        cls, *, basis, widths, height, offset=(0.0, 0.0), refinement=0, shift=None
    ):
        settings = validate_geometry(basis, widths, height, offset)
        if shift is None:
            native = CartesianPairPlan.from_geometry(refinement=refinement, **settings)
        else:
            native = CartesianPairPlan(True, shift, refinement)
            native.replay(**settings)
        return cls(native.shift, native.refinement)


def canonical(triangle, tag):
    v = tuple(map(int, triangle))
    return min(v, v[1:] + v[:1], v[2:] + v[:2]), int(tag)


def topology(plan):
    ow, ot, om, op = CartesianPairPlan(False, plan.shift, plan.refinement).topology()
    cw, ct, cm, cp = CartesianPairPlan(True, plan.shift, plan.refinement).topology()
    denominator = 2**plan.refinement
    on, cn = ow * denominator, cw * denominator
    if not np.array_equal(on, np.round(on)) or not np.array_equal(cn, np.round(cn)):
        raise ValueError("topology weights are not exactly dyadic at this refinement")
    on, cn = on.astype(np.int64), cn.astype(np.int64)
    collapse = np.eye(10, dtype=np.int64)[[0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9]]
    folded = on @ collapse
    lookup = {tuple(row): i for i, row in enumerate(cn)}
    mapping = np.array([lookup[tuple(row)] for row in folded])
    if not np.array_equal(folded, cn[mapping]):
        raise ValueError("dyadic collapse is inconsistent")
    mapped = mapping[ot]
    multiplicities = np.array([len(set(v)) for v in mapped])
    if not np.isin(multiplicities, [2, 3]).all():
        raise ValueError("unsupported collapsed child topology")
    thin = multiplicities == 2
    if Counter(
        canonical(v, tag) for v, tag in zip(mapped[~thin], om[~thin])
    ) != Counter(canonical(v, tag) for v, tag in zip(ct, cm)):
        raise ValueError("surviving oriented tagged triangles differ from contact")
    pmap = []
    for index in range(int(op.max()) + 1):
        targets = set(map(int, cp[mapping[op == index]]))
        if len(targets) != 1:
            raise ValueError("periodic node collapse is inconsistent")
        pmap.append(targets.pop())
    if pmap[0] != 0 or any(i == 0 for i in pmap[1:]):
        raise ValueError("the existing fixed gauge is not preserved")
    groups = [
        [j for j, target in enumerate(pmap) if target == i]
        for i in range(int(cp.max()) + 1)
    ]
    if any(not group for group in groups):
        raise ValueError("contact collapse misses a potential")
    cfull = np.eye(len(groups))[pmap]
    extras = [i for group in groups[1:] for i in group[1:]]
    zfull = np.zeros((len(pmap), len(extras)))
    zfull[extras, np.arange(len(extras))] = 1.0
    # Exact pole graph: repeated contact endpoints enforce equality of the
    # corresponding open potentials. Each group must be one connected component.
    adjacency = {i: set() for i in range(len(pmap))}
    graph_edges = []
    for triangle in ot[thin]:
        pairs = [
            (i, j)
            for i in range(3)
            for j in range(i + 1, 3)
            if mapping[triangle[i]] == mapping[triangle[j]]
        ]
        if len(pairs) != 1:
            raise ValueError("collapsed child needs exactly one repeated endpoint pair")
        i, j = pairs[0]
        a, b = int(op[triangle[i]]), int(op[triangle[j]])
        if pmap[a] != pmap[b] or a == b:
            raise ValueError(
                "pole graph constraint is not a genuine collapsing difference"
            )
        adjacency[a].add(b)
        adjacency[b].add(a)
        graph_edges.append((a, b))
    for group in groups:
        visited, pending = set(), [group[0]]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(adjacency[node] - visited)
        if visited != set(group):
            raise ValueError(
                "thin pole graph does not connect exactly the collapse groups"
            )
    return dict(
        open_numerators=on,
        contact_numerators=cn,
        weight_denominator=denominator,
        triangles=ot,
        tags=om,
        periodic=op,
        contact_triangles=ct,
        contact_tags=cm,
        contact_periodic=cp,
        mapping=mapping,
        pmap=np.array(pmap),
        thin=thin,
        Cfull=cfull,
        Zfull=zfull,
        C=cfull[1:, 1:],
        Z=zfull[1:],
        groups=groups,
        extras=extras,
        graph_edges=graph_edges,
    )


def weak_limit(vertices, velocity, data, materials, phase=1.0):
    size = len(data["pmap"])
    k, f = np.zeros((3, size, size), complex), np.zeros((2, size, 2), complex)
    local = []
    c, z = data["C"], data["Z"]
    a00, da00 = (
        np.zeros((c.shape[1], c.shape[1]), complex),
        np.zeros((c.shape[1], c.shape[1]), complex),
    )
    a01, a11 = (
        np.zeros((c.shape[1], z.shape[1]), complex),
        np.zeros((z.shape[1], z.shape[1]), complex),
    )
    cf0, cf1, zf0 = (
        np.zeros((c.shape[1], 2), complex),
        np.zeros((c.shape[1], 2), complex),
        np.zeros((z.shape[1], 2), complex),
    )
    reference = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    for triangle, tag, thin in zip(data["triangles"], data["tags"], data["thin"]):
        p, v = vertices[triangle], velocity[triangle]
        j0, j1 = (np.column_stack([r[1] - r[0], r[2] - r[0]]) for r in (p, v))
        adj = lambda j: np.array([[j[1, 1], -j[0, 1]], [-j[1, 0], j[0, 0]]])
        d0 = j0[0, 0] * j0[1, 1] - j0[0, 1] * j0[1, 0]
        d1 = j1[0, 0] * j0[1, 1] - j1[0, 1] * j0[1, 0]
        if not np.all(j1[1] == 0) or not np.isfinite([d0, d1]).all():
            raise ValueError("opening Jacobian must be affine in gap with fixed y")
        n0, n1 = reference @ adj(j0), reference @ adj(j1)
        h0, h1, h2 = n0 @ n0.T, n0 @ n1.T + n1 @ n0.T, n1 @ n1.T
        if thin:
            if d0 != 0 or d1 <= 0:
                raise ValueError(
                    "topologically collapsed triangle needs exactly zero area and positive slope"
                )
            terms = np.array([h0, h1, h2]) / (2 * d1)
        else:
            if d0 <= 0:
                raise ValueError(
                    "regular contact triangles need strictly positive area"
                )
            terms = np.array(
                [np.zeros((3, 3)), h0 / (2 * d0), (h1 / d0 - h0 * d1 / d0**2) / 2]
            )
        loads = -np.array([n0, n1]) / 2
        ids = data["periodic"][triangle]
        lc, lz, epsilon = data["Cfull"][ids, 1:], data["Zfull"][ids], materials[tag]
        if thin:
            # N0^T lc is structurally zero: equal potentials on the repeated
            # endpoint group. Project before summation instead of canceling
            # large assembled pole entries against one another.
            if np.any(n0.T @ lc != 0):
                raise ValueError("local pole cancellation is not structurally exact")
            cn1, zn0 = n1.T @ lc, n0.T @ lz
            a01 += epsilon / (2 * d1) * (cn1.T @ zn0)
            a11 += epsilon / (2 * d1) * (zn0.T @ zn0)
            da00 += epsilon / (2 * d1) * (cn1.T @ cn1)
        else:
            a00 += epsilon * (lc.T @ terms[1] @ lc)
            a01 += epsilon * (lc.T @ terms[1] @ lz)
            da00 += epsilon * (lc.T @ terms[2] @ lc)
            cf0 -= epsilon / 2 * (lc.T @ n0)
        cf1 -= epsilon / 2 * (lc.T @ n1)
        zf0 -= epsilon / 2 * (lz.T @ n0)
        for i in range(3):
            f[:, ids[i]] += materials[tag] * loads[:, i]
            for j in range(3):
                k[:, ids[i], ids[j]] += materials[tag] * terms[:, i, j]
        local.append((d0, d1, n0, n1))
    km, k0, k1 = k[:, 1:, 1:]
    f0, f1 = f[:, 1:]
    min_contact = np.linalg.eigvalsh((a00 + a00.conj().T) / 2)[0]
    min_difference = np.linalg.eigvalsh((a11 + a11.conj().T) / 2)[0]
    rotated_contact = min_contact if phase == 1.0 else minimum_rotated_storage(a00, phase)
    rotated_difference = min_difference if phase == 1.0 else minimum_rotated_storage(a11, phase)
    if rotated_contact <= 0 or rotated_difference <= 0:
        raise ValueError("limiting weak blocks failed coercivity")
    u0 = np.linalg.solve(a00, cf0)
    v0 = np.linalg.solve(a11, zf0 - a01.T @ u0)
    du0 = np.linalg.solve(a00, cf1 - da00 @ u0 - a01 @ v0)
    w0 = np.vstack([np.zeros((1, 2)), c @ u0])
    w1 = np.vstack([np.zeros((1, 2)), c @ du0 + z @ v0])
    v1 = np.linalg.solve(a11, z.T @ (f1 - k1 @ w0[1:] - k0 @ w1[1:]))
    pole = np.linalg.norm(k[0] @ data["Cfull"]) / max(
        np.linalg.norm(k[0]) * np.linalg.norm(data["Cfull"]), 1.0
    )
    residuals = [
        np.linalg.norm(k[0] @ w1 + k[1] @ w0 - f[0]) / max(np.linalg.norm(f[0]), 1.0),
        np.linalg.norm(k[0] @ data["Zfull"] @ v1 + k[1] @ w1 + k[2] @ w0 - f[1])
        / max(np.linalg.norm(f[1]), 1.0),
    ]
    projected_agreement = max(
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)
        for a, b in (
            (a00, c.T @ k0 @ c),
            (a01, c.T @ k0 @ z),
            (a11, z.T @ km @ z),
            (da00, c.T @ k1 @ c),
            (cf0, c.T @ f0),
            (cf1, c.T @ f1),
            (zf0, z.T @ f0),
        )
    )
    if (
        projected_agreement > 3e-12
        or pole > 3e-12
        or not np.isfinite(residuals).all()
        or max(residuals) > 1e-10
    ):
        raise ValueError(
            "original Laurent weak residual or pole-cancellation guard failed"
        )
    return (
        w0,
        w1,
        data["Zfull"] @ v0,
        local,
        dict(
            original_residuals=residuals,
            pole_cancellation=pole,
            projected_operator_agreement=projected_agreement,
            contact_minimum_real_eigenvalue=min_contact,
            contact_minimum_rotated_eigenvalue=rotated_contact,
            difference_minimum_rotated_eigenvalue=rotated_difference,
            difference_minimum_real_eigenvalue=min_difference,
            contact_condition=np.linalg.cond(a00),
            difference_condition=np.linalg.cond(a11),
            K=k,
            F=f,
        ),
    )


def sinc_prime(x):
    small = np.abs(x) < 1e-3
    safe = np.where(small, 1.0, x)
    series = -x / 3 + x**3 / 30 - x**5 / 840 + x**7 / 45360
    return np.where(small, series, (safe * np.cos(safe) - np.sin(safe)) / safe**2)


def contour_derivative(points, velocity, wave, area):
    """Differentiate the existing exact edge contour at fixed physical k/A."""
    edge = np.roll(points, -1, axis=0) - points
    dedge = np.roll(velocity, -1, axis=0) - velocity
    midpoint = (np.roll(points, -1, axis=0) + points) / 2
    dmidpoint = (np.roll(velocity, -1, axis=0) + velocity) / 2
    cross = wave[:, 0, None] * edge[None, :, 1] - wave[:, 1, None] * edge[None, :, 0]
    dcross = wave[:, 0, None] * dedge[None, :, 1] - wave[:, 1, None] * dedge[None, :, 0]
    x, dx = wave @ edge.T / 2, wave @ dedge.T / 2
    phase = np.exp(-1j * wave @ midpoint.T)
    sinc = np.sinc(x / np.pi)
    change = phase * (
        dcross * sinc + cross * (sinc_prime(x) * dx - 1j * (wave @ dmidpoint.T) * sinc)
    )
    k2 = np.sum(wave**2, axis=1)
    nonzero = k2 != 0
    value = 1j * change.sum(axis=1) / np.where(nonzero, k2, 1.0) / area
    x0, y0 = points.T
    dx0, dy0 = velocity.T
    darea = 0.5 * np.sum(
        dx0 * np.roll(y0, -1)
        + x0 * np.roll(dy0, -1)
        - np.roll(dx0, -1) * y0
        - np.roll(x0, -1) * dy0
    )
    return np.where(nonzero, value, darea / area + 0j)


def phi2(z):
    small = np.abs(z) < 1e-3
    safe = np.where(small, 1.0, z)
    term = np.ones_like(z) / 2
    series = term.copy()
    for n in range(1, 10):
        term = term * z / (n + 2)
        series += term
    return np.where(small, series, (np.expm1(safe) - safe) / safe**2)


def collapsed_edge_weight(points, slope, wave, area):
    repeated = [
        (i, j)
        for i in range(3)
        for j in range(i + 1, 3)
        if np.array_equal(points[i], points[j])
    ]
    if len(repeated) != 1:
        raise ValueError(
            "The collapsed-child formula requires exactly one repeated endpoint pair"
        )
    i, j = repeated[0]
    other = next(k for k in range(3) if k not in (i, j))
    a, b = points[i], points[other]
    return slope / area * np.exp(-1j * wave @ a) * phi2(-1j * wave @ (b - a))


def components(field, gram, material):
    return np.stack(
        [
            field[:, 0, 0],
            field[:, 0, 1],
            field[:, 1, 0],
            field[:, 1, 1],
            gram[:, 0, 0],
            gram[:, 0, 1],
            gram[:, 1, 1],
            material,
            gram[:, 1, 0],
        ]
    )


def freeze_topology(data):
    frozen = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(
                value.shape
            )
        elif isinstance(value, list):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        frozen[key] = value
    return MappingProxyType(frozen)


class CartesianPairOpeningCell:
    """Frozen finite pair response; derivative per caller length unit, NumPy only."""

    def __init__(self, plan, order=1, *, coercivity_angle=0.0):
        self._coercivity_phase = phase_from_angle(coercivity_angle)
        self._coercivity_angle = float(coercivity_angle)
        if not isinstance(plan, CartesianPairOpeningPlan):
            raise ValueError("an explicit CartesianPairOpeningPlan is required")
        pair = (order, order) if isinstance(order, Integral) else order
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
                for v in pair
            )
        ):
            raise ValueError("order must be a nonnegative integer or integer pair")
        self.plan, self.order, self._data = (
            plan,
            tuple(map(int, pair)),
            freeze_topology(topology(plan)),
        )

    @property
    def coercivity_angle(self):
        return self._coercivity_angle

    @property
    def data(self):
        return self._data

    def coefficients(
        self, *, basis, widths, height, epsilon, background=1.0, offset=(0.0, 0.0)
    ):
        geometry = validate_geometry(basis, widths, height, offset)
        material = materials_array(epsilon, background, self._coercivity_phase)
        points, _, _, _, _ = CartesianPairPlan(
            True, self.plan.shift, self.plan.refinement
        ).replay(**geometry)
        vertices = points[self.data["mapping"]]
        base_velocity = np.zeros((12, 2))
        base_velocity[4:, 0] = [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5]
        velocity = (
            self.data["open_numerators"] / self.data["weight_denominator"]
        ) @ base_velocity
        w0, w1, difference, local, weak = weak_limit(
            vertices, velocity, self.data, material, self._coercivity_phase
        )
        ox, oy = self.order
        n, m = np.meshgrid(
            np.arange(-2 * oy, 2 * oy + 1),
            np.arange(-2 * ox, 2 * ox + 1),
            indexing="ij",
        )
        a = geometry["basis"]
        wave = 2 * np.pi * np.column_stack([m.ravel(), n.ravel()]) @ np.linalg.inv(a)
        area = np.linalg.det(a)
        fields, changes, weights, dweights, edge = [], [], [], [], []
        for triangle, thin, (d0, d1, n0, n1) in zip(
            self.data["triangles"], self.data["thin"], local
        ):
            p, dp = vertices[triangle], velocity[triangle]
            ids = self.data["periodic"][triangle]
            dw = contour_derivative(p, dp, wave, area)
            if thin:
                b = np.eye(2) + (n1.T @ w0[ids] + n0.T @ difference[ids]) / d1
                db = np.zeros(
                    (2, 2), complex
                )  # Only exact W0=0 multiplies this placeholder.
                weight = np.zeros(len(wave), complex)
                edge.append(collapsed_edge_weight(p, d1, wave, area))
            else:
                grad = n0 / d0
                dgrad = n1 / d0 - n0 * d1 / d0**2
                b = np.eye(2) + grad.T @ w0[ids]
                db = dgrad.T @ w0[ids] + grad.T @ w1[ids]
                weight = polygon_ft(np, p, n, m, a).ravel()
            fields.append(b)
            changes.append(db)
            weights.append(weight)
            dweights.append(dw)
        fields, changes = np.array(fields), np.array(changes)
        tagged = material[self.data["tags"]]
        gram = tagged[:, None, None] * (fields.conj().transpose(0, 2, 1) @ fields)
        dgram = tagged[:, None, None] * (
            changes.conj().transpose(0, 2, 1) @ fields
            + fields.conj().transpose(0, 2, 1) @ changes
        )
        values = components(fields, gram, tagged)
        dvalues = components(changes, dgram, np.zeros_like(tagged))
        weights, dweights = np.asarray(weights), np.asarray(dweights)
        shape = (9, 4 * oy + 1, 4 * ox + 1)
        value = (values @ weights).reshape(shape)
        derivative = (dvalues @ weights + values @ dweights).reshape(shape)
        if not np.isfinite(value).all() or not np.isfinite(derivative).all():
            raise ValueError("nonfinite opening coefficient response")
        metadata = dict(
            order=self.order,
            coercivity_angle=self.coercivity_angle,
            refinement=self.plan.refinement,
            shift=self.plan.shift,
            area=area,
            open_free_dofs=len(self.data["pmap"]) - 1,
            contact_free_dofs=self.data["C"].shape[1],
            difference_dofs=self.data["Z"].shape[1],
            collapsed_triangles=int(self.data["thin"].sum()),
            derivative_units="per caller length unit",
            **{k: v for k, v in weak.items() if k not in ("K", "F")},
        )
        details = dict(
            vertices=vertices,
            velocity=velocity,
            fields=fields,
            regular_field_derivative=changes,
            weights=weights,
            dweights=dweights,
            phi2_weights=np.asarray(edge),
            K=weak["K"],
            F=weak["F"],
        )
        return value, derivative, metadata, details

    def closure(self, **settings):
        value, derivative, metadata, details = self.coefficients(**settings)
        slots, dslots, closure_meta = energy_opening(
            value,
            derivative,
            self.order,
            settings["epsilon"],
            settings.get("background", 1.0),
            phase=self._coercivity_phase,
        )
        return slots, dslots, dict(metadata, **closure_meta), details


def energy_opening(coefficients, derivative, order, epsilon, background, *, phase=1.0):
    ox, oy = order
    n, m = np.meshgrid(np.arange(-oy, oy + 1), np.arange(-ox, ox + 1), indexing="ij")
    dn, dm = (
        n.ravel()[:, None] - n.ravel() + 2 * oy,
        m.ravel()[:, None] - m.ravel() + 2 * ox,
    )
    c, dc = coefficients[:, dn, dm], derivative[:, dn, dm]

    def matrices(v):
        return (
            np.block([[v[0], v[1]], [v[2], v[3]]]),
            np.block([[v[4], v[5]], [v[8], v[6]]]),
            v[7],
        )

    (q, g, e), (dq, dg, de) = matrices(c), matrices(dc)
    qi, zi = np.linalg.inv(q), np.linalg.inv(e)
    dqi, dzi = -qi @ dq @ qi, -zi @ de @ zi
    h = qi.conj().T @ g @ qi
    dh = dqi.conj().T @ g @ qi + qi.conj().T @ dg @ qi + qi.conj().T @ g @ dqi
    count = len(e)
    slots = ConvMatrices.from_energy(q, g, e)
    dslots = ConvMatrices(
        dh[:count, :count],
        dh[count:, count:],
        dzi,
        -dh[:count, count:],
        -dh[count:, :count],
    )
    storage, loss = (
        np.linalg.eigvalsh((h + h.conj().T) / 2)[0],
        np.linalg.eigvalsh((h - h.conj().T) / (2j))[0],
    )
    rotated_storage = storage if phase == 1.0 else minimum_rotated_storage(h, phase)
    lossless = np.all(np.imag(epsilon) == 0) and np.imag(background) == 0
    hermitian_error = np.max(np.abs(h - h.conj().T))
    residuals = [
        np.linalg.norm(q.conj().T @ h @ q - g) / max(np.linalg.norm(g), 1.0),
        np.linalg.norm(
            dq.conj().T @ h @ q + q.conj().T @ dh @ q + q.conj().T @ h @ dq - dg
        )
        / max(np.linalg.norm(dg), 1.0),
        np.linalg.norm(de @ zi + e @ dzi)
        / max(np.linalg.norm(de) * np.linalg.norm(zi), 1.0),
    ]
    if (
        not np.isfinite(h).all()
        or not np.isfinite(dh).all()
        or not np.isfinite(dzi).all()
        or rotated_storage <= 0
        or loss <= -1e-9
        or (lossless and hermitian_error >= 1e-9)
        or max(residuals) > 1e-10
    ):
        raise ValueError(
            "finite opening closure failed its native physical/residual guards"
        )
    return (
        ConvMatrices(*(v[None] for v in slots)),
        ConvMatrices(*(v[None] for v in dslots)),
        dict(
            minimum_storage=storage,
            minimum_rotated_storage=rotated_storage,
            minimum_dissipation=loss,
            hermitian_error=hermitian_error,
            lossless=lossless,
            inverse_residuals=residuals,
        ),
    )
