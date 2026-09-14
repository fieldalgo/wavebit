"""Experimental quadratic-potential finite-gap pair cell, NumPy only.

The geometry is affine on each triangle. Six conforming potential nodes give
linear auxiliary fields and quadratic energy densities. Exact polynomial
integrals retain the finite contact cancellation and physical gap derivative.
This explicit instrument does not certify optical convergence or alter NVF
routing. See docs/nvf-pair-quadratic.md.
"""

import numpy as np
from .cartesian_pair import (
    CartesianPairPlan,
    _physical_geometry,
)
from .cartesian_pair_opening import (
    CartesianPairOpeningCell,
    components,
    energy_opening,
    CartesianPairOpeningPlan,
    freeze_topology,
    topology,
    _opening_topology,
    materials_array,
    real_array,
    validate_geometry,
)
from .triangle_moments import quadratic_triangle_weights
from .material_sector import (
    phase_from_angle, minimum_rotated_storage,
)


def _quadratic_topology(plan):
    """Use the next P1 node graph as the six-node P2 graph on this mesh."""
    coarse = topology(plan)
    fine = _opening_topology(plan.shift, plan.refinement+1)
    weights = fine["open_numerators"] / fine["weight_denominator"]
    parents = coarse["open_numerators"] / coarse["weight_denominator"]
    lookup = {tuple(row): i for i, row in enumerate(weights)}
    triangles = []
    for triangle in coarse["triangles"]:
        rows = list(parents[triangle])
        rows += [(parents[triangle[i]]+parents[triangle[j]])/2
                 for i, j in ((0, 1), (1, 2), (2, 0))]
        triangles.append([lookup[tuple(row)] for row in rows])
    data = {name: fine[name] for name in (
        "open_numerators", "contact_numerators", "weight_denominator", "mapping", "periodic",
        "pmap", "Cfull", "Zfull", "C", "Z")}
    data.update(triangles=np.asarray(triangles), tags=coarse["tags"], thin=coarse["thin"])
    # Certify transverse derivative annihilation using integer polynomial
    # coefficients before any floating geometry or material enters the cell.
    reference = _reference_gradients()
    for triangle in data["triangles"][data["thin"]]:
        mapped = fine["mapping"][triangle[:3]]
        i, j = next((i, j) for i in range(3) for j in range(i+1, 3) if mapped[i] == mapped[j])
        direction = {(0, 1): (1, 0), (0, 2): (0, 1), (1, 2): (1, -1)}[i, j]
        c = fine["Cfull"][fine["periodic"][triangle]]
        if np.any((reference @ np.asarray(direction)) @ c != 0):
            raise ValueError("quadratic collapse does not cancel the transverse polynomial")
    return freeze_topology(data)


def _reference_gradients():
    """Six nodal shape gradients at the three reference vertices."""
    barycentric = np.eye(3)
    gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    vertex = (4*barycentric[:, :, None]-1)*gradients
    edge = np.stack([4*(barycentric[:, i, None]*gradients[j]
                       + barycentric[:, j, None]*gradients[i])
                     for i, j in ((0, 1), (1, 2), (2, 0))], axis=1)
    return np.concatenate([vertex, edge], axis=1)


def _bilinear(left, right):
    """Exact reference integral of two linear matrices; no conjugation."""
    return (left.reshape(-1, left.shape[-1]).T @ right.reshape(-1, right.shape[-1])
            + left.sum(axis=0).T @ right.sum(axis=0)) / 24


def _bernstein_components(fields, changes, material):
    values, derivatives, gram, dgram = [], [], [], []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)):
        a, b, da, db = fields[i], fields[j], changes[i], changes[j]
        values.append((a+b)/2)
        derivatives.append((da+db)/2)
        gram.append(material*(a.conj().T@b+b.conj().T@a)/2)
        dgram.append(material*(da.conj().T@b+a.conj().T@db
                               +db.conj().T@a+b.conj().T@da)/2)
    return (components(np.asarray(values), np.asarray(gram), np.full(6, material)),
            components(np.asarray(derivatives), np.asarray(dgram), np.zeros(6)))


class CartesianPairQuadraticGapCell:
    """Explicit P2 gap>=0 response with analytic right derivative at contact.

    plan.refinement is the geometric subdivision (0..2). Its six-node
    potential graph has the DOF count of P1 at refinement+1. Physical passive
    materials must share the declared coercive sector. No JAX replay or
    automatic ordinary-NVF dispatch is provided by this NumPy instrument.
    """

    def __init__(self, plan, order=1, *, coercivity_angle=0.0):
        endpoint = CartesianPairOpeningCell(plan, order, coercivity_angle=coercivity_angle)
        if plan.refinement > 2:
            raise ValueError("quadratic geometric refinement must be in 0..2")
        self._coercivity_angle = endpoint.coercivity_angle
        self._coercivity_phase = phase_from_angle(self._coercivity_angle)
        self._plan, self._order, self._data = plan, endpoint.order, _quadratic_topology(plan)

    @property
    def coercivity_angle(self):
        return self._coercivity_angle

    @property
    def plan(self):
        return self._plan

    @property
    def order(self):
        return self._order

    @property
    def data(self):
        return self._data

    def coefficients(
        self, *, gap, basis, widths, height, epsilon, background=1.0, offset=(0.0, 0.0)
    ):
        geometry = validate_geometry(basis, widths, height, offset)
        material = materials_array(epsilon, background, self._coercivity_phase)
        gap = real_array(gap, (), "gap").item()
        if gap < 0:
            raise ValueError("gap must be nonnegative")
        length = np.sqrt(np.linalg.det(geometry["basis"]))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("finite positive normalization length required")
        s = gap / length
        normalized = {
            key: geometry[key] / length
            for key in ("basis", "widths", "height", "offset")
        }
        # Envelope admission has no small-triangle test and changes no topology.
        _physical_geometry(gap=s, **normalized)
        contact, _, _, _, _ = CartesianPairPlan(
            True, self.plan.shift, self.plan.refinement+1
        ).replay(gap=0.0, **normalized)
        zero = contact[self.data["mapping"]]
        base_velocity = np.zeros((12, 2))
        base_velocity[4:, 0] = [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5]
        velocity = (
            self.data["open_numerators"]
            / self.data["weight_denominator"]
            @ base_velocity
        )
        vertices = zero + s * velocity
        nc, nz = self.data["C"].shape[1], self.data["Z"].shape[1]
        h00, h01, h11 = (
            np.zeros((nc, nc), complex),
            np.zeros((nc, nz), complex),
            np.zeros((nz, nz), complex),
        )
        dh00, dh01, dh11 = np.zeros_like(h00), np.zeros_like(h01), np.zeros_like(h11)
        fc, fz = np.zeros((nc, 2), complex), np.zeros((nz, 2), complex)
        dfc, dfz = np.zeros_like(fc), np.zeros_like(fz)
        local = []
        reference = _reference_gradients()
        adj = lambda j: np.array([[j[1, 1], -j[0, 1]], [-j[1, 0], j[0, 0]]])
        for triangle, tag, thin in zip(
            self.data["triangles"], self.data["tags"], self.data["thin"]
        ):
            p, dp = zero[triangle[:3]], velocity[triangle[:3]]
            j0, j1 = (np.column_stack([v[1] - v[0], v[2] - v[0]]) for v in (p, dp))
            if np.any(j1[1] != 0):
                raise ValueError("only horizontal opening is supported")
            d0 = j0[0, 0] * j0[1, 1] - j0[0, 1] * j0[1, 0]
            d1 = j1[0, 0] * j0[1, 1] - j1[0, 1] * j0[1, 0]
            n0, n1 = reference @ adj(j0), reference @ adj(j1)
            n = n0 + s * n1
            ids = self.data["periodic"][triangle]
            c, z, e = (
                self.data["Cfull"][ids, 1:],
                self.data["Zfull"][ids],
                material[tag],
            )
            x1, y1 = n1.swapaxes(-1, -2) @ c, n1.swapaxes(-1, -2) @ z
            y = n.swapaxes(-1, -2) @ z
            yy, dyy = _bilinear(y, y), _bilinear(y1, y) + _bilinear(y, y1)
            if thin:
                # The integer topology certificate makes N0.T C the zero
                # polynomial; do not infer it from rounded physical products.
                if d0 != 0 or d1 <= 0:
                    raise ValueError(
                        "thin geometry needs exact collapse, pole cancellation and positive slope"
                    )
                factor = e / d1
                h00 += factor * s * _bilinear(x1, x1)
                dh00 += factor * _bilinear(x1, x1)
                h01 += factor * _bilinear(x1, y)
                dh01 += factor * _bilinear(x1, y1)
                h11 += factor * yy
                dh11 += factor * dyy
                fc -= e * s * x1.sum(axis=0).T / 6
                x = s * x1
            else:
                d = d0 + s * d1
                if d0 <= 0 or d <= 0:
                    raise ValueError(
                        "regular triangles must remain positive on this chart"
                    )
                x = n.swapaxes(-1, -2) @ c
                xx, dxx = _bilinear(x, x), _bilinear(x1, x) + _bilinear(x, x1)
                xy, dxy = _bilinear(x, y), _bilinear(x1, y) + _bilinear(x, y1)
                factor = e / d
                h00 += factor * xx
                dh00 += factor * (dxx - xx * d1 / d)
                h01 += factor * xy
                dh01 += factor * (dxy - xy * d1 / d)
                h11 += factor * s * yy
                dh11 += factor * (yy + s * dyy - s * yy * d1 / d)
                fc -= e * x.sum(axis=0).T / 6
            dfc -= e * x1.sum(axis=0).T / 6
            fz -= e * y.sum(axis=0).T / 6
            dfz -= e * y1.sum(axis=0).T / 6
            local.append((d0, d1, n, n1, x, x1, y, y1))
        matrix = np.block([[h00, s * h01], [h01.T, h11]])
        dmatrix = np.block([[dh00, h01 + s * dh01], [dh01.T, dh11]])
        rhs, drhs = np.vstack([fc, fz]), np.vstack([dfc, dfz])
        if (
            not np.isfinite(matrix).all()
            or not np.isfinite(dmatrix).all()
            or not np.isfinite(rhs).all()
        ):
            raise ValueError("nonfinite rescaled weak equations")
        contact_min = np.linalg.eigvalsh((h00 + h00.conj().T) / 2)[0]
        difference_min = np.linalg.eigvalsh((h11 + h11.conj().T) / 2)[0]
        rotated_contact = (contact_min if self._coercivity_phase == 1.0 else
                           minimum_rotated_storage(h00, self._coercivity_phase))
        rotated_difference = (difference_min if self._coercivity_phase == 1.0 else
                              minimum_rotated_storage(h11, self._coercivity_phase))
        if min(rotated_contact, rotated_difference) <= 0:
            raise ValueError("contact/difference diagonal blocks failed coercivity")
        solution = np.linalg.solve(matrix, rhs)
        # Resolve the more ill-conditioned quadratic potential before taking
        # its derivative. Defect correction leaves the physical equations and
        # original unprojected residual guard unchanged.
        for _ in range(2):
            solution += np.linalg.solve(matrix, rhs - matrix @ solution)
        tangent_rhs = drhs - dmatrix @ solution
        derivative = np.linalg.solve(matrix, tangent_rhs)
        for _ in range(2):
            derivative += np.linalg.solve(matrix, tangent_rhs - matrix @ derivative)
        u, v = solution[:nc], solution[nc:]
        du, dv = derivative[:nc], derivative[nc:]
        residual = np.linalg.norm(matrix @ solution - rhs) / max(
            np.linalg.norm(rhs), 1.0
        )
        derivative_residual = np.linalg.norm(
            matrix @ derivative + dmatrix @ solution - drhs
        ) / max(np.linalg.norm(drhs), 1.0)
        full_count = len(self.data["pmap"])
        original, doriginal, full_load, dfull_load = (
            np.zeros((full_count, 2), complex) for _ in range(4)
        )
        fields, changes = [], []
        for triangle, tag, thin, (d0, d1, n, n1, x, x1, y, y1) in zip(
            self.data["triangles"], self.data["tags"], self.data["thin"], local
        ):
            if thin:
                field = np.eye(2) + (x1 @ u + y @ v) / d1
                dfield = (x1 @ du + y1 @ v + y @ dv) / d1
            else:
                d = d0 + s * d1
                numerator = x @ u + s * y @ v
                dnumerator = x1 @ u + x @ du + y @ v + s * (y1 @ v + y @ dv)
                field = np.eye(2) + numerator / d
                dfield = dnumerator / d - numerator * d1 / d**2
            ids, e = self.data["periodic"][triangle], material[tag]
            # Original weak equations, before C/Z projection; area cancels the
            # local gradient denominator exactly, including at contact.
            np.add.at(original, ids, e * _bilinear(n.swapaxes(-1, -2), field))
            np.add.at(doriginal, ids, e * (_bilinear(n1.swapaxes(-1, -2), field) + _bilinear(n.swapaxes(-1, -2), dfield)))
            np.add.at(full_load, ids, -e * n.sum(axis=0) / 6)
            np.add.at(dfull_load, ids, -e * n1.sum(axis=0) / 6)
            fields.append(field)
            changes.append(dfield)
        original_residual = np.linalg.norm(original) / max(
            np.linalg.norm(full_load), 1.0
        )
        original_derivative_residual = np.linalg.norm(doriginal) / max(
            np.linalg.norm(dfull_load), 1.0
        )
        residuals = [
            residual,
            derivative_residual,
            original_residual,
            original_derivative_residual,
        ]
        if not np.isfinite(residuals).all() or max(residuals) > 1e-10:
            raise ValueError("rescaled or original unprojected weak residual failed")
        fields, changes = np.asarray(fields), np.asarray(changes)
        ox, oy = self.order
        n, m = np.meshgrid(
            np.arange(-2 * oy, 2 * oy + 1),
            np.arange(-2 * ox, 2 * ox + 1),
            indexing="ij",
        )
        a = normalized["basis"]
        area = np.linalg.det(a)
        wave = 2 * np.pi * np.column_stack([m.ravel(), n.ravel()]) @ np.linalg.inv(a)
        weights, dweights, values, dvalues = [], [], [], []
        for triangle, tag, field, change in zip(
                self.data["triangles"], self.data["tags"], fields, changes):
            weight, dweight = quadratic_triangle_weights(
                vertices[triangle[:3]], velocity[triangle[:3]], wave)
            value, dvalue = _bernstein_components(field, change, material[tag])
            weights.append(weight/area)
            dweights.append(dweight/area)
            values.append(value)
            dvalues.append(dvalue)
        weights, dweights = np.asarray(weights), np.asarray(dweights)
        values, dvalues = np.asarray(values), np.asarray(dvalues)
        shape = (9, 4*oy+1, 4*ox+1)
        value = (values @ weights).sum(axis=0).reshape(shape)
        per_unit = ((dvalues @ weights + values @ dweights).sum(axis=0)/length).reshape(shape)
        if not np.isfinite(value).all() or not np.isfinite(per_unit).all():
            raise ValueError("nonfinite finite-gap coefficients")
        metadata = dict(
            gap=gap,
            coercivity_angle=self.coercivity_angle,
            normalized_gap=s,
            normalization_length=length,
            area=length**2,
            order=self.order,
            shift=self.plan.shift,
            refinement=self.plan.refinement,
            potential_degree=2,
            open_free_dofs=nc + nz,
            contact_free_dofs=nc,
            difference_dofs=nz,
            matrix_condition=np.linalg.cond(matrix),
            contact_minimum_real_eigenvalue=contact_min,
            contact_minimum_rotated_eigenvalue=rotated_contact,
            difference_minimum_rotated_eigenvalue=rotated_difference,
            difference_minimum_real_eigenvalue=difference_min,
            linear_residual=residual,
            linear_derivative_residual=derivative_residual,
            original_flux_residual=original_residual,
            original_flux_derivative_residual=original_derivative_residual,
            derivative_units="per caller length unit",
        )
        details = dict(
            M=matrix,
            dM_ds=dmatrix,
            f=rhs,
            df_ds=drhs,
            solution=solution,
            derivative_ds=derivative,
            vertices=vertices,
            velocity_ds=velocity,
            fields=fields,
            field_derivative_ds=changes,
            weights=weights,
            weight_derivative_ds=dweights,
            original_flux=original,
            original_flux_derivative_ds=doriginal,
        )
        return value, per_unit, metadata, details

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
