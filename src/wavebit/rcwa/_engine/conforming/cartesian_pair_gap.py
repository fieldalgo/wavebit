"""Experimental exact finite-gap NumPy cell on one immutable contact/open chart.

All outputs retain caller physical units; derivatives are per unit gap.
Analytical block/field derivatives have no division by gap at contact.
"""

import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair import CartesianPairPlan
from wavebit.rcwa._engine.conforming.cartesian_pair import _physical_geometry
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningCell
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import components
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import contour_derivative
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import energy_opening
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import materials_array
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import real_array
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import validate_geometry
from wavebit._common.geometry.polygon import polygon_ft
from wavebit.rcwa._engine.conforming.material_sector import phase_from_angle
from wavebit.rcwa._engine.conforming.material_sector import minimum_rotated_storage


class CartesianPairGapCell:
    """Separate explicit gap>=0 response, never an ordinary contact JVP."""

    def __init__(self, plan, order=1, *, coercivity_angle=0.0):
        endpoint = CartesianPairOpeningCell(plan, order, coercivity_angle=coercivity_angle)
        self._coercivity_angle = endpoint.coercivity_angle
        self._coercivity_phase = phase_from_angle(self._coercivity_angle)
        self._plan, self._order, self._data = (
            endpoint.plan,
            endpoint.order,
            endpoint.data,
        )

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
            True, self.plan.shift, self.plan.refinement
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
        reference = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        adj = lambda j: np.array([[j[1, 1], -j[0, 1]], [-j[1, 0], j[0, 0]]])
        for triangle, tag, thin in zip(
            self.data["triangles"], self.data["tags"], self.data["thin"]
        ):
            p, dp = zero[triangle], velocity[triangle]
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
            x1, y1 = n1.T @ c, n1.T @ z
            y = n.T @ z
            yy, dyy = y.T @ y, y1.T @ y + y.T @ y1
            if thin:
                if d0 != 0 or d1 <= 0 or np.any(n0.T @ c != 0):
                    raise ValueError(
                        "thin geometry needs exact collapse, pole cancellation and positive slope"
                    )
                factor = e / (2 * d1)
                h00 += factor * s * (x1.T @ x1)
                dh00 += factor * (x1.T @ x1)
                h01 += factor * (x1.T @ y)
                dh01 += factor * (x1.T @ y1)
                h11 += factor * yy
                dh11 += factor * dyy
                fc -= e * s * x1.T / 2
                x = s * x1
            else:
                d = d0 + s * d1
                if d0 <= 0 or d <= 0:
                    raise ValueError(
                        "regular triangles must remain positive on this chart"
                    )
                x = n.T @ c
                xx, dxx = x.T @ x, x1.T @ x + x.T @ x1
                xy, dxy = x.T @ y, x1.T @ y + x.T @ y1
                factor = e / (2 * d)
                h00 += factor * xx
                dh00 += factor * (dxx - xx * d1 / d)
                h01 += factor * xy
                dh01 += factor * (dxy - xy * d1 / d)
                h11 += factor * s * yy
                dh11 += factor * (yy + s * dyy - s * yy * d1 / d)
                fc -= e * x.T / 2
            dfc -= e * x1.T / 2
            fz -= e * y.T / 2
            dfz -= e * y1.T / 2
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
        derivative = np.linalg.solve(matrix, drhs - dmatrix @ solution)
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
            np.add.at(original, ids, e / 2 * (n @ field))
            np.add.at(doriginal, ids, e / 2 * (n1 @ field + n @ dfield))
            np.add.at(full_load, ids, -e * n / 2)
            np.add.at(dfull_load, ids, -e * n1 / 2)
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
        ox, oy = self.order
        n, m = np.meshgrid(
            np.arange(-2 * oy, 2 * oy + 1),
            np.arange(-2 * ox, 2 * ox + 1),
            indexing="ij",
        )
        a = normalized["basis"]
        area = np.linalg.det(a)
        wave = 2 * np.pi * np.column_stack([m.ravel(), n.ravel()]) @ np.linalg.inv(a)
        weights = np.array(
            [
                polygon_ft(np, vertices[t], n, m, a).ravel()
                for t in self.data["triangles"]
            ]
        )
        dweights = np.array(
            [
                contour_derivative(vertices[t], velocity[t], wave, area)
                for t in self.data["triangles"]
            ]
        )
        fields, changes = np.array(fields), np.array(changes)
        tagged = material[self.data["tags"]]
        gram = tagged[:, None, None] * (fields.conj().transpose(0, 2, 1) @ fields)
        dgram = tagged[:, None, None] * (
            changes.conj().transpose(0, 2, 1) @ fields
            + fields.conj().transpose(0, 2, 1) @ changes
        )
        values, dvalues = (
            components(fields, gram, tagged),
            components(changes, dgram, np.zeros_like(tagged)),
        )
        shape = (9, 4 * oy + 1, 4 * ox + 1)
        value = (values @ weights).reshape(shape)
        per_unit = ((dvalues @ weights + values @ dweights) / length).reshape(shape)
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
