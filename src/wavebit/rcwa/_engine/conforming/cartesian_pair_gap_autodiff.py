"""Value-only exact finite-gap evaluator; NumPy/JAX x64, frozen topology.

No custom derivative or stop_gradient is used. At gap zero, positive gap
JVPs have physical right-derivative meaning; negative tangents only probe the
formal local algebraic extension and do not admit negative-gap values.
"""

import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair import _numeric_array
from wavebit.rcwa._engine.conforming.cartesian_pair import _require
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningCell
from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit._common.geometry.polygon import polygon_ft
from wavebit.rcwa._engine.conforming.material_sector import passive_pair_materials
from wavebit.rcwa._engine.conforming.material_sector import phase_from_angle
from wavebit.rcwa._engine.conforming.material_sector import minimum_rotated_storage


def _gap_weak_blocks(xp, x, x1, y, factor, thin_factor, thin, s):
    # Contract the element and Cartesian axes before allocating the global
    # blocks. An element-by-unknown-by-unknown outer product costs O(E*N^2)
    # storage although assembly only returns O(N^2). This bilinear product
    # is not Hermitian: passive complex permittivities remain unconjugated.
    def bilinear(weight, left, right):
        return (left.reshape(-1, left.shape[-1]).T
                @ (weight[:, None, None] * right).reshape(-1, right.shape[-1]))

    selected_x = xp.where(thin[:, None, None], x1, x)
    h00 = bilinear(xp.where(thin, thin_factor * s, factor), selected_x, selected_x)
    h01 = bilinear(xp.where(thin, thin_factor, factor), selected_x, y)
    h11 = bilinear(xp.where(thin, thin_factor, factor * s), y, y)
    return h00, h01, h11


class CartesianPairGapEvaluator:
    """Evaluate coefficients and closure together; obtain derivatives by AD."""

    def __init__(self, plan, order=1, *, coercivity_angle=0.0):
        endpoint = CartesianPairOpeningCell(plan, order, coercivity_angle=coercivity_angle)
        self._coercivity_angle = endpoint.coercivity_angle
        self._coercivity_phase = phase_from_angle(self._coercivity_angle)
        self._plan, self._order, self._data = (
            endpoint.plan,
            endpoint.order,
            endpoint.data,
        )
        data = self._data
        self._local_c = data["Cfull"][data["periodic"][data["triangles"]], 1:]
        self._local_z = data["Zfull"][data["periodic"][data["triangles"]]]
        self._restriction = np.eye(len(data["pmap"]))[
            data["periodic"][data["triangles"]]
        ]
        self._thin_pairs = tuple(
            (int(triangle[i]), int(triangle[j]))
            for triangle in data["triangles"][data["thin"]]
            for i in range(3)
            for j in range(i + 1, 3)
            if data["mapping"][triangle[i]] == data["mapping"][triangle[j]]
        )
        if len(self._thin_pairs) != int(np.sum(data["thin"])):
            raise ValueError(
                "each thin triangle needs exactly one repeated endpoint pair"
            )
        ox, oy = self.order
        self._n, self._m = np.meshgrid(
            np.arange(-2 * oy, 2 * oy + 1),
            np.arange(-2 * ox, 2 * ox + 1),
            indexing="ij",
        )
        n, m = np.meshgrid(
            np.arange(-oy, oy + 1), np.arange(-ox, ox + 1), indexing="ij"
        )
        self._dn, self._dm = (
            n.ravel()[:, None] - n.ravel() + 2 * oy,
            m.ravel()[:, None] - m.ravel() + 2 * ox,
        )
        for array in (
            self._local_c,
            self._local_z,
            self._restriction,
            self._n,
            self._m,
            self._dn,
            self._dm,
        ):
            array.flags.writeable = False

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

    def evaluate(
        self,
        *,
        gap,
        basis,
        widths,
        height,
        epsilon,
        background=1.0,
        offset=(0.0, 0.0),
        xp=np,
    ):
        if xp is not np and getattr(xp, "__name__", "") != "jax.numpy":
            raise ValueError("the finite-gap evaluator supports NumPy or JAX only")
        if xp.asarray(0.0).dtype != np.dtype("float64"):
            raise ValueError("the finite-gap evaluator requires x64")
        values, valid = {}, xp.asarray(True)
        for name, value, shape in (
            ("basis", basis, (2, 2)),
            ("widths", widths, (2,)),
            ("height", height, ()),
            ("gap", gap, ()),
            ("offset", offset, (2,)),
        ):
            values[name], good = _numeric_array(value, shape, name, xp, real=True)
            valid = valid & good
        materials, valid_material = passive_pair_materials(
            epsilon, background, self._coercivity_phase, xp
        )
        materials = xp.where(valid_material, materials, xp.ones(3, dtype=xp.complex128))
        a, widths, height, gap, offset = (
            values[name] for name in ("basis", "widths", "height", "gap", "offset")
        )
        determinant = xp.linalg.det(a)
        valid_basis = (
            xp.all(xp.isfinite(a)) & xp.isfinite(determinant) & (determinant > 0)
        )
        _require(valid_basis, "basis determinant must be finite and positive")
        a = xp.where(valid_basis, a, xp.eye(2))
        length = xp.sqrt(xp.where(valid_basis, determinant, 1.0))
        valid_shape = xp.all(widths > 0) & (height > 0) & (gap >= 0)
        _require(valid_shape, "positive widths/height and nonnegative gap are required")
        a, widths, height, s, offset = (
            a / length,
            widths / length,
            height / length,
            gap / length,
            offset / length,
        )
        center = xp.sum(a, axis=1) / 2 + offset
        total = xp.sum(widths) + s
        x = xp.stack([xp.asarray(0.0), widths[0], widths[0] + s, total]) - total / 2
        inner = (
            xp.vstack(
                [
                    xp.column_stack([x, xp.full(4, -height / 2)]),
                    xp.column_stack([x[::-1], xp.full(4, height / 2)]),
                ]
            )
            + center
        )
        fractional = (inner - offset) @ xp.linalg.inv(a).T
        contained = xp.all((fractional > 1e-10) & (fractional < 1 - 1e-10))
        _require(contained, "pair envelope must be strictly contained")
        x0 = xp.stack([xp.asarray(0.0), widths[0], xp.sum(widths)]) - xp.sum(widths) / 2
        inner0 = (
            xp.vstack(
                [
                    xp.column_stack([x0, xp.full(3, -height / 2)]),
                    xp.column_stack([x0[::-1], xp.full(3, height / 2)]),
                ]
            )
            + center
        )
        outer = (
            xp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) @ a.T + offset
        )
        contact = (
            xp.asarray(self.data["contact_numerators"])
            / self.data["weight_denominator"]
        ) @ xp.vstack([outer, inner0])
        zero = contact[xp.asarray(self.data["mapping"])]
        valid_geometry = (
            valid & valid_basis & valid_shape & contained & xp.all(xp.isfinite(zero))
        )
        _require(valid_geometry, "finite valid physical geometry required")
        zero = xp.where(valid_geometry, zero, xp.zeros_like(zero))
        a = xp.where(valid_geometry, a, xp.eye(2))
        safe_s = xp.where(valid_geometry, s, 0.0)
        base_velocity = xp.asarray(
            [[0.0, 0.0]] * 4
            + [[v, 0.0] for v in [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5]]
        )
        velocity = (
            xp.asarray(self.data["open_numerators"]) / self.data["weight_denominator"]
        ) @ base_velocity
        vertices = zero + safe_s * velocity
        triangles = xp.asarray(self.data["triangles"])
        p, dp = zero[triangles], velocity[triangles]
        j0 = xp.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=-1)
        j1 = xp.stack([dp[:, 1] - dp[:, 0], dp[:, 2] - dp[:, 0]], axis=-1)

        def adj(j):
            return xp.stack(
                [
                    xp.stack([j[:, 1, 1], -j[:, 0, 1]], axis=-1),
                    xp.stack([-j[:, 1, 0], j[:, 0, 0]], axis=-1),
                ],
                axis=1,
            )

        reference = xp.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        n0, n1 = (
            xp.einsum("ia,eab->eib", reference, adj(j0)),
            xp.einsum("ia,eab->eib", reference, adj(j1)),
        )
        d0 = j0[:, 0, 0] * j0[:, 1, 1] - j0[:, 0, 1] * j0[:, 1, 0]
        d1 = j1[:, 0, 0] * j0[:, 1, 1] - j1[:, 0, 1] * j0[:, 1, 0]
        thin = xp.asarray(self.data["thin"])
        d = d0 + safe_s * d1
        regular_good = xp.isfinite(d0) & (d0 > 0) & xp.isfinite(d) & (d > 0)
        slope_good = xp.isfinite(d1) & (d1 > 0)
        pair_indices = xp.asarray(self._thin_pairs)
        repeated = xp.all(zero[pair_indices[:, 0]] == zero[pair_indices[:, 1]])
        # Exact graph/topology certificate plus identical gathered live endpoints,
        # not a small-area classification, establishes the thin pole identity.
        valid_triangles = (
            xp.all(xp.where(thin, slope_good, regular_good))
            & repeated
            & xp.all(j1[:, 1] == 0)
            & xp.all(xp.isfinite(vertices))
        )
        _require(
            valid_triangles,
            "positive regular areas and exact thin opening geometry required",
        )
        safe_d, safe_d1 = xp.where(regular_good, d, 1.0), xp.where(slope_good, d1, 1.0)
        n = n0 + safe_s * n1
        c, z = xp.asarray(self._local_c), xp.asarray(self._local_z)
        x1 = xp.einsum("eia,eic->eac", n1, c)
        x = xp.einsum("eia,eic->eac", n, c)
        y = xp.einsum("eia,eic->eac", n, z)

        tagged = materials[xp.asarray(self.data["tags"])]
        factor, thin_factor = tagged / (2 * safe_d), tagged / (2 * safe_d1)

        h00, h01, h11 = _gap_weak_blocks(
            xp, x, x1, y, factor, thin_factor, thin, safe_s
        )
        fc = -xp.sum(
            tagged[:, None, None]
            * xp.where(thin[:, None, None], safe_s * x1, x).transpose(0, 2, 1)
            / 2,
            axis=0,
        )
        fz = -xp.sum(tagged[:, None, None] * y.transpose(0, 2, 1) / 2, axis=0)
        matrix, rhs = xp.block([[h00, safe_s * h01], [h01.T, h11]]), xp.vstack([fc, fz])
        valid_blocks = xp.all(xp.isfinite(matrix)) & xp.all(xp.isfinite(rhs))
        _require(valid_blocks, "finite rescaled equations required")
        nc, nz = self.data["C"].shape[1], self.data["Z"].shape[1]
        safe_h00 = xp.where(valid_blocks, h00, xp.eye(nc))
        safe_h11 = xp.where(valid_blocks, h11, xp.eye(nz))
        contact_min = xp.linalg.eigvalsh((safe_h00 + safe_h00.conj().T) / 2)[0]
        difference_min = xp.linalg.eigvalsh((safe_h11 + safe_h11.conj().T) / 2)[0]
        rotated_contact = (contact_min if self._coercivity_phase == 1.0 else
                           minimum_rotated_storage(safe_h00, self._coercivity_phase, xp))
        rotated_difference = (difference_min if self._coercivity_phase == 1.0 else
                              minimum_rotated_storage(safe_h11, self._coercivity_phase, xp))
        coercive = (rotated_contact > 0) & (rotated_difference > 0)
        _require(coercive, "contact/difference diagonal blocks must be coercive")
        valid_before_solve = (
            valid_geometry & valid_triangles & valid_material & valid_blocks & coercive
        )
        matrix = xp.where(valid_before_solve, matrix, xp.eye(nc + nz))
        rhs = xp.where(valid_before_solve, rhs, xp.zeros_like(rhs))
        solution = xp.linalg.solve(matrix, rhs)
        u, v = solution[:nc], solution[nc:]
        regular_field = xp.eye(2) + (x @ u + safe_s * (y @ v)) / safe_d[:, None, None]
        thin_field = xp.eye(2) + (x1 @ u + y @ v) / safe_d1[:, None, None]
        field = xp.where(thin[:, None, None], thin_field, regular_field)
        restriction = xp.asarray(self._restriction)
        flux = xp.einsum(
            "eij,eic->jc", restriction, tagged[:, None, None] * (n @ field) / 2
        )
        load = xp.einsum("eij,eic->jc", restriction, -tagged[:, None, None] * n / 2)
        residual = xp.linalg.norm(matrix @ solution - rhs) / xp.maximum(
            xp.linalg.norm(rhs), 1.0
        )
        original_residual = xp.linalg.norm(flux) / xp.maximum(xp.linalg.norm(load), 1.0)
        valid_residual = (
            xp.isfinite(residual)
            & xp.isfinite(original_residual)
            & (residual <= 1e-10)
            & (original_residual <= 1e-10)
        )
        _require(valid_residual, "original/projected weak residual exceeds 1e-10")
        nmode, mmode = xp.asarray(self._n), xp.asarray(self._m)
        points = vertices[triangles]
        if xp is np:
            transforms = xp.stack(
                [polygon_ft(xp, triangle, nmode, mmode, a) for triangle in points]
            )
        else:
            import jax

            transforms = jax.vmap(
                lambda triangle: polygon_ft(xp, triangle, nmode, mmode, a)
            )(points)
        gram = tagged[:, None, None] * (field.conj().transpose(0, 2, 1) @ field)
        fields = xp.stack(
            [
                field[:, 0, 0],
                field[:, 0, 1],
                field[:, 1, 0],
                field[:, 1, 1],
                gram[:, 0, 0],
                gram[:, 0, 1],
                gram[:, 1, 1],
                tagged,
                gram[:, 1, 0],
            ]
        )
        result = xp.einsum("ce,enm->cnm", fields, transforms)
        valid_coefficients = (
            valid_before_solve & valid_residual & xp.all(xp.isfinite(result))
        )
        _require(valid_coefficients, "finite coefficients required")
        ox, oy = self.order
        delta = xp.asarray((self._m == 0) & (self._n == 0))
        fallback = (
            xp.asarray([1, 0, 0, 1, 1, 0, 1, 1, 0], dtype=xp.complex128)[:, None, None]
            * delta
        )
        safe_coefficients = xp.where(valid_coefficients, result, fallback)
        coeff = safe_coefficients[:, xp.asarray(self._dn), xp.asarray(self._dm)]
        change = xp.block([[coeff[0], coeff[1]], [coeff[2], coeff[3]]])
        energy = xp.block([[coeff[4], coeff[5]], [coeff[8], coeff[6]]])
        longitudinal = coeff[7]
        slots = ConvMatrices.from_energy(change, energy, longitudinal)
        tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
        adjoint = tensor.conj().T
        storage = xp.linalg.eigvalsh((tensor + adjoint) / 2)[0]
        rotated_storage = (storage if self._coercivity_phase == 1.0 else
                           minimum_rotated_storage(tensor, self._coercivity_phase, xp))
        loss = xp.linalg.eigvalsh((tensor - adjoint) / (2j))[0]
        hermitian_error = xp.max(xp.abs(tensor - adjoint))
        lossless = xp.all(xp.imag(materials) == 0)
        inverse_residual = xp.linalg.norm(
            change.conj().T @ tensor @ change - energy
        ) / xp.maximum(xp.linalg.norm(energy), 1.0)
        z_residual = xp.linalg.norm(
            longitudinal @ slots.epz_i - xp.eye(len(longitudinal))
        )
        valid_closure = (
            xp.all(xp.isfinite(tensor))
            & (rotated_storage > 0)
            & (loss > -1e-9)
            & (~lossless | (hermitian_error < 1e-9))
            & xp.isfinite(inverse_residual)
            & (inverse_residual <= 1e-10)
            & xp.isfinite(z_residual)
            & (z_residual <= 1e-10)
        )
        _require(valid_closure, "finite closure failed physical/inverse guards")
        valid_output = valid_coefficients & valid_closure
        result = xp.where(valid_output, result, xp.nan)
        slots = ConvMatrices(
            *(xp.where(valid_output, slot, xp.nan)[None] for slot in slots)
        )
        metadata = dict(
            gap=gap,
            coercivity_angle=self.coercivity_angle,
            normalization_length=length,
            normalized_gap=s,
            open_free_dofs=nc + nz,
            contact_free_dofs=nc,
            difference_dofs=nz,
            linear_residual=residual,
            original_flux_residual=original_residual,
            contact_minimum=contact_min,
            contact_minimum_rotated_eigenvalue=rotated_contact,
            difference_minimum_rotated_eigenvalue=rotated_difference,
            difference_minimum=difference_min,
            matrix_condition=xp.linalg.cond(matrix),
            minimum_storage=storage,
            minimum_rotated_storage=rotated_storage,
            minimum_dissipation=loss,
            hermitian_error=hermitian_error,
            inverse_residual=inverse_residual,
            z_inverse_residual=z_residual,
            lossless=lossless,
            valid=valid_output,
        )
        details = dict(
            M=matrix,
            f=rhs,
            solution=solution,
            vertices=vertices,
            fields=field,
            weights=transforms,
            original_flux=flux,
            original_load=load,
        )
        return result, slots, metadata, details
