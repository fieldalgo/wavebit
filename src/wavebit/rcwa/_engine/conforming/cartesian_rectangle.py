"""Experimental periodic P1 cells for one separated Cartesian rectangle.

The physical rectangle and its surrounding parallelogram move together with
offset. A fixed ten-triangle template, uniformly refined zero, one or two
times, has 4, 19 or 79 free scalar unknowns. Strict containment and positive
template triangles are required; containment alone does not guarantee that
every rectangle orientation is supported. The public adapter exposes this
bounded finite model; no contact, multiple rectangles or optical accuracy
is claimed.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from wavebit.rcwa._engine.factorization import ConvMatrices
from wavebit._common.conventions import swap_fourier_origin
from wavebit._common.backend import _concrete
from wavebit._common.geometry.polygon import polygon_ft
from wavebit._common.geometry.polygon import rect_verts
from wavebit._common.backend import eager_under_torch_compile


def _require(condition, message):
    if _concrete(condition) is False:
        raise ValueError(message)


def _array(value, shape, name, xp, *, real=True):
    try:
        result = xp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric with shape {shape}') from error
    if result.shape != shape or result.dtype.kind not in 'iufc':
        raise ValueError(f'{name} must be numeric with shape {shape}')
    valid = xp.all(xp.isfinite(result))
    if real:
        valid = valid & xp.all(xp.imag(result) == 0)
    _require(valid, f'{name} must be finite'+(' and real' if real else ''))
    result = xp.asarray(xp.real(result) if real else result,
                        dtype=xp.float64 if real else xp.complex128)
    return result, valid


def _topology(refinement, permutation=None):
    """Fixed interpolation weights and periodic node identities, without geometry."""
    cells = [(4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i+1) % 4
        cells.extend([(i, j, i+4), (j, j+4, i+4)])
    cells, weights = np.array(cells), np.eye(8)
    if permutation is not None:
        weights = weights[np.r_[np.arange(4), np.array(permutation)+4]]
    inclusion = np.array([True, True]+[False]*8)
    for _ in range(refinement):
        midpoint_ids, new = {}, list(weights)
        def midpoint(i, j):
            edge = tuple(sorted((int(i), int(j))))
            if edge not in midpoint_ids:
                midpoint_ids[edge] = len(new)
                new.append((weights[i]+weights[j])/2)
            return midpoint_ids[edge]
        refined = []
        for a, b, c in cells:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        weights, cells = np.array(new), np.array(refined)
        inclusion = np.repeat(inclusion, 4)
    # Outer-node coordinates are exact dyadic numbers from the interpolation
    # weights. Opposite sides differ by a primitive vector for every live A.
    outer = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    lookup, periodic = {}, []
    for i, weight in enumerate(weights):
        if np.any(weight[4:]):
            key = ('interior', i)
        else:
            u, v = (weight[:4]@outer) % 1
            key = ('boundary', u, v)
        if key not in lookup:
            lookup[key] = len(lookup)
        periodic.append(lookup[key])
    periodic = np.array(periodic)
    restriction = np.eye(len(lookup))[periodic[cells]]
    for array in (weights, cells, inclusion, periodic, restriction):
        array.flags.writeable = False
    return weights, cells, inclusion, periodic, restriction


@dataclass(frozen=True)
class CartesianRectanglePlan:
    """Immutable cyclic vertex chart selected from concrete physical geometry.

    Only the permutation is retained. Creation chooses among four cyclic
    shifts of the original ten-triangle template; it neither changes the
    rectangle nor searches for a general triangulation. Equivalent angle
    representations require their own concrete plans. Replay never changes
    the selected chart, and derivatives do not include plan selection.
    """

    permutation: tuple

    def __post_init__(self):
        p = self.permutation
        if (not isinstance(p, tuple) or len(p) != 4
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in p)
                or p not in tuple(tuple((i+j) % 4 for j in range(4)) for i in range(4))):
            raise ValueError('plan permutation must be a cyclic integer tuple')
        object.__setattr__(self, 'permutation', tuple(map(int, p)))

    @classmethod
    def from_geometry(cls, *, basis, width, height, angle=0., offset=(0., 0.)):
        """Select a positive template once, outside tracing or differentiation.

        Fractional coordinates make ranking invariant under joint rotation
        and common scaling. A 1e-12 fractional tie tolerance stabilizes the
        discrete decision; it does not relax containment or positive areas.
        No differentiability or continuity across selection ties is claimed.
        """
        basis, _ = _array(getattr(basis, 'columns', basis), (2, 2), 'basis', np)
        determinant = np.linalg.det(basis)
        _require(np.isfinite(determinant) & (determinant > 0), 'basis must have positive finite determinant')
        width, _ = _array(width, (), 'width', np)
        height, _ = _array(height, (), 'height', np)
        angle, _ = _array(angle, (), 'angle', np)
        offset, _ = _array(offset, (2,), 'offset', np)
        center = basis.sum(axis=1)/2+offset
        inner = rect_verts(np, center[0], center[1], width, height, angle)
        fractional = (inner-offset)@np.linalg.inv(basis).T
        _require((width > 0) & (height > 0)
                 & np.all((fractional > 1e-10) & (fractional < 1-1e-10)),
                 'rectangle must be strictly inside its centered cell')
        # Quantization is only a deterministic tie-breaker for concrete
        # planning. Integer sums avoid roundoff reordering symmetric corners.
        keys = np.rint(fractional*1e12).astype(np.int64)
        start = min(range(4), key=lambda i: (keys[i].sum(), keys[i, 0], keys[i, 1]))
        outer = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        cells = _topology(0)[1]
        candidates = []
        for shift in range(4):
            permutation = tuple((start+shift+i) % 4 for i in range(4))
            points = np.concatenate([outer, fractional[list(permutation)]])[cells]
            jacobian = np.stack([points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]], axis=-1)
            score = np.min(np.linalg.det(jacobian))/2
            physical = np.concatenate([outer@basis.T+offset, inner[list(permutation)]])[cells]
            jacobian = np.stack([physical[:, 1]-physical[:, 0], physical[:, 2]-physical[:, 0]], axis=-1)
            physical_determinants = np.linalg.det(jacobian)
            if (np.isfinite(score) and score > 0
                    and np.all(np.isfinite(physical_determinants) & (physical_determinants > 0))):
                candidates.append((score, permutation))
        if not candidates:
            raise ValueError('no positive cyclic rectangle template was found')
        best = max(score for score, _ in candidates)
        return cls(next(p for score, p in candidates if score >= best-1e-12))


class CartesianRectangleCell:
    """Bounded experimental corrector with physical Cartesian geometry.

    ``truncation_order`` is a nonnegative integer or (x, y) Fourier half-width
    pair. It is independent of ``refinement`` (0, 1 or 2). The latter is a
    deliberately bounded auxiliary space, not an optical convergence setting.

    All geometry lengths use the same physical units; lattice vectors are
    columns. ``offset`` is Cartesian displacement from the caller's cell
    center. Translating the entire integration cell with the rectangle keeps
    the finite space translation covariant, including the Fourier phase.
    Joint rotation of lattice and rectangle is distinct from changing only
    the rectangle angle within the fixed lattice and triangle template.
    ``plan=None`` preserves the historical template. An explicit immutable
    ``CartesianRectanglePlan`` expands the available vertex charts while
    keeping topology fixed throughout live replay and differentiation.
    """

    components = ('Bxx', 'Bxy', 'Byx', 'Byy', 'Gxx', 'Gxy', 'Gyy', 'epsilon', 'Gyx')

    def __init__(self, truncation_order=1, *, refinement=0, plan=None):
        pair = ((truncation_order, truncation_order)
                if isinstance(truncation_order, Integral) else truncation_order)
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in pair)):
            raise ValueError('truncation_order must be a nonnegative integer or pair')
        if (isinstance(refinement, (bool, np.bool_)) or not isinstance(refinement, Integral)
                or refinement not in (0, 1, 2)):
            raise ValueError('refinement must be 0, 1 or 2')
        self.truncation_order = tuple(map(int, pair))
        self.refinement = int(refinement)
        if plan is not None and not isinstance(plan, CartesianRectanglePlan):
            raise ValueError('plan must be a CartesianRectanglePlan or None')
        self.plan = plan
        self._weights, self._cells, self._inclusion, self._periodic, self._restriction = _topology(
            self.refinement, None if plan is None else plan.permutation)
        self.degrees_of_freedom = self._restriction.shape[2]-1
        ox, oy = self.truncation_order
        self._x_modes, self._y_modes = np.arange(-2*ox, 2*ox+1), np.arange(-2*oy, 2*oy+1)
        n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
        self._dn = n.ravel()[:, None]-n.ravel()+2*oy
        self._dm = m.ravel()[:, None]-m.ravel()+2*ox

    def coefficients(self, *, basis, width, height, epsilon, background=1.,
                     angle=0., offset=(0., 0.), xp=np):
        """Nine exact Fourier fields of this finite P1 cell and its metadata.

        Scalars epsilon/background are physical passive permittivities with
        positive real parts. Coefficient axes are (component, y mode, x mode),
        covering exactly the differences required by the requested outer set.
        Invalid eager inputs raise; invalid traced updates produce NaNs.
        """
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise ValueError('cell construction supports NumPy and JAX only')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('cell construction requires x64')
        basis, valid = _array(getattr(basis, 'columns', basis), (2, 2), 'basis', xp)
        valid_basis = valid & xp.isfinite(xp.linalg.det(basis)) & (xp.linalg.det(basis) > 0)
        _require(valid_basis, 'basis must have positive finite determinant')
        basis = xp.where(valid_basis, basis, xp.eye(2))
        values = {}
        for name, value, shape in (('width', width, ()), ('height', height, ()),
                                   ('angle', angle, ()), ('offset', offset, (2,))):
            values[name], good = _array(value, shape, name, xp)
            valid = valid & good
        width, height, angle, offset = (values[k] for k in ('width', 'height', 'angle', 'offset'))
        center = xp.sum(basis, axis=1)/2+offset
        outer = xp.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])@basis.T+offset
        inner = rect_verts(xp, center[0], center[1], width, height, angle)
        fractional = (inner-offset)@xp.linalg.inv(basis).T
        valid_shape = ((width > 0) & (height > 0)
                       & xp.all((fractional > 1e-10) & (fractional < 1-1e-10)))
        _require(valid_shape, 'rectangle must be strictly inside its centered cell')
        vertices = xp.asarray(self._weights)@xp.concatenate([outer, inner])
        points = vertices[xp.asarray(self._cells)]
        jacobian = xp.stack([points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]], axis=-1)
        determinant = xp.linalg.det(jacobian)
        valid_triangles = xp.all(xp.isfinite(determinant) & (determinant > 0))
        _require(valid_triangles, 'fixed template has nonpositive triangles')
        jacobian = xp.where((determinant > 0)[:, None, None], jacobian, xp.eye(2))
        areas = determinant/2
        gradient = xp.asarray([[-1., -1.], [1., 0.], [0., 1.]])@xp.linalg.inv(jacobian)
        epsilon, valid_epsilon = _array(epsilon, (), 'epsilon', xp, real=False)
        background, valid_background = _array(background, (), 'background', xp, real=False)
        valid_material = (valid_epsilon & valid_background & (xp.real(epsilon) > 0)
                          & (xp.imag(epsilon) >= 0) & (xp.real(background) > 0)
                          & (xp.imag(background) >= 0))
        _require(valid_material, 'physical scalar materials must be passive with positive real parts')
        material = xp.where(xp.asarray(self._inclusion), epsilon, background)
        material = xp.where(valid_material, material, xp.ones_like(material))
        grad_global = xp.einsum('eia,eij->eja', gradient, xp.asarray(self._restriction))
        matrix = xp.einsum('e,eia,eja->ij', material*areas, grad_global, grad_global)
        right = -xp.einsum('e,eia->ia', material*areas, grad_global)
        scale = 1/xp.sqrt(xp.abs(xp.diag(matrix)[1:]))
        reduced = scale[:, None]*matrix[1:, 1:]*scale[None]
        solution = scale[:, None]*xp.linalg.solve(reduced, scale[:, None]*right[1:])
        solution = xp.concatenate([xp.zeros((1, 2), dtype=xp.complex128), solution])
        field = xp.eye(2)+xp.einsum('eia,ic->eac', grad_global, solution)
        gram = material[:, None, None]*(xp.swapaxes(xp.conj(field), -1, -2)@field)
        n, m = xp.meshgrid(xp.asarray(self._y_modes), xp.asarray(self._x_modes), indexing='ij')
        if xp is np:
            transforms = xp.stack([polygon_ft(xp, triangle, n, m, basis) for triangle in points])
        else:
            import jax
            transforms = jax.vmap(lambda triangle: polygon_ft(xp, triangle, n, m, basis))(points)
        fields = xp.stack([field[:, 0, 0], field[:, 0, 1], field[:, 1, 0], field[:, 1, 1],
                           gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1], material, gram[:, 1, 0]])
        result = xp.einsum('ce,enm->cnm', fields, transforms)
        residual = xp.linalg.norm(matrix@solution-right)/xp.maximum(xp.linalg.norm(right), 1.)
        valid = valid & valid_basis & valid_shape & valid_triangles & valid_material
        valid = valid & xp.isfinite(residual) & (residual <= 1e-10) & xp.all(xp.isfinite(result))
        _require(valid, 'invalid frozen P1 cell or linear residual')
        metadata = dict(
            cell_residual=residual, degrees_of_freedom=self.degrees_of_freedom,
            refinement=self.refinement, elements=len(self._cells), vertices=len(self._weights),
            periodic_dofs=self.degrees_of_freedom+1, periodic_pairs_per_axis=(2**self.refinement+1,)*2,
            area=xp.linalg.det(basis), inclusion_area=xp.sum(xp.where(xp.asarray(self._inclusion), areas, 0.)),
            minimum_signed_area=xp.min(areas), truncation_order=self.truncation_order)
        if self.plan is not None:
            metadata['inner_permutation'] = self.plan.permutation
        return xp.where(valid, result, xp.nan), metadata

    def conv_matrices(self, **settings):
        """One layer of energy-closure slots on exactly the requested Fourier set."""
        coefficients, _ = self.coefficients(**settings)
        xp = settings.get('xp', np)
        c = coefficients[:, xp.asarray(self._dn), xp.asarray(self._dm)]
        change = xp.block([[c[0], c[1]], [c[2], c[3]]])
        gram = xp.block([[c[4], c[5]], [c[8], c[6]]])
        slots = ConvMatrices.from_energy(change, gram, c[7])
        tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
        adjoint = tensor.conj().T
        storage, dissipation = (tensor+adjoint)/2, (tensor-adjoint)/(2j)
        lossless = ((xp.imag(xp.asarray(settings['epsilon'])) == 0)
                    & (xp.imag(xp.asarray(settings.get('background', 1.))) == 0))
        valid = ((~lossless | (xp.max(xp.abs(tensor-adjoint)) < 1e-9))
                 & (xp.linalg.eigvalsh(storage)[0] > 0)
                 & (xp.linalg.eigvalsh(dissipation)[0] > -1e-9))
        _require(valid, 'Cartesian cell energy closure failed its passivity guard')
        return ConvMatrices(*(xp.where(valid, slot, xp.nan)[None] for slot in slots))


@dataclass(frozen=True)
class _CartesianRectangleNVFPlan:
    """Structural layer choices only; None means a uniform layer."""

    layers: tuple


@eager_under_torch_compile
class CartesianRectangleNVF:
    """Experimental finite NVF for one separated physical rectangle per layer.

    Layers use ordinary ``background``/``shapes`` dictionaries. A layer with
    no shapes is uniform; a patterned layer must contain exactly one
    rectangle. Physical ``lx``/``ly`` lengths, the ordinary ``l`` fallback,
    laboratory angle and Cartesian ``cx``/``cy`` offsets retain their meaning
    on the caller's unreduced lattice. The whole integration cell follows
    the offset. The centered rectangle must fit strictly inside that cell,
    with positive finite triangles on its selected cyclic vertex chart.

    Call ``plan`` on concrete geometry and explicitly pass its immutable
    result to every ``conv_matrices`` replay. Counts and vertex choices stay
    fixed; numerical lattice, pose and scalar materials remain live in
    NumPy/JAX x64. Replan explicitly when changing parameter charts. Replay
    validity checks do not detect every equivalent chart change.

    ``refinement`` is 0, 1 or 2 (4, 19 or 79 scalar unknowns), independent
    of the exact integer or (x, y) Fourier half widths. Materials must be
    finite, nonmagnetic, scalar and passive with positive real permittivity.
    Invalid eager values raise; invalid traced values poison the affected
    layer, and invalid lattice values poison all layers. This explicit
    finite model does not certify optical accuracy or admit contact,
    multiple shapes, general triangulations or arbitrary periodic overlaps.
    It does not change ConformingNVF or AffineProfileNVF admission.
    """

    def __init__(self, truncation_order=1, *, refinement=0):
        prototype = CartesianRectangleCell(truncation_order, refinement=refinement)
        self.truncation_order = prototype.truncation_order
        self.refinement = prototype.refinement
        ox, oy = self.truncation_order
        self._count = (2*ox+1)*(2*oy+1)
        self._cells = {}

    @staticmethod
    def _layers(structure, lattice, xp):
        from wavebit.rcwa._engine.conforming import _epsilon
        from wavebit._common.geometry.recipes import check_layer_spec
        from wavebit._common.geometry.recipes import check_shape_spec

        basis, valid_basis = _array(getattr(lattice, 'columns', lattice), (2, 2), 'lattice', xp)
        determinant = xp.linalg.det(basis)
        valid_basis = valid_basis & xp.isfinite(determinant) & (determinant > 0)
        _require(valid_basis, 'Cartesian rectangle lattice must have positive finite determinant')
        basis = xp.where(valid_basis, basis, xp.eye(2))
        if not isinstance(structure, (tuple, list)) or not structure:
            raise ValueError('Cartesian rectangle NVF needs a nonempty sequence of layer dictionaries')
        layers = []
        for layer_index, layer in enumerate(structure):
            if not isinstance(layer, dict):
                raise ValueError('each Cartesian rectangle NVF layer must be a dictionary')
            check_layer_spec(layer, layer_index)
            background = _epsilon(layer.get('background', 1.), xp)
            shapes = layer.get('shapes', [])
            if not isinstance(shapes, (tuple, list)):
                raise ValueError('Cartesian rectangle NVF shapes must be a sequence of dictionaries')
            if len(shapes) > 1:
                raise NotImplementedError('Cartesian rectangle NVF supports one rectangle per patterned layer')
            geometry = None
            if shapes:
                shape = shapes[0]
                if not isinstance(shape, dict):
                    raise ValueError('each Cartesian rectangle NVF shape must be a dictionary')
                check_shape_spec(shape, 0, layer_index)
                if shape.get('type', 'rectangle') != 'rectangle':
                    raise NotImplementedError('Cartesian rectangle NVF supports only rectangle shapes')
                # Match shape_geometry's ordinary rectangle alias precedence.
                # cx/cy already are offsets; adding the cell center here would
                # apply the origin twice in the physical cell's Fourier phase.
                geometry = dict(width=shape.get('lx', shape.get('l', 0.)),
                                height=shape.get('ly', shape.get('l', 0.)),
                                angle=shape.get('angle', 0.),
                                offset=xp.stack([xp.asarray(shape.get('cx', 0.)),
                                                 xp.asarray(shape.get('cy', 0.))]),
                                epsilon=_epsilon(shape.get('material', 1.), xp))
            layers.append((geometry, background))
        return layers, basis, valid_basis

    def plan(self, structure, lattice):
        """Freeze only layer kinds and cyclic choices on concrete geometry.

        Planning assembles no weak system. Numeric dimensions, positions,
        lattice entries and materials are read again during every replay.
        """
        layers, basis, _ = self._layers(structure, lattice, np)
        choices = []
        for geometry, _ in layers:
            if geometry is None:
                choices.append(None)
            else:
                choices.append(CartesianRectanglePlan.from_geometry(
                    basis=basis, **{key: value for key, value in geometry.items() if key != 'epsilon'}))
        return _CartesianRectangleNVFPlan(tuple(choices))

    def conv_matrices(self, structure, lattice, *, plan, xp=np):
        """Return five complex128 layer slots on the caller's exact Fourier set.

        Pass the same original lattice and truncation order to solve_conv.
        The physical cell applies the materials in the corner frame; the returned
        operators are restated about the cell centre, as solve_conv reads them.
        Do not add a phase to the returned slots.
        ``plan`` is mandatory, including for uniform stacks.
        """
        from wavebit.rcwa._engine.geometry import StaleGeometryError

        if getattr(xp, '__name__', '') == 'array_api_compat.numpy':
            xp = np
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise NotImplementedError('Cartesian rectangle construction supports NumPy and JAX x64')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('Cartesian rectangle construction requires 64-bit precision')
        if (not isinstance(plan, _CartesianRectangleNVFPlan) or not isinstance(plan.layers, tuple)
                or any(choice is not None and not isinstance(choice, CartesianRectanglePlan) for choice in plan.layers)):
            raise StaleGeometryError('Cartesian rectangle NVF plan type is invalid; plan again')
        layers, basis, valid_basis = self._layers(structure, lattice, xp)
        if len(plan.layers) != len(layers):
            raise StaleGeometryError('Cartesian rectangle NVF layer count changed; plan again')
        stacks = []
        for (geometry, background), choice in zip(layers, plan.layers):
            if (geometry is None) != (choice is None):
                raise StaleGeometryError('Cartesian rectangle NVF shape count changed; plan again')
            valid = valid_basis & xp.isfinite(background)
            if geometry is None:
                identity = xp.eye(self._count, dtype=xp.complex128)
                slots = ConvMatrices(background*identity, background*identity, identity/background,
                                     xp.zeros_like(identity), xp.zeros_like(identity))
            else:
                if choice not in self._cells:
                    self._cells[choice] = CartesianRectangleCell(
                        self.truncation_order, refinement=self.refinement, plan=choice)
                layer_slots = self._cells[choice].conv_matrices(
                    basis=basis, background=background, xp=xp, **geometry)
                slots = ConvMatrices(*(slot[0] for slot in layer_slots))
            stacks.append(tuple(xp.where(valid, slot, xp.nan) for slot in slots))
        corner = ConvMatrices(*(xp.stack([layer[i] for layer in stacks]) for i in range(5)))
        return swap_fourier_origin(corner, *((0, n) for n in self.truncation_order))
