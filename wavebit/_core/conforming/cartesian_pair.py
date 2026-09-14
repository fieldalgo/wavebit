"""Experimental finite P1 cells for a horizontal equal-height Cartesian pair.

A positive gap and exact full-height contact use different immutable charts.
The physical pair and its fundamental cell translate together. Only strictly
contained, positive-triangle charts and refinement 0 through 3 are admitted. This
module provides finite coefficients and an explicit experimental adapter.
Opening-gap derivatives and optical convergence remain unqualified.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..backend import _concrete
from ..factorization import ConvMatrices
from ..polygon import polygon_ft

__all__ = ['CartesianPairPlan', 'CartesianPairCell', 'CartesianRectanglePairNVF']

_UNIT = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])



def _require(condition, message):
    if _concrete(condition) is False:
        raise ValueError(message)


def _numeric_array(value, shape, name, xp, *, real):
    try:
        result = xp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be numeric with shape {shape}') from error
    if result.shape != shape or result.dtype.kind not in ('iuf' if real else 'iufc'):
        raise ValueError(f'{name} must be {"real " if real else ""}numeric with shape {shape}')
    valid = xp.all(xp.isfinite(result))
    if not real:
        valid = valid & xp.all((xp.real(result) > 0) & (xp.imag(result) >= 0))
    _require(valid, f'{name} must be finite'+(' and passive with positive real parts' if not real else ''))
    return xp.asarray(result, dtype=xp.float64 if real else xp.complex128), valid


def _planning_array(value, shape, name):
    """Planning is concrete-only; live tracing belongs to the frozen cell."""
    try:
        concrete = np.asarray(value)
    except Exception as error:
        raise ValueError('pair planning requires concrete geometry outside JAX transformations') from error
    return _numeric_array(concrete, shape, name, np, real=True)[0]



def _physical_geometry(basis, widths, height, gap, offset=(0., 0.)):
    basis = _planning_array(basis, (2, 2), 'basis')
    widths = _planning_array(widths, (2,), 'widths')
    offset = _planning_array(offset, (2,), 'offset')
    height = _planning_array(height, (), 'height')
    gap = _planning_array(gap, (), 'gap')
    determinant = np.linalg.det(basis)
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError('basis determinant must be finite and positive')
    if min(widths) <= 0 or height <= 0 or gap < 0:
        raise ValueError('widths/height must be positive and gap nonnegative')
    contact = gap == 0.
    total = widths.sum()+gap
    xs = (np.array([0., widths[0], total]) if contact
          else np.array([0., widths[0], widths[0]+gap, total]))-total/2
    center = basis.sum(axis=1)/2+offset
    bottom = np.column_stack([xs, np.full(len(xs), -height/2)])
    top = np.column_stack([xs[::-1], np.full(len(xs), height/2)])
    inner = np.vstack([bottom, top])+center
    fractional = (inner-offset)@np.linalg.inv(basis).T
    if not np.all((fractional > 1e-10) & (fractional < 1-1e-10)):
        raise ValueError('pair envelope must be strictly contained in the translated cell')
    vertices = np.vstack([_UNIT@basis.T+offset, inner])
    if not np.all(np.isfinite(vertices)):
        raise ValueError('physical vertices must be finite')
    return vertices, contact


def _base_topology(contact, shift):
    nx = 3 if contact else 4
    corners = [0, nx-1, nx, 2*nx-1]
    corners = [corners[(i+shift) % 4] for i in range(4)]
    cells, tags = [], []
    for band in range(nx-1):
        a, b, c, d = 4+band, 5+band, 4+2*nx-2-band, 4+2*nx-1-band
        cells.extend([(a, b, c), (a, c, d)])
        tags.extend([([1, 2] if contact else [1, 0, 2])[band]]*2)
    for side in range(4):
        other = (side+1) % 4
        first, last = corners[side], corners[other]
        cells.append((side, other, 4+first))
        tags.append(0)
        current = first
        while current != last:
            following = (current+1) % (2*nx)
            cells.append((other, 4+following, 4+current))
            tags.append(0)
            current = following
    return np.array(cells, int), np.array(tags, int)


def _signed_twice_areas(points):
    return np.linalg.det(np.stack([points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]], axis=-1))


def _refined_topology(contact, shift, refinement):
    """Dyadic node graph; one extra level supplies P2 edge nodes only.

    Physical P1 plans retain their 0..3 contract. A quadratic potential on
    refinement three needs the level-four nodes, but not a level-four P1 cell.
    """
    if (isinstance(refinement, (bool, np.bool_)) or not isinstance(refinement, Integral)
            or refinement not in range(5)):
        raise ValueError('internal nodal refinement must be in 0..4')
    cells, tags = _base_topology(contact, shift)
    weights = np.eye(10 if contact else 12)
    for _ in range(refinement):
        midpoint_ids, new = {}, list(weights)
        def midpoint(a, b):
            key = tuple(sorted((int(a), int(b))))
            if key not in midpoint_ids:
                midpoint_ids[key] = len(new)
                new.append((weights[a]+weights[b])/2)
            return midpoint_ids[key]
        refined = []
        for a, b, c in cells:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        weights, cells, tags = np.array(new), np.array(refined), np.repeat(tags, 4)
    lookup, periodic = {}, []
    for i, row in enumerate(weights):
        key = ('interior', i) if np.any(row[4:]) else ('boundary', *(row[:4]@_UNIT % 1))
        if key not in lookup:
            lookup[key] = len(lookup)
        periodic.append(lookup[key])
    periodic = np.array(periodic)
    for array in (weights, cells, tags, periodic):
        array.flags.writeable = False
    return weights, cells, tags, periodic


@dataclass(frozen=True)
class CartesianPairPlan:
    """Concrete cyclic annulus and contact state, fixed during live replay.

    ``from_geometry`` selects one of four cyclic charts. It stores no lattice,
    width, material or offset value. Planning under JIT is refused; creating a
    new chart is an explicit operation outside differentiation. Refinement is
    restricted to 0 through 3. Contact means an exact full-height shared edge and
    excludes derivatives that open the gap.
    """

    contact: bool
    shift: int
    refinement: int = 0

    def __post_init__(self):
        if not isinstance(self.contact, (bool, np.bool_)):
            raise ValueError('contact must be a concrete bool')
        for value, allowed in ((self.shift, range(4)), (self.refinement, range(4))):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in allowed:
                raise ValueError('shift and refinement must be integers from 0 through 3')

    @classmethod
    def from_geometry(cls, *, refinement=0, **geometry):
        vertices, contact = _physical_geometry(**geometry)
        basis, offset = np.asarray(geometry['basis']), np.asarray(geometry.get('offset', (0., 0.)))
        fractional = (vertices-offset)@np.linalg.inv(basis).T
        candidates = []
        for shift in range(4):
            cells, _ = _base_topology(contact, shift)
            physical = _signed_twice_areas(vertices[cells])
            score = min(_signed_twice_areas(fractional[cells]))/2
            if np.isfinite(score) and score > 0 and np.all(np.isfinite(physical) & (physical > 0)):
                candidates.append((score, shift))
        if not candidates:
            raise ValueError('no positive cyclic pair chart found')
        best = max(score for score, _ in candidates)
        return cls(contact, next(shift for score, shift in candidates if score >= best-1e-12), refinement)

    def topology(self):
        return _refined_topology(self.contact, self.shift, self.refinement)

    def replay(self, **geometry):
        control, contact = _physical_geometry(**geometry)
        if contact != self.contact:
            raise ValueError('contact/opening changes topology; create an explicit new plan')
        weights, cells, tags, periodic = self.topology()
        vertices = weights@control
        determinant = _signed_twice_areas(vertices[cells])
        if not np.all(np.isfinite(determinant) & (determinant > 0)):
            raise ValueError('frozen pair chart contains nonpositive triangles')
        return vertices, cells, tags, periodic, weights


class CartesianPairCell:
    """Live NumPy/JAXx64 finite coefficients on an explicit CartesianPairPlan.

    Basis vectors are columns; every length has the same physical units.
    ``widths`` and ``epsilon`` are left/right pairs. The two rectangles have
    one common height and Cartesian offset from the original cell center.
    ``order`` is a nonnegative Fourier half-width or (x,y) pair. Exact field
    differences are selected before closure inverses. Invalid eager values
    raise and invalid traced updates poison all coefficients and slots.
    """
    components = ('Bxx', 'Bxy', 'Byx', 'Byy', 'Gxx', 'Gxy', 'Gyy', 'epsilon', 'Gyx')

    def __init__(self, plan, order=1):
        if not isinstance(plan, CartesianPairPlan):
            raise ValueError('an explicit frozen CartesianPairPlan is required')
        pair = (order, order) if isinstance(order, Integral) else order
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0 for n in pair)):
            raise ValueError('order must be a nonnegative integer or pair')
        self.plan, self.order = plan, tuple(map(int, pair))
        self._weights, self._cells, self._tags, self._periodic = plan.topology()
        self._restriction = np.eye(self._periodic.max()+1)[self._periodic[self._cells]]
        self.degrees_of_freedom = int(self._periodic.max())
        ox, oy = self.order
        self._n, self._m = np.meshgrid(np.arange(-2*oy, 2*oy+1), np.arange(-2*ox, 2*ox+1), indexing='ij')
        n, m = np.meshgrid(np.arange(-oy, oy+1), np.arange(-ox, ox+1), indexing='ij')
        self._dn, self._dm = n.ravel()[:, None]-n.ravel()+2*oy, m.ravel()[:, None]-m.ravel()+2*ox

    def _geometry(self, *, basis, widths, height, gap, offset, xp):
        valid = xp.asarray(True)
        values = {}
        for name, value, shape in (('basis', basis, (2, 2)), ('widths', widths, (2,)),
                                   ('height', height, ()), ('gap', gap, ()), ('offset', offset, (2,))):
            values[name], good = _numeric_array(value, shape, name, xp, real=True)
            valid = valid & good
        basis, widths, height, gap, offset = (values[name] for name in ('basis', 'widths', 'height', 'gap', 'offset'))
        determinant = xp.linalg.det(basis)
        valid_basis = xp.all(xp.isfinite(basis)) & xp.isfinite(determinant) & (determinant > 0)
        _require(valid_basis, 'basis determinant must be positive and finite')
        basis = xp.where(valid_basis, basis, xp.eye(2))
        chart_valid = (gap == 0) if self.plan.contact else (gap > 0)
        valid_shape = xp.all(widths > 0) & (height > 0) & chart_valid
        _require(valid_shape, 'positive widths/height and the frozen contact/open chart are required')
        total = xp.sum(widths)+gap
        if self.plan.contact:
            x = xp.stack([xp.asarray(0.), widths[0], total])-total/2
        else:
            x = xp.stack([xp.asarray(0.), widths[0], widths[0]+gap, total])-total/2
        center = xp.sum(basis, axis=1)/2+offset
        inner = xp.vstack([xp.column_stack([x, xp.full(len(x), -height/2)]),
                           xp.column_stack([x[::-1], xp.full(len(x), height/2)])])+center
        fractional = (inner-offset)@xp.linalg.inv(basis).T
        contained = xp.all((fractional > 1e-10) & (fractional < 1-1e-10))
        _require(contained, 'pair envelope must be strictly contained in the translated cell')
        outer = xp.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])@basis.T+offset
        vertices = xp.asarray(self._weights)@xp.vstack([outer, inner])
        points = vertices[xp.asarray(self._cells)]
        jacobian = xp.stack([points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]], axis=-1)
        determinants = xp.linalg.det(jacobian)
        positive = xp.isfinite(determinants) & (determinants > 0)
        valid_triangles = xp.all(xp.isfinite(vertices)) & xp.all(positive)
        _require(valid_triangles, 'frozen chart requires finite positive-area triangles')
        safe_jacobian = xp.where(positive[:, None, None], jacobian, xp.eye(2))
        safe_areas = xp.where(positive, determinants/2, 1.)
        return basis, points, safe_jacobian, safe_areas, valid & valid_basis & valid_shape & contained & valid_triangles

    def coefficients(self, *, basis, widths, height, gap, epsilon, background=1., offset=(0., 0.), xp=np):
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise ValueError('the pair cell supports NumPy or JAX only')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('the pair cell requires x64')
        basis, points, jacobian, areas, valid = self._geometry(
            basis=basis, widths=widths, height=height, gap=gap, offset=offset, xp=xp)
        epsilon, good_epsilon = _numeric_array(epsilon, (2,), 'epsilon', xp, real=False)
        background, good_background = _numeric_array(background, (), 'background', xp, real=False)
        valid_material = good_epsilon & good_background
        materials = xp.concatenate([xp.reshape(background, (1,)), epsilon])
        material = xp.where(valid_material, materials[xp.asarray(self._tags)], 1.+0j)
        gradient = xp.asarray([[-1., -1.], [1., 0.], [0., 1.]])@xp.linalg.inv(jacobian)
        global_gradient = xp.einsum('eia,eij->eja', gradient, xp.asarray(self._restriction))
        matrix = xp.einsum('e,eia,eja->ij', material*areas, global_gradient, global_gradient)
        right = -xp.einsum('e,eia->ia', material*areas, global_gradient)
        scale = 1/xp.sqrt(xp.abs(xp.diag(matrix)[1:]))
        reduced = scale[:, None]*matrix[1:, 1:]*scale[None]
        solution = scale[:, None]*xp.linalg.solve(reduced, scale[:, None]*right[1:])
        solution = xp.vstack([xp.zeros((1, 2), dtype=xp.complex128), solution])
        residual = xp.linalg.norm(matrix@solution-right)/xp.maximum(xp.linalg.norm(right), 1.)
        valid_residual = xp.isfinite(residual) & (residual <= 1e-10)
        _require(valid_residual, 'original weak residual exceeds 1e-10')
        field = xp.eye(2)+xp.einsum('eia,ic->eac', global_gradient, solution)
        gram = material[:, None, None]*(field.conj().transpose(0, 2, 1)@field)
        n, m = xp.asarray(self._n), xp.asarray(self._m)
        if xp is np:
            transforms = xp.stack([polygon_ft(xp, triangle, n, m, basis) for triangle in points])
        else:
            import jax
            transforms = jax.vmap(lambda triangle: polygon_ft(xp, triangle, n, m, basis))(points)
        fields = xp.stack([field[:, 0, 0], field[:, 0, 1], field[:, 1, 0], field[:, 1, 1],
                           gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1], material, gram[:, 1, 0]])
        result = xp.einsum('ce,enm->cnm', fields, transforms)
        valid = valid & valid_material & valid_residual & xp.all(xp.isfinite(result))
        _require(valid, 'invalid finite-cell coefficients')
        return xp.where(valid, result, xp.nan), dict(cell_residual=residual,
            degrees_of_freedom=self.degrees_of_freedom, elements=len(points), contact=self.plan.contact,
            shift=self.plan.shift, refinement=self.plan.refinement, order=self.order,
            area=xp.linalg.det(basis), minimum_signed_area=xp.min(areas))

    def conv_matrices(self, **settings):
        coefficients, metadata = self.coefficients(**settings)
        xp = settings.get('xp', np)
        c = coefficients[:, xp.asarray(self._dn), xp.asarray(self._dm)]
        change = xp.block([[c[0], c[1]], [c[2], c[3]]])
        gram = xp.block([[c[4], c[5]], [c[8], c[6]]])
        slots = ConvMatrices.from_energy(change, gram, c[7])
        tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
        adjoint = tensor.conj().T
        storage, dissipation = (tensor+adjoint)/2, (tensor-adjoint)/(2j)
        min_storage, min_dissipation = xp.linalg.eigvalsh(storage)[0], xp.linalg.eigvalsh(dissipation)[0]
        hermitian_error = xp.max(xp.abs(tensor-adjoint))
        lossless = (xp.all(xp.imag(xp.asarray(settings['epsilon'])) == 0)
                    & xp.all(xp.imag(xp.asarray(settings.get('background', 1.))) == 0))
        valid = (xp.all(xp.isfinite(tensor)) & (min_storage > 0) & (min_dissipation > -1e-9)
                 & (~lossless | (hermitian_error < 1e-9)))
        _require(valid, 'finite-cell closure failed its storage/dissipation guard')
        metadata.update(minimum_storage_eigenvalue=min_storage,
                        minimum_dissipation_eigenvalue=min_dissipation,
                        hermitian_error=hermitian_error, lossless=lossless)
        return ConvMatrices(*(xp.where(valid, slot, xp.nan)[None] for slot in slots)), metadata


@dataclass(frozen=True)
class _PairLayerPlan:
    """Input indices ordered left-to-right, and the frozen finite-cell chart."""

    order: tuple
    cell: CartesianPairPlan

    def __post_init__(self):
        if (not isinstance(self.order, tuple) or self.order not in ((0, 1), (1, 0))
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in self.order)
                or not isinstance(self.cell, CartesianPairPlan)):
            raise ValueError('pair layer plan must hold two input indices and a finite-cell chart')


@dataclass(frozen=True)
class _CartesianRectanglePairNVFPlan:
    """Structural layer choices only; None denotes a uniform layer."""

    layers: tuple


class CartesianRectanglePairNVF:
    """Experimental finite NVF for two horizontal equal-height rectangles.

    Each ordinary background/shapes layer is uniform or contains exactly two
    rectangles. Physical lx/ly lengths use the ordinary l fallback; cx/cy are
    Cartesian offsets from the original cell center. Angles must equal zero.
    No lattice reduction or shearing of physical edges occurs.

    Plan concrete geometry once and pass that immutable plan to every replay.
    Planning records the original input indices in left-to-right order, the
    exact contact state, cyclic mesh choice and refinement (0 through 3). Replay
    never sorts shapes. Heights and cy must be exactly equal; the computed gap
    must stay strictly positive or exactly zero as planned. Nothing is snapped
    to contact. Changes in input ordering, contact state or layer kinds require
    explicit replanning. Strict containment and positive triangles still apply.

    Shared height/y, widths, materials, lattice and offset remain live under
    NumPy/JAX x64. Contact derivatives must preserve the shared edge. Derivatives
    that independently change one height/y or open a contact gap are outside
    this constrained geometry family. A mathematically intended contact whose
    floating-point center arithmetic leaves a nonzero gap is not exact contact
    for this adapter. This finite model supplies no optical convergence bound.
    Existing ConformingNVF, AffineProfileNVF and CartesianRectangleNVF behavior
    is unchanged.
    """

    def __init__(self, truncation_order=1, *, refinement=0):
        prototype = CartesianPairCell(CartesianPairPlan(False, 0, refinement), truncation_order)
        self.truncation_order = prototype.order
        self.refinement = prototype.plan.refinement
        self._count = (2*self.truncation_order[0]+1)*(2*self.truncation_order[1]+1)
        self._cells = {}

    @staticmethod
    def _layers(structure, lattice, xp, *, planning=False):
        from . import _epsilon
        from ..parametric import check_layer_spec, check_shape_spec

        def real(value, shape, name):
            if planning:
                return _planning_array(value, shape, name), xp.asarray(True)
            return _numeric_array(value, shape, name, xp, real=True)

        basis, valid_basis = real(getattr(lattice, 'columns', lattice), (2, 2), 'lattice')
        determinant = xp.linalg.det(basis)
        valid_basis = valid_basis & xp.isfinite(determinant) & (determinant > 0)
        _require(valid_basis, 'Cartesian rectangle pair lattice must have positive finite determinant')
        basis = xp.where(valid_basis, basis, xp.eye(2))
        if not isinstance(structure, (tuple, list)) or not structure:
            raise ValueError('Cartesian rectangle pair NVF needs a nonempty sequence of layer dictionaries')
        layers = []
        for layer_index, layer in enumerate(structure):
            if not isinstance(layer, dict):
                raise ValueError('each Cartesian rectangle pair layer must be a dictionary')
            check_layer_spec(layer, layer_index)
            background = _epsilon(layer.get('background', 1.), xp)
            supplied = layer.get('shapes', [])
            if not isinstance(supplied, (tuple, list)):
                raise ValueError('Cartesian rectangle pair shapes must be a sequence')
            if len(supplied) not in (0, 2):
                raise NotImplementedError('Cartesian rectangle pair NVF needs exactly two shapes in a patterned layer')
            shapes = []
            for shape_index, shape in enumerate(supplied):
                if not isinstance(shape, dict):
                    raise ValueError('each Cartesian rectangle pair shape must be a dictionary')
                check_shape_spec(shape, shape_index, layer_index)
                if shape.get('type', 'rectangle') != 'rectangle':
                    raise NotImplementedError('Cartesian rectangle pair NVF supports rectangle shapes only')
                values, valid = {}, xp.asarray(True)
                for name, value in (('width', shape.get('lx', shape.get('l', 0.))),
                                     ('height', shape.get('ly', shape.get('l', 0.))),
                                     ('cx', shape.get('cx', 0.)), ('cy', shape.get('cy', 0.)),
                                     ('angle', shape.get('angle', 0.))):
                    values[name], good = real(value, (), name)
                    valid = valid & good
                valid = valid & (values['width'] > 0) & (values['height'] > 0) & (values['angle'] == 0)
                _require(valid, 'Cartesian pair rectangles need positive lengths and exactly zero angle')
                values.update(epsilon=_epsilon(shape.get('material', 1.), xp), valid=valid)
                shapes.append(values)
            layers.append((shapes, background))
        return layers, basis, valid_basis

    @staticmethod
    def _pair_geometry(shapes, order, xp, choice=None):
        left, right = (shapes[index] for index in order)
        gap = (right['cx']-left['cx'])-(left['width']+right['width'])/2
        offset_x = (left['cx']+right['cx'])/2+(right['width']-left['width'])/4
        valid = (left['valid'] & right['valid'] & (left['cx'] < right['cx']) & (gap >= 0)
                 & (left['height'] == right['height']) & (left['cy'] == right['cy']))
        if choice is not None:
            valid = valid & ((gap == 0) if choice.contact else (gap > 0))
        _require(valid, 'pair replay requires frozen left/right order, equal height/y and unchanged contact state')
        geometry = dict(widths=xp.stack([left['width'], right['width']]), height=left['height'],
                        gap=gap, offset=xp.stack([offset_x, left['cy']]),
                        epsilon=xp.stack([left['epsilon'], right['epsilon']]))
        return geometry, valid

    def plan(self, structure, lattice):
        """Freeze input ordering and finite-cell topology from concrete values.

        Planning assembles no weak system and stores no live geometry/material
        values. Concrete shape sorting occurs only here, without mutating the
        caller's sequence. Traced planning is refused.
        """
        try:
            layers, basis, _ = self._layers(structure, lattice, np, planning=True)
        except Exception as error:
            raise ValueError(f'pair planning requires valid concrete inputs before JIT: {error}') from error
        choices = []
        for shapes, _ in layers:
            if not shapes:
                choices.append(None)
                continue
            order = tuple(sorted(range(2), key=lambda index: float(shapes[index]['cx'])))
            geometry, _ = self._pair_geometry(shapes, order, np)
            cell = CartesianPairPlan.from_geometry(
                basis=basis, refinement=self.refinement,
                **{key: value for key, value in geometry.items() if key != 'epsilon'})
            choices.append(_PairLayerPlan(order, cell))
        return _CartesianRectanglePairNVFPlan(tuple(choices))

    def conv_matrices(self, structure, lattice, *, plan, xp=np):
        """Five complex128 layer slots on the original exact Fourier set.

        Plan is mandatory, including uniform stacks. Pass the same original
        lattice and truncation order to solve_conv. Invalid lattice data poisons
        every layer under tracing; other invalid numerical inputs poison the
        affected layer. The physical cell already supplies all phases/materials.
        """
        from ..geometry import StaleGeometryError

        if getattr(xp, '__name__', '') == 'array_api_compat.numpy':
            xp = np
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise NotImplementedError('Cartesian rectangle pair construction supports NumPy and JAX x64')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('Cartesian rectangle pair construction requires 64-bit precision')
        if (not isinstance(plan, _CartesianRectanglePairNVFPlan) or not isinstance(plan.layers, tuple)
                or any(choice is not None and (not isinstance(choice, _PairLayerPlan)
                       or choice.cell.refinement != self.refinement) for choice in plan.layers)):
            raise StaleGeometryError('Cartesian rectangle pair plan type/refinement is invalid; plan again')
        layers, basis, valid_basis = self._layers(structure, lattice, xp)
        if len(plan.layers) != len(layers):
            raise StaleGeometryError('Cartesian rectangle pair layer count changed; plan again')
        stacks = []
        for (shapes, background), choice in zip(layers, plan.layers):
            if (not shapes) != (choice is None):
                raise StaleGeometryError('Cartesian rectangle pair layer kind changed; plan again')
            valid = valid_basis & xp.isfinite(background)
            if choice is None:
                identity = xp.eye(self._count, dtype=xp.complex128)
                slots = ConvMatrices(background*identity, background*identity, identity/background,
                                     xp.zeros_like(identity), xp.zeros_like(identity))
            else:
                geometry, good_pair = self._pair_geometry(shapes, choice.order, xp, choice.cell)
                valid = valid & good_pair
                if choice.cell not in self._cells:
                    self._cells[choice.cell] = CartesianPairCell(choice.cell, self.truncation_order)
                layer_slots, _ = self._cells[choice.cell].conv_matrices(
                    basis=basis, background=background, xp=xp, **geometry)
                slots = ConvMatrices(*(slot[0] for slot in layer_slots))
            stacks.append(tuple(xp.where(valid, slot, xp.nan) for slot in slots))
        return ConvMatrices(*(xp.stack([layer[index] for layer in stacks]) for index in range(5)))
