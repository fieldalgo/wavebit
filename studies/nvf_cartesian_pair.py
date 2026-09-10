"""Experimental finite P1 cells for a horizontal equal-height Cartesian pair.

A positive gap and exact full-height contact use different immutable charts.
The physical pair and its fundamental cell translate together. Only strictly
contained, positive-triangle charts and refinement 0/1 are admitted. This
study provides finite coefficients and energy closure, not a public geometry
adapter, an opening-gap derivative, or an optical convergence certificate.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from wavebit._core.backend import _concrete
from wavebit._core.factorization import ConvMatrices
from wavebit._core.polygon import polygon_ft

__all__ = ['CartesianPairPlan', 'CartesianPairCell']

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


@dataclass(frozen=True)
class CartesianPairPlan:
    """Concrete cyclic annulus and contact state, fixed during live replay.

    ``from_geometry`` selects one of four cyclic charts. It stores no lattice,
    width, material or offset value. Planning under JIT is refused; creating a
    new chart is an explicit operation outside differentiation. Refinement is
    restricted to 0 or 1. Contact means an exact full-height shared edge and
    excludes derivatives that open the gap.
    """

    contact: bool
    shift: int
    refinement: int = 0

    def __post_init__(self):
        if not isinstance(self.contact, (bool, np.bool_)):
            raise ValueError('contact must be a concrete bool')
        for value, allowed in ((self.shift, range(4)), (self.refinement, (0, 1))):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in allowed:
                raise ValueError('shift must be 0..3 and refinement must be 0 or 1')

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
        cells, tags = _base_topology(self.contact, self.shift)
        weights = np.eye(10 if self.contact else 12)
        for _ in range(self.refinement):
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
