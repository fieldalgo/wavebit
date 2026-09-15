"""Higher-degree real pair potentials on the original open/contact geometry.

This explicit NumPy family raises both interface and interior polynomial
resolution while retaining the original constrained weak equations and gap
scaling. It does not change ordinary NVF routing or certify optical convergence.
"""
from numbers import Integral
import numpy as np

from .cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from .cartesian_pair_opening import freeze_topology, topology
from .triangle_polynomials import multi_indices, gradient_coefficients, product_integrals, phase_average, values_and_tangent


def _polynomial_topology(plan, degree, *, coarse=None):
    """Integer Bernstein control nets, periodic identity and exact collapsed quotient."""
    if coarse is None:
        coarse = topology(plan)
    indices = np.asarray(multi_indices(degree))
    denominator = degree*coarse['weight_denominator']
    rows = list(degree*coarse['open_numerators'])
    lookup = {tuple(row): i for i,row in enumerate(rows)}
    triangles = []
    for triangle in coarse['triangles']:
        local = []
        for row in indices@coarse['open_numerators'][triangle]:
            key = tuple(row)
            if key not in lookup:
                lookup[key] = len(rows); rows.append(row)
            local.append(lookup[key])
        triangles.append(local)
    opened = np.asarray(rows, dtype=np.int64)
    collapse = np.eye(10, dtype=np.int64)[[0,1,2,3,4,5,5,6,7,8,8,9]]
    folded = opened@collapse
    closed, lookup, mapping = [], {}, []
    for row in folded:
        key = tuple(row)
        if key not in lookup:
            lookup[key] = len(closed); closed.append(row)
        mapping.append(lookup[key])
    closed, mapping = np.asarray(closed, dtype=np.int64), np.asarray(mapping)
    unit = np.array([[0,0],[1,0],[1,1],[0,1]])
    def periodic(rows):
        lookup, result = {}, []
        for row in rows:
            key = (('body', *row) if np.any(row[4:]) else
                   ('boundary', *((row[:4]@unit) % denominator)))
            if key not in lookup:
                lookup[key] = len(lookup)
            result.append(lookup[key])
        return np.asarray(result)
    op, cp = periodic(opened), periodic(closed)
    pmap = np.full(max(op)+1, -1, int)
    for source, target in zip(op, cp[mapping]):
        if pmap[source] not in (-1, target):
            raise ValueError('polynomial periodic collapse is inconsistent')
        pmap[source] = target
    # Gauge is the first Bernstein coefficient, which is a surviving vertex.
    groups = [np.flatnonzero(pmap == i) for i in range(max(cp)+1)]
    if any(len(group) == 0 for group in groups) or len(groups[0]) != 1:
        raise ValueError('polynomial collapse must preserve the original fixed gauge')
    extras = np.concatenate([group[1:] for group in groups])
    c = np.eye(len(groups))[pmap]
    z = np.zeros((len(pmap), len(extras)))
    z[extras, np.arange(len(extras))] = 1.
    triangles = np.asarray(triangles)
    reference = gradient_coefficients(degree)
    for triangle in triangles[coarse['thin']]:
        mapped = mapping[triangle[:3]]
        i,j = next((i,j) for i in range(3) for j in range(i+1,3) if mapped[i] == mapped[j])
        direction = {(0,1):(1,0), (0,2):(0,1), (1,2):(1,-1)}[i,j]
        # Do not infer zero transverse gradients from rounded geometry.
        if np.any((reference@np.asarray(direction))@c[op[triangle]] != 0):
            raise ValueError('polynomial contact quotient does not cancel its transverse derivative')
    return freeze_topology(dict(open_numerators=opened, contact_numerators=closed,
        weight_denominator=denominator, triangles=triangles, tags=coarse['tags'], thin=coarse['thin'],
        mapping=mapping, periodic=op, pmap=pmap, Cfull=c, Zfull=z, C=c[1:,1:], Z=z[1:], extras=extras))


class CartesianPairPolynomialGalerkinGapCell(CartesianPairGalerkinGapCell):
    """Degree 2..4 Bernstein potentials, including original contact/material JVPs.

    The original explicit pair plan fixes the geometry and periodic topology.
    The dense implementation admits ``degree * 2**refinement <= 16``, the edge
    resolution of the existing finest quadratic family. This is a resource
    limit, not a convergence guarantee. Geometry and live directions follow
    ``CartesianPairGalerkinGapCell.closure`` exactly.
    """

    _maximum_degree = 4

    def __init__(self, plan, order=1, *, degree=3, coercivity_angle=0.):
        super().__init__(plan, order, degree=1, coercivity_angle=coercivity_angle)
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, Integral)
                or not 2 <= degree <= self._maximum_degree):
            raise ValueError(f'polynomial degree must be an integer from 2 through {self._maximum_degree}')
        if degree*2**plan.refinement > 16:
            raise ValueError('dense polynomial edge resolution must not exceed 16')
        self._degree = int(degree)
        self._data = _polynomial_topology(plan, self.degree)
        self._gram = product_integrals(self.degree-1)

    def _reference_gradient_coefficients(self):
        return gradient_coefficients(self.degree)

    def _polynomial_phase_average(self, points, velocity, waves):
        return phase_average(points, velocity, waves, self.degree-1)

    def _gradient_basis(self, barycentric, tangent):
        return values_and_tangent(barycentric, tangent, self.degree-1)

    def _elimination(self, potential_count, retained_count):
        # Higher polynomial spaces can nearly represent a retained gradient.
        # Eliminating through the original K avoids subtracting those large
        # nearly coincident spaces before the physical equations are solved.
        return 'retained'

    def _inner(self, left, right):
        # Both inputs use the same real degree-(p-1) Bernstein expansion.
        if left.shape[-1] == 0 or right.shape[-1] == 0:
            return np.zeros((left.shape[-1], right.shape[-1]), dtype=np.result_type(left,right))
        return np.einsum('adi,ab,bdj->ij', left, self._gram, right, optimize=True)

    def closure(self, **settings):
        value, tangent, metadata = super().closure(**settings)
        metadata['closure'] = 'real_polynomial_constrained_gradient_galerkin'
        return value, tangent, metadata
