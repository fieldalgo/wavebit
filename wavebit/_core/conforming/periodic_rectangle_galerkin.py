"""Real Galerkin response for a rectangle approaching its periodic image.

The y gap has one fixed open nodal graph, an exact collapsed graph, and an
analytical right derivative. This NumPy family reuses the original constrained
weak equations of the pair cell; no normal-projector blend or Li fallback is
used. Optical qualification is separate from the finite equations.
"""
from dataclasses import dataclass
import numpy as np

from .cartesian_pair_opening import CartesianPairOpeningPlan, freeze_topology, real_array
from .cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from .cartesian_pair_quadratic import _reference_gradients


@dataclass(frozen=True)
class _PeriodicRectanglePlan:
    refinement: int
    shift: int = 0


def _topology(refinement, degree):
    divisions = 2**refinement
    q = degree*divisions
    nx = ny = 3*q
    ix, iy = np.meshgrid(np.arange(nx+1), np.arange(ny+1), indexing='xy')
    coordinates = np.column_stack([ix.ravel(), iy.ravel()])
    periodic = ((iy % ny)*nx+ix % nx).ravel()
    collapsed_y = np.clip(iy-q, 0, q)
    mapping = (collapsed_y*(nx+1)+ix).ravel()
    contact_periodic = ((collapsed_y % q)*nx+ix % nx).ravel()
    count = nx*ny
    pmap = np.full(count, -1, int)
    for i, target in zip(periodic, contact_periodic):
        if pmap[i] not in (-1, target):
            raise ValueError('periodic collapse is inconsistent')
        pmap[i] = target
    cfull = np.eye(nx*q)[pmap]
    groups = [np.flatnonzero(pmap == i) for i in range(nx*q)]
    # The fixed open gauge belongs to a collapsing boundary group. Its other
    # nodes still need difference potentials; dropping that group's extras
    # would incorrectly clamp a physical gap voltage at the cell corner.
    extras = np.concatenate([group[1:] for group in groups])
    zfull = np.zeros((count, len(extras)))
    zfull[extras, np.arange(len(extras))] = 1.
    triangles, tags, thin = [], [], []
    index = lambda x, y: y*(nx+1)+x
    for j in range(3*divisions):
        for i in range(3*divisions):
            x, y, h = degree*i, degree*j, degree
            first = [index(x, y), index(x+h, y), index(x+h, y+h)]
            second = [index(x, y), index(x+h, y+h), index(x, y+h)]
            if degree == 2:
                first += [index(x+1, y), index(x+2, y+1), index(x+1, y+1)]
                second += [index(x+1, y+1), index(x+1, y+2), index(x, y+1)]
            triangles.extend([first, second])
            tags.extend([int(divisions <= i < 2*divisions and divisions <= j < 2*divisions)]*2)
            thin.extend([not divisions <= j < 2*divisions]*2)
    triangles, thin = np.asarray(triangles), np.asarray(thin)
    reference = (_reference_gradients() if degree == 2 else
                 np.array([[-1., -1.], [1., 0.], [0., 1.]]))
    for triangle in triangles[thin]:
        mapped = mapping[triangle[:3]]
        i, j = next((i, j) for i in range(3) for j in range(i+1, 3) if mapped[i] == mapped[j])
        direction = {(0, 1): (1, 0), (0, 2): (0, 1), (1, 2): (1, -1)}[i, j]
        c = cfull[periodic[triangle]]
        if np.any((reference@np.asarray(direction))@c != 0):
            raise ValueError('periodic collapse does not annihilate the transverse polynomial')
    return freeze_topology(dict(
        coordinates=coordinates, divisions=q, triangles=triangles, tags=np.asarray(tags),
        thin=thin, periodic=periodic, mapping=mapping, pmap=pmap,
        Cfull=cfull, Zfull=zfull, C=cfull[1:, 1:], Z=zfull[1:], extras=extras))


class PeriodicRectangleGalerkinGapCell(CartesianPairGalerkinGapCell):
    """Explicit finite rectangular-cell response through the vertical gap zero.

    A centered rectangle has fixed width and height ``period[1] - gap``.
    ``degree`` is 1 or 2 and geometric ``refinement`` is in 0..3. Period,
    width and translation are fixed in a JVP; gap and the complex inclusion
    and background materials may vary simultaneously. The zero-gap tangent
    opens the periodic seam. This class is NumPy-only and is not dispatched
    automatically by the ordinary NVF factorizer.
    """

    def __init__(self, order=1, *, degree=2, refinement=0, coercivity_angle=0.):
        super().__init__(CartesianPairOpeningPlan(0, refinement), order,
                         degree=degree, coercivity_angle=coercivity_angle)
        self._plan = _PeriodicRectanglePlan(refinement)
        self._data = _topology(refinement, self.degree)

    def _chart_geometry(self, gap, geometry):
        basis = geometry['basis']
        if basis[0, 1] != 0 or basis[1, 0] != 0 or min(np.diag(basis)) <= 0:
            raise ValueError('periodic rectangle requires a positive diagonal basis')
        length = np.sqrt(np.linalg.det(basis))
        px, py = np.diag(basis)/length
        width, s = sum(geometry['widths'])/length, gap/length
        if not 0 < width < px or not 0 <= s < py:
            raise ValueError('width must lie inside the period and gap in [0, period_y)')
        coordinates = self.data['coordinates']/self.data['divisions']
        xx, yy = coordinates.T
        x = np.interp(xx, np.arange(4), [0., (px-width)/2, (px+width)/2, px])
        y = np.clip(yy-1, 0., 1.)*py
        speed = np.where(yy < 1, yy/2, np.where(yy <= 2, 1.5-yy, (yy-3)/2))
        zero = np.column_stack([x, y])+geometry['offset']/length
        velocity = np.column_stack([np.zeros_like(speed), speed])
        return length, s, zero, velocity

    def closure(self, *, gap, period=(700., 700.), width=350., epsilon=12.25,
                background=1., offset=(0., 0.), gap_tangent=1., epsilon_tangent=0.,
                background_tangent=0.):
        period = real_array(period, (2,), 'period')
        width = real_array(width, (), 'width').item()
        if np.any(period <= 0):
            raise ValueError('period must be positive')
        slots, tangent, metadata = super().closure(
            gap=gap, basis=np.diag(period), widths=(width/2, width/2), height=period[1],
            epsilon=(epsilon, epsilon), background=background, offset=offset,
            gap_tangent=gap_tangent, epsilon_tangent=(epsilon_tangent, epsilon_tangent),
            background_tangent=background_tangent)
        metadata['geometry'] = 'periodic_rectangle_vertical_gap'
        return slots, tangent, metadata
