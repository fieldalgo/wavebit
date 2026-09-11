"""Conforming NVF construction for rectangular tilings in orthogonal cells."""
from collections import namedtuple
from numbers import Integral

import numpy as np

from ..backend import _concrete
from ..domain import as_material
from ..factorization import ConvMatrices
from ..geometry import StaleGeometryError
from ..lattice import reduction_transform
from ..parametric import check_layer_spec, check_shape_spec
from .polynomial_body import ComplexMaterialGapModel
from .thin_gap import PERIOD, CONTACT_ROUNDOFF


_IDENTITY = ((1, 0), (0, 1))
_Plan = namedtuple('ConformingNVFPlan', 'layers axes reduction')


def _require(condition, message, error=ValueError):
    if _concrete(condition) is False:
        raise error(message)


def _real(value, xp, name):
    value = xp.asarray(value)
    if value.ndim != 0:
        raise ValueError(f'{name} must be a real scalar')
    valid = xp.isfinite(value) & (xp.imag(value) == 0)
    _require(valid, f'{name} must be finite and real')
    return xp.where(valid, xp.real(value), xp.nan)


def _epsilon(value, xp):
    material = as_material(value, 'conforming NVF material')
    if material.is_magnetic:
        raise NotImplementedError('conforming NVF does not construct permeability cells')
    epsilon = xp.asarray(material.epsilon)
    if epsilon.ndim != 0:
        raise ValueError('conforming NVF permittivity must be a scalar')
    valid = xp.isfinite(epsilon) & (xp.real(epsilon) > 0) & (xp.imag(epsilon) >= 0)
    _require(valid, 'conforming NVF requires finite passive permittivity with positive real part')
    return xp.where(valid, epsilon, xp.nan)


class ConformingNVF:
    """Build energy-factorized matrices from ordinary rectangle layer dictionaries.

    Each patterned layer must be a nonoverlapping sequence along either local axis
    of a right-handed orthogonal lattice, with equal perpendicular extent and
    center. Its span along the tiling axis must be less than that period;
    its perpendicular extent may reach the other period. Internal gaps contain
    the layer background and may open or close during replay. Different layers
    may use different axes.
    Adding pi/2 to a rectangle angle and exchanging its lengths preserves
    the geometry; rectangle edges must remain aligned with lattice axes.
    Uniform layers are supported. The two periods may differ and must be positive;
    translation of a layer is unrestricted, and its axes may rotate together
    with the rectangles. Materials must be scalar and non-magnetic, with
    positive real part and nonnegative imaginary part.
    Every material and background may have its own loss ratio. Both real
    and imaginary parts remain live on replay, including one-sided loss
    derivatives at the passive boundary.

    An integer change of lattice basis may spell the same orthogonal point
    lattice with nonorthogonal vectors. Planning records its unimodular
    reduction; the physical rectangles, original cell origin and retained
    Fourier set are preserved. Genuinely nonorthogonal point lattices remain
    unsupported. During replay the reduced basis must remain orthogonal.

    ``truncation_order`` is an integer or (u, v) pair. Cell resolution settings
    are explicit; their defaults do not certify optical accuracy. NumPy and
    JAX with x64 construct the cell. Exported matrices also feed the Torch
    engine. No Li result or existing projector is substituted for this NVF.

    Call ``plan(structure, lattice)`` on concrete geometry before JAX JIT.
    The resulting immutable plan holds rectangle ordering, each layer's
    tiling axis and the integer lattice reduction. If both axes are valid,
    planning prefers the first reduced axis. Replays read
    all coordinates and materials live and check the supported tiling; invalid
    eager replays raise, and invalid traced geometry produces NaN matrices.
    """

    def __init__(self, truncation_order=3, **cell_settings):
        order = ((truncation_order, truncation_order) if isinstance(truncation_order, Integral)
                 else truncation_order)
        if (not isinstance(order, (tuple, list)) or len(order) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in order)):
            raise ValueError('truncation_order must be a nonnegative integer or integer pair')
        self.truncation_order = tuple(int(n) for n in order)
        self._cell_settings = dict(dict(rows=4, body_degree=2, gap_degree=4, flux_basis='local'), **cell_settings)
        self._cell = ComplexMaterialGapModel(max(self.truncation_order), **self._cell_settings)
        self._cells = {self._cell.order: self._cell}
        u, v = self.truncation_order
        n, m = np.meshgrid(np.arange(-v, v+1), np.arange(-u, u+1), indexing='ij')
        self._m, self._n = m.ravel(), n.ravel()

    def _layers(self, structure, lattice, xp, reduction=_IDENTITY):
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise NotImplementedError('conforming cell construction supports NumPy and JAX x64')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('conforming cell construction requires 64-bit precision')
        basis = xp.asarray(getattr(lattice, 'columns', lattice))
        if basis.shape != (2, 2):
            raise ValueError('conforming NVF needs a 2-by-2 lattice with basis vectors in columns')
        _require(xp.all(xp.isfinite(basis)), 'conforming NVF lattice must be finite')
        real = xp.all(xp.imag(basis) == 0)
        basis = xp.real(basis)
        original = basis
        basis = basis@xp.asarray(reduction)
        # Shape offsets refer to the caller's cell center. Keep their absolute
        # positions when constructing the cell on a different lattice basis.
        center_shift = xp.sum(original-basis, axis=1)/2
        periods = xp.sqrt(xp.sum(basis**2, axis=0))
        safe_periods = xp.where(periods > 0, periods, 1.)
        rotation = basis/safe_periods
        # Keep invalid concrete inputs away from determinant evaluation too.
        rotation = xp.where(xp.isfinite(rotation), rotation, 0.)
        valid = (real & xp.all(xp.isfinite(basis)) & xp.all(periods > 0)
                 & (xp.max(xp.abs(rotation.T@rotation-xp.eye(2))) < 128*np.finfo(float).eps)
                 & (xp.linalg.det(rotation) > 0))
        _require(valid, 'conforming NVF requires a right-handed lattice with an orthogonal reduced basis')
        periods = xp.where(valid, periods, xp.nan)
        angle = xp.arctan2(rotation[1, 0], rotation[0, 0])
        if not isinstance(structure, (tuple, list)) or not structure:
            raise ValueError('conforming NVF needs a nonempty sequence of layer dictionaries')
        layers = []
        for layer_index, layer in enumerate(structure):
            if not isinstance(layer, dict):
                raise ValueError('each conforming NVF layer must be a dictionary')
            check_layer_spec(layer, layer_index)
            background = _epsilon(layer.get('background', 1.), xp)
            shapes = []
            supplied = layer.get('shapes', [])
            if not isinstance(supplied, (tuple, list)):
                raise ValueError('conforming NVF shapes must be a sequence of dictionaries')
            for index, shape in enumerate(supplied):
                if not isinstance(shape, dict):
                    raise ValueError('each conforming NVF shape must be a dictionary')
                check_shape_spec(shape, index, layer_index)
                if shape.get('type', 'rectangle') != 'rectangle':
                    raise NotImplementedError('conforming NVF currently supports rectangular tilings')
                center = xp.stack([_real(shape.get('cx', 0.), xp, 'cx'),
                                   _real(shape.get('cy', 0.), xp, 'cy')])
                center = rotation.T@(center+center_shift)*(PERIOD/periods)
                orientation = _real(shape.get('angle', 0.), xp, 'rectangle angle')
                cosine, sine = xp.cos(orientation-angle), xp.sin(orientation-angle)
                aligned = xp.abs(sine*cosine) < 128*np.finfo(float).eps
                _require(aligned, 'conforming rectangles must align with a reduced lattice axis')
                lx = _real(shape.get('lx', 0.), xp, 'lx')
                ly = _real(shape.get('ly', 0.), xp, 'ly')
                positive = (lx > 0) & (ly > 0)
                _require(positive, 'conforming rectangle lengths must be positive')
                # Accepted angles are multiples of pi/2. Squared direction
                # cosines exchange the local extents without an abs cusp or
                # a traced orientation branch. The alignment guard remains live.
                width = cosine**2*lx+sine**2*ly
                height = sine**2*lx+cosine**2*ly
                epsilon = _epsilon(shape.get('material', 1.), xp)
                shapes.append(dict(
                    lx=width*(PERIOD/periods[0]),
                    ly=height*(PERIOD/periods[1]),
                    gap=(periods[1]-height)*(PERIOD/periods[1]),
                    gap_x=(periods[0]-width)*(PERIOD/periods[0]),
                    cx=center[0], cy=center[1],
                    # Normalization changes neither the cell equation nor
                    # physical passivity after restoring the background.
                    epsilon=epsilon/background,
                    valid=positive & aligned & valid))
            layers.append((shapes, background))
        return layers, rotation, valid, periods[0]/periods[1]

    def plan(self, structure, lattice):
        """Freeze lattice reduction, layer axes and input permutations.

        Try a tiling along the first reduced lattice axis, then the second. Replay
        keeps that choice even if another axis later becomes valid. Re-plan
        explicitly when changing layer/shape counts or ordering, or when the
        chosen axis becomes invalid. Internal gaps may open or close on replay.
        """
        try:
            basis = np.asarray(getattr(lattice, 'columns', lattice))
            if (basis.shape != (2, 2) or not np.all(np.isfinite(basis))
                    or np.any(np.imag(basis) != 0)):
                raise ValueError('conforming NVF lattice must be a finite real 2-by-2 matrix')
            basis = np.real(basis)
            if np.linalg.det(basis) <= 0:
                raise ValueError('conforming NVF lattice must be right-handed and nonsingular')
            reduction = reduction_transform(basis)
            layers, _, _, _ = self._layers(structure, lattice, np, reduction)
        except (TypeError, RuntimeError) as exc:
            raise ValueError('plan conforming NVF on concrete geometry before JIT') from exc
        orders, axes = [], []
        # Validate both possible tiling frames without constructing a cell.
        for shapes, _ in layers:
            for axis in (0, 1):
                local = self._axis_shapes(shapes, axis)
                order = tuple(sorted(range(len(local)), key=lambda i: float(local[i]['cx'])))
                try:
                    if local:
                        self._profile([local[i] for i in order], np, False)
                except ValueError:
                    continue
                orders.append(order)
                axes.append(axis)
                break
            else:
                raise ValueError('conforming rectangles must form a nonoverlapping profile along a lattice axis')
        return _Plan(tuple(orders), tuple(axes), reduction)

    def _cell_orders(self, reduction, axis):
        # k_reduced = T.T @ k_original. Form exactly the caller's finite set,
        # including its holes in a rectangular reduced-coordinate envelope.
        # Selecting after matrix inversion would describe a different closure.
        t = np.asarray(reduction)
        m = t[0, 0]*self._m+t[1, 0]*self._n
        n = t[0, 1]*self._m+t[1, 1]*self._n
        order = int(max(np.max(np.abs(m)), np.max(np.abs(n))))
        if order not in self._cells:
            self._cells[order] = ComplexMaterialGapModel(order, **self._cell_settings)
        if axis == 1:
            m, n = n, -m
        return self._cells[order], n[:, None]-n[None]+2*order, m[:, None]-m[None]+2*order

    @staticmethod
    def _axis_shapes(shapes, axis):
        if axis == 0:
            return shapes
        # Right-handed frame (a2, -a1), with periods (Ly, Lx).
        return [dict(s, lx=s['ly'], ly=s['lx'], cx=s['cy'], cy=-s['cx'],
                     gap=s['gap_x'], gap_x=s['gap']) for s in shapes]

    def _profile(self, shapes, xp, replay):
        values = {key: xp.stack([s[key] for s in shapes])
                  for key in ('lx', 'ly', 'gap', 'cx', 'cy', 'epsilon', 'valid')}
        left, right = values['cx']-values['lx']/2, values['cx']+values['lx']/2
        width, cx, cy = right[-1]-left[0], (right[-1]+left[0])/2, values['cy'][0]
        tol = CONTACT_ROUNDOFF*PERIOD
        gap = values['gap'][0]
        valid = (xp.all(values['valid']) & xp.all(values['lx'] > 0) & xp.all(values['ly'] > 0)
                 & (width > 0) & (width < PERIOD) & (gap >= -tol) & (gap < PERIOD)
                 & xp.all(xp.abs(values['ly']-values['ly'][0]) <= tol)
                 & xp.all(xp.abs(values['cy']-cy) <= tol)
                 & xp.all(right[:-1]-left[1:] <= tol))
        _require(valid, 'conforming rectangles must form a nonoverlapping equal-height profile within one period',
                 StaleGeometryError if replay else ValueError)
        # The analytically continued contact equations accept this same tiny
        # roundoff neighborhood. Snapping negative roundoff to zero would erase
        # the live gap derivative when a rotated lattice norm rounds down.
        # Every internal interval carries background, including at zero width.
        # Keep both live endpoints: averaging a contacting pair would discard
        # the one-sided derivative when that background interval opens.
        seams = xp.stack([right[:-1], left[1:]], axis=-1).ravel()-cx
        materials = xp.concatenate([
            xp.stack([values['epsilon'][:-1], xp.ones_like(values['epsilon'][:-1])], axis=-1).ravel(),
            values['epsilon'][-1:]])
        return gap, width, cx, cy, dict(seams=seams, values=materials), valid

    def conv_matrices(self, structure, lattice, *, plan=None, xp=np):
        """Return a layer stack for solve_conv with matching lattice and retained orders.

        Arrays are complex128. Optical conditions belong to the consuming solver.
        Shape centers are Cartesian offsets from the cell center, matching the
        ordinary shape API. Fourier coefficients use the engine's cell-corner
        origin. A fresh solver without a raster structure selects the 2D route,
        including when a retained axis has order zero.
        """
        if getattr(xp, '__name__', '') == 'array_api_compat.numpy':
            xp = np
        replay = plan is not None
        plan = self.plan(structure, lattice=lattice) if plan is None else plan
        if not isinstance(plan, _Plan):
            raise StaleGeometryError('conforming NVF plan type is invalid; plan again')
        t = plan.reduction
        if (not isinstance(t, tuple) or len(t) != 2
                or any(not isinstance(row, tuple) or len(row) != 2 for row in t)
                or any(isinstance(v, bool) or not isinstance(v, Integral) for row in t for v in row)
                or t[0][0]*t[1][1]-t[0][1]*t[1][0] != 1):
            raise StaleGeometryError('conforming NVF plan lattice reduction must be unimodular and right-handed')
        layers, rotation, valid_lattice, aspect = self._layers(structure, lattice, xp, t)
        if (len(plan.layers) != len(layers)
                or len(plan.axes) != len(layers) or any(axis not in (0, 1) for axis in plan.axes)):
            raise StaleGeometryError('conforming NVF layer count, plan type or tiling axis is invalid; plan again')
        stacks = []
        for (shapes, background), order, axis in zip(layers, plan.layers, plan.axes):
            if sorted(order) != list(range(len(shapes))):
                raise StaleGeometryError('conforming NVF shape count changed; plan again')
            count = len(self._m)
            if not shapes:
                identity = xp.eye(count, dtype=xp.complex128)
                slots = ConvMatrices(background*identity, background*identity, identity/background,
                                     xp.zeros_like(identity), xp.zeros_like(identity))
                valid = valid_lattice & xp.isfinite(background) & (xp.real(background) > 0)
            else:
                local = self._axis_shapes(shapes, axis)
                gap, width, cx, cy, profile, valid = self._profile([local[i] for i in order], xp, replay)
                cell_rotation = rotation if axis == 0 else rotation@xp.asarray([[0., -1.], [1., 0.]])
                cell_aspect = aspect if axis == 0 else 1/aspect
                cell, dn, dm = self._cell_orders(t, axis)
                coefficients, _ = cell.coefficients(gap, profile, width, xp, aspect=cell_aspect)
                modes = xp.asarray(cell.modes)
                # The cell kernels integrate around zero. Ordinary shape
                # centers are offsets from (a1+a2)/2, while the engine's
                # Fourier origin is the cell corner. Translate every field,
                # including its energy, so mixed stacks share one origin.
                phase = xp.exp(-2j*np.pi/PERIOD*(modes[:, None]*(cy+PERIOD/2)
                                                +modes[None]*(cx+PERIOD/2)))
                c = coefficients*phase[None]
                field = xp.stack([xp.stack([c[0], c[1]]), xp.stack([c[2], c[3]])])
                gram = xp.stack([xp.stack([c[4], c[5]]), xp.stack([c[8], c[6]])])
                field, gram = (xp.einsum('ai,ijmn,bj->abmn', cell_rotation, a, cell_rotation)
                               for a in (field, gram))
                def blocks(a):
                    a = a[..., dn, dm]
                    return xp.block([[a[0, 0], a[0, 1]], [a[1, 0], a[1, 1]]])
                slots = ConvMatrices.from_energy(blocks(field), background*blocks(gram),
                                                  background*c[7, dn, dm])
                tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
                adjoint = tensor.conj().T
                storage = (tensor+adjoint)/2
                dissipation = (tensor-adjoint)/(2j)
                lossless = (xp.imag(background) == 0) & xp.all(xp.imag(profile['values']) == 0)
                hermitian_defect = xp.maximum(xp.maximum(xp.max(xp.abs(slots.epx-slots.epx.conj().T)),
                                                         xp.max(xp.abs(slots.epy-slots.epy.conj().T))),
                                             xp.max(xp.abs(slots.epxy-slots.epyx.conj().T)))
                # Positive real physical materials give a coercive storage
                # form; passive imaginary parts give nonnegative dissipation.
                # Test these forms without changing the exported matrices.
                valid = (valid & xp.all(xp.isfinite(tensor))
                         & (~lossless | (hermitian_defect < 1e-9))
                         & (xp.linalg.eigvalsh(storage)[0] > 0)
                         & (xp.linalg.eigvalsh(dissipation)[0] > -1e-9)
                         & xp.all(xp.isfinite(slots.epz_i)))
                _require(valid, 'conforming NVF energy closure failed its passivity guard')
            stacks.append(tuple(xp.where(valid, slot, xp.nan) for slot in slots))
        return ConvMatrices(*(xp.stack([s[i] for s in stacks]) for i in range(5)))


__all__ = ['ConformingNVF']
