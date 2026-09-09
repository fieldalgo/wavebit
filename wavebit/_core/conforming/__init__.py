"""Conforming NVF construction for rectangular tilings in square cells."""
from collections import namedtuple
from numbers import Integral

import numpy as np

from ..backend import _concrete
from ..domain import as_material
from ..factorization import ConvMatrices
from ..geometry import StaleGeometryError
from ..parametric import check_layer_spec, check_shape_spec
from .polynomial_body import PolynomialBodyGapModel
from .thin_gap import PERIOD


_Plan = namedtuple('ConformingNVFPlan', 'layers')


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
    epsilon = _real(material.epsilon, xp, 'conforming NVF permittivity')
    _require(epsilon > 0, 'conforming NVF requires positive real permittivity')
    return xp.where(epsilon > 0, epsilon, xp.nan)


class ConformingNVF:
    """Build energy-factorized matrices from ordinary rectangle layer dictionaries.

    Each patterned layer must be a contiguous horizontal tiling in the local
    axes of a right-handed square lattice: equal rectangle heights and y
    centers, total width less than the period, and no overlap or internal gap.
    Uniform layers are supported. The lattice may have any positive period,
    translation of a layer is unrestricted, and its axes may rotate together
    with the rectangles. Materials must be scalar, positive, real and
    non-magnetic; each layer may have its own background permittivity.

    ``truncation_order`` is an integer or (u, v) pair. Cell resolution settings
    are explicit; their defaults do not certify optical accuracy. NumPy and
    JAX with x64 construct the cell. Exported matrices also feed the Torch
    engine. No Li result or existing projector is substituted for this NVF.

    Call ``plan(structure, lattice)`` on concrete geometry before JAX JIT.
    The resulting immutable plan holds only rectangle ordering. Replays read
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
        self._cell = PolynomialBodyGapModel(max(self.truncation_order), **dict(
            dict(rows=4, body_degree=2, gap_degree=4, flux_basis='local'), **cell_settings))
        self._cap = 2*max(order)
        u, v = self.truncation_order
        n, m = np.meshgrid(np.arange(-v, v+1), np.arange(-u, u+1), indexing='ij')
        self._dn = n.ravel()[:, None]-n.ravel()+self._cap
        self._dm = m.ravel()[:, None]-m.ravel()+self._cap

    def _layers(self, structure, lattice, xp):
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
        period = xp.sqrt(xp.sum(basis[:, 0]**2))
        safe_period = xp.where(period > 0, period, 1.)
        rotation = basis/safe_period
        # Keep invalid concrete inputs away from determinant evaluation too.
        rotation = xp.where(xp.isfinite(rotation), rotation, 0.)
        valid = (real & xp.all(xp.isfinite(basis)) & (period > 0)
                 & (xp.max(xp.abs(rotation.T@rotation-xp.eye(2))) < 128*np.finfo(float).eps)
                 & (xp.linalg.det(rotation) > 0))
        _require(valid, 'conforming NVF requires a right-handed square lattice')
        period = xp.where(valid, period, xp.nan)
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
                center = rotation.T@center*(PERIOD/period)
                orientation = _real(shape.get('angle', 0.), xp, 'rectangle angle')
                aligned = xp.abs(xp.sin(orientation-angle)) < 128*np.finfo(float).eps
                _require(aligned, 'conforming rectangles must align with the first lattice axis')
                shapes.append(dict(
                    lx=_real(shape.get('lx', 0.), xp, 'lx')*(PERIOD/period),
                    ly=_real(shape.get('ly', 0.), xp, 'ly')*(PERIOD/period),
                    cx=center[0], cy=center[1],
                    epsilon=_epsilon(shape.get('material', 1.), xp)/background,
                    valid=aligned & valid))
            layers.append((shapes, background))
        return layers, rotation, valid

    def plan(self, structure, lattice):
        """Freeze only input rectangle permutations, never geometry or material values."""
        try:
            layers, _, _ = self._layers(structure, lattice, np)
        except (TypeError, RuntimeError) as exc:
            raise ValueError('plan conforming NVF on concrete geometry before JIT') from exc
        orders = tuple(tuple(sorted(range(len(shapes)), key=lambda i: float(shapes[i]['cx'])))
                       for shapes, _ in layers)
        # Validate the tiling while planning, without constructing a cell.
        for (shapes, _), order in zip(layers, orders):
            if shapes:
                self._profile([shapes[i] for i in order], np, False)
        return _Plan(orders)

    def _profile(self, shapes, xp, replay):
        values = {key: xp.stack([s[key] for s in shapes])
                  for key in ('lx', 'ly', 'cx', 'cy', 'epsilon', 'valid')}
        left, right = values['cx']-values['lx']/2, values['cx']+values['lx']/2
        width, cx, cy = right[-1]-left[0], (right[-1]+left[0])/2, values['cy'][0]
        tol = 32*np.finfo(float).eps*PERIOD
        gap = PERIOD-values['ly'][0]
        valid = (xp.all(values['valid']) & xp.all(values['lx'] > 0) & xp.all(values['ly'] > 0)
                 & (width > 0) & (width < PERIOD) & (gap >= -tol) & (gap < PERIOD)
                 & xp.all(xp.abs(values['ly']-values['ly'][0]) <= tol)
                 & xp.all(xp.abs(values['cy']-cy) <= tol)
                 & xp.all(xp.abs(right[:-1]-left[1:]) <= tol))
        _require(valid, 'conforming rectangles must form a contiguous equal-height tiling within one period',
                 StaleGeometryError if replay else ValueError)
        # Correct only negative roundoff at contact. The zero/positive branch
        # retains its full one-sided gap derivative and all positive gaps.
        gap = xp.where((gap < 0) & (gap >= -tol), 0., gap)
        seams = (right[:-1]+left[1:])/2-cx
        return gap, width, cx, cy, dict(seams=seams, values=values['epsilon']), valid

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
        plan = self.plan(structure, lattice) if plan is None else plan
        layers, rotation, valid_lattice = self._layers(structure, lattice, xp)
        if not isinstance(plan, _Plan) or len(plan.layers) != len(layers):
            raise StaleGeometryError('conforming NVF layer count or plan type changed; plan again')
        stacks = []
        for (shapes, background), order in zip(layers, plan.layers):
            if sorted(order) != list(range(len(shapes))):
                raise StaleGeometryError('conforming NVF shape count changed; plan again')
            count = len(self._dn)
            if not shapes:
                identity = xp.eye(count, dtype=xp.complex128)
                slots = ConvMatrices(background*identity, background*identity, identity/background,
                                     xp.zeros_like(identity), xp.zeros_like(identity))
                valid = valid_lattice & xp.isfinite(background) & (background > 0)
            else:
                gap, width, cx, cy, profile, valid = self._profile([shapes[i] for i in order], xp, replay)
                coefficients, _ = self._cell.coefficients(gap, profile, width, xp)
                modes = xp.asarray(self._cell.modes)
                # The cell kernels integrate around zero. Ordinary shape
                # centers are offsets from (a1+a2)/2, while the engine's
                # Fourier origin is the cell corner. Translate every field,
                # including its energy, so mixed stacks share one origin.
                phase = xp.exp(-2j*np.pi/PERIOD*(modes[:, None]*(cy+PERIOD/2)
                                                +modes[None]*(cx+PERIOD/2)))
                c = coefficients*phase[None]
                field = xp.stack([xp.stack([c[0], c[1]]), xp.stack([c[2], c[3]])])
                gram = xp.stack([xp.stack([c[4], c[5]]), xp.stack([c[5], c[6]])])
                field, gram = (xp.einsum('ai,ijmn,bj->abmn', rotation, a, rotation) for a in (field, gram))
                def blocks(a):
                    a = a[..., self._dn, self._dm]
                    return xp.block([[a[0, 0], a[0, 1]], [a[1, 0], a[1, 1]]])
                slots = ConvMatrices.from_energy(blocks(field), background*blocks(gram),
                                                  background*c[7, self._dn, self._dm])
                defect = xp.maximum(xp.maximum(xp.max(xp.abs(slots.epx-slots.epx.conj().T)),
                                                xp.max(xp.abs(slots.epy-slots.epy.conj().T))),
                                    xp.max(xp.abs(slots.epxy-slots.epyx.conj().T)))
                valid = valid & (defect < 1e-9)
                _require(valid, 'conforming NVF energy closure failed its Hermitian guard')
            stacks.append(tuple(xp.where(valid, slot, xp.nan) for slot in slots))
        return ConvMatrices(*(xp.stack([s[i] for s in stacks]) for i in range(5)))


__all__ = ['ConformingNVF']
