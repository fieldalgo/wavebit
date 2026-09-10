"""Explicit fractional strip profiles on an unreduced physical affine lattice."""
from numbers import Integral

import numpy as np

from ..factorization import ConvMatrices
from . import _epsilon, _real, _require
from .affine_cell import AffineCellGapModel
from .thin_gap import CONTACT_ROUNDOFF, PERIOD


def _keys(mapping, allowed, required, name):
    if not isinstance(mapping, dict):
        raise ValueError(f'{name} must be a dictionary')
    unknown, missing = set(mapping)-allowed, required-set(mapping)
    if unknown:
        raise ValueError(f'{name} has unknown keys: {sorted(unknown, key=repr)}')
    if missing:
        raise ValueError(f'{name} is missing required keys: {sorted(missing)}')


def _vector(value, xp, name, length=None):
    array = xp.asarray(value)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(f'{name} must be a real vector'+('' if length is None else f' of length {length}'))
    valid = xp.all(xp.isfinite(array)) & xp.all(xp.imag(array) == 0)
    _require(valid, f'{name} must be finite and real')
    return xp.where(valid, xp.real(array), xp.nan)


class AffineProfileNVF:
    """Construct finite NVF matrices from explicit fractional reference profiles.

    ``structure`` is a nonempty list or tuple of layer dictionaries. A layer
    accepts only ``background`` (default vacuum) and ``profile``. An absent or
    None profile is uniform. A patterned profile requires ``width``, ``height``,
    ``seams`` and ``values``, with optional Cartesian ``offset=(dx, dy)``.
    Width and height are fractions of the two caller primitive coordinates:
    0 < width < 1 and 0 < height <= 1. Seams are centered fractional first-axis
    positions in [-width/2, width/2], in nondecreasing order. Coincident seams
    are retained, including their separate live opening derivatives. Only the
    existing contact-roundoff neighborhood is admitted; nothing is snapped.
    There is one scalar passive physical material per interval between seams.
    Materials and background must be nonmagnetic, finite, positive in real
    permittivity and nonnegative in imaginary permittivity.

    ``lattice`` is a Lattice or a finite real 2-by-2 matrix with physical basis
    vectors in columns and positive determinant. It maps the reference profile
    into physical parallelograms. Offset is measured from the caller's cell
    center in the same physical units as the lattice, not fractional units.
    The original basis and Fourier labels are preserved without reduction;
    this contract does not reinterpret ordinary physical rectangle recipes.

    ``truncation_order`` is a nonnegative integer or (x, y) pair. Explicit cell
    settings have the same defaults as ConformingNVF. The finite construction
    supports NumPy and JAX x64; all numerical profile, material, offset and
    lattice values stay live under JIT without a geometry plan. Invalid eager
    inputs raise, and invalid traced values poison the affected layer matrices.
    Invalid lattice values poison every layer. These matrices provide access
    to the finite affine model, not an independent optical convergence bound.
    """

    def __init__(self, truncation_order=3, **cell_settings):
        order = ((truncation_order, truncation_order) if isinstance(truncation_order, Integral)
                 else truncation_order)
        if (not isinstance(order, (tuple, list)) or len(order) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in order)):
            raise ValueError('truncation_order must be a nonnegative integer or integer pair')
        self.truncation_order = tuple(int(n) for n in order)
        self._cell_settings = dict(dict(rows=4, body_degree=2, gap_degree=4, flux_basis='local'), **cell_settings)
        self._cell = AffineCellGapModel(self.truncation_order, **self._cell_settings)

    @staticmethod
    def _basis(lattice, xp):
        basis = xp.asarray(getattr(lattice, 'columns', lattice))
        if basis.shape != (2, 2):
            raise ValueError('affine profile lattice must be a real finite right-handed 2-by-2 matrix')
        finite, real = xp.all(xp.isfinite(basis)), xp.all(xp.imag(basis) == 0)
        basis = xp.real(basis)
        safe = xp.where(xp.isfinite(basis), basis, 0.)
        magnitude = xp.max(xp.abs(safe))
        normalized = safe/xp.where(magnitude > 0, magnitude, 1.)
        valid = finite & real & (xp.linalg.det(normalized) > 0)
        _require(valid, 'affine profile lattice must be a real finite right-handed 2-by-2 matrix')
        # Safe arithmetic for invalid traced lattices; the final live validity
        # mask still poisons every layer rather than substituting this basis.
        return xp.where(valid, basis, xp.eye(2)), valid

    @staticmethod
    def _profile(profile, xp):
        _keys(profile, {'width', 'height', 'seams', 'values', 'offset'},
              {'width', 'height', 'seams', 'values'}, 'affine profile')
        width = _real(profile['width'], xp, 'fractional width')
        height = _real(profile['height'], xp, 'fractional height')
        seams = _vector(profile['seams'], xp, 'fractional seams')
        offset = _vector(profile.get('offset', (0., 0.)), xp, 'Cartesian offset', 2)
        supplied = profile['values']
        if not isinstance(supplied, (list, tuple)):
            supplied = xp.asarray(supplied)
            if supplied.ndim != 1:
                raise ValueError('affine profile values must be a vector of scalar materials')
        if len(supplied) != len(seams)+1:
            raise ValueError('affine profile needs one more material value than seams')
        values = xp.stack([_epsilon(value, xp) for value in supplied])
        edges = xp.concatenate([xp.reshape(-width/2, (1,)), seams, xp.reshape(width/2, (1,))])
        valid = ((width > 0) & (width < 1) & (height > 0)
                 & (1-height >= -CONTACT_ROUNDOFF)
                 & xp.all(xp.isfinite(edges)) & xp.all(xp.diff(edges) >= -CONTACT_ROUNDOFF)
                 & xp.all(xp.isfinite(offset)) & xp.all(xp.isfinite(values)))
        _require(valid, 'affine profile requires fractional width in (0,1), height in (0,1], '
                 'and ordered interior seams')
        return width, height, seams, values, offset, valid

    def conv_matrices(self, structure, lattice, *, xp=np):
        """Return complex128 layer matrices in the caller's original Fourier set.

        Use the same lattice and truncation order in the consuming solve_conv.
        Coefficients are translated from the centered reference profile to the
        engine's cell-corner origin before closure, in the physical Cartesian
        frame already returned by the affine cell. No second rotation is used.
        """
        if getattr(xp, '__name__', '') == 'array_api_compat.numpy':
            xp = np
        if xp is not np and getattr(xp, '__name__', '') != 'jax.numpy':
            raise NotImplementedError('affine profile construction supports NumPy and JAX x64')
        if xp.asarray(0.).dtype != np.dtype('float64'):
            raise ValueError('affine profile construction requires 64-bit precision')
        if not isinstance(structure, (tuple, list)) or not structure:
            raise ValueError('affine profile construction needs a nonempty sequence of layer dictionaries')
        basis, valid_basis = self._basis(lattice, xp)
        cell, stacks = self._cell, []
        for layer in structure:
            _keys(layer, {'background', 'profile'}, set(), 'affine profile layer')
            background = _epsilon(layer.get('background', 1.), xp)
            valid = valid_basis & xp.isfinite(background)
            profile = layer.get('profile')
            if profile is None:
                identity = xp.eye(cell.count, dtype=xp.complex128)
                slots = ConvMatrices(background*identity, background*identity, identity/background,
                                     xp.zeros_like(identity), xp.zeros_like(identity))
            else:
                width, height, seams, values, offset, valid_profile = self._profile(profile, xp)
                relative = dict(seams=PERIOD*seams, values=values/background)
                coefficients, _ = cell.coefficients(PERIOD*(1-height), relative, PERIOD*width,
                                                     xp, basis=basis)
                center = .5+xp.linalg.solve(basis, offset)
                modes = xp.asarray(cell.modes)
                phase = xp.exp(-2j*np.pi*(modes[:, None]*center[1]+modes[None]*center[0]))
                # Select the exact requested labels before either closure
                # inverse. Restore physical epsilon on all four Gram entries
                # and the material entry, not on the change of field.
                c = (coefficients*phase[None])[:, cell.dn, cell.dm]
                field = xp.block([[c[0], c[1]], [c[2], c[3]]])
                gram = background*xp.block([[c[4], c[5]], [c[8], c[6]]])
                slots = ConvMatrices.from_energy(field, gram, background*c[7])
                tensor = xp.block([[slots.epx, -slots.epxy], [-slots.epyx, slots.epy]])
                adjoint = tensor.conj().T
                storage, dissipation = (tensor+adjoint)/2, (tensor-adjoint)/(2j)
                lossless = (xp.imag(background) == 0) & xp.all(xp.imag(values) == 0)
                hermitian_defect = xp.maximum(xp.maximum(xp.max(xp.abs(slots.epx-slots.epx.conj().T)),
                                                         xp.max(xp.abs(slots.epy-slots.epy.conj().T))),
                                             xp.max(xp.abs(slots.epxy-slots.epyx.conj().T)))
                valid = (valid & valid_profile & xp.all(xp.isfinite(tensor))
                         & (~lossless | (hermitian_defect < 1e-9))
                         & (xp.linalg.eigvalsh(storage)[0] > 0)
                         & (xp.linalg.eigvalsh(dissipation)[0] > -1e-9)
                         & xp.all(xp.isfinite(slots.epz_i)))
                _require(valid, 'affine profile NVF energy closure failed its passivity guard')
            stacks.append(tuple(xp.where(valid, slot, xp.nan) for slot in slots))
        return ConvMatrices(*(xp.stack([slots[i] for slots in stacks]) for i in range(5)))
