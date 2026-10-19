"""Physical face conditions, independent of the absorbing-grid implementation."""
from wavebit._common.declarations import spec
from wavebit.errors import ValidationError
from . import _legacy_boundaries
from ._legacy_boundaries import Periodic, PEC, PMC

__all__ = ['Open', 'Periodic', 'Bloch', 'PEC', 'PMC', 'MirrorSymmetry', 'InvariantAxis',
           'Boundaries']


def _publicly(build):
    """``build()``, with an engine refusal spelled for a public caller.

    The engine face a public face lowers to checks its own values; the refusal it raises names
    its own leaves and keywords until the public spelling translates it.
    """
    from ._spelling import public, public_error
    with public():
        try:
            return build()
        except (TypeError, ValueError, NotImplementedError) as exc:
            translated = public_error(exc)
            if translated is exc:
                raise
            raise translated from exc


@spec
class Open:
    """An outgoing face: waves leave the domain through it.

    The absorber that implements it is a numerical choice, ``Numerics(absorber=...)``, and its
    cells are added outside the physical domain. Uniform boundary media continue into that
    padding; a finite body that crosses the face needs an explicitly supported continuation
    and is otherwise refused rather than replaced with vacuum.
    """


@spec
class Bloch:
    """A Bloch-periodic face pair: the field repeats with a phase in radians, or a wavevector.

    Give ``phase`` (radians across the period) or ``k`` (inverse ``Problem.units`` lengths).
    :meth:`from_source` derives both transverse faces from an oblique :class:`PlaneWave`.
    """
    phase: object = None
    k: object = None

    @staticmethod
    def from_source(source, *, units, background=None, invariant_axes=(), **removed):
        """The transverse faces a radian-angle PlaneWave needs at its band centre.

        Returns ``{axis: face}`` for the two axes across the wave's plane: a :class:`Bloch`
        face, :class:`Periodic` where the transverse wavevector is zero, or an
        :class:`InvariantAxis` for each axis named in ``invariant_axes`` (``'y'`` or
        ``('x', 'y')``). The transverse wavevector is fixed across the entire pulse spectrum;
        it does not describe constant-angle broadband illumination.
        """
        if 'axis1d' in removed:
            raise TypeError('Bloch.from_source(axis1d=...) is now Bloch.from_source(invariant_axes=...). '
                            'The old keyword was removed, not aliased; see the public-name '
                            'migration guide, docs/migration/public-names.md.')
        if removed:
            raise TypeError(f'Bloch.from_source() does not take {sorted(removed)}. It takes '
                            f"source, units=, background= and invariant_axes=.")
        from wavebit.materials import Material
        from wavebit.materials._lower import lower_material
        from wavebit.units import require_units
        from ._lower_public import _source
        require_units(units)
        medium = Material(epsilon=1.) if background is None else background
        values = _publicly(lambda: _legacy_boundaries.Bloch.from_source(
            _source(source, units), lower_material(medium, units=units), invariant_axes))
        faces = {}
        for axis, value in values.items():
            if isinstance(value, _legacy_boundaries.Bloch):
                faces[axis] = Bloch(phase=value.phase, k=value.k)
            elif isinstance(value, _legacy_boundaries.Axis1D):
                faces[axis] = InvariantAxis(k=value.k, k_max=value.k_max)
            else:
                faces[axis] = value
        return faces


@spec
class MirrorSymmetry:
    """A mirror plane on this face; ``parity`` is the tangential-E eigenvalue, +1 or -1.

    ``fdtd.Boundaries(x=(fdtd.MirrorSymmetry(parity=-1), fdtd.PEC()), ...)`` models half of a
    scene that is mirror-symmetric about the low x face. ``parity=+1`` keeps tangential E even
    across the plane (PMC-like); ``-1`` makes it odd (PEC-like). A mirror goes on at most one
    face of an axis, with PEC or PMC on the other.
    """
    parity: int

    def __post_init__(self):
        object.__setattr__(self, 'parity', _publicly(self._lowered).parity)

    def _lowered(self):
        return _legacy_boundaries.Symmetry(parity=self.parity)


@spec
class InvariantAxis:
    """A one-cell axis the structure does not vary along; the field varies as ``exp(i k s)`` on it.

    ``fdtd.Boundaries(x=fdtd.Bloch(k=kx), y=fdtd.InvariantAxis(k=ky), z=fdtd.Open())`` is a
    conical 1-D grating: the structure is uniform along y and the grid holds one cell of it.
    The derivative along the axis is the exact ``i k``, not a one-cell difference, so the
    grid's dispersion does not change the wavevector. ``k`` is in inverse ``Problem.units``
    lengths; ``k_max`` is the largest ``|k|`` the preparation is made for (default ``|k|``,
    required when ``k`` is traced). The domain must be one cell along the axis.
    """
    k: object = 0.0
    k_max: object = None

    def __post_init__(self):
        object.__setattr__(self, 'k_max', _publicly(self._lowered).k_max)

    def _lowered(self):
        return _legacy_boundaries.Axis1D(k=self.k, k_max=self.k_max)


#: The public faces; a pair may mix any two of them on one axis.
FACES = (Open, Periodic, Bloch, PEC, PMC, MirrorSymmetry, InvariantAxis)


@spec
class Boundaries:
    """The physical condition on each face: ``fdtd.Boundaries(x=..., y=..., z=...)``.

    One face type applies to both ends of an axis; a ``(low, high)`` pair sets them separately.
    ``Boundaries.open()`` and ``Boundaries.periodic(z=...)`` spell the two common cases.
    """
    x: object
    y: object
    z: object

    def __post_init__(self):
        for axis in 'xyz':
            value = getattr(self, axis)
            pair = value if isinstance(value, (tuple, list)) else (value, value)
            if len(pair) != 2 or any(not isinstance(v, FACES) for v in pair):
                raise ValidationError('expected a physical face or (low, high) pair', path='scene.boundaries.'+axis)
            object.__setattr__(self, axis, tuple(pair))

    def face(self, name):
        if name not in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+'):
            raise ValidationError('unknown face', path='scene.boundaries', actual=name)
        return getattr(self, name[0])[name[1] == '+']

    @classmethod
    def open(cls):
        return cls(x=Open(), y=Open(), z=Open())

    @classmethod
    def periodic(cls, *, z):
        return cls(x=Periodic(), y=Periodic(), z=z)
