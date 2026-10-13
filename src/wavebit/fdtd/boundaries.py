"""Physical face conditions, independent of the absorbing-grid implementation."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from wavebit._common.declarations import Declaration
from wavebit.errors import ValidationError
from ._legacy_boundaries import Periodic, PEC, PMC, Symmetry, Axis1D

__all__ = ['Open', 'Periodic', 'Bloch', 'PEC', 'PMC', 'Symmetry', 'Axis1D', 'Boundaries']

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Open(Declaration):
    """Outgoing boundary. Uniform boundary media continue into outside padding.

    Finite bodies intersecting this face need an explicitly supported continuation;
    the adapter rejects such intersections instead of replacing them with vacuum.
    """
    pass

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Bloch(Declaration):
    """Bloch phase in radians, or wavevector in inverse geometry-length units."""
    phase: object = None
    k: object = None

    @staticmethod
    def from_source(source, *, units, background=None, axis1d=()):
        """Concrete transverse faces for a radian-angle PlaneWave at its band centre.

        The transverse wavevector is fixed across the entire pulse spectrum;
        it does not describe constant-angle broadband illumination.
        """
        from wavebit.materials import Material
        from wavebit.materials._lower import lower_material
        from wavebit.units import require_units
        from ._lower_public import _source
        from ._legacy_boundaries import Bloch as LegacyBloch
        require_units(units)
        medium=Material(epsilon=1.) if background is None else background
        values=LegacyBloch.from_source(_source(source,units),lower_material(medium,units=units),axis1d)
        return {axis:Bloch(phase=value.phase,k=value.k) if isinstance(value,LegacyBloch) else value
                for axis,value in values.items()}

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Boundaries(Declaration):
    x: object
    y: object
    z: object

    def __post_init__(self):
        allowed = (Open, Periodic, Bloch, PEC, PMC, Symmetry, Axis1D)
        for axis in 'xyz':
            value = getattr(self, axis)
            pair = value if isinstance(value, (tuple, list)) else (value, value)
            if len(pair) != 2 or any(not isinstance(v, allowed) for v in pair):
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
