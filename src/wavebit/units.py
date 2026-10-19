"""Physical units at public solver boundaries; internal kernels may be normalized."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from ._common.declarations import Declaration, removed
from .errors import ValidationError

C0 = 299792458.0
EPSILON0 = 8.8541878188e-12
Z0 = 1.0 / (EPSILON0 * C0)
_LENGTH_METRES = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9}


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Units(Declaration):
    length: str = 'm'

    def __post_init__(self):
        if not isinstance(self.length, str) or self.length not in _LENGTH_METRES:
            raise ValidationError("length must be 'm', 'mm', 'um', or 'nm'", code='unknown_length_unit', path='units.length', actual=self.length)

    @property
    def length_unit_meters(self):
        """Meters in one length unit (1e-6 for ``'um'``)."""
        return _LENGTH_METRES[self.length]

    @property
    def time_unit_seconds(self):
        """Seconds in one internal time unit, the time light takes to cross one length unit."""
        return self.length_unit_meters / C0

    # Conversions the solvers make at their own boundaries; not part of the public surface.
    def _time_to_internal(self, seconds):
        return seconds / self.time_unit_seconds

    def _rate_to_internal(self, rate):
        return rate * self.time_unit_seconds

    def _conductivity_to_internal(self, conductivity):
        return conductivity * self.length_unit_meters / (EPSILON0 * C0)

    def __getattr__(self, name):
        # Reached only when normal lookup fails, so it never shadows a field or property.
        if name in _RENAMED:
            new, note = _RENAMED[name]
            raise removed('Units', name, new, note)
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')


#: Units members that were renamed or made private, with what replaces them.
_RENAMED = {
    'length_metres': ('Units.length_unit_meters', None),
    'time_seconds': ('Units.time_unit_seconds', None),
    'to_metres': ('a multiplication by Units.length_unit_meters', None),
    'time_to_internal': ('private', 'divide seconds by Units.time_unit_seconds'),
    'time_from_internal': ('private', 'multiply by Units.time_unit_seconds'),
    'rate_to_internal': ('private', 'multiply a rate in 1/s by Units.time_unit_seconds'),
    'conductivity_to_internal': ('private', 'the solvers convert S/m themselves'),
}


def require_units(units):
    if not isinstance(units, Units):
        raise ValidationError('expected Units(length=...)', code='units_required', path='units', actual=type(units).__name__)
    return units
