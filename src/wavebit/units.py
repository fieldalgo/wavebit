"""Physical units at public solver boundaries; internal kernels may be normalized."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from ._common.declarations import Declaration
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
    def length_metres(self):
        return _LENGTH_METRES[self.length]

    @property
    def time_seconds(self):
        """Seconds represented by one normalized kernel time unit."""
        return self.length_metres / C0

    def to_metres(self, length):
        return length * self.length_metres

    def time_to_internal(self, seconds):
        return seconds / self.time_seconds

    def time_from_internal(self, value):
        return value * self.time_seconds

    def rate_to_internal(self, rate):
        return rate * self.time_seconds

    def conductivity_to_internal(self, conductivity):
        return conductivity * self.length_metres / (EPSILON0 * C0)


def require_units(units):
    if not isinstance(units, Units):
        raise ValidationError('expected Units(length=...)', code='units_required', path='units', actual=type(units).__name__)
    return units
