"""Requested physical observations. Frequency-domain fields retain raw transforms."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from wavebit._common.declarations import Declaration
from . import _legacy_monitors
from ._spelling import Spoken
from ._structure import tuples

__all__ = ['Record', 'Field', 'Flux', 'Diffraction', 'FarField', 'Outputs']


@eager_under_torch_compile
class Record(Spoken, _legacy_monitors.Record):
    """A point time series of one component, at the nearest Yee point of that component.

    ``fdtd.Record('probe', at=(x, y, z), component='Ey')`` is read back as
    ``result['probe']``. ``at`` is a static point in ``Problem.units`` inside the physical
    domain; ``component`` is one of ``'Ex'``, ``'Ey'``, ``'Ez'``, ``'Hx'``, ``'Hy'``, ``'Hz'``.
    """

@eager_under_torch_compile
@dataclass(frozen=True)
class Field(Declaration):
    name: str
    wavelengths: tuple
    plane: object = None
    box: object = None
    components: tuple = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

    def __post_init__(self):
        for name in ('wavelengths', 'components'):
            object.__setattr__(self, name, tuples(getattr(self, name)))

@eager_under_torch_compile
@dataclass(frozen=True)
class Flux(Declaration):
    """Raw two-sided spectral energy density, not monochromatic watts."""
    name: str
    wavelengths: tuple
    plane: object = None
    box: object = None

    def __post_init__(self):
        object.__setattr__(self, 'wavelengths', tuples(self.wavelengths))

@eager_under_torch_compile
@dataclass(frozen=True)
class Diffraction(Declaration):
    """One outgoing port, normalized to the explicitly named incident source."""
    name: str
    plane: object
    wavelengths: tuple
    source: str
    side: str
    orders: object = None
    reference_plane: object = None
    incident_plane: object = None
    raw: bool = False

    def __post_init__(self):
        for name in ('wavelengths', 'orders'):
            object.__setattr__(self, name, tuples(getattr(self, name)))

@eager_under_torch_compile
@dataclass(frozen=True)
class FarField(Declaration):
    name: str
    wavelengths: tuple
    box: object
    propagation: str = 'continuum'

    def __post_init__(self):
        object.__setattr__(self, 'wavelengths', tuples(self.wavelengths))

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Outputs(Declaration):
    monitors: tuple = ()
    final_fields: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'monitors', tuple(self.monitors))
