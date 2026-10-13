"""Fields with explicit sample coordinates, components, units and staggering."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from wavebit._common.declarations import Declaration
from wavebit._common.pytrees import register_dataclass
from wavebit._common.execution import frozen_mapping


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class FieldSamples(Declaration):
    coordinates: object
    values: object
    components: tuple
    units: object
    sampling: str
    metadata: object = field(default_factory=dict)

    def __post_init__(self):
        register_dataclass(type(self))
        object.__setattr__(self, 'components', tuple(self.components))
        for name in ('coordinates', 'metadata'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))
        if isinstance(self.units, dict):
            object.__setattr__(self, 'units', frozen_mapping(self.units))


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class SpectralFieldSamples(FieldSamples):
    """Raw Fourier fields; units and transform normalization remain explicit."""
    frequencies: object = None
    wavelengths: object = None
    transform: str = 'integral exp(+i 2 pi f t) dt'
    window: object = None
    spectrum: str = 'two-sided'


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class TimeSeries(FieldSamples):
    """Time-domain samples with an explicit time axis measured in seconds."""
    times: object = None

    @property
    def time(self):
        """Canonical public time axis, in seconds."""
        return self.times
