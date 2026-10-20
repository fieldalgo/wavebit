"""Fields with explicit sample coordinates, components, units and staggering."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from wavebit._common.declarations import Declaration, renamed_attributes, spec
from wavebit._common.pytrees import register_dataclass
from wavebit._common.execution import frozen_mapping


def _component_units(components, coordinates, *, electric='V/m', magnetic='A/m', **extra):
    """The one ``FieldSamples.units`` shape: each component's unit and the coordinates' unit.

    ``components`` are 'Ex'..'Hz'; E components take ``electric`` and H components
    ``magnetic``. ``extra`` adds a unit for another sampled axis (``time``), only where no
    attribute name already carries it.
    """
    return {**{c: magnetic if c.startswith('H') else electric for c in components},
            'coordinates': coordinates, **extra}


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class FieldSamples(Declaration):
    """Field values with their sample coordinates and components.

    ``units`` maps each component present ('Ex'..'Hz') to its unit and ``'coordinates'`` to the
    length unit of the sample coordinates.
    """
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


#: RawFieldSpectrum fields that were renamed (fieldalgo/wavebit#57), and their replacements.
_SPECTRUM_RENAMED = {'frequencies': 'frequencies_hz', 'spectrum': 'sidedness'}


@spec(renamed=_SPECTRUM_RENAMED)
class RawFieldSpectrum(FieldSamples):
    """Raw Fourier fields; units and transform normalization remain explicit.

    ``frequencies_hz`` are the frequencies of the wavelength axis, in Hz; ``sidedness`` says
    which spectrum convention the values follow ('two-sided').
    """
    frequencies_hz: object = None
    wavelengths: object = None
    transform: str = 'integral exp(+i 2 pi f t) dt'
    window: object = None
    sidedness: str = 'two-sided'

    __getattr__ = renamed_attributes(_SPECTRUM_RENAMED)


#: TimeSeries fields and properties that were renamed, and their replacements.
_TIME_RENAMED = {'times': 'times_seconds', 'time': 'times_seconds'}


@spec(renamed={'times': 'times_seconds'})
class TimeSeries(FieldSamples):
    """Time-domain samples with an explicit time axis, ``times_seconds``, in seconds."""
    times_seconds: object = None

    __getattr__ = renamed_attributes(_TIME_RENAMED)
