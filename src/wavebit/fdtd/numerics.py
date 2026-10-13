"""Grid, integration duration, and absorbing implementation choices."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from wavebit._common.declarations import Declaration
from wavebit._common.execution import frozen_mapping
from wavebit.errors import ValidationError

__all__ = ['UniformGrid', 'Duration', 'PML', 'Numerics']

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class UniformGrid(Declaration):
    spacing: float

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Duration(Declaration):
    seconds: object = None
    steps: object = None
    periods: object = None
    source: object = None

    def __post_init__(self):
        if sum(v is not None for v in (self.seconds, self.steps, self.periods)) != 1:
            raise ValidationError('choose exactly one of seconds, steps, or periods', path='numerics.duration')
        if (self.periods is not None) != (self.source is not None):
            raise ValidationError('periods requires one explicitly named reference source', path='numerics.duration.source')

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class PML(Declaration):
    cells: int = 12
    wavelength: object = None
    reflection: float = 1e-8
    allow_thin: bool = False
    stabilize: object = None

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Numerics(Declaration):
    grid: UniformGrid
    duration: Duration
    courant: float = .5
    absorber: PML = field(default_factory=PML)
    geometry_ranges: object = field(default_factory=dict)
    discretisation: str = 'voxel'
    junction: str = 'split'

    def __post_init__(self):
        object.__setattr__(self, 'geometry_ranges', frozen_mapping(self.geometry_ranges))
