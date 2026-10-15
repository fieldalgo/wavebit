"""Grid, integration duration, and absorbing implementation choices."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from wavebit._common.declarations import Declaration
from wavebit._common.execution import frozen_mapping
from wavebit.errors import ValidationError

__all__ = ['UniformGrid', 'Duration', 'PML', 'Absorber', 'Decay', 'Numerics']

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
    """Absorber controls and optional fixed physical reference media by face.

    ``reference_media={'z-': Material(...)}`` fixes the stretch profile while a
    continued substrate's physical material varies. It does not replace the
    substrate material. Omitted faces follow their physical boundary medium.
    """
    cells: int = 12
    wavelength: object = None
    reflection: float = 1e-8
    allow_thin: bool = False
    stabilize: object = None
    reference_media: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'reference_media', frozen_mapping(self.reference_media))


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Absorber(Declaration):
    """Graded matched conductivity outside Open faces, for media PML cannot admit.

    This is an explicit fallback, not a PML. Its fixed cubic profile has a
    normal-incidence round-trip amplitude near exp(-3); more cells do not remove
    that attenuation floor. Check absorber and spatial convergence separately.
    """
    cells: int = 12


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Decay(Declaration):
    """Optional state-amplitude stopping inside the Duration hard cap.

    ``window_seconds`` is an SI duration, independent of geometry units. Complete
    windows must pass for fields and every supported memory group after source
    decay. This concrete forward CPU policy is not differentiable or a bound on
    spectral error; fixed-duration execution remains the default.
    """
    threshold: float
    window_seconds: float


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Numerics(Declaration):
    grid: UniformGrid
    duration: Duration
    courant: float = .5
    absorber: PML | Absorber = field(default_factory=PML)
    geometry_ranges: object = field(default_factory=dict)
    discretisation: str = 'voxel'
    junction: str = 'split'
    decay: object = None

    def __post_init__(self):
        object.__setattr__(self, 'geometry_ranges', frozen_mapping(self.geometry_ranges))
