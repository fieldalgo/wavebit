"""Immutable physical inputs for a time-domain calculation."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from wavebit._common.declarations import Declaration
from wavebit.materials import Material
from wavebit.units import Units, require_units
from wavebit.errors import ValidationError
from .geometry import Domain, Structure
from .boundaries import Boundaries
from .monitors import Outputs

__all__ = ['Scene', 'Problem']

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Scene(Declaration):
    domain: Domain
    boundaries: Boundaries
    background: object = field(default_factory=lambda: Material(epsilon=1.))
    structures: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'structures', tuple(self.structures))
        if not isinstance(self.domain, Domain) or not isinstance(self.boundaries, Boundaries):
            raise ValidationError('expected Domain and physical Boundaries', path='scene')
        names = []
        for s in self.structures:
            if not isinstance(s, Structure) or not isinstance(s.name, str) or not s.name:
                raise ValidationError('each Structure needs a nonempty stable name', path='scene.structures')
            names.append(s.name)
        if len(names) != len(set(names)):
            raise ValidationError('structure names must be unique', path='scene.structures')

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Problem(Declaration):
    scene: Scene
    units: Units
    sources: tuple = ()
    outputs: Outputs = field(default_factory=Outputs)
    initial: object = None

    def __post_init__(self):
        require_units(self.units)
        object.__setattr__(self, 'sources', tuple(self.sources))
        if not isinstance(self.scene, Scene) or not isinstance(self.outputs, Outputs):
            raise ValidationError('expected Scene and Outputs', path='problem')
