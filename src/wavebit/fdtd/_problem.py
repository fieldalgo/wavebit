"""Immutable physical inputs for a time-domain calculation."""
from dataclasses import field
from wavebit._common.declarations import spec
from wavebit.materials import Material
from wavebit.units import Units, require_units
from wavebit.errors import ValidationError
from ._geometry import Domain, Structure
from ._boundaries import Boundaries
from ._monitors import Outputs

__all__ = ['Scene', 'Problem']


@spec
class Scene:
    """The physical domain, its face conditions, the background medium and the structures."""
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


@spec
class Problem:
    """The complete physical FDTD problem: scene, units, sources, requested outputs, initial state."""
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
