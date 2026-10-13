"""Physical geometry; lengths follow Problem.units and angles are radians."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from wavebit._common.declarations import Declaration
from ._legacy_geometry import Slab, Region, Plane
from ._scene import Domain
from ._structure import tuples

__all__ = ['Domain', 'Slab', 'Region', 'Plane', 'Sphere', 'Box', 'Cylinder', 'Prism', 'Structure']

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Sphere(Declaration):
    radius: object
    center: tuple = (0., 0., 0.)

    def __post_init__(self):
        object.__setattr__(self, 'center', tuples(self.center))

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Box(Declaration):
    size: tuple
    center: tuple = (0., 0., 0.)
    angle: float = 0.
    axis: str = 'z'

    def __post_init__(self):
        for name in ('size', 'center'):
            object.__setattr__(self, name, tuples(getattr(self, name)))

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Cylinder(Declaration):
    radius: object
    height: object
    axis: str = 'z'
    center: tuple = (0., 0., 0.)

    def __post_init__(self):
        object.__setattr__(self, 'center', tuples(self.center))

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Prism(Declaration):
    vertices: tuple
    height: object
    axis: str = 'z'
    center: tuple = (0., 0., 0.)
    angle: float = 0.

    def __post_init__(self):
        for name in ('vertices', 'center'):
            object.__setattr__(self, name, tuples(getattr(self, name)))

@eager_under_torch_compile
@dataclass(frozen=True)
class Structure(Declaration):
    """A named physical body and its shared material model."""
    name: str
    shape: object
    material: object
