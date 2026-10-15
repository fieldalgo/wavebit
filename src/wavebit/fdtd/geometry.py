"""Physical geometry; lengths follow Problem.units and angles are radians."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from wavebit._common.declarations import Declaration
from . import _legacy_geometry, _scene
from ._legacy_geometry import Shape
from ._spelling import Spoken
from ._structure import tuples

__all__ = ['Domain', 'Shape', 'Slab', 'Region', 'Plane', 'Sphere', 'Box', 'Cylinder', 'Prism', 'Structure']


@eager_under_torch_compile
class Domain(Spoken, _scene.Domain):
    """The physical box: ``size=(sx, sy, sz)`` and ``center=(cx, cy, cz)`` in ``Problem.units``.

    Each side is covered by whole cells of the grid, rounded up. An Open face adds its
    absorbing cells outside this box, so the box is what the sources, monitors and structures
    are placed in. Every number is static: it fixes the grid.
    """


@eager_under_torch_compile
class Slab(Spoken, _legacy_geometry.Slab):
    """A layer: one axis named as ``(lo, hi)``, the other two running the whole domain.

    ``fdtd.Slab(z=(-0.2, 0.0))`` is a layer 0.2 thick whose faces are at ``z = -0.2`` and
    ``z = 0``, in ``Problem.units`` and in the scene's coordinates (the domain's centre is
    wherever ``Domain(center=...)`` put it). A slab names exactly one axis; a block is a
    :class:`Region` or a :class:`Box`. The bounds are static: they fix which cells carry the
    material, so a traced bound is refused.

    A slab is **cell-snapped**: each face lands on the nearest cell boundary and the cells
    between carry the material whole. A face between two boundaries moves by up to half a cell,
    a first-order error in the spacing, and :func:`check` reports each move as a
    ``geometry.cell_snapped`` warning with what can stand in for the layer instead. A slab
    whose faces snap to the same boundary is refused rather than run as the background.

    Its other two axes run through the domain's faces, so they cannot be Open. A slab that
    reaches an Open face across its own axis continues into the absorbing layer there.
    """


@eager_under_torch_compile
class Region(Spoken, _legacy_geometry.Region):
    """An axis-aligned box, ``fdtd.Region(size=(sx, sy, sz), center=(cx, cy, cz))``.

    Three uses: the box of a :class:`TFSF` source, the surface of a :class:`Flux` or
    :class:`FarField` monitor or the volume of a :class:`Field` monitor, and the shape of a
    :class:`Structure`. Both triples are static lengths in ``Problem.units`` and the scene's
    coordinates.

    As a shape it is **cell-snapped** like :class:`Slab`, on every axis at once: each face
    lands on the nearest cell boundary, the cells between carry the material whole, a face
    between boundaries moves by up to half a cell (reported by :func:`check`), and the size and
    centre are not differentiable. The smoothed, rotatable and differentiable rectangular body
    is :class:`Box`. A structure may not touch an Open face; across a periodic axis it stays
    inside the period or spans all of it, which is how a lamellar grating spells its uniform
    axis.
    """


@eager_under_torch_compile
class Plane(Spoken, _legacy_geometry.Plane):
    """A plane across one axis, for a source or a monitor: ``fdtd.Plane(z=0.2)``.

    The axis named with a scalar is the plane's normal, and the position snaps to the nearest
    node plane of the grid, where the tangential E lives. The other two axes run the whole
    domain unless given as spans, ``fdtd.Plane(z=0.2, x=(-0.3, 0.3), y=(-0.3, 0.3))``, whose
    ends snap to node planes too. A :class:`Field` or :class:`Flux` monitor reads the spans; a
    :class:`PlaneWave` and a :class:`Diffraction` monitor read the position alone. Every number
    is static: it picks grid indices.
    """

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
    """A named physical body and its shared material model.

    ``continuation`` explicitly extends a fixed, unturned Box through named Open
    faces (for example ``('z-',)`` for a substrate). The Box must fill both
    transverse directions; finite scatterers cannot continue into an absorber.
    """
    name: str
    shape: object
    material: object
    continuation: tuple = ()

    def __post_init__(self):
        from wavebit.errors import ValidationError
        faces = tuple(self.continuation)
        if (len(set(faces)) != len(faces)
                or any(f not in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+') for f in faces)):
            raise ValidationError('continuation needs distinct physical face names such as z-',
                                  code='padding_continuation', path='scene.structures.'+self.name)
        object.__setattr__(self, 'continuation', faces)
