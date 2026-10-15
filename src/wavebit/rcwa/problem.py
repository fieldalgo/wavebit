"""Immutable physical declarations for a periodic, layered RCWA problem.

Layers are bottom-to-top and the bottom interface is z=0. Arrays are borrowed
numerical leaves; callers must not mutate their buffers during a calculation.
"""
from dataclasses import field

from .._common.declarations import spec
from ..materials import Material
from ..units import Units
from ..errors import UnsupportedFeatureError, ValidationError


def _freeze_sequence(value):
    # Python containers are owned structural values; backend arrays remain
    # borrowed numerical leaves, preserving their differentiation graphs.
    return tuple(_freeze_sequence(v) for v in value) if isinstance(value, (tuple, list)) else value


@spec
class Rectangle:
    """A centered rectangle with full widths and a counterclockwise angle in radians."""
    size: tuple
    center: tuple = (0., 0.)
    angle: object = 0.

    def __post_init__(self):
        object.__setattr__(self, 'size', tuple(self.size))
        object.__setattr__(self, 'center', tuple(self.center))
        if len(self.size) != 2 or len(self.center) != 2:
            raise ValidationError('Rectangle size and center each need two components.',
                                  code='shape.dimension', path='geometry')


@spec
class Circle:
    """A circle, explicitly specified by radius in the Problem's length unit."""
    radius: object
    center: tuple = (0., 0.)

    def __post_init__(self):
        object.__setattr__(self, 'center', tuple(self.center))
        if len(self.center) != 2:
            raise ValidationError('Circle center needs two components.',
                                  code='shape.dimension', path='geometry.center')


@spec
class Ellipse:
    """An ellipse with principal radii and counterclockwise rotation in radians."""
    radii: tuple
    center: tuple = (0., 0.)
    angle: object = 0.

    def __post_init__(self):
        object.__setattr__(self, 'radii', tuple(self.radii))
        object.__setattr__(self, 'center', tuple(self.center))
        if len(self.radii) != 2 or len(self.center) != 2:
            raise ValidationError('Ellipse radii and center each need two components.',
                                  code='shape.dimension', path='geometry')


@spec
class Polygon:
    """Reserved: a polygon given by its vertices is not built yet, and is refused here.

    The engine builds rectangles, circles and ellipses, and the polygons their overlaps
    resolve into; it has no region whose outline is a list of vertices. The overlap test
    reads convex primitives and the normal field is built per primitive type, so a polygon
    needs both before it can be solved. Until then a declaration is refused at once with
    ``UnsupportedFeatureError`` rather than accepted here and refused at ``run``.
    """
    vertices: object

    def __post_init__(self):
        raise UnsupportedFeatureError(
            'rcwa.Polygon is not built yet: a region outline given by vertices has no '
            'engine path. Compose the shape from Rectangle, Circle and Ellipse regions '
            '(later regions paint over earlier ones), or sample it into a raster layer, '
            'rcwa.Layer(thickness=..., epsilon=<array>).',
            code='shape.polygon', path='geometry')


@spec
class Region:
    """A typed shape and its material; later regions paint over earlier regions."""
    geometry: object
    material: object

    def __post_init__(self):
        if not isinstance(self.geometry, (Rectangle, Circle, Ellipse)):
            raise ValidationError('Use Rectangle, Circle, or Ellipse geometry.',
                                  code='shape.type', path='region.geometry')


@spec
class Layer:
    """A physical layer described by regions or sampled epsilon (and optional mu).

    epsilon is a sampled constitutive field, not an array of Material objects.
    A raster layer cannot also contain a background or regions.
    """
    thickness: object
    background: object = None
    regions: tuple = ()
    epsilon: object = None
    mu: object = None

    def __post_init__(self):
        object.__setattr__(self, 'regions', tuple(self.regions))
        for name in ('epsilon', 'mu'):
            object.__setattr__(self, name, _freeze_sequence(getattr(self, name)))
        if not all(isinstance(r, Region) for r in self.regions):
            raise ValidationError('regions must contain Region declarations.',
                                  code='layer.regions', path='layer.regions')
        if self.epsilon is not None:
            if self.background is not None or self.regions:
                raise ValidationError('A sampled layer cannot also declare a background or regions.',
                                      code='layer.ambiguous', path='layer')
        elif self.mu is not None:
            raise ValidationError('Put permeability in each Material; layer.mu belongs to sampled epsilon.',
                                  code='layer.mu_without_raster', path='layer.mu')
        elif self.background is None:
            object.__setattr__(self, 'background', Material(epsilon=1.))


@spec
class Stack:
    """Bottom-to-top finite layers between two half spaces.

    Supply period=(Lx,Ly) for a rectangular cell, or lattice for a 2x2 basis
    whose columns are the lattice vectors. Naming both is ambiguous.
    """
    layers: tuple = ()
    top: object = field(default_factory=lambda: Material(epsilon=1.))
    bottom: object = field(default_factory=lambda: Material(epsilon=1.))
    period: object = None
    lattice: object = None

    def __post_init__(self):
        object.__setattr__(self, 'layers', tuple(self.layers))
        if not all(isinstance(layer, Layer) for layer in self.layers):
            raise ValidationError('layers must contain rcwa.Layer declarations.',
                                  code='stack.layers', path='stack.layers')
        if self.period is not None and self.lattice is not None:
            raise ValidationError('Supply either period or lattice, not both.',
                                  code='stack.cell_ambiguous', path='stack')
        if self.period is None and self.lattice is None:
            object.__setattr__(self, 'period', (1., 1.))
        for name in ('period', 'lattice'):
            object.__setattr__(self, name, _freeze_sequence(getattr(self, name)))


@spec
class Excitation:
    """Vacuum wavelength(s) and the sources that own direction and polarization."""
    wavelength: object
    sources: tuple

    def __post_init__(self):
        from .sources import PlaneWave, OrderExcitation, GaussianBeam, ElectricDipole
        object.__setattr__(self, 'sources', tuple(self.sources))
        object.__setattr__(self, 'wavelength', _freeze_sequence(self.wavelength))
        if not self.sources or not all(isinstance(s, (PlaneWave, OrderExcitation, GaussianBeam,
                                                       ElectricDipole)) for s in self.sources):
            raise ValidationError('sources must be a nonempty tuple of RCWA source declarations.',
                                  code='excitation.sources', path='excitation.sources')


@spec
class Outputs:
    """Requested observations; reconstruction and full scattering state are opt-in.

    power_region is 'cell', 'infinite', or ((xmin,xmax),(ymin,ymax)). A finite
    region is required for localized beams and isolated dipoles.
    """
    diffraction: bool = True
    fields: bool = False
    scattering: bool = False
    power_region: object = None

    def __post_init__(self):
        for name in ('diffraction', 'fields', 'scattering'):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f'{name} must be a boolean.', code='outputs.boolean',
                                      path=f'outputs.{name}')
        if isinstance(self.power_region, (tuple, list)):
            object.__setattr__(self, 'power_region', tuple(tuple(v) for v in self.power_region))


@spec
class Problem:
    """The complete physical RCWA problem; no backend or mutable solver state."""
    stack: Stack
    excitation: Excitation
    units: Units = field(default_factory=Units)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        for name, cls in (('stack', Stack), ('excitation', Excitation), ('units', Units),
                          ('outputs', Outputs)):
            if not isinstance(getattr(self, name), cls):
                raise ValidationError(f'{name} must be {cls.__name__}.',
                                      code='problem.type', path=name)
