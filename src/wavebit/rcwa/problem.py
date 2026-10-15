"""Immutable physical declarations for a periodic, layered RCWA problem.

Layers are bottom-to-top and the bottom interface is z=0. Arrays are borrowed
numerical leaves; callers must not mutate their buffers during a calculation.
"""
from dataclasses import field

from .._common.declarations import spec
from ..materials import Material
from ..units import Units
from ..errors import ValidationError


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
    """A polygon given by its vertices, in centered Cartesian cell coordinates.

    ``vertices`` is a sequence of ``N >= 3`` ``(x, y)`` pairs, or an ``(N, 2)`` array, in the
    Problem's length unit and measured from the center of the cell -- the frame a Rectangle's
    ``center`` is given in -- listed in order around the boundary. Either winding is accepted
    and the order is kept as given. The outline must be simple: no edge of zero length, none
    doubling back, no two edges crossing or touching, and a nonzero area. A concrete outline
    is checked here; a traced one when the problem is prepared on concrete values.

    The edges are transformed in closed form, so a tilted edge has no staircase, and a layer
    holding a polygon is always built on the painting route: regions still paint in order,
    later over earlier. A polygon may reach past the cell edge, but not into a periodic image
    of itself or of another region; that is refused when the problem is prepared.

    Vertex coordinates differentiate as other region geometry does: through ``prepared.run``
    on PyTorch, refused under a JAX transform (code ``prepared.host_topology``). Prepare with
    the spelling you run with -- an ``(N, 2)`` array and ``N`` pairs are different layouts --
    and a step that changes which edges cross, or moves an axis-aligned edge off its axis,
    needs a new preparation.

    Under the default NVF factorization the normal field is the painting route's
    nearest-interface field rather than the closed-form field a lone Rectangle gets, so a
    rectangle written as a polygon matches Rectangle to round-off under
    ``factorization='laurent'`` and converges to it under NVF; on one rotated rectangle the
    per-order efficiencies differ by 7.1e-3 at ``orders=(4, 4)`` and 4.8e-4 at ``(8, 8)``.
    That field also limits the vertex derivative: it jumps across its medial axis, and the
    derivative misses the jumps' share -- measured over a window, 10% of the change at
    ``orders=(3, 3)``, 7% at ``(6, 6)`` and 3% at ``(9, 9)`` on an L-shaped outline, whatever
    ``shape_nvf_grid_size``. Under ``factorization='laurent'`` the derivative is exact.
    ``factorization='li'``, ``nvf_projector='circular'`` and tensor materials do not build a
    polygon and refuse it by name.
    """
    vertices: object

    def __post_init__(self):
        from .._common.geometry.polygon import outline_values, polygon_defect
        from .._common.geometry.recipes import outline_layout
        object.__setattr__(self, 'vertices', _freeze_sequence(self.vertices))
        problem = outline_layout(self.vertices)
        if problem is not None:
            raise ValidationError(f'Polygon vertices {problem}.', code='shape.vertices',
                                  path='geometry.vertices')
        try:
            values = outline_values(self.vertices)
        except TypeError:
            return                       # traced: checked when prepared on concrete values
        except ValueError as error:
            raise ValidationError(f'Polygon {error}.', code='shape.vertices',
                                  path='geometry.vertices') from None
        defect = polygon_defect(values)
        if defect is not None:
            raise ValidationError(f'Polygon outline is not simple: {defect}.',
                                  code='shape.outline', path='geometry.vertices')


@spec
class Region:
    """A typed shape and its material; later regions paint over earlier regions."""
    geometry: object
    material: object

    def __post_init__(self):
        if not isinstance(self.geometry, (Rectangle, Circle, Ellipse, Polygon)):
            raise ValidationError('Use Rectangle, Circle, Ellipse, or Polygon geometry.',
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
