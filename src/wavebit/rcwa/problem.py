"""Immutable physical declarations for a periodic, layered RCWA problem.

Layers are bottom-to-top and the bottom interface is z=0. Arrays are borrowed
numerical leaves; callers must not mutate their buffers during a calculation.
"""
from dataclasses import field

import numpy as np

from .._common.declarations import spec
from ..materials import Material
from ..units import Units
from ..errors import ValidationError


def _freeze_sequence(value):
    # Python containers are owned structural values; backend arrays remain
    # borrowed numerical leaves, preserving their differentiation graphs.
    return tuple(_freeze_sequence(v) for v in value) if isinstance(value, (tuple, list)) else value


def _refuse_non_real(shape, name, value):
    """Refuse a string or complex entry in a shape's lengths, centre or angle.

    Each entry is read by the geometry builders as a real number; a complex one reached them
    and was taken for a ``jax.jit`` tracer (it has no real value to read), and a string failed
    inside NumPy. A traced or differentiated entry is judged by its dtype alone.
    """
    for entry in value if isinstance(value, tuple) else (value,):
        dtype = getattr(entry, 'dtype', None)
        kind = None
        if dtype is not None:
            if getattr(dtype, 'is_complex', False):          # a torch dtype
                kind = 'c'
            else:
                try:
                    kind = np.dtype(dtype).kind
                except TypeError:
                    kind = None
        if isinstance(entry, (str, bytes, complex)) or kind in ('c', 'U', 'S', 'O'):
            raise ValidationError(f'{shape}({name}={value!r}): each entry is a real number (a length in '
                                  "the problem's unit, or an angle in radians), not "
                                  f'{type(entry).__name__!r}.', code='shape.value', path=f'geometry.{name}',
                                  expected='real numbers', actual=repr(value))


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
        for name in ('size', 'center', 'angle'):
            _refuse_non_real('Rectangle', name, getattr(self, name))


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
        for name in ('radius', 'center'):
            _refuse_non_real('Circle', name, getattr(self, name))


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
        for name in ('radii', 'center', 'angle'):
            _refuse_non_real('Ellipse', name, getattr(self, name))


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
    of itself or of another region; ``check`` reports that, and ``prepare`` refuses it.

    Vertex coordinates differentiate as other region geometry does: through ``prepared.run``
    on PyTorch and under eager ``jax.grad``, refused under ``jax.jit`` or ``jax.vmap`` (code
    ``prepared.host_topology``). Prepare with
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


def _concrete_scalar(value):
    """``float(value)`` for a concrete real scalar, or ``None`` for a traced or tensor value."""
    import numbers
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in 'fi':
        return float(value)
    return None


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


@spec
class Trapezoid:
    """A line's cross-section that changes with height: the sidewall angle and its rounding.

    For :class:`ProfiledLayer` only. The line is invariant along ``y`` and spans the cell in
    ``y``; its ``x``-``z`` cross-section is a trapezoid whose height is the layer's
    ``thickness``. Every length is in the Problem's unit; ``z`` is measured up from the
    layer's bottom face.

    ``cd`` is the line's width (critical dimension), measured at the height ``cd_at`` names:
    ``'bottom'`` (the default), ``'middle'`` or ``'top'`` of the layer, on the straight
    sidewalls before any rounding. ``center`` is the ``x`` of that width's midpoint, from the
    cell centre.

    ``swa`` is the sidewall angle in radians: the angle between the base and the sidewall,
    measured inside the material. ``pi/2`` is a vertical wall; less than ``pi/2`` narrows
    upward (a taper); more than ``pi/2`` widens upward (an undercut). A pair
    ``(left, right)`` gives each wall its own angle. Internally a wall is offset by
    ``(z - z_cd) * tan(pi/2 - swa)``, which is exactly zero at ``swa = pi/2``, so a vertical
    line is the same structure, to the bit, as a :class:`Rectangle` of width ``cd``.

    ``top_rounding`` is the radius of a convex circular arc replacing each top corner
    (material removed); ``footing`` is the radius of a concave fillet between each wall and
    the base (material added). Each arc is tangent to both faces it joins, so the width is a
    continuously differentiable function of height and of every parameter.

    Each slice of the layer becomes a ``Rectangle`` of the slice's width, on the closed-form
    route: there is no lateral staircase, and the width differentiates on PyTorch and JAX
    as a rectangle's size does. The cell must be given as ``Stack(period=...)``.
    """
    cd: object
    swa: object = 1.5707963267948966
    top_rounding: object = 0.
    footing: object = 0.
    center: object = 0.
    cd_at: str = 'bottom'

    def __post_init__(self):
        if isinstance(self.swa, list):
            object.__setattr__(self, 'swa', tuple(self.swa))
        if isinstance(self.swa, tuple) and len(self.swa) != 2:
            raise ValidationError('Trapezoid swa is one angle or a (left, right) pair.',
                                  code='shape.dimension', path='geometry.swa')
        for name in ('cd', 'swa', 'top_rounding', 'footing', 'center'):
            _refuse_non_real('Trapezoid', name, getattr(self, name))
        if self.cd_at not in ('bottom', 'middle', 'top'):
            raise ValidationError("Trapezoid cd_at is 'bottom', 'middle' or 'top'.",
                                  code='shape.profile', path='geometry.cd_at')
        cd = _concrete_scalar(self.cd)
        if cd is not None and not (np.isfinite(cd) and cd > 0):
            raise ValidationError(f'Trapezoid cd must be a positive length, not {cd!r}.',
                                  code='shape.profile', path='geometry.cd')
        for angle in _pair(self.swa):
            value = _concrete_scalar(angle)
            if value is not None and not (0 < value < np.pi):
                raise ValidationError(f'Trapezoid swa is an angle in radians strictly between 0 and pi '
                                      f'(pi/2 is vertical), not {value!r}.',
                                      code='shape.profile', path='geometry.swa')
        for name in ('top_rounding', 'footing'):
            value = _concrete_scalar(getattr(self, name))
            if value is not None and not (np.isfinite(value) and value >= 0):
                raise ValidationError(f'Trapezoid {name} is a nonnegative radius, not {value!r}.',
                                      code='shape.profile', path=f'geometry.{name}')


@spec
class Loft:
    """A polygonal cross-section interpolated linearly from the bottom outline to the top.

    For :class:`ProfiledLayer` only. ``bottom`` and ``top`` are outlines with the same number
    of vertices, written as for :class:`Polygon` and matched vertex by vertex; at height ``z``
    above the layer's bottom face each vertex is ``bottom + (top - bottom) * z / thickness``.
    A pillar whose top outline is the bottom one shrunk about its centre is a tapered pillar.
    Each slice becomes a :class:`Polygon` on the painting route, with the Polygon's
    factorization rules and derivatives.
    """
    bottom: object
    top: object

    def __post_init__(self):
        for name in ('bottom', 'top'):
            object.__setattr__(self, name, _freeze_sequence(getattr(self, name)))
            Polygon(vertices=getattr(self, name))
        count = (lambda v: v.shape[0] if hasattr(v, 'shape') else len(v))
        if count(self.bottom) != count(self.top):
            raise ValidationError('Loft bottom and top outlines need the same number of vertices, '
                                  'matched in order.', code='shape.vertices', path='geometry.top')


@spec
class Profile:
    """A general cross-section: ``section(z)`` returns the outline at height ``z``.

    For :class:`ProfiledLayer` only. ``z`` is in the Problem's unit, measured up from the
    layer's bottom face, ``0 <= z <= thickness``. ``section`` returns vertices as
    :class:`Polygon` takes them, with the same number of vertices at every height. It is
    called with a Python float when the slicing is planned and with the slice's midpoint
    height (a traced value when the thickness is) when the problem runs, so it should be
    written with operations of the backend it is run on. A problem holding a ``Profile``
    cannot be saved as a job, since a callable has no portable form.
    """
    section: object

    def __post_init__(self):
        if not callable(self.section):
            raise ValidationError('Profile section must be a callable z -> vertices.',
                                  code='shape.profile', path='geometry.section')


#: Region geometries whose cross-section changes with height; a ProfiledLayer holds them.
PROFILE_SHAPES = (Trapezoid, Loft, Profile)


@spec
class Region:
    """A typed shape and its material; later regions paint over earlier regions.

    A shape may be written in any period: its ``center`` (a polygon's vertices) and the same
    plus ``m a1 + n a2`` are one structure, and solve the same to rounding. Overlapping
    regions are painted as written, so two that overlap only through a periodic copy are
    refused with the whole-period move that resolves it; ``factorization='li'`` reads a
    rectangle in the cell wherever it is written and refuses one that crosses the cell edge.
    """
    geometry: object
    material: object

    def __post_init__(self):
        if not isinstance(self.geometry, (Rectangle, Circle, Ellipse, Polygon) + PROFILE_SHAPES):
            raise ValidationError('Use Rectangle, Circle, Ellipse, or Polygon geometry '
                                  '(Trapezoid, Loft or Profile in a ProfiledLayer).',
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
        for i, region in enumerate(self.regions):
            if isinstance(region.geometry, PROFILE_SHAPES):
                raise ValidationError(f'{type(region.geometry).__name__} changes with height; put it in an '
                                      f'rcwa.ProfiledLayer, which slices it, rather than an rcwa.Layer.',
                                      code='layer.profile_geometry', path=f'layer.regions[{i}].geometry')
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
class ProfiledLayer:
    """A layer whose cross-section or material changes with height, sliced by Wavebit.

    Holds regions as :class:`Layer` does, painted in order, and any of them may be a
    :class:`Trapezoid`, :class:`Loft` or :class:`Profile`, whose outline depends on the height
    ``z`` above the layer's bottom face. The layer is solved as a stack of slices, each with
    the cross-section at its midpoint height; ``Numerics(slicing=rcwa.Slicing(...))`` chooses
    how many and where, and ``result.diagnostics['z_slicing']`` reports what was used and the
    estimated error of the staircase in ``z``. The slices are internal: results, layer indices
    and ``thickness`` refer to this one layer. There is no sampled (raster) form: a raster
    quantizes the width to pixels, so a stack holding a sampled layer refuses a ProfiledLayer.

    When every slice would be the same -- a vertical ``Trapezoid`` without rounding, a ``Loft``
    whose two outlines are equal -- the layer is solved as one :class:`Layer` of those
    regions, identical to the bit, unless a geometry value carries a derivative (a JAX
    tracer, or a tensor that requires grad): then it stays sliced, so the derivative sees the
    profile, and agrees with the single layer to round-off.
    """
    thickness: object
    background: object = None
    regions: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'regions', tuple(self.regions))
        if not all(isinstance(r, Region) for r in self.regions):
            raise ValidationError('regions must contain Region declarations.',
                                  code='layer.regions', path='layer.regions')
        if self.background is None:
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
        if not all(isinstance(layer, (Layer, ProfiledLayer)) for layer in self.layers):
            raise ValidationError('layers must contain rcwa.Layer or rcwa.ProfiledLayer declarations.',
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
        from .sources import PlaneWave, IncidentOrders, GaussianBeam, ElectricDipole
        object.__setattr__(self, 'sources', tuple(self.sources))
        object.__setattr__(self, 'wavelength', _freeze_sequence(self.wavelength))
        if not self.sources or not all(isinstance(s, (PlaneWave, IncidentOrders, GaussianBeam,
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
