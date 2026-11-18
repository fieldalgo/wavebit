"""Discretization and algorithm declarations; physical inputs belong to Problem."""
from .._common.declarations import order_pair, spec
from ..errors import ValidationError


@spec
class Slicing:
    """How a :class:`~wavebit.rcwa.ProfiledLayer` is cut into slices along ``z``.

    ``count`` is the number of slices per profiled layer, or ``'auto'`` (the default): the
    preparation solves the stack with ``min_count`` slices and doubles the count until the
    Richardson estimate of the remaining staircase error in the efficiencies falls to
    ``tolerance`` or ``max_count`` is reached, and fixes that count for every run. Each slice
    takes the cross-section at its midpoint height, the midpoint rule, whose error falls as
    ``count**-2`` for a smooth profile.

    ``rule='arclength'`` (the default) places the slice boundaries at equal steps of the
    sidewall's arc length in the ``x``-``z`` plane, so slices crowd where the outline moves
    fastest -- a rounded corner -- and spread where it moves slowly; ``rule='uniform'`` cuts
    equal thicknesses. Boundaries are fractions of the layer's thickness, fixed when the
    problem is prepared, so a run moves only the widths and the thickness, smoothly.

    ``estimate`` says when the staircase error is estimated: ``'prepare'`` (the default) once,
    from the counts the preparation solved; ``'run'`` on every run as well, by also solving
    with half the slices (about 1.5 times the cost); ``None`` not at all. The estimate is
    reported in ``result.diagnostics['z_slicing']``.

    Richardson's deferred approach to the limit: L. F. Richardson, Phil. Trans. R. Soc. A 210,
    307 (1911).
    """
    count: object = 'auto'
    rule: str = 'arclength'
    tolerance: float = 1e-4
    min_count: int = 4
    max_count: int = 64
    estimate: str | None = 'prepare'

    def __post_init__(self):
        import math
        count = self.count
        if not (count == 'auto' or (type(count) is int and count >= 1)):
            raise ValidationError("Slicing count is 'auto' or a positive integer.",
                                  code='numerics.slicing', path='numerics.slicing.count')
        if self.rule not in ('arclength', 'uniform'):
            raise ValidationError("Slicing rule is 'arclength' or 'uniform'.",
                                  code='numerics.slicing', path='numerics.slicing.rule')
        if self.estimate not in ('prepare', 'run', None):
            raise ValidationError("Slicing estimate is 'prepare', 'run' or None.",
                                  code='numerics.slicing', path='numerics.slicing.estimate')
        if type(self.tolerance) not in (float, int) or not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValidationError('Slicing tolerance is a positive absolute efficiency change.',
                                  code='numerics.slicing', path='numerics.slicing.tolerance')
        for name in ('min_count', 'max_count'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValidationError(f'Slicing {name} is a positive integer.',
                                      code='numerics.slicing', path=f'numerics.slicing.{name}')
        if self.max_count < self.min_count:
            raise ValidationError('Slicing max_count must be at least min_count.',
                                  code='numerics.slicing', path='numerics.slicing.max_count')


@spec
class Numerics:
    """RCWA approximation and topology-check policy.

    ``orders`` is the retained Fourier window ``(Mu, Mv)``: an explicit pair of nonnegative
    integers, the half-widths along the first and second lattice vectors.

    ``topology_check='device'`` uses available array guards and refuses other
    traced geometry. ``'host'`` explicitly permits a CPU callback for admitted
    scalar region geometry under JAX transforms; vmap checks run sequentially.

    ``exceptional_points='rebuild'`` (the default) rebuilds layers at and near an exceptional
    point, or warns (eager) / answers NaN (traced) where it cannot. ``'off'`` is for advanced
    use: it skips that work on every path for speed, mostly under ``jax.jit``, and near an
    exceptional point the answer can then be wrong with no warning.

    ``slicing`` is a :class:`Slicing`, read only by a stack holding a
    :class:`~wavebit.rcwa.ProfiledLayer`; ``None`` means ``Slicing()``.

    ``shape_normal_method='interfaces'`` is experimental: under shape NVF it builds every shape
    layer's normal field from the interfaces of the resolved structure, overlapping or not,
    where the default ``'per_shape'`` builds a layer of disjoint shapes shape by shape.
    """
    orders: tuple = (0, 0)
    algorithm: str = 'smm'
    fourier_coefficients: str | None = None
    factorization: str | None = None
    formulation: str = 'auto'
    raster_normal_method: str | None = None
    nvf_projector: str | None = None
    shape_nvf_grid_size: int | None = None
    li_nesting: str | None = None
    detuning: object = None
    topology_check: str = 'device'
    half_space_merge_tolerance: float = 0.0
    exceptional_points: str = 'rebuild'
    wall_resolution: float | None = None
    shape_normal_method: str | None = None
    slicing: object = None
    #: Written to a job only off their default (``jobs._encode``), so a job that does not use
    #: them keeps its key.
    _FIELDS_OMITTED_AT_DEFAULT = ('half_space_merge_tolerance', 'exceptional_points',
                                  'wall_resolution', 'shape_normal_method', 'slicing')

    def __post_init__(self):
        object.__setattr__(self, 'orders', order_pair('Numerics', self.orders,
                                                      code='numerics.orders', path='numerics.orders'))
        choices = {'algorithm': ('smm', 'etm'),
                   'fourier_coefficients': (None, 'sampled', 'integrated'),
                   'factorization': (None, 'li', 'nvf', 'laurent'),
                   'formulation': ('auto', 'first_order', 'second_order'),
                   'raster_normal_method': (None, 'distance', 'contour', 'spectral', 'smooth'),
                   'nvf_projector': (None, 'linear', 'circular'),
                   'li_nesting': (None, 'fixed', 'symmetric'),
                   'shape_normal_method': (None, 'per_shape', 'interfaces'),
                   'topology_check': ('device', 'host'),
                   'exceptional_points': ('rebuild', 'off')}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValidationError(f'{name} must be one of {allowed}.',
                                      code='numerics.choice', path=f'numerics.{name}')
        if self.shape_nvf_grid_size is not None and (type(self.shape_nvf_grid_size) is not int or self.shape_nvf_grid_size < 1):
            raise ValidationError('shape_nvf_grid_size must be a positive integer.',
                                  code='numerics.grid', path='numerics.shape_nvf_grid_size')
        if self.detuning is not None:
            import math
            if type(self.detuning) not in (float, int) or not math.isfinite(self.detuning) or self.detuning <= 0:
                raise ValidationError('detuning is None (exact requested problem) or a positive relative step.',
                                      code='numerics.detuning', path='numerics.detuning')
        import math
        if self.slicing is not None and not isinstance(self.slicing, Slicing):
            raise ValidationError('slicing must be rcwa.Slicing or None.',
                                  code='numerics.slicing', path='numerics.slicing')
        resolution = self.wall_resolution
        if resolution is not None and (type(resolution) not in (float, int)
                                       or not math.isfinite(resolution)
                                       or not 0 < resolution <= 1):
            raise ValidationError('wall_resolution is None (the cell\'s own coordinates) or the '
                                  'slope of the solve\'s coordinates at material walls, in '
                                  '(0, 1]; 0.02 resolves a wall fifty times more finely.',
                                  code='numerics.wall_resolution',
                                  path='numerics.wall_resolution')
        tolerance = self.half_space_merge_tolerance
        if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValidationError('half_space_merge_tolerance is a relative tolerance in [0, 1); '
                                  '0 merges only layers bit-identical to their half-space.',
                                  code='numerics.half_space_merge_tolerance',
                                  path='numerics.half_space_merge_tolerance')

    def _engine_kwargs(self):
        out = dict(truncation_order=self.orders, layer_algorithm=self.algorithm.upper())
        for name in ('fourier_coefficients', 'factorization', 'formulation', 'raster_normal_method',
                     'nvf_projector', 'shape_nvf_grid_size', 'li_nesting', 'shape_normal_method'):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.half_space_merge_tolerance:
            out['half_space_merge_tolerance'] = float(self.half_space_merge_tolerance)
        if self.exceptional_points != 'rebuild':
            out['exceptional_points'] = self.exceptional_points
        if self.wall_resolution is not None:
            out['wall_resolution'] = float(self.wall_resolution)
        return out
