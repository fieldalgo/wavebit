"""Discretization and algorithm declarations; physical inputs belong to Problem."""
from .._common.declarations import order_pair, spec
from ..errors import ValidationError


@spec
class Numerics:
    """RCWA approximation and topology-check policy.

    ``orders`` is the retained Fourier window ``(Mu, Mv)``: an explicit pair of nonnegative
    integers, the half-widths along the first and second lattice vectors.

    ``topology_check='device'`` uses available array guards and refuses other
    traced geometry. ``'host'`` explicitly permits a CPU callback for admitted
    scalar region geometry under JAX transforms; vmap checks run sequentially.
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
    #: Written to a job only off their default (``jobs._encode``), so a job that does not use
    #: them keeps its key.
    _FIELDS_OMITTED_AT_DEFAULT = ('half_space_merge_tolerance',)

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
                   'topology_check': ('device', 'host')}
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
        tolerance = self.half_space_merge_tolerance
        if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValidationError('half_space_merge_tolerance is a relative tolerance in [0, 1); '
                                  '0 merges only layers bit-identical to their half-space.',
                                  code='numerics.half_space_merge_tolerance',
                                  path='numerics.half_space_merge_tolerance')

    def _engine_kwargs(self):
        out = dict(truncation_order=self.orders, layer_algorithm=self.algorithm.upper())
        for name in ('fourier_coefficients', 'factorization', 'formulation', 'raster_normal_method',
                     'nvf_projector', 'shape_nvf_grid_size', 'li_nesting'):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.half_space_merge_tolerance:
            out['half_space_merge_tolerance'] = float(self.half_space_merge_tolerance)
        return out
