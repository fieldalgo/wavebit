"""What a run records: in stage 0, a point time series and nothing else.

``Record(name, at=(x, y, z), component='Ey')`` reads one component at the Yee point of that
component nearest ``at``, once per step. It reads at the time level the component lives on:
an E component after the E update of step ``n`` (time ``(n + 1) dt``), an H component after
the H update (time ``(n + 1/2) dt``). The series carries those times, so a probe on ``Hz``
is not mistaken for a sample at the E times.

Observers only read. They never touch the fields, which is what lets the kernel step apply
them last and lets the JAX scan carry their accumulators as ordinary state. The DFT, flux
and far-field monitors are stage 3 and have no keyword here yet: ``Record(kind=...)`` or
``Flux(...)`` fail by name rather than by a missing attribute deep in a run.
"""
from ._spec import spec, static_float

__all__ = ['Record', 'COMPONENTS']

#: The six components a monitor may name, in the array's component order per field.
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


@spec(positional=('name',))
class Record:
    """A point time series of one component, at the nearest Yee point of that component."""

    name: str
    at: tuple
    component: str = 'Ey'

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Record(name=...): the name is the key the result is read back '
                             'by, res[name]; it must be a non-empty string.')
        if not (isinstance(self.at, (tuple, list)) and len(self.at) == 3):
            raise ValueError(f'Record(at={self.at!r}): at is a point (x, y, z) in the user\'s '
                             f'length unit, three numbers.')
        # The point picks a cell index, so it is static like the grid it indexes.
        object.__setattr__(self, 'at', tuple(static_float('Record', 'at', v) for v in self.at))
        if self.component not in COMPONENTS:
            raise ValueError(f'Record(component={self.component!r}): the components are '
                             f'{COMPONENTS}.')

    @property
    def field(self):
        """``'E'`` or ``'H'``."""
        return self.component[0]

    @property
    def index(self):
        """0, 1, 2 for x, y, z."""
        return 'xyz'.index(self.component[1])
