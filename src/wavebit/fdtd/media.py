"""Media only a time-domain solver can express: a permittivity that changes during the run."""
import math
from dataclasses import dataclass

from wavebit._common.backend import _concrete
from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import Declaration
from wavebit.errors import ValidationError
from wavebit.materials.models import _register, _scalar

__all__ = ['SwitchedMaterial']


@eager_under_torch_compile
@dataclass(frozen=True)
class SwitchedMaterial(Declaration):
    """A lossless isotropic medium whose permittivity jumps once: a temporal boundary.

    ``epsilon_before`` holds until ``switch_time`` (seconds, from the start of the run) and
    ``epsilon_after`` from then on. The electric displacement and the magnetic field are
    continuous across the jump, so a wave inside keeps its wavenumber, changes its frequency by
    ``n_before / n_after`` and splits into a forward and a backward wave with field amplitudes
    ``n_before (n_after +- n_before) / (2 n_after^2)`` of the incident one.

    FDTD only, on a cell-snapped :class:`~wavebit.fdtd.Slab` or :class:`~wavebit.fdtd.Region`
    that overlaps no other structure, stays clear of the absorbing layers and holds no source
    or diffraction/far-field reading. Both permittivities may be traced (differentiated); the
    switching time is static.
    """

    epsilon_before: object
    epsilon_after: object
    switch_time: object

    def __post_init__(self):
        _scalar(self.epsilon_before, 'material.epsilon_before', positive=True)
        _scalar(self.epsilon_after, 'material.epsilon_after', positive=True)
        time = _concrete(self.switch_time) if not isinstance(self.switch_time, (int, float)) \
            else self.switch_time
        if time is None or isinstance(time, bool):
            raise ValidationError('switch_time is a static number of seconds; it decides the step '
                                  'the jump falls on and cannot be traced',
                                  code='static_input', path='material.switch_time')
        time = float(time)
        if not math.isfinite(time) or time <= 0:
            raise ValidationError('switch_time must be a finite positive number of seconds',
                                  code='material_scalar', path='material.switch_time',
                                  actual=self.switch_time)
        object.__setattr__(self, 'switch_time', time)
        _register(type(self), ('epsilon_before', 'epsilon_after'), static=('switch_time',))


def lower_switched(model, *, units):
    """The engine's :class:`~wavebit.materials.time_domain.Switched`, the time in scene units."""
    from wavebit.materials import time_domain as td
    return td.Switched(eps=model.epsilon_before, after=model.epsilon_after,
                       time=units.time_to_internal(model.switch_time))
