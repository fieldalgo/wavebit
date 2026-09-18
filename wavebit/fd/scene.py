"""The scene: what a user declares, validated at construction, and nothing derived.

A ``Scene`` is the whole statement of a run -- the box, its six faces, the cell size, the
step count, the excitation and the monitors -- as one frozen object. Everything derived from
it (the cell counts, ``dt``, the face rules, the dtype) lives in the ``Plan`` that
``fd.plan(scene)`` builds, so a scene can be hashed, replaced field by field, and compared,
while the numbers that depend on it are computed once and in one place.

Units, fixed for the whole package: ``c = 1``, ``eps0 = mu0 = 1``, lengths in the user's
unit, so a frequency is ``1 / wavelength`` in the same unit and ``dt`` is a length. H is
stored as ``Z0 * H``, which is what makes the vacuum update coefficient-free apart from
``dt``. Typed triples in the public API are ``(x, y, z)``; the arrays behind them are
``[component, z, y, x]`` (``docs/conventions.md`` has the two axis-order rules; a typed
pair is x-first, an array is z-first).

Validation here is about the declaration, not the grid: sizes positive, the Courant number
in ``(0, 1)``, a monitor inside the box. ``Domain.size`` is snapped **up** to whole cells
when the grid is built, keeping ``center`` fixed, and the report says so; the scene keeps
the size the user asked for.

Every number here is static -- it fixes the cell counts, ``dt``, the scan length and the
skeleton -- so it is concretised at the door through ``_spec.static_float`` /
``static_int``, which refuse a traced value (a JAX tracer, a Torch tensor with
``requires_grad``) by the field's name. The leaves a gradient may flow through are
``InitialField.amplitude``, the initial fields and the Bloch phase, and none of them lives
here.

Named refusals. A keyword a later stage will take is refused here by name -- ``Scene(
sources=...)`` raises ``NotImplementedError`` naming stage 2 -- rather than by the generic
unknown-keyword message, so the reader learns the plan and not only the spelling.
"""
import math

from ._spec import spec, static_float, static_int
from .boundaries import Boundaries
from .monitors import Record
from .sources import InitialField

__all__ = ['Domain', 'Resolution', 'Run', 'Scene', 'COURANT_DEFAULT']

#: The default Courant number; ``dt = courant * dx / sqrt(3)`` in 3-D. 0.5 is a placeholder
#: halfway down the stability limit until a stage-1 study measures where it should sit.
COURANT_DEFAULT = 0.5

#: The Courant number must stay strictly below 1. At exactly 1 the corner Nyquist mode
#: (``k_i dx = pi`` on all three axes) has ``cos(w dt) = -1``, where the leapfrog recurrence
#: ``f(n+1) + f(n-1) = -2 f(n)`` is defective and admits ``(-1)^n (a + b n)``: linear growth
#: in n, quadratic in energy, excited by round-off or any broadband initial data. Measured
#: 2026-09-18 on a periodic 8^3 box from random E of size 1e-3: courant 1.0 multiplies the
#: energy by 1e5 in 2000 steps; 0.999 stays bounded. Stage 1's CPML needs headroom as well.
COURANT_LIMIT = 1.0

#: Scene keywords that belong to a later stage. Refused by name, with the stage.
_LATER = {
    'materials': 'stage 1 (materials other than vacuum)',
    'structure': 'stage 4 (shapes)',
    'shapes': 'stage 4 (shapes)',
    'sources': 'stage 2 (sources)',
    'source': 'stage 2 (sources)',
    'decay': 'stage 7 (Decay)',
    'pml': 'stage 1 (PML / Absorber)',
}


def _triple(where, value, name):
    if not (isinstance(value, (tuple, list)) and len(value) == 3):
        raise ValueError(f'{where}({name}={value!r}): {name} is a triple (x, y, z).')
    out = tuple(static_float(where, name, v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f'{where}({name}={value!r}): every entry must be finite.')
    return out


@spec
class Domain:
    """The box: ``size=(sx, sy, sz)`` and ``center=(cx, cy, cz)``, lengths in the user's unit."""

    size: tuple
    center: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, 'size', _triple('Domain', self.size, 'size'))
        object.__setattr__(self, 'center', _triple('Domain', self.center, 'center'))
        if any(s <= 0 for s in self.size):
            raise ValueError(f'Domain(size={self.size}): every size is a positive length.')


@spec
class Resolution:
    """The uniform cell size: ``dx=...`` directly, or ``cells_per_wavelength=`` with ``wavelength=``.

    Uniform, one ``dx`` for all three axes. A non-uniform grid is not in v1; a per-axis
    cell size is not in stage 0.
    """

    dx: float = None
    cells_per_wavelength: float = None
    wavelength: float = None

    def __post_init__(self):
        by_dx = self.dx is not None
        by_cpw = self.cells_per_wavelength is not None or self.wavelength is not None
        if by_dx == by_cpw:
            raise ValueError('Resolution takes dx=..., or cells_per_wavelength=... with '
                             'wavelength=..., not both and not neither.')
        if by_cpw:
            if self.cells_per_wavelength is None or self.wavelength is None:
                raise ValueError('Resolution(cells_per_wavelength=, wavelength=): both are '
                                 'needed to make a cell size.')
            cpw = static_float('Resolution', 'cells_per_wavelength', self.cells_per_wavelength)
            wavelength = static_float('Resolution', 'wavelength', self.wavelength)
            if cpw <= 0 or wavelength <= 0:
                raise ValueError('Resolution: cells_per_wavelength and wavelength are positive.')
            object.__setattr__(self, 'cells_per_wavelength', cpw)
            object.__setattr__(self, 'wavelength', wavelength)
            object.__setattr__(self, 'dx', wavelength / cpw)
        else:
            object.__setattr__(self, 'dx', static_float('Resolution', 'dx', self.dx))
            if not (self.dx > 0 and math.isfinite(self.dx)):
                raise ValueError(f'Resolution(dx={self.dx}): dx is a positive length.')


@spec
class Run:
    """How long: ``steps``, and the Courant number that sets ``dt = courant * dx / sqrt(3)``.

    ``periods=`` counts optical periods and needs a wavelength, which a stage-0 scene has
    no source to state; it is refused naming stage 2. ``Decay`` is stage 7.
    """

    steps: int = None
    courant: float = COURANT_DEFAULT
    periods: object = None

    def __post_init__(self):
        if self.periods is not None:
            raise NotImplementedError(
                'Run(periods=...) counts optical periods of a source and needs its wavelength; '
                'that arrives with sources in stage 2. Stage 0 takes Run(steps=...).')
        if self.steps is None:
            raise ValueError('Run(steps=...): the step count is required in stage 0.')
        steps = static_int('Run', 'steps', self.steps)
        if steps < 1:
            raise ValueError(f'Run(steps={self.steps!r}): steps is a positive integer.')
        object.__setattr__(self, 'steps', steps)
        object.__setattr__(self, 'courant', static_float('Run', 'courant', self.courant))
        if not (0 < self.courant < COURANT_LIMIT):
            raise ValueError(
                f'Run(courant={self.courant}): dt = courant * dx / sqrt(3) is the 3-D CFL '
                f'condition, stable for 0 < courant < 1 in vacuum and nowhere else. At '
                f'courant = 1 the corner Nyquist mode has cos(w dt) = -1 and grows linearly '
                f'in n (measured: energy x1e5 in 2000 steps on a periodic 8^3 box).')


@spec
class Scene:
    """One run, fully stated. See the module docstring for what is validated where."""

    domain: Domain
    boundaries: Boundaries
    resolution: Resolution
    run: Run
    initial: InitialField = None
    monitors: tuple = ()

    def __post_init__(self):
        for name, cls in (('domain', Domain), ('boundaries', Boundaries),
                          ('resolution', Resolution), ('run', Run)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f'Scene({name}=...): expected fd.{cls.__name__}, got '
                                f'{type(getattr(self, name)).__name__}.')
        if self.initial is not None and not isinstance(self.initial, InitialField):
            raise TypeError(f'Scene(initial=...): expected fd.InitialField, got '
                            f'{type(self.initial).__name__}.')
        if isinstance(self.monitors, Record):
            object.__setattr__(self, 'monitors', (self.monitors,))
        monitors = tuple(self.monitors)
        for m in monitors:
            if not isinstance(m, Record):
                raise TypeError(f'Scene(monitors=...): every monitor is an fd.Record in stage 0; '
                                f'got {type(m).__name__}.')
        names = [m.name for m in monitors]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f'Scene(monitors=...): monitor names are the keys of the result '
                             f'and must be unique; repeated: {dup}.')
        object.__setattr__(self, 'monitors', monitors)
        # A monitor outside the box would silently clip to the nearest edge point.
        lo = tuple(c - s / 2 for c, s in zip(self.domain.center, self.domain.size))
        hi = tuple(c + s / 2 for c, s in zip(self.domain.center, self.domain.size))
        for m in monitors:
            if any(not (a <= p <= b) for p, a, b in zip(m.at, lo, hi)):
                raise ValueError(f'Record({m.name!r}, at={m.at}) lies outside the box '
                                 f'{lo} .. {hi}.')


# ``Scene(sources=...)`` and friends: the spec decorator's generic refusal would suggest the
# nearest spelling; these names are right and merely early, so the refusal names the stage.
_scene_init = Scene.__init__


def _scene_init_with_stages(self, *args, **kwargs):
    later = [k for k in kwargs if k in _LATER]
    if later:
        k = later[0]
        raise NotImplementedError(f'Scene({k}=...) belongs to {_LATER[k]}; stage 0 is the vacuum '
                                  f'kernel with an initial field. It takes '
                                  f'{[f for f in Scene.__dataclass_fields__]}.')
    _scene_init(self, *args, **kwargs)


Scene.__init__ = _scene_init_with_stages
