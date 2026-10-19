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
``[component, z, y, x]`` (``docs/reference/common/conventions.md`` has the two axis-order rules; a typed
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
decay=...)`` raises ``NotImplementedError`` naming ``Run(decay=Decay(...))`` -- rather than by the generic
unknown-keyword message, so the reader learns the plan and not only the spelling.
"""
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
import math

from wavebit._common.declarations import spec
from wavebit._common.declarations import static_float
from wavebit._common.declarations import static_int
from wavebit.fdtd._legacy_boundaries import Boundaries
from wavebit.fdtd._legacy_monitors import MONITOR_TYPES
from wavebit.fdtd._legacy_monitors import Decay
from wavebit.fdtd._legacy_monitors import Record
from wavebit.fdtd._legacy_sources import InitialField
from wavebit.fdtd._public_names import public_name

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

#: Scene keywords that are spelled another way, or belong to a later stage. Refused by name.
_LATER = {
    'materials': "not the stage 1 spelling: a scene has one Scene(background=...) and a "
                 "tuple of Scene(structures=(fd.Structure(shape, material), ...))",
    'structure': 'stage 4 (shapes); one structure is still Scene(structures=(...,))',
    'shapes': 'stage 4 (shapes); stage 1 has fdtd.SnappedSlab, inside a Scene(structures=...)',
    'decay': 'use Numerics(stopping=fdtd.DecayStop(threshold=..., window_seconds=...)) with an '
             'explicit hard cap',
    'source': "not the stage 2 spelling: sources are a tuple, "
              "Scene(sources=(fdtd.CurrentDipole(...), fd.PlaneWave(...), fd.TFSF(...)))",
    'pml': "not the stage 1 spelling: an absorber is a FACE, so it goes in Boundaries -- "
           "fd.Boundaries.open(fd.PML(cells=12, wavelength=...)) or fd.Boundaries(z=fd.PML("
           "...), ...)",
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
    """How long: ``steps``, ``periods`` or ``time``, and the Courant number (``dt = courant dx/sqrt(3)``).

    ``periods=`` (stage 2) counts optical periods of the sources' band centre: the run takes
    ``ceil(periods * wavelength / dt)`` steps, which ``fd.plan`` computes once the grid is
    known. It needs every source to share one concrete band centre, and a scene with no
    source has nothing to count periods of; both are refused by the plan. ``time=`` (stage 3)
    is an end time in the scene's length unit (``c = 1``): ``ceil(time / dt)`` steps, so the
    last E sample is at or just past ``time``. Exactly one of the three is required.
    Optional ``decay=Decay(threshold=..., window=...)`` treats that length as a hard cap
    and can stop at a complete qualifying state-amplitude window. Both criterion values
    are explicit; see :class:`~wavebit.fdtd._legacy.Decay` for its concrete CPU/source restrictions.
    ``res.plan.steps`` is the actual prefix and ``res.diagnostics['stopping']`` retains
    the cap and the reason. Without ``decay`` the fixed-length execution is unchanged.
    """

    steps: int = None
    courant: float = COURANT_DEFAULT
    periods: object = None
    time: object = None
    decay: object = None

    def __post_init__(self):
        if self.decay is not None and type(self.decay) is not Decay:
            raise TypeError('Numerics(stopping=...): use '
                            'fdtd.DecayStop(threshold=..., window_seconds=...).')
        given = [k for k in ('steps', 'periods', 'time') if getattr(self, k) is not None]
        if len(given) > 1:
            raise ValueError(f'Run({", ".join(f"{k}={getattr(self, k)!r}" for k in given)}): '
                             f'one or the other -- one of steps, periods and time; two lengths '
                             f'for one run would have to disagree somewhere.')
        if self.time is not None:
            time = static_float('Run', 'time', self.time)
            if not (time > 0 and math.isfinite(time)):
                raise ValueError(f'Run(time={self.time!r}): a positive end time, in the '
                                 f"scene's length unit (c = 1).")
            object.__setattr__(self, 'time', time)
        elif self.periods is not None:
            periods = static_float('Run', 'periods', self.periods)
            if not (periods > 0 and math.isfinite(periods)):
                raise ValueError(f'Run(periods={self.periods!r}): a positive number of '
                                 f'optical periods.')
            object.__setattr__(self, 'periods', periods)
        else:
            if self.steps is None:
                raise ValueError('Run(steps=...) or Run(periods=...) (or, from stage 3, '
                                 'Run(time=...)): the length of the run is required; periods '
                                 "counts optical periods of the sources' band centre, time is "
                                 'an end time.')
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


_run_init = Run.__init__

#: Other stopping spellings remain unsupported; Decay is an explicit Run field.
_RUN_LATER = ('until', 'shutoff', 'stop')


def _run_init_with_stages(self, *args, **kwargs):
    later = [k for k in kwargs if k in _RUN_LATER]
    if later:
        raise NotImplementedError(
            f'Run({later[0]}=...): use '
            f'Numerics(stopping=fdtd.DecayStop(threshold=..., window_seconds=...)) '
            f'with one explicit steps, periods or time hard cap; arbitrary stopping '
            f'callbacks and these alternate spellings are not supported.')
    _run_init(self, *args, **kwargs)


Run.__init__ = _eager_under_torch_compile(_run_init_with_stages)


@spec
class Scene:
    """One run, fully stated. See the module docstring for what is validated where.

    ``background=`` is what fills every cell no structure claims: a material, a bare
    permittivity, or ``None`` for vacuum. ``structures=`` is a tuple of
    ``fd.Structure(shape, material)``, applied **in order**, so a later one wins where two
    overlap. Both are stage 1; stage 0 scenes are unchanged, because vacuum with no structure
    is exactly what leaving them out means -- and the coefficients it lowers to are the
    stage-0 scalars, bit for bit.
    """

    domain: Domain
    boundaries: Boundaries
    resolution: Resolution
    run: Run
    initial: InitialField = None
    monitors: tuple = ()
    background: object = None
    structures: tuple = ()
    sources: tuple = ()

    def __post_init__(self):
        from wavebit.fdtd._legacy_geometry import Structure
        from wavebit.materials.time_domain import as_material
        object.__setattr__(self, 'background',
                           as_material(self.background, 'Scene(background=)'))
        if isinstance(self.structures, Structure):
            object.__setattr__(self, 'structures', (self.structures,))
        structures = tuple(self.structures)
        for s in structures:
            if not isinstance(s, Structure):
                raise TypeError(f'Scene(structures=...): every entry is an fd.Structure(shape, '
                                f'material); got {public_name(s)}.')
        object.__setattr__(self, 'structures', structures)
        for name, cls in (('domain', Domain), ('boundaries', Boundaries),
                          ('resolution', Resolution), ('run', Run)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f'Scene({name}=...): expected fd.{cls.__name__}, got '
                                f'{public_name(getattr(self, name))}.')
        if self.initial is not None and not isinstance(self.initial, InitialField):
            raise TypeError(f'Scene(initial=...): expected fd.InitialField, got '
                            f'{public_name(self.initial)}.')
        if isinstance(self.monitors, MONITOR_TYPES):
            object.__setattr__(self, 'monitors', (self.monitors,))
        monitors = tuple(self.monitors)
        for m in monitors:
            if not isinstance(m, MONITOR_TYPES):
                raise TypeError(f'Scene(monitors=...): every monitor is an '
                                f'fdtd.TimeSeriesMonitor, fdtd.FluxMonitor, fdtd.FieldMonitor, '
                                f'fdtd.FarFieldMonitor or fdtd.DiffractionMonitor; got '
                                f'{public_name(m)}.')
        names = [m.name for m in monitors]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f'Scene(monitors=...): monitor names are the keys of the result '
                             f'and must be unique; repeated: {dup}.')
        object.__setattr__(self, 'monitors', monitors)
        from wavebit.fdtd._legacy_sources import SOURCE_TYPES
        if isinstance(self.sources, SOURCE_TYPES):
            object.__setattr__(self, 'sources', (self.sources,))
        sources = tuple(self.sources)
        for src in sources:
            if not isinstance(src, SOURCE_TYPES):
                raise TypeError(f'Scene(sources=...): every entry is an fdtd.CurrentDipole, '
                                f'fd.PlaneWave or fd.TFSF; got {public_name(src)}.')
        names = [src.name for src in sources]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f'Scene(sources=...): source names are the keys of '
                             f'res.sources and must be unique; repeated: {dup}.')
        object.__setattr__(self, 'sources', sources)
        # A monitor outside the box would silently clip to the nearest edge point.
        lo = tuple(c - s / 2 for c, s in zip(self.domain.center, self.domain.size))
        hi = tuple(c + s / 2 for c, s in zip(self.domain.center, self.domain.size))
        for m in monitors:
            if isinstance(m, Record):
                if any(not (a <= p <= b) for p, a, b in zip(m.at, lo, hi)):
                    raise ValueError(f'TimeSeriesMonitor({m.name!r}, at={m.at}) lies outside '
                                     f'the box {lo} .. {hi}.')
                continue
            _monitor_inside(m, lo, hi)


def _monitor_inside(m, lo, hi):
    # A region outside the box would clip silently, as a Record would; refused by name.
    kind = public_name(m)
    box = getattr(m, 'box', None)
    plane = getattr(m, 'plane', None)
    if box is not None:
        for a, axis in enumerate('xyz'):
            if box.low[a] < lo[a] - 1e-12 or box.high[a] > hi[a] + 1e-12:
                raise ValueError(f'{kind}({m.name!r}, box=...) reaches {axis} = '
                                 f'{box.low[a]:g} .. {box.high[a]:g}, outside the domain '
                                 f'{lo[a]:g} .. {hi[a]:g}.')
        return
    if plane is not None:
        axis, position = plane.axis, plane.position
    else:
        axis, position = m.axis, m.position
    if not lo[axis] <= position <= hi[axis]:
        raise ValueError(f'{kind}({m.name!r}) sits at {"xyz"[axis]} = {position:g}, outside the '
                         f'domain {lo[axis]:g} .. {hi[axis]:g}.')


# ``Scene(decay=...)`` and friends: the spec decorator's generic refusal would suggest the
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


Scene.__init__ = _eager_under_torch_compile(_scene_init_with_stages)
