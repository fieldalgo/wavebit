"""Grid, integration duration, and absorbing implementation choices."""
import math
from dataclasses import field
from wavebit._common.declarations import spec, static_float
from wavebit._common.execution import frozen_mapping
from wavebit.errors import ValidationError

__all__ = ['UniformGrid', 'Duration', 'PML', 'ConductiveAbsorber', 'DecayStop', 'Numerics']


@spec
class UniformGrid:
    """One cubic cell size for all three axes, ``fdtd.UniformGrid(spacing=0.025)``, in ``Problem.units``."""
    spacing: float


@spec
class Duration:
    """How long the run lasts: exactly one of ``seconds``, ``steps`` or ``periods``.

    ``fdtd.Duration(periods=40, source='light')`` counts periods of the named source's centre
    wavelength, so the resolved step count depends on that wavelength.
    """
    seconds: object = None
    steps: object = None
    periods: object = None
    source: object = None

    def __post_init__(self):
        if sum(v is not None for v in (self.seconds, self.steps, self.periods)) != 1:
            raise ValidationError('choose exactly one of seconds, steps, or periods', code='numerics.duration', path='numerics.duration')
        if (self.periods is not None) != (self.source is not None):
            raise ValidationError('periods requires one explicitly named reference source', code='numerics.duration', path='numerics.duration.source')


@spec(renamed={'stabilize': 'stabilizing_loss'})
class PML:
    """Perfectly matched layer: the default absorber behind every Open face.

    ``fdtd.PML(cells=12, wavelength=0.8)``. Its ``cells`` are added outside the physical
    domain. ``reflection`` is the nominal normal-incidence round-trip amplitude the grading is
    aimed at; it is a grading parameter, not a bound on the discrete reflection.
    ``stabilizing_loss`` adds a small matched loss that keeps a mode bound near the face
    from growing; it changes oblique-angle absorption, so it is off (``None``) by default.
    ``reference_media={'z-': Material(...)}`` fixes the stretch profile while a continued
    substrate's physical material varies. It does not replace the substrate material. Omitted
    faces follow their physical boundary medium.
    ``kappa_max`` is the peak real coordinate stretch (finite, static, >= 1; default 2).
    Changing it changes the finite termination and discrete reflection. Converge layer
    thickness, physical clearance and run duration together; it is not a stability guarantee.
    """
    cells: int = 12
    wavelength: object = None
    reflection: float = 1e-8
    allow_thin: bool = False
    stabilizing_loss: object = None
    reference_media: object = field(default_factory=dict)
    kappa_max: float = 2.0
    _FIELDS_OMITTED_AT_DEFAULT = ('kappa_max',)

    def __post_init__(self):
        try:
            kappa = static_float('PML', 'kappa_max', self.kappa_max)
            if not (math.isfinite(kappa) and kappa >= 1.0):
                raise ValueError('PML(kappa_max=...): a finite real number >= 1 is required.')
        except ValueError as exc:
            raise ValidationError(str(exc), code='numerics.pml.kappa_max',
                                  path='numerics.absorber.kappa_max') from exc
        object.__setattr__(self, 'kappa_max', kappa)
        object.__setattr__(self, 'reference_media', frozen_mapping(self.reference_media))


@spec
class ConductiveAbsorber:
    """A graded matched conductivity behind every Open face, for media a PML cannot match.

    This is an explicit fallback, not a PML. Its fixed cubic profile has a normal-incidence
    round-trip amplitude near exp(-3); more cells do not remove that attenuation floor. Check
    absorber and spatial convergence separately.
    """
    cells: int = 12


@spec
class DecayStop:
    """Stop the run early once its state has decayed, inside the :class:`Duration` hard cap.

    ``fdtd.Numerics(..., stopping=fdtd.DecayStop(threshold=1e-5, window_seconds=2e-15))``.
    ``window_seconds`` is an SI duration, independent of geometry units. Complete windows must
    pass for fields and every supported memory group after source decay, including
    PlaneWave/TFSF incident columns and layered TFSF drivers. Their E/H, CPML and supported
    material memories keep independent peaks; tiny residuals can legitimately keep the run at
    its cap. This concrete forward CPU policy is not differentiable or a bound on spectral
    error; fixed-duration execution remains the default.
    """
    threshold: float
    window_seconds: float


@spec(renamed={'discretisation': 'discretization', 'decay': 'stopping'})
class Numerics:
    """Numerical policy for the physical scene: grid, duration, absorber and smoothing rules.

    ``discretization`` chooses how smoothed bodies are discretized: ``'voxel'`` (the default,
    each cell's coefficients from the fraction of it the body fills) or ``'stencil'`` (an
    extended stencil, for an isolated Sphere). Cell-snapped shapes always fill whole cells.

    ``junction`` is the rule where a smoothed body stands flush on another body's face:
    ``'split'`` (default), ``'staircase'``, or ``'refuse'``, applied to each supported flush
    pair's standing body. Substrates and unrelated structures retain their ordinary rule. A
    non-default policy without a supported flush pair is an error at ``numerics.junction``.
    Traced junction geometry remains split-only.

    ``stopping`` optionally ends the run early (:class:`DecayStop`); ``duration`` stays the
    hard cap.

    ``memory_bound`` optionally sets the stencil's predicted host preparation limit in bytes.
    It requires ``discretization='stencil'`` with a Sphere and a positive finite real number
    (not a bool). ``None`` keeps the engine's default limit. This prediction is not a cap on
    total process memory.

    ``checkpoint_steps`` optionally recomputes blocks during reverse-mode differentiation of
    fixed-duration Torch CUDA runs. A positive static integer selects the block length;
    ``None`` preserves the ordinary scan. This is a compute/memory tradeoff, not a memory cap.
    Observation and source histories are retained, and a block longer than the run leaves only
    an ordinary tail.
    """
    grid: UniformGrid
    duration: Duration
    courant: float = .5
    absorber: PML | ConductiveAbsorber = field(default_factory=PML)
    geometry_ranges: object = field(default_factory=dict)
    discretization: str = 'voxel'
    junction: str = 'split'
    stopping: object = None
    memory_bound: object = None
    checkpoint_steps: object = None
    #: Written to a job only off their default (``jobs._encode``), so a job that does not use
    #: them keeps its key.
    _FIELDS_OMITTED_AT_DEFAULT = ('memory_bound', 'checkpoint_steps')

    def __post_init__(self):
        object.__setattr__(self, 'geometry_ranges', frozen_mapping(self.geometry_ranges))
