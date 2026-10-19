"""Physical, explicitly sampled Yee-grid problems.

Arrays use (z,y,x), with Cartesian vector components on a leading axis. Geometry
lengths use ``Problem.units``; time, current, conductivity and pole rates use SI.
This is the native-grid formulation, including user-supplied absorbing cells.
"""
from dataclasses import field
import numpy as np

from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import spec
from wavebit._common.execution import Execution, CheckReport, frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from wavebit.observations import TimeSeries, SpectralFieldSamples, Validity
from wavebit.units import require_units, Z0, EPSILON0, C0
from ._numerics import ConductiveAbsorber


@spec
class Oscillator:
    """Sampled SI Lorentz/Drude pole: P'' + gamma P' + omega_0² P = epsilon0 strength E.

    All three coefficients may broadcast to (3,z,y,x). Rates are rad/s or 1/s;
    strength is 1/s². ``omega_0=0`` gives a Drude pole. Initial P and P' are zero.
    """
    omega_0: object
    gamma: object
    strength: object


@spec
class Medium:
    """Relative epsilon-infinity/mu and electric conductivity in S/m at Yee sites."""
    epsilon: object
    mu: object = 1.
    conductivity: object = 0.
    poles: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'poles', tuple(self.poles))
        if any(not isinstance(p, Oscillator) for p in self.poles):
            raise ValidationError('use native.Oscillator with SI coefficients', path='medium.poles')


@spec
class Grid:
    """Static (dz,dy,dx) spacing; optional z widths include the closing cell."""
    spacing: tuple
    z_spacing: object = None


@spec
class Current:
    """Electric current density pattern in A/m² times dimensionless time samples.

    ``samples[n]`` drives the electric update ending at (n+1)*dt. Interpolating
    an analytic waveform onto this staggered update is the caller's choice.
    """
    density: object
    samples: object


@spec
class Initial:
    """E in V/m at t=0; H in A/m at t=-dt/2, each shaped (3,z,y,x)."""
    E: object = None
    H: object = None


@spec
class Record:
    """Native field snapshots; optional (z,y,x) bounds are (start,stop,step).

    Bounds use integer indices, preserve singleton axes and do not crop the solve.
    Components are lowercase ex/ey/ez/hx/hy/hz. Recorded time is always SI seconds.
    """
    every: int = 1
    components: tuple = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
    bounds: object = None


@spec
class Spectrum:
    """Streaming Fourier integrals on collocated planes at integer z nodes."""
    wavelengths: object
    planes: tuple


@spec
class Outputs:
    spectrum: object = None
    record: object = None


@spec
class PML:
    """CPML within the supplied grid's two z ends; x/y remain Bloch-periodic.

    End media are relative (epsilon,mu) pairs in low/high z order. Each end and
    one extra interior cell must match its nondispersive, nonconducting medium.
    """
    cells: int = 12
    wavelength: float = 1.
    media: object = ((1., 1.), (1., 1.))
    reflection: float = 1e-8


@spec
class Tail:
    """Final-window decay diagnostic, without early stopping or a CPML-memory test."""
    window_seconds: float
    tolerance: float = 1e-4


@spec
class Numerics:
    grid: Grid
    time_step_seconds: float
    steps: int
    absorber: object = None
    checkpoint_steps: int = 32
    dft_subsample: int = 1
    tail: object = None

    def __post_init__(self):
        if not isinstance(self.grid, Grid):
            raise ValidationError('expected native.Grid', path='numerics.grid')
        if self.absorber is not None and not isinstance(self.absorber, (PML, ConductiveAbsorber)):
            raise ValidationError('expected native.PML or fdtd.ConductiveAbsorber', path='numerics.absorber')
        if self.tail is not None and not isinstance(self.tail, Tail):
            raise ValidationError('expected native.Tail', path='numerics.tail')


@spec
class Problem:
    medium: Medium
    units: object
    current: object = None
    initial: Initial = field(default_factory=Initial)
    bloch_phase: object = (0., 0., 0.)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        require_units(self.units)
        for name, typ in (('medium', Medium), ('initial', Initial), ('outputs', Outputs)):
            if not isinstance(getattr(self, name), typ):
                raise ValidationError(f'expected native.{typ.__name__}', path=name)
        if self.current is not None and not isinstance(self.current, Current):
            raise ValidationError('expected native.Current', path='current')
        for name, typ in (('spectrum', Spectrum), ('record', Record)):
            value = getattr(self.outputs, name)
            if value is not None and not isinstance(value, typ):
                raise ValidationError(f'expected native.{typ.__name__}', path='outputs.'+name)


def _execution(backend, device, precision):
    execution = Execution(backend=backend, device=device, precision=precision).validate()
    if precision != 'double':
        raise UnsupportedFeatureError('native execution currently requires double precision', path='execution.precision')
    return execution


def _arguments(problem, numerics, execution):
    if not isinstance(problem, Problem) or not isinstance(numerics, Numerics):
        raise ValidationError('expected native.Problem and native.Numerics', path='problem')
    from wavebit._common.backend import as_array
    from ._periodic.runtime import time_loop_backend
    device = execution.device
    if execution.backend == 'jax':
        import jax
        platform, _, index = device.partition(':')
        device = jax.devices(platform)[int(index or 0)]
    runtime = time_loop_backend(execution.backend, device=device)
    xp = runtime.xp
    # Explicit precision; conversion preserves backend derivative graphs.
    def real(v):
        # Refuse complex coefficients in the engine instead of discarding them.
        a = runtime.array(v)
        dtype = xp.complex128 if xp.isdtype(a.dtype, 'complex floating') else xp.float64
        return as_array(xp, a, dtype=dtype, device=runtime.device)
    def vector(v, factor=1.):
        return None if v is None else as_array(xp, v, dtype=xp.complex128, device=runtime.device)*factor
    u = problem.units
    medium = problem.medium
    record, spectrum = problem.outputs.record, problem.outputs.spectrum
    dt = float(numerics.time_step_seconds)/u.time_seconds
    if not np.isfinite(dt) or dt <= 0:
        raise ValidationError('time step must be finite and positive', path='numerics.time_step_seconds')
    tail_steps = None
    if numerics.tail is not None:
        window = numerics.tail.window_seconds
        if not np.isfinite(window) or window <= 0:
            raise ValidationError('window must be finite and positive', path='numerics.tail.window_seconds')
        tail_steps = int(np.ceil(window/numerics.time_step_seconds))
    region = None
    if record is not None and record.bounds is not None:
        if len(record.bounds) != 3:
            raise ValidationError('bounds require (z,y,x) selectors', path='outputs.record.bounds')
        region = tuple(slice(*v) if isinstance(v, (tuple, list)) else v for v in record.bounds)
    absorber = numerics.absorber
    options = dict(spacing=numerics.grid.spacing, z_spacing=numerics.grid.z_spacing,
        dt=dt, steps=numerics.steps, mu=real(medium.mu),
        conductivity=real(medium.conductivity)*u.length_metres/(EPSILON0*C0),
        poles=tuple((real(p.omega_0)*u.time_seconds, real(p.gamma)*u.time_seconds,
                     real(p.strength)*u.time_seconds**2) for p in medium.poles),
        electric=vector(problem.initial.E), magnetic=vector(problem.initial.H, Z0),
        source=None if problem.current is None else vector(problem.current.density, Z0*u.length_metres),
        pulse=None if problem.current is None else vector(problem.current.samples),
        bloch_phase=real(problem.bloch_phase), backend=execution.backend, device=device,
        checkpoint_steps=numerics.checkpoint_steps,
        wavelengths=() if spectrum is None else spectrum.wavelengths,
        monitor_z=() if spectrum is None else spectrum.planes,
        record_field=None if record is None else record.every,
        record_components=None if record is None else record.components,
        record_region=region, dft_subsample=numerics.dft_subsample,
        convergence_window=tail_steps,
        convergence_tol=1e-4 if numerics.tail is None else numerics.tail.tolerance)
    if isinstance(absorber, PML):
        options.update(pml_cells=absorber.cells, pml_wavelength=absorber.wavelength,
                       pml_background=absorber.media, pml_reflection=absorber.reflection)
    elif isinstance(absorber, ConductiveAbsorber):
        options['absorber_cells'] = absorber.cells
    return real(medium.epsilon), options


def _signature(problem, numerics):
    if not isinstance(problem, Problem) or not isinstance(numerics, Numerics):
        raise ValidationError('expected native.Problem and native.Numerics', path='problem')
    from ._api import _static
    from wavebit.rcwa._adapt import structure
    # Numerical grids/output locations are fixed; all field/material/source values
    # and Bloch phases remain live, subject to the engine's numerical validity guard.
    return (_static(numerics, 'numerics'), problem.units.length,
            structure(problem.medium), structure(problem.current), structure(problem.initial),
            structure(problem.bloch_phase), _static(problem.outputs, 'outputs'))


def _validated(problem, numerics, execution):
    from ._periodic.grid import fdtd_grid
    with execution.scope():
        epsilon, options = _arguments(problem, numerics, execution)
        admission = fdtd_grid(epsilon, **options, _validate_only=True)
    return CheckReport(resolved={'shape': admission['shape'],
        'cfl_time_step_seconds': admission['cfl_dt']*problem.units.time_seconds,
        'time_step_seconds': numerics.time_step_seconds, 'steps': numerics.steps},
        estimates={'principal_array_bytes': admission['state_bytes'],
                   'scope': 'E, H and one Fourier accumulator; excludes ADE/CPML, history and temporaries'},
        capabilities={'backend': execution.backend, 'device': execution.device,
                      'precision': execution.precision, 'automatic_stopping': False})


@eager_under_torch_compile
def check(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    """Run the same admission checks as execution, without advancing fields."""
    try:
        execution = _execution(backend, device, precision)
        _signature(problem, numerics)
        return _validated(problem, numerics, execution)
    except (TypeError, ValueError, NotImplementedError) as exc:
        issue = exc if isinstance(exc, ValidationError) else ValidationError(str(exc), path='problem')
        return CheckReport(issues=(issue.issue,))


@spec
class Prepared:
    numerics: Numerics
    execution: Execution
    report: CheckReport
    _signature: object

    def validate(self, problem):
        """Check concrete reuse and input admission without advancing fields."""
        try:
            if _signature(problem, self.numerics) != self._signature:
                raise IncompatibleProblemError('native layout changed; prepare this problem again', path='problem')
            return _validated(problem, self.numerics, self.execution)
        except (TypeError, ValueError, NotImplementedError) as exc:
            issue = exc if isinstance(exc, ValidationError) else ValidationError(str(exc), path='problem')
            return CheckReport(issues=(issue.issue,))

    def run(self, problem):
        if _signature(problem, self.numerics) != self._signature:
            raise IncompatibleProblemError('native layout changed; prepare this problem again', path='problem')
        return _run(problem, self.numerics, self.execution)


@eager_under_torch_compile
def prepare(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    execution = _execution(backend, device, precision)
    report = _validated(problem, numerics, execution)
    return Prepared(numerics=numerics, execution=execution, report=report,
                    _signature=_signature(problem, numerics))


@spec
class Result:
    """SI native fields and raw Fourier integrals, never incident-normalized phasors."""
    E: object
    H: object
    polarization: tuple
    previous_polarization: tuple
    fields: object
    spectrum: object
    spectral_power_z: object
    validity: object
    diagnostics: object
    provenance: object

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('fields', 'diagnostics', 'provenance'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))


def _adapt(raw, problem, numerics):
    from wavebit._common.backend import array_namespace, _device_of
    u = problem.units
    fields = {}
    for component, data in raw['fields'].items():
        magnetic = component.startswith('h')
        fields[component] = TimeSeries(values=data['values']/(Z0 if magnetic else 1.),
            coordinates={a: data[a] for a in ('x', 'y', 'z')}, components=(component,),
            units={'field': 'A/m' if magnetic else 'V/m', 'coordinates': u.length, 'time': 's'},
            times=data['t']*u.time_seconds, sampling='native Yee stagger',
            metadata={'axis_order': ('time', 'z', 'y', 'x')})
    xp = array_namespace(raw['E'])
    spectrum = None
    if problem.outputs.spectrum is not None:
        e = xp.moveaxis(raw['electric_spectrum'], 2, -1)*u.time_seconds
        h = xp.moveaxis(raw['magnetic_spectrum'], 2, -1)*u.time_seconds/Z0
        spectrum = SpectralFieldSamples(values=xp.concat((e, h), axis=-1),
            coordinates={'z': raw['z_nodes'][raw['monitor_z']],
                         'y': xp.arange(e.shape[2], dtype=e.real.dtype,
                                        device=_device_of(e))*numerics.grid.spacing[1],
                         'x': xp.arange(e.shape[3], dtype=e.real.dtype,
                                        device=_device_of(e))*numerics.grid.spacing[2]},
            components=('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'),
            units={'E': 'V s/m', 'H': 'A s/m', 'coordinates': u.length},
            wavelengths=raw['wavelengths'], frequencies=C0/(raw['wavelengths']*u.length_metres),
            sampling='spatially collocated; E/H retain their native Fourier timestamps',
            metadata={'axis_order': ('wavelength', 'plane', 'y', 'x', 'component'),
                      'normalization': 'raw Fourier integral; no incident normalization'})
    tail = raw['convergence']
    if tail is not None:
        tail = dict(tail)
        tail['electric_spectrum_before_window'] *= u.time_seconds
        tail['magnetic_spectrum_before_window'] *= u.time_seconds/Z0
        # The engine's activity is a weighted diagnostic norm, not stored energy.
        tail['material_activity_units'] = 'internal relative norm; not physical energy'
    diagnostics = dict(time_seconds=raw['time']*u.time_seconds,
        magnetic_time_seconds=raw['h_time']*u.time_seconds,
        cfl_time_step_seconds=raw['cfl_dt']*u.time_seconds,
        z_nodes=raw['z_nodes'], z_spacing=raw['z_spacing'], spacing=raw['spacing'],
        bloch_phase=raw['bloch_phase'], dft_subsample=raw['dft_subsample'],
        dft_samples=raw['dft_samples'], tail=tail)
    return Result(E=raw['E'], H=raw['H']/Z0,
        polarization=tuple(p*EPSILON0 for p in raw['polarization']),
        previous_polarization=tuple(p*EPSILON0 for p in raw['previous_polarization']),
        fields=fields, spectrum=spectrum,
        spectral_power_z=raw['power_z']*u.time_seconds**2*u.length_metres**2/Z0,
        validity=Validity(checks={'stable':raw['stable'],
            'finite_fields':xp.all(xp.isfinite(raw['E'])) & xp.all(xp.isfinite(raw['H']))},
            convergence={} if tail is None else tail), diagnostics=diagnostics,
        provenance={'length_unit': u.length, 'E_unit': 'V/m', 'H_unit': 'A/m',
                    'polarization_unit': 'C/m²', 'spectral_power_unit': 'W s²',
                    'axis_order': ('component', 'z', 'y', 'x'),
                    'sampling': 'native Yee stagger; E at t, H at t-dt/2'})


def _run(problem, numerics, execution):
    from ._periodic.grid import fdtd_grid
    with execution.scope():
        epsilon, options = _arguments(problem, numerics, execution)
        return _adapt(fdtd_grid(epsilon, **options), problem, numerics)


@eager_under_torch_compile
def run(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    return _run(problem, numerics, _execution(backend, device, precision))


__all__ = ['Oscillator', 'Medium', 'Grid', 'Current', 'Initial', 'Record', 'Spectrum',
           'Outputs', 'PML', 'Tail', 'Numerics', 'Problem', 'Prepared', 'Result',
           'check', 'prepare', 'run']
