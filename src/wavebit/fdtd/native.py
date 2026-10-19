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
from ._public_names import removed as _removed
from ._yee_outputs import (COMPONENTS, decay_check_fields, engine_components, engine_message,
                           snapshot_components, snapshot_every)


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
class InitialField:
    """The initial field: E in V/m at t=0 and H in A/m at t=-dt/2, each shaped (3,z,y,x)."""
    E: object = None
    H: object = None


@spec(renamed={'bounds': 'slices'})
class Snapshots:
    """Field snapshots every ``every`` steps, of ``components`` ('Ex'..'Hz').

    ``slices`` optionally picks a (z,y,x) part of the grid: per axis an integer index or a
    ``(start, stop[, step])`` tuple of integer indices. It preserves singleton axes and does not
    crop the solve. Recorded time is always SI seconds; ``Result.snapshots`` is keyed by component.
    Example: ``native.Snapshots(every=10, components=('Ex',), slices=(slice(None), 0, 0))``.
    """
    every: int = 1
    components: tuple = COMPONENTS
    slices: object = None

    def __post_init__(self):
        object.__setattr__(self, 'every', snapshot_every(self.every, 'outputs.snapshots.every'))
        object.__setattr__(self, 'components', snapshot_components(
            self.components, 'outputs.snapshots.components'))
        if self.slices is not None:
            if isinstance(self.slices, (str, bytes)) or not hasattr(self.slices, '__len__') \
                    or len(self.slices) != 3:
                raise ValidationError('slices needs three (z,y,x) selectors: an integer index or '
                                      'a (start, stop[, step]) tuple per axis',
                                      path='outputs.snapshots.slices')


@spec(renamed={'planes': 'z_indices'})
class Spectrum:
    """Streaming Fourier integrals on collocated planes at integer z node indices."""
    wavelengths: object
    z_indices: tuple


@spec(renamed={'record': 'snapshots'})
class Outputs:
    spectrum: object = None
    snapshots: object = None


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
class DecayCheck:
    """Diagnose whether the fields have decayed over the run's final ``window_seconds``.

    A diagnostic, not a stopping rule: the run always takes its ``steps``, and the result's
    ``validity.convergence`` reports the final-window change against ``threshold``. CPML memory
    is not assessed. Example: ``native.Numerics(..., decay_check=native.DecayCheck(window_seconds=2e-15))``.
    """
    threshold: float = 1e-4
    window_seconds: float

    def __post_init__(self):
        threshold, window = decay_check_fields(self.threshold, self.window_seconds,
                                               'numerics.decay_check')
        object.__setattr__(self, 'threshold', threshold)
        object.__setattr__(self, 'window_seconds', window)


@spec(renamed={'tail': 'decay_check'})
class Numerics:
    grid: Grid
    time_step_seconds: float
    steps: int
    absorber: object = None
    checkpoint_steps: int = 32
    dft_subsample: int = 1
    decay_check: object = None

    def __post_init__(self):
        if not isinstance(self.grid, Grid):
            raise ValidationError('expected native.Grid', path='numerics.grid')
        if self.absorber is not None and not isinstance(self.absorber, (PML, ConductiveAbsorber)):
            raise ValidationError('expected native.PML or fdtd.ConductiveAbsorber', path='numerics.absorber')
        if self.decay_check is not None and not isinstance(self.decay_check, DecayCheck):
            raise ValidationError('expected native.DecayCheck', path='numerics.decay_check')


@spec
class Problem:
    medium: Medium
    units: object
    current: object = None
    initial: InitialField = field(default_factory=InitialField)
    bloch_phase: object = (0., 0., 0.)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        require_units(self.units)
        for name, typ in (('medium', Medium), ('initial', InitialField), ('outputs', Outputs)):
            if not isinstance(getattr(self, name), typ):
                raise ValidationError(f'expected native.{typ.__name__}', path=name)
        if self.current is not None and not isinstance(self.current, Current):
            raise ValidationError('expected native.Current', path='current')
        for name, typ in (('spectrum', Spectrum), ('snapshots', Snapshots)):
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
    snapshots, spectrum = problem.outputs.snapshots, problem.outputs.spectrum
    dt = float(numerics.time_step_seconds)/u.time_seconds
    if not np.isfinite(dt) or dt <= 0:
        raise ValidationError('time step must be finite and positive', path='numerics.time_step_seconds')
    window_steps = None
    decay = numerics.decay_check
    if decay is not None:
        from ._periodic.grid import window_steps as _window_steps
        window_steps = _window_steps(decay.window_seconds, numerics.time_step_seconds)
        if not window_steps < numerics.steps:
            raise ValidationError(
                f'DecayCheck(window_seconds={decay.window_seconds!r}) covers {window_steps} '
                f'steps; it must be shorter than the run, Numerics(steps={numerics.steps!r})',
                path='numerics.decay_check.window_seconds')
    region = None
    if snapshots is not None and snapshots.slices is not None:
        region = tuple(slice(*v) if isinstance(v, (tuple, list)) else v for v in snapshots.slices)
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
        monitor_z=() if spectrum is None else spectrum.z_indices,
        record_field=None if snapshots is None else snapshots.every,
        record_components=None if snapshots is None else engine_components(snapshots.components),
        record_region=region, dft_subsample=numerics.dft_subsample,
        convergence_window=window_steps,
        convergence_tol=1e-4 if decay is None else decay.threshold)
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


#: Engine keywords a native refusal can name, and the public spelling that set them.
_ENGINE_SPELLING = (('record_region', 'Snapshots(slices=...)'),
                    ('monitor_z', 'Spectrum(z_indices=...)'))


def _grid(epsilon, options, **extra):
    """The engine call, with its refusals spelled as the public declarations that caused them."""
    from ._periodic.grid import fdtd_grid
    try:
        return fdtd_grid(epsilon, **options, **extra)
    except ValueError as exc:
        public = engine_message(exc, _ENGINE_SPELLING)
        if public is exc:
            raise
        raise public from exc


def _validated(problem, numerics, execution):
    with execution.scope():
        epsilon, options = _arguments(problem, numerics, execution)
        admission = _grid(epsilon, options, _validate_only=True)
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


#: Result attributes that were renamed, and their replacements.
_RENAMED_ATTRIBUTES = {'fields': 'snapshots', 'polarization': 'polarization_density',
                       'previous_polarization': 'previous_polarization_density'}


@spec
class Result:
    """SI native fields and raw Fourier integrals, never incident-normalized phasors.

    ``snapshots`` maps each recorded component ('Ex'..'Hz') to its ``TimeSeries``;
    ``polarization_density`` is the material polarization density P in C/m² per pole.
    """
    E: object
    H: object
    polarization_density: tuple
    previous_polarization_density: tuple
    snapshots: object
    spectrum: object
    spectral_power_z: object
    validity: object
    diagnostics: object
    provenance: object

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('snapshots', 'diagnostics', 'provenance'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    def __getattr__(self, name):
        # Reached only when normal lookup fails, so it never shadows a field.
        if name in _RENAMED_ATTRIBUTES:
            raise _removed('native.Result', name, f'native.Result.{_RENAMED_ATTRIBUTES[name]}')
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')


def _adapt(raw, problem, numerics):
    from wavebit._common.backend import array_namespace, _device_of
    u = problem.units
    snapshots = {}
    for component, data in raw['fields'].items():
        # The engine keys its records 'ex'..'hz'; the public key is 'Ex'..'Hz'.
        magnetic = component.startswith('h')
        public = component.capitalize()
        snapshots[public] = TimeSeries(values=data['values']/(Z0 if magnetic else 1.),
            coordinates={a: data[a] for a in ('x', 'y', 'z')}, components=(public,),
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
        polarization_density=tuple(p*EPSILON0 for p in raw['polarization']),
        previous_polarization_density=tuple(p*EPSILON0 for p in raw['previous_polarization']),
        snapshots=snapshots, spectrum=spectrum,
        spectral_power_z=raw['power_z']*u.time_seconds**2*u.length_metres**2/Z0,
        validity=Validity(checks={'stable':raw['stable'],
            'finite_fields':xp.all(xp.isfinite(raw['E'])) & xp.all(xp.isfinite(raw['H']))},
            convergence={} if tail is None else tail), diagnostics=diagnostics,
        provenance={'length_unit': u.length, 'E_unit': 'V/m', 'H_unit': 'A/m',
                    'polarization_unit': 'C/m²', 'spectral_power_unit': 'W s²',
                    'axis_order': ('component', 'z', 'y', 'x'),
                    'sampling': 'native Yee stagger; E at t, H at t-dt/2'})


def _run(problem, numerics, execution):
    with execution.scope():
        epsilon, options = _arguments(problem, numerics, execution)
        return _adapt(_grid(epsilon, options), problem, numerics)


@eager_under_torch_compile
def run(problem, *, numerics, backend='numpy', device='cpu', precision='double'):
    return _run(problem, numerics, _execution(backend, device, precision))


__all__ = ['Oscillator', 'Medium', 'Grid', 'Current', 'InitialField', 'Snapshots', 'Spectrum',
           'Outputs', 'PML', 'DecayCheck', 'Numerics', 'Problem', 'Prepared', 'Result',
           'check', 'prepare', 'run']

#: Names this module no longer has: each is a tombstone naming its replacement, not an alias.
_RENAMED = {'Initial': ('InitialField', None), 'Record': ('Snapshots', None),
            'Tail': ('DecayCheck', 'given as Numerics(decay_check=...), with tolerance= now '
                                   'threshold=')}


def __getattr__(name):
    if name in _RENAMED:
        new, note = _RENAMED[name]
        raise _removed('wavebit.fdtd.native', name, f'wavebit.fdtd.native.{new}', note)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
