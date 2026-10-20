"""Physical periodic-stack FDTD with explicit numerical route selection.

The Stack, Layer, typed shapes and plane-wave excitation are shared with RCWA.
Geometry is in declared length units; all times and material rates are SI.
``algorithm='shared'`` supports NumPy/JAX/Torch; ``'reference'`` selects the
independent NumPy stepping formulation, useful for historical reproduction.
"""
from dataclasses import field
from collections.abc import Mapping
import numpy as np

from wavebit._common.backend import eager_under_torch_compile, array_namespace, _concrete
from wavebit._common.declarations import order_pair, renamed_keywords, spec
from wavebit._common.execution import Execution, CheckReport, frozen_mapping
from wavebit._common.pytrees import register_dataclass
from wavebit.errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from wavebit.materials import Material, Drude, Lorentz
from wavebit.observations import DiffractionOrders, FieldSamples, TimeSeries, Validity
from wavebit.observations.fields import _component_units
from wavebit.rcwa import Stack, Layer, Region, Rectangle, Circle, Ellipse, Polygon, Excitation, PlaneWave
from wavebit.units import require_units, Z0
from ._public_names import removed as _removed
from ._yee_outputs import (COMPONENTS, decay_check_fields, engine_components, public_refusal,
                           refusals, snapshot_components, snapshot_every)


@spec(renamed={'models': 'poles'})
class RasterDispersion:
    """Integer raster labels mapped to physical Drude/Lorentz poles.

    ``poles`` maps each nonzero label to a tuple of the same ``materials.Drude`` /
    ``materials.Lorentz`` objects ``DispersiveMaterial(poles=)`` takes (Debye is refused).
    Label zero is nondispersive. Stack raster epsilon is epsilon-infinity.
    Coefficients are SI rates, independently of the geometry's length unit.
    """
    labels: object
    poles: object

    def __post_init__(self):
        poles = {k: tuple(v) for k, v in self.poles.items()}
        if any(type(k) is not int or k == 0 for k in poles):
            raise ValidationError('poles require nonzero integer labels', code='dispersion.poles', path='dispersion.poles')
        if any(not isinstance(p, (Drude, Lorentz)) for v in poles.values() for p in v):
            raise ValidationError('use physical Drude/Lorentz poles', code='dispersion.poles', path='dispersion.poles')
        object.__setattr__(self, 'poles', frozen_mapping(poles))


#: sigma_f of the Gaussian envelope exp(-((t-t0)/w)**2) is 1/(sqrt(2)*pi*w).
_SIGMA_NOTE = ('frequency_sigma_hz = 1/(sqrt(2)*pi*width_seconds), the spectral standard deviation '
               'of the same Gaussian, as in fdtd.GaussianPulse(frequency_sigma_hz=...)')


@spec(renamed={'width_seconds': ('frequency_sigma_hz', _SIGMA_NOTE)})
class Pulse:
    """Gaussian source envelope, by its spectral standard deviation, with optional causal notches.

    ``frequency_sigma_hz`` is the standard deviation of the amplitude spectrum
    ``exp(-(f-f0)**2/(2*sigma**2))`` in Hz, as in ``fdtd.GaussianPulse(frequency_sigma_hz=)``.
    ``None`` (the default) uses the envelope whose 1/e time half-width is one period at the
    centre wavelength, so sigma = f0/(sqrt(2)*pi) with f0 the centre frequency.
    ``notch_frequencies_hz`` suppresses chosen positive frequencies with causal delayed copies.
    Example: ``periodic.Pulse(frequency_sigma_hz=3e13)``.
    """
    frequency_sigma_hz: object = None
    notch_frequencies_hz: object = None

    def __post_init__(self):
        sigma = self.frequency_sigma_hz
        if sigma is not None:
            if (isinstance(sigma, (bool, np.bool_)) or np.ndim(sigma) or np.iscomplexobj(sigma)
                    or not np.isfinite(sigma) or sigma <= 0):
                raise ValidationError(f'frequency_sigma_hz={sigma!r}: a finite positive static '
                                      f'scalar in Hz is needed', code='pulse.frequency_sigma_hz', path='pulse.frequency_sigma_hz')


@spec
class Snapshots:
    """Field snapshots every ``every`` steps, of ``components`` ('Ex'..'Hz').

    ``bounds`` optionally limits them to a physical box: a mapping from 'x', 'y' or 'z' to a
    ``(low, high)`` pair in the problem's length unit. ``Result.snapshots`` is keyed by component.
    Example: ``periodic.Snapshots(every=5, components=('Ex', 'Hy'), bounds={'z': (0., 0.2)})``.
    """
    every: int = 1
    components: tuple = COMPONENTS
    bounds: object = None

    def __post_init__(self):
        object.__setattr__(self, 'every', snapshot_every(self.every, 'outputs.snapshots.every'))
        object.__setattr__(self, 'components', snapshot_components(
            self.components, 'outputs.snapshots.components'))
        if self.bounds is not None:
            object.__setattr__(self, 'bounds', frozen_mapping(self.bounds))


@spec(renamed={'record': 'snapshots'})
class Outputs:
    planes: object = field(default_factory=dict)
    snapshots: object = None

    def __post_init__(self):
        object.__setattr__(self, 'planes', frozen_mapping(self.planes))
        if self.snapshots is not None and not isinstance(self.snapshots, Snapshots):
            raise ValidationError('expected periodic.Snapshots', code='problem.type', path='outputs.snapshots')


@spec
class DecayCheck:
    """Diagnose whether the fields have decayed over the run's final ``window_seconds``.

    A diagnostic, not a stopping rule: the run always lasts ``Numerics(duration_seconds=)``, and
    the result's ``validity.convergence['decay']`` reports the final-window change against
    ``threshold``.
    The window is rounded up to whole time steps. Example:
    ``periodic.Numerics(..., decay_check=periodic.DecayCheck(window_seconds=20e-15))``.
    """
    threshold: float = 1e-4
    window_seconds: float

    def __post_init__(self):
        threshold, window = decay_check_fields(self.threshold, self.window_seconds,
                                               'numerics.decay_check')
        object.__setattr__(self, 'threshold', threshold)
        object.__setattr__(self, 'window_seconds', window)


_DECAY_NOTE = ('decay_check=DecayCheck(window_seconds=..., threshold=...) takes the window in '
               'seconds, not steps (window_seconds = tail_steps*time_step_seconds, with the time '
               "step from check(...).resolved['time_step_seconds']), and threshold= is the old "
               'tail_tolerance=')


@spec(renamed={'tail_steps': ('decay_check', _DECAY_NOTE),
               'tail_tolerance': ('decay_check', _DECAY_NOTE)})
class Numerics:
    """Fixed numerical choices; spatial quantities use the problem's length unit.

    ``spacing`` is nominal z spacing; ``transverse_spacing`` optionally chooses
    x/y separately. Duration is an SI hard cap rounded up to whole time steps.
    ``decay_check`` only diagnoses the final window; it never stops a run early.
    Reference stepping admits only its explicitly supported subset and refuses
    shared-only controls, rather than silently ignoring them.
    ``orders`` is the diffraction-order window ``(Mu, Mv)``: an explicit pair of nonnegative
    integers, the half-widths along the two lattice vectors (``(M, 0)`` for a one-dimensional
    period), or ``None`` for the engine's default window.
    """
    spacing: float
    duration_seconds: float
    algorithm: str = 'shared'
    courant: float = .8
    pml_cells: int = 12
    pml_reflection: float = 1e-8
    subpixel: bool = True
    orders: object = None
    padding_scale: float = 1.
    grading: object = None
    transverse_spacing: object = None
    nominal_indices: object = None
    nominal_layer_thickness: object = None
    align_layers: bool = False
    variable_depth: bool = False
    layer_smoothing: float = 0.
    shape_smoothing: object = 0.
    shape_integration: str = 'separable'
    extraction_distance: object = None
    extraction_method: str = 'electric'
    source_distance: object = None
    checkpoint_steps: int = 32
    dft_subsample: int = 1
    decay_check: object = None

    def __post_init__(self):
        if self.algorithm not in ('shared', 'reference'):
            raise ValidationError("algorithm must be 'shared' or 'reference'", code='numerics.choice', path='numerics.algorithm')
        for name in ('spacing', 'duration_seconds'):
            v=getattr(self,name)
            if isinstance(v,bool) or np.ndim(v) or not np.isfinite(v) or v<=0:
                raise ValidationError('expected a finite positive static scalar', code='numerics.'+name, path='numerics.'+name)
        if self.decay_check is not None and not isinstance(self.decay_check, DecayCheck):
            raise ValidationError('expected periodic.DecayCheck', code='numerics.type', path='numerics.decay_check')
        if self.orders is not None:
            object.__setattr__(self, 'orders', order_pair('Numerics', self.orders,
                                                          code='numerics.orders',
                                                          path='numerics.orders'))


@spec
class Problem:
    stack: Stack
    excitation: Excitation
    units: object
    dispersion: object = None
    pulse: Pulse = field(default_factory=Pulse)
    outputs: Outputs = field(default_factory=Outputs)

    def __post_init__(self):
        require_units(self.units)
        for name, typ in (('stack',Stack),('excitation',Excitation),('pulse',Pulse),('outputs',Outputs)):
            if not isinstance(getattr(self,name),typ):
                raise ValidationError(f'expected periodic.{typ.__name__}',code='problem.type',path=name)
        if len(self.excitation.sources)!=1 or not isinstance(self.excitation.sources[0],PlaneWave):
            raise UnsupportedFeatureError('periodic FDTD requires exactly one PlaneWave',code='excitation.sources',path='excitation.sources')
        if self.dispersion is not None and not isinstance(self.dispersion,RasterDispersion):
            raise ValidationError('expected periodic.RasterDispersion',code='problem.type',path='dispersion')


def _execution(numerics,backend,device,precision):
    if not isinstance(numerics,Numerics):
        raise ValidationError('expected periodic.Numerics',code='numerics.type',path='numerics')
    execution=Execution(backend=backend,device=device,precision=precision).validate()
    if precision!='double':
        raise UnsupportedFeatureError('periodic execution requires double precision',code='execution.precision',path='execution.precision')
    if numerics.algorithm=='reference' and backend!='numpy':
        raise UnsupportedFeatureError('reference stepping supports NumPy only',code='execution.backend',path='execution.backend')
    return execution


def _arguments(problem,numerics,execution):
    from wavebit.rcwa._adapt import _namespace, _array, geometry_recipe, polarization
    if not isinstance(problem,Problem) or not isinstance(numerics,Numerics):
        raise ValidationError('expected periodic.Problem and periodic.Numerics',code='problem.type',path='problem')
    xp=_namespace(execution)
    source=problem.excitation.sources[0]
    if source.kx is not None or source.ky is not None:
        raise UnsupportedFeatureError('use real theta/phi for periodic FDTD',code='source.direction',path='excitation.sources')
    amplitude=_array(xp,source.amplitude,execution,True)
    if amplitude.shape or _concrete(xp.isfinite(amplitude)) is False:
        raise ValidationError('amplitude must be a finite scalar in V/m',code='source.amplitude',path='excitation.sources.amplitude')
    pair=polarization(source,xp,execution)
    if _concrete(xp.all(xp.isfinite(pair))) is False:
        raise ValidationError('polarization must be a finite nonzero Jones pair',code='source.polarization_value',path='excitation.sources.polarization')
    def material(value):
        if not isinstance(value,Material):
            raise UnsupportedFeatureError('periodic regions and ports require nondispersive Material; use RasterDispersion for electric poles',code='material.type',path='stack.material')
        return value
    stack=problem.stack
    if not stack.layers:
        raise ValidationError('periodic FDTD requires at least one finite layer',code='stack.layers',path='stack.layers')
    sampled=[layer.epsilon is not None for layer in stack.layers]
    if any(sampled) and not all(sampled):
        raise UnsupportedFeatureError('use either sampled layers or typed regions throughout a stack',code='stack.mixed_layer_encoding',path='stack.layers')
    if all(sampled):
        structure=xp.stack([_array(xp,l.epsilon,execution,True) for l in stack.layers])
        mu=xp.stack([xp.ones_like(structure[0]) if l.mu is None else _array(xp,l.mu,execution,True) for l in stack.layers])
    else:
        structure=[dict(background=material(l.background),shapes=[dict(geometry_recipe(r.geometry),material=material(r.material)) for r in l.regions]) for l in stack.layers]
        mu=None
    # Concrete thickness without nominal geometry is intentionally static. A
    # nominal mesh keeps live thickness leaves on their backend without detach.
    thickness=([l.thickness for l in stack.layers] if numerics.nominal_layer_thickness is None
               else xp.stack([_array(xp,l.thickness,execution) for l in stack.layers]))
    band=np.atleast_1d(np.asarray(problem.excitation.wavelength,float))
    if band.ndim!=1 or not len(band) or not np.isfinite(band).all() or np.any(band<=0):
        raise ValidationError('wavelength must be a positive finite scalar or vector',code='wavelength.value',path='excitation.wavelength')
    ts=problem.units.time_unit_seconds
    centre=float((band.min()+band.max())/2)
    snapshots=problem.outputs.snapshots
    options=dict(structure=structure,thickness=thickness,top=material(stack.top),bot=material(stack.bottom),
        period=stack.period,lattice=stack.lattice,wavelength=band,mu=mu,
        theta=0. if source.theta is None else source.theta,phi=source.phi,
        incidence_side=source.side,polarization=pair,
        resolution=1/numerics.spacing,n_periods=numerics.duration_seconds/(ts*centre),
        n_pml=numerics.pml_cells,pml_reflection=numerics.pml_reflection,
        courant=numerics.courant,subpixel=numerics.subpixel,truncation_order=numerics.orders,
        pad_scale=numerics.padding_scale,grading=numerics.grading,return_diagnostics=True,
        field_monitors=dict(problem.outputs.planes) or None,
        record_field=None if snapshots is None else snapshots.every,
        record_components=None if snapshots is None else engine_components(snapshots.components))
    if problem.dispersion is not None:
        poles={}
        for key,terms in problem.dispersion.poles.items():
            poles[key]=tuple((0.,p.gamma*ts,(p.omega_p*ts)**2) if isinstance(p,Drude)
                else (p.omega_0*ts,p.gamma*ts,p.strength*ts**2) for p in terms)
        options.update(dispersion=problem.dispersion.labels,poles=poles)
    if numerics.algorithm=='reference':
        defaults=Numerics(spacing=numerics.spacing,duration_seconds=numerics.duration_seconds,algorithm='reference')
        for name in ('transverse_spacing','nominal_indices','nominal_layer_thickness','align_layers',
                     'variable_depth','layer_smoothing','shape_smoothing','shape_integration',
                     'extraction_distance','extraction_method','source_distance','checkpoint_steps',
                     'dft_subsample','decay_check'):
            value=getattr(numerics,name)
            if isinstance(value,(np.ndarray,Mapping)) or value!=getattr(defaults,name):
                raise UnsupportedFeatureError('this control requires shared stepping',code='numerics.reference_scope',path='numerics.'+name)
        if problem.pulse.frequency_sigma_hz is not None or problem.pulse.notch_frequencies_hz is not None:
            raise UnsupportedFeatureError('custom pulse controls require shared stepping',code='numerics.reference_scope',path='pulse')
        if snapshots is not None and snapshots.bounds is not None:
            raise UnsupportedFeatureError('snapshot bounds require shared stepping',code='numerics.reference_scope',path='outputs.snapshots.bounds')
        return options,None
    device=execution.device
    if execution.backend=='jax':
        import jax
        platform,_,index=device.partition(':');device=jax.devices(platform)[int(index or 0)]
    transverse=None
    if numerics.transverse_spacing is not None:
        spacing=np.asarray(numerics.transverse_spacing)
        if spacing.shape not in ((),(2,)) or np.iscomplexobj(spacing) or not np.isfinite(spacing).all() or np.any(spacing<=0):
            raise ValidationError('expected positive x/y spacing',code='numerics.transverse_spacing',path='numerics.transverse_spacing')
        transverse=1/spacing
    notches=problem.pulse.notch_frequencies_hz
    sigma=problem.pulse.frequency_sigma_hz
    decay=numerics.decay_check
    options.update(device=device,transverse_resolution=transverse,mesh_indices=numerics.nominal_indices,
        mesh_thickness=numerics.nominal_layer_thickness,align_layers=numerics.align_layers,
        variable_depth=numerics.variable_depth,layer_smoothing=numerics.layer_smoothing,
        shape_smoothing=numerics.shape_smoothing,shape_integration=numerics.shape_integration,
        diffraction_distance=numerics.extraction_distance,diffraction_method=numerics.extraction_method,
        source_distance=numerics.source_distance,checkpoint_steps=numerics.checkpoint_steps,
        dft_subsample=numerics.dft_subsample,
        # The window is in seconds; only the engine knows its time step, so it converts.
        convergence_window_time=None if decay is None else decay.window_seconds/ts,
        convergence_tol=1e-4 if decay is None else decay.threshold,
        record_bounds=None if snapshots is None or snapshots.bounds is None else dict(snapshots.bounds),
        # The envelope exp(-((t-t0)/w)**2) has amplitude-spectrum sigma 1/(sqrt(2)*pi*w).
        pulse_width=None if sigma is None else (1/(np.sqrt(2)*np.pi*sigma))/ts,
        pulse_notches=None if notches is None else np.asarray(notches)*ts)
    return options,execution.backend


#: Engine refusals whose advice reverses under the public spelling (a longer pulse_width is a
#: smaller frequency_sigma_hz), reworded whole.
_REWORDED = {
    'pulse_width leaves a requested wavelength in the numerical source tail; reduce pulse_width '
    'or narrow the wavelength band.':
        'The source spectrum is numerically zero at a requested wavelength: raise '
        'Pulse(frequency_sigma_hz=...) (the default is f0/(sqrt(2)*pi) at the centre frequency f0) '
        'or narrow Excitation(wavelength=...).',
    'pulse_width must span at least one time step; increase pulse_width or resolution.':
        'Pulse(frequency_sigma_hz=...) is too broad for the time step: lower frequency_sigma_hz '
        'or decrease Numerics(spacing=...).',
    'convergence_window must be a positive integer smaller than steps.':
        'DecayCheck(window_seconds=...) must be shorter than the run, '
        'Numerics(duration_seconds=...).',
    "Point probes require all six fields; solve with record_components='all'.":
        'Result.sample needs all six Snapshots(components=...), the default.',
}

#: Engine keywords a periodic refusal can name, and the public spelling that set them.
_SPELLING = (('pulse_notches', 'Pulse(notch_frequencies_hz=...)'),
             ('record_bounds', 'Snapshots(bounds=...)'),
             ('record_components', 'Snapshots(components=...)'),
             ('record_field', 'Snapshots(every=...)'),
             ('mesh_indices', 'Numerics(nominal_indices=...)'),
             ('mesh_thickness', 'Numerics(nominal_layer_thickness=...)'),
             ('truncation_order', 'Numerics(orders=...)'),
             ('n_periods', 'Numerics(duration_seconds=...)'))


def _public(exc):
    """``exc`` spelled for a public caller, or ``exc`` itself if it names no engine keyword.

    The engine keeps its own wording, which its own callers and tests read; only this door
    rewords, so a refusal names the declaration the caller wrote.
    """
    import re
    text = str(exc)
    public = _REWORDED.get(text, text)
    if public == text:
        for engine, declared in _SPELLING:
            public = public.replace(engine, declared)
        # A snapshot refusal names an engine component ('for ex;'); the public name is 'Ex'.
        public = re.sub(r"\bfor ([eh][xyz])\b", lambda m: f"for '{m.group(1).capitalize()}'", public)
    if public == text:
        return exc
    return ValidationError(public, code=getattr(exc, 'code', None) or 'periodic.invalid',
                           path=getattr(exc, 'path', '') or 'problem')


def _engine(call, *args, **kwargs):
    """Run an engine call, raising its refusals in the public spelling with a code."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        public = public_refusal(_public(exc), 'periodic')
        if public is exc:
            raise
        raise public from exc


def _signature(problem,numerics):
    if not isinstance(problem,Problem) or not isinstance(numerics,Numerics):
        raise ValidationError('expected periodic.Problem and periodic.Numerics',code='problem.type',path='problem')
    from ._api import _static
    from wavebit.rcwa._adapt import structure
    fixed=dict(units=problem.units.length,numerics=_static(numerics,'numerics'),
        cell=_static((problem.stack.period,problem.stack.lattice),'stack.cell'),
        wavelength=_static(problem.excitation.wavelength,'excitation.wavelength'),
        outputs=_static(problem.outputs,'outputs'))
    if numerics.nominal_layer_thickness is None:
        fixed['thickness']=_static(tuple(l.thickness for l in problem.stack.layers),'stack.layers.thickness')
    if problem.dispersion is not None:
        fixed['labels']=_static(problem.dispersion.labels,'dispersion.labels')
    if numerics.nominal_indices is None:
        fixed['ports']=_static((problem.stack.top,problem.stack.bottom),'stack.ports')
    source=problem.excitation.sources[0]
    phi=source.phi
    static_phi=phi is None or isinstance(phi,(int,float,np.number))
    fixed['direction_layout']=(static_phi, (phi is None or phi==0) if static_phi else None)
    return structure(problem),fixed


#: The declaration each fixed entry of ``_signature`` holds, as a public path; a pair names
#: its two declarations, and ``thickness`` holds one entry per layer.
_FIXED_PATHS={'units':'units.length','numerics':'numerics','cell':('stack.period','stack.lattice'),
              'wavelength':'excitation.wavelength','outputs':'outputs',
              'thickness':'stack.layers[{}].thickness','labels':'dispersion.labels',
              'ports':('stack.top','stack.bottom'),'direction_layout':'excitation.sources[0].phi'}


def _fixed_path(key,new,old):
    path=_FIXED_PATHS[key]
    if isinstance(path,str) and '{}' not in path:
        return path
    if new is None or old is None or len(new)!=len(old):
        return path[0].split('.')[0] if isinstance(path,tuple) else 'stack.layers'
    index=next(i for i,(a,b) in enumerate(zip(new,old)) if a!=b)
    return path[index] if isinstance(path,tuple) else path.format(index)


def _same_layout(problem,numerics,signature):
    """Refuse a problem whose fixed layout differs from a preparation's, naming what changed."""
    from wavebit.rcwa._adapt import structure_change
    layout,fixed=_signature(problem,numerics)
    if (layout,fixed)!=signature:
        old=signature[1]
        if layout!=signature[0]:
            path=structure_change(layout,signature[0]) or 'problem'
        else:
            key=next(k for k in list(fixed)+[k for k in old if k not in fixed]
                     if fixed.get(k)!=old.get(k))
            path=_fixed_path(key,fixed.get(key),old.get(key))
        raise IncompatibleProblemError('periodic layout changed; prepare this problem again',
                                       code='prepared.structure',path=path)


def _validated(problem,numerics,execution):
    from ._periodic.adapter import solve_periodic_fdtd
    with execution._scope():
        options,backend=_arguments(problem,numerics,execution)
        raw=_engine(solve_periodic_fdtd,dict(options,_validate_only=True),backend=backend)
    return CheckReport(resolved={'shape':raw['shape'],'steps':raw['steps'],
        'time_step_seconds':raw['dt']*problem.units.time_unit_seconds},
        capabilities={'algorithm':numerics.algorithm,'backend':execution.backend,
            'precision':execution.precision,'device':execution.device,
            'scope':raw.get('scope','full shared-grid admission without time advancement')})


@eager_under_torch_compile
def check(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    try:
        execution=_execution(numerics,backend,device,precision)
        _signature(problem,numerics)
        return _validated(problem,numerics,execution)
    except (TypeError,ValueError,NotImplementedError) as exc:
        return CheckReport(issues=(public_refusal(exc,'periodic').issue,))


@spec
class Prepared:
    numerics: Numerics
    execution: Execution
    report: CheckReport
    _signature: object

    def validate(self, problem):
        """Check concrete reuse and input admission without advancing fields."""
        try:
            _same_layout(problem, self.numerics, self._signature)
            return _validated(problem, self.numerics, self.execution)
        except (TypeError, ValueError, NotImplementedError) as exc:
            return CheckReport(issues=(public_refusal(exc, 'periodic').issue,))

    def run(self,problem):
        with refusals('periodic'):
            _same_layout(problem,self.numerics,self._signature)
            return _run(problem,self.numerics,self.execution)


@eager_under_torch_compile
def prepare(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    with refusals('periodic'):
        execution=_execution(numerics,backend,device,precision)
        return Prepared(numerics=numerics,execution=execution,report=_validated(problem,numerics,execution),
                        _signature=_signature(problem,numerics))


@spec
class Result:
    """Diffraction orders, plane phasors and optional field snapshots of one periodic run.

    ``snapshots`` maps each recorded component ('Ex'..'Hz') to its ``TimeSeries``.
    ``validity.checks['runtime_stability']`` is ``None`` (not assessed) under
    ``algorithm='reference'``, whose stepping assesses no stability; ``numerical_finite`` covers
    R and T.
    """
    reflection: DiffractionOrders
    transmission: DiffractionOrders
    basis: object
    snapshots: object
    planes: object
    plane_power_z: object
    validity: Validity
    diagnostics: object
    provenance: object
    _engine_diagnostics: object = None

    def __post_init__(self):
        register_dataclass(type(self))
        for name in ('basis','snapshots','planes','plane_power_z','diagnostics','provenance'):
            object.__setattr__(self,name,frozen_mapping(getattr(self,name)))

    def __getattr__(self, name):
        # Reached only when normal lookup fails, so it never shadows a field.
        if name == 'fields':
            raise _removed('periodic.Result', name, 'periodic.Result.snapshots')
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    @property
    def R(self): return self.reflection.efficiency

    @property
    def T(self): return self.transmission.efficiency

    @renamed_keywords('periodic.Result.sample', {'times': 'times_seconds'})
    def sample(self,*,x=0.,y=0.,z,times_seconds=None):
        """Concrete offline interpolation of all six fields at the SI times ``times_seconds``.

        Requires all six Snapshots components (the default). Interpolation across the source
        discontinuity is refused; this NumPy postprocessing does not retain AD.
        """
        from ._periodic.probes import sample_fdtd_fields
        ts=self.provenance['time_unit_seconds']
        raw=_engine(sample_fdtd_fields,self._engine_diagnostics,x=x,y=y,z=z,
                    t=None if times_seconds is None else np.asarray(times_seconds)/ts)
        amplitude=self.provenance['incident_amplitude']
        return TimeSeries(values=np.concatenate((raw['E'],raw['H']/Z0),axis=-1)*amplitude,
            coordinates={'x':x,'y':y,'z':z},components=('Ex','Ey','Ez','Hx','Hy','Hz'),
            units=_component_units(('Ex','Ey','Ez','Hx','Hy','Hz'),self.provenance['length_units']),
            times_seconds=raw['t']*ts,sampling='linear space/time interpolation of Yee records',
            metadata={'region':raw['region'],'axis_order':('time','component')})


def _adapt(raw,diag,problem,numerics,execution):
    raw=raw if isinstance(raw,list) else [raw]
    xp=array_namespace(raw[0].R)
    wavelengths=xp.asarray(np.atleast_1d(problem.excitation.wavelength))
    source=problem.excitation.sources[0];amplitude=source.amplitude;u=problem.units
    depth=sum(l.thickness for l in problem.stack.layers)
    # The periodic cell's measure: its area, or its x period per unit length of an invariant y.
    invariant=('y',) if diag.get('invariant_y') else ()
    period=problem.stack.period[:2-len(invariant)]
    cell_measure=float(np.prod(period))*u.length_unit_meters**(2-len(invariant))
    def orders(which,leg):
        subs=[getattr(r,leg) for r in raw]
        if any(s is None for s in subs): return None
        efficiency=xp.stack([getattr(s,which.upper()) for s in subs])
        nv,nu=efficiency.shape[-2:]
        # Engine windows can be offset; retain their explicit integer labels.
        (cu,hu),(cv,hv)=raw[0].order_window
        z=depth if (source.side=='top')==(which=='r') else 0.
        face=source.side if which=='r' else ('bottom' if source.side=='top' else 'top')
        return DiffractionOrders(wavelengths=wavelengths,
            orders_u=tuple(range(cu-hu,cu+hu+1)),orders_v=tuple(range(cv-hv,cv+hv+1)),
            efficiency=efficiency,amplitude_s=xp.stack([getattr(s,which+'_s') for s in subs])*amplitude,
            amplitude_p=xp.stack([getattr(s,which+'_p') for s in subs])*amplitude,
            reference={'port':face,'wave':'reflected' if which=='r' else 'transmitted',
                'plane':z,'length_units':u.length,'lateral_origin':(0.,0.),'polarization_frame':'per-order s/p',
                'amplitude_units':'V/m','incident_amplitude':amplitude,'incident_amplitude_units':'V/m',
                'cell_measure':cell_measure,'cell_measure_units':'m' if invariant else 'm^2',
                'invariant_axes':invariant},
            reference_wavelength_fields=(),
            metadata={'theta_radians':diag.get('angles'),'normalization':'incident-only power; unit Jones pair'})
    basis={}
    for leg in ('s_inc','p_inc'):
        reflected=orders('r',leg)
        if reflected is not None: basis[leg]=(reflected,orders('t',leg))
    snapshots={}
    for name,data in diag.get('fields',{}).items():
        # The engine keys its records 'ex'..'hz'; the public key is 'Ex'..'Hz'.
        snapshots[name.capitalize()]=TimeSeries(values=data['values']*amplitude/(Z0 if name.startswith('h') else 1.),
            coordinates={a:data[a] for a in ('x','y','z')},components=(name.capitalize(),),
            units=_component_units((name.capitalize(),),u.length),
            times_seconds=data['t']*u.time_unit_seconds,sampling='native Yee stagger',
            metadata={'axis_order':('time','z','y','x'),
                'calibration':diag.get('field_normalization',diag.get('normalization'))})
    planes={};powers={}
    for name,data in diag.get('monitors',{}).items():
        components=('ex','ey','ez','hx','hy','hz')
        values=xp.stack([data['fields'][c]*amplitude/(Z0 if c.startswith('h') else 1.) for c in components],axis=-1)
        planes[name]=FieldSamples(values=values,components=('Ex','Ey','Ez','Hx','Hy','Hz'),
            coordinates=dict({k:data[k] for k in ('x','y','z') if k in data},wavelength=wavelengths),
            units=_component_units(('Ex','Ey','Ez','Hx','Hy','Hz'),u.length),sampling='collocated phasor',
            metadata={'normalization':data.get('normalization'),'axis_order':('wavelength','y','x','component'),
                      'power_units':'W/m' if data['per_unit_y'] else 'W',
                      'cell_measure':data['cell_area']*u.length_unit_meters**(1 if data['per_unit_y'] else 2),
                      'cell_measure_units':'m' if data['per_unit_y'] else 'm^2'})
        powers[name]=data['power_z']*abs(amplitude)**2*u.length_unit_meters**(1 if data['per_unit_y'] else 2)/Z0
    # Only explicitly physical diagnostics cross the public boundary. The private
    # retained snapshot supports existing offline interpolation, without exporting
    # normalized arrays or internal grid objects as user-facing measurements.
    diagnostics={k:diag[k] for k in ('shape','spacing','steps','n_steps','dft_subsample','dft_samples',
        'z_nodes','z_spacing','source_z','pml_bounds','layer_planes','angles','azimuth',
        'diffraction','residual') if k in diag}
    if 'invariant_y' in diag:
        # The axes along which the cell is invariant, by name, as fdtd outputs report them.
        diagnostics['invariant_axes']=('y',) if diag['invariant_y'] else ()
    if 'dt' in diag: diagnostics['time_step_seconds']=diag['dt']*u.time_unit_seconds
    elif 'grid' in diag: diagnostics['time_step_seconds']=diag['grid'].dt*u.time_unit_seconds
    if 'pulse' in diag:
        pulse={}
        for key,value in diag['pulse'].items():
            if key in ('width','peak_time','sampled_peak_time'):
                pulse[key+'_seconds']=value*u.time_unit_seconds
            elif key in ('frequency','notches'):
                pulse[key+'_hz']=value/u.time_unit_seconds
            else: pulse[key]=value
        diagnostics['pulse']=pulse
    def tail(value):
        if isinstance(value,(tuple,list)): return tuple(tail(v) for v in value)
        if not isinstance(value,Mapping): return value
        result={}
        for key,v in value.items():
            if key=='comparison_time': result['comparison_time_seconds']=v*u.time_unit_seconds
            # The engine's tolerance is what the public DecayCheck(threshold=) set.
            elif key=='tolerance': result['threshold']=v
            elif key=='electric_spectrum_before_window': result[key]=v*u.time_unit_seconds
            elif key=='magnetic_spectrum_before_window': result[key]=v*u.time_unit_seconds/Z0
            else: result[key]=tail(v)
        return result
    decay=tail(diag.get('convergence',{})) or None
    if decay is not None:
        decay.update(amplitude_change_units='relative incident amplitude',
            scattering_change_bound_units='relative incident amplitude',
            electric_spectrum_before_window_units='V*s/m, uncalibrated current basis',
            magnetic_spectrum_before_window_units='A*s/m, uncalibrated current basis',
            material_activity_units='internal relative norm; not physical energy')
    diagnostics['decay']=decay
    # Reference stepping assesses no stability: None (not assessed), never a defaulted True.
    checks={'runtime_stability':diag.get('stable') if numerics.algorithm=='shared' else None,
            'numerical_finite':xp.all(xp.isfinite(xp.stack([r.R for r in raw])))
                               & xp.all(xp.isfinite(xp.stack([r.T for r in raw])))}
    return Result(reflection=orders('r','incident'),transmission=orders('t','incident'),basis=basis,
        snapshots=snapshots,planes=planes,plane_power_z=powers,
        validity=Validity(checks=checks,convergence={'decay':decay,'residual':diag.get('residual')}),
        diagnostics=diagnostics,provenance={'solver':'fdtd.periodic','execution':execution,
            'numerics':numerics,'units':u,'coordinate_frame':'+z up; bottom interface z=0; layers '
            'bottom-to-top; x/y about the cell center','time_convention':'exp(-i*omega*t)',
            'time_step_seconds':diagnostics.get('time_step_seconds'),
            'length_units':u.length,'time_unit_seconds':u.time_unit_seconds,
            'incident_amplitude':amplitude,'algorithm':numerics.algorithm,
            'plane_power_units':{name:plane.metadata['power_units'] for name,plane in planes.items()}},
        _engine_diagnostics=diag)


# The scalar settings of the Numerics a result's provenance carries stay static under jax.jit
# (strings, such as Execution's and Units', are static already).
register_dataclass(Numerics,static_fields=('spacing','duration_seconds','algorithm','courant',
    'pml_cells','pml_reflection','subpixel','padding_scale','align_layers','variable_depth',
    'layer_smoothing','shape_integration','extraction_method','checkpoint_steps','dft_subsample',
    'decay_check'))


def _run(problem,numerics,execution):
    from ._periodic.adapter import solve_periodic_fdtd
    with execution._scope():
        options,backend=_arguments(problem,numerics,execution)
        raw,diag=_engine(solve_periodic_fdtd,options,backend=backend)
        return _adapt(raw,diag,problem,numerics,execution)


@eager_under_torch_compile
def run(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    with refusals('periodic'):
        return _run(problem,numerics,_execution(numerics,backend,device,precision))


__all__=['Stack','Layer','Region','Rectangle','Circle','Ellipse','Polygon','Excitation','PlaneWave',
         'RasterDispersion','Pulse','Snapshots','Outputs','DecayCheck','Numerics','Problem',
         'Prepared','Result','check','prepare','run']


def __getattr__(name):
    # A tombstone naming the replacement, not an alias.
    if name == 'Record':
        raise _removed('wavebit.fdtd.periodic', name, 'wavebit.fdtd.periodic.Snapshots')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
