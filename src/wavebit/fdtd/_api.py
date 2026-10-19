"""Validated, immutable FDTD preparation with complete live inputs on each run."""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
import numpy as np
from wavebit._common.execution import Execution, CheckReport, frozen_mapping
from wavebit._common.backend import eager_under_torch_compile
from wavebit.errors import Issue, ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from wavebit.materials import Material, ConductiveMaterial, DispersiveMaterial
from ._problem import Problem
from ._policy_report import numerical_policies
from ._lower_public import lower_problem, validate_sources
from ._execution import _planned, run as execute
from ._prepare.report import report as legacy_report
from ._result import adapt_result
from ._spelling import public as _public_spelling, public_error, to_public


def _static(v,path):
    if v is None or isinstance(v,(str,bool,int,float,complex)):return v
    if callable(v):return v
    if isinstance(v,Mapping):return tuple((k,_static(x,path+'.'+str(k))) for k,x in v.items())
    if isinstance(v,(tuple,list)):return tuple(_static(x,path) for x in v)
    if is_dataclass(v):return (type(v).__name__,tuple((f.name,_static(getattr(v,f.name),path+'.'+f.name)) for f in fields(v)))
    from ._geometry import Shape
    if isinstance(v, Shape):
        # Custom protocol implementations are static declarations. Include inherited
        # slots as well as ordinary attributes; object-array bytes are pointer identity.
        names = set(vars(v)) if hasattr(v, '__dict__') else set()
        for cls in type(v).__mro__:
            slots = getattr(cls, '__slots__', ())
            names.update((slots,) if isinstance(slots, str) else slots)
        names -= {'__dict__', '__weakref__'}
        return (type(v), tuple((name, _static(getattr(v, name), path+'.'+name))
                              for name in sorted(names) if hasattr(v, name)))
    try:
        if getattr(v,'requires_grad',False):raise ValueError('differentiable static value')
        array=np.asarray(v)
        return (array.shape,array.dtype.str,array.tobytes())
    except Exception as exc:
        raise IncompatibleProblemError('this field is static for a prepared calculation',code='static_input',path=path) from exc


def _leaf_shape(v):
    if v is None:return None
    return tuple(getattr(v,'shape',np.shape(v)))


def _material(v):
    if isinstance(v,Material):return ('Material',_leaf_shape(v.epsilon),_leaf_shape(v.mu))
    if isinstance(v,ConductiveMaterial):return ('ConductiveMaterial',_leaf_shape(v.epsilon),_leaf_shape(v.mu),_leaf_shape(v.conductivity))
    if isinstance(v,DispersiveMaterial):return ('DispersiveMaterial',_leaf_shape(v.epsilon_inf),_leaf_shape(v.mu),tuple(type(p).__name__ for p in v.poles))
    return (type(v).__name__,_leaf_shape(v))


def _signature(problem,numerics):
    validate_sources(problem)
    structures=[]
    for s in problem.scene.structures:
        if s.name in numerics.geometry_ranges:
            shape=(type(s.shape).__name__,tuple((f.name,_static(getattr(s.shape,f.name),'scene.structures.'+s.name+'.'+f.name))
                  for f in fields(s.shape) if f.name in ('axis','angle')),
                  len(getattr(s.shape,'vertices',())))
        else:shape=_static(s.shape,'scene.structures.'+s.name+'.shape')
        structures.append((s.name,shape,_material(s.material),s.continuation))
    sources=[]
    for s in problem.sources:
        sources.append((type(s).__name__,tuple((f.name,_static(getattr(s,f.name),'sources.'+s.name+'.'+f.name))
                       for f in fields(s) if f.name not in ('pulse','amplitude','current_moment','phase','delay_seconds')),s.pulse.cutoff))
    initial=None if problem.initial is None else (type(problem.initial).__name__,_static(problem.initial.E,'initial.E'),_static(problem.initial.H,'initial.H'))
    return dict(units=problem.units.length,domain=_static(problem.scene.domain,'scene.domain'),
        boundaries=_static(problem.scene.boundaries,'scene.boundaries'),background=_material(problem.scene.background),
        structures=tuple(structures),sources=tuple(sources),outputs=_static(problem.outputs,'outputs'),initial=initial)


def _execution(backend,device,precision):
    value=Execution(backend=backend,device=device,precision=precision)
    value.validate()
    if device!='cpu' and not (backend=='torch' and device.split(':')[0]=='cuda'):
        raise UnsupportedFeatureError('FDTD supports CPU or the limited Torch CUDA route',code='device',path='execution.device')
    if precision!='double':
        raise UnsupportedFeatureError('the current FDTD executor requires double precision',code='precision',path='execution.precision')
    return value


def _checkpoint(numerics,execution):
    """Resolve the opt-in static schedule without changing a default CPU run."""
    value=numerics.checkpoint_steps
    if value is None:
        return None
    from wavebit._common.declarations import static_int
    path='numerics.checkpoint_steps'
    try:
        if isinstance(value,(bool,np.bool_)) or str(getattr(value,'dtype','')) in ('bool','torch.bool'):
            raise ValueError('a bool is not a checkpoint block length')
        value=static_int('Numerics','checkpoint_steps',value)
        if value<=0:
            raise ValueError('checkpoint_steps must be positive')
    except (TypeError,ValueError) as exc:
        raise ValidationError('checkpoint_steps must be a positive static integer or None',
                              code='checkpoint_steps',path=path) from exc
    if execution.backend!='torch' or execution.device.split(':')[0]!='cuda':
        raise UnsupportedFeatureError('checkpoint_steps currently requires Torch CUDA execution',
                                      code='checkpoint_scope',path=path)
    if numerics.stopping is not None:
        raise UnsupportedFeatureError('checkpoint_steps requires fixed-duration execution without a DecayStop',
                                      code='checkpoint_scope',path='numerics.stopping')
    return value


def _cuda_homogeneous_ade(problem,scene):
    """Admit only the independently qualified homogeneous oscillator geometry.

    Filled cells are checked on the lowered grid, using the planner's own
    rounding rule. Material validity and pole-aware stability remain the
    ordinary scientific guards' responsibility, including on prepared reuse.
    """
    from . import _boundaries as boundaries, _geometry as geometry, _monitors as monitors
    from ._kernel.grid import Grid
    from wavebit.materials import Drude, Lorentz
    from wavebit._common.declarations import static_float

    def refuse(message,path):
        raise UnsupportedFeatureError(message,code='cuda_scope',path=path)

    if type(problem.scene.background) is not Material:
        refuse('CUDA homogeneous ADE requires an ordinary Material background',
               'scene.background')
    if len(problem.scene.structures)!=1:
        refuse('CUDA homogeneous ADE requires exactly one full-grid SnappedSlab','scene.structures')
    structure=problem.scene.structures[0]
    path='scene.structures.'+structure.name
    material=structure.material
    if (type(structure.shape) is not geometry.SnappedSlab or
            type(material) is not DispersiveMaterial):
        refuse('CUDA homogeneous ADE requires one DispersiveMaterial SnappedSlab',path)
    for axis in 'xyz':
        for side in '-+':
            if type(problem.scene.boundaries.face(axis+side)) is not boundaries.Periodic:
                refuse('CUDA homogeneous ADE requires six Periodic faces',
                       'scene.boundaries.'+axis+side)
    if len(material.poles)!=1 or type(material.poles[0]) not in (Lorentz,Drude):
        refuse('CUDA homogeneous ADE requires exactly one Lorentz or Drude pole',
               path+'.material.poles')
    pole=material.poles[0]
    fixed=[('epsilon_inf',material.epsilon_inf),('mu',1. if material.mu is None else material.mu),
           ('poles[0].gamma',pole.gamma)]
    if type(pole) is Lorentz:
        fixed.append(('poles[0].omega_0',pole.omega_0))
    for name,value in fixed:
        where=path+'.material.'+name
        if _leaf_shape(value)!=():
            refuse('CUDA homogeneous ADE requires scalar epsilon, mu and pole rates',where)
        try:static_float('homogeneous ADE',name,value)
        except (TypeError,ValueError) as exc:
            raise IncompatibleProblemError(
                'CUDA homogeneous ADE differentiates only oscillator weight and initial amplitude; '
                'epsilon_inf, mu, gamma and omega_0 must be static scalars',
                code='static_input',path=where) from exc
    grid=Grid.from_domain(scene.domain.size,scene.domain.center,scene.resolution.dx)
    if scene.structures[0].shape.filled_cells(grid)!=tuple((0,n) for n in grid.n):
        refuse('CUDA homogeneous ADE SnappedSlab must fill every cell of the actual rounded grid',
               path+'.shape')
    if problem.sources:
        refuse('CUDA homogeneous ADE supports initial fields only','sources')
    initial=problem.initial
    if initial is None or (initial.E is None and initial.H is None):
        refuse('CUDA homogeneous ADE requires a real static initial field','initial')
    for name,value in (('E',initial.E),('H',initial.H)):
        if value is None:
            continue
        if (type(value) is not np.ndarray or value.dtype.kind not in 'fiu'
                or value.shape!=grid.shape or not np.isfinite(value).all()):
            refuse('CUDA homogeneous ADE initial patterns must be finite real host arrays '
                   'of shape (3,nz,ny,nx)', 'initial.'+name)
    amplitude=initial.amplitude
    if (_leaf_shape(amplitude)!=() or isinstance(amplitude,(bool,np.bool_,complex))
            or str(getattr(amplitude,'dtype','')) in ('bool','torch.bool')
            or 'complex' in str(getattr(amplitude,'dtype',''))):
        refuse('CUDA homogeneous ADE initial amplitude must be a real scalar','initial.amplitude')
    for monitor in problem.outputs.monitors:
        if type(monitor) is not monitors.TimeSeriesMonitor:
            refuse('CUDA homogeneous ADE supports TimeSeriesMonitor and final fields only',
                   'outputs.'+monitor.name)


def _cuda_region(problem,scene):
    """The fixed, strictly interior, scalar-material SnappedBox CUDA family.

    Geometry and initial patterns remain static. Material values are inspected only
    for scalar real shape here; their ordinary scientific runtime guards retain AD.
    """
    from . import _boundaries as boundaries, _geometry as geometry, _monitors as monitors, _sources as sources
    from ._kernel.grid import Grid

    def refuse(message,path):
        raise UnsupportedFeatureError(message,code='cuda_scope',path=path)

    if len(problem.scene.structures)!=1:
        refuse('CUDA SnappedBox requires exactly one structure without SnappedSlab mixtures', 'scene.structures')
    structure=problem.scene.structures[0]
    path='scene.structures.'+structure.name
    if type(structure) is not geometry.Structure or type(structure.shape) is not geometry.SnappedBox:
        refuse('CUDA SnappedBox requires an exact static SnappedBox structure',path)
    if structure.continuation:
        refuse('CUDA SnappedBox does not support continuation',path+'.continuation')
    for axis in 'xyz':
        for side in '-+':
            if type(problem.scene.boundaries.face(axis+side)) is not boundaries.Periodic:
                refuse('CUDA SnappedBox requires six Periodic faces','scene.boundaries.'+axis+side)
    grid=Grid.from_domain(scene.domain.size,scene.domain.center,scene.resolution.dx)
    cells=scene.structures[0].shape.filled_cells(grid)
    if any(not 0<lo<hi<n for (lo,hi),n in zip(cells,grid.n)):
        refuse('CUDA SnappedBox must occupy strictly interior cells on every axis; '
               'seams and full-span axes are not supported',path+'.shape')

    def real_scalar(value):
        return (_leaf_shape(value)==() and not isinstance(value,(bool,np.bool_,complex))
                and str(getattr(value,'dtype','')) not in ('bool','torch.bool')
                and 'complex' not in str(getattr(value,'dtype','')))

    for where,material in (('scene.background',problem.scene.background),
                            (path+'.material',structure.material)):
        if type(material) not in (Material,ConductiveMaterial):
            refuse('CUDA SnappedBox supports scalar Material or ConductiveMaterial only',where)
        leaves=[('epsilon',material.epsilon),('mu',1. if material.mu is None else material.mu)]
        if type(material) is ConductiveMaterial:
            leaves.append(('conductivity',material.conductivity))
        for name,value in leaves:
            if not real_scalar(value):
                refuse('CUDA SnappedBox requires real scalar material values',where+'.'+name)
    if problem.sources:
        refuse('CUDA SnappedBox supports initial fields only','sources')
    initial=problem.initial
    if (type(initial) is not sources.InitialField or
            (initial.E is None and initial.H is None)):
        refuse('CUDA SnappedBox requires an ordinary real static InitialField','initial')
    for name,value in (('E',initial.E),('H',initial.H)):
        if value is None:
            continue
        if (type(value) is not np.ndarray or value.dtype.kind not in 'fiu'
                or value.shape!=grid.shape or not np.isfinite(value).all()):
            refuse('CUDA SnappedBox initial patterns must be finite real host arrays '
                   'of shape (3,nz,ny,nx)','initial.'+name)
    if not real_scalar(initial.amplitude):
        refuse('CUDA SnappedBox initial amplitude must be a real scalar','initial.amplitude')
    for monitor in problem.outputs.monitors:
        if type(monitor) is not monitors.TimeSeriesMonitor:
            refuse('CUDA SnappedBox supports TimeSeriesMonitor and final fields only','outputs.'+monitor.name)


def _cuda_scope(problem,numerics,execution,*,scene=None):
    """Keep the first CUDA route inside the cases covered by device regressions.

    Planning remains on the host. This admission is separate from the scientific
    guards in lowering/planning, which still apply on every preparation and reuse.
    """
    if execution.device == 'cpu':
        return
    from . import _boundaries as boundaries, _geometry as geometry, _monitors as monitors, _sources as sources
    from ._numerics import PML
    from wavebit._common.declarations import static_float

    def refuse(message,path):
        raise UnsupportedFeatureError(message,code='cuda_scope',path=path)

    if numerics.stopping is not None:
        refuse('CUDA requires a fixed duration; DecayStop remains CPU only','numerics.stopping')
    if not isinstance(numerics.absorber,PML):
        refuse('CUDA currently supports PML padding only','numerics.absorber')
    if numerics.discretization!='voxel':
        refuse('CUDA currently requires voxel discretization','numerics.discretization')
    if numerics.geometry_ranges:
        refuse('CUDA does not yet admit ranged geometry','numerics.geometry_ranges')
    faces=tuple((axis+side,problem.scene.boundaries.face(axis+side))
                for axis in 'xyz' for side in '-+')
    has_bloch=any(type(face) is boundaries.Bloch for _,face in faces)
    for name,face in faces:
        if type(face) not in (boundaries.Periodic,boundaries.Open,boundaries.PEC,
                              boundaries.PMC,boundaries.Bloch):
            refuse('CUDA currently supports Periodic, Open, PEC, PMC and static Bloch faces only',
                   'scene.boundaries.'+name)
        if type(face) is boundaries.Bloch:
            for parameter in ('phase','k'):
                value=getattr(face,parameter)
                if value is not None:
                    try:static_float('Bloch',parameter,value)
                    except (TypeError,ValueError) as exc:
                        raise IncompatibleProblemError(
                            'CUDA Bloch phase and wavevector must be static scalars',
                            code='static_input',path='scene.boundaries.'+name+'.'+parameter) from exc
        if has_bloch and type(face) in (boundaries.PEC,boundaries.PMC):
            refuse('CUDA Bloch currently supports Periodic or Open companion faces only',
                   'scene.boundaries.'+name)
    for structure in problem.scene.structures:
        if type(structure.shape) not in (geometry.SnappedSlab,geometry.SnappedBox):
            refuse('CUDA currently supports SnappedSlab or restricted interior SnappedBox structures only',
                   'scene.structures.'+structure.name+'.shape')
    if any(type(s.shape) is geometry.SnappedBox for s in problem.scene.structures):
        if scene is None:
            scene,_=lower_problem(problem,numerics)
        _cuda_region(problem,scene)
    materials=(('scene.background',problem.scene.background),)+tuple(
        ('scene.structures.'+s.name+'.material',s.material) for s in problem.scene.structures)
    homogeneous_ade=any(isinstance(material,DispersiveMaterial) for _,material in materials)
    if homogeneous_ade:
        if scene is None:
            scene,_=lower_problem(problem,numerics)
        _cuda_homogeneous_ade(problem,scene)
    for path,material in materials:
        if (type(material) not in (Material,ConductiveMaterial)
                and not (homogeneous_ade and type(material) is DispersiveMaterial)):
            refuse('CUDA supports Material, ConductiveMaterial and restricted homogeneous ADE',path)
    for source in problem.sources:
        if type(source) not in (sources.PlaneWave, sources.CurrentDipole):
            refuse('CUDA currently supports PlaneWave, fixed-center CurrentDipole sources or initial fields only',
                   'sources.'+source.name)
        if has_bloch and type(source) is sources.CurrentDipole:
            refuse('CUDA Bloch currently supports PlaneWave sources or initial fields only; '
                   'complex CurrentDipole source work is not qualified','sources.'+source.name)
    for monitor in problem.outputs.monitors:
        if type(monitor) not in (monitors.TimeSeriesMonitor,monitors.FieldMonitor,monitors.FluxMonitor):
            refuse('CUDA currently supports TimeSeriesMonitor, FieldMonitor and FluxMonitor only',
                   'outputs.'+monitor.name)


def _compact_poles(problem,execution):
    """Select compact memory only after the bounded CUDA admission succeeds."""
    return (execution.backend=='torch' and execution.device.split(':')[0]=='cuda'
            and any(type(s.material) is DispersiveMaterial for s in problem.scene.structures))


def _snapped(problem,scene,plan):
    """A warning for each cell-snapped structure whose interfaces the grid moved.

    A SnappedSlab or SnappedBox fills whole cells, so an interface between two cell boundaries lands on
    the nearer one: up to half a cell, a first-order error in the spacing. Two spans inside
    the same cells give bit-identical runs -- a 0.317 and a 0.325 film at 0.025 spacing did --
    and the compat report printed the move while this one dropped it. The rows are the
    declared shape's on the run's grid, so a half-space continued into an Open face's padding,
    or a span clipped by the domain, is not reported as a move: only a face inside the grid is.
    """
    grid=plan.grid
    tol=1e-9*plan.dx
    issues=[]
    for declared,lowered in zip(problem.scene.structures,scene.structures):
        if lowered.shape.body() is not None:
            continue                                  # smoothed: faces stay where asked
        asked,got=[],[]
        for label,lo,hi,low,high in declared.shape.span_rows(grid):
            a='xyz'.index(label[-1])
            span=(max(lo,grid.low[a]),min(hi,grid.high[a]))
            if abs(span[0]-low)>tol or abs(span[1]-high)>tol:
                asked.append(('xyz'[a],)+span)
                got.append(('xyz'[a],float(f'{low:.12g}'),float(f'{high:.12g}')))
        if not asked:
            continue
        moves=', '.join(f'{a}=({lo:.6g}, {hi:.6g}) -> ({slo:.6g}, {shi:.6g})'
                        for (a,lo,hi),(_,slo,shi) in zip(asked,got))
        issues.append(Issue(code='geometry.cell_snapped',severity='warning',
            path='scene.structures.'+declared.name,expected=tuple(asked),actual=tuple(got),
            message=f'fdtd.{type(declared.shape).__name__} {declared.name!r} is cell-snapped: its '
                    f'interfaces move to the nearest cell boundary at spacing {plan.dx:g}, '
                    f'{moves}. That is up to half a cell, a first-order error in the spacing, and '
                    f'any two spans inside the same cells run identically. Make the span a whole '
                    f'number of cells or check convergence as the spacing shrinks. A lossless, '
                    f'non-dispersive layer clear of every Open face and of every cell-snapped face '
                    f'can instead be an unturned fdtd.Box reaching more than a cell past both faces '
                    f'of the other two axes, whose smoothed faces stay where they were asked.'))
    return tuple(issues)


def _report(problem,numerics,execution,scene,plan,metadata):
    r=legacy_report(scene,plan)
    resolved=dict(metadata,numerical_policies=numerical_policies(problem,numerics,scene,plan),grid_shape=(plan.nx,plan.ny,plan.nz),spacing=plan.dx,
                  dt_seconds=plan.dt*problem.units.time_unit_seconds,steps=plan.steps,
                  duration_seconds=plan.steps*plan.dt*problem.units.time_unit_seconds,dtype=plan.dtype,
                  checkpoint_steps=plan.schedule,
                  duration_policy='fixed' if numerics.stopping is None else 'hard cap with concrete state-decay stopping',
                  monitor_samples=tuple((o.name,getattr(o,'indices',None)) for o in plan.observers))
    return CheckReport(issues=_snapped(problem,scene,plan),
                       estimates={'state_bytes':r.bytes_per_state,'parameter_bytes':r.bytes_per_params,
        'scope':'main E/H, CPML and material arrays; excludes runtime, temporary, source and observation storage'},
        capabilities={'backend':execution.backend,'precision':execution.precision,'device':execution.device,
                      'runtime_validation':'concrete_inputs' if r.params_known else 'deferred',
                      'warnings':tuple(to_public(w) for w in r.warnings),
                      'automatic_stopping':numerics.stopping is not None,
                      'geometry_ranges':tuple(numerics.geometry_ranges)},resolved=resolved)


def _issue(exc):
    """``exc`` as a structured error, in the public vocabulary (``_spelling``)."""
    exc=public_error(exc)
    return exc if isinstance(exc,ValidationError) else ValidationError(str(exc),code='fdtd_input',path='problem')


def _reraise_public(exc):
    """Raise ``exc`` again, spelled for a public caller; its type is kept."""
    translated=public_error(exc)
    if translated is exc:
        raise exc
    raise translated from exc


@eager_under_torch_compile
def check(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Validate and estimate resources without advancing any field."""
    try:
        execution=_execution(backend,device,precision)
        with execution._scope(),_public_spelling():
            scene,metadata=lower_problem(problem,numerics)
            schedule=_checkpoint(numerics,execution)
            _cuda_scope(problem,numerics,execution,scene=scene)
            plan,_=_planned(scene,schedule=schedule,compact_poles=_compact_poles(problem,execution))
            _signature(problem,numerics)
            return _report(problem,numerics,execution,scene,plan,metadata)
    except (TypeError,ValueError,NotImplementedError) as exc:
        return CheckReport(issues=(_issue(exc).issue,))


@eager_under_torch_compile
@dataclass(frozen=True)
class Prepared:
    numerics: object
    execution: Execution
    report: CheckReport
    contract: object
    _plan: object
    _signature: object

    @eager_under_torch_compile
    def validate(self,problem):
        """Validate concrete reuse; runtime guards still judge transformed values."""
        try:
            with self.execution._scope(),_public_spelling():
                scene,metadata=self._lower(problem)
                return _report(problem,self.numerics,self.execution,scene,self._plan,metadata)
        except (TypeError,ValueError,NotImplementedError) as exc:
            return CheckReport(issues=(_issue(exc).issue,))

    def _lower(self,problem):
        if not isinstance(problem, Problem):
            raise ValidationError('expected fdtd.Problem', code='problem.type', path='problem')
        signature=_signature(problem,self.numerics)
        for key,value in signature.items():
            if value!=self._signature[key]:
                raise IncompatibleProblemError('changed prepared structure; prepare this problem again',code='prepared_structure',path=key)
        scene,metadata=lower_problem(problem,self.numerics,plan=self._plan)
        schedule=_checkpoint(self.numerics,self.execution)
        _cuda_scope(problem,self.numerics,self.execution,scene=scene)
        plan,_=_planned(scene,reuse=self._plan,schedule=schedule,
                        compact_poles=_compact_poles(problem,self.execution))
        if plan.skeleton!=self._plan.skeleton:
            path='numerics.duration' if (plan.steps,plan.dt)!=(self._plan.steps,self._plan.dt) else 'problem'
            raise IncompatibleProblemError('resolved numerical dependencies changed; prepare this problem again',code='prepared_dependencies',path=path)
        return scene,metadata

    @eager_under_torch_compile
    def run(self,problem):
        """Run the complete supplied live problem; never update this preparation."""
        with self.execution._scope(),_public_spelling():
            try:scene,metadata=self._lower(problem)
            except (TypeError,ValueError,NotImplementedError) as exc:
                raise _issue(exc) from exc
            try:
                from ._prepare.fields import static_host_patterns
                cuda_patterns=(self.execution.backend=='torch'
                               and self.execution.device.split(':')[0]=='cuda')
                # Explicit False also isolates a nested public CPU/JAX run from
                # an outer CUDA callback's pattern policy. The token is restored.
                with static_host_patterns(cuda_patterns):
                    raw=execute(scene,backend=self.execution.backend,device=self.execution.device,plan=self._plan)
                metadata=dict(metadata,checkpoint_steps=raw.plan.schedule,numerical_policies=numerical_policies(
                    problem,self.numerics,scene,raw.plan))
                return adapt_result(problem,self.numerics,self.execution,raw,metadata)
            except (TypeError,ValueError,NotImplementedError,RuntimeError) as exc:
                _reraise_public(exc)


@eager_under_torch_compile
def prepare(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Validate concrete inputs and fix the computational structure for later runs."""
    execution=_execution(backend,device,precision)
    with execution._scope(),_public_spelling():
        try:
            scene,metadata=lower_problem(problem,numerics)
            schedule=_checkpoint(numerics,execution)
            _cuda_scope(problem,numerics,execution,scene=scene)
            plan,_=_planned(scene,schedule=schedule,compact_poles=_compact_poles(problem,execution))
            signature=_signature(problem,numerics)
            report=_report(problem,numerics,execution,scene,plan,metadata)
        except (TypeError,ValueError,NotImplementedError) as exc:
            raise _issue(exc) from exc
    homogeneous_ade=(execution.device!='cpu' and any(
        type(s.material) is DispersiveMaterial for s in problem.scene.structures))
    from ._geometry import SnappedBox
    interior_region=(execution.device!='cpu' and any(
        type(s.shape) is SnappedBox for s in problem.scene.structures))
    material_live=('oscillator weight (Lorentz strength or Drude omega_p) subject to constitutive guards'
                   if homogeneous_ade else 'material coefficients subject to constitutive guards')
    source_live=(() if interior_region else
                 ('source amplitude/phase/delay_seconds and pulse subject to fixed plan dependencies',))
    contract=frozen_mapping({'static':tuple(signature),'live':(material_live,)+source_live+('initial amplitude',)
        + (('ranged geometry subject to reserved-support guards',) if execution.device=='cpu' else ()),
        'differentiation':('concrete forward only; DecayStop refuses AD and external transforms' if numerics.stopping is not None
                          else 'CUDA homogeneous ADE: oscillator weight (Lorentz strength or Drude omega_p) '
                               'and initial amplitude support eager reverse derivatives; epsilon_inf, mu, gamma '
                               'and omega_0 are nondifferentiable scalars; initial patterns are static host arrays'
                          if homogeneous_ade else
                          'CUDA SnappedBox: one static strictly interior SnappedBox with scalar real epsilon, mu '
                          'and conductivity, plus initial amplitude, supports eager reverse derivatives; '
                          'six Periodic faces and real static host initial patterns; TimeSeriesMonitor/final fields only'
                          if interior_region else
                               'existing Yee runtime guards apply; invalid transformed states return NaN and failed validity'),
        'resolved_steps':plan.steps,'resolved_dt_seconds':plan.dt*problem.units.time_unit_seconds,
        'resolved_checkpoint_steps':plan.schedule})
    return Prepared(numerics,execution,report,contract,plan,frozen_mapping(signature))


@eager_under_torch_compile
def run(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Prepare and execute one complete physical problem."""
    return prepare(problem,numerics=numerics,backend=backend,device=device,precision=precision).run(problem)
