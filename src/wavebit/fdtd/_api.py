"""Validated, immutable FDTD preparation with complete live inputs on each run."""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
import numpy as np
from wavebit._common.execution import Execution, CheckReport, frozen_mapping
from wavebit._common.backend import eager_under_torch_compile
from wavebit.errors import Issue, ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from wavebit.materials import Material, ConductiveMaterial, DispersiveMaterial
from .problem import Problem
from ._lower_public import lower_problem
from ._execution import _planned, run as execute
from ._prepare.report import report as legacy_report
from .result import adapt_result
from ._spelling import public as _public_spelling, public_error, to_public


def _static(v,path):
    if v is None or isinstance(v,(str,bool,int,float,complex)):return v
    if callable(v):return v
    if isinstance(v,Mapping):return tuple((k,_static(x,path+'.'+str(k))) for k,x in v.items())
    if isinstance(v,(tuple,list)):return tuple(_static(x,path) for x in v)
    if is_dataclass(v):return (type(v).__name__,tuple((f.name,_static(getattr(v,f.name),path+'.'+f.name)) for f in fields(v)))
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
                       for f in fields(s) if f.name not in ('pulse','amplitude','current_moment','phase','delay')),s.pulse.cutoff))
    initial=None if problem.initial is None else (type(problem.initial).__name__,_static(problem.initial.E,'initial.E'),_static(problem.initial.H,'initial.H'))
    return dict(units=problem.units.length,domain=_static(problem.scene.domain,'scene.domain'),
        boundaries=_static(problem.scene.boundaries,'scene.boundaries'),background=_material(problem.scene.background),
        structures=tuple(structures),sources=tuple(sources),outputs=_static(problem.outputs,'outputs'),initial=initial)


def _execution(backend,device,precision):
    value=Execution(backend=backend,device=device,precision=precision)
    value.validate()
    if device!='cpu':
        raise UnsupportedFeatureError('the current FDTD executor supports CPU only',code='device',path='execution.device')
    if precision!='double':
        raise UnsupportedFeatureError('the current FDTD executor requires double precision',code='precision',path='execution.precision')
    return value


def _snapped(problem,scene,plan):
    """A warning for each cell-snapped structure whose interfaces the grid moved.

    A Slab or Region fills whole cells, so an interface between two cell boundaries lands on
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
    resolved=dict(metadata,grid_shape=(plan.nx,plan.ny,plan.nz),spacing=plan.dx,
                  dt_seconds=plan.dt*problem.units.time_seconds,steps=plan.steps,
                  duration_seconds=plan.steps*plan.dt*problem.units.time_seconds,dtype=plan.dtype,
                  monitor_samples=tuple((o.name,getattr(o,'indices',None)) for o in plan.observers))
    return CheckReport(issues=_snapped(problem,scene,plan),
                       estimates={'state_bytes':r.bytes_per_state,'parameter_bytes':r.bytes_per_params,
        'scope':'main E/H, CPML and material arrays; excludes runtime, temporary, source and observation storage'},
        capabilities={'backend':execution.backend,'precision':execution.precision,'device':execution.device,
                      'warnings':tuple(to_public(w) for w in r.warnings),
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
        with execution.scope(),_public_spelling():
            scene,metadata=lower_problem(problem,numerics)
            plan,_=_planned(scene)
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
            with self.execution.scope(),_public_spelling():
                self._lower(problem)
            return CheckReport(resolved=self.report.resolved,capabilities=self.report.capabilities)
        except (TypeError,ValueError,NotImplementedError) as exc:
            return CheckReport(issues=(_issue(exc).issue,))

    def _lower(self,problem):
        if not isinstance(problem, Problem):
            raise ValidationError('expected fdtd.Problem', code='problem.type', path='problem')
        signature=_signature(problem,self.numerics)
        for key,value in signature.items():
            if value!=self._signature[key]:
                raise IncompatibleProblemError('changed prepared structure; prepare this problem again',code='prepared_structure',path=key)
        scene,metadata=lower_problem(problem,self.numerics)
        plan,_=_planned(scene,reuse=self._plan)
        if plan.skeleton!=self._plan.skeleton:
            path='numerics.duration' if (plan.steps,plan.dt)!=(self._plan.steps,self._plan.dt) else 'problem'
            raise IncompatibleProblemError('resolved numerical dependencies changed; prepare this problem again',code='prepared_dependencies',path=path)
        return scene,metadata

    @eager_under_torch_compile
    def run(self,problem):
        """Run the complete supplied live problem; never update this preparation."""
        with self.execution.scope(),_public_spelling():
            try:scene,metadata=self._lower(problem)
            except (TypeError,ValueError,NotImplementedError) as exc:
                raise _issue(exc) from exc
            try:
                raw=execute(scene,backend=self.execution.backend,device=self.execution.device,plan=self._plan)
                return adapt_result(problem,self.numerics,self.execution,raw,metadata)
            except (TypeError,ValueError,NotImplementedError,RuntimeError) as exc:
                _reraise_public(exc)


@eager_under_torch_compile
def prepare(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Validate concrete inputs and fix the computational structure for later runs."""
    execution=_execution(backend,device,precision)
    with execution.scope(),_public_spelling():
        try:
            scene,metadata=lower_problem(problem,numerics)
            plan,_=_planned(scene)
            signature=_signature(problem,numerics)
            report=_report(problem,numerics,execution,scene,plan,metadata)
        except (TypeError,ValueError,NotImplementedError) as exc:
            raise _issue(exc) from exc
    contract=frozen_mapping({'static':tuple(signature),'live':('material coefficients subject to constitutive guards',
        'source amplitude/phase/delay and band subject to fixed plan dependencies','initial amplitude',
        'ranged geometry subject to reserved-support guards'),
        'differentiation':'existing Yee runtime guards apply; invalid transformed states return NaN and failed validity',
        'resolved_steps':plan.steps,'resolved_dt_seconds':plan.dt*problem.units.time_seconds})
    return Prepared(numerics,execution,report,contract,plan,frozen_mapping(signature))


@eager_under_torch_compile
def run(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Prepare and execute one complete physical problem."""
    return prepare(problem,numerics=numerics,backend=backend,device=device,precision=precision).run(problem)
