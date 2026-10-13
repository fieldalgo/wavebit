"""Validated, immutable FDTD preparation with complete live inputs on each run."""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
import numpy as np
from wavebit._common.execution import Execution, CheckReport, frozen_mapping
from wavebit._common.backend import eager_under_torch_compile
from wavebit.errors import ValidationError, UnsupportedFeatureError, IncompatibleProblemError
from wavebit.materials import Material, ConductiveMaterial, DispersiveMaterial
from .problem import Problem
from ._lower_public import lower_problem
from ._execution import _planned, run as execute
from ._prepare.report import report as legacy_report
from .result import adapt_result


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
        structures.append((s.name,shape,_material(s.material)))
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


def _report(problem,numerics,execution,scene,plan,metadata):
    r=legacy_report(scene,plan)
    resolved=dict(metadata,grid_shape=(plan.nx,plan.ny,plan.nz),spacing=plan.dx,
                  dt_seconds=plan.dt*problem.units.time_seconds,steps=plan.steps,
                  duration_seconds=plan.steps*plan.dt*problem.units.time_seconds,dtype=plan.dtype,
                  monitor_samples=tuple((o.name,getattr(o,'indices',None)) for o in plan.observers))
    return CheckReport(estimates={'state_bytes':r.bytes_per_state,'parameter_bytes':r.bytes_per_params,
        'scope':'main E/H, CPML and material arrays; excludes runtime, temporary, source and observation storage'},
        capabilities={'backend':execution.backend,'precision':execution.precision,'device':execution.device,
                      'warnings':r.warnings,'geometry_ranges':tuple(numerics.geometry_ranges)},resolved=resolved)


def _issue(exc):
    return exc if isinstance(exc,ValidationError) else ValidationError(str(exc),code='fdtd_input',path='problem')


@eager_under_torch_compile
def check(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Validate and estimate resources without advancing any field."""
    try:
        execution=_execution(backend,device,precision)
        with execution.scope():
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
            with self.execution.scope():
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
        with self.execution.scope():
            try:scene,metadata=self._lower(problem)
            except (TypeError,ValueError,NotImplementedError) as exc:
                raise _issue(exc) from exc
            raw=execute(scene,backend=self.execution.backend,device=self.execution.device,plan=self._plan)
            return adapt_result(problem,self.numerics,self.execution,raw,metadata)


@eager_under_torch_compile
def prepare(problem,*,numerics,backend='numpy',device='cpu',precision='double'):
    """Validate concrete inputs and fix the computational structure for later runs."""
    execution=_execution(backend,device,precision)
    with execution.scope():
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
