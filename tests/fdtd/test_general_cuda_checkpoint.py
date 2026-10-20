"""Opt-in general CUDA checkpoints preserve public lifecycle and derivatives."""
from types import SimpleNamespace

import numpy as np
import pytest

from wavebit import Execution, fdtd
from wavebit.errors import ValidationError
from wavebit.units import EPSILON0
from tests.fdtd.test_general_cuda import closed_problem, fixed_numerics, host, parity, slab_problem
from tests.fdtd.test_general_cuda import cuda as cuda  # Re-export the shared pytest fixture.
from tests.fdtd.test_general_cuda_observations import observable_losses


@pytest.mark.parametrize('value',[0,-1,True,np.bool_(True),1.5,float('inf'),float('nan'),'11'])
def test_checkpoint_rejects_nonpositive_or_noninteger_block_before_planning(monkeypatch,value):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    monkeypatch.setattr(api,'_planned',lambda *a,**k:pytest.fail('invalid checkpoint reached planning'))
    p=closed_problem();n=fixed_numerics().replace(checkpoint_steps=value)
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda')
    assert not report.ok and report.issues[0].code=='checkpoint_steps'
    assert report.issues[0].path=='numerics.checkpoint_steps'
    with pytest.raises(ValidationError):fdtd.prepare(p,numerics=n,backend='torch',device='cuda')


@pytest.mark.parametrize('backend',['numpy','torch','jax'])
def test_checkpoint_is_not_silently_ignored_by_cpu_backends(monkeypatch,backend):
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    report=fdtd.check(closed_problem(),numerics=fixed_numerics().replace(checkpoint_steps=11),backend=backend)
    assert not report.ok and report.issues[0].code=='checkpoint_scope'
    assert report.issues[0].path=='numerics.checkpoint_steps'


def test_checkpoint_refuses_decay_and_carried_block_lengths(monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    p=closed_problem()
    n=fixed_numerics().replace(checkpoint_steps=11,stopping=fdtd.DecayStop(threshold=1e-3,window_seconds=1e-15))
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda')
    assert not report.ok and report.issues[0].code=='checkpoint_scope'
    assert report.issues[0].path=='numerics.stopping'
    n=fixed_numerics().replace(checkpoint_steps=torch.tensor(11.,requires_grad=True))
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda')
    assert not report.ok and report.issues[0].code=='checkpoint_steps'


@pytest.mark.parametrize('value',[True,False,[True]])
def test_checkpoint_rejects_boolean_tensors(monkeypatch,value):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    numerics=fixed_numerics().replace(checkpoint_steps=torch.tensor(value))
    report=fdtd.check(closed_problem(),numerics=numerics,backend='torch',device='cuda')
    assert not report.ok and report.issues[0].code=='checkpoint_steps'
    assert report.issues[0].path=='numerics.checkpoint_steps'


def test_checkpoint_plan_policy_survives_check_prepare_and_revalidation(monkeypatch):
    import wavebit.fdtd._api as api
    from wavebit.fdtd._lower_public import lower_problem
    from wavebit.fdtd._prepare import plan as engine_plan
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    monkeypatch.setattr(api,'execute',lambda *a,**k:pytest.fail('planning advanced fields'))
    p=closed_problem();n=fixed_numerics(163)
    scene,_=lower_problem(p,n)
    original=engine_plan(scene)
    plain=fdtd.prepare(p,numerics=n)
    assert plain._plan.skeleton==original.skeleton
    assert plain._plan.schedule is None
    n=n.replace(checkpoint_steps=np.int64(11))
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda')
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    assert report.ok and report.resolved['checkpoint_steps']==11
    assert prepared.contract['resolved_checkpoint_steps']==11
    assert prepared._plan.schedule==11 and type(prepared._plan.schedule) is int
    assert prepared._plan.skeleton!=plain._plan.skeleton
    assert prepared.validate(p).ok
    changed=p.replace(initial=p.initial.replace(amplitude=1.5))
    assert prepared.validate(changed).resolved['checkpoint_steps']==11


def test_internal_run_keeps_schedule_on_both_polarization_solves(monkeypatch):
    # This checks private execution plumbing, not public CUDA DiffractionMonitor support.
    import wavebit.fdtd._execution as execution
    class Both:
        name='port'
        polarizations='both'
    class Port:
        orders=SimpleNamespace(incident='second incidence')
        def with_both(self,other):return ('combined',other)
    class Result:
        def __init__(self):self._series={'port':Port()};self.diagnostics={}
        def __getitem__(self,key):return self._series[key]
    plan=SimpleNamespace(schedule=11,skeleton='same',terms=())
    scene=SimpleNamespace(monitors=(Both(),))
    schedules=[]
    def planned(scene,reuse=None,*,schedule=None,compact_poles=False):
        schedules.append((schedule,compact_poles))
        return SimpleNamespace(schedule=schedule,skeleton='same',terms=()),None
    monkeypatch.setattr(execution,'Diffraction',Both)
    monkeypatch.setattr(execution,'_planned',planned)
    monkeypatch.setattr(execution,'_other_polarization',lambda scene:scene)
    monkeypatch.setattr(execution,'runtime',lambda *a:None)
    monkeypatch.setattr(execution,'_solve',lambda *a:Result())
    result=execution.run(scene,plan=plan)
    assert schedules==[(11,False),(11,False)]
    assert result.diagnostics['runs']==2
    assert result['port']==('combined','second incidence')


@pytest.mark.parametrize('block',[1,64])
def test_checkpoint_single_step_blocks_and_run_shorter_than_block(cuda,block):
    # K=1 exercises every block boundary; K>steps exercises a tail with no block.
    steps=17
    sigma=cuda.tensor(600.,dtype=cuda.float64,device='cuda:0',requires_grad=True)
    amplitude=cuda.tensor(1.3,dtype=cuda.float64,device='cpu',requires_grad=True)
    p=closed_problem(conductivity=sigma,amplitude=amplitude)
    n=fixed_numerics(steps).replace(checkpoint_steps=block)
    result=fdtd.run(p,numerics=n,backend='torch',device='cuda:0')
    assert result.provenance['checkpoint_steps']==block
    loss=result.final_fields.values['E'][1].mean()
    gs,ga=cuda.autograd.grad(loss,(sigma,amplitude))
    factor=result.provenance['time_step_seconds']/(2*EPSILON0*2.25)
    ratio=(1-600*factor)/(1+600*factor)
    np.testing.assert_allclose(host(loss),1.3*ratio**steps,rtol=2e-14,atol=0.)
    np.testing.assert_allclose(host(ga),ratio**steps,rtol=2e-12,atol=0.)
    np.testing.assert_allclose(host(gs),1.3*steps*ratio**(steps-1)*(-2*factor/(1+600*factor)**2),rtol=2e-12,atol=0.)


def test_checkpoint_tail_observable_gradients_and_independent_prepared_reuse(cuda):
    steps,block=163,11
    assert steps%block==9
    base=slab_problem();plain_n=fixed_numerics(steps)
    plain=fdtd.prepare(base,numerics=plain_n,backend='torch',device='cuda:0')
    prepared=fdtd.prepare(base,numerics=plain_n.replace(checkpoint_steps=block),backend='torch',device='cuda:0')
    assert prepared.report.resolved['checkpoint_steps']==block
    saved=None
    def solve(prepared,epsilon,amplitude):
        eps=cuda.tensor(epsilon,dtype=cuda.float64,device='cuda:0',requires_grad=True)
        amp=cuda.tensor(amplitude,dtype=cuda.float64,device='cpu',requires_grad=True)
        result=prepared.run(slab_problem(epsilon=eps,amplitude=amp))
        losses=((result.final_fields.values['E']**2).sum(),)+observable_losses(result,base.units)
        gradients=[]
        for index,loss in enumerate(losses):
            pair=cuda.autograd.grad(loss,(eps,amp),retain_graph=index<len(losses)-1)
            gradients.append([float(g.detach().cpu()) for g in pair])
        values=np.array([float(v.detach().cpu()) for v in losses])
        gradients=np.asarray(gradients)
        assert np.isfinite(values).all() and np.isfinite(gradients).all()
        assert np.all(np.abs(values)>1e-12)
        np.testing.assert_allclose(gradients[:,1],2*values/amplitude,rtol=2e-11,atol=0.)
        return result,values,gradients
    for epsilon,amplitude in ((2.25,1.2),(2.4,.8)):
        expected,values,gradients=solve(plain,epsilon,amplitude)
        result,actual_values,actual_gradients=solve(prepared,epsilon,amplitude)
        assert result.provenance['checkpoint_steps']==block
        assert bool(result.validity.checks['runtime_stability'])
        for field in ('E','H'):parity(result.final_fields.values[field],expected.final_fields.values[field])
        parity(result['e'].values,expected['e'].values)
        for component in ('Ey','Hx'):parity(result['field'].values[component],expected['field'].values[component])
        parity(result['flux'].spectral_energy,expected['flux'].spectral_energy)
        np.testing.assert_allclose(actual_values,values,rtol=5e-13,atol=0.)
        np.testing.assert_allclose(actual_gradients,gradients,rtol=1e-12,atol=0.)
        if saved is None:
            saved=(result,result['e'].values.detach().clone(),result.final_fields.values['E'].detach().clone())
        else:
            assert cuda.equal(saved[0]['e'].values,saved[1])
            assert cuda.equal(saved[0].final_fields.values['E'],saved[2])


def test_checkpoint_preserves_invalid_live_material_guard(cuda):
    n=fixed_numerics(13).replace(checkpoint_steps=5)
    prepared=fdtd.prepare(slab_problem(),numerics=n,backend='torch',device='cuda')
    epsilon=cuda.tensor(4.,dtype=cuda.float64,device='cuda',requires_grad=True)
    result=prepared.run(slab_problem(epsilon=epsilon))
    assert not bool(result.validity.checks['runtime_stability'])
    assert cuda.isnan(result['e'].values).all()


@pytest.mark.parametrize('forward_context',['no_grad','inference_mode'])
def test_checkpoint_forward_context_then_live_gradient_reuse(cuda,forward_context):
    """A forward-only use must neither retain inference state nor change later AD."""
    base=slab_problem();numerics=fixed_numerics(163)
    plain=fdtd.prepare(base,numerics=numerics,backend='torch',device='cuda:0')
    scheduled=fdtd.prepare(base,numerics=numerics.replace(checkpoint_steps=11),
                           backend='torch',device='cuda:0')

    def arrays(result):
        return (result.final_fields.values['E'],result.final_fields.values['H'],
                result['e'].values,result['field'].values['Ey'],result['field'].values['Hx'],
                result['flux'].spectral_energy,result.source_references['pw']['incident'],
                result.source_references['pw']['incident_H'])

    # Live leaves created outside the disabled-grad context remain caller-owned.
    epsilon=cuda.tensor(2.25,dtype=cuda.float64,device='cuda:0',requires_grad=True)
    amplitude=cuda.tensor(1.2,dtype=cuda.float64,device='cpu',requires_grad=True)
    input_copies=tuple(x.detach().clone() for x in (epsilon,amplitude))
    problem=slab_problem(epsilon=epsilon,amplitude=amplitude)
    with getattr(cuda,forward_context)():
        expected=plain.run(problem)
        actual=scheduled.run(problem)
    assert actual.provenance['checkpoint_steps']==11
    for result in (expected,actual):
        assert bool(result.validity.checks['runtime_stability'])
        assert all(not x.requires_grad for x in arrays(result))
    for got,want in zip(arrays(actual),arrays(expected)):parity(got,want)
    old_arrays=arrays(expected)+arrays(actual)
    old_copies=tuple(x.detach().clone() for x in old_arrays)

    def differentiated(prepared):
        eps=cuda.tensor(2.4,dtype=cuda.float64,device='cuda:0',requires_grad=True)
        amp=cuda.tensor(.8,dtype=cuda.float64,device='cpu',requires_grad=True)
        copies=tuple(x.detach().clone() for x in (eps,amp))
        result=prepared.run(slab_problem(epsilon=eps,amplitude=amp))
        assert bool(result.validity.checks['runtime_stability'])
        losses=((result.final_fields.values['E']**2).sum(),)+observable_losses(result,base.units)
        derivatives=[]
        for index,loss in enumerate(losses):
            pair=cuda.autograd.grad(loss,(eps,amp),retain_graph=index<len(losses)-1)
            assert tuple(g.device for g in pair)==(eps.device,amp.device)
            derivatives.append([float(g.detach().cpu()) for g in pair])
        values=np.array([float(loss.detach().cpu()) for loss in losses])
        derivatives=np.asarray(derivatives)
        assert np.isfinite(values).all() and np.isfinite(derivatives).all()
        assert np.all(np.abs(values)>1e-12)
        np.testing.assert_allclose(derivatives[:,1],2*values/.8,rtol=2e-11,atol=0.)
        for leaf,copy in zip((eps,amp),copies):
            assert leaf.requires_grad and cuda.equal(leaf.detach(),copy)
        return result,values,derivatives

    expected_grad,want_values,want_derivatives=differentiated(plain)
    actual_grad,got_values,got_derivatives=differentiated(scheduled)
    for got,want in zip(arrays(actual_grad),arrays(expected_grad)):parity(got,want)
    np.testing.assert_allclose(got_values,want_values,rtol=5e-13,atol=0.)
    np.testing.assert_allclose(got_derivatives,want_derivatives,rtol=1e-12,atol=0.)
    for old,copy in zip(old_arrays,old_copies):assert cuda.equal(old,copy)
    for leaf,copy in zip((epsilon,amplitude),input_copies):
        assert leaf.requires_grad and leaf.grad is None and cuda.equal(leaf.detach(),copy)
