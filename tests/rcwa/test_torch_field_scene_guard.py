"""Concrete autograd conditions must not disable stale-field detection."""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
import numpy as np
import pytest

VALUES={'thickness':np.array([150.]),'period':np.array([1500.,1400.]),
        'lattice':np.array([[1500.,120.],[0.,1400.]]),'top':np.array(1.2),'bot':np.array(1.4)}


def solver(backend,name,value,algorithm):
    kw=dict(backend=backend,structure=np.array([[[1.,4.,4.,1.]]]),period=(1500.,1400.),
            truncation_order=(1,0),thickness=[150.],wavelength=900.,theta=.1,phi=.23,
            answers='field',layer_algorithm=algorithm)
    if name=='lattice':
        kw.pop('period')
        kw['factorization']='laurent'
    kw[name]=value
    return _legacy_wavebit_rcwa(**kw)


def field(s):
    return s.reconstruct_field(3,2,3)


@pytest.mark.parametrize('name',sorted(VALUES))
@pytest.mark.parametrize('change',['wavelength','incidence_side'])
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_autograd_condition_does_not_hide_a_changed_scene(name,change,algorithm):
    torch=pytest.importorskip('torch')
    leaf=torch.tensor(VALUES[name],dtype=torch.float64,requires_grad=True)
    s=solver('torch',name,leaf,algorithm)
    s.solve(auto_detune=False)
    before=field(s).detach().numpy()
    setattr(s,change,750. if change=='wavelength' else 'bottom')
    with pytest.raises(RuntimeError,match='scene has changed'):
        field(s)
    s.solve(auto_detune=False)
    after=field(s).detach().numpy()
    assert np.max(np.abs(before-after))>1e-3


@pytest.mark.parametrize('name',sorted(VALUES))
@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_scene_snapshot_preserves_the_live_gradient(name,algorithm):
    torch=pytest.importorskip('torch')
    leaf=torch.tensor(VALUES[name],dtype=torch.float64,requires_grad=True)
    s=solver('torch',name,leaf,algorithm)
    s.solve(auto_detune=False)
    def objective(xp,f):return xp.sum(xp.abs(f)**2)+.31*xp.real(xp.sum(f))
    loss=objective(torch,field(s));loss.backward()
    assert leaf.requires_grad and leaf.grad is not None
    direction=np.linspace(.3,.8,leaf.numel()).reshape(VALUES[name].shape)
    actual=np.sum(leaf.grad.numpy()*direction)
    def independent(t):
        reference=solver('numpy',name,VALUES[name]+t*direction,algorithm)
        reference.solve(auto_detune=False)
        return objective(np,field(reference))
    h=1e-4
    expected=(independent(h)-independent(-h))/(2*h)
    assert float(loss.detach())==pytest.approx(independent(0.),rel=2e-11,abs=2e-11)
    assert abs(expected)>1e-7
    assert actual==pytest.approx(expected,rel=3e-5,abs=2e-7)
    # Reassigning the same concrete numbers remains a valid field scene.
    setattr(s,name,leaf.clone())
    assert torch.isfinite(field(s)).all()


@pytest.mark.parametrize('algorithm',['ETM','SMM'])
def test_inplace_condition_update_invalidates_legacy_field_but_not_result(algorithm):
    torch=pytest.importorskip('torch')
    leaf=torch.tensor([150.],dtype=torch.float64,requires_grad=True)
    s=solver('torch','thickness',leaf,algorithm)
    result=s.solve(auto_detune=False)
    before=result.reconstruct_field([21.],[-17.],[63.]).detach().numpy()
    with torch.no_grad():s.thickness.add_(20.)
    with pytest.raises(RuntimeError,match='scene has changed'):
        field(s)
    np.testing.assert_allclose(result.reconstruct_field([21.],[-17.],[63.]).detach().numpy(),before,atol=2e-12,rtol=2e-12)
