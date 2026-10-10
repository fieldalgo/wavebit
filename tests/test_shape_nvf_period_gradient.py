"""A readable Torch period still owns the shape NVF geometry derivative."""
import numpy as np
import pytest
import wavebit


def solver(backend,shape,period=(600.,750.)):
    return wavebit.rcwa(backend=backend,period=period,wavelength=900.,thickness=[170.],
        truncation_order=(1,1),theta=.17,phi=.29,polarization=(.8,.6j),answers='field',
        factorization='nvf',shape_nvf_grid_size=16,structure=[dict(background=1.,shapes=[
        dict(type=shape,lx=130.,ly=180.,cx=23.,cy=-17.,material=4.+.07j)])])


def loss(s,xp,period,geometry=None):
    r=s.solve(period=period,geometry=geometry,auto_detune=False)
    f=r.reconstruct_field([27.],[-35.],[63.])
    return r.T.sum().real+.07*xp.sum(xp.abs(f)**2)+.11*xp.real(xp.sum(f))


@pytest.mark.parametrize('shape',['rectangle','ellipse'])
@pytest.mark.parametrize('frozen',[False,True])
def test_both_period_components_reach_the_shape_nvf_builder(shape,frozen):
    torch=pytest.importorskip('torch')
    s=solver('torch',shape)
    geometry=s.freeze_geometry() if frozen else None
    p=torch.tensor([600.,750.],dtype=torch.float64,requires_grad=True)
    result=loss(s,torch,p,geometry);result.backward()
    reference=loss(solver('numpy',shape),np,np.array([600.,750.]))
    assert float(result.detach())==pytest.approx(reference,abs=2e-11)
    expected=[]
    for j in (0,1):
        delta=np.zeros(2);delta[j]=.01
        plus=np.array([600.,750.])+delta;minus=np.array([600.,750.])-delta
        expected.append((loss(solver('numpy',shape,plus),np,plus)-loss(solver('numpy',shape,minus),np,minus))/.02)
    assert np.min(np.abs(expected))>1e-7
    np.testing.assert_allclose(p.grad.numpy(),expected,atol=2e-9,rtol=2e-5)
