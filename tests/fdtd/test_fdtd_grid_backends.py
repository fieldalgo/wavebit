"""Single-source Yee core across the RCWA Array API backends."""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest

import wavebit


def case():
    shape=(20,4,5)
    rng=np.random.default_rng(811)
    mask=np.zeros((3,)+shape);mask[:,8:12]=rng.uniform(.2,.8,size=(3,4,4,5))
    source=np.zeros((3,)+shape,complex);source[0,6]=1
    kw=dict(spacing=(.5,)*3,dt=.1,steps=41,pml_cells=3,pml_wavelength=3.,
            source=source,pulse=np.exp(-((np.arange(41)*.1-1)/.4)**2)*np.exp(-2j*np.arange(41)*.1),
            wavelengths=[3.,4.,5.],monitor_z=[0,14])
    return mask,kw


def test_torch_common_core_matches_numpy_and_preserves_all_parameter_gradients():
    torch=pytest.importorskip('torch')
    mask,kw=case()
    parameters=torch.tensor([.2,1.2,.3,.7,.2,-.1],dtype=torch.float64,requires_grad=True)
    def run(p,backend='torch',block=7):
        if backend=='torch':
            m=torch.as_tensor(mask,dtype=p.dtype)
            phase=torch.stack((p[0]*0,p[4],p[5]))
        else:
            m=mask
            phase=np.array([0.,p[4],p[5]])
        return _legacy_wavebit_fdtd_grid(1+p[0]*m,poles=[(p[1],p[2],p[3]*m)],
                                bloch_phase=phase,backend=backend,checkpoint_steps=block,**kw)
    actual=run(parameters)
    expected=run(parameters.detach().numpy(),'numpy')
    for key in ('E','H','electric_spectrum','magnetic_spectrum','power_z'):
        np.testing.assert_allclose(actual[key].detach(),expected[key],rtol=1e-11,atol=1e-12)
    def objective(p,block=7):
        return run(p,block=block)['power_z'].sum()
    derivative=torch.autograd.grad(actual['power_z'].sum(),parameters)[0]
    assert parameters.requires_grad
    delta=1e-5
    for i in range(6):
        perturb=torch.eye(6,dtype=parameters.dtype)[i]*delta
        with torch.no_grad():
            finite=(objective(parameters+perturb)-objective(parameters-perturb))/(2*delta)
        np.testing.assert_allclose(derivative[i].detach(),finite,rtol=2e-5,atol=1e-10)
    other=torch.autograd.grad(objective(parameters,41),parameters)[0]
    np.testing.assert_allclose(derivative.detach(),other.detach(),rtol=1e-11,atol=1e-12)


def test_torch_initial_field_and_pulse_gradients_and_input_immutability():
    torch=pytest.importorskip('torch')
    initial=torch.randn((3,8,3,4),dtype=torch.float64,requires_grad=True)
    pulse=torch.linspace(0,1,13,dtype=torch.float64,requires_grad=True)
    saved=initial.detach().clone()
    source=torch.ones_like(initial)
    result=_legacy_wavebit_fdtd_grid(torch.ones((8,3,4),dtype=torch.float64),spacing=(1.,)*3,
                            dt=.2,steps=13,electric=initial,source=source,pulse=pulse,backend='torch')
    loss=(result['E'].abs()**2).sum()
    de,dp=torch.autograd.grad(loss,(initial,pulse))
    assert torch.isfinite(de).all() and torch.isfinite(dp).all()
    assert de.abs().max()>0 and dp.abs().max()>0
    torch.testing.assert_close(initial.detach(),saved)


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch',
                                      wavebit.Backend.NUMPY, wavebit.Backend.JAX,
                                      wavebit.Backend.TORCH])
def test_named_backends_and_enums_match_rcwa(backend):
    if backend == 'jax':
        pytest.importorskip('jax')
    if backend == 'torch':
        pytest.importorskip('torch')
    out = _legacy_wavebit_fdtd_grid(np.ones((4, 3, 2)), spacing=(1.,)*3, dt=.1, steps=2,
                            backend=backend)
    assert tuple(out['E'].shape) == (3, 4, 3, 2)


@pytest.mark.parametrize('backend', [0, 1, 2, 'JAX', 'pytorch'])
def test_retired_backend_spellings_are_refused(backend):
    with pytest.raises(ValueError, match='backend'):
        _legacy_wavebit_fdtd_grid(np.ones((4, 3, 2)), spacing=(1.,)*3, dt=.1, steps=2,
                          backend=backend)


def test_torch_eager_validation_reads_live_values_without_detaching_inputs():
    torch=pytest.importorskip('torch')
    eps=torch.full((4,3,2),-1.,requires_grad=True)
    with pytest.raises(ValueError):
        _legacy_wavebit_fdtd_grid(eps,spacing=(1.,)*3,dt=.1,steps=2,backend='torch')
    assert eps.requires_grad


def test_tensor_phase_sequence_preserves_graph_and_matches_stacked_input():
    torch=pytest.importorskip('torch')
    angle=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    source=torch.ones((3,5,3,4),dtype=torch.complex128)
    kw=dict(spacing=(1.,)*3,dt=.2,steps=9,source=source,pulse=torch.ones(9),backend='torch')
    def objective(phase):
        return _legacy_wavebit_fdtd_grid(torch.ones((5,3,4),dtype=torch.float64),bloch_phase=phase,**kw)['E'].abs().square().sum()
    a=objective((0.,angle,-angle))
    b=objective(torch.stack((angle*0,angle,-angle)))
    ga=torch.autograd.grad(a,angle)[0]
    gb=torch.autograd.grad(b,angle)[0]
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(ga,gb)
    assert ga.abs()>0
