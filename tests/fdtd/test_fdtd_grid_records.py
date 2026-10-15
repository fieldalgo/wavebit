"""Sampled histories: independent discrete eigenmode and streaming DFT checks."""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest
from tests.fdtd.test_fdtd_grid import initial_mode


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('interval', [1, 6, 50])
def test_records_preserve_exact_modes_staggering_and_final_state(backend, interval):
    if backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
    if backend == 'torch':
        pytest.importorskip('torch')
    e, h, omega = initial_mode()
    out = _legacy_wavebit_fdtd_grid(np.full((16,2,2),2.), spacing=(1.,)*3, dt=.2, steps=20,
                            electric=e, magnetic=h, record_field=interval,
                            checkpoint_steps=4, backend=backend)
    def host(v):
        return v.detach().numpy() if backend == 'torch' else np.asarray(v)
    steps = np.arange(0,20,interval)+1
    for name, initial in [('ey',e[1]), ('hx',h[0])]:
        record=out['fields'][name]
        np.testing.assert_allclose(host(record['values']),
                                   initial[None]*np.exp(-1j*omega*steps*.2)[:,None,None,None],atol=2e-14)
        np.testing.assert_allclose(host(record['t']), steps*.2-(.1 if name=='hx' else 0))
        np.testing.assert_allclose(host(record['z']), np.arange(16)+(.5 if name=='hx' else 0))
    np.testing.assert_allclose(host(out['E']), e*np.exp(-1j*omega*20*.2),atol=2e-14)


def test_streamed_planes_equal_offline_record_dft():
    e,h,w=initial_mode()
    wl=2*np.pi/w
    out=_legacy_wavebit_fdtd_grid(np.full((16,2,2),2.),spacing=(1.,)*3,dt=.2,steps=21,
                          electric=e,magnetic=h,record_field=1,wavelengths=[wl],monitor_z=[4])
    for name,key,c in [('ey','electric_spectrum',1),('hx','magnetic_spectrum',0)]:
        rec=out['fields'][name]
        plane=rec['values'][:,4]
        if name=='hx':
            plane=.5*(plane+rec['values'][:,3])
        offline=np.sum(plane*np.exp(2j*np.pi*rec['t']/wl)[:,None,None],axis=0)*.2
        np.testing.assert_allclose(offline,out[key][0,0,c],atol=1e-13)


@pytest.mark.parametrize('backend',['jax','torch'])
def test_recorded_material_gradient_matches_finite_difference(backend):
    e,h,_=initial_mode()
    def run(value,which):
        out=_legacy_wavebit_fdtd_grid(value*np.ones((16,2,2)) if which=='numpy' else value*ones,
                              spacing=(1.,)*3,dt=.2,steps=15,electric=e,magnetic=h,
                              backend=which,checkpoint_steps=3,record_field=4,record_components=['ey'])
        return (abs(out['fields']['ey']['values'])**2).sum()
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        ones=jnp.ones((16,2,2))
        derivative=jax.jit(jax.grad(lambda v:run(v,'jax')))(2.)
    else:
        torch=pytest.importorskip('torch')
        ones=torch.ones((16,2,2),dtype=torch.float64)
        value=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        derivative=torch.autograd.grad(run(value,'torch'),value)[0].detach().numpy()
    delta=1e-5
    finite=(run(2.+delta,'numpy')-run(2.-delta,'numpy'))/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=1e-6,atol=1e-7)
    assert abs(finite)>1
