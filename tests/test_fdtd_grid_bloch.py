"""Bloch wrapping and native Yee phasing verified against discrete Maxwell modes."""
import numpy as np
import pytest

import wavebit


def test_arbitrary_three_axis_bloch_eigenmode_and_boundary_monitor():
    shape=(9,7,5);spacing=np.array([.8,1.,1.2]);dt=.2;steps=51
    k=np.array([.31,-.22,.43])  # physical x,y,z
    kd=k*spacing[::-1]
    q=2*np.sin(kd/2)/spacing[::-1]
    omega=2*np.arcsin(dt*np.linalg.norm(q)/(2*np.sqrt(2)))/dt
    amplitude=np.cross(q,[1.,.2,.4]);amplitude/=np.linalg.norm(amplitude)
    magnetic=np.cross(q,amplitude)/(2*np.sin(omega*dt/2)/dt)
    coords=np.meshgrid(*[np.arange(n)*d for n,d in zip(shape,spacing)],indexing='ij')
    e=np.empty((3,)+shape,complex);h=np.empty_like(e)
    for component in range(3):
        e_offset=np.eye(3)[component]*.5
        h_offset=(1-np.eye(3)[component])*.5
        for value,offset,amp,time in ((e,e_offset,amplitude[component],0.),
                                      (h,h_offset,magnetic[component],dt/2)):
            exponent=sum(k[2-a]*(coords[a]+offset[2-a]*spacing[a]) for a in range(3))
            value[component]=amp*np.exp(1j*exponent+1j*omega*time)
    phase=k[::-1]*spacing*np.array(shape)
    out=wavebit.fdtd_grid(np.full(shape,2.),spacing=spacing,dt=dt,steps=steps,
                         electric=e,magnetic=h,bloch_phase=phase,
                         wavelengths=[2*np.pi/omega],monitor_z=[0,3])
    np.testing.assert_allclose(out['E'],e*np.exp(-1j*omega*dt*steps),atol=2e-14)
    np.testing.assert_allclose(out['H'],h*np.exp(-1j*omega*dt*steps),atol=2e-14)
    # At resonance the DFT is exactly steps*dt times the co-located spatial mode.
    expected_e=amplitude*np.cos(kd/2)
    expected_h=magnetic*np.array([np.prod(np.cos(np.delete(kd,i)/2)) for i in range(3)])
    np.testing.assert_allclose(out['electric_spectrum'][0,0,:,0,0],expected_e*steps*dt,atol=2e-13)
    np.testing.assert_allclose(out['magnetic_spectrum'][0,0,:,0,0],expected_h*steps*dt,atol=2e-13)


def test_bloch_phase_gradient_and_numpy_jax_parity():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(19)
    electric=rng.normal(size=(3,7,5,4))+1j*rng.normal(size=(3,7,5,4))
    kw=dict(spacing=(1.,)*3,dt=.2,steps=23,electric=electric,wavelengths=[4.],monitor_z=[0])
    phase=jnp.array([.3,-.2,.4])
    def run(p,backend='jax'):
        return wavebit.fdtd_grid(np.ones((7,5,4)),bloch_phase=p,backend=backend,**kw)
    a=jax.jit(run)(phase);b=run(np.asarray(phase),'numpy')
    for key in ('E','H','electric_spectrum','magnetic_spectrum'):
        np.testing.assert_allclose(a[key],b[key],atol=2e-13)
    def objective(p):
        return jnp.sum(abs(run(p)['electric_spectrum'])**2)
    gradient=jax.grad(objective)(phase)
    for i in range(3):
        delta=np.eye(3)[i]*1e-5
        finite=(objective(phase+delta)-objective(phase-delta))/2e-5
        np.testing.assert_allclose(gradient[i],finite,rtol=1e-6,atol=1e-8)


def test_cpml_requires_zero_longitudinal_bloch_phase():
    with pytest.raises(ValueError):
        wavebit.fdtd_grid(np.ones((12,3,3)),spacing=(1.,)*3,dt=.2,steps=10,
                         pml_cells=2,bloch_phase=(.1,0.,0.))


def test_explicit_single_precision_scan_with_x64_enabled_and_cpml_monitor():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    source=jnp.zeros((3,16,3,3),dtype=jnp.complex64).at[0,7].set(1)
    def run(eps):
        return wavebit.fdtd_grid(eps,mu=jnp.float32(1),conductivity=jnp.float32(0),
                                bloch_phase=jnp.zeros(3,dtype=jnp.float32),
                                spacing=(1.,)*3,dt=.2,steps=20,pml_cells=3,
                                source=source,pulse=jnp.sin(jnp.arange(20)*.2),
                                wavelengths=[4.],monitor_z=[9],backend='jax')
    out=jax.jit(run)(jnp.ones((16,3,3),dtype=jnp.float32))
    assert out['E'].dtype==jnp.complex64
    assert np.isfinite(np.asarray(out['electric_spectrum'])).all()
    assert np.max(abs(np.asarray(out['E'])))>0
