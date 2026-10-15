"""Dispersive displacement conservation, CPML and differentiable pole parameters."""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest



def test_uniform_ade_conserves_displacement_without_free_current():
    shape=(3,4,3,2)
    electric=np.ones(shape)
    out=_legacy_wavebit_fdtd_grid(np.full(shape,2.),spacing=(1.,)*3,dt=.1,steps=121,
                         electric=electric,poles=((0.,.5,2.),(2.,.3,3.)))
    np.testing.assert_allclose(2*out['E']+sum(out['polarization']),2*electric,atol=2e-14)
    assert np.max(abs(out['H']))<1e-14
    assert np.max(abs(out['polarization'][0]))>.1


def test_cpml_reduces_late_pulse_energy_and_numpy_jax_agree():
    jax=pytest.importorskip('jax')
    jax.config.update('jax_enable_x64',True)
    shape=(100,2,2)
    source=np.zeros((3,)+shape);source[0,50]=1
    dt=.04;steps=600
    t=(np.arange(steps)+.5)*dt
    pulse=np.exp(-((t-2)/.5)**2)*np.sin(2*np.pi*t)
    common=dict(spacing=(.1,)*3,dt=dt,steps=steps,source=source,pulse=pulse)
    periodic=_legacy_wavebit_fdtd_grid(np.ones(shape),**common)
    pml=_legacy_wavebit_fdtd_grid(np.ones(shape),**common,pml_cells=15)
    compiled=_legacy_wavebit_fdtd_grid(np.ones(shape),**common,pml_cells=15,backend='jax')
    for key in ('E','H'):
        np.testing.assert_allclose(compiled[key],pml[key],atol=1e-13)
    def energy(out):
        return np.sum(abs(out['E'])**2)+np.sum(abs(out['H'])**2)
    assert energy(pml)<1e-5*energy(periodic)


def test_cpml_dispersive_spectrum_and_pole_gradients_match_finite_differences():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    shape=(48,2,2)
    mask=np.zeros(shape);mask[20:28]=1
    source=np.zeros((3,)+shape);source[0,12]=1
    dt=.05;steps=200;t=(np.arange(steps)+.5)*dt
    kw=dict(spacing=(.2,)*3,dt=dt,steps=steps,source=source,
            pulse=np.exp(-((t-2)/.5)**2)*np.sin(3*t),
            pml_cells=8,pml_wavelength=2.4,wavelengths=[2.4],monitor_z=[32])
    def run(parameters,backend='jax',block=13):
        return _legacy_wavebit_fdtd_grid(np.ones(shape),poles=[(parameters[0],parameters[1],parameters[2]*mask)],
                                backend=backend,checkpoint_steps=block,**kw)
    def objective(parameters,block=13):
        return run(parameters,block=block)['power_z'].sum()
    parameters=jnp.array([3.,.5,4.])
    compiled=jax.jit(run)(parameters)
    reference=run(np.asarray(parameters),'numpy')
    for key in ('E','H','electric_spectrum','magnetic_spectrum','power_z'):
        np.testing.assert_allclose(compiled[key],reference[key],atol=1e-13)
    gradient=jax.jit(jax.grad(objective))(parameters)
    delta=1e-4
    for i in range(3):
        perturb=np.eye(3)[i]*delta
        finite=(objective(parameters+perturb)-objective(parameters-perturb))/(2*delta)
        assert abs(float(gradient[i]))>1e-9
        np.testing.assert_allclose(gradient[i],finite,rtol=2e-5,atol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda p:objective(p,200))(parameters),gradient,atol=1e-12)


@pytest.mark.parametrize('extra',[
    {'poles':[(1.,-.1,2.)]}, {'poles':[(100.,0.,2.)]},
    {'poles':[(1.,.1,-2.)]}, {'pml_cells':2,'absorber_cells':2},
    {'pml_cells':4}, {'pml_cells':2,'poles':[(1.,.1,2.)]},
])
def test_invalid_poles_and_cpml_combinations_are_refused(extra):
    with pytest.raises(ValueError):
        _legacy_wavebit_fdtd_grid(np.ones((8,2,2)),spacing=(1.,)*3,dt=.2,steps=10,**extra)


def test_cpml_absorbs_oblique_transverse_fourier_mode():
    jax=pytest.importorskip('jax')
    jax.config.update('jax_enable_x64',True)
    shape=(100,2,32)
    source=np.zeros((3,)+shape,complex)
    source[1,50]=np.exp(2j*np.pi*np.arange(32)/32)[None,:]
    dt=.04;steps=900;t=(np.arange(steps)+.5)*dt
    pulse=np.exp(-((t-6)/1.5)**2)*np.exp(-2j*np.pi*t)
    common=dict(spacing=(.1,)*3,dt=dt,steps=steps,source=source,pulse=pulse,backend='jax')
    periodic=_legacy_wavebit_fdtd_grid(np.ones(shape),**common)
    pml=_legacy_wavebit_fdtd_grid(np.ones(shape),**common,pml_cells=15)
    def energy(out):
        return np.sum(abs(np.asarray(out['E']))**2)+np.sum(abs(np.asarray(out['H']))**2)
    assert energy(pml)<1e-4*energy(periodic)
