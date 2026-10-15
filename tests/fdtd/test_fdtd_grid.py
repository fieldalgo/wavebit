"""Independent Yee-mode dispersion, streaming parity and material sensitivities."""

from wavebit.fdtd._periodic.grid import fdtd_grid as _legacy_wavebit_fdtd_grid
import numpy as np
import pytest



def initial_mode(nz=16, dt=.2, eps=2., dz=1.):
    # Exact forward eigenmode of the discrete staggered update, not continuum
    # exp(-i*k*t): sin(omega*dt/2) = dt*sin(k*dz/2)/(sqrt(eps)*dz).
    k = 2*np.pi/(nz*dz)
    omega = 2*np.arcsin(dt*np.sin(k*dz/2)/(np.sqrt(eps)*dz))/dt
    e = np.zeros((3,nz,2,2), complex)
    h = np.zeros_like(e)
    e[1] = np.exp(1j*k*np.arange(nz)*dz)[:,None,None]
    h[0] = -np.sqrt(eps)*np.exp(1j*k*(np.arange(nz)+.5)*dz+1j*omega*dt/2)[:,None,None]
    return e, h, omega


def test_numpy_grid_propagates_an_exact_discrete_maxwell_eigenmode():
    e,h,w = initial_mode()
    out = _legacy_wavebit_fdtd_grid(np.full((16,2,2),2.), spacing=(1.,1.,1.),dt=.2,steps=71,
                           electric=e, magnetic=h, wavelengths=[2*np.pi/w], monitor_z=[3,8])
    np.testing.assert_allclose(out['E'], e*np.exp(-1j*w*71*.2), atol=2e-14)
    np.testing.assert_allclose(out['H'], h*np.exp(-1j*w*71*.2), atol=2e-14)
    assert np.all(out['power_z']>0)


def test_jax_compiled_checkpointed_scan_matches_numpy_and_is_differentiable():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    e,h,_=initial_mode()
    common=dict(spacing=(1.,1.,1.),dt=.2,steps=43,electric=e,magnetic=h,
                wavelengths=[20.,24.], monitor_z=[3], conductivity=.015)
    reference=_legacy_wavebit_fdtd_grid(np.full((16,2,2),2.), **common)
    def run(eps, block=7):
        return _legacy_wavebit_fdtd_grid(eps, **common, backend='jax',checkpoint_steps=block)
    compiled=jax.jit(run)(jnp.full((16,2,2),2.))
    for key in ('E','H','electric_spectrum','magnetic_spectrum','power_z'):
        np.testing.assert_allclose(compiled[key],reference[key],atol=2e-12)
    def objective(value, block=7):
        result=run(jnp.full((16,2,2),2.)+value*jnp.arange(16)[:,None,None]/16,block)
        return jnp.sum(jnp.abs(result['electric_spectrum'])**2)
    exact=float(jax.grad(objective)(.03))
    delta=1e-5
    finite=float((objective(.03+delta)-objective(.03-delta))/(2*delta))
    assert abs(exact)>1e-3
    np.testing.assert_allclose(exact,finite,rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(jax.grad(lambda v:objective(v,43))(.03),exact,atol=2e-10)


def test_passive_conductivity_damps_an_initial_field():
    e,h,_=initial_mode()
    kw=dict(spacing=(1.,1.,1.),dt=.2,steps=80,electric=e,magnetic=h)
    a=_legacy_wavebit_fdtd_grid(np.full((16,2,2),2.),**kw)
    b=_legacy_wavebit_fdtd_grid(np.full((16,2,2),2.),conductivity=.3,**kw)
    assert np.sum(abs(b['E'])**2)+np.sum(abs(b['H'])**2)<.2*(np.sum(abs(a['E'])**2)+np.sum(abs(a['H'])**2))


def test_current_pulse_drives_fields_and_invalid_cfl_is_refused():
    source=np.zeros((3,8,2,2));source[1,3]=1
    result=_legacy_wavebit_fdtd_grid(np.ones((8,2,2)),spacing=(1.,1.,1.),dt=.2,steps=12,
                            source=source,pulse=np.sin(np.arange(12)*.2))
    assert np.max(abs(result['E']))>.1
    with pytest.raises(ValueError,match='CFL'):
        _legacy_wavebit_fdtd_grid(np.ones((8,2,2)),spacing=(1.,1.,1.),dt=1.,steps=12)


def test_absorber_removes_a_pulse_before_periodic_recirculation():
    nz=80
    z=np.arange(nz)
    e=np.zeros((3,nz,2,2));h=np.zeros_like(e)
    e[1]=np.exp(-((z-40)/6)**2)[:,None,None]
    h[0]=-np.exp(-((z+.5-40+.1)/6)**2)[:,None,None]
    kw=dict(spacing=(1.,1.,1.),dt=.2,steps=400,electric=e,magnetic=h)
    bare=_legacy_wavebit_fdtd_grid(np.ones((nz,2,2)),**kw)
    damped=_legacy_wavebit_fdtd_grid(np.ones((nz,2,2)),**kw,absorber_cells=18)
    def energy(out):
        return np.sum(abs(out['E'])**2)+np.sum(abs(out['H'])**2)
    assert energy(damped)<.03*energy(bare)


def test_jax_pulse_gradient_and_explicit_cpu_device():
    jax=pytest.importorskip('jax')
    import jax.numpy as jnp
    source=np.zeros((3,8,2,2));source[1,3]=1
    pulse=jnp.sin(jnp.arange(12)*.2)
    def objective(scale):
        out=_legacy_wavebit_fdtd_grid(np.ones((8,2,2)),spacing=(1.,1.,1.),dt=.2,steps=12,
                             source=source,pulse=scale*pulse,backend='jax',device='cpu')
        return jnp.sum(abs(out['E'])**2)
    value=objective(1.)
    np.testing.assert_allclose(jax.grad(objective)(1.),2*value,rtol=1e-6)
    with pytest.raises((ValueError, RuntimeError)):
        _legacy_wavebit_fdtd_grid(np.ones((8,2,2)),spacing=(1.,1.,1.),dt=.2,steps=12,
                         backend='jax',device='nonexistent_device')


@pytest.mark.parametrize('axis',[0,1,2])
def test_diagonal_material_and_magnetic_yee_modes_on_each_axis(axis):
    # Physical axes x/y/z map to array axes 2/1/0. A cyclic (k,E,H)
    # basis checks curl signs and spacing independently on every axis.
    size=16
    shape=[2,2,2];shape[2-axis]=size
    eps_value=2.;mu=3.;dt=.2;steps=37
    spacing=np.array([.8,1.,1.2]);delta=spacing[2-axis]
    k=2*np.pi/(size*delta)
    omega=2*np.arcsin(dt*np.sin(k*delta/2)/(np.sqrt(eps_value*mu)*delta))/dt
    coordinate_shape=[1,1,1];coordinate_shape[2-axis]=size
    z=(np.arange(size)*delta).reshape(coordinate_shape)
    e=np.zeros((3,)+tuple(shape),complex);h=np.zeros_like(e)
    ec=(axis+1)%3;hc=(axis+2)%3
    e[ec]=np.exp(1j*k*z)
    h[hc]=np.sqrt(eps_value/mu)*np.exp(1j*k*(z+.5*delta)+1j*omega*dt/2)
    epsilon=np.ones(e.shape);epsilon[ec]=eps_value
    out=_legacy_wavebit_fdtd_grid(epsilon,mu=mu,spacing=spacing,dt=dt,steps=steps,electric=e,magnetic=h)
    np.testing.assert_allclose(out['E'],e*np.exp(-1j*omega*dt*steps),atol=2e-14)
    np.testing.assert_allclose(out['H'],h*np.exp(-1j*omega*dt*steps),atol=2e-14)
