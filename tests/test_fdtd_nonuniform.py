"""Nonuniform Yee meshes: independent eigenproblem, CPML parity and sensitivities."""
import numpy as np
import pytest
import wavebit


def eigenmode():
    widths=.8+.2*np.sin(np.arange(14)*2*np.pi/14)
    dual=(widths+np.roll(widths,1))/2
    incidence=np.roll(np.eye(14),1,axis=1)-np.eye(14)
    mass=2*dual
    stiffness=incidence.T@np.diag(1/widths)@incidence
    values,vectors=np.linalg.eigh(stiffness/np.sqrt(mass[:,None]*mass[None,:]))
    rate=values[3];electric=vectors[:,3]/np.sqrt(mass)
    dt=.15;omega=2*np.arcsin(dt*np.sqrt(rate)/2)/dt
    magnetic=1j*(incidence@electric)/widths/np.sqrt(rate)*np.exp(1j*omega*dt/2)
    e=np.zeros((3,14,2,2),complex);h=np.zeros_like(e)
    e[1]=electric[:,None,None];h[0]=magnetic[:,None,None]
    return widths,e,h,dt,omega


@pytest.mark.parametrize('backend',['numpy','jax','torch'])
def test_nonuniform_grid_matches_independent_discrete_eigenproblem(backend):
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    if backend=='torch':
        pytest.importorskip('torch')
    widths,e,h,dt,omega=eigenmode()
    result=wavebit.fdtd_grid(np.full((14,2,2),2.),spacing=(1.,)*3,z_spacing=widths,
                             dt=dt,steps=41,electric=e,magnetic=h,backend=backend,
                             record_field=7,wavelengths=[2*np.pi/omega],monitor_z=[3])
    host=lambda x:x.detach().numpy() if backend=='torch' else np.asarray(x)
    np.testing.assert_allclose(host(result['E']),e*np.exp(-1j*omega*41*dt),atol=2e-14)
    np.testing.assert_allclose(host(result['H']),h*np.exp(-1j*omega*41*dt),atol=2e-14)
    expected_z=np.r_[0.,np.cumsum(widths[:-1])]+widths/2
    np.testing.assert_allclose(host(result['fields']['hx']['z']),expected_z)
    # Collocation is linear in physical distance, not the average of neighbours.
    weight=widths[2]/(widths[2]+widths[3])
    sample=weight*h[0,3]+(1-weight)*h[0,2]
    times=(np.arange(41)+.5)*dt
    offline=np.sum(np.exp(-1j*omega*(np.arange(41)+1)*dt+1j*omega*times))*dt*sample
    np.testing.assert_allclose(host(result['magnetic_spectrum'])[0,0,0],offline,atol=2e-14)


def case():
    widths=np.r_[.6,.55,.5,.45,.4,.4,.4,.4,.4,.4,.4,.4,.45,.5,.55,.6]
    mask=np.zeros((16,2,3));mask[6:10]=1
    source=np.zeros((3,16,2,3));source[1,5]=1
    kw=dict(spacing=(.4,1.,1.),dt=.1,steps=61,pml_cells=3,z_spacing=widths,
            source=source,pulse=np.exp(-((np.arange(61)*.1-1)/.3)**2),
            wavelengths=[2.],monitor_z=[11],record_field=8)
    return mask,kw


def test_explicit_uniform_mesh_matches_existing_cpml_and_cfl_uses_smallest_cell():
    mask,kw=case();kw.pop('z_spacing')
    a=wavebit.fdtd_grid(1+mask,**kw)
    b=wavebit.fdtd_grid(1+mask,z_spacing=np.full(16,.4),**kw)
    for key in ('E','H','electric_spectrum','power_z'):
        np.testing.assert_allclose(a[key],b[key],atol=1e-13)
    widths=np.full(16,.4);widths[7]=.01
    with pytest.raises(ValueError,match='CFL'):
        wavebit.fdtd_grid(1+mask,z_spacing=widths,**kw)


@pytest.mark.parametrize('backend',['jax','torch'])
def test_nonuniform_cpml_ade_material_gradients(backend):
    mask,kw=case()
    def run(v,which):
        m=mask if which=='numpy' else native_mask
        result=wavebit.fdtd_grid(1+v*m,poles=[(2.,.2,.3*m)],backend=which,**kw)
        return result['E'],(abs(result['fields']['ey']['values'])**2).sum()
    if backend=='jax':
        jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        native_mask=jnp.asarray(mask)
        field,value=jax.jit(lambda v:run(v,'jax'))(.3)
        derivative=jax.jit(jax.grad(lambda v:run(v,'jax')[1]))(.3)
        host=np.asarray
    else:
        torch=pytest.importorskip('torch')
        native_mask=torch.as_tensor(mask,dtype=torch.float64)
        parameter=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        field,value=run(parameter,'torch')
        derivative=torch.autograd.grad(value,parameter)[0].detach().numpy()
        host=lambda x:x.detach().numpy()
    expected,expected_value=run(.3,'numpy')
    np.testing.assert_allclose(host(field),expected,atol=1e-12)
    np.testing.assert_allclose(host(value),expected_value,atol=1e-12)
    delta=1e-5;finite=(run(.3+delta,'numpy')[1]-run(.3-delta,'numpy')[1])/(2*delta)
    np.testing.assert_allclose(derivative,finite,rtol=1e-6,atol=1e-8)
    assert abs(finite)>1e-4


def test_nonuniform_absorber_removes_outgoing_pulse():
    widths=1.+.2*np.cos(np.linspace(-np.pi,np.pi,48))
    z=np.r_[0.,np.cumsum(widths[:-1])];centre=z[24]
    e=np.zeros((3,48,2,2));h=np.zeros_like(e)
    e[1]=np.exp(-((z-centre)/3.)**2)[:,None,None]
    h[0]=-np.exp(-((z+widths/2-centre+.1)/3.)**2)[:,None,None]
    kw=dict(spacing=(1.,)*3,z_spacing=widths,dt=.2,steps=400,electric=e,magnetic=h)
    bare=wavebit.fdtd_grid(np.ones((48,2,2)),**kw)
    absorbed=wavebit.fdtd_grid(np.ones((48,2,2)),absorber_cells=8,**kw)
    energy=lambda r:np.sum(abs(r['E'])**2+abs(r['H'])**2)
    assert energy(absorbed)<.03*energy(bare)


@pytest.mark.parametrize('widths',[np.ones(3),np.zeros(14),np.full(14,np.nan),np.full(14,1.+1j)])
def test_invalid_nonuniform_geometry(widths):
    with pytest.raises(ValueError,match='z_spacing'):
        wavebit.fdtd_grid(np.ones((14,2,2)),spacing=(1.,)*3,z_spacing=widths,dt=.1,steps=2)


def test_unequal_physical_absorber_widths_do_not_use_domain_midpoint():
    # Left absorber reaches z=16, beyond the domain midpoint at z=12.
    widths=np.array([8.,8.]+[1.]*8)
    fields=np.ones((3,10,2,2))
    result=wavebit.fdtd_grid(np.ones((10,2,2)),spacing=(1.,)*3,z_spacing=widths,
                            dt=.1,steps=1,electric=fields,magnetic=fields,absorber_cells=2)
    rate=6*(1-12/16)**3/16
    np.testing.assert_allclose(result['H'][0,1],(1-.05*rate)/(1+.05*rate),atol=1e-15)
