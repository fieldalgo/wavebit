"""Periodic physical boundaries preserve independent optics, port phases and AD."""
import numpy as np
import pytest
from wavebit import Units, Material, Job, run, Drude, Lorentz
from wavebit.fdtd import periodic as p
from wavebit.units import C0, Z0
from wavebit.fdtd._periodic.adapter import solve_periodic_fdtd


def case(algorithm='shared',length='um',scale=1.,side='top',snapshots=False):
    q=p.Problem(units=Units(length=length),
        stack=p.Stack(period=(.2*scale,),top=Material(1.),bottom=Material(1.),
            layers=(p.Layer(thickness=.1*scale,epsilon=np.full((1,2),2.25)),)),
        excitation=p.Excitation(wavelength=(.85*scale,.9*scale,.95*scale),sources=(p.PlaneWave(side=side),)),
        outputs=p.Outputs(planes={'outside':-.2*scale},snapshots=p.Snapshots(every=17) if snapshots else None))
    n=p.Numerics(algorithm=algorithm,spacing=scale/30,duration_seconds=55*.9e-6/C0,orders=0)
    return q,n


@pytest.mark.parametrize('algorithm',['shared','reference'])
@pytest.mark.parametrize('side',['top','bottom'])
def test_port_amplitudes_and_si_plane_power_match_original_and_airy(algorithm,side):
    q,n=case(algorithm,side=side)
    r=p.run(q,numerics=n)
    old,diag=solve_periodic_fdtd(dict(structure=np.full((1,1,2),2.25),thickness=[.1],top=1.,bot=1.,
        period=(.2,),wavelength=(.85,.9,.95),polarization=(1.,0.),incidence_side=side,
        resolution=30.,n_periods=55.,n_pml=12,courant=.8,subpixel=True,truncation_order=0,
        return_diagnostics=True,field_monitors={'outside':-.2}),backend='numpy' if algorithm=='shared' else None)
    for name in ('R','T'):
        np.testing.assert_allclose(getattr(r,name),np.stack([getattr(v,name) for v in old]),rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(r.reflection.amplitude_s,np.stack([v.incident.r_s for v in old]),rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(r.plane_power_z['outside'],diag['monitors']['outside']['power_z']*1e-6/Z0,rtol=2e-12)
    # Independent Fresnel recursion for a 1.5-index slab between vacuum ports.
    phase=np.exp(4j*np.pi*1.5*.1/np.array([.85,.9,.95]))
    exact=(-.2+.2*phase)/(1-.04*phase)
    fine=p.run(q,numerics=n.replace(spacing=n.spacing/2))
    np.testing.assert_allclose(fine.reflection.amplitude_s[:,0,0],exact,atol=.005)
    assert np.linalg.norm(fine.reflection.amplitude_s[:,0,0]-exact)<np.linalg.norm(r.reflection.amplitude_s[:,0,0]-exact)
    np.testing.assert_allclose((r.R+r.T)[:,0,0],1.,atol=.002)
    assert r.reflection.reference['side']==side


@pytest.mark.parametrize('algorithm',['shared','reference'])
def test_geometry_unit_changes_keep_fields_times_and_jobs(algorithm):
    q,n=case(algorithm,snapshots=True); q2,n2=case(algorithm,'nm',1000.,snapshots=True)
    job=Job(solver='fdtd.periodic',problem=q2,numerics=n2)
    copy=Job.from_json(job.to_json());assert copy.key==job.key
    a=p.run(q,numerics=n);b=run(copy)
    np.testing.assert_allclose(a.T,b.T,rtol=3e-12,atol=1e-13)
    for component in a.snapshots:
        x,y=a.snapshots[component],b.snapshots[component]
        np.testing.assert_allclose(x.values,y.values,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(x.time,y.time,rtol=2e-14)
        np.testing.assert_allclose(x.coordinates['z']*1000,y.coordinates['z'],rtol=1e-13,atol=1e-10)
    np.testing.assert_allclose(a.plane_power_z['outside'],b.plane_power_z['outside'],rtol=2e-11)


def test_check_uses_no_time_steps_and_refuses_controls_instead_of_dropping_them(monkeypatch):
    q,n=case()
    from wavebit.fdtd._periodic.runtime import NumpyTimeLoop
    def refuse(*a,**kw): raise AssertionError('advanced a field')
    with monkeypatch.context() as m:
        m.setattr(NumpyTimeLoop,'advance',refuse)
        assert p.check(q,numerics=n).ok
        prepared=p.prepare(q,numerics=n)
    assert prepared.report.resolved['steps']>0
    assert not p.check(q,numerics=n.replace(algorithm='reference',align_layers=True)).ok
    assert not p.check(q,numerics=n,precision='single').ok
    assert not p.check(q,numerics=n.replace(pml_cells=0)).ok
    bad=q.replace(excitation=q.excitation.replace(sources=(p.PlaneWave(polarization=(0.,0.)),)))
    assert not p.check(bad,numerics=n).ok


@pytest.mark.parametrize('backend',['jax','torch'])
def test_live_layer_thickness_gradient_matches_finite_difference(backend):
    m=pytest.importorskip(backend)
    if backend=='jax':m.config.update('jax_enable_x64',True)
    u=Units(length='um')
    q=p.Problem(units=u,stack=p.Stack(period=(.05,),bottom=Material(2.25),layers=(
        p.Layer(thickness=.28,epsilon=np.array([[2.25]])),p.Layer(thickness=.12,epsilon=np.array([[1.5]])))),
        excitation=p.Excitation(wavelength=1.,sources=(p.PlaneWave(),)),pulse=p.Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*3*u.time_seconds)))
    n=p.Numerics(spacing=1/25,duration_seconds=45*u.time_seconds,orders=0,nominal_layer_thickness=(.2,.2),
        align_layers=True,pml_cells=25,extraction_method='directional',layer_smoothing=.04)
    prepared=p.prepare(q,numerics=n,backend=backend)
    def loss(d,engine):
        layers=(q.stack.layers[0].replace(thickness=.4-d),q.stack.layers[1].replace(thickness=d))
        live=q.replace(stack=q.stack.replace(layers=layers))
        return (prepared.run(live) if engine==backend else p.run(live,numerics=n,backend=engine)).R.sum()
    step=1e-5;finite=(loss(.123+step,'numpy')-loss(.123-step,'numpy'))/(2*step)
    if backend=='jax':value,gradient=m.jit(m.value_and_grad(lambda d:loss(d,'jax')))(.123)
    else:
        d=m.tensor(.123,dtype=m.float64,requires_grad=True);value=loss(d,'torch');gradient=m.autograd.grad(value,d)[0].detach();value=value.detach()
    np.testing.assert_allclose(float(value),loss(.123,'numpy'),rtol=2e-10,atol=1e-13)
    np.testing.assert_allclose(float(gradient),finite,rtol=2e-6,atol=1e-9)


def test_si_dispersion_is_portable_and_equivalent_under_unit_changes():
    a,n=case();b,m=case(length='nm',scale=1000.)
    terms=(Drude(1e14,1e13),Lorentz(8e14,1e13,1e28))
    dispersion=p.RasterDispersion(labels=np.ones((1,1,2),dtype=int),poles={1:terms})
    a=a.replace(dispersion=dispersion);b=b.replace(dispersion=dispersion)
    original=Job(solver='fdtd.periodic',problem=a,numerics=n)
    assert Job.from_json(original.to_json()).key==original.key
    np.testing.assert_allclose(p.run(a,numerics=n).T,p.run(b,numerics=m).T,rtol=3e-11,atol=1e-12)


def test_crossed_cell_power_is_watts_and_sampled_fields_are_si():
    q,n=case(snapshots=True)
    q=q.replace(stack=q.stack.replace(period=(.2,.2),layers=(q.stack.layers[0].replace(epsilon=np.full((2,2),2.25)),)))
    result=p.run(q,numerics=n)
    plane=result.planes['outside'];v=plane.values
    integrated=.5*np.real(np.cross(v[...,:3],np.conj(v[...,3:])))[...,2].mean(axis=(-1,-2))*(.2e-6)**2
    np.testing.assert_allclose(result.plane_power_z['outside'],integrated,rtol=2e-14)
    assert plane.metadata['power_unit']=='W'
    # The offline interpolation keeps the E/H half-step and returns physical H.
    from wavebit.fdtd._periodic.probes import sample_fdtd_fields
    sample=result.sample(z=.05)
    raw=sample_fdtd_fields(result._engine_diagnostics,z=.05)
    np.testing.assert_allclose(sample.values[:,:3],raw['E'],rtol=1e-13)
    np.testing.assert_allclose(sample.values[:,3:],raw['H']/Z0,rtol=1e-13)
    np.testing.assert_allclose(sample.time,raw['t']*q.units.time_seconds,rtol=1e-13)


def test_full_result_crosses_jit_and_keeps_physical_snapshots_and_power_metadata():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    q,n=case(snapshots=True)
    held=p.prepare(q,numerics=n,backend='jax')
    assert held.validate(q).ok
    assert not held.validate(q.replace(outputs=p.Outputs())).ok
    def forward(amplitude):
        return held.run(q.replace(excitation=q.excitation.replace(sources=(p.PlaneWave(amplitude=amplitude),))))
    result=jax.jit(forward)(1.3)
    expected=forward(1.3)
    np.testing.assert_allclose(result.plane_power_z['outside'],expected.plane_power_z['outside'],rtol=5e-12)
    np.testing.assert_allclose(result.snapshots['Hy'].values,expected.snapshots['Hy'].values,rtol=5e-12,atol=1e-15)
    assert result.planes['outside'].metadata['power_unit']=='W/m'
