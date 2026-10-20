"""Physical native-grid checks: independent SI recurrence, units, and live AD."""
import numpy as np
import pytest

from wavebit import Units, Job, Execution, run
from wavebit.fdtd import native as n
from wavebit.errors import IncompatibleProblemError
from wavebit.units import C0, EPSILON0, Z0


def problem(length='um', scale=1.):
    shape = (16, 2, 2)
    dt = .02e-6/C0
    steps = 24
    current = np.zeros((3,)+shape); current[0, 4] = 1e7
    p = n.Problem(units=Units(length=length), medium=n.Medium(epsilon=np.ones(shape)*2.),
        current=n.Current(density=current, samples=np.sin(np.arange(steps)*.1)),
        outputs=n.Outputs(spectrum=n.Spectrum(wavelengths=[.8*scale], z_indices=(10,)),
            snapshots=n.Snapshots(every=3, components=('Ex','Hy'), slices=((2,13,2), 0, (None,None,None)))))
    nums = n.Numerics(grid=n.Grid(spacing=(.05*scale,)*3,
        z_spacing=np.linspace(.045,.055,16)*scale), time_step_seconds=dt, steps=steps,
        decay_check=n.DecayCheck(window_seconds=7*dt))
    return p, nums


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_uniform_si_conductor_and_current_match_independent_update(backend):
    if backend != 'numpy':
        m=pytest.importorskip(backend)
        if backend == 'jax': m.config.update('jax_enable_x64',True)
    dt=1e-17; sigma=2300.; eps=2.4; current=1e7
    p=n.Problem(units=Units(length='nm'), medium=n.Medium(epsilon=np.full((3,2,2),eps),conductivity=sigma),
        initial=n.InitialField(E=np.ones((3,3,2,2))*2,H=np.ones((3,3,2,2))*.03),
        current=n.Current(density=np.ones((3,3,2,2))*current,samples=np.ones(11)))
    nums=n.Numerics(grid=n.Grid(spacing=(10.,)*3),time_step_seconds=dt,steps=11)
    r=n.run(p,numerics=nums,backend=backend)
    c=sigma*dt/(2*EPSILON0*eps)
    expected=2.
    for _ in range(11): expected=((1-c)*expected-dt*current/(EPSILON0*eps))/(1+c)
    np.testing.assert_allclose(r.E,expected,rtol=2e-14)
    np.testing.assert_allclose(r.H,.03,rtol=2e-14)
    assert r.diagnostics['magnetic_time_seconds']==pytest.approx(10.5*dt,rel=1e-15,abs=0)


def test_geometry_units_records_and_fourier_integrals_are_invariant():
    p, nums=problem(); p2,nums2=problem('nm',1000.)
    a=n.run(p,numerics=nums); b=n.run(p2,numerics=nums2)
    for name in ('E','H','spectral_power_z'):
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=2e-12,atol=1e-50)
    np.testing.assert_allclose(a.spectrum.values,b.spectrum.values,rtol=1e-12,atol=1e-35)
    for name in a.snapshots:
        x,y=a.snapshots[name],b.snapshots[name]
        np.testing.assert_allclose(x.values,y.values,rtol=1e-12,atol=1e-20)
        np.testing.assert_allclose(x.times_seconds,y.times_seconds,rtol=2e-15)
        np.testing.assert_allclose(x.coordinates['z']*1000,y.coordinates['z'],rtol=2e-15)
    np.testing.assert_allclose(a.spectrum.coordinates['z'],np.cumsum(nums.grid.z_spacing)[:10][-1:])
    e,h=a.spectrum.values[...,:3],a.spectrum.values[...,3:]
    expected=.5*np.real(np.cross(e,np.conj(h)))[...,2].sum(axis=(-1,-2))*(.05e-6)**2
    np.testing.assert_allclose(a.spectral_power_z,expected,rtol=2e-14)
    assert a.spectrum.units['H']=='A s/m'


def test_check_does_not_advance_and_job_reuses_only_the_layout(monkeypatch):
    p, nums=problem()
    from wavebit.fdtd._periodic.runtime import NumpyTimeLoop
    def refuse(*a,**k): raise AssertionError('check advanced fields')
    with monkeypatch.context() as m:
        m.setattr(NumpyTimeLoop,'advance',refuse)
        m.setattr(NumpyTimeLoop,'advance_recorded',refuse)
        assert n.check(p,numerics=nums).ok
        prepared=n.prepare(p,numerics=nums)
    job=Job(solver='fdtd.native',problem=p,numerics=nums,execution=Execution())
    restored=Job.from_json(job.to_json())
    assert restored.key==job.key
    np.testing.assert_array_equal(run(restored).E,prepared.run(p).E)
    live=p.replace(medium=p.medium.replace(epsilon=p.medium.epsilon*1.2))
    np.testing.assert_array_equal(prepared.run(live).E,n.run(live,numerics=nums).E)
    with pytest.raises(IncompatibleProblemError):
        prepared.run(p.replace(outputs=n.Outputs()))
    assert not n.check(p,numerics=nums.replace(time_step_seconds=1.)).ok
    assert not n.check(p.replace(medium=n.Medium(epsilon=-p.medium.epsilon)),numerics=nums).ok


@pytest.mark.parametrize('backend',['jax','torch'])
def test_pole_current_and_bloch_derivatives_follow_si_values(backend):
    m=pytest.importorskip(backend)
    if backend=='jax': m.config.update('jax_enable_x64',True)
    p,nums=problem()
    nums=nums.replace(decay_check=None)
    ts=p.units.time_unit_seconds
    mask=np.zeros((16,2,2));mask[6:9]=1.
    def loss(v,engine):
        mask_value=m.tensor(mask) if engine=='torch' else mask
        density=m.tensor(p.current.density) if engine=='torch' else p.current.density
        q=p.replace(medium=p.medium.replace(poles=(n.Oscillator(omega_0=3/ts,gamma=.2/ts,strength=v[0]*mask_value/ts**2),)),
            bloch_phase=(0.,0.,v[1]),current=p.current.replace(density=density*v[2]))
        return n.run(q,numerics=nums,backend=engine).spectral_power_z.sum()*1e35
    x=np.array([.4,.17,1.2]);step=1e-5
    fd=np.array([(loss(x+np.eye(3)[i]*step,'numpy')-loss(x-np.eye(3)[i]*step,'numpy'))/(2*step) for i in range(3)])
    if backend=='jax':
        value,g=m.jit(m.value_and_grad(lambda v:loss(v,'jax')))(m.numpy.asarray(x))
    else:
        v=m.tensor(x,requires_grad=True);value=loss(v,'torch');g=m.autograd.grad(value,v)[0].detach().numpy();value=value.detach()
    np.testing.assert_allclose(float(value),loss(x,'numpy'),rtol=1e-12)
    np.testing.assert_allclose(g,fd,rtol=2e-7,atol=1e-15)


def test_native_cpml_poles_and_si_initial_h_match_normalized_engine():
    from wavebit.fdtd._periodic.grid import fdtd_grid
    shape=(16,2,2);ts=1e-6/C0
    mask=np.zeros(shape);mask[7:9]=1.
    h=np.zeros((3,)+shape);h[1,5]=.03
    q=n.Problem(units=Units(length='um'),medium=n.Medium(epsilon=np.ones(shape),poles=(
        n.Oscillator(omega_0=2/ts,gamma=.1/ts,strength=.4*mask/ts**2),)),initial=n.InitialField(H=h))
    nums=n.Numerics(grid=n.Grid(spacing=(.1,)*3),time_step_seconds=.02*ts,steps=30,
        absorber=n.PML(cells=3,wavelength=1.))
    a=n.run(q,numerics=nums)
    b=fdtd_grid(np.ones(shape),spacing=(.1,)*3,dt=.02,steps=30,magnetic=h*Z0,
        poles=((2.,.1,.4*mask),),pml_cells=3,pml_wavelength=1.)
    np.testing.assert_allclose(a.E,b['E'],rtol=5e-13,atol=1e-14)
    np.testing.assert_allclose(a.H,b['H']/Z0,rtol=5e-13,atol=1e-16)
    np.testing.assert_allclose(a.polarization_density[0],b['polarization'][0]*EPSILON0,rtol=5e-13,atol=1e-26)
    assert bool(a.validity.checks['runtime_stability'])


def test_si_ringdown_fit_recovers_independent_poles():
    from wavebit.observations import fit_modes
    t=np.arange(300)*1e-16
    frequency=2.3e14;decay=1e13
    signal=(.7+.2j)*np.exp((-decay-2j*np.pi*frequency)*t)
    fit=fit_modes(t,signal)
    np.testing.assert_allclose(fit.frequency_hz,[frequency],rtol=2e-13)
    np.testing.assert_allclose(fit.decay_rate_per_second,[decay],rtol=2e-12)
    np.testing.assert_allclose(fit.amplitude,[.7+.2j],rtol=2e-12)
    assert fit.diagnostics['relative_residual']<1e-12


def test_full_result_crosses_jit_with_live_si_observations():
    jax=pytest.importorskip('jax');jax.config.update('jax_enable_x64',True)
    p,nums=problem();held=n.prepare(p,numerics=nums,backend='jax')
    assert held.validate(p).ok
    assert not held.validate(p.replace(outputs=n.Outputs())).ok
    def forward(amplitude):
        return held.run(p.replace(current=p.current.replace(density=p.current.density*amplitude)))
    result=jax.jit(forward)(1.3)
    expected=forward(1.3)
    np.testing.assert_allclose(result.spectrum.values,expected.spectrum.values,rtol=3e-13,atol=1e-30)
    assert result.validity.checks['numerical_finite']
    # The provenance declarations cross jit; their scalar settings stay static.
    assert result.provenance['execution'].backend=='jax' and result.provenance['units']==p.units
    assert result.provenance['numerics'].steps==nums.steps
    assert result.provenance['numerics'].time_step_seconds==nums.time_step_seconds


def test_validity_decay_and_provenance_use_the_shared_result_keys():
    p,nums=problem();r=n.run(p,numerics=nums)
    assert set(r.validity.checks)=={'runtime_stability','numerical_finite'}
    assert bool(r.validity.checks['runtime_stability']) and bool(r.validity.checks['numerical_finite'])
    assert set(r.validity.convergence)=={'decay'}
    decay=r.validity.convergence['decay']
    assert set(decay)==set(r.diagnostics['decay']) and 'tail' not in r.diagnostics
    assert decay['threshold']==nums.decay_check.threshold and 'tolerance' not in decay
    assert {k for k in decay if k.endswith('_units')}=={'material_activity_units',
        'electric_spectrum_before_window_units','magnetic_spectrum_before_window_units'}
    plain=n.run(p,numerics=nums.replace(decay_check=None))
    assert dict(plain.validity.convergence)=={'decay':None} and plain.diagnostics['decay'] is None
    prov=r.provenance
    assert prov['solver']=='fdtd.native' and prov['numerics'] is nums and prov['units'] is p.units
    assert prov['execution']==Execution() and prov['time_convention']=='exp(-i*omega*t)'
    assert prov['time_step_seconds']==nums.time_step_seconds and isinstance(prov['coordinate_frame'],str)
