"""First general-scene CUDA scope, SI recurrence, device parity and live gradients.

Device parity uses the existing backend gate's peak-normalized 5e-13 budget;
SI recurrence is independent of Wavebit's lowering and stepping implementation.
CUDA cases skip explicitly when the optional hardware/backend is unavailable.
"""
import numpy as np
import pytest

from wavebit import Execution, Material, Units, fdtd
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError
from wavebit.materials import ConductiveMaterial, DispersiveMaterial, Lorentz
from wavebit.units import EPSILON0


def closed_problem(epsilon=2.25, conductivity=600., amplitude=1.):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4, .4, .4)),
        background=ConductiveMaterial(epsilon=epsilon, conductivity=conductivity),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Periodic())),
        initial=fdtd.InitialField(E=(0., 1., 0.), H=(2., 0., 0.), amplitude=amplitude),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('e', at=(0, 0, 0), component='Ey'),),
                             final_fields=True))


def fixed_numerics(steps=20):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=steps))


def slab_problem(epsilon=2.25, amplitude=1.):
    return fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 2.4)), background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure(name='film', shape=fdtd.Slab(z=(-.15, .15)),
                                   material=Material(epsilon=epsilon)),)),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.65), amplitude=amplitude,
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(
            fdtd.Record('e', at=(0., 0., -.5), component='Ey'),
            fdtd.Field('field', plane=fdtd.Plane(z=-.5), wavelengths=(.8,), components=('Ey', 'Hx')),
            fdtd.Flux('flux', plane=fdtd.Plane(z=-.5), wavelengths=(.8,))), final_fields=True))


@pytest.fixture
def cuda():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA-enabled Torch installation and CUDA device')
    return torch


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def parity(actual, expected):
    actual, expected=host(actual), host(expected)
    assert actual.shape==expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    peak=float(np.max(np.abs(expected), initial=0.))
    error=float(np.max(np.abs(actual-expected), initial=0.))
    if peak==0:
        assert error==0
    else:
        assert error/peak < 5e-13


@pytest.mark.parametrize('case,path', [
    ('decay', 'numerics.decay'), ('absorber', 'numerics.absorber'),
    ('stencil', 'numerics.discretisation'), ('range', 'numerics.geometry_ranges'),
    ('boundary', 'scene.boundaries.x-'), ('shape', 'scene.structures.body.shape'),
    ('material', 'scene.background'), ('source', 'sources.tfsf'),
    ('monitor', 'outputs.far')])
def test_cuda_scope_refuses_before_planning_without_a_device(monkeypatch, case, path):
    # Exercise public admission on CPU-only CI as well. No runtime is constructed.
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, '_planned', lambda *a, **k: pytest.fail('unsupported CUDA scope reached planning'))
    p=closed_problem(); n=fixed_numerics()
    if case=='decay': n=n.replace(decay=fdtd.Decay(threshold=1e-3, window_seconds=1e-15))
    if case=='absorber': n=n.replace(absorber=fdtd.Absorber())
    if case=='stencil': n=n.replace(discretisation='stencil')
    if case=='range': n=n.replace(geometry_ranges={'body': ((-.15, .15),)*3})
    if case=='boundary': p=p.replace(scene=p.scene.replace(boundaries=p.scene.boundaries.replace(x=fdtd.Axis1D(k=.2,k_max=.2))))
    if case in ('shape', 'range', 'stencil'):
        p=p.replace(scene=p.scene.replace(structures=(fdtd.Structure(name='body', shape=fdtd.Sphere(radius=.08), material=Material(epsilon=2.)),)))
    if case=='material':
        p=p.replace(scene=p.scene.replace(background=DispersiveMaterial(epsilon_inf=2., poles=(Lorentz(8e14, 1e13, 1e28),))))
    if case=='source':
        p=p.replace(sources=(fdtd.TFSF('tfsf', box=fdtd.Region(size=(.2,.2,.2)),
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),), initial=None)
    if case=='monitor':
        p=p.replace(outputs=fdtd.Outputs(monitors=(fdtd.FarField('far', wavelengths=(.8,), box=fdtd.Region(size=(.2,.2,.2))),)))
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda')
    assert not report.ok
    assert report.issues[0].code=='cuda_scope'
    assert report.issues[0].path==path
    with pytest.raises(UnsupportedFeatureError) as exc:
        fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    assert exc.value.issue.code=='cuda_scope'
    assert exc.value.issue.path==path


def test_cuda_check_and_prepare_do_not_advance_fields(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    monkeypatch.setattr(api, 'execute', lambda *a, **k: pytest.fail('preparation advanced fields'))
    p=closed_problem(); n=fixed_numerics()
    report=fdtd.check(p,numerics=n,backend='torch',device='cuda:0')
    assert report.ok
    assert report.capabilities['device']=='cuda:0'
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda:0')
    assert prepared.validate(p).ok
    assert prepared.contract['resolved_steps']==20


def test_cuda_prepared_reuse_keeps_structural_and_material_guards(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(Execution, 'validate', lambda self: self)
    p=closed_problem(); n=fixed_numerics()
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    monkeypatch.setattr(api, 'execute', lambda *a, **k: pytest.fail('invalid reuse advanced fields'))
    changed=p.replace(scene=p.scene.replace(background=Material(epsilon=2.)))
    assert not prepared.validate(changed).ok
    with pytest.raises(IncompatibleProblemError): prepared.run(changed)
    invalid=p.replace(scene=p.scene.replace(background=ConductiveMaterial(epsilon=-2.,conductivity=600.)))
    assert not prepared.validate(invalid).ok
    with pytest.raises(ValueError): prepared.run(invalid)


def test_cuda_uniform_conductor_matches_independent_si_recurrence(cuda):
    p=closed_problem(); n=fixed_numerics()
    result=fdtd.run(p,numerics=n,backend='torch',device='cuda:0')
    dt=result.provenance['dt_seconds']
    half=600.*dt/(2*EPSILON0*2.25)
    ratio=(1-half)/(1+half)
    expected=ratio**20
    assert result.provenance['device']=='cuda:0'
    for value in result.final_fields.values.values():
        assert value.device.type=='cuda' and value.dtype==cuda.float64
    assert result['e'].values.device.type=='cuda'
    np.testing.assert_allclose(host(result.final_fields.values['E'][1]),expected,rtol=2e-14,atol=0.)
    np.testing.assert_allclose(host(result.final_fields.values['H'][0]),2.,rtol=2e-14,atol=0.)
    np.testing.assert_allclose(host(result['e'].values),ratio**np.arange(1,21),rtol=2e-14,atol=0.)


def test_cuda_slab_cpml_records_dft_flux_and_prepared_live_reuse(cuda):
    p=slab_problem(); n=fixed_numerics(240)
    reference=fdtd.run(p,numerics=n)
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    result=prepared.run(p)
    assert result.validity.checks['runtime_stability']
    for field in ('E','H'): parity(result.final_fields.values[field],reference.final_fields.values[field])
    parity(result['e'].values,reference['e'].values)
    for field in ('Ey','Hx'):
        assert result['field'].values[field].device.type=='cuda'
        parity(result['field'].values[field],reference['field'].values[field])
    assert result['flux'].energy_density.device.type=='cuda'
    parity(result['flux'].energy_density,reference['flux'].energy_density)
    saved=result['e'].values.clone()
    changed=slab_problem(epsilon=2.5,amplitude=1.5)
    assert prepared.validate(changed).ok
    reused=prepared.run(changed)
    fresh=fdtd.run(changed,numerics=n,backend='torch',device='cuda')
    parity(reused['e'].values,fresh['e'].values)
    assert cuda.equal(result['e'].values,saved)
    assert not cuda.allclose(reused['e'].values,saved)


@pytest.mark.parametrize('leaf_device',['cpu','cuda'])
def test_cuda_conductivity_and_initial_amplitude_gradients_are_independent(cuda,leaf_device):
    sigma=cuda.tensor(600.,dtype=cuda.float64,device=leaf_device,requires_grad=True)
    amplitude=cuda.tensor(1.3,dtype=cuda.float64,device=leaf_device,requires_grad=True)
    p=closed_problem(); n=fixed_numerics()
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    live=closed_problem(conductivity=sigma,amplitude=amplitude)
    result=prepared.run(live)
    loss=result.final_fields.values['E'][1].mean()
    gs,ga=cuda.autograd.grad(loss,(sigma,amplitude))
    dt=result.provenance['dt_seconds']
    factor=dt/(2*EPSILON0*2.25)
    ratio=(1-600*factor)/(1+600*factor)
    np.testing.assert_allclose(host(ga),ratio**20,rtol=2e-12,atol=0.)
    expected=1.3*20*ratio**19*(-2*factor/(1+600*factor)**2)
    np.testing.assert_allclose(host(gs),expected,rtol=2e-12,atol=0.)
    assert gs.device.type==leaf_device and ga.device.type==leaf_device


def test_cuda_slab_material_and_source_gradients_match_cpu_finite_differences(cuda):
    n=fixed_numerics(160)
    prepared=fdtd.prepare(slab_problem(),numerics=n,backend='torch',device='cuda')
    eps=cuda.tensor(2.25,dtype=cuda.float64,device='cuda',requires_grad=True)
    amplitude=cuda.tensor(1.2,dtype=cuda.float64,device='cuda',requires_grad=True)
    result=prepared.run(slab_problem(epsilon=eps,amplitude=amplitude))
    loss=(result.final_fields.values['E']**2).sum()
    ge,ga=cuda.autograd.grad(loss,(eps,amplitude))
    def cpu_loss(epsilon,amp):
        result=fdtd.run(slab_problem(epsilon=epsilon,amplitude=amp),numerics=n)
        return np.sum(result.final_fields.values['E']**2)
    step=2.25e-5
    expected=(cpu_loss(2.25+step,1.2)-cpu_loss(2.25-step,1.2))/(2*step)
    np.testing.assert_allclose(host(ge),expected,rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(host(ga),2*host(loss)/1.2,rtol=2e-12,atol=0.)


def test_cuda_invalid_live_material_keeps_runtime_validity_guard(cuda):
    p=slab_problem(); n=fixed_numerics(40)
    prepared=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    # Existing sampling gate refuses concrete epsilon=4 at this grid/band. A
    # differentiable input must propagate failed validity and NaNs, not bypass it.
    assert not fdtd.check(slab_problem(epsilon=4.),numerics=n,backend='torch',device='cuda').ok
    epsilon=cuda.tensor(4.,dtype=cuda.float64,device='cuda',requires_grad=True)
    result=prepared.run(slab_problem(epsilon=epsilon))
    assert not bool(result.validity.checks['runtime_stability'])
    assert cuda.isnan(result['e'].values).all()

def axis_slab_problem(axis):
    """Rotate the physical slab/port declarations, not array buffers or kernels."""
    size=[.1,.1,.1]; size['xyz'.index(axis)]=2.4
    faces={a: fdtd.Open() if a==axis else fdtd.Periodic() for a in 'xyz'}
    plane=lambda value: fdtd.Plane(**{axis:value})
    return fdtd.Problem(units=Units(length='um'),scene=fdtd.Scene(
        domain=fdtd.Domain(size=tuple(size)),background=Material(epsilon=1.),
        boundaries=fdtd.Boundaries(**faces),
        structures=(fdtd.Structure('film',fdtd.Slab(**{axis:(-.15,.15)}),Material(epsilon=2.25)),)),
        sources=(fdtd.PlaneWave('pw',plane=plane(.65),direction='-'+axis,
            pulse=fdtd.GaussianPulse(wavelength=.8,frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.Field('field',plane=plane(-.5),wavelengths=(.8,)),
            fdtd.Flux('flux',plane=plane(-.5),wavelengths=(.8,))),final_fields=True))


@pytest.mark.parametrize('axis',['x','y','z'])
def test_cuda_axis_permuted_slab_cpml_and_spectral_outputs(cuda,axis):
    p=axis_slab_problem(axis); n=fixed_numerics(140)
    reference=fdtd.run(p,numerics=n)
    result=fdtd.run(p,numerics=n,backend='torch',device='cuda')
    for field in ('E','H'):
        assert np.max(np.abs(reference.final_fields.values[field]))>0
        parity(result.final_fields.values[field],reference.final_fields.values[field])
    for component in reference['field'].values:
        parity(result['field'].values[component],reference['field'].values[component])
    parity(result['flux'].energy_density,reference['flux'].energy_density)


def open_pulse_problem():
    return fdtd.Problem(units=Units(length='um'),scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.4,.45,.5)),background=Material(epsilon=1.,mu=1.2),
        boundaries=fdtd.Boundaries.open()),
        initial=fdtd.InitialField(E=(None,lambda x,y,z: np.exp(-100*(x*x+2*y*y+3*z*z)),None)),
        outputs=fdtd.Outputs(monitors=(
            fdtd.Record('e',at=(.05,.05,.05),component='Ey'),
            fdtd.Flux('box',box=fdtd.Region(size=(.2,.2,.2),center=(.025,0,0)),wavelengths=(1.,))),
            final_fields=True))


def test_cuda_three_dimensional_open_pulse_and_closed_flux_faces(cuda):
    p=open_pulse_problem()
    n=fixed_numerics(40).replace(absorber=fdtd.PML(cells=8,wavelength=1.,allow_thin=True))
    reference=fdtd.run(p,numerics=n)
    result=fdtd.run(p,numerics=n,backend='torch',device='cuda')
    for field in ('E','H'): parity(result.final_fields.values[field],reference.final_fields.values[field])
    parity(result['e'].values,reference['e'].values)
    # Compare each nonzero face independently so cancellation cannot hide an
    # axis, orientation or corner error in the net outward flux.
    assert len(result['box'].faces)==6
    for face,expected in reference['box'].faces.items():
        assert np.max(np.abs(expected))>0
        parity(result['box'].faces[face],expected)
    parity(result['box'].energy_density,reference['box'].energy_density)


@pytest.mark.parametrize('conductivity',[None,600.])
def test_cuda_diagonal_epsilon_and_mu_nonuniform_fields(cuda,conductivity):
    material=Material(epsilon=(2.,2.3,2.6),mu=(1.2,1.4,1.6)) if conductivity is None else ConductiveMaterial(
        epsilon=(2.,2.3,2.6),mu=(1.2,1.4,1.6),conductivity=conductivity)
    p=closed_problem().replace(scene=closed_problem().scene.replace(background=material),
        initial=fdtd.InitialField(E=(lambda x,y,z: np.cos(5*np.pi*y),
                                    lambda x,y,z: np.cos(5*np.pi*z),
                                    lambda x,y,z: np.cos(5*np.pi*x)),
                                 H=(lambda x,y,z: .003*np.sin(5*np.pi*z),
                                    lambda x,y,z: .002*np.sin(5*np.pi*x),
                                    lambda x,y,z: .001*np.sin(5*np.pi*y))))
    n=fixed_numerics(28)
    reference=fdtd.run(p,numerics=n)
    result=fdtd.run(p,numerics=n,backend='torch',device='cuda')
    for field in ('E','H'):
        for component in range(3):
            assert np.max(np.abs(reference.final_fields.values[field][component]))>0
            parity(result.final_fields.values[field][component],reference.final_fields.values[field][component])
    parity(result['e'].values,reference['e'].values)

@pytest.mark.parametrize('device,available,count,precision,code', [
    ('cuda',False,0,'double','device_unavailable'),
    ('cuda:1',True,1,'double','device_unavailable'),
    ('cuda:-1',True,1,'double','device'),
    ('cuda:0',True,1,'single','precision')])
def test_cuda_execution_rejection_precedes_scene_lowering(monkeypatch,device,available,count,precision,code):
    import wavebit.fdtd._api as api
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'is_available',lambda: available)
    monkeypatch.setattr(torch.cuda,'device_count',lambda: count)
    monkeypatch.setattr(api,'lower_problem',lambda *a,**k: pytest.fail('invalid execution reached scene lowering'))
    report=fdtd.check(closed_problem(),numerics=fixed_numerics(),backend='torch',device=device,precision=precision)
    assert not report.ok
    assert report.issues[0].code==code
    with pytest.raises(ValueError) as exc:
        fdtd.prepare(closed_problem(),numerics=fixed_numerics(),backend='torch',device=device,precision=precision)
    assert exc.value.issue.code==code


def test_cuda_prepared_cpu_tensor_patterns_are_static_and_do_not_alias(monkeypatch):
    # A fixed pattern's bytes belong to the prepared signature. Amplitude is the
    # live differentiable leaf; replacing or mutating the pattern needs prepare.
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    field=torch.ones((3,8,8,8),dtype=torch.float64)
    p=closed_problem().replace(initial=fdtd.InitialField(E=field))
    prepared=fdtd.prepare(p,numerics=fixed_numerics(),backend='torch',device='cuda')
    assert prepared.validate(p).ok
    field[0,0,0,0]=2.
    assert not prepared.validate(p).ok
    with pytest.raises(IncompatibleProblemError): prepared.run(p)
    live=p.replace(initial=fdtd.InitialField(E=field.clone().requires_grad_()))
    report=fdtd.check(live,numerics=fixed_numerics(),backend='torch',device='cuda')
    assert not report.ok
    assert report.issues[0].code=='static_input'
    assert report.issues[0].path=='initial.E'


def test_cuda_cpu_initial_tensors_keep_precision_warning_and_input_ownership(cuda):
    from wavebit._common.backend import PrecisionLossWarning
    electric=cuda.zeros((3,8,8,8),dtype=cuda.float32,device='cpu')
    electric[1]=1.25
    magnetic=cuda.zeros((3,8,8,8),dtype=cuda.float64,device='cpu')
    magnetic[0]=.03
    saved_e,saved_h=electric.clone(),magnetic.clone()
    amplitude=cuda.tensor(1.3,dtype=cuda.float32,device='cpu',requires_grad=True)
    p=closed_problem().replace(initial=fdtd.InitialField(E=electric,H=magnetic,amplitude=amplitude))
    n=fixed_numerics(13)
    with pytest.warns(PrecisionLossWarning):
        result=fdtd.run(p,numerics=n,backend='torch',device='cuda:0')
    assert result.final_fields.values['E'].dtype==cuda.float64
    assert result.final_fields.values['H'].dtype==cuda.float64
    assert result.final_fields.values['E'].device==cuda.device('cuda:0')
    dt=result.provenance['dt_seconds']
    half=600.*dt/(2*EPSILON0*2.25)
    ratio=(1-half)/(1+half)
    loss=result.final_fields.values['E'][1].mean()
    expected=1.25*float(amplitude.detach())*ratio**13
    np.testing.assert_allclose(host(loss),expected,rtol=2e-14,atol=0.)
    derivative=cuda.autograd.grad(loss,amplitude)[0]
    assert derivative.dtype==cuda.float32 and derivative.device.type=='cpu'
    np.testing.assert_allclose(host(derivative),1.25*ratio**13,rtol=2*np.finfo(np.float32).eps,atol=0.)
    assert electric.dtype==cuda.float32 and cuda.equal(electric,saved_e)
    assert magnetic.dtype==cuda.float64 and cuda.equal(magnetic,saved_h)
    assert amplitude.requires_grad


def test_cuda_mixed_device_material_leaves_preserve_graph_and_cpu_execution(cuda):
    # All arithmetic is float64; the nonuniform field makes both epsilon and mu
    # matter. A CPU conductivity leaf and CUDA epsilon/mu leaves share one graph.
    n=fixed_numerics(23)
    def compute(device):
        eps=cuda.tensor([2.,2.3,2.6],dtype=cuda.float64,device=device,requires_grad=True)
        mu=cuda.tensor(1.4,dtype=cuda.float64,device=device,requires_grad=True)
        sigma=cuda.tensor(600.,dtype=cuda.float64,device='cpu',requires_grad=True)
        p=closed_problem().replace(scene=closed_problem().scene.replace(background=ConductiveMaterial(
            epsilon=eps,mu=mu,conductivity=sigma)),initial=fdtd.InitialField(
                E=(None,lambda x,y,z: np.cos(5*np.pi*z),None)))
        r=fdtd.run(p,numerics=n,backend='torch',device=device)
        loss=(r.final_fields.values['E']**2).sum()
        derivatives=cuda.autograd.grad(loss,(eps,mu,sigma))
        return r,derivatives
    cpu,expected=compute('cpu')
    gpu,actual=compute('cuda:0')
    for field in ('E','H'):
        assert cpu.final_fields.values[field].device.type=='cpu'
        assert gpu.final_fields.values[field].device==cuda.device('cuda:0')
        parity(gpu.final_fields.values[field],cpu.final_fields.values[field])
    for got,want in zip(actual,expected): parity(got,want)
    assert actual[0].device==cuda.device('cuda:0') and actual[1].device==cuda.device('cuda:0')
    assert actual[2].device.type=='cpu'
    assert np.max(np.abs(host(actual[0])))>0 and abs(float(actual[1]))>0 and abs(float(actual[2]))>0


def test_cuda_mixed_device_source_phase_and_delay_match_finite_differences(cuda):
    n=fixed_numerics(171)
    base=slab_problem()
    def problem(phase,delay_scale):
        return base.replace(sources=(base.sources[0].replace(phase=phase,delay=delay_scale*1e-15),))
    prepared=fdtd.prepare(base,numerics=n,backend='torch',device='cuda:0')
    phase=cuda.tensor(.3,dtype=cuda.float64,device='cpu',requires_grad=True)
    delay=cuda.tensor(.2,dtype=cuda.float64,device='cuda:0',requires_grad=True)
    result=prepared.run(problem(phase,delay))
    loss=(result.final_fields.values['E']**2).sum()
    derivatives=cuda.autograd.grad(loss,(phase,delay))
    assert derivatives[0].device.type=='cpu' and derivatives[1].device==cuda.device('cuda:0')
    def value(phi,scale):
        r=fdtd.run(problem(phi,scale),numerics=n)
        return np.sum(r.final_fields.values['E']**2)
    for step in (1e-4,5e-5):
        expected=((value(.3+step,.2)-value(.3-step,.2))/(2*step),
                  (value(.3,.2+step)-value(.3,.2-step))/(2*step))
        assert all(abs(v)>1e-8 for v in expected)
        np.testing.assert_allclose([float(g.detach().cpu()) for g in derivatives],expected,rtol=2e-6,atol=0.)


def test_cuda_live_reuse_rechecks_scope_even_when_material_signature_matches(monkeypatch):
    import wavebit.fdtd._api as api
    class UnvalidatedMaterial(Material):
        pass
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    p=closed_problem().replace(scene=closed_problem().scene.replace(background=Material(epsilon=2.25)))
    prepared=fdtd.prepare(p,numerics=fixed_numerics(),backend='torch',device='cuda')
    changed=p.replace(scene=p.scene.replace(background=UnvalidatedMaterial(epsilon=2.25)))
    # Both models have the same existing material-family/shape signature. The
    # independent CUDA scope still has to reject the unvalidated declaration.
    assert api._signature(changed,prepared.numerics)==api._signature(p,prepared.numerics)
    monkeypatch.setattr(api,'execute',lambda *a,**k: pytest.fail('unsupported reuse reached execution'))
    report=prepared.validate(changed)
    assert not report.ok and report.issues[0].code=='cuda_scope'
    assert report.issues[0].path=='scene.background'
    with pytest.raises(UnsupportedFeatureError) as exc: prepared.run(changed)
    assert exc.value.issue.code=='cuda_scope'



def test_cuda_live_contract_does_not_advertise_ranged_geometry(monkeypatch):
    monkeypatch.setattr(Execution,'validate',lambda self:self)
    p=closed_problem();n=fixed_numerics()
    cpu=fdtd.prepare(p,numerics=n)
    gpu=fdtd.prepare(p,numerics=n,backend='torch',device='cuda')
    assert 'ranged geometry subject to reserved-support guards' in cpu.contract['live']
    assert gpu.contract['live']==tuple(item for item in cpu.contract['live'] if not item.startswith('ranged geometry'))
