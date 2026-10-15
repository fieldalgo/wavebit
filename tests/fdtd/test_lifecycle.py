"""Independent physical units, outside-padding, and immutable live-reuse contracts."""
import dataclasses
import math
import numpy as np
import pytest
import warnings

from wavebit import Material, Units, fdtd
from wavebit.materials import ConductiveMaterial
from wavebit.units import C0, EPSILON0, Z0
from wavebit.compat.legacy import fd as legacy
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError
from wavebit.fdtd._lower_public import lower_problem


def periodic_problem(*,units='um',epsilon=1.,amplitude=1.,outputs=None):
    return fdtd.Problem(units=Units(length=units),scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,1.,1.)),background=Material(epsilon=epsilon),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(),y=fdtd.Periodic(),z=fdtd.Periodic())),
        initial=fdtd.InitialField(E=(None,lambda x,y,z: np.cos(2*np.pi*z),None),amplitude=amplitude),
        outputs=outputs or fdtd.Outputs(monitors=(fdtd.Record('p',at=(0.,0.,0.),component='Ey'),)))


def small_numerics(steps=8):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.1),duration=fdtd.Duration(steps=steps))


def plane_problem(scale=1.,length='um',epsilon=1.,monitors=None):
    return fdtd.Problem(units=Units(length=length),scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1*scale,.1*scale,1.6*scale)),background=Material(epsilon=epsilon),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(),y=fdtd.Periodic(),z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw',plane=fdtd.Plane(z=.5*scale),
            pulse=fdtd.GaussianPulse(wavelength=.8*scale,frequency_sigma_hz=4e13)),),
        outputs=fdtd.Outputs(monitors=monitors or (fdtd.Record('e',at=(0,0,0),component='Ey'),
                                                 fdtd.Record('h',at=(0,0,0),component='Hx'))))


def plane_numerics(scale=1.,steps=700,cells=12):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05*scale),duration=fdtd.Duration(steps=steps),
                         absorber=fdtd.PML(cells=cells))


def test_check_never_advances_fields(monkeypatch):
    import wavebit.fdtd._api as api
    monkeypatch.setattr(api,'execute',lambda *a,**k:pytest.fail('check executed the field loop'))
    report=fdtd.check(plane_problem(),numerics=plane_numerics(steps=10))
    assert report.ok
    assert report.resolved['grid_shape']==(2,2,56)
    assert report.resolved['padding_cells']['z-']==12
    assert report.resolved['physical_domain']['size']==(.1,.1,1.6)
    assert 'excludes' in report.estimates['scope']


def test_padding_is_outside_and_nonvacuum_background_continues():
    problem=plane_problem(epsilon=2.25);numerics=plane_numerics(steps=40)
    problem=problem.replace(sources=(problem.sources[0].replace(pulse=problem.sources[0].pulse.replace(frequency_sigma_hz=2e13)),))
    scene,meta=lower_problem(problem,numerics)
    assert scene.domain.size==pytest.approx((.1,.1,2.8))
    np.testing.assert_array_equal(scene.boundaries.face('z-').background.eps,(2.25,)*3)
    assert meta['physical_domain']['low']==(-.05,-.05,-.8)
    manual=legacy.Scene(domain=legacy.Domain(size=(.1,.1,2.8)),
        boundaries=legacy.Boundaries.periodic(z=legacy.PML(cells=12,background=legacy.Material(eps=2.25))),
        resolution=legacy.Resolution(dx=.05),run=legacy.Run(steps=40),background=legacy.Material(eps=2.25),
        sources=(legacy.PlaneWave('pw',z=.5,band=legacy.Band(wavelength=.8,df=2e13*1e-6/C0)),),
        monitors=(legacy.Record('e',at=(0,0,0),component='Ey'),legacy.Record('h',at=(0,0,0),component='Hx')))
    old=legacy.run(manual);new=fdtd.run(problem,numerics=numerics)
    np.testing.assert_array_equal(new['e'].values,old['e'].values)
    np.testing.assert_allclose(new['h'].values,old['h'].values/Z0,rtol=1e-15,atol=0)
    np.testing.assert_allclose(new['e'].time,old['e'].t*1e-6/C0,rtol=1e-15)


def test_changing_pml_cells_preserves_physical_domain_and_monitor_coordinates():
    p=plane_problem()
    a=fdtd.run(p,numerics=plane_numerics(steps=2,cells=12))
    b=fdtd.run(p,numerics=plane_numerics(steps=2,cells=16))
    assert a.provenance['physical_domain']==b.provenance['physical_domain']
    np.testing.assert_allclose(a['e'].coordinates['sample'],b['e'].coordinates['sample'],atol=1e-15)
    assert b.provenance['allocated_domain']['size'][2]-a.provenance['allocated_domain']['size'][2]==pytest.approx(.4)


def test_same_physical_pulse_in_nm_and_um_has_equal_fields_and_seconds():
    a=fdtd.run(plane_problem(),numerics=plane_numerics())
    b=fdtd.run(plane_problem(1000.,'nm'),numerics=plane_numerics(1000.))
    for name in ('e','h'):
        assert np.max(np.abs(a[name].values))>1e-6
        np.testing.assert_allclose(a[name].values,b[name].values,rtol=2e-11,atol=2e-13)
        np.testing.assert_allclose(a[name].time,b[name].time,rtol=2e-15)


def test_public_h_and_conductivity_obey_si_update_equation():
    sigma=600.;eps=2.25
    p=periodic_problem(outputs=fdtd.Outputs(final_fields=True))
    p=p.replace(scene=p.scene.replace(background=ConductiveMaterial(epsilon=eps,conductivity=sigma)),
                initial=fdtd.InitialField(E=(0,1,0),H=(2,0,0)))
    n=small_numerics(steps=1);r=fdtd.run(p,numerics=n)
    dt=r.provenance['dt_seconds']
    ratio=(EPSILON0*eps-.5*sigma*dt)/(EPSILON0*eps+.5*sigma*dt)
    np.testing.assert_allclose(r.final_fields.values['E'][1],ratio,rtol=2e-15)
    np.testing.assert_allclose(r.final_fields.values['H'][0],2.,rtol=2e-15)
    assert r.final_fields.units['H']=='A/m'
    assert fdtd.run(periodic_problem(),numerics=n).final_fields is None


def test_prepared_uses_live_amplitude_without_mutating_earlier_results():
    p=periodic_problem();n=small_numerics();prepared=fdtd.prepare(p,numerics=n)
    first=prepared.run(p);saved=first['p'].values.copy()
    changed=p.replace(initial=p.initial.replace(amplitude=2.))
    second=prepared.run(changed)
    np.testing.assert_allclose(second['p'].values,2*saved,rtol=2e-15)
    np.testing.assert_array_equal(first['p'].values,saved)
    assert prepared.validate(changed).ok
    with pytest.raises(dataclasses.FrozenInstanceError):prepared.numerics=n


def test_material_reuse_recomputes_coefficients_and_matches_fresh_prepare():
    p=periodic_problem(epsilon=2.25);n=small_numerics(20);prepared=fdtd.prepare(p,numerics=n)
    changed=p.replace(scene=p.scene.replace(background=Material(epsilon=3.)))
    reused=prepared.run(changed);fresh=fdtd.run(changed,numerics=n)
    np.testing.assert_array_equal(reused['p'].values,fresh['p'].values)
    assert not np.allclose(reused['p'].values,prepared.run(p)['p'].values)


def test_period_duration_rejects_a_new_wavelength_before_execution(monkeypatch):
    import wavebit.fdtd._api as api
    p=plane_problem();n=plane_numerics().replace(duration=fdtd.Duration(periods=2,source='pw'))
    prepared=fdtd.prepare(p,numerics=n)
    changed=p.replace(sources=(p.sources[0].replace(pulse=p.sources[0].pulse.replace(wavelength=.9)),))
    monkeypatch.setattr(api,'execute',lambda *a,**k:pytest.fail('incompatible problem advanced fields'))
    with pytest.raises(IncompatibleProblemError,match='duration'):
        prepared.run(changed)
    assert not prepared.validate(changed).ok


def test_single_precision_is_refused_without_a_silent_double_run():
    p=periodic_problem();n=small_numerics()
    assert not fdtd.check(p,numerics=n,precision='single').ok
    with pytest.raises(UnsupportedFeatureError,match='double precision'):
        fdtd.prepare(p,numerics=n,precision='single')


def test_radians_are_lowered_once_and_pulse_width_is_hz():
    p=plane_problem();s=p.sources[0].replace(theta=.2,phi=.3,delay=1e-15)
    p=p.replace(sources=(s,));lowered,_=lower_problem(p,plane_numerics())
    source=lowered.sources[0]
    assert source.theta==pytest.approx(math.degrees(.2))
    assert source.phi==pytest.approx(math.degrees(.3))
    assert source.band.df==pytest.approx(4e13*p.units.time_seconds)
    assert source.delay==pytest.approx(1e-15/p.units.time_seconds)


@pytest.mark.parametrize('backend',['jax','torch'])
def test_live_amplitude_derivative_matches_exact_quadratic(backend):
    if backend=='jax':
        import jax
        import jax.numpy as xp
        jax.config.update('jax_enable_x64',True)
        value=xp.asarray(1.3)
    else:
        import torch
        value=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
    p=periodic_problem();prepared=fdtd.prepare(p,numerics=small_numerics(),backend=backend)
    def loss(a):
        v=prepared.run(p.replace(initial=p.initial.replace(amplitude=a)))['p'].values
        return (v*v).sum()
    if backend=='jax':
        computed=float(jax.grad(loss)(value));expected=float(2*loss(value)/value)
    else:
        y=loss(value);computed=float(torch.autograd.grad(y,value)[0]);expected=float((2*y/value).detach())
    assert computed==pytest.approx(expected,rel=2e-12,abs=1e-13)


def spectral_problem(scale=1.,length='um',amplitude=1.):
    plane=fdtd.Plane(z=-.4*scale);wavelengths=(.8*scale,)
    monitors=(fdtd.Diffraction('t',plane=plane,wavelengths=wavelengths,source='pw',side='transmit',orders=0),
        fdtd.Diffraction('raw',plane=plane,wavelengths=wavelengths,source='pw',side='transmit',orders=0,raw=True),
        fdtd.Flux('q',plane=plane,wavelengths=wavelengths),
        fdtd.Field('f',plane=plane,wavelengths=wavelengths,components=('Ey','Hx')))
    p=plane_problem(scale,length,monitors=monitors)
    return p.replace(sources=(p.sources[0].replace(amplitude=amplitude),))


def test_raw_spectral_energy_integrates_the_cell_and_keeps_one_port():
    r=fdtd.run(spectral_problem(),numerics=plane_numerics(steps=1000))
    outgoing=r['raw'];scattering=r['t']
    assert scattering.amplitude_s.shape==(1,1,1)
    assert scattering.k_parallel.shape==(1,1,1,2)
    assert np.all(np.real(scattering.kz)<0)  # transmitted wave travels toward -z
    assert scattering.reference['outward_normal']==(0,0,-1)
    assert scattering.reference['cell_area_m2']==pytest.approx(1e-14)
    assert scattering.power is None and not hasattr(scattering,'T')
    assert bool(scattering.metadata['source_usable'][0])
    assert float(scattering.efficiency[0,0,0])==pytest.approx(1.,abs=3e-6)
    np.testing.assert_allclose(outgoing.spectral_energy[:,0,0],-r['q'].energy_density,rtol=1e-6,atol=0)
    np.testing.assert_allclose(outgoing.amplitude_s,
        scattering.amplitude_s*scattering.reference['incident_amplitude'][:,None,None],rtol=1e-14)
    assert outgoing.reference['spectral_energy_units']=='J/Hz'
    assert r['f'].units=={'Ey':'V*s/m','Hx':'A*s/m'}
    np.testing.assert_array_equal(outgoing.at_wavelength(0).spectral_energy,outgoing.spectral_energy)


def test_raw_spectral_units_are_invariant_under_geometry_length_unit():
    a=fdtd.run(spectral_problem(),numerics=plane_numerics(steps=700))
    b=fdtd.run(spectral_problem(1000.,'nm'),numerics=plane_numerics(1000.,steps=700))
    for name in ('amplitude_s','amplitude_p','spectral_energy'):
        np.testing.assert_allclose(getattr(a['raw'],name),getattr(b['raw'],name),rtol=2e-11,atol=0)
    for component in ('Ey','Hx'):
        np.testing.assert_allclose(a['f'].values[component],b['f'].values[component],rtol=2e-11,atol=0)
    np.testing.assert_allclose(a['q'].energy_density,b['q'].energy_density,rtol=2e-11,atol=0)


def test_zero_source_retains_zero_raw_spectrum_but_cannot_normalize():
    with np.errstate(invalid='ignore',divide='ignore'):
        r=fdtd.run(spectral_problem(amplitude=0.),numerics=plane_numerics(steps=30))
    assert not bool(r.validity.checks['source_reference']['t'][0])
    assert np.isnan(r['t'].amplitude_s).all() and np.isnan(r['t'].efficiency).all()
    np.testing.assert_array_equal(r['raw'].amplitude_s,0.)
    np.testing.assert_array_equal(r['raw'].amplitude_p,0.)
    np.testing.assert_array_equal(r['raw'].spectral_energy,0.)


def test_material_derivative_matches_finite_difference_and_invalid_jit_is_marked():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    p=periodic_problem(epsilon=2.25);prepared=fdtd.prepare(p,numerics=small_numerics(20),backend='jax')
    def result(eps):
        return prepared.run(p.replace(scene=p.scene.replace(background=Material(epsilon=eps))))
    def loss(eps):return result(eps)['p'].values[-1]
    center=2.25;h=1e-5
    expected=(float(loss(center+h))-float(loss(center-h)))/(2*h)
    assert abs(expected)>1e-3
    assert float(jax.grad(loss)(jnp.asarray(center)))==pytest.approx(expected,rel=2e-8)
    compiled=jax.jit(lambda eps:(result(eps)['p'].values[-1],result(eps).validity.checks['runtime_stability']))
    value,valid=compiled(jnp.asarray(center));assert bool(valid) and np.isfinite(value)
    value,valid=compiled(jnp.asarray(.01));assert not bool(valid) and np.isnan(value)


@pytest.mark.parametrize('kind,component,amplitude',[('electric','Ey',1e-16),('magnetic','Hy',1e-14)])
def test_point_current_si_units_are_invariant_under_geometry_units(kind,component,amplitude):
    def problem(scale,length):
        return fdtd.Problem(units=Units(length=length),scene=fdtd.Scene(
            domain=fdtd.Domain(size=(1.6*scale,)*3),boundaries=fdtd.Boundaries(x=fdtd.PEC(),y=fdtd.PEC(),z=fdtd.PEC())),
            sources=(fdtd.Dipole('d',center=(0.,0.,0.),component=component,kind=kind,current_moment=amplitude,
                pulse=fdtd.GaussianPulse(wavelength=3.*scale,frequency_sigma_hz=.03*C0/1e-6),delay=-18e-6/C0),),
            outputs=fdtd.Outputs(monitors=(fdtd.Record('p',at=(.3*scale,0,0),component=component),)))
    a=fdtd.run(problem(1.,'um'),numerics=small_numerics(30))
    b=fdtd.run(problem(1000.,'nm'),numerics=small_numerics(30).replace(grid=fdtd.UniformGrid(spacing=100.)))
    assert np.max(np.abs(a['p'].values))>1e-8
    np.testing.assert_allclose(a['p'].values,b['p'].values,rtol=1e-10,atol=1e-15)
    np.testing.assert_allclose(a.sources['d']['drive'],b.sources['d']['drive'],rtol=1e-12,atol=amplitude*1e-14)
    assert max(abs(a.sources['d']['drive']))<=amplitude
    assert a.sources['d']['drive_units']==('A*m' if kind=='electric' else 'V*m')


def test_halfspace_continues_to_outer_edge_with_its_own_material():
    p=plane_problem().replace(scene=plane_problem().scene.replace(structures=(
        fdtd.Structure('substrate',fdtd.Slab(z=(-.8,0.)),Material(epsilon=2.25)),)))
    legacy_scene,metadata=lower_problem(p,plane_numerics(steps=2))
    assert legacy_scene.structures[0].shape.z==pytest.approx((-1.4,0.))
    np.testing.assert_array_equal(legacy_scene.boundaries.face('z-').background.eps,(2.25,)*3)
    np.testing.assert_array_equal(legacy_scene.boundaries.face('z+').background.eps,(1.,)*3)
    assert metadata['physical_domain']['low'][2]==-.8


def test_a_continued_halfspace_has_no_proximity_note_and_a_film_near_the_face_does():
    """The padding continues the substrate and its PML is matched to it: nothing to keep clear.

    The note used to say the substrate was -0.3 from the absorber and that the layer "is matched
    to the background" -- false on both counts once the lowering continues it. A film stopping
    short of an ``Open()`` face is still near a layer matched to vacuum, and still warns, in the
    report and as a Python warning alike.
    """
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=2),
                             absorber=fdtd.PML(cells=12))

    def proximity(*structures):
        p = plane_problem()
        p = p.replace(scene=p.scene.replace(structures=structures))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            report = fdtd.check(p, numerics=numerics)
        assert report.ok, report.issues
        listed = [m for m in report.capabilities['warnings'] if 'inner face of the absorber' in m]
        emitted = [str(w.message) for w in caught
                   if issubclass(w.category, fdtd.diagnostics.AbsorberProximityWarning)]
        assert listed == emitted
        return listed

    glass = Material(epsilon=2.25)
    substrate = fdtd.Structure('substrate', fdtd.Slab(z=(-.8, 0.)), glass)
    assert proximity(substrate) == []
    (below,) = proximity(fdtd.Structure('film', fdtd.Slab(z=(-.7, -.6)), glass))
    assert below.startswith("structures[0] is 0.1 from the inner face of the absorber on 'z-'")
    (above,) = proximity(substrate, fdtd.Structure('film', fdtd.Slab(z=(.6, .7)), glass))
    assert above.startswith("structures[1] is 0.1 from the inner face of the absorber on 'z+'")


def test_source_cannot_be_placed_in_numerical_padding():
    p=plane_problem();p=p.replace(sources=(p.sources[0].replace(plane=fdtd.Plane(z=.9)),))
    assert not fdtd.check(p,numerics=plane_numerics(steps=2)).ok
    with pytest.raises(ValueError,match='physical domain'):
        fdtd.prepare(p,numerics=plane_numerics(steps=2))


def test_raw_bloch_amplitudes_use_the_declared_cell_centre():
    p=spectral_problem();source=p.sources[0].replace(theta=.2,phi=.3)
    faces=fdtd.Bloch.from_source(source,units=p.units)
    p=p.replace(sources=(source,),scene=p.scene.replace(boundaries=fdtd.Boundaries(**faces,z=fdtd.Open())))
    r=fdtd.run(p,numerics=plane_numerics(steps=1000))
    expected_phase=np.exp(1j*(2*np.pi/.8*np.sin(.2))*(np.cos(.3)*.05+np.sin(.3)*.05))
    expected=r['t'].amplitude_s*r['t'].reference['incident_amplitude'][:,None,None]*expected_phase
    np.testing.assert_allclose(r['raw'].amplitude_s,expected,rtol=2e-14,atol=0)
    assert r['t'].reference['lateral_origin']==(0.,0.)


def test_diffraction_band_guard_is_separate_from_runtime_stability():
    p=spectral_problem();monitors=tuple(m.replace(wavelengths=(.8,1.5)) for m in p.outputs.monitors)
    p=p.replace(outputs=p.outputs.replace(monitors=monitors))
    r=fdtd.run(p,numerics=plane_numerics(steps=700))
    np.testing.assert_array_equal(r['t'].metadata['source_usable'],(True,False))
    assert r.validity.checks['runtime_stability']
    assert np.isnan(r['t'].efficiency[1]).all()
    assert np.isfinite(r['raw'].amplitude_s).all()
    assert r['t'].at_wavelength(1).metadata['source_usable'].shape==(1,)


def test_ranged_geometry_derivative_and_runtime_guard_survive_public_lifecycle():
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    body=fdtd.Structure('ball',fdtd.Sphere(radius=.22,center=(-.073,.017,-.009)),Material(epsilon=2.25,mu=1.2))
    p=fdtd.Problem(units=Units(length='um'),scene=fdtd.Scene(domain=fdtd.Domain(size=(1.6,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.PEC(),y=fdtd.PEC(),z=fdtd.PEC()),structures=(body,)),
        sources=(fdtd.Dipole('d',center=(-.55,.03,-.02),current_moment=1e-16,
            pulse=fdtd.GaussianPulse(wavelength=3.,frequency_sigma_hz=.03*C0/1e-6),delay=-18e-6/C0),),
        outputs=fdtd.Outputs(final_fields=True))
    n=small_numerics(24).replace(geometry_ranges={'ball':((-.36,.36),)*3})
    prepared=fdtd.prepare(p,numerics=n,backend='jax')
    def run(radius):
        q=p.replace(scene=p.scene.replace(structures=(body.replace(shape=body.shape.replace(radius=radius)),)))
        r=prepared.run(q)
        return (r.final_fields.values['E']**2).sum(),r.validity.checks['runtime_stability']
    loss=lambda radius:run(radius)[0]
    h=3e-6;expected=(float(loss(.22+h))-float(loss(.22-h)))/(2*h)
    assert abs(expected)>1e-3
    assert float(jax.grad(loss)(jnp.asarray(.22)))==pytest.approx(expected,rel=2e-6)
    compiled=jax.jit(run)
    good,valid=compiled(jnp.asarray(.22));assert bool(valid) and np.isfinite(good)
    bad,valid=compiled(jnp.asarray(.4));assert not bool(valid) and np.isnan(bad)
    assert compiled._cache_size()==1


def test_structural_lists_are_owned_but_numeric_array_leaves_remain_borrowed():
    center = [0., 0., .1]
    size = [.6, .3, .2]
    vertices = [[-.1, -.1], [.1, -.1], [0., .1]]
    pulse = fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=3e13)
    bodies = (fdtd.Sphere(radius=.1, center=center),
              fdtd.Box(size=size, center=center),
              fdtd.Cylinder(radius=.1, height=.2, center=center),
              fdtd.Prism(vertices=vertices, height=.2, center=center),
              fdtd.Dipole('d', center=center, pulse=pulse))
    wavelengths = [.8, .9]
    components = ['Ey', 'Hx']
    orders = [1, 0]
    outputs = (fdtd.Field('f', wavelengths=wavelengths, plane=fdtd.Plane(z=0.), components=components),
               fdtd.Flux('q', wavelengths=wavelengths, plane=fdtd.Plane(z=0.)),
               fdtd.Diffraction('d', wavelengths=wavelengths, plane=fdtd.Plane(z=0.),
                                source='pw', side='transmit', orders=orders),
               fdtd.FarField('far', wavelengths=wavelengths, box=fdtd.Box(size=(1.,1.,1.))))
    center[2] = 99.
    size[0] = 99.
    vertices[0][0] = 99.
    wavelengths[0] = 99.
    components[0] = 'Ex'
    orders[0] = 99
    for body in bodies:
        assert body.center == (0., 0., .1)
    assert bodies[1].size == (.6, .3, .2)
    assert bodies[3].vertices == ((-.1, -.1), (.1, -.1), (0., .1))
    for output in outputs:
        assert output.wavelengths == (.8, .9)
    assert outputs[0].components == ('Ey', 'Hx')
    assert outputs[2].orders == (1, 0)
    # Numeric leaves are borrowed explicitly, not copied or coerced to host scalars.
    array = np.array([0., 0., .1])
    assert fdtd.Box(size=(1.,1.,1.), center=array).center is array
    nested = np.array([[-.1,-.1],[.1,-.1],[0.,.1]])
    assert fdtd.Prism(vertices=nested, height=.2).vertices is nested


@pytest.mark.parametrize('invalid', [None, {}, object()])
def test_prepared_rejects_a_wrong_problem_type_before_execution(invalid, monkeypatch):
    from wavebit.errors import ValidationError
    import wavebit.fdtd._api as api
    prepared = fdtd.prepare(periodic_problem(), numerics=small_numerics())
    monkeypatch.setattr(api, 'execute', lambda *args, **kwargs: pytest.fail('wrong input executed'))
    report = prepared.validate(invalid)
    assert not report.ok
    assert report.issues[0].path == 'problem'
    assert report.issues[0].code == 'problem.type'
    with pytest.raises(ValidationError, match='expected fdtd.Problem'):
        prepared.run(invalid)


def _traced_resolution_case(kind):
    """A traced permittivity no seed covers, the reference value and one the door refuses.

    A cell-snapped Slab and an unturned ranged Box have no cross terms, so host planning
    holds no seed for their material: the eager resolution door cannot count a traced value
    and skips it, and only the runtime guard is left to hold the 8-cells floor.
    """
    if kind == 'slab':
        film = fdtd.Structure('film', fdtd.Slab(z=(-.2, 0.)), Material(epsilon=2.25))
        p = plane_problem().replace(scene=plane_problem().scene.replace(structures=(film,)))
        n = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=200),
                          absorber=fdtd.PML(cells=12))
        return p, n, 400.          # 1.15 cells per wavelength
    box = fdtd.Structure('box', fdtd.Box(size=(.44, .44, .44)), Material(epsilon=2.25))
    p = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(domain=fdtd.Domain(size=(1.6,)*3),
        boundaries=fdtd.Boundaries(x=fdtd.PEC(), y=fdtd.PEC(), z=fdtd.PEC()), structures=(box,)),
        sources=(fdtd.Dipole('d', center=(-.55, .03, -.02), current_moment=1e-16,
            pulse=fdtd.GaussianPulse(wavelength=1.6, frequency_sigma_hz=.03*C0/1e-6), delay=-18e-6/C0),),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('e', at=(.3, 0., 0.), component='Ey'),)))
    n = small_numerics(24).replace(geometry_ranges={'box': ((-.36, .36),)*3})
    return p, n, 12.25             # 3.88 cells per wavelength


def _with_epsilon(problem, eps):
    s = problem.scene.structures[0]
    return problem.replace(scene=problem.scene.replace(structures=(s.replace(material=Material(epsilon=eps)),)))


@pytest.mark.parametrize('kind', ['slab', 'unturned ranged box'])
def test_a_traced_permittivity_below_the_resolution_floor_runs_to_nan_and_fails_validity(kind):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    problem, numerics, refused = _traced_resolution_case(kind.split()[0])
    with pytest.raises(ValueError, match='cells per wavelength'):
        fdtd.prepare(_with_epsilon(problem, refused), numerics=numerics)
    prepared = fdtd.prepare(problem, numerics=numerics, backend='jax')

    def run(eps):
        r = prepared.run(_with_epsilon(problem, eps))
        return r['e'].values.sum(), r.validity.checks['runtime_stability']
    compiled = jax.jit(run)
    value, valid = compiled(jnp.asarray(2.25))
    assert bool(valid)
    assert float(value) == pytest.approx(float(prepared.run(problem)['e'].values.sum()), rel=1e-12)
    value, valid = compiled(jnp.asarray(refused))
    assert not bool(valid) and np.isnan(float(value))
    assert np.isnan(float(jax.grad(lambda eps: run(eps)[0])(jnp.asarray(refused))))


def test_a_torch_permittivity_below_the_resolution_floor_runs_to_nan_and_fails_validity():
    import torch
    problem, numerics, refused = _traced_resolution_case('slab')
    prepared = fdtd.prepare(problem, numerics=numerics, backend='torch')
    eps = torch.tensor(refused, dtype=torch.float64, requires_grad=True)
    r = prepared.run(_with_epsilon(problem, eps))
    assert not bool(r.validity.checks['runtime_stability'])
    assert torch.isnan(r['e'].values).all()


def _film(shape):
    p = plane_problem()
    return p.replace(scene=p.scene.replace(structures=(fdtd.Structure('film', shape, Material(epsilon=2.25)),)))


def test_check_reports_a_cell_snapped_interface_the_grid_moved():
    """A 0.317 film at 0.025 spacing runs as the 0.325 one, bit for bit; check says so."""
    n = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025), duration=fdtd.Duration(steps=60),
                      absorber=fdtd.PML(cells=12))
    moved, whole = _film(fdtd.Slab(z=(0., .317))), _film(fdtd.Slab(z=(0., .325)))
    np.testing.assert_array_equal(fdtd.run(moved, numerics=n)['e'].values,
                                  fdtd.run(whole, numerics=n)['e'].values)
    report = fdtd.check(moved, numerics=n)
    assert report.ok
    (issue,) = report.issues
    assert (issue.code, issue.severity, issue.path) == ('geometry.cell_snapped', 'warning', 'scene.structures.film')
    assert issue.expected == (('z', 0., .317),) and issue.actual == (('z', 0., .325),)
    assert 'fdtd.Box' in issue.message and 'fd.' not in issue.message.replace('fdtd.', '')
    assert fdtd.check(whole, numerics=n).issues == ()
    assert [i.code for i in fdtd.prepare(moved, numerics=n).report.issues] == ['geometry.cell_snapped']
    # A half-space continued into the padding moves only its inner face; a Region clipped
    # by the periodic faces moves only along z; a smoothed Box stays where it was asked.
    (issue,) = fdtd.check(_film(fdtd.Slab(z=(-.8, .317))), numerics=n).issues
    assert issue.expected == (('z', -.8, .317),) and issue.actual == (('z', -.8, .325),)
    (issue,) = fdtd.check(_film(fdtd.Region(size=(.3, .3, .317), center=(0., 0., .1585))), numerics=n).issues
    assert [row[0] for row in issue.expected] == ['z']
    assert fdtd.check(_film(fdtd.Box(size=(.2, .2, .317), center=(0., 0., .1585))), numerics=n).issues == ()
