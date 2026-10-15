"""Job schema migration preserves physical numbers, numerical choices and sweeps."""
import numpy as np
import pytest
from wavebit import Units, Job, run
from wavebit.migration import migrate_legacy_jobs
from wavebit._common._jobs_v1 import run as original_run


def document():
    return dict(version='10',stack=dict(top=1.,bot=2.25,period=[700.],layers=[
        dict(thickness=300.,structure=[[1.,1.,12.25,12.25]])]),
        incidence=dict(wavelength=900.,theta=.13,polarization=[[1.,0.],[0.,1.]]),
        numerics=dict(backend='numpy',truncation_order=[2,0],layer_algorithm='ETM'),
        sweep=dict(wavelength=[850.,900.],theta=[0.,.13]))


def test_explicit_units_expand_the_same_sweep_and_keep_per_order_complex_amplitudes():
    from wavebit._common._jobs_v1 import JobSpec
    doc=document();converted=migrate_legacy_jobs(doc,units=Units(length='nm'))
    old=JobSpec.from_dict(doc).expand()
    assert len(converted)==len(old)==4
    for current,previous in zip(converted,old):
        assert current.numerics.algorithm=='etm'
        assert Job.from_json(current.to_json()).key==current.key
        a,b=run(current),original_run(previous)
        np.testing.assert_allclose(a.R[0],b.R,atol=2e-13)
        np.testing.assert_allclose(a.T[0],b.T,atol=2e-13)
        np.testing.assert_allclose(a.reflection.amplitude_s[0],b.incident.r_s,atol=2e-13)
        np.testing.assert_allclose(a.transmission.amplitude_p[0],b.incident.t_p,atol=2e-13)


def test_migration_requires_units_and_keeps_original_version_gate():
    with pytest.raises(TypeError):migrate_legacy_jobs(document())
    with pytest.raises(ValueError):migrate_legacy_jobs(dict(document(),version='2'),units=Units(length='nm'))
    with pytest.raises(ValueError):migrate_legacy_jobs(document(),units='nm')


@pytest.mark.parametrize('shape',[
    dict(type='square',lx=.3,cx=.06,angle=.1,material=4.),
    dict(l=.3,cx=-.04,cy=.03,material=4.),
    dict(type='ellipse',l=.3,cx=.06,angle=.1,material=4.),
])
def test_shape_defaults_and_aliases_preserve_complex_scattering(shape):
    doc=dict(version='10',stack=dict(top=1.,bot=2.25,period=[1.,1.],layers=[
        dict(thickness=.2,structure=dict(background=1.,shapes=[shape]))]),
        incidence=dict(wavelength=1.1,theta=.13,psi=.4,chi=.12),
        numerics=dict(backend='numpy',truncation_order=[1,1],layer_algorithm='ETM',factorization='nvf'))
    from wavebit._common._jobs_v1 import JobSpec
    old=original_run(JobSpec.from_dict(doc))
    current=run(migrate_legacy_jobs(doc,units=Units(length='um'))[0])
    np.testing.assert_allclose(current.reflection.amplitude_s[0],old.incident.r_s,atol=2e-13)
    np.testing.assert_allclose(current.transmission.amplitude_p[0],old.incident.t_p,atol=2e-13)


def test_unrecognized_shape_fields_cannot_be_silently_dropped():
    doc=document();doc['stack']['layers'][0]['structure']=dict(background=1.,shapes=[
        dict(type='circle',lx=100.,material=4.,unused_parameter=9.)])
    with pytest.raises(ValueError,match='unsupported shape fields'):
        migrate_legacy_jobs(doc,units=Units(length='nm'))


def test_table_permeability_and_oblique_cell_preserve_bottom_incidence():
    from wavebit._common._jobs_v1 import JobSpec, Numerics
    from wavebit._common.periodic import Stack, Layer, Incidence
    from wavebit.materials.tabulated import TabulatedMaterial
    table=TabulatedMaterial.from_refractive_index([800.,1000.],[1.5,1.6],mu=1.2)
    spec=JobSpec(stack=Stack(lattice=[[700.,100.],[0.,600.]],top=table,bot=1.,
        layers=[Layer(100.,np.full((2,2),2.25),mu=np.full((2,2),1.1))]),
        incidence=Incidence(wavelength=900.,theta=.12,polarization=(0.,1.),incidence_side='bottom'),
        numerics=Numerics(backend='numpy',truncation_order=(1,1),factorization='laurent'))
    job=migrate_legacy_jobs(spec.to_json(),units=Units(length='nm'))[0]
    assert Job.from_json(job.to_json()).key==job.key
    a,b=run(job),original_run(spec)
    np.testing.assert_allclose(a.R[0],b.R,rtol=1e-12,atol=2e-14)
    np.testing.assert_allclose(a.transmission.amplitude_p[0],b.incident.t_p,rtol=1e-12,atol=2e-14)
