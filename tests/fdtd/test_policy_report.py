"""Reports describe the resolved engine policy, including inapplicable structures."""
import pytest
from wavebit import fdtd as fd, Material, Units
from wavebit.fdtd import _legacy_geometry as old
from tests.fdtd.test_fd_junction import quiet
from tests.fdtd.test_public_junction import example


def case(shapes=(), rule='voxel'):
    return fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=(2.,)*3),
        boundaries=fd.Boundaries(x=fd.PEC(),y=fd.PEC(),z=fd.PEC()),
        structures=tuple(fd.Structure(str(i),s,Material(2.25)) for i,s in enumerate(shapes)))), fd.Numerics(
        grid=fd.UniformGrid(spacing=.1),duration=fd.Duration(steps=2),discretisation=rule)


def policies(report):
    assert report.ok, report.issues
    return report.resolved['numerical_policies']


def test_empty_scene_does_not_claim_default_junction_was_applied():
    p,n=case()
    got=policies(quiet(fd.check,p,numerics=n))
    assert got['junction']['requested']=='split'
    assert got['junction']['pairs']==got['junction']['structures']==()
    assert got['discretisation']['structures']==()


def test_whole_cell_fills_and_unrelated_sphere_are_distinguished():
    p,n=case((fd.Slab(z=(-.75,-.55)),fd.Region(size=(.3,)*3,center=(.55,.55,.55)),fd.Sphere(radius=.22)))
    got=policies(quiet(fd.check,p,numerics=n))
    rows=got['discretisation']['structures']
    assert [r['rule'] for r in rows]==[None,None,'voxel']
    assert [r['reason'] for r in rows]==['whole_cell_fill','whole_cell_fill','voxel_default']
    assert all(r['rule'] is None and r['reason']=='no_resolved_junction' for r in got['junction']['structures'])


def test_stencil_report_reads_the_prepared_operator():
    p,n=case((fd.Sphere(radius=.14),),'stencil')
    n=n.replace(grid=fd.UniformGrid(spacing=1/18))
    prepared=quiet(fd.prepare,p,numerics=n)
    got=policies(prepared.report)
    assert got['discretisation']['structures'][0]['rule']=='stencil'
    assert got['discretisation']['structures'][0]['reason']=='prepared_stencil_operator'
    assert policies(quiet(prepared.validate,p))==got


@pytest.mark.parametrize('shape',[fd.Slab(z=(-.75,-.55)),fd.Region(size=(.3,)*3,center=(.55,.55,.55))])
def test_stencil_scene_refused_by_engine_claims_no_application(shape):
    p,n=case((shape,fd.Sphere(radius=.22)),'stencil')
    report=quiet(fd.check,p,numerics=n)
    assert not report.ok
    assert not report.resolved


class CustomSphere(old.Sphere):
    pass


def test_custom_voxel_shape_is_reported_as_protocol_not_builtin_operator():
    p,n=case((CustomSphere(radius=.22),))
    row=policies(quiet(fd.check,p,numerics=n))['discretisation']['structures'][0]
    assert row['rule'] is None
    assert row['reason']=='custom_shape_protocol'


@pytest.mark.parametrize('shape',[CustomSphere(radius=.22),type('PublicSphere',(fd.Sphere,),{})(radius=.22)])
def test_unsupported_stencil_custom_and_subclass_have_no_success_report(shape):
    p,n=case((shape,),'stencil')
    report=quiet(fd.check,p,numerics=n)
    assert not report.ok and report.issues[0].path=='numerics.discretisation'
    assert not report.resolved


def test_continued_junction_records_only_the_standing_body():
    p,n=example(continuation=True)
    prepared=quiet(fd.prepare,p,numerics=n)
    got=policies(prepared.report)['junction']
    assert got['requested']=='staircase'
    assert [r['rule'] for r in got['structures']]==[None,'staircase',None]
    assert [r['reason'] for r in got['structures']]==['junction_face','standing_body','no_resolved_junction']
    pair,=got['pairs']
    assert (pair['face'],pair['standing'],pair['axis'],pair['rule'])==('substrate','pillar','z','staircase')
    assert policies(quiet(prepared.validate,p))['junction']==got
    with pytest.raises(TypeError):
        prepared.report.resolved['numerical_policies']['junction']['structures'][0]['rule']='staircase'


def test_prepared_reuse_cannot_claim_an_old_target_for_changed_geometry():
    p,n=case((fd.Sphere(radius=.22),))
    prepared=quiet(fd.prepare,p,numerics=n)
    before=policies(prepared.report)
    changed=p.replace(scene=p.scene.replace(structures=(p.scene.structures[0].replace(shape=fd.Region(size=(.3,)*3)),)))
    report=quiet(prepared.validate,changed)
    assert not report.ok and not report.resolved
    assert policies(prepared.report)==before


def test_validate_preserves_inapplicable_whole_cell_fill_warning():
    p,n=case((fd.Slab(z=(-.73,-.52)),))
    prepared=quiet(fd.prepare,p,numerics=n)
    assert any(i.code=='geometry.cell_snapped' for i in prepared.report.issues)
    report=quiet(prepared.validate,p)
    assert report.issues==prepared.report.issues
    assert policies(report)==policies(prepared.report)


@pytest.mark.parametrize('rule',['staircase','refuse'])
def test_unused_nondefault_junction_reports_an_error_not_application(rule):
    p,n=case()
    report=quiet(fd.check,p,numerics=n.replace(junction=rule))
    assert not report.ok and report.issues[0].path=='numerics.junction'
    assert not report.resolved


def test_ranged_reuse_cannot_relabel_a_pillar_that_no_longer_stands_flush():
    p,n=example(extra=False)
    n=n.replace(geometry_ranges={'pillar': ((-.16,.19),(-.19,.16),(.0115,.4635))})
    prepared=quiet(fd.prepare,p,numerics=n)
    sub,pillar=p.scene.structures
    changed=p.replace(scene=p.scene.replace(structures=(sub,
        pillar.replace(shape=pillar.shape.replace(center=(.017,-.013,.2625))))))
    report=quiet(prepared.validate,changed)
    assert not report.ok and not report.resolved
    assert policies(prepared.report)['junction']['pairs'][0]['standing']=='pillar'


def test_report_lists_both_flush_pairs_in_structure_order():
    p,n=example(extra=False)
    sub,pillar=p.scene.structures
    left=pillar.replace(name='left',shape=pillar.shape.replace(center=(-.3,0.,.2125)))
    right=pillar.replace(name='right',shape=pillar.shape.replace(center=(.3,0.,.2125)))
    p=p.replace(scene=p.scene.replace(structures=(left,sub,right)))
    got=policies(quiet(fd.check,p,numerics=n))['junction']
    assert [r['rule'] for r in got['structures']]==['staircase',None,'staircase']
    assert {(r['face'],r['standing']) for r in got['pairs']}=={('substrate','left'),('substrate','right')}


@pytest.mark.parametrize('shape',[type('CustomSlab',(fd.Slab,),{})(z=(-.75,-.55)),
                                 type('CustomRegion',(fd.Region,),{})(size=(.3,)*3)])
def test_stencil_cannot_hide_unsupported_cell_subclass_beside_exact_sphere(shape):
    p,n=case((shape,fd.Sphere(radius=.22)),'stencil')
    report=quiet(fd.check,p,numerics=n)
    assert not report.ok and report.issues[0].path=='numerics.discretisation'
    assert not report.resolved
