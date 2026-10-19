"""Scene-wide discretization routes only to bodies with fractional fills.

On base 8715814b68d0a027cafdb335055b0e0efe66c56b, the first ten cases
failed: eight missing public selector refusals and two cell-fill routing errors.
The explicit per-structure selector is the unchanged numerical oracle for both
allowed sphere rules; compare plan identity, report and nonzero final E/H exactly.
No physical tolerance or reference data is introduced or relaxed here.
"""
import pytest
from wavebit import fdtd as fd, Material, Units
from wavebit.errors import ValidationError
from wavebit.fdtd._lower_public import lower_problem
from wavebit.fdtd import _legacy_geometry as old_geometry


def case(shapes, rule='stencil'):
    p = fd.Problem(units=Units(length='um'), scene=fd.Scene(
        domain=fd.Domain(size=(2.,)*3),
        boundaries=fd.Boundaries(x=fd.PEC(), y=fd.PEC(), z=fd.PEC()),
        structures=tuple(fd.Structure(str(i), shape=s, material=Material(2.25)) for i,s in enumerate(shapes))))
    n = fd.Numerics(grid=fd.UniformGrid(spacing=1/18), duration=fd.Duration(steps=2),
                    discretization=rule)
    return p,n


@pytest.mark.parametrize('shapes', [(), (fd.SnappedSlab(z=(-.3,-.2)),),
    (fd.SnappedBox(size=(.2,)*3),), (fd.Box(size=(.2,)*3),),
    (fd.Cylinder(radius=.1,height=.2),),
    (fd.Prism(vertices=((0.,0.),(.1,0.),(0.,.1)),height=.2),),
    (fd.Sphere(radius=.14),fd.Box(size=(.2,)*3))])
def test_stencil_refuses_unsupported_or_absent_body_at_public_path(shapes):
    with pytest.raises(ValidationError) as error:
        lower_problem(*case(shapes))
    assert error.value.path == 'numerics.discretization'


def test_invalid_selector_is_refused_even_in_empty_scene():
    with pytest.raises(ValidationError) as error:
        lower_problem(*case((), 'stencl'))
    assert error.value.path == 'numerics.discretization'


@pytest.mark.parametrize('cell', [fd.SnappedSlab(z=(-.4,-.3)),fd.SnappedBox(size=(.2,)*3)])
def test_cell_fills_keep_voxel_while_sphere_receives_stencil(cell):
    scene,_ = lower_problem(*case((cell,fd.Sphere(radius=.14))))
    assert [s.discretisation for s in scene.structures] == ['voxel','stencil']


@pytest.mark.parametrize('rule', ['voxel','stencil'])
def test_allowed_sphere_lowering_matches_explicit_existing_selector(rule):
    scene,_ = lower_problem(*case((fd.Sphere(radius=.14),),rule))
    assert scene.structures[0].discretisation == rule
    from wavebit.fdtd import _legacy as old
    import numpy as np
    explicit = scene.replace(structures=(old.Structure(scene.structures[0].shape,
        scene.structures[0].material,discretisation=rule),))
    source = old.TFSF('t', box=old.Region(size=(1.11,)*3), direction='+z',
                      polarization='x', band=old.Band(wavelength=1.,df=.1))
    scene,explicit = (s.replace(sources=(source,), boundaries=old.Boundaries.open(old.PML(cells=6,allow_thin=True)),
                                 run=old.Run(steps=100)) for s in (scene,explicit))
    a,b = old.plan(scene),old.plan(explicit)
    assert a.skeleton == b.skeleton
    ra,rb = old.run(scene),old.run(explicit)
    assert np.any(ra.final.E != 0)
    assert str(ra.report) == str(rb.report)
    np.testing.assert_array_equal(ra.final.E,rb.final.E)
    np.testing.assert_array_equal(ra.final.H,rb.final.H)


@pytest.mark.parametrize('extra,background', [
    ((fd.Sphere(radius=.14,center=(.6,0.,0.)),), Material(1.)),
    ((fd.SnappedSlab(z=(-.4,-.3)),), Material(1.)),
    ((), Material(2.))])
def test_existing_lone_sphere_and_vacuum_doors_remain(extra,background):
    from wavebit.fdtd import _legacy as old
    p,n = case((fd.Sphere(radius=.14),)+extra)
    scene,_ = lower_problem(p.replace(scene=p.scene.replace(background=background)),n)
    with pytest.raises(ValueError, match='Stencil(Structures|Background)Refused'):
        old.plan(scene)


@pytest.mark.parametrize('shape', [fd.SnappedSlab(z=(-.4,-.3)),fd.SnappedBox(size=(.2,)*3),
    fd.Box(size=(.3,)*3),fd.Cylinder(radius=.15,height=.3),
    fd.Prism(vertices=((0.,0.),(.3,0.),(0.,.3)),height=.3)])
def test_default_voxel_preserves_every_existing_shape(shape):
    scene,_ = lower_problem(*case((shape,), 'voxel'))
    assert scene.structures[0].discretisation == 'voxel'


def test_public_check_reports_the_selector_path():
    p,n = case(())
    report = fd.check(p,numerics=n)
    assert not report.ok
    assert any(issue.path == 'numerics.discretization' for issue in report.issues)


class _SphereSubclass(fd.Sphere):
    pass


class _KernelSphereSubclass(old_geometry.Sphere):
    pass


class _CustomBall(fd.Shape):
    def body(self):
        from wavebit.fdtd import _legacy_geometry as old
        return old.Sphere(radius=.14).body()


class _SmoothedSlab(fd.SnappedSlab):
    body = _CustomBall.body


class _SmoothedRegion(fd.SnappedBox):
    body = _CustomBall.body


@pytest.mark.parametrize('shape', [fd.Shape(), _CustomBall(),
    _SphereSubclass(radius=.14), _KernelSphereSubclass(radius=.14), _SmoothedSlab(z=(-.4,-.3)),
    _SmoothedRegion(size=(.2,)*3)])
@pytest.mark.parametrize('beside_sphere', [False, True])
def test_stencil_never_accepts_protocol_or_builtin_subclasses(shape, beside_sphere):
    # On 5cb55352 the two beside_sphere=True Slab/Region cases failed.
    # A Slab/Region subclass can override body() to return fractional geometry.
    # Recognising it by isinstance would silently bypass the public shape gate
    # when a genuine Sphere beside it satisfies the applicability test.
    shapes = (shape,) + ((fd.Sphere(radius=.14),) if beside_sphere else ())
    with pytest.raises(ValidationError) as error:
        lower_problem(*case(shapes))
    assert error.value.path == 'numerics.discretization'


@pytest.mark.parametrize('cell,box', [
    (fd.SnappedSlab(z=(-.4,-.3)), (slice(11,13),slice(None),slice(None))),
    (fd.SnappedBox(size=(.2,)*3), (slice(16,20),)*3)])
def test_routed_whole_cell_fill_reaches_prepared_coefficients_exactly(cell, box):
    import numpy as np
    from wavebit.fdtd import _legacy as old
    from wavebit.fdtd._prepare import lower
    scene,_ = lower_problem(*case((cell,fd.Sphere(radius=.14))))
    # The complete stencil scene still fails the lone-sphere gate. Prepare the
    # routed cell separately to inspect its actual fill without extending scope.
    cell_scene = scene.replace(structures=scene.structures[:1])
    plan = old.plan(cell_scene)
    params = lower(cell_scene,plan)
    mask = np.zeros((36,36,36),dtype=bool)
    mask[box] = True
    expected = (plan.dt/plan.dx)/np.where(mask,2.25,1.)
    np.testing.assert_array_equal(np.broadcast_to(params['be'],(3,36,36,36)),
                                  np.broadcast_to(expected,(3,36,36,36)))
    assert not plan.constitutive
    assert cell_scene.structures[0].shape.body() is None
