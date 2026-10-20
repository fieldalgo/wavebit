"""Independent review of C's physical aperture boundaries; preserve existing tolerance."""
import math
import numpy as np
import pytest
from wavebit import fdtd as fd
from wavebit.errors import ValidationError
from wavebit.fdtd._lower_public import lower_problem
from tests.fdtd.test_monitor_aperture import problem, NUMERICS


@pytest.mark.parametrize('kind',[fd.FieldMonitor,fd.FluxMonitor])
@pytest.mark.parametrize('padded',[False,True])
def test_nonaligned_domain_uses_resolved_physical_box(kind,padded):
    size=.401; dx=.025; center=(.037,-.021,.011)
    width=math.ceil(size/dx)*dx
    low=center[0]-width/2; high=low+width
    # This lies beyond the requested box but inside the documented rounded box.
    span=(low+.0025,low+.05)
    assert span[0] < center[0]-size/2
    p=problem([kind('edge',plane=fd.Plane(z=center[2],x=span),wavelengths=(.8,))])
    boundaries=fd.Boundaries(x=fd.Open() if padded else fd.Periodic(),
                             y=fd.Periodic(),z=fd.Periodic())
    p=p.replace(scene=p.scene.replace(domain=fd.Domain(size=(size,)*3,center=center),boundaries=boundaries))
    _,meta=lower_problem(p,NUMERICS)
    assert meta['physical_domain']['low'][0]==low
    assert meta['physical_domain']['size'][0]==width
    outside=fd.Plane(z=center[2],x=(low-.00001,low+.05))
    p=p.replace(outputs=fd.Outputs(monitors=(kind('edge',plane=outside,wavelengths=(.8,)),)))
    with pytest.raises(ValidationError) as error:
        lower_problem(p,NUMERICS)
    assert error.value.code=='monitor.extent'
    assert error.value.path=='outputs.edge.plane.x'
    assert error.value.expected==(low,high)


@pytest.mark.parametrize('kind',[fd.FieldMonitor,fd.FluxMonitor])
@pytest.mark.parametrize('side',['low','high'])
@pytest.mark.parametrize('distance',['nextafter','tolerance','beyond'])
def test_existing_absolute_boundary_tolerance_is_preserved(kind,side,distance):
    edge=-.2 if side=='low' else .2
    sign=-1 if side=='low' else 1
    endpoint=(np.nextafter(edge,sign*np.inf) if distance=='nextafter' else
              edge+sign*(1e-12 if distance=='tolerance' else 2e-12))
    span=(endpoint,.05) if side=='low' else (-.05,endpoint)
    p=problem([kind('ulp',plane=fd.Plane(z=0.,x=span),wavelengths=(.8,))])
    if distance=='beyond':
        with pytest.raises(ValidationError,match='outside the physical domain') as error:
            lower_problem(p,NUMERICS)
        assert error.value.code=='monitor.extent'
    else:
        lower_problem(p,NUMERICS)


def test_physical_extent_and_sampling_pml_clearance_are_separate():
    nums=NUMERICS.replace(absorber=fd.PML(wavelength=.8))
    monitor=fd.FluxMonitor('face',plane=fd.Plane(z=0.,x=(.125,.2),y=(-.1,.1)),wavelengths=(.8,))
    p=problem([monitor])
    # Unpadded high node does not exist; this is sampling, not monitor_extent.
    report=fd.check(p,numerics=nums)
    assert not report.ok
    assert all(issue.code!='monitor.extent' for issue in report.issues)
    p=p.replace(scene=p.scene.replace(boundaries=fd.Boundaries(
        x=fd.Open(),y=fd.Periodic(),z=fd.Periodic())))
    padded=fd.check(p,numerics=nums)
    assert not padded.ok  # Stored physical face still reaches the absorbing layer.
    assert all(issue.code!='monitor.extent' for issue in padded.issues)
    assert 'absorbing layer' in padded.issues[0].message
    interior=monitor.replace(plane=fd.Plane(z=0.,x=(.1,.15),y=(-.1,.1)))
    assert fd.check(p.replace(outputs=fd.Outputs(monitors=(interior,))),numerics=nums).ok
    outside=monitor.replace(plane=fd.Plane(z=0.,x=(.125,.201),y=(-.1,.1)))
    p=p.replace(outputs=fd.Outputs(monitors=(outside,)))
    report=fd.check(p,numerics=nums)
    assert not report.ok
    assert any(issue.code=='monitor.extent' for issue in report.issues)
