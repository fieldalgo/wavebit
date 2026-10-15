"""Explicit smoothed substrate continuation, physical units, and retained AD."""
import numpy as np
import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import IncompatibleProblemError, UnsupportedFeatureError
from wavebit.fdtd._lower_public import lower_problem
from wavebit.units import C0, Z0


def pillar(radius=.12, height=.3, epsilon=2.25, substrate=2.1, *, scale=1., length='um'):
    plane = -.2375*scale
    source = fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.45*scale),
        pulse=fdtd.GaussianPulse(wavelength=scale, frequency_sigma_hz=.15*C0/1e-6))
    return fdtd.Problem(units=Units(length=length), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.6*scale, .6*scale, 1.3*scale), center=(0., 0., .05*scale)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('glass',
            fdtd.Box(size=(10.*scale, 10.*scale, 2.*scale), center=(0., 0., plane-scale)),
            Material(substrate), continuation=('z-',)),
            fdtd.Structure('pillar', fdtd.Cylinder(radius=radius*scale, height=height*scale,
                center=(.011*scale, -.007*scale, plane+height*scale/2)), Material(epsilon)))),
        sources=(source,), outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.Diffraction('T', plane=fdtd.Plane(z=-.55*scale), source='pw',
                             side='transmit', wavelengths=(scale,), orders=0),)))


def numerics(*, scale=1., steps=40):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05*scale),
        duration=fdtd.Duration(steps=steps), absorber=fdtd.PML(cells=8, wavelength=scale, allow_thin=True),
        geometry_ranges={'pillar': tuple((lo*scale, hi*scale) for lo, hi in
            ((-.145, .145), (-.145, .145), (-.2385, .1635)))})


def test_lowering_preserves_the_smoothed_scene_and_matches_the_normalized_engine():
    # Independently declare the original scene including its inside-domain PML.
    from wavebit.fdtd import _legacy as old
    p, n = pillar(), numerics()
    glass = old.Material(eps=2.1)
    manual = old.Scene(domain=old.Domain(size=(.6, .6, 2.1), center=(0., 0., .05)),
        boundaries=old.Boundaries.periodic(z=(old.PML(cells=8, wavelength=1., background=glass, allow_thin=True),
            old.PML(cells=8, wavelength=1., allow_thin=True))),
        resolution=old.Resolution(dx=.05), run=old.Run(steps=40),
        sources=(old.PlaneWave('pw', z=.45, band=old.Band(wavelength=1., df=.15)),),
        structures=(old.Structure(old.Box(size=(10., 10., 2.), center=(0., 0., -1.2375)), glass),
            old.Structure(old.Cylinder(radius=.12, height=.3, center=(.011, -.007, -.0875),
                range=((-.145, .145), (-.145, .145), (-.2385, .1635))), old.Material(eps=2.25))),
        monitors=(old.Diffraction('T', z=-.55, side='transmit', wavelengths=(1.,), orders=0),))
    lowered, meta = lower_problem(p, n)
    assert lowered.structures[0].shape == manual.structures[0].shape
    assert lowered.structures[0].shape.body() is not None
    assert meta['physical_domain']['size'] == pytest.approx((.6, .6, 1.3))
    result, reference = fdtd.run(p, numerics=n), old.run(manual)
    for i in range(3):
        np.testing.assert_allclose(result.final_fields.values['E'][i], reference.final.E[i], atol=1e-14)
        np.testing.assert_allclose(result.final_fields.values['H'][i], reference.final.H[i]/Z0, atol=1e-14)


def test_same_substrate_in_nanometres_has_identical_fields_and_seconds():
    a = fdtd.run(pillar(), numerics=numerics())
    b = fdtd.run(pillar(scale=1000., length='nm'), numerics=numerics(scale=1000.))
    for field in ('E', 'H'):
        for x, y in zip(a.final_fields.values[field], b.final_fields.values[field]):
            np.testing.assert_allclose(x, y, rtol=3e-12, atol=1e-14)
    assert a.provenance['dt_seconds'] == pytest.approx(b.provenance['dt_seconds'], rel=1e-15)


def test_explicit_continuation_extends_the_box_without_moving_its_interior_face():
    p, n = pillar(), numerics()
    glass = p.scene.structures[0].replace(shape=fdtd.Box(size=(10., 10., .3625), center=(0., 0., -.41875)))
    p = p.replace(scene=p.scene.replace(structures=(glass, p.scene.structures[1])))
    lowered, metadata = lower_problem(p, n)
    box = lowered.structures[0].shape
    assert box.high[2] == pytest.approx(-.2375, abs=1e-15)
    assert box.low[2] < metadata['allocated_domain']['low'][2] - n.grid.spacing
    np.testing.assert_array_equal(lowered.boundaries.face('z-').background.eps, (2.1,)*3)


@pytest.mark.parametrize('change', ['implicit', 'curved', 'turned', 'finite', 'wrong_face', 'closed_face', 'ranged'])
def test_unsupported_continuations_are_refused_before_stepping(change):
    p, n = pillar(), numerics()
    glass = p.scene.structures[0]
    if change == 'implicit': glass = glass.replace(continuation=())
    elif change == 'curved': glass = glass.replace(shape=fdtd.Sphere(radius=2.))
    elif change == 'turned': glass = glass.replace(shape=glass.shape.replace(angle=.1))
    elif change == 'finite': glass = glass.replace(shape=glass.shape.replace(size=(.2, .2, 2.)))
    elif change == 'wrong_face': glass = glass.replace(continuation=('z+',))
    elif change == 'closed_face': glass = glass.replace(continuation=('x-',))
    elif change == 'ranged': n = n.replace(geometry_ranges={**n.geometry_ranges, 'glass': ((-5.,5.),(-5.,5.),(-3.,0.))})
    p = p.replace(scene=p.scene.replace(structures=(glass, p.scene.structures[1])))
    with pytest.raises(UnsupportedFeatureError) as error:
        fdtd.prepare(p, numerics=n)
    assert error.value.code == 'padding_continuation'


def test_prepared_continuation_is_part_of_the_static_contract():
    p, n = pillar(), numerics()
    prepared = fdtd.prepare(p, numerics=n)
    p = p.replace(scene=p.scene.replace(structures=(p.scene.structures[0].replace(continuation=()), p.scene.structures[1])))
    with pytest.raises(IncompatibleProblemError):
        prepared.run(p)


def test_job_roundtrip_retains_continuation_and_fixed_pml_reference():
    from wavebit import Job
    p, n = pillar(), numerics(steps=2)
    n = n.replace(absorber=n.absorber.replace(reference_media={'z-': Material(2.1)}))
    job = Job(solver='fdtd', problem=p, numerics=n)
    restored = Job.from_json(job.to_json())
    assert job.key == restored.key
    assert restored.problem.scene.structures[0].continuation == ('z-',)
    assert restored.numerics.absorber.reference_media['z-'].epsilon == 2.1
    a = fdtd.run(p, numerics=n)
    b = fdtd.run(restored.problem, numerics=restored.numerics)
    for x, y in zip(a.final_fields.values['E'], b.final_fields.values['E']):
        np.testing.assert_array_equal(x, y)


def test_reference_media_cannot_silently_name_a_closed_or_unknown_face():
    from wavebit.errors import ValidationError
    p, n = pillar(), numerics()
    for face in ('x-', 'unknown'):
        with pytest.raises(ValidationError, match='reference_media'):
            fdtd.prepare(p, numerics=n.replace(absorber=n.absorber.replace(reference_media={face: Material(2.1)})))


def test_substrate_and_pillar_gradients_match_fresh_finite_differences():
    torch = pytest.importorskip('torch')
    p, n = pillar(), numerics(steps=100)
    n = n.replace(absorber=n.absorber.replace(reference_media={'z-': Material(2.1)}))
    prepared = fdtd.prepare(p, numerics=n, backend='torch')
    v = torch.tensor([.12, .3, 2.25, 2.1], dtype=torch.float64, requires_grad=True)
    def objective(result):
        return (result.final_fields.values['E'][1] ** 2).sum()
    value = objective(prepared.run(pillar(*v)))
    derivative, = torch.autograd.grad(value, v)
    values = v.detach().numpy()
    finite = []
    for i in range(4):
        step = np.zeros(4); step[i] = 1e-5
        plus = objective(fdtd.run(pillar(*(values+step)), numerics=n))
        minus = objective(fdtd.run(pillar(*(values-step)), numerics=n))
        finite.append((plus-minus)/(2e-5))
    assert np.all(np.abs(finite) > 1e-8)
    np.testing.assert_allclose(derivative.detach(), finite, rtol=2e-5, atol=1e-7)
