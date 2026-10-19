"""Public input refusals and actionable stencil preparation controls.

Memory admission is tested before the sparse solve: reaching geometry is observed
with a sentinel, so these routing checks never allocate the fitted operator.
"""
import dataclasses

import pytest

from wavebit import Material, Units, fdtd
from wavebit.errors import ValidationError
from wavebit.fdtd._prepare import stencil


def _plane_problem(plane):
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(.1, .1, 1.6)),
                         boundaries=fdtd.Boundaries(x=fdtd.Periodic(),
                                                   y=fdtd.Periodic(), z=fdtd.Open())),
        sources=(fdtd.PlaneWave('light', plane=plane,
                 pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(final_fields=True))


def _numerics(**changes):
    return fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                         duration=fdtd.Duration(steps=2), **changes)


@pytest.mark.parametrize('plane', [.5, None, 'z=.5', (.5,), {'z': .5}])
def test_plane_wave_rejects_non_plane_before_reading_coordinates(plane):
    problem = _plane_problem(plane)
    report = fdtd.check(problem, numerics=_numerics())
    assert not report.ok
    (issue,) = report.issues
    assert issue.path == 'sources.light.plane'
    assert 'expected Plane' in issue.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as error:
            door(problem, numerics=_numerics())
        assert error.value.path == issue.path
        assert 'expected Plane' in str(error.value)
    # A real Plane remains accepted; the same declaration just outside is refused.
    assert fdtd.check(_plane_problem(fdtd.Plane(z=.5)), numerics=_numerics()).ok
    outside = fdtd.check(_plane_problem(fdtd.Plane(z=.9)), numerics=_numerics())
    assert not outside.ok and 'outside the physical domain' in outside.issues[0].message


def _monitor_problem(monitor):
    problem = _plane_problem(fdtd.Plane(z=.5))
    return dataclasses.replace(problem, outputs=fdtd.Outputs(monitors=(monitor,)))


_WAVELENGTHS = (.8,)


@pytest.mark.parametrize('build, path, expected', [
    (lambda v: fdtd.FieldMonitor('m', plane=v, wavelengths=_WAVELENGTHS), 'outputs.m.plane', 'Plane'),
    (lambda v: fdtd.FluxMonitor('m', plane=v, wavelengths=_WAVELENGTHS), 'outputs.m.plane', 'Plane'),
    (lambda v: fdtd.FieldMonitor('m', box=v, wavelengths=_WAVELENGTHS), 'outputs.m.box', 'Region'),
    (lambda v: fdtd.FluxMonitor('m', box=v, wavelengths=_WAVELENGTHS), 'outputs.m.box', 'Region'),
    (lambda v: fdtd.FarFieldMonitor('m', box=v, wavelengths=_WAVELENGTHS), 'outputs.m.box', 'Region'),
], ids=['field-plane', 'flux-plane', 'field-box', 'flux-box', 'farfield-box'])
@pytest.mark.parametrize('value', [.5, 'z=.5', (.5,)])
def test_monitor_geometry_that_is_not_geometry_is_refused_at_the_monitor(build, path, expected, value):
    """A non-geometry monitor plane/box names the monitor, not the scene structures (#48)."""
    problem = _monitor_problem(build(value))
    report = fdtd.check(problem, numerics=_numerics())
    assert not report.ok
    (issue,) = report.issues
    assert issue.path == path
    assert 'expected ' + expected in issue.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError) as error:
            door(problem, numerics=_numerics())
        assert error.value.path == path


def test_monitor_plane_of_the_wrong_geometry_keeps_its_specific_message():
    # Geometry of the wrong kind is still explained by the monitor's own check.
    problem = _monitor_problem(fdtd.FieldMonitor('m', plane=fdtd.Box(center=(0, 0, 0), size=(.1,) * 3),
                                          wavelengths=_WAVELENGTHS))
    report = fdtd.check(problem, numerics=_numerics())
    assert not report.ok
    assert 'fdtd.Plane' in report.issues[0].message
    assert fdtd.check(_monitor_problem(fdtd.FieldMonitor('m', plane=fdtd.Plane(z=.3),
                                                  wavelengths=_WAVELENGTHS)),
                      numerics=_numerics()).ok


def _sphere_problem(count=1):
    return fdtd.Problem(
        units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(2.,) * 3),
                         boundaries=fdtd.Boundaries.open(),
                         structures=tuple(fdtd.Structure(str(i),
                             shape=fdtd.Sphere(radius=.14, center=(i * .5, 0., 0.)),
                             material=Material(epsilon=2.25)) for i in range(count))),
        sources=(fdtd.TFSF('light', box=fdtd.SnappedBox(size=(1.7,) * 3),
                 pulse=fdtd.GaussianPulse(wavelength=1., frequency_sigma_hz=1e13)),),
        outputs=fdtd.Outputs(final_fields=True))


class _AssemblyReached(Exception):
    pass


def _stop_at_assembly(*args, **kwargs):
    raise _AssemblyReached


def test_stencil_memory_bound_controls_public_preparation(monkeypatch):
    monkeypatch.setattr(stencil, 'geometry', _stop_at_assembly)
    problem = _sphere_problem()
    tight = _numerics(discretization='stencil', memory_bound=1)
    (issue,) = fdtd.check(problem, numerics=tight).issues
    assert 'StencilMemoryRefused' in issue.message
    assert 'Numerics(' in issue.message and 'memory_bound=<bytes>' in issue.message
    assert 'bound declared in Numerics' in issue.message
    assert 'fdtd.Structure(' not in issue.message
    for door in (fdtd.prepare, fdtd.run):
        with pytest.raises(ValidationError, match='StencilMemoryRefused'):
            door(problem, numerics=tight)
    # The offered public option gets through the real prediction guard.
    generous = tight.replace(memory_bound=2 * 1024 ** 3)
    with pytest.raises(_AssemblyReached):
        fdtd.prepare(problem, numerics=generous)
    # Omitting the option still reads the engine default, not a duplicated limit.
    monkeypatch.setattr(stencil, 'MEMORY_BOUND_BYTES', 1)
    (issue,) = fdtd.check(problem, numerics=_numerics(discretization='stencil')).issues
    assert 'StencilMemoryRefused' in issue.message and 'default bound' in issue.message
    with pytest.raises(_AssemblyReached):
        fdtd.prepare(problem, numerics=generous)


@pytest.mark.parametrize('bound', [0, -1., float('inf'), float('nan'), True, '1GB', [1]])
def test_stencil_memory_bound_rejects_invalid_values(bound):
    numerics = _numerics(discretization='stencil', memory_bound=bound)
    report = fdtd.check(_sphere_problem(), numerics=numerics)
    assert not report.ok
    (issue,) = report.issues
    assert issue.path == 'numerics.memory_bound'
    assert 'positive, finite number of bytes' in issue.message
    with pytest.raises(ValidationError, match='positive, finite number of bytes'):
        fdtd.prepare(_sphere_problem(), numerics=numerics)


@pytest.mark.parametrize('rule, count', [('voxel', 1), ('stencil', 0)])
def test_stencil_memory_bound_rejects_unused_settings(rule, count):
    numerics = _numerics(discretization=rule, memory_bound=1024)
    report = fdtd.check(_sphere_problem(count), numerics=numerics)
    assert not report.ok
    (issue,) = report.issues
    assert issue.path == 'numerics.memory_bound'
    assert "discretisation='stencil' with a Sphere" in issue.message


def test_stencil_multiple_structure_remedy_uses_public_numerics():
    problem = _sphere_problem(2)
    (issue,) = fdtd.check(problem, numerics=_numerics(discretization='stencil')).issues
    assert 'StencilStructuresRefused' in issue.message
    assert "Numerics(discretisation='voxel')" in issue.message
    assert "use discretisation='voxel' on structures" not in issue.message
    # Applying the offered setting removes this refusal without deleting a body.
    report = fdtd.check(problem, numerics=_numerics(discretization='voxel'))
    assert report.ok


def test_a_default_memory_bound_is_not_written_to_a_job():
    """A field in ``_FIELDS_OMITTED_AT_DEFAULT`` is written only off its default.

    This kept every job that does not use ``memory_bound`` on the key it had before the field
    existed; the schema has since moved to ``wavebit.job/v2``, and the rule still holds there.
    """
    import json
    from wavebit import Job
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1., 1., 1.)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),
                             duration=fdtd.Duration(steps=3))
    job = Job(solver='fdtd', problem=problem, numerics=numerics)
    assert 'memory_bound' in fdtd.Numerics._FIELDS_OMITTED_AT_DEFAULT
    assert 'memory_bound' not in json.loads(job.to_json())['numerics']['fields']
    assert Job.from_json(job.to_json()).key == job.key


@pytest.mark.parametrize('bound', [1, 2.5e9])
def test_explicit_stencil_memory_bound_survives_job_roundtrip(bound):
    from wavebit import Job
    default = Job(solver='fdtd', problem=_sphere_problem(),
                  numerics=_numerics(discretization='stencil'))
    job = default.replace(numerics=default.numerics.replace(memory_bound=bound))
    restored = Job.from_json(job.to_json())
    assert restored.numerics.memory_bound == bound
    assert restored.key == job.key != default.key
    assert restored.replace(numerics=restored.numerics.replace(memory_bound=None)).key == default.key
