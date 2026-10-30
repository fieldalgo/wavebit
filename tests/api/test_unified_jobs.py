"""Portable jobs preserve material physics and typed numeric data exactly."""
import json
import numpy as np
import pytest

from wavebit import Job, Material, Units, rcwa, run
from wavebit.errors import UnsupportedFeatureError, ValidationError
from wavebit.materials import TabulatedMaterial, ConductiveMaterial, DispersiveMaterial, Drude, Lorentz, Debye


def job(material=None, raster=None):
    layer = (rcwa.Layer(thickness=.2, background=material or Material(2.25)) if raster is None
             else rcwa.Layer(thickness=np.float32(.2), epsilon=raster))
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8,.9), layers=(layer,)),
                           excitation=rcwa.Excitation(wavelength=np.array([.6,.8],dtype='>f8'),
                                                     sources=(rcwa.PlaneWave(polarization=(1.,1j)),)))
    return Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics())


@pytest.mark.parametrize('index', [False, True])
def test_table_roundtrip_preserves_unit_interpolation_and_permeability(index):
    make = TabulatedMaterial.from_refractive_index if index else TabulatedMaterial
    material = make((500.,1000.), (2.+.1j,3.+.2j), mu=(1.1,1.3), wavelength_unit='nm')
    original = job(material)
    restored = Job.from_json(original.to_json())
    clone = restored.problem.stack.layers[0].background
    assert clone.wavelength_unit == 'nm'
    assert clone.interpolated_quantity == ('index' if index else 'epsilon')
    assert clone.mu == material.mu
    np.testing.assert_allclose(clone.at(.8, units=Units(length='um')).epsilon,
                               material.at(.8, units=Units(length='um')).epsilon)
    assert original.key == restored.key
    np.testing.assert_allclose(run(restored).T, run(original).T, atol=2e-14)


@pytest.mark.parametrize('material', [ConductiveMaterial(2.2,1250.,mu=1.2),
    DispersiveMaterial(2.2,(Drude(1e15,1e13),Lorentz(4e15,2e13,1e29),Debye(1e-15,.3)))])
def test_si_models_are_not_reinterpreted_when_jobs_roundtrip(material):
    original = job(material)
    restored = Job.from_json(original.to_json())
    response = restored.problem.stack.layers[0].background.at(.8, units=Units(length='um'))
    np.testing.assert_allclose(response.epsilon, material.at(.8, units=Units(length='um')).epsilon)
    assert restored.key == original.key


def test_arrays_retain_complex_dtype_endianness_shape_and_scalar_distinction():
    array = np.array([[2.+.1j,3.+.2j],[2.+.1j,3.+.2j]], dtype='>c8')
    original = job(raster=array)
    restored = Job.from_json(original.to_json())
    layer = restored.problem.stack.layers[0]
    assert layer.epsilon.dtype.str == '>c8'
    assert layer.epsilon.shape == (2,2)
    np.testing.assert_array_equal(layer.epsilon, array)
    assert layer.thickness.shape == () and layer.thickness.dtype == np.dtype('float32')
    assert restored.problem.excitation.wavelength.dtype.str == '>f8'
    assert restored.problem.excitation.sources[0].polarization == (1.,1j)
    assert restored.key == original.key


def test_job_key_changes_with_physical_and_execution_policy():
    original = job()
    problem = original.problem.replace(units=Units(length='nm'))
    assert original.replace(problem=problem).key != original.key
    assert original.replace(execution=original.execution.replace(precision='single')).key != original.key
    assert Job.from_json(original.to_json(indent=2)).key == original.key


def test_callable_values_are_refused_instead_of_silently_serialized():
    original = job()
    source = original.problem.excitation.sources[0].replace(amplitude=lambda _:1.)
    problem = original.problem.replace(excitation=original.problem.excitation.replace(sources=(source,)))
    with pytest.raises(ValidationError, match='Callables'):
        original.replace(problem=problem).to_json()


@pytest.mark.parametrize('mutation', ['schema', 'array_shape', 'arbitrary_type', 'unknown_field'])
def test_malformed_job_cannot_bypass_explicit_schema_and_declaration_registry(mutation):
    body = json.loads(job().to_json())
    expected = ValidationError
    if mutation == 'schema':
        body['schema'] = 'wavebit.job/v3'
        expected = UnsupportedFeatureError
    elif mutation == 'array_shape':
        body['problem']['fields']['excitation']['fields']['wavelength']['shape'] = [999]
    elif mutation == 'arbitrary_type':
        body['problem']['type'] = 'os.system'
    else:
        body['numerics']['fields']['unrecognized'] = 1
    with pytest.raises(expected):
        Job.from_json(json.dumps(body))


def test_device_buffers_need_explicit_host_conversion_before_saving():
    torch = pytest.importorskip('torch')
    original = job()
    source = original.problem.excitation.sources[0].replace(amplitude=torch.tensor(1.,requires_grad=True))
    problem = original.problem.replace(excitation=original.problem.excitation.replace(sources=(source,)))
    with pytest.raises(ValidationError, match='host arrays'):
        original.replace(problem=problem).to_json()


def test_fdtd_job_roundtrip_preserves_physical_source_geometry_and_monitor_declarations():
    from wavebit import fdtd
    scene = fdtd.Scene(domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries.periodic(z=fdtd.Open()),
        structures=(fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-.2, 0.)), material=Material(2.25)),))
    problem = fdtd.Problem(units=Units(length='um'), scene=scene,
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5),
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.TimeSeriesMonitor('probe', at=(0., 0., -.5), component='Ey'),
            fdtd.FieldMonitor('spectrum', wavelengths=(.8,), plane=fdtd.Plane(z=-.4), components=('Ey', 'Hx')))))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=3))
    original = Job(solver='fdtd', problem=problem, numerics=numerics)
    restored = Job.from_json(original.to_json())
    assert restored.key == original.key
    assert type(restored.problem.scene.structures[0].shape) is fdtd.SnappedSlab
    assert type(restored.problem.sources[0].plane) is fdtd.Plane
    assert type(restored.problem.outputs.monitors[1]) is fdtd.FieldMonitor
    assert restored.problem.sources[0].pulse.frequency_sigma_hz == 2e13
    assert fdtd.check(restored.problem, numerics=restored.numerics).ok
    np.testing.assert_array_equal(run(restored)['probe'].values, run(original)['probe'].values)


def test_oblique_lattice_job_roundtrip_preserves_physical_basis_and_scattering():
    from wavebit import Lattice
    original = job()
    cell = Lattice(np.array([.8, 0.]), np.array([.2, .9]), False)
    problem = original.problem.replace(stack=original.problem.stack.replace(period=None, lattice=cell))
    original = original.replace(problem=problem)
    restored = Job.from_json(original.to_json())
    assert restored.key == original.key
    assert isinstance(restored.problem.stack.lattice, Lattice)
    assert restored.problem.stack.lattice.axis_aligned is False
    np.testing.assert_array_equal(restored.problem.stack.lattice.columns, cell.columns)
    np.testing.assert_allclose(run(restored).transmission.amplitude_s,
                               run(original).transmission.amplitude_s, atol=2e-14)


def test_a_v1_job_file_is_refused_with_the_migration_guide_named():
    """Jobs written before the public names were renamed are refused, not converted."""
    import json
    from wavebit.errors import UnsupportedFeatureError
    body = json.loads(job().to_json())
    assert body['schema'] == 'wavebit.job/v2'
    body['schema'] = 'wavebit.job/v1'
    with pytest.raises(UnsupportedFeatureError, match='public-names.md') as caught:
        Job.from_json(json.dumps(body))
    assert caught.value.code == 'job.schema'


def test_the_execution_tag_names_its_public_path():
    import json
    assert json.loads(job().to_json())['execution']['type'] == 'wavebit.Execution'


@pytest.mark.parametrize('module, old', [('wavebit.fdtd', 'Flux'),
                                         ('wavebit.fdtd.native', 'Record'),
                                         ('wavebit.fdtd.native', 'Tail'),
                                         ('wavebit.fdtd.periodic', 'Record'),
                                         ('wavebit.rcwa', 'OrderExcitation')])
def test_an_allowlisted_name_that_does_not_resolve_is_an_error(monkeypatch, module, old):
    """A removed name is a tombstone; listed by mistake, it fails here, not when a job is read."""
    import re
    from wavebit import jobs
    monkeypatch.setitem(jobs._CLASS_MODULES, module, jobs._CLASS_MODULES[module] + (old,))
    with pytest.raises(RuntimeError, match=re.escape(f'{module}.{old},')):
        jobs._registry()


@pytest.mark.parametrize('module', ['native', 'periodic'])
def test_every_public_native_and_periodic_declaration_has_a_tag(module):
    """The explicit-grid declarations, renamed ones included, can be saved in a job.

    ``fdtd.periodic`` re-exports the ``rcwa`` geometry, which is tagged once, as ``wavebit.rcwa``.
    """
    import dataclasses
    from wavebit import fdtd, jobs
    namespace = getattr(fdtd, module)
    tagged = {id(cls) for cls in jobs._registry().values()}
    for name in namespace.__all__:
        obj = getattr(namespace, name)
        if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                and name not in ('Prepared', 'Result')):
            assert id(obj) in tagged, f'wavebit.fdtd.{module}.{name}'


def test_every_public_fdtd_declaration_has_a_tag_and_round_trips():
    """A renamed declaration missing from the allowlist would fail only when a user saved it."""
    import dataclasses
    from wavebit import fdtd, jobs
    registry = jobs._registry()
    results = {'Prepared', 'Result', 'SpectralFlux', 'FarFieldSpectrum', 'WaveguideMode',
               'WaveguideModes', 'ModeSpectrum'}
    for name in fdtd.__all__:
        obj = getattr(fdtd, name)
        if isinstance(obj, type) and dataclasses.is_dataclass(obj) and name not in results:
            assert f'wavebit.fdtd.{name}' in registry, name
    pulse = fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)
    box = fdtd.SnappedBox(size=(.2, .2, .2))
    problem = fdtd.Problem(units=Units(length='um'), scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1., 1., 1.)),
        boundaries=fdtd.Boundaries(x=(fdtd.MirrorSymmetry(parity=1), fdtd.PEC()),
                                   y=fdtd.InvariantAxis(k=0.), z=fdtd.Open()),
        structures=(fdtd.Structure('a', shape=fdtd.SnappedSlab(z=(-.2, 0.)), material=Material(2.25)),
                    fdtd.Structure('b', shape=box, material=Material(2.)),
                    fdtd.Structure('c', shape=fdtd.Sphere(radius=.1), material=Material(2.)))),
        sources=(fdtd.CurrentDipole('d', center=(0., 0., .3), pulse=pulse, delay_seconds=1e-15),
                 fdtd.TFSF('t', box=box, pulse=pulse)),
        outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.TimeSeriesMonitor('p', at=(0., 0., 0.)),
            fdtd.FieldMonitor('f', wavelengths=(.8,), plane=fdtd.Plane(z=.1)),
            fdtd.FluxMonitor('x', wavelengths=(.8,), box=box),
            fdtd.FarFieldMonitor('ff', wavelengths=(.8,), box=box),
            fdtd.DiffractionMonitor('r', plane=fdtd.Plane(z=.4), wavelengths=(.8,), source='d',
                                    wave='reflected'))))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=3),
                             absorber=fdtd.ConductiveAbsorber(cells=4), discretization='voxel',
                             stopping=fdtd.DecayStop(threshold=1e-3, window_seconds=1e-15))
    original = Job(solver='fdtd', problem=problem, numerics=numerics)
    restored = Job.from_json(original.to_json())
    assert restored == original and restored.key == original.key


def _coherent_job():
    sources = (rcwa.IncidentOrders(weights=(((0, 0), 1.), ((1, 0), .5j)), theta=.1),
               rcwa.GaussianBeam(waist=1., angular_spectrum_extent=3., sampling=(5, 5)))
    problem = rcwa.Problem(units=Units(length='um'), stack=rcwa.Stack(period=(.8, .9), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=.6, sources=sources),
        outputs=rcwa.Outputs(diffraction=False, power_region=((-.4, .4), (-.45, .45))))
    return Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics(orders=(1, 1)))


def test_incident_orders_and_a_gaussian_beam_round_trip_under_their_public_names():
    original = _coherent_job()
    text = original.to_json()
    assert 'wavebit.rcwa.IncidentOrders' in text and '"weights"' in text
    assert '"angular_spectrum_extent"' in text
    assert 'OrderExcitation' not in text and 'spectral_extent' not in text
    restored = Job.from_json(text)
    assert restored == original and restored.key == original.key
    orders, beam = restored.problem.excitation.sources
    assert orders.weights == (((0, 0), 1.), ((1, 0), .5j))
    assert beam.angular_spectrum_extent == 3.
    for port in ('top', 'bottom'):
        np.testing.assert_array_equal(run(restored).outputs[port].outgoing,
                                      run(original).outputs[port].outgoing)


@pytest.mark.parametrize('old, new', [('"wavebit.rcwa.OrderExcitation"', '"wavebit.rcwa.IncidentOrders"'),
                                      ('"orders"', '"weights"'),
                                      ('"spectral_extent"', '"angular_spectrum_extent"')])
def test_a_v2_job_with_an_old_rcwa_name_is_refused(old, new):
    """An old tag or field is refused, never converted; the refusal names the migration guide."""
    text = _coherent_job().to_json()
    assert text.count(new) == 1
    with pytest.raises(ValidationError, match='public-names.md') as caught:
        Job.from_json(text.replace(new, old))
    assert caught.value.code == 'job.encoding'


def test_every_public_rcwa_declaration_has_a_tag():
    import dataclasses
    from wavebit import jobs
    registry = jobs._registry()
    for name in rcwa.__all__:
        obj = getattr(rcwa, name)
        if (isinstance(obj, type) and dataclasses.is_dataclass(obj)
                and name not in ('Prepared', 'Result', 'BoundaryPower')):
            assert f'wavebit.rcwa.{name}' in registry, name


def _solver_job(solver):
    """A small job of each solver, from the tests that already pin its numbers."""
    if solver == 'rcwa':
        return job()
    if solver == 'fdtd':
        from tests.api.test_run_records import fdtd_job
        return fdtd_job()
    if solver == 'fdtd.native':
        from tests.fdtd.test_physical_native import problem
        problem, numerics = problem()
    else:
        from tests.fdtd.test_physical_periodic import case
        problem, numerics = case()
    return Job(solver=solver, problem=problem, numerics=numerics)


@pytest.mark.parametrize('solver', ['rcwa', 'fdtd', 'fdtd.native', 'fdtd.periodic'])
def test_every_solver_result_carries_the_shared_provenance_core(solver):
    import wavebit
    from wavebit import Execution
    original = _solver_job(solver)
    result = run(original)
    provenance = result.provenance
    assert {'solver', 'execution', 'numerics', 'units', 'coordinate_frame',
            'time_convention'} <= set(provenance)
    assert provenance['solver'] == solver
    assert provenance['execution'] == Execution()
    assert provenance['numerics'] is original.numerics
    assert provenance['units'] == original.problem.units
    assert isinstance(provenance['coordinate_frame'], str) and provenance['coordinate_frame']
    assert provenance['time_convention'] == 'exp(-i*omega*t)'
    if solver == 'rcwa':
        assert provenance['code_version'] == wavebit.__version__
    else:
        assert provenance['time_step_seconds'] > 0
