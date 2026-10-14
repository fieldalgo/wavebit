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
    assert clone.interpolation == ('index' if index else 'epsilon')
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
        body['schema'] = 'wavebit.job/v2'
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
        structures=(fdtd.Structure('film', fdtd.Slab(z=(-.2, 0.)), Material(2.25)),))
    problem = fdtd.Problem(units=Units(length='um'), scene=scene,
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5),
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(monitors=(fdtd.Record('probe', at=(0., 0., -.5), component='Ey'),
            fdtd.Field('spectrum', wavelengths=(.8,), plane=fdtd.Plane(z=-.4), components=('Ey', 'Hx')))))
    numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=3))
    original = Job(solver='fdtd', problem=problem, numerics=numerics)
    restored = Job.from_json(original.to_json())
    assert restored.key == original.key
    assert type(restored.problem.scene.structures[0].shape) is fdtd.Slab
    assert type(restored.problem.sources[0].plane) is fdtd.Plane
    assert type(restored.problem.outputs.monitors[1]) is fdtd.Field
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
