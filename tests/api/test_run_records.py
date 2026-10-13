"""Research records preserve measured data and reject damaged or ambiguous archives."""
import io
import json
from contextlib import nullcontext
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from wavebit import Execution, Job, Material, Units, fdtd, rcwa, run
from wavebit.errors import ValidationError
from wavebit import records
from wavebit._common.provenance import capture_source, sha256


def rcwa_job(backend='numpy'):
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(.8, .9), layers=(rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=np.array([.6, .8], dtype='>f8' if backend == 'numpy' else 'float64'),
            sources=(rcwa.PlaneWave(polarization=(1., 1j)),)),
        outputs=rcwa.Outputs(fields=True, scattering=True))
    return Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics(), execution=Execution(backend=backend))


def fdtd_job():
    problem = fdtd.Problem(units=Units(length='um'),
        scene=fdtd.Scene(domain=fdtd.Domain(size=(.1, .1, 1.6)),
            boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())),
        sources=(fdtd.PlaneWave('pw', plane=fdtd.Plane(z=.5),
            pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
        outputs=fdtd.Outputs(final_fields=True, monitors=(
            fdtd.Record('probe', at=(0., 0., -.5), component='Ey'),
            fdtd.Field('spectrum', wavelengths=(.8,), plane=fdtd.Plane(z=-.4), components=('Ey', 'Hx')))))
    return Job(solver='fdtd', problem=problem, numerics=fdtd.Numerics(
        grid=fdtd.UniformGrid(spacing=.05), duration=fdtd.Duration(steps=6)))


def sampled(result):
    return {'field': result.field(x=(0., .1), y=(0.,), z=(.1,))}


@pytest.fixture(scope='module')
def solved():
    job = rcwa_job()
    result = run(job)
    return job, result


def update_payload(path, name, data):
    (path/name).write_bytes(data)
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['files'][name] = {'sha256': sha256(data), 'bytes': len(data)}
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_rcwa_fields_scattering_coordinates_and_job_roundtrip(tmp_path, solved):
    job, result = solved
    record = records.save_run(tmp_path/'rcwa', job=job, result=result, samples=sampled(result))
    restored = records.load_run(record.path)
    assert restored.job.key == job.key
    assert restored.job.problem.excitation.wavelength.dtype.str == '>f8'
    np.testing.assert_array_equal(restored.outputs['transmitted']['amplitude_s'], result.transmission.amplitude_s)
    np.testing.assert_array_equal(restored.outputs['transmitted']['efficiency'], result.T)
    assert not restored.outputs['transmitted']['efficiency'].flags.writeable
    field = restored.samples['field']
    live = sampled(result)['field']
    assert field.type_name == 'wavebit.observations.fields.FieldSamples'
    assert field['units'] == live.units
    np.testing.assert_array_equal(field['values'], live.values)
    for axis, values in field['coordinates'].items():
        np.testing.assert_array_equal(values, live.coordinates[axis])
    assert field['metadata'] == live.metadata
    assert len(restored.extra['scattering']) == len(result.scattering)
    for saved, live in zip(restored.extra['scattering'], result.scattering):
        for block in ('S11', 'S12', 'S21', 'S22'):
            np.testing.assert_array_equal(saved[block], getattr(live, block))
        for port in ('entry_port', 'exit_port'):
            assert saved[port]['channel'] == getattr(live, port).channel
            np.testing.assert_array_equal(saved[port]['index'], getattr(live, port).index)
    assert 'field_reconstruction' in restored.deferred
    assert 'caller-supplied' in restored.manifest['association']
    assert (record.path/'source.zip').is_file()
    np.testing.assert_allclose(run(restored.job).T, result.T, atol=2e-14)


def test_fdtd_time_spectral_units_source_calibration_and_final_fields(tmp_path):
    job = fdtd_job()
    result = run(job)
    record = records.save_run(tmp_path/'fdtd', job=job, result=result)
    probe = record.outputs['probe']
    assert probe['units'] == 'V/m' and probe['metadata']['time_unit'] == 's'
    np.testing.assert_array_equal(probe['times'], result['probe'].times)
    np.testing.assert_array_equal(probe['values'], result['probe'].values)
    spectrum = record.outputs['spectrum']
    assert spectrum['units'] == {'Ey': 'V*s/m', 'Hx': 'A*s/m'}
    for component in ('Ey', 'Hx'):
        np.testing.assert_array_equal(spectrum['values'][component], result['spectrum'].values[component])
        for axis, values in enumerate(spectrum['coordinates'][component]):
            np.testing.assert_array_equal(values, result['spectrum'].coordinates[component][axis])
    assert record.extra['sources']['pw']['time_unit'] == 's'
    np.testing.assert_array_equal(record.extra['sources']['pw']['incident'], result.sources['pw']['incident'])
    final = record.extra['final_fields']
    assert final['units'] == {'E': 'V/m', 'H': 'A/m'}
    np.testing.assert_array_equal(final['values']['H'], result.final_fields.values['H'])


def test_run_recorded_executes_once_and_keeps_samples(tmp_path, monkeypatch, solved):
    job, result = solved
    calls = []
    def execute(declaration):
        calls.append(declaration.key)
        return result
    monkeypatch.setattr(records, 'run', execute)
    saved = records.run_recorded(job, tmp_path/'run', sample=sampled, include_source=False)
    assert calls == [job.key]
    assert saved.manifest['execution']['source_unchanged']
    assert saved.samples['field']['units']
    assert not (saved.path/'source.zip').exists()
    with pytest.raises(FileExistsError):
        records.run_recorded(job, saved.path)
    assert calls == [job.key]


def test_run_recorded_refuses_changed_source_during_sampling(tmp_path, monkeypatch, solved):
    package = tmp_path/'installed'/'wavebit'
    package.mkdir(parents=True)
    source = package/'__init__.py'
    source.write_text('VERSION = 1\n')
    monkeypatch.setattr(records, '_PACKAGE', package)
    monkeypatch.setattr(records, 'run', lambda _: solved[1])
    def mutate(result):
        source.write_text('VERSION = 2\n')
        return {}
    with pytest.raises(ValidationError, match='changed during execution'):
        records.run_recorded(solved[0], tmp_path/'run', sample=mutate)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_backend_results_save_as_owned_host_data_and_load_without_backend(tmp_path, backend):
    module = pytest.importorskip(backend)
    job = rcwa_job(backend)
    with module.enable_x64() if backend == 'jax' else nullcontext():
        result = run(job)
    samples = None
    if backend == 'torch':
        samples = {'gradient_sample': module.tensor([1.+2j], dtype=module.complex128, requires_grad=True).conj()}
    record = records.save_run(tmp_path/backend, job=job, result=result, samples=samples, include_source=False)
    if samples is not None:
        np.testing.assert_array_equal(record.samples['gradient_sample'], [1.-2j])
        assert samples['gradient_sample'].requires_grad
    assert type(record.outputs['transmitted']['efficiency']) is np.ndarray
    code = """import sys
from wavebit.records import load_run
r = load_run(sys.argv[1])
assert r.job.execution.backend == sys.argv[2]
assert not {'jax', 'jaxlib', 'torch', 'matplotlib'} & set(sys.modules)
"""
    subprocess.run([sys.executable, '-c', code, str(record.path), backend], check=True)


def test_numeric_dtypes_shapes_nonfinite_and_complex_scalars_are_preserved(tmp_path, solved):
    samples = {'scalar': np.complex128(1+2j), 'zero_dim': np.array(2., dtype='>f4'),
               'matrix': np.array([[1+2j, 3j]], dtype='>c8'), 'invalid': np.array([np.nan, np.inf]),
               'float_invalid': float('nan')}
    saved = records.save_run(tmp_path/'run', job=solved[0], result=solved[1], samples=samples, include_source=False)
    for name, value in samples.items():
        if isinstance(value, (np.ndarray, np.generic)):
            assert saved.samples[name].dtype.str == value.dtype.str
            assert saved.samples[name].shape == value.shape
            np.testing.assert_array_equal(saved.samples[name], value)
    assert np.isnan(saved.samples['float_invalid'])


@pytest.mark.parametrize('damage', ['checksum', 'source_identity', 'unknown_file', 'incomplete', 'symlink', 'array_member', 'object_array', 'single_npy'])
def test_corrupt_archives_fail_with_structured_errors(tmp_path, solved, damage):
    path = tmp_path/'run'
    records.save_run(path, job=solved[0], result=solved[1])
    manifest = json.loads((path/'manifest.json').read_text())
    if damage == 'checksum':
        with (path/'arrays.npz').open('ab') as stream:
            stream.write(b'changed')
    elif damage == 'source_identity':
        manifest['source']['identity'] = '0'*64
        (path/'manifest.json').write_text(json.dumps(manifest))
    elif damage == 'unknown_file':
        manifest['files']['../outside'] = {'bytes': 0, 'sha256': sha256(b'')}
        (path/'manifest.json').write_text(json.dumps(manifest))
    elif damage == 'incomplete':
        (path/'manifest.json').unlink()
    elif damage == 'symlink':
        target = tmp_path/'arrays.npz'
        (path/'arrays.npz').rename(target)
        (path/'arrays.npz').symlink_to(target)
    elif damage == 'single_npy':
        payload = io.BytesIO()
        np.save(payload, np.ones(1))
        update_payload(path, 'arrays.npz', payload.getvalue())
    else:
        payload = io.BytesIO()
        with zipfile.ZipFile(path/'arrays.npz') as archive, zipfile.ZipFile(payload, 'w') as broken:
            for i, name in enumerate(archive.namelist()):
                data = archive.read(name)
                if i == 0:
                    if damage == 'array_member':
                        data = b'raw bytes are not an ndarray'
                    else:
                        stream = io.BytesIO()
                        np.save(stream, np.array([object()]), allow_pickle=True)
                        data = stream.getvalue()
                broken.writestr(name, data)
        update_payload(path, 'arrays.npz', payload.getvalue())
    with pytest.raises(ValidationError) as raised:
        records.load_run(path)
    assert raised.value.code == 'record.format'


def test_existing_destination_and_wrong_solver_never_overwrite(tmp_path, solved):
    path = tmp_path/'run'
    records.save_run(path, job=solved[0], result=solved[1], include_source=False)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileExistsError):
        records.save_run(path, job=solved[0], result=solved[1], include_source=False)
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValidationError, match='job solver'):
        records.save_run(tmp_path/'mismatch', job=fdtd_job(), result=solved[1])
    assert not (tmp_path/'mismatch').exists()


def test_source_snapshot_has_modified_and_untracked_code_but_no_unrelated_files(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    package = tmp_path/'src'/'wavebit'
    package.mkdir(parents=True)
    source = package/'__init__.py'
    source.write_text('VERSION = 1\n')
    (package/'private.env').write_text('DO_NOT_CAPTURE_THIS=1')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Record test', '-c', 'user.email=record@example.invalid',
                    '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], check=True)
    source.write_text('VERSION = 2\n')
    (package/'new.py').write_text('VALUE = 3\n')
    (package/'private.env').write_text('DO_NOT_CAPTURE_THIS=2')
    (tmp_path/'notes.txt').write_text('unrelated research notes')
    info, blob, patch = capture_source(package)
    assert info['git']['package_dirty']
    assert b'+VERSION = 2' in patch
    assert b'DO_NOT_CAPTURE_THIS' not in patch
    assert set(info['files']) == {'wavebit/__init__.py', 'wavebit/new.py'}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert archive.read('wavebit/new.py') == b'VALUE = 3\n'
    (package/'new.py').write_text('VALUE = 4\n')
    assert capture_source(package, include_snapshot=False)[0]['identity'] != info['identity']


def test_unsupported_sample_keys_are_refused_before_a_record_is_created(tmp_path, solved):
    with pytest.raises(ValidationError, match='Mapping keys'):
        records.save_run(tmp_path/'run', job=solved[0], result=solved[1],
                         samples={np.int64(1): 'sample'}, include_source=False)
    assert not (tmp_path/'run').exists()


def test_far_field_evaluator_is_deferred_and_explicit_samples_are_saved(tmp_path):
    from wavebit.fdtd.result import FarFieldSpectrum
    job = fdtd_job()
    result = run(job)
    # This evaluator's opaque state must never be inspected or executed by I/O.
    far = FarFieldSpectrum(_raw=object(), _factor=1., wavelengths=(.8,), units='J/Hz/sr')
    result = result.replace(outputs=dict(result.outputs, far=far))
    saved = records.save_run(tmp_path/'far', job=job, result=result, include_source=False,
        samples={'angles': {'theta': np.array([0., .2]), 'phi': 0., 'units': 'rad'},
                 'intensity': {'values': np.array([[.1, .2]]), 'units': far.units}})
    assert 'far' not in saved.outputs
    assert saved.deferred['far']['units'] == 'J/Hz/sr'
    assert saved.samples['angles']['units'] == 'rad'
    np.testing.assert_array_equal(saved.samples['intensity']['values'], [[.1, .2]])
