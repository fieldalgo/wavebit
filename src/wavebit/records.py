"""Portable research runs: declared inputs, measured outputs and source identity.

Archives contain JSON, numeric NPZ arrays and an optional source ZIP. Loading
never executes archived code, imports recorded type names, or recreates a live
solver/autodiff graph. See the run-records guide for the saved data contract.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import time
from types import MappingProxyType
import zipfile

import numpy as np

from .jobs import Job, run
from ._common.provenance import capture_source, environment, sha256, source_identity
from ._common.record_io import RecordedObject, decode, encode, record_error

__all__ = ['RunRecord', 'RecordedObject', 'save_run', 'load_run', 'run_recorded']
_SCHEMA = 'wavebit.run/v2'
_PACKAGE = Path(__file__).resolve().parent
_FILES = {'job.json', 'observations.json', 'arrays.npz', 'source.zip', 'source.patch'}


@dataclass(frozen=True)
class RunRecord:
    """Data from one archive. Declaration fields use mapping access, not methods.

    ``outputs['transmitted']['efficiency']`` is an owned, read-only NumPy array.
    ``samples`` contains any explicitly evaluated fields supplied by the caller.
    Re-execution is explicit: pass ``job`` to ``wavebit.run`` in a chosen environment.
    """
    path: Path
    job: Job
    outputs: Mapping
    samples: Mapping
    validity: RecordedObject
    diagnostics: Mapping
    provenance: Mapping
    extra: Mapping
    deferred: Mapping
    manifest: Mapping


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _json_load(data, path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate key')
            out[key] = value
        return out
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError) as exc:
        raise record_error('Invalid JSON.', path) from exc


def _observations(job, result, samples):
    from .rcwa.results import Result as RCWAResult
    from .fdtd import Result as FDTDResult, FarFieldSpectrum
    expected = RCWAResult if job.solver == 'rcwa' else FDTDResult
    if not isinstance(result, expected) or result.provenance.get('solver') != job.solver:
        raise record_error('The result must belong to the job solver.', 'result')
    if samples is not None and not isinstance(samples, Mapping):
        raise record_error('samples must be a mapping of explicitly evaluated observations.', 'samples')
    outputs, deferred, extra = {}, {}, {}
    for name, value in result.outputs.items():
        if isinstance(value, FarFieldSpectrum):
            deferred[name] = {'type': 'far_field', 'units': value.units,
                              'wavelengths': value.wavelengths,
                              'action': 'Evaluate fields(theta, phi) or intensity(theta, phi) and supply coordinates and values in samples.'}
        else:
            outputs[name] = value
    if isinstance(result, RCWAResult):
        extra['wavelengths'] = result.wavelengths
        if result._scattering:
            extra['scattering'] = result.scattering
        if result._field_records:
            deferred['field_reconstruction'] = {'type': 'rcwa_fields',
                'action': 'Evaluate result.field(x=..., y=..., z=...) and include the returned FieldSamples in samples.'}
    else:
        extra['source_references'] = result.source_references
        extra['final_fields'] = result.final_fields
    return dict(outputs=outputs, samples=dict(samples or {}), validity=result.validity,
                diagnostics=result.diagnostics, provenance=result.provenance,
                extra=extra, deferred=deferred)


def _write(path, *, job, result, samples, source, env, association, execution=None):
    arrays = {}
    observations = encode(_observations(job, result, samples), arrays)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    source_info, snapshot, patch = source
    payloads = {'job.json': job.to_json(indent=2).encode('utf-8'),
                'observations.json': _json_bytes(observations), 'arrays.npz': stream.getvalue()}
    if snapshot is not None:
        payloads['source.zip'] = snapshot
    if patch:
        payloads['source.patch'] = patch
    manifest = {'schema': _SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'job_key': job.key, 'solver': job.solver, 'environment': env,
                'source': source_info, 'association': association, 'execution': execution,
                'contents': 'materialized public outputs, explicit samples and metadata; deferred evaluators and autodiff state are not restored',
                'files': {name: {'sha256': sha256(data), 'bytes': len(data)} for name, data in payloads.items()}}
    path = Path(path)
    # Exclusive directory creation refuses existing files, folders and symlinks.
    # The manifest is the completion marker and is always written last. An I/O
    # failure leaves an incomplete directory; it never deletes another run.
    path.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (path/name).open('xb') as output:
            output.write(data)
    with (path/'manifest.json').open('xb') as output:
        output.write(_json_bytes(manifest))
    return load_run(path)


def save_run(path, *, job, result, samples=None, include_source=True):
    """Save existing results to a new directory, without overwriting another run.

    The caller associates the result with the supplied Job. This function checks
    the solver type but cannot prove that an earlier calculation used these exact
    inputs. ``run_recorded`` performs and records the association itself.
    Backend arrays are copied to the host; save outside jit/grad. Private field
    evaluators are listed in ``deferred``; pass evaluated fields in ``samples``.
    """
    if not isinstance(job, Job):
        raise record_error('job must be a portable Job.', 'job')
    if type(include_source) is not bool:
        raise record_error('include_source must be a bool.', 'include_source')
    # Validate portable inputs before doing any filesystem writes.
    job.to_json()
    return _write(path, job=job, result=result, samples=samples,
                  source=capture_source(_PACKAGE, include_snapshot=include_source), env=environment(),
                  association='caller-supplied job/result; source and environment observed at save time')


def run_recorded(job, path, *, sample=None, include_source=True):
    """Execute a portable Job and save its observations with execution provenance.

    ``sample``, when provided, is a caller-owned function receiving the live
    result and returning a mapping, for example sampled fields with coordinates.
    It runs during creation only; archive loading never calls it.
    """
    if not isinstance(job, Job):
        raise record_error('job must be a portable Job.', 'job')
    if sample is not None and not callable(sample):
        raise record_error('sample must be callable.', 'sample')
    if type(include_source) is not bool:
        raise record_error('include_source must be a bool.', 'include_source')
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError(path)
    before_job = job.to_json()
    before = capture_source(_PACKAGE, include_snapshot=include_source)
    env = environment()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    result = run(job)
    samples = sample(result) if sample is not None else None
    elapsed = time.monotonic()-clock
    after = capture_source(_PACKAGE, include_snapshot=False)
    if job.to_json() != before_job or before[0]['identity'] != after[0]['identity'] or env != environment():
        raise record_error('Inputs, installed source or environment changed during execution; no run was saved.', 'execution')
    return _write(path, job=job, result=result, samples=samples, source=before, env=env,
                  association='executed by wavebit.records.run_recorded',
                  execution={'started_utc': started, 'seconds_including_sampling': elapsed,
                             'inputs_unchanged': True, 'source_unchanged': True})


def load_run(path):
    """Verify and read data, without executing a solver or archived Python code."""
    path = Path(path)
    manifest_path = path/'manifest.json'
    if manifest_path.is_symlink():
        raise record_error('Archive files must not be symbolic links.', 'manifest.json')
    try:
        manifest = _json_load(manifest_path.read_bytes(), 'manifest.json')
    except FileNotFoundError as exc:
        raise record_error('Missing completion manifest; this is not a complete run record.', str(path)) from exc
    if isinstance(manifest, dict) and manifest.get('schema') == 'wavebit.run/v1':
        raise record_error('This is a wavebit.run/v1 record, written before the public names were '
                           'renamed; a v1 record of either solver is refused, not converted. Build '
                           'its job again from current declarations and run it; see the '
                           'public-name migration guide, docs/migration/public-names.md.',
                           'manifest.schema')
    if not isinstance(manifest, dict) or manifest.get('schema') != _SCHEMA:
        raise record_error('Expected wavebit.run/v2.', 'manifest.schema')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'job.json','observations.json','arrays.npz'} <= files.keys() or not files.keys() <= _FILES:
        raise record_error('Unexpected or missing archive files.', 'manifest.files')
    payloads = {}
    for name, descriptor in files.items():
        target = path/name
        if target.is_symlink():
            raise record_error('Archive files must not be symbolic links.', name)
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise record_error('Missing or unreadable archive file.', name) from exc
        if not isinstance(descriptor, dict) or descriptor != {'sha256': sha256(data), 'bytes': len(data)}:
            raise record_error('Archive file checksum or size does not match.', name)
        payloads[name] = data
    job = Job.from_json(payloads['job.json'])
    if job.key != manifest.get('job_key') or job.solver != manifest.get('solver'):
        raise record_error('Job identity does not match the manifest.', 'job.json')
    source = manifest.get('source')
    if not isinstance(source, dict) or not isinstance(source.get('files'), dict):
        raise record_error('Missing source inventory.', 'manifest.source')
    for name, row in source['files'].items():
        member = PurePosixPath(name)
        if (not name.startswith('wavebit/') or '..' in member.parts or '\\' in name
                or member.as_posix() != name or not isinstance(row, dict)
                or set(row) != {'sha256', 'bytes'} or type(row['bytes']) is not int
                or row['bytes'] < 0 or not isinstance(row['sha256'], str)
                or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise record_error('Invalid source inventory entry.', 'manifest.source.files')
    if not source['files'] or source.get('identity') != source_identity(source['files']):
        raise record_error('Source identity differs from its inventory.', 'manifest.source.identity')
    if type(source.get('snapshot_included')) is not bool:
        raise record_error('Expected a boolean snapshot declaration.', 'manifest.source.snapshot_included')
    if source['snapshot_included'] != ('source.zip' in payloads):
        raise record_error('Source snapshot declaration does not match the files.', 'source.zip')
    if 'source.zip' in payloads:
        try:
            with zipfile.ZipFile(io.BytesIO(payloads['source.zip'])) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(source['files']):
                    raise record_error('Source inventory differs from its snapshot.', 'source.zip')
                for name in names:
                    if not name.startswith('wavebit/') or '..' in name.split('/') or '\\' in name:
                        raise record_error('Invalid source member name.', 'source.zip')
                    data = archive.read(name)
                    if source['files'][name] != {'sha256':sha256(data),'bytes':len(data)}:
                        raise record_error('Source checksum differs from its inventory.', name)
        except (zipfile.BadZipFile, KeyError) as exc:
            raise record_error('Invalid source snapshot.', 'source.zip') from exc
    tree = _json_load(payloads['observations.json'], 'observations.json')
    try:
        with np.load(io.BytesIO(payloads['arrays.npz']), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise record_error('Duplicate array names.', 'arrays.npz')
            arrays = {name: archive[name] for name in archive.files}
            if any(not isinstance(array, np.ndarray) for array in arrays.values()):
                raise record_error('Every numeric archive member must be a NumPy array.', 'arrays.npz')
    except (TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        raise record_error('Invalid numeric archive; pickled/object arrays are not supported.', 'arrays.npz') from exc
    used = set()
    restored = decode(tree, arrays, used)
    if used != set(arrays):
        raise record_error('Numeric archive contains unreferenced arrays.', 'arrays.npz')
    expected = {'outputs','samples','validity','diagnostics','provenance','extra','deferred'}
    if not isinstance(restored, Mapping) or set(restored) != expected:
        raise record_error('Unexpected result envelope.', 'observations.json')
    if any(not isinstance(restored[key], Mapping) for key in expected):
        raise record_error('Result envelope entries must contain named fields.', 'observations.json')
    if not isinstance(restored['validity'], RecordedObject):
        raise record_error('Missing validity declaration.', 'observations.json')
    return RunRecord(path.resolve(), job, manifest=MappingProxyType(manifest), **restored)


#: Names this module no longer has (fieldalgo/wavebit#57): each is a tombstone naming its
#: replacement, not an alias.
_RENAMED = {'RecordedFields': 'RecordedObject'}


def __getattr__(name):
    if name in _RENAMED:
        from ._common.declarations import removed
        raise removed('wavebit.records', name, f'wavebit.records.{_RENAMED[name]}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
