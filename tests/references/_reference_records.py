"""The provenance record kept beside each reference golden, and its audit.

Each ``golden_<x>.npz`` or ``golden_<x>.json`` in this directory has a
``golden_<x>.reference.json`` beside it, in the ``wavebit.reference-values/1`` format: which
tool made the values, at which version and build, with which generator and environment, how
it ran, and which tests read them. A record written after the fact says so
(``provenance_status: reconstructed``); every field nobody recorded is null and named in
``not_recorded`` with the source that was consulted. The hashes lock today's values: a golden
changed without a reviewed ``regeneration`` entry fails ``tests/api/test_reference_values.py``.

``audit(record, ...)`` lists everything wrong with one record; the test calls it on every
record here and on synthetic records that must fail.
"""

import fnmatch
import hashlib
import io
import json
import math
import re

import numpy as np

SCHEMA = 'wavebit.reference-values/1'
SUFFIX = '.reference.json'
MAX_RECORD = 1 << 20
MAX_PAYLOAD = 1 << 20    # 512 KiB until 2026-10-28; raised by the maintainer for a 743 KB golden
COMPARISONS = ('same-discretization', 'different-discretization', 'structure-oracle')
EXECUTIONS = ('separate-process', 'in-process', 'unknown')
STATUSES = ('complete', 'reconstructed')
REGENERATION_MODES = ('restamp', 'rebaseline')
PRIVATE_GENERATOR = 'private research repository'
FORBIDDEN_KEYS = {'gate', 'gates', 'tolerance', 'tolerances', 'atol', 'rtol'}
# A string may name a repository path, a URL or a version, never a machine, a private record or
# a session: an absolute, home or drive path, a company planning path, a session link or a
# private repository's name.
FORBIDDEN_TEXT = [
    (re.compile(r'(?:^|[\s(\'"`=])(?:/(?:home|Users|root|tmp|opt|var|private)\b|~/|[A-Za-z]:[\\/])'),
     'an absolute or home path'),
    (re.compile(r'(?:^|[\s(\'"`/])(?:decisions|work|notes)/[\w.-]'), 'a planning-record path'),
    (re.compile(r'claude\.ai/code/session_'), 'a session URL'),
    (re.compile(r'\b(?:wavebit|openphysics)-lab\b'), 'a private repository name'),
]
REQUIRED = {
    'schema': str, 'case': str, 'physics': str, 'comparison': str, 'role': str,
    'references': list, 'generator': dict, 'generated_utc': (str, type(None)),
    'environment': dict, 'cost': dict, 'inputs': dict, 'outcomes': dict, 'payload': dict,
    'values_sha256': (str, type(None)), 'content_sha256': str, 'consumers': list,
    'provenance_status': str, 'not_recorded': dict, 'regeneration': list,
}
REFERENCE_KEYS = {'tool', 'version', 'build', 'modified', 'spdx', 'source', 'install', 'settings',
                  'discretization', 'arrays', 'execution', 'independence'}
GENERATOR_KEYS = {'path', 'sha256', 'command', 'product_commit', 'clean_tree'}
REGENERATION_KEYS = {'date', 'mode', 'reason', 'previous_content_sha256', 'tool_version', 'review',
                     'approval'}


def values_digest(arrays):
    """SHA-256 over named arrays: name, NUL, dtype, NUL, shape, NUL, C-order bytes, by name.

    The same digest as ``_fd_stage0_cases.content_digest``; a ``metadata`` entry is skipped.
    """
    h = hashlib.sha256()
    for key in sorted(arrays):
        if key == 'metadata':
            continue
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(b'\0' + a.dtype.str.encode() + b'\0' + repr(a.shape).encode() + b'\0')
        h.update(a.tobytes())
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      ensure_ascii=False).encode('utf-8')


def content_digest(inputs, values_sha256):
    """SHA-256 of the canonical JSON of the inputs and the values digest."""
    return hashlib.sha256(canonical({'inputs': inputs, 'values_sha256': values_sha256})).hexdigest()


def file_digest(name, data):
    """SHA-256 of a payload's bytes; a text payload is hashed with LF line ends."""
    if not name.endswith('.npz'):
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def payload_arrays(name, data):
    """The payload's value names and, for an archive, its arrays (None for a JSON payload)."""
    if name.endswith('.npz'):
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return sorted(arrays), arrays
    return sorted(json.loads(data.decode('utf-8'))), None


def null_paths(value, prefix=''):
    """Dotted paths of every null leaf; list items are indexed."""
    if value is None:
        return {prefix}
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found |= null_paths(item, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found |= null_paths(item, f'{prefix}[{index}]')
    return found


def _walk(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f'{path}.{key}' if path else key, key, item
            yield from _walk(item, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield f'{path}[{index}]', None, item
            yield from _walk(item, f'{path}[{index}]')


def consumes(source, stem):
    """Whether a test's source names the golden: its stem, or an f-string prefix of it."""
    if stem in source:
        return True
    for prefix in re.findall(r"""f['"][^'"{]*?(golden_\w*)\{""", source):
        if stem.startswith(prefix) and len(prefix) > len('golden_'):
            return True
    return False


def audit(record, text, payload, *, inventory, tree, read):
    """Everything wrong with one record.

    record is the parsed record and text its bytes; payload is the golden's bytes or None;
    inventory is tools/third_party_licenses.json; tree is the set of tracked paths and read
    returns a tracked file's bytes.
    """
    errors = []
    if len(text) > MAX_RECORD:
        errors.append(f'the record is {len(text)} bytes, over {MAX_RECORD}')
    if record.get('schema') != SCHEMA:
        return errors + [f'unknown schema {record.get("schema")!r}; expected {SCHEMA}']
    for key, kind in REQUIRED.items():
        if key not in record:
            errors.append(f'no {key}')
        elif not isinstance(record[key], kind):
            errors.append(f'{key} has the wrong type')
    if errors:
        return errors
    for path, key, value in _walk(record):
        if key is not None and key.lower() in FORBIDDEN_KEYS:
            errors.append(f'{path}: a record holds values, never a {key!r}; the test owns its gate')
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f'{path}: not finite')
        if isinstance(value, str):
            for pattern, what in FORBIDDEN_TEXT:
                if pattern.search(value):
                    errors.append(f'{path}: holds {what}')
    if record['comparison'] not in COMPARISONS:
        errors.append(f'comparison must be one of {COMPARISONS}')
    if record['provenance_status'] not in STATUSES:
        errors.append(f'provenance_status must be one of {STATUSES}')

    nulls = null_paths(record)
    if missing := nulls - set(record['not_recorded']):
        errors.append(f'null but not in not_recorded: {sorted(missing)}')
    if stale := set(record['not_recorded']) - nulls:
        errors.append(f'in not_recorded but not null: {sorted(stale)}')
    for key, why in record['not_recorded'].items():
        if not isinstance(why, str) or not why.strip():
            errors.append(f'not_recorded.{key}: name the source that was consulted')

    # A derived golden also names Wavebit itself, under its own license.
    tools = dict(inventory['external_tools'])
    if 'first_party_license' in inventory:
        tools['wavebit'] = {'spdx': inventory['first_party_license']}
    for index, reference in enumerate(record['references']):
        at = f'references[{index}]'
        if not isinstance(reference, dict) or set(reference) != REFERENCE_KEYS:
            errors.append(f'{at}: needs exactly the keys {sorted(REFERENCE_KEYS)}')
            continue
        tool = reference['tool']
        if tool not in tools:
            errors.append(f'{at}: {tool!r} has no entry in the third-party inventory')
            continue
        if reference['spdx'] != tools[tool]['spdx']:
            errors.append(f"{at}: license {reference['spdx']!r} differs from the inventory's "
                          f"{tools[tool]['spdx']!r}")
        modified = reference['modified']
        if not isinstance(modified, dict) or set(modified) != {'value', 'description'}:
            errors.append(f'{at}.modified needs value and description')
        elif 'AGPL' in tools[tool]['spdx'] and modified['value'] is not False:
            errors.append(f'{at}: an AGPL tool is used only as the unmodified upstream release')
        if reference['execution'] not in EXECUTIONS:
            errors.append(f'{at}.execution must be one of {EXECUTIONS}')
        if not reference['arrays'] or not all(isinstance(g, str) for g in reference['arrays']):
            errors.append(f'{at}.arrays must list the value names it produced')

    generator = record['generator']
    if set(generator) != GENERATOR_KEYS:
        errors.append(f'generator needs exactly the keys {sorted(GENERATOR_KEYS)}')
    elif generator['path'] is not None and generator['path'] != PRIVATE_GENERATOR:
        if generator['path'] not in tree:
            errors.append(f"generator {generator['path']} is not in the repository")
        elif generator['sha256'] is not None and generator['sha256'] != file_digest(
                generator['path'], read(generator['path'])):
            errors.append(f"generator sha256 does not match {generator['path']}")

    previous_dates = []
    for index, entry in enumerate(record['regeneration']):
        at = f'regeneration[{index}]'
        if not isinstance(entry, dict) or set(entry) != REGENERATION_KEYS:
            errors.append(f'{at}: needs exactly the keys {sorted(REGENERATION_KEYS)}')
            continue
        if entry['mode'] not in REGENERATION_MODES:
            errors.append(f'{at}.mode must be one of {REGENERATION_MODES}')
        if not isinstance(entry['reason'], str) or not entry['reason'].strip():
            errors.append(f'{at}: a regeneration states its reason')
        previous_dates.append(entry['date'] or '')
    if previous_dates != sorted(previous_dates):
        errors.append('regeneration entries are not in date order')
    if record['regeneration'] and isinstance(record['regeneration'][-1], dict):
        last = record['regeneration'][-1]
        same = last.get('previous_content_sha256') == record['content_sha256']
        if last.get('mode') == 'restamp' and not same:
            errors.append('the last regeneration is a restamp, but the content changed after it')
        if last.get('mode') == 'rebaseline' and same:
            errors.append('the last regeneration is a rebaseline, but the content did not change')

    file = record['payload'].get('file')
    if payload is None:
        return errors + [f'the payload {file} is not in the repository']
    if len(payload) > MAX_PAYLOAD:
        errors.append(f'the payload is {len(payload)} bytes, over {MAX_PAYLOAD}')
    if record['payload'].get('file_sha256') != file_digest(file, payload):
        errors.append(f'file_sha256 does not match {file}')
    names, arrays = payload_arrays(file, payload)
    if record['payload'].get('arrays') != names:
        errors.append('payload.arrays does not list the values the payload holds')
    if arrays is not None:
        for name, array in arrays.items():
            if array.dtype.kind in 'fc' and not np.isfinite(array).all():
                errors.append(f'{file}: {name} is not finite')
        if record['values_sha256'] != values_digest(arrays):
            errors.append('values_sha256 does not match the payload')
    elif record['values_sha256'] is not None:
        errors.append('a nested JSON payload has no values digest; values_sha256 must be null')
    if record['content_sha256'] != content_digest(record['inputs'], record['values_sha256']):
        errors.append('content_sha256 does not match inputs and values_sha256')
    for name in names:
        owners = [r['tool'] for r in record['references'] if isinstance(r, dict) and
                  any(fnmatch.fnmatchcase(name, g) for g in r.get('arrays') or [])]
        if len(owners) != 1:
            errors.append(f'{name} is produced by {len(owners)} references, not one: {owners}')

    stem = file.rsplit('.', 1)[0]
    if not record['consumers']:
        errors.append('no test reads these values; name the consumers')
    for consumer in record['consumers']:
        if not re.fullmatch(r'tests/(?:.+/)?test_[^/]+\.py', consumer) or consumer not in tree:
            errors.append(f'consumer {consumer} is not a tracked test module')
        elif consumer.endswith('test_reference_values.py'):
            errors.append('the audit is not a consumer')
        elif not consumes(read(consumer).decode('utf-8'), stem):
            errors.append(f'consumer {consumer} does not name {stem}')
    return errors
