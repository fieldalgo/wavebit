"""Every reference golden has a provenance record that matches it.

The record format and its audit live in ``tests/references/_reference_records.py``; the
tests below run the audit on every record in ``tests/references/`` and on synthetic records
that each break one rule. Nothing here reads a golden's numbers for physics: the tests that
consume each golden do that.
"""

import copy
import io
import json
import pathlib

import numpy as np
import pytest

from tests.api.test_docs_contract import tree
from tests.references import _reference_records as rr

ROOT = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = json.loads((ROOT / 'tools/third_party_licenses.json').read_text(encoding='utf-8'))
REFERENCES = 'tests/references/'


def _goldens():
    return sorted(p for p in tree() if p.startswith(REFERENCES + 'golden_') and
                  '/' not in p[len(REFERENCES):] and not p.endswith(rr.SUFFIX))


def _records():
    return sorted(p for p in tree() if p.startswith(REFERENCES) and p.endswith(rr.SUFFIX))


def _read(path):
    return (ROOT / path).read_bytes()


def test_every_golden_has_exactly_one_record_and_every_record_a_golden():
    goldens = {g.rsplit('.', 1)[0] for g in _goldens()}
    assert len(goldens) == len(_goldens()), 'two goldens share a stem'
    records = {r[:-len(rr.SUFFIX)] for r in _records()}
    assert sorted(goldens - records) == [], 'goldens without a provenance record'
    assert sorted(records - goldens) == [], 'records without a golden'


@pytest.mark.parametrize('path', _records())
def test_each_record_matches_its_golden(path):
    text = _read(path)
    record = json.loads(text.decode('utf-8'))
    file = record.get('payload', {}).get('file', '')
    golden = REFERENCES + file
    assert golden.rsplit('.', 1)[0] == path[:-len(rr.SUFFIX)], f'{path} describes {file}'
    payload = _read(golden) if golden in tree() else None
    errors = rr.audit(record, text, payload, inventory=INVENTORY, tree=tree(), read=_read)
    assert errors == [], '\n'.join([path, *errors])


def test_the_shared_digest_vector():
    # The same vector is checked by the other product that uses this record format.
    values = rr.values_digest({'x': np.array([0.1, 0.2], '<f8'), 'z': np.array([1 + 2j], '<c16')})
    assert values == '5b1e6695fa1d3df431519ed94c7096a82b5d1ca875872559f67e1cab86e1ce2e'
    assert rr.content_digest({'eps': {'real': 4.0, 'imag': -0.04}, 'modes': 2}, values) == (
        '63ad3eaeaed8bee68b8cc4ff62997c2fddd92f443327e07d926189d5fde7dea0')


# One good synthetic record; each probe below breaks one rule of it.

def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


PAYLOAD = _npz(R=np.array([0.25, 0.5]), T=np.array([0.75, 0.5]), version=np.array('1.0'))
TEST = b"G = 'golden_probe.npz'\n"
GENERATOR = b'print(1)\n'
TREE = {'tests/references/gen_goldens_probe.py', 'tests/references/test_probe_golden.py'}
FILES = {'tests/references/gen_goldens_probe.py': GENERATOR, 'tests/references/test_probe_golden.py': TEST}
TOOLS = {'external_tools': {'tool': {'spdx': 'GPL-2.0-or-later'}, 'other': {'spdx': 'MIT'},
                            'agpltool': {'spdx': 'AGPL-3.0-or-later'}}}


def _good():
    arrays = rr.payload_arrays('golden_probe.npz', PAYLOAD)[1]
    values = rr.values_digest(arrays)
    record = {
        'schema': rr.SCHEMA, 'case': 'probe', 'physics': 'rcwa', 'comparison': 'same-discretization',
        'role': 'external oracle',
        'references': [{
            'tool': 'tool', 'version': '1.0', 'build': None,
            'modified': {'value': False, 'description': None}, 'spdx': 'GPL-2.0-or-later',
            'source': 'https://example.org/tool', 'install': {'kind': 'pip', 'command': None},
            'settings': None, 'discretization': '3 orders', 'arrays': ['*'],
            'execution': 'separate-process',
            'independence': {'product_loaded_in_reference_process': False,
                             'reference_inputs_built_from_product_objects': False}}],
        'generator': {'path': 'tests/references/gen_goldens_probe.py',
                      'sha256': rr.file_digest('gen_goldens_probe.py', GENERATOR),
                      'command': 'python tests/references/gen_goldens_probe.py',
                      'product_commit': None, 'clean_tree': None},
        'generated_utc': '2026-10-28', 'environment': {'python': '3.12', 'numpy': None},
        'cost': {'not_measured': True}, 'inputs': {'case': 'probe'},
        'outcomes': {'not_run': []},
        'payload': {'file': 'golden_probe.npz', 'format': 'npz',
                    'file_sha256': rr.file_digest('golden_probe.npz', PAYLOAD),
                    'arrays': sorted(arrays)},
        'values_sha256': values, 'content_sha256': rr.content_digest({'case': 'probe'}, values),
        'consumers': ['tests/references/test_probe_golden.py'],
        'provenance_status': 'reconstructed', 'regeneration': [],
    }
    record['not_recorded'] = {path: 'probe' for path in rr.null_paths(record)}
    return record


def _audit(record=None, payload=PAYLOAD, text=None, inventory=TOOLS, files=FILES):
    record = _good() if record is None else record
    text = json.dumps(record).encode() if text is None else text
    return rr.audit(record, text, payload, inventory=inventory, tree=set(files), read=files.__getitem__)


def _restamped(record):
    record['not_recorded'] = {p: 'probe' for p in rr.null_paths({**record, 'not_recorded': {}})}
    return record


def test_the_good_record_passes():
    assert _audit() == []


def _change(edit):
    record = _good()
    edit(record)
    return record


def _probe(errors, phrase):
    assert any(phrase in e for e in errors), errors


def test_r_a_a_wrong_values_hash_fails():
    _probe(_audit(_change(lambda r: r.update(values_sha256='0' * 64))), 'values_sha256 does not match')


def test_r_b_a_one_ulp_payload_change_fails():
    moved = _npz(R=np.array([np.nextafter(0.25, 1), 0.5]), T=np.array([0.75, 0.5]), version=np.array('1.0'))
    errors = _audit(payload=moved)
    _probe(errors, 'file_sha256 does not match')
    _probe(errors, 'values_sha256 does not match')


def test_r_c_a_nan_fails_in_the_payload_and_in_the_record():
    _probe(_audit(payload=_npz(R=np.array([np.nan, 0.5]), T=np.array([0.75, 0.5]),
                               version=np.array('1.0'))), 'R is not finite')
    _probe(_audit(_change(lambda r: r['environment'].update(numpy=float('nan')))), 'not finite')


# Built at run time: the docs contract rejects a literal machine path in any tracked line.
@pytest.mark.parametrize('text', ['/' + 'home/someone/run', 'C' + ':\\work\\run', '~' + '/runs/x',
                                  '--input /' + 'tmp'])
def test_r_d_an_absolute_path_fails(text):
    _probe(_audit(_change(lambda r: r['references'][0].update(settings=text))), 'an absolute or home path')


def test_r_e_an_oversize_record_or_payload_fails():
    _probe(_audit(text=b' ' * (rr.MAX_RECORD + 1)), 'over')
    big = _npz(R=np.zeros(rr.MAX_PAYLOAD // 8 + 1), T=np.zeros(1), version=np.array('1.0'))
    _probe(_audit(payload=big), 'over')


def _regenerated(mode, reason='a reviewed reason', previous=None):
    def edit(r):
        r['regeneration'].append(dict(date='2026-10-28', mode=mode, reason=reason, tool_version='1.0',
                                      previous_content_sha256=previous or r['content_sha256'],
                                      review='pull request', approval='approved by the maintainer'))
    return _restamped(_change(edit))


def test_r_f_a_regeneration_without_a_reason_fails():
    assert _audit(_regenerated('restamp')) == []
    _probe(_audit(_regenerated('restamp', reason=' ')), 'states its reason')


def test_r_g_an_unlisted_null_field_fails():
    _probe(_audit(_change(lambda r: r['environment'].update(cpu=None))), 'null but not in not_recorded')
    _probe(_audit(_change(lambda r: r['not_recorded'].update({'case': 'x'}))), 'in not_recorded but not null')


def test_r_h_a_generator_sha_mismatch_fails():
    _probe(_audit(files=dict(FILES, **{'tests/references/gen_goldens_probe.py': b'print(2)\n'})),
           'generator sha256 does not match')
    _probe(_audit(files={'tests/references/test_probe_golden.py': TEST}), 'not in the repository')


def test_r_i_a_license_other_than_the_inventory_fails():
    _probe(_audit(_change(lambda r: r['references'][0].update(spdx='MIT'))), 'differs from the inventory')


def test_r_j_an_unknown_schema_fails():
    _probe(_audit(_change(lambda r: r.update(schema='wavebit.reference-values/2'))), 'unknown schema')


def test_r_k_the_vector_is_checked_by_the_shared_digest_test():
    # A wrong digest implementation changes the vector; the content digest covers the inputs too.
    assert rr.content_digest({'case': 'probe'}, 'x') != rr.content_digest({'case': 'other'}, 'x')
    _probe(_audit(_change(lambda r: r.update(inputs={'case': 'other'}))), 'content_sha256 does not match')


@pytest.mark.parametrize('key', ['gate', 'tolerance', 'rtol'])
def test_r_l_a_gate_key_fails(key):
    _probe(_audit(_change(lambda r: r['references'][0].update({key: 1e-9}))), 'never a')


def test_r_m_a_record_without_its_payload_fails():
    _probe(_audit(payload=None), 'is not in the repository')


def test_r_n_a_broken_regeneration_chain_fails():
    _probe(_audit(_regenerated('restamp', previous='1' * 64)), 'the content changed after it')
    _probe(_audit(_regenerated('rebaseline')), 'did not change')
    assert _audit(_regenerated('rebaseline', previous='1' * 64)) == []
    record = _regenerated('restamp')
    record['regeneration'].insert(0, dict(record['regeneration'][0], date='2026-11-01'))
    _probe(_audit(record), 'not in date order')


@pytest.mark.parametrize('text', ['see notes/plan.md', 'https://claude.ai/code/session_x', 'in wavebit-lab'])
def test_r_o_a_planning_path_session_url_or_private_repository_fails(text):
    errors = _audit(_change(lambda r: r['references'][0].update(settings=text)))
    assert any('holds' in e for e in errors), errors


def test_r_p_an_array_covered_twice_or_not_at_all_fails():
    second = dict(copy.deepcopy(_good()['references'][0]), tool='other', spdx='MIT', arrays=['R'])
    _probe(_audit(_change(lambda r: r['references'].append(second))), 'produced by 2 references')
    _probe(_audit(_change(lambda r: r['references'][0].update(arrays=['R', 'version']))),
           'T is produced by 0 references')


def test_r_q_an_agpl_tool_marked_modified_fails():
    record = _change(lambda r: r['references'][0].update(tool='agpltool', spdx='AGPL-3.0-or-later',
                                                        modified={'value': True, 'description': 'patched'}))
    _probe(_audit(record), 'unmodified upstream release')


def test_a_consumer_must_be_a_tracked_test_that_names_the_golden():
    _probe(_audit(_change(lambda r: r.update(consumers=[]))), 'no test reads')
    _probe(_audit(files=dict(FILES, **{'tests/references/test_probe_golden.py': b'pass\n'})),
           'does not name golden_probe')
    _probe(_audit(_change(lambda r: r.update(consumers=['tests/references/gen_goldens_probe.py']))),
           'not a tracked test module')
    assert rr.consumes("np.load(f'golden_probe{suffix}.npz')", 'golden_probe_coarse')
    assert not rr.consumes("np.load(f'golden_{name}.npz')", 'golden_probe_coarse')
