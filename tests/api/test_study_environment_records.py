"""Evidence auditors accept their two recorded checkout layouts, never extra imports."""
import hashlib
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('debye_stack', 'oscillator_stack', 'conductive_stack', 'inactive_lorentz')


@pytest.fixture(params=FAMILIES)
def auditor(request):
    family = request.param
    return importlib.import_module(f'studies.fdtd.{family}.fdtd_{family}_audit')


def test_old_root_and_new_ordered_src_root_records_are_accepted(auditor):
    root = auditor.ROOT
    assert auditor._same_project_pythonpath(str(root))
    assert auditor._same_project_pythonpath(os.pathsep.join((str(root / 'src'), str(root))))


@pytest.mark.parametrize('case', [
    'reverse_order', 'source_only', 'duplicate_root', 'extra_local', 'other_checkout',
    'empty_head', 'empty_tail', 'empty_value', 'not_a_string', 'escape',
])
def test_a_record_cannot_smuggle_another_import_directory(auditor, case):
    root = auditor.ROOT
    values = {
        'reverse_order': os.pathsep.join((str(root), str(root / 'src'))),
        'source_only': str(root / 'src'),
        'duplicate_root': os.pathsep.join((str(root), str(root))),
        'extra_local': os.pathsep.join((str(root / 'src'), str(root), str(root / 'tests'))),
        'other_checkout': os.pathsep.join((str(root / 'src'), str(root.parent / 'other'))),
        'empty_head': os.pathsep + str(root),
        'empty_tail': os.pathsep.join((str(root / 'src'), str(root))) + os.pathsep,
        'empty_value': '',
        'not_a_string': [str(root)],
        'escape': os.pathsep.join((str(root / '..'), str(root))),
    }
    assert not auditor._same_project_pythonpath(values[case])


@pytest.mark.parametrize('family', FAMILIES)
def test_original_windows_record_remains_valid_on_its_recorded_host(family, monkeypatch):
    path = ROOT / f'studies/fdtd/{family}/records/fdtd_{family}_targeted.json'
    raw = path.read_bytes()
    record = json.loads(raw)
    original = record['checks'][0]['environment']['PYTHONPATH']
    recorded_root = PureWindowsPath(original)
    assert recorded_root.is_absolute()
    # The retained evidence was recorded on Windows. Model that path parser
    # explicitly instead of interpreting its drive colon as a POSIX separator.
    auditor = importlib.import_module(f'studies.fdtd.{family}.fdtd_{family}_audit')
    monkeypatch.setattr(auditor, 'ROOT', recorded_root)
    monkeypatch.setattr(auditor, 'os', SimpleNamespace(pathsep=';'))

    def recorded_inside(value):
        candidate = recorded_root / PureWindowsPath(value)
        if not candidate.is_relative_to(recorded_root):
            raise ValueError('outside recorded checkout')
        return candidate

    monkeypatch.setattr(auditor, 'inside', recorded_inside)
    assert auditor._same_project_pythonpath(original)
    assert auditor._same_project_pythonpath(';'.join((str(recorded_root / 'src'), original)))
    assert not auditor._same_project_pythonpath(original + ';')
    assert not auditor._same_project_pythonpath(original + ';C:\\another-checkout')
    # This is an actual immutable historical fixture, not a regenerated record.
    immutable = json.loads((ROOT / 'tools/relocations.json').read_text())['immutable']
    entry, = [row for row in immutable.values() if row['target'] == path.relative_to(ROOT).as_posix()]
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
