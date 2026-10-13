"""Recorded XLA controls govern reuse and reach the actual pytest process."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout'
    (root / 'tests').mkdir(parents=True)
    (root / 'tools').mkdir()
    shutil.copy2(ROOT / 'tools/run_verification.py', root / 'tools/run_verification.py')
    shutil.copytree(ROOT / 'tools/_verification', root / 'tools/_verification',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'pytest.ini').write_text('[pytest]\nmarkers =\n    research: research\n')
    (root / 'tests/test_environment.py').write_text(textwrap.dedent('''
        import json
        import os
        from pathlib import Path

        def test_receives_xla_flags():
            path = Path('artifacts/xla-flags-seen.json')
            path.parent.mkdir(exist_ok=True)
            seen = json.loads(path.read_text()) if path.exists() else []
            seen.append(os.environ.get('XLA_FLAGS'))
            path.write_text(json.dumps(seen))
    '''))
    return root


def invoke(root, output, flags, *arguments):
    environment = dict(os.environ)
    if flags is None:
        environment.pop('XLA_FLAGS', None)
    else:
        environment['XLA_FLAGS'] = flags
    command = [sys.executable, str(root / 'tools/run_verification.py'),
               '--root', str(root), '--profile', 'full', '--output', str(output),
               *arguments]
    return subprocess.run(command, cwd=root, env=environment, text=True,
                          capture_output=True, timeout=60)


@pytest.mark.parametrize(('flags', 'changed'), [
    (None, ''),
    ('', None),
    ('--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1',
     '--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1'),
], ids=['unset', 'empty', 'explicit-options'])
def test_xla_flags_are_inherited_recorded_and_required_for_resume(checkout, flags, changed):
    first = checkout / 'artifacts/first'
    result = invoke(checkout, first, flags)
    assert result.returncode == 0, result.stdout + result.stderr
    inputs = json.loads((first / 'inputs.json').read_text())
    receipt = json.loads((first / 'receipt.json').read_text())
    assert 'XLA_FLAGS' in inputs['environment']['controls']
    assert inputs['environment']['controls']['XLA_FLAGS'] == flags
    expected_hash = hashlib.sha256(json.dumps(
        inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert receipt['fingerprint'] == expected_hash
    node = 'tests/test_environment.py::test_receives_xla_flags'
    assert receipt['counts'] == {'passed': 1}
    assert receipt['reused_nodeids'] == []
    seen = checkout / 'artifacts/xla-flags-seen.json'
    assert json.loads(seen.read_text()) == [flags]

    second = checkout / 'artifacts/second'
    result = invoke(checkout, second, flags, '--resume', str(first))
    assert result.returncode == 0, result.stdout + result.stderr
    resumed = json.loads((second / 'receipt.json').read_text())
    assert resumed['fingerprint'] == expected_hash
    assert resumed['reused_nodeids'] == [node]
    assert resumed['counts'] == {'passed': 1}
    assert [phase['name'] for phase in resumed['phases']] == ['collection']
    assert json.loads(seen.read_text()) == [flags]

    refused = checkout / 'artifacts/changed'
    result = invoke(checkout, refused, changed, '--resume', str(second))
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Resume refused' in result.stderr and 'environment' in result.stderr
    assert not refused.exists()
    assert json.loads(seen.read_text()) == [flags]
