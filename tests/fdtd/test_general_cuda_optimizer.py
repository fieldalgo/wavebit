"""Same-leaf optimizer reuse against discrete basis derivatives and exact lifetime checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('optimizer_reuse_tested',
    ROOT/'studies/fdtd/general_cuda/optimizer_reuse.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_basis_gradient_matches_independent_complex_step_residual():
    # A literal two-channel basis, independent of the study's Gram formula.
    basis = np.array([[.3, 1.2, -.7], [.9, -.4, 1.1]])
    scale = np.sum(basis*basis)
    gram = basis@basis.T/scale
    target = study.TARGET[0]*(np.cos(study.TARGET[1])*basis[0]
                              +np.sin(study.TARGET[1])*basis[1])
    def direct(theta):
        value = theta[0]*(np.cos(theta[1])*basis[0]+np.sin(theta[1])*basis[1])-target
        return np.sum(value*value)/(2*scale)
    for seed in study.SEEDS:
        theta = np.array(seed)
        value, gradient = study.oracle(theta, gram)
        np.testing.assert_allclose(value, direct(theta), rtol=2e-12, atol=0.)
        measured = []
        for component in (0, 1):
            perturbed = theta.astype(complex)
            perturbed[component] += 1e-25j
            measured.append(direct(perturbed).imag/1e-25)
        np.testing.assert_allclose(gradient, measured, rtol=2e-12, atol=0.)
    for seed in (0, 1):
        rows = study.trajectory(seed, gram)
        assert len(rows) == 16
        for first, second in zip(rows[:-1], rows[1:]):
            assert first['after'] == second['before']


def test_source_snapshot_preserves_original_bytes_and_rejects_drift(tmp_path, monkeypatch):
    root = tmp_path/'checkout'
    root.mkdir()
    source = root/'source.py'
    source.write_bytes(b'original source\r\n')
    monkeypatch.setattr(study, 'source_files', lambda root: {'source.py': source})
    output = tmp_path/'attempt'
    report = study.begin(root, output, {'mode': 'test'})
    source.write_bytes(b'changed source\n')
    with pytest.raises(RuntimeError, match='source changed'):
        study.finish(root, output, report)
    receipt = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'failed' and receipt['inputs_unchanged'] is False
    assert (output/'source/source.py').read_bytes() == b'original source\r\n'


def test_changed_basis_artifact_invalidates_run_receipt(tmp_path, monkeypatch):
    root = tmp_path/'checkout'
    root.mkdir()
    source = root/'source.py'
    source.write_bytes(b'fixed source')
    monkeypatch.setattr(study, 'source_files', lambda root: {'source.py': source})
    reference = tmp_path/'basis'
    reference.mkdir()
    receipt = reference/'receipt.json'
    receipt.write_bytes(b'{}')
    artifact = reference/'basis.npz'
    artifact.write_bytes(b'original basis')
    output = tmp_path/'attempt'
    report = study.begin(root, output, {'mode': 'test'})
    report['reference'] = dict(receipt=str(receipt),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        basis_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    artifact.write_bytes(b'replaced basis')
    with pytest.raises(RuntimeError, match='reference artifact changed'):
        study.finish(root, output, report)
    saved = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'failed' and saved['inputs_unchanged'] is False
    assert saved['reference_unchanged'] is False


@pytest.mark.parametrize('stage', ['missing', 'identity', 'copy'])
def test_initial_audit_failure_always_has_failed_attempt(tmp_path, monkeypatch, stage):
    root = tmp_path/'checkout'
    root.mkdir()
    source = root/'source.py'
    source.write_bytes(b'source')
    output = tmp_path/'attempt'
    monkeypatch.setattr(study, 'source_files', lambda root: {'source.py': source})
    if stage == 'missing':
        source.unlink()
    elif stage == 'identity':
        def failed_identity(root):
            initial = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
            assert initial['status'] == 'running'
            raise PermissionError('source read denied')
        monkeypatch.setattr(study, 'identity', failed_identity)
    else:
        original_write = Path.write_bytes
        def failed_copy(path, data):
            if path.is_relative_to(output/'source'):
                initial = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
                assert initial['status'] == 'running'
                raise PermissionError('snapshot copy denied')
            return original_write(path, data)
        monkeypatch.setattr(Path, 'write_bytes', failed_copy)
    with pytest.raises(OSError):
        study.begin(root, output, {'mode': 'test'})
    saved = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'failed' and saved['inputs_unchanged'] is False
    assert saved['source_audit_complete'] is False
    assert saved['audit_errors'][0]['phase'] == 'initial_source_snapshot'
    assert saved['failure']['type'] in ('FileNotFoundError', 'PermissionError')


@pytest.mark.parametrize('original_failure', [False, True])
@pytest.mark.parametrize('damage', ['missing_source', 'unreadable_source', 'missing_reference', 'unreadable_basis'])
def test_final_audit_failure_preserves_original_exception_and_records_audit(
        tmp_path, monkeypatch, original_failure, damage):
    root = tmp_path/'checkout'
    root.mkdir()
    source = root/'source.py'
    source.write_bytes(b'source')
    monkeypatch.setattr(study, 'source_files', lambda root: {'source.py': source})
    output = tmp_path/'attempt'
    report = study.begin(root, output, {'mode': 'test'})
    reference = tmp_path/'basis'
    reference.mkdir()
    receipt = reference/'receipt.json'
    receipt.write_bytes(b'{}')
    artifact = reference/'basis.npz'
    artifact.write_bytes(b'basis')
    report['reference'] = dict(receipt=str(receipt),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        basis_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    if damage == 'missing_source':
        source.unlink()
    elif damage == 'missing_reference':
        receipt.unlink()
    else:
        denied = source if damage == 'unreadable_source' else artifact
        original_read = Path.read_bytes
        def failed_read(path):
            if path == denied:
                raise PermissionError('audit read denied')
            return original_read(path)
        monkeypatch.setattr(Path, 'read_bytes', failed_read)
    primary = RuntimeError('original solver failure')
    if original_failure:
        with pytest.raises(RuntimeError) as caught:
            try:
                raise primary
            except RuntimeError as error:
                study.finish(root, output, report, error)
                raise
        assert caught.value is primary
    else:
        with pytest.raises(OSError):
            study.finish(root, output, report)
    saved = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'failed' and saved['inputs_unchanged'] is False
    phase = 'final_source_identity' if damage.endswith('source') else 'final_reference_identity'
    assert saved['audit_errors'][0]['phase'] == phase
    if original_failure:
        assert saved['failure']['message'] == 'original solver failure'
        assert saved['failure']['type'] == 'RuntimeError'
    else:
        assert saved['failure']['type'] in ('FileNotFoundError', 'PermissionError')


def test_unwritable_final_receipt_does_not_replace_original_exception(tmp_path, monkeypatch):
    root = tmp_path/'checkout'
    root.mkdir()
    source = root/'source.py'
    source.write_bytes(b'source')
    monkeypatch.setattr(study, 'source_files', lambda root: {'source.py': source})
    output = tmp_path/'attempt'
    report = study.begin(root, output, {'mode': 'test'})
    def denied(*args):
        raise PermissionError('receipt volume unavailable')
    monkeypatch.setattr(study, 'write_json', denied)
    primary = RuntimeError('original solver failure')
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        except RuntimeError as error:
            study.finish(root, output, report, error)
            raise
    assert caught.value is primary
    assert any('receipt volume unavailable' in note for note in primary.__notes__)


@pytest.mark.parametrize('message,accepted', [
    ('source phase must be finite', True), ('NaN source amplitude', True),
    ('unsupported source dtype', False),
])
def test_invalid_source_gate_requires_finiteness_related_refusal(message, accepted):
    from wavebit.errors import ValidationError
    error = ValidationError(message, code='invalid_input', path='sources.d.phase')
    if accepted:
        assert study.finite_refusal(error)['message'] == message
    else:
        with pytest.raises(AssertionError, match='nonfinite input'):
            study.finite_refusal(error)


@pytest.mark.parametrize('argv', [
    ['--mode', 'run'], ['--mode', 'basis', '--device', 'cuda:0'],
    ['--mode', 'basis', '--block', '7'], ['--mode', 'run', '--reference', 'basis.json', '--block', '7'],
    ['--mode', 'run', '--reference', 'basis.json', '--seed', '2'],
])
def test_cli_refuses_unqualified_or_incomplete_requests(tmp_path, argv):
    with pytest.raises(SystemExit):
        study.arguments([*argv, '--output-dir', str(tmp_path/'attempt')])


def test_cli_refuses_raw_output_inside_product_checkout():
    with pytest.raises(SystemExit):
        study.arguments(['--mode', 'basis', '--output-dir', str(ROOT/'artifacts/optimizer')])


@pytest.fixture(scope='module')
def basis_reference(tmp_path_factory):
    pytest.importorskip('torch')
    return study.build_basis(ROOT, tmp_path_factory.mktemp('optimizer-basis')/'attempt')


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('device,block', [('cpu', None), ('cuda:0', None), ('cuda:0', 7)],
                         ids=['cpu-plain', 'cuda-plain', 'cuda-k7'])
def test_same_leaf_sgd_against_basis_recurrence_and_lifetime(tmp_path, basis_reference, seed, device, block):
    torch = pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('requires CUDA; no CPU substitution')
    report = study.run(ROOT, tmp_path/'attempt', basis_reference, device=device, block=block, seed=seed)
    assert report['status'] == 'passed' and report['inputs_unchanged'] is True
    assert report['leaf_identity_preserved'] is True
    assert len(report['iterations']) == 16 and len(report['negative_recovery']) == 2
    assert report['iterations'][1]['old_outputs_unchanged'] is True
    assert report['versions_after_recovery'] == [18, 18]
    if device.startswith('cuda'):
        assert report['final_cleanup']['allocated'] == report['process_baseline_allocated']
        assert {row['after_release']['allocated'] for row in report['iterations'][1:]} == {
            report['plateau_baseline']}
