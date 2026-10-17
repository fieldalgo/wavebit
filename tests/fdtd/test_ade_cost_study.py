"""CPU-only independent oracle and preflight contract checks for ade_cost."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'ade_cost_tested', Path(__file__).resolve().parents[2]/'studies/fdtd/general_cuda/ade_cost.py')
cost = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cost
spec.loader.exec_module(cost)


def test_roots_matrix_and_two_centered_secants():
    time_scale = 1e-6/299792458.
    dt = .5*.05/np.sqrt(3)*time_scale
    result = cost.oracle(8, dt, time_scale)
    np.testing.assert_allclose(result['final_e'], result['record'][-1], rtol=2e-12, atol=0.)
    values = result['record']
    residual = values[2:]-(result['g']-result['drive']/cost.EPSILON)*values[1:-1] + result['h']*values[:-2] - cost.AMPLITUDE*(1-result['g']+result['h'])
    assert np.max(abs(residual)) < 1e-12
    for step in (1e-4, 5e-5):
        upper = cost.oracle(8, dt, time_scale, parameter=cost.PARAMETER+step)['value']
        lower = cost.oracle(8, dt, time_scale, parameter=cost.PARAMETER-step)['value']
        np.testing.assert_allclose((upper-lower)/(2*step), result['gradients'][1], rtol=2e-6, atol=0.)
    larger = cost.oracle(32, dt, time_scale)
    np.testing.assert_allclose(larger['value'], 64*result['value'], rtol=2e-15, atol=0.)
    np.testing.assert_allclose(larger['gradients'], 64*result['gradients'], rtol=2e-15, atol=0.)


def reference_pair():
    config = dict(device='cuda:0', size=64, checkpoint_steps=8,
                  steps=163, repeat=5, warmup=1, profile=False)
    report = dict(config=config, script_sha256='instrument', source_sha256={'poles': 'source'},
        fixture_identity={'kind': 'homogeneous-lorentz'},
        environment=dict(torch='torch', numpy='numpy', cuda_runtime='cuda', gpu='gpu',
                         gpu_total_bytes=8*cost.GIB, precision='float64'))
    reference = copy.deepcopy(report)
    reference.update(status='passed', source_integrity=dict(unchanged=True), samples=[dict(checks_passed=True,
        retained=dict(peak_reserved_bytes=64*2**20)) for _ in range(5)])
    reference['config']['size'] = 32
    return reference, report


def test_empirical_forecast_cell_scaling_and_margin():
    reference, report = reference_pair()
    result = cost.forecast(reference, report)
    assert result['forecast_reserved_bytes'] == 64*2**20*12
    assert result['headroom_bytes'] == cost.GIB


@pytest.mark.parametrize('mutation', ['block', 'size', 'failed', 'four', 'source',
                                      'environment', 'unchecked', 'too_large', 'conductor', 'drift'])
def test_forecast_rejects_nonmatching_or_unsafe_evidence(mutation):
    reference, report = reference_pair()
    if mutation == 'block':
        reference['config']['checkpoint_steps'] = None
    elif mutation == 'size':
        reference['config']['size'] = 16
    elif mutation == 'failed':
        reference['status'] = 'failed'
    elif mutation == 'four':
        reference['samples'].pop()
    elif mutation == 'source':
        reference['source_sha256']['poles'] = 'different'
    elif mutation == 'environment':
        reference['environment']['torch'] = 'different'
    elif mutation == 'unchecked':
        reference['samples'][0]['checks_passed'] = False
    elif mutation == 'drift':
        reference['source_integrity']['unchanged'] = False
    elif mutation == 'too_large':
        reference['samples'][0]['retained']['peak_reserved_bytes'] = cost.GIB
    else:
        reference['fixture_identity']['kind'] = 'conductor'
    with pytest.raises(ValueError):
        cost.forecast(reference, report)


@pytest.mark.parametrize('extra', [
    ['--device', 'cpu', '--size', '32'],
    ['--device', 'cpu', '--size', '8', '--checkpoint-steps', '8'],
    ['--size', '8'], ['--size', '64'], ['--size', '32', '--forecast-from', 'receipt.json'],
])
def test_cli_rejects_out_of_scope_configurations(tmp_path, extra):
    with pytest.raises(SystemExit):
        cost.arguments(['--source-root', str(tmp_path/'source'), '--output-dir', str(tmp_path/'output'),
                        *extra])
