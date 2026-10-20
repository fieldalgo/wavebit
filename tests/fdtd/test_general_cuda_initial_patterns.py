"""CUDA initial patterns stay static; explicit initial amplitude remains live."""
import contextlib
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fdtd.test_general_cuda import closed_problem, fixed_numerics
from wavebit import fdtd
from wavebit.errors import ValidationError
from wavebit.units import EPSILON0

STEPS = 17
RTOL = 2e-12


def runtime(device):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('actual CUDA required')
    return torch


def callback_problem(leaf, torch):
    def pattern(x, y, z):
        return leaf*torch.ones(x.shape, dtype=torch.float64)
    return closed_problem().replace(initial=fdtd.InitialField(
        E=(0., pattern, 0.), H=None, amplitude=1.))


def verify(prepared, problem, leaf, torch):
    version = leaf._version
    result = prepared.run(problem)
    electric = result.final_fields.values['E'][1]
    loss = electric.square().mean()
    gradient, = torch.autograd.grad(loss, leaf)
    dt = result.provenance['time_step_seconds']
    r = 600.*dt/(2*EPSILON0*2.25)
    q = (1-r)/(1+r)
    amplitude = float(leaf.detach().cpu())
    expected = (amplitude**2*q**(2*STEPS), 2*amplitude*q**(2*STEPS))
    actual = (float(loss.detach().cpu()), float(gradient.detach().cpu()))
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.)
    np.testing.assert_allclose(electric.detach().cpu(), amplitude*q**STEPS,
                               rtol=RTOL, atol=0.)
    assert leaf.is_leaf and leaf.requires_grad and leaf.grad is None
    assert leaf._version == version and gradient.device == leaf.device
    assert bool(result.validity.checks['runtime_stability'])
    return result, actual


def test_cpu_callable_initial_pattern_keeps_original_gradient_contract(record_property):
    torch = runtime('cpu')
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    problem = callback_problem(leaf, torch)
    prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS), backend='torch')
    _, actual = verify(prepared, problem, leaf, torch)
    record_property('exact_conductor_loss_gradient', json.dumps(actual))


@pytest.mark.parametrize('block', [None, 7], ids=['gpu_plain', 'gpu_k7'])
def test_cuda_callable_carried_pattern_is_refused_before_conversion(block, record_property):
    torch = runtime('cuda:0')
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    problem = callback_problem(leaf, torch)
    policy = fixed_numerics(STEPS).replace(checkpoint_steps=block)
    try:
        prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device='cuda:0')
        first, actual = verify(prepared, problem, leaf, torch)
    except ValidationError as exc:
        assert exc.issue.path.startswith('initial')
        assert any(word in str(exc).lower() for word in ('static', 'pattern', 'carried'))
        return
    saved = first.final_fields.values['E'].detach().clone()
    observations = [actual]
    for amplitude in (.7, 1.2):
        with torch.no_grad():
            leaf.fill_(amplitude)
        current, actual = verify(prepared, problem, leaf, torch)
        observations.append(actual)
        torch.testing.assert_close(first.final_fields.values['E'], saved, rtol=0, atol=0)
    torch.testing.assert_close(current.final_fields.values['E'], saved, rtol=0, atol=0)
    record_property('unexpected_pattern_gradient_bypass', json.dumps(observations))
    print({'pattern_leaf_device': str(leaf.device), 'loss_gradient': observations,
           'same_leaf_recovery': True, 'old_outputs_unchanged': True}, flush=True)
    pytest.fail('CUDA callable initial pattern admitted a carried CPU leaf')


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
@pytest.mark.parametrize('kind', ['numpy_callback', 'torch_callback', 'whole_callback',
                                  'numpy_array', 'torch_array'])
def test_static_host_patterns_keep_live_amplitude(device, kind):
    torch = runtime(device)
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    calls = []

    def pattern(x, y, z):
        calls.append(x.shape)
        if kind == 'torch_callback':
            return torch.ones(x.shape, dtype=torch.float64)
        return np.ones(x.shape)

    if kind.endswith('array'):
        value = np.zeros((3, 8, 8, 8))
        value[1] = 1.
        if kind == 'torch_array':
            value = torch.as_tensor(value)
    else:
        value = pattern if kind == 'whole_callback' else (0., pattern, 0.)
    problem = closed_problem().replace(initial=fdtd.InitialField(E=value, H=None, amplitude=leaf))
    prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS), backend='torch', device=device)
    assert calls == []
    saved = None
    for amplitude in (1.2, .7, 1.2):
        with torch.no_grad():
            leaf.fill_(amplitude)
        before = len(calls)
        result, _ = verify(prepared, problem, leaf, torch)
        expected_calls = 0 if kind.endswith('array') else 3 if kind == 'whole_callback' else 1
        assert len(calls)-before == expected_calls
        assert result.final_fields.values['E'].device == torch.device(device)
        if saved is None:
            first = result
            saved = result.final_fields.values['E'].detach().clone()
        torch.testing.assert_close(first.final_fields.values['E'], saved, rtol=0, atol=0)
    torch.testing.assert_close(result.final_fields.values['E'], saved, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['component', 'whole', 'array'])
@pytest.mark.parametrize('output', ['host_live', 'gpu_live', 'gpu_constant'])
def test_gpu_pattern_values_refused_and_policy_recovers(kind, output):
    torch = runtime('cuda:0')
    device = 'cpu' if output == 'host_live' else 'cuda:0'
    leaf = torch.tensor(1., dtype=torch.float64, device=device,
                        requires_grad=output.endswith('live'))
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return leaf*torch.ones(x.shape, dtype=torch.float64, device=device)

    value = ((0., pattern, 0.) if kind == 'component' else pattern if kind == 'whole'
             else leaf*torch.ones((3, 8, 8, 8), dtype=torch.float64, device=device))
    problem = closed_problem().replace(initial=fdtd.InitialField(E=value, H=None))
    with pytest.raises(ValidationError) as caught:
        fdtd.run(problem, numerics=fixed_numerics(STEPS), backend='torch', device='cuda:0')
    assert caught.value.issue.path.startswith('initial')
    assert caught.value.issue.code in ('prepared.static_input', 'execution.cuda_scope')
    if kind != 'array':
        assert calls == [1]
    # The exception must not leave the stricter policy enabled for ordinary CPU AD.
    cpu_leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    cpu_problem = callback_problem(cpu_leaf, torch)
    prepared = fdtd.prepare(cpu_problem, numerics=fixed_numerics(STEPS), backend='torch')
    verify(prepared, cpu_problem, cpu_leaf, torch)


def evaluation_fixture():
    from wavebit.fdtd._prepare import fields
    from wavebit.fdtd._runtime import runtime as make_runtime
    prepared = fdtd.prepare(closed_problem(), numerics=fixed_numerics(STEPS), backend='torch')
    return fields, prepared._plan.grid, make_runtime('torch', device='cpu')


@pytest.mark.parametrize('kind', ['component', 'whole', 'array'])
def test_cpu_policy_rejects_before_conversion_without_host_reads(kind, monkeypatch):
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return leaf

    value = ((None, pattern, None) if kind == 'component' else pattern if kind == 'whole'
             else leaf.expand(3, 8, 8, 8))

    def forbidden(*args, **kwargs):
        raise AssertionError('pattern rejection must precede conversion or host reads')

    monkeypatch.setattr(rt, 'array', forbidden)
    for name in ('cpu', 'numpy', 'item'):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with fields.static_host_patterns(True), pytest.raises(ValidationError) as caught:
        fields.evaluate('E', value, grid, 'E', rt, 'float64')
    assert caught.value.issue.code == 'prepared.static_input'
    assert caught.value.issue.path == 'initial.E'
    assert calls == ([] if kind == 'array' else [1])


def test_cpu_policy_nested_false_exception_reset_and_public_cpu_isolation():
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    value = leaf.expand(3, 8, 8, 8)
    with fields.static_host_patterns(True):
        with fields.static_host_patterns(False):
            result = fields.evaluate('E', value, grid, 'E', rt, 'float64')
            gradient, = torch.autograd.grad(result.sum(), leaf)
            assert gradient == 3*8**3
        with pytest.raises(ValidationError):
            fields.evaluate('E', value, grid, 'E', rt, 'float64')
        problem = callback_problem(leaf, torch)
        prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS), backend='torch')
        verify(prepared, problem, leaf, torch)
        with pytest.raises(ValidationError):
            fields.evaluate('E', value, grid, 'E', rt, 'float64')
    with pytest.raises(RuntimeError, match='sentinel'), fields.static_host_patterns(True):
        raise RuntimeError('sentinel')
    # Legacy/internal evaluation defaults to the original differentiable behavior.
    result = fields.evaluate('E', value, grid, 'E', rt, 'float64')
    gradient, = torch.autograd.grad(result.sum(), leaf)
    assert gradient == 3*8**3


@pytest.mark.parametrize('transform', ['forward', 'jacfwd', 'vmap'])
def test_cpu_transformed_callback_rejected_without_host_reads(transform, monkeypatch):
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError('transformed pattern must not be read on host')

    for name in ('cpu', 'numpy', 'item'):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    def evaluate(value):
        with fields.static_host_patterns(True):
            return fields.evaluate('E', (None, lambda x, y, z: value, None),
                                   grid, 'E', rt, 'float64')

    with pytest.raises(ValidationError) as caught:
        if transform == 'forward':
            with torch.autograd.forward_ad.dual_level():
                evaluate(torch.autograd.forward_ad.make_dual(
                    torch.tensor(1.), torch.tensor(1.)))
        elif transform == 'jacfwd':
            torch.func.jacfwd(evaluate)(torch.tensor(1.))
        else:
            torch.vmap(evaluate)(torch.ones(2))
    assert caught.value.issue.code == 'prepared.static_input'


def test_cpu_jax_pattern_ad_unchanged_and_scoped_tracer_refused():
    pytest.importorskip('torch')
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from wavebit.fdtd._prepare import fields
    from wavebit.fdtd._runtime import runtime as make_runtime
    prepared = fdtd.prepare(closed_problem(), numerics=fixed_numerics(STEPS), backend='jax')
    rt = make_runtime('jax')

    def total(value, policy):
        with fields.static_host_patterns(policy):
            return fields.evaluate('E', (None, lambda x, y, z: value, None),
                                   prepared._plan.grid, 'E', rt, 'float64').sum()

    assert jax.grad(lambda value: total(value, False))(1.) == 8**3
    with pytest.raises(ValidationError) as caught:
        jax.jit(lambda value: total(value, True))(1.)
    assert caught.value.issue.code == 'prepared.static_input'


def test_gpu_same_callback_output_policy_recovery_and_old_results():
    torch = runtime('cuda:0')
    amplitude = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    current = [torch.ones((8, 8, 8), dtype=torch.float64)]
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return current[0]

    problem = closed_problem().replace(initial=fdtd.InitialField(
        E=(0., pattern, 0.), H=None, amplitude=amplitude))
    prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS),
                            backend='torch', device='cuda:0')
    first, _ = verify(prepared, problem, amplitude, torch)
    saved = first.final_fields.values['E'].detach().clone()
    for invalid in (torch.ones((8, 8, 8), dtype=torch.float64, requires_grad=True),
                    torch.ones((8, 8, 8), dtype=torch.float64, device='cuda:0')):
        current[0] = invalid
        before = len(calls)
        with pytest.raises(ValidationError) as caught:
            prepared.run(problem)
        assert caught.value.issue.code in ('prepared.static_input', 'execution.cuda_scope')
        assert caught.value.issue.path == 'initial.E'
        assert len(calls) == before+1
        current[0] = np.ones((8, 8, 8))
        result, _ = verify(prepared, problem, amplitude, torch)
        torch.testing.assert_close(result.final_fields.values['E'], saved, rtol=0, atol=0)
        torch.testing.assert_close(first.final_fields.values['E'], saved, rtol=0, atol=0)


def test_cpu_callback_nested_legacy_and_public_policy_reset(monkeypatch):
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()
    from wavebit.fdtd import _api
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    values = leaf.expand(3, 8, 8, 8)
    events = []
    prepared = fdtd.prepare(closed_problem(), numerics=fixed_numerics(STEPS), backend='torch')
    # Only the public policy dispatch is simulated; this control never creates a CUDA runtime.
    fake_execution = SimpleNamespace(backend='torch', device='cuda:0',
                                     _scope=contextlib.nullcontext)
    nested = dataclasses.replace(prepared, execution=fake_execution)
    monkeypatch.setattr(_api.Prepared, '_lower', lambda self, problem: (None, {}))

    def execute(*args, **kwargs):
        events.append('nested-public-cuda')
        fields.evaluate('E', values, grid, 'E', rt, 'float64')
        raise AssertionError('nested public CUDA policy was not enabled')

    monkeypatch.setattr(_api, 'execute', execute)

    def callback(x, y, z):
        internal = fields.evaluate('E', values, grid, 'E', rt, 'float64')
        assert internal.requires_grad
        events.append('legacy')
        with pytest.raises(ValidationError) as caught:
            nested.run(None)
        assert caught.value.issue.code == 'prepared.static_input'
        # Nested failure restores the callback's legacy policy.
        assert fields.evaluate('E', values, grid, 'E', rt, 'float64').requires_grad
        return leaf

    with fields.static_host_patterns(True), pytest.raises(ValidationError) as caught:
        fields.evaluate('E', (None, callback, None), grid, 'E', rt, 'float64')
    assert caught.value.issue.code == 'prepared.static_input'
    assert events == ['legacy', 'nested-public-cuda']
    assert fields.evaluate('E', values, grid, 'E', rt, 'float64').requires_grad


def test_cpu_callback_exception_list_output_and_decay_contract():
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()
    leaf = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    failure = RuntimeError('callback sentinel')

    def broken(x, y, z):
        raise failure

    with fields.static_host_patterns(True):
        with pytest.raises(RuntimeError) as caught:
            fields.evaluate('E', (None, broken, None), grid, 'E', rt, 'float64')
        assert caught.value is failure
        with pytest.raises(ValidationError) as caught:
            fields.evaluate('E', (None, lambda x, y, z: [leaf], None), grid, 'E', rt, 'float64')
        assert caught.value.issue.code == 'prepared.static_input'
    # The original Decay concrete-only restriction takes precedence on CPU too.
    with pytest.raises(ValueError, match='DecayConcreteOnly'):
        fields.evaluate('E', (None, lambda x, y, z: leaf, None), grid, 'E', rt,
                        'float64', concrete_only=True)


@pytest.mark.parametrize('mode', ['no_grad', 'inference'])
def test_gpu_magnetic_callback_raw_carrier_refused_before_unit_scaling(mode, record_property):
    torch = runtime('cuda:0')
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    version = leaf._version
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return leaf

    policy = fixed_numerics(STEPS)
    context = torch.no_grad if mode == 'no_grad' else torch.inference_mode
    # The identical raw E callback must refuse even when reverse recording is off.
    electric = closed_problem().replace(initial=fdtd.InitialField(
        E=(0., pattern, 0.), H=None, amplitude=1.))
    with context(), pytest.raises(ValidationError) as caught:
        fdtd.run(electric, numerics=policy, backend='torch', device='cuda:0')
    assert caught.value.issue.code == 'prepared.static_input'
    assert caught.value.issue.path == 'initial.E'
    assert calls == [1]
    calls.clear()
    magnetic = closed_problem().replace(initial=fdtd.InitialField(
        E=None, H=(0., pattern, 0.), amplitude=1.))
    try:
        with context():
            result = fdtd.run(magnetic, numerics=policy, backend='torch', device='cuda:0')
    except ValidationError as exc:
        assert exc.issue.code == 'prepared.static_input'
        assert exc.issue.path == 'initial.H'
        assert calls == [1]
        assert leaf.is_leaf and leaf.requires_grad and leaf._version == version
        assert leaf.grad is None
        return
    # A uniform magnetic field has zero curl and no magnetic conductivity here.
    field = result.final_fields.values['H'][1].detach().cpu().numpy()
    np.testing.assert_allclose(field, 1.2, rtol=RTOL, atol=0.)
    assert calls == [1] and leaf._version == version and leaf.grad is None
    observation = {'mode': mode, 'magnetic_field_min': float(field.min()),
                   'magnetic_field_max': float(field.max()),
                   'runtime_stability': bool(result.validity.checks['runtime_stability']),
                   'raw_leaf_requires_grad': leaf.requires_grad,
                   'returned_field_requires_grad': result.final_fields.values['H'].requires_grad}
    record_property('unexpected_magnetic_pattern_bypass', json.dumps(observation))
    print(observation, flush=True)
    pytest.fail('CUDA H callback admitted a carried raw value after internal unit scaling')


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
@pytest.mark.parametrize('mode', ['no_grad', 'inference'])
def test_magnetic_static_callback_live_amplitude_after_forward_warmup(device, mode):
    torch = runtime(device)
    amplitude = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return torch.ones(x.shape, dtype=torch.float64)

    problem = closed_problem().replace(initial=fdtd.InitialField(
        E=None, H=(0., pattern, 0.), amplitude=amplitude))
    prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS),
                            backend='torch', device=device)
    assert not calls
    context = torch.no_grad if mode == 'no_grad' else torch.inference_mode
    with context():
        warmup = prepared.run(problem)
    assert calls == [1]
    saved = warmup.final_fields.values['H'].detach().clone()
    for value in (1.2, .7, 1.2):
        with torch.no_grad():
            amplitude.fill_(value)
        version = amplitude._version
        before = len(calls)
        result = prepared.run(problem)
        field = result.final_fields.values['H']
        gradient, = torch.autograd.grad(field[1].square().mean(), amplitude)
        np.testing.assert_allclose(field[1].detach().cpu(), value, rtol=RTOL, atol=0.)
        np.testing.assert_allclose(gradient.detach().cpu(), 2*value, rtol=RTOL, atol=0.)
        assert field.device == torch.device(device) and gradient.device == amplitude.device
        assert amplitude.is_leaf and amplitude.requires_grad and amplitude.grad is None
        assert amplitude._version == version and len(calls) == before+1
        torch.testing.assert_close(warmup.final_fields.values['H'], saved, rtol=0, atol=0)
    torch.testing.assert_close(field, saved, rtol=0, atol=0)


def test_cpu_magnetic_callback_preserves_pattern_derivative():
    torch = runtime('cpu')
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return leaf

    problem = closed_problem().replace(initial=fdtd.InitialField(
        E=None, H=(0., pattern, 0.), amplitude=1.))
    prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS), backend='torch')
    assert not calls
    result = prepared.run(problem)
    field = result.final_fields.values['H'][1]
    gradient, = torch.autograd.grad(field.square().mean(), leaf)
    np.testing.assert_allclose(field.detach(), 1.2, rtol=RTOL, atol=0.)
    np.testing.assert_allclose(gradient.detach(), 2.4, rtol=RTOL, atol=0.)
    assert calls == [1] and leaf.grad is None and leaf._version == 0


def test_cpu_jax_public_magnetic_callback_preserves_pattern_derivative():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    calls = []

    def loss(value):
        def pattern(x, y, z):
            calls.append(1)
            return value
        problem = closed_problem().replace(initial=fdtd.InitialField(
            E=None, H=(0., pattern, 0.), amplitude=1.))
        prepared = fdtd.prepare(problem, numerics=fixed_numerics(STEPS), backend='jax')
        before = len(calls)
        result = prepared.run(problem)
        assert len(calls) == before+1
        return jnp.mean(result.final_fields.values['H'][1]**2)

    value, gradient = jax.value_and_grad(loss)(1.2)
    np.testing.assert_allclose(value, 1.2**2, rtol=RTOL, atol=0.)
    np.testing.assert_allclose(gradient, 2.4, rtol=RTOL, atol=0.)
    assert calls == [1]


@pytest.mark.parametrize('mode', ['no_grad', 'inference'])
def test_cpu_magnetic_raw_guard_precedes_scaling_and_host_reads(mode, monkeypatch):
    torch = runtime('cpu')
    fields, grid, rt = evaluation_fixture()
    from wavebit.fdtd._lower_public import _scaled_initial
    from wavebit.units import Z0
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    calls = []

    def pattern(x, y, z):
        calls.append(1)
        return leaf

    scaled = _scaled_initial(pattern, Z0)

    def forbidden(*args, **kwargs):
        raise AssertionError('raw magnetic rejection must precede host reads and conversion')

    monkeypatch.setattr(rt, 'array', forbidden)
    for name in ('cpu', 'numpy', 'item'):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    context = torch.no_grad if mode == 'no_grad' else torch.inference_mode
    with context(), fields.static_host_patterns(True), pytest.raises(ValidationError) as caught:
        fields.evaluate('H', (None, scaled, None), grid, 'H', rt, 'float64')
    assert caught.value.issue.code == 'prepared.static_input'
    assert caught.value.issue.path == 'initial.H' and calls == [1]
