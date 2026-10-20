"""Public pulse-band Hessians against independently forced discrete Yee fields.

The small periodic trajectories test the discrete source and derivative contract,
not radiation accuracy or continuum convergence. Bandwidth changes the envelope
width and its starting peak time; the delay baseline remains fixed. All four
leaves are dimensionless coordinates with documented physical SI rescalings.
"""
from functools import lru_cache

import numpy as np
import pytest

from tests.fdtd import _pulse_yee_reference as reference
from wavebit import fdtd
from wavebit.errors import IncompatibleProblemError, ValidationError
from wavebit.units import Z0


CASES = [('cpu', None), ('cuda:0', None), ('cuda:0', 7)]
CASE_IDS = ['cpu_plain', 'gpu_plain', 'gpu_k7']
COMPONENTS = tuple(f+c for f in ('E', 'H') for c in 'xyz')
WEIGHTS = np.array([1., .7, 1.3, .6, 1.1, .9])
RTOL = 2e-12


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def runtime(device):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    return torch


@lru_cache(maxsize=8)
def exact(point, stamp, steps):
    return reference.reference(point, stamp=stamp, steps=steps)


def objective(result, group, torch):
    if group == 'drive':
        return (result.source_references['d']['drive']/reference.MOMENT).square().mean()
    fields = torch.cat((result.final_fields.values['E'], Z0*result.final_fields.values['H']))
    records = torch.stack([result[c].values*(Z0 if c[0] == 'H' else 1.) for c in COMPONENTS])
    w = torch.as_tensor(WEIGHTS, dtype=torch.float64, device=fields.device)
    time_weights = torch.as_tensor(1+.2*np.arange(records.shape[1])/records.shape[1],
                                   dtype=torch.float64, device=fields.device)
    return ((w[:, None, None, None]*fields.square()).mean()
            + .25*(w[:, None]*time_weights*records.square()).mean())


def public_tensors(result):
    return (*result.final_fields.values.values(), *(result[c].values for c in COMPONENTS),
            *(result.source_references['d'][k] for k in ('drive', 'field', 'delivered_power')))


def scaled_error(value, expected, scale):
    assert np.isfinite(value).all() and np.isfinite(expected).all() and scale > 0
    return float(np.max(abs(value-expected), initial=0.)/scale)


def check_forward(result, expected, steps, device):
    assert bool(result.validity.checks['runtime_stability'])
    assert result.provenance['actual_steps'] == steps
    errors = []
    for field in ('E', 'H'):
        actual = result.final_fields.values[field]
        assert str(actual.device) == device and str(actual.dtype) == 'torch.float64'
        wanted = expected['fields'][field]
        assert tuple(actual.shape) == wanted.shape
        errors.append(scaled_error(host(actual), wanted, np.max(abs(wanted))))
    for field in ('E', 'H'):
        # Ey excitation makes Hy vanish. Normalize the whole field group, so
        # an erroneous nonzero Hy is still tested without dividing by zero.
        scale = max(np.max(abs(expected['records'][field+c])) for c in 'xyz')
        for c in 'xyz':
            name = field+c
            actual = result[name].values
            assert str(actual.device) == device and str(actual.dtype) == 'torch.float64'
            assert tuple(actual.shape) == (steps,)
            errors.append(scaled_error(host(actual), expected['records'][name], scale))
    history = result.source_references['d']
    errors.append(scaled_error(host(history['drive']), expected['drive'], reference.MOMENT))
    dt = result.provenance['time_step_seconds']
    np.testing.assert_allclose(dt, reference.DT, rtol=2e-14, atol=0.)
    n = np.arange(steps)
    np.testing.assert_allclose(host(history['times_seconds']), (n+.5)*dt, rtol=2e-14, atol=0.)
    np.testing.assert_allclose(host(history['field_times_seconds']), (n+1)*dt, rtol=2e-14, atol=0.)
    for name in COMPONENTS:
        shift = 1 if name[0] == 'E' else .5
        np.testing.assert_allclose(host(result[name].times_seconds), (n+shift)*dt, rtol=2e-14, atol=0.)
    previous = np.r_[0., host(history['field'])[:-1]]
    power = -host(history['drive'])*(host(history['field'])+previous)/2
    np.testing.assert_allclose(host(history['delivered_power']), power, rtol=1e-13, atol=0.)
    assert max(errors) < RTOL
    return max(errors)


@pytest.mark.parametrize('device,block', CASES, ids=CASE_IDS)
@pytest.mark.parametrize('steps', [17, 41], ids=['n17', 'n41'])
@pytest.mark.parametrize('stamp', ['bspline', 'tent'])
def test_pulse_band_full_hessians_and_same_leaf_reuse(device, block, steps, stamp,
                                                    record_property):
    torch = runtime(device)
    problem, policy = reference.fixture(reference.POINTS[0], stamp=stamp, steps=steps, block=block)
    prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == steps
    assert prepared.contract['resolved_checkpoint_steps'] == block
    # Wavelength and phase begin on CPU; bandwidth and delay on execution device.
    # Lowering must propagate every derivative back to its original leaf owner.
    leaves = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True,
                               device='cpu' if i in (0, 2) else device)
                   for i,v in enumerate(reference.POINTS[0]))
    identities = tuple((id(v), v.data_ptr(), v.device) for v in leaves)
    held = []
    first = None
    for index, point in enumerate((*reference.POINTS, reference.POINTS[0])):
        with torch.no_grad():
            for leaf, value in zip(leaves, point):
                leaf.fill_(value)
        versions = tuple(v._version for v in leaves)
        live, _ = reference.fixture(leaves, stamp=stamp, steps=steps, block=block)
        assert prepared.validate(live).ok
        result = prepared.run(live)
        expected = exact(tuple(point), stamp, steps)
        forward_error = check_forward(result, expected, steps, device)
        analytic = reference.objectives(expected)
        derivative_tensors = []
        for group_index, group in enumerate(('drive', 'fields_records')):
            loss = objective(result, group, torch)
            gradients = torch.autograd.grad(loss, leaves, create_graph=True, retain_graph=True)
            rows = [torch.autograd.grad(g, leaves,
                        retain_graph=(group_index == 0 or row_index < 3))
                    for row_index,g in enumerate(gradients)]
            actual = (float(host(loss)), np.array([float(host(g)) for g in gradients]),
                      np.array([[float(host(g)) for g in row] for row in rows]))
            wanted = analytic[group]
            assert wanted[0] > 0
            assert np.all(abs(wanted[1])/wanted[0] > 1e-6)
            assert np.all(abs(wanted[2])/wanted[0] > 1e-6)
            for got, target in zip(actual, wanted):
                np.testing.assert_allclose(got, target, rtol=RTOL, atol=0.)
            np.testing.assert_allclose(actual[2], actual[2].T, rtol=RTOL, atol=0.)
            for derivatives in (gradients, *rows):
                for leaf, derivative in zip(leaves, derivatives):
                    assert derivative.device == leaf.device and derivative.dtype == leaf.dtype
                    assert leaf.grad is None
                derivative_tensors.extend(derivatives)
            record_property(f'point_{index}_{group}_value_gradient_hessian',
                            repr(tuple(np.asarray(v).tolist() for v in actual)))
        assert tuple((id(v), v.data_ptr(), v.device) for v in leaves) == identities
        assert tuple(v._version for v in leaves) == versions
        np.testing.assert_array_equal([float(host(v)) for v in leaves], point)
        for old in held:
            for value, saved in old:
                assert torch.equal(value, saved)
        current = (*public_tensors(result), *derivative_tensors)
        held.append([(v, v.detach().clone()) for v in current])
        if first is None:
            first = tuple(host(v).copy() for v in public_tensors(result))
        elif index == 2:
            for value, saved in zip(public_tensors(result), first):
                np.testing.assert_array_equal(host(value), saved)
        record_property(f'point_{index}_maximum_forward_scaled_error', forward_error)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu_units', 'gpu_units'])
def test_pulse_physical_um_nm_fresh_preparations(device):
    torch = runtime(device)
    results = []
    point = reference.POINTS[1]
    expected = exact(tuple(point), 'bspline', 17)
    with torch.no_grad():
        for unit in ('um', 'nm'):
            problem, policy = reference.fixture(point, stamp='bspline', steps=17, unit=unit)
            result = fdtd.prepare(problem, numerics=policy, backend='torch', device=device).run(problem)
            check_forward(result, expected, 17, device)
            results.append(result)
    for field in ('E', 'H'):
        left, right = (r.final_fields.values[field] for r in results)
        assert scaled_error(host(right), host(left), np.max(abs(host(left)))) < RTOL
        scale = max(np.max(abs(host(results[0][field+c].values))) for c in 'xyz')
        for c in 'xyz':
            left, right = (r[field+c].values for r in results)
            assert scaled_error(host(right), host(left), scale) < RTOL
    for name in ('drive', 'field', 'delivered_power'):
        left, right = (r.source_references['d'][name] for r in results)
        assert scaled_error(host(right), host(left), np.max(abs(host(left)))) < RTOL


@pytest.mark.parametrize('change', ['cutoff', 'center', 'array_band', 'periods'])
def test_contract_static_pulse_dependencies_refuse_before_execution(monkeypatch, change):
    import wavebit.fdtd._api as api
    runtime('cpu')
    problem, policy = reference.fixture(reference.POINTS[0])
    if change == 'periods':
        policy = policy.replace(duration=fdtd.Duration(periods=.2, source='d'))
    prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device='cpu')
    source = problem.sources[0]
    if change == 'cutoff': source = source.replace(pulse=source.pulse.replace(cutoff=1e-4))
    elif change == 'center': source = source.replace(center=(.014, -.021, .007))
    elif change == 'array_band':
        source = source.replace(pulse=source.pulse.replace(wavelength=np.array([2.])))
    else: source = source.replace(pulse=source.pulse.replace(wavelength=1.94))
    changed = problem.replace(sources=(source,))
    monkeypatch.setattr(api, 'execute', lambda *a, **k: pytest.fail('incompatible input stepped'))
    assert not prepared.validate(changed).ok
    error = ValidationError if change == 'array_band' else IncompatibleProblemError
    with pytest.raises(error) as caught:
        prepared.run(changed)
    if change in ('cutoff', 'center'):
        assert caught.value.issue.code == 'prepared_structure' and caught.value.issue.path == 'sources'
    elif change == 'periods':
        assert caught.value.issue.code == 'prepared_dependencies'
        assert caught.value.issue.path == 'numerics.duration'
    else:
        assert 'scalar' in str(caught.value)
