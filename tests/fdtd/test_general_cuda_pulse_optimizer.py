"""Public same-leaf projected pulse fitting against independent discrete residual derivatives.

Twelve fixed projected SGD updates test a waveform fit, not unique parameter recovery or a
continuum inverse problem. CUDA cases additionally check live allocation lifetime.
"""
import gc
import json
import weakref

import numpy as np
import pytest

from tests.fdtd import _pulse_fit_reference as fit
from tests.fdtd import _pulse_yee_reference as pulse
from tests.fdtd.test_general_cuda_pulse_higher_order import check_forward, host, public_tensors
from wavebit import fdtd
from wavebit.units import Z0

CASES = [('cpu', None), ('cuda:0', None), ('cuda:0', 7)]
CASE_IDS = ['cpu_plain', 'gpu_plain', 'gpu_k7']
RTOL = 2e-12


def _targets(torch, device, target):
    values = np.concatenate((target['fields']['E'], Z0*target['fields']['H']))
    records = np.stack([target['records'][key]*(Z0 if key[0] == 'H' else 1.)
                        for key in pulse.LABELS])
    tensor = lambda value: torch.tensor(value, dtype=torch.float64, device=device)
    return (tensor(values), tensor(records), tensor(pulse.FIELD_WEIGHTS),
            tensor(pulse.time_weights(fit.STEPS)))


def _loss(result, targets, normalization, torch):
    target_fields, target_records, weights, times = targets
    fields = torch.cat((result.final_fields.values['E'], Z0*result.final_fields.values['H']))
    records = torch.stack([result[key].values*(Z0 if key[0] == 'H' else 1.)
                           for key in pulse.LABELS])
    return ((weights[:, None]*times*(records-target_records).square()).mean()
            + .25*(weights[:, None, None, None]*(fields-target_fields).square()).mean())/normalization


def _snapshot_owned(values):
    return [(id(value), value.data_ptr(), value.device, value.dtype, tuple(value.shape),
             tuple(value.stride()), value._version, host(value).copy()) for value in values]


def _check_owned(values, snapshots):
    for value, saved in zip(values, snapshots, strict=True):
        assert (id(value), value.data_ptr(), value.device, value.dtype, tuple(value.shape),
                tuple(value.stride()), value._version) == saved[:-1]
        np.testing.assert_array_equal(host(value), saved[-1])
        assert not value.requires_grad and value.grad_fn is None


def _step(prepared, leaves, optimizer, targets, normalization, row,
          block, device, torch):
    """Function scope releases all live autograd owners before the caller checks."""
    before = np.array([float(host(value)) for value in leaves])
    assert fit.in_bounds(before)
    actual_oracle = fit.oracle(tuple(before))
    expected = actual_oracle['expected']
    versions = tuple(value._version for value in leaves)
    live, _ = pulse.fixture(leaves, stamp='bspline', steps=fit.STEPS, block=block)
    assert prepared.validate(live).ok
    result = prepared.run(live)
    forward_error = check_forward(result, expected, fit.STEPS, device)
    loss = _loss(result, targets, normalization, torch)
    gradient = torch.autograd.grad(loss, leaves, create_graph=True)
    directional = sum(value*float(direction) for value, direction in zip(gradient, fit.DIRECTION))
    hvp = torch.autograd.grad(directional, leaves)
    actual = dict(value=float(host(loss)), gradient=np.array([float(host(v)) for v in gradient]),
                  hvp=np.array([float(host(v)) for v in hvp]))
    for key in ('value', 'gradient', 'hvp'):
        np.testing.assert_allclose(actual[key], actual_oracle[key], rtol=RTOL, atol=0.)
        np.testing.assert_allclose(actual[key], row[key], rtol=RTOL, atol=0.)
    np.testing.assert_allclose(before, row['before'], rtol=RTOL, atol=0.)
    assert tuple(value._version for value in leaves) == versions
    for leaf, g, h in zip(leaves, gradient, hvp):
        assert g.device == h.device == leaf.device and g.dtype == h.dtype == leaf.dtype
        assert leaf.grad is None
    values = (*public_tensors(result), loss, *gradient, *hvp)
    refs = [weakref.ref(value) for value in values]
    owned = tuple(value.detach().clone() for value in values)
    addresses = {(value.device, value.data_ptr()) for value in values}
    assert all((value.device, value.data_ptr()) not in addresses for value in owned)
    for leaf, value in zip(leaves, gradient):
        leaf.grad = value.detach().clone()
    optimizer.step()
    proposal = np.array([float(host(value)) for value in leaves])
    np.testing.assert_allclose(proposal, row['proposal'], rtol=RTOL, atol=0.)
    # Projection is an optimizer operation, outside the differentiated loss.
    with torch.no_grad():
        for leaf, bounds in zip(leaves, fit.BOUNDS):
            leaf.clamp_(float(bounds[0]), float(bounds[1]))
    optimizer.zero_grad(set_to_none=True)
    assert all(leaf.grad is None for leaf in leaves)
    assert tuple(value._version for value in leaves) == tuple(v+2 for v in versions)
    after = np.array([float(host(value)) for value in leaves])
    assert fit.in_bounds(after)
    np.testing.assert_allclose(after, row['after'], rtol=RTOL, atol=0.)
    metrics = dict(before=before.tolist(), proposal=proposal.tolist(), after=after.tolist(), value=actual['value'],
                   gradient=actual['gradient'].tolist(), hvp=actual['hvp'].tolist(),
                   maximum_forward_scaled_error=forward_error)
    return owned, refs, metrics


def _final(prepared, leaves, targets, normalization, expected, device, block, torch):
    with torch.no_grad():
        live, _ = pulse.fixture(leaves, stamp='bspline', steps=fit.STEPS, block=block)
        result = prepared.run(live)
        check_forward(result, expected, fit.STEPS, device)
        return float(host(_loss(result, targets, normalization, torch)))


@pytest.mark.parametrize('device,block', CASES, ids=CASE_IDS)
@pytest.mark.parametrize('seed', [0, 1], ids=['seed0', 'seed1'])
def test_pulse_fitting_sgd_and_hvp_same_leaf_reuse(device, block, seed, record_property):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    cuda = device.startswith('cuda')
    if cuda and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    plan = fit.plan()
    trajectory = fit.trajectory(seed)
    problem, policy = pulse.fixture(fit.SEEDS[seed], stamp='bspline', steps=fit.STEPS, block=block)
    prepared = fdtd.prepare(problem, numerics=policy, backend='torch', device=device)
    assert prepared.contract['resolved_steps'] == 41
    assert prepared.contract['resolved_checkpoint_steps'] == block
    if block is not None:
        assert divmod(fit.STEPS, block) == (5, 6)
    leaves = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True,
                               device='cpu' if i in (0, 2) else device)
                   for i, value in enumerate(fit.SEEDS[seed]))
    identities = tuple((id(v), v.data_ptr(), v.device, v.dtype) for v in leaves)
    optimizer = torch.optim.SGD([{'params': [leaf], 'lr': float(rate)}
                                 for leaf, rate in zip(leaves, plan['rates'])], foreach=False)
    targets = _targets(torch, device, plan['target'])
    if cuda:
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
    else:
        baseline = None
    held, snapshots = (), []
    live_allocations, metrics = [], []
    try:
        for row in trajectory['rows']:
            owned, refs, measured = _step(prepared, leaves, optimizer, targets, plan['normalization'],
                                          row, block, device, torch)
            assert tuple((id(v), v.data_ptr(), v.device, v.dtype) for v in leaves) == identities
            _check_owned(held, snapshots)
            assert all(value.data_ptr() != old.data_ptr() for value in owned for old in held
                       if value.device == old.device)
            held = owned
            del owned
            snapshots = _snapshot_owned(held)
            gc.collect()
            if cuda:
                torch.cuda.synchronize()
            assert all(ref() is None for ref in refs), 'live result/derivative graph retained after SGD step'
            if cuda:
                live_allocations.append(torch.cuda.memory_allocated())
            metrics.append(measured)
        assert len(metrics) == fit.UPDATES == 12
        final_point = tuple(float(host(value)) for value in leaves)
        np.testing.assert_allclose(final_point, trajectory['final_point'], rtol=RTOL, atol=0.)
        final_oracle = fit.oracle(final_point)
        final_expected = final_oracle['expected']
        final_loss = _final(prepared, leaves, targets, plan['normalization'], final_expected, device, block, torch)
        np.testing.assert_allclose(final_loss, final_oracle['value'], rtol=RTOL, atol=0.)
        np.testing.assert_allclose(final_loss, trajectory['final_value'], rtol=RTOL, atol=0.)
        assert final_loss < metrics[0]['value']
        _check_owned(held, snapshots)
        if cuda:
            assert len(set(live_allocations[2:])) == 1, 'live allocations grew after two warmup updates'
        record_property('fixed_rates', json.dumps(np.asarray(plan['rates']).tolist()))
        record_property('updates', json.dumps(metrics))
        record_property('final_loss', final_loss)
        record_property('allocated_bytes_with_one_owned_pack', json.dumps(live_allocations))
    finally:
        optimizer.zero_grad(set_to_none=True)
        held, snapshots = (), []
        gc.collect()
        if cuda:
            torch.cuda.synchronize()
    assert all(leaf.grad is None for leaf in leaves)
    if cuda:
        assert torch.cuda.memory_allocated() == baseline, 'live storage remains after all owned outputs are released'
