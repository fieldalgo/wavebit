"""Finite source leaves must not certify nonfinite waveform arithmetic."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fdtd import _pulse_yee_reference as pulse
from tests.fdtd.test_general_source_band_guard import output_tensors, torch_runtime
from wavebit import fdtd
from wavebit.errors import ValidationError

CASES = [('cpu', None), ('cuda:0', None), ('cuda:0', 7)]
CASE_IDS = ['cpu_plain', 'gpu_plain', 'gpu_k7']


def host(value):
    return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('device,block', CASES, ids=CASE_IDS)
@pytest.mark.parametrize('huge', [1e308, -1e308], ids=['positive_delay', 'negative_delay'])
def test_same_delay_leaf_huge_waveform_then_recovers(device, block, huge, tmp_path,
                                                    record_property):
    torch = torch_runtime(device)
    base, policy = pulse.fixture(pulse.POINTS[0], steps=17, block=block)
    prepared = fdtd.prepare(base, numerics=policy, backend='torch', device=device)
    leaves = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True,
                               device='cpu' if i in (0, 2) else device)
                   for i, v in enumerate(pulse.POINTS[0]))
    for leaf in leaves:
        leaf.grad = torch.full_like(leaf, .125)
    identities = tuple((id(v), v.data_ptr(), v.device, id(v.grad), v.grad.data_ptr())
                       for v in leaves)

    def run():
        versions = tuple((v._version, v.grad._version) for v in leaves)
        saved = tuple(v.detach().clone() for v in leaves)
        try:
            problem, _ = pulse.fixture(leaves, steps=17, block=block)
            assert problem.sources[0].center == base.sources[0].center
            return prepared.run(problem)
        finally:
            assert tuple((id(v), v.data_ptr(), v.device, id(v.grad), v.grad.data_ptr())
                         for v in leaves) == identities
            assert tuple((v._version, v.grad._version) for v in leaves) == versions
            for leaf, wanted in zip(leaves, saved):
                assert leaf.is_leaf and leaf.requires_grad
                torch.testing.assert_close(leaf, wanted, rtol=0, atol=0)
                torch.testing.assert_close(leaf.grad, torch.full_like(leaf, .125), rtol=0, atol=0)

    first = run()
    assert bool(first.validity.checks['runtime_stability'])
    assert bool((first.final_fields.values['E'] != 0).any())
    saved = tuple(v.detach().clone() for v in output_tensors(first))
    concrete, _ = pulse.fixture((1., 1., .4, huge), steps=17, block=block)
    with pytest.raises(ValidationError) as caught:
        prepared.run(concrete)
    assert caught.value.code == 'source_waveform_range'
    assert caught.value.issue.path == 'sources.d'
    with torch.no_grad():
        leaves[3].fill_(huge)
    facts = {'structured_refusal': False}
    try:
        invalid = run()
    except ValidationError as exc:
        assert any(word in str(exc).lower() for word in ('finite', 'waveform', 'overflow'))
        facts.update(structured_refusal=True, code=exc.code, message=str(exc))
    else:
        arrays = {'drive': host(invalid.sources['d']['drive']),
                  'E': host(invalid.final_fields.values['E']),
                  'H': host(invalid.final_fields.values['H'])}
        np.savez(tmp_path/'invalid-waveform.npz', **arrays)
        facts.update(runtime_stability=bool(invalid.validity.checks['runtime_stability']),
                     drive_finite=bool(np.isfinite(arrays['drive']).all()),
                     fields_have_nan=bool(np.isnan(arrays['E']).any()),
                     raw_artifact=str(tmp_path/'invalid-waveform.npz'))
    record_property('invalid_waveform_facts', json.dumps(facts))
    print(facts, flush=True)
    with torch.no_grad():
        leaves[3].fill_(pulse.POINTS[0][3])
    recovered = run()
    assert bool(recovered.validity.checks['runtime_stability'])
    for current, old, wanted in zip(output_tensors(recovered), output_tensors(first), saved):
        torch.testing.assert_close(current, wanted, rtol=0, atol=0)
        torch.testing.assert_close(old, wanted, rtol=0, atol=0)
    assert facts['structured_refusal'] or (
        facts['fields_have_nan'] and facts['runtime_stability'] is False), facts


def waveform_case(name):
    values = {'ordinary': (2., .08, .4, -.3),
              'large_phase': (2., .08, 1e100, -.3),
              'positive_delay': (2., .08, .4, 1e308),
              'negative_delay': (2., .08, .4, -1e308),
              'envelope_square': (2., .08, .4, 1e155),
              'phase_addition': (2., 1e-155, 1e308, -3e307),
              'carrier_frequency': (np.nextafter(0., 1.), .08, .4, -.3),
              'carrier_prefactor': (1e-308, .08, .4, -.3),
              'tau': (2., np.nextafter(0., 1.), .4, -.3),
              't0': (2., 1e-309, .4, -.3)}
    return values[name]


def independent_stages(values, shift=.5, dt=.01):
    wavelength, width, phase, delay = map(np.float64, values)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
        f0 = 1/wavelength
        tau = 1/(2*np.pi*width)
        t0 = pulse.KAPPA*tau
        carrier = 2*np.pi*f0
        s = (np.array([0., 16.])+shift)*dt-t0-delay
        quotient = s/tau
        square = quotient**2
        theta = carrier*s+phase
    return {'f0': f0, 'tau': tau, 't0': t0, 'carrier': carrier,
            's': s, 'quotient': quotient, 'square': square, 'theta': theta}


def certificate_inputs(values, rt, field='E'):
    from wavebit.fdtd._kernel.sources import DipoleTerm
    wavelength, width, phase, delay = values
    source = SimpleNamespace(name='d', amplitude=1., phase=phase, delay=delay,
                             band=SimpleNamespace(wavelength=wavelength, df=width,
                                                  sigmas=pulse.KAPPA))
    scene = SimpleNamespace(sources=(source,))
    term = DipoleTerm('d', field, None, 0, .05, 'bspline')
    plan = SimpleNamespace(terms=(term,), steps=17, dt=.01, decay=None)
    w, df, p, d = (rt.array(v, 'float64') for v in values)
    tau = 1/(2*np.pi*df)
    params = {'sources': {'d': {'f0': 1/w, 'tau': tau, 't0': pulse.KAPPA*tau,
                               'phase': p, 'delay': d}}}
    return scene, plan, params


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
@pytest.mark.parametrize('name', ['ordinary', 'large_phase', 'positive_delay', 'negative_delay',
                                  'envelope_square', 'phase_addition', 'carrier_frequency',
                                  'carrier_prefactor', 'tau', 't0'])
def test_concrete_and_carried_waveform_arithmetic_certificate(device, name):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)
    values = waveform_case(name)
    stages = independent_stages(values)
    finite = all(np.isfinite(v).all() for v in stages.values())
    assert finite is (name in ('ordinary', 'large_phase'))
    if name == 'phase_addition':
        assert all(np.isfinite(v).all() for key, v in stages.items() if key != 'theta')
        assert not np.isfinite(stages['theta']).all()
    if name == 'envelope_square':
        assert np.isfinite(stages['quotient']).all() and np.isfinite(stages['theta']).all()
        assert not np.isfinite(stages['square']).all()
    concrete = certificate_inputs(values, rt)
    incoming = rt.array(.75, 'float64')
    if finite:
        assert _source_waveform_guard(*concrete, rt, incoming) is incoming
        assert _source_waveform_guard(*concrete, rt, None) is None
    else:
        with pytest.raises(ValidationError) as caught:
            _source_waveform_guard(*concrete, rt, None)
        assert caught.value.code == 'source_waveform_range'
        assert caught.value.issue.path == 'sources.d'
    leaves = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True,
                               device='cpu' if i in (0, 2) else device)
                   for i, v in enumerate(values))
    for field in ('E', 'H'):
        actual = _source_waveform_guard(*certificate_inputs(leaves, rt, field), rt, None)
        assert actual.dtype == torch.float64 and str(actual.device) == device
        assert not actual.requires_grad and actual.ndim == 0
        assert bool(torch.isfinite(actual)) is finite
        if finite:
            torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
        failed = _source_waveform_guard(*certificate_inputs(leaves, rt, field), rt,
                                        rt.array(float('nan'), 'float64'))
        assert bool(torch.isnan(failed))


def test_sourcefree_and_decay_preserve_incoming_certificate_identity():
    torch_runtime('cpu')
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    scene, plan, params = certificate_inputs(waveform_case('positive_delay'), rt)
    incoming = rt.array(.75, 'float64')
    # These paths must return before examining malformed/unbounded source data.
    sourcefree = SimpleNamespace(terms=(), steps=17, dt=.01, decay=None)
    assert _source_waveform_guard(SimpleNamespace(sources=()), sourcefree, {}, rt, incoming) is incoming
    plan.decay = object()
    assert _source_waveform_guard(scene, plan, params, rt, incoming) is incoming


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_waveform_certificate_forward_reverse_and_batch(device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)

    def certificate(point):
        return _source_waveform_guard(*certificate_inputs(tuple(point), rt), rt, None)

    def function(point):
        guard = certificate(point)
        return point.square().sum()*(1. if guard is None else guard)

    point = torch.tensor(waveform_case('ordinary'), dtype=torch.float64, device=device)
    direction = torch.full_like(point, .2)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(point, direction)
        guard, tangent = torch.autograd.forward_ad.unpack_dual(certificate(dual))
        assert guard.item() == 1. and tangent is None
        value, derivative = torch.autograd.forward_ad.unpack_dual(function(dual))
        torch.testing.assert_close(value, point.square().sum(), rtol=0, atol=0)
        torch.testing.assert_close(derivative, 2*(point*direction).sum(), rtol=0, atol=0)
    torch.testing.assert_close(torch.func.jacfwd(function)(point), 2*point, rtol=0, atol=0)
    torch.testing.assert_close(torch.func.hessian(function)(point),
                               2*torch.eye(4, dtype=point.dtype, device=device), rtol=0, atol=0)
    batch = torch.tensor([waveform_case(k) for k in ('ordinary', 'positive_delay', 'large_phase')],
                         dtype=torch.float64, device=device)
    guards = torch.func.vmap(certificate)(batch)
    assert bool(torch.isnan(guards[1]))
    torch.testing.assert_close(guards[[0, 2]], torch.ones(2, dtype=point.dtype, device=device),
                               rtol=0, atol=0)


def test_jax_waveform_certificate_jit_grad_hessian_and_batch():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('jax')

    def certificate(point):
        return _source_waveform_guard(*certificate_inputs(tuple(point), rt), rt, None)

    def function(point):
        guard = certificate(point)
        return jnp.sum(point**2)*(1. if guard is None else guard)

    point = jnp.asarray(waveform_case('ordinary'))
    compiled = jax.jit(function)
    np.testing.assert_allclose(compiled(point), np.sum(np.asarray(point)**2), rtol=2e-15, atol=0.)
    np.testing.assert_array_equal(jax.grad(compiled)(point), 2*np.asarray(point))
    np.testing.assert_array_equal(jax.hessian(compiled)(point), 2*np.eye(4))
    np.testing.assert_array_equal(jax.jacfwd(jax.jit(certificate))(point), np.zeros(4))
    batch = jnp.asarray([waveform_case(k) for k in ('ordinary', 'positive_delay', 'large_phase')])
    actual = np.asarray(jax.jit(jax.vmap(certificate))(batch))
    assert np.isnan(actual[1])
    np.testing.assert_array_equal(actual[[0, 2]], np.ones(2))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_waveform_certificate_checks_final_endpoint_and_never_reads_carried_host(device, monkeypatch):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)
    values = (10/3, 1e-308, 0., 0.)
    stages = independent_stages(values, dt=1e307)
    assert np.isfinite(stages['theta'][0]) and not np.isfinite(stages['theta'][1])
    assert all(np.isfinite(v).all() for key, v in stages.items() if key != 'theta')
    leaves = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True,
                               device='cpu' if i in (0, 2) else device)
                   for i, v in enumerate(values))
    scene, plan, params = certificate_inputs(leaves, rt)
    plan.dt = 1e307

    def host_read(*args, **kwargs):
        pytest.fail('carried waveform certificate attempted a host read')

    with monkeypatch.context() as patch:
        for name in ('cpu', 'numpy', 'item'):
            patch.setattr(torch.Tensor, name, host_read)
        actual = _source_waveform_guard(scene, plan, params, rt, None)
    assert bool(torch.isnan(actual)) and not actual.requires_grad


def test_valid_concrete_waveform_logical_views_preserve_guard_identity():
    torch = torch_runtime('cpu')
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    incoming = rt.array(.75, 'float64')
    negative_view = torch._neg_view(torch.tensor(.3, dtype=torch.float64))
    assert negative_view.is_neg()
    values = (2., torch.tensor(.125, dtype=torch.bfloat16), .4, negative_view)
    assert _source_waveform_guard(*certificate_inputs(values, rt), rt, incoming) is incoming


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_adjacent_t0_boundary_certifies_actual_backend_parameters(device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)
    # Wavelength keeps carrier arithmetic finite even when t0 approaches maxfloat.
    # Use nearby binary64 widths, without assuming CPU and CUDA divide identically.
    center = (pulse.KAPPA/(2*np.pi))/np.finfo(np.float64).max
    verdicts = []
    for offset in (-8, -1, 0, 1, 8):
        width = center
        for _ in range(abs(offset)):
            width = np.nextafter(width, np.inf if offset > 0 else 0.)
        values = (5e307, width, 0., 0.)
        scene, plan, params = certificate_inputs(values, rt)
        actual = params['sources']['d']
        # An independent full-time sample audit judges the exact arrays consumed
        # by this backend, rather than a host re-evaluation of reciprocal rounding.
        n = torch.arange(plan.steps, dtype=torch.float64, device=device)
        s = (n+.5)*plan.dt-actual['t0']-actual['delay']
        quotient = s/actual['tau']
        square = quotient.square()
        omega = (2*np.pi)*actual['f0']
        theta = omega*s+actual['phase']
        valid = all(bool(torch.isfinite(actual[k]) & (actual[k] > 0))
                    for k in ('f0', 'tau', 't0'))
        valid = valid and all(bool(torch.isfinite(v).all())
                              for v in (s, quotient, square, -.5*square, omega, theta))
        verdicts.append(valid)
        incoming = rt.array(.75, 'float64')
        if valid:
            assert _source_waveform_guard(scene, plan, params, rt, incoming) is incoming
        else:
            with pytest.raises(ValidationError) as caught:
                _source_waveform_guard(scene, plan, params, rt, incoming)
            assert caught.value.code == 'source_waveform_range'
        live = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True,
                                  device='cpu' if i in (0, 2) else device)
                      for i, v in enumerate(values))
        carried = _source_waveform_guard(*certificate_inputs(live, rt), rt, None)
        assert bool(torch.isfinite(carried)) is valid
    assert set(verdicts) == {False, True}, 'boundary samples must straddle acceptance'


def test_jax_jit_closed_concrete_source_uses_lowered_tracer_certificate():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('jax')

    def function(x, values):
        guard = _source_waveform_guard(*certificate_inputs(values, rt), rt, None)
        return x*x*(1. if guard is None else guard)

    good = jax.jit(lambda x: function(x, waveform_case('ordinary')))
    bad = jax.jit(lambda x: function(x, waveform_case('positive_delay')))
    x = jnp.asarray(.7)
    np.testing.assert_array_equal(good(x), x*x)
    np.testing.assert_array_equal(jax.grad(good)(x), 2*x)
    np.testing.assert_array_equal(jax.grad(jax.grad(good))(x), 2.)
    assert np.isnan(np.asarray(bad(x)))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
@pytest.mark.parametrize('name', ['ordinary', 'positive_delay'])
def test_torch_transforms_keep_closed_source_certificate_off_host(device, name, monkeypatch):
    """Geometry-only transforms still wrap a closed source's boolean certificate."""
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _source_waveform_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)
    values = waveform_case(name)
    valid = name == 'ordinary'

    def function(x):
        guard = _source_waveform_guard(*certificate_inputs(values, rt), rt, None)
        return x*x*(1. if guard is None else guard)

    point = torch.tensor(.7, dtype=torch.float64, device=device)
    direction = torch.tensor(.2, dtype=point.dtype, device=device)
    before = point.clone()
    # Eager constants retain the identity/refusal contract. A genuine forward dual
    # on the unrelated input likewise must keep its derivative; it need not make
    # the closed source itself carried.
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(point, direction)
        if valid:
            value, tangent = torch.autograd.forward_ad.unpack_dual(function(dual))
            torch.testing.assert_close(value, point.square(), rtol=0, atol=0)
            torch.testing.assert_close(tangent, 2*point*direction, rtol=0, atol=0)
        else:
            with pytest.raises(ValidationError) as caught:
                function(dual)
            assert caught.value.code == 'source_waveform_range'

    def forbidden(*args, **kwargs):
        pytest.fail('a closed-source transform certificate attempted a host read')

    with monkeypatch.context() as patch:
        patch.setattr(rt, 'numpy', forbidden)
        value, tangent = torch.func.jvp(function, (point,), (direction,))
        gradient = torch.func.grad(function)(point)
        forward = torch.func.jacfwd(function)(point)
        hessian = torch.func.hessian(function)(point)
        _, mixed = torch.func.jvp(torch.func.grad(function), (point,), (direction,))
        batch = torch.stack((point, point+.1))
        batched = torch.func.vmap(function)(batch)
        batched_grad = torch.func.vmap(torch.func.grad(function))(batch)
    actual = (value, tangent, gradient, forward, hessian, mixed, batched, batched_grad)
    expected = (point.square(), 2*point*direction, 2*point, 2*point,
                torch.full_like(point, 2.), 2*direction, batch.square(), 2*batch)
    for got, wanted in zip(actual, expected):
        if valid:
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        else:
            assert bool(torch.isnan(got).all())
    torch.testing.assert_close(point, before, rtol=0, atol=0)
    assert point.grad is None
