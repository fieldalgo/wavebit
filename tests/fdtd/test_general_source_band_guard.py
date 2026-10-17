"""Carried Gaussian bands retain the concrete positive-carrier contract.

These tests concern f0 > cutoff_sigmas * df, including finite derived terms.
They do not qualify dynamic spatial resolution or general waveform overflow.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fdtd import _pulse_yee_reference as pulse
from wavebit import fdtd
from wavebit.errors import ValidationError


CASES = [('cpu', None), ('cuda:0', None), ('cuda:0', 7)]
CASE_IDS = ['cpu_plain', 'gpu_plain', 'gpu_k7']


def torch_runtime(device):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('actual CUDA device required')
    return torch


def source_problem(wavelength, bandwidth, block):
    return pulse.fixture((wavelength, bandwidth, .4, .12), steps=17, block=block)


def output_tensors(result):
    return (*result.final_fields.values.values(),
            *(result[name].values for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')),
            *(result.sources['d'][key] for key in ('drive', 'field', 'delivered_power')))


@pytest.mark.parametrize('device,block', CASES, ids=CASE_IDS)
def test_same_band_leaves_invalid_carrier_then_recover(device, block):
    torch = torch_runtime(device)
    # The invalid concrete twin is a carrier refusal, not a grid-resolution failure.
    with pytest.raises(ValidationError, match='spectrum reaches zero frequency'):
        twin, numerics = source_problem(1., 2., block)
        fdtd.prepare(twin, numerics=numerics, backend='torch', device=device)

    initial, numerics = source_problem(1., 1., block)
    prepared = fdtd.prepare(initial, numerics=numerics, backend='torch', device=device)
    leaves = (torch.tensor(1., dtype=torch.float64, requires_grad=True),
              torch.tensor(1., dtype=torch.float64, device=device, requires_grad=True))
    for leaf in leaves:
        leaf.grad = torch.full_like(leaf, .125)
    identities = tuple(id(leaf) for leaf in leaves)
    grad_refs = tuple(leaf.grad for leaf in leaves)
    grad_snapshots = tuple(leaf.grad.clone() for leaf in leaves)

    def run():
        versions = tuple(leaf._version for leaf in leaves)
        values = tuple(leaf.detach().clone() for leaf in leaves)
        grad_versions = tuple(leaf.grad._version for leaf in leaves)
        try:
            problem, _ = source_problem(*leaves, block)
            assert problem.sources[0].center == initial.sources[0].center
            return prepared.run(problem)
        finally:
            assert tuple(id(leaf) for leaf in leaves) == identities
            assert tuple(leaf._version for leaf in leaves) == versions
            for i, leaf in enumerate(leaves):
                assert leaf.is_leaf and leaf.requires_grad
                assert str(leaf.device) == ('cpu' if i == 0 else device)
                assert leaf.grad is grad_refs[i] and leaf.grad._version == grad_versions[i]
                torch.testing.assert_close(leaf, values[i], rtol=0, atol=0)
                torch.testing.assert_close(leaf.grad, grad_snapshots[i], rtol=0, atol=0)

    first = run()
    assert bool(first.validity.checks['runtime_stability'])
    assert bool((first.final_fields.values['E'] != 0).any())
    held = tuple(t.detach().clone() for t in output_tensors(first))
    with torch.no_grad():
        leaves[1].fill_(2.)
    refused, invalid_has_nan, invalid_stable = False, False, None
    try:
        invalid = run()
    except ValidationError as exc:
        assert 'spectrum reaches zero frequency' in str(exc)
        refused = True
    else:
        invalid_has_nan = bool(torch.isnan(invalid.final_fields.values['E']).any())
        invalid_stable = bool(invalid.validity.checks['runtime_stability'])
    print(dict(structured_refusal=refused, output_has_nan=invalid_has_nan,
               runtime_stability=invalid_stable), flush=True)

    with torch.no_grad():
        leaves[1].fill_(1.)
    recovered = run()
    assert bool(recovered.validity.checks['runtime_stability'])
    for actual, original, expected in zip(output_tensors(recovered), output_tensors(first), held):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(original, expected, rtol=0, atol=0)
    assert refused or (invalid_has_nan and invalid_stable is False), (
        'a carried band forbidden by its concrete twin returned without failed validity',
        invalid_has_nan, invalid_stable)


def band_scene(wavelength=2., df=.125, *, phase=0., sigmas=2.):
    # k=2 has an exactly representable boundary df=.25 at wavelength=2.
    band = SimpleNamespace(wavelength=wavelength, df=df, sigmas=sigmas)
    source = SimpleNamespace(amplitude=-2., phase=phase, delay=-.3, band=band)
    return SimpleNamespace(sources=(source,), initial=None)


@pytest.mark.parametrize('carried', ['wavelength', 'df', 'both'])
@pytest.mark.parametrize('df,valid', [(.125, True), (np.nextafter(.25, 0.), True),
                                     (.25, False), (np.nextafter(.25, np.inf), False),
                                     (.5, False)],
                         ids=['below', 'adjacent-below', 'equal', 'adjacent-above', 'above'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_carrier_strict_boundary(carried, df, valid, device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    wavelength = (torch.tensor(2., dtype=torch.float64, requires_grad=True)
                  if carried in ('wavelength', 'both') else 2.)
    bandwidth = (torch.tensor(df, dtype=torch.float64, device=device, requires_grad=True)
                 if carried in ('df', 'both') else df)
    actual = _excitation_door_guard(band_scene(wavelength, bandwidth),
                                    runtime('torch', device), None)
    assert actual.dtype == torch.float64 and str(actual.device) == device
    assert actual.ndim == 0 and not actual.requires_grad
    assert bool(torch.isfinite(actual)) is valid
    if valid:
        assert actual.item() == 1.


@pytest.mark.parametrize('wavelength,df', [(np.nextafter(0., 1.), .125), (2., 1e308)],
                         ids=['carrier-overflow', 'cutoff-width-overflow'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_finite_band_leaves_reject_derived_spectral_overflow(wavelength, df, device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    leaves = (torch.tensor(wavelength, dtype=torch.float64, requires_grad=True),
              torch.tensor(df, dtype=torch.float64, device=device, requires_grad=True))
    assert all(bool(torch.isfinite(leaf)) and bool(leaf > 0) for leaf in leaves)
    actual = _excitation_door_guard(band_scene(*leaves), runtime('torch', device), None)
    assert bool(torch.isnan(actual)) and not actual.requires_grad


def test_concrete_band_preserves_guard_identity_and_phase_only_does_not_recheck_band():
    torch = torch_runtime('cpu')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    incoming = torch.tensor(.75, dtype=torch.float64)
    assert _excitation_door_guard(band_scene(), rt, incoming) is incoming
    assert _excitation_door_guard(band_scene(), rt, None) is None
    phase = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    # A fake concrete invalid band distinguishes a new band check from the
    # existing phase-finiteness certificate. Real constructors already refuse it.
    actual = _excitation_door_guard(band_scene(df=.5, phase=phase), rt, incoming)
    torch.testing.assert_close(actual, incoming, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_band_guard_keeps_prior_failed_validity(device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    incoming = torch.tensor(float('nan'), dtype=torch.float64, device=device)
    leaf = torch.tensor(.125, dtype=torch.float64, device=device, requires_grad=True)
    actual = _excitation_door_guard(band_scene(df=leaf), runtime('torch', device), incoming)
    assert bool(torch.isnan(actual)) and not actual.requires_grad


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_torch_band_forward_reverse_and_batched_certificate(device):
    torch = torch_runtime(device)
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch', device)

    def certificate(point):
        return _excitation_door_guard(band_scene(point[0], point[1]), rt, None)

    def guarded_quadratic(point):
        guard = certificate(point)
        return point.square().sum() * (1. if guard is None else guard)

    point = torch.tensor([2., .125], dtype=torch.float64, device=device)
    direction = torch.tensor([.3, -.2], dtype=point.dtype, device=device)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(point, direction)
        guard, guard_tangent = torch.autograd.forward_ad.unpack_dual(certificate(dual))
        assert guard.item() == 1. and guard_tangent is None
        value, tangent = torch.autograd.forward_ad.unpack_dual(guarded_quadratic(dual))
        torch.testing.assert_close(value, point.square().sum(), rtol=0, atol=0)
        torch.testing.assert_close(tangent, 2*(point*direction).sum(), rtol=0, atol=0)
        bad = torch.autograd.forward_ad.make_dual(point*point.new_tensor([1., 2.]), direction)
        assert bool(torch.isnan(certificate(bad)))
    torch.testing.assert_close(torch.func.jacfwd(guarded_quadratic)(point),
                               2*point, rtol=0, atol=0)
    torch.testing.assert_close(torch.func.hessian(guarded_quadratic)(point),
                               2*torch.eye(2, dtype=point.dtype, device=device), rtol=0, atol=0)
    live = point.detach().clone().requires_grad_()
    first = torch.autograd.grad(guarded_quadratic(live), live, create_graph=True)[0]
    torch.testing.assert_close(first, 2*point, rtol=0, atol=0)
    second = torch.stack([torch.autograd.grad(first[i], live, retain_graph=True)[0]
                          for i in range(2)])
    torch.testing.assert_close(second, 2*torch.eye(2, dtype=point.dtype, device=device),
                               rtol=0, atol=0)
    batch = torch.tensor([[2., .125], [2., .25], [2., .5], [1., .125]],
                         dtype=point.dtype, device=device)
    guards = torch.func.vmap(certificate)(batch)
    torch.testing.assert_close(guards[[0, 3]], torch.ones(2, dtype=point.dtype, device=device),
                               rtol=0, atol=0)
    assert bool(torch.isnan(guards[1:3]).all())


def test_jax_band_jit_gradient_hessian_and_batch():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('jax')

    def certificate(point):
        return _excitation_door_guard(band_scene(point[0], point[1]), rt, None)

    def guarded_quadratic(point):
        guard = certificate(point)
        return jnp.sum(point**2) * (1. if guard is None else guard)

    point = jnp.array([2., .125], dtype=jnp.float64)
    compiled = jax.jit(guarded_quadratic)
    np.testing.assert_array_equal(compiled(point), np.sum(np.asarray(point)**2))
    np.testing.assert_array_equal(jax.grad(compiled)(point), 2*np.asarray(point))
    np.testing.assert_array_equal(jax.hessian(compiled)(point), 2*np.eye(2))
    np.testing.assert_array_equal(jax.jacfwd(jax.jit(certificate))(point), np.zeros(2))
    batch = jnp.array([[2., .125], [2., .25], [2., .5], [1., .125]])
    actual = np.asarray(jax.jit(jax.vmap(certificate))(batch))
    np.testing.assert_array_equal(actual[[0, 3]], np.ones(2))
    assert np.isnan(actual[1:3]).all()
