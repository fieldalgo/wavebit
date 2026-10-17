"""Nonfinite source leaves must not produce NaN fields reported as stable.

The same leaf objects are updated in place, as by an optimizer. Rebuilding
the Problem refreshes the derived SI current moment without changing the fixed
prepared structure. These two first regressions use the established small
electric Dipole/CPML fixture; they do not qualify new source or device scope.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wavebit import fdtd
from wavebit.errors import ValidationError
from tests.fdtd.test_general_cuda import closed_problem, fixed_numerics
from tests.fdtd.test_general_cuda_cpml_higher_order import numerics, problem


@pytest.mark.parametrize('invalid_leaf', [0, 1], ids=['amplitude', 'phase'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_same_source_leaf_nan_refused_or_reported_invalid_then_recovers(device, invalid_leaf):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    prepared = fdtd.prepare(problem(), numerics=numerics(), backend='torch', device=device)
    values = (1.2, .37)
    leaves = (torch.tensor(values[0], dtype=torch.float64, requires_grad=True),
              torch.tensor(values[1], dtype=torch.float64, device=device, requires_grad=True))
    identities = tuple(id(leaf) for leaf in leaves)
    first = prepared.run(problem(*leaves))
    assert bool(first.validity.checks['runtime_stability'])
    assert bool(torch.isfinite(first.final_fields.values['E']).all())
    assert bool((first.sources['d']['field'] != 0).any())
    held = (first.final_fields.values['E'].detach().clone(),
            first.sources['d']['field'].detach().clone())

    with torch.no_grad():
        leaves[invalid_leaf].fill_(float('nan'))
    refused = False
    invalid_has_nan = False
    invalid_stable = None
    try:
        invalid = prepared.run(problem(*leaves))
    except ValidationError as exc:
        assert exc.issue.code and ('finite' in str(exc).lower() or 'nan' in str(exc).lower())
        refused = True
    else:
        invalid_has_nan = bool(torch.isnan(invalid.final_fields.values['E']).any())
        invalid_stable = bool(invalid.validity.checks['runtime_stability'])
    print(dict(leaf=('amplitude', 'phase')[invalid_leaf], structured_refusal=refused,
               output_has_nan=invalid_has_nan, runtime_stability=invalid_stable), flush=True)

    with torch.no_grad():
        leaves[invalid_leaf].fill_(values[invalid_leaf])
    recovered = prepared.run(problem(*leaves))
    assert tuple(id(leaf) for leaf in leaves) == identities
    assert all(leaf.is_leaf and leaf.grad is None for leaf in leaves)
    assert bool(recovered.validity.checks['runtime_stability'])
    torch.testing.assert_close(recovered.final_fields.values['E'], held[0], rtol=0, atol=0)
    torch.testing.assert_close(recovered.sources['d']['field'], held[1], rtol=0, atol=0)
    torch.testing.assert_close(first.final_fields.values['E'], held[0], rtol=0, atol=0)
    torch.testing.assert_close(first.sources['d']['field'], held[1], rtol=0, atol=0)
    assert refused or (invalid_has_nan and invalid_stable is False), (
        'nonfinite source reached the solver without a failed runtime validity verdict',
        invalid_has_nan, invalid_stable)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_same_initial_amplitude_nan_refused_or_reported_invalid_then_recovers(device):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    leaf = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    identity = id(leaf)
    prepared = fdtd.prepare(closed_problem(), numerics=fixed_numerics(17),
                            backend='torch', device=device)
    first = prepared.run(closed_problem(amplitude=leaf))
    assert bool(first.validity.checks['runtime_stability'])
    held = {field: value.detach().clone() for field, value in first.final_fields.values.items()}
    assert all(bool(torch.isfinite(value).all()) for value in held.values())
    assert bool((held['E'] != 0).any()) and bool((held['H'] != 0).any())
    with torch.no_grad():
        leaf.fill_(float('nan'))
    refused = False
    invalid_has_nan = False
    invalid_stable = None
    try:
        invalid = prepared.run(closed_problem(amplitude=leaf))
    except ValidationError as exc:
        assert exc.issue.code and ('finite' in str(exc).lower() or 'nan' in str(exc).lower())
        refused = True
    else:
        invalid_has_nan = bool(torch.isnan(invalid.final_fields.values['E']).any())
        invalid_stable = bool(invalid.validity.checks['runtime_stability'])
    print(dict(leaf='initial.amplitude', structured_refusal=refused,
               output_has_nan=invalid_has_nan, runtime_stability=invalid_stable), flush=True)
    with torch.no_grad():
        leaf.fill_(1.2)
    recovered = prepared.run(closed_problem(amplitude=leaf))
    assert id(leaf) == identity and leaf.is_leaf and leaf.grad is None
    assert bool(recovered.validity.checks['runtime_stability'])
    for field in held:
        torch.testing.assert_close(recovered.final_fields.values[field], held[field], rtol=0, atol=0)
        torch.testing.assert_close(first.final_fields.values[field], held[field], rtol=0, atol=0)
    assert refused or (invalid_has_nan and invalid_stable is False), (
        'nonfinite initial amplitude reached the solver without a failed runtime validity verdict',
        invalid_has_nan, invalid_stable)


def excitation_scene(name, value):
    """Lowered excitation leaves, without constructor refusals hiding the runtime door."""
    source = SimpleNamespace(amplitude=-2., phase=0., delay=-.3,
                             band=SimpleNamespace(wavelength=2., df=.08,
                                                  sigmas=np.sqrt(-2*np.log(1e-3))))
    initial = SimpleNamespace(amplitude=.7)
    if name == 'initial':
        initial.amplitude = value
    elif name in ('wavelength', 'df'):
        setattr(source.band, name, value)
    else:
        setattr(source, name, value)
    return SimpleNamespace(sources=(source,), initial=initial)


@pytest.mark.parametrize('name', ['amplitude', 'phase', 'delay', 'wavelength', 'df', 'initial'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')],
                         ids=['nan', 'positive-inf', 'negative-inf'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_live_excitation_nonfinite_door(name, bad, device):
    torch = pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    leaf = torch.tensor(bad, dtype=torch.float64, requires_grad=True)
    guard = _excitation_door_guard(excitation_scene(name, leaf), runtime('torch', device), None)
    assert guard.device.type == device.split(':')[0] and guard.dtype == torch.float64
    assert guard.ndim == 0 and bool(torch.isnan(guard))
    assert leaf.is_leaf and leaf.grad is None and leaf.device.type == 'cpu'


@pytest.mark.parametrize('name', ['wavelength', 'df'])
@pytest.mark.parametrize('bad', [0., -.1])
def test_live_band_requires_positive_values(name, bad):
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    leaf = torch.tensor(bad, dtype=torch.float64, requires_grad=True)
    assert bool(torch.isnan(_excitation_door_guard(
        excitation_scene(name, leaf), runtime('torch'), None)))


@pytest.mark.parametrize('name', ['amplitude', 'initial'])
@pytest.mark.parametrize('bad', [complex(float('nan'), 1), complex(1, float('inf')),
                                complex(-float('inf'), 0)])
@pytest.mark.parametrize('live', [False, True], ids=['concrete', 'live'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'], ids=['cpu', 'gpu'])
def test_complex_amplitudes_check_both_parts(name, bad, live, device):
    torch = pytest.importorskip('torch')
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    value = (torch.tensor(bad, dtype=torch.complex128, device=device, requires_grad=True)
             if live else bad)
    guard = _excitation_door_guard(excitation_scene(name, value), runtime('torch', device), None)
    assert guard.device.type == device.split(':')[0] and guard.dtype == torch.float64
    assert bool(torch.isnan(guard))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')],
                         ids=['nan', 'positive-inf', 'negative-inf'])
def test_concrete_initial_nonfinite_has_failed_public_validity(bad):
    result = fdtd.run(closed_problem(amplitude=bad), numerics=fixed_numerics(3))
    assert not bool(result.validity.checks['runtime_stability'])
    assert np.isnan(result.final_fields.values['E']).any()


def test_valid_concrete_and_empty_column_preserve_incoming_guard():
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('numpy')
    for incoming in (None, np.array(.75), np.array(np.nan)):
        assert _excitation_door_guard(SimpleNamespace(sources=(), initial=None), rt,
                                       incoming) is incoming
        for name, value in [('amplitude', -2.), ('phase', 0.), ('delay', -.3),
                            ('amplitude', 1+2j), ('initial', np.array([1., -2., 0.])),
                            ('initial', 1+2j)]:
            assert _excitation_door_guard(excitation_scene(name, value), rt,
                                           incoming) is incoming


def test_torch_excitation_forward_and_batched_transforms():
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')

    def guarded_square(x):
        guard = _excitation_door_guard(excitation_scene('initial', x), rt, None)
        return x*x * (1. if guard is None else guard)

    x = torch.tensor(-1.2, dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(guarded_square(dual))
        torch.testing.assert_close(primal, x*x, rtol=0, atol=0)
        torch.testing.assert_close(tangent, 2*x, rtol=0, atol=0)
        invalid = torch.autograd.forward_ad.make_dual(
            torch.tensor(float('nan'), dtype=x.dtype), torch.ones_like(x))
        assert bool(torch.isnan(_excitation_door_guard(
            excitation_scene('initial', invalid), rt, None)))
    assert float(torch.func.jacfwd(guarded_square)(x)) == float(2*x)
    values = torch.tensor([-1.2, 0., 2., float('nan')], dtype=x.dtype)
    actual = torch.func.vmap(guarded_square)(values)
    torch.testing.assert_close(actual[:3], values[:3]**2, rtol=0, atol=0)
    assert bool(torch.isnan(actual[-1]))


def test_concrete_torch_logical_views_and_bfloat16():
    torch = pytest.importorskip('torch')
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('torch')
    base = torch.tensor(1+2j, dtype=torch.complex128)
    valid = (base.conj(), torch._neg_view(base), torch.tensor(-2., dtype=torch.bfloat16))
    for value in valid:
        assert _excitation_door_guard(excitation_scene('initial', value), rt, None) is None
    bad = torch.tensor(complex(1, float('inf')), dtype=torch.complex128).conj()
    assert bool(torch.isnan(_excitation_door_guard(excitation_scene('amplitude', bad), rt, None)))
    assert base.item() == 1+2j and base.grad is None


def test_jax_excitation_jit_and_derivatives():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from wavebit.fdtd._prepare.params import _excitation_door_guard
    from wavebit.fdtd._runtime import runtime
    rt = runtime('jax')

    def guarded_square(x):
        guard = _excitation_door_guard(excitation_scene('initial', x), rt, None)
        return x*x * (1. if guard is None else guard)

    compiled = jax.jit(guarded_square)
    assert float(compiled(-1.2)) == pytest.approx(1.44, rel=1e-15)
    assert float(jax.grad(compiled)(-1.2)) == pytest.approx(-2.4, rel=1e-15)
    assert float(jax.grad(jax.grad(compiled))(-1.2)) == 2.
    assert np.isnan(compiled(float('nan')))
