"""Backend adapter (wavebit._core.backend): eig dispatch + differentiability.

Verifies the single per-framework primitive: eigenvalues match NumPy, and gradients through
eig flow correctly on JAX and PyTorch (matching finite differences). Also the input contract
the adapter has to keep backend-independent -- see the stride section at the end.
"""
import numpy as np
import pytest

import wavebit
from wavebit._core.backend import eig, array_namespace, forward_strided


A_np = np.array([[2.0 + 0j, 1.0, 0.2],
                 [0.3, 3.0, 0.5],
                 [0.1, 0.2, 1.0]])


def _sorted_eigvals(w):
    w = np.asarray(w)
    return np.sort_complex(np.round(w, 8))


def test_eig_eigenvalues_numpy_reference():
    w, V = eig(A_np)
    ref = np.linalg.eigvals(A_np)
    assert np.allclose(_sorted_eigvals(w), _sorted_eigvals(ref))
    # reconstruction A V = V diag(w)
    assert np.allclose(A_np @ V, V @ np.diag(w))


def test_array_namespace_numpy():
    xp = array_namespace(A_np)
    assert 'numpy' in xp.__name__


def _grad_scalar_from_eig(xp, backend_scalar):
    # f(p) = sum |sqrt(A)|^2 with sqrt(A) = V diag(sqrt(w)) V^-1, A = A0 + p * P.
    # Gauge-invariant in the eigenvectors (as real RCWA outputs are), so both jax and
    # torch accept it -- torch.linalg.eig backward rejects phase-dependent losses.
    A0 = xp.asarray(A_np)
    P = xp.asarray(np.array([[0, 1.0, 0], [0.0, 0, 0.4], [1.0, 0, 0]], dtype=complex))
    A = A0 + backend_scalar * P
    w, V = eig(A)
    M = V @ (xp.diag(w ** 0.5) @ xp.linalg.inv(V))
    return xp.real(xp.sum(M * xp.conj(M)))


def _fd(p, h=1e-6):
    import numpy as _np
    return (float(_grad_scalar_from_eig(_np, p + h)) - float(_grad_scalar_from_eig(_np, p - h))) / (2 * h)


def test_eig_grad_jax_matches_fd():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    g = float(jax.grad(lambda p: _grad_scalar_from_eig(jnp, p))(jnp.asarray(2.0)))
    assert abs(g - _fd(2.0)) < 1e-4


def test_eig_grad_torch_matches_fd():
    torch = pytest.importorskip("torch")
    p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    _grad_scalar_from_eig(torch, p).backward()
    assert abs(p.grad.item() - _fd(2.0)) < 1e-4


# ---------------------------------------------------------------- Bessel J1

def _j1_reference(x):
    """SciPy's J1, which is the accurate one and is what NumPy already uses.

    Checked against ``mpmath`` at 40 digits over ``(0, 95]``: max absolute error 2.2e-16.
    That check is not run here because mpmath is not a dependency; this records the result
    so the constant below is not a mystery.
    """
    from scipy.special import j1
    return j1(np.asarray(x))


# The series/asymptotic hybrid the traced backends use is not exact. Its error is set by
# cancellation in the series just below the crossover, and 1e-11 is comfortably above the
# measured 1.2e-12 while still being four orders tighter than what torch.special.bessel_j1
# delivered (4.1e-7) before this path was shared.
J1_TOL = 1e-11

# Includes the crossover at 12.5 from both sides, the origin, and the far field where the
# asymptotic expansion is the only branch in play.
J1_PROBE = np.concatenate([
    np.array([0.0, 1e-8, 1e-3]),
    np.linspace(0.05, 12.4, 200),
    np.array([12.45, 12.499, 12.5, 12.501, 12.55]),
    np.linspace(12.6, 95.0, 400),
])


@pytest.mark.parametrize('lib', ['jax', 'torch'])
def test_j1_matches_scipy_on_every_backend(lib):
    """The traced backends must agree with SciPy, not merely with each other.

    Both used to be wrong in different ways and by different amounts: torch took
    ``torch.special.bessel_j1`` (4.1e-7 from truth) while jax took the hybrid with a
    crossover that picked the worse branch on both sides of itself (7.2e-11). They now
    share one implementation, so this test would pass vacuously if it compared them to
    each other -- SciPy is the reference for that reason.
    """
    xp = pytest.importorskip(lib)
    from wavebit._core.backend import j1
    if lib == 'jax':
        import jax
        jax.config.update('jax_enable_x64', True)
        x = xp.numpy.asarray(J1_PROBE, dtype=xp.numpy.float64)
    else:
        x = xp.as_tensor(J1_PROBE, dtype=xp.float64)
    got = np.asarray(j1(x), dtype=float)
    assert np.max(np.abs(got - _j1_reference(J1_PROBE))) < J1_TOL


@pytest.mark.parametrize('lib', ['jax', 'torch'])
def test_j1_single_precision_uses_its_own_crossover(lib):
    """float32 needs a different crossover, and the double-precision one costs 200x.

    The series loses a fixed number of *digits* to cancellation, which is a far larger share
    of single precision's budget than of double's. Optimising separately against mpmath puts
    the crossover at 12.5 in double and 7.0 in single. Running single with the double
    crossover measured 2.7e-04 -- worse than the `torch.special.bessel_j1` this path
    replaced, which is the regression this test exists to prevent.

    The tolerance is loose on purpose: 5e-06 against a measured 2.1e-06 and a float32 epsilon
    of 1.2e-07. Single precision is documented as hazardous and this does not pretend
    otherwise; it only pins that the hybrid stays in the right order of magnitude for it.

    Forces `hybrid` rather than taking the default. Under `auto` torch's float32 goes to the
    framework's own routine, so this would silently stop testing the crossover while keeping
    its name -- which is the failure a test like this exists to avoid.
    """
    xp = pytest.importorskip(lib)
    from wavebit import set_j1_implementation
    from wavebit._core.backend import j1
    previous = set_j1_implementation('hybrid')
    probe = np.linspace(1e-3, 95.0, 1500)
    if lib == 'jax':
        x = xp.numpy.asarray(probe, dtype=xp.numpy.float32)
    else:
        x = xp.as_tensor(probe, dtype=xp.float32)
    try:
        got = np.asarray(j1(x), dtype=float)
    finally:
        set_j1_implementation(previous)
    assert np.max(np.abs(got - _j1_reference(probe))) < 5e-06


@pytest.fixture
def j1_default():
    """Restore the global setting, which is global exactly because j1 has no other channel."""
    from wavebit import set_j1_implementation
    previous = set_j1_implementation('auto')
    yield
    set_j1_implementation(previous)


def test_j1_switch_picks_the_accurate_one_for_each_precision(j1_default):
    """`auto` is not a tie-break: neither implementation wins at both precisions.

    Measured against mpmath (docs/bessel-j1.md): the hybrid is 1.1e-12 in float64 where
    torch's builtin is 4.1e-07, and the builtin is 4.7e-07 in float32 where the hybrid is
    2.1e-06. So `auto` takes the hybrid at 64-bit and the builtin below it, and this test
    pins that it really is switching rather than doing one thing and calling it two.
    """
    torch = pytest.importorskip('torch')
    from wavebit import set_j1_implementation
    from wavebit._core.backend import j1
    probe = np.linspace(1e-3, 95.0, 1500)
    ref = _j1_reference(probe)

    def err(dtype):
        return np.max(np.abs(np.asarray(j1(torch.as_tensor(probe, dtype=dtype)), float) - ref))

    set_j1_implementation('hybrid')
    hybrid64, hybrid32 = err(torch.float64), err(torch.float32)
    set_j1_implementation('native')
    native64, native32 = err(torch.float64), err(torch.float32)
    set_j1_implementation('auto')
    auto64, auto32 = err(torch.float64), err(torch.float32)

    # the premise: they genuinely trade off, so `auto` has something to decide
    assert hybrid64 < native64 / 100, (hybrid64, native64)
    assert native32 < hybrid32 / 2, (native32, hybrid32)
    # and auto takes the better one at each precision
    assert auto64 == pytest.approx(hybrid64, rel=1e-6)
    assert auto32 == pytest.approx(native32, rel=1e-6)


def test_j1_switch_leaves_jax_alone(j1_default):
    """JAX has no second option, so the switch must not pretend otherwise.

    `jax.scipy.special.bessel_jn` returns NaN for small arguments. Asking for 'native'
    deliberately does not raise -- a global switch should not break one backend because
    another lacks an option -- but it must not change what JAX computes either.
    """
    pytest.importorskip('jax')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from wavebit import set_j1_implementation
    from wavebit._core.backend import j1
    x = jnp.asarray(np.linspace(1e-3, 95.0, 500), dtype=jnp.float64)
    out = {}
    for mode in ('auto', 'hybrid', 'native'):
        set_j1_implementation(mode)
        out[mode] = np.asarray(j1(x), dtype=float)
    assert np.array_equal(out['auto'], out['hybrid'])
    assert np.array_equal(out['auto'], out['native'])


def test_j1_switch_rejects_an_unknown_name(j1_default):
    from wavebit import set_j1_implementation, get_j1_implementation
    with pytest.raises(ValueError, match='auto'):
        set_j1_implementation('fastest')
    assert get_j1_implementation() == 'auto'      # unchanged by the failed call


@pytest.mark.parametrize('mode', ['auto', 'hybrid', 'native'])
def test_j1_derivative_survives_every_switch_setting(mode, j1_default):
    """The builtin has no backward of its own, so 'native' has to supply one."""
    torch = pytest.importorskip('torch')
    from scipy.special import j0, j1 as scipy_j1
    from wavebit import set_j1_implementation
    from wavebit._core.backend import j1
    set_j1_implementation(mode)
    for x0 in (0.0, 0.7, 12.4, 12.6, 40.0):
        t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        j1(t).backward()
        exact = 0.5 if x0 == 0 else float(j0(x0) - scipy_j1(x0) / x0)
        assert abs(float(t.grad) - exact) < 1e-10, (mode, x0, float(t.grad), exact)


@pytest.mark.parametrize('lib', ['jax', 'torch'])
def test_j1_derivative_is_right_including_at_zero(lib):
    """``dJ1/dx = J0 - J1/x``, and 1/2 at the origin.

    The origin is the case worth pinning. An ``abs(x)`` guard at the top of ``j1`` looks
    harmless on a domain of ``x >= 0``, and it silently zeroes this derivative, because
    ``d|x|/dx`` is 0 at 0 in both frameworks. The series branch is odd by construction and
    gives 1/2 without needing the guard.
    """
    pytest.importorskip(lib)
    from scipy.special import j0, j1 as sj1
    from wavebit._core.backend import j1
    probes = [0.0, 0.7, 5.0, 12.4, 12.6, 40.0]
    exact = [0.5 if p == 0 else float(j0(p) - sj1(p) / p) for p in probes]

    if lib == 'jax':
        import jax, jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        got = [float(jax.grad(j1)(jnp.asarray(p, dtype=jnp.float64))) for p in probes]
    else:
        import torch
        got = []
        for p in probes:
            t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
            j1(t).backward()
            got.append(float(t.grad))

    assert np.allclose(got, exact, atol=1e-11), list(zip(probes, got, exact))


# ------------------------------------------------------------------- reversed NumPy views
#
# `np.rot90`, `np.flip` and `a[::-1]` return views with a negative stride. NumPy and JAX take
# those; PyTorch refuses them ("At least one stride in the given numpy array is negative"), so
# before `forward_strided` the natural spelling of a rotated cell -- the one a user reaches for
# to check the phi=90 / transposed-grating equivalence -- was a crash on backend=2 and an answer
# on backends 0 and 1. The strides are an artifact of how the array was built, not part of the
# structure being described, so which backend is running must not be able to see them.

STRIDED_CFG = dict(period=(500., 800.), thickness=(180.,), wavelength=620.,
                   fto=(2, 2), theta=0.2, psi=0.3)

#: Asymmetric on purpose: a cell that equals its own rotation would pass the comparison below
#: while measuring nothing.
STRIDED_ROW = [1.0, 6.25, 1.0, 12.25]        # permittivity: 2.5**2 and 3.5**2
STRIDED_CELL = np.rot90(np.array([[STRIDED_ROW] * 4]), axes=(1, 2))


def _totals(res):
    return float(np.sum(np.asarray(res.R))), float(np.sum(np.asarray(res.T)))


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_a_reversed_raster_solves_the_same_as_its_copy(backend):
    """Every backend takes a negative-stride `structure`, and gets the contiguous answer.

    Two claims, and the second is why this is not just a smoke test: torch used to raise here,
    and a normalisation that copied the buffer the wrong way round would now return a *wrong*
    number instead. The copy is the reference.
    """
    if backend == 1:
        pytest.importorskip('jax')
    elif backend == 2:
        pytest.importorskip('torch')
    assert any(s < 0 for s in STRIDED_CELL.strides), 'premise: the input is a reversed view'

    strided = wavebit.rcwa(backend=backend, structure=STRIDED_CELL, **STRIDED_CFG).conv_solve()
    copied = wavebit.rcwa(backend=backend, structure=np.ascontiguousarray(STRIDED_CELL),
                          **STRIDED_CFG).conv_solve()
    assert _totals(strided) == _totals(copied)


def test_torch_takes_a_reversed_period_and_thickness():
    """Not only the raster: every scalar-ish input crosses the same adapter.

    `period` and `thickness` reach `_as_tensor` as whole arrays, so `np.flip(...)` on either
    raised exactly like the rotated cell did. Each is written backwards and flipped, so the two
    solves are handed the same numbers and differ only in how those numbers are laid out.
    """
    pytest.importorskip('torch')
    cell = np.ascontiguousarray(STRIDED_CELL)
    stack = np.concatenate([cell, 1.0 + 0.5 * cell])
    cfg = dict(wavelength=620., fto=(2, 2), theta=0.2, psi=0.3)

    reversed_ = wavebit.rcwa(backend=2, structure=stack, period=np.flip(np.array([800., 500.])),
                             thickness=np.flip(np.array([90., 180.])), **cfg).conv_solve()
    plain = wavebit.rcwa(backend=2, structure=stack, period=(500., 800.),
                         thickness=(180., 90.), **cfg).conv_solve()
    assert _totals(reversed_) == _totals(plain)


def test_normalising_strides_does_not_detach_the_structure():
    """The trap `as_array`'s docstring warns about, on the input this change touched.

    A copy is the obvious way to kill a negative stride and it is also the obvious way to drop
    a caller's optimizer leaf out of the graph. `forward_strided` copies NumPy only, so a torch
    `structure` must reach the solve untouched and every cell must still carry a gradient.
    """
    torch = pytest.importorskip('torch')
    uc = torch.tensor(np.ascontiguousarray(STRIDED_CELL), dtype=torch.float64,
                      requires_grad=True)
    torch.sum(wavebit.rcwa(backend=2, structure=uc, **STRIDED_CFG).conv_solve().T).backward()
    assert uc.requires_grad, 'the leaf was flipped in place'
    assert int((uc.grad.abs() > 1e-14).sum()) == uc.numel()


def test_as_array_carries_a_reversed_view_into_torch():
    """The generic adapter, not only the torch solver's own: both had the same hole.

    `as_array` is what every user-supplied scalar and small array in `_core` passes through, so
    it is the second door the same reversed view can arrive at, and `xp.asarray` raised there
    for the same reason `torch.tensor` did.
    """
    pytest.importorskip('torch')
    import array_api_compat.torch as txp
    from wavebit._core.backend import as_array
    reversed_ = np.flip(np.array([1.0, 2.0, 3.0]))
    got = as_array(txp, reversed_, dtype=txp.float64)
    assert np.array_equal(np.asarray(got), reversed_)


def test_forward_strided_copies_only_what_it_must():
    """It is an identity everywhere except the one case it exists for."""
    torch = pytest.importorskip('torch')
    contiguous = np.ascontiguousarray(STRIDED_CELL)
    assert forward_strided(contiguous) is contiguous
    assert forward_strided(0.4) == 0.4
    assert forward_strided([1.0, 2.0]) == [1.0, 2.0]

    leaf = torch.tensor([1.0, 2.0], requires_grad=True)
    assert forward_strided(leaf) is leaf

    fixed = forward_strided(STRIDED_CELL)
    assert all(s > 0 for s in fixed.strides)
    assert np.array_equal(fixed, STRIDED_CELL)
