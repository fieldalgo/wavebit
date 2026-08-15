"""The exact Rayleigh point solves correctly, and what still refuses does it loudly.

``tests/test_stability.py`` stresses the *evanescent* regime and does it well, but every case
in it is 1D TE at normal incidence, one layer, NumPy, ``period=3000`` / ``wavelength=400``.
That choice puts ``kx_m = 0.1333*m``, which never lands on ``n_top`` or ``n_bot``, so the file
systematically avoids the singularity this one is about.

**The regime.** At a Rayleigh (grazing) condition a diffraction order has ``kz == 0`` in some
medium: the order is exactly at the light line of that layer, neither propagating nor decaying.
The layer's eigenproblem then has a zero eigenvalue, and the matrices ``A`` / ``B`` the conical
path used to invert (D#81) are numerically singular — ``inv`` returned garbage of arbitrary
size, finite, in range, and wrong by orders of magnitude, sometimes with nothing raised.

That inversion is gone: :func:`~wavebit._core.solve_1d.layer_modes_1d_conical` now builds its
``V`` blocks from the layer eigensystem itself, so the exact-Rayleigh solve is an ordinary
solve (see its docstring for the identity and the measurements). What this file pins is the
new contract, on **every** backend:

* the solve at the exact point equals the one-ULP-detuned solve — the physics is continuous
  across the condition, the singularity was in the coordinates, not the structure;
* the lossless stack conserves energy there, with no warning;
* the one case still refused — an eigenvalue that is *exactly* zero, fto=3 below, where the
  0/0 has no consistent nearby problem to stand on — refuses loudly, never quietly.

The stack puts the condition in an *interior* layer rather than at a half-space, which is why
it needs two layers: a patterned pillar over a uniform ``n=1.5`` spacer. With ``period=1000``
and ``wavelength=500``, ``kx_m = 0.5*m``, so order ``m=3`` sits at ``kx = 1.5 = n_spacer``
exactly.

The concreteness tests at the bottom are the guard-level regression this file originally
carried: on JAX and PyTorch a wrong answer once returned with no warning at all, because both
guards keyed their eager-only rule on the *namespace* instead of on whether the value was
concrete. See :func:`wavebit._core.backend._concrete`. The guards still matter — the 2D path
at an exact layer graze (``V = (Omega_R @ W) / q`` loses the grazing column to cancellation)
and any traced-and-jitted solve still rely on them.
"""
import functools
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core.backend import (UnphysicalResultWarning, array_namespace,
                                   check_finite, check_physical)

PERIOD = 1000.0
WL = 500.0
N_SPACER = 1.5

# Patterned pillar (n=3.5 in air) over a uniform n=1.5 spacer. The spacer is the layer whose
# eigenproblem goes singular; the pillar is there to make the case an ordinary metasurface
# rather than a contrived one, and to populate the orders that reach the spacer.
_PILLAR = np.ones((1, 1, 20))
_PILLAR[0, 0, 6:14] = 3.5
_STACK = np.concatenate([_PILLAR, np.full((1, 1, 20), N_SPACER)], axis=0)

_KW = dict(period=[PERIOD], n_top=1.0, n_bot=N_SPACER, theta=0.0, phi=0.0,
           psi=np.pi / 2, thickness=[200.0, 300.0])

# fto values that retain the grazing order. fto=3 is the first truncation that retains it at
# all, and the one whose eigenvalue comes out exactly zero (the refused 0/0); at 4, 6 and 9
# the FFT round-off keeps the eigenvalue at ~1e-16 and the solve must simply be right. These
# were the values that used to come back quietly wrong (R off by 1 to 600 against 0.215).
BAND_FTO = [4, 6, 9]

# The exact answer and the one-ULP-detuned answer may differ by at most this. The instability
# this file guards against produced *random* answers — errors of 1e0 to 1e2 — so the margin
# to a false alarm is ~10 orders; measured agreement is ~1e-9 (see layer_modes_1d_conical).
TOL = 1e-6


def _solve(backend, fto, ucell=None, wavelength=WL, **kw):
    return wavebit.rcwa(backend=backend, wavelength=wavelength, fto=[fto, 0],
                        ucell=_STACK if ucell is None else ucell,
                        **{**_KW, **kw}).conv_solve()


def _totals(result):
    """``(sum R, sum T)`` as Python floats, from any backend's arrays."""
    return tuple(float(np.real(np.asarray(v, dtype=np.complex128)).sum())
                 for v in (result.R, result.T))


@functools.lru_cache(maxsize=None)
def _truth(fto):
    """The same stack one ULP of wavelength off the exact Rayleigh point.

    The physics is continuous across the condition — nothing resonates there, the singularity
    is in the *coordinates*, not in the structure — so the detuned solve is the correct
    answer, and it is an oracle that owes nothing to the exact-point code path being tested:
    off the condition the layer matrices are ordinarily conditioned.

    Cached because every case below wants it and it is a clean solve.
    """
    return _totals(_solve(0, fto, wavelength=np.nextafter(WL, WL + 1)))[0]


# ---------------------------------------------------------------- the regime is real

def test_the_stack_actually_sits_on_a_rayleigh_condition():
    """Guard the premise, the way ``test_stability.py`` guards its own.

    Without this the file could pass forever by having quietly stopped being hard — which is
    precisely how the existing stability suite missed this regime.
    """
    kx = np.arange(-9, 10) * WL / PERIOD
    assert np.min(np.abs(np.abs(kx) - N_SPACER)) == 0.0, (
        'no retained order lands exactly on the spacer index, so this stack is not at a '
        'Rayleigh condition and these tests prove nothing')


# ---------------------------------------------------------------- the regression itself

@pytest.mark.parametrize('backend, name', [(0, 'numpy'), (1, 'jax'), (2, 'torch')])
@pytest.mark.parametrize('fto', BAND_FTO)
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_exact_rayleigh_point_solves_correctly(backend, fto, name, algo):
    """The exact-point solve equals the detuned one, conserves energy, and says nothing.

    This is the regression for the D#81 formulation fix. Before it, these exact solves
    returned finite efficiencies wrong by orders of magnitude — with a warning on a good day
    (``R = 44.3`` at fto=5 ETM) and without one on a bad day (``R = 0.179`` against a true
    ``0.222`` at fto=5 SMM, every value in range, R+T = 0.98 reading as ordinary absorption).
    The error was *random* across fto and connecting algorithm, which is why this asserts
    tight agreement rather than a looser "not absurd" bound: a re-introduced instability
    would miss by orders of magnitude, not by a factor near TOL.
    """
    if backend == 1:
        pytest.importorskip('jax')
        import wavebit.on_jax  # noqa: F401  -- lifts x64 before the ucell is built
    elif backend == 2:
        pytest.importorskip('torch')

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(_solve(backend, fto, connecting_algo=algo))

    flagged = [r for r in rec if issubclass(r.category, UnphysicalResultWarning)]
    assert not flagged, f'{name}/{algo} warned on a solve that is now ordinary: {flagged}'
    assert abs(R - _truth(fto)) < TOL, (
        f'{name}/{algo} fto={fto}: exact-point R={R!r} vs detuned {_truth(fto)!r} — the '
        'exact Rayleigh solve has come apart from its one-ULP neighbour again')
    assert abs(R + T - 1.0) < TOL, (
        f'{name}/{algo} fto={fto}: R+T={R + T!r} on a lossless stack')


# ---------------------------------------------------------------- the 2D path

# The quiet-wrong class: exact-graze 2D stacks that returned R+T of 0.96-0.99 on a lossless
# stack with no warning at all — the exact signature that reads as ordinary absorption.
_QUIET_WRONG_2D = [(2.0, 'ETM'), (2.0, 'SMM'), (1.5, 'ETM')]


def _solve_2d(n_spacer, algo, wavelength=WL):
    pillar = np.ones((1, 2, 20))
    pillar[0, :, 6:14] = 3.5
    ucell = np.concatenate([pillar, np.full((1, 2, 20), n_spacer)], axis=0)
    return wavebit.rcwa(backend=0, wavelength=wavelength, fto=[5, 1], ucell=ucell,
                        period=[PERIOD, PERIOD], n_top=1.0, n_bot=1.5, theta=0.0, phi=0.0,
                        psi=np.pi / 2, thickness=[200.0, 300.0],
                        connecting_algo=algo).conv_solve()


@pytest.mark.parametrize('n_spacer, algo', _QUIET_WRONG_2D)
def test_2d_exact_graze_announces_itself(n_spacer, algo):
    """The 2D path still cannot solve the exact point — so it must say so, every time.

    ``layer_modes_2d`` loses the grazing column of ``V`` to cancellation and no eigen
    identity ports (see its comment), so the 2D contract is detection, not correctness:
    :class:`~wavebit._core.backend.RayleighPointWarning` fires whenever a layer eigenvalue is
    at round-off scale (threshold measured in ``backend._CUTOFF_REL``; 0 false positives over
    60 random non-grazing 2D solves). Before it, 5 of 48 exact-graze 2D solves returned
    R+T of 0.96-0.99 on this lossless stack with nothing fired — absorption that isn't there.
    """
    from wavebit._core.backend import RayleighPointWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            _solve_2d(n_spacer, algo)
        except Exception:
            return                              # loud is an acceptable outcome
    assert any(issubclass(r.category, RayleighPointWarning) for r in rec), (
        f'2D {algo} n_spacer={n_spacer}: exact-graze solve returned with no '
        'RayleighPointWarning — the quiet-wrong hole is open again')


def test_2d_detection_clears_off_the_condition():
    """The same stack detuned well off the point must not warn — that is the FP contract."""
    from wavebit._core.backend import RayleighPointWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        result = _solve_2d(2.0, 'ETM', wavelength=WL * (1 + 1e-8))
    R, T = _totals(result)
    assert not any(issubclass(r.category, RayleighPointWarning) for r in rec)
    assert abs(R + T - 1.0) < 1e-6


def test_exact_zero_eigenvalue_is_loud_on_every_backend():
    """``fto=3`` retains the grazing order with an eigenvalue that is *exactly* zero.

    There the mode partner is 0/0 — no nearby problem to stand on — and the NaN must reach
    :func:`~wavebit._core.backend.check_finite`. Raising and warning are both fine outcomes —
    a returned ``nan`` is not.

    The outcome per backend is collected rather than asserted in the loop, so the test states
    what it saw instead of passing by falling through three ``continue``\\ s.
    """
    outcomes = {}
    for backend, name in ((0, 'numpy'), (1, 'jax'), (2, 'torch')):
        if backend == 1:
            pytest.importorskip('jax')
            import wavebit.on_jax  # noqa: F401
        elif backend == 2:
            pytest.importorskip('torch')

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            try:
                got, _ = _totals(_solve(backend, 3))
            except Exception as exc:
                outcomes[name] = f'raised {type(exc).__name__}'
                continue
        if [r for r in rec if issubclass(r.category, UnphysicalResultWarning)]:
            outcomes[name] = f'warned (R={got:.6f})'
        else:
            outcomes[name] = f'RETURNED QUIETLY: R={got!r}'

    assert len(outcomes) == 3, f'a backend was skipped: {outcomes}'
    quiet = {k: v for k, v in outcomes.items() if v.startswith('RETURNED')}
    assert not quiet, f'exactly singular solve came back with nothing said: {quiet}'


# ---------------------------------------------------------------- where the line falls

def test_guards_read_concrete_jax_and_torch_values():
    """Eager JAX and PyTorch values are right there to be read, so they get checked.

    This is the unit-level statement of the same regression. It replaces a pair of tests that
    asserted the opposite — that both guards were no-ops off NumPy — which is the behaviour
    that let the band above go unreported.
    """
    for mod, mk in (('jax', lambda: _jax_arrays()), ('torch', lambda: _torch_arrays())):
        pytest.importorskip(mod)
        nan, big = mk()
        xp = array_namespace(big)
        with pytest.raises(FloatingPointError):
            check_finite(xp, de_ri=nan)
        with pytest.warns(UnphysicalResultWarning):
            check_physical(xp, big, big, 1.0)


def _jax_arrays():
    import wavebit.on_jax  # noqa: F401
    import jax.numpy as jnp
    return jnp.asarray([np.nan]), jnp.asarray([9.0])


def _torch_arrays():
    import torch
    return torch.tensor([float('nan')]), torch.tensor([9.0])


def test_guards_are_skipped_under_jit_and_do_not_break_the_trace():
    """A ``jit`` tracer has no value to read, so the guards must step aside silently.

    This is the constraint that rules out simply deleting the early return: reading a tracer
    either raises or forces a concretization that breaks the trace. The check therefore keys
    on whether the read succeeds, and a jitted solve is the case where it does not — so a
    traced non-finite result still returns quietly. That gap is real and is the accepted cost.
    """
    jax = pytest.importorskip('jax')
    import wavebit.on_jax  # noqa: F401
    import jax.numpy as jnp

    def f(x):
        xp = array_namespace(x)
        check_finite(xp, de_ri=x)
        check_physical(xp, x, x, 1.0)
        return jnp.sum(x)

    bad = jnp.asarray([np.nan, 9.0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = jax.jit(f)(bad)                   # must not raise, must not warn
    assert np.isnan(float(out))
    assert [r for r in rec if issubclass(r.category, UnphysicalResultWarning)] == []


def test_grad_is_guarded_and_still_differentiates():
    """An un-jitted ``jax.grad`` tracer carries a concrete primal, so the guard runs there.

    That asymmetry is the reason this formulation is worth having rather than a blanket skip
    for JAX: the gradient loop is exactly where D#81's reporter lives, and it is on the
    checked side of the line. The gradient itself must still be finite and still flow.
    """
    jax = pytest.importorskip('jax')
    import wavebit.on_jax  # noqa: F401
    import jax.numpy as jnp

    def loss(n):
        ucell = jnp.reshape(
            jnp.stack([jnp.ones((), dtype=jnp.complex128)] * 2
                      + [jnp.astype(n, jnp.complex128)] * 3
                      + [jnp.ones((), dtype=jnp.complex128)] * 3), (1, 1, 8))
        r = wavebit.rcwa(backend=1, wavelength=550.0, period=[700.0], fto=[10, 0],
                         n_top=1.0, n_bot=1.5, theta=0.1, phi=0.0, psi=np.pi / 2,
                         ucell=ucell, thickness=[220.0]).conv_solve()
        return jnp.real(r.R).sum()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        g = float(jax.grad(loss)(jnp.asarray(3.5)))
    assert np.isfinite(g) and g != 0.0
    # A healthy solve: the guard ran (grad is readable) and found nothing to say.
    assert [r for r in rec if issubclass(r.category, UnphysicalResultWarning)] == []
