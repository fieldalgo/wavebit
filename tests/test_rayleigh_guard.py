"""The guards fire wherever the values are concrete — not only on NumPy.

``tests/test_stability.py`` stresses the *evanescent* regime and does it well, but every case
in it is 1D TE at normal incidence, one layer, NumPy, ``period=3000`` / ``wavelength=400``.
That choice puts ``kx_m = 0.1333*m``, which never lands on ``n_top`` or ``n_bot``, so the file
systematically avoids the singularity this one is about — and being NumPy-only it could not
have seen the gap either way.

**The regime.** At a Rayleigh (grazing) condition a diffraction order has ``kz == 0`` in some
medium: the order is exactly at the light line of that layer, neither propagating nor decaying.
The layer's eigenproblem then has a degenerate zero eigenvalue and its mode matrix is singular,
so the ``inv`` in the stack algebra is inverting a matrix that has no inverse.

Exactly on the condition that is loud — LAPACK raises, or the NaN reaches
:func:`~wavebit._core.backend.check_finite`. The dangerous part is the *band around* it, where
the matrix is merely near-singular: ``inv`` returns without complaint and the answer is finite,
in range, and wrong by orders of magnitude. That is what these tests pin, because it is the
regime a gradient-based optimizer walks into and sits in.

The stack below puts the condition in an *interior* layer rather than at a half-space, which
is why it needs two layers: a patterned pillar over a uniform ``n=1.5`` spacer. With
``period=1000`` and ``wavelength=500``, ``kx_m = 0.5*m``, so order ``m=3`` sits at
``kx = 1.5 = n_spacer`` exactly.

What this file did not used to be able to say: on JAX and PyTorch the same solve returned
``R = 2.47`` and ``R = 243.9`` against a truth of ``0.215`` / ``0.311`` with no warning and no
error at all, because both guards keyed their eager-only rule on the *namespace* instead of on
whether the value was concrete. See :func:`wavebit._core.backend._concrete`.
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

# fto values inside the near-singular band. fto=3 is the first truncation that retains the
# grazing order at all, and it is the loud one (LAPACK raises); 4, 6 and 9 are the quiet ones.
BAND_FTO = [4, 6, 9]


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
    """The same stack with the wavelength nudged off the exact Rayleigh point.

    The physics is continuous across the condition — nothing resonates there, the singularity
    is in the *coordinates*, not in the structure — so a detuned solve is the correct answer
    to ~7 digits, and it is an oracle that owes nothing to the code path being tested.

    Cached because every case below wants it and it is a clean solve: recomputing it per
    parametrization was most of this file's runtime.
    """
    return _totals(_solve(0, fto, wavelength=WL + 1e-4))[0]


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


def test_the_band_really_does_return_a_wrong_answer():
    """The other half of the premise: these fto values are wrong, not merely suspicious.

    If a future change makes the band converge, these tests would still "pass" on a warning
    that no longer means anything. Pinning the wrongness keeps them honest.
    """
    for fto in BAND_FTO:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnphysicalResultWarning)
            got = _totals(_solve(0, fto))[0]
        assert abs(got - _truth(fto)) > 1.0, (
            f'fto={fto} no longer returns a badly wrong R; pick a new band')


# ---------------------------------------------------------------- the regression itself

@pytest.mark.parametrize('backend, name', [(0, 'numpy'), (1, 'jax'), (2, 'torch')])
@pytest.mark.parametrize('fto', BAND_FTO)
def test_near_singular_solve_is_never_silently_wrong(backend, fto, name):
    """A wrong answer has to announce itself on **every** backend.

    This is the regression. Before the guards were keyed on concreteness, ``backend=1`` and
    ``backend=2`` returned finite, badly wrong efficiencies here with an empty warning list —
    the two backends an optimizer actually runs on, silent in exactly the regime that breaks.

    The contract asserted is deliberately weak on *how* it reports: raising and warning are
    both acceptable answers, and which one happens depends on whether the degeneracy reaches
    LAPACK or arrives as a NaN, which is a property of the solve path rather than of the
    problem. Returning quietly is the only outcome ruled out.
    """
    if backend == 1:
        pytest.importorskip('jax')
        import wavebit.on_jax  # noqa: F401  -- lifts x64 before the ucell is built
    elif backend == 2:
        pytest.importorskip('torch')

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            got, _ = _totals(_solve(backend, fto))
        except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            # RuntimeError covers torch._C._LinAlgError, which does not derive from NumPy's.
            # No backend takes this path today — the whole point of these fto values is that
            # they come back quietly — so a skip appearing here is itself worth reading.
            pytest.skip(f'{name} raised before returning: {type(exc).__name__}')

    flagged = [r for r in rec if issubclass(r.category, UnphysicalResultWarning)]
    assert abs(got - _truth(fto)) > 1.0, 'premise: this case should be badly wrong'
    assert flagged, (
        f'{name} returned R={got:.6f} against a true {_truth(fto):.6f} with no warning at '
        'all — a near-singular solve must not come back quietly')


def test_exact_singularity_is_loud_on_every_backend():
    """``fto=3`` retains the grazing order with nothing to stabilise it.

    NumPy and PyTorch trip LAPACK; JAX propagated a NaN instead and used to hand it straight
    back. Raising and warning are both fine outcomes — a returned ``nan`` is not.

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
