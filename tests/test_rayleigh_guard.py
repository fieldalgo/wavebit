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
concrete. See :func:`wavebit._core.backend._concrete`. The guards still matter — the 2D
rebuild's fallback paths (a graze under ``jax.grad`` or a grad-carrying tensor ships the old
construction plus the warning; see ``solve_2d._rebuild_grazing_cluster``) and any
traced-and-jitted solve still rely on them.
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


def _solve(backend, fto, ucell=None, wavelength=WL, auto_detune=False, **kw):
    # auto_detune off by default *here*: these tests pin the exact-point contracts, and the
    # library default would detune past the very regime under test. The default's own
    # contract has its own tests below.
    return wavebit.rcwa(backend=backend, wavelength=wavelength, fto=[fto, 0],
                        ucell=_STACK if ucell is None else ucell,
                        **{**_KW, **kw}).conv_solve(auto_detune=auto_detune)


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

# The former quiet-wrong class: exact-graze 2D stacks that once returned R+T of 0.96-0.99
# on a lossless stack with no warning at all. The 2D contract is now correct-or-loud (see
# ``solve_2d._rebuild_grazing_cluster``): which of the two a stack gets is a property of
# this platform's round-off (the graze cluster's coupled mode rates are noise-split), so
# the portable assertion is on the *pair* of outcomes, and the platform-specific ones skip
# when their premise does not reproduce.
_GRAZE_2D = [(2.0, 'ETM'), (2.0, 'SMM'), (1.5, 'ETM'), (1.5, 'SMM')]


def _solve_2d(n_spacer, algo, wavelength=WL, auto_detune=False):
    pillar = np.ones((1, 2, 20))
    pillar[0, :, 6:14] = 3.5
    ucell = np.concatenate([pillar, np.full((1, 2, 20), n_spacer)], axis=0)
    return wavebit.rcwa(backend=0, wavelength=wavelength, fto=[5, 1], ucell=ucell,
                        period=[PERIOD, PERIOD], n_top=1.0, n_bot=1.5, theta=0.0, phi=0.0,
                        psi=np.pi / 2, thickness=[200.0, 300.0],
                        connecting_algo=algo).conv_solve(auto_detune=auto_detune)


@pytest.mark.parametrize('n_spacer, algo', _GRAZE_2D)
def test_2d_exact_graze_is_correct_or_loud(n_spacer, algo):
    """An exact-graze 2D solve either refuses loudly or returns the right answer.

    This is the flip of the old ``announces_itself`` detection pin, made possible by the
    coupled-cluster rebuild in ``layer_modes_2d``: a representable graze cluster is now
    re-derived from the first-order problem and solves exactly (measured: inside the
    two-sided sqrt-extrapolated oracle bracket, where the old construction was off by
    4.5e-3 .. 4.1e0 with R+T up to 42); a numerically defective one (the ``{1, z}``
    regime) raises with the detune advice. What may never happen again is the third
    outcome this test's predecessor guarded: finite, in-range, quietly wrong.
    """
    R_far, T_far = _totals(_solve_2d(n_spacer, algo, wavelength=WL * (1 + 1e-8)))
    try:
        R, T = _totals(_solve_2d(n_spacer, algo))
    except (FloatingPointError, np.linalg.LinAlgError):
        return                                  # loud is one of the two allowed outcomes
    assert abs(R - R_far) < 1e-4 and abs(R + T - 1.0) < 1e-4, (
        f'2D {algo} n_spacer={n_spacer}: exact-graze solve returned quietly with '
        f'R={R!r} (far-detuned truth {R_far!r}), R+T={R + T!r} — the quiet-wrong hole '
        'is open again')


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_2d_representable_graze_solves_exactly(algo):
    """On this platform the n=2.0 graze cluster is representable — pin the exact solve.

    The rebuilt exact point must agree with its one-ULP neighbour (both through the
    rebuild; the physics is continuous across the condition), conserve energy, and say
    nothing at all — the same contract the conical path earned. Measured agreement is
    ~1e-9 against a two-sided sqrt-extrapolated oracle; asserted at TOL for slack across
    LAPACK builds. If this platform's round-off makes the cluster defective instead, the
    premise is gone and the test skips — ``correct_or_loud`` above still covers the stack.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            R, T = _totals(_solve_2d(2.0, algo))
            R_ulp, _ = _totals(_solve_2d(2.0, algo, wavelength=np.nextafter(WL, WL + 1)))
        except (FloatingPointError, np.linalg.LinAlgError):
            pytest.skip('this platform rounds the n=2.0 graze cluster defective; the '
                        'correct-or-loud contract is pinned separately')
    said = [r for r in rec if issubclass(r.category, UserWarning)]
    assert not said, f'2D {algo}: the rebuilt exact solve warned: {[str(r.message) for r in said]}'
    assert abs(R - R_ulp) < TOL and abs(R + T - 1.0) < TOL


def test_2d_defective_graze_refuses_with_advice():
    """On this platform the n=1.5 graze cluster is numerically defective — pin the refusal.

    Its coupled mode rates split below working precision (the true limit basis is
    ``{1, z}``, which no exponential mode basis represents — the same fact §5 of
    ``docs/rayleigh-points.md`` records for the exactly-zero eigenvalue), so returning
    numbers here would mean returning the measured 1e-4..1e0 errors quietly. The refusal
    must carry the remedy. If this platform's round-off splits the cluster representable
    instead, the premise is gone and the test skips.
    """
    try:
        _solve_2d(1.5, 'ETM')
    except FloatingPointError as exc:
        assert 'auto_detune' in str(exc), f'refusal without the remedy: {exc}'
        return
    except np.linalg.LinAlgError:
        return                                  # loud in LAPACK's own voice — acceptable
    pytest.skip('this platform rounds the n=1.5 graze cluster representable; the '
                'correct-or-loud contract is pinned separately')


def test_2d_detection_clears_off_the_condition():
    """The same stack detuned well off the point must not warn — that is the FP contract."""
    from wavebit._core.backend import RayleighPointWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        result = _solve_2d(2.0, 'ETM', wavelength=WL * (1 + 1e-8))
    R, T = _totals(result)
    assert not any(issubclass(r.category, RayleighPointWarning) for r in rec)
    assert abs(R + T - 1.0) < 1e-6


@pytest.mark.parametrize('psi', [0.0, np.pi / 4], ids=['tm', 'mixed'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_exact_point_solves_for_tm_and_mixed_polarization(psi, algo):
    """The p family (``V_22`` / ``B``) takes the same reformulated path — pin it too.

    The backend/fto matrix above runs at ``psi = pi/2``, which exercises only the s family's
    blocks. ``B`` is exactly as singular as ``A`` at the graze (same eigenvalues for a
    uniform layer), so a regression that broke only the ``B``-side identity would slip past
    a TE-only file. NumPy only: the backend axis is already covered above.
    """
    kw = dict(_KW)
    kw['psi'] = psi
    exact = wavebit.rcwa(backend=0, wavelength=WL, fto=[5, 0], ucell=_STACK,
                         connecting_algo=algo, **kw).conv_solve(auto_detune=False)
    near = wavebit.rcwa(backend=0, wavelength=np.nextafter(WL, WL + 1), fto=[5, 0],
                        ucell=_STACK, connecting_algo=algo,
                        **kw).conv_solve(auto_detune=False)
    R, T = _totals(exact)
    R_n, _ = _totals(near)
    assert abs(R - R_n) < TOL and abs(R + T - 1.0) < TOL


def test_exact_point_field_reconstruction_matches_its_neighbour():
    """The fields were broken over a *band*, not a point — this pins the repaired band.

    Pre-fix, ETM field reconstruction at one ULP off the condition was wrong at 2e18 scale
    and only converged ~1e4 ULP out, while the efficiencies looked fine — so an
    efficiency-only file would let that regression back in. Post-fix the exact-point field
    equals its one-ULP neighbour to ~3e-8 (measured); asserted at 1e-5 for slack across
    LAPACK builds.
    """
    def field(wl):
        mee = wavebit.rcwa(backend=0, wavelength=wl, fto=[5, 0], ucell=_STACK, **_KW)
        mee.conv_solve(auto_detune=False)
        return np.asarray(mee.calculate_field(res_x=16, res_y=1, res_z=16))

    exact = field(WL)
    near = field(np.nextafter(WL, WL + 1))
    assert np.all(np.isfinite(exact))
    scale = np.max(np.abs(near))
    assert np.max(np.abs(exact - near)) / scale < 1e-5


# ---------------------------------------------------------------- auto_detune (the default)

def test_auto_detune_default_lands_every_2d_graze_correct():
    """The library default returns the right answer on every graze stack, however it goes.

    A representable cluster solves exactly on the first pass and the ladder stays silent —
    the flip of this test's predecessor, which pinned a retry on the very stack the rebuild
    now solves. A defective cluster refuses, the ladder retries at
    ``wavelength*(1+1e-12)``, says so in an :class:`AutoDetuneWarning`, and the landing
    point sits inside the rebuild's regime (that is what sizes ``solve_2d._COUPLED_REL``),
    so the rescued answer is the nearby problem's, not the degraded band's. Either way no
    :class:`RayleighPointWarning` reaches the caller and the answer matches a far-detuned
    solve — on stacks that once returned R+T of 0.96-0.99 with nothing said.
    """
    from wavebit._core.backend import AutoDetuneWarning, RayleighPointWarning
    for n_spacer, algo in _GRAZE_2D:
        R_far, T_far = _totals(_solve_2d(n_spacer, algo, wavelength=WL * (1 + 1e-8)))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            result = _solve_2d(n_spacer, algo, auto_detune=True)
        R, T = _totals(result)
        assert not any(issubclass(r.category, RayleighPointWarning) for r in rec), (
            f'{algo} n={n_spacer}: a RayleighPointWarning reached the caller: {rec}')
        notices = [r for r in rec if issubclass(r.category, AutoDetuneWarning)]
        for n in notices:
            assert 'wavelength' in str(n.message)
        assert abs(R - R_far) < 1e-4 and abs(R + T - 1.0) < 1e-4, (
            f'{algo} n={n_spacer}: default solve returned R={R!r} against far-detuned '
            f'{R_far!r} (detuned retry: {bool(notices)})')


def test_auto_detune_rescues_the_refused_point():
    """The exact-zero eigenvalue point (fto=3) returns the nearby answer under the default.

    ``auto_detune=False`` keeps the loud refusal (pinned above); the default catches that
    refusal, retries off the point, and must agree with the one-ULP oracle.
    """
    from wavebit._core.backend import AutoDetuneWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(_solve(0, 3, auto_detune=True))
    assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert abs(R - _truth(3)) < TOL and abs(R + T - 1.0) < TOL


def test_auto_detune_is_silent_on_an_ordinary_solve():
    """Away from any degeneracy the default must change nothing and say nothing."""
    from wavebit._core.backend import AutoDetuneWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R_on, T_on = _totals(_solve(0, 5, wavelength=520.0, auto_detune=True))
    R_off, T_off = _totals(_solve(0, 5, wavelength=520.0))
    assert not any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert R_on == R_off and T_on == T_off


def test_auto_detune_accepts_an_explicit_step():
    """A float ``auto_detune`` is the relative step — the complex64 escape hatch."""
    from wavebit._core.backend import AutoDetuneWarning
    mee = wavebit.rcwa(backend=0, wavelength=WL, fto=[3, 0], ucell=_STACK, **_KW)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(mee.conv_solve(auto_detune=1e-9))
    assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert mee.wavelength == WL * (1 + 1e-9)
    assert abs(R + T - 1.0) < 1e-6


def _theta_exactly_on(n_top, target):
    """An incidence angle with ``n_top * sin(theta) == target`` *bitwise*, or None.

    The equality this file needs is a property of libm's rounding, not of mathematics, so it
    is searched for rather than assumed: step outward one ULP at a time from ``arcsin`` and
    take the first angle whose product lands exactly. A platform whose libm never lands on
    it makes the caller skip — a skipped premise is honest where a silently-vacuous test is
    not.
    """
    base = float(np.arcsin(target / n_top))
    lo = hi = base
    candidates = [base]
    for _ in range(60):
        lo = np.nextafter(lo, 0.0)
        hi = np.nextafter(hi, 2.0)
        candidates += [lo, hi]
    for th in candidates:
        if n_top * np.sin(th) == target:
            return float(th)
    return None


def test_auto_detune_theta_rung_clears_a_wavelength_invariant_graze():
    """The second rung: a graze the wavelength cannot move, the incidence angle can.

    ``kx_0 = n_top*sin(theta)`` carries no wavelength, so when the *zeroth* order sits
    exactly on a uniform layer's index the first rung retries into the identical problem
    and fails again. The ladder then nudges ``theta``, which moves every order at once.
    Constructed so the layer eigenvalue is exactly zero (all quantities binary-exact:
    2.5 * 0.5 == 1.25, 1.25**2 == 1.5625), which is the refusing case.
    """
    from wavebit._core.backend import AutoDetuneWarning
    theta = _theta_exactly_on(2.5, 1.25)
    if theta is None:
        pytest.skip("this platform's libm never lands n_top*sin(theta) on 1.25 exactly")

    def solver():
        return wavebit.rcwa(backend=0, wavelength=WL, fto=[2, 0],
                            ucell=np.full((1, 1, 16), 1.25), period=[PERIOD],
                            n_top=2.5, n_bot=1.5, theta=theta, phi=0.0, psi=np.pi / 2,
                            thickness=[300.0])

    # Premise: the exact point must actually refuse — otherwise the FFT was not noiseless
    # here and the test would pass without touching the ladder.
    try:
        solver().conv_solve(auto_detune=False)
    except (FloatingPointError, np.linalg.LinAlgError):
        pass
    else:
        pytest.skip('the constructed point did not refuse; premise not reproducible here')

    mee = solver()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(mee.conv_solve())
    notices = [r for r in rec if issubclass(r.category, AutoDetuneWarning)]
    assert notices and 'theta' in str(notices[0].message), (
        f'expected the theta rung to report itself, got {[str(n.message) for n in notices]}')
    assert mee.theta != theta, 'the solver must keep the nudged angle it solved with'
    assert abs(R + T - 1.0) < 1e-6


def test_auto_detune_keeps_gradients_flowing_through_the_retry():
    """The retry multiplies the knob, so an autodiff graph survives it.

    ``jax.grad`` through the refused exact-zero point (fto=3): the first pass raises, the
    ladder re-solves at the detuned wavelength, and the gradient that comes back is the
    nearby problem's — finite, and nonzero for a parameter the reflectance depends on.
    """
    jax = pytest.importorskip('jax')
    import wavebit.on_jax  # noqa: F401
    import jax.numpy as jnp

    def loss(thickness):
        r = wavebit.rcwa(backend=1, wavelength=WL, fto=[3, 0], ucell=_STACK,
                         **_KW).conv_solve(thickness=[200.0, thickness])
        return jnp.real(jnp.asarray(r.R)).sum()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = float(jax.grad(loss)(300.0))
    assert np.isfinite(g) and g != 0.0


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
