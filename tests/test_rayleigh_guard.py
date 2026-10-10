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
* the case still refused — an eigenvalue that is *exactly* zero, where the 0/0 has no
  consistent nearby problem to stand on — refuses loudly, never quietly. *Which*
  truncations land there is a property of the LAPACK underneath rather than of this
  file, so it is read at runtime and not assumed; see :func:`_exact_point_or_refusal`.

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
from wavebit._core.backend import array_namespace, check_finite

PERIOD = 1000.0
WL = 500.0
# Both spellings of one spacer, and both are needed. The Rayleigh condition is a statement
# about `kx`, which is an index-scale quantity — order m=3 lands on `N_SPACER` exactly — so
# the assertion that pins the premise reads the index. Everything handed to wavebit reads the
# permittivity. 1.5**2 is exact, so the coincidence stays exact through the conversion.
N_SPACER = 1.5
EPS_SPACER = 2.25

# Patterned pillar (n=3.5 in air) over a uniform n=1.5 spacer. The spacer is the layer whose
# eigenproblem goes singular; the pillar is there to make the case an ordinary metasurface
# rather than a contrived one, and to populate the orders that reach the spacer.
_PILLAR = np.ones((1, 1, 20))
_PILLAR[0, 0, 6:14] = 12.25
_STACK = np.concatenate([np.full((1, 1, 20), EPS_SPACER), _PILLAR], axis=0)

_KW = dict(period=[PERIOD], top=1.0, bot=EPS_SPACER, theta=0.0, phi=0.0,
           psi=np.pi / 2, thickness=[300.0, 200.0])

# truncation_order values that retain the grazing order, and the ones that used to come back
# quietly wrong (R off by 1 to 600 against 0.215). truncation_order=3 is the first truncation
# that retains it at all. Whether a given order's eigenvalue comes out *exactly* zero — the
# refused 0/0 — or merely tiny is a property of the LAPACK underneath, so each case reads its
# own premise through `_exact_point_or_refusal` instead of this list asserting one.
BAND_MAX_ORDER = [4, 6, 9]

# The exact answer and the one-ULP-detuned answer may differ by at most this. The instability
# this file guards against produced *random* answers — errors of 1e0 to 1e2 — so the margin
# to a false alarm is ~10 orders; measured agreement is ~1e-9 (see layer_modes_1d_conical).
TOL = 1e-6


def _solve(backend, truncation_order, structure=None, wavelength=WL, auto_detune=False, **kw):
    # auto_detune off by default *here*: these tests pin the exact-point contracts, and the
    # library default would detune past the very regime under test. The default's own
    # contract has its own tests below.
    return wavebit.rcwa(backend=backend, wavelength=wavelength, truncation_order=[truncation_order, 0],
                        structure=_STACK if structure is None else structure,
                        **{**_KW, **kw}).solve(auto_detune=auto_detune)


def _totals(result):
    """``(sum R, sum T)`` as Python floats, from any backend's arrays."""
    return tuple(float(np.real(np.asarray(v, dtype=np.complex128)).sum())
                 for v in (result.R, result.T))


@functools.lru_cache(maxsize=None)
def _truth(truncation_order):
    """The same stack one ULP of wavelength off the exact Rayleigh point.

    The physics is continuous across the condition — nothing resonates there, the singularity
    is in the *coordinates*, not in the structure — so the detuned solve is the correct
    answer, and it is an oracle that owes nothing to the exact-point code path being tested:
    off the condition the layer matrices are ordinarily conditioned.

    Cached because every case below wants it and it is a clean solve.
    """
    return _totals(_solve('numpy', truncation_order, wavelength=np.nextafter(WL, WL + 1)))[0]


def _exact_point_or_refusal(call):
    """Run an exact-point solve, or hand this case to the refusal contract.

    The band's premise is a claim about round-off: at 4, 6 and 9 the FFT round-off was
    supposed to leave the grazing layer's eigenvalue at ~1e-16, so the mode partner is
    ``0/eps`` rather than ``0/0`` and the solve is ordinary. **Round-off is not portable.**
    Recording every eigenvalue :func:`wavebit._core.solve_1d.eig` returns and taking the
    one nearest zero:

        truncation_order    macOS arm64 / numpy 2.5.1    Windows x86-64 / numpy 2.5.2
                       3    0.0                          0.0
                       4    9.427073e-18                 0.0
                       6    3.938391e-17                 0.0
                       9    2.253776e-16                 0.0

    so on an exact-zero LAPACK *every* retained order is the ``truncation_order=3`` case
    and the solves-correctly contract has no cases left to run. It is not a margin story
    to be repaired by re-picking the orders: ``truncation_order=9``'s macOS residue is
    about one ULP of an order-1 quantity, nowhere near marginal, and it is zero on the
    other box anyway. Nor is it a property of the machine — on the *same* Windows box
    NumPy and JAX return the exact zero while PyTorch returns a residue and solves. The
    premise has to be read per case.

    Reading it through the outcome keeps this file out of the solver's internals and
    matches what the 2D graze tests above already do with their own platform-split
    premise: a refusal is one of the two allowed outcomes, it must be loud and carry the
    remedy, and the case then belongs to
    :func:`test_exact_zero_eigenvalue_is_loud_on_every_backend`. Nothing is lost by
    skipping here — :func:`test_auto_detune_lands_every_band_order` covers these orders
    end to end whichever class the platform puts them in, and a *quietly wrong* answer
    still fails, because it does not raise.

    Since 2026-09-26 the FFT part of that round-off is gone on every platform: the raster
    factorization transforms a staircase as ``a - a_0``, so a uniform layer is exactly
    ``eps * I`` at any grid size, where this stack's 20-pixel spacer used to come out 1.3e-16
    off. What still varies is LAPACK's own eigenvalue. On Linux x86-64 with NumPy 2.5.3 the
    change moved 6 cases here, and 2 of the graze-cluster cases, from the solves-correctly
    contract to the refusal one -- counted before the layers were listed bottom-to-top
    (2026-09-21 on develop), on the same physical stack. On macOS arm64 with NumPy 2.5.1 it
    moved all 23 cases here and 2 of the graze-cluster cases (45 passed before; 20 passed and
    25 took the refusal after; 2026-09-26): the macOS residues in the table above were the
    FFT's, they are now exact zeros, and there too the solves-correctly contract has no cases
    left.
    """
    try:
        return call()
    except FloatingPointError as exc:
        assert 'auto_detune' in str(exc), f'refusal without the remedy: {exc}'
        pytest.skip('this LAPACK lands the grazing eigenvalue exactly on zero, so this '
                    'case is the refusal contract rather than the solves-correctly one')


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

@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('truncation_order', BAND_MAX_ORDER)
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_exact_rayleigh_point_solves_correctly(backend, truncation_order, algo):
    """The exact-point solve equals the detuned one, conserves energy, and says nothing.

    This is the regression for the D#81 formulation fix. Before it, these exact solves
    returned finite efficiencies wrong by orders of magnitude — with a warning on a good day
    (``R = 44.3`` at truncation_order=5 ETM) and without one on a bad day (``R = 0.179`` against a true
    ``0.222`` at truncation_order=5 SMM, every value in range, R+T = 0.98 reading as ordinary absorption).
    The error was *random* across truncation_order and connecting algorithm, which is why this asserts
    tight agreement rather than a looser "not absurd" bound: a re-introduced instability
    would miss by orders of magnitude, not by a factor near TOL.
    """
    if backend == 'jax':
        pytest.importorskip('jax')
        wavebit.enable_x64()
    elif backend == 'torch':
        pytest.importorskip('torch')

    R, T = _exact_point_or_refusal(
        lambda: _totals(_solve(backend, truncation_order, layer_algorithm=algo)))

    assert abs(R - _truth(truncation_order)) < TOL, (
        f'{backend}/{algo} truncation_order={truncation_order}: exact-point R={R!r} vs detuned {_truth(truncation_order)!r} — the '
        'exact Rayleigh solve has come apart from its one-ULP neighbour again')
    assert abs(R + T - 1.0) < TOL, (
        f'{backend}/{algo} truncation_order={truncation_order}: R+T={R + T!r} on a lossless stack')


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
    pillar[0, :, 6:14] = 12.25
    # `n_spacer` names the index, because that is the quantity the graze condition is about;
    # squaring it here is the only conversion, and it is exact for both 2.0 and 1.5.
    cell = np.concatenate([np.full((1, 2, 20), n_spacer ** 2), pillar], axis=0)
    return wavebit.rcwa(backend='numpy', wavelength=wavelength, truncation_order=[5, 1], structure=cell,
                        period=[PERIOD, PERIOD], top=1.0, bot=2.25, theta=0.0, phi=0.0,
                        psi=np.pi / 2, thickness=[300.0, 200.0],
                        layer_algorithm=algo).solve(auto_detune=auto_detune)


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

    The backend/truncation_order matrix above runs at ``psi = pi/2``, which exercises only the s family's
    blocks. ``B`` is exactly as singular as ``A`` at the graze (same eigenvalues for a
    uniform layer), so a regression that broke only the ``B``-side identity would slip past
    a TE-only file. NumPy only: the backend axis is already covered above.
    """
    kw = dict(_KW)
    kw['psi'] = psi
    exact = _exact_point_or_refusal(
        lambda: wavebit.rcwa(backend='numpy', wavelength=WL, truncation_order=[5, 0], structure=_STACK,
                             layer_algorithm=algo, **kw).solve(auto_detune=False))
    near = wavebit.rcwa(backend='numpy', wavelength=np.nextafter(WL, WL + 1), truncation_order=[5, 0],
                        structure=_STACK, layer_algorithm=algo,
                        **kw).solve(auto_detune=False)
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
        mee = wavebit.rcwa(backend='numpy', wavelength=wl, truncation_order=[5, 0], structure=_STACK,
                           answers='field', **_KW)
        mee.solve(auto_detune=False)
        return np.asarray(mee.reconstruct_field(sample_x=16, sample_y=1, sample_z=16))

    exact = _exact_point_or_refusal(lambda: field(WL))
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
    """The exact-zero eigenvalue point (truncation_order=3) returns the nearby answer under the default.

    ``auto_detune=False`` keeps the loud refusal (pinned above); the default catches that
    refusal, retries off the point, and must agree with the one-ULP oracle.
    """
    from wavebit._core.backend import AutoDetuneWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(_solve('numpy', 3, auto_detune=True))
    assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert abs(R - _truth(3)) < TOL and abs(R + T - 1.0) < TOL


@pytest.mark.parametrize('case', ['1D', '1D ky', '2D'])
def test_auto_detune_retries_under_an_error_filter(case):
    """The refused attempt's arithmetic is not the caller's business, even under ``'error'``.

    At the exact point the layer build divides by a zero mode eigenvalue, and the NaN is
    refused or rebuilt downstream. NumPy's own ``RuntimeWarning`` for that division reached the
    caller once the ladder stopped catching warnings (2026-10-03), and under
    ``filterwarnings('error', category=RuntimeWarning)`` it *raised*, so the ladder never
    retried. ``backend.division_at_a_cutoff`` silences exactly those blocks: the 1D conical
    E/H blocks and the 2D ``V = Omega_R W / q``.

    ``'1D ky'`` lights the same stack at ``phi = pi/2``, so ``kx_3**2`` still meets the
    spacer's permittivity while ``ky != 0``. That is an x-cutoff, not a Rayleigh point -- the
    order's ``kz`` is ``|ky|``, not 0 -- and until 2026-10-10 it was this test's hardest case:
    the quotient was ``inf``, not NaN, the warning came from the *product* after the division
    (silencing the divisions alone passed ``'1D'`` and failed this one), the point refused, and
    the retry landed on the family construction's error plateau, 2.7e-6 off here, which the
    test could only bound by 1e-4 against a 1e-8-detuned oracle. The layer is now rebuilt in a
    regular basis (:func:`wavebit._core.solve_1d._regular_conical_modes`), whose divisors are
    guarded where they are discarded: nothing refuses, so nothing retries -- no
    :class:`AutoDetuneWarning` -- no RuntimeWarning reaches the ``'error'`` filter, and ``R`` is
    the 2D path's (a y-invariant raster, which never forms the families) at the exact point to
    1e-12. Removing the 1D ``division_at_a_cutoff`` block fails ``'1D'``; removing the 2D one
    fails ``'2D'``; the rebuild's guards are what ``'1D ky'`` pins now.
    """
    from wavebit._core.backend import AutoDetuneWarning
    tilt = dict(theta=0.3, phi=np.pi / 2)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        warnings.filterwarnings('error', category=RuntimeWarning)
        if case == '1D':
            R, T = _totals(_solve('numpy', 3, auto_detune=True))
        elif case == '1D ky':
            R, T = _totals(_solve('numpy', 3, auto_detune=True, **tilt))
        else:
            R, T = _totals(_solve_2d(1.5, 'ETM', auto_detune=True))
    if case == '1D':
        truth = _truth(3)
    elif case == '1D ky':
        truth, _ = _totals(wavebit.rcwa(
            backend='numpy', wavelength=WL, truncation_order=[3, 1],
            structure=np.repeat(_STACK, 2, axis=1),
            **{**_KW, **tilt, 'period': [PERIOD, 700.0]}).solve(auto_detune=False))
    else:
        truth, _ = _totals(_solve_2d(1.5, 'ETM', wavelength=WL * (1 + 1e-8)))
    assert not any(issubclass(r.category, RuntimeWarning) for r in rec), [str(r.message) for r in rec]
    if case == '1D':
        assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    elif case == '1D ky':
        assert not any(issubclass(r.category, AutoDetuneWarning) for r in rec), (
            [str(r.message) for r in rec])
    # 2D may rebuild on the first pass, platform by platform
    tol = {'1D': TOL, '1D ky': 1e-12, '2D': 1e-4}[case]
    assert abs(R - truth) < tol and abs(R + T - 1.0) < 1e-4


@pytest.mark.parametrize('truncation_order', BAND_MAX_ORDER)
def test_auto_detune_lands_every_band_order(truncation_order):
    """Whichever class this LAPACK puts a band order in, the default returns the truth.

    This is the coverage that survives :func:`_exact_point_or_refusal` skipping a case,
    and it is why that skip costs nothing. On a build that leaves a residue the order is
    an ordinary solve and the ladder stays silent; on one that returns the exact zero it
    refuses, the ladder retries one ULP off, and the rescued answer is the same one.
    Either way no :class:`RayleighPointWarning` reaches the caller and the total matches
    the detuned oracle — so 4, 6 and 9 stay pinned end to end on both kinds of build,
    and a regression that made them quietly wrong under the library default would fail
    here even on a platform where the exact-point matrix has nothing left to run.

    Measured across both outcomes: 1.3e-8 .. 3.6e-8 against ``_truth``, against
    ``TOL = 1e-6``.
    """
    from wavebit._core.backend import AutoDetuneWarning, RayleighPointWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(_solve('numpy', truncation_order, auto_detune=True))
    assert not any(issubclass(r.category, RayleighPointWarning) for r in rec), (
        f'truncation_order={truncation_order}: a RayleighPointWarning reached the caller: {rec}')
    for notice in (r for r in rec if issubclass(r.category, AutoDetuneWarning)):
        assert 'wavelength' in str(notice.message)
    assert abs(R - _truth(truncation_order)) < TOL, (
        f'truncation_order={truncation_order}: the default returned R={R!r} against the '
        f'detuned oracle {_truth(truncation_order)!r}')
    assert abs(R + T - 1.0) < TOL, (
        f'truncation_order={truncation_order}: R+T={R + T!r} on a lossless stack')


def test_auto_detune_is_silent_on_an_ordinary_solve():
    """Away from any degeneracy the default must change nothing and say nothing."""
    from wavebit._core.backend import AutoDetuneWarning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R_on, T_on = _totals(_solve('numpy', 5, wavelength=520.0, auto_detune=True))
    R_off, T_off = _totals(_solve('numpy', 5, wavelength=520.0))
    assert not any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert R_on == R_off and T_on == T_off


def test_auto_detune_accepts_an_explicit_step():
    """A float ``auto_detune`` is the relative step — the complex64 escape hatch."""
    from wavebit._core.backend import AutoDetuneWarning
    mee = wavebit.rcwa(backend='numpy', wavelength=WL, truncation_order=[3, 0], structure=_STACK, **_KW)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(mee.solve(auto_detune=1e-9))
    assert any(issubclass(r.category, AutoDetuneWarning) for r in rec)
    assert mee.wavelength == WL * (1 + 1e-9)
    assert abs(R + T - 1.0) < 1e-6


def _ordinary_solver():
    """The ordinary 520 nm stack: no degeneracy, so the ladder has nothing to retry."""
    return wavebit.rcwa(backend='numpy', wavelength=520.0, truncation_order=[5, 0], structure=_STACK,
                        **_KW)


def _constant_notice():
    """One fixed source line, so the default filter's registry can collapse repeats."""
    warnings.warn('a constant notice from inside the solve', UserWarning, stacklevel=1)


def test_auto_detune_lets_a_sweep_collapse_a_constant_warning():
    """The ladder catches nothing, so a constant-text warning is one report per sweep.

    The ladder used to wrap every solve in ``catch_warnings(record=True)`` to look for the
    Rayleigh warning, then replay the rest with ``warn_explicit``. ``catch_warnings`` bumps the
    filters version on entry and exit, which empties every ``__warningregistry__``. So under the
    default filter a constant-text warning came back once per sweep point on the default
    ``solve()``, where ``auto_detune=False`` showed it once. Measured before the change: 20
    reports for these 20 points.
    """
    solver = _ordinary_solver()
    original = solver._solve_built

    def noisy(geometry):
        _constant_notice()
        return original(geometry)

    solver._solve_built = noisy
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('default')
        for k in range(20):
            solver.solve(wavelength=520.0 + k)            # auto_detune=True, the default
    assert len([r for r in rec if 'constant notice' in str(r.message)]) == 1


def _grazing_on(calls_that_graze):
    """A `_solve_built` stand-in that trips the Rayleigh check on the listed calls (1-based)."""
    from wavebit._core.backend import check_mode_cutoff
    calls = []

    def install(solver):
        original = solver._solve_built

        def solve_built(geometry):
            calls.append(solver.wavelength)
            if len(calls) in calls_that_graze:
                check_mode_cutoff(array_namespace(np.zeros(1)), np.array([0.0, 1.0]))
            return original(geometry)

        solver._solve_built = solve_built
        return calls
    return install


def test_the_ladder_hears_the_rayleigh_check_on_its_own_channel():
    """A Rayleigh hit reaches the ladder without passing through the warnings machinery.

    The caller sees the retry (:class:`AutoDetuneWarning`, the moved wavelength) and not the
    hit of the attempt the ladder discarded. A channel that lost the hit would return the
    exact-point answer silently; one that leaked it would show a RayleighPointWarning for a
    condition the returned solve does not have.
    """
    from wavebit._core.backend import AutoDetuneWarning, RayleighPointWarning
    solver = _ordinary_solver()
    calls = _grazing_on({1})(solver)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        solver.solve()
    assert calls == [520.0, 520.0 * (1 + 1e-12)], calls
    assert not any(issubclass(r.category, RayleighPointWarning) for r in rec), rec
    notices = [str(r.message) for r in rec if issubclass(r.category, AutoDetuneWarning)]
    assert len(notices) == 1 and 'wavelength' in notices[0] and 'did not clear' not in notices[0]


def test_a_condition_the_ladder_cannot_clear_still_warns():
    """If every rung still grazes, the caller hears the Rayleigh warning once, and why."""
    from wavebit._core.backend import AutoDetuneWarning, RayleighPointWarning
    solver = _ordinary_solver()
    calls = _grazing_on({1, 2, 3})(solver)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        solver.solve()
    assert len(calls) == 2, calls                         # theta = 0: the wavelength rung only
    assert len([r for r in rec if issubclass(r.category, RayleighPointWarning)]) == 1
    notices = [str(r.message) for r in rec if issubclass(r.category, AutoDetuneWarning)]
    assert len(notices) == 1 and 'did not clear' in notices[0]


def test_the_rayleigh_check_warns_directly_outside_the_ladder():
    """With no ladder listening (``auto_detune=False``), the check is an ordinary warning."""
    from wavebit._core.backend import RayleighPointWarning
    solver = _ordinary_solver()
    _grazing_on({1})(solver)
    with pytest.warns(RayleighPointWarning):
        solver.solve(auto_detune=False)


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
        return wavebit.rcwa(backend='numpy', wavelength=WL, truncation_order=[2, 0],
                            structure=np.full((1, 1, 16), 1.5625), period=[PERIOD],
                            top=6.25, bot=2.25, theta=theta, phi=0.0, psi=np.pi / 2,
                            thickness=[300.0])

    # Premise: the exact point must actually refuse — otherwise the FFT was not noiseless
    # here and the test would pass without touching the ladder.
    try:
        solver().solve(auto_detune=False)
    except (FloatingPointError, np.linalg.LinAlgError):
        pass
    else:
        pytest.skip('the constructed point did not refuse; premise not reproducible here')

    mee = solver()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        R, T = _totals(mee.solve())
    notices = [r for r in rec if issubclass(r.category, AutoDetuneWarning)]
    assert notices and 'theta' in str(notices[0].message), (
        f'expected the theta rung to report itself, got {[str(n.message) for n in notices]}')
    assert mee.theta != theta, 'the solver must keep the nudged angle it solved with'
    assert abs(R + T - 1.0) < 1e-6


def test_auto_detune_keeps_gradients_flowing_through_the_retry():
    """The retry multiplies the knob, so an autodiff graph survives it.

    ``jax.grad`` through the refused exact-zero point (truncation_order=3): the first pass raises, the
    ladder re-solves at the detuned wavelength, and the gradient that comes back is the
    nearby problem's — finite, and nonzero for a parameter the reflectance depends on.
    """
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax.numpy as jnp

    def loss(thickness):
        r = wavebit.rcwa(backend='jax', wavelength=WL, truncation_order=[3, 0], structure=_STACK,
                         **_KW).solve(thickness=[thickness, 200.0])
        return jnp.real(jnp.asarray(r.R)).sum()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = float(jax.grad(loss)(300.0))
    assert np.isfinite(g) and g != 0.0


def test_exact_zero_eigenvalue_is_loud_on_every_backend():
    """``truncation_order=3`` retains the grazing order with an eigenvalue that is *exactly* zero.

    There the mode partner is 0/0 — no nearby problem to stand on — and the NaN must reach
    :func:`~wavebit._core.backend.check_finite`, which raises. A returned ``nan`` is not an
    acceptable outcome.

    This used to accept a warning as an alternative outcome, from ``check_physical``. That
    branch was dead: measured on all three backends at the time it was removed, every one
    raises here, and none of them reached the efficiency check at all.

    The outcome per backend is collected rather than asserted in the loop, so the test states
    what it saw instead of passing by falling through three ``continue``\\ s.
    """
    outcomes = {}
    for backend in ('numpy', 'jax', 'torch'):
        if backend == 'jax':
            pytest.importorskip('jax')
            wavebit.enable_x64()
        elif backend == 'torch':
            pytest.importorskip('torch')

        try:
            got, _ = _totals(_solve(backend, 3))
        except Exception as exc:
            outcomes[backend] = f'raised {type(exc).__name__}'
            continue
        outcomes[backend] = f'RETURNED QUIETLY: R={got!r}'

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


def _jax_arrays():
    wavebit.enable_x64()
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
    wavebit.enable_x64()
    import jax.numpy as jnp

    def f(x):
        xp = array_namespace(x)
        check_finite(xp, de_ri=x)
        return jnp.sum(x)

    bad = jnp.asarray([np.nan, 9.0])
    out = jax.jit(f)(bad)                       # must not raise
    assert np.isnan(float(out))


def test_grad_is_guarded_and_still_differentiates():
    """An un-jitted ``jax.grad`` tracer carries a concrete primal, so the guard runs there.

    That asymmetry is the reason this formulation is worth having rather than a blanket skip
    for JAX: the gradient loop is exactly where D#81's reporter lives, and it is on the
    checked side of the line. The gradient itself must still be finite and still flow.
    """
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax.numpy as jnp

    def loss(n):
        structure = jnp.reshape(
            jnp.stack([jnp.ones((), dtype=jnp.complex128)] * 2
                      + [jnp.astype(n, jnp.complex128)] * 3
                      + [jnp.ones((), dtype=jnp.complex128)] * 3), (1, 1, 8))
        r = wavebit.rcwa(backend='jax', wavelength=550.0, period=[700.0], truncation_order=[10, 0],
                         top=1.0, bot=2.25, theta=0.1, phi=0.0, psi=np.pi / 2,
                         structure=structure, thickness=[220.0]).solve()
        return jnp.real(r.R).sum()

    g = float(jax.grad(loss)(jnp.asarray(12.25)))   # a permittivity: the old n = 3.5
    assert np.isfinite(g) and g != 0.0
