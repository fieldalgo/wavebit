"""Floating-point hygiene: a clean warning stream, and guards on the values that leave a solve.

A strongly evanescent stack drives ``exp(-k0*q*d)`` into the double subnormal range on purpose,
so this codebase does work at the edges of the format. For a long time it looked like that was
being *reported*: on macOS a single 2D solve produced hundreds of ``RuntimeWarning: overflow /
divide by zero / invalid value encountered in matmul``, and the obvious reading was that the
extreme arithmetic was setting FPU flags.

It was not. The cascade was a NumPy bug in the complex ``matmul`` FP-status check, present
through 2.3.0 and fixed in 2.3.1. Measured on the same machine, same Accelerate BLAS, same
bytes in: 456 warnings on 2.3.0, zero on 2.3.1, identical efficiencies either way. A matrix
with condition number 12 and no entry above 2.1 in magnitude was enough to trigger it.

That is why ``pyproject.toml`` floors NumPy at 2.3.1 instead of the library suppressing anything: on
a supported NumPy there are no spurious flags to hide, and hiding them would only ever cost the
user a real warning later.

What this file pins:

* the warning stream is clean on every user-facing path — both connecting algorithms, all three
  factorizations, the parametric builder, field reconstruction — which is what the version floor
  buys and would catch a regression to a broken NumPy;
* the result is right, checked by energy conservation rather than a frozen number;
* contamination is loud where it announces itself: a non-finite efficiency raises, pointing
  at the structure rather than at the arithmetic.
"""

from wavebit._common.backend import PrecisionLossWarning as _legacy_wavebit_PrecisionLossWarning

from wavebit._common.backend import UnresolvedBranchWarning as _legacy_wavebit_UnresolvedBranchWarning

from wavebit._common.backend import _enable_x64 as _legacy_wavebit_enable_x64

from wavebit.lattice import meshgrid_xy as _legacy_wavebit_meshgrid_xy

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path, repository_root
import json
import warnings

import numpy as np
import pytest

from wavebit._common.backend import PrecisionLossWarning, UnresolvedBranchWarning, array_namespace, check_finite
from wavebit.rcwa._engine.solve_1d import final_matching

# The messages NumPy emits for the FP status flags. Anything else a solve warns about is a
# genuine finding and should not be silenced by this file's assertions.
_FP_WORDS = ('overflow', 'underflow', 'divide by zero', 'invalid value')

WL = 500.0


def _fp_warnings(records):
    return [str(r.message) for r in records
            if issubclass(r.category, RuntimeWarning)
            and any(w in str(r.message) for w in _FP_WORDS)]


def _run(**kwargs):
    """Solve, returning ``(result, fp_warning_messages)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = _legacy_wavebit_rcwa(backend='numpy', wavelength=WL, **kwargs).solve()
    return res, _fp_warnings(rec)


def _energy(res):
    return float(np.sum(res.incident.R) + np.sum(res.incident.T))


# --- the structures: thick, high-truncation, deeply evanescent -----------------------------
# A 600nm rod at 500nm wavelength with truncation_order=8 retains orders whose decay over one layer is
# ~exp(-100): the regime that made the broken NumPy shout loudest.

#: 3.48**2 and 2.0**2 -- the same two media, as the permittivity `structure` now holds.
EPS_SI, EPS_MID = 12.1104, 4.0


def _structure_2d():
    u = np.ones((1, 64, 64))
    u[0, 16:48, 16:48] = EPS_SI
    return u


def _structure_1d():
    u = np.ones((2, 1, 256))
    u[1, 0, 64:192] = EPS_SI
    u[0, 0, :] = EPS_MID
    return u


def test_numpy_is_new_enough_to_report_fp_flags_correctly():
    """The floor in pyproject.toml is the whole fix; state it once, here, in the open."""
    major, minor, patch = (int(p) for p in np.__version__.split('.')[:3])
    assert (major, minor, patch) >= (2, 3, 1), (
        f'NumPy {np.__version__} miscounts FP status flags in complex matmul and will emit '
        'hundreds of spurious overflow/divide-by-zero warnings from any solve. '
        'pyproject.toml requires numpy>=2.3.1.')


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
def test_thick_2d_stack_is_silent_and_conserves_energy(layer_algorithm):
    res, fp = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], truncation_order=[8, 8],
                   psi=0.0, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('psi', [np.pi / 2, 0.0], ids=['te', 'tm'])
def test_thick_1d_stack_is_silent_and_conserves_energy(layer_algorithm, psi):
    res, fp = _run(structure=_structure_1d(), thickness=[400, 800], period=[700], truncation_order=[8, 0],
                   psi=psi, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
def test_conical_stack_is_silent_and_conserves_energy(layer_algorithm):
    res, fp = _run(structure=_structure_1d(), thickness=[400, 800], period=[700], truncation_order=[8, 0],
                   psi=np.pi / 2, phi=0.3, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('strategy',
                         [dict(fourier_coefficients=c, factorization=f)
                          for c in ('sampled', 'integrated') for f in ('li', 'nvf')],
                         ids=['sampled-li', 'sampled-nvf', 'integrated-li', 'integrated-nvf'])
def test_every_factorization_is_silent(strategy):
    """The convolution builders were the loudest sites of all under the broken NumPy."""
    res, fp = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], truncation_order=[6, 6],
                   psi=0.0, bot=EPS_SI, **strategy)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


def test_parametric_path_is_silent():
    """The parametric (staircase-free) builder reaches the shared NVF closing through
    ``nvf_from_coeffs``."""
    structure = [dict(background=1.0, shapes=[dict(type='ellipse', cx=0.0, cy=0.0, lx=0.5,
                                              ly=0.3, angle=np.deg2rad(40), material=5.76),
                                      dict(type='circle', cx=-0.2, cy=-0.3, lx=0.2,
                                              material=9.0)])]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = _legacy_wavebit_rcwa(backend='numpy', structure=structure, truncation_order=(6, 6), top=1.0, bot=1.0,
                             theta=np.deg2rad(5), phi=np.deg2rad(20), period=(1.0, 1.2),
                             wavelength=1.5, thickness=(0.3,), psi=0.35 * np.pi).solve()
    assert _fp_warnings(rec) == []
    assert abs(_energy(res) - 1.0) < 1e-10


def test_field_reconstruction_is_silent_and_finite():
    """Reconstruction re-walks the same modal exponentials the solve cascaded."""
    sim = _legacy_wavebit_rcwa(backend='numpy', structure=_structure_2d(), thickness=[600], period=[700, 700],
                         wavelength=WL, truncation_order=[6, 6], psi=0.0, bot=EPS_SI, answers='field')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res, field = sim.solve_field(sample_x=8, sample_y=8, sample_z=8)
    assert _fp_warnings(rec) == []
    assert bool(np.all(np.isfinite(field)))
    assert abs(_energy(res) - 1.0) < 1e-10


def test_solving_does_not_touch_global_seterr():
    """A library must not change the caller's FP error policy — nothing here sets np.seterr."""
    before = np.geterr()
    try:
        np.seterr(over='raise', divide='raise', invalid='raise')
        during = np.geterr()
        _run(structure=_structure_2d(), thickness=[600], period=[700, 700], truncation_order=[6, 6],
             psi=0.0, bot=EPS_SI)
        assert np.geterr() == during
    finally:
        np.seterr(**before)


def test_caller_raise_policy_survives_a_solve():
    """``np.seterr(all='raise')`` is what a careful user sets while hunting a NaN.

    On a supported NumPy the solve raises no flags at all, so it must run straight through —
    the sharpest statement available that the arithmetic really is clean, rather than merely
    quiet.
    """
    before = np.geterr()
    try:
        np.seterr(over='raise', divide='raise', invalid='raise', under='raise')
        res, _ = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], truncation_order=[8, 8],
                      psi=0.0, bot=EPS_SI)
        assert abs(_energy(res) - 1.0) < 1e-10
    finally:
        np.seterr(**before)


# --- tier 1: non-finite raises --------------------------------------------------------------

def test_check_finite_passes_finite_values():
    xp = array_namespace(np.zeros(1))
    check_finite(xp, R=np.array([0.4, 0.1]), T=np.array([0.5]))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_check_finite_raises_with_a_structural_hint(bad):
    xp = array_namespace(np.zeros(1))
    with pytest.raises(FloatingPointError) as e:
        # check_finite is internal; its call sites still pass the internal de_ri/de_ti
        # names (the public rename to R/T lives in the Result container, not here).
        check_finite(xp, de_ri=np.array([0.4, bad]), de_ti=np.array([0.5]))
    msg = str(e.value)
    assert 'de_ri' in msg and 'de_ti' not in msg      # names only the contaminated array
    assert 'truncation_order' in msg and 'degenerate' in msg       # says what to do about it


def test_finiteness_guard_fires_from_the_solve_path():
    """A contaminated boundary system must raise out of the shared final matching.

    Injecting at ``big_F`` is the honest way in: NaN reaching the efficiencies is exactly what
    a degenerate interface produces, and it exercises the guard where a solve would hit it.
    """
    xp = array_namespace(np.zeros(1))
    C = np.complex128
    eye = np.eye(2, dtype=C)
    big_F = eye.copy()
    big_F[0, 0] = np.inf
    kz = np.ones(1, dtype=C)
    with pytest.raises(FloatingPointError, match='non-finite'):
        final_matching(xp, C, big_F, eye.copy(), eye.copy(), kz, kz,
                       incidence=(0.0, 1.0), theta=0.0, n_top=1.0, n_bot=1.0,
                       n_orders_u=1, n_orders_v=1, kx=np.zeros(1, dtype=C),
                       ky=np.zeros(1, dtype=C))


# --- the line is concreteness, not the namespace ---------------------------------------------
#
# These two used to assert the opposite: that both guards were no-ops on anything that was not
# eager NumPy. That was keyed on the namespace, and it meant an ordinary eager `backend='jax'` /
# `backend='torch'` call — where the value is right there to be read — went unguarded. A
# near-singular stack then returned R = 2.47 (JAX) and 243.9 (PyTorch) against a truth of
# 0.215, finite and with an empty warning list. tests/rcwa/test_rayleigh_guard.py pins that case
# end to end; these pin the adapter-level rule it rests on.

def test_guards_read_eager_jax_values():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    xp = array_namespace(jnp.zeros(1))
    with pytest.raises(FloatingPointError):
        check_finite(xp, R=jnp.asarray([np.nan]))


def test_guards_read_eager_torch_values():
    torch = pytest.importorskip('torch')
    xp = array_namespace(torch.zeros(1))
    with pytest.raises(FloatingPointError):
        check_finite(xp, R=torch.tensor([float('nan')]))


def test_guards_step_aside_for_a_jit_tracer():
    """The reason the early return cannot simply be deleted.

    Under ``jit`` the value is abstract: reading it raises, and forcing it concrete would
    break the trace. So the guards must notice they cannot read and say nothing — which is
    also why a traced non-finite result still returns quietly. Accepted, and documented on
    :func:`wavebit._common.backend._concrete`.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    def f(x):
        xp = array_namespace(x)
        check_finite(xp, R=x)
        return jnp.sum(x)

    out = jax.jit(f)(jnp.asarray([np.nan, 9.0]))
    assert np.isnan(float(out))


# --- the other tier: a branch decided from round-off ---------------------------------------------
#
# check_finite catches contamination that announces itself, and it reads the *result*. This
# tier reads the arithmetic: a sqrt branch is a discrete choice, and at complex64 the quantity
# it is taken from is often pure round-off. The answer stays finite (the degenerate case is
# perfectly unitary), so the tier above it does not fire. See docs/theory/rcwa/branch-cuts.md.

#: A 1/5 index grating, written as its permittivities: 1 and 25.
_BRANCH_STRUCTURE = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1


def _branch_solve(complex_dtype, top=1.0, theta=0.4, phi=0.3, backend='numpy', layer_algorithm='ETM'):
    return _legacy_wavebit_rcwa(
        backend=backend, psi=np.pi / 2, top=top, bot=25.0, theta=theta, phi=phi, truncation_order=[30],
        wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
        complex_dtype=complex_dtype, fourier_coefficients='integrated',
        factorization='li', layer_algorithm=layer_algorithm).solve()


def _branch_warnings(**kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _branch_solve(**kw)
    return [r for r in rec if issubclass(r.category, UnresolvedBranchWarning)]


_COMPLEX_THETA = np.arcsin(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4, dtype=np.complex128))
_COMPLEX_PHI = np.arctan(1.2 / 1.6001, dtype=np.complex128)

_REGIMES = {
    'real angle': dict(),
    'complex angle': dict(theta=_COMPLEX_THETA, phi=_COMPLEX_PHI),
    # (1.4 + 1e-5j)**2 -- the weak absorber, as permittivity. A real angle there is a phase
    # angle with a real k_par (2026-09), so its half-space roots are resolved, and so are ETM's
    # layer labels (2026-10-02).
    'weak absorber': dict(top=1.9599999998999997 + 2.8e-05j, theta=0.6, phi=0.4),
}


@pytest.mark.parametrize('regime', list(_REGIMES))
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_double_precision_resolves_every_branch(backend, regime):
    """Silence at complex128 is the claim that makes the warning worth having."""
    assert _branch_warnings(complex_dtype=0, backend=backend, **_REGIMES[regime]) == []


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('regime', list(_REGIMES))
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_single_precision_reads_no_branch_from_round_off(backend, regime, algorithm):
    """complex64 now answers every branch question these regimes ask, and says nothing.

    Every decision an ETM solve reads is at least ``|v|`` from zero: the half-space's
    ``Re(kz) + Im(kz)`` (Maystre's cut) and the layer label's ``Re(q) - Im(q)``
    (``etm_mode_q``), both since 2026-10-02. Neither is round-off, so the warning stays quiet and
    the answer holds -- within 3e-6 of double precision on every backend here. SMM's layer labels
    still come from a round-off ``Re(q)``, but SMM does not depend on them, so they are not
    checked: they used to warn on every one of these solves while the answer sat within 6.4e-6.
    In 2026-08 the same point was 1.80 off on NumPy, 4.12 on JAX and 5.68 on PyTorch, and ETM
    was still 5e-2 off until its labels moved.

    Every backend, deliberately: unlike the guards in ``src/wavebit/_common/backend.py`` this check is not
    NumPy-only, and the backends it would have skipped are the paths taken for gradients and
    GPUs. Where a decision genuinely sits on a cut it still fires; see the test below.
    """
    kw = dict(_REGIMES[regime], backend=backend, layer_algorithm=algorithm)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        single = _branch_solve(1, **kw).s_inc.r_s
    assert [r for r in rec if issubclass(r.category, UnresolvedBranchWarning)] == []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        double = _branch_solve(0, **kw).s_inc.r_s
    error = np.abs(np.asarray(single, dtype=np.complex128)
                   - np.asarray(double, dtype=np.complex128)).max()
    assert error < 1e-5, error


def _on_a_cut(backend, which, single=True):
    """One branch decision placed 4e-7 off its cut, and the UnresolvedBranchWarnings it raised.

    A margin of ~3e-7 on a root of size 1 is inside complex64's round-off (8 eps ~ 1e-6) and
    far outside complex128's. ``'half-space'`` puts ``kz**2`` beside the negative imaginary axis
    (Maystre's cut); ``'layer'`` puts ``q**2`` beside the positive imaginary one, which is the
    same cut in ``q = -i*kz``.
    """
    from wavebit._common.conventions import boundary_kz, etm_mode_q
    if backend == 'numpy':
        make = lambda v: np.asarray(v, dtype=np.complex64 if single else np.complex128)
    elif backend == 'jax':
        jnp = pytest.importorskip('jax.numpy')
        make = lambda v: jnp.asarray(v, dtype=jnp.complex64 if single else jnp.complex128)
    else:
        torch = pytest.importorskip('torch')
        make = lambda v: torch.tensor(v, dtype=torch.complex64 if single else torch.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        if which == 'half-space':
            eps = make(4e-7 - 1j)
            boundary_kz(array_namespace(eps), eps, make(0j))
        else:
            eigenvalue = make(-4e-7 + 1j)
            etm_mode_q(array_namespace(eigenvalue), eigenvalue, 0.7)
    return [r for r in rec if issubclass(r.category, UnresolvedBranchWarning)]


@pytest.mark.parametrize('which', ['half-space', 'layer'])
@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
def test_a_decision_on_its_cut_still_reports_at_single_precision(backend, which):
    """The warning is not retired with the round-off it used to report: a cut still has one.

    Off a cut, neither the half-space's nor ETM's decision can be flipped by round-off. On one --
    ``kz**2`` or ``q**2`` within round-off of the imaginary axis it is cut along -- it can, and
    then complex64 says so while complex128, four hundred million times finer, stays quiet.
    """
    assert len(_on_a_cut(backend, which)) == 1
    assert _on_a_cut(backend, which, single=False) == []


def test_a_traced_single_precision_solve_says_it_could_not_be_checked():
    """Silence must not mean two different things.

    Under ``jit`` the values are tracers, so the check cannot run — and that is exactly where a
    wrong branch is least likely to be noticed: measured ``|R| = 3.93`` traced at complex64
    against 0.67 at complex128, with nothing said. The dtype *is* known at trace time even when
    the values are not, so a traced single-precision solve reports that it went unchecked. A
    traced double-precision one stays silent, which is what keeps the message meaningful.
    """
    jax = pytest.importorskip('jax')
    _legacy_wavebit_enable_x64()
    import jax.numpy as jnp

    theta = np.arcsin(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4, dtype=np.complex128))
    phi = np.arctan(1.2 / 1.6001, dtype=np.complex128)

    def amplitude(complex_dtype, t):
        res = _legacy_wavebit_rcwa(
            backend='jax', psi=np.pi / 2, top=1.0, bot=25.0, theta=t, phi=phi, truncation_order=[10],
            wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
            complex_dtype=complex_dtype, fourier_coefficients='sampled',
            factorization='li').solve()
        return jnp.asarray(res.s_inc.r_s).ravel()[10]

    def traced(complex_dtype):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            jax.jit(lambda t: amplitude(complex_dtype, t))(jnp.asarray(theta))
        return [str(r.message) for r in rec
                if issubclass(r.category, UnresolvedBranchWarning)]

    assert traced(0) == []
    single = traced(1)
    assert single and 'traced' in single[0]


def test_the_check_does_not_break_tracing():
    """Under jit/grad the value is abstract; the check must bail, not raise.

    This is why the reduction runs in the caller's namespace and only the scalar is read: an
    unreadable scalar means the value was never there to inspect. Uses the sampled path,
    the one that is traceable at all.
    """
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp

    def R(theta):
        res = _legacy_wavebit_rcwa(
            backend='jax', psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[6],
            wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
            complex_dtype=0, fourier_coefficients='sampled',
            factorization='li').solve()
        return jnp.sum(jnp.real(jnp.asarray(res.s_inc.R_s)))

    grad = jax.jit(jax.grad(R))(jnp.asarray(0.4))
    assert np.isfinite(float(grad))


def test_branch_warning_message_carries_no_numbers():
    a = str(_on_a_cut('numpy', 'half-space')[0].message)
    b = str(_on_a_cut('numpy', 'layer')[0].message)
    assert a == b


def test_branch_warning_category_is_exported():
    assert _legacy_wavebit_UnresolvedBranchWarning is UnresolvedBranchWarning


def _branch_error(algorithm, **kw):
    """complex64 against complex128 on one layer algorithm."""
    def r_s(complex_dtype):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = _legacy_wavebit_rcwa(
                backend='numpy', psi=np.pi / 2, bot=25.0, truncation_order=[30], wavelength=900e-9,
                period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
                complex_dtype=complex_dtype, fourier_coefficients='integrated',
                factorization='li', layer_algorithm=algorithm, **kw).solve()
        return np.asarray(res.s_inc.r_s, dtype=np.complex128).ravel()
    return float(np.abs(r_s(1) - r_s(0)).max())


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_single_precision_evanescent_incidence_needs_no_override(algorithm):
    """Neither the half-space root nor ETM's layer labels read round-off any more.

    complex64 rounds ``Re(theta) = pi/2`` up past it, and the engine clamps it just below,
    so the in-plane wavevector carries a genuine imaginary part of ~1e-7. Maystre's cut is
    continuous there. The rule before 2026-10-02 had its cut on the real evanescent axis, and
    there it turned half of the evanescent orders into growing ones: 8.2 against double
    precision. ETM was 5.1e-2 off until the same day, from layer labels read off the sign of
    a round-off ``Re(q)``; ``etm_mode_q`` reads ``Re(q) - Im(q)``. Now ETM 1.9e-6, SMM 3.4e-6.
    """
    assert _branch_error(algorithm, top=1.0, **_REGIMES['complex angle']) < 1e-5


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
def test_a_weak_absorber_at_a_complex_angle_holds_at_single_precision(algorithm):
    """The case that once made a branch cutoff a trade, with no cutoff left to trade.

    In 2026-08 a weakly absorbing superstrate had a genuine ``Re(kz)`` at the scale of complex64
    round-off, so the half-space rule's cutoff could serve it or the complex angle above, not
    both, and ``branch_tolerance`` was exposed to choose (0.99 off double precision at 1e-5
    here). Maystre's cut and ``etm_mode_q`` read no cutoff, and the override was removed on
    2026-10-02. ``theta = 0.6 + 1e-12j`` is the formal reading, ``k_par = n_top*sin(theta)``,
    which a real angle had in 2026-08. Measured now: ETM 3.1e-7, SMM 4.4e-7.
    """
    weak = dict(top=1.9599999998999997 + 2.8e-05j, theta=0.6 + 1e-12j, phi=0.4)
    assert _branch_error(algorithm, **weak) < 1e-5


# --- a value that arrived below the precision it was asked to be solved at --------------------

def _precision_warnings(theta, backend='numpy', complex_dtype=0):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _legacy_wavebit_rcwa(
            backend=backend, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[5],
            wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
            complex_dtype=complex_dtype, fourier_coefficients='sampled', factorization='li')
    return [r for r in rec if issubclass(r.category, PrecisionLossWarning)]


@pytest.mark.parametrize('theta, complex_dtype, expected', [
    (np.float64(0.4), 0, False),        # matches the solve
    (0.4, 0, False),                    # a Python float has no dtype to lose
    (np.float32(0.4), 1, False),        # single data, single solve — consistent
    (np.float32(0.4), 0, True),         # double was asked for; widening recovers nothing
    (np.complex64(0.4 + 0.3j), 0, True),
])
def test_an_input_below_the_solve_precision_is_reported(theta, complex_dtype, expected):
    assert bool(_precision_warnings(theta, complex_dtype=complex_dtype)) is expected


def _in_subprocess(body):
    """Run ``body`` in a fresh interpreter with ``JAX_*`` scrubbed.

    Import order is the whole subject here, so it cannot be tested in-process: by the time any
    other test has run, x64 is long since on. The environment is scrubbed because
    ``tests/rcwa/test_boundary.py`` sets ``JAX_ENABLE_X64=1`` at import and a child would inherit it
    — that difference passed alone and failed in the full suite until it was pinned down.
    """
    import os
    import subprocess
    import sys
    import textwrap
    env = {k: v for k, v in os.environ.items() if not k.startswith('JAX')}
    env['PYTHONPATH'] = os.pathsep.join((str(repository_root() / 'src'), str(repository_root())))
    # `capture_output` means `run` waits for EOF on the pipes, not merely for the
    # child to exit -- anything that inherits them holds the parent here forever.
    # A bound is cheap: the body is an import and an assert, about a second, and
    # 300 is chosen only to stay clear of a cold runner importing jax from disk.
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(body)],
                         capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_importing_wavebit_keeps_the_precision_policy_explicit():
    """Root import preserves the cap; configure JAX before constructing inputs."""
    pytest.importorskip('jax')
    assert 'OK' in _in_subprocess("""
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', False)
import wavebit
assert not jax.config.jax_enable_x64
assert jnp.asarray(0.4 + 0.3j).dtype == jnp.complex64
jax.config.update('jax_enable_x64', True)
assert jnp.asarray(0.4 + 0.3j).dtype == jnp.complex128, 'the cap was not lifted'
print('OK')
""")


def test_importing_wavebit_does_not_drag_jax_in():
    """The half of the old rationale that still holds: a NumPy-only caller pays nothing."""
    assert 'OK' in _in_subprocess("""
        import sys
        import wavebit
        assert 'jax' not in sys.modules, 'import wavebit pulled jax in'
        print('OK')
    """)


def test_the_remaining_x64_ordering_gap_is_reported():
    """JAX arriving *after* wavebit is the ordering the early lift cannot reach.

    Nothing has imported JAX yet when ``import wavebit`` runs, so there is no config to
    change; the backend lifts it later, at the first solve, and an array built
    before that is complex64. The setter widens it without recovering anything — measured
    ``|R| = 3.94`` against a true 0.67, with every dtype in sight reading complex128, which was
    read as a solver bug before it was read as this.
    """
    pytest.importorskip('jax')
    assert 'OK' in _in_subprocess("""
import warnings
from wavebit.rcwa._factory import rcwa as wavebit_rcwa
import jax.numpy as jnp
import numpy as np
from wavebit._common.backend import PrecisionLossWarning

theta = jnp.asarray(0.4 + 0.3j)
assert theta.dtype == jnp.complex64, theta.dtype        # the trap is armed
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    wavebit_rcwa(
        backend='jax', psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[5],
        wavelength=900e-9, period=[2000e-9],
        structure=np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1, thickness=[100e-9],
        complex_dtype=0, fourier_coefficients='sampled', factorization='li')
assert any(issubclass(r.category, PrecisionLossWarning) for r in rec), \\
    'the remaining ordering gap went unreported'
print('OK')
""")


def test_the_remedy_the_warning_names_actually_removes_it():
    """The message tells the caller to configure JAX x64 before arrays; follow it.

    The test above asserts the *category* of the warning and nothing about what it says to do,
    which is the half that can rot: the message used to say "Import wavebit.on_jax first" -- a
    statement that reads as a no-op, names a private path, and needed a comment every one of the
    33 places this suite used it. Nothing checked that following it worked, and nothing would
    have noticed when the module it named moved.

    Same scene as the gap test, one line earlier. The array must come back complex128 and the
    solve must warn about nothing.
    """
    pytest.importorskip('jax')
    assert 'OK' in _in_subprocess("""
import warnings
from wavebit._common.backend import _enable_x64 as wavebit_enable_x64
from wavebit.rcwa._factory import rcwa as wavebit_rcwa
import jax
jax.config.update('jax_enable_x64', True)                  # the remedy, before the array
import jax.numpy as jnp
import numpy as np
from wavebit._common.backend import PrecisionLossWarning

theta = jnp.asarray(0.4 + 0.3j)
assert theta.dtype == jnp.complex128, theta.dtype       # the trap is disarmed
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    wavebit_rcwa(
        backend='jax', psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[5],
        wavelength=900e-9, period=[2000e-9],
        structure=np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1, thickness=[100e-9],
        complex_dtype=0, fourier_coefficients='sampled', factorization='li')
assert not [r for r in rec if issubclass(r.category, PrecisionLossWarning)], \\
    'the documented remedy did not remove the warning it is documented against'
print('OK')
""")


def test_enable_x64_does_not_import_jax_to_find_out_whether_it_can():
    """It uses `importlib.util.find_spec`, and that matters for `import wavebit`.

    `src/wavebit/__init__.py` defines it, so anything the *body* imports at module scope would be
    pulled in by `import wavebit`, which
    `test_material.py::test_import_wavebit_does_not_import_jax` exists to forbid. The imports
    are inside the function, and `find_spec` answers "is JAX installed" without loading it.
    """
    assert 'OK' in _in_subprocess("""
import sys
from wavebit._common.backend import _enable_x64 as wavebit_enable_x64
assert 'jax' not in sys.modules, 'importing wavebit imported jax'
assert callable(wavebit_enable_x64)
print('OK')
""")


@pytest.fixture(scope='module')
def fresh_fdtd_jax_runs():
    """A NumPy reference and two JAX FDTD runs, all in one interpreter that starts clean.

    The FDTD path reaches JAX through its own door -- ``JaxTimeLoop.__init__``, not
    ``src/wavebit/rcwa/_backends/jax_backend.py`` -- and for a while that door did not lift the cap. The effect
    was that the *first* ``wavebit.fdtd(backend='jax', ...)`` of a process ran its Maxwell loop at
    complex64 and every later one ran at complex128, differing in the 5th significant figure,
    with no warning and with ``R.dtype`` reading float64 in both. (``fdtd_grid``, which never
    reached the result assembly that happened to import ``jax_backend``, ran at complex64 on
    *every* call; see the test after these two.) Hence the shape of this:

    * a **subprocess**, because by the time any other test has run, x64 is long since on and
      the first call is no longer first -- this is the same reason :func:`_in_subprocess`
      exists for the RCWA orderings above;
    * a **NumPy** run as the oracle, because NumPy has no cap to lift, so the double-precision
      answer needs no frozen number to compare against;
    * the loop **carry** dtype collected on the way past, because the reported result dtype is
      float64 either way and is exactly what made this invisible.

    One child process, three runs, read by the two tests below.
    """
    pytest.importorskip('jax')
    out = _in_subprocess("""
import json
import numpy as np
from wavebit.fdtd._periodic.api import fdtd as wavebit_fdtd
from wavebit.fdtd._periodic.runtime import JaxTimeLoop

# Small on purpose: seconds, and it still separated the two precisions by 9e-07.
scene = dict(structure=np.full((1, 1, 1), 2.25), thickness=[.2], top=1., bot=1.,
             period=.1, wavelength=[.9, 1., 1.1], resolution=8, n_periods=30,
             n_pml=4, truncation_order=0, tetm='te')

carried = []
advance = JaxTimeLoop.advance
def spy(self, step, state, steps, checkpoint_steps):
    import jax
    carried.extend(str(getattr(v, 'dtype', '')) for v in jax.tree.leaves(state))
    return advance(self, step, state, steps, checkpoint_steps)
JaxTimeLoop.advance = spy

def efficiencies(backend):
    given = wavebit_fdtd(**scene, backend=backend)[0].incident
    return [np.asarray(given.R).tolist(), np.asarray(given.T).tolist()]

reference = efficiencies('numpy')         # NumPy: no cap, so this is the double answer
first = efficiencies('jax')             # the call that used to run at complex64
second = efficiencies('jax')
print(json.dumps(dict(reference=reference, first=first, second=second,
                      carried=sorted(set(carried)))))
""")
    return json.loads(out.strip().splitlines()[-1])


def test_a_jax_fdtd_run_lifts_the_x64_cap_before_its_own_first_array(fresh_fdtd_jax_runs):
    """The first FDTD call of a fresh process must already be in double precision.

    No ``enable_x64()`` anywhere: the lift has to come from inside the call, which is what
    ``JaxTimeLoop.__init__`` now does -- and it is early enough because ``scene_fdtd`` takes
    ``runtime.xp`` and ``runtime.array`` from that constructor before it builds anything.
    Asserted on the value against the NumPy run, not on a dtype; ``R.dtype`` read float64
    throughout the defect.
    """
    carried = fresh_fdtd_jax_runs['carried']
    assert 'complex128' in carried and 'complex64' not in carried, carried
    np.testing.assert_allclose(fresh_fdtd_jax_runs['first'],
                               fresh_fdtd_jax_runs['reference'], atol=1e-12)


def test_two_identical_fdtd_calls_in_one_process_agree(fresh_fdtd_jax_runs):
    """Same inputs, same answer -- whatever the precision, it must not depend on call order.

    This is the invariant the defect broke and that nothing else pins: the two calls differed
    by 9e-07 because the first ran under the cap and the second did not. Exact equality is the
    right bar; the two runs execute identical code on identical inputs. That is a claim about
    the CPU this was measured on -- a device with nondeterministic reductions could need a
    tolerance, which would still have to sit far below the 9e-07 the defect produced.
    """
    np.testing.assert_array_equal(fresh_fdtd_jax_runs['first'], fresh_fdtd_jax_runs['second'])


def test_a_bare_fdtd_grid_run_is_double_precision_in_a_fresh_process():
    """``fdtd_grid`` goes through the same time-loop door and needs its own fresh process.

    Before the lift moved into ``JaxTimeLoop.__init__`` this door was worse off than ``fdtd``:
    it never reaches the scene's result assembly, which is where the ``fdtd`` path happened to
    import ``jax_backend`` and lift the cap, so a process that had not lifted it elsewhere ran
    ``fdtd_grid`` at complex64 on every call -- 5.5e-07 from NumPy, first and second call
    identical, so no call-order invariant can see it. Only a value against NumPy can, and only
    in a process where nothing else opened a door first. That also pins the lift to the shared
    constructor rather than to one of its callers.
    """
    pytest.importorskip('jax')
    out = _in_subprocess("""
import json
import numpy as np
from wavebit.fdtd._periodic.grid import fdtd_grid as wavebit_fdtd_grid

def run(backend):
    eps = np.ones((16, 3, 3)); eps[6:10] = 2.25
    src = np.zeros((3, 16, 3, 3), complex); src[0, 7] = 1
    return np.asarray(wavebit_fdtd_grid(
        eps, spacing=(1.,)*3, dt=.2, steps=60, pml_cells=3, source=src,
        pulse=np.sin(np.arange(60)*.2), wavelengths=[4., 5.], monitor_z=[9, 12],
        backend=backend)['electric_spectrum'])

reference = run('numpy')                  # NumPy: no cap, so this is the double answer
jax_run = run('jax')
print(json.dumps(float(np.max(abs(jax_run - reference))/np.max(abs(reference)))))
""")
    assert json.loads(out.strip().splitlines()[-1]) < 1e-12


def test_a_jax_rcwa_solve_is_double_precision_in_a_fresh_process():
    """The RCWA door, ``src/wavebit/rcwa/_backends/jax_backend.py``, pinned by value and not by the suite's habit.

    Nearly every JAX test here lifts the cap itself before it starts, so in-process nothing
    would notice that door's call going missing: measured with it deleted, a fresh-process
    solve differs from NumPy by 1.0e-06 and warns ``UnresolvedBranchWarning`` -- and the rest
    of this file stays green, because the gap test above asserts a warning category and the
    remedy test lifts the cap by hand first.
    """
    pytest.importorskip('jax')
    out = _in_subprocess("""
import json
import numpy as np
from wavebit.rcwa._factory import rcwa as wavebit_rcwa

kw = dict(psi=np.pi/2, top=1.0, bot=2.25, theta=0.3, phi=0.2, truncation_order=[5],
          wavelength=900e-9, period=[2000e-9],
          structure=np.array([[[0, 1, 0, 1, 1, 1]]])*1.25 + 1, thickness=[300e-9],
          fourier_coefficients='sampled', factorization='li')

def efficiencies(backend):
    given = wavebit_rcwa(backend=backend, **kw).solve().incident
    return np.r_[np.ravel(np.asarray(given.R)), np.ravel(np.asarray(given.T))]

reference = efficiencies('numpy')
print(json.dumps(float(np.max(abs(efficiencies('jax') - reference)))))
""")
    assert json.loads(out.strip().splitlines()[-1]) < 1e-12


def test_meshgrid_xy_on_jax_is_double_precision_in_a_fresh_process():
    """``meshgrid_xy(backend='jax')`` is a door onto JAX of its own, and asks for float64.

    Under the cap that request came back float32, and the coordinates -- and whatever raster
    was drawn from them -- were already truncated when the solver's own lift ran later. A
    period of 0.1 is not representable in float32, so the value comparison separates the two.
    """
    pytest.importorskip('jax')
    out = _in_subprocess("""
import json
import numpy as np
from wavebit.lattice import meshgrid_xy as wavebit_meshgrid_xy

on_jax = [np.asarray(c) for c in wavebit_meshgrid_xy((4, 5), period=(.1, .3), backend='jax')]
on_numpy = wavebit_meshgrid_xy((4, 5), period=(.1, .3), backend='numpy')
print(json.dumps(max(float(np.max(abs(a - b)/np.max(abs(b))))
                     for a, b in zip(on_jax, on_numpy))))
""")
    assert json.loads(out.strip().splitlines()[-1]) < 1e-14


def test_a_jax_door_raises_rather_than_run_under_a_cap_it_could_not_lift(monkeypatch):
    """Every door but ``import wavebit`` asks the shared lift to *require* success.

    ``enable_x64()`` promises to raise rather than quietly do nothing, and the time loop must
    not start at single precision because a config write was refused. The shared body swallows
    the failure for the one caller that may run with JAX absent or mid-import, so the other
    callers' loudness lives entirely in ``required=True`` -- and nothing else here makes the
    write fail.
    """
    jax = pytest.importorskip('jax')
    from wavebit._common.backend import lift_jax_x64
    from wavebit.fdtd._periodic.runtime import JaxTimeLoop

    previous = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', False)

        def refused(*args, **kwargs):
            raise ValueError('refused')
        monkeypatch.setattr(jax.config, 'update', refused)

        assert lift_jax_x64() is False                  # the tolerant caller: a no-op, said so
        with pytest.raises(RuntimeError, match='single precision'):
            _legacy_wavebit_enable_x64()
        with pytest.raises(RuntimeError, match='single precision'):
            JaxTimeLoop()
        with pytest.raises(RuntimeError, match='single precision'):
            _legacy_wavebit_meshgrid_xy((2, 2), period=(1., 1.), backend='jax')
    finally:
        monkeypatch.undo()
        jax.config.update('jax_enable_x64', previous)


def test_the_x64_cap_is_lifted_by_one_body():
    """Exactly one place in the package writes ``jax_enable_x64``: the shared lift.

    The FDTD defect was a door with no call, and the copies it replaced had already drifted
    apart (two swallowed failures, one raised). A second write would restart both. This checks executable
    call syntax, excluding examples inside strings: it cannot say every door *calls* the body -- the fresh-process tests
    above do that, door by door -- only that nothing bypasses it.
    """
    import ast
    package = repository_path('src/wavebit')
    sites = []
    for path in sorted(package.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 'jax_enable_x64'):
                continue
            owner = node.func.value
            if ((isinstance(owner, ast.Attribute) and owner.attr == 'config')
                    or (isinstance(owner, ast.Name) and owner.id == 'config')):
                sites.append(f'{path.relative_to(package.parent.parent).as_posix()}:{node.lineno}')
    assert [s.split(':')[0] for s in sites] == ['src/wavebit/_common/backend.py'], sites


def test_precision_warning_category_is_exported():
    assert _legacy_wavebit_PrecisionLossWarning is PrecisionLossWarning
