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
* contamination is loud, in two tiers: a non-finite efficiency raises, and a finite but
  physically impossible one warns. Both point at the structure rather than at the arithmetic.
"""
import pathlib
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core.backend import (PrecisionLossWarning, UnphysicalResultWarning,
                                     UnresolvedBranchWarning,
                                     array_namespace, check_finite,
                                     check_physical)
from wavebit._core.solve_1d import final_matching

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
        res = wavebit.rcwa(backend=0, wavelength=WL, **kwargs).solve()
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
        res = wavebit.rcwa(backend=0, structure=structure, truncation_order=(6, 6), top=1.0, bot=1.0,
                             theta=np.deg2rad(5), phi=np.deg2rad(20), period=(1.0, 1.2),
                             wavelength=1.5, thickness=(0.3,), psi=0.35 * np.pi).solve()
    assert _fp_warnings(rec) == []
    assert abs(_energy(res) - 1.0) < 1e-10


def test_field_reconstruction_is_silent_and_finite():
    """Reconstruction re-walks the same modal exponentials the solve cascaded."""
    sim = wavebit.rcwa(backend=0, structure=_structure_2d(), thickness=[600], period=[700, 700],
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


# --- tier 2: finite, but not a physical result ----------------------------------------------

def _physical(R, T, n_top=1.0):
    """Run the plausibility check, returning the warnings it emitted.

    ``n_top`` and not ``top``: :func:`check_physical` is a core function, and there the
    quantity genuinely is a refractive index -- it normalises the incident power. Only the
    public boundary flipped to permittivity.
    """
    xp = array_namespace(np.zeros(1))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        check_physical(xp, np.asarray(R), np.asarray(T), n_top)
    return [r for r in rec if issubclass(r.category, UnphysicalResultWarning)]


def test_physical_check_passes_a_lossless_result():
    assert _physical([0.4, 0.1], [0.5]) == []


def test_physical_check_allows_absorption():
    """``R + T < 1`` is just a lossy stack — never a complaint."""
    assert _physical([0.1, 0.05], [0.2]) == []


def test_physical_check_warns_on_negative_efficiency():
    w = _physical([0.6, -0.2], [0.5])
    assert len(w) == 1
    assert 'negative' in str(w[0].message)


def test_physical_check_warns_on_gain_from_a_passive_stack():
    w = _physical([0.9, 0.4], [0.5])
    assert len(w) == 1
    assert 'above 1' in str(w[0].message)


def test_physical_check_reports_both_problems_at_once():
    w = _physical([1.6, -0.2], [0.5])
    assert len(w) == 1
    assert 'negative' in str(w[0].message) and 'above 1' in str(w[0].message)


def test_physical_check_tolerates_roundoff():
    """A lossless solve lands on 1 to ~1e-14; that must never warn."""
    assert _physical([0.5 + 1e-14], [0.5]) == []


def test_physical_check_needs_net_entering_power_for_an_absorbing_superstrate():
    """Without interference, R and T alone cannot determine the absorption balance."""
    assert _physical([0.9, 0.4], [0.5], n_top=1.0 + 0.3j) == []


def test_physical_check_uses_net_entering_power_for_an_absorbing_superstrate():
    xp = array_namespace(np.zeros(1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        check_physical(xp, np.array([.054]), np.array([.984]), 1.5 + .3j,
                       power_entering=np.array(.984))
    assert not caught  # R+T > 1 does not imply gain at this absorbing interface.
    with pytest.warns(UnphysicalResultWarning, match='net entering'):
        check_physical(xp, np.array([.054]), np.array([.99]), 1.5 + .3j,
                       power_entering=np.array(.984))


def test_physical_warning_message_carries_no_numbers():
    """Constant text is what lets the default filter collapse a sweep into one report."""
    a = str(_physical([0.9, 0.4], [0.5])[0].message)
    b = str(_physical([2.7, 0.4], [0.9])[0].message)
    assert a == b


def test_physical_warning_can_be_escalated_to_an_error():
    xp = array_namespace(np.zeros(1))
    with warnings.catch_warnings():
        warnings.simplefilter('error', UnphysicalResultWarning)
        with pytest.raises(UnphysicalResultWarning):
            check_physical(xp, np.array([0.9, 0.4]), np.array([0.5]), 1.0)


def test_warning_category_is_exported():
    """Filtering it is a user-facing action, so the name has to be reachable."""
    assert wavebit.UnphysicalResultWarning is UnphysicalResultWarning


def test_healthy_solves_emit_no_unphysical_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        wavebit.rcwa(backend=0, structure=_structure_2d(), thickness=[600], period=[700, 700],
                       wavelength=WL, truncation_order=[8, 8], psi=0.0, bot=EPS_SI).solve()
    assert [r for r in rec if issubclass(r.category, UnphysicalResultWarning)] == []


# --- the line is concreteness, not the namespace ---------------------------------------------
#
# These two used to assert the opposite: that both guards were no-ops on anything that was not
# eager NumPy. That was keyed on the namespace, and it meant an ordinary eager `backend=1` /
# `backend=2` call — where the value is right there to be read — went unguarded. A
# near-singular stack then returned R = 2.47 (JAX) and 243.9 (PyTorch) against a truth of
# 0.215, finite and with an empty warning list. tests/test_rayleigh_guard.py pins that case
# end to end; these pin the adapter-level rule it rests on.

def test_guards_read_eager_jax_values():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    xp = array_namespace(jnp.zeros(1))
    with pytest.raises(FloatingPointError):
        check_finite(xp, R=jnp.asarray([np.nan]))
    with pytest.warns(UnphysicalResultWarning):
        check_physical(xp, jnp.asarray([9.0]), jnp.asarray([9.0]), 1.0)


def test_guards_read_eager_torch_values():
    torch = pytest.importorskip('torch')
    xp = array_namespace(torch.zeros(1))
    with pytest.raises(FloatingPointError):
        check_finite(xp, R=torch.tensor([float('nan')]))
    with pytest.warns(UnphysicalResultWarning):
        check_physical(xp, torch.tensor([9.0]), torch.tensor([9.0]), 1.0)


def test_guards_step_aside_for_a_jit_tracer():
    """The reason the early return cannot simply be deleted.

    Under ``jit`` the value is abstract: reading it raises, and forcing it concrete would
    break the trace. So the guards must notice they cannot read and say nothing — which is
    also why a traced non-finite result still returns quietly. Accepted, and documented on
    :func:`wavebit._core.backend._concrete`.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    def f(x):
        xp = array_namespace(x)
        check_finite(xp, R=x)
        check_physical(xp, x, x, 1.0)
        return jnp.sum(x)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = jax.jit(f)(jnp.asarray([np.nan, 9.0]))
    assert np.isnan(float(out))
    assert [r for r in rec if issubclass(r.category, UnphysicalResultWarning)] == []


# --- tier three: a branch decided from round-off ---------------------------------------------
#
# check_finite catches contamination that announces itself, check_physical catches a finite
# result that cannot be physical. Both read the *result*. This tier reads the arithmetic: a
# sqrt branch is a discrete choice, and at complex64 the quantity it is taken from is often
# pure round-off. The answer stays finite and stays physical (the degenerate case is perfectly
# unitary), so neither tier above it fires. See docs/branch-cuts.md.

#: A 1/5 index grating, written as its permittivities: 1 and 25.
_BRANCH_STRUCTURE = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1


def _branch_solve(complex_dtype, top=1.0, theta=0.4, phi=0.3, backend=0):
    return wavebit.rcwa(
        backend=backend, psi=np.pi / 2, top=top, bot=25.0, theta=theta, phi=phi, truncation_order=[30],
        wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
        complex_dtype=complex_dtype, fourier_coefficients='integrated',
        factorization='li').solve()


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
@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_double_precision_resolves_every_branch(backend, regime):
    """Silence at complex128 is the claim that makes the warning worth having."""
    assert _branch_warnings(complex_dtype=0, backend=backend, **_REGIMES[regime]) == []


@pytest.mark.parametrize('regime', list(_REGIMES))
@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_single_precision_reads_no_branch_from_round_off(backend, regime):
    """complex64 now answers every branch question these regimes ask, and says nothing.

    Every decision on this ETM solve reads a value at least ``|v|`` from zero: the half-space's
    ``Re(kz) + Im(kz)`` (Maystre's cut) and the layer label's ``Re(q) - Im(q)``
    (``etm_mode_q``), both since 2026-10-02. Neither is round-off, so the warning stays quiet and
    the answer holds -- within 3e-6 of double precision on every backend here. In 2026-08 the
    same point was 1.80 off on NumPy, 4.12 on JAX and 5.68 on PyTorch, and ETM was still 5e-2
    off until its labels moved.

    Every backend, deliberately: unlike the guards in ``_core/backend.py`` this check is not
    NumPy-only, and the backends it would have skipped are the paths taken for gradients and
    GPUs. Where a decision genuinely sits on a cut it still fires; see the test below.
    """
    kw = _REGIMES[regime]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        single = _branch_solve(1, backend=backend, **kw).s_inc.r_s
    assert [r for r in rec if issubclass(r.category, UnresolvedBranchWarning)] == []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        double = _branch_solve(0, backend=backend, **kw).s_inc.r_s
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
    from wavebit._core.conventions import boundary_kz, etm_mode_q
    if backend == 0:
        make = lambda v: np.asarray(v, dtype=np.complex64 if single else np.complex128)
    elif backend == 1:
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
@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
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
    wavebit.enable_x64()
    import jax.numpy as jnp

    theta = np.arcsin(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4, dtype=np.complex128))
    phi = np.arctan(1.2 / 1.6001, dtype=np.complex128)

    def amplitude(complex_dtype, t):
        res = wavebit.rcwa(
            backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=t, phi=phi, truncation_order=[10],
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
        res = wavebit.rcwa(
            backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[6],
            wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
            complex_dtype=0, fourier_coefficients='sampled',
            factorization='li').solve()
        return jnp.sum(jnp.real(jnp.asarray(res.s_inc.R_s)))

    grad = jax.jit(jax.grad(R))(jnp.asarray(0.4))
    assert np.isfinite(float(grad))


def test_branch_warning_message_carries_no_numbers():
    a = str(_on_a_cut(0, 'half-space')[0].message)
    b = str(_on_a_cut(0, 'layer')[0].message)
    assert a == b


def test_branch_warning_category_is_exported():
    assert wavebit.UnresolvedBranchWarning is UnresolvedBranchWarning


def test_branch_tolerance_is_scoped_and_restores():
    import wavebit._core.conventions as conventions
    before = conventions._FLAT
    with wavebit.branch_tolerance(1e-6):
        assert conventions._FLAT == 1e-6
    assert conventions._FLAT == before


def _branch_error(algorithm, cutoff=1e-12, **kw):
    """complex64 against complex128 on one layer algorithm, at one branch cutoff."""
    def r_s(complex_dtype):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = wavebit.rcwa(
                backend=0, psi=np.pi / 2, bot=25.0, truncation_order=[30], wavelength=900e-9,
                period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
                complex_dtype=complex_dtype, fourier_coefficients='integrated',
                factorization='li', layer_algorithm=algorithm, **kw).solve()
        return np.asarray(res.s_inc.r_s, dtype=np.complex128).ravel()
    ref = r_s(0)
    with wavebit.branch_tolerance(cutoff):
        return float(np.abs(r_s(1) - ref).max())


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


def test_raising_the_cutoff_no_longer_costs_a_weak_absorber():
    """The trade that made ``branch_tolerance`` a scoped override is gone.

    In 2026-08 a raised cutoff rescued the complex angle and broke a weakly absorbing
    superstrate, because the half-space rule then decided on the same cutoff. That was 0.99 off
    double precision at 1e-5 here. ``theta = 0.6 + 1e-12j`` is the formal reading,
    ``k_par = n_top*sin(theta)``, which a real angle had in 2026-08. Measured now: 3.8e-7.
    """
    weak = dict(top=1.9599999998999997 + 2.8e-05j, theta=0.6 + 1e-12j, phi=0.4)
    assert _branch_error('SMM', 1e-5, **weak) < 1e-5


# --- tier four: a value that arrived below the precision it was asked to be solved at --------

def _precision_warnings(theta, backend=0, complex_dtype=0):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        wavebit.rcwa(
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
    ``tests/test_boundary.py`` sets ``JAX_ENABLE_X64=1`` at import and a child would inherit it
    — that difference passed alone and failed in the full suite until it was pinned down.
    """
    import os
    import subprocess
    import sys
    import textwrap
    env = {k: v for k, v in os.environ.items() if not k.startswith('JAX')}
    env['PYTHONPATH'] = str(pathlib.Path(__file__).resolve().parent.parent)
    # `capture_output` means `run` waits for EOF on the pipes, not merely for the
    # child to exit -- anything that inherits them holds the parent here forever.
    # A bound is cheap: the body is an import and an assert, about a second, and
    # 300 is chosen only to stay clear of a cold runner importing jax from disk.
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(body)],
                         capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_importing_wavebit_lifts_the_x64_cap_when_jax_is_already_loaded():
    """The half of the trap that can be closed rather than merely reported.

    With x64 off JAX cannot be *asked* for float64 — it is a cap, not a default — and wavebit
    flips it either way, just later. Deferring therefore does not spare the caller the
    process-wide change; it only makes the moment depend on import order, and an array built in
    the gap is silently complex64. So it also happens at ``import wavebit``, when JAX is
    already present.
    """
    pytest.importorskip('jax')
    assert 'OK' in _in_subprocess("""
        import jax.numpy as jnp
        import wavebit
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
        import wavebit
        import jax.numpy as jnp
        import numpy as np
        from wavebit._core.backend import PrecisionLossWarning

        theta = jnp.asarray(0.4 + 0.3j)
        assert theta.dtype == jnp.complex64, theta.dtype        # the trap is armed
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            wavebit.rcwa(
                backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[5],
                wavelength=900e-9, period=[2000e-9],
                structure=np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1, thickness=[100e-9],
                complex_dtype=0, fourier_coefficients='sampled', factorization='li')
        assert any(issubclass(r.category, PrecisionLossWarning) for r in rec), \\
            'the remaining ordering gap went unreported'
        print('OK')
    """)


def test_the_remedy_the_warning_names_actually_removes_it():
    """The message tells the caller to run `wavebit.enable_x64()`. This is that, run.

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
        import wavebit
        wavebit.enable_x64()                                    # the remedy, before the array
        import jax.numpy as jnp
        import numpy as np
        from wavebit._core.backend import PrecisionLossWarning

        theta = jnp.asarray(0.4 + 0.3j)
        assert theta.dtype == jnp.complex128, theta.dtype       # the trap is disarmed
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            wavebit.rcwa(
                backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, truncation_order=[5],
                wavelength=900e-9, period=[2000e-9],
                structure=np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1, thickness=[100e-9],
                complex_dtype=0, fourier_coefficients='sampled', factorization='li')
        assert not [r for r in rec if issubclass(r.category, PrecisionLossWarning)], \\
            'the documented remedy did not remove the warning it is documented against'
        print('OK')
    """)


def test_enable_x64_does_not_import_jax_to_find_out_whether_it_can():
    """It uses `importlib.util.find_spec`, and that matters for `import wavebit`.

    `wavebit/__init__.py` defines it, so anything the *body* imports at module scope would be
    pulled in by `import wavebit`, which
    `test_material.py::test_import_wavebit_does_not_import_jax` exists to forbid. The imports
    are inside the function, and `find_spec` answers "is JAX installed" without loading it.
    """
    assert 'OK' in _in_subprocess("""
        import sys
        import wavebit
        assert 'jax' not in sys.modules, 'importing wavebit imported jax'
        assert callable(wavebit.enable_x64)
        print('OK')
    """)

def test_precision_warning_category_is_exported():
    assert wavebit.PrecisionLossWarning is PrecisionLossWarning
