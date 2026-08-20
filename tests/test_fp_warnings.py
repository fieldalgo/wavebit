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
  factorizations, the analytic builder, field reconstruction — which is what the version floor
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
    return float(np.sum(res.given.R) + np.sum(res.given.T))


# --- the structures: thick, high-truncation, deeply evanescent -----------------------------
# A 600nm rod at 500nm wavelength with max_order=8 retains orders whose decay over one layer is
# ~exp(-100): the regime that made the broken NumPy shout loudest.

#: 3.48**2 and 2.0**2 -- the same two media, as the permittivity `structure` now holds.
EPS_SI, EPS_MID = 12.1104, 4.0


def _structure_2d():
    u = np.ones((1, 64, 64))
    u[0, 16:48, 16:48] = EPS_SI
    return u


def _structure_1d():
    u = np.ones((2, 1, 256))
    u[0, 0, 64:192] = EPS_SI
    u[1, 0, :] = EPS_MID
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
    res, fp = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], max_order=[8, 8],
                   psi=0.0, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('psi', [np.pi / 2, 0.0], ids=['te', 'tm'])
def test_thick_1d_stack_is_silent_and_conserves_energy(layer_algorithm, psi):
    res, fp = _run(structure=_structure_1d(), thickness=[800, 400], period=[700], max_order=[8, 0],
                   psi=psi, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
def test_conical_stack_is_silent_and_conserves_energy(layer_algorithm):
    res, fp = _run(structure=_structure_1d(), thickness=[800, 400], period=[700], max_order=[8, 0],
                   psi=np.pi / 2, phi=0.3, bot=EPS_SI, layer_algorithm=layer_algorithm)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


@pytest.mark.parametrize('strategy',
                         [dict(fourier_coefficients=c, factorization=f)
                          for c in ('sampled', 'integrated') for f in ('li', 'nvf')],
                         ids=['sampled-li', 'sampled-nvf', 'integrated-li', 'integrated-nvf'])
def test_every_factorization_is_silent(strategy):
    """The convolution builders were the loudest sites of all under the broken NumPy."""
    res, fp = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], max_order=[6, 6],
                   psi=0.0, bot=EPS_SI, **strategy)
    assert fp == []
    assert abs(_energy(res) - 1.0) < 1e-10


def test_analytic_shape_path_is_silent():
    """The analytic (staircase-free) builder carries its own copy of the NVF assembly."""
    structure = [dict(background=1.0, shapes=[dict(type='ellipse', cx=0.5, cy=0.6, lx=0.5,
                                              ly=0.3, angle=np.deg2rad(40), material=5.76),
                                      dict(type='circle', cx=0.3, cy=0.3, lx=0.2,
                                              material=9.0)])]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = wavebit.rcwa(backend=0, structure=structure, max_order=(6, 6), top=1.0, bot=1.0,
                             theta=np.deg2rad(5), phi=np.deg2rad(20), period=(1.0, 1.2),
                             wavelength=1.5, thickness=(0.3,), psi=0.35 * np.pi).solve()
    assert _fp_warnings(rec) == []
    assert abs(_energy(res) - 1.0) < 1e-10


def test_field_reconstruction_is_silent_and_finite():
    """Reconstruction re-walks the same modal exponentials the solve cascaded."""
    sim = wavebit.rcwa(backend=0, structure=_structure_2d(), thickness=[600], period=[700, 700],
                         wavelength=WL, max_order=[6, 6], psi=0.0, bot=EPS_SI, answers='field')
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
        _run(structure=_structure_2d(), thickness=[600], period=[700, 700], max_order=[6, 6],
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
        res, _ = _run(structure=_structure_2d(), thickness=[600], period=[700, 700], max_order=[8, 8],
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
    assert 'max_order' in msg and 'degenerate' in msg       # says what to do about it


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
                       n_orders_x=1, n_orders_y=1)


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


def test_physical_check_skips_an_absorbing_superstrate():
    """With complex n_top the incident-power normalization makes the comparison meaningless."""
    assert _physical([0.9, 0.4], [0.5], n_top=1.0 + 0.3j) == []


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
                       wavelength=WL, max_order=[8, 8], psi=0.0, bot=EPS_SI).solve()
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
        backend=backend, psi=np.pi / 2, top=top, bot=25.0, theta=theta, phi=phi, max_order=[30],
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
    # Its genuine Re(kz) sits at the same scale as the round-off above; that overlap is why
    # no fixed cutoff serves both, and why this warns rather than being repaired.
    # (1.4 + 1e-5j)**2 -- the weak absorber, as permittivity.
    'weak absorber': dict(top=1.9599999998999997 + 2.8e-05j, theta=0.6, phi=0.4),
}


@pytest.mark.parametrize('regime', list(_REGIMES))
@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_double_precision_resolves_every_branch(backend, regime):
    """Silence at complex128 is the claim that makes the warning worth having."""
    assert _branch_warnings(complex_dtype=0, backend=backend, **_REGIMES[regime]) == []


@pytest.mark.parametrize('regime', list(_REGIMES))
@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_single_precision_reports_an_unresolvable_branch(backend, regime):
    """complex64 cannot answer the question, and says so instead of guessing quietly.

    Every backend, deliberately: unlike the guards in ``_core/backend.py`` this one is not
    NumPy-only, because the backends it would have skipped return the *worst* answers here —
    NumPy 1.80, JAX 4.12, PyTorch 5.68 at the same point — and they are the paths taken for
    gradients and GPUs.
    """
    assert _branch_warnings(complex_dtype=1, backend=backend, **_REGIMES[regime])


def test_a_traced_single_precision_solve_says_it_could_not_be_checked():
    """Silence must not mean two different things.

    Under ``jit`` the values are tracers, so the check cannot run — and that is exactly where a
    wrong branch is least likely to be noticed: measured ``|R| = 3.93`` traced at complex64
    against 0.67 at complex128, with nothing said. The dtype *is* known at trace time even when
    the values are not, so a traced single-precision solve reports that it went unchecked. A
    traced double-precision one stays silent, which is what keeps the message meaningful.
    """
    jax = pytest.importorskip('jax')
    import wavebit.on_jax          # noqa: F401  -- enables x64 before any array is made
    import jax.numpy as jnp

    theta = np.arcsin(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4, dtype=np.complex128))
    phi = np.arctan(1.2 / 1.6001, dtype=np.complex128)

    def amplitude(complex_dtype, t):
        res = wavebit.rcwa(
            backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=t, phi=phi, max_order=[10],
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
            backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, max_order=[6],
            wavelength=900e-9, period=[2000e-9], structure=_BRANCH_STRUCTURE, thickness=[100e-9],
            complex_dtype=0, fourier_coefficients='sampled',
            factorization='li').solve()
        return jnp.sum(jnp.real(jnp.asarray(res.s_inc.R_s)))

    grad = jax.jit(jax.grad(R))(jnp.asarray(0.4))
    assert np.isfinite(float(grad))


def test_branch_warning_message_carries_no_numbers():
    a = str(_branch_warnings(complex_dtype=1)[0].message)
    b = str(_branch_warnings(complex_dtype=1, **_REGIMES['complex angle'])[0].message)
    assert a == b


def test_branch_warning_category_is_exported():
    assert wavebit.UnresolvedBranchWarning is UnresolvedBranchWarning


def test_branch_tolerance_is_scoped_and_restores():
    import wavebit._core.conventions as conventions
    before = conventions._FLAT
    with wavebit.branch_tolerance(1e-6):
        assert conventions._FLAT == 1e-6
    assert conventions._FLAT == before


def test_branch_tolerance_rescues_a_complex_angle_at_single_precision():
    """The escape hatch has to actually move the answer, or it is decoration.

    Double precision is the reference; at complex64 the default cutoff is 4.5 off it, and
    raising the cutoff brings it to ~1e-6. The same setting is *wrong* for a weak absorber —
    that trade is the documented reason this is a scoped override and not a new default.
    """
    ref = np.asarray(_branch_solve(0, **_REGIMES['complex angle']).s_inc.r_s).ravel()

    def err():
        got = np.asarray(_branch_solve(1, **_REGIMES['complex angle']).s_inc.r_s).ravel()
        return float(np.abs(got - ref).max())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert err() > 1.0
        with wavebit.branch_tolerance(1e-5):
            assert err() < 1e-5


# --- tier four: a value that arrived below the precision it was asked to be solved at --------

def _precision_warnings(theta, backend=0, complex_dtype=0):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        wavebit.rcwa(
            backend=backend, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, max_order=[5],
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
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(body)],
                         capture_output=True, text=True, env=env)
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
    change; ``wavebit.on_jax`` enables x64 later, at the first solve, and an array built
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
                backend=1, psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3, max_order=[5],
                wavelength=900e-9, period=[2000e-9],
                structure=np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1, thickness=[100e-9],
                complex_dtype=0, fourier_coefficients='sampled', factorization='li')
        assert any(issubclass(r.category, PrecisionLossWarning) for r in rec), \\
            'the remaining ordering gap went unreported'
        print('OK')
    """)

def test_precision_warning_category_is_exported():
    assert wavebit.PrecisionLossWarning is PrecisionLossWarning
