"""ETM's layer labels at a complex in-plane wavevector: the half-space's root, capped.

A layer's ``±q`` pair is used in full by the stack algebra, so which root is labelled "down" is
free in exact arithmetic, and the SMM connecting algorithm keeps it free in practice. ETM does
not. Its interface factor ``a = (W^-1 F + V^-1 G)/2`` is the inverse of the transmission from a
layer's ``+`` modes into the medium below, so it is well conditioned only when those modes are
the waves that medium receives: :func:`~wavebit._core.conventions.boundary_kz`'s root.

:func:`~wavebit._core.conventions.mode_q`'s decaying root is that wave at every real ``k_par`` in
a passive layer. At a complex ``k_par`` it is not: a propagating mode whose ``q**2`` lies above
the real axis decays *upward*. The scene below rests an ``eps = 25`` block on an ``eps = 25``
substrate, so such a mode crosses into the substrate with nothing to reflect it. The smallest
singular value of ``a`` is then that reflection, ~1e-14, and the recursion lost ~12 digits. A
1e-14 nudge of ``theta = 0.5+0.3j`` moved ETM's ``r0`` by 2e-4 where SMM's moved by 2e-15, and
the two algorithms disagreed by 5e-5; the 2D path lost the same way (5e-6).

:func:`~wavebit._core.conventions.etm_mode_q` takes the half-space's root instead, and falls back
to the decaying one where that root would grow across the layer by more than ``eps**-0.25``.
Each half is load-bearing, and the tests below carry the controls that show it:

* on the decaying root everywhere (the rule before this file) every case fails;
* on the half-space root everywhere, uncapped, the 50 um layer at ``theta = 0.5+0.3j`` fails:
  near-cutoff modes grew by up to ``e**167`` across it and put ETM 2.7 off SMM.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core import conventions
from wavebit._core.conventions import boundary_kz, etm_mode_q, mode_q

#: The 1/5 index grating of ``tests/test_fp_warnings.py``, as permittivities: 1 and 25. Its
#: wide ``eps = 25`` block rests on the ``eps = 25`` substrate below, which is what lets a
#: mislabelled mode pass the interface unreflected.
STRUCTURE = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1

NUDGE = 1 + 1e-14


def _r0(theta, algorithm, thickness=100e-9, phi=0.3):
    """The s-polarised zeroth-order reflection amplitude of the scene above."""
    solver = wavebit.rcwa(backend='numpy', psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=phi,
                          truncation_order=[30], wavelength=900e-9, period=[2000e-9],
                          structure=STRUCTURE, thickness=[thickness], complex_dtype=0,
                          fourier_coefficients='integrated', factorization='li',
                          layer_algorithm=algorithm)
    with warnings.catch_warnings():
        # R and T are formal ratios at a complex angle, which is the warning's whole content.
        warnings.simplefilter('ignore', wavebit.ComplexIncidenceWarning)
        res = solver.solve()
    amp = np.asarray(res.s_inc.r_s).ravel()
    return complex(amp[len(amp) // 2])


def _moves(theta, algorithm, **kw):
    """``(r0, |r0(theta*NUDGE) - r0(theta)|)``: the answer, and how far rounding alone moves it."""
    r = _r0(theta, algorithm, **kw)
    return r, abs(_r0(theta * NUDGE, algorithm, **kw) - r)


def _decaying(xp, eigenvalue, k0_d):
    """The rule before this file: the decaying root for ETM too."""
    return mode_q(xp, eigenvalue)


def _uncapped(xp, eigenvalue, k0_d):
    """The half-space's root everywhere, however much it grows across the layer."""
    v = xp.sqrt(eigenvalue)
    return xp.where(xp.real(v) - xp.imag(v) < 0, -v, v)


def _with_etm_rule(rule, fn, *args, **kw):
    """Run ``fn`` with :func:`etm_mode_q` swapped for ``rule`` (the control)."""
    original = conventions.etm_mode_q
    conventions.etm_mode_q = rule
    try:
        return fn(*args, **kw)
    finally:
        conventions.etm_mode_q = original


# ------------------------------------------------------------------- the reported symptom

THETAS = [0.5 + 0.3j, 0.5 + 0.03j, 1.2 + 0.5j]


@pytest.mark.parametrize('theta', THETAS, ids=['0.5+0.3j', '0.5+0.03j', '1.2+0.5j'])
@pytest.mark.parametrize('thickness', [100e-9, 50e-6], ids=['100nm', '50um'])
def test_a_nudged_complex_angle_moves_etm_no_more_than_smm(theta, thickness):
    """A 1e-14 nudge of a complex ``theta`` moves ETM's ``r0`` no more than SMM's.

    Both moves are rounding: 1e-15 to 1e-12 here, growing with the thickness on either
    algorithm alike. At that size their order is noise, so ETM is allowed twice SMM's move plus
    1e-14 -- nine decades below what ETM moved before (2e-4 at 100 nm). The two answers are
    also held to each other, since a floor need not show in a nudge to show in the value.

    The control puts the decaying root back. It has to break the agreement, or this scene no
    longer reaches what it is here to test: it left ETM 5e-6 to 2e-4 off SMM.
    """
    smm, smm_move = _moves(theta, 'SMM', thickness=thickness)
    etm, etm_move = _moves(theta, 'ETM', thickness=thickness)
    assert etm_move <= 2 * smm_move + 1e-14, (etm_move, smm_move)
    assert abs(etm - smm) < 1e-12, abs(etm - smm)

    broken = _with_etm_rule(_decaying, _r0, theta, 'ETM', thickness=thickness)
    assert abs(broken - smm) > 1e-7, abs(broken - smm)


def test_the_cap_is_what_keeps_a_thick_layer():
    """Uncapped, the half-space's root broke a 50 um layer; capped, ETM matches SMM.

    At ``theta = 0.5+0.3j`` the modes near cutoff have half-space roots that grow by up to
    ``e**167`` across 50 um. Taken anyway they put ETM 2.7 off SMM, and broke SMM too when it
    was handed the same labels. Past ``eps**-0.25`` the decaying root is kept, and the
    agreement above holds at 50 um.
    """
    theta, thickness = 0.5 + 0.3j, 50e-6
    smm = _r0(theta, 'SMM', thickness=thickness)
    assert abs(_r0(theta, 'ETM', thickness=thickness) - smm) < 1e-12
    broken = _with_etm_rule(_uncapped, _r0, theta, 'ETM', thickness=thickness)
    assert abs(broken - smm) > 1e-2, abs(broken - smm)


#: ``Re(theta)`` offsets below pi/2 at the evanescent point of ``tests/test_fp_warnings.py``.
#: 7.5e-8 is what the engine's float32 clamp (``_incident_kxy``) produces there, which makes
#: ``Im(k_par)`` ~1e-7: the decaying root then took its label from the sign of a ``Re(q)`` that
#: small, and raising the branch cutoff to 1e-5 (``branch_tolerance``, removed the same day as
#: this rule landed) sent those modes to the tie-break. At 1e-4 and 1e-2 that real part is
#: larger and no cutoff helped.
EVANESCENT_OFFSETS = [7.5e-8, 1e-4, 1e-2]


@pytest.mark.parametrize('offset', EVANESCENT_OFFSETS, ids=['f32_clamp', '1e-4', '1e-2'])
def test_evanescent_incidence_needs_no_override(offset):
    """ETM matches SMM at evanescent incidence with no override of any branch cutoff.

    The decaying root left ETM 1.0e-5, 3.6e-5 and 9.8e-6 off SMM at these three offsets.
    """
    alpha = np.arccosh(np.sqrt(1.6 ** 2 + 1.2 ** 2 + 1e-4))
    theta = (np.pi / 2 - offset) + 1j * alpha
    phi = np.arctan(1.2 / 1.6001)
    smm, smm_move = _moves(theta, 'SMM', phi=phi)
    etm, etm_move = _moves(theta, 'ETM', phi=phi)
    assert etm_move <= 2 * smm_move + 1e-14, (etm_move, smm_move)
    assert abs(etm - smm) < 1e-12, abs(etm - smm)

    broken = _with_etm_rule(_decaying, _r0, theta, 'ETM', phi=phi)
    assert abs(broken - smm) > 1e-7, abs(broken - smm)


def _r0_tetm(theta, algorithm, tetm):
    """The same grating on the pure TE / TM path (``phi = 0``), at its own amplitude."""
    solver = wavebit.rcwa(backend='numpy', tetm=tetm, top=1.0, bot=25.0, theta=theta,
                          truncation_order=[30], wavelength=900e-9, period=[2000e-9],
                          structure=STRUCTURE, thickness=[100e-9], complex_dtype=0,
                          fourier_coefficients='integrated', factorization='li',
                          layer_algorithm=algorithm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavebit.ComplexIncidenceWarning)
        res = solver.solve().incident
    amp = np.asarray(res.r_s if tetm == 'te' else res.r_p).ravel()
    return complex(amp[len(amp) // 2])


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_pure_tetm_path_takes_the_same_labels(tetm):
    """``solve_1d``'s own loop, which the conical cases above never enter.

    The decaying root left it 2.5e-6 (TE) and 3.8e-10 (TM) off SMM at ``theta = 0.5+0.3j``.
    """
    theta = 0.5 + 0.3j
    smm, etm = _r0_tetm(theta, 'SMM', tetm), _r0_tetm(theta, 'ETM', tetm)
    etm_move = abs(_r0_tetm(theta * NUDGE, 'ETM', tetm) - etm)
    smm_move = abs(_r0_tetm(theta * NUDGE, 'SMM', tetm) - smm)
    assert etm_move <= 2 * smm_move + 1e-14, (etm_move, smm_move)
    assert abs(etm - smm) < 1e-12, abs(etm - smm)

    broken = _with_etm_rule(_decaying, _r0_tetm, theta, 'ETM', tetm)
    assert abs(broken - smm) > 1e-11, abs(broken - smm)


def _r0_2d(theta, algorithm):
    """A 2D grid of ``eps = 1`` lines around a 5/6-wide ``eps = 25`` block, on ``eps = 25``."""
    cell = np.full((6, 6), 25.0)
    cell[0, :] = 1.0
    cell[:, 0] = 1.0
    solver = wavebit.rcwa(backend='numpy', psi=np.pi / 2, top=1.0, bot=25.0, theta=theta, phi=0.3,
                          truncation_order=[6, 6], wavelength=900e-9,
                          period=[2000e-9, 2000e-9], structure=cell[None],
                          thickness=[100e-9], complex_dtype=0,
                          fourier_coefficients='integrated', factorization='li',
                          layer_algorithm=algorithm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavebit.ComplexIncidenceWarning)
        res = solver.solve()
    amp = np.asarray(res.s_inc.r_s)
    return complex(amp[amp.shape[0] // 2, amp.shape[1] // 2])


def test_the_2d_path_takes_the_same_labels():
    """``solve_2d``'s ETM loop gets the same rule, through ``layer_modes_2d``.

    Before it, ETM sat 5e-6 off SMM here and a 1e-14 nudge moved it by as much.
    """
    theta = 0.5 + 0.3j
    smm, etm = _r0_2d(theta, 'SMM'), _r0_2d(theta, 'ETM')
    etm_move = abs(_r0_2d(theta * NUDGE, 'ETM') - etm)
    smm_move = abs(_r0_2d(theta * NUDGE, 'SMM') - smm)
    assert etm_move <= 2 * smm_move + 1e-14, (etm_move, smm_move)
    assert abs(etm - smm) < 1e-12, abs(etm - smm)

    broken = _with_etm_rule(_decaying, _r0_2d, theta, 'ETM')
    assert abs(broken - smm) > 1e-8, abs(broken - smm)


# ------------------------------------------------------------------------- the rule itself

@pytest.mark.parametrize('lossy', [False, True], ids=['lossless', 'passive'])
def test_the_labels_are_mode_q_at_every_real_k_par(lossy):
    """Bit-identical to :func:`mode_q` wherever ``k_par`` is real and the layer passive.

    ``q**2 = k_par**2 - eps`` then lies in the closed lower half plane, where the decaying root
    is also the half-space's, and the cap cannot fire because that root does not grow. This is
    what keeps every real-angle solve on its released bits.
    """
    rng = np.random.default_rng(20261002)
    n = 200_000
    eps = rng.uniform(-25, 25, n) + (1j * rng.uniform(0, 8, n) if lossy else 0j)
    eigenvalue = rng.uniform(0, 40, n) - eps
    k0_d = rng.uniform(0, 500, n)
    assert np.array_equal(etm_mode_q(np, eigenvalue, k0_d), mode_q(np, eigenvalue))


def test_a_thin_layer_takes_the_half_space_root():
    """At a complex ``k_par`` a thin layer's label is the half-space's own root, ``q = -i*kz``.

    So a layer of a half-space's material has exactly the half-space's waves for its ``+``
    modes, and ``a`` is the identity there -- propagating and evanescent orders alike, on both
    sides of ``Re(k_par) = 0``.
    """
    kx = (0.5 + 0.3j) + 0.45 * np.arange(-30, 31)
    k_par_sq = kx ** 2 + (0.15 + 0.08j) ** 2
    for eps in (1.0 + 0j, 25.0 + 0j, 2.25 + 0.1j):
        kz = boundary_kz(np, eps, k_par_sq)
        q = etm_mode_q(np, k_par_sq - eps, 0.7)
        assert np.max(np.abs(q - (-1j * kz))) < 1e-14 * np.max(np.abs(kz))


def test_the_cap_keeps_the_decaying_root_past_eps_to_the_minus_quarter():
    """The half-space's root is kept until it grows by ``eps**-0.25`` across the layer.

    A propagating mode at a complex ``k_par``: its half-space root has ``Re(q) < 0``. The
    growth ``|exp(-k0*q*d)|`` crosses ``eps**-0.25`` (8192 in double precision) at one
    thickness, and the label switches to the decaying root there and nowhere else.
    """
    eigenvalue = np.asarray([-4.0 + 0.4j])            # q**2 above the real axis: q ~ +-(0.1 + 2i)
    half_space = complex(_uncapped(np, eigenvalue, None)[0])
    assert half_space.real < 0                        # it grows downward
    threshold = -0.25 * np.log(np.finfo(np.float64).eps) / -half_space.real
    below = complex(etm_mode_q(np, eigenvalue, threshold * (1 - 1e-9))[0])
    above = complex(etm_mode_q(np, eigenvalue, threshold * (1 + 1e-9))[0])
    assert below == half_space
    assert above == -half_space and above.real > 0
    assert abs(abs(np.exp(-threshold * half_space)) - 2.0 ** 13) < 1e-6 * 2.0 ** 13
