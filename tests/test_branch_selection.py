"""Which square root a half-space takes, and which one a layer is free to take.

``_core/conventions.py`` holds two selectors that both mean "downward" -- in the propagation
kernels' local frame, from their entry face into the stack, not the public upward height --
and are *not* the same rule. ``docs/time-convention-audit.md`` derives them; this file pins
what the derivation decided, in the regime where the two requirements come apart.

The regime is a **complex in-plane wavevector** — an absorbing superstrate at oblique
incidence, or a complex incidence angle. For every real ``k_par`` a passive medium has
``Im(kz**2) = Im(eps) >= 0``, so ``kz**2`` lies in the closed upper half plane and its
principal root is in the first quadrant: ``Re(kz) > 0`` and ``Im(kz) > 0`` hold *at the same
time* and no rule can distinguish itself. Nothing in the suite reached past that until this
file, which is why the defect below survived: every absorbing-superstrate test in
``test_lossy_media.py`` runs at ``truncation_order=0`` with a lossless substrate, where the two candidate
answers coincide.

What it pins:

* the rule is a **no-op for every real** ``k_par`` — the correction is provably scoped, so no
  released number can move because of it;
* ``r(theta)`` has no step discontinuity where the physics is analytic — the observable the
  defect was found with, and the one it is fixed against;
* the root agrees with the continuation from normal incidence, computed by brute force;
* a **layer's** root is free, which is what makes ``mode_q`` disagreeing with ``boundary_kz``
  legitimate rather than a defect.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit import eps_from_index
from wavebit._core import solve_1d as _s1
from wavebit._core.backend import array_namespace
from wavebit._core.conventions import boundary_kz, mode_q


def _backend_namespace(backend):
    """The array namespace a given ``backend`` code solves in."""
    if backend == 0:
        return array_namespace(np.asarray(0j))
    if backend == 1:
        jnp = pytest.importorskip('jax.numpy')
        return array_namespace(jnp.asarray(np.asarray(0j)))
    torch = pytest.importorskip('torch')
    return array_namespace(torch.zeros(1, dtype=torch.complex128))

WL = 500.0
AU = 0.18 + 3.0j


def _principal(eps, k_par_sq):
    """``boundary_kz`` as it stood before the correction: the principal root, Im tie-break."""
    v = np.sqrt(np.asarray(eps - k_par_sq, dtype=complex))
    flip = np.where(np.abs(v.real) > 1e-12, v.real < 0, v.imag < 0)
    return np.where(flip, -v, v)


# --------------------------------------------------------------- scope of the correction

@pytest.mark.parametrize('lossy', [False, True], ids=['lossless_media', 'passive_media'])
def test_the_rule_is_a_no_op_for_every_real_k_par(lossy):
    """Bit-identical to the principal root wherever ``k_par`` is real.

    This is the guard that keeps the correction from being able to move a released number.
    ``Im(kz**2) = Im(eps) >= 0`` for a passive medium at real ``k_par``, so the segment the
    correction tracks never leaves the closed upper half plane and never meets the cut.

    Measured ``0.0`` — exactly, not approximately — over 4e5 samples on each branch.
    """
    rng = np.random.default_rng(20260811)
    n = 200_000
    eps = rng.uniform(-25, 25, n) + (1j * rng.uniform(0, 8, n) if lossy else 0j)
    k_par_sq = rng.uniform(0, 40, n)
    assert np.array_equal(boundary_kz(np, eps, k_par_sq), _principal(eps, k_par_sq))


def test_both_conditions_hold_at_once_for_a_real_k_par():
    """Why no real-``k_par`` test could ever have caught this.

    ``Re(kz) > 0`` ("carries energy down") and ``Im(kz) >= 0`` ("does not grow down") are
    different demands, and they select the same root for every passive medium at real
    ``k_par``. So a suite that only runs real angles into real-index superstrates cannot
    distinguish the two rules at all — it is not a matter of tolerance.
    """
    rng = np.random.default_rng(7)
    n = 100_000
    eps = rng.uniform(-25, 25, n) + 1j * rng.uniform(0, 8, n)
    kz = boundary_kz(np, eps, rng.uniform(0, 40, n))
    assert (np.real(kz) >= 0).all()
    assert (np.imag(kz) >= 0).all()


# ------------------------------------------------------- the observable that found it

# n_top, n_bot, layer index, thickness, and the angle where the principal root changes sheet.
# `theta_cut` is arcsin(sqrt(Im(eps_bot)/Im(eps_top))); it exists only when BOTH half-spaces
# absorb and the superstrate is the lossier one.
JUMP_CASES = [
    ('lossy_dielectric_on_gold', 1.5 + 0.4j, AU, 2.0, 120.0, 1.2490457724),
    ('gold_like_on_gold', 0.2 + 3.2j, AU, 2.0, 120.0, 1.1644189301),
]


#: The declaration is what these cases want: a uniform 1D stack at no azimuth, where the
#: pure TE/TM formulation is the one under test.
def _amp(n_top, n_bot, n_layer, d, theta, tetm):
    # The cases are written in refractive index because ``theta_cut`` is derived from the
    # half-space *permittivities* below and would read circularly if both were the same
    # spelling. The conversion happens here, where the public API takes over.
    solver = wavebit.rcwa(backend=0, tetm=tetm, top=eps_from_index(n_top),
                            bot=eps_from_index(n_bot), theta=theta,
                            truncation_order=0, wavelength=WL, period=[WL * 2],
                            structure=eps_from_index(np.array([[[n_layer, n_layer]]])),
                            thickness=[d])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = solver.solve().incident
    amp = np.asarray(res.r_s if tetm == 'te' else res.r_p).ravel()
    return complex(amp[len(amp) // 2])


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
@pytest.mark.parametrize('case', JUMP_CASES, ids=[c[0] for c in JUMP_CASES])
def test_reflection_is_analytic_across_the_branch_change(case, tetm):
    """``r(theta)`` must have no step where nothing physical happens.

    ``kz_bot**2(theta) = eps_bot - eps_top*sin^2(theta)`` is a straight segment that stays
    9.03 clear of its only branch point (``kz**2 = 0``) over the whole sweep, so ``r`` is an
    analytic function of ``theta`` there and any step is the branch rule, not the physics.

    The principal root changes sheet mid-segment and produced exactly that step: **2.96 in
    ``r`` across 6.7e-05 rad** for the TE case here, 6.16 for TM — on a quantity of order 1.
    It is invisible to every other kind of check. It does not break energy conservation
    (efficiencies are not even defined against 1 for an absorbing superstrate), it is smooth
    on either side, and both halves are individually plausible.

    Halving the step is the discriminator, and it is a *ratio* test rather than an absolute
    one on purpose: the analytic slope through this angle is large (77 in ``r`` per radian for
    the TM case), so a tolerance on the increment itself would have to be loose enough to
    readmit a small jump. Four halvings of an analytic increment give 1/16 = 0.0625, measured
    to three digits on both polarizations; the same four halvings of a step give 1.0, and the
    step here was 2.96 wide.
    """
    _, n_top, n_bot, n_layer, d, theta_cut = case
    gaps = []
    for h in (8e-4, 4e-4, 2e-4, 1e-4, 5e-5):
        a = _amp(n_top, n_bot, n_layer, d, theta_cut - h, tetm)
        b = _amp(n_top, n_bot, n_layer, d, theta_cut + h, tetm)
        gaps.append(abs(b - a))
    assert 0.05 < gaps[-1] / gaps[0] < 0.08, gaps


# --------------------------------------------------- the root itself, against a brute walk

WALK_CASES = [
    # eps_bot, k_par, and whether the correction moves the root there
    ('lossless_substrate_under_a_lossy_prism', 1.0 + 0j, (1.5 + 0.3j) * np.sin(0.9), False),
    ('lossless_substrate_steep', 1.0 + 0j, (1.5 + 0.3j) * np.sin(1.31), False),
    ('gold_under_a_lossier_top', AU ** 2, (1.5 + 0.4j) * np.sin(1.30), True),
    ('gold_under_a_gold_like_top', AU ** 2, (0.2 + 3.2j) * np.sin(1.30), True),
    ('lossless_metal_under_an_absorbing_top', -9.0 + 0j, (1.5 + 0.4j) * np.sin(0.9), True),
    ('absorbing_on_both_sides', AU ** 2, (1.5 + 0.02j) * np.sin(0.4), False),
]


@pytest.mark.parametrize('case', WALK_CASES, ids=[c[0] for c in WALK_CASES])
def test_the_root_is_the_continuation_from_normal_incidence(case):
    """The closed form must agree with tracking the root by hand.

    ``kz**2 = eps - k_par_sq`` has one branch point, at ``kz**2 = 0``, so the physical root is
    the one reached by deforming continuously from a configuration that is not in doubt. Take
    ``k_par_sq`` to zero: at ``t = 0`` the root is ``sqrt(eps)`` — normal incidence in a
    homogeneous passive medium — and the walk below follows it out to ``t = 1`` by always
    taking whichever of ``+-sqrt`` is nearer the previous step. No rule, no priority, no
    physics beyond "the parameter moved a little, so the root did too".

    ``_off_principal_branch`` is the closed form of that walk, available because
    ``Im(kz2(t))`` is linear in ``t`` and so meets the real axis at most once.
    """
    _, eps, k_par, moves = case
    k_par_sq = complex(k_par) ** 2

    kz = None
    for t in np.linspace(0.0, 1.0, 40_000):
        v = np.sqrt(complex(eps) - t * k_par_sq)
        if kz is None:
            kz = complex(boundary_kz(np, eps, 0.0))       # t = 0: kz2 = eps, not in doubt
        else:
            kz = v if abs(v - kz) <= abs(-v - kz) else -v

    got = complex(boundary_kz(np, eps, k_par_sq))
    assert abs(got - kz) < 1e-9, (got, kz)
    assert (abs(got - complex(_principal(eps, k_par_sq))) > 1e-9) == moves


# ---------------------------------------------------------------- a layer's root is free

LAYER_CASES = [
    ('real_angle', dict(top=1.0, bot=eps_from_index(1.5), theta=0.2)),
    ('absorbing_superstrate', dict(top=eps_from_index(1.5 + 0.3j),
                                   bot=eps_from_index(1.5), theta=0.6)),
    ('absorbing_superstrate_on_gold', dict(top=eps_from_index(1.5 + 0.4j),
                                           bot=eps_from_index(AU), theta=1.3)),
]


def _grating(tetm, thickness, **kw):
    cell = eps_from_index(np.array([[[1.0] * 8 + [3.4] * 8]]))
    return dict(backend=0, tetm=tetm, truncation_order=6, wavelength=WL, period=[300.0], structure=cell,
                thickness=[thickness], **kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = wavebit.rcwa(**cfg).solve().incident
    return np.asarray(res.r_s if cfg['tetm'] == 0 else res.r_p).ravel()


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
@pytest.mark.parametrize('case', LAYER_CASES, ids=[c[0] for c in LAYER_CASES])
def test_a_layer_mode_root_is_free(case, tetm):
    """Relabelling which of ``+-q`` is "down" for a layer mode cannot move the answer.

    This is what makes ``mode_q`` and ``boundary_kz`` selecting *unrelated* roots at complex
    ``k_par`` legitimate rather than a defect: they are answering different questions. A
    half-space's root is a boundary condition and there is only one right answer;
    a layer's ``+-q`` pair is used in full by the stack algebra, so the label is free and
    ``mode_q`` may pick on whatever grounds it likes.

    It picks on numerical grounds: ``Re(q) >= 0`` keeps ``|exp(-k0*q*d)| <= 1``. That is not
    decoration — driven with ``boundary_kz``'s rule instead, 47% of the modes of this layer
    grow, and the same solve that agrees to 3.8e-15 at 50 nm disagrees by 64 at 500 nm and
    raises ``FloatingPointError`` at 4 um. Hence the thin layer here: it is measuring the
    algebra, and the algebra is what is free.
    """
    _, inc = case
    cfg = _grating(tetm, 40.0, **inc)
    base = _run(cfg)

    original = _s1.mode_q
    worst = 0.0
    try:
        for seed in range(6):
            rng = np.random.default_rng(seed)

            def relabelled(xp, w, _o=original, _rng=rng):
                q = _o(xp, w)
                return xp.where(xp.asarray(_rng.random(np.shape(q)) < 0.5), -q, q)

            _s1.mode_q = relabelled
            worst = max(worst, float(np.max(np.abs(_run(cfg) - base))))
    finally:
        _s1.mode_q = original
    assert worst < 1e-10, worst


def test_the_mode_q_tie_break_is_worth_nothing():
    """``mode_q``'s ``Im(q) <= 0`` tie-break decides nothing, and that is the point.

    It fires only where ``Re(q) == 0``, where ``|exp(-k0*q*d)| == 1`` either way — so it
    cannot be serving boundedness, and by the freedom above it cannot be serving the answer.
    Inverting it moves nothing (measured 0.0 exactly on the complex-``k_par`` cases, 1.4e-15
    on the real-angle one). Stated so that nobody re-derives it from ``boundary_kz``'s
    tie-break, which is the opposite sign and *does* decide something.
    """
    def inverted(xp, eigenvalue):
        # `xp.sqrt`, matching `mode_q`. Spelling it `** 0.5` here would make this mirror
        # differ from the real function in two ways at once, and the test could no longer
        # say which of them moved the answer.
        v = xp.sqrt(eigenvalue)
        flip = xp.where(xp.abs(xp.real(v)) > 1e-12, xp.real(v) < 0, xp.imag(v) < 0)
        return xp.where(flip, -v, v)

    cfg = _grating('te', 120.0, top=eps_from_index(1.5 + 0.3j), bot=eps_from_index(1.5),
                   theta=0.6)
    base = _run(cfg)
    original = _s1.mode_q
    try:
        _s1.mode_q = inverted
        assert float(np.max(np.abs(_run(cfg) - base))) < 1e-12
    finally:
        _s1.mode_q = original


def test_the_two_selectors_disagree_at_complex_k_par_by_design():
    """``q = -i*kz`` is an identity for real ``k_par`` and is not one beyond it.

    Pinned because it looks like a bug and is not. ``mode_q`` returns ``+i*kz`` rather than
    ``-i*kz`` for the layer below — the other root — and that is allowed by
    ``test_a_layer_mode_root_is_free``.
    """
    eps, k_par = 2.0 + 0j, (1.5 + 0.3j) * np.sin(0.6)

    real_kz = complex(boundary_kz(np, eps, complex(k_par.real) ** 2))
    real_q = complex(mode_q(np, np.asarray(complex(k_par.real) ** 2 - eps)))
    assert abs(real_q - (-1j * real_kz)) < 1e-12          # identity at real k_par

    kz = complex(boundary_kz(np, eps, complex(k_par) ** 2))
    q = complex(mode_q(np, np.asarray(complex(k_par) ** 2 - eps)))
    assert abs(q - (-1j * kz)) > 1.0                      # and not one here
    assert abs(q - (+1j * kz)) < 1e-12                    # it took the other root
    assert np.real(q) >= 0                                # which is the one that cannot grow


# --------------------------------------------------------------------------------------------
# An evanescent order's `Re(kz)` is a *hard* zero, on every backend.
#
# `kz2 ** 0.5` and `sqrt(kz2)` are one number in exact arithmetic and two numbers in a float.
# `**0.5` has no complex-sqrt path: it goes through `exp(0.5*log z)`, and for a negative real
# `z` that returns `cos(pi/2)` as the real part. `pi/2` is not representable, so `cos` of the
# nearest double is 6.123233995736766e-17 rather than 0, scaled by `|Im(kz)|`.
#
# NumPy and Torch special-case the exponent 0.5 back to `csqrt` and so hide it; JAX and Python's
# own `complex` do not. One source therefore produced different numbers per backend, and not
# cosmetically: an evanescent order's efficiency carries a factor `Re(kz)`, so the spelling alone
# decided whether a cut-off channel reported 0.0 or 1e-19. The same residue is what made the
# incident z-flux of an evanescent order read as ~1e-17 -- recorded once as "numerically zero"
# when it is structurally zero.
#
# These pin the exactness rather than the fix, so they fail if any complex root on this path is
# spelled `** 0.5` again.

BACKENDS = [(0, 'numpy'), (1, 'jax'), (2, 'torch')]

# A grating whose orders straddle the vacuum light line, so one solve carries both kinds and the
# propagating ones witness that the solve is not simply empty.
CUT_OFF_SCENE = dict(wavelength=500., period=[900.], truncation_order=[5, 0],
                     top=1., bot=eps_from_index(1.5), theta=0.3, phi=0.,
                     structure=np.array([[[1.0, eps_from_index(3.5),
                                           1.0, eps_from_index(3.5)]]]),
                     thickness=[300.])
CUT_OFF_U = np.array([np.sin(0.3) + m * 500. / 900. for m in range(-5, 6)])


@pytest.mark.parametrize('backend,name', BACKENDS, ids=[n for _, n in BACKENDS])
def test_evanescent_kz_has_no_real_part_at_all(backend, name):
    """``Re(kz)`` of a cut-off order is bitwise 0, not merely small."""
    xp = _backend_namespace(backend)
    # k_par**2 of 1.44, 4.0, 9.0 -- all past the light line of a vacuum superstrate.
    kz = np.asarray(boundary_kz(xp, xp.asarray(np.asarray(1.0 + 0j)),
                                xp.asarray(np.array([1.44, 4.0, 9.0], dtype=complex))))
    real = [float(v) for v in np.real(kz)]
    assert real == [0.0, 0.0, 0.0], f'{name}: Re(kz) leaked {real}'
    # and the decaying root is still the one taken, so this is not a sign flip in disguise
    assert all(float(v) > 0 for v in np.imag(kz)), f'{name}: took the growing root'


@pytest.mark.parametrize('backend,name', BACKENDS, ids=[n for _, n in BACKENDS])
@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_cut_off_orders_report_exactly_zero_efficiency(backend, tetm, name):
    """A grating's cut-off orders carry real amplitude and exactly zero efficiency.

    The amplitude half is the point: the zero has to come from the ``Re(kz)`` factor, not from
    the channel being empty. Without it this passes on a solver that zeroes them by hand.
    """
    res = wavebit.rcwa(backend=backend, tetm=tetm, **CUT_OFF_SCENE).solve()
    R = np.asarray(res.R, dtype=float).ravel()

    cut = np.abs(CUT_OFF_U) > 1.0
    assert cut.any() and (~cut).any(), 'scene must carry both kinds of order'

    leaked = R[cut][np.nonzero(R[cut])]
    assert leaked.size == 0, f'{name}/{tetm}: cut-off orders reported {leaked}'
    assert R[~cut].sum() > 0.0, f'{name}/{tetm}: no propagating order carried anything'
