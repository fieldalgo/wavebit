"""Which square root a half-space takes, and which one a layer is free to take.

``src/wavebit/_common/conventions.py`` holds two selectors that both mean "downward" -- in the propagation
kernels' local frame, from their entry face into the stack, whose ``z`` is not, in general,
the public height: it coincides with it under bottom incidence and is its mirror
(``h_total - z``) under top incidence -- and are *not* the same rule.
``docs/theory/rcwa/branch-cuts.md`` and ``docs/validation/rcwa/time-convention-audit.md`` derive them; this file pins what
was decided, in the regime where the two requirements come apart.

That regime is ``kz**2`` below the real axis. Two things put it there: a **complex in-plane
wavevector**, and a **gain** half-space. Since 2026-09 only a complex incidence angle reaches a
complex in-plane wavevector, because a real angle in an absorbing superstrate is a phase angle
with a real ``k_par``. For every real ``k_par``, a passive medium has
``Im(kz**2) = Im(eps) >= 0``, so ``kz**2`` lies in the closed upper half plane and its principal
root is in the first quadrant. ``Re(kz) > 0`` and ``Im(kz) > 0`` then hold *at the same time*,
and no rule can distinguish itself there.

Below the axis the half-space takes Maystre's cut, ``Re(kz) + Im(kz) > 0`` (since 2026-10-02).
What this file pins:

* the rule is a **no-op for every real** ``k_par`` in a passive medium, bit for bit;
* the root is the **limit of the half-space's own absorption**, computed by brute force; for a
  gain medium, that is the root reached by switching the gain on;
* nothing jumps as ``Im(k_par)`` passes through zero, neither the root nor a grating's ``r``.
  The rule before this one lacked that property;
* ``r(theta)`` has no step where the physics is analytic. That is the observable the 2026-08
  defect was found with;
* a gain half-space under total internal reflection reflects more than it receives
  (evanescent gain);
* a **layer's** root is free, which is what makes ``mode_q`` disagreeing with ``boundary_kz``
  legitimate rather than a defect.
"""

from wavebit.compat import legacy as _legacy_wavebit
import warnings

import numpy as np
import pytest

from wavebit import eps_from_refractive_index
from wavebit._common import conventions as _conventions
from wavebit._common.backend import array_namespace
from wavebit._common.conventions import boundary_kz, mode_q


def _backend_namespace(backend):
    """The array namespace a given ``backend`` solves in."""
    if backend == 'numpy':
        return array_namespace(np.asarray(0j))
    if backend == 'jax':
        jnp = pytest.importorskip('jax.numpy')
        return array_namespace(jnp.asarray(np.asarray(0j)))
    torch = pytest.importorskip('torch')
    return array_namespace(torch.zeros(1, dtype=torch.complex128))

WL = 500.0
AU = 0.18 + 3.0j


def _principal(eps, k_par_sq):
    """The principal root with an ``Im`` tie-break: ``boundary_kz`` before 2026-08-12.

    It is still the answer at every real ``k_par`` in a passive medium, which is what the
    no-op test below holds the present rule to.
    """
    v = np.sqrt(np.asarray(eps - k_par_sq, dtype=complex))
    flip = np.where(np.abs(v.real) > 1e-12, v.real < 0, v.imag < 0)
    return np.where(flip, -v, v)


# --------------------------------------------------------------- scope of the correction

@pytest.mark.parametrize('lossy', [False, True], ids=['lossless_media', 'passive_media'])
def test_the_rule_is_a_no_op_for_every_real_k_par(lossy):
    """Bit-identical to the principal root wherever ``k_par`` is real.

    This is the guard that keeps a change of the half-space rule from moving a released number.
    ``Im(kz**2) = Im(eps) >= 0`` for a passive medium at real ``k_par``. The principal root is
    then in the closed first quadrant, where ``Re(kz) + Im(kz) >= |kz|``, so Maystre's cut never
    takes the other root. That held for the 2026-08-12 continuation rule too, for its own reason.

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
#
# The cases were found under the formal reading of a real angle, `k_par = n_top*sin(theta)`.
# Since 2026-09 a real angle in an absorbing superstrate is a phase angle with a real `k_par`,
# and only a complex `theta` reaches the formal reading. `_FORMAL` is the smallest imaginary
# part that does: it selects the formal reading and leaves the segment the cases were written
# for where it was. Without it these cases solve at a real `k_par` and pass under any rule.
_FORMAL = 1e-9j
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
    solver = _legacy_wavebit.rcwa(backend='numpy', tetm=tetm, top=eps_from_refractive_index(n_top),
                            bot=eps_from_refractive_index(n_bot), theta=theta + _FORMAL,
                            truncation_order=0, wavelength=WL, period=[WL * 2],
                            structure=eps_from_refractive_index(np.array([[[n_layer, n_layer]]])),
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

    ``kz_bot**2(theta) = eps_bot - eps_top*sin^2(theta)`` is a straight segment. For the
    lossy dielectric it stays 9.03 clear of its only branch point (``kz**2 = 0``) over the whole
    sweep, so ``r`` is an analytic function of ``theta`` there and any step is the branch rule,
    not the physics.

    At ``theta_cut`` the segment crosses the negative real axis, the principal root's cut. The
    principal root changed sheet there and produced exactly that step: **2.96 in ``r`` across
    6.7e-05 rad** for the TE case here, 6.16 for TM, on a quantity of order 1. It is invisible
    to every other kind of check. It does not break energy conservation (efficiencies are not
    even defined against 1 for an absorbing superstrate), it is smooth on either side, and both
    halves are individually plausible. Maystre's cut is the negative *imaginary* axis, which the
    lossy dielectric's segment never meets. The gold-like top's segment does meet it, at
    1.215 rad, 0.045 from the branch point, outside the window tested here. Close to a branch
    point no rule is continuous along every path.

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
    # eps, k_par, and whether the root is not the principal one there
    ('lossless_substrate_under_a_lossy_prism', 1.0 + 0j, (1.5 + 0.3j) * np.sin(0.9), True),
    ('lossless_substrate_steep', 1.0 + 0j, (1.5 + 0.3j) * np.sin(1.31), True),
    ('gold_under_a_lossier_top', AU ** 2, (1.5 + 0.4j) * np.sin(1.30), True),
    ('gold_under_a_gold_like_top', AU ** 2, (0.2 + 3.2j) * np.sin(1.30), False),
    ('lossless_metal_under_an_absorbing_top', -9.0 + 0j, (1.5 + 0.4j) * np.sin(0.9), True),
    ('absorbing_on_both_sides', AU ** 2, (1.5 + 0.02j) * np.sin(0.4), False),
    ('gain_under_total_internal_reflection', 1.0 - 0.05j, 1.5 * np.sin(np.pi / 4) + 0j, True),
    ('gain_propagating', 1.0 - 0.05j, 1.5 * np.sin(0.3) + 0j, False),
]


@pytest.mark.parametrize('case', WALK_CASES, ids=[c[0] for c in WALK_CASES])
def test_the_root_is_the_limit_of_the_half_spaces_own_absorption(case):
    """The rule must agree with tracking the root by hand as absorption is taken away.

    Add ``i*L`` to the half-space's permittivity, with ``L`` large enough to lift ``kz**2`` into
    the upper half plane. The root there is the first-quadrant one, both outgoing and decaying,
    and not in doubt. Take the absorption away at fixed ``k_par`` and follow the root by always
    taking whichever of ``+-sqrt`` is nearer the previous step. No rule, no priority, no physics
    beyond "the parameter moved a little, so the root did too". ``kz**2`` moves straight down, so
    the walk can reach the branch point only along the negative imaginary axis. That makes
    Maystre's cut the closed form of this walk. For a gain medium the walk passes
    ``Im(eps) = 0`` and goes on, so it is also the root reached by switching the gain on.

    Until 2026-10-02 the rule was the continuation from normal incidence instead. It disagreed
    with this walk on the two lossless substrates, where it let an evanescent field grow, and on
    the gold-like top. It also took the principal root under total internal reflection into a
    gain medium, which is the root of a lossy one.
    """
    _, eps, k_par, moves = case
    kz2 = complex(eps) - complex(k_par) ** 2
    lift = abs(kz2.imag) + 10.0

    kz = None
    for s in np.linspace(0.0, 1.0, 40_000):
        v = np.sqrt(kz2 + 1j * lift * (1.0 - s))
        if kz is None:
            kz = v                     # upper half plane: the principal root, first quadrant
        else:
            kz = v if abs(v - kz) <= abs(-v - kz) else -v

    got = complex(boundary_kz(np, eps, complex(k_par) ** 2))
    assert abs(got - kz) < 1e-9, (got, kz)
    assert (abs(got - complex(_principal(eps, complex(k_par) ** 2))) > 1e-9) == moves


def test_the_root_continues_across_the_real_k_par_axis():
    """A vanishing ``Im(k_par)`` must leave the root where the real axis put it.

    Random lossless and absorbing media, evanescent and propagating orders alike. At
    ``Im(k_par) = +-1e-9`` the root stays within 1e-6 of the real-axis one, given a margin
    from the Rayleigh point, where ``kz**2`` sits on the branch point itself.

    The rule before 2026-10-02 failed exactly this. Its cut lay along the real evanescent axis,
    so every evanescent order with ``Re(k_par)*Im(k_par) > 0`` took the growing root at any
    ``Im(k_par)``, round-off included: 6858 of 6858 such orders at ``Im(k_par) = 1e-3``.
    """
    rng = np.random.default_rng(20261002)
    n = 100_000
    for lossy in (False, True):
        eps = rng.uniform(1, 6, n) + (1j * rng.uniform(0, 2, n) if lossy else 0j)
        a = rng.uniform(-6, 6, n)
        clear = np.abs(a ** 2 - eps.real) > 1e-3
        on_axis = boundary_kz(np, eps, a ** 2 + 0j)
        for sign in (1, -1):
            off_axis = boundary_kz(np, eps, (a + sign * 1e-9j) ** 2)
            assert np.max(np.abs(off_axis - on_axis)[clear]) < 1e-6, (lossy, sign)


def _grating_r0(theta):
    """Air over glass with a binary 9/1 grating, eleven orders: the zeroth reflection amplitude."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = _legacy_wavebit.rcwa(backend='numpy', tetm='te', top=1.0, bot=2.25, theta=theta, truncation_order=5,
                           wavelength=0.9, period=[0.7], structure=np.array([[[9.0, 1.0]]]),
                           thickness=[0.15], fourier_coefficients='integrated').solve()
    amp = np.asarray(res.incident.r_s).ravel()
    return complex(amp[len(amp) // 2])


def test_a_grating_is_continuous_as_its_angle_leaves_the_real_axis():
    """The observable for the property above: ``r0`` as ``Im(theta)`` passes through zero.

    At ``theta = 0.2`` both first orders are evanescent in air. Under the rule before
    2026-10-02, ``Im(theta) = +1e-9`` moved ``r0`` by 0.22 and ``-1e-9`` by 0.36. Each sign
    turned the evanescent orders on one side into growing ones. RETICOLO, whose cut this rule
    now takes, moves by nothing. ``test_reticolo_complex_kx_golden`` holds the same grating to
    RETICOLO's numbers at a finite ``Im(theta)``.
    """
    base = _grating_r0(0.2)
    for im in (1e-9, -1e-9):
        assert abs(_grating_r0(0.2 + 1j * im) - base) < 1e-7, im


@pytest.mark.parametrize('theta_deg, gain_exceeds_loss', [(20.0, False), (45.0, True)],
                         ids=['below_critical', 'total_internal_reflection'])
def test_a_gain_substrate_reflects_as_switching_its_gain_on_says(theta_deg, gain_exceeds_loss):
    """Glass onto ``eps = 1 -+ 0.05i``: gain and loss reflect alike below the critical angle only.

    The Fresnel coefficient is evaluated here with the root reached by switching the gain on
    from zero, written out independently of ``boundary_kz``. Below the critical angle that root
    grows forward, ``r_gain = conj(r_loss)``, and the two media reflect alike. Under total
    internal reflection it is the decaying root with ``Re(kz) < 0``, and the reflectance
    exceeds one: evanescent gain (J. O. Grepstad and J. Skaar, Opt. Express 19, 21404, 2011).
    At 45 degrees TE that is ``R = 1.2646``. Until 2026-10-02 this solve returned 0.7907, the
    *lossy* mirror's value: the principal root read the gain as loss.
    """
    theta = np.deg2rad(theta_deg)

    def solved(eps_bot):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = _legacy_wavebit.rcwa(backend='numpy', tetm='te', top=2.25, bot=eps_bot, theta=theta,
                               truncation_order=0, wavelength=WL, period=[WL * 2],
                               structure=np.full((1, 1, 2), 2.25), thickness=[100.0]).solve()
        return float(np.sum(res.R))

    k_top = np.sqrt(2.25 * np.cos(theta) ** 2)
    kz2 = (1.0 - 0.05j) - 2.25 * np.sin(theta) ** 2
    k_bot = np.sqrt(kz2)
    if kz2.real < 0:                   # evanescent before the gain: the gain keeps it decaying
        k_bot = k_bot if k_bot.imag > 0 else -k_bot
    else:                              # propagating before the gain: it keeps carrying power down
        k_bot = k_bot if k_bot.real > 0 else -k_bot
    switched_on = abs((k_top - k_bot) / (k_top + k_bot)) ** 2

    gain, loss = solved(1.0 - 0.05j), solved(1.0 + 0.05j)
    assert abs(gain - switched_on) < 1e-12, (gain, switched_on)
    assert (gain > loss + 1e-3) == gain_exceeds_loss, (gain, loss)
    assert (gain > 1.0) == gain_exceeds_loss


# ---------------------------------------------------------------- a layer's root is free

LAYER_CASES = [
    ('real_angle', dict(top=1.0, bot=eps_from_refractive_index(1.5), theta=0.2)),
    ('absorbing_superstrate', dict(top=eps_from_refractive_index(1.5 + 0.3j),
                                   bot=eps_from_refractive_index(1.5), theta=0.6)),
    ('absorbing_superstrate_on_gold', dict(top=eps_from_refractive_index(1.5 + 0.4j),
                                           bot=eps_from_refractive_index(AU), theta=1.3)),
]


def _grating(tetm, thickness, **kw):
    # SMM, because the freedom pinned below is SMM's. ETM's recursion inverts a transmission into
    # the medium below each layer, so its labels are not free: they take the half-space's root
    # (`etm_mode_q`), pinned in `tests/rcwa/test_etm_layer_labels.py`.
    cell = eps_from_refractive_index(np.array([[[1.0] * 8 + [3.4] * 8]]))
    return dict(backend='numpy', tetm=tetm, truncation_order=6, wavelength=WL, period=[300.0], structure=cell,
                thickness=[thickness], layer_algorithm='SMM', **kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = _legacy_wavebit.rcwa(**cfg).solve().incident
    # `cfg['tetm']` is the word, so it is compared as one: against `0` it was never equal, and
    # the TE half of the test below read `r_p` -- identically zero on a TE solve -- and
    # compared zeros with zeros.
    return np.asarray(res.r_s if cfg['tetm'] == 'te' else res.r_p).ravel()


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
@pytest.mark.parametrize('case', LAYER_CASES, ids=[c[0] for c in LAYER_CASES])
def test_a_layer_mode_root_is_free(case, tetm):
    """Relabelling which of ``+-q`` is "down" for a layer mode cannot move an SMM answer.

    This is what makes ``mode_q`` and ``boundary_kz`` selecting *unrelated* roots at complex
    ``k_par`` legitimate rather than a defect: they are answering different questions. A
    half-space's root is a boundary condition and there is only one right answer;
    a layer's ``+-q`` pair is used in full by the stack algebra, so the label is free and
    ``mode_q`` may pick on whatever grounds it likes -- as long as the layers are joined by
    scattering matrices. ETM's are not free (``tests/rcwa/test_etm_layer_labels.py``).

    It picks on numerical grounds: ``Re(q) >= 0`` keeps ``|exp(-k0*q*d)| <= 1``. That is not
    decoration — driven with ``boundary_kz``'s rule instead, 47% of the modes of this layer
    grow, and the same solve that agrees to 3.8e-15 at 50 nm disagrees by 64 at 500 nm and
    raises ``FloatingPointError`` at 4 um. Hence the thin layer here: it is measuring the
    algebra, and the algebra is what is free.
    """
    _, inc = case
    cfg = _grating(tetm, 40.0, **inc)
    base = _run(cfg)
    assert np.abs(base).max() > 1e-3, 'the amplitude read is empty, so relabelling cannot move it'

    original = _conventions.mode_q
    worst = 0.0
    try:
        for seed in range(6):
            rng = np.random.default_rng(seed)

            def relabelled(xp, w, _o=original, _rng=rng):
                q = _o(xp, w)
                return xp.where(xp.asarray(_rng.random(np.shape(q)) < 0.5), -q, q)

            _conventions.mode_q = relabelled
            worst = max(worst, float(np.max(np.abs(_run(cfg) - base))))
    finally:
        _conventions.mode_q = original
    assert worst < 1e-10, worst


def test_the_mode_q_tie_break_is_worth_nothing():
    """``mode_q``'s ``Im(q) <= 0`` tie-break decides nothing, and that is the point.

    It fires only where ``Re(q) == 0``, where ``|exp(-k0*q*d)| == 1`` either way — so it
    cannot be serving boundedness, and by the freedom above it cannot be serving the answer.
    Inverting it moves this SMM solve by 3.0e-16, and by exactly 0.0 when ``theta`` carries an
    imaginary part of 1e-9. Stated so that nobody re-derives it from ``boundary_kz``'s
    tie-break, which is the opposite sign and *does* decide something. ETM does not read it:
    its labels come from ``etm_mode_q``.
    """
    def inverted(xp, eigenvalue):
        # `xp.sqrt`, matching `mode_q`. Spelling it `** 0.5` here would make this mirror
        # differ from the real function in two ways at once, and the test could no longer
        # say which of them moved the answer.
        v = xp.sqrt(eigenvalue)
        flip = xp.where(xp.abs(xp.real(v)) > 1e-12, xp.real(v) < 0, xp.imag(v) < 0)
        return xp.where(flip, -v, v)

    cfg = _grating('te', 120.0, top=eps_from_refractive_index(1.5 + 0.3j),
                   bot=eps_from_refractive_index(1.5), theta=0.6)
    base = _run(cfg)
    original = _conventions.mode_q
    try:
        _conventions.mode_q = inverted
        assert float(np.max(np.abs(_run(cfg) - base))) < 1e-12
    finally:
        _conventions.mode_q = original


def test_the_two_selectors_disagree_at_complex_k_par_by_design():
    """``q = -i*kz`` is an identity for real ``k_par`` and is not one beyond it.

    Pinned because it looks like a bug and is not. ``mode_q`` returns ``+i*kz`` rather than
    ``-i*kz`` for the layer below — the other root — and that is allowed by
    ``test_a_layer_mode_root_is_free``. ETM's rule, ``etm_mode_q``, takes ``-i*kz`` there
    instead, because its recursion is the one place the label is not free.
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

BACKENDS = ['numpy', 'jax', 'torch']

# A grating whose orders straddle the vacuum light line, so one solve carries both kinds and the
# propagating ones witness that the solve is not simply empty.
CUT_OFF_SCENE = dict(wavelength=500., period=[900.], truncation_order=[5, 0],
                     top=1., bot=eps_from_refractive_index(1.5), theta=0.3, phi=0.,
                     structure=np.array([[[1.0, eps_from_refractive_index(3.5),
                                           1.0, eps_from_refractive_index(3.5)]]]),
                     thickness=[300.])
CUT_OFF_U = np.array([np.sin(0.3) + m * 500. / 900. for m in range(-5, 6)])


@pytest.mark.parametrize('backend', BACKENDS)
def test_evanescent_kz_has_no_real_part_at_all(backend):
    """``Re(kz)`` of a cut-off order is bitwise 0, not merely small."""
    xp = _backend_namespace(backend)
    # k_par**2 of 1.44, 4.0, 9.0 -- all past the light line of a vacuum superstrate.
    kz = np.asarray(boundary_kz(xp, xp.asarray(np.asarray(1.0 + 0j)),
                                xp.asarray(np.array([1.44, 4.0, 9.0], dtype=complex))))
    real = [float(v) for v in np.real(kz)]
    assert real == [0.0, 0.0, 0.0], f'{backend}: Re(kz) leaked {real}'
    # and the decaying root is still the one taken, so this is not a sign flip in disguise
    assert all(float(v) > 0 for v in np.imag(kz)), f'{backend}: took the growing root'


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_cut_off_orders_report_exactly_zero_efficiency(backend, tetm):
    """A grating's cut-off orders carry real amplitude and exactly zero efficiency.

    The amplitude half is the point: the zero has to come from the ``Re(kz)`` factor, not from
    the channel being empty. Without it this passes on a solver that zeroes them by hand.
    """
    res = _legacy_wavebit.rcwa(backend=backend, tetm=tetm, **CUT_OFF_SCENE).solve()
    R = np.asarray(res.R, dtype=float).ravel()

    cut = np.abs(CUT_OFF_U) > 1.0
    assert cut.any() and (~cut).any(), 'scene must carry both kinds of order'

    leaked = R[cut][np.nonzero(R[cut])]
    assert leaked.size == 0, f'{backend}/{tetm}: cut-off orders reported {leaked}'
    assert R[~cut].sum() > 0.0, f'{backend}/{tetm}: no propagating order carried anything'
