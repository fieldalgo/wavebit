"""Permeability on the 1D and conical paths — the half that needs no oracle.

Companion to ``test_magnetic_layer.py``, which does the same job for the 2D path. Landed
2026-09-01, lifting the refusal ``docs/magnetic-materials.md`` §10 recorded; the derivation and
what it cost are in §11 there.

Four instruments, and the order is the order of their strength:

1. **The duality**, which is a theorem rather than a comparison. Maxwell maps onto itself under
   ``(E, H, eps, mu) -> (H, -E, mu, eps)``, and on this path that swap carries the **TE branch
   onto the TM branch** — two independently written operators that must return one number. No
   solver that ignores a permeability can satisfy it, and it needs nothing external.
2. **A closed form** for a uniform magnetic slab, with no wavebit anywhere in it. It carries the
   two cases a solver holding only a refractive index cannot express at all.
3. **The 2D path**, which had ``mu`` since 2026-08-14 against a golden frozen before the code
   existed. A 1D grating is a 2D grating that does not vary in y, so the two must agree — and
   this is the only instrument here that reaches the *conical* operator's coupling blocks.
4. **The two connecting algorithms**, on efficiencies *and on the reconstructed field*. The
   field half is not redundant: ``muz_conv_i`` reaches reconstruction as per-layer state that
   ETM and SMM build by different routes, and an efficiency comparison cannot see it.

**Energy conservation is deliberately not on that list.** This repository's own history says
``R + T = 1`` holds for magnetic scenes that are wrong — a boundary using the same admittance
for both polarizations conserves energy perfectly, and so does a reconstruction whose ``H_z``
is scaled by ``mu_zz``. Every assertion below is against something that can fail.

**A crossing neither this file nor its partner stands at (found 2026-09-07, unfixed).**
``incidence_side='bottom'`` x a magnetic scene is arbitrated by nothing. This file never sets
``incidence_side``; ``test_backside_incidence.py`` never names a permeability. Measured: dropping
the ``eps_top`` / ``eps_bot`` half-space swap from the backside mirror in EITHER 1D solve
(``solve_1d.py``, the pure TE/TM branch and the conical one) leaves the whole suite green while a
passive stack returns ``R + T = 2.69``. It came out of a mutation audit run for the FDTD path,
where the same shape of hole -- a cell of the input cross-product no test visits -- had already
shipped two defects. Closing it is a scene, not a new instrument: one magnetic stack solved from
both sides, checked against the duality this file already uses.
"""
import numpy as np
import pytest

import wavebit
from wavebit import Material

WL, PERIOD, THICK = 900.0, 480.0, 240.0
TO = 4
N = 2 * TO + 1
NPIX = 64

# Off `phi = 0`, and that is load-bearing rather than decorative. At `phi = 0` the conical
# problem has `ky = 0`, its two mode families decouple exactly, and every block that carries a
# `ky` factor (`V_12`) or belongs to the second family (`muy_conv`) becomes unreachable. A
# mutation survey run at `phi = 0` left three of sixteen mutants alive for that reason alone;
# moved here, all sixteen die. `psi` mixes s and p for the same reason.
PHI, PSI, THETA = 0.55, 0.7, 0.18


def _raster(rows, background, inclusion):
    a = np.ones((1, rows, NPIX)) * background
    a[..., 16:40] = inclusion
    return a


def _grating(rows, theta=THETA, **kw):
    """The same patterned magnetic grating, routed by its row count and declaration.

    One row plus a ``tetm`` is the pure TE/TM path, one row plus a ``psi`` the conical one, and
    three identical rows the 2D path at ``n_orders_v == 1`` — the same physical grating each
    time. The raster front door is the one that reaches these paths at all: a shape structure
    sets ``modeling_path='parametric'``, which routes to the 2D solve whatever its geometry.
    """
    return wavebit.rcwa(backend='numpy', top=1.0, bot=Material(2.25, mu=1.3), theta=theta,
                        truncation_order=(TO, 0), period=(PERIOD, PERIOD), wavelength=WL,
                        thickness=[THICK], answers='field',
                        structure=_raster(rows, 1.0, 6.25),
                        mu=_raster(rows, 1.0, 2.4), **kw)


def _solved(rows, **kw):
    solver = _grating(rows, **kw)
    result = solver.solve()
    cell = np.asarray(solver.reconstruct_field(sample_x=20, sample_y=1, sample_z=10))
    return solver, result, (cell[0] if cell.ndim == 5 else cell)


# --------------------------------------------------------------------------- 1. the duality

def _arbitrary(seed, scale, n_layers=2):
    """A non-scalar, non-commuting stack of convolution-shaped matrices.

    Deliberately not a scalar times the identity: a scalar permeability is symmetric under the
    x/y exchange, so it passes whether or not the operator pairs ``mu_x`` with the branch that
    owns it. The 2D work recorded the same trap (``docs/magnetic-materials.md`` §2) — the
    isotropic case is the one that looks like proof and cannot see the error.
    """
    rs = np.random.RandomState(seed)
    a = rs.rand(n_layers, N, N) + 1j * rs.rand(n_layers, N, N) * 0.05
    return (a + np.transpose(a, (0, 2, 1))) * 0.5 * scale + np.eye(N)[None] * 2.0


def _inv_stack(a):
    return np.stack([np.linalg.inv(x) for x in a])


def test_te_with_a_permeability_is_tm_with_the_two_tensors_traded():
    """``(E, H, eps, mu) -> (H, -E, mu, eps)`` carries this path's TE branch onto its TM one.

    The strongest statement available here, because the two branches are *separately written*:
    TE builds ``mu_x (Kx muz_i Kx - eps_y)`` and TM builds ``eps_x (Kx epz_i Kx - mu_y)``, in
    different lines, with different inverses in ``V``, against different half-space
    admittances. Exchanging the two tensors everywhere — layers **and** half-spaces — has to
    map one onto the other exactly.

    Stated as a **trade** rather than an invariance, and the control at the end is what makes
    the assertion mean something: TE and TM are 6.5e-3 apart on this scene when the tensors are
    *not* traded, so a solver that ignored ``mu`` entirely would fail the control, not pass it.

    ``n = sqrt(eps*mu)`` is itself invariant under the swap, which is why both runs are handed
    the same indices and only the material slots move.
    """
    eps = (_arbitrary(11, 0.20), _arbitrary(12, 0.25))
    mu = (_arbitrary(13, 0.10), _arbitrary(14, 0.30))
    # `eps_z == eps_y` and `mu_z == mu_y`, which is not a simplification: on a 1D grating all
    # four of those slots are tangential to a boundary perpendicular to x, so all four are
    # built by the direct rule from one profile. See `layer_modes_1d_conical`.
    eps_slots = (eps[0], eps[1], _inv_stack(eps[1]))
    mu_slots = (mu[0], mu[1], _inv_stack(mu[1]))
    mu_top, mu_bot, eps_top, eps_bot = 1.6, 2.2, 1.3, 1.9
    n_top, n_bot = np.sqrt(eps_top * mu_top), np.sqrt(eps_bot * mu_bot)

    def run(tetm, e, m, m_top, m_bot):
        solver = wavebit.rcwa(backend='numpy', top=n_top ** 2, bot=n_bot ** 2, theta=0.21,
                              tetm=tetm, truncation_order=(TO, 0), period=(PERIOD, PERIOD),
                              wavelength=WL, thickness=[130.0, 220.0],
                              structure=np.ones((2, 1, 32)) * 2.25)
        # Reverse the layer axis of every epsilon and mu slot together.
        return solver.solve_conv(*(slot[::-1] for slot in e), mu=(*(slot[::-1] for slot in m), None, None), mu_top=m_top, mu_bot=m_bot,
                                 eps_top=n_top ** 2 / m_top, eps_bot=n_bot ** 2 / m_bot)

    te = run('te', eps_slots, mu_slots, mu_top, mu_bot)
    tm_traded = run('tm', mu_slots, eps_slots, eps_top, eps_bot)
    for name in ('R', 'T'):
        assert np.allclose(np.asarray(getattr(te.incident, name)),
                           np.asarray(getattr(tm_traded.incident, name)),
                           rtol=0, atol=1e-13)

    # The control. Without it this test is passed by any solver for which TE and TM happen to
    # agree, including one that drops the permeability from both.
    tm_untraded = run('tm', eps_slots, mu_slots, mu_top, mu_bot)
    assert np.max(np.abs(np.asarray(te.incident.R)
                         - np.asarray(tm_untraded.incident.R))) > 1e-4


@pytest.mark.parametrize('theta', [0.21, -0.21])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_duality_carries_the_field_with_the_traded_admittance(side, theta):
    """The same trade, on the reconstructed field: ``TM = -+(n/mu)_B * dual(TE)``, entry for entry.

    ``(E, H) -> (H, -E)`` maps the TE solution onto a field of the traded scene, so it must be
    that scene's TM solution up to the scalar that says how each was lit. Both are lit by a
    unit incident **E**. The dual of TE's unit-E wave carries ``E' = H = h_A k_hat x y_hat``,
    with ``h_A = n/mu`` of the original incidence medium; the traded medium's ``h_B = n/eps``
    (its ``mu`` is the original ``eps``) is ``1/h_A``. So TM equals ``h_B`` times the dual,
    signed by how ``k_hat x y_hat`` sits against the incident ``p_hat``: ``p_hat = s_hat x k_hat``
    lit from above, the opposite, and ``k_hat x s_hat`` lit from below, the same
    (``docs/conventions.md``). A closed form for the one scalar, with ``mu != 1`` on both sides,
    and nothing fitted.

    It is what pins the pure TM input at ``n/mu`` of the incidence medium: a TM state rescaled
    by ``n`` instead is off by ``mu`` here (1.3 above, 1.9 below), and one never rescaled --
    the unit-H input this path had until 2026-09-27 -- is off by ``h_B`` itself. Both angle
    signs, because the incident wave turns over with ``kx`` on the pure 1D path, TE and TM
    together.
    """
    eps = (_arbitrary(11, 0.20), _arbitrary(12, 0.25))
    mu = (_arbitrary(13, 0.10), _arbitrary(14, 0.30))
    eps_slots = (eps[0], eps[1], _inv_stack(eps[1]))
    mu_slots = (mu[0], mu[1], _inv_stack(mu[1]))
    mu_top, mu_bot, eps_top, eps_bot = 1.6, 2.2, 1.3, 1.9
    n_top, n_bot = np.sqrt(eps_top * mu_top), np.sqrt(eps_bot * mu_bot)

    def cell(tetm, e, m, m_top, m_bot):
        solver = wavebit.rcwa(backend='numpy', top=n_top ** 2, bot=n_bot ** 2, theta=theta,
                              tetm=tetm, truncation_order=(TO, 0), period=(PERIOD, PERIOD),
                              wavelength=WL, thickness=[130.0, 220.0], answers='field',
                              incidence_side=side, structure=np.ones((2, 1, 32)) * 2.25)
        result = solver.solve_conv(*(slot[::-1] for slot in e),
                                   mu=(*(slot[::-1] for slot in m), None, None),
                                   mu_top=m_top, mu_bot=m_bot,
                                   eps_top=n_top ** 2 / m_top, eps_bot=n_bot ** 2 / m_bot)
        return np.asarray(result.reconstruct_field(sample_x=7, sample_y=1, sample_z=5))

    te = cell('te', eps_slots, mu_slots, mu_top, mu_bot)
    tm = cell('tm', mu_slots, eps_slots, eps_top, eps_bot)
    dual = np.concatenate([te[..., 3:], -te[..., :3]], axis=-1)
    h_b = (n_top / eps_top) if side == 'top' else (n_bot / eps_bot)
    want = (-1.0 if side == 'top' else 1.0) * h_b * dual
    gap = np.max(np.abs(tm - want)) / np.max(np.abs(want))
    assert gap < 1e-12, (
        f'{side}-lit, theta={theta}: TM is not {"-" if side == "top" else "+"}h_B * dual(TE), '
        f'off by {gap:.2e}; best scalar {np.vdot(dual, tm) / np.vdot(dual, dual):.6f} against '
        f'h_B = {h_b:.6f}')


@pytest.mark.parametrize('theta', [THETA, -0.3])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_pure_tm_field_is_the_conical_p_field(side, theta):
    """The pure 1D TM solve and ``psi = 0`` at ``phi = 0`` light the stack with one wave.

    Lit from below, the incidence medium is ``Material(2.25, mu=1.3)``: ``n/mu = 1.316``, which
    separates the unit-E input every path takes from the unit-H one the pure TM path took until
    2026-09-27 (their ratio was ``n/mu``) and from a rescale by ``n`` alone (1.710). Lit from
    above it is air, where all three agree -- the control that the comparison itself is sound.
    At ``theta = -0.3`` the incident order sits at ``kx < 0``, so its ``p_hat`` is the turned-over
    one on both paths, which the pure path took on with the per-order frame the same day.
    The whole six-component cell, from the result, so the boundary spectra that
    ``field_result`` assembles and the interior the kernels hand over are both in it.
    """
    pure = _grating(1, theta=theta, tetm='tm', phi=0.0, incidence_side=side).solve()
    conical = _grating(1, theta=theta, psi=0.0, phi=0.0, incidence_side=side).solve()
    a = np.asarray(pure.reconstruct_field(sample_x=20, sample_y=1, sample_z=10))
    b = np.asarray(conical.reconstruct_field(sample_x=20, sample_y=1, sample_z=10))
    gap = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert gap < 1e-12, (
        f'{side}-lit: the pure TM field is not the conical p field, off by {gap:.2e}; best '
        f'scalar {np.vdot(a, b) / np.vdot(a, a):.6f}')


# ------------------------------------------------------------------------- 2. the closed form

def _slab_closed_form(eps, mu, theta, tetm):
    """Airy sum for one uniform slab between vacuum. No wavebit in it, by construction.

    The admittance is the whole content: ``kz/mu`` where the continuous amplitude is electric
    and ``kz/eps`` where it is magnetic. ``kz`` itself depends only on ``sqrt(eps*mu)``, which
    is exactly why a solver holding an index alone cannot express either case below.
    """
    k0 = 2 * np.pi / WL
    kx = np.sin(theta)
    kz_0 = np.sqrt(1 - kx ** 2 + 0j)
    kz_1 = np.sqrt(eps * mu - kx ** 2 + 0j)
    if tetm == 'te':
        y_0, y_1 = kz_0 / 1.0, kz_1 / mu
    else:
        y_0, y_1 = kz_0 / 1.0, kz_1 / eps
    r01 = (y_0 - y_1) / (y_0 + y_1)
    r10 = -r01
    t01, t10 = 2 * y_0 / (y_0 + y_1), 2 * y_1 / (y_1 + y_0)
    phase = np.exp(1j * k0 * kz_1 * THICK)
    r = r01 + t01 * t10 * r10 * phase ** 2 / (1 - r10 ** 2 * phase ** 2)
    return float(abs(r) ** 2)


# (eps, mu). The last two are the point of the list: `2.0/2.0` has the vacuum *impedance* with
# a non-vacuum index, and `0.5/2.0` the vacuum *index* with a non-vacuum impedance. A solver
# carrying only `n` returns 0.111 for the first (there is no interface) and 0.000 for the
# second (it sees no interface at all). The true answers are 0 and 0.346.
SLABS = [(4.0, 1.0), (4.0, 2.5), (6.25, 0.4), (2.0, 2.0), (0.5, 2.0)]


@pytest.mark.parametrize('eps,mu', SLABS, ids=[f'eps{e}_mu{m}' for e, m in SLABS])
@pytest.mark.parametrize('theta', [0.0, 0.4], ids=['normal', 'oblique'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_uniform_magnetic_slab_matches_its_closed_form(eps, mu, theta, tetm):
    """A uniform slab is exactly solvable, and the pure 1D path must land on it.

    Absolute, not relative to a fitted scale: the incident amplitude is unit and the
    reflectance is compared as a number. Measured worst case over the twenty rows, 3.3e-16.
    """
    solver = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=theta, phi=0.0, tetm=tetm,
                          truncation_order=(2, 0), period=(PERIOD, PERIOD), wavelength=WL,
                          thickness=[THICK], structure=np.ones((1, 1, 16)) * eps,
                          mu=np.ones((1, 1, 16)) * mu)
    got = float(np.sum(solver.solve().incident.R))
    # After the solve: routing is assigned during it, and the comparison below is only against
    # the pure 1D operator if that is what actually ran.
    assert solver.grating_type == 0
    assert got == pytest.approx(_slab_closed_form(eps, mu, theta, tetm), abs=1e-13)


def test_the_two_cases_an_index_only_solver_cannot_express():
    """Pinned as *values*, because they are the reason a permeability is a separate input.

    Both slabs are invisible to a solver that reduces a material to ``n = sqrt(eps*mu)``, in
    opposite directions, and both are ordinary numbers this path now returns. Kept apart from
    the parametrized test above so that the claim is stated rather than merely covered.
    """
    def reflectance(eps, mu):
        solver = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.0, tetm='te',
                              truncation_order=(2, 0), period=(PERIOD, PERIOD),
                              wavelength=WL, thickness=[THICK],
                              structure=np.ones((1, 1, 16)) * eps,
                              mu=np.ones((1, 1, 16)) * mu)
        return float(np.sum(solver.solve().incident.R))

    # Index 2, impedance 1: matched to vacuum, so nothing reflects at normal incidence. An
    # index-only solver reflects |(1-2)/(1+2)|**2 = 0.111 off a step that is not there.
    assert reflectance(2.0, 2.0) == pytest.approx(0.0, abs=1e-15)

    # Index 1, impedance 2: the mirror image. An index-only solver sees no interface at all
    # and returns exactly 0. Both halves are asserted -- the closed form fixes the value, and
    # the bound below is the statement that the value is nowhere near what an index alone
    # gives, which a tolerance on its own would not say.
    got = reflectance(0.5, 2.0)
    assert got == pytest.approx(_slab_closed_form(0.5, 2.0, 0.0, 'te'), abs=1e-13)
    assert got > 0.3


# ------------------------------------------------------------------- 3. against the 2D path

def test_conical_magnetic_solve_agrees_with_the_2d_path():
    """A 1D grating is a 2D grating that does not vary in y, and ``mu`` cannot change that.

    The only instrument here that reaches the conical operator's *coupling* blocks — ``V_12``'s
    ``muz_conv_i`` and family two's ``muy_conv`` — which is why it runs off ``phi = 0``.

    The 2D side has had a permeability since 2026-08-14, checked against a golden frozen while
    the solver had nowhere to put one, so agreement here inherits that oracle rather than
    asserting the two halves of one new change agree with each other.
    """
    _, conical, _ = _solved(1, psi=PSI, phi=PHI)
    _, two_d, _ = _solved(3, psi=PSI, phi=PHI)
    for name in ('R', 'T'):
        assert np.allclose(np.asarray(getattr(conical.incident, name)),
                           np.asarray(getattr(two_d.incident, name)), rtol=0, atol=1e-13)


def test_pure_te_magnetic_solve_agrees_with_the_2d_path():
    """Same claim for the pure TE/TM operator, whose 2D twin is s-polarized at ``phi = 0``."""
    _, pure, _ = _solved(1, tetm='te', phi=0.0)
    _, two_d, _ = _solved(3, psi=np.pi / 2, phi=0.0)
    assert float(np.sum(pure.incident.R)) == pytest.approx(
        float(np.sum(two_d.incident.R)), abs=1e-13)
    assert float(np.sum(pure.incident.T)) == pytest.approx(
        float(np.sum(two_d.incident.T)), abs=1e-13)


# The three components the pure TE path carries, and where the 2D layout keeps each.
TE_COMPONENTS = [('E_y', 0, 1), ('H_x', 1, 3), ('H_z', 2, 5)]


@pytest.mark.parametrize('name,i_1d,i_2d', TE_COMPONENTS, ids=[c[0] for c in TE_COMPONENTS])
def test_reconstructed_magnetic_field_agrees_with_the_2d_path(name, i_1d, i_2d):
    """``H_z`` is the reason this test exists, and the other two are its control.

    The longitudinal components are the only ones a *solved* field can be wrong in while
    everything else stays exact: they are recovered algebraically at the end, each by dividing
    out the ``zz`` slot of its own tensor. TE's ``H_z = mu_zz^-1 Kx E_y`` carried the ``mu = 1``
    form until this change and was correct only because the operator refused a ``mu``.

    On the 2D path exactly this line shipped wrong for six days (2026-08-20): ``R + T`` stayed
    1, both connecting algorithms still agreed, and fmmax still matched, because none of them
    read the interior ``H_z``. Comparing against a path whose reconstruction *is* pinned from
    outside is what closes it here — and ``E_y`` / ``H_x`` agreeing in the same run is what
    says a failure is localized to the algebraic component rather than to the whole solve.

    ``mu_zz`` is patterned, so the division is a matrix solve against a convolution and not a
    scale; a scalar would commute and could not tell an operand-order error from a correct one.
    """
    _, _, pure = _solved(1, tetm='te', phi=0.0)
    _, _, two_d = _solved(3, psi=np.pi / 2, phi=0.0)
    assert np.allclose(pure[..., i_1d], two_d[..., i_2d], rtol=0, atol=1e-11)


# ------------------------------------------------------- 4. the two connecting algorithms

@pytest.mark.parametrize('label,kw', [('te', dict(tetm='te', phi=0.0)),
                                      ('conical', dict(psi=PSI, phi=PHI))],
                         ids=['te', 'conical'])
def test_both_connecting_algorithms_carry_the_permeability(label, kw):
    """ETM and SMM must answer one question, on the field as well as the efficiencies.

    The field half is the part that is easy to leave out and impossible to see without: the
    per-layer ``muz_conv_i`` that ``H_z`` divides by is state the two algorithms assemble by
    different routes — the ETM appends it as it recurses, the SMM recovers it after the
    cascade — so a slot dropped from one leaves ``R`` and ``T`` identical under both. A
    mutation that dropped it from the SMM side survived an efficiency-only comparison and dies
    against this one.
    """
    out = []
    for algorithm in ('ETM', 'SMM'):
        solver = _grating(1, **kw)
        solver.layer_algorithm = algorithm
        efficiency = np.asarray(solver.solve().incident.T)
        cell = np.asarray(solver.reconstruct_field(sample_x=20, sample_y=1, sample_z=10))
        out.append((efficiency, cell[0] if cell.ndim == 5 else cell))
    assert np.allclose(out[0][0], out[1][0], rtol=0, atol=1e-12)
    assert np.allclose(out[0][1], out[1][1], rtol=0, atol=1e-11)


# ------------------------------------------------------------------------------- the refusal

def test_an_off_diagonal_permeability_is_refused_on_the_1d_paths():
    """Three slots work here and five do not, and the last two are refused rather than dropped.

    The 1D operators have no off-diagonal position for ``mu_xy`` / ``mu_yx``, for the same
    reason they have none for ``eps_xy`` / ``eps_yx``. A dropped off-diagonal would return a
    plausible *uniaxial* answer for a gyromagnetic layer — finite, energy-conserving and wrong.

    Absence is ``None`` and not a zero matrix: the question asked is structural ("was a slot
    named?"), never a value comparison, because under ``jax.jit`` a named ``mu`` is a tracer
    and ``!= 0`` on one is a traced boolean no ``if`` can take.
    """
    solver = wavebit.rcwa(backend='numpy', top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                          truncation_order=(3, 0), period=(PERIOD, PERIOD), wavelength=WL,
                          thickness=[THICK], structure=np.ones((1, 1, 8)) * 2.25)
    eps = [np.stack([np.eye(7, dtype=complex) * v]) for v in (2.6, 2.6, 1 / 2.6)]
    diagonal = tuple(np.stack([np.eye(7, dtype=complex) * v])
                     for v in (1.7, 1.7, 1 / 1.7))
    with pytest.raises(NotImplementedError, match='off-diagonal'):
        solver.solve_conv(*eps, mu=(*diagonal, *(np.zeros((1, 7, 7), dtype=complex),) * 2))
    # The same three slots, with absence spelled as the messages ask, solve.
    assert solver.solve_conv(*eps, mu=(*diagonal, None, None)) is not None
