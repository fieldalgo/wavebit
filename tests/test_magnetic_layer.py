"""Permeability on the 2D path — what holds without an oracle.

The external check is ``tests/references/test_fmmax_magnetic_golden.py``, whose magnetic rows
were frozen before this code existed. This file is the other half: theorems and identities that
would hold even if that golden were wrong.

The one that carries the most weight is the duality. Maxwell under
``(E, H, eps, mu) -> (H, -E, mu, eps)`` maps onto itself, so a slab with its two tensors
exchanged is the same problem with the polarizations traded. That reaches **only** the ``mu``
path — nothing about it can be satisfied by a solver that quietly ignores permeability, and
nothing about it depends on fmmax.
"""
import numpy as np
import pytest

import wavebit

WL, PERIOD, THICK = 700.0, 400.0, 210.0
I1 = np.eye(1, dtype=np.complex128)


def tensor(xx, yy, zz, xy=0.0, yx=0.0):
    return np.array([[xx, xy, 0.0], [yx, yy, 0.0], [0.0, 0.0, zz]], dtype=complex)


ONE = tensor(1.0, 1.0, 1.0)

# (label, eps, mu). Chosen so each one reaches a different part of the operator: scalar touches
# only the diagonal, uniaxial only `muz_i`, biaxial breaks the x/y symmetry, and gyromagnetic is
# the only one that reaches the off-diagonal slots.
CASES = [
    ('scalar_mu', tensor(2.6, 2.6, 2.6), tensor(1.7, 1.7, 1.7)),
    ('uniaxial_mu', tensor(2.6, 2.6, 2.6), tensor(1.4, 1.4, 2.2)),
    ('biaxial_both', tensor(2.3, 3.0, 2.6), tensor(1.3, 1.8, 1.5)),
    ('gyromagnetic', tensor(2.6, 2.6, 2.6), tensor(1.6, 1.6, 1.6, 0.40j, -0.40j)),
]
IDS = [c[0] for c in CASES]


def slots(t):
    """A uniform tensor as the five stacks — the same convention for ``eps`` and for ``mu``."""
    t = np.asarray(t, dtype=complex)
    return [np.stack([t[0, 0] * I1]), np.stack([t[1, 1] * I1]),
            np.stack([np.linalg.inv(t[2, 2] * I1)]),
            np.stack([-t[0, 1] * I1]), np.stack([-t[1, 0] * I1])]


def solve(eps, mu=None, theta=0.0, algo='ETM', backend=0, fto=(0, 0), period=None):
    mee = wavebit.rcwa(backend=backend, top=1.0, bot=1.0, theta=np.deg2rad(theta),
                       phi=0.0, psi=0.0, fto=fto, period=period or (PERIOD, PERIOD),
                       wavelength=WL, thickness=[THICK], connecting_algo=algo)
    res = mee.solve_for_conv(WL, *slots(eps), mu=None if mu is None else tuple(slots(mu)))
    te, tm = res.res_te_inc, res.res_tm_inc
    f = lambda v: float(np.asarray(v.detach() if hasattr(v, 'detach') else v).sum())
    return np.array([f(te.R), f(te.T), f(tm.R), f(tm.T)])


# ---------------------------------------------------------------------------- the duality

@pytest.mark.parametrize('theta', [0.0, 39.0, 62.0])
@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
def test_exchanging_eps_and_mu_trades_s_and_p(name, eps, mu, theta):
    """``R_s(eps, mu) == R_p(mu, eps)``, exactly, at any incidence.

    A theorem, so it needs no reference solver, and it is the sharpest statement available
    about *where* ``mu`` sits: the duality only closes if the transverse permeability lands in
    ``Omega_L`` and ``mu_zz`` in ``Omega_R``, cross-paired (``B`` takes ``mu_yy``, ``D`` takes
    ``mu_xx``). Put ``mu`` anywhere else and this fails while energy conservation, reciprocity
    and a plausible-looking R/T all survive.

    Stated as a *trade* rather than as invariance on purpose. Invariance is what one expects at
    normal incidence and it is wrong there too: a biaxial in-plane tensor still distinguishes x
    from y, so the swap moves the answer by 4.7e-2 while trading it exactly.
    """
    straight = solve(eps, mu, theta=theta)
    swapped = solve(mu, eps, theta=theta)
    traded = np.array([swapped[2], swapped[3], swapped[0], swapped[1]])
    assert np.abs(straight - traded).max() < 1e-12


def test_the_duality_test_is_not_vacuous():
    """It would pass trivially if the swap changed nothing, so check that it does.

    ``scalar_mu`` at normal incidence genuinely is invariant — that case cannot tell the two
    apart and is excluded rather than weakened.
    """
    moved = 0
    for name, eps, mu in CASES:
        if np.abs(solve(eps, mu, theta=39.0) - solve(mu, eps, theta=39.0)).max() > 1e-3:
            moved += 1
    assert moved == len(CASES), f'only {moved} of {len(CASES)} cases respond to the swap'


# --------------------------------------------------------------- the non-magnetic path

@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_a_unit_permeability_is_the_non_magnetic_answer(name, eps, mu, algo):
    """Handing over an explicit ``mu = 1`` must agree with handing over nothing.

    Close rather than bit-identical, deliberately: ``mu=None`` keeps ``Omega2_LR``'s
    hand-expanded form and an explicit ``mu`` multiplies ``Omega_L @ Omega_R`` out. They are the
    same operator by different arithmetic, so they agree to rounding and not below it. The
    branch exists because the expanded form is *only* valid at ``mu = 1`` — see
    :func:`~wavebit._core.solve_2d.layer_modes_2d` — and keeping it means a non-magnetic solve
    is untouched by this feature, which the ``__mu1`` golden rows check against numbers that
    predate it.
    """
    assert np.abs(solve(eps, ONE, theta=39.0, algo=algo)
                  - solve(eps, None, theta=39.0, algo=algo)).max() < 1e-12


@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
def test_the_two_connecting_algorithms_agree_with_mu(name, eps, mu):
    """``mu`` is a property of a layer; ETM and SMM only differ in how layers are joined.

    ``smm_2d`` reaches ``layer_modes_2d`` by its own route, so it could have been left behind
    without anything failing — a stack would simply have come back non-magnetic under one
    algorithm and magnetic under the other.
    """
    assert np.abs(solve(eps, mu, theta=39.0, algo='ETM')
                  - solve(eps, mu, theta=39.0, algo='SMM')).max() < 1e-10


@pytest.mark.parametrize('backend, name', [(1, 'jax'), (2, 'torch')])
def test_the_other_backends_carry_mu_too(backend, name):
    """Same answer under jax and torch as under numpy."""
    pytest.importorskip(name)
    lift = {1: lambda a: __import__('jax.numpy', fromlist=['x']).asarray(a),
            2: lambda a: __import__('torch').as_tensor(a)}[backend]

    eps, mu = CASES[3][1], CASES[3][2]                # the gyromagnetic one
    want = solve(eps, mu, theta=39.0)

    mee = wavebit.rcwa(backend=backend, top=1.0, bot=1.0, theta=np.deg2rad(39.0), phi=0.0,
                       psi=0.0, fto=(0, 0), period=(PERIOD, PERIOD), wavelength=WL,
                       thickness=[THICK])
    res = mee.solve_for_conv(WL, *[lift(s) for s in slots(eps)],
                             mu=tuple(lift(s) for s in slots(mu)))
    f = lambda v: float(np.asarray(v.detach() if hasattr(v, 'detach') else v).sum())
    got = np.array([f(res.res_te_inc.R), f(res.res_te_inc.T),
                    f(res.res_tm_inc.R), f(res.res_tm_inc.T)])
    assert np.abs(got - want).max() < 1e-12


# ------------------------------------------------------------------ physics that must hold

@pytest.mark.parametrize('name, eps, mu', CASES, ids=IDS)
@pytest.mark.parametrize('theta', [0.0, 55.0])
def test_a_hermitian_pair_is_lossless(name, eps, mu, theta):
    """Hermitian in **both** tensors is what makes a magnetic medium lossless.

    Worth stating separately from the permittivity case: a gyromagnetic ``mu_xy = +i g`` with
    ``mu_yx = -i g`` is Hermitian and lossless exactly as its electric twin is, and the first
    version of this operator broke it — ``R + T`` came out 1.035, energy created out of the
    permeability.
    """
    for t in (eps, mu):
        assert np.abs(t - t.conj().T).max() < 1e-12, 'this case was meant to be lossless'
    r_s, t_s, r_p, t_p = solve(eps, mu, theta=theta)
    assert r_s + t_s == pytest.approx(1.0, abs=1e-10)
    assert r_p + t_p == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------------------- the refusals

def test_mu_on_a_one_dimensional_structure_is_refused():
    """The 1D and conical operators carry no permeability slot, so ``mu`` there is a lie.

    Refusing beats ignoring: a dropped ``mu`` returns a finite, energy-conserving,
    entirely plausible non-magnetic answer.
    """
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       fto=(3, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                       structure=np.ones((1, 1, 8)) * 2.25)
    e = [np.stack([np.eye(7, dtype=complex) * v]) for v in (2.6, 2.6, 1 / 2.6)]
    m = tuple(np.stack([np.eye(7, dtype=complex) * v]) for v in (1.7, 1.7, 1 / 1.7, 0.0, 0.0))
    with pytest.raises(NotImplementedError, match='2D path only'):
        mee.solve_for_conv(WL, *e, mu=m)


def test_mu_with_a_z_coupled_permittivity_is_refused():
    """A z row routes to the first-order (4ff) solve, which has no ``mu`` in it.

    The two features are independent and both implemented, which is exactly why the
    *combination* has to say so rather than silently taking whichever path it lands on.
    """
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       fto=(0, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK])
    e = slots(tensor(2.6, 2.6, 2.6))
    z_row = [np.stack([0.2j * I1])] + [np.stack([0.0 * I1])] * 3
    m = tuple(slots(tensor(1.7, 1.7, 1.7)))
    with pytest.raises(NotImplementedError, match='first-order'):
        mee.solve_for_conv(WL, *e, *z_row, mu=m)


# ------------------------------------------------------- the raster pair, checked both ways
#
# `structure` and `mu` are two fields of one raster and either setter can be the one that
# completes the pair. Only the `mu` setter used to check it, so the same mistake was refused or
# accepted purely on the order the two assignments were written in -- and the accepted spelling
# was the dangerous one: the `mu` stayed on the solver, reached no build, and `conv_solve()`
# answered bit-identically to the same scene with no `mu` at all.
#
# Parametrized over all three backends because the check is one shared method called from three
# `structure` setters, one per backend, and a copy that forgot the call would leave exactly the
# silent drop above on that backend alone -- with the other two green.

BACKENDS = [(0, 'numpy'), (1, 'jax'), (2, 'torch')]

# Two rows in y: with one, the scene routes to a 1D solve, where `mu` is refused for an
# unrelated (and correct) reason and the refusal under test would not be the one observed.
RASTER = np.array([[[2.6, 2.6, 1.0, 1.0], [2.6, 1.0, 1.0, 1.0]]])
MU_RASTER = np.array([[[1.7, 1.7, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
SHAPE_LAYER = [dict(background=1.0,
                    shapes=[dict(type='rectangle', cx=PERIOD / 4, cy=PERIOD / 4,
                                 lx=PERIOD / 4, ly=PERIOD / 4, angle=0.0, material=2.6)])]
BLOCK_LAYER = [[1.0, [['rectangle', PERIOD / 4, PERIOD / 4, PERIOD / 4, PERIOD / 4, 2.6, 0.0]]]]


def _raster_solver_with_mu(backend=0):
    return wavebit.rcwa(backend=backend, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                        fto=(2, 2), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK],
                        structure=RASTER, mu=MU_RASTER)


@pytest.mark.parametrize('backend,name', BACKENDS)
@pytest.mark.parametrize('replacement,match', [
    (SHAPE_LAYER, 'no region'),
    (BLOCK_LAYER, 'no region'),
    (np.array([[[2.6, 2.6, 2.6, 1.0, 1.0, 1.0]]]), 'sampled on one raster'),
], ids=['analytic', 'block', 'wrong_grid'])
def test_replacing_the_structure_rechecks_the_mu_beside_it(replacement, match, backend, name):
    """Assigning over the permittivity re-asks what the permeability may sit beside.

    Two ways for the swap to invalidate the pair, and both are checked because both are the
    *same* omission: a shape recipe leaves a layer-shaped ``mu`` with no region to attach to,
    and a differently-shaped raster leaves two fields that no longer sample one grid.
    """
    if backend:
        pytest.importorskip(name)
    mee = _raster_solver_with_mu(backend)
    with pytest.raises(ValueError, match=match):
        mee.structure = replacement


@pytest.mark.parametrize('backend,name', BACKENDS)
def test_a_same_shaped_raster_swap_keeps_the_mu(backend, name):
    """The re-check must not become a refusal of every reassignment.

    The legal swap -- a new permittivity raster on the same grid -- has to go through with the
    permeability still attached and still reaching the solve. Without this the cheapest way to
    pass the test above is to clear or reject ``mu`` on any ``structure`` write, which trades a
    silently dropped field for a loudly dropped one.
    """
    if backend:
        pytest.importorskip(name)
    mee = _raster_solver_with_mu(backend)
    mee.structure = np.array([[[9.0, 9.0, 1.0, 1.0], [9.0, 1.0, 1.0, 1.0]]])
    assert mee.mu is not None
    total = mee.conv_solve().R
    assert float(np.asarray(total.detach() if hasattr(total, 'detach') else total).sum()) > 0


def test_clearing_the_mu_first_makes_the_swap_legal():
    """The refusal is about the pair, not about the solver having ever been magnetic.

    Worth its own case: a check written as "this solver once had a ``mu``" would pass the test
    above while making the documented way out -- drop the permeability, then change the
    structure -- impossible.
    """
    mee = _raster_solver_with_mu()
    mee.mu = None
    mee.structure = SHAPE_LAYER
    assert mee.mu is None
    assert mee.modeling_type_assigned == 2


# ============================================================ the reconstructed field
#
# Everything above this line reads `R` and `T`. Field reconstruction is a different subsystem
# reading a different part of the solve's output, and until 2026-08-20 nothing in this file
# called it -- which is exactly why `wavebit._core.fields._field_coupled` carried
# `H_z = Kx Ey - Ky Ex`, Faraday's z row at `mu = 1`, on the one path that does take a `mu`.
# A magnetic layer's `H_z` came back scaled by `mu_zz`, and nothing else moved: `E`, the
# transverse `H`, `R`, `T`, the duality above and energy conservation were all exactly right,
# because the mode amplitudes come from a solve that does account for `mu` and only this last
# algebraic step dropped it. No theorem already in this file could have seen that.
#
# The other three reconstruction paths are covered by refusing rather than by testing:
# `field_1d` / `field_1d_conical` carry the same `mu = 1` form and are unreachable
# (`test_mu_on_a_one_dimensional_structure_is_refused`), and so is `field_2d_4ff`
# (`test_mu_on_the_first_order_path_is_refused_whichever_way_it_got_there`).

FIELD_WL = 700.0

# The three incidence columns a coupled solve carries are (psi, TE, TM); these select the
# second and third. TE is E along the s direction -- the only linear polarization with an
# `H_z` at all, which is why every case below drives with it.
TE_INC, TM_INC = (False, True, False), (False, False, True)


def _kz_mu(eps, mu, kx):
    """Longitudinal wavenumber over ``k0`` in a medium with both a permittivity and a mu.

    ``eps*mu`` and not ``eps``: the index of a magnetic medium is ``sqrt(eps*mu)``. The branch
    is this project's standing convention, ``Im(kz) >= 0`` for the ``e^{-iwt}`` clock.
    """
    v = np.sqrt(complex(eps) * complex(mu) - kx ** 2)
    return v if v.imag >= 0 else -v


def _closed_form_s_field(eps, mu, thick, eps_bot, mu_bot, theta, z_abs):
    """``(Ey, Hx, Hz)`` inside a uniform magnetic stack, s-polarized. NO WAVEBIT IN HERE.

    The oracle, and the reason this file can arbitrate at all. Every golden under
    ``tests/data`` is regenerated by a script that imports wavebit, so a field golden captured
    while ``H_z`` was missing its ``mu_zz`` would have frozen the wrong number and agreed with
    it forever. A characteristic-matrix recursion cannot be co-migrated -- the same argument
    ``tests/test_material_convention_anchor.py`` makes for ``R``, made one level deeper.

    Vacuum above, unit incident amplitude, ``z_abs`` measured from the top of the stack -- the
    same frame the reconstruction returns, and the normalisation it uses, so the comparison is
    **absolute**: no fitted scale, no fitted phase, no ratio taken to make two curves meet.

    Three ingredients, and only the third is what this file is about:

    * the s-channel admittance is ``kz/mu``, not ``kz`` -- this is what a magnetic layer does
      to the *boundary* conditions, and it already had to be right for ``R`` to be right;
    * the layer's ``kz`` is ``sqrt(eps*mu - kx^2)`` -- what it does to propagation;
    * ``H_z = kx Ey / mu_zz`` -- Faraday's z row, pointwise here because the layer is uniform,
      a convolution against ``mu_zz^-1`` in the solver. **This is the term under test.** It is
      algebraic rather than propagated, which is why dropping it leaves a field that still
      satisfies both interface conditions and still carries the right power.
    """
    k0 = 2 * np.pi / FIELD_WL
    kx = np.sin(theta)                                  # vacuum above: n_top = 1
    kz_l = [_kz_mu(eps[j], mu[j], kx) for j in range(len(eps))]
    eta_l = [kz_l[j] / mu[j] for j in range(len(eps))]
    eta_t = _kz_mu(1.0, 1.0, kx) / 1.0
    eta_b = _kz_mu(eps_bot, mu_bot, kx) / mu_bot

    def M(j, d):
        """Transfer of ``(Ey, Hx)`` down a depth ``d`` of layer ``j``."""
        delta = k0 * kz_l[j] * d
        c, s = np.cos(delta), np.sin(delta)
        return np.array([[c, -1j * s / eta_l[j]], [-1j * eta_l[j] * s, c]], dtype=complex)

    # Bottom face: only a downward wave, so the state there is `t * (1, -eta_b)`. Pull it back
    # to the top through the whole stack, then match the top face -- `(1 + r, -eta_t(1 - r))`
    # -- which fixes `t`, and with it every layer's state.
    down = np.eye(2, dtype=complex)
    for j in range(len(eps)):
        down = M(j, thick[j]) @ down
    p0, p1 = np.linalg.solve(down, np.array([1.0, -eta_b], dtype=complex))
    t = 2 * eta_t / (eta_t * p0 - p1)
    state = np.array([p0, p1], dtype=complex) * t       # (Ey, Hx) at the top of the stack

    tops, s_j = [], state
    for j in range(len(eps)):
        tops.append(s_j)
        s_j = M(j, thick[j]) @ s_j

    edges = np.concatenate([[0.0], np.cumsum(thick)])
    Ey = np.zeros(len(z_abs), dtype=complex)
    Hx = np.zeros(len(z_abs), dtype=complex)
    Hz = np.zeros(len(z_abs), dtype=complex)
    for i, z in enumerate(z_abs):
        j = max(0, min(int(np.searchsorted(edges, z, side='right') - 1), len(eps) - 1))
        Ey[i], Hx[i] = M(j, z - edges[j]) @ tops[j]
        Hz[i] = kx * Ey[i] / mu[j]                      # <- the term under test
    return Ey, Hx, Hz


def _uniform_cell(eps, mu, thick, eps_bot, mu_bot, theta, z_abs, algo='ETM'):
    """The same stack through the public front door, sampled at the same absolute depths.

    A ``2 x 2`` raster per layer rather than ``1 x 1``: with one row in ``y`` the scene routes
    to a 1D solve, where ``mu`` is refused (and rightly). Every pixel carries the same value,
    so the *scene* is a uniform slab while the *path* is the 2D one that takes a permeability
    -- which is what makes a closed form applicable to the path under test.

    ``fto=(0, 0)``: only the zeroth order exists, so the solver is being asked the multilayer
    question the recursion above answers, and any disagreement is about ``mu`` rather than
    about truncation.
    """
    tile = lambda vals: np.array([[[v] * 2] * 2 for v in vals], dtype=complex)
    # A magnetic substrate is a `Material`, not a number: it enters through `mu_bot` rather
    # than through the layer slots, and it reaches the interior by changing the s admittance
    # `kz/mu` at the bottom interface -- so the standing wave above it moves.
    bot = eps_bot if mu_bot == 1.0 else wavebit.Material(eps_bot, mu=mu_bot)
    solver = wavebit.rcwa(backend=0, top=1.0, bot=bot, theta=theta, phi=0.0,
                          fto=(0, 0), period=(PERIOD, PERIOD), wavelength=FIELD_WL,
                          thickness=list(thick), connecting_algo=algo, answers='field')
    solver.structure = tile(eps)
    solver.mu = None if all(m == 1.0 for m in mu) else tile(mu)
    solver.conv_solve()
    # `(False, True, False)` is the TE-incidence column: E along the s direction, which is the
    # polarization the closed form is written for and the only one with an `H_z` at all.
    cell = np.asarray(solver.calculate_field(res_x=1, res_y=1,
                                            res_z=np.asarray(z_abs, dtype=float),
                                            set_field_input=TE_INC))
    return cell[0, :, 0, 0, :]                          # (nz, 6) = Ex Ey Ez Hx Hy Hz


# (label, eps per layer, mu per layer, thickness per layer, eps_bot, mu_bot, theta degrees).
# `mu_zz` differs from 1 by a different amount and in a different way in each magnetic row --
# above one, below one, complex, and per-layer -- so a fix that happened to divide by the
# wrong layer's value is separated from one that divides by the right one. The last row makes
# the *substrate* magnetic as well, which reaches `H_z` only indirectly: it moves the interior
# standing wave through the boundary admittance without appearing in Faraday's z row at all.
FIELD_CASES = [
    ('non-magnetic control', [6.0], [1.0], [180.0], 2.25, 1.0, 35.0),
    ('scalar mu, oblique', [6.0], [2.4], [180.0], 2.25, 1.0, 35.0),
    ('mu below one', [6.0], [0.55], [180.0], 2.25, 1.0, 48.0),
    ('two magnetic layers', [6.0, 2.0], [2.4, 1.6], [180.0, 90.0], 1.0, 1.0, 41.0),
    ('lossy magnetic', [4.0 + 0.8j], [1.9 - 0.3j], [150.0], 2.25, 1.0, 28.0),
    ('magnetic substrate too', [3.0], [1.5], [120.0], 2.25, 1.8, 52.0),
]
FIELD_IDS = [c[0] for c in FIELD_CASES]


def _sample_depths(thick):
    """Interior depths, kept off the interfaces where ``E_z`` is two-valued by choice."""
    return np.linspace(3.0, float(np.sum(thick)) - 3.0, 9)


@pytest.mark.parametrize('name,eps,mu,thick,eps_bot,mu_bot,theta_deg', FIELD_CASES,
                         ids=FIELD_IDS)
def test_a_uniform_magnetic_slab_reconstructs_the_closed_form_interior_field(
        name, eps, mu, thick, eps_bot, mu_bot, theta_deg):
    """The whole s-polarized triple, absolutely, against a recursion that never imports us.

    ``Ey`` and ``Hx`` were already right before this file existed -- they are what the solve
    hands over -- and they are asserted anyway, because they are what says a failing ``Hz`` is
    a reconstruction bug rather than a solve that mis-handled the permeability.
    """
    theta = np.deg2rad(theta_deg)
    z = _sample_depths(thick)
    got = _uniform_cell(eps, mu, thick, eps_bot, mu_bot, theta, z)
    want_ey, want_hx, want_hz = _closed_form_s_field(eps, mu, thick, eps_bot, mu_bot, theta, z)

    for label, g, w in (('Ey', got[:, 1], want_ey),
                        ('Hx', got[:, 3], want_hx),
                        ('Hz', got[:, 5], want_hz)):
        scale = float(np.abs(w).max())
        assert scale > 1e-3, (label, 'nothing to compare -- this case is degenerate')
        assert np.abs(g - w).max() < 1e-11 * scale, (label, np.abs(g - w).max() / scale)

    # The components a pure s incidence has none of. Not decoration: reading the wrong column
    # of the cell would satisfy everything above while comparing the wrong polarization.
    cell_scale = float(np.abs(got).max())
    for label, idx in (('Ex', 0), ('Ez', 2), ('Hy', 4)):
        assert np.abs(got[:, idx]).max() < 1e-11 * cell_scale, label


@pytest.mark.parametrize('name,eps,mu,thick,eps_bot,mu_bot,theta_deg', FIELD_CASES[1:],
                         ids=FIELD_IDS[1:])
def test_the_closed_form_separates_a_magnetic_hz_from_a_non_magnetic_one(
        name, eps, mu, thick, eps_bot, mu_bot, theta_deg):
    """The bug this file was blind to must be far outside the tolerance above, not near it.

    ``mu_zz`` is uniform per layer here, so the shipped-and-wrong answer is exactly
    ``mu_zz * Hz`` and can be written down rather than measured. If that were within reach of
    ``1e-11`` the test above would be pinning arithmetic noise and not the physics.
    """
    theta = np.deg2rad(theta_deg)
    z = _sample_depths(thick)
    _, _, want_hz = _closed_form_s_field(eps, mu, thick, eps_bot, mu_bot, theta, z)
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    owner = np.clip(np.searchsorted(edges, z, side='right') - 1, 0, len(eps) - 1)
    dropped = want_hz * np.array([mu[j] for j in owner])        # the `mu = 1` spelling

    scale = float(np.abs(want_hz).max())
    assert scale > 1e-3, 'this case has no Hz to get wrong'
    assert np.abs(dropped - want_hz).max() > 0.2 * scale


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_both_connecting_algorithms_reconstruct_the_magnetic_field(algo):
    """``muz_i`` reaches field reconstruction by two independent routes, one per algorithm.

    The SMM builds its own per-layer state (``wavebit._core.scattering._smm_sp``) rather than
    reusing the ETM's, so the slot had to be added twice and could have been added once --
    leaving a stack whose ``H_z`` was magnetic under ``'ETM'`` and non-magnetic under
    ``'SMM'``, with ``R`` and ``T`` identical to 1e-10 under both.
    """
    name, eps, mu, thick, eps_bot, mu_bot, theta_deg = FIELD_CASES[1]
    theta = np.deg2rad(theta_deg)
    z = _sample_depths(thick)
    got = _uniform_cell(eps, mu, thick, eps_bot, mu_bot, theta, z, algo=algo)
    _, _, want_hz = _closed_form_s_field(eps, mu, thick, eps_bot, mu_bot, theta, z)
    assert np.abs(got[:, 5] - want_hz).max() < 1e-11 * float(np.abs(want_hz).max())


# The two layers differ in `mu_zz` *and* in pattern, which is what makes the backside case
# below able to fail: the SMM reverses its per-layer state for a substrate-side solve, and a
# `muz_i` list left out of that reversal hands each layer the other one's permeability.
TWO_LAYER_EPS = np.array([[[2.6, 2.6, 1.0, 1.0], [2.6, 1.0, 1.0, 1.0]],
                          [[1.0, 4.0, 4.0, 1.0], [1.0, 1.0, 4.0, 4.0]]])
TWO_LAYER_MU = np.array([[[1.8, 1.0, 1.0, 1.4], [1.0, 1.0, 1.4, 1.4]],
                         [[1.0, 1.0, 2.2, 2.2], [2.2, 1.0, 1.0, 1.0]]])


def _two_layer_cell(algo, side):
    solver = wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=np.deg2rad(31.0),
                          phi=np.deg2rad(17.0), fto=(2, 2), period=(PERIOD, PERIOD),
                          wavelength=FIELD_WL, thickness=[180.0, 120.0], answers='field',
                          connecting_algo=algo, incidence_side=side)
    solver.structure = TWO_LAYER_EPS
    solver.mu = TWO_LAYER_MU
    solver.conv_solve()
    return np.asarray(solver.calculate_field(res_x=5, res_y=5, res_z=4,
                                             set_field_input=TE_INC))


@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_two_algorithms_agree_on_a_magnetic_cell_from_either_side(side):
    """A stack of two *different* magnetic layers, illuminated from each side, both algorithms.

    ``incidence_side='bottom'`` is where the per-layer permeability has to travel with its
    layer through a reversal, and the two connecting algorithms reverse by different means --
    the ETM mirrors the whole scene before it solves, the SMM reverses the state it kept. A
    ``muz_i`` left behind by either reversal gives each layer the other one's ``mu_zz``, which
    is a plausible magnetic cell rather than an obviously broken one: it is still magnetic,
    still energy-conserving, and still identical in ``R`` and ``T``.
    """
    etm, smm = _two_layer_cell('ETM', side), _two_layer_cell('SMM', side)
    assert np.abs(etm - smm).max() < 1e-11 * float(np.abs(etm).max())
    assert float(np.abs(etm[..., 5]).max()) > 1e-3, 'no H_z to compare'


# ------------------------------------------------------- a patterned mu, judged by duality
#
# The closed form above reaches only a uniform layer, where `mu_zz^-1` is a scalar. On a
# patterned layer it is a convolution matrix built by the inverse rule, and getting *that*
# wrong is a different mistake from omitting it. The duality is the only oracle available:
# under `(E, H, eps, mu) -> (H, -E, mu, eps)` a scene with its two patterns exchanged is the
# same problem with the polarizations traded, so a patterned `mu_zz^-1` in `H_z` is checked
# against a patterned `eps_zz^-1` in `E_z` -- a line that predates permeability entirely and
# is pinned from outside by `tests/references/test_fmmax_field_golden.py`.

EPS_PATTERN = np.array([[[2.6, 2.6, 1.0, 1.0], [2.6, 1.0, 1.0, 1.0]]])
MU_PATTERN = np.array([[[1.8, 1.0, 1.0, 1.4], [1.0, 1.0, 1.4, 1.4]]])


def _patterned_cell(eps, mu, variant, backend=0):
    """One patterned magnetic cell. Vacuum half-spaces, which are self-dual."""
    solver = wavebit.rcwa(backend=backend, top=1.0, bot=1.0, theta=np.deg2rad(37.0),
                          phi=np.deg2rad(23.0), fto=(2, 2), period=(PERIOD, PERIOD),
                          wavelength=FIELD_WL, thickness=[THICK], answers='field')
    solver.structure = eps if backend == 0 else _lift(backend, eps)
    solver.mu = mu if backend == 0 else _lift(backend, mu)
    solver.conv_solve()
    cell = solver.calculate_field(res_x=5, res_y=5, res_z=4, set_field_input=variant)
    return np.asarray(cell.detach() if hasattr(cell, 'detach') else cell)[0]


def _lift(backend, a):
    return {1: lambda x: __import__('jax.numpy', fromlist=['x']).asarray(x),
            2: lambda x: __import__('torch').as_tensor(x)}[backend](a)


def test_a_patterned_magnetic_cell_obeys_the_field_level_duality():
    """``E(eps=A, mu=B) == H(eps=B, mu=A)`` and ``H(eps=A, mu=B) == -E(eps=B, mu=A)``.

    All six components, with **no** fitted constant -- the two cells agree entry for entry,
    which is itself the result: the s and p incidence columns are normalised at equal flux, and
    the duality maps one onto the other with unit scale only because that normalisation carries
    the ``mu`` (see ``incident_sp_amplitudes``).

    Off-normal in *both* angles. At ``phi = 0`` the swap leaves ``Ky`` out of ``H_z``, so the
    half of the term that a patterned ``mu`` most easily breaks never gets exercised.
    """
    straight = _patterned_cell(EPS_PATTERN, MU_PATTERN, TE_INC)
    swapped = _patterned_cell(MU_PATTERN, EPS_PATTERN, TM_INC)
    scale = float(np.abs(straight).max())

    assert np.abs(swapped[..., :3] - straight[..., 3:]).max() < 1e-10 * scale     # E' = H
    assert np.abs(swapped[..., 3:] + straight[..., :3]).max() < 1e-10 * scale     # H' = -E


def test_the_field_duality_is_not_vacuous_on_the_longitudinal_components():
    """The swap has to move ``H_z`` by far more than the tolerance that then closes on it.

    Without this, a reconstruction returning ``H_z = 0`` on both scenes would pass the duality
    above -- and so would one that dropped ``mu_zz`` if the two patterns happened to be close
    enough for the error to hide.
    """
    a = _patterned_cell(EPS_PATTERN, MU_PATTERN, TE_INC)
    b = _patterned_cell(MU_PATTERN, EPS_PATTERN, TE_INC)      # same incidence, so not a trade
    for idx, label in ((2, 'Ez'), (5, 'Hz')):
        assert float(np.abs(a[..., idx]).max()) > 1e-3, f'{label} is identically zero'
        moved = np.abs(a[..., idx] - b[..., idx]).max() / np.abs(a[..., idx]).max()
        assert moved > 1e-2, (label, moved)


@pytest.mark.parametrize('backend,name', [(1, 'jax'), (2, 'torch')])
def test_the_other_backends_reconstruct_the_magnetic_field_too(backend, name):
    """One implementation, three namespaces, and ``muz_i @ Uz`` is new arithmetic in each.

    Against numpy rather than against the closed form: what is under test here is that the
    backend dispatch reaches the same line, and numpy is already pinned absolutely above.

    Note the order inside :func:`_patterned_cell` -- the solver is constructed *before* the
    arrays are lifted, because importing the jax backend is what raises ``jax_enable_x64``,
    and an array built before that is silently ``complex64``.
    """
    pytest.importorskip(name)
    want = _patterned_cell(EPS_PATTERN, MU_PATTERN, TE_INC)
    got = _patterned_cell(EPS_PATTERN, MU_PATTERN, TE_INC, backend=backend)
    assert np.abs(got - want).max() < 1e-9 * float(np.abs(want).max())


def test_the_permeability_a_reconstructed_hz_depends_on_is_differentiable():
    """``d|H_z|^2 / d(mu raster)``, against finite differences on the same abscissa.

    The point is *which* dependence it checks. The loss reads `H_z` and nothing else, so the
    only route from the permeability raster to it that this can be measuring is the
    ``muz_i @ Uz`` added on 2026-08-20 -- plus the solve's own, which was already there. A fix
    that reached the right number by detaching ``mu``, or by reading it as a constant, would
    leave every value test above green and this one at zero.

    Central differences on one pixel rather than the whole raster: the agreement is what is
    being checked, and one abscissa is enough to separate "wrong gradient" from "no gradient".
    """
    pytest.importorskip('jax')
    import wavebit.on_jax                                # noqa: F401 -- lifts x64 first
    import jax
    import jax.numpy as jnp

    def loss(mu):
        solver = wavebit.rcwa(backend=1, top=1.0, bot=1.0, theta=np.deg2rad(37.0),
                              phi=np.deg2rad(23.0), fto=(1, 1), period=(PERIOD, PERIOD),
                              wavelength=FIELD_WL, thickness=[THICK], answers='field')
        solver.structure = jnp.asarray(EPS_PATTERN)
        solver.mu = mu
        solver.conv_solve()
        cell = solver.calculate_field(res_x=4, res_y=4, res_z=3,
                                      set_field_input=TE_INC)
        return jnp.sum(jnp.abs(cell[..., 5]) ** 2)       # H_z alone

    mu0 = jnp.asarray(MU_PATTERN)
    grad = jax.grad(loss)(mu0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 1e-6, 'H_z does not depend on mu at all'

    h, idx = 1e-6, (0, 0, 0)
    fd = (float(loss(mu0.at[idx].add(h))) - float(loss(mu0.at[idx].add(-h)))) / (2 * h)
    assert abs(float(grad[idx]) - fd) < 1e-6 * max(abs(fd), 1.0)


def test_mu_on_the_first_order_path_is_refused_whichever_way_it_got_there():
    """``formulation='4ff'`` reaches the 4ff solve with no z row, and must say *that*.

    Companion to :func:`test_mu_with_a_z_coupled_permittivity_is_refused`, which covers the
    other route in. Both refusals are what make ``fields.field_2d_4ff``'s
    ``H_z = Kx Ey - Ky Ex`` -- Faraday's z row at ``mu = 1`` -- a correct line rather than an
    unchecked assumption: that path cannot be handed a permeability, so it never needs the
    ``muz_i`` its 2ff twin carries.

    The message is asserted, not just the exception type. Naming a z-coupled permittivity that
    was never given sends the reader to look for one, and the two routes are told apart by
    nothing else.
    """
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.0, phi=0.0, psi=0.0,
                       fto=(0, 0), period=(PERIOD, PERIOD), wavelength=WL, thickness=[THICK])
    mee.formulation = '4ff'
    e = slots(tensor(2.6, 2.6, 2.6))
    m = tuple(slots(tensor(1.7, 1.7, 1.7)))
    with pytest.raises(NotImplementedError, match="formulation='4ff' was asked for"):
        mee.solve_for_conv(WL, *e, mu=m)
