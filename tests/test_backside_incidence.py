"""Backside illumination — ``incidence_side='bottom'`` — against oracles that predate it.

The contract under test (``docs/conventions.md``): ``n_top`` / ``n_bot`` / ``ucell`` order are
*scene* labels and never move; ``incidence_side`` selects the illumination only. Results are
incidence-relative — ``R`` is power back into the incidence medium, ``T`` power through the
far side, the amplitudes keep the Fresnel contract, and ``theta`` / ``phi`` are measured
against the incidence medium's normal (``n_bot``'s, for ``'bottom'``) — so user
post-processing is identical for both sides.

Three independent ways to be right, none of which shares arithmetic with the implementation
under test:

* **The reversed-stack oracle.** Illuminating a scene from below is, exactly, illuminating
  the hand-reversed scene from above — so the Airy/transfer forms from ``test_lossy_media``
  applied to ``ns[::-1]`` pin the backside amplitudes to the same 1e-12 the top side is held
  to, absorbing media included. This is the check that would catch a backside path that is
  merely energy-conserving but on the wrong stack.
* **Reciprocity.** For a reciprocal stack the flux-normalized transmittance between a mode
  pair is direction-independent — *including* absorbing interiors — while reflectance is
  not symmetric for an asymmetric stack. Asserting both halves matters: ``T`` equal alone
  would pass a solver that ignored ``incidence_side`` entirely on a symmetric case, and the
  ``R`` inequality is what proves the two sides are genuinely different solves.
* **Two implementations.** The ETM solves the z-mirrored scene; the SMM reads the other two
  blocks of the one cascade it always built. They share the modes but not the backside
  logic, so their per-order agreement (measured 1e-16, 1e-14 on field cells) checks each
  against the other.

RETICOLO's own bottom-incidence numbers pin the diffraction orders in
``tests/references/test_reticolo_golden.py`` — this file covers what the golden cannot:
amplitudes, absorbing media, fields, and the SMM.
"""
import numpy as np
import pytest

import wavebit

from test_lossy_media import analytic_r, analytic_t

WL = 500.0
AU = 0.18 + 3.0j          # gold at 500nm — strongly absorbing
SI = 4.0 + 0.05j          # weakly absorbing

# name, indices top..bottom, thicknesses. The stacks are deliberately asymmetric in z —
# a symmetric one cannot tell 'bottom' from 'top' at all.
STACKS = [
    ('asymmetric_lossless', [1.0, 1.5, 2.2, 1.8], [20.0, 35.0]),
    ('lossy_layer', [1.0, AU, 1.5], [20.0]),
    # Incidence medium absorbing (light arrives from inside gold): amplitudes are the
    # meaningful quantities there, same as for an absorbing superstrate on the top side.
    ('absorbing_incidence_medium', [1.0, SI, AU], [30.0]),
]

INCIDENCE = [(tetm, theta) for tetm in ('te', 'tm') for theta in (0.0, 0.4)]
IDS = [f'{p}_theta{t}' for p, t in INCIDENCE]

# A 1D grating stack that is asymmetric in z, with an absorbing layer, so nothing about it
# can hide a side mix-up: two patterned layers of different duty cycle and index.
_GRATING = np.array([[[3.0] * 6 + [1.0] * 6], [[2.2 + 0.3j] * 8 + [1.0] * 4]])
GRATING_KW = dict(backend=0, n_top=1.0, n_bot=2.0, fto=10, period=[0.7], wavelength=0.85,
                  ucell=_GRATING, thickness=[0.15, 0.22])

_GRATING_LOSSLESS = np.array([[[3.0] * 6 + [1.0] * 6], [[2.2] * 8 + [1.0] * 4]])


def _solver(ns, ds, theta, tetm, path, algo, side):
    """A solver on the *scene* ``ns`` (top..bottom), illuminated from ``side``."""
    layers = ns[1:-1]
    common = dict(backend=0, connecting_algo=algo, n_top=ns[0], n_bot=ns[-1], theta=theta,
                  wavelength=WL, thickness=ds, incidence_side=side)
    if path == '1d':
        return wavebit.rcwa(tetm=tetm, fto=0, period=[WL * 2],
                              ucell=np.array([[[n, n]] for n in layers]), **common)
    cell = ([[[n, n]] for n in layers] if path == 'conical'
            else [[[n, n], [n, n]] for n in layers])
    return wavebit.rcwa(phi=0.0, psi=(np.pi / 2 if tetm == 'te' else 0.0),
                          fto=[0, 0] if path == '2d' else 0,
                          period=[WL * 2, WL * 2], ucell=np.array(cell), **common)


def _amps(res, tetm):
    """The centre-order Fresnel pair ``(r, t)`` for the asked polarization."""
    r = np.asarray(res.res.r_s if tetm == 'te' else res.res.r_p).ravel()
    t = np.asarray(res.res.t_s if tetm == 'te' else res.res.t_p).ravel()
    return complex(r[r.size // 2]), complex(t[t.size // 2])


# ------------------------------------------------------ the reversed-stack amplitude oracle

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', STACKS, ids=[c[0] for c in STACKS])
def test_backside_amplitudes_match_the_reversed_stack_oracle(case, tetm, theta, path, algo):
    """``r`` / ``t`` from below equal the analytic answer for the hand-reversed stack.

    The mirror identity is exact — ``z -> -z`` is a symmetry of Maxwell's equations and a
    layer is z-invariant inside itself — so this holds to the same 1e-12 as the top-side
    amplitude tests in ``test_lossy_media``, for every path and both connecting algorithms.
    ``theta`` here is the angle in the *incidence* medium (``ns[-1]``), which is exactly what
    the reversed oracle takes as its superstrate angle: no angle conversion appears on either
    side of the assertion, which is itself a statement about the contract.
    """
    _, ns, ds = case
    want_r = analytic_r(ns[::-1], ds[::-1], WL, theta, tetm)
    want_t = analytic_t(ns[::-1], ds[::-1], WL, theta, tetm)
    res = _solver(ns, ds, theta, tetm, path, algo, 'bottom').conv_solve()
    got_r, got_t = _amps(res, tetm)
    assert abs(got_r - want_r) < 1e-12, (got_r, want_r)
    assert abs(got_t - want_t) < 1e-12, (got_t, want_t)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
def test_backside_efficiencies_conserve_energy_on_a_lossless_stack(tetm, theta, algo):
    """``R + T == 1`` from below, which pins the ``n_bot*cos(theta_bot)`` normalization.

    The amplitude oracle above cannot see a wrong incident-power factor (it never forms a
    power); this is the check that does. Lossless asymmetric stack, so the numbers differ
    from the top-side solve while their sum may not.
    """
    _, ns, ds = STACKS[0]
    res = _solver(ns, ds, theta, tetm, '1d', algo, 'bottom').conv_solve()
    assert float(np.sum(res.R) + np.sum(res.T)) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------------------ reciprocity

@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_transmittance_is_reciprocal_and_reflectance_is_not(tetm):
    """Same in-plane k both ways: ``T`` equal (absorbing interior included), ``R`` not.

    The two sides are compared at the *same* ``k_par`` — ``n_top*sin(theta_top) ==
    n_bot*sin(theta_bot)`` — because that is the pairing reciprocity actually relates; equal
    ``theta`` would compare two unrelated experiments. ``R`` differing (here by ~4e-3) is
    asserted, not tolerated: it proves the backside solve is a different solve, which the
    ``T`` equality alone would not.
    """
    ns, ds = [1.0, 1.5, SI, 2.0], [40.0, 25.0]
    kpar = 0.4
    th_t = float(np.arcsin(kpar / ns[0]))
    th_b = float(np.arcsin(kpar / ns[-1].real))

    top = _solver(ns, ds, th_t, tetm, '1d', 'ETM', 'top').conv_solve()
    bot = _solver(ns, ds, th_b, tetm, '1d', 'ETM', 'bottom').conv_solve()

    T_t, T_b = float(np.sum(top.T)), float(np.sum(bot.T))
    R_t, R_b = float(np.sum(top.R)), float(np.sum(bot.R))
    assert T_t == pytest.approx(T_b, abs=1e-12)
    assert abs(R_t - R_b) > 1e-3, 'an asymmetric stack must reflect differently per side'


@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_grating_transmittance_is_reciprocal_per_order(tetm):
    """The zeroth order transmits equally both ways at matched (reversed) in-plane k.

    Per-order, not summed: the two sides transmit into different order sets, so total ``T``
    differs legitimately (measured ~0.8 apart on this stack) and only the mode-paired
    statement is reciprocity. Holds to 1e-15 at *finite* truncation — the truncated RCWA
    operators are still reciprocal, so this is exact, not converged-to. The reversed sign on
    the bottom angle is the pairing reciprocity states (``k -> -k``); at normal incidence it
    degenerates to the plain equal-``T00`` statement, which is asserted too.
    """
    kw = dict(GRATING_KW, ucell=_GRATING_LOSSLESS)
    fto = kw['fto']

    def T00(side, theta):
        res = wavebit.rcwa(tetm=tetm, theta=theta, incidence_side=side, **kw).conv_solve()
        return float(np.asarray(res.T).ravel()[fto])

    assert T00('top', 0.0) == pytest.approx(T00('bottom', 0.0), abs=1e-12)

    th = 0.2
    th_b = float(np.arcsin(np.sin(th) * kw['n_top'] / kw['n_bot']))
    assert T00('top', th) == pytest.approx(T00('bottom', -th_b), abs=1e-12)

    res_t = wavebit.rcwa(tetm=tetm, theta=0.0, incidence_side='top', **kw).conv_solve()
    res_b = wavebit.rcwa(tetm=tetm, theta=0.0, incidence_side='bottom', **kw).conv_solve()
    dR00 = abs(float(np.asarray(res_t.R).ravel()[fto])
               - float(np.asarray(res_b.R).ravel()[fto]))
    assert dR00 > 1e-2, 'an asymmetric grating must reflect differently per side'


# ------------------------------------------------------------- the two algorithms cross-check

@pytest.mark.parametrize('tetm', ['te', 'tm'], ids=['te', 'tm'])
def test_backside_etm_and_smm_agree_per_order(tetm):
    """One cascade read backwards against one mirrored recursion, order by order.

    The two backside implementations are deliberately different (see
    ``_core/scattering.py``): the ETM re-solves the mirrored scene, the SMM reads ``S22`` /
    ``S12`` off the cascade it already had. Block-swap algebra says those coincide exactly;
    this holds the floating-point realizations to 1e-12 on a stack with diffraction, loss
    and oblique incidence in play at once.
    """
    a = wavebit.rcwa(connecting_algo='ETM', tetm=tetm, theta=0.3,
                       incidence_side='bottom', **GRATING_KW).conv_solve()
    b = wavebit.rcwa(connecting_algo='SMM', tetm=tetm, theta=0.3,
                       incidence_side='bottom', **GRATING_KW).conv_solve()
    assert np.abs(np.asarray(a.R) - np.asarray(b.R)).max() < 1e-12
    assert np.abs(np.asarray(a.T) - np.asarray(b.T)).max() < 1e-12
    r_a, r_b = (a.res.r_s, b.res.r_s) if tetm == 'te' else (a.res.r_p, b.res.r_p)
    assert np.abs(np.asarray(r_a) - np.asarray(r_b)).max() < 1e-12


def test_backside_etm_and_smm_agree_on_the_conical_path():
    a = wavebit.rcwa(connecting_algo='ETM', psi=0.7, phi=0.3, theta=0.25,
                       incidence_side='bottom', **GRATING_KW).conv_solve()
    b = wavebit.rcwa(connecting_algo='SMM', psi=0.7, phi=0.3, theta=0.25,
                       incidence_side='bottom', **GRATING_KW).conv_solve()
    assert np.abs(np.asarray(a.R) - np.asarray(b.R)).max() < 1e-12
    assert np.abs(np.asarray(a.T) - np.asarray(b.T)).max() < 1e-12


# ----------------------------------------------------------------------------------- fields

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_field_cell_comes_back_in_scene_coordinates(algo):
    """The reconstructed cell has scene z whichever side is lit.

    Pinned against the mirror identity itself: the backside cell of a scene must equal the
    top-side cell of the hand-reversed scene read upside down. The scene is z-asymmetric, so
    a missing (or double) un-flip puts the absorbing layer's field profile in the wrong half
    and the comparison fails at O(1) — and for the SMM this is also the only test of its
    backside field-state recovery, which shares no code with its efficiency read.
    """
    kw = dict(GRATING_KW, connecting_algo=algo, tetm='te', theta=0.3)
    s_b = wavebit.rcwa(incidence_side='bottom', **kw)
    _, cell_b = s_b.conv_solve_field(res_x=16, res_z=8)

    kw_rev = dict(kw, ucell=GRATING_KW['ucell'][::-1], thickness=[0.22, 0.15],
                  n_top=GRATING_KW['n_bot'], n_bot=GRATING_KW['n_top'])
    s_t = wavebit.rcwa(**kw_rev)
    _, cell_t = s_t.conv_solve_field(res_x=16, res_z=8)

    assert np.abs(np.asarray(cell_b) - np.asarray(cell_t)[::-1]).max() < 1e-12


def test_backside_field_cells_agree_between_algorithms():
    kw = dict(GRATING_KW, tetm='tm', theta=0.3)
    _, c1 = wavebit.rcwa(connecting_algo='ETM', incidence_side='bottom',
                           **kw).conv_solve_field(res_x=16, res_z=8)
    _, c2 = wavebit.rcwa(connecting_algo='SMM', incidence_side='bottom',
                           **kw).conv_solve_field(res_x=16, res_z=8)
    assert np.abs(np.asarray(c1) - np.asarray(c2)).max() < 1e-12


# ------------------------------------------------------------------- backends and validation

@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_backside_holds_on_every_backend(backend):
    """Same physics through the same single-source core, so the backends may not diverge."""
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    if backend == 1:
        import importlib
        importlib.import_module('wavebit.on_jax')          # enables x64

    ref = wavebit.rcwa(tetm='te', theta=0.3, incidence_side='bottom',
                         **GRATING_KW).conv_solve()
    got = wavebit.rcwa(tetm='te', theta=0.3, incidence_side='bottom',
                         **{**GRATING_KW, 'backend': backend}).conv_solve()
    assert np.abs(np.asarray(ref.R) - np.asarray(got.R)).max() < 1e-12
    assert np.abs(np.asarray(ref.T) - np.asarray(got.T)).max() < 1e-12


def test_an_unknown_side_is_refused_by_name():
    """Everywhere it can be said: the constructor, the property, and the core solves.

    A typo'd side must never quietly illuminate from the top — the results would be
    plausible and wrong for exactly the structures where the side matters.
    """
    with pytest.raises(ValueError, match="'top' or 'bottom'"):
        wavebit.rcwa(incidence_side='back', **GRATING_KW)

    solver = wavebit.rcwa(**GRATING_KW)
    with pytest.raises(ValueError, match="'top' or 'bottom'"):
        solver.incidence_side = 'up'

    from wavebit._core.solve_1d import solve_1d
    with pytest.raises(ValueError, match="'top' or 'bottom'"):
        solve_1d(0, 1.0, np.array([0.0]), 1.0, 1.5, 0.0, [0.1],
                 np.ones((1, 1, 1), complex), np.ones((1, 1, 1), complex),
                 np.ones((1, 1, 1), complex), incidence_side='rear')
