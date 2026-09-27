"""Backside illumination — ``incidence_side='bottom'`` — against oracles that predate it.

The contract under test (``docs/conventions.md``): ``top`` / ``bot`` / ``structure`` order are
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
  against the other. That backside logic belongs to the propagation kernels, and since
  2026-09-21 the engine sends public *top* incidence through it and public bottom incidence
  through their direct branch. The ETM/SMM comparisons below therefore check the two
  algorithms on the direct branch; the top-lit ones elsewhere
  (``tests/test_scattering_core.py``) are what reach the mirrored one now.

RETICOLO's own bottom-incidence numbers pin the diffraction orders in
``tests/references/test_reticolo_golden.py`` — this file covers what the golden cannot:
amplitudes, absorbing media, fields, and the SMM.
"""
import numpy as np
import pytest

import wavebit
from wavebit import eps_from_index

from test_lossy_media import analytic_r, analytic_t

# Same arrangement as ``test_lossy_media``: the stacks stay lists of refractive INDEX, since
# the Airy oracles imported above are written that way, and `eps_from_index` is applied where
# each one enters wavebit.

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
_GRATING = eps_from_index(np.array([
    [[2.2 + 0.3j] * 8 + [1.0] * 4],
    [[3.0] * 6 + [1.0] * 6],
]))
#: The substrate's index, kept beside its permittivity: the reciprocity angle below is a
#: Snell relation and needs the index, while the constructor takes the square.
N_SUB = 2.0
GRATING_KW = dict(backend=0, top=1.0, bot=eps_from_index(N_SUB), truncation_order=10, period=[0.7],
                  wavelength=0.85, structure=_GRATING, thickness=[0.22, 0.15])

_GRATING_LOSSLESS = eps_from_index(np.array([
    [[2.2] * 8 + [1.0] * 4],
    [[3.0] * 6 + [1.0] * 6],
]))


def _solver(ns, ds, theta, tetm, path, algo, side):
    """A solver on the *scene* ``ns`` (top..bottom), illuminated from ``side``."""
    layers = ns[1:-1][::-1]
    common = dict(backend=0, layer_algorithm=algo, top=eps_from_index(ns[0]),
                  bot=eps_from_index(ns[-1]), theta=theta,
                  wavelength=WL, thickness=ds[::-1], incidence_side=side)
    if path == '1d':
        return wavebit.rcwa(tetm=tetm, truncation_order=0, period=[WL * 2],
                              structure=eps_from_index(
                                  np.array([[[n, n]] for n in layers])), **common)
    cell = ([[[n, n]] for n in layers] if path == 'conical'
            else [[[n, n], [n, n]] for n in layers])
    return wavebit.rcwa(phi=0.0, psi=(np.pi / 2 if tetm == 'te' else 0.0),
                          truncation_order=[0, 0] if path == '2d' else 0,
                          period=[WL * 2, WL * 2],
                          structure=eps_from_index(np.array(cell)), **common)


def _amps(res, tetm):
    """The centre-order Fresnel pair ``(r, t)`` for the asked polarization."""
    r = np.asarray(res.incident.r_s if tetm == 'te' else res.incident.r_p).ravel()
    t = np.asarray(res.incident.t_s if tetm == 'te' else res.incident.t_p).ravel()
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
    res = _solver(ns, ds, theta, tetm, path, algo, 'bottom').solve()
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
    res = _solver(ns, ds, theta, tetm, '1d', algo, 'bottom').solve()
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

    top = _solver(ns, ds, th_t, tetm, '1d', 'ETM', 'top').solve()
    bot = _solver(ns, ds, th_b, tetm, '1d', 'ETM', 'bottom').solve()

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
    kw = dict(GRATING_KW, structure=_GRATING_LOSSLESS)
    truncation_order = kw['truncation_order']

    def T00(side, theta):
        res = wavebit.rcwa(tetm=tetm, theta=theta, incidence_side=side, **kw).solve()
        return float(np.asarray(res.T).ravel()[truncation_order])

    assert T00('top', 0.0) == pytest.approx(T00('bottom', 0.0), abs=1e-12)

    th = 0.2
    th_b = float(np.arcsin(np.sin(th) * 1.0 / N_SUB))     # Snell, in indices
    assert T00('top', th) == pytest.approx(T00('bottom', -th_b), abs=1e-12)

    res_t = wavebit.rcwa(tetm=tetm, theta=0.0, incidence_side='top', **kw).solve()
    res_b = wavebit.rcwa(tetm=tetm, theta=0.0, incidence_side='bottom', **kw).solve()
    dR00 = abs(float(np.asarray(res_t.R).ravel()[truncation_order])
               - float(np.asarray(res_b.R).ravel()[truncation_order]))
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

    Written when public bottom incidence was the kernels' backside. Since 2026-09-21 it is
    their direct branch -- call counters on the ETM's ``_mirrored_scene`` and the SMM's
    backside flag read 0 for these solves, and 1 for the same solves lit from the top -- so
    this now compares the two algorithms lit from the bottom on that branch; the mirrored
    recursion and the backwards read are reached by top incidence.
    """
    a = wavebit.rcwa(layer_algorithm='ETM', tetm=tetm, theta=0.3,
                       incidence_side='bottom', **GRATING_KW).solve()
    b = wavebit.rcwa(layer_algorithm='SMM', tetm=tetm, theta=0.3,
                       incidence_side='bottom', **GRATING_KW).solve()
    assert np.abs(np.asarray(a.R) - np.asarray(b.R)).max() < 1e-12
    assert np.abs(np.asarray(a.T) - np.asarray(b.T)).max() < 1e-12
    r_a, r_b = (a.incident.r_s, b.incident.r_s) if tetm == 'te' else (a.incident.r_p, b.incident.r_p)
    assert np.abs(np.asarray(r_a) - np.asarray(r_b)).max() < 1e-12


def test_backside_etm_and_smm_agree_on_the_conical_path():
    a = wavebit.rcwa(layer_algorithm='ETM', psi=0.7, phi=0.3, theta=0.25,
                       incidence_side='bottom', **GRATING_KW).solve()
    b = wavebit.rcwa(layer_algorithm='SMM', psi=0.7, phi=0.3, theta=0.25,
                       incidence_side='bottom', **GRATING_KW).solve()
    assert np.abs(np.asarray(a.R) - np.asarray(b.R)).max() < 1e-12
    assert np.abs(np.asarray(a.T) - np.asarray(b.T)).max() < 1e-12


# ----------------------------------------------------------------------------------- fields

#: The mirror relating a bottom-lit scene to its hand-reversed, top-lit twin is
#: ``diag(1, 1, -1)`` (the same one a backside solve leaves behind), and the field components do
#: not all follow the coordinates through it: ``E`` is polar and loses ``E_z``, ``H`` is axial
#: and loses ``H_x``/``H_y`` instead. The two 1D layouts fill their three slots with different
#: components, so the same rule reads as two different vectors.
MIRROR_1D = {'te': np.array([1.0, -1.0, 1.0]),      # (Ey, Hx, Hz)
             'tm': np.array([-1.0, 1.0, -1.0])}     # (Hy, Ex, Ez)


def _reversed_scene_kw(kw):
    """The same stack written upside down, to be lit from the top."""
    return dict(kw, structure=GRATING_KW['structure'][::-1], thickness=GRATING_KW['thickness'][::-1],
                top=GRATING_KW['bot'], bot=GRATING_KW['top'])


@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_field_cell_comes_back_in_scene_coordinates(algo, tetm):
    """The reconstructed cell is in the scene's frame — both axis *and* components.

    The backside cell must equal the top-side cell of the hand-reversed scene read upside down
    **and with the mirror's vector parity applied**. Both halves are needed and this test
    carried only the first until 2026-08-22, which is how a wrong frame survived: the two sides
    were then bit-identical either way, because the ETM backside path *was* the reversed-scene
    solve and the SMM recovers the same state. It pinned the bookkeeping — which slices, which
    order — and nothing about the frame. What it cannot see, the Poynting sign below can.
    (Since 2026-09-21 the arithmetic is shared from the other side: the bottom-lit solve takes
    the kernels' direct branch, and the top-lit reversed scene their mirrored one, which undoes
    the hand reversal. Measured still bit-identical, ETM and SMM, TE and TM.)

    Both polarizations, because the two 1D layouts need different sign vectors and a TM-only
    error is invisible to the ETM/SMM comparison beneath, which is sign-blind.
    """
    kw = dict(GRATING_KW, layer_algorithm=algo, tetm=tetm, theta=0.3, answers='field')
    _, cell_b = wavebit.rcwa(incidence_side='bottom', **kw).solve_field(sample_x=16, sample_z=8)
    _, cell_t = wavebit.rcwa(**_reversed_scene_kw(kw)).solve_field(sample_x=16, sample_z=8)

    mirrored = MIRROR_1D[tetm] * np.asarray(cell_t)[::-1]
    assert np.abs(np.asarray(cell_b) - mirrored).max() < 1e-12


@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('side,sign', [('top', -1.0), ('bottom', +1.0)])
def test_net_power_flows_away_from_the_lit_face(side, sign, tetm):
    """``S_z`` points into the stack from whichever face is lit — the sign the mirror decides.

    This is the assertion the mirror identity above cannot make. That one compares two cells
    that share their arithmetic, so it is satisfied by *either* frame as long as both sides use
    the same one — and the returned cell was in the mirrored frame for as long as it existed.
    Net Poynting is answerable from the cell alone, against a fact no reconstruction supplies:
    light entering from below carries power upward, so ``S_z`` in scene coordinates is positive
    (``+z`` has pointed up since 2026-09-21; before that it pointed down and the signs were the
    other way round).

    Measured before the fix: ``+0.343`` on both sides, bit-identical, where the two must differ
    in sign.
    """
    kw = dict(GRATING_KW, tetm=tetm, theta=0.3, answers='field')
    _, cell = wavebit.rcwa(incidence_side=side, **kw).solve_field(sample_x=64, sample_z=16)
    c = np.asarray(cell)
    # TE fills the three slots as (Ey, Hx, Hz) and TM as (Hy, Ex, Ez)
    Sz = (0.5 * np.real(-c[..., 0] * np.conj(c[..., 1])) if tetm == 'te'
          else 0.5 * np.real(c[..., 1] * np.conj(c[..., 0])))
    mean = float(np.mean(Sz))
    assert np.sign(mean) == sign, mean
    assert abs(mean) > 1e-2, mean                    # not a sign read off round-off


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_explicit_z_on_a_backside_solve_is_read_in_scene_coordinates(tetm):
    """An explicit ``sample_z`` is absolute in the scene, on both sides, in the caller's order.

    Two things went wrong together and this pins both. The coordinate was bucketed against the
    *mirror's* thicknesses, so it selected the wrong layer; and the axis was then reversed even
    though ``_z_plan`` already returns explicit samples in the caller's own order, so the
    request came back backwards. Against the hand-reversed scene the pair was worth O(1) at
    every requested z — 2.86 on this stack.

    The requested z straddle the internal interface deliberately: an interface belongs to the
    layer above it, and that rule has to be applied once, in scene coordinates, rather than
    re-derived in the mirror where it inverts. The interface sits at height 0.22 -- layer 0 is
    the 0.22-thick one since layers run bottom to top. When the stack was reversed into that
    order on 2026-09-21 these heights were kept as they were, which put both 0.149 and 0.151
    inside layer 0, so the check stopped straddling the interface; they are the old depths
    restated as heights, ``total - depth``, and deliberately not ascending.
    """
    kw = dict(GRATING_KW, tetm=tetm, theta=0.3, answers='field')
    total = sum(kw['thickness'])
    interface = kw['thickness'][0]
    z = np.array([0.35, 0.221, 0.219, 0.07, 0.01])
    gap = z - interface
    assert ((gap > 0) & (gap < 2e-3)).any() and ((gap < 0) & (gap > -2e-3)).any(), (
        'the samples no longer straddle the internal interface')

    def cell(solver_kw, zs):
        s = wavebit.rcwa(**solver_kw)
        s.solve()
        return np.asarray(s.reconstruct_field(sample_x=8, sample_y=1, sample_z=zs))

    got = cell(dict(kw, incidence_side='bottom'), z)
    want = MIRROR_1D[tetm] * cell(_reversed_scene_kw(kw), total - z)
    assert np.abs(got - want).max() < 1e-12, np.abs(got - want).max()

    # ...and the caller's order is the returned order, which the blind reversal also broke
    assert np.array_equal(cell(dict(kw, incidence_side='bottom'), z[::-1]), got[::-1])


def test_backside_field_cells_agree_between_algorithms():
    kw = dict(GRATING_KW, tetm='tm', theta=0.3, answers='field')
    _, c1 = wavebit.rcwa(layer_algorithm='ETM', incidence_side='bottom',
                           **kw).solve_field(sample_x=16, sample_z=8)
    _, c2 = wavebit.rcwa(layer_algorithm='SMM', incidence_side='bottom',
                           **kw).solve_field(sample_x=16, sample_z=8)
    assert np.abs(np.asarray(c1) - np.asarray(c2)).max() < 1e-12


# ------------------------------------------------------------------- backends and validation

@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_backside_holds_on_every_backend(backend):
    """Same physics through the same single-source core, so the backends may not diverge."""
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    if backend == 1:
        wavebit.enable_x64()

    ref = wavebit.rcwa(tetm='te', theta=0.3, incidence_side='bottom',
                         **GRATING_KW).solve()
    got = wavebit.rcwa(tetm='te', theta=0.3, incidence_side='bottom',
                         **{**GRATING_KW, 'backend': backend}).solve()
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
