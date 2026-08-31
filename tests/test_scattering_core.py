"""Scattering-matrix connecting algorithm (wavebit._core.scattering).

SMM has no golden file of its own on purpose: its oracle is the ETM path, which is itself
pinned to the legacy results in test_golden_legacy_parity.py. So the properties tested are

  * agreement    — SMM == ETM on every grating type, in efficiencies *and* in the s/p
                   amplitudes, for the psi incidence and both pure-TE / pure-TM variants;
  * conservation — lossless stacks keep their energy (a check ETM agreement alone would pass
                   even if both were wrong the same way, so it is worth its own assertion);
  * wiring       — layer_algorithm='SMM' reaches the core solve, and an unknown algo is
                   rejected with a useful message;
  * fields       — the near field reconstructed after an SMM solve, and the per-layer state
                   it is built from, match the ETM's, on an axis-aligned and a skewed cell;
  * portability  — same numbers on numpy / jax / torch, and gradients still flow.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core.factorization import to_conv_mat_raster_sampled
from wavebit._core.scattering import smm_1d, smm_1d_conical, smm_2d
from wavebit._core.solve_1d import solve_1d, solve_1d_conical
from wavebit._core.solve_2d import solve_2d

# ---------------------------------------------------------------------------------- cases
# name, kwargs for wavebit.rcwa (grating type follows from `structure` / phi, as in the
# solver). Every raster below is PERMITTIVITY: 9 and 4 where the pre-flip file wrote the
# indices 3 and 2, so each case is the same stack it always was.
UC_1D = np.array([[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]])
UC_1D_SUB = np.array([[[9., 9., 1., 1., 9., 1.]], [[1., 1., 1., 1., 1., 1.]]])
UC_2D = np.array([[[9., 9., 1., 1.], [1., 9., 9., 1.], [1., 1., 9., 9.], [9., 1., 1., 9.]]])

CASES_1D = [
    ('te_1layer', dict(tetm='te', structure=np.array([[[9., 9., 9., 1., 1., 1.]]]), thickness=[0.15],
                       top=4.0, bot=1.0, theta=0.0, wavelength=0.777, period=[0.77],
                       truncation_order=[6, 0])),
    ('tm_1layer', dict(tetm='tm', structure=np.array([[[9., 9., 9., 1., 1., 1.]]]), thickness=[0.15],
                       top=4.0, bot=1.0, theta=0.0, wavelength=0.777, period=[0.77],
                       truncation_order=[6, 0])),
    ('te_2layer_oblique', dict(tetm='te', structure=UC_1D, thickness=[0.15, 0.22], top=4.0,
                               bot=2.25, theta=0.3, wavelength=0.777, period=[0.77],
                               truncation_order=[5, 0])),
    ('tm_2layer_oblique', dict(tetm='tm', structure=UC_1D, thickness=[0.15, 0.22], top=4.0,
                               bot=2.25, theta=0.3, wavelength=0.777, period=[0.77],
                               truncation_order=[5, 0])),
    # bottom layer uniform and equal to the substrate: ETM short-circuits the interface,
    # SMM has no short-circuit to take — they must still land on the same answer.
    ('te_same_material', dict(tetm='te', structure=UC_1D_SUB, thickness=[0.2, 0.1], top=1.0,
                              bot=1.0, theta=0.0, wavelength=0.9, period=[0.7],
                              truncation_order=[5, 0])),
    ('tm_same_material', dict(tetm='tm', structure=UC_1D_SUB, thickness=[0.2, 0.1], top=1.0,
                              bot=1.0, theta=0.0, wavelength=0.9, period=[0.7],
                              truncation_order=[5, 0])),
    # thick, strongly evanescent: the regime where a connecting algorithm goes wrong if it
    # ever inverts a growing exponential
    ('tm_thick', dict(tetm='tm', structure=np.array([[[16., 16., 16., 16., 1., 1., 1., 1.]]]),
                      thickness=[20.0], top=1.0, bot=1.0, theta=0.0, wavelength=0.5,
                      period=[0.6], truncation_order=[12, 0])),
]

CASES_CONICAL = [
    ('con_1layer', dict(structure=np.array([[[9., 9., 9., 1., 1., 1.]]]), thickness=[0.15],
                        top=4.0, bot=1.0, theta=0.2, phi=0.3, psi=0.4, wavelength=0.777,
                        period=[0.77], truncation_order=[6, 0])),
    ('con_2layer_lossy_sub', dict(structure=UC_1D, thickness=[0.15, 0.22], top=1.0, bot=2.25,
                                  theta=0.3, phi=0.5, psi=1.1, wavelength=0.9, period=[0.7],
                                  truncation_order=[5, 0])),
    ('con_phi0', dict(structure=np.array([[[16., 16., 1., 1., 1., 1.]]]), thickness=[0.3],
                      top=2.25, bot=1.0, theta=0.25, phi=0.0, psi=0.7, wavelength=0.6,
                      period=[0.5], truncation_order=[4, 0])),
]

CASES_2D = [
    ('2d_1layer', dict(structure=UC_2D, thickness=[0.3], top=1.0, bot=2.25, theta=0.2,
                       phi=0.4, psi=0.7, wavelength=0.9, period=[0.7, 0.7], truncation_order=[3, 3])),
    # The two layers differ in *mean* permittivity, not just in pattern. Equal means used to
    # trip a ETM misdetection (see the DEGENERATE_MEAN note below); that path is gone, and
    # this stays the way it is because the case is a cross-algorithm check, not a probe of it.
    ('2d_2layer_normal', dict(structure=np.concat([UC_2D, 1 + UC_2D[:, ::-1]]),
                              thickness=[0.2, 0.3], top=1.0, bot=1.0, theta=0.0,
                              phi=0.0, psi=0.4, wavelength=0.5, period=[0.6, 0.6],
                              truncation_order=[2, 2])),
]

ALL_CASES = ([(n, k, '1d') for n, k in CASES_1D]
             + [(n, k, 'conical') for n, k in CASES_CONICAL]
             + [(n, k, '2d') for n, k in CASES_2D])

AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')
EFFICIENCIES = ('R', 'R_s', 'R_p', 'T', 'T_s', 'T_p')


def _solve(algo, kwargs):
    return wavebit.rcwa(backend=0, layer_algorithm=algo, **kwargs).solve()


# ------------------------------------------------------------------------------- agreement

@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_matches_tmm(name, kwargs, kind):
    """Same efficiencies and same s/p amplitudes, for every incidence the solve reports."""
    tmm = _solve('ETM', kwargs)
    smm = _solve('SMM', kwargs)

    for variant in ('given', 's_inc', 'p_inc'):
        a, b = getattr(tmm, variant), getattr(smm, variant)
        if a is None:                      # 1D TE/TM reports only the psi incidence
            assert b is None, variant
            continue
        for field in AMPLITUDES + EFFICIENCIES:
            x, y = np.array(getattr(a, field)), np.array(getattr(b, field))
            assert x.shape == y.shape, (variant, field, x.shape, y.shape)
            assert np.allclose(x, y, atol=1e-10, rtol=0), \
                (variant, field, np.max(np.abs(x - y)))


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_conserves_energy_when_lossless(name, kwargs, kind):
    """Every case here is lossless, so reflection + transmission must sum to 1."""
    smm = _solve('SMM', kwargs)
    total = float(np.sum(np.array(smm.R)) + np.sum(np.array(smm.T)))
    assert abs(total - 1.0) < 1e-9, total


@pytest.mark.parametrize('slot', ['eps_top', 'eps_bot'])
def test_half_space_permittivity_reaches_both_connecting_algorithms(slot):
    """``eps_top`` / ``eps_bot`` are read by ETM and by SMM, and read the same way.

    The 2D solves take each half-space as three numbers -- the index ``n``, the permeability
    ``mu`` and the permittivity ``eps`` -- of which only two are independent, since
    ``n**2 == eps*mu``. Every scene the engine builds supplies a consistent triple, and *that
    is what hides a missing reader*: while ``eps == n**2/mu`` the ETM matching's ``kz*mu/n**2``
    and the SMM's ``kz/eps`` are the same number to within rounding, so ETM could -- and for a
    while did -- declare ``eps_top``, document it, mirror it under backside incidence, and
    never read a line of it, with the whole suite green. ``eps_bot`` was read on both paths the
    whole time, which is the asymmetry that makes this worth parametrizing rather than
    asserting once.

    Breaking the consistency on purpose is what makes a reader visible. The probe hands one
    slot a permittivity far from ``n**2/mu`` and asks two things: that the answer *moves* (a
    slot nothing reads cannot move it), and that both connecting algorithms move to the same
    place. The second is the one that failed: with ``eps_top = 100`` against ``n_top = 2``, SMM
    followed the permittivity and ETM did not, so the same scene answered differently
    depending on a keyword that is supposed to select an algorithm, not a structure.

    The probe is deliberately **not** a physical scene -- an index and a permittivity that
    disagree describe no medium, so ``R + T`` leaves 1 and the solve says so. That warning is
    the honest report of the input and is not what is under test, so it is filtered rather
    than asserted; a case where it happened not to fire would prove nothing either way.
    """
    _, inputs = _core_inputs(CASES_2D[0][1])
    base = solve_2d(**inputs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavebit.UnphysicalResultWarning)
        etm = solve_2d(**inputs, **{slot: 100.0})
        smm = smm_2d(**inputs, **{slot: 100.0})

    moved = max(float(np.max(np.abs(np.array(etm[i]) - np.array(base[i])))) for i in (0, 1))
    assert moved > 1e-3, f'{slot} moved the ETM answer by only {moved:.3e}: nothing reads it'
    for i, port in enumerate(('R', 'T')):
        np.testing.assert_allclose(np.array(etm[i]), np.array(smm[i]),
                                   rtol=1e-10, atol=1e-14,
                                   err_msg=f'{slot}: ETM and SMM disagree in {port}')


# ---------------------------------------------------------------------------------- wiring

def _core_inputs(kwargs):
    """Solver-built arguments for a direct core call (same route the mixin takes)."""
    solver = wavebit.rcwa(backend=0, layer_algorithm='SMM', **kwargs)
    solver.solve()                    # assigns the grating type
    epx, epy, epz, *_ = to_conv_mat_raster_sampled(
        solver.structure, solver.truncation_order[0], solver.truncation_order[1])
    kx, ky = solver.order_kxy()
    # The core solves take an INDEX: `n_top` / `n_bot` are amplitude ratios there, and the
    # solver's read-only properties are exactly the derived roots to hand them.
    common = dict(wavelength=kwargs['wavelength'], kx=kx, n_top=solver.n_top,
                  n_bot=solver.n_bot, theta=solver.theta, thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    if solver.grating_type == 0:
        return solver, dict(common, tetm_index=solver.tetm_index)
    return solver, dict(common, ky=ky, incidence=solver._incidence_pair(), phi=solver.phi)


CORE_FN = {'1d': (smm_1d, solve_1d), 'conical': (smm_1d_conical, solve_1d_conical),
           '2d': (smm_2d, solve_2d)}


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_wiring_matches_direct_core_call(name, kwargs, kind):
    """layer_algorithm='SMM' dispatches to the core with the arguments it expects."""
    smm_fn, _ = CORE_FN[kind]
    solver, inputs = _core_inputs(kwargs)
    via_api = _solve('SMM', kwargs)
    ri, ti = smm_fn(**inputs)
    # To a tolerance, not bit for bit. The property here is that the mixin hands the core the
    # arguments it expects, and every way of getting that wrong -- kx and ky swapped, psi
    # dropped, the layers in the other order -- moves the efficiencies by far more than 1e-12.
    # Exact equality would additionally assert something this test does not own: that two
    # independent runs of the same expressions land on the same last bit. They do on every
    # machine measured here, but the two routes are two separate BLAS call sequences, and on
    # GitHub's ubuntu-latest runners this has disagreed in the low bits on one run and agreed
    # on a re-run of the same commit -- and only on the cases whose mode matrices are 26x26 or
    # larger, which points at a blocked or threaded kernel rather than at anything in this
    # repository (both results are computed in one process on one machine, so a SIMD kernel
    # chosen per CPU model cannot separate them). Where bit-identity *is* the property --
    # freezing a geometry below, the `full=` flag in test_core_full_results.py, legacy parity --
    # the tests that own it still assert it exactly, and CI pins the BLAS thread count so they
    # have a deterministic runtime to assert it on.
    #
    # atol is not zero because a 2D case reports evanescent orders as round-off-level numbers
    # (~1e-29) rather than exact zeros; comparing those on their own relative scale would test
    # the round-off, not the wiring.
    np.testing.assert_allclose(np.array(via_api.R), np.array(ri), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.array(via_api.T), np.array(ti), rtol=1e-12, atol=1e-15)


def test_unknown_connecting_algo_is_rejected():
    kwargs = CASES_1D[0][1]
    with pytest.raises(ValueError, match="'RCWA-ish'"):
        _solve('RCWA-ish', kwargs)


def _incidences_of(solver):
    """The incidences this solve actually carries.

    Read off the solved solver rather than off the case list, so the routing rule stays
    stated once -- in ``_assign_grating_type`` -- instead of being restated here. The pure 1D
    TE/TM path carries one: ``tetm`` declared the other degree of freedom away, and asking it
    for ``'s'`` / ``'p'`` is refused rather than answered.
    """
    return ('given',) if solver.grating_type == 0 else ('given', 's', 'p')


def _field(algo, kwargs, **res):
    solver = wavebit.rcwa(backend=0, layer_algorithm=algo, answers='field', **kwargs)
    solver.solve()
    cell = solver.reconstruct_field(incidences=_incidences_of(solver), **res)
    return solver, np.array(cell)


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_field_matches_tmm(name, kwargs, kind):
    """The reconstructed near field, not just the far field, on every grating type.

    The SMM cascade throws away what field reconstruction walks, so it recovers the per-layer
    amplitudes from its own split-plane amplitudes afterwards. This is the check that the
    recovery is the same physical state and not merely a plausible one — including, on the
    paths that carry them, for the pure-s / pure-p incidence variants, which have their own
    amplitude columns.
    """
    _, tmm = _field('ETM', kwargs, sample_x=6, sample_y=6, sample_z=5)
    _, smm = _field('SMM', kwargs, sample_x=6, sample_y=6, sample_z=5)
    assert tmm.shape == smm.shape
    scale = np.max(np.abs(tmm))
    assert np.max(np.abs(tmm - smm)) < 1e-10 * scale, np.max(np.abs(tmm - smm)) / scale


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_field_state_matches_tmm(name, kwargs, kind):
    """The recovered state itself is the ETM's, layer by layer — not just its field.

    Both algorithms hand ``fields`` the same three things per layer: the incident amplitude
    ``T1``, the block advancing it to the next layer, and the block turning it into that
    layer's up-going amplitude. Those are unique given the modes (which both take from the
    same builders), so they must agree matrix for matrix. Only ``W`` / ``V`` / ``q`` are gauge
    here, and they are shared, not recomputed.

    The exception is the bottom layer — entry ``0``, since the list runs bottom-to-top. Its
    two blocks describe the amplitude handed *past* the stack, which the reconstruction never
    reads, and the two algorithms express that one in different bases (see ``_field_state``).
    What is defined there is their product, which is what the reconstruction actually forms.
    """
    tmm = wavebit.rcwa(backend=0, layer_algorithm='ETM', answers='field', **kwargs)
    smm = wavebit.rcwa(backend=0, layer_algorithm='SMM', answers='field', **kwargs)
    tmm.solve()
    smm.solve()

    a, b = np.array(tmm.T1), np.array(smm.T1)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(a))

    def close(x, y, where):
        # Floored at 1: an interface between identical materials makes B identically zero,
        # and a relative bound on a zero matrix compares two roundoff patterns.
        scale = max(float(np.max(np.abs(x))), 1.0)
        assert np.max(np.abs(x - y)) < 1e-9 * scale, (where, np.max(np.abs(x - y)))

    assert len(tmm.layer_info_list) == len(smm.layer_info_list) == len(kwargs['thickness'])
    for i, (li_t, li_s) in enumerate(zip(tmm.layer_info_list, smm.layer_info_list)):
        if i == 0:                                     # bottom layer: only the product is fixed
            close(np.array(li_t[6]) @ np.array(li_t[5]),
                  np.array(li_s[6]) @ np.array(li_s[5]), (i, 'B@A_i'))
            continue
        for j, nm in ((5, 'A_i'), (6, 'B')):
            close(np.array(li_t[j]), np.array(li_s[j]), (i, nm))


# theta=0, bot=2.25 (index 1.5) and lambda/period=1/4 put order m=+-6 exactly at the substrate cutoff:
# kx = 1.5 = n_bot, so kz_bot = 0 and the substrate's own mode basis is singular. Recovering
# the amplitudes in that basis is the natural-looking choice and it dies here; the gap basis
# the recovery actually uses cannot, by construction of eps_gap.
CUTOFF = dict(structure=np.array([[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]]),
              thickness=[300.0, 500.0], top=1.0, bot=2.25, theta=0.0,
              wavelength=500.0, period=[2000.0], truncation_order=[8, 0])


@pytest.mark.parametrize('extra', [dict(tetm='te'), dict(tetm='tm'), dict(psi=0.3 * np.pi, phi=0.3)],
                         ids=['te', 'tm', 'conical'])
def test_smm_field_with_an_order_exactly_at_the_substrate_cutoff(extra):
    """A retained order sitting at kz_bot = 0 is a real configuration, not a pathology."""
    kwargs = dict(CUTOFF, **extra)
    solver = wavebit.rcwa(backend=0, layer_algorithm='SMM', **kwargs)
    kz = np.sqrt(np.array(solver.n_bot, dtype=complex) ** 2
                 - np.array(solver.order_kxy()[0]) ** 2)
    assert np.min(np.abs(kz)) < 1e-12, 'premise: an order must sit exactly at cutoff'

    _, tmm = _field('ETM', kwargs, sample_x=6, sample_y=6, sample_z=4)
    _, smm = _field('SMM', kwargs, sample_x=6, sample_y=6, sample_z=4)
    assert np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-10 * np.max(np.abs(tmm))


# A rod on a hexagonal cell, sampled on that cell's own parametric grid. The skewed field
# path indexes its output by the lattice coordinates (u, v), not by (x, y) -- both algorithms
# go through the same reconstruction, so agreement here is about the amplitudes reaching it.
HEX_P = 400.0
HEX = np.column_stack(([HEX_P, 0.0], [HEX_P / 2, HEX_P * np.sqrt(3) / 2]))


def _hex_rod(n=48, radius=90.0):
    a1, a2 = HEX[:, 0], HEX[:, 1]
    X, Y = (np.asarray(c) for c in wavebit.grid((n, n), lattice=HEX))
    inside = np.zeros(X.shape, dtype=bool)
    for i in (-1, 0, 1):                       # minimum image over the lattice
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= ((X - s[0]) ** 2 + (Y - s[1]) ** 2) <= radius ** 2
    cell = np.full((1, n, n), 1.0)
    cell[0][inside] = 5.76           # 2.4**2
    return cell


def test_smm_field_on_a_skewed_cell_matches_tmm():
    """The two capabilities were added independently and had never met.

    Non-orthogonal cells arrived while field reconstruction was still ETM-only, so the skewed
    reconstruction has only ever been exercised through the ETM. Nothing in the amplitude
    recovery reads the lattice -- it works in the mode and gap bases, which the basis change
    never touches -- and this is the test that says so rather than assuming it.
    """
    kwargs = dict(structure=_hex_rod(), period=[HEX_P, HEX_P], truncation_order=[4, 4], thickness=[150.0],
                  wavelength=900.0, top=1.0, bot=1.0, theta=0.1, phi=0.3, psi=0.4,
                  lattice=HEX, factorization='nvf')
    _, tmm = _field('ETM', kwargs, sample_x=6, sample_y=6, sample_z=4)
    _, smm = _field('SMM', kwargs, sample_x=6, sample_y=6, sample_z=4)
    assert tmm.shape == smm.shape
    assert np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-10 * np.max(np.abs(tmm))


def test_field_before_solve_raises():
    """No solve, no amplitudes — fail with the reason rather than on a None.

    Declared for the field, so the refusal under test is `_require_solved` and not the
    `answers` check ahead of it; that one has its own coverage in
    `test_answers_declaration.py` and matches on a different sentence.
    """
    solver = wavebit.rcwa(backend=0, layer_algorithm='SMM', answers='field', **CASES_1D[0][1])
    with pytest.raises(RuntimeError, match='run solve'):
        solver.reconstruct_field(sample_x=4, sample_z=4)


def test_frozen_geometry_does_not_change_the_smm_answer():
    """SMM decides nothing discrete, so freezing must be a no-op for it, bit for bit."""
    kwargs = dict(CASES_1D[4][1])          # a substrate-matched bottom layer
    solver = wavebit.rcwa(backend=0, layer_algorithm='SMM', **kwargs)
    with_geom = solver.solve(geometry=solver.freeze_geometry())
    without = _solve('SMM', kwargs)
    assert np.allclose(np.array(with_geom.R), np.array(without.R), atol=0, rtol=0)
    assert np.allclose(np.array(with_geom.T), np.array(without.T), atol=0, rtol=0)


# ------------------------------------------------------------------------------ robustness

# Two differently patterned layers whose *mean* permittivity happens to match. This is the
# structure that exposed the same-material misdetection (test_fixes.test_10): ETM read the
# upper layer as identical to the one below and skipped their interface, losing ~50x on
# energy, while SMM — which never had such a short-circuit — stayed exact. ETM has none
# either now, so no solver can skip an interface; kept as what would catch either of them
# growing one back.
DEGENERATE_MEAN = dict(
    structure=np.concat([UC_2D, UC_2D[:, ::-1]]), thickness=[0.2, 0.3], top=1.0, bot=1.0,
    psi=0.4, wavelength=0.5, period=[0.6, 0.6], truncation_order=[2, 2])


@pytest.mark.parametrize('theta,phi', [(0.0, 0.0), (0.1, 0.0), (0.1, 0.4)])
def test_smm_survives_same_mean_layers(theta, phi):
    """Lossless energy conservation, and ETM agreement, on that stack."""
    kwargs = dict(DEGENERATE_MEAN, theta=theta, phi=phi)
    smm = _solve('SMM', kwargs)
    total = float(np.sum(np.array(smm.R)) + np.sum(np.array(smm.T)))
    assert abs(total - 1.0) < 1e-9, total

    tmm = _solve('ETM', kwargs)
    for field in ('R', 'T'):
        a, b = np.array(getattr(tmm, field)), np.array(getattr(smm, field))
        assert np.allclose(a, b, atol=1e-10, rtol=0), (field, np.max(np.abs(a - b)))


def test_smm_2d_matches_conical_on_y_invariant_stack():
    """Independent check on the coupled path: a y-invariant 2D grating is a conical one.

    The two go through different mode algebra (2D block operator vs the conical s/p pair), so
    agreement here is evidence about the physics, not just about the connecting algorithm.
    """
    pattern = np.array([[[9., 9., 1., 1., 9., 1., 1., 1.]],
                        [[4., 4., 4., 1., 1., 1., 4., 4.]]])
    common = dict(thickness=[0.2, 0.3], top=1.0, bot=1.0, psi=0.4, wavelength=0.5,
                  period=[0.6, 0.6], theta=0.1, phi=0.0, truncation_order=[2, 0])
    conical = _solve('SMM', dict(common, structure=pattern))
    two_d = _solve('SMM', dict(common, structure=np.repeat(pattern, 8, axis=1)))
    for field in ('R', 'T'):
        a = np.array(getattr(conical, field)).ravel()
        b = np.array(getattr(two_d, field)).ravel()
        assert np.allclose(a, b, atol=1e-10, rtol=0), (field, np.max(np.abs(a - b)))


# ----------------------------------------------------------------------------- portability

def _to(backend, x):
    if backend == 'jax':
        wavebit.enable_x64()
        import jax.numpy as jnp
        return jnp.asarray(x)
    import torch
    return torch.as_tensor(np.asarray(x))


@pytest.mark.parametrize('backend', ['jax', 'torch'])
@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_cross_backend(name, kwargs, kind, backend):
    pytest.importorskip(backend)
    smm_fn, _ = CORE_FN[kind]
    _, inputs = _core_inputs(kwargs)
    ri0, ti0 = smm_fn(**inputs)                                   # numpy reference

    bk = dict(inputs)
    for key in ('kx', 'ky', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all'):
        if key in bk:
            bk[key] = _to(backend, inputs[key])
    for key in ('theta', 'psi'):
        if key in bk:
            bk[key] = _to(backend, complex(inputs[key]))
    ri, ti = smm_fn(**bk)

    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


@pytest.mark.parametrize('backend,name', [(1, 'jax'), (2, 'torch')], ids=['jax', 'torch'])
@pytest.mark.parametrize('case', [CASES_1D[2], CASES_CONICAL[1], CASES_2D[0]],
                         ids=['1d', 'conical', '2d'])
def test_smm_field_cross_backend(case, backend, name):
    """The amplitude recovery is Array-API code like the rest — same field on every backend."""
    pytest.importorskip(name)
    if name == 'jax':
        wavebit.enable_x64()
    kwargs = case[1]
    _, reference = _field('SMM', kwargs, sample_x=5, sample_y=5, sample_z=4)

    solver = wavebit.rcwa(backend=backend, layer_algorithm='SMM', answers='field', **kwargs)
    solver.solve(geometry=solver.freeze_geometry())
    cell = np.asarray(solver.reconstruct_field(sample_x=5, sample_y=5, sample_z=4,
                                               incidences=_incidences_of(solver)))
    assert cell.shape == reference.shape
    assert np.max(np.abs(cell - reference)) < 1e-10 * np.max(np.abs(reference))


@pytest.mark.parametrize('index', [0, 1], ids=['te', 'tm'])
def test_gradient_jax_torch_fd(index):
    """d(sum T)/ds where s scales a real perturbation of a layer's eps-conv matrix."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    wavebit.enable_x64()
    import jax.numpy as jnp

    kwargs = dict(CASES_1D[index][1], theta=0.3)
    _, inputs = _core_inputs(kwargs)
    key = 'epy_conv_all' if index == 0 else 'epx_conv_all'
    base = np.array(inputs[key])
    direction = np.real(base).copy()
    conv_keys = ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')

    def f_np(s):
        d = dict(inputs); d[key] = base + s * direction
        return float(np.sum(np.array(smm_1d(**d)[1])))
    h = 1e-6
    g_fd = (f_np(h) - f_np(-h)) / (2 * h)

    def f_jax(s):
        d = dict(inputs)
        for k in conv_keys:
            d[k] = jnp.asarray(inputs[k])
        d[key] = jnp.asarray(base) + s * jnp.asarray(direction)
        d['theta'] = jnp.asarray(complex(inputs['theta']))
        return jnp.sum(smm_1d(**d)[1]).real
    g_jax = float(jax.grad(f_jax)(0.0))

    def f_torch(s):
        d = dict(inputs)
        for k in conv_keys:
            d[k] = torch.as_tensor(np.asarray(inputs[k]))
        d[key] = torch.as_tensor(base) + s * torch.as_tensor(direction)
        d['theta'] = torch.as_tensor(complex(inputs['theta']))
        return torch.sum(smm_1d(**d)[1]).real
    s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    f_torch(s).backward()
    g_torch = float(s.grad)

    assert abs(g_jax - g_torch) < 1e-6, (g_jax, g_torch)
    assert abs(g_jax - g_fd) < 1e-4, (g_jax, g_fd)


def test_jit_traceable_jax():
    """The SMM cascade is jittable: no data-dependent branching on traced values."""
    jax = pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax.numpy as jnp

    _, inputs = _core_inputs(CASES_1D[0][1])
    static = {k: v for k, v in inputs.items() if k not in
              ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')}
    arrays = {k: jnp.asarray(inputs[k]) for k in
              ('kx', 'epx_conv_all', 'epy_conv_all', 'epz_conv_i_all')}
    static['theta'] = jnp.asarray(complex(inputs['theta']))

    @jax.jit
    def run(kx, epx, epy, epz):
        return smm_1d(kx=kx, epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz, **static)

    ri, ti = run(arrays['kx'], arrays['epx_conv_all'], arrays['epy_conv_all'],
                 arrays['epz_conv_i_all'])
    ri0, ti0 = smm_1d(**inputs)
    assert np.allclose(np.array(ri0), np.asarray(ri), atol=1e-10, rtol=0)
    assert np.allclose(np.array(ti0), np.asarray(ti), atol=1e-10, rtol=0)


# --------------------------------------------------------- the scattering matrix the SMM keeps

#: A z-asymmetric two-layer stack between unequal half-spaces: the two sides genuinely differ,
#: so a mirrored or block-swapped mapping cannot pass by symmetry.
S_KW = dict(structure=UC_1D, thickness=[0.15, 0.22], top=4.0, bot=2.25, tetm='te',
            wavelength=0.777, period=[0.77], truncation_order=[5, 0], answers='amplitude')


def _lit(side='top', algo='SMM', **over):
    kw = dict(S_KW, theta=0.3, incidence_side=side, layer_algorithm=algo)
    kw.update(over)
    solver = wavebit.rcwa(backend=0, **kw)
    return solver, solver.solve()


def _delta(block):
    n = np.asarray(block).shape[0]
    column = np.zeros((n, 1), dtype=complex)
    column[n // 2, 0] = 1.0                     # the incident order, as the solve builds it
    return column


@pytest.mark.parametrize('algo', ['SMM', 'ETM'])
@pytest.mark.parametrize('answers', ['efficiency', 'amplitude', 'field'])
def test_the_matrix_is_there_exactly_where_amplitudes_are(algo, answers):
    """SMM from ``'amplitude'`` up, and nowhere else.

    An ``S`` *is* amplitude information -- its blocks applied to the incident vector are the
    reported amplitudes -- so it belongs on the rung that keeps them, and the rung below said
    only power was wanted. Absent after an ETM solve is not an ignored request: the ETM fixes
    one illumination's boundary condition and forms no both-sides operator at all.
    """
    solver, _ = _lit(algo=algo, answers=answers)
    if algo == 'SMM' and answers in ('amplitude', 'field'):
        assert solver.scattering.S11 is not None
    else:
        with pytest.raises(RuntimeError):
            solver.scattering


@pytest.mark.parametrize('kind, kw, says', [
    ('the algorithm forms none', dict(algo='ETM'), 'ETM'),
    ('the rung keeps power only', dict(algo='SMM', answers='efficiency'), 'amplitude'),
])
def test_an_absent_matrix_says_which_of_the_three_reasons_it_is(kind, kw, says):
    """Three reasons produced one ``None`` until 2026-08-22, and then an attribute error on it.

    ``'NoneType' object has no attribute 'S11'`` names neither the cause nor the fix, and the
    three causes want different fixes -- two are declarations, where "solve again" is false
    advice, and one is a state, where it is the whole advice. Same split
    :meth:`_require_field_declaration` already makes for ``reconstruct_field``.
    """
    solver, _ = _lit(**kw)
    with pytest.raises(RuntimeError, match=says) as excinfo:
        solver.scattering
    assert 'scattering matrix' in str(excinfo.value)


def test_two_reasons_at_once_name_the_one_that_would_still_bite():
    """``'ETM'`` *and* ``'efficiency'``: the algorithm is the answer, and the rung is not.

    Raising the rung on an ETM solver changes nothing -- the ETM forms no matrix at any rung
    -- so naming the rung here would send the caller to a fix that does not work, which is the
    exact failure the split exists to avoid. Ordering is what decides it, and nothing else
    reaches that branch: a case with one reason wrong passes whichever order the checks are
    in. Found by mutating the order and watching every test still pass.
    """
    solver, _ = _lit(algo='ETM', answers='efficiency')
    with pytest.raises(RuntimeError, match='ETM') as excinfo:
        solver.scattering
    assert 'answers=' not in str(excinfo.value)


def test_the_unsolved_solver_is_told_to_solve_and_not_to_reconfigure():
    """Declared right, just not run -- the one reason where "solve" is the whole answer."""
    solver = wavebit.rcwa(backend=0, **dict(S_KW, theta=0.3, layer_algorithm='SMM'))
    with pytest.raises(RuntimeError, match='run solve'):
        solver.scattering


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('block, reported', [('S11', 'r_s'), ('S21', 't_s')])
def test_the_kept_blocks_are_the_amplitudes_the_solve_reported(side, block, reported):
    """The two lit-side blocks, applied to the incident order, reproduce the reported answer.

    This is where the orientation claim becomes checkable rather than stated: the solve reads
    its own amplitudes off ``S11`` / ``S21`` from the top and off ``S22`` / ``S12`` from
    below, so a matrix handed over unswapped would disagree here on exactly one side -- and
    agree everywhere a symmetric test looks. It is also what says the ports are the two
    half-spaces and not the cascade's internal gap basis, which no numeric tolerance states.
    """
    solver, result = _lit(side)
    S = solver.scattering
    got = (np.asarray(getattr(S, block)) @ _delta(S.S11)).ravel()
    assert np.abs(got - np.asarray(getattr(result.given, reported)).ravel()).max() < 1e-12


def test_one_solve_carries_both_illumination_directions():
    """The far-side blocks answer the *other* side's solve — the point of keeping the whole S.

    ``S22`` / ``S12`` are the half of the matrix the lit side never reads, so nothing else in
    this file touches them; without this they could be transposed, swapped, or stale and every
    other assertion would still pass. Normal incidence on purpose: ``theta`` is
    incidence-relative, so at an angle the two sides do not share ``kx`` and their cascades
    are built on different order grids -- ``theta = 0`` is where one matrix really does hold
    both answers.
    """
    top_lit, _ = _lit('top', theta=0.0)
    bottom_lit, from_below = _lit('bottom', theta=0.0)
    S = top_lit.scattering
    delta = _delta(S.S11)
    for block, reported in (('S22', 'r_s'), ('S12', 't_s')):
        got = (np.asarray(getattr(S, block)) @ delta).ravel()
        want = np.asarray(getattr(from_below.given, reported)).ravel()
        assert np.abs(got - want).max() < 1e-12, block


def test_the_backside_matrix_is_the_scene_a_caller_would_build_by_hand():
    """Lit from below, the blocks are the flipped stack's, in the flipped stack's order.

    The frame assertion. Each block is an orientation-blind matrix, so the mapping can be
    internally consistent and still be assembled from the wrong end of the scene. Normal
    incidence for the same reason as above.
    """
    lit_from_below, _ = _lit('bottom', theta=0.0)
    by_hand, _ = _lit('top', theta=0.0, structure=UC_1D[::-1], thickness=[0.22, 0.15],
                      top=S_KW['bot'], bot=S_KW['top'])
    a, b = lit_from_below.scattering, by_hand.scattering
    assert a._fields[:4] == ('S11', 'S12', 'S21', 'S22')
    for name, x, y in zip(a._fields, a.blocks, b.blocks):
        assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12, name


# -------------------------------------------------------------------------------- cascading

#: The stack the cascade tests take apart: four layers between unequal half-spaces, cut
#: between layers 1 and 2. Every piece is solved as an independent problem.
WHOLE = np.concatenate([UC_1D, UC_1D[::-1]])
WHOLE_TH = [0.15, 0.22, 0.31, 0.09]
JOINT_KW = dict(tetm='te', wavelength=0.777, period=[0.77], truncation_order=[5, 0],
                answers='amplitude', layer_algorithm='SMM')


def _piece(structure, thickness, top, bot, theta, side='top'):
    solver = wavebit.rcwa(backend=0, structure=structure, thickness=thickness, top=top,
                          bot=bot, theta=theta, incidence_side=side, **JOINT_KW)
    solver.solve()
    return solver.scattering


def _whole():
    return _piece(WHOLE, WHOLE_TH, 4.0, 2.25, 0.3)


@pytest.mark.parametrize('joint', [1.0, 2.25, 9.0])
def test_two_pieces_cascade_into_the_stack_they_were_cut_from(joint):
    """Solve the halves apart, join them, and land on the whole stack's own matrix.

    The joint medium is a *basis*, not a layer: it has no thickness, nothing propagates in it,
    and the star product's ``inv(I - lower.S11 @ upper.S22)`` is the infinite bounce across the
    cut. Parametrised over three joint media because which one is chosen must not matter --
    only that both halves name the same one, and that both sit on the same order grid, which
    is what the converted ``theta`` below arranges.
    """
    upper = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, joint, 0.3)
    theta_lower = float(np.arcsin(np.sin(0.3) * 2.0 / np.sqrt(joint)))   # same k_par, in `joint`
    lower = _piece(WHOLE[2:], WHOLE_TH[2:], joint, 2.25, theta_lower)
    joined, whole = upper.cascade(lower), _whole()
    for name, got, want in zip(joined._fields, joined.blocks, whole.blocks):
        rel = np.abs(np.asarray(got) - np.asarray(want)).max() / np.abs(np.asarray(want)).max()
        assert rel < 1e-12, (name, joint, rel)


def test_the_joined_matrix_keeps_the_outer_ports():
    """The composite opens into the two half-spaces the *ends* opened into, not the joint."""
    upper = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 9.0, 0.3)
    theta_lower = float(np.arcsin(np.sin(0.3) * 2.0 / 3.0))
    lower = _piece(WHOLE[2:], WHOLE_TH[2:], 9.0, 2.25, theta_lower)
    joined = upper.cascade(lower)
    assert np.abs(complex(joined.entry_port.index) - 2.0) < 1e-12      # sqrt(4.0)
    assert np.abs(complex(joined.exit_port.index) - 1.5) < 1e-12       # sqrt(2.25)


def test_cascading_across_a_shifted_order_grid_is_refused():
    """The mistake this whole mechanism exists for, and it cannot be caught by looking.

    ``theta`` is measured in the incidence medium, so handing the lower piece the *same* angle
    across an index step silently shifts its whole ``kx`` grid. Composing anyway returns a
    plausible answer about ten percent wrong -- no NaN, no negative power, nothing to notice.
    Refusing needs the ports, which is why they ride in the matrix.
    """
    upper = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 9.0, 0.3)
    lower = _piece(WHOLE[2:], WHOLE_TH[2:], 9.0, 2.25, 0.3)            # not converted
    with pytest.raises(ValueError, match='in-plane wavevectors'):
        upper.cascade(lower)


def test_cascading_onto_a_different_joint_medium_is_refused():
    """Same order grid, different half-space at the cut -- still not the same basis.

    Normal incidence is what isolates this: ``sin(theta) = 0`` drops ``n_inc`` out of ``kx``,
    so the two grids agree to the bit and only the medium is left to disagree about. It is a
    real disagreement -- a port amplitude is a *mode* amplitude, and the mode normalisation
    carries the medium -- and at any other angle it would hide behind the grid check.
    """
    upper = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 4.0, 0.0)
    lower = _piece(WHOLE[2:], WHOLE_TH[2:], 1.0, 2.25, 0.0)
    assert np.allclose(np.asarray(upper.exit_port.kx), np.asarray(lower.entry_port.kx))
    with pytest.raises(ValueError, match='different half-spaces'):
        upper.cascade(lower)


def test_cascade_says_where_a_scattering_matrix_comes_from():
    """A bare 4-tuple is the obvious thing to try, and the message has to redirect it."""
    upper = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 1.0, 0.3)
    with pytest.raises(TypeError, match='answers'):
        upper.cascade(upper.blocks)


def test_cascade_joins_in_the_order_the_light_meets_the_pieces():
    """Lit from below, the piece the light reaches next is the one geometrically *above*.

    `cascade` is incidence-relative like every other orientation on this class, so `beyond`
    means "next along the path", not "underneath". Lit from the top the two readings agree and
    a geometric mistake would hide; lit from below they are opposite, which is what makes this
    case the one that can tell them apart. Normal incidence so both pieces share `kx` without
    a converted `theta`.
    """
    whole = _piece(WHOLE, WHOLE_TH, 4.0, 2.25, 0.0, side='bottom')
    # from below the light meets layers 2..3 first, then 0..1
    first = _piece(WHOLE[2:], WHOLE_TH[2:], 1.0, 2.25, 0.0, side='bottom')
    second = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 1.0, 0.0, side='bottom')
    joined = first.cascade(second)
    for name, got, want in zip(joined._fields, joined.blocks, whole.blocks):
        rel = np.abs(np.asarray(got) - np.asarray(want)).max() / np.abs(np.asarray(want)).max()
        assert rel < 1e-12, (name, rel)


def test_cascading_the_pieces_the_other_way_round_is_refused():
    """The geometric reading of `beyond`, on the side where it differs, does not quietly work."""
    first = _piece(WHOLE[2:], WHOLE_TH[2:], 1.0, 2.25, 0.0, side='bottom')
    second = _piece(WHOLE[:2], WHOLE_TH[:2], 4.0, 1.0, 0.0, side='bottom')
    with pytest.raises(ValueError, match='different half-spaces'):
        second.cascade(first)
# ------------------------------------------------- cascading a cell whose ky is its own axis

#: ``HEX`` respelt: ``a2 - a1`` in place of ``a2``. ``U = [[1, -1], [0, 1]]`` has ``det U = 1``,
#: so the two bases generate the *same* point lattice and a disc at every lattice point is one
#: and the same infinite array -- but ``truncation_order`` cuts a rectangle in ``(m, n)``, and
#: the two bases carry that rectangle onto different ``G``.
HEX_RESPELT = np.column_stack((HEX[:, 0], HEX[:, 1] - HEX[:, 0]))

#: The stack the oblique cascade cuts, stated as *shapes* rather than as a raster on purpose: a
#: raster is not the same object under a change of basis -- re-basing re-discretizes it and the
#: staircase term converges only as O(1/n) -- while a disc is carried across exactly.
#:
#: ``cx = cy = 0`` is the cell centre, which the two spellings put half of ``a1`` apart:
#: ``(a1 + a2)/2`` against ``(a1 + (a2 - a1))/2 = a2/2``. That is a rigid translation of the
#: whole stack -- both layers carry it -- and neither half of what this file asserts can see
#: one: a port is built from the basis and the media, and the cascade below joins two solves on
#: the *same* spelling.
CUT_HEX = [dict(background=1.0,
                shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=180.0, material=12.25)]),
           dict(background=6.25,
                shapes=[dict(type='circle', cx=0.0, cy=0.0, lx=240.0, material=1.0)])]
CUT_HEX_TH = [150.0, 220.0]
CUT_HEX_KW = dict(backend=0, wavelength=900.0, truncation_order=[3, 3], theta=0.0, phi=0.0,
                  psi=0.7, answers='amplitude', layer_algorithm='SMM')


def _hex_piece(layers, thickness, top, bot, lattice=HEX):
    solver = wavebit.rcwa(structure=layers, thickness=thickness, top=top, bot=bot,
                          lattice=lattice, **CUT_HEX_KW)
    solver.solve()
    return solver.scattering


def test_cascading_across_a_respelt_oblique_cell_is_refused():
    """The port check's ``ky`` half, on the only cell where ``ky`` can move without ``kx``.

    Every other cascade test in this file is 1D (``period=[0.77]``), so both ports carry
    ``ky=None`` and ``_port_disagreement`` returns from the ``kx`` clause before the ``ky`` one
    is ever evaluated. Delete ``or differ(a.ky, b.ky)`` from
    ``wavebit/_core/scattering.py::_port_disagreement`` and the other 131 tests in this file
    still pass; this one is what fails.

    The scene is the mistake, not a contrivance. ``HEX`` and ``HEX_RESPELT`` are two bases of
    one hexagonal lattice (``a2`` against ``a2 - a1``, unimodular), so the two solves describe
    the same physical array of discs and a user who spells the halves differently has written
    nothing wrong anywhere. What differs is which ``G`` the retained ``(m, n)`` rectangle lands
    on, and on a skewed cell ``kx`` and ``ky`` are not separate axes: ``kx`` depends on ``m``
    alone here and is untouched by the shear, while ``ky`` picks up ``2*lambda*m/(a*sqrt(3))``.

    Measured, at ``truncation_order=[3, 3]``, ``a = 400``, ``lambda = 900``:

    * matched ports (both halves on ``HEX``): ``max|dkx| = max|dky| = 0`` exactly, and the join
      reproduces the uncut stack's own blocks to ``2.5e-14`` relative -- half (a) below;
    * respelt lower half: ``max|dkx| = 0`` **exactly**, ``max|dky| = 7.794`` -- 7.8e12 times
      the check's own ``1e-12`` tolerance, so the two numbers are nowhere near each other and
      the assertion has room;
    * what the deleted clause lets through: the star product runs, returns no NaN and no
      negative power, and the composite is wrong by 8.7% / 13.5% / 6.2% / **58.2%** in
      ``S11`` / ``S12`` / ``S21`` / ``S22``.

    ``np.array_equal(exit.kx, entry.kx)`` is the load-bearing line and comes first: without it
    the refusal would only say the ports differ *somehow*, and a check that had kept only the
    ``kx`` clause would pass this test on the ``kx`` disagreement it never had. Normal
    incidence for the same reason -- ``sin(theta) = 0`` drops ``n_inc`` out of ``k_inc``, so the
    two grids agree to the bit rather than to a tolerance. The joint medium is identical on
    both sides so the half-space clause cannot fire first.
    """
    # (a) the machinery works on this cell: cut the stack, solve the halves apart, join.
    whole = _hex_piece(CUT_HEX, CUT_HEX_TH, 1.0, 2.25)
    upper = _hex_piece(CUT_HEX[:1], CUT_HEX_TH[:1], 1.0, 4.0)
    lower = _hex_piece(CUT_HEX[1:], CUT_HEX_TH[1:], 4.0, 2.25)
    joined = upper.cascade(lower)
    for name, got, want in zip(joined._fields, joined.blocks, whole.blocks):
        rel = np.abs(np.asarray(got) - np.asarray(want)).max() / np.abs(np.asarray(want)).max()
        assert rel < 1e-11, (name, rel)                    # clean residual 2.5e-14

    # (b) the same lower half on the other spelling of the same lattice: kx to the bit, ky not.
    respelt = _hex_piece(CUT_HEX[1:], CUT_HEX_TH[1:], 4.0, 2.25, lattice=HEX_RESPELT)
    exit_port, entry_port = upper.exit_port, respelt.entry_port
    assert np.array_equal(np.asarray(exit_port.kx), np.asarray(entry_port.kx))
    assert np.max(np.abs(np.asarray(exit_port.ky) - np.asarray(entry_port.ky))) > 1.0
    assert exit_port.index == entry_port.index and exit_port.mu == entry_port.mu
    with pytest.raises(ValueError, match='in-plane wavevectors'):
        upper.cascade(respelt)