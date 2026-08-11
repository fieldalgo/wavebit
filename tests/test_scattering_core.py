"""Scattering-matrix connecting algorithm (fieldwave._core.scattering).

SMM has no golden file of its own on purpose: its oracle is the ETM path, which is itself
pinned to the legacy results in test_golden_legacy_parity.py. So the properties tested are

  * agreement    — SMM == ETM on every grating type, in efficiencies *and* in the s/p
                   amplitudes, for the psi incidence and both pure-TE / pure-TM variants;
  * conservation — lossless stacks keep their energy (a check ETM agreement alone would pass
                   even if both were wrong the same way, so it is worth its own assertion);
  * wiring       — connecting_algo='SMM' reaches the core solve, and an unknown algo is
                   rejected with a useful message;
  * fields       — the near field reconstructed after an SMM solve, and the per-layer state
                   it is built from, match the ETM's, on an axis-aligned and a skewed cell;
  * portability  — same numbers on numpy / jax / torch, and gradients still flow.
"""
import numpy as np
import pytest

import fieldwave
from fieldwave._core.factorization import to_conv_mat_raster_discrete
from fieldwave._core.scattering import smm_1d, smm_1d_conical, smm_2d
from fieldwave._core.solve_1d import solve_1d, solve_1d_conical
from fieldwave._core.solve_2d import solve_2d

# ---------------------------------------------------------------------------------- cases
# name, kwargs for fieldwave.rcwa (grating type follows from ucell / phi, as in the solver)
UC_1D = np.array([[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]])
UC_1D_SUB = np.array([[[3., 3., 1., 1., 3., 1.]], [[1., 1., 1., 1., 1., 1.]]])
UC_2D = np.array([[[3., 3., 1., 1.], [1., 3., 3., 1.], [1., 1., 3., 3.], [3., 1., 1., 3.]]])

CASES_1D = [
    ('te_1layer', dict(pol=0, ucell=np.array([[[3., 3., 3., 1., 1., 1.]]]), thickness=[0.15],
                       n_top=2.0, n_bot=1.0, theta=0.0, wavelength=0.777, period=[0.77],
                       fto=[6, 0])),
    ('tm_1layer', dict(pol=1, ucell=np.array([[[3., 3., 3., 1., 1., 1.]]]), thickness=[0.15],
                       n_top=2.0, n_bot=1.0, theta=0.0, wavelength=0.777, period=[0.77],
                       fto=[6, 0])),
    ('te_2layer_oblique', dict(pol=0, ucell=UC_1D, thickness=[0.15, 0.22], n_top=2.0,
                               n_bot=1.5, theta=0.3, wavelength=0.777, period=[0.77],
                               fto=[5, 0])),
    ('tm_2layer_oblique', dict(pol=1, ucell=UC_1D, thickness=[0.15, 0.22], n_top=2.0,
                               n_bot=1.5, theta=0.3, wavelength=0.777, period=[0.77],
                               fto=[5, 0])),
    # bottom layer uniform and equal to the substrate: ETM short-circuits the interface,
    # SMM has no short-circuit to take — they must still land on the same answer.
    ('te_same_material', dict(pol=0, ucell=UC_1D_SUB, thickness=[0.2, 0.1], n_top=1.0,
                              n_bot=1.0, theta=0.0, wavelength=0.9, period=[0.7],
                              fto=[5, 0])),
    ('tm_same_material', dict(pol=1, ucell=UC_1D_SUB, thickness=[0.2, 0.1], n_top=1.0,
                              n_bot=1.0, theta=0.0, wavelength=0.9, period=[0.7],
                              fto=[5, 0])),
    # thick, strongly evanescent: the regime where a connecting algorithm goes wrong if it
    # ever inverts a growing exponential
    ('tm_thick', dict(pol=1, ucell=np.array([[[4., 4., 4., 4., 1., 1., 1., 1.]]]),
                      thickness=[20.0], n_top=1.0, n_bot=1.0, theta=0.0, wavelength=0.5,
                      period=[0.6], fto=[12, 0])),
]

CASES_CONICAL = [
    ('con_1layer', dict(ucell=np.array([[[3., 3., 3., 1., 1., 1.]]]), thickness=[0.15],
                        n_top=2.0, n_bot=1.0, theta=0.2, phi=0.3, psi=0.4, wavelength=0.777,
                        period=[0.77], fto=[6, 0])),
    ('con_2layer_lossy_sub', dict(ucell=UC_1D, thickness=[0.15, 0.22], n_top=1.0, n_bot=1.5,
                                  theta=0.3, phi=0.5, psi=1.1, wavelength=0.9, period=[0.7],
                                  fto=[5, 0])),
    ('con_phi0', dict(ucell=np.array([[[4., 4., 1., 1., 1., 1.]]]), thickness=[0.3],
                      n_top=1.5, n_bot=1.0, theta=0.25, phi=0.0, psi=0.7, wavelength=0.6,
                      period=[0.5], fto=[4, 0])),
]

CASES_2D = [
    ('2d_1layer', dict(ucell=UC_2D, thickness=[0.3], n_top=1.0, n_bot=1.5, theta=0.2,
                       phi=0.4, psi=0.7, wavelength=0.9, period=[0.7, 0.7], fto=[3, 3])),
    # The two layers differ in *mean* permittivity, not just in pattern. Equal means used to
    # trip a ETM misdetection (see the DEGENERATE_MEAN note below); that path is gone, and
    # this stays the way it is because the case is a cross-algorithm check, not a probe of it.
    ('2d_2layer_normal', dict(ucell=np.concat([UC_2D, 1 + UC_2D[:, ::-1]]),
                              thickness=[0.2, 0.3], n_top=1.0, n_bot=1.0, theta=0.0,
                              phi=0.0, psi=0.4, wavelength=0.5, period=[0.6, 0.6],
                              fto=[2, 2])),
]

ALL_CASES = ([(n, k, '1d') for n, k in CASES_1D]
             + [(n, k, 'conical') for n, k in CASES_CONICAL]
             + [(n, k, '2d') for n, k in CASES_2D])

AMPLITUDES = ('R_s', 'R_p', 'T_s', 'T_p')
EFFICIENCIES = ('de_ri', 'de_ri_s', 'de_ri_p', 'de_ti', 'de_ti_s', 'de_ti_p')


def _solve(algo, kwargs):
    return fieldwave.rcwa(backend=0, connecting_algo=algo, **kwargs).conv_solve()


# ------------------------------------------------------------------------------- agreement

@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_matches_tmm(name, kwargs, kind):
    """Same efficiencies and same s/p amplitudes, for every incidence the solve reports."""
    tmm = _solve('ETM', kwargs)
    smm = _solve('SMM', kwargs)

    for variant in ('res', 'res_te_inc', 'res_tm_inc'):
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
    total = float(np.sum(np.array(smm.de_ri)) + np.sum(np.array(smm.de_ti)))
    assert abs(total - 1.0) < 1e-9, total


# ---------------------------------------------------------------------------------- wiring

def _core_inputs(kwargs):
    """Solver-built arguments for a direct core call (same route the mixin takes)."""
    solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', **kwargs)
    solver.conv_solve()                    # assigns the grating type
    epx, epy, epz = to_conv_mat_raster_discrete(
        solver.ucell, solver.fto[0], solver.fto[1])
    kx, ky = solver.get_kx_ky_vector(kwargs['wavelength'])
    common = dict(wavelength=kwargs['wavelength'], kx=kx, n_top=solver.n_top,
                  n_bot=solver.n_bot, theta=solver.theta, thickness=solver.thickness,
                  epx_conv_all=epx, epy_conv_all=epy, epz_conv_i_all=epz)
    if solver.grating_type_assigned == 0:
        return solver, dict(common, pol=solver.pol)
    return solver, dict(common, ky=ky, psi=solver.psi)


CORE_FN = {'1d': (smm_1d, solve_1d), 'conical': (smm_1d_conical, solve_1d_conical),
           '2d': (smm_2d, solve_2d)}


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_wiring_matches_direct_core_call(name, kwargs, kind):
    """connecting_algo='SMM' dispatches to the core with the arguments it expects."""
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
    np.testing.assert_allclose(np.array(via_api.de_ri), np.array(ri), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.array(via_api.de_ti), np.array(ti), rtol=1e-12, atol=1e-15)


def test_unknown_connecting_algo_is_rejected():
    kwargs = CASES_1D[0][1]
    with pytest.raises(ValueError, match="'RCWA-ish'"):
        _solve('RCWA-ish', kwargs)


def _field(algo, kwargs, **res):
    solver = fieldwave.rcwa(backend=0, connecting_algo=algo, **kwargs)
    solver.conv_solve()
    cell = solver.calculate_field(set_field_input=(True, True, True), **res)
    return solver, np.array(cell)


@pytest.mark.parametrize('name,kwargs,kind', ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_smm_field_matches_tmm(name, kwargs, kind):
    """The reconstructed near field, not just the far field, on every grating type.

    The SMM cascade throws away what field reconstruction walks, so it recovers the per-layer
    amplitudes from its own split-plane amplitudes afterwards. This is the check that the
    recovery is the same physical state and not merely a plausible one — including for the
    TE / TM incidence variants, which have their own amplitude columns.
    """
    _, tmm = _field('ETM', kwargs, res_x=6, res_y=6, res_z=5)
    _, smm = _field('SMM', kwargs, res_x=6, res_y=6, res_z=5)
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
    tmm = fieldwave.rcwa(backend=0, connecting_algo='ETM', **kwargs)
    smm = fieldwave.rcwa(backend=0, connecting_algo='SMM', **kwargs)
    tmm.conv_solve()
    smm.conv_solve()

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


# theta=0, n_bot=1.5 and lambda/period=1/4 put order m=+-6 exactly at the substrate cutoff:
# kx = 1.5 = n_bot, so kz_bot = 0 and the substrate's own mode basis is singular. Recovering
# the amplitudes in that basis is the natural-looking choice and it dies here; the gap basis
# the recovery actually uses cannot, by construction of eps_gap.
CUTOFF = dict(ucell=np.array([[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]]),
              thickness=[300.0, 500.0], n_top=1.0, n_bot=1.5, theta=0.0,
              wavelength=500.0, period=[2000.0], fto=[8, 0])


@pytest.mark.parametrize('extra', [dict(pol=0), dict(pol=1), dict(pol=0.4, phi=0.3)],
                         ids=['te', 'tm', 'conical'])
def test_smm_field_with_an_order_exactly_at_the_substrate_cutoff(extra):
    """A retained order sitting at kz_bot = 0 is a real configuration, not a pathology."""
    kwargs = dict(CUTOFF, **extra)
    solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', **kwargs)
    kz = np.sqrt(np.array(solver.n_bot, dtype=complex) ** 2
                 - np.array(solver.get_kx_ky_vector(kwargs['wavelength'])[0]) ** 2)
    assert np.min(np.abs(kz)) < 1e-12, 'premise: an order must sit exactly at cutoff'

    _, tmm = _field('ETM', kwargs, res_x=6, res_y=6, res_z=4)
    _, smm = _field('SMM', kwargs, res_x=6, res_y=6, res_z=4)
    assert np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-10 * np.max(np.abs(tmm))


# A rod on a hexagonal cell, sampled on that cell's own parametric grid. The skewed field
# path indexes its output by the lattice coordinates (u, v), not by (x, y) -- both algorithms
# go through the same reconstruction, so agreement here is about the amplitudes reaching it.
HEX_P = 400.0
HEX = np.column_stack(([HEX_P, 0.0], [HEX_P / 2, HEX_P * np.sqrt(3) / 2]))


def _hex_rod(n=48, radius=90.0):
    a1, a2 = HEX[:, 0], HEX[:, 1]
    t = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(t, t)
    pts = U[..., None] * a1 + V[..., None] * a2
    centre = 0.5 * (a1 + a2)
    inside = np.zeros(pts.shape[:2], dtype=bool)
    for i in (-1, 0, 1):                       # minimum image over the lattice
        for j in (-1, 0, 1):
            s = i * a1 + j * a2
            inside |= ((pts[..., 0] - centre[0] - s[0]) ** 2
                       + (pts[..., 1] - centre[1] - s[1]) ** 2) <= radius ** 2
    cell = np.full((1, n, n), 1.0)
    cell[0][inside] = 2.4
    return cell


def test_smm_field_on_a_skewed_cell_matches_tmm():
    """The two capabilities were added independently and had never met.

    Non-orthogonal cells arrived while field reconstruction was still ETM-only, so the skewed
    reconstruction has only ever been exercised through the ETM. Nothing in the amplitude
    recovery reads the lattice -- it works in the mode and gap bases, which the basis change
    never touches -- and this is the test that says so rather than assuming it.
    """
    kwargs = dict(ucell=_hex_rod(), period=[HEX_P, HEX_P], fto=[4, 4], thickness=[150.0],
                  wavelength=900.0, n_top=1.0, n_bot=1.0, theta=0.1, phi=0.3, psi=0.4,
                  lattice=HEX, factorization='nvf')
    _, tmm = _field('ETM', kwargs, res_x=6, res_y=6, res_z=4)
    _, smm = _field('SMM', kwargs, res_x=6, res_y=6, res_z=4)
    assert tmm.shape == smm.shape
    assert np.all(np.isfinite(np.abs(smm)))
    assert np.max(np.abs(tmm - smm)) < 1e-10 * np.max(np.abs(tmm))


def test_field_before_solve_raises():
    """No solve, no amplitudes — fail with the reason rather than on a None."""
    solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', **CASES_1D[0][1])
    with pytest.raises(RuntimeError, match='solve'):
        solver.calculate_field(res_x=4, res_z=4)


def test_frozen_geometry_does_not_change_the_smm_answer():
    """SMM decides nothing discrete, so freezing must be a no-op for it, bit for bit."""
    kwargs = dict(CASES_1D[4][1])          # a substrate-matched bottom layer
    solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', **kwargs)
    with_geom = solver.conv_solve(geometry=solver.freeze_geometry())
    without = _solve('SMM', kwargs)
    assert np.allclose(np.array(with_geom.de_ri), np.array(without.de_ri), atol=0, rtol=0)
    assert np.allclose(np.array(with_geom.de_ti), np.array(without.de_ti), atol=0, rtol=0)


# ------------------------------------------------------------------------------ robustness

# Two differently patterned layers whose *mean* permittivity happens to match. This is the
# structure that exposed the same-material misdetection (test_fixes.test_10): ETM read the
# upper layer as identical to the one below and skipped their interface, losing ~50x on
# energy, while SMM — which never had such a short-circuit — stayed exact. ETM has none
# either now, so no solver can skip an interface; kept as what would catch either of them
# growing one back.
DEGENERATE_MEAN = dict(
    ucell=np.concat([UC_2D, UC_2D[:, ::-1]]), thickness=[0.2, 0.3], n_top=1.0, n_bot=1.0,
    psi=0.4, wavelength=0.5, period=[0.6, 0.6], fto=[2, 2])


@pytest.mark.parametrize('theta,phi', [(0.0, 0.0), (0.1, 0.0), (0.1, 0.4)])
def test_smm_survives_same_mean_layers(theta, phi):
    """Lossless energy conservation, and ETM agreement, on that stack."""
    kwargs = dict(DEGENERATE_MEAN, theta=theta, phi=phi)
    smm = _solve('SMM', kwargs)
    total = float(np.sum(np.array(smm.de_ri)) + np.sum(np.array(smm.de_ti)))
    assert abs(total - 1.0) < 1e-9, total

    tmm = _solve('ETM', kwargs)
    for field in ('de_ri', 'de_ti'):
        a, b = np.array(getattr(tmm, field)), np.array(getattr(smm, field))
        assert np.allclose(a, b, atol=1e-10, rtol=0), (field, np.max(np.abs(a - b)))


def test_smm_2d_matches_conical_on_y_invariant_stack():
    """Independent check on the coupled path: a y-invariant 2D grating is a conical one.

    The two go through different mode algebra (2D block operator vs the conical s/p pair), so
    agreement here is evidence about the physics, not just about the connecting algorithm.
    """
    pattern = np.array([[[3., 3., 1., 1., 3., 1., 1., 1.]],
                        [[2., 2., 2., 1., 1., 1., 2., 2.]]])
    common = dict(thickness=[0.2, 0.3], n_top=1.0, n_bot=1.0, psi=0.4, wavelength=0.5,
                  period=[0.6, 0.6], theta=0.1, phi=0.0, fto=[2, 0])
    conical = _solve('SMM', dict(common, ucell=pattern))
    two_d = _solve('SMM', dict(common, ucell=np.repeat(pattern, 8, axis=1)))
    for field in ('de_ri', 'de_ti'):
        a = np.array(getattr(conical, field)).ravel()
        b = np.array(getattr(two_d, field)).ravel()
        assert np.allclose(a, b, atol=1e-10, rtol=0), (field, np.max(np.abs(a - b)))


# ----------------------------------------------------------------------------- portability

def _to(backend, x):
    if backend == 'jax':
        import fieldwave.on_jax     # noqa: F401  (enables x64; RCWA needs double precision)
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
        # `as _`: a plain `import fieldwave.on_jax` would rebind `fieldwave` as a local.
        import fieldwave.on_jax as _   # noqa: F401  (enables x64; RCWA needs double precision)
    kwargs = case[1]
    _, reference = _field('SMM', kwargs, res_x=5, res_y=5, res_z=4)

    solver = fieldwave.rcwa(backend=backend, connecting_algo='SMM', **kwargs)
    solver.conv_solve(geometry=solver.freeze_geometry())
    cell = np.asarray(solver.calculate_field(res_x=5, res_y=5, res_z=4,
                                             set_field_input=(True, True, True)))
    assert cell.shape == reference.shape
    assert np.max(np.abs(cell - reference)) < 1e-10 * np.max(np.abs(reference))


@pytest.mark.parametrize('pol', [0, 1], ids=['te', 'tm'])
def test_gradient_jax_torch_fd(pol):
    """d(sum de_ti)/ds where s scales a real perturbation of a layer's eps-conv matrix."""
    jax = pytest.importorskip('jax')
    torch = pytest.importorskip('torch')
    import fieldwave.on_jax          # noqa: F401  (enables x64)
    import jax.numpy as jnp

    kwargs = dict(CASES_1D[pol][1], theta=0.3)
    _, inputs = _core_inputs(kwargs)
    key = 'epy_conv_all' if pol == 0 else 'epx_conv_all'
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
    import fieldwave.on_jax          # noqa: F401  (enables x64)
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
