"""The first-order (4ff) layer solve, against the two things that can arbitrate it.

``fieldwave/_core/solve_2d_4ff.py`` exists because the 2ff reduction does not survive a
populated z row — physically, a magnetisation lying in the plane. Nothing dispatches to it yet;
these tests are what says the formulation is right before anything is built on top.

Two oracles, and the split between them is the point:

* **Where the 2ff path is valid, the two must agree.** A z-decoupled tensor has an answer both
  formulations can reach, and the 4ff eigenvalues must be exactly the 2ff modal constants,
  ``lam = i q``. This is the internal cross-check the module is designed around, and no external
  oracle provides it.
* **Where it is not, Berreman arbitrates.** ``tests/test_anisotropic_berreman.py`` already
  carries a 4x4 first-order oracle for uniform layers, written independently and validated
  against RETICOLO and fmmax. For a single order the two operators should agree entry for
  entry, not merely in their spectra.

Both of those are ours, so a misconception this repo holds about the problem survives every
test in this file. The outside check lives in
``tests/references/test_4ff_external_goldens.py`` — RETICOLO and fmmax, 21 cases, on goldens
that were already committed.

Every case here is a uniform layer at one Fourier order. Multi-order is untested — the algebra
is written in blocks and should extend, but "should extend" is exactly the sort of claim this
suite exists to disbelieve.
"""
import numpy as np
import pytest

from fieldwave._core.solve_2d import layer_modes_2d
from fieldwave._core.solve_2d_4ff import (layer_matrix_4ff, split_forward_backward,
                                          solve_uniform_stack, vacuum_sp_states)

WL = 600.0
THICK = 220.0
EO, EE, G = 2.25, 4.0, 0.45


def blocks(eps, ff=1):
    """The mapping ``layer_matrix_4ff`` takes, from a 3x3 tensor."""
    e = np.asarray(eps, dtype=complex)
    I = np.eye(ff, dtype=complex)
    return dict(exx=e[0, 0] * I, exy=e[0, 1] * I, exz=e[0, 2] * I,
                eyx=e[1, 0] * I, eyy=e[1, 1] * I, eyz=e[1, 2] * I,
                ezx=e[2, 0] * I, ezy=e[2, 1] * I, ezz_i=np.linalg.inv(e[2, 2] * I))


def gyro(direction, g=G, ed=EO):
    """``eps_ij = ed d_ij + i e_ijk g_k`` — the magnetisation direction picks the cells."""
    e = np.eye(3, dtype=complex) * ed
    gx, gy, gz = np.asarray(direction, dtype=float) * g
    e[0, 1] += 1j * gz
    e[1, 0] -= 1j * gz
    e[1, 2] += 1j * gx
    e[2, 1] -= 1j * gx
    e[2, 0] += 1j * gy
    e[0, 2] -= 1j * gy
    return e


def rotated_uniaxial(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[EE * c ** 2 + EO * s ** 2, (EE - EO) * s * c, 0],
                     [(EE - EO) * s * c, EE * s ** 2 + EO * c ** 2, 0],
                     [0, 0, EO]], dtype=complex)


Z_DECOUPLED = [
    ('isotropic', np.diag([EO, EO, EO]).astype(complex)),
    ('diagonal biaxial', np.diag([2.1, 3.4, 2.8]).astype(complex)),
    ('rotated uniaxial', rotated_uniaxial(np.deg2rad(30))),
    ('gyrotropic, M along z', gyro((0, 0, 1))),
]

Z_COUPLED = [
    ('Voigt, M along x', gyro((1, 0, 0))),
    ('Voigt, M along y', gyro((0, 1, 0))),
    ('M tilted 45 deg', gyro((np.sin(np.pi / 4), 0, np.cos(np.pi / 4)))),
    ('general z-coupled', np.array([[2.4, 0.2, 0.5], [0.2, 2.9, 0.3], [0.5, 0.3, 2.6]],
                                   dtype=complex)),
]

ANGLES = [0.0, 35.0, 62.0]


def _kxky(theta_deg, alpha_deg=0.0):
    st, a = np.sin(np.deg2rad(theta_deg)), np.deg2rad(alpha_deg)
    return np.array([st * np.cos(a)]), np.array([st * np.sin(a)])


def rt(eps, theta_deg, thick=THICK, alpha_deg=0.0):
    """``{'s': (R, T), 'p': (R, T)}`` for a single layer, in a named polarisation basis."""
    kx, ky = _kxky(theta_deg, alpha_deg)
    psi_s, psi_p = vacuum_sp_states(np, kx, ky, 1)
    layers = [(blocks(eps), thick)]
    return {'s': solve_uniform_stack(WL, kx, ky, layers, 1, psi_s),
            'p': solve_uniform_stack(WL, kx, ky, layers, 1, psi_p)}


def rot_z(eps, alpha_deg):
    """The tensor as seen after turning the whole problem by ``alpha`` about the stack normal."""
    a = np.deg2rad(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.asarray(eps, dtype=complex) @ R.T


@pytest.mark.parametrize('theta_deg', ANGLES)
@pytest.mark.parametrize('label,eps', Z_DECOUPLED, ids=[c[0] for c in Z_DECOUPLED])
def test_reproduces_the_2ff_modal_constants(label, eps, theta_deg):
    """Where 2ff is valid the two formulations must give the same modes, ``lam = i q``.

    ``psi`` goes as ``exp(i k0 lam z)`` and the 2ff modes as ``exp(-k0 q z)``, which is what
    fixes the factor. A sign or transpose error anywhere in the derivation lands here.
    """
    kx, ky = _kxky(theta_deg)
    Kx, Ky = np.diag(kx).astype(complex), np.diag(ky).astype(complex)
    I = np.eye(1, dtype=complex)
    b = blocks(eps)

    _, _, q = layer_modes_2d(np, Kx, Ky, I, b['exx'], b['eyy'], b['ezz_i'],
                             -b['exy'], -b['eyx'], complex)
    lam = np.linalg.eigvals(layer_matrix_4ff(np, Kx, Ky, **b))

    gap = np.abs(np.sort_complex(lam)
                 - np.sort_complex(np.concatenate([1j * q, -1j * q]))).max()
    assert gap < 1e-12, f'{label} at {theta_deg} deg: {gap:.2e}'


@pytest.mark.parametrize('theta_deg', [0.0, 35.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_matches_the_berreman_operator_entry_for_entry(label, eps, theta_deg):
    """At one order the two first-order operators are the same 4x4 and should say so.

    Spectra agreeing would leave a similarity transform unaccounted for; entries agreeing does
    not. Both use ``psi = (Ex, Ey, Hx, Hy)``, so they are directly comparable.
    """
    from test_anisotropic_berreman import _berreman_M

    kx, _ = _kxky(theta_deg)
    Kx, Ky = np.diag(kx).astype(complex), np.zeros((1, 1), dtype=complex)
    mine = layer_matrix_4ff(np, Kx, Ky, **blocks(eps))
    theirs = _berreman_M(eps, float(kx[0]))
    gap = np.abs(mine - theirs).max()
    assert gap < 1e-12, f'{label} at {theta_deg} deg: {gap:.2e}'


@pytest.mark.parametrize('theta_deg', ANGLES)
@pytest.mark.parametrize('label,eps', Z_DECOUPLED, ids=[c[0] for c in Z_DECOUPLED])
def test_the_forward_half_is_the_2ff_half(label, eps, theta_deg):
    """The direction split must select the same modes ``mode_q`` does.

    ``mode_q`` picks a root by sign; this picks a half by energy flux. On a z-decoupled tensor
    they have to land on the same set, which is what licenses trusting the flux rule where no
    2ff answer exists to compare against.
    """
    kx, ky = _kxky(theta_deg)
    Kx, Ky = np.diag(kx).astype(complex), np.diag(ky).astype(complex)
    I = np.eye(1, dtype=complex)
    b = blocks(eps)

    _, _, q = layer_modes_2d(np, Kx, Ky, I, b['exx'], b['eyy'], b['ezz_i'],
                             -b['exy'], -b['eyx'], complex)
    lam_f, _, lam_b, _ = split_forward_backward(np, layer_matrix_4ff(np, Kx, Ky, **b), 1)

    assert len(lam_f) == len(lam_b) == 2
    gap = np.abs(np.sort_complex(lam_f) - np.sort_complex(1j * q)).max()
    assert gap < 1e-12, f'{label} at {theta_deg} deg: {gap:.2e}'


@pytest.mark.parametrize('kx', [2.5, 4.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_an_evanescent_order_is_split_by_decay(label, eps, kx):
    """Past every index in the layer, no mode carries flux and the fallback decides alone.

    This is not an exotic corner: above the light line *every* diffraction order looks like
    this, so it is the ordinary case for the multi-order path and the only case that exercises
    the fallback at all. Scaling the flux threshold by the largest flux in the set — rather than
    by each mode's own norm — passes every propagating test in this file and then reads 1e-17 of
    noise as energy flow here, splitting 3/1.
    """
    M = layer_matrix_4ff(np, np.diag([kx]).astype(complex), np.zeros((1, 1), complex),
                         **blocks(eps))
    lam_f, _, lam_b, _ = split_forward_backward(np, M, 1)

    assert len(lam_f) == len(lam_b) == 2, label
    assert np.imag(lam_f).min() > 0.1, f'{label}: forward modes must decay in +z'
    assert np.imag(lam_b).max() < -0.1, f'{label}: backward modes must decay in -z'


@pytest.mark.parametrize('theta_deg', ANGLES + [55.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_forward_modes_never_grow(label, eps, theta_deg):
    """No case here has gain, so nothing travelling towards +z may amplify along the way.

    Under ``e^{-iwt}`` with ``psi ~ exp(i k0 lam z)``, growth is ``Im(lam) < 0``. This is the
    convention-level statement — get the time sign wrong anywhere and forward modes come back
    growing while every efficiency still looks plausible.
    """
    kx, ky = _kxky(theta_deg)
    M = layer_matrix_4ff(np, np.diag(kx).astype(complex), np.diag(ky).astype(complex),
                         **blocks(eps))
    lam_f, _, lam_b, _ = split_forward_backward(np, M, 1)

    assert np.imag(lam_f).min() > -1e-12, f'{label} at {theta_deg} deg: forward mode grows'
    assert np.imag(lam_b).max() < 1e-12, f'{label} at {theta_deg} deg: backward mode grows'


@pytest.mark.parametrize('theta_deg', [0.0, 35.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_stack_solve_matches_berreman(label, eps, theta_deg):
    """End to end: modes, split, boundary and transfer together.

    Four of these cases are z-coupled, which is to say the 2ff path cannot reach them at all —
    handed one it drops the z row and is 2e-2 to 6e-2 out while conserving energy exactly
    (``tests/test_anisotropic_berreman.py::test_voigt_geometry_is_refused_rather_than_quietly_reduced``).
    """
    from test_anisotropic_berreman import berreman_efficiencies

    got = rt(eps, theta_deg)
    ref = berreman_efficiencies([(np.asarray(eps, dtype=complex), THICK)],
                                np.deg2rad(theta_deg))
    for pol in ('s', 'p'):
        want = (ref[pol]['R_s'] + ref[pol]['R_p'], ref[pol]['T_s'] + ref[pol]['T_p'])
        gap = max(abs(g - w) for g, w in zip(got[pol], want))
        assert gap < 1e-12, f'{label} at {theta_deg} deg, {pol}-pol: {gap:.2e}'


@pytest.mark.parametrize('alpha_deg', [17.0, 90.0, 133.0, 251.0])
@pytest.mark.parametrize('theta_deg', [18.0, 35.0, 62.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_rotating_the_whole_problem_about_z_changes_nothing(label, eps, theta_deg, alpha_deg):
    """Turn the tensor and the incidence azimuth together; the efficiencies must not move.

    This is the only judge of a tilted magnetisation that is neither a solver nor ours. **All
    four surveyed solvers are out, each for a measured reason** — RETICOLO stops conserving
    energy past one off-diagonal pair; fmmax's "general anisotropic" entry point means *magnetic*
    and still takes only the five 2ff slots; S4 has the same shape; and inkstone, the one with a
    full 3x3 API, returns numbers bit-identical to the same tensor with its z row zeroed. The
    table is in ``tests/references/README.md``. So the alternative to a theorem is two of our own
    derivations agreeing, which is what ``_berreman_M`` already is.

    It is also the only thing in the repo that exercises ``ky != 0`` at all. Every other
    anisotropy test sits in the plane of incidence, and so does ``_berreman_M``, which takes
    ``kx`` alone. Measured rather than asserted: five mutations that are exactly the identity at
    ``ky = 0`` — dropping ``ky`` from ``kt``, from ``kz``, from the s direction, from ``Ez``'s
    ``Hx`` term, and from ``Ey'`` — leave every other anisotropy test in this repo passing, and
    every one of them dies here.

    **Oblique only, and that is not a dodge.** At normal incidence the plane of incidence does
    not exist, so ``vacuum_sp_states`` fixes ``shat = y`` by convention; the labels then stop
    following the geometry and a quarter turn genuinely exchanges them — see the test below,
    which pins that instead.
    """
    before = rt(eps, theta_deg)
    after = rt(rot_z(eps, alpha_deg), theta_deg, alpha_deg=alpha_deg)
    for pol in ('s', 'p'):
        gap = max(abs(b - a) for b, a in zip(before[pol], after[pol]))
        assert gap < 1e-12, f'{label} at {theta_deg} deg, turned {alpha_deg} deg, {pol}: {gap:.2e}'


@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_a_quarter_turn_at_normal_incidence_exchanges_s_and_p(label, eps):
    """The companion to the above, and the reason it is restricted to oblique incidence.

    With ``kt = 0`` there is no plane of incidence for s and p to be defined against, so the
    convention names them after the fixed axes. Turning the tensor a quarter turn then maps x to
    y and the two labels swap outright rather than staying put. Asserting the swap keeps that
    from being mistaken for the covariance above failing.
    """
    before, after = rt(eps, 0.0), rt(rot_z(eps, 90.0), 0.0)
    for src, dst in (('s', 'p'), ('p', 's')):
        gap = max(abs(b - a) for b, a in zip(before[src], after[dst]))
        assert gap < 1e-12, f'{label}: {src} should land on {dst}, off by {gap:.2e}'


@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_lossless_layers_conserve_energy(label, eps):
    """Every tensor here is Hermitian, so ``R + T`` must be 1 for each incident mode."""
    e = np.asarray(eps, dtype=complex)
    assert np.abs(e - e.conj().T).max() < 1e-12, 'case is not Hermitian; fix the case, not this'
    for pol, (R, T) in rt(eps, 35.0).items():
        assert R + T == pytest.approx(1.0, abs=1e-10), f'{label}, {pol}-pol'


ABSORBING_N = 0.2 + 4.0j


def _fresnel(n, theta_deg):
    """``(R_s, R_p)`` off a single vacuum/medium interface — the closed form a thick slab tends to."""
    ct, st = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    c2 = np.sqrt(n ** 2 - st ** 2)
    return (abs((ct - c2) / (ct + c2)) ** 2,
            abs((n ** 2 * ct - c2) / (n ** 2 * ct + c2)) ** 2)


@pytest.mark.parametrize('theta_deg', [0.0, 35.0, 62.0])
def test_a_thick_absorbing_slab_is_a_single_interface(theta_deg):
    """The anchor for the regime the transfer cannot reach, and it is a closed form.

    Once a slab is thick enough that nothing returns from its far side, its reflectance *is* the
    single-interface Fresnel value. At 60000 nm and ``n = 0.2 + 4j`` the modal exponent is about
    2500 — eight times past where ``float64`` gives up — so this says the cascade stays exact
    where the eigen-decomposition transfer has no answer at all, against arithmetic rather than
    against another solver.
    """
    kx, ky = _kxky(theta_deg)
    layers = [(blocks(np.diag([ABSORBING_N ** 2] * 3).astype(complex)), 60000.0)]
    psi_s, psi_p = vacuum_sp_states(np, kx, ky, 1)
    want_s, want_p = _fresnel(ABSORBING_N, theta_deg)

    for psi, want, pol in ((psi_s, want_s, 's'), (psi_p, want_p, 'p')):
        R, T = solve_uniform_stack(WL, kx, ky, layers, 1, psi)
        assert R == pytest.approx(want, abs=1e-12), f'{pol}-pol at {theta_deg} deg'
        assert T == 0.0, f'{pol}-pol: nothing may survive 100 wavelengths of this'


def _airy_slab(n, theta_deg, d):
    """Exact ``(R, T)`` for an isotropic slab between vacuum, s-polarised.

    Vacuum on *both* sides, so there is no ``Re(n cos)`` normalisation on ``T`` — including one
    scales the answer by 1.69 at 35 degrees, which reads exactly like a solver that is wrong in
    the transmission only.
    """
    k0, t = 2 * np.pi / WL, np.deg2rad(theta_deg)
    ct, c2 = np.cos(t), np.sqrt(n ** 2 - np.sin(t) ** 2)
    r, t12, t21 = (ct - c2) / (ct + c2), 2 * ct / (ct + c2), 2 * c2 / (ct + c2)
    ph = np.exp(1j * k0 * c2 * d)
    return (abs(r * (1 - ph ** 2) / (1 - r ** 2 * ph ** 2)) ** 2,
            abs(t12 * t21 * ph / (1 - r ** 2 * ph ** 2)) ** 2)


@pytest.mark.parametrize('thick', [500.0, 4000.0, 10000.0, 30000.0])
@pytest.mark.parametrize('n', [1.5 + 0.1j, 1.5 - 0.1j], ids=['lossy', 'gain'])
def test_the_cascade_matches_the_closed_form_across_thicknesses(n, thick):
    """A closed form over the whole range, where interference is still alive.

    ``test_a_thick_absorbing_slab_is_a_single_interface`` pins the far limit, where the answer
    stops depending on thickness at all. This pins the way in: at 30000 nm the modal exponent is
    34 and ``T`` is 2.6e-30, and the cascade still tracks the Airy formula to ~1e-13 relative.

    The gain case (``Im(n) < 0`` under this repo's convention) is included because it is where
    the SMM stability argument does *not* apply — forward modes amplify there, so the propagator
    is no longer a contraction. It stays accurate anyway over this range; what it eventually does
    is return NaN rather than a wrong number, past an exponent of about 700.
    """
    kx, ky = _kxky(35.0)
    layers = [(blocks(np.diag([n ** 2] * 3).astype(complex)), thick)]
    R, T = solve_uniform_stack(WL, kx, ky, layers, 1, vacuum_sp_states(np, kx, ky, 1)[0])
    want_r, want_t = _airy_slab(n, 35.0, thick)

    assert R == pytest.approx(want_r, rel=1e-11), f'n={n}, d={thick}'
    assert T == pytest.approx(want_t, rel=1e-11), f'n={n}, d={thick}'


def test_the_transfer_refuses_before_it_becomes_inaccurate():
    """``connecting_algo='TMM'`` is a second opinion, so its bound must sit where it stops being
    one — not where it crashes.

    The two are different by three decades. The transfer's relative error grows like
    ``eps*exp(2x)`` in the modal exponent, so it is already 1.2e-8 out at ``x = 9`` and 8.5e-4 at
    ``x = 16``, while it does not overflow until about 700. A bound placed at the crash lets all
    of that through silently; this checks the bound is placed at the accuracy instead, by
    requiring that whatever TMM still accepts agrees with the cascade.
    """
    kx, ky = _kxky(35.0)
    absorbing = np.diag([(1.5 + 0.1j) ** 2] * 3).astype(complex)
    psi = vacuum_sp_states(np, kx, ky, 1)[0]

    # just inside the bound: TMM answers, and its answer is still worth having
    ok = [(blocks(absorbing), 7000.0)]
    smm = solve_uniform_stack(WL, kx, ky, ok, 1, psi)
    tmm = solve_uniform_stack(WL, kx, ky, ok, 1, psi, connecting_algo='TMM')
    for a, b in zip(smm, tmm):
        assert a == pytest.approx(b, rel=1e-6), 'TMM accepted a case it cannot do accurately'

    # past it: refused, and told where to go instead
    with pytest.raises(ValueError, match="connecting_algo='SMM'"):
        solve_uniform_stack(WL, kx, ky, [(blocks(absorbing), 30000.0)], 1, psi,
                            connecting_algo='TMM')


@pytest.mark.parametrize('theta_deg', [0.0, 35.0])
@pytest.mark.parametrize('label,eps', Z_DECOUPLED + Z_COUPLED,
                         ids=[c[0] for c in Z_DECOUPLED + Z_COUPLED])
def test_the_two_connecting_algorithms_agree_where_both_are_valid(label, eps, theta_deg):
    """Two ways of joining the same modes must give the same stack.

    The 2ff path makes this same comparison (``tests/`` covers TMM vs SMM there) and it means
    more here, because the two are *not* equivalent in reach: only one survives a thick layer. A
    disagreement in the overlapping range would say the interface matching or the mode ordering
    differs between them, which no external oracle would localise.
    """
    kx, ky = _kxky(theta_deg)
    layers = [(blocks(eps), THICK)]
    for psi, pol in zip(vacuum_sp_states(np, kx, ky, 1), ('s', 'p')):
        smm = solve_uniform_stack(WL, kx, ky, layers, 1, psi)
        tmm = solve_uniform_stack(WL, kx, ky, layers, 1, psi, connecting_algo='TMM')
        gap = max(abs(a - b) for a, b in zip(smm, tmm))
        assert gap < 1e-12, f'{label} at {theta_deg} deg, {pol}-pol: {gap:.2e}'


@pytest.mark.parametrize('theta_deg', [0.0, 35.0])
def test_a_multilayer_stack_matches_berreman(theta_deg):
    """The Redheffer cascade over more than one layer — untouched by every test above.

    Each single-layer case exercises two interfaces and one propagation. Only a stack exercises
    the star product joining layer to layer, and the ordering it has to keep: the mix below is
    deliberately asymmetric, so getting the stack backwards changes the answer.

    Measured, not assumed — traversing ``layers`` bottom-up leaves every other test in this file
    and every external golden passing, and fails only here.
    """
    from test_anisotropic_berreman import berreman_efficiencies

    stack = [(np.asarray(Z_DECOUPLED[1][1], dtype=complex), 140.0),
             (gyro((1, 0, 0)), 90.0),
             (np.diag([(1.4 + 0.05j) ** 2] * 3).astype(complex), 210.0),
             (rotated_uniaxial(np.deg2rad(52)), 175.0)]

    kx, ky = _kxky(theta_deg)
    layers = [(blocks(e), d) for e, d in stack]
    ref = berreman_efficiencies(stack, np.deg2rad(theta_deg))

    for psi, pol in zip(vacuum_sp_states(np, kx, ky, 1), ('s', 'p')):
        got = solve_uniform_stack(WL, kx, ky, layers, 1, psi)
        want = (ref[pol]['R_s'] + ref[pol]['R_p'], ref[pol]['T_s'] + ref[pol]['T_p'])
        gap = max(abs(g - w) for g, w in zip(got, want))
        assert gap < 1e-12, f'{theta_deg} deg, {pol}-pol: {gap:.2e}'
