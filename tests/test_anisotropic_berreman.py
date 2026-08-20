"""Uniform anisotropic stacks against a Berreman 4x4 transfer matrix — an outside oracle.

``test_uniform_anisotropic.py`` checks the five-slot contract against closed forms that only
exist because the layer is uniform *and* the tensor is simple: an Airy slab per eigen-polarisation,
a Jones waveplate at normal incidence. Those run out exactly where the interesting cases start —
an asymmetric tensor at oblique incidence has no one-line answer.

This file supplies one, following

    D. W. Berreman, "Optics in Stratified and Anisotropic Media: 4x4-Matrix Formulation",
    J. Opt. Soc. Am. 62(4), 502 (1972).  doi:10.1364/JOSA.62.000502

which is the reference technique for stratified anisotropic media and was built with exactly this
case in mind — its 6x6 starting point is chosen so that Faraday rotation and optical activity are
expressible, so the gyrotropic tensors below are what it is *for*, not a stretch of it.

The method eliminates ``E_z`` and ``H_z`` and integrates the remaining transverse fields as a
first-order system,

    d/dz (Ex, Ey, Hx, Hy) = i k0 M (Ex, Ey, Hx, Hy)

so a uniform layer is a matrix exponential and a stack is their product. It shares no algebra with
RCWA: no Fourier orders, no eigenmodes, no s/p mode rotation, no scattering or transfer recursion.
Agreement is therefore evidence about the physics rather than about our own conventions — which
matters here, because the sign and ordering conventions are exactly what is in question.

Independently corroborated by RETICOLO (Hugonin and Lalanne, arXiv:2101.00901), whose V9+ toolbox
for arbitrarily anisotropic multilayer stacks reproduces the cases below to ~1e-15 with no
convention fitting — see ``docs/off-diagonal-slot-convention.md`` §8. That check needs Octave and
so does not run here; this file is the part that runs everywhere.

The oracle is checked before it is trusted (``test_oracle_agrees_where_the_solver_is_known_good``):
isotropic, absorbing, diagonal, biaxial, rotated-uniaxial and multilayer, at normal and oblique
incidence. Those all use a **symmetric** tensor, which is all any shipped path produces, and they
agree to ~1e-15.

What it pins beyond the existing file:

* **Oblique incidence with off-diagonal terms.** Nothing else covers it. The closed forms in
  ``test_uniform_anisotropic.py`` are normal-incidence only.
* **Stacks.** A single slab cannot expose an error in how layers are joined.
* **Asymmetric tensors** (``eps_xy != eps_yx``): the block below that was ``xfail`` until
  ``layer_modes_2d`` was corrected. It carried the off-diagonal pair in one order inside
  ``Omega2_LR`` and the other inside ``Omega_R``, so the modes came from one medium and the
  H-field from that medium's transpose. Measured against this oracle the error ran from 8e-3 to
  8e-1 depending on configuration; exchanging the pair inside ``Omega_R`` alone brought every case
  to ~1e-15. No shipped path could reach it — every producer emits ``-eps_xy`` twice.
* **A populated z row** (the Voigt geometry), which the second-order formulation cannot express
  at all and which the solver now answers on the first-order one. This is the only oracle that
  judges the *public* route there at oblique incidence with two off-diagonal pairs at once, and
  the only one that judges it from the back side — where the mirror has to flip the z row's sign
  and nothing else. RETICOLO covers a single pair; ``golden_fmmax.npz`` has no z row to give.

Note that the defect *is* visible in power efficiencies here, unlike the sign-convention error in
``test_uniform_anisotropic.py``, which only moves a complex amplitude. Those are different claims
about different mistakes; only the latter is invisible to R and T.

**Scope.** Every case in this file is a *uniform film*: no in-plane structure, ``max_order=(0, 0)``, one
Fourier order, convolution matrices hand-filled as scaled identities. Nothing that turns a pattern
into those matrices is exercised — not the ``fourier2d_*`` builders, not Li's rules, not NVF, not
the analytic or
boundary builders, not truncation. What this file covers is the layer mode operator, the s/p
rotation, the boundary match and the layer joining. Anisotropy on a *patterned* layer is untested
here and untestable here; it needs the factorization described in
``docs/anisotropic-materials.md`` §3, which does not exist yet.

And the oracle is an algorithm implemented below, not a third-party solver — see
``docs/off-diagonal-slot-convention.md`` §8 for why S4 was tried for that role and set aside.
"""
import numpy as np
import pytest
from scipy.linalg import expm

import wavebit

WL = 600.0


# ------------------------------------------------------------------ the oracle

def _berreman_M(eps, kx):
    """``M`` in ``dpsi/dz = i k0 M psi`` for ``psi = (Ex, Ey, Hx, Hy)``. After Berreman (1972).

    Derived for this repo's ``e^{-iwt}`` convention with fields going as
    ``exp(i k0 (kx x + kz z))``, so ``curl E = i k0 H`` and ``curl H = -i k0 eps E`` with ``H``
    carrying the free-space impedance. ``E_z`` follows from ``D_z = -kx Hy`` and ``H_z = kx Ey``;
    substituting both leaves the four transverse components below.
    """
    e = np.asarray(eps, dtype=complex)
    ezz = e[2, 2]
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = -kx * e[2, 0] / ezz
    M[0, 1] = -kx * e[2, 1] / ezz
    M[0, 3] = 1 - kx ** 2 / ezz
    M[1, 2] = -1.0
    M[2, 0] = -(e[1, 0] - e[1, 2] * e[2, 0] / ezz)
    M[2, 1] = kx ** 2 - e[1, 1] + e[1, 2] * e[2, 1] / ezz
    M[2, 3] = e[1, 2] * kx / ezz
    M[3, 0] = e[0, 0] - e[0, 2] * e[2, 0] / ezz
    M[3, 1] = e[0, 1] - e[0, 2] * e[2, 1] / ezz
    M[3, 3] = -e[0, 2] * kx / ezz
    return M


def _vacuum_modes(kx):
    """``psi`` of the s/p, forward/backward plane waves in vacuum."""
    kz = np.sqrt(1 - kx ** 2 + 0j)
    return (np.array([0, 1, -kz, 0], dtype=complex),      # s forward
            np.array([kz, 0, 0, 1], dtype=complex),       # p forward
            np.array([0, 1, kz, 0], dtype=complex),       # s backward
            np.array([-kz, 0, 0, 1], dtype=complex))      # p backward


def berreman_efficiencies(layers, theta, wl=WL):
    """``{'s': {...}, 'p': {...}}`` of power efficiencies, vacuum on both sides.

    ``layers`` is ``[(eps_3x3, thickness), ...]`` ordered along +z.
    """
    kx = np.sin(theta)
    k0 = 2 * np.pi / wl
    P = np.eye(4, dtype=complex)
    for eps, d in layers:
        P = expm(1j * k0 * _berreman_M(eps, kx) * d) @ P

    s_f, p_f, s_b, p_b = _vacuum_modes(kx)
    out = {}
    for name, inc in (('s', s_f), ('p', p_f)):
        A = np.column_stack([P @ s_b, P @ p_b, -s_f, -p_f])
        r_s, r_p, t_s, t_p = np.linalg.solve(A, -P @ inc)
        out[name] = dict(R_s=abs(r_s) ** 2, R_p=abs(r_p) ** 2,
                         T_s=abs(t_s) ** 2, T_p=abs(t_p) ** 2)
    return out


def berreman_efficiencies_from_below(layers, theta, wl=WL):
    """The same stack illuminated from the bottom — **without mirroring anything**.

    This exists to judge one specific piece of the solver: backside illumination is solved
    there as top-side illumination of the z-mirrored scene, and for a z-coupled tensor that
    mirror is not just a reversed layer list. ``z -> -z`` also sends ``eps_xz -> -eps_xz`` (and
    ``zx``, ``yz``, ``zy`` likewise) while leaving the in-plane block and ``eps_zz`` alone, so
    a solver that reverses the stack and forgets the signs answers for a different medium.

    Nothing is reversed or negated here. ``P`` is the same top-to-bottom transfer the function
    above builds; only the matching changes, because now the wave arrives at the bottom
    travelling ``-z`` (a *backward* mode), the reflection leaves downward and the transmission
    leaves upward through the top ::

        inc_b + r_s s_f + r_p p_f  =  P @ (t_s s_b + t_p p_b)

    Vacuum on both sides, so the incidence angle and ``kx`` are the same number either way and
    ``|r|^2`` / ``|t|^2`` are power directly. Powers rather than amplitudes on purpose: the
    ``p`` direction of an upward wave is a convention, and the two need not share it.
    """
    kx = np.sin(theta)
    k0 = 2 * np.pi / wl
    P = np.eye(4, dtype=complex)
    for eps, d in layers:
        P = expm(1j * k0 * _berreman_M(eps, kx) * d) @ P

    s_f, p_f, s_b, p_b = _vacuum_modes(kx)
    out = {}
    for name, inc in (('s', s_b), ('p', p_b)):
        A = np.column_stack([s_f, p_f, -P @ s_b, -P @ p_b])
        r_s, r_p, t_s, t_p = np.linalg.solve(A, -inc)
        out[name] = dict(R_s=abs(r_s) ** 2, R_p=abs(r_p) ** 2,
                         T_s=abs(t_s) ** 2, T_p=abs(t_p) ** 2)
    return out


# ------------------------------------------------------------------ the solver

def is_z_coupled(eps, tol=1e-12):
    """Whether the tensor's ``z`` row or column is populated — the Voigt geometry.

    What this decides for the adapter below is **which slots to fill**, and through that which
    formulation the solver picks. A z-free tensor is passed as five convolution matrices and
    runs on the second-order path, exactly as every case in this file did before the
    first-order one existed; a z-coupled one is passed as nine and runs on the first-order
    path, at about 8x. Passing the z slots unconditionally would have quietly moved every case
    here onto the new path and left the old one untested.
    """
    e = np.asarray(eps, dtype=complex)
    return max(abs(e[0, 2]), abs(e[2, 0]), abs(e[1, 2]), abs(e[2, 1])) > tol


def solve_tensor_stack(layers, theta, wl=WL, connecting_algo='ETM', formulation=None,
                       incidence_side='top'):
    """Drive the 2D solve with a hand-filled tensor per layer.

    ``structure`` stays ``None``: there is no raster, and it routes ``_assign_grating_type`` to
    the
    2D path, the only one with off-diagonal slots. **The slot convention is the negative of
    the tensor component, for every off-diagonal slot** — ``-eps_xy`` and equally ``-eps_xz``,
    one rule across all six — see ``docs/off-diagonal-slot-convention.md``.

    A tensor with a populated z row is handed over as nine slots rather than refused; the
    solver reads them and takes the first-order formulation. Passing ``formulation='2ff'``
    makes it refuse instead, which is what
    :func:`test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it` checks.
    """
    I = np.eye(1, dtype=np.complex128)
    z_coupled = any(is_z_coupled(eps) for eps, _ in layers)
    slots = [[] for _ in range(9)]
    thick = []
    for eps, d in layers:
        e = np.asarray(eps, dtype=complex)
        for slot, value in zip(slots, (e[0, 0], e[1, 1], None,
                                       -e[0, 1], -e[1, 0],
                                       -e[0, 2], -e[1, 2], -e[2, 0], -e[2, 1])):
            slot.append(np.linalg.inv(e[2, 2] * I) if value is None else value * I)
        thick.append(d)
    conv = [np.stack(s) for s in slots]
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=theta, phi=0.0, psi=0.0,
                         max_order=(0, 0), period=(300.0, 300.0), wavelength=wl, thickness=thick,
                         connecting_algo=connecting_algo, incidence_side=incidence_side)
    if formulation is not None:
        mee.formulation = formulation
    res = mee.solve_conv(*(conv if z_coupled else conv[:5]))
    out = {}
    for name, sub in (('s', res.res_te_inc), ('p', res.res_tm_inc)):
        out[name] = dict(R_s=float(sub.R_s.sum()), R_p=float(sub.R_p.sum()),
                         T_s=float(sub.T_s.sum()), T_p=float(sub.T_p.sum()))
    return out


def max_gap(a, b):
    return max(abs(a[pol][k] - b[pol][k])
               for pol in ('s', 'p') for k in ('R_s', 'R_p', 'T_s', 'T_p'))


# ------------------------------------------------------------------ tensors

def iso(e):
    return np.diag([e, e, e]).astype(complex)


def rotated_uniaxial(eo, ee, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[ee * c ** 2 + eo * s ** 2, (ee - eo) * s * c, 0],
                     [(ee - eo) * s * c, ee * s ** 2 + eo * c ** 2, 0],
                     [0, 0, eo]], dtype=complex)


def gyrotropic(ed, g):
    """Hermitian, hence lossless — the simplest asymmetric tensor there is."""
    return np.array([[ed, 1j * g, 0], [-1j * g, ed, 0], [0, 0, ed]], dtype=complex)


SYMMETRIC_CASES = [
    ('isotropic, normal', [(iso(2.25), 220.0)], 0.0),
    ('isotropic, 35 deg', [(iso(2.25), 220.0)], 35.0),
    ('absorbing, 20 deg', [(iso((0.18 + 3.0j) ** 2), 60.0)], 20.0),
    ('uniaxial along z, 35 deg', [(np.diag([2.25, 2.25, 4.0]).astype(complex), 220.0)], 35.0),
    ('biaxial, 55 deg', [(np.diag([2.1, 3.4, 2.8]).astype(complex), 180.0)], 55.0),
    ('rotated uniaxial, normal', [(rotated_uniaxial(2.25, 4.0, np.deg2rad(30)), 220.0)], 0.0),
    ('rotated uniaxial, 40 deg', [(rotated_uniaxial(2.25, 4.0, np.deg2rad(30)), 220.0)], 40.0),
    ('three-layer stack, 25 deg',
     [(iso(2.0), 90.0), (np.diag([2.25, 3.6, 2.9]).astype(complex), 150.0), (iso(4.0), 70.0)],
     25.0),
]


@pytest.mark.parametrize('label,layers,theta_deg', SYMMETRIC_CASES,
                         ids=[c[0] for c in SYMMETRIC_CASES])
def test_oracle_agrees_where_the_solver_is_known_good(label, layers, theta_deg):
    """Validate the oracle before trusting it, on tensors a shipped path could produce.

    Every case here is symmetric, so it is also a real check of the solver at oblique incidence
    with off-diagonal terms present — which nothing else covers.
    """
    gap = max_gap(berreman_efficiencies(layers, np.deg2rad(theta_deg)),
                  solve_tensor_stack(layers, np.deg2rad(theta_deg)))
    assert gap < 1e-12, f'{label}: oracle and solver disagree by {gap:.2e}'


ASYMMETRIC_CASES = [
    ('gyrotropic, normal', [(gyrotropic(2.25, 0.45), 220.0)], 0.0),
    ('gyrotropic, 45 deg', [(gyrotropic(2.25, 0.45), 220.0)], 45.0),
    ('gyrotropic, 75 deg', [(gyrotropic(2.25, 0.45), 220.0)], 75.0),
    ('gyro / isotropic / gyro, 33 deg',
     [(gyrotropic(2.25, 0.4), 120.0), (iso(2.9), 80.0), (gyrotropic(2.25, -0.4), 120.0)], 33.0),
]


@pytest.mark.parametrize('label,layers,theta_deg', ASYMMETRIC_CASES,
                         ids=[c[0] for c in ASYMMETRIC_CASES])
def test_asymmetric_tensor_matches_the_oracle(label, layers, theta_deg):
    """Oblique incidence, stacks and gyrotropy — none of which the closed forms elsewhere reach.

    These were ``xfail`` until ``layer_modes_2d`` was corrected: its two blocks had carried the
    off-diagonal pair in the *same* order, which is only consistent when the pair is symmetric.
    The error ran 8e-3 to 8e-1 depending on configuration.
    """
    gap = max_gap(berreman_efficiencies(layers, np.deg2rad(theta_deg)),
                  solve_tensor_stack(layers, np.deg2rad(theta_deg)))
    assert gap < 1e-12, f'{label}: oracle and solver disagree by {gap:.2e}'


@pytest.mark.parametrize('label,layers,theta_deg', ASYMMETRIC_CASES,
                         ids=[c[0] for c in ASYMMETRIC_CASES])
def test_scattering_algorithm_agrees_on_asymmetric_tensors(label, layers, theta_deg):
    """The same cases through SMM rather than ETM.

    ``layer_modes_2d`` is shared by both connecting algorithms, so the ordering correction
    reaches SMM for free — but "for free" is exactly the kind of claim that turns out to be
    false, and nothing else here drives SMM with an off-diagonal tensor at all.
    """
    theta = np.deg2rad(theta_deg)
    ref = berreman_efficiencies(layers, theta)
    got = solve_tensor_stack(layers, theta, connecting_algo='SMM')
    gap = max_gap(ref, got)
    assert gap < 1e-12, f'{label}: SMM is {gap:.2e} from the oracle'


def gyro_along(direction, g=0.45, ed=2.25):
    """``eps_ij = ed d_ij + i e_ijk g_k`` — the magnetisation direction picks the cells.

    ``M`` along z fills ``eps_xy`` / ``eps_yx`` (Faraday, expressible here); ``M`` in the plane
    fills the z row and column instead (Voigt, not expressible).
    """
    e = np.eye(3, dtype=complex) * ed
    gx, gy, gz = np.asarray(direction, dtype=float) * g
    e[0, 1] += 1j * gz
    e[1, 0] -= 1j * gz
    e[1, 2] += 1j * gx
    e[2, 1] -= 1j * gx
    e[2, 0] += 1j * gy
    e[0, 2] -= 1j * gy
    return e


VOIGT_CASES = [
    ('M along x', (1, 0, 0)),
    ('M along y', (0, 1, 0)),
    ('M tilted 45 deg', (np.sin(np.pi / 4), 0, np.cos(np.pi / 4))),
]


@pytest.mark.parametrize('label,direction', VOIGT_CASES, ids=[c[0] for c in VOIGT_CASES])
def test_voigt_geometry_matches_the_oracle(label, direction):
    """A magnetisation off the stack normal, answered rather than refused.

    This used to assert a ``NotImplementedError``: the five slots span
    ``[[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]]`` and nothing more, so the adapter refused
    what it could not carry. What made refusing the right answer — rather than dropping the
    z row — is that dropping it is *plausible*: the reduced tensor is a perfectly good medium,
    the solve succeeds, ``R + T`` sits at 1 to machine precision, and the efficiencies are
    wrong by 2.1e-2 for ``M`` along x and 6.2e-2 for ``M`` tilted 45 degrees. Nothing a caller
    is likely to check would notice.

    Now the solver has somewhere to put them (``formulation='4ff'``, selected automatically by
    the presence of the slots) and this is the check that it puts them in the *right* place.
    The oracle is not a rearrangement of the same code: Berreman integrates the first-order
    system directly with no Fourier orders, no mode split, no transfer recursion and no s/p
    rotation, and it takes the plain tensor while the solver takes the negated slots — so the
    slot sign convention is under test here too, in the one place where getting it backwards
    is *not* invisible to a power efficiency.

    ``M tilted 45 degrees`` populates ``xy`` and ``yz`` at once, which is what an ordinary
    magnetised film does and is the case RETICOLO cannot arbitrate (it stops conserving energy
    past one off-diagonal pair). Here it is judged.
    """
    layers = [(gyro_along(direction), 220.0)]
    gap = max_gap(berreman_efficiencies(layers, np.deg2rad(35)),
                  solve_tensor_stack(layers, np.deg2rad(35)))
    assert gap < 1e-12, f'{label}: oracle and solver disagree by {gap:.2e}'


@pytest.mark.parametrize('label,direction', VOIGT_CASES, ids=[c[0] for c in VOIGT_CASES])
def test_a_z_row_is_refused_by_the_formulation_that_cannot_hold_it(label, direction):
    """``formulation='2ff'`` must raise on a z row, not drop it.

    The guard moved from this file into the solver, which is where
    ``docs/anisotropic-materials.md`` §8 said it belonged once the solver could be handed a
    tensor directly. The measurement behind it is quoted in the test above.
    """
    with pytest.raises(ValueError, match='z row'):
        solve_tensor_stack([(gyro_along(direction), 220.0)], np.deg2rad(35),
                           formulation='2ff')


def test_faraday_geometry_stays_on_the_second_order_path():
    """The control: magnetisation along the stack normal has an empty z row.

    It must still agree with the oracle — and it must get there on the cheap path, which is
    the half of the dispatch that is easy to break by widening a signature.
    """
    layers = [(gyro_along((0, 0, 1)), 220.0)]
    gap = max_gap(berreman_efficiencies(layers, np.deg2rad(35)),
                  solve_tensor_stack(layers, np.deg2rad(35)))
    assert gap < 1e-12
    assert not is_z_coupled(gyro_along((0, 0, 1)))


@pytest.mark.parametrize('label,direction', VOIGT_CASES, ids=[c[0] for c in VOIGT_CASES])
def test_the_two_formulations_agree_on_a_stack_holding_both(label, direction):
    """A z-coupled layer and a z-free one in the same stack, against the oracle.

    A single slab cannot expose an error in how layers are joined, and the ETM recursion is
    the piece the first-order path had to grow for itself — the second-order one writes the
    two directions as ``big_W (I + Y)`` / ``big_V (I - Y)``, which assumes the backward modes
    are the forward ones with the partner field negated. That assumption is exact for a
    z-free layer and fails at 6.9e-1 for a Voigt tensor at 35 degrees, so this is the
    arrangement where reusing it would have shown.
    """
    layers = [(iso(2.0), 90.0), (gyro_along(direction), 150.0), (iso(4.0), 70.0)]
    gap = max_gap(berreman_efficiencies(layers, np.deg2rad(35)),
                  solve_tensor_stack(layers, np.deg2rad(35)))
    assert gap < 1e-12, f'{label}: oracle and solver disagree by {gap:.2e}'


@pytest.mark.parametrize('label,direction', VOIGT_CASES, ids=[c[0] for c in VOIGT_CASES])
def test_backside_illumination_of_a_z_coupled_stack(label, direction):
    """``incidence_side='bottom'`` on a tensor whose z row changes sign under the mirror.

    The oracle solves the un-mirrored stack from below directly, so this is a check of the
    mirror itself rather than a restatement of it. An asymmetric stack, because a symmetric one
    would pass with the sign flip missing.

    A magneto-optic medium is also the one place where "the two sides agree" is the wrong
    expectation: non-reciprocity is what the material is for, so the numbers here are not the
    top-side ones and must not be compared to them.
    """
    layers = [(iso(2.0), 90.0), (gyro_along(direction), 150.0), (iso(3.5), 60.0)]
    theta = np.deg2rad(35)
    gap = max_gap(berreman_efficiencies_from_below(layers, theta),
                  solve_tensor_stack(layers, theta, incidence_side='bottom'))
    assert gap < 1e-12, f'{label}: oracle and solver disagree by {gap:.2e}'


@pytest.mark.parametrize('label,direction', VOIGT_CASES, ids=[c[0] for c in VOIGT_CASES])
def test_the_two_sides_differ_by_exactly_what_the_oracle_says(label, direction):
    """Guards the test above from passing for the wrong reason — with a measured bound.

    A solver that ignored ``incidence_side`` altogether would satisfy the backside check on
    any stack whose two sides happen to agree, so the size of the disagreement is part of what
    has to be right. It cannot be asserted as a flat "> 0", because one of these three is
    genuinely reciprocal:

    ==================  ==========================
    magnetisation       two-sided gap, oracle
    ==================  ==========================
    ``M`` along x       1.4e-3
    ``M`` along y       1.2e-15 — reciprocal here
    ``M`` tilted 45     3.6e-2
    ==================  ==========================

    ``M`` along y with the plane of incidence at ``phi = 0`` is the transverse arrangement, and
    the oracle says so independently — which is why the bound is taken from it rather than
    written down. The other two are the reason this test is not vacuous.
    """
    layers = [(iso(2.0), 90.0), (gyro_along(direction), 150.0), (iso(3.5), 60.0)]
    theta = np.deg2rad(35)
    oracle = max_gap(berreman_efficiencies(layers, theta),
                     berreman_efficiencies_from_below(layers, theta))
    solver = max_gap(solve_tensor_stack(layers, theta),
                     solve_tensor_stack(layers, theta, incidence_side='bottom'))
    assert solver == pytest.approx(oracle, abs=1e-12), (
        f'{label}: the solver puts the two sides {solver:.2e} apart, the oracle {oracle:.2e}')


def test_the_oracle_conserves_energy_on_a_lossless_gyrotropic_slab():
    """Guards the oracle itself: a Hermitian tensor is lossless, so R + T must be 1.

    Without this, an error in the Berreman matrix could make the xfail above meaningless.
    """
    eff = berreman_efficiencies([(gyrotropic(2.25, 0.45), 220.0)], np.deg2rad(40))
    for pol in ('s', 'p'):
        total = eff[pol]['R_s'] + eff[pol]['R_p'] + eff[pol]['T_s'] + eff[pol]['T_p']
        assert total == pytest.approx(1.0, abs=1e-12)
