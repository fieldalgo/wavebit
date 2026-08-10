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
* **Asymmetric tensors** (``eps_xy != eps_yx``): the xfail block below. ``layer_modes_2d`` carries
  the off-diagonal pair in one order inside ``Omega2_LR`` and the other inside ``Omega_R``, so the
  modes come from one medium and the H-field from its transpose. Measured against this oracle the
  error runs from 8e-3 to 8e-1 depending on configuration; exchanging the pair inside ``Omega_R``
  alone brings every case to ~1e-15. No shipped path can reach it — every producer emits
  ``-eps_xy`` twice — so these are xfail rather than a bug against a released number.

Note that the defect *is* visible in power efficiencies here, unlike the sign-convention error in
``test_uniform_anisotropic.py``, which only moves a complex amplitude. Those are different claims
about different mistakes; only the latter is invisible to R and T.

**Scope.** Every case in this file is a *uniform film*: no in-plane structure, ``fto=(0, 0)``, one
Fourier order, convolution matrices hand-filled as scaled identities. Nothing that turns a pattern
into those matrices is exercised — not ``dfs2d``, not Li's rules, not NVF, not the analytic or
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

import fieldwave

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


# ------------------------------------------------------------------ the solver

def solve_tensor_stack(layers, theta, wl=WL, connecting_algo='TMM'):
    """Drive the 2D solve with a hand-filled tensor per layer.

    ``ucell`` stays ``None``: there is no raster, and it routes ``_assign_grating_type`` to the
    2D path, the only one with off-diagonal slots. The slot convention is ``-eps_xy`` — see
    ``docs/off-diagonal-slot-convention.md``.
    """
    I = np.eye(1, dtype=np.complex128)
    epx, epy, epz_i, epxy, epyx, thick = [], [], [], [], [], []
    for eps, d in layers:
        e = np.asarray(eps, dtype=complex)
        epx.append(e[0, 0] * I)
        epy.append(e[1, 1] * I)
        epxy.append(-e[0, 1] * I)
        epyx.append(-e[1, 0] * I)
        epz_i.append(np.linalg.inv(e[2, 2] * I))
        thick.append(d)
    mee = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.0, theta=theta, phi=0.0, psi=0.0,
                         fto=(0, 0), period=(300.0, 300.0), wavelength=wl, thickness=thick,
                         connecting_algo=connecting_algo)
    res = mee.solve_for_conv(wl, np.stack(epx), np.stack(epy), np.stack(epz_i),
                             np.stack(epxy), np.stack(epyx))
    out = {}
    for name, sub in (('s', res.res_te_inc), ('p', res.res_tm_inc)):
        out[name] = dict(R_s=float(sub.de_ri_s.sum()), R_p=float(sub.de_ri_p.sum()),
                         T_s=float(sub.de_ti_s.sum()), T_p=float(sub.de_ti_p.sum()))
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
    """The same cases through SMM rather than TMM.

    ``layer_modes_2d`` is shared by both connecting algorithms, so the ordering correction
    reaches SMM for free — but "for free" is exactly the kind of claim that turns out to be
    false, and nothing else here drives SMM with an off-diagonal tensor at all.
    """
    theta = np.deg2rad(theta_deg)
    ref = berreman_efficiencies(layers, theta)
    got = solve_tensor_stack(layers, theta, connecting_algo='SMM')
    gap = max_gap(ref, got)
    assert gap < 1e-12, f'{label}: SMM is {gap:.2e} from the oracle'


def test_the_oracle_conserves_energy_on_a_lossless_gyrotropic_slab():
    """Guards the oracle itself: a Hermitian tensor is lossless, so R + T must be 1.

    Without this, an error in the Berreman matrix could make the xfail above meaningless.
    """
    eff = berreman_efficiencies([(gyrotropic(2.25, 0.45), 220.0)], np.deg2rad(40))
    for pol in ('s', 'p'):
        total = eff[pol]['R_s'] + eff[pol]['R_p'] + eff[pol]['T_s'] + eff[pol]['T_p']
        assert total == pytest.approx(1.0, abs=1e-12)
