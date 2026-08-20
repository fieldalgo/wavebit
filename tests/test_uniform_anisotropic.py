"""A *uniform* anisotropic layer, driven straight through the five convolution slots.

The 2D mode operator has read five permittivity blocks since the NVF factorization landed
(``epx, epy, epz_i, epxy, epyx`` — see ``_core/solve_2d.py::layer_modes_2d``), which makes the
tensor ``[[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]]`` solvable today even though no user-facing
input builds one: ``structure`` is a permittivity raster and becomes an isotropic ``eps``.
``docs/anisotropic-materials.md`` §4 says so and shows the call; this file is what keeps that
claim true.

The oracle is analytic, not a golden. A uniform layer at ``max_order=(0, 0)`` is a plane-wave problem,
so the answer is an Airy slab per eigen-polarisation — stated before the run, and independent of
how the solver reaches it.

What it pins, and none of it is pinned anywhere else:

* **Which slot is which.** ``epx`` carries ``exx`` and is what a TM (p) wave sees; ``epy`` carries
  ``eyy`` and belongs to TE (s). Swapping them conserves energy and stays plausible, so only a
  birefringence split with a known sign catches it.
* **That ``epz_i`` is real physics, not bookkeeping.** ``ezz`` may move the extraordinary branch
  and may not touch the ordinary one. At normal incidence it must do neither.
* **The off-diagonal sign convention.** ``_nvf_assemble`` returns ``-eps_xy`` into the ``epxy``
  slot, so a caller filling the slots by hand has to negate. Getting it backwards mirrors the
  optic axis about ``x``, and *every power efficiency stays bit-identical* — only the complex
  ``T_p`` changes sign. That is why the last test asserts on amplitudes; nothing built out of
  R/T numbers can see it.

The first two cases would pass with the off-diagonal slots ignored entirely, which is why the
rotated-axis tests are the ones that carry weight.
"""
import numpy as np
import pytest

import wavebit

WL = 600.0
D = 220.0
EO, EE = 2.25, 4.0          # ordinary / extraordinary permittivity


# ------------------------------------------------------------------ driver

def solve_uniform(exx, eyy, ezz, exy=0.0, eyx=0.0, *, theta=0.0, d=D, wl=WL):
    """One uniform anisotropic layer, fed to the 2D solve through ``solve_conv``.

    ``structure`` is left ``None``: there is no raster to build, and it also routes
    ``_assign_grating_type`` to the 2D path, which is the only one with off-diagonal slots.
    """
    I = np.eye(1, dtype=np.complex128)
    mee = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=theta, phi=0.0, psi=0.0,
                         max_order=(0, 0), period=(300.0, 300.0), wavelength=wl, thickness=[d])
    return mee.solve_conv(
        np.stack([exx * I]),
        np.stack([eyy * I]),
        np.stack([np.linalg.inv(ezz * I)]),
        np.stack([-exy * I]),          # slot convention: the block holds -eps_xy
        np.stack([-eyx * I]),
    )


# ------------------------------------------------------------------ analytic reference

def _airy(y0, y1, kz1, d, wl):
    """Amplitude ``(r, t)`` of a single slab from the two admittances and the inner ``kz``."""
    r01 = (y0 - y1) / (y0 + y1)
    t01 = 2 * y0 / (y0 + y1)
    r10 = -r01
    t10 = 2 * y1 / (y0 + y1)
    ph = np.exp(1j * (2 * np.pi / wl) * kz1 * d)
    denom = 1 + r01 * r10 * ph ** 2
    return r01 + t01 * t10 * r10 * ph ** 2 / denom, t01 * t10 * ph / denom


def slab_s(eps_yy, sin_t=0.0, d=D, wl=WL):
    """s-polarisation: ``E`` along y, so it sees ``eps_yy`` and nothing else. ``y = kz``."""
    kz0 = np.sqrt(1.0 - sin_t ** 2 + 0j)
    kz1 = np.sqrt(eps_yy - sin_t ** 2 + 0j)
    return _airy(kz0, kz1, kz1, d, wl)


def slab_p(eps_xx, eps_zz, sin_t=0.0, d=D, wl=WL):
    """p-polarisation: ``E`` in the xz plane. ``y = kz/eps``, and ``kz`` is extraordinary.

    With the optic axis along z the TM branch obeys ``kx**2/eps_zz + kz**2/eps_xx = k0**2``,
    which is the only place ``eps_zz`` can enter the problem at all.
    """
    kz0 = np.sqrt(1.0 - sin_t ** 2 + 0j)
    kz1 = np.sqrt(eps_xx * (1.0 - sin_t ** 2 / eps_zz) + 0j)
    return _airy(kz0 / 1.0, kz1 / eps_xx, kz1, d, wl)


def eff(r, t, cos_t=1.0):
    """Power efficiencies; superstrate and substrate are both vacuum here."""
    return abs(r) ** 2, abs(t) ** 2 * cos_t / cos_t


# ------------------------------------------------------------------ tests

def test_isotropic_tensor_reproduces_a_plain_slab():
    """Sanity floor: equal diagonal entries and no off-diagonal must be the isotropic answer."""
    res = solve_uniform(EO, EO, EO)
    r, t = slab_s(EO)
    R, T = eff(r, t)
    assert float(res.s_inc.T.sum()) == pytest.approx(T, abs=1e-12)
    assert float(res.p_inc.T.sum()) == pytest.approx(T, abs=1e-12)
    assert float(res.s_inc.R.sum()) == pytest.approx(R, abs=1e-12)


def test_diagonal_tensor_splits_te_and_tm():
    """``exx`` belongs to TM and ``eyy`` to TE — a swap would keep energy and be silent."""
    exx, eyy, ezz = EO, EE, EO
    res = solve_uniform(exx, eyy, ezz)

    R_s, T_s = eff(*slab_s(eyy))
    R_p, T_p = eff(*slab_p(exx, ezz))

    # The split has to be large, or the assignment is not actually being tested.
    assert abs(T_s - T_p) > 0.3

    assert float(res.s_inc.T.sum()) == pytest.approx(T_s, abs=1e-12)
    assert float(res.s_inc.R.sum()) == pytest.approx(R_s, abs=1e-12)
    assert float(res.p_inc.T.sum()) == pytest.approx(T_p, abs=1e-12)
    assert float(res.p_inc.R.sum()) == pytest.approx(R_p, abs=1e-12)


@pytest.mark.parametrize('theta_deg', [0.0, 35.0, 60.0])
def test_extraordinary_branch_sees_eps_zz(theta_deg):
    """``eps_zz`` enters the TM branch through the extraordinary dispersion, and only there."""
    th = np.deg2rad(theta_deg)
    st, ct = np.sin(th), np.cos(th)
    res = solve_uniform(EO, EO, EE, theta=th)

    R_s, T_s = eff(*slab_s(EO, st), ct)
    R_p, T_p = eff(*slab_p(EO, EE, st), ct)

    assert float(res.s_inc.T.sum()) == pytest.approx(T_s, abs=1e-12)
    assert float(res.s_inc.R.sum()) == pytest.approx(R_s, abs=1e-12)
    assert float(res.p_inc.T.sum()) == pytest.approx(T_p, abs=1e-12)
    assert float(res.p_inc.R.sum()) == pytest.approx(R_p, abs=1e-12)


@pytest.mark.parametrize('theta_deg', [0.0, 35.0])
def test_eps_zz_moves_tm_only(theta_deg):
    """The control for the test above: TE must be blind to ``eps_zz``, TM must not be.

    At normal incidence there is no ``E_z`` at all, so neither branch may move.
    """
    th = np.deg2rad(theta_deg)
    aniso = solve_uniform(EO, EO, EE, theta=th)
    iso = solve_uniform(EO, EO, EO, theta=th)

    te_a = float(aniso.s_inc.T.sum())
    te_i = float(iso.s_inc.T.sum())
    tm_a = float(aniso.p_inc.T.sum())
    tm_i = float(iso.p_inc.T.sum())

    assert te_a == pytest.approx(te_i, abs=1e-14), 'eps_zz must never reach the ordinary wave'
    if theta_deg == 0.0:
        assert tm_a == pytest.approx(tm_i, abs=1e-14), 'no E_z at normal incidence'
    else:
        assert abs(tm_a - tm_i) > 1e-4, 'eps_zz is not reaching the extraordinary wave'


@pytest.mark.parametrize('phi_deg', [15.0, 30.0, 55.0])
def test_rotated_axis_converts_polarization(phi_deg):
    """The off-diagonal slots, against Jones.

    Rotating the in-plane optic axis by ``phi`` makes ``eps_xy = (ee-eo) sin phi cos phi``. At
    normal incidence the two eigen-polarisations decouple, so each is a scalar slab and the lab
    frame answer is ``R(phi) diag(t_e, t_o) R(phi)^T`` — a waveplate. TE in, TM out is the only
    observable that carries the sign of ``eps_xy``.
    """
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    exx = EE * c ** 2 + EO * s ** 2
    eyy = EE * s ** 2 + EO * c ** 2
    exy = (EE - EO) * s * c

    res = solve_uniform(exx, eyy, EO, exy=exy, eyx=exy)

    r_e, t_e = slab_s(EE)
    r_o, t_o = slab_s(EO)
    rot = np.array([[c, -s], [s, c]])
    T_j = rot @ np.diag([t_e, t_o]) @ rot.T          # basis (x, y)
    R_j = rot @ np.diag([r_e, r_o]) @ rot.T

    incident_s = np.array([0.0, 1.0])                # TE incidence is E along y
    Ex_t, Ey_t = T_j @ incident_s
    Ex_r, Ey_r = R_j @ incident_s

    te = res.s_inc
    assert abs(Ex_t) ** 2 > 1e-3, 'no conversion to compare — pick a rotation off the axes'
    assert float(te.T_s.sum()) == pytest.approx(abs(Ey_t) ** 2, abs=1e-12)
    assert float(te.T_p.sum()) == pytest.approx(abs(Ex_t) ** 2, abs=1e-12)
    assert float(te.R_s.sum()) == pytest.approx(abs(Ey_r) ** 2, abs=1e-12)
    assert float(te.R_p.sum()) == pytest.approx(abs(Ex_r) ** 2, abs=1e-12)
    assert float(te.T.sum() + te.R.sum()) == pytest.approx(1.0, abs=1e-12)


def test_off_diagonal_sign_shows_up_in_the_amplitude_only():
    """Negating ``eps_xy`` mirrors the optic axis about x, and **no efficiency notices**.

    Every power efficiency — co-polarised, cross-polarised, total — is bit-identical under the
    flip, because ``|T_p|**2`` cannot see the mirror. Only the complex amplitude does: ``T_p``
    changes sign and ``T_s`` does not.

    So a caller who fills the slots with ``+eps_xy`` instead of ``-eps_xy`` gets an answer that
    is wrong in the transmitted *phase* while every R/T number stays right. Nothing that checks
    efficiencies alone — including the rest of this file — can catch it, which is the reason
    this test asserts on amplitudes.
    """
    phi = np.deg2rad(30.0)
    c, s = np.cos(phi), np.sin(phi)
    exx = EE * c ** 2 + EO * s ** 2
    eyy = EE * s ** 2 + EO * c ** 2
    exy = (EE - EO) * s * c

    right = solve_uniform(exx, eyy, EO, exy=exy, eyx=exy).s_inc
    flipped = solve_uniform(exx, eyy, EO, exy=-exy, eyx=-exy).s_inc

    for name in ('T_s', 'T_p', 'R_s', 'R_p'):
        assert float(getattr(right, name).sum()) == pytest.approx(
            float(getattr(flipped, name).sum()), abs=1e-14), f'{name} should be blind to the mirror'

    T_p_right = complex(right.t_p.sum())
    T_p_flipped = complex(flipped.t_p.sum())
    assert abs(T_p_right) > 1e-2, 'no cross-polarised amplitude to carry the sign'
    assert T_p_flipped == pytest.approx(-T_p_right, abs=1e-12), 'the mirror must flip T_p'
    assert complex(flipped.t_s.sum()) == pytest.approx(complex(right.t_s.sum()), abs=1e-12)


# ------------------------------------------------------------------ asymmetric tensors
#
# Everything above uses eps_xy == eps_yx, which is all any shipped path can produce:
# `_nvf_assemble`, `boundary.py` and `analytic.py` each return ``-eps_xy`` *twice*. The two
# blocks in ``layer_modes_2d`` disagree on the order of that pair, and equal entries hide it.
#
#     Omega2_LR carries  (epxy, epyx)
#     Omega_R   carries  (epyx, epxy)      <- the other way round
#
# A factorization ``Omega2_LR = Omega_L @ Omega_R`` must have an ``Omega_L`` built from the
# geometry alone. Symbolically it is not: with the shipped ordering ``Omega_L`` depends on
# ``epx, epy, epxy, epyx``, and exchanging the pair inside ``Omega_R`` alone reduces it to
# ``[[-Kx Ky epz_i, Kx^2 epz_i - 1], [-Ky^2 epz_i + 1, Kx Ky epz_i]]`` for any tensor.
#
# So the eigenvalues come from one tensor and the H-field from its transpose. No shipped path
# can reach it, which is why this is xfail rather than a bug report against a released number.

DELTA = 0.45          # gyrotropy strength


def jones_from_solver(res):
    """Transmitted 2x2 Jones matrix in the ``(x, y)`` basis, from both incidence channels.

    Calibrated on the symmetric cases above: for TE incidence (``E`` along y) the solver
    reports ``t_s = Ey`` and ``t_p = -Ex`` (the Fresnel p convention's normal-incidence
    sign — the same one that makes ``r_p = -r_s`` at a normal-incidence interface); TM
    incidence fills the other column and is ``E`` along ``-x`` in the same basis.
    """
    te, tm = res.s_inc, res.p_inc
    col_y = np.array([-complex(te.t_p.sum()), complex(te.t_s.sum())])
    col_x = -np.array([-complex(tm.t_p.sum()), complex(tm.t_s.sum())])
    return np.column_stack([col_x, col_y])


def faraday_jones(eps_d, gamma):
    """``eps = [[ed, i*g], [-i*g, ed]]`` -- circular eigenstates, a scalar slab on each.

    ``v+ = (1, -i)/sqrt2`` sees ``ed + g`` and ``v- = (1, +i)/sqrt2`` sees ``ed - g``; the
    interface preserves each, so every circular component is an independent Fabry-Perot.
    """
    _, t_plus = slab_s(eps_d + gamma)
    _, t_minus = slab_s(eps_d - gamma)
    P_plus = 0.5 * np.array([[1, 1j], [-1j, 1]])
    P_minus = 0.5 * np.array([[1, -1j], [1j, 1]])
    return t_plus * P_plus + t_minus * P_minus


def test_gyrotropic_layer_has_the_right_shape():
    """What *does* hold today: the answer is gyrotropic in form, just not in magnitude."""
    res = solve_uniform(EO, EO, EO, exy=+1j * DELTA, eyx=-1j * DELTA)
    J = jones_from_solver(res)
    assert J[0, 0] == pytest.approx(J[1, 1], abs=1e-12), 'diagonal must stay degenerate'
    assert J[0, 1] == pytest.approx(-J[1, 0], abs=1e-12), 'off-diagonal must be antisymmetric'
    assert float(res.s_inc.T.sum()
                 + res.s_inc.R.sum()) == pytest.approx(1.0, abs=1e-12)


def test_gyrotropic_layer_matches_faraday():
    """Faraday rotation through a gyrotropic slab, against the circular-eigenstate answer.

    This was ``xfail`` until ``layer_modes_2d`` was corrected — the error was ~4e-2 and first
    order in ``gamma``, so it did not wash out at weak gyrotropy either.
    """
    res = solve_uniform(EO, EO, EO, exy=+1j * DELTA, eyx=-1j * DELTA)
    assert np.abs(jones_from_solver(res) - faraday_jones(EO, DELTA)).max() < 1e-12
