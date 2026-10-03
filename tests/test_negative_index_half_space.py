"""A negative-index half-space is refused; the same medium as a layer is solved exactly.

``boundary_kz`` reads only ``eps*mu``, and for ``Re(eps) < 0`` and ``Re(mu) < 0`` with
``Re(eps*mu) > 0`` that product looks like an ordinary forward medium, so a half-space root
would be the wrong one. Glass onto the passive pair ``eps = -2+0.1j``, ``mu = -1+0.1j`` returned
``R = 929``, ``T = -928`` at normal incidence against a true ``0.0011``. A pair too lossy to
propagate keeps the right root and is accepted. See ``engine._refuse_negative_index_half_space``.
"""
import numpy as np
import pytest

import wavebit

PERIOD, WL, N_GLASS = 1000.0, 500.0, 1.5
PASSIVE = (-2 + 0.1j, -1 + 0.1j)
NEGATIVE_PAIRS = {
    'passive': PASSIVE,
    'lossless': (-2.0 + 0j, -1.0 + 0j),
    'gain': (-2 - 0.1j, -1 - 0.1j),
}


def _solver(top=N_GLASS ** 2, bot=N_GLASS ** 2, theta=0.0, algo='SMM', backend=0, **kw):
    """One uniform glass layer between the given half-spaces; the zeroth order only."""
    solver = wavebit.rcwa(backend=backend, top=top, bot=bot, theta=theta, phi=0.0, psi=0.0,
                          truncation_order=(0, 0), period=(PERIOD, PERIOD), wavelength=WL,
                          thickness=kw.pop('thickness', [50.0]), layer_algorithm=algo, **kw)
    solver.structure = np.full((1, 2, 2), N_GLASS ** 2, dtype=complex)
    return solver


@pytest.mark.parametrize('name', NEGATIVE_PAIRS)
@pytest.mark.parametrize('side', ['top', 'bot'])
def test_a_negative_index_half_space_is_refused(side, name):
    eps, mu = NEGATIVE_PAIRS[name]
    with pytest.raises(ValueError, match='negative-index half-space'):
        _solver(**{side: wavebit.Material(eps, mu=mu)})


@pytest.mark.parametrize('side', ['top', 'bot'])
def test_the_refused_setter_leaves_the_scene_as_it_was(side):
    """Both, or neither: the refusal runs before ``top``/``bot`` is written."""
    solver = _solver()
    before = (getattr(solver, side), getattr(solver, f'n_{side}'))
    with pytest.raises(ValueError, match='negative-index half-space'):
        setattr(solver, side, wavebit.Material(*PASSIVE[:1], mu=PASSIVE[1]))
    assert (getattr(solver, side), getattr(solver, f'n_{side}')) == before


def _interface_R(theta, eps, mu, pol):
    """Glass onto a passive half-space: the closed form, on the decaying root ``Im(kz) >= 0``."""
    kx = N_GLASS * np.sin(theta)
    k1 = np.sqrt(N_GLASS ** 2 - kx ** 2 + 0j)
    k2 = np.sqrt(complex(eps) * complex(mu) - kx ** 2)
    k2 = k2 if k2.imag >= 0 else -k2
    a, b = (k1, k2 / mu) if pol == 's' else (k1 / N_GLASS ** 2, k2 / eps)
    return abs((a - b) / (a + b)) ** 2


def _check_interface(eps, mu, theta=0.0, algo='SMM'):
    result = _solver(theta=theta, algo=algo, bot=wavebit.Material(eps, mu=mu)).solve()
    for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
        np.testing.assert_allclose(float(np.sum(sub.R)), _interface_R(theta, eps, mu, pol),
                                   atol=1e-12, err_msg=f'{pol}, eps={eps}, mu={mu}')


@pytest.mark.parametrize('eps, mu', [
    (-2 + 0.1j, 1.0),            # a metal: Re(eps) < 0 alone is an ordinary evanescent medium
    (2.0 + 0.1j, -1 + 0.1j),     # mu-negative alone: the same, through the other factor
    (-2 + 0.1j, 1.3 + 0.0j),     # a magnetic metal, so the comparison reads mu at all
    (-2 + 0.01j, 0.1 + 0.5j),    # eps*mu = -0.205-0.999j: below the axis, principal n reads as gain
])
def test_one_negative_real_part_is_not_refused(eps, mu):
    """Only both negative is the backward-wave medium. One alone keeps its root, and the
    reflectance agrees with the closed-form interface for s and p.

    The last pair is passive yet its ``n_bot`` has ``Im(n) < 0``: no branch is decided on
    ``n``, only on ``n**2``, which is what this pins (``engine._half_space_index``)."""
    _check_interface(eps, mu)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('eps', [-1 + 1.2j, -0.1 + 2j])
def test_a_lossy_double_negative_is_solved(eps, algo):
    """Both real parts negative but ``Re(eps*mu) <= 0``: nothing propagates at any ``k_par``,
    every root decays, and Maystre's is the decaying one. Accepted, and exact.

    Kills a refusal on the signs alone, which refused these with a false message."""
    for theta in (0.0, np.deg2rad(60.0)):
        _check_interface(eps, eps, theta=theta, algo=algo)


def test_solve_conv_half_space_terms_are_checked():
    """``solve_conv`` takes its own half-space terms; they need not be the scene's materials.

    Here the scene's substrate is the non-magnetic ``1.99-0.3j``, which is what the refused
    pair multiplies to; handed over as ``eps_bot`` / ``mu_bot`` it returned ``R = 928.78``."""
    eps, mu = PASSIVE
    solver = _solver(bot=complex(eps * mu))
    with pytest.raises(ValueError, match='negative-index half-space'):
        solver.solve_conv(*solver.eps_conv_matrices(), mu_bot=mu, eps_bot=eps)


def _airy(theta, d, eps, mu, pol):
    """Glass / slab / glass, the slab's admittance ``kz/mu`` (TE) or ``kz/eps`` (TM).

    Even in the slab's ``kz``, so the root does not matter -- which is the point."""
    kx = N_GLASS * np.sin(theta)
    k1 = np.sqrt(N_GLASS ** 2 - kx ** 2 + 0j)
    k2 = np.sqrt(complex(eps) * complex(mu) - kx ** 2)
    a, b = (k1, k2 / mu) if pol == 's' else (k1 / N_GLASS ** 2, k2 / eps)
    r12, r23 = (a - b) / (a + b), (b - a) / (b + a)
    t12, t23 = 2 * a / (a + b), 2 * b / (a + b)
    phase = np.exp(1j * k2 * 2 * np.pi / WL * d)
    den = 1 + r12 * r23 * phase ** 2
    return abs((r12 + r23 * phase ** 2) / den) ** 2, abs(t12 * t23 * phase / den) ** 2


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta_deg', [0.0, 60.0])
def test_a_negative_index_slab_matches_airy(theta_deg, algo):
    """The refusal's advice is true: the same passive pair as a 120 nm layer is exact.

    A layer keeps both roots, so its scattering matrix cannot depend on which one is labelled
    forward. Measured 1e-16 against the closed form at both angles and both polarizations."""
    theta, d = np.deg2rad(theta_deg), 120.0
    eps, mu = PASSIVE
    solver = _solver(theta=theta, algo=algo, thickness=[d])
    solver.structure = np.full((1, 2, 2), eps, dtype=complex)
    solver.mu = np.full((1, 2, 2), mu, dtype=complex)
    result = solver.solve()
    for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
        R, T = _airy(theta, d, eps, mu, pol)
        np.testing.assert_allclose([float(np.sum(sub.R)), float(np.sum(sub.T))], [R, T],
                                   atol=1e-12, err_msg=f'{pol} at {theta_deg} deg')


def test_the_refusal_reads_torch_values_eagerly():
    torch = pytest.importorskip('torch')
    eps = torch.tensor(PASSIVE[0], dtype=torch.complex128)
    mu = torch.tensor(PASSIVE[1], dtype=torch.complex128)
    with pytest.raises(ValueError, match='negative-index half-space'):
        _solver(backend=2, bot=wavebit.Material(eps, mu=mu))
