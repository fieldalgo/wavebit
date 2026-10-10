"""A negative-index half-space: solved on either side, lit or not; refused where it is active.

``boundary_kz`` used to see only ``eps*mu``, and for ``Re(eps) < 0`` and ``Re(mu) < 0`` with
``Re(eps*mu) > 0`` that product looks like an ordinary forward medium, so the half-space took
the root that advances its phase away from the interface: glass onto the passive pair
``eps = -2+0.1j``, ``mu = -1+0.1j`` returned ``R = 929``, ``T = -928`` at normal incidence
against a true ``0.0011``. A double negative now takes the mirror of Maystre's cut, the
decaying root (``docs/branch-cuts.md`` section 9), and a magnetic half-space's index is
``sqrt(eps)*sqrt(mu)`` (``conventions.signed_index``), negative for a negative-phase-velocity
medium. Lit from such a medium, ``theta`` / ``phi`` name the direction the incident light
travels: ``k_par = s*N*sin(theta)``, ``s = sign(Re(n))``.

The oracles here need no root rule of their own: closed-form Fresnel coefficients on the
decaying root, the Poynting vector of the field the solve returns, and a thick lossy slab of
the same medium, which as a layer keeps both roots. Every lit-side case uses an asymmetric
medium (``|mu| != 1``) and an oblique azimuth where it has one, so a sign-blind answer or a
twin substituted for the medium cannot pass.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import boundary_kz, gain_cutoff_band, signed_index

PERIOD, WL, N_GLASS = 1000.0, 500.0, 1.5
PASSIVE = (-2 + 0.1j, -1 + 0.1j)
LOSSLESS = (-2 + 0j, -1 + 0j)
HIGH = (-6 + 0.3j, -1.5 + 0.05j)
GAIN = (-2 - 0.1j, -1 - 0.1j)
GAIN_IN_MU = (-2 + 0.1j, -1 - 0.01j)
ASYM = (-3 + 0.1j, -0.5 + 0.05j)
ASYM_LOSSLESS = (-3 + 0j, -0.5 + 0j)
D_LAYER = 37.0


def _host(v):
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v)


def _solver(top=N_GLASS ** 2, bot=N_GLASS ** 2, theta=0.0, algo='SMM', backend='numpy', path='2d',
            side='top', phi=None, **kw):
    """One uniform glass layer between the given half-spaces; the zeroth order only."""
    common = dict(backend=backend, top=top, bot=bot, theta=theta, wavelength=WL,
                  thickness=kw.pop('thickness', [D_LAYER]), layer_algorithm=algo,
                  incidence_side=side, **kw)
    if path in ('te', 'tm'):
        return wavebit.rcwa(**common, tetm=path, truncation_order=0, period=[PERIOD],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    if path == 'conical':
        return wavebit.rcwa(**common, phi=0.4 if phi is None else phi, psi=0.0,
                            truncation_order=0, period=[PERIOD],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    solver = wavebit.rcwa(**common,
                          phi=(0.4 if path == 'first_order' else 0.0) if phi is None else phi,
                          psi=0.0, truncation_order=(0, 0), period=(PERIOD, PERIOD))
    solver.structure = np.full((1, 2, 2), N_GLASS ** 2, dtype=complex)
    if path == 'first_order':
        solver.formulation = 'first_order'
    return solver


def _decaying_root(eps, mu, k_par):
    """The far side's root with ``Im(kz) > 0``; on a lossless order, ``Re(kz/mu) > 0``."""
    kz = np.sqrt(complex(eps) * complex(mu) - k_par ** 2 + 0j)
    if kz.imag < 0 or (kz.imag == 0 and (kz / mu).real < 0):
        kz = -kz
    return kz


def _interface(theta, eps, mu):
    """Glass onto ``(eps, mu)`` through ``D_LAYER`` of glass: ``{pol: (r, t, R, T)}``.

    s: ``Y = kz/mu``. p: ``Z = kz/eps`` on the magnetic amplitude, ``r_p`` the H ratio and
    ``t_p`` the E ratio, ``t_H (n1/mu1)/(n2/mu2)`` with ``n2 = sqrt(eps)*sqrt(mu)``.
    """
    k_par = N_GLASS * np.sin(theta)
    k1 = np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
    k2 = _decaying_root(eps, mu, k_par)
    n2 = np.sqrt(complex(eps)) * np.sqrt(complex(mu))
    phase = np.exp(2j * np.pi / WL * k1 * D_LAYER)
    a, b = k1, k2 / mu
    r, t = (a - b) / (a + b), 2 * a / (a + b)
    s = (r * phase ** 2, t * phase, abs(r) ** 2, abs(t) ** 2 * b.real / a.real)
    a, b = k1 / N_GLASS ** 2, k2 / eps
    r, t = (a - b) / (a + b), 2 * a / (a + b)
    p = (r * phase ** 2, t * N_GLASS / (n2 / mu) * phase, abs(r) ** 2,
         abs(t) ** 2 * b.real / a.real)
    return {'s': s, 'p': p}


def _check_interface(eps, mu, path='2d', algo='SMM', side='top', thetas=(0., 30., 60., 80.)):
    half = wavebit.Material(eps, mu=mu)
    scene = dict(bot=half) if side == 'top' else dict(top=half, bot=N_GLASS ** 2)
    for theta_deg in thetas:
        theta = np.deg2rad(theta_deg)
        result = _solver(theta=theta, algo=algo, path=path, side=side, **scene).solve()
        expected = _interface(theta, eps, mu)
        subs = {'te': {'s': result.incident}, 'tm': {'p': result.incident}}.get(
            path, {'s': result.s_inc, 'p': result.p_inc})
        for pol, sub in subs.items():
            r, t = (sub.r_s, sub.t_s) if pol == 's' else (sub.r_p, sub.t_p)
            got = [complex(np.sum(_host(r))), complex(np.sum(_host(t))),
                   float(np.sum(_host(sub.R))), float(np.sum(_host(sub.T)))]
            np.testing.assert_allclose(got, expected[pol], rtol=0, atol=1e-12,
                                       err_msg=f'{pol} at {theta_deg} deg, {path} {algo}')


def _phase_index(eps, mu, theta):
    """``N = |Re(k)|/k0`` at a real angle: ``N**2 = (a + hypot(a, b/cos(theta)))/2``."""
    a, b = (eps * mu).real, (eps * mu).imag
    return np.sqrt((a + np.hypot(a, b / np.cos(theta))) / 2)


def _lit_interface(theta, eps, mu):
    """``(eps, mu)`` lit at ``theta`` onto glass through ``D_LAYER`` of glass.

    The incidence is the one ``theta`` names: ``k_par = s*N*sin(theta)``, ``s`` the sign of
    ``Re(sqrt(eps)*sqrt(mu))``, and the incident ``kz`` the decaying root (``Re(kz) < 0`` for a
    lossless double negative: the phase recedes while the power arrives). ``t_p`` is the E ratio,
    ``t_H (n1/mu1)/(n2/mu2)`` with the signed ``n1``. Returns ``{pol: (r, t, R, T)}``."""
    n1 = np.sqrt(complex(eps)) * np.sqrt(complex(mu))
    k_par = np.sign(n1.real) * _phase_index(eps, mu, theta) * np.sin(theta)
    k1, k2 = _decaying_root(eps, mu, k_par), np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
    phase = np.exp(2j * np.pi / WL * k2 * D_LAYER)
    a, b = k1 / mu, k2
    r, t = (a - b) / (a + b), 2 * a / (a + b)
    s = (r, t * phase, abs(r) ** 2, abs(t) ** 2 * b.real / a.real)
    a, b = k1 / eps, k2 / N_GLASS ** 2
    r, t = (a - b) / (a + b), 2 * a / (a + b)
    p = (r, t * (n1 / mu) / N_GLASS * phase, abs(r) ** 2, abs(t) ** 2 * b.real / a.real)
    return {'s': s, 'p': p}


def _check_lit_interface(eps, mu, path='2d', algo='SMM', side='top',
                         thetas=(0., 20., 35., 60.)):
    half = wavebit.Material(eps, mu=mu)
    scene = dict(top=half) if side == 'top' else dict(bot=half)
    for theta_deg in thetas:
        theta = np.deg2rad(theta_deg)
        result = _solver(theta=theta, algo=algo, path=path, side=side, phi=0.0, **scene).solve()
        expected = _lit_interface(theta, eps, mu)
        subs = {'te': {'s': result.incident}, 'tm': {'p': result.incident}}.get(
            path, {'s': result.s_inc, 'p': result.p_inc})
        for pol, sub in subs.items():
            r, t = (sub.r_s, sub.t_s) if pol == 's' else (sub.r_p, sub.t_p)
            got = [complex(np.sum(_host(r))), complex(np.sum(_host(t))),
                   float(np.sum(_host(sub.R))), float(np.sum(_host(sub.T)))]
            np.testing.assert_allclose(got, expected[pol], rtol=0, atol=1e-12,
                                       err_msg=f'{pol} at {theta_deg} deg, {path} {algo}')


# --------------------------------------------------------------------- the far side, solved

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'first_order'])
def test_a_negative_index_far_side_matches_the_closed_form(path, side, algo):
    """``r``, ``t``, ``R`` and ``T`` for s and p, 0 to 80 degrees, lit from either side.

    Measured 3.2e-15 at worst. Under Maystre's root the same scene returned ``R = 929``.
    ``t`` reads the index's sign, so it also pins ``n = -sqrt(eps*mu)``."""
    _check_interface(*PASSIVE, path=path, algo=algo, side=side)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('pair', [LOSSLESS, HIGH, (-1 + 0.01j, -1 + 0.01j)],
                         ids=['lossless', 'high_index', 'impedance_matched'])
def test_other_negative_index_media_match_the_closed_form(pair, algo):
    """The lossless pair, where every propagating order has ``Re(kz) < 0`` and no loss to decide
    the root (80 degrees is beyond its critical angle from glass), and two others."""
    _check_interface(*pair, algo=algo)


def test_the_index_is_negative_and_t_p_follows_t_s():
    """``n_bot = -sqrt(eps*mu)``: Veselago's sign, and ``sqrt(eps)*sqrt(mu)`` for a passive pair.

    The sign decides ``t_p``: at normal incidence ``t_p == t_s`` as for any medium. With the
    principal root it was ``-t_s``. Kills a kernel fix that leaves ``_half_space_index`` alone."""
    eps, mu = PASSIVE
    solver = _solver(bot=wavebit.Material(eps, mu=mu))
    assert solver.n_bot == pytest.approx(np.sqrt(eps) * np.sqrt(mu), abs=1e-15)
    assert solver.n_bot.real < 0
    result = solver.solve()
    t_s, t_p = complex(np.sum(result.s_inc.t_s)), complex(np.sum(result.p_inc.t_p))
    assert abs(t_s - t_p) < 1e-14 and abs(t_s) > 0.5


def test_a_passive_far_side_says_nothing():
    """No gain, so no ``GainCutoffWarning``, and its root decision is at least ``|kz|`` from
    zero, so no ``UnresolvedBranchWarning`` -- across 0 to 89 degrees, lossless included."""
    for pair in (PASSIVE, LOSSLESS):
        for theta_deg in np.linspace(0.0, 89.0, 9):
            solver = _solver(theta=np.deg2rad(theta_deg), bot=wavebit.Material(*pair))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                solver.solve()
            assert not [w for w in caught if issubclass(
                w.category, (wavebit.GainCutoffWarning, wavebit.UnresolvedBranchWarning))]
            assert not np.asarray(solver.gain_cutoff_band()['bottom']).any()


@pytest.mark.parametrize('eps, mu', [
    (-2 + 0.1j, 1.0),            # a metal: Re(eps) < 0 alone is an ordinary evanescent medium
    (2.0 + 0.1j, -1 + 0.1j),     # mu-negative alone: the same, through the other factor
    (-2 + 0.1j, 1.3 + 0.0j),     # a magnetic metal, so the comparison reads mu at all
    (-2 + 0.01j, 0.1 + 0.5j),    # eps*mu = -0.205-0.999j: below the axis, principal n reads as gain
])
def test_one_negative_real_part_keeps_maystres_root(eps, mu):
    """Only both negative is the backward-wave medium. One alone keeps Maystre's root, and
    ``r``, ``t``, ``R`` and ``T`` agree with the closed-form interface for s and p.

    The last pair is passive and negative-phase-velocity (``Im(eps*mu) < 0``): its index is
    ``sqrt(eps)*sqrt(mu) = -0.638+0.783j``, the negative of the principal root of the product,
    which reads as gain. ``t_p`` reads that sign (``conventions.signed_index``); with the
    principal root it was ``-t_s`` at normal incidence."""
    _check_interface(eps, mu, thetas=(0.0, 30.0, 60.0))


def test_the_signed_index_is_the_product_of_the_two_roots():
    """``signed_index(eps, mu) == sqrt(eps)*sqrt(mu)`` on passive pairs of every sign, the
    principal root of ``eps*mu`` itself -- to the bit -- wherever that root keeps
    ``Im(n) >= 0``, and its exact negation where it does not.

    The pairs placed on the boundary of the negative-phase-velocity class (``eps*mu`` on the
    negative real axis, to within a few ulp) are where the sign of ``Re(sqrt(eps)*sqrt(mu))``
    alone is round-off: read that way, 18% of them came back with the gain-reading root."""
    rng = np.random.default_rng(11)
    eps = rng.uniform(-5, 5, 4000) + 1j * rng.uniform(0, 2, 4000)
    mu = rng.uniform(-5, 5, 4000) + 1j * rng.uniform(0, 2, 4000)
    mu[::4] = mu[::4].real + 0j                               # lossless factors, both signs
    eps[1::4] = eps[1::4].real + 0j
    # The class boundary: Im(eps*mu) = 0 with Re(eps*mu) < 0, nudged by a few ulp either way.
    e_b = rng.uniform(-5, 5, 2000) + 1j * rng.uniform(0.01, 2, 2000)
    m_re = rng.uniform(-5, 5, 2000)
    m_im = -e_b.imag * m_re / e_b.real
    keep = m_im > 0
    e_b, m_re, m_im = e_b[keep], m_re[keep], m_im[keep]
    m_b = m_re + 1j * (m_im + rng.integers(-3, 4, m_im.size) * np.spacing(m_im))
    eps, mu = np.concatenate([eps, e_b]), np.concatenate([mu, m_b])
    n = signed_index(eps, mu)
    product = np.sqrt(eps + 0j) * np.sqrt(mu + 0j)
    np.testing.assert_allclose(n, product, rtol=1e-12, atol=0)
    principal = np.sqrt(eps * mu)
    same = (n == principal)
    assert np.all(n[~same] == -principal[~same])
    assert np.all((n.view(float) == principal.view(float)).reshape(-1, 2).all(axis=1) == same)
    assert np.all(principal[same].imag >= 0) and np.any(~same)
    # Python numbers: through cmath, a Python complex back, the same root.
    for e, m in [(-2 + 0.1j, -1 + 0.1j), (-2 + 0.01j, 0.1 + 0.5j), (2.25, 1.3), (-2.0, -1.0)]:
        got = signed_index(e, m)
        assert type(got) is complex
        assert got == pytest.approx(np.sqrt(complex(e)) * np.sqrt(complex(m)), abs=1e-15)
    # A signed zero in a lossless factor is lossless, not gain.
    assert signed_index(complex(-2.0, -0.0), -1.0 + 0j).real < 0


@pytest.mark.parametrize('eps', [-(4 + 0j), complex(-3.0, -0.0), np.complex128(-(4 + 0j)),
                                 -4 + 0j, 2.25, -2 + 0.1j, np.complex128(2 + 0.1j)])
def test_naming_mu_one_keeps_the_non_magnetic_index_and_amplitudes(eps):
    """``Material(eps, mu=1.0)`` takes the permeability path and must compute the numbers the
    non-magnetic path does. The hard inputs are lossless metals written with a signed zero,
    ``-(4+0j) == -4-0j``: ``eps*mu`` sits exactly on the negative real axis, where the principal
    root ``-2j`` and its negation are equally lossless. The signed index keeps the principal root
    there (``conventions.signed_index``), as ``sqrt(eps)`` does; turned to ``+2j`` it flipped
    ``t_p`` against the non-magnetic solve of the same medium. Also pinned: ``n_bot`` keeps the
    type the principal root gave it, a NumPy scalar for a NumPy scalar."""
    out = {}
    for name, bot in (('plain', eps), ('named', wavebit.Material(eps, mu=1.0))):
        solver = _solver(bot=bot, theta=0.3, path='tm', algo='ETM')
        result = solver.solve()
        out[name] = (solver.n_bot, _host(result.incident.t_p), _host(result.incident.r_p))
    plain, named = out['plain'], out['named']
    assert type(named[0]) is type(plain[0])
    assert np.asarray(named[0]).tobytes() == np.asarray(plain[0]).tobytes()
    for a, b in zip(plain[1:], named[1:]):
        np.testing.assert_allclose(b, a, rtol=1e-14, atol=1e-15)


def test_on_the_axis_the_signed_index_is_the_principal_root():
    """Lossless ``eps*mu`` exactly on the negative real axis, every placement of signed zeros:
    no wave propagates, ``n`` is imaginary either way, and the index stays the principal
    ``sqrt(eps*mu)`` it always was. A lossless double negative written with signed zeros is
    still ``-sqrt(2)``, and a lossy medium that lands on the axis still decays."""
    zeros = (0.0, -0.0)
    for e, m in ((-4.0, 1.0), (-4.0, 2.0), (2.0, -1.0), (0.5, -0.5)):
        for ze in zeros:
            for zm in zeros:
                eps, mu = complex(e, ze), complex(m, zm)
                principal = np.sqrt(np.complex128(eps * mu))
                for got in (signed_index(eps, mu),
                            signed_index(np.asarray([eps]), np.asarray([mu]))[0]):
                    assert np.asarray(got).tobytes() == np.asarray(principal).tobytes(), (eps, mu)
    for ze in zeros:
        for zm in zeros:
            assert signed_index(complex(-2.0, ze), complex(-1.0, zm)) == -np.sqrt(2.0)
    assert signed_index(-2 + 0.1j, 1 + 0.05j).imag > 0       # Im(eps*mu) = +0 exactly


def _maystre_root(z):
    """Maystre's root of each factor, written out: principal, negated where ``Re + Im < 0``."""
    r = np.sqrt(np.asarray(z, dtype=complex) + 0j)
    return np.where(r.real + r.imag < 0, -r, r)


def test_each_factors_root_is_maystres():
    """``n = sqrt(eps)*sqrt(mu)`` with each factor's root on Maystre's cut (A. A. Govyadinov,
    V. A. Podolskiy and M. A. Noginov, Appl. Phys. Lett. 91, 191103 (2007)), steering the
    principal ``sqrt(eps*mu)``. It is the principal-factor rule to the bit for every medium
    without exactly one gain-metal factor (``Re < 0``, ``Im < 0``) -- every passive medium among
    them -- and its exact negation for those, on the array path and on Python numbers alike, and
    on JAX and PyTorch arrays."""
    rng = np.random.default_rng(29)
    size = 20000
    eps = rng.uniform(-3, 3, size) + 1j * rng.uniform(-1, 1, size)
    mu = rng.uniform(-3, 3, size) + 1j * rng.uniform(-1, 1, size)
    eps[::7] = eps[::7].real + 0j                              # lossless factors, both signs
    mu[1::7] = mu[1::7].real + 0j
    r = np.sqrt(eps * mu)
    turn = lambda q: np.where(((np.conj(r) * q).real < 0) & (r.real != 0), -r, r)
    want = turn(_maystre_root(eps) * _maystre_root(mu))
    before = turn(np.sqrt(eps + 0j) * np.sqrt(mu + 0j))
    n = signed_index(eps, mu)
    assert n.tobytes() == want.tobytes()
    gain_metal = lambda z: (z.real < 0) & (z.imag < 0)
    one = gain_metal(eps) ^ gain_metal(mu)
    assert 0.2 < one.mean() < 0.5
    assert n[~one].tobytes() == before[~one].tobytes()
    assert np.all(n[one] == -before[one])
    for i in range(0, size, 97):
        got = signed_index(complex(eps[i]), complex(mu[i]))
        assert type(got) is complex and (np.conj(got) * n[i]).real > 0
    # A factor on Maystre's cut itself, sqrt(-2j) = 1-1j exactly: its tie keeps the root, as
    # `boundary_kz`'s does, so the index points along the half-space's kz there too.
    for m in (1.0 + 0j, 2.25 + 0j):
        kz = boundary_kz(np, np.asarray([-2j * m]), np.zeros(1), medium=(-2j, m))[0]
        for got in (signed_index(-2j, m), signed_index(np.asarray([-2j]), np.asarray([m]))[0]):
            assert got == kz
    for backend in ('jax', 'torch'):
        lib = pytest.importorskip(backend)
        if backend == 'jax':
            wavebit.enable_x64()
            got = np.asarray(signed_index(lib.numpy.asarray(eps), lib.numpy.asarray(mu)))
        else:
            got = signed_index(lib.tensor(eps), lib.tensor(mu)).numpy()
        assert np.all((np.conj(got) * n).real > 0), backend


def test_a_factor_on_maystres_cut_keeps_its_root_whatever_the_round_off():
    """A factor exactly on the negative imaginary axis (``Re = 0``, ``Im < 0``) has both roots on
    Maystre's cut, ``Re + Im = 0``, and the tie keeps the principal one. The computed sum is
    round-off of either sign, by backend: reading it turned the index of ``(-0.1j, -3)`` -- no
    double negative, no gain-metal factor -- against the ``kz`` it is solved with, so ``t_p`` was
    ``-t_s`` at normal incidence on NumPy, JAX and Python numbers and ``+t_s`` on PyTorch, and
    with ``mu = -3`` so it went for 67 of the factors ``-0.01j .. -3.00j``. The decision is now
    read off the factor (``conventions._maystre_factor``): the index points along ``kz`` for
    every one of them, on every backend, either factor on the axis, and ``t_p = t_s``."""
    b = np.round(np.arange(1, 301) * 0.01, 2)
    axis = np.asarray([complex(0.0, -x) for x in b])
    other = np.full_like(axis, -3 + 0j)
    kz = boundary_kz(np, axis * other, np.zeros(axis.size), medium=(axis, other))
    assert np.all(kz.real > 0) and np.all(kz.imag > 0)       # kz**2 = 3bi: no tie in kz
    for eps, mu in ((axis, other), (other, axis)):
        got = {'numpy': signed_index(eps, mu),
               'python': np.asarray([signed_index(complex(e), complex(m))
                                     for e, m in zip(eps, mu)])}
        for backend in ('jax', 'torch'):
            lib = pytest.importorskip(backend)
            if backend == 'jax':
                wavebit.enable_x64()
                got[backend] = np.asarray(signed_index(lib.numpy.asarray(eps),
                                                       lib.numpy.asarray(mu)))
            else:
                got[backend] = signed_index(lib.tensor(eps), lib.tensor(mu)).numpy()
        for name, n in got.items():
            assert np.all((np.conj(n) * kz).real > 0), name
            np.testing.assert_allclose(n, kz, rtol=1e-15, atol=0, err_msg=name)
    # Through the solver, the materials as Python numbers and as each backend's arrays.
    for backend, wrap in (('numpy', complex), ('numpy', np.asarray), ('jax', 'jax'),
                          ('torch', 'torch')):
        if wrap == 'jax':
            jnp = pytest.importorskip('jax').numpy
            wrap = jnp.asarray
        elif wrap == 'torch':
            torch = pytest.importorskip('torch')
            wrap = lambda v: torch.tensor(v, dtype=torch.complex128)   # noqa: E731
        medium = wavebit.Material(wrap(-0.1j), mu=wrap(-3 + 0j))
        solver = _solver(bot=medium, theta=0.0, path='2d', backend=backend)
        result = solver.solve()
        t_s = complex(np.sum(_host(result.s_inc.t_s)))
        t_p = complex(np.sum(_host(result.p_inc.t_p)))
        assert abs(t_p - t_s) < 1e-13 * abs(t_s), (backend, wrap)
        assert complex(_host(solver.n_bot)).real > 0


@pytest.mark.parametrize('eps, mu, along', [
    ((-2 - 0.1j), 1.0 + 0j, True),         # a gain metal named with mu = 1: evanescent
    ((2 + 0.1j), (-1 - 0.1j), True),       # the gain metal in mu
    ((-2 - 0.01j), (0.1 - 0.5j), True),    # a magnetic metal with gain
    ((-2 + 0.01j), (0.1 + 0.5j), True),    # passive, for contrast
    ((-0.1 - 2j), (0.1 + 2j), False),      # a gain-metal factor that propagates: against kz
])
def test_t_p_is_t_s_at_normal_incidence_where_the_index_points_along_kz(eps, mu, along):
    """At normal incidence an isotropic interface has no s or p: the far side's ``kz`` is
    ``+-n``, and ``t_p = t_s`` exactly when the signed index points along the ``kz`` the
    half-space is solved with, ``t_p = -t_s`` when it points against it.

    With each factor's root principal, every medium with one gain-metal factor that does not
    propagate pointed against its ``kz`` (``t_p = -t_s``); with Maystre's roots they point along
    it. The few that propagate go the other way (study section 3: 5,839 of 400,000 random media
    against 93,797); the last case pins that, so a change to it is a decision, not an accident."""
    medium = wavebit.Material(eps, mu=mu)
    solver = _solver(bot=medium, theta=0.0, path='2d')
    result = solver.solve()
    n = complex(solver.n_bot)
    kz = boundary_kz(np, np.asarray([eps * mu]), np.zeros(1), medium=(eps, mu))[0]
    assert ((np.conj(n) * kz).real > 0) == along
    t_s = complex(np.sum(_host(result.s_inc.t_s)))
    t_p = complex(np.sum(_host(result.p_inc.t_p)))
    assert abs(t_p - (1 if along else -1) * t_s) < 1e-13 * abs(t_s)


@pytest.mark.parametrize('which', ['eps', 'mu'])
def test_t_p_is_continuous_across_a_real_part_seam(which):
    """``Re(eps)`` (with ``mu = -1+0.1j``) or ``Re(mu)`` (with ``eps = -2+0.1j``) crossing zero
    takes a passive far side into and out of the double-negative class, and nothing physical
    happens there: ``n_bot`` and every amplitude move by ``O(x)``. A sign keyed on both real
    parts being negative made ``n_bot`` and ``t_p`` jump by ``2*n`` at the seam, which no power
    quantity and no gradient sees."""
    pair = ((lambda x: (x + 0.3j, -1 + 0.1j)) if which == 'eps' else
            (lambda x: (-2 + 0.1j, x + 0.4j)))
    rows = []
    for x in (-1e-9, -1e-12, 0.0, 1e-12, 1e-9):
        solver = _grating('conical', 'SMM', nim=pair(x))
        result = solver.solve(auto_detune=False)
        rows.append(np.concatenate([[solver.n_bot]] + [
            np.ravel(_host(getattr(v, f))) for v in _variants(result, 'conical')
            for f in ('r_s', 'r_p', 't_s', 't_p')]))
    rows = np.asarray(rows)
    assert np.max(np.abs(np.diff(rows, axis=0))) < 1e-8
    assert np.min(np.abs(rows[:, 0])) > 0.2


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('eps', [-1 + 1.2j, -0.1 + 2j])
def test_a_double_negative_too_lossy_to_propagate_is_solved(eps, algo):
    """Both real parts negative but ``Re(eps*mu) <= 0``: nothing propagates at any ``k_par``,
    every root decays, and both rules pick it."""
    _check_interface(eps, eps, algo=algo, thetas=(0.0, 60.0))


def test_maystres_tie_on_a_double_negative_takes_the_decaying_root():
    """``eps = mu = -1+1j`` puts the normal order on Maystre's cut, ``kz**2 = -2j``, where his
    tie-break kept ``1-1j``, which grows. Accepted before and wrong there; the mirror rule
    takes ``-1+1j``. A unit check, since the scene is a single order."""
    eps = mu = -1 + 1j
    kz = boundary_kz(np, np.asarray([eps * mu]), np.asarray([0j]), (eps, mu))
    assert kz[0] == pytest.approx(-1 + 1j)
    assert boundary_kz(np, np.asarray([eps * mu]), np.asarray([0j]))[0] == pytest.approx(1 - 1j)
    _check_interface(eps, mu, thetas=(0.0,))


def test_the_mirror_rule_on_a_sweep():
    """At every real ``k_par`` of a passive double negative the root decays, and a lossless
    propagating order has ``Re(kz) < 0``. Anywhere else ``medium`` changes nothing, to the bit,
    including complex ``k_par`` and gain. Kills a rule applied without the material test, and
    one applied to a medium with a single negative real part: at a complex ``k_par`` such a
    medium reaches ``Re(kz**2) > 0``, where the two rules part."""
    rng = np.random.default_rng(7)
    k_par_sq = np.concatenate([np.linspace(0, 9, 301), rng.uniform(0, 9, 50)]) + 0j
    for eps, mu in (PASSIVE, LOSSLESS, HIGH, (-1 + 1.2j, -1 + 1.2j)):
        kz = boundary_kz(np, eps * mu, k_par_sq, (eps, mu))
        assert np.all(kz.imag >= 0)
        propagating = (kz.imag == 0) & (kz.real != 0)
        assert np.all(kz[propagating].real < 0)
        np.testing.assert_allclose(kz ** 2, eps * mu - k_par_sq, atol=1e-13)
    k_par = np.sqrt(k_par_sq) + 1j * rng.normal(scale=1.5, size=k_par_sq.shape)
    k_par_sq = np.concatenate([k_par_sq, k_par ** 2])
    for eps, mu in ((2.25, 1.0), (-2 + 0.1j, 1.0), (-2 + 0.01j, 0.1 + 0.5j),
                    (2.25 - 0.05j, 1.2), (2 + 0.1j, -1 + 0.1j), (1 - 0.05j, 1 - 0.05j)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain = boundary_kz(np, eps * mu, k_par_sq)
            told = boundary_kz(np, eps * mu, k_par_sq, (eps, mu))
        assert plain.tobytes() == told.tobytes()
        assert np.any((eps * mu - k_par_sq).real > 0)     # where the two rules part


def test_the_double_negative_rule_is_the_mirror_of_maystres_to_the_bit():
    """``kz_DNG(z) == -conj(maystre(conj(z)))`` on every ``kz**2``, the axes and signed zeros
    included: the double-negative rule is the ordinary one run on the mirror medium
    ``(-conj(eps), -conj(mu))``. Its own cut is the positive imaginary axis of ``kz**2``, where
    the root with ``Re(kz) < 0`` is kept (Maystre keeps ``Re(kz) > 0`` on his); a strict ``<``
    there keeps the other root, which the axis samples catch."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=4000) + 1j * rng.normal(size=4000)
    axis = rng.uniform(0.01, 9, 400)
    z = np.concatenate([z, 1j * axis, -1j * axis, axis + 0j, -axis + 0j, axis * (1 + 1j),
                        axis * (1 - 1j), [0j, complex(0, -0.0), complex(-0.0, 0), 2j, -2j]])
    zero = np.zeros_like(z)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mirrored = boundary_kz(np, z, zero, (-2 + 0.1j, -1 + 0.1j))
        maystre = boundary_kz(np, np.conj(z), zero)
    assert mirrored.tobytes() == (-np.conj(maystre)).tobytes()
    on_the_cut = boundary_kz(np, np.asarray([2j]), np.asarray([0j]), (-1.0, -1.0))[0]
    assert on_the_cut.real < 0 and on_the_cut.imag < 0


def test_a_negative_index_slab_matches_airy():
    """A layer of the medium keeps both roots, so its scattering matrix cannot depend on which
    one is labelled forward. The slab oracle below rests on this."""
    eps, mu = PASSIVE
    for algo in ('ETM', 'SMM'):
        for theta_deg in (0.0, 60.0):
            theta, d = np.deg2rad(theta_deg), 120.0
            solver = _solver(theta=theta, algo=algo, thickness=[d])
            solver.structure = np.full((1, 2, 2), eps, dtype=complex)
            solver.mu = np.full((1, 2, 2), mu, dtype=complex)
            result = solver.solve()
            k_par = N_GLASS * np.sin(theta)
            k1 = np.sqrt(N_GLASS ** 2 - k_par ** 2 + 0j)
            k2 = np.sqrt(eps * mu - k_par ** 2)     # either root: Airy is even in it
            for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
                a, b = (k1, k2 / mu) if pol == 's' else (k1 / N_GLASS ** 2, k2 / eps)
                r12, t12, t23 = (a - b) / (a + b), 2 * a / (a + b), 2 * b / (a + b)
                phase = np.exp(1j * k2 * 2 * np.pi / WL * d)
                den = 1 - r12 ** 2 * phase ** 2
                R = abs(r12 * (1 - phase ** 2) / den) ** 2
                T = abs(t12 * t23 * phase / den) ** 2
                np.testing.assert_allclose([float(np.sum(sub.R)), float(np.sum(sub.T))], [R, T],
                                           atol=1e-12, err_msg=f'{pol} {theta_deg} {algo}')


# ------------------------------------------------- a grating: the thick-slab limit, every path

G1 = np.array([4.0, 4.0, 1.0, 1.0, 2.25, 2.25], dtype=complex)
G2 = np.ones((6, 6), dtype=complex)
G2[1:4, 2:5] = 4.0
G2[4, 0:2] = 2.25
H, P_G = 180.0, 700.0


def _grating(path, algo, nim=PASSIVE, d=None, backend='numpy', answers='amplitude', lit=None,
             side='top', theta=0.3, phi=0.4):
    """glass (lit) / grating / NIM half-space, or / NIM slab of thickness ``d`` / glass.

    With ``lit``, that medium is the lit half-space on ``side`` and glass the far one."""
    eps, mu = nim
    common = dict(backend=backend, top=N_GLASS ** 2, wavelength=WL, layer_algorithm=algo,
                  theta=theta, answers=answers)
    if path in ('te', 'tm', 'conical'):
        grating, kw = G1[None, None, :], dict(period=[P_G], truncation_order=3)
    else:
        grating, kw = G2[None], dict(period=(P_G, 0.9 * P_G), truncation_order=(1, 1))
    kw.update(tetm=path) if path in ('te', 'tm') else kw.update(phi=phi, psi=0.3)
    if lit is not None:
        half = wavebit.Material(*lit)
        common.update(top=half, bot=N_GLASS ** 2) if side == 'top' else common.update(
            bot=half)
        solver = wavebit.rcwa(**common, **kw, thickness=[H], structure=grating,
                              incidence_side=side)
    elif d is None:
        solver = wavebit.rcwa(**common, **kw, bot=wavebit.Material(eps, mu=mu),
                              thickness=[H], structure=grating)
    else:
        solver = wavebit.rcwa(**common, **kw, bot=N_GLASS ** 2, thickness=[d, H],
                              structure=np.concatenate([np.full_like(grating, eps), grating]))
        solver.mu = np.concatenate([np.full_like(grating, mu), np.ones_like(grating)])
    if path == 'first_order':
        solver.formulation = 'first_order'
    return solver


def _variants(result, path):
    return [result.incident] if path in ('te', 'tm') else [result.incident, result.s_inc,
                                                           result.p_inc]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'first_order'])
def test_a_grating_on_the_half_space_is_the_thick_slab_limit(path, algo):
    """Every order's ``R``, and the half-space's ``T`` against the power the slab takes in.

    16 um of ``eps = -2+0.1i``, ``mu = -1+0.1i`` returns ``e**-43`` of the field it receives,
    so the two scenes are one problem to round-off: measured 9e-16 in ``R`` and 3.3e-15 in
    ``T``, against 1e-4 at 1 um. Four orders propagate in the medium, oblique ones included.
    The grating is lossless, so ``R + T = 1`` for the half-space, and its entering power is its
    ``T``."""
    half = _grating(path, algo).solve(auto_detune=False)
    slab = _grating(path, algo, d=16000.0).solve(auto_detune=False)
    for h, s in zip(_variants(half, path), _variants(slab, path)):
        np.testing.assert_allclose(_host(h.R), _host(s.R), rtol=0, atol=1e-12)
        T = float(np.sum(_host(h.T)))
        assert abs(T - float(_host(s.power_entering))) < 1e-12
        assert float(np.sum(_host(s.T))) < 1e-15
        assert abs(float(np.sum(_host(h.R))) + T - 1) < 1e-12
        assert abs(float(_host(h.power_entering)) - T) < 1e-12


def test_the_lossless_half_space_is_the_vanishing_loss_limit():
    """``eps = -2+i*delta``, ``mu = -1+i*delta`` slabs thick enough to swallow the round trip,
    a layer computation each, extrapolated to ``delta = 0`` from ``1e-4`` and ``1e-5``, land on
    the lossless half-space: measured 2.3e-8, where ``1e-5`` alone is 1.9e-5 away (a slope of
    1.9). The slab is 128 mm at ``1e-5``."""
    def total(delta):
        nim = (-2 + 1j * delta, -1 + 1j * delta)
        d = 34.0 / (2 * 2 * np.pi / WL * abs((-np.sqrt(nim[0] * nim[1])).imag))
        result = _grating('conical', 'SMM', nim=nim, d=d).solve(auto_detune=False)
        return np.concatenate([_host(v.R).ravel() for v in _variants(result, 'conical')])

    lossless = _grating('conical', 'SMM', nim=LOSSLESS).solve(auto_detune=False)
    lossless = np.concatenate([_host(v.R).ravel() for v in _variants(lossless, 'conical')])
    a, b = total(1e-4), total(1e-5)
    extrapolated = b + (b - a) * 1e-5 / (1e-4 - 1e-5)
    assert np.max(np.abs(extrapolated - lossless)) < 1e-6
    assert np.max(np.abs(b - lossless)) > 5e-6      # the limit is approached, not sat on


@pytest.mark.parametrize('path, algo', [('te', 'ETM'), ('conical', 'SMM'), ('2d', 'SMM'),
                                        ('first_order', 'ETM')])
def test_the_field_below_the_interface_is_the_slab_field(path, algo):
    """The half-space's ``t_xyz``, carried to depth ``z`` on the decaying root, is the slab
    scene's reconstructed ``E`` at the same depth inside the slab; measured 3e-14. It decays.

    ``reconstruct_field`` does not sample a half-space, so this is the half-space's field as the
    published amplitudes define it, against a layer that needs no root."""
    from wavebit._core.solve_2d import order_grid_kxy
    half_solver = _grating(path, algo, answers='field')
    half = half_solver.solve(auto_detune=False)
    d = 16000.0
    slab = _grating(path, algo, d=d, answers='field').solve(auto_detune=False)
    xs, ys = [-123.0, 0.0, 211.0], ([0.0] if path in ('te', 'conical') else [0.0, 97.0])
    gx, gy = order_grid_kxy(np, *[_host(a) for a in half_solver.order_kxy()])
    gx, gy = gx.ravel(), gy.ravel()
    kz = np.array([_decaying_root(*PASSIVE, np.sqrt(k + 0j)) for k in gx * gx + gy * gy])
    t_xyz = np.reshape(_host(half.incident.t_xyz), (-1, 3))
    X, Y = np.meshgrid(xs, ys)
    k0 = 2 * np.pi / WL
    depths = [1.0, 50.0, 400.0, 2000.0]
    predicted = np.array([(np.exp(1j * k0 * (X[..., None] * gx + Y[..., None] * gy))
                           * np.exp(1j * k0 * kz * z)) @ t_xyz for z in depths])
    got = _host(slab.reconstruct_field(xs, ys, [d - z for z in depths]))[..., :3]
    np.testing.assert_allclose(predicted, got, rtol=0, atol=1e-12)
    magnitude = [np.max(np.abs(a)) for a in predicted]
    assert magnitude[-1] < 0.1 * magnitude[0]
    # and the half-space scene's own field is continuous across its interface
    inside = _host(half.reconstruct_field(xs, ys, [0.0]))[0]
    face = _host(half.boundary_fields('bottom', xs, ys, part='total'))
    np.testing.assert_allclose(inside[..., [0, 1, 3, 4]], face[..., [0, 1, 3, 4]], atol=1e-12)


def test_the_coherent_door_agrees_with_the_ordinary_solve():
    """``excitation={'top': PlaneWave(...)}`` onto a NIM bottom: the first-order cascade with its
    own plane-wave basis (``illumination._plane_basis``) against the ordinary SMM solve. Measured
    2e-16 in power and 3e-14 in field."""
    pol = (0.6, 0.8j)
    opts = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :], thickness=[H],
                truncation_order=(3, 0), top=N_GLASS ** 2, answers='field',
                bot=wavebit.Material(*PASSIVE))
    coherent = wavebit.rcwa(**opts, excitation={'top': wavebit.PlaneWave(
        theta=0.3, phi=0.4, polarization=pol)}).solve()
    ordinary = wavebit.rcwa(**opts, theta=0.3, phi=0.4, polarization=pol,
                            layer_algorithm='SMM').solve(auto_detune=False)
    assert abs(float(coherent.output_top) - float(np.sum(ordinary.R))) < 1e-12
    assert abs(float(coherent.output_bottom) - float(np.sum(ordinary.T))) < 1e-12
    points = dict(sample_x=[-123.0, 211.0], sample_y=[0.0], sample_z=[10.0, 90.0])
    np.testing.assert_allclose(coherent.reconstruct_field(**points),
                               ordinary.reconstruct_field(**points), atol=1e-12)


# ------------------------------------------------------------------- the mirror medium, -conj

def _mirrored(value):
    """``-conj(value)``; a number stays a Python complex, so both scenes take one square root.

    (``cmath.sqrt`` and NumPy's differ in the last bit on some arguments, and a mirror that
    handed the index a NumPy scalar where the original handed a Python one moved ``t_p`` by
    an ulp for that reason alone.)"""
    return complex(-np.conj(value)) if np.ndim(value) == 0 else -np.conj(value)


def _amplitudes_by_name(result):
    """``{'incident.r_s': array, ...}``: every amplitude and power the result carries."""
    out = {}
    for name in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, name, None)
        for field in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
            value = None if sub is None else getattr(sub, field, None)
            if value is not None:
                out[f'{name}.{field}'] = np.ravel(_host(value))
    return out


def _mirror_scene(path, algo, side, theta, far, eps=None, mu=None, thickness=(10.0,),
                  truncation_order=None):
    """Glass ``(2.25, 1)`` lit at ``theta`` onto the half-space ``far`` through the raster layers
    ``eps`` / ``mu`` -- or through none, by ``solve_conv``, when they are ``None``."""
    shape = (1, 1, 4) if path in ('te', 'tm', 'conical') else (1, 2, 2)
    layers = eps is not None
    if not layers:
        eps, mu = np.full(shape, 2.25 + 0j), np.full(shape, 1.0 + 0j)
    glass, medium = wavebit.Material(2.25 + 0j, mu=1.0 + 0j), wavebit.Material(*far)
    top, bot = (glass, medium) if side == 'top' else (medium, glass)
    common = dict(top=top, bot=bot, theta=theta, wavelength=WL, layer_algorithm=algo,
                  thickness=list(thickness), answers='amplitude', incidence_side=side,
                  structure=eps, mu=mu)
    if path in ('te', 'tm'):
        solver = wavebit.rcwa(**common, tetm=path, period=[700.0],
                              truncation_order=truncation_order or 3)
    elif path == 'conical':
        solver = wavebit.rcwa(**common, phi=0.6, psi=0.3, period=[700.0],
                              truncation_order=truncation_order or 3)
    else:
        solver = wavebit.rcwa(**common, phi=0.6, psi=0.3, period=(700.0, 630.0),
                              truncation_order=truncation_order or (1, 1))
        if path == 'first_order':
            solver.formulation = 'first_order'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if layers:
            return _amplitudes_by_name(solver.solve(auto_detune=False))
        e, m = solver.eps_conv_matrices(), solver.mu_conv_matrices()
        solver.thickness = []
        return _amplitudes_by_name(solver.solve_conv(
            *(None if a is None else a[:0] for a in e), mu=tuple(None if a is None else a[:0]
                                                               for a in m),
            eps_top=solver.top.epsilon, mu_top=solver.top.mu, eps_bot=solver.bot.epsilon,
            mu_bot=solver.bot.mu))


MIRROR_PAIRS = {'lossy': (2 + 0.1j, 1.5 + 0.05j), 'lossless': (2.25 + 0j, 1.3 + 0j),
                'lossless, evanescent at 60 deg': (1.2 + 0j, 1.0 + 0j)}


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'first_order'])
def test_the_mirror_medium_conjugates_every_reflection(path, side, algo):
    """Light from a real medium at a real angle: a far half-space ``(eps, mu)`` replaced by
    ``(-conj(eps), -conj(mu))`` sends every reflection amplitude to its conjugate and leaves
    ``|r|`` and ``|t|`` -- every ``R`` and ``T`` -- unchanged. The mirror medium is A. Lakhtakia's
    conjugation symmetry (Microwave Opt. Technol. Lett. 40, 160 (2004)), which conjugates ``E``
    and ``H``; V. Grunhut and R. A. Depine (Eur. Phys. J. D 62, 227 (2011)) apply it to a flat
    interface, where the Fresnel coefficients change phase and not magnitude. The mirror of a
    positive medium is a double negative: its root
    is ``-conj`` of the other's (the mirror of Maystre's cut is ``-conj(maystre(conj z))``), its
    admittances ``kz/mu`` and ``kz/eps`` are the conjugates, and its signed index is
    ``-conj(n)``, so ``t`` goes to ``conj(t)`` too. Every operation the solve does on the bare
    interface is one whose conjugate is the conjugate operation, so this is exact: measured
    bit for bit on every path, side and algorithm, lossy and lossless, propagating and
    evanescent in the far medium, 0, 0.3 rad and 60 degrees. It fails if the double negative
    takes Maystre's root or the principal index."""
    for name, far in MIRROR_PAIRS.items():
        for theta in (0.0, 0.3, np.deg2rad(60.0)):
            plain = _mirror_scene(path, algo, side, theta, far)
            mirror = _mirror_scene(path, algo, side, theta, tuple(_mirrored(v) for v in far))
            for key, value in plain.items():
                want = value if key.endswith(('.R', '.T')) else np.conj(value)
                assert np.array_equal(mirror[key], want), (name, theta, key)


def test_mirroring_everything_below_the_light_conjugates_it_through_layers():
    """The same identity, layer by layer: under ``(eps, mu) -> (-conj(eps), -conj(mu))`` a layer's
    first-order system becomes its conjugate, so its transfer matrix does, and with every layer
    and the far half-space mirrored the reflection conjugates again -- provided every retained
    order propagates in the lit medium, whose admittances are then real. Two lossy layers, a
    positive and a double negative, become a double negative and a positive one; a grating
    symmetric about the cell centre (whose convolution matrices are then symmetric) mirrors the
    same way. Measured 7.8e-15 at worst, relative to ``max(1, |amplitude|)``. It exercises the
    double-negative layers' modes on every path (their labels move no exact answer, so it does
    not see those)."""
    def uniform(path):
        shape = (1, 9) if path in ('te', 'tm', 'conical') else (5, 5)
        return (np.stack([np.full(shape, 2 + 0.1j), np.full(shape, -3 + 0.2j)]),
                np.stack([np.full(shape, 1.2 + 0.05j), np.full(shape, -1.1 + 0.1j)]))

    def grating(path):
        shape = (1, 1, 9) if path in ('te', 'tm', 'conical') else (1, 5, 5)
        eps, mu = np.full(shape, 2.0 + 0.1j), np.full(shape, 1.0 + 0j)
        middle = (0, 0, slice(3, 6)) if len(shape) == 3 and shape[1] == 1 else (
            0, slice(1, 4), 2)
        eps[middle], mu[middle] = -3 + 0.3j, -1.2 + 0.1j
        return eps, mu

    worst = 0.0
    for path in ('te', 'tm', 'conical', '2d', 'first_order'):
        for build, thickness in ((uniform, (60.0, 90.0)), (grating, (60.0,))):
            eps, mu = build(path)
            for far in MIRROR_PAIRS.values():
                for algo in ('ETM', 'SMM'):
                    for side in ('top', 'bottom'):
                        for theta in (0.0, 0.2):
                            kw = dict(thickness=thickness,
                                      truncation_order=1 if path in ('te', 'tm', 'conical')
                                      else (1, 1))
                            plain = _mirror_scene(path, algo, side, theta, far, eps, mu, **kw)
                            mirror = _mirror_scene(path, algo, side, theta,
                                                   tuple(_mirrored(v) for v in far),
                                                   _mirrored(eps), _mirrored(mu), **kw)
                            for key, value in plain.items():
                                want = value if key.endswith(('.R', '.T')) else np.conj(value)
                                gap = np.abs(mirror[key] - want).max() / max(
                                    1.0, np.abs(value).max())
                                worst = max(worst, gap)
    assert worst < 1e-13


# --------------------------------------------------------------------------- what is refused

# --------------------------------------------------------------------------- the lit side

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'first_order'])
def test_a_lit_negative_index_half_space_matches_the_closed_form(path, side, algo):
    """``r``, ``t``, ``R``, ``T`` for s and p at 0, 20, 35 and 60 degrees, lit from an asymmetric
    lossy double negative (``eps = -3+0.1i``, ``mu = -0.5+0.05i``) and from its lossless twin,
    onto glass. Measured 1.3e-15 at worst. This pins the lit side's root (the incident ``kz``
    decays toward the structure), its power normalisation and the sign of ``t_p`` (the signed
    ``n1``); on a planar interface ``kx`` enters only squared, so the direction it takes is
    pinned by the Poynting, refraction and reciprocity tests below."""
    for pair in (ASYM, ASYM_LOSSLESS):
        _check_lit_interface(*pair, path=path, algo=algo, side=side)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('pair', [ASYM_LOSSLESS, (2.25 + 0j, 1.3 + 0j)], ids=['nim', 'ordinary'])
def test_theta_and_phi_are_the_direction_the_incident_power_travels(pair, side):
    """The Poynting vector of the incident field the solve returns (``boundary_fields(...,
    part='incoming')``) points at ``theta = 35`` degrees from the lit side's inward normal, at
    azimuth ``phi = 0.6`` rad, for s, p and a mixed polarization -- in a lossless double
    negative as in an ordinary medium; measured to 1e-13 degrees. Its wavevector points the
    other way in the double negative: ``kx = n*sin(theta)*cos(phi)`` with ``n = -sqrt(1.5)``."""
    theta, phi = np.deg2rad(35.0), 0.6
    half = wavebit.Material(*pair)
    n = np.sqrt(pair[0]) * np.sqrt(pair[1])
    scene = dict(top=half) if side == 'top' else dict(bot=half)
    inward = np.array([0.0, 0.0, -1.0 if side == 'top' else 1.0])
    for psi in (0.0, np.pi / 2, 0.7):
        solver = _solver(theta=theta, phi=phi, side=side, answers='field', **scene)
        solver.psi = psi
        result = solver.solve()
        kx, ky = (complex(np.ravel(_host(k))[0]) for k in solver.order_kxy())
        np.testing.assert_allclose([kx, ky], n.real * np.sin(theta) * np.array(
            [np.cos(phi), np.sin(phi)]), rtol=0, atol=1e-14)
        field = _host(result.boundary_fields(side, [0.0], [0.0], part='incoming'))[0, 0]
        poynting = 0.5 * np.real(np.cross(field[:3], np.conj(field[3:])))
        polar = np.degrees(np.arccos(poynting @ inward / np.linalg.norm(poynting)))
        azimuth = np.arctan2(poynting[1], poynting[0])
        assert abs(polar - 35.0) < 1e-10 and abs(azimuth - phi) < 1e-12, (psi, polar, azimuth)


def test_light_from_a_negative_index_medium_refracts_negatively():
    """Lit from ``n = -sqrt(2)`` (``eps = -2``, ``mu = -1``) at 20 degrees onto glass, the
    transmitted power leaves at ``-18.81`` degrees, on the same side of the normal it came from:
    ``n1 sin(theta1) = n2 sin(theta2)`` with the signed index (J. B. Pendry, Phys. Rev. Lett. 85,
    3966 (2000)). Read off the transmitted field's Poynting vector, oblique 2D, both sides."""
    theta, phi = np.deg2rad(20.0), 0.6
    expected = np.degrees(np.arcsin(-np.sqrt(2) * np.sin(theta) / N_GLASS))
    assert abs(expected + 18.8117) < 1e-4
    for side in ('top', 'bottom'):
        far = 'bottom' if side == 'top' else 'top'
        scene = {('top' if side == 'top' else 'bot'): wavebit.Material(*LOSSLESS)}
        result = _solver(theta=theta, phi=phi, side=side, answers='field', **scene).solve()
        field = _host(result.boundary_fields(far, [0.0], [0.0], part='outgoing'))[0, 0]
        poynting = 0.5 * np.real(np.cross(field[:3], np.conj(field[3:])))
        along = poynting[0] * np.cos(phi) + poynting[1] * np.sin(phi)
        normal = abs(poynting[2])
        assert abs(np.degrees(np.arctan2(along, normal)) - expected) < 1e-10
        assert abs(poynting[0] * np.sin(phi) - poynting[1] * np.cos(phi)) < 1e-14


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_grating_lit_from_a_negative_index_side_is_reciprocal(tetm, algo):
    """A grating lit from a lossless double negative (``eps = -3``, ``mu = -0.5``) at 0.35 rad
    diffracts into glass with the efficiency of the reciprocal problem -- lit from the glass
    against the direction of that order, read in the double negative against the direction of
    the incident wave -- for orders +1 and -1, to round-off. Both problems retain mirrored
    order windows (zero-weight ``excitation`` entries pin them), so truncation cannot part
    them. The reciprocal problem's illumination is direct ``kx`` from an ordinary medium: no
    root rule, index sign or angle semantics of the double negative enters it except as the
    far side the slab test pins."""
    def solve(side, excitation, **incidence):
        solver = wavebit.rcwa(backend='numpy', wavelength=WL, period=[P_G], structure=G1[None, None, :],
                              thickness=[H], truncation_order=3, tetm=tetm, layer_algorithm=algo,
                              top=wavebit.Material(*ASYM_LOSSLESS), bot=N_GLASS ** 2,
                              incidence_side=side, excitation=excitation, **incidence)
        T = np.ravel(_host(solver.solve(auto_detune=False).T))
        (centre, half), _ = solver.order_window
        return list(range(centre - half, centre + half + 1)), np.ravel(
            _host(solver.order_kxy()[0])).real, T
    for m in (1, -1):
        orders, kx, forward = solve('top', {0: 1.0, m: 0.0, -m: 0.0}, theta=0.35)
        kx_inc = kx[orders.index(0)]
        assert kx_inc < 0
        orders_b, _, backward = solve('bottom', {-m: 1.0, 0: 0.0, m: 0.0},
                                      kx=-kx_inc * 2 * np.pi / WL)
        assert forward[orders.index(m)] > 1e-3
        assert abs(forward[orders.index(m)] - backward[orders_b.index(0)]) < 1e-12


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', 'first_order'])
def test_the_incident_field_is_continuous_through_normal_incidence(path, side, algo):
    """``theta = 0`` against ``theta = 1e-9`` on a grating lit from a double negative, at
    ``phi = 0.6`` where there is an azimuth: the amplitudes, ``r_xyz`` / ``t_xyz``, the
    reconstructed field and the incident field itself move by ``O(1e-9)``. As ``theta -> 0+``
    ``k_par`` points along ``phi + pi`` there, so the frame at exactly normal incidence is taken
    at that azimuth (``CoreSolveMixin._frame_azimuth``); at ``phi`` it negated the incident
    field and every amplitude."""
    rows = []
    for theta in (0.0, 1e-9):
        solver = _grating(path, algo, lit=ASYM, side=side, theta=theta, answers='field', phi=0.6)
        result = solver.solve(auto_detune=False)
        sub = result.incident
        ys = [0.0, 77.0] if path in ('2d', 'first_order') else [0.0]
        rows.append(np.concatenate(
            [np.ravel(_host(getattr(sub, f))) for f in ('r_s', 'r_p', 't_s', 't_p', 'r_xyz',
                                                       't_xyz')]
            + [np.ravel(_host(result.reconstruct_field([0.0, 123.0], ys, [10.0, 90.0]))),
               np.ravel(_host(result.boundary_fields(side, [0.0, 50.0], [0.0],
                                                     part='incoming')))]))
    assert np.max(np.abs(rows[1] - rows[0])) < 1e-7
    assert np.max(np.abs(rows[0])) > 0.1


@pytest.mark.parametrize('path', ['te', 'tm', 'conical'])
def test_a_complex_angle_meets_the_real_one(path):
    """``theta = 0.35 + 1e-9i`` against ``0.35``, lit from a double negative. Lossless, the two
    parameterisations are one (``n = s*N``), and every amplitude moves by ``O(1e-9)``. Lossy,
    the formal ``n*sin(theta)`` and ``s*N*sin(theta)`` differ by design (about 0.03 here, as
    for any absorbing medium) but sit on the same side of the normal; reading the real angle
    as the phase direction put them about 1 apart."""
    def solved(pair, theta):
        solver = _grating(path, 'SMM', lit=pair, theta=theta)
        result = solver.solve(auto_detune=False)
        return (np.ravel(_host(solver.order_kxy()[0])),
                np.concatenate([np.ravel(_host(getattr(result.incident, f)))
                                for f in ('R', 'r_s', 'r_p', 't_s', 't_p')]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavebit.ComplexIncidenceWarning)
        kx_real, real = solved(ASYM_LOSSLESS, 0.35)
        kx_complex, formal = solved(ASYM_LOSSLESS, 0.35 + 1e-9j)
        assert np.max(np.abs(kx_complex - kx_real)) < 1e-8
        assert np.max(np.abs(formal - real)) < 1e-7
        kx_real, _ = solved(ASYM, 0.35)
        kx_complex, _ = solved(ASYM, 0.35 + 1e-9j)
    assert np.max(np.abs(kx_complex - kx_real)) < 0.1
    assert kx_real[3].real < 0 and kx_complex[3].real < 0


@pytest.mark.parametrize('pair', [ASYM, ASYM_LOSSLESS], ids=['lossy', 'lossless'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_the_coherent_door_lights_a_negative_index_side_as_the_ordinary_solve(side, pair):
    """``excitation={side: PlaneWave(theta, phi, polarization)}`` from a double negative against
    the ordinary SMM solve lit from it: power and field, at ``(0.35, 0.6)`` and at normal
    incidence with ``phi = 0.6`` and ``2.5``, where the frame takes ``phi + pi``
    (``source_expansion.resolve_wave``). Measured 2e-15 in power and 6e-14 in field."""
    pol = (0.6, 0.8j)
    half = wavebit.Material(*pair)
    scene = dict(top=half, bot=N_GLASS ** 2) if side == 'top' else dict(top=N_GLASS ** 2,
                                                                         bot=half)
    opts = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :], thickness=[H],
                truncation_order=(3, 0), answers='field', **scene)
    for theta, phi in ((0.35, 0.6), (0.0, 0.6), (0.0, 2.5)):
        coherent = wavebit.rcwa(**opts, excitation={side: wavebit.PlaneWave(
            theta=theta, phi=phi, polarization=pol)}).solve()
        ordinary = wavebit.rcwa(**opts, theta=theta, phi=phi, polarization=pol,
                                layer_algorithm='SMM', incidence_side=side).solve(
            auto_detune=False)
        back, through = ((coherent.output_top, coherent.output_bottom) if side == 'top' else
                         (coherent.output_bottom, coherent.output_top))
        assert abs(float(back) - float(np.sum(ordinary.R))) < 1e-12
        assert abs(float(through) - float(np.sum(ordinary.T))) < 1e-12
        points = dict(sample_x=[-123.0, 211.0], sample_y=[0.0], sample_z=[10.0, 90.0])
        np.testing.assert_allclose(_host(coherent.reconstruct_field(**points)),
                                   _host(ordinary.reconstruct_field(**points)), atol=1e-12)


def test_a_gaussian_beam_from_a_negative_index_side_is_refused():
    """Its angular spectrum, transversality and focal phase are written for waves whose phase
    and power travel together; refused by name rather than built forward-phase."""
    opts = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :], thickness=[H],
                truncation_order=(2, 0), top=wavebit.Material(*ASYM_LOSSLESS),
                bot=N_GLASS ** 2, answers='field',
                power_region=((-1000.0, 1000.0), (-1000.0, 1000.0)))
    beam = wavebit.GaussianBeam(waist=2000.0, theta=0.2)
    with pytest.raises(ValueError, match='negative-index'):
        wavebit.rcwa(**opts, excitation=[beam]).solve()


@pytest.mark.parametrize('pair', [GAIN, GAIN_IN_MU], ids=['gain', 'gain_in_mu'])
@pytest.mark.parametrize('side', ['top', 'bot'])
def test_an_active_negative_index_half_space_is_refused(side, pair):
    """No single-frequency root rule exists for gain in a propagating double negative (Skaar
    2006; Nistad and Skaar 2008), so it is refused where it is set, and the setter leaves the
    solver as it was."""
    solver = _solver()
    before = (getattr(solver, side), getattr(solver, f'n_{side}'))
    with pytest.raises(ValueError, match=r'active negative-index half-space(.|\n)*Skaar'):
        setattr(solver, side, wavebit.Material(*pair))
    assert (getattr(solver, side), getattr(solver, f'n_{side}')) == before


def test_the_gain_band_of_a_double_negative_is_the_mirror_image():
    """Gain raises ``kz**2`` in a double negative, so its band near grazing lies above the real
    axis: the band of a double negative ``(eps, mu)`` is that of its double-positive mirror
    ``(-conj(eps), -conj(mu))`` to the bit, gain or loss in either factor, at every ``k_par``. A
    passive double negative is never flagged, and an active one is -- testing the lower half
    plane for every medium flagged none."""
    rng = np.random.default_rng(5)
    k_par_sq = np.linspace(0.0, 9.0, 901)
    for _ in range(300):
        eps = rng.uniform(-5, -0.1) + 1j * rng.uniform(-0.5, 0.5)
        mu = rng.uniform(-3, -0.1) + 1j * rng.uniform(-0.5, 0.5)
        band = gain_cutoff_band(np, eps, mu, k_par_sq)
        mirror = gain_cutoff_band(np, -np.conj(eps), -np.conj(mu), k_par_sq)
        np.testing.assert_array_equal(band, mirror)
    for pair in (PASSIVE, LOSSLESS, ASYM, HIGH):
        assert not gain_cutoff_band(np, *pair, k_par_sq).any()
    flagged = gain_cutoff_band(np, *GAIN, k_par_sq)
    assert flagged.any() and not flagged.all()
    assert np.all((GAIN[0] * GAIN[1] - k_par_sq[flagged]).imag > 0)
    # Gain in mu under more loss in eps leaves the product on the passive side: no band, as for
    # the ordinary mirror of that medium.
    assert not gain_cutoff_band(np, *GAIN_IN_MU, k_par_sq).any()


def test_the_gain_band_flags_an_active_double_negative_under_jit():
    """Under ``jax.jit`` the refusal cannot read the medium, so an active propagating double
    negative is solved with the mirror rule -- and ``gain_cutoff_band()``, which traces, names its
    orders near grazing: glass lit at 60 degrees onto ``eps = -2-0.1i``, ``mu = -1-0.1i`` puts
    the specular order at ``kz**2 = 0.30+0.30i``. Eagerly the same medium is refused."""
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    def band(re_eps):
        solver = _grating('conical', 'SMM', nim=(re_eps + 1j * GAIN[0].imag, GAIN[1]),
                          backend='jax')
        solver.theta = np.deg2rad(60.0)
        return solver.gain_cutoff_band()['bottom']

    flagged = np.asarray(jax.jit(band)(jnp.asarray(GAIN[0].real)))
    assert flagged.any() and not flagged.all()
    with pytest.raises(ValueError, match='active negative-index half-space'):
        band(GAIN[0].real)


def test_solve_conv_half_space_terms_are_read():
    """``solve_conv`` takes its own half-space terms; they need not be the scene's materials.

    The scene's substrate is the non-magnetic ``1.99-0.3j``, which is what the pair multiplies
    to; handed over as ``eps_bot`` / ``mu_bot`` it is solved with its own root, and its index
    with the pair's sign, so ``t_p == t_s`` at normal incidence as on the magnetic scene."""
    eps, mu = PASSIVE
    solver = _solver(bot=complex(eps * mu))
    result = solver.solve_conv(*solver.eps_conv_matrices(), mu_bot=mu, eps_bot=eps)
    expected = _interface(0.0, eps, mu)
    for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
        r, t = (sub.r_s, sub.t_s) if pol == 's' else (sub.r_p, sub.t_p)
        got = [complex(np.sum(r)), complex(np.sum(t)), float(np.sum(sub.R)), float(np.sum(sub.T))]
        np.testing.assert_allclose(got, expected[pol], rtol=0, atol=1e-12)
    assert solver.n_bot.real > 0           # the scene's own index, once the solve is over


@pytest.mark.parametrize('where', ['top', 'bot'])
def test_a_solve_conv_pair_signs_the_index_it_is_solved_with(where):
    """A non-magnetic scene holding ``eps*mu`` of the asymmetric double negative, split by
    ``solve_conv`` into that pair, is the magnetic scene: every amplitude (``t_p`` and
    ``r_xyz`` / ``t_xyz`` included) and the result's field agree to the bit, lit from the pair
    (``kx`` turned, ``theta = 0.6``, ``phi = 0.6``; and the frame at normal incidence) or not.
    The solver's indices are its materials' again afterwards, and its own
    ``reconstruct_field``, which reads the incidence live, refuses after a pair turned the lit
    side's sign rather than mixing the two scenes."""
    def scene(material, theta):
        sides = (dict(top=material, bot=N_GLASS ** 2) if where == 'top' else
                 dict(top=N_GLASS ** 2, bot=material))
        return wavebit.rcwa(backend='numpy', wavelength=WL, period=(P_G, 0.9 * P_G),
                            truncation_order=(1, 1), structure=G2[None], thickness=[H],
                            theta=theta, phi=0.6, psi=0.3, answers='field', **sides)
    points = ([0.0, 100.0], [0.0, 50.0], [10.0, 90.0])
    for theta in (0.6, 0.0):
        reference = scene(wavebit.Material(*ASYM), theta).solve(auto_detune=False)
        solver = scene(complex(ASYM[0] * ASYM[1]), theta)
        result = solver.solve_conv(*solver.eps_conv_matrices(),
                                   **{f'eps_{where}': ASYM[0], f'mu_{where}': ASYM[1]})
        for name in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p', 'r_xyz', 't_xyz'):
            np.testing.assert_array_equal(_host(getattr(result.incident, name)),
                                          _host(getattr(reference.incident, name)), err_msg=name)
        np.testing.assert_array_equal(_host(result.reconstruct_field(*points)),
                                      _host(reference.reconstruct_field(*points)))
        assert solver.n_top.real > 0 and solver.n_bot.real > 0
        if where == 'top':
            with pytest.raises(RuntimeError, match='scene has changed'):
                solver.reconstruct_field(*points)
        else:
            field = _host(result.reconstruct_field(*points))
            np.testing.assert_allclose(_host(solver.reconstruct_field(*points)).reshape(
                field.shape), field, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------- the backends

@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_backends_agree(backend):
    pytest.importorskip(backend)
    if backend == 'jax':
        wavebit.enable_x64()

    def flat(result):
        return np.concatenate([np.ravel(_host(getattr(v, f))) for v in _variants(result, 'conical')
                               for f in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p')])

    reference = flat(_grating('conical', 'SMM').solve(auto_detune=False))
    other = flat(_grating('conical', 'SMM', backend=backend).solve(auto_detune=False))
    np.testing.assert_allclose(other, reference, rtol=0, atol=1e-12)


def test_the_refusals_read_torch_values_eagerly():
    """Torch tensors as the material: a lit passive double negative is solved, as on NumPy, and
    an active one is refused where it is set."""
    torch = pytest.importorskip('torch')
    eps = torch.tensor(ASYM[0], dtype=torch.complex128)
    mu = torch.tensor(ASYM[1], dtype=torch.complex128)
    theta = np.deg2rad(35.0)
    got = _solver(backend='torch', theta=theta, phi=0.6, top=wavebit.Material(eps, mu=mu)).solve()
    want = _solver(theta=theta, phi=0.6, top=wavebit.Material(*ASYM)).solve()
    for name in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p'):
        np.testing.assert_allclose(_host(getattr(got.incident, name)),
                                   _host(getattr(want.incident, name)), rtol=0, atol=1e-13)
    gain = [torch.tensor(v, dtype=torch.complex128) for v in GAIN]
    with pytest.raises(ValueError, match='active negative-index half-space'):
        _solver(backend='torch', bot=wavebit.Material(gain[0], mu=gain[1]))


def test_jit_through_a_lit_negative_index_superstrate():
    """Under ``jax.jit`` the lit medium's ``Re(eps)`` is traced, so the sign of ``kx`` and the
    frame's azimuth at normal incidence come from ``where`` on the traced index rather than
    from a branch: ``R`` and every amplitude match NumPy at 35 degrees and at normal incidence
    with ``phi = 0.6``."""
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    def amplitudes(result):
        sub = result.incident
        return [sub.R, sub.r_s, sub.r_p, sub.t_s, sub.t_p]

    for theta in (0.35, 0.0):
        eager = amplitudes(_grating('conical', 'SMM', lit=ASYM, theta=theta, phi=0.6).solve(
            auto_detune=False))
        traced = jax.jit(lambda re_eps: amplitudes(_grating(
            'conical', 'SMM', lit=(re_eps + 1j * ASYM[0].imag, ASYM[1]), backend='jax',
            theta=theta, phi=0.6).solve(auto_detune=False)))(jnp.asarray(ASYM[0].real))
        for a, b in zip(traced, eager):
            np.testing.assert_allclose(np.asarray(a), _host(b), rtol=0, atol=1e-12)


def test_jit_and_grad_through_a_negative_index_substrate():
    """Under ``jax.jit`` the half-space's values are traced and no refusal can read them; the
    root and the index's sign then come from ``where`` on the traced material. ``R`` and the
    sign-reading ``t_p`` match NumPy, and ``jax.grad`` with respect to the medium's ``Re(eps)``
    matches a central difference (measured 2e-10)."""
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    def solved(re_eps):
        nim = (re_eps + 1j * PASSIVE[0].imag, PASSIVE[1])
        return _grating('conical', 'SMM', nim=nim, backend='jax').solve(auto_detune=False)

    def total_R(re_eps):
        return solved(re_eps).R.sum()

    def both(re_eps):
        result = solved(re_eps)
        return result.R.sum(), result.p_inc.t_p

    eager = _grating('conical', 'SMM').solve(auto_detune=False)
    R, t_p = jax.jit(both)(jnp.asarray(-2.0))
    assert abs(float(R) - float(np.sum(eager.R))) < 1e-12
    np.testing.assert_allclose(np.asarray(t_p), eager.p_inc.t_p, rtol=0, atol=1e-12)
    auto = float(jax.grad(lambda v: total_R(v).real)(jnp.asarray(-2.0)))
    h = 1e-6
    fd = (float(total_R(-2.0 + h)) - float(total_R(-2.0 - h))) / (2 * h)
    assert abs(auto - fd) < 1e-7 * max(abs(fd), 1e-3)


def test_under_jit_the_scenes_own_pairs_sign_nothing():
    """The ordinary solve hands the kernels the scene's own ``(eps, mu)`` pairs, whose index
    already is ``signed_index`` of those very objects. Under ``jax.jit`` a traced material's
    sign cannot be read, and re-signing the index by its own pair wrapped it in a ``where`` and
    marked the indices as pair-signed -- so a non-magnetic lit side's ``kx`` and frame took a
    traced ``where`` too, because a *far* side was magnetic. Both moved jitted results by a few
    ulp against develop for media with nothing negative anywhere.

    Pinned structurally: the scene's own pairs leave both indices the same objects and mark no
    side, and a foreign pair marks only its own side, so the lit sign of a non-magnetic lit side
    stays ``None`` (no ``where`` at all) while the far side is traced."""
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    seen = {}

    def probe(re_eps):
        solver = _grating('conical', 'SMM', nim=(re_eps + 0.1j, 1 + 0.1j), backend='jax')
        bot = solver._bot
        n_top, n_bot = solver._n_top, solver._n_bot
        with solver._indices_signed_by_pairs(solver._top.epsilon, solver._top.mu,
                                             bot.epsilon, bot.mu):
            seen['own'] = (solver._indices_from_pairs, solver._n_top is n_top,
                           solver._n_bot is n_bot, solver._lit_index_is_negative())
        # A foreign far-side pair: the split of the scene's eps*mu into (eps*mu/2, 2).
        with solver._indices_signed_by_pairs(None, None, bot.epsilon * bot.mu / 2, 2.0 + 0j):
            seen['foreign'] = (solver._indices_from_pairs, solver._lit_index_is_negative())
        return solver.solve(auto_detune=False).R.sum()

    jax.jit(probe)(jnp.asarray(2.0))
    assert seen['own'] == (frozenset(), True, True, None)
    assert 'top' not in seen['foreign'][0] and seen['foreign'][1] is None
