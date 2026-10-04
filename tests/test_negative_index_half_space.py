"""A negative-index half-space: solved as the far side, refused where it is lit or active.

``boundary_kz`` used to see only ``eps*mu``, and for ``Re(eps) < 0`` and ``Re(mu) < 0`` with
``Re(eps*mu) > 0`` that product looks like an ordinary forward medium, so the half-space took
the root that advances its phase away from the interface: glass onto the passive pair
``eps = -2+0.1j``, ``mu = -1+0.1j`` returned ``R = 929``, ``T = -928`` at normal incidence
against a true ``0.0011``. A double negative now takes the mirror of Maystre's cut, the
decaying root (``docs/branch-cuts.md`` section 9), and its index is ``-sqrt(eps*mu)``.

The oracles here need no root rule of their own: closed-form Fresnel coefficients on the
decaying root, and a thick lossy slab of the same medium, which as a layer keeps both roots.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit._core.conventions import boundary_kz

PERIOD, WL, N_GLASS = 1000.0, 500.0, 1.5
PASSIVE = (-2 + 0.1j, -1 + 0.1j)
LOSSLESS = (-2 + 0j, -1 + 0j)
HIGH = (-6 + 0.3j, -1.5 + 0.05j)
GAIN = (-2 - 0.1j, -1 - 0.1j)
GAIN_IN_MU = (-2 + 0.1j, -1 - 0.01j)
D_LAYER = 37.0


def _host(v):
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v)


def _solver(top=N_GLASS ** 2, bot=N_GLASS ** 2, theta=0.0, algo='SMM', backend=0, path='2d',
            side='top', **kw):
    """One uniform glass layer between the given half-spaces; the zeroth order only."""
    common = dict(backend=backend, top=top, bot=bot, theta=theta, wavelength=WL,
                  thickness=kw.pop('thickness', [D_LAYER]), layer_algorithm=algo,
                  incidence_side=side, **kw)
    if path in ('te', 'tm'):
        return wavebit.rcwa(**common, tetm=path, truncation_order=0, period=[PERIOD],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    if path == 'conical':
        return wavebit.rcwa(**common, phi=0.4, psi=0.0, truncation_order=0, period=[PERIOD],
                            structure=np.full((1, 1, 2), N_GLASS ** 2, dtype=complex))
    solver = wavebit.rcwa(**common, phi=0.4 if path == '4ff' else 0.0, psi=0.0,
                          truncation_order=(0, 0), period=(PERIOD, PERIOD))
    solver.structure = np.full((1, 2, 2), N_GLASS ** 2, dtype=complex)
    if path == '4ff':
        solver.formulation = '4ff'
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


# --------------------------------------------------------------------- the far side, solved

@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', '4ff'])
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
    """Only both negative is the backward-wave medium. One alone keeps its root, and the
    reflectance agrees with the closed-form interface for s and p.

    The last pair is passive yet its ``n_bot`` has ``Im(n) < 0``: no ``kz`` is decided on
    ``n``, only on ``n**2``, which is what this pins (``engine._half_space_index``). Its ``t_p``
    still reads that ``n`` and is not compared here (``docs/magnetic-materials.md``)."""
    result = _solver(bot=wavebit.Material(eps, mu=mu)).solve()
    expected = _interface(0.0, eps, mu)
    for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
        np.testing.assert_allclose(float(np.sum(sub.R)), expected[pol][2], atol=1e-12)


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


def _grating(path, algo, nim=PASSIVE, d=None, backend=0, answers='amplitude'):
    """glass (lit) / grating / NIM half-space, or / NIM slab of thickness ``d`` / glass."""
    eps, mu = nim
    common = dict(backend=backend, top=N_GLASS ** 2, wavelength=WL, layer_algorithm=algo,
                  theta=0.3, answers=answers)
    if path in ('te', 'tm', 'conical'):
        grating, kw = G1[None, None, :], dict(period=[P_G], truncation_order=3)
    else:
        grating, kw = G2[None], dict(period=(P_G, 0.9 * P_G), truncation_order=(1, 1))
    kw.update(tetm=path) if path in ('te', 'tm') else kw.update(phi=0.4, psi=0.3)
    if d is None:
        solver = wavebit.rcwa(**common, **kw, bot=wavebit.Material(eps, mu=mu),
                              thickness=[H], structure=grating)
    else:
        solver = wavebit.rcwa(**common, **kw, bot=N_GLASS ** 2, thickness=[d, H],
                              structure=np.concatenate([np.full_like(grating, eps), grating]))
        solver.mu = np.concatenate([np.full_like(grating, mu), np.ones_like(grating)])
    if path == '4ff':
        solver.formulation = '4ff'
    return solver


def _variants(result, path):
    return [result.incident] if path in ('te', 'tm') else [result.incident, result.s_inc,
                                                           result.p_inc]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', '4ff'])
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
                                        ('4ff', 'ETM')])
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
    """``excitation={'top': PlaneWave(...)}`` onto a NIM bottom: the 4ff cascade with its own
    plane-wave basis (``illumination._plane_basis``) against the ordinary SMM solve. Measured
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


# --------------------------------------------------------------------------- what is refused

@pytest.mark.parametrize('pair', [PASSIVE, LOSSLESS], ids=['passive', 'lossless'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
def test_a_lit_negative_index_half_space_is_refused_at_the_solve(side, pair):
    """Setting it is fine -- which side is lit is not known yet -- and lighting the other side
    solves. Kills a refusal that forgot ``incidence_side`` and one that came back to the setter."""
    where = 'top' if side == 'top' else 'bot'
    solver = _solver(side=side, **{where: wavebit.Material(*pair)})
    with pytest.raises(ValueError, match='incidence medium and a negative-index'):
        solver.solve()
    solver.incidence_side = 'bottom' if side == 'top' else 'top'
    assert np.isfinite(float(np.sum(solver.solve().R)))


@pytest.mark.parametrize('pair', [GAIN, GAIN_IN_MU], ids=['gain', 'gain_in_mu'])
@pytest.mark.parametrize('side', ['top', 'bot'])
def test_an_active_negative_index_half_space_is_refused(side, pair):
    """No root rule is established for gain in a propagating double negative."""
    solver = _solver()
    before = (getattr(solver, side), getattr(solver, f'n_{side}'))
    with pytest.raises(ValueError, match='active negative-index half-space'):
        setattr(solver, side, wavebit.Material(*pair))
    assert (getattr(solver, side), getattr(solver, f'n_{side}')) == before


def test_solve_conv_half_space_terms_are_read():
    """``solve_conv`` takes its own half-space terms; they need not be the scene's materials.

    The scene's substrate is the non-magnetic ``1.99-0.3j``, which is what the pair multiplies
    to; handed over as ``eps_bot`` / ``mu_bot`` it is the far side and is solved with its own
    root. Handed over as the lit side's, it is refused."""
    eps, mu = PASSIVE
    solver = _solver(bot=complex(eps * mu))
    result = solver.solve_conv(*solver.eps_conv_matrices(), mu_bot=mu, eps_bot=eps)
    expected = _interface(0.0, eps, mu)
    for pol, sub in (('s', result.s_inc), ('p', result.p_inc)):
        np.testing.assert_allclose(float(np.sum(sub.R)), expected[pol][2], atol=1e-12)
    with pytest.raises(ValueError, match='incidence medium and a negative-index'):
        solver.solve_conv(*solver.eps_conv_matrices(), mu_top=mu, eps_top=eps)


def test_the_coherent_door_refuses_a_lit_negative_index_side():
    opts = dict(wavelength=WL, period=[P_G], structure=G1[None, None, :], thickness=[H],
                truncation_order=(2, 0), top=N_GLASS ** 2, bot=wavebit.Material(*PASSIVE))
    with pytest.raises(ValueError, match='incidence medium and a negative-index'):
        wavebit.rcwa(**opts, excitation={'bottom': wavebit.PlaneWave(theta=0.3)}).solve()


# ----------------------------------------------------------------------------- the backends

@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_the_backends_agree(backend):
    pytest.importorskip('jax' if backend == 1 else 'torch')
    if backend == 1:
        wavebit.enable_x64()

    def flat(result):
        return np.concatenate([np.ravel(_host(getattr(v, f))) for v in _variants(result, 'conical')
                               for f in ('R', 'T', 'r_s', 'r_p', 't_s', 't_p')])

    reference = flat(_grating('conical', 'SMM').solve(auto_detune=False))
    other = flat(_grating('conical', 'SMM', backend=backend).solve(auto_detune=False))
    np.testing.assert_allclose(other, reference, rtol=0, atol=1e-12)


def test_the_refusals_read_torch_values_eagerly():
    torch = pytest.importorskip('torch')
    eps = torch.tensor(PASSIVE[0], dtype=torch.complex128)
    mu = torch.tensor(PASSIVE[1], dtype=torch.complex128)
    solver = _solver(backend=2, top=wavebit.Material(eps, mu=mu))
    with pytest.raises(ValueError, match='incidence medium and a negative-index'):
        solver.solve()
    gain = [torch.tensor(v, dtype=torch.complex128) for v in GAIN]
    with pytest.raises(ValueError, match='active negative-index half-space'):
        _solver(backend=2, bot=wavebit.Material(gain[0], mu=gain[1]))


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
        return _grating('conical', 'SMM', nim=nim, backend=1).solve(auto_detune=False)

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
