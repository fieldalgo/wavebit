"""A double-negative *layer*: how its modes are labelled, and the one pair ETM cannot write.

A layer mode comes as a ``+-q`` pair and the recursion has to call one root "down". In exact
arithmetic the choice is free; ETM is not free to make it, because it writes each layer through
the transmission from its downward modes into the region below
(``docs/connecting-algorithms.md`` section 10). For an ordinary layer the half-space's root
(Maystre's cut) is that wave. For a double negative it is the wrong one: Maystre's root
advances the phase downward, and a double negative carries its power against its phase
(V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)). So a uniform double-negative layer resting on
the same medium made ETM's interface factor singular -- ``LinAlgError`` on 171 of 720 such
scenes -- and SMM met the same thing at its gap medium whenever a backward mode's admittance
matched the gap's.

The rules now (``docs/connecting-algorithms.md`` section 11):

* a **uniform** double negative takes the mirror of the plain rule, as a half-space of the same
  medium does (``conventions.etm_mode_q`` / ``mode_q`` with ``mirrored``);
* a **patterned** layer holding a double-negative region takes each mode's power flux
  (``conventions.power_down_labels``), the partition the 4ff path has always split by;
* every other layer keeps its arithmetic, structurally.

The oracles need no label rule of their own: SMM (which the label cannot move in exact
arithmetic), closed-form interface reflection, the scattering matrix's evanescent transmission
against Pendry's closed form, and the same scene with the double negative's loss taken to zero.
Every test below has a control: the rule it pins, reverted, must break it.
"""
import warnings

import numpy as np
import pytest

import wavebit
from wavebit import Material
from wavebit._core import conventions
from wavebit._core.conventions import DoubleNegativeLayer, layer_label_rule

WL, P = 600.0, 700.0
LOSSLESS = (-2.0 + 0j, -1.0 + 0j)
LOSSY = (-2 + 0.1j, -1 + 0.1j)
ASYM = (-6.0 + 0j, -1.5 + 0j)
ASYM_LOSSY = (-3 + 0.1j, -0.5 + 0.05j)
PAIRS = {'lossless': LOSSLESS, 'lossy': LOSSY, 'asym': ASYM, 'asym_lossy': ASYM_LOSSY}


def _host(v):
    return np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v)


def _amplitudes(result):
    """Every amplitude of every polarization the result carries, flattened."""
    out = []
    for name in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, name, None)
        if sub is None:
            continue
        for field in ('r_s', 'r_p', 't_s', 't_p'):
            v = getattr(sub, field, None)
            if v is not None:
                out.append(np.ravel(_host(v)))
    return np.concatenate(out)


def _solver(path, algo, eps, mu, *, theta, top=1.0, bot, thickness, side='top', backend=0,
            period=P, wavelength=WL, truncation_order=None, phi=0.6):
    """A raster stack: ``eps`` / ``mu`` are ``(n_layers, Ny, Nx)`` (1D paths: ``Ny = 1``)."""
    def raster(a):
        # A tracer passes through as itself; anything NumPy can hold becomes a complex array.
        return a if hasattr(a, 'shape') and not isinstance(a, np.ndarray) else np.asarray(
            a, dtype=complex)

    common = dict(backend=backend, top=top, bot=bot, theta=theta, wavelength=wavelength,
                  thickness=thickness, layer_algorithm=algo, answers='amplitude',
                  incidence_side=side, structure=raster(eps), mu=raster(mu))
    if path in ('te', 'tm'):
        return wavebit.rcwa(**common, tetm=path, period=[period],
                            truncation_order=5 if truncation_order is None else truncation_order)
    if path == 'conical':
        return wavebit.rcwa(**common, phi=phi, psi=0.3, period=[period],
                            truncation_order=5 if truncation_order is None else truncation_order)
    return wavebit.rcwa(**common, phi=phi, psi=0.3, period=(period, 0.9 * period),
                        truncation_order=(1, 1) if truncation_order is None else truncation_order)


def _uniform(path, value, n_layers=1):
    shape = (n_layers, 1, 8) if path in ('te', 'tm', 'conical') else (n_layers, 4, 4)
    return np.full(shape, value, dtype=complex)


def _decaying_root(eps, mu, k_par):
    """A half-space's root with ``Im(kz) > 0``; on a lossless order, ``Re(kz/mu) > 0``."""
    kz = np.sqrt(complex(eps) * complex(mu) - k_par ** 2 + 0j)
    if kz.imag < 0 or (kz.imag == 0 and (kz / mu).real < 0):
        kz = -kz
    return kz


def _interface_R(theta, eps, mu):
    """Vacuum onto ``(eps, mu)``: ``(R_s, R_p)`` of the specular order, in closed form."""
    k_par = np.sin(theta)
    k1, k2 = np.sqrt(1 - k_par ** 2 + 0j), _decaying_root(eps, mu, k_par)
    r_s = (k1 - k2 / mu) / (k1 + k2 / mu)
    r_p = (k1 - k2 / eps) / (k1 + k2 / eps)
    return abs(r_s) ** 2, abs(r_p) ** 2


# ------------------------------------------------------------- a uniform layer on its own medium

UNIFORM_CASES = [(path, name, theta) for name in PAIRS for path, theta in (
    ('te', 0.0), ('te', 0.3), ('tm', 0.0), ('tm', 0.3), ('conical', 0.3), ('2d', np.deg2rad(17)))]


@pytest.mark.parametrize('path, name, theta', UNIFORM_CASES,
                         ids=[f'{p}-{n}-{t:.2f}' for p, n, t in UNIFORM_CASES])
def test_a_double_negative_layer_on_its_own_medium_is_that_medium(path, name, theta):
    """A uniform layer of the substrate's own double negative is no interface at all.

    So ETM and SMM must agree on every amplitude, and the specular ``R`` must be vacuum onto the
    half-space, in closed form. Measured 8.7e-16 on 720 such scenes (four media, five paths, two
    superstrates, three periods, three thicknesses, real and complex angles). Under Maystre's
    labels ETM raised ``LinAlgError`` on 171 of them -- 63 of the lossless medium's, 105 of the
    asymmetric lossless one's, 3 of the lossy one's, on the TE, TM, conical and 2D paths, none on
    4ff, which splits its modes by flux -- because the layer's ``+`` modes advanced their phase
    downward and the substrate received none of them. SMM raised it on 12 (see below).
    """
    eps, mu = PAIRS[name]
    kw = dict(theta=theta, bot=Material(eps, mu=mu), thickness=[150.0], period=1000.0,
              wavelength=500.0)
    etm = _solver(path, 'ETM', _uniform(path, eps), _uniform(path, mu), **kw).solve(
        auto_detune=False)
    smm = _solver(path, 'SMM', _uniform(path, eps), _uniform(path, mu), **kw).solve(
        auto_detune=False)
    assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 1e-13
    R_s, R_p = _interface_R(theta, eps, mu)
    subs = {'te': {'s': etm.incident}, 'tm': {'p': etm.incident}}.get(
        path, {'s': etm.s_inc, 'p': etm.p_inc})
    for pol, sub in subs.items():
        R = _host(sub.R)
        specular = np.unravel_index(np.argmax(R), R.shape)
        expected = R_s if pol == 's' else R_p
        # In each order's own s/p frame an isotropic interface does not mix s and p, so an s
        # (p) input reflects R_s (R_p) whatever the azimuth.
        assert abs(R[specular] - expected) < 1e-13, (pol, R[specular], expected)


@pytest.mark.filterwarnings('ignore::wavebit.ComplexIncidenceWarning')
def test_a_complex_angle_takes_the_mirror_too():
    """The mirror is the half-space's own rule, so it holds at a complex ``theta`` as well.

    ``theta = 0.3+0.05j`` on a lossless double negative, every path: ETM against SMM to 1e-13.
    The flux-based rule a patterned layer takes would not do for a uniform one here: away from
    the real axis only the cut is the limiting absorption the substrate itself follows.
    """
    eps, mu = LOSSLESS
    for path in ('te', 'tm', 'conical', '2d'):
        kw = dict(theta=0.3 + 0.05j, bot=Material(eps, mu=mu), thickness=[150.0])
        etm = _solver(path, 'ETM', _uniform(path, eps), _uniform(path, mu), **kw).solve(
            auto_detune=False)
        smm = _solver(path, 'SMM', _uniform(path, eps), _uniform(path, mu), **kw).solve(
            auto_detune=False)
        assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 1e-13, path


def test_the_mirror_labels_are_the_half_spaces_roots():
    """``etm_mode_q`` with ``mirrored`` is ``-i*kz`` of ``boundary_kz``'s double-negative root.

    That is what makes ``a`` the identity for a layer of the half-space's own medium. Checked on
    real and complex ``k_par`` over four media, propagating and evanescent orders alike, against
    the root the half-space takes; and it is the exact mirror of the plain rule,
    ``conj(etm_mode_q(conj(q**2)))``, to the bit.
    """
    k_par = np.concatenate([np.linspace(-3, 3, 41), np.linspace(-3, 3, 41) + 0.07j])
    for eps, mu in PAIRS.values():
        q2 = k_par ** 2 - eps * mu
        q = conventions.etm_mode_q(np, q2, 1.0, True)
        kz = conventions.boundary_kz(np, eps * mu, k_par ** 2, (eps, mu))
        np.testing.assert_allclose(q, -1j * kz, rtol=0, atol=1e-14)
        mirror = np.conj(conventions.etm_mode_q(np, np.conj(q2), 1.0))
        assert np.array_equal(q, mirror)
        # SMM's tie: a lossless propagating mode is labelled by where its power goes.
        tie = conventions.mode_q(np, q2, True)
        assert np.all(np.real(tie) >= 0)
        lossless = np.abs(np.real(np.sqrt(q2))) < 1e-12
        assert np.all(np.imag(tie[lossless]) >= 0)


# ------------------------------------------------------------------------------ SMM's gap medium

@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_smm_no_longer_meets_a_backward_mode_at_its_gap(tetm):
    """SMM joins layers through a gap of ``eps = 1 + kx**2``, ``mu = 1``, so ``kz_gap = 1``.

    A lossless double negative of ``mu = -1`` has, at ``kx = 1``, ``kz = 1`` too, and with
    ``eps = -2`` it is the gap's exact mirror. ``mode_q``'s old tie labelled its propagating mode
    by the phase, against the power, so the gap's wave had no ``+`` content in it and SMM raised
    ``LinAlgError`` at normal incidence (period 1000 nm, 500 nm puts orders +-2 there). The
    mirrored tie labels it by the power, as the half-space is labelled, and the gap factor is 2.
    """
    eps, mu = LOSSLESS
    kw = dict(theta=0.0, bot=Material(eps, mu=mu), thickness=[150.0], period=1000.0,
              wavelength=500.0, truncation_order=3)
    smm = _solver(tetm, 'SMM', _uniform(tetm, eps), _uniform(tetm, mu), **kw).solve(
        auto_detune=False)
    etm = _solver(tetm, 'ETM', _uniform(tetm, eps), _uniform(tetm, mu), **kw).solve(
        auto_detune=False)
    assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 1e-13
    R = _host(smm.incident.R)
    expected = _interface_R(0.0, eps, mu)[0 if tetm == 'te' else 1]
    assert abs(R.max() - expected) < 1e-13


# -------------------------------------------------------------------- a patterned double negative

NPIX = 20
MEDIA = {'air': (1.0, 1.0), 'glass': (2.25, 1.0), 'dngA0': ASYM}
PATTERNED = [
    # (path, background, inclusion, fill, theta): the double negative's limits, a small
    # inclusion in it and a small gap of ordinary medium around it, on the double negative.
    ('te', 'dngA0', 'glass', 0.05, 0.25),
    ('tm', 'glass', 'dngA0', 0.95, 0.7),
    ('conical', 'dngA0', 'air', 0.05, 0.25),
    ('2d', 'dngA0', 'air', 0.05, 0.7),
]


def _patterned(path, bg, inc, fill):
    if path == '2d':
        n = 8
        eps = np.full((1, n, n), MEDIA[bg][0], complex)
        mu = np.full((1, n, n), MEDIA[bg][1], complex)
        k = max(1, int(round(np.sqrt(fill) * n)))
        eps[0, 1:1 + k, 1:1 + k], mu[0, 1:1 + k, 1:1 + k] = MEDIA[inc]
    else:
        eps = np.full((1, 1, NPIX), MEDIA[bg][0], complex)
        mu = np.full((1, 1, NPIX), MEDIA[bg][1], complex)
        k = max(1, int(round(fill * NPIX)))
        eps[..., 2:2 + k], mu[..., 2:2 + k] = MEDIA[inc]
    return eps, mu


def _truncation(path):
    return (2, 2) if path == '2d' else 5


@pytest.mark.parametrize('path, bg, inc, fill, theta', PATTERNED,
                         ids=[c[0] for c in PATTERNED])
def test_a_patterned_double_negative_is_labelled_by_its_power(path, bg, inc, fill, theta):
    """A layer mostly double negative, or mostly ordinary, on the double negative: ETM is SMM.

    Its modes are of both kinds, so neither cut labels them all; the flux does. Measured
    against SMM before and after, on the scenes pinned here: TE 4.5e-11 -> 7.1e-16, TM 7.1e-11 ->
    6.7e-16, conical 7.2e-11 -> 9.9e-16, 2D 2.0e-09 -> 7.7e-16, and on 360 random patterned
    scenes on all four paths the worst real-angle disagreement away from normal incidence went
    from ``LinAlgError`` (25 of them) to 2.3e-13.
    """
    eps, mu = _patterned(path, bg, inc, fill)
    kw = dict(theta=theta, bot=Material(*ASYM), thickness=[40.0],
              truncation_order=_truncation(path))
    etm = _solver(path, 'ETM', eps, mu, **kw).solve(auto_detune=False)
    smm = _solver(path, 'SMM', eps, mu, **kw).solve(auto_detune=False)
    assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 2e-13


BLOCK_CASES = [('te', 0.5, (-25 + 0.5j, -1 + 0.02j)), ('tm', 0.5 + 0.3j, (-25 + 0j, -1 + 0j)),
               ('conical', 0.5, (-25 + 0j, -1 + 0j)), ('conical', 0.5 + 0.3j, (-25 + 0j, -1 + 0j))]


@pytest.mark.filterwarnings('ignore::wavebit.ComplexIncidenceWarning')
@pytest.mark.parametrize('path, theta, block', BLOCK_CASES,
                         ids=[f'{p}-{t}-{b[0].imag:g}' for p, t, b in BLOCK_CASES])
def test_a_high_contrast_double_negative_block_on_its_own_medium(path, theta, block):
    """``docs/connecting-algorithms.md`` section 10's scene, its ``eps = 25`` block made the double
    negative ``(-25, -1)``: permittivities ``[1, X, 1, X, X, X]`` over 2000 nm, 100 nm thick, on a
    half-space of ``X``, ``truncation_order = 30``, 900 nm.

    The wide block rests on its own medium, so its modes cross into the substrate unreflected and
    their labels decide whether ``a`` is invertible. Under Maystre's labels ETM was off SMM by
    7.3e-07 (TE, lossy block), 2.7e-04 (TM, ``theta = 0.5+0.3j``), 1.4e-05 and 1.2e-04 (conical);
    with the flux it is 1e-13. Labelling by decay instead -- what SMM does -- was still off by
    4e-11 to 1e-10 at the complex angle: there a propagating mode's decaying root travels up,
    which is section 10's reason for ETM's own rule, and why the per-mode rule reads the flux.
    """
    pattern = (1, block, 1, block, block, block)
    eps = np.ones((1, 1, 60), complex)
    mu = np.ones((1, 1, 60), complex)
    for i, value in enumerate(pattern):
        if value != 1:
            eps[..., 10 * i:10 * (i + 1)], mu[..., 10 * i:10 * (i + 1)] = value
    kw = dict(theta=theta, bot=Material(*block), thickness=[100.0], period=2000.0,
              wavelength=900.0, truncation_order=30, phi=0.3)
    etm = _solver(path, 'ETM', eps, mu, **kw).solve(auto_detune=False)
    smm = _solver(path, 'SMM', eps, mu, **kw).solve(auto_detune=False)
    assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 1e-12


def test_the_flux_labels_carry_power_down():
    """Every mode a patterned double negative's rule labels ``+`` carries power down or decays.

    Reads the labels straight off the mode builder, on the TE scene above: each propagating
    (``|Im q| > |Re q|``) mode's flux is positive, each other mode decays. The control is the
    builder without the engine's word (``dng=None``), which keeps Maystre's labels and sends
    the backward modes up.
    """
    from wavebit._core.solve_1d import layer_modes_1d, mode_flux
    eps, mu = _patterned('te', 'dngA0', 'glass', 0.05)
    solver = _solver('te', 'ETM', eps, mu, theta=0.25, bot=Material(*ASYM), thickness=[40.0])
    conv, mu_conv = solver._eps_conv_matrices(), solver._mu_conv_matrices()
    kx = solver._separable_kxy(solver.wavelength)[0].astype(complex)
    I = np.eye(len(kx), dtype=complex)
    k0_d = 2 * np.pi / WL * 40.0

    def modes(dng):
        return layer_modes_1d(np, 0, kx, I, conv[0][0], conv[1][0], conv[2][0], complex,
                              mu=tuple(m[0] for m in mu_conv[:3]), k0_d=k0_d, dng=dng)

    def carrying(W, V, q):
        # Propagation-dominated modes whose flux stands clear of round-off, and the flux.
        flux, energy = mode_flux(np, V, W)
        return (np.abs(q.imag) > np.abs(q.real)) & (np.abs(flux) > 1e-12 * energy), flux

    W, V, q = modes(DoubleNegativeLayer(True, False))
    sure, flux = carrying(W, V, q)
    assert sure.sum() >= 3 and np.all(flux[sure] > 0)
    assert np.all(q.real[np.abs(q.imag) <= np.abs(q.real)] > 0)
    sure, flux = carrying(*modes(None))
    assert np.any(flux[sure] < 0)


def test_smm_breaks_a_patterned_layers_tie_by_the_flux_too():
    """SMM keeps the decaying root and so never meets a growing one; only a lossless propagating
    mode (``Re(q)`` flat) is left to choose, and a patterned double negative's is chosen by the
    flux, as a uniform one's is by the mirrored tie. ``mode_q``'s ``Im(q) <= 0`` sends the
    backward ones up, which at a gap-matched admittance is SMM's singular factor."""
    from wavebit._core.solve_1d import layer_modes_1d, mode_flux
    eps, mu = _patterned('te', 'dngA0', 'glass', 0.05)
    solver = _solver('te', 'SMM', eps, mu, theta=0.25, bot=Material(*ASYM), thickness=[40.0])
    conv, mu_conv = solver._eps_conv_matrices(), solver._mu_conv_matrices()
    kx = solver._separable_kxy(solver.wavelength)[0].astype(complex)
    I = np.eye(len(kx), dtype=complex)
    for dng, backward_up in ((DoubleNegativeLayer(True, False), False), (None, True)):
        W, V, q = layer_modes_1d(np, 0, kx, I, conv[0][0], conv[1][0], conv[2][0], complex,
                                 mu=tuple(m[0] for m in mu_conv[:3]), dng=dng)
        assert np.all(q.real >= -1e-12)     # decaying, up to a tie's round-off
        flux, energy = mode_flux(np, V, W)
        tie = (np.abs(q.real) <= 1e-12) & (np.abs(flux) > 1e-12 * energy)
        assert tie.sum() >= 3
        assert np.any(flux[tie] < 0) == backward_up


def test_the_flux_rule_keeps_decay_where_decay_dominates_and_the_cap():
    """Two guards of ``power_down_labels``, on hand-made modes.

    Where a mode's decay dominates (``|Re q| >= |Im q|``) its flux is not read: a truncated
    operator can give an evanescent mode a small flux of either sign (the reason the 4ff split
    reads attenuation there), and following it would hand ETM a root growing by ``exp(Re(q)*k0*d)``.
    And a propagating mode whose power-down root grows past ``eps**-0.25`` across the layer keeps
    its decaying root under ETM, as :func:`etm_mode_q` does; SMM never takes a growing root."""
    q = np.array([1.0 + 0.2j, 0.001 + 1.0j, 0.3 + 1.0j])
    flux = np.array([-0.5, -0.5, -0.5])        # all resolvable, all pointing up
    energy = np.ones(3)
    labels, flip = conventions.power_down_labels(np, q, flux, energy, 1.0, True)
    assert list(flip) == [False, True, True]   # decay-dominated: kept; propagating: turned
    # Thick enough that the third mode's turned root grows past eps**-0.25: kept decaying.
    labels, flip = conventions.power_down_labels(np, q, flux, energy, 200.0, True)
    assert list(flip) == [False, True, False]
    # SMM (no thickness): only a flat real part is a tie the flux may break.
    labels, flip = conventions.power_down_labels(np, np.array([1e-14 + 1.0j, 0.3 + 1.0j]),
                                                 np.array([-0.5, -0.5]), np.ones(2), None, True)
    assert list(flip) == [True, False]


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('path, bg, inc, fill, theta', PATTERNED,
                         ids=[c[0] for c in PATTERNED])
def test_the_labels_travel_with_their_layers_through_the_mirrored_scene(
        path, bg, inc, fill, theta, side):
    """The engine solves public top incidence as the kernels' z-mirrored scene, which reverses
    the layers, so its word on each layer has to reverse with them.

    Two layers, the patterned double negative next to a double-negative half-space and a glass
    layer on the lit side, lit from either side. Lit from the top, the kernels see the stack
    reversed; without the reversal the patterned layer would take the glass layer's word --
    none -- and Maystre's labels, 5e-11 (TE, TM, conical) and 1e-9 (2D) off SMM.
    """
    eps0, mu0 = _patterned(path, bg, inc, fill)
    glass, one = np.full_like(eps0, 2.25), np.ones_like(mu0)
    if side == 'top':                     # the double negative at the bottom, on its medium
        eps, mu = np.concatenate([eps0, glass]), np.concatenate([mu0, one])
        kw = dict(top=1.0, bot=Material(*ASYM), thickness=[40.0, 60.0])
    else:                                 # at the top, under its medium, lit from below
        eps, mu = np.concatenate([glass, eps0]), np.concatenate([one, mu0])
        kw = dict(top=Material(*ASYM), bot=1.0, thickness=[60.0, 40.0])
    kw.update(theta=theta, side=side, truncation_order=_truncation(path))
    etm = _solver(path, 'ETM', eps, mu, **kw).solve(auto_detune=False)
    smm = _solver(path, 'SMM', eps, mu, **kw).solve(auto_detune=False)
    assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 2e-13


def test_solve_conv_with_the_scenes_own_matrices_is_solve():
    """``solve_conv(*eps_conv_matrices(), mu=mu_conv_matrices(), ...)`` is ``solve()``, to the bit.

    The contract holds for a double-negative scene too: ``solve_conv`` reads the scene's
    materials for the labels when the layer counts match. Only labels: a stale word would cost
    conditioning, never the answer.
    """
    eps, mu = _patterned('2d', 'dngA0', 'air', 0.05)
    solver = _solver('2d', 'ETM', eps, mu, theta=0.7, bot=Material(*ASYM), thickness=[40.0],
                     truncation_order=(2, 2))
    direct = _amplitudes(solver.solve(auto_detune=False))
    relayed = _amplitudes(solver.solve_conv(
        *solver.eps_conv_matrices(), mu=solver.mu_conv_matrices(), mu_top=solver.top.mu,
        mu_bot=solver.bot.mu, eps_top=solver.top.epsilon, eps_bot=solver.bot.epsilon))
    assert direct.tobytes() == relayed.tobytes()


def test_a_shape_layer_is_read_from_its_materials():
    """The parametric path names its double negative by ``Material``: a disc of glass in a
    double-negative background, on the double negative, ETM against SMM."""
    layer = {'background': Material(*ASYM),
             'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0, 'lx': 240.0,
                         'material': Material(2.25, mu=1.0)}]}
    out = {}
    for algo in ('ETM', 'SMM'):
        s = wavebit.rcwa(backend=0, top=1.0, bot=Material(*ASYM), theta=0.5, phi=0.6, psi=0.3,
                         wavelength=WL, period=(P, P), thickness=[40.0], truncation_order=(2, 2),
                         layer_algorithm=algo, answers='amplitude', structure=[layer])
        out[algo] = _amplitudes(s.solve(auto_detune=False))
    assert np.abs(out['ETM'] - out['SMM']).max() < 1e-12


# ---------------------------------------------------------------------- what the engine says

def test_the_engine_reads_each_layer_from_its_materials():
    """Per layer: does it hold a double negative, and is it one medium -- for scalar rasters, a
    tensor raster, a 3x3 scalar raster (which is not a tensor raster), and shape layers."""
    s = _solver('te', 'ETM', np.stack([_uniform('te', -2.0)[0], np.full((1, 8), 2.25),
                                       np.r_[[[-2.0] * 4 + [1.0] * 4]]]),
                np.stack([_uniform('te', -1.0)[0], np.full((1, 8), 1.0),
                          np.r_[[[-1.0] * 4 + [1.0] * 4]]]),
                theta=0.3, bot=1.0, thickness=[10.0, 10.0, 10.0])
    assert s._layer_double_negative() == (DoubleNegativeLayer(True, True),
                                          DoubleNegativeLayer(False, True),
                                          DoubleNegativeLayer(True, False))
    # Only where both are negative at one point: an eps-negative pixel beside a mu-negative
    # one is not a double negative.
    apart = _solver('te', 'ETM', np.r_[[[[-2.0] * 4 + [2.0] * 4]]],
                    np.r_[[[[1.0] * 4 + [-1.0] * 4]]], theta=0.3, bot=1.0, thickness=[10.0])
    assert apart._layer_double_negative() == (DoubleNegativeLayer(False, False),)
    # A 3x3 *scalar* raster: a 3x3 cell, not a tensor per pixel.
    grid = np.full((1, 3, 3), -2.0, complex)
    grid[0, 0, 0] = 1.0
    s3 = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.3, wavelength=WL, period=(P, P),
                      thickness=[10.0], truncation_order=(1, 1), structure=grid,
                      mu=np.full((1, 3, 3), -1.0, complex))
    assert s3._layer_double_negative() == (DoubleNegativeLayer(True, False),)
    # Non-magnetic: no word at all.
    plain = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.3, wavelength=WL, period=[P],
                         thickness=[10.0], truncation_order=2, tetm='te',
                         structure=np.full((1, 1, 8), -2.0, complex))
    assert plain._layer_double_negative() is None
    # Shapes: a background with no shapes is uniform; a disc of another material is not.
    layers = [{'background': Material(*ASYM)},
              {'background': Material(*ASYM), 'shapes': [{'type': 'circle', 'cx': 0.0, 'cy': 0.0,
                                                          'lx': 100.0, 'material': 2.25}]},
              {'background': Material(2.0, mu=1.5)}]
    shapes = wavebit.rcwa(backend=0, top=1.0, bot=1.0, theta=0.3, wavelength=WL, period=(P, P),
                          thickness=[10.0] * 3, truncation_order=(1, 1), structure=layers)
    assert shapes._layer_double_negative() == (DoubleNegativeLayer(True, True),
                                               DoubleNegativeLayer(True, False),
                                               DoubleNegativeLayer(False, True))


PRINCIPAL = {
    # Three principal values beside three, or beside a scalar with a negative real part: one
    # flag per material, whatever its spelling. Each of these broadcast a (3,) flag against
    # the modes when the double-negative reader first landed, ETM and SMM alike.
    'ordinary': ([2.0, 3.0, 4.0], [1.1, 1.3, 1.2]),
    'eps_three_mu_negative': ([2.0, 3.0, 4.0], -1 + 0.1j),
    'eps_negative_mu_three': (-2 + 0.5j, [1.1, 1.3, 1.2]),
    'double_negative': ([-2 + 0.1j, -3 + 0.1j, -2.5 + 0.1j], [-1 + 0.1j, -1.2 + 0.1j, -1.1 + 0.1j]),
    'one_component_positive': ([-2 + 0.1j, 3 + 0.1j, -2.5 + 0.1j], [-1 + 0.1j, -1.2 + 0.1j, -1.1 + 0.1j]),
}


def _diagonal(value):
    return np.diag(np.asarray(value, dtype=complex)) if np.ndim(value) == 1 else value


@pytest.mark.parametrize('name', list(PRINCIPAL))
def test_three_principal_values_are_one_flag_and_solve_as_their_tensor(name):
    """``Material([a, b, c], mu=...)`` is the diagonal tensor ``diag(a, b, c)`` spelled short. The
    reader says one thing per material for either spelling -- a double negative only if every
    principal component of both is negative -- and both spellings solve alike, ETM and SMM,
    as a disc and as a uniform background."""
    eps, mu = PRINCIPAL[name]
    holds = name == 'double_negative'

    def solver(material, algo, shape):
        layer = ({'background': 1.0, 'shapes': [{'type': 'rectangle', 'lx': 150.0, 'ly': 150.0,
                                                 'material': material}]}
                 if shape else {'background': material})
        return wavebit.rcwa(backend=0, top=1.0, bot=2.25, theta=0.2, phi=0.1,
                            polarization=(1.0, 0.0), wavelength=600.0, period=(400.0, 400.0),
                            truncation_order=(1, 1), thickness=[100.0], structure=[layer],
                            layer_algorithm=algo, answers='amplitude')

    for shape in (True, False):
        short = solver(Material(eps, mu=mu), 'ETM', shape)
        assert short._layer_double_negative() == (DoubleNegativeLayer(holds, not shape),)
        out = {}
        for algo in ('ETM', 'SMM'):
            for spelling, material in (('short', Material(eps, mu=mu)),
                                       ('tensor', Material(_diagonal(eps), mu=_diagonal(mu)))):
                out[algo, spelling] = _amplitudes(
                    solver(material, algo, shape).solve(auto_detune=False))
        scale = np.abs(out['SMM', 'tensor']).max()
        for key, value in out.items():
            assert np.abs(value - out['SMM', 'tensor']).max() < 1e-12 * scale, (name, shape, key)


def test_an_ordinary_layer_takes_no_new_rule():
    """Every layer without a double negative is handed back the plain rule, structurally:
    ``(None, None)``, so its arithmetic is the one it had -- a non-magnetic layer, a magnetic one
    the engine calls ordinary, and a magnetic one nobody said anything about whose slots are not
    a uniform double negative."""
    I = np.eye(3, dtype=complex)
    assert layer_label_rule(np, None, (I, I, I), None) == (None, None)
    assert layer_label_rule(np, DoubleNegativeLayer(False, True), (-2 * I,) * 3,
                            (-I,) * 3) == (None, None)
    assert layer_label_rule(np, None, (2 * I,) * 3, (-I,) * 3) == (None, None)
    patterned = np.diag([-2.0, 1.0, -2.0]).astype(complex)
    assert layer_label_rule(np, None, (patterned,) * 3, (-I,) * 3) == (None, None)
    assert layer_label_rule(np, None, (-2 * I,) * 3, (-I,) * 3) == (True, None)
    assert layer_label_rule(np, DoubleNegativeLayer(True, False), (patterned,) * 3,
                            (-I,) * 3) == (None, True)


# ------------------------------------------------------------------------------- under jax.jit

def test_a_traced_double_negative_layer_is_selected_not_branched_on():
    """Under ``jax.jit`` the raster is a tracer, so whether a layer is a uniform or a patterned
    double negative is a traced boolean, and both rules are selected with ``where``. ETM through
    the trace matches NumPy on a uniform layer on its own medium (whose ETM was singular under
    Maystre's labels) and on the patterned 2D scene."""
    pytest.importorskip('jax')
    wavebit.enable_x64()
    import jax
    import jax.numpy as jnp

    eps_u, mu_u = ASYM
    eps_p, mu_p = _patterned('2d', 'dngA0', 'air', 0.05)
    cases = (
        ('uniform', lambda x, b: _solver('te', 'ETM', _uniform('te', eps_u) + 0 * x + x,
                                         _uniform('te', mu_u), theta=0.3, backend=b,
                                         bot=Material(eps_u, mu=mu_u), thickness=[150.0],
                                         period=1000.0, wavelength=500.0)),
        ('patterned', lambda x, b: _solver('2d', 'ETM', eps_p + x * (eps_p.real < 0), mu_p,
                                           theta=0.7, backend=b, bot=Material(*ASYM),
                                           thickness=[40.0], truncation_order=(2, 2))),
    )
    for name, build in cases:
        eager = _amplitudes(build(0.0, 0).solve(auto_detune=False))
        traced = jax.jit(lambda x: jnp.asarray(_amplitudes_jax(build(x, 1).solve(
            auto_detune=False))))(jnp.asarray(0.0))
        assert np.abs(np.asarray(traced) - eager).max() < 1e-13, name


def _amplitudes_jax(result):
    import jax.numpy as jnp
    out = []
    for name in ('incident', 's_inc', 'p_inc'):
        sub = getattr(result, name, None)
        if sub is None:
            continue
        for field in ('r_s', 'r_p', 't_s', 't_p'):
            v = getattr(sub, field, None)
            if v is not None:
                out.append(jnp.ravel(v))
    return jnp.concatenate(out)
