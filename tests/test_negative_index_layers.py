"""A double-negative *layer*: how its modes are labelled, and the one pair ETM cannot write.

A layer mode comes as a ``+-q`` pair and the recursion has to call one root "down". In exact
arithmetic the choice is free; ETM is not free to make it, because it writes each layer through
the transmission from its downward modes into the region below
(``docs/connecting-algorithms.md`` section 10). For an ordinary layer the half-space's root
(Maystre's cut) is that wave. For a double negative it is the wrong one: Maystre's root
advances the phase downward, and a double negative carries its power against its phase
(V. G. Veselago, Sov. Phys. Usp. 10, 509 (1968)). So a uniform double-negative layer resting on
the same medium made ETM's interface factor singular -- ``LinAlgError`` on 171 of 720 such
scenes -- and SMM met the same thing at its gap whenever a backward mode's admittance mirrored
the gap's.

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

# A lossless double negative whose backward wave at ``kx = 1`` is the gap's exact mirror: its
# ``kz = sqrt(2)``, so ``kz/mu`` (TE) or ``kz/eps`` (TM) is ``-sqrt(2)``, the gap's negative.
GAP_MIRROR = {'te': (-3.0 + 0j, -1.0 + 0j), 'tm': (-1.0 + 0j, -3.0 + 0j)}


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_smm_no_longer_meets_a_backward_mode_at_its_gap(tetm, monkeypatch):
    """SMM joins layers through a gap whose waves have ``kz/mu`` (s) and ``kz/eps`` (p) equal to
    ``sqrt(1 + |k_par|**2)`` (``scattering._gap_waves``), ``sqrt(2)`` at ``kx = 1``.

    A lossless double negative of ``(-3, -1)`` has, at ``kx = 1``, ``kz = sqrt(2)``, so its TE
    wave's ``kz/mu`` is the gap's exact mirror; ``(-1, -3)`` is the TM one. ``mode_q``'s old tie
    labelled its propagating mode by the phase, against the power, so the gap's wave had no
    ``+`` content in it and SMM raised ``LinAlgError`` at normal incidence (period 1000 nm,
    500 nm puts orders +-2 there) -- the control. The mirrored tie labels it by the power, as the
    half-space is labelled, and the gap factor is 2. (Until 2026-10-09 the gap's ``kz/mu`` was
    1 and its ``kz/eps`` ``1/(1 + k_par**2)``, and ``(-2, -1)`` mirrored both.)
    """
    from wavebit._core import solve_1d
    eps, mu = GAP_MIRROR[tetm]
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

    monkeypatch.setattr(solve_1d, 'layer_label_rule', lambda *a, **k: (None, None))
    with pytest.raises(np.linalg.LinAlgError):
        _solver(tetm, 'SMM', _uniform(tetm, eps), _uniform(tetm, mu), **kw).solve(
            auto_detune=False)


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


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_the_4ff_split_reads_flux_at_the_working_precision(backend):
    """A lossless patterned double negative has complex quadruplets of modes (``lam``, ``-lam``
    and their conjugates) that carry no flux and are propagation-dominated (``|Re| > |Im|``).
    In single precision their computed flux is ~1e-7 of either sign, and the 4ff split's
    threshold of 1e-12 -- about 4500 ulp of float64 -- read it as a direction: JAX and Torch
    found 17 and 16 forward modes of 18 and refused the layer at ``complex_dtype=1``. The
    threshold is now that many ulp of the working dtype, the same 1e-12 in double precision.
    The single-precision answer is held to the double-precision one."""
    pytest.importorskip({0: 'numpy', 1: 'jax', 2: 'torch'}[backend])
    if backend == 1:
        wavebit.enable_x64()
    mask = np.zeros((6, 6))
    mask[1:4, 2:5] = 1
    glass = np.zeros((6, 6))
    glass[4, 0:2] = 1
    eps = (mask * -2.5 + glass * 2.25 + (1 - mask - glass))[None].astype(complex)
    mu = (mask * -1.5 + (1 - mask))[None].astype(complex)

    def reflected(backend, complex_dtype):
        s = wavebit.rcwa(backend=backend, top=2.25, bot=2.25, wavelength=500.0, theta=0.35,
                         phi=0.6, psi=0.3, period=(700.0, 630.0), truncation_order=(1, 1),
                         thickness=[120.0], structure=eps, complex_dtype=complex_dtype,
                         layer_algorithm='SMM')
        s.mu = mu
        s.formulation = '4ff'
        return np.asarray(_host(s.solve(auto_detune=False).R), dtype=float)

    single = reflected(backend, 1)
    double = reflected(0, 0)
    np.testing.assert_allclose(single, double, rtol=0, atol=1e-5)


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


# ------------------------------------------------------------------------- Pendry's perfect lens

def _lens(algo, delta=0.0, backend=0, thickness=100.0):
    """A uniform ``(-2, -1)`` layer on its double-positive mirror ``(2, 1)``, lit from vacuum."""
    nim = (-2 + 1j * delta, -1 + 1j * delta)
    return _solver('te', algo, _uniform('te', nim[0]), _uniform('te', nim[1]), theta=0.25,
                   bot=Material(2.0, mu=1.0), thickness=[thickness], backend=backend,
                   truncation_order=4)


def test_the_perfect_lens_pair_is_finite_and_smm_has_it():
    """The single interface is singular; a layer of it is not.

    Between ``(eps, mu)`` and ``(-eps, -mu)`` the admittances of every evanescent order cancel,
    ``kz/mu + kz/(-mu) = 0``, so the semi-infinite pair has a surface mode at every evanescent
    ``k_par``: its ``r`` and ``t`` there are poles (J. B. Pendry, Phys. Rev. Lett. 85, 3966
    (2000)). A finite layer on its mirror instead carries only the mode that grows toward the
    mirror, and transmits an evanescent order with ``t = 2*ka/(ka + kn) * exp(kn*k0*d)`` --
    Pendry's amplification, finite at any thickness. SMM's scattering matrix has exactly that,
    order by order, and its plane-wave answer is the limit of the lossy one as the loss goes
    to zero, linearly.
    """
    solver = _lens('SMM')
    base = _amplitudes(solver.solve(auto_detune=False))
    S21 = _host(solver.scattering.S21)
    kx = _host(solver.order_kxy()[0]).ravel().real
    k0, d = 2 * np.pi / WL, 100.0
    checked = 0
    for m, k in enumerate(kx):
        if k ** 2 <= 2.0:
            continue                         # propagating in the lens: no amplification
        ka, kn = np.sqrt(k ** 2 - 1), np.sqrt(k ** 2 - 2)
        closed = 2 * ka / (ka + kn) * np.exp(kn * k0 * d)
        assert abs(abs(S21[m, m]) - closed) < 1e-12 * closed, (k, abs(S21[m, m]), closed)
        checked += 1
    assert checked >= 4 and closed > 10
    for delta in (1e-9, 1e-7):
        moved = np.abs(_amplitudes(_lens('SMM', delta).solve(auto_detune=False)) - base).max()
        assert 1e-3 * delta < moved < 1e3 * delta


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_etm_names_the_perfect_lens_pair_where_it_cannot_write_it(backend):
    """ETM writes a layer through its transmission into the medium below, and on Pendry's pair
    that is zero on every evanescent order. Here the zero comes out exact and the inversion
    fails -- ``LinAlgError`` on NumPy and Torch, a non-finite result on JAX -- and the solve says
    which pair it is and that SMM solves it, rather than retrying the wavelength for nothing."""
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    if backend == 2:
        pytest.importorskip('torch')
    with pytest.raises(ValueError, match='perfect-lens pair') as caught:
        _lens('ETM', backend=backend).solve()
    assert caught.value.__cause__ is not None
    assert 'Singular' in str(caught.value.__cause__) or 'non-finite' in str(
        caught.value.__cause__) or 'singular' in str(caught.value.__cause__)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _lens('SMM', backend=backend).solve(auto_detune=False)


def test_a_lossy_lens_is_not_the_pair():
    """Any loss breaks the exact cancellation, and ETM solves the layer again -- the message is
    for the exact pair only, and only where ETM actually failed on it."""
    etm = _amplitudes(_lens('ETM', 1e-9).solve(auto_detune=False))
    smm = _amplitudes(_lens('SMM', 1e-9).solve(auto_detune=False))
    assert np.abs(etm - smm).max() < 1e-13


def _lens_on_its_negation(algo, backend=0):
    """The roles swapped: a uniform ``(2, 1)`` layer on a ``(-2, -1)`` substrate, lit from vacuum."""
    return _solver('te', algo, _uniform('te', 2.0), _uniform('te', 1.0), theta=0.25,
                   bot=Material(-2.0 + 0j, mu=-1.0 + 0j), thickness=[100.0], backend=backend,
                   truncation_order=4)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_etm_names_the_pair_when_the_double_negative_is_the_substrate(backend):
    """The same pair with the double negative as the half-space: ETM writes the ordinary layer
    through its transmission into the substrate, ``(1 + Y_below/Y_layer)/2``, zero on every
    evanescent order. ETM raised a bare ``LinAlgError`` here (NumPy, PyTorch; non-finite on
    JAX) on the TE, TM and 2D paths, because the message only looked for a double-negative
    *layer*; now it names the pair and points to SMM.

    SMM's answer is a closed form. The layer cancels an equal thickness of the substrate (J. B.
    Pendry and S. A. Ramakrishna, J. Phys.: Condens. Matter 15, 6345 (2003)), so the stack
    reflects as vacuum onto ``(-2, -1)`` does; and the lens the other way round,
    ``(-2, -1)`` on ``(2, 1)``, reflects as vacuum onto ``(2, 1)``, the same ``R``."""
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    if backend == 2:
        pytest.importorskip('torch')
    with pytest.raises(ValueError, match='perfect-lens pair') as caught:
        _lens_on_its_negation('ETM', backend=backend).solve()
    assert 'the bottom half-space is a uniform double negative' in str(caught.value)
    assert caught.value.__cause__ is not None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        swapped = _lens_on_its_negation('SMM', backend=backend).solve(auto_detune=False)
        lens = _lens('SMM', backend=backend).solve(auto_detune=False)
    R_s, _ = _interface_R(0.25, -2.0 + 0j, -1.0 + 0j)
    for result in (swapped, lens):
        R = _host(result.incident.R)
        assert abs(R.sum() - R_s) < 1e-13 and abs(R.max() - R_s) < 1e-13


# ------------------------------------------------- two half-spaces that are each other's negation

NEGATION = (3.0 + 0j, 1.5 + 0j)          # (eps, mu); its negation is the double negative
NEGATED = (-3.0 + 0j, -1.5 + 0j)
STACKS = {'none': (), 'own': ('top',), 'both': ('bot', 'top'), 'three': ('top', 'bot', 'top')}
#: Truncations that retain evanescent orders at ``theta = 0.3``: 6 of 11 on the 1D paths at their
#: default, 9 of 25 on the 2D ones at ``(2, 2)`` -- their default ``(1, 1)`` retains none.
EVANESCENT_KEPT = {'te': None, 'tm': None, 'conical': None, '2d': (2, 2), '4ff': (2, 2)}


def _negation_scene(path, algo, stack, *, flip=False, side='top', backend=0, bot=None,
                    theta=0.3, delta=0.0, period=P, truncation_order=None):
    """``NEGATION`` above ``NEGATED`` (``flip``: below it), with uniform layers of the two named
    by ``stack``; ``delta`` adds a loss to the double negative wherever it is."""
    lossy = (NEGATED[0] + 1j * delta, NEGATED[1] + 1j * delta)
    top, below = (lossy, NEGATION) if flip else (NEGATION, lossy)
    below = below if bot is None else bot
    media = {'top': top, 'bot': below}
    names = STACKS[stack] or ('top',)           # the empty stack is stripped after the build
    eps = np.stack([_uniform(path, media[n][0])[0] for n in names])
    mu = np.stack([_uniform(path, media[n][1])[0] for n in names])
    solver = _solver('2d' if path == '4ff' else path, algo, eps, mu, theta=theta,
                     top=Material(top[0], mu=top[1]), bot=Material(below[0], mu=below[1]),
                     thickness=[80.0, 120.0, 50.0][:len(names)], side=side, backend=backend,
                     period=period, truncation_order=truncation_order)
    if path == '4ff':
        solver.formulation = '4ff'
    if STACKS[stack]:
        return solver, lambda: solver.solve(auto_detune=False)
    return solver, lambda: _without_layers(solver)


def _without_layers(solver):
    """The same scene with no layer at all, through ``solve_conv`` and its half-space pairs."""
    eps, mu = solver.eps_conv_matrices(), solver.mu_conv_matrices()
    solver.thickness = []
    return solver.solve_conv(*(None if a is None else a[:0] for a in eps),
                             mu=tuple(None if a is None else a[:0] for a in mu),
                             eps_top=solver.top.epsilon, mu_top=solver.top.mu,
                             eps_bot=solver.bot.epsilon, mu_bot=solver.bot.mu)


NEGATION_CASES = [(path, algo, stack, flip, side)
                  for path in ('te', 'tm', 'conical', '2d', '4ff') for algo in ('ETM', 'SMM')
                  for stack in ('none', 'own', 'three') for flip in (False, True)
                  for side in ('top', 'bottom') if (stack, flip) != ('three', True)]


@pytest.mark.parametrize('path, algo, stack, flip, side', NEGATION_CASES,
                         ids=['-'.join(map(str, c)) for c in NEGATION_CASES])
def test_two_half_spaces_that_are_each_others_lossless_negation_are_refused(
        path, algo, stack, flip, side):
    """``(eps, mu)`` against ``(-eps, -mu)``, both lossless, as the two half-spaces: on every
    evanescent order the decaying waves share ``kz`` and their admittances cancel, so the
    interface reflects with a pole at every evanescent ``k_par`` and the scattering matrix does
    not exist -- the time-harmonic problem has no solution (B. Gralak and D. Maystre, C. R.
    Physique 13, 786 (2012); M. Cassier and P. Joly, arXiv:2504.00235 (2025)). Uniform layers of
    the two media are the same pair, its interface moved: a layer of ``(-eps, -mu)`` cancels an
    equal thickness of ``(eps, mu)`` (J. B. Pendry and S. A. Ramakrishna, J. Phys.: Condens.
    Matter 15, 6345 (2003)).

    Before this was refused the solve did not answer it: with no layer every path raised
    ``LinAlgError`` (NumPy), ``FloatingPointError`` (JAX) or, on Torch, all but 2D and 4ff SMM;
    with layers SMM returned a finite answer whose 4ff evanescent amplitudes were round-off of
    order one where they must be zero, and ETM raised ``LinAlgError`` or a perfect-lens message
    that called the stack finite. Now every path, either algorithm, either side lit and either
    medium on top refuses it before anything is solved, naming both media and how many of the
    retained orders are evanescent.
    """
    solver, solve = _negation_scene(path, algo, stack, flip=flip, side=side,
                                    truncation_order=EVANESCENT_KEPT[path])
    from wavebit._core.solve_2d import order_grid_kxy
    gx, gy = order_grid_kxy(np, *solver.order_kxy())
    evanescent = int(np.sum((NEGATION[0] * NEGATION[1] - gx * gx - gy * gy).real < 0))
    with pytest.raises(ValueError, match="each other's exact negation") as caught:
        solve()
    message = str(caught.value)
    for value in (NEGATION, NEGATED):
        assert f'eps = {value[0]:g}, mu = {value[1]:g}' in message
    assert 'Gralak and D. Maystre' in message and 'arXiv:2504.00235' in message
    assert ('no layer between' in message) == (stack == 'none')
    assert f'{evanescent} of the {gx.size} retained orders are evanescent' in message
    assert 0 < evanescent < gx.size


@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', '4ff'])
def test_the_negation_pair_with_no_evanescent_order_retained_is_solved(path):
    """Only a retained evanescent order makes the pair singular: on a propagating one the mirror
    root is ``-kz``, the admittances are equal and the order is matched, and uniform layers
    couple no two orders. So a truncation that keeps only propagating orders -- the specular one
    alone, or every order of a long period -- is solved, with the matched answer the lossy
    neighbours converge to (``R = 0``, ``T = 1``), every path, ETM and SMM, either side lit,
    either medium on top, with and without layers. The refusal used to stop these too, saying
    that retained evanescent orders made the system singular when there were none.

    Its control: the same scenes lit by an evanescent wave (``theta = pi/2 - 0.5i``), whose one
    retained order is evanescent, are still refused."""
    one_d = path in ('te', 'tm', 'conical')
    specular, long_period = (0, 3) if one_d else ((0, 0), (1, 1))
    for algo in ('ETM', 'SMM'):
        for stack, flip in (('none', False), ('own', False), ('own', True), ('three', False)):
            for side in ('top', 'bottom'):
                for period, truncation_order in ((P, specular), (10 * P, long_period)):
                    common = dict(flip=flip, side=side, period=period,
                                  truncation_order=truncation_order)
                    solver, solve = _negation_scene(path, algo, stack, **common)
                    result = solve()
                    sub = result.incident if result.incident is not None else result.s_inc
                    assert float(np.sum(_host(sub.R)).real) < 1e-28
                    assert abs(float(np.sum(_host(sub.T)).real) - 1) < 1e-13
                    lossy = _negation_scene(path, algo, stack, delta=1e-12, **common)[1]()
                    assert np.abs(_amplitudes(result) - _amplitudes(lossy)).max() < 1e-10
                    with pytest.raises(ValueError, match="each other's exact negation") as caught:
                        _negation_scene(path, algo, stack, theta=np.pi / 2 - 0.5j,
                                        flip=flip, side=side, truncation_order=specular)[1]()
                    assert '1 of the 1 retained orders is evanescent' in str(caught.value)


@pytest.mark.parametrize('backend', [0, 1, 2], ids=['numpy', 'jax', 'torch'])
def test_the_negation_pair_is_refused_on_every_backend(backend):
    """The test reads values, so it runs on JAX and Torch eagerly as on NumPy -- and under an
    un-jitted ``jax.grad``, whose tracer carries its value -- and the spellings of the pair that
    reach it differently are refused too: a half-space that names no permeability (``mu = 1``),
    a layer that names none, and a ``solve_conv`` pair whose permittivity is left to
    ``n**2/mu``."""
    if backend == 1:
        pytest.importorskip('jax')
        wavebit.enable_x64()
    if backend == 2:
        pytest.importorskip('torch')
    for path, stack in (('te', 'none'), ('2d', 'three'), ('conical', 'own')):
        with pytest.raises(ValueError, match="each other's exact negation"):
            _negation_scene(path, 'SMM', stack, backend=backend,
                            truncation_order=EVANESCENT_KEPT[path])[1]()
    glass = wavebit.rcwa(backend=backend, top=2.25, bot=Material(-2.25 + 0j, mu=-1.0 + 0j),
                         theta=0.3, wavelength=WL, period=[P], truncation_order=5, tetm='te',
                         thickness=[70.0], structure=_uniform('te', 2.25))
    with pytest.raises(ValueError, match="each other's exact negation"):
        glass.solve(auto_detune=False)
    solver, _ = _negation_scene('te', 'SMM', 'own', backend=backend)
    with pytest.raises(ValueError, match="each other's exact negation"):
        solver.solve_conv(*solver.eps_conv_matrices(), mu=solver.mu_conv_matrices(),
                          mu_top=NEGATION[1], eps_bot=NEGATED[0], mu_bot=NEGATED[1])
    if backend == 1:
        import jax
        import jax.numpy as jnp

        def total_R(re_eps):
            solver = _solver('te', 'SMM', _uniform('te', NEGATED[0]), _uniform('te', NEGATED[1]),
                             theta=0.3, top=Material(re_eps + 0j, mu=NEGATION[1]),
                             bot=Material(*NEGATED), thickness=[80.0], backend=1)
            return solver.solve(auto_detune=False).incident.R.sum().real

        with pytest.raises(ValueError, match="each other's exact negation"):
            jax.grad(total_R)(jnp.asarray(NEGATION[0].real))


@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', '4ff'])
def test_a_lossy_negation_pair_is_solved_and_has_no_lossless_limit(path):
    """Any loss, however small, is solved, and shows why the exact pair is refused.

    Lit by a propagating wave the pair is matched, ``r = 0`` and ``t = 1`` in the limit, and the
    solve gets there: ``R`` falls with the loss and ETM agrees with SMM. Lit by an evanescent
    one (a complex angle), the incident order sits on the pole: ``|r|`` grows like ``1/delta``,
    ``delta*|r|`` constant to three digits from 1e-6 to 1e-12 -- the ``1/gamma`` of Gralak and
    Maystre, with no limit as the loss goes. A real mismatch of 1e-9 is not the pair either.
    """
    scaled = []
    for delta in (1e-6, 1e-9, 1e-12):
        etm = _negation_scene(path, 'ETM', 'own', delta=delta)[1]()
        smm = _negation_scene(path, 'SMM', 'own', delta=delta)[1]()
        assert np.abs(_amplitudes(etm) - _amplitudes(smm)).max() < 1e-12
        R = smm.incident.R if smm.incident is not None else smm.s_inc.R
        assert float(np.sum(_host(R))) < 10 * delta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lit = _negation_scene(path, 'SMM', 'own', delta=delta,
                                  theta=np.pi / 2 - 0.5j)[1]()
        scaled.append(delta * np.abs(_amplitudes(lit)).max())
    assert scaled[0] > 1e-3
    np.testing.assert_allclose(scaled[1:], scaled[0], rtol=1e-3)
    mismatched = (NEGATED[0] * (1 + 1e-9), NEGATED[1])
    result = _negation_scene(path, 'SMM', 'own', bot=mismatched)[1]()
    assert np.all(np.isfinite(_amplitudes(result)))


@pytest.mark.parametrize('path', ['te', 'tm', 'conical', '2d', '4ff'])
def test_a_third_medium_between_the_pair_is_another_stack(path):
    """The pair is refused only when nothing but its own two media lies between the
    half-spaces. A vacuum layer, a layer with the top medium's ``eps`` but not its ``mu``, one
    with its ``eps`` in the plane but not along z or with its diagonal and an in-plane
    off-diagonal (where a path takes a tensor), or a patterned layer makes a stack with an answer, and ETM and SMM agree on it (measured 2.4e-13 at worst,
    under the grating)."""
    shape = _uniform(path, 0)[0].shape
    patterned_eps, patterned_mu = np.full(shape, NEGATION[0]), np.full(shape, NEGATION[1])
    patterned_eps[..., :2, :3], patterned_mu[..., :2, :3] = 1.0, 1.0
    uniaxial = np.zeros(shape + (3, 3), dtype=complex)
    uniaxial[..., 0, 0] = uniaxial[..., 1, 1] = NEGATION[0]
    uniaxial[..., 2, 2] = 5.0
    layers = [(np.ones(shape), np.ones(shape)),
              (np.full(shape, NEGATION[0]), np.ones(shape)),
              (patterned_eps, patterned_mu)]
    if path not in ('te', 'tm'):                     # `tetm` takes no tensor
        layers.append((uniaxial, np.full(shape, NEGATION[1])))
    if path in ('2d', '4ff'):                        # the 1D operators take no eps_xy
        sheared = uniaxial.copy()
        sheared[..., 2, 2] = NEGATION[0]
        sheared[..., 0, 1] = sheared[..., 1, 0] = 0.5
        layers.append((sheared, np.full(shape, NEGATION[1])))
    for eps, mu in layers:
        got = {}
        for algo in ('ETM', 'SMM'):
            solver = _solver('2d' if path == '4ff' else path, algo, eps[None], mu[None],
                             theta=0.3, top=Material(*NEGATION), bot=Material(*NEGATED),
                             thickness=[60.0])
            if path == '4ff':
                solver.formulation = '4ff'
            got[algo] = _amplitudes(solver.solve(auto_detune=False))
        assert np.all(np.isfinite(got['SMM']))
        assert np.abs(got['ETM'] - got['SMM']).max() < 1e-12


def test_the_lens_message_never_calls_two_half_spaces_finite():
    """Gain that exactly cancels the loss makes an exact negation that is not lossless: a lossy
    double negative ``(-2.25+0.1j, -1)`` under a gain medium ``(2.25-0.1j, 1)``. It shares the
    lossless pair's pole on every evanescent order, and with no layer between them it has no
    scattering matrix either, so ETM's perfect-lens message -- "SMM solves it" -- must not be
    raised for it. It is not refused (only the lossless pair is), and the solve fails as the
    singular system makes it fail."""
    gain, lossy = (2.25 - 0.1j, 1.0 + 0j), (-2.25 + 0.1j, -1.0 + 0j)
    solver = _solver('te', 'ETM', _uniform('te', gain[0]), _uniform('te', gain[1]), theta=0.3,
                     top=Material(*gain), bot=Material(*lossy), thickness=[80.0],
                     period=300.0, wavelength=500.0)
    with pytest.raises(Exception) as caught:
        _without_layers(solver)
    assert 'solves it' not in str(caught.value)
