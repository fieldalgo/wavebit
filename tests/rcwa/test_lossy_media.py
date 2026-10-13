"""Reflection off a stack in *absorbing* media, against the analytic multilayer answer.

A uniform stack has a closed-form reflection coefficient, so this file needs no golden: the
oracle is the Fresnel/Airy recursion below, and the solver is driven with uniform layers at
``truncation_order=0`` where only the zeroth order exists. That makes any disagreement a statement about
physics rather than about a frozen number.

The reference is itself checked (``test_reference_agrees_where_the_solver_is_known_good``)
on lossless stacks and on lossy *layers*, where solver and analytic agree to ~1e-16 for TE and
TM at normal and oblique incidence.

What it pins: every quantity must enter in the same convention the layers do — boundaries and
Fourier builders alike. Three places did not. The first two are stated in the pre-2026-08
conjugated formulation, which is what the code ran on when they were found (see
``_core.conventions`` for what replaced it).

* The substrate permittivity was raw ``n_bot**2``, so the stack reflected as though the
  substrate were ``conj(n_bot)`` — absorbing read as gain. Only the reflection *phase* was
  affected, so it hid wherever interference cannot show it (a bare interface, a vanishing
  layer) and grew with the stack above it: ~2.6% on one layer, ~12% on two.
* ``kz_top`` paired a raw ``n_top**2`` with an already-conjugated ``kx``, and the incident-field
  coefficients used a raw ``n_top``. With an absorbing superstrate the reflected amplitude was
  wrong outright — by ~1% at normal incidence and by more than the amplitude itself at oblique.
* The analytic (shape-based) Fourier builder kept conjugating after the seam was removed, so an
  absorbing *layer* built that way came out as gain — R + T above 1, and ~9% on the reflectance
  of a single gold film. Hence ``test_absorbing_layer_via_the_parametric_path`` below: the raster
  builder was already covered here, the analytic one was not covered anywhere.

Real indices were never affected by any of them, which is why all three survived every
lossless test.

Absorbing-incident power is normalized at the illuminated interface. Real-transverse-k
reflection is checked here, with transmission and mixed-polarization power in
``test_half_space_power.py``. Complex-transverse-k cases below check amplitudes only.
"""

from wavebit.compat import legacy as _legacy_wavebit

import numpy as np
import pytest

from wavebit import eps_from_index

# Every stack in this file is written as a list of refractive INDICES, and stays that way:
# the Airy/transfer oracles below are written in n, and restating them in permittivity would
# make them inherit the quantity the solver takes -- which is the one thing an oracle must
# not do. `eps_from_index` is applied at each call into wavebit instead, which is exactly the
# migration a user with an nk table performs.

WL = 500.0
AU = 0.18 + 3.0j          # gold at 500nm — strongly absorbing
SI = 4.0 + 0.05j          # weakly absorbing


# ------------------------------------------------------------------ analytic reference

def _kz(n, k_par):
    """Out-of-plane wavevector of the downward wave (``e^{-iwt}``, ``Im(n) > 0`` = loss).

    The branch is ``Re(kz) > 0`` — energy flowing down — falling back to ``Im(kz) > 0`` for a
    purely evanescent order. Selecting on ``Im`` alone is wrong as soon as ``k_par`` is complex
    (a lossy superstrate at oblique incidence): a *lossless* layer can legitimately have
    ``Im(kz) < 0`` there, and forcing it positive returns ``|r| > 1``.
    """
    v = np.sqrt(complex(n) ** 2 - k_par ** 2)
    if abs(v.real) > 1e-12:
        return v if v.real > 0 else -v
    return v if v.imag > 0 else -v


def _transverse_k(n, theta):
    """Independent numerical inversion for real phase-angle illumination.

    Complex angles retain their formal Snell parameterization. For absorbing
    real-angle incidence, bracket the real transverse component using the
    principal dispersion root and its phase direction, not the solver formula.
    """
    if np.imag(theta) != 0 or np.imag(n) == 0 or theta == 0:
        return n * np.sin(theta)
    target = abs(float(np.real(theta)))
    eps = complex(n)**2
    def phase_angle(beta):
        return np.arctan2(beta, np.sqrt(eps - beta**2).real)
    low, high = 0., 1.
    while phase_angle(high) < target:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if phase_angle(mid) < target:
            low = mid
        else:
            high = mid
    return np.copysign((low + high) / 2, np.real(theta))


def analytic_r(ns, thicknesses, wavelength, theta, tetm):
    """Reflection coefficient of a uniform stack: ``ns = [n_top, *layers, n_bot]``.

    Airy recursion from the bottom interface up — the standard thin-film result, valid for
    complex (absorbing) indices as written.
    """
    k_par = _transverse_k(ns[0], theta)
    k = [_kz(n, k_par) for n in ns]

    def r_interface(i, j):
        if tetm == 'te':                                    # TE
            return (k[i] - k[j]) / (k[i] + k[j])
        return ((complex(ns[j]) ** 2 * k[i] - complex(ns[i]) ** 2 * k[j])
                / (complex(ns[j]) ** 2 * k[i] + complex(ns[i]) ** 2 * k[j]))

    r = r_interface(len(ns) - 2, len(ns) - 1)
    for i in range(len(ns) - 2, 0, -1):
        beta = 2 * np.pi / wavelength * k[i] * thicknesses[i - 1]
        r_up = r_interface(i - 1, i)
        r = (r_up + r * np.exp(2j * beta)) / (1 + r_up * r * np.exp(2j * beta))
    return r


def solver_RT(ns, thicknesses, wavelength, theta, tetm, algo='ETM'):
    """Total ``(R, T)`` of the same stack through the solver (uniform layers, truncation_order=0)."""
    structure = eps_from_index(np.array([[[n, n]] for n in ns[1:-1][::-1]]))
    solver = _legacy_wavebit.rcwa(backend='numpy', layer_algorithm=algo, tetm=tetm,
                           top=eps_from_index(ns[0]), bot=eps_from_index(ns[-1]),
                           theta=theta, truncation_order=0, wavelength=wavelength,
                           period=[wavelength * 2], structure=structure,
                           thickness=thicknesses[::-1])
    res = solver.solve()
    return float(np.sum(res.R).real), float(np.sum(res.T).real)


def solver_R(ns, thicknesses, wavelength, theta, tetm, algo='ETM'):
    """Total reflectance of the same stack through the solver (uniform layers, truncation_order=0)."""
    return solver_RT(ns, thicknesses, wavelength, theta, tetm, algo)[0]


# name, indices, thicknesses
KNOWN_GOOD = [
    ('lossless_2layer', [1.0, 1.5, 2.2, 1.8], [20.0, 35.0]),
    ('lossy_layer_lossless_substrate', [1.0, AU, 1.5], [20.0]),
    ('lossy_layer_thick', [1.0, SI, 1.5], [120.0]),
    # a vanishing layer leaves a bare interface: magnitude only, so the substrate phase
    # cannot show up here even though the substrate is absorbing
    ('bare_lossy_interface', [1.0, 1.0, AU], [1e-6]),
]

LOSSY_SUBSTRATE = [
    ('lossless_layer_on_gold', [1.0, 1.5, AU], [20.0]),
    ('absorbing_layer_on_gold', [1.0, SI, AU], [30.0]),
    ('two_layers_on_gold', [1.0, 1.5, 2.2, AU], [20.0, 35.0]),
]

INCIDENCE = [(tetm, theta) for tetm in ('te', 'tm') for theta in (0.0, 0.4)]
IDS = [f'{p}_theta{t}' for p, t in INCIDENCE]


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', KNOWN_GOOD, ids=[c[0] for c in KNOWN_GOOD])
def test_reference_agrees_where_the_solver_is_known_good(case, tetm, theta):
    """The oracle itself: lossless stacks and lossy *layers* already match to ~1e-15."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, tetm)) ** 2
    assert solver_R(ns, ds, WL, theta, tetm) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_reflection(case, tetm, theta, algo):
    """Reflection off a stack on an absorbing substrate must match the analytic answer."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, tetm)) ** 2
    assert solver_R(ns, ds, WL, theta, tetm, algo) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_transmission_agrees_between_algorithms(case, tetm, theta):
    """``T`` into an absorbing substrate, ETM against SMM.

    The reflection tests above compared only ``R``, and that gap hid a real break for four
    years' worth of the SMM path: its TM branch normalized by ``Re(conj(kz_bot)/n_bot**2)``
    where every other efficiency in the codebase uses ``Re(kz_bot/n_bot**2)``. The two are equal
    for a real ``n_bot``, so nothing lossless could see it; against gold it returned *negative*
    transmitted power (T = -0.263 where ETM gave +0.088).

    The right factor follows from the flux itself: for TM the downward Poynting component is
    ``Re(E_x H_y*)`` with ``E_x`` proportional to ``(kz/eps) H_y``, giving ``Re(kz/eps)``. The
    conjugate was a leftover of the pre-2026-08 conjugated formulation (see ``conventions``).
    """
    _, ns, ds = case
    tmm = solver_RT(ns, ds, WL, theta, tetm, 'ETM')
    smm = solver_RT(ns, ds, WL, theta, tetm, 'SMM')
    assert smm[0] == pytest.approx(tmm[0], abs=1e-12)
    assert smm[1] == pytest.approx(tmm[1], abs=1e-12)
    assert tmm[1] > 0, 'transmitted power into an absorbing substrate cannot be negative'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
def test_lossless_layer_on_absorbing_substrate_conserves_energy(algo, tetm, theta):
    """``T`` is the power *entering* the substrate, so nothing is lost above it.

    With every layer lossless, R + T must be exactly 1 however absorbing the substrate is —
    the sharpest available statement about ``T``'s normalization, and one no reflection-only
    test can make.
    """
    R, T = solver_RT([1.0, 1.5, AU], [20.0], WL, theta, tetm, algo)
    assert R + T == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
def test_absorbing_layer_identical_to_its_substrate(tetm, theta):
    """An absorbing layer on a substrate of the same material must vanish into it.

    The layer and the substrate have to arrive at the same modes on their own. They did not
    while the substrate skipped the conjugated convention, and a short-circuit that skipped
    the interface entirely was hiding it — the answer stayed right only because skipping the
    layer leaves a bare interface, where the wrong phase cannot show. Nothing is skipped now:
    this is the general transfer recursion solving the matched interface for itself, which is
    also what makes it a live check on the boundary.
    """
    ns, ds = [1.0, AU, AU], [20.0]
    want = abs(analytic_r(ns, ds, WL, theta, tetm)) ** 2   # == the bare air/gold interface

    structure = eps_from_index(np.array([[[AU, AU]]]))
    common = dict(backend='numpy', tetm=tetm, top=1.0, bot=eps_from_index(AU), theta=theta, truncation_order=0,
                  wavelength=WL, period=[WL * 2], structure=structure, thickness=ds)
    tmm = _legacy_wavebit.rcwa(layer_algorithm='ETM', **common).solve()
    smm = _legacy_wavebit.rcwa(layer_algorithm='SMM', **common).solve()

    assert float(np.sum(tmm.R)) == pytest.approx(want, abs=1e-12)
    assert float(np.sum(smm.R)) == pytest.approx(want, abs=1e-12)


def solver_R_analytic(ns, thicknesses, wavelength, theta, tetm):
    """Total reflectance of the same stack built through the ANALYTIC path.

    ``shapes=[]`` makes each layer uniform ``background``, so the structure is identical to the
    one ``solver_RT`` rasters — only the coefficient builder differs. That routes through the 2D
    solve (the parametric path always does), so the polarization comes in as ``psi`` rather than
    ``tetm``, and the period must be 2D.
    """
    structure = [dict(background=eps_from_index(n), shapes=[]) for n in ns[1:-1][::-1]]
    solver = _legacy_wavebit.rcwa(backend='numpy', top=eps_from_index(ns[0]), bot=eps_from_index(ns[-1]),
                            theta=theta, phi=0.0,
                            psi=(np.pi / 2 if tetm == 'te' else 0.0), truncation_order=0,
                            wavelength=wavelength, period=[wavelength * 2, wavelength * 2],
                            structure=structure, thickness=thicknesses[::-1])
    return float(np.sum(np.asarray(solver.solve().R)).real)


ANALYTIC_CASES = KNOWN_GOOD + LOSSY_SUBSTRATE


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', ANALYTIC_CASES, ids=[c[0] for c in ANALYTIC_CASES])
def test_absorbing_layer_via_the_parametric_path(case, tetm, theta):
    """The parametric builder must land on the same physics as the raster one, for complex ``n``.

    The parametric path had no complex-index coverage at all: ``tests/test_parametric``
    was entirely real, where ``conj(n)**2 == n**2``. Running the *uniform* stack through it puts
    it against the absolute oracle rather than against another solver — gold as a layer landed
    ~9% off here and tripped the ``R + T > 1`` guard, because the conjugation turned it into a
    gain medium. Nothing about the shape machinery is exercised; the material convention is
    the entire point.

    Uniform layers are also the only place the two builders can be held to 1e-12 against each
    other. Put a *patterned* absorbing structure through both and they disagree by ~3e-3 at
    ``truncation_order=8`` — not a convention difference but the two NVF factorizations converging to each
    other slowly when ``|eps|`` is large.
    """
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, tetm)) ** 2
    got = solver_R_analytic(ns, ds, WL, theta, tetm)
    assert got == pytest.approx(want, abs=1e-12)
    assert got == pytest.approx(solver_R(ns, ds, WL, theta, tetm), abs=1e-12)   # == raster


def solver_R_painted(n, thickness, wavelength, theta, tetm):
    """Total reflectance of a *uniform* layer of index ``n`` built through the PAINTED path.

    Two overlapping rectangles of the same index, together covering the cell. The overlap is
    what routes the layer to ``_core.boundary.to_conv_mat_boundary_layer`` (a layer with nothing
    overlapping stays on the summed closed form), and same-index-everywhere is what leaves a
    uniform slab at the end of it, so the Airy oracle applies exactly and none of the shape
    machinery is under test. The material convention is the entire point, again.

    Every edge of the pair lands exactly on a cell edge or inside it, and that is not
    cosmetic. The boundary builder does not read a shape's *periodic images* -- it clips loops
    in unwrapped coordinates -- so a rod past the period, or a pair meeting only through an
    edge, is refused (``check_layer_shapes``). A gap of exactly zero is touching, not
    overlapping, so ``ly = 1.0 * P`` spans the cell and is legal.

    Both were wrong here until 2026-08-27: the rods were ``ly = 1.4 * P`` and their x extents
    wrapped as well. It gave the right answer anyway, for a reason that is not a reason -- the
    two rods tile the cell, so whatever the missing images would have covered, the *other* rod
    covered. On the same pair shrunk until they no longer tile, ``<1/eps>`` comes back 0.618
    against the 0.695 the geometry has, and a wrapped overlap's ``sum(T)`` does not converge at
    all (0.309 / 0.632 / 0.913 at truncation order 5 / 9 / 13, against a wrapped raster's
    0.997). The pair here overlaps over 0.4-0.5 of the cell, inside it, which is the case the
    resolver is for.
    """
    P = 400.0
    eps = eps_from_index(n)
    shapes = [dict(type='rectangle', cx=-0.25 * P, cy=0.0, lx=0.50 * P, ly=1.0 * P,
                   material=eps),
              dict(type='rectangle', cx=0.20 * P, cy=0.0, lx=0.60 * P, ly=1.0 * P,
                   material=eps)]
    solver = _legacy_wavebit.rcwa(backend='numpy', top=1.0, bot=2.25, theta=theta, phi=0.0,
                            psi=(np.pi / 2 if tetm == 'te' else 0.0), truncation_order=(3, 3),
                            wavelength=wavelength, period=[P, P],
                            structure=[dict(background=eps, shapes=shapes)],
                            thickness=[thickness])
    res = solver.solve()
    return float(np.sum(np.asarray(res.R)).real), float(np.sum(np.asarray(res.T)).real)


@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('n', [AU, SI], ids=['gold', 'weakly_absorbing'])
def test_absorbing_layer_via_the_painted_path(n, tetm, theta):
    """The boundary (painted-polygon) builder, for complex ``n``. It was the last one left.

    ``src/wavebit/rcwa/_engine/boundary.py`` is a later copy of ``src/wavebit/rcwa/_engine/parametric.py``'s coefficient assembly, and
    it kept the ``conj(n)**2`` that the parametric path was moved off — so a painted absorbing
    layer came out as *gain*: reflectance off by 0.15 on gold and 0.023 on the weak absorber,
    with ``R + T`` up to 1.11.

    Nothing objected, for the same reason the parametric path went uncovered before it:
    ``test_boundary.py`` builds every shape with a **real** material, where ``conj(eps)``
    and ``eps`` are the same number. Its two builders are cross-checked against each other to
    5e-9 and both were on the same real materials, so agreeing proved nothing about this.

    ``R + T <= 1`` is asserted separately from the oracle: it is the statement that survives
    even if the oracle is ever doubted, and it is what the conjugation broke most visibly.
    """
    R, T = solver_R_painted(n, 60.0, WL, theta, tetm)
    want = abs(analytic_r([1.0, n, 1.5], [60.0], WL, theta, tetm)) ** 2
    assert R == pytest.approx(want, abs=1e-12)
    assert R + T <= 1.0, 'an absorbing layer cannot return more power than it was given'


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_complex_n_bot_keeps_full_precision_on_every_backend(backend):
    """A complex ``n_bot`` must reach the boundary at the solve's working precision.

    PyTorch reads a bare Python ``complex`` as ``complex64``, and the ~1e-8 error that
    introduces survives the later promotion to ``complex128`` — invisible against a real
    ``n_bot``, which is exactly the case the other cross-backend tests use.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit.enable_x64()

    ns, ds = [1.0, 1.5, AU], [20.0]
    want = solver_R(ns, ds, WL, 0.3, 'te')                  # numpy reference

    structure = eps_from_index(np.array([[[n, n] for n in ns[1:-1][::-1]]][0])[None])
    solver = _legacy_wavebit.rcwa(backend=backend, tetm='te', top=eps_from_index(ns[0]),
                           bot=eps_from_index(ns[-1]), theta=0.3,
                           truncation_order=2, wavelength=WL, period=[WL * 2], structure=structure,
                           thickness=ds)
    got = float(np.sum(np.asarray(solver.solve().R)))
    assert got == pytest.approx(want, abs=1e-12)


# ------------------------------------------------------- amplitude oracle (absorbing media)
# Reflectance needs an incident power, which is convention-dependent in an absorbing medium.
# The complex amplitude is not, and Wavebit reports it in the same e^{-iwt} convention the
# inputs use, so `R == analytic_r` is an exact check that works with a lossy superstrate too.

WET = 1.5 + 0.02j         # weakly absorbing ambient

AMPLITUDE_CASES = [
    ('lossless', [1.0, 1.5, 2.2, 1.8], [20.0, 35.0]),
    ('lossy_substrate', [1.0, 1.5, AU], [20.0]),
    ('weakly_absorbing_superstrate', [WET, 1.5, 1.8], [20.0]),
    ('strongly_absorbing_superstrate', [AU, 1.5, 1.8], [20.0]),
    ('absorbing_on_both_sides', [WET, 1.5, AU], [20.0]),
]


def analytic_t(ns, thicknesses, wavelength, theta, tetm):
    """Transmission amplitude (E-field ratio) of a uniform stack, via a 2x2 transfer.

    Same conventions as :func:`analytic_r`; for TM the characteristic transfer works in the
    magnetic variable, so the last line converts its ratio back to an E-field one.
    """
    k_par = _transverse_k(ns[0], theta)
    k = [_kz(n, k_par) for n in ns]

    def eta(i):
        return k[i] if tetm == 'te' else k[i] / complex(ns[i]) ** 2

    M = np.eye(2, dtype=complex)
    for i in range(1, len(ns) - 1):
        b = 2 * np.pi / wavelength * k[i] * thicknesses[i - 1]
        M = M @ np.array([[np.cos(b), -1j * np.sin(b) / eta(i)],
                          [-1j * eta(i) * np.sin(b), np.cos(b)]])
    e0, eN = eta(0), eta(len(ns) - 1)
    t = 2 * e0 / (e0 * M[0, 0] + eN * M[1, 1] + e0 * eN * M[0, 1] + M[1, 0])
    if tetm == 'tm':
        t = t * complex(ns[0]) / complex(ns[-1])
    return t


def solver_amps(ns, thicknesses, wavelength, theta, tetm, path, algo='ETM'):
    """The solver's Fresnel amplitudes ``(r, t)`` for the asked polarization.

    ``path`` picks which core solve runs: ``'1d'`` (pure TE/TM), ``'conical'`` or ``'2d'``.
    No normalization happens here — ``r_s``/``r_p``/``t_s``/``t_p`` *are* the Fresnel
    amplitudes, which is the public contract this file pins. (Until 2026-08-12 the result
    exposed raw boundary variables instead, and this function carried the ``/(-i*n_top)``
    footnote that conversion now lives without.)
    """
    layers = ns[1:-1][::-1]
    common = dict(backend='numpy', layer_algorithm=algo, top=eps_from_index(ns[0]),
                  bot=eps_from_index(ns[-1]), theta=theta,
                  wavelength=wavelength, thickness=thicknesses[::-1])
    if path == '1d':
        solver = _legacy_wavebit.rcwa(tetm=tetm, truncation_order=0, period=[wavelength * 2],
                                structure=eps_from_index(
                                    np.array([[[n, n]] for n in layers])), **common)
    else:
        cell = ([[[n, n]] for n in layers] if path == 'conical'
                else [[[n, n], [n, n]] for n in layers])
        solver = _legacy_wavebit.rcwa(phi=0.0, psi=(np.pi / 2 if tetm == 'te' else 0.0), truncation_order=[0, 0],
                                period=[wavelength * 2, wavelength * 2],
                                structure=eps_from_index(np.array(cell)), **common)
    res = solver.solve().incident
    r = np.asarray(res.r_s if tetm == 'te' else res.r_p).ravel()
    t = np.asarray(res.t_s if tetm == 'te' else res.t_p).ravel()
    return complex(r[len(r) // 2]), complex(t[len(t) // 2])


def solver_r(ns, thicknesses, wavelength, theta, tetm, path, algo='ETM'):
    return solver_amps(ns, thicknesses, wavelength, theta, tetm, path, algo)[0]


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', AMPLITUDE_CASES, ids=[c[0] for c in AMPLITUDE_CASES])
def test_reflection_amplitude(case, tetm, theta, path, algo):
    """``r_s`` / ``r_p`` are the Fresnel amplitudes, for absorbing media on either side."""
    _, ns, ds = case
    want = analytic_r(ns, ds, WL, theta, tetm)
    got = solver_r(ns, ds, WL, theta, tetm, path, algo)
    assert abs(got - want) < 1e-12, (got, want)


@pytest.mark.parametrize('beta', [0., .5])
@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
def test_absorbing_incident_reflectance_matches_airy(beta, algo):
    """Each pure-p wave is normalized by its own incident power at the interface."""
    n_top = 1.5 + 0.3j
    theta = np.arcsin(beta / n_top)
    ns, ds = [n_top, 2.0, 1.0], [30.0]

    def eff(path):
        common = dict(backend='numpy', top=eps_from_index(ns[0]), bot=eps_from_index(ns[-1]),
                      theta=theta, wavelength=WL, thickness=ds, layer_algorithm=algo)
        if path == '1d':
            s = _legacy_wavebit.rcwa(tetm='tm', truncation_order=0, period=[WL * 2],
                               structure=eps_from_index(
                                   np.array([[[n, n]] for n in ns[1:-1][::-1]])), **common)
        else:
            s = _legacy_wavebit.rcwa(phi=0.0, psi=0.0, truncation_order=[0, 0],
                               period=[WL * 2, WL * 2],
                               structure=eps_from_index(
                                   np.array([[[n, n], [n, n]] for n in ns[1:-1][::-1]])), **common)
        res = s.solve()
        return float(np.sum(np.asarray(res.R))), res

    R_1d, res_1d = eff('1d')
    R_2d, res_2d = eff('2d')
    expected = abs(analytic_r(ns, ds, WL, theta, 'tm'))**2

    r_1d = complex(np.asarray(res_1d.incident.r_p).ravel()[0])
    assert R_1d == pytest.approx(expected, abs=1e-12)
    assert R_2d == pytest.approx(expected, abs=1e-12)
    r_2d = complex(np.asarray(res_2d.incident.r_p).ravel()[0])
    assert abs(r_2d - r_1d) < 1e-12                                  # amplitudes agree


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('tetm,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', AMPLITUDE_CASES, ids=[c[0] for c in AMPLITUDE_CASES])
def test_transmission_amplitude(case, tetm, theta, path, algo):
    """``t_s`` / ``t_p`` are the Fresnel amplitudes too — the conversion each pack site owns.

    This is the test that distinguishes the three conversions (none for s, ``n_top/n_bot``
    for the pure-1D TM primary, ``/(-i*n_bot)`` for the coupled p channel): drop or swap any
    one of them and its path fails here while reflection stays green.
    """
    _, ns, ds = case
    want = analytic_t(ns, ds, WL, theta, tetm)
    got = solver_amps(ns, ds, WL, theta, tetm, path, algo)[1]
    assert abs(got - want) < 1e-12, (got, want)
