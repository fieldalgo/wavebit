"""Reflection off a stack in *absorbing* media, against the analytic multilayer answer.

A uniform stack has a closed-form reflection coefficient, so this file needs no golden: the
oracle is the Fresnel/Airy recursion below, and the solver is driven with uniform layers at
``fto=0`` where only the zeroth order exists. That makes any disagreement a statement about
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
  of a single gold film. Hence ``test_absorbing_layer_via_the_analytic_path`` below: the raster
  builder was already covered here, the analytic one was not covered anywhere.

Real indices were never affected by any of them, which is why all three survived every
lossless test.

For an absorbing superstrate the *amplitude* is the meaningful check: incident power in an
absorbing medium is convention-dependent, so ``de_ri`` there is not compared to anything.
"""
import importlib

import numpy as np
import pytest

import fieldwave

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


def analytic_r(ns, thicknesses, wavelength, theta, pol):
    """Reflection coefficient of a uniform stack: ``ns = [n_top, *layers, n_bot]``.

    Airy recursion from the bottom interface up — the standard thin-film result, valid for
    complex (absorbing) indices as written.
    """
    k_par = ns[0] * np.sin(theta)
    k = [_kz(n, k_par) for n in ns]

    def r_interface(i, j):
        if pol == 0:                                    # TE
            return (k[i] - k[j]) / (k[i] + k[j])
        return ((complex(ns[j]) ** 2 * k[i] - complex(ns[i]) ** 2 * k[j])
                / (complex(ns[j]) ** 2 * k[i] + complex(ns[i]) ** 2 * k[j]))

    r = r_interface(len(ns) - 2, len(ns) - 1)
    for i in range(len(ns) - 2, 0, -1):
        beta = 2 * np.pi / wavelength * k[i] * thicknesses[i - 1]
        r_up = r_interface(i - 1, i)
        r = (r_up + r * np.exp(2j * beta)) / (1 + r_up * r * np.exp(2j * beta))
    return r


def solver_RT(ns, thicknesses, wavelength, theta, pol, algo='ETM'):
    """Total ``(R, T)`` of the same stack through the solver (uniform layers, fto=0)."""
    ucell = np.array([[[n, n]] for n in ns[1:-1]])
    solver = fieldwave.rcwa(backend=0, connecting_algo=algo, pol=pol, n_top=ns[0],
                           n_bot=ns[-1], theta=theta, fto=0, wavelength=wavelength,
                           period=[wavelength * 2], ucell=ucell, thickness=thicknesses)
    res = solver.conv_solve()
    return float(np.sum(res.de_ri).real), float(np.sum(res.de_ti).real)


def solver_R(ns, thicknesses, wavelength, theta, pol, algo='ETM'):
    """Total reflectance of the same stack through the solver (uniform layers, fto=0)."""
    return solver_RT(ns, thicknesses, wavelength, theta, pol, algo)[0]


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

INCIDENCE = [(pol, theta) for pol in (0, 1) for theta in (0.0, 0.4)]
IDS = [f'{"te" if p == 0 else "tm"}_theta{t}' for p, t in INCIDENCE]


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', KNOWN_GOOD, ids=[c[0] for c in KNOWN_GOOD])
def test_reference_agrees_where_the_solver_is_known_good(case, pol, theta):
    """The oracle itself: lossless stacks and lossy *layers* already match to ~1e-15."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2
    assert solver_R(ns, ds, WL, theta, pol) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_reflection(case, pol, theta, algo):
    """Reflection off a stack on an absorbing substrate must match the analytic answer."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2
    assert solver_R(ns, ds, WL, theta, pol, algo) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_transmission_agrees_between_algorithms(case, pol, theta):
    """``de_ti`` into an absorbing substrate, ETM against SMM.

    The reflection tests above compared only ``de_ri``, and that gap hid a real break for four
    years' worth of the SMM path: its TM branch normalized by ``Re(conj(kz_bot)/n_bot**2)``
    where every other efficiency in the codebase uses ``Re(kz_bot/n_bot**2)``. The two are equal
    for a real ``n_bot``, so nothing lossless could see it; against gold it returned *negative*
    transmitted power (T = -0.263 where ETM gave +0.088).

    The right factor follows from the flux itself: for TM the downward Poynting component is
    ``Re(E_x H_y*)`` with ``E_x`` proportional to ``(kz/eps) H_y``, giving ``Re(kz/eps)``. The
    conjugate was a leftover of the pre-2026-08 conjugated formulation (see ``conventions``).
    """
    _, ns, ds = case
    tmm = solver_RT(ns, ds, WL, theta, pol, 'ETM')
    smm = solver_RT(ns, ds, WL, theta, pol, 'SMM')
    assert smm[0] == pytest.approx(tmm[0], abs=1e-12)
    assert smm[1] == pytest.approx(tmm[1], abs=1e-12)
    assert tmm[1] > 0, 'transmitted power into an absorbing substrate cannot be negative'


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
def test_lossless_layer_on_absorbing_substrate_conserves_energy(algo, pol, theta):
    """``de_ti`` is the power *entering* the substrate, so nothing is lost above it.

    With every layer lossless, R + T must be exactly 1 however absorbing the substrate is —
    the sharpest available statement about ``de_ti``'s normalization, and one no reflection-only
    test can make.
    """
    R, T = solver_RT([1.0, 1.5, AU], [20.0], WL, theta, pol, algo)
    assert R + T == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
def test_absorbing_layer_identical_to_its_substrate(pol, theta):
    """An absorbing layer on a substrate of the same material must vanish into it.

    The layer and the substrate have to arrive at the same modes on their own. They did not
    while the substrate skipped the conjugated convention, and a short-circuit that skipped
    the interface entirely was hiding it — the answer stayed right only because skipping the
    layer leaves a bare interface, where the wrong phase cannot show. Nothing is skipped now:
    this is the general transfer recursion solving the matched interface for itself, which is
    also what makes it a live check on the boundary.
    """
    ns, ds = [1.0, AU, AU], [20.0]
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2   # == the bare air/gold interface

    ucell = np.array([[[AU, AU]]])
    common = dict(backend=0, pol=pol, n_top=1.0, n_bot=AU, theta=theta, fto=0,
                  wavelength=WL, period=[WL * 2], ucell=ucell, thickness=ds)
    tmm = fieldwave.rcwa(connecting_algo='ETM', **common).conv_solve()
    smm = fieldwave.rcwa(connecting_algo='SMM', **common).conv_solve()

    assert float(np.sum(tmm.de_ri)) == pytest.approx(want, abs=1e-12)
    assert float(np.sum(smm.de_ri)) == pytest.approx(want, abs=1e-12)


def solver_R_analytic(ns, thicknesses, wavelength, theta, pol):
    """Total reflectance of the same stack built through the ANALYTIC path.

    ``shapes=[]`` makes each layer uniform ``n_base``, so the structure is identical to the one
    ``solver_RT`` rasters — only the ``n -> eps`` builder differs. That routes through the 2D
    solve (the smooth path always does), so the polarization comes in as ``psi`` rather than
    ``pol``, and the period must be 2D.
    """
    ucell = [dict(n_base=n, shapes=[]) for n in ns[1:-1]]
    solver = fieldwave.rcwa(backend=0, n_top=ns[0], n_bot=ns[-1], theta=theta, phi=0.0,
                            psi=(np.pi / 2 if pol == 0 else 0.0), fto=0,
                            wavelength=wavelength, period=[wavelength * 2, wavelength * 2],
                            ucell=ucell, thickness=thicknesses)
    return float(np.sum(np.asarray(solver.conv_solve().de_ri)).real)


ANALYTIC_CASES = KNOWN_GOOD + LOSSY_SUBSTRATE


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', ANALYTIC_CASES, ids=[c[0] for c in ANALYTIC_CASES])
def test_absorbing_layer_via_the_analytic_path(case, pol, theta):
    """The analytic builder must land on the same physics as the raster one, for complex ``n``.

    The smooth path had no complex-index coverage at all: ``tests/test_analytic``
    was entirely real, where ``conj(n)**2 == n**2``. Running the *uniform* stack through it puts
    it against the absolute oracle rather than against another solver — gold as a layer landed
    ~9% off here and tripped the ``R + T > 1`` guard, because the conjugation turned it into a
    gain medium. Nothing about the shape machinery is exercised; the ``n -> eps`` convention is
    the entire point.

    Uniform layers are also the only place the two builders can be held to 1e-12 against each
    other. Put a *patterned* absorbing structure through both and they disagree by ~3e-3 at
    ``fto=8`` — not a convention difference but the two NVF factorizations converging to each
    other slowly when ``|eps|`` is large.
    """
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2
    got = solver_R_analytic(ns, ds, WL, theta, pol)
    assert got == pytest.approx(want, abs=1e-12)
    assert got == pytest.approx(solver_R(ns, ds, WL, theta, pol), abs=1e-12)   # == raster


def solver_R_painted(n, thickness, wavelength, theta, pol):
    """Total reflectance of a *uniform* layer of index ``n`` built through the PAINTED path.

    Two overlapping rectangles of the same index, together covering the cell. The overlap is
    what routes the layer to ``_core.boundary.to_conv_mat_boundary_layer`` (a non-overlapping
    ``paint`` stays on the summed closed form), and same-index-everywhere is what leaves a
    uniform slab at the end of it, so the Airy oracle applies exactly and none of the shape
    machinery is under test. The ``n -> eps`` convention is the entire point, again.
    """
    P = 400.0
    shapes = [dict(type='rectangle', cx=0.30 * P, cy=0.5 * P, lx=0.75 * P, ly=1.4 * P,
                   n_index=n),
              dict(type='rectangle', cx=0.72 * P, cy=0.5 * P, lx=0.75 * P, ly=1.4 * P,
                   n_index=n)]
    solver = fieldwave.rcwa(backend=0, n_top=1.0, n_bot=1.5, theta=theta, phi=0.0,
                            psi=(np.pi / 2 if pol == 0 else 0.0), fto=(3, 3),
                            wavelength=wavelength, period=[P, P],
                            ucell=[dict(n_base=n, combine='paint', shapes=shapes)],
                            thickness=[thickness])
    res = solver.conv_solve()
    return float(np.sum(np.asarray(res.de_ri)).real), float(np.sum(np.asarray(res.de_ti)).real)


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('n', [AU, SI], ids=['gold', 'weakly_absorbing'])
def test_absorbing_layer_via_the_painted_path(n, pol, theta):
    """The boundary (painted-polygon) builder, for complex ``n``. It was the last one left.

    ``_core/boundary.py`` is a later copy of ``_core/analytic.py``'s coefficient assembly, and
    it kept the ``conj(n)**2`` that the smooth path was moved off — so a painted absorbing
    layer came out as *gain*: reflectance off by 0.15 on gold and 0.023 on the weak absorber,
    with ``R + T`` up to 1.11.

    Nothing objected, for the same reason the analytic path went uncovered before it:
    ``test_boundary.py`` builds every shape with a **real** ``n_index``, where ``conj(n)**2``
    and ``n**2`` are the same number. Its two builders are cross-checked against each other to
    5e-9 and both were on the same real indices, so agreeing proved nothing about this.

    ``R + T <= 1`` is asserted separately from the oracle: it is the statement that survives
    even if the oracle is ever doubted, and it is what the conjugation broke most visibly.
    """
    R, T = solver_R_painted(n, 60.0, WL, theta, pol)
    want = abs(analytic_r([1.0, n, 1.5], [60.0], WL, theta, pol)) ** 2
    assert R == pytest.approx(want, abs=1e-12)
    assert R + T <= 1.0, 'an absorbing layer cannot return more power than it was given'


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
def test_complex_n_bot_keeps_full_precision_on_every_backend(backend):
    """A complex ``n_bot`` must reach the boundary at the solve's working precision.

    PyTorch reads a bare Python ``complex`` as ``complex64``, and the ~1e-8 error that
    introduces survives the later promotion to ``complex128`` — invisible against a real
    ``n_bot``, which is exactly the case the other cross-backend tests use.
    """
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    if backend == 1:
        # via importlib: a plain `import fieldwave.on_jax` here would rebind `fieldwave`
        # as a local for the whole function, breaking the torch branch that skips it.
        importlib.import_module('fieldwave.on_jax')       # enables x64

    ns, ds = [1.0, 1.5, AU], [20.0]
    want = solver_R(ns, ds, WL, 0.3, 0)                     # numpy reference

    ucell = np.array([[[n, n] for n in ns[1:-1]]][0])[None]
    solver = fieldwave.rcwa(backend=backend, pol=0, n_top=ns[0], n_bot=ns[-1], theta=0.3,
                           fto=2, wavelength=WL, period=[WL * 2], ucell=ucell, thickness=ds)
    got = float(np.sum(np.asarray(solver.conv_solve().de_ri)))
    assert got == pytest.approx(want, abs=1e-12)


# ------------------------------------------------------- amplitude oracle (absorbing media)
# Reflectance needs an incident power, which is convention-dependent in an absorbing medium.
# The complex amplitude is not, and FieldWave reports it in the same e^{-iwt} convention the
# inputs use, so `R == analytic_r` is an exact check that works with a lossy superstrate too.

WET = 1.5 + 0.02j         # weakly absorbing ambient

AMPLITUDE_CASES = [
    ('lossless', [1.0, 1.5, 2.2, 1.8], [20.0, 35.0]),
    ('lossy_substrate', [1.0, 1.5, AU], [20.0]),
    ('weakly_absorbing_superstrate', [WET, 1.5, 1.8], [20.0]),
    ('strongly_absorbing_superstrate', [AU, 1.5, 1.8], [20.0]),
    ('absorbing_on_both_sides', [WET, 1.5, AU], [20.0]),
]


def solver_r(ns, thicknesses, wavelength, theta, pol, path, algo='ETM'):
    """The solver's reflection amplitude, normalized to compare against ``analytic_r``.

    ``path`` picks which core solve runs: ``'1d'`` (pure TE/TM), ``'conical'`` or ``'2d'``.
    The coupled paths report the p channel as its *magnetic* amplitude against an incident
    ``i*n_top`` (in the internal convention), hence the division; the s channel is the electric
    amplitude against an incident 1, so it needs none.
    """
    layers = ns[1:-1]
    common = dict(backend=0, connecting_algo=algo, n_top=ns[0], n_bot=ns[-1], theta=theta,
                  wavelength=wavelength, thickness=thicknesses)
    if path == '1d':
        solver = fieldwave.rcwa(pol=pol, fto=0, period=[wavelength * 2],
                                ucell=np.array([[[n, n]] for n in layers]), **common)
    else:
        cell = ([[[n, n]] for n in layers] if path == 'conical'
                else [[[n, n], [n, n]] for n in layers])
        solver = fieldwave.rcwa(phi=0.0, psi=(np.pi / 2 if pol == 0 else 0.0), fto=[0, 0],
                                period=[wavelength * 2, wavelength * 2],
                                ucell=np.array(cell), **common)
    res = solver.conv_solve().res
    amp = np.asarray(res.R_s if pol == 0 else res.R_p).ravel()
    value = complex(amp[len(amp) // 2])
    if path == '1d' or pol == 0:
        return value
    return value / (-1j * complex(ns[0]))


@pytest.mark.parametrize('algo', ['ETM', 'SMM'])
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', AMPLITUDE_CASES, ids=[c[0] for c in AMPLITUDE_CASES])
def test_reflection_amplitude(case, pol, theta, path, algo):
    """Complex reflection amplitude, for absorbing media on either side or both."""
    _, ns, ds = case
    want = analytic_r(ns, ds, WL, theta, pol)
    got = solver_r(ns, ds, WL, theta, pol, path, algo)
    assert abs(got - want) < 1e-12, (got, want)
