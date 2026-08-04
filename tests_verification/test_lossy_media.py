"""Reflection off a stack in *absorbing* media, against the analytic multilayer answer.

A uniform stack has a closed-form reflection coefficient, so this file needs no golden: the
oracle is the Fresnel/Airy recursion below, and the solver is driven with uniform layers at
``fto=0`` where only the zeroth order exists. That makes any disagreement a statement about
physics rather than about a frozen number.

The reference is itself checked (``test_reference_agrees_where_the_solver_is_known_good``)
on lossless stacks and on lossy *layers*, where solver and analytic agree to ~1e-16 for TE and
TM at normal and oblique incidence.

What it pins: every boundary quantity must enter in the same (conjugated) convention the
layers do. Two places did not.

* The substrate permittivity was raw ``n_bot**2``, so the stack reflected as though the
  substrate were ``conj(n_bot)`` — absorbing read as gain. Only the reflection *phase* was
  affected, so it hid wherever interference cannot show it (a bare interface, a vanishing
  layer) and grew with the stack above it: ~2.6% on one layer, ~12% on two.
* ``kz_top`` paired a raw ``n_top**2`` with an already-conjugated ``kx``, and the incident-field
  coefficients used a raw ``n_top``. With an absorbing superstrate the reflected amplitude was
  wrong outright — by ~1% at normal incidence and by more than the amplitude itself at oblique.

Real indices were never affected by either, which is why both survived every lossless test.

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


def solver_R(ns, thicknesses, wavelength, theta, pol, algo='TMM'):
    """Total reflectance of the same stack through the solver (uniform layers, fto=0)."""
    ucell = np.array([[[n, n]] for n in ns[1:-1]])
    solver = fieldwave.rcwa(backend=0, connecting_algo=algo, pol=pol, n_top=ns[0],
                           n_bot=ns[-1], theta=theta, fto=0, wavelength=wavelength,
                           period=[wavelength * 2], ucell=ucell, thickness=thicknesses)
    return float(np.sum(solver.conv_solve().de_ri))


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


@pytest.mark.parametrize('algo', ['TMM', 'SMM'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_reflection(case, pol, theta, algo):
    """Reflection off a stack on an absorbing substrate must match the analytic answer."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2
    assert solver_R(ns, ds, WL, theta, pol, algo) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
def test_absorbing_layer_identical_to_its_substrate(pol, theta):
    """An absorbing layer on a substrate of the same material must vanish into it.

    The short-circuit is forced *off* so this measures the boundary rather than the shortcut:
    the layer and the substrate have to arrive at the same modes on their own. They did not
    while the substrate skipped the conjugated convention, and the short-circuit was hiding
    it — the answer stayed right only because skipping the layer leaves a bare interface,
    where the wrong phase cannot show.
    """
    from fieldwave._core.geometry import StaticGeometry
    ns, ds = [1.0, AU, AU], [20.0]
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2   # == the bare air/gold interface

    ucell = np.array([[[AU, AU]]])
    common = dict(backend=0, pol=pol, n_top=1.0, n_bot=AU, theta=theta, fto=0,
                  wavelength=WL, period=[WL * 2], ucell=ucell, thickness=ds)
    forced = fieldwave.rcwa(connecting_algo='TMM', **common).conv_solve(
        geometry=StaticGeometry(same_material=[False]))
    smm = fieldwave.rcwa(connecting_algo='SMM', **common).conv_solve()

    assert float(np.sum(forced.de_ri)) == pytest.approx(want, abs=1e-12)
    assert float(np.sum(smm.de_ri)) == pytest.approx(want, abs=1e-12)


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


def solver_r(ns, thicknesses, wavelength, theta, pol, path, algo='TMM'):
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


@pytest.mark.parametrize('algo', ['TMM', 'SMM'])
@pytest.mark.parametrize('path', ['1d', 'conical', '2d'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', AMPLITUDE_CASES, ids=[c[0] for c in AMPLITUDE_CASES])
def test_reflection_amplitude(case, pol, theta, path, algo):
    """Complex reflection amplitude, for absorbing media on either side or both."""
    _, ns, ds = case
    want = analytic_r(ns, ds, WL, theta, pol)
    got = solver_r(ns, ds, WL, theta, pol, path, algo)
    assert abs(got - want) < 1e-12, (got, want)
