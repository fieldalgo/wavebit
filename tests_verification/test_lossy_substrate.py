"""Reflection off a stack on an *absorbing* substrate, against the analytic multilayer answer.

A uniform stack has a closed-form reflection coefficient, so this file needs no golden: the
oracle is the Fresnel/Airy recursion below, and the solver is driven with uniform layers at
``fto=0`` where only the zeroth order exists. That makes any disagreement a statement about
physics rather than about a frozen number.

The reference is itself checked (``test_reference_agrees_where_the_solver_is_known_good``)
on lossless stacks and on lossy *layers*, where solver and analytic agree to ~1e-16 for TE and
TM at normal and oblique incidence.

What it pins: with a lossy *substrate* the solver reflects as though the substrate were
``conj(n_bot)`` — an absorbing medium read as a gain medium. Only the phase of the substrate
reflection is affected, so it hides wherever interference does not matter (a bare interface,
or a vanishing layer) and grows with the number of layers above it: ~2.6% on one layer, ~12%
on two. Layers themselves are fine; the superstrate is not exercised here.
"""
import numpy as np
import pytest

import fieldwave

WL = 500.0
AU = 0.18 + 3.0j          # gold at 500nm — strongly absorbing
SI = 4.0 + 0.05j          # weakly absorbing


# ------------------------------------------------------------------ analytic reference

def _kz(n, k_par):
    """Out-of-plane wavevector on the decaying branch (``e^{-iwt}``, ``Im(n) > 0`` = loss)."""
    v = np.sqrt(complex(n) ** 2 - k_par ** 2)
    return v if v.imag >= 0 else -v


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


@pytest.mark.xfail(strict=True, reason='substrate reflects as conj(n_bot): absorbing read as gain')
@pytest.mark.parametrize('algo', ['TMM', 'SMM'])
@pytest.mark.parametrize('pol,theta', INCIDENCE, ids=IDS)
@pytest.mark.parametrize('case', LOSSY_SUBSTRATE, ids=[c[0] for c in LOSSY_SUBSTRATE])
def test_lossy_substrate_reflection(case, pol, theta, algo):
    """Reflection off a stack on an absorbing substrate must match the analytic answer."""
    _, ns, ds = case
    want = abs(analytic_r(ns, ds, WL, theta, pol)) ** 2
    assert solver_R(ns, ds, WL, theta, pol, algo) == pytest.approx(want, abs=1e-12)
