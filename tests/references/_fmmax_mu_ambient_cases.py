"""Cases shared by ``gen_goldens_fmmax_mu_ambient.py`` and ``test_fmmax_mu_ambient_golden.py``.

Same split as the other fmmax pairs: the generator runs in ``fwref`` (fmmax, no wavebit) and the
test in the project env (the reverse), so the shared module imports numpy and nothing else.

**These are MAGNETIC HALF-SPACES** — the superstrate and/or substrate carry ``mu != 1``. Frozen
before the solver could express one, for the reason ``gen_goldens_fmmax_magnetic.py`` gives: an
oracle chosen after the work is an oracle chosen to agree.

Why this file needed a gate of its own. The magnetic *layer* golden could lean on its ``__mu1``
half — every case generated twice, once non-magnetically, and that half held to the solver while
the other could not be. Here there is no such twin: the ambient **is** the thing under test, so
a ``mu = 1`` companion tests nothing about it. The substitute is a closed form. A bare interface
between two non-absorbing media has one, set by the wave admittance::

    s:  Y = kz / mu          p:  Y = kz / eps          kz = sqrt(eps*mu - sin^2(theta))
    r = (Y_inc - Y_sub) / (Y_inc + Y_sub)

and fmmax reproduces it to 5e-16 across eight ``(eps, mu, theta)`` combinations, including
``mu < 1`` and a purely magnetic ``eps = 1``. ``ZERO_THICKNESS_CASES`` below keeps that gate in
the golden rather than leaving it in a scratch file: those rows are checkable against the
formula above by anyone, forever, with no solver involved.

**Ambients are stated as physical ``(eps, mu)``, not as an index.** The solver's spelling of a
magnetic half-space is an API decision that came after this file, and a golden that encoded it
would have to be regenerated if it changed — which is the one thing a golden must never need.
``n = sqrt(eps*mu)`` recovers the refractive index for whatever the API turns out to want.
"""
import numpy as np

WL = 700.0
PERIOD = 400.0
THICK = 210.0


def _t(xx, yy, zz, xy=0.0, yx=0.0):
    return np.array([[xx, xy, 0.0], [yx, yy, 0.0], [0.0, 0.0, zz]], dtype=complex)


ONE = _t(1.0, 1.0, 1.0)
VACUUM = (1.0, 1.0)          # (eps, mu) of a half-space

# name -> (top (eps, mu), bot (eps, mu), layer eps tensor, layer mu tensor, theta_deg)
#
# The layer is deliberately plain in most rows — a non-magnetic isotropic slab — so that a
# discrepancy is the *boundary* and not the layer operator, which its own golden already covers.
# The last two rows combine the two features on purpose: they are the only place a magnetic
# ambient and a magnetic layer meet, and each was validated against a different oracle.
CASES = {
    # substrate only
    'mag_substrate_normal': (VACUUM, (2.6, 1.7), _t(2.2, 2.2, 2.2), ONE, 0.0),
    'mag_substrate_35deg': (VACUUM, (2.6, 1.7), _t(2.2, 2.2, 2.2), ONE, 35.0),
    # superstrate only — the incidence medium, which is where the flux normalisation lives
    'mag_superstrate_normal': ((2.6, 1.7), VACUUM, _t(2.2, 2.2, 2.2), ONE, 0.0),
    'mag_superstrate_28deg': ((2.6, 1.7), VACUUM, _t(2.2, 2.2, 2.2), ONE, 28.0),
    # both, and different from each other
    'mag_both_41deg': ((1.9, 1.3), (2.6, 1.7), _t(2.2, 2.2, 2.2), ONE, 41.0),
    # a purely magnetic half-space: eps = 1, so nothing but mu distinguishes it from vacuum
    'purely_magnetic_substrate_35deg': (VACUUM, (1.0, 2.9), _t(2.2, 2.2, 2.2), ONE, 35.0),
    # mu < 1, which flips the sign of the impedance mismatch relative to every case above
    'diamagnetic_substrate_35deg': (VACUUM, (2.6, 0.6), _t(2.2, 2.2, 2.2), ONE, 35.0),
    # an index-matched pair that differs only in impedance: eps*mu is equal on both sides, so a
    # solver that carries `n` and drops `mu` sees no interface at all and returns R = 0.
    'impedance_only_mismatch_30deg': ((1.0, 1.0), (0.5, 2.0), ONE, ONE, 30.0),
    # the two features together
    'mag_substrate_and_mag_layer_35deg': (VACUUM, (2.6, 1.7), _t(2.2, 2.2, 2.2),
                                          _t(1.4, 1.4, 2.2), 35.0),
    'mag_both_and_gyromagnetic_layer_35deg': ((1.9, 1.3), (2.6, 1.7), _t(2.6, 2.6, 2.6),
                                              _t(1.6, 1.6, 1.6, 0.40j, -0.40j), 35.0),
}

# Bare interfaces — zero-thickness middle layer, so the whole answer is one boundary and the
# closed form above applies. These carry the convention gate that the layer golden got from its
# `__mu1` half.
#
# name -> (substrate (eps, mu), theta_deg); the superstrate is always vacuum.
ZERO_THICKNESS_CASES = {
    'interface_nonmagnetic_normal': ((2.6, 1.0), 0.0),
    'interface_nonmagnetic_35deg': ((2.6, 1.0), 35.0),
    'interface_magnetic_normal': ((2.6, 1.7), 0.0),
    'interface_magnetic_35deg': ((2.6, 1.7), 35.0),
    'interface_purely_magnetic_35deg': ((1.0, 2.9), 35.0),
    'interface_diamagnetic_35deg': ((2.6, 0.6), 35.0),
}


def fresnel(eps_sub, mu_sub, theta_deg, eps_inc=1.0, mu_inc=1.0):
    """Closed-form ``[R_s, T_s, R_p, T_p]`` at one interface. No solver, no oracle.

    ``kz`` is the same for both polarizations and depends only on ``eps*mu``; the polarizations
    differ solely in which of the two the admittance divides by. That is the whole content of
    "a magnetic half-space", and it is why a solver carrying only the refractive index cannot
    express one.
    """
    th = np.deg2rad(theta_deg)
    kz_inc = np.sqrt(complex(eps_inc * mu_inc) - complex(eps_inc * mu_inc) * np.sin(th) ** 2)
    kz_sub = np.sqrt(complex(eps_sub * mu_sub) - complex(eps_inc * mu_inc) * np.sin(th) ** 2)
    out = []
    for a_inc, a_sub in ((mu_inc, mu_sub), (eps_inc, eps_sub)):
        y0, y1 = kz_inc / a_inc, kz_sub / a_sub
        r = (y0 - y1) / (y0 + y1)
        out += [abs(r) ** 2, 1.0 - abs(r) ** 2]
    return np.asarray(out, dtype=float)
