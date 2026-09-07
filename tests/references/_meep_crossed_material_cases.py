"""Small asymmetric crossed rasters with patterned mu or electric pole models.

Lengths are micrometres; Wavebit pole frequencies are angular frequencies (c=1).
Half-spaces are nondispersive and nonmagnetic so the existing E-field extraction and
power normalization apply unchanged. These are three selected moderate-contrast cases,
not a claim about negative-epsilon metals or arbitrary resonant structures.
"""
import numpy as np

MASK = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1]]], dtype=int)
OMEGA = 2*np.pi/.9
CASES = {}
for name, mu, pole in [
    ('magnetic', 1.5, None),
    ('drude', 1., (0., .2*OMEGA, .9*OMEGA**2)),
    ('lorentz', 1., (1.5*OMEGA, .2*OMEGA, 1.2*OMEGA**2)),
]:
    eps = np.where(MASK, 2.25, 1.)
    CASES[name] = dict(wavelength=.9, period=(.6, .8), thickness=(.2,),
                       theta_deg=20., phi_deg=35., n_top=1., n_bot=1.4,
                       index=np.sqrt(eps), mu=np.where(MASK, mu, 1.),
                       dispersion=MASK.copy() if pole else None,
                       poles={1: [pole]} if pole else None,
                       orders_u=(-1, 0, 1), orders_v=(-1, 0, 1),
                       meep_dpml=.5, meep_pad=.6, meep_until=120.)


def epsilon(case, wavelength):
    """Closed-form epsilon, independent of either solver's material evaluator."""
    eps = np.asarray(case['index'], dtype=complex)**2
    if case['poles']:
        omega = 2*np.pi/wavelength
        for identity, poles in case['poles'].items():
            for omega0, gamma, strength in poles:
                eps = eps + (case['dispersion'] == identity) * (
                    strength/(omega0**2 - omega**2 - 1j*gamma*omega))
    return eps
