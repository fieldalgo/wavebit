"""Cases shared by ``gen_goldens_pymoosh_dng.py`` and ``test_pymoosh_dng_golden.py``.

The generator runs in a throwaway environment holding PyMoosh and no ``wavebit``, the test in the
project environment, so what they share imports numpy and nothing else.

**What this oracle is for.** *Lossless* double negatives (``Re(eps) < 0``, ``Re(mu) < 0``, no
loss) in flat multilayers: as the substrate, as the lit superstrate, as an inner layer, and all
three at once. fmmax cannot arbitrate those -- it keeps every mode with ``Im(kz) > 0``, which a
lossless medium cannot read (``_fmmax_dng_cases.py``) -- and PyMoosh can, because it writes the
double-negative choice down (D. Langevin, P. Bennet, A. Khaireh-Walieh, P. Wiecha, O. Teytaud and
A. Moreau, "PyMoosh: a comprehensive numerical toolkit for computing the optical properties of
multilayered structures", J. Opt. Soc. Am. B 41, A67 (2024); the switch is in its code,
its ``coefficient_S`` function, in ``core.py``, not in the paper). Read in PyMoosh 4.0.1:

* every ``gamma`` (its ``kz``) starts as the principal root of ``eps*mu*k0**2 - alpha**2``;
* the **lit** medium's is negated when ``Re(eps) < 0`` and ``Re(mu) < 0``;
* the **last** medium's is negated when ``Re(eps) < 0`` and ``Re(mu) < 0`` and
  ``Re(gamma) != 0`` ("outgoing wave condition for the last medium");
* inner layers are flipped to ``Im(gamma) >= 0``, a label that moves no exact answer.

For every lossless medium that is this solver's root: Maystre's cut for a positive medium and its
mirror for a double negative (``wavebit._core.conventions.boundary_kz``). The two codes share no
code and no formulation beyond Maxwell's equations: PyMoosh composes 2x2 scattering matrices of
scalar TE / TM interfaces, this solver an RCWA stack with ``truncation_order = 0``.

**Its angle is the phase's.** PyMoosh's in-plane wavevector is
``alpha = sqrt(eps*mu)*k0*sin(theta)`` with the principal root, so from a double negative its
``theta`` names the direction of the *phase*, and its ``kx`` has the sign opposite to this
solver's, whose ``theta`` names the direction the light travels (``k_par = s*N*sin(theta)``,
``s = -1``). The golden therefore holds PyMoosh's ``kx`` (``__kx``), and the test hands this
solver that wavevector; a flat stack answers ``kx`` and ``-kx`` alike, so the sign is not what is
tested here (the fmmax golden's grating lit from a double negative tests it).

**What is frozen**, per case, per angle and per polarization (``0`` TE, ``1`` TM): PyMoosh's
``(r, t, R, T)``. ``r`` is referenced to the first interface and ``t`` to the last (the generator
gives the two half-spaces zero thickness). TE's ``r`` and ``t`` are ratios of ``E_y``, TM's of
``H_y``: this solver's ``r_p`` is that ``H`` ratio, and its ``t_p`` is the ``E`` ratio,
``t_H * (n1/mu1)/(n2/mu2)`` with the signed index.

Media are listed from the lit side down, as PyMoosh takes them; ``layers`` gives the inner ones
with their thicknesses. This solver's ``structure`` lists layers from the bottom up, so the test
reverses them. No two adjacent media are each other's negation (that pair has no solution).
"""
import numpy as np

WL = 500.0
PYMOOSH_VERSION = '4.0.1'

GLASS = (2.25, 1.0)
HIGH = (4.0, 1.0)
MAGNETIC = (2.0, 1.3)
AIR = (1.0, 1.0)
#: Lossless double negatives, ``(eps, mu)``, each with ``|mu| != 1``.
DNG_A = (-3.0, -0.5)           # n = -sqrt(1.5) = -1.2247
DNG_B = (-2.0, -1.5)           # n = -sqrt(3)   = -1.7321
DNG_C = (-6.0, -2.0)           # n = -sqrt(12) = -3.4641

#: name -> ``top`` (lit), ``layers`` [(medium, thickness)], ``bot``, and PyMoosh's angles. The
#: last angle of each case puts some medium of the stack past its critical angle.
CASES = {
    'dng_substrate': dict(top=GLASS, layers=[(HIGH, 80.0), (MAGNETIC, 130.0)], bot=DNG_A,
                          thetas=(0.0, 0.25, 0.7, 1.2)),
    'dng_superstrate': dict(top=DNG_B, layers=[(HIGH, 90.0), (GLASS, 60.0)], bot=MAGNETIC,
                            thetas=(0.0, 0.3, 0.75, 1.1)),
    'dng_inner_layer': dict(top=GLASS, layers=[(MAGNETIC, 70.0), (DNG_C, 110.0), (HIGH, 50.0)],
                            bot=AIR, thetas=(0.0, 0.25, 0.7, 1.2)),
    'dng_everywhere': dict(top=DNG_B, layers=[(DNG_C, 100.0), (HIGH, 70.0), (DNG_A, 45.0)],
                           bot=DNG_C, thetas=(0.0, 0.3, 0.75, 1.1)),
}


def media(name):
    """Every medium from the lit side down, as complex ``(eps, mu)``."""
    case = CASES[name]
    return [tuple(complex(v) for v in m) for m in
            [case['top'], *(medium for medium, _ in case['layers']), case['bot']]]


def thicknesses(name):
    """PyMoosh's thickness list: the half-spaces get zero, so ``r`` and ``t`` are referenced to
    the first and the last interface."""
    return [0.0, *(d for _, d in CASES[name]['layers']), 0.0]


def pymoosh_kx(name, theta):
    """PyMoosh's ``alpha / k0``: the principal ``sqrt(eps*mu)`` of the lit medium times
    ``sin(theta)`` -- for a double negative, the opposite sign to this solver's ``k_par``."""
    eps, mu = media(name)[0]
    return (np.sqrt(eps * mu) * np.sin(theta)).real
