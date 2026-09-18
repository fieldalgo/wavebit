"""The CPML profiles: ``b``, ``a`` and ``1/kappa`` per cell of depth, at the E and H points.

Ported from ``wavebit/_core/fdtd.py::cpml_coefficients`` -- a **kernel seed**: the formula and
its units audit are copied here, the function is never imported (nothing under ``wavebit/fd/``
imports the old engines). Roden & Gedney 2000; Taflove & Hagness sec. 7.9.

The recurrence the kernel runs, per psi (``wavebit/fd/_kernel/cpml.py``)::

    psi <- b * psi + a * d
    d   <- d / kappa + psi

and the profiles, at fractional depth ``f = depth / thickness`` into the layer::

    sigma_max = -(m + 1) log(R) / (2 sqrt(eps_r) thickness)
    alpha_max = ALPHA_RELATIVE * 2 pi / wavelength
    sigma = sigma_max f^m      kappa = 1 + (kappa_max - 1) f^m      alpha = alpha_max (1 - f)
    b = exp(-(sigma/kappa + alpha) dt)
    a = sigma (b - 1) / (kappa (sigma + kappa alpha))            [0 where the denominator is]

**The units audit, which both material-dependent quantities failed once, in the obvious
direction.** ``sigma`` in a *stretched coordinate* is normalised by the vacuum permittivity,
so the attenuation a wave accumulates is ``n * integral(sigma)`` and a fixed round-trip
target needs ``sigma_max`` proportional to ``1/sqrt(eps_r)``. Taflove & Hagness eq. 7.61
carries ``sqrt(eps_r)`` the other way up because there it is a *material* conductivity acting
against the local permittivity; transplanting that spelling into the stretch overshoots by
exactly ``eps_r``, which on an ``n = 3.5`` substrate reflected 5.1e-2 instead of 1.9e-3 and
returned ``T = 1.037`` from a lossless stack. And ``alpha`` is a frequency, so it is set
relative to the operating one -- which is why ``wavelength`` is required and has no default.
``alpha`` grades *opposite* to ``sigma``, largest at the inner face: that is what makes this
a complex-frequency-shifted PML and what lets it absorb evanescent fields.

``eps_r`` is ``eps * mu`` of the background the face is matched to; with the diagonal
materials of stage 1 it is taken per axis and the stretch along axis ``a`` uses the
*transverse* pair -- see :func:`face_eps_r`.

**Where each profile is sampled, and it is not a free choice.** ``curl E`` is a forward
difference, so ``D_a E[b]`` lands half a cell along ``a``, on the H points, and reads the
``half`` profile; ``curl H`` is backward, so ``D_a H[b]`` lands on the nodes and reads the
``node`` profile. Depth is measured from the layer's inner face, in length units, with the
grid's own asymmetry left alone: on a low face the wall is the node ``x_0``, which the node
profile reaches at ``depth = thickness``; on a high face the wall is ``x_N``, which no node
occupies, so the node profile stops at ``thickness - dx``. That asymmetry is the Yee grid's,
not a bug, and forcing it symmetric moves the wall.

**The three constants** come with the legacy engine's measurements, which were *not* re-run
here (the re-measurement stage 1 did run is the doubling table in ``tests/test_fd_pml.py``,
which measures the layer as a whole rather than each constant in isolation).
"""
import math

import numpy as np

from ..materials import concrete

__all__ = ['ORDER', 'KAPPA_MAX', 'ALPHA_RELATIVE', 'profiles', 'face_eps_r',
           'damping_profile']

#: CPML grading exponent. Measured on the legacy 1-D probe with everything else fixed: at
#: ``kappa_max = 1`` the reflection runs 1.29e-6 / 9.12e-9 / 9.28e-10 for ``m`` = 3 / 4 / 5,
#: so 4 is worth ~140x over the textbook default of 3. Not re-measured for this kernel.
ORDER = 4.0

#: Peak coordinate stretch. Above 1 it costs a little on the propagating reflection the
#: legacy probe measured (1.29e-6 / 1.36e-6 / 1.63e-6 at kappa 1 / 2 / 5) and is classically
#: justified by grazing and evanescent absorption, which that probe could not see. 2 is the
#: compromise, and it is the one constant here not settled by a measurement. The doubling
#: table's ``kappa = 1`` mutant is the first thing in this repository that can see it.
KAPPA_MAX = 2.0

#: The CFS pole, **as a multiple of the operating frequency** -- which is what makes it
#: dimensionless and the solver scale-free. The legacy measured optimum is flat between 1 and
#: 2 in every medium (1.15e-6 at 2.0 in air, glass and n = 3.5), rising to 3.9e-6 at 4. Zero
#: is not safe: the shifted pole is what absorbs the evanescent field a structure radiates
#: into the layer, and the doubling table's ``alpha = 0`` mutant is what shows it.
ALPHA_RELATIVE = 1.5


def face_eps_r(material, axis):
    """``eps_r = eps * mu`` the stretch along ``axis`` is matched to, or ``None`` if traced.

    For a diagonal medium the wave travelling along ``axis`` is governed by the *transverse*
    permittivity and permeability, so the two transverse pairs are what matter; the larger of
    the two ``eps*mu`` products is taken, because ``sigma_max ~ 1/sqrt(eps_r)`` and the larger
    index is the one that would be under-absorbed. On the isotropic media every stage-1 gate
    uses, the two agree and the choice is invisible.
    """
    values = []
    for c in range(3):
        if c == axis:
            continue
        e, m = concrete(material.eps[c]), concrete(material.mu_value[c])
        if e is None or m is None:
            return None
        values.append(e * m)
    return max(values)


def profiles(cells, dx, dt, wavelength, eps_r, side, reflection):
    """``{'node': (b, a, ik), 'half': (b, a, ik)}`` for one face, as 1-D arrays of ``cells``.

    ``side`` is ``'-'`` or ``'+'``. ``ik`` is ``1/kappa``, which is what the kernel multiplies
    by -- one division here instead of one per cell per step.
    """
    thickness = cells * dx
    index = np.arange(cells, dtype=np.float64)
    out = {}
    for name, offset in (('node', 0.0), ('half', 0.5)):
        if side == '-':
            depth = (cells - index - offset) * dx
        else:
            depth = (index + offset) * dx
        out[name] = _coefficients(depth, thickness, dt, wavelength, eps_r, reflection)
    return out


def _coefficients(depth, thickness, dt, wavelength, eps_r, reflection):
    """``(b, a, 1/kappa)`` at the given depths; the ported formula, see the module docstring."""
    sigma_max = (-(ORDER + 1.0) * math.log(reflection)
                 / (2.0 * math.sqrt(eps_r) * thickness))
    alpha_max = ALPHA_RELATIVE * 2.0 * math.pi / wavelength
    f = np.clip(depth / thickness, 0.0, 1.0)
    sigma = sigma_max * f ** ORDER
    kappa = 1.0 + (KAPPA_MAX - 1.0) * f ** ORDER
    alpha = alpha_max * (1.0 - f)
    b = np.exp(-(sigma / kappa + alpha) * dt)
    denominator = kappa * (sigma + kappa * alpha)
    safe = np.where(denominator == 0.0, 1.0, denominator)
    a = np.where(denominator == 0.0, 0.0, sigma * (b - 1.0) / safe)
    return b, a, 1.0 / kappa


def damping_profile(depth, thickness):
    """The ``Absorber``'s matched rate, ``6 u**3 / thickness`` with ``u = depth / thickness``.

    Ported from ``wavebit/_core/grid_fdtd.py::damping_profile`` (a kernel seed).

    **What the layer attenuates by, and the factor of two that is easy to double-count.** The
    electric and the magnetic rate are *equal*, and that is what "matched" means here: with
    ``sigma_e/eps = sigma_m/mu = r`` the substitution ``E, H -> e^{-rt}E', e^{-rt}H'`` returns
    the lossless pair exactly, so the two matched rates set the attenuation constant to ``r``
    and not to ``2 r``. The integral of ``6 u^3`` over a unit-width layer is ``3/2``, so **one
    pass** through the layer attenuates the continuum amplitude by ``exp(-1.5)`` and the round
    trip to the PEC wall behind it and back -- which is what a normally incident wave actually
    suffers -- is ``exp(-3) = 4.98e-2``, whatever the depth, because the profile is normalised
    by its own thickness. Measured at every depth from 8 to 32 cells in
    ``tests/test_fd_absorber.py``, whose 0-degree column runs 4.81e-2 to 5.04e-2 and which
    lands on ``exp(-6)`` and ``exp(-9)`` when the profile is scaled x2 and x3.

    Matched rates reflect nothing at normal incidence, and everything the stretch would have
    absorbed at an angle.

    **The strength is ported, not measured here, and it is far from this probe's optimum**:
    scaling ``6 u^3`` by 5 takes the normal-incidence number from 4.9e-2 to 1.3e-5 while 30 and
    60 degrees do not move. It is left where it is for the reason ``KAPPA_MAX`` is: what the
    strength is *for* is the evanescent near field a structure radiates into the layer, and
    nothing in this repository can see one yet
    (``tests/test_fd_absorber.py::test_the_profile_strength_is_not_settled_by_this_probe``).
    """
    u = np.clip(depth / thickness, 0.0, 1.0)
    return 6.0 * u ** 3 / thickness
