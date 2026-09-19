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

__all__ = ['ORDER', 'KAPPA_MAX', 'ALPHA_RELATIVE', 'STABILIZING_LOSS', 'DEFAULT_STABILIZE',
           'profiles', 'face_eps_r', 'damping_profile', 'stabilizing_rate',
           'stabilizing_factors', 'MEASURED_STABILIZE', 'sufficient_stabilize', 'spell']

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

#: The stabilizing loss that suffices **at the default reflection target, 1e-8** (and at
#: stronger ones; a weaker target needs more, :data:`MEASURED_STABILIZE`), ``rho``: a matched loss ``rho * sigma(depth)`` on the
#: component **normal** to the face (``E_z`` and ``H_z`` behind a z face), at its own Yee point.
#: ``PML(stabilize=3e-3)`` asks for it. Why a PML needs one at all, why on the normal
#: component, why this value and what it costs: see :func:`stabilizing_rate`.
STABILIZING_LOSS = 3e-3

#: What ``PML(stabilize=None)`` -- the default -- means: **0, the pure stretch**. The loss is
#: opt-in because what it costs at oblique incidence breaks this package's own accuracy gates
#: (:func:`stabilizing_rate`, "Why it is not the default"); flipping this constant to
#: :data:`STABILIZING_LOSS` is the whole of the change the other choice would be.
DEFAULT_STABILIZE = 0.0

#: ``((reflection target, the smallest measured rho that suffices), ...)``, the weakest target
#: last. ``rho`` is relative to ``sigma_max``, and ``sigma_max`` falls with ``log(1/reflection)``,
#: but the loss that suffices does **not** stay fixed relative to it: a weaker target needs a
#: larger ``rho``, and a larger absolute rate too (``rho * log(1/reflection)`` = 0.055 / 0.069
#: / 0.092 / 0.14 down the table). Measured with the one-step-matrix eigen gate
#: (``tests/test_fd_pml_stability.py``) over its 24-mode slab family at 12 and 24 cells (the
#: three weaker targets on the grid ``3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2``): the value below
#: each entry still leaves
#: modes growing (1e-8: 2e-3 leaves 2/24 at 24 cells; 1e-6: 3e-3 leaves 2/24 and 6/24; 1e-4:
#: 7e-3 leaves 4/24; 1e-3: 1.5e-2 leaves 1/24 at 12 cells). Each entry is also 0/24 on the
#: stress family at 12 and 16 cells, and at a target between two entries (1e-7, 1e-5, 3e-4 at
#: 24 cells) under the weaker entry's value. A target of 1e-12 is stable at 3e-3; no target
#: weaker than 1e-3 was measured. :func:`sufficient_stabilize` reads it.
MEASURED_STABILIZE = ((1e-8, 3e-3), (1e-6, 5e-3), (1e-4, 1e-2), (1e-3, 2e-2))


def sufficient_stabilize(reflection):
    """The measured ``rho`` that suffices at ``reflection``, or ``None`` if none was measured.

    The entry of :data:`MEASURED_STABILIZE` for the strongest tabulated target at least as
    weak as ``reflection`` -- the value that sufficed at a weaker target, since a weaker target
    needs more. **Sufficient on the sampled slab families, not a guarantee**: the loss does
    not make the layer passive (that would need ``sigma_loss >~ sigma / kappa**2`` on the normal
    component, about two decades more), it outweighs the feed on the modes that were swept.
    """
    for target, rho in MEASURED_STABILIZE:
        if reflection <= target * (1.0 + 1e-12):
            return rho
    return None


def spell(x):
    """``3e-3`` for 0.003: a one-digit value as the docstrings and messages write it."""
    text = f'{x:.0e}'
    return text.replace('e-0', 'e-').replace('e+0', 'e')


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


def stabilizing_rate(depth, thickness, eps_r, reflection, rho):
    """The stabilizing loss rate at ``depth``: ``rho * sigma_max * f**ORDER``, zero outside.

    That is the stretch's own ``sigma`` profile, with the same ``1/sqrt(eps_r)`` of the units
    audit, times ``rho``. It is a rate (1/time): the kernel applies it to the normal component
    of E and of H alike, ``sigma_e/eps = sigma_m/mu = rate``, as ``ae``/``be`` and ``ah``/``bh``
    carry an ``Absorber``'s -- ``(1 - h)/(1 + h)`` on the field and ``1/(1 + h)`` on the curl,
    ``h = rate dt / 2`` -- written as two slab factors (:func:`stabilizing_factors`).

    **Why a PML needs it: a PEC-backed stretch is an active termination for evanescent waves.**
    A lossless structure with a bound mode just on the evanescent side of the light line --
    the guided resonances of a grating tooth folded to ``k_x = 2 pi / period``, or a slab mode
    of an ``Axis1D`` column -- grows exponentially under the pure CPML, at a few 1e-3 to
    2e-2 e-folds per unit time (``tests/test_fd_pml_stability.py`` measures it on a slab column;
    ``tests/test_fd_decay.py`` on the lamellar grating that first showed it). The mechanism, as
    the stage-3 investigation derived and measured it (not re-derived here): in stretched
    coordinates the layer acts only through its complex depth ``D = integral of s dz``, ``s =
    kappa + sigma/(alpha - i omega)``, and an evanescent order ``exp(-q z)`` meeting [layer +
    PEC] sees a stub of complex length ``D``. ``Im D > 0`` is what absorbs a propagating wave;
    for an evanescent one it is a round-trip *phase* ``2 q Im D``, and the power the stub takes out,
    ``-Im tanh(q D)`` for TM (``H_y`` is Neumann at the PEC) and ``+Im coth(q D)`` for TE,
    changes sign in windows of ``q``. At small ``q`` the TM one is ``-q Im D < 0`` for every
    stretch with ``Im D > 0``: the layer feeds the mode. The same fact read as a medium: the
    uniaxial tensor of a z stretch is ``diag(s, s, 1/s)``, and ``Im(1/s) < 0`` makes its
    **normal** component active. A continuum transfer-matrix root of slab + gap + graded
    CFS-PML + PEC (no FDTD) grows at the rate this kernel converges to as ``dx -> 0``, and it
    grows with ``alpha = 0`` too, so the defect is the backed stretch itself -- not the
    discretization, the recurrence or the CFS pole. Every constant of the stretch was swept
    (``kappa_max`` 1-10, ``alpha`` 0-10, order 2-4, 12-48 cells, targets 1e-4 to 1e-12) over 24
    slab modes: the worst rate moves by at most 3x and never reaches zero.

    **Why the normal component, and why this profile.** The loss has to sit where the
    evanescent tail is, and the tail of a small-``q`` mode fills the whole layer, so
    the loss is graded with the stretch's own ``sigma`` (a loss confined to the back cells,
    or behind the grading, did not stabilize). The active part of the medium is its normal
    component, so the loss goes there: on the 24-mode family it stabilizes at the same
    ``rho`` a loss on all six components does, reflects **nothing at normal incidence**
    (a normally incident wave has no normal component; the 0-degree column is unchanged to the
    digit) and half as much at every oblique angle.

    Other profiles were tried at the same gate: ``f**2`` stabilizes at ``rho = 2e-3`` and
    reflects 2x more at 30 and 60 degrees; ``f**8`` needs ``1e-2`` and buys nothing at 60
    degrees (3.75e-3 against 3.7e-3); ``f**12`` still leaves growing modes at ``2e-2``. To first
    order the oblique reflection is the loss integrated against the propagating wave's own
    round-trip attenuation, which the grading fixes, so the shape barely moves it.

    **Why 3e-3.** The smallest value with no growing mode on the 12-, 16- and 24-cell layers
    over the 24-mode family (eps 1.05-4, 0.08-0.24 thick, ``k_x = 2 pi/1.6, 2 pi/1.2,
    2 pi/0.8``) and a 24-mode stress family (eps up to 12, 0.02-0.08 thick, dx 0.02, four
    ``k_x``). ``2e-3`` leaves 2/24 growing on the 24-cell layer and ``1e-3`` 3-8/24, so the
    margin is 1.5x, not a decade.

    **3e-3 is for the default target, 1e-8, and stronger ones; a weaker target needs more.**
    The profile scales with ``sigma_max``, but the loss that suffices does not scale with it:
    at a weaker ``reflection=`` both ``rho`` and the absolute rate have to rise (5e-3 at 1e-6,
    1e-2 at 1e-4, 2e-2 at 1e-3; :data:`MEASURED_STABILIZE`), and ``3e-3`` there leaves 2-14 of
    the 24 modes growing. The plan warns (``PMLStabilityWarning``) when a stabilized face asks
    for less than :func:`sufficient_stabilize` of its target, or for a target weaker than any
    measured. **It is a measured balance, not passivity**: the normal component of the stretch
    is ``1/s`` with ``Im(1/s) = -sigma omega / ((alpha^2 + omega^2) |s|^2)``, and cancelling
    that point by point would take ``rho`` of order ``1/kappa^2``, about two decades above
    these values. The loss outweighs the feed on the slab families that were swept (and on
    32- and 48-cell layers, slab-to-layer gaps 0.02-0.24 and weak slabs, eps 1.01-1.1); a
    structure outside them can still grow, and the growth warning is what would say so.

    **What it costs**, amplitude reflection at the carrier (``tests/test_fd_pml.py``'s
    doubling probe), pure CPML -> stabilized::

        cells      0 deg              30 deg             60 deg             80 deg
        12    9.2e-05 -> 9.2e-05  4.7e-05 -> 2.1e-04  1.1e-04 -> 3.7e-03  3.4e-02 -> 1.1e-01
        24    2.2e-08 -> 2.2e-08  1.4e-07 -> 1.7e-05  1.2e-04 -> 9.2e-04  4.4e-02 -> 7.9e-02

    so the oblique columns are set by this loss and no longer by the target's
    ``reflection ** cos(theta)``, and a lower ``reflection=`` does not buy them back (the loss
    grows with ``sigma_max``: 24 cells at 60 degrees, 9.2e-4 at 1e-8 and 1.1e-3 at 1e-12). The
    near field of a dipole near the layer (``studies/fdtd_cpml_evanescent.py``'s probe,
    ``R_face`` at ``f0``, the loss on the probed ``z+`` face only) goes from 2e-7..5e-6 to
    5e-6..1.6e-4 (``E_z`` at 0.05 / 0.1 / 0.2 / 0.5 wavelengths: 5.1e-6 / 2.5e-5 / 8.8e-5 /
    1.6e-4; ``E_x``: 1.3e-5 / 5.1e-5 / 1.2e-4 / 6.1e-5) -- half what the same loss on all six
    components costs, as at every oblique angle of the table.

    **Why it is not the default.** Measured with it on by default, 71 of the fd suite's default
    tier failed: every frozen golden with a PML (expected), and also oracle gates -- the Airy
    slab at 30 degrees (5.0e-5 against its 2e-6 line), the closed-form PlaneWave at 20-40
    degrees (1.4e-5 to 2.5e-5 against 2e-6 / 3e-6), the uniform stack against RCWA (energy
    off by 2.8e-4 against 1e-4, and its second-order convergence falling to 1.5), and the
    measured verdicts of the grazing rule (``wavebit/fd/_lower/grazing.py``: at ``cos 0.26``
    the default face returned 3.7e-2 against the ``R_c`` 1.2e-2 the rule is built on, and the
    rule's ``tau = 1e-3`` is exceeded below ``cos theta ~ 0.7`` on 12 cells). A default that
    makes a flat stack or a plane wave less accurate to keep a grating from growing is a choice
    for the owner, not a fix, so the default stays the pure stretch and the growth is reported
    instead: when the energy of the scene's interior is still rising after every source has
    ended, the monitors that see it warn ``fd.GrowingFieldWarning`` (``_assemble.decay_report``,
    ``res.diagnostics['interior_growth']``), which says a longer run makes it worse and names
    ``PML(stabilize=...)`` with the value :func:`sufficient_stabilize` gives for the face's
    target (3e-3 at the default). A run that is merely cut off -- unsettled but not rising --
    is told to run longer and nothing else.
    """
    sigma_max = (-(ORDER + 1.0) * math.log(reflection)
                 / (2.0 * math.sqrt(eps_r) * thickness))
    f = np.clip(np.asarray(depth, dtype=np.float64) / thickness, 0.0, 1.0)
    return rho * sigma_max * f ** ORDER


def stabilizing_factors(cells, dx, dt, eps_r, side, reflection, rho):
    """``{'node': (keep, lift), 'half': (keep, lift)}`` of the stabilizing loss, 1-D.

    With ``h = rate dt / 2``, ``keep = (1 - h)/(1 + h)`` and ``lift = 1/(1 - h)``: the kernel
    multiplies the curl by ``lift`` before the update and the updated field by ``keep`` after
    it, which is the matched-loss update ``keep F + drive cF``, ``drive = 1/(1 + h)``, written
    so that it touches only the slab (``wavebit/fd/_kernel/cpml.py::damp``). Depth is measured
    exactly as :func:`profiles` measures it. The normal ``E`` component sits half a cell along
    the axis and reads ``half``; the normal ``H`` component sits on the node and reads
    ``node``. A ``stabilize=`` so large that ``h`` reaches 1 somewhere is refused: past it the
    factor changes sign and the update is no longer a loss.
    """
    thickness = cells * dx
    index = np.arange(cells, dtype=np.float64)
    out = {}
    for name, offset in (('node', 0.0), ('half', 0.5)):
        if side == '-':
            depth = (cells - index - offset) * dx
        else:
            depth = (index + offset) * dx
        h = 0.5 * dt * stabilizing_rate(depth, thickness, eps_r, reflection, rho)
        if not float(np.max(h)) < 0.5:
            raise ValueError(
                f'PML(stabilize={rho!r}): the loss reaches {float(np.max(h)):.3g} per half '
                f'step at the wall (rate * dt / 2), and the update stops being a loss at 1 '
                f'(refused from 0.5). The measured value is {STABILIZING_LOSS}; a value this '
                f'large is a conductor, not a stabilizer -- use fd.Absorber for that.')
        out[name] = ((1.0 - h) / (1.0 + h), 1.0 / (1.0 - h))
    return out
