"""Planning and runtime warnings, in one place to keep their imports free of cycles.

A warning here is a statement about the *answer*, not about the spelling: the run will
produce numbers, and the numbers will be worse than the caller probably expects. Anything
that would make the numbers meaningless is a refusal instead, and lives at the door that
owns it (``materials.py``, ``geometry.py``, ``boundaries.py``, ``_lower/plan.py``).
"""

__all__ = ['CoarseMeshWarning', 'AbsorberProximityWarning', 'UnsettledFieldWarning',
           'GrowingFieldWarning', 'PMLStabilityWarning', 'BoundedVoxelWarning',
           'SubpixelAccuracyWarning']


class SubpixelAccuracyWarning(UserWarning):
    """The smoothed interface rule has not completed its curved-scattering gate.

    The controlled glass-sphere study measured order 2.023 with adjoint cross
    coupling. Production validation over material contrasts remains separate.
    A benchmark result is not an error estimate for an arbitrary scene; exact
    voxel volume alone does not guarantee electromagnetic convergence.
    See docs/fd-mie-validation.md.
    """


class CoarseMeshWarning(UserWarning):
    """Fewer cells per wavelength inside a material than the convergence study supports.

    The wavelength in a medium is ``lambda / n``, so a structure of index ``n`` is sampled
    ``n`` times more coarsely than the vacuum around it at the same ``dx``. The legacy
    engine's measured convergence on a benchmark grating (``wavebit/_core/fdtd.py``,
    ``CELLS_PER_WAVELENGTH``) runs a worst per-order error of 1.2e-2 at ~11 cells per
    wavelength in the tooth, 2.8e-3 at 23 and 7.5e-4 at 46 -- so 12 is where the second
    digit stops being trustworthy and 8 is where the first one does. Below 8 the scene is
    refused; between 8 and 12 this warns.
    """


class AbsorberProximityWarning(UserWarning):
    """A structure sits within half a wavelength (in its own medium) of an absorbing face.

    A PML is matched to the *background* it terminates, and it terminates the near field of
    whatever is close to it as well. The legacy engine's measured clearance
    (``wavebit/_core/fdtd.py``, ``PAD_WAVELENGTHS``) is half a wavelength in the half-space:
    on the benchmark grating the worst per-order error runs 7.37e-4 at one wavelength of
    clearance, 6.57e-4 at half and 8.81e-4 at an eighth, so half is the optimum and less
    than half costs accuracy. This warns rather than refuses because a structure that
    *crosses* the face is a legitimate scene (the PML then has to be matched to it, or be
    an ``Absorber``); one that merely sits near it is only inaccurate.
    """


class UnsettledFieldWarning(UserWarning):
    """A frequency-domain monitor's region still held field when the run stopped.

    Stage 3's decay diagnostic, always on and never an error: at the end of every run each
    DFT monitor's region energy ``sum |F|^2`` over the last period of its longest wavelength is
    compared with its peak over the run (``wavebit/fd/_assemble.py::decay_report``). A DFT of a
    signal cut off before it decayed is not the spectrum of the physics, so every flux, order
    and normalisation built on it carries the truncation. The message names the monitor and
    the ratio; the line is ``fd._assemble.UNSETTLED_THRESHOLD``, measured. Its advice is to
    run longer, which is right for a field that decays; a field that **grows** after its
    sources have ended warns :class:`GrowingFieldWarning` instead.
    """


class GrowingFieldWarning(UnsettledFieldWarning):
    """The field still **rises** after every source has ended: a longer run makes it worse.

    Stage 3's growth diagnostic (``wavebit/fd/_assemble.py::decay_report``,
    ``res.diagnostics['interior_growth']``): once every source has ended -- a Dipole's envelope
    back below its cutoff, a PlaneWave's or TFSF's column no longer changing on its faces -- the
    total energy of a passive scene's interior -- every cell but the PML layers', an
    ``Absorber``'s included, a dispersive structure's stored energy too -- can only hold or
    fall, so when it averages higher over the last quarter of that span than over the quarter
    before, by more than ``fd._assemble.GROWTH_THRESHOLD`` (measured), and rises as an
    exponential does (``fd._assemble.GROWTH_STEADY``: energy a PML hands back arrives as a
    pulse), something is feeding it. An interior energy that overflows to ``inf`` or ``nan``
    warns this too, once, saying so. The scene is
    judged and not a monitor: a monitor's own region energy, or a Record's ``|v|^2``, can rise
    in a passive scene when an echo or a slow beat moves energy through it. In this package the
    feed is a PML: a stretch backed by its PEC wall is active for evanescent waves, and a
    lossless structure with a bound mode just past the light line grows under the pure
    stretch, the default (``wavebit/fd/_lower/cpml.py::stabilizing_rate``).

    One is raised for every monitor that sees the growth -- its own series rose over the same
    two quarters, or it is unsettled -- **instead of** its :class:`UnsettledFieldWarning`, and
    one for the interior when no monitor sees it. The message names the monitor, the
    interior's factor and the two spans compared, the monitor's own factor, and the knob:
    ``PML(stabilize=...)`` with the value measured for the face's reflection target (the
    weakest target's, over several pure-stretch faces); each face's value against the measured
    one, and that the structure lies outside the sampled families, when every face is already
    stabilized; a defect to report when no face is a PML. A subclass of
    :class:`UnsettledFieldWarning`, so a filter on that warning still sees it.
    """


class BoundedVoxelWarning(UserWarning):
    """Planning found CSG voxels evaluated with volume bounds instead of exact fills.

    One warning covers the static scene's component voxels and names the count, widest
    interval and responsible node. Union and Difference take an upper bound; Intersection
    takes a lower bound. The warning reports geometric uncertainty, not a bound on the
    resulting electromagnetic error. ``fd.check`` includes the same note with or without
    prebuilt params. Lowering emits no warning inside JIT; cached calls use the geometry
    that was checked at plan time. See ``_shapes.Approximations`` for the collector.
    """


class PMLStabilityWarning(UserWarning):
    """A stabilized PML asks for less loss than was measured to stop a bound mode growing.

    ``PML(stabilize=rho)`` adds a loss relative to the stretch's own ``sigma_max``, and the
    ``rho`` that suffices depends on the face's ``reflection=`` target: 3e-3 at the default
    1e-8, 5e-3 at 1e-6, 1e-2 at 1e-4, 2e-2 at 1e-3, measured on slab families with the
    one-step-matrix eigen gate (``wavebit/fd/_lower/cpml.py::MEASURED_STABILIZE``). Below it,
    or at a target weaker than any measured, a lossless structure with a bound mode next to
    the face may still grow; the plan warns and runs, because a smaller loss is a legitimate
    trade (it reflects less at an angle) and the growth depends on the structure.
    """
