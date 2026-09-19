"""The warnings stage 1 raises, in one place so every door can import them without a cycle.

A warning here is a statement about the *answer*, not about the spelling: the run will
produce numbers, and the numbers will be worse than the caller probably expects. Anything
that would make the numbers meaningless is a refusal instead, and lives at the door that
owns it (``materials.py``, ``geometry.py``, ``boundaries.py``, ``_lower/plan.py``).
"""

__all__ = ['CoarseMeshWarning', 'AbsorberProximityWarning', 'UnsettledFieldWarning',
           'PMLStabilityWarning']


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
    the ratio; the line is ``fd._assemble.UNSETTLED_THRESHOLD``, measured.
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
