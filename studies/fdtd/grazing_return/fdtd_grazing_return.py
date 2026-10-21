"""Whether a grazing order still carries a CPML return now that the PML is backed by PEC.

**The question.** The legacy engines' grazing orders (exit ``cos theta ~ 0.26``) carried a
return from the z termination that did not converge with resolution; 24 cells, a 1e-16
target and 640 periods removed it, neither knob alone. ``wavebit.fdtd._legacy`` backs every PML with a
PEC wall, which the legacy ``common`` route did not (it wrapped z periodically), and that
confound was never separated. The stage-3 plan asks to measure it before a Diffraction
monitor is built on top: does the return survive the PEC backing, does it converge, and what
must a scene declare before an order below some ``cos theta`` may be reported.

**The probe.** An ``fd.PlaneWave`` (``Band(wavelength=1, df=0.15)``, the gates' band: the
spectrum spans 0.44-1.56 f0 at the 1e-3 cutoff) in a Bloch cell uniform in x and y
(``fd.Bloch.from_source``; measured on a 1 x 1 x nz column, run now with a ``BLOCH_CELLS``-cell
x period -- see "Running it" below), launched toward ``-z`` at the angle whose band-centre
exit cosine is 0.15 / 0.2 / 0.26 / 0.3 / 0.4 / 0.5 / 0.7, in vacuum, so the exit angle is the
incidence angle. The sheet sits 1 wavelength above the swept ``z-`` PML; the ``z+`` face is
the default 12-cell PML and is never swept. Two ``Record`` probes 5 and 6 cells in front of
the swept layer read the tangential E (``Ey`` for s; ``Ex`` for p). Everything is at courant
0.5, NumPy, CPU.

**The doubling method.** The reference is the same run on the same cells with the ``z-`` face
moved ``ext`` cells further down, ``2 ext dx >= 1.1 T`` for the longest window ``T`` (640
periods), so nothing the reference's layer returns reaches the probes inside any window.
``test - ref`` at the probes is then the swept layer's doing and nothing else. Control 1
moves the reference face 25% further still: the probe series change by exactly 0.0 in all 28
(resolution, angle, polarisation) cases. Control 2: a 40-period run is bit-identical to the
first 40 periods of the 640-period run, so every window is read off one run.

**What is read.** Every number is at the band centre ``f0`` (the angle in the column heading is
exact only there: the Bloch k is fixed, so the angle varies across the band). The two probes'
DFTs are split into the wave travelling toward the layer and the wave leaving it with the Yee
``kz dx`` of the band centre (``E_j = a e^{-i kappa j} + b e^{+i kappa j}``):

- ``R``: ``|b(test - ref)| / |a(ref)|``, a plain rectangular DFT over the window.
- ``R_taper``: the same with a Tukey taper over the second half of the window -- the layer's
  own f0 return with the window's leakage suppressed.
- ``R_user``: ``|b(test)|`` at window ``T`` over the ideal incident ``a`` (the reference's own
  f0 phasor tapered over 640 periods) -- what a plain running DFT of the run itself reads.
- ``total_user``: ``|DFT(test)(T) - ideal| / |ideal|`` at the probe itself, both directions.
- worst in time: ``max|test - ref| / max|ref|`` at the probe.

Measured 2026-09-19 on the one-cell column, NumPy, CPU, 294 test runs
(``studies/fdtd/grazing_return/records/fdtd_grazing_return.log``, every table in full).

**1. The return survives the PEC backing, and it converges -- to the layer's designed round
trip, from below.** A PEC-backed graded CFS-PML has a continuum reflection at f0 of

    |R_c| = exp(-2 w cos(theta) int sigma w / (w^2 + alpha^2) dz) = target ** (g cos theta)
    g = (m + 1) int_0^1 f^m / (1 + (a_rel (1 - f))^2) df = 0.919    (m = 4, a_rel = 1.5)

-- the round trip the grading is aimed at, raised to ``cos theta`` (the path along z is
shorter by it), independent of the thickness. At normal incidence that is the target; at
``cos 0.26`` it is ``target ** 0.24``. ``R_taper``, 640 periods, s, a layer of fixed physical
thickness 1.2 wavelengths at 20 / 40 / 80 cells per wavelength (24 / 48 / 96 cells) against
``R_c``::

    exit cos theta     0.15      0.2      0.26     0.3      0.4      0.5      0.7
    1e-8   20 cpw     4.5e-02  1.9e-02  6.9e-03  3.5e-03  6.0e-04  1.0e-04  3.0e-06
           40 cpw     6.7e-02  2.8e-02  9.7e-03  4.9e-03  8.5e-04  1.5e-04  4.5e-06
           80 cpw     7.6e-02  3.2e-02  1.1e-02  5.6e-03  1.0e-03  1.8e-04  5.7e-06
           R_c        7.9e-02  3.4e-02  1.2e-02  6.2e-03  1.1e-03  2.1e-04  7.2e-06
    1e-12  20 cpw     7.3e-03  2.0e-03  4.3e-04  1.5e-04  1.0e-05  7.2e-07  1.0e-08
           40 cpw     1.5e-02  4.0e-03  8.3e-04  2.9e-04  2.0e-05  1.4e-06  7.5e-09
           80 cpw     1.9e-02  5.2e-03  1.1e-03  3.8e-04  2.8e-05  2.1e-06  1.1e-08
           R_c        2.2e-02  6.2e-03  1.4e-03  4.9e-04  3.9e-05  3.1e-06  1.9e-08
    1e-16  20 cpw     1.1e-03  1.8e-04  2.1e-05  5.3e-06  1.6e-07  5.8e-09  9.3e-09
           40 cpw     3.2e-03  5.5e-04  6.4e-05  1.5e-05  4.0e-07  1.8e-08  1.7e-10
           80 cpw     4.8e-03  8.3e-04  1.0e-04  2.5e-05  7.2e-07  2.7e-08  7.7e-11
           R_c        6.2e-03  1.1e-03  1.5e-04  3.9e-05  1.3e-06  4.5e-08  5.1e-11

The 0.6 and 1.8 wavelength layers do the same (the log's last two tables); at 1.8
wavelengths and 80 cells per wavelength every entry above 1e-9 sits at 0.69-1.01 of ``R_c``,
still rising. So the return **does** converge with resolution at fixed physical thickness,
roughly first order (at ``cos 0.26``, 1e-8: steps of 2.8e-3 then 1.3e-3), but **upward**, toward a
floor that is not zero: a finer grid is worse on a grazing order, never better. That is what
the legacy engine's "grows with resolution" was (0.0028 at r20 -> 0.0033 at r40 on a
``cos 0.26`` order). The PEC wall does not remove the return; it makes it exactly the designed
round trip, which is the one thing a user can compute in advance. The coarse, thin layers
reflect *less* than ``R_c`` at grazing (12 cells at 1e-16, 20 cpw: 3.5e-4 at ``cos 0.15``
against ``R_c`` = 6.2e-3), which is under-resolution, not absorption, and disappears as the
grid refines (80 cpw, 48 cells: 3.7e-3).

**2. The task's grid, ``R_taper`` at 640 periods** (the settled numbers; the log has every
window)::

    20 cpw              0.15     0.2      0.26     0.3      0.4      0.5      0.7
    12 cells  1e-8    3.3e-02  1.4e-02  4.8e-03  2.3e-03  3.9e-04  7.6e-05  2.2e-05
    12 cells  1e-12   3.9e-03  1.0e-03  2.0e-04  7.5e-05  2.1e-05  3.6e-05  1.0e-04
    12 cells  1e-16   3.5e-04  6.5e-05  3.2e-05  3.5e-05  6.5e-05  1.1e-04  2.6e-04
    24 cells  1e-8    4.5e-02  1.9e-02  6.9e-03  3.5e-03  6.0e-04  1.0e-04  3.0e-06
    24 cells  1e-12   7.3e-03  2.0e-03  4.3e-04  1.5e-04  1.0e-05  7.2e-07  1.0e-08
    24 cells  1e-16   1.1e-03  1.8e-04  2.1e-05  5.3e-06  1.6e-07  5.8e-09  9.3e-09
    36 cells  1e-8    4.7e-02  2.1e-02  8.0e-03  4.0e-03  7.2e-04  1.2e-04  3.5e-06
    36 cells  1e-12   8.8e-03  2.6e-03  5.7e-04  2.0e-04  1.5e-05  1.0e-06  5.4e-09
    36 cells  1e-16   1.5e-03  2.9e-04  3.6e-05  8.7e-06  2.4e-07  5.9e-09  1.2e-09
    40 cpw
    12 cells  1e-8    4.3e-02  1.7e-02  5.6e-03  2.7e-03  4.3e-04  7.9e-05  5.1e-06
    12 cells  1e-16   6.6e-04  9.7e-05  2.3e-05  2.0e-05  3.6e-05  5.8e-05  1.2e-04
    24 cells  1e-12   1.1e-02  2.8e-03  5.4e-04  1.8e-04  1.2e-05  8.3e-07  8.0e-09
    24 cells  1e-16   1.9e-03  2.9e-04  3.0e-05  6.8e-06  2.0e-07  7.0e-09  4.5e-09

Two terms, and the table is their sum. The **round trip** ``R_c`` dominates at grazing and is
set by the target alone, so cells do not buy it down (at 1e-8, 36 cells is worse than 12,
because the thicker layer is better resolved and closer to ``R_c``). The **discretisation**
reflection dominates near normal and on strong, thin layers, and cells buy that down: 12
cells at 1e-16 reflects 2.6e-4 / 1.2e-4 at ``cos 0.7`` (20 / 40 cpw), 12-24x the default
layer's, which 24 cells takes to 1e-8. So a target below the default needs the cells to go
with it, or every non-grazing order gets worse -- the legacy's "neither knob alone",
explained. ``p`` equals ``s`` to 2.6e-7 in every entry (the one-cell Bloch column's
tangential-E recurrence is the same for both), so only ``s`` is tabulated.

**3. The window is a separate error, and it belongs to the band, not the layer.** With the
Bloch k fixed, the band crosses its Rayleigh point ``f_R = f0 sin(theta)`` at every angle here,
and the content near ``f_R`` has almost no group velocity: it lingers at the probe with a
``~t^{-3/2}`` tail. A rectangular DFT at f0 over ``T`` periods picks it up. What a *perfect*
layer (the reference) still leaves, ``|DFT_ref(T) - ideal| / |ideal|``, 20 cpw::

    window           0.15     0.2      0.26     0.3      0.4      0.5      0.7
    40 periods     3.5e-02  2.3e-02  1.5e-02  1.1e-02  5.4e-03  2.8e-03  3.0e-04
    160 periods    4.3e-03  2.6e-03  1.7e-03  1.2e-03  6.4e-04  2.9e-04  3.1e-05
    640 periods    5.3e-04  3.2e-04  2.0e-04  1.5e-04  7.8e-05  3.4e-05  3.1e-06

8-10x per 4x of window, ``T^{-1.5}``, and the same at 40 and 80 cpw within 1.7x. In a run with
a near layer the tail is cut short, so the run's own reading settles sooner: ``total_user``
for 24 cells at 1e-16 (20 cpw) is 1.4e-2 / 1.6e-4 / 4.2e-5 at ``cos 0.26`` for 40 / 160 / 640
periods, and 2.8e-3 / 3.7e-5 / 9.7e-9 at ``cos 0.5``. **At 40 periods (the legacy default)
no layer setting helps**: every 1e-12 and 1e-16 row has ``R_user`` 5e-3 to 1.2e-2 at
``cos 0.26``, the band still passing the probe. At 160 periods ``R_user`` matches the settled
``R_taper`` to 10% wherever the return is above 2e-3; below 1e-3 the window term (1e-4 to
9e-4 at ``cos <= 0.3``) is comparable. Worst-in-time does not separate the angles at all
(the default layer: 1.2e-2 at ``cos 0.15``, 7.4e-3 at 0.4), because it is the most grazing
part of the band that sets it; a strong layer lowers it (24 cells, 1e-16: 1.3-2.0e-3), no
layer removes it.

**4. The sheet mirrors the return.** The down-going part of ``test - ref`` equals the up-going
return (median ratio 1.00, 10-90% 0.90-1.08, windows >= 160): the PlaneWave's column is
driven by a hard source, which reflects with ``r = -1`` whatever comes back up the column, and
in a uniform cell the column's return *is* the main grid's -- so the returned wave is
re-launched as incident. A field monitor between the sheet and the layer therefore reads
anywhere from 0 to ``2 R`` by its position (``total_user / R_user`` at 80 cpw, 144 cells,
1e-8: 1.4-1.9 at ``cos <= 0.4``, 0.3 at 0.5), and in a structured scene the column's own
return reaches the structure as a delayed spurious incident wave of size ``R(theta_inc)``. The
Diffraction normalisation must not read the column's DFT as clean incidence at a grazing
incident angle; the same rule has to apply to the incidence angle as to an order.

**The rule** (the stage-3 grazing gate). Tolerance ``tau = 1e-3`` in amplitude: the strictest
tolerance a measured setting still meets down to ``cos 0.2``, and 3x below the legacy's
grazing engine gap. From finding 1, a PEC-backed PML of target ``R`` meets it on an order
leaving through it at f0 exactly when

    cos theta >= cos_min(R) = ln(tau) / (g ln R),   g = 0.919

- **Default face** (``PML(cells=12)``, ``reflection=1e-8``): ``cos_min`` = **0.41**. An order
  (or a PlaneWave's incidence) whose exit ``cos theta`` at any wavelength of a Diffraction
  monitor is below 0.41 in a Bloch+PML run is refused, naming the face it exits through.
  Measured at ``cos 0.4``: 3.9e-4 (20 cpw), 4.3e-4 (40), converging to 1.0e-3 (80 cpw, 1.8
  wavelengths); at 0.3 already 2.3e-3 -> 5.8e-3.
- **Allowed below 0.41** only when the scene declares, on the face that order exits through:
  ``PML(cells >= 24, reflection=R)`` with ``cos theta >= cos_min(R)`` -- ``R = 1e-12`` reaches
  **0.27** (measured at ``cos 0.3``, 1.2 wavelengths: 1.5e-4 / 2.9e-4 / 3.8e-4 at 20 / 40 /
  80 cpw, ``R_c`` 4.9e-4), ``R = 1e-16`` reaches **0.204** (at ``cos 0.26``: 2.1e-5 / 6.4e-5
  / 1.0e-4, ``R_c`` 1.5e-4; at 0.2 the continuum is 1.1e-3, just outside) -- **and**
  ``Run(periods >= 160)``, or the window error of finding 3 (1.7e-3 at ``cos 0.26`` at 160
  periods on a perfect layer, 1.6e-4 in the run itself) is the larger term. The 24 cells are what keep every non-grazing
  order no worse than under the default face (24 cells at 1e-16: <= 1e-8 at ``cos 0.7``,
  against 2.6e-4 at 12 cells). Cost: 12 more cells deep on that face, and 4x the steps of a
  40-period run.
- **Refused outright below 0.204.** No measured setting meets ``tau`` there: 1e-16 returns
  5.2e-3 at ``cos 0.15`` (80 cpw, converged to ``R_c`` = 6.2e-3), and a target beyond 1e-16 is
  not measured (the formula asks 1e-22 at 0.15, where the discretisation term of a 24-cell
  layer is unknown).
- **The residual that remains** when allowed: the layer's return, up to ``R_c = R^{g cos}``
  <= 1e-3 on the grazing order (the grid gives less, approaching it from below) and <= 1e-8 of
  discretisation on the others at 24 cells; up to twice that at a monitor between a
  PlaneWave's sheet and the layer (finding 4); plus the band's window term, ~``T^{-3/2}``, at
  or below 3e-4 at ``cos >= 0.26`` at 160 periods for this band in the run's own reading. The
  message quotes ``R_c`` for the order and ``cos_min`` for the face.

**What this does not measure.** One band (``df = 0.15 f0``); a narrower band moves the
Rayleigh point's weight and the window term, not the layer's return. Vacuum only: in a medium
``sigma_max`` is matched through ``sqrt(eps_r)`` and ``R_c`` should be the same function of the
*medium's* ``cos theta``, not checked here. Only the ``z-`` face and only ``phi = 0``. Targets
beyond 1e-16 and layers beyond 144 cells (1.8 wavelengths at 80 cpw) are not measured. The
tolerance ``tau`` is a choice; ``cos_min`` for another one is the formula above (``tau = 1e-2``:
0.27 at the default face; ``tau = 1e-4``: 0.54).

**Running it.** The tables were measured on a one-cell Bloch column. Two product checks added
since refuse that scene; the script works round both without changing what it measures.

- ``checks.check_one_cell_bloch`` refuses a Bloch phase on a one-cell axis in a scene with a
  source. The script gives x a ``BLOCH_CELLS``-cell period instead. Every cell of a periodic
  grid applies the one-cell difference, ``K = 2 sin(k dx/2)/dx``, to the PlaneWave's single
  Bloch mode, so each probe series is the one-cell column's times one unit-modulus phase (equal
  to round-off), and every number read here is a ratio that cancels it. ``fd.InvariantAxis``
  (``Axis1D``), which that refusal names, is a different column: its derivative is ``i k``
  exactly, not ``K``, and the grazing return moves with it.
- The grazing rule (``src/wavebit/fdtd/_prepare/grazing.py``), set from this study, refuses
  every scene here at ``cos <= 0.4``, references included: the PlaneWave's incidence on the
  default ``z+`` face alone is below that face's ``cos_min`` of 0.41. Inside the script's own
  runs the rule records instead of refusing (``_rule_recorded``). It checks the scene and
  changes no coefficient.

Rerun this way on 2026-10-21, the script reproduced every ``GRAZING_TABLE`` entry it was
checked on, to the printed two digits: the three 20 cpw rows and both window rows at all seven
cosines, and the 80 cpw row at ``cos 0.26`` and 0.7 (37 entries). The continuum row needs no
run.

Run from the repository root. The committed log is the 2026-09-19 measurement that
``GRAZING_TABLE`` is checked against; write a rerun elsewhere::

    mkdir -p artifacts/studies
    python studies/fdtd/grazing_return/fdtd_grazing_return.py [--workers 8] > artifacts/studies/fdtd_grazing_return.log

NumPy, CPU, 356 solves (56 references and controls, 294 tests, 6 truncation runs); each worker
is given two BLAS/OpenMP threads.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import numpy as np                                      # noqa: E402

sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import wavebit.fdtd._legacy as fd                                 # noqa: E402
from wavebit.fdtd._prepare import cpml as cpml_module       # noqa: E402
from wavebit.fdtd._prepare import grazing as grazing_rule   # noqa: E402

COSINES = (0.15, 0.2, 0.26, 0.3, 0.4, 0.5, 0.7)        # exit cos(theta) at the band centre
CELLS = (12, 24, 36)
TARGETS = (1e-8, 1e-12, 1e-16)
WINDOWS = (40, 160, 640)                                # periods of the band centre
DEFAULT_WINDOW = 40                                     # the legacy engine's n_periods default
WAVELENGTH = 1.0
DF = 0.15                                               # tests/fixtures/fdtd/fd_helpers.py BAND
TOP = 12                                                # the z+ layer, never swept
ABOVE = 4                                               # cells between the z+ layer and the sheet
DISTANCE = 1.0                                          # sheet to the z- layer, wavelengths
PROBE = 5                                               # probe cells in front of the z- layer
CONTROL_GROWTH = 1.25                                   # the longer control's extension factor
# The x period, in cells ("Running it" above): ``checks.check_one_cell_bloch`` refuses the
# one-cell column the tables were measured on, and its own docstring states the equivalence --
# every cell of a periodic grid applies the one-cell difference ``(exp(i k dx) - 1)/dx`` to the
# PlaneWave's single Bloch mode ``exp(i k x)``.
BLOCH_CELLS = 4

# (cells per wavelength, polarisation, layer cells). 20 and 40 carry the task's grid; 40
# adds 48 and 72 and 80 carries 48 / 96 / 144, so 0.6 / 1.2 / 1.8 wavelengths of layer are
# measured at three resolutions (the fixed-physical-thickness arm).
ARMS = ((20, 's', (12, 24, 36)), (20, 'p', (12, 24, 36)),
        (40, 's', (12, 24, 36, 48, 72)), (80, 's', (48, 96, 144)))


def band():
    return fd.Band(wavelength=WAVELENGTH, df=DF)


def grid(cpw):
    dx = WAVELENGTH / cpw
    dt = 0.5 * dx / math.sqrt(3.0)
    return dx, dt


def window_steps(cpw, periods):
    """What ``Run(periods=)`` would take: ``ceil(periods * wavelength / dt)``."""
    _, dt = grid(cpw)
    return int(math.ceil(periods * WAVELENGTH / dt - 1e-9))


def extension(cpw):
    """Reference cells past the z- face: nothing it returns reaches the probe in the window.

    A return needs ``2 ext dx`` of path at a speed of at most 1 (the discrete group speed is
    below c), so ``ext >= 0.55 T / dx`` for the longest window ``T`` leaves 10% margin. The
    control moves the face 25% further and must change nothing.
    """
    dx, _ = grid(cpw)
    return int(math.ceil(0.55 * max(WINDOWS) * WAVELENGTH / dx)) + 20


def theta_of(cos):
    return math.degrees(math.acos(cos))


def scene(cpw, cos, pol, cells, target, ext, steps):
    """A ``BLOCH_CELLS x 1 x nz`` Bloch cell: the sheet near the top, the swept PML below."""
    dx, _ = grid(cpw)
    D = int(round(DISTANCE * cpw))
    nz = TOP + ABOVE + D + cells + ext
    z_sheet = -(TOP + ABOVE) * dx
    z_face = z_sheet - D * dx                           # the swept layer\'s inner face
    pw = fd.PlaneWave('pw', z=z_sheet, direction='-z', theta=theta_of(cos),
                      polarization=pol, band=band())
    comp = 'Ey' if pol == 's' else 'Ex'
    monitors = tuple(fd.Record(f'p{i}', at=(0.0, 0.0, z_face + (PROBE + i) * dx),
                               component=comp) for i in (0, 1))
    return fd.Scene(domain=fd.Domain(size=(BLOCH_CELLS * dx, dx, nz * dx),
                                     center=(0.0, 0.0, -nz * dx / 2)),
                    boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                             z=(fd.PML(cells=cells, reflection=target,
                                                       allow_thin=True),
                                                fd.PML(cells=TOP))),
                    resolution=fd.Resolution(dx=dx), run=fd.Run(steps=steps),
                    sources=(pw,), monitors=monitors)


@contextlib.contextmanager
def _rule_recorded():
    """The grazing rule surveys instead of refusing, for this probe's own runs only.

    The rule (``src/wavebit/fdtd/_prepare/grazing.py``, set from this measurement) refuses a
    PlaneWave whose incidence a face cannot absorb to ``tau`` -- below ``cos 0.41`` on the
    default ``z+`` face -- and that return is what this study measures. The rule checks the
    scene and changes no coefficient, so the run is the same; ``res.diagnostics['grazing']``
    still records its verdicts. ``tests/fdtd/test_fd_grazing.py`` measures refused faces the
    same way (``_run_unenforced``, ``_plan_unenforced``). Scoped, so nothing that imports this
    module inherits a disabled rule.
    """
    enforce = grazing_rule.enforce
    grazing_rule.enforce = grazing_rule.survey
    try:
        yield
    finally:
        grazing_rule.enforce = enforce


def series(sc):
    with warnings.catch_warnings(), _rule_recorded():
        warnings.simplefilter('ignore')
        res = fd.run(sc)
    t = np.asarray(res['p0'].t, dtype=float)
    return t, np.stack([np.asarray(res[f'p{i}'].values) for i in (0, 1)])


def yee_kappa(cpw, cos):
    """``kz dx`` of the discrete plane wave at the band centre, the Bloch k fixed at it."""
    dx, dt = grid(cpw)
    w = 2 * math.pi / WAVELENGTH
    kx = w * math.sqrt(1 - cos ** 2)
    s = (math.sin(w * dt / 2) / dt) ** 2 - (math.sin(kx * dx / 2) / dx) ** 2
    return 2 * math.asin(dx * math.sqrt(s))


def tukey(n):
    """1 over the first half of ``n`` samples, a cosine roll-off to 0 over the second."""
    u = np.arange(n) / n
    return np.where(u < 0.5, 1.0, 0.5 * (1 + np.cos(np.pi * (u - 0.5) / 0.5)))


def dft(values, t, taper=None):
    w = 2 * math.pi / WAVELENGTH
    phase = np.exp(1j * w * t) if taper is None else np.exp(1j * w * t) * taper
    return [complex(np.sum(v * phase)) for v in values]


def split(values, t, cpw, cos, taper=None):
    """``(down, up)`` amplitudes at probe 0 from the two probes' DFT at the band centre.

    ``E_j = a exp(-i kappa j) + b exp(+i kappa j)`` with ``kappa`` the Yee ``kz dx``: ``a`` is
    the wave travelling toward the swept layer (-z), ``b`` the one leaving it.
    """
    E0, E1 = dft(values, t, taper)
    k = yee_kappa(cpw, cos)
    b = (E1 - E0 * np.exp(-1j * k)) / (np.exp(1j * k) - np.exp(-1j * k))
    return E0 - b, b


def continuum(cos, target):
    """The PEC-backed continuum CFS-PML round trip at the band centre: ``R ** (g cos)``.

    ``s = kappa + sigma / (alpha - i w)`` (``e^{-iwt}``), so ``|R| = exp(-2 w cos int sigma w
    / (w^2 + alpha^2) dz)``; with the profiles of ``src/wavebit/fdtd/_prepare/cpml.py`` that is ``R **
    (g cos)`` with ``g = (m + 1) int_0^1 f^m / (1 + (a_rel (1 - f))^2) df``, independent of the
    thickness.
    """
    m, a = cpml_module.ORDER, cpml_module.ALPHA_RELATIVE
    f = np.linspace(0.0, 1.0, 20001)
    g = (m + 1) * np.trapezoid(f ** m / (1 + (a * (1 - f)) ** 2), f)
    return target ** (g * cos), g


def reference_task(args):
    cpw, cos, pol, more = args
    ext = extension(cpw) if not more else int(math.ceil(CONTROL_GROWTH * extension(cpw)))
    start = time.time()
    t, v = series(scene(cpw, cos, pol, CELLS[0], TARGETS[0], ext,
                        window_steps(cpw, max(WINDOWS))))
    return (cpw, cos, pol, more), (t, v), time.time() - start


def test_task(args):
    cpw, cos, pol, cells, target, reference = args
    start = time.time()
    t, got = series(scene(cpw, cos, pol, cells, target, 0, window_steps(cpw, max(WINDOWS))))
    t_ref, want = reference
    assert np.array_equal(t, t_ref)
    # The ideal: the reference's f0 phasor with its near-Rayleigh tail tapered away over the
    # longest window (control 3 prints how far that is from converged).
    full = tukey(len(t))
    a_id, _ = split(want, t, cpw, cos, full)
    E_id = dft(want[:1], t, full)[0]
    out = {}
    for W in WINDOWS:
        n = window_steps(cpw, W)
        diff = got[:, :n] - want[:, :n]
        _, b_user = split(got[:, :n], t[:n], cpw, cos)
        user = dict(up_user=abs(b_user) / abs(a_id),
                    total_user=abs(dft(got[:1, :n], t[:n])[0] - E_id) / abs(E_id),
                    ref_user=abs(dft(want[:1, :n], t[:n])[0] - E_id) / abs(E_id))
        a_ref, b_ref = split(want[:, :n], t[:n], cpw, cos)
        a_d, b_d = split(diff, t[:n], cpw, cos)
        taper = tukey(n)
        a_rt, b_rt = split(want[:, :n], t[:n], cpw, cos, taper)
        a_dt, b_dt = split(diff, t[:n], cpw, cos, taper)
        total = abs(dft(diff[:1], t[:n])[0]) / abs(dft(want[:1, :n], t[:n])[0])
        out[W] = dict(up=abs(b_d) / abs(a_ref), echo=abs(a_d) / abs(a_ref), total=total,
                      worst=float(np.max(np.abs(diff[0])) / np.max(np.abs(want[0, :n]))),
                      ref_up=abs(b_ref) / abs(a_ref),
                      up_taper=abs(b_dt) / abs(a_rt), echo_taper=abs(a_dt) / abs(a_rt),
                      ref_up_taper=abs(b_rt) / abs(a_rt), **user)
    return (cpw, cos, pol, cells, target), out, time.time() - start


def truncation_task(args):
    cpw, cos, pol, cells, target = args
    n = window_steps(cpw, DEFAULT_WINDOW)
    _, short = series(scene(cpw, cos, pol, cells, target, 0, n))
    _, long = series(scene(cpw, cos, pol, cells, target, 0, window_steps(cpw, max(WINDOWS))))
    return (cpw, cos, pol), float(np.max(np.abs(short - long[:, :n])))


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def fmt(x):
    return f'{x:8.1e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--json', default='artifacts/studies/fdtd_grazing_return.json')
    args = parser.parse_args()
    b = band()
    print(f'Band(wavelength={WAVELENGTH:g}, df={DF:g}): f0 = 1, spectrum {1 - b.sigmas * DF:.3f}'
          f'..{1 + b.sigmas * DF:.3f} f0 at the 1e-3 cutoff; courant 0.5')
    print(f'cell {BLOCH_CELLS} x 1 x nz, Bloch k at the band centre; '
          f'z+ PML {TOP} cells (default), '
          f'{ABOVE} cells, the sheet, {DISTANCE:g} wavelength, the probes {PROBE} and '
          f'{PROBE + 1} cells in front of the swept z- PML')
    print(f'windows {WINDOWS} periods = steps ' + ', '.join(
        f'{cpw} cpw: {tuple(window_steps(cpw, W) for W in WINDOWS)}'
        for cpw in sorted({a[0] for a in ARMS})))
    print('reference extension (cells): ' + ', '.join(
        f'{cpw} cpw: {extension(cpw)}' for cpw in sorted({a[0] for a in ARMS})))
    print('\nexit angle at f0: cos(theta) continuum -> Yee kz / k at each resolution')
    for cos in COSINES:
        print(f'  cos {cos:<5g} theta {theta_of(cos):6.2f} deg  ' + '  '.join(
            f'{cpw} cpw {yee_kappa(cpw, cos) / (2 * math.pi / cpw):.4f}'
            for cpw in sorted({a[0] for a in ARMS})))
    started = time.time()
    seconds = 0.0
    results, refs, controls, truncations = {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(cpw, cos, pol, more) for cpw, pol, _ in ARMS for cos in COSINES
                for more in (False, True)]
        for key, value, s in pool.map(reference_task, jobs):
            seconds += s
            cpw, cos, pol, more = key
            if more:
                controls[(cpw, cos, pol)] = value
            else:
                refs[(cpw, cos, pol)] = value
        for key, value in refs.items():
            a, b_ = value[1], controls[key][1]
            controls[key] = float(np.max(np.abs(a - b_)) / np.max(np.abs(a)))
        futures = [pool.submit(test_task, (cpw, cos, pol, n, R, refs[(cpw, cos, pol)]))
                   for cpw, pol, cells in ARMS for cos in COSINES for n in cells
                   for R in TARGETS]
        trunc = [pool.submit(truncation_task, (cpw, 0.26, 's', 12, 1e-8))
                 for cpw in sorted({a[0] for a in ARMS})]
        for f in futures:
            key, value, s = f.result()
            results[key] = value
            seconds += s
        for f in trunc:
            key, value = f.result()
            truncations[key] = value

    print(f'\ncontrol 1: the reference with its z- face {CONTROL_GROWTH}x further, over the '
          f'longest window, max |ref - ref_longer| / max |ref|:')
    for (cpw, cos, pol), v in sorted(controls.items()):
        print(f'  {cpw} cpw {pol} cos {cos:<5g}: {v:.1e}')
    print(f'control 2: a run of the {DEFAULT_WINDOW}-period window against the first steps of '
          f'the {max(WINDOWS)}-period run (max |difference|):')
    for key, v in sorted(truncations.items()):
        print(f'  {key}: {v:.1e}')
    print('control 3: the reference against the ideal (its own f0 phasor, tapered over the '
          'longest window). Per exit cos: the reference up-going amplitude over its '
          'incident one, rectangular / tapered (nothing returns there, so this is the band '
          'near-Rayleigh tail leaking into the f0 DFT), and | DFT_ref(T) - ideal | / | ideal | '
          '(the error a perfect layer would still leave at window T):')
    for cpw, pol, cells in ARMS:
        if pol != 's':
            continue
        for W in WINDOWS:
            row = [results[(cpw, cos, pol, cells[0], TARGETS[0])][W] for cos in COSINES]
            print(f'  {cpw} cpw, window {W:4d}: ' + '  '.join(
                f'{r["ref_up"]:.1e}/{r["ref_up_taper"]:.1e}/{r["ref_user"]:.1e}' for r in row))
    print('control 4: p against s (same arm, 20 cpw): max over every entry and window of '
          '| R_p / R_s - 1 |, up-going rectangular and tapered:')
    worst = {'up': 0.0, 'up_taper': 0.0}
    for cos in COSINES:
        for n in CELLS:
            for R in TARGETS:
                for W in WINDOWS:
                    for k in worst:
                        a = results[(20, cos, 'p', n, R)][W][k]
                        b = results[(20, cos, 's', n, R)][W][k]
                        worst[k] = max(worst[k], abs(a / b - 1))
    print('  ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    ratios = [results[(cpw, c, 's', n, R)][W]['echo_taper']
              / results[(cpw, c, 's', n, R)][W]['up_taper']
              for cpw, pol, cells in ARMS if pol == 's' for c in COSINES for n in cells
              for R in TARGETS for W in WINDOWS[1:]]
    print(f'the echo (down-going part of test - ref) over the return, tapered, windows >= '
          f'{WINDOWS[1]}: median {np.median(ratios):.2f}, 10-90% {np.quantile(ratios, 0.1):.2f}'
          f'..{np.quantile(ratios, 0.9):.2f}')

    print('\ncontinuum prediction, PEC-backed CFS-PML at f0: R_c = target ** (g cos), '
          f'g = {continuum(1.0, 1e-8)[1]:.4f}')
    print(f'{"target":>8s} ' + ' '.join(f'{c:8g}' for c in COSINES))
    for R in TARGETS:
        print(f'{R:8.0e} ' + ' '.join(fmt(continuum(c, R)[0]) for c in COSINES))

    for title, key in (
            ('R = |up-going DFT(test - ref)| / |down-going DFT(ref)| at f0, rectangular '
             '(what a plain running DFT reads)', 'up'),
            ('R_taper: the same with a Tukey taper over the second half of the window '
             '(the layer\'s own f0 return, leakage suppressed)', 'up_taper'),
            ('R_user: |up-going DFT(test)| at window T over the ideal incident, rectangular '
             '(what a plain running DFT of the run itself reads as a returned wave)',
             'up_user'),
            ('total_user: |DFT(test)(T) - ideal| / |ideal| of the probe itself, rectangular '
             '(return + echo + window, what a field monitor there is off by)', 'total_user'),
            ('worst in time, max |test - ref| / max |ref| at the probe', 'worst')):
        print(f'\n=== {title}')
        for cpw, pol, cells in ARMS:
            if pol != 's':
                continue
            for W in WINDOWS:
                print(f'\n{cpw} cpw, window {W} periods '
                      f'({window_steps(cpw, W)} steps); columns: exit cos theta')
                print(f'{"cells":>5s} {"target":>7s} {"thick":>6s} '
                      + ' '.join(f'{c:8g}' for c in COSINES))
                for n in cells:
                    for R in TARGETS:
                        print(f'{n:5d} {R:7.0e} {n / cpw:6.2f} ' + ' '.join(
                            fmt(results[(cpw, c, pol, n, R)][W][key]) for c in COSINES))

    for key in ('up', 'up_taper'):
        print(f'\n=== fixed physical thickness, s, {key} at f0 over resolution '
              f'(20 / 40 / 80 cpw); columns: exit cos theta {COSINES}')
        for thick in (0.6, 1.2, 1.8):
            for W in WINDOWS:
                for R in TARGETS:
                    row = []
                    for cos in COSINES:
                        vals = []
                        for cpw in (20, 40, 80):
                            n = int(round(thick * cpw))
                            k = (cpw, cos, 's', n, R)
                            vals.append(results[k][W][key] if k in results else float('nan'))
                        row.append('/'.join(f'{v:.1e}' for v in vals))
                    print(f'{thick:3.1f} wl {R:.0e} W{W:<4d}: ' + ' '.join(row))

    if args.json:
        blob = {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]:g}': {str(W): v for W, v in val.items()}
                for k, val in results.items()}
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as fh:
            json.dump(dict(results=blob, controls={str(k): v for k, v in controls.items()},
                           truncations={str(k): v for k, v in truncations.items()}), fh)
    print(f'\n{len(results)} test runs, {seconds:.0f} s of worker time, '
          f'{time.time() - started:.0f} s wall')


if __name__ == '__main__':
    main()
