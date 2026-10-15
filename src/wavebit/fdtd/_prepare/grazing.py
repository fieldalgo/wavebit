"""The grazing-order rule: which orders a PML face may be asked to absorb, measured first.

**What was measured** (``studies/fdtd/grazing_return/fdtd_grazing_return.py``, 2026-09-19, its log in
``studies/fdtd/grazing_return/records/fdtd_grazing_return.log``): a ``PlaneWave`` in a Bloch cell
leaving toward a PEC-backed PML at exit ``cos theta`` 0.15 .. 0.7, doubling method, over the
layer's cells, its ``reflection=`` target and the run length. The PEC backing does **not** remove the grazing
return: it makes the return exactly the round trip the grading is designed for. In the continuum
a PEC-backed graded CFS-PML reflects

    R_c = reflection ** (g cos theta),   g = (m + 1) int_0^1 f^m / (1 + (a (1 - f))^2) df

(``m = ORDER = 4``, ``a = ALPHA_RELATIVE = 1.5`` of ``src/wavebit/fdtd/_prepare/cpml.py``: ``g =
0.919`` at the face's design wavelength), independent of the thickness. The discrete return
converges to ``R_c`` as the grid refines at a fixed physical thickness, roughly first order but **upward and from below** -- a finer grid is
worse on a grazing order, never better (at 1.2 wavelengths, 1e-8, ``cos 0.26``: 6.9e-3 / 9.7e-3 /
1.1e-2 at 20 / 40 / 80 cells per wavelength against ``R_c`` 1.2e-2). The measured return is the
sum of that round trip, which only the target moves, and a discretisation reflection, which
only cells move (12 cells at 1e-16: 2.6e-4 at ``cos 0.7``, against 2.2e-5 for the default
layer) -- which is why "neither knob alone" was the legacy engine's experience.

**``g`` depends on the wavelength.** The CFS shift is fixed at the face's design wavelength
(``alpha_max = ALPHA_RELATIVE 2 pi / wavelength`` in ``cpml.py``), so at another wavelength the
round trip is ``exp(-2 cos theta int sigma / (1 + (alpha / omega)^2) dz)`` with ``alpha / omega
= ALPHA_RELATIVE (lambda / lambda_design) (1 - f)``: the same law with ``a`` scaled by ``lambda /
lambda_design``. ``g`` is 0.931 / 0.919 / 0.906 / 0.893 at 0.9 / 1.0 / 1.1 / 1.2 of the design
wavelength, and ``cos_min(1e-8)`` 0.403 / 0.408 / 0.414 / 0.420. Every row is judged with the
``g`` of its own wavelength against its face's design wavelength (``plan.pml_wavelengths``; the
column's padded face is matched to the band centre). The study measured the band centre only,
so the other wavelengths rest on the continuum law, the part of the measurement that holds at
every setting measured (the grid approaches ``R_c`` from below). The figures quoted below
(0.41, 0.27, 0.204) are the design wavelength's.

**The rule**, with the amplitude tolerance ``tau = 1e-3``: a face whose target is ``R`` reaches
down to ``cos_min(R) = ln(tau) / (g ln R)`` -- 0.41 for the default ``PML(cells=12,
reflection=1e-8)``. So, for every propagating order that leaves the cell through an absorbing
face along the propagation axis, at every wavelength a ``Diffraction`` monitor asks for, and for
the ``PlaneWave``'s own incidence (its column returns and re-injects at the same size, on the
column's two faces):

* ``cos theta >= cos_min(1e-8) = 0.41`` and ``>= cos_min(R)`` of the face: allowed.
* below 0.41: allowed only if that face declares ``PML(cells >= 24, reflection=R)`` with ``cos
  theta >= cos_min(R)`` **and** the run is ``Run(periods >= 160)`` (of the PlaneWave's band
  centre). 1e-12 reaches 0.27 (measured at ``cos 0.3``: 1.5e-4 / 2.9e-4 / 3.8e-4 at 20 / 40 / 80
  cpw against ``R_c`` 4.9e-4); 1e-16 reaches 0.204 (at ``cos 0.26``: 2.1e-5 / 6.4e-5 / 1.0e-4
  against ``R_c`` 1.5e-4). The 24 cells keep the non-grazing orders from getting worse than
  under the default layer (1e-8 or less at ``cos 0.7``); the 160 periods are needed because at
  40 the band's window error swamps every layer (5e-3 to 1.2e-2 at ``cos 0.26``). Cost: 12 more
  cells on that face and 4x the steps of a 40-period run.
* below ``cos_min(1e-16) = 0.204``: refused outright. No measured setting meets ``tau`` there
  (at ``cos 0.15``, 1e-16 returns 5.2e-3 at 80 cpw against ``R_c`` 6.2e-3); targets past 1e-16
  were not measured.

What is left when an order is allowed: its return up to ``R_c <= 1e-3`` (the grid gives less,
approaching from below), 1e-8 or less of discretisation on the other orders, anywhere from 0
to 2x the return at a monitor between a PlaneWave sheet and the layer (the echo), and a window
term falling as ``T^-1.5``, at or below 3e-4 at ``cos >= 0.26`` after 160 periods for this band.

Only ``PML`` faces are judged: an ``Absorber`` is matched at normal incidence only and absorbs
hopelessly at grazing whatever it is told (``boundaries.Absorber``), and a PEC, PMC or Symmetry
face along the axis is a mirror the scene asked for. An order is judged in the medium of the
absorbing layer it enters (each material filling that layer; the lowest index where the medium
is anisotropic), with ``k_par`` from the cell's faces (a Bloch ``k``, ``0`` on a Periodic axis,
an ``Axis1D`` ``k`` with the single order 0 along it). A medium or a Bloch phase that is traced
cannot be counted; that face is skipped and ``survey`` says so.

**A stabilized face is judged by what it returns.** The opt-in ``PML(stabilize=rho)``
(``cpml.py::stabilizing_rate``) reflects at every oblique angle, first order in ``rho``, so the
pure stretch's ``R_c`` no longer bounds it. So on a face with ``stabilize > 0`` every row whose
pure verdict is ``'ok'`` or ``'needs'`` also carries that face's **exact discrete return on
that order**, ``layer_return.py``: the kernel's own update equations along the face's axis --
the CPML recursion with ``b``, ``a``, ``kappa`` per cell as lowered, the loss's ``keep``/``lift``
on the normal ``E`` and ``H``, the PEC wall where ``src/wavebit/fdtd/_kernel/faces.py`` puts it -- in
the medium the order was counted in, the stretch matched as ``params`` matches it, at the row's
wavelength and the face's design wavelength, with the order's transverse ``k`` as the grid
differences it. It equals the kernel's driven steady state to 4.7e-14 of the incident wave
(``tests/fdtd/test_fd_layer_return.py``) and a column run's measured return to four digits (12
cells: 1.548e-3 at ``cos 0.6``, 7.465e-4 at 0.7) and to 1.5% at worst (the runs below). The
row carries ``'return'`` (``|r|``, the face with the loss), ``'return_pure'`` (``|r_pure|``, the
same face at ``rho = 0``) and ``'return_loss'`` (``|r - r_pure|``, the loss's own part,
reported, not judged) of the judged polarisation furthest past its bound (``'polarisation'``):
TE or TM for a diffracted order, which may carry either, the PlaneWave's own for its incidence
(``'s'`` is TE, ``'p'`` is TM). A ``'weak'`` or ``'floor'`` row is refused already and is not
modelled.

**The bound is ``|r| <= max(tau, |r_pure|)``**, per polarisation; above it the row is
``'lossy'``, a refusal ranked between ``'needs'`` and ``'weak'``. The loss is blamed only where
it takes the order above both what ``tau`` allows and what the pure stretch the rules above
already accept returns. The pure-stretch verdicts are judged first and unchanged (kept as
``'pure_verdict'``), so a stabilized face is never accepted where its pure stretch would not
be, and a pure-stretch face's rows carry no model and are what they always were. The two
bounds this replaced were each wrong, measured on the column (doubling probe, 20 cells per
wavelength unless said; each total within 0.2% of the model, 1.5% at ``cos 0.28``):

* ``|r| <= tau`` refused a stabilized face where its pure twin is accepted: at normal
  incidence, where the loss is inert (``r = r_pure`` exactly), a 12-cell face at 9 / 8.4 cells
  per wavelength returns 1.54e-3 / 2.14e-3 with the loss and without it. The rules above
  deliberately do not hold the grid's own return to ``tau`` (they hold a face to ``R_c``, which
  the grid approaches from below), so the loss may not be blamed for it: at normal incidence
  this bound is the pure verdict exactly.
* ``|r - r_pure| <= tau`` (the loss's own part) allowed ``|r|`` up to ``|r_pure| + tau`` and
  refused where the loss part-cancels the stretch. It accepted ``PML(32, 1e-8,
  stabilize=5e-3)`` at ``cos 0.43`` (s) returning a measured 1.278e-3 and ``PML(32, 1e-12,
  stabilize=5e-4)`` at 0.28 (p) returning 1.390e-3 -- both at or above the stabilize
  ``MEASURED_STABILIZE`` asks for, so nothing warned -- and refused ``PML(12, 1e-16,
  stabilize=3e-3)`` at 0.69 (p) returning 7.96e-4 (the loss's part 1.03e-3 on a pure 2.4e-4).
  On one glass face (``eps 2.25``, 40 cells per wavelength, ``PML(24, 1e-8, stabilize=1e-3)``,
  ``cos 0.43``) it refused s at 5.93e-4 and accepted p at 1.455e-3. Over the reviewer's model
  sweep (cells per wavelength 10 / 20 / 40, ``n`` 1 / 1.5, 12-48 cells, targets 1e-8 / 1e-12 /
  1e-16, ``rho`` 5e-4 .. 5e-3, 59940 (order, polarisation) points the pure rule accepts) it
  accepted 1678 whose total is above ``tau`` and refused 100 whose total is at or below it.
  ``max(tau, |r_pure|)`` refuses none at or below ``tau``, and every one it accepts above
  ``tau`` has a pure stretch returning at least as much (1321; the worst 1.05e-2, at normal
  incidence, where the loss adds nothing, on a grid of 10 cells per vacuum wavelength in glass
  that the plan refuses as 6.7 in the medium).

At 3e-3 in vacuum at 20 cells per wavelength the bound is reached at exit ``cos`` 0.6602 (TE)
/ 0.6571 (TM) on the default 12 cells and 0.4717 / 0.4752 on 24, where the rules above allow
down to 0.41 (the loss's own part reached ``tau`` at 0.6586 / 0.6586 and 0.4700 / 0.4752); in
glass (``eps 2.25``, the same grid) at 0.5491 / 0.5554 on 12 and 0.4111 / 0.3832 on 24.

**What a refusal offers passes.** The refusal quotes the three numbers and what passes at that
order: the fewest cells at this ``rho`` (searched to :data:`LOSSY_CELLS_LIMIT`; 16 at ``cos
0.6`` on 12 cells, 27 at 0.45 on 24) and the largest ``rho`` at these cells (1.9e-3 and 2.5e-3
there), saying when that ``rho`` is below what ``cpml.py::MEASURED_STABILIZE`` gives for the
target -- the growth the loss holds back may return. An offer replaces the face on both
faces of its axis when the scene declares one face there (``z=PML(...)``, spelled so) and on
the named face alone otherwise, spelled with every argument the face was given
(``wavelength=``, ``background=``, ``allow_thin=``), and it is made only if the survey itself,
re-run with the offer in place, accepts **every** row those faces govern -- each medium, each
wavelength, the incidence (:func:`lossy_fixes`, :func:`_passes`). The review applied the
offers of 128 random-grating refusals, each judged on the quoted row alone, and 22 of 247
failed, every one on a grating with a substrate: the other face, or the other medium, returned
another order too much, and ``enforce`` quoted the first refused row rather than the worst.
It quotes the worst now (:func:`worst`: a ``'weak'`` or ``'floor'`` row, which only the target
or the geometry moves, before a ``'lossy'`` one) and names every other refused row; when a
face the offer would not change is refused too, no offer can pass the scene and the refusal
says so instead of offering. That other face is always the axis's own second face, declared
separately (``z=(low, high)``), so the refusal says what the user must change -- both faces
at once, the axis given one face (``z=PML(...)``), which this rule does offer for -- and keeps
the advice that always applies; the sentence it had instead ("that face needs a change of its
own, which its own refusal names once this one is fixed") was circular, since this face is
never offered a fix (the review: 39 of 73 ``'lossy'`` refusals on pair-declared faces, and its
deadlock scene, where four hand attempts at the pair are all refused while the one-face
declaration is offered 37 cells or ``stabilize<=9.2e-4``, both of which pass).
With both, every offer of the review's sweep (its seeds 7 / 8 / 9,
1200 gratings, 727 offers) and of a sweep that also pairs two different z faces (seeds 11 /
12 / 13, 593 offers; 46 refusals that name the other face instead) passes when applied, the
domain grown with the cells -- which the offer now says, since a deeper layer is taken out of
the domain and 100 of 176 offers copied literally onto the domain the scene has are refused
next for what the layer swallows (all 176 pass with the domain grown). An offer is grown in
the medium the scene's own layer holds.
When any order of those faces is ``'needs'`` every offer includes what the pure stretch
needs -- ``cells >= 24``, and ``Run(periods >= 160)`` when the run is shorter -- and on a
``'needs'`` row the tail says the face's target already reaches the order: the review found a
``stabilize``
offered at 12 cells there (still refused, the stretch needing 24) and a ``reflection<=``
printed rounded up (4.695e-10 as 4.7e-10, whose ``cos_min`` 0.350017 turned the row
``'weak'``); a quoted target is now rounded down and checked as printed.

**Every propagating order is judged on a stabilized face**, not only the lowest ``cos`` at each
wavelength as on a pure stretch: the loss's return is not monotone in ``cos`` everywhere.
Swept over ``n`` 1 .. 3.5, 6 .. 48 cells, 8 .. 40 cells per wavelength in the medium, ``rho``
1e-3 .. 3e-2 and targets 1e-8 and 1e-12 (960 sweeps of 161 cosines, the model alone), an
order above the lowest returned up to 1.048x what the lowest does wherever either is above
0.1 ``tau``; under 8 cells per wavelength in the medium it is far from monotone (up to 2e5x).
So no monotonicity is assumed: the orders of one face at one wavelength are marched together
(``layer_return`` over an array of ``K``), and the row quoted is the order furthest past its
bound (``'excess' = |r| / max(tau, |r_pure|)``). The PlaneWave's own incidence is judged on
what its column runs: ``sources.column_scene`` keeps a stabilized scene face, loss and all,
and a padded column face is a default pure stretch.

**What the model does not see.** It is isotropic: a diagonal layer medium is judged in its
lowest index, like the pure rule. On a uniaxial layer (the kernel's driven return, 12 cells,
3e-3, an x face, ``cos`` 0.5 .. 0.7 in the lowest index) that never fell below the kernel's
worse polarisation by more than 1% and overstated it up to 1.9x (``eps = (2.25, 4, 4)``:
1.46e-3 against 7.63e-4 at ``cos 0.5``), so such a face can be refused where it absorbs well
enough. A ``Conductor`` or ``Dispersive`` never reaches a ``PML`` face (``checks.py`` refuses
it). It models an order the grid propagates only, and the plan never asks it for another: an
order the grid makes evanescent is at ``cos`` 0.130 or less at the coarsest grid the plan
takes, a ``'floor'`` row (``layer_return.py``), and a wavelength this rule reads sampled below
8 cells per wavelength in any material -- where an order may be past the grid's Nyquist along
the axis, which the review found as a bare 'math domain error' and a NaN verdict -- is refused
before the rule runs (``checks.check_wavelength_resolution``: a ``Diffraction`` monitor's
wavelengths, and the sources' band, which the incidence row is judged at and which reached
``layer_return``'s own refusal on a column with no ``Diffraction`` monitor at all).

**Cost** (NumPy, Windows, ``survey`` best of 5 x 20; the crossed gratings best of 3): 0.43 ms
against 0.020 ms on the one-order column of ``tests/fdtd/test_fd_grazing.py`` with the loss on both
faces, 4.8 ms against 0.105 ms on a lamellar grating's three wavelengths and every order; on a
crossed grating (``theta 10``, ``phi 20``) 11 ms at period 4 and five wavelengths and 40 ms at
period 8 and nine (1903 vacuum orders), where judging order by order took 92 ms and 645 ms.
``fd.run`` surveys once, with nothing cached: ``fd.plan`` hands the rows its refusal judged to
the result (``plan(scene, _grazing=)``); they are not a plan field, since the survey reads the
materials, which the skeleton does not hold (two scenes with one skeleton can survey
differently). A refusal with its offers, each judged by a survey, takes 14 / 26 ms on the
column (``cos 0.6``; a ``'needs'`` row), 70 ms on the lamellar grating and 0.57 s on the
substrate grating of ``tests/fdtd/test_fd_grazing.py`` (38 cells, bisections on the whole survey).
"""
import functools
import math

import numpy as np

from wavebit.fdtd._legacy_boundaries import AXES
from wavebit.fdtd._legacy_boundaries import Axis1D
from wavebit.fdtd._legacy_boundaries import Bloch
from wavebit.fdtd._legacy_boundaries import PML
from wavebit.fdtd._legacy_boundaries import Periodic
from wavebit.materials.time_domain import as_material
from wavebit.materials.time_domain import concrete
from wavebit.fdtd._prepare.cpml import ALPHA_RELATIVE
from wavebit.fdtd._prepare.cpml import ORDER
from wavebit.fdtd._prepare.cpml import face_eps_r
from wavebit.fdtd._prepare.cpml import spell
from wavebit.fdtd._prepare.cpml import sufficient_stabilize
from wavebit.fdtd._prepare.layer_return import grid_k
from wavebit.fdtd._prepare.layer_return import layer_return

__all__ = ['GRAZING_TOLERANCE', 'GRAZING_EXPONENT', 'exponent', 'GRAZING_CELLS', 'GRAZING_PERIODS',
           'GRAZING_TABLE', 'LOSSY_CELLS_LIMIT', 'POLARISATIONS', 'cos_min', 'continuum_return',
           'order_K', 'face_return', 'judge_returns', 'refuses', 'lossy_fixes', 'worst',
           'survey', 'enforce']

#: The amplitude a PML return may reach on an order that is reported.
GRAZING_TOLERANCE = 1e-3


def _exponent(m=ORDER, a=ALPHA_RELATIVE, samples=200001):
    """``g = (m + 1) int_0^1 f^m / (1 + (a (1 - f))^2) df`` (the module docstring), by Simpson."""
    f = np.linspace(0.0, 1.0, samples)
    y = f ** m / (1.0 + (a * (1.0 - f)) ** 2)
    h = 1.0 / (samples - 1)
    return float((m + 1) * h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


#: ``g`` of ``R_c = reflection ** (g cos theta)``, from the grading's own constants: 0.919.
GRAZING_EXPONENT = _exponent()


@functools.lru_cache(maxsize=512)
def exponent(ratio=1.0):
    """``g`` at ``ratio = wavelength / design wavelength``: the grading's ``a`` scaled by it."""
    if ratio == 1.0:
        return GRAZING_EXPONENT
    return _exponent(a=ALPHA_RELATIVE * ratio)

#: The default target's reach, and the measured floor (the lowest target measured, 1e-16).
DEFAULT_REFLECTION = 1e-8
FLOOR_REFLECTION = 1e-16

#: What a face must declare, and the run must last, below the default face's reach.
GRAZING_CELLS = 24
GRAZING_PERIODS = 160

#: The measurement the refusal quotes: ``R_taper`` at the band centre, 640 periods, s, the
#: layer's own return (``studies/fdtd/grazing_return/records/fdtd_grazing_return.log``), and
#: the continuum law.
GRAZING_TABLE = """\
  measured PML return at f0 (640 periods, s), columns = exit cos theta:
                        0.15     0.2      0.26     0.3      0.4      0.5      0.7
  20 cpw 12 cells 1e-8  3.3e-02  1.4e-02  4.8e-03  2.3e-03  3.9e-04  7.6e-05  2.2e-05
  20 cpw 24 cells 1e-12 7.3e-03  2.0e-03  4.3e-04  1.5e-04  1.0e-05  7.2e-07  1.0e-08
  20 cpw 24 cells 1e-16 1.1e-03  1.8e-04  2.1e-05  5.3e-06  1.6e-07  5.8e-09  9.3e-09
  80 cpw 1.2 wl   1e-8  7.6e-02  3.2e-02  1.1e-02  5.6e-03  1.0e-03  1.8e-04  5.7e-06
  continuum R_c   1e-8  7.9e-02  3.4e-02  1.2e-02  6.2e-03  1.1e-03  2.1e-04  7.2e-06
  window error alone, 40 / 160 periods (20 cpw):
                        3.5e-02  2.3e-02  1.5e-02  1.1e-02  5.4e-03  2.8e-03  3.0e-04
                        4.3e-03  2.6e-03  1.7e-03  1.2e-03  6.4e-04  2.9e-04  3.1e-05
  R_c = reflection**(0.919 cos theta); cos_min(R) = ln(1e-3) / (0.919 ln R):
  0.41 at 1e-8 (the default), 0.27 at 1e-12, 0.204 at 1e-16 (the floor)
  (g = 0.919 at the face's design wavelength; 0.931 / 0.906 at 0.9x / 1.1x of it)"""


def cos_min(reflection, tolerance=GRAZING_TOLERANCE, g=GRAZING_EXPONENT):
    """The lowest exit ``cos theta`` a face with target ``reflection`` returns ``<= tolerance`` at."""
    return math.log(tolerance) / (g * math.log(reflection))


def continuum_return(reflection, cos, g=GRAZING_EXPONENT):
    """``R_c = reflection ** (g cos theta)``: a PEC-backed graded CFS-PML's round trip."""
    return reflection ** (g * cos)


def reflection_for(cos, tolerance=GRAZING_TOLERANCE, g=GRAZING_EXPONENT):
    """The largest target whose ``cos_min`` is ``cos``: ``tolerance ** (1 / (g cos))``."""
    return tolerance ** (1.0 / (g * cos))


# ---------------------------------------------------------------------------------------------
# What the scene asks the faces to absorb
# ---------------------------------------------------------------------------------------------

def _transverse_k(scene, grid, a):
    """``(k, period, single)`` along transverse axis ``a``; ``None`` if not a periodic axis."""
    face = scene.boundaries.face(AXES[a] + '-')
    if isinstance(face, Periodic):
        return 0.0, grid.size[a], False
    if isinstance(face, Axis1D):
        k = concrete(face.k)
        return (None if k is None else k), None, True
    if isinstance(face, Bloch):
        if face.k is not None:
            k = concrete(face.k)
        else:
            phase = concrete(face.phase)
            k = None if phase is None else phase / grid.size[a]
        return k, grid.size[a], False
    return None


#: How far the refusal of a ``'lossy'`` face searches for the cell count that would pass.
LOSSY_CELLS_LIMIT = 96


def _stabilize(face):
    """The face's stabilizing loss ``rho`` as the plan lowers it: 0 for the pure stretch."""
    from wavebit.fdtd._prepare.cpml import DEFAULT_STABILIZE
    return float(DEFAULT_STABILIZE if face.stabilize is None else face.stabilize)


#: The polarisations a row is judged in: TE (E transverse to ``k_par``) and TM, by index into
#: :func:`~wavebit.fdtd._prepare.layer_return.layer_return`'s pair. A diffracted order carries
#: either; the PlaneWave's own incidence carries its own (``'s'`` is TE, ``'p'`` is TM).
POLARISATIONS = ('TE', 'TM')


def order_K(plan, k_par, axis1d):
    """The order's transverse wavenumber as the kernel's differences see it (``grid_k``)."""
    return math.sqrt(sum(grid_k(k, plan.dx, single) ** 2 for k, single in zip(k_par, axis1d)))


def face_return(scene, plan, name, face, n, wavelength, design, K):
    """``(r_TE, r_TM)`` this kernel's face ``name`` returns on the orders ``K``, or ``None``.

    The exact discrete return of ``src/wavebit/fdtd/_prepare/layer_return.py``, at the face's own
    cells, target, loss and design wavelength, in the isotropic medium of index ``n`` the
    order was counted in, with the order's transverse wavenumber ``K`` as the grid's
    differences see it (:func:`order_K`; a 1-D array judges every order of one face at one
    wavelength in one march). ``None`` when the medium the stretch is matched to is traced.
    """
    matched = face.background if face.background is not None else as_material(
        scene.background, 'Scene(background=)')
    eps_r = face_eps_r(matched, 'xyz'.index(name[0]))
    if eps_r is None:
        return None
    return layer_return(face.cells, plan.dx, plan.dt, design, eps_r, name[1], face.reflection,
                        _stabilize(face), wavelength, K, eps=n * n)


def judge_returns(r, r_pure, pols):
    """Per order, what the rule reads off a stabilized face: ``[{...}, ...]``.

    ``r`` and ``r_pure`` are :func:`face_return` with the loss and with ``stabilize = 0`` (the
    face's pure stretch, the same cells), ``pols`` the polarisations judged (indices into
    :data:`POLARISATIONS`). In each, the order's total return ``|r|`` is allowed up to
    ``max(tau, |r_pure|)`` -- the module docstring -- and the fields are those of the judged
    polarisation furthest from that bound: ``'polarisation'``, ``'return'`` (``|r|``),
    ``'return_pure'`` (``|r_pure|``), ``'return_loss'`` (``|r - r_pure|``, the loss's own
    part, reported and not judged), ``'allowed'`` and ``'excess' = |r| / allowed``, above 1
    on a refused order.
    """
    r = np.asarray(r).reshape(2, -1)[list(pols)]
    r0 = np.asarray(r_pure).reshape(2, -1)[list(pols)]
    total, pure = np.abs(r), np.abs(r0)
    allowed = np.maximum(GRAZING_TOLERANCE, pure)
    excess = total / allowed
    pick = np.argmax(excess, axis=0)
    out = []
    for i, p in enumerate(pick.tolist()):
        out.append({'polarisation': POLARISATIONS[pols[p]], 'return': float(total[p, i]),
                    'return_pure': float(pure[p, i]),
                    'return_loss': float(abs(r[p, i] - r0[p, i])),
                    'allowed': float(allowed[p, i]), 'excess': float(excess[p, i])})
    return out


def refuses(row):
    """Whether a stabilized row is ``'lossy'``: ``|r| > max(tau, |r_pure|)`` in its judged
    polarisation (the module docstring)."""
    return row['return'] > max(GRAZING_TOLERANCE, row['return_pure'])


def _pure_verdict(c, reflection, cells, g, periods):
    """The pure-stretch verdict of an order at exit ``cos c`` (the rules of the docstring)."""
    if c < cos_min(FLOOR_REFLECTION, g=g):
        return 'floor'
    if c < cos_min(reflection, g=g):
        return 'weak'
    if c < cos_min(DEFAULT_REFLECTION, g=g) and (
            cells < GRAZING_CELLS or periods is None
            or periods < GRAZING_PERIODS * (1.0 - 1e-9)):
        return 'needs'
    return 'ok'


def accepted_cos(reflection, cells, g, periods):
    """The lowest exit ``cos`` :func:`_pure_verdict` calls ``'ok'``, the only verdict
    :func:`enforce` passes: above the floor's reach and the face's own, and above the default
    target's reach unless the face declares ``GRAZING_CELLS`` and the run lasts
    ``GRAZING_PERIODS``."""
    lowest = max(cos_min(FLOOR_REFLECTION, g=g), cos_min(reflection, g=g))
    if cells < GRAZING_CELLS or periods is None or periods < GRAZING_PERIODS * (1.0 - 1e-9):
        lowest = max(lowest, cos_min(DEFAULT_REFLECTION, g=g))
    return lowest


def refused_bands(scene, plan, n_max):
    """``{(face name, medium label): ((low, high), ...)}``: what the orders' rows would refuse.

    For each medium filling a PML layer that :func:`_survey` judges orders in -- one
    PlaneWave, a Diffraction monitor, a concrete Bloch phase, the faces along the wave's axis
    -- the open intervals of ``n^2`` (the medium's ``min_c eps_c mu_c``) in which some order
    propagates at an exit cos the face's pure stretch does not accept (:func:`accepted_cos`),
    at every wavelength a Diffraction monitor asks for: an order of transverse wavenumber
    ``K`` propagates for ``n^2 k0^2 > K^2`` and exits at ``cos^2 = 1 - K^2 / (n^2 k0^2)``, so it
    is refused for ``K^2 / k0^2 < n^2 < K^2 / (k0^2 (1 - c^2))``. Orders are counted at
    ``n_max``; a larger index is the caller's to refuse. A stabilized face's rows are judged
    on the face's exact return as well (:func:`_judge`), which these bands do not hold; the
    caller keeps a traced medium out of such a layer. ``{}`` where the survey judges no
    orders.
    """
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.monitors import plane_waves
    waves = plane_waves(scene)
    diffraction = [m for m in scene.monitors if isinstance(m, Diffraction)]
    if len(waves) != 1 or not diffraction:
        return {}
    pw = waves[0]
    P = pw.axis
    kpar = [_transverse_k(scene, plan.grid, a) for a in range(3) if a != P]
    if any(k is None or k[0] is None for k in kpar):
        return {}
    wanted = sorted({w for m in diffraction for w in m.wavelengths})
    periods = _periods(scene, plan, pw)
    design = dict(plan.pml_wavelengths)
    out = {}
    for side in ('-', '+'):
        name = AXES[P] + side
        face = scene.boundaries.face(name)
        if not isinstance(face, PML):
            continue
        bands = []
        for w in wanted:
            c = accepted_cos(face.reflection, face.cells, exponent(w / design[name]), periods)
            k0 = 2.0 * math.pi / w
            for _, _, (ku, kv) in _orders(kpar, n_max, w):
                low = (ku * ku + kv * kv) / (k0 * k0)
                if low > 0.0:
                    bands.append((low, low / (1.0 - c * c)))
        for label, _ in _layer_materials(scene, plan.grid, P, side, face.cells):
            out[(name, label)] = tuple(bands)
    return out


def _judge(scene, plan, rows, face, face_wavelength, n, k_pars, axis1d, pols, periods):
    """``rows`` (one wavelength, one face) with their verdicts, judged with the ``g`` of that
    wavelength against the face's design wavelength.

    Every row gets its pure-stretch verdict first. On a face with ``stabilize > 0`` the rows
    whose pure verdict is ``'ok'`` or ``'needs'`` are then judged on the face's exact return
    (:func:`face_return`, one march for all of them, with the loss and without) and are
    ``'lossy'`` where :func:`refuses` says so. A ``'weak'`` or ``'floor'`` row is refused
    already, and the model is not run for it. The same function judges a refusal's offers
    (:func:`lossy_fixes`), with the offered face and run in place of the scene's.
    """
    if not rows:
        return rows
    wavelength = rows[0]['wavelength']
    g = exponent(wavelength / face_wavelength)
    R = face.reflection
    for row in rows:
        c = row['cos']
        row.update(cells=face.cells, reflection=R, periods=periods, g=g,
                   design_wavelength=face_wavelength, R_c=continuum_return(R, c, g=g),
                   cos_min=cos_min(R, g=g))
        row['verdict'] = _pure_verdict(c, R, face.cells, g, periods)
    rho = _stabilize(face)
    if rho <= 0.0:
        return rows
    for row, k_par in zip(rows, k_pars):
        row.update(stabilize=rho, n=n, k_par=tuple(k_par), axis1d=tuple(axis1d),
                   K=order_K(plan, k_par, axis1d),
                   polarisations=tuple(POLARISATIONS[p] for p in pols),
                   pure_verdict=row['verdict'])
    judged = [row for row in rows if row['verdict'] in ('ok', 'needs')]
    if not judged:
        return rows
    # One order (a column's incidence, a refusal's re-judging) takes the scalar march, which
    # is ~3x faster than an array of one; more take one array march.
    K = judged[0]['K'] if len(judged) == 1 else np.array([row['K'] for row in judged])
    name = rows[0]['face']
    lossy = face_return(scene, plan, name, face, n, wavelength, face_wavelength, K)
    if lossy is None:
        return rows
    pure = face_return(scene, plan, name, face.replace(stabilize=0.0), n, wavelength,
                       face_wavelength, K)
    for row, fields in zip(judged, judge_returns(lossy, pure, pols)):
        row.update(fields)
        if refuses(row):
            row['verdict'] = 'lossy'
    return rows


def _layer_media(scene, grid, P, side, cells):
    """``[(label, index or None), ...]``: the materials filling the absorbing layer's cells."""
    return [(label, m.index_bound()) for label, m in _layer_materials(scene, grid, P, side,
                                                                       cells)]


def _layer_materials(scene, grid, P, side, cells):
    """``[(label, material), ...]``: what :func:`_layer_media` reads the index of."""
    k = cells - 1 if side == '-' else grid.n[P] - cells
    background = as_material(scene.background, 'Scene(background=)')
    from wavebit.fdtd._post import layer_share
    owners, full = {}, False
    for index, s in enumerate(scene.structures):
        box = s.shape.support(grid)                    # anything that may reach this layer
        if box[P][0] <= k < box[P][1]:
            share = layer_share(s.shape, grid, P, k)
            if share != 'absent':
                owners[index] = s.material
                full = full or share == 'full'
    # The background is one of the media unless some structure fills the whole cross-section
    # of the layer -- asked of each shape's own fill (``_post.layer_share``), not of a box.
    media = [('the background', background)] if not full else []
    media += [(f'structures[{i}]', m) for i, m in owners.items()]
    return media


def _periods(scene, plan, pw):
    wavelength = pw.band.concrete_wavelength
    return None if wavelength is None else plan.steps * plan.dt / wavelength


def _orders(kpar, n, wavelength):
    """``[((m_u, m_v), cos, (k_u, k_v)), ...]``: every order propagating in index ``n``."""
    k0 = n * 2.0 * math.pi / wavelength
    axes = []
    for k, L, single in kpar:
        if single or L is None:
            axes.append([(0, k)])
            continue
        g = 2.0 * math.pi / L
        m_max = int(math.ceil((k0 + abs(k)) / g)) + 1
        axes.append([(m, k + m * g) for m in range(-m_max, m_max + 1)])
    out = []
    for mu, ku in axes[0]:
        for mv, kv in axes[1]:
            s2 = (ku * ku + kv * kv) / (k0 * k0)
            if s2 < 1.0:
                out.append(((mu, mv), math.sqrt(1.0 - s2), (ku, kv)))
    return out


#: The verdicts, from accepted to furthest from it; :func:`enforce` refuses all but the first.
RANK = {'ok': 0, 'needs': 1, 'lossy': 2, 'weak': 3, 'floor': 4}


def worst(candidates):
    """The candidate furthest from being accepted: the verdict, how far the face's return is
    past what it is allowed, then ``cos/cos_min``.

    ``'excess'`` is on a stabilized face's modelled rows only, so a pure-stretch face's
    ranking is the verdict and then ``cos/cos_min``, as it always was; on a stabilized face
    the row quoted is the order whose return is furthest past ``max(tau, |r_pure|)``, which is
    not always the lowest ``cos``.
    """
    return max(candidates, key=lambda r: (RANK[r['verdict']], r.get('excess', 0.0),
                                          -r['cos'] / r['cos_min']))


def survey(scene, plan):
    """``[row, ...]``: the lowest exit ``cos theta`` each judged face is asked to absorb.

    One row per (PML face, what leaves through it): ``{'face', 'what', 'medium', 'order',
    'wavelength', 'cos', 'R_c', 'cos_min', 'cells', 'reflection', 'periods', 'verdict'}`` where
    ``what`` is ``'orders'`` (a Diffraction scene's propagating orders in that layer's medium)
    or ``'incidence'`` (the PlaneWave's own angle, on its column's faces), and ``verdict`` is
    ``'ok'``, ``'needs'`` (below the default reach, and the face or the run falls short),
    ``'lossy'`` (a face with the stabilizing loss that returns this order above both the
    tolerance and what its pure stretch returns), ``'weak'`` (below the face's own
    ``cos_min``) or ``'floor'`` (below 0.204). A face that cannot be counted (a traced medium,
    band or Bloch phase) has ``verdict='skipped'``. A row on a stabilized face also carries
    ``'stabilize'``, ``'n'``, ``'k_par'``, ``'axis1d'``, ``'K'``, ``'polarisations'`` and
    ``'pure_verdict'``, and, where the pure verdict is ``'ok'`` or ``'needs'``, the fields of
    :func:`judge_returns`; it is judged at every propagating order. A row on a pure-stretch
    face carries none of them and is judged as it always was.

    Nothing is cached: ``fd.plan`` surveys once for its refusal and ``fd.run`` hands those
    very rows to its diagnostics (``plan(scene, _grazing=rows)``), so a run surveys once.
    """
    return _survey(scene, plan)


#: ``_survey(periods=)`` left at this: the run's own length, read off the plan.
_OWN = object()


def _survey(scene, plan, faces=None, periods=_OWN, only=None, wavelengths=None, every=False):
    """:func:`survey`, and what every offer of a refusal is judged by.

    ``faces`` (``{name: PML}``) judges those faces in place of the scene's -- in the medium
    the scene's own face's layer holds, since an offer grows the face in the medium it
    already has -- and ``periods`` the run length in place of the plan's. ``only`` (face
    names) and ``wavelengths`` narrow what is judged, and ``every`` returns every judged row
    instead of the worst of each (face, what, medium).
    """
    from wavebit.fdtd._legacy_monitors import Diffraction
    from wavebit.fdtd._prepare.monitors import plane_waves
    waves = plane_waves(scene)
    if len(waves) != 1:
        return []
    pw = waves[0]
    P = pw.axis
    transverse = [a for a in range(3) if a != P]
    kpar = [_transverse_k(scene, plan.grid, a) for a in transverse]
    if any(k is None for k in kpar):
        return []
    faces = faces or {}
    diffraction = [m for m in scene.monitors if isinstance(m, Diffraction)]
    asked = sorted({w for m in diffraction for w in m.wavelengths})
    wanted = asked if wavelengths is None else [w for w in asked if w in wavelengths]
    if periods is _OWN:
        periods = _periods(scene, plan, pw)
    design = dict(plan.pml_wavelengths)
    axis1d = [k[2] for k in kpar]
    rows = []

    def keep(candidates):
        if every:
            rows.extend(candidates)
        elif candidates:
            rows.append(worst(candidates))

    # The orders leaving through each PML face along P, in that layer's medium, judged at every
    # wavelength a Diffraction monitor asks for (the lowest exit cos at each), the worst kept.
    if diffraction:
        for side in ('-', '+'):
            name = AXES[P] + side
            own = scene.boundaries.face(name)
            if not isinstance(own, PML) or (only is not None and name not in only):
                continue
            face = faces.get(name, own)
            for label, n in _layer_media(scene, plan.grid, P, side, own.cells):
                if n is None or any(k[0] is None for k in kpar):
                    rows.append(dict(face=name, what='orders', medium=label, verdict='skipped',
                                     why='a traced medium or Bloch k cannot be counted'))
                    continue
                candidates = []
                for w in wanted:
                    orders = _orders(kpar, n, w)
                    if not orders:
                        continue
                    # The pure stretch is judged at the lowest exit cos (its round trip falls
                    # monotonically with cos). A stabilized face is judged on every
                    # propagating order: its return is not monotone in cos everywhere (the
                    # module docstring), and each order is its own row.
                    judged = (orders if _stabilize(face) > 0.0
                              else [min(orders, key=lambda oc: oc[1])])
                    candidates += _judge(
                        scene, plan, [dict(face=name, what='orders', medium=label, order=o,
                                           wavelength=w, cos=c) for o, c, _ in judged],
                        face, design[name], n, [k for _, _, k in judged], axis1d, (0, 1),
                        periods)
                keep(candidates)
    # The PlaneWave's own incidence, on its column's two faces (a padded column face is the
    # default PML the column carries past a non-absorbing scene face, matched to the band
    # centre).
    n_bg = as_material(scene.background, 'Scene(background=)').index_bound()
    centre = pw.band.concrete_wavelength
    if n_bg is not None and centre is not None and not any(k[0] is None for k in kpar):
        checked = sorted(set(asked) | {centre})
        if wavelengths is not None:
            checked = [w for w in checked if w in wavelengths]
        angles = []
        for w in checked:
            k0 = n_bg * 2.0 * math.pi / w
            s2 = sum(k[0] ** 2 for k in kpar) / (k0 * k0)
            if s2 < 1.0:
                angles.append((w, math.sqrt(1.0 - s2)))
        if angles:
            from wavebit.fdtd._legacy_boundaries import Absorbing
            for side in ('-', '+'):
                name = AXES[P] + side
                if only is not None and name not in only:
                    continue
                face = scene.boundaries.face(name)
                if not isinstance(face, PML):
                    if isinstance(face, Absorbing):
                        continue
                    face = PML()                     # the column's own padded layer
                    label = 'the background (the column\'s padded PML)'
                    face_wavelength = centre
                else:
                    face = faces.get(name, face)
                    label = 'the background'
                    face_wavelength = design[name]
                # A stabilized scene face is the column's face too (``sources.column_scene``
                # keeps it, loss and all), so the incidence is judged on what the column runs.
                # Its polarisation is known: 's' (E out of the plane of incidence) is TE, 'p'
                # is TM.
                pols = (POLARISATIONS.index('TE' if pw.polarization == 's' else 'TM'),)
                keep([
                    _judge(scene, plan, [dict(face=name, what='incidence', medium=label,
                                              order=(0, 0), wavelength=w, cos=c)],
                           face, face_wavelength, n_bg, [[k[0] for k in kpar]], axis1d, pols,
                           periods)[0]
                    for w, c in angles])
    return rows


def _floor2(x):
    """``x`` rounded down to two significant digits, so a quoted bound is still met."""
    e = math.floor(math.log10(x)) - 1
    return math.floor(x / 10.0 ** e) * 10.0 ** e


def _quoted_target(c, g):
    """The largest two-digit ``reflection=`` whose ``cos_min`` reaches ``c``, as it is printed.

    ``reflection_for(c)`` rounded to ``.1e`` can round **up** (4.695e-10 printed as 4.7e-10,
    whose ``cos_min`` is 0.350017 > 0.35: the quoted target turned the row ``'weak'``), so it
    is rounded down and the printed value itself is checked against the same ``cos_min``.
    """
    q = float(f'{_floor2(reflection_for(c, g=g)):.1e}')
    while cos_min(q, g=g) > c:
        q = float(f'{_floor2(q * (1.0 - 1e-9)):.1e}')
    return q


def _governed(scene, name):
    """The faces an offer for face ``name`` replaces: both faces of its axis when the scene
    declares the same face on both (``z=PML(...)``), else ``name`` alone."""
    low, high = (scene.boundaries.face(name[0] + side) for side in '-+')
    return (name[0] + '-', name[0] + '+') if low == high else (name,)


def _accepted(rows):
    return all(r['verdict'] in ('ok', 'skipped') for r in rows)


def _passes(scene, plan, governed, face, periods, row, full=True):
    """Whether ``face`` on every face of ``governed`` and a run of ``periods`` pass: the
    survey itself, re-run with the offer in place, accepting every row those faces govern --
    every medium, every wavelength, the incidence. ``row``'s own face at ``row``'s wavelength
    is judged first (the same code on fewer orders, a necessary condition), and alone when
    ``full`` is false (the bisection's cheap first pass)."""
    faces = {name: face for name in governed}
    if not _accepted(_survey(scene, plan, faces, periods, only=(row['face'],),
                             wavelengths=(row['wavelength'],))):
        return False
    return not full or _accepted(_survey(scene, plan, faces, periods, only=governed))


def _largest_rho(passes, high):
    """The largest ``stabilize`` in ``[0, high]`` that ``passes``, by bisection, rounded down
    to two digits and judged again as printed (``:.2g``); 0 when nothing above zero passes."""
    low = 0.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if passes(mid):
            low = mid
        else:
            high = mid
    # Judged as printed: _floor2 leaves float residue (1.8000000000000002e-3).
    rho = float(f'{_floor2(low):.2g}') if low > 0.0 else 0.0
    while rho > 0.0 and not passes(rho):
        rho = float(f'{_floor2(rho * (1.0 - 1e-9)):.2g}')
    return rho


def lossy_fixes(row, scene, plan):
    """What passes a ``'lossy'`` row: ``{'periods', 'cells', 'rho', 'rho_cells'}``.

    An offer replaces the row's face on every face it governs (:func:`_governed`: both faces
    of the axis when the scene declares one face there) and is made only if the survey,
    re-run with it in place (:func:`_passes`), accepts every row those faces govern -- both
    media, every wavelength, the incidence -- not only the row quoted: the review applied 247
    offers judged on the quoted row alone to random gratings, and 22 failed, every one on a
    substrate grating whose other face returned another order too much. When any order those
    faces judge has the pure verdict ``'needs'``, every offer carries what the pure stretch
    needs as well: at least :data:`GRAZING_CELLS` cells, and ``periods`` (``GRAZING_PERIODS``
    when the run is shorter, else ``None``). ``cells`` is the fewest cells at this
    ``stabilize`` (searched to :data:`LOSSY_CELLS_LIMIT`), or ``None``. ``rho`` is the largest
    ``stabilize`` at ``rho_cells`` cells (the face's own, or ``GRAZING_CELLS`` where the pure
    stretch needs them), found by bisection on the quoted row, then held to the whole survey
    as printed (and bisected again below it if the survey refuses); 0 when nothing above zero
    passes -- the pure stretch, which passes there: every refused row is ``'lossy'`` or
    ``'needs'`` (:func:`enforce` quotes a ``'weak'`` or ``'floor'`` one first), and at
    ``rho_cells`` and this run the pure verdicts are all ``'ok'`` (judged all the same, and
    not offered if it failed) -- ``None`` when this
    ``stabilize`` already passes at ``rho_cells`` (so ``cells`` is the offer). The loss's
    return is first order in ``rho``, ``|r_pure + rho d|``, convex in ``rho``, and each order's
    bound ``max(tau, |r_pure|)`` does not depend on it, so the ``rho`` that pass form an
    interval from 0 and ``stabilize<=rho`` is the offer.
    """
    governed = _governed(scene, row['face'])
    face = scene.boundaries.face(row['face'])
    periods = row['periods']
    needs = any(r.get('pure_verdict', r['verdict']) == 'needs'
                for r in _survey(scene, plan, only=governed, every=True))
    short = needs and (periods is None or periods < GRAZING_PERIODS * (1.0 - 1e-9))
    run = max(periods or 0.0, GRAZING_PERIODS) if needs else periods

    def passes(f, full=True):
        return _passes(scene, plan, governed, f, run, row, full)
    # No floor at 24 here: below it a 'needs' order's verdict is 'needs', not 'ok', so the
    # search passes it by (a floor here was a mutant no test could tell from the original).
    cells = next((m for m in range(face.cells + 1, LOSSY_CELLS_LIMIT + 1)
                  if passes(face.replace(cells=m))), None)
    rho_cells = max(face.cells, GRAZING_CELLS) if needs else face.cells
    base = face.replace(cells=rho_cells)
    rho = None
    if not passes(base):
        rho = _largest_rho(lambda r: passes(base.replace(stabilize=r), full=False),
                           row['stabilize'])
        if rho > 0.0 and not passes(base.replace(stabilize=rho)):
            rho = _largest_rho(lambda r: passes(base.replace(stabilize=r)), rho)
        if rho == 0.0 and not passes(base.replace(stabilize=None)):
            rho = None                      # argued never to happen (above); not offered then
    return dict(periods=GRAZING_PERIODS if short else None, cells=cells, rho=rho,
                rho_cells=rho_cells)


def _spell_face(face, cells, stabilize):
    """The face as it is written, every argument it was given: ``PML(cells=16,
    reflection=1e-08, stabilize=0.003, wavelength=0.9)`` -- an offer copied from a refusal
    keeps the design wavelength, the match and ``allow_thin`` it was judged with."""
    parts = [f'cells={cells}', f'reflection={face.reflection:g}', stabilize]
    if face.wavelength is not None:
        parts.append(f'wavelength={face.wavelength:g}')
    if face.background is not None:
        parts.append(f'background={face.background!r}')
    if face.allow_thin:
        parts.append('allow_thin=True')
    return 'PML(' + ', '.join(parts) + ')'


def _brief(row):
    """One refused row, in a clause."""
    where = (f"order {row['order']} through {row['face']!r} in {row['medium']}"
             if row['what'] == 'orders' else f"the incidence on {row['face']!r}")
    return (f"{where} at wavelength {row['wavelength']:g} (exit cos theta {row['cos']:.3f}, "
            f"'{row['verdict']}')")


def _refused_too(others):
    """What else the survey refuses: the refusal quotes the worst row, and names the rest."""
    if not others:
        return ''
    return (' Refused too (the row quoted above is the worst): '
            + '; '.join(_brief(r) for r in others) + '.')


def _lossy_message(row, pw, what, scene, plan, others):
    """The refusal of a stabilized face that returns an order above both ``tau`` and what its
    pure stretch returns."""
    c, R, rho = row['cos'], row['reflection'], row['stabilize']
    face = scene.boundaries.face(row['face'])
    governed = _governed(scene, row['face'])
    head = (f"Stabilized face refused: {what} at wavelength {row['wavelength']:g} has exit cos "
            f"theta = {c:.3f}. That face is "
            f"{_spell_face(face, row['cells'], f'stabilize={rho:g}')}, and it sends back "
            f"{row['return']:.2e} of this order "
            f"({row['polarisation']}): above the amplitude tolerance {GRAZING_TOLERANCE:g} and "
            f"above the {row['return_pure']:.2e} the same face returns without the loss, so it "
            f"is the loss that takes the order past what the rule allows. Both are the face's "
            f"exact discrete return, computed for this kernel by "
            f"src/wavebit/fdtd/_prepare/layer_return.py; the loss's own part, |r - r_pure|, is "
            f"{row['return_loss']:.2e}, and the continuum round trip R_c = {row['R_c']:.2e}. "
            f"The loss sits on the field component normal to the face: it is inert at normal "
            f"incidence and sends back more the more oblique the order, first order in "
            f"stabilize. ")
    axis = row['face'][0]
    blocked = [r for r in others if r['face'] not in governed]
    if blocked:
        # An offer changes this face only; a refused row on another face stays refused, so no
        # offer of this face could pass the scene, and none is made. Every row the survey
        # judges is on the propagation axis's two faces, so a blocked row is always the other
        # face of this axis, declared separately (``z=(low, high)``): the two have to move
        # together, which an offer for one face cannot do. Saying only that leaves the user
        # nowhere -- this face is never offered a fix, so "fix that one first" is circular --
        # so the two ways out are named, and the advice that always applies is kept.
        return (head + 'No change to this face alone passes the scene: '
                + '; '.join(_brief(r) for r in blocked)
                + f' is refused too, on the other face of the {axis} axis, which the scene '
                  f'declares separately ({axis}=(low, high)) and an offer for this face leaves '
                  f'as it is. Both faces have to change together, and this rule offers a '
                  f'setting only for the faces one declaration governs: give the axis one face, '
                  f'{axis}=PML(...), and the refusal names what passes on both at once; or set '
                  f'each face yourself until no row here is refused. Or make the order less '
                  f'oblique (the period, the angle or the wavelengths).'
                + _refused_too([r for r in others if r not in blocked]))
    fix = lossy_fixes(row, scene, plan)
    where = f'{row["face"][0]}=' if len(governed) == 2 else f'the {row["face"]!r} face '
    run = (f' and Run(periods>={fix["periods"]})' if fix['periods'] is not None else '')
    fixes = []
    if fix['cells'] is not None:
        fixes.append(f'{where}{_spell_face(face, fix["cells"], "stabilize=%g" % rho)}{run}')
    else:
        fixes.append(f'no layer up to {LOSSY_CELLS_LIMIT} cells at this stabilize')
    if fix['rho'] is not None:
        at = fix['rho_cells']
        if fix['rho'] > 0.0:
            text = f'{where}{_spell_face(face, at, "stabilize<=%.2g" % fix["rho"])}{run}'
        else:
            text = f'{where}{_spell_face(face, at, "stabilize=None")}{run}, the pure stretch'
        need = sufficient_stabilize(R)
        if need is None:
            text += (f' -- and reflection={R:g} is a weaker target than any the loss was '
                     f'measured at (src/wavebit/fdtd/_prepare/cpml.py::MEASURED_STABILIZE)')
        elif fix['rho'] < need * (1.0 - 1e-12):
            text += (f' -- below the {spell(need)} MEASURED_STABILIZE gives for '
                     f'reflection={R:g}, so a bound mode the loss holds may grow again '
                     f'(tests/fdtd/test_fd_pml_stability.py; the decay warning reports it)')
        fixes.append(text)
    tail = ''
    if row['pure_verdict'] == 'needs':
        tail = (f' Below cos theta = {cos_min(DEFAULT_REFLECTION, g=row["g"]):.3f} the pure '
                f'stretch alone needs PML(cells>={GRAZING_CELLS}) and '
                f'Run(periods>={GRAZING_PERIODS}) (this face: {row["cells"]} cells; this run: '
                + ('unknown' if row['periods'] is None else f"{row['periods']:.1f}")
                + f' periods), which every offer above includes; its reflection={R:g} already '
                f'reaches this order (cos_min = {row["cos_min"]:.3f}).')
    scope = ('both faces of the axis, which the scene declares as one face, each offer '
             'judged by this rule on every order, medium and wavelength they take'
             if len(governed) == 2 else
             f'the {row["face"]!r} face, each offer judged by this rule on every order, '
             f'medium and wavelength it takes')
    # Each offer is judged with the domain grown by its extra cells, which is how a layer is
    # deepened without moving the interior; copied literally onto the domain the scene has,
    # a deeper layer passes this rule and is then refused for what it swallows (the review:
    # 100 of 176 offers, every follow-on refusal explicit and naming its own fix).
    deeper = []                             # in the order the offers above are spelled
    for c in (fix['cells'], fix['rho_cells'] if fix['rho'] is not None else None):
        if c is not None and c > row['cells'] and c not in deeper:
            deeper.append(c)
    grown = ''
    if deeper:
        extra = [c - row['cells'] for c in deeper]
        grown = (' A deeper layer is taken out of the domain you gave, not added to it: grow '
                 f'Scene(domain=) along {axis} with the offer you take ('
                 + ' or '.join(f'{e}' for e in extra) + ' more cells on '
                 + ('each of the two faces' if len(governed) == 2
                    else f'the {row["face"]!r} face') + ', '
                 + ' or '.join(f'{e * plan.dx * len(governed):g}' for e in extra)
                 + ' in total), or the layer eats into the interior and what it then reaches '
                   '-- a Diffraction plane, a structure, the domain left too thin for the '
                   'layers themselves -- is refused next.')
    return (head + f'What passes on {scope}, before it is offered: '
            + '; or '.join(fixes) + '.' + grown + ' Or make the order less oblique (the period, '
            'the angle or the wavelengths).' + tail + _refused_too(others))


def _message(row, pw, scene=None, plan=None, others=()):
    c = row['cos']
    g = row['g']
    what = (f"order {row['order']} leaving through {row['face']!r} in {row['medium']}"
            if row['what'] == 'orders' else
            f"the incidence of PlaneWave({pw.name!r}) itself, whose column returns and "
            f"re-injects through {row['face']!r}")
    if row['verdict'] == 'lossy':
        return _lossy_message(row, pw, what, scene, plan, others)
    head = (f"Grazing order refused: {what} at wavelength {row['wavelength']:g} has exit cos "
            f"theta = {c:.3f}. That face is PML(cells={row['cells']}, "
            f"reflection={row['reflection']:g}"
            + (f", stabilize={row['stabilize']:g}" if 'stabilize' in row else '')
            + f"): its continuum round trip on this order is R_c = "
            f"{row['reflection']:g}**({g:.3f}*{c:.3f}) = {row['R_c']:.2e} (g = {g:.3f} at "
            f"{row['wavelength'] / row['design_wavelength']:.3g} of the face's design wavelength "
            f"{row['design_wavelength']:g}), and it reaches down to "
            f"cos_min = {row['cos_min']:.3f} for the amplitude tolerance "
            f"{GRAZING_TOLERANCE:g}. ")
    if row['verdict'] == 'floor':
        fix = (f'Below cos theta = {cos_min(FLOOR_REFLECTION, g=g):.3f} no measured setting meets '
               f'the tolerance (at cos 0.15, reflection=1e-16 returns 5.2e-3 at 80 cells per '
               f'wavelength against R_c 6.2e-3) and targets past 1e-16 were not measured, so '
               f'this is refused outright: change the period, the angle or the wavelengths so '
               f'the order is less grazing, or leaves the window as evanescent.')
    else:
        fix = (f'Below cos theta = {cos_min(DEFAULT_REFLECTION, g=g):.3f} (the default face\'s reach) '
               f'an order is allowed only when that face declares PML(cells>={GRAZING_CELLS}, '
               f'reflection<={_quoted_target(c, g):.1e}) -- the target whose cos_min is this '
               f'order\'s cos theta, rounded down -- and the run is '
               f'Run(periods>={GRAZING_PERIODS}) (this one: '
               + ('unknown' if row['periods'] is None else f"{row['periods']:.1f}")
               + f'). Cost: {GRAZING_CELLS - 12} more cells on that face and 4x the steps of a '
               f'40-period run. What is left then: the return up to R_c <= 1e-3, 1e-8 or less '
               f'on the other orders, up to 2x the return between a sheet and the layer (the '
               f'echo), and a window term at or below 3e-4 after 160 periods.')
    return (head + fix + _refused_too(others) + '\n' + GRAZING_TABLE
            + '\n(studies/fdtd/grazing_return/fdtd_grazing_return.py; src/wavebit/fdtd/_prepare/grazing.py derives the law.)')


def enforce(scene, plan):
    """Refuse the worst row of :func:`survey` that is not ``'ok'`` or ``'skipped'``
    (:func:`worst`: a ``'weak'`` or ``'floor'`` row, which only the geometry or the target
    fixes, before a ``'lossy'`` one), naming every other refused row; else return the rows."""
    from wavebit.fdtd._prepare.monitors import plane_waves
    rows = survey(scene, plan)
    refused = [row for row in rows if row['verdict'] not in ('ok', 'skipped')]
    if refused:
        top = worst(refused)
        raise ValueError(_message(top, plane_waves(scene)[0], scene, plan,
                                  [row for row in refused if row is not top]))
    return rows
