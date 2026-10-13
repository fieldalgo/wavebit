"""C7: a periodic metasurface under an oblique plane wave, every diffraction order against RCWA.

**The question.** Stage 4's use case (a) is a periodic metasurface lit by an oblique plane wave
and read as diffraction orders. Does the public path -- ``fd.plan`` and ``fd.run`` with every
default (the ``'voxel'`` discretisation, ``Structure(junction='split')`` where a pillar stands
flush on its substrate) -- give ``wavebit.rcwa``'s answer for every propagating order's complex
amplitude, co- and cross-polarised, and for its efficiency, and does it converge to it at second
order? The junction campaign (``studies/fdtd/subpixel_junction/fdtd_subpixel_junction.py``) compared the zeroth order of
subwavelength cells at normal incidence; this one asks for the higher orders a metasurface
steers, at oblique and conical incidence.

**Scenes** (:data:`SCENES`; lengths in vacuum wavelengths at the band centre, the cell ``(Px,
Py)``). Each pillar stands flush on a glass substrate that runs through the ``z-`` PML.

* ``M1`` -- an epsilon-4 cylinder (diameter 0.40, height 0.6) on epsilon 2.25 in a 0.85 x 1.00
  cell, theta 35 degrees in the xz plane: 2 reflected and 6 transmitted orders.
* ``M2`` -- a TiO2-like epsilon-5.8 cylinder (diameter 0.32, height 0.6) on epsilon 2.13 in a
  0.6 square cell, theta 30: the transmitted light leaves in its ``(0, 0)`` and ``(-1, 0)``
  orders (the pilot's slightly smaller pillar sent 60% of the incident s power into ``(-1, 0)``
  at 0.95, 21% at 1.0 and 13% at 1.05).
* ``M3`` -- a silicon-like epsilon-12 box (0.30 x 0.20, height 0.3) on epsilon 2.25 in a
  1.30 x 0.50 cell, theta 25, phi 30 (conical): 2 reflected and 4 transmitted orders.

**The same geometry in cells at every rung.** The pillar's axis sits ``o`` cells past the node
line ``floor(cells/2)`` and the substrate plane ``zeta`` of a cell above a node plane; every
height, side and radius is a whole number of cells at every rung, so each face the lowering
reads -- the substrate plane, the pillar's top, a box's sides, a cylinder's tangent lines along
x and y -- sits at the same fraction of a cell at every rung (``layout`` checks it). A face that
moved within the cell from rung to rung would change the error's constant between rungs, which
criteria 3 and 4 read as convergence. ``floor`` rather than ``round`` keeps the axis ``|o|`` or
``|o - 1/2|`` cells from the cell centre at every rung, so the lateral offset the phase
correction removes is itself first order in the cell (the control below relies on it).

**Why these periods and angles.** With the wavevector fixed across the band (``theta_at``), an
order's transverse index at wavelength ``lambda`` is ``lambda q`` -- the band moves every order --
so the cells and angles were searched for ones where every order that propagates anywhere in
0.95 .. 1.05 propagates everywhere in it at exit ``cos theta >= 0.5`` in its medium (the grazing
rule refuses below 0.41 for these 120-period runs -- its lower reach, 0.27 at the 24-cell 1e-12
layer used here, needs 160 periods -- and that layer returns ``1e-12^(0.919 cos)``, under 3e-6
at 0.5), and every other order is at
least 8% past its cutoff at the three wavelengths. The *source's* spectrum is wider than the
three wavelengths: at ``df = 0.06`` it reached the reflected cutoffs near ``f = 1.15 .. 1.17``
(about 2.6 standard deviations) and M2's reflection decayed as a power law, 2.2e-5 of its peak
after 60 periods; at ``df = 0.04`` those cutoffs sit 3.8 to 4.2 deviations out and the same run
decayed to 5.9e-8, its orders unchanged to 1e-5 (the review's measurement, then this study's
pilot). M1 and M3 are off the ``x <-> y`` swap symmetry (rectangular cells; a square cell lit
at phi 45 with equal offsets reads the same transposed, so an axis-order defect would be
invisible -- the first pilot's M1 and M3 were such scenes and were replaced). M2's square cell
with equal offsets and a round pillar is swap-symmetric in its permittivity; only its incidence
breaks the swap. M1 and M2 are also mirror-symmetric in y about the pillar's axis: M2's
cross-polarised amplitudes are zero in RCWA, and M1's ``(m, +-1)`` cross-polarised amplitudes
are equal and opposite. Only M1 propagates orders with ``n != 0``, so only M1 tests the y half of
the phase correction; only M3 has no mirror at all.

**The reference.** ``wavebit.rcwa`` solves the identical geometry with the shape centred on the
cell, both incidences from one solve, at the run's own ``theta_at`` per wavelength, over the
truncation ladder of :data:`SCENES` (half-widths ``(N_u, N_v)``). The cylinders take its shape
path (closed-form Fourier coefficients and the normal-vector-field factorisation, that path's
only rule). The box takes the raster path with Li's rule and the exact series
(``fourier_coefficients='integrated'``) over pixels whose edges carry the box's edges, so the
raster *is* the box: the pilot measured the epsilon-12 box converging there and not on the
shape path (truncation 25 against 30, the half-widths in proportion to the periods: 2.3e-3 on
the shape path, 9.1e-4 on the raster), and the narrow ``y`` side wanting as many orders as the
long one ((26, 15) against (39, 15) 2.5e-4, (39, 10) 1.7e-3), whence the box's equal
half-widths. ``frame`` holds the raster's frame to the shape path's: a box spanning the cell
along one axis has a constant normal, where the normal-vector-field rule *is* Li's, and the two
paths agree to 1e-10 at an offset centre, once along each axis. RCWA centres its shape at the
origin and fd's pillar sits at ``(xc, yc)``: order ``(m, n)``'s amplitude differs by
``exp(-2 pi i (m xc/Px + n yc/Py))`` and its efficiency not at all (checked by ``shift``: RCWA
solved at the offset against the centred solve times that factor). fd's amplitudes are
multiplied by the inverse factor before any comparison. ``u`` below is the ladder's sampled
spread, the precedent's convention, not a proven bound on the top's own error.

**Instruments**, in the order they run. ``register`` (this file's hash and a copy, once, before
any fd row); ``layout`` (the order statuses over the band, the margins above, the face
fractions); ``frame`` and ``shift`` (above, to 1e-10, written to the reference log and required
by the report); ``lowest`` (each scene's lowest rung, both polarisations); ``rcwa`` (the
ladders); ``project`` (the check below, before the finer rungs); ``fd`` (every scene,
polarisation and rung, 120 periods, NumPy below 48 cells per wavelength and JAX from 48);
``long`` (each scene's middle rung, both polarisations, at 180 periods); ``backends`` (each
scene's lowest rung, s, on JAX and Torch against NumPy); ``report`` (the rule below, written to
``REPORT``).

**The registered rule** (written 2026-10-02 before the campaign and amended 2026-10-03, before any
fd row existed, after two adversarial reviews of the draft -- the amendments are listed below).
Every threshold is one this repository registered before: the junction campaign's consistency
band, Richardson pair and decay line, the design's second-order line 1.8. A *cell* is one number
compared: a scene, an incident polarisation, a wavelength, a side, a propagating order and one
of its two outgoing amplitudes (``r_s``, ``r_p`` / ``t_s``, ``t_p``, complex, phase-corrected as
above) -- an *amplitude cell* -- or its efficiency (``R`` / ``T``) -- an *efficiency cell*.
Evanescent orders fd also stores are not cells. For a scene, a polarisation and a rung ``n``,
``A(n)`` is the largest ``|fd - ref|`` over its amplitude cells and ``F(n)`` over its efficiency
cells; ``n_f`` is the finest rung and ``n_c`` the finest rung at most ``n_f/1.5`` (the
precedent's pair: the middle rung for M1 and M2, 80 for M3). ``u_A`` (``u_F``) is the largest
distance, over the pair's amplitude (efficiency) cells, of the three ladder rungs below the top
from the top -- one number per kind, as the precedent's ``u_ref`` was one per scene. Every
maximum propagates NaN, and every comparison passes only on a finite number.

1. *Validity.* Every run the report reads -- the three rungs, the 180-period runs and the
   backend runs -- has a plan that is not refused (a refusal is a FAIL), every value finite, no
   ``GrowingFieldWarning``, ``stable`` not False, its ``theta_at`` the fixed-k angle to 1e-9
   degrees, and every cell the reference names; each rung run and each 180-period run also has
   both Diffraction monitors' decay ratio below 1e-5 and no ``UnsettledFieldWarning``. A rung run
   over the decay line is solved once more at 180 periods and that run stands for it (the
   precedent's one repeat). The run length is not what is read: for each scene and polarisation
   the 180-period run at the middle rung differs from the 120-period one by at most
   ``0.1 A(n_f)`` over the amplitude cells and ``0.1 F(n_f)`` over the efficiency cells (the
   120-period side is the run under test and need not have settled). The backends agree: at each
   scene's lowest rung, at 120 periods whether settled or not (agreement is arithmetic, not
   physics), JAX and Torch are within 1e-10 of NumPy over the amplitude and the efficiency cells,
   relative to the largest amplitude cell.
2. *A reference that can judge.* For each scene and polarisation ``u_A <= A(n_f)/3`` and
   ``u_F <= F(n_f)/3``. Otherwise the pair is *not judgeable*: reported, neither passed nor
   failed.
3. *Second order.* ``A`` and ``F`` each fall at every refinement, and the least-squares slope of
   ``log A`` (and of ``log F``) against ``log n`` over the three rungs is at most -1.8.
4. *Consistency.* In every cell the reference lies within ``3 u + 0.1 |X(n_f) - X(n_c)|`` (``u``
   the cell's kind's) of the segment of Richardson limits ``X(n_f) + (X(n_f) - X(n_c))/((n_f/n_c)^p
   - 1)``, ``p`` from 1.4 to 2 (a segment of the complex plane for an amplitude). Cells RCWA holds
   at zero by symmetry are judged with the same band, and their largest fd value is reported.

*The controls* show the rule can fail on these very numbers, and their outcomes are registered
too: fd's amplitudes referred to a pillar one cell off its axis along x and y (a position error
first order in the cell, larger than fd's own error at every rung) must fail criteria 3 and 4,
and every reference value moved along the real axis by the finest rung's error of its kind must
fail criterion 4. A pair that would pass but whose control does not fail as required is *rule
inert*, not passed; a pair that fails stays failed.

*Before the finer rungs* (``project``): once the ladders and the lowest rungs are in, a pair
whose ``u_A`` (``u_F``) exceeds a third of its projected finest-rung error -- the lowest rung's
``A`` (``F``) times ``(n_0/n_f)^1.8`` -- has its ladder lengthened, one truncation step at a
time, until it does not; the ``fd``, ``long`` and ``backends`` commands refuse to start until
every pair holds. Only the reference moves: no fd row, scene or threshold.

A scene and polarisation passes when 1, 2, 3 and 4 hold and its controls fail; a pair with a run
not yet solved, or one that raised and will be solved again, is *incomplete* and the report
*interim*; a pair that otherwise fails 1 fails; one that holds 1 but not 2 is not judgeable (3
and 4 read the reference); C7 passes when all six pairs pass under the registered protocol (the
report compares this file's hash, ladders aside, with the registration's). Whatever fails is
reported with its numbers -- no threshold here moves after the campaign, and a failure is a
finding to explain, not a tolerance to widen. The energy balance ``sum R + sum T - 1`` of every
run is reported, not judged.

*Stated before the data:* the junction campaign's pillar scenes measured this same worst-cell
statistic (its 'totals', split junction) at least-squares orders 1.74 (A: epsilon 4 on 2.25,
height 0.6, 20/40/80 -- M1's materials and rungs) and 1.73 (B: epsilon 5.8 on 2.13 -- M2's
materials), 1.48 (C') and 1.77 (T') for cylinders, and 1.89 (D', a box) and 2.02 (R, a ridge)
-- ``studies/fdtd/subpixel_junction/records/fdtd_subpixel_junction_report.log``. Cylinders may therefore fail
criterion 3 here for the reason C3 names (the default voxel rule's curved wall,
``docs/validation/fdtd/fd-mie-validation.md``); that campaign's junction-free twins put the split junction's own
share of the error near zero. This campaign does not separate the two again.

**Amendments before the campaign** (2026-10-03; no fd row existed, the RCWA rows were rekeyed):
the pillar's sizes made whole cells at every rung (M1 diameter 0.45 -> 0.40, M2 diameter 0.30 ->
0.32 and height 0.625 -> 0.6, M3 side 0.18 -> 0.20) and the axis placed from ``floor(cells/2)``;
M3's rungs 40/60/80 -> 40/80/120 (the precedent's for its epsilon-12 box: 40/60/80 gave the
coarsest rung a slope leverage of -1.51, so the allowances of criteria 1 and 2 could tilt the
slope by 0.5); ``n_c`` the precedent's (at ratio 4/3 the band rejected an apparent order of 2.15);
the source's ``df`` 0.06 -> 0.04 (above); validity extended to every run read, refusals and
missing cells made FAILs, NaN made to fail every comparison, the decay repeat registered, the
run-length check on both polarisations and over cells only (fd's rows also hold evanescent orders,
de-embedded by gains up to 2e5); the controls' required outcomes registered; ``frame`` and
``shift`` logged and required; the ``project`` check registered; the provenance made to cover
``src/wavebit/rcwa/_engine`` and to separate what fd rows and RCWA rows each depend on. Then, after a
second review and a toy run of every command: ``u`` in the band made one number per kind (per
cell, the cells RCWA holds at zero by symmetry had no slack and criterion 4 judged the order of
fd's grid depolarisation there); the phase control replaced by a one-cell misplacement (at M3's
120 rung the dropped correction was only 1.3 to 1.6 times fd's own error, so the control could
pass criterion 3) and 'rule inert' kept from overriding a FAIL; never-solved and raising runs
made 'incomplete'; the backend check run at 120 periods unsettled or not, over efficiencies
too, and repeats limited to the rung runs; ``project`` made a gate and its numbers logged; the
protocol's hash taken once at start, recorded at registration with each scene's ladder and RCWA
keywords, and compared by the report (ladders may only grow at their top); the run-length
check's 120-period side excused from settling; a missing instrument row made a gap, not a FAIL;
the fd provenance narrowed to the functions and constants an fd row's numbers come from, the
ladder rows' provenance to everything but the ladder (each row names its truncation), and the
run's environment kept with every row.

Run: ``python studies/cross_solver/metasurface/fdtd_metasurface_rcwa.py [command ...]``, the commands of **Instruments**
(all of them, in that order, when none is named). Rows go to
``results/fdtd_metasurface_rcwa*.jsonl``, keyed by spec and provenance; a command run again
solves only what is missing.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import cmath
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = repository_path('studies')
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ.setdefault('JAX_ENABLE_X64', '1')      # before JAX is imported: x64 is a cap
import numpy as np                                                                # noqa: E402
from wavebit.compat import legacy as wavebit                                                                    # noqa: E402
import wavebit.fdtd._legacy as fd                                                           # noqa: E402
from wavebit.observations.diffraction import Result                                          # noqa: E402

if not os.path.normcase(str(Path(fd.__file__).resolve())).startswith(os.path.normcase(str(ROOT))):
    raise SystemExit(f'wavebit.fdtd._legacy is imported from {fd.__file__}, not from {ROOT}')

RESULTS = HERE / 'results'
LOG = RESULTS / 'fdtd_metasurface_rcwa.jsonl'
RCWA_LOG = RESULTS / 'fdtd_metasurface_rcwa_reference.jsonl'
REPORT = RESULTS / 'fdtd_metasurface_rcwa_report.log'

WAVELENGTHS = (0.95, 1.0, 1.05)
BAND = dict(wavelength=1.0, df=0.04)
SHEET, GAP_R, GAP_T, CLEAR = 0.6, 0.5, 0.5, 0.3
PML_CELLS, PML_REFLECTION = 24, 1e-12
PERIODS, LONG_PERIODS = 120, 180
JAX_FROM = 48

#: The rule's lines (module docstring).
DECAY_LINE = 1e-5
THETA_GAP = 1e-9
FLOOR_SHARE = 0.1
BACKEND_GAP = 1e-10
REFERENCE_SHARE = 1.0 / 3.0
ORDER_LINE = 1.8
P_RANGE = (1.4, 2.0)
BAND_U, BAND_STEP = 3.0, 0.1
PAIR_RATIO = 1.5
INSTRUMENT_GAP = 1e-10
#: The layout search's margins (``layout``).
COS_LINE, CUTOFF_MARGIN = 0.5, 0.08

#: name: kind, cell (Px, Py), size (diameter | (lx, ly)), height, epsilon pillar / substrate,
#: incidence (degrees), the axis offset past the node line floor(cells/2) and the substrate
#: plane above a node plane (cells), the rungs (cells per vacuum wavelength), the RCWA
#: truncation ladder (half-widths along x and y, the top last), extra RCWA keywords and, for the
#: box, the raster's pixel (its edges carry the box's).
SCENES = {
    'M1': dict(kind='cyl', P=(0.85, 1.00), size=0.40, H=0.60, eps_p=4.0, eps_s=2.25,
               theta=35.0, phi=0.0, o=(0.25, 0.40), zeta=0.25, rungs=(20, 40, 80),
               ladder=((8, 10), (12, 14), (15, 18), (19, 22), (22, 26)), rcwa={}),
    'M2': dict(kind='cyl', P=(0.60, 0.60), size=0.32, H=0.60, eps_p=5.8, eps_s=2.13,
               theta=30.0, phi=0.0, o=(0.25, 0.25), zeta=0.75, rungs=(25, 50, 100),
               ladder=((12, 12), (15, 15), (18, 18), (21, 21), (24, 24)), rcwa={}),
    'M3': dict(kind='box', P=(1.30, 0.50), size=(0.30, 0.20), H=0.30, eps_p=12.0, eps_s=2.25,
               theta=25.0, phi=30.0, o=(0.35, 0.15), zeta=0.25, rungs=(40, 80, 120),
               ladder=((20, 20), (26, 26), (28, 28), (30, 30), (32, 32)), rcwa={},
               pixel=(0.05, 0.01)),
}
POLS = ('s', 'p')


# -- geometry and the orders -------------------------------------------------------------------

from studies._support._fdtd_legacy import archive_order, legacy_rcwa  # noqa: E402


def half_sizes(name):
    g = SCENES[name]
    return (g['size'] / 2,) * 2 if g['kind'] == 'cyl' else (g['size'][0] / 2, g['size'][1] / 2)


def centre(name, n):
    """``(xc, yc)``: the pillar's axis at rung ``n``, ``o`` cells past the node line
    ``floor(cells/2)`` (the cell is centred on the origin, as RCWA's is)."""
    g = SCENES[name]
    dx = 1.0 / n
    out = []
    for P, o in zip(g['P'], g['o']):
        cells = round(P * n)
        if abs(P * n - cells) > 1e-9:
            raise AssertionError(f'{name}: the period {P} is not whole cells at {n}')
        out.append(-P / 2 + (cells // 2 + o) * dx)
    return tuple(out)


def theta_at(name, wl):
    """The fixed-k angle at ``wl`` in degrees: fd's ``theta_at`` (the band centre is 1)."""
    return math.degrees(math.asin(math.sin(math.radians(SCENES[name]['theta'])) * wl))


def order_q(name, m, n):
    """Order ``(m, n)``'s transverse wavevector over the vacuum one at the band centre."""
    g = SCENES[name]
    s = math.sin(math.radians(g['theta']))
    phi = math.radians(g['phi'])
    return (s * math.cos(phi) + m / g['P'][0], s * math.sin(phi) + n / g['P'][1])


def propagating(name, side, wl, m, n):
    index = 1.0 if side == 'r' else math.sqrt(SCENES[name]['eps_s'])
    return wl * math.hypot(*order_q(name, m, n)) < index


def orders(name, side, reach=4):
    """The orders propagating on ``side`` at every wavelength of the band (``layout`` proves
    that no order propagates at some and not at others)."""
    return [(m, n) for n in range(-reach, reach + 1) for m in range(-reach, reach + 1)
            if all(propagating(name, side, wl, m, n) for wl in WAVELENGTHS)]


def cell_keys(name):
    """Every cell's key (side, order, quantity): what the reference holds and fd must."""
    return [f'{side}|{m}|{n}|{q}' for side, power in (('r', 'R'), ('t', 'T'))
            for (m, n) in orders(name, side) for q in (f'{side}_s', f'{side}_p', power)]


def phase(name, rung, m, n):
    """The factor that refers fd's order ``(m, n)`` amplitude at ``rung`` to the centred pillar."""
    xc, yc = centre(name, rung)
    Px, Py = SCENES[name]['P']
    return complex(np.exp(2j * math.pi * (m * xc / Px + n * yc / Py)))


def z_grid(n, zeta, z_min, z_max, pml):
    """``(dx, z_lo, z_hi)`` with the node planes ``zeta`` of a cell below the plane z = 0."""
    dx = 1.0 / n
    z_off = (-zeta) % 1.0
    k_lo = math.floor((z_min - z_off * dx) / dx + 1e-9) - pml
    k_hi = math.ceil((z_max - z_off * dx) / dx - 1e-9) + pml
    z_lo, z_hi = z_off * dx + k_lo * dx, z_off * dx + k_hi * dx
    frac = ((0.0 - z_lo) / dx) % 1.0
    if min(abs(frac - zeta), 1 - abs(frac - zeta)) > 1e-9:
        raise AssertionError(f'zeta {zeta}: the plane sits {frac} of a cell above a node plane')
    return dx, z_lo, z_hi


def planes(name, n):
    """``(dx, z_lo, z_hi, sheet, r_plane, t_plane)`` of ``name`` at rung ``n``."""
    g = SCENES[name]
    sheet = g['H'] + SHEET
    r_plane, t_plane = sheet + GAP_R, -GAP_T
    dx, z_lo, z_hi = z_grid(n, g['zeta'], t_plane - CLEAR, r_plane + CLEAR, PML_CELLS)
    return dx, z_lo, z_hi, sheet, r_plane, t_plane


def faces(name, n):
    """Each face's fraction of a cell past the node plane or line below it: the substrate plane
    and the pillar's top (z), and the pillar's two extremes along x and along y."""
    g = SCENES[name]
    dx, z_lo = planes(name, n)[:2]
    (Px, Py), (hx, hy) = g['P'], half_sizes(name)
    xc, yc = centre(name, n)

    def frac(v, low):
        f = ((v - low) / dx) % 1.0
        return 0.0 if min(f, 1.0 - f) < 1e-9 else round(f, 9)
    return (frac(0.0, z_lo), frac(g['H'], z_lo), frac(xc - hx, -Px / 2), frac(xc + hx, -Px / 2),
            frac(yc - hy, -Py / 2), frac(yc + hy, -Py / 2))


def build(name, n, pol='s', periods=PERIODS):
    """``(scene, info)``: the public scene of ``name`` at ``n`` cells per wavelength."""
    g = SCENES[name]
    (Px, Py), H = g['P'], g['H']
    dx, z_lo, z_hi, sheet, r_plane, t_plane = planes(name, n)
    xc, yc = centre(name, n)
    glass = fd.Material(eps=g['eps_s'])
    deep = z_lo - 2 * dx                  # the substrate runs through the z- PML
    sub = fd.Structure(fd.Box(size=(10.0, 10.0, abs(deep)), center=(0.0, 0.0, deep / 2)), glass)
    if g['kind'] == 'cyl':
        shape = fd.Cylinder(radius=g['size'] / 2, height=H, axis='z', center=(xc, yc, H / 2))
    else:
        shape = fd.Box(size=(g['size'][0], g['size'][1], H), center=(xc, yc, H / 2))
    pillar = fd.Structure(shape, fd.Material(eps=g['eps_p']))
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=g['theta'], phi=g['phi'],
                      polarization=pol, band=fd.Band(**BAND))
    monitors = (fd.Diffraction('r', z=r_plane, side='reflect', wavelengths=WAVELENGTHS,
                               reference_plane=H, incident_plane=H),
                fd.Diffraction('t', z=t_plane, side='transmit', wavelengths=WAVELENGTHS,
                               reference_plane=0.0, incident_plane=H))
    layer = dict(cells=PML_CELLS, reflection=PML_REFLECTION)
    scene = fd.Scene(domain=fd.Domain(size=(Px, Py, z_hi - z_lo),
                                      center=(0.0, 0.0, 0.5 * (z_hi + z_lo))),
                     boundaries=fd.Boundaries(**fd.Bloch.from_source(pw),
                                              z=(fd.PML(background=glass, **layer),
                                                 fd.PML(**layer))),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                     structures=(sub, pillar), sources=(pw,), monitors=monitors)
    return scene, dict(dx=dx, xc=xc, yc=yc, z_lo=z_lo, z_hi=z_hi)


# -- provenance and rows -----------------------------------------------------------------------

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _tree(root, skip=None):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        if skip and (rel == skip or rel.startswith(skip + '/')):
            continue
        h.update(rel.encode() + b'\0' + bytes.fromhex(_sha(path)))
    return h.hexdigest()


#: The fields of a scene an fd row does not read.
RCWA_ONLY = ('ladder', 'rcwa', 'pixel')
#: The fields of a scene an RCWA row depends on.
RCWA_FIELDS = ('kind', 'P', 'size', 'H', 'eps_p', 'eps_s', 'theta', 'phi', 'rcwa',
               'pixel')


def provenance():
    """``digest`` keys the fd rows: wavebit (fd and the rest it imports), the functions and
    constants of this file an fd row's numbers come from, the scenes' fd fields and the versions.
    ``rcwa`` keys each scene's ladder rows (each row's key names its truncation): their functions,
    that scene's RCWA fields, the band, the rest of wavebit and the versions; ``frame`` and
    ``shift`` key the instrument rows, adding their own functions, the truncation they use and,
    for ``shift``, the pillar's place. ``env`` is the run's environment, kept with every row."""
    base = Path(fd.__file__).resolve().parent
    try:
        got = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, timeout=20)
        commit = got.stdout.strip() if got.returncode == 0 else 'unknown'
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', 'src/wavebit'],
                               capture_output=True, text=True, timeout=20).stdout.strip()
    except OSError:
        commit, dirty = 'unknown', ''
    versions = {'numpy': np.__version__, 'python': platform.python_version()}
    for mod in ('scipy', 'jax', 'jaxlib', 'torch'):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    fd_tree = _tree(base)
    rest = _tree(base.parent, skip='fdtd')
    # an fd row's numbers come from these functions, these constants and the scenes' fd fields
    fd_code = [inspect.getsource(f) for f in (half_sizes, centre, theta_at, z_grid, planes, build,
                                              spec, key_of, _c, fd_row, settled, run_specs)]
    constants = dict(WAVELENGTHS=WAVELENGTHS, BAND=BAND, SHEET=SHEET, GAP_R=GAP_R, GAP_T=GAP_T,
                     CLEAR=CLEAR, PML_CELLS=PML_CELLS, PML_REFLECTION=PML_REFLECTION,
                     PERIODS=PERIODS, LONG_PERIODS=LONG_PERIODS, JAX_FROM=JAX_FROM,
                     DECAY_LINE=DECAY_LINE)
    fd_scenes = {name: {k: v for k, v in g.items() if k not in RCWA_ONLY}
                 for name, g in SCENES.items()}
    study = hashlib.sha256(json.dumps([fd_code, constants, fd_scenes], sort_keys=True,
                                      default=repr).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps([fd_tree, rest, study, versions],
                                       sort_keys=True).encode()).hexdigest()[:16]
    code = [inspect.getsource(f) for f in (rcwa_point, box_raster, orders, propagating, order_q,
                                           theta_at, cell_keys, half_sizes)]
    rcwa = {name: hashlib.sha256(json.dumps(
        [code, {k: g.get(k) for k in RCWA_FIELDS}, repr(WAVELENGTHS), rest, versions],
        sort_keys=True, default=repr).encode()).hexdigest()[:16] for name, g in SCENES.items()}
    # the frame rows add their own function to the ladder's
    frame = {name: hashlib.sha256(json.dumps(
        [rcwa[name], inspect.getsource(part_frame), g['ladder'][0]],
        sort_keys=True).encode()).hexdigest()[:16] for name, g in SCENES.items()}
    threads = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                              'MKL_NUM_THREADS')}
    # the shift rows also read the pillar's place: centre(), phase(), the offsets and rungs
    shift = {name: hashlib.sha256(json.dumps(
        [rcwa[name], inspect.getsource(part_shift), inspect.getsource(centre),
         inspect.getsource(phase), g['o'], g['rungs'], g['ladder'][0]],
        sort_keys=True).encode()).hexdigest()[:16] for name, g in SCENES.items()}
    try:
        import jax
        devices = [str(d) for d in jax.devices()]
        jax_backend = jax.default_backend()
    except Exception:                                                      # noqa: BLE001
        devices, jax_backend = None, None
    env = dict(threads=threads, x64=os.environ.get('JAX_ENABLE_X64'),
               xla_flags=os.environ.get('XLA_FLAGS'), platform=platform.platform(),
               jax_backend=jax_backend, jax_devices=devices)
    return dict(digest=digest, rcwa=rcwa, frame=frame, shift=shift, commit=commit,
                dirty=bool(dirty), fd_tree=fd_tree, wavebit_rest=rest, study=study,
                versions=versions, env=env)


PROV = None


def _json(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return repr(v)


def emit(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ''
    if path.exists() and path.stat().st_size:
        with open(path, 'rb') as fh:
            fh.seek(-1, os.SEEK_END)
            lead = '' if fh.read(1) == b'\n' else '\n'        # a line cut by a killed run
    with open(path, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write(lead + json.dumps(row, default=_json) + '\n')
    brief = {k: row[k] for k in ('key', 'status', 'passed') if k in row}
    if 'wall' in row:
        brief['wall'] = round(row['wall'], 1)
    print(json.dumps(brief), flush=True)


def load(path):
    """``{key: row}``, the last row per key under the current provenance; a line a killed run
    cut short is skipped with a note."""
    out = {}
    if path.exists():
        for k, line in enumerate(open(path, encoding='utf-8')):
            line = line.replace('\0', '').strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f'note: {path.name} line {k + 1} is not a whole row; skipped', flush=True)
                continue
            if row.get('kind') == 'instrument':
                want = PROV.get(row.get('check'), {}).get(row.get('scene'))
            elif row.get('kind') == 'rcwa':
                want = PROV['rcwa'].get(row.get('scene'))
            else:
                want = PROV['digest']
            if row.get('prov') == want:
                out[row['key']] = row
    return out


def _c(v):
    return [float(np.real(v)), float(np.imag(v))]


def spec(name, n, pol, periods=None, backend=None):
    return dict(scene=name, n=n, pol=pol, periods=PERIODS if periods is None else periods,
                backend=backend or ('jax' if n >= JAX_FROM else 'numpy'))


def key_of(s):
    return '|'.join(f'{k}={s[k]}' for k in ('scene', 'n', 'pol', 'periods', 'backend'))


def fd_row(s):
    """One public solve as a JSON row: every order's amplitudes and efficiencies per wavelength
    (as fd reports them, not phase-corrected; the window also holds evanescent orders), decay,
    warnings, energy balance, timings. A plan refused by name is ``'refused'``; anything else
    that raises, during the solve or after it, is ``'error'`` and is solved again next time."""
    base = dict(kind='fd', key=key_of(s), prov=PROV['digest'], spec=s, commit=PROV['commit'],
                dirty=PROV['dirty'], wavebit_rest=PROV['wavebit_rest'][:16], env=PROV['env'])
    t0 = time.perf_counter()
    stage = 'plan'
    try:
        scene, info = build(s['scene'], s['n'], s['pol'], s['periods'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plan = fd.plan(scene)
            stage = 'run'
            res = fd.run(scene, plan=plan, backend=s['backend'])
        stage = 'read'
        values, balance = {}, []
        for side, power in (('r', 'R'), ('t', 'T')):
            d = res[side].orders
            for nn in d.orders_v:
                for mm in d.orders_u:
                    o = d.order(int(mm), int(nn))
                    for q in (f'{side}_s', f'{side}_p', power):
                        values[f'{side}|{int(mm)}|{int(nn)}|{q}'] = [
                            _c(v) for v in np.asarray(archive_order(o, q))]
        for i in range(len(WAVELENGTHS)):
            total = sum(float(np.nansum(np.real(np.asarray(
                res[side].orders.R if side == 'r' else res[side].orders.T)[i])))
                for side in ('r', 't'))
            balance.append(total - 1.0)
        names = sorted({f'{w.category.__name__}: {str(w.message)[:120]}' for w in caught})
        decay = {k: (None if v is None else float(v))
                 for k, v in (res.diagnostics.get('decay') or {}).items()}
        stable = res.diagnostics.get('stable')
        row = dict(base, status='ok', values=values, decay=decay, warnings=names,
                   theta_at=[float(v) for v in np.asarray(res['r'].theta_at)],
                   balance=balance, stable=stable if isinstance(stable, bool) else None,
                   steps=int(plan.steps), grid=list(plan.grid.n),
                   centre=[info['xc'], info['yc']], wall=time.perf_counter() - t0)
        json.dumps(row, default=_json)
        return row
    except Exception as err:                                               # noqa: BLE001
        refused = stage == 'plan' and isinstance(err, ValueError)
        return dict(base, status='refused' if refused else 'error', stage=stage,
                    error=f'{type(err).__name__}: {err}'[:600],
                    trace=traceback.format_exc()[-2000:], wall=time.perf_counter() - t0)


def settled(row):
    return (row.get('status') == 'ok' and len(row.get('decay') or {}) == 2
            and all(v is not None and v < DECAY_LINE for v in row['decay'].values()))


def run_specs(specs):
    """Solve every spec not in the log (or logged as ``'error'``); a rung run at 120 periods
    over the decay line gets its one repeat at 180 periods."""
    done = load(LOG)
    for s in specs:
        row = done.get(key_of(s))
        if row is None or row.get('status') == 'error':
            row = fd_row(s)
            emit(LOG, row)
            done[key_of(s)] = row
        ladder_run = s['backend'] == spec(s['scene'], s['n'], s['pol'])['backend']
        if (ladder_run and s['periods'] == PERIODS and row.get('status') == 'ok'
                and not settled(row)):
            again = dict(s, periods=LONG_PERIODS)
            if done.get(key_of(again)) is None or done[key_of(again)].get('status') == 'error':
                done[key_of(again)] = fd_row(again)
                emit(LOG, done[key_of(again)])
    return done


# -- the commands ------------------------------------------------------------------------------

# the RCWA side ---------------------------------------------------------------------------------

def box_raster(name, size, cx=0.0, cy=0.0):
    """The box ``size`` at ``(cx, cy)`` as a one-layer raster on ``SCENES[name]['pixel']``,
    refused unless every edge of the box is a pixel edge (then the raster is the box)."""
    g = SCENES[name]
    (Px, Py), (px, py) = g['P'], g['pixel']
    nx, ny = round(Px / px), round(Py / py)
    for edge, low, step in ((cx - size[0] / 2, -Px / 2, px), (cx + size[0] / 2, -Px / 2, px),
                            (cy - size[1] / 2, -Py / 2, py), (cy + size[1] / 2, -Py / 2, py)):
        k = (edge - low) / step
        if abs(k - round(k)) > 1e-9:
            raise AssertionError(f'{name}: the box edge {edge} is not a pixel edge')
    X, Y = wavebit.meshgrid_xy((ny, nx), period=[Px, Py])
    if X.shape != (ny, nx):
        raise AssertionError(f'meshgrid_xy gave {X.shape}, not (ny, nx) = {(ny, nx)}')
    inside = (np.abs(X - cx) < size[0] / 2) & (np.abs(Y - cy) < size[1] / 2)
    return np.where(inside, g['eps_p'], 1.0)[None]


def rcwa_point(name, wl, trunc, cx=0.0, cy=0.0, path=None, size=None):
    """``{pol: {key: complex}}`` over every cell at ``wl``: both incidences of one solve at the
    fixed-k angle, the pillar (or a box of ``size``) at ``(cx, cy)``. ``path``: ``'shape'`` or
    ``'raster'`` (a box only); the default is the scene's reference path."""
    g = SCENES[name]
    m = g['eps_p']
    path = path or ('raster' if g['kind'] == 'box' else 'shape')
    extra = dict(g.get('rcwa') or {})
    if g['kind'] == 'cyl':
        structure = [dict(background=1.0, shapes=[dict(type='circle', cx=cx, cy=cy,
                                                       lx=g['size'], material=m)])]
    elif path == 'shape':
        lx, ly = size or g['size']
        structure = [dict(background=1.0, shapes=[dict(type='rectangle', cx=cx, cy=cy, lx=lx,
                                                       ly=ly, material=m)])]
    else:
        structure = box_raster(name, size or g['size'], cx, cy)
        extra.update(fourier_coefficients='integrated', factorization='li')
    res = legacy_rcwa(backend='numpy', top=1.0, bot=g['eps_s'],
                       theta=math.radians(theta_at(name, wl)), phi=math.radians(g['phi']),
                       psi=math.pi / 2, truncation_order=list(trunc),
                       period=list(g['P']), wavelength=wl, structure=structure,
                       thickness=[g['H']], **extra).solve()
    out = {}
    for pol, sub in (('s', res.s_inc), ('p', res.p_inc)):
        one = Result(incident=sub, order_window=res.order_window)
        vals = {}
        for side, power in (('r', 'R'), ('t', 'T')):
            for (mm, nn) in orders(name, side):
                o = one.order(mm, nn)
                for q in (f'{side}_s', f'{side}_p', power):
                    vals[f'{side}|{mm}|{nn}|{q}'] = complex(np.asarray(getattr(o, q)))
        out[pol] = vals
    return out


def worst(values):
    """The largest of ``values``, NaN if any is NaN or there are none (``max(0.0, nan)`` is 0.0
    in Python: every maximum a check passes on goes through here)."""
    a = np.asarray(list(values), dtype=float)
    return float(np.max(a)) if a.size and not np.any(np.isnan(a)) else float('nan')


def part_frame():
    """The box's raster against the shape path where the normal-vector-field rule is Li's: a
    box spanning the cell along one axis (the raster's a pixel longer each way, so both read the
    whole span), at an offset centre, once along each axis. Every cell to 1e-10."""
    done = load(RCWA_LOG)
    rows = []
    for name, g in SCENES.items():
        if g['kind'] != 'box':
            continue
        (Px, Py), (px, py) = g['P'], g['pixel']
        trunc = g['ladder'][0]
        for axis, shape_size, raster_size, centre_ in (
                ('x', (g['size'][0], Py), (g['size'][0], Py + 2 * py), (3 * px, 0.0)),
                ('y', (Px, g['size'][1]), (Px + 2 * px, g['size'][1]), (0.0, 3 * py))):
            key = f'{name}|frame|{axis}'
            if key in done:
                rows.append(done[key])
                continue
            a = rcwa_point(name, 1.0, trunc, *centre_, path='shape', size=shape_size)
            b = rcwa_point(name, 1.0, trunc, *centre_, path='raster', size=raster_size)
            gap = worst(abs(a[pol][k] - b[pol][k]) for pol in POLS for k in a[pol])
            row = dict(kind='instrument', key=key, prov=PROV['frame'][name], scene=name,
                       check='frame', axis=axis, truncation=list(trunc), gap=gap,
                       passed=bool(gap <= INSTRUMENT_GAP))
            emit(RCWA_LOG, row)
            rows.append(row)
    return rows


def part_shift():
    """RCWA at each rung's offset against the centred solve times :func:`phase`'s inverse, on
    the shape path (the identity is the solver's, not a factorisation's)."""
    done = load(RCWA_LOG)
    rows = []
    for name, g in SCENES.items():
        trunc = g['ladder'][0]
        plain = None
        for n in g['rungs']:
            key = f'{name}|shift|{n}'
            if key in done:
                rows.append(done[key])
                continue
            if plain is None:
                plain = rcwa_point(name, 1.0, trunc, path='shape')
            xc, yc = centre(name, n)
            moved = rcwa_point(name, 1.0, trunc, xc, yc, path='shape')
            gaps = []
            for pol in POLS:
                for k, v in plain[pol].items():
                    side, m, nn, q = k.split('|')
                    f = 1.0 if q in ('R', 'T') else 1.0 / phase(name, n, int(m), int(nn))
                    gaps.append(abs(moved[pol][k] - v * f))
            gap = worst(gaps)
            row = dict(kind='instrument', key=key, prov=PROV['shift'][name], scene=name,
                       check='shift', rung=n, truncation=list(trunc), gap=gap,
                       passed=bool(gap <= INSTRUMENT_GAP))
            emit(RCWA_LOG, row)
            rows.append(row)
    return rows


def ladder_key(name, i, trunc):
    return f'{name}|{i}|{trunc[0]}x{trunc[1]}'


def part_rcwa():
    done = load(RCWA_LOG)
    for name, g in SCENES.items():
        for trunc in g['ladder']:
            for i, wl in enumerate(WAVELENGTHS):
                key = ladder_key(name, i, trunc)
                if key in done:
                    continue
                t0 = time.perf_counter()
                vals = rcwa_point(name, wl, trunc)
                row = dict(kind='rcwa', key=key, prov=PROV['rcwa'][name], scene=name,
                           wavelength=wl, truncation=list(trunc),
                           values={pol: {kk: _c(v) for kk, v in vals[pol].items()}
                                   for pol in POLS},
                           wall=time.perf_counter() - t0)
                emit(RCWA_LOG, row)
                done[key] = row


def reference(name):
    """``(ref, ladder)``: per polarisation ``{key: [complex per wavelength]}`` at the top of the
    ladder, and the same for the three truncations below it; ``None`` if a row is missing."""
    rows = load(RCWA_LOG)

    def at(trunc):
        out = {pol: {} for pol in POLS}
        for i in range(len(WAVELENGTHS)):
            row = rows.get(ladder_key(name, i, trunc))
            if row is None:
                return None
            for pol in POLS:
                for key, v in row['values'][pol].items():
                    out[pol].setdefault(key, [None] * len(WAVELENGTHS))[i] = complex(*v)
        return out
    ladder = SCENES[name]['ladder']
    got = [at(trunc) for trunc in ladder[-4:]]
    if any(g is None for g in got):
        return None
    return got[-1], got[:-1]


# the rule --------------------------------------------------------------------------------------

def segment_distance(z, a, b):
    """The distance from ``z`` to the segment ``[a, b]`` of the complex plane."""
    d = b - a
    if abs(d) == 0:
        return abs(z - a)
    t = min(1.0, max(0.0, ((z - a) * np.conj(d)).real / abs(d) ** 2))
    return abs(z - (a + t * d))


def pair_rungs(rungs):
    """``(n_c, n_f)``: the finest rung and the finest one at most ``n_f/1.5``; ``None`` when no
    rung is that coarse (``layout`` refuses such a ladder)."""
    n_f = max(rungs)
    coarse = [n for n in rungs if n <= n_f / PAIR_RATIO]
    return (max(coarse), n_f) if coarse else None


def cells_of(name, pol, rung_row, ref, ladder, misplace=0.0):
    """``[(cell, kind, fd value, ref value, u)]`` for one run against the top of the ladder;
    ``misplace`` refers fd's amplitudes to a pillar that many cells off its true axis along x
    and y (a control: a position error first order in the cell). A cell fd does not hold is
    NaN, which fails every comparison."""
    out = []
    for key, top in ref.items():
        side, m, n, q = key.split('|')
        kind = 'eff' if q in ('R', 'T') else 'amp'
        for i in range(len(WAVELENGTHS)):
            got = rung_row['values'].get(key)
            value = complex(*got[i]) if got else complex('nan')
            if kind == 'amp':
                rung = rung_row['spec']['n']
                value *= phase(name, rung, int(m), int(n))
                if misplace:
                    Px, Py = SCENES[name]['P']
                    value *= cmath.exp(2j * math.pi * misplace / rung * (int(m) / Px
                                                                          + int(n) / Py))
            r = top[i]
            u = worst(abs(lower[key][i] - r) for lower in ladder)
            out.append(((pol, WAVELENGTHS[i], side, int(m), int(n), q), kind, value, r, u))
    return out


def judge(name, pol, runs, ref, ladder, misplace=0.0, displace=0.0):
    """Criteria 2, 3 and 4 for one scene and polarisation (module docstring).

    The controls use it unchanged: ``misplace`` refers fd's amplitudes to a pillar that many
    cells off its axis, and ``displace`` moves every reference value along the real axis by that
    multiple of the finest rung's error of its kind (``A`` or ``F``)."""
    rungs = [r['spec']['n'] for r in runs]
    cells = [cells_of(name, pol, r, ref, ladder, misplace) for r in runs]

    def errors(cs_list):
        return ([worst(abs(c[2] - c[3]) for c in cs if c[1] == 'amp') for cs in cs_list],
                [worst(abs(c[2] - c[3]) for c in cs if c[1] == 'eff') for cs in cs_list])
    A, F = errors(cells)
    if displace:
        move = {'amp': displace * A[-1], 'eff': displace * F[-1]}
        cells = [[(c[0], c[1], c[2], c[3] + move[c[1]], c[4]) for c in cs] for cs in cells]
        A, F = errors(cells)
    uA = worst(c[4] for c in cells[0] if c[1] == 'amp')
    uF = worst(c[4] for c in cells[0] if c[1] == 'eff')
    finite = all(math.isfinite(v) and v > 0 for v in A + F)
    slope_A = float(np.polyfit(np.log(rungs), np.log(A), 1)[0]) if finite else float('nan')
    slope_F = float(np.polyfit(np.log(rungs), np.log(F), 1)[0]) if finite else float('nan')
    falls = all(b < a for a, b in zip(A, A[1:])) and all(b < a for a, b in zip(F, F[1:]))
    second = bool(falls and slope_A <= -ORDER_LINE and slope_F <= -ORDER_LINE)
    judgeable = bool(uA <= REFERENCE_SHARE * A[-1] and uF <= REFERENCE_SHARE * F[-1])
    pair = pair_rungs(rungs)
    if pair is None:
        return dict(A=A, F=F, uA=uA, uF=uF, slope_A=slope_A, slope_F=slope_F, second=second,
                    judgeable=judgeable, consistent=False, outside=[('no Richardson pair',)],
                    cells=len(cells[-1]), pair=None)
    n_c, n_f = pair
    ratio = n_f / n_c
    fine, coarse = cells[rungs.index(n_f)], cells[rungs.index(n_c)]
    outside = []
    for cf, cc in zip(fine, coarse):
        if cf[0] != cc[0]:
            raise AssertionError(f'cells out of step: {cf[0]} against {cc[0]}')
        xf, xc_, r = cf[2], cc[2], cf[3]
        u = uA if cf[1] == 'amp' else uF          # the precedent's: one u per kind and pair
        lo = xf + (xf - xc_) / (ratio ** P_RANGE[0] - 1)
        hi = xf + (xf - xc_) / (ratio ** P_RANGE[1] - 1)
        d = segment_distance(r, lo, hi)
        line = BAND_U * u + BAND_STEP * abs(xf - xc_)
        if not d <= line:
            outside.append((cf[0], d, line))
    return dict(A=A, F=F, uA=uA, uF=uF, slope_A=slope_A, slope_F=slope_F, second=second,
                judgeable=judgeable, consistent=not outside, outside=outside,
                cells=len(fine), pair=(n_c, n_f))


def validity(name, row, settle=True):
    """Criterion 1 for one run: the reasons it is not valid (empty when it is). ``settle=False``
    leaves out the decay line and its warning -- for the 120-period side of the run-length
    check, whose 180-period run is the one that must have settled."""
    if row is None:
        return ['never run']
    if row.get('status') != 'ok':
        return [f'{row.get("status")}: {row.get("error", "")[:200]}']
    bad = []
    if settle and not settled(row):
        bad.append(f'decay {row["decay"]}')
    if any('GrowingField' in w for w in row['warnings']):
        bad.append('growth warning')
    if settle and any('UnsettledField' in w for w in row['warnings']):
        bad.append('unsettled warning')
    if row.get('stable') is False:
        bad.append('stable False')
    gap = worst(abs(t - theta_at(name, wl)) for t, wl in zip(row['theta_at'], WAVELENGTHS))
    if not gap <= THETA_GAP:
        bad.append(f'theta_at off by {gap}')
    missing = [k for k in cell_keys(name) if k not in row['values']]
    if missing:
        bad.append(f'missing cells {missing[:4]}')
    if not all(math.isfinite(x) for v in row['values'].values() for pair in v for x in pair):
        bad.append('non-finite values')
    return bad


def rung_run(rows, name, n, pol):
    """The run that stands for a rung: the 120-period one, or its 180-period repeat when the
    first is over the decay line (criterion 1)."""
    first = rows.get(key_of(spec(name, n, pol)))
    if first is not None and first.get('status') == 'ok' and not settled(first):
        return rows.get(key_of(spec(name, n, pol, periods=LONG_PERIODS))), True
    return first, False


def gap_over_cells(name, a, b, relative=False):
    """``(amplitude gap, efficiency gap)`` between two runs over the cells only, NaN-safe;
    ``relative`` divides by the largest amplitude cell of ``a``."""
    amp, eff, scale = [], [], []
    for key in cell_keys(name):
        q = key.split('|')[-1]
        for x, y in zip(a['values'].get(key, [[math.nan, 0]] * 3),
                        b['values'].get(key, [[math.nan, 0]] * 3)):
            d = abs(complex(*x) - complex(*y))
            (eff if q in ('R', 'T') else amp).append(d)
            if q not in ('R', 'T'):
                scale.append(abs(complex(*x)))
    s = worst(scale) if relative else 1.0
    return worst(amp) / s, worst(eff) / (s if relative else 1.0)


def all_specs():
    return [spec(name, n, pol) for name in SCENES for pol in POLS
            for n in SCENES[name]['rungs']]


def long_specs():
    return [spec(name, SCENES[name]['rungs'][1], pol, periods=LONG_PERIODS) for name in SCENES
            for pol in POLS]


def backend_specs():
    return [spec(name, SCENES[name]['rungs'][0], 's', backend=b) for name in SCENES
            for b in ('numpy', 'jax', 'torch')]


def lowest_specs():
    return [spec(name, SCENES[name]['rungs'][0], pol) for name in SCENES for pol in POLS]


def part_project():
    """The check registered for before the finer rungs (module docstring): ``u`` against a
    third of the projected finest-rung error, per scene and polarisation."""
    rows = load(LOG)
    out = {}
    for name, g in SCENES.items():
        got = reference(name)
        if got is None:
            print(f'{name}: the RCWA ladder is incomplete', flush=True)
            continue
        ref, ladder = got
        n0, nf = g['rungs'][0], g['rungs'][-1]
        for pol in POLS:
            r, _ = rung_run(rows, name, n0, pol)
            if r is None or r.get('status') != 'ok':
                print(f'{name} {pol}: no lowest-rung run', flush=True)
                continue
            cells = cells_of(name, pol, r, ref[pol], [lad[pol] for lad in ladder])
            A0 = worst(abs(c[2] - c[3]) for c in cells if c[1] == 'amp')
            F0 = worst(abs(c[2] - c[3]) for c in cells if c[1] == 'eff')
            uA = worst(c[4] for c in cells if c[1] == 'amp')
            uF = worst(c[4] for c in cells if c[1] == 'eff')
            shrink = (n0 / nf) ** ORDER_LINE
            ok = bool(uA <= REFERENCE_SHARE * A0 * shrink and uF <= REFERENCE_SHARE * F0 * shrink)
            out[(name, pol)] = ok
            emit(RCWA_LOG, dict(kind='project', key=f'{name}|{pol}|project|{len(g["ladder"])}',
                                prov=PROV['rcwa'][name], scene=name, pol=pol,
                                top=list(g['ladder'][-1]), A0=A0, F0=F0, uA=uA, uF=uF,
                                projected_A=A0 * shrink, projected_F=F0 * shrink, passed=ok,
                                time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            print(f'{name} {pol}: A({n0}) {A0:.3e} F({n0}) {F0:.3e}; projected A({nf}) '
                  f'{A0 * shrink:.2e} F({nf}) {F0 * shrink:.2e}; u amp {uA:.2e} eff {uF:.2e}: '
                  f'{"ladder enough" if ok else "LENGTHEN THE LADDER"}', flush=True)
    return out


#: The runs ``tests/fdtd/test_fd_metasurface.py`` repeats, and where their rows go.
EXPORTED = (('M1', 20, 'p'), ('M2', 25, 's'), ('M3', 40, 's'))
EXPORT = ROOT / 'tests' / 'references' / 'fd_metasurface_rows.json'


def part_export():
    """Write :data:`EXPORTED`'s campaign rows (the cells only), each scene's RCWA top over the
    cells, and the errors the report reads at those rungs, for the tests to repeat."""
    rows = load(LOG)
    out = dict(provenance=dict(digest=PROV['digest'], rcwa=PROV['rcwa'], commit=PROV['commit']),
               runs=[])
    for name, n, pol in EXPORTED:
        row, again = rung_run(rows, name, n, pol)
        got = reference(name)
        if row is None or row.get('status') != 'ok' or got is None:
            raise SystemExit(f'{name} {n} {pol}: no campaign row or reference to export')
        ref, ladder = got
        cells = cells_of(name, pol, row, ref[pol], [lad[pol] for lad in ladder])
        out['runs'].append(dict(
            scene=name, n=n, pol=pol, periods=row['spec']['periods'], key=row['key'],
            values={k: row['values'][k] for k in cell_keys(name)},
            reference={k: [_c(v) for v in ref[pol][k]] for k in cell_keys(name)},
            A=worst(abs(c[2] - c[3]) for c in cells if c[1] == 'amp'),
            F=worst(abs(c[2] - c[3]) for c in cells if c[1] == 'eff')))
    EXPORT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'wrote {EXPORT}', flush=True)


REGISTRATION = RESULTS / 'fdtd_metasurface_rcwa_registration.json'


def part_register():
    """Record the protocol as it stands before the campaign: this file's SHA-256 and a copy of
    it next to the logs, with the time and the provenance (written once; never overwritten)."""
    copy = RESULTS / 'fdtd_metasurface_rcwa_registered.py.txt'
    if REGISTRATION.exists():
        print(f'{REGISTRATION.name} exists; the protocol was registered before', flush=True)
        return
    if LOG.exists() and any('"kind": "fd"' in line for line in open(LOG, encoding='utf-8')):
        raise SystemExit(f'{LOG.name} already holds fd rows: a registration now would postdate '
                         f'data')
    text = Path(__file__).read_bytes()
    RESULTS.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(text)
    record = dict(file='studies/cross_solver/metasurface/fdtd_metasurface_rcwa.py', sha256=hashlib.sha256(
        text.replace(b'\r\n', b'\n')).hexdigest(), registered=time.strftime(
        '%Y-%m-%dT%H:%M:%S%z'), protocol=PROTOCOL, digest=PROV['digest'],
        rcwa=PROV['rcwa'], ladders={name: [list(tr) for tr in g['ladder']]
                                    for name, g in SCENES.items()},
        rcwa_keywords={name: g.get('rcwa') or {} for name, g in SCENES.items()},
        commit=PROV['commit'], dirty=PROV['dirty'], copy=copy.name)
    REGISTRATION.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(record), flush=True)


def part_layout():
    rows = []
    for name, g in SCENES.items():
        statuses, cos_min, margin = {}, 1.0, math.inf
        for side in ('r', 't'):
            index = 1.0 if side == 'r' else math.sqrt(g['eps_s'])
            for m, n in itertools.product(range(-5, 6), repeat=2):
                q = math.hypot(*order_q(name, m, n))
                statuses[(side, m, n)] = [wl * q < index for wl in WAVELENGTHS]
                for wl in WAVELENGTHS:
                    u = wl * q / index
                    if u < 1:
                        cos_min = min(cos_min, math.sqrt(1 - u * u))
                    else:
                        margin = min(margin, u - 1)
        mixed = [k for k, st in statuses.items() if any(st) and not all(st)]
        fractions = {n: faces(name, n) for n in g['rungs']}
        same = len(set(fractions.values())) == 1
        paired = pair_rungs(g['rungs']) is not None
        ok = bool(not mixed and cos_min >= COS_LINE and margin >= CUTOFF_MARGIN and same
                  and paired)
        row = dict(scene=name, passed=ok, cos_min=cos_min, cutoff_margin=margin, mixed=mixed,
                   faces={str(n): f for n, f in fractions.items()}, faces_same=same,
                   pair=pair_rungs(g['rungs']),
                   reflected=orders(name, 'r'), transmitted=orders(name, 't'))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def protocol_hash():
    """SHA-256 of this file (LF line ends) with the SCENES ladders and RCWA keywords blanked: the
    registered rule, under which the ``project`` step may lengthen a ladder."""
    text = Path(__file__).read_text(encoding='utf-8').replace('\r\n', '\n')
    head, rest = text.split('\nSCENES = {', 1)
    tail = rest.split('\nPOLS = ', 1)[1]
    fields = {name: {k: v for k, v in g.items() if k not in ('ladder', 'rcwa')}
              for name, g in SCENES.items()}
    return hashlib.sha256((head + repr(fields) + tail).encode()).hexdigest()


def incomplete(row):
    """A run not yet solved, or one that raised and will be solved again: a gap, not a FAIL."""
    return row is None or row.get('status') == 'error'


def part_report():
    rows = load(LOG)
    instruments = {k: r for k, r in load(RCWA_LOG).items() if r.get('kind') == 'instrument'}
    lines, verdicts = [], {}

    def say(text):
        lines.append(text)
        print(text, flush=True)

    registered = (json.loads(REGISTRATION.read_text(encoding='utf-8'))
                  if REGISTRATION.exists() else {})
    changes = []
    if registered.get('protocol') != PROTOCOL:
        changes.append('the rule')
    if protocol_hash() != PROTOCOL:
        changes.append('the file on disk since this run started')
    for name, g in SCENES.items():
        before = [tuple(tr) for tr in registered.get('ladders', {}).get(name, [])]
        if tuple(g['ladder'][:len(before)]) != tuple(before) or not before:
            changes.append(f'{name} ladder (registered {before})')
        if (g.get('rcwa') or {}) != registered.get('rcwa_keywords', {}).get(name, {}):
            changes.append(f'{name} RCWA keywords')
    same_rule = not changes
    say(f'C7 report {time.strftime("%Y-%m-%dT%H:%M:%S%z")} -- provenance {PROV["digest"]} '
        f'(rcwa {PROV["rcwa"]}), commit {PROV["commit"][:12]}'
        f'{" (dirty)" if PROV["dirty"] else ""}; protocol {PROTOCOL[:16]}, registered '
        f'{registered.get("protocol", "never")[:16]} at {registered.get("registered")}: '
        f'{"same rule" if same_rule else "RULE CHANGED: " + "; ".join(changes)}')
    for name, g in SCENES.items():
        before = len(registered.get('ladders', {}).get(name, []))
        say(f'{name} ladder: top {g["ladder"][-1]}, below it {list(g["ladder"][-4:-1])}; '
            f'{len(g["ladder"]) - before} appended since registration')
    layout = {r['scene']: r['passed'] for r in part_layout()}
    say(f'layout: {layout}')
    for name, g in SCENES.items():
        rungs = g['rungs']
        wanted = [f'{name}|shift|{n}' for n in rungs] + (
            [f'{name}|frame|x', f'{name}|frame|y'] if g['kind'] == 'box' else [])
        inst_ok = all(instruments.get(k, {}).get('passed') is True for k in wanted)
        inst_gap = any(k not in instruments for k in wanted)
        say(f'{name} instruments {"pass" if inst_ok else "MISSING OR FAILED"}: ' + ', '.join(
            f'{k.split("|", 1)[1]} {instruments[k]["gap"]:.1e}' if k in instruments
            else f'{k} missing' for k in wanted))
        got = reference(name)
        if got is None:
            say(f'{name}: the RCWA ladder is incomplete')
            for pol in POLS:
                verdicts[(name, pol)] = 'incomplete'
            continue
        ref, ladder = got
        per = {}
        for pol in POLS:
            picked = [rung_run(rows, name, n, pol) for n in rungs]
            runs = [r for r, _ in picked]
            v1 = [(n, validity(name, r)) for n, r in zip(rungs, runs)]
            repeats = [n for n, (_, again) in zip(rungs, picked) if again]
            per[pol] = dict(valid=all(not bad for _, bad in v1), v1=v1, repeats=repeats,
                            gap=any(incomplete(r) for r in runs))
            if any(r is None or r.get('status') != 'ok' for r in runs):
                say(f'{name} {pol}: runs not all ok -- {[(n, b) for n, b in v1 if b]}')
                continue
            ladders = [lad[pol] for lad in ladder]
            j = judge(name, pol, runs, ref[pol], ladders)
            controls = dict(misplaced=judge(name, pol, runs, ref[pol], ladders, misplace=1.0),
                            displaced=judge(name, pol, runs, ref[pol], ladders, displace=1.0))
            inert = []
            if controls['misplaced']['second'] or controls['misplaced']['consistent']:
                inert.append('misplaced control did not fail criteria 3 and 4')
            if controls['displaced']['consistent']:
                inert.append('displaced control did not fail criterion 4')
            zero = [c for c in cells_of(name, pol, runs[-1], ref[pol], ladders)
                    if abs(c[3]) < 1e-10]
            per[pol].update(j, controls=controls, inert=inert,
                            balance=[r['balance'] for r in runs])
            say(f'{name} {pol}: A {", ".join(f"{a:.3e}" for a in j["A"])} (slope '
                f'{j["slope_A"]:.3f}); F {", ".join(f"{f:.3e}" for f in j["F"])} (slope '
                f'{j["slope_F"]:.3f}); u amp {j["uA"]:.2e} eff {j["uF"]:.2e}; {j["cells"]} '
                f'cells; pair {j["pair"]}; repeats {repeats}')
            say(f'   1 valid {per[pol]["valid"]} {[(n, b) for n, b in v1 if b]}; 2 judgeable '
                f'{j["judgeable"]}; 3 second order {j["second"]}; 4 consistent '
                f'{j["consistent"]} ({len(j["outside"])} outside'
                f'{": " + str(j["outside"][:4]) if j["outside"] else ""})')
            if zero:
                say(f'   cells zero by symmetry in RCWA: {len(zero)}, largest |fd| at '
                    f'{rungs[-1]} {worst(abs(c[2]) for c in zero):.2e}')
            say(f'   energy balance per rung (sum R + sum T - 1): '
                f'{[[round(b, 6) for b in bal] for bal in per[pol]["balance"]]}')
            for label, c in controls.items():
                say(f'   control {label}: A {", ".join(f"{a:.2e}" for a in c["A"])} slope '
                    f'{c["slope_A"]:.2f}, F slope {c["slope_F"]:.2f}; 3 {c["second"]}, 4 '
                    f'{c["consistent"]} ({len(c["outside"])} of {c["cells"]} outside)')
            if inert:
                say(f'   controls that did not fail: {inert}')
        # 1. the run length and the backends
        for pol in POLS:
            p = per[pol]
            mid = rows.get(key_of(spec(name, rungs[1], pol)))
            long_ = rows.get(key_of(spec(name, rungs[1], pol, periods=LONG_PERIODS)))
            p['gap'] = p['gap'] or incomplete(mid) or incomplete(long_)
            bad = validity(name, mid, settle=False) + validity(name, long_)
            if 'A' in p and not bad:
                dA, dF = gap_over_cells(name, mid, long_)
                lineA, lineF = FLOOR_SHARE * p['A'][-1], FLOOR_SHARE * p['F'][-1]
                p['floor'] = bool(dA <= lineA and dF <= lineF)
                say(f'{name} {pol} run length at {rungs[1]}: 120 vs 180 periods amp {dA:.2e} '
                    f'eff {dF:.2e} (lines {lineA:.2e}, {lineF:.2e}): {p["floor"]}')
            else:
                p['floor'] = False
                say(f'{name} {pol} run length: not judged -- {bad or "no criteria"}')
        # backend identity is arithmetic, not physics: the 120-period runs, settled or not
        b_rows = {b: rows.get(key_of(spec(name, rungs[0], 's', backend=b)))
                  for b in ('numpy', 'jax', 'torch')}
        backend_gap = any(incomplete(r) for r in b_rows.values())
        bad = {b: validity(name, r, settle=False) for b, r in b_rows.items()}
        if not any(bad.values()):
            gaps = {b: gap_over_cells(name, b_rows['numpy'], b_rows[b], relative=True)
                    for b in ('jax', 'torch')}
            backend_ok = all(a <= BACKEND_GAP and e <= BACKEND_GAP for a, e in gaps.values())
            say(f'{name} backends at {rungs[0]} (s): ' + ', '.join(
                f'{b} amp {a:.2e} eff {e:.2e}' for b, (a, e) in gaps.items())
                + f': {backend_ok}')
        else:
            backend_ok = False
            say(f'{name} backends: not judged -- {bad}')
        for pol in POLS:
            p = per[pol]
            ok1 = (p['valid'] and p.get('floor') is True and backend_ok and layout[name]
                   and inst_ok)
            if not ok1 and (p['gap'] or backend_gap or inst_gap):
                verdict = 'incomplete'
            elif not ok1:
                verdict = 'FAIL'
            elif not p['judgeable']:
                verdict = 'not judgeable'
            elif not (p['second'] and p['consistent']):
                verdict = 'FAIL'
            elif p['inert']:
                verdict = 'rule inert'
            else:
                verdict = 'pass'
            verdicts[(name, pol)] = verdict
            say(f'{name} {pol}: {verdict}')
    complete = (len(verdicts) == len(SCENES) * len(POLS)
                and 'incomplete' not in verdicts.values())
    overall = ('PASS' if complete and same_rule and all(v == 'pass' for v in verdicts.values())
               else 'NOT PASSED' if complete else 'INTERIM')
    say(f'C7: {overall}{"" if same_rule else " (the rule changed since registration)"} -- '
        + ', '.join(f'{n} {p} {v}' for (n, p), v in sorted(verdicts.items())))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    return verdicts


PROTOCOL = None


def main(argv):
    global PROV, PROTOCOL
    PROV = provenance()
    PROTOCOL = protocol_hash()
    print(json.dumps({k: PROV[k] for k in ('digest', 'rcwa', 'commit', 'dirty')}), flush=True)
    parts = argv or ['register', 'layout', 'frame', 'shift', 'lowest', 'rcwa', 'project', 'fd',
                     'long', 'backends', 'report']
    actions = {'register': part_register, 'layout': part_layout, 'frame': part_frame,
               'shift': part_shift, 'rcwa': part_rcwa,
               'lowest': lambda: run_specs(lowest_specs()), 'project': part_project,
               'fd': lambda: run_specs(all_specs()),
               'long': lambda: run_specs(long_specs()),
               'backends': lambda: run_specs(backend_specs()), 'report': part_report,
               'export': part_export}
    for part in parts:
        if part not in actions:
            raise SystemExit(f'unknown command {part!r}: one of {sorted(actions)}')
        if part in ('fd', 'long', 'backends'):
            gate = part_project()
            pairs = [(name, pol) for name in SCENES for pol in POLS]
            if not all(gate.get(pair) is True for pair in pairs):
                raise SystemExit(f'the project check does not hold for '
                                 f'{[pair for pair in pairs if gate.get(pair) is not True]}: '
                                 f'lengthen those ladders before the finer rungs run')
        actions[part]()


if __name__ == '__main__':
    main(sys.argv[1:])
