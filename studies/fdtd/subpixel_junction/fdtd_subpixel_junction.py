"""Can a pillar standing flush on its substrate be lowered, and by which junction rule?

Production integration: ``fd.Structure(junction='split')`` now implements the selected
rule without this harness's patches. ``studies/cross_solver/junction/fdtd_junction_production.py`` validates
the added exact node-plane ties; see ``docs/validation/fdtd/fd-subpixel-occupancy.md``, The junction.
The historical question, registered gates and recorded rows below remain unchanged.

**The question.** Door 5 (``src/wavebit/fdtd/_prepare/checks.py``) refuses a smoothed pillar standing
flush on a smoothed substrate: at the base, voxels hold two smoothed surfaces -- the substrate's
plane and the pillar's wall or base -- and the two-phase mixing rule composes neither. The owner
wants a switch, ``fd.Structure(..., junction=)``. This study decides its default among three
outcomes only: 'split' becomes the default, 'refuse' stays, or the numbers go to the owner. It
never ranks rules by their total error against RCWA: every rule carries fd's own error, and a
first-order rule can cancel it (the film anchor shows exactly that at 40 cells).

**Junction voxels.** J is door 5's own set: the voxels, over the E and H families and the two
common points, where both exact fills lie in ``(1e-9, 1 - 1e-9)``. J3 are those where the
pillar's transverse fill ``f_up`` is fractional too (the rim, three media), J2 the rest (two
media, one flat interface). The patch writes the E edges and the E node; the H families enter
the census only (nothing here is magnetic, so ``bh`` is the scalar ``dt/dx``).

**The write set is J and two sets outside it** where production's single-body rule misreads
the flush pair, and where every writing arm writes arm 2's closed form (so each arm differs
from arm 2 on J only): the plane's nodes whose pillar fill is a round-off 1e-16
(:data:`NODE_EXTRA`, production's ``fill == 0`` test), and the cylinder's *base side*
(:func:`geometry`): rim points with ``alpha <= 1e-9`` whose nearest face in production's
``surface()`` is the pillar's base, whose tangent plane gives the wall no weight. The base side
is empty at the registered placements 0.25 and 0.75 (the first row above the plane is 0.75 of
a cell from the base there); it exists for continuity in the pair's height (V5's height sweep).

**The arms**, per E component c on its own voxel. ``alpha`` is the substrate's exact fill,
``f_up`` the pillar's transverse fill (flush: ``f = (1 - alpha) f_up``), ``a = f_up eps_p +
(1 - f_up) eps_g``, ``eta = f_up/eps_p + (1 - f_up)/eps_g``, ``A = (1-alpha) a + alpha eps_s``,
``B = (1-alpha)/eta + alpha eps_s``, and ``P_up`` the wall-only weight:

* ``2`` (split): ``Xi_t = [(1-alpha) Xi_up,t^-1 + alpha eps_s I]^-1`` with ``Xi_up,t =
  (I - P_up)/a + eta P_up``, ``Xi_zz = (1-alpha)/a + alpha/eps_s``, ``Xi_xz = Xi_yz = 0``;
  painted ``1/eps_c = Xi_cc``; at the E node the full tensor, cross ``(Xi_xy, 0, 0)`` and
  ``1/held = 1/A``. Closed forms (the patch's route): a cylinder's rank-1 ``P_up = g g^T``
  gives ``Xi_cc = (1 - g_c^2)/A + g_c^2/B``; a Box's diagonal ``P_up`` gives ``Xi_cc =
  1/[(1-alpha) mix(eps_g, eps_p, f_up, P_cc) + alpha eps_s]`` for c in {x, y} only. V3 checks
  the lowered numbers against the general 2x2-inverse form, a second route. ``P_up`` follows
  production's own rule for each family, so that ``alpha -> 0`` is today's rule exactly: at the
  E edges it is ``g g^T`` (g from ``Cylinder._transverse_surface``, never ``surface()``) where
  production's tangent-plane test (``_cut._parts`` of the wall plane, ``_mean_of_the_moment``)
  finds the wall in the voxel and 0 elsewhere; at the E node it is ``g g^T`` wherever the
  wall cuts the voxel's square (``f_up`` fractional) with no tangent-plane test, because
  production's node rule reads the nearest face's normal ungated (plan v3 gated both; at T' 40
  two rim nodes sit in J with the wall's plane missing their voxel, and gated they missed the
  ``alpha -> 0`` limit by 1.9e-3), and 0 where ``f_up`` is 0 or 1, where production's contrast
  term is exactly 0 (a round-off cross coefficient there is refused at the term box's edge by
  ``constitutive.validate``). A Box takes ``(M_uu, M_vv)/L`` from
  ``polygon_edge_moments`` without the cap term, gated the same way (a Box plans no node term).
* ``1b``: 2 on J2; on J3 the medium owning the Yee point (the substrate below the plane, else
  the pillar where its signed distance is <= 0, else the background), zero cross.
* ``1a``: the owning medium on all of J.
* ``3w``: door 5 held open for the declared pair, no patch: today's painters in scene order.
* ``PA`` (anchor instrument): ``1.1 Xi_2`` on J, tensor and cross too, ``1/held`` unchanged.
* ``P3`` (rim instrument): arm 2 with ``(1 + 0.05 n/n0) Xi_2`` on J3, ``1/held`` unchanged.
* ``1bL``/``1bR`` (ridge diagnostics): 1b on one wall's J3 only.
* ``none``: no patch at all and door 5 not held (the junction-free controls K0, K1 and W): a
  control that reached J would be refused, and its V4 asks for a census of 0.

**How the arms reach the kernel** (:func:`patched`). Module attributes are wrapped and restored
in ``finally``: ``checks._shared_surface_voxels`` holds door 5 open for the declared (substrate,
pillar) pair only, matched by type and geometry, and records its count; ``coefficients.painters``
records the grid of the lowering in progress (plan time lowers too, for the certificate);
``coefficients._blend`` writes the arm's ``eps`` on the write set for field E, rule 'mix';
``constitutive.lower`` (full signature) writes the node tensor, cross and ``1/held`` there;
``wavebit.fdtd._execution.lower`` digests what the executor receives; ``_run.execute`` taps the
monitor planes' field norms. ``coefficients.build`` is never wrapped (it is bound by name in
``params.py`` and ``constitutive.py``). Each arm is planned once with ``_run._planned`` (its
certificate is arm-specific) and solved with ``_run._solve``. The patch is host NumPy: ``fd.run``
lowers eagerly and jits only the executor, so JAX runs see it; under ``jit``/``grad`` it fails
loudly on the tracer, and it is not used there. An exception raised by the harness's own code
inside a wrapper is re-raised as :class:`HarnessError` and recorded, so a row's ``status`` can
tell a refusal by wavebit (``'refused'``: a ``ValueError`` raised in the package) from a harness
or machine failure (``'error'``, solved again when the command is run again) even where
production turns the failure into a ValueError of its own (``constitutive.validate`` does).

**Instruments** (plan v3 sections 5, 10). V1: both monitors' decay below 1e-5 and every
amplitude and monitor-plane series finite (production reports a NaN monitor as quiet, decay 0).
V2: the lift certified where a term exists (the run-time guard), the CFL bound on the lowered
diagonal and its Gershgorin enclosure, no NaN, no GrowingFieldWarning and no overflow, and each
monitor plane's final ``sum |F|^2`` at most :data:`GROWTH` of its peak (a growing field ends at
its peak; the literal "at most its peak" cannot fail). V3: the pre-lift ``eps`` bit-identical
to 3w outside the write set and equal to the general form on J to 1e-13, nodes likewise, with
the E edge families and (where a term exists) the nodes actually tapped and their J counts
equal to the census, and the executor's parameter digest equal to a host NumPy lowering of the
same arm (bitwise on NumPy, 1e-10 of the scale on JAX). V4: door 5's count equals the study's
census. V5 (host): arm 2's coefficients continuous in the radius, the lateral offset and the
flush pair's height except at production's own tangent-plane switches. V6 (host): arm 2
lowered with the structures listed in either order is bit-identical. Every maximum here
propagates NaN (:func:`worst`), and every pass needs a finite number.

**The rule** (plan v3 section 11, registered before the campaign):

Decision rule (registered before the campaign; the pilot runs anchor_fd, ridge_fd, ridge_walls,
oblique_map, pml_floor and k0_oblique informed its thresholds and are not part of it). Junction
voxels J are door 5's own set (the substrate's and the pillar's exact fills both in (1e-9, 1 -
1e-9)); J3 are those where the pillar's transverse fill is also fractional (the rim) and J2 the
rest. The candidates are 2, the split (the laminate of the pure substrate, fraction alpha, under
the upper part's pillar/background two-phase tensor whose wall weight comes from the side wall
alone and is zero where production's tangent-plane test finds no wall in the voxel, painted as
that tensor's diagonal at each E component's own voxel and as the full tensor with 1/held =
1/<eps> at the E node); 1b (2 on J2, the medium owning the Yee point on J3); 1a (the medium owning
the Yee point on all of J); and 3w (doors held open, today's painters in scene order, substrate
first). The default must be continuous in the geometry and independent of the order the
structures are listed in, so that jax.grad and a finite difference compute one function; of the
candidates only 2 is both, so the measurement decides only whether 2 becomes the default,
'refuse' stays the default, or the numbers go to the owner, and total errors against RCWA are
reported for every arm and never rank them. The steps apply in this order. (1) Anchor: on a
laterally uniform film of each scene's pillar material flush on its substrate (both smoothed
unturned fd.Box in a one-cell column, door 5 held open for that pair only, 24-cell PML matched to
the substrate, 80 periods), at normal incidence and at 45 degrees p, with the substrate plane
0.25 and 0.75 cell above a node plane, over rungs n0, 2n0, 4n0 and 8n0 (n0 the lowest rung the
8-cells-per-wavelength door admits), arm 2's largest error over the three wavelengths against
the exact unpatterned-layer answer must fit a least-squares log-log order of at least 1.8 over
the rungs where the bare-substrate control's ratio K0/e_2 has not grown more than threefold from
n0 (at least three rungs), and the planted arm P_A (arm 2's inverse permittivity, node tensor and
cross term times 1.1 on J) must fit an order below 1.7 for its junction term d(n), the largest
over the wavelengths of |r - r_2| and |t - t_2| on the same grid, fitted over all four rungs;
otherwise the study stops without a decision; the same statistic is reported for 1a and 3w. (2)
Validity: if any run of arm 2 on the pillar scenes A, B, C', D', T' and the ridge R (s and p;
placements 0.25 and 0.75; the registered ladders) is invalid -- both monitors' decay not below
1e-5 at 60 periods nor at 120 on one repeat, the operator uncertified where a constitutive term
exists, the CFL bound broken, a NaN, a final field norm above its peak, coefficients not
bit-identical to 3w outside J or not equal to the closed form inside J to 1e-13, an executor
parameter digest different from Phase 0's, a door-5 count different from the census, or a failed
host continuity or listing-order probe -- 'refuse' stays the default. (3) Consistency: a scene is
judgeable when the junction-free control K1 (the pillar lifted by one cell of the scene's lowest
rung, height kept) is consistent with its own RCWA twin, consistent meaning that at every cell
(placement, wavelength, t00 or r00, and polarisation on R) the RCWA value lies within 3 u_ref +
0.1 |T(n_f) - T(n_c)| of the segment of Richardson limits T(n_f) + (T(n_f) - T(n_c))/((n_f/n_c)^p
- 1), p from 1.4 to 2, n_f the finest registered rung and n_c the finest rung at most n_f/1.5,
u_ref the scene's largest distance of the N = 11, 13 and 15 amplitudes from N = 17 over the three
wavelengths; if fewer than two of the five three-dimensional scenes are judgeable the rule does
not choose; if arm 2 is inconsistent on a judgeable scene, the rule does not choose when 1b is
consistent there, and otherwise 'refuse' stays the default. (4) Rim: for each scene, placement
and (on R) polarisation, the rim has power when the planted arm P3 (arm 2 with its J3 inverse
permittivity, node tensor and cross term multiplied by 1 + 0.05 n/n0) gives a difference max
over cells |T_P3 - T_2| whose least-squares log-log order over the registered rungs has
log-residual RMS at most 0.15, whose finest-rung value is at least ten times the run-length floor
F (ten times arm 2's largest 60-versus-120-period change at the second rung, and at least 1e-7),
and whose order is at least 0.6 below p_sigma, the median over cells of arm 2's Richardson
observed order from its three finest rungs; the rule does not choose if no scene-placement has
rim power, or if on one that has it the rim difference max over cells |T_1b - T_2| is material
(its finest-rung value at least 0.05 of the largest |T_2(n_f) - T_2(n_c)|) and its fitted order
is unresolved (RMS above 0.15) or at most p_sigma - 0.3. (5) Otherwise the default is 'split'
(arm 2). Whenever the rule does not choose or 'refuse' stays, the switch is still offered with
these numbers.

**What phase 0 and its two reviews changed** (each with the check that now holds it):

* The node write set is J plus the plane's nodes whose pillar fill is a round-off 1e-16
  (:data:`NODE_EXTRA`): there production's node rule discards the substrate's contrast, and
  arm 2 inherited it outside J -- V5 read a node Xi_zz jump of 10.8% (C' 40) and 26.7% (T' 40)
  at the rim's entry. So V3's "outside J" is, for nodes, outside the write set.
* The base side (found by the review, held by V5's height sweep): moving the flush pair 0.002
  of a cell across a node voxel's face changed 1/eps at 2-4 rim edges and nodes by 22.5%
  (C' 40), 25.4% (T' 40) and 6.6% (B 24), because production's cylinder reads the base there
  and arm 2 the wall. Arm 2's value is the right one (the base lies on the voxel's face, the
  wall crosses it), so the write set takes those points too. Its control -- the same sweep
  without the base side -- must fail, and does (12 unexplained jumps, the 22% seam).
* The node's wall weight is ungated (found by the review, held by the same sweep): gated as
  plan v3 wrote it, T' 40's two tangent-plane-miss rim nodes missed production's node rule by
  1.9e-3 as the plane left their voxel. The control with the gated node must fail, and does.
  At the registered placements this moves arm 2's node tensor by at most 1.8e-3 (cross 4.2e-4)
  at two J nodes per placement at T' 40 and at C' 60, and by round-off everywhere else.
* V5's write-set-boundary equality holds to 1e-12 plus the band's Lipschitz term
  ``2 (eps_max/eps_min) b``, ``b = min(alpha, 1 - alpha, f, 1 - f)`` on the outside side: a
  voxel with a fill within 1e-9 of 0 or 1 is outside door 5's J and keeps today's mix, off the
  closed form by O(b) (up to 1.1e-8 measured, within 0.76 of the bound). Its step scale is arm
  2's own sweep, so a rule that only jumps is caught: arm 1b on the same sweep must fail, and
  does (197 unexplained jumps).
* V6 asks bit-identity of arm 2's own numbers (the pre-lift eps on the write set, its lifted
  value on J, the nodes) and 1e-14 elsewhere: outside J today's painters differ by a double
  reciprocal between the two listing orders on the cylinder scenes (up to 5.1e-15 of the
  array's scale, 24 entries at T' 40; the Box scenes are bitwise).
* The executor digest is compared with a host NumPy lowering of the same arm in every solve
  (bitwise on NumPy; on JAX to :data:`JAX_GAP` of each array's scale, since production's own
  lift and cross coefficients differ between the backends by up to 3e-13 of it at A 80 and
  T' 60, where a patch that missed the executor is an O(1) change) and, at the lowest rung,
  with phase 0's digest.

**Amendment 1 (2026-09-27, after the first anchor run; the rule is unchanged).** The first anchor
run stopped (report log, ``anchor|1790516169``): in three of the sixteen cases -- FC' at 45
degrees p, placement 0.75, and FT' at 45 degrees p, both placements -- fewer than three rungs
were floor-free. Arm 2's own error there fell at 2.01 / 2.00 / 2.00 (FC') and 2.02 / 2.00 / 1.07
(FT'); the bare-substrate control K0 flattened from 80 cells at 2.7e-4 (eps 2.1) and 1.2e-3
(eps 3.1). Diagnosed with ``floor_probe.py`` (``results/fdtd_subpixel_junction_floor.log``): the
floor is the z PML's round-trip return at the default target 1e-8 -- at 45 degrees p and 160
cells K0 is 2.67e-4 / 9.10e-5 / 9.08e-5 (eps 2.1) and 1.226e-3 / 1.580e-4 / 1.578e-4 (eps 3.1)
at targets 1e-8 / 1e-12 / 1e-16, and at 1e-12 it falls at second order from 80 cells (orders
1.99 and 2.00). So every oblique film case, every arm and the control alike, now runs with
``PML(cells=24, reflection=1e-12)`` (:data:`OBLIQUE_REFLECTION`); normal incidence and the
pillar scenes are unchanged. The gate, its thresholds and the floor rule are as registered. The
first run's rows stay in the log under the previous digest and its verdict in the report log.

**Reading the rule as code** (where the text leaves a choice, fixed here before the campaign):

* Fits: log-log least squares in natural logarithms, RMS in the same units, three rungs at
  least; q(a, 2) over the rungs whose difference is at least F_s; a NaN anywhere makes the
  order NaN, and a NaN order fails every test it enters.
* Anchor: F is ``max(1e-9, 10 max|T_2(80) - T_2(160)|)`` at 8 n0. The study stops for a case
  (film, incidence, placement) unless arm 2's p_2 >= 1.8 over at least three floor-free rungs
  and P_A's d fits an order below 1.7 -- section 11's words, whatever its RMS -- and exceeds F
  at one rung at least (below F everywhere, P_A has shown nothing). 1a and 3w are reported as
  second order on J2 when ``min(p_2, q_a) >= 1.7`` with RMS <= 0.1, or ``d_a < F`` everywhere.
* Pillar scenes: F_s comes from one 120-period repeat of arm 2 at the second rung, at placement
  0.25 (both polarisations on R), and serves both placements. u_ref is the largest distance of
  N = 11, 13, 15 from N = 17 over the three wavelengths; when it exceeds 1e-3 the shifted
  estimate, N = 15, 17, 19 against 21 at wavelength 1, is taken if larger, and the reference is
  the largest N computed. K1-flush and W-flush are reported de-embedded to the flush reference
  plane (``r exp(-2 i k0 g)``, ``t exp(-i k0 g)``, ``k0 = 2 pi/lambda``; exact on RCWA to 4e-16);
  K1 against its twin needs nothing, both refer to ``g + H``.
* Step 3 over several judgeable scenes: if any has arm 2 and 1b both inconsistent, 'refuse'
  stays, even where another has 1b consistent (refuse takes precedence).
* Rows: a missing row, or one whose status is 'error', leaves its step 'incomplete' (run the
  command again). A row refused by wavebit, or failing V1 (after its repeat at twice the
  periods, whose values then stand for it) or V2, is an invalid run: for arm 2 on the pillar
  scenes that is step 2; for every other arm its amplitudes become NaN, so it can only take a
  test's cautious side (not consistent, K1 not judgeable, no rim power, material and
  unresolved, the anchor stops). A row of any other arm failing V3 or V4 means the harness did
  not realise that arm: the report says 'incomplete (harness)'.
* Phase 0 gates the report: every phase-0 row it expects must be present under its current
  digest and pass (a mutant must also have applied). A failed V5 or V6 probe is step 2's
  'refuse'; any other missing or failed row is 'incomplete'.
* Completeness of step 3 includes the K1 rows, the full RCWA ladders at the three wavelengths
  (N = 19/21 at wavelength 1 when u_ref > 1e-3; the K1 twin at 19/21 when its 15/17 spread
  exceeds 2 u_ref), and each K1 twin's total thickness equal to its fd row's top.

**Engines.** NumPy below 48 cells per wavelength and JAX (CPU, x64, no ``XLA_FLAGS``) at 48
and above, on every scene. Measured in phase 0: a one-column film at 320 cells takes 33.9 s on
NumPy (per-step overhead) and 11.9 s on JAX, 2.6 s for the next arm on the same plan, whose
compiled executor JAX keeps; the two agree to 1.3e-15.

Run with the snapshot on ``SNAP`` (or from the repository, where ``SNAP`` defaults to the root)::

    python fdtd_subpixel_junction.py phase0 [--part P ...]   # checks, V3-V6, mutants, smoke, timing
    python fdtd_subpixel_junction.py rcwa [--scene S]        # phase 1
    python fdtd_subpixel_junction.py anchor                  # phase 2, stops if step (1) fails
    python fdtd_subpixel_junction.py pillars [--scene S] [--minimal]    # phase 3
    python fdtd_subpixel_junction.py report [--minimal]      # phase 4: section 11 applied by script
    python fdtd_subpixel_junction.py status [--minimal]      # what each phase's rows hold now

The parts are provenance, v4, v3, v5 (radius, offset), v5z (height), v6, mutants, smoke, agree,
timing-low, timing-jax, timing-rcwa and estimate. ``anchor`` will not start unless every row
phase 0's gate expects is present and passed, and ``pillars`` not unless the anchor passed too
(``--anyway`` overrides both); ``--minimal`` must be given to ``pillars`` and ``report`` alike.

Rows go to ``results/fdtd_subpixel_junction.log`` (solves and phase-0 checks), ``..._rcwa.log``
and ``..._report.log`` as JSON lines, appended; a row already present with the same key and the
same provenance digest is skipped (unless its status is 'error'), so every command resumes. The
solve digest covers the SHA-256 of every ``.py`` file of ``src/wavebit/fdtd`` (line endings
normalised), this file's source above the phase-0 section (every constant, builder, arm and
wrapper), and the NumPy/JAX versions; a phase-0 row adds its own check's source; the RCWA rows
carry their own digest (``rcwa_point``, the K1 gap, the scenes, the rest of ``wavebit``).
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))
from tools.repository_paths import repository_path
import cmath
import contextlib
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

HERE = repository_path('studies')
ROOT = HERE.parent
SNAP = Path(os.environ.get('SNAP', str(ROOT))).resolve()
sys.path.insert(0, str(SNAP))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
os.environ.setdefault('JAX_ENABLE_X64', '1')      # before JAX is imported: x64 is a cap
import numpy as np                                                                # noqa: E402
import wavebit.fdtd._legacy as fd                                                           # noqa: E402
from wavebit.fdtd import _execution as _run                                                       # noqa: E402
from wavebit.fdtd._kernel.grid import Grid                                          # noqa: E402
from wavebit.fdtd._kernel.observe import InteriorEnergy                             # noqa: E402
from wavebit.fdtd._kernel.stencil import StencilTerm                                # noqa: E402
from wavebit.fdtd._prepare import checks, coefficients, constitutive                 # noqa: E402
from wavebit.fdtd._prepare import params as lowering                                  # noqa: E402
from wavebit.fdtd._prepare.occupancy import yee_points                                # noqa: E402
from wavebit.fdtd._runtime import runtime                                           # noqa: E402
from wavebit.fdtd._shapes import occupancy as body_occupancy                        # noqa: E402
from wavebit.fdtd._shapes._cut import _LIVE_AREA, _mean_of_the_moment, _parts       # noqa: E402
from wavebit.fdtd._shapes._exact import polygon_edge_moments                        # noqa: E402

if not os.path.normcase(str(Path(fd.__file__).resolve())).startswith(os.path.normcase(str(SNAP))):
    raise SystemExit(f'wavebit.fdtd._legacy is imported from {fd.__file__}, not from the snapshot {SNAP}')

RESULTS = HERE / 'results'
LOG = RESULTS / 'fdtd_subpixel_junction.log'
RCWA_LOG = RESULTS / 'fdtd_subpixel_junction_rcwa.log'
REPORT_LOG = RESULTS / 'fdtd_subpixel_junction_report.log'

TOL = checks._CLIP_ROUNDOFF
WAVELENGTHS = (0.95, 1.0, 1.05)
ZETAS = (0.25, 0.75)
OBLIQUE = 45.0            # the anchor's p incidence; at the grazing door's edge, never raised
#: The z PML's target on the oblique film cases (amendment 1 in the module docstring): at the
#: default 1e-8 its round-trip return floored K0 at 2.7e-4 / 1.2e-3 from 80 cells; at 1e-12 K0
#: is second order to 160 cells, and 1e-16 moves it by 2e-7.
OBLIQUE_REFLECTION = 1e-12
SHEET, GAP_R, GAP_T, CLEAR = 0.6, 0.5, 0.5, 0.3
WRITING = ('2', '1b', '1a', 'PA', 'P3', '1bL', '1bR')
#: Phase 0 (V5) found it: at the plane's nodes outside J whose pillar fill is a round-off
#: 1e-16, production's node rule discards the substrate's contrast (Xi_zz = 1/<eps> where the
#: voxel's <1/eps> belongs), 3 such nodes per placement at C' 40 and 13 at T' 40. The patch
#: writes arm 2's closed form there too (the substrate's own tensor to round-off), so the
#: node write set is J plus those nodes; the edges' mix is continuous at a zero fill and keeps J.
NODE_EXTRA = True
#: The executor's parameters on JAX against the host NumPy lowering, relative to each array's
#: largest entry (phase 0: up to 3.0e-13 at A 80 and T' 60, 1e-16 on the Box scenes).
JAX_GAP = 1e-10
#: V2's growth line: each monitor plane's final ``sum |F|^2`` over its peak. V1's line is 1e-5
#: of the peak for the change over the last period, so a run V1 accepts sits two decades under
#: this one (phase 0's smoke rows end at 5e-11 .. 1e-10), and a growing field ends at its peak.
GROWTH = 1e-3


from studies._support._fdtd_legacy import legacy_rcwa  # noqa: E402
class HarnessError(Exception):
    """Raised by this file's own code inside a patched call: never a refusal by wavebit. Not a
    RuntimeError, TypeError or ValueError, which production turns into a refusal (a failed
    lowering at plan time becomes constitutive.validate's ValueError); and recorded in
    :data:`FAULTS` as well, in case something catches it all the same."""


#: The harness errors raised since the current solve began (:func:`solve_row` clears it).
FAULTS = []


def worst(*values):
    """The largest of ``values`` (numbers or arrays), NaN if any is NaN; 0.0 for none.

    Python's ``max(0.0, nan)`` is 0.0: every maximum a check passes on goes through here."""
    flat = [np.ravel(np.asarray(v, dtype=np.float64)) for v in values]
    flat = [v for v in flat if v.size]
    return float(np.max(np.concatenate(flat))) if flat else 0.0


def least(*values):
    """The smallest of ``values``, NaN if any is NaN."""
    return -worst(*(-np.asarray(v, dtype=np.float64) for v in values))


#: name: kind, period, size (diameter | side | width), height, eps pillar / substrate, the axis
#: offset from the node line nearest the cell centre in cells, the core rungs (cells per vacuum
#: wavelength at the band centre; plan v3 section 3).
SCENES = {
    'A': dict(kind='cyl', P=0.50, size=0.29, H=0.60, eps_p=4.0, eps_s=2.25, o=(0.0, 0.0),
              rungs=(20, 40, 80)),
    'B': dict(kind='cyl', P=0.50, size=0.26, H=0.625, eps_p=5.8, eps_s=2.13, o=(0.25, 0.25),
              rungs=(24, 48, 96)),
    "C'": dict(kind='cyl', P=0.50, size=0.20, H=0.30, eps_p=12.0, eps_s=2.10, o=(0.25, 0.25),
               rungs=(40, 60, 80, 120)),
    "D'": dict(kind='box', P=0.50, size=0.15, H=0.30, eps_p=12.0, eps_s=2.10, o=(0.25, 0.25),
               rungs=(40, 80, 120)),
    "T'": dict(kind='cyl', P=0.45, size=0.30, H=0.15, eps_p=12.0, eps_s=3.10, o=(0.25, 0.25),
               rungs=(40, 60, 80)),
    'R': dict(kind='ridge', P=0.50, size=0.15, H=0.20, eps_p=12.0, eps_s=2.10, o=(0.25, 0.0),
              rungs=(40, 80, 160, 320)),
}
#: The anchor's films: the scene whose materials and height they take. Rungs n0 .. 8 n0.
FILMS = {'FA': 'A', 'FB': 'B', "FC'": "C'", "FT'": "T'"}
MINIMAL_DROPS = {"C'": 120, "D'": 120}
ALIASES = {'Cp': "C'", 'Dp': "D'", 'Tp': "T'", 'FCp': "FC'", 'FTp': "FT'"}

#: The ten lowering files and their LF-normalised SHA-256 (first 16) at commit adb2ada.
FILES = ('_lower/checks.py', '_lower/coefficients.py', '_lower/constitutive.py', '_lower/stars.py',
         '_shapes/_bodies.py', '_shapes/_exact.py', '_shapes/_cut.py', '_lower/params.py',
         '_run.py', '_kernel/constitutive.py')
AT_1C26BF3 = ('31e687f4fc897d29', '42c296466f5001f6', '840fab8b0963e80b', '613650f52641c7ad',
              '17baf9d581d5d187', '71bd32cac777037e', '509b505f0ce9ff92', '12d8518672638238',
              '0ffa6ad8ffc1d4ec', '5ecc1a369ec9a74e')


# -- scenes ------------------------------------------------------------------------------------

def rungs_of(scene, minimal=False):
    if scene in FILMS:
        n0 = SCENES[FILMS[scene]]['rungs'][0]
        return (n0, 2 * n0, 4 * n0, 8 * n0)
    return tuple(n for n in SCENES[scene]['rungs']
                 if not (minimal and MINIMAL_DROPS.get(scene) == n))


def richardson_pair(rungs):
    """``(n_c, n_f)``: the finest rung and the finest one at most ``n_f/1.5``."""
    n_f = max(rungs)
    return max(n for n in rungs if n <= n_f / 1.5), n_f


def engine(scene, n):
    """NumPy below 48 cells per wavelength, JAX at 48 and above, every scene (section 12)."""
    return 'jax' if n >= 48 else 'numpy'


def spec(scene, n, zeta, arm, variant='flush', pol='s', theta=0.0, periods=None, backend=None):
    return dict(scene=scene, n=n, zeta=zeta, arm=arm, variant=variant, pol=pol, theta=theta,
                periods=periods or (80 if scene in FILMS else 60),
                backend=backend or engine(scene, n))


def key_of(s):
    return '|'.join(f'{k}={s[k]}' for k in ('scene', 'variant', 'n', 'zeta', 'arm', 'pol',
                                             'theta', 'periods', 'backend'))


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


def k1_gap(name):
    """K1's lift: one cell of the scene's lowest rung. The fd builder and the RCWA twin both
    take it from here, and the report checks the twin's thickness against the fd row's top."""
    return 1.0 / SCENES[name]['rungs'][0]


def shapes(name, info, dz=0.0, radius=None, dxy=(0.0, 0.0)):
    """``(substrate, pillar)`` shapes of ``info``'s scene, both moved up by ``dz`` (the flush
    pair's height, V5) and the pillar by ``dxy`` sideways with its radius replaced (V5)."""
    g = SCENES[name]
    deep, H, zc = info['deep'], g['H'], info['base'] + g['H'] / 2 + dz
    sub = fd.Box(size=(10.0, 10.0, abs(deep)), center=(0.0, 0.0, deep / 2 + dz))
    xc, yc = info['xc'] + dxy[0], info['yc'] + dxy[1]
    if info['film']:
        pil = fd.Box(size=(10.0, 10.0, H), center=(0.0, 0.0, zc))
    elif g['kind'] == 'cyl':
        pil = fd.Cylinder(radius=g['size'] / 2 if radius is None else radius, height=H, axis='z',
                          center=(xc, yc, zc))
    elif g['kind'] == 'box':
        pil = fd.Box(size=(g['size'], g['size'], H), center=(xc, yc, zc))
    else:
        pil = fd.Box(size=(g['size'], 10.0, H), center=(xc, 0.0, zc))
    return sub, pil


def build(name, n, zeta, *, variant='flush', film=False, theta=0.0, pol='s', periods=None,
          reverse=False):
    """``(scene, info)``. ``variant``: 'flush' | 'k1' (lifted by :func:`k1_gap`, height kept) |
    'w' (lifted by one cell of this rung) | 'bare' (substrate only, K0).
    """
    g = SCENES[name]
    P, H, ep, es = g['P'], g['H'], g['eps_p'], g['eps_s']
    pml = 24 if film else 12              # matched to the substrate either way
    reflection = OBLIQUE_REFLECTION if (film and theta) else 1e-8     # 1e-8: fd.PML's default
    periods = periods or (80 if film else 60)
    base = {'k1': k1_gap(name), 'w': 1.0 / n}.get(variant, 0.0)
    top = base + H
    sheet = top + SHEET
    r_plane, t_plane = sheet + GAP_R, -GAP_T
    dx, z_lo, z_hi = z_grid(n, zeta, t_plane - CLEAR, r_plane + CLEAR, pml)
    glass = fd.Material(eps=es)
    cells = round(P * n)
    if abs(P * n - cells) > 1e-9:
        raise AssertionError((name, n))
    xc = 0.0 if film else -P / 2 + (round(cells / 2) + g['o'][0]) * dx
    yc = 0.0 if film or g['kind'] == 'ridge' else -P / 2 + (round(cells / 2) + g['o'][1]) * dx
    kind = 'cyl' if (g['kind'] == 'cyl' and not film) else 'box'
    info = dict(name=name, n=n, zeta=zeta, variant=variant, film=film, kind=kind, dx=dx,
                z_lo=z_lo, top=top, base=base, xc=xc, yc=yc, eps=(es, ep, 1.0), plane=0.0,
                deep=z_lo - 2 * dx,       # the substrate runs through the z- PML
                n0=g['rungs'][0])
    sub_shape, pil_shape = shapes(name, info)
    sub = fd.Structure(sub_shape, glass)
    pil = None if variant == 'bare' else fd.Structure(pil_shape, fd.Material(eps=ep))
    structures = (sub,) if pil is None else ((pil, sub) if reverse else (sub, pil))
    pw = fd.PlaneWave('pw', z=sheet, direction='-z', theta=theta, phi=0.0, polarization=pol,
                      band=fd.Band(wavelength=1.0, df=0.06))
    ref_top = 0.0 if variant == 'bare' else top
    monitors = (fd.Diffraction('r', z=r_plane, side='reflect', wavelengths=WAVELENGTHS,
                               reference_plane=ref_top, incident_plane=ref_top,
                               polarizations='given'),
                fd.Diffraction('t', z=t_plane, side='transmit', wavelengths=WAVELENGTHS,
                               reference_plane=0.0, incident_plane=ref_top,
                               polarizations='given'))
    faces = (fd.Bloch.from_source(pw, axis1d='x') if film and theta else
             fd.Bloch.from_source(pw))
    one = film or g['kind'] == 'ridge'
    scene = fd.Scene(domain=fd.Domain(size=(dx if film else P, dx if one else P, z_hi - z_lo),
                                      center=(0.0, 0.0, 0.5 * (z_hi + z_lo))),
                     boundaries=fd.Boundaries(**faces, z=(fd.PML(cells=pml, background=glass,
                                                                  reflection=reflection),
                                                          fd.PML(cells=pml, reflection=reflection))),
                     resolution=fd.Resolution(dx=dx), run=fd.Run(periods=periods),
                     structures=structures, sources=(pw,), monitors=monitors)
    info.update(sub=sub, pil=pil)
    info['bodies'] = (sub.shape.body(), None if pil is None else pil.shape.body())
    return scene, info


def make(s, **extra):
    film = s['scene'] in FILMS
    name = FILMS[s['scene']] if film else s['scene']
    return build(name, s['n'], s['zeta'], variant=s['variant'], film=film, theta=s['theta'],
                 pol=s['pol'], periods=s['periods'], **extra)


# -- geometry and the arms (host NumPy; the kernel's own primitives) -------------------------------

def geometry(info, points, dx, mutant=None, legacy=None):
    """Per point: the substrate's and the pillar's exact fills, ``f_up``, the wall weight
    ``P_up`` (xx, yy, xy) gated for the edges and ungated for the node, production's gate, the
    owning medium, the J / J3 masks and the two write sets outside J (module docstring).

    ``legacy`` rebuilds a rejected arm for V5's controls: 'no-base-side' drops the base side,
    'gated-node' gives the node the edges' gated weight (plan v3 as written)."""
    sub, pil = info['bodies']
    es, ep, eg = info['eps']
    alpha = np.asarray(body_occupancy(sub, points, dx)[0])
    fill = np.asarray(body_occupancy(pil, points, dx)[0])
    _, _, box, _ = pil._boxes(points, dx)
    f_up = np.asarray(pil._area(box, points)) / dx ** 2
    distance, normal = (np.asarray(v) for v in pil.surface(points))
    frac = lambda v: (v > TOL) & (v < 1 - TOL)                               # noqa: E731
    if mutant == 'm3':                    # the mutant: the wall weight from surface()
        live = np.ones(alpha.shape, bool)
        pxx, pyy, pxy = normal[..., 0] ** 2, normal[..., 1] ** 2, normal[..., 0] * normal[..., 1]
        node = (pxx, pyy, pxy)
    elif info['kind'] == 'cyl':
        # The wall's own tangent plane, through production's own test: w = 0 where it misses.
        dw, g = (np.asarray(v) for v in pil._transverse_surface(points[..., :2]))
        plane = np.concatenate([g, np.zeros_like(g[..., :1])], axis=-1)
        _, area, moment = (np.asarray(v) for v in _parts(dw, plane, dx))
        w = np.asarray(_mean_of_the_moment(area, moment, dx, np))
        live = area > _LIVE_AREA * dx * dx
        pxx, pyy, pxy = w[..., 0], w[..., 1], np.where(live, g[..., 0] * g[..., 1], 0.0)
        # The node: production's node rule (constitutive.lower) reads the nearest face's normal
        # with no tangent-plane test, so arm 2's alpha -> 0 limit is g g^T ungated wherever the
        # wall cuts the voxel's square (f_up fractional), and 0 where it does not -- there
        # production's contrast term is exactly 0, and so must the cross coefficient be (a
        # round-off one at the term box's edge is refused by constitutive.validate).
        rim = frac(f_up)
        node = ((pxx, pyy, pxy) if legacy == 'gated-node' else
                tuple(np.where(rim, v, 0.0) for v in (g[..., 0] ** 2, g[..., 1] ** 2,
                                                       g[..., 0] * g[..., 1])))
    else:
        # A full-height upper column: production's own moment with along = dx and no cap.
        length, muu, mvv = (np.asarray(v) for v in
                            polygon_edge_moments(pil._vertices(points), *box, np))
        w = np.asarray(_mean_of_the_moment(length * dx, np.stack([muu * dx, mvv * dx, 0 * muu],
                                                                  axis=-1), dx, np))
        live = length * dx > _LIVE_AREA * dx * dx
        pxx, pyy, pxy = w[..., 0], w[..., 1], np.zeros_like(muu)
        node = (pxx, pyy, pxy)            # a Box plans no constitutive term: never lowered
    J = frac(alpha) & frac(fill)
    owner = np.where(points[..., 2] < info['plane'], es, np.where(distance <= 0, ep, eg))
    # Outside J, where the pillar's exact fill is a round-off 1e-16 and not 0, production's node
    # branch (constitutive.lower: ``fill == 0``) composes the two bodies and drops the
    # substrate's contrast. The node patch covers these too (NODE_EXTRA).
    extra = frac(alpha) & (fill != 0) & ~J
    # The base side: a flush cylinder's rim point with the substrate's fill within 1e-9 of 0,
    # where production's ``surface()`` picks the pillar's base as the nearest face. The base's
    # plane lies on the voxel's face or under it, so production gives the wall no weight there,
    # and jumps to the wall's weight where the wall becomes the nearer face; arm 2's alpha -> 0
    # limit is the wall's weight throughout. The review measured the jump at 22.5% (C' 40). A
    # lifted pillar (K1, W) has a real base over air: nothing of it is a junction.
    base = np.zeros(alpha.shape, bool)
    if (info['kind'] == 'cyl' and info['variant'] == 'flush' and mutant != 'm3'
            and legacy != 'no-base-side'):
        base = (alpha <= TOL) & frac(fill) & (normal[..., 2] < 0) & ~J
    return dict(kind=info['kind'], alpha=alpha, fill=fill, f_up=f_up, P=(pxx, pyy, pxy),
                P_node=node, live=live, J=J, J3=J & frac(f_up), extra=extra, base=base,
                owner=owner, left=points[..., 0] < info['xc'])


def split(geo, eps):
    """Arm 2 in closed form: ``(Xi diagonal (3, ...), cross (xy, xz, yz), 1/held)``."""
    es, ep, eg = eps
    al, fu = geo['alpha'], geo['f_up']
    pxx, pyy, pxy = geo['P']
    a = fu * ep + (1 - fu) * eg
    eta = fu / ep + (1 - fu) / eg
    A = (1 - al) * a + al * es
    B = (1 - al) / eta + al * es
    if geo['kind'] == 'cyl':              # rank-1 (or gated-off) P_up: the full-tensor diagonal
        xx, yy, xy = (1 - pxx) / A + pxx / B, (1 - pyy) / A + pyy / B, (1 / B - 1 / A) * pxy
    else:                                 # diagonal P_up: the Box spelling, x and y only
        xx = 1 / ((1 - al) * coefficients.mix(np, eg, ep, fu, pxx) + al * es)
        yy = 1 / ((1 - al) * coefficients.mix(np, eg, ep, fu, pyy) + al * es)
        xy = np.zeros_like(al)
    zz = (1 - al) / a + al / es           # always harmonic along the plane normal (Cp6)
    zero = np.zeros_like(al)
    return np.stack([xx, yy, zz]), np.stack([xy, zero, zero]), 1 / A


def general(geo, eps):
    """Arm 2 by 2x2 inverses, ``[(1-alpha) Xi_up,t^-1 + alpha eps_s I]^-1``: V3's second route."""
    es, ep, eg = eps
    al, fu = geo['alpha'], geo['f_up']
    pxx, pyy, pxy = geo['P']
    a = fu * ep + (1 - fu) * eg
    eta = fu / ep + (1 - fu) / eg
    P = np.stack([np.stack([pxx, pxy], -1), np.stack([pxy, pyy], -1)], -2)
    up = (np.eye(2) - P) / a[..., None, None] + eta[..., None, None] * P
    xt = np.linalg.inv((1 - al)[..., None, None] * np.linalg.inv(up)
                       + (al * es)[..., None, None] * np.eye(2))
    zero = np.zeros_like(al)
    return (np.stack([xt[..., 0, 0], xt[..., 1, 1], (1 - al) / a + al / es]),
            np.stack([xt[..., 0, 1], zero, zero]), 1 / ((1 - al) * a + al * es))


def arm_tables(arm, geo, eps, scale, form=split):
    """``(Xi, cross, 1/held)`` of ``arm`` at every point; the caller writes them on J only."""
    xi, cross, inv = form(geo, eps)
    J, J3 = geo['J'], geo['J3']
    own = {'1a': J, '1b': J3, '1bL': J3 & geo['left'], '1bR': J3 & ~geo['left']}.get(arm)
    if own is not None:                   # the medium at the Yee point; no cross coefficient
        o = 1 / geo['owner']
        xi, cross, inv = np.where(own, o, xi), np.where(own, 0.0, cross), np.where(own, o, inv)
    s = {'PA': np.where(J, 1.1, 1.0), 'P3': np.where(J3, scale, 1.0)}.get(arm)
    if s is not None:
        xi, cross = xi * s, cross * s
    return xi, cross, inv


def plane_rows(grid, xbox):
    """``xbox`` (x-first) cut to the rows around the plane z = 0: no J point lies elsewhere."""
    kp = int(math.floor(-grid.low[2] / grid.dx + 1e-9))
    (i0, i1), (j0, j1), (k0, k1) = xbox
    return (i0, i1), (j0, j1), (max(k0, kp - 1), min(k1, kp + 3))


def layer_geometry(info, grid, xbox, field, c, mutant=None, legacy=None):
    """``(rows, geometry)`` of family ``c`` (3: the common point) on the plane rows of ``xbox``;
    the node's ``P`` is its own (ungated) weight."""
    rows = plane_rows(grid, xbox)
    pts = (yee_points(grid, rows, field, c) if c < 3
           else constitutive.common_points(grid, rows, field))
    geo = geometry(info, pts, grid.dx, mutant, legacy)
    return rows, (dict(geo, P=geo['P_node']) if c == 3 else geo)


def write_set(geo, c):
    """Where a writing arm writes: J, the base side, and on the node :data:`NODE_EXTRA`'s set."""
    return geo['J'] | geo['base'] | (geo['extra'] if (c == 3 and NODE_EXTRA) else False)


def layer_tables(info, grid, xbox, field, c, arm, mutant=None, form=split):
    """``{write, J, xi, cross, inv}`` shaped like ``xbox``'s region, filled on the plane rows."""
    shape = tuple(hi - lo for lo, hi in reversed(xbox))
    out = dict(write=np.zeros(shape, bool), J=np.zeros(shape, bool), xi=np.ones((3,) + shape),
               cross=np.zeros((3,) + shape), inv=np.ones(shape))
    rows, geo = layer_geometry(info, grid, xbox, field, c, mutant)
    if geo['J'].size:
        xi, cross, inv = arm_tables(arm, geo, info['eps'], 1 + 0.05 * info['n'] / info['n0'], form)
        view = tuple(slice(lo - b[0], hi - b[0]) for (lo, hi), b in zip(reversed(rows),
                                                                       reversed(xbox)))
        out['write'][view], out['J'][view] = write_set(geo, c), geo['J']
        out['xi'][(slice(None),) + view], out['cross'][(slice(None),) + view] = xi, cross
        out['inv'][view] = inv
    return out


def census(info, grid):
    """J and J3 per point family over door 5's shared box: E, H, and the two common points."""
    if info['pil'] is None:
        return dict(total=0, J3=0, fam={})
    a, b = info['sub'].shape.support(grid), info['pil'].shape.support(grid)
    shared = tuple((max(u[0], v[0]), min(u[1], v[1])) for u, v in zip(a, b))
    fam, total, j3 = {}, 0, 0
    if all(hi > lo for lo, hi in shared):
        for field in 'EH':
            for c in range(4):
                _, geo = layer_geometry(info, grid, shared, field, c)
                fam[field + 'xyz*'[c]] = [int(geo['J'].sum()), int(geo['J3'].sum())]
                total, j3 = total + fam[field + 'xyz*'[c]][0], j3 + fam[field + 'xyz*'[c]][1]
    return dict(total=total, J3=j3, fam=fam)


# -- the patch -------------------------------------------------------------------------------

def _same_body(a, b):
    if a is None or b is None or type(a) is not type(b) or len(a.leaves()) != len(b.leaves()):
        return False
    try:
        return all(np.array_equal(np.asarray(x, float), np.asarray(y, float))
                   for x, y in zip(a.leaves(), b.leaves()))
    except (TypeError, ValueError):
        return False


class _Patch:
    """What one patched lowering needs and what it records."""

    def __init__(self, arm, info, mutant):
        self.arm, self.info, self.mutant = arm, info, mutant
        self.active, self.grid, self.tap = True, None, None
        self.door, self.exec, self.obs, self._cache = [], [], [], {}

    def writes(self):
        return self.active and self.arm in WRITING and self.info['pil'] is not None

    def declared(self, a, b):
        sub, pil = self.info['bodies']
        return ((_same_body(a, sub) and _same_body(b, pil))
                or (_same_body(a, pil) and _same_body(b, sub)))

    def layer(self, grid, xbox, c):
        key = (grid, xbox, c)
        if key not in self._cache:
            self._cache[key] = layer_tables(self.info, grid, xbox, 'E', c, self.arm, self.mutant)
        return self._cache[key]


def _digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f'{name}{a.shape}{a.dtype}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _executor_view(scene, plan, params):
    """The leaves the step multiplies by, as NumPy, their digest, dtype and CFL ratio dt/bound."""
    arrays = {'be': np.array(np.asarray(params['be'])), 'bh': np.array(np.asarray(params['bh']))}
    cross = {k: np.array(np.asarray(v)) for k, v in (params.get('constitutive') or {}).items()}
    arrays.update({'constitutive:' + k: v for k, v in cross.items()})
    ratio = plan.dt / plan.dx
    be = np.broadcast_to(arrays['be'], (3, plan.nz, plan.ny, plan.nx))
    upper = float(np.max(be)) / ratio
    for term in plan.constitutive:
        if isinstance(term, StencilTerm) or term.field != 'E' or term.name not in cross:
            continue
        c = cross[term.name]
        diagonal = np.broadcast_to(InteriorEnergy._cut(be, slice(None), term.box) / term.ratio,
                                   c.shape)
        upper = worst(upper, constitutive.enclosure(diagonal, c, np, 'E')[1])
    bound = 1 / math.sqrt(checks.cfl_axis_sum(scene, plan.dx) * upper * 1.0)   # mu = 1
    return dict(sha=_digest(arrays), dtype=str(arrays['be'].dtype), arrays=arrays,
                cfl=plan.dt / bound)


def _field_norms(plan, state):
    """Each monitor plane's ``sum |F|^2`` at the last step and at its peak (NaN kept); whether
    every series and the final fields are finite."""
    try:
        series = {}
        for o in plan.observers:
            m, entry = getattr(o, 'monitor', None), state['obs'].get(o.name)
            if m is None or not isinstance(entry, dict) or 'energy' not in entry:
                continue
            e = np.asarray(entry['energy'], dtype=np.float64)
            series[m] = e if m not in series else series[m] + e
        E, H = np.asarray(state['E']), np.asarray(state['H'])
        return dict(monitors={m: [float(e[-1]), worst(e)] for m, e in series.items()},
                    series_finite=bool(all(np.isfinite(e).all() for e in series.values())),
                    finite=bool(np.isfinite(E).all() and np.isfinite(H).all()))
    except Exception as err:                                               # noqa: BLE001
        FAULTS.append(f'field norms: {type(err).__name__}: {err}'[:300])  # not V2's to fail
        return dict(error=FAULTS[-1])


PATCH_POINTS = ((checks, '_shared_surface_voxels'), (coefficients, 'painters'),
                (coefficients, '_blend'), (constitutive, 'lower'), (_run, 'lower'),
                (_run, 'execute'))


@contextlib.contextmanager
def _ours(where):
    """Anything this file's own code raises inside a patched call becomes a HarnessError."""
    try:
        yield
    except HarnessError:
        raise
    except Exception as err:                                               # noqa: BLE001
        FAULTS.append(f'{where}: {type(err).__name__}: {err}'[:300])
        raise HarnessError(FAULTS[-1]) from err


@contextlib.contextmanager
def patched(arm, info, mutant=None):
    """Lower ``arm`` on the write set for the declared (substrate, pillar) pair; restore on exit.

    ``mutant``: 'm1' leaves the nodes out of J, 'm2' is switched off by :func:`solve` after
    planning (the patch at plan time only), 'm3' takes the wall weight from ``surface()``,
    'm4' holds door 5 open for every pair. Arm 'none' writes nothing and holds no door.
    """
    st = _Patch(arm, info, mutant)
    door0, painters0, blend0, lower0, run_lower0, execute0 = (getattr(m, a)
                                                              for m, a in PATCH_POINTS)

    def door(body_a, body_b, grid, box, include_common=False):
        count, worst_ = door0(body_a, body_b, grid, box, include_common=include_common)
        with _ours('door'):
            mine = st.declared(body_a, body_b)
            if mine:
                st.door.append(count)
            hold = arm != 'none' and (st.mutant == 'm4' or mine)
        return (0, None) if hold else (count, worst_)

    def painters(scene, plan, axes, occ):
        st.grid = plan.grid           # the grid rides in with the call, never a stale one
        return painters0(scene, plan, axes, occ)

    def blend(rt, paint_list, background_value, structure_values, component, field, rule):
        out = blend0(rt, paint_list, background_value, structure_values, component, field, rule)
        if field != 'E' or rule != 'mix' or info['pil'] is None:
            return out
        with _ours('blend'):
            before = np.asarray(out)
            after = before
            if st.writes():
                t = st.layer(st.grid, tuple((0, k) for k in st.grid.n), component)
                after = np.array(np.broadcast_to(before, t['write'].shape), dtype=np.float64)
                after[t['write']] = 1.0 / t['xi'][component][t['write']]
                out = rt.array(after, 'float64')
            if st.tap is not None:
                st.tap.setdefault('edges', {})[component] = (np.array(before), np.array(after))
        return out

    def lower(scene, plan, rt, report=None, proxy_stats=None, proxy_by_index=None, nodes=None):
        mine = {} if nodes is None else nodes
        out = lower0(scene, plan, rt, report=report, proxy_stats=proxy_stats,
                     proxy_by_index=proxy_by_index, nodes=mine)
        if info['pil'] is None:
            return out
        with _ours('constitutive.lower'):
            for term in plan.constitutive:
                if isinstance(term, StencilTerm) or term.field != 'E':
                    continue
                tensor, inverse = (np.array(np.asarray(v), dtype=np.float64)
                                   for v in mine[term.name])
                cross = np.array(np.asarray(out[term.name]), dtype=np.float64)
                before = (tensor.copy(), cross.copy(), inverse.copy())
                xbox = tuple(reversed(term.box))
                if st.writes() and st.mutant != 'm1':
                    t = st.layer(plan.grid, xbox, 3)
                    w = t['write']
                    tensor[:, w], cross[:, w] = t['xi'][:, w], t['cross'][:, w]
                    inverse[w] = t['inv'][w]
                    mine[term.name] = (rt.array(tensor, 'float64'), rt.array(inverse, 'float64'))
                    out[term.name] = rt.array(cross, 'float64')
                if st.tap is not None:
                    st.tap.setdefault('nodes', {})[term.name] = (before, (tensor, cross, inverse),
                                                                 xbox)
        return out

    def run_lower(scene, plan, rt='numpy', device=None):
        params = run_lower0(scene, plan, rt, device)
        with _ours('executor view'):
            st.exec.append(_executor_view(scene, plan, params))
        return params

    def execute(plan, params, rt):
        state = execute0(plan, params, rt)
        st.obs.append(_field_norms(plan, state))
        return state

    mine = (door, painters, blend, lower, run_lower, execute)
    try:
        for (module, attribute), wrapper in zip(PATCH_POINTS, mine):
            setattr(module, attribute, wrapper)
        yield st
    finally:
        for (module, attribute), original in zip(PATCH_POINTS, (door0, painters0, blend0, lower0,
                                                                run_lower0, execute0)):
            setattr(module, attribute, original)
        for (module, attribute), original in zip(PATCH_POINTS, (door0, painters0, blend0, lower0,
                                                                run_lower0, execute0)):
            if getattr(module, attribute) is not original:
                raise AssertionError(f'{module.__name__}.{attribute} was not restored')


def check_signatures():
    """The private functions this study drives, as it read them at adb2ada."""
    want = {_run._planned: ['scene', 'reuse'], _run._solve: ['scene', 'p', 'rt', 'grazing'],
            _run.lower: ['scene', 'plan', 'rt', 'device'],
            _run.execute: ['plan', 'params', 'rt'],
            constitutive.lower: ['scene', 'plan', 'rt', 'report', 'proxy_stats',
                                 'proxy_by_index', 'nodes'],
            coefficients.painters: ['scene', 'plan', 'axes', 'occ'],
            coefficients._blend: ['rt', 'paint_list', 'background_value', 'structure_values',
                                  'component', 'field', 'rule'],
            checks._shared_surface_voxels: ['body_a', 'body_b', 'grid', 'box', 'include_common']}
    for fn, names in want.items():
        got = list(inspect.signature(fn).parameters)
        if got != names:
            raise AssertionError(f'{fn.__module__}.{fn.__name__}{got}: expected {names}')


# -- V3 on the host, and one solve -----------------------------------------------------------------

def _gap(a, b):
    """``max |a - b| / max |b|``: relative to the array's own scale (a cross coefficient that
    is 0 on one backend and 1e-17 on the other is not a relative error of 1e283). NaN kept."""
    a, b = np.asarray(a), np.asarray(b)
    if not b.size:
        return 0.0
    scale = worst(np.abs(b))
    return worst(np.abs(a - b)) / (scale if not scale < 1e-300 else 1e-300)


def host_v3(scene, info, p, st, backend=None):
    """Lower the arm once more on host NumPy, tapped: pre-lift ``eps`` and nodes against 3w
    outside the write set (bitwise) and the general form on it, the tapped J counts against
    the census, and the digest against the executor's (``backend`` None: phase 0's lowering,
    no executor ran, no digest)."""
    st.tap = {}
    host = _executor_view(scene, p, lowering.lower(scene, p, runtime('numpy')))
    tap, st.tap = st.tap, None
    st.last_tap = tap
    cen = census(info, p.grid)
    out = dict(host_sha=host['sha'], census=cen, cfl=host['cfl'])
    if backend is not None:
        if len(st.exec) != 1 or sorted(st.exec[0]['arrays']) != sorted(host['arrays']):
            out.update(digest=False, exec_views=len(st.exec))
        else:
            ex = st.exec[0]
            gap = worst(*(_gap(ex['arrays'][k], host['arrays'][k]) for k in host['arrays']))
            out.update(exec_sha=ex['sha'], dtype=ex['dtype'], cfl=ex['cfl'], exec_gap=gap)
            out['digest'] = bool(ex['sha'] == host['sha'] if backend == 'numpy'
                                 else gap <= JAX_GAP and ex['dtype'] == 'float64')
    outside, inside, nodes_inside = True, [], []
    edges_J = nodes_J = written = 0
    terms = [t for t in p.constitutive if not isinstance(t, StencilTerm) and t.field == 'E']
    counted = True
    if info['pil'] is not None:
        grid = p.grid
        full = tuple((0, k) for k in grid.n)
        for c, (before, after) in sorted(tap.get('edges', {}).items()):
            # J from p.grid and the general form, independently of the patch's own cache.
            t = layer_tables(info, grid, full, 'E', c, st.arm, form=general)
            w = t['write']
            outside &= bool(np.array_equal(after[~w], before[~w]))
            edges_J = edges_J + int(t['J'].sum())
            if st.arm in WRITING and w.any():
                written += int(w.sum())
                inside.append(np.abs(after[w] * t['xi'][c][w] - 1.0))
        for before, after, xbox in tap.get('nodes', {}).values():
            t = layer_tables(info, grid, xbox, 'E', 3, st.arm, form=general)
            w = t['write']
            for b, a in zip(before, after):
                outside &= bool(np.array_equal(a[..., ~w], b[..., ~w]))
            nodes_J = nodes_J + int(t['J'].sum())
            if st.arm in WRITING and w.any():
                written += int(w.sum())
                scale = np.max(np.abs(t['xi'][:, w]), axis=0)
                nodes_inside += [_gap(after[0][:, w], t['xi'][:, w]),
                                 np.abs(after[1][:, w] - t['cross'][:, w]) / scale,
                                 _gap(after[2][w], t['inv'][w])]
        # V3 must have looked: the three E edge families, every E term's nodes, and door 5's
        # own J in them (the census, counted over door 5's shared box).
        fam = cen['fam']
        counted = bool(len(tap.get('edges', {})) == 3
                       and edges_J == sum(fam.get('E' + a, [0])[0] for a in 'xyz')
                       and len(tap.get('nodes', {})) == len(terms)
                       and (not terms or nodes_J == fam.get('E*', [0])[0]))
    inside, nodes_inside = worst(*inside), worst(*nodes_inside)
    out.update(outside=outside, inside=inside, nodes_inside=nodes_inside, edges=edges_J,
               nodes=nodes_J, written=written,
               beyond_J=written - edges_J - nodes_J if written else 0, counted=counted)
    out['passed'] = bool(outside and inside <= 1e-13 and nodes_inside <= 1e-13 and counted
                         and (backend is None or out['digest'] is True))
    return out


def solve(s, mutant=None):
    """Plan ``s`` once under its arm, solve it with ``_run._solve``, check V3 on the host."""
    scene, info = make(s)
    check_signatures()
    with patched(s['arm'], info, mutant) as st:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t0 = time.perf_counter()
            p, rows = _run._planned(scene)
            t1 = time.perf_counter()
            if mutant == 'm2':
                st.active = False
            res = _run._solve(scene, p, runtime(s['backend']), rows)
            t2 = time.perf_counter()
            st.active = True
            if len(st.exec) != 1 or len(st.obs) != 1:
                raise HarnessError(f'{len(st.exec)} executor lowerings and {len(st.obs)} '
                                   f'executions in one solve, not 1 and 1')
            v3 = host_v3(scene, info, p, st, s['backend'])
            t3 = time.perf_counter()
    return dict(res=res, plan=p, st=st, v3=v3, info=info, scene=scene, caught=caught,
                wall=dict(plan=t1 - t0, solve=t2 - t1, host=t3 - t2))


def _status(err):
    """'refused' for a ``ValueError`` raised inside the wavebit package (a door, the
    certificate, the grazing door); 'error' for anything else (the harness, memory, JAX), and
    for a refusal whose cause chain holds a harness error or that followed one."""
    chain, seen = err, set()
    while chain is not None and id(chain) not in seen:
        seen.add(id(chain))
        if isinstance(chain, HarnessError):
            return 'error'
        chain = chain.__cause__ or chain.__context__
    if FAULTS or not isinstance(err, ValueError):
        return 'error'
    tb, last = err.__traceback__, None
    while tb is not None:
        last, tb = tb.tb_frame.f_code.co_filename, tb.tb_next
    package = os.path.normcase(str(SNAP / 'src/wavebit'))
    return ('refused' if last and os.path.normcase(str(Path(last).resolve())).startswith(package)
            else 'error')


def amplitudes(res, pol):
    """The zeroth order's r and t per wavelength ``[[re, im], ...]``, R00, T00, theta_at."""
    out = {}
    for mon, name, power in (('r', 'r_' + pol, 'R'), ('t', 't_' + pol, 'T')):
        sub = res[mon].orders.incident
        a, w = np.asarray(getattr(sub, name)), np.asarray(getattr(sub, power))
        mid = (a.shape[-2] // 2, a.shape[-1] // 2)
        out[mon] = [[float(np.real(a[i][mid])), float(np.imag(a[i][mid]))] for i in range(len(a))]
        out[power] = [float(np.real(w[i][mid])) for i in range(len(w))]
    out['theta_at'] = [float(v) for v in np.asarray(res['r'].theta_at)]
    return out


def solve_row(s, mutant=None):
    """One solve as a JSON row: amplitudes, J counts, V1-V4, digests, timings, provenance."""
    base = dict(kind='solve', key=key_of(s) + (f'|mutant={mutant}' if mutant else ''),
                prov=PROV['digest'], spec=s, mutant=mutant, provenance=PROV['short'])
    t0 = time.perf_counter()
    FAULTS.clear()
    try:
        out = solve(s, mutant)
    except Exception as err:                                               # noqa: BLE001
        return dict(base, status=_status(err), error=f'{type(err).__name__}: {err}'[:600],
                    faults=list(FAULTS), wall=dict(total=time.perf_counter() - t0))
    if FAULTS:                            # swallowed somewhere, and the solve went on
        return dict(base, status='error', error='harness faults', faults=list(FAULTS),
                    wall=dict(total=time.perf_counter() - t0))
    res, p, st, v3, info = out['res'], out['plan'], out['st'], out['v3'], out['info']
    amp = amplitudes(res, s['pol'])
    cen = v3.pop('census')
    decay = {k: (None if v is None else float(v))
             for k, v in (res.diagnostics.get('decay') or {}).items()}
    obs = st.obs[0]
    names = sorted({f'{w.category.__name__}: {str(w.message)[:90]}' for w in out['caught']})
    values = [x for pair in amp['r'] + amp['t'] + [[v, 0.0] for v in amp['R'] + amp['T']]
              for x in pair]
    finite = bool(np.all(np.isfinite(values)) and obs.get('finite') and obs.get('series_finite'))
    stable = res.diagnostics.get('stable')
    interior = res.diagnostics.get('interior_growth')
    # V2's growth: each monitor plane ends at most GROWTH of its peak, production saw no growth
    # (its GrowingFieldWarning judges the interior energy after the sources end) and nothing
    # overflowed (interior_growth inf).
    monitors = obs.get('monitors') or {}
    growth = bool(monitors and all(last <= GROWTH * peak for last, peak in monitors.values())
                  and not any('GrowingField' in n for n in names)
                  and not (interior is not None and not math.isfinite(interior)))
    v2 = dict(certified='vacuous' if not p.constitutive else stable is True,
              cfl=v3.get('cfl'), nan=not finite, growth=growth,
              last_over_peak={m: (last / peak if peak > 0 else None)
                              for m, (last, peak) in monitors.items()},
              interior_growth=interior, stable=stable if isinstance(stable, bool) else None)
    v2['passed'] = bool(v2['certified'] in (True, 'vacuous') and v2['cfl'] is not None
                        and v2['cfl'] < 1 and finite and growth)
    if s['arm'] == 'none':                # the junction-free controls: nothing reaches J
        v4 = cen['total'] == 0 and sum(st.door) == 0
    else:
        v4 = sum(st.door) == cen['total'] and (info['pil'] is None or len(st.door) == 1)
    return dict(base, status='ok', r=amp['r'], t=amp['t'], R=amp['R'], T=amp['T'],
                theta_at=amp['theta_at'], J=cen['total'], J3=cen['J3'],
                J2=cen['total'] - cen['J3'], census=cen['fam'], door=st.door,
                top=info['top'], base=info['base'],
                V1=dict(passed=bool(len(decay) == 2 and finite
                                    and all(v is not None and v < 1e-5 for v in decay.values())),
                        decay=decay),
                V2=v2, V3=v3,
                V4=dict(passed=bool(v4), door=st.door, census=cen['total']),
                steps=int(p.steps), grid=list(p.grid.n), terms=len(p.constitutive),
                skeleton=str(p.skeleton)[:16],
                field_norms=obs, warnings=names,
                wall=dict(out['wall'], total=time.perf_counter() - t0))


# -- the RCWA side -----------------------------------------------------------------------------

def rcwa_point(name, variant, N, wl, theta=0.0):
    """``{pol: {r, t, R, T}}`` of the zeroth order: 'flush' | 'k1' (air gap under the pillar)
    on the shape path; 'film' | 'bare' unpatterned (exact, truncation 0) at fd's fixed k."""
    g = SCENES[name]
    th = math.asin(math.sin(math.radians(theta)) * wl)      # fd's theta_at: k fixed at 1
    if variant in ('film', 'bare'):
        structure = [dict(background=g['eps_p'] if variant == 'film' else 1.0, shapes=[])]
        thickness, trunc = [g['H'] if variant == 'film' else 0.0], [0, 0]
    else:
        m = g['eps_p']
        shape = {'cyl': dict(type='circle', cx=0.0, cy=0.0, lx=g['size'], material=m),
                 'box': dict(type='rectangle', cx=0.0, cy=0.0, lx=g['size'], ly=g['size'],
                             material=m),
                 'ridge': dict(type='rectangle', cx=0.0, cy=0.0, lx=g['size'], ly=g['P'],
                               material=m)}[g['kind']]
        structure, thickness = [dict(background=1.0, shapes=[shape])], [g['H']]
        if variant == 'k1':               # layers run from the top: the gap under the pillar
            structure.append(dict(background=1.0, shapes=[]))
            thickness.append(k1_gap(name))
        trunc = [N, 0] if g['kind'] == 'ridge' else [N, N]
    res = legacy_rcwa(backend='numpy', top=1.0, bot=g['eps_s'], theta=th, phi=0.0, psi=math.pi / 2,
                       truncation_order=trunc, period=[g['P'], g['P']], wavelength=wl,
                       structure=structure, thickness=thickness).solve()
    out = dict(thickness=thickness)       # the report checks a twin's total against fd's top
    for pol in ('s', 'p'):
        sub = res.s_inc if pol == 's' else res.p_inc
        z = {}
        for name_ in ('r_' + pol, 't_' + pol, 'R', 'T'):
            a = np.asarray(getattr(sub, name_))
            z[name_[0]] = complex(a[a.shape[0] // 2, a.shape[1] // 2])
        out[pol] = dict(r=[z['r'].real, z['r'].imag], t=[z['t'].real, z['t'].imag],
                        R=z['R'].real, T=z['T'].real)
    return out


# -- provenance and rows ---------------------------------------------------------------------------

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _tree(root, skip=None):
    """SHA-256 of every ``.py`` file under ``root`` (paths and LF-normalised contents), bar
    the directory ``skip``."""
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        if skip and (rel == skip or rel.startswith(skip + '/')):
            continue
        h.update(rel.encode() + b'\0' + bytes.fromhex(_sha(path)))
    return h.hexdigest()


#: The line that ends the part of this file a solve row depends on (its digest's source).
MARKER = '# -- phase 0 '


def _own_source():
    """This file from its first import to the phase-0 section, LF-normalised: every constant,
    builder, arm, wrapper and row writer a solve row's numbers come from."""
    text = Path(__file__).read_text(encoding='utf-8').replace('\r\n', '\n')
    return text[text.index('\nimport '):text.index('\n' + MARKER)]


def provenance():
    base = Path(fd.__file__).resolve().parent
    files = {f: _sha(repository_path('src/wavebit/fd', f)) for f in FILES}
    try:
        got = subprocess.run(['git', '-C', str(SNAP), 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, timeout=20)
        commit = got.stdout.strip() if got.returncode == 0 else None
    except OSError:
        commit = None
    if commit is None:
        same = tuple(files[f][:16] for f in FILES) == AT_1C26BF3
        commit = ('adb2adae6ea942d877b72b901c9c4653b3515d66 (snapshot; the ten file hashes match)'
                  if same else 'unknown (not a git checkout; the file hashes differ from adb2ada)')
    study = hashlib.sha256(_own_source().encode())
    versions = {'numpy': np.__version__, 'python': platform.python_version()}
    try:
        import jax
        import jaxlib
        versions.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
    except ImportError:
        versions.update(jax=None)
    fd_tree = _tree(base)
    files_digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps([fd_tree, study.hexdigest(), versions],
                                       sort_keys=True).encode()).hexdigest()[:16]
    threads = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                              'MKL_NUM_THREADS')}
    short = dict(commit=commit[:12], files=files_digest[:16], fd_tree=fd_tree[:16],
                 study=study.hexdigest()[:16], **versions, threads=threads.get('OMP_NUM_THREADS'))
    # RCWA rows answer to their own code only: an fd-side change must not recompute them.
    rest = _tree(base.parent, skip='fdtd')
    rcwa = hashlib.sha256(json.dumps([inspect.getsource(rcwa_point), inspect.getsource(k1_gap),
                                      repr(SCENES), rest, versions],
                                     sort_keys=True).encode()).hexdigest()[:16]
    return dict(digest=digest, rcwa=rcwa, short=short, commit=commit, files=files,
                fd_tree=fd_tree, wavebit_rest=rest, study=study.hexdigest(),
                versions=versions, threads=threads, x64=os.environ.get('JAX_ENABLE_X64'),
                xla_flags=os.environ.get('XLA_FLAGS'), platform=platform.platform())


PROV = None


def _json(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return repr(v)


def emit(path, row, quiet=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(row, default=_json) + '\n')
    if not quiet:
        brief = {k: row[k] for k in ('key', 'status', 'check', 'passed') if k in row}
        for k in ('V1', 'V2', 'V3', 'V4'):
            if isinstance(row.get(k), dict):
                brief[k] = row[k].get('passed')
        if 'wall' in row:
            brief['wall'] = round(row['wall'].get('total', 0.0), 1)
        print(json.dumps(brief, default=_json), flush=True)


def load(path, current=True):
    """``{key: row}``, the last row per key; only the current provenance unless told."""
    out = {}
    if path.exists():
        for line in open(path, encoding='utf-8'):
            if line.strip():
                row = json.loads(line)
                # A phase-0 check row is current only under its own check's digest.
                want = (check_prov(row['check']) if row.get('kind') == 'phase0'
                        else PROV['rcwa'] if row.get('kind') == 'rcwa' else PROV['digest'])
                if 'key' in row and (not current or row.get('prov') == want):
                    out[row['key']] = row
    return out


def run_specs(specs, log=LOG):
    """Solve every spec not already in the log under this provenance (a row whose status is
    'error' is solved again); a run at its default periods that fails V1 gets its one repeat
    at twice the periods (60 -> 120, the anchor's 80 -> 160), whose values then stand for it."""
    done = load(log)
    for s in specs:
        k = key_of(s)
        row = done.get(k)
        if row is None or row.get('status') == 'error':
            row = solve_row(s)
            emit(log, row)
            done[k] = row
        default = 80 if s['scene'] in FILMS else 60
        if s['periods'] == default and row.get('status') == 'ok' and not row['V1']['passed']:
            again = dict(s, periods=2 * default)
            if done.get(key_of(again)) is None or done[key_of(again)].get('status') == 'error':
                done[key_of(again)] = solve_row(again)
                emit(log, done[key_of(again)])


# -- phase 0 -----------------------------------------------------------------------------------

def _p0(check, scope, passed, detail, applied=None):
    return dict(kind='phase0', key=f'phase0|{check}|{scope}', check=check, scope=scope,
                passed=None if passed is None else bool(passed), applied=applied, detail=detail,
                prov=check_prov(check), provenance=PROV['short'])


_CHECK_PROV = {}


def check_prov(check):
    """The digest a phase-0 row carries: the numbers' digest and its own check's code."""
    if check not in _CHECK_PROV:
        code = {'V4': (part_v4, v4_cases, plan_under), 'V3': (part_v3, v3_cases, lower_host),
                'V3 film 1b==2': (part_v3, v3_cases, lower_host), 'V6': (part_v6, v6_cases),
                'smoke': (part_smoke, smoke_specs), 'estimate': (part_estimate, _cells_steps)}
        fns = (code.get(check)
               or ((part_v5, v5_cases, v5z_cases, v5_series, v5_verdict)
                   if check.startswith('V5') else
                   (part_mutants, mutant_cases, lower_host, dimer, plan_under)
                   if check.startswith('mutant') else (part_agree,)))
        text = PROV['digest'] + ''.join(inspect.getsource(f) for f in fns)
        _CHECK_PROV[check] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return _CHECK_PROV[check]


def _p0_run(check, scope, fn):
    done = load(LOG)
    key = f'phase0|{check}|{scope}'
    if key in done:
        return done[key]
    t0 = time.perf_counter()
    try:
        row = fn()
    except Exception as err:                                               # noqa: BLE001
        row = _p0(check, scope, False, dict(error=f'{type(err).__name__}: {err}'[:600]))
    row['detail']['wall'] = round(time.perf_counter() - t0, 2)
    emit(LOG, row)
    return row


def plan_under(arm, scene, info, mutant=None):
    """``(plan or error text, patch state)`` of ``scene`` planned under ``arm``."""
    with patched(arm, info, mutant) as st, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return _run._planned(scene)[0], st
        except ValueError as err:
            return f'{err}'[:300], st


def part_provenance():
    emit(LOG, dict(kind='provenance', key=f'provenance|{time.time():.0f}', prov=PROV['digest'],
                   **{k: PROV[k] for k in ('commit', 'files', 'fd_tree', 'wavebit_rest', 'study',
                                           'versions', 'threads', 'x64', 'xla_flags',
                                           'platform')}))


# The case lists: the parts run them and the report expects every one of their rows.

def v4_cases():
    return [(name, n, z) for name in list(SCENES) + list(FILMS) for n in rungs_of(name)
            for z in ZETAS]


def v3_cases():
    out = []
    for name in list(SCENES) + list(FILMS):
        arms = (('2', '1b', '1a', '3w', 'PA') if name in FILMS else
                ('2', '1b', '1a', '3w', 'P3') + (('1bL', '1bR') if name == 'R' else ()))
        out += [(name, rungs_of(name)[0], z, arm) for z in ZETAS for arm in arms]
    return out


def v5_cases():
    """``(check, scene, zeta, what, arm, legacy)``: the radius and offset sweeps at the
    registered placements, and the control a jumping rule must fail (right after the case
    whose arm 2 sweep it takes its step scale from)."""
    out = [('V5', nm, z, w, '2', None) for nm in ("C'", "T'") for z in ZETAS
           for w in ('radius', 'offset')]
    return out[:1] + [('V5 power (1b must fail)', "C'", 0.25, 'radius', '1b', None)] + out[1:]


def v5z_cases():
    """The flush pair's height, swept 0.3 of a cell either side of the ties 0.5 (the edges'
    and nodes' voxels cross the plane) and 0 (the Ez voxels do), and its two controls: the arm
    without the base side, and with the gated node, must fail."""
    out = []
    for nm in ("C'", "T'", 'B', "D'"):
        out.append(('V5', nm, 0.5, 'height', '2', None))
        if nm == "C'":
            out.append(('V5 power (no base side must fail)', nm, 0.5, 'height', '2',
                        'no-base-side'))
        if nm == "T'":
            out.append(('V5 power (gated node must fail)', nm, 0.5, 'height', '2', 'gated-node'))
        out.append(('V5', nm, 0.0, 'height', '2', None))
    return out


def v6_cases():
    return [(name, 0.25) for name in SCENES] + [("C'", 0.75)]


def mutant_cases():
    return [(name, m) for name in ("C'", 'A') for m in ('m1', 'm2', 'm3', 'm4')]


def part_v4():
    """Plan every (scene, rung, zeta) of the core list under arm 2: door 5's count = census."""
    for name, n, z in v4_cases():
        def one(name=name, n=n, z=z):
            scene, info = make(spec(name, n, z, '2'))
            p, st = plan_under('2', scene, info)
            if isinstance(p, str):
                return _p0('V4', f'{name}|{n}|{z}', False, dict(error=p))
            cen = census(info, p.grid)
            return _p0('V4', f'{name}|{n}|{z}', st.door == [cen['total']],
                       dict(door=st.door, census=cen['total'], J3=cen['J3'], fam=cen['fam'],
                            terms=len(p.constitutive)))
        _p0_run('V4', f'{name}|{n}|{z}', one)


def lower_host(arm, scene, info, mutant=None):
    """Plan under ``arm`` and run :func:`host_v3` (no solve): phase 0's V3 and its digest."""
    with patched(arm, info, mutant) as st, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = _run._planned(scene)[0]
        v3 = host_v3(scene, info, p, st)
    return p, v3, st.last_tap


def part_v3():
    """Every arm at each scene's lowest rung, both placements: V3 on the host, and the digest."""
    for name, n, z, arm in v3_cases():
        def one(name=name, z=z, n=n, arm=arm):
            scene, info = make(spec(name, n, z, arm))
            _, v3, _ = lower_host(arm, scene, info)
            return _p0('V3', f'{name}|{n}|{z}|{arm}', v3['passed'], v3)
        _p0_run('V3', f'{name}|{n}|{z}|{arm}', one)
    for name in FILMS:                    # J3 is empty on a film: 1b must be 2, bitwise
        for z in ZETAS:
            n = rungs_of(name)[0]

            def same(name=name, z=z, n=n):
                rows = load(LOG)
                a, b = (rows[f'phase0|V3|{name}|{n}|{z}|{arm}']['detail']['host_sha']
                        for arm in ('2', '1b'))
                return _p0('V3 film 1b==2', f'{name}|{n}|{z}', a == b, dict(sha2=a, sha1b=b))
            _p0_run('V3 film 1b==2', f'{name}|{n}|{z}', same)


def v5_series(name, n, zeta, what, arm='2', legacy=None):
    """Arm ``arm`` at the plane rows over 601 radii, lateral offsets or heights of the flush
    pair (+-0.3 cell): per family and entry the values on the write set and today's rule off it,
    the write-set membership, production's tangent-plane gate, the closed form's distance from
    today's rule and the out-of-band fill ``b``, shape ``(601, points)``.

    The height sweep is centred on a tie (``zeta`` 0.5 or 0) and offset half a step, so the tie
    itself -- a cap on a voxel face, production's own measure-zero convention -- is not sampled;
    it keeps every point of the plane rows, since the fills there change with the height."""
    g = SCENES[name]
    scene, info = make(spec(name, n, zeta, '2'))
    grid = Grid.from_domain(scene.domain.size, scene.domain.center, 1.0 / n)
    es, ep, eg = info['eps']
    dx = info['dx']
    height = what == 'height'
    deltas = np.linspace(-0.3, 0.3, 601) + (0.0005 if height else 0.0)
    near = tuple((max(lo - 1, 0), min(hi + 1, k)) for (lo, hi), k in
                 zip(info['pil'].shape.support(grid), grid.n))    # one fixed point set
    frac = lambda v: (v > TOL) & (v < 1 - TOL)                               # noqa: E731
    series = {}
    for delta in deltas:
        if what == 'radius':
            sub_shape, shape = shapes(name, info, radius=g['size'] / 2 + delta * dx)
        elif what == 'offset':
            sub_shape, shape = shapes(name, info, dxy=(delta * dx, 0.5 * delta * dx))
        else:
            sub_shape, shape = shapes(name, info, dz=delta * dx)
        sub_body, body = sub_shape.body(), shape.body()
        info_d = dict(info, bodies=(sub_body, body), plane=delta * dx if height else 0.0)
        for c in range(4 if info['kind'] == 'cyl' else 3):   # a Box plans no node term
            rows, geo = layer_geometry(info_d, grid, near, 'E', c, legacy=legacy)
            pts = (yee_points(grid, rows, 'E', c) if c < 3
                   else constitutive.common_points(grid, rows, 'E'))
            keep = np.ones(geo['alpha'].shape, bool) if height else frac(geo['alpha'])
            xi, cross, inv = arm_tables(arm, geo, info['eps'], 1.0)
            member = write_set(geo, c)
            # Today's rule: the substrate painted first, then the pillar (edges: mix; node:
            # constitutive.lower's branch with the nearest-surface normal).
            fs, ws = (np.asarray(v) for v in body_occupancy(sub_body, pts, grid.dx))
            fp, wp = (np.asarray(v) for v in body_occupancy(body, pts, grid.dx))
            band = np.minimum(np.minimum(fs, 1 - fs), np.minimum(fp, 1 - fp)).clip(0.0)
            if c < 3:
                held = coefficients.mix(np, eg, es, fs, ws[..., c])
                today = 1 / coefficients.mix(np, held, ep, fp, wp[..., c])
                vals = [np.where(member, xi[c], today)]
                bound = [np.abs(xi[c] - today) / np.abs(today)]
            else:
                ar1 = fs * es + (1 - fs) * eg
                con1 = fs * (1 - fs) * (es - eg) ** 2 / (es * eg * ar1)
                ar2 = fp * ep + (1 - fp) * ar1
                con2 = fp * (1 - fp) * (ep - ar1) ** 2 / (ep * ar1 * ar2)
                nrm = np.asarray(body.surface(pts)[1])
                t_sub = np.stack([1 / ar1, 1 / ar1, 1 / ar1 + con1])
                t_now = np.where(fp == 0, t_sub, np.stack([1 / ar2 + con2 * nrm[..., a] ** 2
                                                           for a in range(3)]))
                x_now = np.where(fp == 0, 0.0, con2 * nrm[..., 0] * nrm[..., 1])
                i_now = np.where(fp == 0, 1 / ar1, 1 / ar2)
                vals = [np.where(member, xi[a], t_now[a]) for a in range(3)]
                vals += [np.where(member, cross[0], x_now), np.where(member, inv, i_now)]
                bound = [np.abs(xi[a] - t_now[a]) / np.abs(t_now[a]) for a in range(3)]
                bound += [np.abs(cross[0] - x_now) / np.abs(t_now[0]),
                          np.abs(inv - i_now) / np.abs(i_now)]
            for k, v in enumerate(vals):
                entry = series.setdefault((c, k), dict(v=[], J=[], gate=[], bound=[], b=[]))
                entry['b'].append(band[keep])
                entry['v'].append(v[keep])
                entry['J'].append(member[keep])
                entry['gate'].append(geo['live'][keep])
                entry['bound'].append(bound[k][keep])
    return {k: {f: np.array(x) for f, x in e.items()} for k, e in series.items()}


def v5_verdict(ser, ref, lip):
    """No step above 50x the median step of ``ref`` (arm 2's own sweep) except where
    production's gate switches; at every write-set membership change the closed form equals
    today's rule on the outside side to ``1e-12 + lip b``. ``(clean, detail)``."""
    jumps = unexplained = changes = 0
    largest, over, ratio, b_out = [], [], [], []
    finite = True
    for (c, k), e in ser.items():
        finite &= bool(np.isfinite(e['v']).all())
        steps = np.abs(np.diff(e['v'], axis=0))
        floor = 1e-12 * worst(np.abs(e['v']))              # round-off is not a step
        base = np.abs(np.diff(ref[(c, k)]['v'], axis=0))
        moving = base[base > floor]
        flips = e['gate'][1:] != e['gate'][:-1]
        if moving.size:
            median = float(np.median(moving))
            big = (steps > 50 * median) & (steps > floor)
            jumps += int(big.sum())
            unexplained += int((big & ~flips).sum())
            largest.append(steps[~flips] / median)
        change = e['J'][1:] != e['J'][:-1]
        changes += int(change.sum())
        if change.any():
            # The side outside the write set: its geometry evaluated by the closed form.
            out_side = np.where(e['J'][1:], e['bound'][:-1], e['bound'][1:])[change]
            band = np.where(e['J'][1:], e['b'][:-1], e['b'][1:])[change]
            over.append(out_side)
            ratio.append(out_side / (1e-12 + lip * band))
            b_out.append(band)
    detail = dict(jumps=jumps, unexplained=unexplained, changes=changes,
                  worst_boundary=worst(*over), boundary_over_allowed=worst(*ratio),
                  largest_band_outside=worst(*b_out), largest_off_switch=worst(*largest),
                  finite=finite)
    clean = bool(finite and unexplained == 0 and detail['boundary_over_allowed'] <= 1.0)
    return clean, detail


def part_v5(cases=None):
    """Continuity, per :func:`v5_verdict`. A control ('V5 power ...') passes when its arm is
    caught (finite, and not clean): a check that cannot fail certifies nothing."""
    cache = {}                            # the last arm 2 sweep only: a control follows its case
    for check, name, zeta, what, arm, legacy in (v5_cases() if cases is None else cases):
        n = SCENES[name]['rungs'][0]

        def one(check=check, name=name, zeta=zeta, what=what, n=n, arm=arm, legacy=legacy):
            ser = v5_series(name, n, zeta, what, arm, legacy)
            key = (name, n, zeta, what)
            if arm == '2' and legacy is None:
                cache.clear()
                cache[key] = ser
            # The step scale is arm 2's: a piecewise-constant rule's median step is its jump.
            ref = cache.get(key) or v5_series(name, n, zeta, what)
            eps = SCENES[name]['eps_p'], SCENES[name]['eps_s'], 1.0
            clean, detail = v5_verdict(ser, ref, 2 * max(eps) / min(eps))
            control = check != 'V5'
            passed = (detail['finite'] and not clean) if control else clean
            return _p0(check, f'{name}|{n}|{zeta}|{what}', passed,
                       dict(detail, node_extra=NODE_EXTRA, legacy=legacy, arm=arm))
        _p0_run(check, f'{name}|{n}|{zeta}|{what}', one)


def part_v6():
    """Listing order: arm 2 lowered with (substrate, pillar) and (pillar, substrate).

    Arm 2's own values -- the pre-lift eps on the write set, the lifted ``be`` there, and the
    node arrays -- must be bit-identical. Everything else must agree to 1e-14 relative and is
    reported: outside J the lowering is today's painters in scene order, and ``mix`` at a fill
    of 0 or 1 returns its input only to within a double reciprocal, so the two orders differ
    there by round-off that no junction rule can remove (phase 0 measured up to 5.1e-15 of the
    scale, 24 entries at T' 40; the Box scenes are bitwise)."""
    for name, zeta in v6_cases():
        n = SCENES[name]['rungs'][0]

        def one(name=name, zeta=zeta, n=n):
            got = []
            for reverse in (False, True):
                scene, info = make(spec(name, n, zeta, '2'), reverse=reverse)
                with patched('2', info) as st, warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    p = _run._planned(scene)[0]
                    st.tap = {}
                    view = _executor_view(scene, p, lowering.lower(scene, p, runtime('numpy')))
                    got.append((view, st.tap, info, p.grid))
            (a, ta, info, grid), (b, tb, _, _) = got
            full = tuple((0, k) for k in grid.n)
            on_j = off_j = post_on_j = True     # the eps before and after the lift, on/off it
            be_a = np.broadcast_to(a['arrays']['be'], (3,) + tuple(reversed(grid.n)))
            be_b = np.broadcast_to(b['arrays']['be'], (3,) + tuple(reversed(grid.n)))
            written = 0
            for c in range(3):
                w = layer_tables(info, grid, full, 'E', c, '2')['write']
                written += int(w.sum())
                x, y = ta['edges'][c][1], tb['edges'][c][1]
                on_j &= bool(np.array_equal(x[w], y[w]))
                off_j &= bool(np.array_equal(x[~w], y[~w]))
                post_on_j &= bool(np.array_equal(be_a[c][w], be_b[c][w]))
            be_same = bool(np.array_equal(a['arrays']['be'], b['arrays']['be']))
            nodes_same = bool(sorted(ta.get('nodes', {})) == sorted(tb.get('nodes', {})) and all(
                np.array_equal(x, y) for k in ta.get('nodes', {})
                for x, y in zip(ta['nodes'][k][1], tb['nodes'][k][1])))
            gap = worst(*(_gap(a['arrays'][k], b['arrays'][k]) for k in a['arrays']))
            diff = np.argwhere(a['arrays']['be'] != b['arrays']['be'])
            return _p0('V6', f'{name}|{n}|{zeta}',
                       written > 0 and on_j and post_on_j and nodes_same and gap <= 1e-14,
                       dict(pre_lift_on_J=on_j, nodes_bitwise=nodes_same,
                            post_lift_on_J=post_on_j, pre_lift_off_J=off_j, written=written,
                            be_bitwise=be_same, max_rel_gap=gap, differing=int(len(diff))))
        _p0_run('V6', f'{name}|{n}|{zeta}', one)


def dimer(s):
    """``s``'s scene plus a sphere through the pillar's wall (4 cells across or more, one cell
    clear of the cell's faces and of the plane): a pair that is not the declared one, so
    door 5 must still refuse it."""
    scene, info = make(s)
    g = SCENES[info['name']]
    ball = fd.Structure(fd.Sphere(radius=0.1, center=(info['xc'] + 0.05, info['yc'],
                                                       0.5 * g['H'])), fd.Material(eps=4.0))
    return scene.replace(structures=scene.structures + (ball,)), info


def part_mutants():
    """m1-m4 on C' 40 and A 20 (placement 0.25): each must fail its named check, and the
    unmutated arm must pass the same check."""
    for name, mutant in mutant_cases():
        n = SCENES[name]['rungs'][0]
        s = spec(name, n, 0.25, '2')

        def one(name=name, n=n, s=s, mutant=mutant):
            scene, info = make(s)
            if mutant in ('m1', 'm3'):
                check = 'V3 nodes inside J' if mutant == 'm1' else 'V3 edges inside J'
                field = 'nodes_inside' if mutant == 'm1' else 'inside'
                _, good, tg = lower_host('2', scene, info)
                _, bad, tb = lower_host('2', scene, info, mutant)
                part = 'nodes' if mutant == 'm1' else 'edges'
                after = lambda tap, k: (tap[part][k][1] if part == 'nodes'     # noqa: E731
                                        else (tap[part][k][1],))
                applied = any(not np.array_equal(x, y) for k in tg.get(part, {})
                              for x, y in zip(after(tg, k), after(tb, k)))
                control, caught = good[field] <= 1e-13, bad[field] > 1e-13
                detail = dict(control=good[field], mutant=bad[field])
            elif mutant == 'm2':
                check = 'V3 executor digest'
                good, bad = solve(s)['v3'], solve(s, 'm2')['v3']
                applied = good.get('exec_sha') != bad.get('exec_sha')
                control, caught = good['digest'] is True, bad['digest'] is False
                detail = dict(control_exec=str(good.get('exec_sha'))[:16],
                              mutant_exec=str(bad.get('exec_sha'))[:16],
                              host=bad['host_sha'][:16])
            else:
                check = 'narrow door hold (a dimer refuses)'
                two, info2 = dimer(s)
                good, _ = plan_under('2', two, info2)
                bad, _ = plan_under('2', two, info2, 'm4')
                control = isinstance(good, str) and 'surfaces pass through' in good
                caught = not isinstance(bad, str)
                applied = isinstance(good, str) != isinstance(bad, str)
                detail = dict(control=good if isinstance(good, str) else 'admitted',
                              mutant=bad if isinstance(bad, str) else 'admitted')
            return _p0(f'mutant {mutant}', f'{name}|{n}|0.25',
                       bool(control and caught and applied),
                       dict(detail, check=check, control_passed=bool(control),
                            caught=bool(caught)), applied=bool(applied))
        _p0_run(f'mutant {mutant}', f'{name}|{n}|0.25', one)


def smoke_specs():
    out = [spec(nm, 40, z, arm) for nm in ("C'", "T'") for z in ZETAS for arm in ('2', '1b', 'P3')]
    return out + [spec('A', 20, z, '2') for z in ZETAS]


def part_smoke():
    run_specs(smoke_specs())
    rows = load(LOG)
    for s in smoke_specs():
        r = rows[key_of(s)]
        ok = r.get('status') == 'ok' and all(r[v]['passed'] for v in ('V1', 'V2', 'V3', 'V4'))

        def one(r=r, ok=ok, s=s):
            return _p0('smoke', key_of(s), ok,
                       dict(status=r.get('status'), error=r.get('error'),
                            **{v: r.get(v, {}).get('passed') for v in ('V1', 'V2', 'V3', 'V4')},
                            decay=r.get('V1', {}).get('decay'), cfl=r.get('V3', {}).get('cfl'),
                            last_over_peak=r.get('V2', {}).get('last_over_peak')))
        _p0_run('smoke', key_of(s), one)


def _complex(row, mon):
    return np.array([complex(*v) for v in row[mon]])


def agree_cases():
    """The agreement checks' ``(check, scope)`` and the specs they solve."""
    base = spec("C'", 40, 0.25, '2')
    pairs = {name: [spec(name, SCENES[name]['rungs'][0], 0.25, '2', pol=pol) for pol in 'sp']
             for name in ('A', 'B', "C'", "D'", "T'")}
    keys = [('NumPy vs JAX', "C'|40|0.25|2"), ('fd.run vs _solve', "C'|40|0.25|2")]
    keys += [('s == p', f"{name}|{v[0]['n']}|0.25") for name, v in pairs.items()]
    return base, pairs, keys


def part_agree():
    """NumPy against JAX and fd.run against _solve at C' 40; s == p at each 3-D scene."""
    base, pairs, _ = agree_cases()
    jx = dict(base, backend='jax')
    run_specs([base, jx])
    rows = load(LOG)

    def gap_of(a, b, sign=1):
        return worst(*(np.abs(_complex(a, m) - sign * _complex(b, m)) for m in ('r', 't')))

    def numpy_jax():
        a, b = rows[key_of(base)], rows[key_of(jx)]
        gap = gap_of(a, b)
        return _p0('NumPy vs JAX', "C'|40|0.25|2", gap <= 1e-10, dict(max_gap=gap))
    _p0_run('NumPy vs JAX', "C'|40|0.25|2", numpy_jax)

    def run_vs_solve():
        scene, info = make(base)
        with patched('2', info), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = fd.run(scene, backend='numpy')
        amp, row = amplitudes(res, 'p' if base['pol'] == 'p' else 's'), rows[key_of(base)]
        same = all(amp[k] == row[k] for k in ('r', 't', 'R', 'T'))
        return _p0('fd.run vs _solve', "C'|40|0.25|2", same, dict(bitwise=same,
                                                                   gap=gap_of(amp, row)))
    _p0_run('fd.run vs _solve', "C'|40|0.25|2", run_vs_solve)

    for name, pair in pairs.items():
        run_specs(pair)

        def s_p(name=name, pair=pair):
            rows = load(LOG)
            a, b = (rows[key_of(x)] for x in pair)
            dt = worst(np.abs(_complex(a, 't') - _complex(b, 't')))
            dr = least(worst(np.abs(_complex(a, 'r') - _complex(b, 'r'))),   # r_p = -r_s at
                       worst(np.abs(_complex(a, 'r') + _complex(b, 'r'))))   # normal incidence
            return _p0('s == p', f"{name}|{pair[0]['n']}|0.25", dt < 1e-12 and dr < 1e-12,
                       dict(t_gap=dt, r_gap=dr))
        _p0_run('s == p', f"{name}|{pair[0]['n']}|0.25", s_p)


def phase0_expected():
    """Every ``(check, scope)`` phase 0 writes that the report depends on (timing is not)."""
    out = [('V4', f'{nm}|{n}|{z}') for nm, n, z in v4_cases()]
    out += [('V3', f'{nm}|{n}|{z}|{arm}') for nm, n, z, arm in v3_cases()]
    out += [('V3 film 1b==2', f'{nm}|{rungs_of(nm)[0]}|{z}') for nm in FILMS for z in ZETAS]
    out += [(c, f"{nm}|{SCENES[nm]['rungs'][0]}|{z}|{w}")
            for c, nm, z, w, _, _ in v5_cases() + v5z_cases()]
    out += [('V6', f"{nm}|{SCENES[nm]['rungs'][0]}|{z}") for nm, z in v6_cases()]
    out += [(f'mutant {m}', f"{nm}|{SCENES[nm]['rungs'][0]}|0.25") for nm, m in mutant_cases()]
    out += [('smoke', key_of(s)) for s in smoke_specs()]
    return out + agree_cases()[2]


def timing_specs():
    low = [spec(nm, rungs_of(nm)[0], 0.25, '2') for nm in SCENES]
    low += [spec('FA', 20, 0.25, '2'), spec('FA', 40, 0.25, '2'),
            spec("FC'", 320, 0.25, '2', pol='p', theta=OBLIQUE),
            spec("FC'", 320, 0.25, 'PA', pol='p', theta=OBLIQUE),
            # a pair in one process: 3w reuses arm 2's compiled executor (the same plan)
            spec("FC'", 160, 0.25, '2', pol='p', theta=OBLIQUE),
            spec("FC'", 160, 0.25, '3w', pol='p', theta=OBLIQUE)]
    jax = [spec('B', 48, 0.25, '2'), spec('B', 48, 0.25, '1b'), spec("C'", 60, 0.25, '2'),
           spec("T'", 60, 0.25, '2'), spec('A', 80, 0.25, '2'), spec("D'", 80, 0.25, '2'),
           spec('R', 320, 0.25, '2')]
    return low, jax


def part_timing(which=('low', 'jax')):
    low, jax = timing_specs()
    run_specs((low if 'low' in which else []) + (jax if 'jax' in which else []))
    if 'rcwa' in which:                   # phase 1's ladder at one wavelength, for its cost
        done = load(RCWA_LOG)
        for N in (9, 11, 13, 15, 17):
            rcwa_row(done, "C'", 'flush', N, 1.0)
        for f in FILMS:
            rcwa_row(done, FILMS[f], 'film', 0, 1.0, OBLIQUE)
            rcwa_row(done, FILMS[f], 'bare', 0, 1.0, OBLIQUE)


def _cells_steps(s):
    """``(cells, steps)`` of a spec without planning it."""
    scene, _ = make(s)
    grid = Grid.from_domain(scene.domain.size, scene.domain.center, 1.0 / s['n'])
    return (int(np.prod(grid.n)),
            int(math.ceil(s['periods'] * 2 * math.sqrt(3) * s['n'] - 1e-9)))


def part_estimate():
    """Hours for phases 1-3 from the measured rows. NumPy: seconds per step ``a + b cells``
    (least squares; the one-column films are all overhead). JAX: ``c + r cells steps`` over the
    rungs after the first JAX solve of a process (which also pays JAX's start-up), ``c`` then
    paid once per distinct plan. Plan and host-V3 time per cell. RCWA: seconds per solve by
    truncation at C' (wavelength 1), times the ladder of section 9."""
    rows = [r for r in load(LOG).values() if r.get('kind') == 'solve' and r.get('status') == 'ok'
            and not r.get('mutant')]
    cells = lambda r: float(np.prod(r['grid']))                          # noqa: E731
    step = {}                             # NumPy seconds per step, a + b cells: films apart
    for film in (False, True):
        npy = [r for r in rows if r['spec']['backend'] == 'numpy'
               and (r['spec']['scene'] in FILMS) == film]
        step[film] = np.linalg.lstsq(np.array([[1.0, cells(r)] for r in npy]),
                                     np.array([r['wall']['solve'] / r['steps'] for r in npy]),
                                     rcond=None)[0]
    # JAX, 3-D: arm 2's timing rungs (each the first solve of its plan), not B 48 (the first
    # JAX solve of its process) nor C' 40 (the agreement check, also a first).
    jx = [r for r in rows if r['spec']['backend'] == 'jax' and r['spec']['scene'] in SCENES
          and r['spec']['scene'] != 'B' and r['spec']['n'] != 40 and r['spec']['arm'] == '2']
    c, rate = np.linalg.lstsq(np.array([[1.0, cells(r) * r['steps']] for r in jx]),
                              np.array([r['wall']['solve'] for r in jx]), rcond=None)[0]
    c = max(float(c), 0.0)
    # JAX, films: per step from the fastest (a reused executor), the compile from the slowest.
    jf = [r for r in rows if r['spec']['backend'] == 'jax' and r['spec']['scene'] in FILMS]
    f_step = min(r['wall']['solve'] / r['steps'] for r in jf) if jf else 3e-5
    f_compile = max(r['wall']['solve'] - f_step * r['steps'] for r in jf) if jf else 10.0
    over = float(np.median([(r['wall']['plan'] + r['wall']['host']) / cells(r) for r in rows]))
    table = {}
    for phase, specs in (('2 anchor', anchor_specs()),
                         ('3 pillars (core)', [s for nm in SCENES for s in pillar_specs(nm)]),
                         ('3 pillars (minimal)', [s for nm in SCENES
                                                  for s in pillar_specs(nm, True)])):
        total, seen, n_jax = 0.0, set(), 0
        for s in specs:
            n_cells, steps = _cells_steps(s)
            if s['backend'] == 'jax':
                plan = (s['scene'], s['n'], s['zeta'], s['variant'], s['pol'], s['theta'],
                        s['periods'])
                film = s['scene'] in FILMS
                total += 0.0 if plan in seen else (f_compile if film else c)
                total += f_step * steps if film else rate * n_cells * steps
                seen.add(plan)
                n_jax += 1
            else:
                a, b = step[s['scene'] in FILMS]
                total += steps * (a + b * n_cells)
            total += over * n_cells
        table[phase] = dict(solves=len(specs), jax=n_jax, hours=round(total / 3600, 2))
    per = {r['N']: r['wall']['total'] for r in load(RCWA_LOG).values()
           if r.get('scene') == "C'" and r.get('variant') == 'flush' and r.get('wl') == 1.0}
    ladder = 5 * 3 * (sum(per.get(N, 0.0) for N in (9, 11, 13, 15, 17))
                      + per.get(15, 0.0) + per.get(17, 0.0))
    table['1 rcwa (3-D ladders and K1 twins; R and films add about a minute)'] = dict(
        solves=5 * 3 * 7, hours=round(ladder / 3600, 2),
        conditional_19_21_per_scene_hours=round(2.5 * per.get(17, 0.0) / 3600 * 2, 2))
    fit = dict(numpy_step_s={('films' if k else '3-D'): [float(v[0]), float(v[1])]
                             for k, v in step.items()},
               jax_compile_s=c, jax_s_per_cell_step=float(rate), jax_film_step_s=f_step,
               jax_film_compile_s=f_compile, overhead_s_per_cell=over)
    emit(LOG, _p0('estimate', 'hours', None, dict(table=table, fit=fit)))
    for k, v in table.items():
        print(f'phase {k}: {json.dumps(v)}')
    print('fit: ' + json.dumps(fit))


PARTS = {'provenance': part_provenance, 'v4': part_v4, 'v3': part_v3, 'v5': part_v5,
         'v5z': lambda: part_v5(v5z_cases()), 'v6': part_v6, 'mutants': part_mutants,
         'smoke': part_smoke, 'agree': part_agree,
         'timing-low': lambda: part_timing(('low',)), 'timing-jax': lambda: part_timing(('jax',)),
         'timing-rcwa': lambda: part_timing(('rcwa',)), 'estimate': part_estimate}


def phase0_table():
    rows = [r for r in load(LOG).values() if r.get('kind') == 'phase0']
    print(f'\n{"check":22s} {"scope":34s} {"result":13s} detail')
    by = {}
    for r in sorted(rows, key=lambda r: r['key']):
        by.setdefault(r['check'], []).append(r)
        d = {k: v for k, v in r['detail'].items()
             if k not in ('fam', 'host_sha', 'exec_sha', 'census')}
        verdict = ('DID NOT APPLY' if r.get('applied') is False else 'INFO'
                   if r['passed'] is None else 'PASS' if r['passed'] else 'FAIL')
        print(f'{r["check"]:22s} {r["scope"]:34s} {verdict:13s} '
              f'{json.dumps(d, default=_json)[:150]}')
    print('\nsummary: ' + '; '.join(f'{c} {sum(bool(r["passed"]) for r in v)}/{len(v)}'
                                    for c, v in sorted(by.items()) if c != 'estimate'))
    print(f'rows: {LOG}')


# -- phases 1-3 ------------------------------------------------------------------------------------

def rcwa_key(name, variant, N, wl, theta):
    return f'{name}|{variant}|N={N}|wl={wl}|theta={theta}'


def rcwa_row(done, name, variant, N, wl, theta=0.0):
    """The RCWA row of one point, solved and appended unless ``done`` has it."""
    k = rcwa_key(name, variant, N, wl, theta)
    if k not in done:
        t0 = time.perf_counter()
        row = dict(kind='rcwa', key=k, prov=PROV['rcwa'], scene=name, variant=variant,
                   N=N, wl=wl, theta=theta, **rcwa_point(name, variant, N, wl, theta))
        row['wall'] = dict(total=time.perf_counter() - t0)
        emit(RCWA_LOG, row)
        done[k] = row
    return done[k]


def film_needs(film):
    """The exact answers of one film: unpatterned layer and bare substrate, both incidences."""
    return [(FILMS[film], v, 0, wl, th) for th in (0.0, OBLIQUE) for v in ('film', 'bare')
            for wl in WAVELENGTHS]


def rcwa_ladder(rc, name):
    """A pillar scene's flush reference, u_ref and K1 twin reference (plan v3 section 9), and
    ``needs``, the RCWA points still missing: the conditional rungs follow from those present,
    so :func:`run_rcwa` asks again until nothing is missing, and the report calls a scene
    incomplete until then. u_ref is the largest distance of N = 11, 13, 15 from N = 17 over the
    three wavelengths; above 1e-3 the shifted estimate (N = 15, 17, 19 against 21 at wavelength
    1) is taken if larger. The twin runs at N = 15 and 17, and at 19 and 21 too when those two
    differ by more than 2 u_ref. R: N = 31 .. 61, u_ref from 31/41/51 against 61, twin 51/61.
    Every reference is the largest N computed at its wavelength."""
    ridge = name == 'R'
    pols = ('s', 'p') if ridge else ('s',)
    ladder, twins = ((31, 41, 51, 61), (51, 61)) if ridge else ((9, 11, 13, 15, 17), (15, 17))
    rows = {v: {(r['N'], r['wl']): r for r in rc.values()
                if r.get('kind') == 'rcwa' and r.get('scene') == name and r.get('variant') == v
                and r.get('theta') == 0.0} for v in ('flush', 'k1')}

    def missing(v, Ns, wls=WAVELENGTHS):
        return [(name, v, N, wl, 0.0) for N in Ns for wl in wls if (N, wl) not in rows[v]]

    def dist(v, a, b):
        return worst(*(abs(_rc(rows[v][a], p, m) - _rc(rows[v][b], p, m)) for p in pols
                       for m in ('r', 't')))

    def reference(v):
        best = {}
        for N, wl in rows[v]:
            best[wl] = max(best.get(wl, N), N)
        return ({(wl, m, p): _rc(rows[v][(N, wl)], p, m) for wl, N in best.items() for p in pols
                 for m in ('r', 't')}, {wl: rows[v][(N, wl)].get('thickness') for wl, N in
                                        best.items()})

    out = dict(ref={}, u=float('inf'), twin={}, extended=[],
               needs=missing('flush', ladder) + missing('k1', twins))
    if out['needs']:
        return out
    top, lower = (61, (31, 41, 51)) if ridge else (17, (11, 13, 15))
    u = worst(*(dist('flush', (N, wl), (top, wl)) for wl in WAVELENGTHS for N in lower))
    if not ridge and not u <= 1e-3:
        out['extended'].append('flush 19/21')
        out['needs'] = missing('flush', (19, 21), (1.0,))
        if out['needs']:
            return out
        u = worst(u, *(dist('flush', (N, 1.0), (21, 1.0)) for N in (15, 17, 19)))
    spread = worst(*(dist('k1', (twins[0], wl), (twins[1], wl)) for wl in WAVELENGTHS))
    if not ridge and not spread <= 2 * u:
        out['extended'].append('k1 19/21')
        out['needs'] = missing('k1', (19, 21))
        if out['needs']:
            return out
    (ref, _), (twin, thick) = reference('flush'), reference('k1')
    out.update(ref=ref, u=u, twin=twin, twin_thickness=thick, spread=spread)
    return out


def run_rcwa(scenes):
    done = load(RCWA_LOG)
    for name in scenes:
        if name in FILMS:
            for need in film_needs(name):
                rcwa_row(done, *need)
            continue
        while True:                       # the conditional rungs follow from those present
            needs = rcwa_ladder(done, name)['needs']
            if not needs:
                break
            for need in needs:
                rcwa_row(done, *need)


def _rc(row, pol, mon):
    return complex(*row[pol][mon])


def anchor_specs():
    out = []
    for f in FILMS:
        for theta, pol in ((0.0, 's'), (OBLIQUE, 'p')):
            for zeta in ZETAS:
                for n in rungs_of(f):
                    out += [spec(f, n, zeta, arm, pol=pol, theta=theta)
                            for arm in ('2', '1a', '3w', 'PA')]
                    out.append(spec(f, n, zeta, 'none', 'bare', pol=pol, theta=theta))
                out.append(spec(f, rungs_of(f)[-1], zeta, '2', pol=pol, theta=theta,
                                periods=160))
    return sorted(out, key=lambda s: s['n'])


def pillar_specs(name, minimal=False):
    rungs = rungs_of(name, minimal)
    pols = ('s', 'p') if name == 'R' else ('s',)
    n_c, n_f = richardson_pair(rungs)
    out = [spec(name, n, z, arm, pol=pol) for n in rungs for z in ZETAS for pol in pols
           for arm in ('2', '1b', 'P3')]
    out += [spec(name, n, 0.25, 'none', 'k1', pol=pol) for n in (n_c, n_f) for pol in pols]
    if name in ('A', 'B', "C'"):
        out += [spec(name, n, 0.25, 'none', 'w') for n in rungs]
    out += [spec(name, n, 0.25, arm, pol=pol) for n in rungs[:2] for arm in ('1a', '3w')
            for pol in pols]
    if name == 'R':
        out += [spec('R', n, 0.25, arm, pol='p') for n in rungs for arm in ('1bL', '1bR')]
    out += [spec(name, rungs[1], 0.25, '2', pol=pol, periods=120) for pol in pols]
    return sorted(out, key=lambda s: (s['n'], s['arm']))


# -- phase 4: the rule by script -----------------------------------------------------------------

def fit(ns, ds):
    """Least-squares order ``q`` (``d ~ n^-q``) and the RMS of its natural-log residuals; NaN
    for fewer than three rungs or any non-finite or non-positive difference."""
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(ds, float))
    if len(x) < 3 or not np.all(np.isfinite(y)):
        return float('nan'), float('nan')
    slope, icept = np.polyfit(x, y, 1)
    return float(-slope), float(np.sqrt(np.mean((y - slope * x - icept) ** 2)))


def observed_order(ns, values):
    """Richardson's observed order from three rungs, ``|T1-T2|/|T2-T3|`` solved for p."""
    h = [1.0 / n for n in ns]
    target = abs(values[0] - values[1]) / max(abs(values[1] - values[2]), 1e-300)
    if not math.isfinite(target):
        return float('nan')
    f = lambda p: (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - target   # noqa: E731
    lo, hi = 1e-3, 8.0
    if f(lo) * f(hi) > 0:
        return float('nan')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def consistent(ref, t_f, t_c, n_f, n_c, u):
    """The Richardson segment test: ``(passed, miss, margin)``; NaN anywhere fails it."""
    a = t_f + (t_f - t_c) / ((n_f / n_c) ** 2 - 1)
    b = t_f + (t_f - t_c) / ((n_f / n_c) ** 1.4 - 1)
    d = b - a
    if not (cmath.isfinite(a) and cmath.isfinite(d) and cmath.isfinite(ref)):
        return False, float('nan'), float('nan')
    s = min(max(((ref - a) * d.conjugate()).real / abs(d) ** 2, 0.0), 1.0) if abs(d) else 0.0
    miss, margin = abs(ref - (a + s * d)), 3 * u + 0.1 * abs(t_f - t_c)
    return bool(miss <= margin), miss, margin


class Book:
    """What the report could not use, by reason: missing (or 'error') rows, rows whose harness
    checks failed, and invalid runs (refused, or failing V1/V2) whose amplitudes became NaN."""

    def __init__(self):
        self.missing, self.harness, self.invalid = [], [], []


def run_state(rows, s):
    """``(state, row)`` of one spec: 'ok', 'missing' (absent or 'error'), 'refused', 'harness'
    (V3 or V4 failed) or 'invalid' (V1 or V2 failed); a run at its default periods failing V1
    stands on its repeat at twice the periods."""
    r = rows.get(key_of(s))
    if r is None or r.get('status') == 'error':
        return 'missing', r
    if r['status'] == 'refused':
        return 'refused', r
    if not (r['V3']['passed'] and r['V4']['passed']):
        return 'harness', r
    default = 80 if s['scene'] in FILMS else 60
    if not r['V1']['passed'] and s['periods'] == default:
        return run_state(rows, dict(s, periods=2 * default))
    if not (r['V1']['passed'] and r['V2']['passed']):
        return 'invalid', r
    return 'ok', r


def cells(rows, scene, arm, n, zetas, pols, variant='flush', periods=None, theta=0.0, book=None,
          raw=False):
    """``{(zeta, wl, mon, pol): T}`` for one (arm, rung), None when a row is missing or failed
    a harness check (noted in ``book``); an invalid run's cells are NaN. ``raw`` reads the row
    itself, V1 not asked and no repeat taken (the run-length floors F and F_s)."""
    out = {}
    for z in zetas:
        for pol in pols:
            s = spec(scene, n, z, arm, variant, pol=pol, theta=theta, periods=periods)
            if raw:
                r = rows.get(key_of(s))
                state = ('missing' if r is None or r.get('status') == 'error' else
                         'refused' if r['status'] == 'refused' else
                         'harness' if not (r['V3']['passed'] and r['V4']['passed']) else
                         'ok' if r['V2']['passed'] else 'invalid')
            else:
                state, r = run_state(rows, s)
            if state in ('missing', 'harness'):
                if book is not None:
                    getattr(book, state).append(key_of(s))
                return None
            nan = state != 'ok'
            if nan and book is not None:
                book.invalid.append(f'{key_of(s)} ({state})')
            for i, wl in enumerate(WAVELENGTHS):
                for mon in ('r', 't'):
                    out[(z, wl, mon, pol)] = (complex('nan') if nan else complex(*r[mon][i]))
    return out


def dmax(a, b):
    return worst(*(abs(a[k] - b[k]) for k in a))


def deembed(t, g):
    """Cells referred to the plane ``g`` lower (normal incidence, vacuum above): ``r
    exp(-2 i k0 g)``, ``t exp(-i k0 g)``, ``k0 = 2 pi/lambda``; exact on RCWA to 4e-16."""
    return {k: v * cmath.exp(-1j * (2 if k[2] == 'r' else 1) * 2 * math.pi / k[1] * g)
            for k, v in t.items()}


def anchor_report(rows, rc, book):
    """Step (1), per film, incidence and placement: ``(cases, stop)``; gaps go to ``book``."""
    cases, stop = [], []
    for f, of in FILMS.items():
        for theta, pol in ((0.0, 's'), (OBLIQUE, 'p')):
            exact = {v: {wl: rc.get(rcwa_key(of, v, 0, wl, theta)) for wl in WAVELENGTHS}
                     for v in ('film', 'bare')}
            if any(r is None for v in exact.values() for r in v.values()):
                book.missing.append(f'{f} theta {theta}: the exact answer (run rcwa)')
                continue
            for zeta in ZETAS:
                ns = rungs_of(f)
                T = {arm: {n: cells(rows, f, arm, n, (zeta,), (pol,), 'bare' if arm == 'none'
                                    else 'flush', theta=theta, book=book) for n in ns}
                     for arm in ('2', '1a', '3w', 'PA', 'none')}
                long = cells(rows, f, '2', ns[-1], (zeta,), (pol,), periods=160, theta=theta,
                             book=book, raw=True)
                short = cells(rows, f, '2', ns[-1], (zeta,), (pol,), theta=theta, book=book,
                              raw=True)
                if any(v is None for a in T.values() for v in a.values()) or not long or not short:
                    continue

                def err(t, v):
                    return worst(*(abs(t[(zeta, wl, m, pol)] - _rc(exact[v][wl], pol, m))
                                   for wl in WAVELENGTHS for m in ('r', 't')))
                e2 = [err(T['2'][n], 'film') for n in ns]
                k0 = [err(T['none'][n], 'bare') for n in ns]
                ratio = [k / e for k, e in zip(k0, e2)]
                free = [n for n, r in zip(ns, ratio) if r <= 3 * ratio[0]]
                p2, rms2 = fit(free, [e2[ns.index(n)] for n in free])
                F = worst(1e-9, 10 * dmax(long, short))
                row = dict(film=f, theta=theta, zeta=zeta, e2=e2, K0=k0, free=free, p2=p2,
                           rms2=rms2, F=F)
                for arm in ('1a', '3w', 'PA'):
                    d = [dmax(T[arm][n], T['2'][n]) for n in ns]
                    q, rms = fit(ns, d)
                    second = (least(p2, q) >= 1.7 and rms <= 0.1) or all(v < F for v in d)
                    row[arm] = dict(d=d, q=q, rms=rms, second_order=bool(second))
                # P_A must fit an order below 1.7 (section 11, whatever its RMS) and show at
                # least once above the run-length floor; a NaN order stops too.
                pa = row['PA']
                pa['power'] = bool(pa['q'] < 1.7 and any(v >= F for v in pa['d']))
                if not (p2 >= 1.8) or len(free) < 3 or not pa['power']:
                    stop.append(f'{f} theta {theta} zeta {zeta}: p2 {p2:.3f}, floor-free {free}, '
                                f"P_A order {pa['q']:.3f} (below 1.7 and above F: {pa['power']})")
                cases.append(row)
    return cases, stop


def pillar_report(rows, rc, book, minimal=False):
    """Per scene: consistency, judgeability, the rim; and the totals that never rank."""
    out = {}
    for name in SCENES:
        rungs = rungs_of(name, minimal)
        pols = ('s', 'p') if name == 'R' else ('s',)
        n_c, n_f = richardson_pair(rungs)
        T = {arm: {n: cells(rows, name, arm, n, ZETAS, pols, book=book) for n in rungs}
             for arm in ('2', '1b', 'P3')}
        lad = rcwa_ladder(rc, name)
        if lad['needs']:
            book.missing.append(f"{name}: RCWA {', '.join(rcwa_key(*k) for k in lad['needs'][:4])}")
        ref, u, tw = lad['ref'], lad['u'], lad['twin']
        k1 = {n: cells(rows, name, 'none', n, (0.25,), pols, 'k1', book=book) for n in (n_c, n_f)}
        rep = cells(rows, name, '2', rungs[1], (0.25,), pols, periods=120, book=book, raw=True)
        short = cells(rows, name, '2', rungs[1], (0.25,), pols, book=book, raw=True)
        if (any(v is None for a in T.values() for v in a.values()) or rep is None or short is None
                or any(v is None for v in k1.values()) or lad['needs']):
            out[name] = dict(complete=False)
            continue
        # The twin's total thickness is the K1 row's top (the same k1_gap on both sides).
        tops = {rows[key_of(spec(name, n, 0.25, 'none', 'k1', pol=pol))]['top']
                for n in (n_c, n_f) for pol in pols}
        twin_tops = {sum(t) for t in lad['twin_thickness'].values()}
        if any(abs(a - b) > 1e-12 for a in tops for b in twin_tops):
            book.harness.append(f'{name}: K1 twin thickness {twin_tops} is not the fd top {tops}')
        F = worst(1e-7, 10 * dmax(rep, short))
        sc = dict(complete=True, F=F, u_ref=u, rungs=rungs, rcwa_extended=lad['extended'])
        # (3) consistency: every cell, both placements (and polarisations on R)
        for arm in ('2', '1b'):
            tests = [consistent(ref[(wl, m, pol)], T[arm][n_f][(z, wl, m, pol)],
                                T[arm][n_c][(z, wl, m, pol)], n_f, n_c, u)
                     for (z, wl, m, pol) in T[arm][n_f]]
            sc['consistent_' + arm] = all(t[0] for t in tests)
            sc['worst_' + arm] = worst(*(t[1] / t[2] for t in tests))
        tests = [consistent(tw[(wl, m, pol)], k1[n_f][c], k1[n_c][c], n_f, n_c, u)
                 for c in k1[n_f] for (_, wl, m, pol) in [c]]
        sc['judgeable'] = all(t[0] for t in tests)
        sc['worst_K1'] = worst(*(t[1] / t[2] for t in tests))
        # (4) the rim, per placement and polarisation
        rim = []
        for z in ZETAS:
            for pol in pols:
                sel = lambda d: {k: v for k, v in d.items() if k[0] == z and k[3] == pol}  # noqa
                t2 = {n: sel(T['2'][n]) for n in rungs}
                sigma = dmax(t2[n_f], t2[n_c])
                three = rungs[-3:]
                orders = [observed_order(three, [t2[n][k] for n in three]) for k in t2[n_f]
                          if len(three) == 3 and all(abs(t2[a][k] - t2[b][k]) >= 10 * F
                                                     for a, b in zip(three, three[1:]))]
                orders = [o for o in orders if np.isfinite(o)]
                p_sigma = float(np.median(orders)) if orders else float('nan')
                entry = dict(zeta=z, pol=pol, sigma=sigma, p_sigma=p_sigma)
                for arm in ('P3', '1b'):
                    d = {n: dmax(sel(T[arm][n]), t2[n]) for n in rungs}
                    use = [n for n in rungs if not d[n] < F]       # a NaN stays, and fails
                    q, rms = fit(use, [d[n] for n in use])
                    entry[arm] = dict(d=[d[n] for n in rungs], q=q, rms=rms, rungs=use)
                p3, b = entry['P3'], entry['1b']
                entry['power'] = bool(p3['rms'] <= 0.15 and p3['d'][-1] >= 10 * F
                                      and p3['q'] <= p_sigma - 0.6)
                entry['material'] = bool(not b['d'][-1] < 0.05 * sigma)
                entry['escalate'] = bool(entry['power'] and entry['material']
                                         and not (b['rms'] <= 0.15 and b['q'] > p_sigma - 0.3))
                rim.append(entry)
        sc['rim'] = rim
        # reported, never ranking: each arm's total against RCWA at each rung
        tot = {}
        for arm in ('2', '1b', 'P3', '1a', '3w'):
            for n in rungs:
                t = T.get(arm, {}).get(n) or cells(rows, name, arm, n, (0.25,), pols)
                if t:
                    tot[f'{arm}@{n}'] = worst(*(abs(v - ref[(k[1], k[2], k[3])])
                                                for k, v in t.items()))
        sc['totals'] = tot
        # 1b's total over arm 2's; an anchor-first-order arm below arm 2 is cancellation
        sc['ratio_1b'] = {n: tot[f'1b@{n}'] / tot[f'2@{n}'] for n in rungs if f'1b@{n}' in tot}
        sc['cancellation'] = [f'{a}@{n}' for a in ('1a', '3w') for n in rungs
                              if f'{a}@{n}' in tot and tot[f'{a}@{n}'] < tot[f'2@{n}']]
        # K1 against its twin (both at g + H), K1 and W against the flush reference (de-embedded
        # by their lift g from g + H to H, the flush reference plane)
        for label, variant, ns, against in (('K1-twin', 'k1', (n_c, n_f), tw),
                                            ('K1-flush', 'k1', (n_c, n_f), ref),
                                            ('W-flush', 'w', rungs, ref)):
            for n in ns:
                t = cells(rows, name, 'none', n, (0.25,), pols if variant == 'k1' else ('s',),
                          variant)
                if t and against:
                    if label != 'K1-twin':
                        t = deembed(t, k1_gap(name) if variant == 'k1' else 1.0 / n)
                    sc.setdefault('controls', {})[f'{label}@{n}'] = worst(
                        *(abs(v - against[(k[1], k[2], k[3])]) for k, v in t.items()))
        out[name] = sc
    return out


def phase0_gate(rows):
    """``(missing, failed, probes)`` over :func:`phase0_expected`: the V5 and V6 probes that
    failed (step 2), and every other row absent, stale or failed (a mutant must have applied)."""
    missing, failed, probes = [], [], []
    for check, scope in phase0_expected():
        r = rows.get(f'phase0|{check}|{scope}')
        if r is None:
            missing.append(f'{check} {scope}')
        elif r['passed'] is not True or (check.startswith('mutant')
                                         and r.get('applied') is not True):
            # a probe that raised has not measured anything: that is the harness's, not step 2's
            probe = check in ('V5', 'V6') and 'error' not in r['detail']
            (probes if probe else failed).append(f'{check} {scope}')
    return missing, failed, probes


def validity(rows, book, minimal=False):
    """Step (2): every registered run of arm 2 on the pillar scenes valid, including its phase-0
    digest at the lowest rung; the missing ones go to ``book``."""
    bad = []
    for name in SCENES:
        low = rungs_of(name, minimal)[0]
        for s in pillar_specs(name, minimal):
            if s['arm'] != '2' or s['variant'] != 'flush':
                continue
            state, r = run_state(rows, s)
            if state == 'missing':
                book.missing.append(key_of(s))
                continue
            if state != 'ok':
                bad.append(f"{key_of(s)}: {state}"
                           + (f" ({r.get('error', '')[:80]})" if state == 'refused' else ''))
                continue
            if s['n'] == low:             # the executor digest equals phase 0's lowering
                p0 = rows.get(f"phase0|V3|{name}|{low}|{s['zeta']}|2")
                if p0 is None or p0['detail'].get('host_sha') != r['V3'].get('host_sha'):
                    bad.append(f'{key_of(s)}: phase-0 digest')
    return bad


def report(minimal=False):
    rows, rc = load(LOG), load(RCWA_LOG)
    missing0, failed0, probes = phase0_gate(rows)
    anchor_book, valid_book, pillar_book = Book(), Book(), Book()
    cases, stop = anchor_report(rows, rc, anchor_book)
    bad = validity(rows, valid_book, minimal) + [f'{p}: failed' for p in probes]
    scenes = pillar_report(rows, rc, pillar_book, minimal)
    print(f'\n== phase 0: {len(phase0_expected())} rows expected, {len(missing0)} missing, '
          f'{len(failed0)} failed, {len(probes)} V5/V6 probes failed')
    print('\n== (1) anchor: arm 2 error order p2 over the floor-free rungs; junction terms d')
    for c in cases:
        print(f"{c['film']:4s} theta {c['theta']:4.0f} zeta {c['zeta']}: p2 {c['p2']:.3f} "
              f"(rms {c['rms2']:.3f}, rungs {c['free']}) F {c['F']:.1e} | "
              + ' | '.join(f"{a} q {c[a]['q']:.2f} rms {c[a]['rms']:.2f}"
                           f"{' 2nd' if c[a]['second_order'] else ''}"
                           for a in ('1a', '3w', 'PA')))
    for arm in ('1a', '3w', 'PA'):          # reported: the arm's anchor order, over every case
        if cases:
            order = least(*(least(c['p2'], c[arm]['q']) for c in cases))
            print(f"anchor order of {arm}: {order:.3f}; second order on J2 in every case: "
                  f"{all(c[arm]['second_order'] for c in cases)}")
    print('\n== pillar scenes (totals against RCWA are reported and never rank)')
    for name, sc in scenes.items():
        if not sc.get('complete'):
            print(f'{name}: incomplete')
            continue
        print(f"{name}: F_s {sc['F']:.1e} u_ref {sc['u_ref']:.1e} judgeable {sc['judgeable']} "
              f"({sc['worst_K1']:.2f}) consistent 2 {sc['consistent_2']} ({sc['worst_2']:.2f}) "
              f"1b {sc['consistent_1b']} ({sc['worst_1b']:.2f})"
              + (f" RCWA extended: {sc['rcwa_extended']}" if sc['rcwa_extended'] else ''))
        print('   totals ' + ' '.join(f'{k} {v:.2e}' for k, v in sc['totals'].items()))
        print('   1b/2 ' + ' '.join(f'{n}: {v:.2f}' for n, v in sc['ratio_1b'].items())
              + (f"; lower by cancellation: {', '.join(sc['cancellation'])}"
                 if sc['cancellation'] else '')
              + '; controls ' + ' '.join(f'{k} {v:.2e}'
                                          for k, v in sc.get('controls', {}).items()))
        for e in sc['rim']:
            print(f"   rim zeta {e['zeta']} {e['pol']}: p_sigma {e['p_sigma']:.2f} sigma "
                  f"{e['sigma']:.1e} | P3 q {e['P3']['q']:.2f} rms {e['P3']['rms']:.2f} "
                  f"power {e['power']} | 1b d_f {e['1b']['d'][-1]:.1e} q {e['1b']['q']:.2f} "
                  f"rms {e['1b']['rms']:.2f} material {e['material']} escalate {e['escalate']}")
    invalid = anchor_book.invalid + pillar_book.invalid
    if invalid:
        print(f'\ninvalid runs read as NaN ({len(invalid)}): ' + '; '.join(invalid[:8]))
    three_d = [n for n in SCENES if n != 'R']
    judge = [n for n in three_d if scenes[n].get('judgeable')]
    if missing0 or failed0:
        decision = 'incomplete (phase 0)'
        why = [f'missing: {m}' for m in missing0[:6]] + [f'failed: {m}' for m in failed0[:6]]
    elif anchor_book.harness:
        decision, why = 'incomplete (harness)', anchor_book.harness[:10]
    elif stop:
        decision, why = 'stopped (no decision)', stop
    elif anchor_book.missing:
        decision, why = 'incomplete', ['anchor: ' + m for m in anchor_book.missing[:10]]
    elif bad:
        decision, why = "'refuse' stays", [f'invalid: {b}' for b in bad[:10]]
    elif valid_book.missing or pillar_book.missing:
        decision = 'incomplete'
        why = [f'missing: {m}' for m in (valid_book.missing + pillar_book.missing)[:10]]
    elif pillar_book.harness:
        decision, why = 'incomplete (harness)', pillar_book.harness[:10]
    elif len(judge) < 2:
        decision, why = 'owner (does not choose)', [f'judgeable 3-D scenes: {judge}']
    else:
        # (3) over every judgeable scene at once: 'refuse' takes precedence
        bad2 = [n for n, sc in scenes.items() if sc.get('judgeable') and not sc['consistent_2']]
        both = [n for n in bad2 if not scenes[n]['consistent_1b']]
        if both:
            decision, why = "'refuse' stays", [f'{n}: 2 and 1b inconsistent' for n in both]
        elif bad2:
            decision, why = 'owner (does not choose)', [f'{n}: 2 inconsistent, 1b not'
                                                        for n in bad2]
        else:
            rims = [(n, e) for n, sc in scenes.items() for e in sc['rim']]
            if not any(e['power'] for _, e in rims):
                decision, why = 'owner (does not choose)', ['no scene-placement has rim power']
            elif any(e['escalate'] for _, e in rims):
                decision = 'owner (does not choose)'
                why = [f"{n} zeta {e['zeta']} {e['pol']}: rim escalates" for n, e in rims
                       if e['escalate']]
            else:
                decision, why = "'split' (arm 2)", ['every step passed']
    print(f'\n== decision: {decision}\n   ' + '\n   '.join(why))
    emit(REPORT_LOG, dict(kind='report', key=f'report|{time.time():.0f}', prov=PROV['digest'],
                          decision=decision, why=why, anchor=cases, scenes=scenes,
                          phase0=dict(missing=missing0, failed=failed0, probes=probes),
                          invalid=invalid, minimal=minimal), quiet=True)
    return decision


def ready(step, anyway):
    """Refuse to start a long phase whose inputs say it cannot lead anywhere (``--anyway``)."""
    rows = load(LOG)
    missing0, failed0, probes = phase0_gate(rows)
    blockers = [f'phase 0 missing: {m}' for m in missing0[:5]]
    blockers += [f'phase 0 failed: {m}' for m in (failed0 + probes)[:5]]
    if step == 'pillars':
        book = Book()
        cases, stop = anchor_report(rows, load(RCWA_LOG), book)
        blockers += [f'anchor stopped: {s}' for s in stop[:5]]
        blockers += [f'anchor missing: {m}' for m in book.missing[:5]]
        blockers += [f'anchor harness: {m}' for m in book.harness[:5]]
    if blockers and not anyway:
        print(f'not starting {step}:\n   ' + '\n   '.join(blockers)
              + '\n(--anyway runs it regardless)')
        return False
    return True


def status(minimal=False):
    """What each phase's rows hold now: every registered run by state, and the RCWA points
    still missing (the conditional rungs included once their condition is known)."""
    rows, rc = load(LOG), load(RCWA_LOG)
    missing0, failed0, probes = phase0_gate(rows)
    print(f'phase 0: {len(phase0_expected())} gate rows; missing {len(missing0)}, failed '
          f'{len(failed0)}, V5/V6 probes failed {len(probes)}'
          + ''.join(f'\n   {m}' for m in (missing0 + failed0 + probes)[:8]))
    films = [k for f in FILMS for k in film_needs(f) if rcwa_key(*k) not in rc]
    print(f'phase 1a (films): {4 * len(film_needs("FA"))} RCWA points, {len(films)} missing')
    for name in SCENES:
        lad = rcwa_ladder(rc, name)
        print(f'phase 1b {name}: RCWA {len(lad["needs"])} missing'
              + (f' (extended: {lad["extended"]})' if lad['extended'] else ''))
    for label, specs in ([('phase 2 anchor', anchor_specs())]
                         + [(f'phase 3 {nm}', pillar_specs(nm, minimal)) for nm in SCENES]):
        states = {}
        for s in specs:
            states.setdefault(run_state(rows, s)[0], []).append(key_of(s))
        print(f'{label}: {len(specs)} runs: '
              + ', '.join(f'{k} {len(v)}' for k, v in sorted(states.items()))
              + ''.join(f'\n   {k}: {v[0]}' for k, v in sorted(states.items()) if k != 'ok'))


def main(argv):
    global PROV
    PROV = provenance()
    cmd = argv[0] if argv else 'phase0'
    opt = lambda name: argv[argv.index(name) + 1] if name in argv else None    # noqa: E731
    scene = opt('--scene')
    scene = ALIASES.get(scene, scene)
    if cmd == 'phase0':
        parts = [a for i, a in enumerate(argv) if i and argv[i - 1] == '--part'] or list(PARTS)
        for part in parts:
            PARTS[part]()
        phase0_table()
    elif cmd == 'rcwa':
        run_rcwa([scene] if scene else list(SCENES) + list(FILMS))
    elif cmd == 'anchor':
        if not ready('anchor', '--anyway' in argv):
            return
        run_specs(anchor_specs())
        book = Book()
        cases, stop = anchor_report(load(LOG), load(RCWA_LOG), book)
        emit(REPORT_LOG, dict(kind='anchor', key=f'anchor|{time.time():.0f}', prov=PROV['digest'],
                              stop=stop, missing=book.missing, harness=book.harness,
                              invalid=book.invalid, cases=cases))
        print('INCOMPLETE (harness): ' + '; '.join(book.harness) if book.harness else
              'STOP: ' + '; '.join(stop) if stop else
              'INCOMPLETE: ' + '; '.join(book.missing) if book.missing else
              'anchor passed: continue to pillars')
    elif cmd == 'pillars':
        if not ready('pillars', '--anyway' in argv):
            return
        for name in ([scene] if scene else list(SCENES)):
            run_specs(pillar_specs(name, '--minimal' in argv))
    elif cmd == 'report':
        report('--minimal' in argv)
    elif cmd == 'status':
        status('--minimal' in argv)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
