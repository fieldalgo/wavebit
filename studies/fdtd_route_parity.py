"""Compare the two historical FDTD engines; run from the repository root.

Integration note: this replay uses studies._fdtd_legacy to keep the archived
top-to-bottom layers, downward z coordinates and corner phase. It validates
the private engines; public API conventions are covered by separate tests.
The narrative below records the original experiment and its route names.

    python studies/fdtd_route_parity.py --scene all --output studies/results/fdtd_route_parity.json
    python studies/fdtd_route_parity.py --list
    python studies/fdtd_route_parity.py --scene slab-1d-dielectric-normal --output smoke.json

**The question.** ``wavebit.fdtd`` has two routes (``src/wavebit/manifest.py``, ``fdtd.routes``).
Omitting ``backend`` selects ``legacy_numpy`` (``_core/fdtd.py``): NumPy only, not
differentiable, top incidence only. An explicit ``backend`` selects ``common``
(``_core/scene_fdtd.py`` over ``_core/grid_fdtd.py``): three backends, differentiable, both
incidence sides. They are **two independent discretisations that share no Maxwell update**,
with different defaults, clearance layouts, sources and normalisations. Every number in
``docs/fdtd-alpha.md`` was measured on the legacy route, which refuses the features the alpha
advertises. So a route decision needs one fact nobody had measured: on the same scene, do the
two routes give the same answer?

**Why this did not exist.** Nothing in the repository ran one scene through both routes. The
tests that mention "legacy" compare within one route, and ``studies/scene_hash.py`` does not
cover FDTD. This repository removes a path only after *measuring* that it adds nothing: the
block path went after three geometries chosen to favour it came out bit-identical. Two
independent discretisations will not agree bit for bit, and should not be expected to. So
this harness does not stop at "how far apart are the routes". It reports three numbers per
scene and per channel, against the same reference:

    e1 = |legacy - reference|     legacy's own error
    e2 = |common - reference|     common's own error
    d  = |legacy - common|        the route difference

If ``d`` is small next to ``e1`` and ``e2``, the two routes make the *same* error. They are
then interchangeable, and the choice between them is about features. If ``d`` is comparable
to ``e1`` and ``e2``, the route changes the answer, and that is the headline. The reference
is the one the alpha studies use: ``wavebit.rcwa(backend=0)`` at a truncation order where it
has converged, or a closed form where one exists. Its own uncertainty ``u`` is measured from
an order series in every run, never assumed. ``e1`` and ``e2`` can be interpreted only at
``>= 10 u``: a pair whose largest-``d`` channel has an error below that is filed
``one_route_below_reference_resolution`` or ``both_below_reference_resolution``, whatever its
``d / min(e)``. A claim that one route is more accurate needs ``|e1 - e2| >= 10 u``. ``d``
involves no reference, so the ``10 u`` rule does not apply to it; it is interpretable only
above both runs' settling (see Settling). The rule for ``u`` is in
:func:`reference_uncertainty`.

**What it answers, and what it cannot.** It answers, per scene, whether the two *engines*
make the same error once the one post-processing difference it can measure is removed, and
on what evidence: materials compared array by array, amplitude and power classified side
by side, each ratio taken on one channel, and each ``d`` held against how settled the runs
are. It cannot say *why* two engines differ once their inputs match: the confounds listed
under ``confounds`` stay folded into every ``d``. It cannot rank the routes where the
reference is exact to round-off (a uniform stack: ``u`` is a floor there, not a
measurement, so every ``e1 != e2`` would "rank"), and it cannot resolve a ``d`` smaller than
either run's own settling. In each of those places the output says so instead of a number.

It compares engines only where their inputs match, and whether they match is measured on
every scene (``materials``), not assumed. The first run, at 0f204f1, found that the
patterned dispersive scenes (the Drude and Lorentz rasters, the metal) did not: legacy
filled Ey's pole strength half a cell off in y from where it samples Ey's ``eps``
(``KNOWN_INPUT_DEFECTS``). There it measured that defect, not the engines, and every pair
classification said ``inputs_differ``. The legacy fix puts the fill on Ey's y half-step,
``tests/test_fdtd_pole_colocation.py`` pins it, and the committed results were rerun after
it. Nor does the harness settle anything the legacy route alone does, or what common gives
at its own defaults (``--arms matched,verbatim`` measures one such call and nothing more).

**Two metrics, neither privileged.** Every pair is classified on amplitude *and* on power,
and both are reported. Amplitude is not the stricter test on these scenes: it is the metric
the de-embedding phase (below) dominates, and power is immune to that phase. A route ranking
is emitted at scene level only where the two metrics agree (:func:`scene_verdict`). On a
metric, a ranking also needs ``|e1 - e2|`` to exceed how far the one measured confound moved
that same quantity (``confound_shift``). Each ranking carries its margin
(``ranking_margin``), and ``families`` shows whether the order survives a change of
resolution.

**Each ratio on one channel.** ``d / min(e1, e2)`` is taken on the channel that carries the
largest ``d``, with that channel's own ``e1`` and ``e2`` (:func:`pair_summary`). A maximum
``d`` over a minimum of maximum errors mixes channels, and can call a scene "same error"
when its worst channel is not. The largest per-channel ratio among channels whose errors are
both resolved is reported beside it; on small-amplitude channels it runs far higher.

**Settling.** A ``d`` below common's ``scattering_change_bound`` (the amplitude change over
its final window) or legacy's field ``residual`` cannot be told apart from what the run time
leaves unsettled. :func:`classify` then returns ``unresolved_against_settling`` and no
ranking, and records which part decided (``settling_decided_by``). Only common's part is a
bound on an amplitude change. Legacy's ``residual`` is a field tail-to-peak ratio, and the
power bound is the amplitude one times ``POWER_SETTLING_FACTOR``, which ignores the channel.
Both are proxies (``settling_is_proxy``). Where a power call is unresolved, the bound scaled
by that channel's own reference power is reported beside it
(``channel_settling_estimate``). ``settling_call_rests_on_proxy`` marks a call that neither
common's bound nor the channel-scaled estimate would make on its own.

**Round-off-floor references.** On a uniform stack or in vacuum ``u`` is ``ROUNDOFF``, so
"resolved at 10 u" tests nothing and ``d / min(e)`` is set by whichever route happens to
land near exact. There, a pair whose smaller error is more than ``RESOLVABLE`` times below
the larger is ``one_route_near_exact`` rather than a ratio class. Every classification
reports ``d / max(e)`` and ``|e1 - e2|`` beside ``d / min(e)``. Channels whose reference is
zero by symmetry are left out of the resolved per-channel ratio. ``families`` lists each
route's worst error across a scene's resolution and ``pad_scale`` variants. A smaller-error
route that changes with ``pad_scale`` alone points at the ``pml_structure_gap`` confound, not
at an engine, and ``cases`` files those scenes apart.

**The de-embedding confound, measured.** Legacy carries each monitor's phase back to the
interface with the continuum ``kz`` (``_core/fdtd.py``, ``_extract``); common with the
discrete Yee ``kz`` (``_core/scene_fdtd.py``, ``yee_longitudinal_phase``). That is a
post-processing choice, not an engine difference; it moves only amplitude phases, and it is
``d``-sized. So every scene carries a fourth arm, ``legacy_yee_deembedded``: legacy's own
amplitudes re-de-embedded with common's formula on legacy's own ``dt``, cells and monitor
planes (:func:`yee_deembedded`), with no free parameter. ``legacy_yee_deembedded`` vs
``common`` is the engine comparison; ``legacy`` vs ``common`` stays reported as the raw one.
A residual stays. On a non-conical TM loop legacy samples its monitors and its incident
amplitude half a cell below the node (``_core/fdtd.py``, ``offset = 0.5``), and its
continuum phase runs over those half cells too. The swap is applied on the node planes, as
the reviewer's ``a4.py`` does, which leaves an ``O((kz_yee - kz) dz)`` residual on those p
channels. Recomputed from the channel rows, moving the monitor term alone onto legacy's
half-cell plane shifts their ``d`` by up to about 20% and changes no classification.

The same swap is the harness's positive control **for the raw arm**. It is a known
post-processing difference about the size of the raw ``d`` (``effect_over_raw_d``), and
``deembedding_control`` records what the raw classification made of it: ``raw_case_differs``,
and ``raw_overstates`` where the raw case sits strictly higher on the
``same_error < intermediate < route_changes_answer`` scale. Against the engine ``d`` it is
several times too large (``effect_over_corrected_d``), so it certifies nothing about the
engine arm's resolution. The engine arm's right-sized control sits at the input: the
material comparison caught a pole fill half a cell off on every patterned pole scene of the
first run, before the legacy fix. ``tests/test_fdtd_route_parity_capture.py`` keeps the
capture able to: it requires the recorded Ey fill to be the one at Ey's y half-step, which a
y-node fill is not, and every wrapper to accept what the function it replaces accepts.

**What "matched" means.** The legacy route runs *as shipped*, with the study's own keywords,
exactly as ``tests/references/gen_goldens_fdtd_table.py::_fdtd`` drives it. The grid it
actually built is then read back from ``diag['grid']``, and the common route is driven onto
that grid and schedule. It gets the same cell sizes and counts, the same ``dt`` (the
``courant`` keyword is recomputed, because the two ``dt`` formulas differ), the same source
envelope width, and the same time after the pulse peak (common's peak sits one width later,
so its ``n_periods`` grows by that width). It also gets the same PML depth and target, and
extraction planes on the same nodes. :func:`matched_common` holds each of these, and every
one is asserted against the run's own diagnostics (``match`` in the output). The material
inputs are part of "matched" too: every per-component ``eps``, ``mu`` and pole strength-times-
fill array each engine steps with is recorded as the measured runs build it
(:class:`MaterialCapture`) and compared node for node over the z range both grids share
(:func:`compare_materials`). A schedule can match exactly while one engine puts a pole's
strength half a cell off, and no check on the grid or ``dt`` sees that. A y-invariant
grating needs one trick. Common always meshes y with two cells and counts ``1/dy**2`` in its
``dt``, while legacy's 2D steppers do not. The y cell is therefore made ``1000 dx`` wide.
That is inert because the fields are exactly y-invariant (``ky0 = 0``, so every y difference
is exactly zero). A true conical mount on a 1D grating cannot match ``dt`` inside
``courant < 1``, so it runs at the legacy ``courant`` keyword. Neither can a pole scene
whose legacy ``dt`` common refuses under its pole-aware CFL bound. Those are the only two
places ``dt`` is exempt from "matched". The scene records ``dt_reason``, the ``dt`` ratio is
a confound, and ``cases`` lists these scenes under ``dt exempt from matching``.

**What cannot be matched** is part of every ``d``. It is listed per scene under
``confounds`` so no reader mistakes the route difference for a pure discretisation
difference. Legacy injects through a TFSF plane fed by a 1D incident grid; common uses
current sheets and subtracts a reference run. They de-embed with continuum vs discrete
``kz`` (measured and removed in the ``legacy_yee_deembedded`` arm, above), and demodulate
the staggered x components differently. The PML is backed
differently (grounded end rows vs a periodic wrap), and the gap between the PML and the
structure differs. The pulse delay and carrier differ, and the order of subpixel averaging
at corners may too. ``--common-pad legacy`` (the default) sizes common's clearance so its
upper gap reaches legacy's; ``--common-pad default`` keeps common's own four-pad layout.
``--arms matched,verbatim`` adds a ``common_verbatim`` arm. That arm is common driven by the
study's keywords verbatim, so the answer a user gets by just adding ``backend=0`` is
measured too.

**Grazing orders and the termination.** An order that leaves at a small exit-medium
``cos(theta)`` is partly sent back by the termination -- the CPML, then legacy's grounded end
rows or common's periodic wrap, then the gap to the monitor -- and the two engines lay that out
differently (``pml_backing``, ``pml_structure_gap``). On the material scenes the transmitted
(0, 1) order leaves the substrate at 0.263. On magnetic-r20 and lorentz-r20/r40 the engine ``d``
sits there, and it was the two terminations' returns (2026-09-15): it did not shrink with the
grid, it is there for the pole-free material as for the Lorentz one, and with both CPMLs at
target 1e-16, 24 cells and 640 periods (the twins below) it fell to 1.1e-5 on magnetic-r20 and
1.0e-5 on lorentz-r40, below those twins' settling bounds, and to 1.9e-4 on lorentz-r20, whose
``dt`` is exempt from matching. On
lorentz-r20 neither change alone removed it. The reflected (0, -1) order (0.242) carries a
smaller return that strengthening only partly removed; drude-r20's engine ``d`` sits on the
specular (0, 0) reflection instead; and at r20 that (0, 0) ``d`` also fell under the deeper
CPML, so a non-grazing ``d`` is not guaranteed free of the termination either. So every channel
reports its ``exit_cos`` and a ``grazing`` flag (``GRAZING_COS``), ``cases`` lists the scenes
whose resolved engine ``d`` sits on a grazing channel, and each material scene has a ``-pml16``
twin: target 1e-16, 24 cells, and a longer run, because a stronger termination settles more
slowly. ``termination_control`` compares each scene with its twin channel by channel. The twin
passes ``pml_reflection`` to both routes, and the match check compares the target legacy reports
it built both CPMLs with against common's.

**Determinism.** Wall clock goes to a separate file (``--timing``, default
``<output>.timing.json``) together with the machine-load snapshots. The ``--output`` file
holds no wall clock, so two runs of one scene can be byte-compared with nothing to strip.
The BLAS/OpenMP thread pins are set before NumPy is imported and are recorded. The source
digests read CRLF as LF, so a Windows and a Linux checkout of one commit give one digest. The
``convergence_window`` diagnostic passed to common is shown not to change the answer by
``--check-inert``, which reruns common without it and compares the arrays bit for bit.

Scenes are the shipped physics scenes behind ``docs/fdtd-alpha.md``: the planar install
smoke with its ``psi`` duplicates removed, the two crossed dielectrics, the vacuum phase
control, the three crossed material rasters and the damped metal, and -- made by this study,
not shipped -- each material raster again with a strengthened termination (``-pml16``,
above). They also include the one
shipped scene that runs pure TE/TM, ``docs/fdtd.md`` section 2's binary grating, beside its
closed-form twin ``slab_uniform``. ``--list`` names them; ``--scene`` takes names, group names,
shell patterns or ``all``. Results are written after each scene, so an interrupted run
keeps its measurements. No external solver is imported.
"""
import os

#: Set before NumPy loads, so every run uses one thread and the same reduction order.
THREAD_PINS = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
for _name, _value in THREAD_PINS.items():
    os.environ.setdefault(_name, _value)

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests' / 'references'))
import wavebit  # noqa: E402
from wavebit._core import fdtd as LEGACY  # noqa: E402
from wavebit._core import scene_fdtd as COMMON  # noqa: E402
from _meep_crossed_cases import CASES as CROSSED, eps_of  # noqa: E402
from _meep_crossed_material_cases import CASES as MATERIALS, epsilon  # noqa: E402
from _meep_crossed_metal_case import CASE as METAL  # noqa: E402
from _meep_1d_cases import CASES as ONE_D, airy_rt  # noqa: E402
from test_meep_crossed_materials_golden import scene as material_scene  # noqa: E402

# A script run from outside a worktree imports the MAIN checkout, where these routes differ
# or do not exist; every number would then describe the wrong code.
if ROOT not in Path(wavebit.__file__).resolve().parents:
    raise SystemExit(f'wavebit imported from {wavebit.__file__}, not from {ROOT}')

#: Common's pulse peak is hard-coded at five widths (``scene_fdtd.py``, ``peak_time=5*width``),
#: legacy's at ``SOURCE_DELAY`` widths. The difference is added to common's run so both spend
#: the same time after the peak; asserted against ``diag['pulse']`` on every run.
COMMON_PEAK_DELAY = 5.0
#: The y cell used on a y-invariant grating, in units of dx. It only has to be large enough
#: that ``1/dy**2`` does not stop common's ``dt`` from reaching legacy's inside ``courant < 1``.
Y_INVARIANT_CELL = 1e3
#: Fraction of common's run inspected by ``convergence_window``, the analogue of legacy's
#: ``residual``. The keyword only adds tail accumulators; ``--check-inert`` proves it.
CONVERGENCE_FRACTION = .1
#: Rayleigh margin below which a channel is not compared (``_meep_crossed_cases.margin``):
#: near a Rayleigh point amplitudes converge as ``1/kz`` and legacy returns NaN deeper in.
MARGIN = .05
#: ``d`` against the smaller of two resolved own errors. At or below the first, the routes
#: make the same error; at or above the second, the route changes the answer.
SAME_ERROR_RATIO = 1/3
ROUTE_MATTERS_RATIO = 1.
#: An own error is interpretable only at ten times the reference's uncertainty.
RESOLVABLE = 10.
#: Floor on ``u`` for a uniform stack, whose RCWA answer is exact up to round-off (it matches
#: ``airy_rt`` to 4.5e-16; ``tests/references/_meep_1d_cases.py``).
ROUNDOFF = 1e-14
#: Two material arrays agree when they differ by less than this, relative to the largest
#: magnitude either holds. Both routes average one raster exactly, in different orders, so
#: matched inputs are separated by round-off alone; a misplaced pole fill moves it to O(1).
MATERIAL_TOL = 1e-9
#: A power moves by at most ``2|a| da + da**2`` when its amplitude moves by ``da``; with
#: ``|a| <= 1`` the amplitude settling bound is doubled for power. It ignores the admittance
#: factor in ``T`` (at most ``n_bot/n_top`` times the cosine ratio) and the channel's own
#: ``|a|``, so it is a proxy, not a bound.
POWER_SETTLING_FACTOR = 2.
#: An order leaving at an exit-medium cos(theta) below this is ``grazing``. Set from the material
#: scenes (2026-09-15): the transmitted (0, 1) order at 0.263 carries a return from the
#: z-termination that is the engine d on three of them and does not shrink with the grid, and
#: the reflected (0, -1) at 0.242 a smaller one. The next most grazing compared order, 0.394 on
#: the crossed dielectrics, has had no termination control, and its d is below its scene's
#: settling bound. So this is a line with one measured side, not a derived bound: every channel
#: reports its own ``exit_cos``.
GRAZING_COS = .3
#: The ``-pml16`` twins' termination: CPML target and depth together (on lorentz-r20 neither
#: alone removed the grazing return), and a longer run, because a stronger termination settles
#: more slowly. With it, legacy's residual sets the amplitude settling bound. Measured on the
#: lorentz scenes (2026-09-15), amplitude bound / engine d by run length:
#:   r20: 160 periods 6.7e-4 / 2.8e-4, 320 1.2e-4 / 2.6e-4, 480 6.3e-5 / 1.8e-4, 640 2.2e-5 / 1.9e-4
#:   r40: 320 periods 1.2e-4 / 2.1e-4, 480 3.9e-5 / 6.5e-5
#: At r20 the d stops moving from 480 and the power call resolves at 640; at r40 the d was still
#: falling at 480. So 640. Each twin records its own bound.
PML16 = dict(target=1e-16, n_pml=24, n_periods=640)
#: On a round-off-floor scene, a reference this small is zero by symmetry (a cross-polarised
#: channel of a uniform stack), and "resolved at 10 u" says nothing about its errors.
ZERO_REFERENCE = 1e-12
#: Input differences whose owner is known from reading the source; none of this is measured by
#: the run. Keyed by the prefix of a name ``compare_materials`` reports, and attached only to
#: a scene where an array with that prefix differs: after the fix, only to a recurrence.
KNOWN_INPUT_DEFECTS = {
    'pole_fill.ey': dict(
        owner='legacy',
        detail="until the fix, legacy sampled Ey's eps at y_half=True (_core/fdtd.py, "
               "_time_loop: grid.eps_for(False, False, 'y', y_half=True)) but built Ey's pole "
               'state through _Grid.pole_state -> _Grid.fill_for, which was called without '
               'y_half and defaulted it to False. The Drude/Lorentz strength therefore sat half '
               'a y cell from the eps it belongs to. Lines at 0f204f1: fdtd.py:1395 (eps), :1403 '
               '(pole_state call), :1225 (fill_for default), :1247 (the call). Entered in 1c521ea.',
        evidence='source reading. In memory, filling Ey\'s pole state at y_half=True makes '
                 'every material array agree on metal-r30 and material-lorentz-r20.',
        status='fixed: pole_state takes y_half and Ey passes y_half=True '
               '(tests/test_fdtd_pole_colocation.py); kept so that a recurrence is attributed'),
}

AMPLITUDES = ('r_s', 'r_p', 't_s', 't_p')
POWERS = ('R', 'T')


# =======================================================================================
# Scenes
# =======================================================================================

from studies._fdtd_legacy import legacy_fdtd, legacy_rcwa  # noqa: E402


def _scenes():
    """name -> scene. Every keyword is the owning study's, cited in ``source``."""
    scenes = {}
    omega = 2*np.pi/900.
    for crossed in (False, True):
        shape = (1, 2 if crossed else 1, 2)
        for material in ('dielectric', 'magnetic', 'drude'):
            for mount, (theta, phi) in (('normal', (0., 0.)), ('oblique', (.3, 0.)),
                                        ('conical', (.3, .4))):
                call = dict(structure=np.full(shape, 2.25), thickness=[100.], top=1., bot=1.44,
                            period=[100., 140.] if crossed else [100.], wavelength=900.,
                            theta=theta, phi=phi, psi=0., truncation_order=dict(x=0, y=0))
                rcwa_structure = np.full(shape, 2.25, dtype=complex)
                if material == 'magnetic':
                    call['mu'] = np.full(shape, 1.3)
                if material == 'drude':
                    call['dispersion'] = np.ones(shape, dtype=int)
                    call['poles'] = {1: [(0., .2*omega, .4*omega**2)]}
                    rcwa_structure = _closed_form_epsilon(call)
                scenes[f"slab-{'crossed' if crossed else '1d'}-{material}-{mount}"] = dict(
                    group='slab', kind='psi', call=call, study=dict(resolution=.04, n_periods=80),
                    rcwa_structure=rcwa_structure, n_top=1., n_bot=1.2,
                    orders_u=(0,), orders_v=(0,), reference_orders=(0, 1), uniform=True,
                    lossless=material != 'drude',
                    closed_form=(_airy_psi(call, rcwa_structure)
                                 if mount == 'normal' and material != 'magnetic' else None),
                    source='studies/fdtd_install_smoke.py (the psi=0 half: psi only reweights '
                           '`given`; both routes always compute both basis columns)')
    for name, resolutions, orders in (('square_18deg', (50/3, 25., 100/3), (9, 13, 17)),
                                      ('rect_28deg', (20., 30.), (13, 17, 21))):
        case = CROSSED[name]
        call = dict(wavelength=case['wavelength'], period=list(case['period']),
                    thickness=list(case['thickness']), structure=eps_of(case['index']),
                    top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
                    theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
                    psi=np.deg2rad(30.))
        for resolution in resolutions:
            scenes[f'{name}-r{resolution:.4g}'] = dict(
                group='crossed', kind='psi', call=call,
                study=dict(resolution=resolution, n_periods=120, n_pml=12, courant=.99,
                           subpixel=True, pad_scale=1., grading=None),
                rcwa_structure=call['structure'], n_top=case['n_top'], n_bot=case['n_bot'],
                orders_u=case['orders_u'], orders_v=case['orders_v'], reference_orders=orders,
                uniform=False, lossless=True, closed_form=None,
                reference_key=name,
                source='studies/fdtd_alpha_convergence.py; tests/references/_meep_crossed_cases.py')
    theta = np.deg2rad(18.)
    vacuum = dict(structure=np.ones((1, 2, 2)), thickness=[.2], top=1., bot=1.,
                  period=[.16, .16], wavelength=.8, psi=.4, theta=theta, phi=np.deg2rad(40.))
    phase = complex(np.exp(2j*np.pi/.8*np.cos(theta)*.2))
    for resolution in (25., 50.):
        for pad in (1., 1.5):
            scenes[f'vacuum-r{resolution:g}-pad{pad:g}'] = dict(
                group='vacuum', kind='psi', call=vacuum,
                study=dict(resolution=resolution, n_periods=120, n_pml=12, pad_scale=pad,
                           courant=.99, subpixel=True, grading=None),
                rcwa_structure=np.ones((1, 2, 2), dtype=complex), n_top=1., n_bot=1.,
                orders_u=(0,), orders_v=(0,), reference_orders=(0, 1), uniform=True,
                lossless=True, reference_key='vacuum',
                closed_form={'s.t_s': phase, 'p.t_p': phase, 's.r_s': 0j, 's.r_p': 0j,
                             's.t_p': 0j, 'p.r_s': 0j, 'p.r_p': 0j, 'p.t_s': 0j},
                source='tests/test_fdtd_boundary_controls.py (analytic phase '
                       'exp(2 pi i/lambda cos(theta) d))')
    for name, resolution in (('magnetic', 20.), ('drude', 20.), ('lorentz', 20.),
                             ('lorentz', 40.)):
        case = MATERIALS[name]
        call = material_scene(case)
        if case['dispersion'] is not None:
            call.update(dispersion=case['dispersion'], poles=case['poles'])
        scenes[f'material-{name}-r{resolution:g}'] = dict(
            group='materials', kind='psi', call=call,
            study=dict(resolution=resolution, n_periods=160, n_pml=12, pad_scale=1.,
                       courant=.99, subpixel=True, grading=None),
            rcwa_structure=epsilon(case, case['wavelength']), n_top=case['n_top'],
            n_bot=case['n_bot'], orders_u=case['orders_u'], orders_v=case['orders_v'],
            reference_orders=(7, 11, 15), uniform=False, lossless=case['poles'] is None,
            closed_form=None, reference_key=f'material-{name}',
            source='tests/references/test_meep_crossed_materials_golden.py::comparison')
    # The same material scenes with a strengthened termination (``termination_control``).
    for name in [n for n in scenes if n.startswith('material-')]:
        base = scenes[name]
        scenes[f'{name}-pml16'] = dict(
            base, group='materials-pml16',
            study=dict(base['study'], n_pml=PML16['n_pml'], pml_reflection=PML16['target'],
                       n_periods=PML16['n_periods']),
            termination_of=name,
            source=base['source'] + '; termination strengthened for termination_control')
        base['termination_twin'] = f'{name}-pml16'
    for resolution in (30., 60.):
        call = material_scene(METAL)
        call.update(dispersion=METAL['dispersion'], poles=METAL['poles'])
        scenes[f'metal-r{resolution:g}'] = dict(
            group='metal', kind='psi', call=call,
            study=dict(resolution=resolution, n_periods=160, n_pml=12, pad_scale=1.,
                       courant=.99, subpixel=True, grading=None),
            rcwa_structure=epsilon(METAL, METAL['wavelength']), n_top=METAL['n_top'],
            n_bot=METAL['n_bot'], orders_u=METAL['orders_u'], orders_v=METAL['orders_v'],
            reference_orders=(11, 15, 19), uniform=False, lossless=False, closed_form=None,
            reference_key='metal',
            source='studies/fdtd_metal_validation.py (coarse and fine legs)')
    # slab_uniform's reference window must hold m = +-1: they propagate in the substrate and
    # are where a broken bridge would put energy, so they are compared (against zero).
    for name, resolutions, orders in (('slab_uniform', (50.,), (1, 2)),
                                      ('binary_grating', (50., 100.), (100, 200, 400))):
        case = ONE_D[name]
        call = dict(structure=eps_of(case['index']), thickness=list(case['thickness']),
                    top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
                    period=[case['period'][0]], wavelength=case['wavelength'], theta=0.)
        uniform = name == 'slab_uniform'
        closed = None
        if uniform:
            r, t = airy_rt(case['wavelength'], case['n_top'], [case['index'].ravel()[0]],
                           list(case['thickness']), case['n_bot'])
            closed = {'te.r_s': complex(r), 'te.t_s': complex(t),
                      'tm.r_p': -complex(r), 'tm.t_p': complex(t)}
        for resolution in resolutions:
            scenes[f'{name}-r{resolution:g}'] = dict(
                group='binary', kind='tetm', legs=('te', 'tm'), call=call,
                study=dict(resolution=resolution),
                rcwa_structure=call['structure'], n_top=case['n_top'], n_bot=case['n_bot'],
                orders_u=case['orders_u'], orders_v=(0,), reference_orders=orders,
                uniform=uniform, lossless=True, closed_form=closed, reference_key=name,
                source='tests/references/gen_goldens_fdtd_table.py::_fdtd (WAVEBIT_CALL with '
                       'the resolution replaced); docs/fdtd.md section 2')
    return scenes


def _closed_form_epsilon(call):
    """The pole sum written out, independent of either solver's material evaluator."""
    eps = np.asarray(call['structure'], dtype=complex)
    omega = 2*np.pi/call['wavelength']
    for label, poles in call['poles'].items():
        for omega0, gamma, strength in poles:
            eps = eps + (np.asarray(call['dispersion']) == label)*(
                strength/(omega0**2 - omega**2 - 1j*gamma*omega))
    return eps


def _airy_psi(call, eps):
    """The normal-incidence closed form as the psi-spelled basis columns (p leg ``(-r, t)``)."""
    r, t = airy_rt(call['wavelength'], np.sqrt(complex(call['top'])),
                   [np.sqrt(complex(eps.ravel()[0]))], call['thickness'],
                   np.sqrt(complex(call['bot'])))
    r, t = complex(r), complex(t)
    return {'s.r_s': r, 's.t_s': t, 'p.r_p': -r, 'p.t_p': t,
            's.r_p': 0j, 's.t_p': 0j, 'p.r_s': 0j, 'p.t_s': 0j}


SCENES = _scenes()
GROUPS = sorted({scene['group'] for scene in SCENES.values()})


def select(patterns):
    chosen = []
    for pattern in patterns:
        hits = [name for name, scene in SCENES.items()
                if pattern == 'all' or scene['group'] == pattern or fnmatch.fnmatchcase(name, pattern)]
        if not hits:
            raise SystemExit(f'--scene {pattern!r} matches nothing; see --list')
        chosen.extend(name for name in hits if name not in chosen)
    return chosen


# =======================================================================================
# Channels, masks, energy
# =======================================================================================

def legs_of(scene):
    """``(leg label, amplitude keys)``. A TE/TM leg carries only its co-polarised pair."""
    if scene['kind'] == 'tetm':
        return [('te', ('r_s', 't_s')), ('tm', ('r_p', 't_p'))]
    return [('s', AMPLITUDES), ('p', AMPLITUDES)]


def margin(scene, m, l, n):
    """``n**2 - |k_par|**2`` of order ``(m, l)``, as ``_meep_crossed_cases.margin`` spells it."""
    call = scene['call']
    theta = float(call.get('theta') or 0.)
    phi = float(call.get('phi') or 0.)
    period = call['period']
    wl = call['wavelength']
    kx = scene['n_top']*np.sin(theta)*np.cos(phi) + m*wl/period[0]
    ky = scene['n_top']*np.sin(theta)*np.sin(phi) + (l*wl/period[1] if len(period) > 1 else 0.)
    return float(n**2 - kx**2 - ky**2)


def _labels(result):
    u = [int(m) for m in np.asarray(result.orders_u).ravel()]
    v = [int(l) for l in np.asarray(getattr(result, 'orders_v', [0])).ravel()]
    return u, v


def _grid(sub, result, key):
    u, v = _labels(result)
    return np.asarray(getattr(sub, key)).reshape(len(v), len(u))


def read(subs, result, scene, legs=None):
    """``'leg.key' -> complex (n_v, n_u)`` on the scene's own order labels."""
    u, v = _labels(result)
    missing = [m for m in scene['orders_u'] if m not in u] + [l for l in scene['orders_v'] if l not in v]
    if missing:
        raise ValueError(f'orders {missing} absent from the solved window {u} x {v}')
    ix = np.ix_([v.index(l) for l in scene['orders_v']], [u.index(m) for m in scene['orders_u']])
    return {f'{leg}.{key}': _grid(subs[leg], result, key)[ix].astype(complex)
            for leg, keys in legs_of(scene) if legs is None or leg in legs
            for key in keys + POWERS}


def energy(subs, result, scene, legs=None):
    """Per leg: ``1 - sum R - sum T`` over every propagating order in the solved window."""
    u, v = _labels(result)
    out = {}
    for leg, _ in legs_of(scene):
        if legs is not None and leg not in legs:
            continue
        total = 0.
        for key, n in (('R', scene['n_top']), ('T', scene['n_bot'])):
            values = _grid(subs[leg], result, key)
            for j, l in enumerate(v):
                for i, m in enumerate(u):
                    if margin(scene, m, l, n) > 0:
                        total += float(np.real(values[j, i]))
        out[leg] = 1. - total
    return out


def window_is_complete(result, scene, reach=40):
    """True when every propagating order lies inside the solved window (energy is complete)."""
    u, v = _labels(result)
    crossed = len(scene['call']['period']) > 1
    for n in (scene['n_top'], scene['n_bot']):
        for m in range(-reach, reach + 1):
            for l in (range(-reach, reach + 1) if crossed else (0,)):
                if margin(scene, m, l, n) > 0 and (m not in u or l not in v):
                    return False
    return True


def mask_of(scene, label):
    key = label.split('.')[1]
    n = scene['n_top'] if key[0] in 'rR' else scene['n_bot']
    return np.array([[margin(scene, m, l, n) > MARGIN for m in scene['orders_u']]
                     for l in scene['orders_v']])


def max_change(a, b, scene, power):
    worst = 0.
    for label in a:
        if (label.split('.')[1] in POWERS) != power:
            continue
        mask = mask_of(scene, label)
        worst = max(worst, float(np.max(np.abs(a[label][mask] - b[label][mask]), initial=0.)))
    return worst


# =======================================================================================
# Reference
# =======================================================================================

def rcwa_channels(scene, order):
    """``(channels, energy per leg, window complete)`` from one RCWA truncation order."""
    kw = {k: v for k, v in scene['call'].items() if k not in ('dispersion', 'poles', 'truncation_order')}
    kw['structure'] = scene['rcwa_structure']
    crossed = np.shape(kw['structure'])[1] > 1
    kw['truncation_order'] = dict(x=order, y=order if crossed else 0)
    if scene['kind'] == 'tetm':
        # Spelled with tetm, like the legacy call: psi would put kx<0 orders in their own frame.
        channels, energies, complete = {}, {}, True
        for leg in scene['legs']:
            result = legacy_rcwa(backend='numpy', tetm=leg, **kw).solve()
            channels.update(read({leg: result.incident}, result, scene, legs=(leg,)))
            energies.update(energy({leg: result.incident}, result, scene, legs=(leg,)))
            complete = complete and window_is_complete(result, scene)
        return channels, energies, complete
    result = legacy_rcwa(backend='numpy', **kw).solve()
    subs = {'s': result.s_inc, 'p': result.p_inc}
    return read(subs, result, scene), energy(subs, result, scene), window_is_complete(result, scene)


def reference_uncertainty(steps, orders):
    """``u`` from the last order step, as the reference audit fixed it.

    ``u = 2 * step``. If the step before shrank by less than 2x the tail is slow, so the
    ``p ~ 1`` bound ``step * N / (N - N_prev)`` is used instead and the scene is flagged.
    """
    step = steps[-1]
    slow = len(steps) > 1 and steps[-1] > steps[-2]/2
    if slow:
        return step*orders[-1]/(orders[-1] - orders[-2]), True
    return 2*step, False


def solve_reference(scene, cache, timing):
    key = scene.get('reference_key')
    sols = {}
    for order in scene['reference_orders']:
        if key is not None and (key, order) in cache:
            sols[order] = cache[key, order]
            continue
        start, cpu = time.perf_counter(), time.process_time()
        sols[order] = rcwa_channels(scene, order)
        timing[f'rcwa_{order}'] = dict(wall_s=time.perf_counter() - start,
                                       cpu_s=time.process_time() - cpu)
        if key is not None:
            cache[key, order] = sols[order]
    orders = list(scene['reference_orders'])
    best = sols[orders[-1]][0]
    entry = dict(solver="studies._fdtd_legacy.legacy_rcwa(backend='numpy'), default factorization", orders=orders,
                 window_complete=bool(sols[orders[-1]][2]))
    u = {}
    for power in (False, True):
        kind = 'power' if power else 'amplitude'
        steps = [max_change(sols[a][0], sols[b][0], scene, power) for a, b in zip(orders, orders[1:])]
        u[kind], slow = reference_uncertainty(steps, orders)
        entry[f'{kind}_steps'] = [dict(orders=[a, b], change=s)
                                  for (a, b), s in zip(zip(orders, orders[1:]), steps)]
        entry[f'{kind}_slow'] = slow
    if scene['closed_form'] is not None:
        j, i = scene['orders_v'].index(0), scene['orders_u'].index(0)
        gap = max(abs(best[label][j, i] - value) for label, value in scene['closed_form'].items())
        # A uniform stack radiates no non-specular order; any the window holds must be zero.
        for label, values in best.items():
            if label.split('.')[1] in AMPLITUDES:
                off = mask_of(scene, label).copy()
                off[j, i] = False
                gap = max(gap, float(np.max(np.abs(values[off]), initial=0.)))
        entry['closed_form_vs_rcwa'] = float(gap)
        entry['closed_form'] = {k: [v.real, v.imag] for k, v in scene['closed_form'].items()}
        u['amplitude'] = max(u['amplitude'], float(gap))
    if scene['uniform']:
        # RCWA of a uniform stack is exact to round-off; an order step of exactly zero would
        # otherwise claim a perfect reference.
        u = {kind: max(value, ROUNDOFF) for kind, value in u.items()}
    entry['u'] = u
    # At the floor, 10 u is 1e-13 and any e1 != e2 clears it: a ranking there measures nothing.
    entry['u_is_roundoff_floor'] = {kind: bool(scene['uniform'] and value <= ROUNDOFF)
                                    for kind, value in u.items()}
    entry['energy'] = sols[orders[-1]][1]
    return best, entry


# =======================================================================================
# Routes
# =======================================================================================

def _caught(caught):
    return [f'{w.category.__name__}: {w.message}' for w in caught]


#: Legacy names a component by where it sits in the Yee cell, ``(z_half, x_half)``, at its
#: call sites in ``_time_loop``; the electric and magnetic tables are the two triples.
_LEGACY_ELECTRIC = {(False, False): 'ey', (False, True): 'ex', (True, False): 'ez'}
_LEGACY_MAGNETIC = {(True, False): 'hx', (False, True): 'hz', (True, True): 'hy'}
#: Common's native stacks are ordered (x, y, z). Electric component c sits half a cell along
#: its own axis, magnetic component c half a cell along the other two (``scene_fdtd.sample``),
#: so only Ez, Hx and Hy sit on the z half-steps.
_COMMON_ORDER = (('ex', 'ey', 'ez'), ('hx', 'hy', 'hz'))
_COMMON_Z_HALF = dict(ex=False, ey=False, ez=True, hx=True, hy=True, hz=False)


class MaterialCapture:
    """Record the material arrays each route's time loop is built from, passing every call on.

    Installed around the *measured* runs rather than a separate dry run. A dry run has to
    stop the first time loop, and a y-invariant psi scene builds its second triple only in
    the second loop, so it would see half the arrays; the pass-through sees every array every
    measured loop used, at no cost. Legacy is read at ``_Grid.eps_for``/``mu_for`` and at
    ``pole_state``, whose fills are taken as ``_PoleSet`` receives them, which is exactly
    where a fill computed at the wrong position would show. Common is read at the
    ``fdtd_grid`` call of its structure run (the one handed ``poles``; its uniform reference
    run is not). Values are ``key -> (z centres, array)``; a key built twice must come back
    bit-identical, or it is listed in ``inconsistent``. Each wrapper restates its target's
    signature, so a new keyword on the target breaks the study on its first scene;
    ``tests/test_fdtd_route_parity_capture.py`` fails first.
    """

    def __init__(self):
        self.arrays = {}
        self.inconsistent = []
        self.common_calls = []
        self._pending = None

    def _put(self, key, z, value):
        value = np.array(value)
        if key in self.arrays:
            if not np.array_equal(self.arrays[key][1], value) and key not in self.inconsistent:
                self.inconsistent.append(key)
            return
        self.arrays[key] = (np.array(z, dtype=float), value)

    def __enter__(self):
        grid_class, pole_class = LEGACY._Grid, LEGACY._PoleSet
        self._saved = [(grid_class, 'eps_for', grid_class.eps_for),
                       (grid_class, 'mu_for', grid_class.mu_for),
                       (grid_class, 'pole_state', grid_class.pole_state),
                       (pole_class, '__init__', pole_class.__init__),
                       (COMMON, 'fdtd_grid', COMMON.fdtd_grid)]
        eps_for, mu_for, pole_state, pole_init, fdtd_grid = (item[2] for item in self._saved)
        capture = self

        def eps_wrapper(grid, z_half, x_half, harmonic_axis, y_half=False):
            value = eps_for(grid, z_half, x_half, harmonic_axis, y_half=y_half)
            name = _LEGACY_ELECTRIC.get((z_half, x_half), f'unmapped{(z_half, x_half)}')
            capture._put(('eps', name), grid.z_cells(z_half)[0], value)
            return value

        def mu_wrapper(grid, z_half, x_half, harmonic_axis, y_half=False):
            value = mu_for(grid, z_half, x_half, harmonic_axis, y_half=y_half)
            name = _LEGACY_MAGNETIC.get((z_half, x_half), f'unmapped{(z_half, x_half)}')
            capture._put(('mu', name), grid.z_cells(z_half)[0], value)
            return value

        def pole_init_wrapper(state, shape, dt, poles, fills, dtype):
            if capture._pending is not None:
                capture._pending.append((list(poles), [np.array(f) for f in fills]))
            return pole_init(state, shape, dt, poles, fills, dtype)

        def pole_state_wrapper(grid, shape, z_half, x_half, dtype, y_half=False):
            capture._pending = []
            try:
                state = pole_state(grid, shape, z_half, x_half, dtype, y_half=y_half)
            finally:
                pending, capture._pending = capture._pending, None
            name = _LEGACY_ELECTRIC.get((z_half, x_half), f'unmapped{(z_half, x_half)}')
            for poles, fills in pending:
                seen = {}
                for (omega0, gamma, strength), fill in zip(poles, fills):
                    term = (float(omega0), float(gamma))
                    seen[term] = seen.get(term, -1) + 1
                    capture._put(('pole', name, term + (seen[term],)), grid.z_cells(z_half)[0],
                                 strength*np.broadcast_to(fill, shape))
            return state

        def grid_wrapper(epsilon, **kw):
            if 'poles' in kw:
                capture.common_calls.append(dict(
                    epsilon=np.array(epsilon), mu=np.array(kw['mu']),
                    poles=[(float(w), float(g), np.array(s)) for w, g, s in kw['poles']],
                    spacing=tuple(float(v) for v in kw['spacing']),
                    monitor_z=[int(v) for v in kw['monitor_z']],
                    uniform=kw.get('z_spacing') is None))
            return fdtd_grid(epsilon, **kw)

        grid_class.eps_for, grid_class.mu_for = eps_wrapper, mu_wrapper
        grid_class.pole_state, pole_class.__init__ = pole_state_wrapper, pole_init_wrapper
        COMMON.fdtd_grid = grid_wrapper
        return self

    def __exit__(self, *exc):
        for owner, name, original in self._saved:
            setattr(owner, name, original)
        return False


def _material_name(key):
    kind, component = key[0], key[1]
    if kind == 'pole':
        omega0, gamma, index = key[2]
        return f'pole_fill.{component}[omega0={omega0:.6g}, gamma={gamma:.6g}, #{index}]'
    return f'{kind}.{component}'


def _compare_array(z_legacy, legacy, z_common, common, dz):
    """Node-for-node comparison over the z positions both grids hold."""
    index = np.rint((z_legacy - z_common[0])/dz).astype(int)
    shared = (index >= 0) & (index < len(z_common))
    shared[shared] &= np.abs(z_common[index[shared]] - z_legacy[shared]) < 1e-6*dz
    if not shared.any():
        return dict(comparable=False, reason='no shared z position')
    a, b = legacy[shared], common[index[shared]]
    if a.shape[1] == 1 and b.shape[1] > 1:
        # Legacy's 2D steppers carry one y row; common's y-invariant mesh has two identical
        # ones, and each is held against legacy's row.
        a = np.broadcast_to(a, b.shape)
    if a.shape != b.shape:
        return dict(comparable=False, reason=f'shapes {a.shape} vs {b.shape} over shared z')
    difference = np.abs(a - b)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    worst = np.unravel_index(int(np.argmax(difference)), difference.shape)
    relative = float(difference[worst])/scale if scale > 0 else 0.
    z_shared = z_legacy[shared]
    return dict(comparable=True, max_difference=float(difference[worst]), relative=relative,
                agrees=relative <= MATERIAL_TOL, worst_z=float(z_shared[worst[0]]),
                worst_yx=[int(worst[1]), int(worst[2])],
                shared_z=[float(z_shared.min()), float(z_shared.max())],
                shared_rows=int(shared.sum()))


def compare_materials(scene, legacy_capture, common_capture, common_diag):
    """Every array one engine steps with, against the other's, on legacy's z positions.

    ``matched`` needs every array both routes build to agree to ``MATERIAL_TOL``, the shared
    z range to cover the layer stack, a pole term on one side to exist on the other, and each
    route to have built each of its arrays consistently. Components legacy never steps (a
    pure TE or TM leg allocates one triple) are listed and do not count against it.
    """
    out = dict(tolerance_relative=MATERIAL_TOL, inconsistent=[], differing=[], missing=[],
               not_stepped_by_legacy=[], compared={})
    calls = common_capture.common_calls
    if not calls:
        out.update(matched=False, reason="common's structure run was not captured")
        return out
    first = calls[0]
    for other in calls[1:]:
        for field in ('epsilon', 'mu'):
            if not np.array_equal(other[field], first[field]):
                out['inconsistent'].append(f'common.{field}')
    if not first['uniform']:
        out.update(matched=False, reason='a graded common mesh is not compared node for node')
        return out
    dz = first['spacing'][0]
    # Common's z nodes are uniform; its reflection plane is node monitor_z[0] and sits at the
    # z the diagnostics report, which fixes the axis. The transmission plane checks it.
    reflect, transmit = first['monitor_z'][:2]
    nz = first['epsilon'].shape[1]
    nodes = float(common_diag['diffraction']['reflection_monitor_z']) + (np.arange(nz) - reflect)*dz
    axis_error = abs(nodes[transmit] - float(common_diag['diffraction']['transmission_monitor_z']))
    if axis_error > 1e-6*dz:
        out.update(matched=False, reason=f'common z axis does not reproduce its planes ({axis_error:g})')
        return out
    common = {}
    for kind, stack, order in (('eps', first['epsilon'], _COMMON_ORDER[0]),
                               ('mu', first['mu'], _COMMON_ORDER[1])):
        for c, name in enumerate(order):
            common[kind, name] = stack[c]
    seen = {}
    for omega0, gamma, strength in first['poles']:
        term = (omega0, gamma)
        seen[term] = seen.get(term, -1) + 1
        strength = np.broadcast_to(strength, first['epsilon'].shape)
        for c, name in enumerate(_COMMON_ORDER[0]):
            common['pole', name, term + (seen[term],)] = strength[c]
    legacy = legacy_capture.arrays
    out['inconsistent'] += [_material_name(k) for k in legacy_capture.inconsistent]
    stepped = {key[1] for key in legacy if key[0] in ('eps', 'mu')}
    thickness = float(sum(scene['call']['thickness']))
    covers = True
    for key in sorted(set(legacy) | set(common), key=_material_name):
        name = _material_name(key)
        if key not in legacy:
            if key[1] not in stepped:
                out['not_stepped_by_legacy'].append(name)
            elif key[0] == 'pole' and not np.any(common[key]):
                continue
            else:
                out['missing'].append(f'{name} (common only)')
            continue
        if key not in common:
            if key[0] == 'pole' and not np.any(legacy[key][1]):
                continue
            out['missing'].append(f'{name} (legacy only)')
            continue
        z_legacy, values = legacy[key]
        z_common = nodes + (dz/2 if _COMMON_Z_HALF.get(key[1], False) else 0.)
        stats = _compare_array(z_legacy, values, z_common, common[key], dz)
        if key[0] == 'pole' and stats.get('comparable'):
            # The product is compared; the fill itself reads more directly.
            strength = next((s for w, g, s in _scene_poles(scene) if (w, g) == key[2][:2]), 0.)
            if strength:
                stats['max_fill_difference'] = stats['max_difference']/abs(strength)
        out['compared'][name] = stats
        if not stats.get('comparable') or not stats['agrees']:
            out['differing'].append(name)
        if stats.get('comparable'):
            covers = covers and stats['shared_z'][0] < 0. and stats['shared_z'][1] > thickness
    out['differing'].sort(key=lambda n: -out['compared'][n].get('relative', np.inf))
    out['attributed'] = {name: dict(KNOWN_INPUT_DEFECTS[prefix], prefix=prefix)
                         for name in out['differing'] for prefix in KNOWN_INPUT_DEFECTS
                         if name.startswith(prefix)}
    out['shared_z_covers_stack'] = covers
    out['matched'] = bool(covers and not out['differing'] and not out['missing']
                          and not out['inconsistent'])
    return out


def _scene_poles(scene):
    return [(float(w), float(g), float(s)) for terms in scene['call'].get('poles', {}).values()
            for w, g, s in terms]


def _one_target(runs):
    used = {float(diag['pml_reflection']) for _, _, diag, _ in runs}
    if len(used) != 1:
        raise AssertionError(f'legacy built its CPML with targets {sorted(used)}, not one')
    return used.pop()


def run_legacy(scene, timing):
    """The legacy route as shipped: the study's keywords, backend omitted."""
    runs = []
    for leg in (scene['legs'] if scene['kind'] == 'tetm' else (None,)):
        kw = dict(scene['call'], **scene['study'], return_diagnostics=True)
        if leg is not None:
            kw['tetm'] = leg
        start, cpu = time.perf_counter(), time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result, diag = legacy_fdtd(**kw)
        timing['legacy' + (f'_{leg}' if leg else '')] = dict(wall_s=time.perf_counter() - start,
                                                             cpu_s=time.process_time() - cpu)
        if 'grid' not in diag:
            raise AssertionError('backend omitted did not reach the legacy route')
        runs.append((leg, result, diag, _caught(caught)))
    first = runs[0][2]['grid']
    for _, _, diag, _ in runs[1:]:
        g = diag['grid']
        if (g.nx, g.nz, g.dx, g.dt, g.k_ref, g.k_trn) != (first.nx, first.nz, first.dx,
                                                         first.dt, first.k_ref, first.k_trn):
            raise AssertionError('legacy TE and TM legs built different grids')
    if scene['kind'] == 'tetm':
        channels, energies, complete = {}, {}, True
        for leg, result, _, _ in runs:
            channels.update(read({leg: result.incident}, result, scene, legs=(leg,)))
            energies.update(energy({leg: result.incident}, result, scene, legs=(leg,)))
            complete = complete and window_is_complete(result, scene)
        return channels, energies, complete, runs
    result = runs[0][1]
    subs = {'s': result.s_inc, 'p': result.p_inc}
    return read(subs, result, scene), energy(subs, result, scene), window_is_complete(result, scene), runs


def legacy_record(scene, runs):
    g = runs[0][2]['grid']
    thickness = float(sum(scene['call']['thickness']))
    k_bottom = g.k_z0 + int(np.ceil(thickness/g.dz))
    wl = scene['call']['wavelength']
    record = dict(
        time_loops_per_call=2 if scene['kind'] == 'psi' else 1, calls=len(runs),
        nx=g.nx, ny=g.ny, nz=g.nz, dx=g.dx, dy=g.dy, dz=g.dz, dt=float(g.dt),
        n_pml=g.n_pml, resolution_effective=1/g.dx,
        courant_keyword=scene['study'].get('courant', LEGACY.COURANT),
        n_periods_keyword=scene['study'].get('n_periods', LEGACY.N_PERIODS),
        pulse_width=LEGACY.SOURCE_CYCLES*wl, peak_time=LEGACY.SOURCE_DELAY*LEGACY.SOURCE_CYCLES*wl,
        pml_target_reflection=_one_target(runs),
        planes_cells=dict(src=g.k_src, ref=g.k_ref, tfsf=g.k_tfsf, z0=g.k_z0, bottom=k_bottom,
                          trn=g.k_trn),
        reflection_monitor_z=float(g.z_of(g.k_ref)), transmission_monitor_z=float(g.z_of(g.k_trn)),
        pml_gap_cells=dict(top=g.k_z0 - g.n_pml, bottom=(g.nz - 1 - g.n_pml) - k_bottom),
        legs=[])
    for leg, _, diag, caught in runs:
        record['legs'].append(dict(leg=leg or 'psi (s and p columns)', n_steps=int(diag['n_steps']),
                                   residual=float(diag['residual']),
                                   incident_balance=float(np.max(np.abs(diag['incident_balance']))),
                                   pml_thickness=[float(v) for v in diag['pml_thickness']],
                                   warnings=caught))
    record['cells'] = g.nz*g.ny*g.nx
    record['time_loops'] = record['time_loops_per_call']*record['calls']
    record['cell_steps'] = sum(record['time_loops_per_call']*record['cells']*leg['n_steps']
                               for leg in record['legs'])
    return record


def pole_cfl_bound(scene, spacing):
    """A lower bound on common's pole-aware time-step limit (``grid_fdtd.fdtd_grid``).

    Common tightens its Maxwell bound ``sqrt(min eps * min mu)/|1/spacing|`` to
    ``1/sqrt(1/bound**2 + rate/4)`` with ``rate = max omega0**2 + max(sum strength/eps)``.
    Its strengths are fill-weighted and its eps is the sampled one, neither of which exists
    before the run, so this takes every pole at full strength over the smallest eps anywhere:
    never above common's own bound, so a ``dt`` below it is accepted.
    """
    call = scene['call']
    eps_lo = min(float(np.real(call['top'])), float(np.real(call['bot'])),
                 float(np.min(np.real(call['structure']))))
    mu_lo = min(1., float(np.min(call['mu']))) if 'mu' in call else 1.
    bound = math.sqrt(eps_lo*mu_lo)/math.sqrt(sum(s**-2 for s in spacing))
    terms = [pole for poles in call.get('poles', {}).values() for pole in poles]
    if not terms:
        return bound
    rate = max(p[0]**2 for p in terms) + sum(p[2] for p in terms)/eps_lo
    return 1/math.sqrt(1/bound**2 + rate/4)


def matched_common(scene, runs, pad_mode, pole_cfl=False):
    """Common-route keywords that rebuild legacy's grid and schedule. Returns ``(kw, notes)``.

    ``pole_cfl=True`` is the fallback after common refused legacy's ``dt`` under its
    pole-aware CFL bound: ``dt`` is then the legacy ``courant`` keyword times
    :func:`pole_cfl_bound`, everything else stays matched, and the ``dt`` ratio is a confound.
    """
    g = runs[0][2]['grid']
    call, study = scene['call'], scene['study']
    wl = call['wavelength']
    period = list(call['period'])
    thickness = float(sum(call['thickness']))
    if study.get('grading') is not None:
        raise ValueError('graded meshes cannot be matched across the routes')
    crossed = np.shape(call['structure'])[1] > 1
    theta = float(call.get('theta') or 0.)
    phi = float(call.get('phi') or 0.)
    y_invariant = not crossed and (theta == 0. or phi == 0.)
    dz = g.dz
    if crossed:
        # (n - 1/2)/L, not n/L: ceil(L * (n/L)) lands on n + 1 for some (n, L) in floating point.
        transverse, dy = ((g.nx - .5)/period[0], (g.ny - .5)/period[1]), g.dy
    else:
        # Common meshes y with two cells of 1/ry. On a y-invariant scene that cell is inert, so
        # it is made wide enough to leave dt to x and z; on a conical one it is not, so dy = dx.
        ry = (1/Y_INVARIANT_CELL if y_invariant else 1.)/g.dx
        transverse, dy = ((g.nx - .5)/period[0], ry), 1/ry
    if pole_cfl:
        courant = float(study.get('courant', LEGACY.COURANT)*pole_cfl_bound(scene, (dz, dy, g.dx))
                        * math.sqrt(dz**-2 + dy**-2 + g.dx**-2))
        dt_reason = 'pole_cfl'
    elif crossed or y_invariant:
        courant = float(g.dt*math.sqrt(dz**-2 + dy**-2 + g.dx**-2))
        dt_reason = None
    else:
        courant = float(study.get('courant', LEGACY.COURANT))
        dt_reason = 'conical'
    if not 0 < courant < 1:
        raise ValueError(f'matching legacy dt needs courant {courant}, which common refuses')
    dt = courant/math.sqrt(dz**-2 + dy**-2 + g.dx**-2)
    n_periods = (study.get('n_periods', LEGACY.N_PERIODS)
                 + LEGACY.SOURCE_CYCLES*(COMMON_PEAK_DELAY - LEGACY.SOURCE_DELAY))
    k_bottom = g.k_z0 + int(np.ceil(thickness/dz))
    n_min = min(scene['n_top'], scene['n_bot'])
    if pad_mode == 'legacy':
        # Common's gap between PML and structure is 2*pad + 2 cells on both sides, legacy's is
        # asymmetric; reach legacy's upper gap, and leave room for common's own source (three
        # cells above the reflection plane) and for the transmission plane.
        pad = max(6, math.ceil((g.k_z0 - g.n_pml - 2)/2), (g.k_z0 - g.k_ref + 2)//2 + 1,
                  (g.k_trn - k_bottom)//2 + 1)
        pad_scale = (pad - .5)*dz*n_min/wl
    else:
        pad_scale = study.get('pad_scale', 1.)
        pad = max(6, math.ceil(pad_scale*wl/n_min/dz))
    steps = math.ceil(n_periods*wl/dt)
    kw = dict(call, backend='numpy', resolution=1/dz, transverse_resolution=transverse,
              courant=courant, n_periods=n_periods, pulse_width=LEGACY.SOURCE_CYCLES*wl,
              n_pml=g.n_pml, pml_reflection=g.pml_reflection,
              subpixel=study.get('subpixel', True), grading=None, pad_scale=pad_scale,
              diffraction_distance=(-float(g.z_of(g.k_ref)), float(g.z_of(g.k_trn)) - thickness),
              convergence_window=max(1, int(CONVERGENCE_FRACTION*steps)),
              return_diagnostics=True)
    if scene['kind'] == 'tetm':
        kw['tetm'] = scene['legs'][0]
    notes = dict(dt_matched=dt_reason is None, dt_reason=dt_reason, y_invariant=bool(y_invariant),
                 expected_pad=pad, expected_steps=steps)
    return kw, notes


def verbatim_common(scene):
    """Common driven by the study's own keywords plus ``backend=0``; courant as legacy's default."""
    kw = dict(scene['call'], **scene['study'], backend='numpy', return_diagnostics=True)
    kw.setdefault('courant', LEGACY.COURANT)
    if scene['kind'] == 'tetm':
        kw['tetm'] = scene['legs'][0]
    return kw


def run_common(scene, kw, timing, label):
    start, cpu = time.perf_counter(), time.process_time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result, diag = legacy_fdtd(**kw)
    timing[label] = dict(wall_s=time.perf_counter() - start, cpu_s=time.process_time() - cpu)
    if 'shape' not in diag:
        raise AssertionError("backend='numpy' did not reach the archived common engine")
    if scene['kind'] == 'tetm':
        subs = {'te': result.s_inc, 'tm': result.p_inc}   # never `given`: it mixes the other column
    else:
        subs = {'s': result.s_inc, 'p': result.p_inc}
    return (read(subs, result, scene), energy(subs, result, scene),
            window_is_complete(result, scene), result, diag, _caught(caught))


def _max_abs(tree):
    values = [np.max(np.abs(np.asarray(v))) for v in tree.values()]
    return float(np.nanmax(values)) if values else None


def common_record(scene, kw, diag, caught):
    nz, ny, nx = (int(v) for v in diag['shape'])
    thickness = float(sum(scene['call']['thickness']))
    dz, dy, dx = (float(v) for v in diag['spacing'])
    bounds = [float(v) for v in diag['pml_bounds']]
    pulse = diag['pulse']
    record = dict(
        keywords={k: _plain(v) for k, v in kw.items()
                  if k not in ('structure', 'mu', 'dispersion', 'return_diagnostics')},
        time_loops=4, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=float(diag['dt']),
        n_steps=int(diag['steps']), n_pml=int(diag['n_pml']),
        pml_reflection=float(diag['pml_reflection']), stable=bool(np.all(np.asarray(diag['stable']))),
        pulse_width=float(pulse['width']), peak_time=float(pulse['peak_time']),
        source_z=float(diag['source_z']),
        reflection_monitor_z=float(diag['diffraction']['reflection_monitor_z']),
        transmission_monitor_z=float(diag['diffraction']['transmission_monitor_z']),
        mesh_indices=[float(v) for v in diag['mesh_indices']],
        pml_gap_cells=dict(top=round(-bounds[1]/dz), bottom=round((bounds[2] - thickness)/dz)),
        warnings=caught)
    record['cells'] = nz*ny*nx
    record['cell_steps'] = record['time_loops']*record['cells']*record['n_steps']
    if 'convergence' in diag:
        c = diag['convergence']
        record['convergence'] = dict(
            window=int(kw['convergence_window']),
            scattering_change_bound=_max_abs(c['scattering_change_bound']),
            scattering_settled=bool(np.all(np.asarray(c['scattering_settled']))),
            field_decayed=bool(np.all(np.asarray(c['field_decayed']))),
            material_decayed=bool(np.all(np.asarray(c['material_decayed']))))
    return record


def match_checks(legacy, common, notes, materials):
    """Each matched quantity, as the two runs report it. ``matched`` needs all of them,
    including the material arrays (:func:`compare_materials`)."""
    dz = legacy['dz']
    post = ((common['n_steps']*common['dt'] - common['peak_time'])
            - (legacy['legs'][0]['n_steps']*legacy['dt'] - legacy['peak_time']))
    checks = dict(
        nx=[legacy['nx'], common['nx']], ny=[legacy['ny'], common['ny']],
        dx_ratio=common['dx']/legacy['dx'], dz_ratio=common['dz']/dz,
        dt_ratio=common['dt']/legacy['dt'],
        reflection_plane_offset_cells=(common['reflection_monitor_z'] - legacy['reflection_monitor_z'])/dz,
        transmission_plane_offset_cells=(common['transmission_monitor_z']
                                         - legacy['transmission_monitor_z'])/dz,
        n_pml=[legacy['n_pml'], common['n_pml']],
        pml_reflection=[legacy['pml_target_reflection'], common['pml_reflection']],
        pulse_width_ratio=common['pulse_width']/legacy['pulse_width'],
        peak_delay_widths=[legacy['peak_time']/legacy['pulse_width'],
                           common['peak_time']/common['pulse_width']],
        post_peak_time_difference_dt=post/legacy['dt'],
        pad_cells=[notes['expected_pad'], (common['pml_gap_cells']['top'] - 2)//2])
    ok = dict(
        grid=(legacy['nx'] == common['nx'] and (legacy['ny'] == common['ny'] or legacy['ny'] == 1)
              and abs(checks['dx_ratio'] - 1) < 1e-12 and abs(checks['dz_ratio'] - 1) < 1e-12),
        dt=abs(checks['dt_ratio'] - 1) < 1e-12,
        planes=(abs(checks['reflection_plane_offset_cells']) < 1e-6
                and abs(checks['transmission_plane_offset_cells']) < 1e-6),
        pml=legacy['n_pml'] == common['n_pml'] and legacy['pml_target_reflection'] == common['pml_reflection'],
        envelope=(abs(checks['pulse_width_ratio'] - 1) < 1e-12
                  and abs(checks['peak_delay_widths'][1] - COMMON_PEAK_DELAY) < 1e-12),
        post_peak_time=abs(checks['post_peak_time_difference_dt']) <= 1.,
        pad=checks['pad_cells'][0] == checks['pad_cells'][1],
        materials=bool(materials['matched']))
    checks['materials_differing'] = materials['differing'] + materials['missing']
    checks['passed'] = ok
    # dt is expected to miss on a 1D conical mount or after common's pole-CFL refusal, and
    # only there (``dt_reason``); everything else must hold.
    checks['dt_exempt'] = notes['dt_reason']
    checks['matched'] = all(v for k, v in ok.items() if k != 'dt' or notes['dt_matched'])
    return checks


def confounds(scene, legacy, common, checks, notes, rows):
    """Differences the matching cannot remove; each is inside every legacy-vs-common number."""
    grazing = []
    for r in rows:
        side = 'reflection' if r['channel'].split('.')[1][0] in 'rR' else 'transmission'
        item = dict(order=r['order'], side=side, exit_cos=r['exit_cos'])
        if r['grazing'] and item not in grazing:
            grazing.append(item)
    # annotate() names termination_control here once the scene's twin is in the same report.
    termination = dict(grazing_channels=grazing, measured_in=None)
    crossed = np.shape(scene['call']['structure'])[1] > 1
    theta = float(scene['call'].get('theta') or 0.)
    phi = float(scene['call'].get('phi') or 0.)
    k_y = abs(2*np.pi/scene['call']['wavelength']*scene['n_top']*np.sin(theta)*np.sin(phi))
    # Every scene here has nonmagnetic half-spaces, so the eps-vs-eps*mu PML grading split
    # (legacy grades on eps alone) never applies; it is named so its absence is visible.
    return [
        dict(id='source_normalisation', applies=True,
             detail='legacy: TFSF plane fed by a 1D incident grid, reflection read in the '
                    'scattered-field region above the TFSF plane; common: two soft current '
                    'sheets, each with a uniform-medium reference run, reflected = actual - '
                    'reference, calibrated by a 2x2 incident matrix; its source sits above its '
                    f"reflection plane. Time loops: legacy {legacy['time_loops']}, "
                    f"common {common['time_loops']}."),
        dict(id='deembedding_phase', applies=True, measured_in='deembedding_control',
             detail='legacy de-embeds with the continuum kz, common with the discrete Yee kz; '
                    'moves amplitude phases at O(h^2 * distance), leaves |r|^2 untouched. '
                    'Removed in the legacy_yee_deembedded arm, which is the engine comparison.'),
        dict(id='transverse_demodulation', applies=bool(crossed or theta != 0.),
             detail='legacy demodulates each staggered component at its own x (Hy, Ex at '
                    'x+dx/2); common interpolates faces to the node and divides by cos(kx dx/2).'),
        dict(id='tm_half_cell_sampling', applies=bool(not crossed and (theta == 0. or phi == 0.)),
             detail='on its 2D TM stepper legacy samples Hy half a cell off the plane common '
                    'samples E on.'),
        dict(id='pml_backing', applies=True, **termination,
             detail='legacy grounds E on the two end rows; common wraps z periodically with a '
                    'closing cell (its lower PML face one cell further). PML grading on '
                    'magnetic half-spaces (eps on legacy, eps*mu on common): not exercised.'),
        dict(id='pml_structure_gap', applies=True, **termination,
             legacy_cells=legacy['pml_gap_cells'], common_cells=common['pml_gap_cells'],
             detail='legacy clears 0.5*pad_scale*lambda/n per side (asymmetric); common lays one '
                    'pad on both sides (2*pad + 2 cells).'),
        dict(id='pulse_delay_and_carrier', applies=True,
             peak_delay_widths=checks['peak_delay_widths'],
             post_peak_time_difference_dt=checks['post_peak_time_difference_dt'],
             detail='peak at 4 widths on legacy and 5 on common (hard-coded both); the post-peak '
                    'time is equalised, the total step count is not. Legacy drives a real sine, '
                    'common a complex exp(-i w t).'),
        dict(id='subpixel_order', applies=not scene['uniform'],
             detail='both state the same rule (harmonic along the component, arithmetic across) '
                    'but average in a different order; they agree on a slab, differ unmeasured '
                    'at tooth corners.'),
        dict(id='pole_cfl_dt', applies=notes['dt_reason'] == 'pole_cfl',
             dt_ratio=checks['dt_ratio'],
             detail="common refused legacy's dt under its conservative coupled Maxwell/oscillator "
                    'CFL bound (grid_fdtd.fdtd_grid), which legacy does not apply; common ran at '
                    "the legacy courant keyword times a lower bound on that limit "
                    '(pole_cfl_bound). Every other matched quantity still holds.'),
        dict(id='conical_dt_and_dy', applies=notes['dt_reason'] == 'conical',
             dt_ratio=checks['dt_ratio'], ky_dy=None if crossed else k_y*common['dy'],
             detail='a 1D conical mount: legacy uses the exact continuum d/dy = i ky, common a '
                    'two-cell Bloch difference with dy = dx, and courant < 1 cannot reach '
                    "legacy's dt."),
    ]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(np.real(value)), float(np.imag(value))]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


# =======================================================================================
# Comparison
# =======================================================================================

def yee_deembedded(scene, legacy, values):
    """Legacy's amplitudes re-de-embedded with common's discrete Yee ``kz``; powers unchanged.

    Legacy carried each monitor back to the interface with ``exp(-i kz d)`` on the continuum
    ``kz``; common uses ``2/h asin(h/2 sqrt(n**2 w**2 - kx**2 - ky**2))`` with the step
    frequency ``w = 2 sin(k0 dt/2)/dt`` and discrete transverse wavenumbers
    (``scene_fdtd.propagation``, ``conventions.yee_longitudinal_phase``). Swapping one for the
    other multiplies each amplitude by ``exp(-i (kz_yee - kz_continuum) d)``: ``d = |z_ref|``
    for reflection and ``z_trn - thickness`` for transmission, both on legacy's node planes
    (a TM loop's monitors sit half a cell below them; the module docstring names that
    residual), on legacy's own ``dt`` and cells. The transverse terms are discrete on a crossed
    scene; on a 1D one ``ky`` stays the continuum value, because legacy's own ``d/dy`` is the
    exact ``i ky`` there. No parameter is free. This is the reviewer's ``a4.py`` computation,
    operation for operation, so the harness reproduces its numbers.
    """
    call = scene['call']
    wl = call['wavelength']
    k0 = 2*np.pi/wl
    # The index from the half-space permittivity, as a4.py spells it; every half-space here
    # is non-magnetic, so it is sqrt(eps*mu) as well, and the scene's own index must agree.
    n_top = np.sqrt(complex(call['top']).real)
    n_bot = np.sqrt(complex(call['bot']).real)
    if abs(n_top - scene['n_top']) > 1e-9 or abs(n_bot - scene['n_bot']) > 1e-9:
        raise AssertionError('half-space index is not sqrt(eps); a magnetic half-space needs mu')
    theta = float(call.get('theta') or 0.)
    phi = float(call.get('phi') or 0.)
    period = call['period']
    crossed = len(period) > 1
    h, dx, dy, dt = legacy['dz'], legacy['dx'], legacy['dy'], legacy['dt']
    omega2 = (2*np.sin(k0*dt/2)/dt)**2
    thickness = float(sum(call['thickness']))
    distance = dict(r=abs(legacy['reflection_monitor_z']),
                    t=legacy['transmission_monitor_z'] - thickness)
    kx0 = k0*n_top*np.sin(theta)*np.cos(phi)
    ky0 = k0*n_top*np.sin(theta)*np.sin(phi)
    out = {}
    for label, grid in values.items():
        key = label.split('.')[1]
        if key not in AMPLITUDES:
            out[label] = grid
            continue
        n = n_top if key[0] == 'r' else n_bot
        mask = mask_of(scene, label)
        corrected = np.array(grid, dtype=complex)
        for j, l in enumerate(scene['orders_v']):
            for i, m in enumerate(scene['orders_u']):
                if not mask[j, i]:
                    continue   # never compared; an evanescent kz would only grow here
                kx = kx0 + 2*np.pi*m/period[0]
                ky = ky0 + (2*np.pi*l/period[1] if crossed else 0.)
                kx_yee = (2*np.sin(kx*dx/2)/dx)**2
                ky_yee = (2*np.sin(ky*dy/2)/dy)**2 if crossed else ky**2
                kz_yee = 2/h*np.arcsin(h/2*np.sqrt(n**2*omega2 - kx_yee - ky_yee + 0j))
                kz_continuum = np.sqrt((n*k0)**2 - kx**2 - ky**2 + 0j)
                corrected[j, i] = corrected[j, i]*np.exp(-1j*(kz_yee - kz_continuum)*distance[key[0]])
        out[label] = corrected
    return out


def _exit_cos(scene, label, m, l):
    """cos(theta) of order ``(m, l)`` in the medium channel ``label`` leaves into: the top
    half-space for a reflection (``r_*``, ``R``), the bottom one for a transmission. From the
    continuum in-plane wavevector, as :func:`yee_deembedded` spells it."""
    call = scene['call']
    k0 = 2*np.pi/call['wavelength']
    theta, phi = float(call.get('theta') or 0.), float(call.get('phi') or 0.)
    period = call['period']
    kx = k0*scene['n_top']*np.sin(theta)*np.cos(phi) + 2*np.pi*m/period[0]
    ky = (k0*scene['n_top']*np.sin(theta)*np.sin(phi)
          + (2*np.pi*l/period[1] if len(period) > 1 else 0.))
    n = scene['n_top'] if label.split('.')[1][0] in 'rR' else scene['n_bot']
    return float(np.sqrt(max(0., 1 - (kx**2 + ky**2)/(n*k0)**2)))


def compare(scene, reference, arms, pairs):
    """Per comparable channel: every arm's value and error, and each pair's distance and ratio."""
    rows = []
    for label in reference:
        mask = mask_of(scene, label)
        for j, l in enumerate(scene['orders_v']):
            for i, m in enumerate(scene['orders_u']):
                if not mask[j, i]:
                    continue
                ref = complex(reference[label][j, i])
                exit_cos = _exit_cos(scene, label, m, l)
                row = dict(channel=label, order=[m, l], reference=[ref.real, ref.imag],
                           exit_cos=exit_cos, grazing=exit_cos < GRAZING_COS)
                for arm, values in arms.items():
                    value = complex(values[label][j, i])
                    finite = bool(np.isfinite(value))
                    row[arm] = [value.real, value.imag] if finite else None
                    row[f'{arm}_vs_reference'] = abs(value - ref) if finite else None
                for a, b in pairs:
                    va, vb = complex(arms[a][label][j, i]), complex(arms[b][label][j, i])
                    d = abs(va - vb) if np.isfinite(va) and np.isfinite(vb) else None
                    row[f'{a}_vs_{b}'] = d
                    errors = (row[f'{a}_vs_reference'], row[f'{b}_vs_reference'])
                    # This channel's own d over this channel's own smaller error.
                    row[f'{a}_vs_{b}_over_min_error'] = (
                        d/min(errors) if d is not None and None not in errors and min(errors) > 0
                        else None)
                rows.append(row)
    return rows


#: The ratio classes in order of how much they say the route matters.
_CASE_SCALE = ('same_error', 'intermediate', 'route_changes_answer')


def classify(e_first, e_second, d, u, settling, roundoff_floor, worst_first, worst_second,
             names, firm=None, channel_settling=None, confound_shift=0.):
    """Which case a pair falls in (see the module docstring).

    ``e_first``, ``e_second`` and ``d`` are all taken on ONE channel, the one carrying the
    largest ``d``; the ranking compares each route's worst error (``worst_*``), which are
    like quantities. In order: a ``d`` below the settling bound is unresolved and ranks
    nothing; a round-off-floor ``u`` withholds the ranking, and there a pair with one error
    more than ``RESOLVABLE`` times below the other is ``one_route_near_exact``; otherwise a
    ranking needs ``|worst_first - worst_second| >= RESOLVABLE u`` and above
    ``confound_shift``. ``firm`` is the part of the settling bound that is not a proxy
    (common's bound on amplitude, the channel-scaled estimate on power).
    """
    resolved = [e > 0 and e >= RESOLVABLE*u for e in (e_first, e_second)]
    smaller, larger = min(e_first, e_second), max(e_first, e_second)
    out = dict(first=names[0], second=names[1], first_resolved=resolved[0],
               second_resolved=resolved[1], d_over_min_error=d/smaller if smaller > 0 else None,
               d_over_max_error=d/larger if larger > 0 else None,
               error_gap=abs(e_first - e_second), ranking=None, more_accurate=None,
               ranking_withheld=None)
    if channel_settling is not None:
        out['channel_settling_estimate'] = channel_settling
        out['d_over_channel_settling_estimate'] = d/channel_settling if channel_settling > 0 else None
    if d < settling['bound']:
        out['case'] = 'unresolved_against_settling'
        out['settling_decided_by'] = settling['decided_by']
        out['settling_is_proxy'] = settling['proxy']
        # True where only a proxy puts d under the bound.
        out['settling_call_rests_on_proxy'] = bool(firm is None or d >= firm)
        out['ranking_withheld'] = (f"d = {d:.3g} is below the settling bound {settling['bound']:.3g} "
                                   f"({settling['decided_by']}): the runs are not settled finely "
                                   'enough to separate them')
        return out
    gap = abs(worst_first - worst_second)
    if roundoff_floor:
        out['ranking_withheld'] = ('u is the round-off floor of an exact reference, so every '
                                   'e1 != e2 would clear 10 u')
    else:
        out['ranking_margin'] = dict(
            gap=gap, relative_gap=gap/max(worst_first, worst_second),
            gap_over_10u=gap/(RESOLVABLE*u) if u > 0 else None, confound_shift=confound_shift,
            gap_over_confound_shift=gap/confound_shift if confound_shift > 0 else None)
        if gap < RESOLVABLE*u or worst_first == worst_second:
            out['ranking_withheld'] = (f'|e1 - e2| = {gap:.3g} is below {RESOLVABLE:g} u = '
                                       f'{RESOLVABLE*u:.3g}')
        elif gap < confound_shift:
            out['ranking_withheld'] = (f'fragile: |e1 - e2| = {gap:.3g} is below {confound_shift:.3g}, '
                                       'the shift the measured de-embedding confound put on the '
                                       'same quantity')
        else:
            winner = names[0] if worst_first < worst_second else names[1]
            out['more_accurate'] = winner
            out['ranking'] = f'{winner} more accurate'
    if all(resolved):
        ratio = d/smaller
        if roundoff_floor and smaller*RESOLVABLE < larger:
            # d/min(e) is then set by one route's near-zero error, not by the pair.
            out['case'] = 'one_route_near_exact'
        else:
            out['case'] = ('same_error' if ratio <= SAME_ERROR_RATIO else
                           'route_changes_answer' if ratio >= ROUTE_MATTERS_RATIO else 'intermediate')
    elif any(resolved):
        out['case'] = 'one_route_below_reference_resolution'
    else:
        out['case'] = 'both_below_reference_resolution'
    return out


def pair_summary(rows, a, b, u, settling, roundoff_floor, confound_shift=0.):
    """One pair on one metric: the channel carrying the largest ``d``, and its classification."""
    col_a, col_b, col_d = f'{a}_vs_reference', f'{b}_vs_reference', f'{a}_vs_{b}'
    usable = [r for r in rows if None not in (r[col_a], r[col_b], r[col_d])]
    entry = dict(channels=len(usable))
    if len(usable) != len(rows):
        entry['nonfinite_channels'] = len(rows) - len(usable)
    if not usable:
        entry['classification'] = None
        return entry
    top = max(usable, key=lambda r: r[col_d])
    worst_a, worst_b = max(r[col_a] for r in usable), max(r[col_b] for r in usable)
    entry.update(d=top[col_d], d_channel=dict(channel=top['channel'], order=top['order'],
                                               e_first=top[col_a], e_second=top[col_b],
                                               reference=top['reference'],
                                               exit_cos=top['exit_cos'], grazing=top['grazing']),
                 e_first_max=worst_a, e_second_max=worst_b, confound_shift=confound_shift)

    def zero_reference(r):
        return roundoff_floor and abs(complex(*r['reference'])) <= ZERO_REFERENCE

    def ratio_entry(r):
        return dict(ratio=r[col_d]/min(r[col_a], r[col_b]), channel=r['channel'], order=r['order'],
                    d=r[col_d], e_first=r[col_a], e_second=r[col_b])
    positive = [r for r in usable if min(r[col_a], r[col_b]) > 0]
    resolved = [r for r in positive
                if min(r[col_a], r[col_b]) >= RESOLVABLE*u and not zero_reference(r)]
    if resolved:
        entry['largest_resolved_channel_ratio'] = ratio_entry(
            max(resolved, key=lambda r: r[col_d]/min(r[col_a], r[col_b])))
    zeros = [r for r in positive if zero_reference(r)]
    if zeros:
        entry['largest_zero_reference_channel_ratio'] = dict(
            ratio_entry(max(zeros, key=lambda r: r[col_d]/min(r[col_a], r[col_b]))),
            note='reference zero by symmetry on a round-off-floor scene: not counted as resolved')
    firm, channel_settling = settling.get('firm'), None
    if 'amplitude_bound' in settling:
        # A power moves by 2|a| da + da**2; |a| from this channel's own reference power
        # (admittance taken as 1, so an estimate, not a bound).
        da = settling['amplitude_bound']
        channel_settling = 2*math.sqrt(abs(top['reference'][0]))*da + da**2
        firm = channel_settling
    entry['classification'] = classify(top[col_a], top[col_b], top[col_d], u, settling,
                                       roundoff_floor, worst_a, worst_b, (a, b), firm=firm,
                                       channel_settling=channel_settling,
                                       confound_shift=confound_shift)
    return entry


def summarise(rows, arms, pairs, u, floors, settling):
    """Both metrics, every classified pair; neither metric is the headline on its own."""
    out = {}
    for kind, power in (('amplitude', False), ('power', True)):
        chosen = [r for r in rows if (r['channel'].split('.')[1] in POWERS) == power]
        entry = dict(channels=len(chosen), u=u[kind], u_is_roundoff_floor=floors[kind],
                     settling=settling[kind], errors={}, pairs={})
        for arm in arms:
            values = [r[f'{arm}_vs_reference'] for r in chosen]
            finite = [v for v in values if v is not None]
            entry['errors'][arm] = max(finite) if finite else None
            if len(finite) != len(values):
                entry['errors'][f'{arm}_nonfinite'] = len(values) - len(finite)
        for a, b in pairs:
            # The engine pair's ranking must clear how far the de-embedding swap moved legacy's
            # worst error on this metric (zero on power, which the swap does not touch).
            shift = 0.
            if a == 'legacy_yee_deembedded' and None not in (entry['errors'].get('legacy'),
                                                             entry['errors'].get(a)):
                shift = abs(entry['errors']['legacy'] - entry['errors'][a])
            entry['pairs'][f'{a}_vs_{b}'] = pair_summary(chosen, a, b, u[kind], settling[kind],
                                                         floors[kind], shift)
        out[kind] = entry
    return out


def settling_bounds(legacy, common):
    """How far each run's answer is from settled, per metric, and which part sets it.

    Only common's ``scattering_change_bound`` bounds an amplitude change. Legacy's ``residual``
    is a field tail-to-peak ratio (``_core/fdtd.py``, ``_time_loop``) and the power bound is a
    channel-blind multiple, so both are marked proxies.
    """
    parts = dict(legacy_residual=max(leg['residual'] for leg in legacy['legs']))
    if common.get('convergence', {}).get('scattering_change_bound') is not None:
        parts['common_scattering_change_bound'] = common['convergence']['scattering_change_bound']
    decided_by = max(parts, key=parts.get)
    amplitude = parts[decided_by]
    return dict(
        amplitude=dict(bound=amplitude, decided_by=decided_by,
                       proxy=decided_by != 'common_scattering_change_bound',
                       firm=parts.get('common_scattering_change_bound')),
        power=dict(bound=POWER_SETTLING_FACTOR*amplitude,
                   decided_by=f'{POWER_SETTLING_FACTOR:g} x {decided_by}', proxy=True,
                   amplitude_bound=amplitude),
        parts=parts)


def withhold_on_inputs(summary, materials):
    """Stamp every matched pair of a scene whose materials differ.

    Its ``d`` and ``e`` stay as measurements, but no case or ranking is an engine claim: the
    engines were stepping different materials.
    """
    if materials['matched']:
        return
    reason = 'materials differ: ' + ', '.join(materials['differing'] + materials['missing'])
    for kind in summary:
        for pair in ('legacy_vs_common', 'legacy_yee_deembedded_vs_common'):
            classification = summary[kind]['pairs'][pair].get('classification')
            if classification is not None:
                classification.update(case='inputs_differ', inputs_differ=True, ranking=None,
                                      more_accurate=None, ranking_withheld=reason)


def deembedding_control(summary, rows):
    """The swap as the raw arm's positive control: its size next to both ``d``, and what the
    raw classification made of it."""
    effects = [r for r in rows if r['channel'].split('.')[1] in AMPLITUDES
               and r['legacy_vs_legacy_yee_deembedded'] is not None]
    raw = summary['amplitude']['pairs']['legacy_vs_common']
    fixed = summary['amplitude']['pairs']['legacy_yee_deembedded_vs_common']
    if not effects or raw['classification'] is None or fixed['classification'] is None:
        return dict(effect=max((r['legacy_vs_legacy_yee_deembedded'] for r in effects), default=None),
                    comparable=False)
    effect = max(r['legacy_vs_legacy_yee_deembedded'] for r in effects)
    # The channel the raw d is taken on, so this ratio does not mix channels.
    on_channel = next((r['legacy_vs_legacy_yee_deembedded'] for r in effects
                       if r['channel'] == raw['d_channel']['channel']
                       and r['order'] == raw['d_channel']['order']), None)
    raw_case, fixed_case = raw['classification']['case'], fixed['classification']['case']

    def over(value, base):
        return value/base if value is not None and base > 0 else None
    return dict(
        what="legacy re-de-embedded with common's Yee kz: zero parameters, post-processing only",
        sized_against='the raw d; it certifies the raw arm only',
        effect=effect, effect_on_raw_d_channel=on_channel, raw_d=raw['d'], corrected_d=fixed['d'],
        effect_over_raw_d=over(effect, raw['d']),
        effect_on_raw_d_channel_over_raw_d=over(on_channel, raw['d']),
        effect_over_corrected_d=over(effect, fixed['d']),
        raw_case=raw_case, corrected_case=fixed_case,
        raw_ratio=raw['classification']['d_over_min_error'],
        corrected_ratio=fixed['classification']['d_over_min_error'],
        raw_case_differs=raw_case != fixed_case,
        raw_overstates=bool(raw_case in _CASE_SCALE and fixed_case in _CASE_SCALE
                            and _CASE_SCALE.index(raw_case) > _CASE_SCALE.index(fixed_case)),
        corrected_unresolved_against_settling=fixed_case == 'unresolved_against_settling')


def scene_verdict(summary, materials):
    """The engine comparison on both metrics; a ranking only where the two metrics agree."""
    def cases(pair):
        return {kind: (summary[kind]['pairs'][pair]['classification'] or {}) for kind in summary}
    engines = cases('legacy_yee_deembedded_vs_common')
    raw = cases('legacy_vs_common')

    def unify(name):
        return 'legacy' if name == 'legacy_yee_deembedded' else name
    picks = [unify(engines[kind].get('more_accurate')) if engines[kind].get('more_accurate') else None
             for kind in ('amplitude', 'power')]
    ranking, withheld = None, None
    if not materials['matched']:
        # Engines stepping different materials are not being compared as engines.
        withheld = 'materials differ: ' + ', '.join(materials['differing'] + materials['missing'])
    elif None in picks:
        withheld = '; '.join(f"{kind}: {engines[kind].get('ranking_withheld')}"
                             for kind, pick in zip(('amplitude', 'power'), picks) if pick is None)
    elif picks[0] != picks[1]:
        withheld = f'amplitude ranks {picks[0]}, power ranks {picks[1]}'
    else:
        ranking = f'{picks[0]} more accurate'
    out = dict(materials_matched=bool(materials['matched']),
               engines=dict(amplitude=engines['amplitude'].get('case'),
                            power=engines['power'].get('case'),
                            cases_agree=engines['amplitude'].get('case') == engines['power'].get('case'),
                            ranking=ranking, ranking_withheld=withheld),
               raw=dict(amplitude=raw['amplitude'].get('case'), power=raw['power'].get('case')))
    if ranking is not None:
        out['engines']['ranking_margin'] = {kind: engines[kind].get('ranking_margin')
                                            for kind in ('amplitude', 'power')}
    if not materials['matched']:
        # A difference in what the engines were given is not a difference between engines.
        out['engines']['confounded_by_inputs'] = materials['differing'] + materials['missing']
        out['engines']['input_defect_owner'] = sorted({v['owner'] for v in
                                                       materials.get('attributed', {}).values()})
    return out


_VARIANT = re.compile(r'^(?P<family>.+)-r(?P<resolution>[0-9.]+)(?:-pad(?P<pad>[0-9.]+))?'
                      r'(?:-(?P<termination>pml16))?$')


def families(scenes):
    """Each route's worst error (engine pair) across a scene's resolution and pad variants.

    Scenes that differ only in ``pad_scale`` share one physics and one grid; a smaller-error
    route that changes between them is an effect of the clearance, which the two routes lay
    out differently (``pml_structure_gap``), not of either engine. Only matched scenes count.
    """
    out = {}
    for name, entry in scenes.items():
        match = _VARIANT.match(name)
        if match is None:
            continue
        row = dict(scene=name, resolution=float(match['resolution']),
                   pad=float(match['pad']) if match['pad'] else None,
                   materials_matched=entry['verdict']['materials_matched'])
        for kind in ('amplitude', 'power'):
            pair = entry['summary'][kind]['pairs']['legacy_yee_deembedded_vs_common']
            if pair.get('classification') is None:
                continue
            first, second = pair['e_first_max'], pair['e_second_max']
            row[kind] = dict(legacy=first, common=second,
                             smaller='legacy' if first < second else 'common' if second < first else 'tie')
        # A strengthened termination is its own family: its errors are not the base scene's.
        family = match['family'] + (f"-{match['termination']}" if match['termination'] else '')
        out.setdefault(family, dict(variants=[]))['variants'].append(row)
    for family in out.values():
        flips, changes = {}, {}
        for kind in ('amplitude', 'power'):
            rows = [r for r in family['variants'] if kind in r and r['materials_matched']]
            by_resolution, by_pad = {}, {}
            for r in rows:
                by_resolution.setdefault(r['resolution'], []).append(r)
                by_pad.setdefault(r['pad'], []).append(r)
            flips[kind] = sorted(r['scene'] for group in by_resolution.values()
                                 if len(group) > 1 and len({g[kind]['smaller'] for g in group}) > 1
                                 for r in group)
            changes[kind] = any(len(group) > 1 and len({g[kind]['smaller'] for g in group}) > 1
                                for group in by_pad.values())
        family['smaller_route_flips_with_pad'] = flips
        family['smaller_route_changes_with_resolution'] = changes
    return out


def cases_by_scene(scenes, family_table):
    """``case -> scene names``, read off each scene's verdict, so no prose can disagree with it."""
    pad_flip = {kind: {name for family in family_table.values()
                       for name in family['smaller_route_flips_with_pad'][kind]}
                for kind in ('amplitude', 'power')}
    out = {}
    for name, entry in scenes.items():
        verdict = entry['verdict']
        if entry['match'].get('dt_exempt'):
            out.setdefault(f"dt exempt from matching ({entry['match']['dt_exempt']})", []).append(name)
        if not verdict['materials_matched']:
            out.setdefault('inputs_differ (engines not compared)', []).append(name)
            continue
        for kind in ('amplitude', 'power'):
            case = verdict['engines'][kind]
            key = f'{kind}: {case}'
            if name in pad_flip[kind] and case not in ('same_error', 'unresolved_against_settling'):
                key += ' (smaller-error route flips with pad_scale: pml_structure_gap)'
            out.setdefault(key, []).append(name)
        if verdict['engines']['ranking']:
            out.setdefault(f"ranking: {verdict['engines']['ranking']}", []).append(name)
        engine = entry['summary']['amplitude']['pairs']['legacy_yee_deembedded_vs_common']
        d_channel = engine.get('d_channel') or {}
        resolved = (engine.get('classification') or {}).get('case') != 'unresolved_against_settling'
        if d_channel.get('grazing') and resolved:
            out.setdefault(f'engine d on a grazing channel (exit cos < {GRAZING_COS:g}; '
                           'see termination_control)', []).append(name)
        if entry['deembedding_control'].get('raw_overstates'):
            out.setdefault('control: raw arm overstates the route difference', []).append(name)
    return out


def termination_control(scenes):
    """What strengthening the termination did, for each material scene run beside its twin.

    Per compared amplitude channel: the engine ``d`` with the study's CPML and with the twin's,
    and how far each engine's own amplitude moved between the two runs. The twin changes three
    things at once -- the CPML target, its depth and the run length -- so a move holds what that
    engine's termination was sending back and whatever the base run had not yet settled. Both
    runs' settling bounds are carried, and a twin ``d`` below its own bound is marked unresolved.
    A grazing channel whose ``d`` falls far below its base value, resolved, was measuring the
    terminations, not the engines.
    """
    out = {}
    for name, entry in scenes.items():
        twin = scenes.get(f'{name}-pml16')
        if twin is None:
            continue
        twin_rows = {(r['channel'], tuple(r['order'])): r for r in twin['channels']}
        channels = []
        for r in entry['channels']:
            s_ = twin_rows.get((r['channel'], tuple(r['order'])))
            if s_ is None or r['channel'].split('.')[1] not in AMPLITUDES:
                continue

            def moved(arm, r=r, s_=s_):
                a, b = r.get(arm), s_.get(arm)
                return abs(complex(*a) - complex(*b)) if a and b else None
            channels.append(dict(channel=r['channel'], order=r['order'], exit_cos=r['exit_cos'],
                                 grazing=r['grazing'], d_base=r['legacy_yee_deembedded_vs_common'],
                                 d_strengthened=s_['legacy_yee_deembedded_vs_common'],
                                 legacy_moved=moved('legacy_yee_deembedded'),
                                 common_moved=moved('common')))

        def worst(grazing, key):
            values = [c[key] for c in channels if c['grazing'] == grazing and c[key] is not None]
            return max(values, default=None)
        bound = twin['settling']['amplitude']['bound']

        def pair(grazing):
            strengthened = worst(grazing, 'd_strengthened')
            return dict(base=worst(grazing, 'd_base'), strengthened=strengthened,
                        strengthened_resolved=None if strengthened is None else strengthened >= bound)
        out[name] = dict(
            twin=f'{name}-pml16', twin_matched=twin['match']['matched'],
            twin_termination=dict(target=twin['legacy']['pml_target_reflection'],
                                  n_pml=twin['legacy']['n_pml'],
                                  n_periods=twin['legacy']['n_periods_keyword']),
            grazing_d=pair(True), other_d=pair(False),
            base_settling_bound=entry['settling']['amplitude']['bound'], twin_settling_bound=bound,
            base_verdict=entry['verdict']['engines'], twin_verdict=twin['verdict']['engines'],
            channels=channels)
    return out


def annotate(report):
    """Cross-scene fields, recomputed from the scenes after each one so they never go stale."""
    report['families'] = families(report['scenes'])
    report['termination_control'] = termination_control(report['scenes'])
    for name, entry in report['scenes'].items():
        base = name[:-len('-pml16')] if name.endswith('-pml16') else name
        for confound in entry['confounds']:
            if confound['id'] in ('pml_backing', 'pml_structure_gap'):
                confound['measured_in'] = ('termination_control' if base in report['termination_control']
                                           else None)
    for family in report['families'].values():
        for kind, names in family['smaller_route_flips_with_pad'].items():
            for name in names:
                report['scenes'][name]['verdict'].setdefault(
                    'smaller_route_flips_with_pad', {})[kind] = True
    report['cases'] = cases_by_scene(report['scenes'], report['families'])


# =======================================================================================
# Provenance and the machine
# =======================================================================================

REFERENCE_FILES = ('tests/references/_meep_crossed_cases.py',
                   'tests/references/_meep_crossed_material_cases.py',
                   'tests/references/_meep_crossed_metal_case.py',
                   'tests/references/_meep_1d_cases.py',
                   'tests/references/test_meep_crossed_materials_golden.py')


def _digest(paths):
    """sha256 over each path and its bytes, with CRLF read as LF.

    Git stores these files with LF and a Windows checkout (``core.autocrlf``) writes CRLF, so
    hashing the bytes as checked out gave one commit two digests: a result measured on Windows
    could not be matched to its source on Linux.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def provenance():
    package = sorted((ROOT / 'src/wavebit').rglob('*.py'))
    references = [ROOT / p for p in REFERENCE_FILES]
    this = Path(__file__).resolve()

    def git(*a):
        return subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    return dict(
        study_sha256=_digest([this]), references_sha256=_digest(references),
        package_sha256=_digest(package), source_sha256=_digest([this, *references, *package]),
        git_head=git('rev-parse', 'HEAD'),
        package_tree_dirty=bool(git('status', '--porcelain', '--', 'wavebit', 'tests/references')),
        python=platform.python_version(), platform=platform.platform(),
        versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'array-api-compat')},
        thread_pins={k: os.environ.get(k) for k in THREAD_PINS})


def machine_snapshot():
    """Other Python processes at this moment; wall clock with any of them alive is contended."""
    try:
        rows = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                              capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return dict(error=str(error))
    others = []
    for line in rows.splitlines():
        cells = [c.strip('"') for c in line.split('","')]
        if len(cells) >= 5 and cells[1].isdigit() and int(cells[1]) != os.getpid():
            others.append(dict(pid=int(cells[1]), memory=cells[4]))
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), other_python=others,
                contended=bool(others))


# =======================================================================================
# Main
# =======================================================================================

def run_scene(name, args, cache, timing):
    scene = SCENES[name]
    reference, ref_entry = solve_reference(scene, cache, timing)
    with MaterialCapture() as legacy_capture:
        legacy_values, legacy_energy, legacy_complete, runs = run_legacy(scene, timing)
    legacy = legacy_record(scene, runs)
    kw, notes = matched_common(scene, runs, args.common_pad)
    refused = None
    try:
        with MaterialCapture() as common_capture:
            common_values, common_energy, common_complete, _, diag, caught = run_common(
                scene, kw, timing, 'common')
    except ValueError as error:
        # Common's pole-aware CFL bound (grid_fdtd.fdtd_grid) can sit below legacy's dt; the
        # refusal is a measured fact about the routes, so it is recorded and dt alone unmatched.
        if 'conservative CFL bound' not in str(error) or 'poles' not in scene['call']:
            raise
        refused = dict(courant=kw['courant'], error=str(error))
        kw, notes = matched_common(scene, runs, args.common_pad, pole_cfl=True)
        with MaterialCapture() as common_capture:
            common_values, common_energy, common_complete, _, diag, caught = run_common(
                scene, kw, timing, 'common')
    common = common_record(scene, kw, diag, caught)
    materials = compare_materials(scene, legacy_capture, common_capture, diag)
    checks = match_checks(legacy, common, notes, materials)
    arms = {'legacy': legacy_values, 'common': common_values,
            'legacy_yee_deembedded': yee_deembedded(scene, legacy, legacy_values)}
    energies = {'reference': ref_entry.pop('energy'), 'legacy': legacy_energy,
                'common': common_energy}
    complete = {'reference': ref_entry['window_complete'], 'legacy': legacy_complete,
                'common': common_complete}
    entry = dict(group=scene['group'], source=scene['source'], kind=scene['kind'],
                 lossless=scene['lossless'],
                 scene={k: _plain(v) for k, v in scene['call'].items()},
                 legacy_keywords=_plain(scene['study']), reference=ref_entry,
                 legacy=legacy, common=common, match=checks)
    if refused is not None:
        entry['matched_dt_refused_by_common'] = refused
    if args.check_inert:
        bare = {k: v for k, v in kw.items() if k != 'convergence_window'}
        bare_values, *_ = run_common(scene, bare, timing, 'common_without_convergence_window')
        entry['convergence_window_inert'] = dict(
            bit_identical=all(np.array_equal(bare_values[k], common_values[k], equal_nan=True)
                              for k in common_values),
            max_difference=max(float(np.nanmax(np.abs(bare_values[k] - common_values[k]),
                                               initial=0.)) for k in common_values))
    if 'verbatim' in args.arms:
        verbatim = verbatim_common(scene)
        try:
            values, verbatim_energy, verbatim_complete, _, vdiag, vcaught = run_common(
                scene, verbatim, timing, 'common_verbatim')
        except ValueError as error:
            # What a user gets by adding backend=0 includes a refusal; that is the measurement.
            entry['common_verbatim'] = dict(refused=str(error))
        else:
            arms['common_verbatim'] = values
            energies['common_verbatim'] = verbatim_energy
            complete['common_verbatim'] = verbatim_complete
            entry['common_verbatim'] = common_record(scene, verbatim, vdiag, vcaught)
    # Raw route difference, the engine comparison, and (when asked) what a user gets verbatim.
    # The fourth distance, legacy vs its own re-de-embedding, is the positive control's size.
    pairs = [('legacy', 'common'), ('legacy_yee_deembedded', 'common')]
    if 'common_verbatim' in arms:
        pairs.append(('legacy', 'common_verbatim'))
    rows = compare(scene, reference, arms, pairs + [('legacy', 'legacy_yee_deembedded')])
    # One settling bound per scene, from the matched runs; a verbatim arm has no convergence
    # window, so it is held against the same bound.
    entry['settling'] = settling_bounds(legacy, common)
    entry['materials'] = materials
    entry['summary'] = summarise(rows, arms, pairs, ref_entry['u'], ref_entry['u_is_roundoff_floor'],
                                 entry['settling'])
    withhold_on_inputs(entry['summary'], materials)
    entry['deembedding_control'] = deembedding_control(entry['summary'], rows)
    entry['verdict'] = scene_verdict(entry['summary'], materials)
    label = 'energy_defect' if scene['lossless'] else 'absorption'
    entry['energy'] = dict(
        quantity=f'1 - sum(R) - sum(T) over propagating orders ({label})',
        window_complete=complete,
        per_leg={leg: {arm: energies[arm][leg] for arm in energies} for leg, _ in legs_of(scene)})
    entry['confounds'] = confounds(scene, legacy, common, checks, notes, rows)
    entry['channels'] = rows
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', nargs='+', metavar='NAME',
                        help=f'scene names, shell patterns, groups ({", ".join(GROUPS)}) or all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timing', type=Path,
                        help='wall-clock file (default: <output stem>.timing.json)')
    parser.add_argument('--arms', default='matched', choices=('matched', 'matched,verbatim'),
                        help='add a common_verbatim arm: common with the study keywords verbatim')
    parser.add_argument('--common-pad', default='legacy', choices=('legacy', 'default'))
    parser.add_argument('--check-inert', action='store_true',
                        help='rerun common without convergence_window and compare bit for bit')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        for name, scene in SCENES.items():
            print(f"{scene['group']:10s} {name}")
        return
    if not args.scene or args.output is None:
        parser.error('--scene and --output are required unless --list')
    names = select(args.scene)
    timing_path = args.timing or args.output.with_name(args.output.stem + '.timing.json')
    report = dict(schema=2, complete=False, **provenance(),
                  scope='The legacy and common FDTD routes on shared scenes, each against one '
                        'converged reference; propagating channels with n^2 - k_par^2 > '
                        f'{MARGIN} only. Engines are compared only where materials match, as '
                        'measured per scene (materials); a difference with a known owner is '
                        'named in materials.attributed. Neither '
                        'legacy-only behaviour nor common at its own defaults is settled here.',
                  definitions=dict(
                      arms='legacy (as shipped); common (matched to legacy); '
                           "legacy_yee_deembedded (legacy re-de-embedded with common's Yee kz, "
                           'the engine comparison); common_verbatim (optional)',
                      e='max |arm - reference| over the channels of one metric (errors)',
                      d='|a - b| per channel; a pair is summarised on the channel carrying '
                        'the largest d, with that channel\'s own e1 and e2; d/min(e), '
                        'd/max(e) and |e1 - e2| are all reported',
                      u='reference uncertainty: 2x the last order step, or step*N/(N-N_prev) '
                        'when the tail is slow; floored by any closed-form gap, and by '
                        f'{ROUNDOFF:g} on a uniform stack (u_is_roundoff_floor: no ranking)',
                      cases='decided on the channel carrying the largest d, from that '
                            "channel's own e1 and e2, in this order. "
                            'unresolved_against_settling: d below the settling bound. An e is '
                            f'resolved at >= {RESOLVABLE:g} u. With both resolved: '
                            'one_route_near_exact: round-off-floor u and one e more than '
                            f'{RESOLVABLE:g}x below the other; otherwise same_error: '
                            f'd <= {SAME_ERROR_RATIO:.3g} min(e1,e2), route_changes_answer: '
                            f'd >= {ROUTE_MATTERS_RATIO:g} min(e1,e2), intermediate: between '
                            'the two. one_route_below_reference_resolution: only one e '
                            'resolved. both_below_reference_resolution: neither, so the '
                            'reference cannot judge the pair whatever d/min(e) is. '
                            'inputs_differ: materials do not match, so no case or ranking is '
                            'an engine claim',
                      ranking=f'needs |e1 - e2| >= {RESOLVABLE:g} u and above the measured '
                              'de-embedding shift of the same quantity (confound_shift); '
                              'ranking_margin carries the numbers',
                      settling='amplitude: max(common scattering_change_bound, legacy '
                               f'residual); power: {POWER_SETTLING_FACTOR:g}x that. Only '
                               "common's part bounds an amplitude change; legacy's residual "
                               '(a field tail/peak ratio) and the power factor are proxies. '
                               'settling_decided_by names the part that decided; '
                               'channel_settling_estimate scales the power bound by the '
                               "channel's own reference power",
                      matched='grid, dt, planes, PML, envelope, post-peak time, pad, and every '
                              'per-component eps/mu/pole-fill array to '
                              f'{MATERIAL_TOL:g} relative; dt is exempt only where the scene '
                              'records dt_reason (a 1D conical mount, or common refusing '
                              "legacy's dt under its pole CFL bound), and the dt ratio is then "
                              'a confound',
                      deembedding_control="legacy vs legacy_yee_deembedded: sized against the "
                                          'raw d and certifies the raw arm only',
                      families='worst errors of the engine pair across resolution and '
                               'pad_scale variants; a smaller-error route that flips with '
                               'pad_scale alone is the pml_structure_gap confound',
                      verdict='engines = legacy_yee_deembedded vs common on amplitude and power; '
                              'a ranking only where both metrics rank the same route',
                      cases_field='scene names by verdict, generated; scenes whose materials '
                                  'differ are listed apart and not compared as engines',
                      exit_cos='cos(theta) of the order in the medium it leaves into (top for '
                               'a reflection, bottom for a transmission), from the continuum '
                               'in-plane wavevector',
                      grazing=f'exit_cos < {GRAZING_COS:g}. Measured on the material scenes: the '
                              'termination sends part of such an order back, the two engines '
                              'differently, and on (0, 1) that return did not shrink with the grid '
                              '(termination_control)',
                      termination_control='each material scene against its -pml16 twin (CPML '
                                          f"target {PML16['target']:g}, {PML16['n_pml']} cells, "
                                          f"{PML16['n_periods']} periods, so a move also holds what "
                                          'the base run had not settled): the engine d with each '
                                          'termination, how far each engine moved, and both '
                                          'settling bounds; the twin passes pml_reflection to both '
                                          'routes and the record holds the target legacy reports',
                      known_input_defects=KNOWN_INPUT_DEFECTS),
                  settings=dict(scenes_requested=args.scene, arms=args.arms.split(','),
                                common_pad=args.common_pad, check_inert=args.check_inert,
                                y_invariant_cell_dx=Y_INVARIANT_CELL,
                                convergence_fraction=CONVERGENCE_FRACTION,
                                grazing_cos=GRAZING_COS, termination_twin=PML16),
                  scenes={})
    timing = dict(note='Wall clock and CPU seconds. Any snapshot with other Python processes '
                       'alive marks the run contended; a contended timing is not a clean one.',
                  start=machine_snapshot(), scenes={})

    def save():
        for path, payload in ((args.output, report), (timing_path, timing)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    cache = {}
    for name in names:
        clock = timing['scenes'][name] = dict(before=machine_snapshot())
        report['scenes'][name] = entry = run_scene(name, args, cache, clock)
        annotate(report)
        clock['after'] = machine_snapshot()
        save()
        verdict = entry['verdict']
        status = ('matched' if entry['match']['matched'] else
                  'NOT MATCHED: ' + ', '.join(k for k, v in entry['match']['passed'].items() if not v)
                  + (f" ({', '.join(entry['match']['materials_differing'])})"
                     if entry['match']['materials_differing'] else ''))
        print(name, status, '| raw', verdict['raw'], '| engines', verdict['engines'], flush=True)
    report['complete'] = True
    timing['end'] = machine_snapshot()
    save()


if __name__ == '__main__':
    main()
