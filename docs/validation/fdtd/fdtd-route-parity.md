# FDTD route parity: do the two engines make the same error?

Written: 2026-09-17

Targets:
- `src/wavebit/compat/legacy.py`
- `src/wavebit/manifest.py`
- `src/wavebit/fdtd/_periodic/legacy.py`
- `src/wavebit/fdtd/_periodic/scene.py`
- `src/wavebit/fdtd/_periodic/grid.py`
- `studies/fdtd/route_parity/fdtd_route_parity.py`
- `tests/fdtd/test_fdtd_route_parity_capture.py`
- `tests/fdtd/test_fdtd_pole_colocation.py`

TL;DR: on 40 scenes, with the grid, monitor planes, CPML, pulse envelope and every material
array matched, `dt` matched on 35 of them, and the one post-processing difference that can be
measured (the de-embedding phase) removed, the legacy engine and the shared engine make the
**same error** on 50 of 80 classified calls, and on 19 scenes on both amplitude and power.
Without that removal the comparison overstates the difference on 27 scenes. **No route ranks at
scene level on any scene**: 23 are withheld because their reference is exact to round-off, which
is a rule rather than a measured tie; on 16 of the other 17 at least one metric is withheld for
want of resolution, and on the last amplitude is withheld as fragile. Power alone ranked the
shared engine on 3 scenes, by 1.2-5.1%, but two of those gaps are within 1.5x a proxy settling
bound and shrink to about 2e-6, below resolution, in their strengthened-termination twins. On the
material scenes most of the grazing-order difference went away when the CPML target, depth and
run length were strengthened together; a resolved remainder stays on both Lorentz scenes. On
power, all four vacuum scenes change their smaller-error route with the padding, and a uniform
slab disagrees with nothing to attribute it to. The record does not measure the shared engine at
its own defaults, does not isolate the source injection, and was taken on a package that has
since changed (section 3). It does not license changing `default_route` or removing either
engine.

## 1. The question, and how the two engines differ

`wavebit.fdtd` has two routes (`src/wavebit/manifest.py`, `fdtd.routes`; `default_route` is
`legacy_numpy`). With `backend` omitted, `src/wavebit/fdtd/_periodic/api.py::fdtd` calls
`src/wavebit/fdtd/_periodic/legacy.py::solve_fdtd`, the **legacy engine**: NumPy only, not differentiable, top
incidence only. With an explicit `backend` it calls `src/wavebit/fdtd/_periodic/scene.py::solve_scene_fdtd`,
the **shared engine** over `src/wavebit/fdtd/_periodic/grid.py`: NumPy, JAX and Torch, differentiable on JAX and
Torch, both incidence sides. They share no Maxwell update. Their defaults and methods differ:

| | Legacy engine | Shared engine |
| --- | --- | --- |
| `courant` | 0.99 (`src/wavebit/fdtd/_periodic/legacy.py::COURANT`) | 0.8 |
| `n_periods` | 80 (`src/wavebit/fdtd/_periodic/legacy.py::N_PERIODS`) | 40 |
| Pulse peak | 4 widths (`src/wavebit/fdtd/_periodic/legacy.py::SOURCE_DELAY`) | 5 widths |
| CPML target | 1e-8 (`src/wavebit/fdtd/_periodic/legacy.py::PML_TARGET_REFLECTION`) | 1e-8 (`pml_reflection`) |
| Source | TFSF plane fed by a 1D incident grid | two current sheets; a uniform-medium reference run calibrates the incident amplitude and is subtracted at the reflection plane |
| De-embedding phase | continuum `kz` | discrete Yee `kz` |
| PML backing | E grounded on the end rows | periodic z wrap with a closing cell |
| PML-to-structure gap | larger above the structure, where the source, reflection and TFSF planes sit, than below | one pad on both sides, equal to within a cell |

Two independent discretisations will not agree bit for bit, so the route difference is never
read alone. For each scene and channel the study reports, against one converged reference:

| Symbol | Meaning |
| --- | --- |
| `e1` | legacy's own error, `abs(legacy - reference)`; in the engine pair, legacy after the de-embedding swap (section 6) |
| `e2` | the shared engine's own error |
| `d` | `abs(legacy - shared)`, the route difference; it involves no reference |
| `u` | the reference's uncertainty, measured from an RCWA order series in every run; floored at `ROUNDOFF` = 1e-14 where the reference is exact |

The reference is always `wavebit.rcwa(backend='numpy')` at the highest truncation order of its series.
Where a closed form exists it checks that answer, and the gap between them widens the amplitude
`u`. If `d` is small next to `e1` and `e2`, the engines make the same error and the choice
between them is about features.

## 2. How to read a verdict

Each pair is classified twice, on complex amplitudes and on powers. The case is read on the one
channel that carries the largest `d`, with that channel's own `e1` and `e2`; the ranking compares
the two engines' worst errors over all channels of the metric. Decisions are made in this order
(`studies/fdtd/route_parity/fdtd_route_parity.py::classify`):

| Step | Rule | Outcome |
| --- | --- | --- |
| 1 | `d` below the settling bound | `unresolved_against_settling`; no ranking |
| 2 | `u` is the round-off floor | ranking withheld, since every `e1 != e2` would clear 10 u |
| 3 | otherwise, the gap between the worst errors below 10 u | ranking withheld ("below 10 u") |
| 4 | otherwise, that gap below the shift the de-embedding swap put on the same worst error | ranking withheld ("fragile"); zero on power |
| 5 | both errors resolved (at least 10 u), round-off floor, one error more than 10x the other | `one_route_near_exact` |
| 5 | both resolved, `d/min(e)` at most 1/3 | `same_error` |
| 5 | both resolved, `d/min(e)` at least 1 | `route_changes_answer` |
| 5 | both resolved, in between | `intermediate` |
| 5 | one or neither error resolved | `one_route_below_reference_resolution`, `both_below_reference_resolution`: the reference cannot judge the pair |
| scene | materials differ | `inputs_differ`; no case is an engine claim |

The settling bound is the larger of the shared engine's `scattering_change_bound` and legacy's
field `residual`; the power bound is twice that (`POWER_SETTLING_FACTOR`). Only the shared
engine's part bounds an amplitude change; legacy's residual and the power factor are proxies,
and the record marks which part decided. A ranking is not compared with the settling bound.

**A scene ranks a route only when both metrics rank the same one**
(`studies/fdtd/route_parity/fdtd_route_parity.py::scene_verdict`). Neither metric is privileged: amplitude is the
one the de-embedding phase dominates, and power is immune to it.

## 3. Reproduce, and provenance

Run from the root of the checkout the numbers should describe (the study refuses a `wavebit`
imported from anywhere else):

```shell
python studies/fdtd/route_parity/fdtd_route_parity.py --list
python studies/fdtd/route_parity/fdtd_route_parity.py --scene all --output artifacts/fdtd_route_parity.json
```

The output goes to `artifacts/`, which is ignored: a rerun is a local generated output, not
evidence that arrives with a clone. The measured run's record, `fdtd_route_parity.json`, is
archived outside the repository; this page states what it measured. It was run with
`--arms matched` and `--common-pad legacy` (both defaults), without `--check-inert`, and with
the `-pml16` twins at target 1e-16, 24 cells and 640 periods (`settings` in the JSON). It holds
no wall clock, and this page states no cost.

| Field | Value |
| --- | --- |
| Machine | Windows 11 (10.0.22631), CPU, one BLAS/OpenMP thread |
| Software | Python 3.12.14, NumPy 2.5.1, SciPy 1.18.0, array-api-compat 1.15.0; NumPy backend only |
| `git_head` | `99226a8fcda0bc55c6bc153ee71ddd2c4ba0ac25`, the commit the run was made at, since rewritten, so the digests below identify the sources; `package_tree_dirty` false |
| Scenes | 40: 36 shipped scenes and 4 `-pml16` twins |

The record carries source digests (`studies/fdtd/route_parity/fdtd_route_parity.py::provenance`). Recomputed with
the study's own rule (`studies/fdtd/route_parity/fdtd_route_parity.py::_digest`, CRLF read as LF) against the
current tree:

| Digest | Record | Current tree |
| --- | --- | --- |
| `references_sha256` | `eac2a1b0…` | matches |
| `study_sha256` | `95b5d973…` | differs |
| `package_sha256` | `30ac7083…` | differs |
| `source_sha256` | `5018c678…` | differs (it covers the three above) |

**The study source differs** because the `-pml16` twins now pass `pml_reflection` to both
engines, where the measured run set legacy's CPML target by replacing its profile function for
the length of the call, and because its docstring has been corrected since. The two were
compared byte for byte when the keyword was added: the
study's whole legacy leg on `material-magnetic-r20`, the magnetic, Drude and Lorentz r20 twins
and `binary_grating-r50`, and the CPML coefficients on all four twins, identical except that the
diagnostics now name the target `pml_reflection`. That comparison is recorded in the commit that
added the keyword, not in a file here. The record's `definitions.termination_control` text, which
says legacy has no target keyword, describes the run that was measured, not the current tree.

**The package differs**, in five files: device placement for RCWA solves on the JAX and Torch
backends (`src/wavebit/rcwa/_engine/engine.py`, `src/wavebit/rcwa/_engine/parametric.py`, `src/wavebit/rcwa/_backends/jax_backend.py`); the
shared engine now steps its uniform-medium reference on a
two-cell transverse column (`src/wavebit/fdtd/_periodic/scene.py`, held against the full-grid reference by
`tests/fdtd/test_fdtd_uniform_reference.py`); and the legacy engine gained `pml_reflection`
(`src/wavebit/fdtd/_periodic/legacy.py`). These 40 scenes were not rerun after any of them. Read every number below as
a measurement of the package at `git_head`.

## 4. Scenes

| Group | Scenes | What they are | Reference, `u` |
| --- | ---: | --- | --- |
| slab | 18 | 1D and crossed planar slabs; dielectric, magnetic, Drude; normal, oblique, conical (`tests/distribution/fdtd_install_smoke.py`, `psi` duplicates removed) | RCWA, exact: round-off floor |
| crossed | 5 | `square_18deg` at resolution 16.67, 25, 33.33; `rect_28deg` at 20, 30 (`studies/cross_solver/alpha_convergence/fdtd_alpha_convergence.py`) | RCWA order series; amplitude `u` 1.4e-4 and 2.1e-4 |
| vacuum | 4 | resolution 25 and 50, `pad_scale` 1 and 1.5 (`tests/fdtd/test_fdtd_boundary_controls.py`) | RCWA, checked against the analytic phase: round-off floor |
| materials | 4 | crossed magnetic, Drude and Lorentz rasters at resolution 20, Lorentz also at 40 (`tests/references/test_meep_crossed_materials_golden.py`) | amplitude `u` 2.0e-5 to 2.5e-4 |
| materials-pml16 | 4 | the same four with CPML target 1e-16, 24 cells, 640 periods; made by this study | as their base |
| metal | 2 | damped negative-epsilon metal at resolution 30 and 60 (`studies/fdtd/metal_validation/fdtd_metal_validation.py`) | amplitude `u` 2.08e-3 |
| binary | 3 | `slab_uniform-r50` (RCWA, checked against the Airy closed form) and the binary grating at resolution 50 and 100 of [`fdtd.md`](../../guide/fdtd/fdtd.md) section 2 | round-off floor; grating amplitude `u` 8.2e-6 |

## 5. What matched

The legacy engine runs with the study's keywords; on the twins that includes the CPML target
(section 3). The grid it built is read back from its diagnostics, and the shared engine is driven
onto it (`studies/fdtd/route_parity/fdtd_route_parity.py::matched_common`): cell sizes and counts, `dt`, envelope
width, time after the pulse peak, CPML depth and target, monitor planes on the same nodes, and a
pad sized to reach legacy's upper gap. The pads match in cells on all 40 scenes; the upper gap
itself comes out one cell larger on the shared engine on 12.

- `match.matched` is true on 40 of 40 scenes. Every per-component `eps`, `mu` and pole-fill
  array each engine steps with agrees node for node within 1e-9 relative (`MATERIAL_TOL`), so
  no scene is `inputs_differ`. Materials are compared because a schedule can match exactly
  while a pole fill does not: the first run of this study found legacy filling Ey's pole
  strength half a cell off in y (`studies/fdtd/route_parity/fdtd_route_parity.py::KNOWN_INPUT_DEFECTS`), fixed
  since and pinned by `tests/fdtd/test_fdtd_pole_colocation.py`.
  `tests/fdtd/test_fdtd_route_parity_capture.py` keeps the capture able to see that defect.
- `dt` is exempt on 5 scenes. The three 1D conical slabs run at ratio 0.8165: a conical mount
  on a 1D grating cannot reach legacy's `dt` inside `courant < 1`. `material-lorentz-r20` and
  its twin run at 0.9829: the shared engine refused legacy's `dt` under its pole-aware CFL
  bound. The `dt` ratio is then a confound.

## 6. De-embedding: a difference that is not an engine difference

Legacy carries each monitor phase back to the interface with the continuum `kz`, the shared
engine with the discrete Yee `kz`. That moves amplitude phases only. The
`legacy_yee_deembedded` arm re-de-embeds legacy's own amplitudes with the shared engine's
formula, on legacy's `dt`, cells and planes, with no free parameter
(`studies/fdtd/route_parity/fdtd_route_parity.py::yee_deembedded`). That arm against the shared engine is the
engine comparison; `legacy` against the shared engine is the raw one. The swap is applied on the
node planes. On a non-conical TM loop legacy samples its monitors half a cell below them, so an
`O((kz_yee - kz) dz)` residual stays on those channels. By the study's own estimate, which is not
in the record, moving the monitor term alone onto the half-cell plane shifts their `d` by up to
about 20% with no case changed; the incident term was not estimated.

| Amplitude case | Raw (`legacy`) | Engine (`legacy_yee_deembedded`) |
| --- | ---: | ---: |
| `same_error` | 4 | 30 |
| `intermediate` | 24 | 1 |
| `route_changes_answer` | 9 | 0 |
| `unresolved_against_settling` | 2 | 5 |
| `both_below_reference_resolution` | 1 | 4 |

The swap alone is 0.57 (`material-lorentz-r40`) to 1.09 (`vacuum-r25-pad1`) times the raw `d`
(`deembedding_control.effect_over_raw_d`). The raw case differs from the engine case on 33
scenes and sits higher on the `same_error < intermediate < route_changes_answer` scale on 27
(`raw_overstates`). Power cases are identical between the two arms on all 40 scenes, as they
must be. The swap is the positive control for the raw arm only: it is much larger than most
engine `d`, so it certifies nothing about the engine arm's resolution.

## 7. Results

Engine cases (`verdict.engines`), counts over 40 scenes:

| Case | Amplitude | Power |
| --- | ---: | ---: |
| `same_error` | 30 | 20 |
| `unresolved_against_settling` | 5 | 9 |
| `both_below_reference_resolution` | 4 | 2 |
| `intermediate` | 1 | 4 |
| `route_changes_answer` | 0 | 3 |
| `one_route_near_exact` | 0 | 2 |

Over the 80 calls: `same_error` 50, `unresolved_against_settling` 14,
`both_below_reference_resolution` 6, `intermediate` 5, `route_changes_answer` 3,
`one_route_near_exact` 2. Both metrics are `same_error` on 19 scenes; the two metrics give
different cases on 15. Where `same_error`, `d/min(e)` runs 0.0070-0.27 on amplitude and
0.0046-0.24 on power.

Per scene, generated from the run record `fdtd_route_parity.json` (archived outside the
repository), whose name for the shared engine is `common`. `unresolved` is
`unresolved_against_settling` and `both_below` is `both_below_reference_resolution`; `d/min(e)`
is on the largest-`d` channel and is reported whatever the case.

| Scene | Amplitude | d/min(e) | Power | d/min(e) | No ranking because (amplitude; power) | Notes |
| --- | --- | ---: | --- | ---: | --- | --- |
| slab-1d-dielectric-normal | same_error | 0.19 | same_error | 0.24 | round-off floor |  |
| slab-1d-dielectric-oblique | same_error | 0.16 | intermediate | 0.37 | round-off floor |  |
| slab-1d-dielectric-conical | same_error | 0.14 | same_error | 0.049 | round-off floor | dt exempt (conical, ratio 0.8165) |
| slab-1d-magnetic-normal | same_error | 0.17 | same_error | 0.21 | round-off floor |  |
| slab-1d-magnetic-oblique | same_error | 0.087 | intermediate | 0.35 | round-off floor |  |
| slab-1d-magnetic-conical | same_error | 0.072 | same_error | 0.045 | round-off floor | dt exempt (conical, ratio 0.8165) |
| slab-1d-drude-normal | same_error | 0.27 | intermediate | 0.35 | round-off floor |  |
| slab-1d-drude-oblique | same_error | 0.24 | intermediate | 0.58 | round-off floor |  |
| slab-1d-drude-conical | same_error | 0.2 | same_error | 0.057 | round-off floor | dt exempt (conical, ratio 0.8165) |
| slab-crossed-dielectric-normal | same_error | 0.1 | same_error | 0.13 | round-off floor |  |
| slab-crossed-dielectric-oblique | same_error | 0.088 | same_error | 0.067 | round-off floor |  |
| slab-crossed-dielectric-conical | same_error | 0.088 | same_error | 0.067 | round-off floor |  |
| slab-crossed-magnetic-normal | same_error | 0.052 | same_error | 0.12 | round-off floor |  |
| slab-crossed-magnetic-oblique | same_error | 0.047 | same_error | 0.057 | round-off floor |  |
| slab-crossed-magnetic-conical | same_error | 0.047 | same_error | 0.057 | round-off floor |  |
| slab-crossed-drude-normal | same_error | 0.15 | same_error | 0.19 | round-off floor |  |
| slab-crossed-drude-oblique | same_error | 0.13 | same_error | 0.094 | round-off floor |  |
| slab-crossed-drude-conical | same_error | 0.13 | same_error | 0.093 | round-off floor |  |
| square_18deg-r16.67 | same_error | 0.09 | unresolved | 1.7 | fragile; settling |  |
| square_18deg-r25 | unresolved | 0.032 | unresolved | 0.12 | settling; settling |  |
| square_18deg-r33.33 | unresolved | 0.07 | unresolved | 0.079 | settling; settling |  |
| rect_28deg-r20 | unresolved | 0.11 | unresolved | 0.081 | settling; settling |  |
| rect_28deg-r30 | unresolved | 0.16 | unresolved | 0.27 | settling; settling |  |
| vacuum-r25-pad1 | same_error | 0.12 | one_route_near_exact | 19 | round-off floor | smaller route flips with pad (power) |
| vacuum-r25-pad1.5 | same_error | 0.12 | route_changes_answer | 1.5 | round-off floor | smaller route flips with pad (power) |
| vacuum-r50-pad1 | same_error | 0.08 | route_changes_answer | 9.6 | round-off floor | smaller route flips with pad (amplitude, power) |
| vacuum-r50-pad1.5 | same_error | 0.12 | one_route_near_exact | 138 | round-off floor | smaller route flips with pad (amplitude, power) |
| material-magnetic-r20 | same_error | 0.13 | same_error | 0.054 | below 10 u; ranks common |  |
| material-drude-r20 | same_error | 0.18 | same_error | 0.027 | below 10 u; ranks common |  |
| material-lorentz-r20 | same_error | 0.11 | unresolved | 0.93 | below 10 u; settling | dt exempt (pole_cfl, ratio 0.9829) |
| material-lorentz-r40 | intermediate | 0.44 | both_below | 84 | below 10 u; below 10 u |  |
| material-magnetic-r20-pml16 | unresolved | 0.00047 | unresolved | 0.001 | settling; settling |  |
| material-drude-r20-pml16 | both_below | 0.24 | unresolved | 0.00012 | below 10 u; settling |  |
| material-lorentz-r20-pml16 | same_error | 0.007 | same_error | 0.0046 | below 10 u; below 10 u | dt exempt (pole_cfl, ratio 0.9829) |
| material-lorentz-r40-pml16 | both_below | 0.012 | unresolved | 0.0035 | below 10 u; settling |  |
| metal-r30 | both_below | 0.017 | same_error | 0.013 | below 10 u; below 10 u |  |
| metal-r60 | both_below | 0.0089 | both_below | 0.009 | below 10 u; below 10 u |  |
| slab_uniform-r50 | same_error | 0.0096 | route_changes_answer | 2.1 | round-off floor |  |
| binary_grating-r50 | same_error | 0.072 | same_error | 0.23 | fragile; ranks common |  |
| binary_grating-r100 | same_error | 0.047 | same_error | 0.17 | below 10 u; below 10 u |  |

## 8. Why no route ranks

`verdict.engines.ranking` is empty on all 40 scenes.

**23 scenes withhold by rule, not by measurement**: the 18 slabs, the 4 vacuum scenes and
`slab_uniform-r50` have a round-off-floor `u` on both metrics, where any `e1 != e2` would clear
10 u. These are not measured ties; the study cannot rank there by construction.

The other 17 withhold for these reasons:

| Amplitude | Power | Scenes |
| --- | --- | ---: |
| settling | settling | 5 |
| below 10 u | below 10 u | 5 |
| below 10 u | settling | 3 |
| below 10 u | ranks the shared engine | 2 |
| fragile | settling | 1 |
| fragile | ranks the shared engine | 1 |

On three scenes power alone ranked the shared engine more accurate, and amplitude was withheld,
so the scene-level rule dropped the ranking. Margins from `ranking_margin`, against the power
settling bound (`summary.power.settling`) and, where a twin exists, the twin's smaller worst
power error (`families`):

| Scene | Relative gap | Gap / 10 u | Gap / settling bound | Gap in the twin | Amplitude withheld because |
| --- | ---: | ---: | ---: | --- | --- |
| material-magnetic-r20 | 5.1% | 3.2 | 1.2 (proxy) | 1.9e-6, unresolved | below 10 u |
| material-drude-r20 | 2.6% | 1.2 | 1.5 (proxy) | 1.4e-6, unresolved | below 10 u |
| binary_grating-r50 | 1.2% | 4.7 | 7.4 (proxy) | no twin | fragile |

On the two material scenes the ranking sits just above a proxy bound, and with the termination
strengthened the two worst power errors come within 1.9e-6 and 1.4e-6 of each other, below the
twins' 10 u and settling bounds; the twins' power verdict is `unresolved_against_settling`. Those
two rankings do not survive their twins. `binary_grating-r50`'s, which clears its settling bound
7.4 times and has no twin, is dropped only by the scene-level rule.

## 9. Where the engines disagree

**Vacuum: the smaller-error route follows the padding.** On power, legacy has the smaller
error at `pad_scale` 1 and the shared engine at 1.5, at both resolutions, so all four scenes are
listed under the flip (`families.vacuum.smaller_route_flips_with_pad`); on amplitude the same flip
appears at resolution 50. Two are `route_changes_answer` and two `one_route_near_exact`, whose
`d/min(e)` of 19 and 138 is larger still. The record files these under `pml_structure_gap`, not
under an engine: legacy's clearance is larger above than below and the shared engine's is not,
and `--common-pad legacy` sizes the shared engine's one pad, used on both sides, to legacy's
upper gap.

| Scene | Legacy gap, top / bottom (cells) | Shared gap, top / bottom (cells) | Power case |
| --- | --- | --- | --- |
| vacuum-r25-pad1 | 18 / 10 | 18 / 18 | `one_route_near_exact` |
| vacuum-r25-pad1.5 | 25 / 16 | 26 / 26 | `route_changes_answer` |
| vacuum-r50-pad1 | 32 / 20 | 32 / 32 | `route_changes_answer` |
| vacuum-r50-pad1.5 | 48 / 30 | 48 / 48 | `one_route_near_exact` |

Every vacuum ratio is taken against a round-off-floor `u`, where one engine landing near exact
sets `d/min(e)`.

**Four 1D slabs are `intermediate` on power** (`d/min(e)` 0.35-0.58): dielectric, magnetic and
Drude at oblique incidence, where the power `d` sits on `p.T(0,0)`, legacy's TM loop; and Drude
at normal incidence, where it sits on `s.T(0,0)`, legacy's TE loop. Their amplitude is
`same_error`. `tm_half_cell_sampling` applies to all four but acts only on the TM loop, and it
applies equally to the dielectric and magnetic slabs at normal incidence, which are `same_error`
on power. Nothing here was measured; their confounds are listed per scene.

**`slab_uniform-r50` is `route_changes_answer` on power, and nothing attributes it.** It has no
pad-flip flag only because its family has one variant. The confounds that apply to it are
`source_normalisation`, `deembedding_phase`, `tm_half_cell_sampling`, `pml_backing`,
`pml_structure_gap` (legacy 36 / 16 cells, shared 36 / 36) and `pulse_delay_and_carrier`. Its
power `d` sits on `tm.T(0,0)`: `d` 3.6e-4 against `e1` 1.69e-4 and `e2` 1.89e-4, so `d` is about
`e1 + e2`; on `te.T(0,0)` `d` is 5.0e-5. At normal incidence on a uniform slab TE and TM are the
same physics, and one observation separates the engines there:

| | Legacy | Shared |
| --- | ---: | ---: |
| `te.T(0,0)` error | 1.387e-4 | 1.886e-4 |
| `tm.T(0,0)` error | 1.690e-4 | 1.886e-4 (equal to TE within 1e-14) |
| Energy defect, TE leg | -1.284e-3 | -1.194e-3 |
| Energy defect, TM leg | -1.592e-3 | -1.194e-3 |

The shared engine gives TE and TM the same answer; legacy does not. Legacy's TE leg lands 5.0e-5
from the shared engine and its TM leg 3.6e-4, on the other side of the reference, where the
shared engine's error is the larger; each leg is its own time loop.
`tm_half_cell_sampling` (legacy samples Hy half a cell off the plane the shared engine samples E
on) is one candidate among the TM leg's differences, and no run tested any of them. It is an
observation, not an attribution.

`material-lorentz-r40` is `intermediate` on amplitude (`d/min(e)` 0.44), on the grazing
`s.t_s(0,1)` channel; section 10 covers it.

## 10. Grazing orders and the termination

An order that leaves at a small exit-medium cos(theta) is partly sent back by the termination
(CPML, backing, gap), and the two engines lay that out differently. The study flags a channel
`grazing` below `GRAZING_COS` = 0.3 (`studies/fdtd/route_parity/fdtd_route_parity.py::GRAZING_COS`). On the
material scenes the transmitted (0, 1) order leaves at 0.263 and the reflected (0, -1) at 0.242.
Each material scene has a `-pml16` twin with CPML target 1e-16, 24 cells and 640 periods
(`studies/fdtd/route_parity/fdtd_route_parity.py::PML16`), and `termination_control` compares the pair channel by
channel (`studies/fdtd/route_parity/fdtd_route_parity.py::termination_control`). The twin changes target, depth
and run length at once, so a move also holds whatever the base run had not settled. All four
twins matched their inputs.

| Scene | Worst grazing `d`, base | Worst grazing `d`, twin | Twin settling bound (proxy) | Twin `d` resolved | Worst other `d`, base / twin | Verdict base (amplitude / power) | Verdict twin |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| material-magnetic-r20 | 2.53e-3 | 1.14e-5 | 1.23e-5 | no | 1.18e-3 / 1.00e-5 | same_error / same_error | unresolved / unresolved |
| material-drude-r20 | 3.05e-4 | 5.94e-6 | 6.28e-6 | no | 7.44e-4 / 9.49e-6 | same_error / same_error | both_below / unresolved |
| material-lorentz-r20 | 2.76e-3 | 1.88e-4 | 2.16e-5 | yes | 1.16e-3 / 9.28e-5 | same_error / unresolved | same_error / same_error |
| material-lorentz-r40 | 3.26e-3 | 2.15e-5 | 1.50e-5 | yes | 3.82e-4 / 7.19e-6 | intermediate / both_below | both_below / unresolved |

- On magnetic-r20 and both Lorentz scenes the base amplitude `d` sat on the grazing
  `s.t_s(0,1)` channel, and the worst grazing `d` fell 221x (magnetic), 15x (lorentz-r20) and
  152x (lorentz-r40) with target, depth and run length strengthened together. Most of it was
  therefore the terminations or what the shorter base runs had not settled, not the engines'
  interiors; the twins do not separate those two. On magnetic-r20 and drude-r20 the twin's
  grazing `d` is below the twin's settling bound.
- A resolved grazing remainder stays on both Lorentz scenes: 1.9e-4 on `p.t_p(0,1)` of
  `material-lorentz-r20-pml16`, whose twin verdict is `same_error` on both metrics but whose `dt`
  is exempt (pole CFL, ratio 0.9829), and 2.1e-5 on the reflected `s.r_p(0,-1)` of
  `material-lorentz-r40-pml16`, whose `dt` is matched. That twin's worst (0, 1) `d`, 1.0e-5, is
  below its settling bound.
- Drude-r20's base amplitude `d` sat on the specular `s.r_s(0,0)` channel. The worst
  non-grazing `d` fell 12x to 118x on all four scenes; the twin's value is resolved on drude-r20
  and lorentz-r20 and below the twin's settling bound on magnetic-r20 and lorentz-r40. A
  non-grazing channel is not free of the change either.

[CPML strength](fdtd-pml-strength.md) records a related effect on a different scene, a
near-Rayleigh binary grating on the shared engine alone: width alone does not reduce the glancing
return, strength alone does but raises the order-0 return, and its strength-only and combined
800-period runs fail its scattering-settling check.

## 11. What this record does not decide

- **The shared engine at its own defaults.** Only the matched arm ran (`settings.arms`), and no
  scene has a `common_verbatim` entry. The verbatim arm would not measure "add `backend='numpy'`" either:
  `studies/fdtd/route_parity/fdtd_route_parity.py::verbatim_common` runs every scene at `courant` 0.99, not 0.8
  (21 scenes by its own default, 19 because the study passes it), and `--common-pad default`
  changes only the matched arm. The record's own `scope` says that the shared engine at its own
  defaults is not settled.
- **The source injection.** `source_normalisation` (TFSF plane against current sheets
  calibrated by a reference run) has no arm and no `measured_in`; it is folded into every `d`, as
  `pulse_delay_and_carrier` is. `transverse_demodulation`, `subpixel_order` and
  `tm_half_cell_sampling` are folded into every `d` of the scenes they apply to (34, 17 and 9).
- **The backing and the gap.** No arm varies `pml_backing` or `pml_structure_gap`. The twins
  change the CPML's target and depth and the run length, not the backing or the gap, although
  the record labels both confounds `measured_in` `termination_control` on the 8 material scenes.
  On vacuum the gap shows up only as a pad flip.
- **Rankings on round-off-floor scenes, and `d` below settling**, by construction.
- **`convergence_window` inertness.** `--check-inert` was not run, so the diagnostic passed to
  the shared engine is not shown bit-inert in this record.
- **Out of scope:** graded meshes (`matched_common` refuses them), bottom incidence, shape
  recipes, JAX and Torch (the shared engine ran on NumPy only), gradients, GPU, legacy-only
  diagnostics, and cost.
- **The current package** (section 3).

It therefore does not license changing `default_route` from `legacy_numpy`, and it does not
license removing either engine. Where the record could resolve the pair, the two engines mostly
made the same error; where it could not, or where they disagreed, it does not say which engine
is right.

## 12. Confounds

Listed per scene under `confounds`; applicability counts over the 40 scenes.

| Confound | Scenes | What differs | Measured |
| --- | ---: | --- | --- |
| `source_normalisation` | 40 | TFSF plane with a 1D incident grid, against current sheets calibrated by a uniform-medium reference run | no |
| `deembedding_phase` | 40 | continuum against Yee `kz` | removed on the node planes by the `legacy_yee_deembedded` arm; a half-cell residual stays on non-conical TM loops (section 6) |
| `pml_backing` | 40 | grounded end rows against a periodic wrap | not varied; labelled `termination_control` on 8 scenes |
| `pml_structure_gap` | 40 | clearance larger above than below, against equal to within a cell | not varied; labelled `termination_control` on 8 scenes; a pad flip on vacuum |
| `pulse_delay_and_carrier` | 40 | peak at 4 against 5 widths; real sine against complex carrier; post-peak time equalised | no |
| `transverse_demodulation` | 34 | per-component x against face interpolation divided by cos(kx dx/2) | no |
| `subpixel_order` | 17 | same averaging rule, different order at tooth corners | no |
| `tm_half_cell_sampling` | 9 | legacy's 2D TM stepper samples Hy half a cell off the shared engine's E plane | no |
| `conical_dt_and_dy` | 3 | continuum d/dy against a two-cell Bloch difference; `dt` ratio 0.8165 | no |
| `pole_cfl_dt` | 2 | the shared engine's pole-aware CFL bound refused legacy's `dt`; ratio 0.9829 | no |
