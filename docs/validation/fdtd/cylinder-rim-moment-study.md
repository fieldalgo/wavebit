# Cylinder rim moment: exact and split prototypes (study)

Written: 2026-11-10. Second round (the cap blend): 2026-11-11.

**This is a study, not an acceptance.** Nothing here changes the default. The prototype is
behind a private, import-time switch, `WAVEBIT_STUDY_CYLINDER_MOMENT`, which must be set
before `wavebit` is imported. The default is `plane`, which is bit-identical to `main` at
7e48d76b. No test expectation, tolerance, hash or golden was changed. The tests and pins
that would move are listed below with their magnitudes.

Targets: `src/wavebit/fdtd/_shapes/_bodies.py` (`Cylinder._split_tensor`,
`Cylinder._exact_tensor`, `exact_moment`, `study_tensor`),
`src/wavebit/fdtd/_prepare/constitutive.py` (`lower`), `studies/fdtd/cylinder_rim_moment/`.

The first round (2026-11-10, below) built `rim`, `exact` and `split`, found that `split`
bounds the rim gradient spike, and found one defect shared by all three: a point outlier
when a cap lies exactly on a voxel face. The owner's decision was to fix the defect first
and measure again. This second round does that for `split` only.

## Second round: TL;DR

* **The tie outlier is fixed.** `split` now gives each cap a smooth share of the voxel, ramped
  inside the voxel over 0.02 dx. `test_height_continuity_includes_the_exact_ties[0.0-cylinder]`
  reads max/median 1.00 (gate 50; first round 75), the same as `plane`.
* **The gradient stays bounded.** End to end, |dL/dR| <= 168 and |dL/dH| <= 6.1 for every delta
  from +/-1e-3 to 1e-16 and at 0, against `plane`'s 1.7e12. AD matches central differences to
  2.5e-7. Crossing the ramp moves |dL/dH| to at most 6.95 (`plane` over the same sweep: 7.08).
* **The rim moment stays accurate.** Over 32 cap offsets, the rim rms error of w is 9.1e-3 /
  3.9e-3 / 1.7e-3 at R/h 4 / 8 / 16 (order ~1.2), against `plane`'s 0.18 / 0.14 / 0.10
  (order ~0.4).
* **Observables still do not improve.** On the C7 ladder `split` agrees with the first
  round's `split` to 3e-12 (no C7 cap is near a face). Against `plane` it moves the errors
  against RCWA by -4% to +10%, with the same orders to 0.02.
* **It is cheaper than the first round.** Geometry costs 1.4-1.5x `plane` (was 2.0-2.6x), a
  repeated forward plus gradient 1.76x (was 1.9x), and peak RSS is +13%.
* **Eight tests and the pillar example's assert fail under `split`.** All are pins of the
  plane's own numbers or frozen archives (list below). There is one new one: the pin that
  the plane's dead-voxel fill gap stays above 1e-2. `split` closes that gap to 3.3e-6.
* **Recommendation: land `split` with re-pins**, after the full suite runs under it. The
  reasons and the exact re-pin list are under "Second round: recommendation".

## Second round: the blend

`Cylinder._split_tensor`. In every voxel, the moment is the side wall's own tangent plane
(radial normal, its distance) times the axial overlap, plus each cap's clipped disc area
times a share `S(t)`. Here `t = |cap - s| - dx/2` is how far the cap plane lies outside the
voxel's axial span (negative inside).

* **The share.** `S = y^3 (10 - 15 y + 6 y^2)`, with `y = clip(-t / eps, 0, 1)` and
  `eps = 0.02 dx`. This is the C2 smootherstep, ramped *inside* the voxel:
  * exactly 1 once the cap is `eps` deep;
  * exactly 0 on the face and outside it;
  * first and second derivatives zero at both ends of the ramp.

  The first round used `plane_share`'s step instead: 1 inside, 1/2 on the face, 0 outside.
* **Why the ramp is inside.** The outlier is at a junction. A pillar stands on a substrate,
  and the cap moves through a grid plane together with the substrate face. Above the tie the
  junction rule owns the voxel, and it writes values that do not count the buried cap. At and
  below the tie the body's moment is used. `plane` gives a cap on the face zero area there,
  because the nearest-surface plane at |d| = dx/2 misses the voxel. Any rule that counts that
  cap jumps when the junction takes over. A centred ramp, 1/2 on the face with area conserved
  between neighbours, was built and measured: 75.2 (tie 0) and 48.9 (tie 0.5) against the gate
  of 50, so the outlier stays. The inside ramp matches the plane's one-sided limit at the tie.
  The price is that a cap within `eps` of a face is under-counted, and a cap on the face is
  counted nowhere, as the plane counts it.
* **Why it brings back no 1/delta and no 1/sqrt.** `|dS/dt| <= 1.875 / eps = 94 / dx`. That is a
  fixed number per cell, whatever the distance of the cap from the face. No normal is
  selected and no arc is clipped, so there is no 1/delta from a normal turning and no 1/sqrt
  from arc tangency. Measured end to end (`blend_gradient.py`), the largest |dL/dH| while a cap
  crosses the ramp is:

  | eps | 0.25 | 0.1 | 0.05 | 0.02 (C1) | 0.02 (C2) | `plane` |
  |---|---|---|---|---|---|---|
  | max \|dL/dH\| | 9.14 | 7.67 | 7.07 | 6.88 | 6.95 | 7.08 |

  AD matches central differences inside the ramp to 4e-8. The narrow ramp costs no
  measurable gradient, because the voxels it touches carry little contrast.
* **Why 0.02 dx.** The ramp under-counts caps near a face. The rim rms error over 32 cap
  offsets (`split_error.py offsets 32`) at R/h 4 / 8 / 16, with the inside C1 ramp, is:

  | eps | R/h 4 | R/h 8 | R/h 16 |
  |---|---|---|---|
  | 0.25 | 1.2e-1 | 9.4e-2 | 6.9e-2 |
  | 0.1 | 6.4e-2 | 4.8e-2 | 3.5e-2 |
  | 0.05 | 3.4e-2 | 2.6e-2 | 1.9e-2 |
  | 0.02 | 9.5e-3 | 4.4e-3 | 2.2e-3 |
  | 1e-6 (a hard step) | 8.9e-3 | 3.7e-3 | 1.4e-3 |

  At 0.25 the error is back at the plane's order (~0.4). At 0.02 it is close to the hard cap.
  The tie gate passes at every eps tried (0.25 and 0.02 measured), because the junction jump
  is spread over 20 or more of the gate's 1e-3 dx steps.
* **Why C2, not C1.** `test_fd_cylinder_grad[y]` puts a cap exactly `eps dx` inside a voxel,
  the end of the ramp, where a C1 smoothstep's second derivative jumps. There its h = 3e-6
  central difference is off by O(h): 7.2e-6 relative against 2e-6. AD is right; the
  difference converges on it as h falls (`records/blend_cylinder_grad_step.txt`). With C2 the
  same difference is 4.3e-8. C1 remains available as `WAVEBIT_STUDY_CYLINDER_CAP_STEP=c1`.
* **The hand-over to `plane`.** The first round switched between `plane` and the split moment
  where `caps > 0`, a second hard switch at the tie. There is now no switch. Beyond
  (sqrt 2 - 1)/2 dx from a cap plane, the side wall's own plane *is* the nearest-surface
  plane:
  * where that plane's normal is radial, the two have the same `d` and `n`;
  * where it is axial or a corner normal, it lies more than (sqrt 2)/2 dx from the voxel
    centre, so its area is 0, and the side wall is farther still.

  So `split` uses its own moment everywhere and never evaluates the nearest-surface plane.
  Against the plane, over 140,871 random voxels beyond that reach: 5.2e-16 dx^2
  (`blend_check.py switch`). The same identity is what removed the cost of the second plane
  walk.

All of it is `xp.where`, `xp.minimum` and `xp.maximum` on `array_namespace`. It runs on NumPy,
JAX and Torch and is differentiable in radius, height and centre. JAX and Torch agree to
<= 5e-13 in `blend_check.py ad` and to <= 1.6e-6 relative in the spike tables (the
largest on the w2 lattice, where gradients reach 2e5). Torch runs the pillar example, and JAX the end-to-end spike and the C7 M1/M2
rungs at 80 and 100 cells per wavelength.

## Second round: (a) spike

Same scripts and deltas as the first round (`spike.py`, `records/blend_c2_spike_*.txt`).
Max over delta = +/-1e-3 ... 1e-16 and 0:

| level | `plane` | first-round `split` | `split` (blend) | normal scale |
|---|---|---|---|---|
| voxel, w2: \|dL/dR\| | 2.95e15 | 0.90 | 0.90 | about 1 |
| voxel, mix: \|dL/dR\| / \|dL/dH\| | 3.27e16 / 1.09e16 | 214 / 102 | 214 / 102 | |
| lattice, w2: \|dL/dR\| / \|dL/dH\| | 1.24e16 / 4.65e15 | 5.2e8 / 8.7e7 | 9.9e3 / 2.4e5 | |
| lattice, mix: \|dL/dR\| / \|dL/dH\| | 1.33e17 / 4.99e16 | 4.90e6 / 3.30e6 | 4.84e6 / 3.24e6 | about 4-5e6 |
| fdtd end to end: \|dL/dR\| / \|dL/dH\| | 1.68e12 / 4.21e11 | 168 / 5.83 | 168 / 6.06 | about 157 / 5.7 |

* **AD vs central difference** (|delta| >= 1e-7 for fdtd, >= 1e-9 elsewhere): <= 2.5e-7 end to
  end, <= 2.0e-6 on the lattice and <= 2e-5 on one voxel.
* **Loss continuity at alignment.** The end-to-end loss steps by 2.35e-3, out of 1069, between
  delta = 0 and -1e-16 (first round 2.3e-3; `plane` 5.7e-4). That is not the cap. At delta <= 0
  the radius sits exactly on a voxel face, where the side wall's tangent plane lies on the face
  and its area steps from 0 to a full face. `plane` has the same step along the whole side wall
  (`blend_check.py sweep`: radius through face tangency, |d(f(1-f) w)| 8.4e-2 for `split` and
  4.1e-2 to 1.8e-1 for `plane`). The cap blend does not touch it.
* **Cap height through a face.** f(1-f) w, every Yee staggering, steps of 1e-4 dx: worst step
  8.0e-4 against a median of 1.0e-4, and no jump in slope beyond 10 per unit height (`plane`:
  9.2e-2 and 9.2e3). The jump lies in `plane`, not in `split`.
* **Lattice w2.** Its dL/dH of 2.4e5 (C1: 5.7e6) comes from voxels with fill ~1e-12. There,
  w swings as the cap share's area, ~(delta/eps)^2, meets a sliver of side wall. The mixing
  rule weights w by f(1-f), so these voxels are inert. The w2 loss is the only one that sees
  them.

The junction tie gate (`records/blend_tie_gate_c2.txt`), max jump over median:

| | tie 0 | tie 0.5 |
|---|---|---|
| `plane` | 1.00 | 1.43 |
| first-round `split` | 75 | (passed) |
| centred ramp, eps 0.25 | 75.2 | 48.9 |
| `split` (blend) | 1.00 | 1.43 |

## Second round: (b) what moves, and what fails

**Fill never changes.** w changes in rim voxels and, beyond the first round, in side-wall
voxels just past a cap plane:
* There the plane said "no surface" and the side wall still clips a corner. These are the
  plane's dead voxels.
* Changed fractional voxels: 136-858 per scene, rms |dw| 0.24-0.44, max up to 1.0, the last
  in sliver-fill voxels (`records/blend_wchange.txt`).
* The constitutive node tensor moves exactly as in the first round, by max 0.49-0.55.

**Pillar example** (Torch): dT/d(R, H, eps) = (0.18543, -0.08235, 0.00619), the same as the first
round. dT/dR moves by -4.1%.

**Tests.** 1475 tests ran in the 36 cylinder-touching files of `tests/fdtd` and `tests/api`,
with `-m "not research"` and `-n 2`, under `split` (C2, eps 0.02;
`records/tests_split_blend_c2.txt`). Result: 8 failed, 1459 passed, 8 skipped. All 8 pass
under `plane`.

| test / pin | tolerance | `split` | what it is |
|---|---|---|---|
| `test_fd_junction_physics::test_public_solver_reproduces_the_measured_study[0]` | atol 1e-10 | 8.39e-4 (rel 4.8%) | archive `tests/references/fd_junction_study_rows.json` |
| `...::test_public_solver_reproduces_the_measured_study[1]` | atol 1e-10 | 8.41e-4 (rel 3.5%) | same |
| `test_fd_metasurface::...[M1-20-p]` | atol 1e-10 | 1.33e-4 | archive `tests/references/fd_metasurface_current_rows.json` |
| `test_fd_metasurface::...[M2-25-s]` | atol 1e-10 | 7.65e-4 | same |
| `test_fd_shapes::test_the_cap_rims_steep_gradient_is_the_planes_own_derivative` | slope < 2e-3 | 0.327 | pins the spike itself; rewrite as a bounded-gradient test |
| `test_fd_shapes::test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs` | rim gap > 0.3 | 0.0186 | pins the plane's rim error |
| `test_fd_shapes::test_a_crease_is_where_w_jumps_and_this_is_how_far` | cyl rim jump > 0.9 | 0.412 | pins the plane's crease |
| `test_fd_shapes::test_w_vanishes_only_where_the_fill_is_already_decided` | cylinder gap at R/h 16 > 1e-2 | 3.26e-6 | **new**: pins the plane's dead-voxel fill gap, which `split` closes |
| `examples/fdtd/pillar_gradient.py` (examples smoke job) | atol 5e-5 | dT/dR off by 7.87e-3 | the example's own assert |

* **Gone since the first round:** `height_continuity[0.0-cylinder]` (passes, see above).
* **Not new:** `test_public_spelling`; it failed in the first round only because this note did
  not yet exist.
* **C1 instead of C2:** the same eight fail, plus `test_fd_cylinder_grad[jax-y]` and
  `[torch-y]` (7.2e-6 against 2e-6, explained above; `records/tests_split_blend_c1.txt`).
* **The four archive rows** are the first round's to the digit. No cap in those scenes lies
  near a face, so the blend changes nothing there.

## Second round: (c) accuracy

**Per-voxel moment against the exact tensor** (`split_error.py`). Rim voxels, worst / rms of
max_c |w_c - w_c(exact)|:

| | R/h 4 | R/h 8 | R/h 16 | rms order |
|---|---|---|---|---|
| `plane`, 32 cap offsets | 0.66 / 0.181 | 0.67 / 0.138 | 0.68 / 0.102 | ~0.4 |
| `split` (blend), 32 cap offsets | 0.10 / 9.1e-3 | 0.082 / 3.9e-3 | 0.056 / 1.7e-3 | ~1.2 |
| `split` (blend), the first round's fixed geometry | 0.059 / 8.5e-3 | 0.040 / 3.6e-3 | 0.037 / 1.35e-3 | ~1.3 |
| first-round `split`, fixed geometry | 0.059 / 8.5e-3 | 0.040 / 3.6e-3 | 0.037 / 1.35e-3 | ~1.3 |

Away from the rim `split` and `plane` agree, at rms 1.4e-2 / 7.0e-3 / 3.5e-3. The first round's
fixed geometry puts every cap 0.043 h from a face, outside a 0.02 ramp, so those rows are the
same as the first round's. The 32-offset rows are what a ramp is judged on.

**Observables against the stored RCWA ladder (C7)**, using the C7 harness unchanged. The
`plane` rows are the stored C7 rows; this study's `plane` reruns reproduce them.

| | rungs | `plane` A | `split` A | `plane` F | `split` F |
|---|---|---|---|---|---|
| M1 s | 20 / 40 / 80 | 2.925e-2 / 7.403e-3 / 2.095e-3 | 2.938e-2 / 7.425e-3 / 2.103e-3 | 1.739e-2 / 4.581e-3 / 1.363e-3 | 1.717e-2 / 4.458e-3 / 1.315e-3 |
| M1 s order | | -1.902 | -1.902 | -1.837 | -1.854 |
| M1 p | 20 / 40 / 80 | 2.775e-2 / 6.603e-3 / 1.811e-3 | 2.793e-2 / 6.721e-3 / 1.853e-3 | 1.363e-2 / 3.495e-3 / 9.060e-4 | 1.364e-2 / 3.476e-3 / 9.080e-4 |
| M1 p order | | -1.969 | -1.957 | -1.956 | -1.954 |
| M2 s | 25 / 50 / 100 | 7.408e-2 / 2.018e-2 / 6.082e-3 | 7.365e-2 / 2.008e-2 / 6.057e-3 | 1.207e-2 / 2.048e-3 / 3.870e-4 | 1.241e-2 / 2.123e-3 / 4.056e-4 |
| M2 s order | | -1.803 | -1.802 | -2.482 | -2.468 |
| M2 p | 25 / 50 | 3.478e-2 / 8.577e-3 | 3.459e-2 / 8.411e-3 | 1.802e-2 / 4.484e-3 | 1.958e-2 / 4.934e-3 |

* M1 p at 80 and M2 at 50 and 100 are new in this round.
* Every M1 s, M1 p and M2 s ladder under `split` is consistent with RCWA (C7's criterion 4),
  as under `plane`. M2 p @100 was not run, so M2 p has no order here. C7 marks the stored
  `plane` M2 p ladder `consistent=False`.
* The orders are unchanged to 0.02.
* Errors in A move by -2% to +2.3%. Errors in F move by -4% to +10% (M2 p @50); `split` is
  worse there, and better in M1 s F.
* `split` agrees with the first round's `split` to 3e-12 on every rung both ran
  (`records/metasurface_split-hard.jsonl`).

The 60x better rim moment still does not reach these observables.

## Second round: (d) cost

`cost.py` (`records/blend_cost.txt`), one process per mode, 1 thread, machine otherwise idle,
best of 3 runs of best-of-5:

| | `plane` | `split` (blend) | first-round `split` |
|---|---|---|---|
| occupancy, NumPy | 0.056 s | 0.082 s (1.5x) | 2.6x |
| occupancy, JAX | 0.237 s | 0.342 s (1.4x) | 2.4x |
| occupancy, Torch | 0.067 s | 0.100 s (1.5x) | 2.0x |
| occupancy peak RSS | 465 MB | 467 MB | +3% |
| plan | 0.23 s | 0.28 s | |
| first value_and_grad (with compile) | 12.4 s | 14.4 s | |
| repeat value_and_grad | 1.26 s | 2.21 s (1.76x) | 1.9x |
| peak RSS, fdtd | 656 MB | 739 MB (+13%) | +17% |

The saving comes from three things:
* `split` no longer evaluates the exact arc;
* it no longer walks the nearest-surface plane (the identity above);
* it no longer selects between two moments.

What remains is the clipped disc for the caps, computed once for the fill and once for the
moment, and the constitutive tensor, computed a second time in `lower`. Sharing those would be
the next saving. It was not done because it crosses the occupancy/constitutive boundary.

## Second round: recommendation

**Land `split` (inside ramp, C2, eps = 0.02 dx) as the smoothed cylinder's moment, with the
re-pins below. Do it only after the full `tests/fdtd` and `tests/api` suites and the
research-marked tests have run under it.**

* **The defect is fixed, and nothing replaces it.** The first round's main reason to keep
  `plane` was that `split` swapped a pinned spike for a new outlier. The outlier is gone, the
  tie gate reads what `plane` reads, and no 1/delta or 1/sqrt slope came back.
* **The aligned rim is not measure-zero in practice.** Round-number designs land on it:
  `R = 0.3` and `H = 0.6` on `dx = 0.1` is exactly delta = 0. There `plane`'s end-to-end
  dL/dR is 4.05e6 against a true ~157, and one ulp away it is 1.7e12. An optimiser that starts
  at round numbers gets a gradient off by 4 to 10 orders of magnitude. `split` gives 157.
* **It also removes a second known plane defect,** the cylinder's dead-voxel fill gap (2.5e-2
  -> 3.3e-6, `test_w_vanishes...`).
* **The cost is moderate:** 1.4-1.5x geometry, 1.76x forward plus gradient, +13% memory.
* **Against landing:** observables do not improve (C7 errors move by -4% to +10%, same
  orders). The ramp
  adds a constant, 0.02 dx, below which a cap near a face is under-counted. Six pins and two
  archive rows have to move. If gradients at round-number designs are not a concern, keep
  `plane`. The re-pin cost is the same later.

**Landing would re-pin exactly:**

1. **Archives:**
   * `tests/references/fd_junction_study_rows.json`: rows 0 and 1 (provenance
     cc2ccf8761b414ba), max change 8.4e-4. The other six rows did not move.
   * `tests/references/fd_metasurface_current_rows.json`: M1-20-p (1.3e-4) and M2-25-s
     (7.6e-4).
2. **`tests/fdtd/test_fd_shapes.py`, tests that pin the plane's numbers:**
   * rewrite `test_the_cap_rims_steep_gradient_is_the_planes_own_derivative` as a
     bounded-gradient test (slope 0.327 against a 1/delta ridge);
   * re-measure `test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs`
     (rim gap 0.0186);
   * re-measure `test_a_crease_is_where_w_jumps_and_this_is_how_far` (cylinder rim jump 0.412);
   * re-measure `test_w_vanishes_only_where_the_fill_is_already_decided` (cylinder row of its
     docstring table, and the `> 1e-2` pin, now 3.3e-6).
3. **`examples/fdtd/pillar_gradient.py`:** the pinned gradient
   (.1933, -.0818, .00625) -> (.1854, -.0824, .00619).
4. **Docs that call the plane the cylinder's moment:** the `_bodies.py` module table and the
   d117e9bb text, and the C2/C5 notes these tests cite.
5. **The switch itself:** delete `rim`, `exact`, `centred` and `c1`, and make eps a named
   constant, not an environment variable.

## Second round: versions

Python 3.12.3, numpy 2.5.4, scipy 1.18.1, jax and jaxlib 0.12.0.1, torch 2.14.1+cpu,
contourpy 1.4.0, pytest 9.1.1, pytest-xdist 3.8.0. Torch was installed by
`tools/ci_constraints.py install tests`, and the rest by `uv pip install` with
`.github/constraints/tests.txt`. All versions match the file. 4 CPUs and 15 GB.

## Second round: not measured

* **The full suites** (`tests/fdtd`, `tests/api`, about 9,100 tests beyond the 36 files) and
  the research-marked tests, under any variant.
* **C7 M2 p @100.** It was stopped at a 2-hour limit while sharing the machine with the test
  run, so M2 p has no order.
* **The full test run and the C7 ladder used earlier forms of the code.** Both used the code
  before the plane walk was dropped, which agrees with the final code to 5.2e-16 dx^2. The C7
  ladder ran with the C1 ramp. No C7 cap lies within a ramp: the rows match the hard cap to
  3e-12, so C1 and C2 give the same rows. The final code was run on the pillar example, the
  cost table, `blend_check.py switch` and the default `plane` checks: the quick verification
  profile, 17 passed, and the five split-failing pins, which pass under `plane`.
* **Other variants with the blend:** `rim` and `exact` were not given it (they keep the hard
  cap). The centred ramp was run only on the tie gate and the moment error.
* **The radius at exact face tangency** (`plane`'s side-wall step, which `split` keeps).
* **Cylinders inside CSG nodes** (they fall back to the plane's cross term), GPU timing, and
  `examples/` other than the pillar example.

## Second round: reproduce

```sh
cd studies/fdtd/cylinder_rim_moment
export WAVEBIT_STUDY_CYLINDER_MOMENT=split          # eps 0.02, inside ramp, C2 by default
python blend_check.py switch; python blend_check.py sweep; python blend_check.py ad
python blend_gradient.py                            # WAVEBIT_STUDY_CYLINDER_CAP_BLEND=0.25 etc.
STUDY_MODES=split python spike.py fdtd              # and voxel|lattice w2|mix
python split_error.py; python split_error.py offsets 32
python metasurface_modes.py split M1,M2 && python metasurface_modes.py report
for m in plane split; do python cost.py occupancy $m; python cost.py fdtd $m; done
python -m pytest <the 36 files> -m "not research" -n 2
```

## First round (2026-11-10)

### TL;DR

* **The rim spike is the plane moment's own, and both prototypes that drop the plane at the
  rim remove it.** Under `plane`, d(loss)/d(radius) for an aligned rim grows as 1/delta:
  1.7e12 at delta = 1e-16 in the end-to-end FDTD loss, against about -157 elsewhere. Under
  `split`, it stays between -168 and -156 for every delta from 1e-3 to 1e-16, at 0, and
  for the negative deltas. AD matches central differences to 3e-7 or better.
* **The exact arc moment brings a spike of its own.** When the circle is tangent to a voxel
  face, the arc length in that voxel has a 1/sqrt(delta) singularity. Rounding as small as
  `0.1 + 0.2` can land a rim on that side. Under `rim` and `exact`, the end-to-end
  d/d(radius) reaches -1.4e6 at delta = -1e-16. This is inherent in the exact per-voxel arc,
  not a bug.
* **`split` is the bounded variant.** In voxels that hold a cap, it uses the cap's exact
  area plus the side wall's own tangent plane. Elsewhere it keeps `plane`. It has no
  normal selection and no exact arc.
* **It does not improve accuracy where it is measured.** Against the stored RCWA ladder, C7
  M1 s moves from 2.925e-2 / 7.403e-3 / 2.095e-3 to 2.938e-2 / 7.425e-3 / 2.103e-3 in
  amplitude, at the same order (-1.902). Efficiency improves by 1-4%. The per-voxel
  moment error at the rim falls by 20-80x, but that does not show in the observables.
* **All three variants share one defect: a point outlier at an exact cap tie.** When a cap lies
  exactly on a voxel face, `be` jumps by 1.2e-2 against a 1.6e-4 neighbour step. That fails
  V5's height-continuity gate (factor 50, measured 75). See "The tie outlier" below.
* **Recommendation: keep.** Do not land any variant as the default now. If the rim
  gradient spike must be removed, land `split` with re-pins, but only after the tie outlier
  is fixed. Reasons are below.

### What was built

The cylinder already had an exact fill: clipped disc area times axial overlap. Its moment
(`area`, `area * n_c^2`) came from the tangent plane at the nearest-surface normal, and
the constitutive cross term used that normal directly (`weights = n_i n_j`). At the rim,
`_Extrusion._extruded_surface` turns the normal through 90 degrees within delta. That is
the C/delta gradient pinned by
`test_the_cap_rims_steep_gradient_is_the_planes_own_derivative`.

`Cylinder._exact_tensor` computes the exact `<n n^T>` of the surface inside a voxel. The
side wall uses the 2-D divergence theorem in disc-centred coordinates. With box
`[X0, X1] x [Y0, Y1]`, `L(X)` is the chord of the line `x = X` inside the disc and box, and
`m` is its first moment:

    R * M_uu = A - X1 L(X1) + X0 L(X0)
    R * M_vv = A - Y1 L(Y1) + Y0 L(Y0)
    R * M_uv = m(Y0) - m(Y1)

All three are multiplied by the axial overlap. The caps are planes: clipped disc area times
`plane_share`. Measured against a 2,000,001-sample arc quadrature on 400 random voxels,
the worst error is 1.2e-5 dx^2. That is quadrature-limited: with 200,001 samples it is
1.3e-4 (`check_exact_tensor.py`).

The four switch values:

| switch | moment in a cap voxel | moment elsewhere | constitutive cross term |
|---|---|---|---|
| `plane` (default) | tangent plane at the nearest-surface normal | same | `n n^T` |
| `rim` | exact | plane | exact tensor / area in cap voxels |
| `exact` | exact | exact | exact tensor / area everywhere |
| `split` | exact cap + the side wall's own tangent plane | plane | split tensor / area in cap voxels |

All of these are written with `xp.where` and `array_namespace`. They run on NumPy, JAX and
Torch and stay differentiable in radius, height and centre. Torch is exercised by the pillar
example and JAX by the end-to-end spike.

### (a) Spike

`spike.py` shrinks the rim to alignment with `R = R0 - delta` and `H = H0 - 3 delta`, for
delta = +/-1e-3 ... 1e-16 and 0.

* **voxel:** one voxel at the rim, w2 loss.
* **lattice:** every Yee staggering, with the mixed 1/eps_c loss at eps = 12.
* **fdtd:** the PEC dipole scene, 24 steps, JAX grad and NumPy central differences.

Max |dL/dR| over all deltas:

| level | plane | rim | exact | split | normal scale |
|---|---|---|---|---|---|
| voxel, w2 | 2.95e15 | 1.01 | 1.01 | 0.90 | about 1 |
| lattice, mix | 1.33e17 | 1.69e9 | 1.05e11 | 4.90e6 | about 4-5e6 |
| fdtd end to end | 1.68e12 (delta = 1e-16); 4.05e6 (delta = 0) | 1.43e6 (delta = -1e-16) | 1.44e6 (delta = -1e-16) | 168 | about 157 |

* Under `split`, |dL/dH| <= 5.83 at every delta.
* Where a central difference can be taken (|delta| >= 1e-7), AD matches it to 3e-7 or
  better in `split`.
* The loss jump across alignment is 5.7e-4 absolute for `plane` and 2.3e-3 absolute for
  `split`, out of 1069, so 2.2e-6 relative.
* Lattice w2 under `rim` and `exact` reaches about 3.9e9, from sliver voxels with fill of
  order 1e-12. Mixing weights w only through contrast ~ f(1-f), so those voxels are
  physically inert.

Records: `records/spike_*.txt`.

### (b) How much results move

* **Fill never changes.** Only `w` and the node tensor change.
* **`rim` and `split`** change the rim layers only: 3.7-30% of fractional voxels, with rms
  |dw| 0.24-0.30 and max 0.5-1.0.
* **`exact`** changes almost every cut voxel, with rms 0.05-0.23. See `records/wchange.txt`.

FDTD observables:

* **Pillar example** (`examples/fdtd/pillar_gradient.py`, Torch), T00 and dT/d(R, H, eps):

  | mode | T00 | dT/d(R, H, eps) |
  |---|---|---|
  | `plane` | 0.980767387 | (0.19333, -0.08175, 0.00625) |
  | `split` | 0.980720500 | (0.18543, -0.08235, 0.00619) |
  | `rim` | 0.980719988 | (0.18513, -0.08244, 0.00618) |
  | `exact` | 0.980731995 | (0.18644, -0.08244, 0.00620) |

  Under `split`, dT/dR moves by -4.1%. T00 moves by 4.7e-5.
* **C7 M1, all orders, all cells:** max |d amp/eff| is 1.8e-3 at 20 cells per wavelength.
  The max relative change, over cells with |ref| > 1e-3, is 3-4%.

Tests and pins that fail under `split` (`records/tests_split.txt`): 1477 tests ran in the
37 cylinder-touching files of `tests/fdtd` and `tests/api`, with `-m "not research"`. The
result was 9 failed, 1468 passed and 8 skipped, and all 9 pass under `plane`. One of the 9,
`test_public_spelling`, failed only because this note did not exist yet.

| test / pin | tolerance | split | what it is |
|---|---|---|---|
| `test_fd_shapes::test_the_cap_rims_steep_gradient_is_the_planes_own_derivative` | slope < 2e-3 | 0.327 | pins the spike itself; it must be rewritten, not re-pinned |
| `test_fd_shapes::test_the_plane_is_the_cylinders_moment_and_this_is_what_that_costs` | rim gap > 0.3 | 0.0186 | pins the plane's rim error; the gap closes |
| `test_fd_shapes::test_a_crease_is_where_w_jumps_and_this_is_how_far` | cyl rim jump > 0.9 | 0.416 | pins the crease jump; the jump shrinks |
| `test_fd_junction_physics::test_public_solver_reproduces_the_measured_study[0]` | atol 1e-10 | max diff 8.4e-4 (rel 4.8%) | stored junction study rows (provenance cc2ccf8761b414ba) |
| `test_fd_junction_physics::test_public_solver_reproduces_the_measured_study[1]` | atol 1e-10 | max diff 8.4e-4 (rel 3.5%) | same |
| `test_fd_junction_physics::test_height_continuity_includes_the_exact_ties[0.0-cylinder]` | max jump <= 50 x median | 1.21e-2 vs 50 x 1.60e-4 = 8.0e-3 | **a defect of the exact cap moment** (all three variants), not a pin; see below |
| `test_fd_metasurface::...[M1-20-p]` | atol 1e-10 | max diff 1.3e-4 (`r -1 0 r_s`) | `tests/references/fd_metasurface_current_rows.json` |
| `test_fd_metasurface::...[M2-25-s]` | atol 1e-10 | max diff 7.6e-4 (`r 0 0 r_s`) | same archive |
| `examples/fdtd/pillar_gradient.py` (examples smoke job) | atol 5e-5 | dT/dR off by 7.9e-3 | the example's own assert |

Under `rim` and `exact`, the same eight tests were rerun (`records/tests_failing_*.txt`):
the same eight fail under both, by similar amounts:

* junction rows: `rim` max diff 8.4e-4, `exact` 7.3e-4
* M2-25-s: `rim` 7.8e-4, `exact` 6.4e-4
* M1-20-p: `rim` 1.3e-4, `exact` 1.2e-4
* height tie: 1.20e-2 under both
* plane-cost gap: 1.4e-3 under both
* crease jump: 0.415 under both
* rim slope: 0.327 under both

All eight pass under `plane` (`records/tests_failing_plane.txt`).

Not run: the remaining 9,100 or so tests in `tests/fdtd` and `tests/api` that do not
mention a cylinder, and the research-marked tests.

#### The tie outlier

`test_height_continuity_includes_the_exact_ties[0.0-cylinder]` moves a pillar through a
grid plane in 41 steps of 1e-3 dx. It fails the same way under `rim` and `exact` (1.20e-2),
so the outlier belongs to the exact cap moment, not to `split`'s side wall. Under `split`, `be` at the exact tie differs from both
one-sided limits (+/-1e-9) by 1.2e-2. Under `plane` the difference is 1.6e-4. The cause is
the hard selection between `plane` (`caps == 0`) and the split moment (`caps > 0`):

* In a rim voxel whose cap lies on its upper face, the split moment holds at the tie and
  below it (|dw| 5e-10).
* Just above the tie, the cap has left the voxel, so `caps == 0` and the voxel falls back to
  the plane moment. Its w differs by about 0.17 in voxels with fill about 0.9.

Counting a cap on a face as whole instead of half was tried. It made the outlier worse
(2.0e-2). A planar interface on a voxel face is a genuine discontinuity of the exact
moment, and only the plane moment hides it. Removing the outlier needs a continuous blend
between the two moments near a cap plane, or a cap moment in the split voxels that matches
the plane's one-sided limit. Neither was built here.

### (c) Accuracy

Two independent references were used.

**Per-voxel moment against the exact tensor** (`split_error.py`, `records/split_error.txt`).
Worst and rms |error| of `<n n^T>/area` over cut voxels, for R/h = 4, 8 and 16:

| | R/h 4 | R/h 8 | R/h 16 |
|---|---|---|---|
| away from the rim, `plane` = `split` (rms) | 1.4e-2 | 6.9e-3 | 3.5e-3 |
| rim, `plane`: worst / rms | 0.59 / 0.183 | 0.64 / 0.140 | 0.65 / 0.104 |
| rim, `split`: worst / rms | 5.9e-2 / 8.5e-3 | 4.0e-2 / 3.6e-3 | 3.7e-2 / 1.35e-3 |

`plane`'s rim error does not converge in the worst case (0.6 at every resolution) and its
rms falls at about order 0.4. `split`'s rms falls at order 1.2-1.4. `exact` has zero error
here by construction.

**Observables against the stored RCWA ladder (C7)**, using the C7 harness unchanged
(`metasurface_modes.py`, `records/metasurface_*.jsonl`). This study's `plane` reruns
reproduce the stored C7 rows: M1 s A 2.9253e-2, M2 s A 7.4079e-2.

| | rung | `plane` A | `split` A | `plane` F | `split` F |
|---|---|---|---|---|---|
| M1 s | 20 / 40 / 80 | 2.925e-2 / 7.403e-3 / 2.095e-3 | 2.938e-2 / 7.425e-3 / 2.103e-3 | 1.739e-2 / 4.581e-3 / 1.363e-3 | 1.717e-2 / 4.458e-3 / 1.315e-3 |
| M1 s order | | -1.902 | -1.902 | -1.837 | -1.854 |
| M1 p | 20 / 40 | 2.775e-2 / 6.603e-3 | 2.793e-2 / 6.721e-3 | 1.363e-2 / 3.495e-3 | 1.364e-2 / 3.476e-3 |
| M2 s | 25 | 7.408e-2 | 7.365e-2 | 1.207e-2 | 1.241e-2 |
| M2 p | 25 | 3.478e-2 | 3.459e-2 | 1.802e-2 | 1.958e-2 |

At M2 25, the largest change `split` makes in any amplitude or efficiency is 2.0e-3
absolute. Relative to the reference this reaches 30-50%, but only in cells that RCWA puts
near 1e-3 (`records/metasurface_report.txt`).

Every M1 s cell under `split` stays consistent with RCWA (C7's criterion 4). Errors in A move by -1% to +2% and errors in F by -4% to +9% (M2 p).
The M1 s order is unchanged. The 20-80x better rim moment does not show at this
contrast (eps 4) and resolution. The observable error is dominated by something else,
probably the field's own staircase and the crease at the rim, which no moment rule fixes.

### (d) Cost

`cost.py` (`records/cost.txt`) measures one process per mode, 1 thread each. A C7 run was
using two other cores at the time, so read the ratios, not the absolute times.

* Geometry: `component_occupancy`, all six components, cylinder R = 12 cells, best of 5.
* Forward + gradient: the PEC cylinder scene, 48 steps, JAX `value_and_grad` over (R, H).

| | `plane` | `rim` | `exact` | `split` |
|---|---|---|---|---|
| occupancy, NumPy | 0.061 s | 0.113 s (1.8x) | 0.089 s (1.5x) | 0.159 s (2.6x) |
| occupancy, JAX | 0.290 s | 0.535 s (1.8x) | 0.415 s (1.4x) | 0.686 s (2.4x) |
| occupancy, Torch | 0.091 s | 0.141 s (1.6x) | 0.122 s (1.3x) | 0.177 s (2.0x) |
| occupancy peak RSS | 512 MB | 522 MB | 511 MB | 528 MB |
| plan | 0.30 s | 0.39 s | 0.36 s | 0.34 s |
| first value_and_grad (with compile) | 16.3 s | 18.9 s | 17.8 s | 20.9 s |
| repeat value_and_grad | 1.70 s | 2.69 s (1.6x) | 2.23 s (1.3x) | 3.17 s (1.9x) |
| peak RSS, fdtd | 643 MB | 765 MB | 749 MB | 751 MB (+17%) |

`split` evaluates the plane, the exact tensor and the side wall's own plane, and computes
the constitutive tensor twice. None of this was optimised. A landed version would compute
only the cap voxels.

### Recommendation (first round, superseded above)

**Keep `plane` as the default. Do not land a variant now.**

1. The only demonstrated benefit is the gradient at an exactly aligned rim. That is a
   measure-zero set, reachable by rounding. Elsewhere `split`'s gradients agree with
   `plane`'s to a few percent, and its observables to 1e-3.
2. Every variant, `split` included, introduces a tie outlier that fails an existing continuity gate. It
   would swap a known, pinned, documented spike for a new one.
3. `rim` and `exact` remove the 1/delta spike but add a 1/sqrt(delta) one on the other side
   of tangency, so they are not fixes.
4. No accuracy gain was measured on observables. Landing any variant costs re-pinning two
   frozen archives (`fd_junction_study_rows`, `fd_metasurface_current_rows`), rewriting
   three geometry pins, and the pillar example's golden.

If an optimiser hits the aligned rim in practice, land `split` with re-pins, after the tie
outlier is fixed with a continuous blend near cap planes. The re-pins are the eight items in
the table above. Before then, re-measure the tie test and the spike table.

### Versions

Python 3.12.3, numpy 2.5.4, scipy 1.18.1, jax and jaxlib 0.12.0.1, torch 2.14.1+cpu,
contourpy 1.4.0, pytest 9.1.1. All match `.github/constraints/tests.txt`. 4 CPUs and 15 GB.
Tests ran with `-n 2`.

### Not measured

* The full `tests/fdtd` and `tests/api` suites (10,586 tests) under any variant. Only the
  37 cylinder-touching files were run under `split`, and only the eight `split` failures
  were rerun under `rim` and `exact`.
* Research-marked tests.
* M1 p at 80 cells per wavelength (stopped after an hour) and M2 at 50 and 100. M2 has
  no order under `split`.
* `rim` and `exact` C7 ladders.
* A Mie-like convergence gate for a cylinder: none exists, and none was built.
* Cylinders inside CSG nodes. `study_tensor` is looked up on the structure's own body, so a
  CSG node falls back to the plane's cross term. That path was not exercised.
* GPU timing.

### Reproduce

```sh
cd studies/fdtd/cylinder_rim_moment
python check_exact_tensor.py
python spike.py voxel w2; python spike.py lattice mix; python spike.py fdtd
python wchange.py; python split_error.py
python metasurface_modes.py split M1,M2 && python metasurface_modes.py report
for m in plane rim exact split; do python cost.py occupancy $m; python cost.py fdtd $m; done
WAVEBIT_STUDY_CYLINDER_MOMENT=split python -m pytest <files> -m "not research" -n 2
```