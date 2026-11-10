# Cylinder rim moment: exact and split prototypes (study)

Written: 2026-11-10

**This is a study, not an acceptance.** Nothing here changes the default. The prototype is
behind a private, import-time switch, `WAVEBIT_STUDY_CYLINDER_MOMENT`, which must be set
before `wavebit` is imported. The default is `plane`, which is bit-identical to `main` at
7e48d76b. No test expectation, tolerance, hash or golden was changed. The tests and pins
that would move are listed below with their magnitudes.

Targets: `src/wavebit/fdtd/_shapes/_bodies.py` (`Cylinder._exact_tensor`,
`exact_moment`, `study_tensor`), `src/wavebit/fdtd/_prepare/constitutive.py` (`lower`),
`studies/fdtd/cylinder_rim_moment/`.

## TL;DR

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

## What was built

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

## (a) Spike

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

## (b) How much results move

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

### The tie outlier

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

## (c) Accuracy

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

## (d) Cost

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

## Recommendation

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

## Versions

Python 3.12.3, numpy 2.5.4, scipy 1.18.1, jax and jaxlib 0.12.0.1, torch 2.14.1+cpu,
contourpy 1.4.0, pytest 9.1.1. All match `.github/constraints/tests.txt`. 4 CPUs and 15 GB.
Tests ran with `-n 2`.

## Not measured

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

## Reproduce

```sh
cd studies/fdtd/cylinder_rim_moment
python check_exact_tensor.py
python spike.py voxel w2; python spike.py lattice mix; python spike.py fdtd
python wchange.py; python split_error.py
python metasurface_modes.py split M1,M2 && python metasurface_modes.py report
for m in plane rim exact split; do python cost.py occupancy $m; python cost.py fdtd $m; done
WAVEBIT_STUDY_CYLINDER_MOMENT=split python -m pytest <files> -m "not research" -n 2
```
