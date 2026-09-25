# Research: symmetric interface stencils

This is an experimental material operator for the remaining C3 interface
problem. Public `fd.run` reaches it only when a structure asks for it
(`fd.Structure(..., discretisation='stencil')`, since 2026-09-24; the default is
unchanged), and it does not certify C3.
The production cylinder has mean/RMS orders 1.60337/1.58655. The older
three-dimensional symmetric triplet candidate misses the small high-index
sphere control: 3.77084% error against the unchanged 3% limit.

## Construction

Let `F = E_t + n D_n` describe a static planar interface state in vacuum
units, with relative permittivity `eps` on the other side. Exact edge
length and dual-face fractions give

```text
Gamma_j(n) = e_j^T + (1/eps - 1) length_j n_j n^T
Pi_j(n)    = e_j^T + (eps - 1) area_j (e_j^T - n_j n^T).
```

The earlier local map `Gamma inv(Pi)` is accurate but asymmetric.
Symmetrizing each local block loses exact static consistency. The new
experiment instead solves for a symmetric, extended matrix `K`. At each
target edge `i`, its normal `n_i` is frozen and the constraints are

```text
sum_j K_ij Pi_j(n_i) = Gamma_i(n_i).
```

These are linear constraints on shared entries `K_ij = K_ji`. Input edges
can be within two cells of the output edge. The baseline `K0` is the
previous symmetric triplet construction; the correction has the smallest
weighted Frobenius norm subject to these constraints. No scattering
measurement enters this solve. The constant-state planar condition is
necessary, not sufficient for the scattering convergence gate. On curved
surfaces the frozen normal omits spatial field variation and curvature
terms in the interface conditions.

The initial one-cell stencil had a negative minimum eigenvalue
`-5.84418e-4` at circle radius 12 cells. The two-cell stencil gave
`+.02390` there, with positive screened spectra at radii 4, 6 and 8.
This chooses the research stencil width; it is not a guarantee for every
scene. Every prepared matrix still needs its own stability checks.

## Artificial support boundary

A hard radial cutoff initially changed the matrix by `1.41418e-3` when
the geometry and grid were rigidly translated together. Two causes were
separated: world-coordinate roundoff changed which edges lay exactly two
cells apart, and the artificial radial support changed discontinuously.

Neighbour topology now uses exact integer/half-integer grid coordinates.
The support weight is

```text
q_i = signed_distance_i / (reach + sqrt(dimension-1)/2)
w_i = max(0, 1-q_i^2)^2
K_ij - K0_ij = w_i w_j Z_ij.
```

The polynomial is the lowest-degree even polynomial with unit value at
zero and zero value and derivative at both support ends. Its coefficients
follow from these conditions; they are not fit parameters. Minimizing
`||Z||_F` penalizes large corrections near the support boundary. It does
not by itself prove continuity through a constraint-rank change: that
also requires a bounded solution as a weight approaches zero. The
distance includes the dual face's half diagonal. Row normalization of
the constraint system preserves feasible solutions and their least norm,
while improving conditioning as the weights approach zero.

The aligned-circle rigid-translation screen then measures at most
`4.04286e-9`, with constraint residual at most `1.65146e-15`. These are
not machine-precision translation results: grazing circle chords have a
square-root sensitivity to roundoff in their endpoints. General geometry
derivatives and all support/topology events remain unvalidated.

## Preparation and time stepping

The prepared sparse matrix is checked for finite entries and exact symmetry.
A sparse factorization checks the coupled finite block. Row and column
permutations must agree, every diagonal pivot must be positive, and the
residual is measured against `L D L^T` in that common ordering. For unit
lower `L`, the recorded energy lower bound is

```text
min(D) / (bound_1(L^-1) bound_infinity(L^-1)) - ||R||_infinity.
```

The nonnegative inverse of `I - abs(strict_lower(L))` bounds `abs(L^-1)`
entrywise. Two triangular solves with vectors of ones give its row and
column sums. Residuals are reconstructed in `floor(sqrt(N))`-row blocks,
so neither a dense inverse nor a complete dense residual is required.
These inequalities hold in real arithmetic; the recorded reconstruction
residual measures the floating-point factorization error.

Uncoupled diagonal entries are included in the lower bound. The maximum
absolute row sum bounds the upper eigenvalue and supplies the existing
curl/time-step check. Region bounds, exact vacuum halos and separation
from the absorbing layer are checked once before stepping. A failed
check refuses the whole research scene. There are no per-cell exceptions
in the compiled numerical loop.

The scope guard permits one concrete, real, isotropic, nonmagnetic,
lossless sphere or the historical invariant-x TE cylinder in vacuum, a
float64 plan, zero initial fields and one unmasked electric constitutive
term. Geometry and material arguments must agree with that scene. Traced
or array-valued preparation arguments are refused without detaching them.
The time-step rate must be finite and strictly between zero and one.

The runtime uses a padded neighbour-index array to multiply by the sparse
off-diagonal part. Diagonal coefficients stay in the usual field-update
slot; compact electric-displacement memory uses the existing update.
No linear system is solved during time stepping. Preparation currently
uses SciPy on the host. Geometry differentiation and the complete large
sphere preparation range are not yet production ready.

The preliminary radius-4/10-cell sparse screens use 3,761/20,904 coupled
edges and return lower bounds `.01127077/.00855516`. Independent smallest
eigenvalues are `.07174972/.06756362`; absolute reconstruction residuals
are below `1.337e-15`. The corresponding factorization/check times are
2.08/81.47 seconds. The cheap diagonal-dominance and baseline-perturbation
lower bounds were negative on both cases, so they were not used to claim
stability. The original dense certificate remains in the historical control
records; both certificates concern the same material matrix.

## Preliminary controls

All controls below use the historical source, duration, contour and Mie
reference. A zero exit means the calculation completed; it is not C3 pass.

| Variant | Cylinder, 40 cells | Small sphere, 40 cells |
| --- | ---: | ---: |
| Pointwise baseline, hard support | 10.18081% | Not run after failure |
| Triplet baseline, hard support | 3.92430% | 2.55467% |
| Triplet baseline, continuous support | 3.51206% | 2.46813% |

The hard-support variant is superseded because of its translation defect,
even though its small physical controls pass. The continuous-support
cylinder has energy lower bound `.04331670`, squared time-step rate
`.28255297`, late-monitor energy fraction `3.72872e-9`, and apparent
absorption cross section `-4.27770e-6`.

The independent static audit covers 18 contrast/normal/size combinations
and 30 additional subcell offsets. Its initial 18-case maximum discrepancy
is `6.85563e-15`. Eight 2D/3D incremental controls across NumPy, JAX eager,
JAX compiled and Torch differ from the independently indexed sparse product
by at most `1.08421e-16`. The first sparse-certificate fast test run reports
18 passed and two slow cases deselected, exit zero.

The revised source-mutant audit has caught 11 actual defects through 22
assertion failures, covering all 20 new test cases. Every altered source
file was restored byte for byte. A subsequent healthy run, including both
physical controls and the documentation contract, collected and reported
85 cases: all passed, exit zero, with no collection or internal errors.
Additional scope and nonfinite-time-step checks are under review separately;
they are not covered by this historical 11-defect record.

The predeclared cylinder study uses all eight historical placements at
40, 56 and 80 cells per wavelength. Its result must retain signed-mean
and RMS fits, the spread over placements, contour displacement, absorption
and late-energy controls. Existing sphere fits for other material operators
cannot certify this candidate.

## Completed cylinder refinement control

The finite sweep completed all 24 conditions with actual exit zero and
unchanged numerical source hashes. No placements were discarded.

| Cells per wavelength | Mean relative error | RMS relative error |
| --- | ---: | ---: |
| 40 | 3.261836% | 3.275148% |
| 56 | 1.538905% | 1.546348% |
| 80 | 0.740709% | 0.744100% |

The mean/RMS fitted orders are **2.137819/2.137121**, with log-fit scatter
**2.128691%/2.093839%**. Both error series are monotone. The 70 possible
four-placement subsets give orders 2.114320 through 2.160873. This passes
the existing cylinder order >=1.8 and scatter <10% control. It replaces
neither the public material operator nor the complete C3 sphere gate.

Across all cases the largest late-monitor energy fraction is `5.190933e-9`,
the largest absolute apparent absorption cross section is `6.990351e-6`,
and moving the contour by four cells changes the normalized scattering
cross section by at most `1.665981e-6` relative. The smallest certified
energy lower bound is `.01654609`; the largest squared time-step rate
is `.28466859`. Static constraint residuals are at most `2.296774e-15`.
There are no captured run warnings. The small-sphere control above still
uses exactly this material construction. The glass and weak controls below
extend it; the large high-index sphere still needs fresh temporal and spatial
controls.

## Regression and source-defect witnesses

The `fa82487` checkpoint fast suite reports **1,997 passed and six expected NumPy
differentiation skips**, exit zero, with **2,003 collected and 2,003
reported**. The two affected slow physical controls both pass, collected
and reported as two cases, exit zero. This run does not repeat the
unaffected 78 slow cases from the preceding radiation commit.

Stage 0/1/2/3 reference scripts exit zero and reproduce **143/288/140/570
arrays bit for bit**, 1,141 total. No reference is rewritten. Numerical
sources remain unchanged throughout this final verification.

The original 11 source mutations produce 22 assertion failures. Three
additional scope/call/time-step mutations produce another 25, including
all 21 unsupported inputs, bypassing the preparation guard, and excessive,
NaN and zero time steps. Every injected source is restored byte for byte;
the combined witnesses cover all 44 research tests at that checkpoint. The first
guard implementation mishandled normalized three-component materials and
the first mask fixture was not serializable. Both failed healthy tests,
were corrected, and the final suite above was run after restoration.

The [complete measurement record](../studies/results/fdtd_subpixel_symmetric_stencil.log)
retains source digests, per-case energy histories, rejected prototypes,
translation and sparse-bound screens, actual process exits and mutant
assertion witnesses. The public material operator is not replaced here.

```bash
python studies/fdtd_subpixel_symmetric_stencil.py cylinder24.json --sweep-cylinder
python studies/fdtd_subpixel_symmetric_stencil.py sphere40.json --shape sphere
python -m pytest tests/test_fd_symmetric_stencil.py --run-slow -p no:cacheprovider -q
```

Use fresh output filenames. The finite sweep refuses duplicate outputs,
records progress after each condition and refuses a source change or an
unresolved late-field control. Its completion status explicitly says
`completed_measurements_not_C3_pass`.

## Glass and weak sphere refinement

The same material construction, with the independently derived Yee radiation
map, completed 18 conditions: epsilon 2.25 and 1.1, 20/28/40 cells per wavelength,
and the three historical offsets `(0,0,0)`, `(.5,1/3,.2)` and `(.25,2/3,.4)`
in cell units. Radius is .25 wavelengths. Numerical source hashes stayed
unchanged and the finite batch exited zero. No placements were discarded.

| Epsilon | Mean errors at 20 / 28 / 40 cells | Mean / RMS order | Mean / RMS fit scatter |
| --- | --- | --- | --- |
| 2.25 | 2.689504% / 1.330501% / .643254% | 2.063617 / 2.063844 | .625985% / .631509% |
| 1.1 | 1.713313% / .860472% / .411694% | 2.057240 / 2.057310 | .231425% / .228431% |

Both mean and RMS error series are monotone, their orders exceed the unchanged
1.8 requirement, and fit scatter is below 10%. The finest errors exceed ten
times the previously measured `9.7e-5` empty-scene floor. These two finite
convergence controls pass. They do not complete C3 or certify the public map.

The origin glass temporal control extends 24 to 48 periods at 20 cells.
Scattering efficiency changes by `3.31443e-7` relative, the complex forward
field by `4.84350e-7`, and the peak-normalized angular cuts by `7.69836e-7`.
This measured extension supports using 24 periods for this study with retained
per-case decay records. It is not a general duration certificate.

At 40 cells the origin glass has apparent absorption `2.46463e-7`, optical
closure `1.14905e-5`, flux-surface closure `-7.58102e-8`, and angular-integral
closure `5.22945e-7`. Angular peak errors are .675997%/.726452%. The two
90-degree errors are +1.07356%/-.459428%. These satisfy the existing central
glass instrument limits. With the enclosing absorption surfaces described below,
the weak-sphere 20-cell optical residuals are +.026644%/-.017002%/-.026287%
across the three placements. The original shifted values through the cut surface
are retained in the record. The old continuum 52%/26%/13% record remains in the
historical radiation investigation.

## Nonlocal material support and measurement surfaces

The material matrix couples multiple nearby edges. A flux box surrounding the
geometric sphere can still cut that coupling. An exact nonzero-row audit of
`K-I` found 22 and 45 material rows outside the old absorption half-width .4
at the two shifted 20-cell placements. The largest affected coefficients
are `1.604388e-8` and `3.880915e-7`. No tolerance discards these coefficients.
Half-width .45 encloses every material row at all three coarse placements.
The existing source, scattering and far-field surfaces are already clear.

Preparation now checks snapped surfaces against the actual material rows.
A cut flux or far-field surface raises `StencilSurfaceRefused`; an intersecting
source injection band raises `StencilSourceRefused`. Unsupported source or
open flux geometry raises `StencilScopeRefused`. Interior field samples remain
allowed. The check happens once before energy certification and time stepping.
No exception is introduced inside a numerical loop.

The absorption half-width is an explicit research input, never silently moved.
Four corrected coarse controls repeat both shifted placements at both contrasts
with half-width .45. The original absorption values are retained as measurements
through a cut surface and cannot certify enclosed-object absorption. Outer
scattering and complex forward fields are compared directly in the accompanying
record. All four corrected calculations exit zero; both observables are exactly
unchanged in every case. Apparent absorption is `1.18292e-6` through `1.32389e-6`,
and the largest late-monitor energy fraction is `5.22536e-9`. These finite-record
absorption values are retained, not described as exact zero. This surface guard
does not change the material matrix.

After source restoration, the affected fast suite collects and reports **118
passing cases**, and the two slow physical controls both pass. Every process
exits zero, with no errors or `INTERNALERROR`; numerical source digests remain
unchanged. Stage 0/1/2/3 again reproduce **143/288/140/570 arrays bit for bit**.
The unrelated full-suite result above is historical and was not rerun for this
research-only surface guard.

Eight new source mutations cause 12 assertion failures, covering all 11 new
surface/source cases. They remove the cut-surface or injection-band checks,
bypass preparation, reject valid enclosed boxes or interior field samples,
remove named scope checks, or ignore the explicit absorption width. Source
restoration is byte-exact. Including the earlier audits, 22 mutation variants
produce 59 assertion failures covering the 55 research cases at `43d5835`; some
variants exercise related defect classes and are not 22 distinct bugs.

The large epsilon-12 sphere, public integration and the corresponding physical
source-defect witnesses remain required before a C3 pass. Geometry differentiation
and general support/rank events are still outside this research implementation.

## Active-row runtime storage

The matrix, constraints and stability checks are unchanged. Only rows with a
nonzero off-diagonal coefficient need the padded sparse product. The runtime
now multiplies those rows and gathers them back through a compact output map;
every inactive row points to one exact zero slot. No small coefficient is
discarded. A purely diagonal matrix has an empty product and the same zero
slot on NumPy, JAX eager, JAX compiled and Torch.

Before changing the runtime, isolated JAX product measurements gave:

| Sphere radius in cells | All / active rows | Full / active storage | Product speed ratio |
| --- | --- | --- | --- |
| 4 | 10,125 / 3,703 | 12,636,000 / 4,661,844 bytes | 2.129 |
| 10 | 59,049 / 20,953 | 73,693,152 / 26,385,540 bytes | 2.753 |

The two layouts give exactly equal product values on these inputs. The largest
difference from an independently indexed sparse product is `1.66534e-16`.
Three timing blocks retain their spread in the measurement record. These are
isolated operation timings, not whole-simulation speedups or radius-20 memory
bounds.

Post-restoration verification collects and reports 128 passing fast cases and
two passing slow physical controls; both processes exit zero without errors or
`INTERNALERROR`. All watched numerical source hashes remain unchanged. The
production reference arrays were verified at `43d5835` immediately before this
research runtime change; production source files are untouched.

Three actual source mutations cause 26 assertion failures: retaining inactive
rows violates the storage contract, replacing the zero slot by one corrupts
both full and purely diagonal increments, and reversing active output indices
corrupts the product on every backend. The latter two include JAX eager and
compiled execution. All ten added cases have mutation witnesses, as do the
eight modified product cases, and restoration is byte-exact. The research
module now contains 65 test cases, including the two physical controls.

## Preparation assembly storage

`fit` and `triplet_baseline` built their sparse triplets in Python lists. At 56
cells that is `10,840,896` appended entries in each of the three constraint
lists -- `3*(2*unknowns - active rows)` at `1,826,928` unknowns and `40,224`
active rows -- and every one of them a boxed Python float or int. They are now
preallocated arrays: at exact counts in `fit`, where one `dim`-block of entries
belongs to each diagonal pair and two to each off-diagonal one, and at the exact
upper bound with a closing slice in `triplet_baseline`, where a block covers at
most the whole lattice. No arithmetic, no ordering and no stored value changes.

Measured on six preparation fixtures in one process -- two exact periodic planar
interfaces and four sphere bands, the largest `23^3` nodes at radius 7 cells,
`959,383` stored nonzeros and 543 LSMR iterations -- with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`:

| | Before (two runs) | After |
| --- | ---: | ---: |
| Peak working set, whole process | 577,585,152 / 577,556,480 bytes | 396,943,360 bytes |
| Six-fixture assembly time | 25.189 / 22.808 s | 23.688 s |

That is `-180,641,792` bytes, `-31.28%`, against a before-to-before spread of
`0.005%`. **No speed claim is made**: the two unchanged runs bracket the changed
one, which is what "not separable from run-to-run spread" looks like.

Everything that is a function of the code is byte identical across the change on
all six fixtures: the assembled matrix's data, indices and indptr, the balanced
least-squares matrix and the right-hand side handed to LSMR, the solution it
returned with its stop code and iteration count, the four padded arrays, and
eleven of the thirteen report scalars.

**Two report scalars are not a function of the code.** `minimum_eigenvalue` and
`eigenpair_residual` come from `eigsh(matrix, k=1, which='SA')` called with no
starting vector, so ARPACK draws a random one. Two runs of the *unchanged*
source move `minimum_eigenvalue` by `1.750e-15` to `9.106e-15` relative over
these six fixtures, and `eigenpair_residual` by up to `66%`; the change moves
them by `1.592e-15` to `6.524e-15` and up to `114%`, inside and beside that
control. The comparison above therefore holds the two out and records the
control next to them rather than calling their agreement a result. This is
recorded as an open defect, not as a tolerance: `replace` gates on
`report['minimum_eigenvalue'] > 0`, and every measurement record that carries
one carries a number a rerun will not reproduce. Nothing here changes it.

The thread count is held fixed on purpose. The assembled matrix data digest
differs between 1, 2 and 4 BLAS threads -- at most `1.1102230246251565e-16` on
90 percent of the nonzeros, with identical sparsity and identical iteration
count -- while two runs at one thread count agree. Every comparison above is at
two threads, and a comparison across thread counts would say nothing.

Reversing the `dim` constraint rows each pair writes -- an actual source
mutation of the new code -- reddens `test_exact_static_planar_states` on both
normals and `test_thirty_planar_interface_offsets`, taking the planar
consistency residual from `2.220446e-15` to `.101084`, and moves the matrix,
right-hand side and solution digests on every fixture. Sixty other cases in the
module survive it, so the planar states are what holds this line.

## Moved into fd preparation

`fit`, `geometry`, `padded`, `energy_bounds` and `triplet_baseline` now live in
`wavebit/fd/_lower/stencil.py`; `studies/fdtd_subpixel_symmetric_stencil.py`
imports them from there and still owns the scene scope refusals, the surface
clearance rule, the stepping term and `replace`. **Nothing public routes to the
new module**: no `plan()` call, no `params` key, no kernel branch, and
`wavebit/fd/_lower/__init__.py` does not export it. The operator is in
preparation, not in `fd.run`, and this section is not a C3 pass.

The move changes no arithmetic. Apart from four lines at the top of `geometry`
-- two comments and two deferred imports -- the 216 moved lines are byte
identical to the ones they replace. Rebuilt on six fixtures from a tar of each
tracked tree, 138 quantities per tree agree exactly: the assembled matrix's
data, indices, indptr, shape and nonzero count; the balanced least-squares
matrix and the right-hand side handed to LSMR; the solution it returned with its
stop code and iteration count; the four padded arrays; and eleven of the
thirteen report scalars. The remaining two are the ARPACK pair above, which two
runs of one unchanged tree already disagree on.

`geometry` and `triplet_baseline` still reach into `studies/` for the exact edge
and face fraction helpers and the triplet orientation maps, so they import them
where they use them. That seam is why the module sits in `_lower` and is not
exported: importing it pulls no study and cannot cycle, and `fit`,
`energy_bounds` and `padded` need no study at all, but the two that do will not
run outside a checkout that has `studies` importable. Closing the seam means
moving those helpers into the package, which is a separate commit with its own
byte compare.

A sign flipped in one moved constraint -- the normal term of `Gamma_i` -- moves
the matrix, right-hand side and solution digests on every fixture and reddens
seven cases in the module: both `test_exact_static_planar_states` normals,
`test_thirty_planar_interface_offsets`,
`test_rigid_translation_keeps_the_material_operator`, and all three
`test_invalid_time_step_is_refused_before_stepping` cases.

### Half of the seam, closed

Written: 2026-09-25, after review. `geometry` no longer reaches into `studies/`
for its fractions. The exact ball chord and the exact disc cut are
`Sphere.exact_edge_fill` and `Sphere.exact_face_fill` in
`wavebit/fd/_shapes/_bodies.py`; the study helpers the old code called are
themselves wrappers over those two, so this is the same arithmetic reached one
call earlier. In two dimensions the study's circle is that sphere's central
plane, and the lift to `(0, y, z)` makes the transverse sum `0*0 + t*t`, which
is the `t**2` the circle form wrote.

The other half cannot be routed: the triplet orientation maps and their
`ORIENTATIONS` table are research material with no public equivalent, and
`geometry`'s default baseline asks for them. So `triplet_baseline` imports the
study through `_study_module`, which **refuses by name** where `studies/` is not
importable -- an installed wheel carries none -- instead of failing several
frames down on a module the caller never asked for, and the message names the
route that does work, `geometry(..., baseline_kind=None)`.

Measured on the 2-D and 3-D preparation fixtures, against a `git archive` of the
parent commit: the assembled matrix's data, indices, indptr, shape and nonzero
count, the four padded arrays, the three energy bounds and eleven of the
thirteen report scalars are byte identical, 92 of 100 quantities, on both the
default triplet baseline and the diagonal one. The eight that move are the
ARPACK pair above on each of the four runs, and the control -- the unchanged
tree run twice -- moves them by the same amounts: `minimum_eigenvalue` by up to
5.5e-15 relative against a control spread of up to 8.1e-15, `eigenpair_residual`
by up to .38 against a control .83.

`fit` also refuses a **complex** input by name. Its three assembly buffers are
preallocated `float64`, and a lossy `eps` reaches all of them through `gamma`,
`pis` and `pi`, where NumPy casts rather than raising. Measured with the refusal
removed, on the four-point planar fixture: twenty `ComplexWarning`s, none of
which a caller filtering warnings sees, and a `complex128` matrix whose fitted
correction is **exactly real** -- `(matrix - baseline).data.imag` is zero
everywhere at `eps = 12 + 1e-9j` and equally at `eps = 12 + 5j`, where the loss
is 40% of the permittivity. What survives of the loss is the baseline's diagonal
`1/eps`; the interface correction, which is the whole operator, is the lossless
one. It is a
refusal and not a promotion to complex buffers because the routine is real
symmetric throughout -- `lsmr` solves a real least-squares problem,
`eigsh(which='SA')` asks for a smallest algebraic eigenvalue, and
`energy_bounds` certifies an LDL with positive pivots. `ell`, `area`, `normals`
and a supplied baseline are doored beside `eps`, because each of them reaches
the same buffers. The public route cannot deliver a lossy material today --
`fd.Material` refuses a complex permittivity and `prepare` refuses a `Conductor`
or a `Dispersive` by name -- and this refusal says so as well, rather than by the
absence of a caller.

Three source mutants, each applied alone and restored from a pristine copy, run
against `tests/test_fd_symmetric_stencil.py` (68 cases) and, for the third,
`tests/test_fd_triplet_sphere.py` beside it (82 in all):

| Mutation | Failures | The case that went red |
| --- | ---: | --- |
| The complex refusal deleted | 1 | `test_a_complex_permittivity_is_refused_where_the_buffers_are_real` |
| `_study_module`'s `except ImportError` dropped | 1 | `test_the_public_route_runs_with_no_study_and_the_research_one_refuses_by_name` |
| `geometry`'s routed edge fraction swapped for the face one | 1 | `test_same_operator_physical_controls[sphere]` |

The third is the one worth reading: the routing this section describes is gated
by a physical control and not only by a byte compare, so a wrong fraction on the
edge reaches a scattering number and is caught there.

## Large high-index sphere: time and bandwidth controls

The radius/wavelength `.25`, epsilon-12 origin sphere now uses the same
research operator at 40 vacuum cells per wavelength. Its initial broad pulse
(`df=.1`, cutoff `1e-3`) at 48 periods remains unresolved: apparent absorption
divided by scattering is 4.2766%, and final interior energy is 1.5972% of its
peak. Its scattering error is not used as spatial-convergence evidence.

The measured source-spectrum study selected `df=.00625`, cutoff `1e-9`.
Its envelope returns to the cutoff level at 327.87946 periods; the Gaussian
is not abruptly switched off there. The pilot's final energy-decay fit has
time constant 28.20397 periods. An exploratory 100-fold energy reduction,
rounded to the historical 24-period blocks, gives 480 periods. This construction
only chooses a trial; the following completed controls test it directly.
An additional 144 periods gives 624. Halving the bandwidth doubles the envelope's
cutoff-return time to 655.75891 periods and gives the 792-period trial by the same rule.

| Periods | Bandwidth `df` | Scattering efficiency | Apparent absorption | Final / peak interior energy |
| --- | --- | --- | --- | --- |
| 480 | .00625 | 3.472272468592 | -3.00605e-9 | 3.90523e-15 |
| 624 | .00625 | 3.472272466258 | +2.89389e-10 | 3.75098e-17 |
| 792 | .003125 | 3.472272466464 | -5.70177e-16 | 2.45007e-23 |

Extending 480 to 624 periods changes scattering by `-6.71950e-10` relative,
the complex forward field by `2.69102e-9`, and peak-normalized angular cuts
by `4.26165e-9`. The narrower-pulse control changes those observables by
`5.91560e-11`, `2.30124e-10`, and `3.59704e-10`. Each calculation and the
finite batch exit zero, with identical numerical source hashes throughout.
These measurements support 480 periods for this origin case; other grid
resolutions and placements still require their own residual/time checks.

The common result remains **19.53794% below Mie** (`4.315415843395`). The
complex forward-field error is 21.48460%, and angular peak error is 25.81477%.
The optical-theorem residual is only `-1.80629e-5`: agreement between power
measurements does not imply agreement with continuum Mie scattering.
Insufficient record length does not explain the remaining error in this case.
One spatial resolution cannot distinguish bulk-grid and material-interface
errors or establish a convergence order. No wavenumber substitution or fitted
coefficient correction is applied. The large-sphere spatial gate and C3 remain
incomplete.

The measurement log retains the unresolved pilot, all three controls, actual
exit codes, preparation checks, source digests, complex fields and angular
cuts, plus summaries and digests of the full retained energy records. The
numerical implementation is unchanged by this report.

## High-index refinement and external comparisons

The same research operator completes the epsilon-12, radius/wavelength `.25`
origin sphere at 56 cells per vacuum wavelength, 480 periods, `df=.00625`,
and cutoff `1e-9`. Its actual numerical process exits zero. The original
resource-report wrapper failed replacing its JSON; a retained Windows process
handle independently observed the numerical child's exit. The calculation was
not restarted. All 59 watched numerical source hashes still match the records.

Scattering efficiency becomes **3.935783692815**, or **8.797116% below Mie**,
compared with 19.53794% at 40 cells. Apparent absorption is `3.72867e-10`,
the largest final monitor-energy ratio is `3.59066e-18`, and final interior
energy is `2.81863e-17` of its peak. These are distinct energy diagnostics.
Preparation takes 493.062 seconds and the complete experiment 4419.203 seconds;
peak observed process memory is 5,389,631,488 bytes. The 80-cell preparation
memory is not yet measured. An independent time extension at 56 cells is also
pending: a small residual alone is not a measured bound on integrated observables.

Two external solvers now measure this same continuum sphere at 40 cells.
Each uses a paired empty/sphere calculation, normalized by its own incident
power, and records both 480 and 624 periods. Both actual process exits and the
finite batch exit are zero. Exact Mie efficiency is `4.315415843395232`.

| Solver and material representation | Cells/wavelength | Scattering efficiency | Relative Mie error |
| --- | ---: | ---: | ---: |
| Wavebit research extended stencil, 624 periods | 40 | 3.472272466258 | -19.53794% |
| FDTDX 0.6.2 native binary sphere, 624 periods | 40 | 3.508191356225 to 3.510442726983 | -18.70560% to -18.65343% |
| Meep 1.34.0 native sphere with dielectric averaging, 624 periods | 40 | 3.412383060872 to 3.413262530065 | -20.92574% to -20.90536% |
| Wavebit research extended stencil, 480 periods | 56 | 3.935783692815 | -8.797116% |

External ranges retain all three enclosing cube half-widths `.4/.7/.75`
wavelengths. The fixtures use a 2.5-wavelength domain, 12 absorbing cells,
an Ex-polarized source at z = `-.85` wavelengths, and equal physical time
steps `0.5/(sqrt(3)*40)`. FDTDX records interpolated complex E/H face fields
and uses trapezoidal surface integration. Meep uses its public flux API,
including opaque incident-field subtraction records. Native geometry,
face placement, source pulse conventions and flux quadrature differ; these
are not identical discrete problems. Meep implementation source was not read.

| Measured control | FDTDX | Meep |
| --- | ---: | ---: |
| Maximum relative scattering change, 480 to 624 periods | 1.51901e-11 | 1.26968e-7 |
| Three-surface scattering range / outer-surface scattering | .0641335% | .0257662% |
| Final / peak interior energy, 624 periods | 2.07213e-18 | 2.04452e-14 |
| Maximum absolute empty closed-surface efficiency | 8.34882e-5 | 2.13635e-7 |
| Maximum relative incident Ex variation on entry faces | .0805132% | 2.09236e-13% |

Apparent absorption at 624 periods remains `.003096` to `.004764` in FDTDX
and approximately `.002986` in Meep. It is retained as a numerical residual
for this lossless sphere, not interpreted as physical absorption. The small
time changes do not certify the absorbing layer or air gap: independent
layer/gap controls, external refinement and external angular comparisons
have not been completed.

The external measurement helpers pass 10 independently collected and reported
tests with exit zero, no errors and no `INTERNALERROR`. Actual source-copy
mutations cover conjugation, face area, outward orientation, projected area,
invalid incident power and energy-window selection; every added case has a
witness. A follow-up guard-bypass mutation makes all four invalid-incident
cases fail by explicit assertions, avoiding reliance on the first zero-input
variant's incidental division error. The measured driver and raw-array
digests are verified. The full records, helper audit and 56-cell preparation
are retained under `high_index_external_comparison` in the existing study log.

At this coarse resolution all three tested solvers underpredict scattering
by roughly 19-21%. The 56-cell improvement supports a substantial spatial
discretization effect. It neither identifies the complete error mechanism nor
establishes solver ranking or a convergence order. The same-operator
40/56/80 by three-placement gate, public integration and physical mutation
review remain incomplete. **C3 has not passed.** No new tolerance, fitted
correction or numerical-source change is introduced by this report.

## A public scene can ask for this operator (2026-09-24)

`fd.Structure(shape, material, discretisation='stencil')` routes one smoothed
structure to this operator. The default, `discretisation='voxel'`, is unchanged
and is what every scene still gets: the four reference archives reproduce
143/288/140/570 arrays bit for bit, all 23 stage-0..3 skeletons are unmoved, and
242 digested quantities over four default scenes -- a vacuum box, a `Slab`, a
`Box` and a *smoothed* `Sphere`, covering the plan skeleton, every lowered
parameter leaf and the final E and H of a run -- are identical to a `git archive`
of `28f6e49`. Nothing here changes the physics this document measures, and this
is not a C3 pass: **C3 has not passed.**

What the routing is. `fd.plan` calls `wavebit/fd/_lower/stencil.py::prepare`
once, eagerly, on the host, for the structure that asked. It calls the same
`geometry` the research driver calls, so the operator is the research operator;
`tests/test_fd_stencil.py` proves that by digest, with no tolerance, on a small
fixture. Preparation is plan time and never lowering: `fit` is a sparse
least-squares solve, a symmetric eigenpair and a sparse factorization, and none
of the three has a JAX or a Torch spelling. Lowering reads the prepared arrays,
moves them to the backend, patches the operator's diagonal into `be` out of
place and puts its off-diagonal coefficients in `params['stencil']`.

Measured 2026-09-24 on a ball of radius .14 at 18 cells per wavelength in a
36-cell box (1830 active rows of width 104), OMP/OPENBLAS/MKL_NUM_THREADS=2,
best of three runs at 40 and 120 steps with the difference taken per step:

| backend | voxel ms/step | stencil ms/step | added |
| --- | ---: | ---: | ---: |
| numpy | 2.010 | 3.020 | +1.010 (+50%) |
| jax   | .291  | .445  | +.154 (+53%) |
| torch | 4.061 | 5.734 | +1.673 (+41%) |

Preparation cost, five separate processes, one preparation each, peak working
set over the process's own working set just before `geometry`:

| unknowns | 76,353 | 117,864 | 135,768 | 249,462 | 350,730 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bytes per unknown | 749.2 | 728.2 | 710.8 | 658.3 | 620.9 |
| unknowns per active row | 41.72 | 42.98 | 43.43 | 44.33 | 44.76 |

The per-unknown cost falls with size, so the largest measured value is used as an
upper bound over the measured range and nothing is extrapolated past it. The
declared preparation bound is 256 MiB, which is 357,913 unknowns at that constant
-- just past the largest preparation measured -- so the refusal never quotes the
constant at a size it was not measured at. The recorded 56-cell research sphere
reads 45.42 unknowns per active row, above every value in the table and below the
46.0 the bound rounds up to.

**Superseded 2026-09-25** (next section): that table measured `geometry` alone,
while the preparation also runs `energy_bounds`' sparse LDL factorization, whose
fill sets the real peak -- 1.35 to 4.4 times these figures -- and the 256 MiB
bound refused every resolution the operator's orders were measured at.

What is refused, once per whole-scene preparation and never per cell: a shape
with no fractional edge or one that is not `fd.Sphere`; a traced radius, centre
or material value; a conductivity, poles, a permeability or an initial E; more
than one smoothed region; a support that reaches an absorbing layer or leaves the
box; a source or flux surface that cuts the operator's nonlocal rows (the
research clearance rule, moved rather than rewritten); a predicted preparation
above the declared bound; and an operator that does not return to the identity on
its own halo, fails its consistency residual, its minimum eigenvalue, its LDL
energy bound or its own time-step rate. None of them falls back to the other
rule. (**Corrected 2026-09-25:** "more than one smoothed region" was a count of
constitutive terms, which merge, and it never fired; the list the public
preparation now enforces is in the next section.)

Two things are open. The operator's skeleton digest is a function of the BLAS
thread count, because the assembled matrix's data is (commit `c939ed3`); a plan
prepared at a different thread count is a different plan, and nothing yet says
so at the door. And `geometry` still reaches into `studies/` for the triplet
baseline (its exact fractions are public now, see "Half of the seam, closed"), so
`discretisation='stencil'` runs from
a checkout and not from an installed wheel; that is named in a refusal
(`StencilSeamRefused`) rather than worked around.

## The scene around the operator (2026-09-25, after review)

The prepared term owns its whole **grown region** -- the smoothed support plus three
nodes a side: its diagonal replaces `be` over every cell of it and its gather reads
the curl over every cell of it. The operator behind it was fitted with vacuum
hard-coded outside the one body (`fit`'s baseline is `1/eps` inside and `1` outside,
and its constraint rows carry `eps - 1`). The public preparation had dropped the
research driver's scene conditions, so three scenes were accepted and silently
changed, measured on the test fixture before the fix:

| scene | default (`voxel`) | `stencil`, before the fix |
| --- | --- | --- |
| `Scene(background=1.44)` | `be/ratio` 0.6944 at the grown region's corner | 1.0 there: a cube of vacuum round the sphere |
| a cell-snapped `Box` of 1.44 beside the ball, inside the region | 864 cells at 1/1.44 | 0 |
| a second smoothed `Sphere` beside the ball | present | `be` `np.array_equal` to the scene without it |

The research driver's `validate_scope` refused every one of these. Each of its
conditions is now a named refusal at plan time, read line by line:

| `validate_scope` condition | public refusal |
| --- | --- |
| centre and radius concrete, finite | `fd.Sphere`'s static door (`_spec.static_float`, on either discretisation): any traced radius or centre -- a JAX tracer, a `requires_grad` leaf, a forward-mode dual, a `torch.func` wrapper -- is refused by name at construction. `StencilNumberRefused` (plan **and** lowering) covers the material leaves: `eps`, `mu`, the background's `eps` and `mu` are plain Python numbers or NumPy scalars; any array or tensor is refused, traced or not |
| `radius > 0`, `eps > 0` | `fd.Sphere`, `fd.Material` constructors |
| `components` is `(0, 1, 2)` for a sphere | fixed by the public preparation |
| `len(scene.structures) == 1` | `StencilStructuresRefused` |
| `len(p.constitutive) == 1`, `old.field == 'E'` | implied by one isotropic non-magnetic ball in vacuum; `old, = plan.constitutive` |
| `p.dtype == 'float64'` | `StencilDtypeRefused` (a Bloch face makes it complex) |
| `not p.has_E0`, `not p.has_H0` | `StencilScopeRefused`, one sentence each |
| background `fd.Material`, `mu is None`, every `eps == 1` | `StencilBackgroundRefused` |
| structure `fd.Material`, `mu is None`, every `eps` equal | permeability: `StencilScopeRefused` at plan and lowering. Conductivity or poles, and anisotropy: at plan time `checks.check_smoothed` (doors 1 and 4) refuses both on either discretisation before the preparation runs, so the stencil's own arms are the **lowering** door only (`StencilScopeRefused` there) |
| `not old.masks` | `StencilScopeRefused` (PEC or PMC wall) |
| `type(body) is fd.Sphere` | `StencilShapeRefused` (a subclass is refused too) |

The one the research driver never needed, because it only ever had one structure:
**nothing else may occupy the grown region.** A TFSF box has to hold the region
inside it with one node of injection band to spare, a PlaneWave plane and its band
have to miss it, a closed `Flux` or `FarField` surface has to enclose or exclude
it, and a flux or diffraction plane has to miss it (judged as an infinite plane,
which is conservative). This is stricter than the research `surface_clearance`,
which judged only the rows the operator changes; a scene it passes passes that too,
and the public preparation no longer calls it. The test fixture's TFSF box grew from
0.9 to 1.11 wavelengths because of it: at 0.9 its faces sat on the region's
outermost node. The research sphere scenes at 40 and 56 cells pass unchanged.

Not done, and refused instead: fitting the operator against a non-vacuum
background is new physics with no measurement behind it. The commit that carries the
background permittivity into `fit`'s baseline and contrast, and measures it, is what
lifts `StencilBackgroundRefused`.

**Lowering.** `fd.plan` refused a traced leaf, but the repository's gradient recipe
builds the plan from concrete values and lowers a traced scene against it
(`tests/test_fd_constitutive.py`). Measured before the fix on the fixture:
`d(sum be)/d(eps)` was -14.3487 by autodiff and by central difference on the
default, and **exactly 0** by both on the stencil, on `jax.grad`, `jax.jit(jax.grad)`
and Torch alike. `fd.lower` now runs the structure's and the background's doors
again and compares the operator's source digest (the SHA-256 of the exact arguments
`geometry` was called with) against the scene it is handed: a leaf that is not a plain
number is `StencilNumberRefused` (it was `StencilTracedRefused`, see the next section),
and a plain scene whose ball, permittivity, material kind or structure count is not
the prepared one is refused by name (`StencilMismatchRefused`, or the sentence for a
loss, an anisotropy, a permeability or a background).

**`fd.run(scene, plan=p)` no longer re-solves.** `run` re-plans the scene to check
the caller's plan and to take the grazing survey, and that re-plan re-solved the
operator. It now runs every door and takes the operator `p` carries when it was
prepared from the same arguments in the same region. Measured on the fixture, best
of three NumPy runs after a warm-up, OMP/OPENBLAS/MKL_NUM_THREADS=2: **2.965 s ->
0.430 s** per run, `geometry` calls per run 1 -> 0, and the scattered flux
`5.126343269106948e-09` on both trees, bit for bit.

**The coefficients are a field of the term.** They used to live in a module-level
dict keyed by the operator's digest that nothing evicted. `StencilTerm.value` is a
digested field now, beside `index`, `output` and `diagonal`, so they are in the
skeleton and they die with the plan; `test_a_prepared_operator_is_released_with_its_plan`
holds weak references to all four arrays and sees them go.

**Preparation memory, remeasured over the whole preparation.** The constant above
was measured over `geometry` alone. The preparation also runs `energy_bounds`, whose
sparse LDL factorization of the coupled block takes most of the time and sets the
peak. Each row below is one `fd.plan` in its own process (OMP/OPENBLAS/MKL=2), peak
working set (`GetProcessMemoryInfo`) over the working set just before the solve
(`stencil._assemble`), read by a probe that wraps it (a scratch script, not
committed; it calls nothing but the public `fd.plan`):

| scene | cells | active rows | unknowns | solve only, MB | whole preparation, MB | bytes/unknown | plan s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test fixture, ball .14, eps 2.25 | 18 | 1,830 | 76,353 | 50 | 77 | 1,009.7 | 2.6 |
| ball .25, eps 2.25 | 24 | 7,836 | 350,730 | 212 | 613 | 1,747.6 | 25.2 |
| ball .25, eps 2.25 | 28 | 10,404 | 467,346 | 279 | 889 | 1,901.5 | 40.9 |
| ball .25, eps 2.25 | 32 | 13,776 | 622,560 | 366 | 1,502 | 2,412.0 | 67.2 |
| research sphere, eps 12 | 40 | 21,060 | 954,978 | 552 | 2,423 | 2,537.2 | 139.0 |
| research sphere, eps 12 | 56 | 40,224 | 1,826,928 | 1,016 | 5,984 | 3,275.6 | 455.8 |

The cost per unknown **rises** with size -- the factorization's fill grows faster
than the unknown count -- so the old 750 bytes under-predicted the preparation by up
to 4.4 times and its 256 MiB bound refused every resolution the orders were
measured at. `BYTES_PER_UNKNOWN` is now 3300, the largest measured value rounded up,
which is an upper bound over the measured range and not one past it.
`PAIRS_PER_ACTIVE_ROW` stays 46.0 (45.35 at 40 cells, 45.42 at 56).
`MEMORY_BOUND_BYTES` was 5824 MiB, the 56-cell origin prediction rounded up to the
MiB; that refused two of the three research placements at 56 cells, and it is 6144 MiB
now (next section). `fd.Structure(..., discretisation='stencil', memory_bound=<bytes>)`
raises it for one structure.

**Thread count.** The prepared skeleton is still a function of the BLAS thread count.
`test_two_processes_agree_on_a_prepared_skeleton` now runs both sides of its
comparison in child processes with the same explicit thread environment and two hash
seeds, so it proves that the same code and the same threads give the same skeleton in
two interpreters, and says that it proves nothing about thread independence. A plan
prepared at another thread count is still a different plan, and nothing at the door
says so; `fd.run(scene, plan=p)` now runs `p`'s own operator in that case rather than
re-solving one that would not match it.

## Second review of the public preparation (2026-09-25)

Two reviews of `199463e` found one blocker, three majors and four minors. Each was
reproduced on `199463e` before it was fixed; the tests named below each have a source
mutant that fails them.

**A structure that asks for the stencil takes plain numbers only.** The doors decided
"traced" with `materials.concrete()`, which returns `None` for a JAX tracer and for a
Torch tensor with `requires_grad`, and reads anything else through `float()`. A Torch
forward-mode dual has `requires_grad=False` and `float()` reads its primal, so
`torch.func.jvp`, `torch.func.jacfwd` and `torch.autograd.forward_ad` passed both doors.
Reproduced on the test fixture: `d(sum be)/d(eps)` at lowering came back **0.0** on all
three (the voxel rule, and every reverse-mode path, read -14.3487; the central
difference is -14.34871), and plain `fd.run(scene, 'torch')` with a dual `eps` returned
the scattered flux 5.126343269e-09 with a tangent of **exactly 0.0**, against a stencil
central difference of 5.639678457e-09 (the voxel rule's forward mode read
5.527620124e-09 against its difference 5.527620120e-09). Rather than add one transform
at a time to a tracing test, the door is now a **type**: every material leaf the operator
reads -- `eps`, `mu`, and the background's `eps` and `mu` -- has to be a
Python `int` or `float` or a NumPy real scalar of at most 64 bits, and anything else is
`StencilNumberRefused` at `fd.plan` and again at `fd.lower`: every `jax.Array`, every
`torch.Tensor` (a gradient leaf, a dual, a `torch.func` wrapper or a plain tensor), and
every NumPy `ndarray`, a 0-d one included. A NumPy scalar is admitted because it is
immutable host data no transform can wrap and `float()` of it is exact; a 0-d array is
not, because it is mutable and is the type every library converts through, and
`float(x)` is the spelling that works. The refusal names the voxel recipe for a
derivative -- `discretisation='voxel'` in **both** scenes, the plan from plain numbers,
`fd.lower` of the traced scene -- and the gate applies it as written on every derivative
path and gets the central difference back to 1e-5. `concrete()` itself is unchanged: the
voxel rule's forward mode is correct, and changing what the default reads as concrete
is not this commit's to do.

| path (fixture, `d(sum be)/d(eps)` at lowering) | `199463e` | now | mutant: door reverted to the tracing test | mutant: number door removed | mutant: every lowering door removed |
| --- | --- | --- | --- | --- | --- |
| `jax.grad`, `jax.jit(jax.grad)`, `jax.jvp`, `jax.jacfwd` | refused | refused | refused | unnamed `ConcretizationTypeError` | **0.0** |
| `jax.vmap` | refused | refused | refused | unnamed `ConcretizationTypeError` | both lanes the prepared value |
| Torch autograd, `torch.func.grad` | refused | refused | refused | **0.0** | **0.0** |
| `torch.func.jvp`, `torch.func.jacfwd`, `forward_ad` | **0.0** | refused | **0.0** | **0.0** | **0.0** |

Each mutant fails both gates below. With the number door reverted, plain `fd.run(scene,
'torch')` with a dual `eps` returns a tangent of 0.0 again.

Gates: `test_lowering_takes_plain_numbers_only_on_every_path`,
`test_fd_run_takes_plain_numbers_only_before_it_prepares_anything`.

**The lowering door works in both directions.** Only a voxel scene against a stencil
plan was refused. A stencil scene against its voxel twin's plan -- the plan the
refusal's own recipe builds -- was accepted, painted with the voxel rule, and on the
traced path returned the voxel gradient: -12.667396009626980 on `jax.grad`,
-12.667396009626977 on `jax.jit(jax.grad)`, -12.667396009626978 on Torch (reproduced on
review 1's off-centre 20-cell fixture). A plan built with no structure did the same.
`stencil.refuse_discretisation` now runs first in `fd.lower`, whether or not the plan
has constitutive terms, and refuses either disagreement (`StencilMismatchRefused`).
Gate: `test_the_discretisation_door_at_lowering_works_both_ways`.

**The operator dies with its plan on JAX too.** `JaxRuntime` compiles the executor once
per plan and keeps it for the life of the process, and both its key and the compiled
closure held the plan. Reproduced: a three-radius sweep through plain
`fd.run(scene, 'jax')`, the caller keeping nothing, left 1, 2 and 3 live operators (2.37,
4.85 and 7.50 MB; about 77 MB each at 56 cells). The executor now keys and closes over
`_run._program(plan)`, the plan with its digested arrays set to `None` -- the step never
reads them; the kernel's coefficients come through the params. The skeleton is kept, and
it holds the arrays' digests, so equality is unchanged, and a plan with no digested field
(every default scene) is passed through as the same object. After the fix the same sweep
leaves 0 operators (3 compile-cache entries, as before). The alternative the brief
named, holding the plan weakly, was measured and does not work: a `WeakKeyDictionary`
keyed by the full plan left 1, 2 and 3 operators, because the compiled closure has to
reference the plan it steps and so keeps its own key alive. Gate:
`test_a_prepared_operator_is_released_with_its_plan_after_a_jax_run`.

**The default memory bound admits every research placement.** The research sphere's
three placements are `(0,0,0)`, `(1/2,1/3,1/5)` and `(1/4,2/3,2/5)` voxels
(`studies/fdtd_subpixel_mie_transform.py::PLACEMENTS`). Their predictions, through the
public selector with the memory door bypassed and no solve (the 40-cell placed spheres
use the research's own .45 absorption half-width, which the occupancy door requires --
at .4 they are `StencilSurfaceRefused`):

| cells | placement | active rows | predicted MiB | measured peak MB | bytes/unknown measured |
| ---: | --- | ---: | ---: | ---: | ---: |
| 40 | (0, 0, 0) | 21,060 | 3,048.81 | 2,423 | 2,537.2 |
| 40 | (1/2, 1/3, 1/5) | 20,916 | 3,027.96 | -- | -- |
| 40 | (1/4, 2/3, 2/5) | 20,906 | 3,026.51 | -- | -- |
| 56 | (0, 0, 0) | 40,224 | 5,823.14 | 5,984 | 3,275.6 |
| 56 | (1/2, 1/3, 1/5) | 40,481 | 5,860.34 | 4,887 (review 2) | 2,654.5 |
| 56 | (1/4, 2/3, 2/5) | 40,489 | 5,861.50 | -- | -- |

The prediction is not monotone in the true need. It is the active-row count times two
constants, and at 56 cells the placement with 0.6% more rows (0.8% more unknowns) needed
18% less memory. The peak is set by `energy_bounds`' sparse factorization, whose fill
depends on the ordering its heuristic finds for the coupling pattern, and a row count
does not fix that; why the origin fills more has not been measured. (At 56 cells the
origin is a cell centre, not a node -- the domain's half width is 65.2 cells, so the
nodes sit at half-integers -- which is also why it has the fewest rows there and the
most at 40.) The row count itself moves with sub-cell placement: over 1,006 placements
at 56 cells (the three above, three symmetric ones and 1,000 seeded uniform ones) it ran
from 40,224 to 40,610 rows, 5,879.02 MiB at the top; at 40 cells the largest was 21,060.
`MEMORY_BOUND_BYTES` is 6144 MiB: 4.5% above the largest of those, 4.8% above the largest
research placement, room for 42,440 rows. `BYTES_PER_UNKNOWN` stays 3300, the largest of
seven measured values, and its comment no longer says it rises with size or that past
56 cells it is more likely low than high: at one size it moved by placement, and past
56 cells nobody has measured it either way. Nothing is claimed at 80 cells, which the
default refuses. Gate: `test_the_default_bound_admits_the_resolutions_the_orders_were_measured_at`
plans all six through the public selector, stopping before the solve.

**The minors.**

- The PlaneWave band gate could not fail: -.47 (node 10) and -.6 (node 7) cannot tell a
  one-node band from none. -.5 (node 9, the region starts at node 10) is refused now, and
  the band-0 mutant fails the test.
- The occupancy span's axis order could not fail: every case was a centred ball.
  `test_a_plane_is_judged_against_the_region_on_its_own_axis` puts the ball off centre
  along x (region x 11..27, y and z 8..25) with three planes where the two readings
  disagree; the transposed-span mutant fails it.
- Refusals now name a spelling that works, and each gate applies it:
  `StencilStructuresRefused`, `StencilDtypeRefused` (`fd.Periodic()`, or voxel),
  `StencilShapeRefused` for a subclass, the magnetic `StencilScopeRefused` ("leave mu
  unnamed", or voxel), `StencilSourceRefused` for a Dipole (a TFSF box or a PlaneWave, or
  voxel), `StencilBackgroundRefused` (voxel), the lowering-only conductivity and
  anisotropy arms (the lossless isotropic scene the plan was prepared from -- voxel is not
  named there, because `check_smoothed` refuses those balls on either discretisation),
  and `StencilCFLRefused`, which now prints the courant number that passes (the rate
  goes as its square) and says it comes after the preparation.
- `Scene(background=fd.Material(eps=1., mu=1.))` was told to use vacuum, which is what
  its author wrote. It is told now that a named `mu` is a permeability path whatever its
  value, and to leave `mu` unnamed; a structure with `mu=1` hears the same.
- The plan-time conductivity and anisotropy arms of `_refuse_structure` could not be
  reached from `fd.plan`. They are **deleted from the plan-time path** and kept at
  lowering, where `check_smoothed` never runs and both are live (the source digest reads
  `eps[0]` only). `test_loss_and_anisotropy_meet_check_smoothed_at_plan_and_the_stencil_door_at_lowering`
  pins the plan side to `check_smoothed`'s sentences.

Still open. `materials.concrete()` still reads a forward-mode dual as a number, so the
**voxel** path's other `concrete()` readers (the CFL bound, the cells-per-wavelength
check) take its primal; they compute no derivative themselves, and the voxel rule's own
forward-mode tangent is right, but nothing gates that class. (`fd.Sphere`'s radius and
centre, listed here as open when this section was written, were closed by the third
review below.)

## Third review of the public preparation (2026-09-25)

Two reviews of `b24a708` confirmed every blocker and major above closed on fixtures of
their own -- 21 derivative paths on four material leaves refused by name, the lowering
door refusing every scene/plan pairing, 0 live operators after a JAX sweep, all six
research placements admitted -- and the default byte-identical to `28f6e49` over 656
quantities on 7 new scenes. One review still would not land it, over one finding the
author had disclosed; three minors came with it.

**A forward-mode radius or centre, on either discretisation.** `fd.Sphere` concretises
its radius and centre through `_spec.static_float`, which refused a JAX tracer and a
`requires_grad` leaf and read anything else through `float()` -- and `float()` of a
forward-mode dual or a `torch.func` wrapper is its primal. Reproduced on
`tests/test_fd_stencil.py`'s fixture through public `fd.run(scene, 'torch')`, on a git
archive of `b24a708`, d(flux)/d(radius):

| discretisation | `torch.func.jvp` | `forward_ad` tangent | central difference (h = 1e-3) |
| --- | ---: | ---: | ---: |
| stencil | **0.0** | **none** | 1.96739e-7 |
| voxel | **0.0** | **none** | 1.84128e-7 |

Refused by name now. `_spec.torch_carried` asks Torch's own two questions -- is this a
`torch.func` wrapper (`is_functorch_wrapped_tensor`), does it carry a forward tangent
(`forward_ad.unpack_dual`) -- and `static_float` and `static_int` refuse either, so the
fix is in the one function every static field shares: `Box` sizes, `Resolution`, `Run`,
`PML`, the sources' static fields, not the sphere alone. It refuses and never changes a
value, so no plain-number input moves by a bit. Neither question is wrapped in a `try`:
a Torch release that renames one makes every static door raise, not open. The stencil
number door's radius and centre arms could never fire -- `fd.Sphere` handed them floats
-- and are deleted rather than kept as dead code; the module docstring and
`Structure`'s now say which door guards which leaf, instead of claiming the type door
"closes the class" for leaves it never saw.

Gates: `test_a_torch_forward_mode_value_in_a_static_door_is_refused_by_name` in
`tests/test_fd_stage1_grad.py` and `tests/test_fd_stage2_grad.py`, over every static door
those files already list (`torch.func.jvp`, `torch.func.jacfwd`, `torch.func.vmap`,
`forward_ad`), and `test_a_forward_mode_radius_or_centre_is_refused_by_the_sphere_on_either_rule`
through `fd.run` on both discretisations. Mutants, each restored and SHA-checked, each
failing its gate: `static_float` without the new arm; `static_int` without it;
`torch_carried` without either question. (Corrected by the fourth round below: those runs
stopped at the first failing row, so which rows each mutant let through was not observed,
and the `static_int` arm only changes a refusal's wording -- a forward-mode float was
already refused there as not an integer.)

**The minors.**

- Two arms of the background door survived their tests (the review's mutants D and V):
  `all` read as `any` admitted `fd.Material(eps=(1., 1., 1.2))` at `fd.lower`, and the
  magnetic test dropped admitted `fd.Material(eps=1., mu=1.3)`. `fd.plan` refuses the
  first in `check_smoothed` on either rule, so the lowering arm is its only guard. Both
  are witnesses in `test_a_non_vacuum_background_is_refused_not_replaced_by_vacuum` now,
  at `fd.lower` against the vacuum plan (and the second at `fd.plan`), and both mutants
  fail it.
- `fd.plan` and `fd.check` printed the voxel rule's accuracy sentence (adjoint
  inverse-material cross coupling, order 2.023) on a stencil scene. The stencil scene
  gets its own now -- the research sweeps' 2.064 (glass) and 2.057 (epsilon 1.1), and
  that its C3 gate is incomplete -- and the voxel sentence is unchanged to the
  character. Gate: `test_a_stencil_scene_is_told_its_own_accuracy_and_not_the_voxel_rules`;
  mutants: the stencil never recognised, the voxel sentence always added.
- `StencilStructuresRefused` and `StencilDtypeRefused` promised that a spelling "is
  accepted" when the next door could still refuse it; they say it removes this refusal
  now. `StencilSourceRefused` read "d is an Dipole"; it names the source and its type.

Verified on the committed source: the fast tier (every `tests/test_fd_*.py` and
`test_docs_contract.py`, `-m "not slow"`, `-n 3`, two BLAS threads) 2098 passed and 8
skipped, exit 0, 2106 JUnit cases, no INTERNALERROR; the four freezers reproduce
143/288/140/570 arrays bit for bit with nothing written; the default-unmoved digest (822
quantities over 9 default scenes, NumPy lowering and runs, JAX and Torch runs on four) is
byte-identical to the one built from `28f6e49`; and the plan notes and `fd.check` text of
the same scenes are byte-identical to `b24a708`'s, built twice on each tree. (The fourth
round found `_spec.py` reworded after these runs -- a docstring, but not re-run; the
fourth round's verification runs on its own committed tree.)

## Fourth round: a static review while the sweep ran (2026-09-25)

A read-only review of `d46935a` -- four lenses (the code, every claim against its
evidence, the sweep's measurement tooling, what C3 still needs), eleven agents, nothing
executed while the 20-hour sweep held the machine -- found no blocker or major in the
code. The lens on what C3 still needs reported two blockers and three majors; its skeptics
confirmed all five as real and graded them one major -- the slow tier had not run since
`21bfcfe` -- and four minors (no written pass rule for C3, a C3 claim that must be scoped
to `discretisation='stencil'`, the `studies/` seam, a spot check without an acceptance
rule). The pass rule and the spot check's acceptance rule are now written down before the
results (the sweep's preregistration, kept outside the repository), and the slow tier is
queued on the declaring commit. Its findings on the code, and what was done:

**`torch_carried` refused too much and missed one thing.** `d46935a` refused every
`torch.func` wrapper. `torch.func.grad(f, argnums=0)(eps, w)` wraps `w` too, with nothing
on it, and `jax.grad` admits the same program, so a static field given `w` was refused
on Torch only. And it gated on the value's class being defined in the `torch` package,
so a forward-mode dual of a `torch.Tensor` subclass defined anywhere else was read
through `float()` and lost its tangent. The redesign asks each wrapper layer, from the
outside in, whether it requires grad, carries a forward tangent or is a vmap batch, and
unwraps a layer that answers no. Writing it turned up a case no one had listed: from
inside a `torch.func.grad` layer, a forward tangent on the tensor it wraps reads as
absent. Measured on Torch 2.13: a `forward_ad` dual handed to `grad` as an argument it
does not differentiate unwraps to a tensor whose `unpack_dual` tangent is `None`, while
`grad`'s output still carries the outer tangent (1.0 for `a*w*w` at `w=.5`) -- so
`float()` there drops it silently. While a forward-mode level is open, a `grad` layer
that answers no is therefore refused, not unwrapped; that over-refuses one wrapper that
carries nothing (a `grad`-wrapped constant inside `jvp`), loudly. (The fifth round below
found this mechanism misread -- the blindness belongs to the Grad interpreter on the stack,
not to the wrapper -- and the design silent on seven captured-value rows.)

`tests/test_fd_stage1_grad.py::test_torch_carried_answers_every_transform` asks a real
static door (`fd.PML(wavelength=)`) about 27 ways of making a Torch value, and checks an
admitted one reads back as its own number:

| design | differs from the table, of 27 | of which a derivative dropped silently |
| --- | ---: | ---: |
| `b24a708` (`requires_grad` only) | 12 (2 refused loudly as not a number; 1 admitted a value that carries nothing, the row this design over-refused) | 9 |
| `d46935a` (every wrapper, module-name gate) | 5 | 1 (the subclass dual) |
| `a666d40` | 0 | 0 -- on these rows; see the fifth round |

Mutants, each run on its own, restored and SHA-checked, with the rows each one turns:

| mutant | table rows it gets wrong | other gates |
| --- | --- | --- |
| no tangent question | `forward_ad` dual, its subclass, `jvp` primal, `jacfwd` argument, and a bare `forward_ad` dual under `vmap` with `in_dims=None` (the record showed four: pytest cut the dict message at four entries) | 12 door rows |
| no batch question | the vmap row, refused as "a real number is needed" | 12 door rows |
| no walk (stop at the outer layer) | `grad` over an inner `grad`/`jacfwd` given the leaf as a non-differentiated argument | none |
| no `grad`-layer rule under forward mode | `grad` over a `forward_ad` dual or a `jvp` primal as a non-differentiated argument; the `grad`-wrapped constant inside `jvp` | none |
| module-name gate for `isinstance` | the subclass dual | none |
| no `requires_grad` question inside it | `grad` over an inner `grad` (the static door's own `requires_grad` arm still refuses the rest) | none |
| `static_float` without it | the table fails at the vmap row | 10 door rows |
| `static_int` without it | none (the table's door is a float field) | the `PML(cells=)` and `Run(steps=)` rows, by wording only |

**The history, corrected.** `vmap` was never silent at a static door: `float()` of a
vmap batch raises (it was refused as "a real number is needed"). A single
`torch.func.grad` was never silent either -- it marks its argument `requires_grad` -- but
a `grad` nested in another `grad` or `jacfwd`, handed the outer leaf as an argument it
does not differentiate, was: `b24a708` read it and returned an outer derivative of 0.0
against 1.0. An integer door refused every forward-mode value as "an integer is needed".
What was silent on `b24a708` is a forward-mode value at a float door -- the measured
`Sphere(radius=)` zero -- and a derivative carried under another transform's layer.

**The sweep's own tooling.** The preregistration (kept with the sweep records, outside the
repository) gained, before any 80-cell result existed: the research driver's own temporal
gate (every case's largest monitor decay < 1e-7), the 9.7e-5 floor, an acceptance rule
for the public spot check (bit-identical, or investigated before any verdict), and the
owner's two decisions of the day -- the cylinder arm is settled by re-running all 24
research conditions through the moved preparation code (not `fd.run`: there is no
`fd.Cylinder` before C4), and the 80-cell origin gets its own 624-period time extension.

## Fifth round: the Grad interpreter, not the wrapper (2026-09-25)

A second review, of `a666d40` and of the queue that runs after the sweep, confirmed one
blocker. A forward-mode value **captured by closure** beneath a reverse transform --
`jvp(lambda r: grad(lambda x: x * fd.Sphere(radius=r).radius)(1.))`, the ordinary way to
write a mixed derivative -- was admitted and its derivative came back 0.0. The mechanism:
while a `torch.func` Grad interpreter is on the stack, `forward_ad.unpack_dual` reads every
forward tangent beneath it as absent, on a plain dual that `grad` never wrapped as much as
on a wrapper; `a666d40` refused only a wrapper sitting at a Grad level, and a captured
value is not one. Measured on Torch 2.13: `jvp` over `grad` of `x * a`, `a` captured,
returns 1.0 with no door and 0.0 when the door reads `float(a)`; jax refuses the same
program by name. For `jacfwd` over `jacrev` -- the pattern `torch.func.hessian` is built
on -- `a666d40` had turned `d46935a`'s loud refusal into this silent zero.

`torch_carried` now asks the stack before it asks the value: when a forward level is open
and a Grad interpreter sits above the lowest forward level -- any Grad layer when no `jvp`
interpreter exists (a user's `dual_level` or `linearize`, beneath every interpreter), else
one above the lowest `jvp` -- no tangent can be seen, and every Torch tensor is refused.
Otherwise the walk of the third round runs, without the per-wrapper rule. The review also
measured that functorch's private `temporarily_pop_interpreter_stack` makes the hidden
tangents readable; a door that edits the interpreter stack it is asking about was not
adopted. The price is loud: under `torch.func.hessian`, or a constant captured into a
`grad` inside a `jvp`, a static field handed a tensor that carries nothing is refused
where jax admits it, and a Python number is the spelling that passes.

The table grew to 39 rows -- every nested row now also passes its value by closure -- with
three `policy` rows, refused by choice although nothing rides on them:

| design | a derivative dropped silently | refused a value that carries nothing |
| --- | ---: | ---: |
| `b24a708` (`requires_grad` only) | 16 | 0 |
| `d46935a` (every wrapper, module-name gate) | 2 | 6 |
| `a666d40` (walk; Grad-level wrapper refused under forward mode) | 7 | 2 |
| now | 0 | 0 (and the 3 policy rows) |

Mutants, each alone, restored and SHA-checked; the table's failure message is now a
string, so every row a mutant turns is listed:

| mutant | table rows it turns | door tests failed |
| --- | --- | ---: |
| no tangent question | `forward_ad` dual, its subclass, `jvp` primal, `jacfwd` argument, bare dual under `vmap(in_dims=None)` | 12 |
| no batch question | stops at the vmap row ("a real number is needed") | 12 |
| no walk | `grad` over an inner `grad` / `jacfwd` given the leaf as a non-differentiated argument | 0 |
| no blind-stack rule | the seven captured rows, the two `grad(argnums=0)` over a dual or `jvp` primal rows, the three policy rows | 0 |
| blind whenever any Grad is on the stack | the `grad`-wrapped constant inside `jvp`; `jacrev(jacfwd)` with a non-differentiated argument (both admit) | 0 |
| blind only above a `jvp` (a user's `dual_level` forgotten) | a `forward_ad` dual captured into `grad`, or handed to it as a non-differentiated argument | 0 |
| module-name gate for `isinstance` | the subclass dual | 0 |
| no `requires_grad` question inside it | `grad` over an inner `grad` / `jacfwd` | 0 |
| `static_float` without it | stops at the vmap row | 10 |
| `static_int` without it | none (the table's door is a float field) | 2, by wording only |

Also from this review: the fourth round's counts and its T1 row are corrected above; the
queue that runs after the sweep opens no console windows, keeps running its test tiers if
a measurement job fails, records collection counts beside the JUnit reports, and judges
each freezer by its printed "N of N arrays reproduce bit for bit" line.
