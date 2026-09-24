# Research: symmetric interface stencils

This is an experimental material operator for the remaining C3 interface
problem. It is not part of public `fd.run`, and it does not certify C3.
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
