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
uses exactly this material construction, but glass, weak and large
high-index spheres require fresh spatial and temporal controls.

## Regression and source-defect witnesses

The final fast suite reports **1,997 passed and six expected NumPy
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
the combined witnesses cover all 44 current research tests. The first
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
