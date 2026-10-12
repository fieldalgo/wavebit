# Complete positive-gap coordinate precision

Written: 2026-09-16

Targets:
- `src/wavebit/_core/conforming/pair_positive_fields.py`
- `src/wavebit/_core/conforming/pair_positive_precision.py`
- `src/wavebit/_core/conforming/pair_positive_energy_precision.py`
- `src/wavebit/_core/conforming/real_trace_coupling.py`
- `src/wavebit/_core/conforming/pair_complex_exponential.py`
- `tests/test_nvf_positive_field_precision.py`
- `tests/test_nvf_positive_metric_precision.py`
- `tests/test_nvf_positive_energy_precision.py`

TL;DR: Large cancelling polynomial and retained Fourier coefficients need
complete precision before projection, through the solve, and into independent
physical verification. A full geometric coordinate change improves the solve
without removing any potential. These finite controls do not establish
optical approximation convergence.

| Question | Decision | Limit |
| --- | --- | --- |
| Does a small original weak residual prove the positive-gap field is accurate? | No. The sextic control has a `1.98e-15` weak residual while its physical energy fails by `1.25e-6`. | Backward equation accuracy alone cannot qualify large cancelling coordinates. |
| Does passing physical energy prove quadrature accuracy? | No. An independent original-node integral rejects an underresolved transverse metal case even when its same-rule energy passes. | Preserve the independent partition and investigate quadrature separately from arithmetic precision. |
| Can a better coordinate system retain the complete space? | Factor the complete material-independent projected geometric Gram with compensated positive pivots. | A nonpositive pivot is rejected; no shift, rank cutoff or column removal is allowed. |
| Can the retained constraint be checked after rounding all coefficients? | Independently integrate the complete physical Fourier mean and compare it with the same identity at `1e-10`. | Keep the rounded-matrix defect visible as a diagnostic; it is not a second physical definition. |
| Is the analytical gap derivative independently checked? | Five separately assembled nearby physical candidates give a fourth-order difference below the existing `3e-9` bound. | This is a finite gap/space control, not a global error estimate. |

## Field and projection

Let `Phi` contain both Cartesian Fourier blocks and `G` contain every physical
common gradient, disappearing `s*Z` gradient and continuous trace gradient.
The field is `E = Phi*x + G*U`. Integrate the unweighted Fourier moments
`Q = integral(Phi.H*G)` directly, then impose `x = I-Q*U`. Consequently the
admissible gradient is `P = G-Phi*Q`. All products and physical variations
retain their high and low components before this subtraction.

The unprojected material blocks are
`K = integral(G.T*epsilon*G)`, `R = integral(Phi.H*epsilon*G)` and
`A = integral(Phi.H*epsilon*Phi)`. Their complete combinations give the same
finite constrained weak equations. The actual material response is solved
once. Neither it nor its derivative is averaged, reflected or projected after
the solve.

The material-independent metric `integral(P.H*P)` supplies a full invertible
coordinate change. Its Cholesky factor retains every positive pivot, including
small ones. Complete residual corrections solve the transformed physical
equations. The coordinate change is only a preconditioner; the derivative is
formed from the original fixed-coordinate physical variation, so no derivative
of that artificial coordinate change enters the physical JVP.

Collapsed triangles require an additional algebraic normalization. The common
gradient uses the coefficient of the Jacobian adjugate divided by the opening
area coefficient. The physical `s*Z` gradient retains the complete affine
adjugate divided by that same coefficient. Neither expression first divides
by a tiny gap and subsequently subtracts or multiplies it away. Moving
transmission fields use the complete sector, cutoff and interpolant derivatives
described in the [transmission note](nvf-pair-transmission-potentials.md).

## Independent physical verification

The verifier receives only complete `x`, the fixed-potential geometric
variation of `x`, and `U`. It reconstructs `E` and its stationary variation at
physical quadrature points. It receives no candidate tensor, tensor derivative,
solved derivative of `U`, or assembled weak matrix.

Unsplit triangles supply every nodal energy. The original clipped partition
supplies all trace/nodal cross terms and trace energy, including both terms of
each cross pair. The Hermitian physical energy verifies the value; the
reciprocal stationary bilinear energy verifies the physical derivative. Their
existing `1e-10` relative bounds remain unchanged.

The same independently sampled field gives `integral(Phi.H*E)`. This must equal
the identity at the original `1e-10` retained-constraint bound. The ordinary
matrix product `x+Q*U-I` can itself lose more than that bound when rounded
coefficients of order millions cancel. Its value remains available as
`rounded_retained_defect`, while `retained_verification='physical_field_integral'`
identifies the actual physical check. Original nodal, weak, flux, derivative,
reciprocity, storage and dissipation checks still apply.

### Original-node integration detects underresolved quadrature

A quintic metal control uses order `(0,1)`, gap `.01`, period 700, widths and
height 300, centered coordinates, reference and physical coefficients
`(12.25,-10+i,1)`, radial order 100 and angular order 32. All 20 real
transmission directions and all three transverse continuous trace columns
remain present. Its energy and stationary-energy defects are `5.19e-22` and
`3.88e-19`, but the original nodal residual is `5.19e-9`. The existing
`1e-10` original-equation guard correctly rejects this state.

Reconstructing the original Bernstein test gradients and the complete solved
field in compensated arithmetic leaves the same `5.19e-9` defect on the
independent clipped partition. This is not loss of coefficient precision.
Reusing the assembly's unsplit nodal rule would instead give `5.18e-29` and
conceal the unresolved integration error; that substitution is not used.

Hold every solved coefficient fixed and increase only the unsplit angular
integration. The 32-to-64 change is `5.19e-9`, matching the independent nodal
defect. The 48-to-64 and 64-to-96 changes are `6.22e-12` and `8.05e-15`.
All three use the same original physical residual normalization. The analogous
derivative changes are `7.93e-12`, `9.51e-15` and `1.23e-17`.
These are integration comparisons of one frozen physical field; they are
not fresh solves or estimates of optical approximation error. An accepted
angular-order argument is a requested integration resolution, not a promise
that every material/geometry combination will pass its physical guards.

The public quintic order-`(0,1)` control is then freshly assembled at angular
order 48. It passes every original physical guard and every tensor/reflected
tensor comparison at the unchanged `3e-10` value and `3e-9` derivative bounds.
The complete run takes 993 seconds and is retained as a regression test.

## Finite measurements and independent oracles

The sextic control uses order `(4,0)`, gap `.01`, period 700, widths and height
300, offset `(-350,-350)`, physical materials `(12.25,-10+i,1)`, flat auxiliary
reference `(2,2,2)`, radial order 120 and angular order 24. The empty auxiliary
mode bank isolates the full polynomial/Fourier interaction. It does not remove
any field requested by that reference.

The preserved old state has common/disappearing coefficient norm about
`3.37e6`. Its separate block coordinate system has condition `2.29e8`.
The complete geometric frame reduces the transformed physical condition to
about `86.89`. A prototype that rounds primitive samples too early still gives
a retained defect `6.16e-9`; precise coordinate/phase evaluation is necessary
as well as compensated matrix products.

With complete primitive fields and coefficients, independently sampled energy
and stationary derivative defects are `2.99e-15` and `8.63e-13`. The physical
retained mean defect is `9.94e-27`, while its rounded-matrix diagnostic is
`1.37e-10`. These numbers describe the independently verified prototype, whose
energy quadratics round only after reconstructing the complete field. The
implementation retains compensated pointwise quadratics as well.

Five fresh candidate assemblies at `.01+k*1e-5`, for `k = 0,-2,-1,1,2`, each
use their own complete geometric frame. The full fourth-order gap difference
has maximum error `5.72e-11` against the analytical derivative, below the
predeclared `3e-9` bound. Conditions remain between `86.8871` and `86.8875`.
This difference checks the complete unqualified candidates; the central
physical energy check above is a separate measurement. Raw diagnostic records
are not distributed.

A separate quintic order-zero prototype retains all 20 real transmission
directions of auxiliary reference `(12.25,-10+i,1)`, unequal widths `.3/.4`,
periods `1.2/.9`, height `.45`, gap `.02`, radial order 100 and angular order 32.
Physical coefficients `(4+.1i,9+.2i,1+.02i)` vary jointly with gap. Full
compensated pointwise energy and derivative defects are `3.20e-21` and
`3.47e-19`; all physical guard thresholds pass. This is a distinct nonempty
field-bank control, not evidence of metal optical convergence.

The committed primitive oracles independently check:

- Every complete sextic polynomial/disappearing gradient and its moving
  variation on five triangles, including a collapsing triangle, at gaps
  `.01`, `1e-9` and `1e-13`. Independent 160-digit potential differences retain
  the `3e-28` relative bound. This does not qualify a whole solve at `1e-13`.
- Full metric factors against 100-digit Hilbert-matrix factors of sizes
  3, 6 and 10, with strict rejection of zero, negative or nonfinite pivots.
- Complete moving polynomial/transmission coordinates against independent
  100-digit potential differences on degree-5, degree-6 and refined meshes.

The actual public sextic control also passes every physical guard and the
unchanged zero off-diagonal reflection checks (`3e-10` for values and `3e-9`
for derivatives). That complete run took 743 seconds before the subsequent
arithmetic cost reductions. A fresh public run with the real-factor Fourier
phases passes the same checks in 615 seconds. These separate runs had
different concurrent workloads, so their timings are not a controlled
whole-solver speed comparison. Separate public corruption controls change the
tensor, its derivative, the verifier's potential, or only its retained field.
Each is rejected by its independent energy or retained-mean check. When only
the verifier's field is changed, the original weak, rounded retained and flux
diagnostics remain below their original bounds; the physical integral detects
the corruption independently.

The public quintic nonempty-bank test also passes the unchanged fourth-order
joint gap/material/background difference at `3e-9`. All five states pass their
own physical checks. Before the Fourier phase and exact-zero arithmetic cost
reductions, this test took 1,811 seconds. It has a bounded 3,600-second test
deadline, instead of the general 1,200-second deadline, to preserve the full
five-state check and its quadrature. No numerical tolerance changes.
The complete public five-state run with those arithmetic reductions also
passes in 1,593 seconds; concurrent workloads differ, so this is another
qualified run time rather than a controlled speedup measurement.

## Cost of complete sector evaluation

The first complete nonempty positive-gap implementation spends most sampled
field time evaluating complex exponentials separately for each corner mode.
The same sector powers can instead be evaluated together in bounded batches.
Each Cartesian potential is the sum of its two complex powers, and its two
derivatives divide those powers by the physical Cartesian coordinate. This
retains the stored reference exponents and angular states.

The complex exponential factors into a real exponential and real sine/cosine.
Range-reduced real arithmetic retains degree 26 for the exponential and
degrees 30/31 for cosine/sine after compensated quadrant reduction. All
factorial coefficients retain their low parts. Independent 100-digit complex
series check real arguments from -500 to 500 and phases from -700 to 700,
including low input components and quadrant boundaries, at `3e-28` relative
accuracy. The same evaluator supplies contact and positive-gap Fourier
phases; independent Decimal phase and variation controls cover both order
1 and order 4 at the existing `3e-29` absolute bound. Exactly zero phase
returns one directly. The complete mode and moving-potential oracles still
apply.

An alternating three-repeat prototype comparison on 12 clipped rows measures
median field-construction times of `11.458 s` and `3.287 s`, a factor of `3.49`.
This is a sampled field comparison, not a whole-response speed claim. The
measurements use an Apple M4 with 16 GiB memory, the installed NumPy 2.5.1
runtime and one BLAS thread;
concurrent verification jobs prevent treating separate full-run timings as
controlled benchmarks. Real/complex matrix products also avoid the exactly
absent imaginary operand of a real coordinate map, while retaining every
high/low cross term. Identically zero operands return an exact zero product
only after finite-input validation; arbitrarily small nonzero operands and
nonzero low parts remain included. A separate five-repeat Fourier-phase
comparison on 512 points and 81 modes gives median times `0.2072 s` and
`0.02665 s` (a factor of `7.77`); this is not a whole-solver speed claim.

Increasing polynomial degree, transmission branches, mesh resolution and
retained Fourier order still changes the optical approximation. Resolving
this numerical precision defect does not supply an optical stopping rule.
