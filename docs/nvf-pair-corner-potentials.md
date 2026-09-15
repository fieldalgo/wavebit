# Real outer-corner potentials in the complete pair tensor

Written: 2026-09-15

Targets:
- `wavebit/_core/conforming/cartesian_pair_corner_galerkin.py`
- `wavebit/_core/conforming/pair_corner_fields.py`
- `wavebit/_core/conforming/pair_corner_positive.py`
- `wavebit/_core/conforming/pair_corner_contact.py`
- `tests/test_nvf_pair_corner_galerkin.py`
- `tests/test_nvf_pair_angular_fields.py`

Fixed real corner powers couple to the cubic/quartic pair potentials and
continuous gap traces in one constitutive solve. The same enlarged field space
supplies the positive-gap tensor, its analytic gap/material JVP, the exact
contact tensor and its right-opening derivative. The explicit NumPy class is
`CartesianPairCornerGalerkinGapCell` in
`wavebit._core.conforming.cartesian_pair_corner_galerkin`.

This finite family uses the existing Cartesian opening plan: two horizontal,
equal-height rectangles inside a positive diagonal periodic cell. Four outer
body corners are enriched by default. The optional `junctions=True` extension
is described in [the junction-family note](nvf-pair-junction-potentials.md).
These controls do not certify joint mesh, polynomial,
Fourier or optical convergence, and ordinary NVF routing is unchanged.

## The added real fields

Let `h` be an original closed control hat at an outer corner, and `n`, `m`
neighboring closed hats. The additional potentials span

```text
h n (1-h)^(alpha-1)
h n m (1-h)^(alpha-2),       alpha in {3/5, 4/5}.
```

The hats are inherited from the original control topology rather than
reassigned on each mesh refinement. The functions vanish on the periodic
boundary and identically on the triangles that collapse at contact. Their
coefficients join the surviving common unknowns; the original disappearing
unknowns and continuous seam traces remain present. Materials never choose
the trial fields or their real coordinate changes.

Part of each expression is already in the polynomial trial space. Removing its
Taylor polynomial changes coordinates without changing the combined space.
With `u=(1-h)^(1/5)`, the remainder has an exact form
`(1-u)^k P(u)/u^m`. Rational polynomial coefficients are formed with Python's
standard-library `Fraction`; `1-u` is evaluated as
`h/(1+u+u^2+u^3+u^4)`. This avoids subtracting nearly equal sampled powers.
Both the field gradient and its moving-point/geometry variation are analytic.

Sampling skips a product only when one of its control hats is exactly zero
at all three triangle vertices. That hat and its gradient vanish throughout
the triangle. The complete column space is retained; there is no magnitude
or rank threshold. Only the resulting support index list is cached per field
instance. Tests compare the defining powers on every triangle at refinements
0 and 1, with changed points and geometry after cache reuse.

Geometry orthogonalization constructs only the field values of its current
common or disappearing block. It uses exactly the same weighted QR rows as
the complete sampler, while avoiding the unused partner block and derivative
arrays during those two passes. The full response assembly and independent
energy verification still retain every field and derivative. A complete
positive-gap control retains bitwise-identical tensor and JVP values after
this change; no general runtime improvement is inferred from concurrent runs.

The constructor accepts degree 3 or 4 and distinct exponents selected from
`(0.55, 0.6, 0.8, 0.95)`. The default `(0.6, 0.8)` retains the original
evaluation described here. Additional radial powers use the
[normalized radial construction](nvf-pair-radial-potentials.md) and its
integration floor. The empty exponent tuple reproduces the original polynomial and
continuous-trace space. Other exponents are refused because their numerical
remainder and quadrature treatment is not implemented by this family.

## Nested angular refinement

`angular_order=1` through `4` adds the following powers for each neighboring pair
already used by the quadratic angular bubbles:

```text
h n m (n-m)^j (1-h)^(alpha-2-j),    j = 1, ..., angular_order.
```

The default is zero, which preserves the original linear/quadratic angular
space. The option works with either the outer corners or `junctions=True`.
All angular levels retain the lower levels, all selected real radial exponents
and every original nodal and disappearing potential.

Near a corner, `n` and `m` each scale with `r=1-h`. Multiplying their product
by `(n-m)^j` and lowering the radial exponent by `j` keeps the potential's
leading radial power `r^alpha`. The new factors resolve angular variation
inside a corner wedge while vanishing along its two bounding rays. They do
not introduce a material-dependent basis or a different contact formula.

The whole prefactor has polynomial degree `3+j`. A Taylor term of radial
degree `k` is removed only when `3+j+k` is no larger than the existing
polynomial degree. When no such term exists, the complete power is evaluated
without a subtraction. The same root-based quadrature and whole positive and
contact equations integrate every new column and its cross terms.

Independent defining-power and moving-gradient checks cover every triangle
at degrees 3 and 4. Whole-field controls include nested dielectric energy,
uniform materials, material linearity, length units, two gap-variation steps
and approach to the same exact metal contact state. These verify the finite
extension; selecting an angular level does not establish joint mesh, radial,
Fourier or optical convergence. Orders three and four use the same complete
homogeneous powers and analytical variations. They extend the selectable space
without changing the default order, radial exponents, physical equations or
acceptance criteria.

The polynomial/angular/Fourier optical comparison is reproducible with
`studies/nvf_pair_angular_resolution.py`. The `--degree` option selects cubic
or quartic common potentials independently of `--angular-order` and
`--refinement`; cubic degree remains the default. Each invocation computes the full
contact state and derivative, checks the original physical guards and both
Fourier reflection identities, and reports p/s transmission, reflection and
optical directional slopes. For example:

```sh
PYTHONPATH=. python studies/nvf_pair_angular_resolution.py --order 4 --angular-order 4 --quadrature 64
```

Two values for `--order`, for example `--order 8 4` or `--order 4 8`, vary
the retained x/y Fourier directions independently. The constitutive cell,
Maxwell solve and reflection check all use that same pair of orders.

Its two small perturbations of the constitutive tensors check the optical
directional stencil, not the physical moving-gap derivative. Independent
physical gap differences remain separate solver tests. The output records the
fixed physical geometry, selected spaces and source hashes; no convergence
verdict is inferred from passing the finite equations.

For the period-700 metal contact with two 300-square rectangles, degree three,
the symmetric mesh and radial exponents `(0.6, 0.8)`, the measured
transmissions and optical directional slopes are:

| Fourier orders (x, y) | Mesh refinement | Angular order | Integration order | p T | s T | p optical slope | s optical slope |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (4, 4) | 0 | 2 | 64 | .485209 | .467190 | -.0363205 | .0308924 |
| (4, 4) | 0 | 4 | 64 | .487462 | .472262 | -.0513455 | .0153370 |
| (4, 4) | 1 | 2 | 64 | .487422 | .473233 | -.0337288 | .0195839 |
| (4, 4) | 1 | 4 | 64 | .488676 | .476527 | -.0451643 | .0172902 |
| (8, 8) | 0 | 2 | 96 | .493193 | .398029 | -.0231899 | .0015036 |
| (8, 8) | 0 | 4 | 96 | .493393 | .399709 | -.0281669 | .0114431 |
| (8, 4) | 0 | 4 | 64 | .483491 | .420411 | -.0510776 | .0146134 |
| (4, 8) | 0 | 4 | 64 | .499815 | .441791 | -.0361071 | .0182143 |

The angular-order-four cases pass every original physical guard and the
unchanged `3e-10` value / `3e-9` gap-derivative reflection limits. Their optical
directional stencil changes are below `8e-8`. Increasing angular resolution
does not settle the remaining Fourier dependence: at angular order four,
the order-4 to order-8 s transmission change is still about `.07255`.
At fixed Fourier and angular order four, refining the mesh changes p/s
transmission by about `.00121` / `.00426` and the optical slopes by
`.00618` / `.00195`. This remaining auxiliary-space dependence must also be
resolved; a stable optical directional stencil alone does not bound it.
At fixed angular order four and the unrefined mesh, increasing only the x
order from four to eight changes s transmission by about `.05185`; increasing
only the y order changes it by about `.03047`. Both directional runs pass the
same physical and reflection guards. Neither directional increase alone
resolves the full two-axis dependence.
These finite responses remain unqualified as an optically converged default.

## Positive gaps and exact contact

Positive quadrature clips each original triangle at the moving profile and
taper boundaries in exact rational barycentric geometry. Zero areas are
decided before conversion to floating point. Each clipped region is integrated
along rays from an actual outer corner. The default radial exponents use
`r=t^5`; expanded exponent sets use `r=t^20` as described in the radial note.
Angular intervals are subdivided when a radial-bound denominator changes by
more than a factor of two. This refines integration without discarding fields.

Splitting all field products into extremely thin regions loses precision in
their derivatives, even when the value integral is accurate. The implementation
therefore integrates complete original-field products on unsplit triangles.
Only products involving the added continuous trace use the moving partition.
The normalized common/disappearing equations are solved together:

```text
[ Mcc       s Mzc^T ] [ uc ] = -[ Sc ]
[ Mzc       Mzz    ] [ uz ]    [ Sz ].
```

For positive gaps, the complete solve uses `vc=uc`, `vz=sqrt(s)*uz`:

```text
[ Mcc             sqrt(s) Mzc^T ] [ vc ] = -[ Sc         ]
[ sqrt(s) Mzc     Mzz           ] [ vz ]    [ sqrt(s) Sz ].
```

This real coordinate change balances the common/disappearing off-diagonal
blocks. The value uses this balanced solve. The derivative uses its scaling
as a fixed preconditioner for the original differentiated equations and the
complete stationary Schur variation described below. Every physical gap term
is differentiated; the artificial coordinate derivative need not be formed
and subtracted. The reported condition number describes the balanced system.
Exact contact retains its separate analytic limit and never evaluates a
square-root quotient at zero.

Every real geometry QR column is retained. Differentiation includes all moving
measures, fields, retained moments and material terms. If original differentiated
equations remain unresolved, the existing `1e-11` internal refinement target
drives a correction to the complete state. It does not replace any tensor or
derivative block with a different method.

Large cancelling potential coefficients can also amplify matrix-assembly
roundoff despite a small backward residual. The positive solve weights its
original weak, retained and flux defects by the complete physical coefficients.
When these forward-weighted defects exceed the same `1e-11` internal target,
it reintegrates the original weak equations from the complete physical field
and refines the value and variation together. At most two corrections use the
existing balanced matrix as a fixed preconditioner. A value correction also
updates the forcing of the differentiated equations before all derivative
variables are corrected. A resolved original state requests no extra quadrature.

The retained and flux defects themselves need accurate arithmetic. For example,
`H - A*x - R*U` can subtract terms much larger than `H`; ordinary matrix products
can leave a rounding residual that the large retained coefficients then amplify.
Refinement must distinguish this arithmetic residue from an actual defect of the
original equations. The NumPy residual evaluator scales rows and columns by
powers of two and splits each matrix entry into three bounded integer chunks
and its full floating remainder. It keeps all 16 cross products. Leading
integer dot sums stay below `2**50`, and compensated high/low accumulation
combines every term before rounding the final small residual. The dimension
changes the integer units, never the field set or a physical tolerance. Partial
products are accumulated one at a time, so their complete expansions need not
remain in memory together.

The same arithmetic evaluates the retained/flux variations and updates the
differentiated weak forcing after a value correction. It changes no matrix,
material, constraint or independent energy check. Exact-input, hundred-digit
oracles test both real and complex cancelling matrix sums, including contraction
dimensions 631 and 8192, separately from the independent full mixed-system tests.
The full original-mesh and symmetric-mesh order-four metal regressions also
pass the unchanged physical guards, as do the expanded subpixel gap, physical
unit, signed gap direction and complex material direction controls.

The physical weak residual integrates common/disappearing products on unsplit
triangles and the actual additional trace products on the moving partition.
Thin rows pair the already normalized measure with their finite physical
fields; their derivative does not divide an accumulated residual by the gap.
All retained potentials, the retained-field coefficients, the tensor and their
full variations are corrected in the same space. The independent energy
verifier is neither an input nor a stopping criterion. The response metadata
reports the number of corrections as `pointwise_refinement_steps`; contact
reports zero because it uses its separate analytic limit.

A metal control with unit period, widths `(0.3,0.4)`, height `0.45`, offset
`(0.01,-0.03)`, gap `0.003`, materials `(12.25,-10+1j)`, retained order four,
cubic degree, mesh refinement one, angular order two and coercivity angle
`1.53` exposed this defect.
At quadrature 64 the former energy and energy-derivative residuals were
`1.21e-10` and `3.54e-10`, exceeding the unchanged `1e-10` guards. The
potential-coefficient norm was about 3,196. Direct physical weak integration,
weighted by those coefficients, reproduced the `9.59e-9` absolute energy
defect; the retained-field moment discrepancy contributed only `3.20e-12`.
The complete public calculation with pointwise refinement now passes all
original guards at that same resolution. Bounded refinement still cannot
recover digits already lost in an inadequately conditioned field
representation: the original guards remain able to reject such a state.

The quadrature-64 public regression took 1,207 seconds on the stated runtime,
just above the 1,200-second CI limit. The regression suite therefore repeats
the same complete potential space at the default quadrature 48. It passes
the original guards in 561 seconds. This changes the integration work of the
fixture, without changing its mesh, angular, radial or retained Fourier fields.

Independent energy verification first constructs the complete original
physical field `E0`. It integrates `E0/E0` on unsplit triangles and the three
remaining `E0/trace`, `trace/E0`, `trace/trace` products on the moving partition.
Its stationary derivative receives the potentials and their fixed-coefficient
geometric field variation, but neither the candidate tensor/JVP nor the solved
potential derivative. All original weak, retained, flux, energy, reciprocity,
storage and dissipation guards remain applicable at their existing thresholds.

At exact contact, the added common fields survive and the complete normalized
disappearing/trace system determines the right-opening derivative. The contact
taper includes both adjacent material sides. No small positive gap substitutes
for zero.

## Measured controls and remaining limits

Measurements use the existing NumPy 2.5.1 runtime on macOS arm64 with Python
3.12 and one BLAS thread. The tests provide the reproduction criteria; the
additional resolution measurements below are observations, not optical goldens.
The measurements below preceded the square-root coordinate balance; current
regressions repeat their acceptance criteria with the balanced full solve.
The tabulated run's raw log is not published. The committed contact-limit test
repeats its acceptance criteria with the class's contact quadrature floor of 48.

For a square period of 700, widths `(300, 300)`, height 300,
permittivities `(12.25, -10+1j)`, background 1, Fourier order 1, degree 3 and
the unrefined opening plan, the following positive results pass every original
physical guard. Both exponents are present. Positive quadrature order is 32;
the contact comparison was integrated at order 64.

| Gap | Tensor norm error from contact | Gap-JVP norm error, multiplied by 700 |
| --- | ---: | ---: |
| `1e-3` | `2.0019297e-2` | `8.8653714e1` |
| `1e-7` | `2.0093163e-6` | `8.7652114e-1` |
| `1e-9` | `2.0095160e-8` | `8.7614797e-2` |

The derivative error ratios are approximately `0.009887` and `0.099957`,
consistent with the square-root-gap taper remainder on these steps. The largest
stationary-energy derivative residual is `6.16e-11`, below the unchanged
`1e-10` threshold. Fourier order 2/degree 3 and order 1/degree 4 also pass the
contact and positive controls at gaps `1e-3` and `1e-7` with positive quadrature
40 and contact quadrature 48.
The default quadrature 48 also passes the positive physical guards for
directional orders `(1, 0)` and `(0, 1)`, and for order `(1, 1)` at refinement 1,
on the asymmetric dielectric test geometry at gap `0.003`.

Tests independently check sampled spatial/moving gradients by finite
variations, empty enrichment against the original full seam family, a uniform
medium, complex-linear material directions, joint gap/material changes and
caller length units. On an asymmetric dielectric pair, quadrature orders
32 and 40 agree at absolute tensor/JVP tolerances `3e-10`/`3e-9`. Fourth-order
gap variations with steps `3e-5` and `1.5e-5` give maximum JVP differences
`7.75e-10` and `8.16e-11`. A deliberately corrupted tensor JVP is rejected by
the independent stationary-energy check.

Passing these finite-space controls establishes the coupling and variations
tested here. It does not establish an accurate optical solution for every
geometry or resolution. Joint optical convergence, including resolution of
material junctions, remains necessary before this family can determine a default.
The implementation uses the existing NumPy runtime and Python standard
library, with no external solver code or new dependency.

## Differentiating the complete balanced response

The positive-gap system balances all disappearing coordinates by the square
root of the normalized gap. This scaling is used as a fixed preconditioner
when solving the original differentiated equations. Differentiating the
scaled solution and subsequently subtracting the derivative of the scaling
would amplify roundoff as the gap closes.

For the original matrix and source `M*u = -S`, define `W = diag(I, gap)`,
`K = W*M` and `F = W*S`. The same complete Schur response has the analytic
stationary derivative

```text
DH = DA + DF[:, reverse].T @ u + u[:, reverse].T @ DF
        + u[:, reverse].T @ DK @ u
```

Both `DK` and `DF` include the physical gap derivative in `W`. All trial
coordinates remain present, and the full potential variation is still solved
and checked in the differentiated weak, retained and flux equations. The
independent physical-energy verifier remains separate and receives neither
the candidate tensor derivative nor the solved potential derivative. The
existing refinement can still reject an insufficiently integrated response;
this derivative formula does not change any acceptance threshold.
