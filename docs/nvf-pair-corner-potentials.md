# Real outer-corner potentials in the complete pair tensor

Written: 2026-09-15

Targets:
- `wavebit/_core/conforming/cartesian_pair_corner_galerkin.py`
- `wavebit/_core/conforming/pair_corner_fields.py`
- `wavebit/_core/conforming/pair_corner_positive.py`
- `wavebit/_core/conforming/pair_corner_contact.py`
- `tests/test_nvf_pair_corner_galerkin.py`

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

The constructor accepts degree 3 or 4 and distinct exponents selected from
`(0.6, 0.8)`. The empty exponent tuple reproduces the original polynomial and
continuous-trace space. Other exponents are refused because their numerical
remainder and quadrature treatment is not implemented by this family.

## Positive gaps and exact contact

Positive quadrature clips each original triangle at the moving profile and
taper boundaries in exact rational barycentric geometry. Zero areas are
decided before conversion to floating point. Each clipped region is integrated
along rays from an actual outer corner with radial coordinate `r=t^5`.
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
blocks. The tensor and its JVP are both evaluated from this complete system.
The derivative includes `d(sqrt(s))=ds/(2*sqrt(s))` in the matrix, source and
mapping back to the original potential derivative. The reported condition
number describes this balanced system. Exact contact retains its separate
analytic limit and never evaluates a square-root quotient at zero.

Every real geometry QR column is retained. Differentiation includes all moving
measures, fields, retained moments and material terms. If original differentiated
equations remain unresolved, the existing `1e-11` internal refinement target
drives a correction to the complete state. It does not replace any tensor or
derivative block with a different method.

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
