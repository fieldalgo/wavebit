# The finite right-opening limit of a Cartesian pair

Written: 2026-09-10

Targets:
- `wavebit/_core/conforming/cartesian_pair_opening.py`
- `wavebit/_core/conforming/cartesian_pair.py`
- `tests/test_nvf_cartesian_pair_opening.py`
- `studies/results/nvf_cartesian_pair_opening.log`

The separate experimental `CartesianPairOpeningCell` computes the contact
value and analytical right derivative of the finite two-rectangle cell as a
physical gap opens. Its NumPy implementation supports the admitted cyclic
charts at refinements 0 and 1. Independent original nodal equations and moving
triangle integration verify both the nine coefficient fields and the selected
energy closure. This establishes a bounded finite response, not continuum or
optical accuracy. Ordinary `CartesianRectanglePairNVF` contact replay and its
contact-preserving JAX tangents retain their existing contract. The explicit
interface is described in the [manual](guide/cartesian-pair-opening.md).

## A physical gap and an exact collapse

Two horizontal rectangles retain fixed widths, one shared height, materials,
basis and whole-cell offset. Increasing gap g moves the left rectangle by
`(-g/2, 0)` and the right one by `(g/2, 0)`. At g = 0 the common full-height edge
is exact. The outer boundary stays fixed. No Cartesian edge is sheared and no
lattice reduction is performed.

The existing [pair triangulation](nvf-cartesian-pair.md) supplies only its
immutable topology. Integer interpolation numerators with denominator
`2**refinement` identify the collapsed nodes exactly. Surviving oriented,
material-tagged triangles must equal the contact mesh. Thin triangles are
identified by those node identities, rather than a small-area tolerance.
Contact replay still requires strict containment and finite positive regular
triangles; each thin triangle must have a positive right-opening area slope.

| Refinement | Open free potentials | Contact free potentials | Collapsing differences | Thin triangles |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8 | 6 | 2 | 4 |
| 1 | 35 | 27 | 8 | 16 |

The fixed gauge is preserved. C injects a contact potential into every member
of its collapse group; Z spans the remaining differences. Each thin triangle
connects its repeated contact endpoint's two original potentials. This exact
graph connects every collapse group and no others, giving the pole kernel and
difference rank without a geometry-dependent numerical rank cutoff.

## Weak operators before the limit

For one triangle, write its affine-in-gap Jacobian as J(g), its signed
determinant as d(g), and its reference nodal gradients as R. With
`N = R adj(J)`, the local stiffness and source are

```text
K(g) = epsilon/(2*d(g)) * N(g) N(g)^T
F(g) = -epsilon/2 * N(g)
```

Only horizontal vertex velocities occur, so both d and N are affine in g.
Regular triangles have positive d(0); thin triangles have d(0) = 0 and d'(0) > 0.
Expanding these original expressions gives

```text
K(g) = K_minus1/g + K0 + g*K1 + O(g^2)
F(g) = F0 + g*F1
```

The scalar complex-material weak matrix is symmetric, not generally Hermitian.
Local `N(0).T @ C` vanishes structurally on a thin triangle. The implementation
checks that identity and projects its local terms before summation, avoiding
cancellation of assembled pole entries. It also retains the full original
Laurent operators and residuals.

With A = CᵀK₀C and D = ZᵀK₋₁Z, the required finite coefficients satisfy

```text
A u0 = C^T F0
D v0 = Z^T (F0 - K0 C u0)
A u1 = C^T (F1 - K1 C u0 - K0 Z v0)
```

Thus the open potential is `C u0 + g (C u1 + Z v0) + O(g²)`. An additional
difference coefficient checks the full order-g residual. Positive-real
permittivities give coercive contact and difference blocks; their measured
Hermitian-part minima and condition numbers are reported. The projected
contact system is independently compared with the original contact nodal
assembly, including the gauge row.

## Vanishing area still contributes a derivative

Regular triangle derivatives include field, area and Fourier-phase motion.
A thin triangle has zero contact weight but a finite limiting field. For its
repeated endpoint a and other endpoint b, its normalized Fourier-weight
right derivative is

```text
W'(0) = d'(0)/det(A_cell) * exp(-i k·a) * phi2(-i k·(b-a))
phi2(z) = (exp(z)-1-z)/z^2,  phi2(0) = 1/2
```

The weighted edge phase matters: a centroid phase or uniform edge average is
not this collapsed-triangle integral. All thin triangles contribute, including
the exterior fan triangles. The energy derivative retains both conjugate field
terms in `epsilon * B.conj().T @ B`. Exact requested Fourier differences are
selected before closure inversions; the derivative follows the inverse-product
rule. Storage, signed dissipation and inverse-equation residuals remain guarded.

## Independent qualification

The test oracle shares discrete topology only. It builds physical vertices
from exact rational inputs, derives Laurent coefficients from original
3-by-3 nodal Vandermonde matrices, and integrates moving reference triangles
with Duffy quadrature. It separately differentiates the left and right closure
solve equations. Positive-gap samples check the already constructed derivative;
they do not define it by extrapolation.

The ten declared cases use the main basis `[[700,315],[0,490]]`, widths
`(140,193)`, height 210 and offset `(49,-21)`, at both refinements and all four
signed-column representations. The independent second fixture uses basis
`[[640,-160],[70,560]]`, widths `(160,110)`, height 190, offset `(31,-19)` and
different lossy materials at both refinements; both admit cyclic chart 1.
Orders `(1,2)` and `(2,1)` follow the exact basis-label permutation. Additional
controls use `(0,0)`, `(1,0)` and `(0,1)`.

Derivatives are returned per caller length unit. Only comparisons multiply by
L = √det(A), corresponding to the dimensionless perturbation g/L. Common
scaling by α = .01 leaves values unchanged and changes the returned derivative
by 1/α. The fixed quadrature/value guard is `3e-10`, the normalized derivative
guard is `3e-9`, and original weak/inverse residuals must be below `1e-10`.

| Frozen ten-case measurement | Largest discrepancy |
| --- | ---: |
| q16/q24 coefficient value | 9.240e-13 |
| q16/q24 normalized derivative | 4.370e-12 |
| Coefficient value versus independent oracle | 3.580e-15 |
| Normalized coefficient derivative versus oracle | 7.470e-15 |
| Normalized closure derivative versus oracle | 3.372e-14 |
| Full original Laurent residual | 1.056e-14 |
| Native positive-gap weak residual | 1.075e-13 |
| Inverse-equation residual | 8.069e-16 |

All six signed-basis comparisons, four common-unit controls, three zero-order
sets and 14 invalid/immutable controls pass. The measured minimum storage and
signed dissipation are 1.00298 and .0800624; these are properties of the tested
finite cases, not a general contrast theorem. A four-point positive-gap ladder
also passes the predeclared remainder check for every component and closure
slot, without claiming a convergence rate or error bound.

The frozen NumPy audit took 1.08183 s on an Apple M4 host with 16 GiB, Python
3.12.9 and NumPy 2.5.1, with one BLAS thread. Repository promotion preserves its
abstract syntax after class renames, relative imports, module wording and
formatting. The dedicated repository suite passes 37 tests in 1.25 s.
Source-pinned measurements are retained in the
[result log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_cartesian_pair_opening.log);
private raw arrays are not required to run the independent tests.

This finite response does not admit arbitrary orientations, unequal heights,
vertical shifts, overlapping pairs, gain or tensor materials. It does not
prove accuracy for all admitted contrasts or geometries, establish an optical
stopping rule, or turn a fixed-contact JAX tangent into a domain-crossing
shape derivative.
