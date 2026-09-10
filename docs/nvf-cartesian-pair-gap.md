# An exact finite pair family through contact

Written: 2026-09-10

Targets:
- `wavebit/_core/conforming/cartesian_pair_gap.py`
- `tests/test_nvf_cartesian_pair_gap.py`
- `studies/results/nvf_cartesian_pair_gap.log`

`CartesianPairGapCell` evaluates the finite horizontal pair at zero or positive
physical gap on one immutable chart. Its NumPy implementation analytically
removes the thin-triangle poles before evaluating the weak equations, fields
and their gap derivatives. The separately qualified
[contact-opening response](nvf-cartesian-pair-opening.md) remains the endpoint
reference. Ordinary pair builders and their contact-preserving JAX tangents
are unchanged. Usage and admission are in the [manual](guide/cartesian-pair-gap.md).

## Exact rescaled coordinates

The physical family holds widths, common height, lattice, materials and
whole-cell offset fixed. Increasing gap g moves the left/right rectangle by
`(-g/2, 0)` / `(g/2, 0)`; the cell boundary stays fixed. Normalize every length
by L = √det(A) and set s = g/L. The exact collapse graph gives C, injecting
contact potentials into their repeated nodes, and Z, spanning their differences.
The physical potential is

```text
w = L * (C u + s Z v).
```

For each normalized triangle let `N(s) = N0 + s N1` and
`d(s) = d0 + s d1`, with original local equations
`K = epsilon/(2*d) * N N^T` and `F = -epsilon*N/2`.
Write `X = N^T C`, `Y = N^T Z`. A regular triangle contributes

```text
H00 = epsilon/(2*d) * X^T X
H01 = epsilon/(2*d) * X^T Y
H11 = epsilon*s/(2*d) * Y^T Y
fC = -epsilon*X^T/2,  fZ = -epsilon*Y^T/2.
```

For a thin triangle `d0=0` and `N0^T C=0` hold structurally, from exact dyadic
collapse identities. With `a=N1^T C`, its finite contributions are

```text
H00 = epsilon*s/(2*d1) * a^T a
H01 = epsilon/(2*d1) * a^T Y
H11 = epsilon/(2*d1) * Y^T Y
fC = -epsilon*s*a^T/2,  fZ = -epsilon*Y^T/2.
```

Assembling these local terms gives

```text
M = [[H00, s*H01], [H01^T, H11]]
M x = f,  x = [u; v]
M x' = f' - M' x.
```

This is an exact change of finite coordinates at positive gap, not a
truncated endpoint series. At zero it gives the contact and difference
limit equations and their derivative. No finite-gap extrapolation defines
the response. All matrix and field derivatives are evaluated analytically.

The full M is deliberately nonsymmetric; its raw Hermitian part is not a
coercivity test. At positive s, `diag(I,s*I) M` is the physical congruence
with columns `[C,s*Z]`. The implementation separately guards the real-energy
parts of H00 and H11 and reports the condition of M. A contained, positive
chart does not imply a uniform conditioning radius.

## Fields, weights and original equations

The normalized local fields are

```text
B_regular = I + (X u + s Y v)/d
B_thin = I + (a u + Y v)/d1.
```

Neither thin fields nor the rescaled solve divides by gap. Their analytical
derivatives include changing local geometry and the implicit solution
response. Full original periodic weak fluxes are also assembled before C/Z
projection, including the gauge row: each local contribution is
`epsilon*N*B/2`, and its derivative includes `N1*B + N*B'`. These residuals
check the original equations without constructing a divergent matrix solely
for readout.

The existing polygon contour transform evaluates triangle weights, including
repeated points at contact. Its analytical moving-contour derivative retains
area and spatial phase motion. All thin triangles contribute at contact,
including the exterior fan triangles. Independent Duffy moments and the
qualified endpoint weighted-edge expression check absolute values and
normalized derivatives. No uniformly relative accuracy is claimed for very
small individual thin weights.

The nine coefficient fields use `epsilon * B.conj().T @ B` for energy, retaining
both conjugate derivative terms. The selected closure and its derivative use
the exact requested Fourier differences before inversion. Final derivatives
are divided by L to return **per caller length unit**.

## Bounded independent qualification

The frozen thirty-point study uses the same ten declared chart/refinement
cases as the endpoint study, at `s=0, 1/64, 1/16`. The main physical fixture
has basis `[[700,315],[0,490]]`, widths `(140,193)`, height 210, offset `(49,-21)`,
permittivities `(4+.8j,9+.3j)` and background `1+.1j`. It is expressed with all
four signed-column bases at refinements 0/1. The second fixture has basis
`[[640,-160],[70,560]]`, widths `(160,110)`, height 190, offset `(31,-19)`,
permittivities `(2.4+.17j,5.8+.46j)` and background `1.2+.08j`; chart 1 is
admitted at both refinements. The ten cases are not ten distinct physical
geometries. They have 8 or 35 free rescaled potentials.

At positive gaps the independent reference forms exact rational physical
3-by-3 nodal Vandermonde matrices, the original full weak system and its
analytical derivative. Moving-simplex Duffy q16/q24 integration provides the
coefficient reference. At zero it uses the independent original Laurent
endpoint equations. Separate differentiated closure solves select integer
labels explicitly. Native positive-gap cell values are an additional check.

The predetermined absolute value and quadrature guard is `3e-10`; derivative
comparisons multiply the per-unit output by L and require `3e-9`. Original,
projected and differentiated weak residuals and inverse identities require
`1e-10`. No tolerance, case or quadrature changed after the source freeze.

| Frozen thirty-point measurement | Largest discrepancy |
| --- | ---: |
| q16/q24 coefficient value | 1.219e-12 |
| q16/q24 normalized derivative | 5.399e-12 |
| Coefficient value versus independent reference | 4.888e-15 |
| Normalized coefficient derivative versus reference | 1.185e-13 |
| Normalized closure derivative versus reference | 2.815e-14 |
| Endpoint normalized coefficient derivative | 9.735e-15 |
| Original unprojected flux residual | 1.058e-15 |
| Original differentiated flux residual | 1.358e-14 |
| All reported reference weak residuals | 2.689e-13 |
| Independently recomputed saved closure identities | 6.747e-16 |

All 18 signed-basis comparisons, 12 common-unit controls, six zero-direction
sets, four component/slot Taylor controls and nine invalid/immutable calls
pass. Orders `(1,2)` / `(2,1)` follow the exact signed-basis label permutation;
`(0,0)`, `(1,0)` and `(0,1)` check the smaller selected closures. Common scaling
by .01 scales physical gap too and changes returned derivatives by 1/.01.
The Taylor ladder verifies the already defined derivative; it is not a
continuum convergence rate or error bound.

M's measured condition spans 49.4730–489.018. The minimum H00/H11 real-energy
eigenvalues are .203997/.136961, and minimum closure storage/signed dissipation
are 1.00298/.0800624. These are measured properties of the finite cases. The
original unscaled conditions were not retained, so no numerical improvement
ratio is inferred.

The original audit took 2.97389 s, with parent elapsed time 3.29534 s. A separate
saved-array audit took .08486 s and verified all 30 main and 22 control
checkpoints, source/launch/completion pins and the exact recomputed comparisons.
It did not rerun a solver. Original oracle closure arrays and full-load
normalizers were not saved; their original comparisons/residuals were
source/hash verified, while stored matrix residuals, conditions and direct
closure identities were independently recomputed. No RSS was measured.

Repository promotion preserves the numerical abstract syntax after only a
class rename, three relative imports and the module heading. The dedicated new suite
passes 79 tests in 3.14 s, recomputing the independent references without private
arrays. A fresh copied-package guide example also passes with tests, studies,
JAX and PyTorch imports blocked. Full-precision
measurements and source pins are in the
[result log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_cartesian_pair_gap.log).
This finite family does not establish continuum, arbitrary-contrast or patterned
optical accuracy, and introduces no ordinary contact-geometry JVP guarantee.

A separate [NumPy/JAX x64 value evaluator](nvf-cartesian-pair-gap-autodiff.md)
now qualifies ordinary automatic differentiation with respect to the physical
gap on the same frozen family. This analytical NumPy reference remains
unchanged, and ordinary contact-adapter tangents retain their existing scope.
