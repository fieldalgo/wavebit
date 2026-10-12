# Raising both interface and interior field resolution

Written: 2026-09-14

Targets:
- `src/wavebit/_core/conforming/cartesian_pair_polynomial_galerkin.py`
- `src/wavebit/_core/conforming/triangle_polynomials.py`
- `tests/test_nvf_polynomial_galerkin.py`
- `studies/nvf_polynomial_galerkin.py`
- `studies/results/nvf_polynomial_galerkin.log`

The explicit pair Galerkin family can now raise the polynomial degree of its
continuous real potential on the original geometry. Cubic and quartic fields
resolve both triangle interiors and shared material edges. At equal numbers
of unknowns, the recorded quartic fields give smaller closed constitutive
error bounds than refined quadratic fields. The metal bounds remain too large
to certify accuracy. This is an additional finite field space, not completed
NVF routing or optical qualification.

## The same field constraints on a larger real space

On a reference triangle with barycentric coordinates lambda, use the complete
degree-p Bernstein basis

```text
B_alpha(lambda) = p! / (alpha_0! alpha_1! alpha_2!) * lambda**alpha,
alpha_0 + alpha_1 + alpha_2 = p.
```

There are (p+1)(p+2)/2 local potential coefficients. Shared edge coefficients
have the same integer control identities in adjacent triangles, so their
traces agree. These are Bernstein coefficients, not sampled nodal values.
The potential remains continuous and its gradient remains tangentially
continuous across an interface. Complex material values enter the equations,
not the trial basis.

Each derivative has integer coefficients in degree-(p-1) Bernstein
polynomials. Their Gram integrals are factorial ratios. Fourier moments use
exponential divided differences, repeating vertex i exactly alpha_i+1 times.
Differentiating a vertex phase appends that vertex with its multiplicity.
There is no division by a geometric area or a phase difference at collapse.

The geometry and periodic/contact quotient are assembled with integer control
weights before evaluating physical coordinates. The first surviving vertex
fixes the original scalar gauge. The contact-identical coefficients C and
disappearing differences Z retain the original scaling

```text
potential = C*c + s*Z*z,      s = gap / sqrt(cell_area).
```

For each collapsed triangle, the integer derivative coefficients verify that
C has zero transverse derivative before any floating geometry is evaluated.
This preserves the original finite contact equations and analytical gap JVP.
No gap cutoff or substitute contact tensor is introduced.

The full equations remain

```text
E = Phi*x + grad(potential)
x + Q*u = I
K*u + S*x - Qtest*H = 0
H = A*x + R*u.
```

The higher-degree family eliminates through the original K. It avoids
forming the large projected weak matrix before imposing the retained field
constraints. Every original nodal reaction, retained constraint, reciprocal
identity and material guard still applies. This changes the elimination,
not the equations, their tolerance, or the exported tensor blocks.

## Admission and resource scope

`CartesianPairPolynomialGalerkinGapCell` is an explicit internal NumPy class.
It accepts degree 2, 3 or 4 (default 3), an existing fixed pair opening plan,
independent x/y Fourier orders, and the same geometry/material directions as
`CartesianPairGalerkinGapCell`. Degree 2 supplies a basis-change control
against the existing quadratic implementation. Neither class is selected
automatically by ordinary NVF.

The dense implementation requires `degree * 2**refinement <= 16`, matching
the finest existing quadratic edge resolution. Degrees 3 and 4 therefore
admit refinements 0 through 2. This cap controls resources; it promises no
accuracy or stability for every admitted geometry. The existing numerical
guards still reject a solve that loses their required accuracy. Higher
degrees, including the investigated degree 6, are not qualified for this
interface. No guard was relaxed to admit them.

The implementation is dense and does not perform local static condensation.
No new third-party package, copied solver implementation, or licensing
obligation is introduced. Backend generalization and production routing
remain separate work.

`estimate_closed_error` accepts the new explicit class. The refinement driver
keeps its degree and uses its declared refinement cap when `max_refinement`
is omitted. An explicit larger cap raises before starting. The original
[electric/flux coercivity bound](nvf-galerkin-error.md) is unchanged: it
controls a closed constitutive cell at fixed Fourier modes, not opening
derivatives or Maxwell observables.

## Independent verification

The reference uses equispaced cardinal Lagrange functions, constructed as
products with prescribed zeros at every other triangle node. It uses neither
the candidate's Bernstein derivatives nor its factorial energy moments.
Original nodal/Laurent equations and independent Duffy integration check the
whole tensor and exact contact variation for degrees 2, 3 and 4, dielectric
and passive metal pairs. Two quadrature orders must agree before comparison.

Other checks cover the quadratic basis change, repeated-phase and collapsed
Fourier moments, complex material holomorphy, additive simultaneous directions,
physical finite variations, uniform media, unit changes at subpixel gaps,
the real-material nested-space energy identity, and invalid resource requests.
The error Gram is also integrated using independently reconstructed cardinal
electric and displacement fields.

These finite identities do not establish an adequately resolved physical
field. The error-bound measurement is deliberately separate from the tests.
Coupling higher-degree fields to shrinking real gap traces requires its own
positive-gap energy and derivative qualification; the bare pair verification
does not grant that qualification.

## Equal-size closed-cell measurements

The complete source-pinned run contains 28 rows: Fourier orders 2 and 4,
degrees 2 through 4 on coarse and once-refined meshes, an additional twice-
refined quadratic control, and two material cases. The cell is 700 by 700,
with two 300-wide, 300-high touching rectangles and background 1. Materials
are (4, 4), angle 0, or (12.25, -10+1j), angle 1.53. All rows pass their
original finite-state guards.

The table compares the larger equal-size pair: 575 open potential unknowns
and 447 surviving closed unknowns after removing the gauge. B denotes the
operator error upper bound, not a measured actual error.

| Fourier order | Material | Quadratic, refinement 2: B | Quartic, refinement 1: B |
| ---: | --- | ---: | ---: |
| 2 | Dielectric | 0.671559 | 0.409763 |
| 2 | Metal | 937.992754 | 573.986786 |
| 4 | Dielectric | 1.204017 | 0.876516 |
| 4 | Metal | 3466.395529 | 2883.559514 |

The smaller equal-size comparison (143 open and 111 closed unknowns) also
improves with quartic fields in all four recorded cases. Those rows, cubic
results, physical residuals, conditioning and floating cancellation allowances
are retained in the raw log. This comparison concerns approximation at equal
unknown counts; it does not claim equal runtime, equal memory, or asymptotic
speed improvement.

The improvement supports resolving edges and interiors together. It does not
justify accepting the metal result: its conservative bound still exceeds the
approximate tensor norm, so the relative error guarantee is unbounded.
Further field resolution and independent optical/Fourier convergence remain
necessary before any default selection or completion claim.
