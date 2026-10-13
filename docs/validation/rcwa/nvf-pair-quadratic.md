# Quadratic auxiliary potentials at matched cell size

The explicit NumPy `CartesianPairQuadraticGapCell` adds a quadratic scalar
potential to the finite contact/opening family. It retains physical complex
materials, a finite analytical gap derivative at contact, and polynomial
Fourier integration. It is an experimental constitutive instrument, not an
optically qualified replacement for ordinary NVF.

Targets: `src/wavebit/rcwa/_engine/conforming/cartesian_pair_quadratic.py`,
`src/wavebit/rcwa/_engine/conforming/triangle_moments.py`,
`tests/rcwa/test_nvf_pair_quadratic.py`, `studies/rcwa/nvf_pair_quadratic/nvf_pair_quadratic.py`.

## What changes in the finite space

Each affine triangle has six potential nodes: three vertices and three edge
midpoints. The scalar potential is quadratic, its gradient is linear, and
`epsilon B^H B` is quadratic within a triangle. The same physical interface
geometry and scalar weak equation are retained. There is no replacement of a
metal's permittivity by its magnitude, no added loss, and no rotation of the
physical material or equations by the admission angle.

The midpoint graph is exactly the next red-refined P1 node graph. The
immutable dyadic collapse and periodic gauge can therefore be reused, while
the six-node shape polynomials differ from four independent P1 children.
For geometric refinement `r`, the comparison is:

| P2 geometric refinement | P2 open free potentials | P2 contact free potentials | P2 difference potentials | P1 refinement with the same counts |
| --- | ---: | ---: | ---: | --- |
| 0 | 35 | 27 | 8 | 1 |
| 1 | 143 | 111 | 32 | 2 |
| 2 | 575 | 447 | 128 | 3 |

These are matched numbers of potentials, not matched execution times or a
claim that either space contains the other. The P2 cell admits geometric
refinements 0 through 2. It is NumPy-only; no ordinary-NVF dispatch or JAX
replay is added.

For a thin triangle, equality of the collapsed potentials makes the
transverse derivative of the injected contact polynomial identically zero.
The implementation checks its integer polynomial coefficients at planning
before inserting floating geometry. The existing change of variables
`w = L(Cu + s Zv)`, with `s = gap/L`, then removes the thin `1/s` pole.
The original, unprojected weak equations and their derivative remain checked
at `1e-10`. Two defect-correction steps in each linear solve reduce numerical
solve error without modifying those equations or the acceptance criterion.

## Fourier integration without a sampling grid

Let `lambda_i` be barycentric coordinates and `z_i = -i k dot vertex_i`.
For a quadratic Bernstein polynomial with multi-index `alpha`, the signed
Fourier integral is

```text
integral B_alpha(lambda) exp(sum z_i lambda_i) dx
  = 2 det(J) exp[z_0 repeated alpha_0+1,
                 z_1 repeated alpha_1+1,
                 z_2 repeated alpha_2+1].
```

The bracket denotes an exponential divided difference, including confluent
nodes. Differentiating a vertex phase appends that node with its multiplicity.
Differentiating `det(J)` retains the finite area-slope contribution when the
triangle collapses. Neither calculation divides by the triangle area.

Repeated phases use the exact confluent limit. Nearby phases use the
complete-homogeneous series around their mean; its local span is below two
and the retained 21 terms put the omitted absolute tail below double
precision roundoff. Other phases use the divided-difference recurrence.
This avoids cancellation at zero or coincident phase differences. There is
no Fourier quadrature order in the candidate cell. The independent tests
use numerical simplex quadrature instead.

## Finite verification

`test_nvf_pair_quadratic.py` has 61 tests covering:

- Signed Bernstein integrals and their moving derivatives against degree-64
  Duffy quadrature, all six vertex orderings, repeated/nearby phases, exact
  collapse and a gap of `1e-12`. Absolute tolerance is `3e-12`.
- Independent six-node Vandermonde shape polynomials, original unscaled
  positive-gap equations, and successive original Laurent equations at
  contact. The oracle shares topology and the previously qualified rational
  geometry map, not the candidate shape gradients, rescaled matrix,
  polynomial integral or Fourier moment code.
- Two independent integration orders, 24 and 32, for equal-size and skew
  cells, unequal widths, lossy dielectrics and passive metals, including
  geometric refinement 2. Values retain the `3e-10` guard and gap derivatives
  the `3e-9` guard after multiplication by `sqrt(det(basis))`.
- Coarse metal cells at Fourier order 8, independently resolved at Duffy
  orders 48 and 64 before comparing the candidate values and derivatives.
- Independent implicit closure variation, exact rectangular material
  transforms, uniform dielectric/metal controls, physical length scaling,
  immutable collapse counts, invalid replays and small positive metal gaps.

The original-equation oracle also corrects its solve defect; accumulation
uses a wider type where the runtime supplies one. Its normalized original
residual must satisfy the same `1e-10` limit. Passing these checks establishes
the stated finite P2 calculation, not continuum optical accuracy.

## Optical decision

The reproducible instrument compares P1 and P2 with the same number of
potentials at Fourier orders 4 and 8. The cell is square with period 700,
two touching 300-by-300 rectangles, background 1, layer thickness 300,
wavelength 900, incident angle 0.2, azimuth 0, and top/bottom permittivity
1/2.25. It records both linear polarizations, two contact-slope windows, and
a gap of `1e-9`. Cartesian Li is a separate same-geometry comparison and is
not assumed converged at either order.

The record is `studies/rcwa/nvf_pair_quadratic/records/nvf_pair_quadratic.log`. For materials
`12.25 | -10+i`, the contact transmission totals are:

| Open potentials | Space | p, order 4 | p, order 8 | s, order 4 | s, order 8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 35 | P1, refinement 1 | 0.248701662 | 0.405985788 | 0.367536644 | 0.356434074 |
| 35 | P2, refinement 0 | 0.268655504 | 0.430894142 | 0.206028027 | 0.325377055 |
| 143 | P1, refinement 2 | 0.520814973 | 0.609365704 | 0.640217485 | 0.549681099 |
| 143 | P2, refinement 1 | 0.453466394 | 0.474055691 | 0.536261627 | 0.497661987 |
| 575 | P1, refinement 3 | 0.486649484 | 0.618325309 | 0.449130307 | 0.518567397 |
| 575 | P2, refinement 2 | 0.297747979 | 0.392365936 | 0.449543501 | 0.455680029 |

Li gives p totals 0.483946395/0.480976241 and s totals
0.480212354/0.402781608 at orders 4/8. A favorable single P2 result at 143
potentials does not survive increasing the potential space to 575. Both
inner-space and outer-order changes remain too large to declare metal
convergence. Raising the potential degree alone is not a solution to this
fixture's metal accuracy or contact-slope problem.

The equal-material dielectric `4 | 4` comparison does improve. At order 8
and 575 open potentials, P1/P2/Li give:

| Quantity | P1 | P2 | Li at the same Fourier order |
| --- | ---: | ---: | ---: |
| p transmission | 0.997770187 | 0.997693922 | 0.997682393 |
| s transmission | 0.986340705 | 0.987748093 | 0.988163075 |
| p contact slope | 6.31004e-6 | 6.60234e-6 | 8.09165e-6 |
| s contact slope | -1.42074e-4 | -1.55046e-4 | -1.58322e-4 |

Slopes are per caller length unit, using the smaller second-order one-sided
window recorded by the instrument. These improvements remain comparisons to
a finite-order Li result, not certified continuum errors.

The study also records the condition number of the compressed field map
`Q = C(B)` separately from the finite weak matrix. Metal examples have
field-map condition numbers in the hundreds even when original-equation
residuals pass. A small weak residual establishes a solved auxiliary cell;
it is not evidence that the truncated optical field is resolved. Neither
coercive admission nor a finite contact derivative closes that gap.

The P2 instrument remains explicitly experimental. Ordinary NVF contact
behavior and metal optical convergence remain unresolved; no default is
changed and no Li fallback is installed.

## Explicit use and remaining scope

```python
import numpy as np
from wavebit.rcwa._engine.conforming.cartesian_pair_opening import CartesianPairOpeningPlan
from wavebit.rcwa._engine.conforming.cartesian_pair_quadratic import CartesianPairQuadraticGapCell

geometry = dict(basis=np.diag([700., 700.]), widths=(300., 300.), height=300.)
plan = CartesianPairOpeningPlan.from_geometry(**geometry, refinement=1)
cell = CartesianPairQuadraticGapCell(plan, order=(2, 1), coercivity_angle=1.53)
slots, gap_derivative, metadata, details = cell.closure(
    gap=0., epsilon=(12.25, -10.+1j), **geometry)
```

The returned slots have the existing layer axis and feed `solve_conv`.
Derivatives are per caller length unit. Coefficient ordering is the same
nine-component ordering as the [finite gap family](nvf-cartesian-pair-gap.md).
Detailed fields now have shape `(triangles, 3, 2, 2)`: the three values are at
the geometric vertices and define a linear field, not three separate
constant subtriangles.

The [material-sector admission](nvf-pair-material-sector.md) and geometry
constraints are unchanged. The angle is a certificate only. All lengths,
widths, height, offset and materials are fixed when taking the analytical
gap derivative. Mixed geometry/material derivatives and automatic
integration into public NVF routing remain outside this instrument.

No new dependency, external solver or copied third-party implementation is
introduced. The triangle identities, six-node construction and reference
polynomials are implemented directly with the existing NumPy dependency.
