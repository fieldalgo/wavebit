# Physical rectangles in an oblique auxiliary cell

Written: 2026-09-10

Targets:
- `studies/nvf_cartesian_cell.py`
- `tests/test_nvf_cartesian_cell.py`
- `studies/results/nvf_cartesian_cell.log`

The experimental `CartesianRectangleCell` keeps a rectangle's physical
edges while its periodic lattice becomes oblique. A small conforming
piecewise-linear potential supplies the nine field, energy and material
Fourier coefficients used by the existing energy closure. Geometry and
material derivatives agree with an independently differentiated weak
system and moving-triangle quadrature. This establishes a differentiable
finite cell for one separated rectangle. It does not establish optical
accuracy or extend the public `ConformingNVF` geometry contract.

## Geometry and finite space

The lattice matrix A contains physical basis vectors in columns. A
rectangle has physical width, height and angle, with center
`A @ (.5, .5) + offset`. Its edges are constructed in Cartesian coordinates;
changing A does not shear those edges. The whole fundamental cell follows
the offset, which preserves translation covariance of this finite space.
This differs from the explicit fractional-profile contract of
[`AffineProfileNVF`](guide/affine-profile-nvf.md).

Two triangles cover the rectangle and eight cover the surrounding
parallelogram. Uniform red refinement divides every triangle into four,
sharing each edge midpoint. Opposite outer edges use the same periodic
unknowns; one potential value fixes the additive gauge. The topology,
interpolation weights and periodic identifications remain fixed when
numerical geometry values change.

| Refinement | Triangles | Free potential unknowns |
| --- | ---: | ---: |
| 0 | 10 | 4 |
| 1 | 40 | 19 |
| 2 | 160 | 79 |

These deliberately small spaces use dense linear algebra. Refinement is
limited to 0, 1 or 2. The rectangle must lie strictly inside the centered
cell, and every template triangle must have positive finite area.
Containment alone is insufficient for the default template: even adding
pi to a contained rectangle's angle can reverse a triangle. That input
is refused unless an appropriate explicit plan is supplied.

## Choosing and replaying an orientation plan

`CartesianRectanglePlan.from_geometry` selects among four cyclic
orderings of the rectangle's vertices using concrete NumPy geometry.
Each candidate retains the original ten triangles and annular diagonals.
The planner maximizes the smallest signed triangle area in fractional
coordinates among positive candidates. A `1e-12` fractional tolerance and
integer-quantized vertex keys resolve planning ties. This tolerance does
not relax the strict containment or positive-area guards.

```python
from studies.nvf_cartesian_cell import CartesianRectangleCell, CartesianRectanglePlan

geometry = dict(basis=[[700., 315.], [0., 490.]], width=350., height=300.,
                angle=3.45, offset=(49., -21.))
plan = CartesianRectanglePlan.from_geometry(**geometry)
cell = CartesianRectangleCell((1, 2), refinement=1, plan=plan)
coefficients, metadata = cell.coefficients(**geometry, epsilon=4.+.8j,
                                         background=1.+.1j)
```

The immutable plan stores only a cyclic permutation. Geometry and
materials remain live numerical inputs, while interpolation weights,
connectivity and periodic identities stay fixed during replay. The
returned metadata includes `inner_permutation` for explicit plans.
`plan=None` retains the original template and metadata. Planning is
separate from JAX tracing: derivatives apply to the chosen finite space,
with no derivative or continuity claim across planning ties. A live
update that invalidates the chosen triangles raises eagerly or returns
all-NaN coefficients under tracing; changing charts requires explicit
replanning. The replay guards do not detect every change of parameter
representation: a different representation can still describe positive
triangles with a different finite triangulation.

A deterministic geometry-only scan covers 15,120 strictly contained
rectangles across lattice skews, lattice heights, aspect ratios, angles
and containment fractions. The default template is positive for 6,981
cases; the four cyclic choices admit all 15,120 sampled cases. Separate
plans for angle-plus-pi and angle-plus-pi/2 with swapped dimensions
produce the same physical meshes in all 30,240 comparisons. Fractional
ranking also preserves joint rotation and common length scaling in the
tested controls. This sampled coverage is not a general triangulation
theorem. Contact continuation, multiple rectangles and crossing periodic
interfaces remain outside this implementation.

## Weak equations and physical coefficients

For each Cartesian unit field, the periodic potential corrector solves

```text
K_ij = sum_e epsilon_e area_e grad(N_i) . grad(N_j)
F_ia = -sum_e epsilon_e area_e partial_a(N_i)
K u = F
B_e = I + grad(u)_e
G_e = epsilon_e B_e^H B_e.
```

The gradients are physical Cartesian gradients on each triangle. The
complex weak matrix is symmetric without complex conjugating its scalar
permittivity. Both materials must have positive real permittivity and
nonnegative imaginary permittivity. Diagonal equilibration changes the
linear solve's conditioning, while its residual is checked against the
original system, including the gauge row.

Every triangle has constant B, G and epsilon. The existing analytic
polygon transform integrates each coefficient with the phase
`exp(-2 pi i (m,n) A^-1 r)` and divides by cell area. The component order
is `Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy, epsilon, Gyx`. The two off-diagonal
G entries remain independent for lossy material. Unequal or zero Fourier
half widths retain the caller's exact difference-mode set.

`coefficients` returns these arrays and finite-space metadata.
`conv_matrices` selects the exact difference-mode matrices before applying
`ConvMatrices.from_energy` and returns one layer of solver slots. Its live
guards require positive storage, nonnegative dissipation to the existing
`1e-9` roundoff tolerance, and a Hermitian lossless tensor. The cell does
not change the consuming solver's lattice or Fourier labels.

## Independent derivatives

The derivative control accumulates the original element matrices by
nodal indices, without using the implementation's restriction tensor or
equilibrated solve. For a triangle Jacobian J it uses

```text
d grad(N) = -grad(N) dJ J^-1
d area = area trace(J^-1 dJ)
K du = dF - dK u.
```

An independent Duffy/Gauss integration includes moving triangle points,
the derivative of physical area and the derivative of the reciprocal
wavevector. Its Fourier phase variation contains both `dk . r` and
`k . dr`, with row-vector `dk = -k dA A^-1`. It differentiates the
conjugate field in G as well as the live scalar material.

The fixed test fixture uses A = `[[700,315],[0,490]]` nm, a 350 by 300 nm
rectangle, physical offset `(49,-21)` nm, inclusion `4+.8j` and background
`1+.1j`. Thirteen real parameters cover all four lattice entries, width,
height, angle, two offsets and both real/imaginary parts of each material.
Length directions are divided by 700 nm; angles and material directions
are dimensionless. Both signs of the tested small perturbations stay
inside the passive material domain.

The frozen preliminary comparison contains 25 direction cases: mixed,
translation, joint scale and joint rotation at each refinement, plus all
13 coordinate directions at refinement 1. Three homogeneous and weak
contrast cases independently check the resulting constitutive derivative.

| Check | Predetermined absolute criterion | Largest preliminary discrepancy |
| --- | ---: | ---: |
| Independent derivative quadrature, 16 to 24 points per Duffy axis | `3e-10` | `1.276e-14` |
| NumPy coefficients versus independent quadrature | `3e-10` | `2.453e-15` |
| JAX JVP versus independent derivative, 28 cases | `3e-9` | `1.550e-14` |

The same compiled JAX function returns all-NaN coefficients for six
invalid live updates covering lattice orientation, containment, template
orientation, material passivity and nonfinite offset. These are tests of
the admitted finite space and its validity checks.

The initial self-contained study module is separately tested after this
frozen preliminary comparison. All 22 initial controls pass in 3.27 s,
including the independent derivative oracle, physical periodic pairs,
analytic material transform, homogeneous constitutive derivatives,
translation/scale/rotation covariance, lossless and lossy passivity, and
exact closure selection for `(0,0)`, `(0,1)`, `(2,0)`, `(1,2)` and `(2,1)`.
Source hashes are unchanged across the run. This test duration includes
JAX compilation on the Apple M4 host with 16 GiB memory; it is not a solver
performance comparison.

With the optional orientation plan, all 38 controls pass in 5.42 s.
The rotated planned fixture uses angle `pi+.31`; its additional 25
directions include all 13 coordinates at refinement 1 and mixed,
translation, scale and rotation directions at all three refinements.
The independent quadrature and JVP criteria remain `3e-10` and `3e-9`.
Equivalent physical representations also agree in the coefficients and
closure slots, and invalid stale-plan replay returns NaNs under JIT.
Module and test hashes remain unchanged across this run. A separate
comparison with the initial implementation finds bitwise-identical
default coefficients, closure slots, metadata and topology arrays for
all three refinements, both lossy/lossless materials and outer order
`(1,2)`. These are finite-space compatibility and derivative controls.

A supplementary five-point finite difference at dimensionless step
`.002` initially fails its `3e-8` criterion for a translation direction,
with error `3.494e-7`. The exact Fourier phase predicts the same stencil
truncation error to `1.321e-13`. Steps `.001` and `.0005` then pass all
17 checked directions under the unchanged criterion, with maximum
`2.185e-8`. This observation qualifies those finite-difference steps; it
does not change the independent quadrature or automatic-derivative gates.

## Scope of the result

The implementation uses the project's existing NumPy/JAX arithmetic and
polygon transform. The weak-system and quadrature controls are original
test code. No optional NGSolve runtime or external FEM source is imported
into this cell.

These small cells do not certify corner resolution, a continuum
derivative, contact behavior or Maxwell transmission. Their finite
spaces depend on the explicitly chosen triangulation; the four-choice
planner does not implement a general geometry adapter. The separate
[independent Maxwell study](nvf-independent-fem.md) remains the optical
qualification work.
