# Material-junction potentials in the complete pair tensor

Written: 2026-09-15

Targets:
- `src/wavebit/_core/conforming/cartesian_pair_corner_galerkin.py`
- `src/wavebit/_core/conforming/pair_corner_fields.py`
- `src/wavebit/_core/conforming/pair_corner_assembly.py`
- `src/wavebit/_core/conforming/pair_corner_contact.py`
- `src/wavebit/_core/conforming/pair_junction_partition.py`
- `src/wavebit/_core/conforming/pair_junction_thin_partition.py`
- `src/wavebit/_core/conforming/pair_junction_transport.py`
- `src/wavebit/_core/conforming/pair_junction_quadrature.py`
- `tests/test_nvf_pair_corner_galerkin.py`
- `tests/test_nvf_pair_junction_quadrature.py`

`CartesianPairCornerGalerkinGapCell(..., junctions=True)` includes the two
material junctions at the ends of the closing gap, in addition to the four
outer corners. All six controls use the same real powers and exact rational
remainders described in [the corner-family note](nvf-pair-corner-potentials.md).
The default `junctions=False` retains the outer-corner family.

The full positive-gap tensor, its joint gap/material JVP, exact contact tensor
and right-opening derivative share the enlarged space. This explicit NumPy
family still requires the fixed Cartesian opening plan, positive diagonal
period, horizontal equal-height pair, degree 3 or 4 and distinct exponents
chosen from `(0.6, 0.8)`. Empty exponents retain the original polynomial and
continuous-trace space. Ordinary NVF dispatch is unchanged.

## The thin triangles carry common junction fields

A junction hat can be one at both ends of a collapsing edge. Its common
potential has a finite tangential field on the collapsing triangle. The
assembly therefore includes its common/common and common/disappearing
products, Fourier moments and material variations there. The two exterior
annulus triangles also collapse; they are treated using their actual affine
geometry instead of assuming every thin point lies inside the gap.

The finite common gradient follows from the linear Jacobian coefficient:
`grad(C)=grad_ref(C)*adj(J1)/d1`. The leading inverse-gap term annihilates
closed common hats. The scaled disappearing field `s*grad(Z)` is evaluated
before dividing or differentiating. The same identities hold at exact contact.

Both ends of a thin triangle can be singular. Splitting the reference radial
interval into two halves and grading each from its own endpoint by `t^5`
resolves both. Small complementary barycentric coordinates are retained
directly. On regular triangles, corner charts include all six controls; a
triangle with three singular vertices is divided into six reference pieces.
Contact boundary integrals grade both singular endpoints when present. The
trace profile vanishes only at the outer corners, not at a material junction.

## Differentiate the integral in the original reference triangle

Moving angular quadrature nodes can produce large cancelling derivative terms
as a clipped region shrinks. The junction implementation evaluates the volume
variation at fixed original barycentric coordinates and includes the moving
boundary flux explicitly. For reference coordinates `(b1,b2)`, triangle
Jacobian determinant `D` and an oriented moving polygon `P`, the identity is

```text
d/ds integral_P D F
  = integral_P (D' F + D partial_s F)
    + integral_boundary_P D F (v1 db2 - v2 db1).
```

Material derivatives enter `partial_s F` together with the field variation.
Volume quadrature points have zero reference-coordinate derivative. Boundary
samples have zero volume weight and the signed flux as derivative weight.
Matching boundaries use identical canonical line points.

The potential and its gradient agree across the moving horizontal taper caps,
so their internal flux cancels. Vertical profile interfaces can carry gradient
jumps and retain both one-sided contributions. Original triangle edges are
fixed in reference coordinates. The method retains all terms of the same
integral; it changes its numerical differentiation coordinates.

The complete common/disappearing solve uses the square-root-gap coordinates
from the corner-family note, including their derivatives. The independent
stationary-energy verifier receives neither the candidate tensor/JVP nor the
solved potential derivative. All original nodal, weak, retained, flux, energy,
reciprocity, storage and dissipation guards retain their existing thresholds.

## Finite controls and scope

Measurements use NumPy 2.5.1, Python 3.12 and one BLAS thread on macOS arm64.
For period 700, widths `(300,300)`, height 300, permittivities
`(12.25,-10+1j)`, background 1, order 1, degree 3 and the unrefined plan,
both exponents pass every physical guard at positive quadrature 32. The
following contact differences use the independently integrated contact state
at quadrature 40. The class uses a contact quadrature floor of 48.

| Gap | Tensor norm difference from contact | Gap-JVP norm difference, multiplied by 700 |
| --- | ---: | ---: |
| `1e-3` | `1.0729116e-2` | `5.1468966e2` |
| `1e-7` | `1.0700941e-6` | `5.2202845e1` |
| `1e-9` | `1.0708106e-8` | `1.4528323e1` |

The largest independent stationary-energy derivative residual on these steps
is `1.15e-14`. The balanced matrix condition number remains about 124.
Fractional junction endpoint remainders approach contact more slowly than the
outer-only taper examples. These values establish finite-family checks;
they do not assign an optical error or a universal contact convergence rate.

On the asymmetric dielectric test geometry, quadrature 32 and 40 agree at
absolute tensor/JVP tolerances `3e-10`/`3e-9`. Fourth-order gap variations
with steps `3e-5` and `1.5e-5` give maximum JVP differences `4.12e-10` and
`6.13e-11`. Empty enrichment, a uniform medium, complex-linear material
variations, joint gap/material directions and length-unit changes also pass.
At order 1, degree 3/refinement 1 and degree 4/refinement 0 also pass the
metal contact control at quadrature 48 and positive gaps `1e-3`/`1e-7` at
quadrature 40. These are additional finite-space controls, not a joint
resolution study.

Independent beta-function integrals check both singular thin endpoints.
Exact rational polygon moments check moving profile areas, linear and
quadratic moments and their derivatives at refinements 0 and 1 through gap
`1e-9`, including both main-gap and annulus thin triangles. Deliberately
corrupting the response JVP must still fail independent energy verification.

Joint mesh, polynomial, Fourier and optical convergence, general geometry and
other backends remain outside these finite-family controls. No external
solver code or new dependency is used.
