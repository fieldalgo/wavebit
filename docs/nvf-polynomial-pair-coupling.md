# Polynomial pair fields with real contact traces

Written: 2026-09-15

This page records the initial polynomial coupling and its measurements.
The subsequent [analytic remainder formulation](nvf-analytic-polynomial-traces.md)
resolves the order-8 contact refusals recorded below. Together with selective
refinement of unresolved equations, it also resolves the recorded refined
tiny-gap refusal.

Targets:
- `src/wavebit/_core/conforming/cartesian_pair_seam_galerkin.py`
- `src/wavebit/_core/conforming/projected_pair_contact.py`
- `src/wavebit/_core/conforming/projected_pair_positive.py`
- `src/wavebit/_core/conforming/pair_contact_energy.py`
- `src/wavebit/_core/conforming/real_trace_coupling.py`
- `tests/test_nvf_pair_seam_galerkin.py`
- `studies/nvf_coupled_pair_seam_galerkin.py`
- `studies/results/nvf_polynomial_pair_contact.log`
- `studies/results/nvf_polynomial_pair_positive.log`

TL;DR: Cubic and quartic nodal potentials now couple to the same continuous
real internal-gap traces as the quadratic family. Nearly coincident fields
require a real change of coordinates and direct physical energy assembly.
Every original equation and physical guard remains in force. This extends
the finite experimental family; it does not qualify production NVF or metal
optical convergence.

| Property | Contract |
| --- | --- |
| Polynomial degree | 2, 3 or 4; default 2 |
| Dense mesh cap | degree times 2**refinement at most 16 |
| Geometry | Internal horizontal equal-height pair with a positive diagonal period |
| Live directions | Gap, both complex materials and background simultaneously |
| Trace coordinates | Real, geometry-only and invertible, with every column retained |
| Contact solve for degrees 3/4 | Direct projected physical fields in streamed real QR coordinates |
| Backend | Explicit NumPy kernel |
| Qualification | Original-equation verification is separate from mesh/Fourier optical convergence |

## Preserve the entire field space

The original continuous real trace potentials are described in the
[quadratic coupling derivation](nvf-coupled-pair-seam-galerkin.md). Higher
nodal degrees nearly reproduce their normal trace and body continuation.
Solving in those nearly repeated coordinates generates unnecessarily large
coefficients and loses digits when reconstructing a field.

Let C denote surviving nodal potentials, sZ the disappearing physical nodal
potentials, L the original added trace potentials, and s the normalized gap.
The new columns are

```text
L_new = (L - s*C*A - s*Z*B)*T.
```

B projects the leading normal trace onto the disappearing nodal gradients
using a homogeneous unit-material metric. A projects the remaining body
continuation onto the common nodal gradients. T is the inverse upper factor from a streamed real QR of the remaining
weighted leading normal fields. Forming their Gram matrix first would square
the condition before choosing the new coordinates.
The residual fields are formed before their Gram products. All three
matrices are real and depend only on fixed geometry and the selected finite
spaces. The unit-material metric uses its own zero coercivity angle; it does
not inherit the physical metal sector.

This triangular coordinate change is invertible at every positive gap and
retains the same normalized contact limit. It introduces no fitted complex
material frame, rank threshold or removed mode. Its exact local variation
includes the s*C term as well as s*Z. Both adjacent material sides retain
one-half of the shrinking tangential taper's endpoint measure.

## Assemble the difference as a field

The surviving contact equations contain the high-mode field

```text
Psi_C = gradient(C) - Phi*Q_C.
```

Computing its stiffness by subtracting the already integrated matrices
K-SQ-Qtest*(R-AQ) loses precision when the two fields nearly coincide.
Instead, the implementation forms the real field Psi_C at integration
points, applies a streamed QR factorization of its geometric weighted rows,
and directly integrates material products in those coordinates. Each QR
update holds one triangle's rows and the current upper factor. No cell-wide
sample matrix is stored, and no singular value is truncated.

The symmetric retained Fourier set makes the geometric projection real:
positive and negative frequencies are equivalent to real cosine and sine
columns. A reality guard checks this identity before taking the real part.
Materials never enter that operation. The invertible QR coordinate transform
can be frozen for a local derivative: it still spans the same varying original
field space, so its arbitrary coordinate variation is unnecessary.

Writing M and S for the direct material gradient energy and Fourier source,
and D for all normalized disappearing nodal and added trace fields, the
complete contact solve is

```text
M_CC*u_C = -S_C
M_DD*u_D = -S_D - M_DC*u_C
M_CC*du_C = -dS_C - dM_CC*u_C - ds*M_DC.T*u_D.
```

The whole tensor and its variation follow from these solved fields. The
implementation then recovers the original nodal potentials, retained field,
and their derivatives. It checks the original weak equations, finite contact
row derivative, retained constraints and flux equations, including their
variations. It does not replace a tensor block or splice an energy derivative
onto a separately solved tensor.

`closed_elimination='physical_projected'` identifies this formulation.
For positive gaps, `elimination='physical_projected'` identifies direct
assembly of all projected nodal and trace fields in streamed real QR
coordinates. The positive diagonal square-root-gap scale is invertible and
is also frozen for the local variation.

At contact, `condition` describes the closed material stiffness in the real
QR coordinates; at a positive gap it describes the full material stiffness
in those coordinates. The quadratic family retains its previous elimination and
condition definitions. `physical_quadrature` reports the contact solve and
verification order: at least max(32, 4*max(order)+16, quadrature).
`integration_quadrature` records the moving/contact trace integration order,
at least max(4*max(order)+16, quadrature) for degrees 3/4. The requested
`quadrature` remains visible separately.

## Independent physical checks

The contact energy verifier receives only the original potentials, retained
field and fixed-coefficient geometric field variation. It receives neither
the candidate tensor nor its derivative. It reconstructs the complete
physical field before multiplying it, retains the vanishing-domain energy
and both taper endpoint sides, and integrates the complete small physical field directly. Correcting only
its free Fourier component would amplify a roundoff-sized quadrature
difference by large cancelling coefficients. The solve uses an independent
exact rectangular Fourier transform, while original retained/flux checks
compare against the original triangle assembly.

The independent positive-gap reference uses cardinal nodal polynomials,
raw sine/cosine traces, convex polygon clipping and separate quadrature.
For degrees 3/4 it directly orthogonalizes the sampled full high-mode fields;
all reference columns survive. It shares no production Bernstein gradients,
trace coordinate transform, moving sweep or contact formula. Quadratures
16 and 24 check the tensor, and two physical finite-variation step sizes
check its gap derivative under the original tolerances.

Further controls cover the closed nodal value, the square-root gap remainder
of the opening derivative, complex material holomorphy and simultaneous
forcing, uniform media, units, translation, unequal directional orders and
invalid degrees/refinements. The refined quartic order-4 contact is checked
at quadratures 32 and 40. A deliberately inconsistent returned tensor
derivative must fail the original/energy/physical guards.

No original residual guard exceeds 1e-10, no reciprocity guard exceeds
1e-9, rotated storage must stay positive, and the existing dissipation
allowance is unchanged. Numerical runs use NumPy 2.5.1 on Apple Silicon.
No external solver code or new dependency is used.

## Differentiate normalized positive-gap rows before integration

A normalized added row contains integrals such as the physical trace energy
divided by s. Applying a quotient rule after integration subtracts large,
near-equal terms at a subpixel gap. On collapsing triangles the sweep instead
keeps the width as s times its exact geometric coefficient and differentiates
the normalized measure directly. In regular body slabs the trace is s*F;
F and its derivative are evaluated analytically before integration. The
physical disappearing gradient s*gradient(Z) is also differentiated before
any division by s. These are exact factors of the existing field space.

The projected positive-gap solve returns the complete original state.
Up to two residual corrections use its physical QR stiffness to solve the
same original differentiated equations. Refinement stops when all three
original residuals are below 1e-11, ten times stricter than the acceptance
guard, so it does not amplify native matrix roundoff in an already accurate
physical solution. Each correction includes the weak,
retained-field and flux residuals together, and updates all potentials,
retained fields and fluxes together. Correcting only the weak residual would
leave the initial flux residual unchanged and spoil the energy variation.
The original three equations, original nodal reactions, stationary physical
energy variation and reciprocity are all checked at their existing tolerances.

## Recorded finite accuracy and optical limits

The contact record contains sixteen completed order-4 controls: degrees 3/4,
refinements 0/1, equal dielectric and dielectric/metal materials, and both
polarizations. The source hashes match the implementation. The fixture is
the same 700-by-700 cell, widths 300/300, height and film thickness 300,
wavelength 900, incidence theta 0.2 and phi 0, top 1, bottom 2.25, and
background 1 used by the [quadratic study](nvf-coupled-pair-seam-galerkin.md).
Equal materials are (4, 4); metal materials are (12.25, -10+1j). The common
coercivity angle is 1.53. Optical slopes use centered variations of the full
analytic contact tensor JVP at steps 0.00005 and 0.000025.

| Degree | Refinement | Metal contact T, p | Metal contact T, s | Contact dT/dgap, p | Contact dT/dgap, s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0 | 0.417863759 | 0.561894924 | -0.039714989 | -0.211042806 |
| 3 | 1 | 0.460712885 | 0.397890361 | -0.020483593 | -0.018586592 |
| 4 | 0 | 0.432212777 | 0.470957986 | 0.001927992 | -0.016452885 |
| 4 | 1 | 0.463091993 | 0.404495071 | -0.150169176 | 0.035682594 |

For the formerly rejected quartic/refinement-1 metal contact, the closed
physical stiffness condition in QR coordinates is about 82.46. The original
weak and contact derivative residuals are below 3.58e-14 and 2.09e-15;
direct physical energy value and derivative residuals are below 3.09e-13
and 4.61e-14. These results establish the finite solve, not optical convergence.
The optical slope changes above remain too large to support a mesh or
Fourier convergence claim.

The positive-gap record deliberately preserves the incomplete run at gaps
0/0.001/1e-9. At order 4, degree 4 and refinement 1, the metal case passes
contact and gap 0.001, then fails the original/energy/physical guards at
1e-9. A separately scoped run at gaps 0/0.001 records both optical
polarizations. The passing positive-gap weak derivative and energy derivative
residuals are about 1.01e-12 and 1.66e-13. This does not qualify the rejected
smaller gap; that unresolved case is not treated as a completed control.

The finite tensor and variation checks cover degrees 2 through 4 against
independent cardinal/clipped fields. Their smaller-order gap sequences do
reach 1e-9 under the unchanged guards. This difference in scope matters:
passing a low-order limit sequence does not certify the refined high-order
sequence, and a refused result is not silently replaced with another physical
model. Li remains an explicitly labelled optical comparison only.


The same contact record also includes the incomplete order-8/refinement-0
runs. Both cubic material cases and the quartic equal-dielectric case are
refused by the unchanged physical guards. The quartic metal case completes
both polarizations. It does not turn the other refused controls into passed
ones. Increasing the retained order therefore still exposes a numerical
qualification boundary in this experimental family, in addition to the
unresolved optical mesh dependence.


The initial coupling regression set contained 96 passing pair/periodic
controls, including the refined quartic contact and gap-0.001 controls.
The 56 documentation checks and the warnings-as-errors Sphinx build pass.
These counts exclude the unresolved high-order tiny-gap qualification target;
its refusal and requested coverage are preserved in the study record above.
