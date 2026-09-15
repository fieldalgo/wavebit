# Complete precision for the free contact Fourier block

The compensated contact solver now retains the low parts of its free Fourier
matrix and material/opening variation, together with those of its common weak
matrices. All entries come from the same original triangle measures. The
physical energy and equation guards retain their existing limits.

Targets:
- `wavebit/_core/conforming/pair_complex_precision.py`
- `wavebit/_core/conforming/triangle_exponential_precision.py`
- `wavebit/_core/conforming/pair_contact_fourier.py`
- `wavebit/_core/conforming/pair_contact_precision.py`
- `tests/test_nvf_contact_fourier_precision.py`
- `tests/test_nvf_pair_compensated_product.py`
- `tests/test_nvf_transmission_common_precision.py`

## A small free-block error can survive an otherwise accurate solve

The complete field is a sum of retained Fourier fields and common potential
gradients. On a high-degree trial space, these two contributions can each be
large while their sum is moderate. Retaining low parts of K, Q and R while
rounding the free block A before forming the projected equations leaves the
term Q.H A Q vulnerable to amplified roundoff.

An internal degree-5, order-4 transmission control had a Fourier coefficient
norm of about 52333. Its fully compensated common solve still failed the
original 1e-10 energy-value guard with a residual of 4.83311e-9. Reconstructing
the entire physical field with compensated potential and phase products gave
4.83308e-9, so merely improving that independent verifier did not resolve the
failure.

The free Fourier matrix differed from independently integrated high/low
moments by at most 7.85e-16. The energy error predicted directly by

```text
x.H @ (A_precise - A_rounded) @ x
```

had relative norm 4.83312e-9. This identifies the free block as the source of
that energy defect. These finite arithmetic checks do not establish optical
convergence. The supported degree contract is described in the
[pair guide](guide/cartesian-pair-gap.md).

Keeping the independently integrated low parts of A and its variation in
that same complete system reduced the energy-value residual to 1.04e-12 and
the energy-variation residual to 1.18e-12. Every original physical guard and
the unchanged full-tensor reflection checks passed. The calculation retained
all degree-5 common fields and every selected transmission direction.

## One expression for distinct and coincident phases

For a triangle with vertex phases z0, z1 and z2, the reference-simplex integral
of exp(sum(b_j*z_j)) is the second divided difference of the exponential.
It is also the top-right entry of

```text
exp([[z0, 1, 0],
     [ 0,z1, 1],
     [ 0, 0,z2]]).
```

The implementation shifts out z0, scales the remaining triangular matrix by
an exact power of two, evaluates its Taylor series, and squares back. The
scaled row norm is at most 1/4; 27 Taylor terms put its analytic truncation
below the retained arithmetic error. A simultaneous directional calculation
propagates all moving vertex phases through the same products and squares.

There is no subtraction of nearly equal exponentials and no division by a
phase difference. Exactly repeated phases and nearly coincident phases use
the same formula. The unweighted zero-phase moment is 1/2 and its phase
variation is the sum of the three vertex directions divided by 6.

Only distinct Fourier *differences* are evaluated, followed by an exact lookup
for every matrix entry. This avoids repeated arithmetic without deleting a
retained direction. Triangle areas and their opening coefficients are formed
from the shared physical vertices in complete arithmetic. All complex material
directions are consumed directly,
including zero-area triangles with a nonzero opening measure. The latter
retain the high/low product of the caller-unit direction and normalized
area coefficient, matching the disappearing-field energy scaling. The
[complete energy-coordinate note](nvf-contact-energy-coordinates.md) describes
why rounding that product early is inconsistent.

The high/low free moments enter the full projected stiffness, its variation,
and the complete response before rounding. The original physical checks
receive the rounded directly integrated matrix. No candidate tensor or JVP
is projected, averaged or replaced to satisfy a check. Existing NumPy and
standard-library arithmetic suffice; no external code or dependency is added.

## Independent checks and cost

Eleven phase fixtures compare both values and variations with independent
400-digit scalar divided differences. They include three repeated phases,
each permutation of a repeated pair, separations of 1e-80, and phases up to
100 radians. The maximum component discrepancy was 5.08e-32, against a
predeclared 3e-28 bound.

Full free matrices and joint material/opening variations are also compared
with independent 100-digit rectangle integrals at orders (0,0), (1,2) and
(1,1), including a refined mesh, using the actual mesh without replacing its
stored area records. A separate constant-mode check compares against independent
Decimal determinants of the physical vertices and their variations. Homogeneous
material checks on a non-dyadic body partition verify the unit Fourier Gram
matrix before and after refinement at the same 3e-28 bound.
Four scalar complex exponential controls cover the scalar and complex cases.

For the order-4 diagnostic, independent smooth Gaussian integration at orders
32 and 40 agreed in every free-matrix entry within 1.04e-30 and in the
variation within 4.71e-32. The direct triangular expression agreed with the
order-40 result within 4.64e-31 and 7.16e-33 respectively. In these local
measurements, the order-40 Gaussian calculation took about 169 seconds and
the direct expression took 4.02 seconds. These times concern the free moments,
not the complete constitutive solve.

This change addresses arithmetic consistency. It does not certify joint
polynomial, mesh, corner-field or Fourier convergence of an optical result.

The free-moment controls and seven existing complete-common regressions
pass. The latter retain the original non-square metal contact, full joint
reflection, quadrature comparison and uniform-material checks.


## Complete matrix products and a smaller projection regression

For normalized high/low inputs, the leading high-by-high product needs its
full error expansion. Each high-by-low or low-by-high product already carries
one small factor, so ordinary multiplication introduces only second-order
rounding on the scale of the original dot product. The implementation retains
both cross products and low-by-low, then combines them with the compensated
leading product before rounding. It removes no field, entry or cross term.

Independent 100-digit decimal dot products check real and complex matrices
with alternating cancelling terms at inner dimensions 1, 21, 64, 161 and 512.
The predeclared error bound is 3e-28 on the summed absolute-product scale.
In a paired, alternating local timing experiment, the complete product was
3.54 times faster for a real (161, 1024, 42) multiplication and 2.48 times
faster for a complex (162, 512, 42) multiplication. These are product timings,
not an end-to-end solver speedup claim.

A smaller actual library regression isolates the quintic polynomial/Fourier
space using a flat auxiliary reference (2, 2, 2), Fourier order (4, 0), physical
materials (12.25, -10+i, 1), period 700 and widths/height 300. It uses radial
order 100 and angular order 24. With complete free moments, the independent
pointwise energy identity and all original physical guards pass. Replaying
the same complete system with the legacy rounded free block fails the
1e-10 energy guard at 9.65e-10, matching the separately predicted 9.67e-10.
This regression protects the free-moment integration into the full solver,
in addition to the scalar and matrix moment tests.

The complete contact path also forms each determinant and opening coefficient
from the physical vertices before rounding. All free, common and independent
energy integrals share these measures. The original/refined unit-cell Fourier
identities and the directly integrated retained-field check are described in
the [contact-coordinate note](nvf-contact-energy-coordinates.md).
