# Analytic remainders for polynomial pair traces

Written: 2026-09-15

The subsequent [guarded contact solver](nvf-contact-solver.md) tries complete
retained elimination before physical QR, using the same trace space and
unchanged original acceptance criteria. The measurements below used the
physical projected solver directly.

Targets:
- `src/wavebit/_core/conforming/polynomial_pair_trace.py`
- `src/wavebit/_core/conforming/cartesian_pair_seam_galerkin.py`
- `tests/test_nvf_polynomial_trace.py`
- `tests/test_nvf_pair_seam_galerkin.py`
- `studies/nvf_coupled_pair_seam_galerkin.py`
- `studies/results/nvf_analytic_polynomial_trace_contact.log`
- `studies/results/nvf_analytic_polynomial_trace_positive.log`

The cubic and quartic pair family now evaluates nearly repeated polynomial
and trigonometric fields as analytic remainders. This preserves every original
trial direction and resolves the previously refused order-8 contact controls.
Selective refinement of unresolved equation groups also resolves the refined
order-4 gap-1e-9 control. Neither numerical improvement establishes optical
mesh or Fourier convergence.

## The small difference is a physical field

The [previous coupling formulation](nvf-polynomial-pair-coupling.md) projects
real trace fields onto nodal gradients before normalizing their remainder.
For a cubic element and eight retained tangential orders, the largest
normalizing singular value was about 1.49e9. Sampling the two nearly equal
functions separately had already lost digits before that normalization.

Let h be the normalized pair height, k=pi/period_y, a=sin(k*h/2), and
u=sin(k*y)/a. The same real Fourier space is spanned by even Chebyshev
polynomials T_(2j)(u) and cos(k*y) times odd polynomials T_(2j+1)(u).
The pair profile f(x) is piecewise linear, with the original positive-gap
and body slopes. The potentials

```text
f(x) * (y*y - h*h/4) * indicator(|y| < h/2)
f(x) * y*(y*y - h*h/4) * indicator(|y| < h/2)
```

belong exactly to the original cubic and quartic nodal spaces, respectively.
Both vanish at the horizontal endpoints and at the outer body faces. Their
restrictions remain polynomials of the admitted degree on every original
triangle, including the collapsing annulus. Subtracting these nodal functions
from selected added traces is an invertible change of the complete space.
It does not delete a mode or introduce a new material-dependent field.

For B2(y)=y*y-h*h/4, write its Chebyshev series as sum(c_j*T_(2j)(u)).
For j>=1 its coefficients are positive sums:

```text
c_j = (1/k**2) * sum_(n>=j) a**(2*n)/n**2
        * binomial(2*n,n-j)/binomial(2*n,n).
```

The binomial ratio is evaluated as a product of bounded positive factors.
If P_N is the truncation through j=N, then P_N-B2 is evaluated as the
negative tail through j>N. The implementation never subtracts two sampled
cell-sized values to obtain this small remainder. Its value and first two
spatial derivatives use the same polynomial recurrence.

For the quartic odd bubble B3=y*B2, the identity

```text
B3 = k*cos(k*y)/(4*a) * d(B2**2)/du
```

supplies the odd remainder. A positive geometry-only coefficient scales each
remainder, followed by a real QR frame. The remaining nodal projection and
streamed QR retain all columns. For the cubic order-8 fixture, the largest
remaining normalizing singular value is about 2.08 rather than 1.49e9.

The power-series tail is bounded over the entire maximum taper interval,
including points outside the nominal height. Bounds for three derivatives
of the even series and the differentiated-square identity cover both
remainders and their first two spatial derivatives. The retained series is
extended until its bound is below floating-point roundoff relative to the
remainder scale. Failure to establish that bound is explicit. Truncating this
convergent evaluation series does not remove any retained Fourier direction.

## Preserve the positive-gap taper and its derivative

Subtracting a nodal bubble also changes the potential inside the taper.
For each selected column the implementation evaluates

```text
hat*P_N - indicator*B = hat*(P_N-B) + (hat-indicator)*B.
```

The small endpoint factor of B comes from the known taper fraction and width.
Inside the nominal body, hat-1 is evaluated as -(1-fraction)**2*(1+2*fraction),
without subtracting two nearly equal values. Outside, the factor is hat.
The implementation differentiates this complete expression, including the
moving taper, the polynomial cap, and the spatial derivatives of the
remainder. At contact the polynomial cap vanishes; its endpoint jump equals
the original trace jump because B is zero there.

All normalized weak equations, retained constraints, fluxes, independent
physical energy, reciprocity, storage and dissipation guards retain their
previous thresholds. The quadratic path and ordinary NVF dispatch are
unchanged. The explicit kernel continues to use NumPy; no dependency or
external solver code is added.

## Independent verification and measured limits

The remainder tests compare both parities, including first and second spatial
derivatives, against 70-digit decimal subtraction with independently evaluated
inverse sine and integer binomial coefficients. Orders 2, 4 and 8 are covered.
Independent cardinal gradients check that both subtracted bubbles belong to
every original triangle at refinements 0 and 1 and gaps 7 and 0.001.
The existing independent clipped-domain reference still uses raw sine/cosine
traces and checks the complete tensor and physical finite variations.

The contact record contains twelve completed optical rows: degrees 3/4,
order 8, refinement 0, equal dielectric and dielectric/metal materials, and
both polarizations, plus both metal polarizations at refinement 1. The fixture and optical solver are those documented in
[the polynomial coupling measurement](nvf-polynomial-pair-coupling.md).
The largest original/physical residual over these controls is below 5.45e-13.
The old refused records remain available with their original source hashes.

| Degree | Refinement | Metal T, p | Metal T, s | Contact dT/dgap, p | Contact dT/dgap, s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0 | 0.427763247 | 0.430755451 | -0.443584124 | -0.175619750 |
| 4 | 0 | 0.460097783 | 0.397736730 | -0.389007719 | 0.000539585 |
| 3 | 1 | 0.462504347 | 0.400914471 | 0.020899510 | 0.111939483 |
| 4 | 1 | 0.470038603 | 0.397210477 | -0.158591627 | -0.251659903 |

The slope columns use the smaller of the two recorded optical variation
steps. Their changes with nodal degree and Fourier order remain substantial;
these are finite accuracy controls, not an optical stopping rule.

## Refine only unresolved equations, while updating the entire state

At a refined tiny gap, the original added-trace derivative rows can still need
residual refinement after the physical QR solve. The original common and
disappearing nodal rows, retained constraints and flux equations can already
be accurate. Feeding their accepted matrix roundoff back through a nearly
coincident closed basis corrupts an otherwise accurate physical derivative.

The correction therefore checks the original C, Z and added-trace equation
groups separately against the existing 1e-11 refinement target. A group below
that target contributes zero forcing to the correction. Retained and flux
residuals use the same rule. The full coupled stiffness then solves the
correction, and every potential, retained field and flux derivative is updated
together. No unknown is frozen or removed. All original equation and energy
guards are applied again to the complete resulting state at their unchanged
1e-10 threshold. This is an arithmetic refinement decision, not a new physical
constraint or a replacement derivative.

The positive-gap record preserves the incomplete initial run and the subsequent
complete run for order 4, degree 4, refinement 1, metal materials and gaps
0, 0.001 and 1e-9. Before the selective correction, the final gap's energy
variation residual was about 1.58e-10 and was refused. Re-evaluating the energy
with full physical fields and factored gradients did not remove that error.
The corrected complete solve passes with energy derivative residual 5.73e-11,
weak derivative residual 2.83e-16, and derivative reciprocity defect 6.66e-13.
That measurement used the original expanded energy verifier.

For positive-gap projected states, verification now reconstructs the complete
physical field and its stationary variation before taking their products.
It consumes the analytically scaled fields `s*GZ` and `d(s*GZ)` from the
normalized quadrature, avoiding a subtraction of inverse-gap terms. The
stationary variation holds auxiliary coefficients fixed and does not consume
the candidate tensor JVP or potential JVP. It includes moving measures,
complex material directions, the retained constraint variation and every
continuous trace field. The original equations, reciprocity, physical guards
and acceptance tolerances remain the same; the computed tensor and JVP are
unchanged. This improves the representation of the independent check and
does not certify field or optical convergence.

For this tiny gap, the full tensor changes from contact by about 5.95e-7;
the contact derivative norm is about 595.023, consistent with the expected
first-order change at gap 1e-9. The positive-gap derivative differs from
contact by about 0.00380 in norm. These are finite contact-limit checks of
this fixed model. The optical dependence on Fourier order and mesh degree in
the table above remains unresolved, as do broader geometry and backend
qualification. No failed response is returned as valid or replaced with Li.

The analytic-remainder regression suite passed 84 controls. After selective
refinement was added, all 20 affected positive-gap controls passed, including
the newly restored refined gap-1e-9 target. These sets overlap; the final
pair/remainder suite contains 85 distinct controls. The 57 documentation
checks and warnings-as-errors Sphinx build also pass.

## Twice-refined closed fields

The subsequent source-pinned record
`studies/results/nvf_polynomial_pair_refinement_two.log` contains eight
completed polarization rows: cubic and quartic fields, Fourier orders 4
and 8, and refinement level 2 for the same dissimilar metal pair. Each
degree has its own source header. Cubic measurements predate the guarded
solver and flat field-product optimizations; quartic measurements use them.
Those optimizations preserve the same finite equations and acceptance limits.

| Degree | Fourier order | T, p | T, s | Contact slope, p | Contact slope, s |
| --- | --- | --- | --- | --- | --- |
| 3 | 4 | 0.462741504 | 0.412446823 | -0.119876028 | 0.007393669 |
| 3 | 8 | 0.470190520 | 0.379490252 | -0.168592579 | 0.015198594 |
| 4 | 4 | 0.473280359 | 0.434992756 | -0.063208885 | 0.049665974 |
| 4 | 8 | 0.476993193 | 0.376108368 | 0.024792239 | -0.010924138 |

Both quartic runs reject the original retained elimination and accept the
complete physical QR state. The largest original/energy residual is
1.03e-12 at order 4 and 2.49e-12 at order 8, below the unchanged 1e-10
limit. Their actual integration orders are 32 and 48 respectively. The
constitutive value/JVP calls took about 234 and 919 seconds in these local
runs; this is still an expensive dense family.

The quartic order-4 mesh change from level 1 to level 2 changes s-polarized
transmission from about 0.404495 to 0.434993. At level 2, increasing Fourier
order from 4 to 8 changes it by about 0.058884, and changes the sign of both
reported contact slopes. Raising degree also changes the slopes substantially.
The additional resolution therefore does not yet demonstrate joint mesh,
degree and Fourier convergence. Small finite-equation residuals still do
not qualify these optical values or contact derivatives for ordinary routing.

The measurements establish the current finite result and its remaining
resolution dependence. They are not a proof that an infinite-resolution
limit fails to exist, and they do not justify replacing NVF with Li or
weakening the acceptance criteria.
