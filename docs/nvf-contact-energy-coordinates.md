# Preserve contact coordinates through the physical energy check

Targets:

- `wavebit/_core/conforming/pair_contact_energy_precision.py`
- `wavebit/_core/conforming/pair_contact_precision.py`
- `wavebit/_core/conforming/pair_contact_quadrature.py`
- `wavebit/_core/conforming/pair_contact_normal_precision.py`
- `wavebit/_core/conforming/pair_contact_normal_mean.py`
- `wavebit/_core/conforming/real_trace_coupling.py`
- `tests/test_nvf_transmission_common_precision.py`
- `tests/test_nvf_contact_fourier_precision.py`
- `tests/test_nvf_contact_map_precision.py`
- `tests/test_nvf_contact_normal_precision.py`
- `tests/test_nvf_contact_normal_mean.py`
- `tests/test_nvf_pair_residual_arithmetic.py`

A complete compensated solve must also preserve its field coefficients when
passing them to the independent physical verifier. Rounding those coefficients
before reconstructing the field can fail the energy derivative even when the
internal finite equations are accurate. A separate consistency issue occurs
when collapsed-triangle free moments round their opening measure before the
normalized disappearing-field energies use that same product.

## The verification handoff is part of the precision contract

The complete physical field contains a retained Fourier contribution and
common potential gradients. Their individual coefficients can be large even
when the resulting field is moderate. In the sextic, order-(4,0) control
below, their norms were about 272,000 and 1,213,000. The disappearing-potential
coefficient norm was about 3,186,000.

The original pointwise check consumed ordinary rounded coefficients and
reported an energy-derivative residual of 5.70e-9, failing the existing 1e-10
limit. Improving the field arithmetic while still consuming those rounded
coefficients left 3.86e-9. An independent rearrangement of the stationary
energy using complete unprojected matrices gave 3.95e-9 with the rounded
coefficients, but 3.92e-26 with their preserved high/low values. This algebraic
comparison isolates the handoff error; it does not replace the physical check.

The solver now hands the pointwise verifier three complete coordinate pairs:

- the retained Fourier coefficients;
- their geometric variation with the common potential coefficients held fixed;
- every common and disappearing potential coefficient.

The verifier receives neither the candidate tensor, its derivative, nor the
solved variation of the common coefficients. It independently reconstructs
the full field and fixed-potential field variation at quadrature points,
including collapsing triangles and both moving boundary terms. All original
pointwise energy expressions and their tolerances remain in force. Products
and accumulated integrals retain high/low parts until the final energy arrays.

The retained Fourier field uses the same scalar phases in two Cartesian
blocks. Applying those blocks separately avoids multiplying their structural
zeros and avoids allocating a lifted phase matrix. It keeps every Fourier
coefficient and every high/low product. Independent 100-digit controls check
both components with cancelling terms for 1, 9 and 81 modes. A paired,
alternating seven-repeat timing of 512 quadrature points, 81 modes and 162
output columns measured a 1.45-fold reduction in this product's median time.
This is a product measurement, not an end-to-end solver timing.

Corruption controls independently alter the candidate tensor, its derivative,
the physical verifier's potential coefficients, or only its retained Fourier
field. The original energy or directly integrated Fourier-mean guard rejects
each change. When only verifier coordinates change, the original weak,
rounded retained and flux diagnostics still pass; those equations alone
cannot mask a broken handoff to the physical reconstruction.

## One opening measure for every collapsed-field term

On a collapsed triangle, the normalized area coefficient is `d1` and the
caller-unit opening direction is `ds`. Disappearing-field energies use the
product `ds * integral(d1 * field_product)`. Their free and common Fourier
moments must retain the same product, including its low part.

Previously, the free variation rounded `ds*d1` before integration. In the
sextic control, changing only that premature rounding altered a free-matrix
entry by at most 1.31e-20. Multiplication by the large retained coefficients
predicted a relative derivative-energy discrepancy of 1.12858e-10. The
observed discrepancy was 1.12861e-10; their difference was 3.93e-15 on the
same scale.

The first opening-measure correction preserved both scalar factors on
collapsed triangles while retaining the original bulk measure records.
The complete vertex-measure correction below additionally preserves the
determinant itself and its opening coefficient on every triangle. Free
Fourier variation, common moments, common opening energy and normalized
disappearing energies use the same measures. Geometry, field count, reference
materials and every physical acceptance threshold are unchanged.

## Complete vertex measures and physical Fourier constraints

The closed determinant and its opening coefficient are now formed from the
shared physical vertices in compensated arithmetic. Computing a determinant
in ordinary precision first can prevent the triangle areas from summing to
the exact cell area. On the 700-period, 300-wide/300-high pair, the unweighted
Fourier Gram defects are `3.90e-16` on the original mesh and `2.41e-16` after
one refinement. Complete vertex measures reduce these defects to `2.51e-31`
and `1.50e-31`. The Fourier test uses both directions at order `(1,1)`.

The same vertex determinants and complete opening coefficients supply the
free matrix, common weak matrices and independent physical energy. Changing
only one of these would compare different finite integrals. Independent
Decimal rectangle and joint-variation oracles now use the actual mesh without
replacing its determinant records. A homogeneous-material control on the
non-dyadic body partition checks the complete free matrix, its material JVP,
and the unit-cell Fourier identity against their exact values at `3e-28`.

The independent verifier also integrates `Phi.H*E` directly. Its output must
equal the identity at the original `1e-10` retained-constraint bound. In the
sextic order-`(4,0)` prototype below, the complete measures reduce that physical
defect from `7.19e-12` to `1.83e-26`; energy and stationary-energy defects are
`1.30e-26` and `3.98e-15`. The rounded product `x+Q*U-I` instead gives
`1.03e-10`, a coefficient-rounding defect. Metadata preserves that diagnostic
as `rounded_retained_defect`, while `retained_verification='physical_field_integral'`
identifies the physical check. The response is neither averaged nor corrected
by the verifier. No candidate tensor or tensor derivative is supplied to it.

## Physical gradient and phase coordinates

The contact gradient maps now form their Jacobians and inverses from the
same complete physical vertices as their measures. In bulk triangles,
`inverse = adj(J)/det(J)` and its variation is
`-inverse*(ds*J1)*inverse`. The direction is multiplied after preserving
`J1`, so both words of the caller-unit velocity reach the gradient and
Fourier-phase variations.

For a collapsing triangle, a common reference gradient annihilates the
collapsed adjugate. Its finite map is `adj(J1)/det'(J0)` and its fixed-common
variation is zero. This expression agrees with the analytic positive-gap
limit without forming a singular inverse. Boundary common gradients use the
same complete inverse; their retained Fourier phases are formed from the
original barycentric points before rounding, as in the physical verifier.

Independent 100-digit physical inverse and gap-difference checks cover every
triangle of a non-square, non-dyadic original and refined mesh, including
finite common-gradient opening limits. Their maximum relative component
errors were `5.66e-32` and `4.38e-31`, below the existing `3e-28` primitive
bound. These maps preserve the same physical coordinates and field space.
They do not change the normal-field bank or quadrature weights. The complete
evaluation of that same normal-field bank is described below.

## Complete disappearing potentials and seam fields

The polynomial normal potential has a different collapsing map from a common
potential. Its reference gradient need not annihilate `adj(J0)`: the finite
scaled opening gradient is `grad_ref(Z)*adj(J0)/det'(J0)`. Bulk triangles use
the ordinary inverse of `J0`. Forming those gradients from complete Bernstein
coordinates and integer differentiation coefficients avoids rounding the
normalized field before the area coefficient and large solved potentials
multiply it.

The seam lift keeps its original stored polynomial coefficients, real frame
and geometric coordinate changes. Values and spatial derivatives are now
evaluated with complete trigonometric factors and a differentiated Clenshaw
recurrence. Exactly even and odd Chebyshev polynomials use their identities
in `2*u*u-1` to reduce the work. Every nonzero coefficient survives; no
polynomial degree or trace column is removed. Common and disappearing
polynomial projections are combined before rounding, including triangles
with no local disappearing node. Boundary jumps retain the same two words.

A paired three-repeat evaluation of 4,096 points at transverse order 4
compared this recurrence with the complete forward Chebyshev recurrence.
Median times were `0.516 -> 0.209` seconds at degree 4 and
`0.243 -> 0.112` seconds at degree 5. Complete values and derivatives differed
by at most `2.04e-31` on the component scale. This measures trace evaluation
alone with one numerical-library thread, not the whole contact solve.

Both the weak equations and independent physical reconstruction use these
fields. Independent 100-digit checks cover the stored trace charts at
degrees 3 through 6 and transverse orders 0, 1 and 4, polynomial-potential
derivatives, and scaled positive-gap inverse limits on non-square original
and refined meshes. All use the existing `3e-28` primitive bound. The
stored trace chart is preserved; this change does not rebuild its reference
coefficients or change its physical span.

In the first quintic order-`(1,0)` joint gap/material control, the complete
field prototype passes all original physical and reflection guards, with
energy and derivative-energy residuals `2.22e-30` and `3.89e-30`. The
corresponding order-`(0,1)` control with angular order 48 also passes the
entire reflection test and original physical guards; its derivative-energy
residual is `1.41e-18`. These controls do not certify refined-mesh or optical
convergence.

The integrated Clenshaw implementation subsequently passed 25 controls:
the independent field checks, the transverse joint response, comparisons
among three complete contact solve paths, and four deliberately corrupted
physical-coordinate cases. Twelve additional controls passed for the
existing common-field paths, non-square cells, a quartic single direction,
and high-degree flat and metal references. The two groups took 1,003 and
903 seconds respectively in concurrent single-library-thread runs; these
are validation durations, not comparative performance measurements.

## Normal Fourier means from the periodic potential

The disappearing normal field includes bulk gradients, the collapsed seam,
and horizontal interface jumps. Its Fourier mean must include all three.
For the original periodic potential `V`, integration by parts gives
`q_b(k) = i*k*integral(exp(-i*k.x)*V(x))`, component by component.
The zero mode is exactly zero. This distributional identity includes the
interface terms without subtracting their large separately integrated values.
It is not a symmetry projection of the response or of its coefficients.

The bounded potential retains its original nodal coefficients, stored trace
chart, common projection, normal projection and full real coordinate change.
Three arithmetic words survive through those cancelling combinations; the
completed potential then enters the two-word Fourier integral. Increasing
Gauss orders must agree coefficient by coefficient at the existing `3e-28`
complete-field criterion,
or the calculation raises. Both words of the resulting mean enter the
normal projection and both complete contact elimination paths. The original
physical equations receive the same mean in their diagnostic coefficients.

On the refined quintic order-`(4,4)` metal geometry, the previous rounded
normal mean differed from this complete integral by `1.82e-9`. Independent
complete gradient-plus-jump integration agreed with the potential result
within `1.80e-24`, while the potential integrals at orders 24 and 32 agreed
within `1.38e-32`. This comparison diagnoses the mean calculation; it does
not qualify a complete tensor or its optical convergence. Independent
100-digit tests separately integrate piecewise constant bulk gradients, a
seam jump and both horizontal jumps. They cover asymmetric widths, unequal
periods, translation, zero modes and both Fourier directions at `3e-28`.

The weak assembly first integrates the raw normal fields using their exact
local nodal support and every seam column. It applies the full Fourier
subtraction after integration: `L = L_raw-q_b.H*R` and
`S_Z = S_raw-q_b.H*A`. Here `A` is the same complete free-material Fourier
matrix used by the retained equations. This removes repeated dense projected
normal samples while preserving every global coordinate. Exact conjugacy of
both words is required before this rearrangement; a failure raises, and no
coefficient is averaged or replaced. The original physical diagnostics
consume the directly integrated raw blocks.

The independent verifier now also differentiates the physical mean with
fixed potential coefficients. It directly integrates `dPhi.H*E`, `Phi.H*dE`,
the moving measure, the collapsing-cell contribution and both interface
jumps. Its result must vanish at the original `1e-10` constraint bound and is
reported as `physical_retained_derivative_residual`. Neither the candidate
response nor the solved common-coefficient variation enters this check.

A uniform-material corruption control adds a skew Cartesian variation only
to the verifier's retained field. Its stationary energy variation cancels,
so the energy and original matrix checks pass. The independently integrated
mean variation rejects it. Four existing corruption controls and the full
transverse joint gap/material response also pass with the new verifier.

The integrated local assembly passes the full quintic transverse joint
control, all three complete elimination paths at degrees five and six,
five flat/metal reference controls, and three non-square/legacy controls.
In the transverse case its energy, energy-variation and physical
mean-variation residuals are `1.27e-28`, `8.57e-28` and `1.05e-32`.
A paired calculation against pointwise Fourier subtraction kept the closed
tensor identical and changed the derivative by at most `7.84e-30` on its
component scale. Response stages took 166.5 and 135.1 seconds in that one
run; this is not a repeated whole-solver performance measurement. Large
refined-mesh and optical convergence require their separate complete checks.

The complete response and its pointwise verifier now read triangle material
coefficients and normal-coordinate indices directly from the original local
data. They no longer build the ordinary sampled fields that these paths
immediately replaced with complete fields. The independent verifier still
reconstructs every physical field and integrates all four energy and mean
quantities; it does not consume the response's assembled matrices.

At degree five, order `(0,1)`, radial/angular orders 100/48 and the joint
metal gap/material/background direction, a before/after comparison produced
bitwise-identical complete response coordinates, original physical matrices
and all four independently integrated quantities. Every original physical
and full reflection guard passed, as did all five corruption controls.
The two energy integrations took 61.7 and 55.9 seconds; response stages took
174.5 and 174.2 seconds under concurrent workloads. These are one-run
qualification timings, not a controlled general speed claim.

## Complete products with anticorrelated sample scales

Row and column normalization alone does not balance a contact edge dot
product. A quadrature weight can be tiny precisely where a corner gradient
is large. Their integrated product remains moderate, but the separate input
maxima can be enormous. Three integer chunks preserve the leading products;
ordinary products of their floating remainders can still lose significant
digits after the final rescaling.

An actual quintic edge product exposed this issue when its structural zero
rows were omitted. Against an independent 100-digit sum, the padded and
compact products had absolute errors of `1.34e-18` and `3.41e-20`. Neither
represented the complete dot to high/low precision. This was an arithmetic
defect, not a change in the trace field space.

The product now balances each inner coordinate with reciprocal powers of
two, accepting a scaling only if both operands round-trip exactly. It also
bounds the roundoff of every noninteger remainder product against the
absolute dot sum. A dot that exceeds its two-word arithmetic budget is
recomputed from error-free scalar products and three passes collecting the
addition remainders. When many dots require correction, a six-integer-chunk
matrix product first reduces that work; its complete floating remainder is
still retained and checked against the same bound. Remaining corrections
use batches of 64 output entries;
they do not remove any matrix entry, basis field or physical term. Extreme
exponent cases use exact standard-library fractions before the final
high/low rounding. Physical acceptance tolerances are unchanged.

Independent Decimal tests cover real and complex inputs, single and multiple
rows, anticorrelated scales spanning 900 binary exponents, and unrelated large
entries that defeat a single global balancing step. The tests use the same
`16*eps**2` absolute-dot-scaled accuracy criterion as the original residual
oracles. Separate controls cover structural zero padding and extreme finite
dyadic cancellation. This arithmetic verification does not certify optical
convergence. A preserved `2419 x 2419` stiffness product with 162 columns
required 390,202 scalar corrections after three chunks and none after six.
The adaptive product took 1.86 seconds; 15 independently selected complete
dots agreed with 150-digit sums within `2.49e-35` on the absolute-dot scale.
This is a matrix-product measurement, not a full solver timing.

Complete contact controls compare the retained solve, full geometric
coordinates and independently sampled QR coordinates at quintic order
`(1,0)` and sextic order `(0,0)`, including joint gap and complex-material
variations. All original physical guards pass, and the complete tensors and
derivatives agree at `3e-12` and `3e-11`. Four independent corruptions of the
response or verifier coordinates are still rejected. These six integration
controls took 586 seconds with one numerical-library thread.

## Physical regression

The earlier library control, before complete vertex measures, uses a flat reference `(2,2,2)`, physical materials
`(12.25,-10+i,1)`, Fourier order `(4,0)`, degree 6, period 700 and widths/height
300. Radial and angular orders are 120 and 24. It now passes all original
physical guards, with energy and derivative-energy residuals of 8.14e-27 and
3.92e-15. It took 94.3 seconds on NumPy 2.5.1 with one numerical-library
thread. The corresponding quintic regression is retained.

The free-moment controls also include non-dyadic geometry and an independent
100-digit calculation of the collapsed opening measure. These checks concern
the same finite equations. They do not certify polynomial, mesh, corner-mode,
Fourier or optical convergence.


A separate order-(4,4), two-branch transmission control retains the complete
original weak matrices and reintegrates only the collapsed-moment product
difference. Re-solving those matrices with the actual library response and
pointwise verifier passes all original guards: energy residuals are 7.85e-21
and 2.47e-12, and complete reflection defects are 1.22e-11 and 1.48e-9.
This replay took 1,043 seconds, dominated by the complete pointwise check.
It is a replay of preserved weak matrices, distinct from a fresh assembly.

A subsequent fresh library assembly of that sextic, two-branch order-(4,4)
cell also passes all original physical and full reflection checks. It uses
radial order 120 and angular order 80 and took 1,585 seconds on the same
runtime. Its energy residuals are 1.11e-20 and 2.47e-12; the full tensor and
derivative differ from the replay by at most 2.42e-22 and 1.39e-17. This
confirms the correction through fresh assembly and physical reconstruction.
The remaining optical changes across polynomial and corner spaces are
recorded in the [transmission study](nvf-pair-transmission-potentials.md).
