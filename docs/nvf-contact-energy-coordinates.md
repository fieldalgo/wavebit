# Preserve contact coordinates through the physical energy check

Targets:

- `wavebit/_core/conforming/pair_contact_energy_precision.py`
- `wavebit/_core/conforming/pair_contact_precision.py`
- `wavebit/_core/conforming/pair_contact_quadrature.py`
- `wavebit/_core/conforming/real_trace_coupling.py`
- `tests/test_nvf_transmission_common_precision.py`
- `tests/test_nvf_contact_fourier_precision.py`

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
