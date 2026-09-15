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

The opening measure now preserves both scalar factors on collapsed triangles.
The free Fourier variation, common moments, common opening energy and
normalized disappearing energies use that consistent scaling. Bulk records
retain their established measure variation. Geometry, field count, reference
materials and every physical acceptance threshold are unchanged.

## Physical regression

The actual library control uses a flat reference `(2,2,2)`, physical materials
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
