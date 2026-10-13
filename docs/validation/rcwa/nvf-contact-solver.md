# Guarded elimination for polynomial pair contact

Written: 2026-09-15

Targets:
- `src/wavebit/rcwa/_engine/conforming/cartesian_pair_seam_galerkin.py`
- `src/wavebit/rcwa/_engine/conforming/real_trace_coupling.py`
- `src/wavebit/rcwa/_engine/conforming/projected_pair_contact.py`
- `tests/rcwa/test_nvf_contact_solver.py`
- `tests/rcwa/test_nvf_pair_seam_galerkin.py`
- `studies/rcwa/nvf_coupled_pair_seam_galerkin/nvf_coupled_pair_seam_galerkin.py`
- `studies/rcwa/nvf_contact_solver/records/nvf_contact_solver_comparison.log`
- `studies/rcwa/nvf_contact_solver/records/nvf_contact_field_products.log`

The cubic and quartic pair family now tries original retained-variable
elimination before constructing the more expensive physical QR coordinates.
It accepts the complete electric tensor and contact derivative only after
all original equation, physical energy and material checks pass. A failed
candidate is discarded in full, and the same field space is solved with the
existing real geometry QR method. This accelerates resolvable finite systems
without substituting a different constitutive model.

This change does not supply mesh, Fourier-order or optical convergence,
generalize geometry, or qualify ordinary NVF routing. Positive-gap assembly
and its projected solve remain unchanged.

## Two complete solves of the same equations

At contact, the electric value depends on the surviving common potentials C.
Disappearing nodal and added real gap potentials enter their original weak
limit and the complete right-opening derivative. Retained elimination first
solves the original common stiffness for its retained electric and flux
loads. The small coupled retained system then enforces both

```text
x + Q_C u_C = I
H = A x + R_C u_C.
```

After recovering u_C, it finds every disappearing potential from its original
weak rows. It differentiates the common and retained equations with complete
forcing from these potentials, geometry and simultaneous complex material
variations. No mode is omitted by a singular-value or material threshold.
This is the existing `retained` elimination in `real_trace_coupling.py`.

The physical method forms real QR coordinates from sampled high-mode fields
and assembles their material energy directly. Both methods span the same
nodal and [analytic polynomial remainder](nvf-analytic-polynomial-traces.md)
space. Selection operates on complete states, never individual tensor or JVP
blocks. The independent contact-energy integration checks both the value and
derivative before a retained candidate is accepted. All original residuals
keep their 1e-10 limits; reciprocity and derivative reciprocity keep 1e-9.
Positive rotated storage, passive dissipation and finite physical diagnostics
are required. A small algebraic residual alone cannot select the fast method.

## Selection and diagnostics

`CartesianPairSeamGalerkinGapCell(..., contact_solver='auto')` is the default.
For degrees three and four at zero gap it tries retained elimination and then,
if necessary, physical QR. `contact_solver='projected'` forces physical QR for
reproducible comparisons. The quadratic family keeps its original retained
solve and refuses forced polynomial projection.

`contact_solver_attempts` records attempts in order; `closed_elimination`
records the accepted method. A rejected retained attempt records its reason
in `retained_contact_rejection`. Failure of the final QR solve or its physical
checks still raises an error. The explicit class remains a NumPy finite
kernel. No new dependency or third-party code is introduced.

Physical rejection messages name every failed metric, its value and its
unchanged acceptance bound. Nonfinite response blocks are identified separately.
This distinguishes an original nodal defect from a stationary-energy or
material failure without rerunning an expensive solve solely to inspect its
metadata. A reported metric identifies the failed check, not its cause;
roundoff, insufficient quadrature and an inaccurate candidate require separate
diagnosis. All original threshold boundaries remain unchanged.

## Verification and measured cost

Tests compare complete automatic and forced-QR values and derivatives for
cubic/quartic fields, dielectric/metal inputs and combined gap/material
directions. They retain the established 3e-10 tensor and 3e-9 dimensionless
gap-derivative tolerances. Existing independent clipped physical references,
positive-gap limits, material variations and high-order controls also apply.
Additional tests show that accepted retained states skip QR, rejected states
are replaced in full, and failure of both methods cannot return a result.
Nonfinite physical diagnostics are refused explicitly.

The source-pinned comparison uses the metallic 700 by 700 pair, two 300-wide,
300-high bodies, materials (12.25, -10+1j), background 1 and angle 1.53.
Fourier order is 4, polynomial degree 3, refinement levels are 1 and 2, and
quadrature is 32 for both strategies. `closure_seconds` measures only the
constitutive value/JVP call, excluding subsequent RCWA power evaluation.
The raw log retains both polarization powers, slopes and original diagnostics.

All eight requested polarization rows completed with source hashes matching
the measured implementation. Both automatic runs accepted retained elimination.

| Refinement | Automatic closure, seconds | Forced physical QR, seconds | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 2.7045 | 8.1501 | 3.01 |
| 2 | 10.8583 | 126.4553 | 11.65 |

These are two measurements per strategy, not a statistical timing benchmark.
Both strategies pass all original guards and give the same optical values
within 8.2e-14 and contact slopes within 3.0e-9 in these comparisons.

A further completed quartic, level-1 automatic run exercises a natural
retained rejection on the same order-4 metal pair. It records both attempts
and accepts physical QR with value-energy residual 3.09e-13 and derivative-
energy residual 4.55e-14. Its two polarization rows are appended separately
with their own source-pinned header. Thus a retained failure does not require
relaxing a guard or accepting a partly corrected state.

Timings describe these local runs, not a hardware-independent speed guarantee.
A rejected retained solve adds cost before physical QR. Acceptance criteria
decide whether it can be used; speed does not override them. Optical
mesh/Fourier variation remains unresolved.

## Applying all physical sample rows together

The physical QR path subsequently reuses the existing `_field_product`
helper when applying coefficient matrices to sampled fields. It reshapes
only the quadrature and Cartesian-component axes, from `(q, 2, n)` to
`(2*q, n)`, and performs one matrix multiplication before restoring the shape.
This avoids many separate two-row multiplications. No sample is averaged,
removed or shared across geometries or materials.

The operation order remains `(G - Phi*Q)*T`: the high-mode physical field is
formed and checked for reality before applying the QR transform. Multiplying
`Q*T` first would change cancellation between nearly coincident fields and
is deliberately avoided. The same row application is used for physical
derivatives, disappearing potentials and boundary energy terms. The geometry
QR, material energy contractions, quadrature orders and acceptance checks
are unchanged.

A separate source-pinned run in `nvf_contact_field_products.log` repeats the
forced-QR cubic comparison after this change. Existing independent physical
references and whole-state QR comparisons are supplemented by forced-QR
order-8 controls for both polynomial degrees and both material cases.
Positive-gap solvers do not use this changed function.

All four requested polarization rows completed and passed the original
guards. The recorded powers are identical to the earlier forced-QR rows;
the largest difference in the differenced optical contact slope is 2.59e-9.

| Refinement | Earlier two-row batches, seconds | All sample rows, seconds |
| --- | ---: | ---: |
| 1 | 8.1501 | 6.4520 |
| 2 | 126.4553 | 119.9303 |

These single local timings include variations in background verification
load. They show a reduction in these runs, rather than a universal speed
factor. In separate whole-state comparisons, the electric tensors were
identical and dimensionless contact derivatives agreed within 7.34e-10.
The physical equations and convergence limitations are unchanged.
