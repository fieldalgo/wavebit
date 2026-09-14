# Reciprocal energy from conjugate-paired normal frames

Written: 2026-09-14

Targets:
- `wavebit/_core/conforming/cartesian_pair_normal_frame.py`
- `tests/test_nvf_paired_frame.py`
- `tests/test_nvf_pair_normal_frame.py`
- `studies/nvf_pair_reciprocal_energy.py`
- `studies/results/nvf_pair_reciprocal_energy.log`

**TL;DR:** The explicit `paired_closure` averages the physical energies
of a normalized normal frame and its conjugate partner. The resulting
finite constitutive matrix is reciprocal and retains passivity and
rotated coercivity. Independent integration of both physical fields,
exact uniform/laminated limits and finite gap/material derivatives
verify this construction. It is a different finite energy model from
the original single frame. It does not establish metal optical
convergence or change ordinary NVF routing.

| Question | Answer |
| --- | --- |
| What is paired? | Two normalized physical field maps with the same retained electric coordinates. |
| What is the new constitutive matrix? | `H_pair = (H + J H^T J)/2`, where `J` reverses Fourier labels within each Cartesian component. |
| Are the materials conjugated? | No. Both energies use the same physical scalar permittivity. |
| What does pairing preserve? | Passivity, the common rotated-storage lower bound, exact uniform and full-height laminated rules, and reciprocity. |
| What happens to the original defect and inverse residual? | They remain visible under `metadata['raw']`; they are never reported as residuals of the paired energy. |
| What is still unqualified? | Optical convergence, general geometry/partition invariance and equivalence to a full Maxwell variational discretization. |

## Two physical field energies

The [single normal frame](nvf-pair-normal-frame.md) constructs `B(r)`
and its Fourier compression `Q=C(B)`. Let `Phi(r)` contain the retained
Cartesian plane waves. The normalized physical field map is

```text
W(r) = B(r) Phi(r) Q^(-1),
C_retained(W) = I.
```

Its original finite energy is `H = integral W^H epsilon W`. A complex
frame does not generally span a space closed under conjugation, which
is why this passive matrix can fail reciprocity in the retained physical
coordinates.

Define a second normalized map by conjugating the physical field and
reversing the source Fourier labels:

```text
W_c(r) = conj(W(r)) J,
C_retained(W_c) = I.
```

The symmetric Fourier set is essential: each mode has its negative in
the same Cartesian component. Because the physical permittivity is
scalar, the second energy is

```text
H_c = integral W_c^H epsilon W_c = J H^T J.
H_pair = (H + H_c)/2.
```

This identity permits a matrix evaluation without constructing another
large sampled field. The independent test takes the opposite route:
it builds both normalized maps at physical quadrature nodes and
integrates their two energies directly. It never transposes the original
closure to generate its expected answer.

The construction is explicitly an average of two trial-field energies.
It is not claimed to equal the energy of their averaged field; that
would introduce different cross terms. It is also not a proof that this
finite constitutive approximation is the full Maxwell variational
discretization. Those distinctions matter when interpreting optical
convergence.

## Reciprocity and passivity together

The operation `S(H)=J H^T J` is an involution. Therefore
`S(H_pair)=H_pair`, the Lorentz reciprocity condition in these Fourier
coordinates. The transpose is not a Hermitian adjoint.

For either physical map and any coefficient vector `e`, passivity follows
from its actual material energy:

```text
Im(e^H H e) = integral Im(epsilon) |W e|^2 >= 0.
```

The same statement holds for `W_c` with the same material. Their average
is therefore passive. Equivalently, the dissipation matrix of `H_c` is
the transpose and permutation of that of `H`, with the same eigenvalues.
The smallest dissipation eigenvalue of the average is at least the
original smallest eigenvalue in exact arithmetic.

For the common material certificate `zeta=exp(-i alpha)`, the rotated
storage of `H_c` is likewise the transpose and permutation of the
rotated storage of `H`. The paired matrix retains its positive lower
bound. This argument needs neither eigenvalue clipping nor a lossless
assumption and leaves the physical material values unrotated.

Changing the original sesquilinear form to a bilinear form is not this
construction. A complex bilinear field energy has no corresponding
positivity bound. Pairing keeps the conjugate test field in each of the
two physical energies.

Any already reciprocal matrix is unchanged by pairing. Thus the exact
uniform medium, the full-height Cartesian inverse/direct rules and
their reciprocal parameter variations are retained. In particular,
the closed-contact laminate derivative has an independent analytical
test based on moving rectangular material transforms.

## Explicit use and diagnostics

```python
import numpy as np
from wavebit._core.conforming.cartesian_pair_normal_frame import CartesianPairNormalFrame

cell = CartesianPairNormalFrame(
    basis=np.diag([700., 560.]), widths=(170., 230.), height=280.,
    order=(2, 1), quadrature=32, coercivity_angle=1.53)
slots, metadata = cell.paired_closure(gap=0., epsilon=(12.25, -10.+1j))
```

The five slots include the layer axis and feed `solve_conv`. The
longitudinal inverse is unchanged. Geometry, material-sector admission,
quadrature requirements and NumPy/JAX x64 scope are those of the
[raw instrument](nvf-pair-normal-frame.md). Ordinary JAX AD differentiates
the live gap and scalar material values; zero-gap derivatives are right
derivatives per caller length unit.

`paired_closure` returns slots and metadata, not the original nine
coefficient grids. Those grids define the single pointwise frame's
energy; labeling them as the paired energy would be incorrect. The
existing `coefficients` and `evaluate` methods continue to expose the
original single-frame instrument for explicit comparison.

The original calculation must pass first. Its finite, base-integral,
coercivity, physical and inverse-residual guards are unchanged. All raw
diagnostics, including the original reciprocity defect, remain in
`metadata['raw']`. Pairing cannot repair a failed raw guard.

The new energy is then checked independently for finite output,
positive rotated storage, dissipation above `-1e-9`, reciprocity error
below `1e-9`, and lossless Hermitian error below `1e-9`. Its maximum
and relative changes from the raw tensor are reported as
`maximum_pairing_change` and `relative_pairing_change`. Invalid JIT
replay poisons every returned slot. A large constitutive change is
not reclassified as a small numerical residual.

## Verification

The paired tests use independently assembled physical nodes, full dense
point phases and full pointwise field products. They explicitly verify
that both normalized fields have identity retained Fourier projection
before comparing their integrated energy to the candidate slots.
The normal extension itself is shared with the separately tested
parametric path; the reference does not claim an independent normal
construction.

Controls cover dielectric, absorbing and metal pairs at contact,
positive gap and gap `1e-9`; exact uniform and laminated values;
an analytical laminated contact derivative; forward/reverse gap
derivatives; live complex material derivatives against two resolved
secant windows; and rejection of corrupted raw energy and invalid
traced input. They establish the paired finite calculation, not
optical continuum accuracy.

## Optical comparison

The study uses a square period of 700, two touching 300-by-300 rectangles,
background 1, layer thickness 300, wavelength 900, incidence angle 0.2,
azimuth zero and top/bottom permittivity 1/2.25. It records both linear
polarizations at Fourier orders 4/8/12/16 and Gauss orders 128/256.
At Gauss order 256, orders 4/8/12 also record two second-order contact
slope windows and gap `1e-9`. Cartesian Li is a separate finite-order
comparison, not a substituted branch or a certified continuum oracle.

The record `studies/results/nvf_pair_reciprocal_energy.log` contains the
original and paired diagnostics and source hashes. Measurements use
NumPy 2.5.1 on macOS arm64. The instrument refuses nonfinite/active
outputs and lossless power-balance errors larger than `1e-9`.

For metal `12.25 | -10+i`, Gauss order 256 gives:

| Fourier order per direction | Raw p | Paired p | Li p | Raw s | Paired s | Li s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.227439403 | 0.277109341 | 0.483946395 | 0.594231419 | 0.612177501 | 0.480212354 |
| 8 | 0.384378678 | 0.402013238 | 0.480976241 | 0.471972924 | 0.454244908 | 0.402781608 |
| 12 | 0.415465282 | 0.431031664 | 0.489085870 | 0.436585134 | 0.435893375 | 0.407928008 |
| 16 | 0.464297818 | 0.471362888 | 0.485772940 | 0.472594987 | 0.455481301 | 0.426820064 |

The paired matrix has zero measured reciprocity defect in these rows,
while keeping the dissipation and rotated-storage guards. At order 12,
the raw relative reciprocity defect is about 0.39733 and pairing changes
the tensor by about 0.19867 in relative Frobenius norm. This is a
substantial change to the finite model, not rounding cleanup.

Optical accuracy is not uniformly improved: the order-4 s comparison is
worse, whereas both order-8 comparisons move toward same-order Li.
Large Fourier-order changes remain. The metal p contact slope at
order 12 is about `+0.0378429` for the paired model, versus
`-0.0216445` for same-order Li. The s slope improves there to
`+0.0170625`, versus Li `+0.0173520`, but at order 8 the paired s
slope is `-0.0205087`, opposite in sign to Li `+0.0115477`.
These are not stable optical derivatives across the tested orders.

From Fourier order 12 to 16 the paired p/s totals change by about
`0.04033 / 0.01959`. At order 16, changing Gauss order 128 to 256
changes them by about `9.72e-7 / 2.40e-6`. The finer integration
therefore does not account for the large outer-order changes. Neither
an exact reciprocity identity nor a small last quadrature change
certifies a converged optical result.

For the real equal-material pair, the raw energy was already reciprocal.
Pairing leaves that finite model unchanged in exact arithmetic; its
existing discrepancies with Li therefore remain. Small differences in
the recorded totals and secants reflect rounding through the optical
solve, not a new dielectric accuracy improvement.

The implemented result is an explicitly selectable finite energy that
preserves reciprocity and passivity together. Optical/contact convergence
and production adoption remain open. Matrix structure alone is not an
optical stopping criterion.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m studies.nvf_pair_reciprocal_energy
```

No new dependency, external solver or copied third-party implementation
is introduced. This remains an explicit experimental energy construction.
