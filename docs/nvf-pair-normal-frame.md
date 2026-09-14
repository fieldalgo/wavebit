# A material-referenced normal frame for a finite rectangular pair

Written: 2026-09-13

Targets:
- `wavebit/_core/conforming/cartesian_pair_normal_frame.py`
- `tests/test_nvf_pair_normal_frame.py`
- `studies/nvf_pair_normal_frame.py`
- `studies/results/nvf_pair_normal_frame.log`
- `studies/nvf_pair_normal_frame_quadrature.py`
- `studies/results/nvf_pair_normal_frame_quadrature.log`

**TL;DR:** An explicit normal frame removes the electrostatic auxiliary solve
and has a positive lower bound on its compressed matrix. Fitted physical
quadrature retains the newly opening material strip, including ordinary JAX
gap and material derivatives. Exact uniform and laminated controls pass.
Metal optical values still change substantially with Fourier order, and
the finite complex frame does not preserve reciprocity. This is an
experimental constitutive instrument, not a replacement for ordinary NVF
or a claim of converged contact optics. Tiled NumPy integration lowers
the measured order-512 quadrature peak from 3.09 GB to 0.49 GB without
changing its finite coefficients.

| Question | Answer |
| --- | --- |
| What is constructed? | The full pointwise material energy of an invertible frame built from the existing unit Hermitian normal projector. |
| What changes inside equal touching bodies? | The frame is exactly the identity there, independently of the artificial internal normal field. |
| What parameters are live? | The nonnegative physical gap and all three scalar permittivities, including background. Other geometry is fixed. |
| What prevents a singular compressed frame? | A common material half-plane gives an explicit positive lower bound, including a bound for base-integral quadrature error. |
| What does a passed derivative establish? | The derivative of the declared fitted finite quadrature. It is not a continuum optical error certificate. |
| What remains unresolved? | Reciprocity of the finite complex closure, optical order convergence, arbitrary shape/partition invariance and automatic integration into ordinary NVF. |

## Pointwise frame and energy

Let `P = n n^H` be the existing unit Hermitian projector, and let
`kappa = (epsilon_a + epsilon_b)/2`. In each physical material region define

```text
B = I + (kappa/epsilon - 1) P
G = epsilon B^H B
  = epsilon I + (|kappa|^2/conj(epsilon) - epsilon) P.
```

At a straight physical interface where `P` is its real normal projector,
the tangential eigenvalue of `B` is one and its normal eigenvalue is
`kappa/epsilon`. The associated normal displacement has factor `kappa` on
both sides. If the two bodies have the same permittivity, `B=I` inside
both bodies. This cancellation removes the effect of their internal
normal variation from the body energy, without changing the background's
physical permittivity or removing the gap's contribution.

Fourier compression is applied **after** these pointwise material products:

```text
Q = C(B),    K = C(G),    E = C(epsilon)
H = Q^(-H) K Q^(-1),     longitudinal inverse = E^(-1).
```

The existing energy closure supplies the five solver slots. In particular,
`K` is not formed as `C(B)^H C(epsilon) C(B)`: that different product would
discard unresolved products before the energy integral is taken. The nine
coefficient components retain the order
`Bxx, Bxy, Byx, Byy, Gxx, Gxy, Gyy, epsilon, Gyx`.

The frame is a material-referenced version of the continuous-field
rearrangement discussed among the
[earlier closure attempts](nvf-contact-followup.md). The rearrangement
itself is not a novelty claim. The present instrument combines the body
reference, pointwise energy, fitted opening integration and a finite
coercivity certificate. It has no auxiliary potential mesh or inner weak
solve; Fourier order and quadrature order remain independent controls.

## Why the compressed frame is invertible

The existing [material admission](nvf-pair-material-sector.md) requires
passivity and a common angle `alpha` such that every physical material has

```text
Re(exp(-i alpha) epsilon_j) > 0.
```

The arithmetic mean `kappa` lies in the same open half-plane and is nonzero.
Choose `psi = exp(i alpha) conj(kappa)/|kappa|`. Then

```text
Re(psi) = Re(exp(-i alpha) kappa)/|kappa| > 0
Re(psi kappa/epsilon_j)
    = |kappa| Re(exp(-i alpha) epsilon_j)/|epsilon_j|^2 > 0.
```

Because `P` is Hermitian and idempotent, the rotated storage of `B` is
bounded below pointwise by the minimum of these scalars, denoted `beta`.
Exact Fourier compression therefore has the same lower bound. No
pointwise determinant or harmonic-corrector invertibility assumption is
needed.

The implementation adds exact identity coefficients to quadrature of
`B-I`. If `delta` is the maximum error in the independently integrated
constant Fourier coefficients, and `N` is the number of retained modes,
the finite frame retains the conservative lower bound

```text
beta - N (|Re(psi)| + beta) delta.
```

Admission requires this bound and the directly measured smallest rotated
eigenvalue to be positive. The bound is returned as
`frame_storage_lower_bound`. The certificate angle only tests admission;
it never rotates the physical materials, coefficients or returned slots.

The energy congruence is passive when its physical integrals are resolved.
The implementation also checks the actual finite closure: positive rotated
storage, minimum dissipation above `-1e-9`, Hermitian error below `1e-9`
for lossless materials, and transverse/longitudinal inverse residuals no
larger than `1e-10`. These algebraic guards cannot bound optical truncation
error or the quadrature error of the nonpolynomial normal field.

## A physical opening strip, including at zero gap

Five horizontal intervals follow the background, first body, gap, second
body and background. Three vertical intervals follow the lower
background, body height and upper background. Each interval uses an even
Gauss rule; their tensor product uses `15 q^2` sample nodes. Material
interfaces are integration boundaries, not raster assignments.

At zero gap the middle interval has zero width but a nonzero derivative
of its integration weight. Inside the body height its limiting normal
projector is `diag(1, 0)`. That value is retained explicitly; differentiating
the closed seam's arbitrary normal tie would give a different derivative.
Outside the body height the usual field is retained so that the moving
background partitions cancel correctly. The placeholder coordinates used
to avoid evaluating the closed tie never contribute a value integral.

Cross-body coincidence substitutions from the ordinary normal plan are
removed in this explicit gap family. Otherwise a geometric contact
tolerance would snap a sufficiently small positive gap back to contact.
Self-periodic records remain valid because the fixed width and height do
not change. The physical material transforms remain analytic.

Normalizing a small signed normal sum is not, on its own, sufficient to
establish a correct opening derivative. A vanishing strip can carry an
order-gap integral even when every fixed sample appears smooth. The
fitted strip here retains that contribution. This observation does not
justify replacing the ordinary normal field with an untested bounded
normalization or declaring a fixed-quadrature derivative converged.

## Verification and explicit scope

The test module checks direct pointwise `B` and `epsilon B^H B` products
at independently constructed physical quadrature nodes. Its dense
Fourier transform and independent inverse closure share neither the
simplified material-weighted formula nor the separable transform from
the candidate. The normal extension itself is shared with the existing
parametric implementation; this is stated explicitly in the reference.

Additional controls include:

- Exact uniform dielectric, absorbing and negative-real material values;
  first material variations recover the exact rectangular material
  convolution with no residual dependence on the normal field.
- Exact Cartesian inverse/direct rules for a full-height laminate,
  including an independent analytic right-opening derivative at contact.
- Reflection with swapped body listings, physical unit covariance,
  certificate-angle independence and the scalar coercivity bound.
- NumPy versus JAX x64, forward and reverse gap derivatives, and live
  complex material derivatives against two resolved secant windows.
- Invalid eager input rejection and all-NaN coefficient/slot output for
  invalid JIT replay. A constructor's minimum Gauss order does not imply
  sufficient quadrature: the constant Fourier check can still refuse it.

The static geometry is a positive diagonal lattice containing two
horizontal equal-height rectangles. Widths and height are positive,
the sum of widths is less than the horizontal period, and height is at
most the vertical period. The gap lies in
`[0, period_x - width_a - width_b)`. The even Gauss order is at least
`max(8, 2*max(order_x, order_y)+4)`; the constant integral error must be
below `1e-11`. Compare at least two quadrature orders separately from
the outer Fourier ladder. A passed constant check is not an estimate
of normal-field integration accuracy.

```python
import numpy as np
from wavebit._core.conforming.cartesian_pair_normal_frame import CartesianPairNormalFrame

cell = CartesianPairNormalFrame(
    basis=np.diag([700., 560.]), widths=(170., 230.), height=280.,
    order=(2, 1), quadrature=32, coercivity_angle=1.53)
coefficients, slots, metadata = cell.evaluate(
    gap=0., epsilon=(12.25, -10.+1j))
```

The slots include the layer axis and can feed `solve_conv`. JAX use passes
`xp=jax.numpy` with x64 enabled; ordinary AD differentiates `gap`, both
body permittivities and `background`. The gap derivative at zero is a
right derivative per caller length unit. Torch, live lattice/size/offset
derivatives, more general geometry and public automatic routing are not
implemented in this instrument.

## Bounded NumPy field storage

The NumPy evaluator integrates the same fitted nodes in tiles of at most
64 y rows. Every tile retains its absolute row indices, so body ownership
and the special collapsed strip remain correct even when a tile crosses
a material boundary. A short final tile uses its actual physical weights.
The Fourier contributions are summed before the exact base terms and
unchanged coercivity/energy guards are applied. No field is interpolated,
truncated or recomputed from a lower-order approximation.

At quadrature order `q`, the full field has `15 q^2` nodes. A NumPy field
evaluation now has at most `320 q` nodes. Field temporary storage therefore
grows linearly with quadrature order for a fixed tile size, although
arithmetic still visits every node. The coefficient grids and convolution
matrices retain their existing Fourier-order costs. Metadata records
`field_tile_rows` and `maximum_field_samples` separately from total
`sample_count`.

JAX retains one array expression. Unrolling the geometric field once per
tile would enlarge its traced program, and the present change makes no
claim about JAX memory. The tiled NumPy and single-expression JAX results
are compared explicitly across a tile boundary. Independent physical
quadrature also exercises seven-row and 64-row tiles, positive and zero
gaps, and a short final tile. All 65 normal-frame tests pass, including
the existing forward/reverse and complex material derivative controls.

The reproducible instrument runs whole-grid and tiled coefficient
integration in fresh processes. Peak process memory includes imports
and quadrature setup; the timed region includes coefficient construction
only. Its subsequent optical calculations reuse the resulting coefficient
arrays and are excluded from those memory/time figures. The record is
`studies/results/nvf_pair_normal_frame_quadrature.log`.

On macOS arm64 with NumPy 2.5.1, Fourier coefficient cap 12 and the
metal contact fixture, the measurements are:

| Gauss order | Whole-grid peak process GB | Tiled peak process GB | Whole-grid coefficient time, s | Tiled coefficient time, s | Largest coefficient difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.354 | 0.160 | 0.169 | 0.140 | 1.23e-15 |
| 256 | 1.131 | 0.292 | 0.848 | 0.594 | 2.00e-15 |
| 512 | 3.085 | 0.487 | 5.104 | 2.735 | 3.46e-15 |

GB is `10^9` bytes. The finest row lowers observed peak process memory
by 84.2%. These single-run times improved in this measurement, but they
are not a whole-solver speed guarantee. No other calculation from this
task was running during the coefficient-only worker measurements;
unrelated activity on the host is not controlled.

At Fourier order 12, the finer inner ladder gives:

| Gauss order | Metal p transmission | Metal s transmission |
| ---: | ---: | ---: |
| 128 | 0.415474961 | 0.436566439 |
| 256 | 0.415465282 | 0.436585134 |
| 512 | 0.415465368 | 0.436583486 |

The last p/s changes are about `8.55e-8 / 1.65e-6`, much smaller than
the observed Fourier-order changes. This is empirical separation of
two errors, not a rigorous quadrature enclosure. Refining this field's
integrals alone does not remove the large outer discrepancy.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m studies.nvf_pair_normal_frame_quadrature
```

The original optical record below describes the initial whole-grid
implementation. The newer source hashes identify the storage refactor;
they do not change the material reference or the mathematical integrals.

## Reciprocity is independent of passivity

The finite complex energy closure has another limitation, separate from
the quadrature error. In the physical Fourier coordinates, reciprocal
constitutive data satisfy

```text
H = J H^T J,
```

where `J` reverses the Fourier labels within each Cartesian component.
Equivalently, transforming from complex exponentials to real sine/cosine
standing waves makes the constitutive matrix transpose-symmetric. This
condition uses a transpose, not a Hermitian adjoint, and applies to
absorbing reciprocal materials too.

The finite frame trial space `B f_N` is generally not closed under complex
conjugation. Thus its passive energy congruence does not, by itself,
supply reciprocal constitutive data in the physical retained coordinates.
The evaluator now reports the maximum-entry `reciprocity_error` and the
Frobenius `relative_reciprocity_defect`. The latter divides by
`max(norm(H), 1)`. Its `valid` flag still means the stated finite,
coercivity, loss and inverse-residual checks passed; it does **not**
certify reciprocity or optical accuracy.

An independent test constructs the real standing-wave basis and compares
its transpose defect to the reported Fourier-coordinate defect. Uniform
materials and exact full-height laminates retain reciprocity. On the
metal contact fixture at Gauss order 512, the relative defects at Fourier
orders 4/8/12 are about `1.2023 / 0.4615 / 0.3974`. At order 12 the
defect remains about 0.3974 over Gauss orders 128/256/512, even though
the integration changes become much smaller. This is a property of the
finite frame closure, not a small numerical integration residual.

The diagnostic leaves the measured tensor intact. A replacement must
establish reciprocity and passivity together with the interface limits
and contact/material derivatives. Positive matrix storage alone is
insufficient for adopting this complex frame in ordinary NVF.

The separate [conjugate-paired energy](nvf-paired-frame.md) now provides
an explicit reciprocal and passive alternative through `paired_closure`.
It preserves the raw diagnostics for comparison and defines a different
finite energy; the original `evaluate` path retains the limitations above.

## Optical decision

The recorded study uses a square period of 700, two touching 300-by-300
rectangles, background 1, layer thickness 300, wavelength 900, incident
angle 0.2, azimuth zero and top/bottom permittivity 1/2.25. It compares
Fourier orders `(4,4)`, `(8,8)` and `(12,12)` independently from Gauss
orders 32/64/128. Both linear polarizations are propagated. At Gauss
order 128, two second-order right-opening windows and gap `1e-9` are
also recorded. Cartesian Li is a separate same-geometry finite-order
comparison, not a substituted NVF branch or an assumed continuum oracle.

The record is `studies/results/nvf_pair_normal_frame.log`, with source
hashes and diagnostics. It was produced with NumPy 2.5.1 on macOS arm64;
wall times are observations, not a controlled performance comparison.

For metal `12.25 | -10+i`, Gauss order 128 gives:

| Fourier order in each direction | Frame p transmission | Li p transmission | Frame s transmission | Li s transmission |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.227437698 | 0.483946395 | 0.594228662 | 0.480212354 |
| 8 | 0.384383287 | 0.480976241 | 0.471958636 | 0.402781608 |
| 12 | 0.415474961 | 0.489085870 | 0.436566439 | 0.407928008 |

The outer-order changes remain large. At order 12, refining quadrature
from 64 to 128 changes p/s by `9.75e-5 / 1.80e-4`; the latter is larger
than its change from 32 to 64. Neither the outer ladder nor the inner
integration can be declared converged from the frame's coercivity alone.
The metal p contact slope also changes sign between orders 8 and 12,
from `-0.0215843` to `+0.0527966` in the smaller window.

For equal material `4 | 4`, Gauss order 128 and Fourier order 8 give
p/s transmission 0.998390026/0.986805646, versus same-order Li
0.997682393/0.988163075. The corresponding p/s contact slopes are
`5.55297e-5 / -1.39617e-4`, versus Li
`8.09165e-6 / -1.58322e-4`. Agreement between the instrument's two
secant windows does not remove this discrepancy with the separate
optical comparison.

At order 12, the equal-material p slope windows differ by `3.61e-8`;
the gap-`1e-9` p transmission change is `5.09e-12`, where propagation
roundoff is significant relative to the expected change. Those rows
are observations, not a precision certificate for a tiny-gap derivative.

The implementation is useful for isolating frame invertibility and
material-weighted contact integration from the auxiliary weak solve.
It does not currently solve the metal convergence problem, preserve
general finite complex reciprocity or establish primitive-partition
independence. Ordinary NVF contact behavior
retains its existing limitations. No default, stopping tolerance or
reference answer is changed to accept these optical discrepancies.

No new dependency, external solver or copied third-party implementation
is introduced. The instrument uses the existing NumPy/JAX operations,
normal field and energy closure.
