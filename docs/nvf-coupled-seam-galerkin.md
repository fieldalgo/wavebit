# Coupling continuous seam fields to the complete periodic tensor

Written: 2026-09-14

Targets:
- `wavebit/_core/conforming/periodic_seam_galerkin.py`
- `tests/test_nvf_seam_galerkin.py`
- `studies/nvf_coupled_seam_galerkin.py`
- `studies/results/nvf_coupled_seam_galerkin.log`
- `studies/results/nvf_coupled_seam_galerkin_fine.log`
- `studies/results/nvf_coupled_seam_galerkin_order7.log`

TL;DR: The quadratic periodic mesh and continuous seam potentials now share
one physical weak solve, including all transverse Fourier components and
their mutual energy. Positive-gap JVPs and an exact contact energy limit are
implemented. This explicit NumPy family does not change ordinary NVF routing
or qualify metal-pair and general-geometry optical convergence.

| Question | Implementation | Qualification boundary |
| --- | --- | --- |
| Is only the old yy block replaced? | No; every retained component shares one enlarged real field space and its original material energy. | A centered rectangle approaching its vertical periodic image, with arbitrary fixed translation and positive rectangular periods. |
| Does the forward model exist at positive gaps? | Actual continuous periodic scalar potentials are integrated with the quadratic mesh fields. | Both physical materials must share the admitted coercive half-plane. |
| How is gap zero evaluated? | Analytic body integrals, endpoint measures and leading thin-gap gradients form the limiting original weak equations. | The normalized disappearing rows themselves need not be differentiable. |
| Which derivatives exist here? | Simultaneous gap, inclusion and background directions, with complex-linear material dependence. | Other geometry is fixed; NumPy only. |
| Does this finish NVF? | It supplies the full finite periodic seam family. | Production routing, other geometries/backends and optical convergence remain separate. |

## One enlarged physical field space

The [quadratic mesh](nvf-pair-galerkin.md) supplies real potentials C u at
contact and s Z v for its opening differences. The
[continuous seam functions](nvf-seam-trace.md) add potentials with physical
gradients L_e = s l_e. Every retained transverse field has independent
coefficients in all three parts. With Phi the retained plane waves,

```text
E = Phi x + L a,       x + Q a = I,
K a + S x - Q_test H = 0,
H = A x + R a.
```

K, Q and R are the original physical gradient energy, unweighted projection
and material-weighted projection. Opening rows are divided by s as in the
mesh kernel; this removes a coordinate singularity without changing its
positive-gap equations. The source uses mode reversal and transpose,
preserving complex material coefficients. No material-dependent trial basis,
post-solve reciprocity repair or Cartesian Li fallback is introduced.

The new upper C-to-seam block is s C_e, the old difference-to-seam row is
Z_e, and the seam self block is its physical energy divided by s. The reverse
blocks have the exact corresponding physical row scaling. In particular,
omitting C_e or Z_e would produce a different finite field problem.

## Moving integration and the positive-gap JVP

Each triangle is cut at the cubic taper's endpoints and physical material
interfaces. A tensor Gauss rule in its x interval and vertical affine section
integrates the original polynomial gradients together with the seam field.
The JVP differentiates the interval endpoints, barycentric coordinates,
physical gradients, phases, materials and integration weights, then solves
the differentiated original weak equations.

Known moving taper endpoints carry their exact fractions 0, 1/2 or 1.
Recovering those fractions by subtracting rounded full-cell coordinates
loses accuracy at small gaps. Likewise, the upper gap's y coordinates stay
local to the seam before the fixed translation enters the Fourier phase.
Triangle area is formed from its certified constant and gap coefficients,
not a subtraction of nearly coincident physical vertices.

The code independently reassembles reactions for every original nodal test,
including the gauge, and checks the reconstructed physical-field energy.
It also differentiates the reciprocal bilinear energy as a separate check
of the reported tensor JVP. For complex materials this is a reversed-mode
transpose energy; replacing it with a conjugate stationary energy would
incorrectly conjugate the material problem.

The energy derivative separates the original mesh energy from the added
seam and mutual terms. Only the latter use the moving taper partition. The
original mesh energy is evaluated on its unsplit triangles, canceling its
artificial O(1/sqrt(gap)) partition-transport terms analytically. This keeps
the same energy and guard while avoiding cancellation of large quadrature
terms. The solve itself still differentiates the original full weak system.

The energy check separates the pure Fourier term as well. Multiplying two
retained plane waves reaches twice the retained frequency. At Fourier order
7, a 32-point rule resolves the mixed weak integrals but leaves about 1e-8
error in this higher-frequency energy term; 48-point integration reduces it
below 1e-13. The pure Fourier energy and its material/gap derivative are now
evaluated by an independent exact rectangle transform and checked against
the original triangle convolution. The mixed and auxiliary energies retain
their physical quadrature checks. This changes neither the assembled weak
system nor its analytical tensor JVP, and retains the 1e-10 energy guard.
Two regressions check 32/48 agreement, and a separate rectangle quadrature
verifies the exact term and its variation.

## Exact contact coefficients

At contact, the seam self energy divided by s is the background-weighted
Gram matrix of the interval trace. The Z_e coupling comes from the leading
gap-normal gradients of the original difference potentials. C_e contains
the body integral of

```text
epsilon [ grad(C)_x (1/2 - y/p_y) T_x - grad(C)_y T/p_y ]
```

and a distributional contribution at each end of the interval. The latter
is one half of the material-weighted x gradient from each adjacent side,
times the limiting signed trace. Both sides must be retained. Endpoints are
identified from the integer mesh topology; comparison with a rounded literal
coordinate can silently omit their entire contribution.

The physical tensor at zero uses the same closed mesh field as before. Its
first derivative depends on the finite differentiated contact rows and the
limiting opening coefficients. It does not require a finite derivative of
every normalized disappearing row. The implementation checks the finite
contact-row derivative and a separate derivative of the original physical
energy; the normalized full weak derivative is explicitly absent at zero.

In the retained k_y=0 normal sector the trace space spans the unrestricted
interface forcing. The resulting yy derivative is exactly

```text
C((epsilon - epsilon**2 / epsilon_background) indicator) / period_y.
```

That identity checks the coupled contact solve. It is not inserted as an
isolated replacement derivative.

## Independent checks and optical scope

The reference reconstructs free physical nodal gradients from rational chart
vertices and independent six-node polynomials. It adds raw standing waves,
integrates all retained/material operators together, and solves the original
unscaled nodal constraints. It uses neither the candidate's C/Z coordinates
nor its stable trace polynomial basis or coupling assembly. Independent
moving-gap references use two quadratures and two finite-difference steps,
with the unchanged 3e-9 normalized derivative guard. The original nodal
potentials use a common length unit, without a C/Z transformation. A
tenth-order central stencil at physical steps 0.2 and 0.1 keeps differencing
above that independent solve's roundoff floor; quadratures 28 and 36 both
satisfy the guard. Smaller low-order stencils were insufficiently accurate
for this oracle and are not used as acceptance evidence.

Other checks cover contact values, the exact normal trace, positive-gap
approach to the full contact derivative, material holomorphy, original
reactions and energies, uniform media, translation, units and subpixel gaps.
These checks establish a finite constitutive response, not a Maxwell
continuum error bound. The separate optical study reports p and s results
across Fourier and mesh orders; it retains physical power and all original
residual checks.

The 24 coupled tests, 53 seam-component tests and 52 documentation checks
pass. The numerical checks used NumPy 2.5.1 on an Apple Silicon host. The
optical fixture is the same lossless 700 by 700 cell, width 350, epsilon
12.25, background 1, film thickness 300, wavelength 900, theta 0.2 and phi
zero used in the original periodic study. The superstrate/substrate
permittivities are 1 and 2.25.

The committed eight-row study records both polarizations, gaps
0, 0.001, 1e-5 and 1e-9, two optical perturbation steps applied to the
analytical contact tensor JVP, all residuals, and source hashes:

| Fourier order | Mesh refinement | p contact dT/dgap | s contact dT/dgap |
| ---: | ---: | ---: | ---: |
| 3 | 0 | 0.00203677376 | 0.00465156953 |
| 3 | 1 | 0.00155608195 | 0.00465156950 |
| 5 | 0 | 0.00187030080 | 0.00018648870 |
| 5 | 1 | 0.00185494947 | 0.00018648864 |

A further four-row record at Fourier order 3 retains the same positive-gap
checks on the two finer meshes:

| Fourier order | Mesh refinement | p contact dT/dgap | s contact dT/dgap |
| ---: | ---: | ---: | ---: |
| 3 | 2 | 0.00148214497 | 0.00465156949 |
| 3 | 3 | 0.00147689623 | 0.00465156944 |

The two records above carry source hashes from `ad907d0`. The subsequent
exact Fourier energy check affects verification, not their constitutive
operators or tensor derivatives. Every fine-mesh row also passes the
original and energy guards. The p slope approaches the finite-order Li
comparison with mesh refinement, while the s slope remains stable. This
is a finite-order check, not a continuum stopping rule.

After exact integration of the pure Fourier verification term, the order-7
record also passes the unchanged original, energy and physical power guards:

| Fourier order | Mesh refinement | p contact dT/dgap | s contact dT/dgap |
| ---: | ---: | ---: | ---: |
| 7 | 0 | 0.00192106198 | -0.00077857504 |
| 7 | 1 | 0.00194293045 | -0.00077857503 |

The s slope now has the negative sign of the independent finite-order
contact reference, rather than the positive coarse-mesh value of the
unenriched family. The p slope is not monotone across these two coarse
meshes and still differs from the Li comparison near 0.00189155. The
order-3 refinement result must not be generalized to all Fourier orders.

The explicit Li comparison's one-sided 1e-5 gap secants are approximately
0.00147641711 and 0.00465156096 for p/s at order 3, and 0.00181767279 and
0.00018650999 at order 5. They retain finite-step error and are not an exact
continuum oracle. The s contact slope is stable between the two meshes and
consistent with this independent comparison. The remaining p discrepancy
is still significant at these coarse meshes.

All eight rows pass physical power and the original/energy/reciprocity
guards. The maximum recorded original or energy residual is below 7e-13.
At gap 1e-9 the transmission changes by about 2e-13 to 5e-12, without the
original closing-gap step. Positive-gap secants approach the contact
derivative more slowly than a quadratic Taylor model, as expected from the
cubic taper's O(gap**1.5) energy remainder.

This result addresses the missing periodic normal trace through the actual
full field solve. It does not establish a general Fourier/mesh stopping
criterion, metal-pair convergence, tilted resonant convergence, or backend
and public-builder integration.

Direct use is through
`wavebit._core.conforming.periodic_seam_galerkin.PeriodicSeamGalerkinGapCell`.
It returns the five convolution slots, their directional derivative and
diagnostics. It remains an explicit NumPy kernel; there is no ordinary NVF
dispatch or JAX/Torch integration. No new external dependency or copied
source is introduced.
