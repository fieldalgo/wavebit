# Continuous real gap traces in the complete pair tensor

Written: 2026-09-14

Targets:
- `wavebit/_core/conforming/cartesian_pair_seam_galerkin.py`
- `wavebit/_core/conforming/real_trace_coupling.py`
- `tests/test_nvf_pair_seam_galerkin.py`
- `studies/nvf_coupled_pair_seam_galerkin.py`
- `studies/results/nvf_coupled_pair_seam_galerkin.log`
- `studies/results/nvf_pair_seam_retained_contact.log`
- `studies/results/nvf_pair_positive_retained.log`

TL;DR: The internal rectangle pair now has continuous real gap potentials in
its complete quadratic Galerkin solve. The independent original nodal system
and positive-gap variations agree with the implementation. Exact contact uses
a physical energy limit, not a tensor block replacement. Metal optical
convergence remains unresolved; adding the opening trace cannot correct an
underresolved closed-body field.

| Property | Finite contract |
| --- | --- |
| Geometry | Two internal horizontal rectangles, equal fixed height, positive diagonal periodic basis |
| Mesh | Existing explicit opening plan, quadratic potentials, refinements 0 through 3 |
| Fourier set | Independent nonnegative x/y orders; added real traces use the y order |
| Live direction | Gap, two complex permittivities and background together |
| Contact | Exact right derivative of the physical finite family |
| Material admission | Passive materials in the existing common coercivity half-plane |
| Backend | Explicit NumPy finite kernel; ordinary NVF routing is unchanged |
| Optical qualification | Separate mesh/Fourier convergence measurement, still incomplete |

The separate [polynomial extension](nvf-polynomial-pair-coupling.md) adds
degrees 3 and 4 with real invertible trace coordinates and direct physical
contact assembly. The quadratic default and measurements below retain their
original scope.

## One physical field space

Let the normalized gap be s, with the seam at x=0, widths a and b and common
height h. Each new potential is f(x) T_j(y). Its normal profile is

```text
f(x) = 0                             outside the two outer faces
       -s*(x+a+s/2)/(2*a)            -a-s/2 < x < -s/2
       x                            -s/2 < x < s/2
       s*(b+s/2-x)/(2*b)              s/2 < x < b+s/2
```

Thus the potential joins 0, -s/2, +s/2 and 0 continuously. T is a real
restricted Fourier basis with the same cubic tangential taper as the
[periodic seam](nvf-seam-trace.md). The taper width decreases as
sqrt(s); body energy from its tangential derivative is O(s**1.5), whereas
the gap normal energy is O(s). No material enters the potential basis.

All old and new fields share the same original equations:

```text
E = Phi*x + L*a
x + Q*a = I
K*a + S*x - Qtest*H = 0
H = A*x + R*a
```

The five output slots come from this whole tensor and its analytical
variation. The new fields couple to every original contact/difference
potential. They do not overwrite a normal block, use a complex fitted frame,
or invoke Cartesian Li inside the implementation.

The common assembly and physical checks now live in `real_trace_coupling.py`.
The periodic geometry delegates to the same code without changing its weak
operator, contact formulas or acceptance guards.

## Moving intersections without losing thin triangles

A horizontal sweep partitions each original triangle at its vertices, cubic
taper knots and intersections with the four moving vertical profile lines.
Physical quadrature nodes, weights, nodal gradients, new trace gradients and
plane waves are differentiated together.

The two collapsing annulus triangles need special arithmetic. Their left
and right edges approach the same cell-scale coordinate, so subtracting
rounded edge positions loses their width as gap decreases. The sweep keeps
the common closed edge and the exact gap times the opening-width coefficient
separately. Barycentric coordinates are interpolated between the two original
edges rather than recovered by inverting a nearly collapsed physical
triangle. This keeps the original nodal reaction and its derivative below
the same 1e-10 guard at a caller-unit gap of 1e-9.

Material edge identities come from the fixed rational nodal graph. Local
control coordinates retain those identities under unit changes; they are
not recovered by subtracting translated cell coordinates or nudging a
nearby floating-point vertex.

## Exact contact and original energy

The body limit F=f/s has a unit jump at x=0. Its normal weak moments therefore
include the background gap measure plus the two regular body integrals. At
each tangential material endpoint, the shrinking cubic taper contributes
one-half of each adjacent material's trace. Both sides are retained in the
contact/trace coupling and Fourier source.

The new physical columns vanish at contact. Their first derivatives and the
finite normalized weak rows determine the contact response. The normalized
disappearing rows themselves can have divergent derivatives; the code does
not invent finite values for those derivatives. It solves the finite contact
row variation and independently checks the stationary reciprocal bilinear
physical energy. The closed value equals the original quadratic mesh value.

Positive gaps reconstruct every original nodal reaction, including the gauge,
and the original complex field energy. The stationary energy variation keeps
the auxiliary coefficients fixed and differentiates the retained constraint.
The unsplit original-mesh terms and moving added terms are integrated
separately to cancel artificial partition transport analytically. Pure
Fourier energy uses an independent exact two-rectangle transform; this also
checks the triangle convolution and its simultaneous material/gap variation.
All residual guards remain 1e-10, reciprocity guards 1e-9, with positive rotated
storage and no negative dissipation beyond the existing roundoff allowance.

## Independent verification and scope

The independent positive-gap reference uses a six-node Vandermonde, raw free
nodal variables, raw sine/cosine traces, convex polygon clipping and separate
triangle quadrature. It does not use the candidate sweep, trace basis,
normalized weak matrix or contact formula. Quadrature orders 16 and 24 agree
with the complete candidate tensor within 3e-10. Fourth-order variations of
the clipped reference at two step sizes check the gap JVP with the unchanged
3e-9 normalized tolerance.

Other checks cover exact closed values, the predicted sqrt(gap) remainder in
the positive-gap JVP, material holomorphy and additive simultaneous directions,
uniform media, unit changes, translations, directional Fourier orders and
refined subpixel gaps. The directional subpixel test retains the nonzero
first-order change; a small positive gap is not required to equal contact
when the actual finite derivative is large.

The 25 pair tests, 26 periodic coupling tests, 53 seam-component tests and
53 documentation checks pass. Numerical verification uses NumPy 2.5.1 on
an Apple Silicon host. No external
solver source or new dependency is integrated. The optical study retains the
existing dielectric/metal fixture, both polarizations, all physical guards,
source hashes and explicit Li comparisons. Li is a comparison at the same
retained order, not a continuum oracle or an automatic fallback.

## Recorded optical result

The first committed record carries the `7d199ef` source snapshot and contains
the eight completed order-4 rows at mesh
refinements 0 and 1: equal dielectric and mixed dielectric/metal materials,
both polarizations, and all four gaps 0, 0.001, 1e-5 and 1e-9. Its header
also requests order 8, but the explicit coverage row marks that larger run
incomplete; no order-8 result is claimed from this record.

The fixture has a 700 by 700 cell, two widths of 300, height 300, film
thickness 300, wavelength 900, theta 0.2, phi 0, top 1, bottom 2.25 and
background 1. Equal materials are (4, 4); the metal pair is (12.25, -10+1j).
Slopes below differentiate the optical output along the analytic contact
tensor JVP, using centered steps 0.00005 and 0.000025. They are not gap
secants used to invent the contact tensor derivative.

| Materials | Refinement | p contact dT/dgap | s contact dT/dgap |
| --- | ---: | ---: | ---: |
| Equal dielectric | 0 | 0.000000505 | -0.000150058 |
| Equal dielectric | 1 | 0.000008522 | -0.000152106 |
| Dielectric/metal | 0 | 0.014727933 | 0.074350490 |
| Dielectric/metal | 1 | -0.186568 | -0.0233903 |

Every recorded row passes the original, energy, reciprocity and physical
power checks. The metal slopes still change strongly with mesh refinement,
and differ from the order-4 Li comparison near -0.017255 and +0.009315.
The finite operator is verified; these data do not meet a metal optical
convergence criterion. Refinement of the closed field, higher Fourier orders,
and positive-gap qualification on these fine meshes still require development. General
shapes, oblique lattices, automatic NVF routing and other backends are outside
this class's current contract.

## Stable elimination of the closed physical equations

At exact contact the added and disappearing physical columns are zero, so
the closed field can be solved before the gap-response coefficients. Forming
a large projected matrix K-SQ-Qtest*(R-AQ) first magnifies roundoff on fine
meshes. The contact implementation now uses the existing original physical
K_C to eliminate the closed nodal variables into retained field and flux
unknowns whenever that retained system is smaller. Coarse cases keep the
algebraically equivalent potential elimination.

With V=K_C**-1 Qtest_C and W=K_C**-1 S_C, the retained matrix is

```text
[ I-Q_C*W        Q_C*V    ] [x] = [I]
[ -A+R_C*W      I-R_C*V  ] [H]   [0]
```

The contact derivative uses the same matrix with the complete differentiated
physical forcing, including every disappearing-potential contribution. This
changes neither the trial space nor the original weak equations. Both
eliminations agree for dielectric and metal controls with simultaneous gap
and material directions. The highest-refinement order-8 metal case is now a
regression under the original, unchanged physical-energy guard. Periodic
contact regressions also include refinement 2 under this elimination.

On the order-8/refinement-3 metal control, the former energy derivative
residual was about 3.26e-10 and was refused by the 1e-10 guard. Original-K
elimination reduces it to about 3.16e-11. No residual threshold, material,
Fourier order or mesh is changed. The reported `condition` still describes
the full projected contact matrix; `closed_elimination` identifies the
actual closed solve.

The second raw record uses `--contact-only` and contains sixteen completed
rows at Fourier orders 4/8 and refinements 2/3, for both materials and
polarizations. Every recorded exact-contact case passes the original
energy/reciprocity and optical power guards. Positive gaps on these fine
meshes are not qualified by this contact-only record.

| Fourier order | Refinement | Metal p contact dT/dgap | Metal s contact dT/dgap |
| ---: | ---: | ---: | ---: |
| 4 | 2 | -0.097133837 | 0.035238120 |
| 4 | 3 | -0.045289485 | 0.021832954 |
| 8 | 2 | -0.020622742 | 0.038088257 |
| 8 | 3 | 0.034894333 | -0.045718330 |

The sign changes at order 8 are a remaining optical failure, even though
the original finite equations are solved accurately. Stable elimination
removes a numerical obstruction; it does not establish metal optical
convergence or justify production NVF dispatch.

## Positive-gap elimination and coincident sweep vertices

Positive gaps now use the same original-K elimination already available in
the underlying Galerkin solver when the full nodal/trace system is larger
than its retained field/flux system. Coarse cases keep potential elimination.
The matrix and its complete analytical variation still describe the same
real field space. Eight controls compare both eliminations for internal and
periodic gaps, dielectric and metal materials, gaps 7 and 1e-9, and simultaneous
gap/material directions. Every original nodal and physical energy check is
retained. At positive gaps, `elimination` identifies the chosen formulation,
and `condition` describes that formulation's solved matrix.

Physical field reconstruction applies the combined quadrature/component
rows in one dense multiplication. It retains the same fields, quadrature
points, energy contractions and residual scales. This removes many separate
small batched multiplications; no general speedup is inferred from that
change alone.

The first order-4/refinement-2 positive-gap run exposed a separate geometry
failure before it could solve the enlarged equations. Two vertices on the
same horizontal edge had different barycentric representations. Independently
rounded control products gave their y coordinates a one-ulp difference. The
sweep then created a fictitious strip whose rounded midpoint was its upper
endpoint, leaving no active edge pair.

The local chart now combines equal-coordinate control weights as integers
before multiplying by physical lengths. Symbolically identical coordinates
therefore have identical floating representations. This is an exact graph
identity; it does not merge nearby physical boundaries by a tolerance or
skip narrow sweep intervals. Independent rational-geometry checks verify
each triangle's area and area derivative at refinement 2 for gaps 0.001 and
1e-9, including the vanishing thin-triangle areas with a relative guard.

The positive-gap raw record preserves the failed source snapshot followed by
the corrected rerun. Both polarizations complete at order 4, refinement 2,
quadrature 32, with the metal fixture and gaps 0/0.001. Current source hashes
match the corrected snapshot. At the positive gap the original nodal value
and derivative residuals are below 3.81e-14; energy value and derivative
residuals are below 4.17e-12, against the unchanged 1e-10 guards.

| Polarization | Contact T | T at gap 0.001 | Contact dT/dgap | Gap secant |
| --- | ---: | ---: | ---: | ---: |
| p | 0.454192751 | 0.454095364 | -0.097133836 | -0.097386443 |
| s | 0.398871799 | 0.398907274 | 0.035238119 | 0.035474886 |

This run took about 95 seconds on its host, including assembly, all physical
checks and both optical readouts; it is not a comparative timing benchmark.
It verifies this formerly blocked finite positive-gap case. It does not
qualify all finer meshes or all gap samples. The contact slope remains the
same underresolved metal result, and the optical convergence failure above
remains open. Closed-cell approximation is now measured separately by the
[conforming electric/flux error evaluator](nvf-galerkin-error.md).
