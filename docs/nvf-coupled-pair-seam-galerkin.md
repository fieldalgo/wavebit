# Continuous real gap traces in the complete pair tensor

Written: 2026-09-14

Targets:
- `wavebit/_core/conforming/cartesian_pair_seam_galerkin.py`
- `wavebit/_core/conforming/real_trace_coupling.py`
- `tests/test_nvf_pair_seam_galerkin.py`
- `studies/nvf_coupled_pair_seam_galerkin.py`
- `studies/results/nvf_coupled_pair_seam_galerkin.log`

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

The 22 pair tests, 24 periodic coupling tests, 53 seam-component tests and
53 documentation checks pass. Numerical verification uses NumPy 2.5.1 on
an Apple Silicon host. No external
solver source or new dependency is integrated. The optical study retains the
existing dielectric/metal fixture, both polarizations, all physical guards,
source hashes and explicit Li comparisons. Li is a comparison at the same
retained order, not a continuum oracle or an automatic fallback.

## Recorded optical result

The committed record contains the eight completed order-4 rows at mesh
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
and stable high-refinement elimination still require development. General
shapes, oblique lattices, automatic NVF routing and other backends are outside
this class's current contract.
