# Continuous lifts of the periodic normal trace

Written: 2026-09-14

Targets:
- `src/wavebit/_core/conforming/periodic_seam_trace.py`
- `tests/test_nvf_seam_trace.py`

TL;DR: A real, continuous positive-gap potential supplies the unrestricted
normal contact trace missing from the fixed polynomial mesh. Its finite yy
response and gap/material derivative pass independent physical-field energy
and contact checks. This is a constitutive component for retained k_y=0
fields, not a full transverse closure or a replacement NVF dispatcher.

| Question | Established result | Limit |
| --- | --- | --- |
| Can the trace come from actual positive-gap fields? | A tapered Fourier trace times a continuous periodic y potential has finite energy at every positive gap. | The construction is for a rectangular bar with a horizontal periodic seam. |
| Does its forward response approach the unrestricted contact derivative? | The body correction energy is o(g); the leading gap Gram spans the required retained Fourier traces. | The remainder is generally O(g**1.5), not a second-order Taylor remainder. |
| Are the computed matrices energies of those fields? | Independent two-dimensional integration of raw standing-wave potentials and reconstructed fields agrees within 3e-10. | The checked component is yy in the k_y=0 sector. |
| Is the material derivative complex-linear? | The basis is real and material-independent; dielectric and passive-metal material/JVP controls pass. | A common material half-plane remains required. |
| Is the whole NVF complete? | No. | Mesh coupling, other retained y modes, metal-pair optical convergence and general-geometry/backend integration remain separate. |

## The field, rather than only its derivative, changes

The [periodic Galerkin analysis](nvf-periodic-galerkin.md) identifies a positive
normal-trace projection error at contact. A fixed continuous polynomial trace
cannot represent the jump at each end of the material interval. Shrinking an
x transition proportionally to gap can give its body field O(g) energy, which
survives in the first derivative and creates another finite approximation error.

Let the normalized periods be p_x and p_y, with cell area one, normalized gap
s, body height h = p_y - s and bar width w. The scalar potential is

```text
potential_j(x, y) = T_j(x) f(y)

f(y) = y                          in 0 <= y <= s/2
       s (p_y/2 - y) / h          in s/2 <= y <= p_y - s/2
       y - p_y                    in p_y - s/2 <= y <= p_y.
```

It is continuous, periodic, and real. Its derivatives are used directly in
the original material energy. T_j is a real retained Fourier polynomial
multiplied by a hat that is one inside the bar and falls to zero over
a width 2 delta centered on each vertical material boundary. Each transition
uses the cubic smoothstep 3 t**2 - 2 t**3; its first x derivative is continuous
at both ends. This also avoids a derivative jump when a taper endpoint crosses
a fixed mesh edge in a future coupled energy. The taper is
material-independent:

```text
cap = min(w, p_x - w) / 4
raw = sqrt(s h / 12)
delta = cap raw / (cap + raw).
```

The cap keeps the two transitions separate and inside the periodic cell.
Delta approaches a constant times sqrt(s) at contact. The gap-normal energy
is O(s); the body tangential energy at the tapered ends is O(s**2/delta),
which is O(s**1.5) and therefore does not contaminate the first derivative.
No sampled normal projector, reference material, or optical target enters
this choice.

## Stable real traces on narrow bars

Raw cosine/sine columns restricted to a narrow interval become nearly
dependent. An equivalent trigonometric basis avoids subtracting those large,
similar columns. Set u = sin(pi x/p_x) / sin(pi w/(2 p_x)) around the bar
center. Even Legendre polynomials P_0(u), P_2(u), ..., P_2n(u), together with
cos(pi x/p_x) times the odd polynomials through P_(2n-1)(u), span exactly the
real Fourier space of order n.

The polynomial recurrence supplies values and two spatial derivatives. A
real, material-independent QR transform orthonormalizes the contact trace;
it drops no mode and uses no rank cutoff. The code checks the resulting Gram
matrix and refuses an unresolved requested basis. Predetermined contact
tests cover orders 0, 3 and 9 at width fractions 0.2, 0.5 and 0.8 for both
dielectric and passive metal materials.

## The finite weak energy and its contact limit

For retained y-directed fields with k_y=0, the unweighted projection of every
auxiliary gradient vanishes. The x component integrates the antisymmetric
f(y) to zero; the y component integrates its periodic derivative to zero.
Consequently, no projection subtraction is needed in this sector.

Let M and X be the x Gram matrices of T and its x derivative, respectively.
A subscript b means background weighting; a subscript e means weighting by
the physical, piecewise x material in the body. Exact y integration gives

```text
K / s = M_b + (s / h) M_e + (s**2 / 12) X_b + (s h / 12) X_e
R / s = (epsilon_background - epsilon) <Phi.H T>_bar
H_yy = A - s (R/s) solve(K/s, (R/s)[reverse].T).
```

A is the direct convolution of the actual rectangle. The source is a mode
reversal and transpose, not a conjugate of the material-weighted R. This
preserves the original complex material equations. Numerical quadrature
splits at every material and taper endpoint and differentiates both the
moving nodes and their weights.

At contact, the normalized weak matrix tends to M_b and the real trace space
contains each retained plane wave restricted to the bar. Its limiting
quadratic minimization therefore yields

```text
H'_yy = C((epsilon - epsilon**2 / epsilon_background) indicator) / period_y.
```

The implementation solves that limiting Gram system; it does not insert an
optical or Li derivative. The positive-gap forward and analytical derivative
approach the same limit. Because the weak matrix divided by gap contains
O(sqrt(s)) terms, its derivative can diverge at zero even though the physical
response has a finite first derivative. The endpoint follows the original
energy limit. Its metadata reports no derivative of that normalized weak
system; positive-gap weak derivative residuals are checked normally.

## Independent qualification and remaining integration

The 53 tests include an independent two-dimensional quadrature in physical
coordinates, using raw standing-wave potentials rather than the candidate's
polynomial frame or analytical y integrals. They compare both the original
weak elimination and the reconstructed complex physical-field energy with
the returned matrix. An eighth-order central stencil checks the independently
integrated moving-gap values at two steps, with the unchanged 3e-9 normalized
derivative guard. The higher-order stencil avoids the roundoff floor of a
passive-metal resonance without loosening that guard.

Other controls cover exact unrestricted contact traces, positive-gap approach
to that derivative, 64/96 quadrature agreement, subpixel gaps, physical units,
translation, periodic potential continuity, sampled gradients, uniform media,
material holomorphy and simultaneous gap/material directions. Sampled fields
also provide an analytical JVP for moving quadrature points and gap together.
Near the upper seam, a small negative y coordinate retains the subpixel voltage
without adding and subtracting a whole period. These small
NumPy checks ran with NumPy 2.5.1 on an Apple Silicon host. They establish
the stated finite component; no Maxwell optical convergence measurement is
inferred from them.

The component's `normal_response` returns one yy matrix and its derivative,
not five convolution slots. The existing finite-element Galerkin response
is unchanged. The separate [coupled seam family](nvf-coupled-seam-galerkin.md)
now assembles their mutual physical energy and retained projections into a
complete finite transverse tensor. Replacing only the old yy derivative, or
splicing this yy block into an otherwise unrelated tensor, would not perform
that coupling and is not implemented. The coupled family's independent
qualification and optical scope are documented separately. Ordinary routing,
other geometries and backend integration remain open.

The implementation uses the existing NumPy dependency and original formulas;
no new external dependency or copied source is introduced.
