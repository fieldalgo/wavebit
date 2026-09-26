# Full-metric conforming cells

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/affine_cell.py`
- `wavebit/_core/conforming/continuous_flux.py`
- `tests/test_nvf_affine_cell.py`
- `tests/test_nvf_affine_contact.py`
- `tests/test_nvf_complex_variational.py`
- `tests/test_nvf_continuous_flux.py`

The internal `AffineCellGapModel` solves a genuinely nonorthogonal cell for
an affine image of the existing reference strip profile. It includes the
mixed spatial derivatives in the weak equations and Fourier energy form,
an affine lift with the exact laminated contact flux, and conversion back
to laboratory electric-field components. Its matrices feed `solve_conv`
with the same physical lattice and retained order set.

The public [`AffineProfileNVF`](guide/affine-profile-nvf.md) adapter now
accepts explicit fractional width, height and ordered material intervals
on the supplied physical lattice. It preserves the original Fourier
labels, restores the physical background before energy closure and
translates the centered profile to the engine's Fourier origin.
The `ConformingNVF` shape adapter
still accepts physical rectangles on orthogonal point lattices, including
the [equivalent-basis construction](nvf-lattice-rebasing.md). A continuous
shear maps a reference rectangle to a parallelogram. The new kernel does
not silently apply that transformation to public rectangle dictionaries.
Physical rectangles repeated on a genuinely oblique point lattice require
a different geometry adapter and, generally, a different partition.

## Coordinates and physical fields

Let A contain the physical lattice vectors in columns, and let
`J = A / norm(A[:,0])`. Reference coordinates have period one in each
direction. Width, seams and gap retain the historical reference length
700 in the kernel inputs, so `g = gap/700`. Reference material intervals
occupy the body strip `-(1-g)/2 < y < (1-g)/2`; the remaining gap wraps
around the periodic boundary. The entire profile is mapped by A.

The spatial metric and weak equation are

```text
M = J^(-1) J^(-T)
integral grad(test)^H epsilon M (F0 + grad(corrector)) = 0.
```

M is real symmetric positive definite and generally has a nonzero xy
entry. Multiplying A by a common positive length scale leaves the metric
and constitutive matrices unchanged. Before determinant evaluation the
implementation scales the finite basis by its largest entry magnitude,
avoiding overflow or underflow from a common physical unit scale.
Nonfinite, complex, singular or left-handed bases raise eagerly. Invalid
traced bases yield NaN coefficients.

For reference field F and its metric energy G, the physical Fourier
coefficients are

```text
B = J^(-T) F J^T
G_physical = J G J^T.
```

For diagonal J these reduce to the existing
[rectangular aspect transform](nvf-rectangular-cells.md). Mode labels
remain the original integers k, with physical reciprocal vectors
`2 pi A^(-T) k`. There is no integer reduction in this kernel.

The kernel now accepts `order=(x,y)` as well as a square integer order.
`truncation_order` records the pair for the consuming engine. Projection
covers the full required coefficient envelope, but Q, G and the
longitudinal matrix retain only the requested modes before inversion.
The [independent directional comparison](nvf-affine-convergence.md)
shows why the two directions must be refined separately.

## The contact lift includes both forcing directions

Let `rho = 1/epsilon_body(x)`, `l = rho/mean(rho)` and
`c = Mxy/Mxx`. The reference affine field is

```text
F0 = [[l, c (l-1)], [0, 1]] = l X + Y
X  = [[1, c], [0, 0]]
Y  = [[0,-c], [0, 1]].
```

Its mean is the identity. At full perpendicular contact, the normal flux
for both columns is constant across every material seam. Using only
`diag(l,1)` would leave a material-dependent normal flux in the second
column whenever Mxy is nonzero. Thus adding mixed stiffness terms alone
would not preserve the exact laminated contact solution.

In physical coordinates the contact field is
`I + (l-1) n n^T`, where n is the normalized reciprocal vector
`A^(-T) [1,0]^T`. The normal follows a reciprocal vector, not the first
direct lattice vector. The resulting finite constitutive matrices obey
the analytic inverse rule in the normal direction and the direct rule
in the tangential direction, including complex scalar materials.

## Mixed weak terms and nonsingular contact rows

The x integration packs now optionally include
`integral epsilon conj(f') f` and its independently oriented reverse,
plus the potential/lift cross loads. Existing diagonal kernels do not
request these extra packs. Local-basis products retain polarized material
differences and positive Gauss weights at zero Fourier frequency; the
global basis keeps its polynomial integration path.

The body and gap use the same periodic potentials and gap-scaled
enrichments as the polynomial rectangular cell. The new solver assembles
all four directional products using exact small polynomial y integrals
and the material-integrated x packs. Enrichment test rows are divided
by g analytically before assembly. The exact laminated lift also makes
the body correction vanish at contact. The implementation now writes that
correction as `v = g w` and cancels g from its equations analytically.
Every row therefore has a finite limit at g=0; there is no small-gap
cutoff, conditional deletion, regularization, or custom derivative.
The residual threshold in the assembled equations before diagonal balancing
remains `1e-8`, and the contact roundoff allowance is unchanged.

This factorization prevents a small numerical body correction at g=0 from
polluting the opening derivative through the narrow interface basis.
Periodic derivative integrals, bubble derivative integrals, and the mean
body x load are set by their exact identities. In particular,
`epsilon_body * l` is constant, so the mean body x load is the integral
of a periodic derivative and equals zero, including for complex material.
The generic internal weak-assembly control retains an unfactored path for
arbitrary supplied lifts; the physical cell uses the factored path for
all valid gaps, with no switch near contact.

These trial potentials are H1-conforming. For positive gap and finite
resolution, their oblique normal flux is not guaranteed to be continuous
pointwise at every interface. The exact contact lift does not establish
general interface accuracy or optical convergence. Resolution settings
remain explicit, and this extension adds no optical stopping criterion.

## Sesquilinear projection

The Fourier Gram form includes

```text
epsilon [ Mxx conj(Fx) Fx + Mxy conj(Fx) Fy
        + Myx conj(Fy) Fx + Myy conj(Fy) Fy ].
```

Both mixed orientations are retained. Trial coefficients and test basis
factors are conjugated; epsilon and the Fourier phase are not. Affine
cross terms use X and Y above, including both `l` and `conj(l)`. The
existing energy closure receives Q, G and the longitudinal material
convolution. It does not symmetrize, clip eigenvalues, or substitute Li.

The kernel has unit background. As with the complex rectangular kernel,
positive-real relative material values may have imaginary parts of either
sign. The `conv_matrices(..., basis=A)` convenience method directly closes
that unit-background problem. A caller using relative values for a
different complex physical background must restore the background in G
and the longitudinal coefficients before closure.

## Bounded Fourier projection intermediates

The affine kernel now assembles the static weak problem from the zero
Fourier coefficient alone. After solving it, the kernel projects the body
and gap fields in consecutive batches of at most 16 x Fourier labels.
Every batch uses the same solved potentials and complete y coefficient
range. Concatenation restores the original x ordering and all nine field,
Gram and material components. The requested coefficient envelope, trial
space, physical map and full-space matrix closure are unchanged.

Previously, all Fourier-weighted x matrices remained live during the
dense static solve. Their storage grew with the number of projected
x labels times the square of the auxiliary x basis size. Batching bounds
the live projection packs by 16 labels and separates them from the static
solve. The output coefficients, dense static equations and subsequent
Maxwell matrices still grow with their respective problem sizes. JAX
derivatives retain whatever intermediates their compiled graph requires;
the NumPy allocation measurements do not establish a JAX memory bound.

On the fixed affine profile, x subdivisions 2, base rows 4, body degree 4
and gap degree 6 give 3,887 auxiliary unknowns. Two coefficient ranges
give these NumPy measurements during `coefficients`:

| coefficient cap | supported outer half width | previous traced peak | batched traced peak | previous median time | batched median time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 12 | 759.36 MB | 726.62 MB | 6.129 s | 6.242 s |
| 64 | 32 | 813.93 MB | 726.62 MB | 16.912 s | 16.668 s |

Traced peak allocation falls by 4.3% and 10.7%, respectively. The dense
static solve still dominates the batched peaks. Models and imports are
constructed before tracing, and native linear-algebra workspaces are
outside its scope; these are not resident-memory figures or full optical
solve benchmarks. Increasing cap 24 to 64 does not add traced peak memory
in this fixed-cell observation, although constructing the higher-order
model and storing its output still cost additional memory.

Each range uses four alternating before/after timing pairs, following one
warmup of each implementation. The cap-24 median increases by 1.8%, whereas
the cap-64 median decreases by 1.4%. All nine coefficient arrays agree exactly in every
comparison. The baseline is the affine implementation at `67cff19`
(landed as `f512bdc`); both
versions run in the same NumPy process with identical inputs and no
overlapping numerical task. This observation does not establish zero
overhead or a universal runtime ratio. The host uses Apple Accelerate;
the recorded OpenBLAS/OMP environment variables do not determine its
actual thread count.

The contact Fourier oracle now also crosses the default batch boundary
at coefficient cap 10, for both real and complex permittivity and three
shears. Positive-gap, three-material comparisons use the separate
diagonal-metric kernel with both local and global x bases. JAX value and
directional-derivative controls exercise both one and multiple batches.
Existing tolerances are retained. Raw timings and allocation scope are
recorded in the
[projection measurement log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_affine_projection_memory.log).

## Independent controls

The checks cover:

- Mixed x packs against independent piecewise Gauss integration for both
  local and global bases, with real and complex material profiles.
- Positive-gap weak equations against independent physical tensor
  quadrature, with complex trial functions and gap degrees 1, 2 and 4.
- All projected field and Gram components against independent tensor
  quadrature, including nonzero Fourier modes and polynomial degrees
  1, 2 and 4.
- Agreement with the existing diagonal-metric kernel across contact,
  positive gap, both x bases, and several body/gap polynomial orders.
- Exact contact Fourier fields and energy, physical rotation and common
  scale covariance, and positive storage/passive loss forms.
- NumPy, JAX and Torch engine consumption, analytic oblique contact
  inverse rules, lossless energy balance, absorbing layers and rigid
  rotation of positive-gap diffraction powers.
- JAX differentiation of shear, aspect, width and independent material
  storage/loss, plus a positive-gap opening direction, against two
  finite-difference steps.

At exact contact the opening derivative of mean energy has a separate
oracle. For a lossless profile the exact body field is stationary. The
first energy variation reduces to a one-dimensional trace minimization:

```text
r = integral conj(f) (epsilon_body-1) (M F0)_y
a = (Myy mass_background)^(-1) r
G'_reference = integral F0^T (1-epsilon_body) M F0
             - Myy a^T mass_background a.
```

The test derives this trace system independently of the coupled cell
solve and checks the physical energy derivative after the J transform.
It uses the default `edge_fraction=1e-5` and a `3e-9` absolute bound.
In the earlier implementation, forward finite differences of a general weighted
Fourier objective on this fine contact mesh did not provide a reliable `3e-6` oracle:
the probe's fourth-order errors ranged from roughly `4e-3` to `3e-5`
as its dimensionless step decreased from `2e-6` to `1e-9`. The contact
test therefore uses the independent energy variation, not an enlarged
finite-difference tolerance. The additional Fourier opening check below
now tests all nine projected components in selected affine cells.

## Contact Fourier variation without subtracting nearby solutions

The new oracle reconstructs the x basis by integrating physical flux hats
piecewise across mesh and material cuts. It evaluates the y nodal functions
and integrated Legendre bubbles independently. It uses no production weak
assembly, Fourier moments, field projection, or automatic differentiation.
The geometry helper supplies only the finite space being checked.

Let `a(x)` be the contact trace correction. It solves

```text
Myy integral conj(f) a = integral conj(f) (epsilon_body-1) (M F0)_y.
```

The first body variation is `W = grad(w - t a)`, where `t` ranges from
`-1/2` to `1/2`. A separate periodic body solve determines w from

```text
integral epsilon_body grad(test)^H M grad(w)
  = integral epsilon_body grad(test)^H M grad(t a)
    - integral_boundary conj(test_x) [M (F0 + e_y a)]_x.
```

For integer y Fourier label n, the disappearing body edge and new gap
contribute phase `(-1)^n`. Thus the field derivative contains the body
Fourier transform of W and the boundary transform of `e_y a`. The Gram
derivative contains both sesquilinear body cross terms with W, and the
boundary difference between the gap and body energies. The material
derivative is the boundary transform of `1-epsilon_body`. Physical tensor
components are then transformed by J as above.

Four cases cover real and complex material, mixed intervals, both shear
signs, body degrees 1/2/4, gap degrees 1/2/4/6, and uniform or graded y
meshes at the default `edge_fraction=1e-5`. Independent quadrature orders
16 and 24 agree within `3e-10`; JAX opening derivatives of all nine
components and all coefficient modes through cap 2 agree with the
independent variation within `3e-9` absolute per dimensionless gap.
The first complex-material case now explicitly splits its five Fourier
labels across three projection batches (batch size 2). It passes the
same independent quadrature and opening-derivative limits; the other
three cases retain the default batch size. This checks contact opening
across projection batches without adding a new physical accuracy claim.

For the ungraded degree-2 case with `epsilon=4+.8j` and
`J=[[1,.45],[0,.7]]`, the largest component discrepancy against a
quadrature-24 oracle was `1.142e-7` before factoring the body correction
and is below `7.58e-11` afterwards. A separate quadrature-32 check changes
the oracle by less than `5.91e-11`. These observations concern numerical
differentiation of the stated finite cells. They establish neither
continuum opening-derivative convergence nor an optical contact derivative.

## Contact opening under x refinement

The first spatial comparison fixes `A=700*[[1,.45],[0,.7]]`, width 350,
physical permittivity `4+.8j`, unit background and exact contact. The
opening parameter is dimensionless g: the reference gap is `700*g` and
the physical normal gap is `490*g` in the same length units. Outer
order 1 retains coefficient cap 2. Rows, body degree and gap degree are
all 2, with ungraded y, local flux basis, edge fraction `1e-5`, grading
strength 8, radius .01 and projection quadrature 12. Only x subdivisions
change from 1 to 2, increasing the auxiliary unknowns from 215 to 431.

Both spaces pass the existing independent q16/q24 bound `3e-10`,
JAX/independent opening-derivative bound `3e-9` and cell residual guard
`1e-8`. The respective maximum discrepancies are `6.07049e-11` /
`1.37889e-10` for quadrature and `7.59735e-11` / `1.61586e-10` for
JAX versus the independent variation. Cell residuals are `2.80323e-11`
and `1.43430e-10`. These limits check the two finite cells separately;
they do not make the two cells' derivatives equal.

| component | maximum absolute x2-minus-x1 derivative change over cap 2 |
| --- | ---: |
| Bxx | 2.18921e-4 |
| Bxy | 6.19188e-4 |
| Byx | 7.34523e-4 |
| Byy | 2.16716e-3 |
| Gxx | 1.09100e-3 |
| Gxy | 2.83072e-3 |
| Gyx | 1.73598e-3 |
| Gyy | 1.12082e-3 |
| material | 6.75322e-16 |

These are derivatives per dimensionless g from the independent q24
arrays. For example, Gxy changes by
`+1.82947843e-3 + 2.16009198e-3j` at mode `(-1,2)`. The mean Gyy
change is only `-2.50216619e-6 - 8.47190175e-7j`, so a mean-energy
comparison alone misses the larger nonzero-mode changes. The mean B
derivative stays at roundoff because the cell field has identity mean;
that identity is not a resolution criterion.

The JAX evaluations, including compilation, take 2.760 s and 2.631 s.
The two q16/q24 oracle pairs take .061/.137 s and .225/.495 s.
Cumulative process peak RSS reaches 729 MB and 885 MB in this sequential
run. These observations are not a runtime scaling claim or a memory bound.
Full settings, source hashes, signed component changes and retained
array identities accompany the measurement log.

An initial x4 extension saves its independent reference arrays but stops
before JAX: q16/q24 discrepancies reach `3.17115e-10` in Gxy,
`3.42389e-10` in Gyx and `4.46982e-10` in Gyy, exceeding the unchanged
`3e-10` bound. Its small independent linear residuals do not override
that check. This unaccepted point cannot extend the accepted spatial
derivative ladder to x4 or establish a convergence rate. The following
stability check preserves that failed record and produces a separately
validated x4 result.

### Stabilized independent variation

The independent oracle now forms the local element coordinate directly,
avoiding loss of digits when subtracting nearly equal physical coordinates
on the smallest graded elements. Kahan accumulation reduces cancellation
in the same weak products. Each trace or body solve uses one balanced LU
for its initial solution and three corrections, with compensated residuals
evaluated against the original weak system. The equations, finite spaces,
quadrature nodes, production kernel and acceptance guards are unchanged.
This addresses arithmetic stability without increasing spatial resolution.

New q16/q24 references qualify all three spaces. The integrated reference
arrays equal the separately checked arithmetic prototype exactly. The x1
and x2 production JAX arrays are reused without reevaluation after checking
all seven production source hashes and the physical/settings metadata;
each is compared anew with the changed oracle. Only x4 receives a new JAX
evaluation. The log separates the old JAX provenance from the new oracle
source and preserves both the original failed checkpoint and the new
complete records, with all nine component arrays saved before validation.

| x subdivisions | production unknowns | max q24-minus-q16 | max JAX-minus-oracle | cell residual |
| --- | ---: | ---: | ---: | ---: |
| 1 | 215 | 8.63001e-12 | 5.29690e-11 | 2.80323e-11 |
| 2 | 431 | 7.20837e-11 | 2.32593e-10 | 1.43430e-10 |
| 4 | 863 | 1.77110e-10 | 1.23021e-9 | 4.23393e-10 |

All three pass the same respective guards `3e-10`, `3e-9` and `1e-8`.
The accepted q24 differences below are derivatives per dimensionless g,
with the physical normal gap still `490*g` nm. The material row denotes
the physical permittivity coefficient, stored as `epsilon`.

| component | max absolute x2-minus-x1 | max absolute x4-minus-x2 |
| --- | ---: | ---: |
| Bxx | 2.18920562e-4 | 2.01228240e-5 |
| Bxy | 6.19188124e-4 | 5.94180900e-5 |
| Byx | 7.34523460e-4 | 7.04298835e-5 |
| Byy | 2.16715844e-3 | 2.07963313e-4 |
| Gxx | 1.09100308e-3 | 8.93730170e-5 |
| Gxy | 2.83072231e-3 | 2.71144089e-4 |
| Gyx | 1.73598458e-3 | 1.66690964e-4 |
| Gyy | 1.12081878e-3 | 1.06536749e-4 |
| material | 6.75322301e-16 | 6.77599955e-16 |

For x4-minus-x2, Gxy changes by
`+1.74892692e-4 + 2.07199573e-4j` at mode `(1,-2)`; its magnitude is
`2.71144089e-4`. The mean Gyy change is only
`-8.81919436e-7 - 2.73702560e-7j`. Both intervals retain their signs and
mode labels in the log. The old and new x2-minus-x1 oracle arrays differ
by at most `2.21337e-10`; the reused JAX arrays and their first interval
are identical. These two finite spatial differences do not establish a
convergence rate, remaining-error bound, continuum derivative accuracy or
optical derivative qualification.

The new x4 JAX evaluation takes 3.342 s including compilation, with
803,422,208 bytes of process peak RSS. Its independent q16/q24 evaluations
take 2.298/4.447 s. The JAX process did not recompute x1/x2, so its peak
cannot be compared with the earlier sequential process as memory scaling.
The full contact regression, including the compensated-residual
cancellation check, passes all five tests in 35.56 s.

Run details are recorded in
[the validation log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_affine_cell_validation.log).
The explicit fractional-profile public contract is implemented separately
from ordinary rectangle input. Its NumPy/JAX acceptance checks cover
physical material Fourier integrals, the analytic reciprocal-normal
contact rule, diagonal rectangle equivalence, exact mode selection,
rigid motions, physical scaling, passive complex backgrounds, live
geometry/material derivatives and invalid updates. General
physical-rectangle input and optical convergence qualification remain open.
