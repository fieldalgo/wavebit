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

This is a cell-kernel extension. The public `ConformingNVF` shape adapter
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
comparison. The baseline is the affine implementation at `67cff19`; both
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

For the ungraded degree-2 case with `epsilon=4+.8j` and
`J=[[1,.45],[0,.7]]`, the largest component discrepancy against a
quadrature-24 oracle was `1.142e-7` before factoring the body correction
and is below `7.58e-11` afterwards. A separate quadrature-32 check changes
the oracle by less than `5.91e-11`. These observations concern numerical
differentiation of the stated finite cells. They establish neither
continuum opening-derivative convergence nor an optical contact derivative.

Run details are recorded in
[the validation log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_affine_cell_validation.log).
General physical-rectangle input, an explicit public affine-shape contract,
and optical convergence qualification remain open.
