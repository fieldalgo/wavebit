# Full-metric conforming cells

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/affine_cell.py`
- `wavebit/_core/conforming/continuous_flux.py`
- `tests/test_nvf_affine_cell.py`
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
by g analytically before assembly. Every row therefore has a finite
limit at g=0; there is no small-gap cutoff, conditional deletion,
regularization, or custom derivative. The original-equation residual
threshold remains `1e-8`, and the contact roundoff allowance is unchanged.

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
For this fine mesh, forward finite differences of a general weighted
Fourier objective at contact do not provide a reliable `3e-6` oracle:
the probe's fourth-order errors ranged from roughly `4e-3` to `3e-5`
as its dimensionless step decreased from `2e-6` to `1e-9`. The contact
test therefore uses the independent energy variation, not an enlarged
finite-difference tolerance. This check does not certify all Fourier
opening derivatives or an optical contact derivative for oblique cells.

Run details are recorded in
[the validation log](https://github.com/fieldalgo/wavebit/blob/develop/studies/results/nvf_affine_cell_validation.log).
General physical-rectangle input, an explicit public affine-shape contract,
and optical convergence qualification remain open.
