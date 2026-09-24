# Integrated anisotropic shapes

Tensor `Material` assignments to shape backgrounds and inclusions default to
`fourier_coefficients='integrated'`. All nine tensor components are supported.
The construction applies separately to epsilon and mu, with each quantity's
own material contrasts. See [full tensor patterns](full-tensor-patterns.md).

```python
import wavebit

crystal = wavebit.Material([2.25, 4., 2.8]).rotated(0.31)
solver = wavebit.rcwa(
    wavelength=730., period=(600., 600.), thickness=[173.],
    structure=[dict(background=1., shapes=[
        dict(type='circle', lx=217., material=crystal),
    ])],
    truncation_order=(3, 3), shape_nvf_grid_size=128,
    fourier_coefficients='integrated',  # the default for tensor shapes
)
result = solver.solve()
```

## What is integrated

Let a region have constant material tensor `e`, a real unit normal extension
`n`, and projector `P = n nᵀ`. The full 3×3 Popov–Nevière rearrangement is

```text
a = nᵀ e n
B(e, P) = I + (P - P e) / a
A(e, P) = e B(e, P)
```

For each homogeneous region, Wavebit samples B's smooth extension throughout
the periodic cell, constructs its trigonometric interpolant, and integrates
that interpolant over the region. With indicator coefficients `F_r` and
interpolant coefficients `b_r`, its contribution at Fourier order `k` is

```text
B_r[k] = sum_q F_r[k-q] b_r[q]
B[k]   = sum_r B_r[k]
A[k]   = sum_r e_r B_r[k]
effective tensor = convolution(A) @ inverse(convolution(B))
```

The background indicator is one minus the union of the resolved regions.
The sums use the complete interpolant spectrum, with sufficient frequency
padding to avoid circular aliasing at all retained difference orders. Even-grid
Nyquist modes are split between both signs; otherwise a real normal-derived
function would acquire an artificial imaginary part between sample points.

This is **not** multiplication of separately truncated material and projector
convolution matrices. All products are integrated before the RCWA truncation
and final block closure. For a uniform material, `A = e B` remains exact,
including when an artificial shape divides regions with identical tensors.
When z is decoupled, zz is integrated directly. With z coupling, all three
components close together, and the resulting zz operator is inverted.

## Geometry and remaining approximations

- Disjoint circles, ellipses and rectangles use analytic indicator transforms.
- Overlaps paint later shapes over earlier ones. The existing contour resolver
  produces disjoint polygon regions, including holes. Rectangle contours are
  exact; curved painted regions use 256-vertex polygon approximations.
- One primitive uses its analytic geometric normal extension. Multiple shapes
  use the resolved polygon interfaces and the existing contrast-weighted normal
  extension. Contrast and equality read all tensor components, separately for
  epsilon and mu; equal tensor traces do not erase an interface.
- The normal extension and PN interpolant remain numerical. Increasing
  `shape_nvf_grid_size` refines them; it does not move the material boundary to
  a pixel edge. Curved multi-shape normals retain polygon approximation too.
- Periodic self-overlap and painting overlaps through cell edges are refused
  by the integrated geometry resolver. Select `fourier_coefficients='sampled'`
  for those geometries. A primitive may cross the cell edge provided its
  periodic images do not overlap.

The supported tensor projector is `linear`. Neither this path nor the sampled
tensor path promises exact energy conservation at finite Fourier order.
Converge the observable in Fourier order and normal grid; curve polygonization
and contact topology are additional limits, not errors removed by a finer grid.

## Derivatives and frozen geometry

NumPy, JAX and PyTorch use the same integral. JAX JIT requires a concrete
`freeze_geometry()` record before tracing. The record stores intersection and
interface topology, while material values and shape coordinates remain live.
Position, size, angle and lattice derivatives therefore include the continuous
boundary contribution rather than a zero derivative through a Boolean mask.

Eager replay rejects stale topology. Under JIT those discrete decisions cannot
be rechecked; re-freeze outside the trace when moving through contacts or
overlap changes. Medial-axis ties in the sampled normal extension can still
affect derivative convergence. The explicit sampled route continues to refuse
shape and lattice derivatives rather than returning misleading zero gradients.

## Validation

`tests/test_tensor_shape_integrated.py` checks weighted Fourier integrals against
independent Gaussian quadrature, including odd/even grids and Nyquist modes;
analytic anisotropic laminate formulas; uniform tensor identities; continuous
translation phases; painting areas and holes; backend agreement; and geometry
and material derivatives against finite differences.

`tests/references/test_tensor_shape_golden.py` compares three gyrotropic
rectangular pillars against the existing independently generated fmmax goldens.
At order 14 and a 256-point normal grid, maximum R/T deviations from the nearer
fmmax formulation were approximately `1.27e-6`, `2.92e-6` and `1.45e-5`.
Each is inside the corresponding spread between fmmax's two formulations
(`4.84e-6`, `2.97e-5`, `1.67e-4`). The reference files were not regenerated.
Those checks cover the stated geometries, not arbitrary high-contrast or
resonant structures.

For a 217 nm circle in a 600 nm square cell with epsilon principal values
`[2.25, 4., 2.8]`, rotated by 0.31 rad, one 173 nm layer at wavelength 730 nm,
theta 0.23, phi 0.19 and Fourier order `(2, 2)`, the `result.R.sum()` values were:

| Normal grid | R |
| --- | --- |
| 32 | 0.0317163161 |
| 64 | 0.0317127505 |
| 128 | 0.0317118686 |
| 256 | 0.0317116465 |
| 1024 | 0.0317115750 |

This isolates normal-grid convergence at fixed Fourier order. It is not an
optically converged reference: R+T is about 0.99794 at that low order.
