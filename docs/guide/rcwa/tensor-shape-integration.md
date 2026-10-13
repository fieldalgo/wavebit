# Integrated anisotropic shapes

:::{note}
This detailed guide documents the retained `wavebit.compat.legacy` interface and its measured scope. For the current problem/numerics lifecycle, start with [the common guide](../common/lifecycle.md). Advanced capabilities are preserved here with explicit compatibility calls; their limits are unchanged.
:::

Tensor `Material` assignments to shape backgrounds and inclusions default to
`fourier_coefficients='integrated'`. All nine tensor components are supported.
The construction applies separately to epsilon and mu, with each quantity's
own material contrasts. See [full tensor patterns](full-tensor-patterns.md).

```python
from wavebit.compat import legacy as wavebit

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
  use the resolved interface plan and the existing contrast-weighted normal
  extension. Contrast and equality read all tensor components, separately for
  epsilon and mu; equal tensor traces do not erase an interface.
- In a layer whose shapes do not overlap, each circle, disk or ellipse enters
  that extension through its exact boundary rather than its planning polygon.
  At each sample it supplies the distance to the nearest boundary among its
  periodic instances, on any lattice, and that instance's level-set direction,
  `sign(f) grad f / |grad f|` for the shape's implicit function `f`: from the
  curve toward the sample on both sides, oriented as the polygon segments'
  vectors are, and the outward normal on the curve itself. At an ellipse's
  centre, where the gradient vanishes, it carries the minor axis. A circle's
  direction is radial, and its `angle`, which describes nothing, is ignored.
  Rectangles, rectilinear contacts and painted overlaps keep their resolved
  polygon interfaces.
- For a circle the level-set direction is the nearest-point vector, from the
  nearest boundary point toward the sample, which the polygon route converges
  to as its facets refine. For an ellipse it is not: off the curve the two
  differ by up to 60-80 degrees, and the nearest-point vector switches sides
  across the ellipse's medial axis, the stretch of the major axis between the
  centres of curvature of its tips. Turning or moving the ellipse carries
  normal-grid samples across that stretch, and autodiff misses the steps they
  make, at every grid ([autodiff](../common/autodiff.md)): each step shrinks as
  `1/n**2` in `n = shape_nvf_grid_size` while the samples crossed grow as
  `n**2`. The level-set direction has no such switch, and it is the
  one-primitive extension's own. A lone ellipse and the same ellipse beside a
  small box of the background's own
  material, which the blend gives no weight, agreed to 7e-16 in the convolution
  matrices (32-point grid, order 3) on a ring ellipse and on an in-cell one
  whose nearest centre is its nearest boundary at every sample. The lone path
  reads the nearest centre's instance; where that is not the nearest
  boundary's -- 9-25% of the samples on three in-cell scenes -- the two read
  different instances, and the convolution matrices differed by 1.8e-5 to
  1.1e-4 on 32 points (3e-6 to 6e-6 on 64) and total transmission at order 6
  by up to 1.6e-5 (4.8e-7).
- The choice was made on what a gradient sees: the gradient integrated over a
  window of a design parameter against the loss's change across it. A central
  difference with a sub-pixel step stays on one smooth piece and agrees with
  autodiff by construction, so it cannot see a step. On a hexagonal cell holding
  a tensor ellipse beside a box and a circle, at order 1, the gradient
  integrated over a 0.1 rad window of the ellipse's angle had the wrong sign
  under the nearest-point vector on 32-, 64- and 128-point grids (-0.84, -0.77
  and -0.79 times the loss's change); at order 2 it was 2.7 and 2.6 times the
  change on 32 and 64 points, and 6.7 times at order 5 on 32. With the level-set
  direction the same windows closed to within 0.5%, 0.2% and 0.05% at order 1,
  0.9% and 0.1% at order 2, and 2.3% at order 5. On the four windows of
  `tests/rcwa/test_tensor_curve_normals.py` -- two of an ellipse's angle and two of
  its position, order 1 -- the level-set direction's integral was within 0.5%,
  1.4%, 0.5% and 0.7% of the change on 32 points and within 0.2% on all four on
  64, where the nearest-point vector's was -0.84, -0.16, 1.30 and -0.17 times
  it on 32 points and -0.77, -0.21, 1.29 and -0.37 on 64.
- An ellipse's centre is the level-set direction's one singular point. It does
  not step: a sample that a motion carries past the centre at a distance `d`
  turns through a half-turn, continuously, over a stretch of the motion of about
  `d (b/a)**2` along the minor axis and `d (a/b)**2` along the major, for
  semi-axes `a > b`. The gradient peaks there roughly as `1/d`. Moving the
  280 x 110 ellipse of that file's square scene along its minor axis past a
  column of samples (32-point grid, order 1), the largest `|dT/dy|` over a
  6 nm window was 6.9e-5, 2.7e-4, 2.0e-3 and 2.0e-2 at `d` = 9.4, 1, 0.1 and
  0.01 nm, against a median of 3.6e-5 to 5.0e-5; integrated on 0.1 nm
  abscissae, the gradient came to 1.004, 1.008, 5.5 and 49 times the loss's
  change, the last two because the abscissae skip a turn 0.015 and 0.0015 nm
  long. The test file's position windows therefore take abscissae finer than
  half the shorter stretch at their nearest pass.
- The other switches remain. A 200 x 90 box turned through the same 0.1 rad
  window in the hexagonal cell, on the polygon route, missed 40%, 23% and 12%
  of the loss's change on 32-, 64- and 128-point grids: its nearest-face
  switches, a bias that shrinks only about as the grid spacing does.
- What the level-set direction costs is convergence in Fourier order. On four
  scenes -- square, oblique and hexagonal cells, tensor backgrounds and
  inclusions, `shape_nvf_grid_size` 256, orders up to 16 or 18 -- the order from
  which all later values of total transmission lie within a band 1e-4 wide was
  16, 11, 11 and 9 with the level-set direction, against 7, 8, 8 and 9 with the
  nearest-point vector. Within +-1e-4 of the limit, the mean of both rules' last
  four orders, they settle from orders 9, 9, 11 and 8 against 7, 6, 7 and 9.
  The rms error over orders 9-13 was 6.0e-5, 4.4e-5, 8.7e-5 and 1.1e-5 against
  2.5e-5, 3.3e-5, 2.6e-5 and 1.6e-5. On the first of those, the scene of
  `tests/rcwa/test_tensor_curve_normals.py`, the order-16 solve took 54 s against
  3.0 s at order 9 and 1.0 s at order 7.
- The normal extension and PN interpolant remain numerical. Increasing
  `shape_nvf_grid_size` refines them; it does not move the material boundary to
  a pixel edge. Curves in painted overlaps retain the polygon approximation too.
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

Eager replay rejects stale topology. For the curves of a disjoint layer the
record holds which periodic instances each curve is searched over, and only
that, and a replay is refused only when a fresh plan would search an instance
the record does not: a curve growing out of the Wigner-Seitz cell of its centre
into a ring wider than the neighbouring cells is refused by name, while a curve
shrinking back into it -- the recorded ring holds the neighbours a fresh plan
searches -- and a rectangle crossing its own image-ring bound replay as fresh
solves. Under JIT those discrete decisions cannot be rechecked; re-freeze
outside the trace when moving through contacts or overlap changes, or when a
curve leaves its cell. Where the sampled normal extension switches -- ties
between periodic instances of a curve, between shapes in the blend, and at
corners -- it biases derivatives taken across the switch, by an amount grid
refinement does not reliably remove; check a gradient against the loss over a
window ([autodiff](../common/autodiff.md)). The explicit sampled route continues to
refuse shape and lattice derivatives rather than returning misleading zero
gradients.

## Validation

`tests/rcwa/test_tensor_shape_integrated.py` checks weighted Fourier integrals against
independent Gaussian quadrature, including odd/even grids and Nyquist modes;
analytic anisotropic laminate formulas; uniform tensor identities; continuous
translation phases; painting areas and holes; backend agreement; and geometry
and material derivatives against finite differences.

`tests/rcwa/test_tensor_curve_normals.py` covers the analytic curve candidates of
disjoint layers. Its oracles are independent of the candidates: off-curve
samples on hexagonal, oblique and square cells against a brute-force ring of
fine inscribed polygons (distance within 1e-3 nm, and the closed-form
level-set direction of the instance the ring finds nearest within 1e-12); the
eps convolution matrices against the polygon candidates of the same layer,
which approach those of a circle as the reciprocal of the vertex count (1.7e-5
at 8192 vertices on a hexagonal disc) -- for an ellipse they approach the
nearest-point vector instead, which patched in comes within 1.7e-7 of them on
a weak ellipse beside a strong box, where the level-set direction stays 3.9e-4
away at 512 and 8192 vertices; and the one-primitive path, which a lone
ellipse beside a shape of no contrast matches to 7e-16 in the convolution
matrices where both read one instance. Gradients are checked against the loss
itself: over two windows of an ellipse's angle and two of its position on a
32-point grid, the gradient integrated across the window is within 4% of the
change in total transmission, and the nearest-point vector, patched in, misses
by more than 20% on every one -- the control that shows the check can fail;
the position windows assert that their abscissae resolve the nearest pass of
the ellipse's centre. An odd grid that samples a circle's centre leaves total
transmission independent of the circle's angle, to the bit, and a disk beside
a box is the same layer as a circle beside it, to the bit, on scalar and
tensor layers. The file also checks derivatives against finite differences
across a former polygon vertex on JAX, JIT and Torch, two-sided smoothness
there on a grid that samples the ellipse's medial axis, a sample lying exactly
on an ellipse, and a circle moved sideways across a column of samples on its
axis; independence from the polygon resolution, including a layer whose only
curve is an ellipse; odd grids that sample a curve's centre, values and
gradient; float32 gradients in several length units, with and without a curve
whose contrast vanishes in one pass, and of an off-centre ellipse beside a
centred circle down to a length unit of 1e-9; electric/magnetic duality;
regression values of the test scene at orders 1 and 7; and
frozen-record replay and refusal on floats and under `jax.grad` and Torch
autograd.

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
