# Differentiating a physical problem

Use live JAX or Torch values in immutable declarations and pass the full updated problem into each prepared run. The declaration's `.replace(...)` operation creates a new input without mutating a previous result or the prepared contract.

For example, an objective can replace a material or layer thickness and return a scalar from `prepared.run(updated_problem)`. Prepare outside the differentiated function with compatible concrete values. Fourier order, array shape, region topology, grid structure and monitor layout are structural choices; changing them requires a new preparation.

Backend transform support and physical differentiability are separate questions. A geometry boundary crossing a fixed observation point may change which material is sampled. FDTD also fixes a discrete grid and source/monitor stencil, so derivatives refer to that discretized problem. Inspect validity information when checks depend on traced values, and verify gradients against an independent perturbation before using them in optimization.

JAX double precision requires explicit x64 configuration; see [backends](backends.md). For the established kernel-level derivations, transform checks, finite-difference comparisons and legacy examples, see the [compatibility autodiff manual](../../migration/autodiff-v1.md) and [FDTD differentiation study guide](../fdtd/fd-differentiation.md). Their evidence retains its stated geometry, backend and discretization scope.

## RCWA: which inputs differentiate on which backend

Through `rcwa.prepare(...)` and `prepared.run(problem)`, the backends do not admit the same live inputs:

| Live input | PyTorch (eager autograd) | JAX (`jax.grad`, `jax.jit`) |
| --- | --- | --- |
| Material coefficients, including the half-space media | supported | supported |
| Wavelength | supported | supported |
| Layer thickness | supported | supported |
| Raster permittivity values | supported | supported; see [held raster preparations](#a-held-raster-preparation-under-jit) |
| Region geometry, such as `Circle.radius`, `Rectangle.size`, `Rectangle.angle` or `Polygon.vertices` | supported | **refused** for every geometry field: `UnsupportedFeatureError`, code `prepared.host_topology` |
| The cell, `Stack.period` | supported | **refused**: `prepared.host_topology`, as for `Stack.lattice` |
| A Jones polarization pair, `PlaneWave(polarization=(s, p))` | supported | supported |

Each supported entry was compared with a central difference on one example, with relative agreement between 4e-10 and 2e-7. The PyTorch geometry entries checked were a circle radius (4.6e-9), a rectangle width (1.8e-8) and a rectangle angle (1.3e-7); the raster entries used `('sampled', 'li')`. A period or Jones pair may be a tuple holding a gradient-tracking tensor, such as `(px, 600.)`, or one tensor, such as `torch.stack([px, py])`. The two spellings give the same PyTorch derivative (central-difference agreement 4.1e-10 for a period length and 1.5e-8 for a Jones component); prepare with the spelling you run with, since the two are different layouts.

So, through this lifecycle, the derivative from geometry to diffraction efficiency is a PyTorch derivative. The JAX refusal applies to `jax.grad` alone as well as under `jax.jit`.

`Polygon.vertices` differentiates as an `(N, 2)` tensor or as `N` pairs of scalars; prepare with the spelling you run with, since the two are different layouts. Its derivative is exact under `factorization='laurent'`. Under the default NVF it is not: the painting route's normal field jumps across its medial axis, and autodiff misses the jumps' share of the change -- 10% across a four-pixel window at `orders=(3, 3)` on an L-shaped outline, 3% at `(9, 9)`, whatever `shape_nvf_grid_size`, and the same holds for any region painted over another. See [polygon regions](../rcwa/boundary-modeling.md#polygon-regions).

### Why JAX-traced geometry is refused

The derivative is the same computation on both backends. Where both admit an input they agree to round-off: one layer-thickness derivative measured 7.695586811926e-4 on PyTorch and 7.695586811925e-4 under `jax.jit`. The backends differ only in a check that runs before that computation.

Preparation records the geometry's combinatorics: which regions overlap and which edges cross. Vertex coordinates stay live and carry the gradient. Each run checks that the new geometry still fits that record, and the check branches on concrete values. A PyTorch tensor carries its value, so every eager run is checked, and a step that crosses a topology boundary raises `IncompatibleProblemError` with code `prepared.topology`; prepare that geometry again.

The two JAX transforms are not alike here. Under `jax.jit` a tracer has a shape and dtype but no value, so the check cannot run. Skipping it could replay a record the geometry no longer fits and return a wrong answer with no error. Under `jax.grad` without `jit`, the tracer's primal is concrete and can be read one element at a time, which is how the compatibility interface takes its geometry decisions. The lifecycle does not currently tell the two apart, and it refuses both when the trace is built. Admitting geometry under `jax.jit` would need a traced form of the check: computed in array operations and reported as NaN with a failed validity check, as the raster partition guard already is.

This is the lifecycle's stated contract, not a defect. The [unified API design](../../design/unified-api.md#7-prepared-reuse-and-differentiation) does not promise unrestricted geometry differentiation, and it requires that unguarded host-only topology decisions be rejected at trace construction. Material, wavelength and layer-thickness derivatives keep their JAX graphs.

For a JAX derivative with respect to a shape, use the compatibility interface: `wavebit.compat.legacy`, with `solver.freeze_geometry()` taken outside the transform and `solver.solve(geometry=...)` inside it. The [compatibility autodiff manual](../../migration/autodiff-v1.md) describes that workflow and when a frozen record goes stale.

### A held raster preparation under jit

Raster values stay live, but some Fourier paths record which pixels share a phase when they are prepared. Under `jax.jit`, a run given a pattern whose phase partition differs from the prepared one returns NaN outputs, and `result.validity.checks['raster_partition']` is false. It does not return a stale answer. On PyTorch, the same run raises `IncompatibleProblemError`.

| `fourier_coefficients`, `factorization` | A moved pattern through one held preparation |
| --- | --- |
| `'sampled'`, `'li'` | correct; nothing is recorded |
| `'sampled'`, `'nvf'` | NaN under jit; PyTorch raises |
| `'integrated'`, `'li'` | NaN under jit; PyTorch raises |

When a pattern moves every step under jit, use `('sampled', 'li')` or prepare each concrete pattern outside the transform.

### Backend speed

The measurements in [backend choice](../../validation/rcwa/backend-choice.md), including its recommendation of jitted JAX for fits on shapes, were taken through the solver interface now kept as `wavebit.compat.legacy` (`freeze_geometry()` and `solve(geometry=...)`). They are not measurements of this lifecycle, where JAX shape derivatives are refused. Backend speed through `prepared.run` has not been measured.
