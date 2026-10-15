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
| Region geometry, such as `Circle.radius`, `Rectangle.size`, `Rectangle.angle` or `Polygon.vertices` | supported | supported under eager `jax.grad`; **refused** under `jax.jit` or `jax.vmap`: `UnsupportedFeatureError`, code `prepared.host_topology` |
| The cell, `Stack.period` | supported | as region geometry, and so is `Stack.lattice` |
| A Jones polarization pair, `PlaneWave(polarization=(s, p))` | supported | supported |

Each supported entry was compared with a central difference on one example, with relative agreement between 4e-10 and 2e-7. The PyTorch geometry entries checked were a circle radius (4.6e-9), a rectangle width (1.8e-8) and a rectangle angle (1.3e-7); the raster entries used `('sampled', 'li')`. Under `jax.grad` the geometry derivatives agree with PyTorch's to between 4e-14 and 3e-13 relative (a circle radius, a rectangle's width, centre and angle, and a period length), and with a central difference to 8e-10 for the radius, 4e-8 for the width and 8e-8 for the angle. Polygon vertices agree with PyTorch to 1.4e-13 under `factorization='laurent'`. A period or Jones pair may be a tuple holding a gradient-tracking tensor, such as `(px, 600.)`, or one tensor, such as `torch.stack([px, py])`. The two spellings give the same PyTorch derivative (central-difference agreement 4.1e-10 for a period length and 1.5e-8 for a Jones component); prepare with the spelling you run with, since the two are different layouts.

So, through this lifecycle, a derivative from geometry to diffraction efficiency is available on PyTorch and under eager `jax.grad`, not under `jax.jit` or `jax.vmap`. A `jax.grad` inside `jax.jit` has no concrete value either, so `jax.jit(jax.grad(loss))` is refused as `jax.jit(loss)` is.

`Polygon.vertices` differentiates as an `(N, 2)` tensor or as `N` pairs of scalars; prepare with the spelling you run with, since the two are different layouts. Its derivative is exact under `factorization='laurent'`. Under the default NVF it is not: the painting route's normal field jumps across its medial axis, and autodiff misses the jumps' share of the change -- 10% across a four-pixel window at `orders=(3, 3)` on an L-shaped outline, 3% at `(9, 9)`, whatever `shape_nvf_grid_size`, and the same holds for any region painted over another. See [polygon regions](../rcwa/boundary-modeling.md#polygon-regions).

### A thickness and a radius derivative

One preparation serves every run of the objective. On JAX the layer thickness differentiates under `jax.jit` as well; the radius differentiates under eager `jax.grad` and is refused under `jax.jit`. On PyTorch the radius differentiates too. The small Fourier order and 64-point NVF grid make a quick demonstration, not a converged design:

```python
import jax
jax.config.update('jax_enable_x64', True)
import torch
from wavebit import Material, Units, UnsupportedFeatureError, rcwa

def problem(radius, thickness):
    return rcwa.Problem(
        units=Units(length='um'),
        stack=rcwa.Stack(
            period=(.6, .6), bottom=Material(epsilon=2.25),
            layers=(rcwa.Layer(thickness=thickness, regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=radius), material=Material(epsilon=12.25)),)),)),
        excitation=rcwa.Excitation(wavelength=.9, sources=(rcwa.PlaneWave(),)))

numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=64)

on_jax = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='jax')
T00 = lambda t: on_jax.run(problem(.12, t)).transmission.order(0, 0).efficiency[0]
print(jax.jit(jax.grad(T00))(.18))           # -2.461434...
T00_r = lambda r: on_jax.run(problem(r, .18)).transmission.order(0, 0).efficiency[0]
print(jax.grad(T00_r)(.12))                  # -8.662999...
try:
    jax.jit(jax.grad(T00_r))(.12)
except UnsupportedFeatureError as error:
    print(error.code)                        # prepared.host_topology

on_torch = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='torch')
radius = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
on_torch.run(problem(radius, .18)).transmission.order(0, 0).efficiency[0].backward()
print(float(radius.grad))                    # -8.662999...
```

Both radius derivatives agree with each other and with a central difference on NumPy (step `1e-6`) to 2e-8 or better.

### Why JAX-traced geometry is refused under jit

The derivative is the same computation on both backends. Where both admit an input they agree to round-off: one layer-thickness derivative measured 7.695586811926e-4 on PyTorch and 7.695586811925e-4 under `jax.jit`. The backends differ only in a check that runs before that computation.

Preparation records the geometry's combinatorics: which regions overlap and which edges cross. Vertex coordinates stay live and carry the gradient. Each run checks that the new geometry still fits that record, and the check branches on concrete values. A PyTorch tensor carries its value, so every eager run is checked, and a step that crosses a topology boundary raises `IncompatibleProblemError` with code `prepared.topology`; prepare that geometry again.

The two JAX transforms are not alike here. Under `jax.jit` a tracer has a shape and dtype but no value, so the check cannot run. Skipping it could replay a record the geometry no longer fits and return a wrong answer with no error. Under `jax.grad` without `jit`, the tracer's primal is concrete. Each run reads it through `jax.lax.stop_gradient` and checks the topology on that copy, as it checks a PyTorch tensor's value; the solve keeps the traced value and its tangent. A grad step that crosses a topology boundary raises `prepared.topology`, as on PyTorch. The lifecycle tells the two apart by trying that read. Under `jax.jit` or `jax.vmap`, including a `jax.grad` inside a `jit`, it fails, and the geometry is refused with `prepared.host_topology` when the trace is built. Admitting geometry under `jax.jit` would need a traced form of the check: computed in array operations and reported as NaN with a failed validity check, as the raster partition guard already is.

This is the lifecycle's stated contract, not a defect. The [unified API design](../../design/unified-api.md#7-prepared-reuse-and-differentiation) does not promise unrestricted geometry differentiation, and it requires that host-only topology decisions be rejected at trace construction when they cannot be checked. Material, wavelength and layer-thickness derivatives keep their JAX graphs under `jax.jit` as well.

For a shape derivative under `jax.jit`, use the compatibility interface: `wavebit.compat.legacy`, with `solver.freeze_geometry()` taken outside the transform and `solver.solve(geometry=...)` inside it. The [compatibility autodiff manual](../../migration/autodiff-v1.md) describes that workflow and when a frozen record goes stale.

### A held raster preparation under jit

Raster values stay live, but some Fourier paths record which pixels share a phase when they are prepared. Under `jax.jit`, a run given a pattern whose phase partition differs from the prepared one returns NaN outputs, and `result.validity.checks['raster_partition']` is false. It does not return a stale answer. On PyTorch, the same run raises `IncompatibleProblemError`.

| `fourier_coefficients`, `factorization` | A moved pattern through one held preparation |
| --- | --- |
| `'sampled'`, `'li'` | correct; nothing is recorded |
| `'sampled'`, `'nvf'` | NaN under jit; PyTorch raises |
| `'integrated'`, `'li'` | NaN under jit; PyTorch raises |

When a pattern moves every step under jit, use `('sampled', 'li')` or prepare each concrete pattern outside the transform.

### Backend speed

The measurements in [backend choice](../../validation/rcwa/backend-choice.md), including its recommendation of jitted JAX for fits on shapes, were taken through the solver interface now kept as `wavebit.compat.legacy` (`freeze_geometry()` and `solve(geometry=...)`). They are not measurements of this lifecycle, where a JAX shape derivative runs under eager `jax.grad` only and is refused under `jax.jit`. Backend speed through `prepared.run` has not been measured.
