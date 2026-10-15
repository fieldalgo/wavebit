# Differentiating a physical problem

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


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
| Fixed-cell, zero-angle rectangles under Cartesian Li (`factorization='li'`): sizes and centres | supported | supported under `jax.jit`, `jax.jit(jax.grad(...))` and `jax.vmap`; guarded edge order |
| Other region geometry, such as `Circle.radius`, `Rectangle.size`, `Rectangle.angle` or `Polygon.vertices` | supported | supported under eager `jax.grad`; JIT/vmap over admitted scalar shapes requires explicit `Numerics(topology_check='host')`; default device policy refuses it with `prepared.host_topology` |
| The cell, `Stack.period` | supported | as region geometry, and so is `Stack.lattice` |
| A Jones polarization pair, `PlaneWave(polarization=(s, p))` | supported | supported |

Before the Cartesian Li runtime guard was added, the supported entries were compared with a central difference on one example, with relative agreement between 4e-10 and 2e-7. The PyTorch geometry entries checked were a circle radius (4.6e-9), a rectangle width (1.8e-8) and a rectangle angle (1.3e-7); the raster entries used `('sampled', 'li')`. Under `jax.grad` the geometry derivatives agree with PyTorch's to between 4e-14 and 3e-13 relative (a circle radius, a rectangle's width, centre and angle, and a period length), and with a central difference to 8e-10 for the radius, 4e-8 for the width and 8e-8 for the angle. Polygon vertices agree with PyTorch to 1.4e-13 under `factorization='laurent'`. A period or Jones pair may be a tuple holding a gradient-tracking tensor, such as `(px, 600.)`, or one tensor, such as `torch.stack([px, py])`. The two spellings give the same PyTorch derivative (central-difference agreement 4.1e-10 for a period length and 1.5e-8 for a Jones component); prepare with the spelling you run with, since the two are different layouts.

Fixed-cell Cartesian Li rectangles have an array-valued edge-order guard. Their sizes and centres can move under JIT and vmap, including a closing gap. `result.validity.checks['rectilinear_topology']` is false after crossed edges, a collapsed rectangle or escape from the cell; powers, amplitudes, retained fields and scattering are then NaN. At contact the derivative follows the side represented by the prepared order. See [the runnable example](../../../examples/rcwa/rectilinear_contact.py). Other geometry still needs the concrete checks described below.

`Polygon.vertices` differentiates as an `(N, 2)` tensor or as `N` pairs of scalars; prepare with the spelling you run with, since the two are different layouts. Its derivative is exact under `factorization='laurent'`, and under the default NVF it integrates to the change too since 2026-10-15: the painting route's normal field used to jump across its medial axis, and autodiff missed the jumps' share of the change -- 9% to 16% across a four-pixel window at `orders=(3, 3)` on an L-shaped outline, whatever `shape_nvf_grid_size`, and 18% to 20% for a region painted over another. The field now averages each switch over half a sample step, and the same windows close to within 0.25%. See [polygon regions](../rcwa/boundary-modeling.md#polygon-regions).

### Eager `jax.grad` takes PyTorch's branches

Without `jit` or `vmap`, a `jax.grad` or `jax.jacfwd` tracer carries its value as a PyTorch tensor does, and the lifecycle reads it through `jax.lax.stop_gradient` wherever an eager decision needs a number. So under eager `jax.grad`, `rcwa.prepare`, `rcwa.check`, `Prepared.validate` and `rcwa.run` work at the point being differentiated, and coherent sources, adaptive `detuning` and rasters with more than 256 distinct phases differentiate as they do on PyTorch. A failing check raises the error PyTorch raises: a negative thickness is a `ValidationError` with code `layer.thickness`, a moved raster partition `prepared.raster_partition`, nonfinite observables a `NumericalError` with code `rcwa.nonfinite`. Under `jax.jit` or `jax.vmap` there is no value to read, and the same cases stay refused: `rcwa.coherent_transform`, `prepared.detuning_transform`, `rcwa.raster_partition_transform`, `prepare.concrete`, `check.concrete` and `validate.concrete`.

On one example each, the two-plane-wave derivatives with respect to a region permittivity, a layer thickness and a radius, the detuned derivative at a Rayleigh point, and the derivatives of two 400-phase rasters agree with a central difference to 2e-7 or better on both backends ([`tests/api/test_eager_grad_parity.py`](../../../tests/api/test_eager_grad_parity.py)). JAX and PyTorch agree to round-off except for the region permittivity: there PyTorch's derivative sits 1.1e-8 (one plane wave) to 5.6e-8 (two) from a Richardson extrapolation, where JAX's sits within 2e-10.

### A thickness and a radius derivative

This example uses the default device policy. One preparation serves every run of the objective. On JAX the layer thickness differentiates under `jax.jit` as well; the radius differentiates under eager `jax.grad` and is refused under `jax.jit`. On PyTorch the radius differentiates too. The small Fourier order and 64-point NVF grid make a quick demonstration, not a converged design. The wavelength sits off the cell's Rayleigh anomalies: at 0.9 µm, where the first substrate orders graze (0.6 µm times the substrate index 1.5), a wavelength derivative is NaN on both backends:

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
        excitation=rcwa.Excitation(wavelength=.85, sources=(rcwa.PlaneWave(),)))

numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=64)

on_jax = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='jax')
T00 = lambda t: on_jax.run(problem(.12, t)).transmission.order(0, 0).efficiency[0]
print(jax.jit(jax.grad(T00))(.18))           # -21.87787...
T00_r = lambda r: on_jax.run(problem(r, .18)).transmission.order(0, 0).efficiency[0]
print(jax.grad(T00_r)(.12))                  # -67.08197...
try:
    jax.jit(jax.grad(T00_r))(.12)
except UnsupportedFeatureError as error:
    print(error.code)                        # prepared.host_topology

on_torch = rcwa.prepare(problem(.12, .18), numerics=numerics, backend='torch')
radius = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
on_torch.run(problem(radius, .18)).transmission.order(0, 0).efficiency[0].backward()
print(float(radius.grad))                    # -67.08197...
```

Both radius derivatives agree with each other and with a central difference on NumPy (step `1e-6`) to 2e-8 or better.

### Why JAX-traced geometry is refused under jit

The derivative is the same computation on both backends. Where both admit an input they agree to round-off: one layer-thickness derivative measured 7.695586811926e-4 on PyTorch and 7.695586811925e-4 under `jax.jit`. The backends differ only in a check that runs before that computation.

Outside the guarded Cartesian Li subset, preparation records the geometry's combinatorics: which regions overlap and which edges cross. Vertex coordinates stay live and carry the gradient. Each run checks that the new geometry still fits that record, and the check branches on concrete values. A PyTorch tensor carries its value, so every eager run is checked, and a step that crosses a topology boundary raises `IncompatibleProblemError` with code `prepared.topology`; prepare that geometry again.

The two JAX transforms are not alike here. Under `jax.jit` a tracer has a shape and dtype but no value, so the check cannot run. Skipping it could replay a record the geometry no longer fits and return a wrong answer with no error. Under `jax.grad` without `jit`, the tracer's primal is concrete. Each run reads it through `jax.lax.stop_gradient` and checks the topology on that copy, as it checks a PyTorch tensor's value; the solve keeps the traced value and its tangent. A grad step that crosses a topology boundary raises `prepared.topology`, as on PyTorch. The lifecycle tells the two apart by trying that read. Under `jax.jit` or `jax.vmap`, including a `jax.grad` inside a `jit`, it fails, and the geometry is refused with `prepared.host_topology` when the trace is built. The Cartesian Li subset supplies that check in array operations. General NVF and painted-curve topology remain refused under the default device policy. Explicit `Numerics(topology_check='host')` instead runs the shared concrete checks through a JAX CPU callback and returns `shape_topology=False` with NaN outputs on invalid reuse. Checks run sequentially under vmap; numerical inputs and derivatives retain their JAX graphs. See [checked shape JIT](../rcwa/guarded-shape-jit.md) for scope and costs.

This is the lifecycle's stated contract, not a defect. The [unified API design](../../design/unified-api.md#7-prepared-reuse-and-differentiation) does not promise unrestricted geometry differentiation, and it requires that host-only topology decisions be rejected at trace construction when they cannot be checked. Material, wavelength and layer-thickness derivatives keep their JAX graphs under `jax.jit` as well.

Current examples for [contact](../../../examples/rcwa/nvf_contact.py), [tangent ellipses](../../../examples/rcwa/nvf_curved_contact.py) and [painted curves](../../../examples/rcwa/nvf_curved_painting.py) use the explicit host policy. The [compatibility autodiff manual](../../migration/autodiff-v1.md) preserves the earlier unchecked traced-replay workflow for historical reproduction; its performance measurements do not establish the callback policy's speed.

### A held raster preparation under jit

Raster values stay live, but some Fourier paths record which pixels share a phase when they are prepared. Under `jax.jit`, a run given a pattern whose phase partition differs from the prepared one returns NaN outputs, and `result.validity.checks['raster_partition']` is false. It does not return a stale answer. On PyTorch, the same run raises `IncompatibleProblemError`.

| `fourier_coefficients`, `factorization` | A moved pattern through one held preparation |
| --- | --- |
| `'sampled'`, `'li'` | correct; nothing is recorded |
| `'sampled'`, `'nvf'` | NaN under jit; PyTorch raises |
| `'integrated'`, `'li'` | NaN under jit; PyTorch raises |

When a pattern moves every step under jit, use `('sampled', 'li')` or prepare each concrete pattern outside the transform. A recording path guards the partition under jit only up to 256 distinct phases; beyond that a traced raster is refused with `rcwa.raster_partition_transform`, while eager `jax.grad` and PyTorch check it on its values.

### Backend speed

The measurements in [backend choice](../../validation/rcwa/backend-choice.md), including its recommendation of jitted JAX for fits on shapes, were taken through the earlier solver interface (`freeze_geometry()` and `solve(geometry=...)`, retired with `wavebit.compat` on 2026-10-15; a historical checkout runs them). They are not measurements of this lifecycle or its Cartesian Li runtime guard or host callback policy. Backend speed through `prepared.run` is measured below.

Through `prepared.run`, one derivative step -- the run and the gradient of its summed transmission -- took the following on one CPU (Apple M4, 2026-10-16), for a circle in a 600 nm cell under NVF with `shape_nvf_grid_size=256`:

| `orders` | radius, PyTorch | radius, eager `jax.grad` | thickness, PyTorch | thickness, eager `jax.grad` | thickness, `jax.jit(jax.grad)` |
| --- | --- | --- | --- | --- | --- |
| `(3, 3)` | 0.071 s | 0.27 s | 0.032 s | 0.15 s | 0.024 s |
| `(5, 5)` | 0.39 s | 0.48 s | 0.33 s | 0.37 s | 0.18 s |
| `(7, 7)` | 1.7 s | 1.7 s | 1.4 s | 1.3 s | 0.84 s |
| `(9, 9)` | 6.4 s | 4.6 s | 4.1 s | 3.6 s | 2.4 s |

Each entry is the median of five calls after a first one, each configuration in a fresh process, on a machine shared with other work; the fastest call was up to 1.7 times quicker than the median. Eager `jax.grad` also pays 13 to 22 s on its first call in a process, compiling each operation it meets, where `jax.jit(jax.grad)` compiled in 2 to 5 s. So for a shape derivative PyTorch is the faster choice at low orders, and the two are level within that spread from `orders=(7, 7)`. For an input `jax.jit` admits, the jitted step was the fastest at every order measured, 1.3 times PyTorch at `(3, 3)` and 1.7 to 1.8 times from `(5, 5)`. The script and its record are in [`studies/rcwa/lifecycle_backend_speed/`](https://github.com/fieldalgo/wavebit/tree/main/studies/rcwa/lifecycle_backend_speed).
