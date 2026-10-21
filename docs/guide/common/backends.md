# Backends, devices and precision

Execution is separate from a physical problem and its numerics:

```python
result = rcwa.run(problem, numerics=numerics,
                  backend="numpy", device="cpu", precision="double")
```

The same keywords apply to `fdtd.run`, `check`, and `prepare`. Backend names are `"numpy"`, `"jax"`, and `"torch"`; precision is `"single"` or `"double"`. NumPy requires CPU. JAX accepts available `cpu`, `gpu` or `tpu` devices, optionally with an index; Torch accepts CPU or available CUDA devices. These device options describe RCWA execution. General-scene `fdtd` requires double precision: NumPy and JAX use CPU, while Torch also admits the [bounded CUDA scene contract](../../validation/fdtd/general-cuda.md). The explicit `fdtd.native` and shared `fdtd.periodic` paths also require double precision and accept available backend devices; the periodic reference algorithm requires NumPy/CPU. Device admission is not a GPU performance or accuracy measurement. Physical feature support remains solver-specific.

Importing Wavebit does not enable JAX x64 globally. Configure double precision before creating JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
from wavebit import rcwa
```

Requesting JAX double precision without x64 enabled raises a structured error. Optional dependencies are imported only when needed. Keep live arrays on the selected backend and device; converting differentiable inputs through NumPy severs their derivative path.

See [automatic differentiation](autodiff.md) for preparation and tracing, and the [compatibility backend manual](../../migration/backends-v1.md) for prior-engine performance notes and historical measured comparisons.

## Shared materials under JAX transformations

For a single plane wave, `Prepared.run` can merge a scalar region layer into its adjacent
half-space when both use the **same live JAX coefficient**. Build the declarations inside the
transformed function: `Material(eps)` for both the layer background and the half-space. If
permeability is live, it must be shared too; every region of the layer must match. This removes
an artificial interface and preserves the shared parameter's derivative through the remaining
half-space and the phase that moves the reference plane back to the original face.

This proof is made afresh on each run, before array conversions. Equal values at `prepare`
time do not freeze a merge. Separate inputs remain independent, even if they happen to have
the same value; `eps + 0` is a separate expression, and passing an entire `Problem` as a JIT
argument creates separate leaves. Those cases, raster materials, dispersive models, PyTorch
material gradients retain their existing merge limits.
A positive `half_space_merge_tolerance` still refuses live materials.

The benefit matters at complex incidence, where explicitly solving a thick layer of its own
half-space's medium can amplify round-off to an order-one error. It does not claim stability
for a genuinely different or patterned layer. Ordinary eager runs with concrete materials keep
their existing path. See [the numerical explanation](../../theory/rcwa/connecting-algorithms.md#10-etms-labels-at-a-complex-in-plane-wavevector).

## Compiled compatibility field solves

The internal compatibility method `solver.solve_field(..., geometry=g)` checks concrete
materials before compiling the solver. It carries that call's merge plan into the solve and
field reconstruction, including a concrete `half_space_merge_tolerance`. The plan is recomputed
after each keyword update; it is not frozen with `g`. Angle and thickness sweeps retain their
derivatives. Use a count for `sample_z` in compiled calls, as required by that interface.

An outer transformation over independent material inputs still keeps those layers; a positive
tolerance still refuses live materials. This change does not infer shared material dependencies
from the separate leaves of a solver passed as a JIT argument. Public field observations use
`Prepared.run` and result-owned reconstruction; they do not require this compatibility method.
