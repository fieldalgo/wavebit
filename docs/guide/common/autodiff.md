# Differentiating a physical problem

Use live JAX or Torch values in immutable declarations and pass the full updated problem into each prepared run. The declaration's `.replace(...)` operation creates a new input without mutating a previous result or the prepared contract.

For example, an objective can replace a material or layer thickness and return a scalar from `prepared.run(updated_problem)`. Prepare outside the differentiated function with compatible concrete values. Fourier order, array shape, region topology, grid structure and monitor layout are structural choices; changing them requires a new preparation.

Backend transform support and physical differentiability are separate questions. A geometry boundary crossing a fixed observation point may change which material is sampled. FDTD also fixes a discrete grid and source/monitor stencil, so derivatives refer to that discretized problem. Inspect validity information when checks depend on traced values, and verify gradients against an independent perturbation before using them in optimization.

JAX double precision requires explicit x64 configuration; see [backends](backends.md). For the established kernel-level derivations, transform checks, finite-difference comparisons and legacy examples, see the [compatibility autodiff manual](../../migration/autodiff-v1.md) and [FDTD differentiation study guide](../fdtd/fd-differentiation.md). Their evidence retains its stated geometry, backend and discretization scope.
