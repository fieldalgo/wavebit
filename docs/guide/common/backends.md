# Backends, devices and precision

Execution is separate from a physical problem and its numerics:

```python
result = rcwa.run(problem, numerics=numerics,
                  backend="numpy", device="cpu", precision="double")
```

The same keywords apply to `fdtd.run`, `check`, and `prepare`. Backend names are `"numpy"`, `"jax"`, and `"torch"`; precision is `"single"` or `"double"`. NumPy requires CPU. JAX accepts available `cpu`, `gpu` or `tpu` devices, optionally with an index; Torch accepts CPU or available CUDA devices. These device options describe RCWA execution. The current public FDTD lifecycle requires CPU and double precision on every backend; it refuses other device/precision combinations. Physical feature support remains solver-specific.

Importing Wavebit does not enable JAX x64 globally. Configure double precision before creating JAX arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
from wavebit import rcwa
```

Requesting JAX double precision without x64 enabled raises a structured error. Optional dependencies are imported only when needed. Keep live arrays on the selected backend and device; converting differentiable inputs through NumPy severs their derivative path.

See [automatic differentiation](autodiff.md) for preparation and tracing, and the [compatibility backend manual](../../migration/backends-v1.md) for prior-engine performance notes and historical measured comparisons.
