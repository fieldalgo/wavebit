# Checking shape topology inside JAX transforms

A prepared shape calculation records discrete decisions: which boundaries intersect,
which regions paint over others, and which periodic images the normal field needs.
Moving a continuous parameter may invalidate those decisions even when a stale
calculation still conserves energy.

The default `Numerics(topology_check='device')` admits the fixed-cell Cartesian Li
rectangle guard. Other geometry under JIT or vmap is refused before tracing because
its concrete topology checks cannot execute as ordinary device array operations.
An eager `jax.grad` can still check its concrete primal.

Use `Numerics(topology_check='host')` to execute the existing concrete checks on the
CPU during a JAX calculation over **scalar-material region layers with one plane
wave**. This supports circles, ellipses, rectangles and admitted polygon paintings,
including their cell and orientation parameters where the selected engine supports
traced evaluation. It does not add tensor, raster or coherent-source transform
support, and it does not change which projector/shape combinations the engine admits.

```python
import jax
jax.config.update('jax_enable_x64', True)
from wavebit import Material, Units, rcwa

def design(radius):
    return rcwa.Problem(units=Units(length='nm'),
        stack=rcwa.Stack(period=(700., 700.), bottom=Material(2.25), layers=(
            rcwa.Layer(thickness=180., regions=(rcwa.Region(
                geometry=rcwa.Circle(radius=radius), material=Material(4.)),)),)),
        excitation=rcwa.Excitation(wavelength=900., sources=(rcwa.PlaneWave(polarization='p'),)))

numerics = rcwa.Numerics(orders=(2, 2), shape_nvf_grid_size=64, topology_check='host')
prepared = rcwa.prepare(design(120.), numerics=numerics, backend='jax')
objective = lambda radius: prepared.run(design(radius)).T.sum()
value, derivative = jax.jit(jax.value_and_grad(objective))(121.)
```

A pure JAX callback sends geometry values to the CPU and returns a validity boolean.
It does not return a replacement geometry or an optical answer. The numerical solve
and all its differentiable inputs keep their JAX graphs. Under vmap the checks run
sequentially for each geometry; they can limit throughput, especially for complex
paintings. This is an explicit host-transfer policy, not a claim of fully device-resident
execution or a backend speed measurement. CPU JAX execution is covered by the regression
suite; accelerator performance has not been measured. One CPU measurement of a jitted step, a circle under `topology_check='host'` and a Cartesian Li
rectangle under the default, is in the differentiation guide's
[backend speed](../common/autodiff.md#backend-speed) section.

`result.validity.checks['shape_topology']` is false when the host check rejects a live
geometry. Diffraction powers and amplitudes, retained fields, boundary fields and
scattering matrices are then NaN. Their connected forward- and reverse-mode derivatives
also propagate NaN, so a rejected solve cannot masquerade as a zero-gradient optimum.
An output independent of a differentiated input can still have a structural zero
derivative; do not use Jacobian finiteness as the admission test.

For RCWA, `result.validity.checks['numerical_finite']` is false if the underlying
solve is nonfinite **or any admission check fails**. Inspect the other checks, such
as `shape_topology`, to identify an admission failure. A true flag does not certify
convergence or that every optional observable is defined (for example, efficiency
can be undefined for a source with no normal incident flux). Optimizers must reject
failed checks before accepting a step or estimating uncertainty.

A valid point follows an invalid point correctly:
no last-call geometry is cached. Eager invalid reuse still raises an
`IncompatibleProblemError`, and `prepared.validate(concrete_problem)` gives an eager
report. Prepare again from concrete inputs when topology changes; no smooth derivative
across that change is promised.

The policy guards reuse, not optical convergence. In particular, painting's finite
normal-field grid and polygon approximations retain their existing derivative limitations.
Compare the derivative with fresh finite differences and refine numerical resolution
independently.

For overlapping polygonised curves, the scalar painted normal field uses a boundary line integral so a disappearing clipped interval loses its contribution continuously. Reprepare when its topology changes. The [clipping-plan continuity validation](../../validation/rcwa/normal-field-known-limits.md) describes the material-label repair, the integral, and its scope; physical contacts, polygon approximation and finite normal grids still require independent convergence and derivative checks.

## Automatic re-preparation during optimization

Use `rcwa.compile(build, observe, numerics=...)` when an optimizer should continue
through changes in the geometry plan. `build(parameters)` returns a `Problem`, and
`observe(result)` selects a scalar or array. The controller checks each concrete
candidate on the host. It reuses its preparation while the geometry fits; otherwise
it prepares the **same candidate**, compiles the new calculation and returns that
candidate's value and derivative. It does not move back to the previous parameters
or ask the optimizer to catch a topology exception.

```python
evaluate = rcwa.compile(design, lambda result: result.T.sum(), numerics=numerics)
value = evaluate(121.)
value, gradient = evaluate.value_and_grad(121.)
value, jacobian = evaluate.value_and_jacobian(121.)
```

Call this controller directly from a Python optimization loop or SciPy. It manages
JAX JIT internally; do not wrap the controller in `jax.jit`, `jax.grad` or `jax.vmap`.
Use `value_and_grad` for a real scalar objective, and `value_and_jacobian` or
`jacobian` for array observations. Parameters must be a concrete finite real
floating-point array or scalar. `build` and `observe` must be pure and traceable by
JAX: keep the model structure and observation shape fixed and pass every changing
input through the parameters, rather than mutable captured state.

This entry point explicitly uses `topology_check='host'` and its single-plane-wave,
scalar-material scope. It preserves the other numerical settings. It keeps one
preparation and compiled functions for its value/derivative methods; a topology
change can incur compilation again. `evaluate.preparation_count` counts successfully
installed preparations. Failed candidates do not replace the last working plan.
Revisiting an older topology can require another preparation: the controller does
not accumulate a history of geometry plans or cached answers.

Only geometry-reuse failures trigger re-preparation. Invalid dimensions, nonfinite
inputs, unsupported requests, nonfinite selected observations and nonfinite
derivatives remain explicit errors. If a compiled guard rejects an old plan despite
the host preflight, it retries once with a fresh preparation at those parameters;
failure of that fresh plan is reported instead of retrying indefinitely.

The [automatic re-preparation example](../../../examples/rcwa/automatic_repreparation.py)
fits a rectangle's position across the transition from overlap to separation, using
SciPy residual and Jacobian calls without a manual retry handler. CPU JAX is covered
by regression tests; GPU execution has not been measured. Re-preparation makes
supported new geometry calculable, but does not guarantee a smooth derivative at
the exact topology boundary, convergence of an optimizer, or normal-field grid
convergence. Low-level `Prepared.run` retains the explicit fixed-plan behavior above.

Runnable examples cover [rectangular contact](../../../examples/rcwa/nvf_contact.py),
[tangent ellipses](../../../examples/rcwa/nvf_curved_contact.py), and
[overlapping curves](../../../examples/rcwa/nvf_curved_painting.py). They preserve their
original geometry, materials, Fourier orders, projector and independent checks.
