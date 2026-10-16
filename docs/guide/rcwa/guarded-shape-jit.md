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
scattering matrices are then NaN. A valid point follows an invalid point correctly:
no last-call geometry is cached. Eager invalid reuse still raises an
`IncompatibleProblemError`, and `prepared.validate(concrete_problem)` gives an eager
report. Prepare again from concrete inputs when topology changes; no smooth derivative
across that change is promised.

The policy guards reuse, not optical convergence. In particular, painting's finite
normal-field grid and polygon approximations retain their existing derivative limitations.
Compare the derivative with fresh finite differences and refine numerical resolution
independently.

Runnable examples cover [rectangular contact](../../../examples/rcwa/nvf_contact.py),
[tangent ellipses](../../../examples/rcwa/nvf_curved_contact.py), and
[overlapping curves](../../../examples/rcwa/nvf_curved_painting.py). They preserve their
original geometry, materials, Fourier orders, projector and independent checks.
