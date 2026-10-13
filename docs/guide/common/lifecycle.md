# Problems, checks, preparation and results

The public `wavebit.rcwa` and `wavebit.fdtd` names are solver modules. Each provides the same three operations:

```python
report = rcwa.check(problem, numerics=numerics,
                    backend="numpy", device="cpu", precision="double")
report.raise_if_invalid()
prepared = rcwa.prepare(problem, numerics=numerics,
                        backend="numpy", device="cpu", precision="double")
result = prepared.run(problem)
# Equivalent one-shot entry point:
result = rcwa.run(problem, numerics=numerics,
                  backend="numpy", device="cpu", precision="double")
```

Replace `rcwa` with `fdtd` for the time-domain lifecycle. A `Problem` contains physical inputs and requested outputs. `Numerics` contains the approximation and discretization. Execution choices select the backend, device and precision. None is hidden in the last solve's mutable state.

## A problem is the input to every run

Declarations are immutable and expose `.replace(...)`. Build a new problem value for a new wavelength, material parameter, shape dimension or excitation. A prepared object does not remember a replacement as its next input; `prepared.run(new_problem)` always requires the complete current problem. Results retain the state needed for their own observations even after later runs.

Preparation freezes structural choices needed by the solver: array shapes and counts, geometry topology and types, discretization, source/monitor layouts, and requested output structure. Reuse is allowed only within the prepared contract. A wavelength vector with the same shape is a live RCWA input; changing the number of wavelengths changes that contract. FDTD reuse also checks the resolved grid and time-stepping structure. Do not infer support for an arbitrary traced value from support for ordinary eager evaluation.

## Validation is explicit

`check` returns a `CheckReport` with structured issues, resolved settings, capabilities and estimates. It does not advance FDTD fields. `raise_if_invalid()` raises a typed validation error. Unsupported physical or numerical combinations are reported as unsupported rather than silently selecting a different model.

Concrete checks can report invalid values before a run. Inside a transformation, some value-dependent conditions cannot raise a Python exception. Public results expose validity information for those checks; inspect it before using a result. A successful trace establishes transform compatibility, not physical convergence.

## Units and execution

All geometry and vacuum wavelengths use `Units(length="m" | "mm" | "um" | "nm")`. Times are seconds, ordinary frequencies Hz, angular rates rad/s, electrical conductivity S/m, angles radians, E V/m and H A/m. Internal normalized FDTD coordinates and magnetic fields are converted at the public boundary.

Backends are `"numpy"`, `"jax"`, and `"torch"`; precision is `"single"` or `"double"`. Select a device explicitly when needed. Importing Wavebit does not load all optional backends or change global precision. For JAX double precision, enable x64 before creating arrays; see [backends](backends.md).

## Runnable examples and compatibility

Start with [RCWA](../../getting-started/rcwa.md) or [FDTD](../../getting-started/fdtd.md). The imported teaching books and archived experiment scripts explicitly use `wavebit.compat.legacy` where they preserve an older numerical workflow. The [migration guide](../../migration/index.md) distinguishes those calls from the current modules.
