(jobs-and-sweeps)=
# 18. From one calculation to an experiment: JobSpec and sweeps

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

Create a solver with `wavebit.rcwa(...)` and call `solver.solve()` to calculate directly in Python. **`JobSpec` is an optional object for saving settings or organizing calculations across multiple conditions.** It groups structure, materials, light and numerical settings; constructing it does not start a solve.

## What are JobSpec, expand() and run()?

All three are provided by Wavebit. Below, `spec` is the variable holding a user-created `JobSpec`.

| Name | Kind | Role |
| --- | --- | --- |
| `wavebit.JobSpec(...)` | Class | Groups a structure and its calculation conditions into one object |
| `spec.expand()` | `JobSpec` method | Splits a sweep into settings for individual wavelengths or angles; does not solve |
| `wavebit.run(point)` | Function | Calculates one set of settings and returns a `Result` |

**These work in a local Python environment.** They can also help build a program that sends settings to a server, but this API itself neither submits cloud jobs nor automatically parallelizes calculations. `run()` executes in the calling environment using the selected backend.

## What does a calculation specification contain?

`JobSpec` groups the following objects. A sweep means **repeating calculations while changing inputs such as wavelength or incidence angle**.

| Object | Contents | Location in `JobSpec` |
| --- | --- | --- |
| `Layer` | One layer’s thickness and material distribution | Element of `Stack.layers` |
| `Stack` | Layer list, upper and lower media, lattice | `stack` |
| `Incidence` | Wavelength, direction, polarization and incident side | `incidence` |
| `Numerics` | Backend, Fourier orders, factorization and other numerical options | `numerics` |
| `Sweep` | Inputs to change and their values | `sweep` |

```{figure} ../../figures/learn/learn/18-jobspec.svg
:alt: Structure, incidence, numerical settings and sweep form a JobSpec; several wavelengths and angles expand into all combinations
:figclass: learning-figure

Figure 18-1. All wavelength–angle combinations become calculation settings while structure and numerical options remain fixed.
```

## Example: a silicon layer at three wavelengths

Assign the wavelength-dependent material object from [Chapter 7](materials.md) to a layer’s `background`. `shapes=[]` makes the entire layer uniform silicon, so Fourier order 0 is sufficient. Lengths are in nm.

```python
from wavebit.compat import legacy as wavebit

silicon = wavebit.materials.load('Si')
spec = wavebit.JobSpec(
    stack=wavebit.Stack(
        layers=[wavebit.Layer(
            thickness=100.,
            structure=dict(background=silicon, shapes=[]),
        )],
        top=1., bot=2.25, period=(600., 600.),
    ),
    incidence=wavebit.Incidence(
        wavelength=900., theta=0., polarization=(1., 0.),
    ),
    numerics=wavebit.Numerics(truncation_order=(0, 0)),
    sweep=wavebit.Sweep.from_dict({'wavelength': [800., 900., 1000.]}),
)
```

`incidence.wavelength=900.` is the base wavelength; this example replaces it with each of the three `sweep` values. So far, only settings have been constructed.

### expand(): split the specification into three points

```python
points = spec.expand()
print([point.incidence.wavelength for point in points])
# [800.0, 900.0, 1000.0]
```

Each element of `points` is **a `JobSpec` with its own wavelength**, not merely a wavelength number. Structure, materials and numerical options remain fixed; only wavelength changes to 800, 900 or 1,000 nm. Calling `expand()` on a specification without a sweep returns a list containing that original specification.

### run(): calculate each point

```python
for point in points:
    result = wavebit.run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

This loop executes three calculations in sequence. The 800 nm calculation uses silicon-table values at 800 nm; the 900 nm calculation uses values at 900 nm. This is what **automatic evaluation of materials at the calculation wavelength** means. Read outputs just as for `solver.solve()`; see [Chapter 12](results.md).

`run()` accepts one condition. Passing a specification containing a sweep directly to `wavebit.run(spec)` raises an error, so call `expand()` before running its points.

(jobs-json)=
## Saving to JSON and restoring

**JSON represents settings as text.** `to_json()` creates a JSON string; `from_json()` reconstructs a `JobSpec` from it. The `Path` code below handles writing and reading the file.

```python
from pathlib import Path

path = Path('silicon-sweep.json')
path.write_text(spec.to_json(indent=2), encoding='utf-8')

restored = wavebit.JobSpec.from_json(path.read_text(encoding='utf-8'))
for point in restored.expand():
    result = wavebit.run(point)
    print(point.incidence.wavelength, result.R.sum(), result.T.sum())
```

Here **JSON stores the loaded wavelength-dependent silicon table, not just the name silicon**. The restored specification evaluates that table at each wavelength. Index-derived materials preserve the index table and optional permeability table; permittivity-derived materials preserve the permittivity table and optional permeability table. Anisotropic tensor components are preserved too.

If instead the constant 900 nm values were assigned, JSON stores those values. They remain fixed after restoring and changing wavelength, as explained for the two material objects in [Chapter 7](materials.md). Wrap them in a `Material` first, as in `value = silicon.at(900.)` and then `wavebit.Material(epsilon=value.epsilon, mu=value.mu)`: `.at()` on a bundled table returns a `MaterialResponse`, which `to_json()` refuses.

`spec.key()` identifies input settings for comparison. If `spec.key() == restored.key()` after saving and restoring, the normalized inputs match. To reproduce results, also record Wavebit and dependency versions, backend and precision. JSON includes a format `version`, currently 10 in this workflow. See the [specification guide](../../migration/jobs-v1.md#what-a-document-must-say-and-what-is-refused) for conversion rules for earlier formats.

## Sweeping multiple inputs together

`Sweep` supports the following inputs.

| Input | Quantity varied |
| --- | --- |
| `wavelength` | Vacuum wavelength |
| `theta`, `phi` | Incident phase angle and azimuth |
| `kx`, `ky` | Direct x and y wavevector components |
| `psi`, `chi` | Polarization angles |

Three wavelengths and four angles create **all twelve combinations**, not three or four index-paired settings. Equally spaced values can be written as `{'linspace': [start, stop, count]}`, as in the example file below.

`theta`·`phi` and `kx`·`ky` are alternative direction inputs from {ref}`Chapter 8 <incident-direction-inputs>` and cannot be mixed. For direct-wavevector sweeps, construct the base incidence as `Incidence(kx=..., ky=...)` too. Holding angles fixed during a wavelength sweep differs physically from holding actual wavevectors fixed.

Update geometry or dispersive numerical rasters in a separate Python loop. Numerical rasters do not retain material tables, so changing wavelength alone does not update their permittivity arrays. When dispersive objects are assigned to shapes, backgrounds or half-spaces, the solver evaluates them at each wavelength.

## Runnable file: save and restore a raster, then sweep wavelength

The file below saves and restores a raster containing ε=1 and ε=12.25 through JSON, then solves at three wavelengths. Material values are constant here, so the comparison varies the structure’s response with wavelength.

Runnable file: {download}`04_jobspec.py <../../../examples/rcwa/learning/learn/04_jobspec.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/04_jobspec.py
:language: python
:linenos:
```

| Wavelength (nm) | R | T |
| --- | --- | --- |
| 800.0 | 0.460322555 | 0.539677445 |
| 900.0 | 0.205331628 | 0.794668372 |
| 1000.0 | 0.171513741 | 0.828486259 |

See the [compatibility job manual](../../migration/jobs-v1.md) for all options and JSON format, and the [API reference](../../migration/legacy-api.md#a-run-as-data) for classes and methods. The [capability-manifest guide](../../reference/common/capability-manifest.md) separately describes the capability list available to external programs.
