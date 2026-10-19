(jobs-and-sweeps)=
# 18. From one calculation to an experiment: Jobs and sweeps

A current `Job` holds a solver name, physical `Problem`, `Numerics` and `Execution`.
Constructing or serializing it does not solve. `wavebit.run(job)` executes one job
in the calling Python environment; it does not submit a cloud job or parallelize a sweep.

| Declaration | What it records |
| --- | --- |
| `Problem` | Units, materials, geometry, illumination and requested outputs |
| `Numerics` | Discretization and algorithm choices |
| `Execution` | Backend, device and precision; NumPy/CPU/double by default |
| `Job` | The three declarations and a solver identifier |

## A silicon layer at three wavelengths

Assign the wavelength-dependent material from [Chapter 7](materials.md) to the
layer background. A uniform layer needs only order zero. Explicitly construct one
job per wavelength with immutable `.replace(...)` operations:

```python
from wavebit import Job, Material, Units, materials, rcwa, run

problem = rcwa.Problem(
    units=Units(length='nm'),
    stack=rcwa.Stack(period=(600., 600.), bottom=Material(2.25), layers=(
        rcwa.Layer(thickness=100., background=materials.load('Si')),)),
    excitation=rcwa.Excitation(wavelength=900., sources=(
        rcwa.PlaneWave(polarization='s'),)))
job = Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics(orders=(0, 0)))
points = [job.replace(problem=problem.replace(
    excitation=problem.excitation.replace(wavelength=wavelength)))
    for wavelength in (800., 900., 1000.)]
for point in points:
    result = run(point)
    print(point.problem.excitation.wavelength, result.R.sum(), result.T.sum())
```

Each job retains its own wavelength and the same silicon table. The solver evaluates
that table at the point's wavelength. A numerical raster instead stores fixed
permittivity values; changing wavelength does not turn those values into a material model.

(jobs-json)=
## Saving and restoring one portable point

```python
from pathlib import Path

path = Path('silicon-point.json')
path.write_text(points[0].to_json(indent=2), encoding='utf-8')
restored = Job.from_json(path.read_text(encoding='utf-8'))
assert restored.key == points[0].key
result = run(restored)
```

`to_json()` stores the current `wavebit.job/v2` schema, including explicit units and
material declarations. Bundled table data is embedded, so a restored job evaluates
the same table. `.key` is a property that identifies canonical inputs. It is not a
result cache or proof of numerical reproducibility; also record package/dependency
versions and the execution environment.

A `wavebit.job/v1` file, written before the [public names were renamed](../../migration/public-names.md),
is refused with code `job.schema` rather than converted: rebuild the problem with the
current names and save it again.

Old `JobSpec` JSON is a different format. It is not silently converted into this
schema: use the [historical job manual](../../migration/jobs-v1.md) to interpret
old units, polarization and sweeps before constructing current declarations.

## Sweeping wavelength and angle together

Use `itertools.product` for all twelve combinations of three wavelengths and four
angles. The loop is explicit and sequential:

```python
from itertools import product

for wavelength, theta in product((800., 900., 1000.), (0., .1, .2, .3)):
    source = problem.excitation.sources[0].replace(theta=theta)
    updated = problem.replace(excitation=problem.excitation.replace(
        wavelength=wavelength, sources=(source,)))
    result = run(job.replace(problem=updated))
```

Angles are radians. Holding angle fixed during a wavelength sweep differs from
holding physical wavevector fixed; see [illumination](../../guide/rcwa/light.md).
The current `Job` has no `expand()` method. Use Python/NumPy to construct the points
or a wavelength vector when the problem supports that spectral batch.

## Runnable round-trip and sweep example

The filename {download}`04_jobspec.py <../../../examples/rcwa/learning/learn/04_jobspec.py>`
is retained for existing links. Its implementation now saves/restores current Jobs,
expands three wavelengths explicitly and checks lossless energy balance at each point.

```{literalinclude} ../../../examples/rcwa/learning/learn/04_jobspec.py
:language: python
:linenos:
```

The original course recorded these totals for the same raster. The stored historical
measurements and source provenance remain unchanged; their format is distinct from
the current Job schema.

| Wavelength (nm) | R | T |
| --- | --- | --- |
| 800.0 | 0.460322555 | 0.539677445 |
| 900.0 | 0.205331628 | 0.794668372 |
| 1000.0 | 0.171513741 | 0.828486259 |

The [capability manifest](../../reference/common/capability-manifest.md) is separate
from a Job and records its own interface version.
