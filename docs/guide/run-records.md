# Save and inspect research runs

`wavebit.records` saves a portable `Job`, measured results and the source used on
that machine. A colleague can inspect the numbers with a base Wavebit install,
without running the solver or installing the backend that produced them.

## Record a calculation

```python
from wavebit import Job, Material, Units, rcwa
from wavebit.records import run_recorded, load_run

problem = rcwa.Problem(
    units=Units(length="um"),
    stack=rcwa.Stack(period=(.8, .8), layers=(
        rcwa.Layer(thickness=.2, background=Material(2.25)),
    )),
    excitation=rcwa.Excitation(wavelength=(.6, .8), sources=(
        rcwa.PlaneWave(polarization="s"),
    )),
    outputs=rcwa.Outputs(fields=True),
)
job = Job(solver="rcwa", problem=problem, numerics=rcwa.Numerics())
record = run_recorded(
    job, "artifacts/runs/film-001",
    sample=lambda result: {
        "inside_film": result.field(x=(0., .1), y=(0.,), z=(.1,)),
    },
)

loaded = load_run(record.path)
transmission = loaded.outputs["transmitted"]["efficiency"]
field = loaded.samples["inside_film"]
print(transmission, field["units"], loaded.manifest["source"]["identity"])
```

The destination must be new. Existing files, directories and symlinks are refused.
Use a distinct directory for each run. The executable
[`record_run.py`](https://github.com/fieldalgo/wavebit/blob/main/examples/common/record_run.py) example checks exact array
round trips; pass `--output artifacts/runs/film-001` to keep its record. Its default
execution uses a temporary directory for smoke checks.

`run_recorded` executes the Job once, then calls the optional `sample(result)`
function. It compares the portable inputs, installed source inventory and recorded
environment before and after execution, and refuses to save if they changed.
The source inventory describes files on disk; it cannot prove that a long-running
interpreter has reloaded earlier edits. Restart the interpreter after editing
source when you need a clean experimental record.

## Save an existing result

```python
from wavebit import run
from wavebit.records import save_run

result = run(job)
record = save_run(
    "artifacts/runs/film-002", job=job, result=result,
    samples={"inside_film": result.field(x=(0.,), y=(0.,), z=(.1,))},
)
```

This checks that Job and Result use the same solver. The caller is responsible for
associating the exact inputs with an already computed result. The manifest labels
this association explicitly; source and environment are observed at save time.
Prefer `run_recorded` when the record should also cover execution.

For FDTD, construct `Job(solver="fdtd", problem=..., numerics=...)` in the same way.
Time series, Fourier fields, diffraction, flux, source calibration traces and
requested final fields retain their numerical values and physical metadata.

## What is preserved

| Data | Saved representation |
| --- | --- |
| Problem, materials, units, numerical and execution settings | Versioned `wavebit.job/v2` JSON and canonical Job key |
| Materialized outputs and explicit samples | Numeric NPZ arrays plus named fields in JSON |
| Array dtype, shape and byte order | Checked against the array payload when loaded |
| Coordinates, component names, units, reference planes, validity and diagnostics | Named fields alongside the values |
| RCWA wavelengths and requested scattering matrices | `record.extra` |
| FDTD source calibration and requested final fields | `record.extra` |
| Python, platform and relevant installed package versions | Manifest environment metadata |
| Installed Wavebit Python, type stubs and runtime material data | Per-file hashes, combined source identity and `source.zip` by default |
| Git context when the package belongs to a checkout | Commit, package dirty state and tracked-package diff to HEAD |

`include_source=False` omits the source ZIP while retaining its inventory and
identity. The Git patch covers tracked package files; untracked Python files are
included in the source ZIP. Unrelated repository files, documents, environment
variables and credentials are not collected. The snapshot is the installed
package, not a full repository checkout or an installable wheel.

Every payload has a SHA-256 checksum. The manifest is written last as a completion
marker. An interrupted write may leave an incomplete directory; `load_run`
refuses it. Checksums detect damaged or inconsistent data; they are not a signature
proving who created the record.

The manifest's schema is `wavebit.run/v2`. A `wavebit.run/v1` record, written before the
[public names were renamed](../migration/public-names.md), is refused with code
`record.format` rather than converted; rerun its Job with the current names.

## Reading and reproducing a result

`load_run(path)` verifies the manifest, Job identity, source inventory and payloads.
It never executes archived Python, calls the solver, imports a type name from the
record, or loads pickle/object arrays. Loaded numerical arrays are owned,
read-only NumPy arrays. Dataclass observations become `RecordedObject` mappings:
use `observation["values"]`, not a live observation method. Their `type_name`
retains the original type as descriptive text.

This is a measured-data archive. It does not preserve gradient graphs, prepared
solver state or deferred evaluators. RCWA field reconstruction and FDTD far-field
evaluation appear under `record.deferred`; explicitly evaluate what you need and
pass it through `samples`. For a far-field sample include `theta`, `phi`, angular
units and the returned field or intensity units together with the values.
Save concrete values outside JAX `jit`/`grad` or Torch compilation. Device arrays
are detached and copied to the host; the portable Job itself requires explicit
host arrays rather than backend buffers.

Re-execution is a deliberate separate step:

```python
from wavebit import run
repeated_result = run(loaded.job)
```

Do that in an environment with the requested backend, precision and source
version. For example, a double-precision JAX Job requires x64 to be enabled before
creating arrays. The record supplies inputs and provenance, not a dependency
installer or a promise of bitwise reproducibility across hardware and versions.
Compare measured outputs and convergence for the physics you are studying.

The repository's test receipts have a separate purpose: see
[recorded verification](../development/verification.md) for quick/full/research
profiles and resumable test execution.
