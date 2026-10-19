# Migrating to the physical API

:::{note}
The general-FDTD declarations were renamed on 2026-10-19 (`fdtd.Record` is now
`fdtd.TimeSeriesMonitor`, `fdtd.Region` is now `fdtd.SnappedBox`, and so on), and the
`fdtd.native` and `fdtd.periodic` ones the same day (`native.Record` and `periodic.Record` are
now `Snapshots`, `native.Tail` is now `native.DecayCheck`, and so on). See
[public names](public-names.md) for every old and new spelling.
:::

`wavebit.compat` has been removed. Calculations use physical Problem/Numerics declarations,
explicit execution and `check` → `prepare` → `prepared.run(problem)`. The underlying numerical
engines and their scientific regression tests remain; no replacement mutable facade is exported.

| Earlier workflow | Current interface |
| --- | --- |
| Mutable `rcwa(...)` | `rcwa.Problem`, `rcwa.Numerics`, `rcwa.check/prepare/run` |
| General `fd` scene/plan/run | `fdtd.Problem`, physical `fdtd.Scene`, separate `fdtd.Numerics` |
| Periodic `fdtd(...)` | [`fdtd.periodic`](../guide/fdtd/periodic-problems.md), explicit `shared` or `reference` algorithm |
| `fdtd_grid(...)` | [`fdtd.native`](../guide/fdtd/native-grids.md), SI current/conductivity/poles, fields, snapshots and Fourier integrals |
| `fit_fdtd_modes` | `observations.fit_modes`, with seconds, Hz and decay rates in 1/s |
| Custom general-FDTD geometry | `fdtd.Shape` static geometry protocol |
| `solver.slice_structure` | `rcwa.sample_structure(problem, quantity='epsilon' or 'mu')` |
| Experimental NVF and scattering operators | `rcwa.experimental` and `rcwa.operators` |
| `JobSpec` v10 | `migration.migrate_legacy_jobs(document, units=Units(...))` → tuple of current `Job` objects |
| Capability manifest v21 | v22: `current_lifecycle` is the public index; detailed `solvers`/result tables explicitly describe private normalized engines. v23 (2026-10-19): the [public names](public-names.md) and `wavebit.job/v2`. v24 (2026-10-19): the [native and periodic names](public-names.md#native-grid-and-periodic-stack-names) |
| Import-time JAX precision change | Configure `jax.config.update('jax_enable_x64', True)` before constructing arrays |

Spatial lengths use `Units`; times, conductivity and material rates use SI. H is A/m.
Layer stacks remain bottom-to-top; angles are radians. Preserve the declared numerical
algorithm, cell origin, phase reference, source normalization and integration measure when
comparing complex fields. A native Fourier field is a time integral, not a calibrated phasor;
a periodic plane power is W/m for invariant-y geometry and W for a crossed cell.

## Existing work

All maintained examples use the physical API. Internal numerical tests and research instruments
import the exact implementation they measure; these private paths have no public stability promise.
Their independent references, tolerances and scientific assertions remain in place.

Changing shapes under JIT uses the supported device guards or explicit
[`topology_check='host'`](../guide/rcwa/guarded-shape-jit.md). Numerical topology checks remain active.
General FDTD exposes [stopping/absorbers](../guide/fdtd/stopping-and-absorbers.md) and explicit
[substrate continuation](../guide/fdtd/fd-differentiation.md). Custom `Shape` methods receive grid
lengths in the Problem's unit; all declared instance attributes are static for prepared reuse.
Custom support must stay clear of Open padding, and custom objects are not portable Job tags.
Unimplemented mode-source/mode-monitor/Gaussian-beam placeholders were never working general-FDTD solvers.

## Job files

```python
from wavebit import Job, Units, run
from wavebit.migration import migrate_legacy_jobs

# old_document is a v10 JSON string or mapping. Supply its actual original length unit.
jobs = migrate_legacy_jobs(old_document, units=Units(length='nm'))
for job in jobs:
    portable = Job.from_json(job.to_json())
    result = run(portable)
```

Conversion expands every sweep point in its original order and preserves numerical choices,
material tables and polarization. It does not run a solver or overwrite the source document.
Unsupported fields fail explicitly. Compare converted observations with the original run before
replacing downstream records. Other historical schema versions retain their original version gate;
changing a JSON version number is not a migration.

## Historical reproduction

The ten older [numbered tutorials](https://github.com/fieldalgo/wavebit/blob/main/tutorials/README.md#historical-tutorials)
are historical teaching records and require the pinned source below. Their
calculation cells and outputs remain unchanged; the tutorial index maps each topic
to current physical examples and guides. The two `00` lifecycle notebooks run with
the current package.

Frozen measurements keep their original bytes, commit and dependency provenance. Use that
recorded checkout/environment to replay them. For scripts deliberately using the final compat
spelling, the pre-retirement baseline is
[3c8a50f8a6506240169538957e99648a4b18ca3a](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
Create a separate checkout at that revision; do not treat current package installation as an
implicit historical environment. Earlier measurement commits take precedence over this convenience pin.

The earlier [API](legacy-api.md), [general-FDTD](general-fdtd-v1.md), [material](materials-v1.md),
[Job](jobs-v1.md), [backend](backends-v1.md) and [autodiff](autodiff-v1.md) records explain old
conventions. Their historical snippets do not describe currently importable names. Documentation
relocation and consumer migration are separate from a new scientific measurement.
