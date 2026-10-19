# Public names in `wavebit.fdtd` (2026-10-19)

Wavebit is still in alpha, so the general-FDTD declarations were renamed in one clean break.
Every class is now named for what it is, every monitor for what it records, and every keyword
carries its unit where the unit is not obvious. The old names were removed, not aliased: an old
attribute (`fdtd.Record`) or keyword (`discretisation=`) fails at once and names its replacement
(the import forms are under [What an old spelling does now](#what-an-old-spelling-does-now)).
Nothing numerical changed. The lowered scene,
the static plan and every output are identical (`tests/fdtd/test_public_names.py` pins the plan
skeletons of eight scenes measured before the rename).

The design and its reasoning are in [issue #57](https://github.com/fieldalgo/wavebit/issues/57).
This page covers `wavebit.fdtd`; `fdtd.native`, `fdtd.periodic`, RCWA and the shared
namespaces follow in later changes and will be added here.

## Classes

| Before | After | Why |
| --- | --- | --- |
| `fdtd.Record` | `fdtd.TimeSeriesMonitor` | it records one component at one point, every step |
| `fdtd.Field` | `fdtd.FieldMonitor` | a monitor, not a field |
| `fdtd.Flux` | `fdtd.FluxMonitor` | a monitor, not a flux |
| `fdtd.Diffraction` | `fdtd.DiffractionMonitor` | a monitor of diffraction orders |
| `fdtd.FarField` | `fdtd.FarFieldMonitor` | a monitor that projects to the far field |
| `fdtd.Slab` | `fdtd.SnappedSlab` | it fills whole Yee cells; `Box` and the other bodies are smoothed |
| `fdtd.Region` | `fdtd.SnappedBox` | a cell-snapped box, the twin of `SnappedSlab` |
| `fdtd.Dipole` | `fdtd.CurrentDipole` | it drives a current moment in A·m, not a dipole moment in C·m |
| `fdtd.Axis1D` | `fdtd.InvariantAxis` | a one-cell axis the structure does not vary along |
| `fdtd.Symmetry` | `fdtd.MirrorSymmetry` | a mirror plane, with `parity=+1` or `-1` |
| `fdtd.Absorber` | `fdtd.ConductiveAbsorber` | a graded conductive layer, as opposed to `PML` |
| `fdtd.Decay` | `fdtd.DecayStop` | a stopping rule, used as `Numerics(stopping=...)` |

`InvariantAxis` has no `algebraic` field: the old `Axis1D(algebraic=True)` was the only value
it accepted.

## Keywords

| Before | After |
| --- | --- |
| `Numerics(discretisation=...)` | `Numerics(discretization=...)` |
| `Numerics(decay=fdtd.Decay(threshold=..., window_seconds=...))` | `Numerics(stopping=fdtd.DecayStop(threshold=..., window_seconds=...))` |
| `PML(stabilize=...)` | `PML(stabilizing_loss=...)` |
| `PlaneWave`, `Dipole`, `TFSF` `(delay=...)` | `PlaneWave`, `CurrentDipole`, `TFSF` `(delay_seconds=...)` |
| `Diffraction(side='reflect' or 'transmit')` | `DiffractionMonitor(wave='reflected' or 'transmitted')` |
| `Bloch.from_source(..., axis1d=...)` | `Bloch.from_source(..., invariant_axes=...)` |

The values are unchanged: `delay` was already in seconds, and `stabilizing_loss` takes the
number `stabilize` took.

## Names before keywords

A declaration with a name takes only that name positionally; everything else is a keyword. This
applies to `Structure`, the five monitors and the three sources:

```python
# Before
fdtd.Structure('film', fdtd.Slab(z=(-0.2, 0.0)), glass)
fdtd.Record('probe', (0.0, 0.0, 0.1))

# After
fdtd.Structure('film', shape=fdtd.SnappedSlab(z=(-0.2, 0.0)), material=glass)
fdtd.TimeSeriesMonitor('probe', at=(0.0, 0.0, 0.1))
```

Every other declaration was already keyword-only. A misspelt keyword is refused with the
nearest valid one.

## One public path per name

Import every declaration from `wavebit.fdtd`. The implementation modules
`wavebit.fdtd.problem`, `geometry`, `boundaries`, `sources`, `monitors`, `numerics`, `result`
and `results` are no longer public; an attribute access such as `fdtd.monitors` fails and says
to use `wavebit.fdtd`. The public submodules are `fdtd.native`, `fdtd.periodic` and
`fdtd.diagnostics`, and `dir(fdtd)` lists only the public names.

## What an old spelling does now

```text
>>> fdtd.Record
AttributeError: wavebit.fdtd.Record is now wavebit.fdtd.TimeSeriesMonitor. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> fdtd.Numerics(grid=grid, duration=duration, discretisation='voxel')
TypeError: Numerics(discretisation=...) is now Numerics(discretization=...). The old keyword was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
```

An attribute access (`fdtd.Record`, `fdtd.monitors`) is the spelling that names the
replacement. `from wavebit.fdtd import Record` raises Python's own `ImportError: cannot import
name 'Record'`, and `import wavebit.fdtd.monitors` raises `ModuleNotFoundError`: Python discards
a module's message for those two forms, as it does for NumPy's removed names. The tables above
give the replacement for each.

## Report keys and issue paths

The same renames reach the data a check, a preparation and a result return:

| Before | After |
| --- | --- |
| `report.resolved['numerical_policies']['discretisation']`, and the same key in `Result.provenance` | `...['numerical_policies']['discretization']` |
| issue path `numerics.discretisation` | `numerics.discretization` |
| issue paths `numerics.decay`, `numerics.decay.window_seconds` | `numerics.stopping`, `numerics.stopping.window_seconds` |

`Prepared.contract` describes the live source leaves as `amplitude/phase/delay_seconds`.

## Job files and run records

A saved `Job` records each declaration by its public name, so the schema is now
`wavebit.job/v2`, for both solvers. The execution settings are tagged by their one public path,
`wavebit.Execution` (it was `wavebit._common.execution.Execution`). A `wavebit.job/v1` file of
either solver is refused with code `job.schema` rather than read with names that no longer
exist: build the Job again from current declarations and save it. Every job key changes, RCWA
jobs included, because the schema and the execution tag are part of what a key hashes. Run
records are `wavebit.run/v2` for the same reason, and a `wavebit.run/v1` record is refused with
code `record.format`. The capability manifest is version 23.

## A complete example

`examples/fdtd/pulse_slab.py` runs a pulse through a film with the new names. The first
calculation in [Getting started](../getting-started/fdtd.md) does the same, step by step.
