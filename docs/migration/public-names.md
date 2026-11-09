# Public names and result keys (2026-10-19, 2026-10-20, 2026-11-08)

Wavebit is still in alpha, so the general-FDTD declarations were renamed in one clean break.
Every class is now named for what it is, every monitor for what it records, and every keyword
carries its unit where the unit is not obvious. The old names were removed, not aliased: an old
attribute (`fdtd.Record`) or keyword (`discretisation=`) fails at once and names its replacement
(the import forms are under [What an old spelling does now](#what-an-old-spelling-does-now)).
Nothing numerical changed. The lowered scene,
the static plan and every output are identical (`tests/fdtd/test_public_names.py` pins the plan
skeletons of eight scenes measured before the rename).

The design and its reasoning are in [issue #57](https://github.com/fieldalgo/wavebit/issues/57).
This page covers `wavebit.fdtd` and, from a second change the same day, the explicit-grid
modules `fdtd.native` and `fdtd.periodic`
([native-grid and periodic-stack names](#native-grid-and-periodic-stack-names)), and, from a
third change the next day, RCWA, the observation objects and the shared modules
([shared, observation and RCWA names](#shared-observation-and-rcwa-names)), and, from a fourth
change the same day, the keys every solver's results and reports use
([result and report keys](#result-and-report-keys)), and, from a fifth change on 2026-11-08
([issue #54](https://github.com/fieldalgo/wavebit/issues/54)), the names of diffraction
amplitudes, which now say whether a value is a ratio, a field or a spectrum
([amplitude ratios, fields and spectra](#amplitude-ratios-fields-and-spectra)).

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
code `record.format`. This change made the capability manifest version 23; the native and
periodic names [below](#native-grid-and-periodic-stack-names) made it version 24.

## A complete example

`examples/fdtd/pulse_slab.py` runs a pulse through a film with the new names. The first
calculation in [Getting started](../getting-started/fdtd.md) does the same, step by step.

## Native-grid and periodic-stack names

`fdtd.native` (a model already sampled at Yee sites) and `fdtd.periodic` (a periodic stack)
were renamed the same way in a second clean break the same day: no aliases, and an old
spelling fails and names its replacement. Each module keeps its own classes, so the tables
below name every one by its module. Several examples import the periodic module as
`from wavebit.fdtd import periodic as fdtd`; there `fdtd.Snapshots` is `periodic.Snapshots`,
not a general-FDTD monitor.

### Native and periodic classes

| Before | After | Why |
| --- | --- | --- |
| `native.Record` | `native.Snapshots` | it keeps snapshots of the grid's fields every `every` steps; `fdtd.TimeSeriesMonitor` is the one-point record |
| `periodic.Record` | `periodic.Snapshots` | the same, within optional physical bounds |
| `native.Initial` | `native.InitialField` | the field at the start, the name `fdtd.InitialField` already has |
| `native.Tail` | `native.DecayCheck` | a final-window decay diagnostic that never stops a run, unlike `fdtd.DecayStop` |

`periodic.DecayCheck` is new: it is the same declaration for periodic stacks and replaces the
two `periodic.Numerics` keywords in the next table.

### Native and periodic keywords

| Before | After |
| --- | --- |
| `native.Outputs(record=...)`, `periodic.Outputs(record=...)` | `native.Outputs(snapshots=...)`, `periodic.Outputs(snapshots=...)` |
| `native.Record(bounds=...)` | `native.Snapshots(slices=...)` |
| `components=('ex', 'hy')` in `native.Record` or `periodic.Record` | `components=('Ex', 'Hy')` in `native.Snapshots` or `periodic.Snapshots` |
| `native.Spectrum(planes=...)` | `native.Spectrum(z_indices=...)` |
| `native.Numerics(tail=native.Tail(window_seconds=w, tolerance=t))` | `native.Numerics(decay_check=native.DecayCheck(window_seconds=w, threshold=t))` |
| `periodic.Numerics(tail_steps=N, tail_tolerance=t)` | `periodic.Numerics(decay_check=periodic.DecayCheck(window_seconds=N*dt_seconds, threshold=t))` |
| `periodic.RasterDispersion(models=...)` | `periodic.RasterDispersion(poles=...)` |
| `periodic.Pulse(width_seconds=W)` | `periodic.Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*W))` |

`slices=` is native only. Its selectors are integer grid indices, per `(z, y, x)` axis an index
or a `(start, stop[, step])` tuple, and the name says so. `periodic.Snapshots` keeps `bounds=`
because its bounds are physical: a mapping from `'x'`, `'y'` or `'z'` to a `(low, high)` pair in
the problem's length unit.

Components are spelled `'Ex'` to `'Hz'`, as every other Wavebit output spells them. A
lowercase `'ex'` is refused, and the message names `'Ex'`. `z_indices=` says what `planes=`
held: integer z-node indices, not positions. `poles=` holds the same `materials.Drude` and
`materials.Lorentz` objects that `DispersiveMaterial(poles=...)` takes. `threshold=` is the old
`tolerance=` (native) or `tail_tolerance=` (periodic), the same number with the same default,
`1e-4`.

`Pulse(width_seconds=W)` was the time width `W` of the envelope `exp(-((t - t0)/W)**2)`.
`frequency_sigma_hz` describes the same Gaussian by the standard deviation of its amplitude
spectrum, in Hz, as `fdtd.GaussianPulse(frequency_sigma_hz=...)` does, so `fdtd` and
`fdtd.periodic` describe a pulse the same way. For this envelope sigma = 1/(sqrt(2)·pi·W):
write that expression to carry an old width over. A longer pulse has a smaller sigma, so a
refusal that said to reduce the width now says to raise `frequency_sigma_hz`. The default
pulse is unchanged: the envelope whose 1/e half-width is one period at the centre wavelength,
sigma = f0/(sqrt(2)·pi).

### Native and periodic results

| Before | After |
| --- | --- |
| `native.Result.fields`, `periodic.Result.fields` | `native.Result.snapshots`, `periodic.Result.snapshots` |
| keys `'ex'` to `'hz'`, each `TimeSeries` with `components == ('ex',)` | keys `'Ex'` to `'Hz'`, with `components == ('Ex',)` |
| `native.Result.polarization` | `native.Result.polarization_density` |
| `native.Result.previous_polarization` | `native.Result.previous_polarization_density` |

`polarization_density` holds what `polarization` held: the material polarization density P in
C/m² for each pole, at the final step and, for `previous_polarization_density`, one step
earlier. The name says it is a density, not the polarization of a wave. Reading an old
attribute raises `AttributeError` and names the new one.

### The decay window

The periodic window was a step count, `tail_steps=N`. It is now a duration,
`DecayCheck(window_seconds=...)`, as the native one already was, and both are rounded to whole
time steps by one rule: up, except that a quotient within a few units in the last place above a
whole number counts as that number, so a window written as `N` time steps covers `N` steps. To
carry over `tail_steps=N`, take the time step from a check:

```python
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_unit_seconds)
dt_seconds = periodic.check(problem, numerics=numerics).resolved['time_step_seconds']
numerics = periodic.Numerics(spacing=1/30, duration_seconds=90*units.time_unit_seconds,
                             decay_check=periodic.DecayCheck(window_seconds=40*dt_seconds))
```

For native the rule was already "round up"; the allowance is new. A window written as
`N*time_step_seconds` whose quotient landed just above `N` through rounding error used to cover
`N + 1` steps and now covers `N`. That changes only the decay diagnostic in
`validity.convergence`, never the fields or the spectrum. A window as long as the run, or
longer, is refused, and the refusal names `DecayCheck(window_seconds=...)`.

### Native and periodic refusals

```text
>>> native.Tail
AttributeError: wavebit.fdtd.native.Tail is now wavebit.fdtd.native.DecayCheck; given as Numerics(decay_check=...), with tolerance= now threshold=. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> periodic.Pulse(width_seconds=3e-15)
TypeError: Pulse(width_seconds=...) is now Pulse(frequency_sigma_hz=...); frequency_sigma_hz = 1/(sqrt(2)*pi*width_seconds), the spectral standard deviation of the same Gaussian, as in fdtd.GaussianPulse(frequency_sigma_hz=...). The old keyword was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
```

The import forms behave as [above](#what-an-old-spelling-does-now): `from wavebit.fdtd.native
import Record` raises Python's own `ImportError`. Issue paths follow the keywords:
`outputs.record` is `outputs.snapshots` (`outputs.snapshots.slices` for native,
`outputs.snapshots.bounds` for periodic), `numerics.tail` (native) and `numerics.tail_steps`
(periodic) are `numerics.decay_check`, and `dispersion.models` is `dispersion.poles`. A refusal
raised inside the engine named the engine's own keyword; it now names the declaration the
caller wrote: `Snapshots(slices=...)` for `record_region`, `Spectrum(z_indices=...)` for
`monitor_z`, and for periodic
`Snapshots(bounds=...)`, `Pulse(frequency_sigma_hz=...)`, `Numerics(duration_seconds=...)`,
`Numerics(nominal_indices=...)` and `Numerics(orders=...)` for `record_bounds`, `pulse_width`,
`n_periods`, `mesh_indices` and `truncation_order`. A type refusal names its module:
`expected periodic.Pulse`, not `expected Pulse`.

### Native and periodic job files

Saved jobs keep the schema `wavebit.job/v2`, which the general-FDTD change introduced. A v2 file
saved between the two changes with an old native or periodic name (a
`wavebit.fdtd.native.Record` tag, or a field such as `tail_steps` or `width_seconds`) is refused
with code `job.encoding`, naming the unknown type or field, rather than read with names that no
longer exist: build the Job again from current declarations and save it. The key of every
native and periodic job changes, because the tags and fields it hashes changed (every native
job holds `InitialField` and `Outputs`, every periodic one `Pulse` and `Numerics`); RCWA and
general-FDTD job keys are unchanged by this change. The capability manifest is version 24.

### Native and periodic numbers

The engine is unchanged and receives the same inputs. Old and new spellings, run side by side
in um, gave bit-identical results for a native case (E, H, polarization density, snapshots,
spectrum and a 7-step decay window) and a periodic case (R, T, r_s, plane phasors, snapshots,
the pulse width and a 40-step decay window). Two things can differ. A periodic pulse width is
recomputed from sigma, so it matches the old width only when sigma is written as the expression
of that width, `frequency_sigma_hz=1/(np.sqrt(2)*np.pi*W)`, and then to within its last bit or
two: it was exact in these cases and differed by at most 2 units in the last place over 10⁵
random widths. A rounded decimal (`2.25e13` for a width of three um time units,
`3*units.time_unit_seconds`) gives a slightly different width, and slightly different numbers. The
other is the native rounding allowance under [the decay window](#the-decay-window), which
changes only the decay diagnostic.

`examples/fdtd/fdtd_record.py` takes periodic snapshots and `examples/fdtd/fdtd_grid_gradient.py`
differentiates a native Fourier integral at `Spectrum(z_indices=...)`, both with the new names.

## Shared, observation and RCWA names

The third clean break (2026-10-20) covers what both solvers share, the objects a result hands
back, and RCWA. As before, nothing is aliased: an old name, keyword or attribute fails and names
its replacement, and nothing numerical changed.

### Shared modules

| Before | After | Why |
| --- | --- | --- |
| `Units.length_metres` | `Units.length_unit_meters` | Meters in one length unit; one spelling of "meter" throughout |
| `Units.time_seconds` | `Units.time_unit_seconds` | Seconds in one internal time unit, not a time in seconds |
| `Units.to_metres(x)` | `x * units.length_unit_meters` | A multiplication |
| `Units.time_to_internal`, `time_from_internal`, `rate_to_internal`, `conductivity_to_internal` | private | Conversions the solvers make at their own boundaries |
| `Lattice.oblique(lx, ly, angle)` | `Lattice.oblique(length_u, length_v, angle)` | The lengths of the two lattice vectors, which are not along x and y |
| `wavebit.lattice.__all__` with `is_axis_aligned`, `jacobian`, `order_kxy`, `periods_of`, `reduced_basis`, `reduction_transform`, `sample_grid` | `DrawnOn`, `Lattice`, `as_lattice`, `meshgrid_xy` | The others work on the raw basis for the solvers |
| `Material.is_anisotropic` | `Material.is_tensor` | It says whether epsilon or mu was given as three principal values or a 3x3 tensor, which an isotropic diagonal tensor also is |
| `TabulatedMaterial.interpolation` | `TabulatedMaterial.interpolated_quantity` | It names the quantity interpolated (`'epsilon'` or `'index'`), not a method |
| `Execution.scope()` | private | `run`, `check` and `prepare` open it themselves |
| `wavebit.materials.time_domain`, `wavebit.materials.tabulated` | private | Their classes are the solvers' normalized-unit models; the public materials are in `wavebit.materials` |
| `NumericalError` was a `ValidationError` (and a `ValueError`) | `NumericalError` is a `WavebitError`, not a `ValidationError` | A failed solve is not a bad input. `WavebitError` is the new root of every Wavebit error |

`except ValueError` no longer catches a `NumericalError`; catch `wavebit.NumericalError` or
`wavebit.WavebitError`. `ValidationError` is still a `ValueError`.

### Observation objects

| Before | After | Why |
| --- | --- | --- |
| `wavebit.SpectralFieldSamples` | `wavebit.RawFieldSpectrum` | Raw Fourier fields, like `RawDiffractionSpectrum` |
| `TimeSeries.times`, `TimeSeries.time` | `TimeSeries.times_seconds` | One name, with its unit |
| `RawFieldSpectrum.frequencies`, `fdtd.SpectralFlux.frequencies` | `.frequencies_hz` | With its unit |
| `RawFieldSpectrum.spectrum`, `RawDiffractionSpectrum.spectrum`, and the `SpectralFlux.metadata` key `spectrum` | `.sidedness` | It holds `'two-sided'`, not a spectrum |
| `fdtd.SpectralFlux.energy_density` | `.spectral_energy` | Energy per unit frequency through the plane, not a density in space |
| `fdtd.Result.sources` | `fdtd.Result.source_references` | The recorded source references, not the sources |
| `DiffractionOrders.cartesian`, `DiffractionOrder.cartesian` | `.amplitude_xyz`, renamed again by the [fifth change](#amplitude-ratios-fields-and-spectra): now `.amplitude_ratio_xyz` (`.spectrum_xyz` on a `RawDiffractionSpectrum`) | The x, y, z amplitude components |
| `DiffractionOrders.kz`, `DiffractionOrder.kz` | `.k_normal` | The wavevector component along the port normal (kz for a z-normal port), with its Cartesian sign: an order leaving through a −z port has `Re(k_normal) < 0` |
| `DiffractionOrders.at_wavelength(i)` | `.at_wavelength_index(i)` | It takes an index, not a wavelength |
| `wavebit.records.RecordedFields` | `wavebit.records.RecordedObject` | It holds any recorded object, not only fields |
| `periodic.Result.sample(times=...)` | `sample(times_seconds=...)` | With its unit |

The constructors refuse the old keywords the same way (`TimeSeries(times=...)` names
`times_seconds=`).

### RCWA

| Before | After | Why |
| --- | --- | --- |
| `rcwa.OrderExcitation(orders=(((m, n), w), ...))` | `rcwa.IncidentOrders(weights=(((m, n), w), ...))` | Incident orders and the complex weight of each |
| `rcwa.GaussianBeam(spectral_extent=...)` | `rcwa.GaussianBeam(angular_spectrum_extent=...)` | The extent of the beam's angular spectrum, in 1/waist |
| `rcwa.sample_structure(n=...)` | `rcwa.sample_structure(samples=...)` | The number of samples |
| `rcwa.operators.Port` fields `kx`, `ky`, `index` | `kx_over_k0`, `ky_over_k0`, `refractive_index` | Normalized wavevector components and the port medium's index |
| `rcwa.experimental` builders `truncation_order=` / `order=` | `orders=` | The same keyword as `rcwa.Numerics(orders=...)` |
| `rcwa.BoundaryPower.integration_measure` | `rcwa.BoundaryPower.power_region` | The same values as before; the name matches `Outputs(power_region=...)` |
| `rcwa.Result.diagnostics['source_groups']` | `diagnostics['source_group_counts']` | It counts the coherent source groups |
| `wavebit.rcwa.layer_thicknesses` | `wavebit.layer_thicknesses` | FDTD uses it too |
| `wavebit.rcwa.plotting` | private | It takes the engine solver, not a public result |

The RCWA warnings a solve can raise have one public home, `wavebit.rcwa.diagnostics`:
`RayleighPointWarning`, `AutoDetuneWarning`, `ExceptionalPointWarning`,
`ComplexIncidenceWarning`, `GainCutoffWarning` and `SvdConvergenceWarning`.

`Port` is a named tuple, so only its attributes can name their replacement: `port.kx` fails
saying `kx_over_k0`, but `Port(kx=...)` raises Python's own `TypeError` and
`port._replace(kx=...)` its own `TypeError` (`ValueError` before Python 3.13) for an unknown
field. The tables above give the replacement.

`rcwa.check` now refuses a `GaussianBeam(angular_spectrum_extent=...)` that is not a positive,
finite real number with code `source.angular_spectrum_extent`; the same values were refused
before, at check too, as `rcwa.invalid` in the engine's words. An `IncidentOrders` refusal has code `source.weights`
(it was `source.orders`).

The `rcwa.check` and `rcwa.run` refusals now name the public settings
(`Numerics(orders=...)`, `Numerics(detuning=...)`, `precision=`,
`GaussianBeam(angular_spectrum_extent=...)`) where they used the engine's keywords. The text of
the RCWA warnings still uses the engine's keywords; that is a later change.

### Shared, observation and RCWA job files and run records

Saved jobs keep the schema `wavebit.job/v2`. A v2 file with an old RCWA name (a
`wavebit.rcwa.OrderExcitation` tag, or a `spectral_extent` field) is refused with code
`job.encoding`: build the Job again from current declarations and save it. The key of every RCWA
job that holds `IncidentOrders` or a `GaussianBeam` changes; other job keys do not.
A `TabulatedMaterial` is still saved under the field `interpolation`, so jobs that hold one keep
their keys.

Run records stay `wavebit.run/v2`. A record saved from now on stores the new field names
(`times_seconds`, `frequencies_hz`, `sidedness`, `amplitude_xyz` (since the
[fifth change](#amplitude-ratios-fields-and-spectra), `amplitude_ratio_xyz`), `k_normal`, `spectral_energy`,
`power_region`, the `Port` fields, `extra['source_references']`, the diagnostics key
`source_group_counts`). A record saved earlier keeps the names it was saved with, since
`load_run` returns the stored fields as `RecordedObject` mappings rather than rebuilding the
classes, but it loads only if its job does: a record whose job holds an `OrderExcitation` or a
`GaussianBeam` is refused with code `job.encoding`, like that job file. The capability manifest is version 25.

### Shared, observation and RCWA numbers

The solvers receive the same inputs. The FDTD plan skeletons are unchanged: the materials
modules that became private are still recorded under their old names. No reference golden or
solver tolerance changed.

## Result and report keys

The fourth change (2026-10-20) gives the facts every solver reports one key each: the same check,
time step, reference value or error has the same name in `rcwa`, `fdtd`, `fdtd.native` and
`fdtd.periodic`. A key is renamed, not aliased: reading an old key raises `KeyError`. Nothing
numerical changed.

### Validity, time step and decay

| Before | After |
| --- | --- |
| native `validity.checks['stable']`, `['finite_fields']` | `['runtime_stability']`, `['numerical_finite']` |
| periodic `validity.checks['stable']`, `['finite']` (R only) | `['runtime_stability']`, `['numerical_finite']` (R and T) |
| general FDTD: no finiteness check | `validity.checks['numerical_finite']` over every monitor output and the final fields |
| `fdtd` `report.resolved['dt_seconds']`, `provenance['dt_seconds']`, `contract['resolved_dt_seconds']` | `'time_step_seconds'`, `'time_step_seconds'`, `'resolved_time_step_seconds'` |
| native `diagnostics['tail']`; `validity.convergence` was that report | `diagnostics['decay']`; `validity.convergence == {'decay': report or None}` |
| periodic `diagnostics['convergence']`; `validity.convergence` was that report plus `residual` | `diagnostics['decay']`; `validity.convergence == {'decay': report or None, 'residual': ...}` |
| inside a decay report: `tolerance`, periodic `units` sub-mapping | `threshold` (as `DecayCheck(threshold=)`), `<quantity>_units` keys |
| `invariant_axes` a count (general FDTD); periodic `diagnostics['invariant_y']` | `invariant_axes`, a tuple of axis names, `()` when none |

Under `fdtd.periodic` `algorithm='reference'` the stepping assesses no stability, so
`validity.checks['runtime_stability']` is `None` (not assessed). It used to read `True` by
default.

### Provenance

Every solver's `provenance` has `solver`, `execution`, `numerics`, `units`, `coordinate_frame`
and `time_convention` (`'exp(-i*omega*t)'`, one spelling everywhere). General FDTD no longer has
the flat `backend`, `device` and `precision` keys: read `provenance['execution'].device`.
`provenance['units']` is the `Units` declaration. native and periodic also report
`time_step_seconds`. periodic's `coordinate_convention` is `coordinate_frame`. rcwa's
`code_version` is the installed package version (it was a fixed `'0.1.0'`).

### Diffraction references

`DiffractionOrders.reference` (and `RawDiffractionSpectrum.reference`) has one core key set
from every producer. The table gives the names of this change; the
[fifth change](#amplitude-ratios-fields-and-spectra) renamed its amplitude keys:

| Key | Meaning | Before |
| --- | --- | --- |
| `port` | the face the orders leave through: `'top'`/`'bottom'` in rcwa and `fdtd.periodic`, a face such as `'z+'` in `fdtd` | fdtd `port` held `'reflect'`/`'transmit'`; periodic `side` |
| `wave` | `'reflected'` or `'transmitted'` | new |
| `plane` | the port plane along its normal, in length units | rcwa and periodic `z` |
| `length_units` | the length unit of `plane` and the coordinates | `length_unit`; periodic `coordinates` |
| `lateral_origin` | coordinates | periodic `'cell_centre'` (now `(0., 0.)`) |
| `incident_amplitude`, `incident_amplitude_units`; now `incident_field` or `incident_spectrum` with their `_units` | the incident amplitude and its unit | rcwa `incident_electric_amplitude_v_per_m` |
| `cell_measure`, `cell_measure_units` | the periodic cell's measure: `'m^2'`, or `'m'`/`'1'` per unit length of each invariant axis | `cell_area_m2`; numeric `integration_measure(_units)` |
| `invariant_axes` | axis names | a count |

rcwa also has `incident_power` and `incident_power_units` (was `incident_power_w`);
`fdtd.periodic` has them too since the fifth change. The unit labels of a reference
(`amplitude_units`, now `amplitude_ratio_units`, or `spectrum_units` on a raw spectrum;
`power_units` where there is a power; `wavevector_units` where there are wavevectors) are in
`reference` for every producer. A
dimensionless amplitude is labeled `'1'`. rcwa's metadata string `integration_measure`
(`'periodic cell'`) is gone.

### Units and sample times

- Unit-label keys end in `_units`: `length_unit` → `length_units`, `time_unit` → `time_units`,
  native `E_unit`/`H_unit`/`polarization_unit`/`spectral_power_unit` → `*_units`, periodic plane
  metadata `power_unit` → `power_units`. One spelling per unit, ASCII: `'V*s/m'`, `'A*s/m'`,
  `'C/m^2'`, `'W*s^2'`.
- `FieldSamples.units` (and `TimeSeries`, `RawFieldSpectrum`) is one mapping everywhere: each
  component present (`'Ex'` to `'Hz'`) plus `'coordinates'`. An fdtd `TimeSeries.units` was a
  string, native snapshots `{'field', 'coordinates', 'time'}`, final fields `{'E', 'H'}`.
- E and H sample times: fdtd `final_fields` metadata `E_time`/`H_time` and native
  `diagnostics['time_seconds']`/`['magnetic_time_seconds']` → `E_time_seconds`/`H_time_seconds`.
- `fdtd.Result.source_references` entries: `time` → `times_seconds`, `magnetic_time` →
  `magnetic_times_seconds`, `field_time` → `field_times_seconds`, `incident` → `incident_E`,
  `plane` → `plane_E` (beside `incident_H`, `plane_H`), `time_unit`/`length_unit` →
  `time_units`/`length_units`; `field_units` is always a mapping.
- `diagnostics['grazing']` rows say `polarization(s)`, `stabilizing_loss` and `invariant_axes`
  (axis names), the public names; they said `polarisation(s)`, `stabilize` and `axis1d`.
- Emitted values use American spelling: `'cell_center'` and `omitted_center` in the capability
  manifest, `'cell center (0,0)'` in `rcwa.operators` metadata, `'explicit cell-centered
  operators'` in rcwa provenance.

### Error codes

Every public refusal has a code of the form `area.item` (lowercase, dotted), listed in
[keywords](../reference/keywords.md). A code that was not of that form was renamed; the most
visible ones:

| Before | After |
| --- | --- |
| `invalid_input` (the default, on about a hundred sites) | a specific code (`monitor.plane`, `numerics.duration`, `source.name_duplicate`, ...); the default is now `input.invalid` |
| fdtd `prepared_structure`; native and periodic `invalid_input` | `prepared.structure` in every solver, with the path of the declaration that changed (`scene.domain`, `stack.layers`, ...) |
| fdtd `prepared_dependencies`, `static_input`, `cuda_scope` | `prepared.dependencies`, `prepared.static_input`, `execution.cuda_scope` |
| fdtd `checkpoint_steps`, `checkpoint_scope`, `fdtd_input` | `numerics.checkpoint_steps`, `numerics.checkpoint_scope`, `fdtd.invalid` |
| fdtd `plane_span`, `monitor_extent`, `source_waveform_range`, `padding_continuation` | `source.plane_span` or `monitor.plane_span`, `monitor.extent`, `source.waveform_range`, `fdtd.padding_continuation` |
| `backend`, `device`, `precision`, `device_unavailable`, `jax_x64_disabled` | `execution.backend`, `execution.device`, `execution.precision`, `execution.device_unavailable`, `execution.jax_x64` |
| `unknown_length_unit`, `units_required` | `units.length`, `units.type` |
| `material_scalar`, `material_shape`, `material_poles`, `material_wavelength`, `fdtd_material` | `material.scalar`, `material.shape`, `material.poles`, `material.wavelength_range`, `material.fdtd` |
| rcwa `output.unavailable` | `outputs.unavailable` |

`fdtd.native` and `fdtd.periodic` `prepare` and `run` now raise a coded `ValidationError`
(`native.invalid`, `periodic.invalid`, or a specific code) where an engine refusal used to escape
as a plain `ValueError`; `check` already reported the same failure.

### `orders` is an explicit pair

`orders=` is the half-width pair `(Mu, Mv)` along the two lattice vectors in every public call:
`rcwa.Numerics`, `fdtd.periodic.Numerics`, `fdtd.DiffractionMonitor` and the `rcwa.experimental`
builders. A bare integer is refused with code `numerics.orders`, `monitor.orders` or
`experimental.orders`, and the message names the pair to write. It used to mean `(M, M)` in
`fdtd.DiffractionMonitor` and the experimental builders, `(M, 0)` in `fdtd.periodic.Numerics`,
and crashed `rcwa.Numerics` with a bare `TypeError`. Write `orders=(0, 0)` for `orders=0`.
NumPy integers are accepted; `True`/`False` are not. The experimental builders' defaults are
`(3, 3)` and `(1, 1)`, and the transmission gap cell's `closure()` metadata key `order` is
`orders`.

### Result-key job files, run records and numbers

No job field is renamed, so job keys and `wavebit.job/v2` are unchanged, with one exception: a
job that wrote `orders` as a bare integer (`fdtd.DiffractionMonitor`, `fdtd.periodic.Numerics`)
is refused with code `job.encoding`, like the declaration itself; write the pair it meant,
`(M, M)` for the monitor and `(M, 0)` for periodic. Run records stay `wavebit.run/v2`: a record
stores a result's mappings as they were, so a record saved earlier keeps its old keys and one
saved now has the new ones; a record loads only if its job does. The capability manifest is
version 26. Every number, plan skeleton and reference golden is unchanged.

## Amplitude ratios, fields and spectra

The fifth change (2026-11-08, [issue #54](https://github.com/fieldalgo/wavebit/issues/54))
makes one name hold one physical quantity in a diffraction result. Before it, `amplitude_s` was
a dimensionless ratio in `rcwa` and `fdtd`, but in `fdtd.periodic` it was that ratio times the
source amplitude, labelled `'V/m'`. `reference['incident_amplitude']` was a field in V/m in
`rcwa` and `fdtd.periodic`, but in `fdtd` a pulse spectrum in V*s/m, one value per wavelength.
Each quantity now has its own name. As before, nothing is aliased: an old amplitude name, as an
attribute or a constructor keyword, fails and names its replacement, and an old `reference` key
is refused.

### Diffraction orders

`wavebit.DiffractionOrders` and its one-order record `wavebit.observations.DiffractionOrder`:

| Before | After | Unit | Source amplitude doubled |
| --- | --- | --- | --- |
| `amplitude_s`, `amplitude_p`, `amplitude_xyz` | `amplitude_ratio_s`, `amplitude_ratio_p`, `amplitude_ratio_xyz` | `'1'` | leaves them unchanged |
| none | `field_s`, `field_p`, `field_xyz`, read-only | V/m | doubles them, phase included |
| `efficiency` | `efficiency` | `'1'` | leaves it unchanged |
| `power`, in `rcwa` only | `power`, now in `fdtd.periodic` too | W, or W/m across an invariant axis | multiplies it by four |

An amplitude ratio is an outgoing order's complex electric field at `reference['plane']` and
`reference['lateral_origin']`, in that order's own outgoing s/p frame (Cartesian, with a
trailing xyz axis, for `amplitude_ratio_xyz`), divided by the incident wave's complex field at
its own reference: the lit face, or `reference['incident_plane']` where a producer names one. For
a transmitted port this is not the port's own plane, where no incident wave runs. Like a Fresnel coefficient it describes the structure, not the drive. `field_*` is that ratio
times `reference['incident_field']`: the outgoing field in V/m. It is computed when it is read,
and it exists only where the drive is a continuous wave given in V/m (`rcwa.run`,
`fdtd.periodic.run`). On `fdtd.run`, whose drive is a pulse, and wherever the ratio itself is
None, `field_*` is None. `fdtd.periodic` gives no Cartesian ratio, so its `field_xyz` is None
too.

### Diffraction reference keys

| Before | After |
| --- | --- |
| `incident_amplitude`, `incident_amplitude_units` (`'V/m'`) in `rcwa` and `fdtd.periodic` | `incident_field`, `incident_field_units` (`'V/m'`): the drive of a continuous wave |
| `incident_amplitude`, `incident_amplitude_units` (`'V*s/m'`) in `fdtd` | `incident_spectrum`, `incident_spectrum_units` (`'V*s/m'`): the measured incident pulse spectrum, one value per wavelength |
| `amplitude_units` (`'1'`, but `'V/m'` in `fdtd.periodic`) | `amplitude_ratio_units`, always `'1'` |
| on a `RawDiffractionSpectrum`, the field `amplitude_units` and `reference['amplitude_units']` (`'V*s/m'`) | `reference['spectrum_units']` (`'V*s/m'`) |
| `fdtd.periodic`: no incident power | `incident_power` with `incident_power_units`, and `power_units`, as in `rcwa` |

`fdtd.periodic`'s `incident_power` is the incident power through one periodic cell, per
wavelength: W for a crossed cell, W/m per unit length of an invariant y. A band shares the
transverse wavevector of its centre wavelength, so each wavelength meets the cell at its own
angle, and its incident power is taken at that angle; `power = efficiency * incident_power`.

`fdtd.periodic` orders also gain two `metadata` keys, always present: `power_defined` and
`efficiency_defined`, one boolean per wavelength, which say where `power`, `efficiency` and
`incident_power` are defined. `rcwa` has keys of the same names, but they do not mark the same
wavelengths. `fdtd.periodic`'s `power` is `efficiency * incident_power`, so wherever no
incident wave propagates both flags are False and all three values are NaN. `rcwa` keeps
`power_defined` True for an evanescent incident wave and still returns its port powers there;
only its `efficiency_defined` is False (see
[Real-angle incidence](../reference/common/conventions.md#real-angle-incidence)). Their
`metadata['normalization']` string changed from `'incident-only power; unit Jones pair'` to
`'amplitudes per unit incident field (unit Jones pair); efficiency over the incident-only power'`.

A `DiffractionOrders`, `DiffractionOrder`, `RawDiffractionSpectrum` or `RawDiffractionOrder`
built with a `reference` that still holds `incident_amplitude`, `incident_amplitude_units` or
`amplitude_units` is refused with code `observation.reference_key`, and the message names the
new key. Code `observation.amplitude_units` refuses three things:
- a unit label other than the one its key allows (`amplitude_ratio_units` `'1'`,
  `spectrum_units` `'V*s/m'`);
- the other kind's label: `spectrum_units` on a calibrated record, `amplitude_ratio_units` on a
  raw spectrum;
- a drive without its label beside it, or a label without its drive: `incident_field` goes with
  `incident_field_units` `'V/m'` and `incident_spectrum` with `incident_spectrum_units` `'V*s/m'`,
  both or neither (a `None` value counts as missing).

Reading an old key from a result's `reference` raises `KeyError`, as for the
[result and report keys](#result-and-report-keys).

### Raw diffraction spectra

`fdtd.DiffractionMonitor(raw=True)` still returns a `wavebit.RawDiffractionSpectrum`, but it is
no longer a kind of `DiffractionOrders`. The two are separate classes that share the axes,
`reference`, `metadata`, `order(m, n)` and `at_wavelength_index(i)`, and
`isinstance(raw, wavebit.DiffractionOrders)` is now False. Its arrays are pulse spectra, not
ratios:

| Before | After |
| --- | --- |
| `RawDiffractionSpectrum.amplitude_s`, `.amplitude_p`, `.amplitude_xyz` | `.spectrum_s`, `.spectrum_p`, `.spectrum_xyz`, in V*s/m |
| `raw.order(m, n)`, a `DiffractionOrder` | a `wavebit.observations.RawDiffractionOrder`, with the same per-order arrays: `spectrum_*`, `spectral_energy`, `k_parallel`, `k_normal` |

A raw spectrum has no `amplitude_ratio_*`, `field_*`, `efficiency` or `power`. Its `reference`
keeps `incident_spectrum` with `incident_spectrum_units` and has no `amplitude_ratio_units`.
Inside the usable band (`metadata['source_usable']`), `raw.spectrum_s` equals, to round-off,
`orders.amplitude_ratio_s * orders.reference['incident_spectrum'][:, None, None]`, where
`orders` is the calibrated monitor with the same `plane`, `reference_plane`, `wave` and
`wavelengths`; the same holds for `p`. Outside that band the relation does not hold: the ratios
are NaN there and the raw spectra stay finite.
`fdtd.run` gives no Cartesian raw spectrum, so its `spectrum_xyz` is None, as `amplitude_xyz`
was.

### What an old amplitude name does now

The [#57 names](#shared-observation-and-rcwa-names) are refused as before, now naming the final
spelling: `DiffractionOrders.cartesian` names `amplitude_ratio_xyz`, and
`RawDiffractionSpectrum.cartesian` names `spectrum_xyz`. `RawDiffractionSpectrum.spectrum`
still names `sidedness`, not the new `spectrum_s`. The raw spectrum's former field
`amplitude_units` names `RawDiffractionSpectrum.reference`, where its unit label is now
`reference['spectrum_units']`.

```text
>>> orders.amplitude_s
AttributeError: DiffractionOrders.amplitude_s is now DiffractionOrders.amplitude_ratio_s. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> raw.amplitude_s
AttributeError: RawDiffractionSpectrum.amplitude_s is now RawDiffractionSpectrum.spectrum_s. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> raw.amplitude_units
AttributeError: RawDiffractionSpectrum.amplitude_units is now RawDiffractionSpectrum.reference; its unit label is reference['spectrum_units']. The old name was removed, not aliased; see the public-name migration guide, docs/migration/public-names.md.
>>> wavebit.DiffractionOrders(wavelengths=(1.,), orders_u=(0,), orders_v=(0,), reference={'incident_amplitude': 1.})
ValidationError: reference.incident_amplitude: reference['incident_amplitude'] is now 'incident_field' (a CW drive, V/m) or 'incident_spectrum' (a pulse, V*s/m) (fieldalgo/wavebit#54)
```

The constructors refuse the old keywords the same way: `DiffractionOrders(amplitude_s=...)`
names `amplitude_ratio_s=`, `RawDiffractionSpectrum(amplitude_s=...)` names `spectrum_s=`,
`RawDiffractionSpectrum(amplitude_units=...)` names `reference=` and its
`reference['spectrum_units']`, and `RawDiffractionSpectrum(spectrum=...)` still names
`sidedness=`.

### Amplitude-ratio job files and run records

No declaration or keyword is renamed (a source's `amplitude=` keeps its name), so jobs, their
keys and `wavebit.job/v2` are unchanged. Run records stay `wavebit.run/v2`. A record saved from
now on stores the new names: `amplitude_ratio_s`, `amplitude_ratio_p` and `amplitude_ratio_xyz`
(on a raw spectrum `spectrum_s`, `spectrum_p` and `spectrum_xyz`), and in `reference`
`incident_field` or `incident_spectrum`, `amplitude_ratio_units` and `spectrum_units`. `field_*`
is a property, not a stored field, so a record does not hold it: multiply a stored
`amplitude_ratio_*` by the stored `reference['incident_field']` to recover it. A record saved
earlier keeps the names it was saved with (`amplitude_s`, `incident_amplitude`, ...), since
`load_run` returns the stored fields as `RecordedObject` mappings rather than rebuilding the
classes. The `amplitude_s` of an earlier `fdtd.periodic` record is the ratio times the source
amplitude: divide it by the record's `reference['incident_amplitude']` to recover the ratio. A
record saved with amplitude 0 holds no ratio. The capability manifest is version 27.

### Amplitude-ratio numbers

- `rcwa`: no number changes.
- `fdtd.periodic`: the amplitude ratios, those in `result.basis` included, are no longer
  multiplied by the source amplitude, so they equal the earlier `amplitude_s` and `amplitude_p`
  only where that amplitude is 1, the default. Efficiencies, `R`, `T`, the planes, the
  snapshots, `sample()` and `plane_power_z` keep their values, with one exception. Under
  `algorithm='reference'`, a wavelength where no incident wave propagates (the band's shared
  transverse wavevector is past that wavelength's light line) now has NaN `incident_power`,
  `power` and `efficiency`, so NaN `R` and `T`, with `metadata['power_defined']` and
  `metadata['efficiency_defined']` False there. Shared stepping refuses such a band, as before.
- `fdtd`: the ratios, efficiencies and raw spectra are unchanged. At oblique Bloch incidence
  `reference['incident_spectrum']` is referred to the cell centre, `reference['lateral_origin']`,
  as the ratios and the raw spectra already were. The earlier `incident_amplitude` was referred to
  the cell's low corner: the new `incident_spectrum` is the earlier `incident_amplitude` times
  `exp(i*(k_u*L_u + k_v*L_v)/2)`, a phase factor of modulus one, where `L_u`, `L_v` are the
  cell's lengths and `k_u`, `k_v` the incident transverse wavevector along them. To compare a
  stored `incident_amplitude` with a new `incident_spectrum`, multiply it by that factor. At
  normal incidence they are equal.

No reference golden was regenerated and no tolerance changed.
