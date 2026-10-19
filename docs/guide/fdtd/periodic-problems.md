# Physical periodic-stack FDTD

`from wavebit.fdtd import periodic` provides immutable `Problem`, `Numerics`,
`check`, `prepare` and `run`. Portable jobs use `solver='fdtd.periodic'`. Names
changed on 2026-10-19 are listed in the
[public-name migration guide](../../migration/public-names.md#native-grid-and-periodic-stack-names).
Stack, Layer, Region, typed shapes, Excitation and PlaneWave are the same
physical declarations as RCWA. Layers run bottom to top, z=0 is the bottom
interface and x=y=0 is the cell centre.

```python
from wavebit import Units
from wavebit.fdtd import periodic as fdtd
import numpy as np

units = Units(length='um')
problem = fdtd.Problem(
    units=units,
    stack=fdtd.Stack(period=(.2,), layers=(
        fdtd.Layer(thickness=.1, epsilon=np.full((1,2), 2.25)),)),
    excitation=fdtd.Excitation(wavelength=(.85,.9,.95),
                               sources=(fdtd.PlaneWave(),)))
numerics = fdtd.Numerics(spacing=1/30,
                         duration_seconds=90*units.time_unit_seconds)
result = fdtd.run(problem, numerics=numerics)
print(result.R.sum(axis=(-2,-1)), result.T.sum(axis=(-2,-1)))
```

Spatial quantities use the declared length unit. Time and material rates use SI.
`duration_seconds` rounds up to complete time steps. `Pulse(frequency_sigma_hz=...)`
and `notch_frequencies_hz` control the physical Gaussian source: sigma is the
standard deviation of the amplitude spectrum in Hz, as in `fdtd.GaussianPulse`, and
the default envelope's 1/e half-width is one period at the centre wavelength
(sigma = f0/(sqrt(2)·pi)). A band shares the transverse Bloch wavevector selected
at its centre wavelength, so incidence angles across the band generally differ.
`diagnostics['angles']` reports them in radians. Theta/phi are supported; Cartesian
kx/ky incidence is refused.

Choose the numerical algorithm explicitly when reproducing earlier results:

| Algorithm | Execution and purpose |
| --- | --- |
| `shared` (default) | Common Yee engine with NumPy/JAX/Torch, fixed nominal meshes and material/shape/angle derivatives |
| `reference` | Independent NumPy formulation retaining the earlier periodic calculation, graded meshes and transient records |

Selecting NumPy as a backend does not change the algorithm. The reference route
refuses shared-only controls; it does not silently discard them. Its check report
states that it validates the grid and materials; monitor extraction is checked
when running. Shared checks execute full grid admission without advancing time.
Both currently require double precision. Configure JAX x64 before arrays.

`spacing` sets the nominal z grid; `transverse_spacing` optionally sets x/y
separately. `nominal_layer_thickness` fixes the grid while live layer thicknesses
change. `variable_depth=True` admits changing total depth with fixed laboratory
coordinates and live interface phases. `nominal_indices` fixes padding/grading
while half-space materials change. Prepared execution requires the same declared
layout and full live input every time. Positive interface/shape smoothing is a
numerical averaging width whose bias must be refined separately from grid error.
The existing shared-engine shape restrictions remain enforced: separable
rectangles or area-integrated rectangles and individual circles/ellipses, with
supported painting/topology constraints. See the engine study guides for their
measured accuracy limits; selecting a typed declaration does not broaden them.

Region/port materials are nondispersive `Material` declarations. Raster epsilon
is epsilon-infinity; `RasterDispersion(labels=..., poles={label: (Drude(...),
Lorentz(...))})` supplies physical electric poles. Label zero is nondispersive.
Unsupported causal models or region-dispersion requests are refused.

Reflected/transmitted `DiffractionOrders` always include a leading wavelength
axis. Efficiencies are normalized by incident-only power; complex amplitudes
are in V/m, referenced at the physical interface and cell centre in each order's
s/p frame. `R` and `T` are convenience views of these efficiencies. Available s/p
basis observations are retained separately in `result.basis`.

`Outputs(planes={'name': z})` streams calibrated phasors. E is V/m and H is A/m;
plane metadata records its calibration. Shared stepping calibrates unit incident
E at the interface, while reference stepping retains its soft-source-spectrum
calibration. `plane_power_z` includes the physical integration measure: W for a
crossed cell, **W/m** for an invariant-y problem. The unit is explicit on each
plane. Positive power points upward. Dividing a sample by a separate reference
run is often the appropriate comparison, as in the
[frequency-plane example](../../../examples/fdtd/fdtd_frequency_planes.py).

`Outputs(snapshots=Snapshots(every=..., components=..., bounds=...))` keeps native
Yee snapshots of the components `'Ex'` to `'Hz'`, including each component's
coordinates and E/H half-step timestamps in seconds; `result.snapshots` is keyed by
component. `bounds` maps `'x'`, `'y'` or `'z'` to a physical `(low, high)` pair in the
length unit and requires shared stepping. Shared snapshots use a constant carrier
calibration to retain a causal waveform, distinct from frequency-by-frequency
plane calibration. `result.sample(x=..., y=..., z=..., times_seconds=...)` interpolates
all six components offline at those SI times, so it needs the default `components`; it
refuses source-discontinuity crossings and does not retain AD. Instantaneous
Poynting is `cross(E.real, H.real)`, without the phasor factor 1/2. See the
[snapshot](../../../examples/fdtd/fdtd_record.py) and
[Poynting](../../../examples/fdtd/fdtd_poynting.py) examples.

`Numerics(decay_check=DecayCheck(window_seconds=..., threshold=...))` diagnoses
decay and scattering changes over the final window without terminating early. The
window is in seconds and is rounded up to whole time steps by the same rule as
`native.DecayCheck`: `40*dt_seconds`, with `dt_seconds` from
`check(...).resolved['time_step_seconds']`, covers 40 steps. Diagnostics retain
basis-specific changes and the scattering bound, with SI times and explicit
raw-spectrum units. These checks do not establish grid, smoothing or PML
convergence. `observations.fit_modes` accepts SI timestamps for offline ringdown
fitting and returns Hz and decay rates in 1/s; window and model-order convergence
remain separate checks.

## Memory under an outer `jax.jit` on CPU

Wavebit has no option for this. It is a JAX compiler setting, and the program it affects is
your own outer `jax.jit`, so you choose it there.

It matters when one layer holds two or more rectangles with `shape_integration='area'`, the
setting every rotated rectangle needs because `'separable'` refuses rotation. In that case a forward-mode derivative
(`jax.jvp`, `jax.jacfwd`) compiled on CPU can need several times the memory of the same
derivative computed eagerly. XLA:CPU's default scheduler releases none of the shape-painting
intermediates until the last one has been created, so all of them are alive at once. The memory-optimized scheduler computes the same
values with a much smaller working set:

```python
options = {'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'}
jacobian = jax.jit(jax.jacfwd(loss), compiler_options=options)
```

To apply it to every jit in a process, add
`--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED` to `XLA_FLAGS` before JAX
initializes its backend, keeping any flags already there.

A layer with a single rotated rectangle and the `'separable'` route are not affected. Reverse
mode (`jax.grad`) was affected much less in the one case measured. The measurements are in
[outer-jit memory of area-painted FDTD derivatives](../../validation/fdtd/fdtd-jit-painting-memory.md).
