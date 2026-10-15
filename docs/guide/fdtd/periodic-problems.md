# Physical periodic-stack FDTD

`from wavebit.fdtd import periodic` provides immutable `Problem`, `Numerics`,
`check`, `prepare` and `run`. Portable jobs use `solver='fdtd.periodic'`.
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
                         duration_seconds=90*units.time_seconds)
result = fdtd.run(problem, numerics=numerics)
print(result.R.sum(axis=(-2,-1)), result.T.sum(axis=(-2,-1)))
```

Spatial quantities use the declared length unit. Time and material rates use SI.
`duration_seconds` rounds up to complete time steps. `Pulse.width_seconds` and
`notch_frequencies_hz` control the physical Gaussian source. A band shares the
transverse Bloch wavevector selected at its centre wavelength, so incidence
angles across the band generally differ. `diagnostics['angles']` reports them
in radians. Theta/phi are supported; Cartesian kx/ky incidence is refused.

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
is epsilon-infinity; `RasterDispersion(labels=..., models={label: (Drude(...),
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

`Record(every=..., components=..., bounds=...)` retains native Yee snapshots,
including each component's coordinates and E/H half-step timestamps in seconds.
Physical bounds require shared stepping. Shared records use a constant carrier
calibration to retain a causal waveform, distinct from frequency-by-frequency
plane calibration. `result.sample(x=..., y=..., z=..., times=...)` interpolates all
six recorded fields offline using SI times; it refuses source-discontinuity
crossings and does not retain AD. Instantaneous Poynting is
`cross(E.real, H.real)`, without the phasor factor 1/2. See the
[field record](../../../examples/fdtd/fdtd_record.py) and
[Poynting example](../../../examples/fdtd/fdtd_poynting.py).

`tail_steps` and `tail_tolerance` diagnose decay and scattering changes without
terminating early. Diagnostics retain basis-specific changes and the scattering
bound, with SI times and explicit raw-spectrum units. These checks do not
establish grid, smoothing or PML convergence. `observations.fit_modes` accepts
SI timestamps for offline ringdown fitting and returns Hz and decay rates in
1/s; window and model-order convergence remain separate checks.
