# FDTD domains, sources and observations


Use `from wavebit import fdtd` for the public time-domain module. `fdtd.Problem` combines a physical `Scene`, named sources, requested `Outputs`, and explicit `Units`. Grid spacing, duration, CFL and the PML belong to `fdtd.Numerics`.

```python
from wavebit import Material, Units, fdtd

problem = fdtd.Problem(
    units=Units(length="um"),
    scene=fdtd.Scene(
        domain=fdtd.Domain(size=(.1, .1, 1.6)),
        boundaries=fdtd.Boundaries(x=fdtd.Periodic(), y=fdtd.Periodic(), z=fdtd.Open()),
        structures=(fdtd.Structure("film", shape=fdtd.SnappedSlab(z=(-.1, .1)),
                                   material=Material(epsilon=2.25)),),
    ),
    sources=(fdtd.PlaneWave("illumination", plane=fdtd.Plane(z=.5),
        pulse=fdtd.GaussianPulse(wavelength=.8, frequency_sigma_hz=2e13)),),
    outputs=fdtd.Outputs(monitors=(
        fdtd.TimeSeriesMonitor("probe", at=(0., 0., -.3), component="Ey"),)),
)
numerics = fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.025),
    duration=fdtd.Duration(periods=20, source="illumination"), absorber=fdtd.PML(cells=12))
report = fdtd.check(problem, numerics=numerics)
assert report.ok
result = fdtd.run(problem, numerics=numerics)
print("samples:", len(result["probe"].times_seconds), "time unit: s; units:", result["probe"].units)
```

## Physical coordinates and units

Domain coordinates and source wavelengths use the problem's declared length unit; angles are radians. A domain's size and center describe its physical region. The PML is a numerical extension outside it, so adding PML cells does not shrink or redefine that region. `Open` denotes an open physical boundary; `PML` controls the numerical absorber separately.

Times and pulse delays (`delay_seconds`) are seconds. `GaussianPulse.frequency_sigma_hz` is in Hz. Plane-wave amplitude is electric field in V/m. `fdtd.CurrentDipole(current_moment=...)` specifies a **current moment**: A·m for electric current, or V·m for its magnetic equivalent. It is different from RCWA `ElectricDipole`, whose electric dipole moment is in C·m; do not copy the same numerical amplitude between them.

Shared material rates are SI quantities. The FDTD adapter converts them to the internal normalized system and converts public observations back to physical units. Unsupported tensor, loss, source, boundary or observation combinations raise a structured error.

## Requested observations

Named monitors determine what is accumulated. `TimeSeriesMonitor` returns a `TimeSeries` with seconds and field units; E is V/m and H is A/m. Every field observation's `units` maps each component present (`'Ex'`..`'Hz'`) to its unit and `'coordinates'` to the problem's length unit. Electric and magnetic samples retain their Yee time staggering. The spectral `FieldMonitor`, `FluxMonitor`, `DiffractionMonitor` and `FarFieldMonitor` have different normalization and validity requirements; use the [reference](../../reference/fdtd/index.md) and inspect returned metadata. A raw diffracted spectrum does not automatically certify normalized R/T or complete entering power.

Use `check` to inspect resolved settings and restrictions, `prepare` to establish a reusable discrete contract, and `prepared.run(problem)` for each complete live input. `check` also lists warnings that leave `report.ok` true: a cell-snapped `SnappedSlab` or `SnappedBox` whose interfaces fall between cell boundaries is reported as `geometry.cell_snapped`, with the requested span in `expected` and the cell boundaries it runs at in `actual`; [how the solver models a problem](model.md) explains snapping and the smoothed alternatives. Earlier results retain their own observations. Changing discrete source/monitor layout, grid topology or incompatible time stepping needs a new preparation.

## Validation and earlier interfaces

The [FDTD curriculum](../../learn/fdtd/index.md) links source normalization, absorber behavior,
refinement, material and gradient checks. Use the physical [periodic-stack](periodic-problems.md)
and [native-grid](native-grids.md) interfaces for those formulations. The
[earlier general-engine manual](../../migration/general-fdtd-v1.md) records normalized
implementation details and historical validated limitations.
