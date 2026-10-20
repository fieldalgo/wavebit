# First FDTD calculation

An FDTD problem describes a physical domain, its structures, sources and requested observations. The grid, duration, CFL factor and absorbing layer belong to `fdtd.Numerics`. Both solvers use shared `Material` definitions and explicit `Units`.

This small pulse calculation records Ey below a dielectric film. Its small transverse domain is periodic in x and y; the z faces are open. The PML is added **outside** the physical domain. This is a quick execution example, not a calibrated spectrum or a refinement study.

## 1. Define the domain and run

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
```

## 2. Read the recorded signal

```python
print("samples:", len(result["probe"].times_seconds), "time unit: s; units:", result["probe"].units)
```


The monitor's time axis is in seconds, and Ey is in V/m; an H component is in A/m. `units` maps the recorded component to its unit and `'coordinates'` to the problem's length unit. `TimeSeriesMonitor` samples the nearest Yee point for its selected component; it is not spatial interpolation at an arbitrary position.

## Optional: reuse a prepared calculation

For repeated compatible runs, use `prepared = fdtd.prepare(problem, numerics=numerics)` and `prepared.run(problem)`. Every run receives the complete current problem. Changing grid topology, duration structure, monitor layout or other prepared constraints requires a new preparation; rejected reuse raises a structured incompatibility error.

See [the FDTD guide](../guide/fdtd/general-fdtd.md) for units, supported observations and validation limits, [how the solver models a problem](../guide/fdtd/model.md) for the grid, cell-snapped and smoothed structures and the accuracy rules a run must pass, and [the FDTD curriculum](../learn/fdtd/index.md) for refinement and physical validation.

## Continue with dispersion and calibrated observations

The [dispersive field example](https://github.com/fieldalgo/wavebit/blob/main/examples/fdtd/dispersive_fields.py)
adds a physical Lorentz material, a time-series monitor, spectral E/H and a separate reference run.
It keeps raw Fourier field units distinct from source-normalized fields and checks live
source-amplitude reuse. Plotting is optional. The [film and grating comparison](https://github.com/fieldalgo/wavebit/blob/main/examples/cross_solver/film_and_grating.py)
then aligns RCWA and FDTD reference planes and compares complex nonzero diffraction orders
under spatial and time-window refinement. Agreement in these declared cases does not
establish convergence for a different material, geometry or pulse.
