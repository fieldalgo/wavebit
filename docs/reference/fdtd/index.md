# FDTD API

See the [runnable first calculation](../../getting-started/fdtd.md) and [common lifecycle](../../guide/common/lifecycle.md).
Names changed on 2026-10-19 are listed in the [public-name migration guide](../../migration/public-names.md).

## Lifecycle

Declare a `Problem`, check it, prepare it once and run it.

```{eval-rst}
.. autofunction:: wavebit.fdtd.check

.. autofunction:: wavebit.fdtd.prepare

.. autofunction:: wavebit.fdtd.run

.. autoclass:: wavebit.fdtd.Problem
   :members:

.. autoclass:: wavebit.fdtd.Prepared
   :members:
```

## Scene and geometry

The physical domain and the structures in it. `Sphere`, `Box`, `Cylinder` and `Prism` are smoothed bodies; `SnappedSlab` and `SnappedBox` fill whole Yee cells.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Scene
   :members:

.. autoclass:: wavebit.fdtd.Domain
   :members:

.. autoclass:: wavebit.fdtd.Structure
   :members:

.. autoclass:: wavebit.fdtd.Sphere
   :members:

.. autoclass:: wavebit.fdtd.Box
   :members:

.. autoclass:: wavebit.fdtd.Cylinder
   :members:

.. autoclass:: wavebit.fdtd.Prism
   :members:

.. autoclass:: wavebit.fdtd.SnappedSlab
   :members:

.. autoclass:: wavebit.fdtd.SnappedBox
   :members:

.. autoclass:: wavebit.fdtd.Plane
   :members:

.. autoclass:: wavebit.fdtd.Shape
   :members:
```

## Boundaries

One physical condition per face; the absorber that implements `Open` is a numerical choice.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Boundaries
   :members:

.. autoclass:: wavebit.fdtd.Open
   :members:

.. autoclass:: wavebit.fdtd.Periodic
   :members:

.. autoclass:: wavebit.fdtd.Bloch
   :members:

.. autoclass:: wavebit.fdtd.PEC
   :members:

.. autoclass:: wavebit.fdtd.PMC
   :members:

.. autoclass:: wavebit.fdtd.MirrorSymmetry
   :members:

.. autoclass:: wavebit.fdtd.InvariantAxis
   :members:
```

## Sources

Radians, seconds and SI amplitudes.

```{eval-rst}
.. autoclass:: wavebit.fdtd.GaussianPulse
   :members:

.. autoclass:: wavebit.fdtd.PlaneWave
   :members:

.. autoclass:: wavebit.fdtd.CurrentDipole
   :members:

.. autoclass:: wavebit.fdtd.TFSF
   :members:

.. autoclass:: wavebit.fdtd.InitialField
   :members:
```

## Monitors

Each monitor is named for what it records.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Outputs
   :members:

.. autoclass:: wavebit.fdtd.TimeSeriesMonitor
   :members:

.. autoclass:: wavebit.fdtd.FieldMonitor
   :members:

.. autoclass:: wavebit.fdtd.FluxMonitor
   :members:

.. autoclass:: wavebit.fdtd.DiffractionMonitor
   :members:

.. autoclass:: wavebit.fdtd.FarFieldMonitor
   :members:
```

## Numerics

Grid, duration, absorber and stopping rule: choices that do not change the physical problem.

```{eval-rst}
.. autoclass:: wavebit.fdtd.Numerics
   :members:

.. autoclass:: wavebit.fdtd.UniformGrid
   :members:

.. autoclass:: wavebit.fdtd.Duration
   :members:

.. autoclass:: wavebit.fdtd.PML
   :members:

.. autoclass:: wavebit.fdtd.ConductiveAbsorber
   :members:

.. autoclass:: wavebit.fdtd.DecayStop
   :members:
```

## Results

```{eval-rst}
.. autoclass:: wavebit.fdtd.Result
   :members:

.. autoclass:: wavebit.fdtd.SpectralFlux
   :members:

.. autoclass:: wavebit.fdtd.FarFieldSpectrum
   :members:
```

## Waveguide modes

Modes of the discrete Yee system on a plane across a uniform cross-section; see the
[waveguide-mode guide](../../guide/fdtd/waveguide-modes.md).

```{eval-rst}
.. autofunction:: wavebit.fdtd.solve_modes

.. autoclass:: wavebit.fdtd.WaveguideModes
   :members:

.. autoclass:: wavebit.fdtd.WaveguideMode
   :members:

.. autoclass:: wavebit.fdtd.ModeMonitor
   :members:

.. autoclass:: wavebit.fdtd.ModeSpectrum
   :members:

.. autoclass:: wavebit.fdtd.ModeSource
   :members:
```
## Native-grid and periodic-stack formulations

See [native-grid units and observations](../../guide/fdtd/native-grids.md) and
[periodic-stack units and observations](../../guide/fdtd/periodic-problems.md). Their
declarations were renamed on 2026-10-19 too (`native.Record` and `periodic.Record` are now
`Snapshots`, `native.Tail` is `native.DecayCheck`, among others); see the
[migration guide](../../migration/public-names.md#native-grid-and-periodic-stack-names).

```{eval-rst}
.. automodule:: wavebit.fdtd.native
   :members:

.. automodule:: wavebit.fdtd.periodic
   :members: Problem, RasterDispersion, Pulse, Outputs, Snapshots, Numerics, DecayCheck, Prepared, Result, check, prepare, run

.. automodule:: wavebit.observations.modes
   :members:
```
