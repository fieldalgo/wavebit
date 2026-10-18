# FDTD API

See the [runnable first calculation](../../getting-started/fdtd.md) and [common lifecycle](../../guide/common/lifecycle.md).

```{eval-rst}
.. autofunction:: wavebit.fdtd.check

.. autofunction:: wavebit.fdtd.prepare

.. autofunction:: wavebit.fdtd.run

.. autoclass:: wavebit.fdtd.Problem
   :members:

.. autoclass:: wavebit.fdtd.Scene
   :members:

.. autoclass:: wavebit.fdtd.Domain
   :members:

.. autoclass:: wavebit.fdtd.Slab
   :members:

.. autoclass:: wavebit.fdtd.Region
   :members:

.. autoclass:: wavebit.fdtd.Plane
   :members:

.. autoclass:: wavebit.fdtd.Sphere
   :members:

.. autoclass:: wavebit.fdtd.Box
   :members:

.. autoclass:: wavebit.fdtd.Cylinder
   :members:

.. autoclass:: wavebit.fdtd.Prism
   :members:

.. autoclass:: wavebit.fdtd.Structure
   :members:

.. autoclass:: wavebit.fdtd.SwitchedMaterial
   :members:

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

.. autoclass:: wavebit.fdtd.Symmetry
   :members:

.. autoclass:: wavebit.fdtd.Axis1D
   :members:

.. autoclass:: wavebit.fdtd.GaussianPulse
   :members:

.. autoclass:: wavebit.fdtd.PlaneWave
   :members:

.. autoclass:: wavebit.fdtd.Dipole
   :members:

.. autoclass:: wavebit.fdtd.TFSF
   :members:

.. autoclass:: wavebit.fdtd.InitialField
   :members:

.. autoclass:: wavebit.fdtd.Record
   :members:

.. autoclass:: wavebit.fdtd.Field
   :members:

.. autoclass:: wavebit.fdtd.Flux
   :members:

.. autoclass:: wavebit.fdtd.Diffraction
   :members:

.. autoclass:: wavebit.fdtd.FarField
   :members:

.. autoclass:: wavebit.fdtd.Outputs
   :members:

.. autoclass:: wavebit.fdtd.UniformGrid
   :members:

.. autoclass:: wavebit.fdtd.Duration
   :members:

.. autoclass:: wavebit.fdtd.PML
   :members:

.. autoclass:: wavebit.fdtd.Absorber
   :members:

.. autoclass:: wavebit.fdtd.Decay
   :members:

.. autoclass:: wavebit.fdtd.Numerics
   :members:

.. autoclass:: wavebit.fdtd.Prepared
   :members:

.. autoclass:: wavebit.fdtd.Result
   :members:

.. autoclass:: wavebit.fdtd.SpectralFlux
   :members:

.. autoclass:: wavebit.fdtd.FarFieldSpectrum
   :members:

```

## Native-grid and periodic-stack formulations

See [native-grid units and observations](../../guide/fdtd/native-grids.md) and
[periodic-stack units and observations](../../guide/fdtd/periodic-problems.md).

```{eval-rst}
.. automodule:: wavebit.fdtd.native
   :members:

.. automodule:: wavebit.fdtd.periodic
   :members: Problem, RasterDispersion, Pulse, Outputs, Record, Numerics, Prepared, Result, check, prepare, run

.. automodule:: wavebit.observations.modes
   :members:
```
