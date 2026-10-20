# Shared API

Physical materials and units are shared by both solvers. A common declaration does not widen a solver's supported numerical scope; `check` reports that scope explicitly.

```{eval-rst}
.. autoclass:: wavebit.Material
   :members:

.. autoclass:: wavebit.MaterialResponse
   :members:

.. autoclass:: wavebit.ConductiveMaterial
   :members:

.. autoclass:: wavebit.DispersiveMaterial
   :members:

.. autoclass:: wavebit.Drude
   :members:

.. autoclass:: wavebit.Lorentz
   :members:

.. autoclass:: wavebit.Debye
   :members:

.. autoclass:: wavebit.TabulatedMaterial
   :members:

.. autofunction:: wavebit.eps_from_refractive_index

.. autofunction:: wavebit.refractive_index_from_eps

.. autofunction:: wavebit.layer_thicknesses

.. autoclass:: wavebit.Units
   :members:

.. autoclass:: wavebit.Lattice
   :members:

.. autoclass:: wavebit.Execution
   :members:

.. autoclass:: wavebit.Backend
   :members:

.. autoclass:: wavebit.CheckReport
   :members:

.. autoclass:: wavebit.Job
   :members:

.. autofunction:: wavebit.run

.. autoclass:: wavebit.DiffractionOrders
   :members:

.. autoclass:: wavebit.FieldSamples
   :members:

.. autoclass:: wavebit.RawFieldSpectrum
   :members:

.. autoclass:: wavebit.RawDiffractionSpectrum
   :members:

.. autoclass:: wavebit.TimeSeries
   :members:

.. autoclass:: wavebit.Validity
   :members:

.. autoclass:: wavebit.WavebitError
   :members:

.. autoclass:: wavebit.ValidationError
   :members:

.. autoclass:: wavebit.UnsupportedFeatureError
   :members:

.. autoclass:: wavebit.IncompatibleProblemError
   :members:

.. autoclass:: wavebit.NumericalError
   :members:

.. autoclass:: wavebit.Issue
   :members:

```

## Measured run records

See [saving and inspecting research runs](../../guide/run-records.md) for the data-only
load contract, explicit field sampling and source-provenance scope.

```{eval-rst}
.. autofunction:: wavebit.records.run_recorded

.. autofunction:: wavebit.records.save_run

.. autofunction:: wavebit.records.load_run

.. autoclass:: wavebit.records.RunRecord
   :members:

.. autoclass:: wavebit.records.RecordedObject
   :members:
```
