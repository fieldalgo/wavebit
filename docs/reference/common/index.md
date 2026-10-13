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

.. autoclass:: wavebit.Units
   :members:

.. autoclass:: wavebit.Lattice
   :members:

.. autoclass:: wavebit.Execution
   :members:

.. autoclass:: wavebit.CheckReport
   :members:

.. autoclass:: wavebit.Job
   :members:

.. autoclass:: wavebit.DiffractionOrders
   :members:

.. autoclass:: wavebit.FieldSamples
   :members:

.. autoclass:: wavebit.SpectralFieldSamples
   :members:

.. autoclass:: wavebit.RawDiffractionSpectrum
   :members:

.. autoclass:: wavebit.TimeSeries
   :members:

.. autoclass:: wavebit.Validity
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
