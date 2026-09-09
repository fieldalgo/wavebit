# API reference

Generated from the docstrings, so it cannot drift from the code. What is *not* here is
deliberate: `wavebit._core` is private, and its module-level notes explain the implementation
to a contributor rather than the interface to a caller.

## Building a solver

```{eval-rst}
.. autofunction:: wavebit.rcwa

```


## The solver object

`rcwa()` returns a solver whose class is chosen by the backend. These are the methods and
properties every one of them carries.

```{eval-rst}
.. autoclass:: wavebit._core.engine.CoreFactorizationMixin
   :members: solve, reconstruct_field, freeze_geometry, solve_conv, eps_conv_matrices, factorization, fourier_coefficients, normal_method, projector, normal_grid


.. autoclass:: wavebit._core.engine.CoreSolveMixin
   :members: answers, excitation, scattering, truncation_order, period, lattice


```

## Describing a scene

```{eval-rst}
.. autoclass:: wavebit.Material
   :members:

.. autofunction:: wavebit.eps_from_index

.. autofunction:: wavebit.index_from_eps

.. autoclass:: wavebit.Lattice
   :members:

.. autofunction:: wavebit.meshgrid_xy

.. autoclass:: wavebit.Layer
   :members:

.. autoclass:: wavebit.Stack
   :members:

.. autoclass:: wavebit.Incidence
   :members:

```


## A run as data

```{eval-rst}
.. autoclass:: wavebit.JobSpec
   :members:

.. autoclass:: wavebit.Numerics
   :members:

.. autoclass:: wavebit.Sweep
   :members:

.. autofunction:: wavebit.run

.. autoclass:: wavebit.SpecError
   :members:

```


## Results and state

```{eval-rst}
.. autoclass:: wavebit.ScatteringMatrix
   :members:

.. autoclass:: wavebit.Port
   :members:

.. autoclass:: wavebit.StaticGeometry
   :members:

.. autoclass:: wavebit.StaleGeometryError
   :members:

.. autoclass:: wavebit.ConvMatrices
   :members:

.. autoclass:: wavebit.ConformingNVF
   :members:

```


## Materials data

```{eval-rst}
.. autofunction:: wavebit.find_material

.. autofunction:: wavebit.find_nk_index

.. autofunction:: wavebit.read_material_table

```


## Plotting

```{eval-rst}
.. autofunction:: wavebit.plot_structure

.. autofunction:: wavebit.slice_structure

```


## Precision, and what warns

```{eval-rst}
.. autofunction:: wavebit.enable_x64

.. autofunction:: wavebit.branch_tolerance

.. autoclass:: wavebit.UnphysicalResultWarning
   :members:

.. autoclass:: wavebit.PrecisionLossWarning
   :members:

.. autoclass:: wavebit.UnresolvedBranchWarning
   :members:

.. autoclass:: wavebit.RayleighPointWarning
   :members:

.. autoclass:: wavebit.AutoDetuneWarning
   :members:

.. autoclass:: wavebit.AxisNameWarning
   :members:

```
