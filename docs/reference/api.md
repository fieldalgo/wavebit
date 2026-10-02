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
   :members: solve, reconstruct_field, freeze_geometry, solve_conv, eps_conv_matrices, mu_conv_matrices, factorization, fourier_coefficients, raster_normal_method, nvf_projector, li_nesting, shape_nvf_grid_size


.. autoclass:: wavebit._core.engine.CoreSolveMixin
   :members: answers, theta, kx, ky, excitation, scattering, truncation_order, period, lattice


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


## Experimental auxiliary NVF

These explicit NumPy interfaces do not change ordinary NVF routing or
certify optical convergence. See the [selection guide](../guide/nvf-method-selection.md).

```{eval-rst}
.. autoclass:: wavebit.CartesianPairOpeningPlan
   :members:

.. autoclass:: wavebit.CartesianPairTransmissionGalerkinGapCell
   :members: closure
   :inherited-members:

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

.. autoclass:: wavebit.TensorConvMatrices
   :members:

.. autoclass:: wavebit.ConformingNVF
   :members:

.. autoclass:: wavebit.AffineProfileNVF
   :members:

.. autoclass:: wavebit.CartesianRectangleNVF
   :members:

.. autoclass:: wavebit.CartesianRectanglePairNVF
   :members:

```


## Materials data

See [Materials and wavelength sweeps](../guide/materials.md) for assignment and evaluation.

```{eval-rst}
.. autoclass:: wavebit.TabulatedMaterial
   :members:

.. autofunction:: wavebit.materials.load

.. autofunction:: wavebit.materials.available

```


## Plotting

```{eval-rst}
.. autofunction:: wavebit.plot_structure

.. autofunction:: wavebit.slice_structure

```


## Precision, and what warns

```{eval-rst}
.. autofunction:: wavebit.enable_x64

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

.. autoclass:: wavebit.ComplexIncidenceWarning
   :members:

```

## Side-resolved coherent illumination

See [the guide and flowchart](../coherent-illumination.md) for physical conventions.

```{eval-rst}
.. autoclass:: wavebit.Source
   :members:

.. autoclass:: wavebit.PlaneWave
   :members:

.. autoclass:: wavebit.OrderExcitation
   :members:

.. autoclass:: wavebit.GaussianBeam
   :members:

.. autoclass:: wavebit.ElectricDipole
   :members:

.. autoclass:: wavebit.CoherentSolver
   :members: solve, reconstruct_field

.. autoclass:: wavebit.CoherentResult
   :members: boundary_fields, boundary_power, reconstruct_field, output_top, output_bottom, absorption

.. autoclass:: wavebit.BoundaryPower
   :members:
```
