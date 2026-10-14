# API reference

:::{note}
This is the complete compatibility reference for the earlier interface. Use `wavebit.compat.legacy` explicitly. Start new applications with the [current lifecycle](../guide/common/lifecycle.md) and [current API reference](../reference/api.md). Numerical evidence quoted here retains its original scope.
:::

Generated from the docstrings, so it cannot drift from the code. What is *not* here is
deliberate: `wavebit._common` is private, and its module-level notes explain the implementation
to a contributor rather than the interface to a caller.

## Building a solver

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.rcwa

```

`fdtd()` is the other door. It hands back the answer rather than a solver, because there is no
per-instance state to keep -- no eigen-decomposition to cache, no frozen geometry, no gradient.

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.fdtd

```


## The solver object

`rcwa()` returns a solver whose class is chosen by the backend. These are the methods and
properties every one of them carries.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.engine.CoreFactorizationMixin
   :no-index:
   :members: solve, reconstruct_field, freeze_geometry, solve_conv, eps_conv_matrices, mu_conv_matrices, gain_cutoff_band, factorization, fourier_coefficients, raster_normal_method, nvf_projector, li_nesting, shape_nvf_grid_size


.. autoclass:: wavebit.rcwa._engine.engine.CoreSolveMixin
   :no-index:
   :members: answers, theta, kx, ky, excitation, scattering, truncation_order, period, lattice


```

## Describing a scene

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.Material
   :no-index:
   :members:

.. autofunction:: wavebit.compat.legacy.eps_from_refractive_index

.. autofunction:: wavebit.compat.legacy.refractive_index_from_eps

.. autofunction:: wavebit.compat.legacy.layer_thicknesses

.. autofunction:: wavebit.compat.legacy.permittivity_at

.. autoclass:: wavebit.compat.legacy.Lattice
   :no-index:
   :members:

.. autofunction:: wavebit.compat.legacy.meshgrid_xy

.. autoclass:: wavebit.compat.legacy.Layer
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.Stack
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.Incidence
   :no-index:
   :members:

```


## A run as data

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.JobSpec
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.Numerics
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.Sweep
   :no-index:
   :members:

.. autofunction:: wavebit.compat.legacy.run

.. autoclass:: wavebit.compat.legacy.SpecError
   :no-index:
   :members:

```


## Experimental auxiliary NVF

These explicit NumPy interfaces do not change ordinary NVF routing or
certify optical convergence. See the [selection guide](../guide/rcwa/nvf-method-selection.md).

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.CartesianPairOpeningPlan
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.CartesianPairTransmissionGalerkinGapCell
   :no-index:
   :members: closure
   :inherited-members:

```

## Results and state

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.ScatteringMatrix
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.Port
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.StaticGeometry
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.StaleGeometryError
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.ConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.TensorConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.ConformingNVF
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.AffineProfileNVF
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.CartesianRectangleNVF
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.CartesianRectanglePairNVF
   :no-index:
   :members:

```


## Materials data

See [Materials and wavelength sweeps](../guide/common/materials.md) for assignment and evaluation.

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.TabulatedMaterial
   :no-index:
   :members:

.. autofunction:: wavebit.materials.catalog.load

.. autofunction:: wavebit.materials.catalog.available

```


## Transient point probes

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.sample_fdtd_fields

.. autofunction:: wavebit.compat.legacy.fdtd_poynting

.. autofunction:: wavebit.compat.legacy.fit_fdtd_modes
```

## Plotting

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.plot_structure

.. autofunction:: wavebit.compat.legacy.slice_structure

.. autofunction:: wavebit.compat.legacy.plot_fdtd_field

.. autofunction:: wavebit.compat.legacy.animate_fdtd_field

```


## Precision, and what warns

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.enable_x64


.. autoclass:: wavebit.compat.legacy.PrecisionLossWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.UnresolvedBranchWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.RayleighPointWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.AutoDetuneWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.AxisNameWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.ComplexIncidenceWarning
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.GainCutoffWarning
   :no-index:
   :members:

```

## Side-resolved coherent illumination

See [the guide and flowchart](../guide/rcwa/coherent-illumination.md) for physical conventions.

```{eval-rst}
.. autoclass:: wavebit.compat.legacy.Source
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.PlaneWave
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.OrderExcitation
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.GaussianBeam
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.ElectricDipole
   :no-index:
   :members:

.. autoclass:: wavebit.compat.legacy.CoherentSolver
   :no-index:
   :members: solve, reconstruct_field

.. autoclass:: wavebit.compat.legacy.CoherentResult
   :no-index:
   :members: boundary_fields, boundary_power, reconstruct_field, output_top, output_bottom, absorption

.. autoclass:: wavebit.compat.legacy.BoundaryPower
   :no-index:
   :members:
```

## Native-grid time domain

```{eval-rst}
.. autofunction:: wavebit.compat.legacy.fdtd_grid
```
