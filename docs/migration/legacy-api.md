# API reference

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
This is the normalized internal-engine reference used to interpret earlier calculations. Its private paths are implementation details. `wavebit.compat` has been removed; the earlier public spelling is available only in the pinned historical checkout linked below. Start new applications with the [current lifecycle](../guide/common/lifecycle.md) and [current API reference](../reference/api.md). Numerical evidence quoted here retains its original scope.
:::

Generated from the retained engine docstrings for contributors checking numerical evidence.
Use the physical public interfaces for application code. Historical public API documentation
is preserved in the source revision, not re-exported as another runtime namespace.

## Building a solver

```{eval-rst}
.. autofunction:: wavebit.rcwa._factory.rcwa

```

`fdtd()` is the other door. It hands back the answer rather than a solver, because there is no
per-instance state to keep -- no eigen-decomposition to cache, no frozen geometry, no gradient.

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.api.fdtd

```


## The solver object

`rcwa()` returns a solver whose class is chosen by the backend. These are the methods and
properties every one of them carries.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.engine.CoreFactorizationMixin
   :no-index:
   :members: solve, reconstruct_field, freeze_geometry, solve_conv, eps_conv_matrices, mu_conv_matrices, gain_cutoff_band, factorization, fourier_coefficients, raster_normal_method, nvf_projector, li_nesting, half_space_merge_tolerance, shape_nvf_grid_size


.. autoclass:: wavebit.rcwa._engine.engine.CoreSolveMixin
   :no-index:
   :members: answers, theta, kx, ky, excitation, scattering, truncation_order, period, lattice


```

## Describing a scene

```{eval-rst}
.. autoclass:: wavebit.materials.response.Material
   :no-index:
   :members:

.. autofunction:: wavebit.materials.response.eps_from_refractive_index

.. autofunction:: wavebit.materials.response.refractive_index_from_eps

.. autofunction:: wavebit._common.periodic.layer_thicknesses

.. autofunction:: wavebit.fdtd._periodic.legacy.permittivity_at

.. autoclass:: wavebit.lattice.Lattice
   :no-index:
   :members:

.. autofunction:: wavebit.lattice.meshgrid_xy

.. autoclass:: wavebit._common.periodic.Layer
   :no-index:
   :members:

.. autoclass:: wavebit._common.periodic.Stack
   :no-index:
   :members:

.. autoclass:: wavebit._common.periodic.Incidence
   :no-index:
   :members:

```


## A run as data

```{eval-rst}
.. autoclass:: wavebit._common._jobs_v1.JobSpec
   :no-index:
   :members:

.. autoclass:: wavebit._common._jobs_v1.Numerics
   :no-index:
   :members:

.. autoclass:: wavebit._common._jobs_v1.Sweep
   :no-index:
   :members:

.. autofunction:: wavebit._common._jobs_v1.run

.. autoclass:: wavebit._common._jobs_v1.SpecError
   :no-index:
   :members:

```


## Experimental auxiliary NVF

These explicit NumPy interfaces do not change ordinary NVF routing or
certify optical convergence. See the [selection guide](../guide/rcwa/nvf-method-selection.md).

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair_opening.CartesianPairOpeningPlan
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair_transmission_galerkin.CartesianPairTransmissionGalerkinGapCell
   :no-index:
   :members: closure
   :inherited-members:

```

## Results and state

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.scattering.ScatteringMatrix
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.scattering.Port
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.geometry.StaticGeometry
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.geometry.StaleGeometryError
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.factorization.ConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.anisotropy.TensorConvMatrices
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.ConformingNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.affine_profile.AffineProfileNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_rectangle.CartesianRectangleNVF
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.conforming.cartesian_pair.CartesianRectanglePairNVF
   :no-index:
   :members:

```


## Materials data

See [Materials and wavelength sweeps](../guide/common/materials.md) for assignment and evaluation.

```{eval-rst}
.. autoclass:: wavebit.materials.TabulatedMaterial
   :no-index:
   :members:

.. autofunction:: wavebit.materials.catalog.load

.. autofunction:: wavebit.materials.catalog.available

```


## Transient point probes

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.probes.sample_fdtd_fields

.. autofunction:: wavebit.fdtd._periodic.probes.fdtd_poynting

.. autofunction:: wavebit.fdtd._periodic.probes.fit_fdtd_modes
```

## Plotting

```{eval-rst}
.. autofunction:: wavebit.rcwa._plotting.plot_structure

.. autofunction:: wavebit.rcwa._plotting.slice_structure

.. autofunction:: wavebit.fdtd._periodic.plotting.plot_fdtd_field

.. autofunction:: wavebit.fdtd._periodic.plotting.animate_fdtd_field

```


## Precision, and what warns

```{eval-rst}
.. autofunction:: wavebit._common.backend._enable_x64


.. autoclass:: wavebit._common.backend.PrecisionLossWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.UnresolvedBranchWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.RayleighPointWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.AutoDetuneWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.AxisNameWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.ComplexIncidenceWarning
   :no-index:
   :members:

.. autoclass:: wavebit._common.backend.GainCutoffWarning
   :no-index:
   :members:

```

## Side-resolved coherent illumination

See [the guide and flowchart](../guide/rcwa/coherent-illumination.md) for physical conventions.

```{eval-rst}
.. autoclass:: wavebit.rcwa._engine.sources.Source
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.PlaneWave
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.OrderExcitation
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.GaussianBeam
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.sources.ElectricDipole
   :no-index:
   :members:

.. autoclass:: wavebit.rcwa._engine.illumination.CoherentSolver
   :no-index:
   :members: solve, reconstruct_field

.. autoclass:: wavebit.rcwa._engine.illumination.CoherentResult
   :no-index:
   :members: boundary_fields, boundary_power, reconstruct_field, output_top, output_bottom, absorption

.. autoclass:: wavebit.rcwa._engine.illumination.BoundaryPower
   :no-index:
   :members:
```

## Native-grid time domain

```{eval-rst}
.. autofunction:: wavebit.fdtd._periodic.grid.fdtd_grid
```
