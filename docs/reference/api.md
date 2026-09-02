# API reference

Generated from the docstrings, so it cannot drift from the code. What is *not* here is
deliberate: `wavebit._core` is private, and its module-level notes explain the implementation
to a contributor rather than the interface to a caller.

## Building a solver

::: wavebit.rcwa

## The solver object

`rcwa()` returns a solver whose class is chosen by the backend. These are the methods and
properties every one of them carries.

::: wavebit._core.engine.CoreFactorizationMixin
    options:
      members:
        - solve
        - reconstruct_field
        - freeze_geometry
        - solve_conv
        - eps_conv_matrices
        - factorization
        - fourier_coefficients
        - normal_method
        - projector
        - normal_grid

::: wavebit._core.engine.CoreSolveMixin
    options:
      members:
        - answers
        - excitation
        - scattering
        - truncation_order
        - period
        - lattice

## Describing a scene

::: wavebit.Material
::: wavebit.eps_from_index
::: wavebit.index_from_eps
::: wavebit.Lattice
::: wavebit.grid
::: wavebit.Layer
::: wavebit.Stack
::: wavebit.Incidence

## A run as data

::: wavebit.JobSpec
::: wavebit.Numerics
::: wavebit.Sweep
::: wavebit.run
::: wavebit.SpecError

## Results and state

::: wavebit.ScatteringMatrix
::: wavebit.Port
::: wavebit.StaticGeometry
::: wavebit.StaleGeometryError
::: wavebit.ConvMatrices

## Materials data

::: wavebit.find_material
::: wavebit.find_nk_index
::: wavebit.read_material_table

## Plotting

::: wavebit.plot_structure
::: wavebit.slice_structure

## Precision, and what warns

::: wavebit.enable_x64
::: wavebit.branch_tolerance
::: wavebit.UnphysicalResultWarning
::: wavebit.PrecisionLossWarning
::: wavebit.UnresolvedBranchWarning
::: wavebit.RayleighPointWarning
::: wavebit.AutoDetuneWarning
::: wavebit.AxisNameWarning
