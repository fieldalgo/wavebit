# 11. Multiple shapes, auxiliary solvers, and common misconceptions

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

## Adding multiple shapes to a layer

You can add shapes to the `shapes` list. When different materials touch or a shape crosses a periodic cell, the shapes cannot be treated only as independent islands. Visible material boundaries and periodic images must be handled together. For overlapping inputs, painting rules determine the actual regions. See the [parametric modeling documentation](../../guide/rcwa/parametric-modeling.md) for supported cases and overlap handling.

An internal line between two shapes with identical material is not a physical permittivity discontinuity. However, how the input is split into shapes can affect the direction field's interior extension and therefore the result at finite truncation. There is no guarantee that “the same object” produces bit-identical finite-order results for every decomposition of its input. Include representation choice in validation when it matters.

## Does ordinary NVF also run FEM?

```{figure} ../../figures/learn/nvf/12-map.svg
:alt: Ordinary NVF and explicitly requested auxiliary cells use different construction paths
:figclass: learning-figure

Figure 11-1. Ordinary NVF and explicitly requested auxiliary cells use different construction paths
```

**The five examples in this course do not call an auxiliary FEM cell.** Ordinary NVF extends boundary-direction information, constructs projector coefficients, and assembles material matrices. Selecting `factorization='nvf'` does not automatically enable a triangular auxiliary solver.

Separate advanced paths include `ConformingNVF` and experimental pair-transmission cells. These solve additional electrostatic or weak-form problems to construct a field basis or material response, then pass the assembled matrices to the existing RCWA engine. Their supported structures, material conditions, approximation spaces, and validation status differ from ordinary NVF. Here we only identify their place in the workflow. For advanced details, continue with the [ConformingNVF guide](../../guide/rcwa/conforming-nvf.md) and the [development-history tutorial](../../validation/rcwa/nvf-contact-followup.md).

## Reading errors and unexpected results

| Situation | Check this first |
| --- | --- |
| shape_nvf_grid_size is rejected for raster input | Change the input image resolution. shape_nvf_grid_size applies to shape NVF. |
| raster_normal_method is rejected for shape input | Shape normals come from geometry. Remove that option. |
| Combining Li and nvf_projector is rejected | This path does not construct an NVF projector. Remove NVF-specific options. |
| Shape Li is rejected for a circle or rotated shape | Use the supported general shape NVF path. |
| Entering n=2 gives an unexpected result | material=2 means ε=2. For n=2, supply ε=4. |
| R+T is 1 but two settings give different T | Energy conservation and convergence are different checks. Compare truncation and spatial representation. |
| Switching to circular makes the result worse | There is no universal accuracy ranking. Compare against a reference for your quantity of interest. |
| Geometry errors or NaN appear under JIT | First check an eager run on concrete inputs and prepare geometry outside the transform. Also check input validity and conditioning. |
| Transmittance is stable but its gradient varies | Values and derivatives converge separately. In particular, check shape_nvf_grid_size together with shape variations for shape input. |
