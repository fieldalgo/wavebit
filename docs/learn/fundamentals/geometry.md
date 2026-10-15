# 05. Two languages for building structures: rasters and shapes

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

A raster uses an array to say “which material fills each cell.” Shape input uses parameters to say “where an object is placed and how large it is.” Both describe the same kind of layered Maxwell problem. The **data type** of `structure` selects the path; there is no separate mode-selection switch.

```{figure} ../../figures/learn/learn/05-geometry.svg
:alt: A comparison of the two input representations only: a raster array on the left and shape parameters on the right
:figclass: learning-figure

Figure 5-1. A comparison of the two input representations only: a raster array on the left and shape parameters on the right
```

| Representation | Input | Suitable design variables |
| --- | --- | --- |
| Raster | An ndarray of shape `(number of layers, number of rows, number of columns)` | Permittivity at each pixel, material distribution |
| Shapes | A `background` and a list of `shapes` for each layer | Center position, width, height, angle |

In a raster, different layers are stacked along the first axis of the same array. Shape input supports `rectangle`, `circle`, and `ellipse`, and since 2026-10-15 a `polygon`, `dict(type='polygon', vertices=..., material=...)`, which is placed and sized by its vertices alone and always takes the painting path described below. `cx` and `cy` are absolute lengths measured from the cell center; `lx` and `ly` are also absolute lengths. **A circle's `lx` is its diameter.** Rectangles and ellipses specify the full size along each of their two axes. The rotation `angle` is in radians.

Executable file: {download}`02_shape.py <../../../examples/rcwa/learning/learn/02_shape.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/02_shape.py
:language: python
:linenos:
```

| Measured result with the course settings | Value |
| --- | --- |
| R | 0.006360910053 |
| T | 0.993639089946 |
| R + T | 1.000000000000 |
| Result array shape (`result.T.shape`) | (5, 5) |

This example uses `shape_nvf_grid_size=64` and order `(2, 2)` to keep execution short while introducing the API. The result is not presented as a converged value for a final design. Analytically calculating the Fourier coefficients of material geometry and sampling the NVF projection field are separate stages.

## An additional rule for shape input: several shapes in one layer

```{figure} ../../figures/learn/learn/05b-painting.svg
:alt: Overlap is resolved in shape-list order: the material of a rectangle placed after a circle covers the overlapping region
:figclass: learning-figure

Figure 5-2. Overlap is resolved in shape-list order: the material of a rectangle placed after a circle covers the overlapping region
```

Place the shapes in order in `shapes`. **The material of a later shape covers an earlier shape.** Permittivities are not added in overlapping regions. For example, placing a shape made of the background material last can represent a hole. The shapes, intersections, and contacts that are accepted depend on the constraints of the current computational path.

For a layer with overlaps, the boundaries are analyzed to partition the geometry into non-overlapping regions. A painting path that includes curves may introduce polygonal boundary approximations. “Shape input” therefore does not mean that every stage handles curves with infinite precision. You can also inspect the input scene with `wavebit.plot_structure(...)` or `wavebit.slice_structure(...)`. See the [earlier-interface API reference](../../migration/legacy-api.md) for the detailed arguments.
