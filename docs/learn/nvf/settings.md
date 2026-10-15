# 06. Two input types and the role of each setting

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Wavebit accepts material structures in two main forms. **Raster and shape describe geometry inputs; NVF is a computational method that incorporates boundary information into material matrices.** A pixel image and a continuous shape represent different actual input geometries, even when both depict the same circle.

```{figure} ../../figures/learn/nvf/07-inputs.svg
:alt: A circle represented by pixels and by a circle shape
:figclass: learning-figure

Figure 6-1. A circle represented by pixels and by a circle shape
```

| Item | Raster: pixel array | Shape: shape dictionaries |
| --- | --- | --- |
| Input | Array of shape `(number of layers, y pixels, x pixels)` | `background` and `shapes` per layer |
| Default Fourier method | `sampled` | `integrated` |
| Default factorization | `li` | `nvf` |
| Enabling NVF | Specify `factorization='nvf'` | The default; may also be specified explicitly |
| Starting point for boundary directions | Estimated from the pixel material pattern | Constructed from shape boundaries and geometry |
| Direction-field method `raster_normal_method` | Used with NVF | Rejected if supplied |
| Projector construction `nvf_projector` | Linear / circular with NVF | Linear / circular with NVF |
| Projector sampling size `shape_nvf_grid_size` | Rejected if supplied | Used with NVF |

## Read the role before the name

`raster_normal_method` and `shape_nvf_grid_size` sound related, but they are not settings to tune together on the same path. This course names their target input explicitly. The descriptive labels below are explanatory; Python code uses the **supported API keywords** themselves.

| Descriptive label | Actual API keyword | Question it answers | Applies to |
| --- | --- | --- | --- |
| Raster direction-field construction method | `raster_normal_method` | How should directions be estimated from pixels and extended? | Raster NVF |
| Shape projector-field sample count per axis | `shape_nvf_grid_size` | At how many points should the auxiliary vector and P be evaluated? | Shape NVF |
| NVF projector construction | `nvf_projector` | How should the decomposition reference P be constructed? | Raster / shape NVF |
| Retained Fourier orders | `truncation_order` | How many Fourier modes should the optical calculation retain? | Both input types |

**The reference API uses the newer names.** The former `normal_method`, `normal_grid`, and `projector` were renamed to `raster_normal_method`, `shape_nvf_grid_size`, and `nvf_projector`, respectively. The former names are not accepted as aliases. `shape_nvf_grid` was only a candidate name, never an actual keyword. Option values and numerical algorithms are unchanged. The reference JobSpec schema version is **10**, and its Python constructor supplies that version. Older saved files require reversing the layer list together with the corresponding thickness and material arrays. Field-sample depth `d` becomes height `h_total − d`, and existing z-coupled tensors require the corresponding coordinate transformation as well. Here `h_total = sum(thickness)` is the total structure height. Changing only a version number does not preserve the same sample. See the [coordinate migration rules](../../reference/common/conventions.md).

## Coefficient construction and product rules are independent

`fourier_coefficients` selects how material Fourier coefficients are obtained, whereas `factorization` selects how discontinuous products are treated. Raster input permits all four combinations of sampled/integrated with li/nvf. A third factorization, the opt-in `'laurent'` (the direct rule for every product), exists for comparison with solvers that use it; this course does not use it. Integrated raster accounts for integrals over each rectangular pixel, but it does not turn the input's staircase geometry into a smooth circle.

For simple circles, ellipses, and rotated rectangles, the shape path can construct material coefficients through analytic integration. A `type='polygon'` outline (since 2026-10-15) is transformed exactly on the painting route instead; see [polygon regions](../../guide/rcwa/boundary-modeling.md#polygon-regions). Not every combination of shapes is handled in closed form, however. Painting overlapping curves, for example, involves polygonal approximations and separate geometry-accuracy controls. NVF projector coefficients also require numerical integration, so “analytic integration of material coefficients” does not mean “an exact closed-form optical solution.”

```{figure} ../../figures/learn/nvf/08-settings.svg
:alt: Construction method, projector type, sample count, and Fourier orders are different settings
:figclass: learning-figure

Figure 6-2. Construction method, projector type, sample count, and Fourier orders are different settings
```

## Raster direction-field construction: raster_normal_method

A raster does not directly contain the mathematical boundary of a circle or rectangle. Boundary directions must be estimated from the pixel array of material values, then extended into the interior. `raster_normal_method` selects this construction. Its default is `distance`.

| Method | Starting point for the direction field | What to keep in mind |
| --- | --- | --- |
| `distance` | Changes in distance to the boundary | Default; the pixel grid can introduce direction errors |
| `contour` | Geometry of contours reconstructed from pixels | Requires the optional contourpy dependency |
| `spectral` | Fourier derivatives of a smoothed pattern | Check the effects of sharp boundaries and tie handling |
| `smooth` | Spatial derivatives of a smoothed pattern | Used for this course's two-phase raster example |

Next, `nvf_projector` uses this information to construct a linear or circular projector. **The four raster_normal_method choices and the two nvf_projector choices are independent.** Shape constructs its own direction field from geometry, so it does not take the raster-specific raster_normal_method.

In the reference implementation, contour/spectral/smooth are paths designed for two-material patterns. Patterns with three or more values, or grayscale patterns, are delegated to distance. Do not infer from the requested name alone that the same method always runs internally. Portions of this geometry-preparation process also run in NumPy/SciPy even when JAX or PyTorch is selected. See [`normal_methods.py`](https://github.com/fieldalgo/wavebit/blob/develop/src/wavebit/rcwa/_engine/normal_methods.py) for the implementation details.

## Shape projector-field sampling: shape_nvf_grid_size

Shape geometry can define auxiliary and projector fields directly from its boundaries. To obtain Fourier coefficients of P numerically, however, **P must be evaluated at multiple positions throughout a periodic domain**. `shape_nvf_grid_size` is the number of points along each axis of that evaluation grid.

| Setting | Evaluation points in one periodic domain | What changes |
| --- | --- | --- |
| `shape_nvf_grid_size=128` | 128×128 = 16,384 | Auxiliary- and projector-field sampling resolution |
| `shape_nvf_grid_size=256` | 256×256 = 65,536 | The same field is evaluated at four times as many points |

This does not turn the shape into 128×128 material pixels. It changes **the spacing at which auxiliary J and P are sampled**, while keeping the geometry and Fourier orders fixed. Doubling it quadruples the number of samples, but does not necessarily quadruple total runtime.

The default is `max(1024, 4Nx+1, 4Ny+1)`. An explicitly supplied grid that is too small may be rejected for failing the minimum required by the retained Fourier orders. A finer grid reduces sampling error in projector coefficients; convergence of values and gradients must be checked separately.

Raster uses the original image resolution, increasing it to an anti-aliasing lower bound when needed. A user-supplied shape_nvf_grid_size does not control that path, so the reference API rejects it explicitly for raster input. To improve raster geometry accuracy, represent the original shape with finer pixels; simply replicating pixels does not recover missing boundary information.

**shape_nvf_grid_size asks “at how many points is the field sampled?”; truncation_order asks “how many Fourier components are retained?”** Increasing one does not repair an inadequate choice of the other. Chapter 9 varies them separately to compare convergence.

This book's shape examples explicitly use `shape_nvf_grid_size=128` for faster learning. **This is not a recommendation to lower the default to 128.** Choose a value for actual work using the convergence procedure below.
