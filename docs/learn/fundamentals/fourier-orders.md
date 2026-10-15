# 10. Fourier orders: accuracy and computational cost

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Fourier order determines the range of spatial-frequency components retained in a calculation. First examine the relation between orders and matrix size, then distinguish what geometry resolution, coefficient calculation, Li and NVF each change.

## Fourier order and matrix size

Write the maximum x and y orders in `truncation_order` as {math}`N_x` and {math}`N_y`. For ordinary single-wave zeroth-order incidence, the retained components have {math}`m=-N_x,\ldots,N_x` and {math}`n=-N_y,\ldots,N_y`. Their combinations give the **total number of Fourier components**, {math}`H`:

```{math}
H=(2N_x+1)(2N_y+1).
```

For example, `truncation_order=dict(x=2, y=1)` gives {math}`N_x=2`, {math}`N_y=1`. There are **five** x orders, −2, −1, 0, 1, 2, and **three** y orders, −1, 0, 1, for a total of {math}`H=5\times3=15` components.

Multiple incident orders can expand or shift the actual order window. In that case, inspect `result.orders_u` and `orders_v` directly.

```{figure} ../../figures/learn/learn/10-orders.svg
:alt: Fourier-order grid and its correspondence to rows and columns of the result array
:figclass: learning-figure

Figure 10-1. Fourier-order grid and its correspondence to rows and columns of the result array
```

```{raw} html
<section class="exercise" aria-labelledby="orders-title"><h3 id="orders-title">Try it: how many components does increasing order retain?</h3><p class="concept">Concept demonstration for single-wave zeroth-order incidence; not an RCWA solve or timing measurement.</p><div class="controls"><label for="order-x">Maximum x order N<sub>x</sub> <output id="order-x-out" for="order-x">2</output><input id="order-x" type="range" min="0" max="10" value="2"></label><label for="order-y">Maximum y order N<sub>y</sub> <output id="order-y-out" for="order-y">1</output><input id="order-y" type="range" min="0" max="10" value="1"></label></div><svg viewBox="0 0 400 245" role="img" aria-label="Conceptual grid of retained Fourier orders. Orange denotes order zero."><g id="order-cells"></g><text x="125" y="241">Orange: (m, n) = (0, 0)</text></svg><div class="readout" id="order-readout" aria-live="polite">Total components H = 15 · Result shape (rows, columns) = (3, 5) · H² = 225</div><noscript><p>Enable JavaScript to manipulate the order grid.</p></noscript></section>
```

Increasing both directions multiplies the number of orders. Storage for many central matrices scales roughly as {math}`H^2`, and dense eigensolve work roughly as {math}`H^3`. Actual runtime depends on modal formulation, layer count, factorization, caching and backend. H² in the control above describes matrix size; it is not a timing benchmark.

## Representing geometry versus representing fields

| Setting | What it changes |
| --- | --- |
| Rows and columns in the material raster | Cells representing geometry and material distribution |
| `truncation_order` | Fourier orders retained in the solution and operators |
| `fourier_coefficients` | Method used to obtain Fourier coefficients from the specified material distribution |
| `factorization` | Rule mapping products of discontinuous materials and fields to finite matrices |
| `shape_nvf_grid_size` | Grid sampling the shape NVF projector field |
| Field reconstruction `sample_x/y/z` | Output positions for an already calculated solution |

A raster divides a layer into rows and columns, storing material values in each cell. Isotropic permittivity and permeability can each be specified by **one scalar**, such as `2.25`. To represent anisotropy, specify each cell’s permittivity or permeability as a **3×3 matrix**. {ref}`Chapter 7 <material-anisotropy>` explains components and inputs.

**A 3×3 matrix is a format for expressing material values.** Even isotropic permittivity can be represented as {math}`\varepsilon_r I`, with equal diagonal entries. The table below distinguishes inputs by whether each cell holds a scalar or a 3×3 matrix.

The raster default `fourier_coefficients='sampled'` obtains coefficients by applying a discrete Fourier transform to array values. `'integrated'` treats each cell as a constant-material region and integrates it. Both methods calculate the supplied pixel distribution.

For shapes, `fourier_coefficients='integrated'` is the default with either isotropic or anisotropic materials. It integrates shape regions to obtain Fourier coefficients, using analytical formulas for nonoverlapping circles, ellipses and rectangles. For overlaps, it identifies and integrates the regions left occupied by each material; overlapping regions involving curves use polygonal approximations.

| Input format | Default Fourier coefficients | Default factorization |
| --- | --- | --- |
| Raster: scalar material per cell | `sampled` | `li` |
| Raster: 3×3 material matrix per cell | `sampled` | `nvf` |
| Shapes: scalar or 3×3 material matrix | `integrated` | `nvf` |

For NVF patterns represented by a **raster with a 3×3 material matrix in each cell**, use `fourier_coefficients='sampled'`. Li supports both `'sampled'` and `'integrated'` for that format. Scalar rasters support both coefficient methods with either Li or NVF.

Coefficient calculation and factorization are separate stages. Even if shape-region integration uses analytical formulas, NVF normal fields and auxiliary functions are calculated on a grid, so convergence in `shape_nvf_grid_size` must be checked separately. Oblique-lattice and anisotropic-material support is discussed below.

## Where do Li and NVF differ?

See the {ref}`calculation-flow diagram in Chapter 2 <factorization-flow>` for their place in the complete solve. Here we compare matrix construction and applicability.

`li` applies direct and inverse rules along the lattice u and v axes. Wavebit’s current Li path supports isotropic materials and anisotropic materials with full 3×3 ε and μ on an **orthogonal lattice whose u and v axes align with physical x and y**. Geometry inputs can be rasters or axis-aligned rectangles.

`nvf` constructs matrices using boundary-normal directions. Diagonal and off-diagonal components are assembled together; do not interpret this as adding a few correction terms to an already completed Li matrix. [Chapter 11](nvf.md) explains normal fields and projectors.

**NVF also supports anisotropic materials on oblique lattices.** Use `nvf_projector="linear"` for NVF patterns containing anisotropic materials. Both Li and NVF assemble material matrices including xz, yz, zx and zy coupling; these use the first-order formulation (`first_order`, formerly 4ff), treating all three directions together. Too few NVF Fourier orders can make it difficult to separate modal propagation directions. This raises an error in ordinary execution and produces NaN during JAX tracing. Increase order and normal-field resolution and check convergence. See [Chapter 7](materials.md) for detailed material support.

This example rotates the principal axes of an anisotropic material by 0.3 rad and calculates a rectangular shape using Li. Only the material axes rotate; the rectangle’s edges remain aligned with x and y.

```python
from wavebit.compat import legacy as wavebit

crystal = wavebit.Material(epsilon=[2.25, 4., 2.8]).rotated(0.3)
solver = wavebit.rcwa(
    wavelength=730., period=(600., 600.), thickness=[173.],
    structure=[{"background": 1., "shapes": [
        {"type": "rectangle", "lx": 240., "ly": 300., "material": crystal},
    ]}],
    factorization="li", truncation_order=dict(x=3, y=3),
)
result = solver.solve()
```

With `factorization="li"`, do not specify NVF-only options `nvf_projector`, `raster_normal_method` or `shape_nvf_grid_size`. The order above is an initial example setting; check order convergence for the quantity of interest.
