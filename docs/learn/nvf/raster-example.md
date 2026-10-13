# 08. Using pixel input and selecting NVF

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

For an image or an existing material array, use {download}`Example 2 <../../../examples/rcwa/learning/nvf/02_raster_nvf.py>`. It constructs a circle on a 64×64 pixel grid, then applies integrated coefficients and NVF. Membership in the circle of radius 160 nm is evaluated at pixel centers, so its geometry is not exactly the continuous circle in Example 1.

```{literalinclude} ../../../examples/rcwa/learning/nvf/02_raster_nvf.py
:language: python
:linenos:
```

```bash
python examples/02_raster_nvf.py
python examples/05_select_li_or_nvf.py
```

The measured result for Example 2 is `R_p=0.002827293719`, `T_p=0.997172706281`, with energy residual `2.22×10⁻¹⁶`. Its difference from Example 1 includes both pixel-geometry differences and differences in normal-field construction. Those two results alone cannot establish which path is closer to the true value.

## Enabling or disabling NVF

For raster input, select `factorization='nvf'` or `'li'`. When selecting Li, omit NVF-specific settings such as `raster_normal_method`, `nvf_projector`, and `shape_nvf_grid_size`. The reference implementation rejects unused options rather than silently ignoring them.

For shape input, NVF is the default. You can explicitly select `factorization='li'` for **axis-aligned rectangles inside an orthogonal cell**. The shapes must fit within the cell. This shape Li path does not support rotated shapes, curves, skewed cells, or rectangles crossing a cell boundary. Being a rectangle does not automatically disable NVF.

{download}`Example 5 <../../../examples/rcwa/learning/nvf/05_select_li_or_nvf.py>` calculates the same supported rectangle with both methods. Its measured output is:

| Method | p transmittance T | Energy residual |
| --- | --- | --- |
| Shape Li | 0.995999620094 | 1.11×10⁻¹⁵ |
| Shape NVF / linear | 0.995802829398 | 1.33×10⁻¹⁵ |

Both methods conserve energy well, but give different answers at finite order. This table illustrates method selection; it is not a benchmark identifying a winner in accuracy or speed.
