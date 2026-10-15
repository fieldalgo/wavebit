# 09. Before trusting a result: how to check convergence

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

Convergence checks ask whether the quantity you need remains sufficiently stable as numerical settings are refined. First decide whether your objective is total transmittance, a particular diffraction order, a phase, or a material or shape gradient.

```{figure} ../../figures/learn/nvf/10-convergence.svg
:alt: Check optical truncation and the projector integration grid separately
:figclass: learning-figure

Figure 9-1. Check optical truncation and the projector integration grid separately
```

{download}`Example 3 <../../../examples/rcwa/learning/nvf/03_convergence.py>` varies these two axes separately for the structure in Example 1. The following values come from actual execution and form a short learning-oriented refinement sequence.

```bash
python examples/rcwa/learning/nvf/03_convergence.py
```

| Comparison | Fourier orders | shape_nvf_grid_size | p transmittance T |
| --- | --- | --- | --- |
| Vary Fourier orders | (1,1) | 128 | 0.997080624533 |
| Vary Fourier orders | (2,2) | 128 | 0.997130496916 |
| Vary Fourier orders | (3,3) | 128 | 0.997065782998 |
| Vary grid | (3,3) | 64 | 0.997065792474 |
| Vary grid | (3,3) | 128 | 0.997065782998 |
| Vary grid | (3,3) | 256 | 0.997065780155 |

For this structure, the displayed changes in Fourier order affect T more than the displayed changes in shape_nvf_grid_size. T also does not move monotonically with order. There is no evidence that the last value is the true answer, and accidental agreement between two points is insufficient. These differences remain even though every row has an energy residual of roughly 10⁻¹⁵.

## A procedure for a real problem

1. Define the quantity of interest and its tolerance first. For example, absolute transmittance differences and relative gradient differences may have different acceptance criteria. No single number is suitable for every problem.
2. For shape input, fix the Fourier orders and increase shape_nvf_grid_size. For raster input, represent the original geometry with finer pixels and compare. Keep the Fourier-coefficient method and normal-field settings fixed during this step.
3. Fix a sufficiently stable spatial representation and increase Fourier orders through several levels. If variation differs along x and y, increase Nx and Ny separately to identify which direction is under-resolved.
4. Finally, refine both resolutions together and check that the result remains stable. For optimization, compare the relevant derivative as well as the value.
5. Where possible, compare against an analytic solution, an independent solver, or a sufficiently validated reference. Check energy conservation and symmetry as well, while distinguishing what each check establishes.

Dense material matrices have size M×M, and the electromagnetic eigenproblem uses larger blocks. Increasing orders in both directions grows M rapidly, with substantial increases in dense-matrix memory and decomposition cost. `shape_nvf_grid_size` primarily increases direction-field sampling and integration costs. Their costs arise differently, so identifying the under-resolved axis is more useful than maximizing every setting at once.

With absorption, do not automatically expect R+T=1. For passive structures with appropriate input/output media and normalization, `1−R−T` is related to absorption. Do not apply the lossless example's energy-check formula unchanged to every lossy problem.
