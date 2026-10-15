# 19. Judging a result: energy conservation is not enough

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

First check the physical problem you defined, then check whether its numerical solution stabilizes. Energy conservation can still hold when refractive index was entered in place of permittivity or incidence was interpreted backward. “R+T=1” and “the accurate answer to my intended scene” are therefore different statements.

```{figure} ../../figures/learn/learn/19-convergence.svg
:alt: Accuracy checklist distinguishing geometry, Fourier order, NVF, z subdivision and output sampling
:figclass: learning-figure

Figure 19-1. Accuracy checklist distinguishing geometry, Fourier order, NVF, z subdivision and output sampling
```

For this course, the first grating was changed to **TM** incidence with integrated coefficients and Li factorization, increasing only x order. These are recorded outputs of {download}`07_convergence.py <../../../examples/rcwa/learning/learn/07_convergence.py>`. Its settings differ from the first TE example; do not combine them as one comparison.

| Nx (Ny=0) | R | T | Energy residual abs(R+T−1) |
| --- | --- | --- | --- |
| 1 | 0.103391197 | 0.896608803 | 2.22e-16 |
| 3 | 0.256502718 | 0.743497282 | 4.00e-15 |
| 5 | 0.274911347 | 0.725088653 | 2.22e-16 |
| 7 | 0.279032278 | 0.720967722 | 1.55e-15 |
| 11 | 0.281219473 | 0.718780527 | 1.31e-14 |

Every row has a small conservation residual, yet R changes substantially. R still changes from `Nx=7→11`. This table **distinguishes conservation from convergence**; it does not declare convergence complete. For a design, choose the tolerance first, then check stability of the target observable over several refinements of order, sampling and boundary approximation.

The table uses a lossless incident medium and finite layers. For lossy incidence, use the relation between `power_entering`, transmission and absorption instead of `R+T`. [Chapter 13](power-and-efficiency.md) defines these power balances.

## What should be checked, and in which order?

1. **Scene:** units, ε versus n, coordinate origin, layer order, overlap order and polarization.
2. **Basic physics:** energy balance where applicable, simple-film limits and symmetry.
3. **Target output:** the desired order, phase, fields or gradients as well as total R/T.
4. **Independent resolutions:** vary Fourier order, material raster, shape NVF grid and z subdivision separately.
5. **External reference:** when available, compare an analytical solution or independent implementation after aligning conventions.

Two nearby orders can agree accidentally, while small variations near a resonance can produce large changes. Other observables and nearby wavelengths help reveal these effects. When using gradients, check objective convergence and gradient convergence separately.

| Symptom | First checks |
| --- | --- |
| Unexpected transmission | Whether material inputs are n or ε; whether the output is total T or T00 |
| Rapid growth in time and memory with order | Two-direction product H, stored field state, shape NVF samples |
| Only the field plot becomes smoother | More output samples versus an improved Fourier solution |
| Polarization conversion appears reversed | Incident and outgoing s/p frames, each order’s basis and time convention |
| Strange results after changing geometry in JIT | Stale StaticGeometry and whether it was refreshed outside tracing |
| Warnings or abnormal values | Read the cause; inspect inputs, precision and proximity to Rayleigh conditions |

A Rayleigh point lies at the transition between a propagating and decaying diffraction order. Branch selection and precision matter nearby. Do not indiscriminately hide `RayleighPointWarning`, `AutoDetuneWarning`, `PrecisionLossWarning`, `UnresolvedBranchWarning` or `GainCutoffWarning`; record what handling actually occurred. `GainCutoffWarning` appears for each gain half-space with an order near grazing; identify its orders with `solver.gain_cutoff_band()`. R + T exceeding 1 alone produces no warning, because it can be correct for gain media. Clear conditions make a calculation more reproducible than forcing a number out of it.
