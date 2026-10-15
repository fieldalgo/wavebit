# 02. Start with boundary conditions: the whole E field is not continuous

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

For a material with scalar relative permittivity ε, `D = ε₀ ε E`. To simplify the notation below, define the normalized displacement `d = D/ε₀`. The vector n is a unit normal to the material boundary, and t is tangent to the boundary in the xy plane. The figures in this chapter resolve the transverse electric field E_xy into those two directions. The z component is also tangent to such vertical boundaries within a layer; Chapter 5 treats it in a separate slot.

At an ordinary material boundary, the tangential electric field `Eₜ` is continuous. In the absence of free surface charge, the normal electric displacement `Dₙ` is continuous as well. The normal electric field `Eₙ`, however, is generally discontinuous when the materials have different ε. This course uses these usual boundary conditions.

```text
E_xy = Eₙ n + Eₜ t
Eₜ₁ = Eₜ₂
dₙ₁ = dₙ₂  →  ε₁Eₙ₁ = ε₂Eₙ₂
```

```{figure} ../../figures/learn/nvf/02-boundary.svg
:alt: The tangential electric field is unchanged, while the normal electric field depends on permittivity
:figclass: learning-figure

Figure 2-1. The tangential electric field is unchanged, while the normal electric field depends on permittivity
```

Consider a local example with `ε₁=1`, `ε₂=4`, `dₙ=1`, and `Eₜ=1`.

| Component | Material 1 | Material 2 | Behavior at the boundary |
| --- | --- | --- | --- |
| Tangential Eₜ | 1 | 1 | Continuous |
| Normal dₙ | 1 | 1 | Continuous |
| Normal Eₙ = dₙ/ε | 1 | 0.25 | Discontinuous |
| Tangential dₜ = εEₜ | 1 | 4 | Discontinuous |

The central idea is therefore to **reflect the different continuity properties of normal and tangential components in the matrix construction**, instead of applying the same product rule to every electric-field component. This table describes relations at one boundary; it is not a solution for the field throughout an actual structure.

```{raw} html
<section class="exercise" aria-labelledby="boundary-title"><h3 id="boundary-title">Try it · Which component changes when ε changes?</h3><p class="explain">This local boundary-condition example fixes ε₁=1, Eₜ=1, and Dₙ/ε₀=1. Units are normalized, and no RCWA calculation runs here.</p><div class="controls"><label for="eps2">Relative permittivity on the right, ε₂ <output id="eps2-value" for="eps2">4.0</output><input id="eps2" type="range" min="1" max="12" step="0.1" value="4"></label></div><p class="explain">Bar length below = right Eₙ / left Eₙ</p><div class="bar-shell"><div class="bar" id="normal-bar"></div></div><div id="boundary-result" class="readout" aria-live="polite">At ε₂=4, Eₙ₂=0.25, Eₜ₂=1, and Dₙ₂/ε₀=1.</div><noscript><p>The table in Chapter 2 shows the same calculation.</p></noscript></section>
```
