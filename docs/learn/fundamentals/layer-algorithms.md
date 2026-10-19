# 15. Connecting layers: ETM and SMM

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

After finding each layer’s modes, the layers still need to be connected to obtain total reflection and transmission. `rcwa.Numerics(algorithm=...)` selects this connection method. **It is separate from material Fourier order and the NVF construction.**

```{figure} ../../figures/learn/learn/14-layers.svg
:alt: ETM and SMM connect the same layer modes using different algorithms
:figclass: learning-figure

Figure 15-1. ETM and SMM connect the same layer modes using different algorithms
```

SMM is the current default. `algorithm="etm"` computes the response to specified incidence using stabilized connection equations. SMM constructs a scattering matrix relating incoming and outgoing waves and cascades those relations. Both share the same modes and physical conventions, so agreement checks the connection algorithms rather than providing a fully independent physical validation.

Both can be used for isotropic materials and anisotropic second- and first-order calculations, historically called `2ff` and `4ff`. When xz, yz, zx or zy components of ε or μ select the first-order path, `algorithm="smm"` remains available. Both methods support top and bottom incidence and electric and magnetic fields inside the structure. `top` and `bot` must be isotropic media.

`Outputs(scattering=True)` requests SMM scattering operators; `result.scattering` returns one matrix per wavelength as a tuple. ETM does not form that operator, so requesting it is refused. Do not assume unrequested result properties are available. Cascading scattering matrices requires compatible `Port` bases, orders and media. Layers are not connected by ordinary matrix multiplication.

Runnable file: {download}`08_material_and_layers.py <../../../examples/rcwa/learning/learn/08_material_and_layers.py>`

```{literalinclude} ../../../examples/rcwa/learning/learn/08_material_and_layers.py
:language: python
:linenos:
```

| Calculation | R | T | A |
| --- | --- | --- | --- |
| ETM | 0.059958423249 | 0.935359040096 | 0.004682536654 |
| SMM | 0.059958423249 | 0.935359040096 | 0.004682536654 |
| Separate Airy formula | 0.059958423249 | 0.935359040096 | 0.004682536654 |

The values were recomputed on 2026-10-19, when the bundled silicon table changed to the cited Green (2008) data; the course record in `learn-measurements.json` was measured with the earlier table.

This example is a 120 nm silicon film using bundled material data at 900 nm. ETM and SMM R/T were compared with a separately written Airy formula for a single film at normal incidence. This closed form provides an independent elementary check for the teaching example; it does not validate patterned structures or contact NVF accuracy.

Combining exponentially growing and decaying modal factors in a naive transfer matrix can be numerically unstable. For ordinary use, focus on ETM and SMM. The `TMM` path available in some first-order diagnostics is not a universal third option for all structures.

A thick glass substrate represented as a finite layer includes back-surface reflection and round-trip interference. [Appendix C on thick substrates](incoherent-substrate.md) explains how this differs from a half-space and what happens when measurement averages the interference fringes.
