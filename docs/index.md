# Wavebit documentation

**Version:** {sub-ref}`release`

**Useful links:** [Source repository](https://github.com/fieldalgo/wavebit) |
[Issue tracker](https://github.com/fieldalgo/wavebit/issues) |
[Contributing](https://github.com/fieldalgo/wavebit/blob/develop/CONTRIBUTING.md) |
[FieldAlgo](https://fieldalgo.com)

Wavebit simulates electromagnetic waves with RCWA and FDTD, and supports differentiating supported calculations with JAX and PyTorch. Shared physical materials, explicit units and an immutable problem lifecycle connect the two solver modules. Each method has its own numerical approximation and validated feature scope.

Start with the [RCWA calculation](getting-started/rcwa.md) for periodic layered structures or the [FDTD pulse calculation](getting-started/fdtd.md) for a spatial domain with sources and named observations. The [learning courses](learn/index.md) explain the underlying physics, while [migration](migration/index.md) keeps earlier scientific workflows explicit.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install, choose a solver and run a first calculation.
:::

:::{grid-item-card} Learn
:link: learn/index
:link-type: doc

Follow lessons and modify runnable examples.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc

Describe materials, geometry, illumination and numerical settings.
:::

:::{grid-item-card} API reference
:link: reference/index
:link-type: doc

Look up signatures, accepted values and result contracts.
:::

:::{grid-item-card} Theory & validation
:link: theory/index
:link-type: doc

Read derivations, measured comparisons and their limits.
:::

:::{grid-item-card} Developer guide
:link: development/index
:link-type: doc

Edit the source, debug a worktree and build the documentation.
:::

::::

## Seven conventions to check

The [common conventions](reference/common/conventions.md) and [current lifecycle](guide/common/lifecycle.md) explain the physical meaning and the boundary between current and compatibility interfaces.

| Convention | Meaning |
| --- | --- |
| **Material values are relative permittivity** | Use `eps_from_refractive_index` when starting from refractive index. |
| **Phasors use `e^{-iωt}`** | Positive imaginary permittivity represents passive electric loss. |
| **Stack layers run bottom to top** | z starts at the bottom interface and increases upward. |
| **Power and amplitude differ** | Efficiencies are power ratios; complex amplitudes retain phase. |
| **Diffraction arrays retain wavelength first** | Followed by order v, then order u; written order pairs are `(u,v)`. |
| **Periodic cells are centered** | Shape coordinates use the cell center and physical length units. |
| **Units are declared explicitly** | Lengths follow `Units`; time is seconds, rates SI, angles radians, E V/m and H A/m. |

## Continue learning

The [fundamentals course](learn/fundamentals/index.md) develops geometry, materials, light, diffraction, power, fields and differentiation in order. The [NVF course](learn/nvf/index.md) covers Fourier products and material interfaces. The [FDTD course](learn/fdtd/index.md) connects sources, absorbers, observations and refinement to published validation.

Runnable [teaching examples](learn/examples.md), solver examples and [notebooks](https://github.com/fieldalgo/wavebit/tree/develop/tutorials) accompany the text. Historical courses state when they use compatibility interfaces; their numerical records were preserved rather than rerun.
