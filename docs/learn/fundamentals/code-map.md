# 20. Code map and further study

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

Read the implementation in **data-flow order** rather than file-size order. This map explains the RCWA workflow used by the source course. Its original text predates FDTD integration; source links are relocated in this edition, and it is not a specification of the new unified API.

```{figure} ../../figures/learn/learn/20-code-map.svg
:alt: Source-code flow from public API through geometry, matrices, modes and layer connections to results
:figclass: learning-figure

Figure 20-1. Source-code flow from public API through geometry, matrices, modes and layer connections to results
```

| Start here | Question to understand |
| --- | --- |
| [src/wavebit/__init__.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/__init__.py), [src/wavebit/rcwa/_factory.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_factory.py) | What is public, and which backend is constructed? |
| [src/wavebit/_common/periodic.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/periodic.py), [materials/](https://github.com/fieldalgo/wavebit/tree/main/src/wavebit/materials/), [_common/_jobs_v1.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/_common/_jobs_v1.py) | How are scenes, materials, light and calculation specifications represented? |
| [src/wavebit/lattice.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/lattice.py), [geometry.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/geometry.py) | What are the cell coordinates and the geometry information held fixed? |
| [src/wavebit/rcwa/_engine/parametric.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/parametric.py), [boundary.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/boundary.py) | How are shape Fourier coefficients and overlap boundaries handled? |
| [src/wavebit/rcwa/_engine/factorization.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/factorization.py), [normal_methods.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/normal_methods.py) | How are products of materials and fields assembled into matrices? |
| [src/wavebit/rcwa/_engine/engine.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/engine.py) | How do inputs connect to actual calculation paths and state? |
| [src/wavebit/rcwa/_engine/solve_1d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_1d.py), [solve_2d.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_2d.py), [solve_2d_first_order.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/solve_2d_first_order.py) | What are the modal equations and layer calculations? |
| [src/wavebit/rcwa/_engine/scattering.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/scattering.py), [fields.py](https://github.com/fieldalgo/wavebit/blob/main/src/wavebit/rcwa/_engine/fields.py) | How are scattering matrices and spatial fields obtained? |
| [src/wavebit/rcwa/_backends/](https://github.com/fieldalgo/wavebit/tree/main/src/wavebit/rcwa/_backends/) | Which computation tools do NumPy, JAX and PyTorch provide to the shared physics? |
| [src/wavebit/rcwa/_engine/conforming/](https://github.com/fieldalgo/wavebit/tree/main/src/wavebit/rcwa/_engine/conforming/) | How do auxiliary NVF constructions differ from an ordinary solve? |

Users need not import all implementation files directly. Write normal programs against the public API; use this map to understand the internals. The former `on_numpy`, `on_jax` and `on_torch` names are not the current source-adapter directories.

## Entry points to advanced features

| Public API group | Stage | Further reading |
| --- | --- | --- |
| `ConvMatrices`, `solve_conv` | Solve modes and layers from caller-prepared constitutive matrices | [Numerical settings and energy closure](../../guide/rcwa/numerics.md) |
| `ConformingNVF` | Separate construction of matrices for supported rectangle arrays | [Direct-use guide](../../guide/rcwa/conforming-nvf.md) |
| `AffineProfileNVF` | Separate affine-profile construction family | [Scope and example](../../guide/rcwa/affine-profile-nvf.md) |
| `CartesianRectangleNVF`, `CartesianRectanglePairNVF` | Specialized constructions for one rectangle or a rectangle pair | [One rectangle](../../guide/rcwa/cartesian-rectangle-nvf.md), [rectangle pair](../../guide/rcwa/cartesian-rectangle-pair-nvf.md) |
| `CartesianPairOpeningPlan`, `CartesianPairTransmissionGalerkinGapCell` | Separate experimental gap and contact constructions | [Gap-family scope](../../guide/rcwa/cartesian-pair-gap.md) |
| `ScatteringMatrix`, `Port` | SMM relations and connections between both-side inputs and outputs | [Earlier-interface API](../../migration/legacy-api.md#results-and-state) |
| `Formulation` | Select and inspect second-order (`second_order`, formerly 2ff) versus first-order (`first_order`, formerly 4ff) solving; a property rather than an ordinary constructor argument | [First-order formulation](../../theory/rcwa/first-order-formulation.md) |
| Bessel J1 implementation selectors | Advanced special-function diagnostics | [Bessel J1](../../validation/rcwa/bessel-j1.md) |

These groups do not share one support scope. Read each guide’s geometry, material and backend restrictions and unresolved limits. Do not assume an ordinary `solve()` automatically selects an experimental construction.

## Vocabulary used in this course

| Term | Meaning here |
| --- | --- |
| Scene | Physical problem combining structure, materials, lattice and light |
| Order | Spatial-frequency indices m,n of a periodic structure |
| Mode | A propagating or decaying state permitted by one layer’s coupled equations |
| Convolution matrix | Matrix representing multiplication between Fourier coefficients |
| Factorization | Rule for handling products of discontinuous functions with finite matrices |
| Phasor | Complex amplitude encoding the magnitude and phase of a time oscillation |
| Port | Input/output connection information including exterior medium and wave basis |
| Closure | Constructing matrices that close constitutive relations in a chosen approximation space |
| Convergence | Checking sufficient stability of a target output under resolution or order changes |
| Automatic differentiation | Connecting derivatives of executed operations to obtain input sensitivities |

## Check your understanding

1. Why is total T larger than T00 in the first example? **The ±1 transmitted orders also carry power.** Check that the output being summed matches the design objective.
2. Is changing circle diameter the same as increasing Fourier order? **The former changes the physical design; the latter changes its numerical approximation.**
3. Does doubling `sample_x` solve higher Fourier orders? **No.** It reconstructs the same solution at more positions.
4. Does circularly polarized light require `nvf_projector='circular'`? **No.** Incident polarization is specified by `polarization` or `psi/chi`.
5. If JAX agrees with central differences, is the physical solution of a contact geometry converged? **That does not follow.** Checking a derivative of one discrete calculation differs from checking physical convergence.

## Evidence and limits of this course

Conventions were checked against [conventions.md](../../reference/common/conventions.md), user options against [keywords.md](../../reference/keywords.md), and capability limits against the implementation and manifest. Where older prose differed from implementation, the implementation and official conventions took precedence. For example, accessing ETM scattering currently raises an exception, and the first raster row lies on the cell’s −y side.

The [build basis and provenance](../../development/learning-provenance.md), {download}`recorded outputs <../../validation/learning-records/learn-measurements.json>` and [review scope](../../development/learning-provenance.md) are preserved together. The original course work checked nine short examples and did not resume the interrupted full regression and installation checks. Those historical checks do not substitute for FDTD integration validation, evaluation of the reorganized repository, or public-release validation.

Consult the repository’s [LICENSE](https://github.com/fieldalgo/wavebit/blob/main/LICENSE), [LICENSE.meent](https://github.com/fieldalgo/wavebit/blob/main/LICENSE.meent), [LICENSE.fmmax](https://github.com/fieldalgo/wavebit/blob/main/LICENSE.fmmax) and associated notices for Wavebit code. The course’s diagrams and explanations were newly authored; external library sources were not copied into it. Reusing other programs’ results, tables or sources requires checking their individual attribution and usage conditions.
