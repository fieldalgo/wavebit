# Wavebit documentation

**Version:** 0.1.0 (unreleased)

**Useful links:** [Source repository](https://github.com/fieldalgo/wavebit) |
[Issue tracker](https://github.com/fieldalgo/wavebit/issues) |
[Contributing](https://github.com/fieldalgo/wavebit/blob/develop/CONTRIBUTING.md) |
[FieldAlgo](https://fieldalgo.com)

Wavebit simulates how light interacts with nanostructures, and differentiates through the whole
calculation — so the same code that predicts a device's response can also optimize it. It is a
rigorous, fully vectorial RCWA solver whose physics is written once and runs on NumPy, JAX and
PyTorch, so a result never depends on which you pick.

Time-domain work has two entry points: the [periodic-stack solver](guide/time-domain.md)
shares RCWA's scene interface; the [general FDTD engine](guide/general-fdtd.md) uses
an explicit spatial domain with sources, boundaries and named monitors. The latter
has its own units, array layout and bounded support contract.

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
:link: guide/README
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
:link: developer/index
:link-type: doc

Edit the source, debug a worktree and build the documentation.
:::

::::

## The seven conventions

Each is legal, energy-conserving and *wrong* when read backwards — no error, no warning.
[`conventions.md`](conventions.md) is the normative contract for the interfaces
summarized here. The [general FDTD guide](guide/general-fdtd.md#units-coordinates-and-backends)
states that engine's separate coordinate, angle and time conventions.

| | |
| --- | --- |
| **Materials are relative permittivity** | not refractive index; `eps_from_index` converts |
| **Time convention is `e^{-iωt}`** | `Im(eps) > 0` is loss |
| **Layers run bottom to top** | `z=0` at the bottom interface; `+z` points upward |
| **Uppercase is power, lowercase is amplitude** | `R` / `T` against `r_s` / `t_p` |
| **Arrays run `v` first, written pairs run `x` first** | they meet at `truncation_order` |
| **The cell is centred at the origin** | and shape lengths are absolute |
| **Units are yours** | `wavelength`, `period` and `thickness` share one |

## Where the rest lives

Tutorials — lessons in order, for someone who has not used the API yet — are runnable
notebooks in the repository under
[`tutorials/`](https://github.com/fieldalgo/wavebit/tree/develop/tutorials).
