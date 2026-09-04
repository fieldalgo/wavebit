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

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: guide/what-is-wavebit
:link-type: doc

New to Wavebit? What it solves and what it is bad at, how to install it, and a binary grating
in fifteen lines.
:::

:::{grid-item-card} User guide
:link: guide/README
:link-type: doc

Describing a scene, saying what the light is, choosing the numerics, near fields, autodiff
and backends.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

The normative contract, every keyword with its default and its refusals, and the API
generated from the docstrings.
:::

:::{grid-item-card} Decisions
:link: decisions/index
:link-type: doc

Why a default is what it is, and what was measured to decide it. Twenty-eight write-ups, cited
from the code they explain.
:::

::::

## The seven conventions

Each is legal, energy-conserving and *wrong* when read backwards — no error, no warning.
[`conventions.md`](conventions.md) is the normative contract.

| | |
| --- | --- |
| **Materials are relative permittivity** | not refractive index; `eps_from_index` converts |
| **Time convention is `e^{-iωt}`** | `Im(eps) > 0` is loss |
| **Layers run top to bottom** | and `+z` points into the stack |
| **Uppercase is power, lowercase is amplitude** | `R` / `T` against `r_s` / `t_p` |
| **Arrays run `v` first, written pairs run `x` first** | they meet at `truncation_order` |
| **The cell is centred at the origin** | and shape lengths are absolute |
| **Units are yours** | `wavelength`, `period` and `thickness` share one |

## Where the rest lives

Tutorials — lessons in order, for someone who has not used the API yet — are runnable
notebooks in the repository under
[`tutorials/`](https://github.com/fieldalgo/wavebit/tree/develop/tutorials).
