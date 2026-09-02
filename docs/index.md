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

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting started**

    ---

    New to Wavebit? A binary grating solved in fifteen lines, and the seven conventions that
    decide whether your input means what you think it means.

    [:octicons-arrow-right-24: Quickstart](#quickstart)

-   :material-book-open-variant:{ .lg .middle } **Guide**

    ---

    Every keyword `rcwa()` takes, what it defaults to and what it refuses; near fields,
    autodiff, backends, and a run described as data.

    [:octicons-arrow-right-24: The scene, keyword by keyword](guide/scene.md)

-   :material-code-braces:{ .lg .middle } **Reference**

    ---

    The normative contract — every sign, axis, name and unit a script can depend on — and the
    API, generated from the docstrings.

    [:octicons-arrow-right-24: Conventions](conventions.md)

-   :material-flask-outline:{ .lg .middle } **Decisions**

    ---

    Why a default is what it is, and what was measured to decide it. Twenty-six write-ups,
    cited from the code they explain.

    [:octicons-arrow-right-24: Backend choice](backend-choice.md)

</div>

## Quickstart

```shell
pip install git+https://github.com/fieldalgo/wavebit.git
```

```python
import numpy as np
import wavebit

eps_hi = wavebit.eps_from_index(3.5)                     # 12.25 -- materials are permittivity
structure = np.array([[[1.0, 1.0, eps_hi, eps_hi]]])     # (layer, y, x)

solver = wavebit.rcwa(
    backend=0, wavelength=900., period=[700.],
    truncation_order=dict(x=10, y=0),
    top=1., bot=wavebit.eps_from_index(1.5),
    theta=0., phi=0., tetm='te',
    structure=structure, thickness=[300.],
)
result = solver.solve()
print(result.R.sum(), result.T.sum())    # 0.20533162825847726 0.7946683717415197
```

`R` and `T` are the reflected and transmitted diffraction efficiencies, one entry per order.
Here they sum to 1 to within a few times 1e-15, as a lossless structure must.

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
