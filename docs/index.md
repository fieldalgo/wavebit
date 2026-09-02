# Wavebit

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

Wavebit simulates how light interacts with nanostructures, and differentiates through the whole
calculation — so the same code that predicts a device's response can also optimize it. One
implementation of the physics runs on NumPy, JAX and PyTorch, so a result never depends on which
you pick.

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

## Three doorways

| | For a reader who | Start at |
| --- | --- | --- |
| **Guide** | knows the API exists and needs the contract for one keyword | [The scene, keyword by keyword](guide/scene.md) |
| **Reference** | needs the normative contract, or a signature | [Conventions](conventions.md) · [API](reference/api.md) |
| **Decisions** | is questioning a default and wants the measurement | [Backend choice](backend-choice.md) is a good sample |

Tutorials — lessons in order, for someone who has not used the API yet — live in the
repository under `tutorials/` as runnable notebooks.

## The seven conventions

Each is legal, energy-conserving and *wrong* when read backwards — no error, no warning.
[`conventions.md`](conventions.md) is the normative contract.

* **Materials are relative permittivity**, not refractive index.
* **Time convention is `e^{-iωt}`**; `Im(eps) > 0` is loss.
* **Layers run top to bottom**, and `+z` points into the stack.
* **Uppercase is power, lowercase is amplitude** — `R`/`T` against `r_s`/`t_p`.
* **Arrays run `v` first, written pairs run `x` first.**
* **The cell is centred at the origin**, and shape lengths are absolute.
* **Units are yours** — `wavelength`, `period` and `thickness` share one.
