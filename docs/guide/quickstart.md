# Quickstart

A binary grating, solved on the NumPy backend.

```python
import numpy as np
import wavebit

# One 300 nm layer, half air and half n = 3.5, on a glass substrate. Materials are relative
# permittivity, so an index off a datasheet goes through `eps_from_index` first.
eps_hi = wavebit.eps_from_index(3.5)                     # 12.25
structure = np.array([[[1.0, 1.0, eps_hi, eps_hi]]])     # (layer, y, x)

solver = wavebit.rcwa(
    backend=0,                                # 0 = NumPy, 1 = JAX, 2 = PyTorch
    wavelength=900., period=[700.],           # one shared length unit; nm here
    truncation_order=dict(x=10, y=0),         # named, so there is no order to recall
    top=1., bot=wavebit.eps_from_index(1.5),  # the two half-spaces, as materials
    theta=0., phi=0., tetm='te',
    structure=structure, thickness=[300.],
)

result = solver.solve()
print(result.R.sum(), result.T.sum())    # 0.20533162825847726 0.7946683717415197
print(result.order(0, 0).T)              # 0.3531011230787259 -- the zeroth transmitted order
```

## Reading the result

`R` and `T` are the reflected and transmitted diffraction efficiencies, **one entry per order**.
Here they sum to 1 to within a few times 1e-15, as a lossless structure must.

```python
result.R.shape                  # (n_orders_v, n_orders_u) -- v first
result.orders_u, result.orders_v    # which order each column and each row is
result.order(0, 0)              # index by order instead, and skip the layout entirely
result.r_s, result.t_p          # the complex Fresnel amplitudes
```

**Uppercase is power, lowercase is amplitude.** `R` / `T` are efficiencies; `r_s` / `r_p` /
`t_s` / `t_p` are the complex amplitudes, directly comparable to the textbook coefficients.

Every eager NumPy solve also checks what it hands back: a NaN or infinity raises
`FloatingPointError` naming the contaminated array, and a finite result that cannot be physical
warns `UnphysicalResultWarning`. If you are modelling gain (`Im(eps) < 0` here), `R + T` above
one is correct and the warning is a false positive — filter that category.

## What to read next

* [Conventions](../conventions.md) — the seven rows that decide whether your input means what
  you think it means. Each is legal, energy-conserving and *wrong* when read backwards.
* [Describing a scene](scene.md) — materials, the two modeling paths, and the cell.
* [Saying what the light is](light.md) — polarization, and arriving on more than one order.
* [Differentiating a solve](autodiff.md) — `jax.grad` works off the public API; `jax.jit` needs
  one more object.

Tutorials are runnable notebooks in the repository under
[`tutorials/`](https://github.com/fieldalgo/wavebit/tree/develop/tutorials);
[`01`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/01-modeling-and-emsolver.ipynb)
walks this same scene in full.
