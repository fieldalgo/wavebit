# Quickstart

A binary grating, solved on the NumPy backend.

```python
import numpy as np
import wavebit

# One 300 nm layer, half air (epsilon_r = 1) and half dielectric (epsilon_r = 12.25),
# on a glass substrate (epsilon_r = 2.25). Enter relative permittivity directly.
eps_hi = 12.25
structure = np.array([[[1.0, 1.0, eps_hi, eps_hi]]])     # (layer, y, x)

solver = wavebit.rcwa(
    backend='numpy',                          # or 'jax' / 'torch'
    wavelength=900., period=[700.],           # one shared length unit; nm here
    truncation_order=dict(x=10, y=0),         # named, so there is no order to recall
    top=1., bot=2.25,  # the two half-spaces, as materials
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

`result.absorption` is the absorbed fraction in all finite layers together;
it excludes absorption in `top` and `bot`. For a lossless incident medium it
equals `1 - result.R.sum() - result.T.sum()`. For an absorbing incident medium,
use `result.absorption`, which includes incident/reflected interference through
`result.power_entering - result.T.sum()`. These power definitions require a real
transverse wavevector; `power_entering` and `absorption` are `NaN` otherwise.

```python
result.R.shape                  # (n_orders_v, n_orders_u) -- v first
result.orders_u, result.orders_v    # which order each column and each row is
result.order(0, 0)              # index by order instead, and skip the layout entirely
result.r_s, result.t_p         # the complex Fresnel amplitudes
result.r_xyz                   # each order's outgoing E as (Ex, Ey, Ez); no per-order frame
result.power_entering          # net entering power / incident-only power
result.absorption              # absorption in the finite stack / incident-only power
```

**Uppercase is power, lowercase is amplitude.** `R` / `T` are efficiencies; `r_s` / `r_p` /
`t_s` / `t_p` are the complex amplitudes, directly comparable to the textbook coefficients.
The direct amplitude properties read `result.incident`, the response to the actual
incident state. For example, `result.r_s` is the same array as `result.incident.r_s`.

Every eager NumPy solve also checks what it hands back: a NaN or infinity raises
`FloatingPointError` naming the contaminated array. There is deliberately no generic
"`R + T` above one" warning, because that can be the right answer for a medium with gain
(`Im(eps) < 0` here) — see [`result-guards.md`](../result-guards.md).

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
