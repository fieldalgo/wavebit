<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/wavebit_logo_dark.svg">
    <img src="images/wavebit_logo.svg" width="45%" alt="Wavebit">
  </picture>
</p>

# Wavebit

[![tests](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml/badge.svg)](https://github.com/fieldalgo/wavebit/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

Wavebit simulates how light interacts with nanostructures, and differentiates through the whole
calculation — so the same code that predicts a device's response can also optimize it.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

One implementation of the physics runs on NumPy, JAX and PyTorch, so a result never depends on
which you pick — cross-backend agreement is part of the test suite rather than a claim.

## Install

Wavebit is not on PyPI yet. Install it from source:

```shell
pip install git+https://github.com/fieldalgo/wavebit.git
```

NumPy alone runs the solver. JAX and PyTorch are optional and bring autodiff and GPU support:

```shell
pip install "wavebit[jax] @ git+https://github.com/fieldalgo/wavebit.git"
```

For a checkout you intend to edit, or the full development environment:

```shell
git clone https://github.com/fieldalgo/wavebit.git && cd wavebit
pip install -e '.[jax]'                # or '.[pytorch]', or plain -e . for NumPy only
conda env create -f environment.yml    # everything, including what the studies need
```

Python ≥ 3.12. The version floors in [`pyproject.toml`](pyproject.toml) are the ones the suite
is verified against rather than the oldest that might work, and each is commented with why it
sits where it does.

## Quickstart

A binary grating, solved on the NumPy backend:

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
print(result.order(0, 0).T)              # 0.3531011230787259 — the zeroth transmitted order
```

`R` and `T` are the reflected and transmitted diffraction efficiencies, one entry per order.
Here they sum to 1 to within a few times 1e-15, as a lossless structure must.

[`tutorials/01-modeling-and-emsolver.ipynb`](tutorials/01-modeling-and-emsolver.ipynb) walks the
same scene in full, and the tutorials after it take one topic each.

## Conventions

Seven rows decide whether your input means what you think it means. Every one of them is legal,
energy-conserving and *wrong* when read backwards — no error, no warning — which is why they are
worth the two minutes. [`docs/conventions.md`](docs/conventions.md) is the normative contract.

| | |
| --- | --- |
| **Materials are relative permittivity** | not refractive index. `eps_from_index` / `index_from_eps` convert, and the retired index-shaped keywords (`ucell`, `n_top`, `n_bot`) are refused by name because the quantity moved with them |
| **Time convention is `e^{-iωt}`** | `Im(eps) > 0` is loss — equivalently `Im(n) > 0`, how nk databases tabulate it — and a wave travelling in `+z` carries `exp(+i*k0*kz*z)` |
| **Layers run top to bottom, `+z` points into the stack** | `structure[0]` is the layer light meets first, arriving from `top`. Row `0` of a layer is `y = 0` |
| **Uppercase is power, lowercase is amplitude** | `R` / `T` are per-order efficiencies; `r_s` / `r_p` / `t_s` / `t_p` are the complex Fresnel amplitudes |
| **Arrays run `v` first, written pairs run `x` first** | `truncation_order=[4, 1]` gives `R.shape == (3, 9)`. `result.order(m, n)` indexes by order and sidesteps the layout entirely |
| **The cell is centred at the origin** | a centred feature is `0` in a shape's `cx` / `cy`, in `reconstruct_field`'s `sample_x` / `sample_y` and in `wavebit.meshgrid_xy` alike. Shape lengths are absolute, not fractions of the period |
| **Units are yours** | `wavelength`, `period` and `thickness` only have to share one. Angles are radians |

## Differentiating a solve

`jax.grad` and PyTorch autograd work straight off the public API — a grad tracer carries a
concrete value, so the pipeline's geometric decisions read it one at a time:

```python
import jax, jax.numpy as jnp

solver = wavebit.rcwa(backend=1, structure=structure, ...)
grad = jax.grad(lambda u: jnp.sum(solver.solve(structure=u).T))(structure)
```

`jax.jit` and `vmap` are the harder bar, because those same decisions branch on values that a
jit tracer does not have. `freeze_geometry()` takes them once, against the concrete structure,
and hands them back as data:

```python
geometry = solver.freeze_geometry()
step = jax.jit(lambda u: jnp.sum(solver.solve(structure=u, geometry=geometry).T))
```

Re-take the freeze every step rather than holding one. It costs 0–1 ms, compares by content so
an unchanged record is free, and it is the only form that cannot carry a stale one forward.
Which backend to use, and whether to jit at all, differs by modeling path and is measured in
[`docs/backend-choice.md`](docs/backend-choice.md).

## Validation

The physics is checked against references, not only against itself:

* **Three outside solvers** — RETICOLO through an Octave bridge, plus fmmax and S4. Their
  numbers are committed as goldens and checked in CI, so the half that needs software this
  project does not ship runs once, by hand. RETICOLO agrees to ~1e-13 and is the only reference
  here that validates individual diffraction *orders* rather than totals.
* **Published benchmark tables** — Li's 1997 crossed-grating checkerboard as tabulated by
  Demésy et al. (2012). All nine transmitted orders agree to a worst 2.06e-4, on the oblique
  primitive cell, which puts that path against numbers from outside this project.
* **Energy conservation** for lossless structures, at machine precision (|R+T−1| ~ 1e-15).
* **Cross-backend agreement** — NumPy, JAX and PyTorch on the same input.
* **Analytic gradients against finite differences**, for the parameters an optimizer moves.

The suite is [`tests/`](tests) and runs in CI on every push; everything with a statable
criterion lives there, including the slower cross-backend and optimizer checks. Every eager
NumPy solve also checks what it is about to hand back — a NaN or infinity raises
`FloatingPointError` naming the contaminated array, and a finite result that cannot be physical
warns `UnphysicalResultWarning`.

## Documentation

| | |
| --- | --- |
| [`docs/guide/`](docs/guide) | The manual: every keyword, what it defaults to and what it refuses — scene, backends, near fields, autodiff, and a run described as data |
| [`tutorials/`](tutorials) | Lessons in order, for someone who has not used the API yet: a first solve, autograd and optimization, device and precision, NVF factorization, oblique cells, polarization, multimode incidence |
| [`docs/`](docs) | Why a default is what it is, and what was measured to decide it. [`conventions.md`](docs/conventions.md) is the normative contract; [`docs/README.md`](docs/README.md) maps the rest |
| [`examples/`](examples) | Wavebit applied to outside problems |
| [`studies/`](studies) | The scripts behind the measurements in `docs/` |
| `python -m wavebit.manifest` | Every option, refusal and constraint as one generated JSON document, for a program that needs to know without reading the source ([`docs/capability-manifest.md`](docs/capability-manifest.md)) |

## Roadmap

Wavebit is being built as a suite rather than a single solver:

1. **RCWA on a single-source core** — one backend-agnostic implementation of the physics behind
   a thin adapter, so NumPy, JAX and PyTorch cannot drift apart. *(done)*
2. **FDTD** — a time-domain solver sharing the same geometry and material layers, enabling
   cross-validation of the same structure between methods.

## Contributing

Bug reports, questions and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

**Adding a file? Start there.** It opens with the rule for which directory a new file belongs
in, answerable without reading what is already in each one.

## Citation

If you use Wavebit in your research, please cite the software:

```bibtex
@software{wavebit,
    title  = {Wavebit: Differentiable Electromagnetic Simulation Suite},
    author = {Kim, Yongha},
    year   = {2026},
    url    = {https://github.com/fieldalgo/wavebit},
}
```

## License

Wavebit is licensed under Apache-2.0. See [LICENSE](LICENSE), which also records the
third-party notices this project carries.

## Contact

[📩 FieldAlgo](mailto:dev@fieldalgo.com)
