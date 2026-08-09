<p align="center"><img src="images/fieldwave_logo.svg" width=45% alt="FieldWave" /></p>

# FieldWave

[![tests](https://github.com/FieldAlgo/fieldwave/actions/workflows/tests.yml/badge.svg)](https://github.com/FieldAlgo/fieldwave/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

FieldWave simulates how light interacts with nanostructures, and differentiates through the
whole calculation — so the same code that predicts a device's response can also optimize it.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

## Lineage

FieldWave is the successor to [meent](https://github.com/kc-ml2/meent), continued by its
original lead author at FieldAlgo. It is derived from meent (MIT License, © 2022 ML2) and is
developed independently under Apache-2.0. If you are coming from meent, see
[Migrating from meent](#migrating-from-meent) — the API was renamed and one geometry
convention changed.

## How to install

FieldWave is not on PyPI yet. Install it from source:

```shell
pip install git+https://github.com/FieldAlgo/fieldwave.git
```

Or, for a checkout you intend to edit:

```shell
git clone https://github.com/FieldAlgo/fieldwave.git
cd fieldwave
pip install -e .
```

The base install needs only Python ≥ 3.10, NumPy ≥ 2.5.1, SciPy ≥ 1.18 and array-api-compat.
JAX and PyTorch are optional and bring autodiff and GPU support:

```shell
pip install -e '.[jax]'       # or '.[pytorch]', or '.[jax,pytorch]'
```

`environment.yml` builds the full development environment, including the tools the benchmarks
need:

```shell
conda env create -f environment.yml   # run from the repo root
conda activate fieldwave
```

> The floors are deliberately high: they are the versions the suite is actually verified
> against, not the oldest that might work. Loosening one later is safe; tightening one after a
> release is not. Two of them also sit above a known defect — NumPy 2.2.5 through 2.3.0
> mis-report the FP status of a complex `matmul`, printing hundreds of
> `RuntimeWarning: ... encountered in matmul` while returning correct efficiencies, and torch
> below 2.13 cannot compile the solver at all (see [docs/torch-compile.md](docs/torch-compile.md)).
> If you need an older one, verify it and the floor can come down.

## Quickstart

A binary grating, solved on the NumPy backend:

```python
import numpy as np
import fieldwave

# One 300 nm layer, half air and half n = 3.5, on a glass substrate.
ucell = np.array([[[1.0, 1.0, 3.5, 3.5]]])   # (layer, y, x) — refractive index

solver = fieldwave.rcwa(
    backend=0,                  # 0 = NumPy, 1 = JAX, 2 = PyTorch
    wavelength=900.,            # every length shares one unit; nm here
    period=[700.],
    fto=[10, 0],                # Fourier truncation order (x, y)
    n_top=1., n_bot=1.5,
    theta=0., phi=0., pol=0.,   # pol: 0 = TE, 1 = TM
    ucell=ucell, thickness=[300.],
)

result = solver.conv_solve()
print(result.de_ri.sum(), result.de_ti.sum())   # 0.20538982716989146 0.7946101728301077
```

`de_ri` and `de_ti` are the reflected and transmitted diffraction efficiencies, one entry per
order. Here they sum to 1 within 1e-15, as a lossless structure must.

The integer codes have named equivalents, if you prefer them to the literals:

```python
from fieldwave import Backend, Pol, Factorization

solver = fieldwave.rcwa(backend=Backend.JAX, pol=Pol.TM, factorization=Factorization.NVF, ...)
```

## Backends

One implementation runs on all three, so a result never depends on which you pick:

![alt text](images/backends.png "Backends")

* [NumPy](https://github.com/numpy/numpy) — the fundamental package for scientific computing with Python; easy and lean to use
* [JAX](https://github.com/google/jax) — autograd and XLA, brought together for high-performance machine learning research
* [PyTorch](https://github.com/pytorch/pytorch) — tensor computation with strong GPU acceleration and neural networks built on a tape-based autograd system

### When to use

|                 | NumPy | JAX | PyTorch | Description |
| --------------- | :---: | :-: | :-----: | :---------: |
| 64bit support   |   O   |  O  |    O    | Default for scientific computing |
| 32bit support   |   O   |  O  |    O    | 32bit (float32 and complex64) data type operation* |
| GPU support     |   X   |  O  |    O    | except Eigendecomposition** |
| TPU support     |   X   |  X  |    X    | Currently there is no workaround to do 32 bit eigendecomposition on TPU |
| AD support      |   X   |  O  |    O    | Automatic Differentiation (Back Propagation) |
| Parallelization |   X   |  O  |    X    | JAX pmap function |

*In 32bit operation, operations on numbers of 8>= digit difference fail without warning or error.
Use only when you do understand what you are doing.
**GPU-native eigendecomposition was unavailable in JAX and PyTorch when this was last checked
(2023-03); it is forced onto the CPU and sent back to the GPU. Re-verify against the versions
you run — it is the one place where the backend choice can dominate the runtime.

For a run that is *differentiated*, the table above is not the whole story — eager PyTorch and
a jitted JAX step trade places with problem size, and a frozen geometry is worth its
compilation only above a certain `fto`. That is measured in
[docs/backend-choice.md](docs/backend-choice.md).

NumPy is simple and light to use. Suggested as a baseline with small ~ medium scale optics problem.
JAX and PyTorch are recommended for cases having large scale or optimization part.
If you want parallelized computing with multiple devices (e.g., GPUs), JAX is ready for that.
But since JAX does jit compilation, it takes much time at the first run.

## Describing a structure

The type of `ucell` selects the modeling path; there is no separate mode switch.

| `ucell` | Path | What it is |
| --- | --- | --- |
| `ndarray`, shape `(n_layers, Ny, Nx)` | **raster** | refractive index sampled on a grid |
| `list` of `[n_base, [shape, ...]]` per layer | **vector** | drawn shapes reduced to piecewise-constant blocks, then Fourier-transformed analytically instead of off a grid — see [`examples/vector_1d.py`](examples/vector_1d.py) |
| `list` of `dict(n_base=..., shapes=[...])` per layer | **analytic** | smooth primitives (rotated rectangle, circle, ellipse) with exact Fourier coefficients — no staircase at all |

The analytic path takes one dict per layer. Note that `cx`, `cy`, `lx` and `ly` are absolute
lengths in the units of `period`, not fractions of it:

```python
ucell = [dict(n_base=1.0, shapes=[
    dict(type='circle', cx=350., cy=350., lx=200., n_index=3.5),
])]

solver = fieldwave.rcwa(backend=0, period=[700., 700.], ucell=ucell, ...)
```

### Shapes that overlap

The two shape paths do different things with an overlap. The vector path **paints**, later
shape over earlier; the analytic path **superposes**, adding every shape's contribution, so
the overlap ends up as neither shape's material. Same recipe, two different structures.

`combine` is how a layer says which it means — `'paint'` or `'add'`:

```python
ucell = [dict(n_base=1.0, combine='add', shapes=[...])]     # superpose, in closed form
ucell = [dict(n_base=1.0, combine='paint', shapes=[...])]   # paint, edges still exact
ucell = [[1.0, [['rectangle', ...], ...], 'paint']]         # vector: paint, edges staircased
```

Both are built on the shape paths now. `'paint'` with an actual overlap resolves the painting
into polygons first and transforms those — so a layer can be painted *and* keep a tilted edge
exact, which no single path used to offer. See
[`docs/boundary-modeling.md`](docs/boundary-modeling.md). The vector path still refuses
`'add'`: a staircase of blocks holds one index per block, so an overlap can only take one of
them.

While no two shapes intersect, both rules describe the same structure, so `combine` is
optional, nothing is refused, and both spellings give the same numbers. It only has to be
given where it changes the answer — and there, an unlabelled *analytic* layer is refused,
because superposition is the rule nobody expects. The vector path paints unlabelled, which is
what it always did and what a raster of the same recipe agrees with.

### Fourier factorization

For the raster path, two independent keywords decide how the permittivity becomes convolution
matrices. They change how fast the result converges with `fto`, not what it converges to:

```python
solver = fieldwave.rcwa(backend=0, fourier_coefficients='sampled', factorization='li', ...)
```

`fourier_coefficients` — how a layer's Fourier coefficients are obtained:

| Value | Notes |
| --- | --- |
| `'sampled'` | Discrete series straight off the pixel grid, with the raster's coefficients in closed form. The default. |
| `'integrated'` | Closed-form transform of the raster's rectangles. The more accurate route, and the one Reticolo uses. |

`factorization` — which rule closes the products of discontinuous functions:

| Value | Notes |
| --- | --- |
| `'li'` | Li's rules, applied along x and y. The default. |
| `'nvf'` | Normal vector field — factorizes along the material interfaces. Made for curved and tilted boundaries, where a staircase converges slowly; how it reads those interfaces off a raster, and what that costs, is in [docs/nvf-upsampling.md](docs/nvf-upsampling.md). |

All four pairings work. The deprecated `fourier_type` integer expressed only three of them —
`0 = ('sampled', 'li')`, `1 = ('integrated', 'li')`, `2 = ('sampled', 'nvf')` — and had no
code at all for `('integrated', 'nvf')`, the pairing a fit of a tilted sidewall wants.

The analytic path always uses the NVF form and ignores both keywords. Its derivation,
convergence data and limitations are written up in
[docs/analytic-modeling.md](docs/analytic-modeling.md).

### Connecting algorithm

How per-layer modes are joined into a stack response. The default is the transfer-matrix
method; the scattering-matrix method is available as an alternative:

```python
solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', ...)   # 'TMM' is the default
```

Both share the same eigenmodes and boundary conventions and agree to ~1e-14; SMM costs
~1.1-1.4x, cascades layer responses that can be reused independently, and is the formulation
most published results are stated in. Field reconstruction (`calculate_field`) needs the TMM
path.

## Conventions

Three of these decide whether your input means what you think it means. They are worth
reading once.

**Time convention.** FieldWave works in `e^{-iωt}` throughout — inputs, internals and
returned amplitudes alike. `Im(n) > 0` is loss, which is how refractive-index databases
tabulate it (`n + 1j*k`), and a wave travelling in `+z` carries `exp(+i*k0*kz*z)`. There is no
second convention underneath; see [`fieldwave/_core/conventions.py`](fieldwave/_core/conventions.py).

**Axes follow the array.** Row `0` of `ucell` is `y = 0`, for every builder and for the
reconstructed field cell alike. The flip that renders a field the conventional way up belongs
to plotting (`field_plot`), not to the data.

**Analytic shape coordinates are absolute.** On the analytic path, `cx`, `cy`, `lx` and `ly`
are lengths in the same units as `period` — not fractions of it. Passing fractions builds a
structure `1/period` times too small and solves it without complaint.

**Units are yours.** `wavelength`, `period` and `thickness` only have to share one unit. The
quickstart uses nanometres; nothing in the solver assumes it.

## Differentiating a simulation

Two steps in the raster pipeline decide something structural rather than compute a value —
which way the interfaces point, and where the piecewise-constant blocks start and stop. Both
branch on values, so neither survives `jax.jit`.

Neither of them needs help to survive `jax.grad`, though. A grad tracer carries a concrete
value, so the decisions read it one at a time and a gradient works straight off the public API:

```python
import jax, jax.numpy as jnp

solver = fieldwave.rcwa(backend=1, ucell=ucell, ...)
grad = jax.grad(lambda u: jnp.sum(solver.conv_solve(ucell=u).de_ti))(ucell)
```

`jax.jit` is the harder bar, and that is what `freeze_geometry()` is for — it takes those
decisions once against the concrete structure and hands them back as data:

```python
geometry = solver.freeze_geometry()
step = jax.jit(lambda u: jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).de_ti))
```

A freeze describes the structure it was taken on. Sweeping the *material values* reuses it
exactly; moving the geometry does not, and an eager `conv_solve` raises `StaleGeometryError`
rather than answering from a record that no longer fits. A traced call cannot check — the
comparison is what the freeze exists to avoid.

So take the freeze again every step. On the vector path it costs a few milliseconds and
recompiles only when the recorded *ordering* changes, which is the same moment a held record
would have gone stale — a length can move 47% for two compilations. That makes re-freezing both
the fastest way to run a moving fit and the only one that cannot carry a stale record forward.

Which backend to use, and whether to jit at all, is not obvious and is not a matter of taste:
below `fto ~ 5` eager PyTorch beats a jitted JAX step outright, and the crossover moves with
problem size. Measurements are in [docs/backend-choice.md](docs/backend-choice.md).

On PyTorch, eager and `.backward()` are the supported path. `torch.compile` is not: it buys
nothing here — the solve is almost entirely LAPACK calls the compiler forwards untouched — and
on torch below 2.13 it does not even run, which is why that is the floor. Measurements and the
full diagnosis are in [docs/torch-compile.md](docs/torch-compile.md).

## Validation

The physics is checked against references, not only against itself:

* **Reticolo**, the established RCWA implementation, through an Octave bridge in
  [`benchmarks/`](benchmarks). The CFS path agrees to ~1e-13 across twelve angle and
  polarization combinations; it is the only reference here that validates individual
  diffraction *orders* rather than totals.
* **Energy conservation** for lossless structures, at machine precision (|R+T−1| ~ 1e-15).
* **Cross-backend agreement** — NumPy, JAX and PyTorch on the same input, since one core
  implementation serves all three.
* **Analytic gradients against finite differences**, for the shape parameters the optimizer
  actually moves.

The suite lives in [`tests_verification/`](tests_verification) and runs in CI on every push.
Everything with a statable criterion lives there, including the slower cross-backend and
optimizer checks — there is no second, unasserted tier.

Every solve also checks what it is about to hand back. A NaN or infinity in `de_ri` / `de_ti`
raises `FloatingPointError` naming the contaminated array; a result that stayed finite but
cannot be physical — a negative efficiency, or R+T above 1 out of a passive stack — warns
`UnphysicalResultWarning`. Both are eager-NumPy only, so JAX and PyTorch tracing is untouched.
If you are modelling gain (`Im(n) < 0` here), R+T above 1 is correct and the warning is a
false positive — filter that category.

## Tutorials

Jupyter notebooks are prepared in [tutorials](tutorials) to give a brief introduction.

## Examples

Comprehensive examples of computational optics with FieldWave can be found in the
[`examples`](examples) folder.

## Contributing

Bug reports, questions and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).

**Adding a file? Start there.** It opens with the rule for which directory a new file belongs
in, answerable without reading what is already in each one. The reasoning, and the renames still
planned, are in [`docs/repository-layout.md`](docs/repository-layout.md).

## Migrating from meent

The physics carries over; the names changed, and one convention did.

| meent | FieldWave |
| --- | --- |
| `import meent` | `import fieldwave` |
| `meent.call_mee(backend=...)` | `fieldwave.rcwa(backend=...)` |
| `MeeNumpy` / `MeeJax` / `MeeTorch` | `SolverNumpy` / `SolverJax` / `SolverTorch` |
| `fourier_type=0` / `=1` / `=2` | `fourier_coefficients` + `factorization` (see above) |
| MIT | Apache-2.0 |

`fourier_type` still works and still means what it did, with a `DeprecationWarning`; it split
because one integer was standing for two independent choices.

Everything else — `conv_solve`, `ucell`, `thickness`, `fto`, `period`, `pol`,
`connecting_algo`, the `de_ri` / `de_ti` results — keeps its name and meaning.

One result does change. meent's CFS path read `ucell` row `0` as the *last* y while its other
paths read it as the first, so the same array described two different structures depending on
meent's `fourier_type`. At `phi = 0` this is invisible; away from it, CFS was ~1e-2 off
Reticolo. FieldWave reads row `0` as `y = 0` everywhere, and CFS now agrees to ~1e-13. If you
are reproducing a meent result at oblique azimuth with meent's `fourier_type=1` —
`fourier_coefficients='integrated'` here — expect a difference: this one is the fix, not a
regression.

## Roadmap

FieldWave is being built as a suite rather than a single solver:

1. **RCWA on a single-source core** — one backend-agnostic implementation of the physics
   behind a thin adapter, so NumPy, JAX and PyTorch cannot drift apart. *(done)*
2. **FDTD** — a time-domain solver sharing the same geometry and material layers, enabling
   cross-validation of the same structure between methods.

## Citation

FieldWave builds on the work published as meent. If you use FieldWave in your research,
please cite:

```bibtex
@article{kim2024meent,
    title={Meent: Differentiable Electromagnetic Simulator for Machine Learning},
    author={Kim, Yongha and Jung, Anthony W. and Kim, Sanmun and
            Octavian, Kevin and Heo, Doyoung and Park, Chaejin and
            Shin, Jeongmin and Nam, Sunghyun and Park, Chanhyung and
            Park, Juho and Han, Sangjun and Lee, Jinmyoung and
            Kim, Seolho and Jang, Min Seok and Park, Chan Y.},
    journal={arXiv preprint arXiv:2406.12904},
    year={2024}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Contact

[📩 FieldAlgo](mailto:dev@fieldalgo.com)
