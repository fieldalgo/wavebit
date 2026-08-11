<p align="center"><img src="images/fieldwave_logo.svg" width=45% alt="FieldWave" /></p>

# FieldWave

[![tests](https://github.com/fieldalgo/fieldwave/actions/workflows/tests.yml/badge.svg)](https://github.com/fieldalgo/fieldwave/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Differentiable electromagnetic simulation suite — by [FieldAlgo](https://fieldalgo.com)**

FieldWave simulates how light interacts with nanostructures, and differentiates through the
whole calculation — so the same code that predicts a device's response can also optimize it.

* **Modeling** — describe a structure parametrically or as a raster
* **EM simulation** — RCWA solver, rigorous and fully vectorial
* **Optimization** — gradients flow end to end, from geometry to diffraction efficiency

## How to install

FieldWave is not on PyPI yet. Install it from source:

```shell
pip install git+https://github.com/fieldalgo/fieldwave.git
```

Or, for a checkout you intend to edit:

```shell
git clone https://github.com/fieldalgo/fieldwave.git
cd fieldwave
pip install -e .
```

The base install needs only Python ≥ 3.10, NumPy ≥ 2.5.1, SciPy ≥ 1.18 and array-api-compat.
JAX and PyTorch are optional and bring autodiff and GPU support:

```shell
pip install -e '.[jax]'       # or '.[pytorch]', or '.[jax,pytorch]'
```

`environment.yml` builds the full development environment, including the tools the studies
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
print(result.R.sum(), result.T.sum())   # 0.2053316282584784 0.7946683717415217
```

`R` and `T` are the reflected and transmitted diffraction efficiencies, one entry per
order. Here they sum to 1 within 1e-15, as a lossless structure must.

`n_top` and `n_bot` may be complex: an absorbing substrate or superstrate is given as an index
like any other, with loss as `Im(n) > 0`, matching the bundled nk tables. `tests/test_lossy_media.py`
pins both against the closed-form multilayer answer. One caveat at the top boundary — incident
power in an absorbing medium is convention-dependent, so with a complex `n_top` the efficiencies
are not comparable to 1 and the amplitudes are what mean something.

The integer codes have named equivalents, if you prefer them to the literals:

```python
from fieldwave import Backend, Pol, Factorization

solver = fieldwave.rcwa(backend=Backend.JAX, pol=Pol.TM, factorization=Factorization.NVF, ...)
```

## Backends

One implementation of the physics sits behind a thin adapter and runs on all three, so a result
never depends on which you pick — cross-backend agreement is part of the test suite, not a
claim.

![NumPy, JAX and PyTorch](images/backends.png "Backends")

* [NumPy](https://github.com/numpy/numpy) — the baseline. No autodiff, no GPU, nothing to
  compile, and the only backend that checks its own output (see [Validation](#validation)).
* [JAX](https://github.com/google/jax) — autodiff and XLA. `jax.grad` works straight off the
  public API; `jax.jit` additionally needs a frozen geometry.
* [PyTorch](https://github.com/pytorch/pytorch) — autodiff eagerly, with no record to freeze
  and so none to go stale. `torch.compile` is not a supported path here.

### What each one can do

|                              | NumPy | JAX | PyTorch |
| ---------------------------- | :---: | :-: | :-----: |
| float64 / complex128         |   ●   |  ●  |    ●    |
| float32 / complex64          |   ●   |  ●  |    ●    |
| Automatic differentiation    |   ○   |  ●  |    ●    |
| GPU                          |   ○   | ●*  |   ●*    |
| In-process device parallelism |  ○   | ● (`pmap`) | ○ |
| TPU                          |   ○   |  ○  |    ○    |

\*The eigendecomposition has no GPU-native implementation in either framework: it is moved to
the CPU and the result sent back. Nothing here has been measured on a GPU, so read that column
as "the framework supports it", not as a performance claim. TPU is out entirely, because a
32-bit eigendecomposition has no route there at all.

That last row is about splitting *one* computation across devices, which only JAX does. It says
nothing about the parallelism a sweep wants: a wavelength, angle or parameter sweep is a set of
independent solves, and distributing those across processes or nodes — Ray, MPI, a Slurm array
— works identically on all three backends.

32-bit halves the memory and is not always safe: two numbers differing by eight or more digits
cancel silently — no warning, no error — and a convolution matrix routinely holds both. Reach
for it only once you have checked that your problem does not.

### Which one to run

Not a matter of taste, and not the same answer for a forward solve as for a differentiated one:

* **Forward solves, small to medium.** NumPy — lightest to install, and the one that warns when
  a result cannot be physical.
* **Differentiated, `fto ≲ 5`.** Eager PyTorch beats a jitted JAX step outright; JAX's
  compilation never repays itself at that size.
* **Differentiated, `fto ≳ 7`, long run.** A jitted JAX step is roughly 2x PyTorch and repays
  its compilation in under a dozen steps.

Where the crossover falls, and whether a freeze is worth it at all, is measured in
[docs/backend-choice.md](docs/backend-choice.md) — on one CPU host, with the run's own
repeatability printed next to the numbers.

## Describing a structure

The type of `ucell` selects the modeling path; there is no separate mode switch.

| `ucell` | Path | What it is |
| --- | --- | --- |
| `ndarray`, shape `(n_layers, Ny, Nx)` | **raster** | refractive index sampled on a grid |
| `list` of `[n_base, [shape, ...]]` per layer | **block** | drawn shapes reduced to piecewise-constant blocks, then Fourier-transformed analytically instead of off a grid — see [`examples/block_1d.py`](examples/block_1d.py) |
| `list` of `dict(n_base=..., shapes=[...])` per layer | **smooth** | smooth primitives (rotated rectangle, circle, ellipse) with exact Fourier coefficients — no staircase at all |

There is a fourth, **boundary**, which the `ucell` type alone cannot select: it is what a smooth
layer becomes when its shapes actually overlap and it asks for `combine='paint'`, and it is the
only path that keeps a tilted edge exact *and* paints. See
[Shapes that overlap](#shapes-that-overlap) below and
[`docs/boundary-modeling.md`](docs/boundary-modeling.md).

The smooth path takes one dict per layer. Note that `cx`, `cy`, `lx` and `ly` are absolute
lengths in the units of `period`, not fractions of it:

```python
ucell = [dict(n_base=1.0, shapes=[
    dict(type='circle', cx=350., cy=350., lx=200., n_index=3.5),
])]

solver = fieldwave.rcwa(backend=0, period=[700., 700.], ucell=ucell, ...)
```

### Shapes that overlap

The two shape paths do different things with an overlap. The block path **paints**, later
shape over earlier; the smooth path **superposes**, adding every shape's contribution, so
the overlap ends up as neither shape's material. Same recipe, two different structures.

`combine` is how a layer says which it means — `'paint'` or `'add'`:

```python
ucell = [dict(n_base=1.0, combine='add', shapes=[...])]     # superpose, in closed form
ucell = [dict(n_base=1.0, combine='paint', shapes=[...])]   # paint, edges still exact
ucell = [[1.0, [['rectangle', ...], ...], 'paint']]         # block path: paint, edges staircased
```

Both are built on the shape paths now. `'paint'` with an actual overlap resolves the painting
into polygons first and transforms those — so a layer can be painted *and* keep a tilted edge
exact, which no single path used to offer. See
[`docs/boundary-modeling.md`](docs/boundary-modeling.md). The block path still refuses
`'add'`: a staircase of blocks holds one index per block, so an overlap can only take one of
them.

While no two shapes intersect, both rules describe the same structure, so `combine` is
optional, nothing is refused, and both spellings give the same numbers. It only has to be
given where it changes the answer — and there, an unlabelled *smooth* layer is refused,
because superposition is the rule nobody expects. The block path paints unlabelled, which is
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

All four pairings work on the raster path, including `('integrated', 'nvf')` — the
staircase-free permittivity series *and* a factorization that handles boundaries not aligned
with x or y, which is what a fit of a tilted sidewall wants. Keeping the two as separate
keywords is what makes that pairing reachable.

The other two paths have one pairing each, because each states a layer as geometry and
integrates over it in closed form: the block path is `('integrated', 'li')` and the smooth path
`('integrated', 'nvf')`. Both keywords resolve to that on their own, so leaving them unset is
not a request and gets the build those paths always ran. Naming a value the path does not
implement raises `NotImplementedError` rather than quietly running the one it does — which is
what both keywords used to do, returning the other value's numbers with nothing in the result
to say so. The smooth path's derivation, convergence data and limitations are written up in
[docs/analytic-modeling.md](docs/analytic-modeling.md).

### Non-orthogonal unit cells

`period=(Lx, Ly)` describes a rectangular cell and is the default. A cell that is *not*
rectangular — hexagonal, or any oblique basis — is given as the two primitive vectors instead:

```python
a = 400.
a1, a2 = np.array([a, 0.]), np.array([a / 2, a * np.sqrt(3) / 2])     # hex

solver = fieldwave.rcwa(backend=0, lattice=np.column_stack((a1, a2)),   # columns are a1, a2
                        factorization='nvf', ucell=ucell, ...)
```

Setting `lattice` makes `period` report the basis-vector lengths `(|a1|, |a2|)`. A hexagonal
array spelled as an `a x a*sqrt(3)` rectangular supercell needs twice the cell area, hence twice
the retained orders, and the solve is `O(N^2.7)` — so the native cell is about **8x** cheaper on
a converged high-contrast structure (~3x on a lightly-truncated sweep).

**The raster is read in lattice coordinates.** `ucell[layer][j, i]` is the permittivity at
`r = (i + 0.5)/Nx * a1 + (j + 0.5)/Ny * a2`. Building a raster on a Cartesian grid and passing a
tilted basis therefore describes a *different structure*, silently — energy still conserves to
1e-15. Analytic shapes are immune: their lengths are absolute Cartesian quantities, so a
rectangle stays a rectangle under a skew.

Two build paths are refused on a skewed cell rather than answering, because both run and return
a plausible wrong number: `factorization='li'` on a raster (the inverse rule follows the raster
axes, which are `a1` / `a2` once skewed — repairing it is Li's covariant formulation) and the
block modeler. Everything else works, including analytic shapes with painted overlaps and field
reconstruction — whose output grid is then indexed by `(v, u)` rather than `(y, x)`.

The measurements, the validation against Li's published checkerboard table, and a worked metalens
phase library are in [docs/oblique-lattice.md](docs/oblique-lattice.md);
[`tutorials/05-oblique-lattice.ipynb`](tutorials/05-oblique-lattice.ipynb) is the walkthrough.

### Connecting algorithm

How per-layer modes are joined into a stack response. The default is the *enhanced*
transmittance-matrix method (Moharam et al. 1995) — the stabilised transfer, not the naive
one; the scattering-matrix method is available as an alternative:

```python
solver = fieldwave.rcwa(backend=0, connecting_algo='SMM', ...)   # 'ETM' is the default
```

Both share the same eigenmodes and boundary conventions and agree to ~1e-14; SMM costs
~1.1-1.4x, cascades layer responses that can be reused independently, and is the formulation
most published results are stated in. Field reconstruction (`calculate_field`) works from
either — the SMM recovers the per-layer amplitudes from its cascade afterwards, which is what
takes it to ~1.3-1.6x.

## Conventions

The full contract — every sign, axis, name and unit a script can depend on — is one page:
[`docs/conventions.md`](docs/conventions.md). The rows that most often decide whether your
input means what you think it means:

**Time convention.** FieldWave works in `e^{-iωt}` throughout — inputs, internals and
returned amplitudes alike. `Im(n) > 0` is loss, which is how refractive-index databases
tabulate it (`n + 1j*k`), and a wave travelling in `+z` carries `exp(+i*k0*kz*z)`. There is no
second convention underneath; see [`fieldwave/_core/conventions.py`](fieldwave/_core/conventions.py).

**Layers are listed top to bottom, and `+z` points into the stack.** `ucell[0]` is the layer
light meets first, arriving from `n_top`. Angles are radians: `theta` from the normal, `phi`
the azimuth of the plane of incidence, `psi` the polarization (`pi/2` = s/TE, `0` = p/TM).

**Uppercase is power, lowercase is amplitude.** `R`/`T` are per-order efficiencies, shape
`(ff_y, ff_x)` — `y` first, the transpose of the `fto=[x, y]` input order — with the incident
order at the centre; `result.order(m, n)` indexes by order so you never touch that layout.
`r_s`/`r_p`/`t_s`/`t_p` are the complex Fresnel amplitudes, directly comparable to the
textbook coefficients.

**Axes follow the array.** Row `0` of `ucell` is `y = 0`, for every builder and for the
reconstructed field cell alike. The flip that renders a field the conventional way up belongs
to plotting (`field_plot`), not to the data.

**Smooth-path shape coordinates are absolute.** On the smooth path, `cx`, `cy`, `lx` and `ly`
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
grad = jax.grad(lambda u: jnp.sum(solver.conv_solve(ucell=u).T))(ucell)
```

`jax.jit` is the harder bar, and that is what `freeze_geometry()` is for — it takes those
decisions once against the concrete structure and hands them back as data:

```python
geometry = solver.freeze_geometry()
step = jax.jit(lambda u: jnp.sum(solver.conv_solve(ucell=u, geometry=geometry).T))
```

A freeze describes the structure it was taken on. Sweeping the *material values* reuses it
exactly; moving the geometry does not, and an eager `conv_solve` raises `StaleGeometryError`
rather than answering from a record that no longer fits. A traced call cannot check — the
comparison is what the freeze exists to avoid.

So take the freeze again every step. On the block path it costs a few milliseconds and
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
  [`tests/references/`](tests/references). The CFS path agrees to ~1e-13 across twelve angle and
  polarization combinations; it is the only reference here that validates individual
  diffraction *orders* rather than totals.
* **Published benchmark tables** — Li's 1997 crossed-grating checkerboard, as tabulated by
  Demésy et al. (2012). All nine transmitted orders agree to a worst 2.06e-4, closer to the
  paper's FEM reference than the paper's own two methods sit to each other. It is posed on the
  oblique primitive cell, so it validates that path against numbers from outside this project.
* **Energy conservation** for lossless structures, at machine precision (|R+T−1| ~ 1e-15).
* **Cross-backend agreement** — NumPy, JAX and PyTorch on the same input, since one core
  implementation serves all three.
* **Analytic gradients against finite differences**, for the shape parameters the optimizer
  actually moves.

The suite lives in [`tests/`](tests) and runs in CI on every push.
Everything with a statable criterion lives there, including the slower cross-backend and
optimizer checks — there is no second, unasserted tier.

Every solve also checks what it is about to hand back. A NaN or infinity in `R` / `T`
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
in, answerable without reading what is already in each one, and says why the directories are
named as they are.

## Roadmap

FieldWave is being built as a suite rather than a single solver:

1. **RCWA on a single-source core** — one backend-agnostic implementation of the physics
   behind a thin adapter, so NumPy, JAX and PyTorch cannot drift apart. *(done)*
2. **FDTD** — a time-domain solver sharing the same geometry and material layers, enabling
   cross-validation of the same structure between methods.

## Citation

If you use FieldWave in your research, please cite the software:

```bibtex
@software{fieldwave,
    title  = {FieldWave: Differentiable Electromagnetic Simulation Suite},
    author = {Kim, Yongha},
    year   = {2026},
    url    = {https://github.com/fieldalgo/fieldwave},
}
```

## License

FieldWave is licensed under Apache-2.0. See [LICENSE](LICENSE), which also records the
third-party notices this project carries.

## Contact

[📩 FieldAlgo](mailto:dev@fieldalgo.com)
