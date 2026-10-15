# Backends, devices and precision

:::{note}
This is the complete compatibility reference for the earlier interface. Use `wavebit.compat.legacy` explicitly. Start new applications with the [current lifecycle](../guide/common/lifecycle.md) and [current API reference](../reference/api.md). Numerical evidence quoted here retains its original scope.
:::

One implementation of the physics sits behind a thin adapter and runs on all three backends, so
a result never depends on which you pick — cross-backend agreement is part of the test suite,
not a claim.

This chapter is about `rcwa()`. The [periodic-stack time-domain guide](../guide/fdtd/time-domain.md)
distinguishes the legacy NumPy default from the explicit-backend route. The separate
[current FDTD guide](../guide/fdtd/general-fdtd.md) covers the physical `fdtd.Problem`,
its separate numerical settings, and its NumPy/JAX/PyTorch CPU, double-precision execution.
The device and precision contracts below belong to the legacy RCWA factory.

![NumPy, JAX and PyTorch](../figures/backends.png "Backends")

* [NumPy](https://github.com/numpy/numpy) — the baseline. No autodiff, no GPU, nothing to
  compile.
* [JAX](https://github.com/google/jax) — autodiff and XLA. `jax.grad` works straight off the
  public API; `jax.jit` additionally needs a frozen geometry ([`autodiff.md`](../guide/common/autodiff.md)).
* [PyTorch](https://github.com/pytorch/pytorch) — autodiff eagerly, with no record to freeze
  and so none to go stale; `torch.autograd` works through `torch.func.vmap`, while the
  `torch.func` grad and jvp transforms are refused by name where their input reaches a modal
  operator, and nested ones wherever the operator carries a derivative (one over a thickness
  is exact) ([`autodiff.md`](autodiff-v1.md#forward-mode)).
  Under `torch.compile` wavebit runs eagerly inside the compiled caller, so its values and
  gradients are eager torch's; `fullgraph=True`, `torch.export` and `torch.jit.trace` are
  refused by name ([`../torch-compile.md`](../development/torch-compile.md)).

```python
solver = wavebit.rcwa(backend='numpy', ...)               # 'numpy', 'jax' or 'torch'
solver = wavebit.rcwa(backend=wavebit.Backend.JAX, ...)   # or the enum member, the same thing
```

The names are the libraries' import names, spelled exactly so: `'JAX'`, `'pytorch'` and the
integers `0` / `1` / `2` -- the spelling until 2026-10-11 -- are refused with the list of names.
`solver.backend` holds the `wavebit.Backend` member, which equals its name
(`solver.backend == 'jax'`).

## What each one can do

|                               | NumPy | JAX | PyTorch |
| ----------------------------- | :---: | :-: | :-----: |
| float64 / complex128          |   ●   |  ●  |    ●    |
| float32 / complex64           |   ●   |  ●  |    ●    |
| Automatic differentiation     |   ○   |  ●  |    ●    |
| GPU                           |   ○   | ●*  |   ●*    |
| In-process device parallelism |   ○   | ● (`pmap`) | ○ |
| TPU                           |   ○   |  ○  |    ○    |

\*The nonsymmetric eigendecomposition is the operation whose GPU story needs stating, and it has
moved recently — what follows is current as of 2026-08. `jax.lax.linalg.eig` chooses between
LAPACK on the host CPU, MAGMA on the GPU and cuSOLVER on the GPU, automatically by cuSOLVER
version, by whether left eigenvectors were asked for (this solver never asks) and by
`jax_use_magma`. PyTorch still documents that a CUDA input to `torch.linalg.eig` synchronizes
that device with the CPU, but 2.10 added a native `cusolverDnXgeev` path on CUDA 12.8 or newer —
both hold for every version this project supports (`jax>=0.11`, `torch>=2.13`). **None of those
paths has been run here** — nothing in this repository has been measured on a GPU — so read that
column as "the framework supports it", not as a performance claim. TPU is out entirely, because
a 32-bit eigendecomposition has no route there at all.

That `pmap` row is about splitting *one* computation across devices, which only JAX does. It
says nothing about the parallelism a sweep wants: a wavelength, angle or parameter sweep is a
set of independent solves, and distributing those across processes or nodes — Ray, MPI, a Slurm
array — works identically on all three backends. [`specs.md`](../guide/common/specs.md) is the shape a run takes
when it has to cross one of those boundaries.

## Asking for a device and a precision

Both keywords are written in wavebit's own vocabulary rather than a namespace's, which is what
lets one signature serve three backends:

```python
wavebit.rcwa(backend='jax', device=1, complex_dtype=1, ...)     # JAX, GPU, single precision
```

| | Accepts |
| --- | --- |
| `device` | `0` / `'cpu'`, `1` / `'gpu'` / `'cuda'`, or the backend's own device object. NumPy has one device and **refuses** a GPU request rather than printing about it |
| `complex_dtype` | `0` (double, the default) / `1` (single), or any of the three namespaces' complex spellings — `np.complex128`, `jnp.complex64`, `torch.complex128` |

32-bit halves the memory and is not always safe: two numbers differing by eight or more digits
cancel silently — no warning, no error — and a convolution matrix routinely holds both. Reach
for it only once you have checked that your problem does not. Evanescent (complex-angle)
incidence was the one case that failed outright at 32 bits until 2026-10-02. Both of its
branch decisions are now resolved there, the half-space root and the ETM layer labels: one
measured point is 1.9e-6 off double precision on ETM and 3.4e-6 on SMM, where it was 5.1e-2
on ETM. The solver warns `UnresolvedBranchWarning` when a branch was still read from
round-off, which takes a root sitting on its cut. See [`../branch-cuts.md`](../theory/rcwa/branch-cuts.md).

Tutorials [`03-jax`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/rcwa/03-device-and-datatype-jax.ipynb) and
[`03-torch`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/rcwa/03-device-and-datatype-torch.ipynb) work through both keywords.

## The JAX 64-bit cap, and import order

JAX caps arrays at 32 bits unless `jax_enable_x64` is on. It is a **cap, not a default**: with
it off, `jnp.asarray(x, jnp.float64)` still yields float32, so double precision cannot be opted
into at all. Several things already lift it, and none covers every ordering:

* `from wavebit.compat import legacy as wavebit` lifts it if JAX is **already** imported.
  Importing the new root package with `import wavebit` leaves this process-wide setting unchanged.
* The JAX RCWA backend lifts it when it is first **used**.
* A JAX FDTD run (`fdtd` or `fdtd_grid`) lifts it when it builds its time loop, and
  `meshgrid_xy(..., backend='jax')` before it samples — each before its first array.

The remaining compatibility gap is JAX arriving after the legacy module and being used before
the first solve — an array built in that window is silently `complex64` while `complex_dtype` still reads 128 everywhere. That
ordering once produced `|R| = 3.94` against a true 0.67 with every dtype reporting double.
`wavebit.enable_x64()` closes it when `wavebit` is the explicit legacy alias below:

```python
from wavebit.compat import legacy as wavebit
wavebit.enable_x64()          # before building any JAX array wavebit will be handed
import jax.numpy as jnp
theta = jnp.asarray(0.4 + 0.3j)        # complex128, not complex64
```

It is idempotent, and it **raises rather than quietly doing nothing** when JAX is not installed
— asking for double precision and silently not getting it is the failure the whole path exists
to stop. `PrecisionLossWarning` is what says so at runtime for the orderings nothing can reach.

## Which one to run

Not a matter of taste, and not the same answer for a forward solve as for a differentiated one:

| Situation | Backend |
| --- | --- |
| Forward solves, small to medium | **NumPy** — lightest to install |
| Differentiating a shape recipe | **jitted JAX** at every size measured — 2.3x PyTorch at `truncation_order=3`, 1.3x at 13 — repaying its compilation in 13 steps at worst |
| Differentiating a raster, `truncation_order ≲ 5` | **eager PyTorch**, outright; JAX's compilation does not repay itself at that size |
| Differentiating a raster, `truncation_order ≳ 7`, long run | **jitted JAX**, roughly 1.5x PyTorch, repaying compilation in under a dozen steps |

Before reaching for any of it on a shape recipe, look at `shape_nvf_grid_size`. At
`truncation_order=3` the default spends far more per step building a normal field than the
solve itself costs, and lowering it is worth more than any backend choice — bounded by
accuracy, not by taste, since it is a rectangle's *angle* that sets the grid.

All of it is measured in [`../backend-choice.md`](../validation/rcwa/backend-choice.md), on one CPU host, with
the run's own repeatability printed next to the numbers.

`torch.compile` is no speed choice. A compiled caller is exact: every public wavebit callable
keeps itself out of the graph (`torch.compiler.disable`), so wavebit runs eagerly inside it and
only your own arithmetic on its outputs is compiled. That is also why it buys wavebit's part
nothing — and it would buy little anyway, since the solve is almost entirely LAPACK calls the
compiler forwards untouched. Until 2026-10 nothing kept wavebit out of the graph, and default
mode returned wrong values and gradients with no error: Inductor (torch 2.13) miscompiled the
complex identity the modal exponential is built on. `fullgraph=True`, `torch.export` and
`torch.jit.trace` are refused by name. On torch below 2.13 compiling does not run at all, which
is why that is the floor. See [`../torch-compile.md`](../development/torch-compile.md).
