# Differentiating a solve

Gradients flow end to end, from the geometry to the diffraction efficiency. `jax.grad` and
PyTorch autograd need nothing from you; `jax.jit` and `vmap` need one extra object.

## The gradient works off the public API

```python
import jax, jax.numpy as jnp

solver = wavebit.rcwa(backend=1, structure=structure, ...)
grad = jax.grad(lambda u: jnp.sum(solver.solve(structure=u).T))(structure)
```

and on PyTorch, eagerly:

```python
u = torch.tensor(raster, dtype=torch.float64, requires_grad=True)
solver = wavebit.rcwa(backend=2, structure=u, ...)
solver.solve().T.sum().backward()          # u.grad
```

The two agree: on the quickstart grating both report a gradient of norm `0.0802694256415…`,
matching to ~1e-15.

What is differentiable is not only the raster. A **`Material`** is a frozen pytree whose
children are `(epsilon, mu)`, so `jax.grad` reaches inside one; and a **`Lattice`** stores its
basis rather than an angle, so building one inside the differentiated function makes the cell's
own shape a design variable — the off-diagonals are the shear, which no spelling of `period`
can express:

```python
def loss(angle):
    A = wavebit.Lattice.oblique(400., 400., angle)
    solver = wavebit.rcwa(backend=1, lattice=A, structure=structure, ...)
    return jnp.sum(solver.solve().T)

jax.grad(loss)(jnp.asarray(np.radians(105.)))
```

The two modeling paths do not differentiate the same question there. A raster is held in
lattice coordinates, so its basis gradient tilts the cell **and shears the pattern with it**,
where a parametric shape's absolute Cartesian lengths tilt the cell only. They differ by 3.6x on
a plain scene — see [`../oblique-lattice.md`](../oblique-lattice.md), "Differentiating the
cell".

## Why `jax.jit` needs more

One step in the raster pipeline decides something structural rather than computing a value —
which way the material interfaces point. It branches on values, so it does not survive a jit
tracer, which carries no value to branch on.

A **grad** tracer does carry a concrete value, which is why the block above needs nothing: the
decisions read it one at a time, at the point the gradient is being taken, and a decision taken
there is one that cannot go stale.

`freeze_geometry()` takes those decisions once, against the concrete structure, and hands them
back as data:

```python
geometry = solver.freeze_geometry()          # a StaticGeometry
step = jax.jit(lambda u: jnp.sum(solver.solve(structure=u, geometry=geometry).T))
```

You need a freeze only under `jax.jit` and `vmap`. PyTorch has no record to freeze, and so none
to go stale.

## Re-take the freeze every step

A freeze describes the structure it was taken on. Sweeping the *material values* reuses it
exactly; moving the geometry does not, and a `solve` that can read its values raises
`StaleGeometryError` rather than answering from a record that no longer fits. **Under `jax.jit`
there is nothing to compare against and that check cannot run** — which is the reason to re-take
the freeze every step rather than hold one:

```python
for _ in range(n_steps):
    geometry = solver.freeze_geometry()      # 0-1 ms; compares by content, so unchanged is free
    value, grads = jax.value_and_grad(step)(params, geometry)
```

The record compares by content, so an unchanged one is free and a jitted step recompiles only
when the recorded *combinatorics* change — which is the same moment a held record would have
gone stale. That makes re-freezing both the fastest way to run a moving fit and the only form
that cannot carry a stale record forward.

The painted-shape path reads "moving the geometry" more narrowly, because its record holds
combinatorics rather than coordinates: a shape sliding is what it replays, bit for bit, and only
a change in *which shape covers which* makes it stale. See
[`../boundary-modeling.md`](../boundary-modeling.md) for which motions those are.

A freeze also records the checks the geometry passed. A record taken over a layer that would be
refused replays the refusal, rather than letting a frozen scene through a door the live one
cannot pass.

## What cannot be traced

| | Why |
| --- | --- |
| An **explicit `sample_z`** in `reconstruct_field`, with respect to `thickness` | bucketing a `z` into a layer needs the thicknesses as numbers. It raises rather than differentiating the wrong geometry; counts stay traceable |
| `torch.compile` | not a supported path — the solve is almost entirely LAPACK calls the compiler forwards untouched, and below torch 2.13 it does not run at all ([`../torch-compile.md`](../torch-compile.md)) |

One trap worth naming because it passes: **build the solver outside the trace.** A jit test
that constructs the solver *inside* the traced function never flattens it as a pytree, so a
keyword can be silently dropped on every jitted solve while a test named for that keyword
passes.

## Which backend, and whether to jit at all

Not a matter of taste, and the answer differs by modeling path — a jitted JAX step wins at
every size on a shape recipe, while on a raster below `truncation_order ~ 5` eager PyTorch beats
it outright. Before choosing either on a shape recipe, look at `shape_nvf_grid_size`, which outweighs
both. The table is in [`backends.md`](backends.md); the measurements are in
[`../backend-choice.md`](../backend-choice.md).

Tutorials [`02-jax`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/02-autograd-and-optimization-jax.ipynb) and
[`02-pytorch`](https://github.com/fieldalgo/wavebit/blob/develop/tutorials/02-autograd-and-optimization-pytorch.ipynb) run a full
optimization on each backend.
