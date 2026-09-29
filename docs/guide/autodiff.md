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

JAX's `solver.grad(parameter_names, forward, loss_fn)` also restores the complete
parameter declarations after evaluation, including tabulated materials and default
polarization. It releases the temporary solve state, even if the forward or loss
raises. Use a saved result for earlier fields, or call `solve()` again before reading
fields through the solver. The loss itself can still reconstruct the current field.

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

## Let `fit` prepare each design before JIT

A freeze describes the structure it was taken on. A material sweep that preserves its phase
interfaces can reuse it; moving interfaces or merging phases can require a new freeze.
A `solve` that can read its values raises
`StaleGeometryError` rather than answering from a record that no longer fits. **Under `jax.jit`
there is nothing to compare against and that check cannot run**. JAX's `solver.fit` therefore
prepares the current geometry outside JIT at every iteration, then passes it explicitly to the
compiled value/gradient and optimizer step. Pass `solver.solve`, `solver.solve_field` or
`solver.solve_field_eager` as `forward`, or a `functools.partial` of one of those methods. Fixed
keyword conditions in a partial are included in preparation; an old `geometry=` in a partial
is replaced each step.

The existing attribute-name API needs no manual freeze:

```python
import optax

best = solver.fit(['structure', 'thickness'], solver.solve,
                  lambda result: -result.T.sum().real,
                  optax.adam(1e-2), iteration=20)
```

For shape optimization, pass initial numeric design parameters and a pure, differentiable
`parameterize` function that maps them to solver conditions. Shape type strings and other
declarations belong in that function, not in the differentiable parameter tree:

```python
def design(width):
    return dict(structure=[dict(background=1., shapes=[
        dict(type='rectangle', cx=0., cy=0., lx=width, ly=190.,
             angle=0., material=4.)])])

initial = dict(width=jnp.asarray(230., dtype=jnp.float64))
solver = wavebit.rcwa(backend=1, period=(700., 700.), thickness=(150.,),
                     wavelength=850., truncation_order=(2, 2), **design(**initial))
best = solver.fit(initial, solver.solve, lambda result: -result.T.sum().real,
                  optax.adam(1.), iteration=20, parameterize=design)
```

`fit` returns the final design parameters and leaves their mapped conditions on the solver.
It evaluates `design` on concrete parameters to prepare geometry, and again inside the
differentiated calculation to retain shape derivatives. The record compares by content:
equal records reuse the compiled step, while a changed record requires recompilation. For
array-bearing raster/tensor records, `fit` compares every field with the preceding record and
retains that object only when the array shapes, dtypes and values all match exactly.
Preparation still has a cost; freezing a record does not freeze the continuous design values.

The painted-shape path reads "moving the geometry" more narrowly, because its record holds
combinatorics rather than coordinates: a shape sliding is what it replays, bit for bit, and only
a change in *which shape covers which* makes it stale. See
[`../boundary-modeling.md`](../boundary-modeling.md) for which motions those are.

A freeze also records the checks the geometry passed. A record taken over a layer that would be
refused replays the refusal, rather than letting a frozen scene through a door the live one
cannot pass.

An arbitrary `forward` may construct another scene or perform several calculations. `fit`
cannot infer that scene from a closure, so it requires an explicit preparation contract:
`prepare_geometry(**conditions)` runs outside JIT each step, and its return is passed as
`forward(geometry=..., **conditions)`. For a wrapper over this solver, for example:

```python
def forward(geometry, **conditions):
    return solver.solve(geometry=geometry, **conditions)

best = solver.fit(['thickness'], forward, lambda result: -result.T.sum().real,
                  optax.adam(1.), iteration=20,
                  prepare_geometry=solver.freeze_geometry)
```

The wrapper must consume the supplied geometry rather than capture an initial freeze.
When `forward` is a `functools.partial`, preparation receives its fixed keyword defaults
as well, with the current conditions taking precedence and `geometry` excluded. For a
built-in solver forward, only solver conditions are passed to preparation; sampling sizes
and call options such as `auto_detune` stay on the forward. A geometry bound by keyword or
position is replaced with the current record.

Without `prepare_geometry`, an opaque forward is refused before optimization. A custom forward
with a mapping of design variables returns those variables without assigning them as solver
attributes; it owns their interpretation. `parameterize`, when supplied, runs before both
preparation and the forward call.

On integrated raster paths, equal-valued pixels remain independent design variables. The
Fourier build keeps all pixel edges during differentiation and JAX tracing, because selecting
one representative per equal block would discard the other pixels' derivatives. Concrete
evaluations still compress equal blocks. Traced integrated builds can therefore cost more
than their compressed eager evaluations, especially on large rasters with many repeated values.

If writing a loop directly, follow the same order: apply the **current** design conditions to
`freeze_geometry`, pass the resulting record as a static argument of the compiled calculation,
then update the parameters. Keep preparation outside JIT. This updates the discrete plan;
it does not make contact or overlap transitions differentiable at the transition itself.

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
