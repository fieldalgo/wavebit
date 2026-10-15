# Fields and diffraction observations

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Request retained fields with `rcwa.Outputs(fields=True)`, run the full problem, then sample that result:

```python
samples = result.field(x=[0., .1], y=[0.], z=[.09])
print(samples.values.shape)  # (n_wavelengths, nz, ny, nx, 6)
print(samples.units)
```

The components are `(Ex,Ey,Ez,Hx,Hy,Hz)`. Public E is in V/m and H is in A/m; coordinates use the problem's length unit. The wavelength axis remains even for a single wavelength. These are complex phasors under `exp(-i omega t)`. `result.boundary_fields(side=..., x=..., y=..., part=...)` samples incoming, outgoing or total boundary phasors from the same retained state.

`result.reflection` and `result.transmission` are diffraction observations with wavelength and order metadata. Select an order with `.order(m,n)` and a wavelength with `.at_wavelength(i)`. Raw amplitudes, normalized efficiencies and entering power are distinct quantities; inspect their validity and reference metadata. A later run cannot change an earlier result's retained field.

The sections below give the sampling rules, the retained state and the differentiation scope of
these two methods. Public H is already in A/m: do not divide it by Z0.

:::{note}
The pure s/p field variants (`incidence='s'`/`'p'`) are not in the current API yet; that passage
names the earlier `wavebit.compat.legacy` call that has them.
:::

## Reading fields from a result

Read fields from the result of a run. Single plane waves, order excitations,
`PlaneWave` lists, `GaussianBeam` and `ElectricDipole` results share these methods:

```python
result = rcwa.run(problem.replace(outputs=problem.outputs.replace(fields=True)),
                  numerics=numerics)
cell = result.field(x=64, y=64, z=32)
boundary = result.boundary_fields(side='top', x=64, y=64, part='total')
```

The result owns its field state. Running another problem, or the same preparation again,
does not change fields read from this result. Resample a result without
running again; run again when the physical problem changes.

The current API keeps no live solver, so nothing can make a result's fields stale. The earlier
interface's `solver.reconstruct_field()` method read the live solver and refused once a
condition used by reconstruction had changed since the solve.

## What to retain

| `Outputs` | Retained results |
| --- | --- |
| `Outputs()` (default) | Computed power results: diffraction efficiencies, complex amplitudes and powers for a single plane wave, boundary power for other sources |
| `Outputs(fields=True)` | Also the per-layer state for boundary and internal fields |

This rule applies to every source type. `boundary_fields` and `field` both require
`Outputs(fields=True)`; without it they raise `UnsupportedFeatureError` (code
`output.unavailable`) naming the request. `Outputs` is part of the prepared contract, so it is
static under JAX `jit`.

A source-based result's `outputs['top']` and `outputs['bottom']` store the power evaluated
over the problem's `Outputs(power_region=...)`, with or without fields. The current result has
no method that integrates the retained fields over another region; run the problem again with
that region. A run without fields uses the boundary fields to compute power, then
releases them instead of keeping a hidden reconstruction state.

## Under `jax.jit`

On JAX, prepare outside the transform and pass the live problem inside it, for example
`jax.jit(lambda p: prepared.run(p).field(x=..., y=..., z=...).values)`. This holds for a single
plane wave; other sources refuse a JAX-traced input. Explicit coordinate arrays and integer
counts both work inside, and so does a traced layer thickness with explicit heights. A region
geometry or cell under `jax.jit` needs the Cartesian Li device guard or explicit
`Numerics(topology_check='host')` for admitted scalar regions. Invalid topology masks
retained fields as well as powers; other traced geometry is refused with
`prepared.host_topology`. Eager `jax.grad` checks its concrete primal. See
[automatic differentiation](../common/autodiff.md) and [checked shape JIT](guarded-shape-jit.md). The earlier interface's compiled
`solver.solve_field(..., geometry=g)` and its static-sampling rules belong to that interface
only.

## Coordinates

Each sampling argument accepts a positive integer count or a nonempty 1-D array of
finite real coordinates. Both methods require `x` and `y`, and `field` requires `z`; there are
no defaults. At the cell centre, `x=[0.]`, `y=[0.]`, a single plane wave is unit, so
`boundary_fields(side=..., x=[0.], y=[0.], part='incoming')` there is the incident polarization
itself, times the source `amplitude`, at any real angle.

```python
cell = result.field(
    x=[-100., 0., 100.], y=[0.], z=[0., 40., 120.],
)
```

Lengths use the problem's length unit. x/y are measured from the
cell centre. z is height from the bottom of the stack, with +z upward, and must lie
between 0 and `sum(thickness)`. Explicit coordinates retain the requested order.

- Counts on x/y include both ends of the cell. A count of one selects the negative
  edge; `[0.]` selects the centre. On a skewed lattice, two counts sample along the
  lattice vectors, with axes `(v,u)`. Two coordinate arrays sample the Cartesian
  product of x/y, including on skewed lattices. Mixed count/array inputs require a
  concrete axis-aligned lattice.
- A z count is **per finite layer**. Counts of two or more include each layer's
  two faces, retaining both limiting values at internal interfaces. A count of one
  reads each layer's bottom face.
- An explicit z at an internal interface selects the upper layer. At the two
  outside interfaces, internal reconstruction returns the finite-layer limit.
- A dipole's source plane is singular and must be excluded from the sample heights.

## Components and shape

Every result returns complex fields in one Cartesian layout, in `.values`:

| Method | Shape | Last axis |
| --- | --- | --- |
| `field` | `(n_wavelengths, nz, ny, nx, 6)` | `(Ex, Ey, Ez, Hx, Hy, Hz)` |
| `boundary_fields` | `(n_wavelengths, ny, nx, 6)` | Same |

y-invariant fields are repeated at all requested y coordinates. `nz` is the number of layers
times the z count, or the length of an explicit z array.

```python
E = cell.values[..., :3]  # V/m
H = cell.values[..., 3:]  # A/m
```

With the source amplitude in V/m, `H` is in A/m. Wavebit solves for `Z0*H` from
the outset and divides by Z0 before returning; the earlier interface returned `Z0*H`.
The time convention is `e^{-iωt}`: the instantaneous real field is
`real(E * exp(-1j*omega*t))`. `abs(E)` and `angle(E)` give each component's magnitude
and phase. See [conventions](../../reference/common/conventions.md) for the source phase reference.

Every input is an electric amplitude: coupled s/p, `PlaneWave`, and the earlier interface's
pure 1D TM solve included. (Until 2026-09-27 the pure 1D TM solve was lit by a unit magnetic
input, so its field was `mu_inc/n_inc` of the conical p field for the same light.) The
result API preserves the solved input's scale; it does not rescale the field by incident
power.

## Boundary waves and polarization variants

`side='top'` or `'bottom'` selects the outside half-space limit at that interface.
`part='incoming'`, `'outgoing'`, or `'total'` selects the wave content. At the same
points, total equals incoming plus outgoing. Internal fields always include all
contributions. Tangential E/H are continuous across an interface without surface
sources; normal components can jump.

Multiple sources are summed as complex fields. The observation region used for
power averaging does not change these pointwise fields.

The earlier interface's coupled solves also retain pure s/p responses, selected with
`incidence='s'` or `'p'` on its result methods. The current `field` and `boundary_fields` take
no `incidence` argument and return the declared excitation; run the `'s'` and `'p'`
polarizations as two problems for the pure responses.

## Differentiating internal fields

Single-plane-wave result fields support JAX and PyTorch differentiation through `prepared.run`
and `field`. A z count follows the relative position in each layer when thickness
changes. Explicit z arrays hold physical heights fixed and support thickness
derivatives while each point stays in the same layer. A moving interface crossing a
point can change its material and normal field discontinuously. Keep sample points
inside the stack; eager validation is skipped when values are abstract under `jit`.

At a layer face, coordinate derivatives use the selected finite layer's limiting
field (the upper layer at an internal interface). They retain the full spatial
derivative, including at an artificial seam between identical materials. This
does not define a unique two-sided derivative across a material discontinuity.

Runs with other sources still use concrete source grouping and layer segmentation, and refuse
a JAX-traced input; this differentiation guarantee does not extend to them.

## Earlier-interface field calls

The earlier `wavebit.compat.legacy` interface keeps `solver.reconstruct_field` and
`solver.solve_field` with their existing layout and sampling contracts: three slots for pure
TE/TM, an incidence axis for coupled fields, and magnetic components in `Z0*H`. Code using
those layouts can continue to run there; compare it with the current result only after
dividing its magnetic components by Z0.
