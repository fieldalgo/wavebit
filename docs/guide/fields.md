# Near fields

Read fields from the result of a solve. Ordinary incidence, order excitations,
`PlaneWave`, `GaussianBeam` and `ElectricDipole` results share these methods:

```python
result = solver.solve(answers='field')
cell = result.reconstruct_field(sample_x=64, sample_y=64, sample_z=32)
boundary = result.boundary_fields('top', sample_x=64, sample_y=64, part='total')
```

The result owns its field state. Changing the solver's conditions or solving another
scene does not change fields read from this result. Resample a solved result without
solving again; solve again when the physical problem changes.

## What to retain

| `answers` | Retained results |
| --- | --- |
| `'efficiency'` | Computed power results |
| `'amplitude'` (default) | Also complex amplitudes and boundary fields |
| `'field'` | Also the per-layer state for internal fields |

This rule applies to every input type. Boundary fields require `'amplitude'` or
`'field'`; internal fields require `'field'`. Missing retained state raises an
error naming the answer level needed for a new solve. The default is `'amplitude'`.
`answers` must be static under JAX `jit`.

A source-based result's `power` stores the power evaluated over the solve's
`power_region`, including at `'efficiency'`. Calling `boundary_power(region)`
again integrates the retained complex fields and requires `'amplitude'` or
`'field'`. An efficiency-only solve uses those fields to compute power, then
releases them instead of keeping a hidden reconstruction state.

## The jitted solver method

On JAX, `solver.solve_field(..., geometry=g)` returns the result and field from a
compiled copy of the solver. It does not install that copy's solve state on the
live solver. Concrete keyword conditions stay on the live solver, following
`solve()`; when they change its scene, its earlier solve state is dropped and
`solver.reconstruct_field()` refuses until another `solve()`. Use the returned
field or result instead. Keywords that are only traced are restored afterwards.
A saved result continues to own its field state independently of the live solver.

Before compiling, `solve_field` checks the frozen geometry against the concrete
scene, including scalar and tensor shape topology and raster interfaces. A changed
topology requires a new `freeze_geometry()`; material sweeps and shape changes
that still fit the frozen plan remain valid. Under an outer `jax.jit` or
`jax.grad`, traced geometry cannot be checked in Python. Prepare the geometry
from the current concrete scene before differentiating; `fit()` does this for
each optimization step.

## Coordinates

Each sampling argument accepts a positive integer count or a nonempty 1-D array of
finite real coordinates. `reconstruct_field` defaults to 20 on each axis;
`boundary_fields` defaults to the cell centre, `sample_x=[0.]`, `sample_y=[0.]` -- the point
where an ordinary incidence is unit, so `boundary_fields(side, part='incoming')` there is the
incident polarization itself at any real angle.

```python
cell = result.reconstruct_field(
    sample_x=[-100., 0., 100.], sample_y=[0.], sample_z=[0., 40., 120.],
)
```

Lengths use the same unit as wavelength and thickness. x/y are measured from the
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

Every result returns complex fields in one Cartesian layout:

| Method | Shape | Last axis |
| --- | --- | --- |
| `reconstruct_field` | `(nz, ny, nx, 6)` | `(Ex, Ey, Ez, Z0Hx, Z0Hy, Z0Hz)` |
| `boundary_fields` | `(ny, nx, 6)` | Same |

Pure TE/TM includes zero-valued components in the same six slots. y-invariant fields
are repeated at all requested y coordinates. `nz` is the number of layers times
the z count, or the length of an explicit z array.

```python
E = cell[..., :3]
H_normalized = cell[..., 3:]
H = H_normalized / 376.730313668
```

If the electric field scale is V/m, `H` is in A/m. Wavebit solves for `Z0*H` from
the outset; the returned magnetic components are not SI H until converted.
The time convention is `e^{-iωt}`: the instantaneous real field is
`real(E * exp(-1j*omega*t))`. `abs(E)` and `angle(E)` give each component's magnitude
and phase. See [conventions](../conventions.md) for the source phase reference.

Every ordinary input is an electric amplitude: coupled s/p, `PlaneWave`, and the pure 1D
TM solve included. (Until 2026-09-27 the pure 1D TM solve was lit by a unit magnetic
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

Ordinary coupled solves also retain pure s/p responses. Select one with
`incidence='s'` or `'p'`; the default `'incident'` returns the specified excitation.
There is no leading incidence axis. Pure TE/TM and source-based results only accept
`'incident'`. A missing variant raises an error rather than substituting another.

## Differentiating internal fields

Ordinary result fields support JAX and PyTorch differentiation through the solve and
reconstruction. A z count follows the relative position in each layer when thickness
changes. Explicit z arrays hold physical heights fixed and support thickness
derivatives while each point stays in the same layer. A moving interface crossing a
point can change its material and normal field discontinuously. Keep sample points
inside the stack; eager validation is skipped when values are abstract under `jit`.

The source-based solver still uses concrete source grouping and layer segmentation;
this ordinary-solve differentiation guarantee does not extend to those operations.

## Solver convenience calls

`solver.reconstruct_field` and `solver.solve_field` remain available with their
existing layout and sampling contracts. Ordinary solver calls use three slots for
pure TE/TM and an incidence axis for coupled fields. Code using those layouts can
continue to run; use the result methods above for the common six-component API.
