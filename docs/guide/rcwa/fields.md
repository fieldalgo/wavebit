# Fields and diffraction observations

Request retained fields with `rcwa.Outputs(fields=True)`, run the full problem, then sample that result:

```python
samples = result.field(x=[0., .1], y=[0.], z=[.09])
print(samples.values.shape)  # (n_wavelengths, nz, ny, nx, 6)
print(samples.units)
```

The components are `(Ex,Ey,Ez,Hx,Hy,Hz)`. Public E is in V/m and H is in A/m; coordinates use the problem's length unit. `samples.units` says so per key: each component and `'coordinates'`. The wavelength axis remains even for a single wavelength. These are complex phasors under `exp(-i omega t)`. `result.boundary_fields(side=..., x=..., y=..., part=...)` samples incoming, outgoing or total boundary phasors from the same retained state.

`result.reflection` and `result.transmission` are diffraction observations with wavelength and order metadata. Select an order with `.order(m,n)` and a wavelength by its position with `.at_wavelength_index(i)`. Raw amplitudes, normalized efficiencies and entering power are distinct quantities; inspect their validity and reference metadata. A later run cannot change an earlier result's retained field.

The sections below give the sampling rules, the retained state and the differentiation scope of
these two methods. Public H is already in A/m: do not divide it by Z0.

:::{note}
The pure s/p field variants (`incidence='s'`/`'p'`) are not in the current API yet; that passage
names the earlier `wavebit.compat.legacy` call that had them (retired 2026-10-15; a historical checkout runs it).
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
`outputs.unavailable`) naming the request. `Outputs` is part of the prepared contract, so it is
static under JAX `jit`.

A source-based result's `outputs['top']` and `outputs['bottom']` store the power evaluated
over the problem's `Outputs(power_region=...)`, with or without fields. The current result has
no method that integrates the retained fields over another region; run the problem again with
that region. A run without fields uses the boundary fields to compute power, then
releases them instead of keeping a hidden reconstruction state.

## Under `jax.jit`

On JAX, prepare outside the transform and pass the live problem inside it, for example
`jax.jit(lambda p: prepared.run(p).field(x=..., y=..., z=...).values)`. This holds for a single
plane wave; other sources refuse an input traced under `jax.jit` (eager `jax.grad` takes them). Explicit coordinate arrays and integer
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

## Normal components inside a patterned layer

Inside a patterned layer the field component normal to a material wall jumps there (it is
`eps*E_n` that is continuous), so the Fourier sum of its coefficients rings near every wall at any
truncation, and the ringing reaches well away from the walls.
`result.field(x=..., y=..., z=..., discontinuous='continuity')` rebuilds that component from the
continuous `D` series instead: sum `D`, then divide by `eps` at each sample (P. Lalanne and
M. P. Jurek, J. Mod. Opt. 45, 1357 (1998)). The default, `'fourier'`, is unchanged. The samples'
metadata records the rule under `'discontinuous'` when it is not the default.

- **What it changes.** `E_x` in the patterned layers of a 1D grating, on the pure TM and the
  conical paths; `E_x` and `E_y` in the patterned layers of a 2D cell. Every other component and
  every uniform layer are returned bit for bit as the default returns them. A 1D `s` wave runs on
  the conical path here, where `E_x` is zero up to rounding and the rule's rounding is its own.
- **How it works on a 2D cell.** Line by line. `E_x` jumps across walls normal to x and is
  continuous across walls normal to y, so at each sample its series is summed over the y orders
  first, then Li's inverse rule (L. Li, J. Opt. Soc. Am. A 13, 1870 (1996)) along the row through
  the sample turns its x coefficients into those of `D_x`, which is summed along x and divided by
  `eps`. `E_y` is the same along the column through the sample.
- **What it trades on a 2D cell.** Each rule is right on one kind of wall and wrong on the
  other, and neither error goes away as the truncation grows. `E_x` meets two kinds of wall:
  across a wall normal to x it is the normal component and must jump (`eps*E_x` is what stays
  continuous), and across a wall normal to y it is tangential and must not jump.
  - The default is continuous everywhere, so it is right across walls normal to y and wrong
    across walls normal to x: there `eps*E_x` mismatches by the full contrast (75 to 92% of the
    larger side in the three test cells below), and the sum rings beside the wall.
  - The continuity rule makes `eps*E_x` continuous across walls normal to x, to 1e-7. But
    across a wall normal to y it reads the two sides off two different rows, each row's
    operator built for that row's walls alone, so there `E_x` steps where it should be
    continuous.
  - Measured on three cells — a 3.5-index square, a 2.5-index checkerboard with four-way
    corners, and a lossy L-shape at oblique incidence — at orders 4, 8, 12 and 16, each jump
    relative to the larger side, at the middle of each pixel edge:
    - The step is 5 to 43% of the local field (median over walls) and up to 75% at the worst
      wall, beside corners.
    - It shows no trend toward zero over that range. The default's mismatch across the other
      walls does not shrink either.
    - A higher order does not remove the step. It is the same kind of error as the one the
      rule removes, moved to the other walls.
  - More than one pixel away from any wall the two rules agree and converge together. There
    the continuity rule is somewhat more accurate: RMS error 0.7 to 2.0% of the field at
    order 16, against 1.5 to 2.3% for the default.
  - The choice therefore depends on what you read and where. Use `'continuity'` for anything
    built from the normal field at sidewalls: `D` or `eps*E_n` across a wall, the field just
    inside a wall normal to the component, or absorbed power beside a sidewall. Keep the
    default where a component runs along walls normal to the other axis and you need it
    continuous there, as in a smooth field map or a tangential boundary value. Away from walls
    either rule will do. A reconstruction right on both kinds of wall would split `E` at each
    point into the part normal to the local wall, read from `D`, and the part along it, read
    from `E`. Wavebit does not have one yet.
- **Why use it.** It is RETICOLO's `res3` rule (`parm.res1.champ=1`): with it the RETICOLO field
  goldens agree to 1e-13 on every slot of the 1D and conical cases, `E_x` included, where the
  default differs by 7e-2 to 1e-1, and to 4e-11 on every slot of a two-layer crossed grating,
  where the default's `E_x` and `E_y` differ by about 0.3. On a 1D TM bar of index 3.5 its RMS
  error in `E_x` is 2.5 to 7 times smaller than the default's at every truncation order from 10
  to 80, against a reference at order 800.
- **What it reads.** `eps` at each sample and, on a 2D cell, the inverse-rule matrix of the row
  and the column through it. A raster gives both from the pixel the sample falls in. Choosing it
  from explicit coordinates takes a concrete period and concrete coordinates: a solve inside
  `jax.jit` traces the period and a map over the coordinates traces them, and both are refused.
  Counts always work, and explicit coordinates under `jax.grad`, under a map over anything else,
  or read inside `jax.jit` from a solve outside it. A shape structure gives them from where that line crosses its outlines, so a wall sits where the
  shape says, between any two samples. A sample exactly on a wall belongs to its positive side
  either way, up to rounding: on shapes, a sample or a shape written more than about 60 periods
  from the cell can read a wall it sits on from the other side. It reads the structure and the cell
  the solve was posed with, from a copy the result keeps, so an edit after a run waits for the next
  run, as it does for the default. The copy is kept whether or not the rule is asked for, and a
  raster is held complex: 16.8 MB per wavelength for each 1024 by 1024 layer.
- **Which shapes.** Walls along x and y: rectangles at a right angle and polygons with only such
  edges.
  - Circles, ellipses, rotated rectangles and slanted polygons are refused. So is a differentiated
    or traced angle, which turns every wall, and under `jax.jit` or `jax.vmap` a traced vertex, whose
    wall direction cannot be read. Across a slanted wall it is `D_n` that is continuous, not `D_x`,
    and no reference here ranks the rule against the default there. A raster of the same shape
    takes the rule on its staircase.
  - One order along an axis is refused, because the solve then cannot resolve the layer along
    that axis while the lines run through the real shape. So is a lattice with a negative side;
    a trace has no sign to read, so there `E_x` and `E_y` come back NaN. A raster on such a lattice
    is the same as on the positive one.
  - Against RETICOLO's `champ=1`, with `factorization='li'`, the rule agrees to 4e-11 on every
    slot, on walls a raster can draw and on walls it cannot, where the default's `E_x` is 0.43
    away. Only Li's rules have that reference. `'nvf'` and `'laurent'` run and are not pinned to
    it. Against a finite-element reference on one 200 nm block of `eps` 12.25 at order 8, the RMS
    error over a plane through the block is 4.3% on `'li'`, 7.3% on `'nvf'` and 23% on `'laurent'`,
    against 21%, 21% and 28% for the default. Away from the walls `'laurent'` gets slightly worse
    with the rule, 16% against 15%.
  - A shape with the background's own `eps` goes through the rule and matches the default to
    rounding, not bit for bit, since which values are equal is not asked under a trace. So does
    every layer of a raster being differentiated, uniform or not: a uniform layer's value is the
    default's to rounding, and its derivative is the rule's.
- **Derivatives.** With respect to a raster's values, a shape's position and lengths or a
  material, a derivative holds each sample point fixed, and agrees with a finite difference off
  the walls. The choice of pixel, or which side of a wall a sample is on, is not differentiated. So
  it is not the derivative of an integral over an area: a wall sweeping across a sample is left
  out, and `E_n` jumps by the contrast there. On that block at order 8, with `'p'` light, off centre
  at `cx = 13` and a 32 by 32 grid of samples, moving the shape cannot change the mean of `|E_x|^2`
  over the cell, and the default's derivative is 1e-17 where the rule's is 2.4e-4; growing it gives
  +3.7e-3 where a fine finite difference of the mean gives -1.8e-3. For an objective over an area,
  take `'fourier'`. An outline's derivative moves vertices it does not check: move both ends of a
  wall together, or the wall leaves the shapes the rule answers for. A read takes the derivative
  its solve took: with the solve outside the differentiated call the derivative is zero, as it is
  for the default. A rectangular cell differentiates through `period=`; a `lattice=` differentiated
  as a whole matrix can shear, and is refused.
- **Metals.** The inverse rule inverts the Toeplitz matrix of `1/eps` along a line. With a negative
  `eps` that matrix is singular at isolated fill factors, which depend on the truncation, and not
  in general where the mean of `1/eps` vanishes. At `eps = -1` and 50% fill it fails on a raster
  with code `rcwa.singular` when a sample is on such a line (on JAX, `E_x` comes back NaN there) or, at
  some orders on a coarse raster, returns a field of order 1e17, and it returns 1e16 to 1e18 on
  shapes. Loss keeps it finite but not small: at
  `eps = -20+0.1j`, order 3, the rule's peak is typically 3 times the default's and up to 4000
  times at the worst fill. Compare two truncations before trusting a peak on a metal line.
- **Not yet.** Tensor materials, skewed lattices, the first-order 2D formulation, solves that
  merged layers into a half-space and any source but a single plane wave refuse it by name, with
  code `field.discontinuous`.

## Earlier-interface field calls

:::{admonition} Earlier interface
:class: earlier-interface

Earlier interface, retired with `wavebit.compat` on 2026-10-15: the names and defaults below are not the current API, and its code runs only on a historical checkout.

The earlier `wavebit.compat.legacy` interface kept `solver.reconstruct_field` and
`solver.solve_field` with their existing layout and sampling contracts: three slots for pure
TE/TM, an incidence axis for coupled fields, and magnetic components in `Z0*H`. Code using
those layouts runs on a historical checkout; compare it with the current result only after
dividing its magnetic components by Z0.
:::
