# Fields and diffraction observations

Request retained fields with `rcwa.Outputs(fields=True)`, run the full problem, then sample that result:

```python
samples = result.field(x=[0., .1], y=[0.], z=[.09])
print(samples.values.shape)  # (n_wavelengths, nz, ny, nx, 6)
print(samples.units)
```

The components are `(Ex,Ey,Ez,Hx,Hy,Hz)`. Public E is in V/m and H is in A/m; coordinates use the problem's length unit. `samples.units` says so per key: each component and `'coordinates'`. The wavelength axis remains even for a single wavelength. These are complex phasors under `exp(-i omega t)`. `result.boundary_fields(side=..., x=..., y=..., part=...)` samples incoming, outgoing or total boundary phasors from the same retained state.

`result.reflection` and `result.transmission` are diffraction observations with wavelength and order metadata. Select an order with `.order(m,n)` and a wavelength by its position with `.at_wavelength_index(i)`. Amplitude ratios (`amplitude_ratio_s`, `amplitude_ratio_p`, `amplitude_ratio_xyz`, with no unit), the outgoing fields they give at the source amplitude (`field_s`, `field_p`, `field_xyz`, in V/m), normalized efficiencies and entering power are distinct quantities; inspect their validity and reference metadata. A later run cannot change an earlier result's retained field.

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
| `Outputs()` (default) | Computed power results: diffraction efficiencies, complex amplitude ratios and powers for a single plane wave, boundary power for other sources |
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
A height within a few rounding steps of a face (four per layer, relative to the stack height)
is read at that face, so a face written as your own sum of thicknesses, such as
`0.1 + 0.2 + 0.3` for layers `(0.3, 0.2, 0.1)`, selects the same side as the face itself; the
z derivative at such a height is kept. A height really outside the stack is refused with
`field.coordinates`, and the message gives the range and every face height.

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
- A stack declared with no layers has no interior to sample: `field` refuses it with
  `field.coordinates`, whatever `z` is. Its `z = 0` is the interface between the two
  half-spaces, and `boundary_fields(side='top', ...)` and `boundary_fields(side='bottom', ...)`
  give the two half-spaces' fields there. A declared layer of zero thickness is still sampled.
- `boundary_fields` takes `side='top'` or `'bottom'` and `part='incoming'`, `'outgoing'` or
  `'total'`; any other value is refused with `field.side` or `field.part`, at that path.

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

Every input is an electric amplitude: coupled s/p, `PlaneWave`, and the pure 1D TM solve
included. (Until 2026-09-27 the pure 1D TM solve was lit by a unit magnetic
input, so its field was `mu_inc/n_inc` of the conical p field for the same light.) The
result API preserves the solved input's scale; it does not rescale the field by incident
power.

A `PlaneWave` written `'s'` or `'p'` on a one-row raster solves on that pure TE/TM path
([the light guide](light.md#the-words-are-a-declaration-the-pair-is-a-value) says when), and its
`field` and `boundary_fields` carry the same six components as the conical path's, to round-off.
The three the polarization does not excite -- `E_x`, `E_z` and `H_y` for `'s'`, `E_y`, `H_x` and
`H_z` for `'p'` -- are exact zeros there, where the conical path returns round-off (1e-16 of the
excited components).

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
a JAX-traced input; this differentiation guarantee does not extend to them. Under eager `jax.grad`
a dipole's segments are concrete too, so a dipole solve carries no derivative with respect to a
layer's thickness or the period: it reads exactly zero or leaves out the cell area, and a finite
difference is the way to read one.

## Normal components inside a patterned layer

Inside a patterned layer the field component normal to a material wall jumps there (it is
`eps*E_n` that is continuous, and `mu*H_n` where the wall changes `mu`), so the Fourier sum of its
coefficients rings near every wall at any truncation, and the ringing reaches well away from the
walls. `result.field(x=..., y=..., z=..., reconstruction='lalanne')` rebuilds that component from
the continuous `D` or `B` series instead: sum `D`, then divide by `eps` at each sample (P. Lalanne and
M. P. Jurek, J. Mod. Opt. 45, 1357 (1998)), and the same for `B` and `mu`. The value is named after
that paper, which treats a 1D TM grating; the 2D, conical and magnetic cases follow RETICOLO's `res3`
(J. P. Hugonin and P. Lalanne). The default, `'plain'`, sums each component's own series. The samples'
metadata records the rule under `'reconstruction'` when it is not the default.

- **What it changes.** `E_x` in the patterned layers of a 1D grating, on the pure TM and the
  conical paths; `E_x` and `E_y` in the patterned layers of a 2D cell. A layer that names a
  permeability takes the same rule for `H`: `H_x` on the pure TE and conical paths, `H_x` and `H_y`
  on a 2D cell. Without a permeability no component of `H` jumps, so `H` is the default's. Every other
  component and every uniform layer are returned bit for bit as the default returns them. A diagonal
  tensor reads each axis's own entry: `eps_xx` for `E_x`, `eps_yy` for `E_y`, `mu_xx` for `H_x`, `mu_yy` for
  `H_y`. A layer merged into a half-space is uniform, and keeps the sum. A dipole's uniform host keeps
  the sum too, and the patterned layers around it take the rule. A 1D `s` wave named by its word
  takes the pure TE path, which has no `E_x` to change; given as a Jones pair it runs on the conical
  path, where `E_x` is zero up to rounding and the rule's rounding is its own. On a one-row raster
  the pure TE or TM path and the conical path give the same rule to 1e-14 of the peak.
- **How it works on a 2D cell.** Line by line. `E_x` jumps across walls normal to x and is
  continuous across walls normal to y, so at each sample its series is summed over the y orders
  first, then Li's inverse rule (L. Li, J. Opt. Soc. Am. A 13, 1870 (1996)) along the row through
  the sample turns its x coefficients into those of `D_x`, which is summed along x and divided by
  `eps`. `E_y` is the same along the column through the sample, and `H_x` and `H_y` are the same
  with `B` and `mu`.
- **What it gets right on a 2D cell.** `E_x` meets two kinds of wall: across a wall normal to x it
  is the normal component and must jump (`eps*E_x` is what stays continuous), and across a wall
  normal to y it is tangential and must not jump.
  - The default is continuous everywhere. Across walls normal to x, `eps*E_x` mismatches by the
    full contrast (75 to 92% of the larger side in three test cells) at every order, and the sum
    rings beside the wall. Across walls normal to y it is continuous but off on both sides: its row
    carries the ringing of the walls normal to x that the row meets.
  - The Lalanne rule makes `eps*E_x` continuous across walls normal to x, to 1e-7. Across a wall
    normal to y it reads the two sides off two different rows. The row that holds the walls is
    right. The other row, such as the air row under a block, has no wall of its own, so its
    operator is the identity and it returns exactly the default's value. `E_x` then steps there
    where the true field is continuous. The step is not a new error: it is the default's error at
    that wall, removed on one side and left on the other. The same step appears at any boundary
    between two rows that differ somewhere, even where the material on both sides is the same.
  - Checked against a finite-element reference (gyptis, second-order edge elements; period
    400 nm, a 200 nm square of `eps` 12.25 in a 150 nm layer, normal incidence at 900 nm, Li's
    factorization, order 16): over 2304 samples on the mid-layer plane the relative RMS error of
    the in-plane field is 1.52% for the rule and 9.10% for the default. 0.3 nm from the middle of
    the square's bottom wall, on the square's side, the rule is off by 1.1% of the local field and
    the default by 25%; on the air side the two are equal. The rule was as accurate as the default
    or better in every region checked, at walls and away from them. This square is the only cell
    with walls along x and y that has a finite-element reference.
  - Five repairs of the remaining step were measured against the same reference. None is in
    Wavebit: four were less accurate than the rule somewhere, and the best (a correction toward the
    right side's wall value, switched off near corners: 1.11% against 1.52% on the square's 8x8
    raster) changed with the raster's resolution and with a relative change of about 1e-5 in one
    pixel's permittivity, and departs from the RETICOLO reference the rule is pinned to.
- **Choosing a rule.** Which rule is right depends on what is read, so neither is right
  everywhere.
  - `'lalanne'` for values beside a wall normal to the component: `D` or `eps*E_n` across the
    wall, the field just inside it, absorbed power beside a sidewall. On walls along x and y.
  - `'plain'` for a spatial derivative across a wall the component runs along, such as an optical
    force from the gradient of `|E|^2` or a curl taken afterwards. On the square above, the
    difference of `|E_x|` over two samples 0.3 nm either side of the bottom wall is +0.002 per nm
    in the reference, +0.004 for the default and -0.082 for the rule, whose step sits between them.
  - `'plain'` for a smooth map of where the field concentrates: it is continuous where the rule
    steps, though not right beside walls normal to the component.
  - `'plain'` in a layer that holds a metal, unless two truncations agree on the rule's peak (see
    *Metals* below), and for the derivative of an area average with respect to a shape's
    position or size (see *Derivatives* below).
  - Either, away from walls (the two rules converge together there), and for `R`, `T` and the
    absorption from the ports, which neither rule changes.
  - On curved or slanted walls, not this rule, which is refused there (see *Which shapes* below).
    Give the true shape with `factorization='nvf'`, check that `R` and `T` have converged, and read
    the default. A rule that splits `E` at each sample into the part normal to the local wall,
    read from `D`, and the part along it, read from `E` (M. Weismann, D. F. G. Gallagher and N. C.
    Panoiu, J. Opt. 17, 125612 (2015)), was more accurate than the default on all 16 dielectric
    circle cells checked against a finite-element reference: 0.06 to 0.67 times its error over the
    plane at order 12, with a window that switches the split off away from the wall. It is not in
    Wavebit.
- **Why the default is `'plain'`.** The default never puts a jump where the field is continuous,
  and its derivatives across walls a component runs along, and of area averages, are right. What
  it misses is the jump at a wall normal to a component, which is what the Lalanne rule is for.
  The rule is a choice for what it measures best, and making it the default would also change
  every field already read with the default.
- **Why use it.** It is RETICOLO's `res3` rule (`parm.res1.champ=1`): with it the RETICOLO field
  goldens agree to 1e-13 on every slot of the 1D and conical cases, `E_x` included, where the
  default differs by 7e-2 to 1e-1, and to 4e-11 on every slot of a two-layer crossed grating,
  where the default's `E_x` and `E_y` differ by about 0.3. With a patterned `mu` it agrees to 4e-11
  on every slot of the 2D, pure TE, pure TM and conical cases, where the default's `H_x` differs by
  0.10 to 0.31. On a 1D TM bar of index 3.5, over 2000 samples at mid-layer, its relative RMS
  error in `E_x` falls from 6.53% to 0.05% as the order goes from 5 to 160, and the default's from
  23.08% to 4.15%, against the rule at order 800.
- **What it reads.** `eps` at each sample and, on a 2D cell, the inverse-rule matrix of the row
  and the column through it, and the same of `mu` in a layer that names one. A raster gives both from the pixel the sample falls in. Choosing it
  from explicit coordinates takes a concrete period and concrete coordinates: a solve inside
  `jax.jit` traces the period and a map over the coordinates traces them, and both are refused.
  Counts always work, and explicit coordinates under `jax.grad`, under a map over anything else,
  or read inside `jax.jit` from a solve outside it. A shape structure gives them from where that line crosses its outlines, so a wall sits where the
  shape says, between any two samples. A sample exactly on a wall belongs to its positive side
  either way, up to rounding: on shapes, a sample or a shape written more than about 60 periods
  from the cell can read a wall it sits on from the other side. It reads the structure and the cell
  the solve was posed with, from a copy the result keeps, so an edit after a run waits for the next
  run, as it does for the default. The copy is kept whether or not the rule is asked for, and a
  raster is held complex: 16.8 MB per wavelength for each 1024 by 1024 layer, nine times that
  (151 MB) for a tensor raster, and as much again when the stack names a raster `mu`.
- **Which shapes.** Walls along x and y: rectangles at a right angle and polygons with only such
  edges.
  - Circles, ellipses, rotated rectangles and slanted polygons are refused. So is a differentiated
    or traced angle, which turns every wall, and under `jax.jit` or `jax.vmap` a traced vertex, whose
    wall direction cannot be read. Across a slanted wall it is `D_n` that is continuous, not `D_x`,
    so the rule leaves the whole contrast in the tangential `E`. Against a 3D finite-element
    reference on 16 circle cells, the rule applied along x and y lost to the default on 4, by up
    to 5.8 times, with a tangential jump that more orders did not shrink. A raster of the same shape
    takes the rule on its staircase. Check its `R` and `T` against the true shape before reading
    its field: drawn as a staircase and solved with `factorization='li'`, a high-index circle had
    `R` off by 18 to 51% on the three cells whose `R` is 0.5 or more, and more steps or pixels did
    not change that; `'nvf'` on the same raster came near the true circle on the one cell tried.
  - A diagonal lattice with a negative side is refused by name: it is read as spelled, on the
    general path, where the rule has no x and y lines to follow. Spell the cell with positive sides
    (a raster then reversed along that axis) to use the rule. A trace has no sign to read, so there
    `E_x` and `E_y` come back NaN.
  - Against RETICOLO's `champ=1`, with `factorization='li'`, the rule agrees to 4e-11 on every
    slot, on walls a raster can draw and on walls it cannot, where the default's `E_x` is 0.43
    away. Only Li's rules have that reference. `'nvf'` and `'laurent'` run and are not pinned to
    it. Against a finite-element reference on one 200 nm block of `eps` 12.25 at order 8, the
    largest component error over a plane through the block, relative to the field's RMS, is 4.3% on
    `'li'`, 7.3% on `'nvf'` and 23% on `'laurent'`, against 21%, 21% and 28% for the default. Away from the walls `'laurent'` gets slightly worse
    with the rule, 16% against 15%. On a patterned `mu` no finite-element reference has been run;
    RETICOLO is the only one.
  - A shape with the background's own `eps` goes through the rule and matches the default to
    rounding, not bit for bit, since which values are equal is not asked under a trace. So does `H`
    in a layer of shapes that names a permeability, even `mu=1.0`. So does
    every layer of a raster being differentiated, uniform or not: a uniform layer's value is the
    default's to rounding, and its derivative is the rule's.
- **Derivatives.** With respect to a raster's values, a shape's position and lengths or a
  material, a derivative holds each sample point fixed, and agrees with a finite difference off
  the walls. The choice of pixel, or which side of a wall a sample is on, is not differentiated. So
  it is not the derivative of an integral over an area: a wall sweeping across a sample is left
  out, and `E_n` jumps by the contrast there. And the rule's field jumps, so a sampled mean of it
  is not the integral. On that block at order 8, with `'p'` light, off centre at `cx = 13` and a
  32 by 32 grid of samples, moving the shape cannot change the mean of `|E_x|^2` over the cell.
  The default's derivative is 1e-17. The rule's is 2.4e-4: the swept walls' parts cancel on a
  move, block to air on one face and air to block on the other, and what remains is the slope of
  the sampled mean, which wobbles as each face nears or leaves its neighbouring samples. Growing
  the shape, the swept parts add: +3.7e-3 where the integral's slope is about -1.9e-3 (a
  straight-line fit over dense samples gives -1.95e-3). For an objective over an area, take
  `'plain'`. An outline's derivative moves vertices it does not check: move both ends of a
  wall together, or the wall leaves the shapes the rule answers for. A read takes the derivative
  its solve took: with the solve outside the differentiated call the derivative is zero, as it is
  for the default. A rectangular cell differentiates through `period=`; a `lattice=` differentiated
  as a whole matrix can shear, and is refused.
- **Metals.** The inverse rule inverts the Toeplitz matrix of `1/eps` along a line, and of `1/mu` in
  a layer that names one. With a negative
  `eps` or `mu` that matrix is singular at isolated fill factors, which depend on the truncation, and not
  in general where the mean of `1/eps` vanishes. At `eps = -1` and 50% fill it fails on a raster
  with code `rcwa.singular` when a sample is on such a line (on JAX, `E_x` comes back NaN there, or
  `H_x` on a singular `1/mu` line) or, at
  some orders on a coarse raster, returns a field of order 1e17, and it returns 1e16 to 1e18 on
  shapes. Loss keeps it finite but not small: at
  `eps = -20+0.1j`, order 3, the rule's peak is typically 3 times the default's and up to 4000
  times at the worst fill. Compare two truncations before trusting a peak on a metal line.
- **One order along an axis.** The lines still read the walls along that axis, though the solve
  cannot resolve them, so `E_x` differs between two samples the modes cannot tell apart. RETICOLO
  does the same, and on Li's factorization the two agree to 1e-11. On any other factorization the
  lines are still Li's and no reference ranks the answer.
- **What it refuses, by name, with code `field.reconstruction`.** RETICOLO has no answer for any of
  these. A slanted or curved wall on shapes; a raster takes the rule on its staircase, which is
  what a RETICOLO ellipse is; check that staircase's `R` and `T` first (see *Which shapes*
  above). A skewed lattice. A patterned `eps` or `mu` with off-diagonal
  entries, where `E_x` needs more than `D_x` and a division. A whole 3x3 tensor, on a raster or a
  shape, under a derivative or a trace, whose zero off-diagonals can hide one; on shapes, principal
  values `Material([xx, yy, zz])` take the rule and keep the derivative. A tensor uniform across
  its layer takes the rule, whatever the other field does. A diagonal lattice with a negative
  side, on shapes and rasters. Matrices handed to `solve_conv`, which need not be any structure.
- **Heights.** The rule reads the layers a sample is in. Explicit heights in a layer whose line is
  singular at those `x` and `y` fail; heights elsewhere answer, and a uniform layer keeps the sum. A
  count reads every layer, and so do heights under a trace, where a singular line in another layer
  leaves the value right and its gradient NaN.

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
