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
z derivative at such a height is kept. This holds under `jax.jit` and `jax.vmap` as well, where
the faces and heights are tracers: the move onto a face is then done on the arrays, and the
height still keeps its own z and thickness derivatives. A height really outside the stack is refused with
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
metadata records the rule under `'reconstruction'` when it is not the default. Two more values,
`'split'` and `'splitw'`, are the rules for the curved walls of circles and of an ellipse. They are experimental and have
their own section, *A curved wall*, below.

**The four values at a glance.** What each one does with a layer, by what the layer holds. "Refused"
is an error with code `field.reconstruction`, never a silent fall back to the sum.

| The layer holds | `'plain'` (default) | `'lalanne'` | `'split'`, `'splitw'` |
| --- | --- | --- | --- |
| nothing patterned | the sum | the sum, bit for bit | the sum, bit for bit |
| a 1D grating | the sum | the rule | refused |
| rectangles at a right angle, or polygons with every edge along x or y | the sum | the rule | refused |
| a raster | the sum | the rule, on the raster's staircase | refused |
| one circle, several circles of one material that do not touch, or one ellipse along the cell's axes | the sum | refused | the rule, experimental |
| a turned rectangle or ellipse, a slanted polygon, an ellipse beside another shape, circles of several materials, circles that touch | the sum | refused | refused |

| Anywhere in the stack | `'plain'` (default) | `'lalanne'` | `'split'`, `'splitw'` |
| --- | --- | --- | --- |
| a skewed lattice, or a diagonal one with a negative side | the sum | refused | refused |
| a patterned permeability | the sum | the rule, for `H` as well | refused |
| a tensor given by its principal values | the sum | the rule, each axis by its own entry | refused |
| a patterned tensor with off-diagonal entries | the sum | refused | refused |

- A rule has to take every patterned layer of the stack. One layer it refuses refuses the call,
  whichever heights are sampled: a stack with a rectangle layer and a circle layer is refused by
  all three rules and read with the sum.
- What a rule changes: `'lalanne'`, `E_x` on a 1D grating and `E_x` and `E_y` on a 2D cell, and the
  same components of `H` where the layer names a permeability; `'split'` and `'splitw'`, `E_x` and
  `E_y` only. Every other component is the sum's, bit for bit.
- How each is checked: `'plain'` and `'lalanne'` against RETICOLO at the same truncation and
  against finite-element fields; `'split'` and `'splitw'` against finite-element fields only,
  which is why they warn (`wavebit.rcwa.diagnostics.ExperimentalWarning`).
- Being taken is not the same as being the better choice. The rest of this section says where
  `'lalanne'` is closer than the sum and where it is not (a layer that holds a metal, a spatial
  derivative), and *A curved wall* does the same for the split rules.

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
    the default. A layer of circles, or of one ellipse along the cell's axes, also takes `'split'`
    and `'splitw'`, which split `E` along the wall's normal (see *A curved wall* below).
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
  - How "along x or y" is decided: from the numbers given, to rounding. An edge of a polygon is
    along an axis when the smaller of its `|dx|` and `|dy|` is at most 1e-12 of the largest
    coordinate step between two neighbouring vertices of that polygon. A rectangle is at a right
    angle when `|sin(2*angle)|` is at most 1e-12. One edge that fails refuses the layer. This
    allows for rounding and nothing more. An outline turned by a quarter turn with a computed
    sine and cosine passes; an edge that rises 1e-9 nm across a 200 nm outline does not, and
    neither does a rectangle at an angle of 1e-11.
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
  Measured on a metal square (`eps` -20+1.5j, side 200 nm in a 400 nm cell, 150 nm thick, normal
  incidence at 900 nm, `factorization='li'`) against a 3D finite-element reference (gyptis,
  second-order edge elements; its next coarser mesh is 0.6% of the plane's RMS field from it),
  over 2304 samples of the mid-layer plane: the default is 32 / 26 / 23 / 18% of the plane's RMS
  field away at orders 8 / 12 / 16 / 20 and the rule 35 / 30 / 27 / 23%. Inside the metal the
  rule is the closer one, 9 / 8 / 6 / 6% against 48 / 20 / 34 / 20%; outside it is not. The
  solve's `R` is 0.351 to 0.352 at those orders where the reference has 0.356. An FDTD solution
  of the same square (Meep, 5 nm cells) is 3.9% from the reference, with `R` 0.342: on a metal
  with corners it is the weaker check of the two. Against that FDTD solution, 15 nm or more off
  the walls, `'nvf'` on the same square is further off than `'li'`, and the same square as a
  dielectric (`eps` 12.25) reads 2.0 / 1.2 / 0.9% for the rule against 5.7 / 4.0 / 2.6% at orders
  8 / 12 / 16.
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

## A curved wall: `reconstruction='split'` and `'splitw'`

Across a curved wall neither `E_x` nor `E_y` is the normal component, so the line rule above has
nothing to follow and is refused. `result.field(x=..., y=..., z=..., reconstruction='splitw')`
splits the in-plane field with the wall's own normal `n` instead (M. Weismann, D. F. G. Gallagher
and N. C. Panoiu, J. Opt. 17, 125612 (2015)): the normal part is read from `D`, the part along the
wall from `E`, each is summed where it is continuous, and the first sum is divided by `eps` at the
sample. `'split'` is the same rule without the window described below. Both are opt-in, and the
default stays `'plain'`.

:::{admonition} Experimental
:class: warning

`'split'` and `'splitw'` are experimental, and every field read with one of them emits
`wavebit.rcwa.diagnostics.ExperimentalWarning` on the calling line.

- **What sets them apart.** `'plain'` and `'lalanne'` are checked two ways: at the same
  truncation against RETICOLO, an independent implementation of the same method, which says the
  code does what the method says, and against finite-element fields, which says how accurate
  the method is. These two rules have the second check only. No other solver implements them.
- **What is this package's own.** The window of `'splitw'`, and the side of a metal's wall both
  rules answer on (*Metals* below). Both were fitted to the measurements in this section, not
  derived.
- **What the measurements cover.** One circle on 22 cells and on seventeen pairs of materials;
  one ellipse and one pair of circles, each as a dielectric and as a metal; and four more
  layers (two circles of different radii, four circles, two more ellipses), each as a
  dielectric and as a metal, against references that are not stored here. One wavelength per
  cell, normal incidence, the `'nvf'` solve.
- **What may change.** Their values, the layers they accept and their names, between releases
  and without the deprecation period the rest of the interface gets.
- **How to use them.** Read the same samples with `'plain'` and at a second truncation. Away
  from the wall the rule and the sum should agree, and a value that still moves between the
  two truncations has not converged under either. How far the difference between two
  truncations goes as an estimate of the error is measured under *Estimating the error
  without a reference* below. On the cells measured the rules are closer to the reference
  than the default; the warning is about how far that has been checked, not a sign that a
  number is wrong.

A script that has taken this in can silence the one category:

```python
import warnings
from wavebit.rcwa import diagnostics

warnings.simplefilter('ignore', diagnostics.ExperimentalWarning)
```

A call the rules refuse raises and does not warn, and under `jax.jit` the warning comes when the
function is traced, not each time the compiled function runs.
:::

- **Where they apply.** A layer of a scalar permittivity in a scalar background, in a
  rectangular cell, that holds one of these, anywhere in the cell:
  - one circle;
  - one ellipse with its axes along the cell's;
  - several circles of one material that do not touch each other or each other's images.

  A stack may hold several such layers. That is what they were measured on. Uniform layers keep
  the sum bit for bit, and so do shapes with their background's permittivity unless one of the
  two is being differentiated.
- **What they change.** `E_x` and `E_y` in those layers. `E_z` and `H` are the default's, bit for
  bit: the reference stored no `H`, so the rules rebuild `E` alone.
- **How they work.** With `P` the convolution matrix of `w n n^T` and `Q` the inverse of the
  matrix of `1/eps`, `D_n = (P Q + Q P)/2 E` and `E_t = (I - P) E`, and
  `E = sum(D_n)/eps + sum(E_t)`. `n` is the circle's radial unit vector, which is right on the wall
  and jumps at the circle's axis and at the cell's edge. `'split'` uses it over the whole cell,
  `w = 1`. `'splitw'` keeps it away from both jumps with a window: `w` is 1 within `b/4` of the
  wall and falls as `cos^2` to 0 at `b`, the smaller of the radius and half the gap between the
  circle and its nearest image. `Q` takes the same floor on a negative permittivity as the `'nvf'`
  solve. For an ellipse or several circles `n` is the solve's own analytic normal field of the
  shapes, and the window is on the distance to the nearest wall, with `b` the smallest inradius:
  a circle's radius, an ellipse's shorter semi-axis.
- **What they are measured against.** A 3D finite-element reference (gyptis, second-order edge
  elements), over 2304 samples on the mid-layer plane, as the relative RMS error of the in-plane
  field, with `factorization='nvf'`:

  | Cell (nm), `eps` 12.25 in air, normal incidence | Order | `'plain'` | `'split'` | `'splitw'` |
  | --- | --- | --- | --- | --- |
  | period 400, radius 100, thickness 150, at 900 | 8 | 13.6% | 3.1% | 2.4% |
  | the same | 16 | 9.6% | 1.4% | 1.3% |
  | period 1000, radius 280, thickness 200, at 1363 (`R` = 0.77) | 8 | 9.1% | 10.9% | 5.4% |
  | the same | 16 | 6.3% | 4.4% | 2.5% |

  The first row is a test (`tests/references/test_gyptis_cylinder_field_golden.py`); the
  reference's own next mesh level is 0.11% from it. An independent FDTD solution of the first
  cell (Meep, 5 and 6.7 nm cells) agrees with the reference to 0.2% of the plane's RMS field
  20 nm or more from the wall, and to under 1% on the same cell with a metal cylinder; closer to
  the wall an FDTD cell averages across the jump and cannot say. On 16 dielectric circle cells (radius 0.15 to
  0.47 of the period, `eps` 2.25 to 25, one with loss), `'splitw'` was 0.06 to 0.67 times the
  default's error at order 12 and at most 0.64 times at order 8, on every one. `'split'` was 0.08
  to 0.85 times at order 12; at order 8 it was further from the reference than the default on two
  of the 16, the third row above among them.
- **An ellipse and two circles.** Two more cells against finite-element references of their own
  (`eps` 12.25 in air, period 400, thickness 150, normal incidence at 900, `factorization='nvf'`),
  over the same 2304 samples, which include the samples on the walls:

  | Layer (nm) | Order | `'plain'` | `'split'` | `'splitw'` |
  | --- | --- | --- | --- | --- |
  | one ellipse, semi-axes 130 and 80 | 8 | 15.5% | 5.0% | 3.2% |
  | the same | 12 | 12.8% | 3.0% | 2.3% |
  | the same | 16 | 11.3% | 2.1% | 1.6% |
  | two circles of radius 60, 180 apart | 8 | 17.2% | 4.8% | 4.2% |
  | the same | 12 | 13.7% | 3.9% | 3.6% |
  | the same | 16 | 11.4% | 2.7% | 2.3% |

  - The order 8 rows are a test (`tests/references/test_gyptis_shapes_field_golden.py`). The
    references' coarser mesh level is 0.38% from them, and an FDTD solution (Meep, 5 nm cells)
    agrees with them to 0.3% of the plane's RMS field 15 nm or more from the walls.
  - Within 10 nm of a wall the default is 35 to 51% of the plane's RMS field off and the rules 3
    to 8%.
  - The window. With `b` the smallest inradius `'splitw'` was never behind `'split'` on these two
    cells. With the smaller `b` one circle uses, half the gap where that is narrower, it was
    behind at order 8 on the two circles, 6.8% against 4.8%. One circle keeps its own `b`: with
    the inradius it was closer on five of the 16 cells above, by 8 to 45%, and 10 to 29% further
    on one, a circle of radius 0.47 of the period.
  - A circle moved across the cell gives the same field moved, exactly. These shapes are sampled
    where they are: moved together they gave the same field to 3e-4 of a field of order 1 with
    `'split'` and 3e-5 with `'splitw'`.
  - The same two layers as a metal (`eps` -20+1.5j), against finite-element references of their
    own that are not stored here. The rules answer inside the metal only, as on one circle, and
    there they were 2.1 to 3.7 times closer than the sum at orders 8, 12 and 16: 30 to 75% of the
    plane's RMS field for the sum, 11 to 32% for `'split'`, and `'splitw'` within a tenth of that.
    Over the whole plane the gain is small, 26 to 56% against 23 to 50%, because the air side keeps
    the sum. The solve's `R` is 1 to 7% from the references' on these two cells and moves with the
    order, and the two metal circles at order 12 are the worst of the six rows for the sum and
    the rules alike. See *Metals: which side of the wall* below before reading a value beside a
    metal.
- **Circles of different radii, four circles, other ellipses.** Four more layers in the same cell,
  at the same wavelength and with the same solve, each against a finite-element field of its own
  that is not stored here (one mesh level each; no test reads them). Plane error at orders
  8 / 12 / 16, as a dielectric (`eps` 12.25 in air):

  | Layer (nm) | `'plain'` | `'split'` | `'splitw'` |
  | --- | --- | --- | --- |
  | two circles, radius 70 and 50 | 13.5 / 11.3 / 9.6% | 3.2 / 2.5 / 2.3% | 3.2 / 2.3 / 1.7% |
  | four circles, radius 60, 40, 30 and 30 | 15.6 / 12.5 / 11.2% | 5.7 / 4.1 / 3.7% | 6.9 / 4.0 / 3.6% |
  | one ellipse, semi-axes 150 and 50 | 15.6 / 12.5 / 10.7% | 12.7 / 6.8 / 4.9% | 8.7 / 4.1 / 2.5% |
  | one ellipse, semi-axes 110 and 90 | 13.9 / 11.3 / 9.8% | 3.2 / 2.1 / 1.5% | 2.3 / 1.7 / 1.4% |

  - One circle of each layer is at the cell's centre; the others are at its corner and at the
    middle of two of its sides, straddle the cell's edge and are read with their images.
  - Both rules were closer than the sum on all four at every order, over the plane and within
    10 nm of a wall.
  - The four circles at order 8: `'splitw'` was behind `'split'`, 6.9% against 5.7%. The window's
    reach is the smallest inradius, 30 nm here, narrower than order 8 resolves; this is what a
    circle that nearly fills the cell showed above. Inside the circles the error falls least on
    this layer and hardly moves with the order (7.1 to 8.9% for `'split'`).
  - The 3:1 ellipse: more than 0.1 period from the wall `'split'` was further from the reference
    than the sum at all three orders (9.7 / 5.4 / 3.9% against 4.5 / 2.9 / 2.2%), though closer
    over the plane; `'splitw'` was further there at order 8 only. On a slender shape the rule
    without a window blurs the field far from the wall.
  - The same four as a metal (`eps` -20+1.5j), over the samples inside the metal, where the
    rules answer: the sum was 24 to 73% of the plane's RMS field off, `'split'` 7 to 30% (2.3 to
    5.1 times closer) and `'splitw'` 8 to 32% (2.1 to 4.4 times). `'splitw'` was not the closer of
    the two inside a metal: further than `'split'` by 1 to 26% in eleven of the twelve cases and
    level in the twelfth.
    Each plane holds one to four samples within 0.1 nm of a wall, whose side a faceted reference
    does not settle; without the samples within 0.5 nm of a wall these figures move by at most
    four points.
  - **The metal 3:1 ellipse is not converged by the solve itself.** Its `R` is 17%, 12% and 3.3%
    below the reference's at the three orders and the sum's plane error is 58%, 39% and 23%. The
    rules still bring the inside of the metal 2.3 to 2.7 times closer, to 30%, 26% and 14%. On the
    other three metal layers `R` is within 2.3% from order 12 on.
  - Not measured on these shapes: a dielectric hole in a metal, more than four circles, circles
    that nearly touch, oblique incidence.
- **Choosing between the two.** `'splitw'`, unless the circle nearly fills the cell. It was the
  closer of the two on 13 of the 16 cells at order 12. The other three have a radius of 0.40, 0.41
  and 0.47 of the period, where the window is narrower than the orders resolve: 2.3%, 2.9% and
  6.2% for `'split'` against 3.5%, 3.8% and 9.6%. A fourth cell at 0.47 went the other way, 3.8%
  against 3.0%, so compare the two there rather than assume.
- **Metals: which side of the wall.** Where the real part of the permittivity changes sign across
  the wall, both rules answer on its negative side only: inside a metal disk, or outside a
  dielectric hole in a metal. The other side keeps the sum, to rounding. Where it does not change
  sign, two dielectrics or two metals, they answer on both sides.
  - **This is an empirical rule, not a derived one.** It was fitted to the six metal circles
    below and then checked on seventeen pairs of materials. It has not been found applying the
    rule to a side that loses by it, and it leaves a gain unused at low contrast.
  - Six metal circles (`eps` -20+1.5j at four sizes, -100+5j, -5+1j; orders 8, 12 and 16):
    the rules were 2.4 to 3.9 times closer than the sum inside the metal, and closer over the
    plane on every cell at every order. At order 12 the plane error falls from 17 to 28% for the
    sum to 13 to 27%.
  - Seventeen pairs on one circle (radius 100 nm, period 400 nm, 150 nm thick, 900 nm at normal
    incidence), each against a finite-element field of its own. Here `'splitw'` was read on both
    sides, and each entry is its error over the samples of one side as a multiple of the sum's
    there, at orders 8 / 12 / 16. Below 1 the rule is the closer. The contrast is
    `|eps_negative| / |eps_positive|`.

    | Circle / around it | Contrast | Negative side | Positive side |
    | --- | --- | --- | --- |
    | 12.25 / -20+1.5j | 1.6 | 0.75 / 0.37 / 0.44 | 0.87 / 0.58 / 0.62 |
    | -20+1.5j / 12.25 | 1.6 | 0.70 / 0.56 / 0.35 | 0.86 / 0.74 / 0.53 |
    | -2+0.5j / air | 2.1 | 0.28 / 0.28 / 0.33 | 0.50 / 0.56 / 0.63 |
    | -50+3j / 12.25 | 4.1 | 0.25 / 0.37 / 0.23 | 0.83 / 0.92 / 0.70 |
    | -20+1.5j / 4 | 5.0 | 0.39 / 0.33 / 0.30 | 0.98 / 0.91 / 0.82 |
    | -5+1j / air | 5.1 | 0.30 / 0.26 / 0.27 | 0.83 / 0.83 / 0.82 |
    | -7+1j / air | 7.1 | 0.19 / 0.23 / 0.26 | 0.56 / 0.82 / 0.86 |
    | -20+1.5j / 2.25 | 8.9 | 0.21 / 0.48 / 0.21 | 0.76 / 1.17 / 0.81 |
    | -10+1j / air | 10 | 0.14 / 0.34 / 0.19 | 0.63 / 1.06 / 0.80 |
    | -15+1.2j / air | 15 | 0.19 / 0.25 / 0.32 | 1.08 / 1.33 / 1.21 |
    | -20+1.5j / air | 20 | 0.29 / 0.37 / 0.38 | 1.40 / 1.20 / 1.23 |
    | air, a round hole / -20+1.5j | 20 | 0.47 / 0.51 / 0.56 | 1.46 / 1.49 / 1.50 |
    | -100+5j / 4 | 25 | 0.28 / 0.46 / 0.29 | 1.41 / 1.32 / 1.03 |
    | -50+3j / air | 50 | 0.67 / 0.40 / 0.28 | 2.20 / 1.36 / 1.32 |
    | -100+5j / air | 100 | 0.32 / 0.36 / 0.38 | 1.56 / 1.21 / 1.48 |

    Two metals, where the rules answer on both sides (inside the circle, then outside it):
    -20+1.5j in -5+1j, 0.08 / 0.06 / 0.05 and 0.14 / 0.12 / 0.11; -5+1j in -20+1.5j,
    0.16 / 0.13 / 0.11 and 0.08 / 0.07 / 0.07.
  - **The negative side gains on every pair** (0.14 to 0.75), and so do both sides of a wall
    between two metals. That part of the rule held everywhere it was measured.
  - **The positive side gains only at low contrast.** Up to 7.1 it gained at every order on this
    circle (0.50 to 0.98, little at 5.0). At 8.9 and 10 it gained at two orders and lost at the
    third. From 15 up it lost at every order (1.03 to 2.20). The rules do not read the contrast:
    they leave the positive side to the sum at any contrast, which is right from 15 up and gives
    up the gain below. On the metal circle in `eps` 12.25 at order 12, the plane error is 39% for
    the sum, 33% as the rules answer, and would be 26% with both sides.
  - Why not open the positive side at low contrast: the edge is not a property of the materials
    alone. The ellipse above with `eps` -5+1j in air, contrast 5.1, did not gain on the positive
    side (0.97 / 0.96 / 1.09), where the circle gains at every order. At lower contrast the other
    shapes agree with the circle: 0.52 / 0.61 / 0.57 for the ellipse with -2+0.5j, and
    0.95 / 0.76 / 0.67 for the two circles of -20+1.5j in `eps` 12.25. A threshold fitted to one
    circle would be one more choice of this package's own, so the rule stays the one that has not
    been seen to lose.
  - What the solve's `R` says about it: nothing reliable. The metal circle in `eps` 12.25 has
    `R` 21%, 10% and 2% from the reference at the three orders and gains on both sides; -100+5j
    in air has `R` within 0.1% at order 16 and loses 1.48 times on the positive side. The inverse
    rule amplifies an error in the solve's high orders, and dividing by the permittivity at the
    sample shrinks it where that is large; that fits the contrast above, and it was not tested
    beyond fitting. A finer projector grid, `nvf_projector='circular'` and
    `formulation='first_order'` moved the errors of the six metal circles by 5% of themselves or
    less, in either direction.
  - So beside a metal on the air side, where a plasmonic field peaks, no rule here is better
    than the default, whose error within 0.1 period of the wall is 26 to 53% of the plane's RMS
    field at order 12 on the six metal circles. Compare two orders before reading a value there
    (*Estimating the error without a reference* below). For walls along x and y, and for one circle
    or one ellipse along the cell's axes, the solve itself can be brought closer: [Resolving walls
    with stretched coordinates](wall-resolution.md).
  - The finite-element fields of the seventeen pairs, and of the three on other shapes, are one
    mesh level each. Three of them repeat cells that have a finer reference and are 0.2 to 0.6%
    of the plane's RMS field from it, so an entry within a few hundredths of 1 is not a ranking.
    They are not stored in this repository, and no test reads them. One radius, one wavelength,
    normal incidence, one loss per permittivity.
- **Estimating the error without a reference.** Read the field at two truncations and take the
  RMS of the difference over the samples, divided by the RMS of the field.
  - On eighteen cells (one circle of radius 100 nm in the 400 nm cell: the dielectric, the
    fifteen pairs of the table above and the two pairs of metals), that figure between orders
    `N` and `N - 4` was 0.5 to 2 times the true error of the field at order `N`, taken against
    the finite-element field, in 34 of 36 cases for `'plain'` (order 12 against 8, and 16 against
    12) and in 33 of 36 for `'splitw'`. The ratio ran from 0.59 to 2.24 and from 0.64 to 2.50.
  - Every case outside that range overestimated: metal cells whose order 12 was unusually far
    and whose error more than halved by order 16. The lowest ratios, near 0.6, are the dielectric,
    the weakest metal (-2+0.5j) and the two pairs of metals with `'plain'`, where the error is
    the sum's ringing and falls slowly with the order.
  - The difference between the two readings, `'splitw'` minus `'plain'`, is not such an
    estimate. It equals the default's error on the dielectric and between two metals (0.99 to
    1.02 times), and it is 0.27 to 0.74 times that error where the permittivity changes sign,
    because outside a metal both readings are off by the same amount.
  - This is a figure over the plane and says nothing about one point at a wall. One radius, one
    wavelength, normal incidence, the `'nvf'` solve, a step of four orders, one mesh level of
    the references, which are not stored here.
- **Where they are not better, or not known.**
  - Right at the wall. On the second cell at order 16, `E_n` 0.3 nm either side of the wall is
    still 18 to 59% off the reference with `'splitw'`. The plane error is a mean; do not read a
    value within a few nanometres of the wall as converged.
  - The median. On the second cell `'splitw'`'s RMS gains 2.5 times and its median error does
    not: far from the wall in the air it is slightly worse than the default.
  - Spatial derivatives. The curl of the rule's `E` does not match the solve's `H_z` even 3 to
    20 nm from the wall, where the default's does, being one series with `H`. For a curl, a
    gradient of `|E|^2` or an optical force, read `'plain'`.
  - Not measured: shapes and arrangements beyond those above, oblique incidence, and `H`. Oblique
    incidence, both formulations, a wave given as a source and a stack of several layers run the
    same operators and are tested for consistency, not against a reference.
  - Other factorizations. Every number here is on an `'nvf'` solve. The rules also answer a
    circle solved with `'li'` or `'laurent'`, and nothing was measured there.
- **Derivatives.** With respect to the radius, the centre, a permittivity, a thickness or the
  period, the derivative agrees with a finite difference off the wall, on JAX and PyTorch. As
  with the line rule it holds each sample fixed and does not differentiate which side of the wall
  a sample is on. `'splitw'`'s reach `b` has a kink where the radius is a quarter of the cell's
  shorter side, and the two one-sided slopes differed there by 1.7 to 3.1% on the first cell
  above. Under `jax.jit` a traced material or thickness works; a traced radius or period needs a
  solve on frozen geometry (`freeze_geometry`). A trace has no value to refuse by name, so there
  a period that turns out negative, or a radius that no longer fits the cell, answers NaN in
  `E_x` and `E_y` of that layer.
- **Cost.** Each call samples the normal on a grid (2048 by 2048 for one circle, 1024 by 1024
  for the other shapes) and builds two dense operators over twice the orders: about a second
  per call at order 8, and 76 MB for each operator at order 16, 2.8 GB at order 40. Sample all
  the points of a layer in one call.
- **What they refuse, by name, with code `field.reconstruction`.** A rectangle or a polygon; an
  ellipse turned off the cell's axes, or beside another shape; circles of several materials;
  circles that touch; a raster; a tensor; a layer that names a permeability; a 1D grating; a
  skewed lattice or one with a negative side; matrices handed to `solve_conv`. For walls along x
  and y, `'lalanne'` is the rule.

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
