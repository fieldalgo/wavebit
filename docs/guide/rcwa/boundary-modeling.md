# Boundary modeling — painting an overlap without staircasing the edges

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


:::{note}
The examples use the current `rcwa.Problem` / `rcwa.Numerics` lifecycle, except the one under
[Differentiability](#differentiability), which is kept in the earlier `wavebit.compat.legacy`
spelling it was measured in (retired 2026-10-15; a historical checkout runs it). In the current
lifecycle the same derivative runs on PyTorch, under eager `jax.grad`, and under `jax.jit` for
the shapes `Numerics(topology_check=...)` admits ([checked shape JIT](guarded-shape-jit.md)).
:::

Written: 2026-08. Targets `src/wavebit/rcwa/_engine/boundary.py`, `src/wavebit/_common/geometry/clipper.py`, `src/wavebit/_common/geometry/polygon.py`,
and the overlap route through `src/wavebit/rcwa/_engine/parametric.py::to_conv_mat_parametric_layers`.

Curved paintings now also support `nvf_projector='circular'` under NVF. The complex
projector retains the analytic extension away from curves and corrects it to the
polygon facet normal near each chord. Original edge contributions remain in the
extension when covered, so material equality does not switch the field. Clipped
regions still supply the material coefficients. See [usage, derivatives and
approximation limits](parametric-modeling.md#overlapping-circles-and-ellipses).
The default linear construction described below still extracts actual material
interfaces and keeps its existing numerical behavior.

---

## The gap this fills

Two things a layer can need at once, and until now no path gave both:

| route | tilted edge | overlapping shapes |
|---|---|---|
| raster | staircase | as drawn (painted) |
| block (removed) | staircase | **painted** |
| parametric, closed form | **exact** | added (superposed) |
| **parametric, resolved** | **exact** | **painted** |

A raster paints — the later sample wins — but it staircases a tilted edge. So did the block
path, which decomposed shapes into axis-aligned blocks. The parametric path transforms a tilted
shape in closed form but *sums* the shapes, so where two overlap the layer would get both
contributions and the permittivity there is neither shape's; asking it to paint used to raise
`NotImplementedError`, and it now refuses an overlap outright and hands the layer here.

The two cannot simply be mixed. `normal_vector_field_analytical` builds its field from each shape
in turn and never asks which boundary survived the painting, so a hybrid would put a normal on an
interface that the painting has already covered up.

**Note this is not free.** An axis-aligned overlap costs a raster nothing either — its pixels
tile the cell whether shapes overlap or not, so there is no staircase to remove. This path
earns its keep only where an edge is actually tilted.

## What it does

Resolve the painting **once** into disjoint polygons, then transform those:

```
shapes in painting order
  -> clipper.resolve_painting        Greiner-Hormann, once per layer
  -> disjoint signed loops           +1 material, -1 hole
  -> polygon.polygon_ft              closed form, exact for any straight edge
  -> interfaces.plan_interfaces     partition coincident edges, including periodic images
  -> interfaces.replay_interfaces   traced endpoints and live material-jump flags
  -> boundary.distance_gradient_...  nearest point on the *actual* interface set
  -> the five NVF convolution matrices
```

A region is a list of loops, and **the orientation is the bookkeeping**: counter-clockwise is
material, clockwise is a hole. That is what closes the recursion: subtracting from a region is
subtracting from each loop, so every clipper call stays a simple-polygon pair and the loop count
grows roughly linearly instead of the `2^m` an inclusion–exclusion formulation would need. A
hole inside a hole comes out counter-clockwise again -- material -- by composition, with no sign
arithmetic anywhere.

Both the area (shoelace) and the transform (`polygon.polygon_ft`) are signed contour integrals,
so a region's value is the plain sum over its loops and no consumer has anything to remember.

That is worth stating as a design rule rather than a convenience, because the alternative was
tried and failed. For a while every loop was stored counter-clockwise with the orientation in a
separate `sign` field. The information is the same, but it then has to be re-applied by hand at
every use: `clipper.region_area` and the since-removed `clipper.region_ft` did,
`boundary.conv_mats` did not, and
a shape painted entirely inside another one was silently solved as a *thicker* one -- with the
matrices still Hermitian and `R + T` still exactly 1. The conversion was also self-defeating:
`plan_difference` already emits a hole clockwise, and `subtract_loop` was flipping it back.

### Actual interfaces after painting (2026-09-07)

Disjoint regions are not sufficient to identify interfaces: two adjacent regions can hold the
same value. The normal field now keeps material labels through the interval planner/replay in
`src/wavebit/_common/geometry/interfaces.py`. Collinear edges are partitioned at every contact endpoint,
including periodic contacts, and their oriented contributions are summed per region before
assigning the two sides. This cancels the clipper's possible zero-area out-and-back spurs
without erasing a real third edge on the same run. An interval is a distance candidate only
when the two sides' current values differ. Material-to-background intervals follow the same rule. The comparison is
per scalar constitutive quantity: the magnetic pass supplies `Material.mu_value` in the slots
normally used for epsilon. Complex equality includes loss, and omitted mu has the physical
value 1. No material equality is frozen into the geometry plan.

The plan stores edge/endpoint references and integer image offsets. Endpoint coordinates are
recomputed during replay, so an interval endpoint moving along another edge carries a gradient.
Changing which intervals exist, their endpoint ordering, or the bounded image groups makes an
eager frozen replay stale. Under JIT, as with the clipper plan, checking is the caller's job
between traced steps. This is local differentiability at fixed topology, not a guarantee at
contact, corners, or a nearest-point tie.

Both the samples and the segments fold on the reduced basis. Image enumeration has a fractional
edge-length bound; conservative per-segment distance bounds prune impossible images. The latter
keeps small curved facets from paying for another edge's larger image ring. JAX still uses the
existing scan and checkpoint implementation.

All resolved partitions now use the same sampling anchor: the area centroid of the unwrapped
painted union, shifted to the cell center and carried back by a Fourier phase. The planner
selects one oriented representative of each unwrapped exterior interval, canceling internal
cuts and spurs before summing its moments. Periodic contacts remain in this unwrapped area
integral even when they are absent from the normal field. Previously one loop was centered and several were not, so an
off-center same-material inclusion changed quadrature simply by changing the loop count. The
material Fourier coefficients remain the original signed polygon transforms.

The regression suite is `tests/rcwa/test_polygon_interfaces.py`; the earlier field defect and the
scope of the fix are recorded in [known limits, §1](../../validation/rcwa/normal-field-known-limits.md#1-polygon-internal-and-periodic-seams--fixed-2026-09-07).
The nearest-point extension away from interfaces, its finite-grid quadrature, and RCWA
truncation convergence are separate questions. Matching a normal on a smooth interface does not
make either of the latter two exact.

## Using it

```python
from wavebit import Material, rcwa

layer = rcwa.Layer(thickness=.3, background=Material(epsilon=1.0), regions=(
    rcwa.Region(geometry=rcwa.Rectangle(size=(.34, .34), center=(-.10, -.10), angle=0.),
                material=Material(epsilon=4.0)),
    rcwa.Region(geometry=rcwa.Rectangle(size=(.34, .34), center=(.06, .06), angle=.3),
                material=Material(epsilon=9.0))))
```

Every material is a `wavebit.Material` holding a **relative permittivity** (`4.0` and `9.0` are
the `n = 2` and `n = 3` rods this example used to spell as `n_index`), and a permeability as well
when the region is magnetic; the earlier interface's retired `n_base` / `n_index` keys are
refused by name, since keeping their numbers under the new keys would paint a different
structure. See [`conventions.md`](../../reference/common/conventions.md).

Regions are in painting order — a later shape covers an earlier one. `Rectangle`, `Circle` and
`Ellipse` are accepted (the earlier interface's `square` and `disk` too); the curved ones are
polygonised at `boundary.POLYGON_VERTICES` (256), which converges on the true shape at
`O(1/n^2)` and puts that error far below the normal field's. So is a `Polygon`, given by its
vertices -- see [Polygon regions](#polygon-regions).

**Routing is on the overlap, and nothing else says it.** While no two shapes intersect,
painting and adding describe the same structure, so the layer stays on the closed-form parametric
path: exact and cheaper. Routing every shape layer here regardless would move the answer on a
layer that has no overlap at all — measured 0.8% apart on disjoint rods, because the two paths
extend the normal differently. A `combine=` key chose the rule by name until 2026-08-27, when
`'add'` was removed (its overlap is `eps_a + eps_b - eps_background`, which is not a material)
and `'paint'` stopped being a choice.

### Which route a layer took

**There is no modeling path of its own for this route**, and the omission is deliberate rather
than pending. The modeling path is a property of the whole stack, decided from what its layers
declare alone — sampled `epsilon` (`raster`) or `regions` (`parametric`). The choice between
resolving and the closed form is taken **per layer**, by the loop in
`to_conv_mat_parametric_layers`, from the shapes' actual coordinates. One parametric stack can
have its first layer resolved into polygons and its second transformed in closed form; a
stack-wide value cannot say that. Both layers are on the `parametric` path (on the earlier
interface both report `modeling_path == 'parametric'`).

Two consequences worth keeping straight:

- The rows of the table above are not settings to pick between. The first two are `structure`
  types (and the third, marked removed, was one); the last is what the parametric one
  *becomes*, layer by layer, where it has to.
- What records the choice is the geometry record's `boundary_plans`: a layer's slot holds a plan
  if that layer resolved, and `None` if it stayed on the closed form. On the earlier interface
  that is `solver.freeze_geometry().boundary_plans`, what to read when you want to know which
  happened; `rcwa.prepare` keeps the record private and its report does not say which route a
  layer took yet.

Nothing else about such a layer changes, which is the other half of why the path was not split.
The two routes share one option contract — `factorization='nvf'` and
`fourier_coefficients='integrated'`, both fixed, neither a choice — and one coordinate
convention, absolute lengths in the units of `period`. A separate path would duplicate every
entry in `engine.py`'s refusal tables and distinguish nothing a user can act on.

This argument used to be made against a document that did not take it: the capability manifest
published `smooth` and `boundary` as sibling paths until MANIFEST_VERSION `'9'` (2026-09-03).
It now publishes `parametric` with an `overlap` property, which is this section in
machine-readable form.

## Differentiability

Split in two halves, which is the whole reason a discrete geometric decision does not stop a
gradient:

- **planning** (`clipper.py`) — which vertices a loop has, which edges cross. NumPy, concrete
  values, outside any trace. It records **indices**, never coordinates: `('src', 'B', 7)`,
  `('x', 3, 5, 8, 9)`.
- **replay** (`polygon.py`, `boundary.py`) — arithmetic only. No value-dependent control flow, no
  in-place writes, so it traces and differentiates on every backend.

`freeze_geometry()` records the plan in `StaticGeometry.boundary_plans`. Because the plan is
pure Python indices it compares **by content**, so re-freezing a structure whose topology has
not changed hits the JAX compilation cache — take the freeze again every step (see the README
on stale geometry).

```python
# Historical API: the earlier interface, retired with wavebit.compat on 2026-10-15; a historical
# checkout runs this snippet as measured.
from wavebit.compat import legacy as wavebit

structure = [dict(background=1.0, shapes=[
    dict(type='rectangle', cx=-0.10, cy=-0.10, lx=0.34, ly=0.34, angle=0.0, material=4.0),
    dict(type='rectangle', cx=0.06, cy=0.06, lx=0.34, ly=0.34, angle=0.3, material=9.0)])]
solver = wavebit.rcwa(backend='jax', structure=structure, ...)
res = solver.solve(geometry=solver.freeze_geometry())
```

A gradient runs from `(cx, cy, lx, ly, angle)` through the replay, the polygon transform and the
normal field to the convolution matrices, and matches finite differences to 2.3e-08. Through the
current lifecycle `rcwa.prepare` takes the freeze and the same derivative runs on PyTorch (prepare
with `backend='torch'` and pass shape values that require grad to `prepared.run`), under eager
`jax.grad`, and under `jax.jit` or `jax.vmap` for the shapes `Numerics(topology_check=...)`
admits: fixed-cell Cartesian Li rectangles by default, the other scalar shapes with `'host'`
([checked shape JIT](guarded-shape-jit.md)).

**That agreement is pointwise, and pointwise is the one comparison that cannot fail here.** The
normal field is sampled, and the nearest-interface direction jumps across the medial axis, so as
a shape moved the solve changed by small steps as well as smoothly; a difference step smaller
than a pixel stays inside one smooth piece and agrees with autodiff by construction. Integrated
across a window several pixels wide, autodiff missed the steps' share of the change: measured
2026-10-15 on one rotated rectangle painted under a second shape, 18% to 20% of the change across
a four-pixel window, the same at `shape_nvf_grid_size` 64, 128 and 256 -- a missing term, not a
grid error. The closed-form route's blended field has no such steps (a lone rectangle closes the
same window to 5e-4), and Laurent's rule builds no field at all, so its painted-layer derivative
is exact.

**Since 2026-10-15 the field averages each switch over half a sample step, and the window
closes.** `boundary.material_projector_fields` keeps the nearest-interface extension and replaces
each of its hard choices -- the nearest interval of a material pair, the nearest of the
influential pairs -- by a soft minimum: a candidate weighs `exp(-b / w)`, `b` its distance to its
bisector with the nearest candidate and `w` half the sample step, and what is averaged is the
projector `n n^T`, which has no sign to flip across an interface. Every sample then moves
smoothly with the geometry, and the same painted pair misses 0.05%, 0.12% and 0.22% of its
window's change at grids 64, 128 and 256
(`test_a_painted_layers_position_derivative_integrates_to_the_change`). The price -- a
projector of rank two inside a switch, a different extension where material pairs of unequal
contrast mix, a straight interface under two region labels visible within a sample of where
the labels change, and the field's build time -- is
measured in [the normal field's known limits](../../validation/rcwa/normal-field-known-limits.md),
section 1b, which also says why the tensor shapes keep the hard field. See
[Polygon regions](#polygon-regions) for the measurement on outlines.

### It carried a position gradient the parametric path could not — and no longer does

**This section's advantage is gone as of 2026-08-17 (`c128fd8`), and the reasoning is kept
because the diagnosis in it is what eventually closed the gap.** The parametric path now carries the
`cx`/`cy` gradient too, so what separates the two paths is the paint-vs-add semantics this
document opens with, and nothing about differentiability.

What used to be true: a multi-shape *smooth* layer differentiated correctly with respect to
`lx`, `ly` and `angle`, and **not** with respect to `cx` and `cy`, because a rectangle's
level-set gradient was piecewise constant and the position reached the sampled field only
through `sign()`, whose derivative is zero. The variation lived entirely on the wedge lines,
which sweep across pixels as the shape moves.

The failure was sharper than "inaccurate", and the number is worth keeping. On the projector
coefficients of two rectangles, `d/dcx` by autodiff was **exactly 0.0** — and so was a central
difference once `h` was small enough to stay between two pixel crossings (0.0 at `h` = 1e-7,
1e-8, 1e-9, against -1.0 at 1e-6). The two agreed perfectly, and were agreeing about nothing:
every bit of the translation dependence sat in jumps. Two *circles*, whose normal is radial and
so genuinely smooth, agreed to nine digits (2.624456e-02 both ways) — the mechanism confirming
itself, since only the shape changed.

That last contrast is what the fix was eventually built on. The seams, not the sampling, were
the problem — confirmed by refining `shape_nvf_grid_size` from 1024 to 4096 and watching the rectangle
error stay flat at 1.8e-2 while the circle pair converged — so the answer was to remove them
rather than to mollify them, which the `tanh` route had tried and failed. Treating +x and -x
(and +y, -y) as four separate faces blended by distance products makes them cancel exactly at
the centre line, with no width anywhere. `sign()` is gone from the branch and the two-rectangle
relative position gradient reads 1e-5 against central differences. Both `xfail`s in
`tests/rcwa/test_shape_position_gradients.py` are lifted.

**This path never had the problem**, because its normal field is the gradient of the distance to
the resolved boundary: a vertex coordinate reaches it as arithmetic rather than through a sign,
and there was no normalized sum of per-shape fields at all. That is still true; it is simply no
longer a point of difference. Sweeping `h` on the five convolution
matrices of a painted pair, every parameter — `cx` and `cy` included — converges onto the
autodiff value by `h` = 1e-7 (`d/dcx` = -3.065387e+02 analytic against -3.065387e+02). The
intermediate-`h` wobble is a real slope sampled across the same pixel jumps, not an absent one.

That last comparison says less than it looks like, and the correction is worth carrying because it
is easy to repeat. Autodiff and a small-`h` central difference **cannot disagree** on either path:
both models make the loss piecewise smooth in `cx`, autodiff returns the slope of the smooth
piece, and an `h` below the pixel spacing stays inside that same piece. Their agreement is
self-consistency, and both are agreeing while they skip whatever the jumps carry. To see which
field is actually right you have to regress the loss across a window several pixels wide, so the
jumps are averaged in rather than stepped over — see the next section, where that is done and the
answer is not the one the `h`-sweep implies.

The distinction matters for what could be done about it. The measurement above holds for a
painted **disjoint** pair as well (3.878496e-01 analytic against 3.878512e-01 at `h` = 1e-7), so
the grad-of-distance field carries a position gradient whether or not anything overlaps — which
makes "give multi-shape parametric layers this field instead" a route the three already rejected at
`normal_projector_coeffs` did not consider. It is not free — it is a different NVF choice, so
forward answers on those layers move — and it has now been tried. (It is also moot since
`c128fd8`: the parametric path carries the gradient on its own field, so there is nothing left for
the swap to buy. The measurement below stands as the record of why the swap was rejected on its
own terms.)

### Tried: the gradient does not survive the swap

Disjoint layers are the whole affected population: an overlapping one already routes here, and a
lone shape is exact through the phase trick.

The comparison isolates the field and nothing else, which is what makes it worth making. `epz_i`
is `inv(ep_dir)` and touches no normal field; between the two paths it agrees to **1e-15** on
every all-rectangle case, because the polygon FT of a rectangle *is* the closed-form rectangle
transform. So `ep_dir` and `delta_ep` are shared bit for bit and the whole difference lives in
`C_xx / C_xy / C_yy`.

The gradient is the half the swap was reached for, and it is the half that fails. Measured the way
the previous section says it has to be — across a window six normal-grid pixels wide, with the
autodiff gradient sampled on the *same* abscissae as the loss, so nothing can alias — the
integral of the gradient has to equal the change in the loss across the window. Two tilted
rectangles, truncation_order 8:

| | ∫ `d/dcx` across the window | loss change across it | ratio |
|---|---|---|---|
| sampled field | +1.5341e-04 | +1.4216e-04 | **+1.08** |
| grad-of-distance field | -3.4743e-04 | +1.3942e-04 | **-2.49** |

The replacement points the wrong way, and not at one unlucky position: at **all 25** positions
across the window, and again at truncation_order 12, where the gradient scatters by 18% between positions and
every one of them still has the wrong sign. The sampled field is high by 8% at truncation_order 8 and 14% at
truncation_order 12.

A circle pair reverses the comparison rather than widening it. There the sampled field accounts
for the loss to 5% and the replacement is 21% out — so on curved shapes the swap makes the
gradient worse as well, which is the verdict the cost reaches by an independent route.

**Re-measured 2026-08-18, and the sampled column is not the field it was.** `c128fd8` replaced
the analytic normal field, so every number above with "sampled" on it was re-run; the
grad-of-distance column moved only by run-to-run noise (-2.53 to -2.49, 0.81 to 0.79), which is
the control that says the re-run is comparable. The verdict is unchanged and the reasoning below
stands. What moved:

| | before `c128fd8` | now |
|---|---|---|
| sampled, rectangles truncation_order 8 | +1.22 | **+1.08** |
| sampled, rectangles truncation_order 12 | +1.24 | **+1.14** |
| sampled, circles truncation_order 8 | +0.99 | **+0.95** |

The rectangle pair improved because that commit is what gave a rectangle a position gradient at
all — 22% high is the defect it removed, and 8% is what is left. The circle pair went the other
way, 1% to 5%, which was not predicted: the distance-product blend changed the field for curved
shapes too, and on the one case where the sampled field was already good it gave a little back.

`studies/rcwa/benchmark_boundary_normal_field/benchmark_boundary_normal_field.py` is that measurement, and it runs in about twenty
minutes. Two things in it are worth knowing before reading its numbers. The window has to be
several normal-grid pixels wide, because the jumps are what the two fields disagree about and a
narrower one steps over them; and the gradient has to be sampled on the *same* abscissae as the
loss, because the derivative fluctuates on a sub-pixel scale and a coarser gradient grid aliases
it into whatever the spacing happens to select.

### But that verdict has a condition, and it is the incidence angle

The measurement above sits at one corner of its own parameter space: high index contrast (3.0 and
2.2) **and** oblique incidence. Varying those one at a time over a fixed base — same shapes, same
period, same thickness, same normal grid, truncation_order 8, three window positions each —
`benchmark_boundary_normal_field_axes.py` gives the swapped field's `err`, where
`err = |∫ d/dcx − ΔL| / (max L − min L)` and 0 means the gradient accounts for the loss change:

| swapped `err` | oblique | normal |
|---|---|---|
| **high contrast** | **2.8 – 4.8** (sign wrong) | 0.19 – 0.36 |
| **low contrast** | 0.013 – 0.020 | 0.009 – 0.034 |

The sampled field over the same eight configurations runs 0.04 – 0.26 — mediocre everywhere and
never catastrophic. So the swap is *better than the field it would replace in three of the four
corners*, by up to 10× (0.013 against 0.128 at low contrast, oblique), and fails only where the
section above measured it. Rejecting it outright overstates what was measured; so would adopting
it.

Two further things fall out, and the first removes an axis:

- **Reflectance and transmittance cannot disagree here.** Every `R` row equals its `T` row to
  every digit printed, which is not a coincidence to be checked but an identity: the structure is
  lossless, so `T = 1 − R`, and both the integral and the loss change simply change sign while
  `err` and the ratio are unchanged. Nothing is learned by measuring both.
- **The angle decides, yet the normal field never sees the angle.** `dC/dcx` is geometry only and
  is the same at every incidence; only the weights `dL/dC` move. So the defect has to live in a
  block that oblique incidence weighs and normal incidence barely does — and running the same
  windowed check per block finds it there. `epz_i`, which touches no normal field, is exact
  (`err` = 0.000) for both constructions, which is what confirms the geometry chain is not
  involved. On the diagonal blocks the swap is already good (0.29 – 0.53) where the sampled field
  is nowhere near (87 – 190). But on **`epxy` / `epyx` both are broken** — 14 – 18 for the swap,
  33 – 51 for the sampled field — and those are exactly the off-diagonal couplings NVF adds over
  Li, the ones normal incidence hardly exercises.

That last line is the useful one: the corner where the swap fails is the corner where a defect
*neither* field escapes gets weighed most heavily. Whether the off-diagonal position gradient can
be fixed for both is open, and is a better question than which of the two fields to prefer.

The cost is that other route. The field is a pixel scan over every segment of every loop and
`POLYGON_VERTICES` is 256, so a circle carries 64× the segments a rectangle does — 512 against 8
for these two pairs. At the 1024 the solver defaults to, the swap costs **2×** on the rectangle
pair and **182×** on the circle pair, which is 0.7 s and 29 s of normal field per layer per
build.

**Those ratios used to be 12× and 486×, and they fell because the baseline got slower, not
because the replacement got faster.** `c128fd8` made the sampled field a distance-product blend
— an SDF per shape and a product over the others, and four blended faces per rectangle on top —
which costs 31.9 ms to 378.6 ms on the rectangle pair and 47.4 ms to 161.6 ms on the circles, at
grid 1024. That is a 12× and 3.4× build-cost regression, paid for the boundary exactness and the
position gradient that commit bought, and it is recorded there. Read the swap ratios as ratios
between two things that both moved. Read those as ratios: absolute milliseconds move between runs on the same machine, and
`studies/README.md` says why. A curved layer would also give up the exact Bessel transform for a
256-gon.

Forward accuracy is **open**, not settled, and the truncations reachable here cannot settle it. At
truncation_order 24 the two constructions still sit 2e-4 apart on a total — as large as the differences any
ratio between them would be reading — so neither has reached the limit such a ratio would have to
be measured against. Over seven disjoint cases the median per-order ratio comes out anywhere from
0.32× to 16× depending only on which of the two is called the limit. A number quoted for this has
to name its reference; without one it is reporting the choice of reference.

One thing the window does settle. At `truncation_order = 4` neither field reproduces the slope — the truncation_order-4
slope is about twice the truncation_order-12 one — and that is the truncation every gradient test in
`tests/rcwa/test_shape_position_gradients.py` runs at. Those tests hold autodiff to finite differences,
which is the comparison the previous section says cannot fail on either path.

### Layers of both kinds in one `structure`

They differentiate together, and the mixture costs nothing: the branch is on `plan is not None`,
which is static under a trace, and the per-layer results are only `stack`ed. Verified on a
three-layer stack — one painted-and-overlapping, one painted-and-disjoint, one summed — through
`jax.jit` and `jax.grad`: every parameter's gradient is finite, and each layer's agreement with
central differences is exactly what that layer would show on its own.

### Which traces need a freeze, and which do not

The routing test of the previous section compares coordinates. Whether a tracer *has* one is
the whole question, and for a long time this code answered it by asking the wrong thing:
`parametric._primitives` spelled its read as `float(value)` and treated any refusal as "this is
traced". A `jax.jit` tracer genuinely has no number. A **`jax.grad` tracer carries a concrete
primal** and was refusing only the spelling — so the checks were skipped in an un-jitted
gradient loop, which is exactly where an optimizer lives, and a rod grown by plain gradient
ascent ran to `lx = 911.71` on a period of 300 with `R + T = 1.000000000000` at every step.

Since 2026-08-28 the read is `backend.checked_float` — `detach`, then `float`, then
`stop_gradient` — and "concrete" is a property of the *value*:

| trace | reads a value? | needs `freeze_geometry()`? |
|---|---|---|
| eager, any backend | yes | no |
| PyTorch autograd | yes | no |
| **`jax.grad`** (un-jitted) | **yes** | **no** |
| `jax.jit`, `jit(grad)`, `vmap` | no | **yes** |

So a gradient takes every decision at the point the gradient is being taken — which is also
the point they have to be right at, and a decision taken there cannot go stale the way a held
record can. Verified against the frozen route on an overlapping pair: **bit-identical**
(−0.646606440 either way), and 4.0e-13 from PyTorch.

Under `jax.jit` there is nothing to read, the freeze is what stands in for it, and the checks
above cannot run at all — including the staleness checks. That is the argument for re-taking
the freeze every step rather than holding one; it costs 0–1 ms and its record compares by
content.

The table is the earlier interface's. Through `rcwa.prepare` and `prepared.run` the eager and
PyTorch rows hold as written. Eager `jax.grad` of a shape is checked on its concrete primal, as
a PyTorch tensor is; a shape traced under `jax.jit` or `jax.vmap` is refused before it reaches
the geometry (code `prepared.host_topology`).

### What a frozen plan does and does not survive

A plan holds no coordinates, only combinatorics — which loop holds which vertex, which edges
cross. That is what makes a shape *moving* the case it is built for: slide a rectangle and the
replay tracks it exactly, bit for bit, which is where the gradient comes from. What it cannot
survive is the combinatorics changing underneath it — which shapes cover which. Three motions
do that, and they overlap as descriptions rather than partitioning the space:

- two shapes that were disjoint begin to overlap, or stop;
- a shape that was completely covered by a later one comes out from under it;
- the resolved loops get re-cut, so a vertex ends up on a different loop.

Eagerly all three now raise `StaleGeometryError` (`IncompatibleProblemError`, code
`prepared.topology`, through `prepared.run`). Before they did, the replay took them. On a
two-rectangle layer it stayed bit-identical while the overlap held and then read 3.9e-4, 2.3e-2,
3.3e-2 wrong as the first rectangle crossed the second, with no step in between to notice. The
covered case is starker: a small circle painted under a larger one has *no loops at all*, so
sliding it out returned the identical number — 0.929791164 at every position tried — while the
true answers ran 0.9295 to 0.9536. The shape had left the structure.

The same motions reach a second kind of record, and that one used to fail somewhere else entirely.
A painted layer with nothing overlapping is summed, which is exact and cheaper, and the freeze
records that decision as a `None` plan — an answer, not a gap. Drag a shape onto another and the
answer is stale too, but the dispatch summed on the record's word and
`analytic._refuse_paint_overlap` stopped it with a `ValueError` about the summing builder, pointing
at the boundary route the caller was already inside and naming neither the record nor the fix. The
same call with no `geometry=` solves the layer fine.

Checking costs one re-plan, including the actual-interface intervals and image bounds.
The earlier clipper-only measurements (0.2% for two rectangles and 1.2% for ten at order 4)
do not measure this expanded check. Image pruning rounds segment extents outward to whole
cells so harmless length changes do not invalidate an unchanged overlap. Crossing an extent
bound, changing periodic contact, or changing the recorded reduced basis still requires a new
freeze (a new `rcwa.prepare`). Under JIT on the earlier interface there is no concrete geometry
to check, so callers must recheck eagerly between traced steps.

## Polygon regions

`rcwa.Polygon(vertices=...)`, or `dict(type='polygon', vertices=..., material=...)` on the
compatibility interface: an outline given by its vertices, `(x, y)` pairs measured from the cell
centre as `cx` / `cy` are, in order around the boundary. Added 2026-10-15; until then the
declaration was refused, because no route built it.

**A layer holding a polygon is always resolved here**, whatever it overlaps.
`parametric.layer_needs_resolving` answers it from the shape's *type*, so the routing needs no
value and is the same eager, under `grad` and under `jit`. The closed form has no normal field
for an outline: its field is built per primitive, face by face and arc by arc. A lone polygon's
plan is the trivial one-loop painting, and every reader of a primitive's
`(cx, cy, lx, ly, angle)` refuses a polygon by name (`recipes.shape_geometry`) rather than reading
its defaults as a zero-size box.

**Either winding is the same region, bit for bit.** The clipper reads each outline's winding on
concrete values and walks a clockwise one backwards (`clipper.resolve_painting`); what the plan
records is an index order, so a traced outline replays in the order it was planned on and nothing
downstream branches on the sign of an area. Taken as given, a clockwise outline would be its own
hole -- which is what S4 does: its `SetRegionPolygon` solved a clockwise rectangle at `sum(T)`
0.859 against the 0.974 of the same list counter-clockwise, without a word.

**The plan is the same in any length unit.** The clipper decides when two edges are parallel and
when a vertex lies on an outline with a tolerance that is relative to what it judges: two edges
are parallel when the sine of the angle between them is below 1e-12, and a vertex is on an outline
when it lies within 1e-12 of that outline's own size from it (`clipper.point_in_polygon`). A
layout spelled `2**k` times larger plans row for row the same, one re-spelled in nm, um or m
paints the same regions up to the rounding of its decimal conversion, and a contact is judged on
its own pair, not on the rest of the layer. Neither tolerance is finer than sixteen units in the
last place of the coordinates it reads. A region may be written whole periods from the cell, but
its coordinates then carry its outline more coarsely, and hundreds to thousands of periods out
that rounding can still misjudge an exact contact or refuse the layer (`rcwa.invalid`, from the
interface plan), as it could before; write regions near the cell, which their centres are
measured from. Until 2026-10-28 the tolerance was compared with cross products and coordinates in
the problem's unit. In the default unit, metres, an overlap in a cell of up to a few micrometres
was painted wrongly -- the later region cut out of the earlier one as a hole (`<eps>` 9.5e-3 off
for a box across an L in a 1 um cell), an earlier region taken as covered, or the overlap summed
-- and touching outlines were refused with a private exception in cells up to about 0.1 mm
(identical ones up to 1 cm), while `rcwa.check` passed and `R + T` stayed 1.

**The outline must be simple**: at least three finite vertices, no zero-length edge, no edge
doubling back, no two edges crossing or touching, and an area. A bow tie would be integrated with
winding numbers of both signs, its lobes cancelling. Checked at the declaration when the values
are concrete (`ValidationError`, codes `shape.vertices` and `shape.outline`), and again in
`check_layer_shapes`, which a traced declaration reaches when it is prepared on values.

**Periodic images are judged on the outline, not on its hull.** The clipper works in unwrapped
coordinates, so an outline reaching into a copy of itself, or into another region's copy, would be
double-counted. For a pair involving a polygon, `overlap.overlapping_outlines` measures the exact
intersection with every image whose bounding box can reach -- its area and boundary length by
Green's theorem, so `2 * area / length` is a depth held to the same `1e-6` of the period the
primitives' separating-axis test uses. A curved partner is taken as the 256-gon this route builds,
with a slack of its corners' stand-off (5.0e-5 of the semi-axis) so that touching the true curve is
not refused. Measured on the cases `tests/rcwa/test_polygon_region.py` holds: a Z 1.2 periods wide
whose arms run into the notches of its neighbours is built (its bounding box and hull both
overlap their images); a U whose arm reaches 0.05 into its own image's, and an L 1e-4 past the
period, are refused; the same L exactly spanning the period, or 1e-9 past it, is built.

**Not built for a polygon**, each refused by name: `factorization='li'` (axis-aligned rectangles
only), `nvf_projector='circular'` (it corrects the primitives' analytic extension near a curve),
tensor materials, the conforming research paths, and the parametric periodic FDTD scene.

What it has been checked against:

| check | result |
| --- | --- |
| S4 `SetRegionPolygon`, direct rule, matched orders, against `factorization='laurent'`: a triangle, an L, an L across the cell edge, the wide Z, an L with a hole and with an inclusion painted inside it, an L on a sheared cell; every order, s and p, orders 3 and 6 | **identity**, 5.8e-13 at worst (`tests/references/test_s4_polygon_golden.py`) |
| a rotated rectangle written as its four corners, against `rcwa.Rectangle`, Laurent | 5.5e-14 per order at orders (4, 4), 1.5e-13 at (8, 8) |
| the same, this route's five matrices from a rectangle dict and from its corners, NVF | to 1e-14 |
| the same, default NVF: corners on this route, the rectangle on the closed form | 7.1e-3 at orders (4, 4), 4.8e-4 at (8, 8): two normal fields, one limit |
| an axis-aligned L at normal incidence, NVF, against an exact-pixel raster under Li's rule (converged: 0.4509 / 0.4511 at orders 12 / 16) | 0.4509, 0.4508 and 0.4507 at orders 4, 8 and 12, with excursions to 0.436 between; Laurent is still at 0.504 at order 12 |

**The vertex derivative is exact under Laurent's rule** -- every coordinate of the L matched a
central difference to 6e-9 against derivatives up to 7e-2, and integrated across a window the
derivative closes on the change to the quadrature's 1.5e-4. **Under NVF it closes too since
2026-10-15**, when the painted field began to average its switches (the
[differentiability section](#differentiability) says why it had not): across the same four-pixel
window on a tilted L it misses 0.15% of the change at orders (3, 3) and grid 128, at most 0.24%
over grids 64 to 512, and 0.06% and 0.08% at orders (6, 6) and (9, 9) -- where the field taken
hard missed 9-16% at every grid, 6.9% and 3.5%. The rectangle written as corners, one corner
moved, closes to 0.014% where it missed 13-17%; a second L, of permittivity 5.76 on glass and lit
from it, whose zeroth transmitted order the hard field missed by 89% at orders (3, 3), closes to
0.25%. `test_under_nvf_the_vertex_derivative_integrates_to_the_change` holds the window, a slow
twin holds it at grid 512, and `studies/rcwa/painted_switch_average` has every number.

**An axis-aligned edge is a knife edge for a frozen plan.** The interface planner rounds each
segment's fractional extent outward to whole cells, and an axis-aligned edge's extent across its
axis is exactly zero, so the first move that tilts one changes the recorded image bounds: the
replay is refused (`StaleGeometryError`; `prepared.topology` through the lifecycle) although
nothing touches. Translating the outline, or moving an axis-aligned face along its normal,
replays. It refuses rather than answers, and the primitives share it -- a painted rectangle
turned off zero angle re-plans the same way -- but an optimiser moving the vertices of an
axis-aligned design re-prepares on its first step.

## Two properties worth knowing before you trust a number

**The nearest-interface direction is discontinuous on the medial axis**, where the closest
boundary point swaps and the direction turns by up to 90 degrees, and until 2026-10-15 the field
sampled it as it is. A *symmetric* shape therefore sat exactly on its own discontinuity: a
square's 45-degree medial axis runs through the pixel centres of a square grid, those pixels were
exact `argmin` ties, and the answer stepped as the shape was perturbed either way. Finite
differences there diverged like `1/h` (a constant jump of 0.42 straddled at every `h` from 1e-3 to
1e-6) while the analytic gradient was the one-sided value, and an optimizer started from a
perfectly symmetric shape saw that step on its first move. The field now averages the switch over
half a sample step, a tied sample reads the average of the two faces' projectors, and the step is
gone. On a 0.5 square at the cell centre, grid 128, orders (4, 4), the central difference of total
transmission in the half-width read 0.144 / 0.261 / 0.146 / 0.146 / 0.146 at `h` = 1e-3 to 1e-7
with the field taken hard -- one step straddled at 1e-4 -- and reads 0.1476 / 0.1478 / 0.1477 /
0.1477 / 0.1477 averaged. The raw probe `distance_gradient_normal_field`, which no solve reads,
keeps the discontinuity.

**The sampled normal field needs `4*m+1` points per axis**, `m` being that axis's entry of
`rcwa.Numerics(orders=...)`. Below it, `k % n` folds distinct
orders onto one bin, and the damage is structural rather than gradual — a constant normal field
stops transforming to the identity, so NVF no longer collapses onto Li's rules and picks up
off-diagonals at `±n` instead. `factorization.check_normal_grid` raises instead of quietly
mis-transforming. This is the same floor `factorization._normal_field_floor` enforces on the
raster path; the constraint belongs to the transform, not to the path that feeds it.

## What has been verified

Against **S4**, an independent RCWA implementation (Stanford Stratified Structure Solver), with
the truncation set matched exactly (`LatticeTruncation='Parallelogramic'`,
`NumBasis=(2*truncation_order+1)**2`) so both codes carry the same orders:

- **The permittivity Fourier coefficients agree to 2.2e-16** inside `|k| <= truncation_order`, on overlapping,
  low-contrast and tilted-overlapping layers alike. Geometry and transform are therefore settled,
  and any residual difference is factorisation, normal field or truncation — nothing else.
- **Painting is the right reading of an overlap.** Where both codes have converged, shape
  addition sits 8.5–68x further from S4 than this path does.
- **The advantage appears exactly where a staircase would.** Comparing per diffraction order and
  pushing both codes' truncation together to `truncation_order` 18, the distance to S4 keeps falling for this
  path while the staircase settles on a floor — its staircase error. Distances over the
  propagating orders, `sum |path - S4|`. The `block` column is the modeling path removed on
  2026-08-26; it is kept because it is the staircase this argument is *about*, and nothing that
  survives makes the same comparison (a raster staircases by sampling and has no `n_split` to
  hold fixed):

  | case | boundary | block | shape addition |
  |---|---|---|---|
  | overlap, axis-aligned | 1.66e-02 | 1.66e-02 | 2.83e-01 |
  | oblique incidence, axis-aligned | 3.09e-02 | 3.08e-02 | 3.31e-01 |
  | one rotated bar | **2.25e-02** | 4.73e-02 | 2.24e-02 |
  | rotated **and** overlapping | **2.02e-02** | 1.43e-01 | 1.74e-01 |
  | one ellipse | **1.52e-02** | 3.29e-01 | 1.51e-02 |
  | overlapping ellipses | **2.58e-02** | 1.91e-01 | 3.44e-01 |
  | overlapping circles | **1.51e-02** | 1.62e-01 | 2.00e-01 |

- **Axis-aligned overlap ties** (1.00x), as it must — there is no staircase there to beat.
- **The ratio is not a fixed number; it grows with truncation.** On the rotated overlapping case
  it runs 1.08 → 1.31 → 2.30 → 3.19 → 4.11 → 5.13 → 6.11 → **7.05** over `truncation_order` 4 → 18. Reading a
  single truncation would have got the sign wrong on a rotated bar: at `truncation_order` 8 the staircase still
  looked *closer*, because S4's own truncation error dominated.
- **Polygonising a curve costs almost nothing, and the staircase costs everything.** On a single
  ellipse — the one case where S4 holds the true closed-form shape and this path holds a 256-gon —
  the permittivity coefficients differ by 3.70e-05 (2.7e-05 relative), which is the polygonisation
  in isolation; every other case is 1e-15 because both sides then hold the same polygons. In the
  *answer* it does not show: 1.52e-02 here against 1.51e-02 for the closed-form parametric path,
  the two tracking each other all the way down. The block staircase sat at 3.29e-01 and
  **flat** — on a curve a staircase does not shrink with truncation at all.

Not settled: **high index contrast**. Over a ladder of `max n` = 1.5 / 1.9 / 2.4 / 2.9 this path
and the staircase drifted from S4 together, but S4's own `truncation_order`-to-`truncation_order` change at `n <= 2.9` is
still 1e-2 at `truncation_order` 12 — larger than the difference being measured, so nothing can be concluded
there. Trust the 1.5–1.9 band. This is a known limit of the factorisation at high contrast rather
than a property of this path; the existing paths meet the same wall.

Against the repository's own paths: two overlapping rods give 0.548127 painted here and
0.824944 if summed — and 0.547893 on the block path, which painted, and so agreed.
On a single shape, where there is nothing to resolve, the permittivity matches the parametric path
to 5e-16.

---

See also: [`parametric-modeling.md`](parametric-modeling.md) for the closed-form shape transforms this
path reuses, and [`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md) for the normal field's grid on the
raster path.

### References

Neither of the two algorithms this path is built on originates here.

- G. Greiner, K. Hormann, *Efficient clipping of arbitrary polygons*, ACM Transactions on
  Graphics **17**(2), 71–83 (1998).
  [doi:10.1145/274363.274364](https://doi.org/10.1145/274363.274364) — the clipping, in
  `src/wavebit/_common/geometry/clipper.py`. What is added here is the degeneracy handling: plain GH has no answer when
  a vertex lies *on* an edge, which grid-snapped designs produce constantly. Its tolerances are
  relative to the outlines they judge, so the plan does not depend on the length unit.
- S.-W. Lee, R. Mittra, *Fourier transform of a polygonal shape function and its application in
  electromagnetics*, IEEE Transactions on Antennas and Propagation **31**(1), 99–103 (1983) —
  the divergence-theorem transform, in `src/wavebit/_common/geometry/polygon.py`.

And one convention borrowed from a neighbouring implementation rather than a paper:

- RETICOLO polygonises an ellipse **area-preservingly** — `retu.m`, "FORME SIMPLIFIEE EN 2 D":
  "la surface totale de l'inclusion etant celle de l'ellipse". That is not a refinement but the
  defining property of its approximation, because an inscribed polygon's area deficit lands on
  the DC coefficient and pollutes every low order. Cite as Hugonin & Lalanne,
  [arXiv:2101.00901](https://arxiv.org/abs/2101.00901).
  **`polygon.ellipse_verts` inscribed** when this was written and therefore carried that deficit.
  Against the closed-form transform of a circle (`D = 0.6`, unit cell, `truncation_order = 11`),
  the error was 2.84e-5 at the default 256 vertices where the area-matched construction gives
  8.4e-10. For scale, the NVF normal-field grid floor is ~1.3e-6 at grid 1024, so polygonisation
  was then the dominant error on a curved painted shape and would stop mattering afterwards.

  The fix is one scale factor on the semi-axes, `s = sqrt(2*pi / (n*sin(2*pi/n)))` for `n`
  vertices. It reads as a magic constant, so the attribution above belongs in a comment beside
  it rather than only here. Since 2026-08-10 (`a7aa28b6`) `ellipse_verts` applies it, and its
  docstring carries the attribution.
