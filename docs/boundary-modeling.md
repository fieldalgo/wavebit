# Boundary modeling — painting an overlap without staircasing the edges

Written: 2026-08. Targets `wavebit/_core/boundary.py`, `_core/clipper.py`, `_core/polygon.py`,
and the overlap route through `_core/parametric.py::to_conv_mat_parametric_layers`.

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
  -> boundary.distance_gradient_...  normal field off the *resolved* boundary
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

## Using it

```python
structure = [dict(background=1.0, shapes=[
    dict(type='rectangle', cx=-0.10, cy=-0.10, lx=0.34, ly=0.34, angle=0.0, material=4.0),
    dict(type='rectangle', cx=0.06, cy=0.06, lx=0.34, ly=0.34, angle=0.3, material=9.0),
])]
```

Every material slot is a **relative permittivity** (`4.0` and `9.0` are the `n = 2` and
`n = 3` rods this example used to spell as `n_index`), or a `wavebit.Material` when the region
is magnetic as well; the retired `n_base` / `n_index` keys are refused by name, since keeping
their numbers under the new keys would paint a different structure. See
[`conventions.md`](conventions.md).

Shapes are in painting order — a later shape covers an earlier one. `rectangle`, `square`,
`circle`, `disk` and `ellipse` are accepted; the curved ones are polygonised at
`boundary.POLYGON_VERTICES` (256), which converges on the true shape at `O(1/n^2)` and puts that
error far below the normal field's.

**Routing is on the overlap, and nothing else says it.** While no two shapes intersect,
painting and adding describe the same structure, so the layer stays on the closed-form parametric
path: exact and cheaper. Routing every shape layer here regardless would move the answer on a
layer that has no overlap at all — measured 0.8% apart on disjoint rods, because the two paths
extend the normal differently. A `combine=` key chose the rule by name until 2026-08-27, when
`'add'` was removed (its overlap is `eps_a + eps_b - eps_background`, which is not a material)
and `'paint'` stopped being a choice.

### Which route a layer took

**There is no modeling path of its own for this route**, and the omission is deliberate rather
than pending. `modeling_path` is a property of the whole solver, decided from the *type* of
`structure` alone — `raster` or `parametric`. The choice between resolving and the closed form
is taken **per layer**, by the loop in `to_conv_mat_parametric_layers`, from the shapes' actual
coordinates. One parametric `structure` can have its first layer resolved into polygons and its
second transformed in closed form; a solver-wide value cannot say that. Both layers report
`modeling_path == 'parametric'`.

Two consequences worth keeping straight:

- The rows of the table above are not settings to pick between. The first two are `structure`
  types (and the third, marked removed, was one); the last is what the parametric one
  *becomes*, layer by layer, where it has to.
- What records the choice is `geometry.boundary_plans`: a layer's slot holds a plan if that layer
  resolved, and `None` if it stayed on the closed form. That is what to read when you want to
  know which happened.

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
solver = wavebit.rcwa(backend=1, structure=structure, ...)
res = solver.solve(geometry=solver.freeze_geometry())
```

A gradient runs from `(cx, cy, lx, ly, angle)` through the replay, the polygon transform and the
normal field to the convolution matrices, and matches finite differences to 2.3e-08.

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
the problem — confirmed by refining `normal_grid` from 1024 to 4096 and watching the rectangle
error stay flat at 1.8e-2 while the circle pair converged — so the answer was to remove them
rather than to mollify them, which the `tanh` route had tried and failed. Treating +x and -x
(and +y, -y) as four separate faces blended by distance products makes them cancel exactly at
the centre line, with no width anywhere. `sign()` is gone from the branch and the two-rectangle
relative position gradient reads 1e-5 against central differences. Both `xfail`s in
`tests/test_shape_position_gradients.py` are lifted.

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

`studies/benchmark_boundary_normal_field.py` is that measurement, and it runs in about twenty
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
`tests/test_shape_position_gradients.py` runs at. Those tests hold autodiff to finite differences,
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

### What a frozen plan does and does not survive

A plan holds no coordinates, only combinatorics — which loop holds which vertex, which edges
cross. That is what makes a shape *moving* the case it is built for: slide a rectangle and the
replay tracks it exactly, bit for bit, which is where the gradient comes from. What it cannot
survive is the combinatorics changing underneath it — which shapes cover which. Three motions
do that, and they overlap as descriptions rather than partitioning the space:

- two shapes that were disjoint begin to overlap, or stop;
- a shape that was completely covered by a later one comes out from under it;
- the resolved loops get re-cut, so a vertex ends up on a different loop.

Eagerly all three now raise `StaleGeometryError`. Before they did, the replay took them. On a
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

Checking costs one re-plan. That is **0.2%** of a frozen `solve` for the two-rectangle
layer above and **1.2%** for ten overlapping shapes at `truncation_order = 4`, falling as `truncation_order` rises (0.4% for
those ten at `truncation_order = 8`) since the solve grows faster than the painting does. The freeze on this
path buys traceability rather than a speed plateau, so that is not traded against anything. Under a trace there is nothing to
re-ask and the replay goes through on trust, the same documented limit every record here has,
and the reason the instruction above is *every step* rather than *once*.

## Two properties worth knowing before you trust a number

**The normal field is discontinuous on the medial axis**, where the closest boundary point swaps
and the direction turns by up to 90 degrees. A *symmetric* shape therefore sits exactly on its own
discontinuity: a square's 45-degree medial axis runs through the pixel centres of a square grid,
those pixels are exact `argmin` ties, and the answer steps as the shape is perturbed either way.
Finite differences there diverge like `1/h` (a constant jump of 0.42 straddled at every `h` from
1e-3 to 1e-6) while the analytic gradient is the correct one-sided value. An optimizer started
from a perfectly symmetric shape can see that step on its first move.

**The sampled normal field needs `4*truncation_order+1` points per axis.** Below it, `k % n` folds distinct
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
path reuses, and [`nvf-upsampling.md`](nvf-upsampling.md) for the normal field's grid on the
raster path.

### References

Neither of the two algorithms this path is built on originates here.

- G. Greiner, K. Hormann, *Efficient clipping of arbitrary polygons*, ACM Transactions on
  Graphics **17**(2), 71–83 (1998).
  [doi:10.1145/274363.274364](https://doi.org/10.1145/274363.274364) — the clipping, in
  `_core/clipper.py`. What is added here is the degeneracy handling: plain GH has no answer when
  a vertex lies *on* an edge, which grid-snapped designs produce constantly.
- S.-W. Lee, R. Mittra, *Fourier transform of a polygonal shape function and its application in
  electromagnetics*, IEEE Transactions on Antennas and Propagation **31**(1), 99–103 (1983) —
  the divergence-theorem transform, in `_core/polygon.py`.

And one convention borrowed from a neighbouring implementation rather than a paper:

- RETICOLO polygonises an ellipse **area-preservingly** — `retu.m`, "FORME SIMPLIFIEE EN 2 D":
  "la surface totale de l'inclusion etant celle de l'ellipse". That is not a refinement but the
  defining property of its approximation, because an inscribed polygon's area deficit lands on
  the DC coefficient and pollutes every low order. Cite as Hugonin & Lalanne,
  [arXiv:2101.00901](https://arxiv.org/abs/2101.00901).
  **`polygon.ellipse_verts` currently inscribes** and therefore carries that deficit. Against
  the closed-form transform of a circle (`D = 0.6`, unit cell, `truncation_order = 11`), the error is 2.84e-5
  at the default 256 vertices where the area-matched construction gives 8.4e-10. For scale, the
  NVF normal-field grid floor is ~1.3e-6 at grid 1024, so polygonisation is currently the
  dominant error on a curved painted shape and would stop mattering afterwards.

  The fix is one scale factor on the semi-axes, `s = sqrt(2*pi / (n*sin(2*pi/n)))` for `n`
  vertices. It reads as a magic constant, so the attribution above belongs in a comment beside
  it rather than only here.
