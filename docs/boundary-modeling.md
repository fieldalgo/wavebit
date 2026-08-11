# Boundary modeling — painting an overlap without staircasing the edges

Written: 2026-08. Targets `fieldwave/_core/boundary.py`, `_core/clipper.py`, `_core/polygon.py`,
and the `combine='paint'` route through `_core/analytic.py::to_conv_mat_smooth_layers`.

---

## The gap this fills

Two things a layer can need at once, and until now no path gave both:

| path | tilted edge | overlapping shapes |
|---|---|---|
| raster | staircase | as drawn |
| block | staircase | **painted** |
| smooth | **exact** | added (superposed) |
| **boundary** | **exact** | **painted** |

(A fifth has since been added for a geometry given as a *field* rather than as shapes — see
[`levelset-modeling.md`](levelset-modeling.md), which reuses this path's transform and its
plan/replay discipline.)

The block path paints later shapes over earlier ones, which is what a raster of the same recipe
agrees with — but it decomposes shapes into axis-aligned blocks, so a tilted edge becomes a
staircase. The smooth path transforms a tilted shape in closed form, but *sums* the shapes, so
where two overlap the layer gets both contributions and the permittivity there is neither
shape's. Asking for `combine='paint'` on the smooth path used to raise `NotImplementedError`.

The two cannot simply be mixed. `normal_vector_field_analytical` builds its field from each shape
in turn and never asks which boundary survived the painting, so a hybrid would put a normal on an
interface that the painting has already covered up.

**Note this is not free.** An axis-aligned overlap costs the block path nothing: axis-aligned
rectangles tile exactly into axis-aligned blocks, overlap or not, so there is no staircase to
remove. This path earns its keep only where an edge is actually tilted.

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

A region is a list of `(loop, sign)`. Both the area and the polygon transform are signed contour
integrals, so a hole subtracts by itself and nothing downstream special-cases one. That is what
closes the recursion: subtracting from a region is subtracting from each loop and keeping the
signs, so every clipper call stays a simple-polygon pair and the loop count grows roughly
linearly instead of the `2^m` an inclusion–exclusion formulation would need.

## Using it

```python
ucell = [dict(n_base=1.0, combine='paint', shapes=[
    dict(type='rectangle', cx=0.40, cy=0.40, lx=0.34, ly=0.34, angle=0.0, n_index=2.0),
    dict(type='rectangle', cx=0.56, cy=0.56, lx=0.34, ly=0.34, angle=0.3, n_index=3.0),
])]
```

Shapes are in painting order — a later shape covers an earlier one. `rectangle`, `square`,
`circle`, `disk` and `ellipse` are accepted; the curved ones are polygonised at
`boundary.POLYGON_VERTICES` (256), which converges on the true shape at `O(1/n^2)` and puts that
error far below the normal field's.

**Routing is on the overlap, not on the spelling.** While no two shapes intersect, painting and
adding describe the same structure, so the layer stays on the closed-form smooth path: exact,
cheaper, and identical to `combine='add'` as it was before. Sending `'paint'` down a different
path regardless would make two names for one structure return different numbers — measured 0.8%
apart on disjoint rods, because the two paths extend the normal differently.

## Differentiability

Split in two halves, which is the whole reason a discrete geometric decision does not stop a
gradient:

- **planning** (`clipper.py`) — which vertices a loop has, which edges cross. NumPy, concrete
  values, outside any trace. It records **indices**, never coordinates: `('src', 'B', 7)`,
  `('x', 3, 5, 8, 9)`.
- **replay** (`polygon.py`, `boundary.py`) — arithmetic only. No value-dependent control flow, no
  in-place writes, so it traces and differentiates on every backend.

`freeze_geometry()` records the plan in `StaticGeometry.boundary_plans`, exactly as it records the
block modeler's grid plan. Because the plan is pure Python indices it compares **by content**, so
re-freezing a structure whose topology has not changed hits the JAX compilation cache — take the
freeze again every step, as with the block path (see the README on stale geometry).

```python
solver = fieldwave.rcwa(backend=1, ucell=ucell, ...)
res = solver.conv_solve(geometry=solver.freeze_geometry())
```

A gradient runs from `(cx, cy, lx, ly, angle)` through the replay, the polygon transform and the
normal field to the convolution matrices, and matches finite differences to 2.3e-08.

## Two properties worth knowing before you trust a number

**The normal field is discontinuous on the medial axis**, where the closest boundary point swaps
and the direction turns by up to 90 degrees. A *symmetric* shape therefore sits exactly on its own
discontinuity: a square's 45-degree medial axis runs through the pixel centres of a square grid,
those pixels are exact `argmin` ties, and the answer steps as the shape is perturbed either way.
Finite differences there diverge like `1/h` (a constant jump of 0.42 straddled at every `h` from
1e-3 to 1e-6) while the analytic gradient is the correct one-sided value. An optimizer started
from a perfectly symmetric shape can see that step on its first move.

**The sampled normal field needs `4*fto+1` points per axis.** Below it, `k % n` folds distinct
orders onto one bin, and the damage is structural rather than gradual — a constant normal field
stops transforming to the identity, so NVF no longer collapses onto Li's rules and picks up
off-diagonals at `±n` instead. `boundary.check_normal_grid` raises instead of quietly
mis-transforming. This is the same floor `factorization._normal_field_floor` enforces on the
raster path; the constraint belongs to the transform, not to the path that feeds it.

## What has been verified

Against **S4**, an independent RCWA implementation (Stanford Stratified Structure Solver), with
the truncation set matched exactly (`LatticeTruncation='Parallelogramic'`,
`NumBasis=(2*fto+1)**2`) so both codes carry the same orders:

- **The permittivity Fourier coefficients agree to 2.2e-16** inside `|k| <= fto`, on overlapping,
  low-contrast and tilted-overlapping layers alike. Geometry and transform are therefore settled,
  and any residual difference is factorisation, normal field or truncation — nothing else.
- **Painting is the right reading of an overlap.** Where both codes have converged, shape
  addition sits 8.5–68x further from S4 than this path does.
- **The advantage appears exactly where a staircase would.** Comparing per diffraction order and
  pushing both codes' truncation together to `fto` 18, the distance to S4 keeps falling for this
  path while the block path's settles on a floor — its staircase error. Distances over the
  propagating orders, `sum |path - S4|`:

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
  it runs 1.08 → 1.31 → 2.30 → 3.19 → 4.11 → 5.13 → 6.11 → **7.05** over `fto` 4 → 18. Reading a
  single truncation would have got the sign wrong on a rotated bar: at `fto` 8 the staircase still
  looked *closer*, because S4's own truncation error dominated.
- **Polygonising a curve costs almost nothing, and the staircase costs everything.** On a single
  ellipse — the one case where S4 holds the true closed-form shape and this path holds a 256-gon —
  the permittivity coefficients differ by 3.70e-05 (2.7e-05 relative), which is the polygonisation
  in isolation; every other case is 1e-15 because both sides then hold the same polygons. In the
  *answer* it does not show: 1.52e-02 here against 1.51e-02 for the closed-form smooth path,
  the two tracking each other all the way down. The block path sits at 3.29e-01 and **flat** —
  on a curve its staircase does not shrink with truncation at all.

Not settled: **high index contrast**. Over a ladder of `max n` = 1.5 / 1.9 / 2.4 / 2.9 this path
and the block path drift from S4 together, but S4's own `fto`-to-`fto` change at `n <= 2.9` is
still 1e-2 at `fto` 12 — larger than the difference being measured, so nothing can be concluded
there. Trust the 1.5–1.9 band. This is a known limit of the factorisation at high contrast rather
than a property of this path; the existing paths meet the same wall.

Against the repository's own paths: two overlapping rods give 0.548127 painted here, 0.824944
summed by the smooth path, and 0.547893 on the block path — which paints, and so should agree.
On a single shape, where there is nothing to resolve, the permittivity matches the smooth path
to 5e-16.

---

See also: [`analytic-modeling.md`](analytic-modeling.md) for the closed-form shape transforms this
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
  **`polygon.ellipse_verts` currently inscribes** and therefore carries that deficit — 2.84e-5
  at the default 256 vertices against 8.4e-10 area-matched. See
  `<the manual's working notes>`.
