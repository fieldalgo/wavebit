# Level-set modeling — a design field, without the staircase and without a density gradient

Written: 2026-08. Targets `fieldwave/_core/levelset.py`, the `edge_ft` half of `_core/polygon.py`,
`_core/analytic.py::nvf_from_coeffs`, and the level-set branch of `RCWAEngine.freeze_geometry`.

---

## TL;DR

A layer can now be given as `dict(n_base=..., phi=<array>, n_index=...)`: material where
`phi > 0`. The zero contour is extracted by marching **triangles**, transformed in closed form,
and the NVF normal field is taken as `grad phi` — so a topology-optimizable design gets the
exact-boundary treatment `boundary` gives a polygon, and `d(answer)/d(phi)` comes back as a
shape derivative rather than a per-pixel density gradient. Measured against the closed-form
circle path: the permittivity coefficients converge at `O(h^2)` to **4.9e-06** relative at
`N = 512`, and the derivative converges onto the closed-form path's own `dJ/dR` — 1.63% at
`N = 64`, **0.12%** at `N = 256`. A cell-spanning stripe is exact to **8.9e-16**. Handed the
*same* `phi` at the *same* resolution, the raster path with NVF is **26–90x further** from the
truth at every truncation tested, and its error is limited by the grid rather than by `fto`.

## The gap this fills

| path | tilted / curved edge | overlapping shapes | topology |
|---|---|---|---|
| raster | staircase | as drawn | free |
| block | staircase | painted | fixed by the recipe |
| smooth | **exact** | added (superposed) | fixed by the recipe |
| boundary | **exact** | **painted** | fixed by the recipe |
| **level set** | **exact** | n/a — the field *is* the geometry | **free** |

The four shape-based paths all state the geometry as a list of primitives, so all four fix the
topology in the recipe. Topology optimisation needs it in a field, which is why level sets are
the standard representation for it — and the only way to feed one to this solver used to be to
threshold it onto the raster path, which gives up both things this path keeps:

- **the boundary.** The block path's staircase error on a curve is `3.29e-01` and **flat** in
  truncation (see [`boundary-modeling.md`](boundary-modeling.md)) — it does not shrink as `fto`
  rises, so a design whose boundary is not grid-aligned sits on that floor at every step of the
  optimisation. And NVF on a raster is *a guess at the smooth boundary the staircase stands for*
  ([`nvf-upsampling.md`](nvf-upsampling.md)), refined by resolution at best slowly: `7.6e-03` at
  `N = 30` to `3.7e-04` at `N = 480`.
- **the gradient.** A raster gives `d/d eps_pixel`, a *density* gradient; turning it into a
  boundary velocity needs a smoothed delta, and the boundary's own motion lives in pixel-scale
  jumps that autodiff steps over rather than reports. That defect is measured, on this
  repository's own sampled-field paths: 15–25% wrong at `fto` 8–12 on straight edges, against
  0.8–3.2% for the boundary path, whose field is `grad d` off a real boundary.

## What it does

```
phi (the design field)
  -> levelset.plan_layer          marching triangles, NumPy, once per topology
  -> crossing specs               (node_a, node_b, base, delta) — indices, never values
  -> levelset.boundary_and_area   t = phi_a / (phi_a - phi_b); segments and the enclosed area
  -> polygon.edge_ft              divergence theorem, exact for any straight segment
  -> levelset.normal_field        grad phi / |grad phi|, one central difference
  -> analytic.nvf_from_coeffs     the five convolution matrices
```

Same planning/replay split as [`boundary-modeling.md`](boundary-modeling.md), and marching
squares is *made of* that split, which is why it costs nothing to keep: the discrete half is the
case table, the continuous half is one linear interpolation per crossing.

## Using it

```python
N = 128
j = (np.arange(N) + 0.5) / N * period
X, Y = np.meshgrid(j, j, indexing='xy')
phi = 0.30 - np.hypot(X - 0.5, Y - 0.5)          # a disc, as a signed distance function

solver = fieldwave.rcwa(ucell=[dict(n_base=1.0, n_index=3.0, phi=phi)], ...)
res = solver.conv_solve(geometry=solver.freeze_geometry())
```

`phi` is sampled at pixel centres, `((j + 0.5)/Nx * Px, (i + 0.5)/Ny * Py)` — the convention the
rest of this codebase samples fields on. It must be at least `4*fto+1` per axis, because the
normal field is taken on the design grid itself; that is the same structural floor
`boundary.check_normal_grid` enforces everywhere, not a preference.

**Re-freeze every step.** A plan records one sign pattern. It compares *by content*, so a step
that has not moved the topology hits the JAX compilation cache; a step that has moved it gets a
new plan, and must — that change **is** the topology change. Replaying a stale plan is the one
way to get a wrong answer here that does not look wrong, so `LevelSetPlan.is_stale_for(phi)`
says directly.

## Three choices, and what each one buys

### Triangles, not squares

Marching *squares* has two ambiguous cases — the saddles — where the contour needs a separate
disambiguation rule, and the enclosed area needs one too. Resolve them differently and the two
disagree; the answer stays smooth and plausible while doing so. That matters here more than in
graphics, because **the area is the `k = 0` Fourier coefficient** and every other order comes
from the contour: a DC error is not a high-order shape approximation, it lands on the low orders
where the answer is decided (the same argument that made `polygon.ellipse_verts` area-matched).

A triangle has eight cases and no ambiguity, so the contour and the area are two readings of one
decomposition and agree by construction. The diagonal alternates on a checkerboard so the
triangulation carries no directional bias. `test_the_area_is_exactly_the_dc_fourier_coefficient`
is the statement.

### No loops are assembled, so percolation is not a special case

The transform is a divergence-theorem contour integral — a **sum over boundary segments**. The
theorem needs `dOmega` closed, not a polygon closed *in the plane*, and a region on a periodic
cell has a boundary that closes on the **torus**, where `exp(-i G.r)` is a genuine function for
every reciprocal `G`. So which loop a segment belongs to never enters the arithmetic, and a
design that percolates across the cell — the case a polygon representation has to cut and re-close
against the cell edge — needs nothing at all.

That is not a convenience argument, it is measured: a stripe spanning the full period, whose two
boundary lines leave the cell and re-enter it, transforms to the closed-form rectangle's
coefficients at **8.9e-16**, at every grid resolution. `polygon.polygon_ft` is now the special
case of `polygon.edge_ft` where the ring happens to close in the plane.

`k = 0` is the one order that would have needed the loops, and the marching-triangle area
supplies it.

### The normal field is `grad phi`

The raster path has to *guess* the normal of the boundary its staircase stands for, from a
distance transform. The boundary path reconstructs a real distance field, but by scanning every
pixel against every segment — `O(pixels x segments)`, which is what makes it expensive on a
contour with a thousand segments rather than a primitive's four. That is the cost this path
would have inherited, and it is the larger of the two terms by a wide margin:

| disc, `N = 128`, `fto = 8` — 522 boundary segments | grid | |
|---|---|---|
| `boundary`, distance scan over the same 522 segments | 33x33, the `4*fto+1` floor | 44.4 ms |
| `levelset`, `grad phi` | 128x128, the design grid | **0.16 ms** |

280x while carrying **15x more points**, because only one of the two is `O(pixels x segments)`.
The scan is on the smallest grid its transform permits; on the solver's default 256x256 normal
grid it takes **4.6 s**. NumPy, one core, and the rest of the layer build is 0.6 ms (replay) plus
7.3 ms (planning, once per topology).

**And it is the more accurate field, not a cheaper approximation of one.** Misalignment against
the exact radial normal of a disc, read in a band around the interface:

| | `N` = 64 | 128 | 256 | |
|---|---|---|---|---|
| `grad phi`, square cell | 1.0e-07 | 7.4e-09 | 4.6e-10 | converges |
| `grad phi`, 2:1 cell | 2.6e-06 | 1.7e-07 | 1.1e-08 | converges |
| distance scan off a 256-gon | — | — | — | **7.5e-05, floor** |

The scan's floor is not sampling, it is the polygonisation: 7.5e-05 is exactly the `pi/256`
edge-tilt bound, so on a polygonised shape nothing tighter is measurable and refining the grid
does not move it. The level set has no polygon in its normal field at all — the contour is
polygonal, the field is not — so there is no such floor and the error falls with the design grid
the optimiser is already refining. (Measured 2026-08; the 256-gon bound is from the boundary
path's own non-square-cell regression test.)

A level set already carries the field whose gradient is that normal. So here it is one periodic
central difference: free, unit-magnitude across the whole cell (which is what makes
`C_xx + C_yy = I`), and differentiable in `phi` like everything else. Where `grad phi` vanishes
outright there is no direction to report and `(1, 0)` is substituted; with `phi` reinitialised to
a signed distance function — which level-set schemes do anyway, for their own reasons — that
locus is measure zero, exactly as the medial axis is for the distance transform.

## What has been verified

Against the **closed-form circle** on the analytic shape path, which needs no discretisation at
all and is therefore an oracle rather than a second opinion.

**Geometry converges at second order**, which is what a midpoint-sampled decomposition owes:

| `N` | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| area error | 1.30e-04 | 3.21e-05 | 7.92e-06 | 2.01e-06 |
| `eps` coefficients, `fto` 6 (relative) | 3.20e-04 | 7.87e-05 | 1.94e-05 | 4.92e-06 |
| boundary segments | 262 | 522 | 1050 | 2102 |

**Through the solve**, `fto` 6, a dielectric disc at `n = 3` on `n_bot = 1.5`: the specular
transmitted order lands 1.78e-03 / 3.31e-04 / 1.82e-05 from the closed-form path at
`N` 64 / 128 / 256, with energy conserved to 1e-14 at every one.

**Against the raster it replaces — the same `phi`, at the same resolution.** One design field,
thresholded onto the raster path with NVF on one side and read as a contour on the other.
Distance from the closed-form circle on the specular transmitted order:

| `fto` | `N` | raster + NVF | level set | |
|---|---|---|---|---|
| 4 | 128 | 3.56e-02 | 1.38e-03 | 26x |
| 6 | 128 | 2.97e-02 | 3.31e-04 | 90x |
| 8 | 128 | 2.57e-02 | 7.77e-04 | 33x |
| 12 | 128 | 2.36e-02 | 4.46e-04 | **53x** |
| 4 | 256 | 1.56e-02 | 4.76e-04 | 33x |
| 8 | 256 | 8.44e-03 | 2.22e-04 | 38x |
| 12 | 256 | 6.42e-03 | 1.12e-04 | **57x** |

**The reference is taken at the same `fto` as the paths under test, and that is what makes the
table mean anything.** It is therefore not "the true answer" but *the same solve with exact
geometry*, so the truncation error is common-mode and what survives the subtraction is the
geometry. The tempting alternative — one converged reference at a high fixed `fto` — cannot
measure geometry at all, because at low `fto` the truncation error dwarfs it. Measured, against a
fixed `fto = 18` reference on the identical runs: 0.4x, 1.7x, 5.9x, 5.6x, 4.0x over `fto` 4 to
12, i.e. the level set reading *worse than the staircase* at `fto` 4, on a difference that is
almost entirely truncation and whose sign is a coincidence. (an experiment in the separate level-set research repository.)

What the table supports, and what it does not:

- **The level set is 26–90x closer at every truncation and grid tested.** That is the finding.
- **The raster's error is set by `N`, not by `fto`.** At `N = 128` it falls only 3.56e-02 to
  2.36e-02 across the whole truncation sweep — a factor of 1.5 — while doubling `N` buys ~3.7x.
  So it has to be paid for in resolution, and the level set at `N = 128` is already ~10x better
  than the raster at `N = 256`, on the grid the optimiser was carrying anyway.
- **The ratio does not grow monotonically with truncation, and an earlier draft of this document
  said it did.** The full sequence at `N = 128` is 26x, 90x, 33x, 38x, 53x over `fto` 4, 6, 8,
  10, 12 — quoting the endpoints as "26x to 53x" read as growth and skipped the 90x in the
  middle. The growth claim was imported from [`boundary-modeling.md`](boundary-modeling.md),
  where it *was* measured (1.08 to 7.05 over `fto` 4 to 18, against S4), and asserted here
  without being measured for this path. It does not hold on this metric. The magnitude is the
  claim; the trend is not.

**The periodic boundary costs nothing.** A full-period stripe: 8.9e-16 against the closed-form
rectangle, at every `N`. A cross percolating in *both* axes (one connected region, genus 1 on the
torus): 9.6e-04 / 4.0e-04 / 1.1e-05 against the paint-resolved two-rod reference.

**The gradient is a shape derivative, not a discretisation's own.** This is the claim that needs
an oracle and cannot be had from autodiff-versus-finite-differences: both agree on one smooth
piece of the loss by construction, on every path in this repository, and agree while skipping
whatever the jumps carry. For `phi = R - r` a uniform offset `c` is exactly `R -> R + c`, so the
closed-form circle path's `dJ/dR` is the same physical derivative reached by an unrelated
discretisation:

| `N` | 64 | 128 | 256 |
|---|---|---|---|
| `dJ/dc` vs the closed form's `dJ/dR` | 1.63% | 0.43% | **0.12%** |

at `fto` 8, on `|de_ti|` of the specular order. It converges at second order, like the geometry.
Two controls sit either side of it: the per-node gradient `dJ/d phi` is finite everywhere and its
sum equals `dJ/dc` to 1e-11 (so the field gradient and the offset gradient are one object), and
the segment vectors sum to zero to 1e-14 (so the segments really do form closed loops, which
nothing else in the build would tell you — they are never linked).

**Not settled:**

- **High index contrast.** Untouched here, and the same wall the other paths meet — see the note
  at the end of [`boundary-modeling.md`](boundary-modeling.md).
- **An optimisation actually run to convergence.** Everything above is a forward and a
  first-derivative measurement. Whether the exact boundary changes where an optimiser *lands*,
  rather than how accurately each step is scored, is a separate experiment and has not been run.
- **A sheared lattice.** Refused rather than approximated: the central difference is taken in
  index steps, which only a rectangular basis makes Cartesian. A merely non-square cell is served
  exactly, because both axes carry their own `Px/Nx`.
- **Recompilation under a real optimiser.** A level-set step usually *does* move the sign pattern,
  which is a new plan and therefore a new JAX compilation. The content-hashed plan removes the
  cost only for steps that do not.

---

See also: [`boundary-modeling.md`](boundary-modeling.md) for the polygon transform and the
plan/replay discipline this path reuses, [`nvf-upsampling.md`](nvf-upsampling.md) for what the
normal field is on the raster path and why refining it does not help, and
[`analytic-modeling.md`](analytic-modeling.md) for the closed-form shape transforms used as the
oracle throughout.

### References

The two algorithms this path is built on do not originate here, and neither does the
factorization it feeds.

- W. E. Lorensen, H. E. Cline, *Marching cubes: a high resolution 3D surface construction
  algorithm*, SIGGRAPH '87, 163–169 (1987).
  [doi:10.1145/37401.37422](https://doi.org/10.1145/37401.37422) — the 2D case is what
  `plan_layer` cuts, on triangles rather than squares to avoid the ambiguous cases; the
  ambiguity and its consequences are C. Montani, R. Scateni, R. Scopigno, *A modified look-up
  table for implicit disambiguation of Marching Cubes*, The Visual Computer **10**(6), 353–355
  (1994).
- S.-W. Lee, R. Mittra, *Fourier transform of a polygonal shape function and its application in
  electromagnetics*, IEEE Transactions on Antennas and Propagation **31**(1), 99–103 (1983) —
  the divergence-theorem transform, in `_core/polygon.py`. Stated there for a closed polygon;
  the segment-sum form used here is the same identity read on the torus.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, JOSA A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880), and P. Götz,
  T. Schuster, K. Frenner, S. Rafler, W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008) — the factorization,
  and the problem of generating its field for an arbitrary shape. A level set answers that
  question by carrying the field already.
- S. Osher, J. A. Sethian, *Fronts propagating with curvature-dependent speed*, J. Comput. Phys.
  **79**(1), 12–49 (1988) — the representation.
