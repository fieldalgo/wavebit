# The raster normal-field constructions, and the `normal_method` option

Written: 2026-08-17. Measured 2026-08-16/17 on CPU, NumPy backend, float64.

Targets:
- `wavebit/_core/normal_methods.py` — the three constructions added beside the original
- `wavebit/_core/factorization.py::normal_vector_field` — the dispatcher
- the `normal_method` solver keyword — `'distance'` (default) | `'contour'` | `'spectral'`
  | `'smooth'`

Companion: [`nvf-upsampling.md`](nvf-upsampling.md) diagnosed the defect this option
addresses and records the baseline measurements this file compares against.

---

## TL;DR

The shipped raster normal field quantizes its directions to essentially {0, 45} degrees
(~14 deg mean error in the boundary band, flat in N — refining the raster does not help).
Three replacement constructions were prototyped and measured at the solver level against
staircase-free analytic references. **`'contour'` is the one to use**: of the 12 measured
cells, 11 are resolved by the reference floor and it won all 11, losing none, by factors up
to 14x. `'smooth'` had the *best* angular numbers and still lost every resolved cell on the
corner-bearing scene (5 of 5, the sixth unresolved) — the measured demonstration that the
best angular field is not the best solver field. The default stays `'distance'`: every
existing result is bit-identical unless the keyword is set. (True of the *keyword*, and no
longer of the construction: `339df21` made `'distance'` itself periodic, which moved its answers
— see [the periodicity section](#periodicity-the-cell-is-a-torus).)

```python
wavebit.rcwa(structure=raster, factorization='nvf', normal_method='contour', ...)
```

The keyword is read exactly where a raster normal field is built — a raster `structure` under
`factorization='nvf'` (and the tensor-raster builders, called directly). Explicitly set
anywhere else it is refused rather than ignored, for the same reason `enhanced_dfs` was
removed outright: a knob nothing reads is a lie waiting to be believed.

## The defect, in one paragraph

`'distance'` reads `np.gradient` of the Euclidean distance to the interface set. Central
differences of a distance-to-a-digital-set over a two-cell span land on small integer
pairs — `{1,0}`, `{2,0}`, `{1,1}` — so in the boundary band the emitted direction comes
from the short list `arctan(p/q)`, in practice {0, 45} deg. The mean angular error is
~14.4 deg in the 0-1 px band with a p90 of ~30 deg, and every band is flat in N from 64
to 512: a finer staircase is the same staircase. Full measurements, including why
upsampling makes it worse, in [`nvf-upsampling.md`](nvf-upsampling.md).

Every raster-native construction in the literature breaks the quantization the same way —
low-pass the indicator *before* differentiating, keep the boundary's location from the raw
raster — and differs only in how the boundary normal is extended into the bulk (Gotz 2008:
Shepard interpolation; fmmax: a quadratic fit over Fourier coefficients; subpixel-smoothed
projection: a filtered level set). The three constructions here are three points on that
same line.

## The constructions

Everything below is eager NumPy/SciPy for every backend, like `'distance'` itself: the
normal field is frozen geometry, a constant to the solve, outside the AD graph (fmmax
stop-gradients its field for the same reason). Backend agreement was measured: numpy, jax
and torch give the same efficiencies to ~1e-14. Sign is free throughout — the
factorization consumes `n n^T` only.

### `'contour'` — subpixel level set, exact distance-to-curve direction

Smooth the two-phase indicator with a periodic Gaussian (`sigma = 0.25 * sqrt(N)` px, N
the geometric-mean base grid size), extract the 0.5 level set as subpixel polylines
(contourpy — the one optional dependency, `pip install wavebit[contour]`), resample
finely, and give every pixel the direction to its exact nearest point on the curve, which
*is* `grad(distance-to-curve)` — the exact version of what `'distance'` approximates on
the integer grid. Below half a pixel the local segment normal stands in.

The sigma policy is the sweep's choice, not a taste call: every fixed-in-px sigma leaves
the angular error flat in N (a fixed kernel keeps a fixed fraction of the staircase
wiggle); every `sqrt(N)` policy converges; the coefficient is the smallest of the top
group, tie-broken by the measured thin-feature erasure threshold (~1.4 sigma px). In
physical units sigma is `0.25/sqrt(N)` of the period — it vanishes as the raster refines,
which is what convergence *is* here.

### `'spectral'` — band-limited spectral gradient, bulk-propagated

FFT the indicator, multiply by an isotropic Gaussian low-pass at a **fixed 8 orders per
period**, read the spectral gradient (`i*k`), normalize on the interface band, and give
every other pixel its nearest band pixel's normal (EDT with `return_indices`). Periodic
by construction, in both stages. No dependency beyond NumPy/SciPy.

This used to read "`np.gradient` is one-sided at the array edge, so `'distance'` is
structurally wrong for a shape crossing the wrap and this construction is not." True as
far as it went, and it went nowhere — the observation sat here as a selling point for
`'spectral'` for two months without anyone chasing it into `'distance'`, where it was a
defect rather than a trade-off. It was also only a third of the story. See
[the periodicity section](#periodicity-the-cell-is-a-torus).

The cutoff is fixed in physical orders deliberately: every N-tied cutoff plateaus exactly
like `'distance'` (tying the band to the raster re-admits the staircase harmonics), and
hard cutoffs floor out on their own Gibbs ringing. The *value* 8 is pinned at the low end
of the band the factorization itself reads (orders to `2*max_order`, `max_order >= 4` in practice) —
a circle harness cannot choose it, because a low-passed circle stays radially symmetric
and smaller is always better there.

### `'smooth'` — derivative-of-Gaussian of the signed distance

The nearest construction to `'distance'` that removes the quantization: periodic *signed*
EDT (the unsigned distance-to-band has a crease at the interface — it rises on both
sides — and smoothing across the crease cancels gradients in exactly the band that
matters), then a derivative-of-Gaussian at `sigma = N/16` px per axis. Kept because it is
instructive and occasionally right (genuinely smooth layers), but see the ranking: it
rounds corners at period/16 and measurably loses on corner-bearing shapes.

## Measurements

### Angular error against the exact normal (circle r=0.30, 0-1 px band, mean)

| field | N=64 | N=128 | N=256 | N=512 | p90 at N=512 |
|---|---|---|---|---|---|
| `distance` | 14.8 | 14.1 | 14.3 | 14.6 | ~30 |
| `contour` | 1.33 | 1.28 | 0.50 | 0.37 | 0.81 |
| `spectral` | 3.08 | 1.43 | 0.38 | 0.16 | 0.36 |
| `smooth` | 0.52 | 0.37 | 0.04 | 0.03 | 0.05 |

Degrees, modulo 180. Only `distance` is flat in N; the 30-40 deg substitution outliers
(a pixel emitting 45 where its neighbours emit 0) are gone from all three replacements.
Note `smooth` winning this table and losing the next one — the tables must be read
together.

### Solver-level A/B, circle (vs the analytic reference at max_order=14)

Same setup as the `li` comparison in [`nvf-upsampling.md`](nvf-upsampling.md): circle
r=0.30, index 2.4, thickness 0.30, wavelength 0.55, theta 12 / phi 30 / psi 40 deg.
Error = max per-order |R - R_ref|, |T - T_ref| over shared orders. All cells resolved
(gaps above the reference's ~1.7e-3 own error).

| N | max_order | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|---|
| 64 | 6 | 2.15e-02 | 1.06e-02 | 1.13e-02 | 9.72e-03 |
| 64 | 10 | 8.75e-03 | 1.13e-03 | 1.03e-03 | 1.04e-03 |
| 128 | 6 | 1.95e-02 | 1.31e-02 | 1.31e-02 | 1.22e-02 |
| 128 | 10 | 8.00e-03 | 3.30e-03 | 3.14e-03 | 3.03e-03 |
| 256 | 6 | 1.43e-02 | 9.64e-03 | 9.79e-03 | 9.16e-03 |
| 256 | 10 | 3.51e-03 | 2.43e-04 | 2.42e-04 | 4.35e-04 |

Every replacement beats `distance` in every circle cell, up to 14x at 256/10 — and
`contour` at N=64 already beats `distance` at N=256.

### Solver-level A/B, rotated rectangle (vs the analytic reference at max_order=18)

The corner-bearing scene: rectangle 0.5 x 0.3 rotated 30 deg, same stack and incidence.
The max_order=14 reference could not resolve this scene (its own error floor ~1.7e-3 swamped
the differences), so the reference was rebuilt at max_order=18; the floor |max_order 16 - max_order 18| is
2.09e-4, and cells are called only above it.

| N | max_order | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|---|
| 64 | 6 | 1.37e-02 | **1.30e-02** | 1.49e-02 | 1.43e-02 |
| 64 | 10 | 3.89e-03 | **3.02e-03** | 4.12e-03 | 3.81e-03 |
| 128 | 6 | 1.09e-02 | **1.06e-02** | 1.08e-02 | 1.39e-02 |
| 128 | 10 | 2.02e-03 | **1.39e-03** | **1.21e-03** | 3.72e-03 |
| 256 | 6 | 1.06e-02 | 1.06e-02 | **1.03e-02** | 1.35e-02 |
| 256 | 10 | 1.32e-03 | **9.48e-04** | **1.04e-03** | 3.50e-03 |

Bold = resolved win over `distance`. Tallies on this scene: `contour` 5 wins, 1
unresolved (256/6), 0 losses; `spectral` 3 wins, 2 losses (64/6 clearly, 64/10 at the
floor's edge), 1 unresolved; `smooth` 5 losses, 1 unresolved — its period/16 smoothing
rounds the corners, and no angular table can see that. (The `spectral` column is the
post-review construction, whose bulk copy is periodic; the change moved its cells by
about the reference floor and split one previous unresolved either way.)

**The combined ranking: `contour` won all 11 resolved cells of 12, zero losses
anywhere.** That is
the whole case for the recommendation, and also the measured warning: the sweep that
froze `smooth`'s policy saw only the circle, and the circle is generous to smoothing in a
way general shapes are not — a corner-bearing scene belongs in any future selection loop.

### Where improvement is possible at all — the truncation floor

The analytic path itself (exact geometry, exact normal field) at the working `max_order`,
against the finest references:

| scene | max_order=6 | max_order=10 |
|---|---|---|
| circle | 1.06e-02 | 1.13e-03 |
| rect30 | 1.16e-02 | 1.57e-03 |

Error is (truncation) + (normal-field error), and no normal field can go below the first
term. Read the tables above against these floors: every max_order=6 cell sits at ~1e-2 for
every method because the floor is ~1.1e-2 there — `contour`'s 1.06e-02 at circle 64/6
*is* the floor, i.e. 100% of the removable error removed. The dramatic cells are the ones
where the normal field dominated the budget (circle max_order=10), and the flat-looking rect30
cells are flat because `distance` was already near the floor on straight edges. Raising
accuracy past these numbers is a matter of `max_order` and raster resolution, not of the normal
field — with the asymmetry that `distance` eats into higher-`max_order` gains (its staircase
error becomes the new floor) while the replacements keep converging.

### fmmax's construction, measured and left out

fmmax's NORMAL scheme (gradient band-limited to the expansion's own orders, one-Newton
quadratic fit) was measured through the same harness as an external yardstick: it wins
all six circle cells (1.5-9.1x) and loses 4 of 6 resolved rect30 cells — the fixed
441-term bandlimit rounds corners too. Since it beats `contour` nowhere and would carry a
jax + fmmax dependency (or a reimplementation), it was not integrated. The transferable
part of its design — the cutoff tied to the expansion rather than to the raster — is
exactly what `'spectral'` adopts.

## Periodicity: the cell is a torus

Fixed 2026-08-17, pre-existing since the port. **`'distance'` was not periodic**, so
rolling a raster by whole pixels — an exact lattice translation, the same structure named
differently — moved `max|dR|` by ~1e-2, at every `max_order`, non-monotonically. Li's rules are
exact to 1e-15 on the same rolls.

It does not converge away, and that is what makes it a defect rather than a residual: the
lift to the `4*max_order+1` floor is a block repeat, the same staircase spelled with more pixels
([`nvf-upsampling.md`](nvf-upsampling.md)), so raising `max_order` never refines the field. It
only lets the convolution matrices read more orders of a field that is wrong by the same
amount.

**Three ingredients, and no partial fix helps.** The construction read the cell as a flat
array in three places:

| | before | after |
|---|---|---|
| `_interface_mask` | compared inside the array only — the last row's neighbour was not the first | wrapping comparison on both axes |
| `distance_transform_edt` | measured to the in-cell interface | 3x3 tiling, cropped back — the periodically nearest one |
| `np.gradient` | one-sided on the first/last row and column | wrapping central differences |

Every one- and two-way combination still leaves 7e-2 to 1e0 on some placement; only all
three give exact zero. `_upsampled_interface_mask` (the tracer route) gained the matching
wrap pair, so the two mask paths stay identical.

**It is not a features-touch-the-edge bug**, which is the natural guess and the wrong one.
A shape three clear pixels from either wall, before and after the roll, still moved by a
full 90 degrees at 74 of 256 pixels — the distance map is nonzero right up to the array
border whatever the shape does.

**What it cost, and what it did not.** The reported case goes 1.7e-2 to ~2e-15 at every
`max_order`. Cost is ~3 ms on an 11.6 s solve at `max_order=[16,12]` (the EDT is 11x dearer in
isolation and irrelevant in context). Accuracy against a converged reference is a wash —
neither arm is better, because this was never an accuracy fix. Two test thresholds moved,
both calibrated against the *old* non-periodic reading rather than against the field:

- a stripe's medial set is **two** full columns in a periodic cell, not one, so its
  zero-gradient share was understated by exactly one column;
- `test_nvf_non_square_grid`'s whole-cell alignment against a radial-from-the-centre
  oracle fell 0.9903 to 0.9887 at n=64, because that oracle stops describing the bulk once
  wall pixels point at the nearest *copy*. Restricted to a band around the circle the
  numbers are **bit-identical** before and after, at all three n — nothing moved at the
  interface.

**The other three.** `'smooth'` was already exactly covariant and needed nothing — it
already tiled 3x3 and smoothed with `mode='wrap'`, which is the in-repo evidence that
periodicity is the answer rather than a tuned tolerance. `'spectral'` leaked only through
the shared mask and is fixed by it.

`'contour'` had a **second, independent defect**, and it needed two changes rather than
one. Its level set was extracted through a `4*sigma+3` px window — about five pixels on a
16 px cell, far narrower than a period — so the aperture cut the periodic copies at a
fixed place while a roll slid the shape past it.

Widening the pad to a full cell is only half of it, and measuring the half is what showed
the other: it takes an interior roll from 1.7e-3 to 1.5e-14 and a *wrap-crossing* one from
2.1e-3 to 9.8e-4, sometimes worse. Even at a full cell a straddling shape still has copies
clipped by the outer boundary — nine whole loops become six whole and six fragments — and
a fragment is harmful twice: its endpoints are not curve ends, so `_assemble` points them
at themselves and the segment normal there is one-sided; and `_resample` samples an open
curve by `linspace` and a closed one by `arange`, so fragments inject sample positions no
whole loop has (67 distinct points became 129). So each closed loop is folded back to the
centre cell and deduplicated, leaving one representative, and the 3x3 tiling regenerates
the copies. Open curves survive only when no closed loop does — a percolating stripe has
no closed 0.5 crossing at all.

Both halves together: 1e-14 on every shape tried, including nested loops, several disjoint
loops, and both roll axes. Cost is neutral to favourable, because the dedup shrinks the
kd-tree's input — 1.06x at N=64, 0.79x at N=1024.

**This one is a covariance fix too, not an accuracy fix, and the temptation is to oversell
it.** The placement dependence it removes is worth ~0.06 deg on a resolved circle
(1.79/1.75/1.81 across centred/wrap/corner at N=64, against 1.79/1.79/1.78 after) and the
angular error itself does not move at all: 1.79 at N=64, 0.86 at N=128, 0.49 at N=256,
before and after alike. A 20% gap does appear at N=48 with r=0.22 — a circle with almost no
pixels on its boundary — and generalizing from it would have been wrong. What is bought is
that the same structure gives the same answer: 1e-4 to 1e-15 at the solve.

Note that a raster with 3+ distinct values delegates to `'distance'`, so all four methods
return bit-identical numbers on one — which is what hid the defect's location originally.
Use a two-phase raster to tell the constructions apart.

Pinned by [`tests/test_nvf_lattice_translation.py`](../tests/test_nvf_lattice_translation.py),
which parametrizes three placements deliberately (a partial fix passes some). `'contour'` was
carried there as a strict xfail until the aperture fix above promoted it; all four constructions
are covariant now and the file has no xfail left. The sibling defect in the *analytic* path was
older and is also fixed (`c128fd8`) — its normal field was distance-blind, not merely aperiodic,
and `test_shape_position_gradients.test_a_multishape_layer_is_periodic_across_the_cell_edge` now
states the property where a test of the opposite name used to pin the defect.

## Semantics worth knowing

- **The default is untouched.** `normal_method='distance'` and an unset keyword are
  bit-identical, pinned by test — the two spellings select the same construction. That is a
  statement about the keyword only. The construction it selects was itself changed on
  2026-08-17 (`339df21`, periodicity) and `tests/data/golden_nvf.npz` was re-frozen with it, so
  "no golden moved" held when this option landed and does not hold across that commit.
- **Refused where unread.** Explicitly set with `factorization='li'` or on either shape
  path, the solve raises `NotImplementedError` naming the keyword. The shape paths build
  their normal fields from the geometry itself (`analytic.normal_vector_field_analytical`);
  there is nothing to select.
- **The frozen-geometry record carries the method.** `freeze_geometry()` under one method
  and `solve(geometry=...)` under another raises `StaleGeometryError`: same raster,
  same mask, different construction is a different field. Records frozen before this
  option existed replay as `'distance'`.
- **Tracers.** The three replacements read the layer's *values* (their indicators need to
  know which phase is which), so a jax-traced layer without concrete values cannot build
  them inline; the error says to `freeze_geometry()` first. `'distance'` keeps its
  mask-only traced path. Torch grad tensors are fine either way (detached eagerly, like
  the shipped field always did).
- **The floor lift keeps the base scale.** A layer block-repeated to the `4*max_order+1` floor
  hands the constructions its *base* shape, and the pixel-unit policies scale by the lift
  factor — smoothing over the base grid's features, not the lifted grid's respelling of
  them. Pinned by a test that runs the same construction base-aware and base-blind and
  requires them to differ.
- **The field reads geometry, not values.** The indicators are corner-canonical, so
  swapping the two material values changes nothing (pinned to <1e-4 deg by test), and
  the frozen record carries the phase count — the one value-derived fact the interface
  mask cannot see (merging two never-adjacent phases moves no interface but changes
  which construction runs; the replay refuses it).
- **Grayscale/continuous rasters take `'distance'`, and that is now a measured choice
  rather than a gap.** All three constructions are two-phase; 3+ distinct values —
  including the filtered densities of a mid-run topology optimization — delegate to
  `'distance'`, documented rather than silent-by-omission. The SSP-style filtered-density
  indicator (arXiv:2503.20189) this paragraph used to name as the follow-up has since
  been built and measured (2026-08-17, branch `claude/nvf-grayscale-contour`): read the
  graded band's own mid-range level set of `Re(eps)` subpixel, take the exact
  distance-to-curve directions off it, delegate binary layers to `'contour'`, and extend
  the frozen record with a digest of the normalized values (a graded band can move
  *inside* its pixels without moving the interface mask, which the geometry-only record
  cannot see — that record fix is the one part a future attempt should keep). Its
  angular fidelity is dramatic — 1.2 deg mean on a graded band against 52 deg for
  `'distance'`, whose mask degenerates over a thick band — **and the advantage does not
  reach the solve.** On SSP-projected scenes at `beta = inf` against same-`max_order` exact
  references, no construction dominates: smoothing-based variants win a round scene by
  up to 3.3x and lose a cornered one ~2-3x on resolved cells, the raw level set roughly
  ties `'distance'` on corners and trails on circles, and at moderate `beta` the
  projection's own blur floors every method equally. The graded permittivity already
  carries the subpixel boundary, so the normal field's job shrinks to a scene-dependent
  residual — the same eps-normal error correlation the staircase decomposition measured,
  now on the other side of the fence. A converged topology-optimization A/B
  independently put contour-vs-distance normals at 0.13% of the outcome. Keep
  `'distance'` on graded rasters; revisit only with a construction that survives the
  corner scene.
- **Covariance under a mirror** is bit-exact both ways for `'distance'` and `'smooth'`.
  `'contour'` and `'spectral'` carry tie-break and resampling-phase asymmetries (worst
  along x, concentrated on the periodic medial ridge where the direction is genuinely
  ambiguous): at the solve, a structure and its x-mirror differ by `max|dR| ~ 1.6e-4`
  against ~1e-15 for the exact two — an order below these methods' accuracy gains at the
  same settings, and pinned by test at 5e-4. Need exact output mirror symmetry? Use
  `'smooth'` or `'distance'`.
- **Sweeps should freeze.** An unfrozen `solve` rebuilds every layer's normal field
  each call; the smoothing constructions raise that cost from milliseconds to
  0.2-5 s/layer at large N. A wavelength sweep over fixed geometry wants
  `freeze_geometry()` once and `solve(geometry=...)` per point — the pattern
  [`backend-choice.md`](backend-choice.md) already recommends for fits.
- **Multi-phase rasters** (3+ distinct values) delegate to `'distance'` under `'contour'`
  and `'smooth'` — the two-phase indicator cannot say which of three materials is inside,
  and the distance mask sees every interface. `'spectral'` uses `Re(eps)` as its source
  there (exact for two phases, approximate beyond), falling back to `'distance'` only if
  the band has no spectral contrast at all.
- **Thin features** under `'contour'`: a strip narrower than ~1.4 sigma px loses its 0.5
  crossing and vanishes from the *field* (the permittivity is untouched); such pixels get
  the nearest surviving contour's direction. The threshold is `0.35/sqrt(N)` of the
  period — it vanishes with real resolution.
- **Axis-aligned bases only.** Square-pixel cells reduce bit-identically to no metric;
  non-square cells work in physical coordinates (pinned by a 2:1-cell test against exact
  normals). An *oblique* lattice keeps working, but only under `'distance'`, the one
  construction ported to it (it carries the Jacobian pullback — see
  [`oblique-lattice.md`](oblique-lattice.md)); a skewed cell with any other method is
  refused up front, next to the other lattice refusals.

## Verification

The prototypes were measured by monkeypatching `normal_vector_field`; after integration
the same 36 cells (3 methods x 2 scenes x 6 grid points) were re-run through the public
API (`normal_method=`) against the same cached references: **worst relative deviation
2e-14** — the integration changed the plumbing and nothing else. A review pass then
hardened the constructions (corner-canonical indicators, spectral's periodic bulk copy
and its 3+-phase delegation, the phase count in the frozen record); re-running the same
cells afterwards, `contour` and `smooth` stayed bit-identical and `spectral`'s cells
moved by at most the reference floor (<=1.7% relative), which is the table above. Backends: numpy, jax and
torch agree on a contour cell to ~1e-14. The measurement harnesses were session
scratch — the numbers recorded here and the tests in
`tests/test_nvf_normal_methods.py` are the durable record, consistent with how
[`nvf-upsampling.md`](nvf-upsampling.md) records its own sweeps.

### References

- P. Gotz, T. Schuster, K. Frenner, S. Rafler, W. Osten, *Normal vector method for the
  RCWA with automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — masked blurred
  gradient at the boundary + Shepard interpolation inward; the first raster-native
  construction, and the "smooth before differentiating" template all of these follow.
- M. F. Schubert et al., *Fourier modal method for inverse design of
  metasurface-enhanced micro-LEDs*, Opt. Express **31**, 42945 (2023).
  [doi:10.1364/OE.503361](https://doi.org/10.1364/OE.503361) — fmmax's vector-field
  generation (gradient bandlimited to the expansion's orders + quadratic fit), the origin
  of `'spectral'`'s fixed-physical-band cutoff.
- V. Liu, S. Fan, *S4: A free electromagnetic solver for layered periodic structures*,
  Comput. Phys. Commun. **183**, 2233 (2012).
  [doi:10.1016/j.cpc.2012.04.026](https://doi.org/10.1016/j.cpc.2012.04.026) — the
  least-squares tangent-field design (after Fisher et al. 2007); shape-constraint-based,
  not raster-native, and the reason none of it is adopted here directly.
- A. M. Hammond, S. G. Johnson et al., *Unifying and accelerating level-set and
  density-based topology optimization by subpixel-smoothed projection*,
  [arXiv:2503.20189](https://arxiv.org/abs/2503.20189) — distance and normal jointly from
  a filtered density; the modern minimal recipe for the same move.
