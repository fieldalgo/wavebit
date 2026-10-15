# The raster normal-field constructions, and the `raster_normal_method` / `nvf_projector` options

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Shape NVF now also accepts `nvf_projector='circular'`, using a positive normal-dyad
Jones construction for rectangles and circles/ellipses, with a polygon-facet
correction for curved paintings. Its scope and contact derivatives are documented
in [parametric modeling](../../guide/rcwa/parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).
The raster constructions and measurements below retain their existing meaning.

Written: 2026-08-17. Measured 2026-08-16/17 on CPU, NumPy backend, float64.


> **Partly fixed 2026-08-29, and still open. Read this before trusting a covariance claim in
> this file.** `'contour'` is the one raster construction that is **not fully
> periodic**: rolling a raster -- the same structure written from a different corner, which
> every answer must be invariant under -- moved `sum(T)` by 3.9e-4 on a square cell and 9.7e-4
> on a 2:1 cell, where `'distance'`, `'spectral'` and `'smooth'` held at 4e-14.
>
> **The cause was `_two_phase_indicator` -- not the contour block.** It keyed the background on
> the *corner pixel* (`eps != eps.flat[0]`), which is invariant under relabeling the two phases
> but not under a translation: a roll can put the corner *inside* the feature, and the
> indicator then inverts wholesale. Ten rolls of the pinned disc came back as *exactly two*
> values -- 0.328909134249 for the six whose corner lands outside, 0.329294462911 for the four
> whose corner lands inside, each class tight to 5e-13, nothing in between. That partition is
> the tell: a sample-phase offset gives a spread. The indicator marks the smaller phase now,
> which collapses all ten to 4.157e-13.
>
> Three things this file said about it were wrong, and are worth keeping as such:
>
> * **the cause.** The caveat blamed `_resample`'s phase following contourpy's loop entry, and
>   scoped the fix as a nearest-segment rewrite deleting `_resample`. That is the amplifier --
>   an inverted indicator is the same curve, 236 points either way, reached in a different
>   traversal. Canonicalizing the traversal directly is worse, measured: shoelace direction
>   alone leaves 4.5e-4 and still two classes; adding a lexicographic entry vertex gives 1.5e-3
>   across ten distinct values, because the entry is itself unstable.
> * **the trigger.** "Where a feature is large compared to the cell" is not it. A disc of
>   radius 12 in a 64 cell broke on exactly the one roll of ten that moved its centre within 12
>   pixels of the corner, at 1.5e-4. Any feature does, given the placement -- and a shape
>   written at the origin is an ordinary thing to write.
> * **the pin.** It was an imperative `pytest.xfail()` call, which raises before the assertion,
>   so it reported XFAIL forever and could not have said the defect was gone. The strict marker
>   that replaced it earned itself immediately: it reported `XPASS(strict)` the moment the
>   indicator was fixed.
>
> **What is left, measured the same day.** Fixing the indicator fully fixes a *single* loop on
> an orthogonal cell (4.2e-13 over ten rolls, from 3.9e-4). It does not fix everything:
>
> | scene | after | before |
> |---|---|---|
> | disc, square cell | 4.2e-13 | 3.9e-4 |
> | disc, `[1,2]` and `[2,1]` cells | 1.2e-13 / 8.3e-14 | -- |
> | **annulus, square cell** | **1.2e-5** | 7.5e-5 |
> | **disc, hex cell** | **2.3e-4** | 3.0e-4 |
> | annulus, hex cell | 2.2e-3 | 2.2e-3 |
>
> `'spectral'`, `'distance'` and `'smooth'` are at 1e-13 on every one of those.
>
> The residual is a different animal, and it is not a tie-break rule that can fix it. Comparing
> the *fields* pixel by pixel: the annulus disagrees on **one pixel**, at the exact centre of
> its hole; the hex disc on 1 to 18 pixels, all at radius 35-42 where the curve is at 30 --
> outside the shape, in the deep gaps between its periodic images. Both sets are the **medial
> axis**, where the nearest boundary point is equidistant in every direction and the normal is
> genuinely undefined. Whichever segment wins that tie is an accident of sampling, and a roll
> changes the accident. This is the same wall a nearest-segment rewrite hits: a prototype of
> one differs from the shipped construction on exactly 8 pixels, all within 4.2 px of a disc's
> centre, and is *less* periodic overall (2.8e-3).
>
> So this is the raster twin of the exact-tangency singularity in the parametric path: a normal
> field asked for a direction at a place geometry does not define one.
>
> The accuracy tables below were taken on interior features, where none of this fired, and
> stand unchanged.


Targets:
- `src/wavebit/rcwa/_engine/normal_methods.py` — the three constructions added beside the original
- `src/wavebit/rcwa/_engine/factorization.py::normal_vector_field` — the dispatcher
- the `raster_normal_method` solver keyword — `'distance'` (default) | `'contour'` | `'spectral'`
  | `'smooth'`
- `src/wavebit/rcwa/_engine/factorization.py::jones_projector_fields` — the complex-basis projector
- the `nvf_projector` solver keyword — `'linear'` (default) | `'circular'`; a second, independent
  axis over the same field

Companion: [`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md) diagnosed the defect this option
addresses and records the baseline measurements this file compares against.

---

## TL;DR

The shipped raster normal field quantizes its directions to essentially {0, 45} degrees
(~14 deg mean error in the boundary band, flat in N — refining the raster does not help).
Three replacement constructions were prototyped and measured at the solver level against
staircase-free analytic references. **`'contour'` is the one to use**: of the 12 measured
cells, 11 are resolved by the reference floor and it won all 11, losing none, by factors up
to 14x — and that holds on an oblique cell too, where all four constructions have run since
2026-08-21 ([the metric section](#the-cell-metric-and-oblique-cells)). `'smooth'` had the *best* angular numbers and still lost every resolved cell on the
corner-bearing scene (5 of 5, the sixth unresolved) — the measured demonstration that the
best angular field is not the best solver field. The default stays `'distance'`: every
existing result is bit-identical unless the keyword is set. (True of the *keyword*, and no
longer of the construction: `339df21` made `'distance'` itself periodic, which moved its answers
— see [the periodicity section](#periodicity-the-cell-is-a-torus).)

```python
wavebit.rcwa(structure=raster, factorization='nvf', raster_normal_method='contour', ...)
```

(This page's snippets use the interface it was written against, which since 2026-10-13 is
`wavebit.compat.legacy`. The public module takes the same two keywords on
`rcwa.Numerics(factorization='nvf', raster_normal_method='contour', nvf_projector=...)`.)

The keyword is read exactly where a raster normal field is built — a raster `structure` under
`factorization='nvf'` (and the tensor-raster builders, called directly). Explicitly set
anywhere else it is refused rather than ignored, for the same reason `enhanced_dfs` was
removed outright: a knob nothing reads is a lie waiting to be believed.

There is a **second, independent** keyword over the same field, `nvf_projector`, and it defaults
to what has always been done. It chooses how the projector is formed *from* the field rather
than how the field is built, so it composes with all four constructions above; the short
version is that `'circular'` removes the medial axis's arbitrariness and costs accuracy,
mirror symmetry and a handedness to do it, and that you should leave it alone unless you have
measured a medial-axis problem on your own scene. Its own section is
[below](#nvf_projector--the-circular-polarization-limit-2026-09-01).

## The defect, in one paragraph

`'distance'` reads `np.gradient` of the Euclidean distance to the interface set. Central
differences of a distance-to-a-digital-set over a two-cell span land on small integer
pairs — `{1,0}`, `{2,0}`, `{1,1}` — so in the boundary band the emitted direction comes
from the short list `arctan(p/q)`, in practice {0, 45} deg. The mean angular error is
~14.4 deg in the 0-1 px band with a p90 of ~30 deg, and every band is flat in N from 64
to 512: a finer staircase is the same staircase. Full measurements, including why
upsampling makes it worse, in [`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md).

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
of the band the factorization itself reads (orders to `2*truncation_order`, `truncation_order >= 4` in practice) —
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
| `contour` | 1.37 | 1.32 | 0.51 | 0.37 | 0.82 |
| `spectral` | 3.08 | 1.43 | 0.38 | 0.16 | 0.36 |
| `smooth` | 0.52 | 0.37 | 0.04 | 0.03 | 0.05 |

Degrees, modulo 180. The `contour` column was re-taken on 2026-09-01, when its resampling
changed; it moved within its own third digit and with no trend (1.33 / 1.28 / 0.50 / 0.37
before), and so did the skew table further down and both A/B tables. Only `distance` is flat
in N; the 30-40 deg substitution outliers
(a pixel emitting 45 where its neighbours emit 0) are gone from all three replacements.
Note `smooth` winning this table and losing the next one — the tables must be read
together.

### Solver-level A/B, circle (vs the analytic reference at truncation_order=14)

Same setup as the `li` comparison in [`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md): circle
r=0.30, index 2.4, thickness 0.30, wavelength 0.55, theta 12 / phi 30 / psi 40 deg.
Error = max per-order |R - R_ref|, |T - T_ref| over shared orders. All cells resolved
(gaps above the reference's ~1.7e-3 own error).

| N | truncation_order | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|---|
| 64 | 6 | 2.15e-02 | 1.06e-02 | 1.13e-02 | 9.72e-03 |
| 64 | 10 | 8.75e-03 | 1.15e-03 | 1.03e-03 | 1.04e-03 |
| 128 | 6 | 1.95e-02 | 1.31e-02 | 1.31e-02 | 1.22e-02 |
| 128 | 10 | 8.00e-03 | 3.31e-03 | 3.14e-03 | 3.03e-03 |
| 256 | 6 | 1.43e-02 | 9.64e-03 | 9.79e-03 | 9.16e-03 |
| 256 | 10 | 3.51e-03 | 2.42e-04 | 2.42e-04 | 4.35e-04 |

Every replacement beats `distance` in every circle cell, up to 14x at 256/10 — and
`contour` at N=64 already beats `distance` at N=256.

### Solver-level A/B, rotated rectangle (vs the analytic reference at truncation_order=18)

The corner-bearing scene: rectangle 0.5 x 0.3 rotated 30 deg, same stack and incidence.
The truncation_order=14 reference could not resolve this scene (its own error floor ~1.7e-3 swamped
the differences), so the reference was rebuilt at truncation_order=18; the floor |truncation_order 16 - truncation_order 18| is
2.09e-4, and cells are called only above it.

| N | truncation_order | `distance` | `contour` | `spectral` | `smooth` |
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

The parametric path itself (exact geometry, exact normal field) at the working `truncation_order`,
against the finest references:

| scene | truncation_order=6 | truncation_order=10 |
|---|---|---|
| circle | 1.06e-02 | 1.13e-03 |
| rect30 | 1.16e-02 | 1.57e-03 |

Error is (truncation) + (normal-field error), and no normal field can go below the first
term. Read the tables above against these floors: every truncation_order=6 cell sits at ~1e-2 for
every method because the floor is ~1.1e-2 there — `contour`'s 1.06e-02 at circle 64/6
*is* the floor, i.e. 100% of the removable error removed. The dramatic cells are the ones
where the normal field dominated the budget (circle truncation_order=10), and the flat-looking rect30
cells are flat because `distance` was already near the floor on straight edges. Raising
accuracy past these numbers is a matter of `truncation_order` and raster resolution, not of the normal
field — with the asymmetry that `distance` eats into higher-`truncation_order` gains (its staircase
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
differently — moved `max|dR|` by ~1e-2, at every `truncation_order`, non-monotonically. Li's rules are
exact to 1e-15 on the same rolls.

It does not converge away, and that is what makes it a defect rather than a residual: the
lift to the `4*truncation_order+1` floor is a block repeat, the same staircase spelled with more pixels
([`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md)), so raising `truncation_order` never refines the field. It
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
`truncation_order`. Cost is ~3 ms on an 11.6 s solve at `truncation_order=[16,12]` (the EDT is 11x dearer in
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

Pinned by [`tests/rcwa/test_nvf_lattice_translation.py`](https://github.com/fieldalgo/wavebit/blob/develop/tests/rcwa/test_nvf_lattice_translation.py),
which parametrizes three placements deliberately (a partial fix passes some). `'contour'` was
carried there as a strict xfail four times -- until the aperture fix above promoted it; again
from 2026-08-28 to 2026-08-29 for the indicator defect in the caveat at the top of this file,
which no amount of contour work was going to reach; and twice more until 2026-09-01, for the
medial axis and for percolating level sets (see *Percolation, and the two rules that fell out
of it* below). **All four constructions are covariant now**, at 1e-14 over 1023 translations
of every scene in the sweep. Rolling
*somewhere* does not test any of it: eight
of the ten rolls in `test_contour_is_periodic_under_a_roll_that_lands_on_the_corner` agreed
with each other even with the defect in place, and the small-disc case turns on exactly one
of the ten. The sibling defect in the *parametric* path was
older and is also fixed (`c128fd8`) — its normal field was distance-blind, not merely aperiodic,
and `test_shape_position_gradients.test_a_multishape_layer_is_periodic_across_the_cell_edge` now
states the property where a test of the opposite name used to pin the defect.

### Percolation, and the two rules that fell out of it

The last thing keeping `'contour'` aperiodic, fixed 2026-09-01, was the **window the level set
is extracted through**. contourpy has no periodic domain: it is handed a finite rectangle and
cuts whatever leaves it. The old window was the raster padded 3x3 with `mode='wrap'`, which is
enough for a loop that closes inside the cell — some copy of it lands whole in the window, and
a fold plus a dedup keep exactly one representative whatever the placement. It does nothing at
all for a level set that **percolates** — one that leaves the cell and comes back, closing only
on the torus — because that is cut wherever the window ends however wide the window is. The cut
fragments got neither the fold nor the dedup, and `_contour_polylines` ended with
`if out: return out`, so a single surviving closed loop discarded every one of them.

Two consequences, and the second is worse:

* the fragment set moved with the placement (5 / 6 / 9 / 12 fragments across rolls of one
  scene), so the field did too: 2.8e-3 on a post-and-arm scene, 8.8e-3 on a Cartesian circle
  big enough to overlap its own images at 20 degrees, 5.3e-5 on a stripe at 45 degrees.
* a layer holding a stripe **and** a disc lost the stripe's boundary from the field entirely —
  9 closed curves and 6 open ones found, 1 kept. Pixels on the stripe's own edge answered with
  the disc's normal, up to **45 degrees** wrong at every placement, with energy conserved. That
  one is not a stability defect; it is a wrong answer, and the periodicity instrument could not
  see it because it was wrong consistently.

**The fix is the window, not the fragments.** Pad instead by a single duplicated row and column:
every torus cell of the raster then appears exactly once, and the cut has an exact partner —
the level set leaving through column `n_u` re-enters at column `0` at a coordinate interpolated
from the same two samples, bit for bit. `_stitch_across_the_seam` matches each fragment's tail
to the head at the same point of the torus and walks the cycle, accumulating the whole-cell
jumps it crosses; that sum is the curve's **winding**. Everything comes back closed: `(0, 0)`
for a component that merely straddles the seam (its fragments reassemble), non-zero for one
that percolates, and `contour_normal_field` then joins each percolating curve's tiled copies
end to end so the tiling has no artificial ends in it either. There is nothing left to fold or
dedup, because there is only ever one copy.

Matching is by position on the torus and **not** by "the opposite edge", which is what makes it
work on a skewed cell: the raster's periodicity is an index shift of a whole cell whatever the
lattice does with those indices, so the metric never enters the stitch. (An earlier probe that
matched opposite edges paired 2 of 4 ends on a square-cell stripe and 0 of 8 at 20 degrees. The
rule was fine; the window was wrong.)

**This regime is ordinary rather than exotic on a skewed cell**, which is what makes it worth
the machinery: at 20 degrees with `|a1| = |a2| = 640` the lattice rows sit 219 apart, so any
circle past radius 109 percolates — and the parametric path refuses such a shape outright, so the
raster is the only route to it.

Two further rules were needed, and neither is about percolation. Both are stated at their
constants in `src/wavebit/rcwa/_engine/normal_methods.py`, with the mutation numbers.

* **`_resample` subdivides each segment; it does not walk arc length.** A reassembled curve has
  no intrinsic starting vertex. Arc-length resampling from one end was only ever covariant
  because contourpy's entry is the first crossing its raster scan meets — a geometric point of
  the loop, which moves with the loop — and a curve stitched from fragments starts wherever the
  seam cut it instead. Subdividing each segment into `ceil(len/h)` equal parts has no phase to
  get wrong, costs about one extra sample per vertex (753 -> 878 on a radius-30 disc), and has
  the side benefit that the sampled polyline is exactly the extracted one, with no chord cutting
  a corner. Putting the arc-length spelling back into the finished construction breaks 8 of the
  14 scenes swept: 7.0e-3 on the overlapping circle, 1.2e-3 on a disc merely straddling the
  seam.
* **The nearest-sample query is answered in a slightly sheared metric.** A sample set with no
  phase in it inherits the raster's symmetries, so on the medial axis two samples are equidistant
  *exactly* rather than nearly, and the last bit decides which one wins. A fixed, translation
  invariant, infinitesimal perturbation makes that choice single valued — Simulation of
  Simplicity, Edelsbrunner & Mücke, ACM TOG **9**, 66 (1990). Without it, seven of fourteen
  scenes move: a cross 1.8e-3 on 63 of 63 rolls, a small disc 4.2e-4 on 57, a saddle 2.9e-4 on
  8, an annulus 9.9e-5 on 48. Every value from 1e-13 to 1e-3 gives an identical clean sweep,
  which is what says it is a round-off device and not a tuned knob.

The second one settles the **medial-axis** residual this file carried as a strict xfail from
2026-08-29, on the very rolls that pin it: `sum(T)` on an annulus 1.19e-5 -> 4.67e-14 and on a
pair of discs 3.16e-4 -> 1.61e-12, and 0 of 1023 rolls on both under the translation identity.
Be exact about what that means. The direction on the medial axis is still undefined
and the answer there is still arbitrary — what is no longer arbitrary is *which* arbitrary
answer, which is all covariance asks for. The accuracy half is the circular-polarization
limit S4 and fmmax use — Antos' complex polarization basis, published as Liu & Fan Eq. (52).
It **landed on 2026-09-01** as `nvf_projector='circular'`, opt-in and off by default, and
[its own section](#nvf_projector--the-circular-polarization-limit-2026-09-01) records what it is
worth: it removes this defect completely and costs accuracy, mirror symmetry and a chirality
to get there. Both papers are cited below and both are owed.

Accuracy stayed put and cost improved. The angular error against an exact circle moves within
its own third digit (1.33 -> 1.37 at N=64, 0.37 -> 0.37 at N=512) with no trend, and the
skewed-disc table likewise. Both A/B tables were re-taken rather than assumed, new against old
through one harness: the circle table's `contour` column is updated above (and its `distance`
column, which this change cannot touch, reproduces the printed numbers exactly, which is what
says the harness is the one those numbers came from); the rotated-rectangle table is left as
printed, because that harness does not reproduce its `distance` column either, and the
new-against-old deltas there are 3.03e-03 -> 3.11e-03 and 1.40e-03 -> 1.42e-03 at the two
cells `contour` wins. On a
non-percolating scene the construction reads ~20% more samples (753 -> 900 on a radius-30 disc
in a 64 cell) for no measurable wall-time change; on a percolating one it is **several times
faster**, because the old window handed the kd-tree three periods of every fragment:

| scene (20-degree cell) | before | after |
|---|---|---|
| circle overlapping its images, N=128 | 601 ms | 84 ms |
| the same, N=512 | 7.00 s | 1.96 s |
| percolating stripe, N=128 | 275 ms | 38 ms |
| the same, N=512 | 2.47 s | 0.42 s |


## `nvf_projector` — the circular-polarization limit (2026-09-01)

Everything above is about how the normal field is **built**. This is about what is done
**with** it, and it is a second, independent keyword for that reason:

```python
wavebit.rcwa(structure=raster, factorization='nvf',
             raster_normal_method='contour',      # how the field is built  (4 values)
             nvf_projector='circular', ...)    # how P is formed from it (2 values)
```

Folding it into `raster_normal_method` would have made one list of eight where there are two lists of
four and two — every construction can be read either way, and the two choices answer different
questions. Same precedent as `fourier_coefficients` x `factorization`. It is read exactly where
`raster_normal_method` is read, and refused rather than ignored everywhere else. (Since
2026-09-09 the shape path under `factorization='nvf'` reads it too; see the note at the top.)

### What it is

The projector has always been `P = n n^T`, built from the field's *direction* alone. On the
medial axis the direction is undefined — not hard to compute, undefined — so both answers
there are arbitrary. `nvf_projector='circular'` builds `P = J J^H` from a Jones vector that carries
the field's magnitude too:

    J   = n_hat cos(phi) + i n_perp sin(phi),   n_perp = (-n_y, n_x)
    phi = (pi/8) (1 + cos(pi |t|)),             |t| scaled to unit maximum

| | |t| = 1 | |t| -> 0 |
|---|---|---|
| `phi` | 0 | `pi/4` |
| `P` | `n n^T`, **to the bit** | `Pxx = Pyy = 1/2`, `Pxy = -i/2`, `Pyx = +i/2` |

The right column does not mention the direction. That is the whole design: the answer on the
medial axis is reached as a **limit**, so there is no threshold and no "is this pixel on the
medial axis?" test anywhere. `trace(P) = 1` at every magnitude, which is what keeps
`C_xx + C_yy = I` and leaves the assembly unchanged.

The unit-maximum scaling is the formula's *domain*, not a normalisation anyone chose: `phi` is
monotone in `|t|` only on `[0, 1]` and turns back up outside it.

`|t|` is `|grad d|` of the periodic distance map — 1 almost everywhere, dropping to 0 exactly
on the medial axis. It is the same array for all four constructions, because it is a property
of the interface set and not of a construction; the constructions return *unit* fields and
have no magnitude of their own to read.

### What it fixes

The medial-axis covariance defect, completely — measured with `_CONTOUR_TIE_SHEAR` switched
off, which is the only configuration in which the instrument can still see it (see below).
`raster_normal_method='contour'`, N=64, `truncation_order=3`, **all 4095 rolls** of the translation
identity, worst defect and how many rolls exceed 1e-12:

| scene | `linear` | `circular` |
|---|---|---|
| annulus | 9.87e-05, 3172 of 4095 | **1.53e-15, 0** |
| disc | 6.52e-05, 2215 of 4095 | **1.93e-15, 0** |
| two discs (saddle) | 3.01e-04, 3858 of 4095 | 2.83e-04, 3858 of 4095 |

**The saddle is not fixed and never will be by this.** Its two nearest boundary points sit on
*different* discs, so the tie is a discrete jump between two separated directions rather than a
direction fading out — `|t|` stays O(1) through it and the limit is never reached. The medial
axis and the saddle are two different geometric facts that happen to break the same instrument.

### What it does not buy on today's `develop`

With the shear **on** — the shipped configuration — both projectors sit on the floor:
0 of 4095 rolls on all three scenes, worst 2.6e-15. The shear (2026-09-01) already settles
the tie the same way at every placement, which is all covariance asks for.

So the two are answers to different halves of one defect: **the shear makes the arbitrary
answer stable; this makes it not be read.** Only an instrument sensitive to the arbitrariness
rather than to the instability can tell them apart — and the one this repository already has
says the news is bad.

### What it costs

**1. Accuracy, everywhere.** The circle A/B above, re-run on this axis (the `linear` column
reproduces the printed table exactly, which is what says it is the same harness):

| N | truncation_order | `linear` | `circular` |
|---|---|---|---|
| 64 | 6 | 1.06e-02 | 1.76e-02 |
| 64 | 10 | 1.15e-03 | 8.39e-03 |
| 128 | 6 | 1.31e-02 | 1.62e-02 |
| 128 | 10 | 3.31e-03 | 6.54e-03 |
| 256 | 6 | 9.64e-03 | 1.11e-02 |
| 256 | 10 | 2.42e-04 | 1.60e-03 |

(`raster_normal_method='contour'`; `distance` loses by the same 1.1x-3x.) **`linear` wins every one
of the twelve cells measured**, by 1.1x to 7x. Both converge on the analytic reference —
`Delta = eps^ - A -> 0` as the truncation rises, so the projector's contribution vanishes and
they must agree in the limit, and they do — `circular` just converges 3-5x further out.

**2. Most of it where the answer is known exactly.** A y-invariant raster with more than one
row goes through the 2D builder and gets block-repeated to the field's grid floor, so the
stripe centres become real medial *columns*. Li's rules are **exact** for that structure.
Total `T` against `factorization='li'`, 8 rows of a 64-wide 50% grating:

| truncation_order | `linear` | `circular` |
|---|---|---|
| 4 | 9.2e-05 | 3.5e-01 |
| 6 | 2.1e-06 | 2.0e-02 |
| 8 | 3.7e-05 | 1.6e-02 |
| 12 | 8.4e-05 | 8.9e-03 |

The mechanism is worth stating because it explains the whole cost column. `'distance'` reports
`(0, 0)` on a medial column, so the **linear projector is zero there and the correction simply
switches off** — the scheme degrades to the direct rule. The circular projector is a unit-trace
projector at *every* pixel including that one, so it applies a half-and-half correction with a
definite handedness instead. "No opinion" and "an arbitrary opinion" are not equally safe.

**3. Mirror symmetry, outright.** `n_perp = (-n_y, n_x)` is a fixed rotation sense and `-i/2`
is left- rather than right-circular, so the construction is **chiral**. Solve a structure and
its x-mirror, `max|dR|`, off-centre scenes at N=48, `truncation_order=5`:

| scene | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|
| circle, `linear` | 2.1e-15 | 7.7e-05 | 1.6e-04 | 2.4e-15 |
| circle, `circular` | 3.96e-04 | 3.79e-04 | 5.22e-04 | 3.50e-04 |
| annulus, `linear` | 3.0e-15 | 2.4e-04 | 7.2e-04 | 2.9e-15 |
| annulus, `circular` | 7.40e-03 | 8.25e-03 | 7.46e-03 | 8.04e-03 |

`'distance'` and `'smooth'` mirror **exactly** under the linear projector, and the circular one
costs them that by up to twelve orders. It is the method and not the implementation: mirroring
the structure and *conjugating* the projector — the other circular state, an equally valid
rank-one projector — agree to **3.4e-15**, so the asymmetry is precisely the basis's
handedness and nothing else. The two handednesses differ from each other by 7.4e-3, so the
choice is real rather than a gauge; the sign implemented is the published one.

A workflow that needs exact mirror symmetry of the output must stay on `'linear'`.

### The recommendation

**Leave it alone.** It is landed because it is the published answer to a real question and
because the question was open in this file for two weeks; it is off by default because on the
shipped configuration it removes nothing measurable and costs accuracy in every cell measured,
mirror symmetry outright, and the most on the one scene whose exact answer is known. Reach for
it if you have a scene whose medial axis you have *measured* to be moving your answer and the
shear has not settled it — and measure the cost on your own scene before keeping it.

The default is untouched, and provably: 25 MB of convolution matrices over 4 scenes x 2
lattices x 4 constructions x 2 builders come back **byte-for-byte identical** to the
pre-change tree.

## Semantics worth knowing

- **The default is untouched.** `raster_normal_method='distance'` and an unset keyword are
  bit-identical, pinned by test — the two spellings select the same construction. That is a
  statement about the keyword only. The construction it selects was itself changed on
  2026-08-17 (`339df21`, periodicity) and `tests/fixtures/legacy_rcwa/golden_nvf.npz` was re-frozen with it, so
  "no golden moved" held when this option landed and does not hold across that commit.
- **Refused where unread.** Explicitly set with `factorization='li'` or on either shape
  path, the solve raises `NotImplementedError` naming the keyword. The shape paths build
  their normal fields from the geometry itself (`parametric.normal_vector_field_analytical`);
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
- **The floor lift keeps the base scale.** A layer block-repeated to the `4*truncation_order+1` floor
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
  been built and measured (2026-08-17, not landed): read the
  graded band's own mid-range level set of `Re(eps)` subpixel, take the exact
  distance-to-curve directions off it, delegate binary layers to `'contour'`, and extend
  the frozen record with a digest of the normalized values (a graded band can move
  *inside* its pixels without moving the interface mask, which the geometry-only record
  cannot see — that record fix is the one part a future attempt should keep). Its
  angular fidelity is dramatic — 1.2 deg mean on a graded band against 52 deg for
  `'distance'`, whose mask degenerates over a thick band — **and the advantage does not
  reach the solve.** On SSP-projected scenes at `beta = inf` against same-`truncation_order` exact
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
  [`backend-choice.md`](../../validation/rcwa/backend-choice.md) already recommends for fits.
- **Multi-phase rasters** (3+ distinct values) delegate to `'distance'` under `'contour'`,
  `'spectral'` and `'smooth'` alike — the two-phase indicator cannot say which of three
  materials is inside, and the distance mask sees every interface.
- **Thin features** under `'contour'`: a strip narrower than ~1.4 sigma px loses its 0.5
  crossing and vanishes from the *field* (the permittivity is untouched); such pixels get
  the nearest surviving contour's direction. The threshold is `0.35/sqrt(N)` of the
  period — it vanishes with real resolution.
- **Any basis, oblique included** (since 2026-08-21 — before that all three refused a
  skewed cell and `'distance'` was the only field an oblique lattice could have). Square-pixel
  cells reduce bit-identically to no metric; everything else works in physical coordinates,
  non-square and skewed alike. See [the metric section](#the-cell-metric-and-oblique-cells).

## The cell metric, and oblique cells

Added 2026-08-21. Until then these three refused a skewed basis, and
[`oblique-lattice.md`](oblique-lattice.md) recorded the reason: only `'distance'` carried the
`J^-T` pullback that turns an index-space gradient into a Cartesian normal. That was a true
statement about the code and a misleading one about the constructions, and it was expensive —
`'distance'` is the least accurate of the four and a skew makes it worse, so the refusal left
an oblique cell with a **18.5 deg** mean angular error where `'contour'` gives **0.42**.

### What each construction actually needed

| | takes an index-space difference? | what a skewed cell needed |
|---|---|---|
| `'contour'` | **no** — it measures between Cartesian points | the level set's coordinates and the pixel centres mapped through the metric; the direction to the nearest point of a curve is then Cartesian by construction |
| `'spectral'` | **no** — it multiplies by a wavevector | both frequency-domain quantities (the low-pass and the gradient) stated on the mode's *Cartesian* `G = 2*pi * M^-T @ (fi, fj)` |
| `'smooth'` | **yes** | both of the below |

Two of the three needed *coordinates*, not a pullback, and neither has one. That is also why
they are the better field on a skewed cell than `'distance'`, which does take a difference on
the grid and pays the quantization for it.

`'smooth'` is the one that leans on scipy's per-axis routines, and a skewed metric is not
diagonal, so both had to be replaced: `distance_transform_edt`'s `sampling` by a kd-tree over
the tiled phase sets, and the separable `gaussian_filter` by a frequency-domain kernel. An
axis-aligned cell keeps both scipy calls, bit-identically; `normal_methods._separable` is the
switch, and it is a tolerance rather than an exact zero so that a square cell spelled
`a2 = P*(cos(pi/2), sin(pi/2))` does not fall off the fast path on a 1e-14 entry.

### Angular error against the exact normal, by shear

Disc r=60 in a 400-unit cell, mean over the interface mask, degrees. `gamma` is the angle
between `a1` and `a2`; 90 is the square cell the tables further up measure.

| gamma | `distance` | `contour` | `spectral` | `smooth` |
|---|---|---|---|---|
| 90 (N=128) | 15.61 | 0.97 | 0.75 | 0.28 |
| 60 (N=128) | 18.18 | 1.00 | 0.89 | 0.16 |
| 30 (N=128) | 23.69 | 1.31 | 0.96 | 0.48 |
| 90 (N=256) | 15.01 | 0.93 | 0.58 | 0.17 |
| 60 (N=256) | 17.50 | 0.66 | 0.37 | 0.12 |
| 30 (N=256) | 24.78 | 0.55 | 0.21 | 0.13 |

The shape of it: `distance` *grows* with shear (15.6 -> 23.7) while the three replacements
stay flat or improve, and they converge in N where `distance` does not — the same two
statements as the axis-aligned table, now under a skew as well.

**Both `'smooth'` repairs are load-bearing and neither alone is enough.** The 2x2, on an r=90
disc at N=256, as `gamma=60 / gamma=30` mean degrees:

| | separable Gaussian | isotropic Gaussian |
|---|---|---|
| scipy's diagonal distance | 2.11 / 2.65 | 1.17 / 2.49 |
| exact kd-tree distance | 1.38 / 2.25 | **0.07 / 1.82** |

And the half-fix is worse than that suggests, because its error grows as the feature shrinks
relative to the period/16 kernel: at r=40 the two-scipy-calls pair scores **7.8 and 11.8 deg**
where the repaired pair scores 0.085 and 0.077 — that is a field no better than `'distance'`,
reached through four times the work.

The r=90 / gamma=30 cell above (1.82) is the one number here that is not about the metric: at
gamma=30 the cell is only 200 units thick across `a1`, so an r=90 disc passes 20 units from its
own periodic copy — comparable to the 17.7-unit kernel, which then reads two shapes as one. A
property of that scene; r=60 and r=40 both sit near 0.1 there.

### Solver-level A/B on a hex cell

Rod r=90 in a hex cell of pitch 400, eps 5.76, thickness 300, wavelength 900, theta 12 / phi
30 / psi 40 deg. The reference is the same rod through the *parametric shape* path (exact
geometry, exact normal field) at `truncation_order=14`, whose own `|14 - 12|` floor is 1.8e-6 — well
under every gap below, so all eight cells are resolved. Error is the worst per-order
`|R - R_ref|`, `|T - T_ref|` over shared orders.

| N | truncation_order | `distance` | `contour` | `spectral` | `smooth` | analytic at that truncation_order |
|---|---|---|---|---|---|---|
| 128 | 6 | 8.33e-05 | 3.27e-05 | 3.21e-05 | 3.38e-05 | 2.82e-05 |
| 128 | 10 | 5.07e-05 | 1.21e-05 | 1.05e-05 | 1.86e-05 | 7.37e-06 |
| 256 | 6 | 6.61e-05 | 3.23e-05 | 3.33e-05 | 3.41e-05 | 2.82e-05 |
| 256 | 10 | 3.83e-05 | 1.11e-05 | 9.92e-06 | 1.76e-05 | 7.37e-06 |

All three beat `distance` in all four cells, by 2.0x to 4.3x, and land near the truncation
floor: at 256/10 `contour` removes 89% of the error that floor leaves room to remove. Read it
against the axis-aligned tables and nothing is new except that the cell is skewed — which is
the result.

### Where it does *not* show: Li's checkerboard

The published oblique benchmark ([`oblique-lattice.md`](oblique-lattice.md) has the table) is
the obvious candidate for an outside oracle here, and it cannot see this. Rasterized on its own
primitive cell and run through the raster NVF path at `truncation_order=12`, the worst deviation from
the paper's FEM column is:

| N | `distance` | `contour` | `spectral` | `smooth` | parametric shapes |
|---|---|---|---|---|---|
| 128 | 6.52e-03 | 6.75e-03 | 6.67e-03 | 6.54e-03 | 2.78e-04 |
| 256 | 3.11e-03 | 3.26e-03 | 3.20e-03 | 3.11e-03 | 2.78e-04 |

All four within 5% of each other, halving with N, and an order above the parametric path at the
same `truncation_order`. That is the *permittivity* staircase, not the normal field: a checkerboard is
straight-edged, and the axis-aligned tables already say `distance` is near the floor on
straight edges — the rotated-rectangle scene looked flat for the same reason. A benchmark has
to be curved to have a normal-field budget to spend, which is why the hex rod above is the
measurement and this is the control.

### What changed for axis-aligned cells too

**The skeleton needs an exact zero, and a transform does not produce one.** The separable
Gaussian returns exact zeros where the two nearest interfaces are equidistant, because a
symmetric kernel over a locally symmetric field cancels term by term in the same
floating-point order. The frequency-domain branch has no such pairwise structure and leaves
~1e-16 there instead, which the unit normalisation inflates into a full vector pointing
wherever the round-off fell — and a lattice translation settles it differently, so the field
stops being covariant at a handful of pixels (4 of 256 on a rod in a hex cell, 90 deg apart).
Found by `test_nvf_lattice_translation`'s skewed parametrization, and by nothing else: no
accuracy number moves by 4 pixels. Snapped at 1e-12 of the largest gradient in the cell, since
the factorization reads an exact zero as "no normal here, collapse to the direct rule", which
is the right answer on a skeleton.

That, and `'smooth'`'s smoothing scale — [its own section](#smooths-sigma-is-a-length-not-a-pixel-count-per-axis),
because it is an axis-aligned defect the port merely happened to find. Everything else is
bit-identical on an axis-aligned cell, square *and* non-square, for all three constructions,
checked scene by scene across the change.

## `'smooth'`'s sigma is a length, not a pixel count per axis

Changed 2026-08-21. It was `base_N/16` *pixels along each axis*, which on a
square cell is the same number and on any other cell is an ellipse stretched toward whichever
axis spans more of the cell. Nothing chose that: the sweep that froze the 1/16 saw square cells
only, so it fixed the coefficient and not which of the two readings it meant. It is also not a
property of the lattice — `a2' = a2 + a1` generates the identical point lattice and would have
moved the kernel — which is the objection that settles it. `sqrt(cell area)/16` is isotropic
and basis-invariant at once, and reduces to `N/16` px bit-identically on a square cell. On the
2:1 cell of `test_a_non_square_cell_measures_in_physical_space` it takes the mean angular error
from **2.22 to 0.39 deg**. `'contour'`'s sigma was already stated this way.

## Verification

The prototypes were measured by monkeypatching `normal_vector_field`; after integration
the same 36 cells (3 methods x 2 scenes x 6 grid points) were re-run through the public
API (`raster_normal_method=`) against the same cached references: **worst relative deviation
2e-14** — the integration changed the plumbing and nothing else. A review pass then
hardened the constructions (corner-canonical indicators, spectral's periodic bulk copy
and its 3+-phase delegation, the phase count in the frozen record); re-running the same
cells afterwards, `contour` and `smooth` stayed bit-identical and `spectral`'s cells
moved by at most the reference floor (<=1.7% relative), which is the table above. Backends: numpy, jax and
torch agree on a contour cell to ~1e-14. The measurement harnesses were session
scratch — the numbers recorded here and the tests in
`tests/rcwa/test_nvf_normal_methods.py` are the durable record, consistent with how
[`nvf-upsampling.md`](../../validation/rcwa/nvf-upsampling.md) records its own sweeps.

`nvf_projector='circular'` (2026-09-01) was verified on five instruments, listed because each
answers something the others cannot. **The translation identity** over all 4095 rolls, shear
on and off — the covariance claim, and the reason the shear had to be switched off to make
it. **A byte comparison against `git archive HEAD`** over 25 MB of convolution matrices (4
scenes x 2 lattices x 4 constructions x 2 builders) — the default path is untouched, which a
tolerance could not have said. **The circle A/B table's own harness**, whose `linear` column
reproduces the printed numbers exactly, which is what says it is that harness. **Li's rules on
a y-invariant raster**, the one scene here whose exact answer is known. And **the mirror
gap**, which is the only instrument sensitive to the arbitrariness rather than to the
instability — it is the one that found the chirality, and it was not being looked for.
Backends: numpy, jax and torch agree on a circular cell to ~1e-14, and the record crosses a
`jax.jit` boundary carrying a finite gradient. Tests in `tests/rcwa/test_nvf_projector.py`.

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
- R. Antos, *Fourier factorization with complex polarization bases in modeling optics of
  discontinuous bi-periodic structures*, Opt. Express **17**, 7269 (2009).
  [doi:10.1364/OE.17.007269](https://doi.org/10.1364/OE.17.007269) — the complex
  polarization basis itself: the projector is built from a Jones vector instead of from
  `n n^T`, so it need not carry a direction at all. This is the origin of the
  circular-polarization limit, and the citation to credit alongside its generalization.
  **Implemented** as `nvf_projector='circular'` — `factorization.jones_projector_fields`.
- V. Liu, S. Fan, *S4: A free electromagnetic solver for layered periodic structures*,
  Comput. Phys. Commun. **183**, 2233 (2012).
  [doi:10.1016/j.cpc.2012.04.026](https://doi.org/10.1016/j.cpc.2012.04.026) — the
  least-squares tangent-field design (after Fisher et al. 2007); shape-constraint-based,
  not raster-native, and the reason none of it is adopted here directly. Its **Eq. (52)**
  is the published form of the circular-polarization limit — `phi = (pi/8)(1 + cos pi|t|)`,
  a generalization of Antos above — so that limit can be implemented from the paper,
  without reading S4's source. It **was**,
  on 2026-09-01, from the paper and the paper alone; S4's `fmm_PolBasisJones.cpp` was not
  opened. S4 states `J` in terms of the *tangent* field, which differs from the normal by a
  fixed 90-degree rotation, so building `J` from the normal directly gives the normal
  projector this module's `C_xx` wants.
- A. M. Hammond, S. G. Johnson et al., *Unifying and accelerating level-set and
  density-based topology optimization by subpixel-smoothed projection*,
  [arXiv:2503.20189](https://arxiv.org/abs/2503.20189) — distance and normal jointly from
  a filtered density; the modern minimal recipe for the same move.
