# The NVF normal field and the grid it is taken on — measurements

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-08-07. Measured on CPU, NumPy backend, float64.

Targets:
- `src/wavebit/rcwa/_engine/factorization.py::_normal_field_floor` — the floor that replaced the knob
- `src/wavebit/rcwa/_engine/factorization.py::normal_vector_field` — the distance-transform construction
- the `enhanced_dfs` parameter — removed outright, not deprecated: nothing read it, so
  leaving it to be set would have been a lie rather than a compatibility shim

---

## TL;DR

`enhanced_dfs=True` repeated every cell `4*truncation_order+1` times before taking the normal field. Across
32 combinations of shape, material, polarization and truncation order it **lost accuracy in 27
and gained in none that stood up**. It is gone — the parameter with it, from the solver
constructors and from `Numerics`, so passing it now raises rather than being quietly ignored.
The normal field is taken on the raster as given, lifted only to the `4*truncation_order+1`-point floor its
transform structurally requires.

## What the knob did

```python
if enhanced_dfs:  min = (4*truncation_order + 1) * N     # per cell
else:             min = 4*truncation_order + 1           # total
```

Verbatim from `meent`. What changed underneath it is where it lands:

| | in `meent` | now |
|---|---|---|
| sampled-path permittivity coefficients | its main purpose | **ignored** — obtained in closed form since `072710d` |
| NVF normal field | (NVF did not exist) | its only remaining reader |

So a flag whose reason for existing was the permittivity survived into a codebase where the
permittivity no longer reads it.

At `truncation_order=6`, `N=240` the two settings differ by a factor of 26 per axis — a 6240x6240 grid
against 240x240, 676x the cells, with a Euclidean distance transform run over it.

## Why refining it does not help

The normal field is not a property the raster *has*. Asked honestly, a raster's boundary is a
staircase and its normals point along the axes — which is exactly the case where NVF collapses
onto Li's rules and buys nothing. So `normal_vector_field` does something else: it takes the
distance transform of the boundary set and reads its gradient, which on a staircase comes out
roughly radial. It is a **guess at the smooth boundary the staircase stands for**.

Refining the grid does not improve the guess, because upsampling adds no information — the
staircase is the same staircase, spelled with more pixels. What it does is resolve that
staircase more sharply, so the guess tracks the steps rather than the shape behind them.

The measurement separates the two cleanly. Building the **same final grid** two ways, from a
real 480x480 raster of a circle, and from a 60x60 raster repeated 8x:

| | `truncation_order=6` | `truncation_order=10` |
|---|---|---|
| N=480, no repeat (real resolution) | **3.65e-04** | **1.89e-03** |
| N=60 repeated 8x (same grid) | 1.28e-02 | 1.18e-02 |
| N=60, no repeat | 5.03e-03 | 5.53e-03 |

Same number of grid points, 35x apart. Refinement that carries information converges;
refinement that duplicates it does not, and costs accuracy.

Raising the real raster resolution behaves as it should:

| N | 30 | 60 | 120 | 240 | 480 |
|---|---|---|---|---|---|
| NVF error (dielectric circle, `truncation_order=6`) | 7.63e-03 | 5.03e-03 | 1.43e-03 | 1.33e-03 | 3.65e-04 |

### How far off the guess is, as an angle

The efficiencies above say the guess costs something; this says how wrong the field itself is,
which is what makes "it does not converge" concrete. Measured 2026-08-16 against the exact
radial normal of the circle the staircase stands for, in bands by distance from the true
boundary:

| distance from boundary | 0-1 px | 1-2 px | 2-4 px | 4-8 px | 8-16 px |
|---|---|---|---|---|---|
| mean angular error | 14.4 deg | 10.4 deg | 7.5 deg | 5.1 deg | 3.6 deg |

**Every column is flat in `N` from 64 to 512.** The error depends on how many *pixels* away
from the boundary a point is and not on how large a pixel is, which is the statement above made
sharp: the guess is a property of the staircase, and a finer staircase is the same staircase.

The mean hides a structure worth knowing, because it is not spread evenly around the arc. In
the boundary band at `N=256` the distribution runs 5.2 / 12.3 / 21.7 / 30.1 deg at the 25th /
50th / 75th / 90th percentile, and by position on the arc:

| arc angle (0 = axis, 45 = diagonal) | 0-10 | 10-25 | 25-40 | 40-50 |
|---|---|---|---|---|
| mean angular error | 7.5 deg | 20.0 deg | 16.2 deg | 2.5 deg |

So the guess is at its *best* where the step pattern is commensurate with the direction it has
to represent -- along the axes, and on the diagonal where steps alternate one-for-one (1.3 deg
there at `N=128`) -- and at its worst in between, around 18-27 deg, where it locks onto local
step runs that do not average to the true angle. The extreme outliers are the other way about:
the worst 3% of pixels, up to ~40 deg, cluster near the diagonal, so a *mean* read at 45 deg and
a *maximum* read at 45 deg point in opposite directions.

**The field does not interpolate, and that is the whole of the split.** Every direction it
returns in the band lands within 0.5 deg of `arctan(p/q)` for small integers -- 3476 of 3476
band pixels at `N=1024`, with `p, q <= 4`. The output is a short list of directions, not a
continuum. Walking outward from the axis it returns `0.0 deg` for essentially every pixel out to
16 deg of true angle, which is why the error there *equals* the arc angle rather than merely
tracking it: 0.7 / 1.2 / 1.8 / 2.3 deg of error at 0.7 / 1.2 / 1.8 / 2.3 deg of arc, and the same
identity holds walking outward from 45 deg. The outliers are the pixels that emit a *different*
entry from the same list -- `45.0` where their neighbours emit `0.0` -- and one substitution is
worth 30-40 deg by itself.

So 0 and 45 deg are cheap and expensive for the same reason: the true normal is *in* the list
there, so most pixels are nearly exact, and a pixel that takes the wrong entry is maximally
wrong.

**What decides which entry.** The direction is `arctan2` of the two central differences of the
distance field, which is what the code computes and nothing more --
`arctan2(d[r+1,c]-d[r-1,c], d[r,c+1]-d[r,c-1])` reproduces the shipped field to 0.0 deg. The
quantisation is in those differences: taken over a two-cell span of a distance-to-a-digital-set,
they land on small integers. In the mask band, 71% of cells have both differences integral, and
the integer pairs that occur are only `{1,0}`, `{2,0}` and `{1,1}` -- so **only 0 and 45 deg are
available there at all**, which is the plateau structure exactly.

Which of the two a cell gets is decided by the staircase under it, and the picture is the whole
argument. `A` is a cell returning 0 deg, `D` one returning 45, `#` interior, `.` exterior, on the
arc near the axis:

    ####AD...
    ####AA...
    ####AA...      <- a flat run: every cell {1,0}, returns 0 deg, and 0 deg is right
    ####AA...
    ####AD...      <- the step: {1,1}, returns 45 deg
    ###DA....

`D` sits on the jog and nowhere else. Along a flat run the distance field rises in one direction
only, so the cross difference is 0; at the one-pixel jog it rises equally both ways and the cell
takes `{1,1}`. Near the axis, runs are long and jogs rare -- 32 of 256 band cells within 12 deg
of the axis, 12.5%, matching the 12% of pixels past 30 deg reported above -- so those few sit 45
deg away from every neighbour. Near the diagonal every cell is a jog, `{1,1}` is the rule rather
than the exception, and 45 deg is the right answer, which is why the mean is excellent there and
the rare non-jog cell becomes the outlier instead.

One candidate is closed on the way: it is *not* the direction to the nearest mask cell, which
agrees only 39% of the time within 1 deg, median disagreement 3.95 deg.

### A zero normal is a different thing, and is not what the outliers are

`normal_vector_field` returns `(0, 0)` on the **skeleton** of the distance field -- the points
equidistant from two or more parts of the boundary. Read the distance field as terrain with the
boundary at sea level and the interior rising away from it: the skeleton is the watershed line
along the top, and standing on it the gradient has no direction to choose, because both ways
lead down equally.

Concentric squares put that ridge where it cannot be missed. `#` is the high-index ring, `X` a
pixel where the field comes back exactly `(0, 0)`:

    ....X##############X....
    ....################....
    ....################....
    ....################....
    ....####X......X####....
    ....####........####....
    ....####........####....
    ....####........####....
    ....####........####....
    ....####........####....
    ....####........####....
    ....####X......X####....
    ....################....
    ....################....
    ....################....
    ....X##############X....

Eight of them, all at corners, and the corners are the whole of it. A corner is where two
perpendicular edges are equidistant, so the two contributions cancel *exactly*. Along the middle
of an arm the ridge is there in the continuum, but on a grid the two sides never cancel to the
last bit, so nothing is flagged and the field carries an ordinary unit vector.

**The circle measured above has none at all** -- zero pixels of 65536 -- because it has no
corners for the cancellation to be exact at. That rules out the obvious reading of the ~40 deg
outliers near the diagonal: they are not degenerate pixels being reported as if they were
directions. Every pixel in the boundary band carries `|n| = 1` to the last digit, the worst 3%
included. Wrong vectors, not missing ones.

The zero case is handled rather than divided by, and not as a patch. The rearrangement divides
by `a = n.eps.n`, which is zero there -- but with `n = 0` the numerators `p`, `q` and `n n^T`
vanish as well, so substituting `a = 1` makes the formulas return `A = eps` and `B = I` on their
own. That is the direct rule, which is exactly what a zero projector should collapse to.
`tests/rcwa/test_nvf_normal_field.py` pins it on the shape drawn above, chosen because concentric
squares put a ridge in the middle of every arm.

It is a curvature effect and nothing else. On a flat interface the same code returns the axis
normal *exactly* -- `max|n_y|` is 0.0, not small -- so nothing here is a bias in where the
boundary is taken to sit. That was checked separately, since a mask built from "cells that
differ from a 4-neighbour" marks both sides of an edge and could plausibly have sat half a
pixel off: the marked band's mean radius tracks the true radius to within 0.03 px at every `N`
up to 1024, with the sign alternating rather than accumulating, and a halfwidth of 0.53 px. The
band straddles the boundary symmetrically, so no sub-pixel offset enters and the mirror
symmetry of the field is exact (`normal_vector_field` is bit-exactly covariant under a flip).

None of this is a defect *refinement* can fix -- that is the finding of this file -- and it is
part of why the analytic and boundary paths exist, both of which build the normal field from
the shape instead of from its raster (a levelset path did too; it is no longer in the
package). It did, however, turn out to be a
defect a different construction can fix on the raster itself: the quantization lives in the
integer-grid differencing, not in the raster, and smoothing the indicator before
differentiating removes it. Three such constructions were measured and shipped behind the
`raster_normal_method` keyword (default unchanged) -- see
[`nvf-normal-methods.md`](../../theory/rcwa/nvf-normal-methods.md), which reads this file's tables as its
baseline.

### One comparison against `li`, on the same raster

Everything above describes how crude the normal field is and says nothing about what that costs
next to the alternative that does not use one. This is a single measurement of that, recorded
because it was not otherwise in this file. **It is one shape at one incidence and is not a claim
about either factorization in general.**

Setup: a circle of radius 0.30 in a unit cell, index 2.4 in vacuum, thickness 0.30, wavelength
0.55, `theta` 12 deg, `phi` 30 deg, `psi` 40 deg. Reference is the analytic (staircase-free)
path on the exact circle at `truncation_order=14`. The two raster runs take the identical raster and differ
only in `factorization`. Error is the largest per-order difference in R or T against the
reference, over the orders both retain.

| N | `truncation_order` | raster `nvf` | raster `li` |
|---|---|---|---|
| 32 | 6 | 6.95e-02 | 8.89e-02 |
| 32 | 10 | 5.63e-02 | 7.45e-02 |
| 64 | 6 | 2.15e-02 | 3.90e-02 |
| 64 | 10 | 8.75e-03 | 2.37e-02 |
| 128 | 6 | 1.95e-02 | 3.98e-02 |
| 128 | 10 | 8.00e-03 | 2.36e-02 |
| 256 | 6 | 1.43e-02 | 3.75e-02 |
| 256 | 10 | 3.51e-03 | 2.08e-02 |

**The reference carries 1.7e-3 of its own**: `|truncation_order=12 - truncation_order=14|` on the same measure. Read the
last row with that in mind -- 3.51e-03 is close enough to that floor that what separates the two
columns there is not resolved by this experiment, while the `N=128` rows sit clearly above it.
Nothing here says what happens on a shape with corners, at another incidence, or at truncations
where either column has stopped converging.

## The sweep

Base raster 240x240 (fine enough that the staircase error is small), truth from the parametric
shape path at `truncation_order=16`, error on the specular transmitted order.

| case | `truncation_order` | sampled | R=1 | R=2 | R=4 |
|---|---|---|---|---|---|
| dielectric circle, TM | 4 | 1.60e-02 | **3.73e-03** | 5.34e-03 | 6.43e-03 |
| | 6 | 1.25e-02 | **1.33e-03** | 2.86e-03 | 3.90e-03 |
| | 8 | 9.92e-03 | **2.22e-03** | 3.72e-03 | 4.71e-03 |
| | 10 | 8.35e-03 | **2.71e-03** | 4.17e-03 | 5.11e-03 |
| dielectric rect 30°, TM | 4 | 2.38e-02 | **3.43e-03** | 5.13e-03 | 6.07e-03 |
| | 6 | 1.81e-02 | **1.67e-03** | 3.29e-03 | 4.23e-03 |
| | 8 | 1.43e-02 | **1.88e-03** | 3.51e-03 | 4.48e-03 |
| | 10 | 1.16e-02 | **2.32e-03** | 3.95e-03 | 4.90e-03 |
| dielectric rect 10°, TM | 4 | 4.31e-03 | 5.29e-03 | 6.57e-03 | 7.13e-03 |
| | 6 | 3.77e-03 | **2.76e-03** | 3.85e-03 | 4.33e-03 |
| | 8 | 3.02e-03 | **2.67e-03** | 3.60e-03 | 4.02e-03 |
| | 10 | 2.41e-03 | **1.84e-03** | 2.67e-03 | 3.05e-03 |

Over all 32 rows (adding metallic circle and rect, and both polarizations): **27 worse with
upsampling, 4 unchanged, 1 better** — and that one is a case whose analytic reference had not
converged (`truncation_order=12` residual 8.7e-03), so it does not count for anything.

Two controls make the separation exact:

- **The permittivity coefficients are invariant under upsampling** — measured at 1e-16, so
  everything above is the normal field and nothing else.
- **Repeating the `structure` raster manually and repeating it via `enhanced_dfs=True` are
  bit-identical**
  (0.00e+00), so the manual sweep measures the real setting.

## Why the floor stays

The floor is not an accuracy preference, it is structural. The normal field has no closed-form
transform (unlike the permittivity, which `_pixel_correction` handles for any order), so
`fourier2d_pointwise` reads it off the grid, and the circulant asks for orders up to `±2*truncation_order`. On a grid
shorter than `4*truncation_order+1`, `k % n` folds distinct orders onto one bin.

The consequence is not gradual. A **constant** normal field must transform to the identity for
NVF to collapse onto Li's rules on a one-cell axis. On a short grid it instead picks up
off-diagonals wherever `(r - c) % n == 0`, and the collapse fails outright — a 1D grating of 8
cells at `truncation_order=10` returned 0.1236 against Li's 0.0324. That is what
`test_nvf_on_a_one_cell_axis_reduces_to_li` pins.

So: lift to the floor, never past it.

## What this cost

`2d_nvf` in `golden_legacy.npz` was recomputed. It is a random 5x5 raster, where the grid goes
from 50x50 to 10x10 and the answer moves a long way (transmitted efficiency 0.684 to 0.352
under TM incidence; energy conserved to 1.000000000000 either way). There is no independent
oracle for that case and there cannot really be one — NVF's premise is that a staircase stands
for a smooth boundary, and a random raster has none. The justification is this document.

## Reproduce

The sweeps here are not checked in as a benchmark; they are one-off measurements against the
parametric shape path. Three neighbouring write-ups cover the questions around this one:
[`parametric-modeling.md`](../../guide/rcwa/parametric-modeling.md) on how the parametric path compares to the raster
and staircase representations, [`backend-choice.md`](backend-choice.md) on what the change
here did to the cost of NVF relative to the other two strategies (it removed the difference),
and [`nvf-normal-methods.md`](../../theory/rcwa/nvf-normal-methods.md) on the raster constructions that remove
the quantization this file measured (the `raster_normal_method` keyword).

### References

- L. Li, *Use of Fourier series in the analysis of discontinuous periodic structures*, JOSA A
  **13**(9), 1870 (1996). [doi:10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)
  — the rules NVF must collapse onto when the normal field is constant, which is what the
  `4*truncation_order+1` floor protects.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, W. Osten, *Normal vector method for convergence
  improvement using the RCWA for crossed gratings*, JOSA A **24**(9), 2880 (2007).
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880) — the method itself.
- P. Götz, T. Schuster, K. Frenner, S. Rafler, W. Osten, *Normal vector method for the RCWA with
  automated vector field generation*, Opt. Express **16**(22), 17295 (2008).
  [doi:10.1364/OE.16.017295](https://doi.org/10.1364/OE.16.017295) — the paper that addresses
  generating the field for an arbitrary shape, which is the problem the distance transform here
  solves. Whether it is the *same* construction has not been checked against the full text; see
  the same caveat in [`parametric-modeling.md`](../../guide/rcwa/parametric-modeling.md).
