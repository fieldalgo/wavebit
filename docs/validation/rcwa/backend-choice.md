# Backend choice for a differentiated run — measurements

Written: 2026-08. Re-measured 2026-08-27, after the block modeling path was removed --
the first table is the *shape* path where a block one used to be. Measured on CPU (Apple
silicon, single process, float64).
**Every number here is from one CPU host; see the caveats before carrying any of it elsewhere.**

**Interface scope (added 2026-10-15).** Every mode below was measured through the solver
interface now kept as `wavebit.compat.legacy`: `solver.freeze_geometry()` and
`solver.solve(geometry=...)`. The public lifecycle, `rcwa.prepare(...)` then `prepared.run(problem)`,
refuses a shape dimension traced under `jax.jit` with `prepared.host_topology`, and differentiates
one under eager `jax.grad` by checking its topology on the tracer's primal. So the "jax jit on
shapes" rows and recommendations apply to the compatibility interface and do not carry over to
that lifecycle, where a shape derivative is a PyTorch or an eager `jax.grad` derivative. Backend speed through `prepared.run`
has not been measured. The [differentiation guide](../../guide/common/autodiff.md#rcwa-which-inputs-differentiate-on-which-backend)
lists which inputs differentiate on which backend there.

Targets:
- `studies/rcwa/benchmark_backends/benchmark_backends.py` — the sweep this document reports
- `src/wavebit/rcwa/_engine/geometry.py::StaticGeometry` — what a freeze records, and what its equality
  costs
- `src/wavebit/rcwa/_engine/factorization.py::compression_indices`, `normal_field_for_raster` — the two
  raster decisions that must be taken eagerly
- `src/wavebit/rcwa/_engine/geometry.py::StaticGeometry` — the frozen records and their staleness checks
- [`nvf-upsampling.md`](nvf-upsampling.md) — why NVF's cost fell between the first run of this
  sweep and the one reported here; [`torch-compile.md`](../../development/torch-compile.md) — why PyTorch is
  measured eager

---

## The question

Wavebit produces the same gradient three ways, and they are not interchangeable:

| mode | how the discrete geometry is decided | frozen record |
|---|---|---|
| **torch** | re-decided every call, eagerly | none |
| **jax eager** | re-decided every call, narrowed inside the grad trace | none |
| **jax jit** | decided once, replayed as a compile-time constant | required |

Only the third can go stale, because only the third carries a record forward. Everything below
is about what that buys, and where it stops paying.

Every row was gated on agreement: the modes' gradients are compared through a fixed random
projection, and the spread never exceeded **2.1e-10** (typically 1e-12). Nothing here reports
the speed of a wrong answer.

## TL;DR

- **On the shape path `jax jit` wins at every size measured**, by 2.3x at
  `truncation_order=3` and 1.3x at 13, repaying its compilation in 13 steps at worst and 2 at
  best. This is the conclusion that moved: on the removed block path PyTorch won below
  `truncation_order≈5` and compilation never repaid there at all.
- **On the raster path PyTorch still wins below `truncation_order≈5`,** and that half of the
  document is unchanged — which is also the control that says the machine did not move.
  Against the 2026-08-20 run it scatters 0.86-1.27x with a median of 1.04.
- **`jax eager` is no longer the loser it was.** It runs 0.9-1.3x PyTorch on the shape path --
  *faster* on five of ten rows -- where it was 2-11x slower on the block path. Its value is
  still elsewhere: it is the mode with no record, so it cannot be served a stale one.
- **Re-freezing every step is free on the shape path and ruinous on the raster path.** A shape
  layer's freeze costs 0-1 ms and its record compares by content; the raster records hold
  arrays, fall back to identity, and recompile every step.
- **At low `truncation_order` the shape path's cost is `shape_nvf_grid_size`, not the backend.** At
  `truncation_order=3` the default 1024 spends 63 ms building a normal field against a 15 ms
  solve. Dropping to 256 makes it 5 ms. No backend choice is worth as much.
- **All three strategies now cost the same.** NVF used to grow with the raster — 359 ms of
  freeze against a 277 ms step at `N=128, truncation_order=5` — because its normal field was taken on a
  grid `4*truncation_order+1` times finer per axis. That is gone ([`nvf-upsampling.md`](nvf-upsampling.md));
  its freeze is
  now under 2 ms and its step is level with the sampled path.

---

## Shape path — the design variable is a *length*

An ellipse stack, `d(sum T)/d(lx)`. This is what the closed-form series exists for: the shape's
outline enters the Fourier transform analytically, so the gradient reaches CD, sidewall angle
and pitch. `shape_nvf_grid_size` is left at its default of 1024, which matters more than it looks —
see [the decomposition](#where-the-time-goes-and-what-the-knob-actually-is) below. Times are ms per step.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| truncation_order=3, L=1 | 141 | 183 | 61 | 66 | 962 | 0 |
| truncation_order=5, L=1 | 241 | 264 | 131 | 132 | 1012 | 0 |
| truncation_order=7, L=1 | 665 | 593 | 431 | 432 | 1395 | 0 |
| truncation_order=9, L=1 | 1944 | 1813 | 1464 | 1463 | 2576 | 0 |
| truncation_order=11, L=1 | 5351 | 4766 | 4079 | 4082 | 5092 | 0 |
| truncation_order=13, L=1 | 12951 | 11899 | 10080 | 9880 | 11178 | 0 |
| truncation_order=5, L=3 | 687 | 715 | 202 | 199 | 1833 | 0 |
| truncation_order=7, L=3 | 1739 | 1643 | 726 | 743 | 2500 | 0 |
| truncation_order=9, L=3 | 4813 | 4958 | 2639 | 2663 | 4502 | 1 |
| truncation_order=11, L=3 | 13366 | 13719 | 7380 | 7312 | 9801 | 0 |

**This table replaced a block-path one, and the conclusion moved with it.** There, PyTorch won
below `truncation_order≈5` and `jax jit` never repaid its compilation at `truncation_order=3`.
Here `jax jit` wins every row, by 1.28x to 3.41x, and most widely at the *small* end (2.3x at
`truncation_order=3` against 1.3x at 13; the 3.41x is three layers at 5). The
reason is in the decomposition below: the shape path's modeling half is a few large array
operations rather than hundreds of small ones, and it is the half a compiler can fuse.

**Read one table at a time.** Nothing is timed until a 45 s warm-up has run, and the sweep
times one fixed configuration three times to state its own repeatability:

| | start | after shapes | after raster | spread |
|---|---|---|---|---|
| probe (ms) | 141 | 137 | 141 | **1.02x** |

Within that spread, and with the four modes on a row timed back to back, a comparison *along a
row* is sound. Across runs it is less so: the same commit once measured 133 ms and 217 ms on
the same configuration on different days, and an A/B over four commits found no code difference
that explains it. Both tables here come from one run for that reason. Treat a ratio as
meaningful and a third significant figure as not.

**The raster table below is the control for that.** It measures code the block-path removal did
not touch. Over its cells above 40 ms — below that a ±1 ms quantum is already 6% — it lands at
0.86-1.27x the 2026-08-20 run with a **median of 1.04**: scatter about parity, inside the
run-to-run variation this section just described, with no drift in either direction. So the
shape-path numbers above differ from the block-path ones they replaced because the *paths*
differ, not because the machine did.

### Re-freezing every step costs nothing here

`jit (re-freeze)` retakes the freeze on every step — the only shape of loop that cannot carry a
stale record forward. It tracks `jit (one freeze)` to within the probe's own spread, and on
four rows it is *faster*, which is the same statement as "the difference is noise".

Two things make it free. The freeze itself costs **0-1 ms** on this path, against 4-77 ms on
the block path, because a shape layer's plan is a list of polygon loops rather than a grid.
And the record compares **by content**: it holds which loop has which vertex and which edges
cross, while the vertex *coordinates* stay traced, which is what carries the gradient. A length
can move a long way without the combinatorics changing, and an unchanged record reuses the
compiled program — measured directly below.

### When compilation pays for itself

Break-even against PyTorch, counting the compile a fit must amortise:

| config | torch − jit(re-freeze) | compile | steps to break even |
|---|---|---|---|
| truncation_order=3, L=1 | +75 ms | 962 | 13 |
| truncation_order=5, L=1 | +110 ms | 1012 | 10 |
| truncation_order=7, L=1 | +234 ms | 1395 | 6 |
| truncation_order=9, L=1 | +481 ms | 2576 | 6 |
| truncation_order=11, L=1 | +1269 ms | 5092 | 5 |
| truncation_order=13, L=1 | +3070 ms | 11178 | 4 |
| truncation_order=5, L=3 | +489 ms | 1833 | 4 |
| truncation_order=7, L=3 | +995 ms | 2500 | 3 |
| truncation_order=9, L=3 | +2150 ms | 4502 | 3 |
| truncation_order=11, L=3 | +6054 ms | 9801 | 2 |

Every row repays, and the worst case is a dozen steps — shorter than any fit. Layers help more
than orders do: three layers at `truncation_order=5` break even in 4 steps where one layer at 5
takes 10, because the compiled program amortises the per-layer modeling that PyTorch pays
again for each layer.

<a name="where-the-time-goes"></a>

### Where the time goes, and what the knob actually is

Splitting a *forward* call into the modeling step (`eps_conv_matrices`) and the solve
(`solve_conv`) locates all of it. The shape moves between calls, which is not a detail: the
analytic normal field is cached on the shape parameters, so timing the same call twice measures
a cache hit — 1.1-1.4 ms whatever `shape_nvf_grid_size` is, which is the one number a fit never sees.

| truncation_order | shape_nvf_grid_size | torch modeling | torch solve | jax modeling | jax solve |
|---|---|---|---|---|---|
| 3 | 256 | 5.3 | 14.0 | 11.0 | 19.8 |
| 3 | **1024** (default) | **62.6** | 15.2 | **50.2** | 21.0 |
| 3 | 2048 | 262.3 | 13.4 | 168.0 | 20.4 |
| 7 | 1024 | 78.4 | 344.3 | 53.9 | 252.2 |
| 11 | 1024 | 151.4 | 3181.6 | 109.4 | 2385.6 |

**1. At low `truncation_order` the shape path is a normal-field benchmark.** At
`truncation_order=3` the default `shape_nvf_grid_size=1024` spends 63 ms of modeling against a 15 ms
solve — four times the solve it exists to feed. The cost is linear in pixel count (256 → 1024
is 16x the pixels and 11.7x the time; 1024 → 2048 is 4x and 4.2x), so it is the distance
transform, and dropping to 256 buys more than any backend choice on this page. It stops
mattering by `truncation_order=7`, where the solve is 4x the modeling.

**2. The old finding about `jax eager` has inverted, and the mechanism says why.** On the block
path JAX was ~9x slower on the modeling step *at every size*, because JAX has no true eager
mode and the block modeller emitted one small array operation per block — 902 of them at
`n_split=20`. PyTorch's C++ dispatcher absorbed that; JAX's did not. The shape path emits a
handful of *large* operations instead, so per-operation dispatch is amortised away and JAX is
**faster** on the modeling half at every point above: 50.2 against 62.6 ms, 109.4 against
151.4. The old number measured dispatch overhead, and there is no longer a stream of tiny
operations for it to accumulate over.

**3. The solve behaves as it did.** JAX is slower at `truncation_order=3` (21.0 against 15.2)
and faster from 7 up (252 against 344; 2386 against 3182), so `jax eager` closing the gap at
high `truncation_order` is not that path improving — it is one large eigendecomposition growing
until it is the whole measurement.

### Where the recompilation lands — and where none does

Re-freezing at every point of an `lx` sweep, `truncation_order=3`, `shape_nvf_grid_size=256`:

| one ellipse | 85 | 90 | 95 | 100 | 103 | 106 | 110 | 115 | 120 | 125 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 1075 | 20 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 |

**One compilation, at the first point, and then nothing** — a 47% move in the design variable
for a single compile. A lone shape has no combinatorics that can change: there is no second
shape to start or stop overlapping it, so its plan is `None` at every point and the same
compiled program serves the whole sweep. On the block path the same sweep recompiled at
`lx≈115`, where two grid edges crossed.

So the boundary has to be shown with a pair. Two rectangles 120 apart, each `lx` wide, which
meet as `lx` passes 120:

| two rectangles | 100 | 105 | 110 | 115 | 118 | **122** | 126 | 130 | 140 |
|---|---|---|---|---|---|---|---|---|---|
| ms | 1033 | 23 | 22 | 23 | 23 | **1070** | 28 | 26 | 28 |

Two compilations: one at the start, one at `lx=122` — the first point past contact, where the
layer stops being a sum of two disjoint transforms and becomes a resolved painting. That is
exactly where a *held* record would have gone stale, so the recompilation and the staleness
boundary are the same event, and re-freezing every step converts a silent wrong answer into a
one-second pause.

---

## Raster path — the design variable is the *permittivity grid*

A binary permittivity raster, `d(sum T)/d(structure)`, swept across all three Fourier
strategies.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| N=16, truncation_order=3, sampled | 21 | 79 | 16 | 16 | 811 | 0 |
| N=16, truncation_order=3, integrated | 19 | 78 | 16 | 709 | 995 | 0 |
| N=16, truncation_order=3, NVF | 20 | 87 | 16 | 668 | 808 | 0 |
| N=32, truncation_order=3, sampled | 20 | 79 | 15 | 16 | 794 | 0 |
| N=32, truncation_order=3, integrated | 21 | 78 | 16 | 709 | 979 | 0 |
| N=32, truncation_order=3, NVF | 22 | 87 | 17 | 643 | 755 | 1 |
| N=32, truncation_order=5, sampled | 133 | 154 | 84 | 84 | 815 | 0 |
| N=32, truncation_order=5, integrated | 140 | 155 | 87 | 770 | 1014 | 0 |
| N=32, truncation_order=5, NVF | 137 | 168 | 87 | 747 | 898 | 1 |
| N=64, truncation_order=3, sampled | 22 | 80 | 15 | 16 | 844 | 0 |
| N=64, truncation_order=3, integrated | 19 | 79 | 16 | 721 | 986 | 0 |
| N=64, truncation_order=3, NVF | 23 | 89 | 18 | 677 | 828 | 1 |
| N=64, truncation_order=5, sampled | 133 | 154 | 85 | 86 | 837 | 0 |
| N=64, truncation_order=5, integrated | 137 | 152 | 84 | 791 | 1077 | 0 |
| N=64, truncation_order=5, NVF | 148 | 163 | 86 | 704 | 868 | 1 |
| N=64, truncation_order=7, sampled | 554 | 492 | 379 | 380 | 1172 | 0 |
| N=64, truncation_order=7, integrated | 571 | 501 | 403 | 1137 | 1445 | 1 |
| N=64, truncation_order=7, NVF | 605 | 522 | 399 | 1057 | 1157 | 2 |
| N=128, truncation_order=3, sampled | 23 | 84 | 16 | 16 | 847 | 0 |
| N=128, truncation_order=3, integrated | 33 | 81 | 16 | 761 | 1061 | 1 |
| N=128, truncation_order=3, NVF | 27 | 95 | 17 | 724 | 843 | 5 |
| N=128, truncation_order=5, sampled | 149 | 156 | 90 | 87 | 932 | 0 |
| N=128, truncation_order=5, integrated | 143 | 154 | 86 | 810 | 1144 | 1 |
| N=128, truncation_order=5, NVF | 144 | 173 | 88 | 727 | 922 | 5 |

**This is the control.** None of the code behind this table was touched by the block-path
removal, and every cell is within ~12% of the 2026-08-20 run — uniformly, and in the same
direction. That is what licenses reading the shape-path table above as a difference between
paths rather than between machines.

Four things this says.

**1. Re-freezing is ruinous for the integrated and NVF records.** Their records hold arrays, where value equality
is neither well defined nor hashable, so `StaticGeometry` falls back to identity and a new
freeze misses the compilation cache every time: ~700 ms against a ~16 ms step. The
content-hash path helps the boundary plans only. On these two strategies a fit must either hold one freeze
(and risk staleness — `_check_compression_still_fits` and `_check_normal_field_still_fits`
refuse it eagerly, but a jitted call cannot check) or not use jit.

**2. The default pairing has nothing to freeze.** `('sampled', 'li')` records nothing at all,
so its `re-freeze` column matches its `one freeze` column exactly and staleness cannot arise.
This is the strategy to reach for in a raster inverse design whose pattern moves every step.

**3. Every raster strategy differentiates unfrozen.** A grad tracer refuses a whole-array
NumPy conversion but permits narrowing one value at a time — its primal is concrete, so each
element reads:

| | `np.asarray(whole_array)` | `bool(one_element)` |
|---|---|---|
| `jax.grad` tracer | `TracerArrayConversionError` | **OK** |
| `jax.jit` tracer | `TracerArrayConversionError` | `TracerBoolConversionError` |

Both eager decisions ask the second way. The compression compares neighbouring rows and columns
directly; the normal field cannot (SciPy wants the mask as an array), so it derives the mask
from the *base* raster instead of reading it off the grid it was lifted onto. Since the lift is
a block repeat the two are the same mask. So `geometry=` is optional for a gradient on every
raster path, which is what makes a fit whose pattern moves every step safe.

**4. Nothing here separates the strategies on cost any more.** The sampled, integrated and
NVF builds land within a
few percent of each other at every size. NVF used to be the outlier, growing with `N` where the
others were flat, because its distance transform ran on the `(4*truncation_order+1)*N` upsampled grid; it no
longer upsamples, and its freeze fell from 359 ms to 5 ms at `N=128, truncation_order=5`. So the choice among
them is now about accuracy, which is what it should have been.

---

## If a fit must hold one freeze — how far it stays valid

Re-freezing every step removes the question on the shape path — its freeze costs 0-1 ms and
its record compares by content — but the raster records give no choice: their identity
comparison makes a re-freeze a recompilation, so a jitted raster fit holds one.

**Everything from here to the end of this section is archived.** It was bisected on the block
path, whose `n_split` is the abscissa of the table below and does not exist any more. It is
kept for the *shape* of the answer, which is what it was about: the boundary is sharp, exact up
to it (a replay at 106.31 matched a rebuild at 0.0e+00) and refused past it (106.33), and it
narrows sharply as a shape is described more finely. The equivalent measurement on the boundary
route — where the abscissa is which shapes overlap, not how many slabs a shape has — has not
been taken; the recompilation sweep above shows its boundary on one pair, which is a different
and much weaker statement.

| shape | parameter (nominal) | n_split | valid window | width |
|---|---|---|---|---|
| ellipse | `lx` (85) | 3 | 55.00 .. 144.46 | 89.5 (−35.3% .. +70.0%) |
| ellipse | `lx` (85) | 4 | 55.00 .. 114.94 | 59.9 (−35.3% .. +35.2%) |
| ellipse | `lx` (85) | 6 | 55.00 .. 90.69 | 35.7 (−35.3% .. +6.7%) |
| ellipse | `lx` (85) | 9 | 77.01 .. 106.32 | 29.3 (−9.4% .. +25.1%) |
| ellipse | `lx` (85) | 12 | 70.86 .. 90.69 | 19.8 (−16.6% .. +6.7%) |
| ellipse | `lx` (85) | 20 | 74.49 .. 86.38 | **11.9 (−12.4% .. +1.6%)** |
| rectangle | `angle` (0.27) | 1 | 0.000 .. 0.574 | 0.574 (−100% .. +113%) |
| rectangle | `angle` (0.27) | 2 | 0.000 .. 0.442 | 0.442 (−100% .. +64%) |
| rectangle | `angle` (0.27) | 4 | 0.252 .. 0.369 | 0.117 (−6.8% .. +36.5%) |
| rectangle | `angle` (0.27) | 8 | 0.203 .. 0.290 | 0.086 (−24.7% .. +7.4%) |
| rectangle | `angle` (0.27) | 12 | 0.252 .. 0.271 | 0.018 (−6.6% .. +0.2%) |
| rectangle | `angle` (0.27) | 20 | 0.267 .. 0.273 | **0.006 (−1.2% .. +0.9%)** |
| rectangle | `lx`, angle=0 | any | — | **unbounded** |

What that table said, in the past tense it now belongs in. `n_split` governed how long a
freeze lived, and not gently: more slabs meant more block edges and more chances for two to
cross, so the rotated rectangle's angle window shrank 96x from 1 slab to 20 and the ellipse's
`lx` window 7.5x from 3 to 20. At `n_split=20` a held record survived about ±1% of the nominal
angle — narrower than any OCD fit's search range, and an unfortunate pairing, since `n_split`
was raised precisely to resolve the curved and tilted sidewalls whose fits wanted it most.

Two of those windows are worth carrying forward, because they are facts about geometry rather
than about slabs. An **axis-aligned rectangle's `lx` never went stale**: its only edges are
`cx ± lx/2`, which cannot reorder — and on the shape path a lone shape of any kind is now in
that position, as the sweep above shows. And the windows were **asymmetric** (`n_split=20`
ellipse: −12.4% / +1.6%), so a record was better taken slightly above the expected convergence
point than at the centre of the search range.

## What to use

For the compatibility interface these were measured on. Through `rcwa.prepare(...)` and
`prepared.run(problem)`, a shape traced under `jax.jit` is refused and one under `jax.grad` runs
eagerly; see the interface scope at the top.

| situation | use |
|---|---|
| OCD fit on shapes, any `truncation_order` | **jax jit, re-freezing every step** |
| OCD fit on shapes, `truncation_order ≤ 5` | first drop `shape_nvf_grid_size` to 256 — it is worth more than the backend |
| OCD fit on shapes, fewer than ~13 steps | torch, and skip the compile |
| Raster inverse design, pattern moves every step | **`('sampled','li')`**, any backend |
| Raster, `truncation_order ≤ 5` | **torch** |
| Raster, integrated or NVF, pattern moves | **torch** (jit's re-freeze recompiles every step) |
| Raster, integrated or NVF, pattern fixed | jax jit, one freeze |
| Anything where a stale record would be silent | **torch**, or re-freeze on the shape path |

The general shape, and it is not what this document said before the block path was removed:
**on shapes JAX is the default and re-freezing every step is both the safe option and the fast
one; on rasters PyTorch still is.** The two paths now want different backends, which is a more
useful answer than the single verdict that replaced — and the reason is in the decomposition:
a compiler helps where there are many array operations to fuse, and the shape path's modeling
half is exactly that while a raster's is one array already.

---

## Caveats

- **CPU only.** Every number here was measured on one CPU host, and none of it has been run on
  a GPU. Do not extrapolate: the per-step times and the compilation cost need not move in the
  same direction, so neither the ratios nor the break-even table carries over on its own.
- **PyTorch was measured eager, which is how it always computes.** Under `torch.compile`
  wavebit runs eagerly inside the compiled caller (since 2026-10; before that, Inductor
  silently miscompiled it). When it was last compiled natively, in 2026-08, it measured
  0.85–1.02x — see `docs/development/torch-compile.md`, which also profiles why (97.9% of the runtime at
  `truncation_order=10` is inside LAPACK/BLAS calls Inductor forwards verbatim). So `torch`
  here means eager torch, and that is not a limitation of the measurement.
- **Single process, no batching.** Nothing here says how the modes behave under `vmap`, across
  a wavelength sweep, or on several devices — and a wavelength sweep is the case where one
  compilation is amortised over many solves, which would favour jit well beyond what is
  measured here.
- **Compile times are single observations** and vary by tens of percent between runs. The
  step times are medians of 3 after a warm-up.
- The break-even table assumes a fit whose discrete description is stable between steps. On
  the shape path that is now the ordinary case rather than an assumption — a lone shape has no
  combinatorics at all — but a layer whose shapes cross in and out of each other every step
  pays the compile every step, and no amount of `truncation_order` rescues that.
- **`shape_nvf_grid_size` is left at its default throughout the shape table**, so those rows carry a
  63 ms normal field that a fit is free not to pay. Read the decomposition before reading the
  step times as a property of the backends.

Reproduce with `python studies/rcwa/benchmark_backends/benchmark_backends.py`. The run this document reports is
checked in as `studies/rcwa/benchmark_backends/records/backends.log`. A run also writes the same numbers as JSON
beside it, which the repository deliberately does not track (`*.json` is ignored): the log is
the record, and a second copy would be a second thing to keep true.
