# Backend choice for a differentiated run — measurements

Written: 2026-08. Measured on CPU (Apple silicon, single process, float64).
**Every number here is from one CPU host; see the caveats before carrying any of it elsewhere.**

Targets:
- `benchmarks/benchmark_backends.py` — the sweep this document reports
- `fieldwave/_core/geometry.py::StaticGeometry` — what a freeze records, and what its equality
  costs
- `fieldwave/_core/factorization.py::compression_indices`, `normal_field_for_ucell` — the two
  raster decisions that must be taken eagerly
- `fieldwave/_core/modeler.py::_check_plan_still_fits` — the vector record's staleness check

---

## The question

FieldWave produces the same gradient three ways, and they are not interchangeable:

| mode | how the discrete geometry is decided | frozen record |
|---|---|---|
| **torch** | re-decided every call, eagerly | none |
| **jax eager** | re-decided every call, narrowed inside the grad trace | none |
| **jax jit** | decided once, replayed as a compile-time constant | required |

Only the third can go stale, because only the third carries a record forward. Everything below
is about what that buys, and where it stops paying.

Every row was gated on agreement: the modes' gradients are compared through a fixed random
projection, and the spread never exceeded **2.0e-10** (typically 1e-12). Nothing here reports
the speed of a wrong answer.

## TL;DR

- **Below `fto≈5`, PyTorch wins outright.** JAX's compilation never pays itself back, and
  `jax jit` with a re-freeze each step is *slower* than eager PyTorch.
- **From `fto≈7` up, `jax jit` is roughly 2x PyTorch** and repays its compilation in under a
  dozen steps.
- **`jax eager` is rarely the right answer on speed.** It is 2-11x slower than PyTorch at
  small scale and level with it at large scale. Its value is elsewhere: it is the mode with no
  record, so it cannot be served a stale one — and every raster strategy now reaches it.
- **Re-freezing every step is free on the vector path and ruinous on the raster path.** The
  vector record compares by content, so an unchanged structure reuses its compilation; the
  raster records hold arrays, fall back to identity, and recompile every step.
- **All three strategies now cost the same.** NVF used to grow with the raster — 359 ms of
  freeze against a 277 ms step at `N=128, fto=5` — because its normal field was taken on a
  grid `4*fto+1` times finer per axis. That is gone (`docs/nvf-upsampling.md`); its freeze is
  now under 2 ms and its step is level with DFS.

---

## Vector path — the design variable is a *length*

An ellipse stack, `d(sum de_ti)/d(lx)`. This is what the closed-form series exists for: the
segment edges enter the Fourier transform analytically, so the gradient reaches CD, sidewall
angle and pitch. Times are ms per step.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| fto=3, split=4, L=1 | 26 | 110 | 17 | 21 | 1653 | 4 |
| fto=3, split=12, L=1 | 22 | 169 | 17 | 29 | 1839 | 12 |
| fto=3, split=20, L=1 | 25 | 251 | 17 | 40 | 2007 | 23 |
| fto=5, split=4, L=1 | 132 | 180 | 82 | 86 | 1703 | 4 |
| fto=5, split=12, L=1 | 130 | 242 | 87 | 92 | 1840 | 12 |
| fto=5, split=20, L=1 | 133 | 322 | 81 | 103 | 2015 | 22 |
| fto=5, split=12, L=3 | 341 | 648 | 124 | 160 | 2717 | 35 |
| fto=7, split=4, L=1 | 554 | 521 | 376 | 381 | 2008 | 4 |
| fto=7, split=12, L=1 | 589 | 575 | 379 | 387 | 2227 | 13 |
| fto=7, split=20, L=1 | 583 | 647 | 372 | 397 | 2374 | 23 |
| fto=7, split=12, L=3 | 1342 | 1545 | 608 | 637 | 3150 | 34 |
| fto=7, split=20, L=3 | 1423 | 1803 | 600 | 676 | 3483 | 70 |
| fto=9, split=12, L=1 | 1886 | 1727 | 1350 | 1369 | 3314 | 12 |
| fto=9, split=20, L=1 | 1917 | 1785 | 1336 | 1364 | 3378 | 23 |
| fto=9, split=20, L=3 | 4408 | 4757 | 2375 | 2466 | 5408 | 68 |
| fto=11, split=20, L=3 | 13594 | 12652 | 7112 | 6985 | 10086 | 77 |

**Read one table at a time.** Nothing is timed until a 45 s warm-up has run, and the sweep
times one fixed configuration three times to state its own repeatability:

| | start | after vector | after raster | spread |
|---|---|---|---|---|
| probe (ms) | 150 | 127 | 145 | **1.18x** |

Within that spread, and with the four modes on a row timed back to back, a comparison *along a
row* is sound. Across runs it is not: the same commit measured 133 ms and 217 ms on the same
configuration on different days, and an A/B over four commits found no code difference that
explains it. Both tables here come from one run for that reason. Treat a ratio as meaningful
and a third significant figure as not.

### Re-freezing every step costs almost nothing here

`jit (re-freeze)` retakes the freeze on every step — the only shape of loop that cannot carry a
stale record forward. It tracks `jit (one freeze)` within the freeze's own cost, because the
vector record compares **by content**: it is which block edge becomes which grid line, in what
order, and the edge *coordinates* stay traced. A length can move a long way without the
ordering changing, and an unchanged ordering reuses the compiled program.

The gap between the two columns is the freeze itself (4-70 ms), which grows with `n_split` and
layer count and is otherwise flat.

### When compilation pays for itself

Break-even against PyTorch, counting the compile a fit must amortise:

| config | torch − jit(re-freeze) | compile | steps to break even |
|---|---|---|---|
| fto=3, split=4 | +5 ms | 1653 | 331 |
| fto=3, split=12 | −7 ms | 1839 | **never** |
| fto=3, split=20 | −15 ms | 2007 | **never** |
| fto=5, split=4 | +46 ms | 1703 | 37 |
| fto=5, split=12 | +38 ms | 1840 | 48 |
| fto=5, split=20 | +30 ms | 2015 | 67 |
| fto=5, split=12, L=3 | +181 ms | 2717 | 15 |
| fto=7, split=4 | +173 ms | 2008 | 12 |
| fto=7, split=12, L=3 | +705 ms | 3150 | 4 |
| fto=9, split=20, L=3 | +1942 ms | 5408 | 3 |
| fto=11, split=20, L=3 | +6609 ms | 10086 | 2 |

The `fto=3` rows sit inside the 1.18x drift the probe measured, so read them as "no
difference worth compiling for" rather than as a sign.

At `fto=3` the modeling dominates; at `fto≥7` the eigen-solve does. `n_split` matters much
less than `fto`: it moves the freeze cost, not the solve.

### Why `jax eager` loses, and where it stops losing

Splitting a forward call into the modeling step (many small operations) and the solve (one
large linear algebra call) locates all of it:

| | modeling | solve | blocks |
|---|---|---|---|
| fto=3, split=4 — torch | 0.4 ms | 12.3 ms | 54 |
| fto=3, split=4 — jax | **4.0 ms** | 23.0 ms | 54 |
| fto=3, split=20 — torch | 2.8 ms | 13.4 ms | 902 |
| fto=3, split=20 — jax | **24.3 ms** | 22.0 ms | 902 |
| fto=11, split=20 — torch | 2.8 ms | 3512 ms | 902 |
| fto=11, split=20 — jax | 27.3 ms | **2543 ms** | 902 |

JAX is ~9x slower on the modeling step at every size, because JAX has no true eager mode:
every operation goes through its own dispatch, and the vector modeler emits one small array
operation per block — 902 of them at `n_split=20`. PyTorch's C++ dispatcher absorbs that;
JAX's does not. Taking a gradient widens it further (forward alone is 46 ms against 16; with
`jax.grad` it is 250 against 22), since each primitive is then traced through JVP and partial
evaluation.

The solve is the other way around: from `fto=7` up, JAX is *faster* there (2543 ms against
3512 at `fto=11`). So `jax eager` catching up at high `fto` is not the eager path improving —
it is the part where it loses shrinking to nothing as one 1250x1250 eigendecomposition comes
to dominate.

### Where the recompilation lands

Sweeping `lx` from 85 to 125 on a 4-slab ellipse, re-freezing at every point:

| lx | 85 | 90 | 95 | 100 | 103 | 106 | 110 | **115** | 120 | 125 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 1868 | 23 | 22 | 22 | 23 | 25 | 23 | **856** | 23 | 23 |

One compilation at the start, one when two block edges cross at `lx≈115`, and nothing in
between — a 47% move in the design variable for two compilations. That crossing is exactly
where a *held* record would have gone stale, so the recompilation and the staleness boundary
are the same event. Re-freezing every step converts a silent wrong answer into a sub-second pause.

---

## Raster path — the design variable is the *permittivity grid*

A binary raster, `d(sum de_ti)/d(ucell)`, swept across all three Fourier strategies.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| N=16, fto=3, DFS | 18 | 67 | 16 | 16 | 700 | 0 |
| N=16, fto=3, CFS | 18 | 70 | 16 | 626 | 917 | 0 |
| N=16, fto=3, NVF | 18 | 70 | 16 | 527 | 709 | 0 |
| N=32, fto=3, DFS | 19 | 66 | 15 | 15 | 695 | 0 |
| N=32, fto=3, CFS | 19 | 77 | 15 | 639 | 929 | 0 |
| N=32, fto=3, NVF | 20 | 73 | 17 | 561 | 683 | 0 |
| N=32, fto=5, DFS | 140 | 146 | 85 | 86 | 792 | 0 |
| N=32, fto=5, CFS | 138 | 150 | 84 | 733 | 975 | 0 |
| N=32, fto=5, NVF | 136 | 147 | 83 | 585 | 727 | 0 |
| N=64, fto=3, DFS | 18 | 67 | 15 | 15 | 688 | 0 |
| N=64, fto=3, CFS | 20 | 72 | 15 | 613 | 933 | 0 |
| N=64, fto=3, NVF | 21 | 73 | 16 | 547 | 727 | 1 |
| N=64, fto=5, DFS | 131 | 140 | 80 | 80 | 771 | 0 |
| N=64, fto=5, CFS | 138 | 153 | 132 | 1032 | 1050 | 0 |
| N=64, fto=5, NVF | 142 | 157 | 93 | 659 | 895 | 1 |
| N=64, fto=7, DFS | 641 | 559 | 416 | 425 | 1340 | 0 |
| N=64, fto=7, CFS | 613 | 550 | 399 | 1026 | 1322 | 0 |
| N=64, fto=7, NVF | 595 | 514 | 399 | 1006 | 1208 | 1 |
| N=128, fto=3, DFS | 19 | 70 | 16 | 16 | 718 | 0 |
| N=128, fto=3, CFS | 24 | 74 | 17 | 649 | 946 | 1 |
| N=128, fto=3, NVF | 21 | 79 | 18 | 608 | 838 | 1 |
| N=128, fto=5, DFS | 140 | 150 | 84 | 85 | 843 | 0 |
| N=128, fto=5, CFS | 143 | 157 | 87 | 799 | 996 | 0 |
| N=128, fto=5, NVF | 148 | 152 | 84 | 619 | 795 | 1 |

Four things this says.

**1. Re-freezing is ruinous for CFS and NVF.** Their records hold arrays, where value equality
is neither well defined nor hashable, so `StaticGeometry` falls back to identity and a new
freeze misses the compilation cache every time: ~600 ms against a ~16 ms step. The content-hash
path helps the vector records only. On these two strategies a fit must either hold one freeze
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

**4. Nothing here separates the strategies on cost any more.** DFS, CFS and NVF land within a
few percent of each other at every size. NVF used to be the outlier, growing with `N` where the
others were flat, because its distance transform ran on the `(4*fto+1)*N` upsampled grid; it no
longer upsamples, and its freeze fell from 359 ms to 1 ms at `N=128, fto=5`. So the choice among
them is now about accuracy, which is what it should have been.

---

## If a fit must hold one freeze — how far it stays valid

Re-freezing every step removes the question on the vector path, but the raster records give no
choice: their identity comparison makes a re-freeze a recompilation, so a jitted raster fit
holds one. The width of that record's validity was bisected against
`_check_plan_still_fits` — the boundary is sharp, exact up to it (a replay at 106.31 matches a
rebuild at 0.0e+00) and refused past it (106.33).

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

**`n_split` governs how long a freeze lives, and it is not a gentle dependence.** More slabs
means more block edges, and more edges means more chances for two of them to cross: the
rotated rectangle's angle window shrinks 96x from 1 slab to 20, and the ellipse's `lx` window
7.5x from 3 slabs to 20. At `n_split=20` a held record survives roughly ±1% of the nominal
angle, which is narrower than any OCD fit's search range.

That is an unfortunate pairing, since `n_split` is raised precisely to resolve curved or
tilted sidewalls — the fits most likely to want it are the ones a held freeze serves worst.

Two windows are unremarkable and worth knowing anyway. An **axis-aligned rectangle's `lx` never
goes stale**: its only edges are `cx ± lx/2`, which cannot reorder. And the windows are
**asymmetric** (`n_split=20` ellipse: −12.4% / +1.6%), so a record is better taken slightly
above the expected convergence point than at the centre of the search range.

## What to use

| situation | use |
|---|---|
| OCD fit, `fto ≤ 5` | **torch** |
| OCD fit, `fto ≥ 7`, long run | **jax jit, re-freezing every step** |
| Raster inverse design, pattern moves every step | **`('sampled','li')`**, any backend |
| Raster with CFS/NVF, pattern moves | **torch** (jit's re-freeze recompiles every step) |
| Raster with CFS/NVF, pattern fixed | jax jit, one freeze |
| Anything where a stale record would be silent | **torch**, or re-freeze on the vector path |

The general shape: **PyTorch is the default, and JAX earns its place at large `fto` on the
vector path.** Where JAX is chosen there, re-freezing every step is now both the safe option
and, within the freeze's own cost, the fast one.

---

## Caveats

- **CPU only.** Every number here was measured on one CPU host, and none of it has been run on
  a GPU. Do not extrapolate: the per-step times and the compilation cost need not move in the
  same direction, so neither the ratios nor the break-even table carries over on its own.
- **PyTorch was measured eager, which is the only way it runs today.** `torch.compile` fails
  on every FieldWave path before torch 2.13, and once it works it measures 0.85–1.02x — see
  `docs/torch-compile.md`, which also profiles why (97.9% of the runtime at `fto=10` is inside
  LAPACK/BLAS calls Inductor forwards verbatim). So `torch` here means eager torch, and that is
  not a limitation of the measurement.
- **Single process, no batching.** Nothing here says how the modes behave under `vmap`, across
  a wavelength sweep, or on several devices — and a wavelength sweep is the case where one
  compilation is amortised over many solves, which would favour jit well beyond what is
  measured here.
- **Compile times are single observations** and vary by tens of percent between runs. The
  step times are medians of 3 after a warm-up.
- The break-even tables assume a fit that holds its discrete description. A fit that reorders
  block edges every step pays the compile every step, and no amount of `fto` rescues that.

Reproduce with `python benchmarks/benchmark_backends.py`. The run this document reports is
checked in as `benchmarks/results/backends.log`; a run also writes machine-readable numbers to
`benchmarks/results/backends.json`, which the repository does not track (`*.json` is ignored).
