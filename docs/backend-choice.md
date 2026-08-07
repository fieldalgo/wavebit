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
- **NVF is the expensive one**, and increasingly so with raster size: at `N=128, fto=5` its
  freeze alone costs 359 ms against a 277 ms step.

---

## Vector path — the design variable is a *length*

An ellipse stack, `d(sum de_ti)/d(lx)`. This is what the closed-form series exists for: the
segment edges enter the Fourier transform analytically, so the gradient reaches CD, sidewall
angle and pitch. Times are ms per step.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| fto=3, split=4, L=1 | **18** | 108 | 16 | 21 | 1544 | 4 |
| fto=3, split=12, L=1 | **21** | 171 | 16 | 28 | 1764 | 12 |
| fto=3, split=20, L=1 | **22** | 250 | 16 | 40 | 1846 | 23 |
| fto=5, split=4, L=1 | 118 | 177 | 80 | **84** | 1532 | 4 |
| fto=5, split=12, L=1 | 124 | 236 | 79 | **91** | 1731 | 11 |
| fto=5, split=20, L=1 | 127 | 316 | 79 | **103** | 1924 | 23 |
| fto=5, split=12, L=3 | 316 | 633 | 120 | **154** | 2571 | 35 |
| fto=7, split=4, L=1 | 518 | 496 | 354 | **362** | 1844 | 4 |
| fto=7, split=12, L=1 | 526 | 552 | 358 | **368** | 2044 | 12 |
| fto=7, split=20, L=1 | 535 | 640 | 355 | **379** | 2202 | 24 |
| fto=7, split=12, L=3 | 1279 | 1491 | 588 | **624** | 3131 | 35 |
| fto=7, split=20, L=3 | 1327 | 1730 | 574 | **640** | 3255 | 69 |
| fto=9, split=12, L=1 | 1715 | 1675 | 1311 | **1312** | 3092 | 11 |
| fto=9, split=20, L=1 | 1707 | 1755 | 1323 | **1349** | 3341 | 23 |
| fto=9, split=20, L=3 | 4203 | 4672 | 2233 | **2366** | 5039 | 71 |
| fto=11, split=20, L=3 | 12332 | 12015 | 6907 | **6618** | 9392 | 68 |

Repeating the whole sweep reproduced every figure to within a few percent, which is the
resolution these numbers deserve.

### Re-freezing every step costs almost nothing here

`jit (re-freeze)` retakes the freeze on every step — the only shape of loop that cannot carry a
stale record forward. It tracks `jit (one freeze)` within the freeze's own cost, because the
vector record compares **by content**: it is which block edge becomes which grid line, in what
order, and the edge *coordinates* stay traced. A length can move a long way without the
ordering changing, and an unchanged ordering reuses the compiled program.

The gap between the two columns is the freeze itself (4-68 ms), which grows with `n_split` and
layer count and is otherwise flat.

### When compilation pays for itself

Break-even against PyTorch, counting the compile a fit must amortise:

| config | torch − jit(re-freeze) | compile | steps to break even |
|---|---|---|---|
| fto=3, split=4 | −3 ms | 1544 | **never** |
| fto=3, split=12 | −7 ms | 1764 | **never** |
| fto=3, split=20 | −18 ms | 1846 | **never** |
| fto=5, split=4 | +34 ms | 1532 | 45 |
| fto=5, split=12 | +33 ms | 1731 | 52 |
| fto=5, split=20 | +24 ms | 1924 | 80 |
| fto=5, split=12, L=3 | +162 ms | 2571 | 16 |
| fto=7, split=4 | +156 ms | 1844 | 12 |
| fto=7, split=12, L=3 | +655 ms | 3131 | 5 |
| fto=9, split=20, L=3 | +1837 ms | 5039 | 3 |
| fto=11, split=20, L=3 | +5714 ms | 9392 | 2 |

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
| ms | 1588 | 22 | 23 | 22 | 22 | 21 | 21 | **766** | 22 | 22 |

One compilation at the start, one when two block edges cross at `lx≈115`, and nothing in
between — a 47% move in the design variable for two compilations. That crossing is exactly
where a *held* record would have gone stale, so the recompilation and the staleness boundary
are the same event. Re-freezing every step converts a silent wrong answer into a 700 ms pause.

---

## Raster path — the design variable is the *permittivity grid*

A binary raster, `d(sum de_ti)/d(ucell)`, swept across all three Fourier strategies.

| config | torch | jax eager | jit (one freeze) | jit (re-freeze) | compile | freeze |
|---|---|---|---|---|---|---|
| N=16, fto=3, DFS | 19 | 67 | 16 | 16 | 747 | 0 |
| N=16, fto=3, CFS | 19 | 79 | 16 | 612 | 895 | 0 |
| N=16, fto=3, NVF | 22 | 79 | 16 | 569 | 750 | 3 |
| N=32, fto=3, DFS | 17 | 67 | 15 | 16 | 679 | 0 |
| N=32, fto=3, CFS | 19 | 69 | 15 | 626 | 836 | 0 |
| N=32, fto=3, NVF | 37 | 87 | 19 | 571 | 863 | 10 |
| N=32, fto=5, DFS | 142 | 141 | 82 | 82 | 709 | 0 |
| N=32, fto=5, CFS | 135 | 144 | 82 | 648 | 907 | 0 |
| N=32, fto=5, NVF | 182 | 177 | 90 | 684 | 1023 | 24 |
| N=64, fto=3, DFS | 20 | 66 | 15 | 16 | 706 | 0 |
| N=64, fto=3, CFS | 21 | 72 | 15 | 625 | 885 | 0 |
| N=64, fto=3, NVF | 95 | 127 | 31 | 739 | 922 | 38 |
| N=64, fto=5, DFS | 130 | 138 | 80 | 80 | 736 | 0 |
| N=64, fto=5, CFS | 140 | 143 | 83 | 672 | 967 | 0 |
| N=64, fto=5, NVF | 320 | 294 | 134 | 1162 | 1367 | 91 |
| N=64, fto=7, DFS | 546 | 480 | 380 | 383 | 1071 | 0 |
| N=64, fto=7, CFS | 593 | 528 | 390 | 993 | 1439 | 0 |
| N=64, fto=7, NVF | 1011 | 888 | 493 | 1976 | 2330 | 172 |
| N=128, fto=3, DFS | 19 | 69 | 16 | 16 | 682 | 0 |
| N=128, fto=3, CFS | 22 | 71 | 16 | 624 | 906 | 0 |
| N=128, fto=3, NVF | 337 | 326 | 87 | 1424 | 1597 | 157 |
| N=128, fto=5, DFS | 133 | 137 | 82 | 83 | 723 | 0 |
| N=128, fto=5, CFS | 133 | 145 | 83 | 676 | 973 | 1 |
| N=128, fto=5, NVF | 978 | 815 | 298 | 2944 | 3169 | 371 |

Four things this says.

**1. Re-freezing is ruinous for CFS and NVF.** Their records hold arrays, where value equality
is neither well defined nor hashable, so `StaticGeometry` falls back to identity and a new
freeze misses the compilation cache every time: 583 ms against a 16 ms step. The content-hash
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
from the *base* raster instead of reading it off the upsampled one. Since the upsample is a
block repeat the two are the same mask, and the base has `repeat_y * repeat_x` fewer values to
narrow. So `geometry=` is optional for a gradient on every raster path, which is what makes a
fit whose pattern moves every step safe.

**4. NVF's cost grows with the raster, the others' do not.** DFS and CFS are flat in `N` (the
convolution matrix depends on `fto`, not `N`); NVF is not, because the distance transform runs
on the `(4·fto+1)·N` upsampled grid. At `N=128, fto=5` that is 2688² cells: a 359 ms freeze
against a 277 ms step.

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
