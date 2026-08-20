# What a solve answers — and why the field state is off by default

Written: 2026-08-20. Measured on an Apple M4 (16 GB), NumPy 2.5.1 against Accelerate, float64,
single thread as configured. Retention is counted by walking the retained objects and charging
each array to its owning buffer, so views are counted once. Peak comes from `tracemalloc`,
which tracks NumPy's own allocator exactly. Times are best-of-N after a warm-up. Neither
memory number comes from a post-solve RSS delta, and neither comes from `ru_maxrss` — §4 says
what each of those two instruments got wrong here.

Targets:
- `wavebit/_core/enums.py` — `Answers`, `_ANSWERS_LADDER`, `covers`
- `wavebit/_core/engine.py` — the `answers` property, `_require_field_declaration`, `_finish`
- `wavebit/_core/solve_1d.py`, `solve_2d.py`, `solve_2d_4ff.py` — the ETM per-layer appends
- `wavebit/_core/scattering.py` — the SMM cascade and its `_field_state` recovery pass

**TL;DR.** `answers` declares at construction what a solve keeps the state to answer:
`'efficiency'`, `'amplitude'` (the default), or `'field'`. The seam that matters is between the
last two, and it is worth a keyword because they differ by three orders of magnitude: the
amplitudes are boundary vectors already solved for, while the field-reconstruction state is
**4.25 matrices of side `2*ff_x*ff_y` per layer** on the 2D path — 240 MB per layer at
`fto=(15, 15)`, 733 MB at `(20, 20)`. Every public route used to keep it unconditionally, and
most large runs are optimization sweeps against a fixed objective that never reconstruct a
field at all.

## 1. The ladder

| rung | adds | costs |
| --- | --- | --- |
| `'efficiency'` | `R`, `T` per order | — |
| `'amplitude'` (default) | `r_s`/`r_p`/`t_s`/`t_p`, their s/p-resolved efficiencies, `res_te_inc` / `res_tm_inc` | one extra solve against an already-factorized matrix |
| `'field'` | the per-layer modes and amplitudes `reconstruct_field` walks | §2 |

It is a ladder and not a menu: each rung answers everything below it, and `Answers.covers`
reads that order out of `_ANSWERS_LADDER` so nothing else has to know it. A lower rung is
therefore always a *smaller* answer and never a different one — checked bit-for-bit rather
than to a tolerance, in `tests/test_answers_declaration.py`.

The two seams are not equally interesting, and it is worth saying which is which. The first is
almost free and exists so a caller can say what they mean; the second is the whole reason the
keyword exists.

## 2. What the field state costs

A layer's entry is `[epz_conv_i, W, V, q, d, a_i, b]`. With `dim = 2*ff_x*ff_y` on the coupled
paths, `W`, `V`, `a_i` and `b` are each a full `dim x dim` block and `epz_conv_i` is a quarter
of one (it is indexed by orders, not by the doubled s/p state), so:

| path | blocks of `dim^2` per layer | why |
| --- | --- | --- |
| 2D (`solve_2d`) | **4.25** | `layer_modes_2d` returns a full `dim x dim` `W` |
| conical (`solve_1d_conical`) | 3.75 | `W = [[W1, W2]]` is `dim/2 x dim`, half the 2D one |
| pure 1D TE/TM | 5.0 | `dim = ff_x`, and all five matrices are square in it |

Read off the retained shapes rather than assumed, at `fto=6`: the pure path holds five
`13 x 13` matrices, the conical one `(13,13) (13,26) (26,26) (26,26) (26,26)`, and the 2D one
`(169,169) (338,338) (338,338) (338,338) (338,338)`. Measured totals at that size are 5.15,
3.90 and 4.26 blocks — above the asymptote by the `q` vector and `T1`, which stop mattering as
`dim` grows (4.25 to three figures by `fto=(8, 8)`).

Measured, one patterned layer, ETM, 2D coupled:

| `fto` | `dim` | one block | retained | buffers |
| --- | --- | --- | --- | --- |
| `(8, 8)` | 578 | 5.1 MB | 21.70 MB | 8 |
| `(15, 15)` | 1922 | 56.4 MB | 239.68 MB | 8 |
| `(20, 20)` | 3362 | 172.7 MB | 733.21 MB | 8 |

and linear in depth — 4.25 blocks *per layer*, so at `fto=(8, 8)`: 21.70 MB at one layer,
43.37 at two, 86.72 at four, 173.42 at eight. On the 16 GB machine these were measured on, a
four-layer stack at `fto=(20, 20)` retains 2.9 GB after a solve that is already finished.

The 4.25 figure is worth stating because 3.75 is the number you get by assuming the 2D path
shares the conical path's `W`. It does not, and the difference is 13% on exactly the case —
a 2D metasurface at `fto=(15, 15)` or `(20, 20)` — where the absolute number is largest.

## 3. Why the amplitudes are on the near side of the seam

Because a phase-target optimization needs them and does not need a field, and the old `full`
flag made it buy both. `full=False` returned `(de_ri, de_ti)` and nothing else: no Fresnel
amplitudes, and no `res_te_inc` / `res_tm_inc`, which the unpolarized average is built from.

The two are also ordered, not independent, and the ordering is forced by the code rather than
chosen: `final_matching` builds `T1` — the thing field reconstruction starts from — *out of*
the pure-TE/TM incidence columns. A solve that kept `T1` while dropping `res_te_inc` would be
doing the extra solve and then discarding the cheap half of its own result. So `'field'`
implies `'amplitude'`, and the ladder is monotone for a structural reason.

## 4. It is a lifetime change at one layer, and a peak change from two up

Peak memory for a **single-layer** solve is unchanged by the declaration — measured at
**+0.01 MB**, which is the noise floor of the instrument. That is not a defect; it is what
"retention" means. The matrices are built either way, because they are the loop's own working
set, and the declaration only decides whether anything still points at them afterwards.

From two layers up it is also a peak change, and the mechanism is the *accumulation* rather
than the per-iteration locals. With the append in place, live memory grows as `k * 4.25`
blocks across `k` layers; without it each layer's matrices become garbage as soon as the next
iteration rebinds the names, and only `F`, `G` and `T` — the ETM recursion state — carry
across. So the appends are conditional, not just the returns.

Measured with `tracemalloc`, `fto=(8, 8)`, one process, ETM:

| layers | peak, `'amplitude'` | peak, `'field'` | peak delta |
| --- | --- | --- | --- |
| 1 | 172.14 MB | 172.14 MB | **+0.01 MB** |
| 2 | 175.98 MB | 196.38 MB | +20.40 MB |
| 4 | 183.67 MB | 244.87 MB | +61.20 MB |
| 8 | 199.03 MB | 341.83 MB | +142.80 MB |

(What each of those retains is §2's column, and it is the larger number in every row: 21.70 MB
per layer against 20.40 MB of peak.)

The peak delta is exactly `(layers - 1) * 20.40 MB`, and 20.40 MB is exactly **4.0** blocks of
`578^2` — not the 4.25 the same solve *retains*. The missing quarter is `epz_conv_i`: the
entry holds a reference to a matrix the solve was already handed in `epz_conv_i_all`, so it is
retained (it keeps that stack alive) without ever being new memory. Retention counts it; peak
cannot see it.

**Do not measure any of this with a post-solve RSS delta, and not with `ru_maxrss` either.**
The RSS delta is at the allocator's discretion. `ru_maxrss` is a true high-water mark but
carries several hundred MB of process noise here — run across fresh processes it reported
`'amplitude'` peaking *above* `'field'` at `fto=(8, 8)`, in the direction opposite to the
effect, at both 1 and 6 layers. `tracemalloc` tracks NumPy's own allocator exactly (a
2000x2000 complex array reports 61.04 MiB) and is the instrument these rows come from.

## 5. What it costs in time

Only on the SMM, and only at the top rung. Measured at `fto=(10, 10)`, 4 layers, best of 3
after a warm-up:

| | `'amplitude'` | `'field'` | |
| --- | --- | --- | --- |
| ETM | 4.396 s | 4.385 s | nil — the state is a by-product of the recursion |
| SMM | 5.928 s | 6.803 s | **+14.8%** — `_field_state` is a genuine second pass |

The SMM figure is worth having because the module docstring has always claimed the recovery is
where "the extra ~10-20%" of SMM's cost goes, and that was a quoted number rather than a
measured one. It measures at 14.8%, inside the claimed band. So on the SMM, declaring anything
below `'field'` buys time and not only memory; on the ETM it buys memory alone.

The `'efficiency'` rung buys **no time at all**: 1.283 s against 1.284 s on the ETM and 1.935 s
against 1.923 s on the SMM, at `fto=(10, 10)`, one layer, best of 5 — differences of one part
in a thousand, which is noise. Re-measured 2026-08-20 at `fto=(8, 8)` (0.536 s against 0.524 s)
and `(15, 15)` (17.0 s against 17.4 s), same answer. It is not a speed knob and never was: what
it skips is one extra linear solve on an already-factorized `final_A`, plus the Fresnel packing,
against a cost the eigendecomposition owns.

What it *does* buy is **the size of the result**, which the timing above cannot see and which
an earlier draft of this section wrongly reported as nothing. Measured on the same scenes:

| | result payload | JAX pytree |
| --- | --- | --- |
| `'efficiency'` | 4.5 KB at `fto=(8, 8)`, 15.0 KB at `(15, 15)` | **2 leaves** |
| `'amplitude'` | 94.8 KB, 315.3 KB | **30 leaves** |

A factor of 21, because the amplitude rung carries ten per-order arrays across three incidence
variants where this one carries two. Whether that matters is exactly the kind of thing the
caller knows and the library does not: an optimisation loop takes `.T.sum()` and drops the rest,
so it buys that caller nothing — while a 200-point spectrum sweep that keeps its results is
0.5 MB against 10.8 MB, and under `vmap` the wider tree is materialised per batch element
whether anything reads it or not.

So the ladder's bottom rung is a real cut, just not on the axis the rest of this document is
about. The lesson is worth more than the rung: **"buys nothing" was a measurement of one axis
reported as a verdict**, and it nearly cost the rung its place.

## 6. Why a declaration and not an argument

Three reasons, in increasing order of how much trouble the alternative causes.

It is knowable when the problem is set up. "Am I going to look at a near field?" is a property
of the job, not of one call — which is the argument `tetm` already makes one level over, and
`answers` is deliberately the same shape of thing: a statement that a question is *not* being
asked, so the machinery for answering it is never built.

It has to be static under `jit`. The core branches on it — which columns enter the cascade,
whether the per-layer appends happen — and a traced value cannot be branched on.

And it must survive the JAX pytree round trip, which is where the interesting failure is. As a
plain instance attribute nothing puts it in `aux_data`, so `_tree_unflatten` never passes it to
the constructor and the rebuilt solver takes the default. Exercised on the real flatten /
unflatten pair, `fto=(3, 3)`, 2 layers:

| `aux_data` carries it | declared | rebuilt reports | retained after re-solve |
| --- | --- | --- | --- |
| yes (as shipped) | `'amplitude'` | `'amplitude'` | 0.000 MB |
| yes (as shipped) | `'field'` | `'field'` | 1.253 MB |
| no (plain attribute) | `'amplitude'` | `'amplitude'` | 0.000 MB |
| no (plain attribute) | `'field'` | **`'amplitude'`** | **0.000 MB** |

The last row is the whole argument: a solver that declared a field comes back without one, and
silently. Worse, with the key absent the two declarations produce **identical `aux_data`**, so
their treedefs coincide and one `jit` compilation serves both — the declaration cannot even be
recovered by recompiling. It works eagerly and under torch, including `torch.compile`, and
fails only across a `jit` boundary, which is the least likely place to notice it.
`tests/test_answers_declaration.py` pins all three halves: the round trip, that the two
declarations produce *different* treedefs, and that the numbers agree across the boundary.

## 7. Under autograd it is worth several times the arrays

A retained tensor anchors its `grad_fn`, and through it everything the graph reaches. PyTorch
eager, `fto=(6, 6)`, 2 layers, `requires_grad` on the permittivity `structure`:

| | retained tensors | arrays | graph nodes | pinned by the graph | total |
| --- | --- | --- | --- | --- | --- |
| `'amplitude'` | 0 | 0.00 MB | 0 | 0.00 MB | 0.00 MB |
| `'field'` | 15 | 14.84 MB | 407 | 72.17 MB | **87.02 MB** |

So the state costs **5.9x** its own arrays once autograd is holding the graph behind it, and
declaring it away drops the graph with it rather than merely the matrices. This is the
optimization case — the one where the run is long, the solve is repeated, and nobody is going
to call `reconstruct_field`.

## 8. What is not on the ladder yet

A fourth rung, `'scattering'`, belongs above `'field'`: the SMM cascade builds `S`, the
per-layer `S_layer`, the below-plane reflections `gamma` and the gap↔layer interface
coefficients `AB`, uses them, and drops them — the same "computed and thrown away" shape the
field state had before this. Adding it is appending one member to `Answers` and one to
`_ANSWERS_LADDER`; nothing else reads the order, which is the point of keeping the order in
one tuple.

Two constraints belong with that rung rather than with this document, so they are written
where the next person will be standing when they add it — beside `_ANSWERS_LADDER` in
`wavebit/_core/enums.py`. In short: `AB` has to ship *with* the blocks, because the blocks are
in the gap basis and `_ab_matrices` is the only thing that decodes them; and `S_ref` / `S_trn`
should not ship at all.
