# Memory-bounded reverse mode on JAX and Torch

Reverse-mode differentiation of an FDTD run normally keeps every time step's intermediates, so
a gradient's memory grows with the step count. `Numerics(checkpoint_steps=K)` trades that
memory for recomputation. Before this record the policy was admitted only on the Torch CUDA
route, with one level of blocks. It is now admitted on JAX and Torch, on CPU and on the Torch
CUDA route, with a nested schedule whose memory grows with the logarithm of the step count.

Tests: `tests/fdtd/test_checkpoint_memory.py`, and the CPU `cpu-k3-nested` cases of
`tests/fdtd/test_general_ade_ad.py` and `tests/fdtd/test_general_ade_record_ad.py`.

## Acceptance, registered before computing

- A checkpointed gradient matches the ordinary scan's to `1e-12` relative in float64, on
  JAX and on Torch CPU, and matches a central difference of a NumPy forward run.
- Peak gradient memory is set by the segment length and not by the step count, shown as a
  memory ratio across two run lengths. No timing is reported.
- Existing goldens are unmoved; no `tests/references/fd_golden_stage*.npz` was regenerated.
- Every new test is checked by deliberately breaking the product code.

## The schedule

`K` is the segment length. A run of at most `K` steps is an ordinary scan. A longer run is cut
into segments of `unit` steps, where `unit` is the largest power of `K` below the run length;
each segment is checkpointed and scheduled the same way one level down, and the remainder
follows, also scheduled the same way. A run of `K**L` steps therefore stores at most about
`L * K` states plus the intermediates of one `K`-step scan, and the forward sweep runs `L + 1`
times. The previous single-level schedule stored `steps / K` states.

This is the uniform multi-level schedule, not the optimal binomial schedule of
A. Griewank and A. Walther, "Algorithm 799: Revolve: an implementation of checkpointing for
the reverse or adjoint mode of computational differentiation", *ACM Transactions on
Mathematical Software* 26(1), 19-45 (2000). Revolve reaches `C(s + r, s)` steps with `s`
stored states and `r` recomputations. The uniform schedule reaches `K**L` steps with
`s = L * K` and `r = L`, which lies inside that bound. It was chosen because each level is one
fixed-length scan, so JAX traces one body per level instead of one per step of a revolve
schedule. `K = 1` makes every step its own segment on one level.

On JAX each segment is a `jax.checkpoint` around a `lax.scan`. On Torch each segment is one
custom `autograd.Function` that keeps only its input state and recomputes itself in backward.

## Why Torch needed its own segment

The first Torch implementation nested `torch.utils.checkpoint` (non-reentrant). Its gradients
were correct but its memory was not bounded. Non-reentrant checkpointing drops each step's
saved tensors but keeps each step's autograd graph nodes. On the forward pass alone, the
graph grew by about 80 kB per step at nearly any grid size: 79 MB on a 10 x 10 x 48 grid and
97 MB on a 20 x 20 x 48 grid after 1,024 steps. A 12,000-step run would hold about a gigabyte
of graph before the backward pass starts.

A `Segment` is one graph node. The graph inside it exists only while that segment's own
backward pass runs. Because it is an `autograd.Function`, every live tensor that the step reads
must be one of its inputs. A tensor that reached the step only through a closure would get no
gradient. The executor therefore passes the run's parameters to `scan` as an argument
(`params=`), and a checkpointed Torch scan without them raises an error.

Under `create_graph=True` the recompute starts from a fresh view of each saved input. Starting
from the input itself makes `autograd.grad` follow that input back through the earlier
segments and return the total derivative instead of this segment's own. A 17-step toy scan
with `K = 3` showed this: its gradient came out 6x too large. The exact-Hessian tests below
cover this case.

## Gradients

Scene: a `0.05 x 0.05 x 2.4` µm cell, periodic in x and y and open in z, with a `SnappedSlab`
film of `ε = 2.25` and a `PlaneWave` Gaussian pulse (0.8 µm, `σ_f = 4e13` Hz). The grid is
0.025 µm and the run is 800 steps. The loss sums a `TimeSeriesMonitor`, a `FieldMonitor`, a
`FluxMonitor` and the final fields. Gradients are taken with respect to the film permittivity
and the source amplitude.

| block `K` | schedule at 800 steps | JAX | Torch CPU |
| --- | --- | --- | --- |
| 1 | 800 one-step segments | ≤ 1e-12 | ≤ 1e-12 |
| 7 | three levels with nested remainders (343, 49, 7) | ≤ 1e-12 | ≤ 1e-12 |
| 30 | one level, 26 segments and a 20-step remainder | ≤ 1e-12 | ≤ 1e-12 |
| 1000 | longer than the run: an ordinary scan | ≤ 1e-12 | ≤ 1e-12 |

Each entry compares the loss and both derivatives with the ordinary scan at `rtol = 1e-12`,
`atol = 0`. The largest relative difference observed during development was 7e-16. With
`K = 7`, the loss matches a NumPy forward run at `1e-12`, the permittivity derivative
matches a central difference at `1e-6` (step `2.25e-5`), and the amplitude derivative equals
`2 L / a` at `1e-12`, since every term of the loss is quadratic in the amplitude.

Second derivatives were checked on Torch CPU. The existing exact-Hessian tests for
homogeneous Lorentz and Drude media (17 steps, closed-form references at `2e-12`) now include
a CPU case with `K = 3`. That case uses a nested segment of 9 steps with remainders of 6 and
2, for both the final-field loss and the record loss.

A traced ranged body is covered too. With a `Sphere` radius as the differentiated input
(`Numerics(geometry_ranges=...)`, 120 steps, eager `torch.autograd` and `jax.grad`), `K = 7`
and `K = 1000` reproduce the plain scan's loss exactly and its radius derivative within
`1e-12` (observed `1.3e-15`); the plain derivative matches a NumPy central difference
(step `1e-4`) within `1e-4` (observed `6e-7`). The traced-geometry preflight compares the
prepared plan with one rebuilt from the concrete scene, and the checkpoint block is part of
the plan's skeleton, so the rebuilt plan takes the block from the prepared one; any other
difference is still refused (`tests/fdtd/test_ranged_shape_checkpoint_grad.py`).

## Memory

Two run lengths, 128 and 1,024 steps (8x), with `K = 8` and without checkpointing. The
scene is the same film in a `0.5 x 0.5` cell (10 x 10 x 48 cells) on JAX and a `1 x 1` cell
(20 x 20 x 48) on Torch, both at 0.05 µm with `σ_f = 2e13` Hz. JAX is measured by XLA's
scratch allocation for the whole jitted gradient (`compiled.memory_analysis()`), which is a
property of the compiled program. Torch is measured by the rise of the process's peak
resident memory during one gradient, in a fresh process, after a forward run of the same
prepared plan.

On Linux the process peak (`ru_maxrss`) covers the whole process, so a gradient smaller than
the imports or the first forward run reads as no rise at all. This happened on the CI
runner. glibc also keeps freed heap pages resident and reuses them. The Linux measurement
therefore resets the peak just before the gradient (writing `5` to `/proc/self/clear_refs`
resets `VmHWM`), after `malloc_trim` and with `MALLOC_MMAP_THRESHOLD_=65536`, so that large
tensors are mapped and unmapped rather than recycled. It reads the rise as `VmHWM` minus the
resident size at the reset. A rise of zero fails the test as a blind measurement. On WSL2
Ubuntu, Torch 2.13 CPU, the ratios were 1.24 with `K = 8` and 6.91 without checkpointing.
The table below is the Windows measurement.

| backend | `checkpoint_steps` | 128 steps | 1,024 steps | ratio |
| --- | --- | --- | --- | --- |
| JAX (XLA scratch) | 8 | 6.5 MB | 9.8 MB | 1.50 |
| JAX (XLA scratch) | None | 22.1 MB | 169.8 MB | 7.69 |
| Torch CPU (peak RSS rise) | 8 | 69.0 MB | 98.2 MB | 1.42 |
| Torch CPU (peak RSS rise) | None | 119.2 MB | 917.9 MB | 7.70 |

With checkpointing, eight times more steps cost one more level, about 1.5x the memory. The
tests require a checkpointed ratio below 2 and an ordinary ratio of at least 4. The second
requirement is a control: it shows that the measurement can fail.

## Mutations

Each mutant was applied to the product code alone, and the named tests were run against it.

| mutant | caught by |
| --- | --- |
| single-level schedule (`unit = K` always) | both memory tests |
| JAX segment without `jax.checkpoint` | JAX memory test |
| Torch `Segment` replaced by nested `torch.utils.checkpoint` | Torch memory test |
| higher-order recompute from the saved inputs themselves | CPU `cpu-k3-nested` Hessians |
| parameters not passed as `Segment` inputs | Torch gradient parity |
| Torch remainder starting one step early | Torch gradient parity |
| JAX remainder starting one step early | JAX gradient parity |
| segment input not copied before the step writes in place | Torch gradient parity, Hessians |
| JAX and Torch CPU refused again at `numerics.checkpoint_scope` | central-difference tests |
| checkpointed Torch scan run without its parameters | the refusal test |

## Limits

- This is a memory/compute trade-off and not a memory cap. Each level adds about `K` stored
  states and one forward sweep.
- Observation and source histories are part of every stored state. A `TimeSeriesMonitor` or
  `Record` therefore adds its own history once per stored state. The bound applies to the
  fields, absorber and material memories.
- NumPy has no reverse mode and is refused; so is `DecayStop`.
- The JAX runtime is device-neutral, but the public API admits JAX only on CPU. JAX on GPU was
  not measured here.
- The Torch CUDA route uses the same `Segment`. Its CUDA tests (checkpoint blocks 1, 3, 7, 11
  and 64, with Hessians) need a CUDA device and were not rerun for this record. Their CPU
  counterparts above pass.
- The periodic-stack solver (`wavebit.fdtd.periodic`) keeps its own single-level block
  schedule and is not changed here.
