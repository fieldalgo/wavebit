# Outer-jit memory of area-painted FDTD derivatives

Written: 2026-10-19.

Targets: `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/runtime.py`,
`studies/cross_solver/painted_shapes/fdtd_painted_shapes.py`, `tests/fdtd/test_fdtd_painted_optics.py`.

**TL;DR.** The extra memory comes from how XLA:CPU orders the compiled program. The time
loop and the size of the program are not the cause. In
`_painted_cell_materials`, the area-integrated shape painting builds many
`[edges, images, ny, nx, segments, gauss]` float64 intermediates, and each feeds a reduction that
XLA:CPU puts in a YNNPACK fusion. A fusion's inputs must exist as whole buffers before it runs.
XLA:CPU's default scheduler in jaxlib 0.11.0 is the concurrency-optimized one, and it computes
every painting intermediate in the program first and runs the reductions last. So all of them are
alive at the same moment. Forward-mode differentiation multiplies the number of those
intermediates: 297 for the primal, 1,071 for one JVP column at resolution 6. Eager execution
dispatches one operation at a time and never holds more than one painting's working set. With
`xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`, the same program reports a 17.5×
smaller XLA temp allocation for `jit(value, jacfwd)` at resolution 20 (11.67 GB → 0.67 GB). Its
peak resident memory is 2.44 GB, against 13.68 GB recorded with the default scheduler, and its
Jacobian matches the eager one to 1.1e-15. Public `wavebit.fdtd.periodic.run(..., backend='jax')`
users with `shape_integration='area'` are affected in the same way; the default `'separable'`
route is not.

## Configuration

Measured on 2026-10-19 at `3f2ad83`: a Linux x86-64 cloud container (Intel Xeon @ 2.80 GHz,
4 vCPU, 15 GiB), Python 3.13.16, NumPy 2.5.4, JAX/jaxlib 0.11.0, float64 throughout. The
workload is `fdtd_painted_shapes.response(p, resolution)` with `p = [.37, .055, .22]` and its
default duration 60. Each row is a separate process, and the processes ran one at a time. Peak is
`ru_maxrss`. *Temp* is `jax.jit(g).lower(p).compile().memory_analysis().temp_size_in_bytes`,
which XLA:CPU allocates as one buffer per execution. The memory-optimized rows set only
`XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`. `jac` is
`jit(lambda p: (f(p), jax.jacfwd(f)(p)))`, the form the study and the slow test used, and `jvp`
is one forward column in the angle.

## Temp size follows the schedule, not the program

| Resolution | Case | Eager peak | jit default: temp / peak | jit memory-optimized: temp / peak |
|---:|---|---:|---:|---:|
| 6 | primal | 1.31 GB | 0.28 / 1.03 GB | 0.015 / 0.80 GB |
| 6 | jvp | 1.42 GB | 1.03 / 2.40 GB | 0.042 / 1.49 GB |
| 6 | jac | 2.37 GB | 1.33 / 2.93 GB | 0.076 / 1.72 GB |
| 8 | jvp | 1.47 GB | 1.61 / 2.92 GB | 0.065 / 1.43 GB |
| 8 | jac | 2.41 GB | 2.08 / 3.76 GB | 0.117 / 1.82 GB |
| 10 | jvp | 1.42 GB | 2.32 / 3.62 GB | 0.092 / 1.47 GB |
| 10 | jac | 2.49 GB | 3.00 / 4.68 GB | 0.166 / 1.90 GB |
| 20 | jac | 2.67 GB | 11.67 GB / not run | 0.67 / 2.44 GB |

Both schedulers compile the same HLO module, so the compile peak before execution is the same
(1.3–1.6 GB for the derivative programs). Each jit peak is about that compile peak plus the temp.
The temp grows roughly with `nx·ny` (4×4 cells at resolution 6; the resolution-20 temp is 8.8× the resolution-6 one). The resolution-20
default temp plus the compile peak, about 13.3 GB, accounts for the 13.68 GB recorded before
this note. The default resolution-20 program was compiled but not run here, because it would not
fit on this machine. The 22.9 GB single allocation that the slow test's docstring reports for
resolution 30 is consistent with this temp buffer. Values do not depend on the scheduler: at resolutions 6–10
the default and memory-optimized runs give identical output sums to the last digit, and every jit result agrees with eager
to about 1e-15. Run times were equal or shorter with the memory-optimized scheduler (5.1 vs 5.5 s,
5.6 vs 7.5 s and 7.5 vs 15.2 s for jvp at 6, 8 and 10; 92 s for the resolution-20 jac).

Adding the CI setting `--xla_cpu_multi_thread_eigen=false` leaves the temp unchanged
(1,031,495,480 bytes for jvp at 6). `CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED` reproduces the
default's temp to the byte, so on this jaxlib the default is the concurrency-optimized scheduler.

## What the buffers are

XLA's buffer-assignment report for one JVP column at resolution 6
(`--xla_dump_to`, `*-memory-usage-report.txt`), grouped by shape:

```text
primal: 239.2 MiB   295 x f64[41,9,4,4,9,2]     jvp: 867.5 MiB  1070 x f64[41,9,4,4,9,2]
         17.4 MiB    43 x f64[41,9,4,4,9]             82.3 MiB   203 x f64[41,9,4,4,9]
```

The shape is `[edge pieces, 3×3 periodic images, ny, nx, cut segments, two Gauss nodes]` from
`_painted_cell_materials` (`scene.py` lines 617–629, the `cdf`/`weight` window loops). The HLO
stack frames on these buffers point to those lines, and the op names are
`jvp(jit(clip))/select_n`, `min`, `mul` and `add`. The consumers are 216 YNN fusions. A typical one:

```text
%ynn_fusion.40 = f64[4,4,2] fusion(<40 operands>), calls=%fused_computation.40,
    backend_config={"fusion_config":{"kind":"__ynn_fusion"}}
  fused_computation.40: 39 parameters (31 x f64[41,9,4,4,9,2]), 30 multiply, 13 add,
    ROOT reduce_sum = f64[4,4,2] reduce(...), dimensions={0,1}
```

This is `total + sum(value*jump, axis=(0, 1))`. Under JVP the product rule splits it into many
terms. All 1,071 large values are defined before the first reduction that uses them runs.
A liveness pass over the scheduled entry computation counts how many are alive at once:

| Schedule (jvp, resolution 6) | Large buffers | Alive at once | Temp |
|---|---:|---:|---:|
| default (concurrency-optimized) | 1,071 | 1,071 | 1031 MB |
| memory-optimized | 1,071 | 37 | 42 MB |

The primal has the same structure with 297 large buffers, all 297 alive at once (276 MB).
Disabling YNN fusion (`--xla_cpu_experimental_ynn_fusion_type=`) only partly helps, because the
producers can then fuse into ordinary reductions: 107 MB for jvp and 433 MB for jac at resolution 6.

## What was ruled out

- **Time-loop carries of tangents.** The program has 13 `while` loops with known trip counts of
  32, 7 and 2 in both the primal and the JVP, so they are not unrolled. The largest loop carry is
  0.45 MB (`c128[3,42,4,4]` fields with their tangents). The checkpointed scans are not where the
  memory goes.
- **Captured constants.** All HLO constants together come to 0.26 MB in the JVP module.
- **Compilation as the main cost.** The compiler accounts for 1.3–1.6 GB of resident memory and
  depends little on resolution. That is the part that "barely falls with resolution". The part
  that grows, and that makes the derivative case 4–5× eager, is the execution temp.
- **`_eager_scan`.** Under an outer jit every leaf is a tracer, so it falls back to plain
  `lax.scan`. Eager `jacfwd` takes the same fallback because its closed-over values are tracers.

The slow test's docstring lists 5.3, 6.2 and 7.2 GB at resolutions 6, 8 and 10, with duration 40.
Those come from a process that runs the eager Jacobian before the jit one. A jit-only process
here measures 2.93, 3.76 and 4.68 GB at duration 60. The differences in the eager peaks (2.67 GB
here against 3.57 GB recorded at resolution 20) were not investigated.

## Public users

The same program arises from the public periodic API. A forward derivative in the rotation angle
of two `Rectangle` regions, with `fdtd.periodic.Numerics(spacing=1/6, shape_integration='area',
shape_smoothing=.05, ...)` and `run(..., backend='jax')` under `jax.jit(jax.jvp(...))`, gives:

| Public route, resolution 6 | Temp | Peak |
|---|---:|---:|
| `area`, jit default | 1031.5 MB | 2.35 GB |
| `area`, jit memory-optimized | 201.5 MB | 1.57 GB |
| `area`, eager | — | 1.58 GB |
| `separable` (unrotated, width derivative), jit default | 0.8 MB | 0.46 GB |

At resolution 10 the public `area` temp is 2321.5 MB with the default scheduler and 453.2 MB
with the memory-optimized one. A rotated rectangle is refused by `separable`, so every public
user who rotates a rectangle, or differentiates its angle, under an outer jit on CPU is on the
affected path. GPU was not measured; XLA:GPU uses a different scheduler.

## Candidate fixes

| Fix | Scope | Expected effect |
|---|---|---|
| `jax.jit(g, compiler_options={'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'})` | one jitted call, such as the slow test | Measured: jvp temp 1031 → 42 MB at resolution 6, identical to setting the flag. Should allow an outer jit at resolution 20 again (2.44 GB measured) |
| `XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED` | process or CI job, like the existing Eigen flag | Same per-program effect for every jit in the job; run times were equal or shorter for these programs, but the effect on other programs was not measured |
| Bound the painting's live set in `_painted_cell_materials`: run the four Yee offsets or the edge groups sequentially through the backend's loop, or contract the `cdf`/`weight` window products into the reduction without materializing `[E,9,ny,nx,9,2]` | library, every scheduler and user | Expected to bring the temp near the memory-optimized figures for any caller. Not implemented or measured. It must stay backend-neutral and keep the numbers unchanged |
| Disable YNN fusion | process | Partial: jac temp 1333 → 433 MB at resolution 6. Experimental flag; not recommended |
| Document the option for users | public users | Done: [periodic-stack FDTD guide](../../guide/fdtd/periodic-problems.md), "Memory under an outer `jax.jit` on CPU". No code change; users opt in on their own jit |
