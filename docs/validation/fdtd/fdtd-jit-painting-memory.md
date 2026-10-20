# Outer-jit memory of area-painted FDTD derivatives

Written: 2026-10-19.

Targets: `src/wavebit/fdtd/_periodic/scene.py`, `src/wavebit/fdtd/_periodic/runtime.py`,
`studies/cross_solver/painted_shapes/fdtd_painted_shapes.py`, `tests/fdtd/test_fdtd_painted_optics.py`.

**TL;DR.** The extra memory comes from how XLA:CPU orders the compiled program. The time loop
and the size of the program are not the cause. When a layer holds two or more area-integrated
rectangles, `_painted_cell_materials` builds many `[edges, images, ny, nx, segments, gauss]`
float64 intermediates. Their reductions are YNNPACK fusions, and a fusion's inputs must exist as
whole buffers. XLA:CPU's default scheduler in jaxlib 0.11.0 is the concurrency-optimized one.
Its order releases none of those intermediates until the last one has been created, so all of
them are alive at the same moment. Forward-mode differentiation multiplies how many there are:
297 for the primal, 1,071 for one JVP column at resolution 6. Eager execution dispatches one
operation at a time and never holds more than one painting's working set.

With `xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`, the same program asks for a
17.5× smaller XLA temp allocation for `jit(value, jacfwd)` at resolution 20 (11.67 GB → 0.67 GB).
It ran at a 2.44 GB peak, where the resolution-20 test's docstring records 13.6 GB with the default
scheduler. Its Jacobian matches the eager one to 1.1e-15.

Public `wavebit.fdtd.periodic.run(..., backend='jax')` users are affected when a layer holds
two or more rectangles with `shape_integration='area'`. A layer with one rotated rectangle is not
affected, and neither is the `'separable'` route. Reverse mode was measured at one resolution and
is affected much less. jax 0.12.0 still shows the effect, at about a quarter of the 0.11.0 ratio.

## Configuration

Measured on 2026-10-19 at `3f2ad83`: a Linux x86-64 cloud container (Intel Xeon @ 2.80 GHz,
4 vCPU, 15 GiB), Python 3.13.16, NumPy 2.5.4, JAX/jaxlib 0.11.0, float64 throughout. The
workload is `fdtd_painted_shapes.response(p, resolution)` with `p = [.37, .055, .22]` and its
default duration 60. Each row is a separate process, and the processes ran one at a time.
The one-rectangle and reverse-mode rows (compile-only) were added later the same day in a fresh
container whose CPU reports Intel Xeon @ 2.10 GHz (4 vCPU, 15 GiB; same Python, NumPy and JAX;
source unchanged since `3f2ad83`). Two slow tests in `tests/fdtd/test_fdtd_painted_optics.py` are
cited below. The resolution-20 test is
`test_painted_eight_complex_channels_and_three_shape_derivatives_at_resolution_20`, and the
resolution-6 test is `test_painted_jacobian_under_jit_equals_the_eager_one_at_resolution_6`.

- **Peak** is `ru_maxrss`.
- **Temp** is `jax.jit(g).lower(p).compile().memory_analysis().temp_size_in_bytes`.
- The memory-optimized rows set only
  `XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`.
- **jac** is `jit(lambda p: (f(p), jax.jacfwd(f)(p)))`, the form the study uses and the resolution-6 test runs.
- **jvp** is one forward column in the angle.

Run times are single calls. The per-process measurement logs and XLA dumps are not committed.
The last section, Reproducing, gives the calls that produce them.

## Temp size follows the schedule, not the program

| Resolution | Case | Eager peak | jit default: temp / peak | jit memory-optimized: temp / peak |
|---:|---|---:|---:|---:|
| 6 | primal | 1.31 GB | 0.28 / 1.03 GB | 0.015 / 0.80 GB |
| 6 | jvp | 1.42 GB | 1.03 / 2.40 GB | 0.042 / 1.49 GB |
| 6 | jac | 2.37 GB | 1.33 / 2.93 GB | 0.076 / 1.72 GB |
| 8 | primal | 1.31 GB | 0.43 / 1.20 GB | 0.024 / 0.82 GB |
| 8 | jvp | 1.47 GB | 1.61 / 2.92 GB | 0.065 / 1.43 GB |
| 8 | jac | 2.41 GB | 2.08 / 3.76 GB | 0.117 / 1.82 GB |
| 10 | primal | 1.32 GB | 0.62 / 1.37 GB | 0.035 / 0.81 GB |
| 10 | jvp | 1.42 GB | 2.32 / 3.62 GB | 0.092 / 1.47 GB |
| 10 | jac | 2.49 GB | 3.00 / 4.68 GB | 0.166 / 1.90 GB |
| 20 | jac | 2.67 GB | 11.67 GB / not run | 0.67 / 2.44 GB |

The two scheduled JVP modules at resolution 6 contain the same instructions in a different order: 2,938 entry
instructions, 1,071 large values and 216 YNN fusions under both. So compilation costs the same
under both. The process holds 0.3–0.4 GB after lowering and
1.3–1.6 GB after compiling a derivative program, so compiling adds 1.0–1.2 GB. Each jit peak is
about that post-compile figure plus the temp. The temp grows roughly with `nx·ny`: the cell is
4×4 at resolution 6, and the resolution-20 temp is 8.8× the resolution-6 one.

The resolution-20 default temp plus the compile figure comes to about 13.3 GB. That matches the
13.6 GB that the resolution-20 test's docstring records for this case on Linux CPU with jax 0.11.0, and
the 13.3 GB it records on a CI runner. The default resolution-20 program was compiled but not run
here, because 13.3 GB would leave too little headroom on this 15 GiB machine. The 22.9 GB single
allocation that the same docstring reports for resolution 30 is consistent with this temp buffer.

Values do not depend on the scheduler. At resolutions 6–10 the default and memory-optimized runs
produce bit-identical output sums, and those sums agree with eager to within 3.4e-15. At
resolution 20 the Jacobian was compared element by element with eager: 1.1e-15.

Derivative run times were equal or shorter with the memory-optimized scheduler. Those times were:

- jvp: 5.1 vs 5.5 s, 5.6 vs 7.5 s and 7.5 vs 15.2 s at resolutions 6, 8 and 10.
- jac: 7.7 vs 7.9 s, 10.8 vs 13.1 s and 16.0 vs 16.8 s.

The primal was slower at resolutions 8 and 10: 2.3 vs 1.7 s and 2.9 vs 2.6 s. The
memory-optimized resolution-20 jac ran in 92 s, against 248 s eager.

Adding the CI setting `--xla_cpu_multi_thread_eigen=false` leaves the temp unchanged
(1,031,495,480 bytes for jvp at resolution 6). `CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED`
reproduces the default's temp to the byte, so on this jaxlib the default is the
concurrency-optimized scheduler.

The same JVP at resolution 6 was compiled again on 2026-10-20, in the 2.10 GHz container, from
`main` at `20a9715`. With jax/jaxlib 0.12.0 it asks for 785 MB with the default scheduler and
197 MB with the memory-optimized one, and `CPU_SCHEDULER_TYPE_CONCURRENCY_OPTIMIZED` again
matches the default to the byte. With jax 0.11.0, the same source still gives 1031 and 42 MB, so
the change comes from the newer XLA, not from the source. On 0.12.0 the effect is smaller (about
4× instead of 24×) but still present. The rest of this record was measured on 0.11.0 only.

## What the buffers are

XLA's buffer-assignment report for resolution 6 (`--xla_dump_to`,
`*-memory-usage-report.txt`) gives these allocations, each counted under the first shape it lists:

```text
primal: 239.2 MiB   295 x f64[41,9,4,4,9,2]     jvp: 867.5 MiB  1070 x f64[41,9,4,4,9,2]
         17.4 MiB    43 x f64[41,9,4,4,9]             82.3 MiB   203 x f64[41,9,4,4,9]
```

The shape is `[edge pieces, 3×3 periodic images, ny, nx, cut segments, two Gauss nodes]`. It
comes from `_painted_cell_materials`: the `cdf` and `weight` window loops and the sums after them,
`scene.py` lines 615–630. The HLO stack frames on these values point to those lines. Their op
names are `jvp(jit(clip))/select_n`, `min` and `max`, `jvp(jit(_where))/select_n`, `mul`, `add`
and `div`.

The scheduled entry computation defines 1,071 values of this shape for the JVP (297 for the
primal). Each of those values has an allocation of its own. One JVP allocation and two primal ones list a
`c128[3,42,4,4]` value first, so the block above counts them under that shape and shows one or two
fewer. 96 of the program's 216 YNN fusions read these values. A typical one:

```text
%ynn_fusion.40 = f64[4,4,2] fusion(<40 operands>), calls=%fused_computation.40,
    backend_config={"fusion_config":{"kind":"__ynn_fusion"}}
  fused_computation.40: 40 parameters (32 of painting size), 30 multiply, 13 add,
    reduce -> f64[41,9,4,4,9], reduce -> f64[41,9,4,4],
    ROOT reduce -> f64[4,4,2], dimensions={0,1}
```

The fusion holds the Gauss-node and segment sums of line 629 and the `sum(value*jump,
axis=(0, 1))` of line 630. Under JVP the product rule splits them into many terms. The
reductions start before the last large value is defined: 48 of those 96 fusions run earlier. But
all 1,071, including the 96 those early fusions read, are also read by the other 48 fusions. Those
run after the last one is defined (instructions 2,757–2,851 of 2,938), so none reaches its last
use until all 1,071 exist.
A liveness pass over the scheduled entry computation counts how many are alive at once:

| Schedule (jvp, resolution 6) | Large values | Alive at once | Temp |
|---|---:|---:|---:|
| default (concurrency-optimized) | 1,071 | 1,071 | 1031 MB |
| memory-optimized | 1,071 | 37 | 42 MB |

The primal has the same structure: all 297 of its large values are alive at once (276 MB).
Disabling YNN fusion (`--xla_cpu_experimental_ynn_fusion_type=`) gives 107 MB for jvp and
433 MB for jac at resolution 6. Only the temp sizes were measured for that setting.

## What was ruled out

- **Time-loop carries of tangents.** The program has 13 `while` loops in both the primal and the
  JVP, with known trip counts of 32, 7 and 2, so they are not unrolled. The largest loop carry is
  0.45 MB (`c128[3,42,4,4]` fields with their tangents). The checkpointed scans are not where the
  memory goes.
- **Captured constants.** All HLO constants together come to 0.26 MB in the JVP module.
- **Compilation as the main cost.** At `3f2ad83` the resolution-6 test's docstring called the
  jit's cost "mostly compilation"; it now cites this record. Compiling adds 1.0–1.2 GB and hardly changes with resolution. The part
  that grows is the execution temp. Here it takes the derivative case from 1.2–2.5× the eager
  peak at resolutions 6–10 to about 3.8× at resolution 20 (13.6 GB against 3.6 GB in the
  docstring of the resolution-20 test).
- **`_eager_scan`.** Under an outer jit every leaf is a tracer, so it falls back to plain
  `lax.scan`. Eager `jacfwd` takes the same fallback because its closed-over values are tracers.

The resolution-6 docstring lists 5.3, 6.2 and 7.2 GB at resolutions 6, 8 and 10, at duration 40.
Those figures come from a process that runs the eager Jacobian before the jit one. A jit-only
process here measures 2.93, 3.76 and 4.68 GB at duration 60. The difference between eager peaks,
2.67 GB here against 3.6 GB in the docstring at resolution 20, was not investigated.

## Public users

The public periodic API reaches the same painting route. The case measured is a forward
derivative in the rotation angle of the study's two `Rectangle` regions in one 0.2 µm layer.
The setup, in the public names current since 2026-10-19 (the run at `3f2ad83` spelled the pulse
`Pulse(width_seconds=2.5*units.time_seconds)` and the time unit `units.time_seconds`; see the
[public-name migration guide](../../migration/public-names.md#native-grid-and-periodic-stack-names)):

- `Stack(period=(.6, .6), bottom=Material(epsilon=1.21))`, `PlaneWave` at 1 µm,
  `Pulse(frequency_sigma_hz=1/(np.sqrt(2)*np.pi*2.5*units.time_unit_seconds))`.
- `Numerics(spacing=1/6, duration_seconds=60*units.time_unit_seconds, pml_cells=6, orders=0,
  align_layers=True, shape_integration='area', shape_smoothing=.05,
  extraction_method='directional')`.
- `run(..., backend='jax')` under `jax.jit(jax.jvp(...))`.

It compiles a somewhat different program from the study. Its default-scheduler temps equal the
study's jvp temps (1031.5 MB at resolution 6, 2321.5 MB at 10). Its memory-optimized temps are
larger: 201.5 against 42.5 MB at resolution 6 and 453.2 against 92.1 MB at 10, a reduction of about
5× instead of 24×.

| Public route, resolution 6 | Temp | Peak |
|---|---:|---:|
| two rotated rectangles, `area`, jit default | 1031.5 MB | 2.35 GB |
| two rotated rectangles, `area`, jit memory-optimized | 201.5 MB | 1.57 GB |
| two rotated rectangles, `area`, eager | — | 1.58 GB |
| one rotated rectangle, `area`, jit default | 10.1 MB | (compile only) |
| one rotated rectangle, `area`, jit memory-optimized | 21.7 MB | (compile only) |
| two unrotated rectangles, `separable`, width derivative, jit default | 0.8 MB | 0.46 GB |

At resolution 10 the two-rectangle public temp is 2321.5 MB with the default scheduler and
453.2 MB with the memory-optimized one.

`_shape_partition` sends a layer with two or more area shapes, which must be rectangles, to
`_painted_cell_materials`. A layer with a single rectangle, circle or ellipse goes to
`_shape_cell_fractions`, and the one-rectangle row shows that route is not affected. Circles and
ellipses were not measured. `separable` refuses rotation, so a user who rotates two or more
rectangles in one layer, or differentiates their angle, is on the affected path whenever an
outer jit runs on CPU.

Reverse mode was measured once, compile-only, at resolution 6. For the study's
`jax.jit(jax.grad(lambda p: jnp.sum(jnp.abs(response(p, 6))**2)))` the temp is 361 MB with the default
scheduler and 203 MB with the memory-optimized one. Its eager peak was not measured. GPU was not
measured; XLA:GPU uses a different scheduler.

## Candidate fixes

| Fix | Scope | Expected effect |
|---|---|---|
| `jax.jit(g, compiler_options={'xla_cpu_scheduler_type': 'CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED'})` | one jitted call, such as the resolution-20 test | Measured: jvp temp 1031 → 42 MB at resolution 6, the same as the flag. Should allow an outer jit at resolution 20 again (2.44 GB measured with the flag) |
| Add `--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED` to `XLA_FLAGS` (keeping the flags already set) | process or CI job, like the existing Eigen flag | The same per-program effect for every jit in the job. Derivative run times were equal or shorter here, but the primal was up to 38% slower at resolutions 8 and 10, and other programs were not measured |
| Bound the painting's live set in `_painted_cell_materials`: run the four Yee offsets or the edge groups sequentially through the backend's loop, or contract the `cdf`/`weight` window products into the reduction without materializing `[E,9,ny,nx,9,2]` | library, every scheduler and user | Expected to bring the temp near the memory-optimized figures for any caller. Not implemented or measured. It must stay backend-neutral and keep the numbers unchanged |
| Disable YNN fusion | process | Partial: jac temp 1333 → 433 MB at resolution 6. Experimental flag; not recommended |
| Document the option for users | public users | Done in the [periodic-stack FDTD guide](../../guide/fdtd/periodic-problems.md), section "Memory under an outer `jax.jit` on CPU". No code change; users opt in on their own jit |

## Reproducing

From a source checkout with `.[jax]` installed, at `3f2ad83` or later:

```python
import jax, numpy as np
jax.config.update('jax_enable_x64', True)
from studies.cross_solver.painted_shapes.fdtd_painted_shapes import response

p, resolution = np.array([.37, .055, .22]), 6
f = lambda p: response(p, resolution)
g = lambda p: jax.jvp(f, (p,), (np.array([1., 0., 0.]),))[1]   # or (f(p), jax.jacfwd(f)(p))
compiled = jax.jit(g).lower(p).compile()
print(compiled.memory_analysis().temp_size_in_bytes)
```

Run it once with no `XLA_FLAGS` and once with
`XLA_FLAGS=--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED`. Add
`--xla_dump_to=<dir> --xla_dump_hlo_as_text` to get the buffer report and the scheduled module.
Peak resident memory is `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` after
`jax.block_until_ready(compiled(p))`.
