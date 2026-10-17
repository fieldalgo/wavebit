# General CUDA cost studies

| Entry | Question it answers |
| --- | --- |
| `checkpoint_cost.py` | How do opt-in checkpoint blocks change reverse-call time and allocator peaks for a bounded general-scene conductor with an exact SI recurrence? |
| `ade_cost.py` | What allocator peaks and reverse-call times does the bounded homogeneous Lorentz CUDA route produce, and how do they differ from its logical state payload? |
| `monitor_cost.py` | How much dispatch and device work does a bounded no-grad prepared run with two Ey Field and two Flux monitors perform? |

## Checkpoint cost

Run in the checkout's configured Torch CUDA environment:

```sh
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 16 32 --steps 163 --blocks 0 8 32 200 --repeat 3 --output artifacts/general-checkpoint.json
```

The study's zero maps to public `checkpoint_steps=None`; public block length
zero is invalid. A block larger than the duration executes
only the ordinary tail. Each configuration has one warm-up and synchronized
forward-plus-reverse repetitions. The default compares an amplitude-only graph
with a graph also differentiating conductivity (`--parameters amplitude` or
`--parameters material` selects one). Timing includes live leaf construction,
prepared execution, the squared-electric-field objective and backward; initial
preparation, scalar copies to the host and JSON writing are excluded.

The uniform conductor and one point record are a cost fixture. Every row must
match the independent SI recurrence and its amplitude and selected material derivatives. The JSON
preserves attempted configurations and failed checks. Allocated/reserved peaks
exclude driver/display memory; the baseline allocation is recorded separately.
Neither this fixture nor checkpointing establishes a total-memory bound or a
guaranteed saving for other fields, observations or source histories.

The legacy matrix remains sizes 4..32 and durations 1..512. A separate volume
scaling case permits **only 64³, 163 steps, material gradients**, with one policy
per fresh process. Run these commands sequentially in the order shown; each has
one warm-up and five timed repetitions:

```sh
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 8 --repeat 5 --output artifacts/general-checkpoint-64-k8.json
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 32 --repeat 5 --output artifacts/general-checkpoint-64-k32.json
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 0 --repeat 5 --output artifacts/general-checkpoint-64-plain.json
```

Before warm-up, the instrument saves free/total device memory and current
allocated/reserved bytes to the attempt. For the 64³ case it forecasts reserved
memory from the observed 32³/163-step material peaks: 350224384 bytes for plain,
69206016 for K8 and 88080384 for K32. Eightfold cell count and a 1.5× margin give
approximately 3.914, 0.773 and 0.984 GiB respectively. It refuses a forecast above
4 GiB or free VRAM below the forecast plus 1 GiB headroom. This estimate cannot
reserve memory or guarantee avoidance of an out-of-memory failure; other GPU
users can change free memory after the check. Larger durations and combined
policy invocations at 64³ are rejected.

Each row records post-warm-up allocated/reserved baselines, absolute and
incremental peaks, per-repetition allocated bytes after releasing the returned
objective/gradients, and allocated/reserved bytes after final collection and
cache release. Raw synchronized times, median and range describe the entire
forward-plus-reverse call. Cleanup and allocator inspection are outside timing.
The SI objective and both derivative tolerances remain `rtol=2e-12, atol=0`.
Failed preflights and execution failures retain their attempt records; use a new
output path on every invocation. Profile separately because instrumentation
changes tensor lifetimes and costs.

The [CUDA validation record](../../../docs/validation/fdtd/general-cuda.md)
reports both measured derivative modes, including a case where checkpointing
increases allocated peak and one where it reduces it at a time cost.

## Monitor dispatch and profiling

```sh
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 4 --repeat 5 --device cuda:0 --profile --output-dir artifacts/general-monitor-profile
```

The short film fixture uses three wavelengths, float64/complex128 and one Torch
CPU thread. It performs one warm-up, then synchronized no-grad prepared runs.
Timing includes live validation, lowering and result adaptation; initial
preparation, result checks, host copies and the profiler are excluded. Optional
profiling follows the uninstrumented repetitions and exports operator groups
and a Chrome trace. CPU dispatch and device activity overlap; do not add their
inclusive times. Shape/memory recording changes cost and tensor lifetimes.

The instrument checks runtime validity, field/flux placement, dtype and finite
values, but the short window is not an Airy accuracy or convergence test.
`--device cpu` checks the instrument without supplying CUDA performance
evidence. The output directory must be new; the attempt record is written
before imports and execution so failed configurations are retained.

The [validation record](../../../docs/validation/fdtd/general-cuda.md) reports
actual CUDA before/after evidence for skipping unused field phases: Ey-only
monitors omit H phases while existing within-field sharing and Yee time levels
are preserved. In the measured 128-step fixture, profiled
exponential calls fell from 1025 to 769 and multiplications from 5888 to 5120.
Five separate uninstrumented repetitions had overlapping time ranges, and
allocated peak was unchanged. The field-maximum/flux summaries matched exactly;
this short early-pulse fixture is not an independent optical accuracy test.

An explicit checkpoint policy can be measured under no-grad as well:

```sh
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 16 --checkpoint-steps 8 --repeat 5 --device cuda:0 --output-dir artifacts/general-monitor-no-grad
```

The recorded before/after experiment holds requested-phase computation fixed and isolates
the disabled-gradient forward path. Allocated peak fell from 7394304 to
5763072 bytes, while reserved peak stayed at 8388608 bytes; the five-repeat
time ranges overlapped. This is a forward-only result for the measured fixture,
not a reverse-mode saving or an unconditional speedup. The policy is still
reported and applies to later gradient-enabled prepared reuse.

## Homogeneous ADE cost

`ade_cost.py` uses the ordinary public preparation/execution path for one
full-grid Lorentz material in a six-periodic-face box. It has no alternate
runtime context or baseline/candidate switch. CUDA selects the compact pole
state; the optional CPU instrument check retains the ordinary pole state.
The fixed fixture has 163 steps, double precision, uniform initial Ey, one
Record and final fields. It differentiates the initial amplitude and Lorentz
strength. Plain execution omits `--checkpoint-steps`; 8 and 32 each leave a
three-step tail.

Set `ADE_RECORDS` to a writable directory outside the checkout, then run each
configuration in a fresh process and a new output directory:

```sh
python studies/fdtd/general_cuda/ade_cost.py --size 32 --checkpoint-steps 8 --output-dir "$ADE_RECORDS/ade-32-k8"
python studies/fdtd/general_cuda/ade_cost.py --size 64 --checkpoint-steps 8 --forecast-from "$ADE_RECORDS/ade-32-k8/attempt.json" --output-dir "$ADE_RECORDS/ade-64-k8"
python studies/fdtd/general_cuda/ade_cost.py --device cpu --size 8 --output-dir "$ADE_RECORDS/ade-cpu-check"
```

Repeat the size-32 then size-64 pair with block 32 or with no block argument
for the other policies. Size 64 requires an authentic passed size-32 receipt
with the same policy, fixture, source/script hashes, product revision and
recorded environment. The maximum reserved peak is scaled by eightfold volume
and a 1.5 margin. The study refuses a forecast above 4 GiB or free device memory
below forecast plus 1 GiB, both before construction and before warm-up. This is
a budget estimate, not a memory reservation or an out-of-memory guarantee.

Each process performs one warm-up and five synchronized measurements. One timed
interval includes fresh leaf and live-problem construction, `prepared.run`
(including live validation, lowering and adaptation), the squared-final-E
objective and `autograd.grad`. Initial preparation, instrument numerical
checks/output host copies, allocator queries and cleanup are outside timing.
The public run's own validation/adaptation host work remains included. The allocator cache
is released after warm-up and after all repetitions, but retained between
repetitions. Returned tensors are released before the next baseline.

Every repetition checks the full Record against independent damped roots and
a scalar recurrence, and the objective plus both derivatives against a
four-state matrix and its Frechet derivative. The Record budget is `1e-11`,
absolute recurrence residual is below `1e-12`, and objective/gradient relative
budget is `2e-12` with zero absolute tolerance. A separate untimed lifecycle
records retained memory after forward and backward; two further runs check
old-output ownership. These allocator peaks are distinct from timed peaks.
The attempt also verifies actual pole `init_mem` bytes against the logical
report. Logical E/H-plus-pole state is 96 bytes/cell for the compact CUDA term,
versus 120 for the ordinary CPU term; neither includes the retained autograd
graph, observations, temporary arrays, allocator reserve or driver memory.

Attempts, source integrity, raw times, baselines, absolute/incremental peaks,
retained allocations and cleanup are recorded even when a check fails. Add
`--profile` only in a separate process/output directory: it performs a warm-up
and one profiled call, with zero unprofiled timing samples. Operator JSON and
a Chrome trace are instrumentation evidence, not throughput measurements.
The [validation record](../../../docs/validation/fdtd/general-cuda.md)
separates historical state-reduction evidence from current product measurements.

The current six-row 32³/64³ production measurement is recorded in the validation
page. All checks passed, and allocated/reserved peaks matched the earlier
compact-state research measurements. Those historical timing comparisons had
overlapping ranges; the product table does not establish a general speedup.
