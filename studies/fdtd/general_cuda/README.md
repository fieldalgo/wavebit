# General CUDA cost studies

| Entry | Question it answers |
| --- | --- |
| `checkpoint_cost.py` | How do opt-in checkpoint blocks change reverse-call time and allocator peaks for a bounded general-scene conductor with an exact SI recurrence? |
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
