# Native CUDA cost study

| Entry | Question it answers |
| --- | --- |
| `native_cuda.py` | At which of the measured grid sizes does eager native CUDA execution outperform one-thread Torch CPU, and how much CUDA allocator memory do forward and reverse calls use? |
| `profile_native_cuda.py` | Which CPU dispatch and CUDA operators contribute to a bounded native solve, with profiler overhead and memory-lifetime effects kept distinct from benchmark timing? |

Run from the checkout's configured CUDA environment:

```sh
python studies/fdtd/native_cuda/native_cuda.py --sizes 8 16 32 64 --steps 24 --repeat 3 --output artifacts/native-cuda.json
```

The JSON is a generated local output. The instrument synchronizes CUDA, warms
each configuration once, and records all repetitions, medians and allocated and
reserved peaks and the live allocation baseline. Forward calls disable gradient
mode; reverse calls include forward and backward with gradient mode enabled.
Every repetition retains its objective/gradient and undergoes finite, CPU/CUDA
parity and quadratic-source checks where applicable. A failed check preserves
the failed row and stops the instrument. It requires a CUDA-enabled Torch
installation and an NVIDIA GPU.
The [validation record](../../../docs/validation/fdtd/native-cuda.md) states the
measured environment, timing boundaries, numerical coverage and limitations.
Its separate 4097/16385-step single-mode validation checks phase, amplitude,
nonmodal residual and mode energy with predeclared numerical budgets. It is
not a cost study or a general long-time stability qualification; no histories
are requested, but the zero drive array still grows with the step count.

For an instrumented operator trace, choose a new output directory:

```sh
python studies/fdtd/native_cuda/profile_native_cuda.py --size 32 --steps 24 --checkpoint-steps 8 --mode forward --device cuda:0 --output-dir artifacts/native-cuda-profile
```

This captures one solve after warm-up. Profiling changes cost and tensor lifetimes;
inclusive operator durations overlap and are not additive wall time. Reports,
traces and attempt receipts are local outputs, including on failure.
