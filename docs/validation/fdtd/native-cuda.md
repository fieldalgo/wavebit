# Native Yee-grid CUDA validation

Written: 2026-10-18

Targets:

- `tests/fdtd/test_native_cuda.py`
- `tests/fdtd/test_native_cuda_admission.py`
- `tests/fdtd/test_periodic_cuda.py`
- `tests/fdtd/test_jax_gpu.py`
- `src/wavebit/fdtd/native.py`
- `src/wavebit/fdtd/_periodic/grid.py`
- `tests/fdtd/test_fdtd_scan_programs.py`
- `studies/fdtd/native_cuda/native_cuda.py`
- `studies/fdtd/native_cuda/profile_native_cuda.py`

The existing public `wavebit.fdtd.native` engine runs the cases below on Torch
CUDA in double precision, including eager reverse-mode derivatives. These checks
add bounded device evidence. The general `wavebit.fdtd` scene API has a
[separate restricted CUDA validation](general-cuda.md). [JAX on GPU](#jax-on-gpu)
is checked only for the cases in its section. Single precision, compilation and comprehensive periodic-stack CUDA accuracy
are unvalidated here.

## Environment and reproduction

The measurement used Wavebit 0.0.4 source at base `2c04ccae`, with the tests,
study and native spectrum-coordinate device fix described here; Windows 11
build 22631.6199, Python 3.12.14,
Torch 2.14.1+cu130, CUDA runtime
13.0, NVIDIA driver 610.88, an AMD Ryzen 5 7500F (6 cores / 12 threads) and an
RTX 4060 Ti with 8 GiB VRAM. NumPy was 2.5.4,
SciPy 1.18.1 and array-api-compat 1.15.0. This is one measured machine, not a
cross-platform qualification.

The 24-step size sweep and 96-step checkpoint tables below retain the baseline
measured **before Bloch-factor reuse**. The separate before/after measurement
records that later optimization. Reproduction commands on current source run
the same workloads with the current implementation.

Create an isolated environment for the checkout, activate it, and install the
CUDA wheel explicitly. For example, in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[pytorch]" pytest
python -c "import torch, wavebit; print(wavebit.__file__); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest tests/fdtd/test_native_cuda.py tests/fdtd/test_native_cuda_admission.py -q
python studies/fdtd/native_cuda/native_cuda.py --sizes 8 16 32 64 --steps 24 --repeat 3 --output artifacts/native-cuda.json
```

For other shells, use their virtual-environment activation command; see the
[source checkout guide](../../development/source-checkout.md). Confirm that
`wavebit.__file__` names this checkout and that CUDA is available before treating
the test result as GPU evidence. Missing Torch or CUDA skips these tests; a skip
does not establish execution. The generated JSON is a local output. The original
raw timing records are not distributed with the checkout; their measured medians
and allocator peaks are transcribed below.

Use `native.run(problem, numerics=numerics, backend='torch', device='cuda:0')`,
or pass the same backend/device to `native.prepare`. Inputs are converted to
the execution device, while gradients remain associated with their original
leaf tensors. See the
[native-grid guide](../../guide/fdtd/native-grids.md) for units and array order.

## Numerical coverage

The native CUDA file passed eight cases on the measured device; an earlier
admission-file run passed 21 cases covering CPU/CUDA and device-independent controls. These
are finite-case checks, not a GPU qualification of every FDTD feature.

| Case | Reference and criterion |
| --- | --- |
| Uniform conducting medium, constant current, 11 steps | Independent SI trapezoidal conductivity recurrence for E and unchanged H; relative tolerance `2e-14`; output dtype/device checked |
| 30-step pulse with z CPML, a Lorentz pole, transverse Bloch phase and nonuniform z spacing | NumPy CPU E/H, polarization, recorded fields, raw Fourier fields and spectral power; each quantity's peak-normalized infinity error at most `5e-12`; prepared execution reused with two source amplitudes |
| Source-amplitude reverse derivative, checkpoint intervals 32 and 7, CPU and CUDA leaves | Exact quadratic scaling of the final squared-E objective to `2e-12` relative, and NumPy central difference to `2e-7`; recording disabled so the full 30-step loop exercises distinct checkpoint partitions |
| Recorded E objective's source derivative | Exact quadratic scaling to relative `2e-12`, with recording enabled |
| Permittivity contrast, oscillator strength and transverse Bloch-phase reverse derivatives | NumPy objective to `2e-12` relative and nonzero central differences at steps `1e-4` and `5e-5` to `2e-6` relative |
| Discrete traveling wave, with zero and nonzero Bloch phase | Independent Yee dispersion and staggered E/H plane wave after 37 and 513 steps, CPU and CUDA, relative `3e-13` |
| Admission and observation placement | Unavailable CUDA, invalid material/CFL/PML inputs and incompatible prepared reuse rejected; check/prepare do not advance fields; field, spectrum and record coordinates, tail and validity tensors remain on the execution device |

The placement check exposed spectrum x/y coordinates allocated on CPU while
spectral values and z coordinates were on CUDA. Constructing those coordinates
on the field's device fixes that result-contract discrepancy without changing
the Yee recurrence. Both CPU and CUDA placement cases pass after the fix.

E, H, polarization and Fourier power have different SI units and scales. The
field comparison therefore normalizes each complete quantity by its own
reference peak, including components close to cancellation zeros; it does not
apply one absolute tolerance across unlike physical quantities. A zero reference
requires exact zero. CPU parity detects device discrepancies; the independent
conductivity recurrence and source-scaling identity supply separate controls.
The material/Bloch finite differences test differentiation of this discrete
problem, not continuum convergence.

### Longer single-mode propagation

A later targeted run passed 16 cases: eight unchanged 37/513-step CPU/CUDA
wave checks, two new CUDA horizons and six CPU mutation controls; 13 unrelated
admission cases were deselected. The longer horizons propagate one complex
lossless mode on a 16×2×2 periodic grid with epsilon 2.4, mu 1.7 and Bloch
phase 0.37 for 4097 and 16385 steps (about 101.68 and 406.66 cycles). The
reference uses independent Yee dispersion with spatial and temporal E/H
staggering.

Before execution, the longer-run acceptance was fixed as
`B(N) = 3e-13 * max(1, N/513)`: a numerical choice allowing coherent roundoff
growth, **not a rigorous floating-point bound**. The existing 37/513-step
`3e-13` checks are unchanged. Elementwise field comparisons use relative B
and zero absolute tolerance, retaining exact zeros of unexcited components.
Modal coefficients are projections onto the initial mode: phase error is in
radians, amplitude error is `abs(abs(coefficient)-1)`, and nonmodal residual
is the relative L2 norm after removing that projection. The relative mode
energy drift uses allowance `2*B + B**2`; this norm is appropriate to this
single complex mode and is not asserted to be conserved for arbitrary
staggered-state superpositions.

| Metric | 4097 steps | 16385 steps |
| --- | ---: | ---: |
| B(N) | 2.395906e-12 | 9.581871e-12 |
| E peak-normalized field error | 1.447407e-13 | 3.357500e-13 |
| H peak-normalized field error | 1.460854e-13 | 3.380075e-13 |
| E modal phase error (rad) | 1.418865e-13 | 3.311240e-13 |
| H modal phase error (rad) | 1.413869e-13 | 3.307354e-13 |
| E modal amplitude error | 2.331468e-15 | 1.132427e-14 |
| H modal amplitude error | 2.775558e-15 | 1.110223e-14 |
| E nonmodal residual | 2.964613e-15 | 4.913670e-15 |
| H nonmodal residual | 2.697725e-15 | 6.326759e-15 |
| Relative mode energy drift | 4.996004e-15 | 2.242651e-14 |

The six CPU controls reject `1e-6` phase, amplitude or orthogonal-component
mutations at both horizons; a phase mutation preserves energy and amplitude,
showing why energy alone is insufficient. Values above transcribe the local
test XML. The run used no-grad with no field/spectrum/tail histories, but the
engine still allocates a zero complex128 drive array of length N (about
64/256 KiB respectively). Memory is therefore O(grid + steps), not independent
of duration. These two horizons do not qualify CPML, dispersive media, arbitrary
modes or general long-time stability.

```sh
python -m pytest tests/fdtd/test_native_cuda_admission.py --run-slow -k "plane_wave_matches or long_native_wave or long_wave_acceptance" -q
```

### Separate periodic slab check

`test_periodic_cuda.py` passed two CUDA cases, illuminating a uniform slab from
the top and bottom (246.21 s for the test run). The slab is 0.1 µm thick, with
relative permittivity 2.25 in vacuum, at wavelengths 0.85, 0.90 and 0.95 µm.
Prepared coarse execution uses spacing 1/30 µm and duration 55 times the vacuum
period at 0.90 µm. R, T and complex s reflection match NumPy CPU at relative
`2e-12`, absolute `1e-14`. Halving the spacing reduces the reflection error
norm against an independent Airy formula and puts each fine complex reflection
within absolute `0.005` of it. This verifies these two finite slab cases; it
does not establish a convergence order, patterned or dispersive accuracy, or
periodic CUDA derivatives.
The same two periodic cases passed again in the final optical rerun alongside
the three [general CUDA slab cases](general-cuda.md) after the observation and
runtime optimizations.

```sh
python -m pytest tests/fdtd/test_periodic_cuda.py --run-slow -q
```

### JAX on GPU

`tests/fdtd/test_jax_gpu.py` reruns the native conductor recurrence, the
CPML, dispersion, Bloch and prepared-reuse case, and the periodic slab of the
two sections above with `backend='jax', device='gpu'`, at the same tolerances,
and adds a source-amplitude derivative compared with JAX on CPU (relative
`1e-10`) and a CPU central secant (relative `1e-6`). All five passed on
2026-10-19 at main `3f2ad839` in 71.46 s, on the same RTX 4060 Ti under WSL2
(Ubuntu 26.04 LTS, NVIDIA driver 610.88) with Python 3.12.14, JAX 0.11.0 and
`jax-cuda13-plugin` 0.11.0, NumPy 2.5.1 and SciPy 1.18.0. Every checked result
array was on the GPU device in complex128. Measured margins: the conductor
fields within 2.2e-16 of the closed form, the dispersive case within 5.4e-16
peak-normalized of NumPy, the gradient identical to the CPU one and 9.1e-13
from the secant, slab R, T and reflection within 0.4 % of their tolerance, and
the refined slab within 1.7e-3 of the Airy reflection. This covers these cases
only, on one GPU; the general `wavebit.fdtd` scene API admits JAX on CPU only.
Without a CUDA-enabled jaxlib the tests skip, and a skip is not execution
evidence.

```sh
python -m pytest tests/fdtd/test_jax_gpu.py --run-slow -q
```

## Measured cost

**Baseline before Bloch-factor reuse:** both tables in this section retain
their original measurements; they are not post-optimization timings.

The study uses cubic grids, 24 steps, a vacuum medium, a point current, two CPML
cells at each z end, float64/complex128 and checkpoint interval 8. The CPML width
is a cost fixture, not a reflection-accuracy recommendation. Each configuration
has one warm-up and three timed repetitions; CUDA is synchronized before and
after each timed call. CPU Torch uses one thread. Reverse rows include forward
execution and `autograd.grad` of the final squared-E objective with respect to
source amplitude.

Times include live input construction, `native.run` preparation, stepping,
result assembly and objective evaluation; reverse rows also include backward.
The initial fixed density tensor allocation/transfer, scalar copies to the host,
JSON writing and warm-up are outside timing. This is eager end-to-end call cost,
not a kernel-only benchmark. No `torch.compile` is used. Forward rows disable
Torch gradient mode; reverse rows enable it. All 16 CPU/CUDA rows passed the
instrument's per-repetition finite-value checks, CPU/GPU objective and gradient
parity, and exact quadratic source-gradient checks (relative `2e-12`).

| Grid | Mode | CPU median (s) | CUDA median (s) | CPU / CUDA | CUDA peak allocated / reserved (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8³ | Forward | 0.011585 | 0.033479 | 0.35 | 0.475 / 2 |
| 8³ | Forward + reverse | 0.055249 | 0.184646 | 0.30 | 0.631 / 2 |
| 16³ | Forward | 0.019155 | 0.039542 | 0.48 | 3.729 / 6 |
| 16³ | Forward + reverse | 0.078316 | 0.169783 | 0.46 | 4.814 / 6 |
| 32³ | Forward | 0.081851 | 0.037559 | 2.18 | 30.260 / 36 |
| 32³ | Forward + reverse | 0.279998 | 0.179967 | 1.56 | 38.783 / 58 |
| 64³ | Forward | 1.475292 | 0.057488 | 25.66 | 238.010 / 292 |
| 64³ | Forward + reverse | 4.233025 | 0.184147 | 22.99 | 306.033 / 372 |

The smaller cases are slower on CUDA; the 32³ and 64³ cases are faster under
this one-thread CPU comparison. These ratios do not predict another CPU thread
count, grid, duration, material, monitor configuration or precision.

Allocator peaks are reset after warm-up, garbage collection and cache release,
and cover the three repetitions, including still-live input tensors. Baseline
allocated bytes before timing are recorded separately (12,288; 98,304; 786,432;
6,291,456 for sizes 8, 16, 32 and 64 respectively). Allocated
bytes are live tensor allocations; reserved bytes include Torch's caching
allocator. Neither counts the complete GPU process, driver or display memory.
CPU memory was not measured. The 8 GiB device capacity does not establish that
larger grids or long reverse-mode trajectories fit. Checkpoint interval,
observation retention and material memories require separate measurements.

### A longer trajectory and checkpoint intervals

The same 64³ fixture was also measured for 96 steps at checkpoint intervals 8,
32 and 96, with one warm-up and three synchronized repetitions per row. All 12
CPU/CUDA rows passed the same applicable numerical checks. The baseline live
CUDA allocation was 6 MiB in every configuration.

| Checkpoint interval | Mode | CPU median (s) | CUDA median (s) | CUDA peak allocated / reserved (MiB) |
| --- | --- | ---: | ---: | ---: |
| 8 | Forward | 4.856461 | 0.213950 | 238.011 / 292 |
| 8 | Forward + reverse | 15.911914 | 0.741333 | 666.034 / 804 |
| 32 | Forward | 4.459965 | 0.215621 | 238.011 / 292 |
| 32 | Forward + reverse | 15.118165 | 0.729725 | 306.104 / 372 |
| 96 | Forward | 4.219480 | 0.215064 | 198.011 / 256 |
| 96 | Forward + reverse | 14.655525 | 0.711200 | 226.289 / 276 |

At interval 8, extending from 24 to 96 steps raised reverse peak allocated
memory from 306.033 to 666.034 MiB. Larger intervals reduced the measured peak
for this 96-step fixture; this does not establish an optimal interval for other
durations or physics. Interval 96 still selects checkpointed execution in one
block, rather than disabling checkpointing. These measurements do not establish
long-time stability or the memory limit of an 8 GiB device.

```sh
python studies/fdtd/native_cuda/native_cuda.py --sizes 64 --steps 96 --checkpoint-steps 8 --repeat 3 --output artifacts/native-cuda-96-cp8.json
python studies/fdtd/native_cuda/native_cuda.py --sizes 64 --steps 96 --checkpoint-steps 32 --repeat 3 --output artifacts/native-cuda-96-cp32.json
python studies/fdtd/native_cuda/native_cuda.py --sizes 64 --steps 96 --checkpoint-steps 96 --repeat 3 --output artifacts/native-cuda-96-cp96.json
```

## Operator profiling

The optional profiling instrument completed actual 32³ forward, 64³ forward
and 64³ forward-plus-reverse CUDA solves, capturing CUDA activity in each case.
It records CPU dispatch and CUDA activity with shapes and memory, exports a
Chrome trace and an operator table, and keeps an attempt receipt on failure.
This verifies the instrument; profiler durations are not benchmark results.
Inclusive operator times overlap, and shape/memory tracing changes overhead and
tensor lifetimes. Do not sum them into an end-to-end time or use their allocator
peaks in place of the uninstrumented cost study.

```sh
python studies/fdtd/native_cuda/profile_native_cuda.py --size 32 --steps 24 --checkpoint-steps 8 --mode forward --device cuda:0 --output-dir artifacts/native-cuda-profile
```

Choose a new output directory for every attempt; the instrument refuses to
overwrite an existing directory. The trace and reports are generated locally.

### Reusing differentiable Bloch factors

The native core now forms forward and inverse Bloch wrap factors once per call,
then reuses them in curls and monitor interpolation throughout that trajectory.
The factors remain in the differentiation graph and are recomputed from every
call's live phases. There is no special branch for zero phase and no detached
constant cache. A new JAX scan-cache test changes all three phases, confirms
program reuse, and checks that the second call's E/H and Fourier fields are
byte-identical to a cold compile at those new phases. The targeted validation
passed 70 physics/derivative cases, 14 existing scan cases and that new case.

A second timing experiment used the same 24-step fixture, checkpoint interval
8, one warm-up and **five** synchronized repetitions before and after the change.
Forward gradient mode remained disabled and CPU Torch remained at one thread.
All CPU/CUDA rows passed their applicable numerical checks. These CUDA values
are measured medians with the observed minimum–maximum range in brackets:

| Grid | Mode | Before (s) | After (s) |
| --- | --- | ---: | ---: |
| 32³ | Forward | 0.034085 [0.031873–0.037173] | 0.029103 [0.028747–0.034839] |
| 32³ | Forward + reverse | 0.176703 [0.141419–0.195175] | 0.161988 [0.146971–0.209357] |
| 64³ | Forward | 0.058163 [0.056666–0.062642] | 0.056074 [0.055931–0.061953] |
| 64³ | Forward + reverse | 0.187348 [0.179126–0.193288] | 0.184639 [0.182334–0.204737] |

The medians decreased in these runs, but every before/after range overlaps,
especially in reverse mode. Five repetitions on one machine do not establish a
universal improvement or a statistical confidence interval. The recorded
`grid.py` SHA-256 is
`c1dbb2967945d83918824177c9009cd443c6127bc961337976d1b8f197228742`
before and
`5609d25df843e7e670ae2f426b2962939fb10164bb0b488733280ca3acc3d696`
after; the instrument records source and script hashes with each new run.

In the separate 32³ forward traces, raw CUDA kernel events decreased from
2,420 to 2,138 and `aten::reciprocal` calls from 148 to 7. These counts show
the removed repeated work; they are not elapsed-time estimates. CUDA kernel
rows and CPU operator attribution can describe the same work and must not be
added together.

```sh
python studies/fdtd/native_cuda/native_cuda.py --sizes 32 64 --steps 24 --checkpoint-steps 8 --repeat 5 --output artifacts/native-cuda-factor-reuse.json
```
