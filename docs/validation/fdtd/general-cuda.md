# Bounded general-scene CUDA execution

Written: 2026-10-18

Targets:

- `src/wavebit/fdtd/_api.py`
- `src/wavebit/fdtd/numerics.py`
- `src/wavebit/fdtd/_runtime.py`
- `src/wavebit/fdtd/_kernel/observe.py`
- `tests/fdtd/test_general_cuda.py`
- `tests/fdtd/test_general_cuda_observations.py`
- `tests/fdtd/test_general_cuda_checkpoint.py`
- `tests/fdtd/test_general_cuda_airy.py`
- `tests/fdtd/test_general_cuda_higher_order.py`
- `tests/fdtd/test_fd_stage3_backends.py`
- `tests/api/test_fdtd_checkpoint_jobs.py`
- `studies/fdtd/general_cuda/checkpoint_cost.py`
- `studies/fdtd/general_cuda/monitor_cost.py`

The general `wavebit.fdtd` API admits a restricted Torch CUDA route in double
precision. Host planning and the existing physical admission rules remain in
place; time stepping and differentiable observations use CUDA. The local
validation ran 35 general-scene and observation cases, including CPU-only
admission checks and actual CUDA execution. It does not qualify every general
FDTD feature or establish continuum convergence. The checkpoint study below
measures one conductor fixture, not general-scene performance broadly.

## Admitted inputs

Pass `backend='torch', device='cuda:0'` to `check`, `prepare` or `run`. Prepared
execution retains the device and validates live inputs on every reuse. Inputs
may originate on CPU; differentiable conversion keeps gradients connected to
the original leaves. Other non-CPU backends and single precision remain refused.

| Part | CUDA admission |
| --- | --- |
| Grid and duration | Existing uniform cubic grid; fixed duration; `discretisation='voxel'`; no `geometry_ranges` |
| Boundaries and padding | `Periodic` and `Open` faces; `PML` padding |
| Materials | Existing nondispersive `Material` and `ConductiveMaterial`, with their ordinary material checks |
| Structures | Cell-snapped `Slab` only |
| Sources | `PlaneWave` or initial fields |
| Observations | `Record`, `Field`, `Flux` and final fields |

Initial E/H patterns are static host declarations: use NumPy arrays, CPU tensors
without `requires_grad`, or host callables. Direct CUDA tensor patterns and
differentiable tensor patterns are not admitted. `InitialField.amplitude` is
live and keeps its gradient through CPU-to-CUDA conversion; this distinction
also applies to prepared reuse. The prepared contract does not advertise ranged
geometry on CUDA.

`Decay`, the alternative `Absorber`, ranged or smoothed geometry, dispersive
materials, Bloch/wall/symmetry/invariant-axis faces, dipole/TFSF sources,
diffraction and far-field monitors are outside this CUDA route. Unsupported
combinations return a `cuda_scope` issue with the relevant public field path
before planning. Admission does not waive sampling, CFL, material, padding or
other scientific checks. A passed `check` or `prepare` advances no fields.

## Evidence and its limits

The original general-scene and observation files account for the 35 cases above.
The integrated run including 21 checkpoint and four Job cases passed all 60
cases. A separate default CPU/JAX lifecycle, unified Job and migration regression
passed 53 cases.

| Test | Independent control or device comparison |
| --- | --- |
| Uniform conductor, periodic box, 20 steps | Exact SI conductivity recurrence for E and its time record, unchanged H; relative `2e-14` |
| Conductivity and initial-amplitude derivatives | Closed-form derivatives of the recurrence to relative `2e-12`, with both CPU and CUDA parameter leaves |
| Slab with CPML, plane-wave source, records, Fourier fields and flux | NumPy CPU parity with the existing peak-normalized infinity budget `5e-13`; live prepared reuse, changed inputs and retained old results |
| Slab permittivity and source derivatives | NumPy material central difference at relative `1e-6`, absolute `1e-8`; exact source quadratic identity at relative `2e-12` |
| Record, Field and Flux objectives, 161 steps | CPU/CUDA objective parity to relative `5e-13`; source quadratic identity to `2e-11`; nonzero material central differences at two step sizes to `2e-6` |
| Slabs normal to x, y and z; open 3D pulse; diagonal epsilon and mu with/without conductivity | CPU field parity to peak-normalized `5e-13`; six nonzero box-flux faces compared independently to prevent cancellation hiding orientation errors |
| Mixed-device material and source leaves | Epsilon/mu/conductivity values and derivatives match CPU; CPU phase and CUDA delay derivatives match nonzero NumPy central differences at two step sizes to relative `2e-6`; leaf-device ownership retained |
| Static initial patterns and precision | Mutating a prepared CPU tensor pattern invalidates reuse; gradient-bearing patterns refused; float32 host patterns retain precision warnings, become float64 execution fields, preserve input ownership and return the amplitude gradient to the original float32 CPU leaf |
| Input and reuse controls | Unsupported scopes refused before planning; preparation advances no fields; incompatible/invalid reuse refused; invalid live differentiable material preserves failed validity and NaNs |

The observable objectives use fixed SI scale factors; they do not normalize by
a live field norm or detach field-dependent values. These comparisons verify
the same discrete problem and its derivatives. CPU parity is not an independent
optical solution. Mesh, finite-duration and precision warnings remain visible;
no tolerance or warning policy was relaxed. The axis,
3D pulse and diagonal-material checks extend finite-case device coverage, not
the set of admitted feature classes or a continuum accuracy claim.

The [native CUDA record](native-cuda.md) gives the measured Windows/Python/Torch
and GPU environment and isolated installation. Its performance table belongs to
the native engine, not this general-scene route. General execution defaults to
the existing plain time scan. The memory results below depend on which inputs
are differentiated; they establish no universal saving, compilation, long-time
stability or multi-GPU claim.

## Independent public slab gate

`test_general_cuda_airy.py` adds an independent optical gate through the public
`PlaneWave`, `Field` and `Flux` interfaces. All six cases passed: three CPU and
three actual CUDA cases. This separate run is not included in the 60 cases
above. Its independent optical evidence is limited to the planar fixture below.
A later run after skipping unused DFT phases and adding the no-grad runtime guard passed all
three general CUDA optical cases again, together with the two separate
periodic top/bottom slab cases. Existing mesh warnings remained visible.

The fixture is a 0.2 um, epsilon-2.25 film in vacuum, with transverse periodic
faces, open z faces and 24-cell PML. The five wavelengths are 0.80, 0.90, 1.00,
1.15 and 1.30 um. Slab and empty-domain runs use the same source and probes;
complex incident fields are subtracted before squaring, and measured reference
fields cancel the source spectrum. Reference amplitudes must exceed 0.1 of
their band peak. Discrete propagation phases move r/t to the slab faces.

Both complex r and t must match the established independent discrete Yee Airy
equations within maximum absolute `2e-6` and relative `1e-5` error. Flux-derived
R/T must match squared field amplitudes to absolute `2e-5`; front/back reference
flux uses the same budget. A vacuum-film control requires exactly zero
reflection and discrete transmission-phase agreement to relative `1e-12`.
These are existing accuracy budgets, not tolerances fitted to this CUDA route.

At both 0.05 and 0.025 um spacing, extending the Fourier window from 24 to 32
source periods must change r/t by less than the same discrete budgets. Halving
spacing must reduce the half-cell-corrected continuum reflection error by a
factor between 3.2 and 5; the coarse error must remain above 0.03. This checks
the established staircase/dispersion correction rather than erasing it. The
gate covers one normal-incidence lossless film, not general geometry,
dispersion, long-time stability or optical derivatives.

## Opt-in reverse-mode checkpoints

`Numerics(checkpoint_steps=K)` selects a positive static integer block length
for fixed-duration Torch CUDA execution. The default `None` preserves the
ordinary scan. CPU backends, other devices and `Decay` are refused for explicit
checkpoint settings; invalid values, booleans (including boolean tensors), and
differentiable or transform-carried block lengths are rejected before planning.
The surrounding CUDA feature limits still apply.

This policy recomputes blocks during reverse mode; it is not a memory cap.
Observation and source histories remain retained. A block longer than the
general-scene run leaves only the ordinary tail, unlike the native runner's
one-block behavior. Changing the schedule requires preparing new numerics.
`report.resolved['checkpoint_steps']`,
`prepared.contract['resolved_checkpoint_steps']` and
`result.provenance['checkpoint_steps']` expose the selected value.

The checkpoint file passed 21 cases, including four actual CUDA cases. Blocks
1 and 64 on a 17-step conductor preserve the independent SI recurrence and
conductivity/initial-amplitude derivatives. A 163-step slab with block 11 has a
nine-step tail; its final fields, Record, Field and Flux objectives and material/
source derivatives agree with the ordinary scan (values `5e-13`, gradients
`1e-12` relative). Multiple backward queries retain the graph correctly, and
two prepared reuses with independently created leaves preserve old results.
The source quadratic identity remains independently checked at `2e-11`, and
invalid live material still returns failed validity and NaNs.

The admission cases include three regressions rejecting boolean Torch tensors
as block lengths.

Four Job tests preserve the fixed pre-existing default key and round-trip
explicit schedules. `checkpoint_steps=None` is omitted from the v1 Job envelope;
an explicit integer is serialized. This establishes the opt-in contract and
bounded derivative equivalence. General-scene memory measurements are separate
from these acceptance tests.

## Bounded second derivatives

Two actual CUDA cases in `test_general_cuda_higher_order.py` passed for a
17-step uniform periodic conductor in double precision: a plain scan and block
3 with a two-step tail. The objective is the mean squared final Ey,
`J = A**2 * q**34`, where `q = (1 - sigma*f)/(1 + sigma*f)` and
`f = dt/(2*epsilon0*epsilon)`. The scalar rational recurrence supplies the
independent value, gradient and full Hessian in amplitude/conductivity order,
including both orders of the mixed derivative, at relative `2e-12` and zero
absolute tolerance. It uses the run's dt and the shared epsilon0 constant;
this is a derivative oracle, not independent validation of the time step or
unit constants.

Amplitude leaves originate on CPU and conductivity leaves on CUDA. Eager
`torch.autograd.grad(..., create_graph=True)` followed by double backward
returns derivatives to the original leaf devices. Reuse with changed leaves
matches fresh preparation and preserves the first result unchanged. These
two cases are separate from the earlier first-derivative test counts. They do
not establish Hessians for arbitrary geometry, CPML, other source/material
families, or `torch.func` transformations.

## Measured checkpoint cost

The prepared conductor fixture uses the environment listed in the native
record, one Torch CPU dispatch thread, CUDA float64 fields, spacing 0.05 um,
epsilon 2.25, conductivity 600 S/m and initial E = (0, 1, 0) with amplitude
1.2. All faces are periodic. It retains one point Ey record and final fields
over 163 steps. The objective is the sum of squared final electric fields.
The two graphs differentiate amplitude alone or both amplitude and conductivity;
their live scalar leaves are on CUDA.

Each configuration has one warm-up and three synchronized repetitions. Times
include live leaf creation, prepared execution, objective and reverse mode;
they exclude preparation, host scalar copies and JSON writing. Memory is the
peak across those repetitions after warm-up, garbage collection and allocator
cache release. Baseline allocated memory was zero in every row. Allocated and
reserved peaks are Torch allocator quantities, excluding driver/display memory;
CPU memory was not measured. These timings are not directly comparable to the
native table, whose timing includes preparation and whose physics differ.

All 16 measured configurations completed with every applicable check passing
on every repetition. The independent SI recurrence is `q = (1 - sigma*f) /
(1 + sigma*f)`, with `f = dt / (2*epsilon0*epsilon)` and objective
`J = size**3 * amplitude**2 * q**(2*steps)`. Both `2*J/amplitude` and the analytic
conductivity derivative using `dq/dsigma = -2*f/(1 + sigma*f)**2` are checked
at relative `2e-12`, absolute zero. This controls both differentiated graphs
independently of their agreement with a plain scan.

Times are median seconds; memory is allocated/reserved MiB (2**20 bytes).
`None` is the plain scan; block 8 makes 20 blocks and a three-step tail, block
32 makes five blocks and a three-step tail, and block 200 leaves only the
ordinary 163-step tail.

| Grid | Block | Amplitude time | Amplitude memory | Amplitude + conductivity time | Amplitude + conductivity memory |
| --- | --- | ---: | ---: | ---: | ---: |
| 16³ | None | 0.487020 | 1.037 / 2 | 0.518653 | 31.321 / 32 |
| 16³ | 8 | 0.708443 | 5.015 / 6 | 0.737885 | 6.207 / 8 |
| 16³ | 32 | 0.673048 | 2.174 / 4 | 0.697212 | 7.865 / 10 |
| 16³ | 200 | 0.519046 | 1.037 / 2 | 0.499353 | 31.321 / 32 |
| 32³ | None | 0.499816 | 8.255 / 12 | 0.537487 | 250.509 / 334 |
| 32³ | 8 | 0.667726 | 39.796 / 54 | 0.728847 | 49.301 / 66 |
| 32³ | 32 | 0.645770 | 17.268 / 24 | 0.661251 | 62.771 / 84 |
| 32³ | 200 | 0.474110 | 8.255 / 12 | 0.514292 | 250.509 / 334 |

For the 32³ material graph, block 8 reduced allocated peak by a factor of 5.08
while increasing median time from 0.537487 to 0.728847 seconds. For the same
physical problem with only amplitude differentiated, it increased allocated
peak from 8.255 to 39.796 MiB. Which leaves require derivatives changes the
storage/recomputation tradeoff. This fixture selects no universally optimal
block and guarantees no saving for other fields or observation/source histories.

The table transcribes two local JSON runs; raw machine records are not
distributed with this page. The amplitude-only run used instrument SHA-256
`ffb6cad9975326b2bcd1ddd240dd4351df4ea074cbce429675ff747c1d2a2263`;
the conductivity run used
`997bf8c7e4de01e6dca1e2469d84a6305aa6e660a32ce1a21388c65a47496848`,
which adds the material derivative control.

## Computing only requested DFT phases

Each observation call now skips the phase of any field it does not request:
an Ey-only monitor no longer computes an unused H phase. Existing phase sharing
across components of the same field is preserved. Electric samples retain time
`(n+1)*dt` and magnetic samples retain `(n+0.5)*dt`. There is no recurrence approximation or cache across time
steps. After this change, 37 CPU checks passed, including 21 new direct offline
quadrature/phase-sensitive derivative cases plus frozen observation references
and the existing time-level mutant. Electric-only, magnetic-only and mixed
components use real and complex input histories across NumPy, JAX and Torch.

The bounded `monitor_cost.py` experiment used the same RTX 4060 Ti environment,
one CPU dispatch thread and a 4×4×84 grid, 128 steps, 12-cell PML, two Ey Field
and two Flux monitors at three wavelengths. Both runs used `torch.no_grad()`
and the plain scan. After one warm-up, five synchronized prepared runs supplied
the timings below; a **separate single instrumented run** supplied operator
counts. Timing includes live validation, lowering and result adaptation but
excludes initial preparation, result checks, host copies and profiling.

| Quantity | Before skipping unused phases | After skipping unused phases |
| --- | ---: | ---: |
| Median seconds [minimum–maximum] | 0.591719 [0.564726–0.605297] | 0.557161 [0.537056–0.638640] |
| Profiled `aten::exp` calls | 1025 | 769 |
| Profiled `aten::mul` calls | 5888 | 5120 |
| Peak allocated / reserved bytes | 420864 / 2097152 | 420864 / 2097152 |

The 256 fewer exponential and 768 fewer multiplication calls match omission
of two unused H phases per step over 128 steps. All placement, dtype, finite and
runtime-validity checks passed. The recorded field-maximum and flux summaries
were exactly equal before and after; this is not a full-field equality test.
The short window contains only a weak early pulse and supplies no independent
optical accuracy claim. Allocator baselines were zero; peaks exclude driver,
display and CPU memory. Profiler shape/memory recording changes cost and tensor
lifetimes, and inclusive CPU/device times must not be added.

The timing ranges overlap. The observed medians do not establish a statistically
significant or universal speedup, and allocated peak did not improve. The table
transcribes local records rather than distributing raw machine traces. Recorded
`src/wavebit/fdtd/_kernel/observe.py` SHA-256 values are
`23e5a6878f8ee14457c1cbcec422fba4f3b197ed9bb55ef4175b1b12f8988a83`
before and
`709828e0ffa334200d51ab100476f403baeaf05bd2ecb71c9b0c9f0d80cb9f01`
after skipping unused phases. This experiment precedes any change to no-grad checkpoint
dispatch.

## Forward-only execution with a checkpoint policy

When Torch gradient recording is disabled by `torch.no_grad()` or
`torch.inference_mode()`, the scheduled scan takes the ordinary forward path.
It avoids the block copies and checkpoint wrappers used for backward
recomputation. The requested checkpoint policy remains in the report, prepared
contract and result provenance; it is used again when the same prepared object
runs with gradients enabled. This does not change admission or turn a forward
result into a differentiable result afterward.

A later run passed 25 cases: the expanded 23-case checkpoint file and two
observable derivative cases. This is a separate result, not a revised count
for the earlier 60-case run. The two new context cases use 163 steps and block
11, comparing final E/H, Record, Fourier E/H, Flux and source histories with
the plain scan under both disabled-gradient contexts. CPU amplitude and CUDA
material leaves remain caller-owned and unchanged. A later differentiated
reuse with new leaves checks four objectives, retained backward graphs,
source quadratic identities, device ownership of gradients and immutable old
results. Values retain relative `5e-13`, derivatives `1e-12`, and the source
identity `2e-11` budgets. Four existing JAX/Torch checkpoint spectral-gradient
cases also passed after the runtime change.

A separate CUDA experiment retained the same requested-phase computation and used the same short
monitor fixture with a 16×16×84 grid, 128 steps and checkpoint block 8 under
`torch.no_grad()`. One warm-up and five synchronized uninstrumented repetitions
used the timing boundary described above. Both versions passed every numerical
check, with exactly equal recorded field-maximum/flux summaries.

| Quantity | Before forward-path guard | After forward-path guard |
| --- | ---: | ---: |
| Median seconds [minimum–maximum] | 0.584975 [0.553745–0.626643] | 0.548744 [0.532387–0.580045] |
| Peak allocated bytes | 7394304 | 5763072 |
| Peak reserved bytes | 8388608 | 8388608 |

The timing ranges overlap, so these medians do not establish a general or
statistically significant speedup. Allocated peak decreased for this measured
forward-only fixture; reserved peak did not. Both allocator baselines were
zero, and the same driver/display/CPU-memory exclusions apply. The short-window
summary equality is neither a full-field comparison nor optical qualification.
Local records identify `src/wavebit/fdtd/_runtime.py` SHA-256 as
`048e38423a374b7880e84f6abb8fef7e5cc2a077a7bce739e4b911bc6e57dcfa`
before and
`6a3a0849bb5527e109b6c7c91d02e9c668c955a6f8a784bab5841f1f1467d11d`
after the guard; raw machine records are not distributed here.

## Reproduction

Use the isolated CUDA environment and confirm the imported source checkout as
described in the [native setup](native-cuda.md#environment-and-reproduction).

```sh
python -m pytest tests/fdtd/test_general_cuda.py tests/fdtd/test_general_cuda_observations.py tests/fdtd/test_general_cuda_checkpoint.py tests/api/test_fdtd_checkpoint_jobs.py -q
python -m pytest tests/fdtd/test_general_cuda_airy.py --run-slow -q
python -m pytest tests/fdtd/test_general_cuda_higher_order.py -q
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 16 32 --steps 163 --blocks 0 8 32 200 --repeat 3 --output artifacts/general-checkpoint.json
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 4 --repeat 5 --device cuda:0 --profile --output-dir artifacts/general-monitor-profile
python studies/fdtd/general_cuda/monitor_cost.py --steps 128 --transverse-cells 16 --checkpoint-steps 8 --repeat 5 --device cuda:0 --output-dir artifacts/general-monitor-no-grad
```

The study's `--blocks 0` maps to public `checkpoint_steps=None`; zero is not a
valid public block length. The default measures both derivative modes; use
`--parameters amplitude` or `--parameters material` to select one. The study
refuses an existing output path and preserves attempts and failed checks in
its JSON. The repository study guide is `studies/fdtd/general_cuda/README.md`.

CUDA numerical cases skip when the required backend or device is unavailable;
the hardware-independent scope checks can still pass. Keep pass and skip counts
separate when reporting a CPU-only run. This local evidence does not replace
required CI on the reviewed candidate or establish release integration.
