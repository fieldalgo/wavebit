# Bounded general-scene CUDA execution

Written: 2026-10-18

Targets:

- `src/wavebit/fdtd/_api.py`
- `src/wavebit/fdtd/_sources.py`
- `src/wavebit/fdtd/_numerics.py`
- `src/wavebit/fdtd/_runtime.py`
- `src/wavebit/fdtd/_prepare/params.py`
- `src/wavebit/fdtd/_kernel/observe.py`
- `tests/fdtd/test_general_cuda.py`
- `tests/fdtd/test_general_source_finite.py`
- `tests/fdtd/test_general_source_band_guard.py`
- `tests/fdtd/test_general_source_resolution_guard.py`
- `tests/fdtd/test_general_source_waveform_guard.py`
- `tests/fdtd/test_general_cuda_initial_patterns.py`
- `tests/fdtd/test_general_cuda_optimizer.py`
- `tests/fdtd/test_general_cuda_region_contract.py`
- `tests/fdtd/test_general_cuda_region.py`
- `tests/fdtd/_region_yee_reference.py`
- `tests/fdtd/test_general_cuda_observations.py`
- `tests/fdtd/test_general_cuda_checkpoint.py`
- `tests/fdtd/test_general_cuda_airy.py`
- `tests/fdtd/test_general_cuda_higher_order.py`
- `tests/fdtd/_pulse_yee_reference.py`
- `tests/fdtd/test_general_cuda_pulse_reference.py`
- `tests/fdtd/test_general_cuda_pulse_higher_order.py`
- `tests/fdtd/_pulse_fit_reference.py`
- `tests/fdtd/test_pulse_fit_reference.py`
- `tests/fdtd/test_general_cuda_pulse_optimizer.py`
- `tests/fdtd/test_general_cuda_cpml_higher_order.py`
- `tests/fdtd/test_general_cuda_dipole.py`
- `tests/fdtd/test_general_cuda_dipole_conductive.py`
- `tests/fdtd/test_general_cuda_dipole_radiation.py`
- `tests/fdtd/test_general_cuda_walls.py`
- `tests/fdtd/test_general_cuda_wall_ad.py`
- `tests/fdtd/test_general_cuda_bloch_modes.py`
- `tests/fdtd/test_general_cuda_bloch_ad.py`
- `tests/fdtd/test_general_cuda_bloch_contract.py`
- `tests/fdtd/test_general_cuda_bloch_planewave.py`
- `tests/fdtd/test_general_cuda_bloch_observations.py`
- `tests/fdtd/test_general_ade_modes.py`
- `tests/fdtd/test_general_ade_ad.py`
- `tests/fdtd/test_general_ade_record_ad.py`
- `tests/fdtd/test_general_ade_units.py`
- `tests/fdtd/test_general_ade_contract.py`
- `tests/fdtd/test_general_ade_memory.py`
- `tests/fdtd/test_general_ade_arithmetic.py`
- `tests/fdtd/test_general_ade_energy.py`
- `tests/fdtd/test_ade_cost_study.py`
- `tests/fdtd/test_fd_stage3_backends.py`
- `tests/api/test_fdtd_checkpoint_jobs.py`
- `studies/fdtd/general_cuda/ade_cost.py`
- `studies/fdtd/general_cuda/optimizer_reuse.py`
- `studies/fdtd/general_cuda/checkpoint_cost.py`
- `studies/fdtd/general_cuda/monitor_cost.py`

The general `wavebit.fdtd` API admits a restricted Torch CUDA route in double
precision. Host planning and the existing physical admission rules remain in
place; time stepping and differentiable observations use CUDA. The local
initial validation ran 35 general-scene and observation cases, including CPU-only
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
| Grid and duration | Existing uniform cubic grid; fixed duration; `discretization='voxel'`; no `geometry_ranges` |
| Boundaries and padding | `Periodic`, `Open`, `PEC` and `PMC`; static `Bloch` with only Bloch/Periodic/Open companions; `PML` padding on open faces |
| Materials | Existing nondispersive `Material` and `ConductiveMaterial`; the separate homogeneous single-pole ADE scope below |
| Structures | Cell-snapped `SnappedSlab`; one strictly interior `SnappedBox` under the separate restrictions below |
| Sources | `PlaneWave`, fixed-center electric/magnetic `CurrentDipole` with `bspline` or `tent` stamp, or initial fields; Bloch scenes admit only PlaneWave/initial fields |
| Observations | `TimeSeriesMonitor`, `FieldMonitor`, `FluxMonitor` and final fields |

Initial E/H patterns are static host declarations: use NumPy arrays, CPU tensors
without `requires_grad`, or host callables. Direct CUDA tensor patterns and
differentiable tensor patterns are not admitted. `InitialField.amplitude` is
live and keeps its gradient through CPU-to-CUDA conversion; this distinction
also applies to prepared reuse. The prepared contract does not advertise ranged
geometry on CUDA.

A callable is evaluated during each run, including reuse of the same callable.
Its returned pattern must satisfy the same static host/non-carried restriction,
including nested component values; callable identity alone does not certify its
result. Raw H callback results are checked before SI normalization, including
under `no_grad` and inference contexts, so unit conversion cannot hide a carried
pattern. `check` and `prepare` do not execute these callbacks. Live initial
amplitude remains supported. This CUDA restriction does not change CPU/JAX or
legacy initial-pattern differentiation.
The dedicated initial-pattern tests passed 43 cases (22 CPU and 21 CUDA),
including same-callable invalid-result refusal and subsequent valid recovery.

`CurrentDipole.current_moment` is a current moment in A·m for electric sources or V·m
for equivalent magnetic sources. Its strength parameter is `current_moment`;
`PlaneWave.amplitude` instead specifies electric-field amplitude in V/m.
Current moment, phase and delay retain live graphs.
The `CurrentDipole` center and stamp are static: changing them requires new preparation.
Position derivatives, support crossing a periodic seam, touching a conducting wall, or in PML
remain refused by the existing source checks.

For CUDA, Bloch `phase` or `k` is static; live boundary derivatives are outside
this scope, including forward-AD duals and transformed/batched wrappers. Ordinary
host scalar tensors without derivatives remain valid static declarations.
Changing its value or phase/k representation requires preparation
again. Bloch with PEC/PMC walls or CurrentDipole sources remains refused. Oblique
PlaneWave angles are radians; `Bloch.from_source(source, units=...)` supplies the
fixed transverse wavevector at the pulse's centre wavelength, not a constant
angle across its spectrum.

`DecayStop`, the alternative `ConductiveAbsorber`, ranged or smoothed geometry, dispersive
scenes outside the homogeneous ADE scope, symmetry/invariant-axis faces, TFSF sources,
diffraction and far-field monitors are outside this CUDA route. Unsupported
combinations return a `cuda_scope` issue with the relevant public field path
before planning. Admission does not waive sampling, CFL, material, padding or
other scientific checks. A passed `check` or `prepare` advances no fields.

## Static interior SnappedBox

The separate bounded SnappedBox route requires exactly one ordinary
`Structure(name, shape=SnappedBox(...), material=...)`, six `Periodic` faces and no external sources.
Each SnappedBox face snaps to the nearest cell boundary on the actual rounded grid.
On every axis its filled interval must satisfy `0 < lo < hi < n`: touching a
periodic seam, spanning a whole axis or collapsing to an empty interval is
outside this route. Multiple SnappedBoxes, SnappedBox/SnappedSlab mixtures, continuation,
`Box`, smoothing and shape derivatives are excluded. The existing SnappedSlab and CPU
routes keep their separate contracts.

Both background and body use scalar, real, nondispersive `Material` or
`ConductiveMaterial`; `mu=None` means one. Their existing finite/positive/CFL
checks still apply on preparation and reuse. The SnappedBox's center and size and
the initial spatial patterns are static. A changed shape or pattern requires
new preparation; changing compatible scalar material values or the real initial
amplitude retains the existing live-input path.

At least one initial E/H pattern must be present as a plain finite real NumPy
array of actual shape `(3,nz,ny,nx)`; the other may be `None`. Tensor patterns,
callables and component lists are excluded here. Only ordinary `TimeSeriesMonitor` and
final-field outputs are admitted. Fixed duration, double precision, voxel
sampling and the existing optional checkpoint policy apply. SnappedBox scenes with
Open/PML, PEC/PMC, Bloch, ADE, sources, FieldMonitor/FluxMonitor or ranged geometry remain
outside this bounded route. Admission does not establish interface convergence
or general geometry differentiation.

### Independent discrete SnappedBox reference

The reference constructs the occupied-cell mask independently from declared
geometry and uses explicit modulo-index forward/backward Yee differences. With
`h=Z0*H`, `r=sigma*dt/(2*epsilon0)`, `q=(epsilon-r)/(epsilon+r)` and
`g=c0*dt/(epsilon+r)`, the update is
`h_new=h-c0*dt/mu*curl_plus(E)` followed by
`E_new=q*E+g*curl_minus(h_new)`. Body epsilon, mu, conductivity and initial
amplitude tangents are propagated by an independently written chain rule,
including the conductive denominator and magnetic coefficient. This reference
does not call the product curl, material-painting, coefficient or AD helpers.
It validates the stated discrete update, not a continuum interface solution.

The declared base fixture uses dx 0.05 um, domain size `(0.61,0.49,0.71)` um and
center `(0.1,-0.15,0.2)` um. Independent upward rounding gives `(nx,ny,nz)=(13,10,15)`
and actual lower corner `(-0.225,-0.4,-0.175)` um. SnappedBox size
`(0.305,0.225,0.365)` um and center `(0.0375,-0.2025,0.1625)` um give half-open
index ranges x `[2:8]`, y `[2:6]`, z `[3:10]`, occupying 168 cells. Three cyclic
axis arrangements and equivalent um/nm declarations exercise orientation and
unit conversion. Initial real fields are sampled at each component's Yee
location; all six components are observed as TimeSeriesMonitor histories and final fields.

The checks use 17 and 41 steps, with checkpoint-7 tails of three and
six steps. Full fields and TimeSeriesMonitor histories use peak-normalized `2e-12`, clocks
use relative `2e-14`, and weighted squared field/time-series norms and their four nonzero
material/amplitude derivatives use relative `2e-12`, absolute zero. A
`|parameter_scale*gradient|/loss > 1e-6` floor prevents vacuous derivative
comparisons. Mask shifts, floor-versus-nearest rounding, axis/wrap/curl-sign
mistakes, omitted mu and conductive-denominator/tangent terms must each exceed
a `1e-4` separation floor. All eight mutants passed this predeclared gate;
the smallest separation was `0.0111898253`. Two central-difference step sizes
independently check the analytic tangents at relative `2e-6`; their largest
observed discrepancy was `1.15306194e-6`.

The body-material reuse cases use `(epsilon,mu,sigma,A)=(2.25,1.3,600,1.2)`
and `(1.8,1.1,900,0.7)`, with conductivity in S/m. CUDA material leaves and a CPU
amplitude leaf remain connected to the same prepared scene. Input patterns,
leaf ownership and earlier outputs remain unchanged. Separate lossless cases
exercise `Material`. A nonvacuum background `(epsilon,mu,sigma)=(1.4,1.1,80)`
uses fresh um/nm runs at rotation 1 and 41 steps, comparing background-material
and amplitude derivatives; this is a unit-pair check, not background prepared
reuse.

| Qualification | Host cases | Actual CUDA cases |
| --- | ---: | ---: |
| Independent mask/Yee/tangents, fields, time series, reuse and units | 25 | 17 |
| Admission, conversion, refusal and same-leaf invalid/recovery contract | 51 | 6 |
| Unique new cases | 76 | 23 |

All **99 unique new cases passed**: the host total comprises 44 device-free
checks and 32 actual CPU cases. The 23-case CUDA run took 10.40 s and emitted
25 warnings. Repeated qualification checks do not add to these counts; 25
existing CPU regressions passed separately.

The largest recorded body-reuse field/time-series error was
`3.97962361e-15`. An independent audit of the recorded body and background
objective gradients found maximum relative error `9.28604067e-14`, below the
unchanged `2e-12` budget. Lossless and unit-pair field assertions also passed,
but their errors were not individually recorded, so the body-reuse maximum is
not a maximum over every field case. These results qualify the bounded discrete
first-derivative contract; they do not establish continuum interface accuracy,
shape derivatives or arbitrary SnappedBox combinations.

## Finite excitation values and prepared recovery

Source amplitude, phase and delay, and `InitialField.amplitude`, must be finite.
Source `Band.wavelength` and `Band.df` must also be positive. Complex amplitudes
are checked in both real and imaginary parts only where the existing field
contract admits complex values; real-plan complex amplitudes remain refused.
For a `CurrentDipole`, the public strength is `current_moment`, from which lowering derives
the source amplitude.

Previously, changing a live source or initial-amplitude leaf to NaN could yield
NaN fields while `result.validity.checks['runtime_stability']` remained true.
The excitation-domain guard now contributes to the existing runtime validity
factor. Existing constructors may refuse readable invalid declarations earlier;
otherwise invalid excitation leaves yield failed runtime validity and NaN
fields. An optimizer-style in-place update does not make a prepared structure
incompatible by itself: restore valid leaf values and run again with the same
preparation. Earlier returned results must remain unchanged.

Valid concrete excitation values leave the incoming stability guard unchanged.
This does not expand initial-pattern, source or complex-field admission. The
check covers the input domains above, not every overflow in derived waveform
arithmetic. The separate live Band cutoff guard below adds the spectral
inequality; the subsequent pole-free resolution guard adds a separate spatial
certificate. Numerical and
derivative qualification remains specific to the exercised cases.

`test_general_source_finite.py` passed **77 new cases**: 44 CPU and 33 actual
CUDA cases. Public valid→invalid→valid runs update the same source amplitude,
phase or initial-amplitude leaf in place, preserve its identity and prior
outputs, and recover the original fields and source history exactly. Direct
guard cases cover NaN and both infinities, nonpositive wavelength/bandwidth,
complex parts, mixed-device leaves, valid concrete guard identity, logical
Torch views and bfloat16 conversion. Torch forward/batched transforms and JAX
JIT/derivatives are helper-level controls; they do not qualify arbitrary full
solver transforms or complex CurrentDipole admission.

Separately, two existing CPU CPML amplitude/phase Hessian cases and three
existing CPU source-gradient/waveform cases passed across their Torch/JAX and
CurrentDipole/TFSF/PlaneWave paths. Ten existing actual CUDA cases passed: four CPML
Hessians, two complex Bloch observation Hessians, two exact conductor Hessians,
and two conductivity/initial-amplitude derivative cases. Their original
numerical budgets remain unchanged; the CUDA run retained 46 mesh/window
warnings. These are bounded regressions, not full CI or optimizer convergence
qualification.

### Live Band carrier cutoff

A carried wavelength or bandwidth now retains the concrete constructor's strict
positive lower-frequency condition. In physical SI units this is
`f0=c0/wavelength > k*frequency_sigma_hz`, where `k=sqrt(-2*log(cutoff))`.
In the engine's normalized Band units the checked quantities are
`f0=1/band.wavelength` and `k*band.df`. Both derived quantities must be finite;
equality at the cutoff fails. The existing positive-finite leaf checks still
apply.

Previously, the same live wavelength/bandwidth leaves could move to a spectrum
reaching zero frequency and return finite fields with `runtime_stability=True`,
while the concrete twin was refused. This discrepancy was reproduced in one
CPU and two actual CUDA cases before the fix. The backend predicate now marks
that invalid carried band with NaN fields and failed runtime validity; restoring
the valid leaf values recovers the original outputs without changing earlier
results, leaf ownership or existing `.grad` buffers.

The carried-band branch is selected before reading either spectral leaf on the
host, including forward/batched Torch and JAX derivative values. It contributes
only a validity certificate; waveform lowering retains the physical leaves and
their derivatives. Ordinary concrete Band handling is unchanged, including the
incoming guard's identity for valid concrete inputs. A carried phase alone does
not recheck an otherwise concrete band. This introduces no new source, material,
geometry or transform admission.

`test_general_source_band_guard.py` passed **43 new cases: 22 CPU and 21 CUDA**.
The public recovery cases use the same wavelength/bandwidth leaves with a
periodic 8³, 17-step Ey CurrentDipole; CUDA checks plain and checkpoint-7 execution.
Helper cases exercise strict boundary values, wavelength-only/bandwidth-only
carrying, mixed leaf devices, overflow of the two derived spectral quantities,
preservation of prior failed validity, and Torch/JAX derivative and batching
certificates. Helper transformations do not establish arbitrary transformed
public solver support.

All **98 related cases also passed**: CPU finite-excitation 44, legacy source
checks three and pulse-Hessian nine; CUDA finite-excitation 33 and pulse-Hessian
nine. The related CUDA run passed 42 cases in 28.07 s with ten warnings. Together
with the 43 new cases this is 141 unique local cases, not full CI or complete
FDTD GPU qualification. All numerical and derivative budgets remained unchanged.
Every recorded property in the eight main CUDA pulse cases matched the pre-guard
qualification strings exactly, preserving its reported forward/value/gradient/
Hessian maxima; the ninth unit-pair case remains assertion-only evidence.
Source snapshots and before/after hashes were unchanged during these runs.

The carrier guard alone does not establish spatial resolution or bound all
intermediate waveform arithmetic. The subsequent pole-free certificate below
checks the existing spatial floor at the current values; neither guard
certifies an optimization's entire parameter range or continuum accuracy.

### Live Band pole-free spatial resolution

When wavelength or bandwidth carries a derivative and **every declared material
has no poles**, a backend certificate checks the existing source/report spatial
floor at each run. In normalized Band units it takes
`lambda_min = min_sources(1 / (1/lambda + k*df))`, including concrete sibling
sources, and checks the background and every structure, including concrete
materials. The shortest wavelength must be finite and positive. The existing
index rule selects the component with largest `abs(epsilon*mu)` (first on ties)
and its passive square root. The wave rule includes equality at **eight cells
per wavelength in the medium**. The existing metal source/report skin-depth
rule remains separate from static DiffractionMonitor wavelengths, which retain their
wave-only rule. Conductivity enters this certificate only under the existing
conductive-film descriptor; this does not add a general conductivity resolution
rule or broaden film admission.

Only the certificate is detached; physical source/material leaves and their
waveform derivatives remain intact. All-concrete Band handling is unchanged.
If **any declared material has any pole**, including an inactive zero-strength
pole, the whole previous resolution path remains in use, including its existing
live-material checks: dynamic Band resolution is not newly certified for that
scene. No source, medium, geometry, observation or transform admission expands.
The host `CoarseMeshWarning` can still state that planning could not read a
traced shortest wavelength. That host-report limitation coexists with this
bounded runtime certificate; it is not a measured coarse-grid error. The eight-
cell floor is a refusal threshold, not an accuracy or convergence guarantee.

The original wavelength update reproduced finite fields with successful runtime
validity below that floor in one CPU and two CUDA cases, although its concrete
twin was refused. A separate bandwidth-only update holds wavelength fixed and
moves from about 9.01 to `7.3330730` cells; the wavelength-update case reaches
`7.55748757` cells. Both retain a positive carrier cutoff. The fixed public
valid→invalid→valid tests fail runtime validity with NaN fields at the invalid
point, then recover exactly while preserving old results and leaf/`.grad`
ownership. They use the periodic 8³, 17-step Ey CurrentDipole, with plain CPU and plain
or checkpoint-7 CUDA execution.

`test_general_source_resolution_guard.py` passed **30 new cases: 25 CPU and five
CUDA**. Helper controls cover the inclusive floor and its adjacent floating-
point values, all-source/all-material selection, prior failed validity, static
DiffractionMonitor wavelengths, descriptor-scoped conductivity, whole-path pole fallback and
constant Torch/JAX derivative certificates without live-band host reads.
These helper transforms do not establish general transformed solver support.
The first CPU batch passed 119 cases; the added bandwidth trajectory then passed
its one new CPU case separately. In total, **125 related cases passed**:
64 CPU material-resolution, 22 CPU and 21 CUDA carrier-cutoff, and nine CPU and
nine CUDA pulse cases. New plus related coverage is **155 unique local cases**,
not full CI. The new CUDA run passed five cases in 2.64 s with nine warnings;
the related CUDA run passed 30 in 27.49 s with 13 warnings.

Source snapshots and before/after hashes were unchanged. Every recorded property
string in the eight main CUDA pulse cases matched the original pulse
qualification exactly, retaining the reported forward/value/gradient/Hessian
maxima; the ninth unit-pair case remains assertion-only. Numerical budgets were
unchanged. Declared-pole scenes and other derived waveform overflow remain
outside this added certificate; independent physical refinement is still needed.

## Fixed-time waveform representability

Finite source inputs alone do not guarantee representable sampled Gaussian
waveforms. The original regression changed a normalized delay to the finite
value `1e308`: the sampled drive became nonfinite and the electric field became
NaN, while `runtime_stability` remained true. One CPU and two CUDA reproductions
(plain and checkpoint-7) exposed this gap. Restoring the original delay recovered
the prepared run, and earlier owned results remained unchanged.

The bounded guard checks fixed-time source arithmetic:
positive finite lowered `tau`, `f0` and `t0`, then finite shifted time, normalized
time, its square, Gaussian exponent `-0.5*(s/tau)**2`, carrier prefactor
`omega=2*pi*f0` and phase `omega*s+phase` at `n=0` and `n=steps-1`
of each actual source row. The row shifts remain 0.5 for electric, 0 for
magnetic and 1 for column sources. It preserves the kernel's arithmetic order.
With positive fixed time
increments, the endpoint bounds cover the intermediate finite samples without a
per-step or full-field scan. Both paths evaluate the predicate using the actual
lowered backend parameters, rather than reconstructing the arithmetic in NumPy.
Fully eager concrete execution reads one backend boolean per source per run:
an invalid declaration raises `ValidationError` with code
`source_waveform_range`, while a valid declaration preserves the incoming guard
identity. On CUDA this entails a scalar synchronization once per source per run,
not once per time step. If the original inputs, lowered parameters or predicate
carry a differentiation or transformation trace, no host read is made; a
constant factor of one or NaN contributes to the existing runtime certificate.
An active Torch transform also keeps closed-over concrete source certificates
on the backend, even when their wrapped values have no derivatives. This avoids
reading a transformed boolean through host storage; ordinary eager concrete
identity/refusal and the physical waveform formulas are unchanged. Separate
closed-source regressions cover valid and invalid constants through JVP, grad,
jacfwd, Hessian, JVP-of-grad, vmap and vmap-of-grad with host reads forbidden.
JIT-derived constants can therefore use a traced factor of one without promising
literal `None` guard identity. Carried-path timing does not measure the concrete
scalar synchronization cost.
`test_general_source_waveform_guard.py` passed **36 new cases: 19 CPU and
17 CUDA**. CPU passed in 5.24 s with four warnings; CUDA passed in 2.97 s
with six coarse-mesh warnings and one Torch JIT future warning. Both signs of
huge delay now fail runtime validity and propagate NaN fields, then recover
exactly after the same leaf is restored; earlier outputs, leaf identity and
existing `.grad` buffers remain unchanged. Concrete invalid values receive the
structured refusal. Additional checks cover individual arithmetic overflows,
the last sample, electric/magnetic row shifts, neighboring backend `t0`
overflow values, valid guard identity and untouched source-free/`DecayStop` paths.
Torch forward/reverse/batch and JAX JIT/gradient/Hessian/batch checks qualify the
constant certificate only, not arbitrary transformed full-solver execution.

The five related test files passed **174 cases** in 78.26 s with 44 warnings:
finite excitation (77), carrier cutoff (43), spatial resolution (30), pulse
higher-order derivatives (18), and projected pulse fitting (6). Together with
the 36 new cases, this change qualified **210 unique cases**. Earlier partial
reruns are not added to that total. Recorded source copies and current source
hashes matched; this is scoped local qualification, not full CI.

This is a representability restriction, not a reformulation of the Gaussian:
a huge finite pulse declaration may be refused even when its ideal envelope
would be zero. It does not change the waveform arithmetic, expand source or
field admission, or qualify the adaptive `DecayStop` path. A true
`runtime_stability` means the implemented certificates passed; it is not a
blanket test that every output, intermediate or derivative is finite.

A bounded cost comparison used the same 8³ bspline pulse fixture, plain scans,
17 or 41 steps, one warmup and five public `prepared.run` calls under
`torch.no_grad()`. Preparation and instrument numerical checks were outside the
timer; public validation, lowering and adaptation stayed inside, with CUDA
synchronization at the timing boundaries. CPU used one thread. The pre-guard
source was `76bdea9b`; only the guard implementation changed. Concrete baseline
imports were checked against the archived pre-guard source. All sampled outputs
passed the existing independent discrete forward checks.

| Input path | Device | Steps | Before median [min, max], ms | After median [min, max], ms |
|---|---|---:|---:|---:|
| carried | CPU | 17 | 10.8012 [10.0707, 12.7233] | 10.7788 [10.2527, 10.8481] |
| carried | CPU | 41 | 18.3311 [18.1400, 19.1682] | 18.9597 [18.7516, 19.0930] |
| carried | CUDA | 17 | 25.0447 [24.4767, 27.9889] | 26.5309 [25.8810, 42.3962] |
| carried | CUDA | 41 | 47.8544 [47.5453, 49.8249] | 54.0605 [49.8878, 64.7790] |
| concrete | CPU | 17 | 9.5441 [9.4136, 11.0372] | 10.6422 [10.0567, 11.2492] |
| concrete | CPU | 41 | 17.6719 [17.3286, 17.7771] | 19.2064 [18.7677, 22.8558] |
| concrete | CUDA | 17 | 22.8888 [21.9422, 32.5485] | 31.4461 [23.7319, 34.6008] |
| concrete | CUDA | 41 | 47.8459 [44.3152, 59.2735] | 52.3676 [48.2676, 62.7320] |

These are five calls in the same order, without a reversed or randomized repeat;
they do not bound general overhead. The carried CUDA 41-step ranges did not
overlap; the other CUDA pairs did. No speed improvement is claimed. CUDA timed
peak allocated memory was unchanged: 140,288 bytes for carried inputs and
137,216 bytes for concrete inputs at both step counts. CPU memory was not
measured. The concrete comparison includes its backend scalar synchronization;
the carried comparison alone cannot measure that cost. This is no-grad evidence,
not a gradient-cost or peak device-memory guarantee.

## Same-leaf SGD reuse

`studies/fdtd/general_cuda/optimizer_reuse.py` checks repeated updates of the
same trainable leaves through a prepared run. The electric Ey CurrentDipole fixture
has 12 physical cells per axis, four Open-face PML cells on each side, an actual
20³ grid and 41 steps. Plain and checkpoint-7 scans use identical physics;
checkpointing leaves a six-step tail. This short thin-PML fixture is a discrete
derivative/lifetime study, not a PML-accuracy or performance benchmark.

Two NumPy runs at source phases zero and pi/2 supply final-field basis vectors
`b0` and `b1`, each concatenating E and Z0 H in consistent SI field units. Let
`S = ||b0||² + ||b1||²`, `G_ij = dot(b_i,b_j)/S`,
`u(phi) = (cos(phi), sin(phi))` and `r = A*u(phi) - 1.1*u(0.2)`.
The normalized fitting loss is `L = r.T*G*r/2`, with derivatives
`dL/dA = u.T*G*r` and `dL/dphi = A*u'.T*G*r`. This is an independent algebraic
value/gradient/update oracle for the **same discrete solver**, not an
independent Maxwell solution. The objective uses final fields only, excludes
the source waveform, and is not an energy in joules.

Each run makes 16 SGD updates with learning rate 0.125, `foreach=False` and
`zero_grad(set_to_none=True)`, starting from `(A,phi)=(0.8,-0.3)` or `(1.4,0.7)`.
A is dimensionless: the source current moment is `1e-16*A` A·m; phase is in
radians. The amplitude leaf stays on CPU, and the phase leaf on the execution
device. Leaf identities, storage addresses, dtypes and devices must remain
unchanged; their version counters must advance with each in-place update.
Fields use a peak-normalized `2e-12` comparison against the basis. Losses,
gradients and updated parameters use relative `2e-12`, absolute zero against
the analytic recurrence, with nonzero comparison controls. Old outputs are retained through the first two iterations
and checked for exact immutability. Separate NaN amplitude/phase updates must
fail validity or receive a finiteness-specific structured refusal, then recover
on the same leaves and prepared object.

For CUDA, after the old output is released, every iteration must return to the
exact persistent target/leaf allocated-byte baseline; no drift allowance or
in-loop cache purge is used. Peak counters are cumulative through zero-based iteration 1,
then reset once and cumulative thereafter; they are not per-iteration peaks.
Final tensor release must return allocated bytes to the process baseline before
cache release. Reserved and driver memory are recorded separately. The reported
loop wall time includes validation, synchronization, host copies, garbage
collection and receipt writing, so it is not a solver-speed comparison.

The study requires a fresh external output directory, matching source and basis
hashes, and a separately supervised CUDA process with a 120-second limit. CUDA
setup requires 5 GiB free and caps the Torch allocator at 4 GiB; this does not
reserve memory or cap driver/other-process use. Exact source snapshots and an
attempt receipt precede execution; missing or changed reference/source files
invalidate the attempt. See the study README for the complete receipt contract.

### Recorded optimizer study

The study's **24 algebra/receipt/CLI control tests passed**. Separately, the
NumPy basis command, two CPU plain-loop commands and four actual CUDA loop
commands (two seeds × plain/K7) passed. These are six CLI study executions,
not six additional executed pytest cases; matching integration cases are
provided in the test file. Each loop completed all 16 updates and both invalid-
input recovery probes. The basis Gram matrix was positive definite, with
condition number `73.9756791383416`.

CPU gradient error was at most `3.025309471976914e-15`. Across the four CUDA
runs, maximum relative loss, gradient and post-update parameter errors were
`5.096959445281874e-15`, `4.421606151350874e-15` and
`1.380285480163295e-15`, respectively, below the unchanged `2e-12` budget.
All old-output checks, leaf-identity/version checks and negative-input recovery
checks passed. This establishes agreement with the prescribed nonlinear SGD
recurrence; no convergence threshold or recovery of the target parameters was
used as an acceptance condition.

Every CUDA iteration after releasing the retained old output returned to
**384512 allocated bytes**: 384000 target-field bytes plus the phase leaf's
512-byte allocator block. The CPU amplitude leaf is outside that device total.
Both seeds produced the same following cumulative peaks after the single reset
at the end of zero-based iteration 1 (covering iterations 2–15):

| Scan policy | Peak allocated bytes | Peak reserved bytes | Allocated after each release |
| --- | ---: | ---: | ---: |
| Plain | 3315712 | 6291456 | 384512 |
| Checkpoint 7 | 7171072 | 12582912 | 384512 |

Checkpointing used more allocated and reserved memory in this small case.
This is not a universal checkpoint ranking or a total-memory bound. There was
no in-loop cache purge. Final tensor cleanup returned allocated bytes to the
zero process baseline; reserved bytes remained 6291456/12582912 until explicit
cache release. Afterwards allocated/reserved were both zero and reported driver
free memory was 7389315072 bytes. These allocator checks do not imply zero
CUDA driver/context allocation.

Instrumented loop totals were 9.2828/9.1116 seconds for plain and
13.2623/13.0117 seconds for K7 (seeds 0/1). They include all the inspection and
receipt work described above and are not a performance comparison. Each run
preserved all 263 source-file hashes and the reference receipt/artifact hashes.
The environment was Python 3.12.14, Torch 2.14.1+cu130, CUDA 13.0, NumPy 2.5.4,
Windows 11, RTX 4060 Ti and one Torch CPU thread. These bounded local results do
not replace reviewed-candidate CI or qualify arbitrary optimizers, objectives,
long trajectories, material optimization or other devices.

The commands below show seed 0. Repeat CPU, CUDA plain and CUDA K7 with
`--seed 1` and distinct new output directories to reproduce both seeds.
The default `--block 0` selects the plain scan; it is not public
`Numerics(checkpoint_steps=0)`.

## Homogeneous single-pole ADE

The bounded dispersive route uses `DispersiveMaterial` in exactly one `SnappedSlab`
covering the whole actual grid, with all six faces `Periodic` and exactly one
`Lorentz` or `Drude` pole. The ordinary uniform-grid, fixed-duration, double-
precision and voxel requirements still apply. There are no material interfaces
or external sources; the background declaration remains ordinary `Material`.
Initial E/H are plain finite real NumPy arrays of shape `(3,nz,ny,nx)` over the
full grid; one may be `None` to supply a zero field. CPU tensors, callables and
component lists are not accepted in this ADE scope. Floating and integer arrays
convert to double execution precision; float32 retains the existing precision
warning. Observations are `TimeSeriesMonitor` and/or final fields, including
TimeSeriesMonitor-only runs with `final_fields=False`. Open/PML, Bloch, walls, multiple poles, Debye, dispersive background,
spatial films and Fourier/flux observations are outside this route.

`epsilon_inf`, `mu`, damping `gamma` and Lorentz `omega_0` are nondifferentiable
scalars; `mu=None` means 1. Their concrete values may change on prepared reuse
when the resolved plan remains compatible and scientific checks pass. This
differs from the fixed spatial patterns, whose changes require preparation
again. The admitted differentiable inputs are real scalar `InitialField.amplitude` and Lorentz
`strength` or Drude `omega_p`, expressed in the existing SI pole convention.
Initial spatial patterns are not differentiable inputs. This scope retains
the existing material validity and pole-aware time-step checks; it is not a
waiver of passivity, finiteness or stability restrictions.

The compact storage route retains the ordinary scalar-recurrence arithmetic
guards. A positive finite pole declaration can still produce an unrepresentable
nonzero drive: concrete values are refused with `PoleArithmeticRefused`, while
traced live values mark the result's `runtime_stability` false and propagate
invalid fields rather than report them stable. Prepared reuse from valid to
invalid and back to valid is covered for both Lorentz and Drude. The dedicated
arithmetic regression adds eight CPU/device-free and two actual CUDA cases;
38 existing mixed-arithmetic checks also passed. These checks preserve the
arithmetic contract rather than expand the admitted material range.

The scalar ADE interior-energy diagnostic selects its storage formula from the
declared pole family, not the numerical recurrence coefficient `hold`. At
`gamma*dt=2`, Drude and Lorentz can both have `hold=0`; this does not make them
Debye poles. The original Drude CUDA case produced NaN diagnostic energy at
this collision, while the tested Lorentz case already passed on its reachable
state. An explicit numeric `energy_is_debye` parameter now preserves the family
through lowering and checkpoint/JAX pytrees. It adds one double (eight bytes)
per pole, not per grid cell; the field recurrence and carried state are unchanged.
The existing cost tables predate this metadata addition.

`test_general_ade_energy.py` compares the zero-hold field and normalized summed
storage with an independently written uniform Ampere/oscillator recurrence at
relative `2e-12`, zero absolute tolerance. The diagnostic uses its integer-E
sample times; it is not an SI joule total or a new continuum-accuracy claim.

The existing startup sets the algorithmic polarization memories and carried
polarization increment to zero. The first E update therefore has no carried
pole increment; the pole is advanced afterwards. This is a discrete initial
state, not a continuum-equilibrated polarization initialization. For a uniform
electric field, a raw fixed-time refinement diagnostic shows first-order
time error for the tested Lorentz and Drude examples. Evaluating a separately
shifted reference at `(n-.5)*dt` explains the startup phase offset but does not
repair the solution or establish second-order raw continuum accuracy.

The two CPU diagnostics hold the grid and physical end time
`6.671281903963041e-15` s fixed and use 64/128/256 steps. The maximum window
errors and successive observed orders are:

| Pole | Raw errors at `n*dt` | Raw orders | Shifted errors at `(n-.5)*dt` | Shifted orders |
| --- | --- | --- | --- | --- |
| Lorentz | 0.0148468 / 0.00748655 / 0.00376165 | 0.98778 / 0.99293 | 0.00234079 / 0.000592359 / 0.000149097 | 1.98245 / 1.99022 |
| Drude | 0.0357578 / 0.0179577 / 0.00900017 | 0.99366 / 0.99658 | 0.000903182 / 0.000229998 / 0.0000580323 | 1.97339 / 1.98670 |

These are diagnostics of the unchanged startup, not fitted acceptance orders.
All three time resolutions still pass the independent discrete-solution gate.

The independent gates use SI characteristic roots and a scalar recurrence for
uniform fields, and an explicitly assembled 15×15 Maxwell/polarization state
matrix for nonzero Fourier modes at physical Yee offsets. Missing-pole and
wrong-startup controls must be detectably different. Field and time-series errors use
peak-normalized `1e-11`; uniform recurrence residuals use `1e-12`. A 45×45
matrix-power Frechet construction supplies amplitude/material gradients and
full Hessians for final-field and signed time-series objectives, at relative
`2e-12` with zero absolute tolerance. Plain and checkpoint scans, mixed-device
leaves, prepared reuse and retained old outputs are separate checks. These
oracles share the declared discrete startup; they qualify eager reverse mode
and double backward, not arbitrary continuum initial states or general transforms.

Public numerical qualification covers 40 distinct CPU cases (26 unchanged
passing cases retained plus 14 modified cases rerun) and **42 actual CUDA cases**
(10.99 s, 40 deselected, 51 expected mesh warnings). The contract suite separately
passed 58 checks needing no device plus seven CPU cases, then six actual CUDA
cases (0.90 s, 65 deselected, seven expected warnings). Together these are
153 distinct new cases; the nine existing scope-refusal checks are separate.
These results use the public admission path, without an experimental context.

| CUDA control | Worst recorded relative error | Budget |
| --- | ---: | ---: |
| Uniform final fields | 4.00635199454821e-13 | 1e-11 |
| Uniform time series | 1.3211653993039363e-14 | 1e-11 |
| Uniform scalar recurrence | 2.7755575615628914e-16 | 1e-12 |
| Fourier final fields | 2.1958220980047683e-13 | 1e-11 |
| Fourier time series | 4.73407123422404e-13 | 1e-11 |
| Magnetic-only initial mode, `E=None`, `mu=None` | 7.496839298600154e-15 | 1e-11 |
| Translated rectangular grid, all SnappedSlab axes, equivalent um/nm declarations | 8.560921989767082e-15 | 1e-11 |
| Prepared reuse with changed concrete epsilon/mu/gamma/omega_0 | 7.137981093552503e-15 | 1e-11 |
| Final-field value, gradient and full Hessian | 4.438322122768973e-14 | 2e-12 |
| TimeSeriesMonitor-only value, gradient and full Hessian | 7.834081628920496e-15 | 2e-12 |

The translated box declares `(0.61,0.49,0.71)` um at 0.05 um spacing; upward
rounding produces `(13,10,15)` cells and physical size `(0.65,0.5,0.75)` um.
The full-grid SnappedSlab must cover those actual cells, not only the nominal extent.
The derivative cases use 12³ cells, 17 steps, plain or block-7 scans with a
three-step tail, two amplitude/material parameter points, a CPU amplitude leaf
and a material leaf on the execution device. TimeSeriesMonitor objectives use all six
staggered components and a nonzero signed linear term as well as weighted squares;
no final-field output is required. The contract checks also retain invalid-live
material behavior, float32/integer conversion, precision warnings and the CPU
epsilon-derivative route. No budget was relaxed. This local qualification does
not establish full CI, release integration, general dispersive CUDA support or
second-order raw continuum initialization.

### Compact CUDA pole state

Within this admitted homogeneous scope, Torch CUDA preparation selects a pole
term carrying only the current and previous polarization vectors. Their
increment is computed once before each E update instead of being carried as a
third vector. The recurrence order and zero-memory startup are unchanged;
first and second derivatives are independently qualified below. The autograd
graph itself changes. This is a state-storage change, not a continuum startup
correction or a new material-admission policy. The term type participates in
plan identity, and compatible prepared reuse selects it again. A compact plan
cannot execute on CPU or a non-Torch backend.

Ordinary CPU, NumPy/JAX and other existing pole consumers retain their original
term. In particular, this change does not extend to source-bearing or `DecayStop`
plans, films, source columns, subclasses, multiple poles, CPML or geometry
ranges. For one full-grid pole in double precision, the logical E/H-plus-pole
report falls from 15 scalar components (120 bytes/cell) to 12 (96 bytes/cell).
The pole histories alone fall from nine to six scalar components. This 20%
logical-state reduction is not a prediction of allocated or reserved peak:
reverse graphs, checkpoint boundaries, observations and temporary tensors have
separate lifetimes.

The preceding numerical tables record qualification before this storage
change. A separate bounded 32³/64³, 163-step research comparison found lower
allocated peaks for plain, block-8 and block-32 scans, with overlapping timing
ranges in every pair. Those historical measurements motivate the change;
they are not measurements of the final product route or evidence of a general
speed advantage. Current-product reproduction uses
`studies/fdtd/general_cuda/ade_cost.py` through ordinary public preparation,
without replacing a runtime or pole factory. It checks independent roots,
recurrence and amplitude/material derivatives outside the timed interval and
separates logical payload, forward-retained graphs and timed allocator peaks.
See `studies/fdtd/general_cuda/README.md` for fresh-process commands, source
identity and the matched size-32 forecast required for size 64.

The final compact implementation requalified the same **153 existing ADE
cases**: 105 CPU/device-free and 48 actual CUDA cases. These are repeat
qualification of the preceding evidence, not 153 additional cases. Separately,
`test_general_ade_memory.py` passed **31 new cases**, 21 CPU/device-free and ten
actual CUDA. They cover selection/refusal, actual state bytes and prepared
reuse; eight CUDA cases compare full Hessians after no-grad/inference warm-up,
using blocks 8/32 over 35 steps with a three-step tail. The earlier discrete
and derivative budgets remain unchanged. The study's independent-oracle suite
passed 17 checks, and eight existing CPU regressions retained ordinary/film
DecayStop memory, film energy and PlaneWave source lowering across their relevant
NumPy/JAX/Torch routes. These groups are separate; none implies full CI.

#### Historical product cost measurements

These six measurements describe product commit `5dd110ad`, after the
scalar-recurrence arithmetic guards for compact storage were restored. Lowering
is inside the measured public call, so later source and initial-amplitude
validity checks are not represented by these timings. Reproducing this table
requires that historical source; running the current study measures the current
checkout instead. Earlier attempts remain historical records; allocated/reserved
peaks were unchanged in that follow-up. These results do not establish a general
speed advantage.

The product study used Python 3.12.14, Torch 2.14.1+cu130, CUDA 13.0, NumPy
2.5.4, Windows 11 and an RTX 4060 Ti (8585216000 device bytes), with one Torch
CPU thread and float64. Each row is a fresh process, one warm-up and five
synchronized repetitions of 163 steps. The fixture differentiates uniform
initial amplitude and Lorentz strength through a full-grid material, one Ey
time series and final fields. Its `sum(E_final**2)` objective uses the same SI
recurrence for every policy.

| Grid | Block | Median seconds [min, max] | Peak allocated bytes | Peak reserved bytes |
| --- | --- | --- | ---: | ---: |
| 32³ | None | 0.6346065 [0.6136485, 0.6656123] | 139294720 | 186646528 |
| 32³ | 8 | 0.9067716 [0.8781101, 0.9841063] | 81847296 | 109051904 |
| 32³ | 32 | 0.9275602 [0.8787053, 0.9474104] | 53517312 | 71303168 |
| 64³ | None | 0.6113849 [0.5887202, 0.6221247] | 1113685504 | 1239416832 |
| 64³ | 8 | 0.8852429 [0.8678227, 0.9109478] | 654371328 | 736100352 |
| 64³ | 32 | 0.8840796 [0.8349297, 0.9061474] | 427860480 | 484442112 |

Timing includes fresh leaves/live-problem construction, public prepared
execution (including its validation, lowering, adaptation and associated host
work), the objective and `autograd.grad`. Initial preparation, instrument
numerical checks/output host copies, allocator inspection and cleanup are
excluded. The cache is released after warm-up and after all five repetitions,
retained between them; each returned tensor bundle is released before the next
baseline. Peaks in the table belong to timed runs, excluding the separate
untimed lifecycle and ownership calls. Allocator quantities exclude CPU,
driver/display and other-process memory.

All 30 measured repetitions passed independent roots/time-series (`1e-11`), absolute
recurrence (`1e-12`) and objective/material/amplitude derivative checks (relative
`2e-12`, zero absolute tolerance). Worst time-series, absolute recurrence and relative
gradient errors were `6.106226635438361e-15`, `3.9985376121265404e-16` and
`2.3453478832892504e-13`. Source hashes remained unchanged and final current
allocated/reserved bytes returned to zero in all six processes. Actual pole
initial-memory bytes matched the report: 1572864 at 32³ and 12582912 at 64³;
E/H-plus-pole state was 3145728 and 25165824 bytes respectively.

At 64³ the separate forward-retained allocated values were 1038181888 (plain),
534827520 (block 8) and 157309440 (block 32) bytes; after backward each retained
12587008 bytes before release. These are not the logical state payload or the
timed peak. These recorded peaks equal the corresponding earlier compact-state
measurements; historical baseline-versus-compact timing ranges overlap. The
product table establishes this bounded memory/time tradeoff, not a general
speedup, scaling law or universally best checkpoint block.

Size 64 used matching successful size-32 receipts. Reserved forecasts were
2239758336/1308622848/855638016 bytes for plain/8/32, below the 4 GiB cap.
Each prewarm-up check saw 7389315072 free bytes and passed the additional 1 GiB
headroom requirement. The forecast does not reserve memory or guarantee against
an out-of-memory failure. No profiler was active during these measurements.

## Evidence and its limits

The original general-scene and observation files account for the 35 cases above.
The integrated run including 21 checkpoint and four Job cases passed all 60
cases. A separate default CPU/JAX lifecycle, unified Job and migration regression
passed 53 cases.

| Test | Independent control or device comparison |
| --- | --- |
| Uniform conductor, periodic box, 20 steps | Exact SI conductivity recurrence for E and its time record, unchanged H; relative `2e-14` |
| Conductivity and initial-amplitude derivatives | Closed-form derivatives of the recurrence to relative `2e-12`, with both CPU and CUDA parameter leaves |
| SnappedSlab with CPML, plane-wave source, records, Fourier fields and flux | NumPy CPU parity with the existing peak-normalized infinity budget `5e-13`; live prepared reuse, changed inputs and retained old results |
| SnappedSlab permittivity and source derivatives | NumPy material central difference at relative `1e-6`, absolute `1e-8`; exact source quadratic identity at relative `2e-12` |
| TimeSeriesMonitor, FieldMonitor and FluxMonitor objectives, 161 steps | CPU/CUDA objective parity to relative `5e-13`; source quadratic identity to `2e-11`; nonzero material central differences at two step sizes to `2e-6` |
| SnappedSlabs normal to x, y and z; open 3D pulse; diagonal epsilon and mu with/without conductivity | CPU field parity to peak-normalized `5e-13`; six nonzero box-flux faces compared independently to prevent cancellation hiding orientation errors |
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

## Static Bloch evidence

This admission change uses the existing complex kernel and CPML recurrence.
The final new coverage contains **66 unique cases**: 25 CPU, 28 actual CUDA,
and 13 rejection controls that need no CUDA device. The CPU material/amplitude
AD case was separately verified with Torch 2.14.1. The existing nine general
scope-refusal checks are separate from these new cases.

| New coverage | CPU | Actual CUDA |
| --- | ---: | ---: |
| Exact discrete modes | 4 | 4 |
| Material/amplitude AD | 1 | 2 |
| Lifecycle and physical contracts | 13 | 13 |
| Oblique PlaneWave | 5 | 5 |
| Complex-CPML observable AD | 2 | 4 |

The initial CUDA run passed 18 mode/AD/contract cases in 5.72 seconds; the
PlaneWave run passed five in 27.37 seconds. A later run passed the expanded
13 contracts plus four observable cases in 103.92 seconds. Its 142 expected
coarse-mesh/finite-window warnings remain visible. The later contract rerun
overlaps the initial run and is not added again to the unique total.
The 13 device-independent controls cover CurrentDipole/wall exclusions, forward-AD/
`torch.func` wrappers and same-primal forward-dual values on prepared reuse.
Their success establishes early rejection, not CUDA execution.

| Gate | Independent control and unchanged budget |
| --- | --- |
| Complex Yee modes, 16³, 300 steps | Three cyclic orientations with signed Bloch phases and one Periodic axis, plus all-Bloch axes; exact E/H fields and records to `1e-10`, temporal recurrence to `1e-11`; omitted/reversed phase controls exceed `1e-4` |
| Material and initial-amplitude AD, 16³, 17 steps | Analytically differentiated discrete eigenfrequency and complex squared-modulus objective; full E/H, objective and both derivatives to relative `2e-12`, absolute zero; plain and K7 with a three-step tail, mixed-device leaves and prepared reuse |
| Static/lifecycle controls | Equivalent phase/k declarations agree; changed phase, representation or initial pattern rejects reuse; CUDA live phase/k and unqualified Bloch/source/wall combinations refuse; paired faces, nonzero phase, complex dtype, source-wavevector consistency and one-cell guards remain active |

For the AD oracle, `K_j = 2*sin(k_j*dx/2)/dx` and
`omega = 2/dt*asin(dt*|K|/(2*sqrt(epsilon*mu)))`. Static initial E/H arrays
are chosen so that `E_N = A*E_0*cos(N*omega*dt)` for each live epsilon. The
objective is `A**2*mean(abs(E_0)**2)*cos(N*omega*dt)**2`; its analytic amplitude
and epsilon derivatives do not differentiate the initialization. The two
changed leaf sets preserve old outputs, original arrays and leaf ownership.
This is an exact discrete-mode derivative gate, not a continuum or Bloch-phase
derivative claim.

The public oblique fixture uses a 0.25×0.2×3 µm physical domain, 0.05 µm spacing,
24-cell PML and a Gaussian pulse. Four cases cover 20° s/p incidence in both
z directions for 1100 steps. The closed form uses the independently written
waveform transform, fixed-k Yee dispersion and staggered discrete polarization;
the hard-source-to-probe distance is derived from the declared geometry.
Public seconds/radians and H in A/m are explicitly converted for this oracle.
Real-frequency errors were at most `5.545210e-7` on CUDA against the existing
`2e-6` budget. A fifth case, 30° polar/35° azimuth p polarization, uses 1600
steps and the damped contour `omega + 0.4i` in geometry-time units; its error
was `6.135543e-8` against `3e-6`.

Across the five CUDA runs, backward/forward peak was at most `8.393595e-16`,
entry-plane reference error `1.111734e-15`, and neighbouring-cell phase error
`1.347744e-15`, each below the unchanged `1e-13` budget. All six field components
are checked, with a nonzero forward floor and complex128 device placement.
The source reference is compared with main-grid samples, while the forward
wave is independently checked against the closed form. These finite vacuum
cases do not establish every incidence angle, material, PML thickness or
complex-observable derivative.

### Complex CPML observations and source derivatives

The additional 83-step fixture uses a 30°/35° p-polarized PlaneWave, static
transverse Bloch phases, a nondispersive slab and four-cell Open-z PML. It
checks plain scans and K7 with a six-step tail. Both CPU cases and all four
actual CUDA cases passed; the CUDA counts include both scan policies.

Two NumPy source-phase basis runs give a real Gram matrix from complex inner
products for each of five groups: final E, final H, E/H records, all six Fourier
field components, and electric/magnetic incident reference histories. Analytic
source-amplitude/phase values, gradients and full Hessians agree at relative
`2e-12`, absolute zero at two changed leaf sets. These are source phase
derivatives, not derivatives of the static Bloch boundary phase. CPU amplitude
and execution-device phase leaves, retained backward graphs and old public
output buffers keep their ownership through prepared reuse.

Separate TimeSeriesMonitor, FieldMonitor and FluxMonitor objectives compare values with NumPy at relative
`5e-13`, the amplitude quadratic identity at `2e-12`, and source-phase/slab-epsilon
central differences at relative `2e-6`, all with zero absolute tolerance. Each
secant uses both `1e-4` and `5e-5` times the parameter value, with nonzero signal
and derivative floors. Only fixed SI unit factors rescale outputs. Short-window
and mesh warnings remain visible: these finite-trajectory AD controls do not
establish PML reflection accuracy or converged optical spectra.

## Fixed-center CurrentDipole evidence

The later CurrentDipole addition passed 42 CPU and 46 actual CUDA cases in separate
runs; nine existing CUDA refusal controls also passed. These receipts are
separate from the earlier 60-case run. Existing fixed-duration, double-precision,
voxel, material, SnappedSlab, boundary and monitor restrictions still apply.

| Test file | CPU / CUDA passes | Evidence |
| --- | ---: | --- |
| `test_general_cuda_dipole.py` | 23 / 27 | Electric/magnetic, three axes, both stamps: one-step SI moment relative `1e-13` and independently reconstructed Yee centroid absolute `1e-13` um. Short slab/PML fields, source histories and observations match NumPy at peak-normalized `5e-13`; static/support refusals and reuse checked. |
| `test_general_cuda_dipole_conductive.py` | 12 / 12 | Electric dipoles: three axes × two stamps × scalar/diagonal epsilon and mu, with scalar conductivity 600 or 2400 S/m. Independent `E1 = -dt*J/(epsilon0*epsilon + sigma*dt/2)` recovers moment and centroid at the same budgets and checks SI source work at the half step. |
| `test_general_cuda_dipole_radiation.py` | 7 / 7 | Vacuum Gaussian bspline source: Larmor energy refinement, electric/magnetic duality and unchanged SI results under um/nm geometry units. |

The core derivative cases use CPU current-moment/delay and CUDA phase/material
leaves. Two NumPy central differences in dimensionless perturbation coordinates
agree at relative `1e-7` plus absolute `1e-10*loss`; the current-moment quadratic
identity agrees at relative `1e-12`. The 41-step checkpoint cases use block 7
and a six-step tail, retain source-field history gradients, and preserve caller
inputs and old histories during fresh and prepared reuse. Their thin PML and
unfinished Fourier-window warnings are intentional; these short parity/AD
cases do not establish radiation accuracy.

The separate radiation fixture uses a vacuum box 0.6 wavelengths wide,
12-cell PML, eight source periods and the default bspline stamp. Energy is
the time integral of the source's delivered-work history, compared with the
independent SI Gaussian-current Larmor value `2.5026540826870543e-24` J.
Actual CUDA values were:

| Cells per wavelength | Delivered energy (J) | Relative Larmor error |
| --- | ---: | ---: |
| 14 | 2.3846422619180062e-24 | -0.0471546673531249 |
| 20 | 2.444240894665441e-24 | -0.02334049616593281 |
| 28 | 2.472762333282175e-24 | -0.01194401959570257 |

Fitted order was `1.9810176952811915`; the second-order Richardson relative
error was `-7.26898350460655e-5`. The unchanged gates require order between
1.9 and 2.1, Richardson magnitude below `1e-3`, and coarse error between
-0.06 and -0.03. Open-box magnetic/electric energy minus one was
`-2.7458726875040362e-5` against the existing `1e-4` criterion; periodic
lattice duality is separately checked at `1e-13`. Mesh warnings remain visible.
This establishes the stated vacuum bspline radiation cases through source work,
not separate far-field or enclosing-surface flux accuracy, tent/nonvacuum
radiation accuracy, position derivatives or arbitrary source transformations.

## PEC/PMC wall evidence

A separate wall qualification passed 38 CPU cases and 41 CUDA cases (79 new
cases), plus nine existing refusal controls. The remaining restrictions above
still apply: Bloch combined with PEC/PMC walls is refused, and symmetry and
invariant-axis faces remain outside the CUDA route.

| Test file | CPU / CUDA passes | Evidence |
| --- | ---: | --- |
| `test_general_cuda_walls.py` | 20 / 20 | PEC/PMC 16³ cavities over 300 steps; both mixed-wall orientations on x/y/z; source support touching each wall refused. |
| `test_general_cuda_wall_ad.py` | 18 / 21 | Independent initial node projection, 41-step initial-amplitude and interior electric/magnetic CurrentDipole derivatives, prepared ownership, and two wall/Open/PML interaction cases. |

Cavity references use independent discrete Yee frequencies and exact staggered
fields: recurrence residual relative `1e-11` and phase/amplitude relative
`1e-10`, retaining the established budgets. Wall-position mutants must exceed
`1e-4` recurrence error. Mixed PEC/PMC walls use their actual length
`L - dx/2` or `L + dx/2`; treating either as L fails. These are exact discrete
mode checks, not a continuum refinement or arbitrary cavity-accuracy claim.

Arbitrary nonzero host arrays independently verify the initial projection
before the first curl: PEC-low zeros tangential E and normal H, PMC-high zeros
tangential H and normal E. PEC-high and PMC-low use exterior ghosts and do not
zero an extra stored layer. Unmasked interiors and caller inputs are unchanged.
At 41 steps, CPU runs use the plain scan; CUDA also exercises block 7 with a
six-step tail. CPU/CUDA amplitude leaves retain the quadratic identity at
`2e-12`; initial-amplitude NumPy secants use `1e-7`. Interior xyz electric and
magnetic CurrentDipoles retain current-moment/phase derivatives against two NumPy
secants at `1e-7` plus `1e-10*loss`, quadratic scaling at `1e-12`, and half-step
source work at `1e-13`. Fresh and prepared reuse preserve old results and input
ownership; changing boundary declarations invalidates preparation.

Two additional cases combine x PEC, y PMC and z Open with four-cell PML,
electric Ey or magnetic Hy, and an 8³ physical domain. They use CPU plain scans
and CUDA block 7 at the same 41-step horizon. These bounded parity/AD checks
exercise the wall/CPML/checkpoint interaction; thin padding does not establish
PML reflection accuracy or radiation accuracy. Existing mesh warnings remain.

## Independent public slab gate

`test_general_cuda_airy.py` adds an independent optical gate through the public
`PlaneWave`, `FieldMonitor` and `FluxMonitor` interfaces. All six cases passed: three CPU and
three actual CUDA cases. This separate run is not included in the 60 cases
above. Its independent optical evidence is limited to the planar fixture below.
A later run after skipping unused DFT phases and adding the no-grad runtime guard passed all
three general CUDA optical cases again, together with the two separate
periodic top/bottom slab cases. Existing mesh warnings remained visible.

The fixture is a 0.2 um, epsilon-2.25 film in vacuum, with transverse periodic
faces, open z faces and 24-cell PML. The five wavelengths are 0.80, 0.90, 1.00,
1.15 and 1.30 um. Film and empty-domain runs use the same source and probes;
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
ordinary scan. CPU backends, other devices and `DecayStop` are refused for explicit
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
nine-step tail; its final fields, TimeSeriesMonitor, FieldMonitor and FluxMonitor objectives and material/
source derivatives agree with the ordinary scan (values `5e-13`, gradients
`1e-12` relative). Multiple backward queries retain the graph correctly, and
two prepared reuses with independently created leaves preserve old results.
The source quadratic identity remains independently checked at `2e-11`, and
invalid live material still returns failed validity and NaNs.

The admission cases include three regressions rejecting boolean Torch tensors
as block lengths.

Four Job tests check that the default leaves a job's key as it is without the
field, and round-trip explicit schedules. `checkpoint_steps=None` is omitted from
the `wavebit.job/v2` envelope; an explicit integer is serialized. This establishes the opt-in contract and
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

## Gaussian pulse wavelength, bandwidth, phase and delay Hessians

This gate exercises the existing electric Ey CurrentDipole route; it changes no runtime
or admission rule. The periodic 8³ fixture has dx 0.05 um, Courant 0.5, relative
epsilon 2.25, mu 1.3 and conductivity 600 S/m. A fixed off-grid center
`(0.013,-0.021,0.007)` um and current moment `I0=1.7e-16` A·m use either
`bspline` or `tent`. The initial fields are zero. The two lengths are 17 and
41 steps; CUDA checkpoint-7 tails are three and six steps.

Let `T=1 um/c0`, `k=sqrt(-2*log(1e-3))` and the dimensionless coordinates be
`p=(l,b,phi,d)`. The physical inputs are

```text
wavelength = 2*l um
frequency_sigma_hz = 0.08*b/T
phase = phi radians
delay = BASE_DELAY + T*d seconds, BASE_DELAY = -k*T/(2*pi*0.08).
```

The delay baseline is fixed. Changing bandwidth therefore changes both envelope
width and peak time `delay+k/(2*pi*frequency_sigma_hz)`; it is not recentered to
hold the peak fixed. The two points are `(1,1,0.4,0.12)` and
`(0.97,1.01,-0.2,0.16)`. For physical coordinates x in um, Hz, radians and seconds,
`x=offset+S*p` with `S=diag(2,0.08/T,1,T)`. Thus `g_p=S*g_x` and
`H_p=S*H_x*S`; physical Hessian entries require division by both scale factors.
For wavelength in metres, replace the first factor by `2e-6`.

The independent reference writes the Gaussian current and all four first and
sixteen second derivatives analytically. At electric-current times `(n+0.5)*dt`,
`u=t-delay-k/(2*pi*f_sigma)`, `v=2*pi*f_sigma*u` and
`Q=-v²/2+i*(2*pi*c0*u/wavelength+phi)` give `I=I0*Im(exp(Q))`.
The first/second forcing derivatives include `Q_i` and `Q_i*Q_j+Q_ij`.
An independently constructed cardinal cubic or width-two tent stamp preserves
moment and centroid. Fifteen lanes (value, four first derivatives, ten symmetric
second derivatives) propagate through explicit periodic indexed Yee updates
with the SI electric-current factor
`-dt/(epsilon0*(epsilon+sigma*dt/(2*epsilon0))*dx³)`.
No product waveform, stamp, curl, coefficient or AD helper supplies this oracle.

Two objectives are checked: `mean((drive/I0)²)` and the weighted squared final
E and Z0H plus six TimeSeriesMonitor means, with weights `(1,0.7,1.3,0.6,1.1,0.9)`,
component normalization `1/6`, time-series factor `1/4`, and time-series sample
weights `1+0.2*n/N`.
Value, all four gradients and the complete 4×4 Hessian, including both mixed
orders, use relative `2e-12`, absolute zero. Fields use full-array peak scaling;
Time series use the corresponding E/H group peak so the structurally zero Hy remains
checked without division by zero. Forward errors use `2e-12`, clocks `2e-14`.

The pure-reference qualification covers 14 unique cases. Its initial secant
attempt passed 13 and failed one: at absolute normalized step `2e-5`, a mixed
Hessian secant error `3.96673119e-6` exceeded `2e-6`. Halving the step reduced the
absolute error by approximately four, consistent with central-difference
`O(h²)` truncation. The final absolute steps `1e-5` and `5e-6` passed the same
`2e-6` budget; the maximum was `9.91691679e-7`. The minimum normalized Hessian
entry was `0.0793395795`, above the unchanged `1e-6` non-vacuity floor. The failed
attempt is retained; no direct derivative or secant tolerance was increased.
All seven forcing/stamp mutants passed the separation floor of `1e-4`.

The public tests compare plain CPU/CUDA and CUDA checkpoint execution, retaining
the same leaves through point 0 → point 1 → point 0. Wavelength and phase leaves
start on CPU, bandwidth and delay on the execution device. They check derivative
ownership, unchanged prior output/gradient/Hessian tensors, source work and
recovery of the original values. A separate fresh-preparation um/nm pair checks
physical equivalence. Static cutoff/center changes, nonscalar bands and changed
wavelength under a period-based duration are checked for refusal before stepping.
All **32 unique cases passed**: 14 pure-reference cases, nine public CPU cases
(five solver/unit cases and four pre-execution refusals), and nine actual CUDA
cases (eight Hessian/reuse cases and one unit pair). The public CPU run passed
in 3.39 s with eight warnings; CUDA passed in 27.42 s with nine traced-band
`CoarseMeshWarning`s. These warnings describe the host planning check; the
subsequent pole-free runtime certificate is separate. They are not evidence of
an independently measured coarse-grid error.

| Recorded CUDA comparison | Maximum relative/scaled error | Budget |
| --- | ---: | ---: |
| Forward fields, time series and source drive in eight main cases | 2.72025560e-14 | 2e-12 |
| Two objective values | 1.31006317e-14 | 2e-12 |
| Four gradients | 6.06181771e-14 | 2e-12 |
| Complete 4×4 Hessians | 9.27036226e-14 | 2e-12 |

The derivative audit independently recomputed all eight main CUDA cases at
three visits and two objectives from their recorded values. The separate CUDA
unit pair passed its assertions but did not record errors for this maximum.
The recorded public CPU forward maximum was `2.72594056e-14`; CPU derivative
assertions passed without a separately recorded maximum. Source hashes remained
unchanged through qualification.

This is discrete forcing and eager double-backward validation, not radiation
accuracy, continuum convergence, arbitrary pulse optimization or transform
support. The two reference points independently satisfy a positive lower
spectral edge and more than ten cells per shortest cutoff-band wavelength in
the medium. The subsequent [live Band cutoff guard](#live-band-carrier-cutoff)
enforces finite `f0` and `k*f_sigma` with strict `f0 > k*f_sigma` for carried
bands. The later [pole-free spatial certificate](#live-band-pole-free-spatial-resolution)
checks the existing floor at current values when no declared material has poles.
Declared-pole scenes retain the previous resolution path. Neither predicate
certifies continuum accuracy or an entire optimization range; independent
refinement and controls on other waveform intermediates remain needed.

## Projected four-coordinate pulse fitting

This regression targets the existing electric Ey CurrentDipole route and changes no
runtime or admission rule. It uses the preceding independent 15-lane pulse
reference with the `bspline` stamp, periodic 8³ grid, 41 steps and fixed medium.
The normalized target is `p*=(0.985,1.005,0.1,0.14)`; the two initial points are
`(1,1,0.4,0.12)` and `(0.97,1.01,-0.2,0.16)`. Wavelength, bandwidth, phase and
delay retain the physical mapping and fixed delay baseline described above.

Let F contain final E and Z0H, and R contain the six TimeSeriesMonitor histories with H
multiplied by Z0. Component weights are `(1,0.7,1.3,0.6,1.1,0.9)` and time-series
sample weights are `1+0.2*n/41`. The residual objective is

```text
C = mean(w_R * R_target**2) + 0.25*mean(w_F * F_target**2)
L(p) = [mean(w_R * (R(p)-R_target)**2)
        + 0.25*mean(w_F * (F(p)-F_target)**2)] / C.
```

The target and residual value, gradient and full Hessian come from the
independent SI reference, including the residual-times-second-derivative term.
The checked HVP uses direction `(1,-0.5,0.25,0.75)`. At the zero-residual target,
let G be the target Gauss–Newton Hessian, `D=diag(1/diag(G))`, and
`eta=0.1/lambda_max(sqrt(D)*G*sqrt(D))`. These quantities are fixed before the
trajectory; each coordinate's SGD learning rate is `eta*D_ii`.

Each of twelve updates first proposes `q=p-eta*D*g`, then projects q componentwise
onto the fixed normalized box
`[0.94,1.04] × [0.97,1.04] × [-0.5,0.7] × [0.08,0.2]`.
The optimizer uses `foreach=False` and clears gradients with
`zero_grad(set_to_none=True)`. Projection occurs outside autograd: the derivative
comparison concerns the smooth residual at each evaluated point, not the
nonsmooth projection or the complete optimization map.

The original unconstrained preflight left this box on the second proposal from
the second seed, with wavelength coordinate approximately `0.92887` below
`0.94`. That was a trajectory-domain failure, not a derivative mismatch. The
projected protocol was specified separately before its qualification; the
original attempt is retained and no derivative tolerance was relaxed.

All **14 unique new cases passed**: eight pure-reference controls, two public
plain CPU trajectories and four actual CUDA trajectories (two seeds with plain
or checkpoint-7 execution, tail six). The public CPU run passed in 9.76 s with
four warnings; CUDA passed in 34.71 s with four warnings. These are test-run
elapsed times, not performance measurements. Qualification preserved source
bytes and the original derivative budgets.

Wavelength and phase remain on the caller CPU; bandwidth and delay are on the
execution device. At every update, the tests compare both the raw SGD proposal
and projected point with the independent trajectory, as well as residual
value/gradient/HVP at the actual local point. Relative tolerance is `2e-12`,
with zero absolute tolerance. They retain the same leaves, preserve old detached
outputs, clear `.grad`, and require weak references to live result/derivative
graphs to die before the next iteration.

An independent audit recomputed all six trajectories and 72 updates. Maximum
relative errors were `1.55049897e-13` for loss, `1.78910054e-13` for gradients and
`5.13582612e-14` for HVP; the maximum recorded forward scaled error was
`1.04404066e-14`.

| Seed | Initial loss | Loss after 12 updates | Updates with projection |
| --- | ---: | ---: | ---: |
| 0 | 0.122965706571319 | 0.021001548655822 | 0 |
| 1 | 0.151257104004487 | 0.011733149330523 | 11 |

The second trajectory ends on the wavelength-coordinate boundary `l=0.94`.
A lower residual is not unique recovery of the target coordinates. All four
CUDA cases recorded 60,416 allocated bytes at each of the twelve post-update
checks with one owned output pack held. Releasing that pack restored the
persistent baseline with target tensors and caller leaves still alive; this is
not total allocation zero. These observations are live-storage checks, not
peak-memory measurements or evidence of checkpoint speed/memory benefit.
The scope remains this discrete fitting regression, without universal
convergence, continuum inverse accuracy or a public optimizer API.

Reproduce the pure controls and all six public trajectories in the qualified
environment; CUDA cases skip when an actual CUDA device is absent. To select
only public CPU or CUDA cases, use `-k cpu_plain` or
`-k 'gpu_plain or gpu_k7'`, respectively.

```sh
python -m pytest tests/fdtd/test_pulse_fit_reference.py -q
python -m pytest tests/fdtd/test_general_cuda_pulse_optimizer.py -q
```

## CPML amplitude/phase second derivatives

The public `test_general_cuda_cpml_higher_order.py` uses electric and
magnetic CurrentDipoles in a uniform vacuum box with six Open faces, four-cell PML and
41 steps. Plain scans and CUDA K7 with a six-step tail share the same oracle.
Two NumPy basis runs at phase 0 and pi/2 supply a Gram matrix `G`; for amplitude
`A` and `u = (cos(phi), sin(phi))`, the quadratic objective is `A**2 * u.T G u`.
Analytically differentiating this expression gives value, gradient and the full
amplitude/phase Hessian at relative `2e-12`, absolute zero. CPU amplitude and
device phase leaves, changed prepared inputs and immutable previous outputs
exercise graph ownership. The reference shares the spatial discretisation:
it establishes bounded derivatives, not independent PML reflection accuracy or
arbitrary material/geometry Hessians. All six cases passed on the retained
functional runtime: two CPU and four actual CUDA cases, in 21.71 seconds. This
is a separate qualification result from the earlier conductor Hessian cases.

### Bounded CPML optimization experiment: not adopted

A CUDA-specific scratch-reuse adjoint was evaluated against the original
functional CPML expression. The final guarded candidate did not demonstrate a
consistent timing improvement or any allocator reduction in the bounded
PML8/K7 comparison, so the runtime optimization was not adopted. The functional
CPML implementation remains; the independent derivative gate is retained.

The experiment used the same RTX 4060 Ti environment, one Torch CPU thread,
float64, a 16³ physical grid, Periodic x/y and Open z, eight-cell PML, a live
slab permittivity and live electric-dipole current moment over 41 steps. Each
fresh process performed one warmup and five repetitions. Timings include
prepared execution, objective, backward, synchronization and allocator queries
after forward and backward. Initial preparation, leaf creation, result checks,
host copies, profiling and cache cleanup are excluded. Cache release after
warmup and each repetition makes these instrumented, cache-released wall times.

| Order | Functional reference median seconds [minimum–maximum] | Guarded candidate median seconds [minimum–maximum] |
| --- | ---: | ---: |
| Reference then candidate | 0.3954916 [0.3694054–0.4154265] | 0.3865133 [0.3495335–0.4249051] |
| Candidate then reference | 0.4004945 [0.3895314–0.4209488] | 0.4036426 [0.3598773–0.4964137] |

Both pairs' ranges overlap. All runs passed the NumPy objective/central-secant,
moment-quadratic and source-work checks, with exactly equal recorded objective
and gradients. Peak allocated/reserved bytes were 6639104/8388608 in both
implementations. Allocator peaks exclude driver/display memory. The short,
thin-PML workload does not establish absorption accuracy. These results neither
identify the cause of the timing variation nor support a general speedup or
memory saving; the experimental candidate is not part of the shipped runtime.
Raw experimental machine records and the rejected candidate are not distributed
with this page; the public command below reproduces the retained Hessian gate.

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

### Separate 64³ material-gradient measurement

A later experiment scales only the same 163-step conductor fixture to 64³
and differentiates both amplitude and conductivity. Each policy ran in a fresh
process, sequentially in order 8, 32, None, with one warm-up and five synchronized
repetitions; one Torch CPU dispatch thread and the same prepared-call timing
boundary were retained. Every repetition passed the independent SI objective
and both derivative checks at relative `2e-12`, absolute zero, and the recorded
values and gradients were exactly equal across policies.

| Block | Median seconds [minimum–maximum] | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: |
| 8 | 0.700060 [0.681144–0.727873] | 413192192 | 463470592 |
| 32 | 0.701149 [0.660019–0.751319] | 526407680 | 589299712 |
| None | 0.549679 [0.535557–0.623938] | 2101355520 | 2350907392 |

Both checkpoint policies reduced allocator peaks and increased median time
relative to the plain scan in this fixture. The checkpoint time ranges overlap;
they do not rank block 8 versus 32 reliably. This does not qualify other
durations, differentiated inputs, source histories or monitor loads.

Before each warm-up, free/total device memory was 7408189440/8585216000 bytes.
Allocated/reserved baselines were zero; after every timed repetition allocated
memory returned to zero, and final collection/cache release returned both to
zero. Peaks are absolute and incremental here because the baselines coincide;
they exclude driver/display/other-process memory. Cleanup is outside timing.

The instrument forecasts reserved memory from the measured 32³ material peaks,
multiplied by eight for cell count and a 1.5 margin: about 0.773/0.984/3.914 GiB for
8/32/None. It refuses forecasts above 4 GiB or free memory below the forecast
plus 1 GiB. This is a budget estimate, not a reservation or guarantee against
out-of-memory failure; concurrent users can change free memory after admission.
The extension accepts only 64³/163 steps/material derivatives and one policy
per process. Local JSON records used instrument SHA-256
`6aaed9733821a062f28fae6b6defa1f8161c1e98095af0f71daa5ce68238a6fa`.

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
one CPU dispatch thread and a 4×4×84 grid, 128 steps, 12-cell PML, two Ey FieldMonitors
and two FluxMonitors at three wavelengths. Both runs used `torch.no_grad()`
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

### Bounded DFT addcmul experiment: not adopted

A separate experiment at `91200897` replaced the functional DFT multiply-plus-add
with out-of-place `torch.addcmul`, guarded by dtype metadata to retain the
original expression when intermediate promotion would differ. It passed 30 CPU
and 30 CUDA primitive cases, nine CPU independent quadrature/derivative checks
and one frozen Bloch reference, six CUDA observation/derivative cases (142.21 s, 190 expected coarse-mesh
and unsettled-field warnings), and three CUDA Airy cases (107.25 s, six mesh
warnings). These numerical checks did not establish a performance benefit.

The same environment and 4×4×84, 128-step plain no-grad monitor fixture above
were used, additionally returning final E/H for full-array checks. Each row is
a fresh process with one warm-up and five synchronized prepared calls; initial
preparation, host copies, post-run numerical checks, cleanup and profiling are
excluded; live validation within the prepared call is included.
Outputs are released between repetitions, with allocator cache retained after
the first timed run. The second pair reverses execution order.

| Execution order | Implementation | Median seconds [minimum–maximum] |
| --- | --- | ---: |
| First pair, first | Original expression | 0.5814155 [0.5355944–0.6884793] |
| First pair, second | Candidate | 0.6112528 [0.5513491–0.6458652] |
| Reverse pair, first | Candidate | 0.5895994 [0.5594044–0.6366238] |
| Reverse pair, second | Original expression | 0.5563477 [0.5421061–0.6044896] |

All four attempts passed full monitor/final-field NumPy comparisons at the
unchanged peak-relative `5e-13` budget (worst `5.207610346140793e-16`) and output
ownership checks. Timed peak allocated/reserved bytes were 420864/2097152 in
every run; final cleanup returned both to zero. Separate single-run profiles
changed `aten::mul` 5120→3840, `aten::add` 2603→1323 and `aten::addcmul` 0→1280;
`aten::copy_` remained 912. Both time ranges overlap, and neither wall time nor
allocator peak shows an advantage. Fewer operators alone do not justify adoption.
The runtime retains the original expression. Candidate code and raw machine
records are not distributed here; these observations describe only this bounded
experiment and do not establish a cause for its timing outcome or a general
ranking of the operations.

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
11, comparing final E/H, time-series, Fourier E/H, flux and source histories with
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
python -m pytest tests/fdtd/test_general_source_finite.py -q
python -m pytest tests/fdtd/test_general_source_band_guard.py -q
python -m pytest tests/fdtd/test_general_source_resolution_guard.py -q
python -m pytest tests/fdtd/test_general_source_waveform_guard.py -q
python -m pytest tests/fdtd/test_general_cuda_initial_patterns.py -q
python -m pytest tests/fdtd/test_general_cuda_optimizer.py -q
python -m pytest tests/fdtd/test_general_cuda_region_contract.py tests/fdtd/test_general_cuda_region.py -q
python -m pytest tests/fdtd/test_general_cuda_higher_order.py -q
python -m pytest tests/fdtd/test_general_cuda_pulse_reference.py tests/fdtd/test_general_cuda_pulse_higher_order.py -q
python -m pytest tests/fdtd/test_general_cuda_cpml_higher_order.py -q
python -m pytest tests/fdtd/test_general_cuda_dipole.py tests/fdtd/test_general_cuda_dipole_conductive.py -q
python -m pytest tests/fdtd/test_general_cuda_dipole_radiation.py --run-slow -q
python -m pytest tests/fdtd/test_general_cuda_walls.py --run-slow -q
python -m pytest tests/fdtd/test_general_cuda_wall_ad.py -q
python -m pytest tests/fdtd/test_general_cuda_bloch_modes.py tests/fdtd/test_general_cuda_bloch_planewave.py --run-slow -q
python -m pytest tests/fdtd/test_general_cuda_bloch_ad.py tests/fdtd/test_general_cuda_bloch_contract.py tests/fdtd/test_general_cuda_bloch_observations.py -q -o junit_family=legacy
python -m pytest tests/fdtd/test_general_ade_modes.py tests/fdtd/test_general_ade_units.py tests/fdtd/test_general_ade_ad.py tests/fdtd/test_general_ade_record_ad.py -q -o junit_family=legacy
python -m pytest tests/fdtd/test_general_ade_energy.py -q
python -m pytest tests/fdtd/test_general_ade_contract.py -q
python -m pytest tests/fdtd/test_general_ade_modes.py -k raw_continuum -q -s
python studies/fdtd/general_cuda/optimizer_reuse.py --mode basis --output-dir ../optimizer-reuse/optimizer-basis
python studies/fdtd/general_cuda/optimizer_reuse.py --mode run --reference ../optimizer-reuse/optimizer-basis/receipt.json --device cpu --seed 0 --output-dir ../optimizer-reuse/optimizer-cpu-0
python studies/fdtd/general_cuda/optimizer_reuse.py --mode run --reference ../optimizer-reuse/optimizer-basis/receipt.json --device cuda:0 --seed 0 --output-dir ../optimizer-reuse/optimizer-cuda-plain-0
python studies/fdtd/general_cuda/optimizer_reuse.py --mode run --reference ../optimizer-reuse/optimizer-basis/receipt.json --device cuda:0 --block 7 --seed 0 --output-dir ../optimizer-reuse/optimizer-cuda-k7-0
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 16 32 --steps 163 --blocks 0 8 32 200 --repeat 3 --output artifacts/general-checkpoint.json
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 8 --repeat 5 --output artifacts/general-checkpoint-64-k8.json
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 32 --repeat 5 --output artifacts/general-checkpoint-64-k32.json
python studies/fdtd/general_cuda/checkpoint_cost.py --sizes 64 --steps 163 --parameters material --blocks 0 --repeat 5 --output artifacts/general-checkpoint-64-plain.json
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
