# Matched absorption of commuting coupled dielectrics

Written: 2026-10-07

Targets:
- `wavebit/fd/_kernel/constitutive_absorber.py`
- `wavebit/fd/_lower/constitutive_absorber.py`
- `wavebit/fd/_lower/constitutive.py`
- `wavebit/fd/_lower/params.py`
- `wavebit/fd/_lower/plan.py`
- `tests/test_fd_constitutive_absorber.py`
- `studies/fdtd_constitutive_absorber_algebra.py`
- `studies/fdtd_constitutive_absorber_probe.py`

**TL;DR.** This implemented bounded route advances compact electric flux with the
same matched-loss factors as the field. It admits a coupled dielectric support
inside `Absorber` only when every finite interpolation link connects equal damping
rates, and observes the corrected positive staggered storage. The first algebra
disclosure completed 81 fixed configurations on three CPU backends; its largest
reported differences are absolute energy/work residuals, not relative field
errors. The 24-solve physical calibration also completed, but every run was still
unsettled and finite-length sensitivity is substantial. The three fixed instrument
allowances were accepted after disclosure. A fixture-only correction was followed
by 148 passing new cases, 646 passing cases in the inclusive targeted selection,
and 20 accepted source-fault witnesses with exact restoration. All 60 registered
physical solves and the saved-data replays completed, but all 60 retained unsettled
field warnings. The observed finite-domain and duration differences remain
substantial. The full and separate slow regressions passed, and all four freezer
comparisons were bit-exact. No termination-accuracy or convergence result is
claimed.

## Contract and what the proof covers

| Item | Bounded admission |
| --- | --- |
| Material | The whole scene uses concrete, positive isotropic, lossless `Material` values with relative permeability one. |
| Geometry | Ordinary voxel shapes with no declared range; existing geometry, occupancy, support, seam and wall rules still apply. Each body entering an absorbing layer remains uniform through it. The exact link certificate, not a shape name alone, decides coupling admission. |
| Useful examples | Cylinders, prisms and rotated boxes extruded through the single absorbing axis, with cross links in the transverse plane. Axis permutations obey the same rule. |
| Boundary | One axis has `Absorber` on one or both ends. Other faces are PEC, PMC, Periodic or Bloch. No PML, Symmetry or reduced Axis1D face anywhere in this first route. |
| Excitation | Electric Dipoles or a source-free scene. Public initial fields, magnetic sources, PlaneWave and TFSF remain outside this route. |
| Execution | Concrete NumPy, JAX and Torch CPU forward execution with a fixed duration. Differentiation, enclosing transformations, GPU execution and automatic Decay stopping are refused for this new term. |
| Prepared plan | All coupled terms and the complete source/memory term tuple must match the supported scene; adding a source cannot silently reuse a source-free allocation. |

This extends one matched-loss case within C9. It does not establish CPML overlap,
magnetic constitutive loss, dispersive or conductive coupled geometry, general C8
smoothing, or the open C7 convergence gate. Their existing measured scopes remain
as documented in [smoothed occupancy](fd-subpixel-occupancy.md).

A potential-coupling support with equal material and host permittivity still owns
its flux term. Its zero cross coefficients are a physical control on the new
route, rather than an automatic switch to the old diagonal route.

## Why damping the field alone is insufficient

For the physical inverse-permittivity operator `K=diag(k)+K_cross`, let `R` be the
diagonal matched-absorber rate at each electric Yee edge, `h=dt`, `r=h/dx`, and

```text
A = (I-h R/2) (I+h R/2)^-1
B = (I+h R/2)^-1.
```

With `K R=R K`, the source-corrected curl `q` advances compact flux and the cross
field increment as

```text
D_new = A D_old + r B q
delta_E_cross = r K_cross(B q).
```

The existing diagonal step supplies `A E_old+r B diag(k)q`. Hence `E=K D` is
preserved in exact arithmetic from the zero initial state. Floating-point drift
is a measurement target, not a bitwise assertion. The ordinary coefficient
builder divides by the loss factor, while the new cross path multiplies its
reciprocal; their rounding need not be identical.

The lift and CFL certificate must read the undamped material operator. Only on
this new route, the diagonal is recovered as `be/(r B)` before the existing star
lift; a lifted change patches `be` by `r B delta_k`. Host planning and runtime
lowering use the same physical diagonal. Invalid or nonfinite coefficients are
refused rather than silently used in a certificate.

## The finite-link certificate

A nonzero xy cross coefficient at a common electric node joins x edges at
`p,p-e_x` to y edges at `p,p-e_y`, with weight one quarter for every live pair.
The certificate compares the represented damping rates at both ends of each
such link, and repeats this for xz and yz. Ghost edges outside the compact box
are omitted exactly as in the bounded interpolation. It does not use a numerical
commutator tolerance or accept cancellation between unequal-rate links.

PEC and PMC masks restrict the operator to the unmasked subspace: `K_Q=Q K Q`,
with the corresponding adjoint projected curl. This principal restriction
preserves positivity and the upper bound. It does not require an unproved
`Q K=K Q` identity. Diagonal damping commutes with the mask, and the all-link
certificate remains sufficient after restriction. A diagonal star lift cannot
create new cross links.

## Positive staggered storage and its observation clock

Let `M_e=K_Q^-1`, `Sigma_e=M_e R_e` and `Sigma_h=R_h`. The magnetic material is
one. For the adjoint discrete curl `C`, the update obeys

```text
(H_plus-H_minus)/h + Sigma_h (H_plus+H_minus)/2 = -C E_old
M_e (E_new-E_old)/h + Sigma_e (E_new+E_old)/2 = C* H_plus + J.
```

For `z=(E_old,H_minus)`, define the Hermitian matrix

```text
P = [[M_e, -h C*/2], [-h C/2, I]],    W=z* P z.
```

The exact one-step identity is

```text
W_new-W_old = -2h Re(E_bar* Sigma_e E_bar + H_bar* Sigma_h H_bar)
             +2h Re(E_bar* J).
```

`P` is positive under the existing physical inverse-material/curl CFL bound.
The source-free identity is dissipative for every nonnegative rate; trapezoidal
loss does not need an additional small-rate condition. Electric source work
remains explicit. A Gaussian pulse has an infinite tail, so passing its declared
cutoff is not the same as a strictly source-free step.

The old observer's `E*D+Re(H_minus*H_plus)` differs from this storage in an
absorber. The scoped observer reads

```text
W = E*D + Re(H_minus*H_plus)
    +(h/2) Re[H_minus* R_h (H_minus+H_plus)].
```

The correction covers every observed absorber cell, including cells outside the
compact material support. NumPy evaluates the following magnetic field before
its destructive update; JAX and Torch use the two actual field arrays. A completed
row at `n dt` uses `E^n`, `H^(n-1/2)` and `H^(n+1/2)`. Fixed-duration sampling retains
its established cadence and completed final row. There is no pending-row Decay
contract for the new memory yet.

Scenes without constitutive-support overlap retain the old term, coefficient
arithmetic and observer. The two observers can therefore report different
values even for physically equal zero-contrast fields. The zero-contrast field
control must not demand equality between those different stored-energy forms.

## Instruments and validation status

| Instrument | Question and limitation |
| --- | --- |
| Independent local operators | Explicit edge incidence builds dense `K`; real/complex, all-axis CPU checks compare returned flux and field with `K D`. |
| Energy and source work | Dense adjoint operators and actual small PEC/PMC Yee steps compare storage at both clocks with loss and nonzero electric work. Private consistent flux seeds are algebraic instruments, not public InitialField support. |
| Algebra calibration | The fixed 81-row disclosure completed; actual/reference arrays and separate zero-reference residuals are reported below. The `algebra` and `step` allowances were accepted afterward, as recorded below. |
| Physical calibration | Two fixed extrusion families on three CPU backends: each row contains the positive-material short/reference pair and the equal-material shaped/omitted pair (24 solves across six rows). These preceded the recorded acceptance of the shared-grid and zero-route rounding allowance; termination sensitivity has no small-error threshold. |
| Fixed physical campaign | All 60 registered solves completed: both families, both transverse source directions, two placements and three spatial rungs, plus designated controls. Every trace/component/wavelength is retained; substantial finite-domain sensitivity remains. |
| Regression | Actual targeted collection, source-mutant assertion witnesses with exact restoration, full and slow suites, and all four read-only freezer comparisons. |

The physical pair measures sensitivity to a finite matched termination. It is
not a continuum waveguide reference or a proof of spatial convergence. Any
causal-prefix claim requires verified common grid coordinates, coefficients and
source stamps. Longer-domain and longer-time controls are empirical sensitivity
checks. Increasing the absorber thickness does not by itself promise a smaller
terminal reflection: the implemented cubic rate profile has a fixed integrated
normal loss.

### Completed algebra disclosure

Attempt 1 completed all 81 fixed instrument calls (27 per CPU backend), with
768 finite, shape-matched comparisons and 1,536 saved actual/reference arrays.
Both the child and its observed parent process exited zero. The source snapshot
was written before importing the test instruments, and all 117 recorded
production/instrument input hashes remained unchanged. These are direct calls to
the independent instruments with the comparison recorder substituted in memory;
they are not a completed pytest or numerical-tolerance gate. Structural and
positivity assertions remained active. See the
[disclosure record](../studies/results/fdtd_constitutive_absorber_algebra_attempt1/result.json)
and [actual parent exit](../studies/results/fdtd_constitutive_absorber_controls/algebra_launch_attempt1/exit.json).

For a nonzero reference `b`, the algebra recorder uses NumPy's complex modulus:
`max(abs(a-b))/max(1,max(abs(b)))`, where `abs(z)=sqrt(real(z)^2+imag(z)^2)`.
This differs from the real/imaginary coordinate norm of the physical instrument
below. A zero-reference energy/work identity instead
records the absolute residual with scale one. Separating those quantities gives:

| Instrument group | Normalization | Comparisons | Largest observed difference |
| --- | --- | ---: | ---: |
| Local algebra, nonzero reference | `max(abs(a-b))/max(1,max(abs(b)))` | 150 | 7.7588603324e-16 |
| Local energy/work identity | Absolute zero-reference residual, scale 1 | 24 | 3.6193270603e-14 |
| Projected Yee steps, nonzero reference | `max(abs(a-b))/max(1,max(abs(b)))` | 450 | 2.2208973639e-15 |
| Projected Yee loss identity | Absolute zero-reference residual, scale 1 | 144 | 2.4371615837e-12 |

The largest local residual is in the complex, damped, nonzero-source-work
configuration. The largest step residual is in the NumPy PEC case with the
absorbing direction along z. That case's sampled storage spans approximately
6752.83 to 7493.36; its direct sampled-storage comparison is 3.6412e-16 under the
nonzero-reference normalization. The 2.4372e-12 scalar residual is therefore not a
relative field error or a fractional energy-loss measurement. No specific cause
of its magnitude is established by this disclosure.

These measurements apply to the fixed small operators, seeded states and finite
step sequences. They do not certify a condition-independent error bound over all
admitted dielectric contrasts, CFL margins, domain sizes or durations. The
recorder's `max(1, reference peak)` convention also means a small nonzero reference
is compared on an absolute scale; it is not a pure relative error near zero.
No numerical allowance was selected during these measurements. The raw algebra
log contains no warning or failed-row report, and this attempt required no retry.
Private seeded states remain an algebraic instrument rather than public initial-
field support for this route.

### Completed physical calibration and its finite-domain limits

Attempt 1 completed all 24 solves: two geometries times three CPU backends, each
with a positive-material short/reference pair and a zero-contrast shaped/omitted
pair. All child exits and the observed parent exit were zero. Recorded arrays were
finite, the runtime `stable` flags were true, and all declared mapping/path checks
held. Source identities were unchanged. These checks do not establish a settled
DFT or small termination sensitivity. See the
[calibration record](../studies/results/fdtd_constitutive_absorber_calibration_attempt1.json)
and [actual parent exit](../studies/results/fdtd_constitutive_absorber_controls/calibration_launch_attempt1/exit.json).

The fixed calibration uses vacuum resolution 20, placement 0.3, transverse
polarization u, time 16, axial lengths 3.2 and 6.4, and physical absorber thickness
0.4. Both domains and observation times are finite. The source band, geometry and
comparison windows were fixed before these responses were observed.

The physical instrument defines `M(a)` as the maximum absolute value over all
stored real and imaginary coordinates: `max(max(abs(real(a))), max(abs(imag(a))))`.
For real arrays this is the ordinary maximum absolute value; for complex arrays
it is not the complex modulus. Its roundoff comparison is `M(a-b)/max(1,M(b))`.
Under that definition the largest backend comparison is 6.7451328148e-15.
Shared-grid/lowered-data alignment reaches 9.0205620751e-16;
the zero-contrast retained-term and omitted-geometry routes have zero difference
in the compared field, clock and source arrays on each backend. Their different
energy observers are deliberately not compared for equality. A source-stamp
comparison is nonzero at the stated alignment scale, so no exact causal-prefix
claim is made.

Those small implementation comparisons do **not** describe the positive-material
short-versus-reference response. The following NumPy summaries use the actual
relative sensitivity `M(short-reference)/M(reference)`, with a nonzero reference
scale for every listed component. Unlike the roundoff comparison, the denominator
is not floored at one. Real time traces are unaffected by this norm distinction;
the complex plane DFT tables use the real/imaginary coordinate norm defined above. These are differences between two finite
calculations, not errors against an exact answer or reflection coefficients.

| Geometry | Component | Whole-record difference at the two record sides |
| --- | --- | ---: |
| cylinder | Ex | 8.972% |
| cylinder | Ey | 40.275% |
| cylinder | Hx | 156.498% to 162.244% |
| cylinder | Hy | 19.463% to 21.139% |
| rotated-box | Ey | 8.973% |
| rotated-box | Ez | 36.240% |
| rotated-box | Hy | 106.041% to 127.835% |
| rotated-box | Hz | 16.907% to 17.618% |

The weak magnetic records exceed 100% under their own small reference peaks.
The dominant electric records differ by about 8.97%; the dominant magnetic
records differ by about 16.9% to 21.1%. The plane DFT comparisons, over all points
in each component's sampled plane at each wavelength, are:

| Geometry | Plane component | Wavelength 0.9 | 1.0 | 1.1 |
| --- | --- | ---: | ---: | ---: |
| cylinder | Ex | 25.776% | 6.088% | 23.601% |
| cylinder | Ey | 30.245% | 9.228% | 29.426% |
| cylinder | Hx | 59.584% | 72.729% | 96.741% |
| cylinder | Hy | 6.369% | 3.860% | 39.499% |
| rotated-box | Ey | 15.994% | 16.599% | 17.787% |
| rotated-box | Ez | 15.194% | 33.741% | 27.445% |
| rotated-box | Hy | 36.118% | 76.861% | 57.866% |
| rotated-box | Hz | 14.244% | 12.035% | 31.801% |

The full record windows and all backend/component values remain in the saved
calibration record. The plane comparisons already reach several tens of percent;
a small backend discrepancy cannot certify a physically accurate termination.
The later time-32 and farther-domain controls, reported below, characterize only
their two declared configurations. They do not justify correcting or extrapolating
the other rows.

Warnings remain part of the result. Every one of the 24 solves reports
`UnsettledFieldWarning`; the positive-material plane tail-to-peak energy ratios
are approximately 0.00185/0.00255 for the cylinder's short/reference runs and
0.00488/0.00363 for the rotated box, above the diagnostic line 1e-5. The phasors
therefore retain finite-time truncation. Eighteen runs also report coarse-mesh
and subpixel-accuracy warnings: 60 `CoarseMeshWarning` and 36
`SubpixelAccuracyWarning` events are retained, including repeated planning/run
messages. The shaped cross sections are 6.4 cells across their smallest extent,
and the positive material has 8.56 cells per wavelength at the source band's
short-wavelength edge. These geometry and mesh warnings are not replaced by the
algebra results. No attempt failed or was retried in this calibration.

The `stable` flag is an existing growth diagnostic, not a convergence certificate.
The saved `interior_spread` values include the diagnostic's infinite sentinel
when its subwindow means do not form a strictly increasing sequence; this is not
nonfinite field data. No source-free sparse-energy identity is claimed for these
still-driven Gaussian-pulse observations.

### Accepted instrument allowances and preserved targeted failure

The [recorded decision](../studies/results/fdtd_constitutive_absorber_acceptance_attempt1/decision.json)
accepted the following allowances at 2026-10-07 05:31:22.974254 UTC, after the
completed algebra and physical disclosures and their independent saved-data
reviews. Its [observed parent process](../studies/results/fdtd_constitutive_absorber_controls/acceptance_launch_attempt1/exit.json)
exited zero. The test's allowance assignment was `None` during both disclosures.
Here `epsilon` is float64 machine epsilon.

| Comparison group | Accepted allowance | Largest disclosed metric | Allowance / observed maximum |
| --- | ---: | ---: | ---: |
| `algebra` | 512 epsilon = 1.1368683772e-13 | 3.6193270603e-14 | 3.1411 |
| `step` | 32768 epsilon = 7.2759576142e-12 | 2.4371615837e-12 | 2.9854 |
| `physical_alignment` | 128 epsilon = 2.8421709430e-14 | 6.7451328148e-15 | 4.2137 |

The first two maxima are the absolute zero-reference energy/work residuals
separated above. The shared `step` allowance also applies to field/state
comparisons; their measured maximum was 2.2208973639e-15, not the larger energy
residual. The physical maximum is a backend comparison of the zero-shaped
cylinder's energy, with absolute difference 5.5297277868e-10 at reference scale
81981.0067292. It is not a shaped-versus-omitted energy comparison.

These fixed CPU instrument allowances are empirical test criteria, not universal
rounding bounds, relative weak-signal guarantees, continuum error bounds or
termination-reflection tolerances. They accept no small-error interpretation of
the finite-domain sensitivity above. Their values remain fixed while later
failures are investigated. The decision and the exact test text before and after
the allowance assignment are retained together.

The independent physical-data review first omitted `normalization_status` while
reconstructing saved metadata. Its failed replay and the successful replay after
that schema correction are retained; no simulation or response was rerun for
this review repair. This review failure is distinct from the 24 successful
calibration worker executions.

[Targeted attempt 1](../studies/results/fdtd_constitutive_absorber_targeted_attempt1.json)
collected 148 new cases and finished with 147 passed and one failed, in 73.69
pytest seconds. The child and observed parent exited one, and the later
compatibility selection was not reached. Source and guard identities were
unchanged during that attempt. The preserved
[raw failure](../studies/results/fdtd_constitutive_absorber_controls/targeted_attempt1/new_constitutive_absorber.log)
is the `pml` scope-refusal fixture: constructing `PML(cells=2)` raised the ordinary
minimum-depth error before the test entered its expected
`ConstitutiveAbsorberScope` assertion. This result does not identify a production
update or C9 refusal defect.

The narrow fixture correction is to construct that deliberately unsupported
boundary with `allow_thin=True`. Static inspection shows that the fixture first
builds the valid extrusion plan, then calls the C9 scope guard directly with the
replaced boundary. Allowing the thin PML object to be constructed therefore lets
the existing C9 face-type check reject PML as intended; it does not admit PML to
the new route. The [preserved correction receipt](../studies/results/fdtd_constitutive_absorber_fixture_repair_attempt1/receipt.json)
records exactly that one keyword addition, with unchanged production, helper,
assertion, case and numerical-allowance content. The executed retry is reported
separately below.

### Targeted retry, actual source faults and registration

[Targeted attempt 2](../studies/results/fdtd_constitutive_absorber_targeted_attempt2.json)
passed all 148 new cases and all 646 cases in the broader targeted selection,
with actual child and parent exit zero. The 148 are a subset of the 646; these
are not 794 distinct cases. Source and guard identities were unchanged during
the retry.

The [original source-fault run](../studies/results/fdtd_constitutive_absorber_mutations_attempt1.json)
accepted all 19 intended assertion or missing-refusal witnesses across the 11
new test functions. Each production change was restored exactly, and the final
restored baseline passed all 148 new cases. A separate witness then removed the
CPML-specific refusal exercised by the repaired fixture, confirming that the
fixture actually detects that missing protection.

The [first supplemental attempt](../studies/results/fdtd_constitutive_absorber_acceptance_fixture_repair/cpml_witness_attempt1/result.json)
exited one because its output parser expected the older class-form spelling of
pytest's missing-exception message. The actual failure was
`Failed: DID NOT RAISE ValueError`, rather than the expected spelling containing
`<class 'ValueError'>`. The production bytes were restored and the restored node
passed even in that rejected attempt. Its log and source snapshots remain
unchanged. After a parser-only correction accepting these two exact spellings,
[attempt 2](../studies/results/fdtd_constitutive_absorber_acceptance_fixture_repair/cpml_witness_attempt2/result.json)
repeated the same actual source fault, accepted the intended missing-refusal
failure, restored the exact source, and passed the restored node. The supplemental
controller and its enclosing parent exited zero; the intentionally failing mutant
pytest child exited one and the restored pytest child exited zero. This gives
20 accepted mechanisms in total, without changing
production behavior, physical instruments or numerical allowances.

The [finalized acceptance](../studies/results/fdtd_constitutive_absorber_acceptance.json)
retains the original decision, allowance values and acceptance time. A separately
archived adapter proves the two test-text changes (allowance assignment, then
`allow_thin=True`), both preserved failures, and the supplemental source witness;
the eight frozen numerical helpers are unchanged. Actual finalization and
registration parent processes both exited zero.

The fixed 60-solve characterization was registered with fingerprint
`eedf93032011801d67cef6a5dd3e92f89d8a233a5a5294d607015c133888db25`.
It retains every declared row and component outcome, including the designated
duration and farther-domain controls. The campaign has no small termination-error
or spatial-convergence gate. The completed calibration configurations are known
responses when repeated; registration does not make those repeats unseen tests.

### Completed registered physical characterization

The [registered report](../studies/results/fdtd_constitutive_absorber_report.json)
contains all 60 fixed NumPy solves, with no replacements or response-based
extension. The [actual campaign parent](../studies/results/fdtd_constitutive_absorber_controls/campaign_launch_attempt1/exit.json)
exited zero. The frozen controller's
[saved-data audit](../studies/results/fdtd_constitutive_absorber_campaign_audit.json)
and its actual parent also completed successfully. An
[independent saved-array review](../studies/results/fdtd_constitutive_absorber_saved_reviews/campaign_attempt1/review.json)
reproduced every comparison from 2,880 arrays, including the four known
calibration repeats, and verified 388 input hashes. This establishes the declared
instrument checks, not a small termination error.

The 48 base solves form 24 short/reference pairs: two shapes, two placements,
two transverse polarizations and three resolutions. Eight additional solves
supply the declared longer-duration and farther-domain controls; four supply
the two zero-contrast shaped/omitted pairs. All 960 base trace-window comparisons,
288 base component/wavelength DFT comparisons, 176 farther-domain trace-window
comparisons, 48 farther-domain DFT comparisons and 72 duration DFT comparisons
are retained. Each physical sensitivity uses `M(actual-reference)/M(reference)`
as defined above; every reported denominator in this campaign is nonzero.

The following descriptive ranges cover both placements. Whole-trace ranges also
cover the two record sides; DFT ranges cover wavelengths 0.9, 1.0 and 1.1. They
show the source-polarized electric component and its transverse magnetic partner,
not an aggregate acceptance statistic. Other components and all four time windows
remain in the report and include the large ratios discussed below.

| Family / source | Cells per vacuum wavelength | E / H components | Whole E trace | Whole H trace | Plane E DFT | Plane H DFT |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cylinder / u | 20 | Ex / Hy | 8.972–8.998% | 19.463–21.139% | 4.922–25.776% | 3.860–39.499% |
| cylinder / u | 30 | Ex / Hy | 9.031–9.031% | 19.075–20.192% | 5.571–25.102% | 4.162–42.197% |
| cylinder / u | 40 | Ex / Hy | 9.038–9.042% | 19.004–19.993% | 5.722–24.915% | 4.231–43.410% |
| cylinder / v | 20 | Ey / Hx | 8.998–9.003% | 19.344–21.033% | 4.922–25.690% | 3.891–39.385% |
| cylinder / v | 30 | Ey / Hx | 9.031–9.039% | 19.039–20.192% | 5.571–25.102% | 4.189–42.203% |
| cylinder / v | 40 | Ey / Hx | 9.038–9.042% | 19.010–19.997% | 5.722–24.915% | 4.237–43.410% |
| rotated-box / u | 20 | Ey / Hz | 8.973–8.995% | 16.907–17.618% | 14.920–17.787% | 11.908–31.801% |
| rotated-box / u | 30 | Ey / Hz | 8.866–8.894% | 16.437–17.469% | 14.674–17.205% | 13.537–35.332% |
| rotated-box / u | 40 | Ey / Hz | 8.821–8.829% | 16.197–17.327% | 14.494–16.926% | 14.191–37.342% |
| rotated-box / v | 20 | Ez / Hy | 9.168–9.191% | 20.514–21.930% | 9.852–17.235% | 14.041–22.244% |
| rotated-box / v | 30 | Ez / Hy | 9.019–9.037% | 20.815–21.783% | 10.602–18.242% | 14.737–20.547% |
| rotated-box / v | 40 | Ez / Hy | 8.991–8.996% | 20.931–21.656% | 10.533–18.856% | 14.616–19.916% |

The source-polarized whole electric traces differ by about 8.82–9.19%, and their
magnetic partners by 16.20–21.93%. These finite-length differences do not disappear
on the sampled finer grids. No order is fitted: the two finite domains are being
compared at the same grid spacing, and neither is an exact open-domain reference.

Large relative values require their actual scales. The cylinder's largest base
trace-window ratio is 251.89% in the weak `plus_Ey` channel at resolution 20,
placement 0, polarization u, time 12–16: the absolute difference is
1.1519e-15 and that window's reference peak is 4.5731e-16. The rotated box's largest
base ratio is 224.38% in `minus_Ez` at resolution 20, placement 0.3, polarization u,
time 8–12: 0.072387 divided by 0.032260. Thus large ratios are not confined to
numerical-scale weak channels, and a late-window denominator is different from
a whole-record peak. No channel is discarded by these summaries.

Only resolution 30, placement 0.3 and polarization u have the duration/farther
controls, one configuration per shape. The table gives the ranges over **all**
four plane components and all three wavelengths for those controls. The longer
time or longer domain supplies the denominator; it is still a finite comparison.

| Comparison | Cylinder | Rotated box |
| --- | ---: | ---: |
| Length 3.2: time 16 vs 32 | 0.847–29.174% | 1.441–106.084% |
| Length 6.4: time 16 vs 32 | 0.952–51.392% | 2.036–54.568% |
| Length 9.6: time 16 vs 32 | 8.496–32.605% | 9.536–90.902% |
| Time 16: length 6.4 vs 9.6 | 3.758–38.590% | 2.056–100.930% |
| Time 32: length 6.4 vs 9.6 | 5.209–67.741% | 11.090–81.001% |

The time-32 controls therefore do not establish settled spectra, and the
length-9.6 calculations do not make length 6.4 an exact reference. Late-window
farther-domain trace ratios reach 339.46% for the cylinder (`minus_Hy`, time 12–16:
0.326322 / 0.096129) and 424.49% for the rotated box (`minus_Ez`, time 12–16:
0.087797 / 0.020683). These are the time-16, length-6.4 versus length-9.6 comparisons.
The designated controls do not bound the other 22 base configurations.

All 78 same-domain, common-duration prefix field/source metrics are bit-identical.
All 36 zero-contrast shaped/omitted field/source comparisons are bit-identical,
without requiring equality of the two energy observers. The four coarse positive-
material calibration worker repeats are also bit-identical; they are provenance
checks of known responses. The maximum instrument alignment comparison is
2.7061686225e-15, within the fixed 128-epsilon allowance. No exact causal-prefix
claim is made between different domain lengths, because the shared source-stamp
comparisons are not all bit-identical.

Every run retained `stable=True`, finite recorded arrays and the declared
instrument checks. Every one of the 60 nevertheless emitted
`UnsettledFieldWarning`, including the longer-time controls. The campaign also
retains 64 `CoarseMeshWarning` occurrences across 16 runs and 116
`SubpixelAccuracyWarning` occurrences across 58 runs. These counts are separate
from the earlier 24-solve calibration. Sparse driven energy curves remain saved
observations, not a source-free full-domain loss proof. The bounded update and
its finite-domain sensitivity have been exercised; low-reflection termination,
continuum accuracy, spatial order and general C9 admission have not been proved.

### Preserved regression failure and archive-format correction

[Validation attempt 1](../studies/results/fdtd_constitutive_absorber_validation_attempt1.json)
collected 4,184 cases and finished its full tier with 4,175 passed, eight skipped
and one failed (1,656 warnings; 1527.02 pytest seconds). Its actual parent exited
one, and the slow/freezer stages were not reached. The sole failure was
`test_nothing_published_points_outside_the_repository`: three preserved review
source snapshots still had a `.py` extension and contained their original
machine-local root paths, so the documentation scan treated them as published
source. This was an archive-format failure, not a failed field calculation or
a numerical-tolerance failure.

The [archive relocation receipt](../studies/results/fdtd_constitutive_absorber_archive_extension_repair_attempt1/receipt.json)
records all four saved review scripts, including the campaign review, moving
from `.py` to `.py.raw` with identical bytes and hashes. The original copy and
execution receipts retain their original names; the explicit relocation map
connects those names to the preserved payloads. Production, tests, frozen helpers,
budgets and raw response data were unchanged. The original failed full log and
parent receipt remain available.

The relocation's actual parent exited zero. A
[focused rerun of the unchanged documentation rule](../studies/results/fdtd_constitutive_absorber_controls/archive_docs_probe_launch_attempt1/exit.json)
then passed its one selected case, with actual parent exit zero. That focused
success was a focused check, separate from the complete retry reported below.
The failed first attempt remains preserved alongside that retry.

### Completed regression gates and evidence preservation

[Validation attempt 2](../studies/results/fdtd_constitutive_absorber_validation.json)
completed on the unchanged production, test, helper and budget identities after
the archive-format correction. All eight recorded collection/run/freezer commands
exited zero, and the
[actual enclosing parent](../studies/results/fdtd_constitutive_absorber_controls/validation_launch_attempt2/exit.json)
exited zero after 2714.1136 seconds.

| Gate | Observed result | Pytest elapsed time |
| --- | --- | ---: |
| Full regression | 4,184 collected; 4,176 passed, eight skipped, no failures or errors | 1382.57 s |
| Separate slow tier | 81 collected and passed; no skips, failures or errors | 1189.89 s |
| Stage-0 freezer | 143 / 143 arrays bit-exact | — |
| Stage-1 freezer | 288 / 288 arrays bit-exact | — |
| Stage-2 freezer | 140 / 140 arrays bit-exact | — |
| Stage-3 freezer | 570 / 570 arrays bit-exact | — |

The full and slow raw logs retain 1,654 and 50 warning occurrences respectively.
The four freezer comparisons reproduce 1,141 arrays in total; they were read-only,
and the golden bytes were unchanged. The original failed validation, focused
archive check, corrected complete run, source identities and raw logs remain
separate records. No numerical tolerance was widened to obtain the retry result.

These regressions support the bounded implementation and preservation of prior
behavior. They do not reduce the measured finite-domain or finite-time sensitivity,
settle the study's phasors, admit CPML overlap, or complete general C9, general C8
smoothing or the open C7 convergence gate.
