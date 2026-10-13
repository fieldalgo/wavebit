# C8: one Debye film in a dielectric

Written: 2026-10-06

Targets: `src/wavebit/fdtd/_prepare/debye_film.py`,
`src/wavebit/fdtd/_prepare/debye_resolution.py`, `src/wavebit/materials/time_domain.py`,
`src/wavebit/fdtd/_kernel/poles.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`src/wavebit/fdtd/_prepare/checks.py`.

## TL;DR

**The registered 40/80/160-cell study classifies all 24 self-orders and all
48 direct-order pairs as second within its fixed intervals.** All 20 workers
completed without a replacement or stop. This result covers the disclosed
Debye material and two placements, not every admitted film. The bounded
implementation and its numerical gates are complete on the feature branch.
All 247 targeted cases and nine actual production mutation witnesses pass
after strengthening one masked refusal test. The full tier passes with
2,994 passed and eight skipped; all 81 slow cases and 1,141 frozen arrays
also pass. The initial documentation-evidence packaging failure and its
correction remain recorded below.

The implemented scope is one concrete isolated flat `Box`, containing one
`Debye` pole, in a plain dielectric host. Normal mixed cells retain a
Debye law with a smaller relaxation time and a transformed static increment.
Tangential cells retain the film relaxation time and scale its increment
by occupancy. Both use the existing split polarization recurrence and its
spatially weighted energy branch. Unlike the prior Drude/Lorentz reduction,
the transformed normal relaxation can tighten the material time-step policy.

This extends a different material family from the earlier
[conductive film](fd-conductive-film.md) and
[Drude/Lorentz film](fd-single-pole-film.md). Those measurements retain their
original scopes. General C8 smoothing is incomplete, and the
[C7 metasurface verdict](fd-metasurface-validation.md) remains unpassed.

## Bounded implementation contract

The public declaration is one
`fd.Structure(fd.Box(...), fd.Dispersive(eps_inf=..., poles=(fd.Debye(...),)))`
with the default voxel discretization. All of the following restrictions
are part of this scope:

| Item | Bounded case |
| --- | --- |
| Geometry | Exactly one concrete public `Box`, `angle=0`, `range=None`, with no partner structure or junction. |
| Extent | Exactly two complete grid axes are uniform through the film, with at least one `dx` of overhang beyond each transverse domain face. The remaining axis is the normal. |
| Thickness | At least one `dx`, so both film faces cannot cut the interior of one electric dual cell. |
| Clearance | The normal support is strictly inside the domain, and the entire support stays clear of every absorbing layer. A transverse absorber is outside this scope. |
| Film | Exact `Dispersive` containing exactly one exact `Debye`, finite positive isotropic `eps_inf`, and `mu=1`. |
| Pole | Concrete finite `delta_eps >= 0` and `tau > 0`. The zero-increment case is admitted. |
| Host | Exact plain `Material`, concrete finite positive isotropic permittivity, `mu=1`, and no loss or pole. |
| Initial state | No initial field. Sources and monitors retain their existing placement and material restrictions. |
| Reuse | Prepared geometry, material, grid and required polarization memory must match; a plan for another material path cannot bypass admission. |

The rules apply after permuting x, y and z. The transverse overhang accounts
for node-centred Yee dual cells extending beyond a domain face. The film
fraction is the analytic interval overlap at each electric component's own
Yee location, with exact empty/full endpoint predicates. A small nonzero
mixed fraction is not rounded into an endpoint by a tolerance band.

Multiple poles, pole subclasses, a dispersive or conducting host, anisotropy,
magnetism, rotated or curved bodies, lossy junctions, and geometry/material/
source-band differentiation remain outside the contract. Source-amplitude
differentiation is not a validation claim of this scope. Existing snapped
material paths and stencil certificates retain their own contracts.

## Exact harmonic and arithmetic laws

Use the repository convention `exp(-i*omega*t)` and put `s=-i*omega`.
Let `f` be the film fraction, `g=1-f`, `a` the film high-frequency
permittivity, `b` the host permittivity, `Delta=delta_eps` and `tau` the
film relaxation time. Then

```text
eps_film(s) = a + Delta/(1 + tau*s)
A = f*b + g*a
B = A + g*Delta.
```

Here `Delta` is a static permittivity increment; it is not the
angular-frequency-squared strength of a Lorentz oscillator. With `a,b>0`,
`Delta>=0` and `tau>0`, the normal harmonic mixture reduces exactly to

```text
eps_n(s) = 1/(f/eps_film(s) + g/b)
         = b*(a*(1 + tau*s) + Delta)/(A*(1 + tau*s) + g*Delta)
         = a*b/A + Delta_n/(1 + tau_n*s)

Delta_n = f*b*b*Delta/(A*B)
tau_n   = A*tau/B.
```

The tangential arithmetic mixture is

```text
eps_t(s) = f*a + g*b + f*Delta/(1 + tau*s).
```

Each electric component therefore retains one passive relaxation:

| Component | `eps_inf_eff` | `delta_eff` | `tau_eff` |
| --- | --- | --- | --- |
| Normal | `a*b/A` | `f*b*b*Delta/(A*B)` | `A*tau/B` |
| Tangential | `f*a+g*b` | `f*Delta` | `tau` |

The high-frequency normal limit is `a*b/A`; its static limit is
`b*(a+Delta)/B`. Since `B>=A>0`, `tau_n<=tau`. Equal high-frequency
permittivities do not remove the normal relaxation-time change when the
film still has a nonzero Debye increment.

At `f=0` the material is the host and has no active memory. At `f=1` it is
the original film. At `Delta=0`, the normal and tangential coefficients are
the ordinary lossless harmonic and arithmetic mixtures and all memory
drive and energy weights vanish. The implementation must handle this case
before dividing by the inactive increment or relaxation time squared.
Non-finite derived active coefficients, and active positive quantities that
underflow to zero, are refused rather than silently replaced by a different
material law.

## Split memory and discrete frequency response

For either component write its local parameters as `epsilon`, `delta` and
`T`. Trapezoidal integration of `T*dP/dt + P = delta*E` gives

```text
u = 2*T/(2*T + dt)
v = dt/(2*T + dt)
gain  = u - v
c     = delta*v
drive = 2*delta*u*v

P^(n+1) = gain*P^n + c*(E^(n+1) + E^n).
```

The `E^(n+1)` term is retained by storing the shifted memory `Q=P-c*E`:

```text
Q^(n+1) = gain*Q^n + drive*E^n
eps_update = epsilon + c
hold = 0.
```

This fits the existing `PoleTerm` recurrence and polarization-current
coupling to Ampere's law. The electric update includes the instantaneous
shift `c`; the memory supplies the remaining delayed response. Dropping the
new electric field from the unsplit equation would define a different,
first-order method.

The bounded ratios `u` and `v` use `m=max(T,dt/2)`, with numerator ratios
`T/m` and `(dt/2)/m` divided by their sum, so forming `2*T+dt` need not
overflow. The drive uses `2*delta*u*v` directly,
avoiding reconstruction from `1+gain` when `gain` rounds close to minus
one. Inactive cells have zero memory coefficients and energy weights. A film
memory term still identifies the admitted geometry when the declared
increment is zero.

For a coefficient-level frequency oracle, both exact mixture formulas hold
with the trapezoidal frequency

```text
omega_hat = 2*tan(omega*dt/2)/dt
s_hat = -i*omega_hat.
```

That identity checks the recurrence's material response. Physical film
measurements must instead compare with the continuum response at the actual
frequency, including spatial and temporal discretization errors.

## Local energy and the conservative time-step policy

At the energy observer's time level the physical polarization is
`P^n=Q^n+c*E^n`, using the previous-memory view aligned with `E^n`.
The existing spatial Debye branch adds the local contribution

```text
inverse_delta * |Q^n + c*E^n|**2 - c*|E^n|**2
inverse_delta = 1/delta.
```

Weights are applied before summing cells and components, including clipped
regions. The subtraction cancels the `c` already present in the field
permittivity weight. Inactive weights are exactly zero; they are not
reconstructed from `1-gain*gain`. For an active relaxation, the trapezoidal
energy identity has the nonnegative dissipated amount
`2*T*|P^(n+1)-P^n|**2/(dt*delta)` per step in the repository normalization.
This identity is not a propagation or long-time stability measurement.

The repository retains a conservative Debye material-rate policy. For this
film it must include the actual transformed cells, whose rates are

```text
R_bulk = Delta/(a*tau*tau)
R_n(f) = delta_n/(tau_n*tau_n*eps_inf_n)
       = f*b*Delta*B/(a*A*A*tau*tau)
R_t(f) = f*Delta/((f*a+g*b)*tau*tau)

R_film = max(R_bulk, max_actual_cells_and_components(R_eff))
dt_bound = 1/sqrt(1/dt_nondispersive**2 + R_film/4).
```

The tangential rate cannot exceed the bulk rate, but the normal ratio
`R_n/R_bulk=f*b*B/(A*A)` can exceed one for positive increment. The
Drude/Lorentz film's unchanged-bulk-rate identity therefore does not carry
over. The maximum is over the actual component occupancies. Bulk and
transformed rates describe this same singleton pole and are combined by a
maximum, not added as independent poles.

For this matching film only, the planner bypasses the ordinary scalar
`pole.rate()` evaluation in favour of the scoped rate. A zero increment
contributes zero before evaluating `tau*tau`, so an extreme but finite
inactive relaxation time cannot create a spurious division by zero or
overflow. Unrelated material paths keep their existing policy. The bound
preserves that policy; passive trapezoidal relaxation by itself does not
intrinsically require this additional time-step restriction.

## Bulk spectral resolution guard

For a positive angular frequency put `x=omega*omega` and
`C=Delta*(2*a+Delta)`. The bulk Debye law satisfies

```text
|eps(omega)|**2 = a*a + C/(1 + tau*tau*x)
|k(omega)|**4 = x*x*(a*a + C/(1 + tau*tau*x)).

d(|k|**4)/dx = 2*a*a*x + C*x*(2 + tau*tau*x)/(1 + tau*tau*x)**2 > 0.
```

Thus the largest bulk modulus on any positive source-frequency interval is
at its upper endpoint. A source with `f0=1/band.wavelength`, width `df`
and multiplier `m=band.sigmas` declares
`[2*pi*(f0-m*df), 2*pi*(f0+m*df)]`; its lower endpoint must be positive.
For each declared source band, the guard uses
`K=omega_max*sqrt(abs(eps(omega_max)))`, with the host bound
`sqrt(b)*omega_max`, and requires `dx*K<=2*pi/8`. This avoids treating the
high-frequency permittivity alone as the wavelength-setting material.
It is a sufficient bulk phase/attenuation bound, not a physical scattering
acceptance test.

The scoped contract also checks each declared monitor wavelength, the
`Resolution` wavelength and an explicit PML wavelength in both film and
host. The sampled minima are eight cells per phase wavelength and one per
skin depth, with the existing warnings below twelve and two respectively.
All band and wavelength values must be concrete, finite and positive;
source-band traces are outside this scope. A scene without usable spectral
metadata can only receive the corresponding unresolved-resolution warning.

These checks concern bulk material scales. They do not certify every
effective interface or oblique longitudinal scale, a convergence order, or
the adequacy of a finite run and its absorbers. Those require their own
physical measurements and controls.

## Disclosed three-backend calibration

The first calibration completed three sequential runs on 2026-10-06,
10:39:57 to 10:40:36 UTC, all exit 0, without adapting a row. Each uses
`eps_inf=2`, `delta_eps=6`, `tau=0.15`, film thickness 0.2 and a vacuum
host. The grid has 40 cells per centre vacuum wavelength, face fraction
0.3, p polarization and 35-degree incidence at centre wavelength 1. The
transverse wavevector stays fixed across wavelengths 0.9, 1.0 and 1.1.
Each run lasts 120 periods with 24-cell PMLs, Courant factor 0.5 and
source `df=0.1`, cutoff `1e-6`.

The reference in `studies/fdtd/debye_film/fdtd_debye_film_probe.py` uses an explicit rational
Debye permittivity and complex Airy reflection/transmission, independently
of the production material evaluator. Its analytic preflight controls pass
without field solves. The NumPy response gives:

| Wavelength | Absolute complex `r` error | Absolute complex `t` error |
| --- | --- | --- |
| 0.9 | 0.003375867488 | 0.008260099430 |
| 1.0 | 0.003564005407 | 0.007349643666 |
| 1.1 | 0.003360434433 | 0.006416354580 |

The maxima across all three backends are 0.0035640054068682516 for `r`
and 0.008260099429717062 for `t`. These are absolute complex-amplitude
errors, not relative errors or power errors. The inherited candidate
ceilings of 0.005 for `r` and 0.010 for `t` were accepted unchanged after
reviewing these disclosed responses. They were not widened, and their
acceptance is not presented as a test of previously unknown calibration
data.

JAX and Torch were compared with NumPy at every complex `r` and `t`
sample using `abs(value-reference)/max(1,abs(reference))`. The largest
scaled difference is 9.710485573708405e-16, within the pre-existing
`128*float64_epsilon` budget of 2.842170943040401e-14. This supports
backend agreement for this configuration, not for every admitted parameter,
normal orientation or derivative.

All three runs report stable fields and finite energy, with 1,041 energy
samples per run. Their largest reflection and transmission monitor decay
ratios are 8.614203e-9 and 2.458165e-11 respectively. Small decay ratios
describe settling of these recorded responses; they do not bound absorber
reflection or determine a spatial convergence order. Each run retains its
`SubpixelAccuracyWarning`: the spectral guard certifies bulk scales only,
and observables still need refinement, duration and absorber controls.

The recorded runtime is Windows with Python 3.12.14, NumPy 2.5.1,
JAX/JAXlib 0.11.0 and Torch 2.13.0. The OMP, OpenBLAS and MKL thread limits
are each one. No hardware performance comparison is claimed.

The manifest is
`studies/fdtd/debye_film/records/fdtd_debye_film_calibration_attempt1.json`; raw results,
logs and frozen probe/controller copies are under
`studies/results/fdtd_debye_film_calibration_attempt1/`. All six raw
result/log hashes and both frozen-source hashes match the manifest. Each
run confirms unchanged before/after source identity, and the controller
records `source_unchanged=true`. The FD source digest is
`6c2ac462df74f2ba1e25d090ce0d0b9b7bf356e7dafde59fae27f311f2d7f0c2`,
using sorted paths relative to `src/wavebit/fd`, NUL separators and
LF-normalized source bytes. These measurements belong to that recorded
source and are known exposure for subsequent registered work.

## Targeted checks and retained mutation attempts

Both combined targeted attempts collected and passed all 247 cases, exit 0,
with no skips, xfailed or xpassed. The total comprises 99 new Debye cases
in nine test functions, 94 existing Drude/Lorentz film cases, 53 existing
conductive-film cases and one existing stencil refusal case.

| Attempt | Targeted result | Test revision |
| --- | --- | --- |
| 1 | 247 passed; pytest 44.10 seconds | Initial Debye test file. |
| 2 | 247 passed; pytest 43.81 seconds | The source-band assertion independently checks the band certificate as well as retaining the public planning refusal. |

The manifests, collection/run logs and actual test/controller snapshots are
under `studies/results/fdtd_debye_film_controls/`, named
`targeted_attempt1` and `targeted_attempt2`. The source remains the recorded
calibration source throughout both attempts. The snapshot and log hashes
match their manifests.

The first mutation attempt caught eight of nine deliberate production
errors. The mutant replacing the dispersive source-band bound with the
high-frequency dielectric value survived: its selected test exited 0.
A later diagnostic wavelength guard also refused that scene, so a generic
public-planner refusal did not isolate whether the source-band certificate
had run. This was a gap in the test's discriminatory power, not an accepted
mutation witness.

The repaired `band-high` fixture obtains the same grid from an otherwise
matching zero-increment scene and directly calls the spectral validator on
the dispersive scene. It requires the source-band-specific refusal before
later planner diagnostics can mask an omitted contribution. The public
planning-refusal assertion remains. No production formula, source bytes or
numerical tolerance changed for this repair.

The second mutation attempt catches all nine actual production mutations
with assertion or missing-refusal failures, each exit 1:

| Mutated behaviour | Required witness |
| --- | --- |
| Omit the normal relaxation-time transformation | Series/parallel response test fails. |
| Shift lossless tangential permittivity | Zero-increment material test fails. |
| Omit the instantaneous energy subtraction | Local split-energy test fails. |
| Reduce memory drive to one quarter | Complex Airy film test fails. |
| Remove the initial-field refusal | Unsupported-scene test detects a missing refusal. |
| Remove the prepared signature check | Plan-reuse test detects a missing refusal. |
| Detach traced increment silently | Traced-parameter test detects a missing refusal. |
| Omit actual transformed-node rates | Material-rate test fails. |
| Omit the dispersive source-band contribution | The isolated band-certificate test detects a missing refusal. |

Every changed production file was restored byte for byte. The final restored
Debye test file collects and passes all 99 cases, exit 0, in 42.22 seconds
as reported by pytest. The canonical record is
`studies/fdtd/debye_film/records/fdtd_debye_film_mutations.json`; both original attempts are
retained as `studies/fdtd/debye_film/records/fdtd_debye_film_mutations_attempt1.json` and
`studies/fdtd/debye_film/records/fdtd_debye_film_mutations_attempt2.json`, with separate raw
logs, original/mutant snapshots and test/harness snapshots. Their raw and
snapshot hashes match, and both source identities are unchanged after
restoration. The surviving first-attempt mutant remains visible in that
record; it is not counted among the nine accepted second-attempt witnesses.

## Registered physical campaign

The 20-worker physical campaign was registered at
2026-10-06T10:51:06 UTC before its new responses, with fingerprint
`281293ccfa40d48716b21623ec9d91c671d10d635e51e221782b763154d7179a`.
The registration is `studies/fdtd/debye_film/records/fdtd_debye_film_registration.json`.
It uses the disclosed material, face fractions 0.3 and 0.1, both s and p
polarization, and NumPy with an exact `Axis1D` transverse derivative.
Each of four placement/polarization groups has five fixed rows:

| Role | Cells per centre wavelength | Periods | PML cells |
| --- | --- | --- | --- |
| Coarse base | 40 | 120 | 24 |
| Middle base | 80 | 120 | 24 |
| Duration control | 80 | 240 | 24 |
| Absorber control | 80 | 240 | 48 |
| Fine base | 160 | 120 | 24 |

All 24 placement/polarization/wavelength/complex-coefficient cells are
classified independently. For each cell the fixed middle-grid proxy is

```text
u = abs(v80_240_PML24 - v80_120_PML24)
  + abs(v80_240_PML48 - v80_240_PML24).
```

Self-order is `log2(abs(v40-v80)/abs(v80-v160))`; its sensitivity interval
is constructed by adding and subtracting `2*u` from each adjacent change.
Direct order is `log2(error_coarse/error_fine)`; its sensitivity interval
adds and subtracts `u` from each absolute complex error. Both adjacent
changes or errors must exceed `1e-10`, and
`2*u/min(adjacent changes or errors)` must be at most 0.1. A resolved
interval is second only when the entire interval lies in `[1.8, 2.3]`;
an upper endpoint at most 1.75 is low, other resolved intervals are mixed,
and failed noise gates are unresolved.

This proxy is empirical, not a rigorous bound on fine-grid noise. Any
allowed replacement or stop follows the fixed registration; an order label
cannot trigger additional mesh, duration or geometry choices. The matching coarse
p/fraction-0.3 row repeats a known calibration response and is disclosed as
such, not treated as independent corroboration. Targeted and mutation
exposures are likewise recorded before registration.

### Measured result and scope of the conclusion

All 20 registered workers exited 0. All four groups select their original
120-period base rows at every mesh; there was no replacement or stop.
Every row reports stable fields, finite energy and unchanged source identity.
All 40 raw result/log hashes match the ledger. The procedural audit passes
and recomputes all 24 cells.

| Classification | Self-order cells, 40/80/160 | Direct pairs, both mesh pairs | Final direct cells, 80 to 160 |
| --- | --- | --- | --- |
| Second | 24 | 48 | 24 |
| Low, mixed or unresolved | 0 | 0 | 0 |
| Total | 24 | 48 | 24 |

The self point estimates range from 1.991828 to 2.255185. The following
table gives the smallest lower endpoint and largest upper endpoint among
the six cell intervals in each placement/polarization group; each range is
an envelope of separate sensitivity intervals, not a statistical confidence
interval.

| Face fraction | Polarization | Self-interval envelope | Final direct-interval envelope |
| --- | --- | --- | --- |
| 0.3 | s | [2.003711, 2.219461] | [2.000774, 2.107204] |
| 0.3 | p | [1.987983, 2.272004] | [1.988840, 2.153655] |
| 0.1 | s | [2.005530, 2.039488] | [2.001116, 2.011915] |
| 0.1 | p | [1.992764, 2.048124] | [1.985806, 2.021544] |

Across all 24 complex `r`/`t` cells, the largest absolute errors are:

| Cells per centre wavelength | Largest absolute complex error |
| --- | --- |
| 40 | 0.008585756353 |
| 80 | 0.002146618968 |
| 160 | 0.000537387641 |

The largest time/PML proxy is 2.834511e-6, for p reflection at fraction 0.3,
wavelength 1.1: its duration contribution is 2.727898e-6 and its absorber
contribution is 1.066127e-7. The largest noise shares are 0.00960248 for
self-order and 0.03230238 for direct order, both below the registered 0.1
limit. Those maxima occur at p reflection, fraction 0.3, wavelength 0.9.
The largest recorded monitor decay ratio across all workers is
1.024557e-8. These diagnostics support the declared finite-run classification;
they do not turn the empirical proxy into a rigorous noise bound.

The repeated known coarse p/fraction-0.3 row has exactly the same recorded
physical values as the NumPy calibration. Only its start/end timestamps and
wall time differ. This is reproducibility of a known response, not an extra
independent validation arm. Both the original calibration and the newly
recorded worker remain preserved.

The complete per-cell values, noise terms and intervals are in
`studies/fdtd/debye_film/records/fdtd_debye_film_report.json`. The worker ledger is
`studies/fdtd/debye_film/records/fdtd_debye_film_campaign.jsonl`, with raw files under
`studies/results/fdtd_debye_film_campaign_jobs/`. The procedural audit is
retained in
`studies/fdtd/debye_film/records/fdtd_debye_film_controls/campaign_audit_attempt1.log`.

The measured decision is second-order behaviour under the registered rule
for this material, thickness, incidence, two face fractions, three wavelengths
and 40/80/160 ladder with its exact transverse derivative. It does not
establish an asymptotic theorem, arbitrary-material or arbitrary-geometry
accuracy, gradient correctness, or a general absorber stability contract.
The bulk spectral guard retains its narrower scale-resolution meaning.

## Regression gates and evidence preservation

The completed sequential validation controller exits 0. The final results
are recorded in `studies/fdtd/debye_film/records/fdtd_debye_film_validation.json`:

| Gate | Collected or compared | Result | Exit |
| --- | --- | --- | --- |
| Full regression | 3,002 cases | 2,994 passed, eight skipped | 0 |
| Separate slow tier | 81 cases | 81 passed, zero skipped | 0 |
| Read-only freezer 0 | 143 arrays | All bit-exact | 0 |
| Read-only freezer 1 | 288 arrays | All bit-exact | 0 |
| Read-only freezer 2 | 140 arrays | All bit-exact | 0 |
| Read-only freezer 3 | 570 arrays | All bit-exact | 0 |

Both test tiers have zero xfailed and zero xpassed. Pytest reports 2,735.35
seconds for the full tier and 1,306.77 seconds for the slow tier; the
controller records 2,736.2631133 and 1,307.3824764 seconds respectively.
The four freezers reproduce 1,141 arrays in total without rewriting or
rebaselining a golden.

The first full attempt stopped on one documentation contract failure:
3,002 collected, 2,990 passed, eight skipped, one failed and three not
reported after the `maxfail` stop; pytest exited 2 and the launcher exited 1.
A redundant plain-text copy of the documentation contract test contained
that test's intentionally exempt path-pattern fixtures. The published-text
scan encountered the copied fixtures outside the live test's exemption.
The exact snapshot is now retained as a `.py.txt.gz` archive; decompression
reproduces its original bytes and SHA-256. The live documentation contract,
production code, tests, numerical tolerances and acceptance rules did not
change. The complete full tier was then repeated and passed as recorded
above.

The failed manifest and logs remain under
`studies/results/fdtd_debye_film_validation_attempt1/`. The mapping from
the original snapshot name to its compressed location, checksums and
byte-exact decompression verification is in
`studies/fdtd/debye_film/records/fdtd_debye_film_evidence_packaging.json`. The preserved
failure records and compressed/uncompressed hashes match that annotation.
The prevalidation comparison also retains unchanged maps for 95 production
files, 80 test files, four golden files and 428 earlier evidence files.

The validation controller's unchanged FD source digest is
`aec5df52a95bd470bb43d966d14bffbcad1abe3b9bb0a8a02dfab0cead3aec26`.
Its inherited helper hashes sorted paths relative to `src/wavebit/fd` and
LF-normalized source bytes without delimiters. The campaign digest above
uses NUL separators; the different algorithms yield different digest
strings for the same source.

These numerical gates do not extend the physical scope of the registered
study. The prior films' results retain their original material scopes;
this campaign does not resolve their open items. General C8 smoothing
remains incomplete and C7 remains unpassed.
