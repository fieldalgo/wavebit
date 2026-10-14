# C8: one Drude or Lorentz film in a dielectric

Written: 2026-10-06

Targets: `src/wavebit/fdtd/_prepare/single_pole_film.py`, `src/wavebit/materials/time_domain.py`,
`src/wavebit/fdtd/_kernel/poles.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/single_pole_resolution.py`.

## TL;DR

**The selected finer study resolves the two Drude self-order exceptions, but
three Lorentz final direct-order cells remain noise-unresolved.** The initial
40-row campaign retains its four self-order exceptions; the separate follow-up
has 11 of 12 self-orders and 9 of 12 final direct orders classified as second.
Targeted tests, nine mutation witnesses, the full and slow regression tiers,
and all four frozen-reference checks pass. This bounded C8 implementation
and its required numerical gates are complete on the feature branch. Its scope is
one isolated flat film containing one passive Drude or Lorentz pole in a
plain dielectric host. The exact harmonic normal mixture remains a single
oscillator, with a shifted resonance and cell-dependent strength. The
tangential arithmetic mixture retains the original resonance and scales its
strength by the film fraction.

This reduction permits reuse of the existing second-order polarization
recurrence. It requires spatial energy weights, including the electric-field
coupling term, but no additional material rate beyond the existing CFL bound.
The formulas below establish that reduction. The field results include
unresolved cases and do not establish universal second-order accuracy.

The earlier [conductive-film result](fd-conductive-film.md) concerns a different
material law and does not validate this feature. General C8 smoothing remains
incomplete, and the [C7 metasurface verdict](fd-metasurface-validation.md)
remains unpassed.

## Bounded implementation contract

The public declaration is one
`fd.Structure(fd.Box(...), fd.Dispersive(eps_inf=..., poles=(pole,)))`, where
`pole` is one `fd.Drude` or `fd.Lorentz` and discretization is the default
`'voxel'`. The contract is implemented; the completed regression gates and
the physical limits below are recorded separately.

| Item | Bounded case |
| --- | --- |
| Geometry | One concrete public `Box`, `angle=0`, `range=None`, with no second structure or junction partner. |
| Extent | Exactly two axes span the complete grid, with at least one `dx` of overhang beyond each transverse face; the remaining axis is the film normal. |
| Thickness | At least one `dx`, so both interfaces do not cut the interior of the same electric dual cell. |
| Clearance | The normal support is strictly within the domain, and all support is outside every absorbing layer. A transverse absorber does not fit this scope. |
| Film | Exact `Dispersive` containing exactly one exact `Drude` or `Lorentz`, with finite positive isotropic `eps_inf` and `mu=1`. |
| Pole | Concrete finite parameters, nonnegative damping and strength; Drude plasma frequency and Lorentz resonance frequency are positive. A zero-strength Lorentz pole is allowed. |
| Host | Plain, concrete, isotropic lossless `Material`, with finite positive permittivity and `mu=1`. |
| Initial state and reuse | No initial field or traced geometry/material values. The prepared plan must retain exactly the film memory belonging to its geometry, material and grid. |

These rules apply to x-, y- and z-normal films. The transverse overhang covers
node-centred Yee dual cells extending beyond domain faces. The existing
source, monitor, resolution, feature-size and absorber checks still apply.
The lowering builds NumPy coefficient profiles and converts them to the
selected runtime; backend correctness remains a validation question until
the corresponding cases are checked. The targeted evidence below covers
specified NumPy, JAX and Torch cases, rather than every admitted combination.

This scope excludes multiple poles, Debye poles, a dispersive host, anisotropy,
magnetism, rotated or curved bodies, lossy junctions, and differentiation of
geometry, material parameters or source bands. These guards do not expressly
refuse source-amplitude differentiation; no amplitude-gradient validation gate
is claimed here.
Existing cell-snapped material paths retain their own contracts. Unsupported
voxel scenes cannot bypass the restrictions by reusing a non-film plan;
existing stencil certificates retain their own refusal path.

## Exact complex mixture

Let `f` be the film fraction of an electric component's dual cell, `g=1-f`,
`a` the film's high-frequency permittivity and `b` the host permittivity.
With `s=-i*omega` for the repository's `exp(-i*omega*t)` convention, write

```text
eps_film(s) = a + S/L(s)
L(s) = s*s + gamma*s + omega0_squared.

Drude:   omega0_squared = 0,             S = omega_p*omega_p.
Lorentz: omega0_squared = omega_0**2,    S = strength.
```

The `strength` parameter has units of angular frequency squared; for a Lorentz
pole its static permittivity increment is `strength/omega_0**2`.
Nonnegative damping and strength give the passive oscillator family, including
zero damping. A positive high-frequency permittivity does not imply that the
frequency-dependent real permittivity stays positive.

The normal harmonic and tangential arithmetic laws are the axis-aligned
specialization of Eqs. (1) and (2) in
[Deinega and Valuev, Optics Letters 32, 3429–3431 (2007)](https://doi.org/10.1364/OL.32.003429).
For `A=f*b+g*a`, direct rational reduction gives

```text
eps_t(s) = f*a + g*b + f*S/L(s)

eps_n(s) = 1/(f/eps_film(s) + g/b)
         = b*(a*L(s) + S)/(A*L(s) + g*S)
         = a*b/A + (f*b*b*S/(A*A))/(L(s) + g*S/A).
```

Thus each electric component has one oscillator,
`eps_eff(s)=eps_inf_eff + S_eff/(s*s+gamma*s+Omega_squared)`:

| Component | `eps_inf_eff` | `S_eff` | `Omega_squared` |
| --- | --- | --- | --- |
| Tangential | `f*a+g*b` | `f*S` | `omega0_squared` |
| Normal, mixed | `a*b/A` | `f*b*b*S/(A*A)` | `omega0_squared+g*S/A` |

Damping is unchanged. The normal interface shifts a Drude pole to a nonzero
resonance wherever `0<f<1`; equal high-frequency permittivities do not remove
that shift. No pole fit, off-diagonal constitutive tensor, instantaneous
permittivity correction or separate conductivity term is introduced.

The fraction is the analytic interval overlap at each electric component's
own Yee position. Exact empty and full cells use the host and the original
film respectively. Endpoint predicates retain every nonzero mixed fraction;
there is no tolerance band that snaps a small fraction to zero or one.
Derived coefficients outside finite float64 range are refused explicitly.

## Discrete oscillator and field coupling

Central differences of
`P'' + gamma*P' + Omega_squared*P = S_eff*E` give

```text
denominator = 1 + gamma*dt/2
step_gain   = (2 - Omega_squared*dt*dt)/denominator
hold        = (1 - gamma*dt/2)/denominator
drive       = S_eff*dt*dt/denominator

P^(n+1) = step_gain*P^n - hold*P^(n-1) + drive*E^n.
```

`DispersiveFilmTerm` uses the existing `PoleTerm` recurrence, with spatial
coefficients. Its polarization increment enters Ampere's law as
`-(P^(n+1)-P^n)`. The kernel expresses this as `-(dx/dt)*delta` before the
ordinary electric-update coefficient applies its `dt/(dx*eps)` factor.
There is no new simultaneous field/material solve.

For a discrete-frequency coefficient oracle, the denominator is

```text
Omega_squared - 4*sin(omega*dt/2)**2/dt**2
              - i*gamma*sin(omega*dt)/dt.
```

Both mixtures retain the same rational identity after this replacement of
the time derivatives. Agreement with that oracle checks the recurrence's
material law. A physical refinement study must compare with the continuum
film at the actual frequency and include the solver's spatial and temporal
discretization errors.

Empty cells and zero-strength Lorentz cells have zero drive and zero local
energy weights. A single film term remains reserved even when the Lorentz
strength is zero, so prepared-plan identity stays explicit.

## Local energy and the CFL identity

At the observer's time level, let `P_new=P^(n+1)`, `P_old=P^n`,
`delta=P_new-P_old` and `E=E^n`. In the repository's energy normalization,
the local contribution of an active oscillator is

```text
|delta/dt|**2 / S_eff
+ (Omega_squared/S_eff)*Re(P_new*conj(P_old))
- Re(E*conj(delta)).
```

The implementation supplies `1/S_eff`, `Omega_squared/S_eff` and an active
mask directly. The observer applies these weights before reducing spatial
cells and components, including clipped regions. Inactive weights are zero,
including the field-coupling weight. This avoids divisions at empty cells or
zero strength and avoids reconstructing a small resonance by subtracting
nearly equal recurrence coefficients.

Explicit oscillator metadata also distinguishes this second-order memory
when `gamma*dt=2` makes `hold=0`; that coefficient value alone does not identify
a first-order Debye law. The energy formula is an identity of the discrete
recurrence, not a substitute for a propagation or long-time stability study.

For a normal mixed cell the relevant oscillator rate obeys the exact identity

```text
Omega_squared + S_eff/eps_inf_eff
  = omega0_squared + g*S/A + f*b*S/(a*A)
  = omega0_squared + S/a.
```

For a tangential cell,
`Omega_squared+S_eff/eps_inf_eff = omega0_squared+f*S/(f*a+g*b)`, which is
no greater than the same bulk rate. The interface resonance therefore does
not add a larger material rate than the existing planner already includes.

More explicitly, put `q=sqrt(S_eff/eps_inf_eff)` in the normalized coupled
Maxwell/oscillator operator `B=[[C_eps,-q],[0,Omega]]`. Splitting `B B*` into
the curl block and local oscillator outer products gives

```text
||B||**2 <= ||C_eps||**2 + max_cell(Omega_squared+q*q).
```

The existing bound requiring `dt*||B||<2` therefore covers this additional
spatial oscillator. Centred nonnegative damping dissipates its leapfrog
energy. This argument does not extend the existing absorber contract or
establish a general curved-interface stability result.

## Conservative bulk spectral guard

The scoped guard in `src/wavebit/fdtd/_prepare/single_pole_resolution.py` considers
every source's declared positive interval and every requested spectral
sample. It uses the bulk dispersive response, rather than only the
high-frequency permittivity. Band values and wavelengths must be concrete,
finite and positive; traced bands, including Torch forward-mode values,
are refused.

For a source with centre frequency `f0=1/band.wavelength`, width `df` and
cutoff multiplier `m=band.sigmas`, the interval is
`[2*pi*(f0-m*df), 2*pi*(f0+m*df)]` in angular frequency. Its lower endpoint
must be positive. Put `x=omega**2`, `x_max=omega_max**2`, and define

```text
R(x) = x/sqrt((omega0_squared-x)**2 + gamma*gamma*x)
K*K  = a*x_max + S*max_interval(R).
```

The triangle inequality gives `|k_bulk(omega)| <= K` throughout that
interval. The maximum of `R` is found from its two endpoints and the sole
possible interior stationary point,

```text
x_star = 2*omega0_squared**2/(2*omega0_squared-gamma*gamma),
```

included only when its denominator is positive and it lies inside the
interval. The implementation evaluates the equivalent scaled expression to
avoid overflow while forming the candidate; it refuses non-finite derived
quantities rather than silently dropping a possible interior maximum.
A positive-strength, undamped Lorentz resonance inside the
interval is refused because no finite bound exists there. A zero-strength
Lorentz material instead uses the constant-permittivity result directly,
including at its nominal resonance; no inactive `0/0` is evaluated.

The host bound is `sqrt(b)*omega_max`. Both film and host must satisfy
`dx*K <= 2*pi/8`. Because `Re(k)` and `Im(k)` are bounded by `|k|`, this is
a sufficient bound on bulk phase and attenuation scales across the declared
band. It can reject a mesh that a less conservative analysis would admit.

Independently, every wavelength requested by any monitor, by `Resolution`,
or explicitly by a PML is evaluated in the film and host. With the passive
square root `k=omega*sqrt(eps(omega))`, the required minima are

| Sampled scale | Minimum cells |
| --- | --- |
| Phase wavelength, `2*pi/Re(k)` | 8 |
| Skin depth, `1/Im(k)` | 1 |

A zero real or imaginary part has no finite constraint of its respective
kind. An undamped, positive-strength resonance requested exactly is refused.
Notes warn below twelve cells per phase wavelength or two per skin depth.
With no source band and no declared wavelength, a warning records that the
bulk mesh resolution could not be judged. Scoped resolution and proximity
notes use the same zero-strength-safe material evaluation.

This guard bounds bulk material scales. It does not certify every effective
normal-interface resonance, every oblique longitudinal wavelength, or the
accuracy of a scattering observable. Those still require physical refinement
and time/absorber controls.

## Disclosed calibration

Two NumPy field rows were measured on 2026-10-06 at 08:31 UTC, one for each
pole family. These are disclosed calibration responses, not a registered
convergence verdict. Both use a thickness-0.2 film in vacuum, `eps_inf=1`,
`gamma=1`, 40 cells per centre wavelength, face fraction 0.3, p polarization,
35-degree incidence at centre wavelength 1, and the same fixed transverse
wavevector at wavelengths 0.9, 1.0 and 1.1. Each run lasts 60 periods with
24-cell PMLs, Courant factor 0.5 and source `df=0.1`, cutoff `1e-6`.

The Drude material has `omega_p=9`; the Lorentz material has `omega_0=8` and
`strength=40`. The independent complex Airy reference is in
`studies/fdtd/single_pole_film/fdtd_single_pole_film_probe.py`. The recorded runtime is Python
3.12.14 with NumPy 2.5.1 on a Windows CPU runtime; the three numerical thread
environment limits are each one. These rows establish no hardware performance
comparison.

| Film | Largest absolute complex `r` error | Largest absolute complex `t` error | Largest monitor decay ratio |
| --- | --- | --- | --- |
| Drude | 0.002942635749 | 0.001145744654 | 2.344237e-9 |
| Lorentz | 0.002826098019 | 0.005957733867 | 8.541341e-8 |

Each maximum is over the three measured wavelengths. These are absolute
complex-amplitude errors, not relative or power errors. Both workers exited
0, reported stable fields and finite energy, and retained their source
identity throughout the run. The coarse-grid regression retains the
existing absolute ceilings of 0.005 for `r` and 0.010 for `t`; neither ceiling
was loosened after these two responses.

The calibration also retained its warnings. In both rows the lower absorber
gap is 0.775, below the proximity rule's half-wavelength distances of 0.868111
for Drude and 0.944059 for Lorentz. The resulting
`AbsorberProximityWarning` was not suppressed or reclassified as a pass. Small
decay ratios show settling of these recorded responses; they do not bound
absorber reflection. The `SubpixelAccuracyWarning` likewise retains the
bulk-only mesh qualification. The registered campaigns below include their
own declared time and absorber controls.

The controller manifest `fdtd_single_pole_film_calibration_attempt1.json`
retains the planned rows, pre-run frozen helper/controller copies, source and
artifact hashes, runtime identities and both worker exits. Each raw result
records its source identity and the successful before/after identity
comparison; the controller also records `source_unchanged=true`. The manifest,
the controller log and the raw results and logs (the
`fdtd_single_pole_film_calibration_attempt1` directory) are archived outside
the repository. No row was adapted within this attempt.
The FD source digest for these responses is
`67d4e3d04546626259e5d0d5e0c57f0a04560c51dedc6fafc469ffc5866f87ae`,
using sorted paths relative to `src/wavebit/fd`, NUL separators and LF-normalized
source bytes. These responses are evidence for that recorded source and remain
known exposure for any later registered campaign.

## Targeted tests and preserved attempts

The completed combined targeted run collected and passed all 148 cases,
exit 0, in 30.81 seconds as reported by pytest. It combines 94 new single-pole
cases, 53 existing conductive-film cases and one existing stencil refusal
case. The new coefficient and zero-strength tests cover all three normal
axes on NumPy; local energy covers x and z on NumPy, JAX and Torch, including
zero strength and `hold=0`. The oblique p-polarized Drude and Lorentz film
regressions run on all three backends against the independent complex Airy
reference. These finite-grid checks do not establish a convergence order.

The earlier attempts remain part of the evidence:

| Attempt | Result | Cause or correction |
| --- | --- | --- |
| 1 | 147 collected; execution exit 3 before any tests ran | The requested temporary directory's parent was missing during worker setup. |
| 2 | 45 passed, one failed; exit 2 | A new fixture supplied both `dx` and `wavelength` to `Resolution`, an invalid constructor combination. |
| 3 | 110 passed, one failed; exit 2 | The new stencil refusal fixture expected the wrong existing error message. |
| 4 | 148 collected and passed; exit 0 | Corrected fixtures and the existing stencil refusal case are included. |

The fixes changed the test setup and expectations, without widening a numerical
tolerance. Production changes after the two calibration rows were limited to
docstrings; these fixture repairs required no production numerical change.
The contemporaneous manifests, collection/run logs and retained test sources
(the study's `fdtd_single_pole_film_controls` directory) are archived outside
the repository. Attempt 1 and 2
test copies were reconstructed afterward by reversing the two fixture fixes;
their bytes match the hashes recorded during those attempts. They are not
described as snapshots taken before execution. That qualification is retained
in `targeted_test_snapshot_provenance.json` in the same archived directory.
Attempt 4 preserves the actual test and controller snapshots.

### Supplementary backend comparison

Six sequential rows repeated the two already disclosed calibration
configurations on NumPy, JAX and Torch: 40 cells, p incidence, face fraction
0.3, 60 periods and 24-cell PMLs, for Drude and Lorentz. All six workers
exited 0 and the source identity remained unchanged. At each of the three
wavelengths, complex `r` and `t` from JAX and Torch were compared with NumPy
using `|value-reference|/max(1,|reference|)`.

The largest scaled difference is 3.065954806021319e-15, within the existing
`128*float64_epsilon` budget of 2.842170943040401e-14, fixed before these
repeated responses. The manifest, individual differences and raw artifact
hashes are in the `manifest.json` of the `fdtd_single_pole_film_backends`
run, archived outside the repository.
This is supplementary agreement on known configurations; it adds no physical
acceptance rule and does not certify other material parameters, orientations
or derivatives. It does not resolve the convergence exceptions reported below.

## Actual production mutation witnesses

Each of the nine new test functions has an accepted assertion-failure witness
from an actual production mutation: removing the normal resonance shift,
perturbing lossless tangential permittivity, omitting oscillator kinetic
energy, reducing tangential oscillator drive to one quarter, removing the
initial-field refusal, removing prepared-plan identity, silently detaching
traced strength, omitting the material CFL coupling, and dropping the
interior spectral-bound critical point.

All nine mutated runs exited 1 on the intended assertion or missing refusal.
Every production file was restored byte for byte. The restored new-test suite
collected and passed all 94 cases, exit 0, with unchanged before/after source
identities. The canonical manifest is the run record
`fdtd_single_pole_film_mutations.json`; it, the original attempt and the raw
logs are archived outside the repository. Startup failures are not counted
as mutation witnesses.

## Registered physical campaign

The fixed 40-row campaign was registered at 2026-10-06T08:47:04 UTC with
fingerprint
`36813ffd5a6199a66a7fafd8ebed7fbf8b73241f2243a06f7f97acda6254c371`.
Its record, `fdtd_single_pole_film_registration.json`, is archived outside the
repository. The two disclosed materials each have face fractions 0.3 and 0.1,
both s and p incidence, and a 40/80/160 ladder. Each of these eight groups has the three
60-period base rows plus two 80-cell controls: 120 periods with 24-cell PMLs,
then the same duration with 48-cell PMLs. The declared time/absorber proxy and
interval classifications are fixed before these campaign responses.

All 48 material/placement/polarization/wavelength/complex-coefficient cells
are retained independently. The registration explicitly discloses the two
calibrations and the subsequent regression/mutation exposures; it does not
claim that those responses were unknown. Its FD digest is
`d444c895045c25c2e936411296fb66d717c8c830a1b2fcba381dbd792ff31135`
under the same NUL-separated normalized-source rule used above. The source
hash differs from calibration after the docstring updates.

### Initial 40-row result

All 40 workers exited 0; every selected mesh row uses its 60-period base run,
with no replacement or registered stop. The report is complete and its audit
recomputes all 48 cells successfully. These facts establish completion and
record consistency, not a blanket physical pass. The classifications are:

| Classification | Self-order cells, 40/80/160 | Direct-order pairs, both mesh pairs | Final direct-order cells, 80 to 160 |
| --- | --- | --- | --- |
| Second | 44 | 93 | 47 |
| Mixed | 2 | 1 | 0 |
| Low | 1 | 0 | 0 |
| Unresolved | 1 | 2 | 1 |
| Total | 48 | 96 | 48 |

For the noise proxy `u=|80,long24-80,base24|+|80,long48-80,long24|`, resolution
requires `2*u <= 0.1*min(adjacent changes)` for self-order, or one tenth of
the smaller endpoint error for a direct order. Changes or errors at or below
`1e-10` are unresolved. The self interval widens each change by `2*u`; each
direct-error interval uses `u`. Second requires the entire order interval
inside [1.8, 2.3], low requires its upper end at most 1.75, and other resolved
intervals are mixed. This empirical middle-grid proxy is not a rigorous
fine-grid error bound.

All four exceptional cells are `r` at face fraction 0.3. Every other one of
the 44 self-order cells is second. The intervals and noise shares below make
the different reasons visible:

| Material, incidence, wavelength | Self point | Self interval | Class | Proxy `u` | `2*u/min(changes)` |
| --- | --- | --- | --- | --- | --- |
| Drude, s, 1.0 | 1.778600 | [1.778005, 1.779194] | Mixed | 3.478103e-8 | 0.000319038 |
| Drude, s, 1.1 | 1.736571 | [1.735433, 1.737709] | Low | 5.938629e-8 | 0.000606812 |
| Lorentz, p, 1.0 | 1.801265 | [1.765002, 1.838040] | Mixed | 1.939314e-6 | 0.0196674 |
| Lorentz, p, 1.1 | 1.315645 | None: noise gate failed | Unresolved | 5.091672e-6 | 0.179538 |

The Drude exceptions are resolved under the registered noise test. The Lorentz
1.0 point estimate alone is just above 1.8, but its interval crosses that
boundary and therefore remains mixed. At Lorentz 1.1, the time contribution
is 4.794236e-6 and the PML contribution is 2.974358e-7; their sum exceeds the
self-order noise allowance. Its point value cannot be presented as a measured
low order.

The direct-reference orders for those same four cells are:

| Material, incidence, wavelength | 40 to 80: point [interval], class | 80 to 160: point [interval], class |
| --- | --- | --- |
| Drude, s, 1.0 | 1.816230 [1.816013, 1.816447], second | 1.915719 [1.914912, 1.916527], second |
| Drude, s, 1.1 | 1.782572 [1.782158, 1.782985], mixed | 1.901403 [1.899885, 1.902921], second |
| Lorentz, p, 1.0 | 1.828921 [1.815572, 1.842340], second | 1.920374 [1.870890, 1.970865], second |
| Lorentz, p, 1.1 | 1.413409, no interval, unresolved | 1.642245, no interval, unresolved |

For Lorentz 1.1 the direct noise shares are 0.122418 and 0.382129, both above
0.1. Outside those two unresolved pairs, only the Drude 1.1 coarse pair is
mixed; the remaining 93 pairs are second. The increasing Drude direct orders on these meshes
do not establish their asymptotic limit.

Across all 48 complex cells, the largest absolute error decreases from
0.00707519 at 40 cells to 0.00176016 at 80 and 0.000439706 at 160. That maximum
does not replace the cellwise verdicts above. All raw values, reference errors,
noise components and absorptance are retained in
`studies/fdtd/single_pole_film/records/fdtd_single_pole_film_report.json`; the successful procedural
audit is in `fdtd_single_pole_film_campaign_audit.log`, archived outside the
repository.

### Registered follow-up selected after the results

The separate refinement v1 was registered at 2026-10-06T09:07:15 UTC with
fingerprint
`93cc246c0b5ba735ebb8e6baade1fb0a2ee23185a6372189d53db6f7f7546045`.
Its record, `fdtd_single_pole_film_refinement_v1_registration.json`, is
archived outside the repository. The selection consists of exactly two
groups: face fraction 0.3 with Drude s incidence, and face fraction 0.3 with Lorentz p incidence. Both `r` and `t`
at all three wavelengths remain, giving 12 cells rather than only the four
original exceptions.

Each group has 80/160/320 base rows at 120 periods with 24-cell PMLs. Its
160-cell controls run for 240 periods, first with 24-cell PMLs and then 48.
These are ten fixed workers. The new middle-grid proxy uses the two
160-cell changes; its interval and classification rules retain the registered
form. Only an otherwise valid but unsettled base row may receive a single
240-period replacement. An invalid replacement or fixed control stops the
run; there is no duration beyond 240, additional rung or adaptation based on
a low, mixed or unresolved verdict.

Both 80-cell/120-period/PML24 configurations were already measured as controls
in the original study. They are repeated into new raw files for a physical
payload comparison that records provenance and reproducibility only. Their
agreement is neither independent corroboration nor an acceptance gate and
cannot trigger an adaptive rerun.

The groups were selected after the initial results were known, while this
follow-up's numerical protocol was registered before its new responses.
The original registration, criteria, 40-row report and four exceptions remain
unchanged.

### Refinement v1 result and its remaining limits

All ten workers exited 0, with stable fields, finite recorded energy and
unchanged source identity. Every selected mesh row uses its 120-period base;
there was no replacement or registered stop. The audit recomputes all 12
cells and confirms the fixed worker set. The classifications are:

| Classification | Self-order cells, 80/160/320 | Direct-order pairs, both mesh pairs | Final direct-order cells, 160 to 320 |
| --- | --- | --- | --- |
| Second | 11 | 20 | 9 |
| Low or mixed | 0 | 0 | 0 |
| Unresolved | 1 | 4 | 3 |
| Total | 12 | 24 | 12 |

All six Drude s cells are second in self-order and in both direct pairs.
The two formerly mixed/low Drude reflection cells are now second over
80/160/320; the formerly mixed Lorentz p reflection at wavelength 1.0 is
also second:

| Cell, all at face fraction 0.3 | Self point [interval] | Final direct point [interval], 160 to 320 |
| --- | --- | --- |
| Drude s, `r`, 1.0 | 1.900206 [1.900108, 1.900305] | 1.959480 [1.959339, 1.959621] |
| Drude s, `r`, 1.1 | 1.883170 [1.882954, 1.883386] | 1.952856 [1.952549, 1.953164] |
| Lorentz p, `r`, 1.0 | 1.899335 [1.853184, 1.946355] | 1.890972 [1.828581, 1.954954] |

This supports second-order behaviour for those observables on the finer,
longer measured ladder. It does not replace their original coarse-ladder
labels or establish an asymptotic limit. Mesh and runtime both changed,
so the comparison alone does not identify the cause of the original result.

The sole unresolved self-order remains Lorentz p `r` at wavelength 1.1.
Its point value is 1.737042, but no interval is admitted: the noise share
`2*u/min(changes)` is 0.302887, above the fixed 0.1 limit. Its 80-to-160 direct
point 1.728487 is also unresolved, with noise share 0.210258. These point
values are not low-order verdicts.

All three unresolved final direct-order cells are Lorentz p at fraction 0.3.
The reflection at 0.9 and transmission at 1.1 must remain visible even though
their self-order intervals are second:

| Coefficient, wavelength | Self-order status | Final direct point, no interval | Proxy `u` | 320-cell absolute error | `2*u/min(endpoint errors)` |
| --- | --- | --- | --- | --- | --- |
| `r`, 0.9 | Second: [1.882175, 2.093633] | 1.951823, unresolved | 3.948515e-6 | 4.708698e-5 | 0.167712 |
| `r`, 1.1 | Unresolved | 1.654273, unresolved | 2.567338e-6 | 7.758427e-6 | 0.661819 |
| `t`, 1.1 | Second: [1.925928, 2.067141] | 2.108154, unresolved | 1.776261e-6 | 2.742639e-5 | 0.129529 |

Each final direct noise share exceeds 0.1. The fine-grid error is smaller
than the adjacent grid change, so the same proxy can permit a self-order
interval while refusing a direct order. Counting only the 11 second
self-orders would hide this limit. All other 20 direct pairs are second;
the four unresolved pairs are exactly the three final pairs above and the
Lorentz `r`, 1.1 coarse pair.

The two repeated known 80-cell/120-period/PML24 rows match the original
controls exactly in every listed physical field and in canonical physical
JSON. This remains a provenance/reproducibility check, with no independent
corroboration or acceptance role. Complete values are retained in
`studies/fdtd/single_pole_film/records/fdtd_single_pole_film_refinement_v1_report.json`.
The audit, `fdtd_single_pole_film_refinement_v1_audit.json`, and the raw
worker records are archived outside the repository.

The study ends at its declared 320-cell/240-period cap with these unresolved
records. No further sweep is part of this validation. The empirical
middle-grid proxy still does not rigorously bound fine-grid noise.

Neither the original exceptions nor the remaining follow-up noise limits
permit a blanket second-order claim. General C8 smoothing is incomplete and
C7 remains unpassed; the earlier conductive-film measurements retain their
original conductivity/induced-Debye scope.

## Regression gates

The full tier collected 2,902 cases and completed with 2,894 passed, eight
skipped, zero xfailed and zero xpassed, exit 0. Pytest reports 2,332.12
seconds; the controller records 2,332.821182 seconds. The separate slow tier
collected and passed all 81 cases, with no skips, xfailed or xpassed, exit 0.
Pytest reports 1,189.84 seconds; the controller records 1,190.355486 seconds.

All four read-only freezers exited 0 and reproduced 143, 288, 140 and 570
arrays bit for bit: 1,141 arrays in total. No golden was rewritten or
rebaselined. The numerical validation controller completed with exit 0.

The controller retains counts and exits in the run record
`fdtd_single_pole_film_validation.json`, archived outside the repository. Its
unchanged FD source digest is
`e2eb97482f5b4d45c67db4464de193aca27a0f39b8d33ba5daa12bc719af60cf`.
This inherited validation helper hashes sorted relative paths and
LF-normalized source bytes without delimiters. The campaign digest above
uses NUL separators; the two algorithms identify the same source with
different digest strings. Passing regression checks does not resolve the
physical noise limits above.
