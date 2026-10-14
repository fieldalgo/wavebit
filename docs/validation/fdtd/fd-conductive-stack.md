# C8: separated parallel conductive films

Written: 2026-10-07

Targets: `src/wavebit/fdtd/_prepare/conductive_stack.py`,
`src/wavebit/fdtd/_prepare/conductive_film.py`, `src/wavebit/fdtd/_prepare/plan.py`,
`src/wavebit/fdtd/_prepare/params.py`, `src/wavebit/fdtd/_prepare/coefficients.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_kernel/poles.py`,
`src/wavebit/fdtd/_kernel/observe.py`, `src/wavebit/fdtd/_prepare/report.py`.

## Verified bounded scope

This extension composes the existing conductive-film response for a finite
number of separated parallel films. Six disclosed calibration rows, 136 new
test cases, a 510-case targeted suite and ten actual source mutations are
complete. The registered 40-worker campaign and its saved-data audit also
completed: 42 of 48 self intervals and 45 of 48 final direct intervals meet
the second-order criterion, with six exceptional reflection cells detailed
below. The full tier passed 3,933 cases with eight existing skips, the
separate slow tier passed all 81 cases, and all 1,141 frozen arrays reproduce
bit for bit. There is no blanket second-order or general-material accuracy
conclusion.

Every structure must be a concrete exact `Conductor` on an unturned, unranged
voxel `Box`. Every member must independently satisfy the singleton-film scope:
positive isotropic permittivity, nonnegative isotropic conductivity, unit
permeability, two transverse axes spanning the grid with the required overhang,
finite normal thickness at least one cell, and support clear of domain walls
and absorbers. The host is one plain positive isotropic lossless nonmagnetic
`Material`; public initial fields and traced geometry, materials or bands are
excluded. Zero conductivity retains its declared member and reserved memory.

All films share one normal axis. Their actual padded half-open grid supports
must be pairwise disjoint: after sorting supports for this check, each upper
index must be at most the next lower index. A physical gap alone does not
establish this condition. Overlap diagnostics name an adjacent original member
pair and its support intervals. Direct contact, intersecting supports, mixed
material families, lossy hosts and arbitrary curved bodies remain outside this
scope. It does not complete C8 or resolve the earlier C7 findings.

## Local composition and prepared identity

The singleton's analytic Yee fills, induced Debye recurrence and split energy
weights are reused with the same arithmetic. Each support receives a direct
paste of its complete singleton permittivity and conductivity profiles into a
host profile. This avoids subtracting and adding a potentially much larger
host permittivity. Each film has its own induced interface memory; global
field energy is counted once and local polarization storage is added once per
member. Unturned boxes are cross-free, so there is no additional tensor
interpolation halo beyond the declared shape support.

The ordered whole stack is part of every memory's prepared signature. Changing
any member, order, host or geometry requires a new plan. Direct lowering also
rejects missing, extra, renamed, foreign or mismatched memories. Sorting for
the separation check does not change the declaration order.

The inherited conservative local Debye CFL policy takes the maximum local
rate over all members, since their supports are disjoint. It does not sum
rates from spatially separate regions. The curl bound retains the smallest
physical high-frequency permittivity.

For a bulk conductor, `k^2 = eps*omega^2 + i*sigma*omega`, and
`|k|^4 = eps^2*omega^4 + sigma^2*omega^2` increases for positive frequency.
The whole source-band upper endpoint is checked for every member and the
host against `dx*|k| <= 2*pi/8`. Endpoint arithmetic is exact rational
arithmetic of the public floating-point band constants, with verified upward
square roots. This does not certify exact transcendental constants. Every
requested monitor, Resolution and explicit PML wavelength also receives
member-specific phase and skin-depth checks. These sufficient bulk mesh floors
are not observable-error or convergence bounds.

## Registered physical instrument

Two fixed vacuum-host cases use top-to-bottom `(eps, sigma, thickness)`:

| Case | Films | Vacuum gaps |
| --- | --- | --- |
| lossy-pair | `(4,8,.10)`, `(2.25,4,.15)` | `.15` |
| mixed-triple | `(3,0,.10)`, `(2,6,.10)`, `(5,10,.15)` | `.125`, `.175` |

The independent characteristic matrix uses declared `eps+i*sigma/omega` and
explicit host gaps. Complex reflection and transmission are referenced to the
outer physical faces. The central angle is 35 degrees with fixed transverse
wave number; sampled wavelengths are `.9`, `1`, `1.1`. Analytic controls cover
host phase, zero-depth deletion, singleton Airy agreement, lossless power,
passivity, polarization signs and reversal reciprocity before field responses.

The registered protocol, the run record `fdtd_conductive_stack_registration.json`
(archived outside the repository),
was sealed on 2026-10-07 at 00:51:36 UTC, after calibration and the targeted
and mutation tests. Its fingerprint is
`3c7789eba8bbc92316b1392f0bc43d8e18836ef1373564c6c62dcd6f09f66a97`.
It binds source and instrument hashes, frozen source snapshots and all
disclosed development responses before the first campaign worker.

There are 40 mandatory sequential workers: two cases, placements `.3` and
`.1`, polarizations `s` and `p`, and the following five settings per group.
Here `n` is cells per central vacuum wavelength and `T` is run duration in
central periods. The source band has central wavelength `1`, `df=.1` and
cutoff `1e-6`; all campaign workers use NumPy.

| Setting, in execution order | n | T | PML cells |
| --- | ---: | ---: | ---: |
| Coarse base | 40 | 120 | 24 |
| Middle base | 80 | 120 | 24 |
| Middle longer duration | 80 | 240 | 24 |
| Middle longer duration and thicker absorber | 80 | 240 | 48 |
| Fine base | 160 | 120 | 24 |

An otherwise valid but unsettled base row may receive one `T240` replacement;
the existing middle-grid `T240/PML24` control is reused if needed. A hard
failure, invalid replacement or invalid fixed control stops the campaign.
The settlement threshold is a finite reported decay no greater than `1e-5`.
No longer run, extra spatial rung, changed geometry or selected subset of
wavelengths is authorized by this registration.

The two `n40/p/placement .3` NumPy rows repeat known calibration
configurations. Their payload comparisons are provenance checks, not
independent corroboration. Warnings remain in the raw rows; geometry is not
moved in response to a warning.

## Disclosed calibration results

Six sequential rows cover both cases on NumPy, JAX and Torch, with
`n40`, `p`, placement `.3`, `T120` and `PML24`. They completed with finite
energy and stable diagnostics. The production source map was unchanged, and
the independent analytic preflight passed before the first field response.

| Case | Maximum absolute complex r error | Maximum absolute complex t error |
| --- | ---: | ---: |
| lossy-pair | 0.0015464762804496444 | 0.004865877549259704 |
| mixed-triple | 0.005551693331482326 | 0.010841844816499236 |

These are absolute errors in complex amplitudes, not relative or percentage
errors. Maximum reported decay was `1.329755298531103e-8`. The greatest
backend difference, scaled by `max(1, abs(NumPy value))`, was
`2.036633330088357e-15`, below the fixed `128*float64 epsilon` budget
(`2.842170943040401e-14`). The
[calibration record](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/records/fdtd_conductive_stack_calibration_attempt1.json)
links all six raw responses and logs, their hashes, the pre-response
production archive and analytic controls. Its outer receipt, the calibration
launch's `exit.json` (archived outside the repository), records the actual
controller exit.

After disclosure, shared optical regression ceilings of `.007` for `r` and
`.014` for `t` were selected as `1.2` times each measured maximum, rounded
upward to the next `.001`. The resulting margins above the observed maxima
are about 26.1% and 29.1%. This
[acceptance record](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/records/fdtd_conductive_stack_calibration_acceptance.json)
preserves the test before and after that change. The limits apply only to
these six known coarse-grid configurations; they are neither universal
accuracy bounds nor a campaign absolute-error gate. The campaign's order,
noise and decay criteria were not fitted to these amplitudes.

## Completed local verification

The new test module passed all 136 collected cases across nine test
functions. The targeted suite passed 510 cases, including those 136, with
no failures or skips. The cases exercise actual recurrence transfer response,
all-member profile placement, singleton preservation, reported versus
allocated state memory, local storage and the closed-Yee loss identity,
optical regressions on three backends, scope refusals, prepared identity,
tracing refusals and every-member CFL and spectral guards. These results
are recorded in the
[targeted manifest](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/records/fdtd_conductive_stack_targeted.json),
with separate raw logs for the new tests (`new_stack.log`) and the targeted
suite (`targeted.log`); both raw logs are archived outside the repository.

Ten deliberate changes to actual production files each caused the intended
assertion failure, covering all nine functions. They altered later-member
permittivity, storage and conductivity; omitted memory allocation; admitted
overlapping supports; bypassed whole-stack identity or foreign-memory
rejection; detached a Torch forward carrier; or checked only the first
member's CFL rate or source band. Every source was restored to its original
bytes, and the restored new-test suite passed all 136 cases again. The
mutation manifest, `fdtd_conductive_stack_mutations.json` (archived outside
the repository), links each exact original and mutant snapshot, one-case
failure log and restoration hash; the
[restored baseline log](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/records/fdtd_conductive_stack_mutations_attempt1/restored_targeted_baseline.log)
stays in the repository. The targeted and mutation exit receipts (each
launch's `exit.json`, archived with the raw records) both record exit zero.
Mutation detection checks the stated failure
mechanisms; it does not prove the absence of other defects.

## Cellwise convergence and uncertainty rules

Each case, placement, polarization, wavelength and complex coefficient
(`r` or `t`) is retained independently. This gives
`2*2*2*3*2 = 48` cells, 48 self-convergence intervals, 96 direct-error
intervals (40-to-80 and 80-to-160), and 48 final-pair direct intervals.
No worst-cell aggregate replaces these records.

For a fixed cell, let `v40`, `v80`, `v160` denote the selected complex
responses and `vref` the continuum multilayer value. Define the positive
magnitudes

```text
d0 = abs(v40 - v80)      d1 = abs(v80 - v160)
en = abs(vn - vref)
u_time = abs(v80,T240,PML24 - v80,T120,PML24)
u_pml  = abs(v80,T240,PML48 - v80,T240,PML24)
u = u_time + u_pml
```

The proxy always retains the original `T120` middle-grid response, even if
the selected `v80` comes from a valid `T240` replacement. For either adjacent
pair of errors or changes `(a,b)`, the interval is resolved only when both
conditions hold:

```text
min(a,b) > 1e-10
2*u <= .1*min(a,b)
```

The self-convergence point estimate is `log2(d0/d1)`, with sensitivity
interval

```text
[log2((d0-2*u)/(d1+2*u)), log2((d0+2*u)/(d1-2*u))].
```

Each adjacent direct-error estimate is `log2(e_coarse/e_fine)`, with interval

```text
[log2((e_coarse-u)/(e_fine+u)), log2((e_coarse+u)/(e_fine-u))].
```

The resolution conditions keep all interval arguments positive. A finite
point estimate alone is insufficient for a resolved verdict.

| Registered label | Rule |
| --- | --- |
| second | Entire interval lies within `[1.8,2.3]`. |
| low | Resolved interval has upper endpoint at most `1.75`. |
| mixed | Any other resolved interval. |
| unresolved | Noise or vanishing-signal gate fails, or a required row is missing or invalid. |

The middle-grid time/PML proxy is empirical. It is not a rigorous bound on
the error of another spatial rung, and its interval is not a statistical
confidence interval. An unresolved label does not establish a spatial order;
a mixed or low label does not by itself identify the physical or numerical
cause. Measured `1-R-T` and continuum absorptance remain available per
wavelength; physical absorption is not treated as energy-closure error.

## Registered campaign outcome

All 40 mandatory workers completed with no replacements or early stop.
Every base row met the settlement rule, so all selected durations remain
`T120`. Every raw row reports finite energy and stable fields. Maximum
reported decay was `1.6715715847730152e-08`, at
`mixed-triple / placement .1 / p / n80 / T240 / PML24` in the reflection
monitor, below the registered `1e-5` limit. All 40 rows retain
`SubpixelAccuracyWarning`; four also retain `CoarseMeshWarning`. No growing
or unstable-field warning was recorded.

Both the campaign process and its separate saved-data audit exited zero (each
launch's `exit.json` receipt is archived outside the repository). The audit
reproduces all per-cell records from the raw responses. The ledger,
`fdtd_conductive_stack_campaign.jsonl` (archived outside the repository with
the raw responses), links each of the 40 response JSON files and raw logs by
hash; the
[complete report](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/records/fdtd_conductive_stack_report.json)
retains all 48 cells. Both known NumPy calibration payloads repeat exactly;
that agreement is a provenance check, not an additional independent result.

The results do not meet the second-order criterion in every cell:

| Measurement | Intervals | second | low | mixed | unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Self, 40/80/160 | 48 | 42 | 4 | 2 | 0 |
| Direct, 40-to-80 | 48 | 42 | 5 | 1 | 0 |
| Direct, 80-to-160 | 48 | 45 | 0 | 2 | 1 |
| All direct pairs | 96 | 87 | 5 | 3 | 1 |

All exceptions occur in reflection from `lossy-pair` at placement `.3`.
The following table includes every early direct exception as well as the
self and final-pair outcomes. Intervals are rounded for display; labels
come from the full-precision registered rules.

| Polarization | Wavelength | Self interval | Direct 40-to-80 interval | Direct 80-to-160 interval |
| --- | ---: | --- | --- | --- |
| s | 0.9 | low [1.586867, 1.586967] | low [1.666163, 1.666199] | second [1.855373, 1.855498] |
| s | 1.0 | low [1.522563, 1.522716] | low [1.610345, 1.610399] | second [1.828735, 1.828921] |
| s | 1.1 | low [1.668261, 1.668772] | low [1.662362, 1.662546] | mixed [1.796293, 1.796916] |
| p | 0.9 | low [1.411215, 1.449272] | low [1.539084, 1.552412] | mixed [1.789333, 1.834114] |
| p | 1.0 | mixed [1.728791, 1.760437] | low [1.720622, 1.732065] | second [1.800794, 1.840621] |
| p | 1.1 | mixed [2.259428, 2.459224] | mixed [2.260149, 2.338504] | unresolved (point 2.128533 only) |

All 24 `mixed-triple` cells, all transmission cells and the `lossy-pair`
reflection cells at placement `.1` meet the second-order interval criterion
on both direct pairs and in self-convergence. These are findings for the
registered cells, not a guarantee for arbitrary stacks or placements.

Every one of the 48 absolute complex-amplitude errors decreases from 40 to
80 to 160 cells. Their maxima over cases, placements, polarizations and
wavelengths are:

| Cells per central wavelength | Maximum absolute r error | Maximum absolute t error |
| --- | ---: | ---: |
| 40 | 0.0056355259405069325 | 0.011467960114785165 |
| 80 | 0.0014133743183063106 | 0.0028551806560730641 |
| 160 | 0.00035456043928385264 | 0.00071412093991657128 |

At every rung, the reflection maximum belongs to `mixed-triple / placement
.1 / p / wavelength 1.0`, and the transmission maximum to the same case and
placement at wavelength `.9`. These maxima are absolute complex-amplitude
errors, not percentages. The largest-error cells are not the cells with
exceptional convergence labels; the cellwise record is therefore necessary.

For the six exceptional reflection cells, the measured duration and absorber
contributions are:

| Polarization | Wavelength | Duration proxy | PML proxy | Final resolution share `2*u/min(e80,e160)` |
| --- | ---: | ---: | ---: | ---: |
| s | 0.9 | 1.61371037e-09 | 4.16672901e-09 | 6.80185623e-05 |
| s | 1.0 | 1.10461152e-09 | 3.28969759e-09 | 0.000100604331 |
| s | 1.1 | 2.58629201e-09 | 3.90080999e-09 | 0.000335115104 |
| p | 0.9 | 1.78718696e-06 | 3.5382486e-08 | 0.0241567821 |
| p | 1.0 | 7.10276311e-07 | 3.83984941e-08 | 0.0215149262 |
| p | 1.1 | 3.21494924e-06 | 1.20804978e-07 | 0.197350725 |

The final `p / wavelength 1.1` interval is unresolved because its share is
`0.1973507248100786`, exceeding the fixed `0.1` gate. Its total proxy is
`3.3357542222225952e-6`, of which about 96.38% comes from the `T120` to
`T240` change. Its point estimate `2.1285334949630847` is descriptive only;
there is no accepted interval. A small reported decay alone did not make
this difference negligible relative to the remaining spatial error.

The `s / wavelength 1.1` final interval lies entirely below `1.8`, giving
the resolved `mixed` label. The `p / wavelength .9` final interval crosses
`1.8` and is also `mixed`. The three `s` self intervals remain `low` with
very small measured time/PML contributions. Several final direct intervals
improve relative to their coarse pairs, but these three spatial rungs do
not establish an asymptotic order or explain the exceptional cells.
The empirical middle-grid proxy cannot prove that duration or absorbers
account for all uncertainty, nor identify the cause of the low or mixed
orders. There is no blanket second-order conclusion.

## Regression evidence and remaining scope

The completed validation manifest, `fdtd_conductive_stack_validation.json`
(archived outside the repository), records every collected and reported test
node, source guard, child command, raw log and actual process exit. Its outer
exit receipt, the validation launch's `exit.json` (archived with it), also
records exit zero.

| Gate | Completed result |
| --- | --- |
| Full tier | 3,941 collected: 3,933 passed, eight existing skips, no failures. |
| Separate slow tier | 81 collected and passed, no skips or failures. |
| Stage 0 frozen reference | 143 of 143 arrays bit for bit. |
| Stage 1 frozen reference | 288 of 288 arrays bit for bit. |
| Stage 2 frozen reference | 140 of 140 arrays bit for bit. |
| Stage 3 frozen reference | 570 of 570 arrays bit for bit. |

The raw full and slow logs (`full.log` and `slow.log` of the validation
attempt, archived outside the repository) retain individual outcomes. Every
freezer reports that the committed golden
was reproduced and nothing was written. The guards retain the exact four
golden files, all 1,738 prior evidence files and all 86 pre-existing FD test
files. The extension adds one FD test module; the final production and FD-test
inventories contain 104 and 87 Python files respectively.

A separate independent campaign replay (its record, `attempt1.json` of the
review campaign, is archived outside the repository)
checks 2,120 recorded input hashes, the exact 40-worker schedule, all per-cell
intervals and both known calibration repeats. Its independently recursive
Fresnel reference differs from the recorded characteristic-matrix answer by
at most `3.608224830031759e-16`. This checks the stored reference independently;
it does not turn the known calibration repeats into independent field runs.

The [documentation runner](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/fdtd_conductive_stack_documentation.py)
checks the final document sources with the documentation contract and a
warning-as-error Sphinx build. The [final evidence auditor](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/conductive_stack/fdtd_conductive_stack_audit.py)
verifies exact source and raw-log hashes, actual process exits, collected and
reported test nodes, the unchanged skip identities, mutation restoration,
preserved prior evidence and scalar continuum/per-cell arithmetic. These
final checks retain their own source identities and execution records in the
study's controls directory, `fdtd_conductive_stack_controls`, which is archived
outside the repository. The saved-evidence auditor reruns neither fields nor
tests; its caller records its exit separately. Passing those checks certifies the stated evidence and
regressions, not a universal spatial order or general C8 completion.

The first final-audit process (its `exit.json` receipt is archived outside the
repository) exited one: it compared a Windows-spelled mutation log path
directly with the POSIX spelling of the same file. The file and its SHA-256 were identical.
A saved-data review found the same path-spelling assumption in the interpreter
and controller-report comparisons. The archived replay source,
`audit_variant.py.raw`, resolves those three same-file comparisons while
retaining every raw hash and non-path field check. It also binds its own
executed source into the input map. The exact patch and observed path pairs,
`repair_proposal.json`, retain the original source and failed attempt. Both
files of the audit path repair are archived outside the repository. This is a
disclosed repair to the saved-evidence checker; the original ten helpers, numerical sources,
tests, calibration, registered physical criteria and numerical results
remain byte-identical. The original auditor is not claimed to have passed.

The physical instrument samples only these two vacuum-host geometries at one
central angle and three wavelengths. Neither those samples nor the algebraic
composition tests establish accuracy for every admitted stack. Mixed
dispersive families, touching interfaces, arbitrary bodies and traced stack
parameters remain outside the feature; the outstanding C7 findings and
general C8 completion are unchanged.
