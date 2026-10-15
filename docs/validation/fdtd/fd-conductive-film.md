# C8: one conductive film in a dielectric

Written: 2026-10-06

Targets: `src/wavebit/fdtd/_prepare/conductive_film.py`, `src/wavebit/fdtd/_prepare/checks.py`,
`src/wavebit/fdtd/_prepare/params.py`, `src/wavebit/fdtd/_kernel/observe.py`,
`tests/fdtd/test_fd_conductive_film.py`, `studies/fdtd/conductive_film/fdtd_conductive_film_probe.py`,
`studies/fdtd/conductive_film/fdtd_conductive_film_campaign.py`,
`studies/fdtd/conductive_film/fdtd_conductive_film_final_confirmation.py`.

## TL;DR

The first C8 implementation admits a flat conductive film whose two faces can
cut through Yee cells. It realizes the complex normal and tangential material
responses with one induced Debye memory at each mixed normal-electric cell.
Simply averaging conductivity in every direction does not represent those
interface conditions.

**Validation status: the corrected source passes the full and slow regression
tiers and reproduces all four frozen references exactly.** The bounded film campaign and
its final-source confirmation place all 24 complex response cells within the
registered second-order intervals. All nine mutation witnesses and the targeted
baseline also passed on the corrected source. This is not completion of C8. The earlier
[C7 metasurface result](fd-metasurface-validation.md) remains unpassed; adding
conductive-film support does not resolve its dielectric box's slow convergence.
Curved conductors and general dispersive-interface smoothing remain later C8 work.

## Exact public scope

Use a single `fd.Structure(fd.Box(...), fd.Conductor(eps=..., sigma=...))`
with the default `discretisation='voxel'`. The following conditions all apply:

| Item | Admitted case |
| --- | --- |
| Geometry | One public `Box`, `angle=0`, `range=None`, with concrete sizes and centre. |
| Film extent | Uniform along exactly two complete grid axes; on each of them the box extends at least one `dx` past both domain faces. The remaining axis is the film normal. |
| Thickness | At least one `dx`, so its two interfaces do not cut the interior of one electric dual cell. Existing resolution and feature-size checks still apply. |
| Clearance | The film's normal support is strictly inside the domain and its full support stays outside every absorbing layer. A transverse absorber therefore does not fit this scope. |
| Film material | Concrete isotropic `Conductor`, finite real `eps > 0`, finite `sigma >= 0`, and `mu=1` (explicit or default). |
| Host | Plain, concrete, isotropic lossless `Material` with finite `eps > 0` and `mu=1`. |
| Other scene content | No second structure, junction partner or initial field. Sources and monitors still obey their existing placement and material restrictions. |
| Reuse | A prepared film plan belongs to its concrete geometry and material values. Replan after changing them; traced values and differentiation are refused. |

For a centred domain with transverse widths `Lx` and `Ly`, a z-normal box can
use transverse sizes `Lx + 2*dx` and `Ly + 2*dx`, or a larger overhang. Merely
ending at the domain faces is insufficient: node-centred Yee dual cells extend
past a face. The film's normal centre and thickness still need enough room for
their support, sources, monitors and absorbers. The same rule applies after
permuting x, y and z.

`Slab` and `Region` retain their existing cell-snapped material rule. The new
film is not a fallback for a finite conductive box, a rotated box, a sphere,
an anisotropic or magnetic conductor, a conductive host, or a `Dispersive`
body. Those unsupported smoothed cases remain refused. A prepared non-film
plan cannot be supplied to bypass these restrictions. Later scopes admit
isolated flat `Dispersive` films (from 2026-10-06: [Drude/Lorentz](fd-single-pole-film.md),
[Debye](fd-debye-film.md) and their successors in the [index](index.md)) and separated
parallel conductive films (2026-10-07, [record](fd-conductive-stack.md)).

For this admitted film, mesh checks include conductivity when calculating the
phase index: `n_phase = Re(sqrt(eps + i*sigma/omega))`. The existing floor of
eight cells per wavelength uses this index at each Diffraction wavelength
and at the source band's shortest wavelength; a known wavelength in a scene
without a source is checked by the resolution notes. The existing warning
below twelve cells remains. Using only `sqrt(eps)` would miss the shorter
phase wavelength at high conductivity. This change is restricted to the
conductive-film scope; it does not change the historical snapped-material
checks or establish a new film-accuracy guarantee. With no source, monitor
wavelength or other wavelength metadata, the planner can only warn that the
mesh could not be judged.

## Complex interface response

Let `f` be the conductor fraction of the dual cell around an electric Yee
point, `g = 1-f`, `a` the conductor's real permittivity, `b` the host
permittivity, and `sigma` the conductivity. With the repository's time
convention `exp(-i*omega*t)`, define `s = -i*omega`; the conductor response is
`a + sigma/s = a + i*sigma/omega`. Conductivity has inverse-length units when
the scene uses `c = eps0 = mu0 = 1`.

The normal harmonic and tangential arithmetic response follow the interface
tensor in Eqs. (1) and (2) of
[Deinega and Valuev, Optics Letters 32, 3429–3431 (2007)](https://doi.org/10.1364/OL.32.003429).
For this axis-aligned film the tensor is diagonal. The reduction below is
algebra for this restricted geometry; it does not implement the paper's
general curved-interface tensor scheme.

```text
eps_t(s) = f*a + g*b + f*sigma/s

eps_n(s) = 1 / (f/(a + sigma/s) + g/b)
         = b*(a*s + sigma) / (A*s + g*sigma),  A = f*b + g*a.
```

For a mixed cell, `0 < f < 1`, with positive conductivity, polynomial division
gives a passive Debye response:

```text
eps_n(s) = eps_inf + delta/(1 + tau*s)
eps_inf  = a*b/A
delta    = f*b*b/(g*A)
tau      = A/(g*sigma).
```

Both `delta` and `tau` are positive. There is no residual normal DC
conductivity in a mixed cell: the dielectric fraction interrupts the series
current path. The low-frequency normal permittivity is `b/g`, whereas a
wholly conducting cell retains its conductivity. Even when `a == b`, the
normal mixed cell still needs a relaxation pole; equal real permittivities
do not remove the conductivity contrast.

The tangential components use `eps = f*a + g*b` and `sigma_t = f*sigma`.
Exact empty and full cells use the host and conductor respectively, without
an interface pole. At `sigma=0`, the normal response is the existing lossless
harmonic mixture and the memory drive is zero. Fractions come from analytic
one-dimensional interval overlaps at each electric component's own Yee
position. Endpoint handling uses geometric containment predicates, without
snapping a narrow band of small nonzero fractions to zero or one.

## Time stepping and finite endpoint formulas

Trapezoidal discretization of `tau*dP/dt + P = delta*E` gives
`P_new = gain*P_old + c*(E_new + E_old)`. The existing Debye implementation
uses the shifted memory `Q = P - c*E`, so the ordinary electric update can
use `eps_update = eps_inf + c`, followed by an explicit pole recurrence.
For `D = 2*A + dt*g*sigma`, the coefficients are

```text
gain  = (2*A - dt*g*sigma)/D
c     = f*b*b*sigma*dt/(A*D)
drive = 4*f*b*b*sigma*dt/(D*D)

Q_new = gain*Q_old + drive*E_old.
```

These formulas avoid constructing `delta` and `tau`, which separately diverge
as `f` approaches one. They are evaluated only on active mixed normal cells.
All inactive components and endpoint cells have zero interface drive and
energy weights; full normal conductor cells use the ordinary conductivity
update. Derived non-finite or non-passive coefficients are refused before
stepping.

`src/wavebit/fdtd/_prepare/conductive_film.py` builds compact profiles along the film
normal and one `InterfacePoleTerm`. It reuses the existing
`src/wavebit/fdtd/_kernel/poles.py` recurrence and its polarization-current addition
to Ampere's law. There is no new off-diagonal constitutive operator in this
scope. Its concrete signature includes geometry, material values and grid
identity; lowering checks that the prepared plan retains the required memory.

The coefficient-level frequency oracle evaluates the complex mixture at
`omega_hat = 2*tan(omega*dt/2)/dt`, the trapezoidal frequency. Agreement there
checks the discrete recurrence's material law. A physical convergence study
must instead compare against the continuum film at the actual `omega`,
including the solver's spatial and temporal discretization errors.

## Energy and the planning bound

The induced pole's strength varies by cell. In the kernel's energy
normalization its contribution is a sum of local terms

```text
inverse_delta * |Q + c*E|^2 - c*|E|^2,
inverse_delta = g*A/(f*b*b).
```

The field update already includes `c` in its permittivity; the subtraction
accounts for that shift. The observer applies these spatial weights before
reducing, respecting both clipped regions and component selections. Inactive
weights are zero. Reconstructing the strength from `1 - gain*gain` would lose
accuracy near a full cell and encounter zero denominators at inactive cells.
The supplied finite weights avoid that reconstruction. This bookkeeping
identity alone is not a propagation or long-time stability experiment.

The existing conservative Debye planning policy also includes the largest
active-cell rate

```text
S = max(delta/(tau*tau*eps_inf))
  = max(f*g*b*sigma*sigma/(a*A*A))
  <= sigma*sigma/(4*a*a).

dt_bound = 1 / sqrt(1/dt_nondispersive^2 + S/4).
```

The maximum is taken over the actual normal mixed cells, not a fitted
constant. The upper bound follows from `A*A >= 4*f*g*a*b`. This preserves
the planner's existing policy for a declared Debye pole; it is not a claim
that passive trapezoidal relaxation intrinsically requires this time step.

## Registered validation scope

`tests/fdtd/test_fd_conductive_film.py` defines checks for the complex series/parallel
response, the zero-conductivity limit, local energy weights, unsupported-scene
refusals, prepared-plan identity, traced-value refusal and the derived planning
bound. The coefficient tests cover x-, y- and z-normal films on NumPy; the
energy test covers x and z on the available NumPy, JAX and Torch backends. An
oblique p-polarized film regression exercises the actual update and complex
Airy reflection/transmission on those backends. This finite-grid regression
does not itself establish a convergence order. Equal real permittivities and
zero conductivity are material-law controls, not additional physical arms of
the bounded campaign described next.

The refusal matrix also covers anisotropic conductivity, a sub-cell-thick
film, insufficient transverse overhang and a transverse absorber. Prepared-plan
tests remove or duplicate the interface memory and try supported and unsupported
conductors against a non-film plan. Traced conductivity is refused on Torch's
reverse and forward paths and under JAX tracing.

`studies/fdtd/conductive_film/fdtd_conductive_film_campaign.py` defines a bounded 20-worker campaign,
using the continuum Airy oracle in `studies/fdtd/conductive_film/fdtd_conductive_film_probe.py`.
The film has `eps=4`, `sigma=8` and thickness 0.2 in vacuum, with 35-degree
incidence at centre wavelength 1.0 and a fixed transverse wavevector. The x
axis uses `Axis1D`, whose derivative is exactly `i*kx`. The two placements
put a face at fractional node offsets 0.3 and 0.1. Each placement has both
s and p incidence and a 40/80/160 mesh ladder; both physical faces retain
their respective fractional offsets along each ladder. The three measured
wavelengths are 0.9, 1.0 and 1.1, with their own incidence angles. The source
band has `df=0.1` and cutoff `1e-6`.

The base runs last 60 periods with 24-cell absorbers. Each placement and
polarization also has two controls at 80 cells: 120 periods with 24 absorber
cells, then the same duration with 48. These five rows per group give the
20 planned workers, all on NumPy. Only an otherwise valid but unsettled base
row may receive one 120-period replacement. An invalid replacement or fixed
control stops the campaign; there is no extra rung or duration beyond 120.
The registered record discloses the earlier narrow-band and broad-band
calibration exposures and their provenance limitations.

The campaign was registered at 2026-10-06T05:13:04 UTC with fingerprint
`27d10f1eea7acf4696358d039eef0c502a391577ae0903e565baa44c0fc62ddc`.
Its source copies, known exposures and runtime identities are retained in the
registration record `fdtd_conductive_film_registration.json`, archived outside the
repository.

Every placement, polarization, wavelength and complex `r`/`t` coefficient
is retained: 24 cells in total. The empirical noise proxy is the sum of
the two 80-cell changes, `|long24-base24| + |long48-long24|`. Twice that
proxy must be at most one tenth of the smaller adjacent mesh change for a
self-order, or of the smaller endpoint error for a direct order. Changes
or errors at or below `1e-10` remain unresolved. Self-order intervals widen
each change by twice the proxy; direct-order intervals widen each error by
the proxy. Second means the whole interval lies inside [1.8, 2.3], low
means its upper end is at most 1.75, and other resolved intervals are mixed.
The middle-grid proxy is not a rigorous fine-grid error bound.

The campaign preserves power and absorptance alongside the complex values.
The film is lossy, so `R + T = 1` is not its conservation target; measured
`1-R-T` is compared with the continuum absorptance. The two-placement
campaign contains no separate equal-permittivity, zero-conductivity or
cell-snapped physical ladder. Those would be additional experiments, not
unreported arms of this protocol.

## Measured film result

The original campaign and the separate final-source confirmation each
completed all 20 workers with exit 0. In each, every selected row uses the base
60-period duration; no replacement or registered stop was needed. All 24
self-order intervals and all 48 direct-order intervals (two mesh pairs per
cell) classify as second. The self-order point estimates span
1.931852–2.038735; the final 80-to-160 direct orders span 1.963506–2.016076.
The complete final-source per-wavelength complex records are in
`studies/fdtd/conductive_film/records/fdtd_conductive_film_final_source_v1_report.json`; the original
`studies/fdtd/conductive_film/records/fdtd_conductive_film_report.json` is preserved separately.

| Face fraction | Incidence | Largest absolute complex error, 40 cells | 80 cells | 160 cells | Second self-orders |
| --- | --- | --- | --- | --- | --- |
| 0.3 | s | 0.00950350 | 0.00236188 | 0.000589592 | 6 / 6 |
| 0.3 | p | 0.00918852 | 0.00229349 | 0.000573507 | 6 / 6 |
| 0.1 | s | 0.01046196 | 0.00260126 | 0.000649418 | 6 / 6 |
| 0.1 | p | 0.00995963 | 0.00248416 | 0.000620901 | 6 / 6 |

Each row's maximum includes both `r` and `t` at all three wavelengths. These
are absolute complex-amplitude differences, not relative errors or power
errors. Across the whole campaign, the largest errors decrease from
0.01046196 to 0.00260126 to 0.000649418. The largest time-plus-absorber proxy is
1.810162e-6. The largest twice-proxy share of the smaller adjacent mesh change
is 0.0086202, below the registered 0.1 limit; no cell is hidden as unresolved.
These results support second-order behaviour for the two tested placements
of this film at these meshes, wavelengths and angle. They do not certify all
admitted materials, orientations, sources or asymptotically finer grids.

The final-source audit recomputed all 24 cell records and compared all 20
workers with their original counterparts. All listed physical fields agree
exactly as decoded values and as canonical JSON, including complex amplitudes,
powers, absorptance, decay, energy diagnostics, time step, grid and source
configuration. The complete cell records are also identical. Result-file hashes
differ because source identities, timestamps and wall times are separate
metadata. This comparison is recorded in the audit record
`fdtd_conductive_film_final_source_v1_audit.json`, archived outside the repository; it
establishes unchanged measured film behaviour across the refusal-routing correction.

**Report recovery, without new calculations.** After the twentieth successful
worker the controller exited 1 while serializing a NumPy Boolean in the
order record. It left a 1988-byte partial JSON report. The registered script
and probe remain unchanged. A separate utility,
`studies/fdtd/conductive_film/fdtd_conductive_film_report_recovery.py`, preserved the partial file,
reproduced the identical serialization failure and partial bytes, and used
the registered probe's scalar conversion to write the same numeric report
with native JSON types. No criterion, worker result or production source was
changed; no worker was rerun. The original controller output exists in the
tool transcript rather than a saved controller log; the saved failure log
is explicitly a reproduction.

The recovery audit recomputes all 24 cells and verifies the registration,
source identities, worker ledger and all 40 raw worker result/log files.
Its canonical record, `fdtd_conductive_film_report_recovery.json`, is archived outside
the repository; it includes the preserved failure's provenance, artifact hashes and
descriptive summary.
The final-source physical confirmation, full and slow tiers, and frozen
reference checks are complete. Curved conductive bodies,
Drude/Lorentz interfaces, mixed lossy junctions and their derivatives remain
outside this film scope, and the C7 verdict is unchanged.

## Targeted tests and mutation evidence

On the first registered source, after exact restoration of the production
sources, the targeted suite passed
all 53 cases, exit 0. Each of its nine test functions has an actual production
mutation witness: omitted normal memory drive, perturbed lossless tangential
permittivity, omitted interface energy, omitted tangential conductivity,
removed initial-field refusal, removed prepared-plan identity check, silently
detached traced conductivity, omitted induced-pole time-step rate, and omitted
conductivity from the phase-index mesh check. Every witness failed on an
assertion or missing refusal; startup exceptions are not counted as witnesses.

The first mutation-log matcher recognized six failures and missed three
pytest-rewritten `E assert` failures. The original manifest, harness snapshot and raw
logs remain unchanged. A subsequent acceptance audit checked the original,
mutated and restored source hashes, replacement text, raw log hashes, one
assertion failure with exit 1 per witness, and the restored 53-pass baseline.
It accepted all nine without rerunning a mutation or a field solve. This is a
correction to evidence parsing, not a relaxed production test. The canonical
record is `fdtd_conductive_film_mutations.json`; it points to the preserved first
attempt and audit. Earlier targeted fixture failures and calibration exposures
remain in the study's controls directory, `fdtd_conductive_film_controls`, and in
the registration. The record, the attempt, the audit and the controls directory are
archived outside the repository.

After the stencil-refusal correction, a combined run passed all 54 cases:
the 53 new film cases and the existing stencil case that failed in the full
suite. The same nine production mutations were then repeated on the corrected
source. All nine produced accepted assertion failures, every source was
restored byte for byte, and the restored 53-case film suite passed with exit 0.
This second attempt is recorded separately in
`fdtd_conductive_film_final_source_v1_mutations_attempt2.json`, archived outside the
repository; it does not overwrite the original witnesses or their parsing history.

Ruff 0.16.2 checked 614 tracked and new Python files with `--no-cache`; all
seven batches exited 0. The later recovery utility also passed with exit 0,
bringing the recorded coverage to 615 files. The original manifest
`fdtd_conductive_film_ruff_validation.json` retains the file inventory, complete log,
exits and unchanged source identities; the added-file check is in
`fdtd_conductive_film_ruff_recovery_validation.json`.
The corrected production module and new confirmation adapter also passed
`--no-cache`, bringing the cumulative inventory to 616 files. Their log,
current source hashes and inventory are recorded in
`fdtd_conductive_film_ruff_final_source_validation.json`. All three Ruff manifests
are archived outside the repository.
The FD digest, like the campaign's, hashes each sorted path relative to the
hashed subtree and LF-normalized file with NUL separators. This differs intentionally from the older regression
wrapper's delimiter-free digest and must not be compared as the same algorithm.

## Regression evidence and the compatibility correction

The first full regression attempt stopped at
`tests/fdtd/test_fd_stencil.py::test_lowering_refuses_a_scene_the_operator_was_not_prepared_for`:
2,443 tests passed, eight were skipped and one failed in 2,078.58 seconds.
Pytest exited 2 after stopping on the first failure; the validation controller
exited 1. This is an incomplete failed run, not a passing full suite. Its raw
log, `full_attempt1.log` in the controls directory, and its validation manifest,
`fdtd_conductive_film_validation_attempt1.json`, are archived outside the repository.

The new unsupported-film bypass guard intercepted a stencil scene before the
stencil operator could issue its existing certificate error. The correction
narrows that guard to voxel discretization, preserving the stencil refusal.
It changes guard dispatch rather than the film's numerical formulas.
The transition record `fdtd_conductive_film_source_transition.json`, archived outside
the repository, verifies that overlaying the archived pre-correction module on the final package reproduces
both original source digests. Its normalized diff contains only that refusal
routing change. The final FD digest is
`09fe9737be88b1f2e297dbfe8e92de4e44fc46a51f8e9a657ec3516ad98989f5`
under the NUL-separated algorithm above.

The scoped `.gitattributes` rules preserve the tested bytes across checkouts:
the new study helpers, test and film module use LF, while the existing
`observe.py`, `checks.py` and `params.py` retain their tested CRLF endings.
Raw C8 evidence files and snapshots bypass text conversion. This protects the
registrations' and mutation witnesses' exact-byte hashes; it changes no
numerical formula or working source byte. The LF-normalized source digests
alone would not detect a checkout's change to those raw-byte identities.

The original campaign and mutation records remain valid for their recorded
source. The repeated mutation gate has passed on the corrected source. Its
20-worker numerical confirmation was registered at 2026-10-06T06:03:00 UTC
with fingerprint
`4e179b38732fbf0b997ceea1f493fa6b133d76b19163ec59fd7ba21ce8475afd`.
`studies/fdtd/conductive_film/fdtd_conductive_film_final_confirmation.py` imports the unchanged
campaign and probe, redirects artifacts to the `final_source_v1` prefix, and
applies the same scalar-only JSON conversion established in the recovery.
It changes neither the numerical protocol nor the classification rules.
The registration records all 20 earlier outcomes as known: this is a source
confirmation, not a new experiment with unseen outcomes. Its physical report
and audit passed, with all 20 physical payloads identical to the originals.

The final-source full tier collected 2,806 cases and passed with 2,798 passed
and eight skipped, exit 0, in 2,355.36 seconds as reported by pytest
(2,356.09 seconds including the wrapper). There were no failures or worker
crashes. Its manifest, `fdtd_conductive_film_validation.json`, and its raw log,
`full.log` in the controls directory, are archived outside the repository.
The wrapper identifies the same final FD source with its delimiter-free digest,
`b7f5ed7b04c8d75b6c4c6fac127bb19683cddb4e8b64d1832e99af4ff5d84535`.

This document's `Targets:` header was added after that full-tier run,
introducing one additional documentation parameter case. The final document
contract check separately collected and passed all 67 cases, exit 0, including
that addition; it is not included in the 2,806 collected above. The count,
complete collection/run logs and unchanged source hashes are recorded in
`fdtd_conductive_film_docs_final_validation.json`, archived outside the repository.

The separate slow tier collected and passed all 81 cases, exit 0, in
1,201.38 seconds as reported by pytest (1,201.92 seconds including the
wrapper), without failures or worker crashes. The four read-only freezers
reproduced 143, 288, 140 and 570 arrays respectively: all 1,141 arrays matched
bit for bit, all four processes exited 0, and no golden was rewritten.
These results are in the same validation manifest, with `slow.log` and
`freezer0.log` through `freezer3.log` in the controls directory, all archived
outside the repository. Production source identity remained unchanged throughout
the successful gates.

The final Sphinx build uses `-W`, with output and exit status retained in
`fdtd_conductive_film_sphinx_final_validation.json` and in `sphinx_final.log` in the
controls directory, both archived outside the repository.
