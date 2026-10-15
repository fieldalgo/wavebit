# Automatic stopping after a transient

> **Historical API:** snippets and descriptions using `wavebit.compat` on this page require a historical checkout.
> The namespace is absent from current installations; the pre-retirement baseline is
> [revision `3c8a50f8`](https://github.com/fieldalgo/wavebit/tree/3c8a50f8a6506240169538957e99648a4b18ca3a).
> Reproduce recorded experiments with their own cited commit and environment; their measurements have not been rerun.


Written: 2026-10-07

Targets: `src/wavebit/fdtd/_scene.py`, `src/wavebit/fdtd/monitors.py`,
`src/wavebit/fdtd/_prepare/plan.py`, `src/wavebit/fdtd/_execution.py`, `src/wavebit/fdtd/_prepare/report.py`,
`src/wavebit/fdtd/_decay.py`, `src/wavebit/fdtd/_runtime.py`,
`tests/fdtd/test_fd_decay_stop.py`, `studies/fdtd/decay_stop/fdtd_decay_stop_probe.py`.

TL;DR: explicit amplitude windows can end a concrete CPU transient before its
hard cap. Every field and evolving material/absorber array has its own peak;
a stop remains a heuristic, not a spectral accuracy certificate.

## What the stopping condition measures

An optional `Run(decay=Decay(threshold=..., window=...))` requests early
termination while retaining the explicit `steps`, `time` or `periods` limit.
Both the threshold and the window are caller choices. They are not defaults
fitted to a scattering example and do not promise a spectral error bound.
The window is a time in the same length unit as `Run(time=...)`, with `c=1`.
Since 2026-10-13 `Decay` is reachable only through `wavebit.compat.legacy.fd`;
the public `wavebit.fdtd` module runs a fixed `Duration` and has no automatic stop.

The stop condition reads the complete electric and magnetic field arrays,
every CPML history array and each supported material's evolving memory array.
These are separate groups: arrays with different physical units are never
added. Each group uses its largest absolute real or imaginary entry, relative
to that group's largest value seen since initialization. The decision compares
the exact product of represented finite values, so a rounded or underflowed
quotient cannot manufacture a pass; reported ratios are display values. This is a relative
amplitude condition on the state, not electromagnetic energy. Avoiding squares
also avoids turning a small nonzero state into zero merely by squaring it.

The window is rounded up to a whole number of steps. Every step in each
complete, non-overlapping window is read. The run can stop at a window boundary
only when every group's maximum over that window satisfies the requested
relative threshold. Peaks include the initial fields. A field must have been
nonzero at least once; an empty scene does not count as a successful decay.
A nonfinite reading cannot subsequently be forgotten and relabelled converged.

The existing post-run decay diagnostic measures a tail's change relative to
its peak. It retains its previous meaning. A constant nonzero field can have
zero change, so it is not a stopping criterion for this feature. In particular,
a quiet point monitor is insufficient: a field elsewhere, a polarization
memory, or an absorbing-layer history may still be active.

## Source eligibility and supported execution

The Gaussian `Band` waveform continues beyond its nominal cutoff; this
feature does not truncate or otherwise change it. A complete test window must
start after every declared Dipole's envelope has fallen below both its band
cutoff and the requested amplitude threshold, on the falling side of the
pulse. This eligibility rule also waits for delayed sources. It is a bound on
the envelope, not a statement that the remaining drive is exactly zero.

Initial fields and Dipoles are the first supported excitation scope.
PlaneWave and TFSF sources carry auxiliary columns that can return delayed
echoes and are excluded from automatic stopping. Traced, differentiated and
batched execution is excluded: a data-dependent termination time is not the
fixed scan differentiated by the existing solver. Ordinary fixed-duration
runs retain their existing source and derivative contracts.

Only concrete CPU NumPy, JAX and Torch routes are admitted. Material
admission rules remain those of the ordinary scene. Every admitted dynamic
term must have an explicit stop-state mapping; a newly introduced unsupported
term must refuse rather than silently omit its state. This extension does
not complete general C8 smoothing, resolve the C7 convergence questions, or
establish GPU accuracy or speed.

## Returned time interval and cost

The run advances one trajectory. It does not restart the simulation at each
window. Source samples and monitor time series end at the actual last step;
frequency-domain accumulation and endpoint corrections use that same step.
For an actual count N, E ends at N*dt and H at (N-1/2)*dt; the
completed interior-energy rows are T1 through T(N-1). The pending
electric-only TN sample is excluded.
The returned plan and report describe the interval actually executed. The
stopping diagnostics retain the requested limit and identify whether the
criterion was met or the hard limit was reached. A final partial window cannot
be reported as a complete successful window.

A shortened result's plan is not the original capped scene's reusable plan.
Keep the plan from `fd.plan(scene)` when reusing that scene. Its stopping
policy is part of its identity, so changing the policy requires a new plan.

Automatic stopping allocates histories for the requested cap. It does not
reduce their planned allocation. It also reads state norms every step and
uses a complete per-step interior-energy history, so arbitrary termination
cannot fabricate an unsampled final energy row. These additional reads cost
time; early termination is not by itself a claim of faster wall-clock
execution. Steady or static residual states can legitimately run to the cap.

JAX CPU can flush subnormal values during native floating reductions. The
optional stopping runtime therefore orders nonnegative float64 bit patterns
as integers for both the component maximum and window merge, then converts
the selected bits back. Signs are removed and any nonfinite component yields
infinity. Field updates and fixed-duration execution do not use these helpers.

## Measured stopping behavior

Calibration attempt 1, recorded in `fdtd_decay_stop_calibration_attempt1.json` and archived
outside the repository, ran a uniform periodic conductor and an open point source on NumPy,
JAX and Torch CPU. Each row compared automatic stopping with a fixed run to the same
actual step and another fixed run to twice the cap. All six worker exits were
0 and all saved arrays were finite. The chosen threshold was 1e-5 and the
window was 1; these are test settings, not recommended defaults.

| Scene | Result on all three backends | Largest relative temporal DFT difference from twice-cap run |
| --- | --- | --- |
| Uniform conductor | `decay`: 490 of 1386 cap steps | 7.457789111015084e-7 |
| Open point source | `cap`: 1109 of 1109 cap steps | 1.0057980092140956e-10 |

The conductor's independent scalar recurrence predicts the exact stopping
step, 490. Its largest analytic phasor discrepancy is 6.580664815877184e-15.
Adaptive and equal-length fixed outputs are byte-identical in these six rows,
including source histories and energy values at common fixed timestamps.
This tests the discrete recurrence and returned interval; a uniform zero-curl
field does not measure spatial accuracy.

The open-source rows are **not successful early stops**. Their final receipts
show four CPML groups, `psi:x±:H:2` and `psi:y±:H:2`, with own peaks between
1.81e-16 and 3.02e-16 and ratios between 0.747 and 0.929. All other groups in
that receipt are below threshold. Strict normalization by each group's own
peak can leave floating-point-scale residuals above the relative threshold;
their physical or numerical cause is not established by these records.

These final ratios describe a 59-step partial block after 15 complete
70-step windows. They are not a history of the groups that blocked earlier
complete-window decisions. A partial block cannot qualify as convergence.
The small DFT difference compares two finite durations on the same grid; it
does not turn the cap outcome into a decay success or an error bound.

The point-source histories exactly match the fixed run and the first 1109
samples of the 2218-step reference. The Gaussian drive remains nonzero after
eligibility, with a largest recorded subsequent value of
7.926365073918224e-7. No waveform cutoff was introduced.

Every conductor solve retained a warning that its wavelength-based material
mesh check did not run. Every open-source solve retained a warning for 10.28
cells per wavelength. Neither fixture establishes continuum accuracy. The
conductor's lower step count also did not produce a wall-time speedup in
these executions: compilation, solve order and detector overhead prevent a
fair performance comparison.

The [post-disclosure acceptance](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/decay_stop/records/fdtd_decay_stop_calibration_acceptance.json)
sets prefix and analytic allowances to 128*eps64 = 2.842170943040401e-14 and
the temporal DFT comparison allowance to 1e-6. These are empirical checks for
the fixed cases, not public accuracy guarantees. The unit tests' separate
max(1, reference peak) scaling is not a weak-signal relative certificate.
Calibration responses remain known data when the broader campaign repeats
these two scenes.

## Engineering checks

The first structural checks exposed a real JAX subnormal-reduction defect.
The optional runtime now preserves positive float64 bit patterns through both
the component reduction and window merge, as described above. Later failures
were test-fixture defects; their corrections did not weaken the stopping
condition or numerical allowances.

| Preserved attempt and its record | Observed result | Correction or conclusion |
| --- | --- | --- |
| Structural 1, `fdtd_decay_stop_structural_attempt1.json` | 53 passed, 1 failed | JAX reduced a nonzero subnormal amplitude to zero; fixed both optional reduction sites. |
| Structural 2, `fdtd_decay_stop_structural_attempt2.json` | 86 passed, 16 failed | Synthetic Torch `where` fixtures needed array booleans. |
| Structural 3, `fdtd_decay_stop_structural_attempt3.json` | 102 passed | Expanded structural checks passed before physical calibration. |
| Targeted 1, `fdtd_decay_stop_targeted_attempt1.json` | 165 passed, 9 failed | Magnetic-Dipole fixtures omitted their explicit magnetic kind. |
| Targeted 2, `fdtd_decay_stop_targeted_attempt2.json` | New tests: 174 passed; compatibility: 625 passed, 3 failed | A test-only skeleton helper needed to omit an absent optional decay field. |
| Targeted 3, `fdtd_decay_stop_targeted_attempt3.json` | New tests: 174 passed; compatibility: 628 passed | Both commands exited 0 after the fixture corrections. |
| Full regression 1, `fdtd_decay_stop_validation_attempt1.json` | 4,026 passed, 8 existing skips, 1 failed | Another legacy test reconstructed the static plan without excluding the absent optional decay field. Slow tests and freezers were not started in this attempt. |

The attempt records named in the first column are archived outside the repository.

The three magnetic-kind additions and the optional-field test-helper correction are
preserved as two `repair.patch` files in the repair record
`fdtd_decay_stop_fixture_repair_after_calibration`, with the earlier failing logs and
source snapshots; that record is archived outside the repository. They do
not change the calibration fixtures, production equations, or five frozen
study helpers.

The original calibration and acceptance remain unchanged. A registration adapter,
`registration_adapter.py.raw` in the same archived repair record, separately binds the
two test-only repairs, their exact before/after bytes, and the observed failing and
following gates. It proves that deleting the
three constructor keywords and the one helper tuple entry recovers the
original complete test syntax trees. Numerical criteria and the original
physical controller's non-registration functions remain unchanged.

## Registered physical comparison

The registered campaign, recorded in `fdtd_decay_stop_campaign.json` and archived outside
the repository, completed all 18 fixed workers, with no replacement or parameter change.
Each worker performed the automatic, fixed-prefix and twice-cap solves.
Every registered contract passed. Three rows ended by decay and 15 reached
the cap; the cap rows are not counted as successful convergence.

| Case | Actual/cap steps on NumPy, JAX and Torch | Interpretation |
| --- | --- | --- |
| Uniform conductor | 490/1386, decay | Independent scalar map predicts the exact first passing window. |
| PEC cavity | 832/832, cap | Persistent fields do not qualify as decayed. |
| Delayed Dipoles | 2079/2079, cap | Late-source eligibility and every state group remain part of the criterion. |
| Retained Debye memory | 832/832, cap | A field-only counterfactual passes at 490, while retained polarization blocks the complete criterion. |
| Weak PML return | 1663/1663, cap | The result retains a finite-duration comparison without claiming convergence. |
| Open Dipole | 1109/1109, cap | The conservative small-group limitation above remains visible. |

All saved actual-prefix comparisons are byte-identical, including complete
energy rows at the fixed executor's timestamps. The largest analytic
relative discrepancy across the conductor and Debye checks is
7.660992160844328e-15. The largest temporal DFT discrepancy for a claimed
decay remains 7.457789111015084e-7. The six repeated calibration configurations
have the same array hashes and stopping receipts; they are known exposures.

The Debye counterfactual checks the stated all-memory contract. Static
retained polarization is not evidence of a later physical error. Neither
this example nor the other cap rows proves a future-echo bound, and the
fixed-grid temporal comparisons make no spatial convergence claim.
A saved-array replay, `fdtd_decay_stop_campaign_audit.json` (archived outside the
repository), independently reconstructs the stopping receipts, prefix/spectral metrics
and both scalar recurrences from the saved evidence without a field solve.

## Regression and deliberate faults

All 20 actual production-source mutations, recorded in `fdtd_decay_stop_mutations.json`
and archived outside the repository, caused their intended assertion or missing-refusal
failures across the ten new and four changed test functions. They cover false early windows,
rounded ratios, dropped memory groups, incorrect sample clocks and DFT
endpoints, AD conversion, unsupported execution, and the fixed-run cadence.
Every original source byte was restored; the combined restored selection
passed all 185 cases. A separate
[failure review](https://github.com/fieldalgo/wavebit/blob/develop/studies/fdtd/decay_stop/records/fdtd_decay_stop_mutation_review.json)
records the actual first failure from each log.

The first full regression then found one additional stale fixture in the
junction test. Its one-constant correction, `repair.patch` in the repair record
`fdtd_decay_stop_postcampaign_fixture_repair` (archived outside the repository),
excludes absent `decay` from the independently reconstructed legacy plan.
The entire test module is otherwise syntax-tree identical. The numerical
implementation, physical fixtures, acceptance and registered responses remain
unchanged; the failed full run and its original source snapshots are retained.

A separate post-campaign witness, `completed.json` in that repair record,
passed the 174 new cases plus this corrected old test, then removed the actual
production omission of absent `decay`. The repaired test failed at its intended
skeleton equality. Exact production restoration was followed by that test
passing again. This brings the total to 21 real source-mutation mechanisms
covering ten new and five changed test functions; the earlier 20-mechanism,
185-case restoration record remains unchanged.

A validation adapter, `validation_adapter.py.raw` in the same repair record,
binds that exact test-only change to fresh validation snapshots. It preserves
the original five study helpers and the original prevalidation record, and
runs the same full, separate slow and read-only freezer commands.

The second full regression and separate slow run, recorded in
`fdtd_decay_stop_validation_attempt2.json` and archived outside the repository, completed
with 4,027 passed and the same eight existing skips in the non-slow selection, followed by all 81 slow cases passing. The disjoint selections
contain 4,116 cases including the skips; documentation-contract cases are
already part of the full selection and are not added again.

All four read-only freezer commands reproduced 143, 288, 140 and 570 arrays
bit for bit, respectively: 1,141 arrays in total. No golden was regenerated.
The enclosing validation process exited 0. These engineering gates preserve
previous behavior within the tested scope; they do not establish an error
bound for automatic stopping or complete the other FDTD development stages.
Final documentation validation comprises the documentation-contract checks
and a warning-fatal Sphinx build of these fixed sources.
