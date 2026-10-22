# PlaneWave acceptance: accuracy and measurement qualification

Short stage-2 `plane_wave` and stage-3 `bloch_stack` / `conical` goldens detect
changes to recorded computations. Their 300/500-step runs deliberately end
before settling. Keep those fixtures and their regression limits; a matching
golden alone does not establish physical accuracy.

An oblique p column uses a centre-normalized differentiated two-node current.
Its independent discrete transfer tests in `tests/fdtd/test_fd_planewave.py`
compare the prescribed input waveform with the resulting field. They retain
the real-frequency 2e-6 and damped-frequency 3e-6 limits, backward leakage and
cross-polarization limits of 1e-13, and DC limit of 1e-6. Normalizing scattered
fields by the measured incident field can hide a source gain defect, so these
source tests remain necessary alongside scattering tests.

## Fixed limits and separate uncertainty caps

`tests/fdtd/test_fd_diffraction.py` retains the planar stack's amplitude,
magnitude, phase, power, closure, decay and 20/40/80-cell convergence criteria.
`tests/fdtd/test_fd_axis1d.py` retains the conical grating's amplitude, per-order
power, closure, reference convergence and 20/40-cell improvement criteria.
In particular, the p stack's phase is first order; this campaign does not
replace that established test with a second-order requirement.

The additional manual campaign qualifies every p refinement using these fixed
limits, denoted by `tau` for each observable:

| Observable | Planar | Conical |
|---|---:|---:|
| Complex amplitudes | .015 at 20/40 cells, .005 at 80 | .0355 |
| Per-order power | .0012 | .01 |
| Total R + T | .0001 | .001 |
| Relative co-polarized magnitude | .02 | — |
| Co-polarized phase (radians) | .06 | — |
| Cross-polarized amplitude | 1e-12 | included in complex amplitudes |

The planar 20-cell row uses its limits only to qualify measurement uncertainty;
its accuracy is judged by the original convergence test. The conical 40-cell
row must also satisfy the original measured improvement factor of 1.8. The
manual campaign is supplemental and cannot substitute for those tests.

1. Run independent records at T times 1, 1.25, 1.5, 1.75 and 2. T is 20 periods
   for the planar fixture and 160 for the conical fixture. The maximum of all
   ten pairwise differences must be at most 0.1 tau, separately per observable.
2. At 2T, double the PML cells from 12 to 24, and independently move the inner
   PML faces away from the structure. Double the lower and upper clearances
   of .8 and 1.4 length units. Keep structure interfaces, source, monitors,
   transverse wavevector and reference planes fixed. Compare all three
   configurations pairwise, again at most 0.1 tau.
3. Check the uniform-stack RCWA reference at truncations 0 and 1, and the
   conical reference at 15 and 30. Each channel must differ by at most .05 tau.
   Compute the angle from the prescribed fixed transverse wavevector, without
   taking an angle from the measured FDTD result.
4. Require finite observables and interior energy, a stable run, a known
   source end before the record end, complete interior-energy evidence,
   and no growth warning. A finite interior-growth ratio qualifies the diagnostic.
   A missing ratio also qualifies when the complete post-source window spans
   at least four of the scene's longest declared periods, all four late eighths have samples, the original
   signal is nonzero and the last-quarter energy peak is at most 1e-7 of the
   full-run peak. This is the engine's documented quiet-floor case; missing,
   short or nonfinite traces still block acceptance. Instantaneous energy need
   not decrease monotonically.

The 10% and 5% allocations are explicit engineering acceptance choices. They
do not prove asymptotic stability. Measured grid/time differences never enlarge
the accuracy tolerance. Complex comparisons retain phase information, and an
intermediate-time excursion cannot be hidden by equal endpoints.

The touching conductive-film fixture additionally runs at n80, grid fraction
.375, 24-cell PML, Courant .2, df .1 and cutoff 1e-6 for 60, 90 and 120 periods.
The maximum pairwise complex r/t difference must be at most 2e-7. This checks
residual time dependence; it does not overturn that fixture's historical
failed spatial-order campaign.

The previous-source goldens remain historical behavior records. For an intended
source change, their affected-scene mismatches establish what changed; they do
not independently establish a physical error. Before a refreeze, require the
independent source/accuracy/qualification checks, cross-backend agreement,
derivatives and unaffected-scene pins. After a justified refreeze, require the
new affected-scene pins and their deliberate-error mutants as well. This order
avoids requiring the new source to reproduce the old source's transient field.

## Stabilized measurement protocol

The original broad-pulse campaign below is retained as historical evidence.
`--protocol stabilized` selects a separate qualification fixture: 80 periods
for planar and 320 for conical as T, 48-cell PML with `stabilize=3e-3`, and
96 cells for the thickness control. The five time factors, independent farther
boundary, source bandwidth, reference calculations and every accuracy and
uncertainty limit stay the same. The domain expands outward when cells are
added, preserving the inner PML faces and physical observation geometry.
An explicit protocol distinguishes a changed experiment from a reinterpretation
of the original failed result. Neither protocol asserts infinite-time stability.

The holding-column diagnostic regressions explicitly inject the historical hard
source inside their test scope. This keeps the trapped mode, growth thresholds
and diagnostic mutants under test after the production p source changes.
A separate test uses the production current and requires a verified post-source
quiet window. The grazing `adds-p` control uses 320 and 640 periods: its return
is 0.001369600 and 0.001369955 versus the independent layer model 0.001370.
The old 160-period current-driven measurement was 0.001349794. The longer pair
must agree within the existing relative 1e-3 pin tolerance; the model comparison
and accept/refuse threshold remain unchanged.

## Frequency-focused measurement protocol

`--protocol spectral` keeps the stabilized inner geometry and durations, but
uses `df=.05, cutoff=1e-6` for the validation pulse and preserves a physical
PML depth of 2.4: 48/96/192 cells at 20/40/80 cells per wavelength, doubled
for the thickness control. Holding 48 cells at every refinement halved the
physical depth at each step and left the 40-cell planar boundary difference
at 1.532 times its fixed uncertainty cap. That failed experiment is retained;
the physical-depth control does not relax the cap. Requested wavelengths, material
interfaces, transverse wavevector, observation planes and all reference values
and tolerances remain unchanged. A smaller start cutoff suppresses the spectral
leakage from abruptly starting a Gaussian; a narrower pulse reduces excitation
of the conical grating's long-lived modes outside the requested frequencies.
This qualifies the requested-frequency linear scattering measurement. It does
not certify the original broad pulse or eliminate its documented PML limitations.
The broad-pulse source transfer and original accuracy tests remain separate.

Each refinement also runs at 2T with `df=.07, cutoff=1e-6`. Its observations must
agree with the baseline 2T within 0.1 tau, with valid diagnostics and the same
absolute accuracy bounds. Every requested incident amplitude must be finite,
and the smallest magnitude must be at least 5% of the largest. A weak incident
signal blocks acceptance instead of buying a looser normalized-error bound.
The complete protocol therefore has 43 independent runs, including the three
unchanged contact residual runs. The conical fine-grid worst complex error must
improve on the coarse-grid error by at least the original factor of 1.8.

`--backend jax` runs on CPU; NumPy remains the default. The stabilized conical
40-cell, 320-period control agreed between NumPy and JAX CPU to 8.9e-16 in every
observed channel (preselected absolute comparison bound 5e-10). Each campaign
records the actual backend and input hashes. This is not GPU validation.

## Outcomes and reproduction

Missing runs, nonfinite values, missing diagnostics or excessive uncertainty
give `BLOCKED`. A qualified measurement outside its accuracy limit gives
`FAIL`. Only qualified measurements within fixed limits give `PASS`. An
observed failure takes precedence over blocked evidence in an aggregate.

Run the frequency-focused campaign from a configured source checkout. The
default protocol remains `original` for historical reproduction:

```sh
PYTHONPATH=src:. python -m tests.fdtd.check_planewave_acceptance --protocol spectral --backend jax --workers 2 --output artifacts/planewave-spectral
PYTHONPATH=src:. python -m tests.fdtd.check_planewave_acceptance --output artifacts/planewave-acceptance
PYTHONPATH=src:. python -m tests.fdtd.check_planewave_acceptance --protocol stabilized --workers 5 --output artifacts/planewave-stabilized
python -m pytest tests/fdtd/test_fd_acceptance.py
```

The output directory must be new. It retains input hashes, versions, individual
records and diagnostics, channel labels and ratios to each fixed budget. A
partial run has no final result and cannot be accepted. A manual `PASS` covers
only this measurement campaign. Source, original accuracy/convergence, backend,
derivative and unaffected-scene regression tests remain separate requirements.

Refreezing is conditional on all those requirements passing, a reviewed list
of changed keys, unchanged unrelated scenes and successful reproduction at the
existing golden tolerance. Never widen a limit just because a new algorithm
misses it. Historical external-solver measurements retain their original
input/version scope; consuming a stored reference is not a fresh external run.

## Original protocol CPU result (2026-10-21)

The 38 independent runs did not qualify a refreeze. The table reports the
largest channel ratio to each uncertainty cap; values above one block acceptance.

| Fixture | Time/cap | Boundary/cap | Qualification |
|---|---:|---:|---|
| Planar 20 | .265 | 5.732 | BLOCKED: source-end/growth evidence and boundary sensitivity |
| Planar 40 | .219 | 2.616 | BLOCKED: source-end/growth evidence and boundary sensitivity |
| Planar 80 | .681 | .800 | BLOCKED: source-end/growth evidence |
| Conical 20 | 6.115 | 3.914 | BLOCKED: time and boundary sensitivity |
| Conical 40 | 10.175 | 8.466 | BLOCKED: growth warnings, time and boundary sensitivity |

The planar source-end diagnostic stays at the record end. The conical 40-cell
160-period run already warns about growth (interior ratio 1.109, threshold 1.1).
All reference-convergence caps passed. The contact residual passed: maximum
pairwise complex r/t change 5.673e-8 against 2e-7, with complete quiet-floor
energy evidence at all three durations. This is a residual result only.

The initial classifier treated every missing growth ratio as missing evidence.
Inspection of the documented diagnostic semantics exposed that error: a fully
sampled field below its existing energy floor also has no growth ratio. The
corrected classifier was checked with 11 counterexample tests, then applied to
the saved traces. All physical values, reference values, numerical ratios and
accuracy limits stayed identical; the initial blocked contact verdict was kept.

Source checks passed 120 tests, original planar/conical accuracy and convergence
passed 9, and the public Torch CPU source oracle passed 5. Stage-2/3 backend,
gradient, decay and grazing tests gave 452 passed and 18 failed: 14 old-golden
mismatches and four old-drive fixture/pin failures (three holding-column tests
and the `adds-p` grazing pin). Both gradient modules passed all 136 tests.
NumPy stage 0–3 capture changed 60 of 573 arrays, confined to the three intended
scenes; the other 513 arrays were byte-identical and both captures reproduced
themselves. No committed golden changed. Linux CPU evidence does not establish
GPU, other-OS, full-CI or independent-review acceptance.

## Physical-depth protocol CPU result (2026-10-22)

The frequency-focused protocol passed all 43 required records: 27 new runs and
16 reused records from the two completed 20-cell cases with identical inputs.
Reuse was checked against scene representations, solver and fixture hashes,
versions and recording/scoring functions; every verdict was recomputed. The
stabilized and fixed-cell-depth candidates remain incomplete failed experiments,
with their completed rows preserved. They are not counted as complete passes.

| Fixture | Time/cap | Boundary/cap | Bandwidth/cap | Accuracy/limit | Result |
|---|---:|---:|---:|---:|---|
| Planar 20 | .000169 | .009242 | .0000344 | qualification only | PASS |
| Planar 40 | .000399 | .023732 | .000164 | .6620 | PASS |
| Planar 80 | .000641 | .032761 | .000288 | .6563 | PASS |
| Conical 20 | .020356 | .004375 | .047166 | .9697 | PASS |
| Conical 40 | .021398 | .005024 | .033646 | .3882 | PASS |

Each entry is the largest channel ratio. Planar 20-cell accuracy remains the
original convergence test's responsibility, as specified before execution.
The conical worst complex error fell from .03442459 to .01378228, a factor of
2.498 against the required 1.8. The fresh contact residual was 5.672528e-8
against 2e-7. Its JAX/previous NumPy observations agreed within 1.5e-15.
Four additional prescribed-input source tests cover both pulse bandwidths
and both launch geometries at the unchanged 2e-6 limit. All 163 decay/grazing
regressions passed with the explicit historical-drive diagnostic fixtures and
the converged grazing record. The unchanged solver's earlier source, original
accuracy/convergence and derivative results remain earlier evidence, not new runs.

These results qualify the requested-frequency measurement and the intended
source change. They do not establish broad-pulse PML stability, repair the
contact fixture's historical spatial-order failure, or certify GPU, other
operating systems, full CI or independent review.

The scoped refreeze retains the 300/500-step scenes and 1e-12 pin tolerance.
Only `plane_wave`, `bloch_stack` and `conical` receive fresh values; all unrelated
array bytes are retained. Both full captures reproduced twice per CPU backend.
Archive metadata preserves the separate provenance of fresh and retained scenes.

## Original-bandwidth contact convergence follow-up

A separate 36-record CPU campaign with the current source and original contact
pulse (`df=.1`) passed the registered asymptotic criterion on s/p polarization
and grid phases .375/.125. It includes an additional fine-grid doubled-time and
doubled-PML check. The finest reflection-order intervals span 1.891–2.175;
complex h² coefficient changes are at most 1.470% against 5%. Fine-grid
time-plus-boundary uncertainty consumes at most .148 of its fixed budget.
The original coarse-pair failure remains unchanged. The new evidence supports
the pre-asymptotic cancellation explanation within this fixture; it is not a
general support decision. Full criteria and results are in
[the contact follow-up](fd-conductive-contact.md#original-bandwidth-follow-up-2026-10-22).


### Original broadband PML follow-up (2026-10-22)

Sixteen additional conical-p controls retained `df=.15`, `n=40` and the
original observation planes. Each boundary choice was run for 160 and 320
periods. A deeper PML (96 cells), optionally with `stabilizing_loss=.003`,
reduced the measured late-time interior growth ratio below one. Doubling its
width or moving the inner boundary passed the fixed boundary-sensitivity cap.
However, the time differences still exceeded `0.1 tau` by factors 4.54–4.86.
Increasing stabilizing loss to `.03` gave time ratios 3.94 and 4.47 for 96 and
192 cells, respectively; its boundary ratio was 0.628 (pass requires at most
one). Absolute reference errors below `tau` do not override these time failures.

Four separate source-free slab eigenvalue controls support the known pure-PML
bound-mode growth mechanism: growth rates were 0.01590 and 0.00845 for 12 and
24 cells with zero stabilizing loss, versus below 4e-13 with loss `.003`
(the existing stability threshold is 1e-9). This small source-free case does
not prove stability or settling of every driven broadband scene. The original
broadband case remains unqualified; these two-time controls do not replace the
five-time acceptance protocol above. No solver, golden or tolerance was changed
for this follow-up.

## Long broadband records and intermediate-time checks (2026-10-22)

The original conical bandwidth `df=.15` was retained at 40 cells per wavelength,
with cutoff `1e-6`, 96 PML cells (physical depth 2.4), and the same geometry,
transverse wavevector, monitors, reference and fixed limits. Four new CPU/JAX
records compared 640 and 1280 periods at stabilizing losses .003 and .03.

| PML loss | 640–1280 time difference / cap | Interior-growth ratios | Two-time result |
|---|---:|---|---|
| .003 | 1.771860 | .720598 / .518109 | FAIL |
| .03 | .988485 | .474283 / .223998 | PASS |

Both largest changes were in the total power at wavelength 1.0. Their absolute
accuracy ratios remained below .389, which cannot replace time qualification.
The .003 comparison failed its preregistered trigger for a five-time extension;
that extension was not run. A separately registered .03 record at 960 periods
then gave a 640–960 difference of **1.138452 times the same cap**. The 960–1280
maximum was .277307. Thus the marginal endpoint pass does not pass even this
three-time comparison. These records establish neither the full five-time and
boundary qualification nor a new default PML setting.

One additional .003/640-period run recorded Ey and Ez at points on both monitor
planes. Adding the four passive records changed no scattering observation
(maximum difference 0, limit 1e-12). The late 320–640 Hann spectrum peaked at
.778116 with frequency resolution .003125. Two-exponential matrix-pencil fits
resolved a persistent oscillation near ±.777, distinct from the +1-order Yee
cutoff .786622. Fits starting at 480 gave decay rates about .00102–.00107,
corresponding to an amplitude e-folding time of roughly 1000 centre periods.
Relative residuals were 1.3–1.4% for three point components and 9.1% for transmitted
Ey; earlier-window fits were less accurate. A one-exponential fit was inadequate.
Two temporal exponentials do not identify two spatial eigenmodes.

This is evidence for a long-lived oscillation after growth is suppressed, not a
spatial-mode identification or an infinite-time stability proof. The point
records do not justify a Q estimate or correcting the measured DFT by a fitted
tail. The related [ringdown analysis](fdtd-ringdown.md) and
[pulse-convergence analysis](fdtd-pulse-convergence.md) describe why window and
model dependence matter; their different fixtures do not establish the mechanism
of this oscillation. All failed time comparisons remain failed at their original
limits. No solver, source bandwidth or acceptance tolerance changed in this follow-up.

## Source-specific M1 regression qualification (2026-10-22)

The original hard-source M1 p regression recorded an n20 maximum amplitude
error of .027752897808 against stored RCWA. The current-source result is
.027752997432. That single-grid increase was preserved as a failed historical
maximum comparison; it did not establish a loss of spatial convergence.
Six fresh CPU records now qualify the current source using the original C7
formulas: JAX n20/40/80 at 120 periods, JAX n40 at 180, and NumPy/Torch n20 at 120.
The stored RCWA ladders and shift instruments were reused, not rerun.

| Cells per wavelength | Maximum amplitude error A | Maximum power error F |
|---|---:|---:|
| 20 | .0277529974 | .0136295190 |
| 40 | .0066034566 | .0034952817 |
| 80 | .0018113039 | .0009058605 |

The fitted convergence orders are 1.968772 and 1.955651, above the unchanged 1.8
requirement. Reference uncertainties .0002322194 and .0001661714 remain below
one third of the fine errors. All 72 cell comparisons satisfy the original
Richardson consistency criterion; both deliberate displacement controls fail
as required. The n40 time changes are 5.76523e-7 and 2.21174e-7, below caps
.0001811304 and .0000905860, respectively. All run-validity and layout checks
pass. Maximum NumPy/JAX/Torch relative disagreement is 6.59e-15, below 1e-10.
This qualifies M1 p only, not the whole metasurface campaign.

`tests/references/fd_metasurface_rows.json` remains byte-for-byte unchanged.
The regression explicitly replays its original hard drive and retains both its
1e-10 pins and its recorded RCWA error caps. A separate
`tests/references/fd_metasurface_current_rows.json` holds the qualified current
NumPy row, using the same pin tolerance and error-cap formula
`recorded * (1 + 1e-9) + 1e-12`. It retains exactly the same RCWA reference values,
channels and 120-period scene. Its metadata records qualification results,
source/version identities, raw-record hashes and the historical archive hash.
No historical failure is reclassified, and no physical tolerance is widened.

From a configured checkout with its root and `src` on `PYTHONPATH`, reproduce
with `python -m tests.fdtd.check_metasurface_current --run --records
artifacts/m1-current --output artifacts/m1-current-report.json`. The directory
must be new. `--workers 2` permits two CPU workers; the default is one. The
original one-time 180-period repeat applies only to an unsettled rung. To grade
existing records, omit `--run`; different source/environment provenance is
rejected. `--regression artifacts/m1-current-candidate.json` exports a separate
candidate only after every physical criterion passes and refuses to overwrite
an existing file. Review the evidence before promoting a candidate. This
manual qualification does not replace CI, independent review or other-platform
validation.
