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
   source end before the record end, an available interior-growth diagnostic,
   and no growth warning. Instantaneous energy need not decrease monotonically.

The 10% and 5% allocations are explicit engineering acceptance choices. They
do not prove asymptotic stability. Measured grid/time differences never enlarge
the accuracy tolerance. Complex comparisons retain phase information, and an
intermediate-time excursion cannot be hidden by equal endpoints.

The touching conductive-film fixture additionally runs at n80, grid fraction
.375, 24-cell PML, Courant .2, df .1 and cutoff 1e-6 for 60, 90 and 120 periods.
The maximum pairwise complex r/t difference must be at most 2e-7. This checks
residual time dependence; it does not overturn that fixture's historical
failed spatial-order campaign.

## Outcomes and reproduction

Missing runs, nonfinite values, missing diagnostics or excessive uncertainty
give `BLOCKED`. A qualified measurement outside its accuracy limit gives
`FAIL`. Only qualified measurements within fixed limits give `PASS`. An
observed failure takes precedence over blocked evidence in an aggregate.

Run the manual campaign from a configured source checkout:

```sh
PYTHONPATH=src:. python -m tests.fdtd.check_planewave_acceptance --output artifacts/planewave-acceptance
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
