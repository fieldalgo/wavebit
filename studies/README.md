# Studies

Open questions, answered once and written down. Each script here exists because somebody did
not know the answer; the table below links the write-up where the answer was recorded and what
it decided. Each is standalone and runs from the repo root.

**This is not a performance regression suite.** If you arrived expecting `asv continuous` and a
dashboard, there is none: these run by hand, they run rarely, and a run that does not end in a
`docs/` change was wasted. Comparisons against external solvers are *not* here — they live in
[`../tests/references/`](../tests/references), because their expected answer is known and that
makes them assertions, not studies.

| Script | Question it answers |
| --- | --- |
| `benchmark_backends.py` | Which backend, and whether to freeze the geometry, for a *differentiated* run. Sweeps the vector and raster paths across four modes and a range of `fto`, and reports the compile cost that decides between them. Write-up: [../docs/backend-choice.md](../docs/backend-choice.md). |
| `benchmark_analytic.py` | How the analytic (staircase-free) shape path converges against the raster and vector staircases, for tilted and curved boundaries. Write-up: [../docs/analytic-modeling.md](../docs/analytic-modeling.md). |
| `benchmark_truncation_shape.py` | Whether S4's circular cut of the Fourier order set beats the full rectangle FieldWave keeps. Sweeps both shapes through the same permittivity raster and solver, on NVF and on Laurent as a control. Write-up: [../docs/fourier-truncation-shape.md](../docs/fourier-truncation-shape.md). |

## Rejected, and why

Some questions end in "do not do this". The code stays out of `develop` — it would be dead
weight — but the answer should not have to be rediscovered, and the branch holding it should not
have to be diffed to find that out.

**Eigendecomposition-free propagator, by matrix square root.** Replacing the per-layer `eig` with
`sqrtm` + `expm` is algebraically exact, not an approximation, and it measures that way:
efficiencies, amplitudes, phases and gradients all match the eig path to 1e-13. It is rejected
anyway, because no branch rule is safe for every structure. Whether a given structure is safe
turns out to depend on the *raster* rather than the optics — the same grating at 48 cells carries
a growing mode where 40, 44, 52 and 56 do not — and the only way to check is to run the
eigendecomposition the method exists to remove. The error is invisible on a thin layer and
unbounded on a thick one, and neither the square root's residual nor energy conservation detects
it. This conclusion is specific to the matrix-square-root formulation; R-DIT and PIM never form a
square root and are not covered by it.

- Branch `claude/rcwa-no-eigendecomposition-b92d89`, tag `study/no-merge/matfun-propagator`.
  **Not merged, and deliberately not rebased.** The value is the measurements, and they are tied
  to the tree they were taken on: `study_matfun_envelope.py` reproduces them against that tree,
  not a later one. Rebasing forward would leave new code carrying old numbers.
- A consequence worth knowing before you sweep: **that branch is out of scope for API renames and
  for `develop`-wide refactors.** Its tree predates the result API rename and the
  `factorization.py` renaming, so it still reads `de_ri` / `de_ti` and
  `to_conv_mat_raster_discrete`. That is expected, not a defect to fix.
- `git diff develop..claude/rcwa-no-eigendecomposition-b92d89` looks enormous. That is staleness,
  not a revert; `git log develop..<branch>` shows the two commits that are actually there.

## Reading a timing table

`benchmark_backends.py` warms the machine for 45 s before it times anything, and times one
fixed configuration three times across the run — at the start, between the two sweeps, and at
the end. That probe is printed with the results, and it is the number to check first: a run
whose three marks spread widely is measuring the machine, not the code. On the host these were
taken from, the same configuration has measured 133 ms and 217 ms on different days with no
code change between them, so treat a ratio within one run as meaningful and a third significant
figure as not.

Results stream to `results/` as they land, so an interrupted run still leaves what it finished.
The `.log` is checked in; the machine-readable `.json` beside it is not (`*.json` is ignored).
