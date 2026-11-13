# Spectral fitting and normal-field resolution

| Entry | Question it answers |
| --- | --- |
| `nvf_fitting_grid.py` | How far do four fitted dimensions move when the normal-field grid changes, with the target and Fourier order held fixed? |
| `check_grid.py` | Are fixed-geometry values and derivatives stable under grid refinement, and does AD agree with finite differences? |
| `measurement.json` | Did the historical solver `18bbcbfd` meet both refinement tolerances through grid 1024? |
| `integral-fixed-point.json` | How did fixed-point spectra, Jacobians and finite differences compare after the integral normal field landed? |
| `integral-fitting.json` | Does refitting the integral construction meet the unchanged parameter and spectral refinement criteria? |

The runnable workflow is documented in
[the fitting guide](../../../docs/guide/rcwa/fitting-grid-convergence.md).
The study uses JAX forward differentiation and NumPy/SciPy fitting; it runs on CPU
and can take substantially longer than the small example. Output is a new JSON
file chosen by `--output`; each completed fit is checkpointed atomically with
`completed=false` until the final decision. Exit code 2 preserves an unconverged
completed run. An existing output is refused.

`--batch-size` bounds a small unrolled group of scalar JIT calls over the same
30 optical observations (default 5); the JSON records it and the execution mode.
The scalar lane is traced once and reused. This avoids an outer spectral `vmap`,
which can evaluate both branches of the normal-field kernel's conditional, and
avoids tracing the entire spectrum as one large program. Larger batches can cost
substantially more compilation memory; start with the default. Batching does not
change the observations or numerical tolerances. `Model(...)(p, derivative=False)`
uses a separate value-only function for finite-difference checks.

`measurement.json` is a reviewed CPU double-precision observation on solver
`18bbcbfd`, not a golden
or an independent reference. Its provenance records the solver baseline and reproducer commit,
versions and unchanged acceptance settings. It preserves the generated data,
target Jacobian, all completed fit stages and the negative convergence decision.
The study command used `--batch-size 30` and the default grid ladder. Solver
changes, especially a different normal-field construction, require a fresh run;
this record does not qualify those changes.

Before fitting, run `check_grid.py` for each grid, using a new output each time:

```shell
python studies/rcwa/nvf_fitting_grid/check_grid.py --grid 128 --fd --solver-revision 2056a39e --output artifacts/check-128.json
python studies/rcwa/nvf_fitting_grid/check_grid.py --grid 256 --previous artifacts/check-128.json --solver-revision 2056a39e --output artifacts/check-256.json
```

Use the revision of the solver actually imported, then continue with 512 and 1024
using the preceding output. The script retains all 30 observations and four
Jacobian columns. At the fixed generating geometry, an adjacent spectrum change
must be at most `1e-5`; each Jacobian column's maximum difference divided by the
finer column's maximum magnitude must be at most `1e-3`. `--fd` checks the original
four step sizes and their halves with Richardson extrapolation, also at `1e-3`.
Exit code 2 retains a completed measurement with a failed check. A first grid
without `--fd` has no comparison checks; exit code 0 is not a convergence claim.
These pointwise checks do not replace refitting or independent optical validation.

`integral-fixed-point.json` retains all four completed fixed-point measurements
on solver `2056a39e`, their original instrument hashes and the adjacent-grid
comparisons. Grid-128 AD/FD passes. Spectra pass from 256 to 512 and from 512 to
1024, but the angle Jacobian still fails the `1e-3` stability criterion at 1024.
This is a measured limit, not a reason to relax the criterion.

`integral-fitting.json` retains the completed refitting ladder on solver
`2056a39e`, with a new grid-128 target held fixed and the original tolerances.
Its fit-stability decision is separate from the fixed-point Jacobian criterion.
The provenance identifies the actual solver and committed study source.
