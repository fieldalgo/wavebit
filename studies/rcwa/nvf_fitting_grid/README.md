# Spectral fitting and normal-field resolution

| Entry | Question it answers |
| --- | --- |
| `nvf_fitting_grid.py` | How far do four fitted dimensions move when the normal-field grid changes, with the target and Fourier order held fixed? |
| `measurement.json` | Did the declared 128/256/512/1024 ladder meet both refinement tolerances on the measured solver? |

The runnable workflow is documented in
[the fitting guide](../../../docs/guide/rcwa/fitting-grid-convergence.md).
The study uses JAX forward differentiation and NumPy/SciPy fitting; it runs on CPU
and can take substantially longer than the small example. Output is a new JSON
file chosen by `--output`; each completed fit is checkpointed atomically with
`completed=false` until the final decision. Exit code 2 preserves an unconverged
completed run. An existing output is refused.

`--batch-size` changes only how many of the same 30 optical observations run
simultaneously (default 5); the JSON records it. A larger batch can reduce repeated
geometry work at the cost of memory. Use the same numerical parameters when
comparing batches.

`measurement.json` is a reviewed CPU double-precision observation, not a golden
or an independent reference. Its provenance records the solver baseline and reproducer commit,
versions and unchanged acceptance settings. It preserves the generated data,
target Jacobian, all completed fit stages and the negative convergence decision.
The study command used `--batch-size 30` and the default grid ladder. Solver
changes, especially a different normal-field construction, require a fresh run;
this record does not qualify those changes.
