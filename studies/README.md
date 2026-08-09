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
