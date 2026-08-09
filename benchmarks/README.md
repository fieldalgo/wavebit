# Benchmarks

Performance and convergence analyses. Each script is standalone and runs from the repo root.

| Script | Question it answers |
| --- | --- |
| `benchmark_backends.py` | Which backend, and whether to freeze the geometry, for a *differentiated* run. Sweeps the vector and raster paths across four modes and a range of `fto`, and reports the compile cost that decides between them. Write-up: [../docs/backend-choice.md](../docs/backend-choice.md). |
| `benchmark_analytic.py` | How the analytic (staircase-free) shape path converges against the raster and vector staircases, for tilted and curved boundaries. Write-up: [../docs/analytic-modeling.md](../docs/analytic-modeling.md). |
| `benchmark_oblique_lattice.py` | Non-orthogonal cells: what they cost, and whether they are right. Five checks — Li's published checkerboard table, S4 against a square-lattice control, a unimodular shear, the supercell it replaces, and the cost ratio. Write-up: [../docs/oblique-lattice.md](../docs/oblique-lattice.md). |
| `reti_1D.py`, `reti_1Dc.py`, `reti_2D.py`, `interface/` | Comparisons against RETICOLO, GRCWA and TORCWA. |

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
