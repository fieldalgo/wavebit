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
| `fdtd_directional.py` | Does E/H direction separation isolate boundary returns from a slowly decaying near-Rayleigh mode? [Measurement](../docs/fdtd-directional.md). |
| `fdtd_pml_strength.py` | How do CPML strength, physical thickness and refinement affect returning near-Rayleigh waves after source narrowing? [Measurement](../docs/fdtd-pml-strength.md). |
| `fdtd_rayleigh.py` | Can a cutoff classification agree while near-Rayleigh scattering remains unsettled? [Measurement](../docs/fdtd-rayleigh.md). |
| `fdtd_enz.py` | Which errors remain in a lossy ENZ film after its two incidence columns settle? [Measurement](../docs/fdtd-enz.md). |
| `fdtd_layer_alignment.py` | Does fitting every layer boundary reduce planar dispersive-interface errors under refinement? [Measurement](../docs/fdtd-layer-alignment.md). |
| `fdtd_oblique_pulse.py` | Which oblique-metal errors come from pulse bandwidth, interface placement or unresolved diffraction? [Measurement](../docs/fdtd-oblique-pulse.md). |
| `fdtd_mixed_metal.py` | How do low-loss mixed Drude/Lorentz amplitudes and damping sensitivities settle and refine? [Measurement](../docs/fdtd-mixed-metal.md). |
| `fdtd_lorentz_decay.py` | Can material motion survive field decay, and how does a resonant slab settle toward Airy? [Measurement](../docs/fdtd-lorentz-decay.md). |
| `fdtd_bragg_cavity.py` | Does a long-lived cavity reveal incomplete time settling or spatial detuning? [Measurement](../docs/fdtd-bragg-cavity.md). |
| `fdtd_dft_sampling.py` | How much monitor time does DFT subsampling save, at what quadrature error? [Measurement](../docs/fdtd-dft-sampling.md). |
| `fdtd_pulse_convergence.py` | Does temporal settling certify slab scattering accuracy? [Measurement](../docs/fdtd-pulse-convergence.md) separates calibrated amplitude changes from spatial error. |
| `fdtd_metal_validation.py` | How a damped negative-epsilon pattern changes with spatial resolution, runtime, PML and padding. Preserves the discrepancy against Meep resolution 60; see [../docs/fdtd-alpha.md](../docs/fdtd-alpha.md). |
| `fdtd_alpha_controls.py` | How runtime, PML depth and padding change the two crossed dielectric results at fixed grids. Write-up: [../docs/fdtd-alpha.md](../docs/fdtd-alpha.md). |
| `fdtd_alpha_convergence.py` | How two crossed dielectric scenes refine against RCWA, with actual grids, residuals and source provenance. Write-up: [../docs/fdtd-alpha.md](../docs/fdtd-alpha.md). |
| `fdtd_install_smoke.py` | Whether a clean wheel installation computes 36 planar material/incidence combinations without optional backends. Run outside the checkout with the clean environment Python. |
| `scene_hash.py` | Not a question -- an instrument. One byte digest over 72 scenes (4 cell shapes x 3 modeling routes x 2 incidence sides x 3 backends), efficiencies and reconstructed fields, refusals hashed by name. Run it before and after a change that must move no number. |
| `benchmark_backends.py` | Which backend, and whether to freeze the geometry, for a *differentiated* run. Sweeps the shape and raster paths across four modes and a range of `truncation_order`, and reports the compile cost that decides between them. Write-up: [../docs/backend-choice.md](../docs/backend-choice.md). |
| `sweep_cell_angle.py` | How to sweep a unit-cell angle, and the one way it goes wrong. Three arms -- a raster redrawn at each angle, one drawn once and reused, and parametric circles as the staircase-free control -- plus the axis ratio of the lit pixels, which is the mechanism rather than the symptom. Write-up: [../docs/oblique-lattice.md](../docs/oblique-lattice.md), "Sweeping the cell angle". |
| `benchmark_analytic.py` | How the analytic (staircase-free) shape path converges against a raster staircase, for tilted and curved boundaries. Write-up: [../docs/parametric-modeling.md](../docs/parametric-modeling.md). |
| `benchmark_truncation_shape.py` | Whether S4's circular cut of the Fourier order set beats the full rectangle Wavebit keeps. Sweeps both shapes through the same permittivity raster and solver, on NVF and on Laurent as a control. Write-up: [../docs/fourier-truncation-shape.md](../docs/fourier-truncation-shape.md). |
| `benchmark_boundary_normal_field.py` | Whether giving a multi-shape smooth layer the boundary route's gradient-of-distance normal field fixes its position gradient. Integrates the autodiff gradient across a window and compares it against the change in the loss — the one comparison autodiff and finite differences cannot both pass by agreeing with each other. Write-up: [../docs/boundary-modeling.md](../docs/boundary-modeling.md). |

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
  `to_conv_mat_raster_discrete`. That is expected, not a defect to fix. It also predates the
  material convention, so its structures are **refractive index** where everything here is now
  relative permittivity — do not carry a scene across in either direction without converting,
  since neither reading raises.
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
