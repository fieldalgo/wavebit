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
| `fdtd_painted_shapes.py` | Do overlapping painted rectangles and their angle/relative-position/width derivatives converge to an independent inclusion-exclusion Fourier profile? [Measurement](../docs/fdtd-painted-shapes.md). |
| `fdtd_multilayer_optimization.py` | Do a constrained six-variable coating design and its gradients improve under grid refinement and across wavelengths absent from training? [Measurement](../docs/fdtd-multilayer-optimization.md). |
| `fdtd_rectangle_gradients.py` | How do transverse width/position derivatives converge to analytic-shape RCWA and the independent translation phase law? [Measurement](../docs/fdtd-rectangle-gradients.md). |
| `fdtd_rotation_gradients.py` | How do exact area-integrated rotation and width derivatives approach independent analytic RCWA as the Yee mesh is refined? [Measurement](../docs/fdtd-rotation-gradients.md). |
| `fdtd_rotation_smoothing.py` | Does translation averaging improve derivatives of the same graded profile, and how large is its sharp-profile bias? [Measurement](../docs/fdtd-rotation-smoothing.md). |
| `fdtd_adjoint_comparison.py` | How large are adjoint-versus-difference gaps compared with each solver's continuum-gradient error? [Measurement](../docs/fdtd-adjoint-validation.md). |
| `fdtd_ellipse_gradients.py` | Do rotated-ellipse angle and axis-length derivatives approach a matching independent Bessel-series optical target? [Measurement](../docs/fdtd-ellipse-gradients.md). |
| `fdtd_circle_gradients.py` | How do circular-cell diameter derivatives approach an independent Bessel-series optical target, and how large is the smoothing bias? [Measurement](../docs/fdtd-circle-gradients.md). |
| `fdtd_variable_depth.py` | Do moving physical reference planes preserve total-depth derivatives and converge to independent graded-film optics? [Measurement](../docs/fdtd-variable-depth.md). |
| `fdtd_layer_smoothing.py` | Does C1 interface averaging improve thickness derivatives, and how much bias does its physical width add? [Measurement](../docs/fdtd-layer-smoothing.md). |
| `fdtd_layer_gradients.py` | How do internal-interface position and mesh refinement affect thickness-derivative error against a fixed-reference-plane film solution? [Measurement](../docs/fdtd-layer-gradients.md). |
| `fdtd_directional.py` | Does E/H direction separation isolate boundary returns from a slowly decaying near-Rayleigh mode? [Measurement](../docs/fdtd-directional.md). |
| `fdtd_pml_strength.py` | How do CPML strength, physical thickness and refinement affect returning near-Rayleigh waves after source narrowing? [Measurement](../docs/fdtd-pml-strength.md). |
| `fdtd_ringdown.py` | Does the remaining late-time signal behave like fixed damped modes or a cutoff-related algebraic tail? [Measurement](../docs/fdtd-ringdown.md). |
| `fdtd_pulse_notches.py` | Does suppressing a discrete cutoff in the source reduce finite-time scattering error across duration, grid and notch choices? [Measurement](../docs/fdtd-pulse-notches.md). |
| `fdtd_source_clearance.py` | How do fixed-source boundary clearance and fixed-boundary source movement separately affect near-cutoff scattering? [Measurement](../docs/fdtd-source-clearance.md). |
| `fdtd_lattice_reference.py` | How much near-cutoff scattering error comes from the Yee lattice and from finite CPML boundaries, before any time truncation? [Measurement](../docs/fdtd-lattice-reference.md). |
| `fdtd_lattice_time_check.py` | Does a shaped pulse with stronger, wider CPML approach the independently solved finite-grid harmonic limit? [Measurement](../docs/fdtd-lattice-reference.md). |
| `fdtd_transverse_resolution.py` | Which axis limits near-cutoff lattice accuracy, and can selective refinement reduce the required cells and updates? [Measurement](../docs/fdtd-transverse-resolution.md). |
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
| `fdtd_route_parity.py` | Do the two FDTD engines -- the legacy NumPy scene solver with its TFSF source, and the shared NumPy/JAX/Torch Yee engine with current sheets calibrated by a uniform-medium reference run, here on NumPy -- make the same error on the same scene? 36 shipped scenes, each engine against one converged reference, with grid, schedule and every material array matched and each unmatched confound listed; plus the four material scenes again with the CPML target, depth and run length strengthened together (`-pml16`), to see how the engine difference on grazing orders moves. Wall clock goes to a separate timing file. Write-up: [../docs/fdtd-route-parity.md](../docs/fdtd-route-parity.md). |
| `fdtd_install_smoke.py` | Whether a clean wheel installation computes 36 planar material/incidence combinations without optional backends. Run outside the checkout with the clean environment Python. |
| `fdtd_grid_benchmark.py` | What JIT compilation costs on the native Yee grid, and what warm execution saves against NumPy: one 32x8x8, 256-step complex update, the compile-and-first call timed apart from the warm median, with the two fields held within 1e-12 of each other. Timings belong to the host they ran on (see "Reading a timing table" below). Write-up: [../docs/fdtd.md](../docs/fdtd.md), "Streaming fields and compiled native grids". |
| `fdtd_slab_add_cost.py` | What a slab add -- one additive term writing a (3, 8, 8, 8) block into the field each step -- costs on NumPy, JAX and Torch against the dense full-volume alternative, on a stand-in Yee update with no engine imported: twelve interior slabs, twelve on zero faces, one at the corner, twelve of (3, 16, 16, 16), at N=64/96/128, plus the eager terms timed alone and one JAX slab swept over the {0, 8}^3 corners; whether XLA keeps `E.at[slab].add` in place inside a `lax.scan` (HLO op counts, buffer assignment, kernel loop bounds); and what Torch's in-place slab `add_` costs and does under autograd -- forward and backward timed apart against a `mark_dirty` Function and out-of-place pad, and the contract (a fresh field written before anything reads it reproduces the out-of-place gradient; a read view first raises); and, JAX-only (`--section scatter` runs it alone), three slab primitives -- the slice add, an index scatter-add with the four index arrays built once per term, and the same scatter on E reshaped flat inside the scan body -- on the interior s=8 and s=16 layouts and each moved off alignment, the final fields held together, and the scatter's reverse mode checked against the slice add's and both timed at every N with one term and with twelve, the transposed scan body's ops counted. CPU-only; the measured tables are in the docstring. |
| `fdtd_cpml_slab_cost.py` | Where a CPML face's psi should live, and what the face costs per step. A PML face stretches the derivative along its axis inside a 12-cell slab and carries four psi arrays there; psi can sit over the SLAB (gather the bare difference, advance psi, write the stretched value back) or over the FULL VOLUME (one multiply-add per psi, most of it on zeros, and both ends of an axis in one array). Both layouts on NumPy, JAX and Torch at N=64/96/128 float64, with one absorbing axis (two faces) and with all six, on the slab-add study's stand-in Yee kernel and with no engine imported: ms/step over a no-absorber baseline, the exact psi and index bytes of every layout, three ways of writing the slab layout on JAX (the slice add, an index scatter-add, and one 4-D scatter per face per curl carrying both curl components at once), the same face pair swept per axis, and the reverse mode of every layout at N=64 and 128 with the transposed scan body's op counts. CPU-only; the measured tables are in the docstring. |
| `fdtd_tfsf_oblique_leak.py` | Whether a TFSF box can take an OBLIQUE incident field from a 1-D auxiliary grid (the incident-field-array method: 3-D points projected onto the propagation direction, the aux field interpolated there). An empty box in vacuum on a plain-NumPy stand-in Yee grid with `wavebit.fd`'s update, the corrections derived from the stencil; the peak scattered field outside over the peak incident at 0/15/30/45/60 deg in the xz plane, 10/20/40 (and 80) cells per wavelength, s and p, for four aux variants -- plain (same dx and dt), matched dispersion, matched with cubic interpolation, and that with the grid's own discrete polarisation vectors -- plus band, Courant, box-size and off-plane arms and a sign mutant. Axis-aligned is exactly 0; 45 deg and the body diagonal are round-off with any matched aux (a frequency-independent `dx/sqrt(2)`, `dx/sqrt(3)` cell); every other direction leaks 2e-3 / 4e-4 / 1e-4 / 3e-5 at 10/20/40/80 cells with the best variant and 3-8e-2 at 10 with the plain one. Decided v1's TFSF: axis-aligned only, oblique refused by name. CPU-only; the measured tables are in the docstring. |
| `fdtd_cpml_evanescent.py` | What the CPML's two CFS constants (`KAPPA_MAX`, `ALPHA_RELATIVE` in `wavebit/fd/_lower/cpml.py`) buy on a source's evanescent near field, and what they cost on propagating waves. An `fd.Dipole` (Ez and Ex) 0.05/0.1/0.2/0.5 wavelengths from a 12-cell `z+` PML at 40 cells per wavelength, measured by the doubling method (the same run with the `z+` face moved out until nothing returns inside the window; a 40-cell-longer control reference differs by exactly 0), with the return at a probe halfway between the dipole and the layer, over `KAPPA_MAX` in (1, 2, 3, 5) x `ALPHA_RELATIVE` in (0, 0.5, 1.5, 3), only the `z+` face swept. The propagating table of `tests/test_fd_pml.py` (5 depths x 4 angles, its own helper) is re-measured at all 16 settings, plus its 12-cell row at 40 cells per wavelength. The near-field return is 1e-7 to 1e-6 of the field at the face at every admissible setting, far below the propagating 12-cell number (9.2e-5); against `alpha = 0.5`, `alpha = 1.5` helps only at 0.05 wavelengths (1.5x) and costs up to 60x from 0.2 on; `alpha = 3` loses 40-55x on the near field and 40-290x on the table. `(1, 0.5)` improves every propagating entry (12-cell row 0.43x, normal incidence 9.2e-5 to 1.7e-6) and the worst near-field return 5x, at 1.4-1.6x more near-field return at 0.05 wavelengths. The default `(2, 1.5)` is kept only under the rule that no near-field case may get worse. Report only; no constant moved. NumPy, CPU, ~14 min wall on 5 workers; the tables are in the docstring and `results/fdtd_cpml_evanescent.log`. |
| `fdtd_dipole_stamp.py` | Which stamp an `fd.Dipole` spreads its current with: tent or cubic B-spline. Both sum to 1 and reproduce linear functions, so the moment and centroid are exact for both; what differs is how much the radiated power depends on where `center` sits inside its cell. An `Ez` dipole in a 3-wavelength open vacuum box at 16 cells per wavelength, moved 0, 1/4 and 1/2 cell along the diagonal, radiation resistance from the source's own drive and work series at three frequencies of the band. The B-spline's position spread is 1.2e-4 at the band centre against the tent's 2.4e-2 (a `1/k^4` spectral tail against `1/k^2`), and it sits closer to the continuum `k^2/(6 pi)` (-3.5% against -8.8%, the rest being grid dispersion). `fd.Dipole` takes the B-spline. NumPy, CPU, seconds; the table is in the docstring and `results/fdtd_dipole_stamp.log`. |
| `fdtd_grazing_return.py` | Whether a grazing order still carries a CPML return now that `wavebit.fd` backs its PML with PEC, and what a scene must declare before such an order may be reported. An `fd.PlaneWave` in a 1 x 1 x nz Bloch cell leaving at band-centre cos theta 0.15 / 0.2 / 0.26 / 0.3 / 0.4 / 0.5 / 0.7 toward the `z-` PML, measured by the doubling method (a 25%-longer control reference differs by exactly 0), the return split from the incidence with the Yee kz at f0, over cells (12, 24, 36) x target (1e-8, 1e-12, 1e-16) x window (40, 160, 640 periods) at 20 and 40 cells per wavelength, plus 0.6 / 1.2 / 1.8 wavelengths of layer at 20 / 40 / 80. The return survives the PEC backing and converges with resolution, upward, to the layer's designed round trip `target ** (0.919 cos theta)` (1.2e-2 at cos 0.26 for the default 1e-8; 80 cpw lands at 0.69-1.01 of it), so a finer grid is worse on a grazing order, never better; cells buy down only the discretisation term. A separate window error from the band's Rayleigh point falls as T^-1.5 and swamps every layer at 40 periods (1.5e-2 at cos 0.26); the PlaneWave's hard-driven column mirrors the return back as incident. The rule, at tau = 1e-3: the default face refuses exit cos theta < 0.41; `PML(cells>=24, reflection=1e-12)` reaches 0.27 and `1e-16` 0.204, with `Run(periods>=160)`; below 0.204 refused outright. Report only; nothing in `wavebit/fd` changed. NumPy, CPU, 4400 s wall on 8 workers; the tables are in the docstring and `results/fdtd_grazing_return.log`. |
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
