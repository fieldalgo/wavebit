# Verification suite

Everything whose expected answer can be stated before the run. CI runs all of it on every push
(`pytest tests -q`); there is no second, unasserted tier.

The test for whether something belongs here: **can you write the `assert` line first?** If
writing it would mean asserting the thing you are trying to find out, it is a study and belongs
in [`../studies/`](../studies) with a [`../docs/`](../docs) write-up instead. See
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Where the expected answers come from

Grouped by oracle, because that is what decides how much a green run is worth. A suite that only
compares the code to itself proves continuity, not correctness.

| Oracle | Files | What a pass means |
| --- | --- | --- |
| **Layer-aligned meshes** | `test_fdtd_layer_alignment.py` | Do fitted layer boundaries preserve geometry, bottom records and pole gradients, and converge to independent multilayer optics? |
| **Scene pulse width** | `test_fdtd_pulse_width.py` | Does bandwidth control preserve defaults, physical scaling and backend material gradients, reject unresolvable sources, and settle oblique metal amplitudes toward Fresnel/Airy? |
| **Mixed dispersive metals** | `test_fdtd_mixed_metal.py`, `references/test_native_material_external.py` | Do low-loss Drude/Lorentz scenes match an independent discrete boundary problem and Airy refinement, preserve all six material gradients, and agree with frozen Meep fields at matched interface sampling? |
| **An outside solver** | `test_reticolo_golden.py`, `test_oblique_lattice.py` (published Li 1997 table) | The strongest evidence here: a different group's code, in a different language. RETICOLO is the only reference that validates individual diffraction *orders* rather than totals. |
| **Region records** | `test_fdtd_record_region.py`, `test_fdtd_scene_record_bounds.py` | Do cropped histories preserve exact Yee modes, coordinates, retained array size and material gradients? |
| **DFT sampling** | `test_fdtd_dft_sampling.py` | Do sampled spectra preserve exact-carrier quadrature, staggered phases, prefix weights and backend gradients? |
| **Pulse-decay diagnostics** | `test_fdtd_material_decay.py`, `test_fdtd_convergence.py`, `test_fdtd_cavity_convergence.py` | Does the reported decay and DFT tail match an exact conductive-field solution, preserve JAX/Torch sensitivities, and reject absent or persistent signals? |
| **Physics** | `test_lossy_media.py`, `test_stability.py`, `test_fp_warnings.py`, `test_polarization.py` (circular states against the gyrotropic-slab Airy answer), `test_slot_order.py` (every builder splatted into `solve_conv`), energy checks throughout | R+T = 1 on a lossless stack, Airy recursion on uniform layers. Independent of how we implement it. |
| **Internal consistency** | `test_equivalence.py`, `test_solve_1d_core.py`, `test_solve_2d_core.py`, `test_field_core.py`, `test_parametric.py` (the parametric path against a fine midpoint DFT, and against the raster it replaces) | Three formulations and three backends must agree. Catches drift, not a shared mistake. |
| **Analytic gradients vs finite differences** | `test_gradients.py`, `test_shape_position_gradients.py`, `test_eig_grad.py`, `test_optimization_smoke.py` | The derivative is right, not just the value — and the optimizer loop makes progress. |
| **Frozen goldens** | `test_golden_legacy_parity.py`, `test_modeler_core.py` | Continuity only. These are the previous implementation's own output, so they pin *change*, never correctness. `test_factorization_core.py` left this row on 2026-08-18 — its golden was retired and every check it carried is now definitional (see `data/README.md`). |
| **The solver's own refusals** | `test_manifest.py` | The published capability manifest describes the solver that exists: each `structure` spelling selects the path it claims, the material quantity it declares is the one the front door actually takes and every retired spelling it lists still raises by name, every option the manifest withholds is refused at build time and every option it offers builds, and each claim cites a file that still exists. See [`../docs/capability-manifest.md`](../docs/capability-manifest.md). |

## Goldens

`golden_*.npz` are frozen outputs; each has a `gen_goldens_*.py` beside it that regenerates it.
They are checked in so the tests run anywhere.

**Regenerating a golden is not a way to make a failing test pass.** A golden captured the
behaviour of an implementation that was believed correct at the time; rewriting it converts a
detected regression into an accepted one. Regenerate only to add cases, or after deliberately
changing what is compared — and say which in the commit message.

`gen_goldens_reticolo.py` is the exception that needs Octave and a RETICOLO tree; everything
that consumes its output needs neither, which is the point.
