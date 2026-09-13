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
| **Independent weak-system variation and triangle quadrature** | `test_nvf_cartesian_cell.py` | Concrete orientation planning matches equivalent vertex charts; frozen P1 replay preserves physical geometry, material transforms and finite derivatives under NumPy/JAX. This does not establish general geometry or optical convergence. |
| **Independent pair weak equations, moving-triangle integration and closure variation** | `test_nvf_cartesian_pair.py` | Frozen positive-gap/contact charts preserve two physical equal-height rectangles, exact Fourier sets and finite NumPy/JAX derivatives. References are recomputed with declared per-case quadrature; ordinary contact JVP excludes gap opening, and optical convergence remains unqualified. |
| **Independent rational Laurent weak limit and moving-simplex integrals** | `test_nvf_cartesian_pair_opening.py` | The separate NumPy finite right-opening response agrees with original nodal equations and differentiated closure solves at refinements 0/1, with exact label covariance and per-length units; ordinary contact JVP and optical scope remain unchanged. |
| **Independent original nodal and moving-simplex equations** | `test_nvf_cartesian_pair_gap.py` | The explicit NumPy finite gap family agrees at contact and positive gaps, with analytical derivatives, per-unit scaling and exact selected closures; it establishes no optical stopping rule. |
| **Independent finite gap automatic differentiation** | `test_nvf_cartesian_pair_gap_autodiff.py` | Original nodal/Laurent and Duffy references check gap-only eager/JIT/reverse derivatives, exact selected closures and invalid replay without widening ordinary contact tangents. |
| **Independent energy-closure variation and public geometry mapping** | `test_nvf_cartesian_rectangle_api.py` | Ordinary rectangle inputs retain physical geometry, scalar materials and live derivatives through an explicitly planned Cartesian NVF builder, including exact unequal/zero Fourier sets. |
| **Independent pair closure variation and native public geometry mapping** | `test_nvf_cartesian_rectangle_pair_api.py` | Frozen input indices preserve the two physical rectangles and material association; shared height/y/contact tangents, exact Fourier sets, uniform optical controls and package-only optional-backend behavior remain explicit. |
| **An outside solver** | `test_reticolo_golden.py`, `test_oblique_lattice.py` (published Li 1997 table) | The strongest evidence here: a different group's code, in a different language. RETICOLO is the only reference that validates individual diffraction *orders* rather than totals. |
| **Physics** | `test_lossy_media.py`, `test_stability.py`, `test_fp_warnings.py`, `test_polarization.py` (circular states against the gyrotropic-slab Airy answer), `test_slot_order.py` (every builder splatted into `solve_conv`), energy checks throughout | R+T = 1 on a lossless stack, Airy recursion on uniform layers. Independent of how we implement it. |
| **Exact interface normals and material continuity** | `test_nvf_seam_materials.py` | Distinct equal-modulus materials retain the exact seam normal and a Hermitian rank-one projector; coefficient and optical changes shrink with the material perturbation, listings agree, and JAX/Torch gradients match secants. The study's opening creates a positive geometric gap. This does not certify optical convergence. |
| **Painted material limits and geometric invariants** | `test_painted_material_continuity.py` | Real, absorbing and metallic equality ramps have no fixed step. Exact straight-interface normals, rank-one projectors, physical rotation/reflection, passivity, chunking, and JAX/Torch material derivatives constrain the contrast envelope independently of its construction. |
| **Passive constitutive forms and independent spectral normalization** | `test_passive_nvf.py` | An explicit anticommutator counterexample, uniform negative-real/absorbing materials, positive storage, rotated 1D rules, an independent inverse square root, and JAX/Torch derivatives constrain the experimental passive closure. These are not optical convergence gates. |
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
