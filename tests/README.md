# Verification suite

Everything whose expected answer can be stated before the run, in two tiers that are both
asserted: the default set, which CI runs on every push (`pytest tests -q`), and the
[research tier](#the-research-tier), which runs when asked for. There is no unasserted tier.

The test for whether something belongs here: **can you write the `assert` line first?** If
writing it would mean asserting the thing you are trying to find out, it is a study and belongs
in [`../studies/`](../studies) with a [`../docs/`](../docs) write-up instead. See
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Where the expected answers come from

Grouped by oracle, because that is what decides how much a green run is worth. A suite that only
compares the code to itself proves continuity, not correctness.

| Oracle | Files | What a pass means |
| --- | --- | --- |
| **Public Fourier origin** | `test_conv_matrices_accessor.py`, `_conv_frame.py` (independent test-side translation) | Closed-form coefficients and hand-built stacks use the public centre; round trips preserve every answer. |
| **Installed distribution records and lossless energy conservation** | `run_installed_smoke.py` | Does a non-editable installation outside the checkout contain matching metadata, all third-party license notices and material tables, and solve without importing an optional backend? |
| **Separately assembled translated physical equations** | `test_nvf_pair_centered_origin.py` | Centering the same periodic cell preserves every value and joint gap/material derivative slot, unequal Fourier orders and original geometry guards. |
| **Exact mesh reflection, polynomial identities and physical gap differences** | `test_nvf_pair_symmetric_mesh.py` | The explicit symmetric pair retains its full potential span, original contact quotient, reflection covariance and finite gap variation without projecting its response. |
| **Original nodal constraints, rational Laurent equations and Duffy integration** | `test_nvf_pair_galerkin.py` | A real constrained-gradient response retains finite contact/material derivatives, reciprocal passive energy and holomorphic material dependence without repairing its tensor. This does not establish optical convergence. |
| **Independent cardinal functions, original nodal equations and physical error integrals** | `test_nvf_polynomial_galerkin.py` | Higher polynomial fields preserve the complete finite tensor, contact/material derivatives and closed error bound without substituting an optical convergence claim. |
| **Independent periodic quotient and rational original equations** | `test_nvf_periodic_galerkin.py` | The periodic rectangle retains its full opening-potential space, finite contact JVP and physical material properties. The optical trace error is measured separately. |
| **Independent continuous physical-field energy and unrestricted normal trace** | `test_nvf_seam_trace.py` | Continuous positive-gap lifts supply the exact normal contact derivative in the retained k_y=0 sector. This component does not establish a full transverse closure or optical convergence. |
| **Independent original unscaled nodal equations plus continuous seam fields** | `test_nvf_seam_galerkin.py` | All transverse components share the enlarged physical field space, its positive-gap JVP and exact contact energy limit. This does not certify optical convergence or ordinary NVF routing. |
| **Independent clipped triangles, raw nodal variables and real sine/cosine gap traces** | `test_nvf_pair_seam_galerkin.py` | The complete internal-pair tensor and its gap/material variations agree with a separately assembled physical field space; optical convergence is not implied. |
| **Corner gradients, physical variations and exact contact limits** | `test_nvf_pair_corner_galerkin.py` | Real outer-corner and optional junction fields preserve the original constraints, material linearity, length units, finite gap variations and independent energy rejection. |
| **Independent sector laws, spatial differences and full physical guards** | `test_nvf_pair_transmission.py` | Fixed transmission directions preserve interface potential/flux continuity, complete transported derivatives, contact reflection, length units and independent positive-gap/material variations. |
| **Independent Decimal powers, potential differences and higher-degree physical controls** | `test_nvf_transmission_common_precision.py` | Higher-degree transmission modes retain full common gradients, uniform limits, family bounds and physical material/opening variations. |
| **Independent Decimal moving transmission potentials** | `test_nvf_transmission_moving_precision.py` | Complete common gradients and their moving-sample variations retain every polynomial and complex transmission component, including refined charts. |
| **Independent nested transmission cutoffs** | `test_nvf_transmission_cutoff_precision.py` | Every original and added cutoff potential retains its complete gradient, moving variation and physical guards. |
| **Independent Decimal physical chart inverses and gap differences** | `test_nvf_contact_map_precision.py` | Complete common contact maps retain physical gradients and opening limits on non-dyadic original and refined meshes. |
| **Complete contact normal fields** | `test_nvf_contact_normal_precision.py` | Do unchanged normal polynomial and seam fields match independent Decimal derivatives and opening limits while preserving the full transverse physical response? |
| **Independent bulk gradients and distributional interface jumps** | `test_nvf_contact_normal_mean.py` | Does periodic potential integration recover every normal Fourier mean, including zero modes and translated non-square cells, at the complete-field bound? |
| **Independent Decimal complex elimination** | `test_nvf_complete_solve.py` | Full high/low pivots preserve every solution column, low-word nonsingularity and exact singular rejection without a rank threshold. |
| **Independently sampled complete physical QR** | `test_nvf_contact_geometric_frame.py` | Retained elimination and integrated full contact metrics preserve the entire quintic/sextic tensor and joint material/opening JVP while retaining all physical energy and reflection checks. |
| **Independent Decimal complex exponentials** | `test_nvf_complex_exponential.py` | Complete real-factor exponentials retain radial scale, quadrant boundaries, low input parts and finite-input checks. |
| **Independent Decimal positive physical potentials** | `test_nvf_positive_field_precision.py` | Complete polynomial and disappearing gradients retain moving-field accuracy through tiny positive gaps before a whole response is solved. |
| **Independent Decimal full metric factors** | `test_nvf_positive_metric_precision.py` | Complete geometric factors retain all positive pivots and reject nonpositive ones without shifts or field removal. |
| **Independent corruption of positive tensors and physical fields** | `test_nvf_positive_energy_precision.py` | Physical energy and retained Fourier integrals reject altered tensors, derivatives or verifier-only coefficients under the original bounds. |
| **Independent complete transmission parameters** | `test_nvf_transmission_mode_precision.py` | Complete roots and states satisfy independent Decimal characteristic and transfer equations, including weak contrasts and all reflected field branches. |
| **Independent near-homogeneous transmission roots** | `test_nvf_transmission_near_homogeneous.py` | Nearly equal reference coefficients retain resolved noninteger roots and closed modes while preserving the existing root branch choices. |
| **Homogeneous angular powers, independent variations and nested real energy** | `test_nvf_pair_angular_fields.py` | Angular refinement retains the complete positive and contact family, including uniform media, gap/material derivatives and length units. |
| **Independent triangle moments and uniform material variations** | `test_nvf_pair_angular_quadrature.py` | Separate radial and angular integration orders preserve exact polynomial integrals, default sampling and full contact/positive material responses. |
| **Independent Decimal potentials, physical gradients and moving gradients** | `test_nvf_pair_oscillatory_coordinates.py` | The complete symmetric oscillatory coordinates remain accurate through strong cancellation, non-dyadic simplex coordinates and singular corners, including bounded table storage. |
| **Independent Decimal common potentials and original contact equations** | `test_nvf_pair_common_precision.py` | Complete real root-20 contact coordinates retain non-dyadic and singular-point accuracy, the full joint metal derivative, quadrature agreement and uniform-material identities. |
| **Independent Decimal cancelling matrix products** | `test_nvf_pair_compensated_product.py` | Complete high/low contact products and structured Cartesian Fourier fields retain every cross term and the original precision bound. |
| **Independent three-word coordinate arithmetic** | `test_nvf_extended_precision.py` | Complete sums, products, division, logarithms and complex powers preserve strongly cancelling fields before two-word assembly. |
| **Independent corruption of contact tensors and physical coefficients** | `test_nvf_contact_energy_coordinates.py` | Complete pointwise energy, Fourier mean and physical mean variation reject changed tensors or coordinates, including an energy-invisible skew variation. |
| **Independent Decimal divided differences and rectangle Fourier integrals** | `test_nvf_contact_fourier_precision.py` | Complete free moments retain scalar, repeated-phase, non-square-order, refined-mesh and joint material/opening accuracy using the physical vertex measures and unit-cell Fourier identity. |
| **Exact contact integration moments and Decimal Fourier phases** | `test_nvf_pair_contact_quadrature.py` | Compensated Gaussian rules and every triangle partition preserve analytical moments, and the complete Fourier phase and its variation agree with an independent Decimal reference. |
| **Independent complex powers and complete real physical variations** | `test_nvf_pair_oscillatory_radial.py` | A fixed real logarithmic cosine/sine pair preserves both radial directions, field derivatives and the complete finite contact/positive response. |
| **Independent exact powers and complete radial variations** | `test_nvf_pair_radial_fields.py` | Expanded real exponents retain their declared span and derivatives without nearby-power cancellation. |
| **Exact balanced Schur variations** | `test_nvf_pair_balanced_response.py` | Original states and tensor derivatives match exact rational equations and independent complex variations across small gap scales. |
| **Independent mixed equations and pointwise physical weak integrals** | `test_nvf_pair_positive_refinement.py` | Complete value and derivative refinement resolves large-coefficient roundoff while preserving normalized thin rows and every original physical guard. |
| **Independent cancelling matrix sums** | `test_nvf_pair_residual_arithmetic.py` | Can complete real and complex products resolve small constraint defects with cancellation, anticorrelated sample scales, structural zero padding and extreme finite inputs? |
| **Analytic beta-function energies and exact moving polynomial moments** | `test_nvf_pair_junction_quadrature.py` | Both singular thin endpoints and every profile-region transport derivative remain resolved through small positive gaps and mesh refinement. |
| **Decimal arithmetic and independent cardinal polynomials** | `test_nvf_polynomial_trace.py` | Small trigonometric remainders retain spatial derivative accuracy, and their subtracted bubbles belong to the original nodal space. |
| **Independent physical residual integrals, real energy identity and exact laminated tensors** | `test_nvf_galerkin_error.py` | Closed constitutive error bounds contain independent exact errors, preserve units and drive refinement with explicit exhaustion. Maxwell and opening-derivative accuracy are outside this bound. |
| **Independent sampled admissible flux projection** | `test_nvf_equilibrated_flux.py` | Does exact weighted flux minimization match physical QR integrals, tighten the bound, preserve the electric tensor and retain all original safeguards? |
| **Whole-state contact elimination and physical QR controls** | `test_nvf_contact_solver.py` | Can retained elimination avoid unnecessary QR while preserving every physical guard and replacing a rejected candidate in full? |
| **Independent integration of two normalized physical field maps** | `test_nvf_paired_frame.py` | The paired energy preserves passivity and reciprocity, exact uniform/laminated limits and finite NumPy/JAX gap/material derivatives. Failed raw guards remain failures, and original defects remain visible. This does not certify optical convergence. |
| **Independent weak-system variation and triangle quadrature** | `test_nvf_cartesian_cell.py` | Concrete orientation planning matches equivalent vertex charts; frozen P1 replay preserves physical geometry, material transforms and finite derivatives under NumPy/JAX. This does not establish general geometry or optical convergence. |
| **Independent pair weak equations, moving-triangle integration and closure variation** | `test_nvf_cartesian_pair.py` | Frozen positive-gap/contact charts preserve two physical equal-height rectangles, exact Fourier sets and finite NumPy/JAX derivatives. References are recomputed with declared per-case quadrature; ordinary contact JVP excludes gap opening, and optical convergence remains unqualified. |
| **Independent rational Laurent weak limit and moving-simplex integrals** | `test_nvf_cartesian_pair_opening.py` | The separate NumPy finite right-opening response agrees with original nodal equations and differentiated closure solves at refinements 0/1, with exact label covariance and per-length units; ordinary contact JVP and optical scope remain unchanged. |
| **Independent original nodal and moving-simplex equations** | `test_nvf_cartesian_pair_gap.py` | The explicit NumPy finite gap family agrees at contact and positive gaps, with analytical derivatives, per-unit scaling and exact selected closures; it establishes no optical stopping rule. |
| **Independent quadratic nodal equations and Bernstein simplex integrals** | `test_nvf_pair_quadratic.py` | Six-node potentials retain finite contact/positive-gap values and derivatives, exact polynomial transforms, physical units and passive uniform controls. This does not establish optical convergence. |
| **Independent pointwise frame energy, exact laminated integrals and real standing waves** | `test_nvf_pair_normal_frame.py` | Material-weighted fitted quadrature retains a coercive frame, bounded NumPy field storage, exact uniform/laminated limits and finite NumPy/JAX gap/material derivatives. A real-basis transpose check verifies reported reciprocity defects. The normal extension is shared; complex reciprocity and optical convergence remain unqualified. |
| **Independent finite gap automatic differentiation** | `test_nvf_cartesian_pair_gap_autodiff.py` | Original nodal/Laurent and Duffy references check gap-only eager/JIT/reverse derivatives, exact selected closures and invalid replay without widening ordinary contact tangents. |
| **Independent energy-closure variation and public geometry mapping** | `test_nvf_cartesian_rectangle_api.py` | Ordinary rectangle inputs retain physical geometry, scalar materials and live derivatives through an explicitly planned Cartesian NVF builder, including exact unequal/zero Fourier sets. |
| **Independent pair closure variation and native public geometry mapping** | `test_nvf_cartesian_rectangle_pair_api.py` | Frozen input indices preserve the two physical rectangles and material association; shared height/y/contact tangents, exact Fourier sets, uniform optical controls and package-only optional-backend behavior remain explicit. |
| **An outside solver** | `test_reticolo_golden.py`, `test_oblique_lattice.py` (published Li 1997 table) | The strongest evidence here: a different group's code, in a different language. RETICOLO is the only reference that validates individual diffraction *orders* rather than totals. |
| **Physics** | `test_lossy_media.py`, `test_stability.py`, `test_fp_warnings.py`, `test_polarization.py` (circular states against the gyrotropic-slab Airy answer), `test_smm_field_labels.py` (the plane wave inside a reflectionless uniform layer at a complex angle, every path's field), `test_slot_order.py` (every private builder splatted into `_solve_conv`), energy checks throughout | R+T = 1 on a lossless stack, Airy recursion on uniform layers. Independent of how we implement it. |
| **Exact interface normals and material continuity** | `test_nvf_seam_materials.py` | Distinct equal-modulus materials retain the exact seam normal and a Hermitian rank-one projector; coefficient and optical changes shrink with the material perturbation, listings agree, and JAX/Torch gradients match secants. The study's opening creates a positive geometric gap. This does not certify optical convergence. |
| **Painted material limits and geometric invariants** | `test_painted_material_continuity.py` | Real, absorbing and metallic equality ramps have no fixed step. Exact straight-interface normals, rank-one projectors, physical rotation/reflection, passivity, chunking, and JAX/Torch material derivatives constrain the contrast envelope independently of its construction. |
| **Passive constitutive forms and independent spectral normalization** | `test_passive_nvf.py` | An explicit anticommutator counterexample, uniform negative-real/absorbing materials, positive storage, rotated 1D rules, an independent inverse square root, and JAX/Torch derivatives constrain the experimental passive closure. These are not optical convergence gates. |
| **Internal consistency** | `test_equivalence.py`, `test_solve_1d_core.py`, `test_solve_2d_core.py`, `test_field_core.py`, `test_parametric.py` (the parametric path against a fine midpoint DFT, and against the raster it replaces) | Three formulations and three backends must agree. Catches drift, not a shared mistake. |
| **Analytic gradients vs finite differences** | `test_gradients.py`, `test_shape_position_gradients.py`, `test_eig_grad.py`, `test_optimization_smoke.py` | The derivative is right, not just the value — and the optimizer loop makes progress. |
| **Reverse mode and finite differences, for forward mode** | `test_forward_mode.py` | `jax.jvp`, `jax.jacfwd` and `torch.autograd.forward_ad` through `solve()` return the reverse-mode derivative and a central difference's on the paths it lists, the cell included; a second order where a modal operator moves under both derivatives, and `torch.func`, is refused by name. |
| **Independent pixel derivatives under integrated compression** | `test_integrated_raster_gradients.py` | Equal-valued pixels keep their individual permittivity/permeability derivatives against fresh NumPy differences under JAX grad, both JIT orders, PyTorch, and optimization. |
| **Eager geometry refusals and freshly prepared NumPy fields** | `test_field_geometry_precheck.py` | Compiled field entry rejects stale scalar/tensor shape plans and tensor raster interfaces before compilation, while valid geometry/material sweeps and tensor field derivatives remain correct. |
| **Independent NumPy derivatives and preserved solver declarations** | `test_grad_state.py` | JAX gradient queries release temporary solve state, preserve dispersion and incidence declarations on success/failure, and retain field-loss derivatives without leaking tracers. |
| **Frozen goldens** | `test_golden_legacy_parity.py`, `test_modeler_core.py` | Continuity only. These are the previous implementation's own output, so they pin *change*, never correctness. `test_factorization_core.py` left this row on 2026-08-18 — its golden was retired and every check it carried is now definitional (see `data/README.md`). |
| **Wavelength-dependent material assignments** | `test_dispersive_material.py` | Do table-backed regions update during sweeps, survive serialization and JAX trees, and differentiate consistently with an independent Airy film? |
| **The solver's own refusals** | `test_manifest.py` | The published capability manifest describes the solver that exists: each `structure` spelling selects the path it claims, the material quantity it declares is the one the front door actually takes and every retired spelling it lists still raises by name, every option the manifest withholds is refused at build time and every option it offers builds, and each claim cites a file that still exists. See [`../docs/capability-manifest.md`](../docs/capability-manifest.md). |

## The research tier

2,715 of the 9,737 tests here are marked `research`. Their subject is NVF research code: the
experimental conforming module `wavebit/_core/conforming/`, reached directly or through the
`studies/` models built on it; the NVF study instruments in `studies/` (energy closure, cell
bounds, the gap, mirror and FEM references); and the passive closure kernel in
`wavebit/_core/passive_nvf.py`, which nothing in the package calls. They are asserted like
everything else, which is why they stay in this directory -- the routing rule in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) sends anything with an `assert` line here.

They are left out of the default run for two reasons: they study that experimental code rather
than the ordinary solver, and they are where the suite's time goes. Run one file at a time on
the development laptop, 14 of these files took just under four hours between them, two of them
over an hour each. That is not a claim that nothing else can move them. Many drive the ordinary
engine: the conforming module builds its matrices as `ConvMatrices` (often through
`ConvMatrices.from_energy`) and checks its layers with `parametric`'s spec checks, and the tests
and the `studies/nvf_*` models they import build `wavebit.rcwa` solvers and hand those matrices
to `_solve_conv` on the kernels' side, or through public doors that restate the frame.

**When it runs.** The tier is a development aid for the research code, not a gate on the rest
of the package. It runs while that code is being worked on: `wavebit/_core/conforming/`,
`wavebit/_core/passive_nvf.py`, a `studies/nvf_*` model, or a research test. A change to the
product code runs the default set, which is also all CI runs, and lands on that. Because many
research tests drive the ordinary engine, such a change can break one. That is accepted: the
test is repaired when research work next runs the tier.

```shell
pytest tests -q                                  # the default set, 7,022 tests: what CI runs
pytest -m research tests -q                      # the research tier alone, 2,715 tests
pytest -m "research or not research" tests -q    # all 9,737
pytest tests/test_nvf_pair_galerkin.py -q        # a file named on the command line runs whole
```

The rule is in `tests/conftest.py`, and `pyproject.toml` registers the marker. An explicit `-m`
always decides on its own; naming a file counts as asking for it, so the reproduction commands
in the `docs/` write-ups still run what they name.

**What stays in the default set**, although it touches the same code:

- The contract of the public conforming classes -- `ConformingNVF`, `AffineProfileNVF`,
  `CartesianRectangleNVF`, `CartesianRectanglePairNVF`: refused options and inputs, plan and
  schema checks, invalid-replay poisoning under `jit`, the export reaching the engine on the
  optional backends, and the installed package not needing `studies/`. Nine files carry it and
  mark their numerical tests one by one, with no module-level mark: `test_conforming_nvf.py`,
  `test_conforming_affine_profile.py`, `test_conforming_common_loss.py`,
  `test_conforming_lattice_rebasing.py`, `test_conforming_layer_axes.py`,
  `test_conforming_rectangular_lattice.py`, `test_conforming_separated_rectangles.py`,
  `test_nvf_cartesian_rectangle_api.py` and `test_nvf_cartesian_rectangle_pair_api.py`. 111 of
  their tests stay, and they run in about 30 seconds between them. The one exception among the
  `test_conforming_*` files is `test_conforming_complex_material.py`, which is numerical
  throughout and marked `research` as a whole.
- The contract of the two exported auxiliary pieces, `CartesianPairOpeningPlan` and
  `CartesianPairTransmissionGalerkinGapCell`: the plan's refusals (a `bool` shift, a refinement
  out of range) and its immutability in `test_nvf_cartesian_pair_opening.py`, the refinement
  refusals again in `test_nvf_pair_refinement.py`, and the cell's option refusals in
  `test_nvf_transmission_common_precision.py` and `test_nvf_transmission_cutoff_precision.py`:
  five unmarked tests, 16 cases in under a second. The rest of those four files is marked test
  by test.
- Anything whose claim is about the ordinary solver, even when it borrows a conforming cell to
  make it: `test_nvf_energy_builder.py` (`ConvMatrices.from_energy` and the engine),
  `test_cell_shape_gate.py` (every oblique-capable path), and the two tests in
  `test_nvf_pair_transmission.py` that hold the method-selection example to leaving the
  auxiliary cell off.
- The ordinary NVF, whatever its file is called: raster and shape normal fields, projectors,
  the seam contrast, Gamma, and the contact and resonance probes.

**A new test of research code** carries `pytestmark = pytest.mark.research` after its imports.
A file that has to run under `python -m unittest` alone, like the NGSolve reference files,
guards that import the way they do.

## Goldens

`golden_*.npz` are frozen outputs; each has a `gen_goldens_*.py` beside it that regenerates it.
They are checked in so the tests run anywhere.

**Regenerating a golden is not a way to make a failing test pass.** A golden captured the
behaviour of an implementation that was believed correct at the time; rewriting it converts a
detected regression into an accepted one. Regenerate only to add cases, or after deliberately
changing what is compared — and say which in the commit message.

`gen_goldens_reticolo.py` is the exception that needs Octave and a RETICOLO tree; everything
that consumes its output needs neither, which is the point.
