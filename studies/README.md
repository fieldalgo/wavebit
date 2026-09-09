# Studies

Open questions, answered once and written down. Each script here exists because somebody did
not know the answer; the table below links the write-up where the answer was recorded and what
it decided. Each is standalone and runs from the repo root.

**This is not a performance regression suite.** If you arrived expecting `asv continuous` and a
dashboard, there is none: these run by hand, they run rarely, and a run that does not end in a
`docs/` change was wasted. Comparisons with established external golden answers live in
[`../tests/references/`](../tests/references). An independent calculation whose convergence
is still being investigated belongs here until a reference value is established.

| Script | Question it answers |
| --- | --- |
| `nvf_harmonic_energy.py` | Can a material-defined electrostatic basis and an energy Gram closure remove artificial seams while retaining contact derivatives? Split equality and lossless energy pass; inner and outer convergence remain open. [Investigation](../docs/nvf-harmonic-energy.md). |
| `nvf_harmonic_energy_jax.py` | Are the finite candidate's derivatives correct under JIT, and physically correct at contact? P controls pass; the expanded s contact derivative fails despite agreement with its own secants. A separate 6 ppm eigenvalue-gradient bias is also recorded. [Measurements](../docs/nvf-cell-corrector.md). |
| `nvf_cell_bounds.py` | How much static cell uncertainty remains after CG converges? Primal/complementary enclosures separate it from optical error; also consumes independently projected fields with a [lower-memory full-space closure](../docs/nvf-energy-closure.md#full-space-projected-fields). [Investigation](../docs/nvf-cell-corrector.md). |
| `nvf_cell_corrector.py` | Can a conforming 2D auxiliary solve reduce cell error without a permittivity raster? Optional NGSolve export narrows the static enclosure substantially; moving-contact derivatives remain unresolved. [Investigation](../docs/nvf-cell-corrector.md). |
| `nvf_thin_gap_energy.py` | Can a conforming potential jump retain the vanishing gap's normal field in the forward energy? Both contact polarization gates now pass on the rectangular control, with JIT geometry/material derivatives. General geometry and partition invariance remain open. [Derivation and limits](../docs/nvf-thin-gap-energy.md). |
| `nvf_gap_optics.py` | How do the historical fixed-wavelength optical readouts reuse the library's conforming cells without becoming a production dependency? |
| `nvf_partitioned_gap_energy.py` | Can that forward gap model also preserve material partitions? Material-jump integration combines split equality, live material derivatives and both contact gates on translated horizontal rectangle tilings. An independent moving real-seam derivative still fails at mesh nodes. [Investigation](../docs/nvf-partitioned-gap-energy.md). |
| `nvf_harmonic_coordinates.py` | Can a material-defined coordinate carry a moving real field jump on that fixed mesh? The closed mixed-bar static and optical derivative gates now pass, retaining partition/contact checks. A smaller positive-gap mesh-node sensitivity discrepancy remains. [Derivation and limits](../docs/nvf-harmonic-coordinates.md). |
| `nvf_continuous_flux.py` | Can the same body flux also remain continuous across artificial mesh nodes? An integrated flux basis passes the previous positive-gap seam gate and retains partition, contact and material derivatives. General geometry and optical convergence remain open. [Derivation and measurements](../docs/nvf-continuous-flux.md). |
| `nvf_gap_convergence.py` | How much of the remaining rectangular error comes from the auxiliary cell and how much from outer Fourier truncation? Hierarchical gap bubbles and directed refinement improve the cell; directional comparisons reveal opposing x/y errors even at outer order 24. [Comparison](../docs/nvf-gap-convergence.md), [directional follow-up](../docs/nvf-directional-convergence.md). |
| `nvf_interface_resolution.py` | Can a material-defined mesh resolve real seams without changing under redundant partitions or losing derivatives? Smooth density grading narrows the tested static enclosure by about 64% at the same dimension and improves the optical cell comparison. Outer convergence remains open. [Derivation and measurements](../docs/nvf-interface-resolution.md). |
| `nvf_fourier_orders.py` | Which Fourier direction needs more resolution? Shared scalar/pair order handling lets the convergence instrument vary x and y independently while preserving the physical problem. Body y refinement also improves the cell comparison with fewer unknowns. [Directional measurements](../docs/nvf-directional-convergence.md). |
| `nvf_polynomial_body.py` | Can higher-degree body potentials reduce the remaining cell error without changing contact traces? Conforming y bubbles narrow the tested static enclosure at the same dimension and improve the optical cell comparison, with independent integration and JIT/contact gates. The cell also exports standard matrices to the Wavebit engine. [Implementation and measurements](../docs/nvf-polynomial-body.md), [engine integration](../docs/nvf-energy-engine.md). |
| `nvf_polynomial_flux.py` | Does higher-degree continuous x flux improve accuracy at equal dimension? The implemented degree-2/3 fluxes pass independent integration and derivative checks, but refining the original linear flux gives tighter static enclosures at equal dimension. A higher-resolution original-flux setting further improves the optical cell comparison at increased cost. [Implementation and decision](../docs/nvf-polynomial-flux.md). |
| `nvf_mirror_convergence.py` | Can exact mirror sectors reduce the cost of the same NVF closure? Square and four-corner comparisons reach order 48; targeted p-x/s-y checks extend to 56/64. The latest s increment falls below 1e-5, but its wider window remains above it. Direct modal storage and interface panels reduce measured full p-sector order-32 memory by 33%, retaining every mode. [Convergence](../docs/nvf-mirror-convergence.md), [modal memory and controls](../docs/nvf-mirror-modal-memory.md). |
| `nvf_energy_closure.py` | Can the same NumPy energy closure avoid an explicit inverse? One LU factorization and two solves retain complex/non-Hermitian behavior; tiled mirror assembly and guards reduce measured order-32 closure peak memory from 2.24 to .99 GB. [Algebra, ownership and controls](../docs/nvf-energy-closure.md). |
| `results/nvf_rcwa_memory.log` | Can the production 2D runtime release dense matrices earlier while preserving its results and field state? Before/after NumPy allocation and resident-memory observations, followed by alternating timing comparisons of diagonal scaling and direct boundary solves, for the fixed affine fixture. [Memory scope](../docs/what-a-solve-answers.md#9-releasing-transient-2d-matrices-after-their-last-use), [diagonal arithmetic](../docs/what-a-solve-answers.md#10-keeping-diagonal-actions-out-of-dense-matrix-multiplication), [boundary solve](../docs/what-a-solve-answers.md#11-solving-only-the-requested-boundary-columns). |
| `nvf_mirror_cartesian.py` | Does the independently factorized Cartesian control show the same high-order drift? Its parity restriction matches the unreduced interval rules; the order-40 comparison still cannot certify optical convergence. This is a separate control, not an NVF fallback. [Comparison](../docs/nvf-mirror-convergence.md#independent-cartesian-control). |
| `nvf_scattering_doubling.py` | Can differentiable propagation avoid the recorded eigenvector-gradient bias? A bounded short-slice series followed by scattering doubling passes the original material gate and exact layer-graze controls. An opt-in polynomial-body adapter retains JIT gap, seam, material and contact checks. [Derivation and limits](../docs/nvf-scattering-doubling.md). |
| `nvf_mirror_doubling.py` | Can the same eigenvector-free propagation use mirror sectors under JAX? The combined live closure and propagation pass contact/graze/derivative controls, match modal values through order 24 and reduce the measured order-8 warm value-plus-material-derivative time by a factor of 3.75. The modal path remains faster for the tested NumPy forward sweep. [Implementation and measurements](../docs/nvf-mirror-doubling.md). |
| `nvf_posed_gap.py` | Can the conforming NVF model retain live translation, whole-lattice rotation and incidence derivatives? Complex phase/covariance and ten-parameter derivative controls pass. A localized version of the same flux basis enables finer grading and passes the mixed-material contact gate on the stated fixture. [Pose and controls](../docs/nvf-posed-gap.md), [stable cell coordinates](../docs/nvf-local-flux.md). |
| `nvf_contact_sweep.py` | Does that contact gate persist across real material profiles, angles and outer orders? A common maximal field projection separates outer order from cell changes. Multiple seams require more x resolution; coarse failures remain recorded. [Controls and refinement](../docs/nvf-local-flux.md). |
| `nvf_derivative_reference.py` | Is a tiny finite-difference step a resolved derivative reference? Two Richardson estimates must agree within a bounded window selected without consulting AD; otherwise the check fails. [Roundoff measurements and criterion](../docs/nvf-local-flux.md). |
| `nvf_gap_reference.py` | Can an independently fitted 2D auxiliary field check the mixed gap? Optional NGSolve primal/reciprocal solves give static enclosures with verified Fourier quadrature, including affine physical gradients. This is not a 3D Maxwell reference. [Rectangular comparison](../docs/nvf-gap-convergence.md), [affine comparison](../docs/nvf-affine-convergence.md). |
| `nvf_affine_convergence.py` | Which direction limits the full-metric affine cell and its optical truncation? Independent fitted FEM separates the errors; complete order rectangles expose directional and mixed changes without a convergence verdict. [Measurements and limits](../docs/nvf-affine-convergence.md). |
| `results/nvf_affine_projection_memory.log` | Does batching the affine auxiliary Fourier projection reduce memory, and what does it cost in time? Alternating NumPy comparisons retain the same finite cell and all nine coefficient arrays. [Allocation scope and controls](../docs/nvf-affine-cells.md#bounded-fourier-projection-intermediates). |
| `nvf_mirror_reference.py` | Can independently fitted FEM fields satisfy the original mirror guard? Even/odd corrector spaces now give a verified cap-112 export using element-owned integration points. The completed order-40/48 candidate/FEM transmission differences stay below 9.58e-7 while both retain outer-order changes above 1e-5, isolating the current truncation problem. [Derivation and controls](../docs/nvf-mirror-reference.md). |
| `nvf_fem_quadrature.py` | Can fitted-interface integration avoid neighboring-material lookup errors? Element-owned Duffy points feed both half/full static FEM exports; exact material Fourier controls through order 112 and 13 optional FEM tests pass. [Diagnosis and correction](../docs/nvf-mirror-reference.md#element-ownership-at-high-quadrature). |
| `nvf_fem_reference.py` | Can conforming 3D H(curl) FEM establish independent converged answers for tilted and affine NVF interfaces? The affine p/s solve shares one factorization, passes independent slab controls, and separates omitted Robin power. Boundary storage, physical-edge sizing and fixed-mesh local degree controls permit further refinement; optical qualification remains open. [Investigation](../docs/nvf-independent-fem.md), [physical-edge refinement](../docs/nvf-independent-fem.md#refining-the-physical-dielectric-edges), [physical per-order comparison](../docs/nvf-independent-fem.md#comparing-the-physical-diffraction-orders-with-wavebit). |
| `nvf_fem_space.py` | Can local polynomial refinement separate film-space error from physical remeshing? Optional NGSolve entity-degree control preserves the original global spaces, periodic identifications and quadrature metadata. [Definition and controls](../docs/nvf-independent-fem.md#raising-the-film-degree-on-a-fixed-mesh). |
| `nvf_material_projection.py` | Does defining the normal field from the material Fourier coefficients remove primitive-partition dependence without damaging contact derivatives? The partition check passes, but the derivative gate fails. Reproduction and rejection: [../docs/nvf-contact-followup.md](../docs/nvf-contact-followup.md#further-closure-attempts-no-accepted-replacement-2026-09-08). |
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
