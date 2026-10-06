# Docs

Measurements and decisions, written so they can be cited. `wavebit/_core/` links to several
of these from code comments — they are why a constant has the value it has, so a document here
is not finished until it states what its numbers decided.

**Which prose belongs here at all is decided by [`../CONTRIBUTING.md`](../CONTRIBUTING.md), and
that table wins over anything below.** What follows is the contract for a document once it is
here: what it must contain, and what it may point at.

House style: title, `Written:` date, a `Targets:` list of the files it explains, a TL;DR, then
tables. Say what was measured on what hardware, and where the result stops being trustworthy.
Every path in `Targets:` must resolve — `tests/test_docs_contract.py` checks it.

Two spellings both count as resolving, and both are intended. A path from the repository root
(`wavebit/_core/parametric.py`) always works, and so does the `_core/...` shorthand this
codebase uses for its own internals — there is only one `_core/`, and writing the prefix out
every time is longer without being clearer. `Targets:` lists use the full path; running prose
may use either.

## A document here points inward only

`docs/` is published. Everyone who clones this repository gets it, so **a document may only
reference things that arrive with the clone**, plus the published literature:

| May cite | Must not cite |
| --- | --- |
| code, tests, studies and other docs in this repository | a path on anyone's machine — `/Users/...`, `~/project/...`, `C:\...` |
| a DOI, an arXiv id, a public URL | a private study, an unpublished draft, an internal note |
| a released version of another project | a sibling checkout or scratch directory, however stable it feels |

The rule is not decoration; it is what this repository already got wrong. Four documents
pointed at directories that exist on one laptop, and the pointers named more than a location:
a private collaboration, an old project name, a folder of internal drafts. To a reader they
were dangling links, and to the author they were a disclosure nobody chose to make. The
committed git author field says *who* wrote this, which is attribution and belongs; a path says
*what else is on their disk*, which is neither.

**When the source really is outside, state the substance instead of the address.** A private
measurement is still evidence — write down what it measured and what the number was, and say
plainly that the raw record is not public. `branch-cuts.md` §7 is the pattern: it names the
golden that ships, dates the run it was cut from, and stops there.

One thing worth noticing about how the leaks were distributed: they were all in the documents
whose subject lives outside the code. A document explaining `_core/` has no occasion to name a
directory. **If a draft here keeps wanting to point somewhere else, that is a signal it belongs
somewhere else** — question 7 of the CONTRIBUTING table, not this folder.

## One exception, and it is a directory

[`guide/`](guide) **is** the user manual, and the rules below do not apply inside it. A page
there states what a keyword does, what it defaults to and what it refuses; it carries no
`Written:` date and no `Targets:` list, because it records neither a measurement nor a decision.
Its own [`README.md`](guide/README.md) states its contract.

Everything else in this folder — every document sitting directly beside this one — follows what
comes next. The outward-pointing rule above binds `guide/` too; only the house style differs.

## What a document here is not

- **Not the user manual.** That is [`guide/`](guide), one directory down. These answer *why it
  is built this way*, and overlap in subject with the manual is fine — the two have different
  readers.
- **Not a changelog or a project record.** A document states what is true now and what the
  measurement decided. Past defects appear only where the current behaviour is otherwise
  inexplicable — a warning that exists because of one, a slot whose sign looks wrong until you
  know why.
- **Not a scratch pad.** An unfinished measurement lives in `studies/` until it decides
  something; scratch lives outside the checkout entirely.

| Document | Question it answers | Decision |
| --- | --- | --- |
| [Yee lattice radiation](fd-lattice-radiation.md) | Can a radiation map satisfy the grid's dispersion, reciprocity and energy instead of imposing continuum phase on discrete fields? | Expose the independently validated lattice transform explicitly, preserve continuum reference outputs, and keep material-boundary error separate. |
| [Smoothed occupancy](fd-subpixel-occupancy.md) | How does fractional geometry reach material coefficients, and what can a tilted-interface study establish? | Sample each component at its own point, report bounds during planning, and keep convergence claims tied to measured error and scatter. |
| [One conductive film](fd-conductive-film.md) | Can a mixed conductor–dielectric cell retain the complex interface law and converge against a continuum film? | Use an induced normal Debye memory and arithmetic tangential conductivity for the restricted flat-film scope; the two tested placements converge at second order, while general C8 smoothing remains open. |
| [One Drude or Lorentz film](fd-single-pole-film.md) | Can a flat film with one oscillator retain the exact harmonic normal response with the existing polarization recurrence? | The bounded implementation and required numerical gates are complete on the feature branch; the selected finer study resolves the Drude self-order exceptions but retains three noise-unresolved Lorentz final direct orders. |
| [One Debye film](fd-debye-film.md) | Can one passive relaxation retain the exact harmonic normal response with the existing split memory? | Use the transformed relaxation time and actual-cell material rate; all 24 self-orders and 48 direct pairs are second in the registered scope, and the bounded implementation's numerical gates pass. |
| [One film with two Debye poles](fd-two-debye-film.md) | Can two passive relaxations retain their coupled harmonic normal response with two split memory slots? | Use coupled passive modes and summed local rates; all 24 self-orders and 48 direct pairs are second in the registered scope, with full/slow regression and frozen-reference gates complete. |
| [A finite Debye sum in one film](fd-debye-sum-film.md) | Can the isolated-film interface law represent a finite passive relaxation sum without a separate implementation for each count? | Use certified coupled modes while preserving the one- and two-pole routes; all 24 self-orders and 48 direct pairs are second in the registered four-pole scope, with the full/slow regression and all four frozen-reference gates complete. |
| [A finite Drude/Lorentz sum in one film](fd-oscillator-sum-film.md) | Can coupled physical polarizations represent a finite passive oscillator sum with unequal damping? | Retain each damping coordinate, add one rank-one restoring and energy term, and preserve the earlier routes; the bounded implementation, six-row calibration, 684 targeted checks and ten mutation witnesses are complete; the 40-worker campaign completes with 45 second, two mixed and one unresolved self-order, plus four unresolved final direct pairs; full/slow regression and all four frozen-reference gates are complete. |
| [Mie validation](fd-mie-validation.md) | Do the analytic reference, public sphere and external Meep/FDTDX runs justify curved-interface convergence, and does adjoint constitutive coupling preserve the correct energy? | Retain placement RMS and fit scatter, isolate discarded cross terms, and validate compact D/B memory separately from the full C3 convergence gate. |
| [Sphere clipping under JIT](fd-sphere-clip-jit.md) | Why can a sphere's voxel fill change by hundreds when compiled inside CSG? | Factor radial differences of squares so endpoint predicates stay consistent; check 32 origins, four bodies, four execution paths, and inactive-guard invariance. |
| [Directional diffraction](fdtd-directional.md) | Can returning waves and a nearby long-lived resonance hide in electric-only extraction? | Separate E/H directions using discrete Maxwell admittance, and retain independent temporal and boundary checks. |
| [CPML strength](fdtd-pml-strength.md) | Can absorption strength and thickness be varied independently near a Rayleigh threshold? | Expose the nominal amplitude target without changing defaults, and recheck time and mesh convergence after boundary changes. |
| [Ringdown fits](fdtd-ringdown.md) | Does a small exponential-fit residual establish a physical resonance or Q? | Recover known material/Maxwell modes, then identify a window-dependent near-cutoff tail that needs broader convergence checks. |
| [Source spectral zeros](fdtd-pulse-notches.md) | Does removing a cutoff frequency from the source make scattering converge faster? | Recover matched Meep/native mode suppression, then separate reduced returns from remaining grid and accumulated time error. |
| [Source and boundary clearance](fdtd-source-clearance.md) | Does varying padding measure boundary error when it also moves the source? | Hold the source and measurement planes fixed, and compare source-only movement separately. |
| [Independent harmonic lattice reference](fdtd-lattice-reference.md) | How much near-cutoff error remains with infinite time and exact outgoing grid boundaries? | Separate lattice, finite CPML and finite-time differences using independently assembled scalar harmonic equations. |
| [Independent transverse resolution](fdtd-transverse-resolution.md) | Can selective x/y refinement reduce lattice error without refining the whole stack? | Compare axis controls at automatic and fixed time steps, and preserve transverse material covariance. |
| [Scene incidence sensitivities](fdtd-scene-angles.md) | Which illumination and normalization terms must an angle derivative include? | Check fixed-grid theta derivatives against independent dispersion and Airy formulas, including axis crossings. |
| [Scene azimuth sensitivities](fdtd-scene-azimuth.md) | How can phi vary through zero without changing the discrete experiment? | Keep y sampling fixed and verify joint angle derivatives, field rotations and continuum refinement. |
| [Half-space sensitivities](fdtd-halfspace-gradients.md) | Which boundary and calibration terms belong in an exterior-material derivative? | Fix nominal mesh indices and verify CPML derivatives, magnetic Airy refinement and public-API Meep differences. |
| [Internal layer sensitivities](fdtd-layer-gradients.md) | Can layer boundaries move without changing the mesh, and how accurate are their derivatives? | Preserve total depth with nominal mesh thicknesses; check AD, position-dependent Airy refinement, independent Meep perturbations and a refined coating design. |
| [Continuous internal-interface sampling](fdtd-layer-smoothing.md) | Can internal interfaces cross cells with continuous first derivatives, and what physical bias remains? | Average interface positions at a static width; separate sharp and graded continuum errors and check a refined coating design. |
| [Constrained multilayer optimization](fdtd-multilayer-optimization.md) | Do jointly optimized coating lengths and materials retain physical improvement under independent refinement? | Share minimum-length coordinates with RCWA, compare joint gradients and validate dense spectra beyond the training wavelengths. |
| [Transverse rectangle derivatives](fdtd-rectangle-gradients.md) | Do shared shape coordinates preserve periodic painting and physical scattering derivatives? | Check exact geometry, projected-edge coincidences, refinement against RCWA and independent Meep perturbations. |
| [Rotated rectangle area integration](fdtd-rotation-gradients.md) | Do exact rotated-cell areas give traceable shape and scattering derivatives? | Separate geometric exactness from material averaging; check independent area and optical refinement. |
| [Rotated rectangle translation averaging](fdtd-rotation-smoothing.md) | Can exact boundary averaging smooth derivatives while separating profile bias from cell error? | Integrate bilinear polygon moments and compare independent graded Fourier and Meep references. |
| [Adjoint and discrete-gradient checks](fdtd-adjoint-validation.md) | Do Meep adjoint, central differences and Wavebit autodiff measure the same derivative? | Separate the geometry chain, fixed-grid derivative and continuum-reference checks. |
| [Multiple rotated area shapes](fdtd-painted-shapes.md) | How do periodic boundary events retain later-shape painting and live topology changes? | Check independent Cartesian integrals, complete graphs and graded optical references. |
| [Rotated elliptical shape derivatives](fdtd-ellipse-gradients.md) | Do ellipse axes, rotation and material sensitivities agree with independent geometry and optics? | Check implicit conic chords, angular integration, affine Bessel profiles and public-API Meep controls. |
| [Circular shape derivatives](fdtd-circle-gradients.md) | Do disk cells preserve shape gradients and approach independent graded optics? | Check chord integrals, stable arc averaging, Bessel-series RCWA and public-API Meep controls. |
| [Moving total depth](fdtd-variable-depth.md) | Do independent layer lengths preserve physical reference phases on a fixed laboratory mesh? | Check both incidence sides, phase derivatives, graded-film refinement and independent Meep perturbations. |
| [Near-Rayleigh channels](fdtd-rayleigh.md) | Can a coarse grid classify a diffraction channel on the wrong side of cutoff? | Expose physical/discrete cutoff agreement and independent plane placement; retain the unresolved TM accuracy limit. |
| [ENZ and scattering convergence](fdtd-enz.md) | Can one settled incidence column hide ENZ transients in the other? | Report both calibrated columns and a Jones-input bound; separate temporal settling from ADE and spatial error. |
| [Layer-aligned meshes](fdtd-layer-alignment.md) | Does placing every layer boundary on a node improve dispersive TM optics? | Fit layer thicknesses explicitly and separate planar interface error from time and boundary effects. |
| [Oblique pulse bandwidth](fdtd-oblique-pulse.md) | Can source tails hide metal-interface errors? | Expose pulse width; independently check temporal settling, interface placement and reference convergence. |
| [Mixed low-loss metal](fdtd-mixed-metal.md) | Do combined Drude/Lorentz spectra and gradients converge? | Separate material settling, spatial error and differentiation; match interface sampling for Meep comparisons. |
| [Lorentz decay](fdtd-lorentz-decay.md) | Can material motion outlive a small field? | Track per-pole activity and expose a combined decay flag. |
| [Bragg cavity](fdtd-bragg-cavity.md) | Do long-lived resonances pass temporal diagnostics too early? | Short runs fail; spatial detuning remains independent. |
| [DFT sampling](fdtd-dft-sampling.md) | What is the cost/error tradeoff of skipping monitor evaluations? | Keep stride 1 by default; compare explicit strides for each pulse. |
| [Pulse settling](fdtd-pulse-convergence.md) | Does a small temporal tail certify accurate scattering? | Report calibrated amplitude changes separately; spatial refinement remains necessary. |
| [`backend-choice.md`](backend-choice.md) | Which backend, and whether to freeze the geometry, for a *differentiated* run? | Shapes: `jax jit` at every size, re-freezing every step. Rasters: PyTorch below `truncation_order≈5`. And `normal_grid` outweighs both |
| [`parametric-modeling.md`](parametric-modeling.md) | How does the staircase-free shape path converge against a raster staircase? | The parametric path for tilted and curved boundaries |
| [`boundary-modeling.md`](boundary-modeling.md) | Can a layer be painted *and* keep a tilted edge exact? | Yes — resolve the painting into polygons once, then transform those |
| [`nvf-upsampling.md`](nvf-upsampling.md) | What grid should the NVF normal field be taken on? | Not the fine one: it lost accuracy in 27 of 32 combinations and cost the most |
| [`nvf-normal-methods.md`](nvf-normal-methods.md) | Can the raster normal field do better than the {0, 45}-degree staircase gradient? And is it periodic? | Yes — `normal_method='contour'` won 11/12 resolved solver cells and lost none; the best *angular* field (`'smooth'`) loses on corners, which is why the solver A/B decides. Periodic only since 2026-08-17: `'distance'` read the cell as a flat array in three places, so a lattice translation of the raster moved `R` by 1e-2 at every `truncation_order` |
| the same file, [`projector`](nvf-normal-methods.md#projector--the-circular-polarization-limit-2026-09-01) | The medial axis has no direction to report. Can the projector stop asking for one? | Yes, and it is a trade rather than a win. `projector='circular'` (Antos 2009 / Liu & Fan Eq. 52) reaches a direction-free projector as a *limit*, taking the translation defect 9.87e-05 -> 1.5e-15 over all 4095 rolls — and loses all twelve accuracy cells by 1.1x-7x, gives up the exact mirror symmetry `'distance'` and `'smooth'` had, and is worst on a 1D grating, where the exact answer is known. Off by default |
| [`bessel-j1.md`](bessel-j1.md) | Do the three backends compute the same `J_1`, and how accurate is it? | They do now — one shared hybrid, 1e-12, with a crossover per precision |
| [`torch-compile.md`](torch-compile.md) | Is `torch.compile` worth chasing for this solver? | No — the solve is almost all LAPACK, which the compiler forwards untouched |
| [`oblique-lattice.md`](oblique-lattice.md) | What does `lattice` do to a non-orthogonal (oblique, hexagonal) cell, and what does it cost? | Implemented — ~8x cheaper than the supercell; NVF only, and the raster is read in lattice coordinates |
| [`fourier-truncation-shape.md`](fourier-truncation-shape.md) | Does S4's circular cut of the Fourier order set beat the full rectangle Wavebit keeps? | **Not implemented** — structure-dependent, within ±20%, and negative on a common case |
| [`anisotropic-materials.md`](anisotropic-materials.md) | Can an anisotropic or magnetic material be given to the solver, and what does each missing case cost? | Uniform tensors work, z-coupling dispatched to automatically since 2026-08-13, patterned in-plane tensors and `mu != 1` since 2026-08-14; what is left is patterned *and* z-coupled |
| [`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md) | How do you factorize a layer whose permittivity is a *tensor* at every pixel, when Li's rule needs one discontinuous factor and a tensor gives three? | Popov–Nevière — finish the projector algebra pointwise, transform once. Doing it as matrix products passes energy, covariance and fmmax while being wrong by 0.5 on a uniform slab |
| [`magnetic-materials.md`](magnetic-materials.md) | Can a layer or a half-space have `mu != 1`, and where does permeability enter an operator written for `eps`? | Implemented on the 2D path — layers as the E/H dual of the five slots, half-spaces as one pair of admittances, patterned layers through the same Popov-Nevière rearrangement unchanged — and on the 1D/conical paths since 2026-09-01 (§11), where the same duality carries the TE branch onto the TM one and three of the five slots fit. Written as `Material(epsilon, mu)` on the three assignment paths and as a `mu=` array beside a raster, and built by running the permittivity's own builder a second time. Field reconstruction carries it too since 2026-08-20 (§9) — `H_z` is Faraday's z row and needs `mu_zz^-1`, which it did not have. Five traps energy conservation cannot see: the hand-expanded `Omega2_LR`, the incident flux normalisation, the slot order that was inverted between builder and operator until 2026-08-18, an `H_z` scaled by `mu_zz` while every other component was exact, and a conical mutation survey run at `phi = 0`, where the two mode families decouple and a third of the new terms are unreachable |
| [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) | If I fill the five permittivity slots by hand, what goes in them — and why did an asymmetric tensor come out wrong? | The slot holds `-eps_xy`; an ordering defect on asymmetric pairs, found and fixed; and (§7.1) the builders' slot *order*, which disagreed with `solve_conv` silently until 2026-08-18 |
| [`connecting-algorithms.md`](connecting-algorithms.md) | Which `layer_algorithm`, and why is a plain transfer matrix not one of the choices? | `ETM` — same answer as `SMM` to 8e-16 and 1.25-1.57x cheaper; a plain transfer refuses at 25 orders |
| [`first-order-4ff.md`](first-order-4ff.md) | How do you validate a solver no other solver can check? | Judge the tilted-magnetisation geometry by a rotation-covariance theorem; decide mode direction per mode, not per set |
| [`five-slot-limit.md`](five-slot-limit.md) | Why does a nine-component tensor have only five slots, and what do other solvers do with the missing four? | Structural to 2ff, not ours; three of four reference solvers claim a full 3×3 and do not deliver one |
| [`branch-cuts.md`](branch-cuts.md) | Which root does each `sqrt` take, and why did evanescent (complex-angle) incidence return a grating that scatters nothing? | Read the incident `kz`, never respell it — fixed. At single precision the branch is unresolvable and no threshold repairs it, so it warns |
| [`result-guards.md`](result-guards.md) | What does a solve check about the result it hands back, and why is there no "R+T above 1" warning? | One guard: a non-finite efficiency raises. The physicality check was removed 2026-09-03 — it assumed passivity and never inspected the structure, so it warned on every active medium, and on 85 of 243 FDTD solves whose excess is second-order Yee truncation error |
| [`rayleigh-points.md`](rayleigh-points.md) | What happens when an order sits *exactly* on a light line, and why did the answer depend on the solve path? | The 1D paths solve the point (conical reformulated off its singular `inv(A)`); 2D detects it; `auto_detune` retries off it by default — loudly, never silently |
| [`time-convention-audit.md`](time-convention-audit.md) | Which square root does a half-space take under `e^{-iωt}`, and what did the flip from `e^{+iωt}` leave behind? | `Re(kz) > 0` plus a sheet test; two live defects found, both invisible at a real angle |
| [`formulation.md`](formulation.md) | The full derivation, Maxwell to efficiencies, natively in `e^{-iωt}` — no translation from the `e^{+jωt}` paper | Every boxed result checked against `_core/` symbol-for-symbol; LaTeX twin lives in the paper's tree |
| [`factorization-terms.md`](factorization-terms.md) | Does `Laurent` name a rule or a matrix, and is the convolution matrix really Toeplitz? | A rule (= `direct`); and no — it is BTTB. `laurent_conv_2d` carried a rule word on a rule-neutral builder and is now `conv_matrix_2d` |
| [`amplitude-sp-frame.md`](amplitude-sp-frame.md) | Which s/p frame is each order's complex amplitude quoted in, and why did a transposed grating disagree with its twin on `r_s` but not on `R`? | `s_hat = z_hat x k_hat_par` — but every `Re(kx) < 0` order was reported reversed until 2026-08-16, invisible to `\|r\|^2` and therefore to every oracle here |
| [`multimode-incidence.md`](multimode-incidence.md) | Can the illumination arrive on more than one diffraction order at once? | Yes — `excitation={0: 1.0, 1: 0.6-0.8j}`. It buys the solve count (2.95x for three beams) and, because the retained window widens around the lit orders, beats solving beam by beam from `truncation_order` 8 up on a third of the orders. Every lit order must propagate, or `R + T` exceeds one on a passive stack while looking like an efficiency |
| [`what-a-solve-answers.md`](what-a-solve-answers.md) | Why is field reconstruction off by default, and what does the state actually cost? | `answers='amplitude'` — the field state is 4.25 matrices of side `2*n_orders_u*n_orders_v` **per layer** (240 MB at `truncation_order=(15, 15)`, 733 MB at `(20, 20)`) against boundary vectors for the amplitudes, and a declaration rather than a flag because it must be static under `jit`: as a plain attribute the pytree round trip drops it, a solver that declared a field comes back without one, and the two declarations share a treedef so one compilation serves both |
| [`conventions.md`](conventions.md) | Every sign, axis, name and unit a user's script can depend on — the release contract, including **which quantity a material number is** | Normative; a change that falsifies a sentence there is breaking by definition. Materials are relative permittivity (`Material(epsilon, mu=1.0)`) since 2026-08-20 — the one row whose *value* moved and not just its name, which is why the retired spellings are refused rather than accepted |
| [`fdtd.md`](fdtd.md) | Does a second method, with different failure modes, agree with the RCWA path — and what does building one cost? | Implemented for 1D gratings (TE/TM, normal and oblique), a conical mount at and away from normal, and crossed gratings in 3D: worst per-order gap 6.6e-4 (TE) / 6.9e-4 (TM) against RCWA at a 5 nm grid, where Meep gives 4.9e-4 / 7.4e-4 on the same grid, falling to 1.4e-4 at second order throughout. Meep's column is committed (`tests/references/golden_meep_1d.npz`) rather than quoted, since 2026-09-06; all three columns are frozen in `golden_fdtd_table.npz` and CI reads §2's tables back out of the markdown, since 2026-09-07. Re-deriving the table moved it twice, and §2 says what moved — the second time the cause was that the conditions were the phrase "at the shipped defaults", which stayed true-looking while `PAD_WAVELENGTHS` was halved underneath it, so the record is now a file of values (`tests/references/_meep_1d_cases.py`) that a test holds against the module. Four findings outlast the agreement: a **dimensional constant in a scale-free solver** (the PML's frequency shift, written as a bare `0.05` = a reciprocal length) is a bug with no symptom at the design wavelength and returns `R+T = 1.22` in the mid-infrared; a wrong absorber is indistinguishable from a wrong discretisation, and I misdiagnosed it once — a correct CPML *is* dz-independent, so the "specify a physical thickness" conclusion was a workaround for a broken layer that also cost 4x the runtime; `T[+1] == T[-1]` on a mirror-symmetric grating is a free oracle-free error bar, exact here and 8.7e-5 off in Meep; and the amplitudes failed to converge twice for two unrelated reasons that `\|r\|^2` cannot see. That fourth finding generalised: every mount added since broke the phase while leaving `R + T` intact — a TFSF source that injected a non-conical wave, a reversed frame sign on the transmitted leg, and a uniform auxiliary grid beneath a graded one. 3D does not beat RCWA on speed, its break-even being hundreds of wavelengths of device; what justifies it is the class of question a harmonic solve cannot state. NumPy only — the time loop needs a `scan` primitive `_core/backend.py` does not have |
| [`fdtd-route-parity.md`](fdtd-route-parity.md) | On the same scene, with grid, `dt` and materials matched, do the legacy FDTD engine and the shared explicit-backend engine make the same error? | Mostly: `same_error` on 50 of 80 classified calls once the de-embedding phase is removed, and no scene ranks either engine. On power the vacuum scenes' smaller error follows the padding and a uniform slab disagrees with nothing to attribute it to; the shared engine at its own defaults and the source injection were not measured, so neither `default_route` nor either engine changes |
| [`capability-manifest.md`](capability-manifest.md) | How does a program outside this process learn what the solver supports, without retyping it? | One generated JSON document — capabilities and constraints only; recommending is somebody else's job. `material_quantity` is machine-readable because that answer changed once, and an old consumer's numbers stay legal |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
