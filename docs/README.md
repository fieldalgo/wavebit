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
| [`nvf-galerkin-error.md`](nvf-galerkin-error.md) | Can conforming electric and displacement fields bound the closed auxiliary-cell error and drive mesh refinement? | Exact physical residual integrals and a coercivity bound separate nodal solve accuracy from constitutive approximation; optical accuracy remains separate. |
| [`nvf-contact-solver.md`](nvf-contact-solver.md) | Can complete retained elimination accelerate contact solves without losing physical QR safeguards? | Both methods solve the same finite space; every original guard precedes acceptance. |
| [`nvf-polynomial-pair-coupling.md`](nvf-polynomial-pair-coupling.md) | Can cubic and quartic pair fields retain the full real contact trace without losing precision in nearly coincident gradients? | Invertible real coordinates and direct field products preserve the original physical guards; optical qualification remains separate. |
| [`nvf-analytic-polynomial-traces.md`](nvf-analytic-polynomial-traces.md) | How do analytic nodal-bubble remainders stabilize the complete high-order trace space? | Same-space decimal/cardinal checks and contact measurements distinguish numerical improvements from unresolved optical convergence. |
| [`nvf-pair-corner-potentials.md`](nvf-pair-corner-potentials.md) | Can fixed real outer-corner powers preserve one complete positive/contact field space and its derivative? | Rational corner remainders and separately integrated smooth/trace products pass finite physical controls; joint optical convergence remains separate. |
| [`nvf-pair-junction-potentials.md`](nvf-pair-junction-potentials.md) | Can the closing-gap material junctions join the same complete finite field space and right-opening derivative? | Both-end grading, reference transport derivatives and balanced full equations retain the original guards; joint optical convergence remains separate. |
| [`nvf-pair-radial-potentials.md`](nvf-pair-radial-potentials.md) | How can additional real radial powers retain accurate whole-field derivatives despite near-dependent remainders? | Exact normalized root factors reduce the measured contact quadrature discrepancy; the enlarged optical slope still needs joint resolution. |
| [`nvf-pair-symmetric-potentials.md`](nvf-pair-symmetric-potentials.md) | Can the complete pair trial space preserve geometric reflection and its physical gap derivative? | Exact mesh and prefactor identities preserve reflection, while real coordinates formed before assembly retain the original physical checks. |
| [`nvf-contact-fourier-precision.md`](nvf-contact-fourier-precision.md) | Why must the free Fourier block retain the same precision as the complete common weak equations? | Direct triangular exponential moments resolve amplified free-block roundoff with independent Decimal and Gaussian checks. |
| [`nvf-retained-contact-solve.md`](nvf-retained-contact-solve.md) | Can complete contact equations use a smaller retained solve while preserving every physical field and variation? | Direct physical stiffness elimination and complete complex pivots retain the original constraints and independent energy checks. |
| [`nvf-contact-energy-coordinates.md`](nvf-contact-energy-coordinates.md) | Why must the independent contact energy check retain complete field coordinates and a consistent opening measure? | Sextic controls separate coefficient handoff and collapsed-measure rounding errors without changing physical guards. |
| [`nvf-positive-coordinate-precision.md`](nvf-positive-coordinate-precision.md) | How can complete higher-degree positive-gap equations retain accurate physical fields through large coefficient cancellation? | Primitive coordinate precision and a full geometric frame preserve the finite equations; independent field integrals retain the original physical thresholds. |
| [`nvf-pair-transmission-potentials.md`](nvf-pair-transmission-potentials.md) | Can fixed local transmission modes join the complete pair tensor and its material/gap variation? | One-sided sector derivatives and explicit reference coefficients preserve the finite construction; angular integration and unresolved optical space changes remain separate. |
| [`nvf-polynomial-galerkin.md`](nvf-polynomial-galerkin.md) | Can higher interface and interior polynomial resolution reduce closed constitutive error at equal unknown counts? | Independent cardinal-field checks verify degrees 2 through 4; improved measured bounds still do not certify metal optics. |
| [`nvf-pair-galerkin.md`](nvf-pair-galerkin.md) | Can a real constrained-gradient space retain reciprocity, passivity and holomorphic material dependence across contact? | Original nodal/Laurent checks verify the finite response and its derivatives; metal optical convergence remains separate. |
| [`nvf-periodic-galerkin.md`](nvf-periodic-galerkin.md) | Does a constrained real field resolve periodic closing-gap optics? | The p derivative approaches Li; s retains a significant trace-space error despite the correct closed-bar value. |
| [`nvf-seam-trace.md`](nvf-seam-trace.md) | Can continuous positive-gap potentials supply the missing unrestricted normal trace? | Their k_y=0 normal response passes independent field-energy and exact contact checks; full mesh coupling remains separate. |
| [`nvf-coupled-seam-galerkin.md`](nvf-coupled-seam-galerkin.md) | Can those real seam potentials share the complete mesh tensor and its contact derivative? | Mutual physical energies and retained projections form one finite weak solve; optical convergence and ordinary routing remain separate. |
| [`nvf-coupled-pair-seam-galerkin.md`](nvf-coupled-pair-seam-galerkin.md) | Can real internal-gap potentials retain the full metal pair tensor and its contact derivative? | Independent clipped nodal equations verify the finite coupling; metal optical convergence remains open. |
| [`nvf-paired-frame.md`](nvf-paired-frame.md) | Can a conjugate pair of normalized field energies retain reciprocity and passivity together? | The explicit paired closure preserves both with finite gap/material derivatives and exact uniform/laminated controls; optical convergence remains separate. |
| [`nvf-pair-normal-frame.md`](nvf-pair-normal-frame.md) | Can a material-referenced normal frame remove the auxiliary weak solve while retaining contact derivatives? | Coercivity and exact finite controls pass; tiled integration lowers memory, but complex reciprocity and metal optics remain unresolved. |
| [`nvf-pair-quadratic.md`](nvf-pair-quadratic.md) | Does a quadratic auxiliary potential improve matched-size contact cells? | Exact polynomial transforms and finite derivatives are verified; dielectric comparisons improve while metal convergence remains unresolved. |
| [`nvf-pair-refinement.md`](nvf-pair-refinement.md) | Can the finite contact family be refined without quadratic element storage? | Refinements 2/3 and contracted weak assembly pass independent finite derivative controls; dielectric comparisons improve but metal optical convergence remains unresolved. |
| [`nvf-pair-material-sector.md`](nvf-pair-material-sector.md) | Can the exact contact/opening cell admit passive metals? | A common half-plane certificate extends the finite NumPy/JAX family without changing its physical equations; independent derivatives pass, but coarse optical values and slopes fail convergence controls. |
| [`nvf-cartesian-cell.md`](nvf-cartesian-cell.md) | Can a physical rectangle retain its edges and derivatives on an oblique periodic lattice? | A small P1 template and explicit public builder keep the planned finite mesh fixed during live replay; contact, general geometry and optical convergence remain open. |
| [`nvf-cartesian-pair.md`](nvf-cartesian-pair.md) | Can two equal-height physical rectangles share an exact contact edge inside an oblique finite cell? | Separate frozen contact/open charts and an explicit public adapter pass independent finite weak, mapping and constrained derivative controls; ordinary contact JVP, general pairs and optical convergence retain their stated limits. |
| [`nvf-cartesian-pair-opening.md`](nvf-cartesian-pair-opening.md) | Can the finite pair retain its right derivative as a physical contact gap opens? | Exact collapse and analytical Laurent operators pass independent NumPy coefficient/closure controls on refinements 0/1; ordinary contact JVP and optical accuracy remain separate. |
| [`nvf-cartesian-pair-gap.md`](nvf-cartesian-pair-gap.md) | Can exact finite coordinates evaluate the pair through zero and positive physical gaps? | Original nodal and moving-simplex controls verify the bounded NumPy family and derivative; conditioning and optical scope remain separate. |
| [`nvf-cartesian-pair-gap-autodiff.md`](nvf-cartesian-pair-gap-autodiff.md) | Can ordinary automatic differentiation retain the finite gap response at exact contact? | Independent original equations qualify gap-only NumPy/JAX x64 values and derivatives on frozen charts; optical and other-gradient scope remain separate. |
| [`backend-choice.md`](backend-choice.md) | Which backend, and whether to freeze the geometry, for a *differentiated* run? | Shapes: `jax jit` at every size, re-freezing every step. Rasters: PyTorch below `truncation_order≈5`. And `shape_nvf_grid_size` outweighs both |
| [`parametric-modeling.md`](parametric-modeling.md) | How does the staircase-free shape path converge against a raster staircase? | The parametric path for tilted and curved boundaries |
| [`boundary-modeling.md`](boundary-modeling.md) | Can a layer be painted *and* keep a tilted edge exact? | Yes — resolve the painting into polygons once, then transform those |
| [`nvf-upsampling.md`](nvf-upsampling.md) | What grid should the NVF normal field be taken on? | Not the fine one: it lost accuracy in 27 of 32 combinations and cost the most |
| [`nvf-normal-methods.md`](nvf-normal-methods.md) | Can the raster normal field do better than the {0, 45}-degree staircase gradient? And is it periodic? | Yes — `raster_normal_method='contour'` won 11/12 resolved solver cells and lost none; the best *angular* field (`'smooth'`) loses on corners, which is why the solver A/B decides. Periodic only since 2026-08-17: `'distance'` read the cell as a flat array in three places, so a lattice translation of the raster moved `R` by 1e-2 at every `truncation_order` |
| the same file, [`nvf_projector`](nvf-normal-methods.md#nvf_projector--the-circular-polarization-limit-2026-09-01) | The medial axis has no direction to report. Can the projector stop asking for one? | Yes, and it is a trade rather than a win. `nvf_projector='circular'` (Antos 2009 / Liu & Fan Eq. 52) reaches a direction-free projector as a *limit*, taking the translation defect 9.87e-05 -> 1.5e-15 over all 4095 rolls — and loses all twelve accuracy cells by 1.1x-7x, gives up the exact mirror symmetry `'distance'` and `'smooth'` had, and is worst on a 1D grating, where the exact answer is known. Off by default |
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
| [`rayleigh-points.md`](rayleigh-points.md) | What happens when an order sits *exactly* on a light line, and why did the answer depend on the solve path? | The 1D paths solve the point (conical reformulated off its singular `inv(A)`); 2D detects it; `auto_detune` retries off it by default — loudly, never silently |
| [`time-convention-audit.md`](time-convention-audit.md) | Which square root does a half-space take under `e^{-iωt}`, and what did the flip from `e^{+iωt}` leave behind? | `Re(kz) > 0` plus a sheet test; two live defects found, both invisible at a real angle |
| [`formulation.md`](formulation.md) | The full derivation, Maxwell to efficiencies, natively in `e^{-iωt}` — no translation from the `e^{+jωt}` paper | Every boxed result checked against `_core/` symbol-for-symbol; LaTeX twin lives in the paper's tree |
| [`factorization-terms.md`](factorization-terms.md) | Does `Laurent` name a rule or a matrix, and is the convolution matrix really Toeplitz? | A rule (= `direct`); and no — it is BTTB. `laurent_conv_2d` carried a rule word on a rule-neutral builder and is now `conv_matrix_2d` |
| [`amplitude-sp-frame.md`](amplitude-sp-frame.md) | Which s/p frame is each order's complex amplitude quoted in, and why did a transposed grating disagree with its twin on `r_s` but not on `R`? | `s_hat = z_hat x k_hat_par` — but every `Re(kx) < 0` order was reported reversed until 2026-08-16, invisible to `\|r\|^2` and therefore to every oracle here |
| [`multimode-incidence.md`](multimode-incidence.md) | Can the illumination arrive on more than one diffraction order at once? | Yes — `excitation={0: 1.0, 1: 0.6-0.8j}`. It buys the solve count (2.95x for three beams) and, because the retained window widens around the lit orders, beats solving beam by beam from `truncation_order` 8 up on a third of the orders. Every lit order must propagate, or `R + T` exceeds one on a passive stack while looking like an efficiency |
| [`what-a-solve-answers.md`](what-a-solve-answers.md) | Why is field reconstruction off by default, and what does the state actually cost? | `answers='amplitude'` — the field state is 4.25 matrices of side `2*n_orders_u*n_orders_v` **per layer** (240 MB at `truncation_order=(15, 15)`, 733 MB at `(20, 20)`) against boundary vectors for the amplitudes, and a declaration rather than a flag because it must be static under `jit`: as a plain attribute the pytree round trip drops it, a solver that declared a field comes back without one, and the two declarations share a treedef so one compilation serves both |
| [`conventions.md`](conventions.md) | Every sign, axis, name and unit a user's script can depend on — the release contract, including **which quantity a material number is** | Normative; a change that falsifies a sentence there is breaking by definition. Materials are relative permittivity (`Material(epsilon, mu=1.0)`) since 2026-08-20 — the one row whose *value* moved and not just its name, which is why the retired spellings are refused rather than accepted |
| [`capability-manifest.md`](capability-manifest.md) | How does a program outside this process learn what the solver supports, without retyping it? | One generated JSON document — capabilities and constraints only; recommending is somebody else's job. `material_quantity` is machine-readable because that answer changed once, and an old consumer's numbers stay legal |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
