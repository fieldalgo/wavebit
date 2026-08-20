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
(`wavebit/_core/analytic.py`) always works, and so does the `_core/...` shorthand this
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
somewhere else** — question 6 of the CONTRIBUTING table, not this folder.

## What a document here is not

- **Not the user manual.** These answer *why it is built this way*; how to use the library is
  the tutorials and the manual. Overlap in subject is fine — the two have different readers.
- **Not a changelog or a project record.** A document states what is true now and what the
  measurement decided. Past defects appear only where the current behaviour is otherwise
  inexplicable — a warning that exists because of one, a slot whose sign looks wrong until you
  know why.
- **Not a scratch pad.** An unfinished measurement lives in `studies/` until it decides
  something; scratch lives outside the checkout entirely.

| Document | Question it answers | Decision |
| --- | --- | --- |
| [`backend-choice.md`](backend-choice.md) | Which backend, and whether to freeze the geometry, for a *differentiated* run? | PyTorch below `fto≈5`; `jax jit` from `fto≈7` up; re-freeze every step |
| [`analytic-modeling.md`](analytic-modeling.md) | How does the staircase-free shape path converge against the raster and block staircases? | The smooth path for tilted and curved boundaries |
| [`boundary-modeling.md`](boundary-modeling.md) | Can a layer be painted *and* keep a tilted edge exact? | Yes — resolve the painting into polygons once, then transform those |
| [`nvf-upsampling.md`](nvf-upsampling.md) | What grid should the NVF normal field be taken on? | Not the fine one: it lost accuracy in 27 of 32 combinations and cost the most |
| [`nvf-normal-methods.md`](nvf-normal-methods.md) | Can the raster normal field do better than the {0, 45}-degree staircase gradient? And is it periodic? | Yes — `normal_method='contour'` won 11/12 resolved solver cells and lost none; the best *angular* field (`'smooth'`) loses on corners, which is why the solver A/B decides. Periodic only since 2026-08-17: `'distance'` read the cell as a flat array in three places, so a lattice translation of the raster moved `R` by 1e-2 at every `fto` |
| [`bessel-j1.md`](bessel-j1.md) | Do the three backends compute the same `J_1`, and how accurate is it? | They do now — one shared hybrid, 1e-12, with a crossover per precision |
| [`torch-compile.md`](torch-compile.md) | Is `torch.compile` worth chasing for this solver? | No — the solve is almost all LAPACK, which the compiler forwards untouched |
| [`oblique-lattice.md`](oblique-lattice.md) | What does `lattice` do to a non-orthogonal (oblique, hexagonal) cell, and what does it cost? | Implemented — ~8x cheaper than the supercell; NVF only, and the raster is read in lattice coordinates |
| [`fourier-truncation-shape.md`](fourier-truncation-shape.md) | Does S4's circular cut of the Fourier order set beat the full rectangle Wavebit keeps? | **Not implemented** — structure-dependent, within ±20%, and negative on a common case |
| [`anisotropic-materials.md`](anisotropic-materials.md) | Can an anisotropic or magnetic material be given to the solver, and what does each missing case cost? | Uniform tensors work, z-coupling dispatched to automatically since 2026-08-13, patterned in-plane tensors and `mu != 1` since 2026-08-14; what is left is patterned *and* z-coupled |
| [`patterned-anisotropic-factorization.md`](patterned-anisotropic-factorization.md) | How do you factorize a layer whose permittivity is a *tensor* at every pixel, when Li's rule needs one discontinuous factor and a tensor gives three? | Popov–Nevière — finish the projector algebra pointwise, transform once. Doing it as matrix products passes energy, covariance and fmmax while being wrong by 0.5 on a uniform slab |
| [`magnetic-materials.md`](magnetic-materials.md) | Can a layer or a half-space have `mu != 1`, and where does permeability enter an operator written for `eps`? | Implemented on the 2D path — layers as the E/H dual of the five slots, half-spaces as one pair of admittances, patterned layers through the same Popov-Nevière rearrangement unchanged. Written as `Material(epsilon, mu)` on the three assignment paths and as a `mu=` array beside a raster, and built by running the permittivity's own builder a second time. Three traps energy conservation cannot see: the hand-expanded `Omega2_LR`, the incident flux normalisation, and the slot order that was inverted between builder and operator until 2026-08-18 |
| [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) | If I fill the five permittivity slots by hand, what goes in them — and why did an asymmetric tensor come out wrong? | The slot holds `-eps_xy`; an ordering defect on asymmetric pairs, found and fixed; and (§7.1) the builders' slot *order*, which disagreed with `solve_for_conv` silently until 2026-08-18 |
| [`connecting-algorithms.md`](connecting-algorithms.md) | Which `connecting_algo`, and why is a plain transfer matrix not one of the choices? | `ETM` — same answer as `SMM` to 8e-16 and 1.25-1.57x cheaper; a plain transfer refuses at 25 orders |
| [`first-order-4ff.md`](first-order-4ff.md) | How do you validate a solver no other solver can check? | Judge the tilted-magnetisation geometry by a rotation-covariance theorem; decide mode direction per mode, not per set |
| [`five-slot-limit.md`](five-slot-limit.md) | Why does a nine-component tensor have only five slots, and what do other solvers do with the missing four? | Structural to 2ff, not ours; three of four reference solvers claim a full 3×3 and do not deliver one |
| [`branch-cuts.md`](branch-cuts.md) | Which root does each `sqrt` take, and why did evanescent (complex-angle) incidence return a grating that scatters nothing? | Read the incident `kz`, never respell it — fixed. At single precision the branch is unresolvable and no threshold repairs it, so it warns |
| [`rayleigh-points.md`](rayleigh-points.md) | What happens when an order sits *exactly* on a light line, and why did the answer depend on the solve path? | The 1D paths solve the point (conical reformulated off its singular `inv(A)`); 2D detects it; `auto_detune` retries off it by default — loudly, never silently |
| [`time-convention-audit.md`](time-convention-audit.md) | Which square root does a half-space take under `e^{-iωt}`, and what did the flip from `e^{+iωt}` leave behind? | `Re(kz) > 0` plus a sheet test; two live defects found, both invisible at a real angle |
| [`formulation.md`](formulation.md) | The full derivation, Maxwell to efficiencies, natively in `e^{-iωt}` — no translation from the `e^{+jωt}` paper | Every boxed result checked against `_core/` symbol-for-symbol; LaTeX twin lives in the paper's tree |
| [`factorization-terms.md`](factorization-terms.md) | Does `Laurent` name a rule or a matrix, and is the convolution matrix really Toeplitz? | A rule (= `direct`); and no — it is BTTB. `laurent_conv_2d` carried a rule word on a rule-neutral builder and is now `conv_matrix_2d` |
| [`amplitude-sp-frame.md`](amplitude-sp-frame.md) | Which s/p frame is each order's complex amplitude quoted in, and why did a transposed grating disagree with its twin on `r_s` but not on `R`? | `s_hat = z_hat x k_hat_par` — but every `Re(kx) < 0` order was reported reversed until 2026-08-16, invisible to `\|r\|^2` and therefore to every oracle here |
| [`what-a-solve-answers.md`](what-a-solve-answers.md) | Why is field reconstruction off by default, and what does the state actually cost? | `answers='amplitude'` — the field state is 4.25 matrices of side `2*ff_x*ff_y` **per layer** (240 MB at `fto=(15, 15)`, 733 MB at `(20, 20)`) against boundary vectors for the amplitudes, and a declaration rather than a flag because it must be static under `jit`: as a plain attribute the pytree round trip drops it, a solver that declared a field comes back without one, and the two declarations share a treedef so one compilation serves both |
| [`conventions.md`](conventions.md) | Every sign, axis, name and unit a user's script can depend on — the release contract, including **which quantity a material number is** | Normative; a change that falsifies a sentence there is breaking by definition. Materials are relative permittivity (`Material(epsilon, mu=1.0)`) since 2026-08-20 — the one row whose *value* moved and not just its name, which is why the retired spellings are refused rather than accepted |
| [`capability-manifest.md`](capability-manifest.md) | How does a program outside this process learn what the solver supports, without retyping it? | One generated JSON document — capabilities and constraints only; recommending is somebody else's job. `material_quantity` is machine-readable because that answer changed once, and an old consumer's numbers stay legal |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
