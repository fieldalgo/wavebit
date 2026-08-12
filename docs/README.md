# Docs

Measurements and decisions, written so they can be cited. `fieldwave/_core/` links to several
of these from code comments — they are why a constant has the value it has, so a document here
is not finished until it states what its numbers decided.

**Which prose belongs here at all is decided by [`../CONTRIBUTING.md`](../CONTRIBUTING.md), and
that table wins over anything below.** What follows is the contract for a document once it is
here: what it must contain, and what it may point at.

House style: title, `Written:` date, a `Targets:` list of the files it explains, a TL;DR, then
tables. Say what was measured on what hardware, and where the result stops being trustworthy.
Every path in `Targets:` must resolve — `tests/test_docs_contract.py` checks it.

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
| [`bessel-j1.md`](bessel-j1.md) | Do the three backends compute the same `J_1`, and how accurate is it? | They do now — one shared hybrid, 1e-12, with a crossover per precision |
| [`torch-compile.md`](torch-compile.md) | Is `torch.compile` worth chasing for this solver? | No — the solve is almost all LAPACK, which the compiler forwards untouched |
| [`oblique-lattice.md`](oblique-lattice.md) | What does `lattice` do to a non-orthogonal (oblique, hexagonal) cell, and what does it cost? | Implemented — ~8x cheaper than the supercell; NVF only, and the raster is read in lattice coordinates |
| [`fourier-truncation-shape.md`](fourier-truncation-shape.md) | Does S4's circular cut of the Fourier order set beat the full rectangle FieldWave keeps? | **Not implemented** — structure-dependent, within ±20%, and negative on a common case |
| [`anisotropic-materials.md`](anisotropic-materials.md) | Can an anisotropic or magnetic material be given to the solver, and what does each missing case cost? | Uniform tensors work, z-coupling included since the 4ff path landed; what is left is a *patterned* anisotropic layer and `mu` |
| [`off-diagonal-slot-convention.md`](off-diagonal-slot-convention.md) | If I fill the five permittivity slots by hand, what goes in them — and why did an asymmetric tensor come out wrong? | The slot holds `-eps_xy`; an ordering defect on asymmetric pairs, found and fixed |
| [`connecting-algorithms.md`](connecting-algorithms.md) | Which `connecting_algo`, and why is a plain transfer matrix not one of the choices? | `ETM` — same answer as `SMM` to 8e-16 and 1.25-1.57x cheaper; a plain transfer refuses at 25 orders |
| [`first-order-4ff.md`](first-order-4ff.md) | How do you validate a solver no other solver can check? | Judge the tilted-magnetisation geometry by a rotation-covariance theorem; decide mode direction per mode, not per set |
| [`five-slot-limit.md`](five-slot-limit.md) | Why does a nine-component tensor have only five slots, and what do other solvers do with the missing four? | Structural to 2ff, not ours; three of four reference solvers claim a full 3×3 and do not deliver one |
| [`branch-cuts.md`](branch-cuts.md) | Which root does each `sqrt` take, and why did evanescent (complex-angle) incidence return a grating that scatters nothing? | Read the incident `kz`, never respell it — fixed. At single precision the branch is unresolvable and no threshold repairs it, so it warns |
| [`time-convention-audit.md`](time-convention-audit.md) | Which square root does a half-space take under `e^{-iωt}`, and what did the flip from `e^{+iωt}` leave behind? | `Re(kz) > 0` plus a sheet test; two live defects found, both invisible at a real angle |
| [`formulation.md`](formulation.md) | The full derivation, Maxwell to efficiencies, natively in `e^{-iωt}` — no translation from the `e^{+jωt}` paper | Every boxed result checked against `_core/` symbol-for-symbol; LaTeX twin lives in the paper's tree |
| [`conventions.md`](conventions.md) | Every sign, axis, name and unit a user's script can depend on — the release contract | Normative; a change that falsifies a sentence there is breaking by definition |
| [`capability-manifest.md`](capability-manifest.md) | How does a program outside this process learn what the solver supports, without retyping it? | One generated JSON document — capabilities and constraints only; recommending is somebody else's job |

## Docs vs tutorials

Audience, not format. **`docs/` is for a reader who already knows the API** and needs to know
what was measured or why a default is what it is; [`../tutorials/`](../tutorials) is for a
reader who does not know it yet.

A tutorial is outgrown — you read it once. A document here is *cited*, including from code. So
a notebook whose purpose is to record a measurement belongs here as markdown instead: stored as
JSON it will not diff, will not review, and will carry stale outputs.
